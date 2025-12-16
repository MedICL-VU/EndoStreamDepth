import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import time
from einops import rearrange
from PIL import Image
import logging
from .dinov2 import DINOv2

from .mamba import MambaModel
from .rnn_transformer import TransformerRNN

from .original_dpt import DPTHead
from .hybrid_fusion import HybridFusion

from .util.loss import ScaleAndShiftInvariantLoss, SiLogLoss, GradientEdgeLoss, temporal_consistency_loss
from utils.helpers import *

from utils.eval_metrics.metrics import compute_depth_metrics




class EndoStreamDepth(nn.Module):
    def __init__(
        self,
        vit_size='vitl',
        dpt_dim=256,
        out_channels=[256, 512, 1024, 1024],
        patch_size=14,
        **kwargs
    ):
        super(EndoStreamDepth, self).__init__()

        encoder = vit_size
        model_configs = {
            'vits': {'encoder': 'vits', 'dpt_dim': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'dpt_dim': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        dpt_dim = model_configs[encoder]['dpt_dim']
        out_channels = model_configs[encoder]['out_channels']

        self.patch_size = patch_size

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
        }


        self.hybrid_configs = kwargs.get('hybrid_configs')
        if self.hybrid_configs is None or self.hybrid_configs.use_hybrid is False:
            self.hybrid_configs = None
        else:
            self.teacher_model = nn.Module()
            self.teacher_model.pretrained = DINOv2(model_name='vitl', patch_size=patch_size)
            self.teacher_model.depth_head = DPTHead(self.teacher_model.pretrained.embed_dim, dpt_dim=256, out_channels=[256, 512, 1024, 1024])
            self.teacher_model.eval()

            self.hybrid_fusion = HybridFusion(d_model=64, **self.hybrid_configs)


        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder, patch_size=patch_size)

        self.use_mamba = kwargs['use_mamba']
        if self.use_mamba:
            self.downsample_mamba = kwargs['downsample_mamba']
            self.mamba_in_dpt_layer = kwargs['mamba_in_dpt_layer']


            if kwargs.get('use_xlstm', False):
                from .xlstm_block import xLSTMModel
                self.mamba = xLSTMModel(dpt_dim, training_mode=kwargs['training'], **kwargs)

            elif kwargs.get('use_transformer_rnn', False):
                self.mamba = TransformerRNN(dpt_dim, **kwargs)
            else:
                self.mamba = MambaModel(dpt_dim, **kwargs)

            logging.info(f"downsample_mamba: {self.downsample_mamba}")
            logging.info(f"mamba_in_dpt_layer: {self.mamba_in_dpt_layer}")

        self.depth_head = DPTHead(self.pretrained.embed_dim, dpt_dim=dpt_dim, out_channels=out_channels)

        self.output_conv1_level4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.output_conv2_level4 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.output_conv1_level3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.output_conv2_level3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.output_conv1_level2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.output_conv2_level2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def dpt_features_to_mamba(self, input_shape, dpt_features, in_dpt_layer):
        # reshape to (B, T*h*w, c) for mamba
        if len(input_shape)==4:
            B, C, H, W = input_shape
            T = 1
        else:
            B, T, C, H, W = input_shape
        BT, c, h, w = dpt_features.shape
        assert BT == B*T, f"Expected batch dimension {B*T}, got {BT}" # sanity check

        downsample_factor = self.downsample_mamba[in_dpt_layer]


        if downsample_factor != 1.0:
            original_dpt_features = dpt_features.clone()
            original_dpt_features = rearrange(original_dpt_features, '(b t) c h w -> b t c h w', b=B, t=T)
            dpt_features = F.adaptive_avg_pool2d(dpt_features, (int(h*downsample_factor), int(w*downsample_factor)))


        dpt_features = rearrange(dpt_features, '(b t) c h w -> b t (h w) c', b=B, t=T)

        mamba_kwargs = dict(Thw = (1, H, W), dpt_shape=(h,w), downsample_factor=downsample_factor, in_dpt_layer=in_dpt_layer)

        # mamba_out = torch.zeros_like(dpt_features)
        mamba_out = []

        # process every frame with selected width
        for i in range(T):
            seq_out = self.mamba.forward_single_frame(dpt_features[:,i,...], **mamba_kwargs)
            if downsample_factor != 1.0:
                assert self.mamba.mamba_type == 'add'
                if seq_out.ndim == 3:
                    spatial_out = rearrange(seq_out, 'b (h w) c -> b c h w', h=int(h*downsample_factor), w=int(w*downsample_factor))
                else:
                    spatial_out = seq_out
                spatial_out = F.interpolate(spatial_out, (h,w), mode="bilinear", align_corners=True)

                seq_out = rearrange(spatial_out, 'b c h w -> b (h w) c')
                seq_out = self.mamba.final_layer(seq_out)

                spatial_out = rearrange(seq_out, 'b (h w) c -> b c h w', h=h, w=w)
                spatial_out = spatial_out + original_dpt_features[:,i,...]
                seq_out = rearrange(spatial_out, 'b c h w -> b (h w) c')

            mamba_out.append(seq_out)

        # reshape back to spatial format (B*T, c, h, w)
        mamba_out = torch.stack(mamba_out, dim=1)
        mamba_out = rearrange(mamba_out, 'b t (h w) c -> (b t) c h w', h=h, w=w, b=B)


        return mamba_out

    def get_dpt_features(self, x, input_shape=None):

        self.input_resolution = (x.shape[-1], x.shape[-2]) # w,h

        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        if self.hybrid_configs is None:
            intermediate_features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder])
            # 'vits': [2, 5, 8, 11],
            # 'vitl': [4, 11, 17, 23],
            if self.use_mamba:
                out = self.depth_head.forward_with_mamba(intermediate_features,
                                                         patch_h, patch_w, return_multiple=True,
                                                         temporal_layer=self.mamba_in_dpt_layer,
                                                         mamba_fn=self.dpt_features_to_mamba, shape_placeholder=input_shape)
            else:
                out = self.depth_head(intermediate_features, patch_h, patch_w)

            # logging.info(f'out: {out}')

        else:
            # using hybrid model
            # input resolution: (w,h), assuming height is short side, change if needed
            base_resolution = self.hybrid_configs['teacher_resolution']
            if self.input_resolution[1]>base_resolution:
                main_w = int((base_resolution/self.input_resolution[1])*self.input_resolution[0])
                main_w = (main_w // 14) * 14 # multiple of 14
                main_x = F.interpolate(x, (base_resolution, main_w), mode="bilinear", align_corners=True)
                high_res_x = x
            else:
                ## TODO: resolution < 518, directly run teacher model stream
                main_x = x
                high_res_x = x

            # STEP 2: get intermediate features
            student_intermediate_features = self.pretrained.get_intermediate_layers(high_res_x, self.intermediate_layer_idx[self.encoder])
            teacher_intermediate_features = self.teacher_model.pretrained.get_intermediate_layers(main_x, self.intermediate_layer_idx['vitl'])

            # STEP 3: get path_4s for fusion
            teacher_dpt_features = self.teacher_model.depth_head.get_path4(teacher_intermediate_features, main_x.shape[-2]//self.patch_size, main_x.shape[-1]//self.patch_size)
            student_path4 = self.depth_head.get_path4(student_intermediate_features, patch_h, patch_w)
            fused_path4 = self.hybrid_fusion(student_path4, teacher_dpt_features, path_idx=0)


            # STEP 4: run DPT decoder and mamba using fused path_4
            out = self.depth_head.forward_with_mamba(student_intermediate_features, patch_h, patch_w, temporal_layer=self.mamba_in_dpt_layer, mamba_fn=self.dpt_features_to_mamba, shape_placeholder=input_shape,
                                                     fused_path4=fused_path4)

        return out

    def final_head(self, x, patch_h, patch_w):

        out  = self.depth_head.scratch.output_conv1(x)

        bs = out.shape[0]
        target_h = int(patch_h * self.patch_size)
        target_w = int(patch_w * self.patch_size)

        # Process in batches of 60 frames
        # out = F.interpolate(out, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)), mode="bilinear", align_corners=True)
        # int max is 2147483647; for B,C=128,H=518,W=518, can only handle 60 frames
        # for vit-s using raw 2k resolution, can only handle 30 frames (2147483647/(32*1064*1904)=33)
        outputs = []
        for i in range(0, bs, 30):
            batch = out[i:i+30]  # Take up to 60 frames
            batch_out = F.interpolate(batch, (target_h, target_w),
                                    mode="bilinear", align_corners=True)
            outputs.append(batch_out)

        out = torch.cat(outputs, dim=0)
        out = self.depth_head.scratch.output_conv2(out)
        if out.max() <=0:
            logging.warning("Depth is all zeros")
        # depth = F.relu(out).squeeze(1)
        depth = out.squeeze(1) * 100
        # depth = out.squeeze(1)
        return depth

    def multi_level_final_head(self, x, c3vd=True):
        """
        x: list/tuple of 4 feature maps [x0, x1, x2, x3]
        returns: depth0, depth1, depth2, depth3  (each [B, H, W])
        """
        x0, x1, x2, x3 = x[0], x[1], x[2], x[3]

        # Per-level feature -> intermediate conv
        out0 = self.depth_head.scratch.output_conv1(x0)  # level 1 (highest res)
        out1 = self.output_conv1_level2(x1)  # level 2
        out2 = self.output_conv1_level3(x2)  # level 3
        out3 = self.output_conv1_level4(x3)  # level 4 (lowest res)

        def _to_depth(feat, level_idx: int, chunk_size: int = 30, target_h=518, target_w=518, max_depth=100):
            """
            Upsample in chunks to (target_h, target_w), apply final conv,
            scale and return [B, H, W].
            """
            bs = feat.shape[0]

            # Upsample in chunks to avoid int32 overflow in F.interpolate
            if feat.shape[-2] != target_h or feat.shape[-1] != target_w:
                upsampled = []
                for i in range(0, bs, chunk_size):
                    batch = feat[i:i + chunk_size]
                    batch_out = F.interpolate(
                        batch,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=True,
                    )
                    upsampled.append(batch_out)
                feat = torch.cat(upsampled, dim=0)

            # Final head conv (shared across levels; change if you have per-level conv2)
            if level_idx == 0:
                out = self.depth_head.scratch.output_conv2(feat)
            if level_idx == 1:
                out = self.output_conv2_level2(feat)
            if level_idx == 2:
                out = self.output_conv2_level3(feat)
            if level_idx == 3:
                out = self.output_conv2_level4(feat)

            if out.max() <= 0:
                logging.warning(f"Depth at level {level_idx} is all zeros")

            depth = out.squeeze(1) * max_depth
            return depth

        if c3vd:
            depth0 = _to_depth(out0, level_idx=0)
            depth1 = _to_depth(out1, level_idx=1, target_h=259, target_w=259)
            depth2 = _to_depth(out2, level_idx=2, target_h=130, target_w=130)
            depth3 = _to_depth(out3, level_idx=3, target_h=65, target_w=65)
        else:
            depth0 = _to_depth(out0, level_idx=0, target_h=476, target_w=476, max_depth=1)
            depth1 = _to_depth(out1, level_idx=1, target_h=238, target_w=238, max_depth=1)
            depth2 = _to_depth(out2, level_idx=2, target_h=119, target_w=119, max_depth=1)
            depth3 = _to_depth(out3, level_idx=3, target_h=60, target_w=60, max_depth=1)

        return depth0, depth1, depth2, depth3

    def make_multilevel_targets(self, gt_depth, valid_mask, pred_depth_tuple):
        """
        gt_depth:    [B, H, W] or [B, 1, H, W]
        valid_mask:  [B, H, W] or [B, 1, H, W]  (binary 0/1)
        pred_depth_tuple: tuple/list of predictions, each [B, h_i, w_i]

        returns:
            gt_levels:   list of [B, h_i, w_i]
            mask_levels: list of [B, h_i, w_i] (binary)
        """
        # Ensure [B, 1, H, W]
        if gt_depth.dim() == 3:
            gt = gt_depth.unsqueeze(1)
        else:
            gt = gt_depth

        if valid_mask.dim() == 3:
            mask = valid_mask.unsqueeze(1)
        else:
            mask = valid_mask

        gt_levels = []
        mask_levels = []

        for pred_i in pred_depth_tuple:
            h, w = pred_i.shape[-2], pred_i.shape[-1]

            if (h, w) == gt.shape[-2:]:
                gt_i = gt
                mask_i = mask
            else:
                # depth: bilinear
                gt_i = F.interpolate(
                    gt,
                    size=(h, w),
                    mode="area",
                    # align_corners=False,
                )

                # mask: nearest, then re-binarize
                mask_i = F.interpolate(
                    mask.float(),
                    size=(h, w),
                    mode="nearest",
                )
                mask_i = (mask_i > 0.5).float()

            gt_levels.append(gt_i.squeeze(1))  # [B, h, w]
            mask_levels.append(mask_i.squeeze(1))  # [B, h, w]

        return gt_levels, mask_levels




    def train_sequence(self, batch, loss_type='l1', vis_training=False, savedir='debug_training', dataset='cv3d', **kwargs):
        # both have shape (B, T, C, H, W)
        video, gt_depth, valid_mask = batch
        video = video.to(torch.cuda.current_device())


        gt_depth = gt_depth.to(torch.cuda.current_device())
        valid_mask = valid_mask.to(torch.cuda.current_device())

        self.mamba.start_new_sequence()

        B, T, C, H, W = video.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # reshape to (B*T, C, H, W) for ViT
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        dpt_features = self.get_dpt_features(video, input_shape=(B, T, C, H, W)) # (B*T, c, h, w) where c=dpt_dim, h,w are also downsampled versions

        # pred_depth = self.final_head(dpt_features, patch_h, patch_w) # (B*T, H, W)

        # pred_depth = self.multi_level_final_head(dpt_features)

        # print(torch.unique(gt_depth.float()))
        dataset = list(dataset)
        if 'cv3d' in dataset:
            pred_depth = self.multi_level_final_head(dpt_features)
            pred_depth = tuple(d * 100 for d in pred_depth)
            max_depth = 100
            #print(f'max_depth: {max_depth}')
        else:
            pred_depth = self.multi_level_final_head(dpt_features, c3vd=False)
            max_depth = 1
            # print(f'max_depth: {max_depth}')

        # loss
        gt_depth = rearrange(gt_depth, 'b t h w -> (b t) h w')
        valid_mask = rearrange(valid_mask, 'b t h w -> (b t) h w')


        gt_depths, valid_masks = self.make_multilevel_targets(gt_depth, valid_mask, pred_depth)

        # pred_depth = pred_depth * valid_mask
        # valid_mask = valid_mask.bool()

        # valid_mask = gt_depth >=0
        # valid_mask = gt_depth > 0
        #criterion = SiLogLoss().cuda(device)
        #criterion = utils.SiLogLoss()

        # for x in range(20):
        #     print()
        #     print()
        #     print(f"gt_depth[{x}]:")
        #     print(torch.unique(gt_depth[x].float()))
        # print(torch.unique(pred_depth.float()))

        if 'l1' in loss_type:
            # loss_si = SiLogLoss()(pred_depth, gt_depth, (valid_mask == 1) & (gt_depth >= 0.01) & (gt_depth <= 100))
            # loss_log_l1 = F.l1_loss(torch.log(pred_depth[valid_mask] + 1e-6),
            #         torch.log(gt_depth[valid_mask] + 1e-6))
            # loss_grad = GradientEdgeLoss()(pred_depth, gt_depth, valid_mask)
            # loss = loss_si + loss_grad + loss_log_l1
            level_weights = [1, 1, 1, 1]  # all1
            # level_weights = [1.0, 0.75, 0.5, 0.25]  # 0.25 factor
            # level_weights = [1, 0.5, 0.25, 0.125] # 0.5 factor
            loss_si = 0.0
            loss_log_l1 = 0.0
            loss_grad = 0.0
            weight_sum = 0.0
            loss_temp = 0
            for lvl, (pred_i, gt_i, mask_i) in enumerate(zip(pred_depth, gt_depths, valid_masks)):
                # shapes: [B, h_i, w_i]

                w_level = level_weights[lvl]

                mask_i = mask_i.bool()
                depth_valid = (mask_i == 1) & (gt_i >= 0.01) & (gt_i <= max_depth)

                if not depth_valid.any():
                    continue  # skip if nothing valid at this level

                weight_sum += w_level

                # --- SiLog loss (with combined depth_valid mask) ---
                loss_si += w_level * SiLogLoss()(pred_i, gt_i, depth_valid)

                # --- log L1 loss on valid pixels ---
                pred_valid = pred_i[depth_valid]
                gt_valid = gt_i[depth_valid]

                if lvl == 0:
                    loss_log_l1 += w_level * F.l1_loss(
                        torch.log(pred_valid + 1e-6),
                        torch.log(gt_valid + 1e-6)
                    )


                # --- gradient loss (use geometric valid mask like original code) ---
                    loss_grad += w_level * GradientEdgeLoss()(pred_i, gt_i, mask_i)

                    pred_temporal = rearrange(pred_i, '(b t) h w -> b t h w', b=B)
                    valid_temporal = rearrange(mask_i, '(b t) h w -> b t h w', b=B)
                    loss_temp = 0.01 * temporal_consistency_loss(pred_temporal, valid_temporal)

            # normalize by total used weight, so scale is stable
            # if weight_sum > 0:
            #     loss_si /= weight_sum
            #     loss_log_l1 /= weight_sum
            #     loss_grad /= weight_sum
            # print('no norm')

            loss = loss_si + loss_grad + loss_log_l1 + loss_temp

        pred_depth, gt_depth, valid_mask = pred_depth[0], gt_depths[0], valid_masks[0]
        pred_depth = pred_depth * valid_mask



        ### debug, visualize training data
        grid = None
        if vis_training:
            if dist.get_rank() == 0:
                with torch.no_grad():
                    pred_depth_vis = rearrange(pred_depth.clone().cpu(), '(b t) h w -> b t h w', b=B)
                    gt_depth_vis = rearrange(gt_depth.clone().cpu(), '(b t) h w -> b t h w', b=B)
                    video_vis = rearrange(video.clone().cpu(), '(b t) c h w -> b t c h w', b=B)

                    import os; os.makedirs(savedir, exist_ok=True)
                    for i in range(B):
                        try:
                            pred_save = depth_to_np_arr(pred_depth_vis[i])
                            video_save = torch_batch_to_np_arr(video_vis[i])
                            gt_save = depth_to_np_arr(gt_depth_vis[i])
                            grid = save_gifs_as_grid(video_save, pred_frames=pred_save, gt_frames=gt_save, duration=160,
                                                output_path=f'{savedir}/{vis_training}_{i}.gif', fixed_height=224)
                        except Exception as e:
                            logging.info(f"Visualization error for iter {vis_training}: {e}")
                            pass
            dist.barrier()


        return loss, grid


    @torch.no_grad()
    def forward(self, batch, use_mamba, gif_path, resolution, out_mp4 ,save_depth_npy=False, save_vis_map=False, dataset='cv3d', **kwargs):

        # both have shape (B, T, C, H, W)
        if isinstance(batch, list) or isinstance(batch, tuple):
            video, gt_depth = batch

        elif isinstance(batch, torch.Tensor):
            video = batch
            gt_depth = None

        preds = []

        loss = 0
        if use_mamba:
            self.mamba.start_new_sequence()

        for i in range(video.shape[1]):
            warmup_frames = 5
            if kwargs.get('print_time', False) and i==warmup_frames:
                torch.cuda.synchronize()
                start = time.time()
            frame = video[:, i, :, :, :].to(torch.cuda.current_device())
            B, C, H, W = frame.shape
            # print(B, C, H, W)


            patch_h, patch_w = frame.shape[-2] // self.patch_size, frame.shape[-1] // self.patch_size

            # dpt_features = self.get_dpt_features(frame)
            dpt_features = self.get_dpt_features(frame, input_shape=(B,C,H,W))

            # pred_depth = self.final_head(dpt_features, patch_h, patch_w)
            if 'cv3d' in dataset:
                pred_depth = self.multi_level_final_head(dpt_features)
                max_depth = 100
            else:
                pred_depth = self.multi_level_final_head(dpt_features, c3vd=False)
                max_depth = 1

            # pred_depth = self.multi_level_final_head(dpt_features)
            pred_depth = pred_depth[0]
            pred_depth = pred_depth * max_depth
            # print(pred_depth.shape)
            pred_depth = torch.clip(pred_depth, min=0)



            if gt_depth is not None and pred_depth.shape != gt_depth[:, i, :, :].shape:
                pred_depth = F.interpolate(pred_depth.unsqueeze(1), gt_depth[:, i, :, :].unsqueeze(1).shape[-2:], mode="bilinear", align_corners=True).squeeze(1)
                # print(torch.unique(pred_depth))
                # print(torch.unique(gt_depth[:, i, :, :].unsqueeze(1)))
            if gt_depth is not None:
                gt_frame = gt_depth[:, i, :, :].to(torch.cuda.current_device())
                valid_mask = gt_frame > 0
                # if loss_type == 'l1':
                #     loss += F.l1_loss(pred_depth[valid_mask], gt_frame[valid_mask])
                # elif loss_type == 'scaleshift':
                #     # loss += ScaleAndShiftInvariantLoss()(pred_depth, gt_frame, mask=valid_mask)
                #     loss += F.l1_loss(pred_depth[valid_mask], gt_frame[valid_mask])
                # else:
                loss += F.l1_loss(pred_depth[valid_mask], gt_frame[valid_mask])

            preds.append(pred_depth)

        if kwargs.get('print_time', False):
            try:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f'wall time taken: {end - start:.2f}; fps: {((video.shape[1]-warmup_frames) / (end - start)):.2f}; num frames: {video.shape[1]-warmup_frames}')
            except Exception as e:
                logging.info(f"Error in printing time: {e}")
                pass

        if kwargs.get('dummy_timing', False):
            return 0,0


        return self.save_and_return(video, gt_depth, preds, loss, save_depth_npy, gif_path, save_vis_map, out_mp4, resolution, kwargs)




    @torch.compiler.disable
    def save_and_return(self, video, gt_depth, preds, loss, save_depth_npy, gif_path, save_vis_map, out_mp4, resolution, kwargs):
        video = (torch.nn.functional.interpolate(video.reshape(-1, video.shape[2], *video.shape[-2:]),
                                                 size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)
                 .reshape(video.shape[0], video.shape[1], video.shape[2], *gt_depth.shape[-2:])
                 if video.shape[-2:] != gt_depth.shape[-2:] else video)

        grid = None
        if gt_depth is not None and kwargs.get('use_metrics', True):
            l1_loss = loss / video.shape[1]

            # calculating metrics across sequence
            #preds_tensor = torch.stack(preds, dim=1).cpu().float() # (1, T, H, W)
            #gt_depth = gt_depth.cpu().float() # (1, T, H, W)
            preds_cpu = [p.detach().to('cpu', non_blocking=True) for p in preds]  # list of [1,H,W] (CPU)
            preds_tensor = torch.stack(preds_cpu, dim=1).float()  # [1,T,H,W] on CPU
            gt_depth = gt_depth.cpu().float()

            loss = compute_depth_metrics(preds_tensor.squeeze(0), gt_depth.squeeze(0))
            loss['l1_loss'] = l1_loss.item()


        if save_depth_npy:
            test_idx = gif_path.rstrip('.gif').split('_')[-1]
            npy_path = os.path.join(os.path.dirname(gif_path), 'depth_npy_files') #, test_idx)
            os.makedirs(npy_path, exist_ok=True)
            for i in range(len(preds)):
                np.save(f'{npy_path}/frame_{i}.npy', preds[i].cpu().float().numpy().squeeze(0))

        if kwargs.get('out_video', True):
            try:
                pred0 = []
                valid_mask = gt_depth[0, 0, :] > 0
                for i in range(len(preds)):
                    pred0.append((valid_mask*preds[i][0]).cpu())
                pred0 = torch.stack(pred0)
                disparity = kwargs.get("disparity", False)  # default False
                pred_save = depth_to_np_arr(pred0, disparity=disparity)
                video_save = torch_batch_to_np_arr(video[0])
                if gt_depth is not None:
                    gt_save = depth_to_np_arr(gt_depth[0], disparity=disparity)
                else:
                    gt_save = None

                # inferno heat map
                if save_vis_map:
                    test_idx = gif_path.rstrip('.gif').split('_')[-1]
                    vis_map_path = os.path.join(os.path.dirname(gif_path), 'vis_maps') #, test_idx)
                    os.makedirs(vis_map_path, exist_ok=True)
                    for i in range(len(pred_save)):
                        Image.fromarray(pred_save[i]).save(f'{vis_map_path}/frame_{i}.png')

                os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                if not out_mp4:
                    grid = save_gifs_as_grid(video_save,gt_save,pred_save, output_path=gif_path, fixed_height=resolution)
                else:
                    grid = save_grid_to_mp4(video_save,gt_save,pred_save, output_path=gif_path.replace('.gif', '.mp4'), fixed_height=video.shape[-2])


            except Exception as e:
                logging.info(f"Error in saving video: {e}")
                pass

        return loss, grid



    # not using mamba
    def train_single(self, batch, loss_type='l1', vis_training=False, savedir='debug_training'):

        images, gt_depth, valid_mask = batch
        images = images.to(torch.cuda.current_device()).squeeze(1)
        # gt_depth = gt_depth.to(torch.cuda.current_device()).squeeze(1) *100
        gt_depth = gt_depth.to(torch.cuda.current_device()).squeeze(1)
        valid_mask = valid_mask.to(torch.cuda.current_device()).squeeze(1)

        assert images.ndim == 4, f"{images.shape}; image ndim should only be 4"

        patch_h, patch_w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size

        dpt_features = self.get_dpt_features(images)
        pred_depth = self.final_head(dpt_features, patch_h, patch_w) # (B, H, W)

        pred_depth = pred_depth * valid_mask
        valid_mask = valid_mask.bool()

        # valid_mask = gt_depth >=0
        # valid_mask = gt_depth > 0
        # loss = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
        loss_si = SiLogLoss()(pred_depth, gt_depth, (valid_mask == 1) & (gt_depth >= 0.01) & (gt_depth <= 100))
        loss_log_l1 = F.l1_loss(torch.log(pred_depth[valid_mask] + 1e-6),
                                    torch.log(gt_depth[valid_mask] + 1e-6))
        loss_grad = GradientEdgeLoss()(pred_depth, gt_depth, valid_mask)
        loss = loss_si + loss_grad + loss_log_l1



        grid = None
        if vis_training:
            if dist.get_rank() == 0:
                with torch.no_grad():
                    import os; os.makedirs(savedir, exist_ok=True)
                    try:
                        pred_depth = torch.clip(pred_depth, min=0)
                        pred_save = depth_to_np_arr(pred_depth)
                        video_save = torch_batch_to_np_arr(images)
                        gt_save = depth_to_np_arr(gt_depth)
                        grid = save_gifs_as_grid(video_save, pred_frames=pred_save, gt_frames=gt_save,
                                            output_path=f'{savedir}/{vis_training}.gif', fixed_height=224)
                    except Exception as e:
                        logging.info(f"Visualization error for iter {vis_training}: {e}")
                        pass
            dist.barrier()


        return loss, grid




#
#
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# import time
# from einops import rearrange
# from PIL import Image
# import logging
# from .dinov2 import DINOv2
#
# from .mamba import MambaModel
# from .rnn_transformer import TransformerRNN
#
# from .original_dpt import DPTHead
# from .hybrid_fusion import HybridFusion
#
# from .util.loss import ScaleAndShiftInvariantLoss, SiLogLoss
# from utils.helpers import *
#
# from utils.eval_metrics.metrics import compute_depth_metrics
#
#
# class FlashDepth(nn.Module):
#     def __init__(
#             self,
#             vit_size='vitl',
#             dpt_dim=256,
#             out_channels=[256, 512, 1024, 1024],
#             patch_size=14,
#             **kwargs
#     ):
#         super(FlashDepth, self).__init__()
#
#         encoder = vit_size
#         model_configs = {
#             'vits': {'encoder': 'vits', 'dpt_dim': 64, 'out_channels': [48, 96, 192, 384]},
#             'vitl': {'encoder': 'vitl', 'dpt_dim': 256, 'out_channels': [256, 512, 1024, 1024]},
#         }
#
#         dpt_dim = model_configs[encoder]['dpt_dim']
#         out_channels = model_configs[encoder]['out_channels']
#
#         self.patch_size = patch_size
#
#         self.intermediate_layer_idx = {
#             'vits': [2, 5, 8, 11],
#             'vitl': [4, 11, 17, 23],
#         }
#
#         self.hybrid_configs = kwargs.get('hybrid_configs')
#         if self.hybrid_configs is None or self.hybrid_configs.use_hybrid is False:
#             self.hybrid_configs = None
#         else:
#             self.teacher_model = nn.Module()
#             self.teacher_model.pretrained = DINOv2(model_name='vitl', patch_size=patch_size)
#             self.teacher_model.depth_head = DPTHead(self.teacher_model.pretrained.embed_dim, dpt_dim=256,
#                                                     out_channels=[256, 512, 1024, 1024])
#             self.teacher_model.eval()
#
#             self.hybrid_fusion = HybridFusion(d_model=64, **self.hybrid_configs)
#
#         self.encoder = encoder
#         self.pretrained = DINOv2(model_name=encoder, patch_size=patch_size)
#
#         self.use_mamba = kwargs['use_mamba']
#         if self.use_mamba:
#             self.downsample_mamba = kwargs['downsample_mamba']
#             self.mamba_in_dpt_layer = kwargs['mamba_in_dpt_layer']
#
#             if kwargs.get('use_xlstm', False):
#                 from .xlstm_block import xLSTMModel
#                 self.mamba = xLSTMModel(dpt_dim, training_mode=kwargs['training'], **kwargs)
#
#             elif kwargs.get('use_transformer_rnn', False):
#                 self.mamba = TransformerRNN(dpt_dim, **kwargs)
#             else:
#                 self.mamba = MambaModel(dpt_dim, **kwargs)
#
#             logging.info(f"downsample_mamba: {self.downsample_mamba}")
#             logging.info(f"mamba_in_dpt_layer: {self.mamba_in_dpt_layer}")
#
#         self.depth_head = DPTHead(self.pretrained.embed_dim, dpt_dim=dpt_dim, out_channels=out_channels)
#
#     def dpt_features_to_mamba(self, input_shape, dpt_features, in_dpt_layer):
#         # reshape to (B, T*h*w, c) for mamba
#         if len(input_shape) == 4:
#             B, C, H, W = input_shape
#             T = 1
#         else:
#             B, T, C, H, W = input_shape
#         BT, c, h, w = dpt_features.shape
#         assert BT == B * T, f"Expected batch dimension {B * T}, got {BT}"  # sanity check
#
#         downsample_factor = self.downsample_mamba[in_dpt_layer]
#
#         if downsample_factor != 1.0:
#             original_dpt_features = dpt_features.clone()
#             original_dpt_features = rearrange(original_dpt_features, '(b t) c h w -> b t c h w', b=B, t=T)
#             dpt_features = F.adaptive_avg_pool2d(dpt_features, (int(h * downsample_factor), int(w * downsample_factor)))
#
#         dpt_features = rearrange(dpt_features, '(b t) c h w -> b t (h w) c', b=B, t=T)
#
#         mamba_kwargs = dict(Thw=(1, H, W), dpt_shape=(h, w), downsample_factor=downsample_factor,
#                             in_dpt_layer=in_dpt_layer)
#
#         # mamba_out = torch.zeros_like(dpt_features)
#         mamba_out = []
#
#         for i in range(T):
#             seq_out = self.mamba.forward_single_frame(dpt_features[:, i, ...], **mamba_kwargs)
#             if downsample_factor != 1.0:
#                 assert self.mamba.mamba_type == 'add'
#                 if seq_out.ndim == 3:
#                     spatial_out = rearrange(seq_out, 'b (h w) c -> b c h w', h=int(h * downsample_factor),
#                                             w=int(w * downsample_factor))
#                 else:
#                     spatial_out = seq_out
#                 spatial_out = F.interpolate(spatial_out, (h, w), mode="bilinear", align_corners=True)
#
#                 seq_out = rearrange(spatial_out, 'b c h w -> b (h w) c')
#                 seq_out = self.mamba.final_layer(seq_out)
#
#                 spatial_out = rearrange(seq_out, 'b (h w) c -> b c h w', h=h, w=w)
#                 spatial_out = spatial_out + original_dpt_features[:, i, ...]
#                 seq_out = rearrange(spatial_out, 'b c h w -> b (h w) c')
#
#             mamba_out.append(seq_out)
#
#         # reshape back to spatial format (B*T, c, h, w)
#         mamba_out = torch.stack(mamba_out, dim=1)
#         mamba_out = rearrange(mamba_out, 'b t (h w) c -> (b t) c h w', h=h, w=w, b=B)
#
#         return mamba_out
#
#     def get_dpt_features(self, x, input_shape=None):
#
#         self.input_resolution = (x.shape[-1], x.shape[-2])  # w,h
#
#         patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
#
#         if self.hybrid_configs is None:
#             intermediate_features = self.pretrained.get_intermediate_layers(x,
#                                                                             self.intermediate_layer_idx[self.encoder])
#
#             if self.use_mamba:
#                 out = self.depth_head.forward_with_mamba(intermediate_features, patch_h, patch_w,
#                                                          temporal_layer=self.mamba_in_dpt_layer,
#                                                          mamba_fn=self.dpt_features_to_mamba,
#                                                          shape_placeholder=input_shape)
#             else:
#                 out = self.depth_head(intermediate_features, patch_h, patch_w)
#
#             # logging.info(f'out: {out}')
#
#         else:
#             # using hybrid model
#             # input resolution: (w,h), assuming height is short side, change if needed
#             base_resolution = self.hybrid_configs['teacher_resolution']
#             if self.input_resolution[1] > base_resolution:
#                 main_w = int((base_resolution / self.input_resolution[1]) * self.input_resolution[0])
#                 main_w = (main_w // 14) * 14  # multiple of 14
#                 main_x = F.interpolate(x, (base_resolution, main_w), mode="bilinear", align_corners=True)
#                 high_res_x = x
#             else:
#                 ## TODO: resolution < 518, directly run teacher model stream
#                 main_x = x
#                 high_res_x = x
#
#             # STEP 2: get intermediate features
#             student_intermediate_features = self.pretrained.get_intermediate_layers(high_res_x,
#                                                                                     self.intermediate_layer_idx[
#                                                                                         self.encoder])
#             teacher_intermediate_features = self.teacher_model.pretrained.get_intermediate_layers(main_x,
#                                                                                                   self.intermediate_layer_idx[
#                                                                                                       'vitl'])
#
#             # STEP 3: get path_4s for fusion
#             teacher_dpt_features = self.teacher_model.depth_head.get_path4(teacher_intermediate_features,
#                                                                            main_x.shape[-2] // self.patch_size,
#                                                                            main_x.shape[-1] // self.patch_size)
#             student_path4 = self.depth_head.get_path4(student_intermediate_features, patch_h, patch_w)
#             fused_path4 = self.hybrid_fusion(student_path4, teacher_dpt_features, path_idx=0)
#
#             # STEP 4: run DPT decoder and mamba using fused path_4
#             out = self.depth_head.forward_with_mamba(student_intermediate_features, patch_h, patch_w,
#                                                      temporal_layer=self.mamba_in_dpt_layer,
#                                                      mamba_fn=self.dpt_features_to_mamba, shape_placeholder=input_shape,
#                                                      fused_path4=fused_path4)
#
#         return out
#
#     def final_head(self, x, patch_h, patch_w):
#
#         out = self.depth_head.scratch.output_conv1(x)
#
#         bs = out.shape[0]
#         target_h = int(patch_h * self.patch_size)
#         target_w = int(patch_w * self.patch_size)
#
#         # Process in batches of 60 frames
#         # out = F.interpolate(out, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)), mode="bilinear", align_corners=True)
#         # int max is 2147483647; for B,C=128,H=518,W=518, can only handle 60 frames
#         # for vit-s using raw 2k resolution, can only handle 30 frames (2147483647/(32*1064*1904)=33)
#         outputs = []
#         for i in range(0, bs, 30):
#             batch = out[i:i + 30]  # Take up to 60 frames
#             batch_out = F.interpolate(batch, (target_h, target_w),
#                                       mode="bilinear", align_corners=True)
#             outputs.append(batch_out)
#
#         out = torch.cat(outputs, dim=0)
#         out = self.depth_head.scratch.output_conv2(out)
#         if out.max() <= 0:
#             logging.warning("Depth is all zeros")
#         depth = F.relu(out).squeeze(1)
#
#         return depth
#
#     def train_sequence(self, batch, loss_type='l1', vis_training=False, savedir='debug_training', **kwargs):
#         # both have shape (B, T, C, H, W)
#         video, gt_depth, valid_mask = batch
#         video = video.to(torch.cuda.current_device())
#
#         # multiplying gt disparity by 100 -> 1/meters to 100/meters;
#         # magic number for training stability because gt is in meters but depthanythingv2 original output values in the hundreds
#         # gt_depth = gt_depth.to(torch.cuda.current_device()) * 100
#         gt_depth = gt_depth.to(torch.cuda.current_device())
#         valid_mask = valid_mask.to(torch.cuda.current_device())
#
#         self.mamba.start_new_sequence()
#
#         B, T, C, H, W = video.shape
#         patch_h, patch_w = H // self.patch_size, W // self.patch_size
#
#         # reshape to (B*T, C, H, W) for ViT
#         video = rearrange(video, 'b t c h w -> (b t) c h w')
#         dpt_features = self.get_dpt_features(video, input_shape=(B, T, C, H,
#                                                                  W))  # (B*T, c, h, w) where c=dpt_dim, h,w are also downsampled versions
#         pred_depth = self.final_head(dpt_features, patch_h, patch_w)  # (B*T, H, W)
#         # depth = out.squeeze(1) * 100
#         # pred_depth = pred_depth * 100
#
#         # loss
#         gt_depth = rearrange(gt_depth, 'b t h w -> (b t) h w')
#         valid_mask = gt_depth >= 0
#         # valid_mask = rearrange(valid_mask, 'b t h w -> (b t) h w')
#
#         # pred_depth = pred_depth * valid_mask
#
#         # valid_mask = gt_depth >=0
#         # valid_mask = gt_depth > 0
#         # criterion = SiLogLoss().cuda(device)
#         # criterion = utils.SiLogLoss()
#
#         # for x in range(20):
#         #     print()
#         #     print()
#         #     print(f"gt_depth[{x}]:")
#         #     print(torch.unique(gt_depth[x].float()))
#         # print(torch.unique(pred_depth.float()))
#
#         if 'l1' in loss_type:
#             # loss = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
#             # loss = SiLogLoss()(pred_depth, gt_depth, (valid_mask == 1) & (gt_depth >= 0.01) & (gt_depth <= 100))
#
#         elif 'scaleshift' in loss_type:
#             loss = ScaleAndShiftInvariantLoss()(pred_depth, gt_depth, mask=valid_mask)
#
#         # tried implementing temporal loss from Video Depth Anything paper; didn't seem to help results
#         # keeping for reference
#         if 'temporal' in loss_type and kwargs['timestep'] > 500:
#             l1_loss = loss
#
#             # Reshape back to B,T,H,W for temporal processing
#             pred_temporal = rearrange(pred_depth, '(b t) h w -> b t h w', b=B)
#             gt_temporal = rearrange(gt_depth, '(b t) h w -> b t h w', b=B)
#             valid_temporal = rearrange(valid_mask, '(b t) h w -> b t h w', b=B)
#
#             temporal_loss = 0
#             K = int(
#                 loss_type.split('k')[1])  # # Maximum time step difference to consider (e.g., 2 means up to t and t+2)
#
#             for k in range(1, K + 1):
#                 pred_diff_k = pred_temporal[:, k:] - pred_temporal[:, :-k]
#                 gt_diff_k = gt_temporal[:, k:] - gt_temporal[:, :-k]
#
#                 valid_diff_k = (valid_temporal[:, k:] & valid_temporal[:, :-k])
#
#                 # Small change condition based on mean depth
#                 # for each pixel independently, it checks if the depth change (gt_diff_k) is less than 20% of that pixel’s mean depth (mean_depth_k).
#                 mean_depth_k = (gt_temporal[:, k:] + gt_temporal[:, :-k]) / 2
#                 # relative_threshold_k = 0.2 * mean_depth_k
#                 # small_change_mask_k = torch.abs(gt_diff_k) < relative_threshold_k
#                 relative_change_k = gt_diff_k.abs() / (mean_depth_k.abs() + 1e-6)
#                 small_change_mask_k = (relative_change_k < 0.2)
#
#                 temporal_mask_k = valid_diff_k & small_change_mask_k
#
#                 if temporal_mask_k.sum() > 0:
#                     loss_k = F.l1_loss(pred_diff_k[temporal_mask_k], gt_diff_k[temporal_mask_k])
#                     temporal_loss += loss_k
#
#             loss = l1_loss + 0.5 * temporal_loss
#             # loss = dict(total=loss, l1_loss=l1_loss, temporal_loss=temporal_loss*0.5)
#
#         ### debug, visualize training data
#         grid = None
#         if vis_training:
#             if dist.get_rank() == 0:
#                 with torch.no_grad():
#                     pred_depth_vis = rearrange(pred_depth.clone().cpu(), '(b t) h w -> b t h w', b=B)
#                     gt_depth_vis = rearrange(gt_depth.clone().cpu(), '(b t) h w -> b t h w', b=B)
#                     video_vis = rearrange(video.clone().cpu(), '(b t) c h w -> b t c h w', b=B)
#
#                     import os;
#                     os.makedirs(savedir, exist_ok=True)
#                     for i in range(B):
#                         try:
#                             pred_save = depth_to_np_arr(pred_depth_vis[i])
#                             video_save = torch_batch_to_np_arr(video_vis[i])
#                             gt_save = depth_to_np_arr(gt_depth_vis[i])
#                             grid = save_gifs_as_grid(video_save, pred_frames=pred_save, gt_frames=gt_save, duration=160,
#                                                      output_path=f'{savedir}/{vis_training}_{i}.gif', fixed_height=224)
#                         except Exception as e:
#                             logging.info(f"Visualization error for iter {vis_training}: {e}")
#                             pass
#             dist.barrier()
#
#         return loss, grid
#
#     @torch.no_grad()
#     def forward(self, batch, use_mamba, gif_path, resolution, out_mp4, save_depth_npy=False, save_vis_map=False,
#                 **kwargs):
#
#         # both have shape (B, T, C, H, W)
#         if isinstance(batch, list) or isinstance(batch, tuple):
#             video, gt_depth = batch
#         elif isinstance(batch, torch.Tensor):
#             video = batch
#             gt_depth = None
#
#         preds = []
#
#         loss = 0
#         if use_mamba:
#             self.mamba.start_new_sequence()
#
#         for i in range(video.shape[1]):
#             warmup_frames = 5
#             if kwargs.get('print_time', False) and i == warmup_frames:
#                 torch.cuda.synchronize()
#                 start = time.time()
#             frame = video[:, i, :, :, :].to(torch.cuda.current_device())
#             B, C, H, W = frame.shape
#
#             patch_h, patch_w = frame.shape[-2] // self.patch_size, frame.shape[-1] // self.patch_size
#
#             # dpt_features = self.get_dpt_features(frame)
#             dpt_features = self.get_dpt_features(frame, input_shape=(B, C, H, W))
#
#             pred_depth = self.final_head(dpt_features, patch_h, patch_w)
#
#             pred_depth = torch.clip(pred_depth, min=0)
#
#             if gt_depth is not None and pred_depth.shape != gt_depth[:, i, :, :].shape:
#                 pred_depth = F.interpolate(pred_depth.unsqueeze(1), gt_depth[:, i, :, :].unsqueeze(1).shape[-2:],
#                                            mode="bilinear", align_corners=True).squeeze(1)
#                 # print(torch.unique(pred_depth))
#                 # print(torch.unique(gt_depth[:, i, :, :].unsqueeze(1)))
#             if gt_depth is not None:
#                 gt_frame = gt_depth[:, i, :, :].to(torch.cuda.current_device())
#                 valid_mask = gt_frame > 0
#                 # if loss_type == 'l1':
#                 #     loss += F.l1_loss(pred_depth[valid_mask], gt_frame[valid_mask])
#                 # elif loss_type == 'scaleshift':
#                 #     # loss += ScaleAndShiftInvariantLoss()(pred_depth, gt_frame, mask=valid_mask)
#                 #     loss += F.l1_loss(pred_depth[valid_mask], gt_frame[valid_mask])
#                 # else:
#                 loss += F.l1_loss(pred_depth[valid_mask], gt_frame[valid_mask])
#
#             preds.append(pred_depth)
#
#         if kwargs.get('print_time', False):
#             try:
#                 torch.cuda.synchronize()
#                 end = time.time()
#                 logging.info(
#                     f'wall time taken: {end - start:.2f}; fps: {((video.shape[1] - warmup_frames) / (end - start)):.2f}; num frames: {video.shape[1] - warmup_frames}')
#             except Exception as e:
#                 logging.info(f"Error in printing time: {e}")
#                 pass
#
#         if kwargs.get('dummy_timing', False):
#             return 0, 0
#
#         return self.save_and_return(video, gt_depth, preds, loss, save_depth_npy, gif_path, save_vis_map, out_mp4,
#                                     resolution, kwargs)
#
#     @torch.compiler.disable
#     def save_and_return(self, video, gt_depth, preds, loss, save_depth_npy, gif_path, save_vis_map, out_mp4, resolution,
#                         kwargs):
#         video = (torch.nn.functional.interpolate(video.reshape(-1, video.shape[2], *video.shape[-2:]),
#                                                  size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)
#                  .reshape(video.shape[0], video.shape[1], video.shape[2], *gt_depth.shape[-2:])
#                  if video.shape[-2:] != gt_depth.shape[-2:] else video)
#
#         grid = None
#         if gt_depth is not None and kwargs.get('use_metrics', True):
#             l1_loss = loss / video.shape[1]
#
#             # calculating metrics across sequence
#             # preds_tensor = torch.stack(preds, dim=1).cpu().float() # (1, T, H, W)
#             # gt_depth = gt_depth.cpu().float() # (1, T, H, W)
#             preds_cpu = [p.detach().to('cpu', non_blocking=True) for p in preds]  # list of [1,H,W] (CPU)
#             preds_tensor = torch.stack(preds_cpu, dim=1).float()  # [1,T,H,W] on CPU
#             gt_depth = gt_depth.cpu().float()
#
#             loss = compute_depth_metrics(preds_tensor.squeeze(0), gt_depth.squeeze(0))
#             loss['l1_loss'] = l1_loss.item()
#
#         if save_depth_npy:
#             test_idx = gif_path.rstrip('.gif').split('_')[-1]
#             npy_path = os.path.join(os.path.dirname(gif_path), 'depth_npy_files')  # , test_idx)
#             os.makedirs(npy_path, exist_ok=True)
#             for i in range(len(preds)):
#                 np.save(f'{npy_path}/frame_{i}.npy', preds[i].cpu().float().numpy().squeeze(0))
#
#         if kwargs.get('out_video', True):
#             try:
#                 pred0 = []
#                 valid_mask = gt_depth[0, 0, :] > 0
#                 for i in range(len(preds)):
#                     pred0.append((valid_mask * preds[i][0]).cpu())
#                 pred0 = torch.stack(pred0)
#                 disparity = kwargs.get("disparity", False)  # default False
#                 pred_save = depth_to_np_arr(pred0, disparity=disparity)
#                 video_save = torch_batch_to_np_arr(video[0])
#                 if gt_depth is not None:
#                     gt_save = depth_to_np_arr(gt_depth[0], disparity=disparity)
#                 else:
#                     gt_save = None
#
#                 # inferno heat map
#                 if save_vis_map:
#                     test_idx = gif_path.rstrip('.gif').split('_')[-1]
#                     vis_map_path = os.path.join(os.path.dirname(gif_path), 'vis_maps')  # , test_idx)
#                     os.makedirs(vis_map_path, exist_ok=True)
#                     for i in range(len(pred_save)):
#                         Image.fromarray(pred_save[i]).save(f'{vis_map_path}/frame_{i}.png')
#
#                 os.makedirs(os.path.dirname(gif_path), exist_ok=True)
#                 if not out_mp4:
#                     grid = save_gifs_as_grid(video_save, gt_save, pred_save, output_path=gif_path,
#                                              fixed_height=resolution)
#                 else:
#                     grid = save_grid_to_mp4(video_save, gt_save, pred_save,
#                                             output_path=gif_path.replace('.gif', '.mp4'), fixed_height=video.shape[-2])
#
#
#             except Exception as e:
#                 logging.info(f"Error in saving video: {e}")
#                 pass
#
#         return loss, grid
#
#     # not using mamba
#     def train_single(self, batch, loss_type='l1', vis_training=False, savedir='debug_training'):
#
#         images, gt_depth = batch
#         images = images.to(torch.cuda.current_device()).squeeze(1)
#         # gt_depth = gt_depth.to(torch.cuda.current_device()).squeeze(1) *100
#         gt_depth = gt_depth.to(torch.cuda.current_device()).squeeze(1)
#
#         assert images.ndim == 4, f"{images.shape}; image ndim should only be 4"
#
#         patch_h, patch_w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
#
#         dpt_features = self.get_dpt_features(images)
#         pred_depth = self.final_head(dpt_features, patch_h, patch_w)  # (B, H, W)
#
#         # valid_mask = gt_depth >=0
#         valid_mask = gt_depth > 0
#         loss = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
#
#         grid = None
#         if vis_training:
#             if dist.get_rank() == 0:
#                 with torch.no_grad():
#                     import os;
#                     os.makedirs(savedir, exist_ok=True)
#                     try:
#                         pred_depth = torch.clip(pred_depth, min=0)
#                         pred_save = depth_to_np_arr(pred_depth)
#                         video_save = torch_batch_to_np_arr(images)
#                         gt_save = depth_to_np_arr(gt_depth)
#                         grid = save_gifs_as_grid(video_save, pred_frames=pred_save, gt_frames=gt_save,
#                                                  output_path=f'{savedir}/{vis_training}.gif', fixed_height=224)
#                     except Exception as e:
#                         logging.info(f"Visualization error for iter {vis_training}: {e}")
#                         pass
#             dist.barrier()
#
#         return loss, grid
