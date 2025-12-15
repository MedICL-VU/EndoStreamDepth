import os 
import re
import random
import logging
import shutil
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from endostreamdepth.model import EndoStreamDepth
from .helpers import LinearWarmupExponentialDecay, get_warmup_lambda

def dist_init():
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_idx = rank // local_world_size
    num_nodes = world_size // local_world_size

    seed = 42 + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # dist.barrier()   # device_ids=[local_rank]
    return dict(
        rank=rank, 
        world_size=world_size,
        local_rank=local_rank, 
        local_world_size=local_world_size,
        node_idx=node_idx, 
        num_nodes=num_nodes
    )

def setup_model(cfg, process_dict):
    '''
    1. instantiate model
    2. apply gradient checkpointing
    3. wrap with DDP
    4. setup optimizer and scheduler
    5. load checkpoint if needed
    '''
    # MODEL
    model = EndoStreamDepth(**dict(
        batch_size=cfg.training.batch_size, 
        hybrid_configs=cfg.hybrid_configs,
        training=not cfg.inference,
        **cfg.model,
    ))
    model = model.to(torch.cuda.current_device())

    # Setup optimizer and scheduler
    if not cfg.inference:
        
        optim_list = []

        for param in model.parameters():
            param.requires_grad = False

        if cfg.training.lr.vit:
            for param in model.pretrained.parameters():
                param.requires_grad = True

            vit_params = {"params": model.pretrained.parameters(), "lr": cfg.training.lr.vit}
            optim_list.append(vit_params)

        # if cfg.training.lr.dpt:
            # for param in model.depth_head.parameters():
            #     param.requires_grad = True
            #
            # dpt_params = {"params": model.depth_head.parameters(), "lr": cfg.training.lr.dpt}
            # optim_list.append(dpt_params)

        if cfg.training.lr.dpt:
            for param in model.depth_head.parameters():
                param.requires_grad = True

            deep_sup_modules = [
                model.output_conv1_level2,
                model.output_conv2_level2,
                model.output_conv1_level3,
                model.output_conv2_level3,
                model.output_conv1_level4,
                model.output_conv2_level4,
            ]
            for m in deep_sup_modules:
                for p in m.parameters():
                    p.requires_grad = True

            dpt_params = {
                "params": list(model.depth_head.parameters())
                          + list(model.output_conv1_level2.parameters())
                          + list(model.output_conv2_level2.parameters())
                          + list(model.output_conv1_level3.parameters())
                          + list(model.output_conv2_level3.parameters())
                          + list(model.output_conv1_level4.parameters())
                          + list(model.output_conv2_level4.parameters()),
                "lr": cfg.training.lr.dpt,
            }
            optim_list.append(dpt_params)
            

        if cfg.training.lr.mamba and cfg.model.use_mamba:
            for param in model.mamba.parameters():
                param.requires_grad = True
            mamba_params = {"params": model.mamba.parameters(), "lr": cfg.training.lr.mamba}
            optim_list.append(mamba_params)


        if cfg.hybrid_configs.use_hybrid:
            for param in model.teacher_model.parameters():
                param.requires_grad = False

            assert cfg.hybrid_configs.teacher_model_path is not None, 'teacher model path is not specified'
            teacher_ckpt_path = cfg.hybrid_configs.teacher_model_path
            teacher_ckpt = torch.load(teacher_ckpt_path, map_location='cpu')
            current_teacher_keys = set(model.teacher_model.state_dict().keys())
            pretrained_teacher_keys = set(teacher_ckpt['model'].keys())
            missing_keys = current_teacher_keys - pretrained_teacher_keys
            unexpected_keys = pretrained_teacher_keys - current_teacher_keys
            overlapping_keys = current_teacher_keys & pretrained_teacher_keys
            assert len(overlapping_keys)==407, 'check teacher model keys'
            model.teacher_model.load_state_dict(teacher_ckpt['model'], strict=False)


            logging.info(f"loaded teacher model from {teacher_ckpt_path.split('mamba/')[1]}!!")

            for param in model.hybrid_fusion.parameters():
                param.requires_grad = True

            fusion_group = {"params": model.hybrid_fusion.parameters(), "lr": cfg.training.lr.fusion}
            optim_list.append(fusion_group)

                    
        optimizer = torch.optim.AdamW(
            optim_list,
            betas=[0.9, 0.95]
        )
        from torch.optim.lr_scheduler import LambdaLR
        warmup_lambda = get_warmup_lambda(cfg.training.lr.warmup_steps)
        scheduler = LambdaLR(optimizer, lr_lambda=[warmup_lambda]*len(optim_list))

    else:
        optimizer, scheduler = None, None

    # Load checkpoint if specified
    train_step = 0
    if cfg.load is not None or cfg.inference:
        model = model.cpu()
        train_step = load_checkpoint(cfg, model, optimizer, scheduler)
        
        # # Move optimizer states to CUDA
        if optimizer is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(torch.cuda.current_device())

        model = model.to(torch.cuda.current_device())


    # Apply gradient checkpointing before DDP wrapping
    if cfg.training.gradient_checkpointing and not cfg.inference:
        apply_activation_checkpointing(
            getattr(model, 'pretrained'),
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda _: True  
        )
        apply_activation_checkpointing(
            getattr(model, 'depth_head'),
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda _: True  
        )
        
        # not compatible
        # if cfg.model.use_mamba:
        #     apply_activation_checkpointing(
        #         getattr(model, 'mamba'),
        #         checkpoint_wrapper_fn=checkpoint_wrapper,
        #         check_fn=lambda _: True  
        #     )

        if cfg.hybrid_configs.use_hybrid:
            apply_activation_checkpointing(
                getattr(model, 'teacher_model'),
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=lambda _: True  
            )

        logging.info("Using gradient checkpointing!")

    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[process_dict['local_rank']],
        find_unused_parameters=True,
    )

    dist.barrier()

    return model, optimizer, scheduler, train_step

def load_checkpoint(cfg, model, optimizer, lr_scheduler):
    
    if cfg.load is not None and cfg.inference and '.pth' in cfg.load: 
        checkpoint_path = cfg.load
        pretrained = False
        logging.info(f"force loading checkpoint {checkpoint_path}...")   

    else: 
        existing_ckpts = get_existing_ckpts(cfg)
        if len(existing_ckpts) > 0:
            checkpoint_path = existing_ckpts[-1]
            pretrained = False
        else:
            checkpoint_path = cfg.load
            pretrained = True
            logging.info(f"assuming checkpoint is pretrained and init new optimizer...")    
    
        logging.info(f"existing ckpts: {existing_ckpts}")
        logging.info(f"loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    


    if pretrained:
        model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        manual_filters = []

        # Check overlapping keys between pretrained and current model
        current_keys = set(model.state_dict().keys())
        pretrained_keys = set(model_state_dict.keys())
        
        overlapping = current_keys & pretrained_keys
        missing = current_keys - pretrained_keys
        unexpected = pretrained_keys - current_keys
        
        logging.info(f"Number of overlapping keys: {len(overlapping)}")
        logging.info(f"Number of missing keys: {len(missing)}")
        logging.info(f"Number of unexpected keys: {len(unexpected)}")

                
        # if len(missing) > 0:
        #     logging.info(f"Missing keys: {missing}")
        # if len(unexpected) > 0:
        #     logging.info(f"Unexpected keys: {unexpected}")
        # if len(overlapping) > 0:
        #     logging.info(f"Overlapping keys: {overlapping}")
    
        # if args.use_new_prediction_head or 'pretrained' in checkpoint_path:
        #     manual_filters.extend(['head'])

        # if len(manual_filters) > 0:
        #     model_state_dict = {k: v for k, v in model_state_dict.items() 
        #                       if not any([attr in k for attr in manual_filters])}
        
        model.load_state_dict(model_state_dict, strict=False)
        train_step = 0
    else:

        model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model.load_state_dict(model_state_dict, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        
        if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logging.info(f"scheduler loaded successfully!")
        train_step = checkpoint.get('step', 0)
    
    logging.info(f"checkpoint loaded successfully!")
    return train_step

def save_checkpoint(cfg, model, optimizer, lr_scheduler, train_step):


    checkpoint = {
        'model': model.module.state_dict(),
        'step': train_step
    }
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

    save_path = os.path.join(cfg.config_dir, f"iter_{train_step}.pth")
    if dist.get_rank() == 0:
        torch.save(checkpoint, save_path)
        logging.info(f"Saved checkpoint to {save_path} at step {train_step}")
        # cleanup(args)

    dist.barrier()

def cleanup(cfg, keep_latest_n=2, keep_freq=10000):
    '''
    keep_latest_n: number of latest checkpoints to keep
    keep_freq: overwrite keep_latest_n and continue to store checkpoints at these frequencies
    TODO: save based on validation accuracy
    '''

    existing_ckpts = get_existing_ckpts(cfg)
    for ckpt in existing_ckpts[:-keep_latest_n]:
        step = int(ckpt.split('iter_')[1].split('.')[0])
        if step % keep_freq != 0:
            if os.path.isfile(ckpt):
                os.remove(ckpt)


def get_existing_ckpts(cfg):
    existing_ckpts = [
        item for item in os.listdir(cfg.config_dir)
        if os.path.isfile(os.path.join(cfg.config_dir, item)) and re.match(r'^iter_\d+.pth$', item)
    ]

    existing_ckpts = sorted(existing_ckpts, key=lambda x: int(x.split('iter_')[1].split('.')[0]))
    existing_ckpts = [os.path.join(cfg.config_dir, ckpt) for ckpt in existing_ckpts]
    return existing_ckpts



def has_valid_gradients(model, train_step, loss, max_grad_norm=20.0, max_loss=10.0):
    """Check if gradients and loss are within valid ranges."""
    _tensor_cache = torch.tensor([0], device=torch.cuda.current_device())

    if train_step < 5000:
        max_grad_norm *= 5

    if train_step < 500:    
        max_loss *= 200
    
    # Check loss magnitude
    if loss.item() > max_loss or not torch.isfinite(loss):
        logging.warning(f"WARNING: skip optimizer.step(), as rank {dist.get_rank()} step {train_step} found loss value of {loss.item()}")
        _tensor_cache[0] = 1
    
    # Check gradients if loss was okay
    if _tensor_cache[0] == 0:
        with torch.no_grad():
            # First check for inf/nan
            for n, p in model.module.named_parameters():
                if (p.requires_grad) and (p.grad is not None):
                    invalid_grad_cnt = p.grad.numel() - torch.isfinite(p.grad).sum().item()
                    if invalid_grad_cnt > 0:
                        logging.warning(f"WARNING: skip optimizer.step(), as rank {dist.get_rank()} step {train_step} found {invalid_grad_cnt} invalid grads for {n}")
                        _tensor_cache[0] = 1
                        break
            
            # Then check gradient norms
            if _tensor_cache[0] == 0:
                for n, p in model.module.named_parameters():
                    if (p.requires_grad) and (p.grad is not None):
                        grad_norm = torch.norm(p.grad.detach(), p=2)
                        if grad_norm > max_grad_norm:
                            logging.warning(f"WARNING: skip optimizer.step(), as rank {dist.get_rank()} step {train_step} found large gradient norm {grad_norm:.1f} > {max_grad_norm} for {n}")
                            _tensor_cache[0] = 1
                            break
    
    # Gather results from all processes
    skip_optim_step_list = [torch.tensor([0], device=torch.cuda.current_device()) for _ in range(dist.get_world_size())]
    dist.all_gather(skip_optim_step_list, _tensor_cache)
    
    return not any(t.item() == 1 for t in skip_optim_step_list)