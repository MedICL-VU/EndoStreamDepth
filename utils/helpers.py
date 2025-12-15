import math
import numpy as np 
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
import matplotlib
import logging
import json, argparse
import re




def parse_dict_arg(s):
    """Parse a string representation of a dictionary into an actual dictionary"""
    if not s:
        return {}
    try:
        # First, standardize the string by removing whitespace and using double quotes
        s = s.strip()
        s = s.replace("'", '"')
        
        # Handle the case where the string doesn't start/end with braces
        if not s.startswith('{'):
            s = '{' + s
        if not s.endswith('}'):
            s = s + '}'
            
        # Split by commas, but first replace = with :
        parts = re.split(r',(?![^\[]*\])', s.replace('=', ':').strip('{}'))
        pairs = [pair.strip() for pair in parts]

        
        # Build proper JSON string
        json_pairs = []
        for pair in pairs:
            if pair:  # Skip empty pairs
                key, value = pair.split(':')
                # Ensure key is properly quoted
                key = '"' + key.strip().strip('"') + '"'
                # Handle value based on type
                value = value.strip()
                if value.lower() == 'true':
                    value = 'true'
                elif value.lower() == 'false':
                    value = 'false'
                elif value.replace('.', '').replace('-', '').isdigit():
                    pass  # Keep numbers as is
                elif value.startswith('[') and value.endswith(']'):
                    # Handle list values
                    try:
                        # Try to parse as list of numbers
                        nums = [int(x.strip()) if x.strip().isdigit() else float(x.strip())
                               for x in value.strip('[]').split(',')]
                        value = str(nums).replace("'", '"')  # Convert to JSON-compatible string
                    except ValueError:
                        # If not numbers, treat as strings
                        value = value.replace("'", '"')
                else:
                    value = '"' + value.strip('"') + '"'
                json_pairs.append(f"{key}:{value}")
                
        json_str = '{' + ','.join(json_pairs) + '}'
        return json.loads(json_str)
        
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")


def test_step(model, test_dataloader):
    model.eval()
    total_loss = 0
    total_samples = 0
    test_loss_details = {}
    
    with torch.no_grad():
        for batch in test_dataloader:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, loss_details = model(batch)
            
            if dist.get_world_size() > 1:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            
            batch_size = batch[0]['img'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Accumulate loss details
            if dist.get_rank() == 0:
                loss_details = gather_and_avg_dict(loss_details)
                for k, v in loss_details.items():
                    if 'conf' not in k:
                        test_loss_details[k] = test_loss_details.get(k, 0) + v * batch_size
        
    # Calculate averages
    avg_loss = total_loss / total_samples
    for k in test_loss_details:
        test_loss_details[k] = test_loss_details[k] / total_samples
    
    model.train()
    return avg_loss, test_loss_details



class LinearWarmupExponentialDecay(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, peak_lr: float, min_lr: float, decay_steps: int = None, last_epoch: int = -1):
        """
        Custom learning rate scheduler with a linear warm-up phase followed by an exponential decay phase.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of steps for the linear warm-up.
            peak_lr (float): Peak learning rate after warm-up.
            min_lr (float): Minimum learning rate after decay.
            decay_steps (int): Number of steps over which to decay from peak_lr to min_lr.
                             If None, uses warmup_steps * 10 as default.
            last_epoch (int): The index of the last iteration. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.decay_steps = decay_steps if decay_steps is not None else warmup_steps * 40 # 2000 / 80k
        self.eps = 1e-9  # Starting learning rate for warm-up
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate for the current step.
        """
        if self.last_epoch < self.warmup_steps:
            # Linear warm-up phase
            warmup_factor = (self.peak_lr - self.eps) / self.warmup_steps
            lr = [self.eps + warmup_factor * self.last_epoch for _ in self.optimizer.param_groups]
        else:
            if self.min_lr == self.peak_lr:
                lr = [self.peak_lr for _ in self.optimizer.param_groups]
            else:
                # Exponential decay phase
                steps_since_warmup = self.last_epoch - self.warmup_steps
                decay_factor = math.exp(
                    math.log(self.min_lr / self.peak_lr) * 
                    min(1.0, steps_since_warmup / self.decay_steps)
                )
                lr = [max(self.min_lr, self.peak_lr * decay_factor) for _ in self.optimizer.param_groups]
        
        return lr

def get_warmup_lambda(num_warmup_steps):
    """
    Creates a lambda function for learning rate warm-up.
    Starts at 0.02 (1/50) of the base learning rate and 
    linearly increases to 1.0 over num_warmup_steps.
    
    Args:
        num_warmup_steps: Number of warm-up steps.
    
    Returns:
        A lambda function that returns a multiplier for the base learning rate.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear increase from 0.0001 to 1.0
            return 0.0001 + (1.0 - 0.0001) * (current_step / (num_warmup_steps - 1))
        return 1.0
    return lr_lambda
    
def save_images_as_grid(imgs, fixed_height=256, spacing=5, max_per_row=5):
    """
    Save a grid of images with a maximum number of images per row.

    :param imgs: List of NumPy images
    :param fixed_height: Fixed height for each image in the grid
    :param spacing: Space between images in pixels
    :param max_per_row: Maximum number of images per row
    """
    row_widths = []
    row_images = []
    current_row = []

    from PIL import Image
    # Process images and organize them into rows
    for np_img in imgs:
        img = Image.fromarray(np_img)
        aspect_ratio = img.width / img.height
        new_width = int(fixed_height * aspect_ratio)
        resized_img = img.resize((new_width, fixed_height))

        if len(current_row) < max_per_row:
            current_row.append(resized_img)
        else:
            row_widths.append(sum(img.width for img in current_row) + spacing * (len(current_row) - 1))
            row_images.append(current_row)
            current_row = [resized_img]

    # Add last row
    if current_row:
        row_widths.append(sum(img.width for img in current_row) + spacing * (len(current_row) - 1))
        row_images.append(current_row)

    total_width = max(row_widths)
    total_height = fixed_height * len(row_images) + spacing * (len(row_images) - 1)

    # Create a new blank image with a white background
    grid_img = Image.new('RGB', (total_width, total_height), color='white')

    # Paste each resized image into the grid
    y_offset = 0
    for row in row_images:
        x_offset = 0
        for img in row:
            grid_img.paste(img, (x_offset, y_offset))
            x_offset += img.width + spacing
        y_offset += fixed_height + spacing

    # Return the grid image
    return grid_img


def torch_batch_to_np_arr(batch, assume_neg1_pos1=False):
    '''
    Convert a torch batch of images B,3,H,W to list of np images with shape H,W,3
    Args:
        batch: torch tensor of shape B,3,H,W
        assume_neg1_pos1: if True, assumes input is in [-1,1] range and uses 127.5 * x + 128 conversion
                         if False, normalizes input to [0,1] based on min/max values
    '''
    np_imgs = []

    if assume_neg1_pos1:
        batch = torch.clamp(127.5 * batch + 128.0, 0, 255).float().cpu().numpy().transpose(1,2,0)
    else:
        batch = (batch - batch.min()) / (batch.max() - batch.min() + 1e-8)
        batch = (255.0 * batch).float().cpu().numpy().transpose(0,2,3,1)

    for i in range(len(batch)):
        np_imgs.append(batch[i].astype(dtype=np.uint8))

    return np_imgs

def depth_to_np_arr(depth, disparity=False):
    # torch batch of B,H,W
    if isinstance(depth, list):
        depth = torch.stack(depth)
    if not disparity:
        depth = depth.detach().cpu().float()
        #print(np.unique(depth))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) # nan happens when max == min
    else:
        valid = depth > 0
        inv = torch.zeros_like(depth, dtype=torch.float32)
        inv[valid] = 1000.0 / depth[valid].float()
        #inv[valid] = torch.clamp(inv[valid], max=100.0)  # cap anything above 100
        mask = (inv >= 0) & (inv <= 100)
        inv = inv * mask  # out-of-range becomes 0

        if valid.any():
            v = inv[valid]
            # lo, hi = torch.quantile(v, torch.tensor([0.025, 0.975]))
            # inv[valid] = (v - lo) / (hi - lo + 1e-12)
            inv[valid] = (v - v.min()) / (v.max() - v.min() + 1e-12)

        inv = inv.detach().cpu().float()
        depth = inv


    depth = depth.numpy()
    cmap = matplotlib.colormaps.get_cmap('inferno')

    np_depth = []
    for i in range(len(depth)):
        d = depth[i]
        if d.min() == d.max():
            logging.info(f"Depth min and max are the same for {i}")
            d = np.zeros_like(d)
        image = cmap(d)[:, :, :3] * 255
        np_depth.append(image.astype(np.uint8))

    return np_depth

def vis_depth_metric(depth, input_is_inverse=True):
    if input_is_inverse:
        invalid_mask = depth <= 0
        depth[invalid_mask] = 0.001
        depth = 1 / depth
    
    # Convert to numpy if it's a torch tensor
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    
    # Define rainbow colors (RGB values)
    colors = {
        0: [255, 0, 0],      # Red (0-10)
        2: [255, 127, 0],   # Orange (10-20)
        5: [255, 255, 0],   # Yellow (20-30)
        10: [0, 255, 0],     # Green (30-40)
        15: [0, 255, 255],   # Cyan (40-50)
        20: [0, 127, 255],   # Light Blue (50-60)
        30: [0, 0, 255],     # Blue (60-70)
        50: [0, 0, 0]        # Black (>70)
    }
    
    depth_maps = []
    for d in depth:  # Iterate over batch dimension
        # Initialize RGB image
        rgb_map = np.zeros((*d.shape, 3), dtype=np.uint8)
        
        # Apply colors based on depth ranges
        thresholds = sorted(colors.keys())
        for i in range(len(thresholds)-1):
            lower = thresholds[i]
            upper = thresholds[i+1]
            mask = (d >= lower) & (d < upper)
            rgb_map[mask] = colors[lower]
        
        # Handle values above the last threshold
        rgb_map[d >= thresholds[-1]] = colors[thresholds[-1]]
        
        depth_maps.append(rgb_map)
    
    return depth_maps

def gather_and_avg_dict(local_dict):
    # world_size = dist.get_world_size()

    # if dist.get_rank() == 0:
    #     import ipdb; ipdb.set_trace()
    # dist.barrier()
    
    # mean_dict = {}
    # for key, value in local_dict.items():
    #     if isinstance(value, torch.Tensor):
    #         dist.reduce(value.detach(), dst=0, op=dist.ReduceOp.AVG)
    #         mean_dict[key] = value.clone()
    #     else:
    #         # Keep existing logic for non-tensor values
    #         if isinstance(value, (list, tuple, np.ndarray)):
    #             value = float(value)
    #         try:
    #             gathered_values = [None for _ in range(world_size)]
    #             dist.all_gather_object(gathered_values, value)
    #             mean_dict[key] = sum(gathered_values) / len(gathered_values)
    #         except Exception as e:
    #             print(f"Warning: Failed to gather value for key {key}: {e}")
    #             mean_dict[key] = value
    
    # return mean_dict

    # just use rank 0 for now
    if dist.get_rank() == 0:
        return local_dict
    else:
        # print('not rank 0, returning empty dict')
        return {}

def save_gifs_as_grid(video_frames, gt_frames, pred_frames, output_path, fixed_height=256, spacing=5, duration=110):
    """
    Create a GIF with two or three columns: video, (optional) ground truth, and predictions.
    Each frame will show the corresponding images side by side.

    Args:
        video_frames: List of NumPy arrays for the video frames
        gt_frames: List of NumPy arrays for the ground truth depth, or None
        pred_frames: List of NumPy arrays for the predicted depth
        fixed_height: Fixed height for each image in pixels
        spacing: Space between columns in pixels
    
    Returns:
        tuple: (PIL Image object containing the animated GIF grid, 
               numpy array of shape (T, C, H, W) containing concatenated frames)
    """
    from PIL import Image
    
    frames = []
    concat_frames = []  # Will store the numpy arrays
    n_frames = len(video_frames)
    assert len(pred_frames) == n_frames, "Video and prediction frames must have same length"
    if gt_frames is not None:
        assert len(gt_frames) == n_frames, "Ground truth frames must have same length"
    
    for i in range(n_frames):
        # Convert and resize each frame
        video_img = Image.fromarray(video_frames[i])
        pred_img = Image.fromarray(pred_frames[i])
        
        # Create list of images to process
        frame_images = [video_img]
        if gt_frames is not None:
            gt_img = Image.fromarray(gt_frames[i])
            frame_images.append(gt_img)
        frame_images.append(pred_img)
        
        # Maintain aspect ratio while resizing
        resized_images = []
        for img in frame_images:
            aspect_ratio = img.width / img.height
            new_width = int(fixed_height * aspect_ratio)
            resized = img.resize((new_width, fixed_height), Image.Resampling.LANCZOS)
            resized_images.append(resized)
        
        # Calculate total width needed
        total_width = sum(img.width for img in resized_images) + spacing * (len(resized_images) - 1)
        # Create a new frame with white background
        frame = Image.new('RGB', (total_width, fixed_height), color='white')
        
        # Paste images with spacing
        x_offset = 0
        for img in resized_images:
            frame.paste(img, (x_offset, 0))
            x_offset += img.width + spacing
            
        frames.append(frame)
        
        # Create concatenated numpy array for this timestep
        np_images = [np.array(img) for img in resized_images]
        # Convert to shape (C, H, W)
        np_images = [img.transpose(2, 0, 1) for img in np_images]
        # Concatenate along width dimension
        concat_frame = np.concatenate(np_images, axis=2)  # Concatenate along W dimension
        concat_frames.append(concat_frame)
    
    # Create animated GIF
    first_frame = frames[0]
    first_frame.save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    # Stack all concatenated frames along time dimension
    concat_frames = np.stack(concat_frames, axis=0)  # Shape: (T, C, H, W)
    
    np_frames = [np.array(frame) for frame in frames]
    grid_img = save_images_as_grid(np_frames, fixed_height=fixed_height, spacing=spacing, max_per_row=1)
    
    return {'stacked_frames': concat_frames, 'grid_img': grid_img}


def save_grid_to_mp4(video_frames, gt_frames, pred_frames, output_path, fixed_height=256, spacing=5, fps=24):
    """
    Create MP4 videos: one with all frames side by side, and one with only predictions.
    
    Args:
        video_frames: List of NumPy arrays for the video frames
        gt_frames: List of NumPy arrays for the ground truth depth, or None
        pred_frames: List of NumPy arrays for the predicted depth
        output_path: Path where the MP4 file will be saved (pred-only video will be saved with '_pred' suffix)
        fixed_height: Fixed height for each image in pixels
        spacing: Space between columns in pixels
        fps: Frames per second for the output video
    """
    from PIL import Image
    import cv2
    
    frames = []
    concat_frames = []  # Will store the numpy arrays
    n_frames = len(video_frames)
    assert len(pred_frames) == n_frames, "Video and prediction frames must have same length"
    if gt_frames is not None:
        assert len(gt_frames) == n_frames, "Ground truth frames must have same length"
    #video_frames, gt_frames, pred_frames = resize_all_to_518(video_frames, gt_frames, pred_frames)

    # Process first frame to get dimensions for video writer
    video_img = Image.fromarray(video_frames[0])
    pred_img = Image.fromarray(pred_frames[0])
    
    frame_images = [video_img]
    if gt_frames is not None:
        gt_img = Image.fromarray(gt_frames[0])
        frame_images.append(gt_img)
    frame_images.append(pred_img)
    
    # Resize first frame images to get final dimensions
    for img in frame_images:
        aspect_ratio = img.width / img.height
        new_width = int(fixed_height * aspect_ratio)
        img.thumbnail((new_width, fixed_height), Image.Resampling.LANCZOS)
    
    # Calculate total width needed
    total_width = sum(img.width for img in frame_images) + spacing * (len(frame_images) - 1)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (total_width, fixed_height))
    
    # Create a second video writer for predictions-only
    pred_output_path = output_path.rsplit('.', 1)[0] + '_pred.mp4'
    pred_img = Image.fromarray(pred_frames[0])
    aspect_ratio = pred_img.width / pred_img.height
    pred_width = int(fixed_height * aspect_ratio)
    pred_writer = cv2.VideoWriter(pred_output_path, fourcc, fps, (pred_width, fixed_height))
    
    # Process all frames
    for i in range(n_frames):
        # Convert and resize each frame
        video_img = Image.fromarray(video_frames[i])
        pred_img = Image.fromarray(pred_frames[i])
        
        frame_images = [video_img]
        if gt_frames is not None:
            gt_img = Image.fromarray(gt_frames[i])
            frame_images.append(gt_img)
        frame_images.append(pred_img)
        
        # Maintain aspect ratio while resizing
        for img in frame_images:
            aspect_ratio = img.width / img.height
            new_width = int(fixed_height * aspect_ratio)
            img.thumbnail((new_width, fixed_height), Image.Resampling.LANCZOS)
        
        # Create a new frame with white background
        frame = Image.new('RGB', (total_width, fixed_height), color='white')
        
        # Paste images with spacing
        x_offset = 0
        for img in frame_images:
            frame.paste(img, (x_offset, 0))
            x_offset += img.width + spacing
        
        frames.append(frame)
        
        # Create concatenated numpy array for this timestep
        np_images = [np.array(img) for img in frame_images]
        # Convert to shape (C, H, W)
        np_images = [img.transpose(2, 0, 1) for img in np_images]
        # Concatenate along width dimension
        concat_frame = np.concatenate(np_images, axis=2)  # Concatenate along W dimension
        concat_frames.append(concat_frame)
        
        # Convert PIL image to OpenCV format (RGB to BGR) and write to video
        opencv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video_writer.write(opencv_frame)
        
        # Write prediction-only frame
        pred_img = Image.fromarray(pred_frames[i])
        pred_img.thumbnail((pred_width, fixed_height), Image.Resampling.LANCZOS)
        pred_opencv = cv2.cvtColor(np.array(pred_img), cv2.COLOR_RGB2BGR)
        pred_writer.write(pred_opencv)
    
    # Release both video writers
    video_writer.release()
    pred_writer.release()

    # Stack all concatenated frames along time dimension
    concat_frames = np.stack(concat_frames, axis=0)  # Shape: (T, C, H, W)
    
    np_frames = [np.array(frame) for frame in frames]
    grid_img = save_images_as_grid(np_frames, fixed_height=fixed_height, spacing=spacing, max_per_row=1)
    
    return {
        'stacked_frames': concat_frames,
        'grid_img': grid_img,
        'video_path': output_path,
        'pred_video_path': pred_output_path
    }




# for now, just copy to under temporal_mask = valid_diff & small_change_mask in code

def vis_temporal_mask():
    with torch.no_grad():
        # Create visualization of masks and differences
        B, T, H, W = gt_temporal.shape
        vis_dict = {
            'valid_diff': valid_diff[0].float(),  # (T-1, H, W)
            'small_change': small_change_mask[0].float(),  # (T-1, H, W)
            'temporal_mask': temporal_mask[0].float(),  # (T-1, H, W)
            'gt_diff': gt_diff[0].abs(),  # (T-1, H, W)
            'pred_diff': pred_diff[0].abs(),  # (T-1, H, W)
            'relative_threshold': relative_threshold[0]  # (T-1, H, W)
        }
        
        # Save visualizations
        import matplotlib.pyplot as plt
        import os
        os.makedirs(f'{savedir}/temporal_vis', exist_ok=True)
        
        for t in range(T-1):  # For each temporal difference
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Frame {t} to {t+1}')
            
            # Plot each mask/metric
            for idx, (name, data) in enumerate(vis_dict.items()):
                ax = axes[idx//3, idx%3]
                if name != 'relative_threshold':
                    im = ax.imshow(data[t].float().cpu().numpy())
                    plt.colorbar(im, ax=ax)
                else:
                    im = ax.imshow(data[t].float().cpu().numpy(), vmin=0, vmax=0.5)
                    plt.colorbar(im, ax=ax)
                ax.set_title(name)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{savedir}/temporal_vis/frame_{vis_training}_{t}.png')
            plt.close()
            
        # Also save a video of the masks
        import cv2
        import numpy as np
        
        mask_video = []
        for t in range(T-1):
            # Create a color-coded visualization
            mask_frame = np.zeros((H, W, 3), dtype=np.uint8)
            # Red: valid_diff
            mask_frame[valid_diff[0, t].float().cpu().bool()] = [255, 0, 0]
            # Green: small_change
            mask_frame[small_change_mask[0, t].float().cpu().bool()] = [0, 255, 0]
            # Yellow: both (actual temporal mask)
            mask_frame[temporal_mask[0, t].float().cpu().bool()] = [255, 255, 0]
            mask_video.append(mask_frame)
        
        # Save as video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{savedir}/temporal_vis/masks_{vis_training}.mp4', 
                            fourcc, 5.0, (W, H))
        for frame in mask_video:
            out.write(frame)
        out.release()



import cv2
import numpy as np

def resize_all_to_518(video_frames, gt_frames=None, pred_frames=None):
    """
    Resize all frames to exactly 518x518 using INTER_LINEAR.

    Args:
        video_frames: list of np arrays (HxW or HxWx3/4)
        gt_frames:    list of depth arrays (HxW), or None
        pred_frames:  list of depth arrays (HxW), or None

    Returns:
        (video_resized, gt_resized, pred_resized)
    """
    target = (518, 518)  # (W,H) for cv2

    # --- video (RGB) ---
    vid_resized = []
    for img in video_frames:
        if img.ndim == 3 and img.shape[2] == 4:  # drop alpha if RGBA
            img = img[:, :, :3]
        vid_resized.append(cv2.resize(img, target, interpolation=cv2.INTER_LINEAR))

    # --- GT depth ---
    gt_resized = None
    if gt_frames is not None:
        gt_resized = [cv2.resize(d.astype(np.float32), target, interpolation=cv2.INTER_LINEAR)
                      for d in gt_frames]

    # --- Pred depth ---
    pred_resized = None
    if pred_frames is not None:
        pred_resized = [cv2.resize(d.astype(np.float32), target, interpolation=cv2.INTER_LINEAR)
                        for d in pred_frames]

    return vid_resized, gt_resized, pred_resized
