import argparse
import numpy as np
import torch
from PIL import Image

from .boundary_metrics import SI_boundary_F1, SI_boundary_Recall

def compute_depth_metrics(pred_depth, gt_depth, align=True, use_boundary=True, max_depth=70):
    """
    Compute standard depth evaluation metrics over valid pixels:
      - Abs Rel:    Mean absolute relative error.
      - Sq Rel:     Mean squared relative error.
      - RMSE:       Root mean squared error.
      - Log RMSE:   Root mean squared logarithmic error.
      - δ < 1.25:   Percentage of pixels with max(pred/gt, gt/pred) < 1.25.
      - δ < 1.25^2: Same as above with 1.25^2.
      - δ < 1.25^3: Same as above with 1.25^3.
    
    Both inputs are NumPy arrays.
    **Assume all inputs are inverse!!!!
    **convention is to align in disparity, calculate metrics in depth
    """


    if torch.is_tensor(pred_depth):
        pred_depth = pred_depth.detach().cpu().numpy()
    if torch.is_tensor(gt_depth):
        gt_depth = gt_depth.detach().cpu().numpy()

    # scale_std, shift_std = depth_consistency_via_scaleshift_std(pred_depth, gt_depth, max_depth)
    # results = depth_consistency_via_scaleshift_std(pred_depth, gt_depth, max_depth)

    # results = depth_consistency_scale_std(pred_depth, gt_depth,
    #                                               max_depth=70,
    #                                               window_sizes=[10, 50, 100, 150, 198, 200, 500, 1000])


    # during preprocessing, we set invalid pixels to -1
    gt_valid_pixel_mask = (gt_depth >= 0)

    if align:
        pred_depth, s, t = align_depths_lstsq(pred_depth, gt_depth, max_depth=max_depth)
        

    # metrics in depth space
    pred_depth = 1 / np.clip(pred_depth, 1e-3, a_max=None)
    gt_depth = 1 / np.clip(gt_depth, 1e-3, a_max=None)

    pred_depth = np.clip(pred_depth, 1e-3, a_max=max_depth)
    gt_depth = np.clip(gt_depth, 1e-3, a_max=max_depth)

    boundary_f1 = -1
    if use_boundary:
        boundary_f1s = 0 
        if pred_depth.ndim == 3:    
            for i in range(pred_depth.shape[0]):
                boundary_f1s += SI_boundary_F1(pred_depth[i], gt_depth[i])
            boundary_f1 = boundary_f1s / pred_depth.shape[0]
        else:
            boundary_f1 = SI_boundary_F1(pred_depth, gt_depth)
 
    valid_mask = (gt_depth > 0) & gt_valid_pixel_mask# & (gt_depth <= max_depth) 
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]
    
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    max_ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    threshold_1 = np.mean(max_ratio < 1.25)
    
    return {
        'abs_rel': float(abs_rel),
        'δ < 1.25': float(threshold_1),
        'boundary_f1': float(boundary_f1),
        # 'scale_std': float(scale_std),
        # 'shift_std': float(shift_std),
    }



# ---------- low-level helper ---------- #
def _solve_scale_lstsq(pred, gt, max_depth=None, eps=1e-8):
    """
    Least-squares scale:   s*pred ≈ gt   ⇒   s = (pred·gt)/(pred·pred).
    Uses only valid pixels < max_depth (if provided).
    """
    valid = np.isfinite(gt) & np.isfinite(pred)
    if max_depth is not None:
        valid &= (gt < max_depth) & (pred < max_depth)

    if valid.sum() == 0:
        return 1.0                                # degenerate frame

    p = pred[valid].ravel()
    g = gt[valid].ravel()
    return float(np.dot(p, g) / (np.dot(p, p) + eps))


# ---------- main metric ---------- #
def depth_consistency_scale_std(
    pred_depths: np.ndarray,
    gt_depths:   np.ndarray,
    max_depth:   float | None = None,
    window_sizes=None,
    relative:    bool = True,
    return_series: bool = True,
):
    """
    Temporal scale-consistency metric.

    Parameters
    ----------
    pred_depths, gt_depths : (T, H, W) arrays
    max_depth              : ignore pixels deeper than this (optional)
    window_sizes           : iterable of prefix lengths, or None for whole sequence
    relative               : if True, use std-dev of (s_t / mean_s)  (unit-free)
    return_series          : if True, also return prefix-std curve (length T)

    Returns
    -------
    * If window_sizes is None and return_series is False:
         float
    * If window_sizes is None and return_series is True:
         (float, np.ndarray[T])
    * If window_sizes is not None and return_series is False:
         dict  {f"scale_std_{w}": value, ...}
    * If window_sizes is not None and return_series is True:
         (dict, np.ndarray[T])
    """
    # 1. sanity checks ---------------------------------------------------------
    assert pred_depths.ndim == gt_depths.ndim == 3, "inputs must be (T,H,W)"
    assert pred_depths.shape == gt_depths.shape,    "prediction/GT shapes differ"

    # 2. per-frame scales ------------------------------------------------------
    scales = [
        _solve_scale_lstsq(pred, gt, max_depth)
        for pred, gt in zip(pred_depths, gt_depths)
    ]
    scales = np.asarray(scales, dtype=float)
    T = scales.size

    def _std_of(slice_):
        """Optionally normalise slice by its own mean before std-dev."""
        if relative:
            m = slice_.mean()
            if m == 0:
                return 0.0
            slice_ = slice_ / m
        return slice_.std()

    # 3. running prefix-std curve (if asked) -----------------------------------
    # if return_series:
    #     prefix_std = np.empty(T, dtype=float)
    #     for i in range(T):                     # O(T²) but T is video length
    #         prefix_std[i] = _std_of(scales[: i + 1])
    # else:
    #     prefix_std = None

    if return_series:
        if relative:
            scales_rel = scales / scales.mean()
        else:
            scales_rel = scales 

    # 4. whole-seq or windowed report -----------------------------------------
    if window_sizes is None:
        result = float(_std_of(scales))
    else:
        result = {}
        for w in window_sizes:
            if w <= T:
                result[f"scale_std_{w}"] = float(_std_of(scales[:w]))

    return (result, scales_rel) if return_series else result

def depth_consistency_via_scaleshift_std(pred_depths, gt_depths, max_depth, window_sizes=None):
    scales = []
    shifts = []

    assert pred_depths.ndim == 3 and gt_depths.ndim == 3, "check consistency metric dimensions"
    assert pred_depths.shape[0] == gt_depths.shape[0], "check consistency metric dimensions"

    num_frames = pred_depths.shape[0]
    
    for pred, gt in zip(pred_depths, gt_depths):
        _, scale, shift = align_depths_lstsq(pred, gt, max_depth)
        scales.append(scale)
        shifts.append(shift)
    
    window_sizes=[10, 50, 100, 150, 200, 500, 1000]

    if window_sizes is None:
        scale_var = np.std(scales)
        shift_var = np.std(shifts)
        
        return scale_var, shift_var
    else:
        results = {}
        for window in window_sizes:
            if window > num_frames:
                continue
                
            scale_var = np.std(scales[:window])
            shift_var = np.std(shifts[:window])
            
            results[f'scale_std_{window}'] = float(scale_var)
            results[f'shift_std_{window}'] = float(shift_var)
        
        return results



def get_args_parser():
    parser = argparse.ArgumentParser(description="Minimal Depth Evaluation with Scaling")
    parser.add_argument("--depth_gt", type=str, required=True,
                        help="Path to ground truth depth image (png or npz)")
    parser.add_argument("--depth_pred", type=str, required=True,
                        help="Path to predicted depth image (png or npz)")
    parser.add_argument("--depth_max", type=float, default=70,
                        help="Maximum depth value to consider")
    parser.add_argument("--align_with_lstsq", action='store_true', default=False,
                        help="Apply scaling and shift alignment using least squares")
    # parser.add_argument("--dataset_name", type=str, default=None, choices=['bonn', 'tum', 'davis', 'sintel', 'PointOdyssey', 'FlyingThings3D'], help="choose dataset for depth evaluation")
    return parser

def load_depth(filename):
    """
    Load a depth map from a file.
      - For PNG files, e.g. Bonn, tum, assumes a 16-bit depth image (divides by 5000).
      - For NPZ files, assumes the depth is stored under the key 'depth'.
      - For DPT files, e.g. sintel, assumes the depth is stored under the key 'depth'.
    """
    if filename.lower().endswith('.png'):
        depth_png = np.array(Image.open(filename))
        if np.max(depth_png) <= 255:
            raise ValueError("Expected a 16-bit depth PNG (max value > 255)")
        depth = depth_png.astype(np.float64) / 5000.0
        # Set missing depths (0 values) to -1
        depth[depth_png == 0] = -1.0
        return depth
    elif filename.lower().endswith('.npz'):
        data = np.load(filename)
        return data['depth']
    elif filename.lower().endswith('.dpt'):
        f = open(filename, "rb")
        depth = np.fromfile(f, dtype=np.float32, count=-1)
        return depth
    else:
        raise ValueError("Unsupported file type: {}".format(filename))


def align_depths_lstsq(pred, gt, max_depth):
    """
    Align the predicted depth to the ground truth using least squares.
    
    Solves for s and t in:
         gt ≈ s * pred + t
    only over valid pixels where gt > eps.
    
    Returns:
        aligned_pred: The aligned predicted depth (s * pred + t)
        s, t: The scale and shift factors.
    """

    # we don't want to align with pixels > max depth, but we are aligning in disparity space
    valid_mask = (gt >= 1/max_depth)
    
    pred_valid = pred[valid_mask].reshape(-1, 1)
    gt_valid = gt[valid_mask].reshape(-1, 1)
    A = np.hstack([pred_valid, np.ones_like(pred_valid)])
    result, _, _, _ = np.linalg.lstsq(A, gt_valid, rcond=None)
    s, t = result.flatten()
    aligned_pred = s * pred + t
    return aligned_pred, s, t

def compute_depth_metrics_backup(aligned_pred, gt_depth, eps=1e-5):
    """
    Compute standard depth evaluation metrics over valid pixels:
      - Abs Rel:    Mean absolute relative error.
      - Sq Rel:     Mean squared relative error.
      - RMSE:       Root mean squared error.
      - Log RMSE:   Root mean squared logarithmic error.
      - δ < 1.25:   Percentage of pixels with max(pred/gt, gt/pred) < 1.25.
      - δ < 1.25^2: Same as above with 1.25^2.
      - δ < 1.25^3: Same as above with 1.25^3.
    
    Both inputs are NumPy arrays.
    """
    valid_mask = (gt_depth > eps)
    num_valid_pixels = np.sum(valid_mask)
    pred_valid = aligned_pred[valid_mask]
    gt_valid = gt_depth[valid_mask]
    
    pred_tensor = torch.tensor(pred_valid, dtype=torch.float32)
    gt_tensor = torch.tensor(gt_valid, dtype=torch.float32)
    
    abs_rel = torch.mean(torch.abs(pred_tensor - gt_tensor) / gt_tensor).item()
    sq_rel  = torch.mean(((pred_tensor - gt_tensor) ** 2) / gt_tensor).item()
    rmse    = torch.sqrt(torch.mean((pred_tensor - gt_tensor) ** 2)).item()
    log_rmse = torch.sqrt(torch.mean((torch.log(pred_tensor) - torch.log(gt_tensor)) ** 2)).item()
    
    max_ratio = torch.max(pred_tensor / gt_tensor, gt_tensor / pred_tensor)
    threshold_1 = (max_ratio < 1.25).float().mean().item()
    threshold_2 = (max_ratio < 1.25**2).float().mean().item()
    threshold_3 = (max_ratio < 1.25**3).float().mean().item()
    
    return {
        'Abs Rel': abs_rel,
        'Sq Rel': sq_rel,
        'RMSE': rmse,
        'Log RMSE': log_rmse,
        'δ < 1.25': threshold_1,
        'δ < 1.25^2': threshold_2,
        'δ < 1.25^3': threshold_3,
        'valid_pixels': num_valid_pixels
    }

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Load ground truth and predicted depth maps.
    gt_depth = load_depth(args.depth_gt)
    pred_depth = load_depth(args.depth_pred)

    if args.depth_max:
        gt_depth = np.clip(gt_depth, a_min=1e-5, a_max=args.depth_max)
        pred_depth = np.clip(pred_depth, a_min=1e-5, a_max=args.depth_max)
    
    # Optionally align the predicted depth to the ground truth with least square.
    if args.align_with_lstsq:
        aligned_pred, s, t = align_depths_lstsq(pred_depth, gt_depth)
        print(f"Alignment factors: scale = {s:.4f}, shift = {t:.4f}")
    else:
        scale_factor = np.median(gt_depth) / np.median(pred_depth)
        aligned_pred = pred_depth * scale_factor
        print(f"Alignment factors: scale = {scale_factor:.4f}")

    # Compute and print evaluation metrics.
    metrics = compute_depth_metrics(aligned_pred, gt_depth)
    print("Depth Evaluation Metrics:")
    for key, val in metrics.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
    main()