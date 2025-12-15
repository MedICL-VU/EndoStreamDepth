from typing import List, Tuple

import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt



def fgbg_depth(
    d: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find foreground-background relations between neighboring pixels.

    Args:
    ----
        d (np.ndarray): Depth matrix.
        t (float): Threshold for comparison.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations.

    """
    right_is_big_enough = (d[..., :, 1:] / d[..., :, :-1]) > t
    left_is_big_enough = (d[..., :, :-1] / d[..., :, 1:]) > t
    bottom_is_big_enough = (d[..., 1:, :] / d[..., :-1, :]) > t
    top_is_big_enough = (d[..., :-1, :] / d[..., 1:, :]) > t
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )


def fgbg_depth_with_mask(
    d: np.ndarray, t: float, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes foreground-background (boundary) booleans for depth/disparity d,
    ignoring pairs where either pixel is invalid or where the denominator would be zero.

    Args:
        d (np.ndarray): Disparity (inverse depth) image, shape (H, W).
        t (float): Threshold for ratio comparison.
        valid_mask (np.ndarray): Boolean mask of shape (H, W); True where pixel is valid.

    Returns:
        tuple of 4 boolean arrays (left, top, right, bottom), each indicating a boundary.
        Each array shape matches the comparison's shape:
        - left, right => (H, W-1)
        - top, bottom => (H-1, W)
    """

    # -------------------------------------------------------------------------
    # 1) Identify valid pairs (both pixels must be valid)
    #    We do this for horizontal pairs (same row, adjacent columns)
    #    and vertical pairs (same column, adjacent rows).
    # -------------------------------------------------------------------------
    valid_pairs_h = valid_mask[..., :, :-1] & valid_mask[..., :, 1:]  # shape: (H, W-1)
    valid_pairs_v = valid_mask[..., :-1, :] & valid_mask[..., 1:, :]  # shape: (H-1, W)

    # -------------------------------------------------------------------------
    # 2) Prepare the arrays that will hold the "is big enough" flags.
    #    - right_is_big_enough, left_is_big_enough => shape (H, W-1)
    #    - bottom_is_big_enough, top_is_big_enough => shape (H-1, W)
    # -------------------------------------------------------------------------
    right_is_big_enough  = np.zeros_like(d[..., :, :-1], dtype=bool)
    left_is_big_enough   = np.zeros_like(d[..., :, :-1], dtype=bool)
    bottom_is_big_enough = np.zeros_like(d[..., :-1, :], dtype=bool)
    top_is_big_enough    = np.zeros_like(d[..., :-1, :], dtype=bool)

    # -------------------------------------------------------------------------
    # 3) Horizontal ratios
    #    We skip pairs where the denominator is 0 to avoid divide-by-zero.
    # -------------------------------------------------------------------------
    # For right_is_big_enough: ratio = d[..., :, 1:] / d[..., :, :-1]
    valid_h_for_right = valid_pairs_h & (d[..., :, :-1] != 0)
    ratio_right = d[..., :, 1:][valid_h_for_right] / d[..., :, :-1][valid_h_for_right]
    right_is_big_enough[valid_h_for_right] = (ratio_right > t)

    # For left_is_big_enough: ratio = d[..., :, :-1] / d[..., :, 1:]
    valid_h_for_left = valid_pairs_h & (d[..., :, 1:] != 0)
    ratio_left = d[..., :, :-1][valid_h_for_left] / d[..., :, 1:][valid_h_for_left]
    left_is_big_enough[valid_h_for_left] = (ratio_left > t)

    # -------------------------------------------------------------------------
    # 4) Vertical ratios
    # -------------------------------------------------------------------------
    # For bottom_is_big_enough: ratio = d[..., 1:, :] / d[..., :-1, :]
    valid_v_for_bottom = valid_pairs_v & (d[..., :-1, :] != 0)
    ratio_bottom = d[..., 1:, :][valid_v_for_bottom] / d[..., :-1, :][valid_v_for_bottom]
    bottom_is_big_enough[valid_v_for_bottom] = (ratio_bottom > t)

    # For top_is_big_enough: ratio = d[..., :-1, :] / d[..., 1:, :]
    valid_v_for_top = valid_pairs_v & (d[..., 1:, :] != 0)
    ratio_top = d[..., :-1, :][valid_v_for_top] / d[..., 1:, :][valid_v_for_top]
    top_is_big_enough[valid_v_for_top] = (ratio_top > t)

    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )

def boundary_f1(
    pr: np.ndarray,
    gt: np.ndarray,
    t: float,
    valid_mask,
    return_p: bool = False,
    return_r: bool = False,
) -> float:
    """Calculate Boundary F1 score.

    Args:
    ----
        pr (np.ndarray): Predicted depth matrix.
        gt (np.ndarray): Ground truth depth matrix.
        t (float): Threshold for comparison.
        return_p (bool, optional): If True, return precision. Defaults to False.
        return_r (bool, optional): If True, return recall. Defaults to False.

    Returns:
    -------
        float: Boundary F1 score, or precision, or recall depending on the flags.

    """
    # ap, bp, cp, dp = fgbg_depth_with_mask(pr, t, valid_mask)
    # ag, bg, cg, dg = fgbg_depth_with_mask(gt, t, valid_mask)

    ap, bp, cp, dp = fgbg_depth(pr, t)
    ag, bg, cg, dg = fgbg_depth(gt, t)

    r = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ag), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bg), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cg), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dg), 1)
    )
    p = 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ap), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bp), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cp), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dp), 1)
    )
    if r + p == 0:
        return 0.0
    if return_p:
        return p
    if return_r:
        return r

    return 2 * (r * p) / (r + p)


def get_thresholds_and_weights(
    t_min: float, t_max: float, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate thresholds and weights for the given range.

    Args:
    ----
        t_min (float): Minimum threshold.
        t_max (float): Maximum threshold.
        N (int): Number of thresholds.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray]: Array of thresholds and corresponding weights.

    """
    thresholds = np.linspace(t_min, t_max, N)
    weights = thresholds / thresholds.sum()
    return thresholds, weights




def SI_boundary_F1(
    predicted_depth: np.ndarray,
    target_depth: np.ndarray,
    t_min: float = 1.05,
    t_max: float = 1.25,
    N: int = 10,
) -> float:
    """Calculate Scale-Invariant Boundary F1 Score for depth-based ground-truth.

    Args:
    ----
        predicted_depth (np.ndarray): Predicted depth matrix.
        target_depth (np.ndarray): Ground truth depth matrix.
        t_min (float, optional): Minimum threshold. Defaults to 1.05.
        t_max (float, optional): Maximum threshold. Defaults to 1.25.
        N (int, optional): Number of thresholds. Defaults to 10.

    Returns:
    -------
        float: Scale-Invariant Boundary F1 Score.

    """
    assert predicted_depth.ndim == target_depth.ndim == 2
    thresholds, weights = get_thresholds_and_weights(t_min, t_max, N)
    valid_mask = target_depth >= 0
    f1_scores = np.array(
        [
            # boundary_f1(invert_depth(predicted_depth), invert_depth(target_depth), t)
            boundary_f1(predicted_depth, target_depth, t, valid_mask)
            for t in thresholds
        ]
    )
    return np.sum(f1_scores * weights)


def SI_boundary_Recall(
    predicted_depth: np.ndarray,
    target_mask: np.ndarray,
    t_min: float = 1.05,
    t_max: float = 1.25,
    N: int = 10,
    alpha_threshold: float = 0.1,
) -> float:
    """Calculate Scale-Invariant Boundary Recall Score for mask-based ground-truth.

    Args:
    ----
        predicted_depth (np.ndarray): Predicted depth matrix.
        target_mask (np.ndarray): Ground truth binary mask.
        t_min (float, optional): Minimum threshold. Defaults to 1.05.
        t_max (float, optional): Maximum threshold. Defaults to 1.25.
        N (int, optional): Number of thresholds. Defaults to 10.
        alpha_threshold (float, optional): Threshold for alpha masking. Defaults to 0.1.

    Returns:
    -------
        float: Scale-Invariant Boundary Recall Score.

    """
    assert predicted_depth.ndim == target_mask.ndim == 2
    thresholds, weights = get_thresholds_and_weights(t_min, t_max, N)
    thresholded_target = target_mask > alpha_threshold

    recall_scores = np.array(
        [
            edge_recall_matting(
                invert_depth(predicted_depth), thresholded_target, t=float(t)
            )
            for t in thresholds
        ]
    )
    weighted_recall = np.sum(recall_scores * weights)
    return weighted_recall

def invert_depth(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverts a depth map with numerical stability.

    Args:
    ----
        depth (np.ndarray): Depth map to be inverted.
        eps (float): Minimum value to avoid division by zero (default is 1e-6).

    Returns:
    -------
    np.ndarray: Inverted depth map.

    """
    inverse_depth = 1.0 / depth.clip(min=eps)
    return inverse_depth


def fgbg_depth_thinned(
    d: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find foreground-background relations between neighboring pixels with Non-Maximum Suppression.

    Args:
    ----
        d (np.ndarray): Depth matrix.
        t (float): Threshold for NMS.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations with NMS applied.

    """
    right_is_big_enough = nms_horizontal(d[..., :, 1:] / d[..., :, :-1], t)
    left_is_big_enough = nms_horizontal(d[..., :, :-1] / d[..., :, 1:], t)
    bottom_is_big_enough = nms_vertical(d[..., 1:, :] / d[..., :-1, :], t)
    top_is_big_enough = nms_vertical(d[..., :-1, :] / d[..., 1:, :], t)
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )


def fgbg_binary_mask(
    d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find foreground-background relations between neighboring pixels in binary masks.

    Args:
    ----
        d (np.ndarray): Binary depth matrix.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four matrices indicating
        left, top, right, and bottom foreground-background relations in binary masks.

    """
    assert d.dtype == bool
    right_is_big_enough = d[..., :, 1:] & ~d[..., :, :-1]
    left_is_big_enough = d[..., :, :-1] & ~d[..., :, 1:]
    bottom_is_big_enough = d[..., 1:, :] & ~d[..., :-1, :]
    top_is_big_enough = d[..., :-1, :] & ~d[..., 1:, :]
    return (
        left_is_big_enough,
        top_is_big_enough,
        right_is_big_enough,
        bottom_is_big_enough,
    )


def edge_recall_matting(pr: np.ndarray, gt: np.ndarray, t: float) -> float:
    """Calculate edge recall for image matting.

    Args:
    ----
        pr (np.ndarray): Predicted depth matrix.
        gt (np.ndarray): Ground truth binary mask.
        t (float): Threshold for NMS.

    Returns:
    -------
        float: Edge recall value.

    """
    assert gt.dtype == bool
    ap, bp, cp, dp = fgbg_depth_thinned(pr, t)
    ag, bg, cg, dg = fgbg_binary_mask(gt)
    return 0.25 * (
        np.count_nonzero(ap & ag) / max(np.count_nonzero(ag), 1)
        + np.count_nonzero(bp & bg) / max(np.count_nonzero(bg), 1)
        + np.count_nonzero(cp & cg) / max(np.count_nonzero(cg), 1)
        + np.count_nonzero(dp & dg) / max(np.count_nonzero(dg), 1)
    )



def connected_component(r: np.ndarray, c: np.ndarray) -> List[List[int]]:
    """Find connected components in the given row and column indices.

    Args:
    ----
        r (np.ndarray): Row indices.
        c (np.ndarray): Column indices.

    Yields:
    ------
        List[int]: Indices of connected components.

    """
    indices = [0]
    for i in range(1, r.size):
        if r[i] == r[indices[-1]] and c[i] == c[indices[-1]] + 1:
            indices.append(i)
        else:
            yield indices
            indices = [i]
    yield indices


def nms_horizontal(ratio: np.ndarray, threshold: float) -> np.ndarray:
    """Apply Non-Maximum Suppression (NMS) horizontally on the given ratio matrix.

    Args:
    ----
        ratio (np.ndarray): Input ratio matrix.
        threshold (float): Threshold for NMS.

    Returns:
    -------
        np.ndarray: Binary mask after applying NMS.

    """
    mask = np.zeros_like(ratio, dtype=bool)
    r, c = np.nonzero(ratio > threshold)
    if len(r) == 0:
        return mask
    for ids in connected_component(r, c):
        values = [ratio[r[i], c[i]] for i in ids]
        mi = np.argmax(values)
        mask[r[ids[mi]], c[ids[mi]]] = True
    return mask


def nms_vertical(ratio: np.ndarray, threshold: float) -> np.ndarray:
    """Apply Non-Maximum Suppression (NMS) vertically on the given ratio matrix.

    Args:
    ----
        ratio (np.ndarray): Input ratio matrix.
        threshold (float): Threshold for NMS.

    Returns:
    -------
        np.ndarray: Binary mask after applying NMS.

    """
    return np.transpose(nms_horizontal(np.transpose(ratio), threshold))


if __name__ == "__main__":
    import cv2 

    
    # gt: 1920x1080 -> downsample to 770x518, then upsample: 0.2716 (low res upper bound)
    # depth anything video output: 770x518; f1 0.2038; precision 0.116/0.154; recall 0.209 / 0.388
    # ours: 0.17; precision: 0.077 / 0.129; recall: 0.213/0.383
    

    #pred_depth = '/root/gene/video-depth/evaluation/baselines/Video-Depth-Anything/outputs/spring/0045/depth_npys/frame_0.npy'
    pred_depth = '/root/gene/video-depth/configs/feb18/all-enc-dec-above-1000/spring0045/depth_npy_files/0/frame_0.npy'
    gt_depth = '/root/gene/video-depth/evaluation/testdata/spring/scenes/0045/depths/disp1_left_0001.npy'
    pred_depth = np.load(pred_depth)
    gt_depth = np.load(gt_depth)
    print("shape", pred_depth.shape, gt_depth.shape)
    
    pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), 
                          interpolation=cv2.INTER_AREA)
    
    # Get edge masks for visualization
    valid_mask = gt_depth >= 0
    t = 1.15  # You can adjust this threshold
    
    # Get edge masks
    # pred_left, pred_top, pred_right, pred_bottom = fgbg_depth_with_mask(pred_depth, t, valid_mask)
    # gt_left, gt_top, gt_right, gt_bottom = fgbg_depth_with_mask(gt_depth, t, valid_mask)

    pred_left, pred_top, pred_right, pred_bottom = fgbg_depth(pred_depth, t)
    gt_left, gt_top, gt_right, gt_bottom = fgbg_depth(gt_depth, t)
    
    # Combine horizontal and vertical edges into single masks
    pred_edges = np.zeros_like(pred_depth, dtype=bool)
    gt_edges = np.zeros_like(gt_depth, dtype=bool)
    
    # Pad the smaller masks to match the original image size
    pred_edges[:, :-1] |= pred_left | pred_right
    pred_edges[:-1, :] |= pred_top | pred_bottom
    gt_edges[:, :-1] |= gt_left | gt_right
    gt_edges[:-1, :] |= gt_top | gt_bottom
    
    # Calculate percentage of boundary pixels
    total_pixels = pred_depth.size
    pred_boundary_pixels = np.count_nonzero(pred_edges)
    gt_boundary_pixels = np.count_nonzero(gt_edges)
    
    pred_boundary_percentage = (pred_boundary_pixels / total_pixels) * 100
    gt_boundary_percentage = (gt_boundary_pixels / total_pixels) * 100
    
    # Save the visualizations
    plt.figure(figsize=(20, 5))
    
    plt.subplot(141)
    plt.imshow(pred_depth)
    plt.title('Predicted Depth')
    plt.colorbar()
    
    plt.subplot(142)
    plt.imshow(gt_depth)
    plt.title('Ground Truth Depth')
    plt.colorbar()
    
    plt.subplot(143)
    plt.imshow(pred_edges, cmap='binary')
    plt.title(f'Predicted Edges ({pred_boundary_percentage:.2f}%)')
    
    plt.subplot(144)
    plt.imshow(gt_edges, cmap='binary')
    plt.title(f'Ground Truth Edges ({gt_boundary_percentage:.2f}%)')
    
    plt.tight_layout()
    plt.savefig('depth_and_edge_visualization.png')
    plt.close()
    
    print(f"Visualization saved as 'depth_and_edge_visualization.png'")
    print(f"Boundary F1 score: {SI_boundary_F1(pred_depth, gt_depth)}")
    print(f"Predicted boundary pixels: {pred_boundary_pixels} ({pred_boundary_percentage:.2f}%)")
    print(f"Ground truth boundary pixels: {gt_boundary_pixels} ({gt_boundary_percentage:.2f}%)")
    print(f"Total pixels: {total_pixels}")
