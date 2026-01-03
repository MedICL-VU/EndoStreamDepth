import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.transform import resize
import os
import glob
from pathlib import Path
import cv2
import csv
from natsort import natsorted
from boundary_metric import SI_boundary_F1, SI_boundary_F1_masked

# from results_util import *   # you can remove this if everything is defined locally


def scale_predictions(gt, est):

    # Flatten the ground truth and estimated depth arrays
    gt_flat = gt.flatten()
    est_flat = est.flatten()

    # Calculate the scaling factor using least median squares
    def error_func(scale, gt, est):
        return np.median((gt - scale * est) ** 2)

    # Initial guess for the scaling factor
    initial_scale = 1.0
    from scipy.optimize import leastsq
    # Use least median squares to find the optimal scaling factor
    result = leastsq(error_func, initial_scale, args=(gt_flat, est_flat))
    optimal_scale = result[0][0]

    # Scale the estimated depth array
    scaled_est = est * optimal_scale

    return scaled_est


# ----------------- Metrics & Helpers -----------------

def eval_depth(pred, target, mask=None, eps=1e-8):
    """
    Computes global metrics + depth-range metrics.
    Returns:
        dict of metrics
    """
    assert pred.shape == target.shape
    pred = pred * 100
    pred = np.clip(pred, 0, 100)
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    # pred = scale_predictions(target, pred)

    valid = (pred > eps) & (target > eps) if mask is None else (mask & (pred > eps) & (target > eps))
    if not np.any(valid):
        raise ValueError("No valid pixels after masking.")

    p = pred[valid]
    t = target[valid]

    # ---------- Global metrics ----------
    diff = p - t
    diff_log = np.log(p + eps) - np.log(t + eps)

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    abs_rel = float(np.mean(np.abs(diff) / (t + eps)))
    sq_rel  = float(np.mean((diff**2) / (t + eps)))

    thresh = np.maximum(t / (p + eps), p / (t + eps))
    d1 = float(np.mean(thresh < 1.25))
    # d1 = float(np.mean(thresh < 1.1))
    d2 = float(np.mean(thresh < 1.25**2))
    d3 = float(np.mean(thresh < 1.25**3))

    rmse_log = float(np.sqrt(np.mean(diff_log**2)))
    log10    = float(np.mean(np.abs(np.log10(p + eps) - np.log10(t + eps))))
    silog    = float(np.sqrt(np.mean(diff_log**2) - 0.5*(np.mean(diff_log)**2)))

    # ---------- Depth bins ----------
    bins = {
        "near": t < 30,
        "mid":  (t >= 30) & (t < 70),
        "far":  t >= 70,
    }

    range_metrics = {}
    for name, m in bins.items():
        if np.any(m):
            pp = p[m]
            tt = t[m]
            dd = pp - tt

            range_metrics[f"mae_{name}"] = float(np.mean(np.abs(dd)))
            range_metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(dd**2)))
            range_metrics[f"absrel_{name}"] = float(np.mean(np.abs(dd) / (tt + eps)))

            # range d1
            th = np.maximum(tt / (pp + eps), pp / (tt + eps))
            range_metrics[f"d1_{name}"] = float(np.mean(th < 1.25))
        else:
            # no pixels in this bin
            range_metrics[f"mae_{name}"]   = np.nan
            range_metrics[f"rmse_{name}"]  = np.nan
            range_metrics[f"absrel_{name}"] = np.nan
            range_metrics[f"d1_{name}"] = np.nan

    out = {
        "d1": d1, "d2": d2, "d3": d3,
        "abs_rel": abs_rel, "sq_rel": sq_rel,
        "rmse": rmse, "rmse_log": rmse_log,
        "log10": log10, "silog": silog,
        "mae": mae,
    }
    out.update(range_metrics)
    return out



def resize_depth_with_mask(depth, mask, out_hw):
    """
    depth: (H, W) float32/float64 depth map (metric depth)
    mask:  (H, W) bool or {0,1} valid mask (1=valid)
    out_hw: (H_out, W_out)

    Returns:
        depth_out: (H_out, W_out) float32 depth, zeros outside mask
        mask_out:  (H_out, W_out) bool mask
    """
    H_out, W_out = int(out_hw[0]), int(out_hw[1])
    assert depth.ndim == 2 and mask.ndim == 2, "depth/mask must be (H,W)"

    # 1) Resize mask first (crisp edges)
    mask_u8 = (mask > 0).astype(np.uint8)
    mask_resized_u8 = cv2.resize(mask_u8, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
    mask_out = mask_resized_u8.astype(bool)

    # 2) Resize depth with INTER_AREA
    depth = depth.astype(np.float32, copy=False)
    depth_resized = cv2.resize(depth, (W_out, H_out), interpolation=cv2.INTER_AREA).astype(np.float32)

    # 3) Apply resized mask to resized depth
    depth_out = depth_resized.copy()
    depth_out[~mask_out] = 0.0

    return depth_out, mask_out


# def frames_match(gt_path, pred_path):
#     s_gt = Path(gt_path).name
#     s_pr = Path(pred_path).name
#
#     frame_index_gt = int(s_gt.replace('_depth.tiff', ''))
#     if 'color' in s_pr:
#         frame_index_pred = int(s_pr.replace('_color_pred.npy', ''))
#     else:
#         frame_index_pred = int(s_pr[6:].replace('.npy', ''))
#
#     return frame_index_pred == frame_index_gt

def frames_match(gt_path, pred_path):
    s_gt = Path(gt_path).name
    s_pr = Path(pred_path).name

    # GT index: e.g. "0074_depth.tiff" -> 74
    frame_index_gt = int(s_gt.replace('_depth.tiff', ''))

    # Pred index:
    if 'color' in s_pr:
        # e.g. "0074_color_pred.npy" -> 74
        frame_index_pred = int(s_pr.replace('_color_pred.npy', ''))
    else:
        # e.g. "frame_0.npy" -> 0
        frame_index_pred = int(s_pr[6:].replace('.npy', ''))

    # -------- special case: desc_t4_a_p2 offset --------
    # GT files:  0074_depth.tiff ... 0147_depth.tiff  (74–147)
    # Pred files: frame_0.npy ... frame_73.npy        (0–73)
    # So: gt_index = pred_index + 74
    if 'desc_t4_a_p2' in gt_path or 'desc_t4_a_p2' in pred_path:
        frame_index_pred += 74

    return frame_index_pred == frame_index_gt

# ----------------- Main quantitative function -----------------

def get_quantative_results(depth_list, pred_list, disparity=False):
    """
    Returns:
        mean_results: dict of per-frame-averaged metrics for this dataset
        sum_results:  dict of summed metrics over frames (for global aggregation)
        nsamples:     number of valid frames
    """
    metric_keys = [
        'd1', 'd2', 'd3',
        'abs_rel', 'sq_rel',
        'rmse', 'rmse_log', 'log10', 'silog',
        'mae',
        'mae_near', 'mae_mid', 'mae_far',
        'rmse_near', 'rmse_mid', 'rmse_far',
        'absrel_near', 'absrel_mid', 'absrel_far',
        'd1_near', 'd1_mid', 'd1_far', 'f1_boundary'
    ]

    sum_results = {k: 0.0 for k in metric_keys}
    nsamples = 0

    for i in range(len(depth_list)):
        if frames_match(depth_list[i], pred_list[i]):
            pred = np.load(pred_list[i])
            #pred = cv2.resize(pred.astype(np.float32), (384, 384), interpolation=cv2.INTER_CUBIC)
            # pred = pred * 100
            depth = tiff.imread(depth_list[i])
            depth = depth / 65535.0 * 100.0
            # depth = depth / 65535.0
            depth, mask = resize_depth_with_mask(depth, depth > 0, pred.shape)


            if disparity:
                pred = 1000.0 / pred

            cur_results = eval_depth(pred, depth, mask=mask)

            try:
                kernel = np.ones((5, 5), np.uint8)  # 3×3 erosion kernel
                mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
                mask = mask.astype(bool)
                f1 = SI_boundary_F1_masked(pred, depth, mask)

                #f1 = SI_boundary_F1(pred, depth)  # pred and depth already resized + masked
            except Exception as e:
                print("Boundary F1 error:", e)
                f1 = 0.0

            cur_results['f1_boundary'] = float(f1)

            for k in metric_keys:
                sum_results[k] += cur_results[k] if not np.isnan(cur_results[k]) else 0.0
            nsamples += 1
        else:
            print('frame match failed, {},{}'.format(depth_list[i], pred_list[i]))

    if nsamples == 0:
        raise RuntimeError("No valid samples for this dataset (frame matching failed for all).")

    mean_results = {k: sum_results[k] / nsamples for k in metric_keys}

    print('==========================================================================================')
    print(" | ".join([f"{k:>12}" for k in metric_keys]))
    print(" | ".join([f"{mean_results[k]:12.4f}" for k in metric_keys]))
    print('==========================================================================================')

    return mean_results, sum_results, nsamples


# ----------------- Script entry: per-dataset + overall CSV -----------------

if __name__ == '__main__':
    test_set_name = [
        'trans_t1_a', 'trans_t1_b', 'trans_t2_a', 'trans_t2_b', 'trans_t2_c',
        'trans_t3_a', 'trans_t3_b', 'trans_t4_a', 'trans_t4_b'
    ]
#     test_set_name = [
#         'cecum_t1_a',
#         'cecum_t2_a',
#         'cecum_t3_a',
#         'sigmoid_t3_a',
#         'trans_t2_a',
#         'trans_t3_a',
#         'trans_t4_a',
#         'desc_t4_a_p2',
#     ]

    checkpoint_name = ''
    dataset_dir = ''
    pred_dir =''
    pred_dir = ''


    checkpoint_name = 'test123'
    pred_dir = f'/home/hao/hao/EndoStreamDepth/configs/{checkpoint_name}/test/cv3d'
    temporal = True
    # a = natsorted(glob.glob(os.path.join(pred_dir, 'trans_t2_c', 'depth_npy_files', '*.npy')))

    out_csv = os.path.join(pred_dir, f'{checkpoint_name}_cv3d_results_final.csv')
    #out_csv = os.path.join(pred_dir, f'{checkpoint_name}_cv3d_results_resize_384.csv')

    metric_keys = [
        'd1', 'd2', 'd3',
        'abs_rel', 'sq_rel',
        'rmse', 'rmse_log', 'log10', 'silog',
        'mae',
        'mae_near', 'mae_mid', 'mae_far',
        'rmse_near', 'rmse_mid', 'rmse_far',
        'absrel_near', 'absrel_mid', 'absrel_far',
        'd1_near', 'd1_mid', 'd1_far', 'f1_boundary'
    ]

    # For CSV rows
    rows = []  # each row: [dataset_name, d1, d2, ...]
    # Store per-dataset mean_results so we can average over videos
    per_dataset_means = []  # list of dicts

    # NEW: global accumulators over all frames (frame-weighted average)
    global_sum_results = {k: 0.0 for k in metric_keys}
    global_nsamples = 0

    for test_name in test_set_name:
        print('Processing {}'.format(test_name))

        gt_list = sorted(glob.glob(os.path.join(dataset_dir, test_name, '*_depth.tiff')))
        print('Processing {} with {} frames'.format(test_name, len(gt_list)))
        if not temporal:
            pred_list = sorted(glob.glob(os.path.join(pred_dir, test_name, '*_pred.npy')))
        else:
            pred_list = natsorted(glob.glob(os.path.join(pred_dir, test_name, 'depth_npy_files', '*.npy')))

        print('getting quantitative results')
        mean_results, sum_results, nsamples = get_quantative_results(gt_list, pred_list)

        # Store per-dataset row (already "average over frames in this video")
        row = [test_name] + [mean_results[k] for k in metric_keys]
        rows.append(row)

        # Keep mean_results for overall "average over videos"
        per_dataset_means.append(mean_results)

        # NEW: accumulate per-frame sums across datasets
        for k in metric_keys:
            global_sum_results[k] += sum_results[k]
        global_nsamples += nsamples

        print(f'dataset: {test_name} is done')

    # ----------------- Overall: average over videos (equal weight per dataset) -----------------
    if len(per_dataset_means) > 0:
        overall_mean = {}
        for k in metric_keys:
            vals = [m[k] for m in per_dataset_means if m[k] != 0]
            if len(vals) == 0:
                overall_mean[k] = 0.0
            else:
                overall_mean[k] = float(np.mean(vals))

        overall_row = ['overall_videos'] + [overall_mean[k] for k in metric_keys]
        rows.append(overall_row)

        # print overall (video-level)
        print('========================== OVERALL (avg over datasets/videos) ==========================')
        header_str = " | ".join([f"{'dataset':>12}"] + [f"{k:>12}" for k in metric_keys])
        print(header_str)
        overall_str = " | ".join([f"{overall_row[0]:>12}"] +
                                 [f"{v:12.4f}" for v in overall_row[1:]])
        print(overall_str)
        print('=======================================================================================')
    else:
        raise RuntimeError("No datasets processed, per_dataset_means is empty.")

    # ----------------- Overall performance over ALL FRAMES -----------------
    if global_nsamples > 0:
        overall_frames_mean = {k: global_sum_results[k] / global_nsamples for k in metric_keys}

        overall_frames_row = ['overall_frames'] + [overall_frames_mean[k] for k in metric_keys]
        rows.append(overall_frames_row)

        print('========================== OVERALL (avg over ALL FRAMES) =============================')
        header_str = " | ".join([f"{'dataset':>12}"] + [f"{k:>12}" for k in metric_keys])
        print(header_str)
        overall_str = " | ".join([f"{overall_frames_row[0]:>12}"] +
                                 [f"{v:12.4f}" for v in overall_frames_row[1:]])
        print(overall_str)
        print('=======================================================================================')
    else:
        raise RuntimeError("global_nsamples is 0, something went wrong.")

    # ----------------- Save CSV -----------------
    header = ['dataset'] + metric_keys

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

    print(f'Saved summary CSV to: {out_csv}')
