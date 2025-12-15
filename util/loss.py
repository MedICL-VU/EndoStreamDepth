# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np


KEY_OUTPUT = 'metric_depth'


def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction


# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, pred, target, mask=None, interpolate=True, return_interpolated=False):

        if pred.shape[-1] != target.shape[-1] and interpolate:
            pred = nn.functional.interpolate(
                pred, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = pred
        else:
            intr_input = pred


        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            pred = pred[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(pred + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", pred.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(pred), torch.max(pred))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input


def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, pred, target, mask=None, interpolate=False, return_interpolated=False):
        if pred.shape[-1] != target.shape[-1] and interpolate:
            pred = nn.functional.interpolate(
                pred, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = pred
        else:
            assert pred.shape == target.shape, f"Shape mismatch: Expected same shape but got {pred.shape} and {target.shape}."
            intr_input = pred

        grad_gt = grad(target)
        grad_pred = grad(pred)
        
        if mask is None:
            loss = nn.functional.l1_loss(grad_pred[0], grad_gt[0])
            loss = loss + nn.functional.l1_loss(grad_pred[1], grad_gt[1])
        else:
            mask_g = grad_mask(mask)
            loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
            loss = loss + nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])

        if not return_interpolated:
            return loss
        return loss, intr_input


class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N,one, H, W = gt.shape
        # print("gt shape:", gt.shape)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        # implementation according to the paper.
        # ord_label = torch.ones(N, self.ord_num * 2, H, W).to(gt.device)
        # ord_label[:, 0::2, :, :] = ord_c0
        # ord_label[:, 1::2, :, :] = ord_c1
        # reimplementation for fast speed.
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # N, C, H, W = prob.shape
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_label(gt)
        # print("prob shape: {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]
        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss"""
    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        super(DiscreteNLLLoss, self).__init__()
        self.name = 'CrossEntropy'
        self.ignore_index = -(depth_bins + 1)
        # self._loss_func = nn.NLLLoss(ignore_index=self.ignore_index)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        # depth : N1HW
        # output : NCHW

        # Quantize depth log-uniformly on [1, self.beta] into self.depth_bins bins
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = depth * (self.depth_bins - 1)
        depth = torch.round(depth) 
        depth = depth.long()
        return depth
        

    
    def _dequantize_depth(self, depth):
        """
        Inverse of quantization
        depth : NCHW -> N1HW
        """
        # Get the center of the bin




    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        # assert torch.all(input <= 0), "Input should be negative"

        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        # assert torch.all(input)<=1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = self.quantize_depth(target)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # Set the mask to ignore_index
            mask = mask.long()
            input = input * mask + (1 - mask) * self.ignore_index
            target = target * mask + (1 - mask) * self.ignore_index


        input = input.flatten(2)  # N, nbins, H*W
        target = target.flatten(1)  # N, H*W
        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss
        return loss, intr_input
    



def compute_scale_and_shift(prediction, target, mask=None):
    if mask is None:
        mask = torch.ones_like(prediction)
        
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask=None, interpolate=True, return_interpolated=False):
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction

        
        prediction, target = prediction.squeeze(), target.squeeze()
        if mask is None:
            mask = torch.ones_like(prediction).to(torch.int)
        else:
            mask = mask.squeeze()
            
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])

        if not return_interpolated:
            return loss
        return loss, intr_input

def temporal_consistency_loss(pred_temporal: torch.Tensor,
                              valid_temporal: torch.Tensor,
                              eps: float = 1e-6) -> torch.Tensor:
    """
    Temporal consistency loss with per-video normalization.

    Args:
        pred_temporal:  [B, T, H, W] predicted depth (float, mm or m)
        valid_temporal: [B, T, H, W] mask (0/1 or bool), same shape
        eps: small value to avoid division by zero

    Returns:
        scalar tensor: temporal loss
    """
    # Ensure boolean mask
    valid = valid_temporal > 0
    B, T, H, W = pred_temporal.shape
    device = pred_temporal.device

    # Clone and normalize per video (per b)
    pred_norm = pred_temporal.clone()

    for b in range(B):
        v_b = valid[b]  # [T, H, W]
        if not v_b.any():
            continue

        depth_b = pred_temporal[b][v_b]  # all valid pixels across time

        # Per-video median + mean absolute deviation
        median = depth_b.median()
        mad = (depth_b - median).abs().mean()

        pred_norm[b] = (pred_temporal[b] - median) / (mad + eps)

    # Now compute pairwise L1 between consecutive frames in normalized space
    loss = pred_temporal.new_tensor(0.0, device=device)
    pair_count = 0

    for b in range(B):
        for t in range(T - 1):
            v_pair = valid[b, t] & valid[b, t + 1]  # only where both frames are valid
            if not v_pair.any():
                continue

            diff = pred_norm[b, t + 1] - pred_norm[b, t]  # [H, W]
            loss = loss + diff[v_pair].abs().mean()
            pair_count += 1

    if pair_count == 0:
        # no valid temporal pairs -> no penalty
        return pred_temporal.new_tensor(0.0, device=device)

    return loss / pair_count


class GradientEdgeLoss(nn.Module):
    """
    Gradient-based edge/structure loss for depth.
    - If use_log=True: operates on log(depth) => relative edges.
    - If use_log=False: operates on raw depth => absolute edges.
    """
    def __init__(self, use_log=True, eps=1e-6):
        super().__init__()
        self.use_log = use_log
        self.eps = eps

    @staticmethod
    def _grad2d(x):
        """
        Forward differences in x and y.
        Input:  (B,1,H,W)
        Output: gx, gy with common shape (B,1,H-1,W-1)
        """
        # x-direction: diff along W
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]      # (B,1,H,  W-1)
        # y-direction: diff along H
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]      # (B,1,H-1,W)

        # crop to common (H-1, W-1)
        gx = gx[:, :, :-1, :]                      # (B,1,H-1,W-1)
        gy = gy[:, :, :, :-1]                      # (B,1,H-1,W-1)
        return gx, gy

    def forward(self, pred_depth, gt_depth, valid_mask=None):
        """
        pred_depth, gt_depth: (B,1,H,W) or (B,H,W) in metric depth.
        valid_mask: (B,1,H,W) or (B,H,W) bool or {0,1}, optional.
        """
        # Ensure (B,1,H,W)
        if pred_depth.dim() == 3:
            pred_depth = pred_depth.unsqueeze(1)
        if gt_depth.dim() == 3:
            gt_depth = gt_depth.unsqueeze(1)

        if valid_mask is None:
            valid_mask = torch.ones_like(pred_depth, dtype=torch.bool)
        else:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.bool()

        # Optionally log-transform
        if self.use_log:
            pred = torch.clamp(pred_depth, min=self.eps).log()
            gt   = torch.clamp(gt_depth,   min=self.eps).log()
        else:
            pred = pred_depth
            gt   = gt_depth

        # Gradients
        pgx, pgy = self._grad2d(pred)
        ggx, ggy = self._grad2d(gt)

        # Downsample mask to match gradient shape
        vm = valid_mask[:, :, 1:, 1:]              # (B,1,H-1,W-1)

        if vm.sum() == 0:
            return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)

        # Differences of gradients, masked everywhere (no edge threshold)
        diff_gx = (pgx - ggx)[vm]
        diff_gy = (pgy - ggy)[vm]

        loss = (diff_gx.abs().mean() + diff_gy.abs().mean())
        return loss


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, epsilon=1e-6):
        super().__init__()
        self.lambd = lambd
        self.epsilon = epsilon

    def forward(self, pred, target, valid_mask):
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = pred.clamp(min=self.epsilon)
        target = target.clamp(min=self.epsilon)

        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])

        mean_squared = diff_log.pow(2).mean()
        mean_mean = diff_log.mean().pow(2)

        val = mean_squared - self.lambd * mean_mean
        val = torch.clamp(val, min=0.0)  # ✅ prevent sqrt of negative

        return torch.sqrt(val)
