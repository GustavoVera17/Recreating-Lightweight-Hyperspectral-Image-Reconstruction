"""
Evaluation metrics for Hyperspectral Image Reconstruction.
Supports Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM),
and Spectral Angle Mapper (SAM) strictly in radians (as reported in Yamawaki et al., 2023).
"""

import torch
import torch.nn.functional as F
import numpy as np
from math import log10
from skimage.metrics import structural_similarity as ssim_metric

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes PSNR between predicted and target HSI tensors.
    Assumes values in range [0, max_val].
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return float(20.0 * log10(max_val / (mse ** 0.5)))

def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes mean structural similarity index across all spectral channels.
    Expects input tensors of shape (C, H, W) or (1, C, H, W).
    """
    if pred.ndim == 4:
        pred = pred[0]
    if target.ndim == 4:
        target = target[0]
        
    pred_np = torch.clamp(pred, 0.0, 1.0).detach().cpu().numpy().transpose(1, 2, 0)
    target_np = torch.clamp(target, 0.0, 1.0).detach().cpu().numpy().transpose(1, 2, 0)
    
    val = ssim_metric(target_np, pred_np, data_range=1.0, channel_axis=-1)
    return float(val)

def calculate_sam(pred: torch.Tensor, target: torch.Tensor, in_degrees: bool = False) -> float:
    """
    Computes the Spectral Angle Mapper (SAM).
    
    Mathematical formula:
        SAM(x, y) = arccos( (x . y) / (||x||_2 * ||y||_2) )
        
    In Yamawaki et al. (2023), SAM is reported in RADIANS (e.g., CAVE: 0.302, Harvard: 0.133).
    If in_degrees=True, converts result to degrees.
    """
    if pred.ndim == 4:
        # Flatten spatial dimensions: (B, C, H*W)
        p_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        t_flat = target.view(target.shape[0], target.shape[1], -1)
    else:
        # (C, H, W) -> (1, C, H*W)
        p_flat = pred.unsqueeze(0).view(1, pred.shape[0], -1)
        t_flat = target.unsqueeze(0).view(1, target.shape[0], -1)
        
    dot_prod = torch.sum(p_flat * t_flat, dim=1)
    norm_p = torch.norm(p_flat, p=2, dim=1)
    norm_t = torch.norm(t_flat, p=2, dim=1)
    
    denom = norm_p * norm_t
    # Where norm is 0, angle is 0
    cos_theta = torch.ones_like(dot_prod)
    valid_mask = denom > 1e-8
    cos_theta[valid_mask] = dot_prod[valid_mask] / denom[valid_mask]
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    sam_rad = torch.acos(cos_theta)
    mean_sam = torch.mean(sam_rad).item()
    
    if in_degrees:
        return float(mean_sam * 180.0 / np.pi)
    return float(mean_sam)
