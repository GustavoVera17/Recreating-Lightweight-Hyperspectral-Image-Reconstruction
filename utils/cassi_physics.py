"""
Physical forward simulation and inverse unshifting for Single-Disperser CASSI systems.
Follows Equations (1) and (2) of Yamawaki et al. (2023).
"""

import torch
import numpy as np

class CASSIForwardModel:
    """
    Simulates the forward observation of a Single-Disperser Coded Aperture Snapshot Spectral Imager (CASSI).
    
    Y(h, w) = sum_lambda T(h, w - psi(lambda)) * X(h, w - psi(lambda), lambda)
    
    Dispersion function psi(lambda) = lambda * shift_step (standard is 1 pixel per band).
    """
    def __init__(self, mask_path: str = None, mask_array: np.ndarray = None, shift_step: int = 1):
        if mask_array is not None:
            self.mask = mask_array.astype(np.float32)
        elif mask_path is not None:
            self.mask = np.load(mask_path).astype(np.float32)
        else:
            raise ValueError("Either mask_path or mask_array must be provided.")
            
        self.H, self.W = self.mask.shape
        self.shift_step = shift_step

    def forward(self, cube: np.ndarray) -> tuple:
        """
        cube: shape (H, W, bands) with values in [0, 1].
        Returns:
            measurement: 2D snapshot array (H, W + (bands - 1) * shift_step)
            shifted_clean: 3D aligned dispersed ground truth cube (H, W + (bands - 1) * shift_step, bands)
        """
        H, W, bands = cube.shape
        assert H == self.H and W == self.W, f"Cube spatial size ({H}, {W}) does not match mask ({self.H}, {self.W})"
        
        total_W = W + (bands - 1) * self.shift_step
        masked_cube = cube * self.mask[:, :, np.newaxis]
        
        shifted_masked = np.zeros((H, total_W, bands), dtype=np.float32)
        shifted_clean = np.zeros((H, total_W, bands), dtype=np.float32)
        
        for i in range(bands):
            offset = i * self.shift_step
            shifted_masked[:, offset : offset + W, i] = masked_cube[:, :, i]
            shifted_clean[:, offset : offset + W, i] = cube[:, :, i]
            
        measurement = np.sum(shifted_masked, axis=2)
        return measurement, shifted_clean

    @staticmethod
    def unshift(shifted_cube: torch.Tensor, original_w: int = 512, shift_step: int = 1) -> torch.Tensor:
        """
        Reverses the spectral dispersion to reconstruct the aligned (H, W, bands) cube.
        shifted_cube: (B, bands, H, total_W) or (bands, H, total_W)
        Returns:
            aligned_cube: (B, bands, H, original_w) or (bands, H, original_w)
        """
        is_batched = (shifted_cube.ndim == 4)
        if not is_batched:
            shifted_cube = shifted_cube.unsqueeze(0)
            
        B, bands, H, total_W = shifted_cube.shape
        aligned = torch.zeros((B, bands, H, original_w), dtype=shifted_cube.dtype, device=shifted_cube.device)
        
        for i in range(bands):
            offset = i * shift_step
            aligned[:, i, :, :] = shifted_cube[:, i, :, offset : offset + original_w]
            
        return aligned if is_batched else aligned[0]
