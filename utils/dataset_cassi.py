"""
PyTorch Dataset for Coded Aperture Snapshot Spectral Imaging (CASSI) on CAVE Multispectral Database.
Supports:
- Shared reproducible hardware Bernoulli mask (p=0.5).
- RAM caching for high-speed GPU training.
- Dynamic stochastic patch extraction (48x48) with random geometric augmentations (flips, rotations).
- Full scene extraction with original and unshifted ground truth for validation.
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .cassi_physics import CASSIForwardModel

# Exact 12 train / 20 test split as reported in Yamawaki et al. (2023)
YAMAWAKI_TRAIN_SCENES = [
    "balloons_ms", "beads_ms", "cd_ms", "chart_and_stuffed_toy_ms",
    "clay_ms", "cloth_ms", "egyptian_statue_ms", "face_ms",
    "fake_and_real_beers_ms", "fake_and_real_food_ms",
    "fake_and_real_lemon_slices_ms", "watercolors_ms"
]

class CASSIDataset(Dataset):
    def __init__(
        self,
        cave_raw_root: str = r"C:\CODE2026\CASSIproy2026\CASSIproy2026\CAVE_Raw",
        mask_path: str = None,
        patch_size: int = 48,
        num_patches_per_img: int = 50,
        is_train: bool = True,
        augment: bool = True
    ):
        super(CASSIDataset, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches_per_img
        self.is_train = is_train
        self.augment = augment and is_train
        
        if mask_path is None:
            mask_path = os.path.join(os.path.dirname(__file__), "mask_cassi_512x512.npy")
            
        self.physics = CASSIForwardModel(mask_path=mask_path)
        
        # Discover all available scene directories in CAVE_Raw
        all_scene_folders = {}
        for root, dirs, files in os.walk(cave_raw_root):
            for d in dirs:
                if d.endswith("_ms"):
                    full_p = os.path.join(root, d)
                    # Check if contains PNGs
                    if len(glob.glob(os.path.join(full_p, "**", "*.png"), recursive=True)) >= 31:
                        all_scene_folders[d] = full_p
                        
        # Filter scenes based on train/test split
        self.scene_names = []
        self.scene_paths = []
        for name, path in sorted(all_scene_folders.items()):
            is_in_train_list = (name in YAMAWAKI_TRAIN_SCENES)
            if self.is_train and is_in_train_list:
                self.scene_names.append(name)
                self.scene_paths.append(path)
            elif (not self.is_train) and (not is_in_train_list):
                self.scene_names.append(name)
                self.scene_paths.append(path)
                
        mode_str = "TRAIN (12 scenes)" if self.is_train else "TEST (20 scenes)"
        print(f"[*] Initializing CASSIDataset [{mode_str}]. Found {len(self.scene_paths)} scenes.")
        
        # Preload cubes and simulate CASSI measurements into RAM
        self.cubes_clean_raw = []   # Original aligned cubes (512, 512, 31)
        self.meas_cache = []        # Dispersed 2D snapshots (512, 542)
        self.shifted_cache = []     # Dispersed 3D targets (512, 542, 31)
        
        for p in self.scene_paths:
            cube = self._load_cave_cube(p)
            meas, shifted = self.physics.forward(cube)
            self.cubes_clean_raw.append(cube)
            self.meas_cache.append(meas)
            self.shifted_cache.append(shifted)
            
        print(f"[+] Loaded {len(self.scene_paths)} cubes into RAM cache successfully.")

    def _load_cave_cube(self, folder_path: str) -> np.ndarray:
        search_pattern = os.path.join(folder_path, "**", "*.png")
        all_pngs = sorted(glob.glob(search_pattern, recursive=True))
        
        band_files = []
        for f in all_pngs:
            filename = os.path.basename(f)
            name_no_ext = os.path.splitext(filename)[0]
            if name_no_ext[-2:].isdigit():
                band_files.append(f)
                
        band_files = sorted(band_files)[:31]
        if len(band_files) != 31:
            raise ValueError(f"Error loading {folder_path}: found {len(band_files)} bands, required 31.")
            
        cube = []
        for file in band_files:
            img = Image.open(file)
            img_arr = np.array(img, dtype=np.float32)
            if img_arr.ndim == 3:
                img_arr = img_arr[:, :, 0]
            max_val = 65535.0 if np.max(img_arr) > 255.0 else 255.0
            img_arr = img_arr / max_val
            cube.append(img_arr)
            
        return np.stack(cube, axis=2) # (512, 512, 31)

    def __len__(self) -> int:
        return len(self.scene_paths) * self.num_patches

    def __getitem__(self, idx: int):
        scene_idx = idx // self.num_patches
        meas = self.meas_cache[scene_idx]
        shifted = self.shifted_cache[scene_idx]
        
        max_h = meas.shape[0] - self.patch_size
        max_w = meas.shape[1] - self.patch_size
        
        # Dynamic stochastic sampling (NO deterministic re-seeding)
        start_h = np.random.randint(0, max_h + 1)
        start_w = np.random.randint(0, max_w + 1)
        
        patch_meas = meas[start_h : start_h + self.patch_size, start_w : start_w + self.patch_size].copy()
        patch_cube = shifted[start_h : start_h + self.patch_size, start_w : start_w + self.patch_size, :].copy()
        
        # Data augmentation for training
        if self.augment:
            if np.random.rand() > 0.5:
                patch_meas = np.fliplr(patch_meas).copy()
                patch_cube = np.fliplr(patch_cube).copy()
            if np.random.rand() > 0.5:
                patch_meas = np.flipud(patch_meas).copy()
                patch_cube = np.flipud(patch_cube).copy()
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                patch_meas = np.rot90(patch_meas, k=rot_k).copy()
                patch_cube = np.rot90(patch_cube, k=rot_k).copy()
                
        tensor_meas = torch.from_numpy(patch_meas).unsqueeze(0)             # (1, 48, 48)
        tensor_cube = torch.from_numpy(patch_cube).permute(2, 0, 1)        # (31, 48, 48)
        return tensor_meas, tensor_cube

    def get_full_scene(self, idx: int) -> tuple:
        """
        Returns full uncropped measurement, shifted target, and clean aligned ground truth:
        meas: (1, 1, 512, 542)
        shifted: (1, 31, 512, 542)
        clean_raw: (1, 31, 512, 512)
        scene_name: str
        """
        meas = self.meas_cache[idx]
        shifted = self.shifted_cache[idx]
        clean_raw = self.cubes_clean_raw[idx]
        scene_name = self.scene_names[idx]
        
        tensor_meas = torch.from_numpy(meas).unsqueeze(0).unsqueeze(0)
        tensor_shifted = torch.from_numpy(shifted).permute(2, 0, 1).unsqueeze(0)
        tensor_clean = torch.from_numpy(clean_raw).permute(2, 0, 1).unsqueeze(0)
        return tensor_meas, tensor_shifted, tensor_clean, scene_name
