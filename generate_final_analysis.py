"""
Generate Comprehensive Analysis Figures and Metrics Summary for YamawakiNet Replication.
Outputs:
- results/training_curves.png
- results/reconstruction_visual_samples.png
- results/final_evaluation_report.md
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.yamawaki_net import YamawakiNet
from utils.dataset_cassi import CASSIDataset
from utils.cassi_physics import CASSIForwardModel
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam

def hsi_to_rgb(hsi_cube):
    """
    Convert (31, H, W) HSI to pseudo-RGB image using bands [26, 14, 5] (approx 660nm, 540nm, 450nm).
    """
    if isinstance(hsi_cube, torch.Tensor):
        hsi_cube = hsi_cube.detach().cpu().numpy()
    
    r = hsi_cube[26]
    g = hsi_cube[14]
    b = hsi_cube[5]
    rgb = np.stack([r, g, b], axis=-1)
    # Normalize per image to [0, 1] for visualization
    rgb = np.clip(rgb, 0.0, 1.0)
    p2, p98 = np.percentile(rgb, (1, 99))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    return rgb

def plot_training_curves():
    csv_path = os.path.join(BASE_DIR, "results", "training_log.csv")
    if not os.path.exists(csv_path):
        print(f"[!] Warning: {csv_path} not found.")
        return
    
    epochs, train_loss, val_psnr, val_ssim, val_sam = [], [], [], [], []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['Epoch']))
            train_loss.append(float(row['Train_Loss']))
            val_psnr.append(float(row['Val_PSNR_dB']))
            val_ssim.append(float(row['Val_SSIM']))
            val_sam.append(float(row['Val_SAM_rad']))
            
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("YamawakiNet (DFHM) 400-Epoch Training Evolution on CAVE Dataset", fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    axes[0, 0].plot(epochs, train_loss, color='#d9534f', lw=2, label='L1 Loss (Train)')
    axes[0, 0].set_title("Training L1 Loss", fontsize=13, fontweight='semibold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].legend()
    
    # 2. Validation PSNR
    axes[0, 1].plot(epochs, val_psnr, color='#0275d8', lw=2, label='Val PSNR (dB)')
    axes[0, 1].axhline(y=27.44, color='red', linestyle='--', label='Paper Reported Mean (27.44 dB)')
    best_psnr = max(val_psnr)
    best_ep = epochs[val_psnr.index(best_psnr)]
    axes[0, 1].scatter([best_ep], [best_psnr], color='gold', s=100, zorder=5, edgecolors='black', label=f'Peak: {best_psnr:.2f} dB (Ep {best_ep})')
    axes[0, 1].set_title("Validation PSNR (dB)", fontsize=13, fontweight='semibold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend()
    
    # 3. Validation SSIM
    axes[1, 0].plot(epochs, val_ssim, color='#5cb85c', lw=2, label='Val SSIM')
    axes[1, 0].axhline(y=0.830, color='red', linestyle='--', label='Paper Reported Mean (0.830)')
    axes[1, 0].set_title("Validation SSIM", fontsize=13, fontweight='semibold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].legend()
    
    # 4. Validation SAM (radians)
    axes[1, 1].plot(epochs, val_sam, color='#6f42c1', lw=2, label='Val SAM (rad)')
    axes[1, 1].axhline(y=0.302, color='red', linestyle='--', label='Paper Reported Mean (0.302 rad)')
    axes[1, 1].set_title("Validation SAM (rad)", fontsize=13, fontweight='semibold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("SAM (radians)")
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].legend()
    
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "results", "training_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Training curves saved to: {out_path}")

def generate_visual_samples():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(BASE_DIR, "results", "weights", "best_model.pth")
    
    dataset_test = CASSIDataset(patch_size=48, num_patches_per_img=1, is_train=False, augment=False)
    model = YamawakiNet(in_channels=1, out_channels=31, num_features=512, num_blocks=9, S=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Select 4 representative scenes
    scenes_to_plot = [
        "fake_and_real_tomatoes_ms",
        "fake_and_real_sushi_ms",
        "real_and_fake_apples_ms",
        "flowers_ms"
    ]
    
    scene_indices = []
    for sc in scenes_to_plot:
        if sc in dataset_test.scene_names:
            scene_indices.append(dataset_test.scene_names.index(sc))
    if not scene_indices:
        scene_indices = [0, 3, 4, 14]
        
    fig, axes = plt.subplots(len(scene_indices), 4, figsize=(18, 4.5 * len(scene_indices)))
    
    with torch.no_grad():
        for row_idx, sc_idx in enumerate(scene_indices):
            meas, shifted, clean, name = dataset_test.get_full_scene(sc_idx)
            meas_gpu = meas.to(device)
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                pred_shifted = model(meas_gpu)
            pred_aligned = CASSIForwardModel.unshift(pred_shifted.cpu(), original_w=512)
            
            psnr = calculate_psnr(pred_aligned, clean)
            ssim = calculate_ssim(pred_aligned, clean)
            sam = calculate_sam(pred_aligned, clean, in_degrees=False)
            
            clean_cube = clean.squeeze(0)
            pred_cube = pred_aligned.squeeze(0)
            clean_rgb = hsi_to_rgb(clean_cube)
            pred_rgb = hsi_to_rgb(pred_cube)
            
            # 2D Measurement visual
            meas_img = meas.squeeze().cpu().numpy()
            
            # Error map (Mean Absolute Error across bands)
            error_map = np.mean(np.abs(pred_cube.numpy() - clean_cube.numpy()), axis=0)
            
            # Plot
            # 1. 2D Measurement
            axes[row_idx, 0].imshow(meas_img, cmap='gray')
            axes[row_idx, 0].set_title(f"CASSI 2D Measurement\n{name}", fontsize=11)
            axes[row_idx, 0].axis('off')
            
            # 2. Ground Truth RGB
            axes[row_idx, 1].imshow(clean_rgb)
            axes[row_idx, 1].set_title("Ground Truth (Pseudo-RGB)", fontsize=11)
            axes[row_idx, 1].axis('off')
            
            # 3. Reconstructed RGB
            axes[row_idx, 2].imshow(pred_rgb)
            axes[row_idx, 2].set_title(f"YamawakiNet Reconstruction\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}", fontsize=11)
            axes[row_idx, 2].axis('off')
            
            # 4. Error Map
            im = axes[row_idx, 3].imshow(error_map, cmap='inferno', vmin=0, vmax=0.15)
            axes[row_idx, 3].set_title(f"MAE Error Map (SAM: {sam:.4f} rad)", fontsize=11)
            axes[row_idx, 3].axis('off')
            fig.colorbar(im, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)
            
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "results", "reconstruction_visual_samples.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Visual samples gallery saved to: {out_path}")

if __name__ == '__main__':
    plot_training_curves()
    generate_visual_samples()
