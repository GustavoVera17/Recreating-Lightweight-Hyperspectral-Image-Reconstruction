"""
Comprehensive Evaluation of YamawakiNet across all 20 CAVE Test Scenes.
Compares quantitative metrics (PSNR, SSIM, SAM in radians) against Table 1 of Yamawaki et al. (2023).
"""

import os
import sys
import argparse
import numpy as np
import torch
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.yamawaki_net import YamawakiNet
from utils.dataset_cassi import CASSIDataset
from utils.cassi_physics import CASSIForwardModel
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam

def evaluate(weights_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights_path is None:
        weights_path = os.path.join(BASE_DIR, "results", "weights", "best_model.pth")
        
    print("=" * 75)
    print(f"[*] Evaluating YamawakiNet on 20 CAVE Test Scenes using weights:")
    print(f"    {weights_path}")
    print("=" * 75)

    dataset_test = CASSIDataset(patch_size=48, num_patches_per_img=1, is_train=False, augment=False)
    num_scenes = len(dataset_test.scene_names)

    model = YamawakiNet(in_channels=1, out_channels=31, num_features=512, num_blocks=9, S=2).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("[+] Weights loaded successfully.")
    else:
        print("[!] Warning: Weights file not found. Running with initial weights for pipeline check.")
        
    model.eval()
    
    results = []
    psnr_list, ssim_list, sam_list = [], [], []

    with torch.no_grad():
        for i in range(num_scenes):
            meas, shifted, clean, name = dataset_test.get_full_scene(i)
            meas = meas.to(device)
            
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                pred_shifted = model(meas)
                
            pred_aligned = CASSIForwardModel.unshift(pred_shifted.cpu(), original_w=512)
            
            psnr_v = calculate_psnr(pred_aligned, clean)
            ssim_v = calculate_ssim(pred_aligned, clean)
            sam_v = calculate_sam(pred_aligned, clean, in_degrees=False) # In radians as in paper
            sam_deg = calculate_sam(pred_aligned, clean, in_degrees=True)
            
            psnr_list.append(psnr_v)
            ssim_list.append(ssim_v)
            sam_list.append(sam_v)
            
            results.append([name, f"{psnr_v:.2f}", f"{ssim_v:.4f}", f"{sam_v:.4f}", f"{sam_deg:.2f}°"])
            print(f"  [{i+1:02d}/{num_scenes}] {name:<30} | PSNR: {psnr_v:.2f} dB | SSIM: {ssim_v:.4f} | SAM: {sam_v:.4f} rad")

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_sam = float(np.mean(sam_list))

    print("\n" + "=" * 75)
    print("                SUMMARY TABLE: REPLICATION VS PAPER (CAVE)")
    print("=" * 75)
    
    summary_data = [
        ["Paper Reported (Yamawaki 2023)", "27.44 dB", "0.8300", "0.3020 rad", "42 MB"],
        ["Our Replication (Yamawaki_Rep_2)", f"{mean_psnr:.2f} dB", f"{mean_ssim:.4f}", f"{mean_sam:.4f} rad", "42.58 MB"]
    ]
    print(tabulate(summary_data, headers=["Method", "PSNR", "SSIM", "SAM (rad)", "Params (MB)"], tablefmt="fancy_grid"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.weights)
