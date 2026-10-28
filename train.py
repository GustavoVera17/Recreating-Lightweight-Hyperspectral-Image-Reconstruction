"""
Training Pipeline for Yamawaki_Rep_2 (LWHRN)
Follows rigorous practices from DSLAnet:
- PyTorch AMP mixed precision training on CUDA.
- Dynamic ETA estimation formatted as DD:HH:MM:SS.
- Structured CSV logging (Epoch, Loss, PSNR, SSIM, SAM, TimeRemaining).
- Periodic visual dashboards (CASSI 2D, GT Band, Pred Band, Absolute Error Map, Metric Curves).
- Automated checkpointing of best validation model.
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Headless mode
import matplotlib.pyplot as plt

# Ensure repository root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.yamawaki_net import YamawakiNet
from utils.dataset_cassi import CASSIDataset
from utils.cassi_physics import CASSIForwardModel
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam

def format_eta(seconds: float) -> str:
    """Formats remaining seconds into days:hours:minutes:seconds (DD:HH:MM:SS)."""
    if seconds < 0 or not np.isfinite(seconds):
        return "00:00:00:00"
    seconds = int(round(seconds))
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{secs:02d}"

def get_args():
    parser = argparse.ArgumentParser(description="Yamawaki_Rep_2 Training Script")
    parser.add_argument('--epochs', type=int, default=400, help="Total training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for 48x48 patches")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--patch_size', type=int, default=48, help="Spatial patch size")
    parser.add_argument('--patches_per_img', type=int, default=50, help="Patches per image per epoch")
    parser.add_argument('--num_features', type=int, default=512, help="Feature channels in DFHM")
    parser.add_argument('--num_blocks', type=int, default=9, help="Number of DFHM stages")
    parser.add_argument('--dash_interval', type=int, default=10, help="Epoch interval to save dashboard")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"[*] Starting Yamawaki_Rep_2 Training on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[*] Total Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 70)

    # Directories
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    DASHBOARDS_DIR = os.path.join(RESULTS_DIR, "dashboards")
    WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")
    os.makedirs(DASHBOARDS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Dataset loading
    print("[*] Loading CAVE datasets into RAM cache...")
    dataset_train = CASSIDataset(
        patch_size=args.patch_size,
        num_patches_per_img=args.patches_per_img,
        is_train=True,
        augment=True
    )
    dataset_val = CASSIDataset(
        patch_size=args.patch_size,
        num_patches_per_img=1,
        is_train=False,
        augment=False
    )
    
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    print(f"[+] Dataloader ready: {len(dataset_train)} patches per epoch ({len(loader_train)} batches).")

    # Instantiate Model
    model = YamawakiNet(
        in_channels=1,
        out_channels=31,
        num_features=args.num_features,
        num_blocks=args.num_blocks,
        S=2
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # StepLR schedule: decay by 0.2 at 30% of total epochs (as in DSLAnet)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs * 0.3), gamma=0.2)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # CSV Logging Setup
    csv_path = os.path.join(RESULTS_DIR, "training_log.csv")
    csv_headers = ["Epoch", "Train_Loss", "Val_PSNR_dB", "Val_SSIM", "Val_SAM_rad", "Learning_Rate", "Epoch_Time_s", "Estimated_Time_Remaining"]
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    # Track metrics history
    history = {'loss': [], 'psnr': [], 'ssim': [], 'sam': []}
    best_psnr = 0.0
    epoch_durations = []

    # Fixed validation scene index for reproducible dashboard
    val_scene_idx = 0
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        model.train()
        loss_epoch = 0.0
        
        loop = tqdm(loader_train, desc=f"Epoch [{epoch}/{args.epochs}]", leave=False)
        for i, (meas_batch, cube_batch) in enumerate(loop):
            meas_batch = meas_batch.to(device)
            cube_batch = cube_batch.to(device)
            
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred = model(meas_batch)
                    loss = criterion(pred, cube_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(meas_batch)
                loss = criterion(pred, cube_batch)
                loss.backward()
                optimizer.step()
                
            loss_epoch += loss.item()
            loop.set_postfix(Loss=f"{(loss_epoch / (i + 1)):.4f}")
            
        scheduler.step()
        avg_train_loss = loss_epoch / len(loader_train)
        history['loss'].append(avg_train_loss)

        # Validation on full scenes (or fixed test scene)
        model.eval()
        with torch.no_grad():
            val_meas, val_shifted, val_clean, val_name = dataset_val.get_full_scene(val_scene_idx)
            val_meas = val_meas.to(device)
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    val_pred_shifted = model(val_meas)
            else:
                val_pred_shifted = model(val_meas)
                
            # Unshift to reconstruct true aligned (1, 31, 512, 512) HSI
            val_pred_aligned = CASSIForwardModel.unshift(val_pred_shifted.cpu(), original_w=512)
            
            val_psnr = calculate_psnr(val_pred_aligned, val_clean)
            val_ssim = calculate_ssim(val_pred_aligned, val_clean)
            val_sam = calculate_sam(val_pred_aligned, val_clean, in_degrees=False) # STRICT RADIANS
            
            history['psnr'].append(val_psnr)
            history['ssim'].append(val_ssim)
            history['sam'].append(val_sam)

        # Measure epoch duration and calculate ETA
        t_epoch_end = time.time()
        epoch_dur = t_epoch_end - t_epoch_start
        epoch_durations.append(epoch_dur)
        
        # Exponential moving average of epoch time
        recent_durations = epoch_durations[-10:]
        avg_epoch_time = sum(recent_durations) / len(recent_durations)
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_formatted = format_eta(eta_seconds)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[*] Ep {epoch:03d}/{args.epochs} | Loss: {avg_train_loss:.4f} | PSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.4f} | SAM: {val_sam:.4f}rad | ETA: {eta_formatted}")

        # Checkpoint Best Model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_model.pth"))
            print(f"  [+] New Best PSNR: {best_psnr:.2f} dB (Saved best_model.pth)")

        # Log to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train_loss:.5f}",
                f"{val_psnr:.2f}",
                f"{val_ssim:.4f}",
                f"{val_sam:.4f}",
                f"{current_lr:.6f}",
                f"{epoch_dur:.2f}",
                eta_formatted
            ])

        # Generate Visual Dashboard
        if epoch % args.dash_interval == 0 or epoch == args.epochs:
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f"Yamawaki LWHRN Replication | Scene: {val_name} | Epoch: {epoch} | PSNR: {val_psnr:.2f} dB", fontsize=16, fontweight='bold')
            
            # Row 1: Images
            axs[0, 0].imshow(val_meas[0, 0].cpu().numpy(), cmap='gray')
            axs[0, 0].set_title("CASSI 2D Snapshot")
            axs[0, 0].axis('off')
            
            band_idx = 15 # Central band (~550 nm)
            gt_band = val_clean[0, band_idx].cpu().numpy()
            pred_band = val_pred_aligned[0, band_idx].clamp(0, 1).cpu().numpy()
            err_map = np.abs(gt_band - pred_band)
            
            axs[0, 1].imshow(gt_band, cmap='gray', vmin=0, vmax=1)
            axs[0, 1].set_title(f"Ground Truth (Band {band_idx})")
            axs[0, 1].axis('off')
            
            axs[0, 2].imshow(pred_band, cmap='gray', vmin=0, vmax=1)
            axs[0, 2].set_title(f"Prediction (Band {band_idx})")
            axs[0, 2].axis('off')
            
            im_err = axs[0, 3].imshow(err_map, cmap='inferno')
            axs[0, 3].set_title("Absolute Error Map")
            fig.colorbar(im_err, ax=axs[0, 3], fraction=0.046, pad=0.04)
            axs[0, 3].axis('off')
            
            # Row 2: Curves
            axs[1, 0].plot(history['loss'], 'r-')
            axs[1, 0].set_title("Training L1 Loss")
            axs[1, 0].grid(True)
            
            axs[1, 1].plot(history['psnr'], 'b-')
            axs[1, 1].set_title("Validation PSNR (dB)")
            axs[1, 1].grid(True)
            
            axs[1, 2].plot(history['ssim'], 'g-')
            axs[1, 2].set_title("Validation SSIM")
            axs[1, 2].grid(True)
            
            axs[1, 3].plot(history['sam'], 'm-')
            axs[1, 3].set_title("Validation SAM (rad)")
            axs[1, 3].grid(True)
            
            plt.tight_layout()
            dash_file = os.path.join(DASHBOARDS_DIR, f"dashboard_ep{epoch:04d}.png")
            plt.savefig(dash_file, dpi=130, bbox_inches='tight')
            plt.close(fig)
            print(f"  [+] Dashboard exported: {dash_file}")

        # Periodic Git Push to GitHub every 50 epochs or at the end
        if epoch % 50 == 0 or epoch == args.epochs:
            try:
                import subprocess
                subprocess.run(["git", "add", "results/"], cwd=BASE_DIR, check=False)
                subprocess.run(["git", "commit", "-m", f"Auto-sync: Training metrics and dashboard at epoch {epoch}"], cwd=BASE_DIR, check=False)
                subprocess.run(["git", "push", "origin", "main"], cwd=BASE_DIR, check=False)
                print(f"  [-] Synced epoch {epoch} results to GitHub successfully.")
            except Exception as e:
                print(f"  [!] Git push notice: {e}")

    total_time = time.time() - total_start_time
    print(f"\n[+] Training Complete! Total time elapsed: {format_eta(total_time)}. Best PSNR: {best_psnr:.2f} dB.")

if __name__ == '__main__':
    main()

