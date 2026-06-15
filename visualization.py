import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_results(lr_dem_norm, sr_dem_norm, hr_dem_raw, vv_sar_norm, vh_sar_norm, 
                     epoch, save_dir, global_stats, phase='train'):
    """Visualize training results using raw elevation values.

    Args:
        lr_dem_norm: Normalized low-resolution DEM [B, 1, H, W].
        sr_dem_norm: Normalized super-resolved DEM [B, 1, H, W].
        hr_dem_raw: High-resolution DEM with raw elevation values [B, 1, H, W].
        vv_sar_norm: Normalized VV SAR [B, 1, H, W].
        vh_sar_norm: Normalized VH SAR [B, 1, H, W].
        epoch: Current epoch.
        save_dir: Output directory.
        global_stats: Global statistics.
        phase: Training phase, either 'train' or 'val'.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Get DEM mean and standard deviation for denormalization.
    dem_mean = global_stats['dem']['mean']
    dem_std = global_stats['dem']['std']
    
    # Select the first sample for visualization.
    lr_norm = lr_dem_norm[0, 0].detach().cpu().numpy()
    sr_norm = sr_dem_norm[0, 0].detach().cpu().numpy()
    hr = hr_dem_raw[0, 0].detach().cpu().numpy()
    vv = vv_sar_norm[0, 0].detach().cpu().numpy()
    vh = vh_sar_norm[0, 0].detach().cpu().numpy()
    
    # Denormalize LR and SR data.
    lr = lr_norm * dem_std + dem_mean
    sr = sr_norm * dem_std + dem_mean
    
    # Create subplots.
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # LR DEM after denormalization.
    im1 = axes[0].imshow(lr, cmap='terrain')
    axes[0].set_title(f'LR DEM\nRange: [{lr.min():.1f}, {lr.max():.1f}]m')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, label='Elevation (m)')
    
    # SR DEM after denormalization.
    im2 = axes[1].imshow(sr, cmap='terrain')
    axes[1].set_title(f'SR DEM\nRange: [{sr.min():.1f}, {sr.max():.1f}]m')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, label='Elevation (m)')
    
    # HR DEM
    im3 = axes[2].imshow(hr, cmap='terrain')
    axes[2].set_title(f'HR DEM (Ground Truth)\nRange: [{hr.min():.1f}, {hr.max():.1f}]m')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, label='Elevation (m)')
    
    # VV SAR
    im4 = axes[3].imshow(vv, cmap='gray')
    axes[3].set_title('VV SAR (Normalized)')
    axes[3].axis('off')
    plt.colorbar(im4, ax=axes[3], fraction=0.046)
    
    # VH SAR
    im5 = axes[4].imshow(vh, cmap='gray')
    axes[4].set_title('VH SAR (Normalized)')
    axes[4].axis('off')
    plt.colorbar(im5, ax=axes[4], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{phase}_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_comparison(lr_dem, sr_dem, hr_dem, save_path=None, title_prefix=""):
    """Visualize DEM comparison results.

    Args:
        lr_dem: Low-resolution DEM [H, W].
        sr_dem: Super-resolved DEM [H, W].
        hr_dem: High-resolution DEM [H, W].
        save_path: Output path.
        title_prefix: Title prefix.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # LR DEM
    im1 = axes[0].imshow(lr_dem, cmap='terrain')
    axes[0].set_title(f'{title_prefix}LR DEM\nRange: [{lr_dem.min():.1f}, {lr_dem.max():.1f}]m')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, label='Elevation (m)')
    
    # SR DEM
    im2 = axes[1].imshow(sr_dem, cmap='terrain')
    axes[1].set_title(f'{title_prefix}SR DEM\nRange: [{sr_dem.min():.1f}, {sr_dem.max():.1f}]m')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, label='Elevation (m)')
    
    # HR DEM
    im3 = axes[2].imshow(hr_dem, cmap='terrain')
    axes[2].set_title(f'{title_prefix}HR DEM (GT)\nRange: [{hr_dem.min():.1f}, {hr_dem.max():.1f}]m')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, label='Elevation (m)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_error_map(sr_dem, hr_dem, save_path=None, title="Error Map"):
    """Visualize an error map.

    Args:
        sr_dem: Super-resolved DEM [H, W].
        hr_dem: High-resolution DEM [H, W].
        save_path: Output path.
        title: Figure title.
    """
    error_map = np.abs(sr_dem - hr_dem)
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(error_map, cmap='hot')
    plt.title(f'{title}\nMAE: {np.mean(error_map):.2f}m, Max Error: {np.max(error_map):.2f}m')
    plt.colorbar(im, label='Absolute Error (m)')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_dir):
    """Plot training curves.

    Args:
        train_losses: Training loss values.
        val_losses: Validation loss values.
        train_metrics: Training metrics.
        val_metrics: Validation metrics.
        save_dir: Output directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves.
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # PSNR curves.
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['psnr'] for m in train_metrics], 'b-', label='Train PSNR')
    plt.plot(epochs, [m['psnr'] for m in val_metrics], 'r-', label='Val PSNR')
    plt.title('PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    
    # SSIM curves.
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['ssim'] for m in train_metrics], 'b-', label='Train SSIM')
    plt.plot(epochs, [m['ssim'] for m in val_metrics], 'r-', label='Val SSIM')
    plt.title('SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)
    
    # RMSE curves.
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [m['rmse'] for m in train_metrics], 'b-', label='Train RMSE')
    plt.plot(epochs, [m['rmse'] for m in val_metrics], 'r-', label='Val RMSE')
    plt.title('RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (m)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
