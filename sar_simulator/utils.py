import random
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_sar_pair_by_target_batch(sim_raw, target_raw, eps=1e-6):
    """
    Linearly normalize simulated and target SAR with target batch statistics.

    Statistics are computed per channel over the current batch and spatial
    dimensions. Using the same target-derived mean/std for both tensors keeps
    absolute intensity scale visible to the loss.
    """

    mean = target_raw.mean(dim=(0, 2, 3), keepdim=True)
    std = target_raw.std(dim=(0, 2, 3), keepdim=True).clamp_min(eps)

    sim_norm = (sim_raw - mean) / std
    target_norm = (target_raw - mean) / std

    return sim_norm, target_norm, mean, std


def append_epoch_log(log_file, epoch, loss, l1_loss, ssim_loss, ssim):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch:03d} "
            f"{loss:.4f} "
            f"{l1_loss:.4f} "
            f"{ssim_loss:.4f} "
            f"{ssim:.4f}\n"
        )


def init_log_file(log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("epoch loss l1loss ssim_loss ssim\n")


def visualize_random_sample(
    model,
    dataset,
    device,
    output_dir,
    stats,
    epoch,
    split_name,
):
    model.eval()
    ensure_dir(output_dir)

    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]

    dem = sample["dem_true"].unsqueeze(0).to(device)
    if "sar_target_raw" in sample:
        target_raw = sample["sar_target_raw"].unsqueeze(0).to(device)
    else:
        target_raw = sample["sar_target_norm"].unsqueeze(0).to(device)
    if "sar_target_original_raw" in sample:
        target_original_raw = sample["sar_target_original_raw"].unsqueeze(0).to(device)
    else:
        target_original_raw = target_raw
    angle = sample["inc_angle_map"].unsqueeze(0).to(device)

    with torch.no_grad():
        sim_raw = model(dem, angle)
        geometry = model.compute_geometry(dem, angle) if hasattr(model, "compute_geometry") else {}

    dem_np = dem[0, 0].detach().cpu().numpy()
    lia_np = None
    if "lia_deg" in geometry:
        lia_np = geometry["lia_deg"][0, 0].detach().cpu().numpy()

    target_vv_original = target_original_raw[0, 0].detach().cpu().numpy()
    target_vh_original = target_original_raw[0, 1].detach().cpu().numpy()
    target_vv_filtered = target_raw[0, 0].detach().cpu().numpy()
    target_vh_filtered = target_raw[0, 1].detach().cpu().numpy()
    sim_vv = sim_raw[0, 0].detach().cpu().numpy()
    sim_vh = sim_raw[0, 1].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)

    im = axes[0, 0].imshow(target_vv_original, cmap="gray")
    axes[0, 0].set_title("Target VV Original")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(target_vv_filtered, cmap="gray")
    axes[0, 1].set_title("Target VV Filtered")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow(sim_vv, cmap="gray")
    axes[0, 2].set_title("Simulated VV")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im = axes[0, 3].imshow(dem_np, cmap="terrain")
    axes[0, 3].set_title("DEM")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(target_vh_original, cmap="gray")
    axes[1, 0].set_title("Target VH Original")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(target_vh_filtered, cmap="gray")
    axes[1, 1].set_title("Target VH Filtered")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im = axes[1, 2].imshow(sim_vh, cmap="gray")
    axes[1, 2].set_title("Simulated VH")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    if lia_np is None:
        lia_np = np.zeros_like(dem_np)
    im = axes[1, 3].imshow(lia_np, cmap="viridis")
    axes[1, 3].set_title("LIA")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{split_name} | Epoch {epoch:03d} | Sample {idx}", fontsize=14)
    plt.tight_layout()

    save_path = Path(output_dir) / f"{split_name}_epoch_{epoch:03d}_sample_{idx:05d}.png"
    plt.savefig(save_path)
    plt.close(fig)

    return save_path
