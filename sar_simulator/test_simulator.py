from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import H5SimulatorDataset
from models import PreciseAngleSimulator
from metrics import channel_metrics
from utils import normalize_sar_pair_by_target_batch, ensure_dir


# =============================================================================
# Config
# =============================================================================

TEST_H5_PATH = Path("Test_data/test_data.h5")

OUTPUT_DIR = Path("sar_simulator_project-R4/output_test")
MODEL_PATH = OUTPUT_DIR / "best_precise_simulator.pth"

TEST_METRICS_FILE = OUTPUT_DIR / "test_metrics.txt"

BATCH_SIZE = 128
NUM_WORKERS = 4
SSIM_WIN_SIZE = 7


@torch.no_grad()
def test_model():
    ensure_dir(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Testing enhanced SAR simulator")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Test H5: {TEST_H5_PATH}")
    print(f"Model  : {MODEL_PATH}")

    test_dataset = H5SimulatorDataset(TEST_H5_PATH)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    model = PreciseAngleSimulator().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    metric_sum = {
        "VV": {
            "psnr": 0.0,
            "ssim": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
        },
        "VH": {
            "psnr": 0.0,
            "ssim": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
        },
    }

    n_batches = 0

    for batch in tqdm(test_loader, desc="Testing"):
        dem = batch["dem_true"].to(device, non_blocking=True)
        target_raw = batch["sar_target_raw"].to(device, non_blocking=True)
        angle = batch["inc_angle_map"].to(device, non_blocking=True)

        sim_raw = model(dem, angle)
        sim_norm, target_norm, _, _ = normalize_sar_pair_by_target_batch(
            sim_raw,
            target_raw,
        )

        batch_metrics = channel_metrics(
            pred=sim_norm,
            target=target_norm,
            win_size=SSIM_WIN_SIZE,
        )

        for pol in ["VV", "VH"]:
            for key in ["psnr", "ssim", "rmse", "mae"]:
                metric_sum[pol][key] += batch_metrics[pol][key]

        n_batches += 1

    final_metrics = {
        pol: {
            key: metric_sum[pol][key] / n_batches
            for key in metric_sum[pol]
        }
        for pol in metric_sum
    }

    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)

    for pol in ["VV", "VH"]:
        print(f"{pol}:")
        print(f"  PSNR: {final_metrics[pol]['psnr']:.4f} dB")
        print(f"  SSIM: {final_metrics[pol]['ssim']:.4f}")
        print(f"  RMSE: {final_metrics[pol]['rmse']:.4f}")
        print(f"  MAE : {final_metrics[pol]['mae']:.4f}")

    with open(TEST_METRICS_FILE, "w", encoding="utf-8") as f:
        f.write("SAR Simulator Test Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test H5: {TEST_H5_PATH}\n")
        f.write(f"SSIM window size: {SSIM_WIN_SIZE}\n\n")

        for pol in ["VV", "VH"]:
            f.write(f"{pol}:\n")
            f.write(f"  PSNR: {final_metrics[pol]['psnr']:.4f} dB\n")
            f.write(f"  SSIM: {final_metrics[pol]['ssim']:.4f}\n")
            f.write(f"  RMSE: {final_metrics[pol]['rmse']:.4f}\n")
            f.write(f"  MAE : {final_metrics[pol]['mae']:.4f}\n\n")

    print(f"\nMetrics saved to: {TEST_METRICS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    test_model()
