import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import H5SimulatorDataset
from models import PreciseAngleSimulator
from losses import L1SSIMLoss
from metrics import torch_ssim
from utils import (
    set_seed,
    ensure_dir,
    normalize_sar_pair_by_target_batch,
    init_log_file,
    append_epoch_log,
    visualize_random_sample,
)


# =============================================================================
# Config
# =============================================================================

TRAIN_H5_PATH = Path(r"E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Train_data\train_data_real_lr.h5")
VAL_H5_PATH = Path(r"E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Val_data\val_data_real_lr.h5")

OUTPUT_DIR = Path(r"E:\TP_DSM_SR_PIMSR\PIMSR\PIMSR-1\sar_simulator_project-R4\explicit_lia_batchnorm_filtered_output")

BEST_MODEL_PATH = OUTPUT_DIR / "best_precise_simulator.pth"
LAST_MODEL_PATH = OUTPUT_DIR / "last_precise_simulator.pth"

TRAIN_LOG_FILE = OUTPUT_DIR / "train_log.txt"
VAL_LOG_FILE = OUTPUT_DIR / "val_log.txt"

TRAIN_VIS_DIR = OUTPUT_DIR / "visualization_train"
VAL_VIS_DIR = OUTPUT_DIR / "visualization_val"

BATCH_SIZE = 128
NUM_EPOCHS = 100
NUM_WORKERS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SSIM_LOSS_WEIGHT = 1.0

SSIM_WIN_SIZE = 7
SEED = 42


# =============================================================================
# Train and validation functions
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim_loss = 0.0
    total_ssim = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        dem = batch["dem_true"].to(device, non_blocking=True)
        target_raw = batch["sar_target_raw"].to(device, non_blocking=True)
        angle = batch["inc_angle_map"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        sim_raw = model(dem, angle)
        sim_norm, target_norm, _, _ = normalize_sar_pair_by_target_batch(
            sim_raw,
            target_raw,
        )

        loss, l1_loss, ssim_loss = criterion(sim_norm, target_norm)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            ssim_val = torch_ssim(
                sim_norm.detach(),
                target_norm,
                win_size=SSIM_WIN_SIZE,
            )

        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_ssim_loss += ssim_loss.item()
        total_ssim += ssim_val.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "l1": total_l1 / n_batches,
        "ssim_loss": total_ssim_loss / n_batches,
        "ssim": total_ssim / n_batches,
    }


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim_loss = 0.0
    total_ssim = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        dem = batch["dem_true"].to(device, non_blocking=True)
        target_raw = batch["sar_target_raw"].to(device, non_blocking=True)
        angle = batch["inc_angle_map"].to(device, non_blocking=True)

        sim_raw = model(dem, angle)
        sim_norm, target_norm, _, _ = normalize_sar_pair_by_target_batch(
            sim_raw,
            target_raw,
        )

        loss, l1_loss, ssim_loss = criterion(sim_norm, target_norm)

        ssim_val = torch_ssim(
            sim_norm,
            target_norm,
            win_size=SSIM_WIN_SIZE,
        )

        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_ssim_loss += ssim_loss.item()
        total_ssim += ssim_val.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "l1": total_l1 / n_batches,
        "ssim_loss": total_ssim_loss / n_batches,
        "ssim": total_ssim / n_batches,
    }


def main():
    set_seed(SEED)

    ensure_dir(OUTPUT_DIR)
    ensure_dir(TRAIN_VIS_DIR)
    ensure_dir(VAL_VIS_DIR)

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"Starting enhanced SAR simulator training at {time.ctime()}")
    print("=" * 80)
    print(f"Device: {device}")

    train_dataset = H5SimulatorDataset(TRAIN_H5_PATH)
    val_dataset = H5SimulatorDataset(VAL_H5_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")

    model = PreciseAngleSimulator().to(device)
    criterion = L1SSIMLoss(
        ssim_weight=SSIM_LOSS_WEIGHT,
        win_size=SSIM_WIN_SIZE,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        verbose=True,
    )

    init_log_file(TRAIN_LOG_FILE)
    init_log_file(VAL_LOG_FILE)

    best_val_ssim = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_metrics["ssim"])

        append_epoch_log(
            TRAIN_LOG_FILE,
            epoch,
            train_metrics["loss"],
            train_metrics["l1"],
            train_metrics["ssim_loss"],
            train_metrics["ssim"],
        )

        append_epoch_log(
            VAL_LOG_FILE,
            epoch,
            val_metrics["loss"],
            val_metrics["l1"],
            val_metrics["ssim_loss"],
            val_metrics["ssim"],
        )

        torch.save(model.state_dict(), LAST_MODEL_PATH)

        if val_metrics["ssim"] > best_val_ssim:
            best_val_ssim = val_metrics["ssim"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_flag = " | best saved"
        else:
            best_flag = ""

        train_vis_path = visualize_random_sample(
            model=model,
            dataset=train_dataset,
            device=device,
            output_dir=TRAIN_VIS_DIR,
            stats=None,
            epoch=epoch,
            split_name="train",
        )

        val_vis_path = visualize_random_sample(
            model=model,
            dataset=val_dataset,
            device=device,
            output_dir=VAL_VIS_DIR,
            stats=None,
            epoch=epoch,
            split_name="val",
        )

        duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Train L1 {train_metrics['l1']:.4f} | "
            f"Train SSIM Loss {train_metrics['ssim_loss']:.4f} | "
            f"Train SSIM {train_metrics['ssim']:.4f} || "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val L1 {val_metrics['l1']:.4f} | "
            f"Val SSIM Loss {val_metrics['ssim_loss']:.4f} | "
            f"Val SSIM {val_metrics['ssim']:.4f} | "
            f"LR {current_lr:.2e} | "
            f"{duration:.2f}s"
            f"{best_flag}"
        )

        print(f"  Train visualization: {train_vis_path}")
        print(f"  Val visualization  : {val_vis_path}")

    print("=" * 80)
    print(f"Training finished. Best validation SSIM: {best_val_ssim:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
