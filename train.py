import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import create_output_dirs, get_config_from_args, print_config
from networks import ResidualSARDEMGenerator, SARLoss
from trainer import SARDEMTrainer
from logger import TrainingLogger, ConsoleLogger
from visualization import visualize_results, plot_training_curves
from dataset import create_dataloader_from_h5


def set_random_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def count_parameters(model):
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, epoch, best_rmse, val_metrics, save_path):
    """Save a training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_rmse': best_rmse,
        'val_metrics': val_metrics
    }, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load a training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_rmse = checkpoint['best_rmse']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return start_epoch, best_rmse


def main():
    """Run the training workflow."""
    config = get_config_from_args()

    set_random_seed(config.random_seed)

    print_config(config)

    output_dir = create_output_dirs(config.output_dir)

    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = 'cpu'
    device = torch.device(config.device)

    with open(config.global_stats, 'r') as f:
        global_stats = json.load(f)

    print("Creating data loaders from HDF5 files...")
    train_loader = create_dataloader_from_h5(
        h5_file_path=config.train_h5_path, 
        global_stats_path=config.global_stats,
        csv_file_path=config.train_csv_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True
    )
    val_loader = create_dataloader_from_h5(
        h5_file_path=config.val_h5_path,
        global_stats_path=config.global_stats,
        csv_file_path=config.val_csv_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False
    )

    print("Creating model...")
    model = ResidualSARDEMGenerator(
        sar_input_mode=config.sar_input_mode,
        fusion_scales=config.fusion_scales,
    ).to(device)
    model_params = count_parameters(model)

    criterion = nn.L1Loss()
    sar_loss_fn = SARLoss(None if config.sar_weight == 0 else config.simulator_weights).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    trainer = SARDEMTrainer(
        model=model,
        criterion=criterion,
        sar_loss_fn=sar_loss_fn,
        optimizer=optimizer,
        device=device,
        global_stats=global_stats,
        recon_weight=config.recon_weight,
        sar_weight=config.sar_weight,
    )

    training_logger = TrainingLogger(output_dir / 'logs', resume=bool(config.resume))
    console_logger = ConsoleLogger()

    start_epoch = 0
    best_rmse = float('inf')
    best_metrics = None

    if config.resume:
        print(f"Resuming from checkpoint: {config.resume}")
        start_epoch, best_rmse = load_checkpoint(
            config.resume, model, optimizer, scheduler
        )
        print(f'Resumed from epoch {start_epoch}, best RMSE: {best_rmse:.4f}')

    console_logger.print_training_start(device, model_params)

    training_start_time = time.time()
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()

        train_loss, train_recon, train_sar, train_metrics = trainer.train_epoch(train_loader, epoch)
        train_time = time.time() - epoch_start_time

        val_start_time = time.time()
        val_loss, val_recon, val_sar, val_metrics = trainer.validate_epoch(val_loader, epoch)
        val_time = time.time() - val_start_time

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        train_loss_dict = {'total': train_loss, 'recon': train_recon, 'sar': train_sar}
        val_loss_dict = {'total': val_loss, 'recon': val_recon, 'sar': val_sar}
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_history.append(train_metrics.copy())
        val_metrics_history.append(val_metrics.copy())

        training_logger.log_epoch('train', epoch, train_loss_dict, train_metrics, current_lr, train_time)
        training_logger.log_epoch('val', epoch, val_loss_dict, val_metrics, current_lr, val_time)

        console_logger.print_metrics('Train', epoch, config.num_epochs, train_loss_dict, train_metrics, current_lr,
                                     train_time)
        console_logger.print_metrics('Val', epoch, config.num_epochs, val_loss_dict, val_metrics, current_lr, val_time)
        console_logger.print_epoch_summary(epoch, train_loss, train_metrics, val_loss, val_metrics)

        if val_metrics['rmse'] < best_rmse:
            best_rmse = val_metrics['rmse']
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            best_model_path = output_dir / 'checkpoints' / 'best_model.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, best_rmse, val_metrics, best_model_path)
            training_logger.log_best_model(epoch, val_metrics, best_model_path)
            console_logger.print_best_model(epoch, best_rmse)

        if epoch % config.save_freq == 0:
            checkpoint_path = output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch:03d}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, best_rmse, val_metrics, checkpoint_path)

        if epoch % config.vis_freq == 0:
            train_sample = trainer.get_sample_prediction(train_loader)
            visualize_results(
                train_sample['lr_dem_norm'], train_sample['sr_dem_norm'], train_sample['hr_dem_raw'],
                train_sample['vv_sar_norm'], train_sample['vh_sar_norm'],
                epoch, output_dir / 'visualizations', global_stats, 'train'
            )
            val_sample = trainer.get_sample_prediction(val_loader)
            visualize_results(
                val_sample['lr_dem_norm'], val_sample['sr_dem_norm'], val_sample['hr_dem_raw'],
                val_sample['vv_sar_norm'], val_sample['vh_sar_norm'],
                epoch, output_dir / 'visualizations', global_stats, 'val'
            )

    total_training_time = time.time() - training_start_time
    plot_training_curves(
        train_losses,
        val_losses,
        train_metrics_history,
        val_metrics_history,
        output_dir / 'visualizations'
    )
    console_logger.print_training_complete()

    # best_metrics = {'psnr': 0, 'ssim': 0, 'rmse': best_rmse, 'mae': 0}
    # training_logger.log_training_summary(config.num_epochs, best_epoch, best_metrics, total_training_time)

    if best_metrics is None:
        best_metrics = {'psnr': 0, 'ssim': 0, 'rmse': best_rmse, 'mae': 0}

    training_logger.log_training_summary(config.num_epochs, best_epoch, best_metrics, total_training_time)

    print(f"\nTraining Summary:")
    print(f"  SAR input mode: {config.sar_input_mode}")
    print(f"  Fusion scales: {config.fusion_scales}")
    print(f"  Total time: {total_training_time:.1f}s ({total_training_time / 3600:.2f}h)")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best RMSE: {best_rmse:.4f}m")
    print(f"  Model saved to: {output_dir / 'checkpoints' / 'best_model.pth'}")


if __name__ == '__main__':
    main()
