import time
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir, resume=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file paths.
        self.train_log_csv = self.log_dir / 'train_log.csv'
        self.val_log_csv = self.log_dir / 'val_log.csv'
        self.train_log_txt = self.log_dir / 'train_log.txt'
        self.val_log_txt = self.log_dir / 'val_log.txt'
        
        # Initialize log files. In resume mode, keep existing logs and only
        # create missing files.
        if resume:
            self._init_missing_logs()
        else:
            self._init_csv_logs()
            self._init_txt_logs()

    def _init_missing_logs(self):
        missing_csv = [p for p in [self.train_log_csv, self.val_log_csv] if not p.exists()]
        missing_txt = [p for p in [self.train_log_txt, self.val_log_txt] if not p.exists()]
        if missing_csv:
            self._init_csv_logs(missing_csv)
        if missing_txt:
            self._init_txt_logs(missing_txt)
    
    def _init_csv_logs(self, log_files=None):
        """Initialize CSV log files."""
        header = [
            'Epoch',
            'Total_Loss',
            'Recon_Loss',
            'SAR_Loss',
            'PSNR(dB)',
            'SSIM',
            'RMSE(m)',
            'MAE(m)',
            'LR',
            'Time(s)'
        ]
        
        if log_files is None:
            log_files = [self.train_log_csv, self.val_log_csv]

        for log_file in log_files:
            with open(log_file, 'w') as f:
                f.write(','.join(header) + '\n')
    
    def _init_txt_logs(self, log_files=None):
        """Initialize TXT log files."""
        header_text = (
            "Training started at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n" +
            "-" * 120 + "\n" +
            "Epoch  Total_Loss   Recon_Loss   SAR_Loss    PSNR(dB)  SSIM    RMSE(m)   MAE(m)    LR        Time(s)\n" +
            "-" * 120 + "\n"
        )
        
        if log_files is None:
            log_files = [self.train_log_txt, self.val_log_txt]

        for log_file in log_files:
            with open(log_file, 'w') as f:
                f.write(header_text)
    
    def log_epoch(self, phase, epoch, loss_dict, metrics, lr, time_cost):
        """
        Log training results for one epoch.

        Args:
            phase: Training phase, either 'train' or 'val'.
            epoch: Epoch index.
            loss_dict: Loss values, such as {'total': float, 'recon': float, 'sar': float}.
            metrics: Metric values, such as {'psnr': float, 'ssim': float, 'rmse': float, 'mae': float}.
            lr: Learning rate.
            time_cost: Elapsed time in seconds.
        """
        # Select the log files for the current phase.
        if phase == 'train':
            csv_file = self.train_log_csv
            txt_file = self.train_log_txt
        else:
            csv_file = self.val_log_csv
            txt_file = self.val_log_txt
        
        # Write the CSV log row.
        self._write_csv_line(csv_file, epoch, loss_dict, metrics, lr, time_cost)
        
        # Write the TXT log row.
        self._write_txt_line(txt_file, epoch, loss_dict, metrics, lr, time_cost)
    
    def _write_csv_line(self, file_path, epoch, loss_dict, metrics, lr, time_cost):
        """Write one CSV log row."""
        values = [
            f"{epoch:03d}",
            f"{loss_dict['total']:.6f}",
            f"{loss_dict['recon']:.6f}",
            f"{loss_dict['sar']:.6f}",
            f"{metrics['psnr']:.2f}",
            f"{metrics['ssim']:.4f}",
            f"{metrics['rmse']:.2f}",
            f"{metrics['mae']:.2f}",
            f"{lr:.2e}",
            f"{time_cost:.1f}"
        ]
        with open(file_path, 'a') as f:
            f.write(','.join(values) + '\n')
    
    def _write_txt_line(self, file_path, epoch, loss_dict, metrics, lr, time_cost):
        """Write one TXT log row."""
        line = (f"{epoch:3d}   "
               f"{loss_dict['total']:9.6f}  "
               f"{loss_dict['recon']:9.6f}  "
               f"{loss_dict['sar']:9.6f}  "
               f"{metrics['psnr']:7.2f}  "
               f"{metrics['ssim']:6.4f}  "
               f"{metrics['rmse']:8.2f}  "
               f"{metrics['mae']:8.2f}  "
               f"{lr:8.2e}  "
               f"{time_cost:6.1f}")
        with open(file_path, 'a') as f:
            f.write(line + '\n')
    
    def log_best_model(self, epoch, metrics, model_path):
        """Log best-model information."""
        best_log_file = self.log_dir / 'best_model.txt'
        with open(best_log_file, 'w') as f:
            f.write(f"Best model saved at epoch {epoch}\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Metrics:\n")
            f.write(f"  PSNR: {metrics['psnr']:.2f} dB\n")
            f.write(f"  SSIM: {metrics['ssim']:.4f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.2f} m\n")
            f.write(f"  MAE: {metrics['mae']:.2f} m\n")
            f.write(f"Saved at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def log_training_summary(self, total_epochs, best_epoch, best_metrics, total_time):
        """Log the training summary."""
        summary_file = self.log_dir / 'training_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total epochs: {total_epochs}\n")
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Total training time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)\n")
            f.write(f"\nBest validation metrics:\n")
            f.write(f"  PSNR: {best_metrics['psnr']:.2f} dB\n")
            f.write(f"  SSIM: {best_metrics['ssim']:.4f}\n")
            f.write(f"  RMSE: {best_metrics['rmse']:.2f} m\n")
            f.write(f"  MAE: {best_metrics['mae']:.2f} m\n")
            f.write(f"\nTraining completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


class ConsoleLogger:
    """Console logger."""
    
    @staticmethod
    def print_metrics(phase, epoch, num_epochs, loss_dict, metrics, lr, time_cost):
        """Print training metrics to the console."""
        print(f"{phase} Epoch: [{epoch:03d}/{num_epochs-1:03d}]")
        print(f"  Loss: total={loss_dict['total']:.6f}, recon={loss_dict['recon']:.6f}, sar={loss_dict['sar']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.2f}m, MAE: {metrics['mae']:.2f}m")
        print(f"  LR: {lr:.2e}, Time: {time_cost:.1f}s")
        print("-" * 80)
    
    @staticmethod
    def print_epoch_summary(epoch, train_loss, train_metrics, val_loss, val_metrics):
        """Print an epoch summary."""
        print(f'Train - Loss: {train_loss:.4f}, PSNR: {train_metrics["psnr"]:.2f}, RMSE: {train_metrics["rmse"]:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, PSNR: {val_metrics["psnr"]:.2f}, RMSE: {val_metrics["rmse"]:.4f}')
    
    @staticmethod
    def print_best_model(epoch, rmse):
        """Print best-model information."""
        print(f'New best model saved with RMSE: {rmse:.4f} at epoch {epoch}')
    
    @staticmethod
    def print_training_start(device, model_params=None):
        """Print training-start information."""
        print("=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        print(f'Using device: {device}')
        if model_params:
            print(f'Model parameters: {model_params:,}')
        print(f'Started at: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        print("=" * 60)
    
    @staticmethod
    def print_training_complete():
        """Print training-completion information."""
        print("=" * 60)
        print("TRAINING COMPLETED!")
        print(f'Completed at: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        print("=" * 60)
