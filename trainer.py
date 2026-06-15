import torch
from tqdm import tqdm
import sys
from metrics import calculate_batch_metrics

class SARDEMTrainer:
    """SAR-enhanced DEM super-resolution trainer.
    """

    def __init__(
        self,
        model,
        criterion,
        sar_loss_fn,
        optimizer,
        device,
        global_stats,
        recon_weight,
        sar_weight,
    ):
        """
        Initialize the trainer.

        Args:
            model: Neural network model.
            criterion: Reconstruction loss function.
            sar_loss_fn: SAR consistency loss function.
            optimizer: Optimizer.
            device: Compute device.
            global_stats: Global statistics.
            recon_weight: Reconstruction loss weight.
            sar_weight: SAR consistency loss weight.
        """
        self.model = model
        self.criterion = criterion
        self.sar_loss_fn = sar_loss_fn
        self.optimizer = optimizer
        self.device = device
        self.global_stats = global_stats
        self.recon_weight = recon_weight
        self.sar_weight = sar_weight

    def train_epoch(self, dataloader, epoch):
        """Train one epoch.

        Args:
            dataloader: Training data loader.
            epoch: Current epoch index.

        Returns:
            tuple: Average total loss, reconstruction loss, SAR loss, and metrics.
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_sar_loss = 0

        # Accumulate metrics across batches.
        total_metrics = {'psnr': 0.0, 'ssim': 0.0, 'rmse': 0.0, 'mae': 0.0}

        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}',file=sys.stdout)

        for batch_idx, batch in enumerate(pbar):
            # Move batch data to the target device.
            lr_dem_norm = batch['lr_dem_norm'].to(self.device)
            hr_dem_norm = batch['hr_dem_norm'].to(self.device)
            hr_dem_raw = batch['hr_dem_raw'].to(self.device)
            vv_sar_norm = batch['vv_sar_norm'].to(self.device)
            vh_sar_norm = batch['vh_sar_norm'].to(self.device)
            inc_angle_maps = batch['inc_angle_map'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass.
            sr_dem_norm = self.model(lr_dem_norm, vv_sar_norm, vh_sar_norm)

            # Compute losses.
            loss_dict = self._compute_losses(sr_dem_norm, hr_dem_norm, hr_dem_raw, inc_angle_maps)
            total_loss_batch = loss_dict['total']

            # Backward pass.
            total_loss_batch.backward()
            self.optimizer.step()

            # Compute evaluation metrics.
            batch_metrics = calculate_batch_metrics(sr_dem_norm, hr_dem_raw, self.global_stats)

            # Accumulate losses and metrics.
            total_loss += loss_dict['total'].item()
            total_recon_loss += loss_dict['recon'].item()
            total_sar_loss += loss_dict['sar'].item()

            for key in total_metrics:
                if key in batch_metrics:
                    total_metrics[key] += batch_metrics[key]

            # Update the progress bar.
            pbar.set_postfix({
                'Loss': f'{loss_dict["total"].item():.4f}',
                'Recon': f'{loss_dict["recon"].item():.4f}',
                'SAR': f'{loss_dict["sar"].item():.4f}',
                'PSNR': f'{batch_metrics.get("psnr", 0):.2f}'
            })

        # Compute averages.
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_sar_loss = total_sar_loss / num_batches

        avg_metrics = {key: total_metrics[key] / num_batches for key in total_metrics}

        return avg_loss, avg_recon_loss, avg_sar_loss, avg_metrics

    def validate_epoch(self, dataloader, epoch):
        """Validate one epoch.

        Args:
            dataloader: Validation data loader.
            epoch: Current epoch index.

        Returns:
            tuple: Average total loss, reconstruction loss, SAR loss, and metrics.
        """
        self.model.eval()
        total_loss, total_recon_loss, total_sar_loss = 0, 0, 0
        total_metrics = {'psnr': 0.0, 'ssim': 0.0, 'rmse': 0.0, 'mae': 0.0}

        pbar = tqdm(dataloader, desc=f'Validation Epoch {epoch}', file=sys.stdout)

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                lr_dem_norm = batch['lr_dem_norm'].to(self.device)
                hr_dem_norm = batch['hr_dem_norm'].to(self.device)
                hr_dem_raw = batch['hr_dem_raw'].to(self.device)
                vv_sar_norm = batch['vv_sar_norm'].to(self.device)
                vh_sar_norm = batch['vh_sar_norm'].to(self.device)
                inc_angle_maps = batch['inc_angle_map'].to(self.device)

                sr_dem_norm = self.model(lr_dem_norm, vv_sar_norm, vh_sar_norm)

                loss_dict = self._compute_losses(sr_dem_norm, hr_dem_norm, hr_dem_raw, inc_angle_maps)

                batch_metrics = calculate_batch_metrics(sr_dem_norm, hr_dem_raw, self.global_stats)

                total_loss += loss_dict['total'].item()
                total_recon_loss += loss_dict['recon'].item()
                total_sar_loss += loss_dict['sar'].item()

                for key in total_metrics:
                    if key in batch_metrics:
                        total_metrics[key] += batch_metrics[key]

                pbar.set_postfix({
                    'Loss': f'{loss_dict["total"].item():.4f}',
                    'Recon': f'{loss_dict["recon"].item():.4f}',
                    'SAR': f'{loss_dict["sar"].item():.4f}',
                    'PSNR': f'{batch_metrics.get("psnr", 0):.2f}'
                })

        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_sar_loss = total_sar_loss / num_batches

        avg_metrics = {key: total_metrics[key] / num_batches for key in total_metrics}

        return avg_loss, avg_recon_loss, avg_sar_loss, avg_metrics

    def _compute_losses(self, sr_dem_norm, hr_dem_norm, hr_dem_raw, inc_angle_maps):
        """Compute loss values.

        Args:
            sr_dem_norm: Normalized super-resolved DEM.
            hr_dem_norm: Normalized high-resolution DEM.
            hr_dem_raw: High-resolution DEM in raw scale.
            inc_angle_maps: Incidence angle maps.

        Returns:
            dict: Loss values.
        """
        recon_loss = self.criterion(sr_dem_norm, hr_dem_norm)

        if self.sar_weight == 0:
            sar_loss = recon_loss.new_tensor(0.0)
        else:
            sr_dem = sr_dem_norm * self.global_stats['dem']['std'] + self.global_stats['dem']['mean']
            hr_dem = hr_dem_raw
            sar_loss, _ = self.sar_loss_fn(sr_dem, hr_dem, inc_angle_maps)

        # Total loss.
        total_loss = self.recon_weight * recon_loss + self.sar_weight * sar_loss

        return {
            'total': total_loss,
            'recon': recon_loss,
            'sar': sar_loss
        }

    def get_sample_prediction(self, dataloader):
        """Get predictions for one batch for visualization.

        Args:
            dataloader: Data loader.

        Returns:
            dict: Inputs and outputs.
        """
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            lr_dem_norm = batch['lr_dem_norm'].to(self.device)
            hr_dem_norm = batch['hr_dem_norm'].to(self.device)
            hr_dem_raw = batch['hr_dem_raw'].to(self.device)
            vv_sar_norm = batch['vv_sar_norm'].to(self.device)
            vh_sar_norm = batch['vh_sar_norm'].to(self.device)
            inc_angle_map = batch['inc_angle_map'].to(self.device)

            sr_dem_norm = self.model(lr_dem_norm, vv_sar_norm, vh_sar_norm)

            return {
                'lr_dem_norm': lr_dem_norm,
                'sr_dem_norm': sr_dem_norm,
                'hr_dem_norm': hr_dem_norm,
                'hr_dem_raw': hr_dem_raw,
                'vv_sar_norm': vv_sar_norm,
                'vh_sar_norm': vh_sar_norm,
                'inc_angle_map': inc_angle_map  # Return incidence angle maps for visualization.
            }
