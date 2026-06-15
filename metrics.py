import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import mean_squared_error


def calculate_batch_metrics(sr_dem_norm, hr_dem_raw, stats):

    with torch.no_grad():
        sr_np = sr_dem_norm.detach().cpu().numpy()
        hr_np = hr_dem_raw.detach().cpu().numpy()
        sr_denorm = sr_np * stats['dem']['std'] + stats['dem']['mean']

        psnr_sum, ssim_sum, rmse_sum, mae_sum = 0, 0, 0, 0
        batch_size = sr_np.shape[0]

        for i in range(batch_size):
            sr_sample = sr_denorm[i, 0]
            hr_sample = hr_np[i, 0]

            hr_variance = np.var(hr_sample)
            if hr_variance < 1e-4:
                ssim_val = 1.0
            else:

                overall_min = np.min([np.min(hr_sample), np.min(sr_sample)])
                overall_max = np.max([np.max(hr_sample), np.max(sr_sample)])

                data_range = overall_max - overall_min
                if data_range < 1e-6:
                    data_range = 1.0

                raw_ssim = ssim_metric(
                    hr_sample,
                    sr_sample,
                    data_range=data_range,  # Use the combined range.
                    win_size=5,
                    channel_axis=None
                )

                ssim_val = np.clip(raw_ssim, -1.0, 1.0)

            ssim_sum += ssim_val

            mse = mean_squared_error(hr_sample, sr_sample)
            if mse < 1e-10:
                psnr = 100
            else:
                psnr_data_range = np.max(hr_sample) - np.min(hr_sample)
                if psnr_data_range < 1e-6: psnr_data_range = 1.0
                psnr = 20 * np.log10(psnr_data_range / np.sqrt(mse))
            psnr_sum += psnr
            rmse_sum += np.sqrt(mse)
            mae_sum += np.mean(np.abs(hr_sample - sr_sample))

        return {
            'psnr': psnr_sum / batch_size,
            'ssim': ssim_sum / batch_size,
            'rmse': rmse_sum / batch_size,
            'mae': mae_sum / batch_size
        }


def calculate_single_metrics(sr_dem, hr_dem):

    hr_variance = np.var(hr_dem)
    if hr_variance < 1e-4:
        ssim = 1.0
    else:
        data_range = np.max(hr_dem) - np.min(hr_dem)
        if data_range < 1e-6:
            data_range = 1.0
        ssim = ssim_metric(
            hr_dem,
            sr_dem,
            data_range=data_range,
            win_size=5,
            channel_axis=None
        )

    # PSNR
    mse = mean_squared_error(hr_dem, sr_dem)
    if mse < 1e-10:
        psnr = 100
    else:
        psnr_data_range = np.max(hr_dem) - np.min(hr_dem)
        if psnr_data_range < 1e-6: psnr_data_range = 1.0
        psnr = 20 * np.log10(psnr_data_range / np.sqrt(mse))

    # RMSE
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(hr_dem - sr_dem))

    return {
        'psnr': psnr,
        'ssim': ssim,
        'rmse': rmse,
        'mae': mae
    }


def denormalize_dem(dem_norm, stats):
    """Denormalize DEM data.

    Args:
        dem_norm: Normalized DEM data.
        stats: Statistics containing mean and standard deviation.

    Returns:
        Denormalized DEM data.
    """
    return dem_norm * stats['dem']['std'] + stats['dem']['mean']


def print_metrics(phase, epoch, num_epochs, loss_dict, metrics, lr, time_cost):
    """Print training metrics."""
    print(f"{phase} Epoch: [{epoch:03d}/{num_epochs - 1:03d}]")
    print(f"  Loss: total={loss_dict['total']:.6f}, recon={loss_dict['recon']:.6f}, sar={loss_dict['sar']:.6f}")
    print(f"  PSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.2f}m, MAE: {metrics['mae']:.2f}m")
    print(f"  LR: {lr:.2e}, Time: {time_cost:.1f}s")
    print("-" * 80)
