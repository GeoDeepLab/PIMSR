import torch
import torch.nn.functional as F
import math


def torch_ssim(pred, target, win_size=7, data_range=None):
    """
    pred, target: [B, C, H, W]
    Return scalar SSIM averaged over batch and channels.
    """

    if data_range is None:
        data_range = target.max() - target.min()
        data_range = torch.clamp(data_range, min=1.0)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    padding = win_size // 2

    mu_x = F.avg_pool2d(pred, win_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, win_size, stride=1, padding=padding)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(pred * pred, win_size, stride=1, padding=padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target * target, win_size, stride=1, padding=padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(pred * target, win_size, stride=1, padding=padding) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / (denominator + 1e-8)

    return ssim_map.mean()


def channel_metrics(pred, target, win_size=7):
    """
    pred, target: [B, 2, H, W]
    Return metrics for VV and VH separately.
    """

    results = {}

    for ch, name in enumerate(["VV", "VH"]):
        p = pred[:, ch:ch + 1]
        t = target[:, ch:ch + 1]

        error = p - t
        mse = torch.mean(error ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(error))

        data_range = t.max() - t.min()
        data_range = torch.clamp(data_range, min=1.0)

        psnr = 20.0 * torch.log10(data_range / (rmse + 1e-8))
        ssim = torch_ssim(p, t, win_size=win_size, data_range=data_range)

        results[name] = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "rmse": rmse.item(),
            "mae": mae.item(),
        }

    return results