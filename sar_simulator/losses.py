import torch.nn as nn

from metrics import torch_ssim


class L1SSIMLoss(nn.Module):
    def __init__(self, ssim_weight=1.0, win_size=7):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.win_size = win_size
        self.l1 = nn.L1Loss()

    def forward(self, y_pred, y_true):
        l1_loss = self.l1(y_pred, y_true)
        ssim_loss = 1.0 - torch_ssim(
            y_pred,
            y_true,
            win_size=self.win_size,
        )
        total_loss = l1_loss + self.ssim_weight * ssim_loss

        return total_loss, l1_loss, ssim_loss
