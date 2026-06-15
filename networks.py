

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from sar_simulator.models import PreciseAngleSimulator
from model_blocks import ResidualBlockCBAM, DEMFeatureExtractor, CBAM

class ResidualSARDEMGenerator(nn.Module):
    """SAR-enhanced DEM super-resolution generator.

    The model uses residual learning to generate high-resolution DEM data from
    SAR data and low-resolution DEM input with multi-scale feature extraction
    and fusion.
    """

    def __init__(self, sar_input_mode='both', fusion_scales=None):
        super().__init__()
        valid_sar_modes = {'both', 'vv', 'vh', 'none'}
        if sar_input_mode not in valid_sar_modes:
            raise ValueError(f"Unsupported sar_input_mode: {sar_input_mode}")
        self.sar_input_mode = sar_input_mode

        if fusion_scales is None:
            fusion_scales = {'16', '32', '64'}
        elif isinstance(fusion_scales, str):
            fusion_scales = {scale.strip() for scale in fusion_scales.split(',') if scale.strip()}
        else:
            fusion_scales = {str(scale) for scale in fusion_scales}
        valid_scales = {'16', '32', '64'}
        if not fusion_scales or not fusion_scales.issubset(valid_scales):
            raise ValueError(f"fusion_scales must be a non-empty subset of {sorted(valid_scales)}")
        self.fusion_scales = fusion_scales

        # SAR encoders for VV and VH with the same architecture.
        self.sar_encoder = nn.ModuleDict({
            'vv': self._make_sar_encoder(),
            'vh': self._make_sar_encoder()
        })

        # DEM encoder.
        self.dem_encoder = nn.Sequential(
            # 16x16 -> 16x16
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlockCBAM(64),
            ResidualBlockCBAM(64)
        )

        # DEM gradient feature extractor.
        self.dem_feature_extractor = DEMFeatureExtractor(64, 64)

        # Multi-scale feature fusion modules.
        self.fusion_16 = nn.Sequential(
            nn.Conv2d(64 * 3, 128, 3, padding=1),  # (64+64+64) -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAM(128)
        )

        self.fusion_32 = nn.Sequential(
            nn.Conv2d(64 * 3, 64, 3, padding=1),  # (64+64+64) -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAM(64)
        )

        self.fusion_64 = nn.Sequential(
            nn.Conv2d(64 * 3, 32, 3, padding=1),  # (64+64+64) -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32)
        )

        # Decoder with upsampling and fusion.
        self.decoder_32 = nn.Sequential(
            nn.Conv2d(128, 64 * (2 ** 2), kernel_size=3, padding=1),  # Output channels become 64 * 4 = 256.
            nn.PixelShuffle(2),  # Rearrange (B, 256, H, W) to (B, 64, H*2, W*2).
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlockCBAM(64)
        )

        self.decoder_64 = nn.Sequential(
            nn.Conv2d(64, 64 * (2 ** 2), kernel_size=3, padding=1),  # Output channels become 64 * 4 = 256.
            nn.PixelShuffle(2),  # Rearrange (B, 256, H, W) to (B, 64, H*2, W*2).
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlockCBAM(64)
        )

        # Final residual prediction.
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def _make_sar_encoder(self):
        """Create one SAR encoder branch."""
        return nn.ModuleDict({
            'conv64': nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ResidualBlockCBAM(64),
            ),
            'conv32': nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ResidualBlockCBAM(64),
            ),
            'conv16': nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ResidualBlockCBAM(64),
            )
        })

    def forward(self, x_lr_dem, x_vv, x_vh):
        """
        Forward pass.

        Args:
            x_lr_dem: Low-resolution DEM [B, 1, 16, 16].
            x_vv: VV-polarized SAR [B, 1, 64, 64].
            x_vh: VH-polarized SAR [B, 1, 64, 64].
        Returns:
            sr_dem: High-resolution DEM [B, 1, 64, 64].
        """
        # Upsample LR DEM with bicubic interpolation to form the base SR.
        base_sr = F.interpolate(x_lr_dem, scale_factor=4, mode='bicubic', align_corners=False)

        # Extract multi-scale SAR features.
        vv_feat = {}
        vh_feat = {}

        batch_size, _, height, width = x_vv.shape
        zero_64 = x_vv.new_zeros(batch_size, 64, height, width)
        zero_32 = x_vv.new_zeros(batch_size, 64, height // 2, width // 2)
        zero_16 = x_vv.new_zeros(batch_size, 64, height // 4, width // 4)

        if self.sar_input_mode in {'both', 'vv'}:
            vv_64 = self.sar_encoder['vv']['conv64'](x_vv)
            vv_32 = self.sar_encoder['vv']['conv32'](vv_64)
            vv_16 = self.sar_encoder['vv']['conv16'](vv_32)
            vv_feat = {'64': vv_64, '32': vv_32, '16': vv_16}
        else:
            vv_feat = {'64': zero_64, '32': zero_32, '16': zero_16}

        if self.sar_input_mode in {'both', 'vh'}:
            vh_64 = self.sar_encoder['vh']['conv64'](x_vh)
            vh_32 = self.sar_encoder['vh']['conv32'](vh_64)
            vh_16 = self.sar_encoder['vh']['conv16'](vh_32)
            vh_feat = {'64': vh_64, '32': vh_32, '16': vh_16}
        else:
            vh_feat = {'64': zero_64, '32': zero_32, '16': zero_16}

        # Extract low-resolution DEM features.
        dem_feat = self.dem_encoder(x_lr_dem)

        gradient_feat = self.dem_feature_extractor(dem_feat)
        enhanced_dem_feat = dem_feat + gradient_feat  # Residual connection for enhanced DEM features.

        # Fuse features and upsample through the U-Net-style decoder.
        vv_16 = vv_feat['16'] if '16' in self.fusion_scales else torch.zeros_like(vv_feat['16'])
        vh_16 = vh_feat['16'] if '16' in self.fusion_scales else torch.zeros_like(vh_feat['16'])
        feat_16 = torch.cat([enhanced_dem_feat, vv_16, vh_16], dim=1)
        fused_16 = self.fusion_16(feat_16)

        up_32 = self.decoder_32(fused_16)
        vv_32 = vv_feat['32'] if '32' in self.fusion_scales else torch.zeros_like(vv_feat['32'])
        vh_32 = vh_feat['32'] if '32' in self.fusion_scales else torch.zeros_like(vh_feat['32'])
        feat_32 = torch.cat([up_32, vv_32, vh_32], dim=1)
        fused_32 = self.fusion_32(feat_32)

        up_64 = self.decoder_64(fused_32)
        vv_64 = vv_feat['64'] if '64' in self.fusion_scales else torch.zeros_like(vv_feat['64'])
        vh_64 = vh_feat['64'] if '64' in self.fusion_scales else torch.zeros_like(vh_feat['64'])
        feat_64 = torch.cat([up_64, vv_64, vh_64], dim=1)
        fused_64 = self.fusion_64(feat_64)

        # Predict the residual.
        residual = self.final_conv(fused_64)

        # Add the residual to the base SR.
        sr_dem = base_sr + residual

        return sr_dem


class SARLoss(nn.Module):
    """SAR consistency loss.

    Use the SAR simulator to measure SAR-domain consistency between generated
    and ground-truth DEM data.
    """

    def __init__(self, simulator_weights=None):
        super(SARLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

        # Load the SAR simulator.
        if simulator_weights:
            self.sar_simulator = PreciseAngleSimulator()
            try:
                checkpoint = torch.load(simulator_weights, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.sar_simulator.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.sar_simulator.load_state_dict(checkpoint)
                print(f"Loaded SAR simulator from {simulator_weights}")
                self.sar_simulator.eval()
                for param in self.sar_simulator.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Error loading SAR simulator weights: {e}")
                self.sar_simulator = None
        else:
            self.sar_simulator = None

    def forward(self, sr_dem, hr_dem, incidence_angle_map):
        """Compute the SAR consistency loss."""
        if self.sar_simulator is None:
            return torch.tensor(0.0, device=sr_dem.device), {}

        with torch.no_grad():
            sar_hr = self.sar_simulator(hr_dem.detach(), incidence_angle_map.detach())
            vv_sim_hr = sar_hr[:, 0:1]
            vh_sim_hr = sar_hr[:, 1:2]

        sar_sr = self.sar_simulator(sr_dem, incidence_angle_map.detach())
        vv_sim_sr = sar_sr[:, 0:1]
        vh_sim_sr = sar_sr[:, 1:2]

        vv_loss = self.l1_loss(vv_sim_sr, vv_sim_hr)
        vh_loss = self.l1_loss(vh_sim_sr, vh_sim_hr)

        # Average VV and VH losses.
        total_loss = (vv_loss + vh_loss) / 2

        # Return loss details.
        loss_dict = {
            'vv_loss': vv_loss.item(),
            'vh_loss': vh_loss.item()
        }

        return total_loss, loss_dict
