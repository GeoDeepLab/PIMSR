import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, kernel_size=3, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                ch,
                ch,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_res=2):
        super().__init__()
        layers = [ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2)]
        for _ in range(num_res):
            layers.append(ResBlock(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, num_res=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(in_ch, out_ch, kernel_size=3),
        )

        layers = [ConvBNAct(out_ch + skip_ch, out_ch, kernel_size=3)]
        for _ in range(num_res):
            layers.append(ResBlock(out_ch))
        self.fuse = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class ExplicitLocalIncidenceAngle(nn.Module):
    """
    Differentiable local geometry module.

    DEM is assumed to be a north-up raster. Image columns point east, while
    image rows point south, so the row derivative sign is flipped to obtain
    dz/dy in the north direction.
    """

    def __init__(self, pixel_size=30.0, look_azimuth_deg=90.0, smooth_dem=True):
        super().__init__()
        self.pixel_size = float(pixel_size)
        self.smooth_dem = smooth_dem

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0) / (8.0 * self.pixel_size)

        sobel_row = torch.tensor(
            [[[-1.0, -2.0, -1.0],
              [0.0, 0.0, 0.0],
              [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0) / (8.0 * self.pixel_size)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_row", sobel_row)
        self.register_buffer(
            "look_azimuth_rad",
            torch.tensor(math.radians(float(look_azimuth_deg)), dtype=torch.float32),
        )

    def _smooth(self, dem):
        if not self.smooth_dem:
            return dem
        dem_pad = F.pad(dem, (1, 1, 1, 1), mode="replicate")
        return F.avg_pool2d(dem_pad, kernel_size=3, stride=1)

    def gradients(self, dem):
        dem_smooth = self._smooth(dem)
        dem_pad = F.pad(dem_smooth, (1, 1, 1, 1), mode="replicate")
        dz_dx = F.conv2d(dem_pad, self.sobel_x)
        dz_drow = F.conv2d(dem_pad, self.sobel_row)
        dz_dy = -dz_drow
        return dz_dx, dz_dy

    def compute(self, dem, eia_deg):
        eps = 1e-6
        dz_dx, dz_dy = self.gradients(dem)

        ones = torch.ones_like(dz_dx)
        normal = torch.cat([-dz_dx, -dz_dy, ones], dim=1)
        normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(eps)

        theta = torch.deg2rad(eia_deg)
        beta = self.look_azimuth_rad.to(device=theta.device, dtype=theta.dtype)

        sin_theta = torch.sin(theta)
        look_x = sin_theta * torch.sin(beta)
        look_y = sin_theta * torch.cos(beta)
        look_z = torch.cos(theta)

        cos_lia = (
            normal[:, 0:1] * look_x
            + normal[:, 1:2] * look_y
            + normal[:, 2:3] * look_z
        )
        # Keep acos away from exactly 1.0. During DEM SR training the simulator
        # is part of the gradient path; acos'(1) is infinite and can turn an
        # otherwise finite SAR loss into NaN gradients.
        cos_lia = cos_lia.clamp(min=1e-4, max=1.0 - 1e-4)
        lia_deg = torch.rad2deg(torch.acos(cos_lia)).clamp(min=0.0, max=89.0)

        return {
            "lia_deg": lia_deg,
            "cos_lia": cos_lia,
            "dz_dx": dz_dx,
            "dz_dy": dz_dy,
        }

    def forward(self, dem, eia_deg):
        return self.compute(dem, eia_deg)["lia_deg"]


class PreciseAngleSimulator(nn.Module):
    """
    Explicit LIA + physics guided SAR simulator with RD inspired geometry correction.

    Class name and interface are unchanged:
        sar_sim_raw = model(dem_input, true_inc_angle_map_deg)

    Input:
        dem_input: [B, 1, H, W]
        true_inc_angle_map_deg: [B, 1, H, W]

    Output:
        simulated raw SAR intensity: [B, 2, H, W]
        channel 0: VV
        channel 1: VH
    """

    def __init__(
        self,
        pixel_size=30.0,
        look_azimuth_deg=90.0,
        base_ch=32,
        max_log_gain=1.5,
        max_residual_scale=0.2,
        patch_param_delta_scale=0.8,
        smooth_dem=True,
    ):
        super().__init__()

        self.max_log_gain = max_log_gain
        self.max_residual_scale = max_residual_scale
        self.patch_param_delta_scale = patch_param_delta_scale
        self.look_azimuth_deg = float(look_azimuth_deg)

        self.local_geometry = ExplicitLocalIncidenceAngle(
            pixel_size=pixel_size,
            look_azimuth_deg=look_azimuth_deg,
            smooth_dem=smooth_dem,
        )

        # Learnable physical parameters.
        self.log_A_vv = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_K_vv = nn.Parameter(torch.log(torch.tensor(4.0)))
        self.log_A_vh = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_K_vh = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.log_C_diffuse = nn.Parameter(torch.log(torch.tensor(0.05)))

        # Patch-wise physical deltas condition the global physical parameters
        # without requiring a dataset index or lookup table.
        self.physical_param_head = nn.Sequential(
            nn.Conv2d(9, base_ch, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 5, kernel_size=1),
        )

        # Geometry features:
        # dem_norm, eia_norm, lia_norm, dz_dx, dz_dy, slope_mag,
        # radar_facing, radar_away, roughness, dz_dlook, foreshortening,
        # shadow_factor, shadow_prob, layover_prob, curvature, base_vv, base_vh
        in_ch = 17

        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base_ch),
            ResBlock(base_ch),
            ResBlock(base_ch),
        )

        self.down1 = DownBlock(base_ch, base_ch * 2, num_res=2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, num_res=2)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8, num_res=2)

        self.bottleneck = nn.Sequential(
            ResBlock(base_ch * 8, dilation=1),
            ResBlock(base_ch * 8, dilation=2),
            ResBlock(base_ch * 8, dilation=3),
            ResBlock(base_ch * 8, dilation=1),
        )

        self.up2 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, num_res=2)
        self.up1 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, num_res=2)
        self.up0 = UpBlock(base_ch * 2, base_ch, base_ch, num_res=2)

        self.refine = nn.Sequential(
            ResBlock(base_ch),
            ResBlock(base_ch),
            ConvBNAct(base_ch, base_ch),
        )

        self.log_gain_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 2, 3, padding=1),
        )

        self.residual_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 2, 3, padding=1),
        )

        self.bias_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 2, 3, padding=1),
        )

        self.scatter_mod_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 2, 3, padding=1),
        )

        self.direct_sar_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 2, 3, padding=1),
        )

        self.mix_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 2, 3, padding=1),
        )

    def _normalize_dem(self, dem_input):
        dem_mean = dem_input.mean(dim=(2, 3), keepdim=True)
        dem_var = (dem_input - dem_mean).pow(2).mean(dim=(2, 3), keepdim=True)
        dem_std = dem_var.clamp_min(1e-6).sqrt()
        return (dem_input - dem_mean) / dem_std

    def _roughness(self, dem_norm, slope_mag):
        eps = 1e-6
        dem_pad = F.pad(dem_norm, (2, 2, 2, 2), mode="replicate")
        local_mean = F.avg_pool2d(dem_pad, kernel_size=5, stride=1)
        local_sq_mean = F.avg_pool2d(dem_pad * dem_pad, kernel_size=5, stride=1)
        local_var = (local_sq_mean - local_mean.pow(2)).clamp_min(0.0)
        local_std = (local_var + eps).sqrt()

        slope_scale = slope_mag.mean(dim=(2, 3), keepdim=True).clamp_min(eps)
        slope_norm = slope_mag / slope_scale

        roughness = 0.5 + torch.sigmoid(local_std + slope_norm - 1.0)
        return roughness.clamp(min=0.5, max=1.5)

    def _radar_facing_terms(self, dz_dx, dz_dy):
        beta = torch.tensor(
            math.radians(self.look_azimuth_deg),
            device=dz_dx.device,
            dtype=dz_dx.dtype,
        )
        slope_mag = torch.sqrt(dz_dx.pow(2) + dz_dy.pow(2) + 1e-6)
        unit_x = dz_dx / slope_mag
        unit_y = dz_dy / slope_mag
        rel = unit_x * torch.sin(beta) + unit_y * torch.cos(beta)
        facing = F.relu(rel)
        away = F.relu(-rel)
        return facing, away

    def _look_direction_terms(self, dz_dx, dz_dy, true_inc_angle_map_deg, cos_lia):
        beta = torch.tensor(
            math.radians(self.look_azimuth_deg),
            device=dz_dx.device,
            dtype=dz_dx.dtype,
        )
        theta = torch.deg2rad(true_inc_angle_map_deg)

        dz_dlook = dz_dx * torch.sin(beta) + dz_dy * torch.cos(beta)
        look_slope = torch.tanh(dz_dlook)

        slope_threshold = torch.tan(theta).clamp_min(1e-4)
        layover_prob = torch.sigmoid((dz_dlook - slope_threshold) / 0.15)

        foreshortening = 1.0 / cos_lia.clamp_min(0.12)
        foreshortening = torch.clamp(foreshortening, min=1.0, max=6.0) / 6.0

        shadow_prob = torch.sigmoid((0.18 - cos_lia) / 0.04)
        shadow_factor = 1.0 - 0.85 * shadow_prob

        return {
            "dz_dlook": dz_dlook,
            "look_slope": look_slope,
            "layover_prob": layover_prob,
            "foreshortening": foreshortening,
            "shadow_prob": shadow_prob,
            "shadow_factor": shadow_factor,
        }

    def _curvature(self, dem_norm):
        lap_kernel = torch.tensor(
            [[[[0.0, 1.0, 0.0],
               [1.0, -4.0, 1.0],
               [0.0, 1.0, 0.0]]]],
            device=dem_norm.device,
            dtype=dem_norm.dtype,
        )
        dem_pad = F.pad(dem_norm, (1, 1, 1, 1), mode="replicate")
        curvature = F.conv2d(dem_pad, lap_kernel)
        return torch.tanh(curvature)

    def _patch_physical_params(
        self,
        dem_norm,
        eia_norm,
        lia_norm,
        dz_dx,
        dz_dy,
        slope_mag,
        roughness,
        radar_facing,
        radar_away,
    ):
        param_features = torch.cat(
            [
                dem_norm,
                eia_norm,
                lia_norm,
                torch.tanh(dz_dx),
                torch.tanh(dz_dy),
                torch.tanh(slope_mag),
                roughness,
                radar_facing,
                radar_away,
            ],
            dim=1,
        )
        pooled_features = F.adaptive_avg_pool2d(param_features, output_size=1)
        delta = self.patch_param_delta_scale * torch.tanh(
            self.physical_param_head(pooled_features)
        )

        log_params = torch.stack(
            [
                self.log_A_vv,
                self.log_K_vv,
                self.log_A_vh,
                self.log_K_vh,
                self.log_C_diffuse,
            ]
        ).view(1, 5, 1, 1)
        params = torch.exp(log_params + delta)

        return {
            "A_vv": params[:, 0:1],
            "K_vv": params[:, 1:2],
            "A_vh": params[:, 2:3],
            "K_vh": params[:, 3:4],
            "C_diffuse": params[:, 4:5],
        }

    def _physical_base(self, cos_lia, roughness, shadow_factor, foreshortening, params):
        A_vv = params["A_vv"]
        K_vv = params["K_vv"]
        A_vh = params["A_vh"]
        K_vh = params["K_vh"]
        C_diffuse = params["C_diffuse"]

        diffuse_component = C_diffuse * roughness * cos_lia
        specular_component_vv = A_vv * cos_lia.pow(K_vv)
        terrain_brightening = 1.0 + 1.5 * foreshortening
        base_vv = (specular_component_vv + diffuse_component) * terrain_brightening

        base_vh = A_vh * cos_lia.pow(K_vh) * (1.0 + 0.5 * roughness)

        base_sar = torch.cat([base_vv, base_vh], dim=1)
        return (base_sar * shadow_factor).clamp_min(1e-8)

    def compute_geometry(self, dem_input, true_inc_angle_map_deg):
        dem_norm = self._normalize_dem(dem_input)
        eia_norm = (true_inc_angle_map_deg - 35.0) / 10.0
        geometry = self.local_geometry.compute(dem_input, true_inc_angle_map_deg)

        dz_dx = geometry["dz_dx"]
        dz_dy = geometry["dz_dy"]
        slope_mag = torch.sqrt(dz_dx.pow(2) + dz_dy.pow(2) + 1e-8)
        roughness = self._roughness(dem_norm, slope_mag)
        lia_norm = (geometry["lia_deg"] - 35.0) / 10.0
        radar_facing, radar_away = self._radar_facing_terms(dz_dx, dz_dy)
        look_terms = self._look_direction_terms(
            dz_dx,
            dz_dy,
            true_inc_angle_map_deg,
            geometry["cos_lia"],
        )
        curvature = self._curvature(dem_norm)
        params = self._patch_physical_params(
            dem_norm,
            eia_norm,
            lia_norm,
            dz_dx,
            dz_dy,
            slope_mag,
            roughness,
            radar_facing,
            radar_away,
        )
        base_sar = self._physical_base(
            geometry["cos_lia"],
            roughness,
            look_terms["shadow_factor"],
            look_terms["foreshortening"],
            params,
        )

        return {
            "lia_deg": geometry["lia_deg"],
            "cos_lia": geometry["cos_lia"],
            "dz_dx": dz_dx,
            "dz_dy": dz_dy,
            "slope_mag": slope_mag,
            "roughness": roughness,
            "dz_dlook": look_terms["dz_dlook"],
            "look_slope": look_terms["look_slope"],
            "layover_prob": look_terms["layover_prob"],
            "foreshortening": look_terms["foreshortening"],
            "shadow_prob": look_terms["shadow_prob"],
            "shadow_factor": look_terms["shadow_factor"],
            "curvature": curvature,
            "base_sar": base_sar,
            "A_vv": params["A_vv"],
            "K_vv": params["K_vv"],
            "A_vh": params["A_vh"],
            "K_vh": params["K_vh"],
            "C_diffuse": params["C_diffuse"],
        }

    def forward(self, dem_input, true_inc_angle_map_deg):
        geometry = self.compute_geometry(dem_input, true_inc_angle_map_deg)

        dem_norm = self._normalize_dem(dem_input)
        eia_norm = (true_inc_angle_map_deg - 35.0) / 10.0
        lia_norm = (geometry["lia_deg"] - 35.0) / 10.0

        dz_dx = geometry["dz_dx"]
        dz_dy = geometry["dz_dy"]
        slope_mag = geometry["slope_mag"]
        roughness = geometry["roughness"]
        look_slope = geometry["look_slope"]
        foreshortening = geometry["foreshortening"]
        shadow_factor = geometry["shadow_factor"]
        shadow_prob = geometry["shadow_prob"]
        layover_prob = geometry["layover_prob"]
        curvature = geometry["curvature"]
        base_sar = geometry["base_sar"]

        radar_facing, radar_away = self._radar_facing_terms(dz_dx, dz_dy)

        features = torch.cat(
            [
                dem_norm,
                eia_norm,
                lia_norm,
                torch.tanh(dz_dx),
                torch.tanh(dz_dy),
                torch.tanh(slope_mag),
                radar_facing,
                radar_away,
                roughness,
                look_slope,
                foreshortening,
                shadow_factor,
                shadow_prob,
                layover_prob,
                curvature,
                base_sar,
            ],
            dim=1,
        )

        x0 = self.stem(features)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        xb = self.bottleneck(x3)

        y2 = self.up2(xb, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)

        feat = self.refine(y0)

        log_gain = self.max_log_gain * torch.tanh(self.log_gain_head(feat))
        residual = self.max_residual_scale * torch.tanh(self.residual_head(feat))
        bias = 0.03 * F.softplus(self.bias_head(feat))
        scatter_mod = torch.exp(1.2 * torch.tanh(self.scatter_mod_head(feat)))

        corrected_base = base_sar * scatter_mod * torch.exp(log_gain) + residual + bias

        # Direct branch gives the network enough freedom to match SAR intensity
        # distributions when the physics prior is under-calibrated for a patch.
        direct_sar = F.softplus(self.direct_sar_head(feat) - 2.5) + 1e-8
        physical_mix = torch.sigmoid(self.mix_head(feat))

        sar_sim = physical_mix * corrected_base + (1.0 - physical_mix) * direct_sar
        sar_sim = sar_sim.clamp_min(1e-8)

        return sar_sim
