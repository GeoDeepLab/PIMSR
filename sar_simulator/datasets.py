import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


class H5SimulatorDataset(Dataset):
    """
    HDF5 dataset for SAR simulator pretraining.

    Required HDF5 keys:
        hr_dem_raw
        vv_sar_raw
        vh_sar_raw
        inc_angle_map
    """

    def __init__(
        self,
        h5_path,
        normalization_stats=None,
        dem_key="hr_dem_raw",
        vv_key="vv_sar_raw",
        vh_key="vh_sar_raw",
        angle_key="inc_angle_map",
        sar_filter="lee",
        sar_filter_window=5,
    ):
        self.h5_path = str(h5_path)
        self.stats = normalization_stats

        self.dem_key = dem_key
        self.vv_key = vv_key
        self.vh_key = vh_key
        self.angle_key = angle_key
        self.sar_filter = sar_filter
        self.sar_filter_window = sar_filter_window

        self._h5_file = None

        with h5py.File(self.h5_path, "r") as hf:
            self.length = hf[self.dem_key].shape[0]

    def __len__(self):
        return self.length

    def _open_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def _clean_sar(self, sar_data):
        sar_data = np.nan_to_num(sar_data, nan=0.0, posinf=0.0, neginf=0.0)
        sar_data = np.clip(sar_data, 0.0, None)
        return sar_data.astype(np.float32, copy=False)

    def _local_mean_var(self, sar_data, window_size):
        pad = window_size // 2
        padded = np.pad(sar_data, pad_width=pad, mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (window_size, window_size),
        )

        local_mean = windows.mean(axis=(-2, -1))
        local_var = windows.var(axis=(-2, -1))

        return local_mean, local_var

    def _lee_filter_sar(self, sar_data):
        window_size = int(self.sar_filter_window)
        if window_size < 3:
            return sar_data.astype(np.float32, copy=False)
        if window_size % 2 == 0:
            window_size += 1

        local_mean, local_var = self._local_mean_var(sar_data, window_size)
        noise_var = np.percentile(local_var, 10.0)

        if noise_var <= 1e-12:
            return sar_data.astype(np.float32, copy=False)

        weight = local_var / (local_var + noise_var + 1e-12)
        filtered = local_mean + weight * (sar_data - local_mean)
        filtered = np.clip(filtered, 0.0, None)

        return filtered.astype(np.float32, copy=False)

    def _filter_sar_noise(self, sar_data):
        sar_data = self._clean_sar(sar_data)

        if self.sar_filter is None:
            return sar_data

        if self.sar_filter == "lee":
            return self._lee_filter_sar(sar_data)

        if self.sar_filter == "mean":
            local_mean, _ = self._local_mean_var(sar_data, int(self.sar_filter_window))
            return np.clip(local_mean, 0.0, None).astype(np.float32, copy=False)

        raise ValueError(f"Unsupported SAR filter: {self.sar_filter}")

    def __getitem__(self, idx):
        hf = self._open_h5()

        dem_data = hf[self.dem_key][idx].astype(np.float32)
        vv_original = np.nan_to_num(
            hf[self.vv_key][idx].astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        vh_original = np.nan_to_num(
            hf[self.vh_key][idx].astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        vv_data = self._filter_sar_noise(vv_original)
        vh_data = self._filter_sar_noise(vh_original)
        angle_data = hf[self.angle_key][idx].astype(np.float32)

        sar_target_original_raw = np.stack([vv_original, vh_original], axis=0)
        sar_target_raw = np.stack([vv_data, vh_data], axis=0)

        return {
            "dem_true": torch.from_numpy(dem_data).unsqueeze(0),
            "sar_target_original_raw": torch.from_numpy(sar_target_original_raw),
            "sar_target_raw": torch.from_numpy(sar_target_raw),
            "sar_target_norm": torch.from_numpy(sar_target_raw),
            "inc_angle_map": torch.from_numpy(angle_data).unsqueeze(0),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def __del__(self):
        if getattr(self, "_h5_file", None) is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
