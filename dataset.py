
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd


class RealWorldHDF5Dataset(Dataset):

    def __init__(self, h5_file_path, global_stats_path, csv_file_path=None):
        """
        Args:
            h5_file_path (str): HDF5 file path.
            global_stats_path (str): Global statistics JSON file path.
            csv_file_path (str, optional): Original index CSV file path for the HDF5 file.
        """
        self.h5_file_path = Path(h5_file_path)
        self.h5_file = None

        # Load global statistics.
        with open(global_stats_path, 'r') as f:
            self.stats = json.load(f)

        # Load the CSV index file.
        self.data_index = None
        if csv_file_path:
            try:
                self.data_index = pd.read_csv(csv_file_path)
            except FileNotFoundError:
                print(f"Index CSV file not found:{csv_file_path}")

        # Read dataset length from the HDF5 file.
        with h5py.File(self.h5_file_path, 'r') as f:
            self.dataset_len = len(f['hr_dem_raw'])
            # Check whether the HDF5 and CSV lengths match.
            if self.data_index is not None and self.dataset_len != len(self.data_index):
                print(f"Mismatch between the number of HDF5 samples ({self.dataset_len}) and the number of CSV rows ({len(self.data_index)})!")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        """
        Get one data sample.
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, 'r')

        hr_dem_raw = self.h5_file['hr_dem_raw'][idx]
        lr_dem_raw = self.h5_file['lr_dem_raw'][idx]
        vv_sar_raw = self.h5_file['vv_sar_raw'][idx]
        vh_sar_raw = self.h5_file['vh_sar_raw'][idx]
        inc_angle_map = self.h5_file['inc_angle_map'][idx]

        vv_log = np.log1p(vv_sar_raw)
        vh_log = np.log1p(vh_sar_raw)

        lr_dem_norm = (lr_dem_raw - self.stats['dem']['mean']) / self.stats['dem']['std']
        vv_sar_norm = (vv_log - self.stats['vv_log']['mean']) / self.stats['vv_log']['std']
        vh_sar_norm = (vh_log - self.stats['vh_log']['mean']) / self.stats['vh_log']['std']

        hr_dem_norm = (hr_dem_raw - self.stats['dem']['mean']) / self.stats['dem']['std']

        sample = {
            'lr_dem_norm': torch.from_numpy(lr_dem_norm).float().unsqueeze(0),
            'hr_dem_raw': torch.from_numpy(hr_dem_raw).float().unsqueeze(0),
            'hr_dem_norm': torch.from_numpy(hr_dem_norm).float().unsqueeze(0), 
            'vv_sar_norm': torch.from_numpy(vv_sar_norm).float().unsqueeze(0),
            'vh_sar_norm': torch.from_numpy(vh_sar_norm).float().unsqueeze(0),
            'inc_angle_map': torch.from_numpy(inc_angle_map).float().unsqueeze(0),
        }

        return sample

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()


def create_dataloader_from_h5(h5_file_path, global_stats_path, csv_file_path, batch_size, num_workers, shuffle=True):

    dataset = RealWorldHDF5Dataset(
        h5_file_path=h5_file_path,
        global_stats_path=global_stats_path,
        csv_file_path=csv_file_path
    )

    # Create and return the DataLoader.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=shuffle
    )
