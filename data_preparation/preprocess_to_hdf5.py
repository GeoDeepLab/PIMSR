import h5py
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Dataset root directory; CSV paths are relative to this directory.
DATA_ROOT = Path(r"E:\TP_DSM_SR_PIMSR\DATA\datasets\不同区域测试\Rocky\rocky_mountains_test_data")# E:\TP_DSM_SR_PIMSR\DATA\datasets\不同区域测试\Rocky\rocky_mountains_test_data     E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Test_data_SRTM

# CSV index files to process.
CSV_FILES_TO_PROCESS = [
    ('data_index.csv', 'test_data_rocky_real_lr.h5'),
]

DATA_CONFIG = {
    'hr_dem_raw': {'csv_col': 'dem_path', 'height': 64, 'width': 64},
    'lr_dem_raw': {'csv_col': 'lr_path', 'height': 16, 'width': 16},
    'vv_sar_raw': {'csv_col': 'vv_path', 'height': 64, 'width': 64},
    'vh_sar_raw': {'csv_col': 'vh_path', 'height': 64, 'width': 64},
    'inc_angle_map': {'csv_col': 'angle_path', 'height': 64, 'width': 64}
}

# =============================================================================

def process_csv_to_hdf5(csv_filename, h5_filename):
    """Read one CSV file and pack referenced TIF data into an HDF5 file.

    This version supports input datasets with different spatial sizes.
    """
    csv_path = DATA_ROOT / csv_filename
    h5_path = DATA_ROOT / h5_filename

    print("-" * 80)
    print(f"Starting to process: {csv_path}")
    print(f"Output will be saved to: {h5_path}")

    try:
        data_index = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        return

    num_samples = len(data_index)
    print(f"Found {num_samples} samples.")

    with h5py.File(h5_path, 'w') as hf:
        datasets = {}
        print("Creating HDF5 datasets:")
        for h5_dset_name, config in DATA_CONFIG.items():
            shape = (num_samples, config['height'], config['width'])
            print(f"  - '{h5_dset_name}' (Shape: {shape})")
            datasets[h5_dset_name] = hf.create_dataset(
                name=h5_dset_name,
                shape=shape,
                dtype=np.float32
            )

        print("\nStarting to iterate through the data and write to the HDF5 file.")
        for idx, row in tqdm(data_index.iterrows(), total=num_samples, desc=f"Processing {csv_filename}"):
            for h5_dset_name, config in DATA_CONFIG.items():
                csv_col = config['csv_col']
                expected_shape = (config['height'], config['width'])

                if csv_col not in row:
                    print(f"\nColumn '{csv_col}' was not found in the CSV file. Skipping...")
                    continue

                try:
                    tif_path = DATA_ROOT / row[csv_col]

                    with rasterio.open(tif_path) as src:
                        image_data = src.read(1).astype(np.float32)

                    # Check whether the image size matches the expected shape.
                    if image_data.shape != expected_shape:
                        print(f"\nImage size {image_data.shape} for sample {idx} does not match the expected "
                              f"{expected_shape}. File: {tif_path}")
                        # Skip mismatched samples instead of resizing them.
                        continue

                    # Write image data to the corresponding HDF5 dataset slot.
                    datasets[h5_dset_name][idx] = image_data

                except FileNotFoundError:
                    print(f"\nFile not found at index {idx}: {tif_path}")
                    continue
                except Exception as e:
                    print(f"\nError processing file {tif_path} at index {idx}: {e}")
                    continue
            # Data writing for this row is complete.

    print(f"\nProcessing completed! Data has been successfully saved to {h5_path}")
    print("-" * 80 + "\n")


if __name__ == '__main__':
    for csv_file, h5_file in CSV_FILES_TO_PROCESS:
        process_csv_to_hdf5(csv_file, h5_file)
    print("All files have been processed.")