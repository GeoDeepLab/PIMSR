import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm
import json


def update_stats(arr, state):
    arr = arr.astype(np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return state

    state["sum"] += arr.sum()
    state["sum_sq"] += np.square(arr).sum()
    state["count"] += arr.size
    return state


def finalize_stats(state):
    if state["count"] == 0:
        return {"mean": None, "std": None}

    mean = state["sum"] / state["count"]
    var = state["sum_sq"] / state["count"] - mean ** 2
    std = np.sqrt(max(var, 0.0))

    return {
        "mean": float(mean),
        "std": float(std)
    }


def calculate_and_save_stats_fast(root_dir, csv_file, output_path):
    if output_path.exists():
        print(f"The statistics file already exists at: {output_path}")
        user_input = input("Do you want to recompute it? (y/n): ").lower()
        if user_input != "y":
            print("Operation cancelled.")
            return

    print("Start calculating global normalization statistics.")

    df = pd.read_csv(root_dir / csv_file)

    required_cols = ["dem_path", "vv_path", "vh_path"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    states = {
        "dem": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "vv_log": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "vh_log": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing DEM/VV/VH"):
        dem_path = root_dir / row["dem_path"]
        vv_path = root_dir / row["vv_path"]
        vh_path = root_dir / row["vh_path"]

        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
            states["dem"] = update_stats(dem, states["dem"])

        with rasterio.open(vv_path) as src:
            vv = src.read(1).astype(np.float32)
            vv_log = np.log1p(vv)
            states["vv_log"] = update_stats(vv_log, states["vv_log"])

        with rasterio.open(vh_path) as src:
            vh = src.read(1).astype(np.float32)
            vh_log = np.log1p(vh)
            states["vh_log"] = update_stats(vh_log, states["vh_log"])

    stats = {
        "dem": finalize_stats(states["dem"]),
        "vv_log": finalize_stats(states["vv_log"]),
        "vh_log": finalize_stats(states["vh_log"]),
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)

    print("\nStatistical data calculation complete.")
    print(f"Global statistics have been saved to: {output_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    TRAIN_DATA_ROOT = Path(r"E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Train_data")
    TRAIN_CSV = "data_index.csv"
    STATS_JSON_PATH = TRAIN_DATA_ROOT / "global_stats.json"

    calculate_and_save_stats_fast(TRAIN_DATA_ROOT, TRAIN_CSV, STATS_JSON_PATH)