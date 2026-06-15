import subprocess
from pathlib import Path


DEFAULT_GLOBAL_STATS = r"E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Train_data\global_stats.json"

TEST_DATASETS = {
    "test_data": {
        "h5_path": r"E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Test_data\test_data_real_lr.h5",
        "csv_path": r"E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Test_data\data_index.csv",
        "data_root": r"E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Test_data",
    },
    "test_data_srtm": {
        "h5_path": r"E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Test_data_SRTM\test_data_srtm_real_lr.h5",
        "csv_path": r"E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Test_data_SRTM\data_index.csv",
        "data_root": r"E:\TP_DSM_SR_PIMSR\PIMSR\datasets\Test_data_SRTM",
    },
}


def parse_dataset_keys(dataset_arg):
    """Parse a comma-separated dataset selector."""
    if dataset_arg in {"all", "both"}:
        return list(TEST_DATASETS)

    keys = [key.strip() for key in dataset_arg.split(",") if key.strip()]
    invalid = [key for key in keys if key not in TEST_DATASETS]
    if invalid:
        valid = ", ".join(TEST_DATASETS)
        raise ValueError(f"Unknown dataset key(s): {', '.join(invalid)}. Valid keys: {valid}, all")
    return keys


def test_complete(output_dir):
    return (Path(output_dir) / "test_summary.txt").exists()


def run_test_case(
    *,
    python_executable,
    test_script,
    dataset_key,
    dataset_config,
    global_stats,
    model_weights,
    output_dir,
    sar_input_mode,
    fusion_scales,
    batch_size,
    num_workers,
    vis_freq,
    save_geotiff,
    dry_run=False,
    force_rerun=False,
    extra_args=None,
):
    output_dir = Path(output_dir)
    model_weights = Path(model_weights)
    extra_args = extra_args or []

    if not model_weights.exists() and not dry_run:
        print(f"[MISSING] Skip because model weights do not exist: {model_weights}")
        return

    if not force_rerun and test_complete(output_dir):
        print(f"[SKIP] Completed: {output_dir}")
        return

    cmd = [
        python_executable,
        str(test_script),
        "--test_h5_path",
        dataset_config["h5_path"],
        "--test_dataset_csv",
        dataset_config["csv_path"],
        "--test_data_root",
        dataset_config["data_root"],
        "--global_stats",
        global_stats,
        "--model_weights",
        str(model_weights),
        "--output_dir",
        str(output_dir),
        "--sar_input_mode",
        sar_input_mode,
        "--fusion_scales",
        fusion_scales,
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--vis_freq",
        str(vis_freq),
    ]

    cmd.append("--save_geotiff" if save_geotiff else "--no-save_geotiff")
    cmd.extend(extra_args)

    print("=" * 80)
    print(f"Dataset: {dataset_key}")
    if not model_weights.exists():
        print(f"[DRY RUN] Model weights do not exist in this environment: {model_weights}")
    print("Running:", " ".join(cmd))
    print("=" * 80)

    if dry_run:
        return

    subprocess.run(cmd, check=True)
