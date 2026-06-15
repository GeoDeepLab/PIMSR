import argparse
from pathlib import Path


def create_output_dirs(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    return output_dir


def print_config(config):
    path_keys = [
        "train_h5_path",
        "train_csv_path",
        "val_h5_path",
        "val_csv_path",
        "global_stats",
        "simulator_weights",
        "output_dir",
    ]
    training_keys = [
        "batch_size",
        "num_epochs",
        "learning_rate",
        "num_workers",
        "lr_step_size",
        "lr_gamma",
        "recon_weight",
        "sar_weight",
    ]
    optional_keys = [
        "sar_input_mode",
        "fusion_scales",
        "experiment_name",
    ]

    print("=" * 60 + "\nTRAINING CONFIGURATION\n" + "=" * 60)
    print("\n[Data Paths]")
    for key in path_keys:
        print(f"  {key}: {getattr(config, key)}")

    print("\n[Training Parameters]")
    for key in training_keys:
        print(f"  {key}: {getattr(config, key)}")

    print(
        "\n[Other Settings]\n"
        f"  device: {config.device}\n"
        f"  model_name: {config.model_name}\n"
        f"  save_freq: {config.save_freq}\n"
        f"  vis_freq: {config.vis_freq}\n"
        f"  resume: {config.resume if config.resume else 'None'}\n"
        f"  random_seed: {config.random_seed}"
    )
    for key in optional_keys:
        if hasattr(config, key):
            print(f"  {key}: {getattr(config, key)}")
    print("=" * 60)


def build_base_parser():
    parser = argparse.ArgumentParser(description="SAR Enhanced DEM Super Resolution Training")
    parser.add_argument(
        "--train_h5_path",
        type=str,
        default=r"Train_data/train_data.h5",
        help="Path to the training HDF5 file with real LR data",
    )
    parser.add_argument(
        "--train_csv_path",
        type=str,
        default=r"Train_data/data_index.csv",
        help="Path to the training index CSV file",
    )
    parser.add_argument(
        "--val_h5_path",
        type=str,
        default=r"Val_data/val_data.h5",
        help="Path to the validation HDF5 file with real LR data",
    )
    parser.add_argument(
        "--val_csv_path",
        type=str,
        default=r"Val_data/data_index.csv",
        help="Path to the validation index CSV file",
    )
    parser.add_argument(
        "--global_stats",
        type=str,
        default=r"Train_data/global_stats.json",
        help="Global statistics JSON file",
    )
    parser.add_argument(
        "--simulator_weights",
        type=str,
        default=r"best_precise_simulator.pth",
        help="SAR simulator pretrained weights",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"output_dir",
        help="Output directory",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--lr_step_size", type=int, default=25, help="Learning rate step size")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="Learning rate gamma")
    parser.add_argument("--recon_weight", type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument("--sar_weight", type=float, default=0.2, help="SAR loss weight")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--save_freq", type=int, default=1, help="Save frequency")
    parser.add_argument("--vis_freq", type=int, default=1, help="Visualization frequency")
    parser.add_argument("--random_seed", type=int, default=1008, help="Random seed for reproducibility")
    parser.add_argument("--model_name", type=str, default="ResidualSARDEMGenerator", help="Model name")
    parser.add_argument(
        "--sar_input_mode",
        type=str,
        choices=["none", "vv", "vh", "both"],
        default="both",
        help="SAR input mode: none, vv, vh, or both.",
    )
    parser.add_argument(
        "--fusion_scales",
        type=str,
        default="16,32,64",
        help="DEM-SAR fusion scales, e.g. 16, 32, 64, 16,32, 16,64, 32,64, 16,32,64.",
    )
    return parser


def get_config_from_args():
    """Build the training configuration from command-line arguments."""
    parser = build_base_parser()

    config = parser.parse_args()

    if config.device == "auto":
        import torch

        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    requested_scales = [scale.strip() for scale in config.fusion_scales.split(",") if scale.strip()]
    config.fusion_scales = ",".join(requested_scales)

    valid_fusion_scales = {"16", "32", "64", "16,32", "16,64", "32,64", "16,32,64"}
    if config.fusion_scales not in valid_fusion_scales:
        parser.error("--fusion_scales must be one of: 16, 32, 64, 16,32, 16,64, 32,64, 16,32,64")

    return config
