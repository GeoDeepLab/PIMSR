import argparse
import sys
from pathlib import Path

from batch_resume_utils import run_experiment_with_resume


FUSION_SCALE_EXPERIMENTS = {
    "only_16": "16",
    "only_32": "32",
    "only_64": "64",
    "16_32": "16,32",
    "16_64": "16,64",
    "32_64": "32,64",
}


def main():
    parser = argparse.ArgumentParser(description="Batch run DEM-SAR scale fusion ablations with auto resume.")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(Path("results") / "structure_ablation" / "scale_fusion"),
        help="Root directory for scale fusion ablation outputs.",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Run from scratch even if logs/checkpoints already exist.",
    )
    args, passthrough_args = parser.parse_known_args()

    output_root = Path(args.output_root)

    for name, scales in FUSION_SCALE_EXPERIMENTS.items():
        output_dir = output_root / f"fusion_{name}"
        run_experiment_with_resume(
            python_executable=args.python,
            train_script="train.py",
            fixed_args=["--fusion_scales", scales],
            output_dir=output_dir,
            passthrough_args=passthrough_args,
            dry_run=args.dry_run,
            force_rerun=args.force_rerun,
        )


if __name__ == "__main__":
    main()
