import argparse
import sys
from pathlib import Path

from batch_resume_utils import run_experiment_with_resume


SAR_WEIGHTS = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]


def weight_name(weight):
    return str(weight).replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description="Batch run SAR loss weight experiments with auto resume.")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(Path("results") / "sar_weight_experiments"),
        help="Root directory for all SAR weight experiment outputs.",
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

    for weight in SAR_WEIGHTS:
        output_dir = output_root / f"sar_weight_{weight_name(weight)}"
        run_experiment_with_resume(
            python_executable=args.python,
            train_script="train.py",
            fixed_args=["--sar_weight", str(weight)],
            output_dir=output_dir,
            passthrough_args=passthrough_args,
            dry_run=args.dry_run,
            force_rerun=args.force_rerun,
        )


if __name__ == "__main__":
    main()
