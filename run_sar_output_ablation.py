import argparse
import sys
from pathlib import Path

from batch_resume_utils import run_experiment_with_resume


SAR_INPUT_MODES = ["none", "vv", "vh", "both"]


def main():
    parser = argparse.ArgumentParser(description="Batch run SAR input/output ablations with auto resume.")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(Path("results") / "structure_ablation" / "sar_output"),
        help="Root directory for SAR output ablation outputs.",
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

    for mode in SAR_INPUT_MODES:
        output_dir = output_root / f"sar_input_{mode}"
        run_experiment_with_resume(
            python_executable=args.python,
            train_script="train.py",
            fixed_args=["--sar_input_mode", mode],
            output_dir=output_dir,
            passthrough_args=passthrough_args,
            dry_run=args.dry_run,
            force_rerun=args.force_rerun,
        )


if __name__ == "__main__":
    main()
