import argparse
import sys
from pathlib import Path

from batch_test_utils import (
    DEFAULT_GLOBAL_STATS,
    TEST_DATASETS,
    parse_dataset_keys,
    run_test_case,
)
from run_sar_weight_experiments import SAR_WEIGHTS, weight_name


def parse_weight_value(weight_text):
    weight = float(weight_text)
    return int(weight) if weight.is_integer() else weight


def main():
    parser = argparse.ArgumentParser(description="Batch test SAR loss weight experiments.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=f"Dataset key(s) to test: {', '.join(TEST_DATASETS)}, or all. Use comma-separated values.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="all",
        help="SAR weights to test, e.g. all or 0,0.01,0.05,0.1,0.2.",
    )
    parser.add_argument(
        "--weight_root",
        type=str,
        default=str(Path("results") / "sar_weight_experiments"),
        help="Root directory of trained SAR weight experiments.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(Path("results") / "test"),
        help="Root directory for test outputs.",
    )
    parser.add_argument("--global_stats", type=str, default=DEFAULT_GLOBAL_STATS)
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument("--test_script", type=str, default="test.py", help="Test script path.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--vis_freq", type=int, default=10)
    parser.add_argument("--no-save_geotiff", dest="save_geotiff", action="store_false")
    parser.set_defaults(save_geotiff=True)
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--force_rerun", action="store_true", help="Run even if test_summary.txt exists.")
    args, passthrough_args = parser.parse_known_args()

    try:
        dataset_keys = parse_dataset_keys(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))

    if args.weights == "all":
        weights = SAR_WEIGHTS
    else:
        weights = [parse_weight_value(weight.strip()) for weight in args.weights.split(",") if weight.strip()]

    weight_root = Path(args.weight_root)
    output_root = Path(args.output_root)

    for dataset_key in dataset_keys:
        dataset_config = TEST_DATASETS[dataset_key]

        for weight in weights:
            experiment_name = f"sar_weight_{weight_name(weight)}"
            experiment_dir = weight_root / experiment_name
            run_test_case(
                python_executable=args.python,
                test_script=args.test_script,
                dataset_key=dataset_key,
                dataset_config=dataset_config,
                global_stats=args.global_stats,
                model_weights=experiment_dir / "checkpoints" / "best_model.pth",
                output_dir=output_root / dataset_key / "sar_weight_experiments" / experiment_name,
                sar_input_mode="both",
                fusion_scales="16,32,64",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                vis_freq=args.vis_freq,
                save_geotiff=args.save_geotiff,
                dry_run=args.dry_run,
                force_rerun=args.force_rerun,
                extra_args=passthrough_args,
            )


if __name__ == "__main__":
    main()
