import argparse
import sys
from pathlib import Path

from batch_test_utils import (
    DEFAULT_GLOBAL_STATS,
    TEST_DATASETS,
    parse_dataset_keys,
    run_test_case,
)


# sar_input_both is intentionally skipped here because it is the full model.
SAR_OUTPUT_TESTS = {
    "none": {
        "experiment_dir": "sar_input_none",
        "sar_input_mode": "none",
        "fusion_scales": "16,32,64",
    },
    "vv": {
        "experiment_dir": "sar_input_vv",
        "sar_input_mode": "vv",
        "fusion_scales": "16,32,64",
    },
    "vh": {
        "experiment_dir": "sar_input_vh",
        "sar_input_mode": "vh",
        "fusion_scales": "16,32,64",
    },
}

FUSION_SCALE_TESTS = {
    "only_16": "16",
    "only_32": "32",
    "only_64": "64",
    "16_32": "16,32",
    "16_64": "16,64",
    "32_64": "32,64",
}


def main():
    parser = argparse.ArgumentParser(description="Batch test SAR output and scale-fusion ablations.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=f"Dataset key(s) to test: {', '.join(TEST_DATASETS)}, or all. Use comma-separated values.",
    )
    parser.add_argument(
        "--sar_output_root",
        type=str,
        default=str(Path("results") / "structure_ablation" / "sar_output"),
        help="Root directory of trained SAR output ablation experiments.",
    )
    parser.add_argument(
        "--scale_fusion_root",
        type=str,
        default=str(Path("results") / "structure_ablation" / "scale_fusion"),
        help="Root directory of trained scale-fusion ablation experiments.",
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
    output_root = Path(args.output_root)
    sar_output_root = Path(args.sar_output_root)
    scale_fusion_root = Path(args.scale_fusion_root)

    for dataset_key in dataset_keys:
        dataset_config = TEST_DATASETS[dataset_key]

        for name, spec in SAR_OUTPUT_TESTS.items():
            experiment_dir = sar_output_root / spec["experiment_dir"]
            run_test_case(
                python_executable=args.python,
                test_script=args.test_script,
                dataset_key=dataset_key,
                dataset_config=dataset_config,
                global_stats=args.global_stats,
                model_weights=experiment_dir / "checkpoints" / "best_model.pth",
                output_dir=output_root / dataset_key / "structure_ablation" / "sar_output" / spec["experiment_dir"],
                sar_input_mode=spec["sar_input_mode"],
                fusion_scales=spec["fusion_scales"],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                vis_freq=args.vis_freq,
                save_geotiff=args.save_geotiff,
                dry_run=args.dry_run,
                force_rerun=args.force_rerun,
                extra_args=passthrough_args,
            )

        for name, scales in FUSION_SCALE_TESTS.items():
            experiment_dir = scale_fusion_root / f"fusion_{name}"
            run_test_case(
                python_executable=args.python,
                test_script=args.test_script,
                dataset_key=dataset_key,
                dataset_config=dataset_config,
                global_stats=args.global_stats,
                model_weights=experiment_dir / "checkpoints" / "best_model.pth",
                output_dir=output_root / dataset_key / "structure_ablation" / "scale_fusion" / f"fusion_{name}",
                sar_input_mode="both",
                fusion_scales=scales,
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
