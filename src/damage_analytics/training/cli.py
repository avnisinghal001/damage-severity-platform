"""Explicit Task 03 commands; test evaluation has its own deliberately named entrypoint."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from damage_analytics.io import write_json
from damage_analytics.training.config import load_training_config
from damage_analytics.training.manifests import open_release, read_pair


def main(argv: list[str] | None = None) -> int:
    """Run available integrity/audit checks even when PyTorch is not installed."""
    parser = argparse.ArgumentParser(description="Task 03: pinned 9-class U-Net segmentation")
    parser.add_argument("--config", type=Path, default=Path("config/task03.toml"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"))
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("verify-data", "audit-data", "inspect-loader", "smoke"):
        commands.add_parser(command)
    for command in ("train", "overfit"):
        commands.add_parser(command).add_argument("--run-id", required=True)
    for command in ("validate", "final-test", "predict"):
        sub = commands.add_parser(command)
        sub.add_argument("--checkpoint", required=True, type=Path)
        if command == "predict":
            sub.add_argument("--image", required=True, type=Path)
            sub.add_argument("--prediction-id", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_training_config(args.config)
        if args.device:
            config = replace(config, device=args.device)
        if args.command == "verify-data":
            release = open_release(config, full_integrity=True)
            counts = {}
            for split in ("train", "validation"):
                for sample in release.samples(split):
                    read_pair(release, sample)
                counts[split] = len(release.samples(split))
            report = {
                "passed": True,
                "dataset": release.identity(),
                "pairs_decoded": counts,
                "test_access": "Sealed artifact byte hashes/partition metadata only; no test "
                "image decoding, transforms, prediction, or metrics",
                "source": "Task 02 conditional research-use limitations remain applicable",
            }
            write_json(config.output_dir / "data_verification.json", report)
            print(json.dumps(report, indent=2))
        elif args.command == "audit-data":
            from damage_analytics.training.audit import audit_resizing

            report = audit_resizing(config, open_release(config))
            print(json.dumps(report, indent=2))
        elif args.command == "inspect-loader":
            from damage_analytics.training.dataset import SegmentationDataset, make_loader

            release = open_release(config)
            dataset = SegmentationDataset(
                release, release.samples("train"), config.data, training=True, seed=config.seed
            )
            batch = next(
                iter(
                    make_loader(
                        dataset,
                        config.optimization.batch_size,
                        config.num_workers,
                        config.seed,
                        shuffle=False,
                    )
                )
            )
            report = {
                "image_shape": list(batch["images"].shape),
                "mask_shape": list(batch["masks"].shape),
                "image_dtype": str(batch["images"].dtype),
                "mask_dtype": str(batch["masks"].dtype),
                "mask_values": batch["masks"].unique().tolist(),
                "sample_ids": [sample.sample_id for sample in batch["samples"]],
                "dataset": release.identity(),
                "seed": config.seed,
                "config": config.snapshot(),
                "test_accessed": False,
            }
            write_json(config.output_dir / "loader_inspection.json", report)
            print(json.dumps(report, indent=2))
        elif args.command == "smoke":
            from damage_analytics.training.smoke import synthetic_smoke

            print(json.dumps(synthetic_smoke(config.output_dir / "synthetic_smoke"), indent=2))
        elif args.command in ("train", "overfit"):
            from damage_analytics.training.engine import train_model

            folder = train_model(config, args.run_id, overfit=args.command == "overfit")
            print(f"Experiment: {folder}")
            if args.command == "overfit":
                return (
                    0
                    if json.loads((folder / "status.json").read_text())["overfit_target_met"]
                    else 2
                )
        elif args.command == "predict":
            from damage_analytics.training.inference import predict_image

            print(predict_image(config, args.checkpoint, args.image, args.prediction_id))
        else:
            from damage_analytics.training.evaluation import evaluate_checkpoint

            print(
                evaluate_checkpoint(
                    config, args.checkpoint, final_test=args.command == "final-test"
                )
            )
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency: {exc.name}. Installation was not attempted. "
            "See README Task 03 setup commands.",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        print(f"Task 03 error: {exc}", file=sys.stderr)
        return 2
    return 0
