"""Explicit Task 07 commands; preparation/training/test work is never launched by audit."""

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

from damage_analytics.task07.config import load_config


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Use a safe new output ID")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 07 external-dataset research")
    parser.add_argument("--config", type=Path, default=Path("config/task07.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", choices=("sdnet2018", "damsegment"), required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--release", type=Path, required=True)
    validate.add_argument("--full", action="store_true")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--run-id", required=True)
    train = commands.add_parser("train")
    train.add_argument("--release", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--backbone", choices=("mobilenet_v3_small", "resnet18"), required=True)
    train.add_argument("--device", choices=("cpu", "mps", "auto"))
    train.add_argument("--epochs", type=int)
    train.add_argument("--allow-provisional", action="store_true")
    for name in ("evaluate", "final-test"):
        evaluate = commands.add_parser(name)
        evaluate.add_argument("--release", type=Path, required=True)
        evaluate.add_argument("--checkpoint", type=Path, required=True)
        evaluate.add_argument("--device", choices=("cpu", "mps"), default="cpu")
        evaluate.add_argument("--confirm-compatible-labels", action="store_true")
        evaluate.add_argument("--source-release", type=Path)
    cross = commands.add_parser("cross-check")
    cross.add_argument("--left", type=Path, required=True)
    cross.add_argument("--right", type=Path, required=True)
    cross.add_argument("--report-id", required=True)
    cross.add_argument("--rc-manifest", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("--metrics", type=Path, action="append", default=[])
    compare.add_argument("--run-id", action="append", default=[])
    compare.add_argument("--report-id", required=True)
    segmentation = commands.add_parser("evaluate-segmentation")
    segmentation.add_argument("--release", type=Path, required=True)
    segmentation.add_argument("--predictions", type=Path, required=True)
    segmentation.add_argument("--report-id", required=True)
    segmentation.add_argument("--rgb", action="store_true")
    segmentation.add_argument("--final-test", action="store_true")
    gallery = commands.add_parser("detection-gallery")
    gallery.add_argument("--release", type=Path, required=True)
    gallery.add_argument("--report-id", required=True)
    gallery.add_argument("--count", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "audit":
            from damage_analytics.task07.audit import audit_archives, audit_rc

            result = audit_archives(config)
            rc = audit_rc(config.output_dir)
            from damage_analytics.task07.reports import audit_markdown

            audit_markdown(config.output_dir / "audit_report.md", result, rc)
            print(
                json.dumps(
                    {
                        name: {task: data["images"] for task, data in info["views"].items()}
                        for name, info in result["datasets"].items()
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "prepare":
            from damage_analytics.task07.prepare import prepare

            print(prepare(config, args.dataset))
            return 0
        if args.command == "smoke":
            from damage_analytics.task07.smoke import smoke

            print(smoke(config, config.output_dir / "smoke" / safe_name(args.run_id)))
            return 0
        from damage_analytics.task07.release import open_release

        if args.command == "cross-check":
            from damage_analytics.task07.crosscheck import crosscheck

            output = config.output_dir / "cross_checks" / f"{safe_name(args.report_id)}.json"
            if output.exists():
                raise FileExistsError("Report already exists")
            result = crosscheck(
                open_release(args.left),
                open_release(args.right),
                output,
                args.rc_manifest,
                radius=config.phash_distance,
                limit=config.max_near_candidates,
            )
            print(
                {
                    "exact_pairs": len(result["exact_pairs"]),
                    "near_candidates": len(result["near_candidates"]),
                }
            )
            return 0
        if args.command == "compare":
            from damage_analytics.io import sha256_file
            from damage_analytics.task07.evaluation import compare

            paths = list(args.metrics)
            for run in args.run_id:
                folder = config.output_dir / "experiments" / safe_name(run)
                meta = json.loads((folder / "run.json").read_text())
                digest = sha256_file(folder / "best.pt")[:16]
                paths.append(
                    config.output_dir
                    / "evaluations"
                    / digest
                    / meta["dataset"]["release_id"]
                    / "validation/metrics.json"
                )
            if not paths:
                raise ValueError("Provide --metrics or --run-id")

            compare(
                paths,
                config.output_dir / "comparisons" / f"{safe_name(args.report_id)}.json",
            )
            return 0
        release = open_release(args.release, full=args.command == "validate" and args.full)
        if args.command == "validate":
            print(
                json.dumps({"identity": release.identity(), "quality": release.quality}, indent=2)
            )
        elif args.command == "train":
            from damage_analytics.task07.engine import train

            settings = config.training
            if args.device:
                settings = replace(settings, device=args.device)
            if args.epochs is not None:
                settings = replace(settings, epochs=args.epochs)
            print(
                train(
                    release,
                    settings,
                    config.output_dir,
                    args.run_id,
                    args.backbone,
                    config.seed,
                    allow_provisional=args.allow_provisional,
                )
            )
        elif args.command in ("evaluate", "final-test"):
            from damage_analytics.task07.evaluation import evaluate

            print(
                evaluate(
                    args.checkpoint,
                    release,
                    config.output_dir,
                    args.device,
                    final_test=args.command == "final-test",
                    confirm_compatible_labels=args.confirm_compatible_labels,
                    source_release=args.source_release,
                )
            )
        elif args.command == "evaluate-segmentation":
            from damage_analytics.task07.segmentation import evaluate_masks

            print(
                evaluate_masks(
                    release,
                    args.predictions,
                    config.output_dir / "segmentation_evaluations" / safe_name(args.report_id),
                    rgb=args.rgb,
                    final_test=args.final_test,
                )
            )
        elif args.command == "detection-gallery":
            from damage_analytics.task07.visualization import detection_gallery

            print(
                detection_gallery(
                    release,
                    config.output_dir / "detection_galleries" / safe_name(args.report_id),
                    args.count,
                )
            )
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(f"Task 07: {exc}", file=sys.stderr)
        return 2
    return 0
