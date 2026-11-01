"""Explicit Task 08 operations: no final-test command or automatic real training."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from damage_analytics.task07.cli import safe_name
from damage_analytics.task08.config import load_config, load_release, load_releases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 08 crack classification reliability")
    parser.add_argument("--config", type=Path, default=Path("config/task08.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    integrity = commands.add_parser("integrity")
    integrity.add_argument("--report-id", required=True)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--report-id", required=True)
    analysis.add_argument("--run-dir", type=Path, action="append")
    analysis.add_argument("--include-external", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("--run-id", required=True)
    train.add_argument("--backbone", choices=("mobilenet_v3_small", "resnet18"), required=True)
    train.add_argument("--device", choices=("cpu", "mps", "auto"))
    train.add_argument("--seed", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--augmentation", choices=("baseline", "orientation"))
    train.add_argument("--allow-provisional", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--dataset", choices=("sdnet2018", "damsegment"), required=True)
    evaluate.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    evaluate.add_argument("--confirm-compatible-labels", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("--run-id", action="append", required=True)
    compare.add_argument("--report-id", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "integrity":
            from damage_analytics.task08.integrity import audit

            folder = config.output_dir / "integrity" / safe_name(args.report_id)
            audit(config, load_releases(config), folder)
            print(folder / "integrity_report.md")
        elif args.command == "analyze":
            from damage_analytics.task08.analysis import analyze

            folder = config.output_dir / "analysis" / safe_name(args.report_id)
            releases = {"sdnet2018": load_release(config, "sdnet2018")}
            if args.include_external:
                releases["damsegment"] = load_release(config, "damsegment")
            analyze(
                config,
                releases,
                folder,
                tuple(args.run_dir) if args.run_dir else None,
                include_external=args.include_external,
            )
            print(folder / "failure_analysis.md")
        elif args.command == "train":
            from damage_analytics.task07.engine import train

            settings = config.training
            for field in ("device", "epochs", "augmentation"):
                if getattr(args, field) is not None:
                    settings = replace(settings, **{field: getattr(args, field)})
            seed = config.seed if args.seed is None else args.seed
            if seed < 0:
                raise ValueError("Seed must be nonnegative")
            print(
                train(
                    load_release(config, "sdnet2018"),
                    settings,
                    config.output_dir,
                    safe_name(args.run_id),
                    args.backbone,
                    seed,
                    allow_provisional=args.allow_provisional,
                )
            )
        elif args.command == "evaluate":
            from damage_analytics.task07.evaluation import evaluate

            print(
                evaluate(
                    args.checkpoint,
                    load_release(config, args.dataset),
                    config.output_dir,
                    args.device,
                    confirm_compatible_labels=args.confirm_compatible_labels,
                    source_release=config.sdnet_release,
                )
            )
        elif args.command == "compare":
            from damage_analytics.task08.comparison import compare_runs

            runs = [config.output_dir / "experiments" / safe_name(name) for name in args.run_id]
            output = config.output_dir / "comparisons" / f"{safe_name(args.report_id)}.json"
            compare_runs(runs, load_release(config, "sdnet2018"), output)
            print(output)
        elif args.command == "smoke":
            from damage_analytics.task07.config import load_config as load_task07
            from damage_analytics.task07.smoke import smoke

            base = replace(
                load_task07(Path("config/task07.toml")), seed=config.seed, training=config.training
            )
            print(smoke(base, config.output_dir / "smoke" / safe_name(args.run_id)))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(f"Task 08: {exc}", file=sys.stderr)
        return 2
    return 0
