"""Explicit preparation, verification and validation-only training commands for Task 10."""

import argparse
import json
import sys
from pathlib import Path

from damage_analytics.io import write_json
from damage_analytics.task10.config import load_config, safe_id


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path("config/task10.toml"))
    commands = result.add_subparsers(dest="command", required=True)
    inspection = commands.add_parser(
        "inspect", help="Read archive schema, label registration and headers"
    )
    inspection.add_argument("--report-id", default="archive_inspection")
    commands.add_parser(
        "prepare", help="CRC/readability audit, grouping, quarantine and frozen manifests"
    )
    validate = commands.add_parser("validate", help="Authenticate a prepared dataset release")
    validate.add_argument("--release", type=Path, required=True)
    validate.add_argument(
        "--full", action="store_true", help="Also hash all arrays and original rows"
    )
    review = commands.add_parser("review", help="Export inventories and train-only contact sheets")
    review.add_argument("--release", type=Path, required=True)
    review.add_argument("--report-id", required=True)
    review.add_argument("--limit", type=int, default=32)
    smoke = commands.add_parser(
        "smoke", help="Two-epoch tiny synthetic CPU test; no pretrained weights"
    )
    smoke.add_argument("--run-id", required=True)
    training = commands.add_parser(
        "train", help="Explicit ResNet18 transfer learning; no test evaluation"
    )
    training.add_argument("--release", type=Path, required=True)
    training.add_argument("--run-id", required=True)
    training.add_argument(
        "--allow-provisional",
        action="store_true",
        help="Acknowledge inferred color encoding and incomplete source identities",
    )
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate a pinned checkpoint on validation ONLY"
    )
    evaluation.add_argument("--release", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    for command in (training, evaluation):
        command.add_argument("--device", choices=("cpu", "mps", "auto"), default="auto")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
        if arguments.command == "inspect":
            from damage_analytics.task10.archives import inspect

            output = config.output_dir / "inspections" / f"{safe_id(arguments.report_id)}.json"
            if output.exists():
                raise FileExistsError("Report ID already exists")
            write_json(output, inspect(config))
        elif arguments.command == "prepare":
            from damage_analytics.task10.prepare import prepare

            output = prepare(config)
        elif arguments.command == "smoke":
            from damage_analytics.task10.smoke import run_smoke

            output = run_smoke(config, arguments.run_id)
        else:
            from damage_analytics.task10.release import open_release

            release = open_release(arguments.release, full=getattr(arguments, "full", False))
            if arguments.command == "validate":
                print(
                    json.dumps(
                        {
                            "status": "verified",
                            "dataset": release.identity(),
                            "quality": release.quality,
                        },
                        indent=2,
                    )
                )
                return 0
            if arguments.command == "review":
                from damage_analytics.task10.review import build_review

                output = build_review(
                    release, config.output_dir, arguments.report_id, arguments.limit
                )
                print(output)
                return 0
            from damage_analytics.task10.engine import evaluate, train

            if arguments.command == "train":
                output = train(
                    config,
                    release,
                    arguments.run_id,
                    device_name=arguments.device,
                    allow_provisional=arguments.allow_provisional,
                )
            else:
                output = evaluate(
                    config, release, arguments.checkpoint, device_name=arguments.device
                )
        print(output)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Task 10 blocked: {exc}", file=sys.stderr)
        return 2
