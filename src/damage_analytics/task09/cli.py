"""Task 09 command boundaries exclude final tests and default to one CPU thread."""

import argparse
import sys
from pathlib import Path

from damage_analytics.task09.config import load_config, safe_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 09 segmentation and native damage features")
    parser.add_argument("--config", type=Path, default=Path("config/task09.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--report-id", required=True)
    audit.add_argument("--full-development", action="store_true")
    replay = commands.add_parser("replay")
    replay.add_argument("--report-id", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--report-id", required=True)
    train = commands.add_parser("train")
    train.add_argument("--model", choices=("unet", "lraspp_mobilenet_v3_large"), required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    infer = commands.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--image", type=Path, required=True)
    infer.add_argument("--report-id", required=True)
    infer.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    compare = commands.add_parser("compare")
    compare.add_argument("--run-id", action="append", required=True)
    compare.add_argument("--report-id", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "smoke":
            from damage_analytics.task09.smoke import smoke

            print(smoke(config.output_dir / "smoke" / safe_id(args.run_id)))
            return 0
        from damage_analytics.training.manifests import open_release

        release = open_release(config.training())
        if args.command == "audit":
            from damage_analytics.task09.audit import audit

            folder = config.output_dir / "audits" / safe_id(args.report_id)
            audit(config, release, folder, full_development=args.full_development)
            print(folder)
        elif args.command == "replay":
            import json

            import torch

            from damage_analytics.task09.evaluation import replay_baseline
            from damage_analytics.task09.reports import summarize

            torch.set_num_threads(config.cpu_threads)
            folder = config.output_dir / "reviews" / safe_id(args.report_id)
            folder.mkdir(parents=True, exist_ok=False)
            from damage_analytics.task09.audit import audit

            dataset_audit = audit(config, release, folder / "audit")
            results = {
                baseline.name: replay_baseline(
                    config, release, baseline, folder / "evaluations" / baseline.name
                )
                for baseline in config.baselines
            }
            summarize(folder, dataset_audit, results)
            print(
                json.dumps(
                    {
                        name: result["metrics"]["foreground_miou"]
                        for name, result in results.items()
                    }
                )
            )
            print(folder / "REPORT.md")
        elif args.command == "plan":
            from damage_analytics.task09.experiments import plan

            output = config.output_dir / "plans" / f"{safe_id(args.report_id)}.json"
            plan(config, release, output)
            print(output)
        elif args.command == "train":
            from damage_analytics.task09.experiments import train

            print(train(config, release, args.model, args.run_id, device_name=args.device))
        elif args.command == "evaluate":
            from damage_analytics.io import sha256_file
            from damage_analytics.task09.evaluation import fresh_evaluation

            output = (
                config.output_dir
                / "evaluations"
                / sha256_file(args.checkpoint)[:16]
                / "validation"
            )
            fresh_evaluation(config, release, args.checkpoint, output, args.device)
            print(output)
        elif args.command == "infer":
            from damage_analytics.task09.pipeline import predict_file

            output = config.output_dir / "inference" / safe_id(args.report_id)
            predict_file(config, release, args.checkpoint, args.image, output, device=args.device)
            print(output / "damage_features.json")
        elif args.command == "compare":
            from damage_analytics.task09.experiments import compare_runs

            output = config.output_dir / "comparisons" / f"{safe_id(args.report_id)}.json"
            compare_runs(config, release, args.run_id, output)
            print(output)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(f"Task 09: {exc}", file=sys.stderr)
        return 2
    return 0
