"""Task 04 commands: validation-only audit, manual experiments and image-space analytics."""

import argparse
import json
import sys
from pathlib import Path

from damage_analytics.analytics.config import load_analytics_config


def main(argv: list[str] | None = None) -> int:
    """No command starts training except an explicitly invoked train subcommand."""
    parser = argparse.ArgumentParser(
        description="Task 04: controlled segmentation and scoped damage analytics"
    )
    parser.add_argument("--config", type=Path, default=Path("config/task04.toml"))
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("audit-baseline", "plan", "supervision", "review", "resolution-audit"):
        subs.add_parser(name)
    train = subs.add_parser("train")
    train.add_argument("--experiment", required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--exploratory-epochs", type=int)
    train.add_argument("--device", choices=("cpu", "mps"))
    train.add_argument("--nondeterministic", action="store_true")
    train.add_argument("--workers", type=int)
    train.add_argument("--persistent-workers", action="store_true")
    profile = subs.add_parser("profile")
    profile.add_argument("--report-id", required=True)
    profile.add_argument("--experiment", default="control")
    profile.add_argument("--suite", choices=("stages", "loaders", "batches"), default="stages")
    profile.add_argument("--samples", type=int, default=64)
    profile.add_argument("--repeats", type=int, default=3)
    profile.add_argument("--warmup-batches", type=int, default=2)
    profile.add_argument("--workers", type=int, choices=(0, 2, 4), default=0)
    profile.add_argument("--persistent-workers", action="store_true")
    profile.add_argument("--cache-mb", type=int, default=0)
    profile.add_argument("--batch-size", type=int)
    profile.add_argument("--device", choices=("cpu", "mps"))
    profile.add_argument("--cpu-threads", type=int)
    profile.add_argument("--channels-last", action="store_true")
    profile.add_argument("--detect-anomaly", action="store_true")
    profile.add_argument("--nondeterministic", action="store_true")
    summary = subs.add_parser("profile-summary")
    summary.add_argument("--profile-id", action="append", required=True)
    summary.add_argument("--report-id", required=True)
    evaluate = subs.add_parser("evaluate")
    evaluate.add_argument("--run-id", required=True)
    comparison = subs.add_parser("compare")
    comparison.add_argument("--run-id", action="append", required=True)
    comparison.add_argument("--comparison-id", required=True)
    predict = subs.add_parser("predict")
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--image", required=True, type=Path)
    predict.add_argument("--report-id", required=True)
    mask = subs.add_parser("analyze-mask")
    mask.add_argument("--mask", required=True, type=Path)
    mask.add_argument("--image", type=Path)
    mask.add_argument("--report-id", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_analytics_config(args.config)
        if args.command == "audit-baseline":
            from damage_analytics.analytics.baseline import audit_baseline

            folder, report = audit_baseline(config)
            print(
                json.dumps(
                    {
                        "output": str(folder),
                        "status": report["status"],
                        "fresh_foreground_miou": report["fresh_foreground_miou"],
                        "delta": report["foreground_miou_delta"],
                    },
                    indent=2,
                )
            )
            return 0 if report["status"] == "verified" else 2
        if args.command == "plan":
            from damage_analytics.analytics.experiments import plan_experiments

            plan = plan_experiments(config)
            print(
                json.dumps(
                    {
                        "experiments": [row["name"] for row in plan["experiments"]],
                        "training_launched": False,
                    },
                    indent=2,
                )
            )
        elif args.command == "resolution-audit":
            from damage_analytics.analytics.experiments import audit_resolutions

            report = audit_resolutions(config)
            print(
                json.dumps(
                    {"resolutions": list(report["resolutions"]), "training_launched": False},
                    indent=2,
                )
            )
        elif args.command == "supervision":
            from damage_analytics.analytics.supervision import assess_supervision

            report = assess_supervision(config)
            print(json.dumps(report, indent=2))
        elif args.command == "review":
            from damage_analytics.analytics.review import targeted_review

            print(targeted_review(config))
        elif args.command == "profile-summary":
            from damage_analytics.training.profile_reports import summarize_profiles

            print(summarize_profiles(config.output_dir, args.profile_id, args.report_id))
        elif args.command == "profile":
            from dataclasses import replace

            from damage_analytics.analytics.experiments import experiment_config
            from damage_analytics.training.profiling import profile_destination, profile_pipeline

            effective = experiment_config(config, args.experiment)
            if args.detect_anomaly:
                import torch

                torch.autograd.set_detect_anomaly(True)
            if args.device:
                effective = replace(effective, device=args.device)
            if args.nondeterministic:
                effective = replace(effective, deterministic=False)
            if args.channels_last:
                effective = replace(
                    effective, performance=replace(effective.performance, channels_last=True)
                )
            destination = profile_destination(config.output_dir, args.report_id)
            profile_existed = destination.exists()
            print(
                profile_pipeline(
                    effective,
                    destination,
                    cpu_threads=config.cpu_threads
                    if args.cpu_threads is None
                    else args.cpu_threads,
                    suite=args.suite,
                    samples=args.samples,
                    repeats=args.repeats,
                    warmup_batches=args.warmup_batches,
                    workers=args.workers,
                    persistent=args.persistent_workers,
                    cache_mb=args.cache_mb,
                    batch_size=args.batch_size,
                )
            )
        elif args.command == "train":
            from dataclasses import replace

            from damage_analytics.analytics.experiments import run_experiment

            if args.device:
                config = replace(config, device=args.device)
            if args.nondeterministic:
                config = replace(config, deterministic=False)
            if args.workers is not None:
                config = replace(config, num_workers=args.workers)
            if args.persistent_workers:
                config = replace(
                    config, performance=replace(config.performance, persistent_workers=True)
                )
            print(
                run_experiment(
                    config,
                    args.experiment,
                    args.run_id,
                    exploratory_epochs=args.exploratory_epochs,
                )
            )
        elif args.command == "evaluate":
            from damage_analytics.analytics.comparison import evaluate_experiment

            print(evaluate_experiment(config, args.run_id)[0])
        elif args.command == "compare":
            from damage_analytics.analytics.comparison import compare_experiments

            print(compare_experiments(config, args.run_id, args.comparison_id))
        elif args.command == "predict":
            from damage_analytics.analytics.inference import structured_predict

            print(structured_predict(config, args.checkpoint, args.image, args.report_id))
        else:
            from damage_analytics.analytics.inference import analyze_mask_file

            print(analyze_mask_file(config, args.mask, args.report_id, args.image))
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}; installation was not attempted.", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError, TypeError, KeyError) as exc:
        if args.command == "profile":
            import traceback

            traceback.print_exc()
            if "config" in locals() and not locals().get("profile_existed", True):
                from damage_analytics.io import write_json
                from damage_analytics.training.profiling import profile_destination

                failed = profile_destination(config.output_dir, args.report_id)
                if failed.exists() and not (failed / "failure.json").exists():
                    write_json(failed / "failure.json", {"error": str(exc)})
        print(f"Task 04 error: {exc}", file=sys.stderr)
        return 2
    return 0
