"""Small command-line entrypoint; all processing lives in reusable modules."""

import argparse
import sys
from pathlib import Path
from zipfile import BadZipFile

from damage_analytics.archives import extract_archives, inspect_archives
from damage_analytics.config import load_config


def main(argv: list[str] | None = None) -> int:
    """Run a dataset stage; return 2 for invalid inputs or failed verification."""
    parser = argparse.ArgumentParser(description="RC1841 dataset audit, curation, and splitting")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "command",
        choices=(
            "inspect",
            "ingest",
            "analyze",
            "visualize",
            "curate",
            "verify-curation",
            "curation-report",
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.command in ("curate", "verify-curation", "curation-report"):
            return _task02(args.command, args.config or Path("config/task02.toml"))
        config = load_config(args.config or Path("config/task01.toml"))
        if args.command == "inspect":
            summary = inspect_archives(config)
            print(f"Archive inventory written to {config.output_dir}")
            for role, details in summary.items():
                print(f"  {role}: {details['files']} files")
        elif args.command == "ingest":
            extract_archives(config)
            print(f"Extracted safely to {config.extracted_dir}")
        elif args.command == "analyze":
            from damage_analytics.analysis import analyze_dataset
            from damage_analytics.validation import validate_dataset

            records, validation = validate_dataset(config)
            analyze_dataset(config, records, validation)
            print(
                f"Validation {validation['status']}: {len(records)} valid pairs, "
                f"{validation['errors']} errors, {validation['warnings']} warnings"
            )
            print(f"Reports and masks: {config.output_dir}")
            return 2 if validation["errors"] else 0
        else:
            from damage_analytics.visualization import create_gallery

            samples = create_gallery(config)
            print(f"Gallery: {config.output_dir / 'sample_gallery.png'} ({len(samples)} samples)")
    except (OSError, ValueError, TypeError, BadZipFile, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency: {exc.name}. Follow README installation instructions.",
            file=sys.stderr,
        )
        return 2
    return 0


def _task02(command: str, path: Path) -> int:
    """Dispatch Task 02 without importing its dependencies for Task 01 archive commands."""
    import json

    from damage_analytics.curation.config import load_curation_config
    from damage_analytics.curation.pipeline import latest_release, run_curation, verify_latest

    config = load_curation_config(path)
    if command == "curate":
        release, report = run_curation(config)
        print(f"Release: {release}")
        print(
            f"Retained={report['valid_samples']}; excluded={report['excluded_samples']}; "
            f"verdict={report['ml_readiness']['verdict']}"
        )
        return 0 if report["ml_readiness"]["technical_checks_passed"] else 2
    if command == "verify-curation":
        result = verify_latest(config)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    print((latest_release(config) / "quality_report.json").read_text())
    return 0
