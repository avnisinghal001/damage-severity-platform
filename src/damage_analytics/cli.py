"""Small command-line entrypoint; all processing lives in reusable modules."""

import argparse
import sys
from pathlib import Path
from zipfile import BadZipFile

from damage_analytics.archives import extract_archives, inspect_archives
from damage_analytics.config import load_config


def main(argv: list[str] | None = None) -> int:
    """Run one Task 01 stage; return 2 for invalid inputs or failed validation."""
    parser = argparse.ArgumentParser(description="RC1841 Task 01 dataset pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/task01.toml"))
    parser.add_argument("command", choices=("inspect", "ingest", "analyze", "visualize"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
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
            print(f"Validation {validation['status']}: {len(records)} valid pairs, "
                  f"{validation['errors']} errors, {validation['warnings']} warnings")
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
        print(f"Missing dependency: {exc.name}. Follow README installation instructions.",
              file=sys.stderr)
        return 2
    return 0
