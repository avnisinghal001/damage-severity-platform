"""Validate explicit external checkpoint pins and create a private Compose environment file."""

import argparse
import os
import secrets
import shlex
import tomllib
from pathlib import Path

from damage_analytics.serving.classifiers import check_pin


def configure(config: Path, output: Path) -> None:
    values = tomllib.loads(config.read_text())
    environment = {
        "POSTGRES_USER": "damage",
        "POSTGRES_DB": "damage",
        "POSTGRES_PASSWORD": secrets.token_hex(24),
    }
    for role in ("segmentation", "crack", "severity"):
        row = values[role]
        path = (config.parent / row["path"]).resolve()
        check_pin(path, row["sha256"])
        environment[f"{role.upper()}_CHECKPOINT_HOST"] = str(path)
        environment[f"{role.upper()}_CHECKPOINT_SHA256"] = row["sha256"]
    # O_EXCL prevents accidentally changing credentials for an existing database volume.
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        stream.write("# Generated local Task 12 environment; do not commit.\n")
        stream.write(
            "".join(f"{key}={shlex.quote(value)}\n" for key, value in environment.items())
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/task12.toml"))
    parser.add_argument("--output", type=Path, default=Path(".env.task12"))
    args = parser.parse_args()
    configure(args.config, args.output)
    print(f"Wrote {args.output}; all three external checkpoints matched their pins")


if __name__ == "__main__":
    main()
