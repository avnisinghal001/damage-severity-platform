"""Explicit versioned migration command; application startup never mutates the schema."""

import os
from pathlib import Path

import psycopg


def migrate(database_url: str, directory: Path = Path("migrations")) -> list[int]:
    applied = []
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("SELECT pg_advisory_lock(120012)")
        exists = connection.execute("SELECT to_regclass('damage_schema_migrations')").fetchone()[0]
        versions = (
            [
                row[0]
                for row in connection.execute(
                    "SELECT version FROM damage_schema_migrations ORDER BY version"
                ).fetchall()
            ]
            if exists
            else []
        )
        if versions not in ([], [1], [1, 2]):
            raise ValueError("Incompatible migration versions; refusing to change database schema")
        for version, filename in ((1, "001_initial.sql"), (2, "002_integrated_predictions.sql")):
            if version not in versions:
                connection.execute((directory / filename).read_text(), prepare=False)
                applied.append(version)
    return applied


if __name__ == "__main__":
    print({"applied": migrate(os.environ["DAMAGE_API_DATABASE_URL"])})
