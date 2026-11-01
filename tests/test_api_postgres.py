"""Opt-in live PostgreSQL test against an ALREADY migrated disposable test database."""

import os

import pytest

pytest.importorskip("psycopg", reason="PostgreSQL driver is not installed")
pytestmark = pytest.mark.skipif(
    not os.getenv("DAMAGE_TEST_DATABASE_URL"),
    reason="Set a dedicated migrated test database explicitly",
)


def test_postgres_save_fetch_and_foreign_key_cleanup(tmp_path):
    from test_api_core import image_bytes, make_service

    from damage_analytics.api.config import Settings
    from damage_analytics.api.repository import PostgresRepository

    settings = Settings(
        os.environ["DAMAGE_TEST_DATABASE_URL"],
        tmp_path / "unused.pt",
        "0" * 64,
        storage_dir=tmp_path / "files",
    )
    repository = PostgresRepository(settings.database_url)
    assert repository.ready(), "Apply migration 001 manually to the dedicated test database"
    record = make_service(settings).predict(image_bytes(), "image/png")
    try:
        repository.save(record)
        assert repository.get(record.id) == record
    finally:
        with repository._connection() as connection:
            connection.execute("DELETE FROM inspections WHERE id = %s", (record.id,))
            assert (
                connection.execute(
                    "SELECT inspection_id FROM predictions WHERE inspection_id = %s", (record.id,)
                ).fetchone()
                is None
            )
