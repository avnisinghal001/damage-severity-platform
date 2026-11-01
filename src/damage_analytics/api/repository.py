"""Minimal transactional PostgreSQL adapter; connections are scoped to each operation."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class Inspection:
    """Persistent metadata only; all binary image/mask content lives in the artifact store."""

    id: UUID
    created_at: datetime
    upload_sha256: str
    image_key: str
    image_sha256: str
    width: int
    height: int
    source_media_type: str
    source_mode: str
    mask_key: str
    mask_sha256: str
    model_metadata: dict
    analytics: dict


class Repository(Protocol):
    """Small persistence boundary for the service and dependency-free test doubles."""

    def ready(self) -> bool:
        """Check database access and the expected migration/schema."""
        ...

    def save(self, inspection: Inspection) -> None:
        """Atomically insert inspection and prediction metadata."""
        ...

    def get(self, identifier: UUID) -> Inspection | None:
        """Read one complete inspection/prediction pair."""
        ...


_SELECT = """
SELECT i.id, i.created_at, i.upload_sha256, i.image_key, i.image_sha256,
       i.width, i.height, i.source_media_type, i.source_mode,
       p.mask_key, p.mask_sha256, p.model_metadata, p.analytics
FROM inspections i JOIN predictions p ON p.inspection_id = i.id
"""


class PostgresRepository:
    """Use bound parameters and one transaction for both tables; never migrate implicitly."""

    def __init__(self, database_url: str, timeout_seconds: int = 3) -> None:
        self._database_url = database_url
        self._timeout = timeout_seconds

    def _connection(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(
            self._database_url,
            connect_timeout=self._timeout,
            options="-c statement_timeout=5000",
            row_factory=dict_row,
        )

    def ready(self) -> bool:
        with self._connection() as connection:
            versions = connection.execute(
                "SELECT version FROM damage_schema_migrations ORDER BY version"
            ).fetchall()
            if versions != [{"version": 1}]:
                return False
            connection.execute(_SELECT + " WHERE false")
        return True

    def save(self, inspection: Inspection) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO inspections
                (id, created_at, upload_sha256, image_key, image_sha256, width, height,
                 source_media_type, source_mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                tuple(
                    getattr(inspection, key)
                    for key in (
                        "id",
                        "created_at",
                        "upload_sha256",
                        "image_key",
                        "image_sha256",
                        "width",
                        "height",
                        "source_media_type",
                        "source_mode",
                    )
                ),
            )
            connection.execute(
                """
                INSERT INTO predictions
                (inspection_id, mask_key, mask_sha256, checkpoint_sha256,
                 model_metadata, analytics)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
                """,
                (
                    inspection.id,
                    inspection.mask_key,
                    inspection.mask_sha256,
                    inspection.model_metadata["checkpoint_sha256"],
                    json.dumps(inspection.model_metadata, allow_nan=False),
                    json.dumps(inspection.analytics, allow_nan=False),
                ),
            )

    def get(self, identifier: UUID) -> Inspection | None:
        with self._connection() as connection:
            row = connection.execute(_SELECT + " WHERE i.id = %s", (identifier,)).fetchone()
        return Inspection(**row) if row else None
