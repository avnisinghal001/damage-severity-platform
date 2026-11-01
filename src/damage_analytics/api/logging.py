"""Small JSON logger; never log uploads, original filenames, credentials or raw exceptions."""

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Keep request correlation and operational fields explicit and machine-readable."""

    def format(self, record: logging.LogRecord) -> str:
        values = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for name in (
            "request_id",
            "inspection_id",
            "error_type",
            "method",
            "path",
            "status",
            "duration_ms",
        ):
            if hasattr(record, name):
                values[name] = getattr(record, name)
        return json.dumps(values)


def configure_logging() -> None:
    """Configure only this package logger once; leave Uvicorn/root logging alone."""
    logger = logging.getLogger("damage_analytics.api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
