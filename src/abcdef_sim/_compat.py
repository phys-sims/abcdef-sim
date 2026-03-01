"""Runtime compatibility helpers for third-party dependencies."""

from __future__ import annotations

import datetime as _datetime


def ensure_datetime_utc() -> None:
    """Backfill ``datetime.UTC`` on Python < 3.11 for dependency compatibility."""

    if not hasattr(_datetime, "UTC"):
        setattr(_datetime, "UTC", _datetime.timezone.utc)
