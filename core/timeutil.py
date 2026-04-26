"""UTC clock helpers — prefer over deprecated datetime.utcnow() (Python 3.12+)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """Current time in UTC (timezone-aware)."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Interpret naive datetimes as UTC; normalize others to UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
