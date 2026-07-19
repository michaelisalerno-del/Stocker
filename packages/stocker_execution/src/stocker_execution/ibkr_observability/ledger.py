"""Append-only persistence for observed IBKR top-of-book records."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from stocker_execution.ibkr_observability.models import (
    ObservationClassification,
    QuoteObservationRecord,
)

_SAFE_OBSERVATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class QuoteLedgerError(RuntimeError):
    """Raised when an observation cannot be safely appended."""


def _json_compatible(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise QuoteLedgerError("quote-ledger timestamps must be timezone-aware")
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise QuoteLedgerError("quote-ledger numeric values must be finite or null")
    return value


def append_quote_observation(root: Path, record: QuoteObservationRecord) -> Path:
    """Append one immutable observation without asserting an achieved fill."""

    if _SAFE_OBSERVATION_ID.fullmatch(record.observation_id) is None:
        raise QuoteLedgerError("observation id contains unsafe path characters")
    if record.fill_claim:
        raise QuoteLedgerError("IBKR quote observations cannot contain a fill claim")
    if record.classification is ObservationClassification.LIVE_TOP_OF_BOOK_OBSERVED and (
        not record.snapshot_complete
        or record.bid is None
        or record.ask is None
        or record.snapshot_completion_timestamp_utc is None
    ):
        raise QuoteLedgerError("complete live classification lacks a complete bid/ask snapshot")
    payload = _json_compatible(asdict(record))
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record.observation_id}.json"
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise QuoteLedgerError("quote observation already exists") from exc
    except OSError as exc:
        raise QuoteLedgerError(f"quote observation append failed: {type(exc).__name__}") from exc
    return path
