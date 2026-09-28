"""Approved, auditable contract selections for frozen historical reference sessions.

Saxo chart daily boundaries are not assumed to be exchange-session volume. Until that
mapping is verified, accept an explicit selection audit, not an inferred continuous future.
The selected contracts' one-minute bars are still obtained only from Saxo.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field

from stocker_execution.config import ContractSelection, Environment, Market, Strict


class SelectedSession(Strict):
    day: date
    contract: ContractSelection
    prior_session_volume_evidence: str = Field(min_length=20, max_length=2000)


class SelectionAudit(Strict):
    provider: Literal["SAXO"]
    environment: Environment
    market: Market
    as_of: date
    current_uic: int = Field(gt=0)
    selection_rule: Literal["PRIOR_COMPLETED_EXCHANGE_SESSION_VOLUME"]
    approval: str = Field(min_length=20, max_length=2000)
    sessions: list[SelectedSession] = Field(min_length=5, max_length=5)


def load_selections(
    path: Path, environment: str, market: str, day: date, current_uic: int
) -> tuple[SelectionAudit, str]:
    if path.stat().st_size > 128 * 1024:
        raise ValueError("REFERENCE_SELECTION_AUDIT_SIZE_LIMIT")
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) > 5:
        raise ValueError("REFERENCE_SELECTION_AUDIT_INVALID")
    books = [SelectionAudit.model_validate(v) for v in data]
    matches = [b for b in books if b.market == market]
    if len(matches) != 1:
        raise ValueError("REFERENCE_SELECTION_AUDIT_MISSING_OR_DUPLICATE")
    book = matches[0]
    dates = [s.day for s in book.sessions]
    if (
        book.environment != environment
        or book.as_of != day
        or book.current_uic != current_uic
        or dates != sorted(set(dates))
        or max(dates) >= day
        or any(s.contract.environment != environment for s in book.sessions)
    ):
        raise ValueError("REFERENCE_SELECTION_AUDIT_DATE_OR_IDENTITY_MISMATCH")
    return book, hashlib.sha256(raw).hexdigest()
