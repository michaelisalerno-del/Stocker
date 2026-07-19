"""Point-in-time sector membership validation and lookup."""

from __future__ import annotations

import pandas as pd

REQUIRED_SECTOR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "sector",
    "effective_from",
    "effective_to",
    "known_at",
    "stable_source_id",
    "source_provider",
    "source_dataset_id",
    "source_hash",
)


def validate_sector_membership_ledger(ledger: pd.DataFrame) -> list[str]:
    """Return deterministic fail-closed issues for an effective-dated sector ledger."""

    issues = [f"missing_{column}" for column in REQUIRED_SECTOR_COLUMNS if column not in ledger]
    if issues:
        return issues
    frame = ledger.copy()
    for column in ("effective_from", "effective_to", "known_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame["effective_from"].isna().any():
        issues.append("invalid_effective_from")
    if frame["known_at"].isna().any():
        issues.append("invalid_known_at")
    duplicates = frame.duplicated(["symbol", "effective_from"], keep=False)
    if duplicates.any():
        issues.append("duplicate_symbol_effective_date")
    for _, group in frame.sort_values("effective_from").groupby("symbol", sort=True):
        prior_end: pd.Timestamp | None = None
        for _, row in group.iterrows():
            start = row["effective_from"]
            if prior_end is not None and pd.notna(prior_end) and start < prior_end:
                issues.append("overlapping_sector_membership")
                break
            prior_end = row["effective_to"]
    return sorted(set(issues))


def sector_at(ledger: pd.DataFrame, *, symbol: str, effective_date: pd.Timestamp) -> str | None:
    """Return the sector known before an effective date, or ``None``."""

    if validate_sector_membership_ledger(ledger):
        return None
    date = pd.Timestamp(effective_date)
    date = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
    frame = ledger.copy()
    for column in ("effective_from", "effective_to", "known_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    active = frame.loc[
        frame["symbol"].eq(symbol)
        & frame["effective_from"].le(date)
        & frame["known_at"].lt(date)
        & (frame["effective_to"].isna() | frame["effective_to"].gt(date))
    ]
    if len(active) != 1:
        return None
    return str(active.iloc[0]["sector"])
