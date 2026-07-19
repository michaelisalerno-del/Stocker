"""Monthly point-in-time universe qualification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from stocker_research.observable_event_ranking_v1.sector_context import (
    sector_at,
    validate_sector_membership_ledger,
)
from stocker_research.observable_event_ranking_v1.sessions import official_session_schedule


@dataclass(frozen=True)
class UniverseRules:
    """Frozen historical universe rules."""

    previous_close_min: float = 5.0
    prior_valid_sessions_min: int = 60
    trailing_sessions: int = 20
    median_daily_dollar_activity_min: float = 20_000_000.0
    bar_coverage_min: float = 0.95
    min_same_sector_total: int = 6


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _active_security_row(
    security_master: pd.DataFrame, symbol: str, month_start: pd.Timestamp
) -> pd.Series | None:
    required = {
        "symbol",
        "effective_from",
        "effective_to",
        "known_at",
        "security_type",
        "currency",
        "country",
        "stable_source_id",
        "source_hash",
    }
    if not required.issubset(security_master.columns):
        return None
    frame = security_master.copy()
    for column in ("effective_from", "effective_to", "known_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    active = frame.loc[
        frame["symbol"].eq(symbol)
        & frame["effective_from"].le(month_start)
        & frame["known_at"].lt(month_start)
        & (frame["effective_to"].isna() | frame["effective_to"].gt(month_start))
    ]
    return active.iloc[0] if len(active) == 1 else None


def build_monthly_universe_ledger(
    *,
    month_starts: list[pd.Timestamp],
    daily_stats: pd.DataFrame,
    security_master: pd.DataFrame,
    sector_membership: pd.DataFrame,
    rules: UniverseRules = UniverseRules(),
) -> pd.DataFrame:
    """Build immutable effective-month qualifications from strictly prior data."""

    required_stats = {
        "symbol",
        "session",
        "close",
        "daily_dollar_activity",
        "bar_coverage",
        "valid_session",
        "unresolved_problem",
        "source_provider",
        "source_dataset_id",
        "source_hash",
    }
    missing = sorted(required_stats.difference(daily_stats.columns))
    if missing:
        raise ValueError(f"daily statistics missing columns: {missing}")
    sector_issues = validate_sector_membership_ledger(sector_membership)
    stats = daily_stats.copy()
    stats["session"] = pd.to_datetime(stats["session"], utc=True)
    if stats.empty:
        return pd.DataFrame()
    symbols = sorted(set(stats["symbol"].astype(str)))
    rows: list[dict[str, Any]] = []
    for raw_month in sorted(month_starts):
        month = _utc(raw_month)
        calendar_start = min(stats["session"].min(), month - pd.Timedelta(days=730))
        official = official_session_schedule(
            calendar_start.strftime("%Y-%m-%d"),
            (month - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        official_sessions = pd.DatetimeIndex(official["session"])
        previous_official_session = official_sessions[-1] if len(official_sessions) else None
        trailing_official_sessions = set(official_sessions[-rules.trailing_sessions :])
        for symbol in symbols:
            reasons: list[str] = []
            metadata = _active_security_row(security_master, symbol, month)
            if metadata is None:
                reasons.append("missing_point_in_time_security_identity")
            else:
                if metadata["security_type"] != "common_stock":
                    reasons.append("not_us_common_stock")
                if metadata["currency"] != "USD":
                    reasons.append("not_usd_quoted")
                if metadata["country"] != "US":
                    reasons.append("not_us_security")
            history = stats.loc[
                stats["symbol"].eq(symbol) & stats["session"].lt(month)
            ].sort_values("session", kind="mergesort")
            history = history.loc[history["session"].isin(set(official_sessions))]
            valid_history = history.loc[history["valid_session"].astype(bool)]
            if len(valid_history) < rules.prior_valid_sessions_min:
                reasons.append("insufficient_prior_valid_sessions")
            previous_rows = (
                history.iloc[0:0]
                if previous_official_session is None
                else history.loc[history["session"].eq(previous_official_session)]
            )
            if len(previous_rows) != 1 or not bool(previous_rows.iloc[0]["valid_session"]):
                reasons.append("missing_or_invalid_previous_official_session")
                previous_close = None
                source_row = previous_rows.iloc[0] if len(previous_rows) == 1 else None
            else:
                previous_close = float(previous_rows.iloc[0]["close"])
                source_row = previous_rows.iloc[0]
            if previous_close is None or previous_close < rules.previous_close_min:
                reasons.append("previous_close_below_minimum")
            trailing = history.loc[history["session"].isin(trailing_official_sessions)]
            trailing_complete = bool(
                len(trailing) == rules.trailing_sessions
                and trailing["session"].nunique() == rules.trailing_sessions
                and trailing["valid_session"].astype(bool).all()
            )
            if not trailing_complete:
                reasons.append("insufficient_trailing_sessions")
                median_activity = None
                coverage = None
            else:
                median_activity = float(trailing["daily_dollar_activity"].median())
                coverage = float(trailing["bar_coverage"].mean())
                if median_activity < rules.median_daily_dollar_activity_min:
                    reasons.append("insufficient_dollar_activity_proxy")
                if coverage < rules.bar_coverage_min:
                    reasons.append("insufficient_five_minute_coverage")
            unresolved_rows = history.loc[history["unresolved_problem"].astype(bool)]
            if not unresolved_rows.empty:
                latest_problem = unresolved_rows["session"].max()
                if "problem_resolution_id" not in history:
                    latest_resolution = None
                else:
                    resolved_rows = history.loc[history["problem_resolution_id"].notna()]
                    latest_resolution = (
                        None if resolved_rows.empty else resolved_rows["session"].max()
                    )
                if latest_resolution is None or latest_resolution <= latest_problem:
                    reasons.append("unresolved_data_or_corporate_action_problem")
            sector = None
            if sector_issues:
                reasons.append("missing_point_in_time_sector_membership")
            else:
                sector = sector_at(sector_membership, symbol=symbol, effective_date=month)
                if sector is None:
                    reasons.append("missing_point_in_time_sector_membership")
            rows.append(
                {
                    "effective_date": month,
                    "symbol": symbol,
                    "stable_source_id": None
                    if metadata is None
                    else str(metadata["stable_source_id"]),
                    "security_source_hash": None
                    if metadata is None
                    else str(metadata["source_hash"]),
                    "source_provider": None
                    if source_row is None
                    else str(source_row["source_provider"]),
                    "source_dataset_id": None
                    if source_row is None
                    else str(source_row["source_dataset_id"]),
                    "source_hash": None if source_row is None else str(source_row["source_hash"]),
                    "sector": sector,
                    "previous_session_close": previous_close,
                    "prior_valid_sessions": len(valid_history),
                    "trailing_median_daily_dollar_activity_proxy": median_activity,
                    "trailing_bar_coverage": coverage,
                    "preliminary_eligible": not reasons,
                    "eligible": not reasons,
                    "qualification_reasons": json.dumps(
                        sorted(set(reasons)), separators=(",", ":")
                    ),
                }
            )
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        return ledger
    for (_effective_date, _sector), group in ledger.loc[
        ledger["preliminary_eligible"].astype(bool)
    ].groupby(["effective_date", "sector"], dropna=False, sort=True):
        if len(group) >= rules.min_same_sector_total:
            continue
        indices = group.index
        ledger.loc[indices, "eligible"] = False
        for index in indices:
            existing = json.loads(str(ledger.at[index, "qualification_reasons"]))
            ledger.at[index, "qualification_reasons"] = json.dumps(
                sorted(set([*existing, "insufficient_same_sector_peers"])),
                separators=(",", ":"),
            )
    return ledger.sort_values(["effective_date", "symbol"], kind="mergesort").reset_index(drop=True)
