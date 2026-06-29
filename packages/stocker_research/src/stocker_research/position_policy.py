"""Research-side target-position policy overlays."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any

import pandas as pd

from stocker_research.hypothesis import HypothesisHoldingPolicy

MAX_POLICY_ACTION_SAMPLES = 50


@dataclass(frozen=True)
class PositionPolicyResult:
    """Adjusted target positions and deterministic policy-action counts."""

    adjusted_positions: pd.Series
    policy_applied: bool
    policy_name: str
    warnings: list[str] = field(default_factory=list)
    policy_actions: list[dict[str, Any]] = field(default_factory=list)
    raw_exposure: float = 0.0
    adjusted_exposure: float = 0.0
    positions_forced_flat_count: int = 0
    late_entries_blocked_count: int = 0
    overnight_carry_prevented_count: int = 0
    sessions_seen: int = 0
    incomplete_session_warning_count: int = 0
    session_close_source: str = "not_applicable"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary without the position series."""

        payload = asdict(self)
        payload.pop("adjusted_positions", None)
        return payload


def _is_daily_timeframe(timeframe: str) -> bool:
    normalized = timeframe.lower().strip()
    return normalized in {"1d", "d", "day", "daily"}


def _requires_session_flat_policy(policy: HypothesisHoldingPolicy, timeframe: str) -> bool:
    return (
        not _is_daily_timeframe(timeframe)
        and policy.preferred_style == "intraday"
        and policy.allow_overnight is False
    )


def _raw_positions(frame: pd.DataFrame, positions: pd.Series) -> pd.Series:
    return positions.reset_index(drop=True).astype(float).reindex(frame.index).fillna(0.0)


def _session_labels(timestamps: pd.Series) -> pd.Series:
    return timestamps.dt.date


@lru_cache(maxsize=128)
def _cached_calendar_closes(
    market_calendar: str,
    start_date: date,
    end_date: date,
) -> tuple[tuple[str, str], ...]:
    try:
        import pandas_market_calendars as mcal
    except ImportError:
        return ()

    calendar = mcal.get_calendar(market_calendar)
    schedule = calendar.schedule(
        start_date=start_date,
        end_date=end_date,
    )
    rows: list[tuple[str, str]] = []
    for session, row in schedule.iterrows():
        close = pd.Timestamp(row["market_close"]).tz_convert("UTC")
        rows.append((str(pd.Timestamp(session).date()), close.isoformat()))
    return tuple(rows)


def _calendar_closes(
    timestamps: pd.Series,
    market_calendar: str | None,
) -> tuple[dict[date, pd.Timestamp], list[str], str]:
    if not market_calendar:
        return {}, ["session_close_proxy_last_observed_timestamp"], "last_observed_timestamp"
    try:
        rows = _cached_calendar_closes(
            market_calendar,
            timestamps.min().date(),
            timestamps.max().date(),
        )
    except ImportError:
        return (
            {},
            ["market_calendar_unavailable_using_last_observed_timestamp"],
            "last_observed_timestamp",
        )
    except Exception:
        return (
            {},
            ["market_calendar_load_failed_using_last_observed_timestamp"],
            "last_observed_timestamp",
        )
    closes: dict[date, pd.Timestamp] = {}
    for session, close in rows:
        closes[pd.Timestamp(session).date()] = pd.Timestamp(close)
    if not closes:
        return (
            {},
            ["market_calendar_unavailable_using_last_observed_timestamp"],
            "last_observed_timestamp",
        )
    return closes, [], f"market_calendar:{market_calendar}"


def apply_holding_policy_to_positions(
    frame: pd.DataFrame,
    positions: pd.Series,
    *,
    policy: HypothesisHoldingPolicy,
    timeframe: str,
    market_calendar: str | None = None,
) -> PositionPolicyResult:
    """Apply a deterministic session-flat overlay to raw template target positions."""

    reset_frame = frame.reset_index(drop=True)
    raw = _raw_positions(reset_frame, positions)
    raw_exposure = float(raw.abs().mean()) if not raw.empty else 0.0
    if not _requires_session_flat_policy(policy, timeframe):
        warnings = []
        if _is_daily_timeframe(timeframe):
            warnings.append("daily_timeframe_not_session_flat_adjusted")
        return PositionPolicyResult(
            adjusted_positions=raw,
            policy_applied=False,
            policy_name="none",
            warnings=warnings,
            raw_exposure=raw_exposure,
            adjusted_exposure=raw_exposure,
            sessions_seen=0,
        )
    if "timestamp" not in reset_frame:
        return PositionPolicyResult(
            adjusted_positions=raw,
            policy_applied=False,
            policy_name="session_flat_intraday",
            warnings=["missing_timestamp_no_position_policy_applied"],
            raw_exposure=raw_exposure,
            adjusted_exposure=raw_exposure,
        )

    timestamps = pd.to_datetime(reset_frame["timestamp"], utc=True).reset_index(drop=True)
    session_labels = _session_labels(timestamps)
    calendar_closes, warnings, close_source = _calendar_closes(timestamps, market_calendar)
    adjusted = raw.copy()
    adjusted.iloc[:] = 0.0
    policy_actions: list[dict[str, Any]] = []
    positions_forced_flat_count = 0
    late_entries_blocked_count = 0
    overnight_carry_prevented_count = 0
    incomplete_session_warning_count = 0
    previous_session_last_raw = 0.0
    sessions_seen = 0

    for session_label, group_index in session_labels.groupby(session_labels).groups.items():
        sessions_seen += 1
        indices = [int(index) for index in group_index]
        observed_close = timestamps.take(indices).max()
        session_close = calendar_closes.get(session_label)
        if session_close is None:
            session_close = observed_close
            if market_calendar:
                incomplete_session_warning_count += 1
                warnings.append(f"missing_calendar_session:{session_label}")
        elif observed_close < session_close:
            incomplete_session_warning_count += 1
            warnings.append(f"incomplete_session:{session_label}")

        carry_block_active = bool(
            abs(previous_session_last_raw) > 0.0 and abs(float(raw.iloc[indices[0]])) > 0.0
        )
        if carry_block_active:
            overnight_carry_prevented_count += 1
            policy_actions.append(
                {
                    "action": "overnight_carry_prevented",
                    "timestamp": str(timestamps.iloc[indices[0]]),
                    "session": str(session_label),
                }
            )

        previous_adjusted = 0.0
        previous_raw_same_session = 0.0
        for index in indices:
            desired = float(raw.iloc[index])
            if carry_block_active:
                if abs(desired) == 0.0:
                    carry_block_active = False
                else:
                    previous_raw_same_session = desired
                    continue

            minutes_to_close = float((session_close - timestamps.iloc[index]).total_seconds() / 60)
            force_flat = minutes_to_close <= policy.flatten_before_close_minutes
            raw_entry = abs(previous_raw_same_session) == 0.0 and abs(desired) > 0.0
            late_entry = raw_entry and minutes_to_close <= policy.entry_cutoff_before_close_minutes
            if force_flat:
                if late_entry:
                    late_entries_blocked_count += 1
                    policy_actions.append(
                        {
                            "action": "late_entry_blocked",
                            "timestamp": str(timestamps.iloc[index]),
                            "session": str(session_label),
                        }
                    )
                if abs(desired) > 0.0:
                    positions_forced_flat_count += 1
                    policy_actions.append(
                        {
                            "action": "forced_flat_before_close",
                            "timestamp": str(timestamps.iloc[index]),
                            "session": str(session_label),
                        }
                    )
                adjusted.iloc[index] = 0.0
            elif late_entry:
                late_entries_blocked_count += 1
                adjusted.iloc[index] = 0.0
                policy_actions.append(
                    {
                        "action": "late_entry_blocked",
                        "timestamp": str(timestamps.iloc[index]),
                        "session": str(session_label),
                    }
                )
            elif abs(previous_adjusted) == 0.0 and not raw_entry and abs(desired) > 0.0:
                adjusted.iloc[index] = 0.0
            else:
                adjusted.iloc[index] = desired
            previous_adjusted = float(adjusted.iloc[index])
            previous_raw_same_session = desired

        previous_session_last_raw = float(raw.iloc[indices[-1]])

    adjusted_exposure = float(adjusted.abs().mean()) if not adjusted.empty else 0.0
    return PositionPolicyResult(
        adjusted_positions=adjusted,
        policy_applied=True,
        policy_name="session_flat_intraday",
        warnings=sorted(set(warnings)),
        policy_actions=policy_actions,
        raw_exposure=raw_exposure,
        adjusted_exposure=adjusted_exposure,
        positions_forced_flat_count=positions_forced_flat_count,
        late_entries_blocked_count=late_entries_blocked_count,
        overnight_carry_prevented_count=overnight_carry_prevented_count,
        sessions_seen=sessions_seen,
        incomplete_session_warning_count=incomplete_session_warning_count,
        session_close_source=close_source,
    )


def summarize_position_policy_effect(
    results: list[PositionPolicyResult],
) -> dict[str, Any]:
    """Aggregate policy-action counts across walk-forward windows."""

    if not results:
        return {
            "policy_applied": False,
            "policy_name": "none",
            "warnings": [],
            "policy_actions": [],
            "policy_action_count": 0,
            "policy_actions_omitted_count": 0,
            "raw_exposure": 0.0,
            "adjusted_exposure": 0.0,
            "positions_forced_flat_count": 0,
            "late_entries_blocked_count": 0,
            "overnight_carry_prevented_count": 0,
            "sessions_seen": 0,
            "incomplete_session_warning_count": 0,
            "session_close_source": "not_applicable",
        }
    all_actions = [action for result in results for action in result.policy_actions]
    return {
        "policy_applied": any(result.policy_applied for result in results),
        "policy_name": results[0].policy_name,
        "warnings": sorted({warning for result in results for warning in result.warnings}),
        "policy_actions": all_actions[:MAX_POLICY_ACTION_SAMPLES],
        "policy_action_count": len(all_actions),
        "policy_actions_omitted_count": max(0, len(all_actions) - MAX_POLICY_ACTION_SAMPLES),
        "raw_exposure": float(sum(result.raw_exposure for result in results) / len(results)),
        "adjusted_exposure": float(
            sum(result.adjusted_exposure for result in results) / len(results)
        ),
        "positions_forced_flat_count": sum(
            result.positions_forced_flat_count for result in results
        ),
        "late_entries_blocked_count": sum(result.late_entries_blocked_count for result in results),
        "overnight_carry_prevented_count": sum(
            result.overnight_carry_prevented_count for result in results
        ),
        "sessions_seen": sum(result.sessions_seen for result in results),
        "incomplete_session_warning_count": sum(
            result.incomplete_session_warning_count for result in results
        ),
        "session_close_source": results[0].session_close_source,
    }
