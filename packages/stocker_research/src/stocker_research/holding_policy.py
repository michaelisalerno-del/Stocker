"""Holding policy and overnight/weekend exposure analysis."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
from pydantic import BaseModel

from stocker_backtest.vectorized import VectorizedBacktestResult
from stocker_research.classification import ResearchClassification
from stocker_research.hypothesis import HypothesisHoldingPolicy


class HoldingPolicyReport(BaseModel):
    """Measured holding behavior for selected target positions."""

    max_holding_bars: int
    estimated_holding_sessions: int
    overnight_exposure_count: int
    weekend_exposure_count: int
    overnight_return_contribution: float
    weekend_return_contribution: float
    gap_return_contribution: float
    gap_return_contribution_pct: float
    intraday_return_contribution: float | None
    session_flat_compliant: bool
    holding_policy_violations: list[str]
    holding_policy_warning_reasons: list[str]
    attribution_note: str
    evidence_tier: str


class HoldingPolicyDecision(BaseModel):
    """Classification impact from holding-policy analysis."""

    classification: ResearchClassification
    reasons: list[str]
    evidence_tier: str
    swing_exceptional_pass: bool


def _is_daily_timeframe(timeframe: str) -> bool:
    normalized = timeframe.lower().strip()
    return normalized in {"1d", "d", "day", "daily"}


def _aligned_positions(frame: pd.DataFrame, positions: pd.Series) -> pd.Series:
    return positions.reset_index(drop=True).astype(float).reindex(frame.index).fillna(0.0)


def _contains_weekend(start: date, end: date) -> bool:
    current = start + timedelta(days=1)
    while current <= end:
        if current.weekday() >= 5:
            return True
        current += timedelta(days=1)
    return False


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _is_unconditionally_allowed(value: object) -> bool:
    return value is True


def _max_holding_stats(timestamps: pd.Series, held: pd.Series) -> tuple[int, int]:
    max_bars = 0
    max_sessions = 0
    current_bars = 0
    current_dates: set[date] = set()
    for index, is_held in enumerate(held):
        if bool(is_held):
            current_bars += 1
            timestamp = pd.Timestamp(timestamps.iloc[index])
            current_dates.add(timestamp.date())
            max_bars = max(max_bars, current_bars)
            max_sessions = max(max_sessions, len(current_dates))
            continue
        current_bars = 0
        current_dates = set()
    return max_bars, max_sessions


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


def _calendar_close_by_date(
    timestamps: pd.Series,
    market_calendar: str | None,
) -> dict[date, pd.Timestamp]:
    if not market_calendar or timestamps.empty:
        return {}
    try:
        rows = _cached_calendar_closes(
            market_calendar,
            timestamps.min().date(),
            timestamps.max().date(),
        )
    except Exception:
        return {}
    return {pd.Timestamp(session).date(): pd.Timestamp(close) for session, close in rows}


def _intraday_close_violations(
    timestamps: pd.Series,
    positions: pd.Series,
    *,
    policy: HypothesisHoldingPolicy,
    market_calendar: str | None = None,
) -> tuple[list[str], bool]:
    violations: list[str] = []
    flat_by_close = True
    frame = pd.DataFrame({"timestamp": timestamps, "position": positions})
    frame["session_date"] = frame["timestamp"].dt.date
    proxy_close_by_date = frame.groupby("session_date")["timestamp"].transform("max")
    calendar_closes = _calendar_close_by_date(timestamps, market_calendar)
    if calendar_closes:
        calendar_close_by_date = pd.to_datetime(
            frame["session_date"].map(calendar_closes),
            utc=True,
        )
        close_by_date = calendar_close_by_date.fillna(proxy_close_by_date)
    else:
        close_by_date = proxy_close_by_date
    minutes_to_close = (close_by_date - frame["timestamp"]).dt.total_seconds() / 60
    held_near_close = (frame["position"].abs() > 0) & (
        minutes_to_close <= policy.flatten_before_close_minutes
    )
    if bool(held_near_close.any()):
        violations.append("failed_flatten_before_close")
        flat_by_close = False

    previous_position = frame["position"].shift(1).fillna(0.0)
    entries = (previous_position.abs() == 0.0) & (frame["position"].abs() > 0.0)
    late_entries = entries & (minutes_to_close <= policy.entry_cutoff_before_close_minutes)
    if bool(late_entries.any()):
        violations.append("entry_after_cutoff")
        flat_by_close = False
    return violations, flat_by_close


def analyze_holding_policy(
    frame: pd.DataFrame,
    positions: pd.Series,
    *,
    result: VectorizedBacktestResult | None,
    selected_net_return: float | None = None,
    timeframe: str,
    policy: HypothesisHoldingPolicy,
    window_ids: pd.Series | None = None,
    market_calendar: str | None = None,
) -> HoldingPolicyReport:
    """Analyze holding behavior and overnight/weekend return contribution."""

    if frame.empty:
        return HoldingPolicyReport(
            max_holding_bars=0,
            estimated_holding_sessions=0,
            overnight_exposure_count=0,
            weekend_exposure_count=0,
            overnight_return_contribution=0.0,
            weekend_return_contribution=0.0,
            gap_return_contribution=0.0,
            gap_return_contribution_pct=0.0,
            intraday_return_contribution=None,
            session_flat_compliant=False,
            holding_policy_violations=["empty_holding_analysis"],
            holding_policy_warning_reasons=[],
            attribution_note="No rows were available for holding-policy analysis.",
            evidence_tier="rejected_holding_risk",
        )

    timestamps = pd.to_datetime(frame["timestamp"], utc=True).reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce").reset_index(drop=True)
    open_ = pd.to_numeric(frame.get("open", frame["close"]), errors="coerce").reset_index(drop=True)
    aligned_positions = _aligned_positions(frame.reset_index(drop=True), positions)
    aligned_window_ids = (
        window_ids.reset_index(drop=True).reindex(frame.index).fillna("window_000")
        if window_ids is not None
        else pd.Series(["window_000"] * len(frame))
    )
    held = aligned_positions.abs() > 0
    max_holding_bars = 0
    estimated_holding_sessions = 0
    for _, group_index in aligned_window_ids.groupby(aligned_window_ids).groups.items():
        indices = [int(index) for index in group_index]
        group_positions = held.take(indices).reset_index(drop=True)
        group_timestamps = timestamps.take(indices).reset_index(drop=True)
        group_max_bars, group_max_sessions = _max_holding_stats(
            group_timestamps,
            group_positions,
        )
        max_holding_bars = max(max_holding_bars, group_max_bars)
        estimated_holding_sessions = max(estimated_holding_sessions, group_max_sessions)
    daily_timeframe = _is_daily_timeframe(timeframe)

    overnight_count = 0
    weekend_count = 0
    gap_contribution = 0.0
    weekend_contribution = 0.0
    intraday_contribution = 0.0
    for index in range(1, len(frame)):
        if aligned_window_ids.iloc[index] != aligned_window_ids.iloc[index - 1]:
            continue
        previous_position = float(aligned_positions.iloc[index - 1])
        if previous_position == 0.0:
            continue
        previous_timestamp = pd.Timestamp(timestamps.iloc[index - 1])
        current_timestamp = pd.Timestamp(timestamps.iloc[index])
        previous_date = previous_timestamp.date()
        current_date = current_timestamp.date()
        date_changed = current_date != previous_date
        is_weekend_gap = date_changed and _contains_weekend(previous_date, current_date)
        if daily_timeframe or date_changed:
            overnight_count += 1
            previous_close = float(close.iloc[index - 1])
            current_open = float(open_.iloc[index])
            if previous_close:
                gap_return = previous_position * (current_open / previous_close - 1.0)
                gap_contribution += gap_return
                if is_weekend_gap:
                    weekend_contribution += gap_return
        if is_weekend_gap:
            weekend_count += 1
        if not daily_timeframe and not date_changed:
            previous_close = float(close.iloc[index - 1])
            current_close = float(close.iloc[index])
            if previous_close:
                intraday_contribution += previous_position * (current_close / previous_close - 1.0)

    total_return = abs(float(selected_net_return)) if selected_net_return is not None else 0.0
    if total_return == 0.0 and result is not None:
        total_return = abs(float(result.net_return))
    if total_return == 0.0:
        total_return = abs(gap_contribution) + abs(intraday_contribution)
    gap_pct = abs(gap_contribution) / total_return if total_return else 0.0

    violations: list[str] = []
    warnings: list[str] = []
    attribution_note = "Intraday timestamps allow session-boundary holding analysis."
    if daily_timeframe:
        session_flat_compliant = False
        attribution_note = (
            "daily data cannot prove session-flat tradability; daily-bar strategies are "
            "swing research vehicles unless session-flat intraday data exists."
        )
        _append_unique(warnings, "daily_bars_are_swing_research_vehicle")
        _append_unique(warnings, "session_flat_unproven")
    else:
        close_violations, flat_by_close = _intraday_close_violations(
            timestamps,
            aligned_positions,
            policy=policy,
            market_calendar=market_calendar,
        )
        violations.extend(close_violations)
        session_flat_compliant = flat_by_close and overnight_count == 0

    if session_flat_compliant:
        _append_unique(warnings, "session_flat_compliant")
    if overnight_count:
        _append_unique(warnings, "held_overnight")
    if weekend_count and policy.allow_weekend == "exceptional_only":
        _append_unique(warnings, "weekend_exceptional_only")
    if gap_pct > policy.max_gap_return_contribution_pct:
        _append_unique(warnings, "gap_dependent_returns")

    if overnight_count and policy.allow_overnight is False:
        _append_unique(violations, "overnight_not_allowed")
    if weekend_count and policy.allow_weekend is False:
        _append_unique(violations, "weekend_not_allowed")
    if estimated_holding_sessions > policy.max_holding_sessions:
        _append_unique(violations, "max_holding_sessions_exceeded")

    evidence_tier = "intraday_preferred" if session_flat_compliant else "swing_research_vehicle"
    if violations:
        evidence_tier = "rejected_holding_risk"

    return HoldingPolicyReport(
        max_holding_bars=max_holding_bars,
        estimated_holding_sessions=estimated_holding_sessions,
        overnight_exposure_count=overnight_count,
        weekend_exposure_count=weekend_count,
        overnight_return_contribution=float(gap_contribution),
        weekend_return_contribution=float(weekend_contribution),
        gap_return_contribution=float(gap_contribution),
        gap_return_contribution_pct=float(gap_pct),
        intraday_return_contribution=None if daily_timeframe else float(intraday_contribution),
        session_flat_compliant=session_flat_compliant,
        holding_policy_violations=violations,
        holding_policy_warning_reasons=warnings,
        attribution_note=attribution_note,
        evidence_tier=evidence_tier,
    )


def build_holding_policy_decision(
    report: HoldingPolicyReport,
    policy: HypothesisHoldingPolicy,
    *,
    selected_excess_vs_benchmark: float,
    selected_excess_vs_null: float,
    trade_count: int,
    max_drawdown: float,
) -> HoldingPolicyDecision:
    """Return the classification impact implied by holding risk."""

    reasons = [*report.holding_policy_warning_reasons, *report.holding_policy_violations]
    if "weekend_not_allowed" in report.holding_policy_violations:
        return HoldingPolicyDecision(
            classification="rejected_weekend_risk",
            reasons=[*reasons, "weekend_risk_too_high"],
            evidence_tier="rejected_holding_risk",
            swing_exceptional_pass=False,
        )
    if "overnight_not_allowed" in report.holding_policy_violations:
        return HoldingPolicyDecision(
            classification="rejected_overnight_risk",
            reasons=[*reasons, "overnight_risk_too_high"],
            evidence_tier="rejected_holding_risk",
            swing_exceptional_pass=False,
        )
    if report.holding_policy_violations:
        return HoldingPolicyDecision(
            classification="rejected_holding_policy_violation",
            reasons=reasons,
            evidence_tier="rejected_holding_risk",
            swing_exceptional_pass=False,
        )
    if report.session_flat_compliant:
        return HoldingPolicyDecision(
            classification="candidate_intraday_test",
            reasons=reasons or ["session_flat_compliant"],
            evidence_tier="intraday_preferred",
            swing_exceptional_pass=False,
        )
    if report.weekend_exposure_count > policy.max_weekend_exposure_count and not (
        _is_unconditionally_allowed(policy.allow_weekend)
    ):
        return HoldingPolicyDecision(
            classification="rejected_weekend_risk",
            reasons=[*reasons, "weekend_risk_too_high"],
            evidence_tier="rejected_holding_risk",
            swing_exceptional_pass=False,
        )
    if report.overnight_exposure_count > policy.max_overnight_exposure_count and not (
        _is_unconditionally_allowed(policy.allow_overnight)
    ):
        return HoldingPolicyDecision(
            classification="rejected_overnight_risk",
            reasons=[*reasons, "overnight_risk_too_high"],
            evidence_tier="rejected_holding_risk",
            swing_exceptional_pass=False,
        )
    if report.gap_return_contribution_pct > policy.max_gap_return_contribution_pct:
        return HoldingPolicyDecision(
            classification="rejected_overnight_risk",
            reasons=[*reasons, "gap_dependent_returns"],
            evidence_tier="rejected_holding_risk",
            swing_exceptional_pass=False,
        )

    swing_exceptional_checks = [
        selected_excess_vs_benchmark >= policy.min_swing_excess_vs_benchmark,
        selected_excess_vs_null >= policy.min_swing_excess_vs_null,
        trade_count >= policy.min_swing_trade_count,
        max_drawdown >= -abs(policy.max_swing_drawdown),
    ]
    swing_exceptional_pass = all(swing_exceptional_checks)
    if policy.require_exceptional_evidence_for_swing and not swing_exceptional_pass:
        return HoldingPolicyDecision(
            classification="interesting_swing_needs_more_tests",
            reasons=[*reasons, "swing_not_exceptional"],
            evidence_tier="swing_research_vehicle",
            swing_exceptional_pass=False,
        )
    return HoldingPolicyDecision(
        classification="candidate_swing_exceptional",
        reasons=[*reasons, "swing_exceptional_evidence"],
        evidence_tier="swing_exceptional",
        swing_exceptional_pass=swing_exceptional_pass,
    )


def report_to_dict(report: HoldingPolicyReport) -> dict[str, Any]:
    """Return a JSON-ready holding policy report."""

    return report.model_dump(mode="json")
