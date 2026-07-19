"""Outcome-free E1 event primitives and decision-grid timing."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
from statistics import median
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stocker_research.observable_event_ranking_v1.contract import DECISION_CLOCKS
from stocker_research.observable_event_ranking_v1.targets import target_reference_times

NEW_YORK = ZoneInfo("America/New_York")


def leave_one_out_medians(values: Sequence[float]) -> list[float]:
    """Return the median of all *other* observations for every position."""

    if len(values) < 2:
        raise ValueError("leave-one-out median requires at least two observations")
    return [
        float(median([other for index, other in enumerate(values) if index != own]))
        for own in range(len(values))
    ]


def _session_grids(local_date: datetime) -> list[datetime]:
    grids: list[datetime] = []
    for clock in DECISION_CLOCKS:
        hour, minute = (int(part) for part in clock.split(":"))
        grids.append(datetime.combine(local_date.date(), time(hour, minute), tzinfo=NEW_YORK))
    return grids


def assign_decision_time(
    *,
    confirmation_time: datetime,
    availability_time: datetime,
    exact_grid_available_before_scoring: bool = False,
) -> datetime | None:
    """Assign an event to its first causally valid New York decision grid.

    Exact-grid confirmation is deferred unless earlier source availability and an
    explicit proof flag are both present.
    """

    if confirmation_time.tzinfo is None or availability_time.tzinfo is None:
        raise ValueError("confirmation and availability timestamps must be timezone-aware")
    confirmation_local = confirmation_time.astimezone(NEW_YORK)
    availability_local = availability_time.astimezone(NEW_YORK)
    effective = max(confirmation_local, availability_local)
    for grid in _session_grids(confirmation_local):
        if effective < grid:
            return grid.astimezone(UTC)
        if (
            confirmation_local == grid
            and availability_local < grid
            and exact_grid_available_before_scoring
        ):
            return grid.astimezone(UTC)
    return None


_BAR_COLUMNS = {
    "symbol",
    "sector",
    "session",
    "bar_end",
    "close",
    "fully_completed",
    "gap_status",
    "universe_eligible",
}


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _window_reasons(group: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    bar_end = pd.to_datetime(group["bar_end"], utc=True)
    interval_ok = bar_end.diff().eq(pd.Timedelta(minutes=5))
    contiguous = interval_ok.rolling(6, min_periods=6).sum().eq(6)
    completed = group["fully_completed"].astype(bool).rolling(7, min_periods=7).sum().eq(7)
    gap_free = group["gap_status"].eq("complete").rolling(7, min_periods=7).sum().eq(7)
    reasons: list[str] = []
    for position in range(len(group)):
        row_reasons: list[str] = []
        if not bool(completed.iloc[position]):
            row_reasons.append("incomplete_bar")
        if not bool(contiguous.iloc[position]) or not bool(gap_free.iloc[position]):
            row_reasons.append("source_gap_crossing_window")
        reasons.append("|".join(row_reasons))
    return completed & contiguous & gap_free, pd.Series(reasons, index=group.index, dtype="string")


def build_relative_context(
    bars: pd.DataFrame,
    *,
    market_stocks_min: int = 20,
    sector_other_stocks_min: int = 5,
) -> pd.DataFrame:
    """Construct causal E1 return context from completed contiguous five-minute bars."""

    _require_columns(bars, _BAR_COLUMNS)
    frame = bars.copy()
    frame["session"] = pd.to_datetime(frame["session"], utc=True)
    frame["bar_end"] = pd.to_datetime(frame["bar_end"], utc=True)
    frame = frame.sort_values(["session", "symbol", "bar_end"], kind="mergesort")
    pieces: list[pd.DataFrame] = []
    for (_, _), group in frame.groupby(["session", "symbol"], sort=True, observed=True):
        part = group.copy()
        close = part["close"].astype("float64")
        part["stock_return_5m"] = close / close.shift(1) - 1.0
        part["recent_15m_return"] = close / close.shift(3) - 1.0
        part["preceding_15m_return"] = close.shift(3) / close.shift(6) - 1.0
        part["stock_return_30m"] = close / close.shift(6) - 1.0
        valid, reasons = _window_reasons(part)
        part["context_valid"] = valid & part["universe_eligible"].astype(bool)
        part["context_unavailable_reason"] = reasons
        pieces.append(part)
    context = pd.concat(pieces, ignore_index=True) if pieces else frame.copy()
    relative_columns = (
        "stock_return_5m",
        "recent_15m_return",
        "preceding_15m_return",
        "stock_return_30m",
    )
    for column in relative_columns:
        context[f"market_median_{column}"] = np.nan
        context[f"sector_median_{column}"] = np.nan
    context["market_peer_count"] = 0
    context["sector_peer_count"] = 0

    for _, simultaneous in context.groupby(["session", "bar_end"], sort=True, observed=True):
        eligible = simultaneous.loc[simultaneous["context_valid"].astype(bool)]
        for row_index, row in eligible.iterrows():
            market_peers = eligible.loc[eligible["symbol"] != row["symbol"]]
            sector_peers = market_peers.loc[market_peers["sector"] == row["sector"]]
            context.at[row_index, "market_peer_count"] = len(market_peers)
            context.at[row_index, "sector_peer_count"] = len(sector_peers)
            if len(eligible) >= market_stocks_min:
                for column in relative_columns:
                    context.at[row_index, f"market_median_{column}"] = float(
                        market_peers[column].median()
                    )
            if len(sector_peers) >= sector_other_stocks_min:
                for column in relative_columns:
                    context.at[row_index, f"sector_median_{column}"] = float(
                        sector_peers[column].median()
                    )

    enough_market = context["market_peer_count"].ge(market_stocks_min - 1)
    enough_sector = context["sector_peer_count"].ge(sector_other_stocks_min)
    initially_valid = context["context_valid"].astype(bool)
    context.loc[initially_valid & ~enough_market, "context_unavailable_reason"] += (
        "|insufficient_market_peers"
    )
    context.loc[initially_valid & ~enough_sector, "context_unavailable_reason"] += (
        "|insufficient_sector_peers"
    )
    context["context_unavailable_reason"] = context["context_unavailable_reason"].str.strip("|")
    context["context_valid"] = initially_valid & enough_market & enough_sector
    context["market_relative_return_5m"] = (
        context["stock_return_5m"] - context["market_median_stock_return_5m"]
    )
    context["recent_market_relative"] = (
        context["recent_15m_return"] - context["market_median_recent_15m_return"]
    )
    context["preceding_market_relative"] = (
        context["preceding_15m_return"] - context["market_median_preceding_15m_return"]
    )
    context["market_relative_return_15m"] = context["recent_market_relative"]
    context["market_relative_return_30m"] = (
        context["stock_return_30m"] - context["market_median_stock_return_30m"]
    )
    context["recent_sector_relative"] = (
        context["recent_15m_return"] - context["sector_median_recent_15m_return"]
    )
    context["preceding_sector_relative"] = (
        context["preceding_15m_return"] - context["sector_median_preceding_15m_return"]
    )
    context["sector_relative_return_15m"] = context["recent_sector_relative"]
    context["sector_relative_return_30m"] = (
        context["stock_return_30m"] - context["sector_median_stock_return_30m"]
    )
    context["market_relative_acceleration"] = (
        context["recent_market_relative"] - context["preceding_market_relative"]
    )
    context["sector_relative_acceleration"] = (
        context["recent_sector_relative"] - context["preceding_sector_relative"]
    )
    context["bar_clock"] = context["bar_end"].dt.tz_convert(NEW_YORK).dt.strftime("%H:%M")
    return context.sort_values(["session", "bar_end", "symbol"], kind="mergesort").reset_index(
        drop=True
    )


def robust_location_scale(values: Iterable[float], *, epsilon: float = 1e-8) -> tuple[float, float]:
    """Return deterministic median and robust scale with IQR and epsilon fallbacks."""

    clean = np.asarray(values, dtype="float64")
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    location = float(np.median(clean))
    mad_scale = float(1.4826 * np.median(np.abs(clean - location)))
    if np.isfinite(mad_scale) and mad_scale >= epsilon:
        return location, mad_scale
    q25, q75 = np.quantile(clean, [0.25, 0.75], method="linear")
    iqr_scale = float((q75 - q25) / 1.349)
    return location, iqr_scale if np.isfinite(iqr_scale) and iqr_scale >= epsilon else epsilon


def causal_robust_scale(
    context: pd.DataFrame,
    *,
    trailing_sessions: int = 60,
    min_observations: int = 50,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Scale accelerations using same-clock prior sessions only, with a prior-pool fallback."""

    required = {
        "session",
        "bar_end",
        "market_relative_acceleration",
        "sector_relative_acceleration",
        "context_valid",
    }
    _require_columns(context, required)
    frame = context.copy()
    frame["session"] = pd.to_datetime(frame["session"], utc=True)
    frame["bar_end"] = pd.to_datetime(frame["bar_end"], utc=True)
    if "bar_clock" not in frame:
        frame["bar_clock"] = frame["bar_end"].dt.tz_convert(NEW_YORK).dt.strftime("%H:%M")
    for output in ("market_relative_acceleration_z", "sector_relative_acceleration_z"):
        frame[output] = np.nan
    sessions = list(pd.Index(frame["session"].drop_duplicates()).sort_values())
    for session_position, session in enumerate(sessions):
        prior_session_values = sessions[
            max(0, session_position - trailing_sessions) : session_position
        ]
        prior = frame.loc[
            frame["session"].isin(prior_session_values) & frame["context_valid"].astype(bool)
        ]
        current_indices = frame.index[
            frame["session"].eq(session) & frame["context_valid"].astype(bool)
        ]
        for row_index in current_indices:
            clock = frame.at[row_index, "bar_clock"]
            same_clock = prior.loc[prior["bar_clock"].eq(clock)]
            scale_population = same_clock if len(same_clock) >= min_observations else prior
            if len(scale_population) < min_observations:
                continue
            for source, output in (
                ("market_relative_acceleration", "market_relative_acceleration_z"),
                ("sector_relative_acceleration", "sector_relative_acceleration_z"),
            ):
                location, scale = robust_location_scale(
                    scale_population[source].to_numpy(dtype="float64"), epsilon=epsilon
                )
                current_value = float(cast(Any, frame.at[row_index, source]))
                frame.at[row_index, output] = (current_value - location) / scale
    frame["event_strength"] = frame[
        ["market_relative_acceleration_z", "sector_relative_acceleration_z"]
    ].min(axis=1, skipna=False)
    return frame


@dataclass(frozen=True)
class EventCalibration:
    """Outcome-free rows and their single frozen q90 threshold."""

    threshold: float
    rows: pd.DataFrame
    calibration_months: tuple[str, ...]


def calibrate_q90_threshold(context: pd.DataFrame) -> EventCalibration:
    """Fit q90 once on the earliest six valid months without reading outcomes."""

    _require_columns(
        context,
        {
            "session",
            "event_strength",
            "recent_market_relative",
            "recent_sector_relative",
            "context_valid",
        },
    )
    frame = context.copy()
    sessions = pd.to_datetime(frame["session"], utc=True)
    frame["calibration_month"] = sessions.dt.tz_localize(None).dt.to_period("M").astype(str)
    valid = frame.loc[
        frame["context_valid"].astype(bool)
        & frame["recent_market_relative"].gt(0.0)
        & frame["recent_sector_relative"].gt(0.0)
        & frame["event_strength"].notna()
    ].copy()
    months = tuple(sorted(valid["calibration_month"].unique())[:6])
    if len(months) < 6:
        raise ValueError("six valid outcome-free calibration months are required")
    rows = valid.loc[valid["calibration_month"].isin(months)].copy()
    forbidden = [
        column
        for column in rows
        if "future" in str(column).lower() or "target" in str(column).lower()
    ]
    if forbidden:
        raise ValueError(f"calibration rows contain outcome columns: {forbidden}")
    threshold = float(np.quantile(rows["event_strength"].to_numpy(dtype="float64"), 0.90))
    return EventCalibration(threshold=threshold, rows=rows, calibration_months=months)


def trigger_e1_events(context: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Apply the one frozen E1 trigger with inclusive threshold equality."""

    triggered = context.loc[
        context["context_valid"].astype(bool)
        & context["recent_market_relative"].gt(0.0)
        & context["recent_sector_relative"].gt(0.0)
        & context["event_strength"].ge(threshold)
    ].copy()
    return triggered.sort_values(["session", "bar_end", "symbol"], kind="mergesort").reset_index(
        drop=True
    )


def _stable_id(prefix: str, *values: Any) -> str:
    identity = "|".join(str(value) for value in values)
    return f"{prefix}_{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


@dataclass(frozen=True)
class EventDeduplicationResult:
    """First-event population plus explicit trigger accounting."""

    primary_events: pd.DataFrame
    diagnostics: pd.DataFrame
    raw_trigger_count: int
    first_event_count: int
    later_trigger_count: int
    grid_assignment_count: int
    rejected_count: int


def deduplicate_and_assign_events(triggers: pd.DataFrame) -> EventDeduplicationResult:
    """Keep the first E1 per stock/session and assign it to a valid later grid."""

    _require_columns(
        triggers,
        {"symbol", "session", "bar_end", "feature_availability_time", "session_close"},
    )
    frame = triggers.copy()
    for column in ("session", "bar_end", "feature_availability_time", "session_close"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["causal_confirmation_time"] = frame[["bar_end", "feature_availability_time"]].max(axis=1)
    frame = frame.sort_values(
        ["session", "symbol", "causal_confirmation_time", "bar_end"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["trigger_ordinal"] = frame.groupby(["session", "symbol"], sort=True).cumcount() + 1
    frame["deduplication_status"] = np.where(
        frame["trigger_ordinal"].eq(1), "first_event", "later_trigger_diagnostic"
    )
    frame["grid_assignment_rejection_reason"] = np.where(
        frame["trigger_ordinal"].eq(1), "", "later_trigger_not_primary"
    )
    first = frame.loc[frame["trigger_ordinal"].eq(1)].copy()
    assigned_rows: list[pd.Series] = []
    for row_index, row in first.iterrows():
        decision = assign_decision_time(
            confirmation_time=row["bar_end"].to_pydatetime(),
            availability_time=row["feature_availability_time"].to_pydatetime(),
            exact_grid_available_before_scoring=bool(
                row.get("exact_grid_available_before_scoring", False)
            ),
        )
        reason = ""
        if decision is None:
            reason = "no_subsequent_decision_grid"
        else:
            references = target_reference_times(decision)
            if pd.Timestamp(references.exit_60m) > row["session_close"]:
                reason = "primary_outcome_interval_exceeds_session"
            else:
                row["source_bar_end"] = row["bar_end"]
                row["event_confirmation_time"] = row["causal_confirmation_time"]
                row["assigned_decision_time"] = pd.Timestamp(decision)
                row["planned_entry_reference_time"] = pd.Timestamp(
                    references.delayed_entry_reference
                )
                row["planned_exit_reference_time"] = pd.Timestamp(references.exit_60m)
                row["event_id"] = _stable_id(
                    "evt", row["symbol"], row["session"].isoformat(), row["bar_end"].isoformat()
                )
                row["slate_id"] = _stable_id("slate", pd.Timestamp(decision).isoformat())
                assigned_rows.append(row)
        frame.at[row_index, "grid_assignment_rejection_reason"] = reason
    primary = pd.DataFrame(assigned_rows)
    if not primary.empty:
        primary = primary.sort_values(
            ["assigned_decision_time", "symbol"], kind="mergesort"
        ).reset_index(drop=True)
    return EventDeduplicationResult(
        primary_events=primary,
        diagnostics=frame,
        raw_trigger_count=len(frame),
        first_event_count=len(first),
        later_trigger_count=int(frame["trigger_ordinal"].gt(1).sum()),
        grid_assignment_count=len(primary),
        rejected_count=len(first) - len(primary),
    )
