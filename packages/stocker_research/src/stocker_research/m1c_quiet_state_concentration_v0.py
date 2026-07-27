"""Pure concentration-audit helpers for the frozen causal M1C quiet state.

This module is retrospective research only.  It neither changes the frozen
M1C model nor exposes a broker, order, account, position, or execution seam.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd

BOTTOM_5_THRESHOLD: Final[float] = 0.115697407847643
BOTTOM_10_THRESHOLD: Final[float] = 0.135896965695626
BOTTOM_20_THRESHOLD: Final[float] = 0.167095528962669
HIGH_TAIL_THRESHOLD: Final[float] = 0.488333710794033
ORIGINAL_DECISION: Final[str] = "blocked_insufficient_low_tail_support"
MAXIMUM_RUN_GAP_MINUTES: Final[int] = 15
MINIMUM_EPISODE_SPACING_MINUTES: Final[int] = 30
PROTECTED_HISTORICAL_START: Final[str] = "2026-01-01"

MonthExplanation = Literal[
    "month_concentration_driven_by_source_exposure",
    "month_concentration_driven_by_higher_low_tail_incidence",
    "month_concentration_driven_by_checkpoint_persistence",
    "month_concentration_has_multiple_causes",
    "month_concentration_unexplained",
]
SurpriseExplanation = Literal[
    "surprise_concentration_disperses_after_event_clustering",
    "surprise_concentration_persists_by_stock",
    "surprise_concentration_persists_by_month",
    "surprise_concentration_persists_by_stock_month",
    "surprise_concentration_is_small_count_fragile",
    "surprise_concentration_unexplained",
]
WeightScheme = Literal[
    "original",
    "equal_month",
    "equal_stock",
    "equal_stock_month",
]


def audit_claims() -> dict[str, bool | float | str]:
    """Return the binding claims boundary for every V0 audit artifact."""

    return {
        "research_only": True,
        "original_low_movement_decision_preserved": True,
        "original_decision": ORIGINAL_DECISION,
        "retrospective_gate_relaxation_allowed": False,
        "m1c_frozen": True,
        "m1c_bottom_5_threshold": BOTTOM_5_THRESHOLD,
        "m1c_bottom_10_threshold": BOTTOM_10_THRESHOLD,
        "m1c_bottom_20_threshold": BOTTOM_20_THRESHOLD,
        "primary_quiet_state": "bottom_10_percent",
        "prospective_record_only": True,
        "option_shadow_outcomes_only": True,
        "defined_risk_short_premium_only": True,
        "naked_short_options_allowed": False,
        "paper_orders_allowed": False,
        "live_orders_allowed": False,
        "broker_order_methods_allowed": False,
        "strategy_promotion": False,
        "protected_historical_start": PROTECTED_HISTORICAL_START,
    }


def _require_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _timestamps(frame: pd.DataFrame, column: str = "entry_timestamp") -> pd.Series:
    parsed = pd.to_datetime(frame[column], errors="raise", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{column} contains a missing timestamp")
    return parsed


def _event_identity(prefix: str, *parts: object) -> str:
    payload = "|".join((prefix, *(str(part) for part in parts)))
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def reconstruct_frozen_tail(
    predictions: pd.DataFrame,
    original_tail: pd.DataFrame,
) -> dict[str, int | float | bool]:
    """Reconstruct inclusive bottom-10 membership against original row evidence."""

    _require_columns(
        predictions,
        {"row_id", "M1C_probability"},
        label="checkpoint predictions",
    )
    _require_columns(
        original_tail,
        {"row_id", "M1C_probability"},
        label="original low-tail rows",
    )
    if predictions["row_id"].astype(str).duplicated().any():
        raise ValueError("checkpoint row identities must be unique")
    if original_tail["row_id"].astype(str).duplicated().any():
        raise ValueError("original low-tail row identities must be unique")
    probabilities = pd.to_numeric(predictions["M1C_probability"], errors="raise")
    original_probabilities = pd.to_numeric(original_tail["M1C_probability"], errors="raise")
    if (
        not np.isfinite(probabilities.to_numpy(float)).all()
        or not np.isfinite(original_probabilities.to_numpy(float)).all()
    ):
        raise ValueError("M1C probabilities must be finite")
    reconstructed = predictions.loc[
        probabilities.le(BOTTOM_10_THRESHOLD),
        ["row_id", "M1C_probability"],
    ].copy()
    expected_ids = set(original_tail["row_id"].astype(str))
    reconstructed_ids = set(reconstructed["row_id"].astype(str))
    identity_mismatches = len(expected_ids.symmetric_difference(reconstructed_ids))
    comparison = original_tail.merge(
        reconstructed,
        on="row_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_original", "_reconstructed"),
    )
    matched = comparison.loc[comparison["_merge"].eq("both")]
    maximum_difference = (
        0.0
        if matched.empty
        else float(
            np.max(
                np.abs(
                    matched["M1C_probability_original"].to_numpy(float)
                    - matched["M1C_probability_reconstructed"].to_numpy(float)
                )
            )
        )
    )
    membership_column_mismatches = 0
    if "m1c_bottom_10_percent" in predictions:
        observed = predictions["m1c_bottom_10_percent"].astype(bool).to_numpy()
        calculated = probabilities.le(BOTTOM_10_THRESHOLD).to_numpy()
        membership_column_mismatches = int(np.count_nonzero(observed != calculated))
    membership_mismatches = identity_mismatches + membership_column_mismatches
    return {
        "row_identity_mismatches": identity_mismatches,
        "maximum_m1c_probability_difference": maximum_difference,
        "tail_membership_mismatches": membership_mismatches,
        "passed": (
            identity_mismatches == 0 and maximum_difference <= 1e-12 and membership_mismatches == 0
        ),
    }


def cluster_quiet_state_runs(
    eligible_rows: pd.DataFrame,
    *,
    threshold: float = BOTTOM_10_THRESHOLD,
) -> pd.DataFrame:
    """Cluster consecutive eligible bottom-tail checkpoints into quiet runs."""

    _require_columns(
        eligible_rows,
        {
            "row_id",
            "stock",
            "session",
            "checkpoint",
            "entry_timestamp",
            "M1C_probability",
        },
        label="quiet-run input",
    )
    if abs(float(threshold) - BOTTOM_10_THRESHOLD) > 0.0:
        raise ValueError("the bottom-10 M1C threshold is frozen")
    ordered = eligible_rows.copy()
    ordered["entry_timestamp"] = _timestamps(ordered)
    ordered["_quiet"] = pd.to_numeric(
        ordered["M1C_probability"],
        errors="raise",
    ).le(BOTTOM_10_THRESHOLD)
    ordered = ordered.sort_values(
        ["stock", "session", "entry_timestamp", "checkpoint", "row_id"],
        kind="mergesort",
    )
    if ordered.duplicated(["stock", "session", "checkpoint"]).any():
        raise ValueError("eligible checkpoint identities must be unique")

    output: list[dict[str, Any]] = []

    def emit(members: list[pd.Series]) -> None:
        if not members:
            return
        first = members[0]
        last = members[-1]
        record = first.to_dict()
        record.pop("_quiet", None)
        member_ids = tuple(str(member["row_id"]) for member in members)
        record.update(
            {
                "quiet_run_id": _event_identity(
                    "quiet-run",
                    first["stock"],
                    first["session"],
                    first["row_id"],
                ),
                "trigger_row_id": str(first["row_id"]),
                "trigger_checkpoint": int(cast(Any, first["checkpoint"])),
                "run_start_timestamp": pd.Timestamp(first["entry_timestamp"]),
                "run_end_timestamp": pd.Timestamp(last["entry_timestamp"]),
                "last_checkpoint": int(cast(Any, last["checkpoint"])),
                "member_count": len(members),
                "member_row_ids": member_ids,
                "multiple_checkpoint_rows": len(members) > 1,
            }
        )
        output.append(record)

    for _, group in ordered.groupby(["stock", "session"], sort=True):
        active: list[pd.Series] = []
        previous_row: pd.Series | None = None
        for _, row in group.iterrows():
            quiet = bool(row["_quiet"])
            continues = False
            if quiet and active and previous_row is not None and bool(previous_row["_quiet"]):
                gap = (
                    pd.Timestamp(row["entry_timestamp"])
                    - pd.Timestamp(previous_row["entry_timestamp"])
                ).total_seconds() / 60.0
                continues = 0.0 <= gap <= MAXIMUM_RUN_GAP_MINUTES
            if quiet:
                if not continues:
                    emit(active)
                    active = []
                active.append(row)
            else:
                emit(active)
                active = []
            previous_row = row
        emit(active)
    return pd.DataFrame(output)


def fresh_quiet_episodes(
    eligible_rows: pd.DataFrame,
    *,
    threshold: float = BOTTOM_10_THRESHOLD,
) -> pd.DataFrame:
    """Apply the frozen downward crossing and thirty-minute spacing rule."""

    _require_columns(
        eligible_rows,
        {
            "row_id",
            "stock",
            "session",
            "checkpoint",
            "entry_timestamp",
            "M1C_probability",
        },
        label="fresh-episode input",
    )
    if abs(float(threshold) - BOTTOM_10_THRESHOLD) > 0.0:
        raise ValueError("the bottom-10 M1C threshold is frozen")
    ordered = eligible_rows.copy()
    ordered["entry_timestamp"] = _timestamps(ordered)
    ordered = ordered.sort_values(
        ["stock", "session", "entry_timestamp", "checkpoint", "row_id"],
        kind="mergesort",
    )
    if ordered.duplicated(["stock", "session", "checkpoint"]).any():
        raise ValueError("eligible checkpoint identities must be unique")
    output: list[dict[str, Any]] = []
    for (stock, session), group in ordered.groupby(["stock", "session"], sort=True):
        previous_probability: float | None = None
        previous_episode: pd.Timestamp | None = None
        episode_number = 0
        for _, row in group.iterrows():
            probability = float(cast(Any, row["M1C_probability"]))
            if not math.isfinite(probability):
                raise ValueError("M1C probabilities must be finite")
            crossing = probability <= BOTTOM_10_THRESHOLD and (
                previous_probability is None or previous_probability > BOTTOM_10_THRESHOLD
            )
            timestamp = pd.Timestamp(row["entry_timestamp"])
            elapsed = (
                math.nan
                if previous_episode is None
                else (timestamp - previous_episode).total_seconds() / 60.0
            )
            if crossing and (not math.isfinite(elapsed) or elapsed >= 30.0):
                episode_number += 1
                record = row.to_dict()
                record.update(
                    {
                        "quiet_episode_id": _event_identity(
                            "quiet",
                            stock,
                            session,
                            row["checkpoint"],
                            timestamp.isoformat(),
                        ),
                        "trigger_checkpoint": int(cast(Any, row["checkpoint"])),
                        "trigger_timestamp": timestamp,
                        "prospective_entry_timestamp": timestamp,
                        "previous_m1c_probability": previous_probability,
                        "current_m1c_probability": probability,
                        "episode_number": episode_number,
                        "minutes_since_previous_quiet_episode": elapsed,
                    }
                )
                output.append(record)
                previous_episode = timestamp
            previous_probability = probability
    return pd.DataFrame(output)


def cluster_surprise_events(
    rows: pd.DataFrame,
    *,
    sigma_threshold: float,
) -> pd.DataFrame:
    """Cluster overlapping 15-minute surprise excursions within stock-session."""

    if sigma_threshold not in {1.5, 2.0}:
        raise ValueError("surprise threshold must be 1.5 or 2.0 sigma")
    _require_columns(
        rows,
        {
            "row_id",
            "stock",
            "session",
            "checkpoint",
            "entry_timestamp",
            "excursion_sigma_ratio_15m",
        },
        label="surprise-event input",
    )
    working = rows.copy()
    working["entry_timestamp"] = _timestamps(working)
    ratios = pd.to_numeric(working["excursion_sigma_ratio_15m"], errors="raise")
    working = working.loc[ratios.ge(sigma_threshold)].sort_values(
        ["stock", "session", "entry_timestamp", "checkpoint", "row_id"],
        kind="mergesort",
    )
    output: list[dict[str, Any]] = []

    def emit(members: list[pd.Series]) -> None:
        if not members:
            return
        first = members[0]
        ratios_array = np.asarray(
            [float(cast(Any, member["excursion_sigma_ratio_15m"])) for member in members],
            dtype=float,
        )
        maximum_index = int(np.argmax(ratios_array))
        maximum_row = members[maximum_index]
        record = maximum_row.to_dict()
        member_ids = tuple(str(member["row_id"]) for member in members)
        first_timestamp = pd.Timestamp(first["entry_timestamp"])
        last_timestamp = pd.Timestamp(members[-1]["entry_timestamp"])
        record.update(
            {
                "surprise_event_id": _event_identity(
                    f"surprise-{str(sigma_threshold).replace('.', '_')}",
                    first["stock"],
                    first["session"],
                    first["row_id"],
                ),
                "trigger_row_id": str(first["row_id"]),
                "trigger_checkpoint": int(cast(Any, first["checkpoint"])),
                "trigger_timestamp": first_timestamp,
                "last_trigger_timestamp": last_timestamp,
                "member_count": len(members),
                "member_row_ids": member_ids,
                "multiple_checkpoint_rows": len(members) > 1,
                "maximum_excursion_sigma_ratio": float(np.max(ratios_array)),
                "extreme_surprise_mover": bool(np.max(ratios_array) >= 2.0),
                "excursion_window_end": max(
                    pd.Timestamp(member["entry_timestamp"]) + pd.Timedelta(minutes=15)
                    for member in members
                ),
            }
        )
        output.append(record)

    for _, group in working.groupby(["stock", "session"], sort=True):
        active: list[pd.Series] = []
        active_window_end: pd.Timestamp | None = None
        previous_trigger: pd.Timestamp | None = None
        for _, row in group.iterrows():
            trigger = pd.Timestamp(row["entry_timestamp"])
            overlaps = (
                bool(active)
                and previous_trigger is not None
                and active_window_end is not None
                and (trigger - previous_trigger).total_seconds() / 60.0 <= 30.0
                and trigger <= active_window_end
            )
            if not overlaps:
                emit(active)
                active = []
                active_window_end = None
            active.append(row)
            row_window_end = trigger + pd.Timedelta(minutes=15)
            active_window_end = (
                row_window_end
                if active_window_end is None
                else max(active_window_end, row_window_end)
            )
            previous_trigger = trigger
        emit(active)
    return pd.DataFrame(output)


def analysis_weights(
    frame: pd.DataFrame,
    *,
    scheme: WeightScheme,
) -> pd.Series:
    """Return deterministic normalized descriptive weights."""

    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    base = (
        pd.to_numeric(frame["row_weight"], errors="raise")
        if "row_weight" in frame
        else pd.Series(1.0, index=frame.index)
    )
    if not np.isfinite(base.to_numpy(float)).all() or bool(base.le(0.0).any()):
        raise ValueError("descriptive base weights must be finite and positive")
    if scheme == "original":
        result = base / float(base.sum())
    else:
        group_columns = {
            "equal_month": ["month"],
            "equal_stock": ["stock"],
            "equal_stock_month": ["stock", "month"],
        }.get(scheme)
        if group_columns is None:
            raise ValueError(f"unknown descriptive weighting scheme: {scheme}")
        _require_columns(frame, set(group_columns), label=f"{scheme} input")
        group_mass = (
            frame.assign(_base_weight=base)
            .groupby(group_columns, sort=True, dropna=False)["_base_weight"]
            .transform("sum")
        )
        represented_groups = frame.groupby(
            group_columns,
            sort=True,
            dropna=False,
        ).ngroups
        result = base / group_mass / float(represented_groups)
    return pd.Series(result.to_numpy(float), index=frame.index, dtype=float)


def small_count_feasibility(
    *,
    month_counts: Mapping[str, int],
    stock_counts: Mapping[str, int],
    month_limit: float,
    stock_limit: float,
    total_months: int = 4,
    total_stocks: int = 20,
) -> dict[str, int | float | bool]:
    """Diagnose concentration discreteness without relaxing a frozen gate."""

    months = {str(key): int(value) for key, value in month_counts.items()}
    stocks = {str(key): int(value) for key, value in stock_counts.items()}
    if any(value < 0 for value in (*months.values(), *stocks.values())):
        raise ValueError("event counts cannot be negative")
    event_count = sum(months.values())
    if event_count != sum(stocks.values()):
        raise ValueError("month and stock allocations must describe the same events")
    if not 0.0 < month_limit <= 1.0 or not 0.0 < stock_limit <= 1.0:
        raise ValueError("concentration limits must lie in (0, 1]")
    maximum_month_count = max(months.values(), default=0)
    maximum_stock_count = max(stocks.values(), default=0)
    maximum_month_share = maximum_month_count / event_count if event_count else 0.0
    maximum_stock_share = maximum_stock_count / event_count if event_count else 0.0
    one_event_share = 1.0 / event_count if event_count else 0.0
    month_capacity = math.floor(month_limit * event_count + 1e-12)
    stock_capacity = math.floor(stock_limit * event_count + 1e-12)
    month_excess = max(0, maximum_month_count - month_capacity)
    stock_excess = max(0, maximum_stock_count - stock_capacity)
    driven_by_one_or_two = 0 < max(month_excess, stock_excess) <= 2
    fragile = bool(event_count and (one_event_share > 0.05 or driven_by_one_or_two))
    return {
        "event_count": event_count,
        "largest_observed_allocation_count": max(maximum_month_count, maximum_stock_count),
        "largest_possible_concentration_under_observed_allocation": max(
            maximum_month_share,
            maximum_stock_share,
        ),
        "observed_maximum_month_share": maximum_month_share,
        "observed_maximum_stock_share": maximum_stock_share,
        "minimum_theoretical_maximum_month_share": (
            math.ceil(event_count / total_months) / event_count if event_count else 0.0
        ),
        "minimum_theoretical_maximum_stock_share": (
            math.ceil(event_count / total_stocks) / event_count if event_count else 0.0
        ),
        "one_event_share": one_event_share,
        "one_event_changes_concentration_by_more_than_five_points": one_event_share > 0.05,
        "events_over_month_limit": month_excess,
        "events_over_stock_limit": stock_excess,
        "failed_condition_driven_by_one_or_two_events": driven_by_one_or_two,
        "small_count_concentration_fragile": fragile,
    }


def classify_month_concentration(
    *,
    maximum_composition_share: float,
    source_exposure_share: float,
    fresh_episode_share: float,
    frozen_limit: float,
) -> MonthExplanation:
    """Choose exactly one frozen explanatory category."""

    values = (
        maximum_composition_share,
        source_exposure_share,
        fresh_episode_share,
        frozen_limit,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        return "month_concentration_unexplained"
    exposure_fails = source_exposure_share > frozen_limit
    persistence_is_binding = maximum_composition_share > frozen_limit >= fresh_episode_share
    incidence_lift = maximum_composition_share - source_exposure_share
    higher_incidence_is_material = incidence_lift >= 0.02
    causes = sum((exposure_fails, persistence_is_binding, higher_incidence_is_material))
    if causes >= 2 or (persistence_is_binding and source_exposure_share >= 0.25):
        return "month_concentration_has_multiple_causes"
    if exposure_fails and abs(incidence_lift) < 0.02:
        return "month_concentration_driven_by_source_exposure"
    if persistence_is_binding:
        return "month_concentration_driven_by_checkpoint_persistence"
    if higher_incidence_is_material:
        return "month_concentration_driven_by_higher_low_tail_incidence"
    return "month_concentration_unexplained"


def classify_surprise_concentration(
    *,
    clustered_maximum_stock_share: float,
    clustered_maximum_month_share: float,
    clustered_maximum_stock_month_share: float,
    stock_limit: float,
    month_limit: float,
    small_count_fragile: bool,
) -> SurpriseExplanation:
    """Choose exactly one frozen surprise-concentration explanation."""

    if small_count_fragile and (
        clustered_maximum_stock_share > stock_limit
        or clustered_maximum_month_share > month_limit
        or clustered_maximum_stock_month_share > max(stock_limit, month_limit)
    ):
        return "surprise_concentration_is_small_count_fragile"
    if clustered_maximum_stock_month_share > max(stock_limit, month_limit):
        return "surprise_concentration_persists_by_stock_month"
    if clustered_maximum_stock_share > stock_limit:
        return "surprise_concentration_persists_by_stock"
    if clustered_maximum_month_share > month_limit:
        return "surprise_concentration_persists_by_month"
    if max(
        clustered_maximum_stock_share,
        clustered_maximum_month_share,
        clustered_maximum_stock_month_share,
    ) <= max(stock_limit, month_limit):
        return "surprise_concentration_disperses_after_event_clustering"
    return "surprise_concentration_unexplained"


__all__ = [
    "BOTTOM_5_THRESHOLD",
    "BOTTOM_10_THRESHOLD",
    "BOTTOM_20_THRESHOLD",
    "HIGH_TAIL_THRESHOLD",
    "ORIGINAL_DECISION",
    "analysis_weights",
    "audit_claims",
    "classify_month_concentration",
    "classify_surprise_concentration",
    "cluster_quiet_state_runs",
    "cluster_surprise_events",
    "fresh_quiet_episodes",
    "reconstruct_frozen_tail",
    "small_count_feasibility",
]
