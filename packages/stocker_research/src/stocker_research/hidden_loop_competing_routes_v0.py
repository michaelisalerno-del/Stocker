"""Structural helpers for the hidden-loop competing-routes quick screen V0.

This module is retrospective research infrastructure only.  It exposes no
price-return, direction, order, broker, execution, deployment, or production
runtime surface.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TARGET_A: Final[str] = "loop_p_2-5-6-2"
TARGET_B: Final[str] = "loop_p_2-6-2"
TARGET_C: Final[str] = "loop_p_4-6-4"
PREREGISTERED_TARGETS: Final[tuple[str, ...]] = (TARGET_A, TARGET_B, TARGET_C)

HIDDEN_A: Final[str] = "unregistered_primitive_like__5-6-5"
HIDDEN_B: Final[str] = "unregistered_primitive_like__2-3-2"
HIDDEN_C: Final[str] = "unregistered_primitive_like__2-5-2"
HIDDEN_D: Final[str] = "unregistered_primitive_like__4-7-4"
HIDDEN_OTHER: Final[str] = "OTHER_UNREGISTERED_FAMILY"
FROZEN_HIDDEN_FAMILIES: Final[tuple[str, ...]] = (
    HIDDEN_A,
    HIDDEN_B,
    HIDDEN_C,
    HIDDEN_D,
    HIDDEN_OTHER,
)

NO_REGISTERED_COMPLETION: Final[str] = "NO_REGISTERED_COMPLETION"
OTHER_REGISTERED_COMPLETION: Final[str] = "OTHER_REGISTERED_COMPLETION"
PROTECTED_START: Final[pd.Timestamp] = pd.Timestamp("2025-08-23T00:00:00Z")
RANDOM_STATE: Final[int] = 20260722

SAFETY_FLAGS: Final[dict[str, bool | str]] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "target_specific_registered_routes": True,
    "hidden_family_competing_route_test": True,
    "registered_loop_recurrence_test": True,
    "sequential_causal_updates": True,
    "economic_outcomes_opened": False,
    "directional_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}

STATUS_VALUES: Final[frozenset[str]] = frozenset(
    {"supported", "descriptive_only", "not_supported", "insufficient_support"}
)

PRIMARY_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "target_specific_hidden_routes_and_registered_recurrence_supported",
        "hidden_target_specific_routes_supported_only",
        "registered_recurrence_supported_only",
        "hidden_diversion_supported_only",
        "descriptive_competing_route_structure_only",
        "no_competing_route_increment",
        "blocked_predecessor_population_not_reconstructable",
        "blocked_exact_target_support_failure",
        "blocked_transition_census_support_failure",
        "blocked_sequential_model_support_failure",
        "blocked_protected_boundary_failure",
        "blocked_chronology_or_leakage_failure",
        "blocked_quick_competing_route_resource_limit",
        "blocked_model_convergence_failure",
        "blocked_reproducibility_or_audit_failure",
    }
)


def reject_protected_dates(frame: pd.DataFrame, *, column: str = "session") -> None:
    """Fail if a materialised row reaches the protected boundary."""

    if column not in frame.columns:
        raise ValueError(f"protected-date column is missing: {column}")
    values = pd.to_datetime(frame[column], utc=True, errors="raise")
    if bool(values.ge(PROTECTED_START).any()):
        raise ValueError("protected rows were materialised")


def candidate_threshold(probabilities: pd.Series | np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Return the frozen linear 80th percentile candidate threshold."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("candidate probabilities must be a finite non-empty vector")
    return float(np.quantile(values, 0.80, method="linear"))


def stable_frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash selected columns after a stable lexical sort."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"hash columns are missing: {missing}")
    ordered = frame.loc[:, list(columns)].astype(str).sort_values(list(columns), kind="mergesort")
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


def deduplicate_registered_completions(completions: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate the frozen ledger without collapsing orientation metadata."""

    identity = [
        "symbol",
        "session",
        "completion_timestamp_utc",
        "semantic_loop_id",
        "orientation_id",
    ]
    required = {
        *identity,
        "completion_bar_ordinal",
        "completion_available_timestamp_utc",
        "motif_type",
    }
    missing = sorted(required.difference(completions.columns))
    if missing:
        raise ValueError(f"registered completion fields are missing: {missing}")
    result = completions.copy()
    result["completion_timestamp_utc"] = pd.to_datetime(
        result["completion_timestamp_utc"], utc=True, errors="raise"
    )
    result["completion_available_timestamp_utc"] = pd.to_datetime(
        result["completion_available_timestamp_utc"], utc=True, errors="raise"
    )
    result = result.sort_values(
        [*identity, "completion_bar_ordinal"], kind="mergesort"
    ).drop_duplicates(identity, keep="first")
    result["year"] = pd.to_datetime(result["session"], errors="raise").dt.year
    result["year_month"] = result["session"].astype(str).str[:7]
    result["clock_bin"] = (
        result["completion_timestamp_utc"]
        .dt.tz_convert("America/New_York")
        .dt.floor("30min")
        .dt.strftime("%H:%M")
    )
    result["event_id"] = (
        result["symbol"].astype(str)
        + "|"
        + result["session"].astype(str)
        + "|"
        + result["completion_timestamp_utc"].astype(str)
        + "|"
        + result["semantic_loop_id"].astype(str)
        + "|"
        + result["orientation_id"].astype(str)
    )
    reject_protected_dates(result)
    if result.duplicated(identity).any() or result["event_id"].duplicated().any():
        raise AssertionError("registered completion identities remain duplicated")
    return result.sort_values(
        ["session", "completion_timestamp_utc", "symbol", "semantic_loop_id", "orientation_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def freeze_target_class_mapping(development_completions: pd.DataFrame) -> dict[str, Any]:
    """Freeze exact target support using development data only."""

    required = {"semantic_loop_id", "session", "symbol", "year_month"}
    missing = sorted(required.difference(development_completions.columns))
    if missing:
        raise ValueError(f"target support fields are missing: {missing}")
    support: dict[str, dict[str, Any]] = {}
    retained: list[str] = []
    for target in PREREGISTERED_TARGETS:
        group = development_completions.loc[
            development_completions["semantic_loop_id"].astype(str).eq(target)
        ]
        stock_share = group["symbol"].value_counts(normalize=True)
        values = {
            "outcomes": int(len(group)),
            "sessions": int(group["session"].nunique()),
            "stocks": int(group["symbol"].nunique()),
            "months": int(group["year_month"].nunique()),
            "maximum_stock_share": float(stock_share.max()) if not stock_share.empty else 1.0,
        }
        checks = {
            "outcomes_at_least_50": values["outcomes"] >= 50,
            "sessions_at_least_30": values["sessions"] >= 30,
            "stocks_at_least_8": values["stocks"] >= 8,
            "months_at_least_4": values["months"] >= 4,
            "maximum_stock_share_at_most_0_30": values["maximum_stock_share"] <= 0.30,
        }
        passed = all(checks.values())
        support[target] = {**values, "checks": checks, "supported": passed}
        if passed:
            retained.append(target)
    classes = [NO_REGISTERED_COMPLETION, OTHER_REGISTERED_COMPLETION, *retained]
    return {
        **SAFETY_FLAGS,
        "fit_period": "2024_only",
        "assessment_support_inspected_before_freeze": False,
        "preregistered_targets": list(PREREGISTERED_TARGETS),
        "support": support,
        "retained_exact_targets": retained,
        "pooled_exact_targets": [
            target for target in PREREGISTERED_TARGETS if target not in retained
        ],
        "final_target_classes": classes,
        "at_least_one_exact_target_supported": bool(retained),
        "at_least_three_final_classes": len(classes) >= 3,
    }


def map_registered_route(semantic_loop_id: str | None, retained_targets: Sequence[str]) -> str:
    """Map one causal next completion into the development-frozen classes."""

    if semantic_loop_id is None:
        return NO_REGISTERED_COMPLETION
    identity = str(semantic_loop_id)
    return identity if identity in set(retained_targets) else OTHER_REGISTERED_COMPLETION


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted p-values."""

    values = np.asarray(p_values, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or bool(((values < 0.0) | (values > 1.0)).any())
    ):
        raise ValueError("p-values must be a finite one-dimensional vector in [0, 1]")
    if values.size == 0:
        return []
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return cast(list[float], output.tolist())


def transition_hypothesis_manifest() -> list[dict[str, Any]]:
    """Return the four fixed transition hypotheses without data-dependent additions."""

    return [
        {
            "hypothesis_id": "H1",
            "precursor_kind": "hidden",
            "precursor_identity": HIDDEN_A,
            "target_identity": TARGET_A,
            "lookbacks": [12],
            "expected_sign": "positive",
        },
        {
            "hypothesis_id": "H2",
            "precursor_kind": "hidden",
            "precursor_identity": HIDDEN_A,
            "target_identity": TARGET_B,
            "lookbacks": [12],
            "expected_sign": "positive",
        },
        {
            "hypothesis_id": "H3",
            "precursor_kind": "registered",
            "precursor_identity": TARGET_C,
            "target_identity": TARGET_C,
            "lookbacks": [6, 12],
            "expected_sign": "positive",
        },
        {
            "hypothesis_id": "H4",
            "precursor_kind": "hidden",
            "precursor_identity": HIDDEN_B,
            "target_identity": "ANY_REGISTERED_COMPLETION",
            "lookbacks": [6],
            "expected_sign": "negative",
        },
    ]


def lookback_is_complete(available_ordinals: Sequence[int], completion: int, lookback: int) -> bool:
    """Return whether exactly the required strict prior bars are available."""

    if lookback not in (6, 12):
        raise ValueError("lookback must be six or twelve bars")
    available = {int(value) for value in available_ordinals}
    return set(range(int(completion) - lookback, int(completion))).issubset(available)


def precursor_present(
    *,
    completion_bar_ordinal: int,
    lookback_bars: int,
    precursor_kind: str,
    precursor_identity: str,
    registered_events: pd.DataFrame,
    hidden_events: pd.DataFrame,
) -> bool:
    """Evaluate one fixed strict-pre-completion precursor predicate."""

    start = int(completion_bar_ordinal) - int(lookback_bars)
    end = int(completion_bar_ordinal) - 1
    if precursor_kind == "registered":
        if not {"completion_bar_ordinal", "semantic_loop_id"}.issubset(registered_events.columns):
            raise ValueError("registered precursor fields are missing")
        ordinal = pd.to_numeric(registered_events["completion_bar_ordinal"], errors="raise")
        return bool(
            (
                ordinal.between(start, end)
                & registered_events["semantic_loop_id"].astype(str).eq(precursor_identity)
            ).any()
        )
    if precursor_kind == "hidden":
        if not {"completion_bar_ordinal", "hidden_family_class"}.issubset(hidden_events.columns):
            raise ValueError("hidden precursor fields are missing")
        ordinal = pd.to_numeric(hidden_events["completion_bar_ordinal"], errors="raise")
        return bool(
            (
                ordinal.between(start, end)
                & hidden_events["hidden_family_class"].astype(str).eq(precursor_identity)
            ).any()
        )
    raise ValueError(f"unknown precursor kind: {precursor_kind}")


def sample_matched_pseudo_completions(
    observed: pd.DataFrame,
    eligible_timestamps: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Sample stock/month/clock matches while excluding the source target identity."""

    observed_required = {
        "event_id",
        "symbol",
        "session",
        "year_month",
        "clock_bin",
        "semantic_loop_id",
        "completion_bar_ordinal",
        "completion_timestamp_utc",
    }
    eligible_required = {
        "symbol",
        "session",
        "year_month",
        "clock_bin",
        "completion_bar_ordinal",
        "completion_timestamp_utc",
        "full_prior_history",
        "semantic_loop_ids_at_timestamp",
    }
    missing_observed = sorted(observed_required.difference(observed.columns))
    missing_eligible = sorted(eligible_required.difference(eligible_timestamps.columns))
    if missing_observed or missing_eligible:
        raise ValueError(
            "matched pseudo-completion fields are missing: "
            f"observed={missing_observed}, eligible={missing_eligible}"
        )
    pool = eligible_timestamps.loc[eligible_timestamps["full_prior_history"].astype(bool)].copy()
    pool = pool.sort_values(
        ["symbol", "year_month", "clock_bin", "session", "completion_bar_ordinal"],
        kind="mergesort",
    ).reset_index(drop=True)
    pool["_semantic_loop_id_set"] = pool["semantic_loop_ids_at_timestamp"].map(
        lambda value: frozenset(str(item) for item in cast(Sequence[Any], value))
    )
    pool_groups = {
        (str(symbol), str(year_month), str(clock_bin)): group
        for (symbol, year_month, clock_bin), group in pool.groupby(
            ["symbol", "year_month", "clock_bin"], sort=False
        )
    }
    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for source in observed.sort_values("event_id", kind="mergesort").itertuples(index=False):
        key = (str(source.symbol), str(source.year_month), str(source.clock_bin))
        candidates = pool_groups.get(key, pool.iloc[:0])
        target = str(source.semantic_loop_id)
        candidates = candidates.loc[
            ~candidates["_semantic_loop_id_set"].map(
                lambda value, target=target: target in cast(frozenset[str], value)
            )
        ]
        if candidates.empty:
            continue
        sampled = candidates.iloc[int(generator.integers(0, len(candidates)))]
        rows.append(
            {
                "source_event_id": str(source.event_id),
                "source_symbol": str(source.symbol),
                "source_session": str(source.session),
                "source_completion_bar_ordinal": int(cast(Any, source.completion_bar_ordinal)),
                "source_completion_timestamp_utc": pd.Timestamp(
                    cast(Any, source.completion_timestamp_utc)
                ),
                "semantic_loop_id": target,
                "orientation_id": str(getattr(source, "orientation_id", "")),
                "symbol": str(sampled["symbol"]),
                "session": str(sampled["session"]),
                "year_month": str(sampled["year_month"]),
                "clock_bin": str(sampled["clock_bin"]),
                "completion_bar_ordinal": int(cast(Any, sampled["completion_bar_ordinal"])),
                "completion_timestamp_utc": pd.Timestamp(
                    cast(Any, sampled["completion_timestamp_utc"])
                ),
            }
        )
    return pd.DataFrame(rows)


def sequential_update_ordinals(
    *,
    opening_ordinal: int,
    first_completion_ordinal: int | None,
    available_ordinals: Sequence[int],
    horizon_bars: int = 12,
    maximum_elapsed: int = 6,
) -> tuple[int, ...]:
    """Build causal update ordinals, retaining the row containing first completion."""

    available = {int(value) for value in available_ordinals}
    horizon_end = int(opening_ordinal) + int(horizon_bars)
    stop = horizon_end
    if first_completion_ordinal is not None:
        stop = min(stop, int(first_completion_ordinal))
    rows = []
    for elapsed in range(maximum_elapsed + 1):
        ordinal = int(opening_ordinal) + elapsed
        if ordinal > stop:
            break
        if ordinal in available:
            rows.append(ordinal)
    return tuple(rows)


def next_registered_route(
    registered_events: pd.DataFrame,
    *,
    update_ordinal: int,
    horizon_end_ordinal: int,
    retained_targets: Sequence[str],
) -> tuple[str, str | None, int | None]:
    """Return the first strict-future completion inside the original horizon."""

    required = {"completion_bar_ordinal", "semantic_loop_id", "orientation_id"}
    missing = sorted(required.difference(registered_events.columns))
    if missing:
        raise ValueError(f"next-route fields are missing: {missing}")
    ordinal = pd.to_numeric(registered_events["completion_bar_ordinal"], errors="raise")
    eligible = registered_events.loc[
        ordinal.gt(int(update_ordinal)) & ordinal.le(int(horizon_end_ordinal))
    ].sort_values(
        ["completion_bar_ordinal", "semantic_loop_id", "orientation_id"], kind="mergesort"
    )
    if eligible.empty:
        return NO_REGISTERED_COMPLETION, None, None
    first = eligible.iloc[0]
    identity = str(first["semantic_loop_id"])
    return (
        map_registered_route(identity, retained_targets),
        identity,
        int(first["completion_bar_ordinal"]),
    )


def target_prefix_snapshot(
    prefix_history: pd.DataFrame,
    *,
    current_ordinal: int,
    target_identity: str,
    canonical_orientation_id: str,
    transition_length: int,
) -> dict[str, float]:
    """Summarise one exact target's causal active-prefix state."""

    required = {"bar_ordinal", "semantic_loop_id", "orientation_id", "progress_states"}
    missing = sorted(required.difference(prefix_history.columns))
    if missing:
        raise ValueError(f"prefix fields are missing: {missing}")
    current = prefix_history.loc[
        pd.to_numeric(prefix_history["bar_ordinal"], errors="raise").astype(int).eq(current_ordinal)
    ]
    target = current.loc[current["semantic_loop_id"].astype(str).eq(target_identity)]
    active = not target.empty
    depth = int(pd.to_numeric(target["progress_states"], errors="raise").max()) if active else 0
    target_active_ordinals = set(
        pd.to_numeric(
            prefix_history.loc[
                prefix_history["semantic_loop_id"].astype(str).eq(target_identity), "bar_ordinal"
            ],
            errors="raise",
        ).astype(int)
    )
    run_start = current_ordinal
    if active:
        while run_start - 1 in target_active_ordinals:
            run_start -= 1
    bars_since = current_ordinal - run_start if active else 0
    return {
        "active": float(active),
        "depth": float(depth),
        "fraction": float(depth / transition_length) if transition_length > 0 else 0.0,
        "bars_since_first_active": float(bars_since),
        "bars_since_first_active_missing": float(not active),
        "canonical_orientation_match": float(
            active & target["orientation_id"].astype(str).eq(canonical_orientation_id).any()
        ),
        "conflicting_prefix_active": float(
            current["semantic_loop_id"].astype(str).ne(target_identity).any()
        ),
    }


def registered_history_features(
    events: pd.DataFrame, *, opening_ordinal: int, current_ordinal: int
) -> dict[str, float]:
    """Calculate causal exact registered-loop history at one update."""

    required = {"completion_bar_ordinal", "semantic_loop_id"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"registered history fields are missing: {missing}")
    ordinal = pd.to_numeric(events["completion_bar_ordinal"], errors="raise").astype(int)
    observed = events.loc[ordinal.le(current_ordinal)].copy()
    observed_ordinal = pd.to_numeric(
        observed.get("completion_bar_ordinal", pd.Series(dtype=int)), errors="raise"
    ).astype(int)
    within_six = observed.loc[observed_ordinal.gt(current_ordinal - 6)]
    within_twelve = observed.loc[observed_ordinal.gt(current_ordinal - 12)]
    since_opening = observed.loc[observed_ordinal.gt(opening_ordinal)]
    latest = int(observed_ordinal.max()) if not observed.empty else current_ordinal
    missing_latest = observed.empty
    return {
        "prior_target_a_within_6": float(
            within_six["semantic_loop_id"].astype(str).eq(TARGET_A).any()
        ),
        "prior_target_b_within_6": float(
            within_six["semantic_loop_id"].astype(str).eq(TARGET_B).any()
        ),
        "prior_target_c_within_6": float(
            within_six["semantic_loop_id"].astype(str).eq(TARGET_C).any()
        ),
        "loop_p_4_6_4_completed_previous_6_bars": float(
            within_six["semantic_loop_id"].astype(str).eq(TARGET_C).any()
        ),
        "loop_p_4_6_4_completed_previous_12_bars": float(
            within_twelve["semantic_loop_id"].astype(str).eq(TARGET_C).any()
        ),
        "any_registered_loop_completed_since_opening": float(not since_opening.empty),
        "bars_since_latest_registered_completion": float(current_ordinal - latest)
        if not missing_latest
        else 0.0,
        "bars_since_latest_registered_completion_missing": float(missing_latest),
    }


def hidden_history_features(
    events: pd.DataFrame, *, opening_ordinal: int, current_ordinal: int
) -> dict[str, float]:
    """Calculate the complete frozen hidden-family history bundle at one update."""

    required = {"completion_bar_ordinal", "hidden_family_class"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"hidden history fields are missing: {missing}")
    ordinal = pd.to_numeric(events["completion_bar_ordinal"], errors="raise").astype(int)
    observed = events.loc[ordinal.gt(opening_ordinal) & ordinal.le(current_ordinal)].copy()
    aliases = {
        HIDDEN_A: "hidden_5_6_5",
        HIDDEN_B: "hidden_2_3_2",
        HIDDEN_C: "hidden_2_5_2",
        HIDDEN_D: "hidden_4_7_4",
        HIDDEN_OTHER: "hidden_other",
    }
    output: dict[str, float] = {}
    for identity, alias in aliases.items():
        group = observed.loc[observed["hidden_family_class"].astype(str).eq(identity)]
        seen = not group.empty
        latest = (
            int(pd.to_numeric(group["completion_bar_ordinal"], errors="raise").max())
            if seen
            else current_ordinal
        )
        output[f"{alias}_seen_since_opening"] = float(seen)
        output[f"bars_since_{alias}_completion"] = float(current_ordinal - latest) if seen else 0.0
        output[f"bars_since_{alias}_completion_missing"] = float(not seen)
    if observed.empty:
        most_recent = "NONE"
    else:
        recent = observed.sort_values(
            ["completion_bar_ordinal", "hidden_family_class"], kind="mergesort"
        ).iloc[-1]
        most_recent = str(recent["hidden_family_class"])
    output["any_hidden_event_since_opening"] = float(not observed.empty)
    output["hidden_event_count_since_opening"] = float(len(observed))
    for identity, alias in [*aliases.items(), ("NONE", "none")]:
        output[f"most_recent_hidden_family__{alias}"] = float(most_recent == identity)
    return output


def model_feature_sets(retained_targets: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Return the fixed nested C0/C1/C2 numeric feature ladder."""

    baseline = (
        "candidate_B0_probability",
        "checkpoint_12",
        "opening_transition_probability",
        "opening_posterior_entropy",
        "opening_top_state_probability",
        "opening_state_margin",
        *(f"elapsed_bar_{value}" for value in range(1, 7)),
        *(f"current_state_p_{value}" for value in range(8)),
        "current_posterior_entropy",
        "current_top_state_probability",
        "current_top_second_margin",
        "current_expected_state_age",
        "current_persistence_probability",
        "current_transition_probability",
        "regime_transitions_since_opening",
    )
    prefix: list[str] = []
    aliases = {TARGET_A: "target_a", TARGET_B: "target_b", TARGET_C: "target_c"}
    for target in retained_targets:
        alias = aliases[target]
        prefix.extend(
            [
                f"{alias}_prefix_active",
                f"{alias}_prefix_depth",
                f"{alias}_prefix_fraction",
                f"{alias}_prefix_bars_since_first_active",
                f"{alias}_prefix_bars_since_first_active_missing",
                f"{alias}_prefix_canonical_orientation_match",
                f"{alias}_conflicting_prefix_active",
            ]
        )
    c0 = (*baseline, *prefix)
    registered = (
        "prior_target_a_within_6",
        "prior_target_b_within_6",
        "prior_target_c_within_6",
        "loop_p_4_6_4_completed_previous_6_bars",
        "loop_p_4_6_4_completed_previous_12_bars",
        "any_registered_loop_completed_since_opening",
        "bars_since_latest_registered_completion",
        "bars_since_latest_registered_completion_missing",
    )
    hidden: list[str] = []
    for alias in ("hidden_5_6_5", "hidden_2_3_2", "hidden_2_5_2", "hidden_4_7_4", "hidden_other"):
        hidden.extend(
            [
                f"{alias}_seen_since_opening",
                f"bars_since_{alias}_completion",
                f"bars_since_{alias}_completion_missing",
            ]
        )
    hidden.extend(
        [
            "any_hidden_event_since_opening",
            "hidden_event_count_since_opening",
            "most_recent_hidden_family__hidden_5_6_5",
            "most_recent_hidden_family__hidden_2_3_2",
            "most_recent_hidden_family__hidden_2_5_2",
            "most_recent_hidden_family__hidden_4_7_4",
            "most_recent_hidden_family__hidden_other",
            "most_recent_hidden_family__none",
        ]
    )
    c1 = (*c0, *registered)
    return {"C0": tuple(c0), "C1": tuple(c1), "C2": tuple((*c1, *hidden))}


def candidate_normalised_weights(
    panel: pd.DataFrame,
    *,
    candidate_column: str = "candidate_id",
    total_weight_column: str = "candidate_total_weight",
) -> pd.Series:
    """Give every opening candidate its predecessor total weight."""

    required = {candidate_column, total_weight_column}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"candidate weighting fields are missing: {missing}")
    counts = panel.groupby(candidate_column, sort=False)[candidate_column].transform("size")
    candidate_weights = panel.groupby(candidate_column, sort=False)[total_weight_column].nunique()
    if bool(candidate_weights.ne(1).any()):
        raise ValueError("candidate total weight changes between sequential rows")
    return panel[total_weight_column].astype(float) / counts.astype(float)


def frozen_quantile_boundaries(values: Sequence[float], quantiles: Sequence[float]) -> list[float]:
    """Freeze deterministic linear quantile boundaries on development only."""

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("quantile values must be finite and non-empty")
    requested = np.asarray(quantiles, dtype=float)
    if bool(((requested <= 0.0) | (requested >= 1.0)).any()):
        raise ValueError("quantiles must lie strictly between zero and one")
    return cast(list[float], np.quantile(vector, requested, method="linear").tolist())


def assign_frozen_bin(values: Sequence[float], boundaries: Sequence[float]) -> np.ndarray[Any, Any]:
    """Assign one-based bins using development-frozen boundaries."""

    vector = np.asarray(values, dtype=float)
    cuts = np.asarray(boundaries, dtype=float)
    return np.searchsorted(cuts, vector, side="right") + 1


@dataclass(frozen=True, slots=True)
class FittedMultinomial:
    """Serializable deterministic standardisation plus multinomial model."""

    name: str
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercept: tuple[float, ...]
    iterations: tuple[int, ...]
    converged: bool

    def to_json(self) -> dict[str, Any]:
        """Return a stable JSON-compatible model specification."""

        return {
            "name": self.name,
            "feature_names": list(self.feature_names),
            "classes": list(self.classes),
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "coefficient": [list(row) for row in self.coefficients],
            "intercept": list(self.intercept),
            "iterations": list(self.iterations),
            "converged": self.converged,
            "model": {
                "penalty": "l2",
                "C": 0.25,
                "solver": "lbfgs",
                "max_iter": 300,
                "multi_class": "multinomial",
                "class_weight": None,
                "random_state": RANDOM_STATE,
                "n_jobs": 1,
            },
        }


def _feature_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray[Any, Any]:
    missing = sorted(set(feature_names).difference(frame.columns))
    if missing:
        raise ValueError(f"model features are missing: {missing}")
    matrix: np.ndarray[Any, Any] = np.asarray(
        frame.loc[:, list(feature_names)].apply(pd.to_numeric, errors="raise").to_numpy(float),
        dtype=float,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("model feature matrix is not finite")
    return matrix


def fit_multinomial(
    name: str,
    development: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    target_column: str = "next_registered_route",
    weight_column: str = "sequential_row_weight",
) -> FittedMultinomial:
    """Fit exactly one deterministic weighted L2 multinomial model."""

    if target_column not in development or weight_column not in development:
        raise ValueError("model target or weight is missing")
    matrix = _feature_matrix(development, feature_names)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    standardised = (matrix - mean) / scale
    labels = development[target_column].astype(str).to_numpy()
    weights = development[weight_column].astype(float).to_numpy()
    if np.unique(labels).size < 3:
        raise ValueError("multinomial model requires at least three observed classes")
    kwargs: dict[str, Any] = {
        "penalty": "l2",
        "C": 0.25,
        "solver": "lbfgs",
        "max_iter": 300,
        "class_weight": None,
        "random_state": RANDOM_STATE,
    }
    signature = inspect.signature(LogisticRegression)
    if "multi_class" in signature.parameters:
        kwargs["multi_class"] = "multinomial"
    if "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = 1
    model = LogisticRegression(**kwargs)
    model.fit(standardised, labels, sample_weight=weights)
    iterations = tuple(int(value) for value in np.asarray(model.n_iter_).tolist())
    return FittedMultinomial(
        name=name,
        feature_names=tuple(str(value) for value in feature_names),
        classes=tuple(str(value) for value in model.classes_.tolist()),
        scaler_mean=tuple(float(value) for value in mean.tolist()),
        scaler_scale=tuple(float(value) for value in scale.tolist()),
        coefficients=tuple(
            tuple(float(value) for value in row) for row in np.asarray(model.coef_).tolist()
        ),
        intercept=tuple(float(value) for value in np.asarray(model.intercept_).tolist()),
        iterations=iterations,
        converged=bool(max(iterations, default=0) < 300),
    )


def predict_multinomial(model: FittedMultinomial, frame: pd.DataFrame) -> np.ndarray[Any, Any]:
    """Reconstruct probabilities directly from the serialised parameters."""

    matrix = _feature_matrix(frame, model.feature_names)
    mean = np.asarray(model.scaler_mean, dtype=float)
    scale = np.asarray(model.scaler_scale, dtype=float)
    coefficient = np.asarray(model.coefficients, dtype=float)
    intercept = np.asarray(model.intercept, dtype=float)
    logits = ((matrix - mean) / scale) @ coefficient.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    probability: np.ndarray[Any, Any] = np.asarray(
        exponential / exponential.sum(axis=1, keepdims=True), dtype=float
    )
    return probability


def model_from_json(value: Mapping[str, Any]) -> FittedMultinomial:
    """Load one serialised model without relying on a pickled estimator."""

    return FittedMultinomial(
        name=str(value["name"]),
        feature_names=tuple(str(item) for item in value["feature_names"]),
        classes=tuple(str(item) for item in value["classes"]),
        scaler_mean=tuple(float(item) for item in value["scaler_mean"]),
        scaler_scale=tuple(float(item) for item in value["scaler_scale"]),
        coefficients=tuple(tuple(float(item) for item in row) for row in value["coefficient"]),
        intercept=tuple(float(item) for item in value["intercept"]),
        iterations=tuple(int(item) for item in value["iterations"]),
        converged=bool(value["converged"]),
    )


def multiclass_metrics(
    labels: Sequence[str],
    probabilities: np.ndarray[Any, Any],
    classes: Sequence[str],
    weights: Sequence[float],
) -> dict[str, float]:
    """Calculate all preregistered weighted multiclass metrics."""

    truth = np.asarray(labels, dtype=str)
    probability = np.asarray(probabilities, dtype=float)
    class_values = np.asarray(classes, dtype=str)
    sample_weight = np.asarray(weights, dtype=float)
    if probability.shape != (truth.size, class_values.size):
        raise ValueError("probability matrix shape differs from labels/classes")
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
    ):
        raise ValueError("multiclass probabilities are invalid")
    if sample_weight.shape != (truth.size,) or bool((sample_weight <= 0.0).any()):
        raise ValueError("metric weights are invalid")
    lookup = {value: index for index, value in enumerate(class_values.tolist())}
    try:
        target_index = np.asarray([lookup[value] for value in truth.tolist()], dtype=int)
    except KeyError as error:
        raise ValueError(f"unknown realised class: {error.args[0]}") from error
    total = float(sample_weight.sum())
    realised = probability[np.arange(truth.size), target_index]
    log_loss = float(np.sum(sample_weight * -np.log(np.clip(realised, 1e-15, 1.0))) / total)
    one_hot = np.zeros_like(probability)
    one_hot[np.arange(truth.size), target_index] = 1.0
    brier = float(np.sum(sample_weight * np.sum((probability - one_hot) ** 2, axis=1)) / total)
    order = np.argsort(-probability, axis=1, kind="mergesort")
    ranks = np.argmax(order == target_index[:, None], axis=1) + 1
    top_one = float(np.sum(sample_weight * (ranks == 1)) / total)
    top_two = float(np.sum(sample_weight * (ranks <= 2)) / total)
    reciprocal_rank = float(np.sum(sample_weight / ranks) / total)
    mean_realised = float(np.sum(sample_weight * realised) / total)
    entropy_by_row = -np.sum(probability * np.log(np.clip(probability, 1e-15, 1.0)), axis=1)
    entropy = float(np.sum(sample_weight * entropy_by_row) / total)
    confidence = probability.max(axis=1)
    correctness = (ranks == 1).astype(float)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for bin_index in range(10):
        selected = bins == bin_index
        if not bool(selected.any()):
            continue
        bin_weight = sample_weight[selected]
        share = float(bin_weight.sum() / total)
        accuracy = float(np.sum(bin_weight * correctness[selected]) / bin_weight.sum())
        calibration = float(np.sum(bin_weight * confidence[selected]) / bin_weight.sum())
        ece += share * abs(accuracy - calibration)
    return {
        "multiclass_log_loss": log_loss,
        "multiclass_brier": brier,
        "top_one_accuracy": top_one,
        "top_two_accuracy": top_two,
        "mean_reciprocal_rank": reciprocal_rank,
        "mean_probability_realised_class": mean_realised,
        "expected_calibration_error": float(ece),
        "prediction_entropy": entropy,
        "effective_candidate_count": float(math.exp(entropy)),
    }


def counterfactual_probability_difference(
    model: FittedMultinomial,
    frame: pd.DataFrame,
    *,
    zero_features: Sequence[str],
    target_classes: Sequence[str],
) -> np.ndarray[Any, Any]:
    """Return original minus feature-zeroed target probability without refitting."""

    missing = sorted(set(zero_features).difference(frame.columns))
    if missing:
        raise ValueError(f"counterfactual features are missing: {missing}")
    original = predict_multinomial(model, frame)
    counterfactual = frame.copy()
    counterfactual.loc[:, list(zero_features)] = 0.0
    changed = predict_multinomial(model, counterfactual)
    indices = [model.classes.index(value) for value in target_classes]
    difference: np.ndarray[Any, Any] = np.asarray(
        original[:, indices].sum(axis=1) - changed[:, indices].sum(axis=1),
        dtype=float,
    )
    return difference


def matched_control_relations(
    panel: pd.DataFrame,
    *,
    treated_column: str,
    untreated_history_column: str,
    stratum_columns: Sequence[str],
    minimum_controls: int = 5,
) -> pd.DataFrame:
    """Match every treated update to equal-weight same-stage untreated controls."""

    required = {"sequential_row_id", treated_column, untreated_history_column, *stratum_columns}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"matched-control fields are missing: {missing}")
    controls = panel.loc[panel[untreated_history_column].astype(int).eq(0)].copy()
    grouped = {
        tuple(row if isinstance(row, tuple) else (row,)): group
        for row, group in controls.groupby(list(stratum_columns), sort=True, dropna=False)
    }
    rows: list[dict[str, Any]] = []
    treated = panel.loc[panel[treated_column].astype(int).eq(1)].sort_values(
        "sequential_row_id", kind="mergesort"
    )
    for source in treated.itertuples(index=False):
        key = tuple(getattr(source, column) for column in stratum_columns)
        candidates = grouped.get(key)
        if candidates is None or len(candidates) < minimum_controls:
            continue
        weight = 1.0 / len(candidates)
        for control in candidates.itertuples(index=False):
            rows.append(
                {
                    "treated_row_id": str(source.sequential_row_id),
                    "control_row_id": str(control.sequential_row_id),
                    "control_weight_within_treated": weight,
                    **{column: getattr(source, column) for column in stratum_columns},
                }
            )
    return pd.DataFrame(rows)


def session_bootstrap_multiplicities(
    sessions: Sequence[str], *, draws: int = 25, seed: int = RANDOM_STATE
) -> list[dict[str, int]]:
    """Return fixed-seed whole-session bootstrap multiplicities."""

    unique = np.asarray(sorted(set(str(value) for value in sessions)), dtype=object)
    if unique.size == 0:
        raise ValueError("bootstrap requires at least one session")
    generator = np.random.default_rng(seed)
    outputs: list[dict[str, int]] = []
    for _ in range(draws):
        sampled = generator.choice(unique, size=unique.size, replace=True)
        values, counts = np.unique(sampled, return_counts=True)
        outputs.append(
            {str(value): int(count) for value, count in zip(values, counts, strict=True)}
        )
    return outputs


def permute_hidden_bundle(
    panel: pd.DataFrame,
    *,
    bundle_columns: Sequence[str],
    group_columns: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Permute complete hidden-history row bundles among stocks within stage."""

    required = {*bundle_columns, *group_columns, "symbol"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"hidden permutation fields are missing: {missing}")
    result = panel.copy()
    generator = np.random.default_rng(seed)
    for _, indices in result.groupby(list(group_columns), sort=True, dropna=False).groups.items():
        positions = np.asarray(sorted(indices), dtype=int)
        if positions.size <= 1:
            continue
        permutation = generator.permutation(positions.size)
        bundles = result.loc[positions, list(bundle_columns)].to_numpy(copy=True)
        result.loc[positions, list(bundle_columns)] = bundles[permutation]
    return result


def choose_primary_decision(
    *,
    blocker: str | None,
    target_a_status: str,
    target_b_status: str,
    target_c_status: str,
    diversion_status: str,
    hidden_increment_status: str,
) -> str:
    """Apply the preregistered primary decision matrix."""

    statuses = {
        target_a_status,
        target_b_status,
        target_c_status,
        diversion_status,
        hidden_increment_status,
    }
    if not statuses.issubset(STATUS_VALUES):
        raise ValueError(f"unknown hypothesis status: {sorted(statuses - STATUS_VALUES)}")
    if blocker is not None:
        if blocker not in PRIMARY_DECISIONS or not blocker.startswith("blocked_"):
            raise ValueError(f"unknown blocker: {blocker}")
        return blocker
    hidden_target = target_a_status == "supported" or target_b_status == "supported"
    recurrence = target_c_status == "supported"
    diversion = diversion_status == "supported"
    if hidden_target and recurrence:
        return "target_specific_hidden_routes_and_registered_recurrence_supported"
    if hidden_target:
        return "hidden_target_specific_routes_supported_only"
    if recurrence:
        return "registered_recurrence_supported_only"
    if diversion:
        return "hidden_diversion_supported_only"
    if "descriptive_only" in statuses:
        return "descriptive_competing_route_structure_only"
    return "no_competing_route_increment"
