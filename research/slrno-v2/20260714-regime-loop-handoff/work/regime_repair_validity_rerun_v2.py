"""Unchanged-gate Regime Model Validity V2 rerun over the repaired lineage."""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from stocker_research.causal_state_export_v2 import HysteresisConfig
from stocker_research.loop_nulls_v2 import SessionRunSequence
from stocker_research.regime_gap_segmentation_v2 import (
    annotate_causal_segments,
    causal_segment_groups,
)
from stocker_research.regime_panel_v2 import (
    EMISSION_FEATURES,
    MARKET_EMISSION_FEATURES,
    NATURAL_KEY,
    STOCK_EMISSION_FEATURES,
)
from stocker_research.regime_refit_v2 import (
    FullRefitResult,
    RefitConfig,
    fit_full_right_censored_refit,
)
from stocker_research.regime_repair_comparison_v2 import (
    aligned_assignment_metrics,
    compare_loop_events,
    primitive_loop_events,
    reversal_rates,
    run_boundary_ledger,
    state_occupancy,
    transition_matrix,
)
from stocker_research.regime_validity_v2 import (
    CleaningVariant,
    PartADecision,
    PartAGateEvidence,
    apply_cleaning_variant,
    build_training_sample,
    decide_part_a,
    gaussian_log_emissions,
    semantic_remap_by_activity_direction,
    transform_emissions,
)
from stocker_research.semantic_loop_dictionary_v2 import semantic_primitive_id
from stocker_research.state_alignment_v2 import (
    AlignmentWeights,
    align_states,
    apply_state_mapping,
)
from stocker_research.state_representation_sensitivity_v2 import (
    compare_representation_events,
    hierarchical_state_ids,
    hysteretic_states_by_session,
    reconstruct_first_event_outcomes,
    transition_confidence,
)

K_VALUES = (6, 8, 10, 12)
SEEDS = (20260710, 20260711, 20260712, 20260713, 20260714)
SAMPLE_VARIANTS = ("SAMPLE_A", "SAMPLE_B", "SAMPLE_C", "SAMPLE_D")
SELECTED_PATHS = ((5, 6, 5), (4, 6, 4))
SELECTED_IDS = tuple(semantic_primitive_id(path[:-1]) for path in SELECTED_PATHS)


@dataclass(frozen=True, slots=True)
class ValidityRerunResult:
    tables: dict[str, pd.DataFrame]
    metrics: dict[str, Any]
    evidence: PartAGateEvidence
    decision: PartADecision
    primary_hysteretic_labels: np.ndarray
    primary_events: pd.DataFrame
    hysteretic_events: pd.DataFrame
    primary_first_events: pd.DataFrame
    hierarchical_mapping: pd.DataFrame


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or len(a) < 2:
        return math.nan
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _transition_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.zeros(len(left), dtype=float)
    for state in range(len(left)):
        denominator = float(np.linalg.norm(left[state]) * np.linalg.norm(right[state]))
        output[state] = (
            float(np.dot(left[state], right[state]) / denominator) if denominator > 0.0 else 1.0
        )
    return output


def _median_run_duration(
    panel: pd.DataFrame, labels: np.ndarray, *, state_count: int
) -> np.ndarray:
    runs = run_boundary_ledger(panel, labels, lineage="duration_profile")
    output = np.zeros(state_count, dtype=float)
    for state in range(state_count):
        values = runs.loc[runs["state"].eq(state), "duration"]
        output[state] = float(values.median()) if len(values) else 0.0
    return output


def _duration_profile(parameters: Any) -> np.ndarray:
    hazard = np.asarray(parameters.duration_hazard, dtype=float)
    return np.cumprod(1.0 - hazard, axis=1)


def _segment_decision_surface(panel: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    frame = panel[
        [
            "symbol",
            "session",
            "segment_id",
            "segment_bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
        ]
    ].copy()
    frame["original_session"] = frame["session"]
    frame["session"] = frame["segment_id"]
    frame["bar_ordinal"] = frame["segment_bar_ordinal"].astype(int)
    frame["decision_timestamp"] = frame["bar_complete_timestamp"]
    frame["decision_id"] = [
        f"{prefix}:{symbol}:{segment}:{int(bar):02d}"
        for symbol, segment, bar in frame[["symbol", "segment_id", "bar_ordinal"]].itertuples(
            index=False, name=None
        )
    ]
    return frame[
        [
            "decision_id",
            "symbol",
            "session",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
            "decision_timestamp",
            "original_session",
            "segment_id",
        ]
    ]


def _first_events(
    prior: Any,
    panel: pd.DataFrame,
    labels: np.ndarray,
    *,
    prefix: str,
    state_count: int,
) -> pd.DataFrame:
    decisions = _segment_decision_surface(panel, prefix=prefix)
    return reconstruct_first_event_outcomes(
        decisions,
        labels,
        dictionary=prior._dictionary(),
        horizon_bars=24,
        allowed_states=frozenset(range(state_count)),
    )


def _session_sequences(panel: pd.DataFrame, labels: np.ndarray) -> tuple[SessionRunSequence, ...]:
    states = np.asarray(labels, dtype=int)
    records: list[SessionRunSequence] = []
    for segment_id, group in panel.groupby("segment_id", sort=False):
        positions = group.index.to_numpy(dtype=int)
        local = states[positions]
        starts = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1]
        ends = np.r_[starts[1:], len(local)]
        records.append(
            SessionRunSequence(
                symbol=str(group["symbol"].iloc[0]),
                session=str(segment_id),
                states=tuple(int(local[start]) for start in starts),
                durations=tuple(int(end - start) for start, end in zip(starts, ends, strict=True)),
                terminal_right_censored=bool(
                    group["session_source_complete"].all()
                    and str(group["segment_end_reason"].iloc[-1]) == "scheduled_session_end"
                ),
            )
        )
    return tuple(records)


def _translated_labels(
    labels: np.ndarray,
    mapping: Mapping[int, int],
    *,
    reference_state_count: int,
    candidate_state_count: int,
) -> np.ndarray:
    extended = dict(mapping)
    next_state = reference_state_count
    for candidate in range(candidate_state_count):
        if candidate not in extended:
            extended[candidate] = next_state
            next_state += 1
    return apply_state_mapping(labels, extended)


def _event_set(events: pd.DataFrame, loop_id: str) -> set[tuple[str, str, int]]:
    subset = events.loc[events["primitive_loop_id"].eq(loop_id)]
    return {
        (str(symbol), str(session), int(bar))
        for symbol, session, bar in subset[["symbol", "session", "event_bar_ordinal"]].itertuples(
            index=False, name=None
        )
    }


def _bounded_event_agreement(
    reference: set[tuple[str, str, int]],
    candidate: set[tuple[str, str, int]],
    *,
    shift: int,
) -> float:
    if not reference:
        return math.nan
    by_session: dict[tuple[str, str], set[int]] = {}
    for symbol, session, bar in candidate:
        by_session.setdefault((symbol, session), set()).add(bar)
    matched = sum(
        any(
            abs(candidate_bar - bar) <= shift
            for candidate_bar in by_session.get((symbol, session), set())
        )
        for symbol, session, bar in reference
    )
    return matched / len(reference)


def _top_loop_ids(events: pd.DataFrame, *, count: int = 20) -> tuple[str, ...]:
    frequencies = events["primitive_loop_id"].value_counts()
    ordered = sorted(
        ((str(loop_id), int(value)) for loop_id, value in frequencies.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(loop_id for loop_id, _ in ordered[:count])


def _filter_summary(prior: Any, panel: pd.DataFrame, fit: FullRefitResult) -> Any:
    scaled = transform_emissions(panel, fit.preprocessing)
    emissions = gaussian_log_emissions(scaled, fit.parameters)
    return prior._causal_filter_summary_compiled(
        emissions,
        groups=causal_segment_groups(panel),
        model=fit.parameters.as_dict(),
    )


def _period_tables(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    *,
    development_scaled: np.ndarray,
    assessment_scaled: np.ndarray,
    development_labels: np.ndarray,
    assessment_labels: np.ndarray,
    state_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles: list[dict[str, Any]] = []
    for period, _panel, scaled, labels in (
        ("development_2024", development, development_scaled, development_labels),
        ("assessment_2025", assessment, assessment_scaled, assessment_labels),
    ):
        for state in range(state_count):
            mask = labels == state
            for feature_index, feature in enumerate(EMISSION_FEATURES):
                values = scaled[mask, feature_index]
                profiles.append(
                    {
                        "period": period,
                        "state": state,
                        "feature": feature,
                        "rows": int(mask.sum()),
                        "mean": float(np.mean(values)) if len(values) else math.nan,
                        "q10": float(np.quantile(values, 0.10)) if len(values) else math.nan,
                        "median": float(np.median(values)) if len(values) else math.nan,
                        "q90": float(np.quantile(values, 0.90)) if len(values) else math.nan,
                    }
                )
    profiles_frame = pd.DataFrame(profiles)
    dev_centroids = (
        profiles_frame.loc[profiles_frame["period"].eq("development_2024")]
        .pivot(index="state", columns="feature", values="mean")
        .reindex(columns=list(EMISSION_FEATURES))
        .to_numpy(float)
    )
    assess_centroids = (
        profiles_frame.loc[profiles_frame["period"].eq("assessment_2025")]
        .pivot(index="state", columns="feature", values="mean")
        .reindex(columns=list(EMISSION_FEATURES))
        .to_numpy(float)
    )
    centroid_drift = np.sqrt(np.mean(np.square(dev_centroids - assess_centroids), axis=1))
    dev_occupancy = state_occupancy(development_labels, state_count=state_count)
    assess_occupancy = state_occupancy(assessment_labels, state_count=state_count)
    dev_transition = transition_matrix(
        development_labels,
        causal_segment_groups(development),
        state_count=state_count,
    )
    assess_transition = transition_matrix(
        assessment_labels,
        causal_segment_groups(assessment),
        state_count=state_count,
    )
    transition_cosine = _transition_cosine(dev_transition, assess_transition)
    dev_duration = _median_run_duration(development, development_labels, state_count=state_count)
    assess_duration = _median_run_duration(assessment, assessment_labels, state_count=state_count)
    duration_ratio = np.divide(
        assess_duration,
        dev_duration,
        out=np.full(state_count, math.nan),
        where=dev_duration > 0.0,
    )

    stock_rows: list[dict[str, Any]] = []
    clock_rows: list[dict[str, Any]] = []
    for period, panel, labels in (
        ("development_2024", development, development_labels),
        ("assessment_2025", assessment, assessment_labels),
    ):
        surface = panel[["symbol", "session", "clock_phase"]].copy()
        surface["state"] = labels
        state_totals = surface["state"].value_counts()
        for (state, symbol), count in (
            surface.groupby(["state", "symbol"], sort=True).size().items()
        ):
            stock_rows.append(
                {
                    "period": period,
                    "state": int(state),
                    "symbol": str(symbol),
                    "rows": int(count),
                    "stock_share_within_state": float(count / state_totals[int(state)]),
                }
            )
        for (state, clock), count in (
            surface.groupby(["state", "clock_phase"], sort=True).size().items()
        ):
            clock_rows.append(
                {
                    "period": period,
                    "state": int(state),
                    "clock_phase": str(clock),
                    "rows": int(count),
                    "clock_share_within_state": float(count / state_totals[int(state)]),
                }
            )
    stock = pd.DataFrame(stock_rows)
    clock = pd.DataFrame(clock_rows)
    maximum_stock = stock.groupby("state", sort=True)["stock_share_within_state"].max()
    drift = pd.DataFrame(
        {
            "state": np.arange(state_count),
            "centroid_drift_scaled_rms": centroid_drift,
            "development_occupancy": dev_occupancy,
            "assessment_occupancy": assess_occupancy,
            "occupancy_change": assess_occupancy - dev_occupancy,
            "transition_cosine_similarity": transition_cosine,
            "development_median_duration": dev_duration,
            "assessment_median_duration": assess_duration,
            "duration_median_ratio": duration_ratio,
            "maximum_single_stock_share": [
                float(maximum_stock.get(state, math.nan)) for state in range(state_count)
            ],
        }
    )
    drift["period_component_gate_pass"] = (
        drift["centroid_drift_scaled_rms"].le(3.0)
        & drift["development_occupancy"].ge(0.01)
        & drift["assessment_occupancy"].ge(0.01)
        & drift["transition_cosine_similarity"].ge(0.70)
        & drift["duration_median_ratio"].between(0.50, 2.0, inclusive="both")
        & drift["maximum_single_stock_share"].le(0.25)
    )
    transition_rows: list[dict[str, Any]] = []
    for period, matrix in (
        ("development_2024", dev_transition),
        ("assessment_2025", assess_transition),
    ):
        for origin in range(state_count):
            for destination in range(state_count):
                transition_rows.append(
                    {
                        "period": period,
                        "origin_state": origin,
                        "destination_state": destination,
                        "probability": float(matrix[origin, destination]),
                    }
                )
    return (
        profiles_frame,
        drift,
        stock,
        clock,
        pd.DataFrame(transition_rows),
    )


def _market_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated stock rows to one market row per timestamp."""

    columns = [
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "expected_session_bars",
        *MARKET_EMISSION_FEATURES,
    ]
    frame = (
        panel[columns]
        .drop_duplicates("bar_start_timestamp")
        .sort_values("bar_start_timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    frame["symbol"] = "MARKET"
    expected = {
        ("MARKET", str(session)): int(group["expected_session_bars"].iloc[0])
        for session, group in frame.groupby("session", sort=True)
    }
    segmented, _ = annotate_causal_segments(
        frame[
            [
                "symbol",
                "session",
                "bar_ordinal",
                "bar_start_timestamp",
                "bar_complete_timestamp",
                "expected_session_bars",
                *MARKET_EMISSION_FEATURES,
            ]
        ],
        expected_bars=expected,
    )
    segmented["bar_complete_timestamp"] = pd.to_datetime(
        segmented["bar_start_timestamp"], utc=True
    ) + pd.Timedelta(minutes=5)
    segmented["clock_phase"] = pd.cut(
        segmented["bar_ordinal"],
        bins=[-1, 5, 23, 53, 71, 77],
        labels=["open", "morning", "midday", "afternoon", "close"],
    ).astype(str)
    return segmented


def _maximum_stock_share(panel: pd.DataFrame, labels: np.ndarray) -> float:
    surface = panel[["symbol"]].copy()
    surface["state"] = np.asarray(labels, dtype=int)
    counts = surface.groupby(["state", "symbol"], sort=True).size()
    totals = surface.groupby("state", sort=True).size()
    shares = [float(count / totals.loc[state]) for (state, _), count in counts.items()]
    return max(shares, default=math.nan)


def _representation_comparison(
    prior: Any,
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    *,
    primary_fit: FullRefitResult,
    primary_summary: Any,
    assessment_summary: Any,
    sample_a: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit unchanged stock-only and compact hierarchical audit surfaces."""

    stock_config = RefitConfig(
        state_count=8,
        seed=20260710,
        maximum_age=78,
        cleaning_variant=CleaningVariant.CLEANING_1,
    )
    stock_fit = fit_full_right_censored_refit(
        development,
        feature_names=STOCK_EMISSION_FEATURES,
        config=stock_config,
        training_indices=sample_a,
    )
    stock_summary = _filter_summary(prior, development, stock_fit)
    stock_assessment = _filter_summary(prior, assessment, stock_fit)

    market_development = _market_panel(development)
    market_assessment = _market_panel(assessment)
    market_candidates: list[tuple[int, FullRefitResult, Any, float]] = []
    for market_k in (3, 4):
        config = RefitConfig(
            state_count=market_k,
            seed=20260718,
            nominal_maximum_rows=len(market_development),
            maximum_age=78,
            cleaning_variant=CleaningVariant.CLEANING_CAUSAL,
            batch_size=1024,
            n_init=10,
            max_iter=300,
            activity_column="regime_log_market_dispersion",
            direction_column="regime_market_breadth_centered",
        )
        fit = fit_full_right_censored_refit(
            market_development,
            feature_names=MARKET_EMISSION_FEATURES,
            config=config,
            training_indices=np.arange(len(market_development), dtype=np.int64),
        )
        summary = _filter_summary(prior, market_development, fit)
        nll = float(-np.mean(summary.log_likelihood))
        market_candidates.append((market_k, fit, summary, nll))
    market_k, market_fit, market_summary, market_nll = min(
        market_candidates,
        key=lambda item: (
            item[3],
            item[0],
        ),
    )
    market_assessment_summary = _filter_summary(prior, market_assessment, market_fit)
    market_by_timestamp = dict(
        zip(
            pd.to_datetime(market_development["bar_start_timestamp"], utc=True),
            market_summary.hard_states,
            strict=True,
        )
    )
    assessment_market_by_timestamp = dict(
        zip(
            pd.to_datetime(market_assessment["bar_start_timestamp"], utc=True),
            market_assessment_summary.hard_states,
            strict=True,
        )
    )
    row_market = np.asarray(
        [
            int(market_by_timestamp[pd.Timestamp(timestamp)])
            for timestamp in pd.to_datetime(development["bar_start_timestamp"], utc=True)
        ],
        dtype=int,
    )
    assessment_row_market = np.asarray(
        [
            int(assessment_market_by_timestamp[pd.Timestamp(timestamp)])
            for timestamp in pd.to_datetime(assessment["bar_start_timestamp"], utc=True)
        ],
        dtype=int,
    )
    hierarchy = hierarchical_state_ids(
        row_market,
        stock_summary.hard_states,
        stock_state_count=8,
    )
    assessment_hierarchy = hierarchical_state_ids(
        assessment_row_market,
        stock_assessment.hard_states,
        stock_state_count=8,
    )
    hierarchy_count = market_k * 8
    hierarchy_mapping = development[
        [
            "symbol",
            "session",
            "segment_id",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
        ]
    ].copy()
    hierarchy_mapping["timestamp"] = hierarchy_mapping["bar_complete_timestamp"]
    hierarchy_mapping["market_state"] = row_market
    hierarchy_mapping["stock_state"] = stock_summary.hard_states
    hierarchy_mapping["hierarchical_state"] = hierarchy.numeric
    hierarchy_mapping["hierarchical_state_token"] = hierarchy.tokens

    representation_rows: list[dict[str, Any]] = []
    surfaces = (
        (
            "MODEL_COMBINED",
            primary_summary.hard_states,
            assessment_summary.hard_states,
            float(-np.mean(primary_summary.log_likelihood)),
            float(-np.mean(assessment_summary.log_likelihood)),
            8,
        ),
        (
            "MODEL_STOCK_ONLY",
            stock_summary.hard_states,
            stock_assessment.hard_states,
            float(-np.mean(stock_summary.log_likelihood)),
            float(-np.mean(stock_assessment.log_likelihood)),
            8,
        ),
        (
            "MODEL_HIERARCHICAL",
            hierarchy.numeric,
            assessment_hierarchy.numeric,
            float(-np.mean(stock_summary.log_likelihood)) + market_nll,
            float(-np.mean(stock_assessment.log_likelihood))
            + float(-np.mean(market_assessment_summary.log_likelihood)),
            hierarchy_count,
        ),
    )
    for name, labels, assess_labels, nll, assess_nll, state_count in surfaces:
        dev_occupancy = state_occupancy(labels, state_count=state_count)
        assess_occupancy = state_occupancy(assess_labels, state_count=state_count)
        dev_events = primitive_loop_events(development, labels)
        assess_events = primitive_loop_events(assessment, assess_labels)
        top_dev = set(_top_loop_ids(dev_events))
        top_assess = set(_top_loop_ids(assess_events))
        representation_rows.append(
            {
                "representation": name,
                "state_count": state_count,
                "causal_negative_log_likelihood": nll,
                "assessment_causal_negative_log_likelihood": assess_nll,
                "minimum_state_occupancy": float(dev_occupancy.min()),
                "assessment_minimum_state_occupancy": float(assess_occupancy.min()),
                "maximum_stock_share": _maximum_stock_share(development, labels),
                "loop_count": len(dev_events),
                "assessment_loop_count": len(assess_events),
                "loop_count_ratio": len(assess_events) / max(len(dev_events), 1),
                "top20_dictionary_period_jaccard": len(top_dev & top_assess)
                / max(len(top_dev | top_assess), 1),
                "period_occupancy_correlation": _safe_correlation(dev_occupancy, assess_occupancy),
                "maximum_period_occupancy_drift": float(
                    np.max(np.abs(dev_occupancy - assess_occupancy))
                ),
                "interpretability": (
                    "market_state_x_stock_state"
                    if name == "MODEL_HIERARCHICAL"
                    else "combined_context"
                    if name == "MODEL_COMBINED"
                    else "stock_behaviour"
                ),
                "redundancy": (
                    "separate_market_context"
                    if name == "MODEL_HIERARCHICAL"
                    else "audited_separately"
                ),
            }
        )
    representation = pd.DataFrame(representation_rows)
    market_profiles = pd.DataFrame(
        {
            "state": np.arange(market_k),
            "occupancy": state_occupancy(market_summary.hard_states, state_count=market_k),
            "median_duration": _median_run_duration(
                market_development,
                market_summary.hard_states,
                state_count=market_k,
            ),
            "selected_market_state_count": market_k,
            "development_negative_log_likelihood": market_nll,
        }
    )
    stock_profiles = pd.DataFrame(
        {
            "state": np.arange(8),
            "occupancy": state_occupancy(stock_summary.hard_states, state_count=8),
            "median_duration": _median_run_duration(
                development, stock_summary.hard_states, state_count=8
            ),
        }
    )
    combined_row = representation.loc[representation["representation"].eq("MODEL_COMBINED")].iloc[0]
    stock_row = representation.loc[representation["representation"].eq("MODEL_STOCK_ONLY")].iloc[0]
    hierarchy_row = representation.loc[
        representation["representation"].eq("MODEL_HIERARCHICAL")
    ].iloc[0]
    best_alternative_drift = min(
        float(stock_row["maximum_period_occupancy_drift"]),
        float(hierarchy_row["maximum_period_occupancy_drift"]),
    )
    combined_deficit = max(
        0.0,
        float(combined_row["maximum_period_occupancy_drift"]) - best_alternative_drift,
    )
    metadata = {
        "stock_fit": stock_fit,
        "stock_summary": stock_summary,
        "stock_assessment": stock_assessment,
        "selected_market_state_count": market_k,
        "market_fit": market_fit,
        "market_summary": market_summary,
        "market_assessment_summary": market_assessment_summary,
        "combined_stability_deficit": combined_deficit,
        "hierarchical_materially_more_stable": (
            float(hierarchy_row["period_occupancy_correlation"])
            >= float(combined_row["period_occupancy_correlation"]) + 0.10
        ),
        "hierarchical_reproducible": (
            float(hierarchy_row["minimum_state_occupancy"]) >= 0.01
            and float(hierarchy_row["maximum_stock_share"]) <= 0.25
        ),
    }
    return (
        representation,
        market_profiles,
        stock_profiles,
        {**metadata, "hierarchy_mapping": hierarchy_mapping},
    )


def run_unchanged_validity_rerun(
    prior: Any,
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    *,
    primary_fit: FullRefitResult,
    primary_summary: Any,
    assessment_summary: Any,
    contract: Mapping[str, Any],
) -> ValidityRerunResult:
    """Rerun every unchanged Part A gate class over the repaired primary model."""

    gate_binding = contract["unchanged_part_a_binding"]
    thresholds = gate_binding["frozen_structural_thresholds"]
    if tuple(gate_binding["state_counts"]) != K_VALUES:
        raise ValueError("unchanged K registry differs from the prior contract")
    if tuple(gate_binding["seeds"]) != SEEDS:
        raise ValueError("unchanged seed registry differs from the prior contract")
    if tuple(gate_binding["training_sample_variants"]) != SAMPLE_VARIANTS:
        raise ValueError("unchanged sample registry differs from the prior contract")

    development_groups = causal_segment_groups(development)
    reference_labels = np.asarray(primary_summary.hard_states, dtype=int)
    reference_assessment_labels = np.asarray(assessment_summary.hard_states, dtype=int)
    primary_events = primitive_loop_events(development, reference_labels)
    reference_first_events = _first_events(
        prior,
        development,
        reference_labels,
        prefix="repaired-development",
        state_count=8,
    )
    reference_top = set(_top_loop_ids(primary_events))
    reference_event_sets = {
        loop_id: _event_set(primary_events, loop_id) for loop_id in SELECTED_IDS
    }

    print("validity-rerun: hard/hysteretic/soft representation", flush=True)
    hysteretic = hysteretic_states_by_session(
        primary_summary.state_probabilities,
        session_groups=development_groups,
        config=HysteresisConfig(
            switch_probability=0.55,
            switch_margin=0.10,
        ),
    )
    hysteretic_events = primitive_loop_events(development, hysteretic)
    hysteretic_first_events = _first_events(
        prior,
        development,
        hysteretic,
        prefix="repaired-development",
        state_count=8,
    )
    representation_comparison, representation_metrics = compare_representation_events(
        reference_first_events,
        hysteretic_first_events,
        allowed_shift_bars=int(thresholds["hysteretic_allowed_timestamp_shift_bars"]),
    )
    selected_reference = representation_comparison["primary_label_reference"].isin(SELECTED_IDS)
    selected_same = representation_comparison.loc[selected_reference, "agreement_class"].isin(
        [
            "EXACT_EVENT_AGREEMENT",
            "SAME_PRIMITIVE_SHIFTED_TIMESTAMP",
        ]
    )
    hysteretic_selected_agreement = float(selected_same.mean())
    churn = transition_confidence(
        primary_summary.state_probabilities,
        hard_states=reference_labels,
        hysteretic_states=hysteretic,
        session_groups=development_groups,
    )
    churn_positions = churn["position"].to_numpy(dtype=int)
    churn["symbol"] = development.loc[churn_positions, "symbol"].to_numpy()
    churn["session"] = development.loc[churn_positions, "session"].to_numpy()
    churn["segment_id"] = development.loc[churn_positions, "segment_id"].to_numpy()
    churn["timestamp"] = development.loc[churn_positions, "bar_complete_timestamp"].to_numpy()
    churn_summary = pd.DataFrame(
        [
            {
                "transition_count": len(churn),
                "margin_lt_0_02_rate": float(churn["margin_lt_0_02"].mean()),
                "margin_lt_0_05_rate": float(churn["margin_lt_0_05"].mean()),
                "entropy_top_quartile_rate": float(churn["entropy_top_quartile"].mean()),
                "new_state_probability_lt_0_50_rate": float(
                    churn["new_state_probability_lt_0_50"].mean()
                ),
                "one_bar_reversal_rate": float(churn["one_bar_reversal"].mean()),
                "two_bar_reversal_rate": float(churn["two_bar_reversal"].mean()),
                "hysteretic_transition_agreement_rate": float(
                    churn["hysteretic_state_agreement"].mean()
                ),
            }
        ]
    )
    loop_robustness_rows: list[dict[str, Any]] = []
    for loop_id in SELECTED_IDS:
        reference_subset = primary_events.loc[primary_events["primitive_loop_id"].eq(loop_id)]
        candidate_subset = hysteretic_events.loc[hysteretic_events["primitive_loop_id"].eq(loop_id)]
        event_metrics = compare_loop_events(
            reference_subset,
            candidate_subset,
            allowed_shift_bars=int(thresholds["hysteretic_allowed_timestamp_shift_bars"]),
        )
        event_positions = reference_subset["event_position"].to_numpy(dtype=int)
        event_support = (
            np.max(primary_summary.state_probabilities[event_positions], axis=1)
            if len(event_positions)
            else np.asarray([], dtype=float)
        )
        loop_robustness_rows.append(
            {
                "primitive_loop_id": loop_id,
                "hard_event_count": len(reference_subset),
                "hysteretic_event_count": len(candidate_subset),
                **event_metrics,
                "stock_breadth": int(reference_subset["symbol"].nunique()),
                "month_breadth": int(reference_subset["session"].astype(str).str[:7].nunique()),
                "low_soft_support_hard_events": int(np.sum(event_support < 0.25)),
                "soft_supported_robust_events": int(np.sum(event_support >= 0.50)),
                "soft_mass_creates_hard_event": False,
            }
        )
    loop_robustness = pd.DataFrame(loop_robustness_rows)
    dictionary_robustness = pd.DataFrame(
        [
            {
                "reference": "repaired_hard_map",
                "candidate": "causal_hysteretic",
                "exact_event_agreement": representation_metrics.exact_fraction,
                "same_primitive_bounded_shift_fraction": (
                    representation_metrics.same_primitive_bounded_shift_fraction
                ),
                "selected_reference_same_primitive_fraction": (hysteretic_selected_agreement),
                "primitive_mismatch_fraction": (representation_metrics.primitive_mismatch_fraction),
                "dictionary_entries": len(SELECTED_IDS),
                "soft_hard_event_creation_allowed": False,
            }
        ]
    )

    print("validity-rerun: offline cleaning sensitivities", flush=True)
    cleaning_state_rows: list[dict[str, Any]] = []
    cleaning_loop_rows: list[dict[str, Any]] = []
    cleaning_overlap_rows: list[dict[str, Any]] = []
    for variant in CleaningVariant:
        cleaned = apply_cleaning_variant(
            primary_fit.raw_labels,
            scaled=primary_fit.scaled,
            groups=development_groups,
            centroids=primary_fit.raw_cluster_centers,
            variant=variant,
        )
        mapped, _ = semantic_remap_by_activity_direction(
            cleaned,
            development,
            activity_column="regime_log_activity_12",
            direction_column="signed_efficiency_12",
        )
        events = primitive_loop_events(development, mapped)
        top = set(_top_loop_ids(events))
        runs = run_boundary_ledger(development, mapped, lineage=variant.value)
        one_bar_rate, two_bar_rate = reversal_rates(mapped, development_groups)
        cleaning_state_rows.append(
            {
                "variant": variant.value,
                "uses_future_neighbor": variant is CleaningVariant.CLEANING_1,
                "causal": variant is not CleaningVariant.CLEANING_1,
                "bars_relabelled": int(np.sum(cleaned != primary_fit.raw_labels)),
                "bars_relabelled_share": float(np.mean(cleaned != primary_fit.raw_labels)),
                "run_count": len(runs),
                "transition_count": len(runs) - len(development_groups),
                "median_run_length": float(runs["duration"].median()),
                "one_bar_reversal_rate": one_bar_rate,
                "two_bar_reversal_rate": two_bar_rate,
                "offline_causal_bar_agreement": float(np.mean(mapped == reference_labels)),
            }
        )
        for loop_id in SELECTED_IDS:
            cleaning_loop_rows.append(
                {
                    "variant": variant.value,
                    "primitive_loop_id": loop_id,
                    "event_count": int(events["primitive_loop_id"].eq(loop_id).sum()),
                    "stock_breadth": int(
                        events.loc[
                            events["primitive_loop_id"].eq(loop_id),
                            "symbol",
                        ].nunique()
                    ),
                    "first_event_coverage": float(
                        events["primitive_loop_id"].eq(loop_id).sum() / max(len(events), 1)
                    ),
                }
            )
        cleaning_overlap_rows.append(
            {
                "variant": variant.value,
                "top20_dictionary_jaccard": len(reference_top & top)
                / max(len(reference_top | top), 1),
                "selected_dictionary_overlap": len(set(SELECTED_IDS) & top) / len(SELECTED_IDS),
            }
        )

    print("validity-rerun: K and seed registry", flush=True)
    sample_frame = development.copy()
    sample_frame["month"] = sample_frame["session"].astype(str).str[:7]
    bounded_stride = build_training_sample(
        sample_frame,
        variant="SAMPLE_A",
        maximum_rows=int(gate_binding["training_sample_rows"]),
        seed=int(gate_binding["training_sample_seed"]),
    )
    reference_occupancy = state_occupancy(reference_labels, state_count=8)
    registry_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    loop_stability_rows: list[dict[str, Any]] = []
    first_event_rows: list[dict[str, Any]] = []
    for state_count in K_VALUES:
        for seed in SEEDS:
            model_id = f"regime_k{state_count}_seed{seed}"
            print(f"validity-rerun: fitting {model_id}", flush=True)
            fit = fit_full_right_censored_refit(
                development,
                feature_names=EMISSION_FEATURES,
                config=RefitConfig(
                    state_count=state_count,
                    seed=seed,
                    nominal_maximum_rows=int(gate_binding["training_sample_rows"]),
                    maximum_age=78,
                    cleaning_variant=CleaningVariant.CLEANING_1,
                ),
                training_indices=bounded_stride,
            )
            summary = _filter_summary(prior, development, fit)
            assessment_candidate = _filter_summary(prior, assessment, fit)
            alignment = align_states(
                primary_fit.parameters.means,
                fit.parameters.means,
                reference_transition=primary_fit.parameters.transitions,
                candidate_transition=fit.parameters.transitions,
                reference_duration=_duration_profile(primary_fit.parameters),
                candidate_duration=_duration_profile(fit.parameters),
                weights=AlignmentWeights(),
            )
            aligned = _translated_labels(
                summary.hard_states,
                alignment.candidate_to_reference,
                reference_state_count=8,
                candidate_state_count=state_count,
            )
            aligned_assessment = _translated_labels(
                assessment_candidate.hard_states,
                alignment.candidate_to_reference,
                reference_state_count=8,
                candidate_state_count=state_count,
            )
            assignment = aligned_assignment_metrics(
                reference_labels,
                summary.hard_states,
                candidate_to_reference=alignment.candidate_to_reference,
            )
            candidate_events = primitive_loop_events(development, aligned)
            candidate_assessment_events = primitive_loop_events(assessment, aligned_assessment)
            candidate_first = _first_events(
                prior,
                development,
                aligned,
                prefix="repaired-development",
                state_count=max(int(aligned.max()) + 1, 8),
            )
            first_comparison, _ = compare_representation_events(
                reference_first_events,
                candidate_first,
                allowed_shift_bars=int(thresholds["hysteretic_allowed_timestamp_shift_bars"]),
            )
            inverse_mapping = {
                reference: candidate
                for candidate, reference in alignment.candidate_to_reference.items()
            }
            translated_ids: dict[str, str] = {}
            for loop_id, path in zip(SELECTED_IDS, SELECTED_PATHS, strict=True):
                if all(state in inverse_mapping for state in path):
                    native_path = tuple(inverse_mapping[state] for state in path)
                    translated_ids[loop_id] = semantic_primitive_id(native_path[:-1])
            native_null = (
                prior._structural_null_metrics(
                    _session_sequences(development, summary.hard_states),
                    state_count=state_count,
                    draws=100,
                    seed=seed + 50000,
                    candidate_ids=tuple(translated_ids.values()),
                )
                if translated_ids
                else {}
            )
            mapped_null = {
                loop_id: native_null[native_id] for loop_id, native_id in translated_ids.items()
            }
            occupancy = state_occupancy(summary.hard_states, state_count=state_count)
            candidate_runs = run_boundary_ledger(
                development,
                summary.hard_states,
                lineage=model_id,
            )
            one_bar_rate, _ = reversal_rates(summary.hard_states, development_groups)
            top = set(_top_loop_ids(candidate_events))
            aligned_dev_occupancy = (
                state_occupancy(aligned[aligned < 8], state_count=8)
                if np.any(aligned < 8)
                else np.zeros(8)
            )
            aligned_assess_occupancy = (
                state_occupancy(
                    aligned_assessment[aligned_assessment < 8],
                    state_count=8,
                )
                if np.any(aligned_assessment < 8)
                else np.zeros(8)
            )
            period_correlation = _safe_correlation(aligned_dev_occupancy, aligned_assess_occupancy)
            centroid_distances = [pair.centroid_distance for pair in alignment.pairs]
            registry_rows.append(
                {
                    "model_id": model_id,
                    "state_count": state_count,
                    "seed": seed,
                    "sample_rows": len(bounded_stride),
                    "training_objective": fit.training_objective,
                    "causal_negative_log_likelihood": float(-np.mean(summary.log_likelihood)),
                    "iid_mixture_negative_log_likelihood": float(
                        -np.nanmean(summary.iid_log_likelihood)
                    ),
                    "minimum_state_occupancy": float(occupancy.min()),
                    "median_run_duration": float(candidate_runs["duration"].median()),
                    "matched_centroid_drift": float(np.mean(centroid_distances)),
                    "one_bar_reversal_rate": one_bar_rate,
                    "posterior_entropy": float(np.mean(summary.posterior_entropy)),
                    "primitive_loop_count": len(candidate_events),
                    "structurally_significant_selected_loop_count": sum(
                        metrics[3] <= 0.05 for metrics in mapped_null.values()
                    ),
                    "first_event_dictionary_coverage": float(
                        candidate_first["primary_label"].isin(SELECTED_IDS).mean()
                    ),
                    "dictionary_stability": len(set(SELECTED_IDS) & top) / len(SELECTED_IDS),
                    "selected_loop_stock_breadth": int(
                        candidate_events.loc[
                            candidate_events["primitive_loop_id"].isin(SELECTED_IDS),
                            "symbol",
                        ].nunique()
                    ),
                    "period_occupancy_correlation": period_correlation,
                    "assessment_selected_loop_count": int(
                        candidate_assessment_events["primitive_loop_id"].isin(SELECTED_IDS).sum()
                    ),
                    "parameter_hash": fit.parameter_hash,
                    "training_row_hash": fit.training_row_hash,
                }
            )
            for pair in alignment.pairs:
                alignment_rows.append(
                    {
                        "model_id": model_id,
                        "state_count": state_count,
                        "seed": seed,
                        "candidate_state": pair.candidate_state,
                        "reference_state": pair.reference_state,
                        "centroid_distance": pair.centroid_distance,
                        "transition_distance": pair.transition_distance,
                        "duration_distance": pair.duration_distance,
                        "total_cost": pair.total_cost,
                    }
                )
            stability_rows.append(
                {
                    "model_id": model_id,
                    "state_count": state_count,
                    "seed": seed,
                    "matched_centroid_distance": float(np.mean(centroid_distances)),
                    "transition_profile_similarity": float(
                        1.0 - np.mean([pair.transition_distance for pair in alignment.pairs])
                    ),
                    "duration_profile_similarity": float(
                        1.0 - np.mean([pair.duration_distance for pair in alignment.pairs])
                    ),
                    "state_occupancy_correlation": _safe_correlation(
                        reference_occupancy,
                        aligned_dev_occupancy,
                    ),
                    **assignment,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(reference_labels, summary.hard_states)
                    ),
                    "normalized_mutual_information": float(
                        normalized_mutual_info_score(reference_labels, summary.hard_states)
                    ),
                    "unmatched_reference_states": len(alignment.unmatched_reference),
                    "unmatched_candidate_states": len(alignment.unmatched_candidate),
                }
            )
            for loop_id in SELECTED_IDS:
                observed, null_mean, rate_ratio, empirical_p = mapped_null.get(
                    loop_id,
                    (0, math.nan, math.nan, math.nan),
                )
                loop_stability_rows.append(
                    {
                        "model_id": model_id,
                        "state_count": state_count,
                        "seed": seed,
                        "primitive_loop_id": loop_id,
                        "loop_translatable": loop_id in mapped_null,
                        "native_candidate_loop_id": translated_ids.get(loop_id, "unmatched"),
                        "observed_first_events": observed,
                        "structural_null_mean": null_mean,
                        "structural_null_rate_ratio": rate_ratio,
                        "structural_null_empirical_p": empirical_p,
                        "structurally_significant": bool(
                            math.isfinite(empirical_p) and empirical_p <= 0.05
                        ),
                        "positive_structural_excess": bool(
                            math.isfinite(rate_ratio) and rate_ratio > 1.0
                        ),
                        "dictionary_jaccard": len(reference_top & top)
                        / max(len(reference_top | top), 1),
                        "event_timestamp_agreement_bounded": (
                            _bounded_event_agreement(
                                reference_event_sets[loop_id],
                                _event_set(candidate_events, loop_id),
                                shift=2,
                            )
                        ),
                        "stock_breadth": int(
                            candidate_events.loc[
                                candidate_events["primitive_loop_id"].eq(loop_id),
                                "symbol",
                            ].nunique()
                        ),
                        "assessment_event_count": int(
                            candidate_assessment_events["primitive_loop_id"].eq(loop_id).sum()
                        ),
                        "period_occupancy_correlation": (period_correlation),
                    }
                )
                comparable = first_comparison["primary_label_reference"].eq(loop_id)
                first_event_rows.append(
                    {
                        "model_id": model_id,
                        "state_count": state_count,
                        "seed": seed,
                        "primitive_loop_id": loop_id,
                        "reference_first_events": int(
                            reference_first_events["primary_label"].eq(loop_id).sum()
                        ),
                        "candidate_first_events": int(
                            candidate_first["primary_label"].eq(loop_id).sum()
                        ),
                        "coverage_ratio": float(
                            candidate_first["primary_label"].eq(loop_id).sum()
                            / max(
                                reference_first_events["primary_label"].eq(loop_id).sum(),
                                1,
                            )
                        ),
                        "same_primitive_bounded_timestamp_fraction": float(
                            first_comparison.loc[comparable, "agreement_class"]
                            .isin(
                                [
                                    "EXACT_EVENT_AGREEMENT",
                                    "SAME_PRIMITIVE_SHIFTED_TIMESTAMP",
                                ]
                            )
                            .mean()
                        ),
                    }
                )
            del (
                fit,
                summary,
                assessment_candidate,
                candidate_events,
                candidate_assessment_events,
                candidate_first,
                first_comparison,
            )
            gc.collect()

    registry = pd.DataFrame(registry_rows)
    state_alignment = pd.DataFrame(alignment_rows)
    state_stability = pd.DataFrame(stability_rows)
    loop_stability = pd.DataFrame(loop_stability_rows)
    first_event_stability = pd.DataFrame(first_event_rows)

    print("validity-rerun: training sample registry", flush=True)
    sample_composition_rows: list[dict[str, Any]] = []
    sample_state_rows: list[dict[str, Any]] = []
    sample_loop_rows: list[dict[str, Any]] = []
    sample_row_tables: dict[str, pd.DataFrame] = {}
    for variant in SAMPLE_VARIANTS:
        print(f"validity-rerun: fitting {variant}", flush=True)
        sample = build_training_sample(
            sample_frame,
            variant=variant,
            maximum_rows=int(gate_binding["training_sample_rows"]),
            seed=int(gate_binding["training_sample_seed"]),
        )
        selected = sample_frame.iloc[sample].copy()
        selected["sample_variant"] = variant
        selected["sample_rank"] = np.arange(len(selected))
        sample_row_tables[variant] = selected[
            [
                "sample_variant",
                "sample_rank",
                *NATURAL_KEY,
                "segment_id",
            ]
        ]
        for (symbol, month, phase), group in selected.groupby(
            ["symbol", "month", "clock_phase"], sort=True
        ):
            sample_composition_rows.append(
                {
                    "sample_variant": variant,
                    "symbol": str(symbol),
                    "month": str(month),
                    "clock_phase": str(phase),
                    "rows": len(group),
                    "sample_rows": len(sample),
                }
            )
        fit = fit_full_right_censored_refit(
            development,
            feature_names=EMISSION_FEATURES,
            config=RefitConfig(
                state_count=8,
                seed=20260710,
                maximum_age=78,
                cleaning_variant=CleaningVariant.CLEANING_1,
            ),
            training_indices=sample,
        )
        summary = _filter_summary(prior, development, fit)
        alignment = align_states(
            primary_fit.parameters.means,
            fit.parameters.means,
            reference_transition=primary_fit.parameters.transitions,
            candidate_transition=fit.parameters.transitions,
            reference_duration=_duration_profile(primary_fit.parameters),
            candidate_duration=_duration_profile(fit.parameters),
        )
        aligned = apply_state_mapping(summary.hard_states, alignment.candidate_to_reference)
        candidate_first = _first_events(
            prior,
            development,
            aligned,
            prefix="repaired-development",
            state_count=8,
        )
        first_comparison, _ = compare_representation_events(
            reference_first_events,
            candidate_first,
            allowed_shift_bars=2,
        )
        sample_state_rows.append(
            {
                "sample_variant": variant,
                "sample_rows": len(sample),
                "training_row_hash": fit.training_row_hash,
                "parameter_hash": fit.parameter_hash,
                "bar_level_aligned_agreement": float(np.mean(aligned == reference_labels)),
                "adjusted_rand_index": float(
                    adjusted_rand_score(reference_labels, summary.hard_states)
                ),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(reference_labels, summary.hard_states)
                ),
                "minimum_state_occupancy": float(
                    state_occupancy(summary.hard_states, state_count=8).min()
                ),
            }
        )
        for loop_id in SELECTED_IDS:
            reference_count = int(reference_first_events["primary_label"].eq(loop_id).sum())
            candidate_count = int(candidate_first["primary_label"].eq(loop_id).sum())
            comparable = first_comparison["primary_label_reference"].eq(loop_id)
            sample_loop_rows.append(
                {
                    "sample_variant": variant,
                    "primitive_loop_id": loop_id,
                    "first_event_count": candidate_count,
                    "event_agreement": float(
                        first_comparison.loc[comparable, "agreement_class"]
                        .isin(
                            [
                                "EXACT_EVENT_AGREEMENT",
                                "SAME_PRIMITIVE_SHIFTED_TIMESTAMP",
                            ]
                        )
                        .mean()
                    ),
                    "dictionary_coverage_ratio": candidate_count / max(reference_count, 1),
                }
            )
        del fit, summary, candidate_first, first_comparison
        gc.collect()

    print("validity-rerun: period and representation stability", flush=True)
    assessment_scaled = transform_emissions(assessment, primary_fit.preprocessing)
    (
        semantic_profiles,
        period_drift,
        stock_heterogeneity,
        clock_heterogeneity,
        transition_drift,
    ) = _period_tables(
        development,
        assessment,
        development_scaled=primary_fit.scaled,
        assessment_scaled=assessment_scaled,
        development_labels=reference_labels,
        assessment_labels=reference_assessment_labels,
        state_count=8,
    )
    (
        representation,
        market_profiles,
        stock_profiles,
        representation_metadata,
    ) = _representation_comparison(
        prior,
        development,
        assessment,
        primary_fit=primary_fit,
        primary_summary=primary_summary,
        assessment_summary=assessment_summary,
        sample_a=bounded_stride,
    )

    state_stability_k8 = state_stability.loc[state_stability["state_count"].eq(8)]
    minimum_k8_nmi = float(state_stability_k8["normalized_mutual_information"].min())
    k8_loop = loop_stability.loc[loop_stability["state_count"].eq(8)]
    positive_counts = k8_loop.groupby("primitive_loop_id", sort=True)[
        "positive_structural_excess"
    ].sum()
    k8_seed_gate = all(
        int(positive_counts.get(loop_id, 0))
        >= int(thresholds["k8_positive_structural_excess_seed_count"])
        for loop_id in SELECTED_IDS
    )
    sample_state = pd.DataFrame(sample_state_rows)
    sample_loop = pd.DataFrame(sample_loop_rows)
    minimum_sample_coverage = float(sample_loop["dictionary_coverage_ratio"].min())
    minimum_sample_event_agreement = float(sample_loop["event_agreement"].min())
    primary_occupancy = state_occupancy(reference_labels, state_count=8)
    maximum_stock_share = float(
        stock_heterogeneity.groupby("state")["stock_share_within_state"].max().max()
    )
    semantic_drift_pass = bool(
        period_drift["period_component_gate_pass"].all()
        and float(period_drift["centroid_drift_scaled_rms"].max())
        <= float(thresholds["maximum_period_centroid_drift_scaled_rms"])
        and float(period_drift["centroid_drift_scaled_rms"].median())
        <= float(thresholds["maximum_median_period_centroid_drift_scaled_rms"])
    )
    combined_deficit = float(representation_metadata["combined_stability_deficit"])
    representation_sensitive = bool(
        minimum_k8_nmi < float(thresholds["minimum_seed_bar_agreement_nmi"])
        or minimum_sample_event_agreement
        < float(thresholds["minimum_training_sample_dictionary_coverage_ratio"])
        or hysteretic_selected_agreement < float(thresholds["hysteretic_same_primitive_minimum"])
    )
    state_language_pass = bool(
        primary_occupancy.min() >= float(thresholds["minimum_state_occupancy"])
        and maximum_stock_share <= float(thresholds["maximum_single_stock_share_per_state"])
        and semantic_drift_pass
        and combined_deficit <= float(thresholds["maximum_combined_stability_deficit"])
    )
    loop_language_pass = bool(
        hysteretic_selected_agreement >= float(thresholds["hysteretic_same_primitive_minimum"])
        and k8_seed_gate
        and minimum_sample_coverage
        >= float(thresholds["minimum_training_sample_dictionary_coverage_ratio"])
    )
    evidence = PartAGateEvidence(
        source_available=True,
        exact_reconstruction_pass=True,
        independent_audit_reproducible=False,
        mathematical_audit_pass=True,
        posterior_duration_pass=True,
        critical_future_leakage=False,
        hysteretic_same_primitive_fraction=hysteretic_selected_agreement,
        k8_selected_loop_seed_gate_pass=k8_seed_gate,
        minimum_state_occupancy=float(primary_occupancy.min()),
        maximum_single_stock_share=maximum_stock_share,
        semantic_drift_pass=semantic_drift_pass,
        training_sample_dictionary_coverage_ratio=minimum_sample_coverage,
        combined_stability_deficit=combined_deficit,
        representation_sensitive=representation_sensitive,
        usable_with_sensitivity=state_language_pass and loop_language_pass,
        recoverable_local_defect=False,
        hierarchical_materially_more_stable=bool(
            representation_metadata["hierarchical_materially_more_stable"]
        ),
        hierarchical_reproducible=bool(representation_metadata["hierarchical_reproducible"]),
    )
    decision = decide_part_a(evidence)
    gate_rows = [
        {
            "gate": "posterior_duration",
            "value": 1.0,
            "threshold": 1.0,
            "passed": True,
        },
        {
            "gate": "minimum_state_occupancy",
            "value": float(primary_occupancy.min()),
            "threshold": float(thresholds["minimum_state_occupancy"]),
            "passed": float(primary_occupancy.min())
            >= float(thresholds["minimum_state_occupancy"]),
        },
        {
            "gate": "maximum_single_stock_share",
            "value": maximum_stock_share,
            "threshold": float(thresholds["maximum_single_stock_share_per_state"]),
            "passed": maximum_stock_share
            <= float(thresholds["maximum_single_stock_share_per_state"]),
        },
        {
            "gate": "semantic_drift",
            "value": float(semantic_drift_pass),
            "threshold": 1.0,
            "passed": semantic_drift_pass,
        },
        {
            "gate": "minimum_k8_seed_nmi",
            "value": minimum_k8_nmi,
            "threshold": float(thresholds["minimum_seed_bar_agreement_nmi"]),
            "passed": minimum_k8_nmi >= float(thresholds["minimum_seed_bar_agreement_nmi"]),
        },
        {
            "gate": "hysteretic_same_primitive",
            "value": hysteretic_selected_agreement,
            "threshold": float(thresholds["hysteretic_same_primitive_minimum"]),
            "passed": hysteretic_selected_agreement
            >= float(thresholds["hysteretic_same_primitive_minimum"]),
        },
        {
            "gate": "k8_positive_structural_excess",
            "value": float(min(int(positive_counts.get(loop_id, 0)) for loop_id in SELECTED_IDS)),
            "threshold": float(thresholds["k8_positive_structural_excess_seed_count"]),
            "passed": k8_seed_gate,
        },
        {
            "gate": "training_sample_dictionary_coverage",
            "value": minimum_sample_coverage,
            "threshold": float(thresholds["minimum_training_sample_dictionary_coverage_ratio"]),
            "passed": minimum_sample_coverage
            >= float(thresholds["minimum_training_sample_dictionary_coverage_ratio"]),
        },
        {
            "gate": "combined_stability_deficit",
            "value": combined_deficit,
            "threshold": float(thresholds["maximum_combined_stability_deficit"]),
            "passed": combined_deficit <= float(thresholds["maximum_combined_stability_deficit"]),
        },
    ]
    metrics = {
        "minimum_k8_seed_nmi": minimum_k8_nmi,
        "minimum_sample_dictionary_coverage": minimum_sample_coverage,
        "minimum_sample_event_agreement": minimum_sample_event_agreement,
        "hysteretic_selected_same_primitive_fraction": (hysteretic_selected_agreement),
        "k8_positive_structural_excess_counts": {
            loop_id: int(positive_counts.get(loop_id, 0)) for loop_id in SELECTED_IDS
        },
        "minimum_state_occupancy": float(primary_occupancy.min()),
        "maximum_single_stock_share": maximum_stock_share,
        "semantic_drift_pass": semantic_drift_pass,
        "combined_stability_deficit": combined_deficit,
        "selected_market_state_count": int(representation_metadata["selected_market_state_count"]),
        "decision": decision.value,
        "part_b_opened": False,
        "dictionary_promotion_enabled": False,
    }
    tables: dict[str, pd.DataFrame] = {
        "repaired_regime_validity_metrics.csv": pd.DataFrame(gate_rows),
        "repaired_k_seed_model_registry.csv": registry,
        "repaired_state_alignment.csv": state_alignment,
        "repaired_state_stability_by_k_seed.csv": state_stability,
        "repaired_loop_stability_by_k_seed.csv": loop_stability,
        "repaired_first_event_stability_by_k_seed.csv": first_event_stability,
        "training_sample_composition.csv": pd.DataFrame(sample_composition_rows),
        "repaired_training_sample_state_stability.csv": sample_state,
        "repaired_training_sample_loop_stability.csv": sample_loop,
        "repaired_state_semantic_profiles.parquet": semantic_profiles,
        "repaired_state_period_drift.csv": period_drift,
        "state_stock_heterogeneity.csv": stock_heterogeneity,
        "state_clock_heterogeneity.csv": clock_heterogeneity,
        "state_transition_drift.csv": transition_drift,
        "repaired_state_representation_event_comparison.parquet": (representation_comparison),
        "repaired_dictionary_robustness.csv": dictionary_robustness,
        "loop_robustness_by_representation.csv": loop_robustness,
        "hard_state_churn_summary.csv": churn_summary,
        "state_transition_confidence.parquet": churn,
        "cleaning_variant_state_metrics.csv": pd.DataFrame(cleaning_state_rows),
        "cleaning_variant_loop_metrics.csv": pd.DataFrame(cleaning_loop_rows),
        "cleaning_variant_dictionary_overlap.csv": pd.DataFrame(cleaning_overlap_rows),
        "combined_stock_hierarchical_comparison.csv": representation,
        "market_regime_profiles.csv": market_profiles,
        "stock_state_profiles.csv": stock_profiles,
    }
    for variant, frame in sample_row_tables.items():
        tables[f"training_sample_rows/{variant.lower()}_rows.parquet"] = frame
    return ValidityRerunResult(
        tables=tables,
        metrics=metrics,
        evidence=evidence,
        decision=decision,
        primary_hysteretic_labels=hysteretic,
        primary_events=primary_events,
        hysteretic_events=hysteretic_events,
        primary_first_events=reference_first_events,
        hierarchical_mapping=representation_metadata["hierarchy_mapping"],
    )


__all__ = [
    "K_VALUES",
    "SAMPLE_VARIANTS",
    "SEEDS",
    "SELECTED_IDS",
    "ValidityRerunResult",
    "run_unchanged_validity_rerun",
]
