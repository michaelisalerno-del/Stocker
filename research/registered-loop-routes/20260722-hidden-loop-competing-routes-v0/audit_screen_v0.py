#!/usr/bin/env python3
"""Independent lightweight audit for the hidden competing-routes screen V0."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.hidden_loop_competing_routes_v0 import (  # noqa: E402
    FROZEN_HIDDEN_FAMILIES,
    PREREGISTERED_TARGETS,
    PROTECTED_START,
    SAFETY_FLAGS,
    benjamini_hochberg,
    candidate_threshold,
    choose_primary_decision,
    freeze_target_class_mapping,
    hidden_history_features,
    next_registered_route,
    registered_history_features,
    sequential_update_ordinals,
    session_bootstrap_multiplicities,
    target_prefix_snapshot,
    transition_hypothesis_manifest,
)

DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
BRIDGE_PRIMARY = (
    REPO_ROOT
    / "research"
    / "hidden-loop-economics"
    / "20260721-hidden-loop-economics-registered-bridge-v0"
    / "artifacts"
    / "primary"
)
PREDECESSOR_RUNNER = (
    REPO_ROOT
    / "research"
    / "registered-loop-precursors"
    / "20260721-registered-loop-precursor-hidden-veto-v0"
    / "run_screen_v0.py"
)
DEFAULT_PROVIDER_ROOT = (
    Path.home() / "StockerLocal" / "data" / "processed" / "source=eodhd" / "instrument_type=stock"
)
BOOTSTRAP_DRAWS = 25
BOOTSTRAP_SEED = 20260723
HIDDEN_NULL_DRAWS = 5
HIDDEN_NULL_SEED = 20260724
REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "target_class_mapping.json",
    "registered_completion_ledger.parquet",
    "transition_hypothesis_manifest.json",
    "transition_census.csv",
    "transition_null_metrics.csv",
    "transition_multiplicity.csv",
    "candidate_population_reconstruction.json",
    "sequential_candidate_panel.parquet",
    "sequential_weight_audit.csv",
    "feature_manifest.json",
    "model_configurations.json",
    "model_coefficients.json",
    "assessment_predictions.parquet",
    "pooled_metrics.csv",
    "target_class_metrics.csv",
    "checkpoint_metrics.csv",
    "elapsed_bar_metrics.csv",
    "monthly_metrics.csv",
    "prefix_state_metrics.csv",
    "target_specific_contrasts.csv",
    "matched_candidate_route_metrics.csv",
    "bootstrap_metrics.csv",
    "hidden_history_null_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "determinism_check.json",
)


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def _load_independent_states(provider_root: Path) -> pd.DataFrame:
    specification = importlib.util.spec_from_file_location(
        "hidden_competing_routes_independent_state_source", PREDECESSOR_RUNNER
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load predecessor state source: {PREDECESSOR_RUNNER}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    states, _, _, _ = module.load_v2_states(provider_root)
    return cast(pd.DataFrame, states)


def _candidate_ids(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["symbol"].astype(str)
        + "|"
        + frame["session"].astype(str)
        + "|"
        + frame["decision_ordinal"].astype(int).astype(str).str.zfill(2)
    )


def _expected_candidate_membership(oof: pd.DataFrame, threshold: float) -> set[tuple[str, str]]:
    key = ["symbol", "session", "decision_ordinal"]
    development_keys = oof.loc[oof["B0_oof_probability"].astype(float).ge(threshold), key]
    development = pd.read_parquet(
        BRIDGE_PRIMARY / "bridge_development_panel.parquet", columns=key
    ).merge(development_keys, on=key, how="inner", validate="one_to_one")
    assessment = pd.read_parquet(
        BRIDGE_PRIMARY / "bridge_assessment_predictions.parquet",
        columns=[*key, "B0_probability"],
    )
    assessment = assessment.loc[assessment["B0_probability"].astype(float).ge(threshold)]
    return {
        *(("development", value) for value in _candidate_ids(development)),
        *(("assessment", value) for value in _candidate_ids(assessment)),
    }


def _expected_sequential_rows(
    candidates: pd.DataFrame, registered: pd.DataFrame, states: pd.DataFrame
) -> pd.DataFrame:
    registered_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in registered.groupby(["symbol", "session"], sort=False)
    }
    state_groups = {
        (str(symbol), str(session)): group.sort_values("bar_ordinal", kind="mergesort")
        for (symbol, session), group in states.groupby(["symbol", "session"], sort=False)
    }
    empty_registered = pd.DataFrame(columns=["completion_bar_ordinal"])
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        key = (str(candidate.symbol), str(candidate.session))
        state_group = state_groups[key]
        registered_group = registered_groups.get(key, empty_registered)
        opening = int(candidate.repo_bar_start_ordinal)
        future = registered_group.loc[
            registered_group["completion_bar_ordinal"].astype(int).gt(opening)
            & registered_group["completion_bar_ordinal"].astype(int).le(opening + 12)
        ]
        first = int(future["completion_bar_ordinal"].min()) if not future.empty else None
        ordinals = sequential_update_ordinals(
            opening_ordinal=opening,
            first_completion_ordinal=first,
            available_ordinals=state_group["bar_ordinal"].astype(int).tolist(),
        )
        timestamp_lookup = state_group.set_index("bar_ordinal")["bar_complete_timestamp"]
        for update in ordinals:
            elapsed = update - opening
            rows.append(
                {
                    "period": str(candidate.period),
                    "candidate_id": str(candidate.candidate_id),
                    "symbol": str(candidate.symbol),
                    "session": str(candidate.session),
                    "elapsed_bar": elapsed,
                    "update_bar_ordinal": update,
                    "update_timestamp_utc": pd.Timestamp(timestamp_lookup.loc[update]),
                    "sequential_row_id": f"{candidate.candidate_id}|E{elapsed}",
                }
            )
    expected = pd.DataFrame(rows)
    return (
        expected.sort_values(
            ["period", "session", "symbol", "update_timestamp_utc", "elapsed_bar"],
            kind="mergesort",
        )
        .drop_duplicates(["period", "symbol", "session", "update_timestamp_utc"], keep="first")
        .reset_index(drop=True)
    )


def _independent_model_coefficients(
    specification: Mapping[str, Any], development: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    features = [str(value) for value in specification["feature_names"]]
    matrix = development.loc[:, features].apply(pd.to_numeric, errors="raise").to_numpy(float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    kwargs: dict[str, Any] = {
        "penalty": "l2",
        "C": 0.25,
        "solver": "lbfgs",
        "max_iter": 300,
        "class_weight": None,
        "random_state": 20260722,
    }
    signature = inspect.signature(LogisticRegression)
    if "multi_class" in signature.parameters:
        kwargs["multi_class"] = "multinomial"
    if "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = 1
    model = LogisticRegression(**kwargs)
    model.fit(
        (matrix - mean) / scale,
        development["next_registered_route"].astype(str).to_numpy(),
        sample_weight=development["sequential_row_weight"].to_numpy(float),
    )
    return (
        mean,
        scale,
        np.asarray(model.coef_, dtype=float),
        np.asarray(model.intercept_, dtype=float),
        tuple(str(value) for value in model.classes_),
    )


def _manual_probabilities(specification: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    features = [str(value) for value in specification["feature_names"]]
    matrix = frame.loc[:, features].apply(pd.to_numeric, errors="raise").to_numpy(float)
    mean = np.asarray(specification["scaler_mean"], dtype=float)
    scale = np.asarray(specification["scaler_scale"], dtype=float)
    coefficient = np.asarray(specification["coefficient"], dtype=float)
    intercept = np.asarray(specification["intercept"], dtype=float)
    logits = ((matrix - mean) / scale) @ coefficient.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    vector = np.asarray(values, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    if vector.size == 0:
        return math.nan
    return float(np.sum(vector * sample_weight) / sample_weight.sum())


def _manual_log_loss_brier(
    frame: pd.DataFrame, probabilities: np.ndarray, classes: Sequence[str]
) -> tuple[float, float]:
    lookup = {str(value): index for index, value in enumerate(classes)}
    targets = np.asarray(
        [lookup[str(value)] for value in frame["next_registered_route"]], dtype=int
    )
    weights = frame["sequential_row_weight"].to_numpy(float)
    realised = probabilities[np.arange(len(frame)), targets]
    log_loss = float(np.sum(weights * -np.log(np.clip(realised, 1e-15, 1.0))) / weights.sum())
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(frame)), targets] = 1.0
    brier = float(np.sum(weights * np.sum((probabilities - one_hot) ** 2, axis=1)) / weights.sum())
    return log_loss, brier


def _weighted_metric_triplet(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    classes: Sequence[str],
    weight_column: str,
) -> dict[str, float]:
    lookup = {str(value): index for index, value in enumerate(classes)}
    targets = np.asarray(
        [lookup[str(value)] for value in frame["next_registered_route"]], dtype=int
    )
    weights = frame[weight_column].to_numpy(float)
    realised = probabilities[np.arange(len(frame)), targets]
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(frame)), targets] = 1.0
    top_two = np.argsort(-probabilities, axis=1, kind="mergesort")[:, :2]
    return {
        "multiclass_log_loss": float(
            np.sum(weights * -np.log(np.clip(realised, 1e-15, 1.0))) / weights.sum()
        ),
        "multiclass_brier": float(
            np.sum(weights * np.sum((probabilities - one_hot) ** 2, axis=1)) / weights.sum()
        ),
        "top_two_accuracy": float(
            np.sum(weights * (top_two == targets[:, None]).any(axis=1)) / weights.sum()
        ),
    }


def _bootstrap_draw_values(
    predictions: pd.DataFrame,
    coefficients: Mapping[str, Any],
    session_counts: Mapping[str, int],
) -> dict[str, float]:
    sampled = predictions.copy()
    sampled["bootstrap_multiplicity"] = sampled["session"].astype(str).map(session_counts).fillna(0)
    sampled = sampled.loc[sampled["bootstrap_multiplicity"].gt(0)].copy()
    sampled["bootstrap_weight"] = (
        sampled["sequential_row_weight"] * sampled["bootstrap_multiplicity"]
    )
    model_metrics: dict[str, dict[str, float]] = {}
    for name in ("C0", "C1", "C2"):
        classes = [str(value) for value in coefficients[name]["classes"]]
        probabilities = sampled.loc[
            :, [f"{name}_probability__{value}" for value in classes]
        ].to_numpy(float)
        model_metrics[name] = _weighted_metric_triplet(
            sampled, probabilities, classes, "bootstrap_weight"
        )
    values = {
        "C1_minus_C0_log_loss_improvement": model_metrics["C0"]["multiclass_log_loss"]
        - model_metrics["C1"]["multiclass_log_loss"],
        "C1_minus_C0_brier_improvement": model_metrics["C0"]["multiclass_brier"]
        - model_metrics["C1"]["multiclass_brier"],
        "C1_minus_C0_top_two_change": model_metrics["C1"]["top_two_accuracy"]
        - model_metrics["C0"]["top_two_accuracy"],
        "C2_minus_C1_log_loss_improvement": model_metrics["C1"]["multiclass_log_loss"]
        - model_metrics["C2"]["multiclass_log_loss"],
        "C2_minus_C1_brier_improvement": model_metrics["C1"]["multiclass_brier"]
        - model_metrics["C2"]["multiclass_brier"],
        "C2_minus_C1_top_two_change": model_metrics["C2"]["top_two_accuracy"]
        - model_metrics["C1"]["top_two_accuracy"],
    }
    contrast_map = {
        "A_hidden_5_6_5_to_target_a_probability_effect": (
            "contrast_effect__A_hidden_5_6_5_to_target_a",
            "hidden_5_6_5_seen_since_opening",
        ),
        "B_hidden_5_6_5_to_target_b_probability_effect": (
            "contrast_effect__B_hidden_5_6_5_to_target_b",
            "hidden_5_6_5_seen_since_opening",
        ),
        "C_recent_4_6_4_to_target_c_probability_effect": (
            "contrast_effect__C_recent_4_6_4_to_target_c",
            "loop_p_4_6_4_completed_previous_6_bars",
        ),
        "D_hidden_2_3_2_to_any_registered_probability_effect": (
            "contrast_effect__D_hidden_2_3_2_to_any_registered",
            "hidden_2_3_2_seen_since_opening",
        ),
    }
    for metric, (effect_column, observed_column) in contrast_map.items():
        group = sampled.loc[
            sampled[observed_column].astype(float).gt(0) & sampled[effect_column].notna()
        ]
        values[metric] = _weighted_mean(group[effect_column], group["bootstrap_weight"])
    candidates = predictions.sort_values("elapsed_bar", kind="mergesort").drop_duplicates(
        "candidate_id"
    )
    candidate_multiplicity = candidates["session"].astype(str).map(session_counts).fillna(0)
    candidates = candidates.loc[candidate_multiplicity.gt(0)]
    candidate_weights = (
        candidates["candidate_total_weight"] * candidate_multiplicity.loc[candidates.index]
    )
    values["candidate_level_any_registered_rate"] = _weighted_mean(
        candidates["original_registered_candidate"], candidate_weights
    )
    return values


def _permute_hidden_bundle(
    panel: pd.DataFrame, bundle_columns: Sequence[str], *, seed: int
) -> pd.DataFrame:
    result = panel.copy()
    generator = np.random.default_rng(seed)
    groups = ["period", "session", "opening_checkpoint", "elapsed_bar"]
    for _, indices in result.groupby(groups, sort=True, dropna=False).groups.items():
        positions = np.asarray(sorted(indices), dtype=int)
        if positions.size <= 1:
            continue
        bundles = result.loc[positions, list(bundle_columns)].to_numpy(copy=True)
        result.loc[positions, list(bundle_columns)] = bundles[generator.permutation(positions.size)]
    return result


def _fit_null_probabilities(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    features: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    train = development.loc[:, list(features)].to_numpy(float)
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    kwargs: dict[str, Any] = {
        "penalty": "l2",
        "C": 0.25,
        "solver": "lbfgs",
        "max_iter": 300,
        "class_weight": None,
        "random_state": 20260722,
    }
    signature = inspect.signature(LogisticRegression)
    if "multi_class" in signature.parameters:
        kwargs["multi_class"] = "multinomial"
    if "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = 1
    model = LogisticRegression(**kwargs)
    model.fit(
        (train - mean) / scale,
        development["next_registered_route"].astype(str).to_numpy(),
        sample_weight=development["sequential_row_weight"].to_numpy(float),
    )
    probabilities = model.predict_proba(
        (assessment.loc[:, list(features)].to_numpy(float) - mean) / scale
    )
    return np.asarray(probabilities, dtype=float), tuple(str(value) for value in model.classes_)


def audit_artifacts(output: Path, *, provider_root: Path = DEFAULT_PROVIDER_ROOT) -> dict[str, Any]:
    """Independently verify frozen artifacts and fail closed on any discrepancy."""

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    checks["required_artifacts_present"] = all(
        (output / value).exists() for value in REQUIRED_ARTIFACTS
    )
    if not checks["required_artifacts_present"]:
        missing = [value for value in REQUIRED_ARTIFACTS if not (output / value).exists()]
        return {**SAFETY_FLAGS, "passed": False, "checks": checks, "missing": missing}
    contract = read_json(output / "contract.json")
    decision = read_json(output / "decision.json")
    source = read_json(output / "source_manifest.json")
    boundary = read_json(output / "protected_boundary_audit.json")
    mapping = read_json(output / "target_class_mapping.json")
    hypothesis_manifest = read_json(output / "transition_hypothesis_manifest.json")
    feature_manifest = read_json(output / "feature_manifest.json")
    configurations = read_json(output / "model_configurations.json")
    coefficients = read_json(output / "model_coefficients.json")
    reconstruction = read_json(output / "candidate_population_reconstruction.json")
    determinism = read_json(output / "determinism_check.json")
    registered = pd.read_parquet(output / "registered_completion_ledger.parquet")
    hidden = pd.read_parquet(output / "hidden_completion_ledger.parquet")
    prefixes = pd.read_parquet(output / "prefix_activity_ledger.parquet")
    transition_panel = pd.read_parquet(output / "transition_event_panel.parquet")
    transition_census = pd.read_csv(output / "transition_census.csv")
    transition_null = pd.read_csv(output / "transition_null_metrics.csv")
    multiplicity = pd.read_csv(output / "transition_multiplicity.csv")
    candidates = pd.read_parquet(output / "candidate_population.parquet")
    oof = pd.read_parquet(output / "b0_development_oof_predictions.parquet")
    panel = pd.read_parquet(output / "sequential_candidate_panel.parquet")
    predictions = pd.read_parquet(output / "assessment_predictions.parquet")
    pooled = pd.read_csv(output / "pooled_metrics.csv")
    contrasts = pd.read_csv(output / "target_specific_contrasts.csv")
    matched = pd.read_csv(output / "matched_candidate_route_metrics.csv")
    relations = pd.read_parquet(output / "matched_candidate_relations.parquet")
    bootstrap = pd.read_csv(output / "bootstrap_metrics.csv")
    hidden_null = pd.read_csv(output / "hidden_history_null_metrics.csv")
    concentration = pd.read_csv(output / "concentration_metrics.csv")
    states = _load_independent_states(provider_root)

    checks["safety_flags"] = all(
        contract.get(key) == value and decision.get(key) == value
        for key, value in SAFETY_FLAGS.items()
    )
    checks["dates_and_protected_boundary"] = (
        source["development_period"] == ["2024-01-01", "2024-12-31"]
        and source["assessment_period"] == ["2025-01-01", "2025-08-22"]
        and int(source["protected_rows_materialised"]) == 0
        and int(boundary["protected_rows_materialised"]) == 0
        and bool(boundary["passed"])
        and pd.Timestamp(source["maximum_timestamp_read"]) < PROTECTED_START
        and pd.to_datetime(panel["session"], utc=True).max() < PROTECTED_START
        and pd.to_datetime(states["bar_complete_timestamp"], utc=True).max() < PROTECTED_START
    )
    checks["frozen_registered_targets"] = (
        tuple(mapping["preregistered_targets"]) == PREREGISTERED_TARGETS
    )
    checks["frozen_hidden_families"] = tuple(contract["hidden_families"]) == FROZEN_HIDDEN_FAMILIES
    recalculated_mapping = freeze_target_class_mapping(registered.loc[registered["year"].eq(2024)])
    checks["development_only_target_mapping"] = (
        mapping["fit_period"] == "2024_only"
        and mapping["assessment_support_inspected_before_freeze"] is False
        and recalculated_mapping["retained_exact_targets"] == mapping["retained_exact_targets"]
        and recalculated_mapping["final_target_classes"] == mapping["final_target_classes"]
    )
    identity = [
        "symbol",
        "session",
        "completion_timestamp_utc",
        "semantic_loop_id",
        "orientation_id",
    ]
    checks["registered_event_deduplication"] = not registered.duplicated(identity).any()
    available_ordinals = {
        (str(symbol), str(session)): set(group["bar_ordinal"].astype(int))
        for (symbol, session), group in states.groupby(["symbol", "session"], sort=False)
    }
    eligibility_checks: list[bool] = []
    transition_support = cast(Mapping[str, Any], decision["transition_support"])
    lookback_support = cast(Mapping[str, Any], transition_support["lookbacks"])
    hypothesis_targets = {
        "H1": PREREGISTERED_TARGETS[0],
        "H2": PREREGISTERED_TARGETS[1],
        "H3": PREREGISTERED_TARGETS[2],
        "H4": "ANY_REGISTERED_COMPLETION",
    }
    for period, year in (("development", 2024), ("assessment", 2025)):
        period_registered = registered.loc[registered["year"].eq(year)].copy()
        for lookback in (6, 12):
            complete = []
            for row in period_registered.itertuples(index=False):
                ordinal = int(row.completion_bar_ordinal)
                available = available_ordinals[(str(row.symbol), str(row.session))]
                complete.append(set(range(ordinal - lookback, ordinal)).issubset(available))
            period_registered[f"eligible_{lookback}"] = complete
            eligible = period_registered.loc[period_registered[f"eligible_{lookback}"]]
            archived_support = cast(Mapping[str, Any], lookback_support[f"{period}_{lookback}"])
            eligibility_checks.append(
                int(archived_support["eligible_events"]) == len(eligible)
                and int(archived_support["ineligible_events"])
                == len(period_registered) - len(eligible)
                and int(archived_support["sessions"]) == eligible["session"].nunique()
                and int(archived_support["stocks"]) == eligible["symbol"].nunique()
                and int(archived_support["months"]) == eligible["year_month"].nunique()
            )
            for census_row in transition_census.loc[
                transition_census["period"].eq(period)
                & transition_census["lookback_bars"].eq(lookback)
            ].itertuples(index=False):
                target = hypothesis_targets[str(census_row.hypothesis_id)]
                target_events = (
                    period_registered
                    if target == "ANY_REGISTERED_COMPLETION"
                    else period_registered.loc[
                        period_registered["semantic_loop_id"].astype(str).eq(target)
                    ]
                )
                target_eligible = target_events.loc[target_events[f"eligible_{lookback}"]]
                observed_ids = set(
                    transition_panel.loc[
                        transition_panel["record_type"].eq("observed")
                        & transition_panel["period"].eq(period)
                        & transition_panel["hypothesis_id"].eq(census_row.hypothesis_id)
                        & transition_panel["lookback_bars"].eq(lookback),
                        "source_event_id",
                    ].astype(str)
                )
                eligibility_checks.append(
                    int(census_row.eligible_events) == len(target_eligible)
                    and int(census_row.ineligible_events)
                    == len(target_events) - len(target_eligible)
                    and observed_ids == set(target_eligible["event_id"].astype(str))
                )
    checks["lookback_specific_eligibility"] = set(
        transition_census["lookback_bars"].astype(int)
    ) == {6, 12} and all(eligibility_checks)
    expected_hypotheses = transition_hypothesis_manifest()
    checks["every_fixed_transition_hypothesis"] = hypothesis_manifest[
        "hypotheses"
    ] == expected_hypotheses and set(transition_census["hypothesis_id"]) == {"H1", "H2", "H3", "H4"}
    source_lookup = registered.set_index("event_id")
    null_rows = transition_panel.loc[transition_panel["record_type"].eq("null")]
    source_symbols = null_rows["source_event_id"].map(source_lookup["symbol"].astype(str))
    source_months = null_rows["source_event_id"].map(source_lookup["year_month"].astype(str))
    state_clock_lookup = {
        (str(row.symbol), str(row.session), int(row.bar_ordinal)): pd.Timestamp(
            row.bar_start_timestamp
        )
        .tz_convert("America/New_York")
        .floor("30min")
        .strftime("%H:%M")
        for row in states.itertuples(index=False)
    }
    pseudo_clock = pd.Series(
        [
            state_clock_lookup[(str(row.symbol), str(row.session), int(row.completion_bar_ordinal))]
            for row in null_rows.itertuples(index=False)
        ],
        index=null_rows.index,
    )
    checks["stock_clock_matched_null"] = (
        null_rows["symbol"].astype(str).eq(source_symbols.astype(str)).all()
        and null_rows["session"].astype(str).str[:7].eq(source_months.astype(str)).all()
        and null_rows["source_clock_bin"]
        .astype(str)
        .eq(null_rows["source_event_id"].map(source_lookup["clock_bin"].astype(str)))
        .all()
        and null_rows["clock_bin"].astype(str).eq(pseudo_clock.astype(str)).all()
        and null_rows["clock_bin"].astype(str).eq(null_rows["source_clock_bin"].astype(str)).all()
        and transition_panel.loc[transition_panel["record_type"].eq("null"), "draw"].nunique() == 25
        and transition_null.loc[transition_null["record_type"].eq("draw"), "draw"].nunique() == 25
    )
    recalculated_q = benjamini_hochberg(multiplicity["p_value"].astype(float).tolist())
    q_difference = float(
        np.max(np.abs(np.asarray(recalculated_q) - multiplicity["q_value"].to_numpy(float)))
    )
    checks["bh_correction"] = q_difference <= 1e-12 and len(multiplicity) == 4
    threshold = candidate_threshold(oof["B0_oof_probability"])
    checks["frozen_b0_threshold"] = abs(
        threshold - float(reconstruction["frozen_threshold"])
    ) <= 1e-12 and bool(reconstruction["threshold_verified_to_1e_12"])
    assessment_candidates = candidates.loc[candidates["period"].eq("assessment")]
    expected_candidate_membership = _expected_candidate_membership(oof, threshold)
    actual_candidate_membership = set(
        zip(candidates["period"].astype(str), candidates["candidate_id"].astype(str), strict=True)
    )
    checks["candidate_population_reconstruction"] = (
        len(assessment_candidates) == int(reconstruction["assessment_candidate_rows"])
        and assessment_candidates["candidate_B0_probability"].astype(float).ge(threshold).all()
        and candidates["candidate_id"].is_unique
        and actual_candidate_membership == expected_candidate_membership
    )
    expected_rows = _expected_sequential_rows(candidates, registered, states)
    sequential_comparison = panel.loc[
        :, ["sequential_row_id", "update_bar_ordinal", "update_timestamp_utc"]
    ].merge(
        expected_rows.loc[:, ["sequential_row_id", "update_bar_ordinal", "update_timestamp_utc"]],
        on="sequential_row_id",
        how="outer",
        suffixes=("_actual", "_expected"),
        indicator=True,
        validate="one_to_one",
    )
    checks["sequential_update_timestamps"] = (
        panel["elapsed_bar"].between(0, 6).all()
        and panel["update_bar_ordinal"]
        .astype(int)
        .eq(panel["opening_bar_ordinal"].astype(int) + panel["elapsed_bar"].astype(int))
        .all()
        and not panel.duplicated(["period", "symbol", "session", "update_timestamp_utc"]).any()
        and sequential_comparison["_merge"].eq("both").all()
        and sequential_comparison["update_bar_ordinal_actual"]
        .astype(int)
        .eq(sequential_comparison["update_bar_ordinal_expected"].astype(int))
        .all()
        and pd.to_datetime(sequential_comparison["update_timestamp_utc_actual"], utc=True)
        .eq(pd.to_datetime(sequential_comparison["update_timestamp_utc_expected"], utc=True))
        .all()
    )
    checks["stop_after_completion"] = (
        panel["update_bar_ordinal"]
        .astype(int)
        .le(panel["first_registered_completion_bar_ordinal"].fillna(math.inf))
        .all()
    )
    checks["original_horizon_target"] = (
        panel["next_completion_bar_ordinal"].dropna().astype(int)
        <= panel.loc[
            panel["next_completion_bar_ordinal"].notna(), "original_horizon_end_ordinal"
        ].astype(int)
    ).all()
    weight_difference = (
        panel.groupby("candidate_id")["sequential_row_weight"]
        .sum()
        .sub(panel.groupby("candidate_id")["candidate_total_weight"].first())
        .abs()
    )
    checks["candidate_normalised_weighting"] = float(weight_difference.max()) <= 1e-12
    state_columns = [f"current_state_p_{value}" for value in range(8)]
    source_state_columns = [f"state_p_{value}" for value in range(8)]
    state_surface = states.loc[
        :,
        [
            "symbol",
            "session",
            "bar_ordinal",
            "bar_complete_timestamp",
            "posterior_entropy_reproduced",
            "expected_state_age",
            "persistence_probability",
            "transition_probability",
            "causal_hard_state",
            *source_state_columns,
        ],
    ]
    current_comparison = panel.merge(
        state_surface,
        left_on=["symbol", "session", "update_bar_ordinal"],
        right_on=["symbol", "session", "bar_ordinal"],
        how="left",
        validate="many_to_one",
    )
    source_probability = current_comparison.loc[:, source_state_columns].to_numpy(float)
    ordered_probability = np.sort(source_probability, axis=1)
    state_groups = {
        (str(symbol), str(session)): group.sort_values("bar_ordinal", kind="mergesort")
        for (symbol, session), group in states.groupby(["symbol", "session"], sort=False)
    }
    expected_transition_counts: list[int] = []
    for row in panel.itertuples(index=False):
        group = state_groups[(str(row.symbol), str(row.session))]
        path = group.loc[
            group["bar_ordinal"]
            .astype(int)
            .between(int(row.opening_bar_ordinal), int(row.update_bar_ordinal)),
            "causal_hard_state",
        ].to_numpy(int)
        expected_transition_counts.append(int((path[1:] != path[:-1]).sum()))
    checks["current_regime_fields"] = (
        np.isfinite(panel.loc[:, state_columns].to_numpy(float)).all()
        and np.allclose(panel.loc[:, state_columns].sum(axis=1), 1.0, atol=1e-10)
        and panel["current_transition_probability"].between(0, 1).all()
        and np.allclose(panel.loc[:, state_columns].to_numpy(float), source_probability, atol=1e-12)
        and np.allclose(
            panel["current_posterior_entropy"].to_numpy(float),
            current_comparison["posterior_entropy_reproduced"].to_numpy(float),
            atol=1e-12,
        )
        and np.allclose(
            panel["current_top_state_probability"].to_numpy(float),
            ordered_probability[:, -1],
            atol=1e-12,
        )
        and np.allclose(
            panel["current_top_second_margin"].to_numpy(float),
            ordered_probability[:, -1] - ordered_probability[:, -2],
            atol=1e-12,
        )
        and np.allclose(
            panel["current_expected_state_age"].to_numpy(float),
            current_comparison["expected_state_age"].to_numpy(float),
            atol=1e-12,
        )
        and np.allclose(
            panel["current_persistence_probability"].to_numpy(float),
            current_comparison["persistence_probability"].to_numpy(float),
            atol=1e-12,
        )
        and np.allclose(
            panel["current_transition_probability"].to_numpy(float),
            current_comparison["transition_probability"].to_numpy(float),
            atol=1e-12,
        )
        and np.array_equal(
            panel["regime_transitions_since_opening"].to_numpy(int),
            np.asarray(expected_transition_counts, dtype=int),
        )
        and pd.to_datetime(panel["update_timestamp_utc"], utc=True)
        .eq(pd.to_datetime(current_comparison["bar_complete_timestamp"], utc=True))
        .all()
    )
    target_metadata = cast(Mapping[str, Any], feature_manifest["prefix_metadata"])
    prefix_checks: list[bool] = []
    prefix_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in prefixes.groupby(["symbol", "session"], sort=False)
    }
    for row in panel.itertuples(index=False):
        history = prefix_groups.get(
            (str(row.symbol), str(row.session)),
            pd.DataFrame(
                columns=["bar_ordinal", "semantic_loop_id", "orientation_id", "progress_states"]
            ),
        )
        for target in mapping["retained_exact_targets"]:
            alias = {
                PREREGISTERED_TARGETS[0]: "target_a",
                PREREGISTERED_TARGETS[1]: "target_b",
                PREREGISTERED_TARGETS[2]: "target_c",
            }[target]
            metadata = target_metadata[target]
            snapshot = target_prefix_snapshot(
                history,
                current_ordinal=int(row.update_bar_ordinal),
                target_identity=target,
                canonical_orientation_id=str(metadata["canonical_orientation_id"]),
                transition_length=int(metadata["full_transition_length"]),
            )
            columns = {
                "active": f"{alias}_prefix_active",
                "depth": f"{alias}_prefix_depth",
                "fraction": f"{alias}_prefix_fraction",
                "bars_since_first_active": f"{alias}_prefix_bars_since_first_active",
                "bars_since_first_active_missing": (
                    f"{alias}_prefix_bars_since_first_active_missing"
                ),
                "canonical_orientation_match": f"{alias}_prefix_canonical_orientation_match",
                "conflicting_prefix_active": f"{alias}_conflicting_prefix_active",
            }
            prefix_checks.append(
                all(
                    abs(float(getattr(row, column)) - snapshot[name]) <= 1e-12
                    for name, column in columns.items()
                )
            )
    checks["target_specific_prefix_fields"] = all(prefix_checks)
    registered_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in registered.groupby(["symbol", "session"], sort=False)
    }
    hidden_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in hidden.groupby(["symbol", "session"], sort=False)
    }
    history_checks: list[bool] = []
    target_checks: list[bool] = []
    for row in panel.itertuples(index=False):
        key = (str(row.symbol), str(row.session))
        registered_group = registered_groups.get(
            key,
            pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id", "orientation_id"]),
        )
        hidden_group = hidden_groups.get(
            key, pd.DataFrame(columns=["completion_bar_ordinal", "hidden_family_class"])
        )
        registered_features = registered_history_features(
            registered_group,
            opening_ordinal=int(row.opening_bar_ordinal),
            current_ordinal=int(row.update_bar_ordinal),
        )
        hidden_features = hidden_history_features(
            hidden_group,
            opening_ordinal=int(row.opening_bar_ordinal),
            current_ordinal=int(row.update_bar_ordinal),
        )
        history_checks.append(
            all(
                abs(float(getattr(row, key_name)) - value) <= 1e-12
                for key_name, value in registered_features.items()
            )
            and all(
                abs(float(getattr(row, key_name)) - value) <= 1e-12
                for key_name, value in hidden_features.items()
            )
        )
        expected_class, expected_identity, expected_ordinal = next_registered_route(
            registered_group,
            update_ordinal=int(row.update_bar_ordinal),
            horizon_end_ordinal=int(row.original_horizon_end_ordinal),
            retained_targets=mapping["retained_exact_targets"],
        )
        target_checks.append(
            str(row.next_registered_route) == expected_class
            and (
                None
                if pd.isna(row.next_registered_semantic_loop_id)
                else str(row.next_registered_semantic_loop_id)
            )
            == expected_identity
            and (
                None
                if pd.isna(row.next_completion_bar_ordinal)
                else int(row.next_completion_bar_ordinal)
            )
            == expected_ordinal
        )
    checks["recent_registered_and_hidden_history"] = all(history_checks)
    checks["original_horizon_target_construction"] = all(target_checks)
    all_features = set(panel.columns)
    checks["C0_C1_C2_feature_ladder"] = all(
        set(configurations[name]["features"]).issubset(all_features) for name in ("C0", "C1", "C2")
    ) and set(configurations["C0"]["features"]) < set(configurations["C1"]["features"]) < set(
        configurations["C2"]["features"]
    )
    development_panel = panel.loc[panel["period"].eq("development")]
    coefficient_differences: list[float] = []
    coefficient_class_checks: list[bool] = []
    for name in ("C0", "C1", "C2"):
        specification = cast(Mapping[str, Any], coefficients[name])
        mean, scale, coefficient, intercept, classes = _independent_model_coefficients(
            specification, development_panel
        )
        coefficient_differences.extend(
            np.abs(mean - np.asarray(specification["scaler_mean"], dtype=float)).ravel()
        )
        coefficient_differences.extend(
            np.abs(scale - np.asarray(specification["scaler_scale"], dtype=float)).ravel()
        )
        coefficient_differences.extend(
            np.abs(coefficient - np.asarray(specification["coefficient"], dtype=float)).ravel()
        )
        coefficient_differences.extend(
            np.abs(intercept - np.asarray(specification["intercept"], dtype=float)).ravel()
        )
        coefficient_class_checks.append(
            classes == tuple(str(value) for value in specification["classes"])
        )
    maximum_coefficient_difference = float(max(coefficient_differences, default=0.0))
    checks["model_coefficients"] = maximum_coefficient_difference <= 1e-12 and all(
        coefficient_class_checks
    )
    manual_rows = predictions.sort_values("sequential_row_id", kind="mergesort").head(100)
    probability_differences: list[float] = []
    metric_differences: list[float] = []
    pooled_lookup = pooled.set_index("model")
    for model in ("C0", "C1", "C2"):
        specification = cast(Mapping[str, Any], coefficients[model])
        manual = _manual_probabilities(specification, manual_rows)
        classes = [str(value) for value in specification["classes"]]
        archived = manual_rows.loc[
            :, [f"{model}_probability__{value}" for value in classes]
        ].to_numpy(float)
        probability_differences.extend(np.abs(manual - archived).ravel())
        full_manual = _manual_probabilities(specification, predictions)
        log_loss, brier = _manual_log_loss_brier(predictions, full_manual, classes)
        metric_differences.extend(
            [
                abs(log_loss - float(pooled_lookup.loc[model, "multiclass_log_loss"])),
                abs(brier - float(pooled_lookup.loc[model, "multiclass_brier"])),
            ]
        )
    maximum_probability_difference = float(max(probability_differences, default=0.0))
    maximum_metric_difference = float(max(metric_differences, default=0.0))
    checks["manual_probability_reconstruction_100_rows"] = maximum_probability_difference <= 1e-12
    checks["multiclass_log_loss_and_brier"] = maximum_metric_difference <= 1e-12
    c2 = cast(Mapping[str, Any], coefficients["C2"])
    contrast_differences: list[float] = []
    for row in contrasts.loc[contrasts["scope_type"].eq("pooled")].itertuples(index=False):
        zero_features = cast(list[str], json.loads(str(row.features_zeroed)))
        target_classes = cast(list[str], json.loads(str(row.target_classes)))
        observed_column = str(row.feature_observed)
        treated = predictions.loc[predictions[observed_column].astype(float).gt(0)].copy()
        if (
            treated.empty
            or not target_classes
            or not all(value in c2["classes"] for value in target_classes)
        ):
            continue
        original = _manual_probabilities(c2, treated)
        counterfactual = treated.copy()
        counterfactual.loc[:, zero_features] = 0.0
        changed = _manual_probabilities(c2, counterfactual)
        indices = [list(c2["classes"]).index(value) for value in target_classes]
        effect = (original[:, indices] - changed[:, indices]).sum(axis=1)
        recalculated = _weighted_mean(effect, treated["sequential_row_weight"])
        contrast_differences.append(
            abs(recalculated - float(row.probability_effect_original_minus_counterfactual))
        )
    checks["target_specific_counterfactual_contrasts"] = (
        float(max(contrast_differences, default=0.0)) <= 1e-12
    )
    if relations.empty:
        checks["matched_candidate_panel"] = matched["treated_rows"].eq(0).all()
    else:
        relation_sums = relations.groupby(["precursor", "treated_row_id"])[
            "control_weight_within_treated"
        ].sum()
        checks["matched_candidate_panel"] = (
            np.allclose(relation_sums.to_numpy(float), 1.0, atol=1e-12)
            and relations.groupby(["precursor", "treated_row_id"]).size().ge(5).all()
        )
    draw_bootstrap = bootstrap.loc[bootstrap["record_type"].eq("draw")]
    bootstrap_checks: list[bool] = [draw_bootstrap["draw"].nunique() == 25]
    bootstrap_multiplicities = session_bootstrap_multiplicities(
        predictions["session"].astype(str).tolist(),
        draws=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
    )
    row_session = predictions.set_index("sequential_row_id")["session"].astype(str)
    for draw, counts in enumerate(bootstrap_multiplicities):
        draw_group = draw_bootstrap.loc[draw_bootstrap["draw"].eq(draw)]
        archived_values = draw_group.set_index("metric")["value"]
        recalculated_values = _bootstrap_draw_values(predictions, coefficients, counts)
        for metric, value in recalculated_values.items():
            archived_value = float(archived_values.loc[metric])
            bootstrap_checks.append(
                (math.isnan(value) and math.isnan(archived_value))
                or abs(value - archived_value) <= 1e-12
            )
        if relations.empty:
            expected_relation_rows = 0
            expected_weighted_relations = 0
            expected_treated_rows = 0
            expected_weighted_treated = 0
        else:
            relation_multiplicity = (
                relations["treated_row_id"].astype(str).map(row_session).map(counts).fillna(0)
            )
            preserved = relations.loc[relation_multiplicity.gt(0)]
            treated_multiplicity = (
                preserved[["treated_row_id"]]
                .drop_duplicates()["treated_row_id"]
                .astype(str)
                .map(row_session)
                .map(counts)
                .fillna(0)
            )
            expected_relation_rows = len(preserved)
            expected_weighted_relations = int(relation_multiplicity.loc[preserved.index].sum())
            expected_treated_rows = len(treated_multiplicity)
            expected_weighted_treated = int(treated_multiplicity.sum())
        bootstrap_checks.append(
            draw_group["matched_relation_rows"].eq(expected_relation_rows).all()
            and draw_group["matched_relation_weighted_rows"].eq(expected_weighted_relations).all()
            and draw_group["matched_treated_rows"].eq(expected_treated_rows).all()
            and draw_group["matched_treated_weighted_rows"].eq(expected_weighted_treated).all()
        )
    for row in bootstrap.loc[bootstrap["record_type"].eq("summary")].itertuples(index=False):
        values = (
            draw_bootstrap.loc[draw_bootstrap["metric"].eq(row.metric), "value"]
            .dropna()
            .to_numpy(float)
        )
        alpha = (1.0 - float(row.interval_level)) / 2.0
        bootstrap_checks.append(
            abs(float(row.lower) - float(np.quantile(values, alpha, method="linear"))) <= 1e-12
            and abs(float(row.upper) - float(np.quantile(values, 1 - alpha, method="linear")))
            <= 1e-12
        )
    checks["whole_session_bootstrap"] = all(bootstrap_checks)
    concentration_lookup = concentration.set_index(["scope", "gate"])
    support = cast(Mapping[str, Any], decision["sequential_model_support"])
    concentration_checks = []
    for gate, decision_key in (
        ("maximum_candidate_stock_share", "maximum_candidate_stock_share"),
        ("maximum_weighted_stock_share", "maximum_weighted_stock_share"),
    ):
        row = concentration_lookup.loc[("assessment", gate)]
        concentration_checks.append(
            abs(float(row["value"]) - float(support[decision_key])) <= 1e-12
            and not bool(row["passed"])
        )
    checks["concentration_metrics"] = all(concentration_checks)
    draw_null = hidden_null.loc[hidden_null["record_type"].eq("draw")]
    null_checks = [draw_null["draw"].nunique() == HIDDEN_NULL_DRAWS]
    c1_specification = cast(Mapping[str, Any], coefficients["C1"])
    c1_classes = tuple(str(value) for value in c1_specification["classes"])
    c1_probabilities = _manual_probabilities(c1_specification, predictions)
    c1_metrics = _weighted_metric_triplet(
        predictions, c1_probabilities, c1_classes, "sequential_row_weight"
    )
    c1_features = set(str(value) for value in configurations["C1"]["features"])
    c2_features = [str(value) for value in configurations["C2"]["features"]]
    hidden_bundle = [value for value in c2_features if value not in c1_features]
    development = panel.loc[panel["period"].eq("development")].copy()
    assessment = panel.loc[panel["period"].eq("assessment")].copy()
    for draw in range(HIDDEN_NULL_DRAWS):
        shuffled_development = _permute_hidden_bundle(
            development, hidden_bundle, seed=HIDDEN_NULL_SEED + draw
        )
        shuffled_assessment = _permute_hidden_bundle(
            assessment, hidden_bundle, seed=HIDDEN_NULL_SEED + draw
        )
        null_probabilities, null_classes = _fit_null_probabilities(
            shuffled_development, shuffled_assessment, c2_features
        )
        null_metrics = _weighted_metric_triplet(
            shuffled_assessment,
            null_probabilities,
            null_classes,
            "sequential_row_weight",
        )
        recalculated = {
            "multiclass_log_loss_improvement": c1_metrics["multiclass_log_loss"]
            - null_metrics["multiclass_log_loss"],
            "multiclass_brier_improvement": c1_metrics["multiclass_brier"]
            - null_metrics["multiclass_brier"],
            "top_two_accuracy_change": null_metrics["top_two_accuracy"]
            - c1_metrics["top_two_accuracy"],
        }
        archived_draw = draw_null.loc[draw_null["draw"].eq(draw)].set_index("metric")
        null_checks.extend(
            abs(value - float(archived_draw.loc[metric, "null_increment"])) <= 1e-12
            for metric, value in recalculated.items()
        )
        null_checks.append(
            shuffled_development.loc[:, list(c1_features)].equals(
                development.loc[:, list(c1_features)]
            )
            and shuffled_assessment.loc[:, list(c1_features)].equals(
                assessment.loc[:, list(c1_features)]
            )
            and shuffled_development["next_registered_route"].equals(
                development["next_registered_route"]
            )
            and shuffled_assessment["next_registered_route"].equals(
                assessment["next_registered_route"]
            )
        )
    for row in hidden_null.loc[hidden_null["record_type"].eq("summary")].itertuples(index=False):
        group = draw_null.loc[draw_null["metric"].eq(row.metric)]
        null_checks.append(
            int((float(row.real_increment) > group["null_increment"].astype(float)).sum())
            == int(row.null_draws_exceeded)
        )
    checks["hidden_history_null"] = all(null_checks)
    assessment_panel = panel.loc[panel["period"].eq("assessment")]
    candidate_stock_share = assessment_candidates["symbol"].value_counts(normalize=True)
    weighted_stock_share = assessment_panel.groupby("symbol")["sequential_row_weight"].sum()
    weighted_stock_share = weighted_stock_share / weighted_stock_share.sum()
    weighted_target_share = assessment_panel.groupby("next_registered_route")[
        "sequential_row_weight"
    ].sum()
    weighted_target_share = weighted_target_share / weighted_target_share.sum()
    model_support_checks = {
        "candidate_rows": len(assessment_candidates) >= 850,
        "candidate_sessions": assessment_candidates["session"].nunique() >= 140,
        "candidate_stocks": assessment_candidates["symbol"].nunique() >= 15,
        "registered_candidates": assessment_candidates["original_registered_completion"]
        .astype(int)
        .sum()
        >= 180,
        "sequential_rows": len(assessment_panel) >= 4000,
        "rows_after_hidden": assessment_panel["any_hidden_event_since_opening"].astype(bool).sum()
        >= 150,
        "candidate_stock_concentration": float(candidate_stock_share.max()) <= 0.10,
        "weighted_stock_concentration": float(weighted_stock_share.max()) <= 0.10,
        "target_concentration": float(weighted_target_share.max()) <= 0.85,
    }
    transition_support_checks = []
    for lookback in (6, 12):
        values = cast(Mapping[str, Any], lookback_support[f"assessment_{lookback}"])
        transition_support_checks.extend(
            [
                int(values["eligible_events"]) >= 500,
                int(values["sessions"]) >= 120,
                int(values["stocks"]) >= 15,
                int(values["months"]) >= 6,
                float(values["minimum_matched_null_coverage"]) >= 0.90,
            ]
        )
    independent_blocker: str | None = None
    if not all(transition_support_checks):
        independent_blocker = "blocked_transition_census_support_failure"
    elif not all(model_support_checks.values()):
        independent_blocker = "blocked_sequential_model_support_failure"
    elif not all(bool(configurations[name]["converged"]) for name in ("C0", "C1", "C2")):
        independent_blocker = "blocked_model_convergence_failure"
    if independent_blocker is not None:
        expected_statuses = {
            "target_a_precursor_status": "insufficient_support",
            "target_b_precursor_status": "insufficient_support",
            "target_c_recurrence_status": "insufficient_support",
            "hidden_2_3_2_diversion_status": "insufficient_support",
            "registered_history_increment_status": "insufficient_support",
            "hidden_history_increment_status": "insufficient_support",
        }
    else:
        expected_statuses = {
            key: str(decision["decision_gates"]["hypotheses"][hypothesis]["status"])
            for key, hypothesis in (
                ("target_a_precursor_status", "H1"),
                ("target_b_precursor_status", "H2"),
                ("target_c_recurrence_status", "H3"),
                ("hidden_2_3_2_diversion_status", "H4"),
            )
        }
        expected_statuses["registered_history_increment_status"] = str(
            decision["registered_history_increment_status"]
        )
        expected_statuses["hidden_history_increment_status"] = str(
            decision["hidden_history_increment_status"]
        )
    reconstructed_decision = choose_primary_decision(
        blocker=independent_blocker,
        target_a_status=expected_statuses["target_a_precursor_status"],
        target_b_status=expected_statuses["target_b_precursor_status"],
        target_c_status=expected_statuses["target_c_recurrence_status"],
        diversion_status=expected_statuses["hidden_2_3_2_diversion_status"],
        hidden_increment_status=expected_statuses["hidden_history_increment_status"],
    )
    decision_inputs = cast(Mapping[str, Any], decision["decision_inputs"])
    checks["decision_logic"] = (
        reconstructed_decision == decision["primary_decision"]
        and decision_inputs["blocker"] == independent_blocker
        and all(decision[key] == value for key, value in expected_statuses.items())
        and all(decision_inputs[key] == value for key, value in expected_statuses.items())
    )
    checks["determinism_result"] = bool(determinism["passed"])
    details.update(
        {
            "maximum_manual_probability_difference": maximum_probability_difference,
            "maximum_manual_metric_difference": maximum_metric_difference,
            "maximum_contrast_difference": float(max(contrast_differences, default=0.0)),
            "maximum_bh_q_difference": q_difference,
            "maximum_independent_coefficient_difference": maximum_coefficient_difference,
            "independently_calculated_blocker": independent_blocker,
            "manual_probability_rows": int(len(manual_rows)),
            "checks_passed": int(sum(checks.values())),
            "checks_total": int(len(checks)),
        }
    )
    return {
        **SAFETY_FLAGS,
        "passed": all(checks.values()),
        "checks": checks,
        **details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider-root", type=Path, default=DEFAULT_PROVIDER_ROOT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output = arguments.output.expanduser().resolve()
    audit = audit_artifacts(output, provider_root=arguments.provider_root.expanduser().resolve())
    (output / "lightweight_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True, default=_json_default))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
