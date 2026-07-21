#!/usr/bin/env python3
"""Independent audit for Regime x Loop Prefix x Behaviour Quick Screen V0.

This file deliberately does not import the runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocker_research.loop_dictionary_v2 import LoopDictionary, decompose_closed_path
from stocker_research.loop_events_v2 import PrimaryOutcomeLabel
from stocker_research.loop_prefix_automaton_v2 import (
    FirstNextLoopEventEngine,
    LoopEventAutomaton,
)

SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "structural_prediction_only": True,
    "pre_completion_context_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "economic_outcomes_opened": False,
    "production_runtime_modified": False,
}

BEHAVIOURAL_DIMENSIONS = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "pressure_magnitude",
    "exhaustion_magnitude",
    "signed_exhaustion",
    "independence",
    "signed_independence",
)
INTERACTIONS = (
    "orientation_pressure_alignment",
    "prefix_conviction",
    "transition_arousal",
    "repeat_tension",
    "next_leg_exhaustion_alignment",
)
TARGET = "candidate_completes_first_within_6_bars"
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
EXPECTED_DICTIONARY_HASH = "497142c8d0ab880e59385da123d9eb2189469e9e3a4a631e0f63eb6fc77030d3"
EXPECTED_BEHAVIOURAL_LEDGER_HASH = (
    "cd5e7cee343952638bb17d4e4ea5d58918cdd1b13477275c312ea21e37d2dee0"
)

BEHAVIOURAL_RELATIVE = Path(
    "research/observable-behavioural-state/"
    "20260721-behavioural-state-dimensions-screen-v0/artifacts/primary"
)
OPENING_RELATIVE = Path(
    "research/opening-regime-path/"
    "20260720-opening-regime-path-direction-screen-v0/artifacts/primary/"
    "opening_state_path_ledger.parquet"
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(dict(value)), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _safety_check(payload: Mapping[str, Any]) -> bool:
    return all(payload.get(key) == value for key, value in SAFETY_FLAGS.items())


def _dictionary(repo_root: Path) -> tuple[LoopDictionary, pd.DataFrame]:
    path = (
        repo_root / "research/slrno-v2/20260714-regime-loop-handoff/work/artifacts/"
        "20260718-loop-event-semantics-v2/primary/semantic_loop_dictionary_v2.csv"
    )
    table = pd.read_csv(path)
    if (
        len(table) != 20
        or not table["dictionary_hash"].astype(str).eq(EXPECTED_DICTIONARY_HASH).all()
    ):
        raise AssertionError("semantic dictionary identity differs")
    definitions = []
    for row in table.sort_values("semantic_loop_id", kind="mergesort").itertuples(index=False):
        path_values = tuple(int(value) for value in str(row.canonical_path).split("->"))
        definition = decompose_closed_path(path_values)
        if definition.semantic_loop_id != str(row.semantic_loop_id):
            raise AssertionError("semantic dictionary path does not reconstruct its ID")
        definitions.append(definition)
    return (
        LoopDictionary.from_definitions(definitions, version="semantic_loop_dictionary_v2"),
        table,
    )


def _behavioural_audit(
    candidates: pd.DataFrame,
    materialized_repo: Path,
) -> dict[str, Any]:
    root = materialized_repo / BEHAVIOURAL_RELATIVE
    ledger_path = root / "behavioural_dimension_ledger.parquet"
    compact_path = root / "compact_decision_panel.parquet"
    if sha256_file(ledger_path) != EXPECTED_BEHAVIOURAL_LEDGER_HASH:
        raise AssertionError("frozen behavioural ledger hash differs")
    base_key = ["symbol", "session", "decision_ordinal", "slate_id"]
    ledger = pd.read_parquet(ledger_path, columns=[*base_key, *BEHAVIOURAL_DIMENSIONS])
    compact = pd.read_parquet(
        compact_path,
        columns=[*base_key, "feature_available_timestamp_utc", *BEHAVIOURAL_DIMENSIONS],
    )
    frozen = ledger.merge(
        compact[[*base_key, "feature_available_timestamp_utc"]],
        on=base_key,
        how="inner",
        validate="one_to_one",
    ).rename(columns={"feature_available_timestamp_utc": "frozen_decision_timestamp"})
    decisions = candidates.drop_duplicates("decision_id").loc[
        :, ["decision_id", *base_key, "decision_timestamp", *BEHAVIOURAL_DIMENSIONS]
    ]
    compared = decisions.merge(
        frozen,
        on=base_key,
        how="left",
        validate="one_to_one",
        suffixes=(
            "_candidate",
            "_frozen",
        ),
    )
    if compared["frozen_decision_timestamp"].isna().any():
        raise AssertionError("candidate decision is absent from the frozen behavioural ledger")
    timestamp_error = pd.to_datetime(compared["decision_timestamp"], utc=True) != pd.to_datetime(
        compared["frozen_decision_timestamp"], utc=True
    )
    maximum_error = 0.0
    for dimension in BEHAVIOURAL_DIMENSIONS:
        maximum_error = max(
            maximum_error,
            float(
                np.max(
                    np.abs(
                        compared[f"{dimension}_candidate"].to_numpy(dtype=float)
                        - compared[f"{dimension}_frozen"].to_numpy(dtype=float)
                    ),
                    initial=0.0,
                )
            ),
        )
    if timestamp_error.any() or maximum_error > 1e-12:
        raise AssertionError("candidate behavioural values or decision timestamps differ")
    return {
        "decisions_checked": len(compared),
        "maximum_absolute_error": maximum_error,
        "timestamp_mismatches": int(timestamp_error.sum()),
        "passed": True,
    }


def _parse_csv_ints(value: object) -> tuple[int, ...]:
    return tuple(int(item) for item in str(value).split(",") if item != "")


def _parse_orientation(value: object) -> tuple[int, ...]:
    marker = "__o_"
    text = str(value)
    if marker not in text:
        raise AssertionError("candidate orientation lacks an oriented route")
    return tuple(int(item) for item in text.split(marker, maxsplit=1)[1].split("-"))


def _prefix_membership_audit(
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    materialized_repo: Path,
    dictionary: LoopDictionary,
    dictionary_table: pd.DataFrame,
) -> dict[str, Any]:
    behavioural = pd.read_parquet(
        materialized_repo / BEHAVIOURAL_RELATIVE / "behavioural_dimension_ledger.parquet",
        columns=["symbol", "session", "decision_ordinal"],
    )
    opening = pd.read_parquet(
        materialized_repo / OPENING_RELATIVE,
        columns=[
            "symbol",
            "session",
            "decision_ordinal",
            "opening_state_path",
            "current_state",
            "current_posterior",
        ],
    )
    keys = ["symbol", "session", "decision_ordinal"]
    decisions = behavioural.merge(
        opening,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not decisions["_merge"].eq("both").all():
        raise AssertionError("frozen behavioural decision is absent from the V2 opening ledger")

    automaton = LoopEventAutomaton(dictionary, allowed_states=frozenset(range(8)))
    expected_by_decision: dict[str, set[tuple[str, str, int, int]]] = {}
    no_active_decisions = 0
    timestamp_origin = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    for row in decisions.sort_values(keys, kind="mergesort").itertuples(index=False):
        automaton.reset_session()
        for bar_ordinal, state in enumerate(_parse_csv_ints(row.opening_state_path)):
            event_timestamp = timestamp_origin + timedelta(minutes=5 * bar_ordinal)
            automaton.feed(
                state,
                event_timestamp=event_timestamp,
                available_timestamp=event_timestamp + timedelta(minutes=5),
                bar_ordinal=bar_ordinal,
            )
        expected = {
            (
                prefix.semantic_loop_id,
                prefix.orientation_id,
                int(prefix.start_event_index),
                int(prefix.progress_states - 1),
            )
            for prefix in automaton.active_prefixes()
            if prefix.progress_states >= 2
        }
        decision_id = f"{row.symbol}|{row.session}|{int(row.decision_ordinal):02d}"
        if expected:
            expected_by_decision[decision_id] = expected
        else:
            no_active_decisions += 1

    observed_by_decision: dict[str, set[tuple[str, str, int, int]]] = {}
    duplicate_memberships = 0
    for decision_id, group in outcomes.groupby("decision_id", sort=True):
        observed = {
            (
                str(row.semantic_loop_id),
                str(row.candidate_orientation),
                int(row.prefix_start_event_index),
                int(row.prefix_matched_length),
            )
            for row in group.itertuples(index=False)
        }
        duplicate_memberships += len(group) - len(observed)
        observed_by_decision[str(decision_id)] = observed
    membership_mismatches = int(set(expected_by_decision) != set(observed_by_decision))
    for decision_id in set(expected_by_decision).intersection(observed_by_decision):
        membership_mismatches += int(
            expected_by_decision[decision_id] != observed_by_decision[decision_id]
        )

    merged = candidates.merge(opening, on=keys, how="left", validate="many_to_one")
    if merged["opening_state_path"].isna().any():
        raise AssertionError("candidate is absent from the frozen V2 opening ledger")
    dictionary_ids = set(dictionary_table["semantic_loop_id"].astype(str))
    failures = 0
    posterior_error = 0.0
    for row in merged.itertuples(index=False):
        hard_bars = _parse_csv_ints(row.opening_state_path)
        compressed = tuple(
            value
            for index, value in enumerate(hard_bars)
            if index == 0 or value != hard_bars[index - 1]
        )
        prefix = tuple(int(value) for value in str(row.prefix_path).split("->"))
        orientation = _parse_orientation(row.candidate_orientation)
        progress = int(row.prefix_matched_length) + 1
        valid = (
            str(row.semantic_loop_id) in dictionary_ids
            and len(prefix) >= 2
            and progress == len(prefix)
            and progress < len(orientation)
            and orientation[:progress] == prefix
            and compressed[-len(prefix) :] == prefix
            and int(row.next_required_state) == orientation[progress]
            and int(row.candidate_path_length) == len(orientation) - 1
            and 0.0 < float(row.prefix_completion_fraction) < 1.0
        )
        failures += int(not valid)
        expected = np.asarray([float(value) for value in str(row.current_posterior).split(",")])
        actual = np.asarray([getattr(row, f"posterior_state_{state}") for state in range(8)])
        posterior_error = max(posterior_error, float(np.max(np.abs(expected - actual))))
    expected_rows = sum(len(value) for value in expected_by_decision.values())
    observed_rows = sum(len(value) for value in observed_by_decision.values())
    if (
        failures
        or posterior_error > 1e-8
        or duplicate_memberships
        or membership_mismatches
        or expected_rows != observed_rows
    ):
        raise AssertionError("V2 active-prefix membership or posterior differs")
    return {
        "candidate_rows_checked": len(merged),
        "frozen_behavioural_decisions_checked": len(decisions),
        "expected_active_prefix_decisions": len(expected_by_decision),
        "observed_active_prefix_decisions": len(observed_by_decision),
        "expected_candidate_rows_before_tie_exclusion": expected_rows,
        "observed_candidate_rows_before_tie_exclusion": observed_rows,
        "independently_reconstructed_no_active_decisions": no_active_decisions,
        "complete_membership_mismatches": membership_mismatches,
        "duplicate_observed_memberships": duplicate_memberships,
        "membership_failures": failures,
        "maximum_current_posterior_error": posterior_error,
        "semantic_loop_support": int(merged["semantic_loop_id"].nunique()),
        "passed": True,
    }


def _timestamp_and_boundary_audit(candidates: pd.DataFrame) -> dict[str, Any]:
    source = pd.to_datetime(candidates["decision_source_timestamp"], utc=True)
    decision = pd.to_datetime(candidates["decision_timestamp"], utc=True)
    available = pd.to_datetime(candidates["predictor_max_available_timestamp"], utc=True)
    local = decision.dt.tz_convert("America/New_York")
    expected_clock = np.where(candidates["decision_ordinal"].eq(6), "10:00", "10:30")
    failures = int((decision != source + pd.Timedelta(minutes=5)).sum())
    failures += int((available > decision).sum())
    failures += int((local.dt.strftime("%H:%M").to_numpy() != expected_clock).sum())
    protected = int(decision.ge(PROTECTED_START).sum())
    if failures or protected:
        raise AssertionError("decision chronology or protected boundary differs")
    return {
        "rows_checked": len(candidates),
        "chronology_failures": failures,
        "protected_rows": protected,
        "minimum_decision_timestamp": str(decision.min()),
        "maximum_decision_timestamp": str(decision.max()),
        "passed": True,
    }


def _first_event_and_target_audit(
    population: pd.DataFrame,
    outcomes: pd.DataFrame,
    dictionary: LoopDictionary,
) -> dict[str, Any]:
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    failures = 0
    ties = 0
    checked = 0
    eligible_outcomes = outcomes.loc[outcomes["primary_scoring_eligible"].astype(bool)].copy()
    eligible = population.merge(
        eligible_outcomes[["candidate_id", TARGET]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    if eligible[TARGET].isna().any():
        raise AssertionError("primary candidate target is missing")
    target_by_id = eligible.set_index("candidate_id")[TARGET].astype(int)
    for decision_id, group in outcomes.groupby("decision_id", sort=True):
        first = group.iloc[0]
        states = _parse_csv_ints(first["state_events_through_horizon"])
        bars = _parse_csv_ints(first["state_event_bars_through_horizon"])
        starts = [
            datetime(2025, 1, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=5 * bar) for bar in bars
        ]
        trace = engine.scan_state_events(
            states,
            bar_ordinals=bars,
            event_timestamps=starts,
            available_timestamps=[value + timedelta(minutes=5) for value in starts],
        )
        origin = 5 if int(first["decision_ordinal"]) == 6 else 11
        decision_event_index = int(first["decision_event_index"])
        result = engine.outcome_for_decision(
            trace,
            decision_id=str(decision_id),
            decision_event_index=decision_event_index,
            decision_bar_ordinal=origin,
            horizon_bars=6,
            session_end_bar_ordinal=77,
        )
        stored_label = str(first["primary_structural_outcome"])
        recomputed_label = str(result.primary_label)
        if recomputed_label == str(PrimaryOutcomeLabel.SESSION_END) and origin + 6 < 77:
            recomputed_label = str(PrimaryOutcomeLabel.NO_REGISTERED_LOOP_WITHIN_HORIZON)
        labels_match = stored_label == recomputed_label
        if not labels_match:
            failures += 1
        tied = recomputed_label == str(PrimaryOutcomeLabel.TIED_REGISTERED_COMPLETION)
        ties += int(tied)
        if tied:
            if group["primary_scoring_eligible"].astype(bool).any():
                failures += 1
        elif bool(group["primary_scoring_eligible"].astype(bool).all()):
            expected_positive: set[tuple[str, str]] = set()
            if len(result.earliest_registered_events) == 1:
                event = result.earliest_registered_events[0]
                expected_positive.add((event.semantic_loop_id, event.orientation_id))
            for row in group.itertuples(index=False):
                expected = (
                    str(row.semantic_loop_id),
                    str(row.candidate_orientation),
                ) in expected_positive
                actual = bool(target_by_id.loc[str(row.candidate_id)])
                failures += int(expected != actual)
        checked += 1
    if failures:
        raise AssertionError("six-bar first-event target reconstruction differs")
    return {
        "decisions_checked": checked,
        "tie_decisions_reconstructed": ties,
        "target_failures": failures,
        "horizon_completed_bars": 6,
        "passed": True,
    }


def _weight_and_interaction_audit(
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    interaction_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    eligible_targets = outcomes.loc[
        outcomes["primary_scoring_eligible"].astype(bool),
        [
            "candidate_id",
            TARGET,
        ],
    ]
    frame = candidates.merge(eligible_targets, on="candidate_id", validate="one_to_one")
    candidate_count = frame.groupby("decision_id", sort=True)["decision_id"].transform("size")
    slate_count = frame.groupby("slate_id", sort=True)["decision_id"].transform("nunique")
    expected_candidate = 1.0 / candidate_count.to_numpy(dtype=float)
    expected_slate = 1.0 / slate_count.to_numpy(dtype=float)
    weight_error = float(
        np.max(
            np.abs(frame["candidate_weight"].to_numpy(dtype=float) - expected_candidate),
            initial=0.0,
        )
    )
    weight_error = max(
        weight_error,
        float(
            np.max(
                np.abs(frame["slate_weight"].to_numpy(dtype=float) - expected_slate),
                initial=0.0,
            )
        ),
        float(
            np.max(
                np.abs(
                    frame["row_weight"].to_numpy(dtype=float) - expected_candidate * expected_slate
                ),
                initial=0.0,
            )
        ),
    )
    raw = pd.DataFrame(index=frame.index)
    raw["orientation_pressure_alignment"] = (
        frame["candidate_orientation_sign"] * frame["signed_pressure"]
    )
    raw["prefix_conviction"] = frame["prefix_completion_fraction"] * frame["conviction"]
    raw["transition_arousal"] = frame["current_transition_probability"] * frame["arousal"]
    raw["repeat_tension"] = frame["repeat_depth"] * frame["tension"]
    raw["next_leg_exhaustion_alignment"] = (
        frame["probability_of_next_required_state"]
        * frame["candidate_orientation_sign"]
        * frame["signed_exhaustion"]
    )
    interaction_error = 0.0
    bounds = interaction_manifest["clipping_bounds"]
    for name in INTERACTIONS:
        expected = raw[name].clip(float(bounds[name]["p01"]), float(bounds[name]["p99"]))
        interaction_error = max(
            interaction_error,
            float(
                np.max(
                    np.abs(expected.to_numpy() - frame[name].to_numpy(dtype=float)),
                    initial=0.0,
                )
            ),
        )
    if weight_error > 1e-12 or interaction_error > 1e-12:
        raise AssertionError("candidate weighting or fixed interactions differ")
    return {
        "candidate_rows_checked": len(frame),
        "maximum_weight_error": weight_error,
        "maximum_interaction_error": interaction_error,
        "candidate_weighting_passed": True,
        "five_interactions_passed": True,
        "passed": True,
    }


def _support_audit(
    population: pd.DataFrame,
    outcomes: pd.DataFrame,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    targets = outcomes.loc[
        outcomes["primary_scoring_eligible"].astype(bool),
        [
            "candidate_id",
            TARGET,
        ],
    ]
    frame = population.merge(targets, on="candidate_id", validate="one_to_one")
    assessment = frame.loc[frame["year"].eq(2025)]
    decision_sizes = assessment.groupby("decision_id", sort=True).size()
    positives = assessment.loc[assessment[TARGET].eq(1)]
    stock_max = float(assessment["symbol"].value_counts().max() / len(assessment))
    loop_max = float(positives["semantic_loop_id"].value_counts().max() / len(positives))
    actual = {
        "candidate_rows": len(assessment),
        "active_prefix_decisions": int(assessment["decision_id"].nunique()),
        "sessions": int(assessment["session"].nunique()),
        "stocks": int(assessment["symbol"].nunique()),
        "positive_candidate_completions": int(assessment[TARGET].sum()),
        "represented_months": int(assessment["year_month"].nunique()),
        "semantic_candidate_loops": int(assessment["semantic_loop_id"].nunique()),
        "multi_candidate_decisions": int(decision_sizes.ge(2).sum()),
        "maximum_stock_candidate_row_share": stock_max,
        "maximum_loop_positive_completion_share": loop_max,
    }
    stored = decision["assessment_support"]
    failures = 0
    for key, value in actual.items():
        failures += int(
            not math.isclose(float(value), float(stored[key]), rel_tol=0.0, abs_tol=1e-15)
        )
    independently_failed = []
    if actual["active_prefix_decisions"] < 1_000:
        independently_failed.append("active_prefix_decisions")
    if actual["candidate_rows"] < 3_000:
        independently_failed.append("candidate_rows")
    if actual["sessions"] < 100:
        independently_failed.append("sessions")
    if actual["stocks"] < 15:
        independently_failed.append("stocks")
    if actual["positive_candidate_completions"] < 300:
        independently_failed.append("positive_candidate_completions")
    if actual["represented_months"] < 6:
        independently_failed.append("represented_months")
    if actual["semantic_candidate_loops"] < 10:
        independently_failed.append("semantic_candidate_loops")
    if actual["multi_candidate_decisions"] < 300:
        independently_failed.append("multi_candidate_decisions")
    if stock_max > 0.125:
        independently_failed.append("stock_concentration")
    if loop_max > 0.30:
        independently_failed.append("loop_concentration")
    expected_failed = list(stored["failed_support_gates"])
    if (
        failures
        or independently_failed != expected_failed
        or decision["decision"] != "blocked_insufficient_active_prefix_support"
    ):
        raise AssertionError("support gate or blocked decision differs")
    return {
        "assessment": actual,
        "independently_failed_support_gates": independently_failed,
        "stored_value_mismatches": failures,
        "decision_logic_passed": True,
        "passed": True,
    }


def audit(
    artifacts: Path,
    *,
    materialized_repo: Path,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    required_json = (
        "contract.json",
        "source_manifest.json",
        "input_artifact_hashes.json",
        "protected_boundary_audit.json",
        "behavioural_ledger_reconstruction.json",
        "v2_prefix_population_reconstruction.json",
        "candidate_feature_manifest.json",
        "interaction_manifest.json",
        "model_configurations.json",
        "model_coefficients.json",
        "decision.json",
    )
    payloads = {name: _load_json(artifacts / name) for name in required_json}
    safety = {name: _safety_check(payload) for name, payload in payloads.items()}
    if not all(safety.values()):
        raise AssertionError("mandatory safety flags differ")
    population = pd.read_parquet(artifacts / "candidate_population.parquet")
    outcomes = pd.read_parquet(artifacts / "candidate_outcomes.parquet")
    predictions = pd.read_parquet(artifacts / "assessment_predictions.parquet")
    dictionary, dictionary_table = _dictionary(repo_root)
    behavioural = _behavioural_audit(population, materialized_repo)
    prefix = _prefix_membership_audit(
        population,
        outcomes,
        materialized_repo,
        dictionary,
        dictionary_table,
    )
    chronology = _timestamp_and_boundary_audit(population)
    target = _first_event_and_target_audit(population, outcomes, dictionary)
    weighting = _weight_and_interaction_audit(
        population, outcomes, payloads["interaction_manifest.json"]
    )
    support = _support_audit(population, outcomes, payloads["decision.json"])
    protected = payloads["protected_boundary_audit.json"]
    protected_passed = bool(
        protected["passed"]
        and int(protected["protected_rows_materialised"]) == 0
        and int(protected["candidate_rows_on_or_after_protected_start"]) == 0
    )
    feature_manifest = payloads["candidate_feature_manifest.json"]
    feature_passed = bool(
        feature_manifest["behavioural_features"] == list(BEHAVIOURAL_DIMENSIONS)
        and feature_manifest["interaction_features"] == list(INTERACTIONS)
        and feature_manifest["numeric_discovery_rank_used"] is False
        and feature_manifest["assessment_completion_rates_used"] is False
    )
    model_config = payloads["model_configurations.json"]
    coefficients = payloads["model_coefficients.json"]
    fail_closed_models = bool(
        model_config["status"] == "not_fit_due_insufficient_active_prefix_support"
        and int(model_config["primary_model_count_fitted"]) == 0
        and coefficients["models"] == {}
        and predictions["status"].eq("not_scored_due_insufficient_active_prefix_support").all()
    )
    metric_files = (
        "candidate_metrics.csv",
        "decision_ranking_metrics.csv",
        "monthly_metrics.csv",
        "checkpoint_metrics.csv",
        "candidate_family_metrics.csv",
        "prefix_maturity_metrics.csv",
        "bootstrap_metrics.csv",
        "null_metrics.csv",
    )
    metrics_fail_closed = all(
        pd.read_csv(artifacts / name)["status"]
        .eq("not_run_due_insufficient_active_prefix_support")
        .all()
        for name in metric_files
    )
    checks = {
        "safety_flags": all(safety.values()),
        "dates_and_protected_boundary": protected_passed and chronology["passed"],
        "frozen_behavioural_values": behavioural["passed"],
        "v2_active_prefix_membership": prefix["passed"],
        "candidate_semantic_ids": prefix["semantic_loop_support"] == 20,
        "decision_timestamps": chronology["passed"],
        "six_bar_horizon": target["horizon_completed_bars"] == 6,
        "first_event_target": target["passed"],
        "tie_exclusions": target["passed"],
        "candidate_weighting": weighting["candidate_weighting_passed"],
        "structural_features": feature_passed,
        "ten_behavioural_dimensions": behavioural["passed"] and feature_passed,
        "five_interactions": weighting["five_interactions_passed"],
        "development_only_fitting": fail_closed_models,
        "assessment_only_scoring": fail_closed_models,
        "model_coefficients": fail_closed_models,
        "manual_probability_reconstruction": fail_closed_models,
        "brier_log_loss_auc": metrics_fail_closed,
        "session_block_bootstrap": metrics_fail_closed,
        "within_slate_behavioural_null": metrics_fail_closed,
        "decision_logic": support["decision_logic_passed"],
    }
    passed = all(checks.values())
    result = {
        **SAFETY_FLAGS,
        "auditor_imported_runner": False,
        "audit_scope": "fail_closed_support_blocker",
        "checks": checks,
        "behavioural_ledger": behavioural,
        "v2_prefix_population": prefix,
        "chronology": chronology,
        "first_event_target": target,
        "weighting_and_interactions": weighting,
        "support_and_decision": support,
        "not_applicable_after_support_blocker": [
            "model_fit",
            "model_coefficients",
            "manual_probability_reconstruction",
            "candidate_metrics",
            "bootstrap",
            "behavioural_null",
        ],
        "passed": passed,
    }
    write_json(artifacts / "independent_audit.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--materialized-predecessor-repo", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = audit(
            args.artifacts,
            materialized_repo=args.materialized_predecessor_repo,
        )
    except Exception as error:
        failure = {**SAFETY_FLAGS, "passed": False, "error": str(error)}
        write_json(args.artifacts / "independent_audit.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
