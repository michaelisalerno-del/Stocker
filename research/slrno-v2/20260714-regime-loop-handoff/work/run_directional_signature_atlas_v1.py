#!/usr/bin/env python3
"""Run the research-only Directional Signature Atlas V1 experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd

from stocker_research.directional_signature_atlas.analysis import (
    evaluate_candidate_census,
    evaluate_neutral_veto_census,
    freeze_discovery_library,
    score_frozen_library,
    signature_breakdowns,
    validate_discovery_library,
    validate_neutral_veto_library,
)
from stocker_research.directional_signature_atlas.contract import (
    contract_sha256,
    load_contract,
)
from stocker_research.directional_signature_atlas.historical import (
    build_causal_anchor_panel,
    build_feature_ledger,
    build_outcome_ledgers,
    feature_family_map,
    load_frozen_movement_bundle,
)
from stocker_research.directional_signature_atlas.io import (
    canonical_json_bytes,
    sha256_file,
    write_deterministic_csv,
    write_deterministic_json,
    write_deterministic_parquet,
)
from stocker_research.directional_signature_atlas.models import (
    CLASSES,
    apply_atlas_controller,
    baseline_predictions,
    prediction_metrics,
)
from stocker_research.directional_signature_atlas.prospective import ProspectiveLedger
from stocker_research.directional_signature_atlas.robustness import (
    null_test_results,
    stress_signature_library,
)
from stocker_research.directional_signature_atlas.signatures import (
    SearchCaps,
    SupportRules,
    extract_shallow_tree_candidates,
    generate_bounded_candidates,
)
from stocker_research.directional_signature_atlas.track_b import (
    construct_relative_outcomes,
    relative_baseline_economic_metrics,
    relative_strength_baseline,
)

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260717-directional-signature-atlas-v1.json"
FEATURE_SCHEMA_PATH = HERE / "contracts/20260717-directional-signature-atlas-v1-feature-schema.json"
AUDITOR_PATH = HERE / "audit_directional_signature_atlas_v1.py"
PRIOR_CONTRACT_PATH = HERE / "contracts/20260714-long-short-neutral-detector-v1.json"
PRIOR_RUNNER_PATH = HERE / "run_long_short_neutral_detector_v1.py"
CORE_PATH = HERE / "frozen_loop_movement_shadow_core.py"
BUNDLE_ROOT = HERE / "shadow_validation/frozen_loop_movement_shadow_v1/frozen_bundle"
DEFAULT_OUTPUT_ROOT = HERE / "artifacts/20260717-directional-signature-atlas-v1/primary"
AS_OF = pd.Timestamp("2026-06-29T19:55:00Z")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HERE,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _snapshot_sources(
    prior_runner: ModuleType,
    prior_contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], str]:
    sources: dict[str, Path] = dict(prior_runner.source_paths(prior_contract))
    provider_root = Path(prior_contract["data"]["provider_root"])
    sources["provider_VTI"] = provider_root / "symbol=VTI.US/timeframe=5m/data.parquet"
    sources.update(
        {
            "state_preprocessing": BUNDLE_ROOT
            / "artifacts/state/frozen_emission_preprocessing.csv",
            "state_parameters": BUNDLE_ROOT / "artifacts/state/frozen_semimarkov_parameters.npz",
            "fixed_cycles": BUNDLE_ROOT / "artifacts/state/fixed_cycle_shuffled_nulls.csv",
            "path_parameters": BUNDLE_ROOT / "artifacts/path/model_parameters.npz",
            "movement_manifest": BUNDLE_ROOT / "artifacts/price/feature_manifest.json",
            "movement_parameters": BUNDLE_ROOT / "artifacts/price/outcome_model_parameters.npz",
            "atlas_contract": CONTRACT_PATH,
            "atlas_feature_schema": FEATURE_SCHEMA_PATH,
            "atlas_runner": Path(__file__).resolve(),
            "atlas_auditor": AUDITOR_PATH,
            "atlas_test_core": HERE.parents[3] / "tests/test_directional_signature_atlas_core.py",
            "atlas_test_discovery": HERE.parents[3]
            / "tests/test_directional_signature_atlas_discovery.py",
            "atlas_test_prospective": HERE.parents[3]
            / "tests/test_directional_signature_atlas_prospective.py",
        }
    )
    payload = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in sorted(sources.items())
    }
    identity_payload = {name: value["sha256"] for name, value in payload.items()}
    snapshot_hash = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
    return payload, snapshot_hash


def _feature_schema() -> dict[str, Any]:
    payload = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("feature schema must be an object")
    return payload


def _population_contract() -> dict[str, Any]:
    payload = json.loads(PRIOR_CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("prior population contract must be an object")
    return payload


def _write_build_artifacts(
    output_root: Path,
    *,
    contract: dict[str, Any],
    feature_schema: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    data_snapshot_hash: str,
    run_id: str,
    events: pd.DataFrame,
    population_counts: pd.DataFrame,
    population_coverage: pd.DataFrame,
    anchors: pd.DataFrame,
    anchor_audit: dict[str, Any],
    feature_ledger: pd.DataFrame,
    outcomes: pd.DataFrame,
    delayed_outcomes: pd.DataFrame,
    first_touch: pd.DataFrame,
    outcome_coverage: pd.DataFrame,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONTRACT_PATH, output_root / "frozen_experiment_contract.json")
    shutil.copyfile(FEATURE_SCHEMA_PATH, output_root / "feature_schema.json")
    write_prior_experiment_coverage(output_root)
    write_deterministic_json(
        {
            "contract_id": contract["contract_id"],
            "contract_sha256": contract_sha256(CONTRACT_PATH),
            "feature_schema_sha256": sha256_file(FEATURE_SCHEMA_PATH),
            "data_snapshot_sha256": data_snapshot_hash,
            "sources": sources,
        },
        output_root / "source_identities.json",
    )
    write_deterministic_csv(
        population_counts,
        output_root / "population_counts.csv",
        sort_by=["period", "symbol_norm", "session_date"],
    )
    write_deterministic_csv(
        population_coverage,
        output_root / "population_data_coverage.csv",
        sort_by=["period", "symbol_norm"],
    )
    write_deterministic_csv(
        outcome_coverage,
        output_root / "outcome_data_coverage.csv",
        sort_by=["period", "symbol_norm"],
    )
    write_deterministic_parquet(
        feature_ledger,
        output_root / "outcome_free_feature_ledger.parquet",
        sort_by=["opportunity_id"],
    )
    feature_hash = sha256_file(output_root / "outcome_free_feature_ledger.parquet")
    write_deterministic_json(
        {
            "run_id": run_id,
            "contract_sha256": contract_sha256(CONTRACT_PATH),
            "feature_schema_sha256": sha256_file(FEATURE_SCHEMA_PATH),
            "data_snapshot_sha256": data_snapshot_hash,
            "feature_ledger_sha256": feature_hash,
            "feature_rows": len(feature_ledger),
            "outcomes_joined_at_seal": false_value(),
        },
        output_root / "pre_outcome_feature_manifest.json",
    )
    write_deterministic_parquet(
        outcomes,
        output_root / "primary_economic_outcome_ledger.parquet",
        sort_by=["opportunity_id"],
    )
    write_deterministic_parquet(
        delayed_outcomes,
        output_root / "one_bar_delay_outcome_ledger.parquet",
        sort_by=["opportunity_id"],
    )
    write_deterministic_parquet(
        first_touch,
        output_root / "secondary_first_touch_outcome_ledger.parquet",
        sort_by=["opportunity_id"],
    )
    movement_columns = [
        "run_id",
        "opportunity_id",
        "period",
        "symbol",
        "session",
        "decision_clock",
        "decision_timestamp",
        "state_run_entry_at_decision",
        "predicted_future_range_bps",
        "predicted_absolute_movement_bps",
        "state_context_future_range_bps",
        "state_context_absolute_movement_bps",
        "range_to_cost_ratio",
        "movement_permission",
        "scheduled_bars_remaining",
    ]
    write_deterministic_parquet(
        feature_ledger[movement_columns],
        output_root / "movement_permission_ledger.parquet",
        sort_by=["opportunity_id"],
    )
    motif_census = (
        feature_ledger.groupby(
            ["period", "decision_clock", "state_motif_2", "state_motif_3", "state_motif_4"],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("rows")
        .reset_index()
    )
    write_deterministic_csv(
        motif_census,
        output_root / "state_motif_census.csv",
        sort_by=["period", "decision_clock", "state_motif_2", "state_motif_3", "state_motif_4"],
    )
    write_deterministic_json(anchor_audit, output_root / "causal_anchor_audit.json")
    run_metadata = {
        "run_id": run_id,
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_sha256(CONTRACT_PATH),
        "feature_schema_sha256": sha256_file(FEATURE_SCHEMA_PATH),
        "data_snapshot_sha256": data_snapshot_hash,
        "feature_ledger_sha256": feature_hash,
        "git_sha": _git_sha(),
        "population_rows": len(events),
        "feature_rows": len(feature_ledger),
        "outcome_rows": len(outcomes),
        "scored_outcome_rows": int(outcomes["target"].ne("UNAVAILABLE").sum()),
        "unavailable_outcome_rows": int(outcomes["target"].eq("UNAVAILABLE").sum()),
        "decision_ordinals": contract["population"]["decision_ordinals"],
        "research_only": True,
        "execution_enabled": False,
    }
    write_deterministic_json(run_metadata, output_root / "run_metadata.json")
    return run_metadata


def write_prior_experiment_coverage(output_root: Path) -> None:
    """Freeze the prior-work distinction established before atlas scoring."""

    rows = [
        {
            "experiment": "Long/Short/Neutral Detector V1",
            "already_tested": (
                "fixed ordinals 12/36; one general multinomial price-context equation; "
                "first-touch classification"
            ),
            "not_tested": (
                "bounded one-to-three-condition signature census with independent "
                "chronological validation and fixed-terminal economic labels"
            ),
        },
        {
            "experiment": "Selective Payoff Equations V1",
            "already_tested": "setup-selected payoff equations using broad loop-score information",
            "not_tested": "unfiltered fixed-clock population and compact interpretable atlas rules",
        },
        {
            "experiment": "Regime Utility Ablation V1",
            "already_tested": "incremental regime utility on existing structural populations",
            "not_tested": "directional signature discovery across compact feature families",
        },
        {
            "experiment": "Loop Burst Mechanism V1",
            "already_tested": "loop-burst mechanism attribution",
            "not_tested": "fixed-clock absolute or contemporaneous relative direction atlas",
        },
        {
            "experiment": "Causal Loop State Paths V1",
            "already_tested": "causal state-path reconstruction and route attribution",
            "not_tested": "small pre-move directional rule persistence",
        },
        {
            "experiment": "Dynamic Loop x Regime Profitability V1",
            "already_tested": "loop-regime profitability surfaces",
            "not_tested": "setup-free fixed decision clocks with neutral abstention",
        },
        {
            "experiment": "Dynamic Loop Temporary Payoff Edge State V2",
            "already_tested": "temporary payoff-state dynamics",
            "not_tested": (
                "bounded signature atlas with multiplicity and threshold-neighbour controls"
            ),
        },
        {
            "experiment": "Sequential Loop Competitor Veto V1",
            "already_tested": "sequential competitor-veto policy",
            "not_tested": "static fixed-clock directional census",
        },
        {
            "experiment": "Directed Economic Loop-Regime Rotation V1",
            "already_tested": "directed structural rotation attribution",
            "not_tested": "independent simple long and short signature libraries",
        },
        {
            "experiment": "Fixed One-Bar Entry Latency V1",
            "already_tested": "one-bar entry latency on frozen structural opportunities",
            "not_tested": (
                "direction discovery; used here only as a predeclared execution-delay stress"
            ),
        },
    ]
    write_deterministic_json(
        {
            "contract_id": "20260717-directional-signature-atlas-v1",
            "exact_signature_atlas_previously_tested": False,
            "experiments": rows,
            "conclusion": (
                "Components existed separately, but no prior experiment combined the setup-free "
                "fixed-clock terminal target, bounded one-to-three-condition census, chronological "
                "freeze, separate long/short/neutral libraries, movement gate, relative Track B, "
                "multiplicity controls, and prospective outcome-free logging."
            ),
        },
        output_root / "prior_experiment_coverage.json",
    )


def false_value() -> bool:
    """Keep the pre-outcome seal explicit and JSON-native."""

    return False


def seal_outcome_free_feature_ledger(
    output_root: Path,
    feature_ledger: pd.DataFrame,
    *,
    run_id: str,
    data_snapshot_hash: str,
) -> str:
    """Materialize and hash the causal feature ledger before any outcome read."""

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "outcome_free_feature_ledger.parquet"
    write_deterministic_parquet(feature_ledger, destination, sort_by=["opportunity_id"])
    feature_hash = sha256_file(destination)
    write_deterministic_json(
        {
            "run_id": run_id,
            "contract_sha256": contract_sha256(CONTRACT_PATH),
            "feature_schema_sha256": sha256_file(FEATURE_SCHEMA_PATH),
            "data_snapshot_sha256": data_snapshot_hash,
            "feature_ledger_sha256": feature_hash,
            "feature_rows": len(feature_ledger),
            "outcomes_joined_at_seal": False,
        },
        output_root / "pre_outcome_feature_manifest.json",
    )
    return feature_hash


def build_ledgers(output_root: Path) -> dict[str, Any]:
    contract = load_contract(CONTRACT_PATH)
    schema = _feature_schema()
    prior_contract = _population_contract()
    prior_runner = _load_module("atlas_prior_lsn", PRIOR_RUNNER_PATH)
    core = _load_module("atlas_frozen_movement_core", CORE_PATH)
    sources, data_snapshot_hash = _snapshot_sources(prior_runner, prior_contract)
    run_id = (
        f"directional-signature-atlas-v1-{contract_sha256(CONTRACT_PATH)[:12]}-"
        f"{data_snapshot_hash[:12]}"
    )
    events, population_counts, population_coverage = prior_runner.build_population_from_sources(
        prior_contract
    )
    expected_rows = int(contract["population"]["expected_population_rows"])
    if len(events) != expected_rows:
        raise AssertionError(f"fixed-clock population drifted: {len(events)} != {expected_rows}")
    frozen = load_frozen_movement_bundle(BUNDLE_ROOT, core)
    symbols = list(contract["population"]["symbols_2024_2025"])
    anchors, anchor_audit = build_causal_anchor_panel(
        core,
        frozen,
        provider_root=Path(contract["population"]["provider_root"]),
        symbols=symbols,
        as_of=AS_OF,
        decision_ordinals=contract["population"]["decision_ordinals"],
    )
    feature_ledger = build_feature_ledger(
        events,
        anchors,
        feature_schema=schema,
        round_trip_cost_bps=float(contract["costs"]["round_trip_bps"]),
    )
    feature_ledger.insert(0, "run_id", run_id)
    feature_ledger["contract_hash"] = contract_sha256(CONTRACT_PATH)
    feature_ledger["data_snapshot_hash"] = data_snapshot_hash
    feature_ledger["feature_schema_hash"] = sha256_file(FEATURE_SCHEMA_PATH)
    if feature_ledger["opportunity_id"].duplicated().any() or len(feature_ledger) != len(events):
        raise AssertionError("feature ledger does not preserve the exact opportunity population")
    seal_outcome_free_feature_ledger(
        output_root,
        feature_ledger,
        run_id=run_id,
        data_snapshot_hash=data_snapshot_hash,
    )
    # No provider outcome path is opened before the ledger and its hash exist above.
    outcomes, first_touch, outcome_coverage = build_outcome_ledgers(
        events,
        prior_contract,
        prior_runner,
        round_trip_cost_bps=float(contract["costs"]["round_trip_bps"]),
        horizon_bars=int(contract["population"]["horizon_bars"]),
    )
    delayed_outcomes, _, _ = build_outcome_ledgers(
        events,
        prior_contract,
        prior_runner,
        round_trip_cost_bps=float(contract["costs"]["round_trip_bps"]),
        horizon_bars=int(contract["population"]["horizon_bars"]),
        entry_delay_bars=2,
    )
    for frame in (outcomes, first_touch, delayed_outcomes):
        frame["run_id"] = run_id
    return _write_build_artifacts(
        output_root,
        contract=contract,
        feature_schema=schema,
        sources=sources,
        data_snapshot_hash=data_snapshot_hash,
        run_id=run_id,
        events=events,
        population_counts=population_counts,
        population_coverage=population_coverage,
        anchors=anchors,
        anchor_audit=anchor_audit,
        feature_ledger=feature_ledger,
        outcomes=outcomes,
        delayed_outcomes=delayed_outcomes,
        first_touch=first_touch,
        outcome_coverage=outcome_coverage,
    )


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return _safe_json(value.item())
    return str(value)


def _write_csv_allow_empty(
    frame: pd.DataFrame,
    path: Path,
    *,
    sort_by: list[str],
) -> None:
    if frame.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, lineterminator="\n")
        return
    write_deterministic_csv(frame, path, sort_by=sort_by)


def _joined_scoring_frame(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(output_root / "outcome_free_feature_ledger.parquet")
    outcomes = pd.read_parquet(output_root / "primary_economic_outcome_ledger.parquet")
    outcome_columns = [
        "opportunity_id",
        "target",
        "gross_long_return_bps",
        "net_long_return_bps",
        "gross_short_return_bps",
        "net_short_return_bps",
        "absolute_terminal_move_bps",
        "future_high_low_range_bps",
        "mfe_long_bps",
        "mae_long_bps",
        "mfe_short_bps",
        "mae_short_bps",
        "entry_timestamp",
        "entry_open",
        "terminal_timestamp",
        "terminal_close",
        "score_status",
    ]
    joined = features.merge(
        outcomes[outcome_columns],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
    )
    if joined["target"].isna().any() or len(joined) != len(features):
        raise AssertionError("outcome join changed the frozen opportunity population")
    joined["long_net_bps"] = joined["net_long_return_bps"]
    joined["short_net_bps"] = joined["net_short_return_bps"]
    return joined, outcomes


def score_experiment(output_root: Path) -> dict[str, Any]:
    """Run bounded Track A discovery, unchanged chronology, controller, and baselines."""

    contract = load_contract(CONTRACT_PATH)
    schema = _feature_schema()
    joined, outcomes = _joined_scoring_frame(output_root)
    scored = joined.loc[joined["target"].ne("UNAVAILABLE")].copy()
    discovery = scored.loc[scored["period"].eq(contract["chronology"]["discovery"]["period"])]
    validation = scored.loc[scored["period"].eq(contract["chronology"]["validation"]["period"])]
    final = scored.loc[
        scored["period"].eq(contract["chronology"]["final_opened_holdout"]["period"])
    ]
    if any(frame.empty for frame in (discovery, validation, final)):
        raise AssertionError("chronological discovery/validation/final split is incomplete")
    feature_families = feature_family_map(schema)
    ordered_bins = {
        str(row["name"]): list(row["bins"]) for row in schema["features"] if row.get("bins")
    }
    search = contract["search"]
    caps = SearchCaps(
        univariate_and_pairwise=int(search["univariate_and_pairwise_cap"]),
        triples=int(search["three_condition_cap"]),
        tree=int(search["tree_candidate_cap"]),
        retained=int(search["discovery_stage_retained_cap"]),
    )
    candidates, registry = generate_bounded_candidates(
        discovery,
        feature_families,
        caps,
        minimum_parent_support=int(contract["support"]["minimum_rows"]),
    )
    tree_candidates, tree_registry = extract_shallow_tree_candidates(
        discovery,
        feature_families,
        maximum_depth=int(search["tree_maximum_depth"]),
        minimum_leaf_rows=int(search["tree_minimum_leaf_rows"]),
        cap=int(search["tree_candidate_cap"]),
        seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]),
    )
    by_id = {candidate.signature_id: candidate for candidate in [*candidates, *tree_candidates]}
    candidates = [by_id[key] for key in sorted(by_id)]
    registry_by_id = {str(row["signature_id"]): row for row in [*registry, *tree_registry]}
    registry = [registry_by_id[key] for key in sorted(registry_by_id)]
    support = contract["support"]
    support_rules = SupportRules(
        minimum_rows=int(support["minimum_rows"]),
        minimum_sessions=int(support["minimum_independent_sessions"]),
        minimum_stocks=int(support["minimum_independent_stocks"]),
        maximum_stock_fraction=float(support["maximum_single_stock_row_fraction"]),
        minimum_months=int(support["minimum_calendar_months"]),
        minimum_directional_outcomes=int(support["minimum_relevant_direction_outcomes"]),
    )
    census = evaluate_candidate_census(
        discovery,
        candidates,
        registry,
        support_rules=support_rules,
        ordered_bins=ordered_bins,
        fdr_q=float(contract["multiplicity"]["broad_discovery_q"]),
    )
    neutral_census, neutral_discovery_library = evaluate_neutral_veto_census(
        discovery,
        candidates,
        support_rules=support_rules,
        fdr_q=float(contract["multiplicity"]["broad_discovery_q"]),
        cap=5,
    )
    neutral_validation_metrics, neutral_survivor_library = validate_neutral_veto_library(
        validation,
        neutral_discovery_library,
        support_rules=support_rules,
        holm_alpha=float(contract["multiplicity"]["retained_family_alpha"]),
    )
    neutral_final_metrics, _ = validate_neutral_veto_library(
        final,
        neutral_survivor_library,
        support_rules=support_rules,
        holm_alpha=float(contract["multiplicity"]["retained_family_alpha"]),
    )
    discovery_library = freeze_discovery_library(
        census,
        candidates,
        discovery,
        retained_stage_cap=int(search["discovery_stage_retained_cap"]),
        per_direction_cap=int(search["frozen_discovery_long_cap"]),
    )
    validation_metrics, survivor_library = validate_discovery_library(
        validation,
        discovery_library,
        support_rules=support_rules,
        holm_alpha=float(contract["multiplicity"]["retained_family_alpha"]),
        per_direction_cap=int(search["validation_survivor_long_cap"]),
    )
    final_metrics = score_frozen_library(
        final,
        survivor_library,
        bootstrap_draws=int(contract["multiplicity"]["session_block_bootstrap_draws"]),
        bootstrap_seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]),
    )
    discovery_probabilities = {
        label: float((discovery["target"].eq(label).sum() + 1.0) / (len(discovery) + 3.0))
        for label in CLASSES
    }
    atlas_decisions = apply_atlas_controller(
        scored,
        survivor_library,
        base_probabilities=discovery_probabilities,
    )
    baselines = baseline_predictions(discovery, scored)
    all_predictions = pd.concat([baselines, atlas_decisions], ignore_index=True)
    metric_outcomes = scored[
        [
            "opportunity_id",
            "target",
            "long_net_bps",
            "short_net_bps",
            "round_trip_cost_bps",
        ]
    ]
    predictive, economic = prediction_metrics(all_predictions, metric_outcomes)
    breakdowns = signature_breakdowns(scored, discovery_library)
    delayed_outcomes = pd.read_parquet(output_root / "one_bar_delay_outcome_ledger.parquet")
    stress_results, leave_one_out = stress_signature_library(
        scored,
        discovery_library,
        delayed_outcomes,
        ordered_bins=ordered_bins,
    )
    random_generator = np.random.default_rng(
        int(contract["multiplicity"]["session_block_bootstrap_seed"])
    )
    random_signatures = []
    used_random_ids: set[str] = set()
    frozen_ids = {str(entry["signature"]["signature_id"]) for entry in discovery_library}
    for entry in discovery_library:
        condition_count = len(entry["signature"]["conditions"])
        target_rows = int(entry["discovery_metrics"]["rows"])
        pool = census.loc[
            census["condition_count"].eq(condition_count)
            & ~census["signature_id"].isin(frozen_ids | used_random_ids)
        ].copy()
        if pool.empty:
            continue
        pool["support_distance"] = (pool["rows"] - target_rows).abs()
        nearest = pool.sort_values(["support_distance", "signature_id"], kind="mergesort").head(
            min(20, len(pool))
        )
        chosen_id = str(
            nearest.iloc[int(random_generator.integers(0, len(nearest)))]["signature_id"]
        )
        used_random_ids.add(chosen_id)
        random_signatures.append(by_id[chosen_id])
    nulls = null_test_results(
        scored,
        discovery_library,
        random_signatures,
        atlas_decisions,
        seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]),
    )

    parquet_census = census.drop(
        columns=["conditions", "rejection_reasons", "support_reasons"],
        errors="ignore",
    )
    write_deterministic_parquet(
        parquet_census,
        output_root / "complete_candidate_registry.parquet",
        sort_by=["signature_id"],
    )
    concise_columns = [
        "signature_id",
        "direction",
        "stage",
        "condition_count",
        "rows",
        "sessions",
        "stocks",
        "mean_directional_net_bps",
        "directional_lift",
        "raw_p_value",
        "fdr_q_value",
        "discovery_score",
        "discovery_supported_effect",
        "discovery_eligible",
        "rejection_reasons_json",
        "conditions_json",
    ]
    _write_csv_allow_empty(
        census[concise_columns],
        output_root / "complete_candidate_registry.csv",
        sort_by=["signature_id"],
    )
    for stage, filename in (
        ("univariate", "univariate_signature_census.csv"),
        ("pairwise", "pairwise_signature_census.csv"),
        ("three_condition", "three_condition_signature_census.csv"),
        ("shallow_tree", "shallow_tree_extracted_signatures.csv"),
    ):
        _write_csv_allow_empty(
            census.loc[census["stage"].eq(stage), concise_columns],
            output_root / filename,
            sort_by=["signature_id"],
        )
    _write_csv_allow_empty(
        census.loc[census["discovery_supported_effect"]],
        output_root / "discovery_signature_metrics.csv",
        sort_by=["direction", "discovery_score", "signature_id"],
    )
    _write_csv_allow_empty(
        census[["signature_id", "rejection_reasons_json"]],
        output_root / "candidate_rejection_reasons.csv",
        sort_by=["signature_id"],
    )
    neutral_columns = [
        "neutral_veto_id",
        "condition_count",
        "conditions_json",
        "rows",
        "sessions",
        "stocks",
        "neutral_rate",
        "base_neutral_rate",
        "neutral_lift",
        "mean_long_net_bps",
        "mean_short_net_bps",
        "raw_p_value",
        "fdr_q_value",
        "neutral_score",
        "neutral_discovery_eligible",
        "rejection_reasons_json",
    ]
    _write_csv_allow_empty(
        neutral_census[neutral_columns],
        output_root / "neutral_veto_census.csv",
        sort_by=["neutral_veto_id"],
    )
    _write_csv_allow_empty(
        neutral_validation_metrics,
        output_root / "neutral_veto_validation_metrics.csv",
        sort_by=["neutral_veto_id"],
    )
    _write_csv_allow_empty(
        neutral_final_metrics,
        output_root / "neutral_veto_final_opened_holdout_metrics.csv",
        sort_by=["neutral_veto_id"],
    )
    write_deterministic_json(
        _safe_json(neutral_survivor_library), output_root / "neutral_veto_library.json"
    )
    write_deterministic_json(
        _safe_json(discovery_library), output_root / "frozen_discovery_signature_library.json"
    )
    _write_csv_allow_empty(
        validation_metrics,
        output_root / "validation_signature_metrics.csv",
        sort_by=["direction", "discovery_score", "signature_id"],
    )
    write_deterministic_json(
        _safe_json(survivor_library), output_root / "frozen_validation_survivor_library.json"
    )
    _write_csv_allow_empty(
        final_metrics,
        output_root / "final_opened_holdout_signature_metrics.csv",
        sort_by=["direction", "discovery_score", "signature_id"],
    )
    for direction, filename in (
        ("LONG", "long_signature_library.json"),
        ("SHORT", "short_signature_library.json"),
    ):
        write_deterministic_json(
            _safe_json(
                [
                    entry
                    for entry in survivor_library
                    if entry["signature"]["direction"] == direction
                ]
            ),
            output_root / filename,
        )
    write_deterministic_parquet(
        atlas_decisions,
        output_root / "atlas_level_decisions.parquet",
        sort_by=["opportunity_id"],
    )
    write_deterministic_parquet(
        baselines,
        output_root / "baseline_predictions.parquet",
        sort_by=["model_id", "opportunity_id"],
    )
    _write_csv_allow_empty(
        predictive,
        output_root / "predictive_calibration_metrics.csv",
        sort_by=["model_id", "period"],
    )
    _write_csv_allow_empty(
        economic,
        output_root / "economic_metrics.csv",
        sort_by=["model_id", "period"],
    )
    _write_csv_allow_empty(
        breakdowns,
        output_root / "concentration_results.csv",
        sort_by=["signature_id", "dimension", "value"],
    )
    _write_csv_allow_empty(
        stress_results,
        output_root / "cost_and_delay_stress_results.csv",
        sort_by=["signature_id", "period", "stress", "removed"],
    )
    _write_csv_allow_empty(
        leave_one_out,
        output_root / "leave_one_stock_out_attribution.csv",
        sort_by=["signature_id", "period", "removed"],
    )
    _write_csv_allow_empty(
        nulls,
        output_root / "null_test_results.csv",
        sort_by=["null"],
    )
    base_rates = (
        scored.groupby("period", sort=True)["target"]
        .value_counts(normalize=True)
        .rename("rate")
        .reset_index()
    )
    _write_csv_allow_empty(
        base_rates,
        output_root / "base_rates_by_period.csv",
        sort_by=["period", "target"],
    )
    summary = {
        "candidate_signatures_examined": len(census),
        "univariate_candidates": int(census["stage"].eq("univariate").sum()),
        "pairwise_candidates": int(census["stage"].eq("pairwise").sum()),
        "three_condition_candidates": int(census["stage"].eq("three_condition").sum()),
        "tree_candidates": int(census["stage"].eq("shallow_tree").sum()),
        "discovery_eligible_candidates": int(census["discovery_eligible"].sum()),
        "discovery_supported_effect_candidates": int(census["discovery_supported_effect"].sum()),
        "frozen_discovery_long": sum(
            entry["signature"]["direction"] == "LONG" for entry in discovery_library
        ),
        "frozen_discovery_short": sum(
            entry["signature"]["direction"] == "SHORT" for entry in discovery_library
        ),
        "validation_survivor_long": sum(
            entry["signature"]["direction"] == "LONG" for entry in survivor_library
        ),
        "validation_survivor_short": sum(
            entry["signature"]["direction"] == "SHORT" for entry in survivor_library
        ),
        "neutral_discovery_survivors": len(neutral_discovery_library),
        "neutral_validation_survivors": len(neutral_survivor_library),
        "neutral_final_same_sign": int(
            neutral_final_metrics.get("neutral_lift", pd.Series(dtype=float)).gt(0.0).sum()
        ),
        "atlas_directional_outputs_validation": int(
            atlas_decisions.loc[atlas_decisions["period"].eq(2025), "predicted_state"]
            .isin(["LONG", "SHORT"])
            .sum()
        ),
        "atlas_directional_outputs_final": int(
            atlas_decisions.loc[atlas_decisions["period"].eq(2026), "predicted_state"]
            .isin(["LONG", "SHORT"])
            .sum()
        ),
        "research_only": True,
        "execution_enabled": False,
    }
    write_deterministic_json(summary, output_root / "track_a_summary.json")
    summary["track_b"] = run_track_b(
        scored,
        output_root=output_root / "track_b",
        contract=contract,
        feature_families=feature_families,
        ordered_bins=ordered_bins,
        caps=caps,
        support_rules=support_rules,
    )
    write_deterministic_json(_safe_json(summary), output_root / "track_a_summary.json")
    generate_plots(
        output_root,
        scored=scored,
        base_rates=base_rates,
        census=census,
        discovery_library=discovery_library,
        validation_metrics=validation_metrics,
        neutral_validation_metrics=neutral_validation_metrics,
        neutral_final_metrics=neutral_final_metrics,
        economic=economic,
    )
    write_prospective_schemas(output_root)
    write_artifact_manifest(output_root)
    return summary


def generate_plots(
    output_root: Path,
    *,
    scored: pd.DataFrame,
    base_rates: pd.DataFrame,
    census: pd.DataFrame,
    discovery_library: list[dict[str, Any]],
    validation_metrics: pd.DataFrame,
    neutral_validation_metrics: pd.DataFrame,
    neutral_final_metrics: pd.DataFrame,
    economic: pd.DataFrame,
) -> None:
    """Write a compact deterministic scientific plot set."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    colors = {"LONG": "#2878B5", "SHORT": "#C44E52", "NEUTRAL": "#8A8A8A"}

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(
            plot_root / name,
            dpi=140,
            metadata={"Software": "Stocker Directional Signature Atlas V1"},
        )
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    pivot = base_rates.pivot(index="period", columns="target", values="rate").fillna(0.0)
    pivot[[label for label in ("LONG", "SHORT", "NEUTRAL") if label in pivot]].plot(
        kind="bar",
        ax=axis,
        color=[colors[label] for label in pivot.columns],
        width=0.78,
    )
    axis.set(title="Economic target base rates", xlabel="Opened period", ylabel="Rate")
    axis.legend(frameon=False)
    save(fig, "base_rates_by_period.png")

    validation_by_id = (
        validation_metrics.set_index("signature_id")
        if not validation_metrics.empty
        else pd.DataFrame()
    )
    effect_rows: list[dict[str, Any]] = []
    for entry in discovery_library[:10]:
        signature_id = str(entry["signature"]["signature_id"])
        effect_rows.append(
            {
                "signature": signature_id[-20:],
                "stage": "discovery",
                "mean_net_bps": float(entry["discovery_metrics"]["mean_directional_net_bps"]),
            }
        )
        if not validation_by_id.empty and signature_id in validation_by_id.index:
            effect_rows.append(
                {
                    "signature": signature_id[-20:],
                    "stage": "validation",
                    "mean_net_bps": float(
                        cast(Any, validation_by_id.loc[signature_id, "mean_directional_net_bps"])
                    ),
                }
            )
    effect = pd.DataFrame(effect_rows)
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    if not effect.empty:
        effect.pivot(index="signature", columns="stage", values="mean_net_bps").plot.bar(
            ax=axis, color=["#4C72B0", "#DD8452"]
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="Frozen discovery rules: discovery versus validation",
        xlabel="Signature suffix",
        ylabel="Directional mean net bps",
    )
    axis.tick_params(axis="x", labelrotation=65)
    save(fig, "frozen_signature_period_effects.png")

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for complexity, group in census.groupby("condition_count", sort=True):
        axis.scatter(
            group["rows"],
            group["mean_directional_net_bps"],
            s=12,
            alpha=0.40,
            label=f"{int(cast(Any, complexity))} condition(s)",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xscale("log")
    axis.set(
        title="Support versus discovery effect", xlabel="Rows (log scale)", ylabel="Mean net bps"
    )
    axis.legend(frameon=False, fontsize=8)
    save(fig, "support_vs_effect.png")

    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.scatter(
        census["long_lift"],
        census["short_lift"],
        c=census["condition_count"],
        cmap="viridis",
        s=12,
        alpha=0.45,
    )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.axvline(0.0, color="black", linewidth=0.7)
    axis.set(title="Long lift versus short lift", xlabel="Long lift", ylabel="Short lift")
    save(fig, "long_lift_vs_short_lift.png")

    movement = scored.assign(
        absolute_terminal_move_bps=np.maximum(
            scored["long_net_bps"].to_numpy(float), scored["short_net_bps"].to_numpy(float)
        )
        + 10.0
    )
    movement_summary = (
        movement.groupby(["period", "movement_permission"], sort=True)["absolute_terminal_move_bps"]
        .mean()
        .unstack()
        .fillna(np.nan)
    )
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    movement_summary.rename(
        columns={False: "permission failed", True: "permission passed"}
    ).plot.bar(ax=axis, color=["#999999", "#55A868"])
    axis.set(
        title="Direction-neutral movement permission",
        xlabel="Opened period",
        ylabel="Mean absolute terminal move (bps)",
    )
    save(fig, "movement_permission_impact.png")

    selected_models = [
        "directional_signature_atlas_v1",
        "one_bar_momentum",
        "one_bar_reversal",
        "current_state_alone",
        "current_state_plus_history",
    ]
    comparison = economic.loc[
        economic["model_id"].isin(selected_models) & economic["period"].isin([2025, 2026])
    ]
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    if not comparison.empty:
        comparison.pivot(
            index="model_id", columns="period", values="net_bps_per_full_opportunity"
        ).plot.bar(ax=axis, color=["#4C72B0", "#DD8452"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Atlas and simple baselines", xlabel="Model", ylabel="Net bps per opportunity")
    axis.tick_params(axis="x", labelrotation=35)
    save(fig, "atlas_vs_baselines.png")

    neutral_rows: list[pd.DataFrame] = []
    if not neutral_validation_metrics.empty:
        neutral_rows.append(
            neutral_validation_metrics[["neutral_veto_id", "neutral_lift"]].assign(
                period="validation"
            )
        )
    if not neutral_final_metrics.empty:
        neutral_rows.append(
            neutral_final_metrics[["neutral_veto_id", "neutral_lift"]].assign(period="final")
        )
    fig, axis = plt.subplots(figsize=(8.0, 4.4))
    if neutral_rows:
        pd.concat(neutral_rows, ignore_index=True).pivot(
            index="neutral_veto_id", columns="period", values="neutral_lift"
        ).plot.bar(ax=axis, color=["#8172B3", "#64B5CD"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(title="Frozen neutral-veto lift", xlabel="Neutral veto", ylabel="Neutral-rate lift")
    axis.tick_params(axis="x", labelrotation=65)
    save(fig, "neutral_veto_period_effects.png")

    relative = pd.read_parquet(output_root / "track_b/relative_outcome_ledger.parquet")
    relative_rates = (
        relative.groupby("period", sort=True)["target"]
        .value_counts(normalize=True)
        .rename("rate")
        .reset_index()
        .pivot(index="period", columns="target", values="rate")
        .fillna(0.0)
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    relative_rates.plot.bar(
        ax=axis,
        color=[colors.get(str(label), "#777777") for label in relative_rates.columns],
    )
    axis.set(title="Track B relative target rates", xlabel="Opened period", ylabel="Rate")
    save(fig, "track_b_relative_target_rates.png")


def write_prospective_schemas(output_root: Path) -> None:
    forecast_fields = [
        "run_id",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "feature_schema_hash",
        "opportunity_id",
        "symbol",
        "session",
        "decision_clock",
        "decision_timestamp",
        "entry_timestamp",
        "terminal_timestamp",
        "causal_features",
        "feature_availability_timestamps",
        "movement_permission",
        "long_signature_decisions",
        "short_signature_decisions",
        "long_vote_count",
        "short_vote_count",
        "conflict_state",
        "final_atlas_state",
        "reason_codes",
        "forecast_freeze_timestamp",
        "research_only",
        "execution_enabled",
    ]
    settlement_fields = [
        "opportunity_id",
        "terminal_timestamp",
        "gross_long_payoff_bps",
        "gross_short_payoff_bps",
        "costs_bps",
        "net_long_payoff_bps",
        "net_short_payoff_bps",
        "primary_target",
        "secondary_first_touch_target",
        "settlement_timestamp",
        "settlement_code_version",
    ]
    write_deterministic_json(
        {
            "schema_id": "directional-signature-atlas-v1-prospective-forecast",
            "append_only": True,
            "immutable": True,
            "required_fields": forecast_fields,
            "outcomes_forbidden": True,
            "research_only": True,
            "execution_enabled": False,
        },
        output_root / "prospective_forecast_ledger_schema.json",
    )
    write_deterministic_json(
        {
            "schema_id": "directional-signature-atlas-v1-prospective-settlement",
            "append_only": True,
            "separate_from_forecast": True,
            "required_fields": settlement_fields,
            "forecast_overwrite_allowed": False,
            "research_only": True,
            "execution_enabled": False,
        },
        output_root / "prospective_settlement_ledger_schema.json",
    )


def write_artifact_manifest(output_root: Path) -> None:
    excluded = {
        "artifact_manifest.json",
        "independent_audit.json",
        "prospective_forecast_dry_run.json",
        "prospective_settlement_dry_run.json",
    }
    files = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        files.append(
            {
                "relative_path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_deterministic_json(
        {
            "contract_id": "20260717-directional-signature-atlas-v1",
            "machine_readable_file_count": len(files),
            "files": files,
            "research_only": True,
            "execution_enabled": False,
        },
        output_root / "artifact_manifest.json",
    )


def prospective_forecast_dry_run(output_root: Path, prospective_root: Path) -> dict[str, Any]:
    metadata = json.loads((output_root / "run_metadata.json").read_text(encoding="utf-8"))
    long_library = json.loads((output_root / "long_signature_library.json").read_text())
    short_library = json.loads((output_root / "short_signature_library.json").read_text())
    ledger = ProspectiveLedger(prospective_root, opened_through="2026-06-26")
    opportunity_id = "atlas|2026|DRYRUN|2026-07-20|12"
    record = {
        "run_id": metadata["run_id"],
        "git_sha": metadata["git_sha"],
        "contract_hash": metadata["contract_sha256"],
        "data_snapshot_hash": metadata["data_snapshot_sha256"],
        "feature_schema_hash": metadata["feature_schema_sha256"],
        "opportunity_id": opportunity_id,
        "symbol": "DRYRUN",
        "session": "2026-07-20",
        "decision_clock": "clock_12",
        "decision_timestamp": "2026-07-20T14:30:00+00:00",
        "entry_timestamp": "2026-07-20T14:35:00+00:00",
        "terminal_timestamp": "2026-07-20T16:30:00+00:00",
        "causal_features": {"decision_clock": "clock_12", "movement_permission": False},
        "feature_availability_timestamps": {
            "decision_clock": "2026-07-20T14:30:00+00:00",
            "movement_permission": "2026-07-20T14:30:00+00:00",
        },
        "movement_permission": False,
        "long_signature_decisions": {
            entry["signature"]["signature_id"]: False for entry in long_library
        },
        "short_signature_decisions": {
            entry["signature"]["signature_id"]: False for entry in short_library
        },
        "long_vote_count": 0,
        "short_vote_count": 0,
        "conflict_state": False,
        "final_atlas_state": "NEUTRAL",
        "reason_codes": ["movement_permission_failed", "dry_run"],
        "forecast_freeze_timestamp": "2026-07-20T14:30:00+00:00",
        "research_only": True,
        "execution_enabled": False,
    }
    ledger.append_forecast(record)
    result = {
        "opportunity_id": opportunity_id,
        "forecast_path": str(ledger.forecast_path),
        "forecast_record_count": 1,
        "research_only": True,
        "execution_enabled": False,
    }
    write_deterministic_json(result, output_root / "prospective_forecast_dry_run.json")
    return result


def prospective_settlement_dry_run(output_root: Path, prospective_root: Path) -> dict[str, Any]:
    ledger = ProspectiveLedger(prospective_root, opened_through="2026-06-26")
    opportunity_id = "atlas|2026|DRYRUN|2026-07-20|12"
    ledger.append_settlement(
        {
            "opportunity_id": opportunity_id,
            "terminal_timestamp": "2026-07-20T16:30:00+00:00",
            "gross_long_payoff_bps": 0.0,
            "gross_short_payoff_bps": 0.0,
            "costs_bps": 10.0,
            "net_long_payoff_bps": -10.0,
            "net_short_payoff_bps": -10.0,
            "primary_target": "NEUTRAL",
            "secondary_first_touch_target": "NEITHER",
            "settlement_timestamp": "2026-07-20T16:31:00+00:00",
            "settlement_code_version": _git_sha(),
            "research_only": True,
            "execution_enabled": False,
        }
    )
    result = {
        "opportunity_id": opportunity_id,
        "forecast_path_unchanged": True,
        "settlement_path": str(ledger.settlement_path),
        "settlement_record_count": 1,
        "research_only": True,
        "execution_enabled": False,
    }
    write_deterministic_json(result, output_root / "prospective_settlement_dry_run.json")
    return result


def run_track_b(
    absolute_scored: pd.DataFrame,
    *,
    output_root: Path,
    contract: dict[str, Any],
    feature_families: dict[str, str],
    ordered_bins: dict[str, list[Any]],
    caps: SearchCaps,
    support_rules: SupportRules,
) -> dict[str, Any]:
    """Run Track B only after Track A artifacts and conclusion are materialized."""

    output_root.mkdir(parents=True, exist_ok=True)
    relative_outcomes = construct_relative_outcomes(absolute_scored)
    relative = absolute_scored.copy()
    relative = relative.drop(
        columns=["target", "long_net_bps", "short_net_bps", "round_trip_cost_bps"]
    ).merge(
        relative_outcomes[
            [
                "opportunity_id",
                "target",
                "long_net_bps",
                "short_net_bps",
                "round_trip_cost_bps",
                "future_residual_return_bps",
                "future_residual_percentile",
                "peer_count",
            ]
        ],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
    )
    relative = relative.loc[relative["target"].ne("UNAVAILABLE")].copy()
    discovery = relative.loc[relative["period"].eq(2024)]
    validation = relative.loc[relative["period"].eq(2025)]
    final = relative.loc[relative["period"].eq(2026)]
    candidates, registry = generate_bounded_candidates(
        discovery,
        feature_families,
        caps,
        minimum_parent_support=support_rules.minimum_rows,
    )
    tree_candidates, tree_registry = extract_shallow_tree_candidates(
        discovery,
        feature_families,
        maximum_depth=int(contract["search"]["tree_maximum_depth"]),
        minimum_leaf_rows=int(contract["search"]["tree_minimum_leaf_rows"]),
        cap=int(contract["search"]["tree_candidate_cap"]),
        seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 1000,
    )
    candidate_map = {
        candidate.signature_id: candidate for candidate in [*candidates, *tree_candidates]
    }
    candidates = [candidate_map[key] for key in sorted(candidate_map)]
    registry_map = {str(row["signature_id"]): row for row in [*registry, *tree_registry]}
    registry = [registry_map[key] for key in sorted(registry_map)]
    census = evaluate_candidate_census(
        discovery,
        candidates,
        registry,
        support_rules=support_rules,
        ordered_bins=ordered_bins,
        fdr_q=float(contract["multiplicity"]["broad_discovery_q"]),
    )
    discovery_library = freeze_discovery_library(
        census,
        candidates,
        discovery,
        retained_stage_cap=int(contract["search"]["discovery_stage_retained_cap"]),
        per_direction_cap=int(contract["search"]["frozen_discovery_long_cap"]),
    )
    validation_metrics, survivor_library = validate_discovery_library(
        validation,
        discovery_library,
        support_rules=support_rules,
        holm_alpha=float(contract["multiplicity"]["retained_family_alpha"]),
        per_direction_cap=int(contract["search"]["validation_survivor_long_cap"]),
    )
    final_metrics = score_frozen_library(
        final,
        survivor_library,
        bootstrap_draws=int(contract["multiplicity"]["session_block_bootstrap_draws"]),
        bootstrap_seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 1000,
    )
    base_probabilities = {
        label: float((discovery["target"].eq(label).sum() + 1.0) / (len(discovery) + 3.0))
        for label in CLASSES
    }
    relative_atlas = apply_atlas_controller(
        relative,
        survivor_library,
        base_probabilities=base_probabilities,
    )
    relative_predictive, relative_economic = prediction_metrics(
        relative_atlas,
        relative[
            [
                "opportunity_id",
                "target",
                "long_net_bps",
                "short_net_bps",
                "round_trip_cost_bps",
            ]
        ],
    )
    strength = relative_strength_baseline(relative_outcomes, absolute_scored)
    strength_economic = relative_baseline_economic_metrics(strength, relative_outcomes)
    write_deterministic_parquet(
        relative_outcomes,
        output_root / "relative_outcome_ledger.parquet",
        sort_by=["opportunity_id"],
    )
    write_deterministic_parquet(
        census.drop(columns=["conditions", "rejection_reasons"], errors="ignore"),
        output_root / "relative_candidate_registry.parquet",
        sort_by=["signature_id"],
    )
    _write_csv_allow_empty(
        census,
        output_root / "relative_discovery_signature_metrics.csv",
        sort_by=["signature_id"],
    )
    write_deterministic_json(
        _safe_json(discovery_library), output_root / "relative_discovery_library.json"
    )
    _write_csv_allow_empty(
        validation_metrics,
        output_root / "relative_validation_metrics.csv",
        sort_by=["signature_id"],
    )
    write_deterministic_json(
        _safe_json(survivor_library), output_root / "relative_survivor_library.json"
    )
    _write_csv_allow_empty(
        final_metrics,
        output_root / "relative_final_opened_holdout_metrics.csv",
        sort_by=["signature_id"],
    )
    write_deterministic_parquet(
        relative_atlas,
        output_root / "relative_atlas_decisions.parquet",
        sort_by=["opportunity_id"],
    )
    write_deterministic_parquet(
        strength,
        output_root / "relative_strength_baseline_predictions.parquet",
        sort_by=["opportunity_id"],
    )
    _write_csv_allow_empty(
        relative_predictive,
        output_root / "relative_predictive_metrics.csv",
        sort_by=["model_id", "period"],
    )
    _write_csv_allow_empty(
        pd.concat(
            [
                relative_economic.assign(metric_basis="relative_atlas"),
                strength_economic.assign(metric_basis="relative_strength_baseline"),
            ],
            ignore_index=True,
        ),
        output_root / "relative_economic_metrics.csv",
        sort_by=["period", "metric_basis"],
    )
    summary = {
        "status": "completed_secondary_equal_universe_only",
        "sector_relative_status": "unavailable_no_frozen_sector_membership",
        "candidate_signatures_examined": len(census),
        "discovery_multiplicity_qualified": int(census["discovery_eligible"].sum()),
        "frozen_exploratory_signatures": len(discovery_library),
        "validation_survivors": len(survivor_library),
        "final_scored_signatures": len(final_metrics),
        "absolute_profitability_claim_allowed": False,
    }
    write_deterministic_json(summary, output_root / "track_b_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "build",
            "score",
            "run",
            "prospective-dry-run",
            "settlement-dry-run",
        ),
        default="run",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prospective-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase in {"build", "run"}:
        metadata = build_ledgers(args.output_root)
        print(json.dumps(metadata, sort_keys=True, indent=2))
    if args.phase in {"score", "run"}:
        summary = score_experiment(args.output_root)
        print(json.dumps(summary, sort_keys=True, indent=2))
    if args.phase in {"prospective-dry-run", "settlement-dry-run"}:
        if args.prospective_root is None:
            raise ValueError("--prospective-root is required for prospective dry runs")
        if args.phase == "prospective-dry-run":
            result = prospective_forecast_dry_run(args.output_root, args.prospective_root)
        else:
            result = prospective_settlement_dry_run(args.output_root, args.prospective_root)
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
