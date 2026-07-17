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
    atlas_concentration,
    evaluate_candidate_census,
    evaluate_neutral_veto_census,
    freeze_discovery_library,
    paired_simple_baseline_metrics,
    score_frozen_library,
    signature_attribution_rows,
    signature_breakdowns,
    signature_from_dict,
    signature_metrics,
    signature_probability_metrics,
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
    recompute_cross_sectional_after_stock_deletion,
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
from stocker_research.directional_signature_atlas.prospective import (
    ProspectiveLedger,
    build_forecast_record,
    build_settlement_record,
    canonical_library_hash,
)
from stocker_research.directional_signature_atlas.robustness import (
    null_test_results,
    stress_neutral_veto_library,
    stress_signature_library,
)
from stocker_research.directional_signature_atlas.signatures import (
    SearchCaps,
    SupportRules,
    candidate_search_space_counts,
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
ATLAS_PACKAGE_ROOT = (
    HERE.parents[3] / "packages/stocker_research/src/stocker_research/directional_signature_atlas"
)
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
            "frozen_movement_core": CORE_PATH,
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
    for path in sorted(ATLAS_PACKAGE_ROOT.glob("*.py")):
        sources[f"atlas_module_{path.stem}"] = path
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
    feature_rows = cast(list[dict[str, Any]], payload.get("features", []))
    names = [str(row.get("name")) for row in feature_rows]
    if len(names) != len(set(names)):
        raise ValueError("feature schema contains duplicate names")
    missing_order = [
        str(row["name"])
        for row in feature_rows
        if row.get("type") == "ordered_categorical" and not row.get("bins")
    ]
    if missing_order:
        raise ValueError(f"ordered feature bins are not frozen: {missing_order}")
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
    sealed_feature_hash: str,
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
    feature_path = output_root / "outcome_free_feature_ledger.parquet"
    manifest_path = output_root / "pre_outcome_feature_manifest.json"
    if not feature_path.is_file() or not manifest_path.is_file():
        raise AssertionError("pre-outcome feature seal disappeared before outcome write")
    if sha256_file(feature_path) != sealed_feature_hash:
        raise AssertionError("pre-outcome feature ledger changed after outcome read")
    sealed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sealed_manifest.get("feature_ledger_sha256") != sealed_feature_hash:
        raise AssertionError("pre-outcome feature manifest changed after outcome read")
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
        "feature_ledger_sha256": sealed_feature_hash,
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

    rows: list[dict[str, Any]] = [
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
    evidence_slugs: list[str | None] = [
        "20260714-long-short-neutral-detector-v1",
        "20260714-selective-payoff-equations-v1",
        "20260711-regime-utility-ablation-v1",
        "20260711-loop-burst-mechanism-v1",
        "20260714-causal-loop-state-path-v1",
        "20260713-dynamic-loop-context-edge-v1",
        "20260714-dynamic-loop-edge-state-v2",
        "20260715-sequential-loop-competitor-veto-v1",
        "20260716-directed-economic-loop-regime-rotation-v1",
        "20260716-fixed-one-bar-entry-latency-v1",
    ]
    runner_names = {
        "20260714-long-short-neutral-detector-v1": "run_long_short_neutral_detector_v1.py",
        "20260714-selective-payoff-equations-v1": "run_selective_payoff_equations_v1.py",
        "20260711-regime-utility-ablation-v1": "run_regime_utility_ablation_v1.py",
        "20260711-loop-burst-mechanism-v1": "run_loop_burst_mechanism_v1.py",
        "20260714-causal-loop-state-path-v1": "run_causal_loop_state_path_v1.py",
        "20260713-dynamic-loop-context-edge-v1": "run_dynamic_loop_context_edge_v1.py",
        "20260714-dynamic-loop-edge-state-v2": "run_dynamic_loop_edge_state_v2.py",
        "20260715-sequential-loop-competitor-veto-v1": "run_sequential_loop_competitor_veto_v1.py",
        "20260716-directed-economic-loop-regime-rotation-v1": (
            "run_directed_economic_loop_regime_rotation_v1.py"
        ),
        "20260716-fixed-one-bar-entry-latency-v1": "run_fixed_one_bar_entry_latency_v1.py",
    }
    auditor_names = {
        slug: runner.replace("run_", "audit_", 1) for slug, runner in runner_names.items()
    }
    for row, slug in zip(rows, evidence_slugs, strict=True):
        if slug is None:
            row["evidence_status"] = "exact_named_experiment_not_found"
            row["closest_repository_surface"] = (
                "20260713-dynamic-loop-context-edge-v1; not silently treated as the exact title"
            )
            row["evidence"] = []
            continue
        candidates = {
            "contract": HERE / f"contracts/{slug}.json",
            "report": HERE / f"reports/{slug}.md",
            "runner": HERE / runner_names[slug],
            "auditor": HERE / auditor_names[slug],
            "artifact_manifest": HERE / f"artifacts/{slug}/primary/artifact_manifest.json",
        }
        evidence: list[dict[str, Any]] = []
        for role, path in candidates.items():
            evidence.append(
                {
                    "role": role,
                    "path": str(path),
                    "available": path.is_file(),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
            )
        row["evidence_status"] = (
            "complete" if all(item["available"] for item in evidence) else "partially_missing"
        )
        row["evidence"] = evidence
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
    frozen_source_mapping = {
        "prior_fixed_clock_contract_sha256": "contract",
        "prior_fixed_clock_runner_sha256": "runner",
        "state_preprocessing_sha256": "state_preprocessing",
        "state_parameters_sha256": "state_parameters",
        "fixed_cycles_sha256": "fixed_cycles",
        "loop_path_parameters_sha256": "path_parameters",
        "movement_feature_manifest_sha256": "movement_manifest",
        "movement_parameters_sha256": "movement_parameters",
        "frozen_movement_core_sha256": "frozen_movement_core",
        "vti_provider_sha256": "provider_VTI",
    }
    source_pins = contract["frozen_sources"]
    drifted_pins = [
        key
        for key, source_name in frozen_source_mapping.items()
        if str(source_pins[key]) != str(sources[source_name]["sha256"])
    ]
    if drifted_pins:
        raise AssertionError(f"frozen source identity drifted: {drifted_pins}")
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
    sealed_feature_hash = seal_outcome_free_feature_ledger(
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
        sealed_feature_hash=sealed_feature_hash,
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


def _assign_chronology_stage(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
    session = pd.to_datetime(output["session"])
    stage = np.full(len(output), "outside_frozen_chronology", dtype=object)
    for name in (
        "development_context",
        "discovery",
        "validation",
        "final_opened_holdout",
    ):
        specification = contract["chronology"][name]
        mask = session.ge(pd.Timestamp(specification["start"])) & session.lt(
            pd.Timestamp(specification["end_exclusive"])
        )
        stage[mask] = name
    output["chronology_stage"] = stage
    if output["chronology_stage"].eq("outside_frozen_chronology").any():
        raise AssertionError("opportunities fall outside the frozen chronology")
    return output


def _stage(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame.loc[frame["chronology_stage"].eq(name)].copy()


def _open_scoring_stage(
    output_root: Path,
    features: pd.DataFrame,
    contract: dict[str, Any],
    stage_name: str,
) -> pd.DataFrame:
    """Open only one registered outcome stage and join its causal features."""

    specification = contract["chronology"][stage_name]
    outcomes = pd.read_parquet(
        output_root / "primary_economic_outcome_ledger.parquet",
        filters=[
            ("session", ">=", str(specification["start"])),
            ("session", "<", str(specification["end_exclusive"])),
        ],
    )
    stage_features = _stage(features, stage_name)
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
    joined = stage_features.merge(
        outcomes[outcome_columns],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
    )
    if joined["target"].isna().any() or len(joined) != len(stage_features):
        raise AssertionError(f"{stage_name} outcome join changed the frozen population")
    joined["long_net_bps"] = joined["net_long_return_bps"]
    joined["short_net_bps"] = joined["net_short_return_bps"]
    return joined.loc[joined["target"].ne("UNAVAILABLE")].copy()


def run_movement_permitted_surface(
    scored: pd.DataFrame,
    *,
    output_root: Path,
    contract: dict[str, Any],
    feature_families: dict[str, str],
    ordered_bins: dict[str, list[Any]],
    caps: SearchCaps,
    support_rules: SupportRules,
) -> dict[str, Any]:
    """Run a separately frozen search only where direction-neutral movement passes."""

    output_root.mkdir(parents=True, exist_ok=True)
    permitted = scored.loc[scored["movement_permission"].eq(True).fillna(False)].copy()
    discovery = _stage(permitted, "discovery")
    validation = _stage(permitted, "validation")
    final = _stage(permitted, "final_opened_holdout")
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
        seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 500,
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
    validation_metrics, survivors = validate_discovery_library(
        validation,
        discovery_library,
        support_rules=support_rules,
        holm_alpha=float(contract["multiplicity"]["retained_family_alpha"]),
        per_direction_cap=int(contract["search"]["validation_survivor_long_cap"]),
        bootstrap_draws=int(contract["multiplicity"]["session_block_bootstrap_draws"]),
        bootstrap_seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 500,
    )
    final_metrics = score_frozen_library(
        final,
        survivors,
        bootstrap_draws=int(contract["multiplicity"]["session_block_bootstrap_draws"]),
        bootstrap_seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 500,
    )
    write_deterministic_parquet(
        permitted,
        output_root / "movement_permitted_scoring_population.parquet",
        sort_by=["opportunity_id"],
    )
    _write_csv_allow_empty(
        census,
        output_root / "complete_candidate_registry.csv",
        sort_by=["signature_id"],
    )
    write_deterministic_json(
        _safe_json(discovery_library), output_root / "frozen_discovery_library.json"
    )
    _write_csv_allow_empty(
        validation_metrics,
        output_root / "validation_metrics.csv",
        sort_by=["signature_id"],
    )
    write_deterministic_json(
        _safe_json(survivors), output_root / "frozen_validation_survivors.json"
    )
    _write_csv_allow_empty(
        final_metrics,
        output_root / "final_opened_holdout_metrics.csv",
        sort_by=["signature_id"],
    )
    population = (
        permitted.groupby("chronology_stage", sort=True)
        .agg(
            rows=("opportunity_id", "size"),
            sessions=("session", "nunique"),
            stocks=("symbol", "nunique"),
        )
        .reset_index()
    )
    _write_csv_allow_empty(
        population,
        output_root / "population_summary.csv",
        sort_by=["chronology_stage"],
    )
    summary = {
        "surface": "movement_permitted_population",
        "movement_rule": contract["movement_permission"]["rule"],
        "rows": len(permitted),
        "discovery_rows": len(discovery),
        "candidate_signatures_examined": len(census),
        "discovery_eligible": int(census["discovery_eligible"].sum()) if len(census) else 0,
        "frozen_discovery_signatures": len(discovery_library),
        "validation_survivors": len(survivors),
        "final_scored_signatures": len(final_metrics),
        "all_rows_permission_pass": bool(permitted["movement_permission"].eq(True).all()),
    }
    write_deterministic_json(summary, output_root / "summary.json")
    return summary


def secondary_structural_population_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Predeclared attribution surfaces; never a selected primary population."""

    rows: list[dict[str, Any]] = []
    surfaces = {
        "named_loop_candidate": "top_parent_loop",
        "control_orientation": "top_loop_orientation",
        "loop_family": "parent_loop_family",
        "regime_state": "current_state",
    }
    for surface, column in surfaces.items():
        values = frame[column].astype(object).where(frame[column].notna(), "UNAVAILABLE")
        working = frame.assign(_surface_value=values)
        for (stage, value), group in working.groupby(
            ["chronology_stage", "_surface_value"], dropna=False, sort=True
        ):
            rows.append(
                {
                    "surface": surface,
                    "source_feature": column,
                    "chronology_stage": str(stage),
                    "value": str(value),
                    "rows": len(group),
                    "sessions": group["session"].nunique(),
                    "stocks": group["symbol"].nunique(),
                    "long_count": int(group["target"].eq("LONG").sum()),
                    "short_count": int(group["target"].eq("SHORT").sum()),
                    "neutral_count": int(group["target"].eq("NEUTRAL").sum()),
                    "long_rate": float(group["target"].eq("LONG").mean()),
                    "short_rate": float(group["target"].eq("SHORT").mean()),
                    "neutral_rate": float(group["target"].eq("NEUTRAL").mean()),
                    "mean_long_net_bps": float(group["long_net_bps"].mean()),
                    "mean_short_net_bps": float(group["short_net_bps"].mean()),
                    "selected_as_primary_population": False,
                }
            )
    return pd.DataFrame(rows)


def secondary_structural_signature_metrics(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
) -> pd.DataFrame:
    """Apply every frozen rule on every predeclared structural attribution cell."""

    rows: list[dict[str, Any]] = []
    surfaces = {
        "named_loop_candidate": "top_parent_loop",
        "control_orientation": "top_loop_orientation",
        "loop_family": "parent_loop_family",
        "regime_state": "current_state",
    }
    for surface, column in surfaces.items():
        values = frame[column].astype(object).where(frame[column].notna(), "UNAVAILABLE")
        working = frame.assign(_surface_value=values)
        for (stage, value), group in working.groupby(
            ["chronology_stage", "_surface_value"], dropna=False, sort=True
        ):
            for entry in library:
                signature = signature_from_dict(entry["signature"])
                metrics = signature_metrics(group, signature)
                rows.append(
                    {
                        "surface": surface,
                        "source_feature": column,
                        "chronology_stage": str(stage),
                        "value": str(value),
                        "signature_id": signature.signature_id,
                        "direction": signature.direction,
                        "rows": metrics["rows"],
                        "sessions": metrics["sessions"],
                        "stocks": metrics["stocks"],
                        "mean_directional_net_bps": metrics["mean_directional_net_bps"],
                        "directional_lift": metrics["directional_lift"],
                        "selected_as_primary_population": False,
                    }
                )
    return pd.DataFrame(rows)


def qualify_provisional_leads(
    library: list[dict[str, Any]],
    final_metrics: pd.DataFrame,
    stress: pd.DataFrame,
    calibration: pd.DataFrame,
    comparators: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply every post-validation success criterion without replacing rules."""

    final_by_id = final_metrics.set_index("signature_id") if len(final_metrics) else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    for entry in library:
        signature_id = str(entry["signature"]["signature_id"])
        reasons: list[str] = []
        if final_by_id.empty or signature_id not in final_by_id.index:
            reasons.append("not_scored_in_final_opened_holdout")
            rows.append(
                {
                    "signature_id": signature_id,
                    "direction": entry["signature"]["direction"],
                    "provisional_prospective_lead": False,
                    "rejection_reasons_json": json.dumps(reasons, separators=(",", ":")),
                }
            )
            continue
        final_row = cast(pd.Series, final_by_id.loc[signature_id])
        if float(final_row["mean_directional_net_bps"]) <= 0.0:
            reasons.append("final_payoff_not_positive")
        if float(final_row["directional_lift"]) <= 0.0:
            reasons.append("final_lift_not_positive")
        if float(final_row["twice_cost_mean_net_bps"]) <= 0.0:
            reasons.append("final_twice_cost_not_positive")
        if float(final_row["top_stock_absolute_contribution_share"]) > 0.25:
            reasons.append("final_stock_concentration")
        if float(final_row["maximum_single_stock_row_fraction"]) > 0.25:
            reasons.append("final_stock_row_concentration")
        if float(final_row["top_month_absolute_contribution_share"]) > 0.35:
            reasons.append("final_month_concentration")
        if float(final_row["positive_stock_fraction"]) <= 0.5:
            reasons.append("final_stock_consistency")
        if float(final_row["positive_month_fraction"]) <= 0.5:
            reasons.append("final_month_consistency")
        if int(final_row["rows"]) < 80:
            reasons.append("final_insufficient_rows")
        if int(final_row["sessions"]) < 30:
            reasons.append("final_insufficient_sessions")
        if int(final_row["stocks"]) < 8:
            reasons.append("final_insufficient_stocks")
        if int(final_row["months"]) < 3:
            reasons.append("final_insufficient_months")
        relevant_count = (
            int(final_row["long_count"])
            if entry["signature"]["direction"] == "LONG"
            else int(final_row["short_count"])
        )
        if relevant_count < 15:
            reasons.append("final_insufficient_directional_outcomes")
        required_positive = {
            "one_bar_execution_delay_same_terminal": "delay_not_positive",
            "remove_best_stock": "best_stock_removal_not_positive",
            "remove_top_five_stocks": "top_five_stock_removal_not_positive",
        }
        all_signature_stress = stress.loc[stress["signature_id"].eq(signature_id)]
        for stage in ("validation", "final_opened_holdout"):
            signature_stress = all_signature_stress.loc[
                all_signature_stress["chronology_stage"].eq(stage)
            ]
            for stress_name, reason in required_positive.items():
                values = signature_stress.loc[
                    signature_stress["stress"].eq(stress_name), "mean_directional_net_bps"
                ]
                if values.empty or not values.gt(0.0).all():
                    reasons.append(f"{stage}_{reason}")
            for episode_name in ("remove_best_episode", "remove_top_five_episodes"):
                episode = signature_stress.loc[signature_stress["stress"].eq(episode_name)]
                if episode.empty or not episode["status"].eq("available").all():
                    reasons.append(f"{stage}_{episode_name}_unavailable")
                elif not episode["mean_directional_net_bps"].gt(0.0).all():
                    reasons.append(f"{stage}_{episode_name}_not_positive")
            neighbours = signature_stress.loc[
                signature_stress["stress"].eq("adjacent_threshold_neighbour")
            ]
            if len(neighbours) and not neighbours["mean_directional_net_bps"].gt(0.0).all():
                reasons.append(f"{stage}_adjacent_threshold_incompatible")
        signature_calibration = calibration.loc[calibration["signature_id"].eq(signature_id)]
        if signature_calibration.empty or not signature_calibration["reasonably_calibrated"].all():
            reasons.append("probability_not_reasonably_calibrated")
        signature_comparators = comparators.loc[comparators["signature_id"].eq(signature_id)]
        if signature_comparators.empty or not (
            signature_comparators["stronger_than_momentum"].all()
            and signature_comparators["stronger_than_reversal"].all()
        ):
            reasons.append("not_stronger_than_momentum_and_reversal")
        reasons = sorted(set(reasons))
        row = {
            "signature_id": signature_id,
            "direction": entry["signature"]["direction"],
            "provisional_prospective_lead": not reasons,
            "rejection_reasons_json": json.dumps(reasons, separators=(",", ":")),
        }
        rows.append(row)
        if not reasons:
            qualified.append({**entry, "lead_qualification": row})
    return pd.DataFrame(rows), qualified


def scientific_decision_label(
    track_a: dict[str, Any],
    track_b: dict[str, Any] | None = None,
) -> str:
    """Derive one frozen label from strict, machine-readable survival fields."""

    long_count = int(track_a.get("provisional_prospective_lead_long", 0))
    short_count = int(track_a.get("provisional_prospective_lead_short", 0))
    if long_count and short_count:
        return "persistent_long_and_short_signatures_found_prospective_validation_required"
    if long_count:
        return "persistent_long_signatures_only"
    if short_count:
        return "persistent_short_signatures_only"
    if int(track_a.get("neutral_validation_and_final_strict_stable", 0)):
        return "neutral_veto_more_reliable_than_direction"
    if track_b is not None and (
        int(track_b.get("persistent_relative_final_signatures", 0))
        and bool(track_b.get("atlas_beats_relative_strength_validation_and_final"))
    ):
        return "relative_direction_more_predictable_than_absolute"
    movement = cast(dict[str, Any], track_a.get("movement_permitted_surface", {}))
    if int(movement.get("validation_survivors", 0)) and int(
        movement.get("positive_final_scored_signatures", 0)
    ):
        return "movement_permission_useful_direction_unresolved"
    validation_survivors = int(track_a.get("validation_survivor_long", 0)) + int(
        track_a.get("validation_survivor_short", 0)
    )
    if validation_survivors:
        return "signature_effects_concentrated_or_unstable"
    if int(track_a.get("frozen_discovery_long", 0)) + int(
        track_a.get("frozen_discovery_short", 0)
    ):
        return "discovery_signatures_failed_validation"
    return "no_persistent_directional_signatures"


def score_experiment(output_root: Path) -> dict[str, Any]:
    """Run bounded Track A discovery, unchanged chronology, controller, and baselines."""

    contract = load_contract(CONTRACT_PATH)
    schema = _feature_schema()
    feature_ledger = pd.read_parquet(output_root / "outcome_free_feature_ledger.parquet")
    feature_ledger = _assign_chronology_stage(feature_ledger, contract)
    discovery = _open_scoring_stage(output_root, feature_ledger, contract, "discovery")
    if discovery.empty:
        raise AssertionError("chronological discovery stage is incomplete")
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
    search_space = candidate_search_space_counts(discovery, feature_families)
    write_deterministic_json(
        {
            **search_space,
            "balanced_directional_candidate_allocation": {
                "univariate": int(census["stage"].eq("univariate").sum()),
                "pairwise": int(census["stage"].eq("pairwise").sum()),
            },
            "broad_examined_directional_candidates": int(
                census["stage"].isin(["univariate", "pairwise"]).sum()
            ),
            "broad_directional_cap": int(search["univariate_and_pairwise_cap"]),
            "enumerated_but_not_examined_due_to_cap": int(
                search_space["observed_univariate_directional_candidates"]
                + search_space["observed_pairwise_directional_candidates"]
                - census["stage"].isin(["univariate", "pairwise"]).sum()
            ),
            "selection": "outcome_free_feature_round_robin_with_hash_ordered_feature_pairs",
        },
        output_root / "candidate_search_space.json",
    )
    neutral_census, neutral_discovery_library = evaluate_neutral_veto_census(
        discovery,
        candidates,
        support_rules=support_rules,
        fdr_q=float(contract["multiplicity"]["broad_discovery_q"]),
        ordered_bins=ordered_bins,
        cap=5,
    )
    discovery_library = freeze_discovery_library(
        census,
        candidates,
        discovery,
        retained_stage_cap=int(search["discovery_stage_retained_cap"]),
        per_direction_cap=int(search["frozen_discovery_long_cap"]),
    )
    discovery_library_path = output_root / "frozen_discovery_signature_library.json"
    neutral_discovery_path = output_root / "frozen_neutral_discovery_library.json"
    write_deterministic_json(_safe_json(discovery_library), discovery_library_path)
    write_deterministic_json(_safe_json(neutral_discovery_library), neutral_discovery_path)
    discovery_library_hash = sha256_file(discovery_library_path)
    neutral_discovery_hash = sha256_file(neutral_discovery_path)
    write_deterministic_json(
        {
            "stage": "discovery",
            "outcome_start": contract["chronology"]["discovery"]["start"],
            "outcome_end_exclusive": contract["chronology"]["discovery"]["end_exclusive"],
            "frozen_discovery_library_sha256": discovery_library_hash,
            "frozen_neutral_discovery_library_sha256": neutral_discovery_hash,
            "validation_or_final_opened_before_seal": False,
        },
        output_root / "discovery_stage_seal.json",
    )
    movement_root = output_root / "movement_permitted"
    movement_root.mkdir(parents=True, exist_ok=True)
    movement_discovery = discovery.loc[
        discovery["movement_permission"].eq(True).fillna(False)
    ].copy()
    movement_feature_families = {
        feature: family
        for feature, family in feature_families.items()
        if feature != "movement_permission"
        and feature in movement_discovery
        and movement_discovery[feature].dropna().nunique() > 1
    }
    movement_candidates, movement_registry = generate_bounded_candidates(
        movement_discovery,
        movement_feature_families,
        caps,
        minimum_parent_support=support_rules.minimum_rows,
    )
    movement_trees, movement_tree_registry = extract_shallow_tree_candidates(
        movement_discovery,
        movement_feature_families,
        maximum_depth=int(search["tree_maximum_depth"]),
        minimum_leaf_rows=int(search["tree_minimum_leaf_rows"]),
        cap=int(search["tree_candidate_cap"]),
        seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 500,
    )
    movement_by_id = {
        candidate.signature_id: candidate for candidate in [*movement_candidates, *movement_trees]
    }
    movement_candidates = [movement_by_id[key] for key in sorted(movement_by_id)]
    movement_registry_by_id = {
        str(row["signature_id"]): row for row in [*movement_registry, *movement_tree_registry]
    }
    movement_registry = [movement_registry_by_id[key] for key in sorted(movement_registry_by_id)]
    movement_census = evaluate_candidate_census(
        movement_discovery,
        movement_candidates,
        movement_registry,
        support_rules=support_rules,
        ordered_bins=ordered_bins,
        fdr_q=float(contract["multiplicity"]["broad_discovery_q"]),
    )
    movement_search_space = candidate_search_space_counts(
        movement_discovery, movement_feature_families
    )
    write_deterministic_json(
        {
            **movement_search_space,
            "broad_examined_directional_candidates": int(
                movement_census["stage"].isin(["univariate", "pairwise"]).sum()
            ),
            "broad_directional_cap": int(search["univariate_and_pairwise_cap"]),
            "selection": "outcome_free_balanced_allocation_gate_feature_excluded",
        },
        movement_root / "candidate_search_space.json",
    )
    movement_discovery_library = freeze_discovery_library(
        movement_census,
        movement_candidates,
        movement_discovery,
        retained_stage_cap=int(search["discovery_stage_retained_cap"]),
        per_direction_cap=int(search["frozen_discovery_long_cap"]),
    )
    movement_discovery_path = movement_root / "frozen_discovery_library.json"
    write_deterministic_json(_safe_json(movement_discovery_library), movement_discovery_path)
    movement_discovery_hash = sha256_file(movement_discovery_path)
    write_deterministic_json(
        {
            "stage": "discovery",
            "frozen_library_sha256": movement_discovery_hash,
            "validation_or_final_opened_before_seal": False,
        },
        movement_root / "discovery_stage_seal.json",
    )
    validation = _open_scoring_stage(output_root, feature_ledger, contract, "validation")
    if validation.empty:
        raise AssertionError("chronological validation stage is incomplete")
    neutral_validation_metrics, neutral_survivor_library = validate_neutral_veto_library(
        validation,
        neutral_discovery_library,
        support_rules=support_rules,
        holm_alpha=float(contract["multiplicity"]["retained_family_alpha"]),
    )
    validation_metrics, survivor_library = validate_discovery_library(
        validation,
        discovery_library,
        support_rules=support_rules,
        holm_alpha=float(contract["multiplicity"]["retained_family_alpha"]),
        per_direction_cap=int(search["validation_survivor_long_cap"]),
        bootstrap_draws=int(contract["multiplicity"]["session_block_bootstrap_draws"]),
        bootstrap_seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]),
    )
    survivor_path = output_root / "frozen_validation_survivor_library.json"
    neutral_survivor_path = output_root / "neutral_veto_library.json"
    write_deterministic_json(_safe_json(survivor_library), survivor_path)
    write_deterministic_json(_safe_json(neutral_survivor_library), neutral_survivor_path)
    survivor_hash = sha256_file(survivor_path)
    neutral_survivor_hash = sha256_file(neutral_survivor_path)
    write_deterministic_json(
        {
            "stage": "validation",
            "outcome_start": contract["chronology"]["validation"]["start"],
            "outcome_end_exclusive": contract["chronology"]["validation"]["end_exclusive"],
            "frozen_validation_survivor_library_sha256": survivor_hash,
            "frozen_neutral_survivor_library_sha256": neutral_survivor_hash,
            "final_opened_before_seal": False,
        },
        output_root / "validation_stage_seal.json",
    )
    movement_validation = validation.loc[
        validation["movement_permission"].eq(True).fillna(False)
    ].copy()
    movement_validation_metrics, movement_survivors = validate_discovery_library(
        movement_validation,
        movement_discovery_library,
        support_rules=support_rules,
        holm_alpha=float(contract["multiplicity"]["retained_family_alpha"]),
        per_direction_cap=int(search["validation_survivor_long_cap"]),
        bootstrap_draws=int(contract["multiplicity"]["session_block_bootstrap_draws"]),
        bootstrap_seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 500,
    )
    movement_survivor_path = movement_root / "frozen_validation_survivors.json"
    write_deterministic_json(_safe_json(movement_survivors), movement_survivor_path)
    movement_survivor_hash = sha256_file(movement_survivor_path)
    write_deterministic_json(
        {
            "stage": "validation",
            "frozen_survivor_library_sha256": movement_survivor_hash,
            "final_opened_before_seal": False,
        },
        movement_root / "validation_stage_seal.json",
    )
    final = _open_scoring_stage(output_root, feature_ledger, contract, "final_opened_holdout")
    if final.empty:
        raise AssertionError("chronological final opened-holdout stage is incomplete")
    if sha256_file(discovery_library_path) != discovery_library_hash:
        raise AssertionError("discovery library changed after validation was opened")
    if sha256_file(survivor_path) != survivor_hash:
        raise AssertionError("validation library changed after final was opened")
    neutral_final_metrics, neutral_final_survivor_library = validate_neutral_veto_library(
        final,
        neutral_survivor_library,
        support_rules=support_rules,
        holm_alpha=float(contract["multiplicity"]["retained_family_alpha"]),
    )
    final_metrics = score_frozen_library(
        final,
        survivor_library,
        bootstrap_draws=int(contract["multiplicity"]["session_block_bootstrap_draws"]),
        bootstrap_seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]),
    )
    movement_final = final.loc[final["movement_permission"].eq(True).fillna(False)].copy()
    movement_final_metrics = score_frozen_library(
        movement_final,
        movement_survivors,
        bootstrap_draws=int(contract["multiplicity"]["session_block_bootstrap_draws"]),
        bootstrap_seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 500,
    )
    if sha256_file(movement_discovery_path) != movement_discovery_hash:
        raise AssertionError("movement discovery library changed after later stages opened")
    if sha256_file(movement_survivor_path) != movement_survivor_hash:
        raise AssertionError("movement validation library changed after final opened")
    # Only after both library seals exist may reporting/baseline code open all
    # retrospective outcomes together.
    joined, outcomes = _joined_scoring_frame(output_root)
    joined = _assign_chronology_stage(joined, contract)
    scored = joined.loc[joined["target"].ne("UNAVAILABLE")].copy()
    for name, opened in (
        ("discovery", discovery),
        ("validation", validation),
        ("final_opened_holdout", final),
    ):
        expected_ids = set(_stage(scored, name)["opportunity_id"].astype(str))
        if expected_ids != set(opened["opportunity_id"].astype(str)):
            raise AssertionError(f"{name} stage identity changed after all outcomes opened")
    discovery_probabilities = {
        label: float((discovery["target"].eq(label).sum() + 1.0) / (len(discovery) + 3.0))
        for label in CLASSES
    }
    atlas_decisions = apply_atlas_controller(
        joined,
        survivor_library,
        base_probabilities=discovery_probabilities,
    )
    first_touch = pd.read_parquet(output_root / "secondary_first_touch_outcome_ledger.parquet")
    baselines = baseline_predictions(discovery, joined, first_touch=first_touch)
    all_predictions = pd.concat([baselines, atlas_decisions], ignore_index=True)
    metric_outcomes = joined[
        [
            "opportunity_id",
            "target",
            "long_net_bps",
            "short_net_bps",
            "round_trip_cost_bps",
        ]
    ]
    predictive, economic = prediction_metrics(all_predictions, metric_outcomes)
    individual_concentration = signature_breakdowns(scored, discovery_library)
    atlas_concentration_rows = atlas_concentration(scored, atlas_decisions)
    breakdowns = pd.concat([individual_concentration, atlas_concentration_rows], ignore_index=True)
    attribution_rows = signature_attribution_rows(scored, discovery_library)
    delayed_outcomes = pd.read_parquet(output_root / "one_bar_delay_outcome_ledger.parquet")
    stress_results, leave_one_out = stress_signature_library(
        scored,
        discovery_library,
        delayed_outcomes,
        ordered_bins=ordered_bins,
        causal_population=joined,
    )
    neutral_stress_results = stress_neutral_veto_library(
        scored,
        neutral_survivor_library,
        delayed_outcomes,
        ordered_bins=ordered_bins,
    )
    probability_metrics = pd.concat(
        [
            signature_probability_metrics(validation, survivor_library),
            signature_probability_metrics(final, survivor_library),
        ],
        ignore_index=True,
    )
    comparator_metrics = pd.concat(
        [
            paired_simple_baseline_metrics(validation, survivor_library),
            paired_simple_baseline_metrics(final, survivor_library),
        ],
        ignore_index=True,
    )
    lead_qualification, prospective_leads = qualify_provisional_leads(
        survivor_library,
        final_metrics,
        stress_results,
        probability_metrics,
        comparator_metrics,
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
    if sha256_file(neutral_survivor_path) != neutral_survivor_hash:
        raise AssertionError("frozen neutral survivor library changed after final open")
    if sha256_file(discovery_library_path) != discovery_library_hash:
        raise AssertionError("frozen discovery library changed after final open")
    _write_csv_allow_empty(
        validation_metrics,
        output_root / "validation_signature_metrics.csv",
        sort_by=["direction", "discovery_score", "signature_id"],
    )
    if sha256_file(survivor_path) != survivor_hash:
        raise AssertionError("frozen validation library changed during final reporting")
    _write_csv_allow_empty(
        final_metrics,
        output_root / "final_opened_holdout_signature_metrics.csv",
        sort_by=["direction", "discovery_score", "signature_id"],
    )
    _write_csv_allow_empty(
        probability_metrics,
        output_root / "individual_signature_calibration_metrics.csv",
        sort_by=["signature_id", "chronology_stage"],
    )
    _write_csv_allow_empty(
        comparator_metrics,
        output_root / "individual_signature_baseline_comparison.csv",
        sort_by=["signature_id", "chronology_stage"],
    )
    _write_csv_allow_empty(
        lead_qualification,
        output_root / "provisional_prospective_lead_qualification.csv",
        sort_by=["direction", "signature_id"],
    )
    write_deterministic_json(
        _safe_json(prospective_leads),
        output_root / "provisional_prospective_lead_library.json",
    )
    for direction, filename in (
        ("LONG", "long_signature_library.json"),
        ("SHORT", "short_signature_library.json"),
    ):
        write_deterministic_json(
            _safe_json(
                [
                    entry
                    for entry in prospective_leads
                    if entry["signature"]["direction"] == direction
                ]
            ),
            output_root / filename,
        )
    long_library = [
        entry for entry in prospective_leads if entry["signature"]["direction"] == "LONG"
    ]
    short_library = [
        entry for entry in prospective_leads if entry["signature"]["direction"] == "SHORT"
    ]
    metadata_path = output_root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "long_library_sha256": canonical_library_hash(long_library),
            "short_library_sha256": canonical_library_hash(short_library),
            "neutral_library_sha256": canonical_library_hash(neutral_survivor_library),
            "causal_feature_names": sorted(feature_families),
            "prospective_lead_count": len(prospective_leads),
        }
    )
    write_deterministic_json(metadata, metadata_path)
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
        sort_by=["signature_id", "dimension"],
    )
    _write_csv_allow_empty(
        attribution_rows,
        output_root / "signature_attribution_breakdowns.csv",
        sort_by=["signature_id", "chronology_stage", "dimension", "value"],
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
        neutral_stress_results,
        output_root / "neutral_veto_stability_results.csv",
        sort_by=["neutral_veto_id", "chronology_stage", "stress", "removed"],
    )
    _write_csv_allow_empty(
        nulls,
        output_root / "null_test_results.csv",
        sort_by=["null"],
    )
    structural_metrics = secondary_structural_population_metrics(scored)
    structural_signature_metrics = secondary_structural_signature_metrics(scored, discovery_library)
    _write_csv_allow_empty(
        structural_metrics,
        output_root / "secondary_structural_population_metrics.csv",
        sort_by=["surface", "chronology_stage", "value"],
    )
    _write_csv_allow_empty(
        structural_signature_metrics,
        output_root / "secondary_structural_signature_metrics.csv",
        sort_by=["surface", "chronology_stage", "value", "signature_id"],
    )
    permitted_scored = scored.loc[scored["movement_permission"].eq(True).fillna(False)].copy()
    write_deterministic_parquet(
        permitted_scored,
        movement_root / "movement_permitted_scoring_population.parquet",
        sort_by=["opportunity_id"],
    )
    _write_csv_allow_empty(
        movement_census,
        movement_root / "complete_candidate_registry.csv",
        sort_by=["signature_id"],
    )
    _write_csv_allow_empty(
        movement_validation_metrics,
        movement_root / "validation_metrics.csv",
        sort_by=["signature_id"],
    )
    _write_csv_allow_empty(
        movement_final_metrics,
        movement_root / "final_opened_holdout_metrics.csv",
        sort_by=["signature_id"],
    )
    movement_population = (
        permitted_scored.groupby("chronology_stage", sort=True)
        .agg(
            rows=("opportunity_id", "size"),
            sessions=("session", "nunique"),
            stocks=("symbol", "nunique"),
        )
        .reset_index()
    )
    _write_csv_allow_empty(
        movement_population,
        movement_root / "population_summary.csv",
        sort_by=["chronology_stage"],
    )
    movement_surface_summary = {
        "surface": "movement_permitted_population",
        "movement_rule": contract["movement_permission"]["rule"],
        "rows": len(permitted_scored),
        "discovery_rows": len(movement_discovery),
        "candidate_signatures_examined": len(movement_census),
        "discovery_eligible": int(movement_census["discovery_eligible"].sum())
        if len(movement_census)
        else 0,
        "frozen_discovery_signatures": len(movement_discovery_library),
        "validation_survivors": len(movement_survivors),
        "final_scored_signatures": len(movement_final_metrics),
        "positive_final_scored_signatures": int(
            (
                movement_final_metrics.get(
                    "mean_directional_net_bps", pd.Series(dtype=float)
                ).gt(0.0)
                & movement_final_metrics.get(
                    "directional_lift", pd.Series(dtype=float)
                ).gt(0.0)
                & movement_final_metrics.get(
                    "twice_cost_mean_net_bps", pd.Series(dtype=float)
                ).gt(0.0)
            ).sum()
        ),
        "all_rows_permission_pass": bool(permitted_scored["movement_permission"].eq(True).all()),
        "stage_seals_verified": True,
    }
    write_deterministic_json(movement_surface_summary, movement_root / "summary.json")
    base_rates = (
        scored.groupby("chronology_stage", sort=True)["target"]
        .value_counts(normalize=True)
        .rename("rate")
        .reset_index()
    )
    _write_csv_allow_empty(
        base_rates,
        output_root / "base_rates_by_period.csv",
        sort_by=["chronology_stage", "target"],
    )
    neutral_final_survivor_ids = {
        str(entry["neutral_veto_id"]) for entry in neutral_final_survivor_library
    }
    neutral_strict_stable_ids: set[str] = set()
    for veto_id in neutral_final_survivor_ids:
        stable = True
        veto_stress = neutral_stress_results.loc[
            neutral_stress_results["neutral_veto_id"].eq(veto_id)
            & neutral_stress_results["chronology_stage"].isin(
                ["validation", "final_opened_holdout"]
            )
        ]
        for stage in ("validation", "final_opened_holdout"):
            stage_stress = veto_stress.loc[veto_stress["chronology_stage"].eq(stage)]
            for stress_name in (
                "one_bar_execution_delay_same_terminal",
                "remove_best_stock",
                "remove_top_five_stocks",
            ):
                rows = stage_stress.loc[stage_stress["stress"].eq(stress_name)]
                stable &= bool(len(rows) and rows["neutral_lift"].gt(0.0).all())
            neighbours = stage_stress.loc[
                stage_stress["stress"].eq("adjacent_threshold_neighbour")
            ]
            stable &= bool(neighbours.empty or neighbours["neutral_lift"].gt(0.0).all())
            twice = stage_stress.loc[stage_stress["stress"].eq("twice_cost")]
            stable &= bool(
                len(twice)
                and twice["mean_long_net_bps"].lt(0.0).all()
                and twice["mean_short_net_bps"].lt(0.0).all()
            )
        if stable:
            neutral_strict_stable_ids.add(veto_id)
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
        "provisional_prospective_lead_long": sum(
            entry["signature"]["direction"] == "LONG" for entry in prospective_leads
        ),
        "provisional_prospective_lead_short": sum(
            entry["signature"]["direction"] == "SHORT" for entry in prospective_leads
        ),
        "neutral_discovery_survivors": len(neutral_discovery_library),
        "neutral_validation_survivors": len(neutral_survivor_library),
        "neutral_final_same_sign_descriptive": int(
            neutral_final_metrics.get("neutral_lift", pd.Series(dtype=float)).gt(0.0).sum()
        ),
        "neutral_final_strict_survivors": len(neutral_final_survivor_ids),
        "neutral_validation_and_final_strict_stable": len(neutral_strict_stable_ids),
        "atlas_directional_outputs_validation": int(
            atlas_decisions.loc[
                atlas_decisions["chronology_stage"].eq("validation"), "predicted_state"
            ]
            .isin(["LONG", "SHORT"])
            .sum()
        ),
        "atlas_directional_outputs_final": int(
            atlas_decisions.loc[
                atlas_decisions["chronology_stage"].eq("final_opened_holdout"),
                "predicted_state",
            ]
            .isin(["LONG", "SHORT"])
            .sum()
        ),
        "movement_permitted_surface": movement_surface_summary,
        "research_only": True,
        "execution_enabled": False,
    }
    summary["track_a_scientific_decision"] = scientific_decision_label(summary)
    write_deterministic_json(summary, output_root / "track_a_summary.json")
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
    pivot = base_rates.pivot(index="chronology_stage", columns="target", values="rate").fillna(0.0)
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
        movement.loc[movement["movement_permission"].notna()]
        .groupby(["chronology_stage", "movement_permission"], sort=True)[
            "absolute_terminal_move_bps"
        ]
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
        economic["model_id"].isin(selected_models)
        & economic["chronology_stage"].isin(["discovery", "validation", "final_opened_holdout"])
    ]
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    if not comparison.empty:
        comparison.pivot(
            index="model_id",
            columns="chronology_stage",
            values="net_bps_per_full_opportunity",
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

    relative_path = output_root / "track_b/relative_outcome_ledger.parquet"
    if relative_path.is_file():
        relative = pd.read_parquet(relative_path)
        relative_rates = (
            relative.groupby("chronology_stage", sort=True)["target"]
            .value_counts(normalize=True)
            .rename("rate")
            .reset_index()
            .pivot(index="chronology_stage", columns="target", values="rate")
            .fillna(0.0)
        )
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        relative_rates.plot.bar(
            ax=axis,
            color=[colors.get(str(label), "#777777") for label in relative_rates.columns],
        )
        axis.set(title="Track B relative target rates", xlabel="Opened stage", ylabel="Rate")
        save(fig, "track_b_relative_target_rates.png")


def write_prospective_schemas(output_root: Path) -> None:
    forecast_fields = [
        "run_id",
        "git_sha",
        "contract_hash",
        "data_snapshot_hash",
        "training_data_snapshot_hash",
        "feature_schema_hash",
        "long_library_hash",
        "short_library_hash",
        "neutral_library_hash",
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
        "settlement_status",
        "unavailable_reason",
        "research_only",
        "execution_enabled",
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


def generate_track_b_plot(output_root: Path) -> None:
    """Generate the secondary plot only after the audited Track B gate opens."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relative = pd.read_parquet(output_root / "track_b/relative_outcome_ledger.parquet")
    relative_rates = (
        relative.groupby("chronology_stage", sort=True)["target"]
        .value_counts(normalize=True)
        .rename("rate")
        .reset_index()
        .pivot(index="chronology_stage", columns="target", values="rate")
        .fillna(0.0)
    )
    colors = {"LONG": "#2878B5", "SHORT": "#C44E52", "NEUTRAL": "#8A8A8A"}
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    relative_rates.plot.bar(
        ax=axis,
        color=[colors.get(str(label), "#777777") for label in relative_rates.columns],
    )
    axis.set(title="Track B relative target rates", xlabel="Opened stage", ylabel="Rate")
    figure.tight_layout()
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        plot_root / "track_b_relative_target_rates.png",
        dpi=140,
        metadata={"Software": "Stocker Directional Signature Atlas V1"},
    )
    plt.close(figure)


def write_artifact_manifest(output_root: Path) -> None:
    excluded = {
        "artifact_manifest.json",
        "independent_audit.json",
        "prospective_forecast_dry_run.json",
        "prospective_settlement_dry_run.json",
        "track_a_independent_audit.json",
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
    contract = load_contract(CONTRACT_PATH)
    long_library = json.loads((output_root / "long_signature_library.json").read_text())
    short_library = json.loads((output_root / "short_signature_library.json").read_text())
    neutral_library = json.loads((output_root / "neutral_veto_library.json").read_text())
    causal_feature_names = list(map(str, metadata["causal_feature_names"]))
    expected_identity = {
        "run_id": str(metadata["run_id"]),
        "git_sha": str(metadata["git_sha"]),
        "contract_hash": str(metadata["contract_sha256"]),
        "training_data_snapshot_hash": str(metadata["data_snapshot_sha256"]),
        "feature_schema_hash": str(metadata["feature_schema_sha256"]),
        "long_library_hash": str(metadata["long_library_sha256"]),
        "short_library_hash": str(metadata["short_library_sha256"]),
        "neutral_library_hash": str(metadata["neutral_library_sha256"]),
    }
    completion = {
        key: int(value)
        for key, value in contract["prospective_completion"].items()
        if key.startswith("minimum_")
    }
    ledger = ProspectiveLedger(
        prospective_root,
        opened_through=str(contract["prospective_completion"]["opened_historical_through"]),
        required_causal_feature_names=causal_feature_names,
        expected_identity=expected_identity,
        completion_requirements=completion,
    )
    opportunity_id = "atlas|2026|DRYRUN|2026-07-20|12"
    decision_timestamp = "2026-07-20T14:30:00+00:00"
    feature_row: dict[str, Any] = {
        "opportunity_id": opportunity_id,
        "symbol": "DRYRUN",
        "session": "2026-07-20",
        "decision_clock": "clock_12",
        "decision_timestamp": decision_timestamp,
        "movement_permission": False,
    }
    for feature in causal_feature_names:
        feature_row.setdefault(feature, None)
        feature_row[f"{feature}__available_at"] = None
    for feature in ("decision_clock", "movement_permission"):
        if feature in causal_feature_names:
            feature_row[f"{feature}__available_at"] = decision_timestamp
    record = build_forecast_record(
        feature_row,
        metadata=metadata,
        long_library=long_library,
        short_library=short_library,
        neutral_library=neutral_library,
        causal_feature_names=causal_feature_names,
        forecast_input_snapshot_hash=hashlib.sha256(
            json.dumps(feature_row, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        forecast_freeze_timestamp=decision_timestamp,
    )
    record["reason_codes"] = sorted([*record["reason_codes"], "dry_run"])
    ledger.append_forecast(record)
    result = {
        "opportunity_id": opportunity_id,
        "forecast_path": str(ledger.forecast_path),
        "forecast_record_count": 1,
        "completion_status": ledger.completion_status(),
        "research_only": True,
        "execution_enabled": False,
    }
    write_deterministic_json(result, output_root / "prospective_forecast_dry_run.json")
    return result


def prospective_settlement_dry_run(output_root: Path, prospective_root: Path) -> dict[str, Any]:
    contract = load_contract(CONTRACT_PATH)
    metadata = json.loads((output_root / "run_metadata.json").read_text(encoding="utf-8"))
    expected_identity = {
        "run_id": str(metadata["run_id"]),
        "git_sha": str(metadata["git_sha"]),
        "contract_hash": str(metadata["contract_sha256"]),
        "training_data_snapshot_hash": str(metadata["data_snapshot_sha256"]),
        "feature_schema_hash": str(metadata["feature_schema_sha256"]),
        "long_library_hash": str(metadata["long_library_sha256"]),
        "short_library_hash": str(metadata["short_library_sha256"]),
        "neutral_library_hash": str(metadata["neutral_library_sha256"]),
    }
    completion = {
        key: int(value)
        for key, value in contract["prospective_completion"].items()
        if key.startswith("minimum_")
    }
    ledger = ProspectiveLedger(
        prospective_root,
        opened_through=str(contract["prospective_completion"]["opened_historical_through"]),
        required_causal_feature_names=list(map(str, metadata["causal_feature_names"])),
        expected_identity=expected_identity,
        completion_requirements=completion,
    )
    opportunity_id = "atlas|2026|DRYRUN|2026-07-20|12"
    ledger.append_settlement(
        build_settlement_record(
            {
                "opportunity_id": opportunity_id,
                "terminal_timestamp": "2026-07-20T16:30:00+00:00",
                "gross_long_return_bps": 0.0,
                "gross_short_return_bps": 0.0,
                "round_trip_cost_bps": 10.0,
                "net_long_return_bps": -10.0,
                "net_short_return_bps": -10.0,
                "target": "NEUTRAL",
                "first_touch_target": "NEITHER",
            },
            settlement_timestamp="2026-07-20T16:31:00+00:00",
            settlement_code_version=_git_sha(),
        )
    )
    result = {
        "opportunity_id": opportunity_id,
        "forecast_path_unchanged": True,
        "settlement_path": str(ledger.settlement_path),
        "settlement_record_count": 1,
        "completion_status": ledger.completion_status(),
        "economic_read_gate_blocked": False,
        "research_only": True,
        "execution_enabled": False,
    }
    try:
        ledger.read_settlements()
    except PermissionError:
        result["economic_read_gate_blocked"] = True
    write_deterministic_json(result, output_root / "prospective_settlement_dry_run.json")
    return result


def qualify_relative_persistence(
    library: list[dict[str, Any]],
    validation_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    stress: pd.DataFrame,
    nulls: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    """Apply the frozen absolute-signature stability standard to Track B."""

    support = contract["support"]
    validation = (
        validation_metrics.set_index("signature_id")
        if not validation_metrics.empty
        else pd.DataFrame()
    )
    final = final_metrics.set_index("signature_id") if not final_metrics.empty else pd.DataFrame()
    null_failed = bool(
        not nulls.empty
        and nulls["similar_persistent_validation_performance"].fillna(False).astype(bool).any()
    )
    rows: list[dict[str, Any]] = []
    for entry in library:
        signature_id = str(entry["signature"]["signature_id"])
        direction = str(entry["signature"]["direction"])
        reasons: list[str] = []
        for stage, metrics in (
            ("validation", validation),
            ("final_opened_holdout", final),
        ):
            if metrics.empty or signature_id not in metrics.index:
                reasons.append(f"{stage}_metrics_missing")
                continue
            metric = cast(pd.Series, metrics.loc[signature_id])
            if float(metric["mean_directional_net_bps"]) <= 0.0:
                reasons.append(f"{stage}_payoff_not_positive")
            if float(metric["directional_lift"]) <= 0.0:
                reasons.append(f"{stage}_lift_not_positive")
            if float(metric["twice_cost_mean_net_bps"]) <= 0.0:
                reasons.append(f"{stage}_twice_cost_not_positive")
            if int(metric["rows"]) < int(support["minimum_rows"]):
                reasons.append(f"{stage}_insufficient_rows")
            if int(metric["sessions"]) < int(support["minimum_independent_sessions"]):
                reasons.append(f"{stage}_insufficient_sessions")
            if int(metric["stocks"]) < int(support["minimum_independent_stocks"]):
                reasons.append(f"{stage}_insufficient_stocks")
            if int(metric["months"]) < int(support["minimum_calendar_months"]):
                reasons.append(f"{stage}_insufficient_months")
            relevant_count = int(
                metric["long_count"] if direction == "LONG" else metric["short_count"]
            )
            if relevant_count < int(support["minimum_relevant_direction_outcomes"]):
                reasons.append(f"{stage}_insufficient_directional_outcomes")
            if float(metric["maximum_single_stock_row_fraction"]) > float(
                support["maximum_single_stock_row_fraction"]
            ):
                reasons.append(f"{stage}_stock_row_concentration")
            if float(metric["top_stock_absolute_contribution_share"]) > float(
                support["maximum_top_stock_absolute_payoff_share"]
            ):
                reasons.append(f"{stage}_stock_payoff_concentration")
            if float(metric["top_month_absolute_contribution_share"]) > float(
                support["maximum_top_month_absolute_payoff_share"]
            ):
                reasons.append(f"{stage}_month_payoff_concentration")
            if float(metric["positive_stock_fraction"]) <= float(
                support["minimum_positive_stock_fraction"]
            ):
                reasons.append(f"{stage}_stock_consistency")
            if float(metric["positive_month_fraction"]) <= float(
                contract["validation_survival"][
                    "positive_month_fraction_strictly_greater_than"
                ]
            ):
                reasons.append(f"{stage}_month_consistency")
            if float(metric["opposite_direction_excess"]) > float(
                support["maximum_opposite_direction_excess"]
            ):
                reasons.append(f"{stage}_opposite_direction_not_controlled")

            stage_stress = stress.loc[
                stress["signature_id"].eq(signature_id)
                & stress["chronology_stage"].eq(stage)
            ]
            for stress_name in (
                "one_bar_execution_delay_same_terminal",
                "remove_best_stock",
                "remove_top_five_stocks",
                "remove_best_month",
            ):
                stress_rows = stage_stress.loc[stage_stress["stress"].eq(stress_name)]
                if stress_rows.empty or not stress_rows["mean_directional_net_bps"].gt(0.0).all():
                    reasons.append(f"{stage}_{stress_name}_not_positive")
            neighbours = stage_stress.loc[
                stage_stress["stress"].eq("adjacent_threshold_neighbour")
            ]
            if len(neighbours) and not neighbours["mean_directional_net_bps"].gt(0.0).all():
                reasons.append(f"{stage}_adjacent_threshold_incompatible")
            for episode_name in ("remove_best_episode", "remove_top_five_episodes"):
                episode = stage_stress.loc[stage_stress["stress"].eq(episode_name)]
                if episode.empty or not episode["status"].eq("available").all():
                    reasons.append(f"{stage}_{episode_name}_unavailable")
                elif not episode["mean_directional_net_bps"].gt(0.0).all():
                    reasons.append(f"{stage}_{episode_name}_not_positive")
        if null_failed:
            reasons.append("null_family_similar_persistent_validation_performance")
        reasons = sorted(set(reasons))
        rows.append(
            {
                "signature_id": signature_id,
                "direction": direction,
                "strict_persistent_relative_signature": not reasons,
                "rejection_reasons_json": json.dumps(reasons, separators=(",", ":")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "signature_id",
            "direction",
            "strict_persistent_relative_signature",
            "rejection_reasons_json",
        ],
    )


def run_track_b(
    absolute_features: pd.DataFrame,
    *,
    historical_root: Path,
    output_root: Path,
    contract: dict[str, Any],
    feature_families: dict[str, str],
    ordered_bins: dict[str, list[Any]],
    caps: SearchCaps,
    support_rules: SupportRules,
) -> dict[str, Any]:
    """Run Track B only after Track A artifacts and conclusion are materialized."""

    output_root.mkdir(parents=True, exist_ok=True)
    minimum_peers = int(contract["track_b"]["minimum_contemporaneous_peers"])

    def construct_stage(absolute: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        outcomes = construct_relative_outcomes(absolute, minimum_peers=minimum_peers)
        relative_stage = absolute.drop(
            columns=["target", "long_net_bps", "short_net_bps", "round_trip_cost_bps"]
        ).merge(
            outcomes[
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
        return outcomes, relative_stage.loc[
            relative_stage["target"].ne("UNAVAILABLE")
        ].copy()

    discovery_absolute = _open_scoring_stage(
        historical_root, absolute_features, contract, "discovery"
    )
    discovery_outcomes, discovery = construct_stage(discovery_absolute)
    relative_search_space = candidate_search_space_counts(discovery, feature_families)
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
    write_deterministic_json(
        {
            **relative_search_space,
            "balanced_directional_candidate_allocation": {
                "univariate": int(census["stage"].eq("univariate").sum()),
                "pairwise": int(census["stage"].eq("pairwise").sum()),
            },
            "broad_examined_directional_candidates": int(
                census["stage"].isin(["univariate", "pairwise"]).sum()
            ),
            "broad_directional_cap": int(
                contract["search"]["univariate_and_pairwise_cap"]
            ),
            "selection": "outcome_free_balanced_allocation",
        },
        output_root / "candidate_search_space.json",
    )
    discovery_library = freeze_discovery_library(
        census,
        candidates,
        discovery,
        retained_stage_cap=int(contract["search"]["discovery_stage_retained_cap"]),
        per_direction_cap=int(contract["search"]["frozen_discovery_long_cap"]),
    )
    discovery_path = output_root / "relative_discovery_library.json"
    write_deterministic_json(_safe_json(discovery_library), discovery_path)
    discovery_hash = sha256_file(discovery_path)
    write_deterministic_json(
        {
            "stage": "discovery",
            "frozen_library_sha256": discovery_hash,
            "validation_or_final_opened_before_seal": False,
        },
        output_root / "discovery_stage_seal.json",
    )
    validation_absolute = _open_scoring_stage(
        historical_root, absolute_features, contract, "validation"
    )
    validation_outcomes, validation = construct_stage(validation_absolute)
    validation_metrics, survivor_library = validate_discovery_library(
        validation,
        discovery_library,
        support_rules=support_rules,
        holm_alpha=float(contract["multiplicity"]["retained_family_alpha"]),
        per_direction_cap=int(contract["search"]["validation_survivor_long_cap"]),
        bootstrap_draws=int(contract["multiplicity"]["session_block_bootstrap_draws"]),
        bootstrap_seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 1000,
    )
    survivor_path = output_root / "relative_survivor_library.json"
    write_deterministic_json(_safe_json(survivor_library), survivor_path)
    survivor_hash = sha256_file(survivor_path)
    write_deterministic_json(
        {
            "stage": "validation",
            "frozen_survivor_library_sha256": survivor_hash,
            "final_opened_before_seal": False,
        },
        output_root / "validation_stage_seal.json",
    )
    final_absolute = _open_scoring_stage(
        historical_root, absolute_features, contract, "final_opened_holdout"
    )
    final_outcomes, final = construct_stage(final_absolute)
    if sha256_file(discovery_path) != discovery_hash or sha256_file(survivor_path) != survivor_hash:
        raise AssertionError("Track B frozen library changed across chronological opens")
    final_metrics = score_frozen_library(
        final,
        survivor_library,
        bootstrap_draws=int(contract["multiplicity"]["session_block_bootstrap_draws"]),
        bootstrap_seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 1000,
    )
    # Full retrospective outcomes and delay diagnostics may be opened only after
    # both Track B rule-library seals have been materialized and hash-checked.
    absolute_joined, _ = _joined_scoring_frame(historical_root)
    absolute_joined = _assign_chronology_stage(absolute_joined, contract)
    absolute_scored = absolute_joined.loc[
        absolute_joined["target"].ne("UNAVAILABLE")
    ].copy()
    delayed_outcomes = pd.read_parquet(
        historical_root / "one_bar_delay_outcome_ledger.parquet"
    )
    absolute_delayed = absolute_joined.drop(
        columns=[
            "target",
            "gross_long_return_bps",
            "net_long_return_bps",
            "gross_short_return_bps",
            "net_short_return_bps",
            "long_net_bps",
            "short_net_bps",
            "score_status",
            "entry_timestamp",
            "entry_open",
            "terminal_timestamp",
            "terminal_close",
        ],
        errors="ignore",
    ).merge(
        delayed_outcomes[
            [
                "opportunity_id",
                "target",
                "gross_long_return_bps",
                "net_long_return_bps",
                "gross_short_return_bps",
                "net_short_return_bps",
            ]
        ],
        on="opportunity_id",
        how="left",
        validate="one_to_one",
    )
    absolute_delayed["long_net_bps"] = absolute_delayed["net_long_return_bps"]
    absolute_delayed["short_net_bps"] = absolute_delayed["net_short_return_bps"]
    absolute_delayed = absolute_delayed.loc[
        absolute_delayed["target"].ne("UNAVAILABLE")
    ].copy()
    development_outcomes, development = construct_stage(
        _stage(absolute_scored, "development_context")
    )
    relative_outcomes = pd.concat(
        [development_outcomes, discovery_outcomes, validation_outcomes, final_outcomes],
        ignore_index=True,
    )
    relative = pd.concat(
        [development, discovery, validation, final], ignore_index=True
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
    relative_delayed_outcomes = construct_relative_outcomes(
        absolute_delayed,
        minimum_peers=minimum_peers,
    )
    delayed_for_stress = relative_delayed_outcomes.rename(
        columns={
            "long_net_bps": "net_long_return_bps",
            "short_net_bps": "net_short_return_bps",
            "future_residual_return_bps": "gross_long_return_bps",
        }
    )
    relative_stress, _ = stress_signature_library(
        relative.assign(gross_long_return_bps=relative["future_residual_return_bps"]),
        discovery_library,
        delayed_for_stress,
        ordered_bins=ordered_bins,
    )
    relative_stress = relative_stress.loc[
        ~relative_stress["stress"].eq(
            "leave_one_stock_out_direct_cross_section_recomputed"
        )
    ].copy()
    track_b_loso_rows: list[dict[str, Any]] = []
    for stage in ("discovery", "validation", "final_opened_holdout"):
        absolute_stage = _stage(absolute_joined, stage)
        for removed_symbol in sorted(absolute_stage["symbol"].astype(str).unique()):
            reduced_absolute = recompute_cross_sectional_after_stock_deletion(
                absolute_stage, removed_symbol
            )
            _, reduced_relative = construct_stage(reduced_absolute)
            for entry in discovery_library:
                signature = signature_from_dict(entry["signature"])
                track_b_loso_rows.append(
                    {
                        **signature_metrics(reduced_relative, signature),
                        "chronology_stage": stage,
                        "period": int(reduced_relative["period"].iloc[0])
                        if len(reduced_relative)
                        else int(absolute_stage["period"].iloc[0]),
                        "stress": "leave_one_stock_out_recomputed",
                        "removed": removed_symbol,
                        "relative_outcomes_recomputed": True,
                        "direct_cross_sectional_features_recomputed": True,
                        "structural_state_loop_movement_context": (
                            "frozen_from_original_causal_timestamp_not_reestimated"
                        ),
                    }
                )
    relative_leave_one_out = pd.DataFrame(track_b_loso_rows)
    relative_stress = pd.concat(
        [relative_stress, relative_leave_one_out], ignore_index=True, sort=False
    )
    relative_concentration = signature_breakdowns(relative, discovery_library)
    rng = np.random.default_rng(
        int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 1000
    )
    frozen_ids = {str(entry["signature"]["signature_id"]) for entry in discovery_library}
    random_signatures = []
    used: set[str] = set()
    for entry in discovery_library:
        count = len(entry["signature"]["conditions"])
        rows = int(entry["discovery_metrics"]["rows"])
        pool = census.loc[
            census["condition_count"].eq(count) & ~census["signature_id"].isin(frozen_ids | used)
        ].copy()
        if pool.empty:
            continue
        pool["distance"] = (pool["rows"] - rows).abs()
        nearest = pool.sort_values(["distance", "signature_id"], kind="mergesort").head(20)
        chosen = str(nearest.iloc[int(rng.integers(0, len(nearest)))]["signature_id"])
        used.add(chosen)
        random_signatures.append(candidate_map[chosen])
    relative_nulls = null_test_results(
        relative,
        discovery_library,
        random_signatures,
        relative_atlas,
        seed=int(contract["multiplicity"]["session_block_bootstrap_seed"]) + 1000,
    )
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
    if sha256_file(discovery_path) != discovery_hash:
        raise AssertionError("Track B discovery library changed during reporting")
    _write_csv_allow_empty(
        validation_metrics,
        output_root / "relative_validation_metrics.csv",
        sort_by=["signature_id"],
    )
    if sha256_file(survivor_path) != survivor_hash:
        raise AssertionError("Track B validation library changed during final reporting")
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
        sort_by=["chronology_stage", "metric_basis"],
    )
    _write_csv_allow_empty(
        relative_nulls,
        output_root / "relative_null_test_results.csv",
        sort_by=["null"],
    )
    _write_csv_allow_empty(
        relative_stress,
        output_root / "relative_stability_results.csv",
        sort_by=["signature_id", "chronology_stage", "stress", "removed"],
    )
    _write_csv_allow_empty(
        relative_leave_one_out,
        output_root / "relative_leave_one_stock_out.csv",
        sort_by=["signature_id", "chronology_stage", "removed"],
    )
    _write_csv_allow_empty(
        relative_concentration,
        output_root / "relative_concentration_results.csv",
        sort_by=["signature_id", "dimension"],
    )
    relative_qualification = qualify_relative_persistence(
        survivor_library,
        validation_metrics,
        final_metrics,
        relative_stress,
        relative_nulls,
        contract,
    )
    _write_csv_allow_empty(
        relative_qualification,
        output_root / "relative_persistence_qualification.csv",
        sort_by=["signature_id"],
    )
    persistent_relative_ids = set(
        relative_qualification.loc[
            relative_qualification["strict_persistent_relative_signature"].astype(bool),
            "signature_id",
        ].astype(str)
    )
    beats_relative_strength = True
    for stage in ("validation", "final_opened_holdout"):
        atlas_row = relative_economic.loc[
            relative_economic["chronology_stage"].eq(stage)
        ]
        baseline_row = strength_economic.loc[
            strength_economic["chronology_stage"].eq(stage)
        ]
        beats_relative_strength &= bool(
            len(atlas_row)
            and len(baseline_row)
            and float(atlas_row.iloc[0]["net_bps_per_full_opportunity"])
            > float(baseline_row.iloc[0]["mean_residual_bps_per_opportunity"])
        )
    summary = {
        "status": "completed_secondary_equal_universe_only",
        "sector_relative_status": "unavailable_no_frozen_sector_membership",
        "candidate_signatures_examined": len(census),
        "discovery_multiplicity_qualified": int(census["discovery_eligible"].sum()),
        "frozen_exploratory_signatures": len(discovery_library),
        "validation_survivors": len(survivor_library),
        "final_scored_signatures": len(final_metrics),
        "persistent_relative_final_signatures": len(persistent_relative_ids),
        "persistent_relative_signature_ids": sorted(persistent_relative_ids),
        "atlas_beats_relative_strength_validation_and_final": beats_relative_strength,
        "absolute_profitability_claim_allowed": False,
        "portfolio_cost_translation_status": "not_implemented_absolute_profitability_forbidden",
        "null_families": int(len(relative_nulls)),
        "stability_rows": int(len(relative_stress)),
    }
    write_deterministic_json(summary, output_root / "track_b_summary.json")
    return summary


def run_track_b_phase(output_root: Path) -> dict[str, Any]:
    """Run secondary Track B only after an independent Track A audit passes."""

    audit_path = output_root / "track_a_independent_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("Track B requires a passing independent Track A audit marker")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("scope") != "track_a" or audit.get("passed") is not True:
        raise RuntimeError("Track B requires a passing independent Track A audit")
    metadata = json.loads((output_root / "run_metadata.json").read_text(encoding="utf-8"))
    expected_audit_identity = {
        "audit_id": "20260717-directional-signature-atlas-v1-independent-audit",
        "run_id": metadata["run_id"],
        "git_sha": metadata["git_sha"],
        "contract_sha256": metadata["contract_sha256"],
        "data_snapshot_sha256": metadata["data_snapshot_sha256"],
        "feature_schema_sha256": metadata["feature_schema_sha256"],
        "feature_ledger_sha256": metadata["feature_ledger_sha256"],
    }
    mismatches = [
        field
        for field, expected in expected_audit_identity.items()
        if str(audit.get(field)) != str(expected)
    ]
    audit_complete = int(audit.get("passed_check_count", -1)) == int(
        audit.get("check_count", -2)
    )
    audit_safe = audit.get("research_only") is True and audit.get("execution_enabled") is False
    if mismatches or not audit_complete or not audit_safe:
        raise RuntimeError(
            "Track B audit marker identity/safety mismatch: "
            f"fields={mismatches}; complete={audit_complete}; safe={audit_safe}"
        )
    contract = load_contract(CONTRACT_PATH)
    schema = _feature_schema()
    features = pd.read_parquet(output_root / "outcome_free_feature_ledger.parquet")
    features = _assign_chronology_stage(features, contract)
    search = contract["search"]
    caps = SearchCaps(
        univariate_and_pairwise=int(search["univariate_and_pairwise_cap"]),
        triples=int(search["three_condition_cap"]),
        tree=int(search["tree_candidate_cap"]),
        retained=int(search["discovery_stage_retained_cap"]),
    )
    support = contract["support"]
    support_rules = SupportRules(
        minimum_rows=int(support["minimum_rows"]),
        minimum_sessions=int(support["minimum_independent_sessions"]),
        minimum_stocks=int(support["minimum_independent_stocks"]),
        maximum_stock_fraction=float(support["maximum_single_stock_row_fraction"]),
        minimum_months=int(support["minimum_calendar_months"]),
        minimum_directional_outcomes=int(support["minimum_relevant_direction_outcomes"]),
    )
    feature_families = feature_family_map(schema)
    ordered_bins = {
        str(row["name"]): list(row["bins"]) for row in schema["features"] if row.get("bins")
    }
    summary = run_track_b(
        features,
        historical_root=output_root,
        output_root=output_root / "track_b",
        contract=contract,
        feature_families=feature_families,
        ordered_bins=ordered_bins,
        caps=caps,
        support_rules=support_rules,
    )
    track_a_summary_path = output_root / "track_a_summary.json"
    track_a_summary = json.loads(track_a_summary_path.read_text(encoding="utf-8"))
    track_a_summary["track_b"] = summary
    track_a_summary["scientific_decision"] = scientific_decision_label(
        track_a_summary, summary
    )
    write_deterministic_json(track_a_summary, track_a_summary_path)
    generate_track_b_plot(output_root)
    write_artifact_manifest(output_root)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "build",
            "score",
            "run",
            "track-b",
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
    if args.phase == "track-b":
        summary = run_track_b_phase(args.output_root)
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
