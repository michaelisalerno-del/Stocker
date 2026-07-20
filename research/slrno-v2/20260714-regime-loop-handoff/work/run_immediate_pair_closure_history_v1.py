#!/usr/bin/env python3
# ruff: noqa: E501
"""Run the fixed-model immediate pair-closure history diagnostic V1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages" / "stocker_research" / "src"
for import_root in (PACKAGE_ROOT, WORK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from stocker_research.causal_state_export_v2 import HysteresisConfig  # noqa: E402
from stocker_research.pair_closure_history_v1 import (  # noqa: E402
    MODEL_CONTEXT_LEVELS,
    build_pair_closure_population,
    canonical_json_bytes,
    expanding_month_predictions,
    frozen_assessment_predictions,
    pair_orientation_metrics,
    pair_replication_table,
    paired_session_bootstrap,
    prediction_metric_table,
    primary_decision_inputs,
    quarter_and_stock_deletion_metrics,
    safety_flags,
)
from stocker_research.regime_gap_segmentation_v2 import causal_segment_groups  # noqa: E402
from stocker_research.regime_panel_v2 import (  # noqa: E402
    RegimePanelConfig,
    build_regime_panel,
)
from stocker_research.regime_validity_v2 import (  # noqa: E402
    EmissionPreprocessing,
    SemiMarkovParameters,
    gaussian_log_emissions,
    transform_emissions,
)
from stocker_research.state_representation_sensitivity_v2 import (  # noqa: E402
    hysteretic_states_by_session,
)

EXPERIMENT_ID = "20260720-immediate-pair-closure-history-v1"
CONTRACT_PATH = WORK_DIR / "contracts" / f"{EXPERIMENT_ID}.json"
ARTIFACT_ROOT = WORK_DIR / "artifacts" / EXPERIMENT_ID
PRIMARY_DIR = ARTIFACT_ROOT / "primary"
EXACT_DIR = ARTIFACT_ROOT / "exact_rerun"
REPORT_PATH = WORK_DIR / "reports" / f"{EXPERIMENT_ID}.md"
REPAIRED_ROOT = WORK_DIR / "artifacts" / "20260719-right-censored-regime-refit-v2" / "primary"
MODEL_PATH = REPAIRED_ROOT / "full_refit_parameters.npz"
MAPPING_PATH = REPAIRED_ROOT / "full_refit_semantic_mapping.csv"
PREPROCESSING_PATH = REPAIRED_ROOT / "full_refit_preprocessing.csv"
PART_A_DECISION_PATH = REPAIRED_ROOT / "repaired_part_a_decision.json"
PROVIDER_ROOT = (
    Path.home()
    / "StockerLocal/data/processed/source=eodhd/instrument_type=stock"
)
SYMBOLS = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "AXTI",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "OKLO",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)
DEVELOPMENT_START = pd.Timestamp("2024-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
ASSESSMENT_START = pd.Timestamp("2025-01-01", tz="UTC")
ASSESSMENT_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
PRIMARY_REPRESENTATION = "CAUSAL_HYSTERETIC_SEMANTIC"
SENSITIVITY_REPRESENTATION = "CAUSAL_HARD_SEMANTIC"
EXPECTED_ROWS = {"DEVELOPMENT_2024": 424_583, "ASSESSMENT_2025": 424_827}
EXPECTED_SNAPSHOTS = {
    "DEVELOPMENT_2024": "48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661",
    "ASSESSMENT_2025": "29e82d6539810e5fcebc13e860d07474c38ee0349fe38aedce0378f9aefb67a4",
}
EXPECTED_MODEL_HASH = "4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425"
EXPECTED_FILE_HASHES = {
    MODEL_PATH: "e267edc80298ff4d2915aa57b00f915cf34659b978ccdbc796d8103c8be4b9d2",
    MAPPING_PATH: "840e105d3c3a6ef0c63d754656d473a7b5703003b6fea4b08e5786af8c98b523",
    PREPROCESSING_PATH: "b03681a1ef94e2118d0ff873334c7bb489730f784557bc7d742921feafb2b0db",
}
IMPLEMENTATION_PATHS = (
    Path("packages/stocker_research/src/stocker_research/pair_closure_history_v1.py"),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/"
        "run_immediate_pair_closure_history_v1.py"
    ),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/"
        "audit_immediate_pair_closure_history_v1.py"
    ),
    Path(
        "research/slrno-v2/20260714-regime-loop-handoff/work/contracts/"
        "20260720-immediate-pair-closure-history-v1.json"
    ),
)
SCIENTIFIC_FILES = (
    "run_metadata.json",
    "source_identity_manifest.json",
    "model_effective_configuration.json",
    "state_assignment_summary.csv",
    "pair_closure_population.parquet",
    "censoring_summary.csv",
    "development_oof_predictions.parquet",
    "assessment_predictions.parquet",
    "model_metrics.csv",
    "paired_session_bootstrap_draws.csv",
    "paired_session_bootstrap_summary.csv",
    "quarter_metrics.csv",
    "leave_one_stock_out.csv",
    "pair_orientation_metrics.csv",
    "pair_replication.csv",
    "concentration.csv",
    "decision_inputs.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(
        json.dumps(payload, sort_keys=True, indent=2, default=str).encode("utf-8") + b"\n"
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        date_format="%Y-%m-%dT%H:%M:%S.%f%z",
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=9,
        row_group_size=50_000,
    )


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_contract() -> tuple[dict[str, Any], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for key, expected in safety_flags().items():
        if contract.get(key) != expected:
            raise RuntimeError(f"contract safety flag differs: {key}")
    if contract.get("part_b_contract_reopened") is not False:
        raise RuntimeError("diagnostic cannot reopen blocked Part B")
    if contract.get("numeric_state_semantic_validity_claimed") is not False:
        raise RuntimeError("diagnostic cannot claim numeric semantic validity")
    for path, expected in EXPECTED_FILE_HASHES.items():
        if _sha256_file(path) != expected:
            raise RuntimeError(f"bound repaired-model file changed: {path.name}")
    part_a = json.loads(PART_A_DECISION_PATH.read_text(encoding="utf-8"))
    if part_a["decision"] != "regime_representation_unstable_loop_dictionary_must_pause":
        raise RuntimeError("Part A decision changed; freeze a new diagnostic contract")
    if part_a["dictionary_work_may_resume"] is not False:
        raise RuntimeError("unexpected repaired Part A authorization")
    return contract, _sha256_file(CONTRACT_PATH)


def _load_model() -> tuple[SemiMarkovParameters, EmissionPreprocessing, dict[int, int]]:
    with np.load(MODEL_PATH, allow_pickle=False) as stored:
        parameters = SemiMarkovParameters(
            means=np.asarray(stored["means"]).copy(),
            variances=np.asarray(stored["variances"]).copy(),
            duration_hazard=np.asarray(stored["duration_hazard"]).copy(),
            transitions=np.asarray(stored["transitions"]).copy(),
            initial=np.asarray(stored["initial"]).copy(),
            occupancy=np.asarray(stored["occupancy"]).copy(),
        )
        preprocessing = EmissionPreprocessing(
            feature_names=tuple(str(value) for value in stored["preprocessing_feature_names"]),
            medians=np.asarray(stored["preprocessing_medians"]).copy(),
            centers=np.asarray(stored["preprocessing_centers"]).copy(),
            scales=np.asarray(stored["preprocessing_scales"]).copy(),
        )
        stored_model_hash = str(np.asarray(stored["state_model_hash"]).item())
    parameters.validate()
    preprocessing.validate()
    if stored_model_hash != EXPECTED_MODEL_HASH:
        raise RuntimeError("repaired model identity differs")
    mapping_frame = pd.read_csv(MAPPING_PATH)
    mapping = {
        int(row.raw_cluster_state): int(row.semantic_state)
        for row in mapping_frame.itertuples(index=False)
    }
    if set(mapping) != set(range(8)) or set(mapping.values()) != set(range(8)):
        raise RuntimeError("semantic mapping is not a permutation of eight states")
    return parameters, preprocessing, mapping


def _semantic_probabilities(raw_probabilities: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    output = np.zeros_like(raw_probabilities, dtype=float)
    for raw_state, semantic_state in mapping.items():
        output[:, semantic_state] = raw_probabilities[:, raw_state]
    if not np.allclose(output.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("semantic probability mapping lost mass")
    return output


def _state_summary(
    panel: pd.DataFrame,
    states: np.ndarray,
    *,
    period: str,
    representation: str,
) -> dict[str, object]:
    transitions = 0
    for positions in causal_segment_groups(panel):
        values = states[positions]
        transitions += int(np.sum(values[1:] != values[:-1]))
    counts = np.bincount(states, minlength=8)
    return {
        "period": period,
        "representation": representation,
        "bars": len(panel),
        "symbols": panel["symbol"].nunique(),
        "sessions": panel["session"].nunique(),
        "segments": panel["segment_id"].nunique(),
        "state_transitions": transitions,
        **{f"state_{state}_bars": int(counts[state]) for state in range(8)},
    }


def _build_period_population(
    *,
    period: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parameters: SemiMarkovParameters,
    preprocessing: EmissionPreprocessing,
    mapping: dict[int, int],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    print(f"pair-closure: build bounded {period} panel", flush=True)
    build = build_regime_panel(
        RegimePanelConfig(
            provider_root=PROVIDER_ROOT,
            symbols=SYMBOLS,
            benchmark_symbol="VTI",
            start=start,
            end=end,
        )
    )
    panel = build.frame
    if len(panel) != EXPECTED_ROWS[period]:
        raise RuntimeError(f"{period} row count changed: {len(panel)}")
    if build.data_snapshot_hash != EXPECTED_SNAPSHOTS[period]:
        raise RuntimeError(f"{period} source snapshot changed")
    if pd.to_datetime(panel["bar_start_timestamp"], utc=True).dt.year.max() >= 2026:
        raise RuntimeError("protected 2026 row was opened")
    print(f"pair-closure: causal filter {period}", flush=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/stocker_pair_closure_history_numba_v1")
    from regime_validity_pipeline_v2 import _causal_filter_summary_compiled

    scaled = transform_emissions(panel, preprocessing)
    emissions = gaussian_log_emissions(scaled, parameters)
    groups = causal_segment_groups(panel)
    summary = _causal_filter_summary_compiled(
        emissions,
        groups=groups,
        model=parameters.as_dict(),
    )
    raw_hard = np.asarray(summary.hard_states, dtype=int)
    raw_hysteretic = hysteretic_states_by_session(
        summary.state_probabilities,
        session_groups=groups,
        config=HysteresisConfig(0.55, 0.10),
    )
    semantic_hard = np.asarray([mapping[int(value)] for value in raw_hard], dtype=int)
    semantic_hysteretic = np.asarray([mapping[int(value)] for value in raw_hysteretic], dtype=int)
    semantic_probability = _semantic_probabilities(summary.state_probabilities, mapping)
    population_parts: list[pd.DataFrame] = []
    state_rows: list[dict[str, object]] = []
    for representation, states in (
        (PRIMARY_REPRESENTATION, semantic_hysteretic),
        (SENSITIVITY_REPRESENTATION, semantic_hard),
    ):
        print(f"pair-closure: population {period} {representation}", flush=True)
        population = build_pair_closure_population(
            panel,
            semantic_states=states,
            state_probabilities=semantic_probability,
            posterior_entropy=summary.posterior_entropy,
            departure_probability=summary.departure_probability,
            representation=representation,
        )
        population["period"] = period
        population_parts.append(population)
        state_rows.append(
            _state_summary(
                panel,
                states,
                period=period,
                representation=representation,
            )
        )
    identity = {
        "period": period,
        "rows": len(panel),
        "row_key_hash": build.row_key_hash,
        "feature_table_hash": build.feature_table_hash,
        "data_snapshot_hash": build.data_snapshot_hash,
        "source_hashes": dict(sorted(build.source_hashes.items())),
        "source_row_counts": dict(sorted(build.source_row_counts.items())),
        "gap_rows": len(build.gap_ledger),
    }
    output = pd.concat(population_parts, ignore_index=True).sort_values(
        ["representation", "symbol", "session", "decision_timestamp", "decision_id"],
        kind="mergesort",
    )
    del panel, build, scaled, emissions, summary, semantic_probability
    gc.collect()
    return output.reset_index(drop=True), identity, pd.DataFrame.from_records(state_rows)


def _concentration(population: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    available = population.loc[population["target_available"].astype(bool)]
    for (period, representation), group in available.groupby(
        ["period", "representation"], sort=True
    ):
        counts = group["symbol"].value_counts()
        rows.append(
            {
                "period": period,
                "representation": representation,
                "rows": len(group),
                "stocks": group["symbol"].nunique(),
                "sessions": group["session"].nunique(),
                "maximum_stock_share": float(counts.iloc[0] / len(group)),
                "largest_stock": str(counts.index[0]),
                "largest_stock_rows": int(counts.iloc[0]),
                "maximum_month_share": float(
                    pd.to_datetime(group["decision_timestamp"], utc=True)
                    .dt.strftime("%Y-%m")
                    .value_counts(normalize=True)
                    .iloc[0]
                ),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["period", "representation"])


def _artifact_manifest(directory: Path) -> dict[str, Any]:
    records = []
    for name in SCIENTIFIC_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"file": name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "artifacts": records,
        **safety_flags(),
    }
    _write_json(directory / "artifact_manifest.json", payload)
    return payload


def run(output_dir: Path) -> dict[str, Any]:
    """Run one deterministic artifact build without opening price outcomes."""

    contract, contract_hash = _load_contract()
    implementation_hash = _implementation_hash()
    parameters, preprocessing, mapping = _load_model()
    development, development_identity, development_states = _build_period_population(
        period="DEVELOPMENT_2024",
        start=DEVELOPMENT_START,
        end=DEVELOPMENT_END,
        parameters=parameters,
        preprocessing=preprocessing,
        mapping=mapping,
    )
    assessment, assessment_identity, assessment_states = _build_period_population(
        period="ASSESSMENT_2025",
        start=ASSESSMENT_START,
        end=ASSESSMENT_END,
        parameters=parameters,
        preprocessing=preprocessing,
        mapping=mapping,
    )
    run_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "contract_hash": contract_hash,
                "implementation_hash": implementation_hash,
                "model_hash": EXPECTED_MODEL_HASH,
                "development_snapshot": development_identity["data_snapshot_hash"],
                "assessment_snapshot": assessment_identity["data_snapshot_hash"],
            }
        )
    ).hexdigest()[:24]
    output_dir.mkdir(parents=True, exist_ok=True)
    print("pair-closure: expanding 2024 folds", flush=True)
    development_predictions = pd.concat(
        [
            expanding_month_predictions(
                development.loc[development["representation"].eq(representation)]
            )
            for representation in (PRIMARY_REPRESENTATION, SENSITIVITY_REPRESENTATION)
        ],
        ignore_index=True,
    ).sort_values(["representation", "decision_id", "model"], kind="mergesort")
    print("pair-closure: frozen 2025 assessment", flush=True)
    assessment_predictions = pd.concat(
        [
            frozen_assessment_predictions(
                development.loc[development["representation"].eq(representation)],
                assessment.loc[assessment["representation"].eq(representation)],
            )
            for representation in (PRIMARY_REPRESENTATION, SENSITIVITY_REPRESENTATION)
        ],
        ignore_index=True,
    ).sort_values(["representation", "decision_id", "model"], kind="mergesort")
    all_predictions = pd.concat(
        [development_predictions, assessment_predictions], ignore_index=True
    )
    model_metrics = prediction_metric_table(all_predictions)
    bootstrap_draws, bootstrap_summary = paired_session_bootstrap(all_predictions)
    quarter_metrics, stock_deletions = quarter_and_stock_deletion_metrics(assessment_predictions)
    print("pair-closure: pair orientation uncertainty", flush=True)
    pair_metrics = pd.concat(
        [
            pair_orientation_metrics(development, period="DEVELOPMENT_2024"),
            pair_orientation_metrics(assessment, period="ASSESSMENT_2025"),
        ],
        ignore_index=True,
    )
    pair_replication = pair_replication_table(pair_metrics)
    population = pd.concat([development, assessment], ignore_index=True).sort_values(
        ["period", "representation", "symbol", "session", "decision_timestamp"],
        kind="mergesort",
    )
    censoring = (
        population.groupby(
            ["period", "representation", "target_available", "censor_reason"],
            sort=True,
            dropna=False,
        )
        .size()
        .rename("rows")
        .reset_index()
    )
    concentration = _concentration(population)
    decision_inputs = primary_decision_inputs(
        bootstrap_summary,
        quarter_metrics,
        stock_deletions,
        primary_representation=PRIMARY_REPRESENTATION,
        sensitivity_representation=SENSITIVITY_REPRESENTATION,
    )
    git_sha = _git("rev-parse", "HEAD")
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "git_sha": git_sha,
        "branch": _git("branch", "--show-current"),
        "contract_hash": contract_hash,
        "implementation_hash": implementation_hash,
        "state_model_hash": EXPECTED_MODEL_HASH,
        "development_population_rows": len(development),
        "assessment_population_rows": len(assessment),
        "development_evaluable_rows": int(development["target_available"].sum()),
        "assessment_evaluable_rows": int(assessment["target_available"].sum()),
        "random_seeds": contract["random_seeds"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: _version(name) for name in ("numpy", "pandas", "pyarrow", "numba", "scipy")
        },
        **safety_flags(),
        "part_b_contract_reopened": False,
        "numeric_state_semantic_validity_claimed": False,
    }
    source_manifest = {
        "provider": "EODHD",
        "dataset_identity": "StockerLocal/source=eodhd/instrument_type=stock/timeframe=5m",
        "volume_meaning": "provider_reported_historical_volume_activity",
        "bar_timestamp_convention": (
            "provider timestamp is bar start; availability is bar start plus five minutes"
        ),
        "development": development_identity,
        "assessment": assessment_identity,
        "protected_2026_opened": False,
        "run_id": run_id,
        "contract_hash": contract_hash,
        **safety_flags(),
    }
    configuration = {
        "experiment_id": EXPERIMENT_ID,
        "model_context_levels": {
            name: [list(level) for level in levels] for name, levels in MODEL_CONTEXT_LEVELS.items()
        },
        "frequency_model": {
            "tau": 64.0,
            "alpha": 0.5,
            "beta": 0.5,
        },
        "primary_baseline": "M2_IMMEDIATE_PAIR",
        "primary_history_model": "M5_LAST_FIVE_STATES",
        "primary_representation": PRIMARY_REPRESENTATION,
        "sensitivity_representation": SENSITIVITY_REPRESENTATION,
        "hysteresis": {"switch_probability": 0.55, "switch_margin": 0.10},
        "bootstrap": {"draws": 2_000, "seed": 20_260_720},
        "numeric_state_semantic_validity_claimed": False,
        "promotable": False,
        **safety_flags(),
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    _write_json(output_dir / "source_identity_manifest.json", source_manifest)
    _write_json(output_dir / "model_effective_configuration.json", configuration)
    _write_csv(
        output_dir / "state_assignment_summary.csv",
        pd.concat([development_states, assessment_states], ignore_index=True).sort_values(
            ["period", "representation"]
        ),
    )
    _write_parquet(output_dir / "pair_closure_population.parquet", population)
    _write_csv(output_dir / "censoring_summary.csv", censoring)
    _write_parquet(output_dir / "development_oof_predictions.parquet", development_predictions)
    _write_parquet(output_dir / "assessment_predictions.parquet", assessment_predictions)
    _write_csv(output_dir / "model_metrics.csv", model_metrics)
    _write_csv(output_dir / "paired_session_bootstrap_draws.csv", bootstrap_draws)
    _write_csv(output_dir / "paired_session_bootstrap_summary.csv", bootstrap_summary)
    _write_csv(output_dir / "quarter_metrics.csv", quarter_metrics)
    _write_csv(output_dir / "leave_one_stock_out.csv", stock_deletions)
    _write_csv(output_dir / "pair_orientation_metrics.csv", pair_metrics)
    _write_csv(output_dir / "pair_replication.csv", pair_replication)
    _write_csv(output_dir / "concentration.csv", concentration)
    _write_json(output_dir / "decision_inputs.json", decision_inputs)
    manifest = _artifact_manifest(output_dir)
    return {
        "run_id": run_id,
        "artifact_count": len(manifest["artifacts"]),
        "development_population_rows": len(development),
        "assessment_population_rows": len(assessment),
        "preliminary_decision": decision_inputs["preliminary_statistical_decision"],
    }


def compare_exact_rerun() -> dict[str, Any]:
    """Compare every scientific file in primary and exact-rerun directories."""

    comparisons = []
    for name in (*SCIENTIFIC_FILES, "artifact_manifest.json"):
        primary = PRIMARY_DIR / name
        exact = EXACT_DIR / name
        if not primary.is_file() or not exact.is_file():
            raise FileNotFoundError(name)
        primary_hash = _sha256_file(primary)
        exact_hash = _sha256_file(exact)
        comparisons.append(
            {
                "file": name,
                "primary_sha256": primary_hash,
                "exact_rerun_sha256": exact_hash,
                "byte_identical": primary_hash == exact_hash,
            }
        )
    passed = all(bool(row["byte_identical"]) for row in comparisons)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "byte_identical": passed,
        "compared_file_count": len(comparisons),
        "comparisons": comparisons,
        **safety_flags(),
    }
    for role, directory in (("primary", PRIMARY_DIR), ("exact_rerun", EXACT_DIR)):
        _write_json(directory / "exact_rerun_manifest.json", {**payload, "directory_role": role})
    if not passed:
        raise RuntimeError("scientific exact rerun differs")
    return payload


def finalize() -> dict[str, Any]:
    """Bind audit and rerun status into the final non-promotable decision/report."""

    inputs = json.loads((PRIMARY_DIR / "decision_inputs.json").read_text(encoding="utf-8"))
    exact = json.loads((PRIMARY_DIR / "exact_rerun_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((PRIMARY_DIR / "independent_audit.json").read_text(encoding="utf-8"))
    reproducibility = bool(exact["byte_identical"] and audit["audit_passed"])
    if not reproducibility:
        decision = "blocked_data_chronology_or_reproducibility_failure"
    else:
        decision = str(inputs["preliminary_statistical_decision"])
    payload = {
        **inputs,
        "decision": decision,
        "exact_rerun_byte_identical": bool(exact["byte_identical"]),
        "independent_audit_passed": bool(audit["audit_passed"]),
        "promotion_authorized": False,
        "economic_testing_authorized": False,
        "part_b_contract_reopened": False,
        "numeric_state_semantic_validity_claimed": False,
        **safety_flags(),
    }
    _write_json(PRIMARY_DIR / "scientific_decision.json", payload)
    metrics = pd.read_csv(PRIMARY_DIR / "paired_session_bootstrap_summary.csv")
    primary = metrics.loc[
        metrics["representation"].eq(PRIMARY_REPRESENTATION)
        & metrics["evaluation_period"].eq("ASSESSMENT_2025")
    ].iloc[0]
    pairs = pd.read_csv(PRIMARY_DIR / "pair_replication.csv")
    primary_pairs = pairs.loc[pairs["representation"].eq(PRIMARY_REPRESENTATION)]
    selected = primary_pairs.loc[primary_pairs["development_selected"].astype(bool)]
    replicated = primary_pairs.loc[
        primary_pairs["assessment_significant_same_direction"].astype(bool)
    ]
    population = pd.read_parquet(PRIMARY_DIR / "pair_closure_population.parquet")
    development_rows = int(population["period"].eq("DEVELOPMENT_2024").sum())
    assessment_rows = int(population["period"].eq("ASSESSMENT_2025").sum())
    report = f"""# Immediate Regime-Pair Closure History Diagnostic V1

Safety boundary: `research_only=True`, `execution_enabled=False`, `order_placement=disabled`, `live_ordering_enabled=False`, `strategy_promotion=False`.

## Scope

Fixed-model structural sensitivity only. The experiment asks whether a five-state context (the current run plus four preceding runs) adds information beyond the immediate A→B pair for predicting the next-state A→B→A closure. It uses no future price, return, payoff, spread, broker, order, position, or 2026 row.

## Scientific limitation

The repaired regime representation remains semantically unstable across refits. Numeric pair identities in this report are valid only inside the hash-bound fitted model and are non-promotable. Blocked Part B was not reopened.

## Population

- Development 2024 pair decisions: {development_rows:,} across 22 stocks.
- Unchanged assessment 2025 pair decisions: {assessment_rows:,} across 22 stocks.
- Primary representation: causal hysteretic semantic labels; causal hard labels are a required sensitivity.
- Volume provenance: provider-reported EODHD historical activity only; volume was not a model input to this diagnostic.

## Primary result

- M5-minus-M2 assessment log-loss improvement: {float(primary["log_loss_improvement"]):.8f}.
- Paired session-block 95% interval: [{float(primary["log_loss_ci_low"]):.8f}, {float(primary["log_loss_ci_high"]):.8f}].
- M5-minus-M2 Brier improvement: {float(primary["brier_improvement"]):.8f}.
- Primary development-selected supported pair orientations: {len(selected):,}.
- Primary same-direction, BH-significant 2025 replications: {len(replicated):,}.

## Decision

`{decision}`

This is evidence about fixed-model structural predictability only. It is not directional price evidence, economic payoff evidence, executable edge evidence, or authorization for another loop dictionary, strategy, or economic test.

## Reproducibility

- Exact rerun byte-identical: `{bool(exact["byte_identical"])}`.
- Independent audit passed: `{bool(audit["audit_passed"])}`.
- No orders, broker connections, positions, accounts, or execution interfaces were used.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compare-exact", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    arguments = parser.parse_args()
    selected = sum([arguments.output_dir is not None, arguments.compare_exact, arguments.finalize])
    if selected != 1:
        parser.error("choose exactly one operation")
    if arguments.output_dir is not None:
        print(json.dumps(run(arguments.output_dir.resolve()), sort_keys=True))
    elif arguments.compare_exact:
        print(json.dumps(compare_exact_rerun(), sort_keys=True))
    else:
        print(json.dumps(finalize(), sort_keys=True))


if __name__ == "__main__":
    main()
