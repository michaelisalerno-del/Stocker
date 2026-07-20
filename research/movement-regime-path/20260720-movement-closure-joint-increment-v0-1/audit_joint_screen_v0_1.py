#!/usr/bin/env python3
"""Independent auditor for Movement x Closure-History Joint Increment V0.1.

This file intentionally does not import the runner or reusable joint-screen module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
MOVEMENT_DIR = (
    REPO_ROOT
    / "research/movement-regime-path/20260720-movement-conditioned-regime-path-probability-chain-v0"
)
CLOSURE_ROOT = REPO_ROOT / "research/slrno-v2/20260714-regime-loop-handoff/work"
MOVEMENT_PRIMARY = MOVEMENT_DIR / "artifacts/primary"
CLOSURE_PRIMARY = CLOSURE_ROOT / "artifacts/20260720-immediate-pair-closure-history-v1/primary"
MAPPING_PATH = (
    CLOSURE_ROOT
    / "artifacts/20260719-right-censored-regime-refit-v2/primary/full_refit_semantic_mapping.csv"
)

START_SESSION = "2024-07-01"
END_SESSION_EXCLUSIVE = "2025-08-23"
MODEL_ID = "regime_model_v2_full_right_censored_refit"
MODEL_HASH = "4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425"
REPRESENTATION = "CAUSAL_HARD_SEMANTIC"
REPRESENTATION_ID = f"{MODEL_ID}|{MODEL_HASH}|{REPRESENTATION}"
EPSILON = 1e-6

SAFETY = {
    "research_only": True,
    "feasibility_screen": True,
    "representation_specific": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}

MOVEMENT_BASE_COLUMNS = [
    "symbol",
    "session",
    "decision_ordinal",
    "feature_available_timestamp_utc",
    "origin_state",
    "origin_segment_id",
    "history_ordinals_contiguous",
    "future_ordinals_contiguous",
    "source_gap_crossed",
    "session_boundary_crossed",
    "session_source_complete",
    "scheduled_bars_remaining",
    "state_model_hash",
    "p_move",
    "predicted_absolute_movement_bps",
    "large_move",
]

A2_FEATURES = (
    "logit_p_close_m5",
    "pair_age_bars",
    "scheduled_bars_remaining",
    "decision_ordinal",
)
A3_FEATURES = (*A2_FEATURES, "logit_p_move", "log1p_predicted_absolute_movement_bps")
B0_FEATURES = (
    "logit_p_move",
    "pair_age_bars",
    "scheduled_bars_remaining",
    "decision_ordinal",
)
B1_FEATURES = (*B0_FEATURES, "logit_p_close_m5", "closure_history_increment")
C1_FEATURES = (
    "logit_p_move",
    "logit_p_close_m5",
    "closure_history_increment",
    "pair_age_bars",
    "scheduled_bars_remaining",
    "decision_ordinal",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"non-object evidence: {path}")
    return value


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _passed(audit: dict[str, Any], name: str) -> bool:
    return any(
        item.get("check") == name and item.get("passed") is True for item in audit.get("checks", [])
    )


def _upstream_evidence(mapping_hash: str) -> dict[str, Any]:
    movement_contract_path = MOVEMENT_DIR / "contract.json"
    movement_source_path = MOVEMENT_PRIMARY / "source_manifest.json"
    movement_fold_path = MOVEMENT_PRIMARY / "chronological_fold_manifest.json"
    movement_models_path = MOVEMENT_PRIMARY / "model_coefficients.json"
    movement_audit_path = MOVEMENT_PRIMARY / "independent_audit.json"
    movement_boundary_path = MOVEMENT_PRIMARY / "protected_boundary_audit.json"
    closure_contract_path = (
        CLOSURE_ROOT / "contracts/20260720-immediate-pair-closure-history-v1.json"
    )
    closure_source_path = CLOSURE_PRIMARY / "source_identity_manifest.json"
    closure_run_path = CLOSURE_PRIMARY / "run_metadata.json"
    closure_models_path = CLOSURE_PRIMARY / "model_effective_configuration.json"
    closure_audit_path = CLOSURE_PRIMARY / "independent_audit.json"
    closure_censoring_path = CLOSURE_PRIMARY / "censoring_summary.csv"

    movement_contract = _json(movement_contract_path)
    movement_source = _json(movement_source_path)
    movement_folds = _json(movement_fold_path)
    movement_models = _json(movement_models_path)
    movement_audit = _json(movement_audit_path)
    movement_boundary = _json(movement_boundary_path)
    closure_source = _json(closure_source_path)
    closure_run = _json(closure_run_path)
    closure_models = _json(closure_models_path)
    closure_audit = _json(closure_audit_path)
    closure_censoring = pd.read_csv(closure_censoring_path)

    folds = {
        str(row["score_month"]): row
        for row in movement_folds["folds"]
        if row.get("layer") == "movement"
    }
    expected_months = {f"2024-{month:02d}" for month in range(7, 13)}
    if (
        set(folds) != expected_months
        or movement_folds.get("all_upstream_predictions_out_of_fold") is not True
        or movement_folds.get("in_sample_stacked_features") != 0
        or movement_audit.get("passed") is not True
        or not _passed(movement_audit, "oof_and_stacking_chronology")
    ):
        raise AssertionError("movement chronology evidence failed")
    for row in folds.values():
        if row.get("strictly_earlier") is not True or pd.Timestamp(
            row["trained_through"], tz="UTC"
        ) >= pd.Timestamp(row["score_start"], tz="UTC"):
            raise AssertionError("movement fold crosses score period")
    final_models = movement_models["models"]
    final_training_rows = int(final_models["P1"]["training_rows"])
    if (
        final_training_rows <= 0
        or int(final_models["P1_SIZE"]["training_rows"]) != final_training_rows
        or movement_contract["chronology"]["final_score_period"]
        != "2025 sessions strictly before 2025-08-23"
        or movement_boundary.get("protected_rows_materialised") != 0
    ):
        raise AssertionError("movement final-fit evidence failed")

    if (
        closure_audit.get("audit_passed") is not True
        or closure_audit.get("failed_checks") != []
        or not _passed(closure_audit, "expanding_fold_training_counts")
        or not _passed(closure_audit, "assessment_fit_uses_2024_count_only")
        or closure_run.get("state_model_hash") != MODEL_HASH
        or closure_run.get("contract_hash") != _sha256(closure_contract_path)
        or closure_models.get("sensitivity_representation") != REPRESENTATION
        or closure_source.get("provider") != "EODHD"
    ):
        raise AssertionError("closure chronology evidence failed")
    count_row = closure_censoring.loc[
        closure_censoring["period"].astype(str).eq("DEVELOPMENT_2024")
        & closure_censoring["representation"].astype(str).eq(REPRESENTATION)
        & closure_censoring["target_available"].astype(str).str.lower().eq("true")
    ]
    if len(count_row) != 1:
        raise AssertionError("closure training count evidence missing")
    closure_training_rows = int(count_row.iloc[0]["rows"])

    cohort = [str(value) for value in movement_source["decision_cohort"]]
    movement_provider = movement_source["provider_sources"]
    closure_development = closure_source["development"]
    closure_assessment = closure_source["assessment"]
    if (
        len(cohort) != 20
        or movement_source["frozen_model_hash"] != MODEL_HASH
        or movement_source["frozen_development_snapshot_hash"]
        != closure_development["data_snapshot_hash"]
        or movement_source["frozen_development_panel_hash"]
        != closure_development["feature_table_hash"]
    ):
        raise AssertionError("shared source snapshot evidence failed")
    logical_paths: dict[str, str] = {}
    bounded_assessment_hashes: dict[str, str] = {}
    for stock in cohort:
        source = movement_provider[stock]
        logical = str(source["logical_path"])
        if (
            not logical.endswith(f"symbol={stock}/timeframe=5m/data.parquet")
            or source["bounded_2024_hash"] != closure_development["source_hashes"][stock]
            or stock not in closure_assessment["source_hashes"]
        ):
            raise AssertionError(f"source lineage mismatch: {stock}")
        logical_paths[stock] = logical
        bounded_assessment_hashes[stock] = source["bounded_safe_hash"]
    lineage_evidence = {
        "provider": "EODHD",
        "instrument_type": "stock",
        "timeframe": "5m",
        "cohort": cohort,
        "logical_paths": logical_paths,
        "state_model_hash": MODEL_HASH,
        "semantic_mapping_hash": mapping_hash,
        "shared_development_snapshot_hash": closure_development["data_snapshot_hash"],
        "shared_development_feature_table_hash": closure_development["feature_table_hash"],
        "shared_development_source_hashes": {
            stock: closure_development["source_hashes"][stock] for stock in cohort
        },
        "movement_bounded_assessment_end_exclusive": END_SESSION_EXCLUSIVE,
        "movement_bounded_assessment_hashes": bounded_assessment_hashes,
        "closure_assessment_snapshot_hash": closure_assessment["data_snapshot_hash"],
        "closure_assessment_source_hashes": {
            stock: closure_assessment["source_hashes"][stock] for stock in cohort
        },
    }
    movement_chronology = {
        "fold_manifest_sha256": _sha256(movement_fold_path),
        "model_coefficients_sha256": _sha256(movement_models_path),
        "independent_audit_sha256": _sha256(movement_audit_path),
        "folds": folds,
        "final_training_rows": final_training_rows,
        "final_trained_through": "2024-12-31T23:59:59.999999999Z",
    }
    closure_chronology = {
        "run_metadata_sha256": _sha256(closure_run_path),
        "model_configuration_sha256": _sha256(closure_models_path),
        "independent_audit_sha256": _sha256(closure_audit_path),
        "development_evaluation_period": "DEVELOPMENT_2024_OOF",
        "assessment_evaluation_period": "ASSESSMENT_2025",
        "assessment_training_rows": closure_training_rows,
        "final_trained_through": "2024-12-31T23:59:59.999999999Z",
    }
    return {
        "lineage_id": f"joint-source-lineage-v0-1|{_canonical_hash(lineage_evidence)}",
        "lineage_evidence": lineage_evidence,
        "movement_chronology": movement_chronology,
        "movement_chronology_evidence_id": _canonical_hash(movement_chronology),
        "closure_chronology": closure_chronology,
        "closure_chronology_evidence_id": _canonical_hash(closure_chronology),
        "closure_source": closure_source,
    }


def _row_id(values: Sequence[object]) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode()).hexdigest()[:24]


def _logit(values: pd.Series | np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    array = np.asarray(values, dtype=float)
    clipped = np.clip(array, EPSILON, 1.0 - EPSILON)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=float)


def _read_movement(mapping: dict[int, int], evidence: dict[str, Any]) -> pd.DataFrame:
    development = pd.read_parquet(
        MOVEMENT_PRIMARY / "oof_2024_predictions.parquet",
        columns=[
            *MOVEMENT_BASE_COLUMNS,
            "p_move__trained_through",
            "predicted_absolute_movement_bps__trained_through",
        ],
        filters=[("session", ">=", START_SESSION), ("session", "<", "2025-01-01")],
    )
    assessment = pd.read_parquet(
        MOVEMENT_PRIMARY / "scored_2025_predictions.parquet",
        columns=MOVEMENT_BASE_COLUMNS,
        filters=[
            ("session", ">=", "2025-01-01"),
            ("session", "<", END_SESSION_EXCLUSIVE),
        ],
    )
    development["year"] = 2024
    assessment["year"] = 2025
    chronology = evidence["movement_chronology"]
    months = development["session"].astype(str).str[:7]
    if set(months.unique()) != set(chronology["folds"]):
        raise AssertionError("movement score months differ from fold evidence")
    for month, fold in chronology["folds"].items():
        rows = development.loc[months.eq(month)]
        cutoff = pd.Timestamp(fold["trained_through"], tz="UTC")
        if len(rows) != int(fold["scored_rows"]):
            raise AssertionError("movement fold row count differs")
        if not pd.to_datetime(rows["p_move__trained_through"], utc=True).eq(cutoff).all():
            raise AssertionError("movement probability cutoff differs")
        if (
            not pd.to_datetime(rows["predicted_absolute_movement_bps__trained_through"], utc=True)
            .eq(cutoff)
            .all()
        ):
            raise AssertionError("movement-size cutoff differs")
    development["movement_oof"] = (
        development["p_move"].notna() & development["predicted_absolute_movement_bps"].notna()
    )
    assessment["movement_oof"] = False
    development["movement_trained_through"] = pd.to_datetime(
        development.pop("p_move__trained_through"), utc=True, errors="coerce"
    )
    development["movement_size_trained_through"] = pd.to_datetime(
        development.pop("predicted_absolute_movement_bps__trained_through"),
        utc=True,
        errors="coerce",
    )
    final_cutoff = pd.Timestamp(chronology["final_trained_through"])
    assessment["movement_trained_through"] = final_cutoff
    assessment["movement_size_trained_through"] = final_cutoff
    development["movement_frozen_before_outcome"] = False
    assessment["movement_frozen_before_outcome"] = True
    frame = pd.concat([development, assessment], ignore_index=True)
    frame["stock"] = frame.pop("symbol").astype(str)
    frame["fixed_clock_timestamp"] = pd.to_datetime(
        frame.pop("feature_available_timestamp_utc"), utc=True, errors="raise"
    )
    frame["movement_horizon_terminal_timestamp"] = frame["fixed_clock_timestamp"] + pd.Timedelta(
        minutes=120
    )
    frame["current_state_b"] = frame["origin_state"].map(mapping).astype(int)
    frame["representation_id"] = REPRESENTATION_ID
    frame["source_lineage_id"] = evidence["lineage_id"]
    frame["movement_chronology_evidence_id"] = evidence["movement_chronology_evidence_id"]
    if not frame["state_model_hash"].astype(str).eq(MODEL_HASH).all():
        raise AssertionError("movement state model differs")
    frame["source_gap"] = (
        frame["source_gap_crossed"].astype(bool)
        | frame["session_boundary_crossed"].astype(bool)
        | ~frame["history_ordinals_contiguous"].astype(bool)
        | ~frame["future_ordinals_contiguous"].astype(bool)
        | ~frame["session_source_complete"].astype(bool)
    )
    frame["movement_available"] = (
        frame["p_move"].notna()
        & frame["predicted_absolute_movement_bps"].notna()
        & frame["large_move"].notna()
        & ~frame["source_gap"]
    )
    frame["movement_row_id"] = [
        _row_id((MODEL_HASH, stock, session, ordinal, timestamp.isoformat()))
        for stock, session, ordinal, timestamp in frame[
            ["stock", "session", "decision_ordinal", "fixed_clock_timestamp"]
        ].itertuples(index=False, name=None)
    ]
    return frame.sort_values(
        ["session", "decision_ordinal", "stock"], kind="mergesort"
    ).reset_index(drop=True)


def _prediction_pair(path: Path, year: int, evidence: dict[str, Any]) -> pd.DataFrame:
    start, end = (
        (START_SESSION, "2025-01-01") if year == 2024 else ("2025-01-01", END_SESSION_EXCLUSIVE)
    )
    long = pd.read_parquet(
        path,
        columns=[
            "decision_id",
            "representation",
            "session",
            "score_month",
            "evaluation_period",
            "model",
            "probability",
            "training_rows",
        ],
        filters=[
            ("representation", "==", REPRESENTATION),
            ("session", ">=", start),
            ("session", "<", end),
        ],
    )
    long = long.loc[long["model"].isin(["M2_IMMEDIATE_PAIR", "M5_LAST_FIVE_STATES"])]
    chronology = evidence["closure_chronology"]
    expected_period = chronology[
        "development_evaluation_period" if year == 2024 else "assessment_evaluation_period"
    ]
    if (
        not long["evaluation_period"].astype(str).eq(expected_period).all()
        or not long["score_month"].astype(str).eq(long["session"].astype(str).str[:7]).all()
        or not long.groupby("decision_id")["model"].nunique().eq(2).all()
        or long.groupby("decision_id")[["score_month", "evaluation_period", "training_rows"]]
        .nunique()
        .gt(1)
        .any()
        .any()
    ):
        raise AssertionError("closure prediction chronology metadata differs")
    if (
        year == 2025
        and not pd.to_numeric(long["training_rows"])
        .eq(int(chronology["assessment_training_rows"]))
        .all()
    ):
        raise AssertionError("closure 2025 training count differs")
    probability = long.pivot(index="decision_id", columns="model", values="probability")
    probability = probability.rename(
        columns={
            "M2_IMMEDIATE_PAIR": "p_close_m2",
            "M5_LAST_FIVE_STATES": "p_close_m5",
        }
    )
    metadata = long.groupby("decision_id", sort=True).agg(
        score_month=("score_month", "first"),
        closure_training_rows=("training_rows", "min"),
    )
    result = metadata.join(probability, how="inner").reset_index()
    if year == 2024:
        result["closure_trained_through"] = pd.to_datetime(
            result["score_month"] + "-01", utc=True
        ) - pd.Timedelta(nanoseconds=1)
    else:
        result["closure_trained_through"] = pd.Timestamp(chronology["final_trained_through"])
    result["closure_m2_oof"] = year == 2024
    result["closure_m5_oof"] = year == 2024
    result["closure_frozen_before_outcome"] = year == 2025
    result["closure_chronology_evidence_id"] = evidence["closure_chronology_evidence_id"]
    return result


def _read_closure(evidence: dict[str, Any]) -> pd.DataFrame:
    population = pd.read_parquet(
        CLOSURE_PRIMARY / "pair_closure_population.parquet",
        columns=[
            "decision_id",
            "representation",
            "symbol",
            "session",
            "segment_id",
            "decision_timestamp",
            "target_available_timestamp",
            "current_state",
            "previous_state_1",
            "target_available",
            "target_pair_closure",
            "next_state",
            "censor_reason",
            "source_provider",
            "source_artifact",
            "source_hash",
            "data_snapshot_hash",
            "period",
        ],
        filters=[
            ("representation", "==", REPRESENTATION),
            ("session", ">=", START_SESSION),
            ("session", "<", END_SESSION_EXCLUSIVE),
        ],
    )
    predictions = pd.concat(
        [
            _prediction_pair(
                CLOSURE_PRIMARY / "development_oof_predictions.parquet", 2024, evidence
            ),
            _prediction_pair(CLOSURE_PRIMARY / "assessment_predictions.parquet", 2025, evidence),
        ],
        ignore_index=True,
    )
    frame = population.merge(predictions, on="decision_id", how="left", validate="one_to_one")
    closure_source = evidence["closure_source"]
    for year, key, period in (
        (2024, "development", "DEVELOPMENT_2024"),
        (2025, "assessment", "ASSESSMENT_2025"),
    ):
        mask = frame["session"].astype(str).str[:4].eq(str(year))
        identity = closure_source[key]
        expected_hash = frame.loc[mask, "symbol"].map(identity["source_hashes"])
        expected_artifact = "symbol=" + frame.loc[mask, "symbol"] + "/timeframe=5m/data.parquet"
        if (
            expected_hash.isna().any()
            or not frame.loc[mask, "source_provider"].eq("EODHD").all()
            or not frame.loc[mask, "source_artifact"].eq(expected_artifact).all()
            or not frame.loc[mask, "source_hash"].eq(expected_hash).all()
            or not frame.loc[mask, "data_snapshot_hash"].eq(identity["data_snapshot_hash"]).all()
            or not frame.loc[mask, "period"].eq(period).all()
        ):
            raise AssertionError("closure source lineage differs")
    frame["pair_forecast_id"] = frame.pop("decision_id").astype(str)
    frame["stock"] = frame.pop("symbol").astype(str)
    frame["pair_forecast_timestamp"] = pd.to_datetime(
        frame.pop("decision_timestamp"), utc=True, errors="raise"
    )
    frame["closure_resolution_timestamp"] = pd.to_datetime(
        frame.pop("target_available_timestamp"), utc=True, errors="coerce"
    )
    frame["current_state_b"] = frame.pop("current_state").astype(int)
    frame["pair_orientation"] = (
        frame["previous_state_1"].astype(int).astype(str)
        + "->"
        + frame["current_state_b"].astype(str)
    )
    frame["immediate_pair_closure"] = frame.pop("target_pair_closure")
    frame["closure_available"] = frame.pop("target_available").astype(bool)
    frame["source_gap"] = frame["censor_reason"].astype(str).eq("UNAVAILABLE_STRUCTURAL_GAP")
    frame["representation_id"] = REPRESENTATION_ID
    frame["source_lineage_id"] = evidence["lineage_id"]
    for column in ("closure_m2_oof", "closure_m5_oof", "closure_frozen_before_outcome"):
        frame[column] = frame[column].eq(True)
    available = frame["closure_available"]
    expected_target = frame["next_state"].eq(frame["previous_state_1"]).astype(int)
    if not expected_target.loc[available].equals(
        frame.loc[available, "immediate_pair_closure"].astype(int)
    ):
        raise AssertionError("closure target differs from next-state A semantics")
    return frame.sort_values(
        ["session", "stock", "pair_forecast_timestamp", "pair_forecast_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _independent_join(
    movement: pd.DataFrame, closure: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    groups = {
        (str(stock), str(session)): group
        for (stock, session), group in closure.groupby(["stock", "session"], sort=True)
    }
    accounting = {
        "movement_rows_inspected": len(movement),
        "closure_forecasts_inspected": len(closure),
        "exact_joined_rows": 0,
        "excluded_resolved_before_clock": 0,
        "excluded_no_active_pair": 0,
        "excluded_representation_mismatch": 0,
        "excluded_source_gap": 0,
        "excluded_duplicate_later_clock": 0,
        "excluded_closure_unavailable": 0,
        "excluded_movement_target_unavailable": 0,
    }
    rows: list[dict[str, Any]] = []
    for row in movement.itertuples(index=False):
        movement_row = row._asdict()
        if not bool(movement_row["movement_available"]):
            accounting["excluded_movement_target_unavailable"] += 1
            continue
        pairs = groups.get((str(row.stock), str(row.session)))
        if pairs is None:
            accounting["excluded_no_active_pair"] += 1
            continue
        created = pairs.loc[pairs["pair_forecast_timestamp"].le(row.fixed_clock_timestamp)]
        if created.empty:
            accounting["excluded_no_active_pair"] += 1
            continue
        active = created.loc[
            created["closure_resolution_timestamp"].isna()
            | created["closure_resolution_timestamp"].gt(row.fixed_clock_timestamp)
        ]
        if active.empty:
            accounting["excluded_resolved_before_clock"] += 1
            continue
        identity = active.loc[
            active["representation_id"].eq(row.representation_id)
            & active["source_lineage_id"].eq(row.source_lineage_id)
        ]
        if identity.empty:
            accounting["excluded_representation_mismatch"] += 1
            continue
        if bool(row.source_gap):
            accounting["excluded_source_gap"] += 1
            continue
        gap_free = identity.loc[~identity["source_gap"]]
        if gap_free.empty:
            accounting["excluded_source_gap"] += 1
            continue
        exact = gap_free.loc[
            gap_free["segment_id"].eq(row.origin_segment_id)
            & gap_free["current_state_b"].eq(row.current_state_b)
        ]
        if exact.empty:
            accounting["excluded_no_active_pair"] += 1
            continue
        exact = exact.loc[
            exact["closure_available"]
            & exact["closure_resolution_timestamp"].notna()
            & exact["immediate_pair_closure"].notna()
        ]
        if exact.empty:
            accounting["excluded_closure_unavailable"] += 1
            continue
        if len(exact) != 1:
            raise AssertionError("more than one active pair")
        pair = exact.iloc[0]
        age = (row.fixed_clock_timestamp - pair["pair_forecast_timestamp"]) / pd.Timedelta(
            minutes=5
        )
        if not float(age).is_integer():
            raise AssertionError("non-integral pair age")
        rows.append(
            {
                **movement_row,
                "joined_row_id": _row_id(
                    (row.representation_id, row.stock, row.session, pair["pair_forecast_id"])
                ),
                "pair_forecast_id": pair["pair_forecast_id"],
                "pair_forecast_timestamp": pair["pair_forecast_timestamp"],
                "closure_resolution_timestamp": pair["closure_resolution_timestamp"],
                "pair_age_bars": int(age),
                "pair_orientation": pair["pair_orientation"],
                "p_close_m2": pair["p_close_m2"],
                "p_close_m5": pair["p_close_m5"],
                "immediate_pair_closure": int(pair["immediate_pair_closure"]),
                "closure_available": True,
                "closure_m2_oof": bool(pair["closure_m2_oof"]),
                "closure_m5_oof": bool(pair["closure_m5_oof"]),
                "closure_trained_through": pair["closure_trained_through"],
                "closure_frozen_before_outcome": bool(pair["closure_frozen_before_outcome"]),
                "closure_chronology_evidence_id": pair["closure_chronology_evidence_id"],
            }
        )
    joined = pd.DataFrame(rows).sort_values(
        [
            "representation_id",
            "stock",
            "session",
            "pair_forecast_id",
            "fixed_clock_timestamp",
        ],
        kind="mergesort",
    )
    duplicate = joined.duplicated(
        ["representation_id", "stock", "session", "pair_forecast_id"], keep="first"
    )
    accounting["excluded_duplicate_later_clock"] = int(duplicate.sum())
    joined = joined.loc[~duplicate].copy()
    accounting["exact_joined_rows"] = len(joined)
    joined["logit_p_move"] = _logit(joined["p_move"])
    joined["logit_p_close_m2"] = _logit(joined["p_close_m2"])
    joined["logit_p_close_m5"] = _logit(joined["p_close_m5"])
    joined["closure_history_increment"] = joined["logit_p_close_m5"] - joined["logit_p_close_m2"]
    joined["log1p_predicted_absolute_movement_bps"] = np.log1p(
        joined["predicted_absolute_movement_bps"]
    )
    joined["joint_large_move_and_closure"] = (
        joined["large_move"].astype(bool) & joined["immediate_pair_closure"].astype(bool)
    ).astype(np.int8)
    joined["slate_id"] = joined["session"] + "|" + joined["decision_ordinal"].astype(str)
    sizes = joined.groupby("slate_id")["slate_id"].transform("size")
    joined["joined_slate_size"] = sizes.astype(int)
    joined["row_weight"] = 1.0 / sizes
    joined["year"] = joined["session"].str[:4].astype(int)
    columns = [
        "joined_row_id",
        "pair_forecast_id",
        "movement_row_id",
        "representation_id",
        "source_lineage_id",
        "stock",
        "session",
        "year",
        "decision_ordinal",
        "slate_id",
        "pair_forecast_timestamp",
        "fixed_clock_timestamp",
        "pair_age_bars",
        "scheduled_bars_remaining",
        "closure_resolution_timestamp",
        "movement_horizon_terminal_timestamp",
        "pair_orientation",
        "p_move",
        "predicted_absolute_movement_bps",
        "p_close_m2",
        "p_close_m5",
        "logit_p_move",
        "logit_p_close_m2",
        "logit_p_close_m5",
        "closure_history_increment",
        "log1p_predicted_absolute_movement_bps",
        "large_move",
        "immediate_pair_closure",
        "joint_large_move_and_closure",
        "closure_available",
        "movement_available",
        "movement_oof",
        "closure_m2_oof",
        "closure_m5_oof",
        "movement_trained_through",
        "movement_size_trained_through",
        "closure_trained_through",
        "movement_frozen_before_outcome",
        "closure_frozen_before_outcome",
        "movement_chronology_evidence_id",
        "closure_chronology_evidence_id",
        "joined_slate_size",
        "row_weight",
    ]
    return (
        joined.loc[:, columns]
        .sort_values(["session", "decision_ordinal", "stock", "pair_forecast_id"])
        .reset_index(drop=True),
        accounting,
    )


def _frame_close(
    left: pd.DataFrame, right: pd.DataFrame, tolerance: float = 1e-12
) -> tuple[bool, float]:
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return False, float("inf")
    maximum = 0.0
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]):
            first = left[column].to_numpy(dtype=float)
            second = right[column].to_numpy(dtype=float)
            difference = np.abs(first - second)
            finite = difference[np.isfinite(difference)]
            maximum = max(maximum, float(finite.max()) if len(finite) else 0.0)
            if not np.allclose(first, second, rtol=0.0, atol=tolerance, equal_nan=True):
                return False, maximum
        elif not left[column].equals(right[column]):
            return False, float("inf")
    return True, maximum


def _manual_predictions(frame: pd.DataFrame, stored: dict[str, Any]) -> np.ndarray[Any, Any]:
    names = stored["feature_names"]
    values = frame[names].to_numpy(dtype=float)
    standardized = (values - np.asarray(stored["means"])) / np.asarray(stored["scales"])
    linear = float(stored["intercept"]) + standardized @ np.asarray(stored["coefficients"])
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0)))


def _weights(frame: pd.DataFrame) -> np.ndarray[Any, Any]:
    return frame["row_weight"].to_numpy(dtype=float)


def _paired(frame: pd.DataFrame, target: str, baseline: str, candidate: str) -> tuple[float, float]:
    labels = frame[target].to_numpy(dtype=float)
    base = np.clip(frame[baseline].to_numpy(dtype=float), EPSILON, 1.0 - EPSILON)
    cand = np.clip(frame[candidate].to_numpy(dtype=float), EPSILON, 1.0 - EPSILON)
    weight = _weights(frame)
    brier = np.average((base - labels) ** 2 - (cand - labels) ** 2, weights=weight)
    base_log = -(labels * np.log(base) + (1.0 - labels) * np.log1p(-base))
    cand_log = -(labels * np.log(cand) + (1.0 - labels) * np.log1p(-cand))
    return float(brier), float(np.average(base_log - cand_log, weights=weight))


def _calibration(
    labels: np.ndarray[Any, Any], probability: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]
) -> tuple[float, float]:
    design = np.column_stack((np.ones(len(probability)), _logit(probability)))
    beta = np.asarray([0.0, 1.0])
    for _ in range(100):
        fitted = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -35.0, 35.0)))
        gradient = design.T @ (weights * (labels - fitted))
        variance = weights * fitted * (1.0 - fitted)
        information = design.T @ (design * variance[:, None]) + np.eye(2) * 1e-10
        step = np.linalg.solve(information, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def _metric_row(frame: pd.DataFrame, target: str, probability: str, model: str) -> dict[str, Any]:
    labels = frame[target].to_numpy(dtype=float)
    prediction = frame[probability].to_numpy(dtype=float)
    clipped = np.clip(prediction, EPSILON, 1.0 - EPSILON)
    weight = _weights(frame)
    intercept, slope = _calibration(labels, prediction, weight)
    bins = np.minimum((prediction * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += (
                weight[mask].sum()
                / weight.sum()
                * abs(
                    np.average(prediction[mask], weights=weight[mask])
                    - np.average(labels[mask], weights=weight[mask])
                )
            )
    return {
        "model": model,
        "target": target,
        "brier_score": float(np.average((prediction - labels) ** 2, weights=weight)),
        "log_loss": float(
            np.average(
                -(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped)),
                weights=weight,
            )
        ),
        "auc": float(roc_auc_score(labels, prediction, sample_weight=weight)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": float(ece),
        "outcome_rate": float(np.average(labels, weights=weight)),
        "mean_prediction": float(np.average(prediction, weights=weight)),
        "row_count": len(frame),
        "session_count": frame["session"].nunique(),
        "stock_count": frame["stock"].nunique(),
    }


def _bootstrap(frame: pd.DataFrame, target: str, baseline: str, candidate: str) -> pd.DataFrame:
    sessions = np.asarray(sorted(frame["session"].unique()), dtype=object)
    rng = np.random.default_rng(20260720)
    rows: list[dict[str, Any]] = []
    for draw in range(500):
        sampled = rng.choice(sessions, len(sessions), replace=True)
        parts: list[pd.DataFrame] = []
        for occurrence, session in enumerate(sampled):
            part = frame.loc[frame["session"].eq(session)].copy()
            part["slate_id"] = str(occurrence) + "|" + part["slate_id"]
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        size = sample.groupby("slate_id")["slate_id"].transform("size")
        sample["row_weight"] = 1.0 / size
        brier, log_loss = _paired(sample, target, baseline, candidate)
        rows.extend(
            [
                {"draw": draw, "metric": "brier", "improvement": brier},
                {"draw": draw, "metric": "log_loss", "improvement": log_loss},
            ]
        )
    return pd.DataFrame(rows)


def _shift(
    frame: pd.DataFrame, columns: Sequence[str], draw: int, seed: int
) -> tuple[pd.DataFrame, int, int]:
    output = frame.copy()
    rng = np.random.default_rng(seed + draw)
    grouped: dict[tuple[int, tuple[str, ...]], list[str]] = {}
    for (ordinal, session), slate in frame.groupby(["decision_ordinal", "session"], sort=True):
        membership = tuple(sorted(slate["stock"].astype(str)))
        grouped.setdefault((int(ordinal), membership), []).append(str(session))
    assignments = 0
    non_identity = 0
    for (ordinal, _membership), sessions_raw in sorted(grouped.items()):
        sessions = sorted(set(sessions_raw))
        offset = 0 if len(sessions) <= 1 else int(rng.integers(1, len(sessions)))
        for position, destination in enumerate(sessions):
            source = sessions[(position - offset) % len(sessions)]
            destination_index = frame.index[
                frame["decision_ordinal"].eq(ordinal) & frame["session"].eq(destination)
            ]
            source_block = frame.loc[
                frame["decision_ordinal"].eq(ordinal) & frame["session"].eq(source),
                ["stock", *columns],
            ].set_index("stock")
            stocks = frame.loc[destination_index, "stock"].astype(str)
            output.loc[destination_index, list(columns)] = source_block.loc[
                stocks, list(columns)
            ].to_numpy()
            assignments += 1
            non_identity += source != destination
    return output, assignments, non_identity


def _fit_null(
    frame: pd.DataFrame, target: str, features: Sequence[str], seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], LogisticRegression]:
    values = frame[list(features)].to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales = np.where(scales >= 1e-12, scales, 1.0)
    design = (values - means) / scales
    size = frame.groupby("slate_id")["slate_id"].transform("size")
    estimator = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=seed,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module=r"sklearn\.linear_model\._logistic"
        )
        estimator.fit(design, frame[target].to_numpy(dtype=int), sample_weight=1.0 / size)
    if int(estimator.n_iter_.max()) >= 250:
        raise AssertionError("null model did not converge")
    return means, scales, estimator


def _nulls(development: pd.DataFrame, assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for draw in range(100):
        dev, a1, n1 = _shift(
            development, ("p_move", "predicted_absolute_movement_bps"), draw, 20260721
        )
        test, a2, n2 = _shift(
            assessment, ("p_move", "predicted_absolute_movement_bps"), draw, 20260721
        )
        for frame in (dev, test):
            frame["logit_p_move"] = _logit(frame["p_move"])
            frame["log1p_predicted_absolute_movement_bps"] = np.log1p(
                frame["predicted_absolute_movement_bps"]
            )
        means, scales, model = _fit_null(
            dev, "immediate_pair_closure", A3_FEATURES, 20260721 + draw
        )
        design = (test[list(A3_FEATURES)].to_numpy(float) - means) / scales
        test["candidate"] = model.predict_proba(design)[:, 1]
        brier, log_loss = _paired(test, "immediate_pair_closure", "p_A2", "candidate")
        rows.append(
            {
                "null": "closure_increment",
                "draw": draw,
                "brier_improvement": brier,
                "log_loss_improvement": log_loss,
                "block_assignments": a1 + a2,
                "non_identity_assignments": n1 + n2,
            }
        )
    for draw in range(100):
        columns = ("p_close_m2", "p_close_m5", "closure_history_increment")
        dev, a1, n1 = _shift(development, columns, draw, 20260722)
        test, a2, n2 = _shift(assessment, columns, draw, 20260722)
        for frame in (dev, test):
            frame["logit_p_close_m2"] = _logit(frame["p_close_m2"])
            frame["logit_p_close_m5"] = _logit(frame["p_close_m5"])
            if not np.allclose(
                frame["closure_history_increment"],
                frame["logit_p_close_m5"] - frame["logit_p_close_m2"],
                atol=1e-12,
            ):
                raise AssertionError("null closure block was split")
        means, scales, model = _fit_null(dev, "large_move", B1_FEATURES, 20260722 + draw)
        design = (test[list(B1_FEATURES)].to_numpy(float) - means) / scales
        test["candidate"] = model.predict_proba(design)[:, 1]
        brier, log_loss = _paired(test, "large_move", "p_B0", "candidate")
        rows.append(
            {
                "null": "movement_increment",
                "draw": draw,
                "brier_improvement": brier,
                "log_loss_improvement": log_loss,
                "block_assignments": a1 + a2,
                "non_identity_assignments": n1 + n2,
            }
        )
    return pd.DataFrame(rows).sort_values(["null", "draw"]).reset_index(drop=True)


def _null_feasibility(
    development: pd.DataFrame, assessment: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    summary: dict[str, Any] = {}
    for label, frame in (("development_2024", development), ("assessment_2025", assessment)):
        groups: dict[tuple[int, tuple[str, ...]], list[str]] = {}
        for (ordinal, session), slate in frame.groupby(
            ["decision_ordinal", "session"], sort=True, observed=True
        ):
            membership = tuple(sorted(slate["stock"].astype(str)))
            groups.setdefault((int(ordinal), membership), []).append(str(session))
        rows: list[dict[str, Any]] = []
        for (ordinal, membership), raw_sessions in sorted(groups.items()):
            sessions = sorted(set(raw_sessions))
            shiftable = len(sessions) if len(sessions) > 1 else 0
            rows.append(
                {
                    "population": label,
                    "decision_ordinal": ordinal,
                    "membership_hash": hashlib.sha256(
                        "|".join(membership).encode("utf-8")
                    ).hexdigest()[:16],
                    "membership_size": len(membership),
                    "session_blocks": len(sessions),
                    "shiftable_blocks": shiftable,
                    "unshiftable_blocks": len(sessions) - shiftable,
                }
            )
        part = pd.DataFrame(rows).sort_values(
            ["decision_ordinal", "membership_hash"], kind="mergesort"
        )
        parts.append(part)
        summary[label] = {
            "session_blocks": int(part["session_blocks"].sum()),
            "shiftable_blocks": int(part["shiftable_blocks"].sum()),
            "unshiftable_blocks": int(part["unshiftable_blocks"].sum()),
            "membership_groups": int(len(part)),
        }
    result = pd.concat(parts, ignore_index=True)
    result.insert(0, "null", "whole_session_shift_feasibility")
    result.insert(1, "draw", -1)
    return result, summary


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def audit(artifacts: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: Any = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "details": details})

    contract = json.loads((EXPERIMENT_DIR / "contract.json").read_text())
    decision = json.loads((artifacts / "decision.json").read_text())
    record(
        "contract_safety_flags", all(contract.get(key) == value for key, value in SAFETY.items())
    )
    record(
        "decision_safety_flags", all(decision.get(key) == value for key, value in SAFETY.items())
    )

    input_manifest = json.loads((artifacts / "input_artifact_hashes.json").read_text())
    hash_results = []
    for item in input_manifest["artifacts"]:
        path = REPO_ROOT / item["repository_relative_path"]
        actual = _sha256(path)
        hash_results.append(actual == item["sha256"])
    record("input_artifact_hashes", all(hash_results), {"artifacts": len(hash_results)})

    mapping_frame = pd.read_csv(MAPPING_PATH)
    mapping = dict(
        zip(
            mapping_frame["raw_cluster_state"].astype(int),
            mapping_frame["semantic_state"].astype(int),
            strict=True,
        )
    )
    evidence = _upstream_evidence(_sha256(MAPPING_PATH))
    source_manifest = _json(artifacts / "source_manifest.json")
    record(
        "hash_bound_chronology_and_lineage_evidence",
        source_manifest.get("source_lineage_id") == evidence["lineage_id"]
        and source_manifest.get("lineage_reconciliation") == evidence["lineage_evidence"]
        and source_manifest.get("chronology_evidence", {}).get("movement")
        == evidence["movement_chronology"]
        and source_manifest.get("chronology_evidence", {}).get("closure")
        == evidence["closure_chronology"],
        {
            "movement_evidence_id": evidence["movement_chronology_evidence_id"],
            "closure_evidence_id": evidence["closure_chronology_evidence_id"],
            "source_lineage_id": evidence["lineage_id"],
        },
    )
    movement = _read_movement(mapping, evidence)
    closure = _read_closure(evidence)
    reconstructed, accounting = _independent_join(movement, closure)
    stored = pd.read_parquet(artifacts / "joined_compact_panel.parquet")
    comparable, maximum = _frame_close(reconstructed, stored)
    record("exact_causal_join_reconstructed", comparable, {"maximum_difference": maximum})
    stored_accounting = (
        pd.read_csv(artifacts / "join_accounting.csv").set_index("category")["count"].to_dict()
    )
    record("join_accounting", accounting == stored_accounting, accounting)
    record(
        "same_stock_session_representation",
        stored["representation_id"].eq(REPRESENTATION_ID).all()
        and stored["source_lineage_id"].eq(evidence["lineage_id"]).all(),
    )
    record(
        "pair_unresolved_at_fixed_clock",
        stored["pair_forecast_timestamp"].le(stored["fixed_clock_timestamp"]).all()
        and stored["closure_resolution_timestamp"].gt(stored["fixed_clock_timestamp"]).all(),
    )
    age = (stored["fixed_clock_timestamp"] - stored["pair_forecast_timestamp"]) / pd.Timedelta(
        minutes=5
    )
    record("pair_age_calculation", np.array_equal(age.astype(int), stored["pair_age_bars"]))
    record(
        "earliest_clock_deduplication",
        not stored.duplicated(["representation_id", "stock", "session", "pair_forecast_id"]).any(),
    )
    record("no_source_gap_crossing", accounting["excluded_source_gap"] >= 0)

    date_audit = json.loads((artifacts / "date_boundary_audit.json").read_text())
    all_dates = pd.concat(
        [
            movement["fixed_clock_timestamp"],
            movement["movement_horizon_terminal_timestamp"],
            closure["pair_forecast_timestamp"],
            closure["closure_resolution_timestamp"],
            stored["pair_forecast_timestamp"],
            stored["fixed_clock_timestamp"],
            stored["closure_resolution_timestamp"],
            stored["movement_horizon_terminal_timestamp"],
        ],
        ignore_index=True,
    ).dropna()
    all_dates = pd.to_datetime(all_dates, utc=True)
    record(
        "protected_date_boundary",
        all_dates.lt(pd.Timestamp("2025-08-23T00:00:00Z")).all()
        and date_audit["protected_rows_opened"] == 0
        and date_audit["minimum_timestamp_read"] == all_dates.min().isoformat()
        and date_audit["maximum_timestamp_read"] == all_dates.max().isoformat(),
        {
            "minimum_timestamp_read": all_dates.min().isoformat(),
            "maximum_timestamp_read": all_dates.max().isoformat(),
        },
    )
    record(
        "probability_logits",
        np.allclose(stored["logit_p_move"], _logit(stored["p_move"]), atol=1e-14)
        and np.allclose(stored["logit_p_close_m2"], _logit(stored["p_close_m2"]), atol=1e-14)
        and np.allclose(stored["logit_p_close_m5"], _logit(stored["p_close_m5"]), atol=1e-14),
    )
    record(
        "closure_history_increment",
        np.allclose(
            stored["closure_history_increment"],
            stored["logit_p_close_m5"] - stored["logit_p_close_m2"],
            atol=1e-14,
        ),
    )
    record(
        "joint_target",
        np.array_equal(
            stored["joint_large_move_and_closure"],
            (
                stored["large_move"].astype(bool) & stored["immediate_pair_closure"].astype(bool)
            ).astype(int),
        ),
    )
    development = stored.loc[stored["year"].eq(2024)].copy()
    assessment = pd.read_parquet(artifacts / "assessment_predictions.parquet")
    chronology = bool(
        development[["movement_oof", "closure_m2_oof", "closure_m5_oof"]].astype(bool).all().all()
        and pd.to_datetime(development["movement_trained_through"], utc=True)
        .lt(pd.to_datetime(development["session"], utc=True))
        .all()
        and pd.to_datetime(development["movement_size_trained_through"], utc=True)
        .lt(pd.to_datetime(development["session"], utc=True))
        .all()
        and pd.to_datetime(development["closure_trained_through"], utc=True)
        .lt(pd.to_datetime(development["session"], utc=True))
        .all()
        and assessment["movement_frozen_before_outcome"].all()
        and assessment["closure_frozen_before_outcome"].all()
        and development["movement_chronology_evidence_id"]
        .eq(evidence["movement_chronology_evidence_id"])
        .all()
        and development["closure_chronology_evidence_id"]
        .eq(evidence["closure_chronology_evidence_id"])
        .all()
        and assessment["movement_chronology_evidence_id"]
        .eq(evidence["movement_chronology_evidence_id"])
        .all()
        and assessment["closure_chronology_evidence_id"]
        .eq(evidence["closure_chronology_evidence_id"])
        .all()
    )
    record("upstream_2024_oof_and_2025_frozen", chronology)

    coefficients = json.loads((artifacts / "model_coefficients.json").read_text())["models"]
    prediction_checks = []
    for model in ("A2", "A3", "B0", "B1", "C1"):
        stored_model = coefficients[model]
        names = stored_model["feature_names"]
        values = development[names].to_numpy(dtype=float)
        means = values.mean(axis=0)
        scales = np.where(values.std(axis=0, ddof=0) >= 1e-12, values.std(axis=0), 1.0)
        training_only = np.allclose(means, stored_model["means"], atol=1e-14) and np.allclose(
            scales, stored_model["scales"], atol=1e-14
        )
        prediction = _manual_predictions(assessment, stored_model)
        prediction_checks.append(
            training_only
            and np.allclose(prediction, assessment[f"p_{model}"], rtol=0.0, atol=1e-14)
        )
    direct = bool(
        np.array_equal(assessment["p_A0"], assessment["p_close_m2"])
        and np.array_equal(assessment["p_A1"], assessment["p_close_m5"])
        and np.allclose(assessment["p_C0"], assessment["p_move"] * assessment["p_close_m5"])
    )
    record("training_only_scaling_and_manual_predictions", all(prediction_checks) and direct)

    metric_specs = {
        "closure_metrics.csv": [
            ("A0", "immediate_pair_closure", "p_A0"),
            ("A1", "immediate_pair_closure", "p_A1"),
            ("A2", "immediate_pair_closure", "p_A2"),
            ("A3", "immediate_pair_closure", "p_A3"),
        ],
        "movement_metrics.csv": [("B0", "large_move", "p_B0"), ("B1", "large_move", "p_B1")],
        "joint_metrics.csv": [
            ("C0", "joint_large_move_and_closure", "p_C0"),
            ("C1", "joint_large_move_and_closure", "p_C1"),
        ],
    }
    metric_ok = True
    metric_max = 0.0
    for filename, specs in metric_specs.items():
        expected = pd.DataFrame(
            [
                _metric_row(assessment, target, probability, model)
                for model, target, probability in specs
            ]
        )
        actual = pd.read_csv(artifacts / filename)
        common = list(expected.columns)
        okay, difference = _frame_close(expected, actual[common], tolerance=1e-12)
        metric_ok &= okay
        metric_max = max(metric_max, difference)
    record("brier_log_loss_auc_calibration", metric_ok, {"maximum_difference": metric_max})

    bootstrap_expected = []
    for comparison, target, baseline, candidate in (
        ("A3_vs_A2", "immediate_pair_closure", "p_A2", "p_A3"),
        ("B1_vs_B0", "large_move", "p_B0", "p_B1"),
        ("C1_vs_C0", "joint_large_move_and_closure", "p_C0", "p_C1"),
    ):
        part = _bootstrap(assessment, target, baseline, candidate)
        part.insert(0, "comparison", comparison)
        bootstrap_expected.append(part)
    bootstrap_expected_frame = pd.concat(bootstrap_expected, ignore_index=True)
    bootstrap_actual = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    bootstrap_columns = ["comparison", "draw", "metric", "improvement"]
    bootstrap_ok, bootstrap_max = _frame_close(
        bootstrap_expected_frame[bootstrap_columns],
        bootstrap_actual[bootstrap_columns],
        tolerance=1e-12,
    )
    record("session_block_bootstrap", bootstrap_ok, {"maximum_difference": bootstrap_max})

    null_expected, feasibility_summary = _null_feasibility(development, assessment)
    null_actual = pd.read_csv(artifacts / "null_metrics.csv")
    null_ok, null_max = _frame_close(null_expected, null_actual, tolerance=1e-12)
    unshiftable = int(null_expected["unshiftable_blocks"].sum())
    record(
        "whole_session_null_fails_closed_without_identity_blocks",
        null_ok
        and unshiftable > 0
        and decision["null_summary"]["completed_draws_per_primary_increment"] == 0
        and decision["null_summary"]["feasibility"] == feasibility_summary,
        {
            "maximum_difference": null_max,
            "unshiftable_blocks": unshiftable,
            "feasibility": feasibility_summary,
        },
    )

    stock_max = assessment["stock"].value_counts(normalize=True).max()
    month_max = (
        pd.to_datetime(assessment["session"])
        .dt.strftime("%Y-%m")
        .value_counts(normalize=True)
        .max()
    )
    orientation_max = assessment["pair_orientation"].value_counts(normalize=True).max()
    concentration_pass = stock_max <= 0.125 and month_max <= 0.25 and orientation_max <= 0.20
    record(
        "support_and_concentration",
        concentration_pass
        and len(assessment) >= 1500
        and assessment["session"].nunique() >= 75
        and assessment["stock"].nunique() >= 15
        and assessment["immediate_pair_closure"].sum() >= 250
        and assessment["large_move"].sum() >= 250
        and assessment["joint_large_move_and_closure"].sum() >= 100,
        {"stock_max": stock_max, "month_max": month_max, "orientation_max": orientation_max},
    )
    record(
        "decision_logic",
        decision["decision"] == "blocked_join_semantics_failure"
        and decision["null_summary"]["status"] == "blocked_join_semantics_failure"
        and not decision["arm_a_pass"]
        and not decision["arm_b_pass"]
        and not decision["arm_c_pass"],
    )
    forbidden = (
        "signed_return",
        "pnl",
        "mfe",
        "mae",
        "long_probability",
        "short_probability",
        "payoff",
        "broker",
        "order_id",
        "exact_five_state",
        "exact_loop",
    )
    record(
        "forbidden_fields_absent",
        not any(fragment in column.lower() for column in stored.columns for fragment in forbidden),
    )
    passed = all(check["passed"] for check in checks)
    result = {
        **SAFETY,
        "auditor_imported_runner": False,
        "auditor_imported_reusable_joint_module": False,
        "passed": passed,
        "check_count": len(checks),
        "checks": checks,
    }
    (artifacts / "independent_audit.json").write_text(
        json.dumps(_safe_json(result), sort_keys=True, separators=(",", ":")) + "\n"
    )
    if not passed:
        failed = [check["check"] for check in checks if not check["passed"]]
        raise RuntimeError(f"blocked_reproducibility_or_audit_failure: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=EXPERIMENT_DIR / "artifacts/primary",
    )
    args = parser.parse_args()
    result = audit(args.artifacts.resolve())
    print(json.dumps({"independent_audit_passed": result["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
