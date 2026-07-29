#!/usr/bin/env python3
# ruff: noqa: E402 -- local package paths are installed before research imports.
"""Lightweight independent audit for the frozen hidden-loop quick screen V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
FROZEN_FAMILIES = (
    "unregistered_primitive_like__5-6-5",
    "unregistered_primitive_like__2-3-2",
    "unregistered_primitive_like__2-5-2",
    "unregistered_primitive_like__4-7-4",
)

PREDECESSOR_PRIMARY = (
    REPO_ROOT
    / "research"
    / "unregistered-loop-families"
    / "20260721-opening-trajectory-unregistered-families-v0"
    / "artifacts"
    / "primary"
)
PATH_LEDGER = PREDECESSOR_PRIMARY / "unregistered_path_ledger.parquet"
FAMILY_MAPPING = PREDECESSOR_PRIMARY / "hidden_family_mapping.json"
PREDECESSOR_COEFFICIENTS = PREDECESSOR_PRIMARY / "model_coefficients.json"
OPENING_PANEL = (
    REPO_ROOT
    / "research"
    / "behavioural-trajectory"
    / "20260721-behavioural-trajectory-late-loops-v01"
    / "artifacts"
    / "primary"
    / "decision_panel.parquet"
)
SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "frozen_hidden_families": True,
    "post_completion_economic_diagnostic": True,
    "registered_loop_bridge_test": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}
REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "hidden_event_population_reconstruction.json",
    "hidden_event_economic_ledger.parquet",
    "matched_control_ledger.parquet",
    "economic_metrics.csv",
    "family_economic_metrics.csv",
    "monthly_economic_metrics.csv",
    "economic_bootstrap_metrics.csv",
    "economic_multiplicity_results.csv",
    "hidden_to_registered_lead_ledger.parquet",
    "hidden_to_registered_transition_counts.csv",
    "hidden_to_registered_exact_pairs.csv",
    "structural_lead_null_metrics.csv",
    "bridge_target_manifest.json",
    "bridge_crossfit_manifest.json",
    "bridge_model_configurations.json",
    "bridge_model_coefficients.json",
    "bridge_assessment_predictions.parquet",
    "bridge_metrics.csv",
    "bridge_monthly_metrics.csv",
    "bridge_checkpoint_metrics.csv",
    "bridge_bootstrap_metrics.csv",
    "bridge_null_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "determinism_check.json",
    "report.md",
)


def binary_model_metrics(
    labels: pd.Series, probabilities: pd.Series, weights: pd.Series
) -> dict[str, float]:
    """Independently reproduce the four binding weighted bridge metrics."""

    target = labels.to_numpy(dtype=int)
    prediction = probabilities.to_numpy(dtype=float)
    sample_weight = weights.to_numpy(dtype=float)
    total_weight = float(sample_weight.sum())
    return {
        "log_loss": float(log_loss(target, prediction, sample_weight=sample_weight, labels=[0, 1])),
        "brier_score": float(np.sum(sample_weight * (prediction - target) ** 2) / total_weight),
        "auc": float(roc_auc_score(target, prediction, sample_weight=sample_weight)),
        "average_precision": float(
            average_precision_score(target, prediction, sample_weight=sample_weight)
        ),
    }


def session_block_bootstrap_indices(
    frame: pd.DataFrame, *, draws: int, seed: int
) -> list[np.ndarray]:
    sessions = np.asarray(sorted(frame["session"].astype(str).unique()), dtype=object)
    values = frame["session"].astype(str).to_numpy()
    positions = {session: np.flatnonzero(values == session) for session in sessions}
    generator = np.random.default_rng(seed)
    return [
        np.concatenate(
            [positions[str(session)] for session in generator.choice(sessions, len(sessions))]
        )
        for _ in range(draws)
    ]


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    for index in range(len(ranked) - 1, -1, -1):
        running = min(running, float(ranked[index]) * len(ranked) / (index + 1))
        adjusted[index] = min(1.0, running)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return restored.astype(float).tolist()


def registered_completion_targets(
    origin_bar_ordinal: int, completions: pd.DataFrame
) -> dict[str, bool | int | str | None]:
    frame = completions.loc[
        pd.to_numeric(completions["completion_bar_ordinal"]).gt(origin_bar_ordinal)
    ].copy()
    frame["bars_after_origin"] = (
        pd.to_numeric(frame["completion_bar_ordinal"]).astype(int) - origin_bar_ordinal
    )
    frame = frame.sort_values(["bars_after_origin", "semantic_loop_id"], kind="mergesort")
    first = frame.iloc[0] if not frame.empty else None
    return {
        "registered_within_6_bars": bool(frame["bars_after_origin"].le(6).any()),
        "registered_within_12_bars": bool(frame["bars_after_origin"].le(12).any()),
        "bars_to_first_registered_completion": (
            int(first["bars_after_origin"]) if first is not None else None
        ),
        "first_registered_semantic_loop_id": (
            str(first["semantic_loop_id"]) if first is not None else None
        ),
        "first_registered_motif_type": (str(first["motif_type"]) if first is not None else None),
    }


def expanding_logistic_crossfit(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    folds: int,
    warmup_fraction: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Independent four-fold chronological reconstruction of development U1."""

    sessions = np.asarray(sorted(frame["session"].astype(str).unique()), dtype=object)
    warmup_count = max(2, int(np.ceil(len(sessions) * warmup_fraction)))
    blocks = [block for block in np.array_split(sessions[warmup_count:], folds) if len(block)]
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    manifest: list[dict[str, Any]] = []
    session_values = frame["session"].astype(str)
    for fold, block in enumerate(blocks, start=1):
        prediction_start = str(block[0])
        train = frame.loc[session_values.lt(prediction_start) & frame[target].notna()]
        predict = frame.loc[session_values.isin([str(value) for value in block])]
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        train_matrix = scaler.fit_transform(train.loc[:, list(features)].to_numpy(dtype=float))
        estimator = LogisticRegression(
            penalty="l2",
            C=0.25,
            solver="liblinear",
            max_iter=300,
            class_weight=None,
            n_jobs=1,
            random_state=20260721,
        )
        labels = train[target].to_numpy(dtype=int)
        estimator.fit(
            train_matrix,
            labels,
            sample_weight=train["row_weight"].to_numpy(dtype=float),
        )
        predictions.loc[predict.index] = estimator.predict_proba(
            scaler.transform(predict.loc[:, list(features)].to_numpy(dtype=float))
        )[:, 1]
        manifest.append(
            {
                "fold": fold,
                "train_session_start": str(train["session"].min()),
                "train_session_end": str(train["session"].max()),
                "prediction_session_start": prediction_start,
                "prediction_session_end": str(block[-1]),
                "train_rows": len(train),
                "prediction_rows": len(predict),
                "positive_train_rows": int(labels.sum()),
            }
        )
    return predictions, pd.DataFrame(manifest)


def permute_feature_within_slates(frame: pd.DataFrame, *, feature: str, seed: int) -> pd.DataFrame:
    result = frame.copy()
    generator = np.random.default_rng(seed)
    values = result[feature].to_numpy(copy=True)
    for positions in result.groupby("slate_id", sort=True).indices.values():
        index = np.asarray(positions, dtype=int)
        values[index] = generator.permutation(values[index])
    result[feature] = values
    return result


def bridge_permutation_sha256(frame: pd.DataFrame) -> str:
    columns = [
        "symbol",
        "session",
        "decision_ordinal",
        "slate_id",
        "p_unregistered_within_6_bars",
    ]
    hashed = pd.util.hash_pandas_object(frame.loc[:, columns], index=False).to_numpy(
        dtype=np.uint64
    )
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def stock_clock_session_permutation(
    events: pd.DataFrame, eligible_sessions: pd.DataFrame, *, seed: int
) -> pd.DataFrame:
    result = events.copy()
    generator = np.random.default_rng(seed)
    group_columns = ["symbol", "clock_bin"]
    if "period" in events and "period" in eligible_sessions:
        group_columns.append("period")
    permuted = result["session"].astype(object).to_numpy(copy=True)
    for key, positions in result.groupby(group_columns, sort=True).indices.items():
        values = key if isinstance(key, tuple) else (key,)
        pool = eligible_sessions.copy()
        for column, value in zip(group_columns, values, strict=True):
            if column != "clock_bin":
                pool = pool.loc[pool[column].astype(str).eq(str(value))]
        sessions = np.asarray(sorted(pool["session"].astype(str).unique()), dtype=object)
        index = np.asarray(positions, dtype=int)
        permuted[index] = generator.permutation(sessions)[: len(index)]
    result["session"] = permuted
    return result


def choose_primary_decision(
    *, economic_status: str, registered_lead_status: str, predictive_bridge_status: str
) -> str:
    if economic_status == "supported" and predictive_bridge_status == "supported":
        return "hidden_loops_economic_and_registered_bridge_supported"
    if economic_status == "supported":
        return "hidden_loop_economic_consequence_only"
    if predictive_bridge_status == "supported":
        return "hidden_loop_registered_bridge_only"
    if registered_lead_status == "supported":
        return "hidden_loop_structural_lead_only"
    statuses = (economic_status, registered_lead_status, predictive_bridge_status)
    if all(value == "insufficient_support" for value in statuses):
        return "blocked_support_failure"
    if "descriptive_only" in statuses:
        return "descriptive_hidden_loop_effects_only"
    return "no_hidden_loop_economic_or_bridge_increment"


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def manual_probability(frame: pd.DataFrame, specification: dict[str, Any]) -> np.ndarray:
    features = [str(value) for value in specification["feature_names"]]
    matrix = frame.loc[:, features].to_numpy(dtype=float)
    mean = np.asarray(specification["scaler_mean"], dtype=float)
    scale = np.asarray(specification["scaler_scale"], dtype=float)
    coefficient = np.asarray(specification["coefficient"], dtype=float)
    logits = ((matrix - mean) / scale) @ coefficient + float(specification["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0)))


def predecessor_probability(frame: pd.DataFrame, specification: dict[str, Any]) -> np.ndarray:
    normalised = {
        "feature_names": specification["features"],
        "scaler_mean": specification["scaler_mean"],
        "scaler_scale": specification["scaler_scale"],
        "coefficient": specification["coefficient"][0],
        "intercept": specification["intercept"][0],
    }
    return manual_probability(frame, normalised)


def load_market_source(
    source_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    materialised_rows: list[pd.DataFrame] = []
    for source_path in source_manifest["raw_market_source"]["source_files_touched"]:
        frame = pd.read_parquet(
            Path(source_path),
            columns=["symbol", "timestamp", "open", "high", "low", "close"],
            filters=[
                ("timestamp", ">=", pd.Timestamp("2024-01-01T00:00:00Z").to_pydatetime()),
                ("timestamp", "<", pd.Timestamp("2025-08-23T00:00:00Z").to_pydatetime()),
            ],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        materialised = frame.loc[:, ["symbol", "timestamp"]].copy()
        materialised["year_month"] = local.dt.strftime("%Y-%m")
        materialised_rows.append(materialised)
        minute = local.dt.hour * 60 + local.dt.minute
        frame = frame.loc[minute.ge(570) & minute.lt(960)].copy()
        numeric = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
        frame["qa_valid"] = np.isfinite(numeric).all(axis=1) & (numeric > 0.0).all(axis=1)
        rows.append(frame)
    materialised = pd.concat(materialised_rows, ignore_index=True)
    row_counts = (
        materialised.groupby(["symbol", "year_month"], sort=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    audit = {
        "minimum_timestamp_read": str(materialised["timestamp"].min()),
        "maximum_timestamp_read": str(materialised["timestamp"].max()),
        "rows_by_symbol_and_month": row_counts.to_dict(orient="records"),
        "materialised_rows": int(len(materialised)),
        "protected_rows_materialised": int(
            materialised["timestamp"].ge(pd.Timestamp("2025-08-23T00:00:00Z")).sum()
        ),
    }
    market = pd.concat(rows, ignore_index=True).set_index(["symbol", "timestamp"])
    return market, audit


def _metric_increment(frame: pd.DataFrame) -> dict[str, float]:
    b0 = binary_model_metrics(
        frame["registered_completion_within_12_bars"],
        frame["B0_probability"],
        frame["row_weight"],
    )
    b1 = binary_model_metrics(
        frame["registered_completion_within_12_bars"],
        frame["B1_probability"],
        frame["row_weight"],
    )
    return {
        "log_loss_improvement": float(b0["log_loss"]) - float(b1["log_loss"]),
        "brier_improvement": float(b0["brier_score"]) - float(b1["brier_score"]),
        "auc_improvement": float(b1["auc"]) - float(b0["auc"]),
        "average_precision_improvement": float(b1["average_precision"])
        - float(b0["average_precision"]),
    }


def _maximum_positive_month_share(frame: pd.DataFrame) -> float:
    positive = frame.loc[
        frame["opening_pressure_net_return_20bps"].gt(0.0),
        ["event_month", "opening_pressure_net_return_20bps"],
    ]
    total = float(positive["opening_pressure_net_return_20bps"].sum())
    return float(
        positive.groupby("event_month")["opening_pressure_net_return_20bps"].sum().max() / total
    )


def independent_economic_status(
    economics: pd.DataFrame, bootstrap: pd.DataFrame
) -> tuple[str, dict[str, bool], dict[str, bool]]:
    primary = economics.loc[
        economics["horizon_bars"].eq(12) & economics["hidden_family_class"].isin(FROZEN_FAMILIES)
    ]
    development = primary.loc[primary["period"].eq("development")]
    assessment = primary.loc[primary["period"].eq("assessment")]

    def summary(metric: str) -> pd.Series:
        return bootstrap.loc[
            bootstrap["row_type"].eq("summary")
            & bootstrap["scope"].eq("FOUR_FROZEN_FAMILIES_POOLED")
            & bootstrap["metric"].eq(metric)
        ].iloc[0]

    support = {
        "assessment_events_at_least_1000": assessment["event_id"].nunique() >= 1000,
        "sessions_at_least_100": assessment["session"].nunique() >= 100,
        "stocks_at_least_15": assessment["symbol"].nunique() >= 15,
        "months_at_least_6": assessment["event_month"].nunique() >= 6,
        "matched_control_coverage_at_least_80pct": float(
            assessment["matched_control_available"].mean()
        )
        >= 0.80,
    }
    months = assessment.groupby("event_month")["opening_pressure_net_return_20bps"].mean()
    checkpoints = assessment.groupby("source_checkpoint")[
        "opening_pressure_net_return_20bps"
    ].mean()
    gate = {
        "assessment_primary_net_positive": float(
            assessment["opening_pressure_net_return_20bps"].mean()
        )
        > 0.0,
        "assessment_cohort_relative_net_positive": float(
            assessment["cohort_relative_net_return_20bps"].mean()
        )
        > 0.0,
        "primary_net_80pct_lower_nonnegative": float(
            summary("primary_net_return_20bps")["interval_80_lower"]
        )
        >= 0.0,
        "cohort_relative_80pct_lower_nonnegative": float(
            summary("cohort_relative_net_return_20bps")["interval_80_lower"]
        )
        >= 0.0,
        "matched_control_excess_positive": float(
            assessment["event_excess_vs_matched_control_bps"].mean()
        )
        > 0.0,
        "matched_control_80pct_lower_nonnegative": float(
            summary("matched_control_excess")["interval_80_lower"]
        )
        >= 0.0,
        "development_assessment_same_sign": bool(
            np.sign(development["opening_pressure_net_return_20bps"].mean())
            == np.sign(assessment["opening_pressure_net_return_20bps"].mean())
        ),
        "at_least_five_positive_assessment_months": int((months > 0.0).sum()) >= 5,
        "neither_checkpoint_materially_adverse": bool((checkpoints >= -5.0).all()),
        "maximum_stock_share_at_most_15pct": float(
            assessment.groupby("symbol").size().max() / len(assessment)
        )
        <= 0.15,
        "maximum_month_positive_return_share_at_most_30pct": (
            _maximum_positive_month_share(assessment) <= 0.30
        ),
    }
    support_passed = all(support.values())
    status = (
        "supported"
        if support_passed and all(gate.values())
        else "not_supported"
        if support_passed
        else "insufficient_support"
    )
    return status, support, gate


def independent_registered_lead_status(
    lead: pd.DataFrame, null: pd.DataFrame
) -> tuple[str, dict[str, bool], bool]:
    assessment = lead.loc[
        lead["period"].eq("assessment") & lead["hidden_family_class"].isin(FROZEN_FAMILIES)
    ]
    support = {
        "assessment_events_at_least_1000": len(assessment) >= 1000,
        "sessions_at_least_100": assessment["session"].nunique() >= 100,
        "stocks_at_least_15": assessment["symbol"].nunique() >= 15,
        "registered_completions_within_12_at_least_100": int(
            assessment["registered_within_12_bars"].sum()
        )
        >= 100,
    }
    null_six = null.loc[null["metric"].eq("registered_completion_rate_6"), "value"].to_numpy(
        dtype=float
    )
    lead_passed = float(assessment["registered_within_6_bars"].mean()) > float(
        np.quantile(null_six, 0.90)
    )
    support_passed = all(support.values())
    status = (
        "supported"
        if support_passed and lead_passed
        else "not_supported"
        if support_passed
        else "insufficient_support"
    )
    return status, support, lead_passed


def independent_predictive_bridge_status(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    crossfit: dict[str, Any],
    *,
    registered_lead_status: str,
) -> tuple[str, dict[str, bool], dict[str, bool], dict[str, float], dict[str, int]]:
    increments = _metric_increment(assessment)
    bootstrap_summary = bootstrap.loc[bootstrap["row_type"].eq("summary")].set_index("metric")
    month_pivot = monthly.pivot(index="year_month", columns="model", values="log_loss")
    checkpoint_pivot = checkpoint.pivot(
        index="decision_ordinal", columns="model", values="log_loss"
    )
    month_increment = month_pivot["B0"] - month_pivot["B1"]
    checkpoint_increment = checkpoint_pivot["B0"] - checkpoint_pivot["B1"]
    class_share = float(
        assessment["registered_completion_within_12_bars"].value_counts(normalize=True).max()
    )
    support = {
        "assessment_rows_at_least_5500": len(assessment) >= 5500,
        "sessions_at_least_140": assessment["session"].nunique() >= 140,
        "stocks_at_least_15": assessment["symbol"].nunique() >= 15,
        "eight_assessment_months": assessment["year_month"].nunique() == 8,
        "positive_targets_at_least_500": int(
            assessment["registered_completion_within_12_bars"].sum()
        )
        >= 500,
        "crossfit_coverage_at_least_90pct": float(crossfit["coverage_after_warmup"]) >= 0.90,
        "no_target_class_above_80pct": class_share <= 0.80,
    }
    null_exceeded = {
        metric: int((float(value) > null[metric]).sum()) for metric, value in increments.items()
    }
    gate = {
        "B1_improves_log_loss": increments["log_loss_improvement"] > 0.0,
        "B1_improves_brier": increments["brier_improvement"] > 0.0,
        "B1_does_not_reduce_auc": increments["auc_improvement"] >= 0.0,
        "log_loss_90pct_lower_nonnegative": float(
            bootstrap_summary.loc["log_loss_improvement", "interval_90_lower"]
        )
        >= 0.0,
        "brier_90pct_lower_nonnegative": float(
            bootstrap_summary.loc["brier_improvement", "interval_90_lower"]
        )
        >= 0.0,
        "five_positive_assessment_months": int((month_increment > 0.0).sum()) >= 5,
        "neither_checkpoint_materially_adverse": bool((checkpoint_increment >= -0.001).all()),
        "real_increment_exceeds_nine_nulls": max(
            null_exceeded["log_loss_improvement"], null_exceeded["brier_improvement"]
        )
        >= 9,
        "maximum_stock_share_at_most_10pct": float(
            assessment.groupby("symbol").size().max() / len(assessment)
        )
        <= 0.10,
    }
    support_passed = all(support.values())
    conditions_passed = support_passed and all(gate.values())
    status = (
        "insufficient_support"
        if not support_passed
        else "supported"
        if conditions_passed and registered_lead_status == "supported"
        else "descriptive_only"
        if conditions_passed
        else "not_supported"
    )
    return status, support, gate, increments, null_exceeded


def run_audit(output: Path, *, write: bool = True) -> dict[str, Any]:
    output = output.resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    missing = [name for name in REQUIRED_ARTIFACTS if not (output / name).is_file()]
    check("required_artifacts", not missing, {"missing": missing})
    contract = read_json(output / "contract.json")
    decision = read_json(output / "decision.json")
    safety_passed = all(
        contract.get(key) == expected and decision.get(key) == expected
        for key, expected in SAFETY_FLAGS.items()
    )
    check("safety_flags", safety_passed, SAFETY_FLAGS)
    source = read_json(output / "source_manifest.json")
    protected = read_json(output / "protected_boundary_audit.json")
    market, independently_materialised = load_market_source(source)
    maximum_timestamp = pd.Timestamp(source["maximum_timestamp_read"])
    boundary_passed = (
        maximum_timestamp < pd.Timestamp("2025-08-23T00:00:00Z")
        and source["protected_rows_materialised"] == 0
        and protected["protected_rows_materialised"] == 0
        and independently_materialised["protected_rows_materialised"] == 0
        and source["minimum_timestamp_read"] == independently_materialised["minimum_timestamp_read"]
        and source["maximum_timestamp_read"] == independently_materialised["maximum_timestamp_read"]
        and source["rows_by_symbol_and_month"]
        == independently_materialised["rows_by_symbol_and_month"]
    )
    check(
        "dates_and_protected_boundary",
        boundary_passed,
        {
            "minimum": source["minimum_timestamp_read"],
            "maximum": source["maximum_timestamp_read"],
            "protected_rows": source["protected_rows_materialised"],
            "independently_materialised_rows": independently_materialised["materialised_rows"],
        },
    )
    family_mapping = read_json(FAMILY_MAPPING)
    check(
        "frozen_hidden_family_identities",
        tuple(family_mapping["selected_families"]) == FROZEN_FAMILIES
        and tuple(contract["frozen_hidden_family_identities"]) == FROZEN_FAMILIES,
        list(FROZEN_FAMILIES),
    )

    predecessor_events = pd.read_parquet(PATH_LEDGER)
    predecessor_events["event_timestamp_utc"] = pd.to_datetime(
        predecessor_events["event_timestamp_utc"], utc=True
    )
    predecessor_events["decision_timestamp_utc"] = pd.to_datetime(
        predecessor_events["decision_timestamp_utc"], utc=True
    )
    predecessor_events["event_available_timestamp_utc"] = pd.to_datetime(
        predecessor_events["event_available_timestamp_utc"], utc=True
    )
    identity = ["symbol", "session", "event_timestamp_utc", "family_id"]
    eligible = predecessor_events.loc[
        predecessor_events["decision_timestamp_utc"].lt(
            predecessor_events["event_available_timestamp_utc"]
        )
    ].copy()
    latest = eligible.sort_values("decision_timestamp_utc").drop_duplicates(identity, keep="last")
    latest["event_id"] = (
        latest["symbol"].astype(str)
        + "|"
        + latest["session"].astype(str)
        + "|"
        + latest["event_timestamp_utc"].astype(str)
        + "|"
        + latest["family_id"].astype(str)
    )
    economics = pd.read_parquet(output / "hidden_event_economic_ledger.parquet")
    economic_events = economics.drop_duplicates("event_id")
    checkpoint_comparison = economic_events[["event_id", "source_checkpoint"]].merge(
        latest[["event_id", "decision_ordinal"]],
        on="event_id",
        how="outer",
        validate="one_to_one",
    )
    check(
        "event_deduplication_and_latest_source_checkpoint",
        not latest.duplicated(identity).any()
        and len(latest) == economic_events["event_id"].nunique()
        and checkpoint_comparison["source_checkpoint"]
        .eq(checkpoint_comparison["decision_ordinal"])
        .all()
        and len(predecessor_events) - len(latest)
        == read_json(output / "hidden_event_population_reconstruction.json")[
            "deduplicated_rows_removed"
        ],
        {"predecessor": len(predecessor_events), "unique": len(latest)},
    )
    timing_passed = bool(
        economics["entry_timestamp_utc"]
        .eq(economics["event_completion_timestamp_utc"] + pd.Timedelta(minutes=5))
        .all()
        and economics["exit_bar_start_timestamp_utc"]
        .eq(
            economics["entry_timestamp_utc"]
            + pd.to_timedelta((economics["horizon_bars"] - 1) * 5, unit="minutes")
        )
        .all()
        and economics["exit_timestamp_utc"]
        .eq(economics["exit_bar_start_timestamp_utc"] + pd.Timedelta(minutes=5))
        .all()
    )
    check("next_bar_entry_and_fixed_exits", timing_passed, {"rows": len(economics)})

    opening = pd.read_parquet(OPENING_PANEL)
    opening = opening.loc[opening["decision_ordinal"].isin([6, 12])].copy()
    direction_source = opening.loc[
        :, ["symbol", "session", "decision_ordinal", "signed_pressure"]
    ].drop_duplicates(["symbol", "session", "decision_ordinal"])
    direction_check = economic_events.merge(
        direction_source,
        left_on=["symbol", "session", "source_checkpoint"],
        right_on=["symbol", "session", "decision_ordinal"],
        validate="one_to_one",
    )
    expected_direction = np.sign(direction_check["signed_pressure"].to_numpy(dtype=float)).astype(
        int
    )
    direction_passed = np.array_equal(
        expected_direction,
        direction_check["opening_pressure_direction"].to_numpy(dtype=int),
    ) and np.array_equal(
        -expected_direction,
        direction_check["opposite_opening_pressure_direction"].to_numpy(dtype=int),
    )
    check("opening_and_opposite_direction", direction_passed, {"rows": len(direction_check)})
    friction_difference = np.max(
        np.abs(
            economics["opening_pressure_signed_return_bps"].to_numpy(dtype=float)
            - 20.0
            - economics["opening_pressure_net_return_20bps"].to_numpy(dtype=float)
        )
    )
    check("twenty_basis_point_friction", friction_difference <= 1e-12, friction_difference)

    sample = economics.loc[economics["horizon_bars"].eq(12)].head(100)
    raw_differences: list[float] = []
    relative_differences: list[float] = []
    momentum_mismatches = 0
    for event in sample.itertuples(index=False):
        entry = market.loc[(str(event.symbol), pd.Timestamp(event.entry_timestamp_utc))]
        exit_row = market.loc[(str(event.symbol), pd.Timestamp(event.exit_bar_start_timestamp_utc))]
        raw = 10_000.0 * (float(exit_row["close"]) / float(entry["open"]) - 1.0)
        raw_differences.append(abs(raw - float(event.raw_return_bps)))
        other_returns = []
        momentum_others = []
        decision_bar = pd.Timestamp(event.decision_timestamp_utc) - pd.Timedelta(minutes=5)
        completion_bar = pd.Timestamp(event.event_completion_timestamp_utc)
        own_momentum = 10_000.0 * (
            float(market.loc[(str(event.symbol), completion_bar)]["close"])
            / float(market.loc[(str(event.symbol), decision_bar)]["close"])
            - 1.0
        )
        for symbol in FROZEN_SOURCE_SYMBOLS:
            if symbol == str(event.symbol):
                continue
            try:
                other_returns.append(
                    10_000.0
                    * (
                        float(
                            market.loc[(symbol, pd.Timestamp(event.exit_bar_start_timestamp_utc))][
                                "close"
                            ]
                        )
                        / float(
                            market.loc[(symbol, pd.Timestamp(event.entry_timestamp_utc))]["open"]
                        )
                        - 1.0
                    )
                )
                momentum_others.append(
                    10_000.0
                    * (
                        float(market.loc[(symbol, completion_bar)]["close"])
                        / float(market.loc[(symbol, decision_bar)]["close"])
                        - 1.0
                    )
                )
            except KeyError:
                continue
        relative = int(event.opening_pressure_direction) * (raw - float(np.mean(other_returns)))
        relative_differences.append(abs(relative - float(event.cohort_relative_signed_return_bps)))
        momentum_sign = int(np.sign(own_momentum - float(np.mean(momentum_others))))
        if momentum_sign != int(event.completion_momentum_direction):
            momentum_mismatches += 1
    check(
        "entry_exit_prices_and_cohort_relative_returns",
        max(raw_differences, default=0.0) <= 1e-10
        and max(relative_differences, default=0.0) <= 1e-10,
        {
            "rows": len(sample),
            "maximum_raw_difference": max(raw_differences, default=0.0),
            "maximum_relative_difference": max(relative_differences, default=0.0),
        },
    )
    check(
        "completion_momentum_direction",
        momentum_mismatches == 0,
        {"rows": len(sample), "mismatches": momentum_mismatches},
    )

    controls = pd.read_parquet(output / "matched_control_ledger.parquet")
    control_counts = controls.groupby(["event_id", "horizon_bars"]).size()
    matched_rows = economics.loc[economics["matched_control_available"]]
    control_count_passed = all(
        int(control_counts.loc[(row.event_id, row.horizon_bars)])
        == int(row.matched_control_count)
        >= 5
        for row in matched_rows.itertuples(index=False)
    )
    control_timing = controls.merge(
        economics[["event_id", "horizon_bars", "entry_timestamp_utc", "exit_timestamp_utc"]],
        on=["event_id", "horizon_bars"],
        suffixes=("_control", "_event"),
        validate="many_to_one",
    )
    control_timing_passed = bool(
        control_timing["entry_timestamp_utc_control"]
        .eq(control_timing["entry_timestamp_utc_event"])
        .all()
        and control_timing["exit_timestamp_utc_control"]
        .eq(control_timing["exit_timestamp_utc_event"])
        .all()
    )
    control_context = economic_events[
        [
            "event_id",
            "session",
            "source_checkpoint",
            "decision_timestamp_utc",
            "event_completion_available_timestamp_utc",
        ]
    ]
    control_pairs = (
        controls[["event_id", "control_symbol", "control_direction"]]
        .drop_duplicates()
        .merge(control_context, on="event_id", validate="many_to_one")
    )
    control_pressure = direction_source.rename(
        columns={
            "symbol": "control_symbol",
            "decision_ordinal": "source_checkpoint",
            "signed_pressure": "control_signed_pressure",
        }
    )
    control_pairs = control_pairs.merge(
        control_pressure,
        on=["control_symbol", "session", "source_checkpoint"],
        how="left",
        validate="many_to_one",
    )
    control_direction_passed = np.array_equal(
        control_pairs["control_direction"].to_numpy(dtype=int),
        np.sign(control_pairs["control_signed_pressure"].to_numpy(dtype=float)).astype(int),
    )
    prior_candidates = control_pairs.merge(
        predecessor_events[["symbol", "session", "event_available_timestamp_utc"]].rename(
            columns={"symbol": "control_symbol"}
        ),
        on=["control_symbol", "session"],
        how="left",
    )
    invalid_prior = prior_candidates.loc[
        prior_candidates["event_available_timestamp_utc"].le(
            prior_candidates["event_completion_available_timestamp_utc"]
        )
    ]
    eligibility_rule_passed = bool(
        economics["matched_control_eligibility_rule"]
        .eq("no_unregistered_completion_by_focal_completion")
        .all()
    )
    check(
        "matched_control_eligibility_and_timing",
        control_count_passed
        and control_timing_passed
        and control_direction_passed
        and invalid_prior.empty
        and eligibility_rule_passed,
        {
            "matched_events": len(matched_rows),
            "control_rows": len(controls),
            "invalid_prior_event_controls": len(invalid_prior),
        },
    )

    assessment_economics = economics.loc[
        economics["period"].eq("assessment")
        & economics["hidden_family_class"].isin(FROZEN_FAMILIES)
        & economics["horizon_bars"].eq(12)
    ].reset_index(drop=True)
    bootstrap_indices = session_block_bootstrap_indices(
        assessment_economics, draws=50, seed=20260721
    )
    expected_draws = [
        float(assessment_economics.iloc[index]["opening_pressure_net_return_20bps"].mean())
        for index in bootstrap_indices
    ]
    economic_bootstrap = pd.read_csv(output / "economic_bootstrap_metrics.csv")
    archived_draws = economic_bootstrap.loc[
        economic_bootstrap["row_type"].eq("draw")
        & economic_bootstrap["scope"].eq("FOUR_FROZEN_FAMILIES_POOLED")
        & economic_bootstrap["metric"].eq("primary_net_return_20bps"),
        "value",
    ].to_numpy(dtype=float)
    bootstrap_difference = float(np.max(np.abs(np.asarray(expected_draws) - archived_draws)))
    check(
        "whole_session_economic_bootstrap",
        len(archived_draws) == 50 and bootstrap_difference <= 1e-12,
        bootstrap_difference,
    )
    multiplicity = pd.read_csv(output / "economic_multiplicity_results.csv")
    q_values = benjamini_hochberg(multiplicity["p_value"].astype(float).tolist())
    q_difference = float(
        np.max(np.abs(np.asarray(q_values) - multiplicity["q_value"].to_numpy(dtype=float)))
    )
    check("family_BH_correction", q_difference <= 1e-12, q_difference)

    lead = pd.read_parquet(output / "hidden_to_registered_lead_ledger.parquet")
    lead_horizon_passed = bool(
        lead.loc[lead["registered_within_6_bars"], "bars_to_first_registered_completion"]
        .between(1, 6)
        .all()
        and lead.loc[lead["registered_within_12_bars"], "bars_to_first_registered_completion"]
        .between(1, 12)
        .all()
    )
    check("strict_hidden_to_registered_horizons", lead_horizon_passed, {"rows": len(lead)})
    completions = pd.read_parquet(output / "registered_completion_ledger.parquet")
    completion_groups = {
        (str(symbol), str(session)): group
        for (symbol, session), group in completions.groupby(["symbol", "session"], sort=False)
    }
    lead_sample = lead.head(100)
    lead_mismatches = 0
    for event in lead_sample.itertuples(index=False):
        candidates = completion_groups.get(
            (str(event.symbol), str(event.session)),
            pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id", "motif_type"]),
        )
        targets = registered_completion_targets(int(event.completion_bar_ordinal), candidates)
        if bool(targets["registered_within_6_bars"]) != bool(event.registered_within_6_bars):
            lead_mismatches += 1
        if bool(targets["registered_within_12_bars"]) != bool(event.registered_within_12_bars):
            lead_mismatches += 1
        if targets["registered_within_12_bars"]:
            first_bar = int(event.completion_bar_ordinal) + int(
                targets["bars_to_first_registered_completion"]
            )
            expected_ids = sorted(
                candidates.loc[
                    candidates["completion_bar_ordinal"].eq(first_bar), "semantic_loop_id"
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            if expected_ids != json.loads(str(event.first_registered_semantic_loop_ids_json)):
                lead_mismatches += 1
    check("registered_timeline_reconstruction", lead_mismatches == 0, lead_mismatches)
    exact_pairs = pd.read_csv(output / "hidden_to_registered_exact_pairs.csv")
    supported_pairs = exact_pairs.loc[exact_pairs["support_eligible"]]
    pair_support_passed = bool(
        supported_pairs["development_transitions"].ge(30).all()
        and supported_pairs["assessment_transitions"].ge(20).all()
        and exact_pairs.loc[~exact_pairs["support_eligible"], "assessment_q_value"].isna().all()
    )
    pair_q_difference = 0.0
    if not supported_pairs.empty:
        reproduced_q = benjamini_hochberg(
            supported_pairs["assessment_p_value"].astype(float).tolist()
        )
        pair_q_difference = float(
            np.max(
                np.abs(
                    np.asarray(reproduced_q)
                    - supported_pairs["assessment_q_value"].to_numpy(dtype=float)
                )
            )
        )
    check(
        "exact_transition_support_and_BH",
        pair_support_passed and pair_q_difference <= 1e-12,
        {"supported_pairs": len(supported_pairs), "maximum_q_difference": pair_q_difference},
    )

    observed = lead.loc[
        lead["period"].eq("assessment") & lead["hidden_family_class"].isin(FROZEN_FAMILIES)
    ].reset_index(drop=True)
    eligible_sessions = (
        pd.read_parquet(output / "bridge_assessment_predictions.parquet")[["symbol", "session"]]
        .drop_duplicates()
        .assign(period="assessment")
    )
    null_rows = pd.read_csv(output / "structural_lead_null_metrics.csv")
    first_permutation = stock_clock_session_permutation(observed, eligible_sessions, seed=20260722)
    first_targets = []
    for event in first_permutation.itertuples(index=False):
        candidates = completion_groups.get(
            (str(event.symbol), str(event.session)),
            pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id", "motif_type"]),
        )
        first_targets.append(
            registered_completion_targets(int(event.completion_bar_ordinal), candidates)
        )
    first_rate = float(
        np.mean([bool(value["registered_within_6_bars"]) for value in first_targets])
    )
    archived_first_rate = float(
        null_rows.loc[
            null_rows["draw"].eq(1) & null_rows["metric"].eq("registered_completion_rate_6"),
            "value",
        ].iloc[0]
    )
    check(
        "stock_clock_structural_lead_null",
        null_rows["draw"].nunique() == 50 and abs(first_rate - archived_first_rate) <= 1e-12,
        {"first_draw_difference": abs(first_rate - archived_first_rate)},
    )

    development = pd.read_parquet(output / "bridge_development_panel.parquet")
    assessment = pd.read_parquet(output / "bridge_assessment_predictions.parquet")
    target_mismatches = 0
    for row in assessment.head(100).itertuples(index=False):
        candidates = completion_groups.get(
            (str(row.symbol), str(row.session)),
            pd.DataFrame(columns=["completion_bar_ordinal", "semantic_loop_id", "motif_type"]),
        )
        expected = registered_completion_targets(int(row.repo_bar_start_ordinal), candidates)
        target_mismatches += int(
            bool(expected["registered_within_12_bars"])
            != bool(row.registered_completion_within_12_bars)
        )
    check("twelve_bar_registered_bridge_target", target_mismatches == 0, target_mismatches)
    crossfit = read_json(output / "bridge_crossfit_manifest.json")
    chronology_passed = (
        all(
            str(row["train_session_end"]) < str(row["prediction_session_start"])
            for row in crossfit["fold_manifest"]
        )
        and float(crossfit["coverage_after_warmup"]) >= 0.90
    )
    t1_features = tuple(str(value) for value in crossfit["feature_specification"])
    raw_outcome = (
        opening["raw_outcome"]
        .astype(str)
        .replace(
            {
                "REGISTERED_PRIMITIVE": "REGISTERED_COMPLETION",
                "REGISTERED_REPEAT": "REGISTERED_COMPLETION",
                "REGISTERED_COMPOSITE": "REGISTERED_COMPLETION",
            }
        )
    )
    opening["audit_unregistered_event"] = math.nan
    scoring = raw_outcome.isin(
        ["UNREGISTERED_LOOP", "REGISTERED_COMPLETION", "NO_REGISTERED_COMPLETION"]
    )
    opening.loc[scoring, "audit_unregistered_event"] = (
        raw_outcome.loc[scoring].eq("UNREGISTERED_LOOP").astype(float)
    )
    crossfit_source = opening.loc[
        opening["year"].eq(2024)
        & opening["audit_unregistered_event"].notna()
        & opening.loc[:, list(t1_features)].notna().all(axis=1)
    ].copy()
    reproduced_crossfit, reproduced_manifest = expanding_logistic_crossfit(
        crossfit_source,
        features=t1_features,
        target="audit_unregistered_event",
        folds=4,
        warmup_fraction=0.2,
    )
    reproduced = crossfit_source.loc[
        reproduced_crossfit.notna(),
        [
            "symbol",
            "session",
            "decision_ordinal",
        ],
    ].copy()
    reproduced["reproduced_probability"] = reproduced_crossfit.loc[
        reproduced_crossfit.notna()
    ].to_numpy(dtype=float)
    crossfit_comparison = reproduced.merge(
        development[
            [
                "symbol",
                "session",
                "decision_ordinal",
                "oof_p_unregistered_within_6_bars",
            ]
        ],
        on=["symbol", "session", "decision_ordinal"],
        how="outer",
        validate="one_to_one",
    )
    crossfit_difference = float(
        np.max(
            np.abs(
                crossfit_comparison["reproduced_probability"].to_numpy(dtype=float)
                - crossfit_comparison["oof_p_unregistered_within_6_bars"].to_numpy(dtype=float)
            )
        )
    )
    chronology_passed = (
        chronology_passed
        and len(reproduced_manifest) == 4
        and len(crossfit_comparison) == len(development)
        and crossfit_difference <= 1e-12
    )
    check(
        "expanding_chronological_U1_crossfit",
        chronology_passed,
        {
            "fold_manifest": crossfit["fold_manifest"],
            "maximum_probability_difference": crossfit_difference,
        },
    )

    predecessor_coefficients = read_json(PREDECESSOR_COEFFICIENTS)["primary_models"]["U1"]
    frozen_u1 = predecessor_probability(assessment, predecessor_coefficients)
    frozen_u1_difference = float(
        np.max(np.abs(frozen_u1 - assessment["p_unregistered_within_6_bars"].to_numpy(dtype=float)))
    )
    check("frozen_U1_assessment_probability", frozen_u1_difference <= 1e-12, frozen_u1_difference)
    model_configurations = read_json(output / "bridge_model_configurations.json")
    coefficients = read_json(output / "bridge_model_coefficients.json")
    b0_features = tuple(model_configurations["B0"]["features"])
    b1_features = tuple(model_configurations["B1"]["features"])
    frozen_model = model_configurations["model"]
    feature_passed = (
        b1_features == (*b0_features, "p_unregistered_within_6_bars")
        and frozen_model
        == {
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "n_jobs": 1,
            "random_state": 20260721,
        }
        and int(model_configurations["primary_model_fits"])
        + int(model_configurations["determinism_refits"])
        == 4
        and int(model_configurations["bridge_null_refits"]) == 10
    )
    coefficient_shapes_passed = True
    maximum_scaler_difference = 0.0
    maximum_kkt_residual = 0.0
    for model, features in (("B0", b0_features), ("B1", b1_features)):
        specification = coefficients[model]
        arrays = [
            np.asarray(specification[key], dtype=float)
            for key in ("scaler_mean", "scaler_scale", "coefficient")
        ]
        coefficient_shapes_passed = coefficient_shapes_passed and bool(
            tuple(specification["feature_names"]) == features
            and all(len(values) == len(features) for values in arrays)
            and all(np.isfinite(values).all() for values in arrays)
            and (arrays[1] > 0.0).all()
            and np.isfinite(float(specification["intercept"]))
        )
        raw_matrix = development.loc[:, list(features)].to_numpy(dtype=float)
        expected_mean = raw_matrix.mean(axis=0)
        expected_scale = raw_matrix.std(axis=0, ddof=0)
        expected_scale[expected_scale == 0.0] = 1.0
        maximum_scaler_difference = max(
            maximum_scaler_difference,
            float(np.max(np.abs(arrays[0] - expected_mean))),
            float(np.max(np.abs(arrays[1] - expected_scale))),
        )
        transformed = (raw_matrix - arrays[0]) / arrays[1]
        labels = development["registered_completion_within_12_bars"].to_numpy(dtype=float)
        weights = development["row_weight"].to_numpy(dtype=float)
        intercept = float(specification["intercept"])
        probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(transformed @ arrays[2] + intercept, -709.0, 709.0))
        )
        residual = weights * (probabilities - labels)
        coefficient_gradient = arrays[2] + 0.25 * (transformed.T @ residual)
        intercept_gradient = intercept + 0.25 * float(residual.sum())
        maximum_kkt_residual = max(
            maximum_kkt_residual,
            float(np.max(np.abs(coefficient_gradient))),
            abs(intercept_gradient),
        )
    check(
        "B0_B1_features_coefficients_and_fit_budget",
        feature_passed
        and coefficient_shapes_passed
        and maximum_scaler_difference <= 1e-12
        and maximum_kkt_residual <= 1e-4,
        {
            "B0": len(b0_features),
            "B1": len(b1_features),
            "primary_plus_determinism_fits": int(model_configurations["primary_model_fits"])
            + int(model_configurations["determinism_refits"]),
            "null_refits": int(model_configurations["bridge_null_refits"]),
            "maximum_scaler_difference": maximum_scaler_difference,
            "maximum_weighted_L2_KKT_residual": maximum_kkt_residual,
        },
    )
    sample_100 = assessment.head(100)
    probability_differences = []
    for model in ("B0", "B1"):
        manual = manual_probability(sample_100, coefficients[model])
        probability_differences.append(
            float(np.max(np.abs(manual - sample_100[f"{model}_probability"].to_numpy(dtype=float))))
        )
    check(
        "manual_probability_reconstruction_100_rows",
        max(probability_differences) <= 1e-12,
        max(probability_differences),
    )
    archived_bridge_metrics = pd.read_csv(output / "bridge_metrics.csv")
    metric_differences = []
    for model in ("B0", "B1"):
        calculated = binary_model_metrics(
            assessment["registered_completion_within_12_bars"],
            assessment[f"{model}_probability"],
            assessment["row_weight"],
        )
        archived = archived_bridge_metrics.loc[
            archived_bridge_metrics["model"].eq(model)
            & archived_bridge_metrics["subgroup"].eq("pooled_assessment")
        ].iloc[0]
        for metric in ("log_loss", "brier_score", "auc", "average_precision"):
            metric_differences.append(abs(float(calculated[metric]) - float(archived[metric])))
    check("bridge_core_metrics", max(metric_differences) <= 1e-12, max(metric_differences))

    bridge_indices = session_block_bootstrap_indices(assessment, draws=50, seed=20260723)
    expected_bridge_draws = [
        _metric_increment(assessment.iloc[index])["log_loss_improvement"]
        for index in bridge_indices
    ]
    archived_bridge_bootstrap = pd.read_csv(output / "bridge_bootstrap_metrics.csv")
    archived_bridge_draws = archived_bridge_bootstrap.loc[
        archived_bridge_bootstrap["row_type"].eq("draw")
        & archived_bridge_bootstrap["metric"].eq("log_loss_improvement"),
        "value",
    ].to_numpy(dtype=float)
    bridge_bootstrap_difference = float(
        np.max(np.abs(np.asarray(expected_bridge_draws) - archived_bridge_draws))
    )
    check(
        "fixed_prediction_session_bridge_bootstrap",
        len(archived_bridge_draws) == 50 and bridge_bootstrap_difference <= 1e-12,
        bridge_bootstrap_difference,
    )
    archived_null = pd.read_csv(output / "bridge_null_metrics.csv")
    permutation_integrity = True
    permutation_changed = False
    expected_development_hashes: list[str] = []
    expected_assessment_hashes: list[str] = []
    feature = "p_unregistered_within_6_bars"
    for draw in range(1, 11):
        seed = 20260724 + draw - 1
        for panel_name, panel in (("development", development), ("assessment", assessment)):
            permuted = permute_feature_within_slates(panel, feature=feature, seed=seed)
            if panel_name == "development":
                expected_development_hashes.append(bridge_permutation_sha256(permuted))
            else:
                expected_assessment_hashes.append(bridge_permutation_sha256(permuted))
            permutation_integrity = permutation_integrity and panel.drop(columns=[feature]).equals(
                permuted.drop(columns=[feature])
            )
            for positions in panel.groupby("slate_id", sort=True).indices.values():
                index = np.asarray(positions, dtype=int)
                permutation_integrity = permutation_integrity and bool(
                    np.array_equal(
                        np.sort(panel.iloc[index][feature].to_numpy(dtype=float)),
                        np.sort(permuted.iloc[index][feature].to_numpy(dtype=float)),
                    )
                )
            permutation_changed = permutation_changed or not np.array_equal(
                panel[feature].to_numpy(dtype=float),
                permuted[feature].to_numpy(dtype=float),
            )
    real_increment = _metric_increment(assessment)
    reproduced_exceeded = {
        metric: int((float(value) > archived_null[metric]).sum())
        for metric, value in real_increment.items()
    }
    archived_exceeded = {
        str(metric): int(value)
        for metric, value in decision["predictive_bridge_gate"]["null_draws_exceeded"].items()
    }
    trace_passed = bool(
        archived_null["seed"].astype(int).tolist() == list(range(20260724, 20260734))
        and archived_null["development_permutation_sha256"].astype(str).tolist()
        == expected_development_hashes
        and archived_null["assessment_permutation_sha256"].astype(str).tolist()
        == expected_assessment_hashes
    )
    check(
        "within_slate_bridge_null",
        archived_null["draw"].tolist() == list(range(1, 11))
        and archived_null.drop(columns=["draw"]).notna().all().all()
        and permutation_integrity
        and permutation_changed
        and trace_passed
        and reproduced_exceeded == archived_exceeded,
        {
            "draws": len(archived_null),
            "reproduced_nulls_exceeded": reproduced_exceeded,
            "permutation_fingerprints_match": trace_passed,
            "additional_null_refits": 0,
        },
    )
    reproduced_economic_status, economic_support, economic_gate = independent_economic_status(
        economics, economic_bootstrap
    )
    reproduced_registered_status, lead_support, lead_passed = independent_registered_lead_status(
        lead, null_rows
    )
    archived_bridge_monthly = pd.read_csv(output / "bridge_monthly_metrics.csv")
    archived_bridge_checkpoint = pd.read_csv(output / "bridge_checkpoint_metrics.csv")
    (
        reproduced_bridge_status,
        bridge_support,
        bridge_gate,
        reproduced_increment,
        reproduced_null_exceeded,
    ) = independent_predictive_bridge_status(
        development,
        assessment,
        archived_bridge_monthly,
        archived_bridge_checkpoint,
        archived_bridge_bootstrap,
        archived_null,
        crossfit,
        registered_lead_status=reproduced_registered_status,
    )
    reproduced_statuses = {
        "economic_status": reproduced_economic_status,
        "registered_lead_status": reproduced_registered_status,
        "predictive_bridge_status": reproduced_bridge_status,
    }
    expected_decision = choose_primary_decision(**reproduced_statuses)
    archived_statuses = {key: str(decision[key]) for key in reproduced_statuses}
    increment_difference = max(
        abs(
            float(reproduced_increment[key])
            - float(decision["predictive_bridge_gate"]["increments"][key])
        )
        for key in reproduced_increment
    )
    gate_inputs_passed = bool(
        economic_support == decision["economic_gate"]["support_checks"]
        and economic_gate == decision["economic_gate"]["gate_checks"]
        and lead_support == decision["registered_lead_gate"]["support_checks"]
        and lead_passed
        == bool(decision["registered_lead_gate"]["observed_exceeds_null_90th_percentile"])
        and bridge_support == decision["predictive_bridge_gate"]["support_checks"]
        and bridge_gate == decision["predictive_bridge_gate"]["gate_checks"]
        and reproduced_null_exceeded
        == {
            str(key): int(value)
            for key, value in decision["predictive_bridge_gate"]["null_draws_exceeded"].items()
        }
        and increment_difference <= 1e-12
    )
    check(
        "decision_logic",
        reproduced_statuses == archived_statuses
        and gate_inputs_passed
        and expected_decision == decision["primary_decision"],
        {
            "reproduced_statuses": reproduced_statuses,
            "archived_statuses": archived_statuses,
            "gate_inputs_match": gate_inputs_passed,
            "expected": expected_decision,
            "actual": decision["primary_decision"],
        },
    )
    determinism = read_json(output / "determinism_check.json")
    refit_coefficient_difference = 0.0
    for model in ("B0", "B1"):
        archived_specification = coefficients[model]
        refit_specification = determinism["refit_model_configurations"][model]
        for key in ("scaler_mean", "scaler_scale", "coefficient"):
            refit_coefficient_difference = max(
                refit_coefficient_difference,
                float(
                    np.max(
                        np.abs(
                            np.asarray(archived_specification[key], dtype=float)
                            - np.asarray(refit_specification[key], dtype=float)
                        )
                    )
                ),
            )
        refit_coefficient_difference = max(
            refit_coefficient_difference,
            abs(
                float(archived_specification["intercept"]) - float(refit_specification["intercept"])
            ),
        )
    check(
        "fast_determinism_check",
        bool(determinism["passed"])
        and bool(determinism["final_decision_match"])
        and bool(determinism["status_reconstruction_match"])
        and determinism["reproduced_statuses"] == reproduced_statuses
        and refit_coefficient_difference <= 1e-12,
        {
            "maximum_refit_coefficient_difference": refit_coefficient_difference,
            "maximum_probability_difference": determinism["maximum_probability_difference"],
            "maximum_return_difference_bps": determinism["maximum_return_difference_bps"],
            "final_decision_match": determinism["final_decision_match"],
            "status_reconstruction_match": determinism["status_reconstruction_match"],
        },
    )
    passed = all(bool(item["passed"]) for item in checks)
    result = {
        **SAFETY_FLAGS,
        "audit": "lightweight_independent_audit",
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
        "passed": passed,
    }
    if write:
        write_json(output / "lightweight_audit.json", result)
    return result


FROZEN_SOURCE_SYMBOLS = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=EXPERIMENT_DIR / "artifacts" / "primary")
    return parser.parse_args()


def main() -> int:
    result = run_audit(parse_args().artifacts, write=True)
    print(json.dumps(result, sort_keys=True, indent=2, default=str))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
