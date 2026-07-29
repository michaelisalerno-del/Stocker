#!/usr/bin/env python3
"""Independent lightweight audit for Movement-Regime-Path Probability Chain V0.

This file intentionally does not import the runner or its reusable V0 module.
It does not reconstruct or refit the historical regime model.
"""

# ruff: noqa: E402 -- thread limits must be set before numerical-library imports.

from __future__ import annotations

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import math
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
START = pd.Timestamp("2024-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
BOOTSTRAP_SEED = 20260720
NULL_SEED = 20260721
BOOTSTRAP_DRAWS = 500
NULL_DRAWS = 100
SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "feasibility_screen": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
B1_ADDITIONS = (
    "p_move",
    "predicted_absolute_movement_bps",
    "movement_model_uncertainty",
)
STRUCTURAL_DIRECTION_COLUMNS = (
    "p_transition_burst_movement_conditioned",
    "p_short_closure_movement_conditioned",
)
FORBIDDEN_FRAGMENTS = (
    "loop_id",
    "selected_loop",
    "profitable_loop",
    "payoff",
    "future_",
    "outcome",
    "excursion_resolution",
    "model_score",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrow_hash(frame: pd.DataFrame) -> str:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: Any = None) -> None:
        self.checks.append({"check": name, "passed": bool(condition), "detail": detail})

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(bool(item["passed"]) for item in self.checks)


def loo_median(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.asarray([np.median(np.delete(array, index)) for index in range(len(array))])


def classify(origin: int, path: Sequence[int]) -> tuple[int, bool, int | None, int | None]:
    states = [int(origin), *(int(value) for value in path)]
    transitions = sum(left != right for left, right in zip(states[:-1], states[1:], strict=True))
    departure: int | None = None
    returned: int | None = None
    for step, state in enumerate(states[1:], start=1):
        if departure is None and state != origin:
            departure = step
        elif departure is not None and state == origin:
            returned = step
            break
    unique = (
        len(set(states[departure - 1 : returned + 1]))
        if departure is not None and returned is not None
        else None
    )
    closure = bool(
        departure is not None
        and returned is not None
        and transitions >= 2
        and unique is not None
        and unique <= 3
    )
    return transitions, closure, returned, unique


def manual_prediction(model: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    features = list(model["feature_names"])
    values = frame[features].to_numpy(dtype=float)
    transformed = (values - np.asarray(model["means"], dtype=float)) / np.asarray(
        model["scales"], dtype=float
    )
    linear = float(model["intercept"]) + transformed @ np.asarray(
        model["coefficients"], dtype=float
    )
    if model["kind"] == "ridge":
        return np.asarray(linear, dtype=float)
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))


def binary_metrics(target: pd.Series, probability: pd.Series) -> dict[str, float]:
    clipped = probability.clip(1e-12, 1.0 - 1e-12)
    return {
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
        "auc": float(roc_auc_score(target, probability)) if target.nunique() == 2 else math.nan,
    }


def independent_decision(evidence: Mapping[str, Any]) -> str:
    blocker = evidence.get("blocker")
    if blocker:
        return str(blocker)
    movement = bool(
        float(evidence["p1_minus_p0_brier_improvement"]) > 0.0
        and float(evidence["p1_minus_p0_log_loss_improvement"]) > 0.0
    )
    structural = bool(
        float(evidence["b1_minus_b0_brier_improvement"]) > 0.0
        and float(evidence["b1_minus_b0_log_loss_improvement"]) > 0.0
    )
    directional = bool(
        float(evidence["d1_minus_d0_brier_improvement"]) > 0.0
        and float(evidence["d1_minus_d0_log_loss_improvement"]) > 0.0
    )
    promising = bool(
        movement
        and structural
        and directional
        and int(evidence["b1_positive_months"]) >= 5
        and int(evidence["d1_positive_months"]) >= 5
        and float(evidence["b1_bootstrap_90_lower"]) >= 0.0
        and float(evidence["d1_bootstrap_90_lower"]) >= 0.0
        and float(evidence["b1_null_percentile"]) >= 0.90
        and float(evidence["d1_null_percentile"]) >= 0.90
        and float(evidence["path_spearman"]) > float(evidence["observable_spearman"])
        and float(evidence["path_top_one_minus_median"])
        > float(evidence["observable_top_one_minus_median"])
        and bool(evidence["concentration_passed"])
        and bool(evidence["exact_rerun_passed"])
        and bool(evidence["independent_audit_passed"])
    )
    if promising:
        return "promising_probability_chain_for_intensive_v1"
    if structural and not directional:
        return "structural_increment_without_directional_value"
    if movement and not structural:
        return "movement_predictable_but_no_structural_increment"
    return "no_incremental_probability_chain"


def shift_blocks(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    draw: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    output = frame.copy()
    rng = np.random.default_rng(seed + draw)
    groups: dict[tuple[int, tuple[str, ...]], list[str]] = {}
    for (ordinal, session), slate in frame.groupby(
        ["decision_ordinal", "session"], sort=True, observed=True
    ):
        membership = tuple(sorted(slate["symbol"].astype(str)))
        groups.setdefault((int(ordinal), membership), []).append(str(session))
    manifest: list[dict[str, Any]] = []
    for (ordinal, membership), raw_sessions in sorted(groups.items()):
        sessions = sorted(set(raw_sessions))
        offset = int(rng.integers(1, len(sessions))) if len(sessions) > 1 else 0
        for destination_position, destination in enumerate(sessions):
            source = sessions[(destination_position - offset) % len(sessions)]
            destination_index = frame.index[
                frame["decision_ordinal"].eq(ordinal) & frame["session"].astype(str).eq(destination)
            ]
            source_block = frame.loc[
                frame["decision_ordinal"].eq(ordinal) & frame["session"].astype(str).eq(source),
                ["symbol", *columns],
            ].set_index("symbol")
            destination_symbols = frame.loc[destination_index, "symbol"].astype(str)
            output.loc[destination_index, list(columns)] = source_block.loc[
                destination_symbols, list(columns)
            ].to_numpy()
            manifest.append(
                {
                    "draw": draw,
                    "decision_ordinal": ordinal,
                    "destination_session": destination,
                    "source_session": source,
                    "offset": offset,
                    "membership_size": len(membership),
                }
            )
    return output, manifest


def fit_logistic_independent(
    frame: pd.DataFrame,
    target: str,
    features: Sequence[str],
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    values = frame[list(features)].to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
    design = (values - means) / scales
    sizes = frame["slate_id"].groupby(frame["slate_id"], sort=True).transform("size")
    weights = 1.0 / sizes.to_numpy(dtype=float)
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(design, frame[target].to_numpy(dtype=int), sample_weight=weights)
    if int(np.max(model.n_iter_)) >= 250:
        raise RuntimeError("independent null logistic did not converge")
    return means, scales, model


def predict_independent(
    frame: pd.DataFrame,
    features: Sequence[str],
    means: np.ndarray,
    scales: np.ndarray,
    model: LogisticRegression,
) -> np.ndarray:
    design = (frame[list(features)].to_numpy(dtype=float) - means) / scales
    return np.asarray(model.predict_proba(design)[:, 1], dtype=float)


def audit_sources(
    audit: Audit,
    artifacts: Path,
    provider_root: Path,
    source_manifest: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> None:
    all_months: list[pd.DataFrame] = []
    for symbol, source in sorted(source_manifest["provider_sources"].items()):
        stored = "VTI.US" if symbol == "VTI" else symbol
        path = provider_root / f"symbol={stored}" / "timeframe=5m" / "data.parquet"
        frame = pd.read_parquet(
            path,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
            filters=[
                ("timestamp", ">=", START.to_pydatetime()),
                ("timestamp", "<", PROTECTED_START.to_pydatetime()),
            ],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        audit.check(
            f"source_{symbol}_protected_rows_zero", not frame["timestamp"].ge(PROTECTED_START).any()
        )
        audit.check(
            f"source_{symbol}_bounded_hash",
            arrow_hash(frame) == source["bounded_safe_hash"],
        )
        all_months.append(frame[["timestamp"]])
    combined = pd.concat(all_months, ignore_index=True)
    counts = combined["timestamp"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    audit.check(
        "protected_boundary_minimum",
        str(combined["timestamp"].min()) == boundary["minimum_timestamp_read"],
    )
    audit.check(
        "protected_boundary_maximum",
        str(combined["timestamp"].max()) == boundary["maximum_timestamp_read"],
    )
    audit.check(
        "protected_boundary_month_counts",
        {str(key): int(value) for key, value in counts.items()}
        == {str(key): int(value) for key, value in boundary["row_count_by_year_month"].items()},
    )


def audit_targets(audit: Audit, compact: pd.DataFrame) -> None:
    stock_return = compact["future_close"] / compact["decision_close"] - 1.0
    delayed = compact["future_close"] / compact["delayed_entry_open"] - 1.0
    expected_median = np.empty(len(compact), dtype=float)
    expected_delayed_median = np.empty(len(compact), dtype=float)
    for _, indices in compact.groupby("slate_id", sort=True).groups.items():
        index = np.asarray(list(indices), dtype=int)
        expected_median[index] = loo_median(stock_return.iloc[index])
        expected_delayed_median[index] = loo_median(delayed.iloc[index])
    residual = 10000.0 * (stock_return - expected_median)
    delayed_residual = 10000.0 * (delayed - expected_delayed_median)
    audit.check(
        "stock_return_recalculated", np.allclose(stock_return, compact["stock_return"], atol=1e-12)
    )
    audit.check(
        "cohort_relative_median_recalculated",
        np.allclose(expected_median, compact["cohort_median_return_minus_i"], atol=1e-12),
    )
    audit.check(
        "residual_return_recalculated",
        np.allclose(residual, compact["residual_return_bps"], atol=1e-9),
    )
    audit.check(
        "delayed_residual_recalculated",
        np.allclose(delayed_residual, compact["delayed_residual_return_bps"], atol=1e-9),
    )
    audit.check(
        "absolute_movement_recalculated",
        np.allclose(np.abs(residual), compact["absolute_movement_bps"], atol=1e-9),
    )
    thresholds = {
        ordinal: float(
            compact.loc[
                compact["year"].eq(2024) & compact["decision_ordinal"].eq(ordinal),
                "absolute_movement_bps",
            ].quantile(0.75, interpolation="linear")
        )
        for ordinal in (12, 36)
    }
    expected_large = compact["absolute_movement_bps"].ge(
        compact["decision_ordinal"].map(thresholds)
    )
    audit.check(
        "training_only_movement_thresholds",
        np.allclose(
            compact["movement_threshold_bps"],
            compact["decision_ordinal"].map(thresholds),
            atol=1e-12,
        ),
    )
    audit.check(
        "large_move_recalculated",
        np.array_equal(expected_large.astype(int), compact["large_move"].astype(int)),
    )
    audit.check(
        "direction_target_recalculated",
        np.array_equal(
            compact["residual_return_bps"].gt(0).astype(int), compact["up_given_move"].astype(int)
        ),
    )
    transition_values: list[int] = []
    closure_values: list[int] = []
    return_steps: list[float] = []
    unique_values: list[float] = []
    for row in compact.itertuples():
        path = [int(value) for value in str(row.future_state_path).split(",")]
        transitions, closure, returned, unique = classify(int(row.origin_state), path)
        transition_values.append(transitions)
        closure_values.append(int(closure))
        return_steps.append(float(returned) if returned is not None else math.nan)
        unique_values.append(float(unique) if unique is not None else math.nan)
    audit.check(
        "transition_counts_recalculated",
        np.array_equal(transition_values, compact["transition_count"].astype(int)),
    )
    audit.check(
        "short_closure_recalculated",
        np.array_equal(closure_values, compact["short_closure"].astype(int)),
    )
    audit.check("source_gap_crossing_zero", not compact["source_gap_crossed"].astype(bool).any())
    audit.check(
        "session_boundary_crossing_zero", not compact["session_boundary_crossed"].astype(bool).any()
    )


def audit_oof_and_features(
    audit: Audit,
    oof: pd.DataFrame,
    feature_manifest: Mapping[str, Any],
) -> None:
    prediction_columns = (
        "p_move",
        "predicted_absolute_movement_bps",
        "p_transition_burst_movement_conditioned",
        "p_short_closure_movement_conditioned",
        "p_up_given_move_observable",
        "p_up_given_move_with_path",
    )
    chronology_pass = True
    for prediction in prediction_columns:
        populated = oof[prediction].notna()
        trained = pd.to_datetime(oof.loc[populated, f"{prediction}__trained_through"])
        scored = pd.to_datetime(oof.loc[populated, "session"])
        chronology_pass &= bool(trained.lt(scored).all())
    audit.check("oof_and_stacking_chronology", chronology_pass)
    all_features: list[str] = []
    for model in ("P0", "P1", "B0", "B1", "C0", "C1", "D0", "D1"):
        all_features.extend(str(value) for value in feature_manifest[model])
    forbidden = sorted(
        feature
        for feature in set(all_features)
        if any(fragment in feature.lower() for fragment in FORBIDDEN_FRAGMENTS)
        or feature.lower() in {"symbol", "month", "exact_loop_id"}
    )
    audit.check("feature_matrices_forbidden_fields_absent", not forbidden, forbidden)
    audit.check("symbol_identity_absent", "symbol" not in all_features)
    audit.check("month_identity_absent", "month" not in all_features)


def audit_predictions_and_metrics(
    audit: Audit,
    artifacts: Path,
    scored: pd.DataFrame,
    coefficient_document: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    models = coefficient_document["models"]
    mappings = {
        "P0": "p_move_p0",
        "P1": "p_move",
        "P0_SIZE": "predicted_log_absolute_movement_p0",
        "P1_SIZE": "predicted_log_absolute_movement",
        "B0": "p_transition_burst_regime_only",
        "B1": "p_transition_burst_movement_conditioned",
        "C0": "p_short_closure_given_burst_regime_only",
        "C1": "p_short_closure_given_burst_movement_conditioned",
        "D0": "p_up_given_move_observable",
        "D1": "p_up_given_move_with_path",
    }
    for model_name, prediction_column in mappings.items():
        reconstructed = manual_prediction(models[model_name], scored)
        audit.check(
            f"manual_prediction_{model_name}",
            np.allclose(reconstructed, scored[prediction_column], atol=2e-12, rtol=1e-11),
        )
    audit.check(
        "ridge_absolute_movement_reconstruction",
        np.allclose(
            np.maximum(0.0, np.expm1(scored["predicted_log_absolute_movement"])),
            scored["predicted_absolute_movement_bps"],
            atol=1e-10,
        ),
    )
    expected_long_observable = scored["p_move"] * scored["p_up_given_move_observable"]
    expected_short_observable = scored["p_move"] * (1 - scored["p_up_given_move_observable"])
    expected_long_path = scored["p_move"] * scored["p_up_given_move_with_path"]
    expected_short_path = scored["p_move"] * (1 - scored["p_up_given_move_with_path"])
    expected_neutral = 1 - scored["p_move"]
    observable_score = (
        scored["p_move"]
        * (2 * scored["p_up_given_move_observable"] - 1)
        * scored["predicted_absolute_movement_bps"]
    )
    path_score = (
        scored["p_move"]
        * (2 * scored["p_up_given_move_with_path"] - 1)
        * scored["predicted_absolute_movement_bps"]
    )
    for name, expected in (
        ("p_long_observable", expected_long_observable),
        ("p_short_observable", expected_short_observable),
        ("p_long_with_path", expected_long_path),
        ("p_short_with_path", expected_short_path),
        ("p_neutral", expected_neutral),
        ("observable_chain_score", observable_score),
        ("path_chain_score", path_score),
    ):
        audit.check(f"chain_{name}", np.allclose(expected, scored[name], atol=1e-12))

    layer_specs = {
        "movement": ("large_move", "p_move_p0", "p_move", None),
        "burst": (
            "transition_burst",
            "p_transition_burst_regime_only",
            "p_transition_burst_movement_conditioned",
            None,
        ),
        "closure": (
            "short_closure",
            "p_short_closure_given_burst_regime_only",
            "p_short_closure_given_burst_movement_conditioned",
            scored["transition_burst"].eq(1),
        ),
        "direction": (
            "up_given_move",
            "p_up_given_move_observable",
            "p_up_given_move_with_path",
            scored["large_move"].eq(1),
        ),
    }
    increments: dict[str, dict[str, float]] = {}
    for layer, (target, baseline, candidate, mask) in layer_specs.items():
        selected = scored if mask is None else scored.loc[mask]
        base = binary_metrics(selected[target], selected[baseline])
        candidate_metrics = binary_metrics(selected[target], selected[candidate])
        metric_file = pd.read_csv(artifacts / f"{layer}_metrics.csv")
        overall = metric_file.loc[metric_file["scope"].eq("overall")].set_index("model")
        base_name, candidate_name = {
            "movement": ("P0", "P1"),
            "burst": ("B0", "B1"),
            "closure": ("C0", "C1"),
            "direction": ("D0", "D1"),
        }[layer]
        audit.check(
            f"{layer}_brier_logloss_auc",
            np.allclose(
                overall.loc[base_name, ["brier", "log_loss", "auc"]].to_numpy(float),
                [base["brier"], base["log_loss"], base["auc"]],
                atol=2e-10,
                equal_nan=True,
            )
            and np.allclose(
                overall.loc[candidate_name, ["brier", "log_loss", "auc"]].to_numpy(float),
                [
                    candidate_metrics["brier"],
                    candidate_metrics["log_loss"],
                    candidate_metrics["auc"],
                ],
                atol=2e-10,
                equal_nan=True,
            ),
        )
        increments[layer] = {
            "brier": base["brier"] - candidate_metrics["brier"],
            "log_loss": base["log_loss"] - candidate_metrics["log_loss"],
        }
    return increments


def recalc_ranking(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slate_id, slate in scored.groupby("slate_id", sort=True):
        ordered = slate.sort_values("symbol", kind="mergesort")
        median = float(ordered["residual_return_bps"].median())
        for model, score_column in (
            ("observable_chain", "observable_chain_score"),
            ("path_chain", "path_chain_score"),
        ):
            ranked = ordered.sort_values(
                [score_column, "symbol"], ascending=[False, True], kind="mergesort"
            )
            top_one = ranked.iloc[0]
            rows.append(
                {
                    "slate_id": slate_id,
                    "session": str(ordered["session"].iloc[0]),
                    "month": str(ordered["session"].iloc[0])[:7],
                    "decision_ordinal": int(ordered["decision_ordinal"].iloc[0]),
                    "model": model,
                    "spearman": float(
                        spearmanr(ordered[score_column], ordered["residual_return_bps"]).statistic
                    ),
                    "top_one_symbol": str(top_one["symbol"]),
                    "top_one_realised_residual_return_bps": float(top_one["residual_return_bps"]),
                    "top_one_minus_slate_median_bps": float(
                        top_one["residual_return_bps"] - median
                    ),
                    "top_two_average_minus_slate_median_bps": float(
                        ranked.iloc[:2]["residual_return_bps"].mean() - median
                    ),
                    "top_one_delayed_residual_return_bps": float(
                        top_one["delayed_residual_return_bps"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def audit_ranking_concentration_delayed(
    audit: Audit,
    artifacts: Path,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    expected = recalc_ranking(scored)
    actual = pd.read_csv(artifacts / "chain_ranking_metrics.csv")
    actual = actual.loc[actual["scope"].eq("slate")].sort_values(
        ["slate_id", "model"], kind="mergesort"
    )
    expected = expected.sort_values(["slate_id", "model"], kind="mergesort")
    numeric = (
        "spearman",
        "top_one_realised_residual_return_bps",
        "top_one_minus_slate_median_bps",
        "top_two_average_minus_slate_median_bps",
        "top_one_delayed_residual_return_bps",
    )
    audit.check(
        "chain_ranking_rows_recalculated",
        np.allclose(actual[list(numeric)], expected[list(numeric)], atol=2e-9, equal_nan=True)
        and actual["top_one_symbol"].tolist() == expected["top_one_symbol"].tolist(),
    )
    concentration = pd.read_csv(artifacts / "concentration_metrics.csv")
    expected_max_rows = float(scored["symbol"].value_counts(normalize=True).max())
    expected_max_top = float(
        expected.groupby("model", sort=True)["top_one_symbol"].value_counts(normalize=True).max()
    )
    actual_rows = float(
        concentration.loc[concentration["surface"].eq("decision_rows"), "fraction"].max()
    )
    actual_top = float(
        concentration.loc[
            concentration["surface"].eq("top_one") & concentration["group_type"].eq("stock"),
            "fraction",
        ].max()
    )
    audit.check(
        "concentration_recalculated",
        np.isclose(actual_rows, expected_max_rows, atol=1e-12)
        and np.isclose(actual_top, expected_max_top, atol=1e-12),
    )
    delayed = pd.read_csv(artifacts / "delayed_entry_sensitivity.csv")
    delayed_pass = True
    for row in delayed.itertuples():
        gross = expected.loc[expected["model"].eq(row.model), "top_one_delayed_residual_return_bps"]
        delayed_pass &= math.isclose(
            float(row.mean_top_one_delayed_residual_bps_after_synthetic_friction),
            float((gross - float(row.synthetic_round_trip_friction_bps)).mean()),
            abs_tol=2e-9,
        )
    audit.check("delayed_entry_sensitivity_recalculated", delayed_pass)
    return expected


def audit_monthly_calibration(
    audit: Audit,
    artifacts: Path,
    scored: pd.DataFrame,
) -> None:
    monthly = pd.read_csv(artifacts / "monthly_metrics.csv")
    specs = {
        "movement": ("large_move", "p_move_p0", "p_move"),
        "burst": (
            "transition_burst",
            "p_transition_burst_regime_only",
            "p_transition_burst_movement_conditioned",
        ),
        "closure": (
            "short_closure",
            "p_short_closure_given_burst_regime_only",
            "p_short_closure_given_burst_movement_conditioned",
        ),
        "direction": (
            "up_given_move",
            "p_up_given_move_observable",
            "p_up_given_move_with_path",
        ),
    }
    pass_monthly = True
    for row in monthly.itertuples():
        target, baseline, candidate = specs[row.layer]
        selected = scored.loc[scored["session"].astype(str).str[:7].eq(row.month)]
        if row.layer == "closure":
            selected = selected.loc[selected["transition_burst"].eq(1)]
        if row.layer == "direction":
            selected = selected.loc[selected["large_move"].eq(1)]
        base = binary_metrics(selected[target], selected[baseline])
        candidate_metrics = binary_metrics(selected[target], selected[candidate])
        pass_monthly &= np.allclose(
            [
                row.baseline_brier,
                row.candidate_brier,
                row.baseline_log_loss,
                row.candidate_log_loss,
            ],
            [
                base["brier"],
                candidate_metrics["brier"],
                base["log_loss"],
                candidate_metrics["log_loss"],
            ],
            atol=2e-9,
        )
    audit.check("monthly_metrics_recalculated", pass_monthly)
    calibration = pd.read_csv(artifacts / "calibration_bins.csv")
    audit.check(
        "calibration_has_ten_bins_per_model",
        calibration.groupby(["layer", "model"]).size().eq(10).all(),
    )
    calibration_pass = True
    model_specs = {
        ("movement", "P0"): ("large_move", "p_move_p0", None),
        ("movement", "P1"): ("large_move", "p_move", None),
        ("burst", "B0"): ("transition_burst", "p_transition_burst_regime_only", None),
        ("burst", "B1"): ("transition_burst", "p_transition_burst_movement_conditioned", None),
        ("closure", "C0"): ("short_closure", "p_short_closure_given_burst_regime_only", "burst"),
        ("closure", "C1"): (
            "short_closure",
            "p_short_closure_given_burst_movement_conditioned",
            "burst",
        ),
        ("direction", "D0"): ("up_given_move", "p_up_given_move_observable", "move"),
        ("direction", "D1"): ("up_given_move", "p_up_given_move_with_path", "move"),
    }
    for row in calibration.itertuples():
        target, probability, population = model_specs[(row.layer, row.model)]
        selected = scored
        if population == "burst":
            selected = selected.loc[selected["transition_burst"].eq(1)]
        elif population == "move":
            selected = selected.loc[selected["large_move"].eq(1)]
        bins = pd.cut(
            selected[probability], np.linspace(0, 1, 11), include_lowest=True, labels=False
        )
        local = selected.loc[bins.eq(int(row.bin) - 1)]
        expected_prediction = float(local[probability].mean()) if len(local) else math.nan
        expected_rate = float(local[target].mean()) if len(local) else math.nan
        calibration_pass &= int(row.rows) == len(local)
        calibration_pass &= bool(
            np.isclose(row.mean_prediction, expected_prediction, atol=2e-9, equal_nan=True)
        )
        calibration_pass &= bool(
            np.isclose(row.observed_rate, expected_rate, atol=2e-9, equal_nan=True)
        )
    audit.check("calibration_values_recalculated", calibration_pass)


def audit_bootstrap(
    audit: Audit,
    artifacts: Path,
    scored: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    sessions = np.asarray(sorted(scored["session"].astype(str).unique()), dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: dict[str, list[float]] = {
        "p1_minus_p0_brier_improvement": [],
        "b1_minus_b0_brier_improvement": [],
        "c1_minus_c0_brier_improvement": [],
        "d1_minus_d0_brier_improvement": [],
        "path_minus_observable_spearman": [],
        "path_minus_observable_top_one": [],
    }
    ranking_session = (
        ranking.groupby(["session", "model"], sort=True)
        .agg(spearman=("spearman", "mean"), top_one=("top_one_minus_slate_median_bps", "mean"))
        .reset_index()
    )
    for _draw in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(sessions, size=len(sessions), replace=True).astype(str).tolist()
        counts = Counter(sample)
        weights = scored["session"].astype(str).map(counts).to_numpy(float)

        def difference(
            target: str,
            base: str,
            candidate: str,
            mask: pd.Series | None = None,
            *,
            draw_weights: np.ndarray = weights,
        ) -> float:
            selected = np.ones(len(scored), dtype=bool) if mask is None else mask.to_numpy(bool)
            selected &= draw_weights > 0
            truth = scored.loc[selected, target].to_numpy(float)
            base_values = scored.loc[selected, base].to_numpy(float)
            candidate_values = scored.loc[selected, candidate].to_numpy(float)
            local_weights = draw_weights[selected]
            return float(
                np.average((truth - base_values) ** 2, weights=local_weights)
                - np.average((truth - candidate_values) ** 2, weights=local_weights)
            )

        values["p1_minus_p0_brier_improvement"].append(
            difference("large_move", "p_move_p0", "p_move")
        )
        values["b1_minus_b0_brier_improvement"].append(
            difference(
                "transition_burst",
                "p_transition_burst_regime_only",
                "p_transition_burst_movement_conditioned",
            )
        )
        values["c1_minus_c0_brier_improvement"].append(
            difference(
                "short_closure",
                "p_short_closure_given_burst_regime_only",
                "p_short_closure_given_burst_movement_conditioned",
                scored["transition_burst"].eq(1),
            )
        )
        values["d1_minus_d0_brier_improvement"].append(
            difference(
                "up_given_move",
                "p_up_given_move_observable",
                "p_up_given_move_with_path",
                scored["large_move"].eq(1),
            )
        )
        selected_ranking = ranking_session.loc[ranking_session["session"].isin(counts)].copy()
        selected_ranking["weight"] = selected_ranking["session"].map(counts).astype(float)
        aggregated: dict[str, dict[str, float]] = {}
        for model, model_frame in selected_ranking.groupby("model", sort=True):
            aggregated[model] = {
                "spearman": float(
                    np.average(model_frame["spearman"], weights=model_frame["weight"])
                ),
                "top_one": float(np.average(model_frame["top_one"], weights=model_frame["weight"])),
            }
        values["path_minus_observable_spearman"].append(
            aggregated["path_chain"]["spearman"] - aggregated["observable_chain"]["spearman"]
        )
        values["path_minus_observable_top_one"].append(
            aggregated["path_chain"]["top_one"] - aggregated["observable_chain"]["top_one"]
        )
    stored = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    passed = True
    for metric, raw in values.items():
        actual = (
            stored.loc[stored["metric"].eq(metric) & stored["draw"].ge(0)]
            .sort_values("draw")["value"]
            .to_numpy(float)
        )
        passed &= np.allclose(actual, raw, atol=2e-9)
        summary = stored.loc[stored["metric"].eq(metric) & stored["draw"].eq(-1)].iloc[0]
        passed &= np.allclose(
            [summary.ci90_lower, summary.ci90_upper, summary.ci95_lower, summary.ci95_upper],
            [
                np.quantile(raw, 0.05),
                np.quantile(raw, 0.95),
                np.quantile(raw, 0.025),
                np.quantile(raw, 0.975),
            ],
            atol=2e-9,
        )
    audit.check("bootstrap_500_session_blocks_recalculated", passed)


def audit_nulls(
    audit: Audit,
    artifacts: Path,
    oof: pd.DataFrame,
    scored: pd.DataFrame,
    coefficient_document: Mapping[str, Any],
) -> None:
    feature_document = coefficient_document["models"]
    b1_features = tuple(feature_document["B1"]["feature_names"])
    d1_features = tuple(feature_document["D1"]["feature_names"])
    burst_train = oof.loc[oof["p_move"].notna()].copy()
    direction_train = oof.loc[
        oof["p_short_closure_movement_conditioned"].notna() & oof["large_move"].eq(1)
    ].copy()
    stored = pd.read_csv(artifacts / "null_metrics.csv")
    passed = True
    for draw in range(NULL_DRAWS):
        shifted_train, train_manifest = shift_blocks(
            burst_train, B1_ADDITIONS, draw=draw, seed=NULL_SEED
        )
        shifted_score, score_manifest = shift_blocks(
            scored, B1_ADDITIONS, draw=draw, seed=NULL_SEED + 10_000
        )
        means, scales, model = fit_logistic_independent(
            shifted_train, "transition_burst", b1_features, random_state=20260720 + draw
        )
        candidate = predict_independent(shifted_score, b1_features, means, scales, model)
        truth = scored["transition_burst"].to_numpy(float)
        baseline = scored["p_transition_burst_regime_only"].to_numpy(float)
        value = float(np.mean((truth - baseline) ** 2) - np.mean((truth - candidate) ** 2))
        row = stored.loc[
            stored["comparison"].eq("B1_minus_B0_brier_improvement") & stored["draw"].eq(draw)
        ].iloc[0]
        passed &= math.isclose(value, float(row.null_value), abs_tol=2e-9)
        passed &= (
            row.train_shift_manifest_hash
            == hashlib.sha256(canonical_json(train_manifest).encode()).hexdigest()
        )
        passed &= (
            row.score_shift_manifest_hash
            == hashlib.sha256(canonical_json(score_manifest).encode()).hexdigest()
        )

        shifted_direction_train, direction_train_manifest = shift_blocks(
            direction_train, STRUCTURAL_DIRECTION_COLUMNS, draw=draw, seed=NULL_SEED + 20_000
        )
        shifted_direction_score, direction_score_manifest = shift_blocks(
            scored, STRUCTURAL_DIRECTION_COLUMNS, draw=draw, seed=NULL_SEED + 30_000
        )
        means, scales, model = fit_logistic_independent(
            shifted_direction_train, "up_given_move", d1_features, random_state=20260820 + draw
        )
        candidate = predict_independent(shifted_direction_score, d1_features, means, scales, model)
        mask = scored["large_move"].eq(1).to_numpy(bool)
        truth = scored.loc[mask, "up_given_move"].to_numpy(float)
        baseline = scored.loc[mask, "p_up_given_move_observable"].to_numpy(float)
        value = float(np.mean((truth - baseline) ** 2) - np.mean((truth - candidate[mask]) ** 2))
        row = stored.loc[
            stored["comparison"].eq("D1_minus_D0_brier_improvement") & stored["draw"].eq(draw)
        ].iloc[0]
        passed &= math.isclose(value, float(row.null_value), abs_tol=2e-9)
        passed &= (
            row.train_shift_manifest_hash
            == hashlib.sha256(canonical_json(direction_train_manifest).encode()).hexdigest()
        )
        passed &= (
            row.score_shift_manifest_hash
            == hashlib.sha256(canonical_json(direction_score_manifest).encode()).hexdigest()
        )
    audit.check("null_100_whole_session_shifts_recalculated", passed)


def run_audit(artifacts: Path, provider_root: Path) -> dict[str, Any]:
    audit = Audit()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    decision = json.loads((artifacts / "decision.json").read_text(encoding="utf-8"))
    feature_document = json.loads((artifacts / "feature_manifest.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((artifacts / "source_manifest.json").read_text(encoding="utf-8"))
    boundary = json.loads((artifacts / "protected_boundary_audit.json").read_text(encoding="utf-8"))
    coefficients = json.loads((artifacts / "model_coefficients.json").read_text(encoding="utf-8"))
    input_hashes = json.loads(
        (artifacts / "input_artifact_hashes.json").read_text(encoding="utf-8")
    )
    for document_name, document in (
        ("contract", contract),
        ("decision", decision),
        ("feature_manifest", feature_document),
        ("source_manifest", source_manifest),
        ("model_coefficients", coefficients),
    ):
        audit.check(
            f"{document_name}_safety_flags",
            all(document.get(key) == value for key, value in SAFETY_FLAGS.items()),
        )
    hash_pass = True
    for relative, expected_hash in input_hashes.items():
        path = REPO_ROOT / relative
        hash_pass &= path.is_file() and sha256_file(path) == expected_hash
    audit.check("input_artifact_hashes", hash_pass)

    compact = pd.read_parquet(artifacts / "compact_decision_panel.parquet")
    oof = pd.read_parquet(artifacts / "oof_2024_predictions.parquet")
    scored = pd.read_parquet(artifacts / "scored_2025_predictions.parquet")
    audit.check("fixed_decision_ordinals", set(compact["decision_ordinal"].astype(int)) == {12, 36})
    audit.check("maximum_25000_decision_rows", len(compact) <= 25_000)
    audit.check(
        "protected_compact_rows_zero",
        not pd.to_datetime(compact["decision_timestamp_utc"], utc=True).ge(PROTECTED_START).any(),
    )
    audit.check(
        "protected_scored_rows_zero",
        not pd.to_datetime(scored["decision_timestamp_utc"], utc=True).ge(PROTECTED_START).any(),
    )
    audit_sources(audit, artifacts, provider_root, source_manifest, boundary)
    audit_targets(audit, compact)
    audit_oof_and_features(audit, oof, feature_document)
    increments = audit_predictions_and_metrics(audit, artifacts, scored, coefficients)
    ranking = audit_ranking_concentration_delayed(audit, artifacts, scored)
    audit_monthly_calibration(audit, artifacts, scored)
    audit_bootstrap(audit, artifacts, scored, ranking)
    audit_nulls(audit, artifacts, oof, scored, coefficients)

    evidence = decision["evidence"]
    audit.check("decision_logic", independent_decision(evidence) == decision["decision"])
    audit.check(
        "decision_metric_binding",
        np.isclose(
            evidence["p1_minus_p0_brier_improvement"], increments["movement"]["brier"], atol=2e-10
        )
        and np.isclose(
            evidence["b1_minus_b0_brier_improvement"], increments["burst"]["brier"], atol=2e-10
        )
        and np.isclose(
            evidence["d1_minus_d0_brier_improvement"], increments["direction"]["brier"], atol=2e-10
        ),
    )
    result = {
        **SAFETY_FLAGS,
        "auditor_imports_runner": False,
        "historical_regime_refit_performed": False,
        "passed": audit.passed,
        "check_count": len(audit.checks),
        "failed_checks": [item["check"] for item in audit.checks if not item["passed"]],
        "checks": audit.checks,
    }
    write_json(artifacts / "independent_audit.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=Path.home()
        / "StockerLocal"
        / "data"
        / "processed"
        / "source=eodhd"
        / "instrument_type=stock",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_audit(args.artifacts.resolve(), args.provider_root.expanduser().resolve())
    except Exception as error:
        result = {
            **SAFETY_FLAGS,
            "auditor_imports_runner": False,
            "historical_regime_refit_performed": False,
            "passed": False,
            "check_count": 0,
            "failed_checks": ["auditor_exception"],
            "exception_type": type(error).__name__,
            "exception": str(error),
        }
        args.artifacts.mkdir(parents=True, exist_ok=True)
        write_json(args.artifacts / "independent_audit.json", result)
    print(canonical_json(result), end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
