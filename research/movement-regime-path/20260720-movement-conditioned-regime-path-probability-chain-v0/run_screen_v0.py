#!/usr/bin/env python3
"""Run the bounded Movement-Conditioned Regime-Path Probability Chain V0 screen."""

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
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from stocker_research.movement_regime_path_screen_v0 import (
    SAFETY_FLAGS,
    FrozenLinearModel,
    assert_allowed_feature_names,
    assert_stacking_chronology,
    bounded_monthly_smoke_population,
    circular_shift_session_blocks,
    classify_state_path,
    decide_screen,
    expanding_month_folds,
    fit_fixed_logistic,
    fit_fixed_ridge,
    leave_one_out_median,
    movement_thresholds,
    probability_chain,
    sampled_sessions,
)
from stocker_research.regime_gap_segmentation_v2 import causal_segment_groups
from stocker_research.regime_panel_v2 import (
    EMISSION_FEATURES,
    NATURAL_KEY,
    RegimePanelConfig,
    bounded_source_hash,
    build_regime_panel,
    canonical_frame_hash,
    provider_path,
)
from stocker_research.regime_validity_v2 import (
    EmissionPreprocessing,
    SemiMarkovParameters,
    causal_filter_summary,
    gaussian_log_emissions,
    transform_emissions,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
DEFAULT_EXACT = EXPERIMENT_DIR / "artifacts" / "exact_rerun"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
AUDITOR_PATH = EXPERIMENT_DIR / "audit_screen_v0.py"

SLRNO_WORK = REPO_ROOT / "research" / "slrno-v2" / "20260714-regime-loop-handoff" / "work"
REFIT_DIR = SLRNO_WORK / "artifacts" / "20260719-right-censored-regime-refit-v2" / "primary"
ATLAS_CONTRACT = SLRNO_WORK / "contracts" / "20260717-directional-signature-atlas-v1.json"
REFIT_CONTRACT = SLRNO_WORK / "contracts" / "20260719-right-censored-regime-refit-v2.json"
PARAMETERS_PATH = REFIT_DIR / "full_refit_parameters.npz"
PREPROCESSING_PATH = REFIT_DIR / "full_refit_preprocessing.csv"
POSTERIOR_AUDIT_PATH = REFIT_DIR / "posterior_audit_input.parquet"
REFIT_CONFIG_PATH = REFIT_DIR / "full_refit_effective_configuration.json"
PANEL_HASHES_PATH = REFIT_DIR / "panel_hashes.json"
SOURCE_IDENTITY_PATH = REFIT_DIR / "pre_repair_source_identity.json"

START = pd.Timestamp("2024-01-01T00:00:00Z")
DEVELOPMENT_END_EXCLUSIVE = pd.Timestamp("2025-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
READ_END_INCLUSIVE = PROTECTED_START - pd.Timedelta(nanoseconds=1)
DECISION_ORDINALS = (12, 36)
HORIZON = 24
BOOTSTRAP_DRAWS = 500
NULL_DRAWS = 100
BOOTSTRAP_SEED = 20260720
NULL_SEED = 20260721

DECISION_SYMBOLS = (
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
REGIME_CONTEXT_SYMBOLS = (
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

MOVEMENT_FEATURES = (
    "absolute_return_1",
    "absolute_return_3",
    "absolute_return_6",
    "absolute_return_12",
    "realized_volatility_3",
    "realized_volatility_6",
    "realized_volatility_12",
    "current_true_range_bps",
    "mean_true_range_bps_6",
    "mean_true_range_bps_12",
    "current_session_absolute_return",
    "current_session_range_bps",
    "distance_from_session_high_bps",
    "historical_activity_proxy_shock",
    "cross_sectional_dispersion_return_6",
)
P0_FEATURES = (
    "decision_ordinal_indicator",
    "realized_volatility_12",
    "current_session_range_bps",
)
DIRECTION_FEATURES = (
    "cohort_relative_return_1_bps",
    "cohort_relative_return_3_bps",
    "cohort_relative_return_6_bps",
    "cohort_relative_return_12_bps",
    "recent_acceleration",
    "close_location_current_bar",
    "wick_imbalance",
    "signed_distance_from_session_high_bps",
    "signed_distance_from_session_low_bps",
    "cohort_breadth_return_6_positive",
)
POSTERIOR_FEATURES = tuple(f"posterior_state_{state}" for state in range(8))
REGIME_FEATURES = (
    *POSTERIOR_FEATURES,
    "posterior_entropy",
    "maximum_posterior_probability",
    "current_hard_state_age",
    "completed_transitions_previous_6",
    "completed_transitions_previous_12",
    "previous_completed_state_run_duration",
    "scheduled_bars_remaining",
)
B1_ADDITIONS = (
    "p_move",
    "predicted_absolute_movement_bps",
    "movement_model_uncertainty",
)
C1_ADDITIONS = (
    "p_move",
    "predicted_absolute_movement_bps",
    "p_transition_burst_movement_conditioned",
)
D0_FEATURES = (*DIRECTION_FEATURES, "p_move", "predicted_absolute_movement_bps")
D1_FEATURES = (
    *D0_FEATURES,
    "p_transition_burst_movement_conditioned",
    "p_short_closure_movement_conditioned",
    "posterior_entropy",
    "current_hard_state_age",
)
ALL_CAUSAL_FEATURES = tuple(
    dict.fromkeys((*P0_FEATURES, *MOVEMENT_FEATURES, *DIRECTION_FEATURES, *REGIME_FEATURES))
)
DECISION_CROSS_SECTION_FEATURES = (
    "decision_ordinal_indicator",
    "cohort_relative_return_1_bps",
    "cohort_relative_return_3_bps",
    "cohort_relative_return_6_bps",
    "cohort_relative_return_12_bps",
    "cohort_breadth_return_6_positive",
    "cross_sectional_dispersion_return_6",
)
ORIGIN_CAUSAL_FEATURES = tuple(
    feature for feature in ALL_CAUSAL_FEATURES if feature not in DECISION_CROSS_SECTION_FEATURES
)

EXPECTED_DEVELOPMENT_SNAPSHOT = "48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661"
EXPECTED_PANEL_HASH = "801c0bf9d69ecdd58b21fb2ba4392137048b466668344ebfc4c8faf6a0d3e2f1"
EXPECTED_MODEL_HASH = "4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def logical_source_path(symbol: str) -> str:
    stored = "VTI.US" if symbol == "VTI" else symbol
    return f"source=eodhd/instrument_type=stock/symbol={stored}/timeframe=5m/data.parquet"


def _load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected or contract.get("safety", {}).get(key) != expected:
            raise RuntimeError(f"contract safety flag differs: {key}")
    if tuple(contract["population"]["symbols"]) != DECISION_SYMBOLS:
        raise RuntimeError("contract decision cohort differs")
    if tuple(contract["population"]["decision_ordinals"]) != DECISION_ORDINALS:
        raise RuntimeError("contract fixed clocks differ")
    if int(contract["population"]["horizon_bars"]) != HORIZON:
        raise RuntimeError("contract horizon differs")
    return cast(dict[str, Any], contract)


def _required_inputs() -> tuple[Path, ...]:
    return (
        ATLAS_CONTRACT,
        REFIT_CONTRACT,
        PARAMETERS_PATH,
        PREPROCESSING_PATH,
        POSTERIOR_AUDIT_PATH,
        REFIT_CONFIG_PATH,
        PANEL_HASHES_PATH,
        SOURCE_IDENTITY_PATH,
        REPO_ROOT / "packages/stocker_research/src/stocker_research/regime_panel_v2.py",
        REPO_ROOT / "packages/stocker_research/src/stocker_research/regime_validity_v2.py",
    )


def _load_frozen_model() -> tuple[EmissionPreprocessing, SemiMarkovParameters]:
    preprocessing_frame = pd.read_csv(PREPROCESSING_PATH)
    preprocessing = EmissionPreprocessing(
        feature_names=tuple(preprocessing_frame["feature"].astype(str)),
        medians=preprocessing_frame["imputer_median"].to_numpy(dtype=float),
        centers=preprocessing_frame["scaler_center"].to_numpy(dtype=float),
        scales=preprocessing_frame["scaler_scale"].to_numpy(dtype=float),
    )
    preprocessing.validate()
    with np.load(PARAMETERS_PATH) as stored:
        parameters = SemiMarkovParameters(
            means=np.asarray(stored["means"]).copy(),
            variances=np.asarray(stored["variances"]).copy(),
            duration_hazard=np.asarray(stored["duration_hazard"]).copy(),
            transitions=np.asarray(stored["transitions"]).copy(),
            initial=np.asarray(stored["initial"]).copy(),
            occupancy=np.asarray(stored["occupancy"]).copy(),
        )
        stored_model_hash = str(np.asarray(stored["state_model_hash"]).item())
    parameters.validate()
    if stored_model_hash != EXPECTED_MODEL_HASH:
        raise RuntimeError("frozen full-refit model hash differs")
    if preprocessing.feature_names != tuple(EMISSION_FEATURES):
        raise RuntimeError("frozen preprocessing feature order differs")
    return preprocessing, parameters


def reproduce_frozen_posterior(
    preprocessing: EmissionPreprocessing, parameters: SemiMarkovParameters
) -> dict[str, Any]:
    frame = pd.read_parquet(POSTERIOR_AUDIT_PATH)
    scaled = transform_emissions(frame, preprocessing)
    summary = causal_filter_summary(
        gaussian_log_emissions(scaled, parameters),
        groups=causal_segment_groups(frame),
        model=parameters.as_dict(),
    )
    expected_probabilities = frame[[f"state_probability_{state}" for state in range(8)]].to_numpy(
        dtype=float
    )
    result = {
        "rows": len(frame),
        "hard_state_agreement": float(
            np.mean(summary.hard_states == frame["state"].to_numpy(dtype=int))
        ),
        "maximum_probability_absolute_error": float(
            np.max(np.abs(summary.state_probabilities - expected_probabilities))
        ),
        "maximum_expected_age_absolute_error": float(
            np.max(np.abs(summary.expected_age - frame["age"].to_numpy(dtype=float)))
        ),
        "maximum_entropy_absolute_error": float(
            np.max(
                np.abs(summary.posterior_entropy - frame["posterior_entropy"].to_numpy(dtype=float))
            )
        ),
    }
    if (
        result["hard_state_agreement"] != 1.0
        or result["maximum_probability_absolute_error"] > 1e-8
        or result["maximum_expected_age_absolute_error"] > 1e-8
        or result["maximum_entropy_absolute_error"] > 1e-8
    ):
        raise RuntimeError("frozen posterior probabilities cannot be reproduced")
    return result


def _verify_development_sources(provider_root: Path) -> dict[str, Any]:
    source_identity = json.loads(SOURCE_IDENTITY_PATH.read_text(encoding="utf-8"))
    expected = source_identity["development_source_hashes"]
    actual: dict[str, str] = {}
    counts: dict[str, int] = {}
    for symbol in (*REGIME_CONTEXT_SYMBOLS, "VTI"):
        digest, rows = bounded_source_hash(
            provider_path(provider_root, symbol),
            start=START,
            end=DEVELOPMENT_END_EXCLUSIVE - pd.Timedelta(nanoseconds=1),
        )
        actual[symbol] = digest
        counts[symbol] = rows
        if digest != expected[symbol]:
            raise RuntimeError(f"2024 bounded source hash differs for {symbol}")
    snapshot = hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if snapshot != EXPECTED_DEVELOPMENT_SNAPSHOT:
        raise RuntimeError("2024 bounded source snapshot differs")
    return {"hashes": actual, "row_counts": counts, "snapshot_hash": snapshot}


def _safe_source_month_counts(provider_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    parts: list[pd.DataFrame] = []
    safe_hashes: dict[str, str] = {}
    for symbol in (*REGIME_CONTEXT_SYMBOLS, "VTI"):
        path = provider_path(provider_root, symbol)
        frame = pd.read_parquet(
            path,
            columns=["timestamp"],
            filters=[
                ("timestamp", ">=", START.to_pydatetime()),
                ("timestamp", "<", PROTECTED_START.to_pydatetime()),
            ],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        if frame["timestamp"].ge(PROTECTED_START).any():
            raise RuntimeError("protected provider row materialised")
        frame["symbol"] = symbol
        parts.append(frame)
        safe_hashes[symbol] = bounded_source_hash(path, start=START, end=READ_END_INCLUSIVE)[0]
    all_rows = pd.concat(parts, ignore_index=True)
    all_rows["year_month"] = all_rows["timestamp"].dt.strftime("%Y-%m")
    counts = all_rows.groupby("year_month", sort=True).size().rename("row_count").reset_index()
    counts["minimum_timestamp"] = str(all_rows["timestamp"].min())
    counts["maximum_timestamp"] = str(all_rows["timestamp"].max())
    return counts, safe_hashes


def _hard_state_history(panel: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    output = panel.copy()
    output["hard_state"] = states.astype(np.int16)
    output["current_hard_state_age"] = 0
    output["completed_transitions_previous_6"] = 0
    output["completed_transitions_previous_12"] = 0
    output["previous_completed_state_run_duration"] = 0
    for _, positions in output.groupby("segment_id", sort=False).groups.items():
        index = np.asarray(list(positions), dtype=np.int64)
        local = output.loc[index, "hard_state"].to_numpy(dtype=int)
        changes = np.r_[0, (local[1:] != local[:-1]).astype(int)]
        age = np.empty(len(local), dtype=np.int16)
        previous_duration = np.zeros(len(local), dtype=np.int16)
        current_age = 0
        last_completed = 0
        for cursor in range(len(local)):
            if cursor == 0 or local[cursor] != local[cursor - 1]:
                if cursor > 0:
                    last_completed = current_age
                current_age = 1
            else:
                current_age += 1
            age[cursor] = current_age
            previous_duration[cursor] = last_completed
        transition_series = pd.Series(changes, dtype=float)
        output.loc[index, "current_hard_state_age"] = age
        output.loc[index, "previous_completed_state_run_duration"] = previous_duration
        output.loc[index, "completed_transitions_previous_6"] = (
            transition_series.rolling(6, min_periods=1).sum().to_numpy(dtype=np.int16)
        )
        output.loc[index, "completed_transitions_previous_12"] = (
            transition_series.rolling(12, min_periods=1).sum().to_numpy(dtype=np.int16)
        )
    return output


def _add_price_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    grouped = frame.groupby("segment_id", sort=False)
    previous_close = grouped["close"].shift(1)
    one_return = frame["close"] / previous_close - 1.0
    frame["simple_return_1"] = one_return
    for window in (1, 3, 6, 12):
        if window == 1:
            cumulative = one_return
        else:
            cumulative = frame["close"] / grouped["close"].shift(window) - 1.0
        frame[f"simple_return_{window}"] = cumulative
        frame[f"absolute_return_{window}"] = cumulative.abs()
    grouped = frame.groupby("segment_id", sort=False)
    for window in (3, 6, 12):
        frame[f"realized_volatility_{window}"] = grouped["simple_return_1"].transform(
            lambda values, size=window: values.rolling(size, min_periods=size).std(ddof=0)
        )
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["current_true_range_bps"] = 10000.0 * true_range / previous_close
    for window in (6, 12):
        frame[f"mean_true_range_bps_{window}"] = grouped["current_true_range_bps"].transform(
            lambda values, size=window: values.rolling(size, min_periods=size).mean()
        )
    session_group = frame.groupby(["symbol", "session"], sort=False)
    session_open = session_group["open"].transform("first")
    session_high = session_group["high"].cummax()
    session_low = session_group["low"].cummin()
    frame["current_session_absolute_return"] = (frame["close"] / session_open - 1.0).abs()
    frame["current_session_range_bps"] = 10000.0 * (session_high - session_low) / session_open
    frame["distance_from_session_high_bps"] = (
        10000.0 * (session_high - frame["close"]) / session_high
    )
    frame["signed_distance_from_session_high_bps"] = 10000.0 * (frame["close"] / session_high - 1.0)
    frame["signed_distance_from_session_low_bps"] = 10000.0 * (frame["close"] / session_low - 1.0)
    frame["historical_activity_proxy_shock"] = frame["log_relative_historical_volume"]
    frame["close_location_current_bar"] = frame["close_location_value"]
    frame["wick_imbalance"] = frame["lower_wick_pct_of_range"] - frame["upper_wick_pct_of_range"]
    latest_six = grouped["simple_return_1"].transform(
        lambda values: values.rolling(6, min_periods=6).sum()
    )
    preceding_six = latest_six.groupby(frame["segment_id"], sort=False).shift(6)
    frame["recent_acceleration"] = latest_six - preceding_six
    return frame


def _add_decision_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for window in (1, 3, 6, 12):
        column = f"simple_return_{window}"
        destination = f"cohort_relative_return_{window}_bps"
        output[destination] = np.nan
        for _, indices in output.groupby("slate_id", sort=True).groups.items():
            index = list(indices)
            values = output.loc[index, column].to_numpy(dtype=float)
            if np.isfinite(values).all() and len(values) >= 2:
                output.loc[index, destination] = 10000.0 * (values - leave_one_out_median(values))
    grouped = output.groupby("slate_id", sort=True)["simple_return_6"]
    output["cohort_breadth_return_6_positive"] = grouped.transform(
        lambda values: float((values > 0.0).mean())
    )
    output["cross_sectional_dispersion_return_6"] = grouped.transform(
        lambda values: float(values.std(ddof=1))
    )
    return output


def build_compact_panel(
    provider_root: Path,
    preprocessing: EmissionPreprocessing,
    parameters: SemiMarkovParameters,
    *,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = RegimePanelConfig(
        provider_root=provider_root,
        symbols=REGIME_CONTEXT_SYMBOLS,
        benchmark_symbol="VTI",
        start=START,
        end=READ_END_INCLUSIVE,
    )
    built = build_regime_panel(config)
    panel = built.frame
    if panel["bar_start_timestamp"].ge(PROTECTED_START).any():
        raise RuntimeError("protected row entered regime panel")
    development = panel.loc[panel["bar_start_timestamp"].lt(DEVELOPMENT_END_EXCLUSIVE)]
    development_hash = canonical_frame_hash(
        development,
        columns=(*NATURAL_KEY, *EMISSION_FEATURES),
    )
    if development_hash != EXPECTED_PANEL_HASH:
        raise RuntimeError("archived 2024 panel feature hash differs")
    scaled = transform_emissions(panel, preprocessing)
    summary = causal_filter_summary(
        gaussian_log_emissions(scaled, parameters),
        groups=causal_segment_groups(panel),
        model=parameters.as_dict(),
    )
    panel = _hard_state_history(panel, summary.hard_states)
    panel = _add_price_features(panel)
    panel["posterior_entropy"] = summary.posterior_entropy
    panel["maximum_posterior_probability"] = summary.state_probabilities.max(axis=1)
    for state in range(8):
        panel[f"posterior_state_{state}"] = summary.state_probabilities[:, state]
    panel["scheduled_bars_remaining"] = (
        panel["expected_session_bars"].astype(int) - panel["bar_ordinal"].astype(int) - 1
    )

    rows: list[dict[str, Any]] = []
    decision_symbols = set(DECISION_SYMBOLS)
    for (symbol, session), session_frame in panel.groupby(["symbol", "session"], sort=True):
        if str(symbol) not in decision_symbols:
            continue
        by_ordinal = {
            int(row.bar_ordinal): row
            for row in session_frame.sort_values("bar_ordinal", kind="mergesort").itertuples()
        }
        for ordinal in DECISION_ORDINALS:
            required_ordinals = range(ordinal - 12, ordinal + HORIZON + 1)
            if any(value not in by_ordinal for value in required_ordinals):
                continue
            origin = by_ordinal[ordinal]
            required = [by_ordinal[value] for value in required_ordinals]
            segment_ids = {str(value.segment_id) for value in required}
            sessions = {str(value.session) for value in required}
            source_gap = len(segment_ids) != 1
            crosses_session = len(sessions) != 1
            if source_gap or crosses_session or bool(origin.source_data_error_in_session):
                continue
            future_states = [
                int(by_ordinal[value].hard_state) for value in range(ordinal + 1, ordinal + 25)
            ]
            topology = classify_state_path(
                int(origin.hard_state),
                future_states,
                source_gap=source_gap,
                crosses_session=crosses_session,
            )
            record: dict[str, Any] = {
                "symbol": str(symbol),
                "session": str(session),
                "year": int(str(session)[:4]),
                "decision_ordinal": ordinal,
                "decision_ordinal_indicator": 0.0 if ordinal == 12 else 1.0,
                "slate_id": f"{session}|{ordinal:02d}",
                "decision_timestamp_utc": pd.Timestamp(origin.bar_start_timestamp),
                "feature_available_timestamp_utc": pd.Timestamp(origin.bar_complete_timestamp),
                "origin_segment_id": str(origin.segment_id),
                "history_ordinals_contiguous": True,
                "future_ordinals_contiguous": True,
                "source_gap_crossed": False,
                "session_boundary_crossed": False,
                "session_source_complete": bool(origin.session_source_complete),
                "decision_close": float(origin.close),
                "delayed_entry_open": float(by_ordinal[ordinal + 2].open),
                "future_close": float(by_ordinal[ordinal + 24].close),
                "origin_state": int(origin.hard_state),
                "future_state_path": ",".join(str(value) for value in future_states),
                "transition_count": topology.transition_count,
                "transition_burst": int(topology.transition_burst),
                "short_closure": int(topology.short_closure),
                "first_return_step": topology.first_return_step,
                "first_closure_unique_states": topology.first_closure_unique_states,
                "state_model_hash": EXPECTED_MODEL_HASH,
                "representation_status": "representation_specific_feasibility_evidence",
            }
            for feature in ORIGIN_CAUSAL_FEATURES:
                value = getattr(origin, feature)
                record[feature] = float(value)
            for window in (1, 3, 6, 12):
                record[f"simple_return_{window}"] = float(
                    getattr(origin, f"simple_return_{window}")
                )
            rows.append(record)
    compact = pd.DataFrame(rows).sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    compact = compact.reset_index(drop=True)
    compact = _add_decision_cross_section(compact)
    finite_features = np.isfinite(compact.loc[:, list(ALL_CAUSAL_FEATURES)].to_numpy(float)).all(
        axis=1
    )
    compact = compact.loc[finite_features].copy()
    slate_size = compact.groupby("slate_id", sort=True)["symbol"].transform("size")
    compact = compact.loc[slate_size.ge(15)].copy()
    compact["slate_size"] = compact.groupby("slate_id", sort=True)["symbol"].transform("size")

    compact["stock_return"] = compact["future_close"] / compact["decision_close"] - 1.0
    compact["delayed_entry_return"] = compact["future_close"] / compact["delayed_entry_open"] - 1.0
    compact["cohort_median_return_minus_i"] = np.nan
    compact["delayed_cohort_median_return_minus_i"] = np.nan
    for _, indices in compact.groupby("slate_id", sort=True).groups.items():
        index = list(indices)
        stock_return = compact.loc[index, "stock_return"].to_numpy(dtype=float)
        delayed_return = compact.loc[index, "delayed_entry_return"].to_numpy(dtype=float)
        compact.loc[index, "cohort_median_return_minus_i"] = leave_one_out_median(stock_return)
        compact.loc[index, "delayed_cohort_median_return_minus_i"] = leave_one_out_median(
            delayed_return
        )
    compact["residual_return_bps"] = 10000.0 * (
        compact["stock_return"] - compact["cohort_median_return_minus_i"]
    )
    compact["delayed_residual_return_bps"] = 10000.0 * (
        compact["delayed_entry_return"] - compact["delayed_cohort_median_return_minus_i"]
    )
    compact["absolute_movement_bps"] = compact["residual_return_bps"].abs()
    compact["log_absolute_movement"] = np.log1p(compact["absolute_movement_bps"])

    thresholds = movement_thresholds(compact)
    compact["movement_threshold_bps"] = compact["decision_ordinal"].map(thresholds)
    compact["large_move"] = (
        compact["absolute_movement_bps"] >= compact["movement_threshold_bps"]
    ).astype(np.int8)
    compact["up_given_move"] = compact["residual_return_bps"].gt(0.0).astype(np.int8)
    compact = compact.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if len(compact) > 25_000:
        raise RuntimeError("compact decision population exceeds 25,000 rows")
    if max_rows is not None:
        compact = bounded_monthly_smoke_population(compact, max_rows)
    context = {
        "combined_source_hashes": built.source_hashes,
        "combined_source_row_counts": built.source_row_counts,
        "combined_snapshot_hash": built.data_snapshot_hash,
        "combined_panel_feature_hash": built.feature_table_hash,
        "development_panel_feature_hash": development_hash,
        "panel_rows_materialised": len(panel),
        "panel_minimum_timestamp": str(panel["bar_start_timestamp"].min()),
        "panel_maximum_timestamp": str(panel["bar_start_timestamp"].max()),
        "protected_rows_materialised": int(panel["bar_start_timestamp"].ge(PROTECTED_START).sum()),
        "movement_thresholds": {str(key): value for key, value in thresholds.items()},
    }
    return compact, context


def _can_fit_binary(frame: pd.DataFrame, target: str, *, minimum_rows: int = 20) -> bool:
    return len(frame) >= minimum_rows and frame[target].nunique() == 2


def build_oof_stack(panel_2024: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = panel_2024.copy().reset_index(drop=True)
    prediction_columns = (
        "p_move_p0",
        "p_move",
        "predicted_log_absolute_movement_p0",
        "predicted_log_absolute_movement",
        "predicted_absolute_movement_bps_p0",
        "predicted_absolute_movement_bps",
        "p_transition_burst_regime_only",
        "p_transition_burst_movement_conditioned",
        "p_short_closure_regime_only",
        "p_short_closure_movement_conditioned",
        "p_up_given_move_observable",
        "p_up_given_move_with_path",
    )
    for column in prediction_columns:
        frame[column] = np.nan
        frame[f"{column}__trained_through"] = pd.NaT
    fold_manifest: list[dict[str, Any]] = []
    folds = expanding_month_folds(frame)
    for fold in folds:
        train = frame.iloc[fold.train_indices].copy()
        score = frame.iloc[fold.score_indices].copy()
        trained_through = str(train["session"].max())
        p0 = fit_fixed_logistic(
            train,
            train["large_move"],
            features=P0_FEATURES,
            slate_column="slate_id",
            model_id=f"P0_{fold.fold_id}",
        )
        p1 = fit_fixed_logistic(
            train,
            train["large_move"],
            features=MOVEMENT_FEATURES,
            slate_column="slate_id",
            model_id=f"P1_{fold.fold_id}",
        )
        r0 = fit_fixed_ridge(
            train,
            train["log_absolute_movement"],
            features=P0_FEATURES,
            slate_column="slate_id",
            model_id=f"P0_SIZE_{fold.fold_id}",
        )
        r1 = fit_fixed_ridge(
            train,
            train["log_absolute_movement"],
            features=MOVEMENT_FEATURES,
            slate_column="slate_id",
            model_id=f"P1_SIZE_{fold.fold_id}",
        )
        values = {
            "p_move_p0": p0.predict(score),
            "p_move": p1.predict(score),
            "predicted_log_absolute_movement_p0": r0.predict(score),
            "predicted_log_absolute_movement": r1.predict(score),
        }
        values["predicted_absolute_movement_bps_p0"] = np.maximum(
            0.0, np.expm1(values["predicted_log_absolute_movement_p0"])
        )
        values["predicted_absolute_movement_bps"] = np.maximum(
            0.0, np.expm1(values["predicted_log_absolute_movement"])
        )
        for column, prediction in values.items():
            frame.loc[fold.score_indices, column] = prediction
            frame.loc[fold.score_indices, f"{column}__trained_through"] = trained_through
        fold_manifest.append(
            {
                "layer": "movement",
                "fold_id": fold.fold_id,
                "score_month": fold.score_month,
                "training_months": list(fold.training_months),
                "training_rows": len(train),
                "scored_rows": len(score),
                "trained_through": trained_through,
                "score_start": str(score["session"].min()),
                "strictly_earlier": trained_through < str(score["session"].min()),
            }
        )

    for fold in folds:
        score_index = fold.score_indices
        score = frame.iloc[score_index].copy()
        prior = frame.iloc[fold.train_indices]
        train = prior.loc[prior["p_move"].notna()].copy()
        if not _can_fit_binary(train, "transition_burst"):
            continue
        train["movement_model_uncertainty"] = train["p_move"] * (1.0 - train["p_move"])
        score["movement_model_uncertainty"] = score["p_move"] * (1.0 - score["p_move"])
        b0 = fit_fixed_logistic(
            train,
            train["transition_burst"],
            features=REGIME_FEATURES,
            slate_column="slate_id",
            model_id=f"B0_{fold.fold_id}",
        )
        b1 = fit_fixed_logistic(
            train,
            train["transition_burst"],
            features=(*REGIME_FEATURES, *B1_ADDITIONS),
            slate_column="slate_id",
            model_id=f"B1_{fold.fold_id}",
        )
        trained_through = str(train["session"].max())
        for column, prediction in (
            ("p_transition_burst_regime_only", b0.predict(score)),
            ("p_transition_burst_movement_conditioned", b1.predict(score)),
        ):
            frame.loc[score_index, column] = prediction
            frame.loc[score_index, f"{column}__trained_through"] = trained_through
        fold_manifest.append(
            {
                "layer": "burst",
                "fold_id": fold.fold_id,
                "score_month": fold.score_month,
                "training_rows": len(train),
                "scored_rows": len(score),
                "trained_through": trained_through,
                "upstream_predictions_oof_only": True,
            }
        )

    for fold in folds:
        score_index = fold.score_indices
        score = frame.iloc[score_index].copy()
        prior = frame.iloc[fold.train_indices]
        train = prior.loc[
            prior["p_transition_burst_movement_conditioned"].notna()
            & prior["transition_burst"].eq(1)
        ].copy()
        if not _can_fit_binary(train, "short_closure"):
            continue
        c0 = fit_fixed_logistic(
            train,
            train["short_closure"],
            features=REGIME_FEATURES,
            slate_column="slate_id",
            model_id=f"C0_{fold.fold_id}",
        )
        c1 = fit_fixed_logistic(
            train,
            train["short_closure"],
            features=(*REGIME_FEATURES, *C1_ADDITIONS),
            slate_column="slate_id",
            model_id=f"C1_{fold.fold_id}",
        )
        conditional_c0 = c0.predict(score)
        conditional_c1 = c1.predict(score)
        trained_through = str(train["session"].max())
        values = {
            "p_short_closure_regime_only": score["p_transition_burst_regime_only"].to_numpy(float)
            * conditional_c0,
            "p_short_closure_movement_conditioned": score[
                "p_transition_burst_movement_conditioned"
            ].to_numpy(float)
            * conditional_c1,
        }
        for column, prediction in values.items():
            frame.loc[score_index, column] = prediction
            frame.loc[score_index, f"{column}__trained_through"] = trained_through
        fold_manifest.append(
            {
                "layer": "closure",
                "fold_id": fold.fold_id,
                "score_month": fold.score_month,
                "training_rows": len(train),
                "training_burst_rows_only": True,
                "scored_rows": len(score),
                "trained_through": trained_through,
                "upstream_predictions_oof_only": True,
            }
        )

    for fold in folds:
        score_index = fold.score_indices
        score = frame.iloc[score_index].copy()
        prior = frame.iloc[fold.train_indices]
        train = prior.loc[
            prior["p_short_closure_movement_conditioned"].notna() & prior["large_move"].eq(1)
        ].copy()
        if not _can_fit_binary(train, "up_given_move"):
            continue
        d0 = fit_fixed_logistic(
            train,
            train["up_given_move"],
            features=D0_FEATURES,
            slate_column="slate_id",
            model_id=f"D0_{fold.fold_id}",
        )
        d1 = fit_fixed_logistic(
            train,
            train["up_given_move"],
            features=D1_FEATURES,
            slate_column="slate_id",
            model_id=f"D1_{fold.fold_id}",
        )
        trained_through = str(train["session"].max())
        for column, prediction in (
            ("p_up_given_move_observable", d0.predict(score)),
            ("p_up_given_move_with_path", d1.predict(score)),
        ):
            frame.loc[score_index, column] = prediction
            frame.loc[score_index, f"{column}__trained_through"] = trained_through
        fold_manifest.append(
            {
                "layer": "direction",
                "fold_id": fold.fold_id,
                "score_month": fold.score_month,
                "training_rows": len(train),
                "training_large_move_rows_only": True,
                "scored_rows": len(score),
                "trained_through": trained_through,
                "upstream_predictions_oof_only": True,
            }
        )

    assert_stacking_chronology(
        frame,
        prediction_columns=(
            "p_move",
            "predicted_absolute_movement_bps",
            "p_transition_burst_movement_conditioned",
            "p_short_closure_movement_conditioned",
            "p_up_given_move_observable",
            "p_up_given_move_with_path",
        ),
    )
    frame["movement_model_uncertainty"] = frame["p_move"] * (1.0 - frame["p_move"])
    return frame, fold_manifest


def fit_final_models(
    panel_2024: pd.DataFrame,
    oof: pd.DataFrame,
    score_2025: pd.DataFrame,
    *,
    smoke_run: bool = False,
) -> tuple[pd.DataFrame, dict[str, FrozenLinearModel], dict[str, int]]:
    burst_minimum = 20 if smoke_run else 100
    closure_minimum = 20 if smoke_run else 50
    direction_minimum = 20 if smoke_run else 100
    scored = score_2025.copy().reset_index(drop=True)
    models: dict[str, FrozenLinearModel] = {}
    models["P0"] = fit_fixed_logistic(
        panel_2024,
        panel_2024["large_move"],
        features=P0_FEATURES,
        slate_column="slate_id",
        model_id="P0",
    )
    models["P1"] = fit_fixed_logistic(
        panel_2024,
        panel_2024["large_move"],
        features=MOVEMENT_FEATURES,
        slate_column="slate_id",
        model_id="P1",
    )
    models["P0_SIZE"] = fit_fixed_ridge(
        panel_2024,
        panel_2024["log_absolute_movement"],
        features=P0_FEATURES,
        slate_column="slate_id",
        model_id="P0_SIZE",
    )
    models["P1_SIZE"] = fit_fixed_ridge(
        panel_2024,
        panel_2024["log_absolute_movement"],
        features=MOVEMENT_FEATURES,
        slate_column="slate_id",
        model_id="P1_SIZE",
    )
    scored["p_move_p0"] = models["P0"].predict(scored)
    scored["p_move"] = models["P1"].predict(scored)
    scored["predicted_log_absolute_movement_p0"] = models["P0_SIZE"].predict(scored)
    scored["predicted_log_absolute_movement"] = models["P1_SIZE"].predict(scored)
    scored["predicted_absolute_movement_bps_p0"] = np.maximum(
        0.0, np.expm1(scored["predicted_log_absolute_movement_p0"])
    )
    scored["predicted_absolute_movement_bps"] = np.maximum(
        0.0, np.expm1(scored["predicted_log_absolute_movement"])
    )
    scored["movement_model_uncertainty"] = scored["p_move"] * (1.0 - scored["p_move"])

    burst_train = oof.loc[oof["p_move"].notna()].copy()
    if not _can_fit_binary(burst_train, "transition_burst", minimum_rows=burst_minimum):
        raise RuntimeError("insufficient chronology-safe burst training support")
    models["B0"] = fit_fixed_logistic(
        burst_train,
        burst_train["transition_burst"],
        features=REGIME_FEATURES,
        slate_column="slate_id",
        model_id="B0",
    )
    models["B1"] = fit_fixed_logistic(
        burst_train,
        burst_train["transition_burst"],
        features=(*REGIME_FEATURES, *B1_ADDITIONS),
        slate_column="slate_id",
        model_id="B1",
    )
    scored["p_transition_burst_regime_only"] = models["B0"].predict(scored)
    scored["p_transition_burst_movement_conditioned"] = models["B1"].predict(scored)

    closure_train = oof.loc[
        oof["p_transition_burst_movement_conditioned"].notna() & oof["transition_burst"].eq(1)
    ].copy()
    if not _can_fit_binary(closure_train, "short_closure", minimum_rows=closure_minimum):
        raise RuntimeError("insufficient chronology-safe closure training support")
    models["C0"] = fit_fixed_logistic(
        closure_train,
        closure_train["short_closure"],
        features=REGIME_FEATURES,
        slate_column="slate_id",
        model_id="C0",
    )
    models["C1"] = fit_fixed_logistic(
        closure_train,
        closure_train["short_closure"],
        features=(*REGIME_FEATURES, *C1_ADDITIONS),
        slate_column="slate_id",
        model_id="C1",
    )
    scored["p_short_closure_given_burst_regime_only"] = models["C0"].predict(scored)
    scored["p_short_closure_given_burst_movement_conditioned"] = models["C1"].predict(scored)
    scored["p_short_closure_regime_only"] = (
        scored["p_transition_burst_regime_only"] * scored["p_short_closure_given_burst_regime_only"]
    )
    scored["p_short_closure_movement_conditioned"] = (
        scored["p_transition_burst_movement_conditioned"]
        * scored["p_short_closure_given_burst_movement_conditioned"]
    )

    direction_train = oof.loc[
        oof["p_short_closure_movement_conditioned"].notna() & oof["large_move"].eq(1)
    ].copy()
    if not _can_fit_binary(direction_train, "up_given_move", minimum_rows=direction_minimum):
        raise RuntimeError("insufficient chronology-safe direction training support")
    models["D0"] = fit_fixed_logistic(
        direction_train,
        direction_train["up_given_move"],
        features=D0_FEATURES,
        slate_column="slate_id",
        model_id="D0",
    )
    models["D1"] = fit_fixed_logistic(
        direction_train,
        direction_train["up_given_move"],
        features=D1_FEATURES,
        slate_column="slate_id",
        model_id="D1",
    )
    scored["p_up_given_move_observable"] = models["D0"].predict(scored)
    scored["p_up_given_move_with_path"] = models["D1"].predict(scored)
    observable = probability_chain(
        scored["p_move"],
        scored["p_up_given_move_observable"],
        scored["predicted_absolute_movement_bps"],
    )
    path = probability_chain(
        scored["p_move"],
        scored["p_up_given_move_with_path"],
        scored["predicted_absolute_movement_bps"],
    )
    scored["p_long_observable"] = observable["p_long"]
    scored["p_short_observable"] = observable["p_short"]
    scored["p_long_with_path"] = path["p_long"]
    scored["p_short_with_path"] = path["p_short"]
    scored["p_neutral"] = observable["p_neutral"]
    scored["observable_chain_score"] = observable["score"]
    scored["path_chain_score"] = path["score"]
    training_support = {
        "movement_rows": len(panel_2024),
        "burst_rows": len(burst_train),
        "closure_burst_rows": len(closure_train),
        "direction_large_move_rows": len(direction_train),
    }
    return scored, models, training_support


def _safe_auc(target: pd.Series, probability: pd.Series) -> float:
    return float(roc_auc_score(target, probability)) if target.nunique() == 2 else math.nan


def _binary_metrics(target: pd.Series, probability: pd.Series) -> dict[str, float]:
    clipped = probability.clip(1e-12, 1.0 - 1e-12)
    return {
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
        "auc": _safe_auc(target, probability),
    }


def _calibration_rows(
    frame: pd.DataFrame,
    *,
    layer: str,
    model: str,
    target: str,
    probability: str,
) -> list[dict[str, Any]]:
    prepared = frame[[target, probability]].dropna().copy()
    prepared["bin"] = pd.cut(
        prepared[probability],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        labels=False,
    )
    rows: list[dict[str, Any]] = []
    for bin_number in range(10):
        selected = prepared.loc[prepared["bin"].eq(bin_number)]
        rows.append(
            {
                "layer": layer,
                "model": model,
                "bin": bin_number + 1,
                "lower": bin_number / 10.0,
                "upper": (bin_number + 1) / 10.0,
                "rows": len(selected),
                "mean_prediction": float(selected[probability].mean())
                if len(selected)
                else math.nan,
                "observed_rate": float(selected[target].mean()) if len(selected) else math.nan,
            }
        )
    return rows


def _quintile_rows(
    frame: pd.DataFrame,
    *,
    layer: str,
    model: str,
    probability: str,
    realised: str,
) -> list[dict[str, Any]]:
    prepared = frame[[probability, realised]].dropna().copy()
    prepared["quintile"] = pd.qcut(prepared[probability].rank(method="first"), 5, labels=False)
    rows: list[dict[str, Any]] = []
    for quintile, selected in prepared.groupby("quintile", sort=True):
        rows.append(
            {
                "scope": "quintile",
                "layer": layer,
                "model": model,
                "quintile": int(quintile) + 1,
                "rows": len(selected),
                "mean_realised": float(selected[realised].mean()),
                "mean_probability": float(selected[probability].mean()),
            }
        )
    return rows


def evaluate_binary_layer(
    frame: pd.DataFrame,
    *,
    layer: str,
    target: str,
    baseline_name: str,
    baseline_probability: str,
    candidate_name: str,
    candidate_probability: str,
    realised: str,
    balanced: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    selected = frame.dropna(subset=[target, baseline_probability, candidate_probability]).copy()
    metric_rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float]] = {}
    for name, probability in (
        (baseline_name, baseline_probability),
        (candidate_name, candidate_probability),
    ):
        metrics = _binary_metrics(selected[target], selected[probability])
        summary[name] = metrics
        row: dict[str, Any] = {
            "scope": "overall",
            "layer": layer,
            "model": name,
            "rows": len(selected),
            **metrics,
        }
        if balanced:
            row["balanced_accuracy_at_0_5_descriptive"] = float(
                balanced_accuracy_score(selected[target], selected[probability].ge(0.5))
            )
        if layer == "movement":
            top_count = max(1, int(math.ceil(0.20 * len(selected))))
            top = selected.nlargest(top_count, probability)
            row["top_quintile_realised_absolute_movement_bps"] = float(top[realised].mean())
        metric_rows.append(row)
        metric_rows.extend(
            _quintile_rows(
                selected,
                layer=layer,
                model=name,
                probability=probability,
                realised=realised,
            )
        )
    calibration = pd.DataFrame(
        [
            *_calibration_rows(
                selected,
                layer=layer,
                model=baseline_name,
                target=target,
                probability=baseline_probability,
            ),
            *_calibration_rows(
                selected,
                layer=layer,
                model=candidate_name,
                target=target,
                probability=candidate_probability,
            ),
        ]
    )
    monthly_rows: list[dict[str, Any]] = []
    selected["month"] = pd.to_datetime(selected["session"]).dt.strftime("%Y-%m")
    for month, month_frame in selected.groupby("month", sort=True):
        base = _binary_metrics(month_frame[target], month_frame[baseline_probability])
        candidate = _binary_metrics(month_frame[target], month_frame[candidate_probability])
        monthly_rows.append(
            {
                "layer": layer,
                "month": month,
                "rows": len(month_frame),
                "baseline_model": baseline_name,
                "candidate_model": candidate_name,
                "baseline_brier": base["brier"],
                "candidate_brier": candidate["brier"],
                "brier_improvement": base["brier"] - candidate["brier"],
                "baseline_log_loss": base["log_loss"],
                "candidate_log_loss": candidate["log_loss"],
                "log_loss_improvement": base["log_loss"] - candidate["log_loss"],
                "baseline_auc": base["auc"],
                "candidate_auc": candidate["auc"],
            }
        )
    increment = {
        "brier_improvement": summary[baseline_name]["brier"] - summary[candidate_name]["brier"],
        "log_loss_improvement": summary[baseline_name]["log_loss"]
        - summary[candidate_name]["log_loss"],
        "baseline_brier": summary[baseline_name]["brier"],
        "candidate_brier": summary[candidate_name]["brier"],
        "baseline_log_loss": summary[baseline_name]["log_loss"],
        "candidate_log_loss": summary[candidate_name]["log_loss"],
        "baseline_auc": summary[baseline_name]["auc"],
        "candidate_auc": summary[candidate_name]["auc"],
    }
    return (
        pd.DataFrame(metric_rows),
        calibration,
        pd.DataFrame(monthly_rows),
        increment,
    )


def chain_ranking(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for slate_id, slate in frame.groupby("slate_id", sort=True):
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
            top_two = ranked.iloc[:2]
            correlation = spearmanr(
                ordered[score_column].to_numpy(float),
                ordered["residual_return_bps"].to_numpy(float),
            ).statistic
            rows.append(
                {
                    "scope": "slate",
                    "slate_id": slate_id,
                    "session": str(ordered["session"].iloc[0]),
                    "month": str(ordered["session"].iloc[0])[:7],
                    "decision_ordinal": int(ordered["decision_ordinal"].iloc[0]),
                    "model": model,
                    "slate_size": len(ordered),
                    "spearman": float(correlation),
                    "top_one_symbol": str(top_one["symbol"]),
                    "top_one_realised_residual_return_bps": float(top_one["residual_return_bps"]),
                    "top_one_minus_slate_median_bps": float(
                        top_one["residual_return_bps"] - median
                    ),
                    "top_two_average_minus_slate_median_bps": float(
                        top_two["residual_return_bps"].mean() - median
                    ),
                    "top_one_delayed_residual_return_bps": float(
                        top_one["delayed_residual_return_bps"]
                    ),
                }
            )
    result = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    overall_rows: list[dict[str, Any]] = []
    for model, selected in result.groupby("model", sort=True):
        values = {
            "spearman": float(selected["spearman"].mean()),
            "top_one_realised_residual_return_bps": float(
                selected["top_one_realised_residual_return_bps"].mean()
            ),
            "top_one_minus_slate_median_bps": float(
                selected["top_one_minus_slate_median_bps"].mean()
            ),
            "top_two_average_minus_slate_median_bps": float(
                selected["top_two_average_minus_slate_median_bps"].mean()
            ),
        }
        overall_rows.append(
            {
                "scope": "overall",
                "slate_id": "ALL",
                "session": "ALL",
                "month": "ALL",
                "decision_ordinal": -1,
                "model": model,
                "slate_size": int(selected["slate_size"].sum()),
                **values,
            }
        )
        summary.update({f"{model}_{key}": value for key, value in values.items()})
    output = pd.concat([result, pd.DataFrame(overall_rows)], ignore_index=True, sort=False)
    path = result.loc[result["model"].eq("path_chain")].set_index("slate_id")
    observable = result.loc[result["model"].eq("observable_chain")].set_index("slate_id")
    summary["spearman_difference"] = float(path["spearman"].mean() - observable["spearman"].mean())
    summary["top_one_difference_bps"] = float(
        path["top_one_minus_slate_median_bps"].mean()
        - observable["top_one_minus_slate_median_bps"].mean()
    )
    return output, summary


def concentration_metrics(
    frame: pd.DataFrame, ranking: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol, count in frame["symbol"].value_counts(sort=False).sort_index().items():
        rows.append(
            {
                "surface": "decision_rows",
                "model": "population",
                "group_type": "stock",
                "group": symbol,
                "count": int(count),
                "fraction": float(count / len(frame)),
            }
        )
    slate_rows = ranking.loc[ranking["scope"].eq("slate")]
    for model, model_frame in slate_rows.groupby("model", sort=True):
        for symbol, count in (
            model_frame["top_one_symbol"].value_counts(sort=False).sort_index().items()
        ):
            rows.append(
                {
                    "surface": "top_one",
                    "model": model,
                    "group_type": "stock",
                    "group": symbol,
                    "count": int(count),
                    "fraction": float(count / len(model_frame)),
                }
            )
        for month, count in model_frame["month"].value_counts(sort=False).sort_index().items():
            rows.append(
                {
                    "surface": "top_one",
                    "model": model,
                    "group_type": "month",
                    "group": month,
                    "count": int(count),
                    "fraction": float(count / len(model_frame)),
                }
            )
    metrics = pd.DataFrame(rows)
    max_row_fraction = float(metrics.loc[metrics["surface"].eq("decision_rows"), "fraction"].max())
    max_top_one_fraction = float(
        metrics.loc[
            metrics["surface"].eq("top_one") & metrics["group_type"].eq("stock"),
            "fraction",
        ].max()
    )
    summary = {
        "maximum_stock_decision_row_fraction": max_row_fraction,
        "maximum_stock_top_one_fraction": max_top_one_fraction,
        "decision_row_gate": max_row_fraction <= 0.125,
        "top_one_gate": max_top_one_fraction <= 0.20,
        "concentration_passed": max_row_fraction <= 0.125 and max_top_one_fraction <= 0.20,
    }
    return metrics, summary


def delayed_sensitivity(ranking: pd.DataFrame) -> pd.DataFrame:
    slate_rows = ranking.loc[ranking["scope"].eq("slate")]
    rows: list[dict[str, Any]] = []
    for model, model_frame in slate_rows.groupby("model", sort=True):
        gross = model_frame["top_one_delayed_residual_return_bps"]
        for friction in (0.0, 5.0, 10.0, 20.0):
            rows.append(
                {
                    "model": model,
                    "synthetic_round_trip_friction_bps": friction,
                    "slates": len(model_frame),
                    "mean_top_one_delayed_residual_bps_gross": float(gross.mean()),
                    "mean_top_one_delayed_residual_bps_after_synthetic_friction": float(
                        (gross - friction).mean()
                    ),
                    "interpretation": "secondary_economic_reference_not_fill_model",
                }
            )
    return pd.DataFrame(rows)


def bootstrap_metrics(
    scored: pd.DataFrame,
    ranking: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sessions = sorted(scored["session"].astype(str).unique())
    samples = sampled_sessions(sessions, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    ranking_slates = ranking.loc[ranking["scope"].eq("slate")].copy()
    ranking_session = (
        ranking_slates.groupby(["session", "model"], sort=True)
        .agg(
            spearman=("spearman", "mean"),
            top_one=("top_one_minus_slate_median_bps", "mean"),
        )
        .reset_index()
    )
    metric_values: dict[str, list[float]] = {
        "p1_minus_p0_brier_improvement": [],
        "b1_minus_b0_brier_improvement": [],
        "c1_minus_c0_brier_improvement": [],
        "d1_minus_d0_brier_improvement": [],
        "path_minus_observable_spearman": [],
        "path_minus_observable_top_one": [],
    }
    for sample in samples:
        counts = Counter(sample)
        weights = scored["session"].astype(str).map(counts).to_numpy(dtype=float)

        def paired_brier(
            target: str,
            baseline: str,
            candidate: str,
            mask: pd.Series | None = None,
            *,
            draw_weights: np.ndarray = weights,
        ) -> float:
            local_mask = np.ones(len(scored), dtype=bool) if mask is None else mask.to_numpy(bool)
            local_mask &= draw_weights > 0
            local_weights = draw_weights[local_mask]
            truth = scored.loc[local_mask, target].to_numpy(float)
            base = scored.loc[local_mask, baseline].to_numpy(float)
            candidate_values = scored.loc[local_mask, candidate].to_numpy(float)
            return float(
                np.average((truth - base) ** 2, weights=local_weights)
                - np.average((truth - candidate_values) ** 2, weights=local_weights)
            )

        metric_values["p1_minus_p0_brier_improvement"].append(
            paired_brier("large_move", "p_move_p0", "p_move")
        )
        metric_values["b1_minus_b0_brier_improvement"].append(
            paired_brier(
                "transition_burst",
                "p_transition_burst_regime_only",
                "p_transition_burst_movement_conditioned",
            )
        )
        metric_values["c1_minus_c0_brier_improvement"].append(
            paired_brier(
                "short_closure",
                "p_short_closure_given_burst_regime_only",
                "p_short_closure_given_burst_movement_conditioned",
                scored["transition_burst"].eq(1),
            )
        )
        metric_values["d1_minus_d0_brier_improvement"].append(
            paired_brier(
                "up_given_move",
                "p_up_given_move_observable",
                "p_up_given_move_with_path",
                scored["large_move"].eq(1),
            )
        )
        selected = ranking_session.loc[ranking_session["session"].isin(counts)].copy()
        selected["bootstrap_weight"] = selected["session"].map(counts).astype(float)
        aggregates = {}
        for model, model_frame in selected.groupby("model", sort=True):
            aggregates[model] = {
                "spearman": float(
                    np.average(model_frame["spearman"], weights=model_frame["bootstrap_weight"])
                ),
                "top_one": float(
                    np.average(model_frame["top_one"], weights=model_frame["bootstrap_weight"])
                ),
            }
        metric_values["path_minus_observable_spearman"].append(
            aggregates["path_chain"]["spearman"] - aggregates["observable_chain"]["spearman"]
        )
        metric_values["path_minus_observable_top_one"].append(
            aggregates["path_chain"]["top_one"] - aggregates["observable_chain"]["top_one"]
        )
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for metric, raw_values in metric_values.items():
        values = np.asarray(raw_values, dtype=float)
        intervals = {
            "ci90_lower": float(np.quantile(values, 0.05)),
            "ci90_upper": float(np.quantile(values, 0.95)),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
        }
        summary[metric] = intervals
        rows.append(
            {
                "metric": metric,
                "draw": -1,
                "value": float(values.mean()),
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                **intervals,
            }
        )
        for draw, value in enumerate(values):
            rows.append(
                {
                    "metric": metric,
                    "draw": draw,
                    "value": float(value),
                    "draws": BOOTSTRAP_DRAWS,
                    "seed": BOOTSTRAP_SEED,
                    "ci90_lower": math.nan,
                    "ci90_upper": math.nan,
                    "ci95_lower": math.nan,
                    "ci95_upper": math.nan,
                }
            )
    return pd.DataFrame(rows), summary


def fast_nulls(
    oof: pd.DataFrame,
    scored: pd.DataFrame,
    models: Mapping[str, FrozenLinearModel],
    *,
    real_burst_improvement: float,
    real_direction_improvement: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    burst_train = oof.loc[oof["p_move"].notna()].copy()
    direction_train = oof.loc[
        oof["p_short_closure_movement_conditioned"].notna() & oof["large_move"].eq(1)
    ].copy()
    scored_large = scored["large_move"].eq(1)
    rows: list[dict[str, Any]] = []
    burst_nulls: list[float] = []
    direction_nulls: list[float] = []
    for draw in range(NULL_DRAWS):
        shifted_burst_train, train_manifest = circular_shift_session_blocks(
            burst_train,
            value_columns=B1_ADDITIONS,
            draw=draw,
            seed=NULL_SEED,
        )
        shifted_burst_score, score_manifest = circular_shift_session_blocks(
            scored,
            value_columns=B1_ADDITIONS,
            draw=draw,
            seed=NULL_SEED + 10_000,
        )
        null_b1 = fit_fixed_logistic(
            shifted_burst_train,
            shifted_burst_train["transition_burst"],
            features=(*REGIME_FEATURES, *B1_ADDITIONS),
            slate_column="slate_id",
            model_id=f"NULL_B1_{draw:03d}",
            random_state=20260720 + draw,
        )
        null_burst_probability = null_b1.predict(shifted_burst_score)
        truth_burst = scored["transition_burst"].to_numpy(float)
        b0_probability = scored["p_transition_burst_regime_only"].to_numpy(float)
        burst_improvement = float(
            np.mean((truth_burst - b0_probability) ** 2)
            - np.mean((truth_burst - null_burst_probability) ** 2)
        )
        burst_nulls.append(burst_improvement)
        train_hash = hashlib.sha256(canonical_json(train_manifest).encode("utf-8")).hexdigest()
        score_hash = hashlib.sha256(canonical_json(score_manifest).encode("utf-8")).hexdigest()
        rows.append(
            {
                "comparison": "B1_minus_B0_brier_improvement",
                "draw": draw,
                "null_value": burst_improvement,
                "real_value": real_burst_improvement,
                "seed": NULL_SEED,
                "train_shift_manifest_hash": train_hash,
                "score_shift_manifest_hash": score_hash,
            }
        )

        structural_columns = (
            "p_transition_burst_movement_conditioned",
            "p_short_closure_movement_conditioned",
        )
        shifted_direction_train, train_direction_manifest = circular_shift_session_blocks(
            direction_train,
            value_columns=structural_columns,
            draw=draw,
            seed=NULL_SEED + 20_000,
        )
        shifted_direction_score, score_direction_manifest = circular_shift_session_blocks(
            scored,
            value_columns=structural_columns,
            draw=draw,
            seed=NULL_SEED + 30_000,
        )
        null_d1 = fit_fixed_logistic(
            shifted_direction_train,
            shifted_direction_train["up_given_move"],
            features=D1_FEATURES,
            slate_column="slate_id",
            model_id=f"NULL_D1_{draw:03d}",
            random_state=20260820 + draw,
        )
        null_direction_probability = null_d1.predict(shifted_direction_score)
        truth_direction = scored.loc[scored_large, "up_given_move"].to_numpy(float)
        d0_probability = scored.loc[scored_large, "p_up_given_move_observable"].to_numpy(float)
        direction_improvement = float(
            np.mean((truth_direction - d0_probability) ** 2)
            - np.mean((truth_direction - null_direction_probability[scored_large]) ** 2)
        )
        direction_nulls.append(direction_improvement)
        rows.append(
            {
                "comparison": "D1_minus_D0_brier_improvement",
                "draw": draw,
                "null_value": direction_improvement,
                "real_value": real_direction_improvement,
                "seed": NULL_SEED,
                "train_shift_manifest_hash": hashlib.sha256(
                    canonical_json(train_direction_manifest).encode("utf-8")
                ).hexdigest(),
                "score_shift_manifest_hash": hashlib.sha256(
                    canonical_json(score_direction_manifest).encode("utf-8")
                ).hexdigest(),
            }
        )
    burst_values = np.asarray(burst_nulls)
    direction_values = np.asarray(direction_nulls)
    summary = {
        "B1": {
            "draws": NULL_DRAWS,
            "null_q90": float(np.quantile(burst_values, 0.90)),
            "real_improvement": real_burst_improvement,
            "empirical_percentile": float(np.mean(real_burst_improvement > burst_values)),
        },
        "D1": {
            "draws": NULL_DRAWS,
            "null_q90": float(np.quantile(direction_values, 0.90)),
            "real_improvement": real_direction_improvement,
            "empirical_percentile": float(np.mean(real_direction_improvement > direction_values)),
        },
    }
    return pd.DataFrame(rows), summary


def support_summary(scored: pd.DataFrame) -> dict[str, Any]:
    support = {
        "scored_2025_decision_rows": len(scored),
        "scored_2025_sessions": int(scored["session"].nunique()),
        "stocks": int(scored["symbol"].nunique()),
        "actual_large_move_rows": int(scored["large_move"].sum()),
        "transition_burst_rows": int(scored["transition_burst"].sum()),
        "short_closure_rows": int(scored["short_closure"].sum()),
    }
    support["closure_status"] = (
        "secondary_sufficient_support"
        if support["short_closure_rows"] >= 100
        else "secondary_insufficient_support"
    )
    support["primary_support_passed"] = bool(
        support["scored_2025_decision_rows"] >= 3000
        and support["scored_2025_sessions"] >= 100
        and support["stocks"] >= 15
        and support["actual_large_move_rows"] >= 500
        and support["transition_burst_rows"] >= 300
    )
    return support


def feature_manifest() -> dict[str, Any]:
    for surface in (
        P0_FEATURES,
        MOVEMENT_FEATURES,
        DIRECTION_FEATURES,
        REGIME_FEATURES,
        (*REGIME_FEATURES, *B1_ADDITIONS),
        (*REGIME_FEATURES, *C1_ADDITIONS),
        D0_FEATURES,
        D1_FEATURES,
    ):
        assert_allowed_feature_names(surface)
    return {
        **SAFETY_FLAGS,
        "outcome_free_causal_feature_ledger": True,
        "provider_volume_interpretation": "historical_activity_proxy",
        "movement_feature_count": len(MOVEMENT_FEATURES),
        "P0": list(P0_FEATURES),
        "P1": list(MOVEMENT_FEATURES),
        "B0": list(REGIME_FEATURES),
        "B1": [*REGIME_FEATURES, *B1_ADDITIONS],
        "C0": list(REGIME_FEATURES),
        "C1": [*REGIME_FEATURES, *C1_ADDITIONS],
        "D0": list(D0_FEATURES),
        "D1": list(D1_FEATURES),
        "forbidden_exact": [
            "exact_loop_id",
            "selected_loop_membership",
            "profitable_loop_label",
            "payoff_history",
            "future_state",
            "future_path",
            "future_movement",
            "symbol_identity",
            "month_identity",
            "previously_selected_model_score",
            "excursion_resolution_label",
        ],
        "outcome_columns_not_in_feature_matrices": [
            "future_state_path",
            "transition_burst",
            "short_closure",
            "residual_return_bps",
            "large_move",
            "up_given_move",
        ],
    }


def _decision_evidence(
    movement: Mapping[str, float],
    burst: Mapping[str, float],
    direction: Mapping[str, float],
    monthly: pd.DataFrame,
    bootstrap: Mapping[str, Any],
    nulls: Mapping[str, Any],
    chain: Mapping[str, float],
    concentration: Mapping[str, Any],
    support: Mapping[str, Any],
    *,
    scientific_run: bool,
    exact_rerun_passed: bool,
    independent_audit_passed: bool,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "p1_minus_p0_brier_improvement": movement["brier_improvement"],
        "p1_minus_p0_log_loss_improvement": movement["log_loss_improvement"],
        "b1_minus_b0_brier_improvement": burst["brier_improvement"],
        "b1_minus_b0_log_loss_improvement": burst["log_loss_improvement"],
        "d1_minus_d0_brier_improvement": direction["brier_improvement"],
        "d1_minus_d0_log_loss_improvement": direction["log_loss_improvement"],
        "b1_positive_months": int(
            monthly.loc[monthly["layer"].eq("burst"), "brier_improvement"].gt(0.0).sum()
        ),
        "d1_positive_months": int(
            monthly.loc[monthly["layer"].eq("direction"), "brier_improvement"].gt(0.0).sum()
        ),
        "b1_bootstrap_90_lower": bootstrap["b1_minus_b0_brier_improvement"]["ci90_lower"],
        "d1_bootstrap_90_lower": bootstrap["d1_minus_d0_brier_improvement"]["ci90_lower"],
        "b1_null_percentile": nulls["B1"]["empirical_percentile"],
        "d1_null_percentile": nulls["D1"]["empirical_percentile"],
        "observable_spearman": chain["observable_chain_spearman"],
        "path_spearman": chain["path_chain_spearman"],
        "observable_top_one_minus_median": chain["observable_chain_top_one_minus_slate_median_bps"],
        "path_top_one_minus_median": chain["path_chain_top_one_minus_slate_median_bps"],
        "concentration_passed": bool(concentration["concentration_passed"]),
        "exact_rerun_passed": exact_rerun_passed,
        "independent_audit_passed": independent_audit_passed,
        "scientific_run": scientific_run,
    }
    if not scientific_run or not bool(support["primary_support_passed"]):
        evidence["blocker"] = "blocked_insufficient_probability_chain_support"
    return evidence


def _report(
    decision: Mapping[str, Any],
    support: Mapping[str, Any],
    thresholds: Mapping[str, float],
    increments: Mapping[str, Mapping[str, float]],
    chain: Mapping[str, float],
    nulls: Mapping[str, Any],
    concentration: Mapping[str, Any],
    delayed: pd.DataFrame,
) -> str:
    lines = [
        "# Movement-Conditioned Regime-Path Probability Chain V0",
        "",
        "Retrospective research-only feasibility screen. Scientific status: "
        "`representation_specific_feasibility_evidence`. This is not prospective "
        "validation, a strategy, achieved P&L, or executable net-edge evidence.",
        "",
        "## Safety",
        "",
        "`research_only=true`, `feasibility_screen=true`, `execution_enabled=false`, "
        "`order_placement=disabled`, `broker_integration_required=false`, "
        "`strategy_promotion=false`, `production_runtime_modified=false`.",
        "",
        "## Population and support",
        "",
        f"- Rows: {support['scored_2025_decision_rows']}",
        f"- Sessions: {support['scored_2025_sessions']}",
        f"- Stocks: {support['stocks']}",
        f"- Actual large moves: {support['actual_large_move_rows']}",
        f"- Transition bursts: {support['transition_burst_rows']}",
        f"- Short closures: {support['short_closure_rows']} ({support['closure_status']})",
        f"- Clock-12 q75: {float(thresholds['12']):.6f} bps",
        f"- Clock-36 q75: {float(thresholds['36']):.6f} bps",
        "",
        "## Layer comparisons",
        "",
    ]
    for label, key in (
        ("P1 versus P0", "movement"),
        ("B1 versus B0", "burst"),
        ("C1 versus C0 (secondary)", "closure"),
        ("D1 versus D0", "direction"),
    ):
        value = increments[key]
        lines.append(
            f"- {label}: Brier improvement {value['brier_improvement']:.8f}; "
            f"log-loss improvement {value['log_loss_improvement']:.8f}; "
            f"AUC {value['baseline_auc']:.6f} -> {value['candidate_auc']:.6f}."
        )
    lines.extend(
        [
            "",
            "## Chain-ranking diagnostic",
            "",
            f"- Mean Spearman: observable {chain['observable_chain_spearman']:.6f}; "
            f"path {chain['path_chain_spearman']:.6f}.",
            f"- Mean top-one minus slate median: observable "
            f"{chain['observable_chain_top_one_minus_slate_median_bps']:.6f} bps; path "
            f"{chain['path_chain_top_one_minus_slate_median_bps']:.6f} bps.",
            "- These are ranking diagnostics, not expected or achieved P&L.",
            "",
            "## Nulls, concentration, and delayed reference",
            "",
            f"- B1 real improvement percentile under 100 session shifts: "
            f"{nulls['B1']['empirical_percentile']:.3f}.",
            f"- D1 real improvement percentile under 100 session shifts: "
            f"{nulls['D1']['empirical_percentile']:.3f}.",
            f"- Maximum stock decision-row fraction: "
            f"{concentration['maximum_stock_decision_row_fraction']:.4f}.",
            f"- Maximum stock top-one fraction: "
            f"{concentration['maximum_stock_top_one_fraction']:.4f}.",
        ]
    )
    for row in delayed.itertuples():
        if float(row.synthetic_round_trip_friction_bps) in (0.0, 20.0):
            lines.append(
                f"- {row.model} delayed top-one reference at "
                f"{row.synthetic_round_trip_friction_bps:.0f} bps synthetic friction: "
                f"{row.mean_top_one_delayed_residual_bps_after_synthetic_friction:.6f} bps."
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{decision['decision']}`",
            "",
            "Movement predictability, structural-path predictability, directional "
            "predictability, gross economic association, and executable net edge remain "
            "separate conclusions. This V0 cannot establish the last conclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def run_screen(
    output_dir: Path,
    *,
    provider_root: Path,
    max_rows: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scientific_run = max_rows is None
    _load_contract()
    missing = [path for path in _required_inputs() if not path.is_file()]
    if missing:
        blocker = {
            **SAFETY_FLAGS,
            "decision": "blocked_missing_required_frozen_artifacts",
            "missing": [path.relative_to(REPO_ROOT).as_posix() for path in missing],
        }
        write_json(output_dir / "decision.json", blocker)
        return blocker
    preprocessing, parameters = _load_frozen_model()
    posterior_reproduction = reproduce_frozen_posterior(preprocessing, parameters)
    development_sources = _verify_development_sources(provider_root)
    month_counts, safe_source_hashes = _safe_source_month_counts(provider_root)
    compact, panel_context = build_compact_panel(
        provider_root,
        preprocessing,
        parameters,
        max_rows=max_rows,
    )
    if compact.empty:
        raise RuntimeError("compact decision panel is empty")
    if compact["decision_timestamp_utc"].ge(PROTECTED_START).any():
        raise RuntimeError("protected decision row entered compact panel")
    panel_2024 = compact.loc[compact["year"].eq(2024)].copy().reset_index(drop=True)
    score_2025 = compact.loc[compact["year"].eq(2025)].copy().reset_index(drop=True)
    oof, folds = build_oof_stack(panel_2024)
    scored, models, training_support = fit_final_models(
        panel_2024,
        oof,
        score_2025,
        smoke_run=not scientific_run,
    )
    support = support_summary(scored)

    movement_metrics, movement_calibration, movement_monthly, movement_increment = (
        evaluate_binary_layer(
            scored,
            layer="movement",
            target="large_move",
            baseline_name="P0",
            baseline_probability="p_move_p0",
            candidate_name="P1",
            candidate_probability="p_move",
            realised="absolute_movement_bps",
        )
    )
    burst_metrics, burst_calibration, burst_monthly, burst_increment = evaluate_binary_layer(
        scored,
        layer="burst",
        target="transition_burst",
        baseline_name="B0",
        baseline_probability="p_transition_burst_regime_only",
        candidate_name="B1",
        candidate_probability="p_transition_burst_movement_conditioned",
        realised="transition_burst",
    )
    burst_rows = scored.loc[scored["transition_burst"].eq(1)].copy()
    closure_metrics, closure_calibration, closure_monthly, closure_increment = (
        evaluate_binary_layer(
            burst_rows,
            layer="closure",
            target="short_closure",
            baseline_name="C0",
            baseline_probability="p_short_closure_given_burst_regime_only",
            candidate_name="C1",
            candidate_probability="p_short_closure_given_burst_movement_conditioned",
            realised="short_closure",
        )
    )
    large_rows = scored.loc[scored["large_move"].eq(1)].copy()
    direction_metrics, direction_calibration, direction_monthly, direction_increment = (
        evaluate_binary_layer(
            large_rows,
            layer="direction",
            target="up_given_move",
            baseline_name="D0",
            baseline_probability="p_up_given_move_observable",
            candidate_name="D1",
            candidate_probability="p_up_given_move_with_path",
            realised="residual_return_bps",
            balanced=True,
        )
    )
    monthly = pd.concat(
        [movement_monthly, burst_monthly, closure_monthly, direction_monthly],
        ignore_index=True,
    )
    calibration = pd.concat(
        [movement_calibration, burst_calibration, closure_calibration, direction_calibration],
        ignore_index=True,
    )
    ranking, chain_summary = chain_ranking(scored)
    concentration, concentration_summary = concentration_metrics(scored, ranking)
    delayed = delayed_sensitivity(ranking)
    bootstraps, bootstrap_summary = bootstrap_metrics(scored, ranking)
    null_metrics, null_summary = fast_nulls(
        oof,
        scored,
        models,
        real_burst_improvement=burst_increment["brier_improvement"],
        real_direction_improvement=direction_increment["brier_improvement"],
    )
    evidence = _decision_evidence(
        movement_increment,
        burst_increment,
        direction_increment,
        monthly,
        bootstrap_summary,
        null_summary,
        chain_summary,
        concentration_summary,
        support,
        scientific_run=scientific_run,
        exact_rerun_passed=False,
        independent_audit_passed=False,
    )
    decision_label = decide_screen(evidence)
    decision = {
        **SAFETY_FLAGS,
        "decision": decision_label,
        "decision_status": "provisional_pending_exact_rerun_and_audit",
        "scientific_run": scientific_run,
        "scientific_status": "representation_specific_feasibility_evidence",
        "not_cluster_invariant": True,
        "not_strategy_or_edge_evidence": True,
        "support": support,
        "training_support": training_support,
        "evidence": evidence,
        "exact_rerun_passed": False,
        "independent_audit_passed": False,
    }

    input_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in _required_inputs()
    }
    source_manifest = {
        **SAFETY_FLAGS,
        "decision_cohort": list(DECISION_SYMBOLS),
        "regime_context_cohort": list(REGIME_CONTEXT_SYMBOLS),
        "benchmark": "VTI",
        "provider_sources": {
            symbol: {
                "logical_path": logical_source_path(symbol),
                "bounded_2024_hash": development_sources["hashes"][symbol],
                "bounded_safe_hash": safe_source_hashes[symbol],
                "bounded_safe_row_count": panel_context["combined_source_row_counts"][symbol],
            }
            for symbol in (*REGIME_CONTEXT_SYMBOLS, "VTI")
        },
        "frozen_model_hash": EXPECTED_MODEL_HASH,
        "frozen_development_snapshot_hash": development_sources["snapshot_hash"],
        "frozen_development_panel_hash": panel_context["development_panel_feature_hash"],
        "posterior_reproduction": posterior_reproduction,
        "provider_volume_interpretation": "historical_activity_proxy",
    }
    boundary_audit = {
        **SAFETY_FLAGS,
        "read_predicate": {
            "timestamp_gte": str(START),
            "timestamp_lt": str(PROTECTED_START),
        },
        "minimum_timestamp_read": str(month_counts["minimum_timestamp"].iloc[0]),
        "maximum_timestamp_read": str(month_counts["maximum_timestamp"].iloc[0]),
        "row_count_by_year_month": {
            str(row.year_month): int(row.row_count) for row in month_counts.itertuples()
        },
        "source_hashes": safe_source_hashes,
        "protected_rows_opened": 0,
        "protected_rows_materialised": panel_context["protected_rows_materialised"],
        "compact_panel_protected_rows": int(
            compact["decision_timestamp_utc"].ge(PROTECTED_START).sum()
        ),
        "proven": True,
    }
    model_coefficients = {
        **SAFETY_FLAGS,
        "models": {name: model.as_dict() for name, model in sorted(models.items())},
    }
    thresholds = panel_context["movement_thresholds"]
    increments = {
        "movement": movement_increment,
        "burst": burst_increment,
        "closure": closure_increment,
        "direction": direction_increment,
    }
    report = _report(
        decision,
        support,
        thresholds,
        increments,
        chain_summary,
        null_summary,
        concentration_summary,
        delayed,
    )

    write_json(output_dir / "source_manifest.json", source_manifest)
    write_json(output_dir / "input_artifact_hashes.json", input_hashes)
    write_json(output_dir / "protected_boundary_audit.json", boundary_audit)
    write_json(output_dir / "feature_manifest.json", feature_manifest())
    write_json(
        output_dir / "movement_thresholds.json",
        {**SAFETY_FLAGS, "fit_year": 2024, "quantile": 0.75, "thresholds_bps": thresholds},
    )
    write_json(
        output_dir / "chronological_fold_manifest.json",
        {
            **SAFETY_FLAGS,
            "folds": folds,
            "all_upstream_predictions_out_of_fold": True,
            "in_sample_stacked_features": 0,
        },
    )
    write_json(output_dir / "model_coefficients.json", model_coefficients)
    write_parquet(output_dir / "compact_decision_panel.parquet", compact)
    write_parquet(output_dir / "oof_2024_predictions.parquet", oof)
    write_parquet(output_dir / "scored_2025_predictions.parquet", scored)
    write_csv(output_dir / "movement_metrics.csv", movement_metrics)
    write_csv(output_dir / "burst_metrics.csv", burst_metrics)
    write_csv(output_dir / "closure_metrics.csv", closure_metrics)
    write_csv(output_dir / "direction_metrics.csv", direction_metrics)
    write_csv(output_dir / "chain_ranking_metrics.csv", ranking)
    write_csv(output_dir / "monthly_metrics.csv", monthly)
    write_csv(output_dir / "calibration_bins.csv", calibration)
    write_csv(output_dir / "bootstrap_metrics.csv", bootstraps)
    write_csv(output_dir / "null_metrics.csv", null_metrics)
    write_csv(output_dir / "concentration_metrics.csv", concentration)
    write_csv(output_dir / "delayed_entry_sensitivity.csv", delayed)
    write_json(output_dir / "decision.json", decision)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    if output_dir.resolve() == DEFAULT_PRIMARY.resolve():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


SCIENTIFIC_ARTIFACTS = (
    "source_manifest.json",
    "input_artifact_hashes.json",
    "protected_boundary_audit.json",
    "feature_manifest.json",
    "compact_decision_panel.parquet",
    "movement_thresholds.json",
    "chronological_fold_manifest.json",
    "oof_2024_predictions.parquet",
    "scored_2025_predictions.parquet",
    "model_coefficients.json",
    "movement_metrics.csv",
    "burst_metrics.csv",
    "closure_metrics.csv",
    "direction_metrics.csv",
    "chain_ranking_metrics.csv",
    "monthly_metrics.csv",
    "calibration_bins.csv",
    "bootstrap_metrics.csv",
    "null_metrics.csv",
    "concentration_metrics.csv",
    "delayed_entry_sensitivity.csv",
    "decision.json",
    "report.md",
)
PRE_FINAL_REPRODUCIBILITY_ARTIFACTS = tuple(
    name for name in SCIENTIFIC_ARTIFACTS if name not in {"decision.json", "report.md"}
)


def compare_core_artifacts(
    primary: Path,
    exact: Path,
    artifact_names: Sequence[str] = SCIENTIFIC_ARTIFACTS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    passed = True
    for name in artifact_names:
        primary_path = primary / name
        exact_path = exact / name
        if not primary_path.is_file() or not exact_path.is_file():
            identical = False
            primary_hash = "missing"
            exact_hash = "missing"
        else:
            primary_hash = sha256_file(primary_path)
            exact_hash = sha256_file(exact_path)
            identical = primary_hash == exact_hash
        passed &= identical
        rows.append(
            {
                "artifact": name,
                "primary_sha256": primary_hash,
                "exact_rerun_sha256": exact_hash,
                "identical": identical,
            }
        )
    return {
        **SAFETY_FLAGS,
        "status": "passed" if passed else "failed",
        "all_scientific_artifacts_identical": passed,
        "artifact_count": len(rows),
        "comparisons": rows,
    }


def _run_auditor(artifacts: Path, provider_root: Path) -> bool:
    command = [
        sys.executable,
        str(AUDITOR_PATH),
        "--artifacts",
        str(artifacts),
        "--provider-root",
        str(provider_root),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        return False
    audit_path = artifacts / "independent_audit.json"
    if not audit_path.is_file():
        return False
    return bool(json.loads(audit_path.read_text(encoding="utf-8")).get("passed"))


def _refresh_final_report(output_dir: Path, decision: Mapping[str, Any]) -> None:
    report_path = output_dir / "report.md"
    if not report_path.is_file():
        return
    marker = "\n## Reproducibility and audit\n"
    report = report_path.read_text(encoding="utf-8").split(marker, maxsplit=1)[0]
    report += (
        marker
        + f"- Decision status: `{decision['decision_status']}`.\n"
        + "- Exact rerun: "
        + ("passed" if decision["exact_rerun_passed"] else "failed or pending")
        + ".\n- Independent audit: "
        + ("passed" if decision["independent_audit_passed"] else "failed or pending")
        + ".\n"
    )
    report_path.write_text(report, encoding="utf-8")
    if output_dir.resolve() == DEFAULT_PRIMARY.resolve():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")


def _write_v1_recommendation_if_permitted(output_dir: Path, decision: Mapping[str, Any]) -> None:
    if output_dir.resolve() != DEFAULT_PRIMARY.resolve():
        return
    if decision.get("decision") != "promising_probability_chain_for_intensive_v1":
        return
    recommendation = """# Intensive V1 recommendation

V0 granted research permission for a more intensive V1; it did not establish a
strategy, achieved P&L, executable net edge, or prospective validity.

Recommended V1 scope: more fixed clocks; 12-, 24-, and 36-bar horizons; a larger
stock universe; multiple valid deterministic regime fits; cross-refit path agreement;
soft-state path alternatives; more bootstrap and null draws; full leave-one-stock-out
refits; partial pooling by stock; one compact nonlinear movement model; and a
prospective freeze.

Safety remains `research_only=true`, `feasibility_screen=true`,
`execution_enabled=false`, `order_placement=disabled`,
`broker_integration_required=false`, `strategy_promotion=false`, and
`production_runtime_modified=false`.
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "intensive_v1_recommendation.md").write_text(recommendation, encoding="utf-8")


def finalize_decision(
    output_dir: Path,
    *,
    exact_rerun_passed: bool,
    independent_audit_passed: bool,
) -> dict[str, Any]:
    decision_path = output_dir / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    evidence = decision["evidence"]
    evidence["exact_rerun_passed"] = exact_rerun_passed
    evidence["independent_audit_passed"] = independent_audit_passed
    if not exact_rerun_passed or not independent_audit_passed:
        evidence["blocker"] = "blocked_reproducibility_or_audit_failure"
    elif evidence.get("blocker") == "blocked_reproducibility_or_audit_failure":
        evidence.pop("blocker")
    decision["decision"] = decide_screen(evidence)
    decision["decision_status"] = "final"
    decision["exact_rerun_passed"] = exact_rerun_passed
    decision["independent_audit_passed"] = independent_audit_passed
    write_json(decision_path, decision)
    _refresh_final_report(output_dir, decision)
    _write_v1_recommendation_if_permitted(output_dir, decision)
    return cast(dict[str, Any], decision)


def run_exact_and_audit(
    *,
    primary: Path,
    exact: Path,
    provider_root: Path,
    max_rows: int | None,
    audit: bool,
) -> dict[str, Any]:
    if not (primary / "decision.json").is_file():
        run_screen(primary, provider_root=provider_root, max_rows=max_rows)
    run_screen(exact, provider_root=provider_root, max_rows=max_rows)
    # A previously completed primary contains final audit status while a fresh exact
    # rerun is still provisional. Compare immutable scientific outputs first, then
    # compare decision/report bytes after both lineages have been finalized below.
    comparison = compare_core_artifacts(
        primary,
        exact,
        PRE_FINAL_REPRODUCIBILITY_ARTIFACTS,
    )
    exact_passed = bool(comparison["all_scientific_artifacts_identical"])
    primary_audit = False
    exact_audit = False
    if exact_passed and audit:
        finalize_decision(primary, exact_rerun_passed=True, independent_audit_passed=False)
        finalize_decision(exact, exact_rerun_passed=True, independent_audit_passed=False)
        primary_audit = _run_auditor(primary, provider_root)
        exact_audit = _run_auditor(exact, provider_root)
    audits_passed = primary_audit and exact_audit if audit else False
    primary_decision = finalize_decision(
        primary,
        exact_rerun_passed=exact_passed,
        independent_audit_passed=audits_passed,
    )
    exact_decision = finalize_decision(
        exact,
        exact_rerun_passed=exact_passed,
        independent_audit_passed=audits_passed,
    )
    if audits_passed:
        primary_audit = _run_auditor(primary, provider_root)
        exact_audit = _run_auditor(exact, provider_root)
        audits_passed = primary_audit and exact_audit
        if not audits_passed:
            primary_decision = finalize_decision(
                primary,
                exact_rerun_passed=exact_passed,
                independent_audit_passed=False,
            )
            exact_decision = finalize_decision(
                exact,
                exact_rerun_passed=exact_passed,
                independent_audit_passed=False,
            )
    final_artifacts = compare_core_artifacts(
        primary,
        exact,
        (*SCIENTIFIC_ARTIFACTS, "independent_audit.json"),
    )
    comparison.update(
        {
            "pre_final_derived_artifacts_excluded": ["decision.json", "report.md"],
            "derived_artifacts_compared_after_finalization": [
                "decision.json",
                "report.md",
                "independent_audit.json",
            ],
            "primary_independent_audit_passed": primary_audit,
            "exact_rerun_independent_audit_passed": exact_audit,
            "final_primary_decision": primary_decision["decision"],
            "final_exact_decision": exact_decision["decision"],
            "decisions_identical": primary_decision == exact_decision,
            "all_final_artifacts_identical": final_artifacts["all_scientific_artifacts_identical"],
            "final_artifact_comparisons": final_artifacts["comparisons"],
        }
    )
    write_json(primary / "exact_rerun_manifest.json", comparison)
    write_json(exact / "exact_rerun_manifest.json", comparison)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--exact-rerun", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
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
    provider_root = args.provider_root.expanduser().resolve()
    if args.exact_rerun:
        primary = args.output.resolve() if args.output != DEFAULT_PRIMARY else DEFAULT_PRIMARY
        manifest = run_exact_and_audit(
            primary=primary,
            exact=DEFAULT_EXACT,
            provider_root=provider_root,
            max_rows=args.max_rows,
            audit=bool(args.audit),
        )
        print(canonical_json(manifest), end="")
        if not manifest["all_scientific_artifacts_identical"]:
            raise SystemExit(2)
        if args.audit and not (
            manifest["primary_independent_audit_passed"]
            and manifest["exact_rerun_independent_audit_passed"]
        ):
            raise SystemExit(3)
        return
    decision = run_screen(
        args.output.resolve(),
        provider_root=provider_root,
        max_rows=args.max_rows,
    )
    if args.audit:
        audit_passed = _run_auditor(args.output.resolve(), provider_root)
        decision = finalize_decision(
            args.output.resolve(),
            exact_rerun_passed=False,
            independent_audit_passed=audit_passed,
        )
    print(canonical_json(decision), end="")


if __name__ == "__main__":
    main()
