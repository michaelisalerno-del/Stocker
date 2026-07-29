#!/usr/bin/env python3
"""Run the bounded Opening Regime-Path Direction Screen V0."""

# ruff: noqa: E402 -- numerical thread limits must be fixed before imports.

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-opening-regime-path-matplotlib")

import argparse
import hashlib
import json
import math
import subprocess
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stocker_research.opening_regime_path_screen_v0 import (
    DECISION_ORDINALS,
    SAFETY_FLAGS,
    FrozenLogisticModel,
    cohort_relative_returns_bps,
    current_regime_features,
    decide_screen,
    delayed_entry_and_terminal,
    development_movement_thresholds,
    fit_fixed_logistic,
    interaction_features,
    opening_path_features,
    permute_structural_bundle_within_slates,
    reject_invalid_decision_history,
    session_block_bootstrap_draws,
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
READ_END_INCLUSIVE = PROTECTED_START - pd.Timedelta(microseconds=1)
STATE_COUNT = 8
EXPECTED_SESSION_BARS = 78
MAX_COMPACT_ROWS = 22_000
BOOTSTRAP_DRAWS = 300
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

OBSERVABLE_FEATURES = (
    "opening_gap_bps",
    "open_to_decision_raw_return_bps",
    "open_to_decision_cohort_relative_return_bps",
    "latest_one_bar_return_bps",
    "latest_three_bar_return_bps",
    "opening_realized_volatility_bps",
    "opening_high_low_range_bps",
    "current_true_range_bps",
    "mean_completed_bar_true_range_bps",
    "close_location_within_opening_range",
    "distance_from_opening_high_bps",
    "distance_from_opening_low_bps",
    "positive_close_fraction",
    "directional_close_persistence_ratio",
    "historical_activity_proxy_shock",
    "cross_sectional_dispersion_bps",
)
CURRENT_REGIME_FEATURES = (
    *(f"current_state_{state}" for state in range(STATE_COUNT)),
    *(f"previous_completed_state_{state}" for state in range(STATE_COUNT)),
    *(f"posterior_state_{state}" for state in range(STATE_COUNT)),
    "maximum_posterior_probability",
    "posterior_entropy",
    "current_state_age",
    "opening_state_equals_current",
)
TOPOLOGY_FEATURES = (
    "opening_transition_count",
    "opening_unique_state_count",
    "opening_state_revisit_count",
    "opening_return_to_origin_count",
    "opening_two_state_closure_count",
    "opening_three_state_closure_count",
    "opening_any_short_closure",
    "opening_most_recent_path_was_closure",
    "opening_alternation_ratio",
    "opening_transition_rate",
    "opening_mean_completed_run_duration",
    "opening_maximum_completed_run_duration",
    "opening_minimum_completed_run_duration",
    "opening_most_recent_completed_run_duration",
    "opening_time_since_latest_transition",
    "opening_state_occupancy_entropy",
    "opening_largest_state_occupancy_fraction",
)
INTERACTION_FEATURES = tuple(
    f"current_state_{state}_x_{source}"
    for state in range(STATE_COUNT)
    for source in (
        "any_short_closure",
        "opening_return_to_origin_count",
        "transition_rate",
        "current_state_age",
    )
)
ADDITIVE_STRUCTURAL_FEATURES = (*CURRENT_REGIME_FEATURES, *TOPOLOGY_FEATURES)
STRUCTURAL_BUNDLE_FEATURES = (*ADDITIVE_STRUCTURAL_FEATURES, *INTERACTION_FEATURES)
MODEL_FEATURES = {
    "M0": ("checkpoint_60m",),
    "M1": ("checkpoint_60m", *OBSERVABLE_FEATURES),
    "M2": ("checkpoint_60m", *OBSERVABLE_FEATURES, *ADDITIVE_STRUCTURAL_FEATURES),
    "M3": (
        "checkpoint_60m",
        *OBSERVABLE_FEATURES,
        *ADDITIVE_STRUCTURAL_FEATURES,
        *INTERACTION_FEATURES,
    ),
}
TARGETS = (
    "large_remaining_move",
    "up_given_large_move",
    "remaining_direction_up",
)
PRIMARY_TARGETS = ("large_remaining_move", "up_given_large_move")

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
    if tuple(contract["population"]["decision_ordinals_completed_bar_count"]) != (
        DECISION_ORDINALS
    ):
        raise RuntimeError("contract checkpoint ordinals differ")
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
        REPO_ROOT
        / "packages/stocker_research/src/stocker_research/opening_regime_path_screen_v0.py",
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
        raise RuntimeError("frozen repaired state model hash differs")
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
    expected = frame[[f"state_probability_{state}" for state in range(STATE_COUNT)]].to_numpy(
        dtype=float
    )
    result = {
        "rows": len(frame),
        "hard_state_agreement": float(
            np.mean(summary.hard_states == frame["state"].to_numpy(dtype=int))
        ),
        "maximum_probability_absolute_error": float(
            np.max(np.abs(summary.state_probabilities - expected))
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
        raise RuntimeError("frozen repaired posterior cannot be reproduced")
    return result


def _verify_development_sources(provider_root: Path) -> dict[str, Any]:
    expected = json.loads(SOURCE_IDENTITY_PATH.read_text(encoding="utf-8"))[
        "development_source_hashes"
    ]
    actual: dict[str, str] = {}
    counts: dict[str, int] = {}
    for symbol in (*REGIME_CONTEXT_SYMBOLS, "VTI"):
        digest, rows = bounded_source_hash(
            provider_path(provider_root, symbol),
            start=START,
            end=DEVELOPMENT_END_EXCLUSIVE - pd.Timedelta(microseconds=1),
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


def _source_boundary_audit(
    provider_root: Path,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, int]]:
    parts: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for symbol in (*REGIME_CONTEXT_SYMBOLS, "VTI"):
        path = provider_path(provider_root, symbol)
        timestamp_frame = pd.read_parquet(
            path,
            columns=["timestamp"],
            filters=[
                ("timestamp", ">=", START.to_pydatetime()),
                ("timestamp", "<", PROTECTED_START.to_pydatetime()),
            ],
        )
        timestamp_frame["timestamp"] = pd.to_datetime(
            timestamp_frame["timestamp"], utc=True, errors="raise"
        )
        if timestamp_frame["timestamp"].ge(PROTECTED_START).any():
            raise RuntimeError("protected provider row materialised")
        timestamp_frame["symbol"] = symbol
        parts.append(timestamp_frame)
        hashes[symbol], counts[symbol] = bounded_source_hash(
            path, start=START, end=READ_END_INCLUSIVE
        )
    all_rows = pd.concat(parts, ignore_index=True)
    all_rows["year_month"] = all_rows["timestamp"].dt.strftime("%Y-%m")
    monthly = all_rows.groupby("year_month", sort=True).size().rename("row_count").reset_index()
    return monthly, hashes, counts


def _qa_manifest() -> list[dict[str, Any]]:
    qa_root = Path.home() / "StockerLocal" / "data" / "reports" / "vendor_qa"
    rows: list[dict[str, Any]] = []
    for symbol in (*REGIME_CONTEXT_SYMBOLS, "VTI"):
        stored = "VTI.US" if symbol == "VTI" else symbol
        candidates = sorted(qa_root.glob(f"{stored}*_5m_eodhd_qa.json"))
        if not candidates and symbol == "VTI":
            candidates = sorted(qa_root.glob("VTI*_5m_eodhd_qa.json"))
        if not candidates:
            rows.append(
                {
                    "symbol": symbol,
                    "status": "not_available",
                    "logical_path": None,
                    "sha256": None,
                    "adjusted_close_present": None,
                    "adjusted_close_differences": None,
                }
            )
            continue
        path = candidates[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        adjusted = payload.get("adjusted_close", {})
        rows.append(
            {
                "symbol": symbol,
                "status": payload.get("status", "unknown"),
                "logical_path": f"external_vendor_qa/{path.name}",
                "sha256": sha256_file(path),
                "adjusted_close_present": adjusted.get("present"),
                "adjusted_close_differences": adjusted.get("different_from_close_count"),
            }
        )
    if any(row["status"] == "fail" for row in rows):
        raise RuntimeError("existing vendor QA contains a failed symbol")
    return rows


def _opening_observables(session_frame: pd.DataFrame, origin: int) -> dict[str, float]:
    opening = session_frame.loc[session_frame["bar_ordinal"].between(0, origin)].sort_values(
        "bar_ordinal", kind="mergesort"
    )
    by_ordinal = opening.set_index("bar_ordinal", verify_integrity=True)
    session_open = float(opening.iloc[0]["open"])
    prior_close = float(opening.iloc[0]["prior_session_close"])
    decision_close = float(opening.iloc[-1]["close"])
    opening_high = float(opening["high"].max())
    opening_low = float(opening["low"].min())
    prior_bar_close = opening["close"].shift(1)
    prior_bar_close.iloc[0] = prior_close
    one_step = opening["close"].to_numpy(dtype=float) / prior_bar_close.to_numpy(dtype=float) - 1.0
    true_ranges = np.maximum.reduce(
        [
            opening["high"].to_numpy(dtype=float) - opening["low"].to_numpy(dtype=float),
            np.abs(opening["high"].to_numpy(dtype=float) - prior_bar_close.to_numpy(dtype=float)),
            np.abs(opening["low"].to_numpy(dtype=float) - prior_bar_close.to_numpy(dtype=float)),
        ]
    )
    signed_sum = float(np.sum(one_step))
    absolute_sum = float(np.sum(np.abs(one_step)))
    range_width = opening_high - opening_low
    return {
        "opening_gap_bps": 10_000.0 * (session_open / prior_close - 1.0),
        "open_to_decision_raw_return_bps": 10_000.0 * (decision_close / session_open - 1.0),
        "latest_one_bar_return_bps": 10_000.0
        * (decision_close / float(by_ordinal.loc[origin - 1, "close"]) - 1.0),
        "latest_three_bar_return_bps": 10_000.0
        * (decision_close / float(by_ordinal.loc[origin - 3, "close"]) - 1.0),
        "opening_realized_volatility_bps": 10_000.0 * float(np.std(one_step, ddof=0)),
        "opening_high_low_range_bps": 10_000.0 * range_width / session_open,
        "current_true_range_bps": 10_000.0 * true_ranges[-1] / prior_bar_close.iloc[-1],
        "mean_completed_bar_true_range_bps": 10_000.0
        * float(np.mean(true_ranges / prior_bar_close.to_numpy(dtype=float))),
        "close_location_within_opening_range": (
            (decision_close - opening_low) / range_width if range_width > 0.0 else 0.5
        ),
        "distance_from_opening_high_bps": 10_000.0 * (opening_high - decision_close) / opening_high,
        "distance_from_opening_low_bps": 10_000.0 * (decision_close - opening_low) / opening_low,
        "positive_close_fraction": float((opening["close"] > opening["open"]).mean()),
        "directional_close_persistence_ratio": (
            abs(signed_sum) / absolute_sum if absolute_sum > 0.0 else 0.0
        ),
        "historical_activity_proxy_shock": float(
            opening.iloc[-1]["log_relative_cumulative_historical_volume"]
        ),
    }


def _whole_session_is_valid(session_frame: pd.DataFrame) -> bool:
    ordered = session_frame.sort_values("bar_ordinal", kind="mergesort")
    ordinals = pd.to_numeric(ordered["bar_ordinal"], errors="coerce")
    prices = ordered[["open", "high", "low", "close"]].to_numpy(dtype=float)
    return bool(
        len(ordered) == EXPECTED_SESSION_BARS
        and ordinals.tolist() == list(range(EXPECTED_SESSION_BARS))
        and ordered["segment_id"].astype(str).nunique() == 1
        and ordered["session"].astype(str).nunique() == 1
        and ordered["expected_session_bars"].astype(int).eq(EXPECTED_SESSION_BARS).all()
        and ordered["session_source_complete"].astype(bool).all()
        and not ordered["source_data_error_in_session"].astype(bool).any()
        and np.isfinite(prices).all()
        and bool((prices > 0.0).all())
        and not pd.to_datetime(ordered["bar_start_timestamp"], utc=True).ge(PROTECTED_START).any()
    )


def build_compact_panel(
    provider_root: Path,
    preprocessing: EmissionPreprocessing,
    parameters: SemiMarkovParameters,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    built = build_regime_panel(
        RegimePanelConfig(
            provider_root=provider_root,
            symbols=REGIME_CONTEXT_SYMBOLS,
            benchmark_symbol="VTI",
            start=START,
            end=READ_END_INCLUSIVE,
        )
    )
    panel = built.frame.sort_values(list(NATURAL_KEY), kind="mergesort").reset_index(drop=True)
    if panel["bar_start_timestamp"].ge(PROTECTED_START).any():
        raise RuntimeError("protected row entered regime panel")
    development = panel.loc[panel["bar_start_timestamp"].lt(DEVELOPMENT_END_EXCLUSIVE)]
    development_hash = canonical_frame_hash(development, columns=(*NATURAL_KEY, *EMISSION_FEATURES))
    if development_hash != EXPECTED_PANEL_HASH:
        raise RuntimeError("archived 2024 panel feature hash differs")
    state_panel = panel.loc[
        panel["symbol"].isin(DECISION_SYMBOLS) & panel["bar_ordinal"].le(11)
    ].copy()
    source_panel_indices = state_panel.index.to_numpy(dtype=np.int64)
    state_panel = state_panel.reset_index(drop=True)
    summary = causal_filter_summary(
        gaussian_log_emissions(transform_emissions(state_panel, preprocessing), parameters),
        groups=causal_segment_groups(state_panel),
        model=parameters.as_dict(),
    )
    panel["causal_hard_state"] = np.int16(-1)
    panel.loc[source_panel_indices, "causal_hard_state"] = summary.hard_states.astype(np.int16)
    for state in range(STATE_COUNT):
        column = f"causal_posterior_state_{state}"
        panel[column] = np.nan
        panel.loc[source_panel_indices, column] = summary.state_probabilities[:, state]

    rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    symbols = set(DECISION_SYMBOLS)
    for (symbol_value, session_value), session_frame in panel.groupby(
        ["symbol", "session"], sort=True
    ):
        symbol = str(symbol_value)
        session = str(session_value)
        if symbol not in symbols or not _whole_session_is_valid(session_frame):
            continue
        ordered = session_frame.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
        for decision_ordinal in DECISION_ORDINALS:
            try:
                history = reject_invalid_decision_history(
                    ordered, decision_ordinal=decision_ordinal
                )
                anchor = delayed_entry_and_terminal(ordered, decision_ordinal=decision_ordinal)
            except ValueError:
                continue
            origin = history.bar_start_ordinal
            opening = ordered.loc[ordered["bar_ordinal"].between(0, origin)]
            state_path = opening["causal_hard_state"].to_numpy(dtype=int)
            current = opening.iloc[-1]
            posterior = np.asarray(
                [current[f"causal_posterior_state_{state}"] for state in range(STATE_COUNT)],
                dtype=float,
            )
            topology = opening_path_features(state_path)
            regime = current_regime_features(state_path, posterior, state_count=STATE_COUNT)
            interactions = interaction_features(
                int(state_path[-1]), topology, state_count=STATE_COUNT
            )
            observables = _opening_observables(ordered, origin)
            record: dict[str, Any] = {
                "symbol": symbol,
                "session": session,
                "year": int(session[:4]),
                "year_month": session[:7],
                "decision_ordinal": decision_ordinal,
                "repo_bar_start_ordinal": origin,
                "decision_time_america_new_york": ("10:00" if decision_ordinal == 6 else "10:30"),
                "checkpoint_60m": float(decision_ordinal == 12),
                "slate_id": f"{session}|{decision_ordinal:02d}",
                "decision_bar_start_timestamp_utc": history.decision_timestamp,
                "feature_available_timestamp_utc": history.feature_available_timestamp,
                "entry_bar_ordinal": anchor.entry_bar_ordinal,
                "delayed_entry_open": anchor.delayed_entry_open,
                "terminal_bar_ordinal": anchor.terminal_bar_ordinal,
                "terminal_close": anchor.terminal_close,
                "raw_remaining_return_bps": 10_000.0
                * (anchor.terminal_close / anchor.delayed_entry_open - 1.0),
                "current_state": int(state_path[-1]),
                "opening_state": int(state_path[0]),
                "opening_completed_bar_count": len(state_path),
                "state_model_hash": EXPECTED_MODEL_HASH,
                "representation_status": "representation_specific_feasibility_evidence",
                **observables,
                **topology,
                **regime,
                **interactions,
            }
            rows.append(record)
            ledger_rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "decision_ordinal": decision_ordinal,
                    "slate_id": record["slate_id"],
                    "repo_bar_start_ordinal": origin,
                    "feature_available_timestamp_utc": history.feature_available_timestamp,
                    "opening_state_path": ",".join(str(value) for value in state_path),
                    "opening_bar_ordinals": ",".join(str(value) for value in range(origin + 1)),
                    "current_state": int(state_path[-1]),
                    "current_posterior": ",".join(f"{value:.17g}" for value in posterior),
                    **topology,
                }
            )
    compact = pd.DataFrame(rows).sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    compact = compact.reset_index(drop=True)
    ledger = pd.DataFrame(ledger_rows).sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    ledger = ledger.reset_index(drop=True)
    pre_cross_observables = [
        feature
        for feature in OBSERVABLE_FEATURES
        if feature
        not in {
            "open_to_decision_cohort_relative_return_bps",
            "cross_sectional_dispersion_bps",
        }
    ]
    finite_base = np.isfinite(compact.loc[:, pre_cross_observables].to_numpy(dtype=float)).all(
        axis=1
    )
    compact = compact.loc[finite_base].copy()
    valid_keys = set(
        zip(
            compact["symbol"],
            compact["session"],
            compact["decision_ordinal"],
            strict=True,
        )
    )
    ledger = ledger.loc[
        [
            (row.symbol, row.session, row.decision_ordinal) in valid_keys
            for row in ledger.itertuples()
        ]
    ].copy()

    for _, indices in compact.groupby("slate_id", sort=True).groups.items():
        index = list(indices)
        opening_returns = compact.loc[index, "open_to_decision_raw_return_bps"].to_numpy(
            dtype=float
        )
        residuals, _ = cohort_relative_returns_bps(opening_returns)
        compact.loc[index, "open_to_decision_cohort_relative_return_bps"] = residuals
        compact.loc[index, "cross_sectional_dispersion_bps"] = float(
            np.std(opening_returns, ddof=1)
        )
        outcome = compact.loc[index, "raw_remaining_return_bps"].to_numpy(dtype=float)
        outcome_residual, medians = cohort_relative_returns_bps(outcome)
        compact.loc[index, "cohort_median_return_minus_i_bps"] = medians
        compact.loc[index, "residual_remaining_return_bps"] = outcome_residual
    compact["slate_size"] = compact.groupby("slate_id", sort=True)["symbol"].transform("size")
    compact = compact.loc[compact["slate_size"].ge(2)].copy()
    finite_features = np.isfinite(
        compact.loc[:, [*OBSERVABLE_FEATURES, *STRUCTURAL_BUNDLE_FEATURES]].to_numpy(dtype=float)
    ).all(axis=1)
    compact = compact.loc[finite_features].copy()
    compact["slate_size"] = compact.groupby("slate_id", sort=True)["symbol"].transform("size")
    compact = compact.loc[compact["slate_size"].ge(2)].copy()
    compact = compact.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if len(compact) > MAX_COMPACT_ROWS:
        raise RuntimeError("compact opening population exceeds 22,000 rows")
    thresholds = development_movement_thresholds(compact)
    compact["movement_threshold_bps"] = compact["decision_ordinal"].map(thresholds)
    compact["large_remaining_move"] = (
        compact["residual_remaining_return_bps"].abs() >= compact["movement_threshold_bps"]
    ).astype(np.int8)
    compact["up_given_large_move"] = (
        compact["residual_remaining_return_bps"].gt(0.0).astype(np.int8)
    )
    compact["remaining_direction_up"] = compact["up_given_large_move"].astype(np.int8)
    valid_keys = set(
        zip(
            compact["symbol"],
            compact["session"],
            compact["decision_ordinal"],
            strict=True,
        )
    )
    ledger = ledger.loc[
        [
            (row.symbol, row.session, row.decision_ordinal) in valid_keys
            for row in ledger.itertuples()
        ]
    ].sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
    context = {
        "combined_source_hashes": built.source_hashes,
        "combined_source_row_counts": built.source_row_counts,
        "combined_snapshot_hash": built.data_snapshot_hash,
        "combined_panel_feature_hash": built.feature_table_hash,
        "development_panel_feature_hash": development_hash,
        "panel_rows_materialised": len(panel),
        "state_rows_causally_filtered": len(state_panel),
        "state_filter_maximum_bar_ordinal": 11,
        "state_filter_symbols": list(DECISION_SYMBOLS),
        "panel_minimum_timestamp": str(panel["bar_start_timestamp"].min()),
        "panel_maximum_timestamp": str(panel["bar_start_timestamp"].max()),
        "protected_rows_materialised": int(panel["bar_start_timestamp"].ge(PROTECTED_START).sum()),
        "gap_ledger_rows": len(built.gap_ledger),
        "movement_thresholds": {str(key): value for key, value in thresholds.items()},
    }
    return compact, ledger.reset_index(drop=True), context


def _target_population(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "up_given_large_move":
        return frame.loc[frame["large_remaining_move"].eq(1)].copy()
    return frame.copy()


def fit_model_ladder(
    development: pd.DataFrame,
) -> tuple[dict[str, dict[str, FrozenLogisticModel]], dict[str, Any]]:
    models: dict[str, dict[str, FrozenLogisticModel]] = {}
    serialized: dict[str, Any] = {}
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    for target in TARGETS:
        training = _target_population(development, target).reset_index(drop=True)
        models[target] = {}
        serialized[target] = {}
        for model_name in ("M0", "M1", "M2", "M3"):
            model = fit_fixed_logistic(
                training,
                training[target].astype(int),
                features=MODEL_FEATURES[model_name],
                slate_column="slate_id",
                model_id=f"{target}__{model_name}",
            )
            models[target][model_name] = model
            serialized[target][model_name] = model.as_dict()
    return models, serialized


def score_models(
    frame: pd.DataFrame,
    models: Mapping[str, Mapping[str, FrozenLogisticModel]],
) -> pd.DataFrame:
    scored = frame.copy()
    for target in TARGETS:
        for model_name in ("M0", "M1", "M2", "M3"):
            scored[f"p__{target}__{model_name}"] = models[target][model_name].predict(scored)
    return scored


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def calibration_intercept_slope(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return math.nan, math.nan
    predictor = _logit(probabilities)
    if float(np.std(predictor)) < 1e-12:
        base = float(np.mean(labels))
        return float(_logit(np.asarray([base]))[0]), 0.0

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + parameters[1] * predictor
        predicted = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        loss = float(
            -np.sum(
                labels * np.log(np.clip(predicted, 1e-12, 1.0))
                + (1 - labels) * np.log(np.clip(1.0 - predicted, 1e-12, 1.0))
            )
        )
        gradient = np.asarray(
            [np.sum(predicted - labels), np.sum((predicted - labels) * predictor)],
            dtype=float,
        )
        return loss, gradient

    result = minimize(
        lambda parameters: objective(parameters)[0],
        x0=np.asarray([0.0, 1.0]),
        jac=lambda parameters: objective(parameters)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    if not np.isfinite(result.x).all():
        return math.nan, math.nan
    return float(result.x[0]), float(result.x[1])


def calibration_table(
    frame: pd.DataFrame,
    *,
    target: str,
    model_name: str,
    scope: str,
) -> pd.DataFrame:
    population = _target_population(frame, target)
    labels = population[target].to_numpy(dtype=int)
    probabilities = population[f"p__{target}__{model_name}"].to_numpy(dtype=float)
    bin_numbers = np.minimum((np.clip(probabilities, 0.0, 1.0) * 10).astype(int), 9)
    rows: list[dict[str, Any]] = []
    for bin_number in range(10):
        mask = bin_numbers == bin_number
        rows.append(
            {
                "scope": scope,
                "target": target,
                "model": model_name,
                "bin": bin_number + 1,
                "probability_lower": bin_number / 10.0,
                "probability_upper": (bin_number + 1) / 10.0,
                "row_count": int(mask.sum()),
                "mean_predicted_probability": (
                    float(np.mean(probabilities[mask])) if mask.any() else math.nan
                ),
                "observed_rate": float(np.mean(labels[mask])) if mask.any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def metric_record(
    frame: pd.DataFrame,
    *,
    target: str,
    model_name: str,
    scope: str,
) -> dict[str, Any]:
    population = _target_population(frame, target)
    labels = population[target].to_numpy(dtype=int)
    probabilities = np.clip(
        population[f"p__{target}__{model_name}"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12
    )
    intercept, slope = calibration_intercept_slope(labels, probabilities)
    bins = calibration_table(population, target=target, model_name=model_name, scope=scope)
    populated = bins.loc[bins["row_count"].gt(0)]
    ece = float(
        np.sum(
            populated["row_count"]
            / len(population)
            * (populated["observed_rate"] - populated["mean_predicted_probability"]).abs()
        )
    )
    return {
        "scope": scope,
        "target": target,
        "model": model_name,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "auc": (
            float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else math.nan
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": ece,
        "base_rate": float(np.mean(labels)),
        "row_count": len(population),
        "session_count": int(population["session"].astype(str).nunique()),
        "stock_count": int(population["symbol"].astype(str).nunique()),
    }


def evaluate_predictions(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    calibration_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        for model_name in ("M0", "M1", "M2", "M3"):
            pooled_rows.append(
                metric_record(assessment, target=target, model_name=model_name, scope="pooled")
            )
            calibration_parts.append(
                calibration_table(assessment, target=target, model_name=model_name, scope="pooled")
            )
            for decision_ordinal in DECISION_ORDINALS:
                subset = assessment.loc[assessment["decision_ordinal"].eq(decision_ordinal)]
                checkpoint_rows.append(
                    metric_record(
                        subset,
                        target=target,
                        model_name=model_name,
                        scope=f"checkpoint_{decision_ordinal}",
                    )
                )
            for month in sorted(assessment["year_month"].astype(str).unique()):
                subset = assessment.loc[assessment["year_month"].astype(str).eq(month)]
                monthly_rows.append(
                    metric_record(
                        subset,
                        target=target,
                        model_name=model_name,
                        scope=month,
                    )
                )
    pooled = pd.DataFrame(pooled_rows)
    movement = pooled.loc[pooled["target"].eq("large_remaining_move")].reset_index(drop=True)
    direction = pooled.loc[pooled["target"].ne("large_remaining_move")].reset_index(drop=True)
    return (
        movement,
        direction,
        pd.DataFrame(monthly_rows),
        pd.DataFrame(checkpoint_rows),
        pd.concat(calibration_parts, ignore_index=True),
    )


def _metric_lookup(frame: pd.DataFrame, target: str, model_name: str, metric: str) -> float:
    row = frame.loc[frame["target"].eq(target) & frame["model"].eq(model_name)]
    if len(row) != 1:
        raise RuntimeError(f"metric lookup is not unique: {target}/{model_name}/{metric}")
    return float(row.iloc[0][metric])


def _improvement(
    frame: pd.DataFrame,
    target: str,
    candidate: str,
    baseline: str,
    metric: str,
) -> float:
    return _metric_lookup(frame, target, baseline, metric) - _metric_lookup(
        frame, target, candidate, metric
    )


def movement_scale_reference(
    development_scored: pd.DataFrame, assessment_scored: pd.DataFrame
) -> tuple[pd.Series, dict[str, Any]]:
    scale = pd.Series(index=assessment_scored.index, dtype=float)
    manifest: dict[str, Any] = {
        "method": (
            "2024_checkpoint_specific_median_absolute_remaining_movement_by_M1_probability_decile"
        ),
        "checkpoints": {},
    }
    for decision_ordinal in DECISION_ORDINALS:
        dev = development_scored.loc[development_scored["decision_ordinal"].eq(decision_ordinal)]
        score = assessment_scored.loc[assessment_scored["decision_ordinal"].eq(decision_ordinal)]
        probabilities = dev["p__large_remaining_move__M1"].to_numpy(dtype=float)
        edges = np.quantile(probabilities, np.linspace(0.0, 1.0, 11))
        edges = np.maximum.accumulate(edges)
        dev_bins = np.minimum(np.searchsorted(edges[1:-1], probabilities, side="right"), 9)
        overall = float(dev["residual_remaining_return_bps"].abs().median())
        medians: list[float] = []
        for bin_number in range(10):
            values = dev.loc[dev_bins == bin_number, "residual_remaining_return_bps"].abs()
            medians.append(float(values.median()) if not values.empty else overall)
        score_probabilities = score["p__large_remaining_move__M1"].to_numpy(dtype=float)
        score_bins = np.minimum(np.searchsorted(edges[1:-1], score_probabilities, side="right"), 9)
        scale.loc[score.index] = np.asarray(medians, dtype=float)[score_bins]
        manifest["checkpoints"][str(decision_ordinal)] = {
            "probability_edges": edges.tolist(),
            "median_absolute_remaining_movement_bps": medians,
            "fallback_overall_median_bps": overall,
        }
    if scale.isna().any() or not np.isfinite(scale.to_numpy()).all():
        raise RuntimeError("movement-scale reference is incomplete")
    return scale, manifest


def add_signed_scores(assessment: pd.DataFrame, movement_scale: pd.Series) -> pd.DataFrame:
    scored = assessment.copy()
    scored["predicted_remaining_movement_scale_bps"] = movement_scale
    for model_name in ("M0", "M1", "M2", "M3"):
        scored[f"expected_signed_remaining_move_score__{model_name}"] = (
            scored[f"p__large_remaining_move__{model_name}"]
            * (2.0 * scored[f"p__up_given_large_move__{model_name}"] - 1.0)
            * scored["predicted_remaining_movement_scale_bps"]
        )
    return scored


def economic_selection_ledger(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slate_id, slate in assessment.groupby("slate_id", sort=True):
        ordered = slate.sort_values("symbol", kind="mergesort")
        candidates: dict[str, tuple[pd.Series | None, bool]] = {}
        for model_name in ("M0", "M1", "M2", "M3"):
            score_column = f"expected_signed_remaining_move_score__{model_name}"
            selected = ordered.sort_values(
                [score_column, "symbol"], ascending=[False, True], kind="mergesort"
            ).iloc[0]
            candidates[model_name] = (
                selected if float(selected[score_column]) > 0.0 else None,
                float(selected[score_column]) > 0.0,
            )
        candidates["highest_opening_relative_momentum"] = (
            ordered.sort_values(
                ["open_to_decision_cohort_relative_return_bps", "symbol"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0],
            True,
        )
        candidates["strongest_opening_reversal"] = (
            ordered.sort_values(
                ["open_to_decision_cohort_relative_return_bps", "symbol"],
                ascending=[True, True],
                kind="mergesort",
            ).iloc[0],
            True,
        )
        candidates["highest_M1_large_move_probability"] = (
            ordered.sort_values(
                ["p__large_remaining_move__M1", "symbol"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0],
            True,
        )
        random_seed = int.from_bytes(
            hashlib.sha256(f"20260722|{slate_id}".encode()).digest()[:8], "big"
        )
        candidates["random_within_slate"] = (
            ordered.iloc[random_seed % len(ordered)],
            True,
        )
        for candidate, (selected, selected_flag) in candidates.items():
            rows.append(
                {
                    "slate_id": str(slate_id),
                    "session": str(ordered.iloc[0]["session"]),
                    "decision_ordinal": int(ordered.iloc[0]["decision_ordinal"]),
                    "candidate": candidate,
                    "selected": selected_flag,
                    "selected_symbol": str(selected["symbol"]) if selected is not None else None,
                    "raw_remaining_return_bps": (
                        float(selected["raw_remaining_return_bps"]) if selected is not None else 0.0
                    ),
                    "cohort_relative_remaining_return_bps": (
                        float(selected["residual_remaining_return_bps"])
                        if selected is not None
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def economic_reference_metrics(selection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, candidate_frame in selection.groupby("candidate", sort=True):
        selected = candidate_frame.loc[candidate_frame["selected"]]
        for friction in (0.0, 10.0, 20.0):
            rows.append(
                {
                    "record_type": "candidate_summary",
                    "candidate": candidate,
                    "friction_bps": friction,
                    "slate_count": int(candidate_frame["slate_id"].nunique()),
                    "selected_slate_count": len(selected),
                    "selection_rate": float(len(selected) / len(candidate_frame)),
                    "mean_raw_remaining_return_bps": (
                        float((selected["raw_remaining_return_bps"] - friction).mean())
                        if not selected.empty
                        else math.nan
                    ),
                    "mean_cohort_relative_remaining_return_bps": (
                        float((selected["cohort_relative_remaining_return_bps"] - friction).mean())
                        if not selected.empty
                        else math.nan
                    ),
                    "paired_mean_cohort_relative_difference_vs_M1_bps": math.nan,
                }
            )
    pivot = selection.pivot(
        index="slate_id", columns="candidate", values="cohort_relative_remaining_return_bps"
    )
    for candidate in ("M0", "M2", "M3"):
        rows.append(
            {
                "record_type": "candidate_minus_M1_paired",
                "candidate": candidate,
                "friction_bps": 0.0,
                "slate_count": len(pivot),
                "selected_slate_count": int(
                    selection.loc[selection["candidate"].eq(candidate), "selected"].sum()
                ),
                "selection_rate": math.nan,
                "mean_raw_remaining_return_bps": math.nan,
                "mean_cohort_relative_remaining_return_bps": math.nan,
                "paired_mean_cohort_relative_difference_vs_M1_bps": float(
                    (pivot[candidate] - pivot["M1"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _binary_loss(labels: np.ndarray, probabilities: np.ndarray, metric: str) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    if metric == "brier":
        return float(np.mean((labels - clipped) ** 2))
    if metric == "log_loss":
        return float(log_loss(labels, clipped, labels=[0, 1]))
    raise ValueError(metric)


def bootstrap_metrics(assessment: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    specifications = (
        (
            "movement_M2_minus_M1_brier_improvement",
            "large_remaining_move",
            "M2",
            "M1",
            "brier",
        ),
        (
            "movement_M3_minus_M2_brier_improvement",
            "large_remaining_move",
            "M3",
            "M2",
            "brier",
        ),
        (
            "direction_M2_minus_M1_brier_improvement",
            "up_given_large_move",
            "M2",
            "M1",
            "brier",
        ),
        (
            "direction_M3_minus_M2_brier_improvement",
            "up_given_large_move",
            "M3",
            "M2",
            "brier",
        ),
        (
            "movement_M2_minus_M1_log_loss_improvement",
            "large_remaining_move",
            "M2",
            "M1",
            "log_loss",
        ),
        (
            "movement_M3_minus_M2_log_loss_improvement",
            "large_remaining_move",
            "M3",
            "M2",
            "log_loss",
        ),
        (
            "direction_M2_minus_M1_log_loss_improvement",
            "up_given_large_move",
            "M2",
            "M1",
            "log_loss",
        ),
        (
            "direction_M3_minus_M2_log_loss_improvement",
            "up_given_large_move",
            "M3",
            "M2",
            "log_loss",
        ),
    )
    selection_pivot = selection.pivot(
        index="slate_id", columns="candidate", values="cohort_relative_remaining_return_bps"
    )
    selection_pivot["session"] = selection_pivot.index.to_series().str.split("|").str[0]
    economic_by_session = (
        (selection_pivot["M3"] - selection_pivot["M1"])
        .groupby(selection_pivot["session"], sort=True)
        .mean()
    )
    session_frames = {
        str(session): frame.copy() for session, frame in assessment.groupby("session", sort=True)
    }
    rows: list[dict[str, Any]] = []
    draws = session_block_bootstrap_draws(
        assessment["session"].astype(str), draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    for draw in draws:
        sampled = pd.concat(
            [session_frames[session] for session in draw.sampled_sessions],
            ignore_index=True,
        )
        for metric_name, target, candidate, baseline, loss_name in specifications:
            population = _target_population(sampled, target)
            labels = population[target].to_numpy(dtype=int)
            candidate_loss = _binary_loss(
                labels,
                population[f"p__{target}__{candidate}"].to_numpy(dtype=float),
                loss_name,
            )
            baseline_loss = _binary_loss(
                labels,
                population[f"p__{target}__{baseline}"].to_numpy(dtype=float),
                loss_name,
            )
            rows.append(
                {
                    "record_type": "draw",
                    "draw": draw.draw,
                    "metric": metric_name,
                    "value": baseline_loss - candidate_loss,
                    "confidence_level": math.nan,
                    "lower": math.nan,
                    "upper": math.nan,
                }
            )
        economic_values = economic_by_session.loc[list(draw.sampled_sessions)].to_numpy(dtype=float)
        rows.append(
            {
                "record_type": "draw",
                "draw": draw.draw,
                "metric": "M3_minus_M1_top_one_cohort_relative_bps",
                "value": float(np.mean(economic_values)),
                "confidence_level": math.nan,
                "lower": math.nan,
                "upper": math.nan,
            }
        )
    draws_frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for metric, values in draws_frame.groupby("metric", sort=True)["value"]:
        array = values.to_numpy(dtype=float)
        for confidence, lower_q, upper_q in ((0.90, 0.05, 0.95), (0.95, 0.025, 0.975)):
            summaries.append(
                {
                    "record_type": "interval",
                    "draw": -1,
                    "metric": metric,
                    "value": float(np.mean(array)),
                    "confidence_level": confidence,
                    "lower": float(np.quantile(array, lower_q)),
                    "upper": float(np.quantile(array, upper_q)),
                }
            )
    return pd.concat([draws_frame, pd.DataFrame(summaries)], ignore_index=True)


def _null_real_values(pooled_metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "movement_M2_minus_M1_brier_improvement": _improvement(
            pooled_metrics,
            "large_remaining_move",
            "M2",
            "M1",
            "brier_score",
        ),
        "movement_M3_minus_M2_brier_improvement": _improvement(
            pooled_metrics,
            "large_remaining_move",
            "M3",
            "M2",
            "brier_score",
        ),
        "direction_M2_minus_M1_brier_improvement": _improvement(
            pooled_metrics,
            "up_given_large_move",
            "M2",
            "M1",
            "brier_score",
        ),
        "direction_M3_minus_M2_brier_improvement": _improvement(
            pooled_metrics,
            "up_given_large_move",
            "M3",
            "M2",
            "brier_score",
        ),
    }


def structure_null_metrics(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
) -> pd.DataFrame:
    real = _null_real_values(pooled_metrics)
    rows: list[dict[str, Any]] = []
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    for draw in range(NULL_DRAWS):
        permuted_development = permute_structural_bundle_within_slates(
            development.reset_index(drop=True),
            structural_columns=STRUCTURAL_BUNDLE_FEATURES,
            seed=NULL_SEED,
            draw=draw,
        )
        permuted_assessment = permute_structural_bundle_within_slates(
            assessment.reset_index(drop=True),
            structural_columns=STRUCTURAL_BUNDLE_FEATURES,
            seed=NULL_SEED + 1,
            draw=draw,
        )
        for target, label in (
            ("large_remaining_move", "movement"),
            ("up_given_large_move", "direction"),
        ):
            training = _target_population(permuted_development, target).reset_index(drop=True)
            scored_population = _target_population(permuted_assessment, target)
            m2 = fit_fixed_logistic(
                training,
                training[target].astype(int),
                features=MODEL_FEATURES["M2"],
                slate_column="slate_id",
                model_id=f"null_{draw}_{target}_M2",
            )
            m3 = fit_fixed_logistic(
                training,
                training[target].astype(int),
                features=MODEL_FEATURES["M3"],
                slate_column="slate_id",
                model_id=f"null_{draw}_{target}_M3",
            )
            labels = scored_population[target].to_numpy(dtype=int)
            p_m1 = assessment.loc[scored_population.index, f"p__{target}__M1"].to_numpy(dtype=float)
            p_m2 = m2.predict(scored_population)
            p_m3 = m3.predict(scored_population)
            values = {
                f"{label}_M2_minus_M1_brier_improvement": _binary_loss(labels, p_m1, "brier")
                - _binary_loss(labels, p_m2, "brier"),
                f"{label}_M3_minus_M2_brier_improvement": _binary_loss(labels, p_m2, "brier")
                - _binary_loss(labels, p_m3, "brier"),
            }
            for metric, value in values.items():
                rows.append(
                    {
                        "record_type": "draw",
                        "draw": draw,
                        "metric": metric,
                        "value": value,
                        "real_value": real[metric],
                        "null_90th_percentile": math.nan,
                        "real_percentile_under_null": math.nan,
                    }
                )
    draws = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for metric, values in draws.groupby("metric", sort=True)["value"]:
        array = values.to_numpy(dtype=float)
        summaries.append(
            {
                "record_type": "summary",
                "draw": -1,
                "metric": metric,
                "value": float(np.mean(array)),
                "real_value": real[metric],
                "null_90th_percentile": float(np.quantile(array, 0.90)),
                "real_percentile_under_null": float(np.mean(array <= real[metric])),
            }
        )
    return pd.concat([draws, pd.DataFrame(summaries)], ignore_index=True)


def concentration_and_support(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(assessment)
    for symbol, count in assessment["symbol"].value_counts(sort=False).sort_index().items():
        rows.append(
            {
                "dimension": "stock_row_share",
                "value": str(symbol),
                "row_count": int(count),
                "share": float(count / total),
                "gate": 0.10,
                "passes": bool(count / total <= 0.10),
            }
        )
    for state, count in assessment["current_state"].value_counts(sort=False).sort_index().items():
        rows.append(
            {
                "dimension": "current_state_row_share",
                "value": str(state),
                "row_count": int(count),
                "share": float(count / total),
                "gate": 0.40,
                "passes": bool(count / total <= 0.40),
            }
        )
    large = assessment.loc[assessment["large_remaining_move"].eq(1)]
    closure_rows = int(assessment["opening_any_short_closure"].eq(1.0).sum())
    support = {
        "assessment_rows": total,
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "actual_large_remaining_moves": len(large),
        "actual_large_moves_by_checkpoint": {
            str(ordinal): int(large["decision_ordinal"].eq(ordinal).sum())
            for ordinal in DECISION_ORDINALS
        },
        "represented_months": int(assessment["year_month"].nunique()),
        "maximum_stock_row_share": float(assessment["symbol"].value_counts().max() / total),
        "maximum_current_state_row_share": float(
            assessment["current_state"].value_counts().max() / total
        ),
        "opening_short_closure_rows": closure_rows,
        "interaction_support_status": (
            "sufficient" if closure_rows >= 100 else "interaction_support_insufficient"
        ),
    }
    support["primary_support_passes"] = bool(
        support["assessment_rows"] >= 3000
        and support["assessment_sessions"] >= 100
        and support["assessment_stocks"] >= 15
        and support["actual_large_remaining_moves"] >= 600
        and all(count >= 250 for count in support["actual_large_moves_by_checkpoint"].values())
        and support["represented_months"] >= 6
        and support["maximum_stock_row_share"] <= 0.10
        and support["maximum_current_state_row_share"] <= 0.40
    )
    rows.extend(
        [
            {
                "dimension": "opening_short_closure_support",
                "value": "all",
                "row_count": closure_rows,
                "share": float(closure_rows / total),
                "gate": 100.0,
                "passes": closure_rows >= 100,
            },
            {
                "dimension": "primary_support",
                "value": "all",
                "row_count": total,
                "share": 1.0,
                "gate": math.nan,
                "passes": support["primary_support_passes"],
            },
        ]
    )
    return pd.DataFrame(rows), support


def _bootstrap_lower(bootstrap: pd.DataFrame, metric: str, confidence: float = 0.90) -> float:
    row = bootstrap.loc[
        bootstrap["record_type"].eq("interval")
        & bootstrap["metric"].eq(metric)
        & bootstrap["confidence_level"].eq(confidence)
    ]
    if len(row) != 1:
        raise RuntimeError(f"bootstrap interval missing: {metric}")
    return float(row.iloc[0]["lower"])


def _null_summary(null: pd.DataFrame, metric: str) -> pd.Series:
    row = null.loc[null["record_type"].eq("summary") & null["metric"].eq(metric)]
    if len(row) != 1:
        raise RuntimeError(f"null summary missing: {metric}")
    return row.iloc[0]


def _positive_month_count(monthly: pd.DataFrame, target: str, candidate: str, baseline: str) -> int:
    candidate_rows = monthly.loc[
        monthly["target"].eq(target) & monthly["model"].eq(candidate),
        ["scope", "brier_score"],
    ].set_index("scope")
    baseline_rows = monthly.loc[
        monthly["target"].eq(target) & monthly["model"].eq(baseline),
        ["scope", "brier_score"],
    ].set_index("scope")
    improvement = baseline_rows["brier_score"] - candidate_rows["brier_score"]
    return int(improvement.gt(0.0).sum())


def _checkpoint_improvements(
    checkpoint: pd.DataFrame, target: str, candidate: str, baseline: str
) -> dict[str, float]:
    output: dict[str, float] = {}
    for ordinal in DECISION_ORDINALS:
        subset = checkpoint.loc[checkpoint["scope"].eq(f"checkpoint_{ordinal}")]
        output[str(ordinal)] = _improvement(subset, target, candidate, baseline, "brier_score")
    return output


def decide_from_artifacts(
    pooled: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    support: Mapping[str, Any],
) -> dict[str, Any]:
    movement_brier = _improvement(pooled, "large_remaining_move", "M2", "M1", "brier_score")
    movement_log = _improvement(pooled, "large_remaining_move", "M2", "M1", "log_loss")
    direction_brier = _improvement(pooled, "up_given_large_move", "M2", "M1", "brier_score")
    direction_log = _improvement(pooled, "up_given_large_move", "M2", "M1", "log_loss")
    movement_checkpoints = _checkpoint_improvements(checkpoint, "large_remaining_move", "M2", "M1")
    direction_checkpoints = _checkpoint_improvements(checkpoint, "up_given_large_move", "M2", "M1")
    movement_null = _null_summary(null, "movement_M2_minus_M1_brier_improvement")
    direction_null = _null_summary(null, "direction_M2_minus_M1_brier_improvement")
    concentration_passes = bool(support["primary_support_passes"])
    movement_evidence = {
        "brier_improvement": movement_brier,
        "log_loss_improvement": movement_log,
        "bootstrap_90_lower_brier": _bootstrap_lower(
            bootstrap, "movement_M2_minus_M1_brier_improvement"
        ),
        "bootstrap_90_lower_log_loss": _bootstrap_lower(
            bootstrap, "movement_M2_minus_M1_log_loss_improvement"
        ),
        "positive_months": _positive_month_count(monthly, "large_remaining_move", "M2", "M1"),
        "structural_null_percentile": float(movement_null["real_percentile_under_null"]),
        "structural_null_90th_percentile": float(movement_null["null_90th_percentile"]),
        "checkpoint_brier_improvements": movement_checkpoints,
    }
    movement_passes = bool(
        movement_evidence["brier_improvement"] > 0.0
        and movement_evidence["log_loss_improvement"] > 0.0
        and movement_evidence["bootstrap_90_lower_brier"] >= 0.0
        and movement_evidence["bootstrap_90_lower_log_loss"] >= 0.0
        and movement_evidence["positive_months"] >= 5
        and movement_evidence["structural_null_percentile"] > 0.90
        and min(movement_checkpoints.values()) >= -0.001
        and concentration_passes
    )
    direction_evidence = {
        "brier_improvement": direction_brier,
        "log_loss_improvement": direction_log,
        "auc_change": _metric_lookup(pooled, "up_given_large_move", "M2", "auc")
        - _metric_lookup(pooled, "up_given_large_move", "M1", "auc"),
        "bootstrap_90_lower_brier": _bootstrap_lower(
            bootstrap, "direction_M2_minus_M1_brier_improvement"
        ),
        "bootstrap_90_lower_log_loss": _bootstrap_lower(
            bootstrap, "direction_M2_minus_M1_log_loss_improvement"
        ),
        "positive_months": _positive_month_count(monthly, "up_given_large_move", "M2", "M1"),
        "structural_null_percentile": float(direction_null["real_percentile_under_null"]),
        "structural_null_90th_percentile": float(direction_null["null_90th_percentile"]),
        "checkpoint_brier_improvements": direction_checkpoints,
    }
    direction_passes = bool(
        direction_evidence["brier_improvement"] > 0.0
        and direction_evidence["log_loss_improvement"] > 0.0
        and direction_evidence["auc_change"] >= 0.0
        and direction_evidence["bootstrap_90_lower_brier"] >= 0.0
        and direction_evidence["bootstrap_90_lower_log_loss"] >= 0.0
        and direction_evidence["positive_months"] >= 5
        and direction_evidence["structural_null_percentile"] > 0.90
        and min(direction_checkpoints.values()) >= -0.001
        and concentration_passes
    )

    interaction_targets: dict[str, Any] = {}
    interaction_passes = False
    for target, label in (
        ("large_remaining_move", "movement"),
        ("up_given_large_move", "direction"),
    ):
        brier = _improvement(pooled, target, "M3", "M2", "brier_score")
        log_increment = _improvement(pooled, target, "M3", "M2", "log_loss")
        null_row = _null_summary(null, f"{label}_M3_minus_M2_brier_improvement")
        evidence = {
            "brier_improvement": brier,
            "log_loss_improvement": log_increment,
            "bootstrap_90_lower_brier": _bootstrap_lower(
                bootstrap, f"{label}_M3_minus_M2_brier_improvement"
            ),
            "bootstrap_90_lower_log_loss": _bootstrap_lower(
                bootstrap, f"{label}_M3_minus_M2_log_loss_improvement"
            ),
            "positive_months": _positive_month_count(monthly, target, "M3", "M2"),
            "structural_null_percentile": float(null_row["real_percentile_under_null"]),
        }
        evidence["passes"] = bool(
            brier > 0.0
            and log_increment > 0.0
            and evidence["bootstrap_90_lower_brier"] >= 0.0
            and evidence["bootstrap_90_lower_log_loss"] >= 0.0
            and evidence["structural_null_percentile"] > 0.90
            and evidence["positive_months"] >= 5
            and support["interaction_support_status"] == "sufficient"
        )
        interaction_targets[label] = evidence
        interaction_passes |= bool(evidence["passes"])

    blocker = None
    if not support["primary_support_passes"]:
        blocker = "blocked_insufficient_opening_path_support"
    evidence_bundle = {
        "movement_increment_passes": movement_passes,
        "direction_increment_passes": direction_passes,
        "interaction_increment_passes": interaction_passes,
        "integrity_blocker": blocker,
        "movement": movement_evidence,
        "direction": direction_evidence,
        "interaction": interaction_targets,
        "support": dict(support),
    }
    return {
        **SAFETY_FLAGS,
        "decision": decide_screen(evidence_bundle),
        "decision_status": "provisional_pending_exact_rerun_and_independent_audit",
        "evidence": evidence_bundle,
        "exact_rerun_passed": False,
        "independent_audit_passed": False,
        "economic_reference_can_rescue_probability_failure": False,
    }


def feature_manifest() -> dict[str, Any]:
    definitions = {
        "opening_gap_bps": "10000 * (regular_session_open / prior_regular_session_close - 1)",
        "open_to_decision_raw_return_bps": (
            "10000 * (completed_decision_close / regular_session_open - 1)"
        ),
        "open_to_decision_cohort_relative_return_bps": (
            "open_to_decision_raw_return_bps minus leave-one-stock-out simultaneous-slate median"
        ),
        "latest_one_bar_return_bps": "10000 * (decision_close / prior_completed_bar_close - 1)",
        "latest_three_bar_return_bps": (
            "10000 * (decision_close / completed_close_three ordinals earlier - 1)"
        ),
        "opening_realized_volatility_bps": (
            "population standard deviation of completed one-step opening close returns times 10000"
        ),
        "opening_high_low_range_bps": (
            "completed opening high-low range divided by session open times 10000"
        ),
        "current_true_range_bps": (
            "decision-bar true range divided by previous completed close times 10000"
        ),
        "mean_completed_bar_true_range_bps": (
            "mean completed opening true range divided by each prior close times 10000"
        ),
        "close_location_within_opening_range": (
            "(decision close - opening low) / (opening high - opening low)"
        ),
        "distance_from_opening_high_bps": "10000 * (opening high - decision close) / opening high",
        "distance_from_opening_low_bps": "10000 * (decision close - opening low) / opening low",
        "positive_close_fraction": "fraction of completed opening bars with close above open",
        "directional_close_persistence_ratio": (
            "absolute signed sum divided by absolute sum of completed one-step opening returns"
        ),
        "historical_activity_proxy_shock": (
            "log1p completed cumulative provider activity divided by causal prior "
            "same-stock same-ordinal expanding mean"
        ),
        "cross_sectional_dispersion_bps": (
            "simultaneous-slate sample standard deviation of open-to-decision raw returns in bps"
        ),
    }
    return {
        **SAFETY_FLAGS,
        "feature_availability": "completed decision bar only",
        "provider_volume_label": "historical_activity_proxy",
        "observable_features": list(OBSERVABLE_FEATURES),
        "observable_feature_count": len(OBSERVABLE_FEATURES),
        "observable_definitions": definitions,
        "current_regime_features": list(CURRENT_REGIME_FEATURES),
        "opening_path_topology_features": list(TOPOLOGY_FEATURES),
        "interaction_features": list(INTERACTION_FEATURES),
        "interaction_families": [
            "current_state_one_hot_x_any_short_closure",
            "current_state_one_hot_x_opening_return_to_origin_count",
            "current_state_one_hot_x_transition_rate",
            "current_state_one_hot_x_current_state_age",
        ],
        "model_features": {key: list(value) for key, value in MODEL_FEATURES.items()},
        "forbidden": [
            "future_state",
            "future_closure",
            "future_run_duration",
            "future_loop",
            "payoff_history",
            "profitable_loop_labels",
            "exact_historical_economic_loop_scores",
            "outcome_selected_state_identity",
            "exact_arbitrary_state_string_search",
        ],
    }


def _make_plots(
    output_dir: Path,
    calibration: pd.DataFrame,
    economic: pd.DataFrame,
) -> None:
    plot_specs = (
        ("large_remaining_move", "movement_calibration_by_model.png", "Large remaining move"),
        ("up_given_large_move", "direction_calibration_by_model.png", "Up, given large move"),
    )
    for target, filename, title in plot_specs:
        figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        subset = calibration.loc[
            calibration["target"].eq(target)
            & calibration["scope"].eq("pooled")
            & calibration["row_count"].gt(0)
        ]
        for model_name in ("M0", "M1", "M2", "M3"):
            model = subset.loc[subset["model"].eq(model_name)]
            axis.plot(
                model["mean_predicted_probability"],
                model["observed_rate"],
                marker="o",
                linewidth=1.5,
                label=model_name,
            )
        axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
        axis.set(xlabel="Mean predicted probability", ylabel="Observed rate", title=title)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend(frameon=False, ncol=2)
        figure.savefig(output_dir / filename, dpi=140, metadata={"Software": "Stocker research"})
        plt.close(figure)

    subset = economic.loc[
        economic["record_type"].eq("candidate_summary") & economic["candidate"].isin(["M1", "M3"])
    ].copy()
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    x = np.arange(3)
    width = 0.34
    for offset, model_name in ((-width / 2, "M1"), (width / 2, "M3")):
        model = subset.loc[subset["candidate"].eq(model_name)].sort_values("friction_bps")
        axis.bar(
            x + offset,
            model["mean_cohort_relative_remaining_return_bps"],
            width,
            label=model_name,
        )
    axis.axhline(0.0, color="#777777", linewidth=1)
    axis.set_xticks(x, ["0 bps", "10 bps", "20 bps"])
    axis.set(
        xlabel="Synthetic friction",
        ylabel="Mean cohort-relative remaining return (bps)",
        title="Top-one delayed economic reference",
    )
    axis.legend(frameon=False)
    figure.savefig(
        output_dir / "M1_vs_M3_top_one_economic_reference.png",
        dpi=140,
        metadata={"Software": "Stocker research"},
    )
    plt.close(figure)


def _markdown_metric_table(frame: pd.DataFrame, target: str) -> str:
    subset = frame.loc[frame["target"].eq(target)].copy()
    lines = ["| Model | Brier | Log loss | AUC |", "|---|---:|---:|---:|"]
    for row in subset.sort_values("model").itertuples():
        lines.append(
            f"| {row.model} | {row.brier_score:.6f} | {row.log_loss:.6f} | {row.auc:.6f} |"
        )
    return "\n".join(lines)


def render_report(
    decision: Mapping[str, Any],
    panel: pd.DataFrame,
    pooled: pd.DataFrame,
    checkpoint: pd.DataFrame,
    monthly: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    economic: pd.DataFrame,
    support: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> str:
    development = panel.loc[panel["year"].eq(2024)]
    assessment = panel.loc[panel["year"].eq(2025)]
    movement = decision["evidence"]["movement"]
    direction = decision["evidence"]["direction"]
    interaction = decision["evidence"]["interaction"]
    m3_economic = economic.loc[
        economic["record_type"].eq("candidate_summary")
        & economic["candidate"].eq("M3")
        & economic["friction_bps"].eq(0.0)
    ].iloc[0]
    paired = economic.loc[
        economic["record_type"].eq("candidate_minus_M1_paired") & economic["candidate"].eq("M3")
    ].iloc[0]
    checkpoint_lines: list[str] = []
    for target in PRIMARY_TARGETS:
        label = "Movement" if target == "large_remaining_move" else "Direction among large moves"
        checkpoint_lines.append(f"### {label}\n")
        for ordinal in DECISION_ORDINALS:
            subset = checkpoint.loc[
                checkpoint["scope"].eq(f"checkpoint_{ordinal}") & checkpoint["target"].eq(target)
            ]
            checkpoint_lines.append(
                f"- {ordinal * 5}-minute checkpoint: M2-minus-M1 Brier improvement "
                f"{_improvement(subset, target, 'M2', 'M1', 'brier_score'):.6f}; "
                f"M3-minus-M2 {_improvement(subset, target, 'M3', 'M2', 'brier_score'):.6f}."
            )
    month_lines: list[str] = []
    for target in PRIMARY_TARGETS:
        month_lines.append(
            f"- {target}: M2 beat M1 on Brier in "
            f"{_positive_month_count(monthly, target, 'M2', 'M1')} represented months; "
            f"M3 beat M2 in {_positive_month_count(monthly, target, 'M3', 'M2')}."
        )
    return f"""# Opening Regime-Path Direction Screen V0 report

## Boundary and interpretation

This is a retrospective, research-only, representation-specific feasibility
screen. It is not prospective validation, a strategy, achieved P&L, or evidence
of executable net edge. The economic-reference calculation is delayed, gross,
and secondary; it cannot rescue failed probability gates.

## Population

- Development: {len(development)} rows, {development["session"].nunique()} sessions,
  {development["symbol"].nunique()} stocks, 2024-01-01 through 2024-12-31.
- Assessment: {len(assessment)} rows, {assessment["session"].nunique()} sessions,
  {assessment["symbol"].nunique()} stocks, 2025-01-01 through 2025-08-22.
- Development q75 movement thresholds: 30-minute {float(thresholds["6"]):.6f} bps;
  60-minute {float(thresholds["12"]):.6f} bps.
- Assessment large moves: {support["actual_large_remaining_moves"]}; opening short
  closure rows: {support["opening_short_closure_rows"]}.
- Protected market rows materialised: 0.

## Pooled movement models

{_markdown_metric_table(pooled, "large_remaining_move")}

## Pooled direction models among actual large moves

{_markdown_metric_table(pooled, "up_given_large_move")}

## Additive and interaction increments

- Movement M2-minus-M1: Brier {movement["brier_improvement"]:.6f}; log loss
  {movement["log_loss_improvement"]:.6f}; bootstrap 90% lower bounds
  {movement["bootstrap_90_lower_brier"]:.6f} and
  {movement["bootstrap_90_lower_log_loss"]:.6f}; structural-null percentile
  {movement["structural_null_percentile"]:.3f}.
- Direction M2-minus-M1: Brier {direction["brier_improvement"]:.6f}; log loss
  {direction["log_loss_improvement"]:.6f}; AUC change {direction["auc_change"]:.6f};
  structural-null percentile {direction["structural_null_percentile"]:.3f}.
- Movement M3-minus-M2: Brier
  {interaction["movement"]["brier_improvement"]:.6f}; log loss
  {interaction["movement"]["log_loss_improvement"]:.6f}.
- Direction M3-minus-M2: Brier
  {interaction["direction"]["brier_improvement"]:.6f}; log loss
  {interaction["direction"]["log_loss_improvement"]:.6f}.

## Checkpoints

{chr(10).join(checkpoint_lines)}

## Monthly stability

{chr(10).join(month_lines)}

## Delayed economic-reference diagnostic

- M3 top-one selected {int(m3_economic["selected_slate_count"])} slates; mean gross
  cohort-relative remaining return at zero synthetic friction was
  {m3_economic["mean_cohort_relative_remaining_return_bps"]:.6f} bps.
- Paired M3-minus-M1 top-one cohort-relative result was
  {paired["paired_mean_cohort_relative_difference_vs_M1_bps"]:.6f} bps per slate.

## Concentration and decision

- Maximum stock row share: {support["maximum_stock_row_share"]:.6f}.
- Maximum current-state row share: {support["maximum_current_state_row_share"]:.6f}.
- Interaction support: `{support["interaction_support_status"]}`.
- Final category: `{decision["decision"]}`.
- Exact rerun: {"passed" if decision["exact_rerun_passed"] else "pending or failed"}.
- Independent audit: {"passed" if decision["independent_audit_passed"] else "pending or failed"}.
"""


def _write_static_manifests(
    output_dir: Path,
    contract: Mapping[str, Any],
    provider_root: Path,
    development_sources: Mapping[str, Any],
    source_months: pd.DataFrame,
    safe_hashes: Mapping[str, str],
    safe_counts: Mapping[str, int],
    qa_rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    panel: pd.DataFrame,
    posterior_reproduction: Mapping[str, Any],
) -> None:
    write_json(output_dir / "contract.json", contract)
    required_hashes = []
    for path in _required_inputs():
        required_hashes.append(
            {
                "repository_relative_path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_json(
        output_dir / "input_artifact_hashes.json",
        {
            **SAFETY_FLAGS,
            "artifacts": required_hashes,
            "frozen_state_model_hash": EXPECTED_MODEL_HASH,
            "posterior_reproduction": dict(posterior_reproduction),
        },
    )
    sources = []
    for symbol in (*REGIME_CONTEXT_SYMBOLS, "VTI"):
        sources.append(
            {
                "symbol": symbol,
                "logical_path": logical_source_path(symbol),
                "development_bounded_hash": development_sources["hashes"][symbol],
                "development_bounded_rows": development_sources["row_counts"][symbol],
                "complete_safe_bounded_hash": safe_hashes[symbol],
                "complete_safe_bounded_rows": safe_counts[symbol],
            }
        )
    write_json(
        output_dir / "source_manifest.json",
        {
            **SAFETY_FLAGS,
            "provider": "EODHD five-minute historical activity proxy",
            "symbol_predicate_applied_before_materialisation": True,
            "date_predicate_applied_before_materialisation": True,
            "source_paths_are_logical_not_local_absolute": True,
            "sources": sources,
            "vendor_qa": list(qa_rows),
            "source_rows_by_year_month": source_months.to_dict(orient="records"),
            "combined_safe_snapshot_hash": context["combined_snapshot_hash"],
            "combined_safe_panel_feature_hash": context["combined_panel_feature_hash"],
        },
    )
    compact_months = (
        panel.groupby(["year", "year_month"], sort=True).size().rename("row_count").reset_index()
    )
    write_json(
        output_dir / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "read_start": str(START),
            "read_end_inclusive": str(READ_END_INCLUSIVE),
            "protected_start": str(PROTECTED_START),
            "minimum_timestamp_read": context["panel_minimum_timestamp"],
            "maximum_timestamp_read": context["panel_maximum_timestamp"],
            "source_rows_by_year_month": source_months.to_dict(orient="records"),
            "compact_rows_by_year_month": compact_months.to_dict(orient="records"),
            "protected_files_touched": [],
            "protected_rows_opened": 0,
            "protected_rows_materialised": context["protected_rows_materialised"],
            "passed": context["protected_rows_materialised"] == 0,
        },
    )
    write_json(output_dir / "feature_manifest.json", feature_manifest())


def run_screen(output_dir: Path, *, provider_root: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _load_contract()
    missing = [path for path in _required_inputs() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {[path.name for path in missing]}")
    preprocessing, parameters = _load_frozen_model()
    posterior_reproduction = reproduce_frozen_posterior(preprocessing, parameters)
    development_sources = _verify_development_sources(provider_root)
    source_months, safe_hashes, safe_counts = _source_boundary_audit(provider_root)
    qa_rows = _qa_manifest()
    panel, state_ledger, context = build_compact_panel(provider_root, preprocessing, parameters)
    if context["protected_rows_materialised"] != 0:
        raise RuntimeError("protected boundary failure")
    development = panel.loc[panel["year"].eq(2024)].reset_index(drop=True)
    assessment = panel.loc[panel["year"].eq(2025)].reset_index(drop=True)
    if len(panel) > MAX_COMPACT_ROWS:
        raise RuntimeError("compact row resource limit exceeded")
    models, coefficients = fit_model_ladder(development)
    development_scored = score_models(development, models)
    assessment_scored = score_models(assessment, models)
    movement_scale, scale_manifest = movement_scale_reference(development_scored, assessment_scored)
    assessment_scored = add_signed_scores(assessment_scored, movement_scale)
    movement, direction, monthly, checkpoint, calibration = evaluate_predictions(assessment_scored)
    pooled = pd.concat([movement, direction], ignore_index=True)
    selection = economic_selection_ledger(assessment_scored)
    economic = economic_reference_metrics(selection)
    bootstrap = bootstrap_metrics(assessment_scored, selection)
    null = structure_null_metrics(development, assessment_scored, pooled)
    concentration, support = concentration_and_support(assessment_scored)
    decision = decide_from_artifacts(pooled, monthly, checkpoint, bootstrap, null, support)

    _write_static_manifests(
        output_dir,
        contract,
        provider_root,
        development_sources,
        source_months,
        safe_hashes,
        safe_counts,
        qa_rows,
        context,
        panel,
        posterior_reproduction,
    )
    write_parquet(output_dir / "opening_decision_panel.parquet", panel)
    write_parquet(output_dir / "opening_state_path_ledger.parquet", state_ledger)
    write_json(
        output_dir / "development_movement_thresholds.json",
        {
            **SAFETY_FLAGS,
            "method": "2024_checkpoint_specific_q75_absolute_cohort_relative_remaining_return",
            "thresholds_bps": context["movement_thresholds"],
        },
    )
    write_json(
        output_dir / "model_configurations.json",
        {
            **SAFETY_FLAGS,
            "models": {key: list(value) for key, value in MODEL_FEATURES.items()},
            "targets": list(TARGETS),
            "preprocessing_fit_interval": "2024_only",
            "normalization": "unweighted_2024_training_mean_and_population_standard_deviation",
            "model_row_weight": "1 / valid_slate_size",
            "logistic": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 250,
                "class_weight": None,
                "n_jobs": 1,
            },
            "movement_scale_reference": scale_manifest,
        },
    )
    write_json(output_dir / "model_coefficients.json", {**SAFETY_FLAGS, "models": coefficients})
    prediction_columns = [
        "symbol",
        "session",
        "year_month",
        "decision_ordinal",
        "repo_bar_start_ordinal",
        "slate_id",
        "feature_available_timestamp_utc",
        "entry_bar_ordinal",
        "delayed_entry_open",
        "terminal_bar_ordinal",
        "terminal_close",
        "raw_remaining_return_bps",
        "cohort_median_return_minus_i_bps",
        "residual_remaining_return_bps",
        "movement_threshold_bps",
        *TARGETS,
        *MODEL_FEATURES["M3"],
        *(f"p__{target}__{model}" for target in TARGETS for model in ("M0", "M1", "M2", "M3")),
        "predicted_remaining_movement_scale_bps",
        *(f"expected_signed_remaining_move_score__{model}" for model in ("M0", "M1", "M2", "M3")),
    ]
    prediction_columns = list(dict.fromkeys(prediction_columns))
    write_parquet(
        output_dir / "assessment_predictions.parquet",
        assessment_scored.loc[:, prediction_columns],
    )
    write_csv(output_dir / "movement_metrics.csv", movement)
    write_csv(output_dir / "direction_metrics.csv", direction)
    write_csv(output_dir / "monthly_metrics.csv", monthly)
    write_csv(output_dir / "checkpoint_metrics.csv", checkpoint)
    write_csv(output_dir / "calibration_bins.csv", calibration)
    write_csv(output_dir / "bootstrap_metrics.csv", bootstrap)
    write_csv(output_dir / "null_metrics.csv", null)
    write_csv(output_dir / "economic_reference_metrics.csv", economic)
    write_csv(output_dir / "concentration_metrics.csv", concentration)
    write_json(output_dir / "decision.json", decision)
    _make_plots(output_dir, calibration, economic)
    report = render_report(
        decision,
        panel,
        pooled,
        checkpoint,
        monthly,
        bootstrap,
        null,
        economic,
        support,
        context["movement_thresholds"],
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    if output_dir.resolve() == DEFAULT_PRIMARY.resolve():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    return decision


SCIENTIFIC_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "input_artifact_hashes.json",
    "protected_boundary_audit.json",
    "feature_manifest.json",
    "opening_decision_panel.parquet",
    "opening_state_path_ledger.parquet",
    "development_movement_thresholds.json",
    "model_configurations.json",
    "model_coefficients.json",
    "assessment_predictions.parquet",
    "movement_metrics.csv",
    "direction_metrics.csv",
    "monthly_metrics.csv",
    "checkpoint_metrics.csv",
    "calibration_bins.csv",
    "bootstrap_metrics.csv",
    "null_metrics.csv",
    "economic_reference_metrics.csv",
    "concentration_metrics.csv",
    "movement_calibration_by_model.png",
    "direction_calibration_by_model.png",
    "M1_vs_M3_top_one_economic_reference.png",
)


def compare_scientific_artifacts(primary: Path, exact: Path) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    passed = True
    for name in SCIENTIFIC_ARTIFACTS:
        primary_path = primary / name
        exact_path = exact / name
        primary_hash = sha256_file(primary_path) if primary_path.is_file() else "missing"
        exact_hash = sha256_file(exact_path) if exact_path.is_file() else "missing"
        byte_identical = primary_hash == exact_hash and primary_hash != "missing"
        strict_numerical_identical = False
        if not byte_identical and name.endswith(".parquet"):
            try:
                pd.testing.assert_frame_equal(
                    pd.read_parquet(primary_path),
                    pd.read_parquet(exact_path),
                    check_exact=True,
                    check_like=False,
                )
                strict_numerical_identical = True
            except (AssertionError, FileNotFoundError):
                strict_numerical_identical = False
        identical = byte_identical or strict_numerical_identical
        passed &= identical
        comparisons.append(
            {
                "artifact": name,
                "primary_sha256": primary_hash,
                "exact_rerun_sha256": exact_hash,
                "byte_identical": byte_identical,
                "strict_numerical_identical": strict_numerical_identical,
                "identical": identical,
            }
        )
    return {
        **SAFETY_FLAGS,
        "status": "passed" if passed else "failed",
        "all_scientific_artifacts_identical": passed,
        "artifact_count": len(comparisons),
        "comparisons": comparisons,
    }


def _run_auditor(artifacts: Path, provider_root: Path) -> bool:
    completed = subprocess.run(
        [
            sys.executable,
            str(AUDITOR_PATH),
            "--artifacts",
            str(artifacts),
            "--provider-root",
            str(provider_root),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    audit_path = artifacts / "independent_audit.json"
    if completed.returncode != 0 or not audit_path.is_file():
        return False
    return bool(json.loads(audit_path.read_text(encoding="utf-8")).get("passed"))


def finalize_decision(
    output_dir: Path,
    *,
    exact_rerun_passed: bool,
    independent_audit_passed: bool,
) -> dict[str, Any]:
    decision_path = output_dir / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if exact_rerun_passed and independent_audit_passed:
        decision["decision"] = decide_screen(decision["evidence"])
    else:
        decision["decision"] = "blocked_reproducibility_or_audit_failure"
    decision["decision_status"] = "final"
    decision["exact_rerun_passed"] = exact_rerun_passed
    decision["independent_audit_passed"] = independent_audit_passed
    write_json(decision_path, decision)
    panel = pd.read_parquet(output_dir / "opening_decision_panel.parquet")
    movement = pd.read_csv(output_dir / "movement_metrics.csv")
    direction = pd.read_csv(output_dir / "direction_metrics.csv")
    pooled = pd.concat([movement, direction], ignore_index=True)
    checkpoint = pd.read_csv(output_dir / "checkpoint_metrics.csv")
    monthly = pd.read_csv(output_dir / "monthly_metrics.csv")
    bootstrap = pd.read_csv(output_dir / "bootstrap_metrics.csv")
    null = pd.read_csv(output_dir / "null_metrics.csv")
    economic = pd.read_csv(output_dir / "economic_reference_metrics.csv")
    thresholds = json.loads(
        (output_dir / "development_movement_thresholds.json").read_text(encoding="utf-8")
    )["thresholds_bps"]
    concentration = pd.read_csv(output_dir / "concentration_metrics.csv")
    assessment = panel.loc[panel["year"].eq(2025)]
    _, support = concentration_and_support(assessment)
    report = render_report(
        decision,
        panel,
        pooled,
        checkpoint,
        monthly,
        bootstrap,
        null,
        economic,
        support,
        thresholds,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    if output_dir.resolve() == DEFAULT_PRIMARY.resolve():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "report.md").write_text(report, encoding="utf-8")
    del concentration
    return cast(dict[str, Any], decision)


def run_complete_screen(
    *,
    primary: Path,
    exact: Path,
    provider_root: Path,
) -> dict[str, Any]:
    if not all((primary / name).is_file() for name in SCIENTIFIC_ARTIFACTS):
        run_screen(primary, provider_root=provider_root)
    run_screen(exact, provider_root=provider_root)
    comparison = compare_scientific_artifacts(primary, exact)
    exact_passed = bool(comparison["all_scientific_artifacts_identical"])
    existing_primary_audit = primary / "independent_audit.json"
    primary_audit = bool(
        exact_passed
        and existing_primary_audit.is_file()
        and json.loads(existing_primary_audit.read_text(encoding="utf-8")).get("passed")
    )
    if exact_passed and not primary_audit:
        primary_audit = _run_auditor(primary, provider_root)
    exact_audit = _run_auditor(exact, provider_root) if exact_passed else False
    audits_passed = primary_audit and exact_audit
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
    final_comparisons: list[dict[str, Any]] = []
    for name in ("decision.json", "report.md", "independent_audit.json"):
        primary_path = primary / name
        exact_path = exact / name
        primary_hash = sha256_file(primary_path) if primary_path.is_file() else "missing"
        exact_hash = sha256_file(exact_path) if exact_path.is_file() else "missing"
        final_comparisons.append(
            {
                "artifact": name,
                "primary_sha256": primary_hash,
                "exact_rerun_sha256": exact_hash,
                "identical": primary_hash == exact_hash and primary_hash != "missing",
            }
        )
    manifest = {
        **comparison,
        "primary_independent_audit_passed": primary_audit,
        "exact_rerun_independent_audit_passed": exact_audit,
        "final_primary_decision": primary_decision["decision"],
        "final_exact_rerun_decision": exact_decision["decision"],
        "decisions_identical": primary_decision == exact_decision,
        "final_artifact_comparisons": final_comparisons,
        "all_final_artifacts_identical": all(row["identical"] for row in final_comparisons),
    }
    write_json(primary / "exact_rerun_manifest.json", manifest)
    write_json(exact / "exact_rerun_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--primary-only", action="store_true")
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
    primary = args.output.resolve()
    if args.primary_only:
        decision = run_screen(primary, provider_root=provider_root)
        print(canonical_json(decision), end="")
        return
    exact = DEFAULT_EXACT if primary == DEFAULT_PRIMARY else primary.parent / "exact_rerun"
    manifest = run_complete_screen(
        primary=primary,
        exact=exact,
        provider_root=provider_root,
    )
    print(canonical_json(manifest), end="")
    if not manifest["all_scientific_artifacts_identical"]:
        raise SystemExit(2)
    if not (
        manifest["primary_independent_audit_passed"]
        and manifest["exact_rerun_independent_audit_passed"]
    ):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
