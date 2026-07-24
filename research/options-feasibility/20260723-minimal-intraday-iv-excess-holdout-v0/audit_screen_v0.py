#!/usr/bin/env python3
"""Independently audit the frozen experiment and its fail-closed resource stop."""

from __future__ import annotations

# ruff: noqa: E402 -- deterministic numerical limits must precede imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts/primary"
REPORTS = EXPERIMENT_DIR / "reports"
ATTRIBUTION_PRIMARY = (
    REPO_ROOT
    / "research/options-feasibility/20260723-stock-layer-iv-excess-attribution-v0"
    / "artifacts/primary"
)
DENSE_PANEL = (
    REPO_ROOT
    / "research/route-competition/20260722-broad-conflict-advance-hazard-v02"
    / "artifacts/primary/dense_advance_panel.parquet"
)
STOCK_CACHE = (
    REPO_ROOT
    / "data/cache/minimal-intraday-iv-excess-holdout-v0"
    / "frozen_h0_stock_surface.parquet"
)
STATE_CACHE = STOCK_CACHE.with_name("frozen_state_surface.parquet")
OPTIONS_CACHE = (
    REPO_ROOT
    / "data/vendor/eodhd/options/minimal-intraday-iv-excess-holdout-v0"
    / "canonical/exact_holdout_options.parquet"
)

for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.minimal_intraday_iv_excess_holdout_v0 import (
    BOOTSTRAP_SEED,
    EXCLUDED_FEATURES,
    GROUP_I,
    GROUP_O,
    HORIZONS,
    NULL_SEEDS,
    SAFETY_FLAGS,
    TARGET_COLUMN,
    ModelGateInputs,
    TailGateInputs,
    assert_safety_flags,
    build_group_i,
    build_group_o,
    decide_experiment,
    fit_minimal_models,
    frozen_tail_membership,
    model_increment,
    model_metric_row,
    model_specification,
    movement_timing_metrics,
    select_minimal_front_options_surface,
    tail_comparison_metrics,
    tail_metrics,
    validate_exact_previous_session_options,
)
from stocker_research.stock_options_cross_market_quick_v0 import manual_model_prediction

EXPECTED_HISTORICAL_PANEL_SHA256 = (
    "f62ef0144c12c813cbc665ba6d5ba1a235a6f77101a04b9f491c77b24c295529"
)
EXPECTED_DENSE_PANEL_SHA256 = "a916b792e15e8630dadc09bed64d71be5533ce9f3b2bd93af06605d0faaa0cc3"
EXPECTED_STOCK_CACHE_SHA256 = "d81655b54a5c5e2e8b2d324e2e6520716700965612404202793ec1b3d9b0e846"
EXPECTED_STATE_CACHE_SHA256 = "68b1cc53c1570d53054d685966eef96f533d8760368ebfc148766bb8f3a6bcc0"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    data = np.asarray(values, dtype=float)
    mass = np.asarray(weights, dtype=float)
    order = np.argsort(data, kind="mergesort")
    data = data[order]
    mass = mass[order]
    positions = (np.cumsum(mass) - 0.5 * mass) / mass.sum()
    return float(np.interp(quantile, positions, data, left=data[0], right=data[-1]))


def maximum_difference(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != right.shape:
        return math.inf
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def historical_source() -> Path:
    source = read_json(ATTRIBUTION_PRIMARY / "source_manifest.json")
    path = Path(str(cast(Mapping[str, Any], source["sources"])["frozen_branch_c_panel"]))
    if not path.is_file() or sha256_file(path) != EXPECTED_HISTORICAL_PANEL_SHA256:
        raise ValueError("historical Branch C panel is missing or drifted")
    return path


def audit_historical_models() -> dict[str, Any]:
    historical_path = historical_source()
    historical = pd.read_parquet(historical_path)
    development = historical.loc[historical["period"].astype(str).eq("development")].copy()
    reference = historical.loc[historical["period"].astype(str).eq("assessment")].copy()
    models = fit_minimal_models(development)
    stored = read_json(PRIMARY / "model_coefficients.json")
    stored_m0 = cast(Mapping[str, Any], stored["M0"])
    stored_m1 = cast(Mapping[str, Any], stored["M1"])
    independent_m0 = model_specification(models.m0)
    independent_m1 = model_specification(models.m1)
    coefficient_difference = max(
        maximum_difference(
            cast(Sequence[float], stored_m0["coefficients"]),
            cast(Sequence[float], independent_m0["coefficients"]),
        ),
        maximum_difference(
            cast(Sequence[float], stored_m1["coefficients"]),
            cast(Sequence[float], independent_m1["coefficients"]),
        ),
        abs(float(stored_m0["intercept"]) - float(independent_m0["intercept"])),
        abs(float(stored_m1["intercept"]) - float(independent_m1["intercept"])),
    )
    development_m0 = models.m0.predict(development)
    development_m1 = models.m1.predict(development)
    thresholds = read_json(PRIMARY / "frozen_tail_thresholds.json")
    threshold_m0 = independent_weighted_quantile(
        development_m0,
        development["row_weight"].to_numpy(float),
        0.95,
    )
    threshold_m1 = independent_weighted_quantile(
        development_m1,
        development["row_weight"].to_numpy(float),
        0.95,
    )
    threshold_difference = max(
        abs(threshold_m0 - float(thresholds["M0_top_5_percent_threshold"])),
        abs(threshold_m1 - float(thresholds["M1_top_5_percent_threshold"])),
    )
    sample = reference.sort_values("row_id", kind="mergesort").head(100)
    manual_m0 = manual_model_prediction(sample, stored_m0)
    manual_m1 = manual_model_prediction(sample, stored_m1)
    manual_probability_difference = max(
        maximum_difference(manual_m0, models.m0.predict(sample)),
        maximum_difference(manual_m1, models.m1.predict(sample)),
    )
    predecessor = pd.read_parquet(
        ATTRIBUTION_PRIMARY / "assessment_predictions.parquet",
        columns=["row_id", "G0_probability"],
    )
    current = reference.loc[:, ["row_id"]].copy()
    current["M0_probability"] = models.m0.predict(reference)
    joined = current.merge(predecessor, on="row_id", how="outer", indicator=True)
    row_mismatches = int(joined["_merge"].ne("both").sum())
    m0_probability_difference = (
        math.inf
        if row_mismatches
        else maximum_difference(joined["M0_probability"], joined["G0_probability"])
    )
    return {
        "historical_panel_path": str(historical_path),
        "historical_panel_sha256": sha256_file(historical_path),
        "development_rows": len(development),
        "prior_reference_rows": len(reference),
        "maximum_coefficient_difference": coefficient_difference,
        "maximum_threshold_difference": threshold_difference,
        "manual_probability_rows_per_model": 100,
        "maximum_manual_probability_difference": manual_probability_difference,
        "M0_G0_row_mismatches": row_mismatches,
        "M0_G0_maximum_probability_difference": m0_probability_difference,
        "passed": bool(
            coefficient_difference <= 1e-12
            and threshold_difference <= 1e-12
            and manual_probability_difference <= 1e-12
            and row_mismatches == 0
            and m0_probability_difference <= 1e-12
        ),
    }


def audit_h0_surface() -> dict[str, Any]:
    if not STOCK_CACHE.is_file() or not STATE_CACHE.is_file():
        raise ValueError("frozen H0/state cache is missing")
    stock_cache_sha256 = sha256_file(STOCK_CACHE)
    state_cache_sha256 = sha256_file(STATE_CACHE)
    if (
        stock_cache_sha256 != EXPECTED_STOCK_CACHE_SHA256
        or state_cache_sha256 != EXPECTED_STATE_CACHE_SHA256
    ):
        raise ValueError("frozen H0/state cache provenance drifted")
    if sha256_file(DENSE_PANEL) != EXPECTED_DENSE_PANEL_SHA256:
        raise ValueError("dense predecessor panel drifted")
    current = pd.read_parquet(STOCK_CACHE)
    predecessor = pd.read_parquet(DENSE_PANEL)
    predecessor = predecessor.loc[predecessor["advance_eligible"].astype(int).eq(1)].copy()
    historical = current.loc[
        pd.to_datetime(current["session"]).le(pd.Timestamp("2025-08-22"))
    ].copy()
    first = predecessor.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    second = historical.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    row_mismatches = abs(len(first) - len(second)) + sum(
        left != right
        for left, right in zip(
            first["row_id"].astype(str),
            second["row_id"].astype(str),
            strict=False,
        )
    )
    feature_difference = math.inf
    if row_mismatches == 0:
        feature_difference = maximum_difference(
            first.loc[:, [*GROUP_I, "row_weight"]].to_numpy(float).ravel(),
            second.loc[:, [*GROUP_I, "row_weight"]].to_numpy(float).ravel(),
        )
    states = pd.read_parquet(STATE_CACHE, columns=["bar_start_timestamp"])
    maximum_state_date = pd.to_datetime(states["bar_start_timestamp"], utc=True).max()
    holdout = current.loc[current["period"].astype(str).eq("holdout")]
    return {
        "stock_cache_path": str(STOCK_CACHE),
        "stock_cache_sha256": stock_cache_sha256,
        "state_cache_path": str(STATE_CACHE),
        "state_cache_sha256": state_cache_sha256,
        "historical_row_mismatches": row_mismatches,
        "maximum_h0_feature_weight_difference": feature_difference,
        "holdout_candidate_rows_before_options": len(holdout),
        "holdout_candidate_sessions": int(holdout["session"].nunique()),
        "holdout_candidate_stocks": int(holdout["symbol"].nunique()),
        "maximum_state_timestamp": str(maximum_state_date),
        "protected_state_rows_materialised": int(
            pd.to_datetime(states["bar_start_timestamp"], utc=True)
            .ge(pd.Timestamp("2026-01-01T00:00:00Z"))
            .sum()
        ),
        "passed": bool(
            row_mismatches == 0
            and feature_difference <= 1e-12
            and maximum_state_date < pd.Timestamp("2026-01-01T00:00:00Z")
            and stock_cache_sha256 == EXPECTED_STOCK_CACHE_SHA256
            and state_cache_sha256 == EXPECTED_STATE_CACHE_SHA256
        ),
    }


def verify_blocked_artifacts() -> bool:
    """Require explicit blocker placeholders without altering primary results."""

    reason = "blocked_quick_resource_limit"
    parquet_schemas: dict[str, list[str]] = {
        "holdout_selected_option_pairs.parquet": [
            "symbol",
            "session",
            "required_options_date",
            "blocked_reason",
        ],
        "holdout_panel.parquet": [
            "row_id",
            "symbol",
            "session",
            "checkpoint",
            "blocked_reason",
        ],
        "holdout_predictions.parquet": [
            "row_id",
            "M0_probability",
            "M1_probability",
            "blocked_reason",
        ],
    }
    for name, columns in parquet_schemas.items():
        path = PRIMARY / name
        if not path.is_file():
            return False
        frame = pd.read_parquet(path)
        if list(frame.columns) != columns or not frame.empty:
            return False
    csv_schemas: dict[str, list[str]] = {
        "holdout_join_audit.csv": ["scope", "rows", "blocked_reason"],
        "holdout_model_metrics.csv": ["model", "metric", "value", "blocked_reason"],
        "holdout_monthly_metrics.csv": ["month", "model", "blocked_reason"],
        "holdout_checkpoint_metrics.csv": ["group", "model", "blocked_reason"],
        "tail_metrics.csv": ["model", "tail", "blocked_reason"],
        "tail_comparison_metrics.csv": ["comparison", "blocked_reason"],
        "tail_overlap_metrics.csv": ["comparison", "blocked_reason"],
        "movement_timing_metrics.csv": ["horizon_minutes", "blocked_reason"],
        "bootstrap_metrics.csv": ["record_type", "statistic", "blocked_reason"],
        "intraday_h0_null_metrics.csv": ["null_refit", "seed", "blocked_reason"],
        "concentration_metrics.csv": ["scope", "group", "blocked_reason"],
    }
    for name, columns in csv_schemas.items():
        path = PRIMARY / name
        if not path.is_file():
            return False
        frame = pd.read_csv(path)
        if list(frame.columns) != columns or len(frame) != 1:
            return False
        if str(frame.iloc[0]["blocked_reason"]) != reason:
            return False
    return True


def metric_difference(expected: Mapping[str, object], actual: Mapping[str, object]) -> float:
    """Compare a calculated metric record with its serialized counterpart."""

    difference = 0.0
    for name, expected_value in expected.items():
        if name not in actual:
            return math.inf
        actual_value = actual[name]
        if isinstance(expected_value, str):
            if str(actual_value) != expected_value:
                return math.inf
            continue
        if isinstance(expected_value, (int, float, np.integer, np.floating)):
            left = float(expected_value)
            right = float(cast(Any, actual_value))
            if not math.isfinite(left) or not math.isfinite(right):
                if not (math.isnan(left) and math.isnan(right)):
                    return math.inf
            else:
                difference = max(difference, abs(left - right))
    return difference


def audit_completed_artifacts(
    *,
    historical: Mapping[str, Any],
    h0: Mapping[str, Any],
    exact_features: bool,
    request_gate: bool,
) -> bool:
    """Audit a completed run without rewriting any scientific result artifact."""

    required_paths = (
        PRIMARY / "holdout_selected_option_pairs.parquet",
        PRIMARY / "holdout_panel.parquet",
        PRIMARY / "holdout_predictions.parquet",
        PRIMARY / "holdout_model_metrics.csv",
        PRIMARY / "tail_metrics.csv",
        PRIMARY / "movement_timing_metrics.csv",
        PRIMARY / "bootstrap_metrics.csv",
        PRIMARY / "intraday_h0_null_metrics.csv",
        PRIMARY / "decision.json",
    )
    if not all(path.is_file() for path in required_paths) or not OPTIONS_CACHE.is_file():
        raise ValueError("completed-run audit inputs are missing")

    pairs = pd.read_parquet(PRIMARY / "holdout_selected_option_pairs.parquet")
    panel = pd.read_parquet(PRIMARY / "holdout_panel.parquet")
    predictions = pd.read_parquet(PRIMARY / "holdout_predictions.parquet")
    if len(panel) < 100 or pairs.empty:
        raise ValueError("completed-run audit lacks the required holdout sample")

    session_dates = pd.to_datetime(panel["session"], errors="raise")
    date_gate = bool(
        session_dates.between("2025-09-01", "2025-12-31").all()
        and not session_dates.ge("2026-01-01").any()
    )
    chronology_gate = True
    for row in pairs[["session", "required_options_date", "options_observation_date"]].itertuples(
        index=False
    ):
        try:
            validate_exact_previous_session_options(
                signal_date=pd.Timestamp(row.session).date(),
                required_options_date=pd.Timestamp(row.required_options_date).date(),
                actual_options_date=pd.Timestamp(row.options_observation_date).date(),
            )
        except ValueError:
            chronology_gate = False
            break

    chain = pd.read_parquet(OPTIONS_CACHE)
    chain["trade_date"] = pd.to_datetime(chain["trade_date"], errors="raise").dt.date
    chain_groups = {
        (str(symbol), cast(object, observed)): group.copy()
        for (symbol, observed), group in chain.groupby(
            ["underlying_symbol", "trade_date"],
            sort=False,
            observed=True,
        )
    }
    selected_contract_mismatches = 0
    for row in pairs.itertuples(index=False):
        observed = pd.Timestamp(row.required_options_date).date()
        exact_chain = chain_groups.get((str(row.symbol), observed))
        if exact_chain is None:
            selected_contract_mismatches += 1
            continue
        realised = float(row.atm_iv) - float(row.iv_minus_realised_20d)
        selected = select_minimal_front_options_surface(
            exact_chain,
            previous_close=float(row.previous_close_underlying_price),
            realised_volatility_20d=realised,
        )
        for name in (
            "front_expiration_date",
            "front_strike",
            "front_call_contract_id",
            "front_put_contract_id",
        ):
            expected = selected.get(name)
            actual = getattr(row, name)
            if name == "front_strike":
                selected_contract_mismatches += int(
                    not math.isclose(float(cast(Any, expected)), float(actual), abs_tol=1e-12)
                )
            else:
                selected_contract_mismatches += int(str(expected) != str(actual))

    historical_frame = pd.read_parquet(historical_source())
    development = historical_frame.loc[
        historical_frame["period"].astype(str).eq("development")
    ].copy()
    models = fit_minimal_models(development)
    calculated_m0 = models.m0.predict(panel)
    calculated_m1 = models.m1.predict(panel)
    probability_difference = max(
        maximum_difference(calculated_m0, panel["M0_probability"]),
        maximum_difference(calculated_m1, panel["M1_probability"]),
    )
    stored_coefficients = read_json(PRIMARY / "model_coefficients.json")
    sample = panel.sort_values("row_id", kind="mergesort").head(100)
    manual_difference = max(
        maximum_difference(
            manual_model_prediction(
                sample,
                cast(Mapping[str, Any], stored_coefficients["M0"]),
            ),
            calculated_m0[sample.index],
        ),
        maximum_difference(
            manual_model_prediction(
                sample,
                cast(Mapping[str, Any], stored_coefficients["M1"]),
            ),
            calculated_m1[sample.index],
        ),
    )

    movement_difference = 0.0
    entry = panel["entry_price"].to_numpy(float)
    atm_iv = panel["atm_iv"].to_numpy(float)
    for horizon in HORIZONS:
        movement = np.abs(np.log(panel[f"close_{horizon}m"].to_numpy(float) / entry))
        sigma = atm_iv * math.sqrt(horizon / (252.0 * 390.0))
        expectation = sigma * math.sqrt(2.0 / math.pi)
        residual = movement - expectation
        movement_difference = max(
            movement_difference,
            maximum_difference(movement, panel[f"absolute_log_return_{horizon}m"]),
            maximum_difference(expectation, panel[f"iv_expected_absolute_{horizon}m"]),
            maximum_difference(residual, panel[f"iv_absolute_residual_{horizon}m"]),
        )
    target = (
        panel["absolute_log_return_15m"].to_numpy(float)
        > panel["iv_expected_absolute_15m"].to_numpy(float)
    ).astype(int)
    movement_difference = max(
        movement_difference,
        maximum_difference(target, panel[TARGET_COLUMN]),
    )

    thresholds = read_json(PRIMARY / "frozen_tail_thresholds.json")
    expected_m0_tail = frozen_tail_membership(
        calculated_m0,
        float(thresholds["M0_top_5_percent_threshold"]),
    )
    expected_m1_tail = frozen_tail_membership(
        calculated_m1,
        float(thresholds["M1_top_5_percent_threshold"]),
    )
    tail_membership_mismatches = int(
        np.count_nonzero(expected_m0_tail != panel["M0_top_5pct"].to_numpy(bool))
        + np.count_nonzero(expected_m1_tail != panel["M1_top_5pct"].to_numpy(bool))
    )

    base_prediction_columns = [
        "row_id",
        "symbol",
        "session",
        "checkpoint",
        "row_weight",
        "M0_probability",
        "M1_probability",
        "M0_top_5pct",
        "M1_top_5pct",
        TARGET_COLUMN,
        *(f"absolute_log_return_{horizon}m" for horizon in HORIZONS),
        *(f"iv_expected_absolute_{horizon}m" for horizon in HORIZONS),
        *(f"iv_absolute_residual_{horizon}m" for horizon in HORIZONS),
    ]
    null_evidence_columns = [
        column
        for null_index in range(3)
        for column in (
            f"M1_null_{null_index}_source_row_id",
            f"M1_null_{null_index}_probability",
        )
    ]
    expected_prediction_columns = [*base_prediction_columns, *null_evidence_columns]
    prediction_schema_gate = bool(
        list(predictions.columns) == expected_prediction_columns
        and not panel["row_id"].duplicated().any()
        and not predictions["row_id"].duplicated().any()
    )
    if not prediction_schema_gate:
        raise ValueError("holdout prediction artifact schema or row identity drifted")
    prediction_rows = panel.loc[:, base_prediction_columns].merge(
        predictions.loc[:, base_prediction_columns],
        on="row_id",
        how="outer",
        suffixes=("_panel", "_artifact"),
        indicator=True,
        validate="one_to_one",
    )
    joined_row_mismatches = int(prediction_rows["_merge"].ne("both").sum())
    prediction_artifact_difference = 0.0
    if joined_row_mismatches:
        prediction_artifact_difference = math.inf
    else:
        for name in base_prediction_columns:
            if name == "row_id":
                continue
            left = prediction_rows[f"{name}_panel"]
            right = prediction_rows[f"{name}_artifact"]
            if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_bool_dtype(left):
                prediction_artifact_difference = max(
                    prediction_artifact_difference,
                    maximum_difference(
                        pd.to_numeric(left, errors="raise"),
                        pd.to_numeric(right, errors="raise"),
                    ),
                )
            elif not left.astype(str).equals(right.astype(str)):
                prediction_artifact_difference = math.inf
                break
    h0_cache = pd.read_parquet(STOCK_CACHE, columns=["row_id", *GROUP_I])
    feature_join = panel.loc[:, ["row_id", *GROUP_I]].merge(
        h0_cache,
        on="row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_panel", "_cache"),
        indicator=True,
    )
    joined_row_mismatches += int(feature_join["_merge"].ne("both").sum())
    feature_difference = 0.0
    if joined_row_mismatches:
        feature_difference = math.inf
    else:
        for name in GROUP_I:
            feature_difference = max(
                feature_difference,
                maximum_difference(
                    feature_join[f"{name}_panel"],
                    feature_join[f"{name}_cache"],
                ),
            )
        group_o = build_group_o(panel)
        group_i = build_group_i(panel)
        feature_difference = max(
            feature_difference,
            maximum_difference(
                group_o.to_numpy(float).ravel(),
                panel.loc[:, list(GROUP_O)].to_numpy(float).ravel(),
            ),
            maximum_difference(
                group_i.to_numpy(float).ravel(),
                panel.loc[:, list(GROUP_I)].to_numpy(float).ravel(),
            ),
        )

    stored_metrics = pd.read_csv(PRIMARY / "holdout_model_metrics.csv")
    metric_difference_value = 0.0
    for model_name, probability_column in (
        ("M0", "M0_probability"),
        ("M1", "M1_probability"),
    ):
        calculated = model_metric_row(
            panel,
            model=model_name,
            probability_column=probability_column,
        )
        rows = stored_metrics.loc[stored_metrics["model"].astype(str).eq(model_name)]
        if len(rows) != 1:
            metric_difference_value = math.inf
            break
        metric_difference_value = max(
            metric_difference_value,
            metric_difference(calculated, rows.iloc[0].to_dict()),
        )

    stored_tails = pd.read_csv(PRIMARY / "tail_metrics.csv")
    tail_difference = 0.0
    for model_name, membership_column in (
        ("M0", "M0_top_5pct"),
        ("M1", "M1_top_5pct"),
    ):
        calculated_tail = tail_metrics(
            panel.loc[panel[membership_column].astype(bool)],
            model=model_name,
        )
        rows = stored_tails.loc[stored_tails["model"].astype(str).eq(model_name)]
        if len(rows) != 1:
            tail_difference = math.inf
            break
        tail_difference = max(
            tail_difference,
            metric_difference(calculated_tail, rows.iloc[0].to_dict()),
        )

    stored_timing = pd.read_csv(PRIMARY / "movement_timing_metrics.csv")
    calculated_timing = movement_timing_metrics(panel.loc[panel["M1_top_5pct"].astype(bool)])
    timing_difference = 0.0
    for row in calculated_timing.to_dict(orient="records"):
        horizon = int(cast(Any, row["horizon_minutes"]))
        stored = stored_timing.loc[
            pd.to_numeric(stored_timing["horizon_minutes"], errors="coerce").eq(horizon)
        ]
        if len(stored) != 1:
            timing_difference = math.inf
            break
        timing_difference = max(
            timing_difference,
            metric_difference(cast(Mapping[str, object], row), stored.iloc[0].to_dict()),
        )

    bootstrap = pd.read_csv(PRIMARY / "bootstrap_metrics.csv")
    draws = bootstrap.loc[bootstrap["record_type"].astype(str).eq("draw")]
    statistic_names = (
        "M1_minus_M0_log_loss_improvement",
        "M1_minus_M0_brier_improvement",
        "M1_minus_M0_auc_improvement",
        "M1_minus_M0_average_precision_improvement",
        "M1_top_5pct_mean_iv_residual",
        "M1_top_5pct_median_iv_residual",
        "M1_top_5pct_exceed_iv_rate",
        "M1_minus_M0_top_5pct_mean_iv_residual",
        "M1_minus_M0_top_5pct_median_iv_residual",
        "M1_minus_M0_top_5pct_exceed_iv_rate",
    )
    bootstrap_gate = bool(
        len(draws) == 10
        and set(pd.to_numeric(draws["seed"], errors="raise").astype(int)) == {BOOTSTRAP_SEED}
        and set(pd.to_numeric(draws["draw"], errors="raise").astype(int)) == set(range(10))
        and "session_multiplicities_json" in draws.columns
    )
    bootstrap_draw_difference = 0.0
    session_labels = panel["session"].astype(str)
    unique_sessions = set(session_labels)
    for draw in range(10):
        stored_draw = draws.loc[pd.to_numeric(draws["draw"], errors="raise").eq(draw)]
        if len(stored_draw) != 1:
            bootstrap_draw_difference = math.inf
            break
        try:
            stored_multiplicities = cast(
                dict[str, Any],
                json.loads(str(stored_draw.iloc[0]["session_multiplicities_json"])),
            )
        except (TypeError, json.JSONDecodeError):
            bootstrap_draw_difference = math.inf
            break
        if (
            set(stored_multiplicities) != unique_sessions
            or any(
                not isinstance(value, int) or value < 0 for value in stored_multiplicities.values()
            )
            or sum(int(value) for value in stored_multiplicities.values()) != len(unique_sessions)
        ):
            bootstrap_draw_difference = math.inf
            break
        multiplicity = session_labels.map(stored_multiplicities).to_numpy(int)
        selected = multiplicity > 0
        sample_frame = panel.loc[selected].copy()
        sample_frame["row_weight"] = sample_frame["row_weight"].to_numpy(float) * multiplicity[
            selected
        ].astype(float)
        sample_m0 = model_metric_row(
            sample_frame,
            model="M0",
            probability_column="M0_probability",
        )
        sample_m1 = model_metric_row(
            sample_frame,
            model="M1",
            probability_column="M1_probability",
        )
        sample_increment = model_increment(sample_m0, sample_m1)
        sample_m1_tail = tail_metrics(
            sample_frame.loc[sample_frame["M1_top_5pct"].astype(bool)],
            model="M1",
        )
        sample_m0_tail = tail_metrics(
            sample_frame.loc[sample_frame["M0_top_5pct"].astype(bool)],
            model="M0",
        )
        sample_comparison = tail_comparison_metrics(sample_m1_tail, sample_m0_tail)
        calculated_draw: dict[str, object] = {
            "record_type": "draw",
            "draw": draw,
            "seed": BOOTSTRAP_SEED,
            "session_multiplicities_json": str(stored_draw.iloc[0]["session_multiplicities_json"]),
            "M1_minus_M0_log_loss_improvement": sample_increment["log_loss_improvement"],
            "M1_minus_M0_brier_improvement": sample_increment["brier_improvement"],
            "M1_minus_M0_auc_improvement": sample_increment["auc_improvement"],
            "M1_minus_M0_average_precision_improvement": sample_increment[
                "average_precision_improvement"
            ],
            "M1_top_5pct_mean_iv_residual": sample_m1_tail["mean_iv_residual"],
            "M1_top_5pct_median_iv_residual": sample_m1_tail["median_iv_residual"],
            "M1_top_5pct_exceed_iv_rate": sample_m1_tail["exceed_iv_rate"],
            "M1_minus_M0_top_5pct_mean_iv_residual": sample_comparison[
                "mean_iv_residual_difference"
            ],
            "M1_minus_M0_top_5pct_median_iv_residual": sample_comparison[
                "median_iv_residual_difference"
            ],
            "M1_minus_M0_top_5pct_exceed_iv_rate": sample_comparison["exceed_iv_rate_difference"],
        }
        bootstrap_draw_difference = max(
            bootstrap_draw_difference,
            metric_difference(calculated_draw, stored_draw.iloc[0].to_dict()),
        )
    bootstrap_gate &= bootstrap_draw_difference <= 1e-12
    interval_rows = bootstrap.loc[bootstrap["record_type"].astype(str).eq("interval")]
    expected_interval_keys = {
        (statistic, level) for statistic in statistic_names for level in (0.80, 0.90, 0.95)
    }
    actual_interval_keys = {
        (str(row.statistic), float(row.interval_level))
        for row in interval_rows.itertuples(index=False)
    }
    bootstrap_gate &= bool(
        len(interval_rows) == len(expected_interval_keys)
        and actual_interval_keys == expected_interval_keys
    )
    for row in interval_rows.itertuples(index=False):
        statistic = str(row.statistic)
        level = float(row.interval_level)
        values = pd.to_numeric(draws[statistic], errors="raise").to_numpy(float)
        alpha = (1.0 - level) / 2.0
        bootstrap_gate &= bool(
            math.isclose(float(row.lower), float(np.quantile(values, alpha)), abs_tol=1e-12)
            and math.isclose(
                float(row.upper),
                float(np.quantile(values, 1.0 - alpha)),
                abs_tol=1e-12,
            )
        )

    nulls = pd.read_csv(PRIMARY / "intraday_h0_null_metrics.csv")
    null_gate = bool(
        len(nulls) == 3
        and tuple(pd.to_numeric(nulls["seed"], errors="raise").astype(int)) == NULL_SEEDS
        and set(pd.to_numeric(nulls["null_refit"], errors="raise").astype(int)) == set(range(3))
        and nulls["bundle"].astype(str).eq("complete_Group_I").all()
        and ~nulls["M0_refit"].astype(bool).any()
        and nulls["M1_refit"].astype(bool).all()
        and "model_specification_json" in nulls.columns
    )
    evaluation = panel.copy()
    evaluation["period"] = "holdout"
    real_m0_metrics = model_metric_row(
        evaluation,
        model="M0",
        probability_column="M0_probability",
    )
    real_m1_metrics = model_metric_row(
        evaluation,
        model="M1",
        probability_column="M1_probability",
    )
    real_increment = model_increment(real_m0_metrics, real_m1_metrics)
    null_metric_difference = 0.0
    panel_by_row_id = panel.set_index(panel["row_id"].astype(str), drop=False)
    for null_index, seed in enumerate(NULL_SEEDS):
        stored_null = nulls.loc[pd.to_numeric(nulls["null_refit"], errors="raise").eq(null_index)]
        if len(stored_null) != 1:
            null_metric_difference = math.inf
            break
        source_column = f"M1_null_{null_index}_source_row_id"
        probability_column = f"M1_null_{null_index}_probability"
        source_row_ids = predictions[source_column].astype(str)
        if not source_row_ids.isin(panel_by_row_id.index).all():
            null_metric_difference = math.inf
            break
        permutation_evidence = panel.loc[
            :,
            ["row_id", "session", "checkpoint"],
        ].copy()
        permutation_evidence["source_row_id"] = source_row_ids.to_numpy()
        for _, group in permutation_evidence.groupby(
            ["session", "checkpoint"],
            sort=True,
            observed=True,
        ):
            if set(group["row_id"].astype(str)) != set(group["source_row_id"]):
                null_metric_difference = math.inf
                break
        if not math.isfinite(null_metric_difference):
            break
        source_rows = panel_by_row_id.loc[source_row_ids]
        if (
            not (
                source_rows["session"].astype(str).to_numpy()
                == panel["session"].astype(str).to_numpy()
            ).all()
            or not (
                source_rows["checkpoint"].to_numpy(int) == panel["checkpoint"].to_numpy(int)
            ).all()
        ):
            null_metric_difference = math.inf
            break
        permuted_holdout = panel.copy()
        permuted_holdout.loc[:, list(GROUP_I)] = source_rows.loc[
            :,
            list(GROUP_I),
        ].to_numpy(float)
        try:
            stored_model = cast(
                dict[str, Any],
                json.loads(str(stored_null.iloc[0]["model_specification_json"])),
            )
        except (TypeError, json.JSONDecodeError):
            null_metric_difference = math.inf
            break
        if stored_model.get("numeric_features") != [*GROUP_O, *GROUP_I] or stored_model.get(
            "category_controls"
        ) != ["stock"]:
            null_metric_difference = math.inf
            break
        manual_null_probability = manual_model_prediction(
            permuted_holdout,
            stored_model,
        )
        null_metric_difference = max(
            null_metric_difference,
            maximum_difference(
                manual_null_probability,
                predictions[probability_column],
            ),
        )
        permuted_holdout["null_probability"] = predictions[probability_column].to_numpy(float)
        null_metrics = model_metric_row(
            permuted_holdout,
            model=f"M1_null_{null_index}",
            probability_column="null_probability",
        )
        null_increment = model_increment(real_m0_metrics, null_metrics)
        calculated_null: dict[str, object] = {
            "null_refit": null_index,
            "seed": seed,
            "bundle": "complete_Group_I",
            "strata": "training_or_holdout_x_session_x_checkpoint",
            "M0_refit": False,
            "M1_refit": True,
            **null_increment,
            "real_exceeds_null_log_loss_improvement": float(
                cast(Any, real_increment["log_loss_improvement"])
            )
            > float(cast(Any, null_increment["log_loss_improvement"])),
            "real_exceeds_null_brier_improvement": float(
                cast(Any, real_increment["brier_improvement"])
            )
            > float(cast(Any, null_increment["brier_improvement"])),
            "real_exceeds_null_auc_improvement": float(cast(Any, real_increment["auc_improvement"]))
            > float(cast(Any, null_increment["auc_improvement"])),
            "real_exceeds_null_average_precision_improvement": float(
                cast(Any, real_increment["average_precision_improvement"])
            )
            > float(cast(Any, null_increment["average_precision_improvement"])),
        }
        null_metric_difference = max(
            null_metric_difference,
            metric_difference(calculated_null, stored_null.iloc[0].to_dict()),
        )
    null_gate &= null_metric_difference <= 1e-12

    decision = read_json(PRIMARY / "decision.json")
    assert_safety_flags(decision)
    model_values = cast(Mapping[str, Any], decision["model_gate"])
    tail_values = cast(Mapping[str, Any], decision["tail_gate"])
    model_inputs = ModelGateInputs(
        **{name: model_values[name] for name in ModelGateInputs.__dataclass_fields__}
    )
    tail_inputs = TailGateInputs(
        **{name: tail_values[name] for name in TailGateInputs.__dataclass_fields__}
    )
    recalculated_decision = decide_experiment(model=model_inputs, tail=tail_inputs)
    decision_gate = all(
        decision.get(name) == recalculated_decision.get(name)
        for name in (
            "overall_decision",
            "minimal_model_status",
            "frozen_top_5pct_status",
            "options_only_tail_comparison_status",
            "movement_timing_status",
        )
    )

    passed = bool(
        historical["passed"]
        and h0["passed"]
        and exact_features
        and request_gate
        and date_gate
        and chronology_gate
        and selected_contract_mismatches == 0
        and probability_difference <= 1e-12
        and prediction_artifact_difference <= 1e-12
        and manual_difference <= 1e-12
        and movement_difference <= 1e-12
        and tail_membership_mismatches == 0
        and joined_row_mismatches == 0
        and feature_difference <= 1e-12
        and metric_difference_value <= 1e-12
        and tail_difference <= 1e-12
        and timing_difference <= 1e-12
        and bootstrap_gate
        and null_gate
        and decision_gate
    )
    write_json(
        PRIMARY / "determinism_check.json",
        {
            **SAFETY_FLAGS,
            "status": "passed" if passed else "failed",
            "selected_contract_mismatches": selected_contract_mismatches,
            "joined_row_mismatches": joined_row_mismatches,
            "maximum_feature_difference": feature_difference,
            "maximum_probability_difference": max(
                probability_difference,
                prediction_artifact_difference,
            ),
            "tail_membership_mismatches": tail_membership_mismatches,
            "maximum_movement_difference": movement_difference,
            "bootstrap_repeated": False,
            "null_draws_repeated": False,
            "passed": passed,
        },
    )
    write_json(
        PRIMARY / "lightweight_audit.json",
        {
            **SAFETY_FLAGS,
            "audit_scope": "completed_binding_holdout",
            "historical_model_audit": dict(historical),
            "historical_h0_audit": dict(h0),
            "manual_probability_rows_per_model": 100,
            "maximum_manual_probability_difference": manual_difference,
            "maximum_prediction_artifact_difference": prediction_artifact_difference,
            "maximum_metric_difference": metric_difference_value,
            "maximum_tail_metric_difference": tail_difference,
            "maximum_timing_metric_difference": timing_difference,
            "maximum_bootstrap_draw_difference": bootstrap_draw_difference,
            "maximum_intraday_h0_null_metric_difference": null_metric_difference,
            "session_bootstrap_evidence_recalculated_without_redraw": bootstrap_gate,
            "intraday_h0_null_evidence_reconstructed_without_refit": null_gate,
            "decision_logic_audited": decision_gate,
            "binding_experiment_completed": True,
            "passed": passed,
        },
    )
    return passed


def main() -> int:
    contract = read_json(EXPERIMENT_DIR / "contract.json")
    assert_safety_flags(contract)
    feature_manifest = read_json(PRIMARY / "minimal_feature_manifest.json")
    assert_safety_flags(feature_manifest)
    download = read_json(PRIMARY / "holdout_options_download_manifest.json")
    assert_safety_flags(download)
    request_plan = read_json(PRIMARY / "holdout_options_request_plan.json")
    assert_safety_flags(request_plan)
    historical = audit_historical_models()
    h0 = audit_h0_surface()
    exact_features = bool(
        cast(Mapping[str, Any], feature_manifest["models"])["M0"]["numeric_features"]
        == list(GROUP_O)
        and cast(Mapping[str, Any], feature_manifest["models"])["M1"]["numeric_features"]
        == [*GROUP_O, *GROUP_I]
        and not set((*GROUP_O, *GROUP_I)).intersection(EXCLUDED_FEATURES)
    )
    request_gate = bool(
        request_plan["requests_planned"] == 1_700
        and request_plan["compact"] is False
        and request_plan["front_dte_minimum"] == 7
        and request_plan["front_dte_maximum"] == 45
        and request_plan["back_expiry_46_to_90_requested"] is False
    )
    if download.get("status") == "complete":
        audit_passed = audit_completed_artifacts(
            historical=historical,
            h0=h0,
            exact_features=exact_features,
            request_gate=request_gate,
        )
        print(
            "independent completed-run audit passed" if audit_passed else "independent audit failed"
        )
        return 0 if audit_passed else 2

    coverage = pd.read_csv(PRIMARY / "holdout_options_coverage.csv")
    coverage_gate = bool(
        len(coverage) == 80
        and int(coverage["requests_planned"].sum()) == int(download["requests_planned"])
        and int(coverage["requests_completed"].sum()) == int(download["requests_completed"])
        and int(coverage["requests_remaining"].sum()) == int(download["requests_remaining"])
        and coverage["pair_coverage_status"].astype(str).eq("blocked_before_pair_selection").all()
        and coverage["selected_pair_sessions"].isna().all()
    )
    record_audit_gate = bool(
        int(download["records_returned"]) == int(download["maximum_new_records"])
        and int(download["exact_date_records_returned"])
        + int(download["extra_date_records_rejected"])
        == int(download["records_returned"])
        and int(download["exact_date_records_retained"])
        + int(download["exact_date_records_rejected_incomplete"])
        == int(download["exact_date_records_returned"])
        and int(download["canonical_records_accepted"])
        + int(download["canonical_records_rejected"])
        == int(download["exact_date_records_retained"])
        and int(download["complete_receipt_records_returned"])
        + int(download["incomplete_request_records_returned"])
        == int(download["records_returned"])
        and int(download["complete_receipt_bytes_downloaded"])
        + int(download["incomplete_request_bytes_downloaded"])
        == int(download["bytes_downloaded"])
        and int(download["bytes_downloaded"]) <= int(download["maximum_new_bytes"])
        and int(download["protected_date_records_rejected"])
        <= int(download["extra_date_records_rejected"])
    )
    resource_gate = bool(
        download["overall_decision"] == "blocked_quick_resource_limit"
        and int(download["requests_completed"]) < int(download["requests_planned"])
        and int(download["requests_completed"]) + int(download["requests_remaining"])
        == int(download["requests_planned"])
        and download["resource_ceiling_reached"] is True
        and download["partial_cache_not_used_for_modeling"] is True
        and download["partial_stock_subgroup_not_selected"] is True
        and record_audit_gate
        and coverage_gate
    )
    decision = read_json(PRIMARY / "decision.json")
    assert_safety_flags(decision)
    decision_gate = bool(
        decision["overall_decision"] == download["overall_decision"]
        and decision["binding_holdout_outcomes_opened"] is False
        and decision["partial_download_used_for_modeling"] is False
        and all(
            decision[name] == "blocked"
            for name in (
                "minimal_model_status",
                "frozen_top_5pct_status",
                "options_only_tail_comparison_status",
                "movement_timing_status",
                "holdout_options_coverage_status",
            )
        )
    )
    protected_boundary_gate = bool(
        h0["protected_state_rows_materialised"] == 0
        and not OPTIONS_CACHE.is_file()
        and decision["binding_holdout_outcomes_opened"] is False
    )
    placeholder_gate = verify_blocked_artifacts()
    audit_passed = bool(
        historical["passed"]
        and h0["passed"]
        and exact_features
        and request_gate
        and resource_gate
        and decision_gate
        and protected_boundary_gate
        and placeholder_gate
    )
    write_json(
        PRIMARY / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "protected_start": "2026-01-01",
            "protected_state_rows_materialised": h0["protected_state_rows_materialised"],
            "protected_option_records_rejected": download["protected_date_records_rejected"],
            "protected_option_records_materialised": 0,
            "binding_holdout_outcomes_opened": False,
            "passed": protected_boundary_gate,
        },
    )
    write_json(
        PRIMARY / "holdout_data_authorisation.json",
        {
            **SAFETY_FLAGS,
            "authorized_calendar_range": ["2025-09-01", "2025-12-31"],
            "actual_xnys_sessions_planned": 85,
            "protected_start": "2026-01-01",
            "other_protected_period_opened": False,
            "economic_option_pnl_outcome_opened": False,
            "binding_holdout_outcomes_opened": False,
            "blocked_before_outcome_materialisation": True,
            "passed": protected_boundary_gate,
        },
    )
    write_json(
        PRIMARY / "source_manifest.json",
        {
            **SAFETY_FLAGS,
            "starting_branch": "agent/stock-layer-iv-excess-attribution-quick-v0",
            "starting_sha": "aad974a4e2c12fe3aeb5290540f84ed61480036e",
            "final_branch": "agent/minimal-intraday-iv-excess-holdout-v0",
            "historical": historical,
            "h0_surface": h0,
            "holdout_options_download": download,
            "raw_vendor_data_committed": False,
            "canonical_vendor_data_committed": False,
            "binding_holdout_panel_materialised": False,
        },
    )
    write_json(
        PRIMARY / "determinism_check.json",
        {
            **SAFETY_FLAGS,
            "status": "blocked",
            "historical_models_rebuilt": True,
            "maximum_coefficient_difference": historical["maximum_coefficient_difference"],
            "maximum_probability_difference": historical["maximum_manual_probability_difference"],
            "maximum_threshold_difference": historical["maximum_threshold_difference"],
            "selected_contract_mismatches": None,
            "joined_row_mismatches": None,
            "maximum_feature_difference": None,
            "tail_membership_mismatches": None,
            "maximum_movement_difference": None,
            "bootstrap_repeated": False,
            "null_draws_repeated": False,
            "blocker": "holdout option acquisition incomplete at resource ceiling",
            "passed": False,
        },
    )
    write_json(
        PRIMARY / "lightweight_audit.json",
        {
            **SAFETY_FLAGS,
            "audit_scope": "fail_closed_resource_blocker_and_pre_holdout_surfaces",
            "historical_model_audit": historical,
            "historical_h0_audit": h0,
            "feature_manifest_exact": exact_features,
            "request_plan_exact": request_gate,
            "resource_blocker_confirmed": resource_gate,
            "all_returned_record_accounting_audited": record_audit_gate,
            "all_stock_month_request_cells_audited": coverage_gate,
            "protected_boundary_enforcement_audited": protected_boundary_gate,
            "blocked_placeholder_artifacts_verified": placeholder_gate,
            "holdout_pair_selection_audited": False,
            "holdout_target_audited": False,
            "bootstrap_audited": False,
            "intraday_h0_null_audited": False,
            "decision_logic_audited": decision_gate,
            "binding_experiment_completed": False,
            "passed": audit_passed,
        },
    )
    print(
        "independent audit passed fail-closed blocker enforcement"
        if audit_passed
        else "independent audit failed"
    )
    return 0 if audit_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
