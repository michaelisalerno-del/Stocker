"""Independent audit for raw-OHLC entry-onset discovery V1.

The audit deliberately does not import the experiment runner or any frozen
detector helper.  It reconstructs the 2024 provider-OHLC tape, causal feature
surface, lagged path scale and classes, monthly models, prior-month alert
thresholds, hysteresis episodes, explanations, matched clock controls,
statistics, gates, and integrity bindings from the frozen contract.

This is research-only entry-sign analysis.  It does not calculate P&L, create
positions, or place orders.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


WORK = Path(__file__).resolve().parent
CONTRACT_PATH = WORK / "contracts/20260712-raw-ohlc-entry-onset-discovery-v1.json"
PRE_SCORE_PATH = WORK / "contracts/20260712-raw-ohlc-entry-onset-discovery-v1-pre-score.json"
RUNNER_PATH = WORK / "run_raw_ohlc_entry_onset_discovery_v1.py"
CLEAN_RUNNER_PATH = WORK / "run_clean_slate_causal_ohlc_entries_v1.py"
RAW_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/"
    "instrument_type=stock"
)
ENVIRONMENT_ROOT = Path("/Users/michaelsalerno/StockerLocal")
ARTIFACT_ROOT = Path(
    "/private/tmp/stocker_raw_ohlc_entry_onset_discovery_v1_20260712"
)

CONTRACT_ID = "raw_ohlc_entry_onset_discovery_v1"
SYMBOLS = (
    "AAL", "AAOI", "APLD", "ASTS", "AXTI", "CIFR", "HIMS", "IONQ",
    "IREN", "MARA", "MP", "MRNA", "MSTR", "NVTS", "OKLO", "QBTS",
    "RGTI", "RIOT", "RIVN", "SMCI", "SOFI", "WULF",
)
HORIZONS = (6, 12, 24)
PREDICTION_MONTHS = tuple(f"2024-{month:02d}" for month in range(6, 13))
VALIDATION_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
ALGORITHMS = ("clock_logit", "full_logit", "full_hgb")
CANDIDATES = ("full_logit", "full_hgb")
SIDES = ("long", "short")
CLASSES = (0, 1, 2)
CLASS_FOR_SIDE = {"long": 1, "short": 2}
OPPOSITE_CLASS = {"long": 2, "short": 1}
CLASS_COLUMNS = ("p_no_entry", "p_long_first", "p_short_first")
SESSION_MINUTE = 570
SESSION_END_MINUTE = 960
SESSION_BARS = 78
SCALE_MIN_COUNT = 3
SCALE_MAX_COUNT = 12
SCALE_FLOOR_BPS = 1.0
FIRE_QUANTILE = 0.95
REARM_QUANTILE = 0.75
BLOCK_LENGTH = 5
BOOTSTRAP_DRAWS = 5000
LOWER_QUANTILE = 0.0125
UPPER_QUANTILE = 0.9875
SEED = 20260712
EPSILON = 1e-12

RAW_COLUMNS = ("timestamp", "open", "high", "low", "close")
CLOCK_FEATURES = (
    "clock_fraction",
    "clock_fraction_squared",
    "clock_sin_1",
    "clock_cos_1",
    "clock_sin_2",
    "clock_cos_2",
)
FULL_FEATURES = (
    *CLOCK_FEATURES,
    "log_close_open",
    "log_high_low",
    "signed_body_fraction",
    "absolute_body_fraction",
    "upper_wick_fraction",
    "lower_wick_fraction",
    "close_location",
    "close_return_1",
    "close_return_3",
    "close_return_6",
    "close_return_12",
    "mean_abs_close_return_3",
    "mean_abs_close_return_6",
    "mean_abs_close_return_12",
    "std_close_return_3",
    "std_close_return_6",
    "std_close_return_12",
    "mean_log_range_3",
    "mean_log_range_6",
    "mean_log_range_12",
    "log_range_ratio_6",
    "log_range_ratio_12",
    "session_log_return",
    "distance_to_session_high",
    "distance_from_session_low",
    "session_range_location",
    "running_log_range",
    "distance_to_rolling_high_6",
    "distance_from_rolling_low_6",
    "distance_to_rolling_high_12",
    "distance_from_rolling_low_12",
    "availability_3",
    "availability_6",
    "availability_12",
)
REASON_GROUPS = {
    "clock": CLOCK_FEATURES,
    "current_bar_geometry": (
        "log_close_open", "log_high_low", "signed_body_fraction",
        "absolute_body_fraction", "upper_wick_fraction",
        "lower_wick_fraction", "close_location",
    ),
    "recent_directional_motion": (
        "close_return_1", "close_return_3", "close_return_6",
        "close_return_12",
    ),
    "volatility_and_range_level": (
        "mean_abs_close_return_3", "mean_abs_close_return_6",
        "mean_abs_close_return_12", "std_close_return_3",
        "std_close_return_6", "std_close_return_12", "mean_log_range_3",
        "mean_log_range_6", "mean_log_range_12", "running_log_range",
    ),
    "range_change": ("log_range_ratio_6", "log_range_ratio_12"),
    "session_drift": ("session_log_return",),
    "location_relative_to_extremes": (
        "distance_to_session_high", "distance_from_session_low",
        "session_range_location", "distance_to_rolling_high_6",
        "distance_from_rolling_low_6", "distance_to_rolling_high_12",
        "distance_from_rolling_low_12",
    ),
    "history_availability": (
        "availability_3", "availability_6", "availability_12",
    ),
}
GROUP_ORDER = tuple(REASON_GROUPS)
FEATURE_INDEX = {name: index for index, name in enumerate(FULL_FEATURES)}
GROUP_INDICES = {
    group: np.asarray([FEATURE_INDEX[name] for name in names], dtype=int)
    for group, names in REASON_GROUPS.items()
}
REASON_TEXT = {
    "clock": "where the completed bar sits in the regular session",
    "current_bar_geometry": "the completed bar's body, range, wicks, and close location",
    "recent_directional_motion": "continuous recent close-to-close motion",
    "volatility_and_range_level": "the recent level of price variation and bar range",
    "range_change": "the current range relative to its recent range level",
    "session_drift": "continuous displacement from the session's first open",
    "location_relative_to_extremes": "continuous location relative to session and rolling extremes",
    "history_availability": "how much exact contiguous history was available",
}
PRE_OUTCOME_LEDGER_COLUMNS = (
    "anchor_id",
    "fold_month",
    "symbol_norm",
    "session_date",
    "decision_timestamp",
    "bar_ordinal",
    "segment_index",
    "segment_position",
    "algorithm",
    "horizon",
    "causal_scale_bps",
    "availability_12",
    "clock_bin_15",
    "clock_bin_30",
    "availability_bucket",
    *CLASS_COLUMNS,
)

EXPECTED_REGULAR_ROWS = 424583
EXPECTED_UNION_SESSIONS = 252
EXPECTED_SYMBOL_SESSIONS = 5539
EXPECTED_GAPS = 2612
EXPECTED_ANNUAL_ROWS = {6: 365075, 12: 330577, 24: 264817}
EXPECTED_VALIDATION_ROWS = {6: 186112, 12: 168639, 24: 135240}
EXPECTED_JUNE_ROWS = {6: 27733, 12: 25143, 24: 20214}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provider_path(symbol: str) -> Path:
    return RAW_ROOT / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def source_paths() -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "runner": RUNNER_PATH,
        "frozen_clean_slate_runner": CLEAN_RUNNER_PATH,
        "environment_pyproject": ENVIRONMENT_ROOT / "pyproject.toml",
        "environment_uv_lock": ENVIRONMENT_ROOT / "uv.lock",
    }
    paths.update(
        {f"provider_full_file_{symbol}": provider_path(symbol) for symbol in SYMBOLS}
    )
    return paths


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def assert_contract(contract: dict[str, Any]) -> None:
    if contract["contract_id"] != CONTRACT_ID:
        raise AssertionError("unexpected contract id")
    if not (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["broker_connection_enabled"] is False
        and contract["paper_or_demo_execution_enabled"] is False
        and contract["deployment_enabled"] is False
        and contract["strategy_promotion_permitted"] is False
        and contract["economic_edge_claim_permitted"] is False
        and contract["pnl_evaluation_permitted"] is False
    ):
        raise AssertionError("research safety boundary changed")
    if tuple(contract["universe"]["symbols"]) != SYMBOLS:
        raise AssertionError("universe changed")
    if tuple(contract["periods"]["validation_months"]) != VALIDATION_MONTHS:
        raise AssertionError("validation months changed")
    if tuple(contract["decision_and_path"]["horizons_bars"]) != HORIZONS:
        raise AssertionError("horizons changed")
    if tuple(contract["features"]["clock_features"]) != CLOCK_FEATURES:
        raise AssertionError("clock feature order changed")
    if tuple(contract["features"]["full_features_in_order"]) != FULL_FEATURES:
        raise AssertionError("full feature order changed")
    forbidden_feature_tokens = (
        "regime", "state", "loop", "cycle", "b0", "template", "volume",
        "order_flow", "quote", "tick", "news", "fundamental", "prediction",
    )
    if any(
        any(token in feature.lower() for token in forbidden_feature_tokens)
        for feature in FULL_FEATURES
    ):
        raise AssertionError("forbidden prior/non-OHLC feature entered the whitelist")
    observed_groups = {
        key: tuple(value)
        for key, value in contract["features"]["fixed_reason_groups"].items()
    }
    if observed_groups != REASON_GROUPS:
        raise AssertionError("reason groups changed")
    if tuple(contract["sources"]["columns_read"]) != RAW_COLUMNS:
        raise AssertionError("raw column whitelist changed")
    if contract["sources"]["provider_volume_label"] != "historical_volume_not_used":
        raise AssertionError("volume provenance changed")
    if any(
        contract["periods"][f"{year}_read_permitted"]
        for year in (2023, 2025, 2026)
    ):
        raise AssertionError("later/backward period access unexpectedly permitted")
    algorithms = {row["name"]: row for row in contract["algorithms"]}
    if tuple(row["name"] for row in contract["algorithms"]) != ALGORITHMS:
        raise AssertionError("algorithm order changed")
    expected_logit = {
        "penalty": "l2",
        "C": 0.2,
        "fit_intercept": True,
        "solver": "lbfgs",
        "max_iter": 500,
        "tol": 1e-6,
        "random_state": SEED,
    }
    if algorithms["clock_logit"]["parameters"] != expected_logit:
        raise AssertionError("clock logit changed")
    if algorithms["full_logit"]["parameters"] != expected_logit:
        raise AssertionError("full logit changed")
    expected_hgb = {
        "loss": "log_loss",
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 7,
        "max_depth": 3,
        "min_samples_leaf": 500,
        "l2_regularization": 10.0,
        "max_bins": 64,
        "early_stopping": False,
        "random_state": SEED,
    }
    if algorithms["full_hgb"]["parameters"] != expected_hgb:
        raise AssertionError("HGB changed")
    onset = contract["thresholds_and_onsets"]
    if (
        onset["fire_probability_quantile"] != FIRE_QUANTILE
        or onset["rearm_probability_quantile"] != REARM_QUANTILE
    ):
        raise AssertionError("alert quantiles changed")
    inference = contract["metrics"]["inference"]
    if (
        inference["draws"] != BOOTSTRAP_DRAWS
        or inference["random_state"] != SEED
        or inference["lower_quantile"] != LOWER_QUANTILE
        or inference["upper_quantile"] != UPPER_QUANTILE
    ):
        raise AssertionError("inference contract changed")
    for key, expected in {
        "expected_target_blind_eligible_rows_2024": EXPECTED_ANNUAL_ROWS,
        "expected_target_blind_validation_rows_july_december": EXPECTED_VALIDATION_ROWS,
        "expected_target_blind_calibration_rows_june": EXPECTED_JUNE_ROWS,
    }.items():
        observed = {
            int(horizon): int(rows)
            for horizon, rows in contract["decision_and_path"][key].items()
        }
        if observed != expected:
            raise AssertionError(f"contract support map changed: {key}")
    chronology = contract["periods"]["fold_chronology"]
    expected_chronology = {
        "same_score_month_outcomes_read_before_that_month_probability": False,
        "prior_completed_validation_month_path_labels_may_train_later_folds": True,
        "global_bundle_written_after_some_prior_validation_month_training_labels_are_read": True,
        "global_bundle_written_before_final_all_fold_evaluation_join": True,
        "global_bundle_written_before_any_validation_outcome_is_read": False,
        "interpretation": (
            "causal monthly expanding OOF development, not a globally sealed "
            "July-December holdout"
        ),
    }
    if chronology != expected_chronology:
        raise AssertionError("fold chronology contract changed")
    integrity = contract["integrity"]
    for key, expected in {
        "each_score_month_probability_generated_before_same_month_outcomes_are_read": True,
        "prior_completed_validation_month_labels_used_in_later_expanding_folds": True,
        "global_bundle_written_before_final_all_fold_evaluation_join": True,
        "global_bundle_written_before_any_validation_outcome_is_read": False,
    }.items():
        if integrity[key] is not expected:
            raise AssertionError(f"integrity chronology changed: {key}")


def ast_source_boundary(runner_path: Path) -> dict[str, Any]:
    """Check that the runner stays on the frozen raw-2024 OHLC surface."""

    source = runner_path.read_text()
    tree = ast.parse(source, filename=str(runner_path))
    forbidden_imports: list[str] = []
    suspicious_paths: list[str] = []
    forbidden_calculation_calls: list[str] = []
    clean_helper_attributes: set[str] = set()
    provider_reads = 0
    read_columns: list[list[str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "clean_slate"
        ):
            clean_helper_attributes.add(node.attr)
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports = [node.module or ""]
        else:
            imports = []
        for name in imports:
            lowered = name.lower()
            if any(
                token in lowered
                for token in ("regime", "semimarkov", "loop", "detector", "template")
            ):
                forbidden_imports.append(name)

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            else:
                call_name = ""
            lowered_call = call_name.lower()
            if any(
                token in lowered_call
                for token in ("profit_factor", "sharpe", "drawdown", "pnl")
            ):
                forbidden_calculation_calls.append(call_name)
            if call_name == "read_parquet":
                keywords = {item.arg: item.value for item in node.keywords}
                if "filters" in keywords and "columns" in keywords:
                    provider_reads += 1
                    column_node = keywords["columns"]
                    if isinstance(column_node, (ast.List, ast.Tuple)):
                        read_columns.append(
                            [
                                item.value
                                for item in column_node.elts
                                if isinstance(item, ast.Constant)
                                and isinstance(item.value, str)
                            ]
                        )

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            lowered = value.lower()
            path_like = "/" in value or value.endswith((".parquet", ".csv", ".json"))
            if path_like:
                cleaned = lowered.replace("20260712", "")
                if any(year in cleaned for year in ("2023", "2025", "2026")):
                    suspicious_paths.append(value)

    if forbidden_imports:
        raise AssertionError(f"forbidden prior-research import: {forbidden_imports}")
    if suspicious_paths:
        raise AssertionError(f"later/backward data path in runner: {suspicious_paths}")
    if forbidden_calculation_calls:
        raise AssertionError(
            f"forbidden economic-output calculation: {forbidden_calculation_calls}"
        )
    permitted_helper_attributes = {
        "SOURCE_COLUMNS", "SYMBOLS", "CLOCK_FEATURES", "FULL_FEATURES",
        "provider_path", "__file__", "load_tape", "build_feature_surface",
    }
    if not clean_helper_attributes.issubset(permitted_helper_attributes):
        raise AssertionError(
            "runner accesses a non-OHLC frozen-helper surface: "
            f"{sorted(clean_helper_attributes - permitted_helper_attributes)}"
        )
    # The runner calls a whole-file-hash-bound earlier OHLC loader.  The audit
    # never imports that helper; instead it inspects the helper's provider read.
    clean_tree = ast.parse(
        CLEAN_RUNNER_PATH.read_text(), filename=str(CLEAN_RUNNER_PATH)
    )
    clean_provider_reads = 0
    for node in ast.walk(clean_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_parquet"
        ):
            keywords = {item.arg for item in node.keywords}
            if {"filters", "columns"}.issubset(keywords):
                clean_provider_reads += 1
    if provider_reads != 0 or clean_provider_reads != 1:
        raise AssertionError(
            "unexpected provider read boundary: "
            f"runner={provider_reads}, frozen_helper={clean_provider_reads}"
        )
    return {
        "ast_nodes": sum(1 for _ in ast.walk(tree)),
        "forbidden_imports": forbidden_imports,
        "later_or_backward_paths": suspicious_paths,
        "forbidden_economic_calculation_calls": forbidden_calculation_calls,
        "hash_bound_frozen_helper_attributes": sorted(clean_helper_attributes),
        "runner_provider_reads": provider_reads,
        "frozen_helper_predicate_column_whitelisted_provider_reads": clean_provider_reads,
        "literal_read_columns": read_columns,
    }


def _read_provider_2024(path: Path) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=list(RAW_COLUMNS),
        filters=[
            ("timestamp", ">=", datetime(2024, 1, 1, tzinfo=timezone.utc)),
            ("timestamp", "<", datetime(2025, 1, 1, tzinfo=timezone.utc)),
        ],
    )


def load_regular_tape() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    per_symbol: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        raw = _read_provider_2024(provider_path(symbol)).copy()
        timestamp = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
        if timestamp.isna().any():
            raise AssertionError(f"unparseable timestamp for {symbol}")
        if not timestamp.ge(pd.Timestamp("2024-01-01", tz="UTC")).all():
            raise AssertionError("pre-2024 provider row")
        if not timestamp.lt(pd.Timestamp("2025-01-01", tz="UTC")).all():
            raise AssertionError("post-2024 provider row")
        raw["timestamp"] = timestamp
        if raw["timestamp"].duplicated(keep=False).any():
            raise AssertionError(f"duplicate provider timestamp for {symbol}")
        local = timestamp.dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        regular = minute.ge(SESSION_MINUTE) & minute.lt(SESSION_END_MINUTE)
        numeric = raw[["open", "high", "low", "close"]].to_numpy(float)
        finite_positive = np.isfinite(numeric).all(axis=1) & (numeric > 0).all(axis=1)
        order_ok = (
            (numeric[:, 2] <= np.minimum(numeric[:, 0], numeric[:, 3]))
            & (np.maximum(numeric[:, 0], numeric[:, 3]) <= numeric[:, 1])
        )
        valid = finite_positive & order_ok
        accepted = regular.to_numpy(bool) & valid
        selected = raw.loc[accepted].copy()
        selected_local = selected["timestamp"].dt.tz_convert("America/New_York")
        selected_minute = selected_local.dt.hour * 60 + selected_local.dt.minute
        if not (((selected_minute - SESSION_MINUTE) % 5) == 0).all():
            raise AssertionError(f"off-grid regular timestamp for {symbol}")
        selected["symbol"] = symbol
        selected["symbol_norm"] = symbol
        selected["session_date"] = selected_local.dt.strftime("%Y-%m-%d")
        selected["month"] = selected["session_date"].str[:7]
        selected["month_key"] = selected["month"]
        selected["bar_ordinal"] = (
            (selected_minute - SESSION_MINUTE) // 5
        ).to_numpy(np.int16)
        selected["clock_ordinal"] = selected["bar_ordinal"]
        selected = selected.sort_values("timestamp", kind="stable").reset_index(drop=True)
        previous = selected.groupby("session_date", sort=False)["timestamp"].shift()
        continuation = (selected["timestamp"] - previous).eq(pd.Timedelta(minutes=5))
        gap_count = int((previous.notna() & ~continuation).sum())
        per_symbol.append(
            {
                "symbol_norm": symbol,
                "raw_predicate_2024_rows": int(len(raw)),
                "outside_regular_session_rows": int((~regular).sum()),
                "nonfinite_or_invalid_ohlc_rows": int((~valid).sum()),
                "duplicate_timestamp_rows": 0,
                "accepted_regular_rows": int(len(selected)),
                "symbol_sessions": int(selected["session_date"].nunique()),
                "within_session_nonfive_minute_gaps": gap_count,
            }
        )
        frames.append(selected)

    tape = pd.concat(frames, ignore_index=True).sort_values(
        ["symbol_norm", "session_date", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    if tape.duplicated(["symbol_norm", "timestamp"]).any():
        raise AssertionError("duplicate symbol timestamp after concatenation")
    prior_timestamp = tape.groupby(
        ["symbol_norm", "session_date"], sort=False
    )["timestamp"].shift()
    continuation = (tape["timestamp"] - prior_timestamp).eq(pd.Timedelta(minutes=5))
    segment_start = ~continuation
    tape["segment_id"] = segment_start.cumsum().to_numpy(np.int64) - 1
    tape["segment_index"] = (
        segment_start.groupby(
            [tape["symbol_norm"], tape["session_date"]], sort=False
        ).cumsum().to_numpy(np.int16)
        - 1
    )
    tape["segment_position"] = tape.groupby(
        "segment_id", sort=False
    ).cumcount().to_numpy(np.int16)
    tape["segment_size"] = tape.groupby(
        "segment_id", sort=False
    )["timestamp"].transform("size").to_numpy(np.int16)
    tape["source_position"] = np.arange(len(tape), dtype=np.int64)
    diagnostics = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "raw_predicate_2024_rows": int(
            sum(row["raw_predicate_2024_rows"] for row in per_symbol)
        ),
        "regular_valid_rows": int(len(tape)),
        "union_sessions": int(tape["session_date"].nunique()),
        "symbol_sessions": int(
            tape[["symbol_norm", "session_date"]].drop_duplicates().shape[0]
        ),
        "within_session_nonfive_minute_gaps": int(
            (prior_timestamp.notna() & ~continuation).sum()
        ),
        "skipped_support_reasons": {
            "nonfinite_or_invalid_ohlc": int(
                sum(row["nonfinite_or_invalid_ohlc_rows"] for row in per_symbol)
            ),
            "duplicate_timestamp": 0,
            "outside_regular_session": int(
                sum(row["outside_regular_session_rows"] for row in per_symbol)
            ),
            "nonfive_minute_gap": int((prior_timestamp.notna() & ~continuation).sum()),
            "insufficient_same_session_horizon": {},
        },
        "per_symbol": per_symbol,
    }
    observed = (
        diagnostics["regular_valid_rows"],
        diagnostics["union_sessions"],
        diagnostics["symbol_sessions"],
        diagnostics["within_session_nonfive_minute_gaps"],
    )
    expected = (
        EXPECTED_REGULAR_ROWS,
        EXPECTED_UNION_SESSIONS,
        EXPECTED_SYMBOL_SESSIONS,
        EXPECTED_GAPS,
    )
    if observed != expected:
        raise AssertionError(f"target-blind tape drift: {observed!r} != {expected!r}")
    return tape, diagnostics


def _rolling(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    column: str,
    window: int,
    operation: str,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    roll = frame.groupby(list(group_columns), sort=False)[column].rolling(
        window, min_periods=window if min_periods is None else min_periods
    )
    if operation == "mean":
        result = roll.mean()
    elif operation == "median":
        result = roll.median()
    elif operation == "std":
        result = roll.std(ddof=0)
    elif operation == "maximum":
        result = roll.max()
    elif operation == "minimum":
        result = roll.min()
    else:
        raise AssertionError(operation)
    return result.reset_index(level=list(group_columns), drop=True).sort_index()


def reconstruct_features(tape: pd.DataFrame) -> pd.DataFrame:
    frame = tape.copy()
    fraction = frame["bar_ordinal"].to_numpy(float) * 5.0 / 385.0
    frame["clock_fraction"] = fraction
    frame["clock_fraction_squared"] = fraction**2
    frame["clock_sin_1"] = np.sin(2.0 * np.pi * fraction)
    frame["clock_cos_1"] = np.cos(2.0 * np.pi * fraction)
    frame["clock_sin_2"] = np.sin(4.0 * np.pi * fraction)
    frame["clock_cos_2"] = np.cos(4.0 * np.pi * fraction)

    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    bar_range = highs - lows
    usable = bar_range > EPSILON
    frame["log_close_open"] = np.log(closes / opens)
    frame["log_high_low"] = np.log(highs / lows)
    frame["signed_body_fraction"] = 0.0
    frame["absolute_body_fraction"] = 0.0
    frame["upper_wick_fraction"] = 0.0
    frame["lower_wick_fraction"] = 0.0
    frame["close_location"] = 0.5
    frame.loc[usable, "signed_body_fraction"] = (
        closes[usable] - opens[usable]
    ) / bar_range[usable]
    frame.loc[usable, "absolute_body_fraction"] = np.abs(
        closes[usable] - opens[usable]
    ) / bar_range[usable]
    frame.loc[usable, "upper_wick_fraction"] = (
        highs[usable] - np.maximum(opens[usable], closes[usable])
    ) / bar_range[usable]
    frame.loc[usable, "lower_wick_fraction"] = (
        np.minimum(opens[usable], closes[usable]) - lows[usable]
    ) / bar_range[usable]
    frame.loc[usable, "close_location"] = (
        closes[usable] - lows[usable]
    ) / bar_range[usable]

    segment_keys = ("symbol_norm", "session_date", "segment_index")
    frame["_log_close"] = np.log(frame["close"])
    segments = frame.groupby(list(segment_keys), sort=False)
    for window in (1, 3, 6, 12):
        frame[f"close_return_{window}"] = (
            frame["_log_close"] - segments["_log_close"].shift(window)
        )
    frame["_absolute_return_1"] = frame["close_return_1"].abs()
    for window in (3, 6, 12):
        frame[f"mean_abs_close_return_{window}"] = _rolling(
            frame, segment_keys, "_absolute_return_1", window, "mean"
        )
        frame[f"std_close_return_{window}"] = _rolling(
            frame, segment_keys, "close_return_1", window, "std"
        )
        frame[f"mean_log_range_{window}"] = _rolling(
            frame, segment_keys, "log_high_low", window, "mean"
        )
    for window in (6, 12):
        frame[f"log_range_ratio_{window}"] = (
            frame["log_high_low"] - frame[f"mean_log_range_{window}"]
        )

    sessions = frame.groupby(["symbol_norm", "session_date"], sort=False)
    session_open = sessions["open"].transform("first")
    running_high = sessions["high"].cummax()
    running_low = sessions["low"].cummin()
    frame["session_log_return"] = np.log(frame["close"] / session_open)
    frame["distance_to_session_high"] = np.log(frame["close"] / running_high)
    frame["distance_from_session_low"] = np.log(frame["close"] / running_low)
    running_range = running_high.to_numpy(float) - running_low.to_numpy(float)
    running_usable = running_range > EPSILON
    frame["session_range_location"] = 0.5
    frame.loc[running_usable, "session_range_location"] = (
        closes[running_usable] - running_low.to_numpy(float)[running_usable]
    ) / running_range[running_usable]
    frame["running_log_range"] = np.log(running_high / running_low)
    for window in (6, 12):
        rolling_high = _rolling(frame, segment_keys, "high", window, "maximum")
        rolling_low = _rolling(frame, segment_keys, "low", window, "minimum")
        frame[f"distance_to_rolling_high_{window}"] = np.log(
            frame["close"] / rolling_high
        )
        frame[f"distance_from_rolling_low_{window}"] = np.log(
            frame["close"] / rolling_low
        )
    available = frame["segment_position"].to_numpy(float) + 1.0
    for window in (3, 6, 12):
        frame[f"availability_{window}"] = np.minimum(available / window, 1.0)
    return frame.loc[:, FULL_FEATURES].copy()


def reconstruct_lagged_scale(tape: pd.DataFrame) -> pd.Series:
    frame = tape[[
        "symbol_norm", "session_date", "segment_index", "high", "low", "close"
    ]].copy()
    keys = ("symbol_norm", "session_date", "segment_index")
    prior_close = frame.groupby(list(keys), sort=False)["close"].shift()
    log_range = np.log(frame["high"] / frame["low"])
    high_gap = np.abs(np.log(frame["high"] / prior_close))
    low_gap = np.abs(np.log(frame["low"] / prior_close))
    # At a segment boundary no prior close exists; the first bar's true range is
    # its own high/low range, exactly as frozen in the contract implementation.
    true_range = log_range.to_numpy(float).copy()
    true_range = np.fmax(true_range, high_gap.to_numpy(float))
    true_range = np.fmax(true_range, low_gap.to_numpy(float))
    frame["_true_range_bps"] = 10000.0 * true_range
    rolling = _rolling(
        frame,
        keys,
        "_true_range_bps",
        SCALE_MAX_COUNT,
        "median",
        min_periods=SCALE_MIN_COUNT,
    )
    scale = rolling.groupby(
        [tape["symbol_norm"], tape["session_date"], tape["segment_index"]],
        sort=False,
    ).shift()
    return scale.clip(lower=SCALE_FLOOR_BPS).rename("causal_scale_bps")


def _path_for_anchor(
    tape: pd.DataFrame,
    position: int,
    horizon: int,
    scale_bps: float,
) -> dict[str, Any]:
    anchor = tape.iloc[position]
    path = tape.iloc[position + 1 : position + horizon + 1]
    if len(path) != horizon or not path["segment_id"].eq(anchor["segment_id"]).all():
        raise AssertionError("path crossed a segment")
    reference = float(path.iloc[0]["open"])
    upper = reference * math.exp(scale_bps / 10000.0)
    lower = reference * math.exp(-scale_bps / 10000.0)
    status = "no_hit_by_horizon"
    target_class = 0
    confirmation_step: float = math.nan
    confirmation_prefix_end = horizon
    for step, row in enumerate(path.itertuples(index=False), start=1):
        future_open = float(row.open)
        if future_open >= upper:
            status = "long_first"
            target_class = 1
            confirmation_step = float(step)
            confirmation_prefix_end = step
            break
        if future_open <= lower:
            status = "short_first"
            target_class = 2
            confirmation_step = float(step)
            confirmation_prefix_end = step
            break
        upper_touch = float(row.high) >= upper
        lower_touch = float(row.low) <= lower
        if upper_touch and lower_touch:
            status = "intrabar_ambiguous"
            target_class = 0
            confirmation_step = float(step)
            confirmation_prefix_end = step
            break
        if upper_touch:
            status = "long_first"
            target_class = 1
            confirmation_step = float(step)
            confirmation_prefix_end = step
            break
        if lower_touch:
            status = "short_first"
            target_class = 2
            confirmation_step = float(step)
            confirmation_prefix_end = step
            break

    highs = path["high"].to_numpy(float)
    lows = path["low"].to_numpy(float)
    upside_bps = float(10000.0 * np.max(np.log(highs / reference)))
    downside_bps = float(10000.0 * np.max(np.log(reference / lows)))
    prefix = path.iloc[:confirmation_prefix_end]
    prefix_upside = float(
        10000.0 * np.max(np.log(prefix["high"].to_numpy(float) / reference))
    )
    prefix_downside = float(
        10000.0 * np.max(np.log(reference / prefix["low"].to_numpy(float)))
    )
    return {
        "target_class": int(target_class),
        "status": status,
        "reference_price": reference,
        "upper_barrier": upper,
        "lower_barrier": lower,
        "upside_mfe_bps": upside_bps,
        "downside_mfe_bps": downside_bps,
        "first_confirmation_step": confirmation_step,
        "prefix_upside_bps": prefix_upside,
        "prefix_downside_bps": prefix_downside,
        "rapid_long_confirmation_within_3": int(
            target_class == 1 and confirmation_step <= 3
        ),
        "rapid_short_confirmation_within_3": int(
            target_class == 2 and confirmation_step <= 3
        ),
    }


def build_horizon_surface(
    tape: pd.DataFrame,
    features: pd.DataFrame,
    scale: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    valid = (
        scale.notna().to_numpy()
        & (tape["segment_position"].to_numpy(int) + horizon < tape["segment_size"].to_numpy(int))
    )
    positions = np.flatnonzero(valid)
    offsets = np.arange(1, horizon + 1, dtype=np.int64)
    future_positions = positions[:, None] + offsets[None, :]
    segment = tape["segment_id"].to_numpy(np.int64)
    if not np.all(segment[future_positions] == segment[positions, None]):
        raise AssertionError("vectorized path crossed a segment")
    opens = tape["open"].to_numpy(float)[future_positions]
    highs = tape["high"].to_numpy(float)[future_positions]
    lows = tape["low"].to_numpy(float)[future_positions]
    reference = opens[:, 0]
    scale_value = scale.iloc[positions].to_numpy(float)
    upper = reference * np.exp(scale_value / 10000.0)
    lower = reference * np.exp(-scale_value / 10000.0)

    event_code = np.zeros(opens.shape, dtype=np.int8)
    gap_up = opens >= upper[:, None]
    gap_down = opens <= lower[:, None]
    event_code[gap_up] = 1
    event_code[gap_down] = 2
    inside = ~(gap_up | gap_down)
    upper_touch = highs >= upper[:, None]
    lower_touch = lows <= lower[:, None]
    event_code[inside & upper_touch & lower_touch] = 3
    event_code[inside & upper_touch & ~lower_touch] = 1
    event_code[inside & lower_touch & ~upper_touch] = 2
    has_event = (event_code != 0).any(axis=1)
    first_index = np.argmax(event_code != 0, axis=1)
    first_code = np.where(
        has_event, event_code[np.arange(len(positions)), first_index], 0
    ).astype(np.int8)
    target_class = np.where(
        first_code == 1, 1, np.where(first_code == 2, 2, 0)
    ).astype(np.int8)
    status = np.full(len(positions), "no_hit_by_horizon", dtype=object)
    status[first_code == 1] = "long_first"
    status[first_code == 2] = "short_first"
    status[first_code == 3] = "intrabar_ambiguous"
    first_step = np.where(has_event, first_index + 1, np.nan).astype(float)
    upside = 10000.0 * np.log(np.max(highs, axis=1) / reference)
    downside = 10000.0 * np.log(reference / np.min(lows, axis=1))
    prefix_end = np.where(has_event, first_index, horizon - 1)
    prefix_mask = np.arange(horizon)[None, :] <= prefix_end[:, None]
    prefix_high = np.max(np.where(prefix_mask, highs, -np.inf), axis=1)
    prefix_low = np.min(np.where(prefix_mask, lows, np.inf), axis=1)
    prefix_upside = 10000.0 * np.log(prefix_high / reference)
    prefix_downside = 10000.0 * np.log(reference / prefix_low)

    base_columns = [
        "symbol_norm", "session_date", "month_key", "timestamp", "bar_ordinal",
        "segment_id", "segment_index", "segment_position", "source_position",
    ]
    surface = tape.iloc[positions].loc[:, base_columns].reset_index(drop=True)
    surface = surface.rename(columns={"timestamp": "decision_timestamp"})
    feature_values = features.iloc[positions].reset_index(drop=True)
    surface = pd.concat([surface, feature_values], axis=1)
    surface["anchor_id"] = (
        surface["symbol_norm"].astype(str)
        + "|"
        + surface["decision_timestamp"].astype(str)
        + f"|h{horizon}"
    )
    surface["horizon"] = np.int16(horizon)
    surface["causal_scale_bps"] = scale_value
    surface["target_class"] = target_class
    surface["status"] = status
    surface["reference_price"] = reference
    surface["upper_barrier"] = upper
    surface["lower_barrier"] = lower
    surface["upside_mfe_bps"] = upside
    surface["downside_mfe_bps"] = downside
    surface["first_confirmation_step"] = first_step
    surface["prefix_upside_bps"] = prefix_upside
    surface["prefix_downside_bps"] = prefix_downside
    surface["rapid_long_confirmation_within_3"] = (
        (target_class == 1) & (first_step <= 3)
    ).astype(np.int8)
    surface["rapid_short_confirmation_within_3"] = (
        (target_class == 2) & (first_step <= 3)
    ).astype(np.int8)
    surface["path_end_timestamp"] = surface["decision_timestamp"] + pd.to_timedelta(
        5 * horizon, unit="minute"
    )
    surface["clock_bin_15"] = (
        surface["bar_ordinal"].to_numpy(int) // 3
    ).astype(np.int16)
    surface["clock_bin_30"] = (
        surface["bar_ordinal"].to_numpy(int) // 6
    ).astype(np.int16)
    surface["availability_bucket"] = history_availability_bucket(
        surface["availability_12"]
    )
    surface = surface.sort_values(
        ["symbol_norm", "session_date", "decision_timestamp"], kind="stable"
    ).reset_index(drop=True)
    if len(surface) != EXPECTED_ANNUAL_ROWS[horizon]:
        raise AssertionError(f"annual h{horizon} support drift")
    if int(surface["month_key"].isin(VALIDATION_MONTHS).sum()) != EXPECTED_VALIDATION_ROWS[horizon]:
        raise AssertionError(f"validation h{horizon} support drift")
    if int(surface["month_key"].eq("2024-06").sum()) != EXPECTED_JUNE_ROWS[horizon]:
        raise AssertionError(f"June h{horizon} support drift")
    return surface


def prefix_causality_check(tape: pd.DataFrame, features: pd.DataFrame) -> int:
    candidates = np.unique(np.linspace(0, len(tape) - 1, num=96, dtype=int))
    checked = 0
    for position in candidates:
        row = tape.iloc[position]
        prefix = tape.loc[
            tape["symbol_norm"].eq(row["symbol_norm"])
            & tape["session_date"].eq(row["session_date"])
            & tape["timestamp"].le(row["timestamp"])
        ].copy().reset_index(drop=True)
        rebuilt = reconstruct_features(prefix).iloc[-1].to_numpy(float)
        reference = features.iloc[position].to_numpy(float)
        if not np.allclose(rebuilt, reference, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise AssertionError(f"feature prefix causality mismatch at {position}")
        checked += 1
    return checked


def training_medians(matrix: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        medians = np.nanmedian(matrix, axis=0)
    if not np.isfinite(medians).all():
        raise AssertionError("a fold feature has no finite training median")
    return medians.astype(float)


def impute(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    rows, columns = np.where(np.isnan(result))
    result[rows, columns] = medians[columns]
    if not np.isfinite(result).all():
        raise AssertionError("nonfinite imputed design")
    return result


def equal_symbol_session_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each symbol equal mass, then each represented session equal mass."""

    key = frame[["symbol_norm", "session_date"]].astype(str)
    session_rows = key.groupby(
        ["symbol_norm", "session_date"], sort=False
    )["session_date"].transform("size").to_numpy(float)
    sessions_per_symbol = (
        key.drop_duplicates()
        .groupby("symbol_norm", sort=False)["session_date"]
        .size()
    )
    raw = (
        1.0
        / key["symbol_norm"].map(sessions_per_symbol).to_numpy(float)
        / session_rows
    )
    weights = raw / raw.mean()
    if not np.isclose(weights.mean(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("weights do not average one")
    totals = pd.Series(weights).groupby(key["symbol_norm"].to_numpy(str)).sum()
    if not np.allclose(totals, totals.iloc[0], rtol=0.0, atol=1e-8):
        raise AssertionError("symbols do not have equal total weight")
    return weights


def weighted_inverse_cdf_quantile(
    values: Iterable[float],
    weights: Iterable[float],
    quantile: float,
    tie_break: Iterable[str],
) -> float:
    value_array = np.asarray(list(values), dtype=float)
    weight_array = np.asarray(list(weights), dtype=float)
    tie_array = np.asarray(list(tie_break), dtype=str)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    if not valid.any():
        raise AssertionError("weighted quantile has no support")
    value_array = value_array[valid]
    weight_array = weight_array[valid]
    tie_array = tie_array[valid]
    order = np.lexsort((tie_array, value_array))
    ordered_value = value_array[order]
    ordered_weight = weight_array[order]
    target = float(quantile) * float(ordered_weight.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weight), target, side="left"))
    return float(ordered_value[min(index, len(ordered_value) - 1)])


def model_parameters(algorithm: str) -> dict[str, Any]:
    if algorithm in {"clock_logit", "full_logit"}:
        return {
            "penalty": "l2",
            "C": 0.2,
            "fit_intercept": True,
            "solver": "lbfgs",
            "max_iter": 500,
            "tol": 1e-6,
            "random_state": SEED,
        }
    if algorithm == "full_hgb":
        return {
            "loss": "log_loss",
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 7,
            "max_depth": 3,
            "min_samples_leaf": 500,
            "l2_regularization": 10.0,
            "max_bins": 64,
            "early_stopping": False,
            "random_state": SEED,
        }
    raise AssertionError(algorithm)


def replay_models(
    surfaces: dict[int, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, int, str], dict[str, Any]],
]:
    """Fit every monthly fold independently and retain objects for explanations."""

    probability_parts: list[pd.DataFrame] = []
    preprocessing_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, int, str], dict[str, Any]] = {}
    for horizon in HORIZONS:
        surface = surfaces[horizon]
        for month in PREDICTION_MONTHS:
            boundary = pd.Timestamp(f"{month}-01", tz="UTC")
            train_mask = surface["path_end_timestamp"].lt(boundary)
            score_mask = surface["month_key"].eq(month)
            if (train_mask & score_mask).any():
                raise AssertionError("monthly training and scoring overlap")
            train = surface.loc[train_mask].copy()
            score = surface.loc[score_mask].copy()
            if train.empty or score.empty:
                raise AssertionError(f"empty fold {month} h{horizon}")
            target = train["target_class"].to_numpy(np.int8)
            if tuple(np.unique(target)) != CLASSES:
                raise AssertionError(f"missing training class {month} h{horizon}")
            weights = equal_symbol_session_weights(train)
            for algorithm in ALGORITHMS:
                names = CLOCK_FEATURES if algorithm == "clock_logit" else FULL_FEATURES
                raw_train = train.loc[:, names].to_numpy(float)
                raw_score = score.loc[:, names].to_numpy(float)
                medians = training_medians(raw_train)
                design_train = impute(raw_train, medians)
                design_score = impute(raw_score, medians)
                scaler: StandardScaler | None = None
                scaler_mean = np.full(len(names), np.nan)
                scaler_scale = np.full(len(names), np.nan)
                if algorithm.endswith("logit"):
                    scaler = StandardScaler().fit(design_train, sample_weight=weights)
                    design_train = scaler.transform(design_train)
                    design_score = scaler.transform(design_score)
                    estimator: Any = LogisticRegression(**model_parameters(algorithm))
                    scaler_mean = scaler.mean_.astype(float)
                    scaler_scale = scaler.scale_.astype(float)
                else:
                    estimator = HistGradientBoostingClassifier(
                        **model_parameters(algorithm)
                    )
                estimator.fit(design_train, target, sample_weight=weights)
                if tuple(int(value) for value in estimator.classes_) != CLASSES:
                    raise AssertionError("estimator class order changed")
                probabilities = estimator.predict_proba(design_score).astype(float)
                if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
                    raise AssertionError("probability rows do not sum to one")
                output = score[
                    [
                        "anchor_id",
                        "symbol_norm",
                        "session_date",
                        "month_key",
                        "decision_timestamp",
                        "bar_ordinal",
                        "segment_index",
                        "segment_position",
                        "horizon",
                        "causal_scale_bps",
                        "availability_12",
                        "clock_bin_15",
                        "clock_bin_30",
                        "availability_bucket",
                    ]
                ].copy()
                output["fold_month"] = month
                output["algorithm"] = algorithm
                output["p_no_entry"] = probabilities[:, 0]
                output["p_long_first"] = probabilities[:, 1]
                output["p_short_first"] = probabilities[:, 2]
                output = output.loc[:, PRE_OUTCOME_LEDGER_COLUMNS]
                probability_parts.append(output)

                for order, feature in enumerate(names):
                    preprocessing_rows.append(
                        {
                            "fold_month": month,
                            "algorithm": algorithm,
                            "horizon": horizon,
                            "feature_order": order,
                            "feature": feature,
                            "training_median": float(medians[order]),
                            "scaler_mean": float(scaler_mean[order]),
                            "scaler_scale": float(scaler_scale[order]),
                        }
                    )
                if isinstance(estimator, LogisticRegression):
                    for class_index, class_label in enumerate(estimator.classes_):
                        coefficient_rows.append(
                            {
                                "fold_month": month,
                                "algorithm": algorithm,
                                "horizon": horizon,
                                "class_value": int(class_label),
                                "feature_order": -1,
                                "feature": "__intercept__",
                                "coefficient": float(estimator.intercept_[class_index]),
                            }
                        )
                        coefficient_rows.extend(
                            {
                                "fold_month": month,
                                "algorithm": algorithm,
                                "horizon": horizon,
                                    "class_value": int(class_label),
                                "feature_order": order,
                                "feature": feature,
                                "coefficient": float(coefficient),
                            }
                            for order, (feature, coefficient) in enumerate(
                                zip(names, estimator.coef_[class_index], strict=True)
                            )
                        )
                fold_rows.append(
                    {
                        "fold_month": month,
                        "algorithm": algorithm,
                        "horizon": horizon,
                        "train_rows": int(len(train)),
                        "score_rows": int(len(score)),
                        "train_symbols": int(train["symbol_norm"].nunique()),
                        "train_symbol_sessions": int(
                            train.groupby(
                                ["symbol_norm", "session_date"], sort=False
                            ).ngroups
                        ),
                        "maximum_training_path_end_timestamp": (
                            train["path_end_timestamp"]
                        ).max(),
                        "minimum_scoring_timestamp": score["decision_timestamp"].min(),
                        "same_score_month_training_label_rows": int(
                            train["month_key"].eq(month).sum()
                        ),
                        "prior_completed_validation_month_training_label_rows": int(
                            train["month_key"].isin(VALIDATION_MONTHS).sum()
                        ),
                        "same_score_month_outcomes_read_before_probability": False,
                        "class_0_rows": int((target == 0).sum()),
                        "class_1_rows": int((target == 1).sum()),
                        "class_2_rows": int((target == 2).sum()),
                        "weight_min": float(weights.min()),
                        "weight_max": float(weights.max()),
                        "fitted_iterations": (
                            int(estimator.n_iter_)
                            if isinstance(estimator, HistGradientBoostingClassifier)
                            else int(estimator.n_iter_[0])
                        ),
                    }
                )
                fitted[(month, horizon, algorithm)] = {
                    "estimator": estimator,
                    "medians": medians,
                    "scaler": scaler,
                    "features": names,
                }

    ledger = pd.concat(probability_parts, ignore_index=True).sort_values(
        ["algorithm", "horizon", "symbol_norm", "session_date", "decision_timestamp"],
        kind="stable",
    ).reset_index(drop=True)
    return (
        ledger,
        pd.DataFrame(preprocessing_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(fold_rows),
        fitted,
    )


def derive_thresholds(probabilities: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for horizon in HORIZONS:
            selected = probabilities.loc[
                probabilities["algorithm"].eq(algorithm)
                & probabilities["horizon"].eq(horizon)
            ]
            for month in VALIDATION_MONTHS:
                prior = f"2024-{int(month[-2:]) - 1:02d}"
                calibration = selected.loc[selected["fold_month"].eq(prior)].copy()
                weights = equal_symbol_session_weights(calibration)
                for side, column in (
                    ("long", "p_long_first"),
                    ("short", "p_short_first"),
                ):
                    rows.append(
                        {
                            "score_month": month,
                            "threshold_source_month": prior,
                            "algorithm": algorithm,
                            "horizon": int(horizon),
                            "side": side,
                            "fire_threshold": weighted_inverse_cdf_quantile(
                                calibration[column],
                                weights,
                                FIRE_QUANTILE,
                                calibration["anchor_id"],
                            ),
                            "rearm_threshold": weighted_inverse_cdf_quantile(
                                calibration[column],
                                weights,
                                REARM_QUANTILE,
                                calibration["anchor_id"],
                            ),
                            "source_rows": int(len(calibration)),
                            "source_weight_sum": float(weights.sum()),
                            "fire_quantile": FIRE_QUANTILE,
                            "rearm_quantile": REARM_QUANTILE,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["algorithm", "horizon", "score_month", "side"], kind="stable"
    ).reset_index(drop=True)


def history_availability_bucket(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return np.where(array < 0.5, 0, np.where(array < 1.0, 1, 2)).astype(np.int8)


def derive_hysteresis_onsets(
    probabilities: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = probabilities.loc[
        probabilities["fold_month"].isin(VALIDATION_MONTHS)
        & probabilities["algorithm"].isin(CANDIDATES)
    ].copy()
    threshold_index = thresholds.set_index(
        ["score_month", "algorithm", "horizon", "side"]
    )
    onset_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    group_keys = [
        "algorithm", "horizon", "symbol_norm", "session_date", "segment_index"
    ]
    for group_key, group in validation.groupby(group_keys, sort=False):
        algorithm, horizon, symbol, session_date, segment_index = group_key
        long_armed = True
        short_armed = True
        for row in group.sort_values("decision_timestamp", kind="stable").itertuples(
            index=False
        ):
                month = str(row.fold_month)
                long_threshold = threshold_index.loc[
                    (month, algorithm, horizon, "long")
                ]
                short_threshold = threshold_index.loc[
                    (month, algorithm, horizon, "short")
                ]
                long_before, short_before = long_armed, short_armed
                if not long_armed and row.p_long_first < float(
                    long_threshold["rearm_threshold"]
                ):
                    long_armed = True
                if not short_armed and row.p_short_first < float(
                    short_threshold["rearm_threshold"]
                ):
                    short_armed = True
                long_rearmed = (not long_before) and long_armed
                short_rearmed = (not short_before) and short_armed
                long_fire = bool(
                    long_armed
                    and row.p_long_first >= float(long_threshold["fire_threshold"])
                    and row.p_long_first > row.p_short_first
                )
                short_fire = bool(
                    short_armed
                    and row.p_short_first >= float(short_threshold["fire_threshold"])
                    and row.p_short_first > row.p_long_first
                )
                conflict = long_fire and short_fire
                emitted: str | None = None
                onset_id: str | None = None
                if not conflict and long_fire:
                    emitted = "long"
                    long_armed = False
                elif not conflict and short_fire:
                    emitted = "short"
                    short_armed = False
                if emitted is not None:
                    onset_id = f"{algorithm}|{row.anchor_id}|{emitted}"
                    chosen = row.p_long_first if emitted == "long" else row.p_short_first
                    opposite = row.p_short_first if emitted == "long" else row.p_long_first
                    onset_rows.append(
                        {
                            "onset_id": onset_id,
                            "anchor_id": row.anchor_id,
                            "candidate_algorithm": algorithm,
                            "horizon": int(horizon),
                            "side": emitted,
                            "fold_month": month,
                            "symbol_norm": symbol,
                            "session_date": session_date,
                            "decision_timestamp": row.decision_timestamp,
                            "bar_ordinal": int(row.bar_ordinal),
                            "segment_index": int(segment_index),
                            "segment_position": int(row.segment_position),
                            "causal_scale_bps": float(row.causal_scale_bps),
                            "availability_12": float(row.availability_12),
                            "availability_bucket": int(row.availability_bucket),
                            "clock_bin_15": int(row.clock_bin_15),
                            "clock_bin_30": int(row.clock_bin_30),
                            "p_no_entry": float(row.p_no_entry),
                            "p_long_first": float(row.p_long_first),
                            "p_short_first": float(row.p_short_first),
                            "chosen_probability": float(chosen),
                            "opposite_probability": float(opposite),
                        }
                    )
                state_rows.append(
                    {
                        "anchor_id": row.anchor_id,
                        "algorithm": algorithm,
                        "horizon": int(horizon),
                        "fold_month": month,
                        "symbol_norm": symbol,
                        "session_date": session_date,
                        "decision_timestamp": row.decision_timestamp,
                        "segment_index": int(segment_index),
                        "long_armed_before": long_before,
                        "short_armed_before": short_before,
                        "long_rearmed": long_rearmed,
                        "short_rearmed": short_rearmed,
                        "long_fire": long_fire,
                        "short_fire": short_fire,
                        "conflict": conflict,
                        "emitted_side": emitted,
                        "onset_id": onset_id,
                        "long_armed_after": long_armed,
                        "short_armed_after": short_armed,
                    }
                )
    onsets = pd.DataFrame(onset_rows).sort_values("onset_id", kind="stable").reset_index(drop=True)
    states = pd.DataFrame(state_rows).sort_values(
        ["algorithm", "horizon", "symbol_norm", "session_date", "decision_timestamp"],
        kind="stable",
    ).reset_index(drop=True)
    if onsets["onset_id"].duplicated().any():
        raise AssertionError("duplicate onset")
    return onsets, states


def surface_feature_rows(surfaces: dict[int, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        [
            surface.loc[
                surface["month_key"].isin(VALIDATION_MONTHS),
                ["anchor_id", "horizon", *FULL_FEATURES],
            ]
            for surface in surfaces.values()
        ],
        ignore_index=True,
    )


def extract_reasons(
    onsets: pd.DataFrame,
    surfaces: dict[int, pd.DataFrame],
    fitted: dict[tuple[str, int, str], dict[str, Any]],
) -> pd.DataFrame:
    reason_features = surface_feature_rows(surfaces).drop(
        columns=["availability_12"]
    )
    joined = onsets.merge(
        reason_features,
        on=["anchor_id", "horizon"],
        how="left",
        validate="many_to_one",
    )
    if joined.loc[:, FULL_FEATURES].isna().all(axis=1).any():
        raise AssertionError("onset feature join failed")
    reason_rows: list[dict[str, Any]] = []
    for (month, horizon, algorithm, side), group in joined.groupby(
        ["fold_month", "horizon", "candidate_algorithm", "side"], sort=False
    ):
        model = fitted[(str(month), int(horizon), str(algorithm))]
        estimator = model["estimator"]
        medians = np.asarray(model["medians"], dtype=float)
        raw = group.loc[:, FULL_FEATURES].to_numpy(float)
        design = impute(raw, medians)
        chosen = CLASS_FOR_SIDE[str(side)]
        opposite = OPPOSITE_CLASS[str(side)]
        if algorithm == "full_logit":
            scaler = model["scaler"]
            if not isinstance(scaler, StandardScaler):
                raise AssertionError("logit reason scaler missing")
            standardized = scaler.transform(design)
            class_to_row = {
                int(value): index for index, value in enumerate(estimator.classes_)
            }
            chosen_row = class_to_row[chosen]
            opposite_row = class_to_row[opposite]
            difference = (
                estimator.coef_[chosen_row] - estimator.coef_[opposite_row]
            )
            intercept = float(
                estimator.intercept_[chosen_row] - estimator.intercept_[opposite_row]
            )
            feature_contributions = standardized * difference[None, :]
            group_values = np.column_stack(
                [
                    feature_contributions[:, GROUP_INDICES[name]].sum(axis=1)
                    for name in GROUP_ORDER
                ]
            )
            reconstructed = intercept + group_values.sum(axis=1)
            decision = estimator.decision_function(standardized)
            expected_margin = decision[:, chosen_row] - decision[:, opposite_row]
            if not np.allclose(reconstructed, expected_margin, rtol=1e-11, atol=1e-11):
                raise AssertionError("logit reason contributions do not reconstruct margin")
            baseline_margin = expected_margin
            explanation_kind = "exact_standardized_logit_margin_contribution"
        elif algorithm == "full_hgb":
            probabilities = estimator.predict_proba(design)
            class_to_row = {
                int(value): index for index, value in enumerate(estimator.classes_)
            }
            chosen_row = class_to_row[chosen]
            opposite_row = class_to_row[opposite]
            baseline_margin = (
                probabilities[:, chosen_row] - probabilities[:, opposite_row]
            )
            group_columns: list[np.ndarray] = []
            for name in GROUP_ORDER:
                perturbed = design.copy()
                indices = GROUP_INDICES[name]
                perturbed[:, indices] = medians[indices][None, :]
                alternative = estimator.predict_proba(perturbed)
                alternative_margin = (
                    alternative[:, chosen_row] - alternative[:, opposite_row]
                )
                group_columns.append(baseline_margin - alternative_margin)
            group_values = np.column_stack(group_columns)
            intercept = math.nan
            reconstructed = np.full(len(group), np.nan)
            explanation_kind = "local_median_replacement_probability_margin_sensitivity"
        else:
            raise AssertionError(f"unexpected reason algorithm {algorithm}")

        for row_index, onset in enumerate(group.itertuples(index=False)):
            ranking = sorted(
                range(len(GROUP_ORDER)),
                key=lambda index: (
                    -abs(float(group_values[row_index, index])), GROUP_ORDER[index]
                ),
            )
            rank_by_index = {
                index: rank + 1 for rank, index in enumerate(ranking[:3])
            }
            for group_index, reason_group in enumerate(GROUP_ORDER):
                value = float(group_values[row_index, group_index])
                direction = (
                    "supports_chosen"
                    if value > 1e-15
                    else "opposes_chosen"
                    if value < -1e-15
                    else "neutral"
                )
                reason_rows.append(
                    {
                        "onset_id": onset.onset_id,
                        "anchor_id": onset.anchor_id,
                        "candidate_algorithm": algorithm,
                        "horizon": int(horizon),
                        "side": side,
                        "symbol_norm": onset.symbol_norm,
                        "fold_month": month,
                        "reason_group": reason_group,
                        "plain_language": REASON_TEXT[reason_group],
                        "contribution_type": explanation_kind,
                        "contribution_value": value,
                        "contribution_direction": direction,
                        "top_rank": rank_by_index.get(group_index),
                        "is_top_group": group_index in rank_by_index,
                        "directional_intercept": float(intercept),
                        "directional_margin": float(baseline_margin[row_index]),
                        "reconstruction_error": float(
                            reconstructed[row_index] - baseline_margin[row_index]
                        ),
                    }
                )
    reasons = pd.DataFrame(reason_rows).sort_values(
        ["onset_id", "reason_group"], kind="stable"
    ).reset_index(drop=True)
    if len(reasons) != len(onsets) * len(GROUP_ORDER):
        raise AssertionError("reason row support mismatch")
    return reasons


def recurring_reason_summary(reasons: pd.DataFrame) -> pd.DataFrame:
    top = reasons.loc[reasons["is_top_group"]].copy()
    recurring = (
        top.groupby(
            [
                "candidate_algorithm", "horizon", "side", "reason_group",
                "contribution_direction",
            ],
            sort=False,
        )
        .agg(
            onset_count=("onset_id", "size"),
            months=("fold_month", "nunique"),
            stocks=("symbol_norm", "nunique"),
        )
        .reset_index()
    )
    recurring["recurring_observable_sign"] = (
        recurring["months"].ge(5)
        & recurring["stocks"].ge(15)
        & recurring["contribution_direction"].ne("neutral")
    )
    return recurring


def select_matched_clock_controls(
    onsets: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    clock = probabilities.loc[
        probabilities["algorithm"].eq("clock_logit")
        & probabilities["fold_month"].isin(VALIDATION_MONTHS)
    ].copy()
    rows: list[dict[str, Any]] = []
    for (algorithm, horizon, side), group in onsets.groupby(
        ["candidate_algorithm", "horizon", "side"], sort=False
    ):
        used: set[str] = set()
        side_probability = (
            "p_long_first" if side == "long" else "p_short_first"
        )
        horizon_pool = clock.loc[clock["horizon"].eq(horizon)].copy()
        local_pools = {
            key: selected.sort_values(
                [side_probability, "anchor_id"],
                ascending=[False, True],
                kind="stable",
            ).reset_index(drop=True)
            for key, selected in horizon_pool.groupby(
                ["symbol_norm", "fold_month"], sort=False
            )
        }
        ordered_candidates = group.sort_values("onset_id", kind="stable")
        for onset in ordered_candidates.itertuples(index=False):
            pool = local_pools[(onset.symbol_norm, onset.fold_month)]
            base = ~pool["anchor_id"].eq(onset.anchor_id) & ~pool["anchor_id"].isin(used)
            tier_masks = (
                (
                    base
                    & pool["session_date"].ne(onset.session_date)
                    & pool["clock_bin_15"].eq(onset.clock_bin_15)
                    & pool["availability_bucket"].eq(onset.availability_bucket)
                ),
                (
                    base
                    & pool["clock_bin_30"].eq(onset.clock_bin_30)
                    & pool["availability_bucket"].eq(onset.availability_bucket)
                ),
                base & pool["clock_bin_30"].eq(onset.clock_bin_30),
                base,
            )
            selected: pd.Series | None = None
            selected_tier = -1
            for tier, mask in enumerate(tier_masks):
                eligible = pool.loc[mask]
                if eligible.empty:
                    continue
                selected = eligible.iloc[0]
                selected_tier = tier
                break
            base_row = {
                "onset_id": onset.onset_id,
                "candidate_algorithm": algorithm,
                "horizon": int(horizon),
                "side": side,
                "candidate_anchor_id": onset.anchor_id,
                "candidate_symbol_norm": onset.symbol_norm,
                "candidate_session_date": onset.session_date,
                "candidate_fold_month": onset.fold_month,
                "candidate_bar_ordinal": int(onset.bar_ordinal),
                "control_anchor_id": None,
                "control_symbol_norm": None,
                "control_session_date": None,
                "control_decision_timestamp": pd.NaT,
                "control_bar_ordinal": None,
                "control_clock_probability": math.nan,
                "match_tier": selected_tier,
                "matched": False,
            }
            if selected is None:
                rows.append(base_row)
                continue
            control_id = str(selected["anchor_id"])
            used.add(control_id)
            rows.append(
                {
                    **base_row,
                    "control_anchor_id": control_id,
                    "control_symbol_norm": str(selected["symbol_norm"]),
                    "control_session_date": selected["session_date"],
                    "control_decision_timestamp": selected["decision_timestamp"],
                    "control_bar_ordinal": int(selected["bar_ordinal"]),
                    "control_clock_probability": float(selected[side_probability]),
                    "match_tier": int(selected_tier),
                    "matched": True,
                }
            )
    controls = pd.DataFrame(rows).sort_values("onset_id", kind="stable").reset_index(drop=True)
    duplicate = controls.loc[controls["matched"]].duplicated(
        ["candidate_algorithm", "horizon", "side", "control_anchor_id"]
    )
    if duplicate.any():
        raise AssertionError("matched controls reused within a candidate family")
    return controls


def path_outcomes(surfaces: dict[int, pd.DataFrame]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    base_columns = [
        "anchor_id", "symbol_norm", "session_date", "month_key",
        "decision_timestamp", "bar_ordinal", "horizon", "causal_scale_bps",
        "target_class", "status", "first_confirmation_step", "upside_mfe_bps",
        "downside_mfe_bps",
    ]
    for surface in surfaces.values():
        selected = surface.loc[
            surface["month_key"].isin(VALIDATION_MONTHS), base_columns
        ].copy()
        full = surface.loc[
            surface["month_key"].isin(VALIDATION_MONTHS)
        ].reset_index(drop=True)
        scale = selected["causal_scale_bps"].to_numpy(float)
        selected["upside_mfe_scale_units"] = (
            selected["upside_mfe_bps"].to_numpy(float) / scale
        )
        selected["downside_mfe_scale_units"] = (
            selected["downside_mfe_bps"].to_numpy(float) / scale
        )
        selected["long_pre_confirmation_adverse_scale_units"] = (
            full["prefix_downside_bps"].to_numpy(float) / scale
        )
        selected["short_pre_confirmation_adverse_scale_units"] = (
            full["prefix_upside_bps"].to_numpy(float) / scale
        )
        parts.append(selected)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["horizon", "anchor_id"], kind="stable"
    ).reset_index(drop=True)


def _oriented_outcome_values(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    result = frame.copy()
    original_columns = list(result.columns)
    chosen = CLASS_FOR_SIDE[side]
    opposite = OPPOSITE_CLASS[side]
    target = result["target_class"].to_numpy(int)
    result["conservative_correct"] = (target == chosen).astype(np.int8)
    result["wrong_first"] = (target == opposite).astype(np.int8)
    result["no_hit"] = result["status"].eq("no_hit_by_horizon").astype(np.int8)
    result["ambiguous"] = result["status"].eq("intrabar_ambiguous").astype(np.int8)
    if not (
        result[["conservative_correct", "wrong_first", "no_hit", "ambiguous"]]
        .sum(axis=1)
        .eq(1)
        .all()
    ):
        raise AssertionError("oriented status partition failed")
    if side == "long":
        favourable = result["upside_mfe_scale_units"].to_numpy(float)
        adverse = result["downside_mfe_scale_units"].to_numpy(float)
        prefix_adverse = result[
            "long_pre_confirmation_adverse_scale_units"
        ].to_numpy(float)
    else:
        favourable = result["downside_mfe_scale_units"].to_numpy(float)
        adverse = result["upside_mfe_scale_units"].to_numpy(float)
        prefix_adverse = result[
            "short_pre_confirmation_adverse_scale_units"
        ].to_numpy(float)
    result["favourable_excursion_scale_units"] = favourable
    result["adverse_excursion_scale_units"] = adverse
    result["pre_confirmation_adverse_scale_units"] = prefix_adverse
    result["directional_dominance_scale_units"] = favourable - adverse
    result["rapid_correct"] = (
        result["conservative_correct"].eq(1)
        & result["first_confirmation_step"].le(3)
    ).astype(np.int8)
    result["resolved"] = (
        result["conservative_correct"].eq(1) | result["wrong_first"].eq(1)
    ).astype(np.int8)
    result["clock_quartile"] = np.minimum(
        result["bar_ordinal"].to_numpy(int) * 4 // SESSION_BARS, 3
    ).astype(np.int8)
    added = [
        "conservative_correct", "wrong_first", "no_hit", "ambiguous",
        "resolved", "rapid_correct", "favourable_excursion_scale_units",
        "adverse_excursion_scale_units", "pre_confirmation_adverse_scale_units",
        "directional_dominance_scale_units", "clock_quartile",
    ]
    return result.loc[:, [*original_columns, *added]]


def score_onsets_and_controls(
    onsets: pd.DataFrame,
    controls: pd.DataFrame,
    surfaces: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome = path_outcomes(surfaces)
    path_columns = [
        "anchor_id", "target_class", "status", "first_confirmation_step",
        "upside_mfe_scale_units", "downside_mfe_scale_units",
        "long_pre_confirmation_adverse_scale_units",
        "short_pre_confirmation_adverse_scale_units",
    ]
    candidate = onsets.merge(
        outcome[path_columns], on="anchor_id", how="left", validate="one_to_one"
    )
    candidate_parts = [
        _oriented_outcome_values(candidate.loc[candidate["side"].eq(side)], side)
        for side in SIDES
    ]
    scored_onsets = pd.concat(candidate_parts, ignore_index=True).sort_values(
        "onset_id", kind="stable"
    ).reset_index(drop=True)
    matched = controls.loc[controls["matched"]].copy()
    control_paths = outcome[path_columns].rename(
        columns={"anchor_id": "control_anchor_id"}
    )
    scored_controls = matched.merge(
        control_paths, on="control_anchor_id", how="left", validate="one_to_one"
    ).rename(
        columns={
            "candidate_symbol_norm": "symbol_norm",
            "candidate_session_date": "session_date",
            "candidate_fold_month": "fold_month",
            "candidate_bar_ordinal": "bar_ordinal",
        }
    )
    control_parts = [
        _oriented_outcome_values(
            scored_controls.loc[scored_controls["side"].eq(side)], side
        )
        for side in SIDES
    ]
    scored_controls = pd.concat(control_parts, ignore_index=True).sort_values(
        "onset_id", kind="stable"
    ).reset_index(drop=True)
    event_columns = [
        "onset_id", "conservative_correct", "wrong_first", "no_hit", "ambiguous",
        "rapid_correct", "favourable_excursion_scale_units",
        "adverse_excursion_scale_units", "pre_confirmation_adverse_scale_units",
        "directional_dominance_scale_units", "first_confirmation_step",
    ]
    pairs = scored_onsets[
        [
            "onset_id", "candidate_algorithm", "horizon", "side", "symbol_norm",
            "session_date", "fold_month", "clock_quartile", *event_columns[1:],
        ]
    ].merge(
        scored_controls[event_columns],
        on="onset_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_control"),
    )
    return scored_onsets, scored_controls, pairs


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    value = np.asarray(list(values), dtype=float)
    weight = np.asarray(list(weights), dtype=float)
    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    if not valid.any():
        return math.nan
    return float(np.average(value[valid], weights=weight[valid]))


def multiclass_probability_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "multiclass_log_loss": math.nan,
            "multiclass_brier": math.nan,
            "top_class_accuracy": math.nan,
            "macro_ovr_auc": math.nan,
        }
    weights = equal_symbol_session_weights(frame)
    actual = frame["target_class"].to_numpy(int)
    probability = frame[
        ["p_no_entry", "p_long_first", "p_short_first"]
    ].to_numpy(float)
    selected_probability = probability[np.arange(len(frame)), actual]
    one_hot = np.eye(3, dtype=float)[actual]
    log_loss = weighted_mean(-np.log(np.clip(selected_probability, 1e-15, 1.0)), weights)
    brier = weighted_mean(np.sum((probability - one_hot) ** 2, axis=1), weights)
    accuracy = weighted_mean((np.argmax(probability, axis=1) == actual).astype(float), weights)
    try:
        auc = float(
            roc_auc_score(
                actual,
                probability,
                labels=[0, 1, 2],
                average="macro",
                multi_class="ovr",
                sample_weight=weights,
            )
        )
    except ValueError:
        auc = math.nan
    return {
        "rows": int(len(frame)),
        "multiclass_log_loss": log_loss,
        "multiclass_brier": brier,
        "top_class_accuracy": accuracy,
        "macro_ovr_auc": auc,
    }


def probability_outputs(
    probabilities: pd.DataFrame,
    surfaces: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = path_outcomes(surfaces)[
        ["anchor_id", "horizon", "target_class", "status"]
    ]
    scored = probabilities.loc[
        probabilities["fold_month"].isin(VALIDATION_MONTHS)
    ].merge(outcomes, on=["anchor_id", "horizon"], how="left", validate="many_to_one")
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for (algorithm, horizon), group in scored.groupby(["algorithm", "horizon"], sort=False):
        slices: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all", group)]
        slices.extend(
            ("month", str(key), selected)
            for key, selected in group.groupby("fold_month", sort=False)
        )
        slices.extend(
            ("stock", str(key), selected)
            for key, selected in group.groupby("symbol_norm", sort=False)
        )
        slices.extend(
            (
                "leave_one_stock_out",
                symbol,
                group.loc[~group["symbol_norm"].eq(symbol)],
            )
            for symbol in SYMBOLS
        )
        for slice_type, slice_value, selected in slices:
            metric_rows.append(
                {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    **multiclass_probability_summary(selected),
                }
            )
        weights = equal_symbol_session_weights(group)
        actual = group["target_class"].to_numpy(int)
        for class_label, probability_column in enumerate(
            ("p_no_entry", "p_long_first", "p_short_first")
        ):
            predicted = group[probability_column].to_numpy(float)
            bucket = np.minimum((predicted * 10.0).astype(int), 9)
            for bin_index in range(10):
                selected = bucket == bin_index
                calibration_rows.append(
                    {
                        "algorithm": algorithm,
                        "horizon": int(horizon),
                        "class_value": class_label,
                        "probability_bin": bin_index,
                        "lower_bound": bin_index / 10.0,
                        "upper_bound": (bin_index + 1) / 10.0,
                        "rows": int(selected.sum()),
                        "weighted_mean_probability": weighted_mean(
                            predicted[selected], weights[selected]
                        ),
                        "weighted_observed_rate": weighted_mean(
                            (actual[selected] == class_label).astype(float),
                            weights[selected],
                        ),
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(calibration_rows),
    )


EVENT_METRICS = (
    "correct_first",
    "wrong_first",
    "no_hit",
    "ambiguous",
    "rapid_correct_confirmation_within_3_bars",
    "favourable_excursion_scale_units",
    "adverse_excursion_scale_units",
    "pre_confirmation_adverse_scale_units",
    "directional_dominance_scale_units",
)


def _event_role_summary(
    frame: pd.DataFrame,
    prefix: str,
    weights: np.ndarray,
) -> dict[str, Any]:
    values = {
        metric: frame[f"{prefix}_{metric}"].to_numpy(float)
        for metric in EVENT_METRICS
    }
    correct = weighted_mean(values["correct_first"], weights)
    wrong = weighted_mean(values["wrong_first"], weights)
    resolved_denominator = correct + wrong
    confirmation = frame[f"{prefix}_first_confirmation_step"].to_numpy(float)
    return {
        "conservative_correct_first_precision": correct,
        "wrong_first_rate": wrong,
        "no_hit_rate": weighted_mean(values["no_hit"], weights),
        "ambiguous_rate": weighted_mean(values["ambiguous"], weights),
        "resolved_only_precision": (
            correct / resolved_denominator if resolved_denominator > 0 else math.nan
        ),
        "rapid_correct_confirmation_within_3_rate": weighted_mean(
            values["rapid_correct_confirmation_within_3_bars"], weights
        ),
        "mean_favourable_excursion_scale_units": weighted_mean(
            values["favourable_excursion_scale_units"], weights
        ),
        "mean_adverse_excursion_scale_units": weighted_mean(
            values["adverse_excursion_scale_units"], weights
        ),
        "mean_pre_confirmation_adverse_scale_units": weighted_mean(
            values["pre_confirmation_adverse_scale_units"], weights
        ),
        "mean_directional_dominance_scale_units": weighted_mean(
            values["directional_dominance_scale_units"], weights
        ),
        "mean_first_confirmation_step_resolved": weighted_mean(
            confirmation, weights
        ),
    }


def paired_event_summary(frame: pd.DataFrame) -> dict[str, Any]:
    weights = equal_symbol_session_weights(
        frame.rename(
            columns={
                "candidate_symbol_norm": "symbol_norm",
                "candidate_session_date": "session_date",
            }
        )
    )
    candidate = _event_role_summary(frame, "candidate", weights)
    control = _event_role_summary(frame, "control", weights)
    candidate_precision = candidate["conservative_correct_first_precision"]
    control_precision = control["conservative_correct_first_precision"]
    return {
        "matched_pairs": int(len(frame)),
        **{f"candidate_{key}": value for key, value in candidate.items()},
        **{f"control_{key}": value for key, value in control.items()},
        "absolute_precision_lift": candidate_precision - control_precision,
        "relative_precision_lift": (
            candidate_precision / control_precision - 1.0
            if control_precision > 0
            else math.nan
        ),
        "rapid_success_lift": (
            candidate["rapid_correct_confirmation_within_3_rate"]
            - control["rapid_correct_confirmation_within_3_rate"]
        ),
        "directional_dominance_lift": (
            candidate["mean_directional_dominance_scale_units"]
            - control["mean_directional_dominance_scale_units"]
        ),
        "pre_confirmation_adverse_not_worse_lift": (
            control["mean_pre_confirmation_adverse_scale_units"]
            - candidate["mean_pre_confirmation_adverse_scale_units"]
        ),
        "confirmation_speed_lift_bars": (
            control["mean_first_confirmation_step_resolved"]
            - candidate["mean_first_confirmation_step_resolved"]
        ),
    }


def event_outputs(
    onsets: pd.DataFrame,
    controls: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    for (algorithm, horizon, side), onset_group in onsets.groupby(
        ["algorithm", "horizon", "side"], sort=False
    ):
        control_group = controls.loc[
            controls["candidate_algorithm"].eq(algorithm)
            & controls["horizon"].eq(horizon)
            & controls["side"].eq(side)
        ]
        pair_group = pairs.loc[
            pairs["candidate_algorithm"].eq(algorithm)
            & pairs["horizon"].eq(horizon)
            & pairs["side"].eq(side)
        ]
        base = {
            "candidate_algorithm": algorithm,
            "horizon": int(horizon),
            "side": side,
            "onsets": int(len(onset_group)),
            "onset_stocks": int(onset_group["symbol_norm"].nunique()),
            "matched_pairs": int(len(pair_group)),
            "matched_control_rate": float(control_group["matched"].mean()),
            **paired_event_summary(pair_group),
        }
        rows.append(base)
        for month in VALIDATION_MONTHS:
            month_onsets = onset_group.loc[onset_group["month_key"].eq(month)]
            month_pairs = pair_group.loc[
                pair_group["candidate_month_key"].eq(month)
            ]
            month_rows.append(
                {
                    "candidate_algorithm": algorithm,
                    "horizon": int(horizon),
                    "side": side,
                    "month": month,
                    "onsets": int(len(month_onsets)),
                    "onset_stocks": int(month_onsets["symbol_norm"].nunique()),
                    **paired_event_summary(month_pairs),
                }
            )
        for deleted in SYMBOLS:
            selected = pair_group.loc[
                pair_group["candidate_symbol_norm"].ne(deleted)
            ]
            deletion_rows.append(
                {
                    "candidate_algorithm": algorithm,
                    "horizon": int(horizon),
                    "side": side,
                    "deleted_symbol": deleted,
                    **paired_event_summary(selected),
                }
            )
        for quartile in range(4):
            quartiles = np.minimum(
                pair_group["candidate_bar_ordinal"].to_numpy(int) * 4
                // SESSION_BARS,
                3,
            )
            selected = pair_group.loc[quartiles == quartile]
            if selected.empty:
                continue
            slice_rows.append(
                {
                    "candidate_algorithm": algorithm,
                    "horizon": int(horizon),
                    "side": side,
                    "slice_kind": "clock_quartile",
                    "slice_value": quartile,
                    **paired_event_summary(selected),
                }
            )
    return (
        pd.DataFrame(rows).sort_values(
            ["candidate_algorithm", "horizon", "side"], kind="stable"
        ).reset_index(drop=True),
        pd.DataFrame(month_rows).sort_values(
            ["candidate_algorithm", "horizon", "side", "month"], kind="stable"
        ).reset_index(drop=True),
        pd.DataFrame(deletion_rows).sort_values(
            ["candidate_algorithm", "horizon", "side", "deleted_symbol"],
            kind="stable",
        ).reset_index(drop=True),
        pd.DataFrame(slice_rows).sort_values(
            ["candidate_algorithm", "horizon", "side", "slice_kind", "slice_value"],
            kind="stable",
        ).reset_index(drop=True),
    )


def moving_block_interval(values: np.ndarray, offset: int) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if len(data) < BLOCK_LENGTH:
        raise AssertionError("insufficient session-date bootstrap support")
    starts = np.arange(len(data) - BLOCK_LENGTH + 1)
    blocks = math.ceil(len(data) / BLOCK_LENGTH)
    rng = np.random.default_rng(SEED + offset)
    sampled_starts = rng.choice(starts, size=(BOOTSTRAP_DRAWS, blocks), replace=True)
    positions = (
        sampled_starts[:, :, None] + np.arange(BLOCK_LENGTH)[None, None, :]
    ).reshape(BOOTSTRAP_DRAWS, -1)[:, : len(data)]
    sample_mean = data[positions].mean(axis=1)
    lower, upper = np.quantile(
        sample_mean, [LOWER_QUANTILE, UPPER_QUANTILE], method="linear"
    )
    return float(data.mean()), float(lower), float(upper)


def event_bootstraps(pairs: pd.DataFrame) -> pd.DataFrame:
    metric_columns = {
        "precision_lift": "lift_correct_first",
        "rapid_success_lift": "lift_rapid_correct_confirmation_within_3_bars",
        "directional_dominance_lift": "lift_directional_dominance_scale_units",
        "pre_confirmation_adverse_not_worse_lift": (
            "lift_pre_confirmation_adverse_not_worse"
        ),
    }
    rows: list[dict[str, Any]] = []
    offset = 0
    for (algorithm, horizon, side), group in pairs.groupby(
        ["candidate_algorithm", "horizon", "side"], sort=False
    ):
        for metric, column in metric_columns.items():
            symbol_day = group.groupby(
                ["candidate_session_date", "candidate_symbol_norm"], sort=False
            )[column].mean()
            daily = symbol_day.groupby("candidate_session_date", sort=True).mean()
            observed, lower, upper = moving_block_interval(
                daily.to_numpy(float), offset
            )
            rows.append(
                {
                    "candidate_algorithm": algorithm,
                    "horizon": int(horizon),
                    "side": side,
                    "metric": metric,
                    "session_dates": int(len(daily)),
                    "observed_mean": observed,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
            offset += 1
    return pd.DataFrame(rows).sort_values(
        ["candidate_algorithm", "horizon", "side", "metric"], kind="stable"
    ).reset_index(drop=True)


def independent_decision(
    probability_metrics: pd.DataFrame,
    event_metrics: pd.DataFrame,
    monthly_events: pd.DataFrame,
    deletion_events: pd.DataFrame,
    bootstraps: pd.DataFrame,
    onsets: pd.DataFrame,
    controls: pd.DataFrame,
    recurring_reasons: pd.DataFrame,
) -> dict[str, Any]:
    probability_hypotheses: dict[str, Any] = {}
    for algorithm in CANDIDATES:
        selected = probability_metrics.loc[
            probability_metrics["algorithm"].eq(algorithm)
        ]
        by_horizon = {
            str(int(row.horizon)): {
                "log_loss_better_than_clock": bool(
                    row.log_loss_improvement_vs_clock > 0
                ),
                "brier_better_than_clock": bool(
                    row.brier_improvement_vs_clock > 0
                ),
            }
            for row in selected.itertuples(index=False)
        }
        retained = bool(
            len(by_horizon) == len(HORIZONS)
            and all(all(gates.values()) for gates in by_horizon.values())
        )
        probability_hypotheses[algorithm] = {
            "retained_as_internal_probability_hypothesis": retained,
            "by_horizon": by_horizon,
        }

    entry_hypotheses: dict[str, Any] = {}
    for algorithm in CANDIDATES:
        algorithm_onsets = onsets.loc[onsets["algorithm"].eq(algorithm)]
        candidate_support: dict[str, Any] = {}
        for horizon in HORIZONS:
            horizon_onsets = algorithm_onsets.loc[
                algorithm_onsets["horizon"].eq(horizon)
            ]
            month_counts = horizon_onsets.groupby("month_key", sort=True).size()
            candidate_support[str(horizon)] = {
                "minimum_onsets": bool(len(horizon_onsets) >= 500),
                "minimum_monthly_onsets": bool(
                    all(int(month_counts.get(month, 0)) >= 50 for month in VALIDATION_MONTHS)
                ),
                "minimum_stocks": bool(horizon_onsets["symbol_norm"].nunique() >= 15),
            }
        side_results: dict[str, Any] = {}
        for side in SIDES:
            by_horizon: dict[str, Any] = {}
            for horizon in HORIZONS:
                row = event_metrics.loc[
                    event_metrics["candidate_algorithm"].eq(algorithm)
                    & event_metrics["horizon"].eq(horizon)
                    & event_metrics["side"].eq(side)
                ].iloc[0]
                probability = probability_metrics.loc[
                    probability_metrics["algorithm"].eq(algorithm)
                    & probability_metrics["horizon"].eq(horizon)
                ].iloc[0]
                month = monthly_events.loc[
                    monthly_events["candidate_algorithm"].eq(algorithm)
                    & monthly_events["horizon"].eq(horizon)
                    & monthly_events["side"].eq(side)
                ]
                deletion = deletion_events.loc[
                    deletion_events["candidate_algorithm"].eq(algorithm)
                    & deletion_events["horizon"].eq(horizon)
                    & deletion_events["side"].eq(side)
                ]
                bootstrap = bootstraps.loc[
                    bootstraps["candidate_algorithm"].eq(algorithm)
                    & bootstraps["horizon"].eq(horizon)
                    & bootstraps["side"].eq(side)
                ].set_index("metric")
                bootstrap_metrics = (
                    "precision_lift",
                    "rapid_success_lift",
                    "directional_dominance_lift",
                    "pre_confirmation_adverse_not_worse_lift",
                )
                gates = {
                    **candidate_support[str(horizon)],
                    "minimum_side_onsets": bool(row.onsets >= 100),
                    "minimum_side_stocks": bool(row.onset_stocks >= 10),
                    "minimum_match_rate": bool(row.matched_control_rate >= 0.95),
                    "probability_log_loss_better_than_clock": bool(
                        probability.log_loss_improvement_vs_clock > 0
                    ),
                    "probability_brier_better_than_clock": bool(
                        probability.brier_improvement_vs_clock > 0
                    ),
                    "minimum_absolute_precision_lift": bool(
                        row.absolute_precision_lift >= 0.05
                    ),
                    "minimum_relative_precision_lift": bool(
                        row.relative_precision_lift >= 0.10
                    ),
                    "minimum_rapid_success_lift": bool(
                        row.rapid_success_lift >= 0.02
                    ),
                    "positive_directional_dominance_lift": bool(
                        row.directional_dominance_lift > 0
                    ),
                    "pre_confirmation_adverse_not_worse": bool(
                        row.pre_confirmation_adverse_not_worse_lift >= 0
                    ),
                    "bootstrap_lower_bounds_positive": bool(
                        all(
                            metric in bootstrap.index
                            and float(bootstrap.loc[metric, "ci_lower"]) > 0
                            for metric in bootstrap_metrics
                        )
                    ),
                    "minimum_positive_precision_lift_months": bool(
                        month["absolute_precision_lift"].gt(0).sum() >= 4
                    ),
                    "minimum_positive_precision_lift_stock_deletions": bool(
                        deletion["absolute_precision_lift"].gt(0).sum() >= 18
                    ),
                }
                by_horizon[str(horizon)] = {
                    "passed": bool(all(gates.values())),
                    "gates": gates,
                }
            side_results[side] = {
                "retained_as_internal_entry_sign_hypothesis": bool(
                    all(item["passed"] for item in by_horizon.values())
                ),
                "by_horizon": by_horizon,
            }
        entry_hypotheses[algorithm] = side_results

    recurring = recurring_reasons.loc[
        recurring_reasons["recurring_positive_sign"]
        | recurring_reasons["recurring_negative_sign"]
    ]
    retained_probability = [
        name
        for name, value in probability_hypotheses.items()
        if value["retained_as_internal_probability_hypothesis"]
    ]
    retained_entry = [
        f"{algorithm}:{side}"
        for algorithm, sides in entry_hypotheses.items()
        for side, value in sides.items()
        if value["retained_as_internal_entry_sign_hypothesis"]
    ]
    return {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "probability_hypotheses": probability_hypotheses,
        "entry_onset_hypotheses": entry_hypotheses,
        "recurring_reason_rows": int(len(recurring)),
        "retained_probability_hypotheses": retained_probability,
        "retained_entry_onset_hypotheses": retained_entry,
        "interpretation": (
            "Any pass is only a recurring 2024 internal entry-sign hypothesis "
            "for genuinely new shadow sessions; it is not a strategy or edge claim."
        ),
    }


def compare_frames(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    label: str,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-10,
) -> float:
    if list(observed.columns) != list(expected.columns):
        raise AssertionError(
            f"{label} columns differ: {list(observed.columns)!r} != {list(expected.columns)!r}"
        )
    if len(observed) != len(expected):
        raise AssertionError(f"{label} length differs: {len(observed)} != {len(expected)}")
    maximum = 0.0
    for column in expected.columns:
        left = observed[column].reset_index(drop=True)
        right = expected[column].reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(left) or pd.api.types.is_datetime64_any_dtype(right):
            left_time = pd.to_datetime(left, utc=True, errors="coerce")
            right_time = pd.to_datetime(right, utc=True, errors="coerce")
            if not left_time.equals(right_time):
                raise AssertionError(f"{label}:{column} timestamp mismatch")
        elif pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_value = left.to_numpy(float)
            right_value = right.to_numpy(float)
            if not np.allclose(
                left_value, right_value, rtol=rtol, atol=atol, equal_nan=True
            ):
                difference = np.abs(left_value - right_value)
                finite = np.isfinite(difference)
                position = int(np.nanargmax(difference)) if finite.any() else -1
                raise AssertionError(
                    f"{label}:{column} numerical mismatch near {position}: "
                    f"observed={left_value[position]!r} expected={right_value[position]!r}"
                )
            difference = np.abs(left_value - right_value)
            finite_values = difference[np.isfinite(difference)]
            if len(finite_values):
                maximum = max(maximum, float(finite_values.max()))
        else:
            left_text = left.where(left.notna(), "<NA>").astype(str)
            right_text = right.where(right.notna(), "<NA>").astype(str)
            if not left_text.equals(right_text):
                mismatch = int(
                    np.flatnonzero(
                        left_text.to_numpy(str) != right_text.to_numpy(str)
                    )[0]
                )
                raise AssertionError(f"{label}:{column} text mismatch at {mismatch}")
    return maximum


def probability_comparisons_independent(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["horizon", "slice_type", "slice_value"]
    baseline = metrics.loc[metrics["algorithm"].eq("clock_logit")].set_index(keys)
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_rows = metrics.loc[metrics["algorithm"].eq(candidate)].set_index(keys)
        if not candidate_rows.index.equals(baseline.index):
            candidate_rows = candidate_rows.reindex(baseline.index)
        for key, candidate_row in candidate_rows.iterrows():
            clock = baseline.loc[key]
            rows.append(
                {
                    "candidate_algorithm": candidate,
                    "horizon": int(key[0]),
                    "slice_type": key[1],
                    "slice_value": key[2],
                    "rows": int(candidate_row["rows"]),
                    "log_loss_improvement_vs_clock": float(
                        clock["multiclass_log_loss"]
                        - candidate_row["multiclass_log_loss"]
                    ),
                    "brier_improvement_vs_clock": float(
                        clock["multiclass_brier"] - candidate_row["multiclass_brier"]
                    ),
                    "accuracy_lift_vs_clock": float(
                        candidate_row["top_class_accuracy"]
                        - clock["top_class_accuracy"]
                    ),
                    "auc_lift_vs_clock": float(
                        candidate_row["macro_ovr_auc"] - clock["macro_ovr_auc"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def event_statistics_independent(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "onsets": 0,
            "stocks": 0,
            "conservative_correct_first_precision": math.nan,
            "wrong_first_rate": math.nan,
            "no_hit_rate": math.nan,
            "ambiguous_rate": math.nan,
            "resolved_only_precision": math.nan,
            "rapid_correct_within_3_rate": math.nan,
            "mean_favourable_excursion_scale_units": math.nan,
            "mean_adverse_excursion_scale_units": math.nan,
            "mean_pre_confirmation_adverse_scale_units": math.nan,
            "mean_directional_dominance_scale_units": math.nan,
            "mean_first_confirmation_step": math.nan,
        }
    weights = equal_symbol_session_weights(frame)
    correct = frame["conservative_correct"].to_numpy(float)
    resolved = correct + frame["wrong_first"].to_numpy(float)
    denominator = float(np.sum(weights * resolved))
    return {
        "onsets": int(len(frame)),
        "stocks": int(frame["symbol_norm"].nunique()),
        "conservative_correct_first_precision": weighted_mean(correct, weights),
        "wrong_first_rate": weighted_mean(frame["wrong_first"], weights),
        "no_hit_rate": weighted_mean(frame["no_hit"], weights),
        "ambiguous_rate": weighted_mean(frame["ambiguous"], weights),
        "resolved_only_precision": (
            float(np.sum(weights * correct) / denominator)
            if denominator > 0
            else math.nan
        ),
        "rapid_correct_within_3_rate": weighted_mean(frame["rapid_correct"], weights),
        "mean_favourable_excursion_scale_units": weighted_mean(
            frame["favourable_excursion_scale_units"], weights
        ),
        "mean_adverse_excursion_scale_units": weighted_mean(
            frame["adverse_excursion_scale_units"], weights
        ),
        "mean_pre_confirmation_adverse_scale_units": weighted_mean(
            frame["pre_confirmation_adverse_scale_units"], weights
        ),
        "mean_directional_dominance_scale_units": weighted_mean(
            frame["directional_dominance_scale_units"], weights
        ),
        "mean_first_confirmation_step": weighted_mean(
            frame["first_confirmation_step"], weights
        ),
    }


def independent_event_slices(
    group: pd.DataFrame,
) -> Iterable[tuple[str, str, pd.DataFrame]]:
    yield "pooled", "all", group
    for key, selected in group.groupby("fold_month", sort=False):
        yield "month", str(key), selected
    for key, selected in group.groupby("symbol_norm", sort=False):
        yield "stock", str(key), selected
    for symbol in SYMBOLS:
        yield "leave_one_stock_out", symbol, group.loc[
            ~group["symbol_norm"].eq(symbol)
        ]
    for key, selected in group.groupby("clock_quartile", sort=False):
        yield "clock_quartile", str(int(key)), selected


def evaluate_events_independent(
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    for sample_kind, frame in (
        ("candidate", candidates),
        ("matched_clock_control", controls),
    ):
        for (algorithm, horizon, side), group in frame.groupby(
            ["candidate_algorithm", "horizon", "side"], sort=False
        ):
            for slice_type, slice_value, subset in independent_event_slices(group):
                metric_rows.append(
                    {
                        "candidate_algorithm": algorithm,
                        "horizon": int(horizon),
                        "side": side,
                        "sample_kind": sample_kind,
                        "slice_type": slice_type,
                        "slice_value": slice_value,
                        **event_statistics_independent(subset),
                    }
                )
    work = pairs.copy()
    work["precision_lift"] = (
        work["conservative_correct_candidate"]
        - work["conservative_correct_control"]
    )
    work["rapid_success_lift"] = (
        work["rapid_correct_candidate"] - work["rapid_correct_control"]
    )
    work["directional_dominance_lift"] = (
        work["directional_dominance_scale_units_candidate"]
        - work["directional_dominance_scale_units_control"]
    )
    work["pre_confirmation_adverse_improvement"] = (
        work["pre_confirmation_adverse_scale_units_control"]
        - work["pre_confirmation_adverse_scale_units_candidate"]
    )
    work["favourable_excursion_lift"] = (
        work["favourable_excursion_scale_units_candidate"]
        - work["favourable_excursion_scale_units_control"]
    )
    lift_rows: list[dict[str, Any]] = []
    for (algorithm, horizon, side), group in work.groupby(
        ["candidate_algorithm", "horizon", "side"], sort=False
    ):
        for slice_type, slice_value, subset in independent_event_slices(group):
            weights = equal_symbol_session_weights(subset) if not subset.empty else np.asarray([])
            candidate_precision = (
                weighted_mean(subset["conservative_correct_candidate"], weights)
                if not subset.empty
                else math.nan
            )
            control_precision = (
                weighted_mean(subset["conservative_correct_control"], weights)
                if not subset.empty
                else math.nan
            )
            lift_rows.append(
                {
                    "candidate_algorithm": algorithm,
                    "horizon": int(horizon),
                    "side": side,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "matched_pairs": int(len(subset)),
                    "candidate_precision": candidate_precision,
                    "control_precision": control_precision,
                    "precision_lift": candidate_precision - control_precision,
                    "relative_precision_lift": (
                        (candidate_precision - control_precision) / control_precision
                        if control_precision > 0
                        else math.nan
                    ),
                    "rapid_success_lift": weighted_mean(
                        subset["rapid_success_lift"], weights
                    ) if not subset.empty else math.nan,
                    "directional_dominance_lift": weighted_mean(
                        subset["directional_dominance_lift"], weights
                    ) if not subset.empty else math.nan,
                    "pre_confirmation_adverse_improvement": weighted_mean(
                        subset["pre_confirmation_adverse_improvement"], weights
                    ) if not subset.empty else math.nan,
                    "favourable_excursion_lift": weighted_mean(
                        subset["favourable_excursion_lift"], weights
                    ) if not subset.empty else math.nan,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(lift_rows)


def paired_interval_independent(
    frame: pd.DataFrame,
    value_column: str,
    random_state: int,
) -> dict[str, Any]:
    symbol_day = (
        frame.groupby(["session_date", "symbol_norm"], sort=True)[value_column]
        .mean()
        .unstack("symbol_norm")
        .sort_index()
        .reindex(columns=sorted(frame["symbol_norm"].unique()))
    )
    matrix = symbol_day.to_numpy(float)
    n_dates = len(matrix)
    if n_dates < BLOCK_LENGTH:
        raise AssertionError("insufficient bootstrap dates")
    observed = float(np.nanmean(np.nanmean(matrix, axis=0)))
    rng = np.random.default_rng(random_state)
    blocks_needed = math.ceil(n_dates / BLOCK_LENGTH)
    start_max = n_dates - BLOCK_LENGTH
    finite = np.isfinite(matrix)
    filled = np.where(finite, matrix, 0.0)
    samples = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, start_max + 1, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + BLOCK_LENGTH) for start in starts]
        )[:n_dates]
        counts = np.bincount(indices, minlength=n_dates).astype(float)
        numerator = (filled * counts[:, None]).sum(axis=0)
        denominator = (finite * counts[:, None]).sum(axis=0)
        symbol_values = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        samples[draw] = np.nanmean(symbol_values)
    return {
        "observed": observed,
        "lower": float(np.quantile(samples, LOWER_QUANTILE)),
        "upper": float(np.quantile(samples, UPPER_QUANTILE)),
        "draws": BOOTSTRAP_DRAWS,
    }


def bootstraps_independent(pairs: pd.DataFrame) -> pd.DataFrame:
    work = pairs.copy()
    work["precision_lift"] = work["conservative_correct_candidate"] - work["conservative_correct_control"]
    work["rapid_success_lift"] = work["rapid_correct_candidate"] - work["rapid_correct_control"]
    work["directional_dominance_lift"] = work["directional_dominance_scale_units_candidate"] - work["directional_dominance_scale_units_control"]
    work["pre_confirmation_adverse_improvement"] = work["pre_confirmation_adverse_scale_units_control"] - work["pre_confirmation_adverse_scale_units_candidate"]
    rows: list[dict[str, Any]] = []
    metrics = (
        "precision_lift", "rapid_success_lift", "directional_dominance_lift",
        "pre_confirmation_adverse_improvement",
    )
    for (algorithm, horizon, side), group in work.groupby(
        ["candidate_algorithm", "horizon", "side"], sort=False
    ):
        for metric_index, metric in enumerate(metrics):
            interval = paired_interval_independent(
                group,
                metric,
                SEED + int(horizon) * 100 + (0 if side == "long" else 10)
                + (0 if algorithm == "full_logit" else 1) + metric_index,
            )
            rows.append(
                {
                    "candidate_algorithm": algorithm,
                    "horizon": int(horizon),
                    "side": side,
                    "metric": metric,
                    **interval,
                    "lower_quantile": LOWER_QUANTILE,
                    "upper_quantile": UPPER_QUANTILE,
                    "block_sessions": BLOCK_LENGTH,
                }
            )
    return pd.DataFrame(rows)


def match_summary_independent(onsets: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    counts = onsets.groupby(
        ["candidate_algorithm", "horizon", "side"], sort=False
    ).size().rename("onsets")
    matched = controls.groupby(
        ["candidate_algorithm", "horizon", "side"], sort=False
    )["matched"].sum().rename("matched_controls")
    summary = pd.concat([counts, matched], axis=1).fillna(0).reset_index()
    summary["onsets"] = summary["onsets"].astype(int)
    summary["matched_controls"] = summary["matched_controls"].astype(int)
    summary["match_rate"] = summary["matched_controls"] / summary["onsets"]
    return summary


def reason_dictionary_independent() -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "artifact_filename_semantics": (
            "pre_outcome is legacy shorthand for pre_final_evaluation_join; "
            "prior completed validation-month labels may have trained later folds"
        ),
        "interpretation": {
            "positive": "the causal OOF fitted-model sensitivity supports the emitted side",
            "negative": "the causal OOF fitted-model sensitivity opposes the emitted side",
            "zero": "no local directional contribution at machine precision",
            "full_logit": "exact additive standardized chosen-versus-opposite logit contribution",
            "full_hgb": "local probability-margin sensitivity to replacing the group by fold-training medians; not additive and not causal",
        },
        "groups": {
            group: {
                "features": list(features),
                "plain_language": REASON_TEXT[group],
            }
            for group, features in REASON_GROUPS.items()
        },
        "top_groups_per_onset": 3,
        "recurring_month_requirement": 5,
        "recurring_stock_requirement": 15,
    }


def final_decisions_independent(
    comparisons: pd.DataFrame,
    onsets: pd.DataFrame,
    match_summary: pd.DataFrame,
    event_lifts: pd.DataFrame,
    bootstraps: pd.DataFrame,
    recurring_reasons: pd.DataFrame,
) -> dict[str, Any]:
    probability: dict[str, Any] = {}
    for algorithm in CANDIDATES:
        detail: dict[str, Any] = {}
        all_horizons = True
        for horizon in HORIZONS:
            subset = comparisons.loc[
                comparisons["candidate_algorithm"].eq(algorithm)
                & comparisons["horizon"].eq(horizon)
            ]
            pooled = subset.loc[subset["slice_type"].eq("pooled")].iloc[0]
            months = subset.loc[subset["slice_type"].eq("month")]
            deletions = subset.loc[
                subset["slice_type"].eq("leave_one_stock_out")
            ]
            gates = {
                "pooled_log_loss_better": bool(
                    pooled["log_loss_improvement_vs_clock"] > 0
                ),
                "pooled_brier_better": bool(
                    pooled["brier_improvement_vs_clock"] > 0
                ),
                "months_both_better_at_least_4": int(
                    (
                        months["log_loss_improvement_vs_clock"].gt(0)
                        & months["brier_improvement_vs_clock"].gt(0)
                    ).sum()
                ) >= 4,
                "stock_deletions_both_better_at_least_18": int(
                    (
                        deletions["log_loss_improvement_vs_clock"].gt(0)
                        & deletions["brier_improvement_vs_clock"].gt(0)
                    ).sum()
                ) >= 18,
            }
            passed = all(gates.values())
            detail[f"h{horizon}"] = {
                "passed": passed,
                "gates": gates,
                "log_loss_improvement_vs_clock": float(
                    pooled["log_loss_improvement_vs_clock"]
                ),
                "brier_improvement_vs_clock": float(
                    pooled["brier_improvement_vs_clock"]
                ),
            }
            all_horizons &= passed
        probability[algorithm] = {
            "retained": all_horizons,
            "interpretation": (
                "2024 internal probability hypothesis only"
                if all_horizons
                else "rejected as a recurring probability hypothesis"
            ),
            "horizons": detail,
        }

    candidate_side: dict[str, Any] = {}
    for algorithm in CANDIDATES:
        for side in SIDES:
            key = f"{algorithm}_{side}"
            horizon_detail: dict[str, Any] = {}
            all_horizons = bool(probability[algorithm]["retained"])
            for horizon in HORIZONS:
                all_onsets = onsets.loc[
                    onsets["candidate_algorithm"].eq(algorithm)
                    & onsets["horizon"].eq(horizon)
                ]
                side_onsets = all_onsets.loc[all_onsets["side"].eq(side)]
                pooled = event_lifts.loc[
                    event_lifts["candidate_algorithm"].eq(algorithm)
                    & event_lifts["horizon"].eq(horizon)
                    & event_lifts["side"].eq(side)
                    & event_lifts["slice_type"].eq("pooled")
                ].iloc[0]
                months = event_lifts.loc[
                    event_lifts["candidate_algorithm"].eq(algorithm)
                    & event_lifts["horizon"].eq(horizon)
                    & event_lifts["side"].eq(side)
                    & event_lifts["slice_type"].eq("month")
                ]
                deletions = event_lifts.loc[
                    event_lifts["candidate_algorithm"].eq(algorithm)
                    & event_lifts["horizon"].eq(horizon)
                    & event_lifts["side"].eq(side)
                    & event_lifts["slice_type"].eq("leave_one_stock_out")
                ]
                match = match_summary.loc[
                    match_summary["candidate_algorithm"].eq(algorithm)
                    & match_summary["horizon"].eq(horizon)
                    & match_summary["side"].eq(side)
                ].iloc[0]
                interval = bootstraps.loc[
                    bootstraps["candidate_algorithm"].eq(algorithm)
                    & bootstraps["horizon"].eq(horizon)
                    & bootstraps["side"].eq(side)
                ].set_index("metric")
                monthly_total = all_onsets.groupby("fold_month").size()
                gates = {
                    "candidate_onsets_at_least_500": len(all_onsets) >= 500,
                    "every_month_candidate_onsets_at_least_50": bool(
                        monthly_total.reindex(VALIDATION_MONTHS, fill_value=0).ge(50).all()
                    ),
                    "candidate_stocks_at_least_15": all_onsets["symbol_norm"].nunique() >= 15,
                    "side_onsets_at_least_100": len(side_onsets) >= 100,
                    "side_stocks_at_least_10": side_onsets["symbol_norm"].nunique() >= 10,
                    "matched_control_rate_at_least_0_95": float(match["match_rate"]) >= 0.95,
                    "absolute_precision_lift_at_least_0_05": float(pooled["precision_lift"]) >= 0.05,
                    "relative_precision_lift_at_least_0_10": float(pooled["relative_precision_lift"]) >= 0.10,
                    "rapid_success_lift_at_least_0_02": float(pooled["rapid_success_lift"]) >= 0.02,
                    "directional_dominance_lift_positive": float(pooled["directional_dominance_lift"]) > 0,
                    "pre_confirmation_adverse_not_worse": float(pooled["pre_confirmation_adverse_improvement"]) >= 0,
                    "bootstrap_precision_positive": float(interval.loc["precision_lift", "lower"]) > 0,
                    "bootstrap_rapid_positive": float(interval.loc["rapid_success_lift", "lower"]) > 0,
                    "bootstrap_dominance_positive": float(interval.loc["directional_dominance_lift", "lower"]) > 0,
                    "bootstrap_adverse_not_worse": float(interval.loc["pre_confirmation_adverse_improvement", "lower"]) >= 0,
                    "positive_precision_lift_months_at_least_4": int(months["precision_lift"].gt(0).sum()) >= 4,
                    "positive_precision_lift_stock_deletions_at_least_18": int(deletions["precision_lift"].gt(0).sum()) >= 18,
                }
                passed = all(gates.values())
                horizon_detail[f"h{horizon}"] = {
                    "passed": passed,
                    "gates": gates,
                    "onsets": len(side_onsets),
                    "matched_control_rate": float(match["match_rate"]),
                    "precision_lift": float(pooled["precision_lift"]),
                    "rapid_success_lift": float(pooled["rapid_success_lift"]),
                    "directional_dominance_lift": float(pooled["directional_dominance_lift"]),
                    "pre_confirmation_adverse_improvement": float(
                        pooled["pre_confirmation_adverse_improvement"]
                    ),
                }
                all_horizons &= passed
            candidate_side[key] = {
                "retained": all_horizons,
                "interpretation": (
                    "recurring 2024 internal entry-sign hypothesis; prospective shadow required"
                    if all_horizons
                    else "rejected or descriptive only"
                ),
                "horizons": horizon_detail,
            }
    recurring = (
        recurring_reasons.loc[recurring_reasons["recurring_observable_sign"]]
        if not recurring_reasons.empty
        else recurring_reasons
    )
    return {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "economic_edge_claim_permitted": False,
        "probability_hypotheses": probability,
        "candidate_side_hypotheses": candidate_side,
        "recurring_observable_signs": json_safe(recurring.to_dict("records")),
        "any_entry_sign_retained": any(
            item["retained"] for item in candidate_side.values()
        ),
    }


def main() -> None:
    if not ARTIFACT_ROOT.is_dir():
        raise FileNotFoundError(f"scored artifacts are absent: {ARTIFACT_ROOT}")
    checks: list[dict[str, Any]] = []

    def record(name: str, detail: Any) -> None:
        checks.append({"name": name, "pass": True, "detail": json_safe(detail)})

    contract = json.loads(CONTRACT_PATH.read_text())
    assert_contract(contract)
    record("frozen_contract_and_research_safety", True)
    record("runner_ast_raw_ohlc_later_period_and_economic_boundary", ast_source_boundary(RUNNER_PATH))

    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    if not (
        pre_score["contract_id"] == CONTRACT_ID
        and pre_score["research_only"] is True
        and pre_score["live_ordering_enabled"] is False
        and pre_score["order_placement"] == "disabled"
        and pre_score["frozen_before_scoring"] is True
        and pre_score["provider_volume_label"] == "historical_volume_not_used"
        and pre_score["scientific_status"]
        == "2024_internal_monthly_expanding_oof_entry_sign_discovery"
        and pre_score["later_period_outcomes_read"] is False
        and pre_score["validation_outcomes_read_before_manifest_freeze"] is False
        and pre_score[
            "same_score_month_outcomes_read_before_that_month_probability"
        ] is False
        and pre_score[
            "prior_completed_validation_month_path_labels_permitted_for_later_folds"
        ] is True
        and pre_score[
            "global_bundle_expected_before_final_all_fold_evaluation_join"
        ] is True
        and pre_score[
            "global_bundle_expected_before_any_validation_outcome_is_read"
        ] is False
    ):
        raise AssertionError("pre-score identity/safety/freeze mismatch")
    if pd.Timestamp(pre_score["frozen_at_utc"]).tzinfo is None:
        raise AssertionError("pre-score timestamp is not timezone-aware")
    actual_sources = {name: sha256_file(path) for name, path in source_paths().items()}
    if pre_score["sha256"] != actual_sources:
        raise AssertionError("whole source/provider hashes changed after pre-score freeze")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if pre_score["environment_versions"] != versions:
        raise AssertionError("environment version drift")
    record("whole_provider_runner_contract_helper_environment_hashes", len(actual_sources))

    tape, diagnostics = load_regular_tape()
    features = reconstruct_features(tape)
    scale = reconstruct_lagged_scale(tape)
    if list(features.columns) != list(FULL_FEATURES):
        raise AssertionError("independent feature order drift")
    record(
        "exact_2024_regular_tape_segments_and_lagged_scale",
        {
            "rows": len(tape),
            "sessions": diagnostics["union_sessions"],
            "symbol_sessions": diagnostics["symbol_sessions"],
            "gaps": diagnostics["within_session_nonfive_minute_gaps"],
            "first_eligible_segment_position": int(
                tape.loc[scale.notna(), "segment_position"].min()
            ),
        },
    )
    record("sampled_explicit_feature_prefix_causality", prefix_causality_check(tape, features))

    surfaces = {
        horizon: build_horizon_surface(tape, features, scale, horizon)
        for horizon in HORIZONS
    }
    record(
        "exact_target_blind_scale_and_path_support",
        {str(horizon): len(surface) for horizon, surface in surfaces.items()},
    )
    probabilities, preprocessing, coefficients, folds, fitted = replay_models(surfaces)
    thresholds = derive_thresholds(probabilities)
    onsets, states = derive_hysteresis_onsets(probabilities, thresholds)
    controls = select_matched_clock_controls(onsets, probabilities)
    reasons = extract_reasons(onsets, surfaces, fitted)
    recurring = recurring_reason_summary(reasons)

    pre_errors = {
        "probabilities": compare_frames(
            pd.read_parquet(ARTIFACT_ROOT / "probabilities_pre_outcome.parquet"),
            probabilities,
            "probabilities_pre_outcome",
            atol=1e-8,
        ),
        "thresholds": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "thresholds_pre_outcome.csv"),
            thresholds,
            "thresholds_pre_outcome",
            atol=1e-10,
        ),
        "states": compare_frames(
            pd.read_parquet(ARTIFACT_ROOT / "onset_state_ledger_pre_outcome.parquet"),
            states,
            "onset_state_pre_outcome",
            atol=1e-8,
        ),
        "onsets": compare_frames(
            pd.read_parquet(ARTIFACT_ROOT / "candidate_onsets_pre_outcome.parquet"),
            onsets,
            "candidate_onsets_pre_outcome",
            atol=1e-8,
        ),
        "controls": compare_frames(
            pd.read_parquet(ARTIFACT_ROOT / "matched_clock_controls_pre_outcome.parquet"),
            controls,
            "matched_controls_pre_outcome",
            atol=1e-8,
        ),
        "reasons": compare_frames(
            pd.read_parquet(ARTIFACT_ROOT / "onset_reasons_pre_outcome.parquet"),
            reasons,
            "onset_reasons_pre_outcome",
            atol=1e-8,
        ),
        "recurring": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "recurring_reason_summary_pre_outcome.csv"),
            recurring,
            "recurring_reasons_pre_outcome",
            atol=1e-10,
        ),
        "preprocessing": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "fold_preprocessing.csv"),
            preprocessing,
            "fold_preprocessing",
            atol=1e-10,
        ),
        "coefficients": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "logit_coefficients.csv"),
            coefficients,
            "logit_coefficients",
            atol=1e-9,
        ),
        "folds": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "fold_metadata.csv"),
            folds,
            "fold_metadata",
            atol=1e-10,
        ),
    }
    record("exact_folds_weights_preprocessing_models_probabilities", pre_errors)
    if not folds["same_score_month_training_label_rows"].eq(0).all():
        raise AssertionError("same-score-month label entered its own fold")
    if not folds["same_score_month_outcomes_read_before_probability"].eq(False).all():
        raise AssertionError("same-score-month outcome chronology changed")
    if not folds[
        "prior_completed_validation_month_training_label_rows"
    ].gt(0).any():
        raise AssertionError("later expanding folds did not consume prior completed labels")
    record(
        "exact_per_fold_oof_chronology_and_prior_month_expansion",
        {
            "same_month_training_labels": int(
                folds["same_score_month_training_label_rows"].sum()
            ),
            "maximum_prior_validation_training_labels": int(
                folds[
                    "prior_completed_validation_month_training_label_rows"
                ].max()
            ),
        },
    )
    record(
        "exact_prior_month_thresholds_hysteresis_onsets_controls_reasons",
        {
            "probability_rows": len(probabilities),
            "onsets": len(onsets),
            "matched_controls": int(controls["matched"].sum()),
            "reason_rows": len(reasons),
        },
    )
    if json.loads(
        (ARTIFACT_ROOT / "reason_dictionary_pre_outcome.json").read_text()
    ) != reason_dictionary_independent():
        raise AssertionError("reason dictionary mismatch")
    record("exact_frozen_plain_language_reason_dictionary", True)

    freeze_path = ARTIFACT_ROOT / "prediction_onset_control_reason_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if not (
        freeze["stage"]
        == "global_bundle_written_before_final_all_fold_evaluation_join"
        and freeze["artifact_filename_semantics"]
        == (
            "pre_outcome is legacy shorthand for pre_final_evaluation_join; "
            "prior completed validation-month labels trained later folds"
        )
        and freeze["validation_paths_present"] is False
        and freeze["final_all_fold_evaluation_path_table_present"] is False
        and freeze[
            "same_score_month_outcomes_read_before_that_month_probability"
        ] is False
        and freeze[
            "prior_completed_validation_month_path_labels_used_in_later_folds"
        ] is True
        and freeze[
            "global_bundle_written_before_any_validation_outcome_is_read"
        ] is False
        and freeze[
            "global_bundle_written_before_final_all_fold_evaluation_join"
        ] is True
        and freeze["horizon_cooldown_used"] is False
        and freeze["terminal_return_or_economic_outcome_used"] is False
        and freeze["frozen_source_manifest_sha256"] == sha256_file(PRE_SCORE_PATH)
        and freeze["frozen_source_sha256"] == actual_sources
    ):
        raise AssertionError("pre-outcome freeze boundary mismatch")
    for name, expected_hash in freeze["artifact_sha256"].items():
        if sha256_file(ARTIFACT_ROOT / name) != expected_hash:
            raise AssertionError(f"pre-outcome artifact changed: {name}")
    freeze_hash = sha256_file(freeze_path)
    record("global_bundle_frozen_before_final_all_fold_evaluation_join", freeze_hash)

    paths = path_outcomes(surfaces)
    path_error = compare_frames(
        pd.read_parquet(ARTIFACT_ROOT / "validation_paths.parquet"),
        paths,
        "validation_paths",
        atol=1e-9,
    )
    record("exact_first_passage_classes_and_path_excursions", {"rows": len(paths), "error": path_error})

    probability_metrics, calibration = probability_outputs(probabilities, surfaces)
    comparisons = probability_comparisons_independent(probability_metrics)
    probability_errors = {
        "metrics": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "probability_metrics.csv"),
            probability_metrics,
            "probability_metrics",
        ),
        "calibration": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "probability_calibration.csv"),
            calibration,
            "probability_calibration",
        ),
        "comparisons": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "probability_comparisons.csv"),
            comparisons,
            "probability_comparisons",
        ),
    }
    record("exact_equal_symbol_session_probability_metrics_and_slices", probability_errors)

    candidates, scored_controls, pairs = score_onsets_and_controls(
        onsets, controls, surfaces
    )
    scored_errors = {
        "candidates": compare_frames(
            pd.read_parquet(ARTIFACT_ROOT / "scored_candidate_onsets.parquet"),
            candidates,
            "scored_candidates",
        ),
        "controls": compare_frames(
            pd.read_parquet(ARTIFACT_ROOT / "scored_matched_clock_controls.parquet"),
            scored_controls,
            "scored_controls",
        ),
        "pairs": compare_frames(
            pd.read_parquet(ARTIFACT_ROOT / "paired_event_evidence.parquet"),
            pairs,
            "paired_event_evidence",
        ),
    }
    record("exact_oriented_candidate_control_and_paired_path_evidence", scored_errors)

    event_metrics, event_lifts = evaluate_events_independent(
        candidates, scored_controls, pairs
    )
    bootstraps = bootstraps_independent(pairs)
    match_summary = match_summary_independent(onsets, controls)
    event_errors = {
        "metrics": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "event_metrics.csv"),
            event_metrics,
            "event_metrics",
        ),
        "lifts": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "event_lifts.csv"),
            event_lifts,
            "event_lifts",
        ),
        "bootstraps": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "event_bootstrap_intervals.csv"),
            bootstraps,
            "event_bootstraps",
        ),
        "matches": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "control_match_summary.csv"),
            match_summary,
            "control_match_summary",
        ),
    }
    record("exact_event_metrics_slices_lifts_and_equal_symbol_day_bootstraps", event_errors)

    decision = final_decisions_independent(
        comparisons, onsets, match_summary, event_lifts, bootstraps, recurring
    )
    if json.loads((ARTIFACT_ROOT / "decision.json").read_text()) != json_safe(decision):
        raise AssertionError("decision gates mismatch")
    record("exact_all_required_probability_and_candidate_side_gates", decision["any_entry_sign_retained"])

    source_binding = json.loads((ARTIFACT_ROOT / "source_hashes.json").read_text())
    expected_source_binding = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "sha256": actual_sources,
        "environment_versions": versions,
        "pre_score_manifest_sha256": sha256_file(PRE_SCORE_PATH),
        "fold_chronology": {
            "same_score_month_outcomes_read_before_that_month_probability": False,
            "prior_completed_validation_month_path_labels_used_in_later_folds": True,
            "global_bundle_written_before_any_validation_outcome_is_read": False,
            "global_bundle_written_before_final_all_fold_evaluation_join": True,
        },
        "pre_outcome_freeze_manifest_sha256": freeze_hash,
    }
    if source_binding != expected_source_binding:
        raise AssertionError("source binding artifact mismatch")
    record("exact_pre_score_pre_outcome_and_source_binding", True)

    summary = json.loads((ARTIFACT_ROOT / "summary.json").read_text())
    expected_summary_scalars = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "data": "regular-session five-minute provider OHLC; volume not read",
        "probability_rows": len(probabilities),
        "validation_path_rows": len(paths),
        "candidate_onsets": len(onsets),
        "matched_controls": int(controls["matched"].sum()),
        "pre_score_manifest_sha256": sha256_file(PRE_SCORE_PATH),
        "pre_outcome_freeze_manifest_sha256": freeze_hash,
        "fold_chronology": expected_source_binding["fold_chronology"],
        "artifact_filename_semantics": (
            "pre_outcome is legacy shorthand for pre_final_evaluation_join"
        ),
    }
    for key, expected in expected_summary_scalars.items():
        if summary[key] != expected:
            raise AssertionError(f"summary scalar mismatch: {key}")
    if summary["tape_diagnostics"] != json_safe(diagnostics):
        raise AssertionError("summary tape diagnostics mismatch")
    if summary["decisions"] != json_safe(decision):
        raise AssertionError("summary decision mismatch")
    for name, expected_hash in summary["result_artifact_sha256"].items():
        if sha256_file(ARTIFACT_ROOT / name) != expected_hash:
            raise AssertionError(f"summary result hash mismatch: {name}")
    record("exact_summary_diagnostics_decision_and_result_hashes", True)

    manifest_path = ARTIFACT_ROOT / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    actual_pre_audit_files = {
        path.name: sha256_file(path)
        for path in sorted(ARTIFACT_ROOT.iterdir())
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "independent_audit.json"}
    }
    if manifest["files_excluding_this_manifest"] != actual_pre_audit_files:
        raise AssertionError("pre-audit artifact manifest mismatch")
    if not (
        manifest["stage"] == "pre_independent_audit_complete_artifact_manifest"
        and manifest["pre_score_manifest_sha256"] == sha256_file(PRE_SCORE_PATH)
        and manifest["pre_outcome_freeze_manifest_sha256"] == freeze_hash
        and manifest["fold_chronology"]
        == expected_source_binding["fold_chronology"]
    ):
        raise AssertionError("pre-audit manifest binding mismatch")
    record("exact_complete_pre_audit_artifact_manifest", len(actual_pre_audit_files))

    result = {
        "audit": "raw_ohlc_entry_onset_discovery_v1_independent",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "scientific_status": "2024_internal_monthly_expanding_oof_entry_sign_discovery",
        "same_score_month_outcomes_read_before_that_month_probability": False,
        "prior_completed_validation_month_path_labels_used_in_later_folds": True,
        "global_bundle_written_before_any_validation_outcome_is_read": False,
        "global_bundle_written_before_final_all_fold_evaluation_join": True,
        "later_or_backward_periods_read": False,
        "forbidden_prior_detector_inputs_used": False,
        "terminal_return_or_pnl_evaluation_used": False,
        "all_passed": True,
        "passed": len(checks),
        "failed": 0,
        "checks": checks,
    }
    audit_path = ARTIFACT_ROOT / "independent_audit.json"
    audit_path.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n")
    refreshed_files = {
        path.name: sha256_file(path)
        for path in sorted(ARTIFACT_ROOT.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    refreshed_manifest = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "stage": "independent_audit_complete_artifact_manifest",
        "pre_score_manifest_sha256": sha256_file(PRE_SCORE_PATH),
        "pre_outcome_freeze_manifest_sha256": freeze_hash,
        "fold_chronology": expected_source_binding["fold_chronology"],
        "files_excluding_this_manifest": refreshed_files,
    }
    manifest_path.write_text(
        json.dumps(refreshed_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
