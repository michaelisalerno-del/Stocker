"""Research-only clean-slate causal OHLC entry experiment.

The experiment deliberately excludes regimes, states, loops, cycles, B0,
templates, named pattern flags, prior detector outputs, and volume.  It fits
three fixed algorithms on raw causal five-minute OHLC transforms and produces
monthly expanding out-of-fold predictions for July--December 2024.

The default scoring path is deliberately two-stage: predictions and fixed
actions are persisted and hashed before validation prices or targets are
attached.  ``--validate-only`` performs only source, bar, feature, and exact
support validation; it does not fit models, create predictions, attach
outcomes, or write result artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


WORK = Path(__file__).resolve().parent
CONTRACT_PATH = WORK / "contracts/20260712-clean-slate-causal-ohlc-entries-v1.json"
PRE_SCORE_PATH = (
    WORK / "contracts/20260712-clean-slate-causal-ohlc-entries-v1-pre-score.json"
)
RAW_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/"
    "instrument_type=stock"
)
ENVIRONMENT_ROOT = Path("/Users/michaelsalerno/StockerLocal")
OUT = Path("/private/tmp/stocker_clean_slate_causal_ohlc_entries_v1_20260712")

CONTRACT_ID = "clean_slate_causal_ohlc_entries_v1"
SEED = 20260712
SOURCE_COLUMNS = ("timestamp", "open", "high", "low", "close")
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
HORIZONS = (6, 12, 24)
VALIDATION_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
ACTION_THRESHOLDS = (10.0, 20.0, 40.0)
COSTS = (0, 1, 2, 5, 10)
PRIMARY_THRESHOLD = 10.0
PRIMARY_COST = 5
SESSION_BARS = 78
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5
EPSILON = 1e-12

EXPECTED_REGULAR_ROWS = 424583
EXPECTED_UNION_SESSIONS = 252
EXPECTED_SYMBOL_SESSIONS = 5539
EXPECTED_GAPS = 2612
EXPECTED_ROWS = {6: 383168, 12: 347620, 24: 280982}
EXPECTED_VALIDATION_ROWS = {6: 195292, 12: 177276, 24: 143472}

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
if len(FULL_FEATURES) != 40:
    raise AssertionError("the frozen full feature list must contain exactly 40 fields")

RIDGE_PARAMETERS = {
    "alpha": 10.0,
    "fit_intercept": True,
    "solver": "lsqr",
    "tol": 1e-6,
}
HGB_PARAMETERS = {
    "loss": "squared_error",
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
ALGORITHM_FEATURES = {
    "clock_ridge": CLOCK_FEATURES,
    "full_ridge": FULL_FEATURES,
    "full_hgb": FULL_FEATURES,
}
ALGORITHMS = tuple(ALGORITHM_FEATURES)
CANDIDATES = ("full_ridge", "full_hgb")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n")


def provider_path(symbol: str) -> Path:
    return RAW_ROOT / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def source_paths() -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "runner": Path(__file__).resolve(),
        "environment_pyproject": ENVIRONMENT_ROOT / "pyproject.toml",
        "environment_uv_lock": ENVIRONMENT_ROOT / "uv.lock",
    }
    for symbol in SYMBOLS:
        paths[f"provider_full_file_{symbol}"] = provider_path(symbol)
    return paths


def current_source_hashes() -> dict[str, str]:
    paths = source_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen source files: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AssertionError(
            f"contract drift for {label}: observed={observed!r}, expected={expected!r}"
        )


def load_contract_and_verify(
    require_pre_score: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    contract = json.loads(CONTRACT_PATH.read_text())
    _require_equal(contract["contract_id"], CONTRACT_ID, "contract_id")
    safety = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "broker_connection_enabled": False,
        "paper_or_demo_execution_enabled": False,
        "deployment_enabled": False,
        "strategy_promotion_permitted": False,
        "economic_edge_claim_permitted": False,
    }
    for key, expected in safety.items():
        _require_equal(contract[key], expected, key)
    _require_equal(contract["periods"]["data_year"], 2024, "data year")
    _require_equal(
        tuple(contract["periods"]["validation_months"]),
        VALIDATION_MONTHS,
        "validation months",
    )
    for year in (2023, 2025, 2026):
        _require_equal(
            contract["periods"][f"{year}_read_permitted"], False, f"{year} read"
        )
    _require_equal(
        tuple(contract["sources"]["columns_read"]), SOURCE_COLUMNS, "source columns"
    )
    _require_equal(
        contract["sources"]["provider_volume_label"],
        "historical_volume_not_used",
        "volume label",
    )
    _require_equal(tuple(contract["universe"]["symbols"]), SYMBOLS, "symbols")
    _require_equal(contract["universe"]["size"], len(SYMBOLS), "universe size")
    _require_equal(tuple(contract["cohort"]["horizons_bars"]), HORIZONS, "horizons")
    _require_equal(
        {
            int(key): int(value)
            for key, value in contract["cohort"][
                "expected_exact_2024_rows_by_horizon"
            ].items()
        },
        EXPECTED_ROWS,
        "exact cohort rows",
    )
    _require_equal(
        {
            int(key): int(value)
            for key, value in contract["cohort"][
                "expected_exact_validation_rows_by_horizon"
            ].items()
        },
        EXPECTED_VALIDATION_ROWS,
        "validation cohort rows",
    )
    _require_equal(
        tuple(contract["feature_policy"]["clock_features"]),
        CLOCK_FEATURES,
        "clock features",
    )
    _require_equal(
        tuple(contract["feature_policy"]["full_features_in_order"]),
        FULL_FEATURES,
        "full features",
    )
    _require_equal(contract["feature_policy"]["epsilon"], EPSILON, "epsilon")
    expected_algorithm_specs = [
        ("clock_ridge", "clock_features", "Ridge", RIDGE_PARAMETERS),
        ("full_ridge", "full_features_in_order", "Ridge", RIDGE_PARAMETERS),
        (
            "full_hgb",
            "full_features_in_order",
            "HistGradientBoostingRegressor",
            HGB_PARAMETERS,
        ),
    ]
    observed_algorithm_specs = [
        (item["name"], item["features"], item["estimator"], item["parameters"])
        for item in contract["algorithms"]
    ]
    _require_equal(observed_algorithm_specs, expected_algorithm_specs, "algorithms")
    _require_equal(
        float(contract["actions"]["primary_threshold_gross_bps"]),
        PRIMARY_THRESHOLD,
        "primary threshold",
    )
    _require_equal(
        tuple(
            float(value)
            for value in contract["actions"]["descriptive_thresholds_gross_bps"]
        ),
        ACTION_THRESHOLDS[1:],
        "descriptive thresholds",
    )
    _require_equal(tuple(contract["costs"]["grid_bps_per_side"]), COSTS, "cost grid")
    _require_equal(
        contract["costs"]["primary_bps_per_side"], PRIMARY_COST, "primary cost"
    )
    _require_equal(
        contract["metrics"]["inference"]["draws"],
        BOOTSTRAP_DRAWS,
        "bootstrap draws",
    )
    _require_equal(
        contract["metrics"]["inference"]["random_state"], SEED, "random state"
    )
    gate = contract["retention_gate_per_candidate"]
    _require_equal(
        gate["minimum_nonoverlapping_trades_each_horizon"], 500, "trade support gate"
    )
    _require_equal(
        gate["minimum_trades_each_validation_month_and_horizon"],
        50,
        "monthly support gate",
    )
    _require_equal(gate["minimum_stocks_each_horizon"], 15, "stock support gate")
    _require_equal(gate["minimum_long_trades_each_horizon"], 100, "long support gate")
    _require_equal(gate["minimum_short_trades_each_horizon"], 100, "short support gate")
    _require_equal(gate["minimum_stocks_each_side_and_horizon"], 10, "side stock gate")
    _require_equal(
        gate["minimum_relative_pooled_mse_improvement_vs_clock_each_horizon"],
        0.0025,
        "MSE gate",
    )
    _require_equal(
        gate["minimum_mse_improvement_months_each_horizon"], 4, "monthly MSE gate"
    )

    if not require_pre_score:
        return contract, None
    if not PRE_SCORE_PATH.is_file():
        raise FileNotFoundError(
            f"missing frozen pre-score manifest {PRE_SCORE_PATH}; run --validate-only, "
            "freeze exact source hashes, then score"
        )
    manifest = json.loads(PRE_SCORE_PATH.read_text())
    _require_equal(manifest["contract_id"], CONTRACT_ID, "pre-score contract id")
    for key, expected in {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }.items():
        _require_equal(manifest[key], expected, f"pre-score {key}")
    actual_hashes = current_source_hashes()
    _require_equal(manifest["sha256"], actual_hashes, "pre-score source hashes")
    if "environment_versions" in manifest:
        _require_equal(
            manifest["environment_versions"],
            environment_versions(),
            "pre-score environment versions",
        )
    return contract, manifest


def _year_filter() -> list[tuple[str, str, datetime]]:
    return [
        ("timestamp", ">=", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        ("timestamp", "<", datetime(2025, 1, 1, tzinfo=timezone.utc)),
    ]


def load_tape(
    symbols: Sequence[str] = SYMBOLS,
    *,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Load only 2024 rows and validate regular-session provider OHLC.

    Parquet timestamp predicates are mandatory: provider files span several
    years, and this function must not materialize 2023, 2025, or 2026 rows.
    """

    if tuple(symbols) != SYMBOLS:
        raise AssertionError("clean-slate universe drift")
    frames: list[pd.DataFrame] = []
    per_symbol: list[dict[str, Any]] = []
    for symbol in symbols:
        path = provider_path(symbol)
        frame = pd.read_parquet(
            path,
            columns=list(SOURCE_COLUMNS),
            filters=_year_filter(),
            engine="pyarrow",
        )
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], utc=True, errors="raise"
        )
        if frame.empty:
            raise AssertionError(f"no predicate-filtered 2024 rows for {symbol}")
        if (
            not frame["timestamp"].ge(pd.Timestamp("2024-01-01", tz="UTC")).all()
            or not frame["timestamp"].lt(pd.Timestamp("2025-01-01", tz="UTC")).all()
        ):
            raise AssertionError(f"non-2024 row materialized for {symbol}")
        duplicate_count = int(frame["timestamp"].duplicated(keep=False).sum())
        if duplicate_count:
            raise AssertionError(f"duplicate 2024 provider timestamps for {symbol}")
        prices = frame[["open", "high", "low", "close"]].to_numpy(float)
        finite_positive = np.isfinite(prices).all(axis=1) & (prices > 0.0).all(axis=1)
        order_valid = (prices[:, 2] <= np.minimum(prices[:, 0], prices[:, 3])) & (
            np.maximum(prices[:, 0], prices[:, 3]) <= prices[:, 1]
        )
        valid_ohlc = finite_positive & order_valid
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        local_minutes = local.dt.hour * 60 + local.dt.minute
        regular = local_minutes.ge(570) & local_minutes.lt(960)
        accepted = valid_ohlc & regular.to_numpy(bool)
        selected = frame.loc[accepted].copy()
        selected_local = selected["timestamp"].dt.tz_convert("America/New_York")
        selected_minutes = selected_local.dt.hour * 60 + selected_local.dt.minute
        if not ((selected_minutes - 570) % 5 == 0).all():
            raise AssertionError(f"off-grid regular-session timestamp for {symbol}")
        selected["symbol_norm"] = symbol
        selected["session_date"] = selected_local.dt.strftime("%Y-%m-%d")
        selected["month_key"] = selected["session_date"].str.slice(0, 7)
        selected["bar_ordinal"] = ((selected_minutes - 570) // 5).astype(np.int16)
        selected = selected.sort_values("timestamp", kind="stable").reset_index(
            drop=True
        )
        gap = (
            selected.groupby("session_date", sort=False)["timestamp"]
            .diff()
            .ne(pd.Timedelta(minutes=5))
        )
        first = selected.groupby("session_date", sort=False).cumcount().eq(0)
        within_session_gap = gap & ~first
        selected["segment_index"] = (
            gap.groupby(selected["session_date"], sort=False).cumsum().astype(np.int16)
            - 1
        )
        per_symbol.append(
            {
                "symbol_norm": symbol,
                "raw_predicate_2024_rows": int(len(frame)),
                "outside_regular_session_rows": int((~regular).sum()),
                "nonfinite_or_invalid_ohlc_rows": int((~valid_ohlc).sum()),
                "duplicate_timestamp_rows": duplicate_count,
                "accepted_regular_rows": int(len(selected)),
                "symbol_sessions": int(selected["session_date"].nunique()),
                "within_session_nonfive_minute_gaps": int(within_session_gap.sum()),
            }
        )
        frames.append(selected)
    tape = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol_norm", "session_date", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )
    tape["source_position"] = np.arange(len(tape), dtype=np.int64)
    segment_keys = ["symbol_norm", "session_date", "segment_index"]
    tape["segment_position"] = (
        tape.groupby(segment_keys, sort=False).cumcount().astype(np.int16)
    )
    tape["segment_size"] = (
        tape.groupby(segment_keys, sort=False)["timestamp"]
        .transform("size")
        .astype(np.int16)
    )
    diagnostics: dict[str, Any] = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "raw_predicate_2024_rows": int(
            sum(item["raw_predicate_2024_rows"] for item in per_symbol)
        ),
        "regular_valid_rows": int(len(tape)),
        "union_sessions": int(tape["session_date"].nunique()),
        "symbol_sessions": int(
            tape.groupby(["symbol_norm", "session_date"], sort=False).ngroups
        ),
        "within_session_nonfive_minute_gaps": int(
            sum(item["within_session_nonfive_minute_gaps"] for item in per_symbol)
        ),
        "skipped_support_reasons": {
            "nonfinite_or_invalid_ohlc": int(
                sum(item["nonfinite_or_invalid_ohlc_rows"] for item in per_symbol)
            ),
            "duplicate_timestamp": int(
                sum(item["duplicate_timestamp_rows"] for item in per_symbol)
            ),
            "outside_regular_session": int(
                sum(item["outside_regular_session_rows"] for item in per_symbol)
            ),
            "nonfive_minute_gap": int(
                sum(item["within_session_nonfive_minute_gaps"] for item in per_symbol)
            ),
            "insufficient_same_session_horizon": {},
        },
        "per_symbol": per_symbol,
    }
    _require_equal(
        diagnostics["regular_valid_rows"], EXPECTED_REGULAR_ROWS, "regular row count"
    )
    _require_equal(
        diagnostics["union_sessions"], EXPECTED_UNION_SESSIONS, "union session count"
    )
    _require_equal(
        diagnostics["symbol_sessions"], EXPECTED_SYMBOL_SESSIONS, "symbol-session count"
    )
    _require_equal(
        diagnostics["within_session_nonfive_minute_gaps"], EXPECTED_GAPS, "gap count"
    )
    if tape.duplicated(["symbol_norm", "timestamp"]).any():
        raise AssertionError("duplicate symbol timestamp after concatenation")
    if return_diagnostics:
        return tape, diagnostics
    return tape


def _segment_rolling(
    frame: pd.DataFrame,
    column: str,
    window: int,
    operation: str,
) -> pd.Series:
    keys = ["symbol_norm", "session_date", "segment_index"]
    grouped = frame.groupby(keys, sort=False)[column]
    rolling = grouped.rolling(window, min_periods=window)
    if operation == "mean":
        values = rolling.mean()
    elif operation == "std_population":
        values = rolling.std(ddof=0)
    elif operation == "max":
        values = rolling.max()
    elif operation == "min":
        values = rolling.min()
    else:
        raise AssertionError(f"unknown rolling operation {operation}")
    return values.reset_index(level=keys, drop=True).sort_index()


def build_feature_surface(tape: pd.DataFrame) -> pd.DataFrame:
    """Engineer the exact ordered causal 40-feature surface."""

    frame = tape.copy()
    q = frame["bar_ordinal"].to_numpy(float) * 5.0 / 385.0
    frame["clock_fraction"] = q
    frame["clock_fraction_squared"] = q * q
    frame["clock_sin_1"] = np.sin(2.0 * math.pi * q)
    frame["clock_cos_1"] = np.cos(2.0 * math.pi * q)
    frame["clock_sin_2"] = np.sin(4.0 * math.pi * q)
    frame["clock_cos_2"] = np.cos(4.0 * math.pi * q)

    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    bar_range = highs - lows
    nonzero_range = bar_range > EPSILON
    frame["log_close_open"] = np.log(closes / opens)
    frame["log_high_low"] = np.log(highs / lows)
    signed_body = np.zeros(len(frame), dtype=float)
    absolute_body = np.zeros(len(frame), dtype=float)
    upper_wick = np.zeros(len(frame), dtype=float)
    lower_wick = np.zeros(len(frame), dtype=float)
    close_location = np.full(len(frame), 0.5, dtype=float)
    signed_body[nonzero_range] = (
        closes[nonzero_range] - opens[nonzero_range]
    ) / bar_range[nonzero_range]
    absolute_body[nonzero_range] = (
        np.abs(closes[nonzero_range] - opens[nonzero_range]) / bar_range[nonzero_range]
    )
    upper_wick[nonzero_range] = (
        highs[nonzero_range] - np.maximum(opens[nonzero_range], closes[nonzero_range])
    ) / bar_range[nonzero_range]
    lower_wick[nonzero_range] = (
        np.minimum(opens[nonzero_range], closes[nonzero_range]) - lows[nonzero_range]
    ) / bar_range[nonzero_range]
    close_location[nonzero_range] = (
        closes[nonzero_range] - lows[nonzero_range]
    ) / bar_range[nonzero_range]
    frame["signed_body_fraction"] = signed_body
    frame["absolute_body_fraction"] = absolute_body
    frame["upper_wick_fraction"] = upper_wick
    frame["lower_wick_fraction"] = lower_wick
    frame["close_location"] = close_location

    segment_keys = ["symbol_norm", "session_date", "segment_index"]
    log_close = np.log(frame["close"])
    frame["_log_close"] = log_close
    for window in (1, 3, 6, 12):
        frame[f"close_return_{window}"] = log_close - frame.groupby(
            segment_keys, sort=False
        )["_log_close"].shift(window)
    frame["_abs_close_return_1"] = frame["close_return_1"].abs()
    for window in (3, 6, 12):
        frame[f"mean_abs_close_return_{window}"] = _segment_rolling(
            frame, "_abs_close_return_1", window, "mean"
        )
        frame[f"std_close_return_{window}"] = _segment_rolling(
            frame, "close_return_1", window, "std_population"
        )
        frame[f"mean_log_range_{window}"] = _segment_rolling(
            frame, "log_high_low", window, "mean"
        )
    for window in (6, 12):
        frame[f"log_range_ratio_{window}"] = (
            frame["log_high_low"] - frame[f"mean_log_range_{window}"]
        )

    session_keys = ["symbol_norm", "session_date"]
    session_group = frame.groupby(session_keys, sort=False)
    first_open = session_group["open"].transform("first")
    running_high = session_group["high"].cummax()
    running_low = session_group["low"].cummin()
    frame["session_log_return"] = np.log(frame["close"] / first_open)
    frame["distance_to_session_high"] = np.log(frame["close"] / running_high)
    frame["distance_from_session_low"] = np.log(frame["close"] / running_low)
    running_range = running_high.to_numpy(float) - running_low.to_numpy(float)
    running_nonzero = running_range > EPSILON
    session_location = np.full(len(frame), 0.5, dtype=float)
    session_location[running_nonzero] = (
        closes[running_nonzero] - running_low.to_numpy(float)[running_nonzero]
    ) / running_range[running_nonzero]
    frame["session_range_location"] = session_location
    frame["running_log_range"] = np.log(running_high / running_low)

    for window in (6, 12):
        rolling_high = _segment_rolling(frame, "high", window, "max")
        rolling_low = _segment_rolling(frame, "low", window, "min")
        frame[f"distance_to_rolling_high_{window}"] = np.log(
            frame["close"] / rolling_high
        )
        frame[f"distance_from_rolling_low_{window}"] = np.log(
            frame["close"] / rolling_low
        )
    bars_available = frame["segment_position"].to_numpy(float) + 1.0
    for window in (3, 6, 12):
        frame[f"availability_{window}"] = np.minimum(bars_available / window, 1.0)

    observed = tuple(name for name in FULL_FEATURES if name in frame.columns)
    _require_equal(observed, FULL_FEATURES, "engineered feature order")
    values = frame.loc[:, FULL_FEATURES].to_numpy(float)
    if np.isinf(values).any():
        raise AssertionError("infinite engineered feature")
    forbidden_tokens = ("regime", "state", "loop", "cycle", "b0", "template", "volume")
    if any(
        any(token in name.lower() for token in forbidden_tokens)
        for name in FULL_FEATURES
    ):
        raise AssertionError("forbidden feature entered clean-slate whitelist")
    return frame.drop(columns=["_log_close", "_abs_close_return_1"])


def build_horizon_surface(feature_surface: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build a target-blind exact-support cohort for one frozen horizon."""

    if horizon not in HORIZONS:
        raise AssertionError(f"unfrozen horizon {horizon}")
    supported = feature_surface["segment_position"].to_numpy(
        int
    ) + horizon < feature_surface["segment_size"].to_numpy(int)
    columns = [
        "source_position",
        "symbol_norm",
        "session_date",
        "month_key",
        "timestamp",
        "bar_ordinal",
        "segment_index",
        "segment_position",
        *FULL_FEATURES,
    ]
    frame = feature_surface.loc[supported, columns].copy()
    frame = frame.rename(columns={"timestamp": "decision_timestamp"})
    frame["horizon"] = np.int16(horizon)
    frame["entry_timestamp"] = frame["decision_timestamp"] + pd.Timedelta(minutes=5)
    frame["exit_timestamp"] = frame["decision_timestamp"] + pd.Timedelta(
        minutes=5 * horizon
    )
    frame["anchor_id"] = (
        frame["symbol_norm"].astype(str)
        + "|"
        + frame["decision_timestamp"].astype(str)
        + f"|h{horizon}"
    )
    frame = frame.sort_values(
        ["symbol_norm", "session_date", "decision_timestamp"], kind="stable"
    ).reset_index(drop=True)
    if len(frame) != EXPECTED_ROWS[horizon]:
        raise AssertionError(
            f"exact h{horizon} cohort drift: {len(frame)} != {EXPECTED_ROWS[horizon]}"
        )
    validation = frame["month_key"].isin(VALIDATION_MONTHS)
    if int(validation.sum()) != EXPECTED_VALIDATION_ROWS[horizon]:
        raise AssertionError(
            f"validation h{horizon} cohort drift: {int(validation.sum())} != "
            f"{EXPECTED_VALIDATION_ROWS[horizon]}"
        )
    if frame["anchor_id"].duplicated().any():
        raise AssertionError(f"duplicate h{horizon} anchor id")
    return frame


def attach_outcomes(
    tape: pd.DataFrame,
    anchors: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Attach next-open and fixed-close outcomes to an already chosen cohort."""

    positions = anchors["source_position"].to_numpy(np.int64)
    entry_positions = positions + 1
    exit_positions = positions + horizon
    timestamps = tape["timestamp"].to_numpy()
    if not (
        timestamps[entry_positions] - anchors["decision_timestamp"].to_numpy()
        == np.timedelta64(5, "m")
    ).all():
        raise AssertionError(f"inexact h{horizon} entry support")
    if not (
        timestamps[exit_positions] - anchors["decision_timestamp"].to_numpy()
        == np.timedelta64(5 * horizon, "m")
    ).all():
        raise AssertionError(f"inexact h{horizon} exit support")
    symbols = tape["symbol_norm"].to_numpy(str)
    sessions = tape["session_date"].to_numpy(str)
    expected_symbols = anchors["symbol_norm"].to_numpy(str)
    expected_sessions = anchors["session_date"].to_numpy(str)
    if not (
        (symbols[entry_positions] == expected_symbols)
        & (symbols[exit_positions] == expected_symbols)
        & (sessions[entry_positions] == expected_sessions)
        & (sessions[exit_positions] == expected_sessions)
    ).all():
        raise AssertionError(f"cross-symbol/session outcome at h{horizon}")
    entry_open = tape["open"].to_numpy(float)[entry_positions]
    exit_close = tape["close"].to_numpy(float)[exit_positions]
    output = anchors[
        [
            "anchor_id",
            "symbol_norm",
            "session_date",
            "month_key",
            "decision_timestamp",
            "entry_timestamp",
            "exit_timestamp",
            "bar_ordinal",
            "horizon",
        ]
    ].copy()
    output["next_bar_open"] = entry_open
    output["exit_close"] = exit_close
    output["target_bps"] = 10000.0 * (exit_close / entry_open - 1.0)
    if not np.isfinite(
        output[["next_bar_open", "exit_close", "target_bps"]].to_numpy(float)
    ).all():
        raise AssertionError(f"non-finite outcome at h{horizon}")
    return output


def fold_masks(surface: pd.DataFrame, month: str) -> tuple[np.ndarray, np.ndarray]:
    if month not in VALIDATION_MONTHS:
        raise AssertionError(f"unfrozen validation month {month}")
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    train = surface["exit_timestamp"].lt(month_start).to_numpy(bool)
    score = surface["month_key"].eq(month).to_numpy(bool)
    if not train.any() or not score.any() or np.logical_and(train, score).any():
        raise AssertionError(f"invalid expanding fold masks for {month}")
    return train, score


def equal_symbol_weights(symbols: Sequence[str] | pd.Series | np.ndarray) -> np.ndarray:
    values = pd.Series(np.asarray(symbols, dtype=str), copy=False)
    counts = values.map(values.value_counts(sort=False)).to_numpy(float)
    represented = int(values.nunique())
    weights = len(values) / (represented * counts)
    if not math.isclose(float(weights.mean()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("equal-symbol weights do not have mean one")
    totals = (
        pd.Series(weights).groupby(values.to_numpy(), sort=False).sum().to_numpy(float)
    )
    if not np.allclose(totals, totals[0], rtol=0.0, atol=1e-9):
        raise AssertionError("symbols do not have equal total training weight")
    return weights


def training_medians(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        medians = np.nanmedian(values, axis=0)
    if np.isnan(medians).any():
        missing = np.flatnonzero(np.isnan(medians)).tolist()
        raise AssertionError(f"all-missing training feature columns {missing}")
    return medians


def apply_medians(values: np.ndarray, medians: np.ndarray) -> np.ndarray:
    output = np.asarray(values, float).copy()
    rows, columns = np.where(np.isnan(output))
    output[rows, columns] = medians[columns]
    if not np.isfinite(output).all():
        raise AssertionError("non-finite value after training-only imputation")
    return output


def action_from_prediction(
    prediction_bps: Sequence[float] | np.ndarray, threshold_bps: float
) -> np.ndarray:
    prediction = np.asarray(prediction_bps, float)
    if threshold_bps not in ACTION_THRESHOLDS:
        raise AssertionError(f"unfrozen action threshold {threshold_bps}")
    if not np.isfinite(prediction).all():
        raise AssertionError("non-finite prediction before action mapping")
    return np.where(
        prediction >= threshold_bps,
        1,
        np.where(prediction <= -threshold_bps, -1, 0),
    ).astype(np.int8)


def greedy_nonoverlap(
    frame: pd.DataFrame, actions: Sequence[int] | np.ndarray, horizon: int
) -> np.ndarray:
    """Accept earliest non-abstaining actions; a prior exit-time action is eligible."""

    action_values = np.asarray(actions, dtype=np.int8)
    if len(action_values) != len(frame):
        raise AssertionError("action length mismatch")
    accepted = np.zeros(len(frame), dtype=np.int8)
    ordered = frame.assign(
        _action=action_values, _position=np.arange(len(frame))
    ).sort_values(["symbol_norm", "session_date", "decision_timestamp"], kind="stable")
    for _, group in ordered.groupby(["symbol_norm", "session_date"], sort=False):
        blocked_until: pd.Timestamp | None = None
        for timestamp_value, action_value, position_value in zip(
            group["decision_timestamp"],
            group["_action"],
            group["_position"],
            strict=True,
        ):
            action = int(action_value)
            if action == 0:
                continue
            timestamp = pd.Timestamp(timestamp_value)
            if blocked_until is None or timestamp >= blocked_until:
                accepted[int(position_value)] = action
                blocked_until = timestamp + pd.Timedelta(minutes=5 * horizon)
    return accepted


def _threshold_label(threshold: float) -> str:
    if not float(threshold).is_integer():
        raise AssertionError("artifact columns require integer frozen thresholds")
    return f"{int(threshold)}bps"


def fit_monthly_oof_predictions(
    tape: pd.DataFrame,
    feature_surface: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, int]]:
    prediction_parts: list[pd.DataFrame] = []
    preprocessing_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    validation_rows: dict[int, int] = {}
    for horizon in HORIZONS:
        surface = build_horizon_surface(feature_surface, horizon)
        validation_rows[horizon] = int(
            surface["month_key"].isin(VALIDATION_MONTHS).sum()
        )
        for month in VALIDATION_MONTHS:
            train_mask, score_mask = fold_masks(surface, month)
            train = surface.loc[train_mask].copy()
            score = surface.loc[score_mask].copy()
            # Only eligible pre-fold targets are attached.  Validation targets are
            # not constructed by this function.
            training_outcomes = attach_outcomes(tape, train, horizon)
            target = training_outcomes["target_bps"].to_numpy(float)
            weights = equal_symbol_weights(train["symbol_norm"])
            for algorithm in ALGORITHMS:
                features = ALGORITHM_FEATURES[algorithm]
                train_raw = train.loc[:, features].to_numpy(float)
                score_raw = score.loc[:, features].to_numpy(float)
                medians = training_medians(train_raw)
                train_imputed = apply_medians(train_raw, medians)
                score_imputed = apply_medians(score_raw, medians)
                scale_mean = np.full(len(features), np.nan)
                scale_scale = np.full(len(features), np.nan)
                if algorithm.endswith("ridge"):
                    scaler = StandardScaler()
                    scaler.fit(train_imputed, sample_weight=weights)
                    train_values = scaler.transform(train_imputed)
                    score_values = scaler.transform(score_imputed)
                    scale_mean = scaler.mean_.astype(float)
                    scale_scale = scaler.scale_.astype(float)
                    estimator: Ridge | HistGradientBoostingRegressor = Ridge(
                        **RIDGE_PARAMETERS
                    )
                else:
                    train_values = train_imputed
                    score_values = score_imputed
                    estimator = HistGradientBoostingRegressor(**HGB_PARAMETERS)
                estimator.fit(train_values, target, sample_weight=weights)
                prediction = estimator.predict(score_values).astype(float)
                if not np.isfinite(prediction).all():
                    raise AssertionError(
                        f"non-finite {algorithm} prediction h{horizon} {month}"
                    )
                output = score[
                    [
                        "anchor_id",
                        "symbol_norm",
                        "session_date",
                        "decision_timestamp",
                        "bar_ordinal",
                    ]
                ].copy()
                output["fold_month"] = month
                output["algorithm"] = algorithm
                output["horizon"] = np.int16(horizon)
                output["prediction_bps"] = prediction
                prediction_parts.append(output)
                for index, feature in enumerate(features):
                    preprocessing_rows.append(
                        {
                            "fold_month": month,
                            "algorithm": algorithm,
                            "horizon": horizon,
                            "feature_order": index,
                            "feature": feature,
                            "training_median": float(medians[index]),
                            "scaler_mean": float(scale_mean[index]),
                            "scaler_scale": float(scale_scale[index]),
                        }
                    )
                if isinstance(estimator, Ridge):
                    coefficient_rows.append(
                        {
                            "fold_month": month,
                            "algorithm": algorithm,
                            "horizon": horizon,
                            "feature_order": -1,
                            "feature": "__intercept__",
                            "coefficient": float(estimator.intercept_),
                        }
                    )
                    for index, (feature, coefficient) in enumerate(
                        zip(features, estimator.coef_, strict=True)
                    ):
                        coefficient_rows.append(
                            {
                                "fold_month": month,
                                "algorithm": algorithm,
                                "horizon": horizon,
                                "feature_order": index,
                                "feature": feature,
                                "coefficient": float(coefficient),
                            }
                        )
                fold_rows.append(
                    {
                        "fold_month": month,
                        "algorithm": algorithm,
                        "horizon": horizon,
                        "train_rows": len(train),
                        "score_rows": len(score),
                        "train_symbols": train["symbol_norm"].nunique(),
                        "score_symbols": score["symbol_norm"].nunique(),
                        "maximum_training_exit_timestamp": training_outcomes[
                            "exit_timestamp"
                        ].max(),
                        "minimum_scoring_timestamp": score["decision_timestamp"].min(),
                        "target_mean_bps": float(target.mean()),
                        "target_std_bps": float(target.std(ddof=0)),
                        "weight_min": float(weights.min()),
                        "weight_max": float(weights.max()),
                        "fitted_iterations": int(estimator.n_iter_)
                        if isinstance(estimator, HistGradientBoostingRegressor)
                        else None,
                    }
                )
        del surface
        gc.collect()
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(
            [
                "algorithm",
                "horizon",
                "symbol_norm",
                "session_date",
                "decision_timestamp",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    for threshold in ACTION_THRESHOLDS:
        label = _threshold_label(threshold)
        action = action_from_prediction(predictions["prediction_bps"], threshold)
        predictions[f"action_{label}"] = action
        accepted = np.zeros(len(predictions), dtype=np.int8)
        for (algorithm, horizon), group in predictions.groupby(
            ["algorithm", "horizon"], sort=False
        ):
            positions = group.index.to_numpy(int)
            accepted[positions] = greedy_nonoverlap(
                group, action[positions], int(horizon)
            )
        predictions[f"accepted_action_{label}"] = accepted
    expected_total = sum(EXPECTED_VALIDATION_ROWS.values()) * len(ALGORITHMS)
    if len(predictions) != expected_total:
        raise AssertionError(
            f"prediction ledger row drift: {len(predictions)} != {expected_total}"
        )
    key = ["anchor_id", "algorithm"]
    if predictions.duplicated(key).any():
        raise AssertionError("duplicate prediction ledger key")
    return (
        predictions,
        pd.DataFrame(preprocessing_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(fold_rows),
        validation_rows,
    )


def validate_prediction_ledger(predictions: pd.DataFrame) -> None:
    allowed = {
        "anchor_id",
        "fold_month",
        "symbol_norm",
        "session_date",
        "decision_timestamp",
        "bar_ordinal",
        "algorithm",
        "horizon",
        "prediction_bps",
        *(f"action_{_threshold_label(threshold)}" for threshold in ACTION_THRESHOLDS),
        *(
            f"accepted_action_{_threshold_label(threshold)}"
            for threshold in ACTION_THRESHOLDS
        ),
    }
    _require_equal(set(predictions.columns), allowed, "pre-outcome prediction columns")
    forbidden = ("target", "outcome", "price", "open", "high", "low", "close", "return")
    bad = [
        column
        for column in predictions.columns
        if any(token in column.lower() for token in forbidden)
    ]
    if bad:
        raise AssertionError(
            f"validation prices/outcomes leaked into pre-score ledger: {bad}"
        )
    for threshold in ACTION_THRESHOLDS:
        label = _threshold_label(threshold)
        expected = action_from_prediction(predictions["prediction_bps"], threshold)
        if not np.array_equal(
            predictions[f"action_{label}"].to_numpy(np.int8), expected
        ):
            raise AssertionError(f"action mapping drift at {threshold} bps")
        accepted = predictions[f"accepted_action_{label}"].to_numpy(np.int8)
        if (~np.isin(accepted, [-1, 0, 1])).any():
            raise AssertionError(f"invalid accepted action at {threshold} bps")
        if (accepted != 0).sum() > (expected != 0).sum():
            raise AssertionError("non-overlap created actions")


def write_prediction_freeze(
    predictions: pd.DataFrame,
    preprocessing: pd.DataFrame,
    coefficients: pd.DataFrame,
    folds: pd.DataFrame,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    validate_prediction_ledger(predictions)
    prediction_path = OUT / "prediction_actions_pre_outcome.parquet"
    preprocessing_path = OUT / "fold_preprocessing.csv"
    coefficient_path = OUT / "ridge_coefficients.csv"
    fold_path = OUT / "fold_metadata.csv"
    predictions.to_parquet(prediction_path, index=False)
    preprocessing.to_csv(preprocessing_path, index=False)
    coefficients.to_csv(coefficient_path, index=False)
    folds.to_csv(fold_path, index=False)
    files = (prediction_path, preprocessing_path, coefficient_path, fold_path)
    freeze = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "stage": "predictions_and_fixed_actions_written_before_validation_outcome_join",
        "validation_prices_or_outcomes_present": False,
        "prediction_rows": len(predictions),
        "sha256": {path.name: sha256(path) for path in files},
        "frozen_source_manifest_sha256": sha256(PRE_SCORE_PATH),
        "frozen_source_sha256": source_manifest["sha256"],
    }
    freeze_path = OUT / "prediction_action_freeze.json"
    write_json(freeze_path, freeze)
    freeze["freeze_manifest_sha256"] = sha256(freeze_path)
    return freeze


def verify_prediction_freeze(freeze: dict[str, Any]) -> None:
    freeze_path = OUT / "prediction_action_freeze.json"
    if sha256(freeze_path) != freeze["freeze_manifest_sha256"]:
        raise AssertionError("prediction freeze manifest changed before outcome join")
    actual = {name: sha256(OUT / name) for name in freeze["sha256"]}
    if actual != freeze["sha256"]:
        raise AssertionError(
            "pre-outcome prediction artifact changed before outcome join"
        )
    if freeze["validation_prices_or_outcomes_present"] is not False:
        raise AssertionError("prediction freeze incorrectly contains outcomes")


def build_validation_outcomes(
    tape: pd.DataFrame, feature_surface: pd.DataFrame
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        surface = build_horizon_surface(feature_surface, horizon)
        validation = surface.loc[surface["month_key"].isin(VALIDATION_MONTHS)].copy()
        parts.append(attach_outcomes(tape, validation, horizon))
        del surface, validation
        gc.collect()
    outcomes = (
        pd.concat(parts, ignore_index=True)
        .sort_values(
            ["horizon", "symbol_norm", "session_date", "decision_timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if outcomes["anchor_id"].duplicated().any():
        raise AssertionError("duplicate validation outcome key")
    return outcomes


def prediction_statistics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    actual_values = np.asarray(actual, float)
    predicted_values = np.asarray(predicted, float)
    error = predicted_values - actual_values
    mse = float(np.mean(error * error))
    mae = float(np.mean(np.abs(error)))
    actual_std = float(actual_values.std(ddof=0))
    predicted_std = float(predicted_values.std(ddof=0))
    pearson = (
        float(np.corrcoef(actual_values, predicted_values)[0, 1])
        if actual_std > 0.0 and predicted_std > 0.0
        else math.nan
    )
    spearman = (
        float(
            pd.Series(actual_values).corr(
                pd.Series(predicted_values), method="spearman"
            )
        )
        if actual_std > 0.0 and predicted_std > 0.0
        else math.nan
    )
    variance = float(np.mean((predicted_values - predicted_values.mean()) ** 2))
    slope = (
        float(
            np.mean(
                (predicted_values - predicted_values.mean())
                * (actual_values - actual_values.mean())
            )
            / variance
        )
        if variance > 0.0
        else math.nan
    )
    intercept = (
        float(actual_values.mean() - slope * predicted_values.mean())
        if math.isfinite(slope)
        else float(actual_values.mean())
    )
    return {
        "rows": len(actual_values),
        "mse_bps2": mse,
        "mae_bps": mae,
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "calibration_intercept_bps": intercept,
        "calibration_slope": slope,
        "target_mean_bps": float(actual_values.mean()),
        "prediction_mean_bps": float(predicted_values.mean()),
        "target_std_bps": actual_std,
        "prediction_std_bps": predicted_std,
    }


def evaluate_predictions(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    all_session_dates = sorted(scored["session_date"].unique())
    for (algorithm, horizon), group in scored.groupby(
        ["algorithm", "horizon"], sort=False
    ):
        stats = prediction_statistics(group["target_bps"], group["prediction_bps"])
        row: dict[str, Any] = {"algorithm": algorithm, "horizon": int(horizon), **stats}
        for threshold in ACTION_THRESHOLDS:
            action = group[f"action_{_threshold_label(threshold)}"].to_numpy(int)
            label = int(threshold)
            row[f"long_rate_{label}bps"] = float((action == 1).mean())
            row[f"short_rate_{label}bps"] = float((action == -1).mean())
            row[f"abstain_rate_{label}bps"] = float((action == 0).mean())
        pooled_rows.append(row)
        order = np.lexsort(
            (
                group["anchor_id"].astype(str).to_numpy(),
                group["prediction_bps"].to_numpy(float),
            )
        )
        decile = np.empty(len(group), dtype=np.int8)
        decile[order] = np.minimum(np.arange(len(group)) * 10 // len(group), 9)
        decile_frame = group.assign(prediction_decile=decile)
        for bucket, selected in decile_frame.groupby("prediction_decile", sort=True):
            decile_rows.append(
                {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "prediction_decile": int(bucket),
                    "rows": len(selected),
                    "prediction_mean_bps": float(selected["prediction_bps"].mean()),
                    "target_mean_bps": float(selected["target_bps"].mean()),
                    "positive_target_rate": float(
                        selected["target_bps"].gt(0.0).mean()
                    ),
                }
            )
        for month, selected in group.groupby("fold_month", sort=True):
            monthly_rows.append(
                {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "month": month,
                    **prediction_statistics(
                        selected["target_bps"], selected["prediction_bps"]
                    ),
                }
            )
        errors = group["prediction_bps"].to_numpy(float) - group["target_bps"].to_numpy(
            float
        )
        loss_frame = pd.DataFrame(
            {
                "session_date": group["session_date"].to_numpy(str),
                "symbol_norm": group["symbol_norm"].to_numpy(str),
                "squared_error_bps2": errors * errors,
                "absolute_error_bps": np.abs(errors),
            }
        )
        symbol_daily = loss_frame.groupby(["session_date", "symbol_norm"], sort=False)[
            ["squared_error_bps2", "absolute_error_bps"]
        ].mean()
        fixed_sleeve_daily = (
            symbol_daily.groupby("session_date", sort=True).sum() / len(SYMBOLS)
        ).reindex(all_session_dates, fill_value=0.0)
        for session_date, selected in fixed_sleeve_daily.iterrows():
            daily_rows.append(
                {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "session_date": session_date,
                    "mse_bps2": float(selected["squared_error_bps2"]),
                    "mae_bps": float(selected["absolute_error_bps"]),
                }
            )
    pooled = pd.DataFrame(pooled_rows)
    monthly = pd.DataFrame(monthly_rows)
    for frame in (pooled, monthly):
        clock = frame.loc[
            frame["algorithm"].eq("clock_ridge"),
            [
                *(["month"] if "month" in frame.columns else []),
                "horizon",
                "mse_bps2",
                "mae_bps",
            ],
        ].rename(columns={"mse_bps2": "clock_mse_bps2", "mae_bps": "clock_mae_bps"})
        join_keys = ["horizon"] + (["month"] if "month" in frame.columns else [])
        merged = frame.merge(clock, on=join_keys, how="left", validate="many_to_one")
        merged["relative_mse_improvement_vs_clock"] = (
            1.0 - merged["mse_bps2"] / merged["clock_mse_bps2"]
        )
        merged["relative_mae_improvement_vs_clock"] = (
            1.0 - merged["mae_bps"] / merged["clock_mae_bps"]
        )
        frame.drop(frame.index, inplace=True)
        for column in merged.columns:
            frame[column] = merged[column]
    return pooled, monthly, pd.DataFrame(decile_rows), pd.DataFrame(daily_rows)


def evaluate_prediction_stock_deletions(scored: pd.DataFrame) -> pd.DataFrame:
    """Report unchanged-prediction leave-one-stock-out prediction slices."""

    rows: list[dict[str, Any]] = []
    for deleted_symbol in SYMBOLS:
        retained = scored.loc[scored["symbol_norm"].ne(deleted_symbol)]
        for (algorithm, horizon), selected in retained.groupby(
            ["algorithm", "horizon"], sort=False
        ):
            rows.append(
                {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "deleted_symbol": deleted_symbol,
                    **prediction_statistics(
                        selected["target_bps"], selected["prediction_bps"]
                    ),
                }
            )
    output = pd.DataFrame(rows)
    clock = output.loc[
        output["algorithm"].eq("clock_ridge"),
        ["horizon", "deleted_symbol", "mse_bps2", "mae_bps"],
    ].rename(
        columns={
            "mse_bps2": "clock_mse_bps2",
            "mae_bps": "clock_mae_bps",
        }
    )
    output = output.merge(
        clock,
        on=["horizon", "deleted_symbol"],
        how="left",
        validate="many_to_one",
    )
    output["relative_mse_improvement_vs_clock"] = (
        1.0 - output["mse_bps2"] / output["clock_mse_bps2"]
    )
    output["relative_mae_improvement_vs_clock"] = (
        1.0 - output["mae_bps"] / output["clock_mae_bps"]
    )
    if len(output) != len(SYMBOLS) * len(ALGORITHMS) * len(HORIZONS):
        raise AssertionError("prediction stock-deletion row drift")
    return output


def gross_returns(
    direction: Sequence[int] | np.ndarray,
    next_open: Sequence[float] | np.ndarray,
    exit_close: Sequence[float] | np.ndarray,
) -> np.ndarray:
    side = np.asarray(direction, float)
    entry = np.asarray(next_open, float)
    exit_value = np.asarray(exit_close, float)
    return side * (exit_value / entry - 1.0)


def build_accepted_entries(scored: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    base_columns = [
        "anchor_id",
        "fold_month",
        "algorithm",
        "horizon",
        "symbol_norm",
        "session_date",
        "decision_timestamp",
        "entry_timestamp",
        "exit_timestamp",
        "bar_ordinal",
        "clock_quartile",
        "prediction_bps",
        "next_bar_open",
        "exit_close",
        "target_bps",
    ]
    for threshold in ACTION_THRESHOLDS:
        label = _threshold_label(threshold)
        accepted = scored[f"accepted_action_{label}"].ne(0)
        selected = scored.loc[accepted, base_columns].copy()
        selected["threshold_bps"] = threshold
        selected["direction"] = scored.loc[
            accepted, f"accepted_action_{label}"
        ].to_numpy(np.int8)
        selected["gross_return"] = gross_returns(
            selected["direction"], selected["next_bar_open"], selected["exit_close"]
        )
        selected["gross_return_bps"] = 10000.0 * selected["gross_return"]
        selected["holding_bars"] = selected["horizon"].astype(np.int16)
        parts.append(selected)
    entries = (
        pd.concat(parts, ignore_index=True)
        .sort_values(
            [
                "algorithm",
                "horizon",
                "threshold_bps",
                "symbol_norm",
                "session_date",
                "decision_timestamp",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if entries.duplicated(["anchor_id", "algorithm", "threshold_bps"]).any():
        raise AssertionError("duplicate accepted entry")
    return entries


def portfolio_stats(daily: Sequence[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(daily, float)
    if len(values) == 0 or not np.isfinite(values).all() or (values <= -1.0).any():
        raise AssertionError("invalid daily return path")
    equity = np.cumprod(1.0 + values)
    cumulative = float(equity[-1] - 1.0)
    annualized = float((1.0 + cumulative) ** (252.0 / len(values)) - 1.0)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    volatility = std * math.sqrt(252.0)
    sharpe = float(values.mean() / std * math.sqrt(252.0)) if std > 0.0 else math.nan
    path = np.r_[1.0, equity]
    drawdown = path / np.maximum.accumulate(path) - 1.0
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "descriptive_sharpe_zero_rate": sharpe,
        "maximum_drawdown": float(drawdown.min(initial=0.0)),
        "mean_daily_return": float(values.mean()),
    }


def daily_for_entries(
    entries: pd.DataFrame,
    session_dates: Sequence[str],
    cost_bps_per_side: int,
    deleted_symbol: str | None = None,
) -> tuple[pd.Series, np.ndarray]:
    selected = entries.copy()
    divisor = len(SYMBOLS)
    if deleted_symbol is not None:
        selected = selected.loc[selected["symbol_norm"].ne(deleted_symbol)].copy()
        divisor -= 1
    net = selected["gross_return"].to_numpy(float) - 2.0 * cost_bps_per_side / 10000.0
    if selected.empty:
        return pd.Series(0.0, index=pd.Index(session_dates, name="session_date")), net
    if (net <= -1.0).any():
        raise AssertionError("entry return exceeded unlevered sleeve collateral")
    selected["log_growth"] = np.log1p(net)
    sleeve = np.expm1(
        selected.groupby(["session_date", "symbol_norm"], sort=False)[
            "log_growth"
        ].sum()
    )
    daily = (sleeve.groupby("session_date").sum() / divisor).reindex(
        session_dates, fill_value=0.0
    )
    return daily, 10000.0 * net


def _profit_factor(net_bps: np.ndarray) -> float:
    positive = net_bps[net_bps > 0.0]
    negative = net_bps[net_bps < 0.0]
    if len(negative):
        return float(positive.sum() / -negative.sum())
    if len(positive):
        return math.inf
    return math.nan


def evaluate_actions(
    entries: pd.DataFrame,
    session_dates: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    daily_parts: list[pd.DataFrame] = []
    month_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    validation_bar_denominator = SESSION_BARS * len(SYMBOLS) * len(session_dates)
    for algorithm in ALGORITHMS:
        for horizon in HORIZONS:
            for threshold in ACTION_THRESHOLDS:
                group = entries.loc[
                    entries["algorithm"].eq(algorithm)
                    & entries["horizon"].eq(horizon)
                    & entries["threshold_bps"].eq(threshold)
                ]
                for cost in COSTS:
                    daily, net_bps = daily_for_entries(group, session_dates, cost)
                    long_mask = group["direction"].eq(1).to_numpy(bool)
                    short_mask = group["direction"].eq(-1).to_numpy(bool)
                    metric_rows.append(
                        {
                            "algorithm": algorithm,
                            "horizon": int(horizon),
                            "threshold_bps": float(threshold),
                            "cost_bps_per_side": cost,
                            "accepted_trades": len(group),
                            "long_trades": int(long_mask.sum()),
                            "short_trades": int(short_mask.sum()),
                            "stocks_with_trade": int(group["symbol_norm"].nunique()),
                            "stocks_with_long_trade": int(
                                group.loc[long_mask, "symbol_norm"].nunique()
                            ),
                            "stocks_with_short_trade": int(
                                group.loc[short_mask, "symbol_norm"].nunique()
                            ),
                            "mean_net_trade_bps": float(net_bps.mean())
                            if len(net_bps)
                            else math.nan,
                            "median_net_trade_bps": float(np.median(net_bps))
                            if len(net_bps)
                            else math.nan,
                            "long_mean_net_bps": float(net_bps[long_mask].mean())
                            if long_mask.any()
                            else math.nan,
                            "short_mean_net_bps": float(net_bps[short_mask].mean())
                            if short_mask.any()
                            else math.nan,
                            "win_rate": float((net_bps > 0.0).mean())
                            if len(net_bps)
                            else math.nan,
                            "profit_factor": _profit_factor(net_bps),
                            "exposure_fraction": float(
                                group["holding_bars"].sum() / validation_bar_denominator
                            ),
                            **portfolio_stats(daily.to_numpy(float)),
                        }
                    )
                    daily_parts.append(
                        pd.DataFrame(
                            {
                                "algorithm": algorithm,
                                "horizon": int(horizon),
                                "threshold_bps": float(threshold),
                                "cost_bps_per_side": cost,
                                "session_date": list(session_dates),
                                "daily_return": daily.to_numpy(float),
                            }
                        )
                    )
                    for month in VALIDATION_MONTHS:
                        month_sessions = [
                            date for date in session_dates if date.startswith(month)
                        ]
                        month_daily = daily.reindex(month_sessions, fill_value=0.0)
                        selected = group.loc[
                            group["session_date"].str.startswith(month)
                        ]
                        selected_net = (
                            10000.0 * selected["gross_return"].to_numpy(float)
                            - 2.0 * cost
                        )
                        month_rows.append(
                            {
                                "algorithm": algorithm,
                                "horizon": int(horizon),
                                "threshold_bps": float(threshold),
                                "cost_bps_per_side": cost,
                                "month": month,
                                "session_dates": len(month_sessions),
                                "accepted_trades": len(selected),
                                "mean_net_trade_bps": float(selected_net.mean())
                                if len(selected_net)
                                else math.nan,
                                **portfolio_stats(month_daily.to_numpy(float)),
                            }
                        )
                    for side_value, side_name in ((1, "long"), (-1, "short")):
                        selected = group.loc[group["direction"].eq(side_value)]
                        selected_net = (
                            10000.0 * selected["gross_return"].to_numpy(float)
                            - 2.0 * cost
                        )
                        slice_rows.append(
                            {
                                "algorithm": algorithm,
                                "horizon": int(horizon),
                                "threshold_bps": float(threshold),
                                "cost_bps_per_side": cost,
                                "slice_type": "side",
                                "slice_value": side_name,
                                "accepted_trades": len(selected),
                                "stocks_with_trade": int(
                                    selected["symbol_norm"].nunique()
                                ),
                                "mean_net_trade_bps": float(selected_net.mean())
                                if len(selected_net)
                                else math.nan,
                                "win_rate": float((selected_net > 0.0).mean())
                                if len(selected_net)
                                else math.nan,
                            }
                        )
                    for quartile in range(4):
                        selected = group.loc[group["clock_quartile"].eq(quartile)]
                        selected_net = (
                            10000.0 * selected["gross_return"].to_numpy(float)
                            - 2.0 * cost
                        )
                        slice_rows.append(
                            {
                                "algorithm": algorithm,
                                "horizon": int(horizon),
                                "threshold_bps": float(threshold),
                                "cost_bps_per_side": cost,
                                "slice_type": "clock_quartile",
                                "slice_value": str(quartile),
                                "accepted_trades": len(selected),
                                "stocks_with_trade": int(
                                    selected["symbol_norm"].nunique()
                                ),
                                "mean_net_trade_bps": float(selected_net.mean())
                                if len(selected_net)
                                else math.nan,
                                "win_rate": float((selected_net > 0.0).mean())
                                if len(selected_net)
                                else math.nan,
                            }
                        )
                    for deleted_symbol in SYMBOLS:
                        deleted_daily, _ = daily_for_entries(
                            group, session_dates, cost, deleted_symbol
                        )
                        deletion_rows.append(
                            {
                                "algorithm": algorithm,
                                "horizon": int(horizon),
                                "threshold_bps": float(threshold),
                                "cost_bps_per_side": cost,
                                "deleted_symbol": deleted_symbol,
                                **portfolio_stats(deleted_daily.to_numpy(float)),
                            }
                        )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(daily_parts, ignore_index=True),
        pd.DataFrame(month_rows),
        pd.DataFrame(slice_rows),
        pd.DataFrame(deletion_rows),
    )


def moving_block_interval(
    values: Sequence[float] | np.ndarray,
    seed_offset: int = 0,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    block_size: int = BOOTSTRAP_BLOCK,
) -> tuple[float, float, float]:
    data = np.asarray(values, float)
    if len(data) < block_size or not np.isfinite(data).all():
        raise AssertionError("invalid moving-block bootstrap input")
    starts = np.arange(len(data) - block_size + 1)
    blocks_needed = math.ceil(len(data) / block_size)
    rng = np.random.default_rng(SEED + seed_offset)
    sampled_starts = rng.choice(starts, size=(draws, blocks_needed), replace=True)
    positions = (
        sampled_starts[:, :, None] + np.arange(block_size)[None, None, :]
    ).reshape(draws, -1)[:, : len(data)]
    sampled_mean = data[positions].mean(axis=1)
    lower, upper = np.quantile(sampled_mean, [0.025, 0.975], method="linear")
    return float(data.mean()), float(lower), float(upper)


def build_bootstraps(
    scored: pd.DataFrame,
    daily_prediction: pd.DataFrame,
    daily_action: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for horizon_index, horizon in enumerate(HORIZONS):
            prediction = daily_prediction.loc[daily_prediction["horizon"].eq(horizon)]
            clock = prediction.loc[
                prediction["algorithm"].eq("clock_ridge")
            ].sort_values("session_date")
            selected = prediction.loc[
                prediction["algorithm"].eq(candidate)
            ].sort_values("session_date")
            if (
                not selected["session_date"]
                .reset_index(drop=True)
                .equals(clock["session_date"].reset_index(drop=True))
            ):
                raise AssertionError("daily prediction bootstrap alignment failure")
            paired = scored.loc[
                scored["horizon"].eq(horizon)
                & scored["algorithm"].isin(["clock_ridge", candidate]),
                [
                    "anchor_id",
                    "symbol_norm",
                    "session_date",
                    "algorithm",
                    "target_bps",
                    "prediction_bps",
                ],
            ]
            clock_rows = (
                paired.loc[paired["algorithm"].eq("clock_ridge")]
                .drop(columns=["algorithm"])
                .rename(
                    columns={
                        "target_bps": "clock_target_bps",
                        "prediction_bps": "clock_prediction_bps",
                    }
                )
            )
            candidate_rows = (
                paired.loc[paired["algorithm"].eq(candidate)]
                .drop(columns=["algorithm"])
                .rename(
                    columns={
                        "target_bps": "candidate_target_bps",
                        "prediction_bps": "candidate_prediction_bps",
                    }
                )
            )
            paired_rows = candidate_rows.merge(
                clock_rows,
                on=["anchor_id", "symbol_norm", "session_date"],
                how="inner",
                validate="one_to_one",
            )
            if len(paired_rows) != len(candidate_rows) or not np.array_equal(
                paired_rows["candidate_target_bps"].to_numpy(float),
                paired_rows["clock_target_bps"].to_numpy(float),
            ):
                raise AssertionError("rowwise candidate-clock target alignment failure")
            target = paired_rows["candidate_target_bps"].to_numpy(float)
            paired_rows["mse_improvement_bps2"] = (
                paired_rows["clock_prediction_bps"].to_numpy(float) - target
            ) ** 2 - (
                paired_rows["candidate_prediction_bps"].to_numpy(float) - target
            ) ** 2
            symbol_daily = paired_rows.groupby(
                ["session_date", "symbol_norm"], sort=False
            )["mse_improvement_bps2"].mean()
            mse_improvement = (
                (symbol_daily.groupby("session_date", sort=True).sum() / len(SYMBOLS))
                .reindex(clock["session_date"].to_numpy(str), fill_value=0.0)
                .to_numpy(float)
            )
            level_difference = clock["mse_bps2"].to_numpy(float) - selected[
                "mse_bps2"
            ].to_numpy(float)
            if not np.allclose(
                mse_improvement, level_difference, rtol=1e-12, atol=1e-9
            ):
                raise AssertionError(
                    "rowwise and fixed-sleeve MSE improvements disagree"
                )
            observed, lower, upper = moving_block_interval(
                mse_improvement, horizon_index
            )
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": "clock_ridge",
                    "horizon": horizon,
                    "metric": "daily_mse_improvement_bps2",
                    "session_dates": len(selected),
                    "observed_mean": observed,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
            action = daily_action.loc[
                daily_action["horizon"].eq(horizon)
                & daily_action["threshold_bps"].eq(PRIMARY_THRESHOLD)
                & daily_action["cost_bps_per_side"].eq(PRIMARY_COST)
            ]
            clock_action = action.loc[
                action["algorithm"].eq("clock_ridge")
            ].sort_values("session_date")
            candidate_action = action.loc[
                action["algorithm"].eq(candidate)
            ].sort_values("session_date")
            if (
                not candidate_action["session_date"]
                .reset_index(drop=True)
                .equals(clock_action["session_date"].reset_index(drop=True))
            ):
                raise AssertionError("daily action bootstrap alignment failure")
            observed, lower, upper = moving_block_interval(
                candidate_action["daily_return"].to_numpy(float), 100 + horizon_index
            )
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": "zero_cash",
                    "horizon": horizon,
                    "metric": "absolute_daily_return",
                    "session_dates": len(candidate_action),
                    "observed_mean": observed,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
            difference = candidate_action["daily_return"].to_numpy(
                float
            ) - clock_action["daily_return"].to_numpy(float)
            observed, lower, upper = moving_block_interval(
                difference, 200 + horizon_index
            )
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": "clock_ridge",
                    "horizon": horizon,
                    "metric": "daily_return_advantage",
                    "session_dates": len(candidate_action),
                    "observed_mean": observed,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def final_decisions(
    prediction_metrics: pd.DataFrame,
    monthly_prediction_metrics: pd.DataFrame,
    action_metrics: pd.DataFrame,
    monthly_action_metrics: pd.DataFrame,
    stock_deletions: pd.DataFrame,
    bootstraps: pd.DataFrame,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for candidate in CANDIDATES:
        prediction = prediction_metrics.loc[
            prediction_metrics["algorithm"].eq(candidate)
        ]
        monthly_prediction = monthly_prediction_metrics.loc[
            monthly_prediction_metrics["algorithm"].eq(candidate)
        ]
        action = action_metrics.loc[
            action_metrics["algorithm"].eq(candidate)
            & action_metrics["threshold_bps"].eq(PRIMARY_THRESHOLD)
            & action_metrics["cost_bps_per_side"].eq(PRIMARY_COST)
        ]
        monthly_action = monthly_action_metrics.loc[
            monthly_action_metrics["algorithm"].eq(candidate)
            & monthly_action_metrics["threshold_bps"].eq(PRIMARY_THRESHOLD)
            & monthly_action_metrics["cost_bps_per_side"].eq(PRIMARY_COST)
        ]
        deletion = stock_deletions.loc[
            stock_deletions["algorithm"].eq(candidate)
            & stock_deletions["threshold_bps"].eq(PRIMARY_THRESHOLD)
            & stock_deletions["cost_bps_per_side"].eq(PRIMARY_COST)
        ]
        bootstrap = bootstraps.loc[bootstraps["candidate"].eq(candidate)]
        monthly_mse_wins = (
            monthly_prediction.assign(
                _win=monthly_prediction["relative_mse_improvement_vs_clock"].gt(0.0)
            )
            .groupby("horizon", sort=True)["_win"]
            .sum()
        )
        checks = {
            "minimum_nonoverlapping_trades_each_horizon": bool(
                len(action) == 3 and action["accepted_trades"].ge(500).all()
            ),
            "minimum_trades_each_validation_month_and_horizon": bool(
                len(monthly_action) == 18
                and monthly_action["accepted_trades"].ge(50).all()
            ),
            "minimum_stocks_each_horizon": bool(
                len(action) == 3 and action["stocks_with_trade"].ge(15).all()
            ),
            "minimum_long_trades_each_horizon": bool(
                len(action) == 3 and action["long_trades"].ge(100).all()
            ),
            "minimum_short_trades_each_horizon": bool(
                len(action) == 3 and action["short_trades"].ge(100).all()
            ),
            "minimum_stocks_each_side_and_horizon": bool(
                len(action) == 3
                and action["stocks_with_long_trade"].ge(10).all()
                and action["stocks_with_short_trade"].ge(10).all()
            ),
            "minimum_relative_pooled_mse_improvement_vs_clock_each_horizon": bool(
                len(prediction) == 3
                and prediction["relative_mse_improvement_vs_clock"].ge(0.0025).all()
            ),
            "mae_not_worse_than_clock_each_horizon": bool(
                len(prediction) == 3
                and prediction["mae_bps"].le(prediction["clock_mae_bps"]).all()
            ),
            "paired_daily_mse_improvement_bootstrap_lower_above_zero_each_horizon": bool(
                len(bootstrap.loc[bootstrap["metric"].eq("daily_mse_improvement_bps2")])
                == 3
                and bootstrap.loc[
                    bootstrap["metric"].eq("daily_mse_improvement_bps2"), "ci_lower"
                ]
                .gt(0.0)
                .all()
            ),
            "positive_pearson_correlation_each_horizon": bool(
                len(prediction) == 3 and prediction["pearson_correlation"].gt(0.0).all()
            ),
            "minimum_mse_improvement_months_each_horizon": bool(
                len(monthly_mse_wins) == 3 and monthly_mse_wins.ge(4).all()
            ),
            "positive_mean_net_trade_bps_each_horizon": bool(
                len(action) == 3 and action["mean_net_trade_bps"].gt(0.0).all()
            ),
            "positive_cumulative_return_each_horizon": bool(
                len(action) == 3 and action["cumulative_return"].gt(0.0).all()
            ),
            "absolute_daily_return_bootstrap_lower_above_zero_each_horizon": bool(
                len(bootstrap.loc[bootstrap["metric"].eq("absolute_daily_return")]) == 3
                and bootstrap.loc[
                    bootstrap["metric"].eq("absolute_daily_return"), "ci_lower"
                ]
                .gt(0.0)
                .all()
            ),
            "paired_daily_return_advantage_vs_clock_lower_above_zero_each_horizon": bool(
                len(bootstrap.loc[bootstrap["metric"].eq("daily_return_advantage")])
                == 3
                and bootstrap.loc[
                    bootstrap["metric"].eq("daily_return_advantage"), "ci_lower"
                ]
                .gt(0.0)
                .all()
            ),
            "positive_cumulative_return_each_month_and_horizon": bool(
                len(monthly_action) == 18
                and monthly_action["cumulative_return"].gt(0.0).all()
            ),
            "positive_cumulative_return_every_stock_deletion_and_horizon": bool(
                len(deletion) == len(SYMBOLS) * len(HORIZONS)
                and deletion["cumulative_return"].gt(0.0).all()
            ),
            "positive_long_mean_net_bps_each_horizon": bool(
                len(action) == 3 and action["long_mean_net_bps"].gt(0.0).all()
            ),
            "positive_short_mean_net_bps_each_horizon": bool(
                len(action) == 3 and action["short_mean_net_bps"].gt(0.0).all()
            ),
        }
        checks["retained"] = bool(all(checks.values()))
        candidates[candidate] = {
            "checks": checks,
            "decision": (
                "retain_2024_internal_entry_hypothesis_for_new_shadow_sessions"
                if checks["retained"]
                else "reject_without_rescue_tuning"
            ),
        }
    return {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "strategy_promotion": False,
        "economic_edge_claim": False,
        "candidates": candidates,
        "retained_candidates": [
            name for name, result in candidates.items() if result["checks"]["retained"]
        ],
    }


def validate_only() -> dict[str, Any]:
    contract, _ = load_contract_and_verify(require_pre_score=False)
    tape, diagnostics = load_tape(return_diagnostics=True)
    features = build_feature_surface(tape)
    horizon_rows: dict[str, int] = {}
    validation_rows: dict[str, int] = {}
    for horizon in HORIZONS:
        surface = build_horizon_surface(features, horizon)
        horizon_rows[str(horizon)] = len(surface)
        validation_rows[str(horizon)] = int(
            surface["month_key"].isin(VALIDATION_MONTHS).sum()
        )
        diagnostics["skipped_support_reasons"]["insufficient_same_session_horizon"][
            str(horizon)
        ] = len(tape) - len(surface)
        del surface
    output = {
        "contract_id": contract["contract_id"],
        "mode": "target_blind_validate_only",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "feature_count": len(FULL_FEATURES),
        "feature_order": FULL_FEATURES,
        "horizon_rows": horizon_rows,
        "validation_rows": validation_rows,
        "data_diagnostics": diagnostics,
        "source_sha256_for_freeze": current_source_hashes(),
        "environment_versions": environment_versions(),
        "artifacts_written": False,
        "models_fitted": False,
        "validation_outcomes_attached": False,
    }
    print(json.dumps(safe(output), indent=2, sort_keys=True))
    return output


def run_scoring() -> dict[str, Any]:
    contract, source_manifest = load_contract_and_verify(require_pre_score=True)
    assert source_manifest is not None
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    tape, diagnostics = load_tape(return_diagnostics=True)
    features = build_feature_surface(tape)
    predictions, preprocessing, coefficients, folds, validation_rows = (
        fit_monthly_oof_predictions(tape, features)
    )
    for horizon in HORIZONS:
        diagnostics["skipped_support_reasons"]["insufficient_same_session_horizon"][
            str(horizon)
        ] = len(tape) - EXPECTED_ROWS[horizon]
    model_specification = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "clock_features": CLOCK_FEATURES,
        "full_features_in_order": FULL_FEATURES,
        "ridge_parameters": RIDGE_PARAMETERS,
        "hgb_parameters": HGB_PARAMETERS,
        "thresholds_bps": ACTION_THRESHOLDS,
        "costs_bps_per_side": COSTS,
    }
    write_json(OUT / "model_specification.json", model_specification)
    freeze = write_prediction_freeze(
        predictions, preprocessing, coefficients, folds, source_manifest
    )
    # This verification is the procedural boundary.  Validation prices and
    # outcomes are first constructed only after it succeeds.
    verify_prediction_freeze(freeze)
    outcomes = build_validation_outcomes(tape, features)
    outcomes.to_parquet(OUT / "validation_outcomes_2024.parquet", index=False)
    scored = predictions.merge(
        outcomes,
        on=[
            "anchor_id",
            "symbol_norm",
            "session_date",
            "decision_timestamp",
            "bar_ordinal",
            "horizon",
        ],
        how="left",
        validate="many_to_one",
    )
    if scored["target_bps"].isna().any() or len(scored) != len(predictions):
        raise AssertionError("prediction-to-validation-outcome join failure")
    scored["clock_quartile"] = np.minimum(
        scored["bar_ordinal"].to_numpy(int) * 4 // SESSION_BARS, 3
    ).astype(np.int8)
    scored.to_parquet(OUT / "scored_predictions_2024.parquet", index=False)
    prediction_metrics, monthly_prediction, deciles, daily_prediction = (
        evaluate_predictions(scored)
    )
    prediction_deletions = evaluate_prediction_stock_deletions(scored)
    entries = build_accepted_entries(scored)
    entries.to_parquet(OUT / "accepted_entries_2024.parquet", index=False)
    session_dates = sorted(
        date for date in tape["session_date"].unique() if date[:7] in VALIDATION_MONTHS
    )
    action_metrics, daily_action, monthly_action, action_slices, deletions = (
        evaluate_actions(entries, session_dates)
    )
    bootstraps = build_bootstraps(scored, daily_prediction, daily_action)
    decision = final_decisions(
        prediction_metrics,
        monthly_prediction,
        action_metrics,
        monthly_action,
        deletions,
        bootstraps,
    )
    prediction_metrics.to_csv(OUT / "prediction_metrics.csv", index=False)
    monthly_prediction.to_csv(OUT / "monthly_prediction_metrics.csv", index=False)
    prediction_deletions.to_csv(OUT / "prediction_stock_deletions.csv", index=False)
    deciles.to_csv(OUT / "prediction_deciles.csv", index=False)
    daily_prediction.to_parquet(OUT / "daily_prediction_metrics.parquet", index=False)
    action_metrics.to_csv(OUT / "action_metrics.csv", index=False)
    daily_action.to_parquet(OUT / "daily_action_returns.parquet", index=False)
    monthly_action.to_csv(OUT / "monthly_action_metrics.csv", index=False)
    action_slices.to_csv(OUT / "action_slices.csv", index=False)
    deletions.to_csv(OUT / "stock_deletions.csv", index=False)
    bootstraps.to_csv(OUT / "bootstrap_intervals.csv", index=False)
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "data_diagnostics.json", diagnostics)
    write_json(
        OUT / "source_hashes.json",
        {
            **source_manifest,
            "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH),
            "prediction_action_freeze_sha256": sha256(
                OUT / "prediction_action_freeze.json"
            ),
        },
    )
    primary_prediction = prediction_metrics.loc[
        prediction_metrics["algorithm"].isin(CANDIDATES)
    ].to_dict(orient="records")
    primary_action = action_metrics.loc[
        action_metrics["algorithm"].isin(CANDIDATES)
        & action_metrics["threshold_bps"].eq(PRIMARY_THRESHOLD)
        & action_metrics["cost_bps_per_side"].eq(PRIMARY_COST)
    ].to_dict(orient="records")
    summary = {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "regular_rows": len(tape),
        "symbols": len(SYMBOLS),
        "union_sessions": len(set(tape["session_date"])),
        "validation_session_dates": len(session_dates),
        "validation_rows_by_horizon": validation_rows,
        "prediction_rows": len(predictions),
        "accepted_entry_rows_all_thresholds": len(entries),
        "primary_prediction_metrics": primary_prediction,
        "primary_action_metrics": primary_action,
        "bootstraps": bootstraps.to_dict(orient="records"),
        "decision": decision,
    }
    write_json(OUT / "summary.json", summary)
    files = sorted(path for path in OUT.iterdir() if path.is_file())
    write_json(
        OUT / "artifact_manifest.json",
        {
            "contract_id": CONTRACT_ID,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "provider_volume_label": "historical_volume_not_used",
            "files": [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ],
        },
    )
    print(json.dumps(safe(summary), indent=2, sort_keys=True))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate 2024 source bars, features, and exact support without fitting or scoring",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_only:
        validate_only()
    else:
        run_scoring()


if __name__ == "__main__":
    main()
