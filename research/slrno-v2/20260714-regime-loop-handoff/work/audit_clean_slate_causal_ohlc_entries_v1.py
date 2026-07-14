"""Independent audit for clean-slate causal OHLC entries V1.

This module deliberately does not import the experiment runner.  It rebuilds the
2024 bar tape, features, cohorts, models, actions, and statistical summaries from
the frozen contract and provider OHLC files.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


WORK = Path(__file__).resolve().parent
CONTRACT_PATH = WORK / "contracts/20260712-clean-slate-causal-ohlc-entries-v1.json"
PRE_SCORE_PATH = WORK / "contracts/20260712-clean-slate-causal-ohlc-entries-v1-pre-score.json"
RUNNER_PATH = WORK / "run_clean_slate_causal_ohlc_entries_v1.py"
RAW_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/"
    "instrument_type=stock"
)
ARTIFACT_ROOT = Path("/private/tmp/stocker_clean_slate_causal_ohlc_entries_v1_20260712")
ENVIRONMENT_ROOT = Path("/Users/michaelsalerno/StockerLocal")

SYMBOLS = (
    "AAL", "AAOI", "APLD", "ASTS", "AXTI", "CIFR", "HIMS", "IONQ", "IREN",
    "MARA", "MP", "MRNA", "MSTR", "NVTS", "OKLO", "QBTS", "RGTI", "RIOT",
    "RIVN", "SMCI", "SOFI", "WULF",
)
HORIZONS = (6, 12, 24)
MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
MODELS = ("clock_ridge", "full_ridge", "full_hgb")
THRESHOLDS = (10.0, 20.0, 40.0)
COSTS = (0, 1, 2, 5, 10)
PRIMARY_THRESHOLD = 10
PRIMARY_COST = 5
SEED = 20260712
SESSION_MINUTE = 570
SESSION_END_MINUTE = 960
SESSION_BARS = 78
BLOCK_LENGTH = 5
BOOTSTRAP_DRAWS = 5000
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
FORBIDDEN_INPUT_WORDS = (
    "regime", "state", "loop", "cycle", "b0", "template", "volume",
    "order_flow", "quote", "tick", "news", "fundamental",
)
EXPECTED_REGULAR_ROWS = 424583
EXPECTED_UNION_SESSIONS = 252
EXPECTED_SYMBOL_SESSIONS = 5539
EXPECTED_GAPS = 2612
EXPECTED_ALL_ROWS = {6: 383168, 12: 347620, 24: 280982}
EXPECTED_VALIDATION_ROWS = {6: 195292, 12: 177276, 24: 143472}


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
        "environment_pyproject": ENVIRONMENT_ROOT / "pyproject.toml",
        "environment_uv_lock": ENVIRONMENT_ROOT / "uv.lock",
    }
    paths.update(
        {f"provider_full_file_{symbol}": provider_path(symbol) for symbol in SYMBOLS}
    )
    return paths


def assert_contract(contract: dict[str, Any]) -> None:
    if contract["contract_id"] != "clean_slate_causal_ohlc_entries_v1":
        raise AssertionError("unexpected contract id")
    if not (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
    ):
        raise AssertionError("research safety boundary changed")
    if tuple(contract["universe"]["symbols"]) != SYMBOLS:
        raise AssertionError("universe changed")
    if tuple(contract["cohort"]["horizons_bars"]) != HORIZONS:
        raise AssertionError("horizons changed")
    if tuple(contract["periods"]["validation_months"]) != MONTHS:
        raise AssertionError("validation months changed")
    if tuple(contract["feature_policy"]["clock_features"]) != CLOCK_FEATURES:
        raise AssertionError("clock feature order changed")
    if tuple(contract["feature_policy"]["full_features_in_order"]) != FULL_FEATURES:
        raise AssertionError("full feature order changed")
    if contract["sources"]["columns_read"] != list(RAW_COLUMNS):
        raise AssertionError("source column whitelist changed")
    if contract["sources"]["provider_volume_label"] != "historical_volume_not_used":
        raise AssertionError("volume provenance changed")
    if contract["periods"]["2025_read_permitted"]:
        raise AssertionError("2025 access unexpectedly permitted")
    if contract["periods"]["2023_read_permitted"]:
        raise AssertionError("2023 access unexpectedly permitted")
    if contract["periods"]["2026_read_permitted"]:
        raise AssertionError("2026 access unexpectedly permitted")
    if contract["actions"]["primary_threshold_gross_bps"] != PRIMARY_THRESHOLD:
        raise AssertionError("primary threshold changed")
    if tuple(contract["actions"]["descriptive_thresholds_gross_bps"]) != (20.0, 40.0):
        raise AssertionError("descriptive thresholds changed")
    if tuple(contract["costs"]["grid_bps_per_side"]) != COSTS:
        raise AssertionError("cost grid changed")


def ast_source_boundary(runner_path: Path) -> dict[str, Any]:
    """Reject imports and literal data paths outside the fresh 2024 OHLC surface."""

    source = runner_path.read_text()
    tree = ast.parse(source, filename=str(runner_path))
    forbidden_imports: list[str] = []
    suspicious_paths: list[str] = []
    parquet_read_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            lowered = name.lower()
            if any(word in lowered for word in ("regime", "loop", "semimarkov", "detector")):
                forbidden_imports.append(name)
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "read_parquet":
                    parquet_read_calls += 1
                    keywords = {item.arg for item in node.keywords}
                    if not {"columns", "filters"}.issubset(keywords):
                        raise AssertionError(
                            "runner parquet read lacks a column whitelist or year predicate"
                        )
            continue
        value = node.value
        lowered = value.lower()
        path_like = "/" in value or value.endswith((".parquet", ".csv", ".json"))
        if path_like and any(year in lowered for year in ("2023", "2025", "2026")):
            # Contract and artifact names legitimately carry the run date 20260712.
            cleaned = lowered.replace("20260712", "")
            if any(year in cleaned for year in ("2023", "2025", "2026")):
                suspicious_paths.append(value)
    if forbidden_imports:
        raise AssertionError(f"runner imports forbidden prior-model modules: {forbidden_imports}")
    if suspicious_paths:
        raise AssertionError(f"runner contains later/backward-period paths: {suspicious_paths}")
    if parquet_read_calls != 1:
        raise AssertionError(f"unexpected runner parquet read count: {parquet_read_calls}")
    return {
        "ast_nodes": sum(1 for _ in ast.walk(tree)),
        "forbidden_imports": forbidden_imports,
        "suspicious_paths": suspicious_paths,
        "predicate_column_whitelisted_parquet_reads": parquet_read_calls,
    }


def _read_provider_2024(path: Path) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    stop = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return pd.read_parquet(
        path,
        columns=list(RAW_COLUMNS),
        filters=[("timestamp", ">=", start), ("timestamp", "<", stop)],
    )


def load_regular_tape() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only 2024 timestamp/OHLC and independently create exact segments."""

    frames: list[pd.DataFrame] = []
    per_symbol: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        frame = _read_provider_2024(provider_path(symbol)).copy()
        timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if timestamp.isna().any():
            raise AssertionError(f"unparseable 2024 timestamp for {symbol}")
        frame["timestamp"] = timestamp
        if not timestamp.ge(pd.Timestamp("2024-01-01", tz="UTC")).all():
            raise AssertionError("provider predicate returned a pre-2024 row")
        if not timestamp.lt(pd.Timestamp("2025-01-01", tz="UTC")).all():
            raise AssertionError("provider predicate returned a post-2024 row")
        if frame["timestamp"].duplicated(keep=False).any():
            raise AssertionError(f"duplicate 2024 provider timestamps for {symbol}")
        local = timestamp.dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        regular = minute.ge(SESSION_MINUTE) & minute.lt(SESSION_END_MINUTE)
        numeric = frame[["open", "high", "low", "close"]].to_numpy(float)
        finite_positive = np.isfinite(numeric).all(axis=1) & (numeric > 0).all(axis=1)
        order_ok = (
            (numeric[:, 2] <= np.minimum(numeric[:, 0], numeric[:, 3]))
            & (np.maximum(numeric[:, 0], numeric[:, 3]) <= numeric[:, 1])
        )
        valid = finite_positive & order_ok
        accepted = regular.to_numpy(bool) & valid
        selected = frame.loc[accepted].copy()
        selected_local = selected["timestamp"].dt.tz_convert("America/New_York")
        selected_minute = selected_local.dt.hour * 60 + selected_local.dt.minute
        if not (((selected_minute - SESSION_MINUTE) % 5) == 0).all():
            raise AssertionError(f"off-grid regular-session timestamp for {symbol}")
        selected["session_date"] = selected_local.dt.strftime("%Y-%m-%d")
        selected["symbol"] = symbol
        selected["symbol_norm"] = symbol
        selected["month"] = selected["session_date"].str[:7]
        selected["month_key"] = selected["month"]
        selected["clock_ordinal"] = ((selected_minute - SESSION_MINUTE) // 5).to_numpy(np.int16)
        selected["bar_ordinal"] = selected["clock_ordinal"]
        selected = selected.sort_values("timestamp", kind="stable").reset_index(drop=True)
        prior = selected.groupby("session_date", sort=False)["timestamp"].shift()
        continuation = (selected["timestamp"] - prior).eq(pd.Timedelta(minutes=5))
        gap_count = int((prior.notna() & ~continuation).sum())
        per_symbol.append(
            {
                "symbol_norm": symbol,
                "raw_predicate_2024_rows": int(len(frame)),
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
        ["symbol", "session_date", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    if tape.duplicated(["symbol", "timestamp"]).any():
        raise AssertionError("duplicate symbol timestamp")

    prior_timestamp = tape.groupby(["symbol", "session_date"], sort=False)["timestamp"].shift()
    delta = tape["timestamp"] - prior_timestamp
    continuation = delta.eq(pd.Timedelta(minutes=5))
    gap_count = int((prior_timestamp.notna() & ~continuation).sum())
    segment_start = ~continuation
    tape["segment_id"] = segment_start.cumsum().to_numpy(np.int64) - 1
    tape["segment_index"] = (
        segment_start.groupby([tape["symbol"], tape["session_date"]], sort=False)
        .cumsum()
        .to_numpy(np.int16)
        - 1
    )
    tape["segment_position"] = tape.groupby("segment_id", sort=False).cumcount().to_numpy(np.int16)
    tape["segment_size"] = tape.groupby("segment_id", sort=False)["timestamp"].transform("size").to_numpy(np.int16)

    local = tape["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    tape["clock_ordinal"] = ((minute - SESSION_MINUTE) // 5).to_numpy(np.int16)
    tape["bar_ordinal"] = tape["clock_ordinal"]
    tape["month"] = tape["session_date"].str[:7]
    tape["month_key"] = tape["month"]
    tape["source_position"] = np.arange(len(tape), dtype=np.int64)
    diagnostics: dict[str, Any] = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "raw_predicate_2024_rows": int(sum(row["raw_predicate_2024_rows"] for row in per_symbol)),
        "regular_valid_rows": len(tape),
        "union_sessions": tape["session_date"].nunique(),
        "symbol_sessions": tape[["symbol", "session_date"]].drop_duplicates().shape[0],
        "within_session_nonfive_minute_gaps": gap_count,
        "skipped_support_reasons": {
            "nonfinite_or_invalid_ohlc": int(sum(row["nonfinite_or_invalid_ohlc_rows"] for row in per_symbol)),
            "duplicate_timestamp": 0,
            "outside_regular_session": int(sum(row["outside_regular_session_rows"] for row in per_symbol)),
            "nonfive_minute_gap": gap_count,
            "insufficient_same_session_horizon": {},
        },
        "per_symbol": per_symbol,
    }
    if diagnostics["regular_valid_rows"] != EXPECTED_REGULAR_ROWS:
        raise AssertionError("regular row count drift")
    if diagnostics["union_sessions"] != EXPECTED_UNION_SESSIONS:
        raise AssertionError("union session count drift")
    if diagnostics["symbol_sessions"] != EXPECTED_SYMBOL_SESSIONS:
        raise AssertionError("symbol-session count drift")
    if diagnostics["within_session_nonfive_minute_gaps"] != EXPECTED_GAPS:
        raise AssertionError("gap count drift")
    return tape, diagnostics


def reconstruct_features(tape: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the frozen ordered features with no observation after each anchor."""

    frame = tape.copy()
    q = frame["bar_ordinal"].to_numpy(float) * 5.0 / 385.0
    frame["clock_fraction"] = q
    frame["clock_fraction_squared"] = q**2
    frame["clock_sin_1"] = np.sin(2.0 * np.pi * q)
    frame["clock_cos_1"] = np.cos(2.0 * np.pi * q)
    frame["clock_sin_2"] = np.sin(4.0 * np.pi * q)
    frame["clock_cos_2"] = np.cos(4.0 * np.pi * q)
    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    spread = highs - lows
    usable = spread > EPSILON
    frame["log_close_open"] = np.log(closes / opens)
    frame["log_high_low"] = np.log(highs / lows)
    for name, neutral in (
        ("signed_body_fraction", 0.0),
        ("absolute_body_fraction", 0.0),
        ("upper_wick_fraction", 0.0),
        ("lower_wick_fraction", 0.0),
        ("close_location", 0.5),
    ):
        frame[name] = neutral
    frame.loc[usable, "signed_body_fraction"] = (
        closes[usable] - opens[usable]
    ) / spread[usable]
    frame.loc[usable, "absolute_body_fraction"] = np.abs(
        closes[usable] - opens[usable]
    ) / spread[usable]
    frame.loc[usable, "upper_wick_fraction"] = (
        highs[usable] - np.maximum(opens[usable], closes[usable])
    ) / spread[usable]
    frame.loc[usable, "lower_wick_fraction"] = (
        np.minimum(opens[usable], closes[usable]) - lows[usable]
    ) / spread[usable]
    frame.loc[usable, "close_location"] = (
        closes[usable] - lows[usable]
    ) / spread[usable]

    segment_keys = ["symbol_norm", "session_date", "segment_index"]
    frame["_log_close"] = np.log(frame["close"])
    segments = frame.groupby(segment_keys, sort=False)
    for window in (1, 3, 6, 12):
        frame[f"close_return_{window}"] = frame["_log_close"] - segments[
            "_log_close"
        ].shift(window)
    frame["_absolute_return_1"] = frame["close_return_1"].abs()

    def rolling_series(column: str, window: int, operation: str) -> pd.Series:
        rolling = frame.groupby(segment_keys, sort=False)[column].rolling(
            window, min_periods=window
        )
        if operation == "mean":
            result = rolling.mean()
        elif operation == "std":
            result = rolling.std(ddof=0)
        elif operation == "maximum":
            result = rolling.max()
        elif operation == "minimum":
            result = rolling.min()
        else:
            raise AssertionError(operation)
        return result.reset_index(level=segment_keys, drop=True).sort_index()

    for window in (3, 6, 12):
        frame[f"mean_abs_close_return_{window}"] = rolling_series(
            "_absolute_return_1", window, "mean"
        )
        frame[f"std_close_return_{window}"] = rolling_series(
            "close_return_1", window, "std"
        )
        frame[f"mean_log_range_{window}"] = rolling_series(
            "log_high_low", window, "mean"
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
    running_spread = running_high.to_numpy(float) - running_low.to_numpy(float)
    frame["session_range_location"] = 0.5
    running_usable = running_spread > EPSILON
    frame.loc[running_usable, "session_range_location"] = (
        closes[running_usable] - running_low.to_numpy(float)[running_usable]
    ) / running_spread[running_usable]
    frame["running_log_range"] = np.log(running_high / running_low)

    for window in (6, 12):
        rolling_high = rolling_series("high", window, "maximum")
        rolling_low = rolling_series("low", window, "minimum")
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


def build_target_blind_surface(
    tape: pd.DataFrame, features: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    supported = tape["segment_position"].to_numpy(int) + horizon < tape[
        "segment_size"
    ].to_numpy(int)
    anchors = np.flatnonzero(supported)
    surface = tape.loc[
        anchors,
        [
            "source_position",
            "symbol_norm",
            "session_date",
            "month_key",
            "timestamp",
            "bar_ordinal",
            "segment_index",
            "segment_position",
        ],
    ].copy()
    surface = surface.rename(columns={"timestamp": "decision_timestamp"})
    surface = pd.concat(
        [surface.reset_index(drop=True), features.iloc[anchors].reset_index(drop=True)],
        axis=1,
    )
    surface["horizon"] = np.int16(horizon)
    surface["entry_timestamp"] = surface["decision_timestamp"] + pd.Timedelta(minutes=5)
    surface["exit_timestamp"] = surface["decision_timestamp"] + pd.Timedelta(
        minutes=5 * horizon
    )
    surface["anchor_id"] = (
        surface["symbol_norm"].astype(str)
        + "|"
        + surface["decision_timestamp"].astype(str)
        + f"|h{horizon}"
    )
    surface = surface.sort_values(
        ["symbol_norm", "session_date", "decision_timestamp"], kind="stable"
    ).reset_index(drop=True)
    if len(surface) != EXPECTED_ALL_ROWS[horizon]:
        raise AssertionError(f"h{horizon} cohort row mismatch")
    if int(surface["month_key"].isin(MONTHS).sum()) != EXPECTED_VALIDATION_ROWS[horizon]:
        raise AssertionError(f"h{horizon} validation row mismatch")
    if surface["anchor_id"].duplicated().any():
        raise AssertionError(f"duplicate h{horizon} anchor")
    return surface


def attach_prices_and_target(
    tape: pd.DataFrame, anchors: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    positions = anchors["source_position"].to_numpy(np.int64)
    entry_positions = positions + 1
    exit_positions = positions + horizon
    timestamps = tape["timestamp"].to_numpy()
    decision = anchors["decision_timestamp"].to_numpy()
    if not (
        (timestamps[entry_positions] - decision == np.timedelta64(5, "m")).all()
        and (
            timestamps[exit_positions] - decision
            == np.timedelta64(5 * horizon, "m")
        ).all()
    ):
        raise AssertionError(f"inexact h{horizon} execution support")
    for column in ("symbol_norm", "session_date"):
        tape_values = tape[column].astype(str).to_numpy()
        anchor_values = anchors[column].astype(str).to_numpy()
        if not (
            (tape_values[entry_positions] == anchor_values).all()
            and (tape_values[exit_positions] == anchor_values).all()
        ):
            raise AssertionError(f"cross-boundary h{horizon} outcome")
    entry = tape["open"].to_numpy(float)[entry_positions]
    exit_value = tape["close"].to_numpy(float)[exit_positions]
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
    output["next_bar_open"] = entry
    output["exit_close"] = exit_value
    output["target_bps"] = 10000.0 * (exit_value / entry - 1.0)
    return output


def equal_symbol_weights(symbols: Iterable[str]) -> np.ndarray:
    series = pd.Series(list(symbols), dtype=str)
    counts = series.value_counts(sort=False)
    represented = len(counts)
    total = len(series)
    return series.map(lambda symbol: total / (represented * counts[symbol])).to_numpy(float)


def training_medians(matrix: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        medians = np.nanmedian(matrix, axis=0)
    if not np.isfinite(medians).all():
        raise AssertionError("a fold feature has no finite training median")
    return medians


def impute(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    rows, columns = np.where(np.isnan(result))
    result[rows, columns] = medians[columns]
    if not np.isfinite(result).all():
        raise AssertionError("nonfinite imputed design")
    return result


def threshold_label(threshold: float) -> str:
    return f"{int(threshold)}bps"


def independent_nonoverlap(
    frame: pd.DataFrame, actions: np.ndarray, horizon: int
) -> np.ndarray:
    accepted = np.zeros(len(frame), dtype=np.int8)
    ordered = frame.assign(_side=actions, _row=np.arange(len(frame))).sort_values(
        ["symbol_norm", "session_date", "decision_timestamp"], kind="stable"
    )
    for _, group in ordered.groupby(["symbol_norm", "session_date"], sort=False):
        eligible_at: pd.Timestamp | None = None
        for timestamp, side, row in zip(
            group["decision_timestamp"], group["_side"], group["_row"], strict=True
        ):
            if int(side) == 0:
                continue
            current = pd.Timestamp(timestamp)
            if eligible_at is None or current >= eligible_at:
                accepted[int(row)] = int(side)
                eligible_at = current + pd.Timedelta(minutes=5 * horizon)
    return accepted


def replay_all_models(
    tape: pd.DataFrame, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    """Independently replay target-blind validation prediction and fixed actions."""

    prediction_parts: list[pd.DataFrame] = []
    preprocessing_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    surfaces: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS:
        surface = build_target_blind_surface(tape, features, horizon)
        surfaces[horizon] = surface
        for month in MONTHS:
            boundary = pd.Timestamp(f"{month}-01", tz="UTC")
            train_mask = surface["exit_timestamp"].lt(boundary).to_numpy(bool)
            score_mask = surface["month_key"].eq(month).to_numpy(bool)
            if np.logical_and(train_mask, score_mask).any():
                raise AssertionError("training/scoring fold overlap")
            train = surface.loc[train_mask]
            score = surface.loc[score_mask]
            training_outcomes = attach_prices_and_target(tape, train, horizon)
            target = training_outcomes["target_bps"].to_numpy(float)
            weights = equal_symbol_weights(train["symbol_norm"])
            if not np.isclose(weights.mean(), 1.0, rtol=0.0, atol=1e-12):
                raise AssertionError("training weights are not normalized")
            symbol_totals = pd.Series(weights).groupby(
                train["symbol_norm"].to_numpy(str), sort=False
            ).sum()
            if not np.allclose(symbol_totals, symbol_totals.iloc[0], atol=1e-9, rtol=0.0):
                raise AssertionError("training symbols do not have equal total weight")
            for algorithm in MODELS:
                names = CLOCK_FEATURES if algorithm == "clock_ridge" else FULL_FEATURES
                raw_train = train.loc[:, names].to_numpy(float)
                raw_score = score.loc[:, names].to_numpy(float)
                medians = training_medians(raw_train)
                design_train = impute(raw_train, medians)
                design_score = impute(raw_score, medians)
                scaler_mean = np.full(len(names), np.nan)
                scaler_scale = np.full(len(names), np.nan)
                if algorithm.endswith("ridge"):
                    scaler = StandardScaler().fit(design_train, sample_weight=weights)
                    design_train = scaler.transform(design_train)
                    design_score = scaler.transform(design_score)
                    estimator: Any = Ridge(
                        alpha=10.0,
                        fit_intercept=True,
                        solver="lsqr",
                        tol=1e-6,
                    )
                    scaler_mean = scaler.mean_.astype(float)
                    scaler_scale = scaler.scale_.astype(float)
                else:
                    estimator = HistGradientBoostingRegressor(
                        loss="squared_error",
                        learning_rate=0.05,
                        max_iter=100,
                        max_leaf_nodes=7,
                        max_depth=3,
                        min_samples_leaf=500,
                        l2_regularization=10.0,
                        max_bins=64,
                        early_stopping=False,
                        random_state=SEED,
                    )
                estimator.fit(design_train, target, sample_weight=weights)
                prediction = estimator.predict(design_score).astype(float)
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
                    coefficient_rows.extend(
                        {
                            "fold_month": month,
                            "algorithm": algorithm,
                            "horizon": horizon,
                            "feature_order": order,
                            "feature": feature,
                            "coefficient": float(coefficient),
                        }
                        for order, (feature, coefficient) in enumerate(
                            zip(names, estimator.coef_, strict=True)
                        )
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
                        "fitted_iterations": (
                            int(estimator.n_iter_)
                            if isinstance(estimator, HistGradientBoostingRegressor)
                            else None
                        ),
                    }
                )
    ledger = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["algorithm", "horizon", "symbol_norm", "session_date", "decision_timestamp"],
        kind="stable",
    ).reset_index(drop=True)
    for threshold in THRESHOLDS:
        label = threshold_label(threshold)
        action = np.where(
            ledger["prediction_bps"].to_numpy(float) >= threshold,
            1,
            np.where(ledger["prediction_bps"].to_numpy(float) <= -threshold, -1, 0),
        ).astype(np.int8)
        ledger[f"action_{label}"] = action
        accepted = np.zeros(len(ledger), dtype=np.int8)
        for (_, horizon), group in ledger.groupby(["algorithm", "horizon"], sort=False):
            rows = group.index.to_numpy(int)
            accepted[rows] = independent_nonoverlap(group, action[rows], int(horizon))
        ledger[f"accepted_action_{label}"] = accepted
    return (
        ledger,
        pd.DataFrame(preprocessing_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(fold_rows),
        surfaces,
    )


def prediction_summary(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    observed = np.asarray(list(actual), dtype=float)
    forecast = np.asarray(list(predicted), dtype=float)
    error = forecast - observed
    observed_std = float(observed.std(ddof=0))
    forecast_std = float(forecast.std(ddof=0))
    centered_forecast = forecast - forecast.mean()
    variance = float(np.mean(centered_forecast**2))
    slope = (
        float(np.mean(centered_forecast * (observed - observed.mean())) / variance)
        if variance > 0
        else math.nan
    )
    return {
        "rows": len(observed),
        "mse_bps2": float(np.mean(error**2)),
        "mae_bps": float(np.mean(np.abs(error))),
        "pearson_correlation": (
            float(np.corrcoef(observed, forecast)[0, 1])
            if observed_std > 0 and forecast_std > 0
            else math.nan
        ),
        "spearman_correlation": (
            float(pd.Series(observed).corr(pd.Series(forecast), method="spearman"))
            if observed_std > 0 and forecast_std > 0
            else math.nan
        ),
        "calibration_intercept_bps": (
            float(observed.mean() - slope * forecast.mean())
            if math.isfinite(slope)
            else float(observed.mean())
        ),
        "calibration_slope": slope,
        "target_mean_bps": float(observed.mean()),
        "prediction_mean_bps": float(forecast.mean()),
        "target_std_bps": observed_std,
        "prediction_std_bps": forecast_std,
    }


def _with_clock_loss(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["horizon"] + (["month"] if "month" in frame.columns else [])
    clock = frame.loc[
        frame["algorithm"].eq("clock_ridge"),
        [*keys, "mse_bps2", "mae_bps"],
    ].rename(columns={"mse_bps2": "clock_mse_bps2", "mae_bps": "clock_mae_bps"})
    output = frame.merge(clock, on=keys, how="left", validate="many_to_one")
    output["relative_mse_improvement_vs_clock"] = (
        1.0 - output["mse_bps2"] / output["clock_mse_bps2"]
    )
    output["relative_mae_improvement_vs_clock"] = (
        1.0 - output["mae_bps"] / output["clock_mae_bps"]
    )
    return output


def recompute_prediction_outputs(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    session_grid = sorted(scored["session_date"].astype(str).unique())
    for (algorithm, horizon), group in scored.groupby(["algorithm", "horizon"], sort=False):
        pooled: dict[str, Any] = {
            "algorithm": algorithm,
            "horizon": int(horizon),
            **prediction_summary(group["target_bps"], group["prediction_bps"]),
        }
        for threshold in THRESHOLDS:
            action = group[f"action_{threshold_label(threshold)}"].to_numpy(int)
            label = int(threshold)
            pooled[f"long_rate_{label}bps"] = float((action == 1).mean())
            pooled[f"short_rate_{label}bps"] = float((action == -1).mean())
            pooled[f"abstain_rate_{label}bps"] = float((action == 0).mean())
        pooled_rows.append(pooled)

        ordering = np.lexsort(
            (
                group["anchor_id"].astype(str).to_numpy(),
                group["prediction_bps"].to_numpy(float),
            )
        )
        buckets = np.empty(len(group), dtype=np.int8)
        buckets[ordering] = np.minimum(np.arange(len(group)) * 10 // len(group), 9)
        ranked = group.assign(prediction_decile=buckets)
        for bucket, selected in ranked.groupby("prediction_decile", sort=True):
            decile_rows.append(
                {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "prediction_decile": int(bucket),
                    "rows": len(selected),
                    "prediction_mean_bps": float(selected["prediction_bps"].mean()),
                    "target_mean_bps": float(selected["target_bps"].mean()),
                    "positive_target_rate": float(selected["target_bps"].gt(0).mean()),
                }
            )
        for month, selected in group.groupby("fold_month", sort=True):
            monthly_rows.append(
                {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "month": month,
                    **prediction_summary(selected["target_bps"], selected["prediction_bps"]),
                }
            )
        error = group["prediction_bps"].to_numpy(float) - group["target_bps"].to_numpy(float)
        loss = pd.DataFrame(
            {
                "session_date": group["session_date"].to_numpy(str),
                "symbol_norm": group["symbol_norm"].to_numpy(str),
                "squared_error_bps2": error**2,
                "absolute_error_bps": np.abs(error),
            }
        )
        symbol_day = loss.groupby(["session_date", "symbol_norm"], sort=False)[
            ["squared_error_bps2", "absolute_error_bps"]
        ].mean()
        day = (symbol_day.groupby("session_date", sort=True).sum() / len(SYMBOLS)).reindex(
            session_grid, fill_value=0.0
        )
        for session_date, values in day.iterrows():
            daily_rows.append(
                {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "session_date": session_date,
                    "mse_bps2": float(values["squared_error_bps2"]),
                    "mae_bps": float(values["absolute_error_bps"]),
                }
            )

    pooled = _with_clock_loss(pd.DataFrame(pooled_rows))
    monthly = _with_clock_loss(pd.DataFrame(monthly_rows))

    deletion_rows: list[dict[str, Any]] = []
    for deleted_symbol in SYMBOLS:
        retained = scored.loc[scored["symbol_norm"].ne(deleted_symbol)]
        for (algorithm, horizon), group in retained.groupby(["algorithm", "horizon"], sort=False):
            deletion_rows.append(
                {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "deleted_symbol": deleted_symbol,
                    **prediction_summary(group["target_bps"], group["prediction_bps"]),
                }
            )
    deletions = pd.DataFrame(deletion_rows)
    deletion_clock = deletions.loc[
        deletions["algorithm"].eq("clock_ridge"),
        ["horizon", "deleted_symbol", "mse_bps2", "mae_bps"],
    ].rename(columns={"mse_bps2": "clock_mse_bps2", "mae_bps": "clock_mae_bps"})
    deletions = deletions.merge(
        deletion_clock,
        on=["horizon", "deleted_symbol"],
        how="left",
        validate="many_to_one",
    )
    deletions["relative_mse_improvement_vs_clock"] = (
        1.0 - deletions["mse_bps2"] / deletions["clock_mse_bps2"]
    )
    deletions["relative_mae_improvement_vs_clock"] = (
        1.0 - deletions["mae_bps"] / deletions["clock_mae_bps"]
    )
    return pooled, monthly, deletions, pd.DataFrame(decile_rows), pd.DataFrame(daily_rows)


def independently_accept_entries(scored: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    base = [
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
    for threshold in THRESHOLDS:
        accepted_column = f"accepted_action_{threshold_label(threshold)}"
        mask = scored[accepted_column].ne(0)
        selected = scored.loc[mask, base].copy()
        selected["threshold_bps"] = threshold
        selected["direction"] = scored.loc[mask, accepted_column].to_numpy(np.int8)
        selected["gross_return"] = selected["direction"].to_numpy(float) * (
            selected["exit_close"].to_numpy(float)
            / selected["next_bar_open"].to_numpy(float)
            - 1.0
        )
        selected["gross_return_bps"] = selected["gross_return"] * 10000.0
        selected["holding_bars"] = selected["horizon"].astype(np.int16)
        parts.append(selected)
    return pd.concat(parts, ignore_index=True).sort_values(
        [
            "algorithm",
            "horizon",
            "threshold_bps",
            "symbol_norm",
            "session_date",
            "decision_timestamp",
        ],
        kind="stable",
    ).reset_index(drop=True)


def portfolio_summary(values: Iterable[float]) -> dict[str, float]:
    daily = np.asarray(list(values), dtype=float)
    equity = np.cumprod(1.0 + daily)
    cumulative = float(equity[-1] - 1.0)
    standard_deviation = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    path = np.r_[1.0, equity]
    return {
        "cumulative_return": cumulative,
        "annualized_return": float((1 + cumulative) ** (252.0 / len(daily)) - 1.0),
        "annualized_volatility": standard_deviation * math.sqrt(252.0),
        "descriptive_sharpe_zero_rate": (
            float(daily.mean() / standard_deviation * math.sqrt(252.0))
            if standard_deviation > 0
            else math.nan
        ),
        "maximum_drawdown": float((path / np.maximum.accumulate(path) - 1).min(initial=0.0)),
        "mean_daily_return": float(daily.mean()),
    }


def fixed_sleeve_daily(
    entries: pd.DataFrame,
    session_dates: list[str],
    cost: int,
    deleted_symbol: str | None = None,
) -> tuple[pd.Series, np.ndarray]:
    selected = entries.copy()
    divisor = len(SYMBOLS)
    if deleted_symbol is not None:
        selected = selected.loc[selected["symbol_norm"].ne(deleted_symbol)].copy()
        divisor -= 1
    net = selected["gross_return"].to_numpy(float) - 2.0 * cost / 10000.0
    if selected.empty:
        return pd.Series(0.0, index=pd.Index(session_dates, name="session_date")), net
    selected["log_growth"] = np.log1p(net)
    sleeve = np.expm1(
        selected.groupby(["session_date", "symbol_norm"], sort=False)["log_growth"].sum()
    )
    daily = (sleeve.groupby("session_date").sum() / divisor).reindex(
        session_dates, fill_value=0.0
    )
    return daily, 10000.0 * net


def profit_factor(values: np.ndarray) -> float:
    positive = values[values > 0]
    negative = values[values < 0]
    if len(negative):
        return float(positive.sum() / -negative.sum())
    return math.inf if len(positive) else math.nan


def recompute_action_outputs(
    entries: pd.DataFrame, session_dates: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    exposure_denominator = SESSION_BARS * len(SYMBOLS) * len(session_dates)
    for algorithm in MODELS:
        for horizon in HORIZONS:
            for threshold in THRESHOLDS:
                group = entries.loc[
                    entries["algorithm"].eq(algorithm)
                    & entries["horizon"].eq(horizon)
                    & entries["threshold_bps"].eq(threshold)
                ]
                for cost in COSTS:
                    daily, net = fixed_sleeve_daily(group, session_dates, cost)
                    long_mask = group["direction"].eq(1).to_numpy(bool)
                    short_mask = group["direction"].eq(-1).to_numpy(bool)
                    metric_rows.append(
                        {
                            "algorithm": algorithm,
                            "horizon": horizon,
                            "threshold_bps": threshold,
                            "cost_bps_per_side": cost,
                            "accepted_trades": len(group),
                            "long_trades": int(long_mask.sum()),
                            "short_trades": int(short_mask.sum()),
                            "stocks_with_trade": int(group["symbol_norm"].nunique()),
                            "stocks_with_long_trade": int(group.loc[long_mask, "symbol_norm"].nunique()),
                            "stocks_with_short_trade": int(group.loc[short_mask, "symbol_norm"].nunique()),
                            "mean_net_trade_bps": float(net.mean()) if len(net) else math.nan,
                            "median_net_trade_bps": float(np.median(net)) if len(net) else math.nan,
                            "long_mean_net_bps": float(net[long_mask].mean()) if long_mask.any() else math.nan,
                            "short_mean_net_bps": float(net[short_mask].mean()) if short_mask.any() else math.nan,
                            "win_rate": float((net > 0).mean()) if len(net) else math.nan,
                            "profit_factor": profit_factor(net),
                            "exposure_fraction": float(group["holding_bars"].sum() / exposure_denominator),
                            **portfolio_summary(daily),
                        }
                    )
                    daily_frames.append(
                        pd.DataFrame(
                            {
                                "algorithm": algorithm,
                                "horizon": horizon,
                                "threshold_bps": threshold,
                                "cost_bps_per_side": cost,
                                "session_date": session_dates,
                                "daily_return": daily.to_numpy(float),
                            }
                        )
                    )
                    for month in MONTHS:
                        month_sessions = [date for date in session_dates if date.startswith(month)]
                        month_daily = daily.reindex(month_sessions, fill_value=0.0)
                        selected = group.loc[group["session_date"].str.startswith(month)]
                        selected_net = selected["gross_return"].to_numpy(float) * 10000.0 - 2 * cost
                        monthly_rows.append(
                            {
                                "algorithm": algorithm,
                                "horizon": horizon,
                                "threshold_bps": threshold,
                                "cost_bps_per_side": cost,
                                "month": month,
                                "session_dates": len(month_sessions),
                                "accepted_trades": len(selected),
                                "mean_net_trade_bps": float(selected_net.mean()) if len(selected_net) else math.nan,
                                **portfolio_summary(month_daily),
                            }
                        )
                    for side, label in ((1, "long"), (-1, "short")):
                        selected = group.loc[group["direction"].eq(side)]
                        selected_net = selected["gross_return"].to_numpy(float) * 10000 - 2 * cost
                        slice_rows.append(
                            {
                                "algorithm": algorithm,
                                "horizon": horizon,
                                "threshold_bps": threshold,
                                "cost_bps_per_side": cost,
                                "slice_type": "side",
                                "slice_value": label,
                                "accepted_trades": len(selected),
                                "stocks_with_trade": int(selected["symbol_norm"].nunique()),
                                "mean_net_trade_bps": float(selected_net.mean()) if len(selected_net) else math.nan,
                                "win_rate": float((selected_net > 0).mean()) if len(selected_net) else math.nan,
                            }
                        )
                    for quartile in range(4):
                        selected = group.loc[group["clock_quartile"].eq(quartile)]
                        selected_net = selected["gross_return"].to_numpy(float) * 10000 - 2 * cost
                        slice_rows.append(
                            {
                                "algorithm": algorithm,
                                "horizon": horizon,
                                "threshold_bps": threshold,
                                "cost_bps_per_side": cost,
                                "slice_type": "clock_quartile",
                                "slice_value": str(quartile),
                                "accepted_trades": len(selected),
                                "stocks_with_trade": int(selected["symbol_norm"].nunique()),
                                "mean_net_trade_bps": float(selected_net.mean()) if len(selected_net) else math.nan,
                                "win_rate": float((selected_net > 0).mean()) if len(selected_net) else math.nan,
                            }
                        )
                    for deleted_symbol in SYMBOLS:
                        deleted_daily, _ = fixed_sleeve_daily(
                            group, session_dates, cost, deleted_symbol
                        )
                        deletion_rows.append(
                            {
                                "algorithm": algorithm,
                                "horizon": horizon,
                                "threshold_bps": threshold,
                                "cost_bps_per_side": cost,
                                "deleted_symbol": deleted_symbol,
                                **portfolio_summary(deleted_daily),
                            }
                        )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(daily_frames, ignore_index=True),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(slice_rows),
        pd.DataFrame(deletion_rows),
    )


def moving_block(values: np.ndarray, offset: int) -> tuple[float, float, float]:
    data = np.asarray(values, float)
    starts = np.arange(len(data) - BLOCK_LENGTH + 1)
    blocks = math.ceil(len(data) / BLOCK_LENGTH)
    rng = np.random.default_rng(SEED + offset)
    sampled_starts = rng.choice(starts, size=(BOOTSTRAP_DRAWS, blocks), replace=True)
    positions = (
        sampled_starts[:, :, None] + np.arange(BLOCK_LENGTH)[None, None, :]
    ).reshape(BOOTSTRAP_DRAWS, -1)[:, : len(data)]
    sample_means = data[positions].mean(axis=1)
    lower, upper = np.quantile(sample_means, [0.025, 0.975], method="linear")
    return float(data.mean()), float(lower), float(upper)


def recompute_bootstraps(
    scored: pd.DataFrame, daily_prediction: pd.DataFrame, daily_action: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in ("full_ridge", "full_hgb"):
        for horizon_index, horizon in enumerate(HORIZONS):
            daily = daily_prediction.loc[daily_prediction["horizon"].eq(horizon)]
            clock_daily = daily.loc[daily["algorithm"].eq("clock_ridge")].sort_values("session_date")
            candidate_daily = daily.loc[daily["algorithm"].eq(candidate)].sort_values("session_date")
            pair = scored.loc[
                scored["horizon"].eq(horizon)
                & scored["algorithm"].isin(["clock_ridge", candidate]),
                ["anchor_id", "symbol_norm", "session_date", "algorithm", "target_bps", "prediction_bps"],
            ]
            clock_rows = pair.loc[pair["algorithm"].eq("clock_ridge")].drop(columns="algorithm").rename(
                columns={"target_bps": "clock_target", "prediction_bps": "clock_prediction"}
            )
            candidate_rows = pair.loc[pair["algorithm"].eq(candidate)].drop(columns="algorithm").rename(
                columns={"target_bps": "candidate_target", "prediction_bps": "candidate_prediction"}
            )
            joined = candidate_rows.merge(
                clock_rows,
                on=["anchor_id", "symbol_norm", "session_date"],
                how="inner",
                validate="one_to_one",
            )
            target = joined["candidate_target"].to_numpy(float)
            joined["loss_advantage"] = (
                joined["clock_prediction"].to_numpy(float) - target
            ) ** 2 - (
                joined["candidate_prediction"].to_numpy(float) - target
            ) ** 2
            symbol_day = joined.groupby(["session_date", "symbol_norm"], sort=False)[
                "loss_advantage"
            ].mean()
            improvement = (
                (symbol_day.groupby("session_date", sort=True).sum() / len(SYMBOLS))
                .reindex(clock_daily["session_date"].to_numpy(str), fill_value=0.0)
                .to_numpy(float)
            )
            observed, lower, upper = moving_block(improvement, horizon_index)
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": "clock_ridge",
                    "horizon": horizon,
                    "metric": "daily_mse_improvement_bps2",
                    "session_dates": len(candidate_daily),
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
            clock_action = action.loc[action["algorithm"].eq("clock_ridge")].sort_values("session_date")
            candidate_action = action.loc[action["algorithm"].eq(candidate)].sort_values("session_date")
            observed, lower, upper = moving_block(
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
            observed, lower, upper = moving_block(
                candidate_action["daily_return"].to_numpy(float)
                - clock_action["daily_return"].to_numpy(float),
                200 + horizon_index,
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


def independent_decision(
    prediction_metrics: pd.DataFrame,
    monthly_prediction: pd.DataFrame,
    action_metrics: pd.DataFrame,
    monthly_action: pd.DataFrame,
    action_deletions: pd.DataFrame,
    bootstraps: pd.DataFrame,
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for candidate in ("full_ridge", "full_hgb"):
        prediction = prediction_metrics.loc[prediction_metrics["algorithm"].eq(candidate)]
        prediction_months = monthly_prediction.loc[monthly_prediction["algorithm"].eq(candidate)]
        action = action_metrics.loc[
            action_metrics["algorithm"].eq(candidate)
            & action_metrics["threshold_bps"].eq(PRIMARY_THRESHOLD)
            & action_metrics["cost_bps_per_side"].eq(PRIMARY_COST)
        ]
        action_months = monthly_action.loc[
            monthly_action["algorithm"].eq(candidate)
            & monthly_action["threshold_bps"].eq(PRIMARY_THRESHOLD)
            & monthly_action["cost_bps_per_side"].eq(PRIMARY_COST)
        ]
        deletion = action_deletions.loc[
            action_deletions["algorithm"].eq(candidate)
            & action_deletions["threshold_bps"].eq(PRIMARY_THRESHOLD)
            & action_deletions["cost_bps_per_side"].eq(PRIMARY_COST)
        ]
        inference = bootstraps.loc[bootstraps["candidate"].eq(candidate)]
        month_wins = (
            prediction_months.assign(
                _win=prediction_months["relative_mse_improvement_vs_clock"].gt(0)
            )
            .groupby("horizon", sort=True)["_win"]
            .sum()
        )
        checks = {
            "minimum_nonoverlapping_trades_each_horizon": bool(
                len(action) == 3 and action["accepted_trades"].ge(500).all()
            ),
            "minimum_trades_each_validation_month_and_horizon": bool(
                len(action_months) == 18 and action_months["accepted_trades"].ge(50).all()
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
                len(inference.loc[inference["metric"].eq("daily_mse_improvement_bps2")]) == 3
                and inference.loc[
                    inference["metric"].eq("daily_mse_improvement_bps2"), "ci_lower"
                ].gt(0).all()
            ),
            "positive_pearson_correlation_each_horizon": bool(
                len(prediction) == 3 and prediction["pearson_correlation"].gt(0).all()
            ),
            "minimum_mse_improvement_months_each_horizon": bool(
                len(month_wins) == 3 and month_wins.ge(4).all()
            ),
            "positive_mean_net_trade_bps_each_horizon": bool(
                len(action) == 3 and action["mean_net_trade_bps"].gt(0).all()
            ),
            "positive_cumulative_return_each_horizon": bool(
                len(action) == 3 and action["cumulative_return"].gt(0).all()
            ),
            "absolute_daily_return_bootstrap_lower_above_zero_each_horizon": bool(
                len(inference.loc[inference["metric"].eq("absolute_daily_return")]) == 3
                and inference.loc[
                    inference["metric"].eq("absolute_daily_return"), "ci_lower"
                ].gt(0).all()
            ),
            "paired_daily_return_advantage_vs_clock_lower_above_zero_each_horizon": bool(
                len(inference.loc[inference["metric"].eq("daily_return_advantage")]) == 3
                and inference.loc[
                    inference["metric"].eq("daily_return_advantage"), "ci_lower"
                ].gt(0).all()
            ),
            "positive_cumulative_return_each_month_and_horizon": bool(
                len(action_months) == 18 and action_months["cumulative_return"].gt(0).all()
            ),
            "positive_cumulative_return_every_stock_deletion_and_horizon": bool(
                len(deletion) == len(SYMBOLS) * len(HORIZONS)
                and deletion["cumulative_return"].gt(0).all()
            ),
            "positive_long_mean_net_bps_each_horizon": bool(
                len(action) == 3 and action["long_mean_net_bps"].gt(0).all()
            ),
            "positive_short_mean_net_bps_each_horizon": bool(
                len(action) == 3 and action["short_mean_net_bps"].gt(0).all()
            ),
        }
        checks["retained"] = bool(all(checks.values()))
        decisions[candidate] = {
            "checks": checks,
            "decision": (
                "retain_2024_internal_entry_hypothesis_for_new_shadow_sessions"
                if checks["retained"]
                else "reject_without_rescue_tuning"
            ),
        }
    return {
        "contract_id": "clean_slate_causal_ohlc_entries_v1",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "strategy_promotion": False,
        "economic_edge_claim": False,
        "candidates": decisions,
        "retained_candidates": [
            name for name, result in decisions.items() if result["checks"]["retained"]
        ],
    }


def compare_frames(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    label: str,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> float:
    if list(observed.columns) != list(expected.columns):
        raise AssertionError(
            f"{label} columns differ: {list(observed.columns)} != {list(expected.columns)}"
        )
    if len(observed) != len(expected):
        raise AssertionError(f"{label} rows differ: {len(observed)} != {len(expected)}")
    maximum = 0.0
    for column in observed.columns:
        left = observed[column]
        right = expected[column]
        if pd.api.types.is_datetime64_any_dtype(left) or pd.api.types.is_datetime64_any_dtype(right):
            left_time = pd.to_datetime(left, utc=True).astype("int64").to_numpy()
            right_time = pd.to_datetime(right, utc=True).astype("int64").to_numpy()
            if not np.array_equal(left_time, right_time):
                raise AssertionError(f"{label}:{column} timestamp mismatch")
        elif pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_value = left.to_numpy(float)
            right_value = right.to_numpy(float)
            if not np.allclose(
                left_value, right_value, rtol=rtol, atol=atol, equal_nan=True
            ):
                difference = np.abs(left_value - right_value)
                where = int(np.nanargmax(difference)) if np.isfinite(difference).any() else -1
                raise AssertionError(
                    f"{label}:{column} numerical mismatch near row {where}: "
                    f"observed={left_value[where]!r} expected={right_value[where]!r} "
                    f"abs_error={difference[where]!r} "
                    f"observed_row={observed.iloc[where].to_dict()!r} "
                    f"expected_row={expected.iloc[where].to_dict()!r}"
                )
            delta = np.abs(left_value - right_value)
            finite = delta[np.isfinite(delta)]
            if len(finite):
                maximum = max(maximum, float(finite.max()))
        else:
            left_text = left.where(left.notna(), "<NA>").astype(str).reset_index(drop=True)
            right_text = right.where(right.notna(), "<NA>").astype(str).reset_index(drop=True)
            if not left_text.equals(right_text):
                mismatch = int(np.flatnonzero(left_text.to_numpy() != right_text.to_numpy())[0])
                raise AssertionError(f"{label}:{column} text mismatch at row {mismatch}")
    return maximum


def prefix_causality_check(tape: pd.DataFrame, features: pd.DataFrame) -> int:
    candidates = np.unique(
        np.linspace(0, len(tape) - 1, num=96, dtype=int)
    )
    checked = 0
    for position in candidates:
        row = tape.iloc[position]
        session = tape.loc[
            tape["symbol_norm"].eq(row["symbol_norm"])
            & tape["session_date"].eq(row["session_date"])
            & tape["timestamp"].le(row["timestamp"])
        ].copy().reset_index(drop=True)
        rebuilt = reconstruct_features(session).iloc[-1].to_numpy(float)
        reference = features.iloc[position].to_numpy(float)
        if not np.allclose(rebuilt, reference, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise AssertionError(f"feature prefix causality mismatch at tape row {position}")
        checked += 1
    return checked


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    if not ARTIFACT_ROOT.is_dir():
        raise FileNotFoundError(f"experiment output is absent: {ARTIFACT_ROOT}")
    checks: list[dict[str, Any]] = []

    def record(name: str, detail: Any) -> None:
        checks.append({"name": name, "pass": True, "detail": json_safe(detail)})

    contract = json.loads(CONTRACT_PATH.read_text())
    assert_contract(contract)
    record("frozen_contract_and_research_safety", True)
    record("runner_ast_forbidden_input_and_later_path_boundary", ast_source_boundary(RUNNER_PATH))

    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    if not (
        pre_score["contract_id"] == "clean_slate_causal_ohlc_entries_v1"
        and pre_score["research_only"] is True
        and pre_score["live_ordering_enabled"] is False
        and pre_score["order_placement"] == "disabled"
    ):
        raise AssertionError("pre-score safety or identity mismatch")
    actual_sources = {name: sha256_file(path) for name, path in source_paths().items()}
    if pre_score["sha256"] != actual_sources:
        raise AssertionError("frozen source hash mismatch")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if pre_score.get("environment_versions") != versions:
        raise AssertionError("frozen runtime environment mismatch")
    record("whole_provider_runner_contract_environment_hashes", len(actual_sources))

    expected_model_specification = {
        "contract_id": "clean_slate_causal_ohlc_entries_v1",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "clock_features": list(CLOCK_FEATURES),
        "full_features_in_order": list(FULL_FEATURES),
        "ridge_parameters": {
            "alpha": 10.0,
            "fit_intercept": True,
            "solver": "lsqr",
            "tol": 1e-6,
        },
        "hgb_parameters": {
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
        },
        "thresholds_bps": list(THRESHOLDS),
        "costs_bps_per_side": list(COSTS),
    }
    observed_model_specification = json.loads(
        (ARTIFACT_ROOT / "model_specification.json").read_text()
    )
    if observed_model_specification != expected_model_specification:
        raise AssertionError("artifact model specification mismatch")
    if any(
        any(word in feature.lower() for word in FORBIDDEN_INPUT_WORDS)
        for feature in FULL_FEATURES
    ):
        raise AssertionError("forbidden feature in independent whitelist")
    record("exact_feature_algorithm_threshold_and_cost_whitelist", len(FULL_FEATURES))

    freeze = json.loads((ARTIFACT_ROOT / "prediction_action_freeze.json").read_text())
    if not (
        freeze["stage"]
        == "predictions_and_fixed_actions_written_before_validation_outcome_join"
        and freeze["validation_prices_or_outcomes_present"] is False
        and freeze["frozen_source_manifest_sha256"] == sha256_file(PRE_SCORE_PATH)
        and freeze["frozen_source_sha256"] == actual_sources
    ):
        raise AssertionError("pre-outcome freeze boundary mismatch")
    for filename, expected_hash in freeze["sha256"].items():
        if sha256_file(ARTIFACT_ROOT / filename) != expected_hash:
            raise AssertionError(f"pre-outcome frozen artifact changed: {filename}")
    record("prediction_and_action_freeze_precedes_outcomes", freeze["prediction_rows"])

    tape, diagnostics = load_regular_tape()
    features = reconstruct_features(tape)
    if list(features.columns) != list(FULL_FEATURES):
        raise AssertionError("independent feature order mismatch")
    record("exact_2024_regular_tape_and_segments", {
        "rows": len(tape),
        "sessions": diagnostics["union_sessions"],
        "symbol_sessions": diagnostics["symbol_sessions"],
        "gaps": diagnostics["within_session_nonfive_minute_gaps"],
    })
    record("sampled_explicit_prefix_causality", prefix_causality_check(tape, features))

    predictions, preprocessing, coefficients, folds, surfaces = replay_all_models(
        tape, features
    )
    allowed_pre_outcome_columns = {
        "anchor_id",
        "fold_month",
        "symbol_norm",
        "session_date",
        "decision_timestamp",
        "bar_ordinal",
        "algorithm",
        "horizon",
        "prediction_bps",
        *(f"action_{threshold_label(value)}" for value in THRESHOLDS),
        *(f"accepted_action_{threshold_label(value)}" for value in THRESHOLDS),
    }
    if set(predictions.columns) != allowed_pre_outcome_columns:
        raise AssertionError("independent pre-outcome ledger columns mismatch")
    forbidden_ledger_tokens = ("target", "outcome", "price", "open", "high", "low", "close", "return")
    if any(
        any(token in column.lower() for token in forbidden_ledger_tokens)
        for column in predictions.columns
    ):
        raise AssertionError("independent pre-outcome ledger contains an outcome field")

    observed_predictions = pd.read_parquet(
        ARTIFACT_ROOT / "prediction_actions_pre_outcome.parquet"
    )
    preprocessing_error = compare_frames(
        pd.read_csv(ARTIFACT_ROOT / "fold_preprocessing.csv"),
        preprocessing,
        "fold_preprocessing",
        rtol=1e-9,
        atol=1e-10,
    )
    coefficient_error = compare_frames(
        pd.read_csv(ARTIFACT_ROOT / "ridge_coefficients.csv"),
        coefficients,
        "ridge_coefficients",
        rtol=1e-9,
        atol=1e-9,
    )
    fold_error = compare_frames(
        pd.read_csv(ARTIFACT_ROOT / "fold_metadata.csv"),
        folds,
        "fold_metadata",
        rtol=1e-9,
        atol=1e-9,
    )
    prediction_error = compare_frames(
        observed_predictions,
        predictions,
        "pre_outcome_predictions",
        rtol=1e-9,
        atol=1e-8,
    )
    record(
        "exact_folds_medians_weighted_scalers_models_predictions_actions_nonoverlap",
        {
            "rows": len(predictions),
            "prediction_max_abs_error": prediction_error,
            "preprocessing_max_abs_error": preprocessing_error,
            "coefficient_max_abs_error": coefficient_error,
            "fold_max_abs_error": fold_error,
        },
    )

    # Validation prices are first opened here, after the independent target-blind
    # ledger and the on-disk pre-outcome freeze have both been verified.
    outcome_parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        validation = surfaces[horizon].loc[surfaces[horizon]["month_key"].isin(MONTHS)]
        outcome_parts.append(attach_prices_and_target(tape, validation, horizon))
        diagnostics["skipped_support_reasons"]["insufficient_same_session_horizon"][str(horizon)] = (
            len(tape) - EXPECTED_ALL_ROWS[horizon]
        )
    outcomes = pd.concat(outcome_parts, ignore_index=True).sort_values(
        ["horizon", "symbol_norm", "session_date", "decision_timestamp"], kind="stable"
    ).reset_index(drop=True)
    outcome_error = compare_frames(
        pd.read_parquet(ARTIFACT_ROOT / "validation_outcomes_2024.parquet"),
        outcomes,
        "validation_outcomes",
        rtol=1e-12,
        atol=1e-10,
    )
    record("exact_next_open_fixed_close_targets", {
        "rows": len(outcomes),
        "max_abs_error": outcome_error,
    })

    join_keys = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "decision_timestamp",
        "bar_ordinal",
        "horizon",
    ]
    scored = predictions.merge(
        outcomes, on=join_keys, how="left", validate="many_to_one"
    )
    scored["clock_quartile"] = np.minimum(
        scored["bar_ordinal"].to_numpy(int) * 4 // SESSION_BARS, 3
    ).astype(np.int8)
    scored_error = compare_frames(
        pd.read_parquet(ARTIFACT_ROOT / "scored_predictions_2024.parquet"),
        scored,
        "scored_predictions",
        rtol=1e-9,
        atol=1e-8,
    )
    record("exact_post_freeze_outcome_join", scored_error)

    (
        prediction_metrics,
        monthly_prediction,
        prediction_deletions,
        prediction_deciles,
        daily_prediction,
    ) = recompute_prediction_outputs(scored)
    prediction_table_errors = {
        "pooled": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "prediction_metrics.csv"),
            prediction_metrics,
            "prediction_metrics",
        ),
        "monthly": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "monthly_prediction_metrics.csv"),
            monthly_prediction,
            "monthly_prediction_metrics",
        ),
        "stock_deletions": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "prediction_stock_deletions.csv"),
            prediction_deletions,
            "prediction_stock_deletions",
        ),
        "deciles": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "prediction_deciles.csv"),
            prediction_deciles,
            "prediction_deciles",
        ),
        "fixed_symbol_daily": compare_frames(
            pd.read_parquet(ARTIFACT_ROOT / "daily_prediction_metrics.parquet"),
            daily_prediction,
            "daily_prediction_metrics",
        ),
    }
    record("exact_prediction_metrics_equal_symbol_daily_and_loso", prediction_table_errors)

    entries = independently_accept_entries(scored)
    entries_error = compare_frames(
        pd.read_parquet(ARTIFACT_ROOT / "accepted_entries_2024.parquet"),
        entries,
        "accepted_entries",
        rtol=1e-9,
        atol=1e-9,
    )
    session_dates = sorted(
        date for date in tape["session_date"].unique() if date[:7] in MONTHS
    )
    action_metrics, daily_action, monthly_action, action_slices, action_deletions = (
        recompute_action_outputs(entries, session_dates)
    )
    action_table_errors = {
        "accepted_entries": entries_error,
        "metrics": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "action_metrics.csv"),
            action_metrics,
            "action_metrics",
        ),
        "daily": compare_frames(
            pd.read_parquet(ARTIFACT_ROOT / "daily_action_returns.parquet"),
            daily_action,
            "daily_action_returns",
        ),
        "monthly": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "monthly_action_metrics.csv"),
            monthly_action,
            "monthly_action_metrics",
        ),
        "slices": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "action_slices.csv"),
            action_slices,
            "action_slices",
        ),
        "deletions": compare_frames(
            pd.read_csv(ARTIFACT_ROOT / "stock_deletions.csv"),
            action_deletions,
            "stock_deletions",
        ),
    }
    record("exact_execution_costs_fixed_sleeves_metrics_and_slices", action_table_errors)

    bootstraps = recompute_bootstraps(scored, daily_prediction, daily_action)
    bootstrap_error = compare_frames(
        pd.read_csv(ARTIFACT_ROOT / "bootstrap_intervals.csv"),
        bootstraps,
        "bootstrap_intervals",
        rtol=1e-9,
        atol=1e-10,
    )
    record("exact_paired_non_circular_moving_block_bootstraps", bootstrap_error)
    decision = independent_decision(
        prediction_metrics,
        monthly_prediction,
        action_metrics,
        monthly_action,
        action_deletions,
        bootstraps,
    )
    observed_decision = json.loads((ARTIFACT_ROOT / "decision.json").read_text())
    if observed_decision != decision:
        raise AssertionError("retention decision/gate mismatch")
    record("exact_all_horizon_retention_gates", decision["retained_candidates"])

    observed_diagnostics = json.loads((ARTIFACT_ROOT / "data_diagnostics.json").read_text())
    if observed_diagnostics != json_safe(diagnostics):
        raise AssertionError("data diagnostics mismatch")
    record("exact_skipped_data_diagnostics", True)

    source_artifact = json.loads((ARTIFACT_ROOT / "source_hashes.json").read_text())
    expected_source_artifact = {
        **pre_score,
        "pre_score_manifest_sha256": sha256_file(PRE_SCORE_PATH),
        "prediction_action_freeze_sha256": sha256_file(
            ARTIFACT_ROOT / "prediction_action_freeze.json"
        ),
    }
    if source_artifact != expected_source_artifact:
        raise AssertionError("source binding artifact mismatch")
    record("exact_pre_score_and_prediction_freeze_source_binding", True)

    summary = json.loads((ARTIFACT_ROOT / "summary.json").read_text())
    scalar_summary = {
        "contract_id": "clean_slate_causal_ohlc_entries_v1",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "regular_rows": len(tape),
        "symbols": len(SYMBOLS),
        "union_sessions": len(set(tape["session_date"])),
        "validation_session_dates": len(session_dates),
        "validation_rows_by_horizon": {
            str(horizon): EXPECTED_VALIDATION_ROWS[horizon] for horizon in HORIZONS
        },
        "prediction_rows": len(predictions),
        "accepted_entry_rows_all_thresholds": len(entries),
    }
    for key, expected in scalar_summary.items():
        if summary[key] != expected:
            raise AssertionError(f"summary scalar mismatch: {key}")
    expected_summary_prediction = prediction_metrics.loc[
            prediction_metrics["algorithm"].isin(["full_ridge", "full_hgb"])
        ].reset_index(drop=True)
    observed_summary_prediction = pd.DataFrame(
        summary["primary_prediction_metrics"]
    ).reindex(columns=expected_summary_prediction.columns)
    compare_frames(
        observed_summary_prediction,
        expected_summary_prediction,
        "summary_primary_prediction_metrics",
    )
    expected_summary_action = action_metrics.loc[
            action_metrics["algorithm"].isin(["full_ridge", "full_hgb"])
            & action_metrics["threshold_bps"].eq(PRIMARY_THRESHOLD)
            & action_metrics["cost_bps_per_side"].eq(PRIMARY_COST)
        ].reset_index(drop=True)
    observed_summary_action = pd.DataFrame(summary["primary_action_metrics"]).reindex(
        columns=expected_summary_action.columns
    )
    compare_frames(
        observed_summary_action,
        expected_summary_action,
        "summary_primary_action_metrics",
    )
    observed_summary_bootstraps = pd.DataFrame(summary["bootstraps"]).reindex(
        columns=bootstraps.columns
    )
    compare_frames(
        observed_summary_bootstraps, bootstraps, "summary_bootstraps"
    )
    if summary["decision"] != decision:
        raise AssertionError("summary decision mismatch")
    record("exact_summary_binding", True)

    artifact_manifest = json.loads((ARTIFACT_ROOT / "artifact_manifest.json").read_text())
    if not (
        artifact_manifest["research_only"] is True
        and artifact_manifest["live_ordering_enabled"] is False
        and artifact_manifest["order_placement"] == "disabled"
        and artifact_manifest["provider_volume_label"] == "historical_volume_not_used"
    ):
        raise AssertionError("artifact manifest safety mismatch")
    listed = {row["name"]: row for row in artifact_manifest["files"]}
    actual_files = {
        path.name: path
        for path in ARTIFACT_ROOT.iterdir()
        if path.is_file() and path.name not in {"artifact_manifest.json", "independent_audit.json"}
    }
    if set(listed) != set(actual_files):
        raise AssertionError("artifact manifest file-set mismatch")
    for name, path in actual_files.items():
        if listed[name]["bytes"] != path.stat().st_size or listed[name]["sha256"] != sha256_file(path):
            raise AssertionError(f"artifact hash mismatch: {name}")
    record("exact_artifact_manifest_hashes", len(actual_files))

    result = {
        "audit": "clean_slate_causal_ohlc_entries_v1_independent",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "later_periods_read": False,
        "forbidden_prior_detector_inputs_used": False,
        "all_passed": True,
        "passed": len(checks),
        "failed": 0,
        "checks": checks,
    }
    audit_path = ARTIFACT_ROOT / "independent_audit.json"
    audit_path.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n")
    manifest_files = sorted(
        path
        for path in ARTIFACT_ROOT.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    refreshed_manifest = {
        "contract_id": "clean_slate_causal_ohlc_entries_v1",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in manifest_files
        ],
    }
    (ARTIFACT_ROOT / "artifact_manifest.json").write_text(
        json.dumps(refreshed_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
