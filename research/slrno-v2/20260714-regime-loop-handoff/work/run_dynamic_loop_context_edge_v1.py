#!/usr/bin/env python3
"""Research-only dynamic loop x context profitability and drift test.

The script consumes frozen causal forecasts and a frozen execution ledger.  It
cannot connect to a broker, submit an order, or modify application code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


WORK = Path(__file__).resolve().parent
CONTRACT_PATH = WORK / "contracts/20260713-dynamic-loop-context-edge-v1.json"
PRE_SCORE_PATH = WORK / "contracts/20260713-dynamic-loop-context-edge-v1-pre-score.json"
DEFAULT_OUT = Path("/private/tmp/stocker_dynamic_loop_context_edge_v1_20260713")

SEED = 20260713
HORIZONS = (6, 12, 24)
PERIODS = (2025, 2023)
FAMILIES = (
    "loop_only",
    "loop_current_regime",
    "loop_previous_regime",
    "loop_regime_path",
    "loop_direction",
    "loop_volatility",
    "loop_range",
    "loop_session",
    "loop_volume",
    "loop_joint_all",
)
PRIMARY = "loop_current_regime"
SOURCE_STRATEGY = "breakout_loop_scores_range_p75"
WINDOW = 60
MIN_SUPPORT = 20
PSEUDOCOUNT = 50.0
COST_PER_SIDE = 5
ROUND_TRIP_COST_BPS = 10.0
UNIVERSE_SIZE = 20
BLOCK_SESSIONS = 20
BLOCK_MIN_SUPPORT = 5
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BLOCK = 5
LOOP_COLUMNS = tuple(f"loop_score_{index:02d}" for index in range(1, 21))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
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


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def input_paths(contract: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "runner": Path(__file__).resolve(),
        "anchor_panel_2023": Path(contract["inputs"]["anchor_panels"]["2023"]),
        "anchor_panel_2024": Path(
            contract["inputs"]["anchor_panels"]["2024_threshold_fit"]
        ),
        "anchor_panel_2025": Path(contract["inputs"]["anchor_panels"]["2025"]),
        "accepted_signal_ledger": Path(
            contract["inputs"]["accepted_signal_ledger"]
        ),
        "fixed_cycles": Path(contract["inputs"]["fixed_cycles"]),
        "execution_manifest": Path(contract["inputs"]["execution_manifest"]),
    }
    for period in PERIODS:
        root = Path(contract["inputs"]["provider_roots"][str(period)])
        for symbol in contract["population"]["symbols"]:
            paths[f"provider_{period}_{symbol}"] = provider_path(root, symbol)
    return paths


def load_and_verify_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    if not (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["broker_connection_enabled"] is False
        and contract["paper_or_demo_execution_enabled"] is False
        and contract["strategy_promotion_allowed"] is False
    ):
        raise AssertionError("research-only safety boundary drift")
    if tuple(contract["population"]["horizons_bars"]) != HORIZONS:
        raise AssertionError("horizon drift")
    if contract["population"]["source_strategy"] != SOURCE_STRATEGY:
        raise AssertionError("source strategy drift")
    if contract["dynamic_selector"]["training_window_completed_sessions"] != WINDOW:
        raise AssertionError("window drift")
    if contract["dynamic_selector"]["minimum_filled_trade_support"] != MIN_SUPPORT:
        raise AssertionError("support drift")
    if float(contract["dynamic_selector"]["shrinkage_pseudocount"]) != PSEUDOCOUNT:
        raise AssertionError("shrinkage drift")
    paths = input_paths(contract)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    actual = {name: sha256(path) for name, path in paths.items()}
    if actual != pre_score["sha256"]:
        changed = sorted(
            set(actual)
            | set(pre_score["sha256"])
            - {
                name
                for name in set(actual) & set(pre_score["sha256"])
                if actual[name] == pre_score["sha256"][name]
            }
        )
        raise AssertionError(f"pre-score source hash mismatch: {changed}")
    return contract, pre_score


def context_thresholds(anchor_2024_path: Path) -> dict[str, float]:
    frame = pd.read_parquet(
        anchor_2024_path,
        columns=["session_return", "mean_abs_return_12", "bar_range_pct"],
    )
    if frame.empty or not np.isfinite(frame.to_numpy(float)).all():
        raise AssertionError("invalid 2024 context-threshold panel")
    direction_low, direction_high = np.quantile(
        frame["session_return"].to_numpy(float), [1.0 / 3.0, 2.0 / 3.0]
    )
    return {
        "direction_tertile_low": float(direction_low),
        "direction_tertile_high": float(direction_high),
        "volatility_median": float(np.median(frame["mean_abs_return_12"])),
        "range_median": float(np.median(frame["bar_range_pct"])),
        "training_rows": int(len(frame)),
    }


def load_cycles(path: Path) -> pd.DataFrame:
    cycles = pd.read_csv(path)
    required = {"cycle_id", "cycle", "transition_length"}
    if set(cycles.columns) != required or len(cycles) != 20:
        raise AssertionError("fixed-cycle dictionary drift")
    cycles = cycles.reset_index(drop=True)
    cycles["cycle_index"] = np.arange(1, 21, dtype=int)
    cycles["start_state"] = cycles["cycle"].str.split("->").str[0].astype(int)
    cycles["member_states"] = cycles["cycle"].map(
        lambda value: tuple(sorted({int(item) for item in str(value).split("->")}))
    )
    return cycles


def load_anchor_context(
    period: int,
    path: Path,
    cycles: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "state",
        "previous_state_1",
        "history_token",
        "bar_index_in_session",
        "session_return",
        "mean_abs_return_12",
        "bar_range_pct",
        *LOOP_COLUMNS,
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["start_timestamp"] = pd.to_datetime(
        frame["start_timestamp"], utc=True, errors="raise"
    )
    frame["session_date"] = frame["session_date"].astype(str)
    if frame["anchor_id"].duplicated().any():
        raise AssertionError(f"duplicate anchor {period}")
    if set(pd.to_datetime(frame["session_date"]).dt.year.unique()) != {period}:
        raise AssertionError(f"anchor period drift {period}")
    loop_values = frame.loc[:, LOOP_COLUMNS].to_numpy(float)
    if not np.isfinite(loop_values).all() or (loop_values < 0.0).any():
        raise AssertionError("invalid causal loop scores")
    top_index = np.argmax(loop_values, axis=1)
    frame["top_loop_index"] = top_index + 1
    frame["top_loop"] = cycles["cycle_id"].to_numpy(str)[top_index]
    frame["top_loop_cycle"] = cycles["cycle"].to_numpy(str)[top_index]
    frame["top_loop_probability"] = loop_values[np.arange(len(frame)), top_index]
    member_states = cycles["member_states"].to_numpy(object)[top_index]
    compatible = np.fromiter(
        (
            int(state) in members
            for state, members in zip(
                frame["state"].to_numpy(int), member_states, strict=True
            )
        ),
        dtype=bool,
        count=len(frame),
    )
    if not compatible.all():
        raise AssertionError("top-loop/current-state membership failure")
    token = frame["history_token"].to_numpy(int)
    decoded_previous_1 = (token % 72) // 8
    decoded_state = token % 8
    if not np.array_equal(decoded_previous_1, frame["previous_state_1"].to_numpy(int)):
        raise AssertionError("history-token previous-state decode failure")
    if not np.array_equal(decoded_state, frame["state"].to_numpy(int)):
        raise AssertionError("history-token current-state decode failure")
    direction = np.full(len(frame), "flat", dtype=object)
    direction[
        frame["session_return"].to_numpy(float)
        <= thresholds["direction_tertile_low"]
    ] = "down"
    direction[
        frame["session_return"].to_numpy(float)
        >= thresholds["direction_tertile_high"]
    ] = "up"
    frame["direction_bucket"] = direction
    frame["volatility_bucket"] = np.where(
        frame["mean_abs_return_12"].to_numpy(float)
        >= thresholds["volatility_median"],
        "high",
        "low",
    )
    frame["range_bucket"] = np.where(
        frame["bar_range_pct"].to_numpy(float) >= thresholds["range_median"],
        "large",
        "small",
    )
    ordinal = frame["bar_index_in_session"].to_numpy(int)
    if (ordinal < 0).any() or (ordinal > 53).any():
        raise AssertionError("anchor outside frozen entry clock")
    frame["session_bucket"] = np.select(
        [ordinal <= 11, ordinal <= 35], ["open", "middle"], default="late"
    )
    keep = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "state",
        "previous_state_1",
        "history_token",
        "bar_index_in_session",
        "session_return",
        "mean_abs_return_12",
        "bar_range_pct",
        "top_loop_index",
        "top_loop",
        "top_loop_cycle",
        "top_loop_probability",
        "direction_bucket",
        "volatility_bucket",
        "range_bucket",
        "session_bucket",
    ]
    return frame.loc[:, keep]


def volume_context_for_period(
    period: int, root: Path, symbols: list[str]
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    source_rows = 0
    valid_ratios = 0
    zero_volumes = 0
    for symbol in symbols:
        path = provider_path(root, symbol)
        frame = pd.read_parquet(path, columns=["timestamp", "volume"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "volume"])
        end = pd.Timestamp(f"{period + 1}-01-01", tz="UTC")
        start = pd.Timestamp(f"{period - 2}-01-01", tz="UTC")
        frame = frame.loc[frame["timestamp"].ge(start) & frame["timestamp"].lt(end)].copy()
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        frame = frame.loc[minute.ge(570) & minute.lt(960)].copy()
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        frame["session_date"] = local.dt.strftime("%Y-%m-%d")
        frame["bar_ordinal"] = frame.groupby("session_date", sort=False).cumcount()
        frame["baseline_volume"] = frame.groupby("bar_ordinal", sort=False)[
            "volume"
        ].transform(
            lambda values: values.shift(1).rolling(20, min_periods=10).median()
        )
        frame["volume_ratio"] = np.where(
            frame["baseline_volume"].to_numpy(float) > 0.0,
            frame["volume"].to_numpy(float) / frame["baseline_volume"].to_numpy(float),
            np.nan,
        )
        source_rows += len(frame)
        zero_volumes += int(frame["volume"].eq(0.0).sum())
        frame = frame.loc[
            pd.to_datetime(frame["session_date"]).dt.year.eq(period)
        ].copy()
        valid_ratios += int(np.isfinite(frame["volume_ratio"].to_numpy(float)).sum())
        frame["symbol_norm"] = symbol
        frame["volume_bucket"] = np.where(
            ~np.isfinite(frame["volume_ratio"].to_numpy(float)),
            "unknown",
            np.where(frame["volume_ratio"].to_numpy(float) >= 1.0, "high", "low"),
        )
        rows.append(
            frame.loc[
                :, ["symbol_norm", "timestamp", "session_date", "volume_ratio", "volume_bucket"]
            ].rename(columns={"timestamp": "start_timestamp"})
        )
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["symbol_norm", "start_timestamp"]).any():
        raise AssertionError(f"duplicate provider volume key {period}")
    sessions = sorted(result["session_date"].unique())
    audit = {
        "period": period,
        "provider_rows_in_causal_lookback": source_rows,
        "target_year_rows": len(result),
        "target_sessions": len(sessions),
        "valid_volume_ratios": valid_ratios,
        "unknown_volume_ratios": int(len(result) - valid_ratios),
        "zero_historical_volume_rows_in_lookback": zero_volumes,
        "volume_label": "historical_volume_activity_proxy",
        "quotes_or_ticks_used": False,
    }
    return result, sessions, audit


def family_key(frame: pd.DataFrame, family: str) -> pd.Series:
    loop = frame["top_loop"].astype(str)
    if family == "loop_only":
        return loop
    if family == "loop_current_regime":
        return loop + "|c" + frame["state"].astype(int).astype(str)
    if family == "loop_previous_regime":
        return loop + "|p" + frame["previous_state_1"].astype(int).astype(str)
    if family == "loop_regime_path":
        return (
            loop
            + "|c"
            + frame["state"].astype(int).astype(str)
            + "|p"
            + frame["previous_state_1"].astype(int).astype(str)
        )
    if family == "loop_direction":
        return loop + "|d=" + frame["direction_bucket"].astype(str)
    if family == "loop_volatility":
        return loop + "|v=" + frame["volatility_bucket"].astype(str)
    if family == "loop_range":
        return loop + "|r=" + frame["range_bucket"].astype(str)
    if family == "loop_session":
        return loop + "|s=" + frame["session_bucket"].astype(str)
    if family == "loop_volume":
        return loop + "|q=" + frame["volume_bucket"].astype(str)
    if family == "loop_joint_all":
        return (
            loop
            + "|c"
            + frame["state"].astype(int).astype(str)
            + "|p"
            + frame["previous_state_1"].astype(int).astype(str)
            + "|d="
            + frame["direction_bucket"].astype(str)
            + "|v="
            + frame["volatility_bucket"].astype(str)
            + "|r="
            + frame["range_bucket"].astype(str)
            + "|s="
            + frame["session_bucket"].astype(str)
            + "|q="
            + frame["volume_bucket"].astype(str)
        )
    raise AssertionError(f"unknown family {family}")


def build_analysis_ledger(
    contract: dict[str, Any],
    thresholds: dict[str, float],
    cycles: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, list[str]], pd.DataFrame]:
    source = pd.read_parquet(Path(contract["inputs"]["accepted_signal_ledger"]))
    source = source.loc[
        source["period"].astype(str).isin([str(period) for period in PERIODS])
        & source["strategy"].eq(SOURCE_STRATEGY)
        & source["horizon"].isin(HORIZONS)
    ].copy()
    if source.duplicated(["period", "horizon", "anchor_id"]).any():
        raise AssertionError("duplicate frozen signal")
    symbols = list(contract["population"]["symbols"])
    sessions_by_period: dict[int, list[str]] = {}
    volume_audits: list[dict[str, Any]] = []
    outputs: list[pd.DataFrame] = []
    for period in PERIODS:
        anchor_path = Path(contract["inputs"]["anchor_panels"][str(period)])
        anchors = load_anchor_context(period, anchor_path, cycles, thresholds)
        volume, sessions, volume_audit = volume_context_for_period(
            period,
            Path(contract["inputs"]["provider_roots"][str(period)]),
            symbols,
        )
        sessions_by_period[period] = sessions
        volume_audits.append(volume_audit)
        anchors = anchors.merge(
            volume.drop(columns="session_date"),
            on=["symbol_norm", "start_timestamp"],
            how="left",
            validate="one_to_one",
        )
        anchors["volume_bucket"] = anchors["volume_bucket"].fillna("unknown")
        selected = source.loc[source["period"].astype(str).eq(str(period))].copy()
        selected["start_timestamp"] = pd.to_datetime(
            selected["start_timestamp"], utc=True, errors="raise"
        )
        merge_columns = [
            column
            for column in anchors.columns
            if column
            not in {"anchor_id", "symbol_norm", "session_date", "start_timestamp"}
        ]
        merged = selected.merge(
            anchors.loc[:, ["anchor_id", *merge_columns]],
            on="anchor_id",
            how="left",
            validate="many_to_one",
        )
        if merged["top_loop"].isna().any():
            raise AssertionError(f"signal-to-anchor join failure {period}")
        if not merged["symbol_norm"].isin(symbols).all():
            raise AssertionError(f"universe drift {period}")
        merged["session_index"] = merged["session_date"].map(
            {date: index for index, date in enumerate(sessions)}
        )
        if merged["session_index"].isna().any():
            raise AssertionError(f"signal session absent from provider tape {period}")
        merged["session_index"] = merged["session_index"].astype(int)
        merged["net_return_bps"] = np.where(
            merged["status"].eq("filled"),
            merged["gross_return_bps"].to_numpy(float) - ROUND_TRIP_COST_BPS,
            np.nan,
        )
        for family in FAMILIES:
            merged[f"key__{family}"] = family_key(merged, family)
        outputs.append(merged)
    ledger = pd.concat(outputs, ignore_index=True)
    return ledger, sessions_by_period, pd.DataFrame(volume_audits)


def rolling_selector(
    ledger: pd.DataFrame, sessions_by_period: dict[int, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored_frames: list[pd.DataFrame] = []
    chronology_rows: list[dict[str, Any]] = []
    primary_state_rows: list[dict[str, Any]] = []
    for period in PERIODS:
        sessions = sessions_by_period[period]
        if len(sessions) != 250:
            raise AssertionError(f"expected 250 regular sessions in {period}, got {len(sessions)}")
        for horizon in HORIZONS:
            frame = ledger.loc[
                ledger["period"].astype(str).eq(str(period))
                & ledger["horizon"].eq(horizon)
            ].copy()
            if frame.empty:
                raise AssertionError(f"empty signal surface {period} h{horizon}")
            all_keys: dict[str, list[str]] = {}
            key_to_loop: dict[str, dict[str, str]] = {}
            for family in FAMILIES:
                pairs = frame[[f"key__{family}", "top_loop"]].drop_duplicates()
                if pairs[f"key__{family}"].duplicated().any():
                    raise AssertionError(f"context key crosses loops: {family}")
                all_keys[family] = sorted(pairs[f"key__{family}"].astype(str))
                key_to_loop[family] = dict(
                    zip(
                        pairs[f"key__{family}"].astype(str),
                        pairs["top_loop"].astype(str),
                        strict=True,
                    )
                )
            ages = {family: {key: 0 for key in all_keys[family]} for family in FAMILIES}
            for score_index in range(WINDOW, len(sessions)):
                train_start = score_index - WINDOW
                past = frame.loc[
                    frame["status"].eq("filled")
                    & frame["session_index"].ge(train_start)
                    & frame["session_index"].lt(score_index)
                ].copy()
                current = frame.loc[frame["session_index"].eq(score_index)].copy()
                chronology_rows.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "score_session": sessions[score_index],
                        "score_session_index": score_index,
                        "training_first_session": sessions[train_start],
                        "training_last_session": sessions[score_index - 1],
                        "training_filled_trades": len(past),
                        "same_session_outcomes_used": False,
                    }
                )
                if past.empty:
                    raise AssertionError("empty rolling training window")
                global_mean = float(past["net_return_bps"].mean())
                loop_group = past.groupby("top_loop", sort=False)["net_return_bps"].agg(
                    ["size", "sum"]
                )
                loop_estimate: dict[str, float] = {}
                loop_support: dict[str, int] = {}
                loop_supported: dict[str, bool] = {}
                for loop in sorted(frame["top_loop"].astype(str).unique()):
                    if loop in loop_group.index:
                        n_loop = int(loop_group.loc[loop, "size"])
                        sum_loop = float(loop_group.loc[loop, "sum"])
                    else:
                        n_loop = 0
                        sum_loop = 0.0
                    loop_support[loop] = n_loop
                    loop_supported[loop] = n_loop >= MIN_SUPPORT
                    loop_estimate[loop] = (
                        (sum_loop + PSEUDOCOUNT * global_mean) / (n_loop + PSEUDOCOUNT)
                        if n_loop > 0
                        else global_mean
                    )
                family_maps: dict[str, dict[str, dict[str, Any]]] = {}
                for family in FAMILIES:
                    if family == "loop_only":
                        cell_group = None
                    else:
                        cell_group = past.groupby(f"key__{family}", sort=False)[
                            "net_return_bps"
                        ].agg(["size", "sum"])
                    states: dict[str, dict[str, Any]] = {}
                    for key in all_keys[family]:
                        loop = key_to_loop[family][key]
                        if family == "loop_only":
                            support = loop_support[loop]
                            estimate = loop_estimate[loop]
                            individualized = False
                        else:
                            if cell_group is not None and key in cell_group.index:
                                support = int(cell_group.loc[key, "size"])
                                cell_sum = float(cell_group.loc[key, "sum"])
                            else:
                                support = 0
                                cell_sum = 0.0
                            individualized = support >= MIN_SUPPORT
                            estimate = (
                                (cell_sum + PSEUDOCOUNT * loop_estimate[loop])
                                / (support + PSEUDOCOUNT)
                                if individualized
                                else loop_estimate[loop]
                            )
                        active = bool(loop_supported[loop] and estimate > 0.0)
                        ages[family][key] = ages[family][key] + 1 if active else 0
                        states[key] = {
                            "estimate": float(estimate),
                            "support": int(support),
                            "loop_support": int(loop_support[loop]),
                            "individualized": bool(individualized),
                            "active": active,
                            "age": int(ages[family][key]),
                        }
                        if family == PRIMARY:
                            primary_state_rows.append(
                                {
                                    "period": period,
                                    "horizon": horizon,
                                    "session_date": sessions[score_index],
                                    "session_index": score_index,
                                    "cell_key": key,
                                    "top_loop": loop,
                                    "estimate_net_bps": float(estimate),
                                    "cell_support": int(support),
                                    "loop_support": int(loop_support[loop]),
                                    "individualized": bool(individualized),
                                    "active": active,
                                    "active_age_sessions": int(ages[family][key]),
                                }
                            )
                    family_maps[family] = states
                for family in FAMILIES:
                    key_column = f"key__{family}"
                    mapping = family_maps[family]
                    current[f"selector__{family}__estimate_net_bps"] = current[
                        key_column
                    ].map(lambda key: mapping[str(key)]["estimate"])
                    current[f"selector__{family}__support"] = current[key_column].map(
                        lambda key: mapping[str(key)]["support"]
                    )
                    current[f"selector__{family}__loop_support"] = current[
                        key_column
                    ].map(lambda key: mapping[str(key)]["loop_support"])
                    current[f"selector__{family}__individualized"] = current[
                        key_column
                    ].map(lambda key: mapping[str(key)]["individualized"])
                    current[f"selector__{family}__active"] = current[key_column].map(
                        lambda key: mapping[str(key)]["active"]
                    )
                    current[f"selector__{family}__active_age_sessions"] = current[
                        key_column
                    ].map(lambda key: mapping[str(key)]["age"])
                scored_frames.append(current)
    scored = pd.concat(scored_frames, ignore_index=True)
    if scored.empty:
        raise AssertionError("no scored rows")
    return scored, pd.DataFrame(chronology_rows), pd.DataFrame(primary_state_rows)


def portfolio_statistics(values: np.ndarray) -> dict[str, float]:
    daily = np.asarray(values, dtype=float)
    equity = np.cumprod(1.0 + daily)
    cumulative = float(equity[-1] - 1.0) if len(equity) else 0.0
    annualized = (
        float((1.0 + cumulative) ** (252.0 / len(daily)) - 1.0)
        if len(daily) and cumulative > -1.0
        else math.nan
    )
    volatility = float(np.std(daily, ddof=1) * math.sqrt(252.0)) if len(daily) > 1 else 0.0
    sharpe = (
        float(np.mean(daily) / np.std(daily, ddof=1) * math.sqrt(252.0))
        if len(daily) > 1 and np.std(daily, ddof=1) > 0.0
        else math.nan
    )
    running_max = np.maximum.accumulate(np.r_[1.0, equity])
    drawdown = np.r_[1.0, equity] / running_max - 1.0
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "descriptive_sharpe_zero_rate": sharpe,
        "maximum_drawdown": float(drawdown.min(initial=0.0)),
        "mean_daily_return": float(daily.mean()) if len(daily) else 0.0,
    }


def daily_returns(
    signals: pd.DataFrame, sessions: list[str], active_column: str | None
) -> pd.Series:
    selected = signals.loc[signals["status"].eq("filled")].copy()
    if active_column is not None:
        selected = selected.loc[selected[active_column].astype(bool)].copy()
    if selected.empty:
        return pd.Series(0.0, index=sessions, dtype=float)
    selected["net_return"] = selected["net_return_bps"].to_numpy(float) / 10000.0
    if selected["net_return"].le(-1.0).any():
        raise AssertionError("net sleeve trade lost full collateral")
    selected["log_growth"] = np.log1p(selected["net_return"].to_numpy(float))
    sleeve = np.expm1(
        selected.groupby(["session_date", "symbol_norm"], sort=False)["log_growth"].sum()
    )
    daily = sleeve.groupby("session_date").sum() / UNIVERSE_SIZE
    return daily.reindex(sessions, fill_value=0.0)


def evaluate_selectors(
    scored: pd.DataFrame, sessions_by_period: dict[int, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    quarter_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    selectors: list[str] = ["unfiltered", *FAMILIES]
    for period in PERIODS:
        score_sessions = sessions_by_period[period][WINDOW:]
        for horizon in HORIZONS:
            base = scored.loc[
                scored["period"].astype(str).eq(str(period))
                & scored["horizon"].eq(horizon)
            ].copy()
            for selector in selectors:
                active_column = (
                    None if selector == "unfiltered" else f"selector__{selector}__active"
                )
                active = (
                    np.ones(len(base), dtype=bool)
                    if active_column is None
                    else base[active_column].astype(bool).to_numpy()
                )
                selected = base.loc[active]
                filled = selected.loc[selected["status"].eq("filled")]
                net_bps = filled["net_return_bps"].to_numpy(float)
                daily = daily_returns(base, score_sessions, active_column)
                stats = portfolio_statistics(daily.to_numpy(float))
                positive = net_bps[net_bps > 0.0]
                negative = net_bps[net_bps < 0.0]
                profit_factor = (
                    float(positive.sum() / -negative.sum())
                    if len(negative) and -negative.sum() > 0.0
                    else (math.inf if len(positive) else math.nan)
                )
                individualized_fraction = (
                    0.0
                    if selector in {"unfiltered", "loop_only"}
                    else float(
                        base[f"selector__{selector}__individualized"].astype(bool).mean()
                    )
                )
                metric_rows.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "selector": selector,
                        "score_sessions": len(score_sessions),
                        "accepted_signals": len(selected),
                        "filled_trades": len(filled),
                        "active_signal_fraction": float(active.mean()),
                        "individualized_decision_fraction": individualized_fraction,
                        "mean_net_trade_bps": float(net_bps.mean()) if len(net_bps) else math.nan,
                        "median_net_trade_bps": float(np.median(net_bps)) if len(net_bps) else math.nan,
                        "win_rate": float((net_bps > 0.0).mean()) if len(net_bps) else math.nan,
                        "profit_factor": profit_factor,
                        **stats,
                    }
                )
                daily_frame = pd.DataFrame(
                    {
                        "period": period,
                        "horizon": horizon,
                        "selector": selector,
                        "session_date": score_sessions,
                        "daily_return": daily.to_numpy(float),
                    }
                )
                daily_rows.append(daily_frame)
                daily_frame["quarter"] = (
                    pd.to_datetime(daily_frame["session_date"]).dt.year.astype(str)
                    + "_q"
                    + pd.to_datetime(daily_frame["session_date"]).dt.quarter.astype(str)
                )
                for quarter, group in daily_frame.groupby("quarter", sort=True):
                    quarter_rows.append(
                        {
                            "period": period,
                            "horizon": horizon,
                            "selector": selector,
                            "quarter": quarter,
                            "sessions": len(group),
                            **portfolio_statistics(group["daily_return"].to_numpy(float)),
                        }
                    )
                if selector in {PRIMARY, "loop_only", "unfiltered"}:
                    for symbol in sorted(base["symbol_norm"].unique()):
                        deletion_base = base.loc[base["symbol_norm"].ne(symbol)].copy()
                        deletion_daily = daily_returns(
                            deletion_base, score_sessions, active_column
                        ) * (UNIVERSE_SIZE / (UNIVERSE_SIZE - 1))
                        deletion_rows.append(
                            {
                                "period": period,
                                "horizon": horizon,
                                "selector": selector,
                                "deleted_symbol": symbol,
                                "sessions": len(deletion_daily),
                                **portfolio_statistics(deletion_daily.to_numpy(float)),
                            }
                        )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(daily_rows, ignore_index=True),
        pd.DataFrame(quarter_rows),
        pd.DataFrame(deletion_rows),
    )


def moving_block_samples(values: np.ndarray, seed_offset: int) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED + seed_offset)
    starts = np.arange(len(data) - BOOTSTRAP_BLOCK + 1)
    blocks = math.ceil(len(data) / BOOTSTRAP_BLOCK)
    selected = rng.choice(starts, size=(BOOTSTRAP_DRAWS, blocks), replace=True)
    positions = (
        selected[:, :, None] + np.arange(BOOTSTRAP_BLOCK)[None, None, :]
    ).reshape(BOOTSTRAP_DRAWS, -1)[:, : len(data)]
    return data[positions].mean(axis=1)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, position in enumerate(order):
        running = max(running, (total - rank) * values[position])
        adjusted[position] = min(1.0, running)
    return adjusted


def bootstrap_primary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("absolute", None),
        ("versus_unfiltered", "unfiltered"),
        ("versus_loop_only", "loop_only"),
    )
    seed_offset = 0
    for period in PERIODS:
        for horizon in HORIZONS:
            surface = daily.loc[
                daily["period"].eq(period) & daily["horizon"].eq(horizon)
            ]
            candidate = surface.loc[surface["selector"].eq(PRIMARY)].sort_values(
                "session_date"
            )
            for comparison, baseline_name in comparisons:
                if baseline_name is None:
                    values = candidate["daily_return"].to_numpy(float)
                else:
                    baseline = surface.loc[
                        surface["selector"].eq(baseline_name)
                    ].sort_values("session_date")
                    if not candidate["session_date"].reset_index(drop=True).equals(
                        baseline["session_date"].reset_index(drop=True)
                    ):
                        raise AssertionError("paired bootstrap date alignment failure")
                    values = candidate["daily_return"].to_numpy(float) - baseline[
                        "daily_return"
                    ].to_numpy(float)
                samples = moving_block_samples(values, seed_offset)
                seed_offset += 1
                lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
                p_one_sided = (1.0 + float((samples <= 0.0).sum())) / (
                    BOOTSTRAP_DRAWS + 1.0
                )
                rows.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "comparison": comparison,
                        "candidate": PRIMARY,
                        "baseline": baseline_name or "zero",
                        "sessions": len(values),
                        "mean_daily_difference": float(values.mean()),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                        "p_one_sided": p_one_sided,
                    }
                )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = holm_adjust(result["p_one_sided"].to_numpy(float))
    result["passes_holm_0_05"] = (
        result["holm_adjusted_p"].lt(0.05)
        & result["mean_daily_difference"].gt(0.0)
    )
    return result


def age_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[0, 5, 10, 20, 40, np.inf],
        labels=["1-5", "6-10", "11-20", "21-40", "41+"],
        right=True,
    ).astype(str)


def lifecycle_metrics(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for period in PERIODS:
        for horizon in HORIZONS:
            surface = scored.loc[
                scored["period"].astype(str).eq(str(period))
                & scored["horizon"].eq(horizon)
                & scored[f"selector__{PRIMARY}__active"].astype(bool)
                & scored["status"].eq("filled")
            ].copy()
            surface["age_bin"] = age_bucket(
                surface[f"selector__{PRIMARY}__active_age_sessions"].astype(int)
            )
            for label in ("1-5", "6-10", "11-20", "21-40", "41+"):
                selected = surface.loc[surface["age_bin"].eq(label)]
                values = selected["net_return_bps"].to_numpy(float)
                rows.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "selector": PRIMARY,
                        "age_bin_sessions": label,
                        "filled_trades": len(values),
                        "mean_net_trade_bps": float(values.mean()) if len(values) else math.nan,
                        "win_rate": float((values > 0.0).mean()) if len(values) else math.nan,
                    }
                )
            age = surface[f"selector__{PRIMARY}__active_age_sessions"].to_numpy(int)
            values = surface["net_return_bps"].to_numpy(float)
            early = values[(age >= 1) & (age <= 10)]
            late = values[age >= 21]
            comparison_rows.append(
                {
                    "period": period,
                    "horizon": horizon,
                    "early_1_10_trades": len(early),
                    "early_1_10_mean_net_bps": float(early.mean()) if len(early) else math.nan,
                    "late_21_plus_trades": len(late),
                    "late_21_plus_mean_net_bps": float(late.mean()) if len(late) else math.nan,
                    "late_minus_early_mean_net_bps": (
                        float(late.mean() - early.mean()) if len(early) and len(late) else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(comparison_rows)


def active_episodes(primary_states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["period", "horizon", "cell_key"]
    for (period, horizon, cell_key), group in primary_states.groupby(
        group_columns, sort=True
    ):
        group = group.sort_values("session_index")
        active = group["active"].astype(bool).to_numpy()
        indices = group["session_index"].to_numpy(int)
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])
        for episode_number, (start, end) in enumerate(zip(starts, ends, strict=True), 1):
            selected = group.iloc[start : end + 1]
            if not np.all(np.diff(selected["session_index"].to_numpy(int)) == 1):
                raise AssertionError("nonconsecutive active episode")
            rows.append(
                {
                    "period": int(period),
                    "horizon": int(horizon),
                    "cell_key": str(cell_key),
                    "top_loop": str(selected["top_loop"].iloc[0]),
                    "episode_number": episode_number,
                    "start_session": str(selected["session_date"].iloc[0]),
                    "end_session": str(selected["session_date"].iloc[-1]),
                    "length_sessions": len(selected),
                    "start_estimate_net_bps": float(selected["estimate_net_bps"].iloc[0]),
                    "end_estimate_net_bps": float(selected["estimate_net_bps"].iloc[-1]),
                    "start_support": int(selected["cell_support"].iloc[0]),
                    "end_support": int(selected["cell_support"].iloc[-1]),
                    "individualized_all_sessions": bool(selected["individualized"].all()),
                }
            )
    return pd.DataFrame(rows)


def descriptive_drift(
    ledger: pd.DataFrame, sessions_by_period: dict[int, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_rows: list[pd.DataFrame] = []
    adjacent_rows: list[dict[str, Any]] = []
    streak_rows: list[dict[str, Any]] = []
    for period in PERIODS:
        sessions = sessions_by_period[period]
        for horizon in HORIZONS:
            frame = ledger.loc[
                ledger["period"].astype(str).eq(str(period))
                & ledger["horizon"].eq(horizon)
                & ledger["status"].eq("filled")
            ].copy()
            frame["block_index"] = frame["session_index"] // BLOCK_SESSIONS
            grouped = (
                frame.groupby(
                    ["block_index", f"key__{PRIMARY}", "top_loop", "state"],
                    sort=True,
                )["net_return_bps"]
                .agg(["size", "mean", "median"])
                .reset_index()
                .rename(
                    columns={
                        f"key__{PRIMARY}": "cell_key",
                        "size": "filled_trades",
                        "mean": "mean_net_trade_bps",
                        "median": "median_net_trade_bps",
                    }
                )
            )
            grouped["period"] = period
            grouped["horizon"] = horizon
            grouped["supported"] = grouped["filled_trades"].ge(BLOCK_MIN_SUPPORT)
            grouped["profitable"] = grouped["mean_net_trade_bps"].gt(0.0)
            grouped["block_start_session"] = grouped["block_index"].map(
                lambda index: sessions[int(index) * BLOCK_SESSIONS]
            )
            grouped["block_end_session"] = grouped["block_index"].map(
                lambda index: sessions[
                    min((int(index) + 1) * BLOCK_SESSIONS, len(sessions)) - 1
                ]
            )
            cell_rows.append(grouped)
            supported = grouped.loc[grouped["supported"]].copy()
            max_block = math.ceil(len(sessions) / BLOCK_SESSIONS)
            for block in range(max_block - 1):
                left = supported.loc[supported["block_index"].eq(block)]
                right = supported.loc[supported["block_index"].eq(block + 1)]
                joined = left[["cell_key", "mean_net_trade_bps", "profitable"]].merge(
                    right[["cell_key", "mean_net_trade_bps", "profitable"]],
                    on="cell_key",
                    suffixes=("_left", "_right"),
                    how="inner",
                )
                left_positive = set(left.loc[left["profitable"], "cell_key"])
                right_positive = set(right.loc[right["profitable"], "cell_key"])
                union = left_positive | right_positive
                jaccard = len(left_positive & right_positive) / len(union) if union else math.nan
                positive_denominator = int(joined["profitable_left"].sum())
                retention = (
                    float(
                        (
                            joined["profitable_left"]
                            & joined["profitable_right"]
                        ).sum()
                        / positive_denominator
                    )
                    if positive_denominator
                    else math.nan
                )
                correlation = (
                    float(
                        spearmanr(
                            joined["mean_net_trade_bps_left"],
                            joined["mean_net_trade_bps_right"],
                        ).statistic
                    )
                    if len(joined) >= 3
                    else math.nan
                )
                adjacent_rows.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "left_block": block,
                        "right_block": block + 1,
                        "common_supported_cells": len(joined),
                        "sign_agreement": (
                            float(
                                (
                                    joined["profitable_left"]
                                    == joined["profitable_right"]
                                ).mean()
                            )
                            if len(joined)
                            else math.nan
                        ),
                        "positive_cell_retention": retention,
                        "profitable_cell_jaccard": jaccard,
                        "spearman_cell_mean": correlation,
                    }
                )
            for cell_key, cell in supported.groupby("cell_key", sort=True):
                block_values = dict(
                    zip(
                        cell["block_index"].astype(int),
                        cell["profitable"].astype(bool),
                        strict=True,
                    )
                )
                episode = 0
                run_start: int | None = None
                prior_block: int | None = None
                for block in sorted(block_values):
                    profitable = block_values[block]
                    if profitable and (run_start is None or prior_block != block - 1):
                        if run_start is not None and prior_block is not None:
                            episode += 1
                            streak_rows.append(
                                {
                                    "period": period,
                                    "horizon": horizon,
                                    "cell_key": cell_key,
                                    "episode": episode,
                                    "start_block": run_start,
                                    "end_block": prior_block,
                                    "profitable_blocks": prior_block - run_start + 1,
                                    "approximate_sessions": (prior_block - run_start + 1)
                                    * BLOCK_SESSIONS,
                                }
                            )
                        run_start = block
                    if not profitable and run_start is not None and prior_block is not None:
                        episode += 1
                        streak_rows.append(
                            {
                                "period": period,
                                "horizon": horizon,
                                "cell_key": cell_key,
                                "episode": episode,
                                "start_block": run_start,
                                "end_block": prior_block,
                                "profitable_blocks": prior_block - run_start + 1,
                                "approximate_sessions": (prior_block - run_start + 1)
                                * BLOCK_SESSIONS,
                            }
                        )
                        run_start = None
                    prior_block = block
                if run_start is not None and prior_block is not None:
                    episode += 1
                    streak_rows.append(
                        {
                            "period": period,
                            "horizon": horizon,
                            "cell_key": cell_key,
                            "episode": episode,
                            "start_block": run_start,
                            "end_block": prior_block,
                            "profitable_blocks": prior_block - run_start + 1,
                            "approximate_sessions": (prior_block - run_start + 1)
                            * BLOCK_SESSIONS,
                        }
                    )
    return (
        pd.concat(cell_rows, ignore_index=True),
        pd.DataFrame(adjacent_rows),
        pd.DataFrame(streak_rows),
    )


def make_decision(
    metrics: pd.DataFrame,
    bootstraps: pd.DataFrame,
    lifetime: pd.DataFrame,
) -> dict[str, Any]:
    primary = metrics.loc[metrics["selector"].eq(PRIMARY)]
    positive_economics = bool(
        len(primary) == 6
        and primary["cumulative_return"].gt(0.0).all()
        and primary["mean_net_trade_bps"].gt(0.0).all()
    )
    versus_unfiltered = bootstraps.loc[
        bootstraps["comparison"].eq("versus_unfiltered")
    ]
    versus_loop = bootstraps.loc[bootstraps["comparison"].eq("versus_loop_only")]
    bootstrap_advantage = bool(
        len(versus_unfiltered) == 6
        and len(versus_loop) == 6
        and versus_unfiltered["ci_lower"].gt(0.0).all()
        and versus_loop["ci_lower"].gt(0.0).all()
    )
    all_holm = bool(len(bootstraps) == 18 and bootstraps["passes_holm_0_05"].all())
    coverage = bool(
        len(primary) == 6
        and primary["individualized_decision_fraction"].ge(0.20).all()
    )
    context_supported = positive_economics and bootstrap_advantage and all_holm and coverage
    finite_lifetime = bool(
        len(lifetime) == 6
        and lifetime["early_1_10_mean_net_bps"].gt(0.0).all()
        and lifetime["late_minus_early_mean_net_bps"].lt(0.0).all()
    )
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "broker_connection_enabled": False,
        "paper_or_demo_execution_enabled": False,
        "primary_selector": PRIMARY,
        "primary_cost_bps_per_side": COST_PER_SIDE,
        "checks": {
            "positive_cumulative_and_mean_net_trade_both_periods_all_horizons": positive_economics,
            "paired_bootstrap_lower_bounds_above_zero_vs_unfiltered_and_loop_only_both_periods_all_horizons": bootstrap_advantage,
            "all_18_primary_bootstrap_endpoints_pass_holm_0_05": all_holm,
            "individualized_decision_fraction_at_least_0_20_both_periods_all_horizons": coverage,
            "context_edge_supported": context_supported,
            "early_1_10_positive_and_late_21_plus_lower_both_periods_all_horizons": finite_lifetime,
            "overall_hypothesis_supported": bool(context_supported and finite_lifetime),
        },
        "decision": (
            "dynamic_loop_context_and_finite_lifetime_supported_retrospectively"
            if context_supported and finite_lifetime
            else "dynamic_loop_context_profitability_hypothesis_not_supported"
        ),
        "economic_edge_claim": False,
        "strategy_promotion": False,
        "prospective_validation_claim": False,
        "live_or_paper_use_authorized": False,
    }


def artifact_manifest(out: Path) -> dict[str, Any]:
    files = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    contract, pre_score = load_and_verify_contract()
    out.mkdir(parents=True)
    thresholds = context_thresholds(
        Path(contract["inputs"]["anchor_panels"]["2024_threshold_fit"])
    )
    cycles = load_cycles(Path(contract["inputs"]["fixed_cycles"]))
    ledger, sessions_by_period, volume_audit = build_analysis_ledger(
        contract, thresholds, cycles
    )
    scored, chronology, primary_states = rolling_selector(ledger, sessions_by_period)
    metrics, daily, quarters, deletions = evaluate_selectors(scored, sessions_by_period)
    bootstraps = bootstrap_primary(daily)
    lifecycle, lifetime = lifecycle_metrics(scored)
    episodes = active_episodes(primary_states)
    cell_blocks, adjacent, streaks = descriptive_drift(ledger, sessions_by_period)
    decision = make_decision(metrics, bootstraps, lifetime)

    output_columns = [
        "period",
        "horizon",
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "bar_ordinal",
        "status",
        "direction",
        "gross_return_bps",
        "net_return_bps",
        "holding_bars",
        "state",
        "previous_state_1",
        "history_token",
        "top_loop_index",
        "top_loop",
        "top_loop_cycle",
        "top_loop_probability",
        "direction_bucket",
        "volatility_bucket",
        "range_bucket",
        "session_bucket",
        "volume_ratio",
        "volume_bucket",
        "session_index",
    ]
    for family in FAMILIES:
        output_columns.extend(
            [
                f"key__{family}",
                f"selector__{family}__estimate_net_bps",
                f"selector__{family}__support",
                f"selector__{family}__loop_support",
                f"selector__{family}__individualized",
                f"selector__{family}__active",
                f"selector__{family}__active_age_sessions",
            ]
        )
    scored.loc[:, output_columns].sort_values(
        ["period", "horizon", "session_date", "symbol_norm", "start_timestamp"],
        kind="stable",
    ).to_parquet(out / "scored_signal_ledger.parquet", index=False)
    primary_states.sort_values(
        ["period", "horizon", "cell_key", "session_index"], kind="stable"
    ).to_parquet(out / "primary_cell_states.parquet", index=False)
    chronology.to_csv(out / "chronology_audit.csv", index=False)
    metrics.to_csv(out / "selector_metrics.csv", index=False)
    daily.to_parquet(out / "daily_portfolio_returns.parquet", index=False)
    quarters.to_csv(out / "quarter_metrics.csv", index=False)
    deletions.to_csv(out / "stock_deletion_metrics.csv", index=False)
    bootstraps.to_csv(out / "primary_bootstraps.csv", index=False)
    lifecycle.to_csv(out / "lifecycle_age_metrics.csv", index=False)
    lifetime.to_csv(out / "lifetime_comparisons.csv", index=False)
    episodes.to_csv(out / "active_episodes.csv", index=False)
    cell_blocks.to_csv(out / "cell_block_profitability.csv", index=False)
    adjacent.to_csv(out / "adjacent_block_drift.csv", index=False)
    streaks.to_csv(out / "profitable_cell_streaks.csv", index=False)
    volume_audit.to_csv(out / "historical_volume_audit.csv", index=False)
    write_json(out / "condition_thresholds_2024.json", thresholds)
    write_json(out / "decision.json", decision)
    write_json(
        out / "source_hashes.json",
        {
            **pre_score,
            "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH),
            "provider_volume_label": "historical_volume_activity_proxy",
            "bid_ask_quotes_used": False,
            "tick_or_quote_count_used": False,
        },
    )
    summary = {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scientific_status": contract["scientific_status"],
        "periods": list(PERIODS),
        "horizons": list(HORIZONS),
        "symbols": len(contract["population"]["symbols"]),
        "source_strategy": SOURCE_STRATEGY,
        "forecast_identity": contract["forecast_identity"],
        "rolling_window_sessions": WINDOW,
        "minimum_support": MIN_SUPPORT,
        "shrinkage_pseudocount": PSEUDOCOUNT,
        "primary_cost_bps_per_side": COST_PER_SIDE,
        "families": list(FAMILIES),
        "score_rows": len(scored),
        "score_sessions_by_period": {
            str(period): len(sessions_by_period[period]) - WINDOW for period in PERIODS
        },
        "decision": decision,
        "primary_metrics": metrics.loc[metrics["selector"].eq(PRIMARY)].to_dict(
            orient="records"
        ),
        "primary_bootstraps": bootstraps.to_dict(orient="records"),
        "lifetime_comparisons": lifetime.to_dict(orient="records"),
        "active_episode_summary": (
            episodes.groupby(["period", "horizon"])["length_sessions"]
            .agg(["size", "median", "mean", "max"])
            .reset_index()
            .to_dict(orient="records")
            if not episodes.empty
            else []
        ),
        "drift_summary": (
            adjacent.groupby(["period", "horizon"])[
                [
                    "common_supported_cells",
                    "sign_agreement",
                    "positive_cell_retention",
                    "profitable_cell_jaccard",
                    "spearman_cell_mean",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .to_dict(orient="records")
        ),
        "provider_volume_label": "historical_volume_activity_proxy",
        "quotes_or_ticks_used": False,
        "economic_edge_claim": False,
        "strategy_promotion": False,
        "prospective_validation_claim": False,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "artifact_manifest.json", artifact_manifest(out))


if __name__ == "__main__":
    main()
