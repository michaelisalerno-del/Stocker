#!/usr/bin/env python3
"""Research-only causal setup-condition test on 2024 monthly-OOF movement forecasts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WORK = Path(__file__).resolve().parent
CONTRACT_PATH = WORK / "contracts/20260712-causal-setup-conditions-v1.json"
PRE_SCORE_PATH = WORK / "contracts/20260712-causal-setup-conditions-v1-pre-score.json"
OOF_ROOT = Path("/private/tmp/stocker_regime_utility_ablation_v1_20260711")
OOF_PATH = OOF_ROOT / "oof_predictions_2024.parquet"
OOF_AUDIT_PATH = OOF_ROOT / "independent_audit.json"
THRESHOLD_PATH = Path(
    "/private/tmp/stocker_frozen_regime_loop_pnl_sanity_v1_20260712/"
    "prediction_thresholds_2024.csv"
)
RAW_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/"
    "instrument_type=stock"
)
OUT = Path("/private/tmp/stocker_causal_setup_conditions_v1_20260712")

SEED = 20260712
HORIZONS = (6, 12, 24)
COSTS = (0, 1, 2, 5, 10)
PRIMARY_COST = 5
UNIVERSE_SIZE = 22
SESSION_BARS = 78
ACTIVATION_BARS = 3
MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
THRESHOLDS = {
    6: 210.3204137212535,
    12: 283.3242044166901,
    24: 372.9191260003554,
}
SETUPS = (
    {
        "setup": "oco_anchor_breakout_all",
        "family": "oco_baseline",
        "exante": "all",
        "condition": "oco",
    },
    {
        "setup": "close_confirmed_all",
        "family": "close_confirmation",
        "exante": "all",
        "condition": "none",
    },
    {
        "setup": "history_gate_close_confirmed",
        "family": "movement_gate",
        "exante": "history_gate",
        "condition": "none",
    },
    {
        "setup": "history_gate_strong_close",
        "family": "strong_close",
        "exante": "history_gate",
        "condition": "strong_close",
    },
    {
        "setup": "history_gate_compression_close",
        "family": "compression",
        "exante": "history_gate_compression",
        "condition": "none",
    },
    {
        "setup": "history_gate_trend_aligned_close",
        "family": "trend_alignment",
        "exante": "history_gate",
        "condition": "trend_aligned",
    },
)
HYPOTHESES = (
    ("H1_close_confirmation", "close_confirmed_all", "oco_anchor_breakout_all"),
    ("H2_movement_gate", "history_gate_close_confirmed", "close_confirmed_all"),
    ("H3_strong_close", "history_gate_strong_close", "history_gate_close_confirmed"),
    ("H4_compression", "history_gate_compression_close", "history_gate_close_confirmed"),
    ("H5_trend_alignment", "history_gate_trend_aligned_close", "history_gate_close_confirmed"),
)


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


def provider_path(symbol: str) -> Path:
    return RAW_ROOT / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def source_paths(symbols: list[str]) -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "runner": Path(__file__).resolve(),
        "oof_predictions": OOF_PATH,
        "oof_independent_audit": OOF_AUDIT_PATH,
        "frozen_thresholds": THRESHOLD_PATH,
    }
    for symbol in symbols:
        paths[f"provider_2024_{symbol}"] = provider_path(symbol)
    return paths


def load_contract_and_verify(symbols: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    if not (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["strategy_promotion_permitted"] is False
    ):
        raise AssertionError("research-only boundary drift")
    if contract["movement_gate"]["thresholds_bps"] != {
        str(key): value for key, value in THRESHOLDS.items()
    }:
        raise AssertionError("movement threshold contract drift")
    actual = {name: sha256(path) for name, path in source_paths(symbols).items()}
    if actual != pre_score["sha256"]:
        raise AssertionError("pre-score source hash mismatch")
    return contract, pre_score


def load_oof() -> pd.DataFrame:
    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "month_key",
        "state",
        "history_token",
        *(f"prediction__history__future_range_bps__h{horizon}" for horizon in HORIZONS),
    ]
    frame = pd.read_parquet(OOF_PATH, columns=columns)
    frame["start_timestamp"] = pd.to_datetime(frame["start_timestamp"], utc=True)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    frame = frame.sort_values(
        ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)
    if len(frame) != 34169 or frame["anchor_id"].duplicated().any():
        raise AssertionError("OOF setup cohort drift")
    if tuple(sorted(frame["month_key"].unique())) != MONTHS:
        raise AssertionError("OOF setup month drift")
    audit = json.loads(OOF_AUDIT_PATH.read_text())
    if audit["all_passed"] is not True:
        raise AssertionError("parent OOF audit is not passing")
    return frame


def validate_threshold_source() -> None:
    frame = pd.read_csv(THRESHOLD_PATH)
    selected = frame.loc[frame["representation"].eq("raw_history")]
    observed = {
        int(row.horizon): float(row.prediction_p75_bps)
        for row in selected.itertuples(index=False)
    }
    if observed != THRESHOLDS:
        raise AssertionError("frozen threshold source mismatch")


def load_tape(symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = pd.read_parquet(
            provider_path(symbol),
            columns=["timestamp", "open", "high", "low", "close"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.loc[
            frame["timestamp"].ge(pd.Timestamp("2024-01-01", tz="UTC"))
            & frame["timestamp"].lt(pd.Timestamp("2025-01-01", tz="UTC"))
        ].dropna(subset=["timestamp", "open", "high", "low", "close"])
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        minutes = local.dt.hour * 60 + local.dt.minute
        frame = frame.loc[minutes.ge(570) & minutes.lt(960)].copy()
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        frame["session_date"] = local.dt.strftime("%Y-%m-%d")
        frame["symbol_norm"] = symbol
        frame["bar_ordinal"] = frame.groupby("session_date", sort=False).cumcount()
        frame["range_pct"] = (frame["high"] - frame["low"]) / frame["open"]
        if frame.empty or frame["timestamp"].duplicated().any():
            raise AssertionError(f"invalid 2024 provider tape {symbol}")
        frames.append(frame)
    tape = pd.concat(frames, ignore_index=True).sort_values(
        ["symbol_norm", "session_date", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    tape["tape_position"] = np.arange(len(tape), dtype=np.int64)
    return tape


def close_confirmation(
    tape: pd.DataFrame,
    positions: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
) -> dict[str, np.ndarray]:
    opens = tape["open"].to_numpy(float)
    highs = tape["high"].to_numpy(float)
    lows = tape["low"].to_numpy(float)
    closes = tape["close"].to_numpy(float)
    confirmed = np.zeros(len(positions), dtype=bool)
    direction = np.zeros(len(positions), dtype=np.int8)
    step = np.zeros(len(positions), dtype=int)
    strong = np.zeros(len(positions), dtype=bool)
    body_fraction = np.full(len(positions), np.nan)
    outer_fraction = np.full(len(positions), np.nan)
    entry = np.full(len(positions), np.nan)
    for row, position in enumerate(positions):
        for offset in range(1, ACTIVATION_BARS + 1):
            index = int(position + offset)
            if closes[index] > upper[row]:
                side = 1
            elif closes[index] < lower[row]:
                side = -1
            else:
                continue
            confirmed[row] = True
            direction[row] = side
            step[row] = offset
            bar_range = highs[index] - lows[index]
            if bar_range > 0.0:
                body_fraction[row] = abs(closes[index] - opens[index]) / bar_range
                outer_fraction[row] = (
                    (closes[index] - lows[index]) / bar_range
                    if side == 1
                    else (highs[index] - closes[index]) / bar_range
                )
                strong[row] = (
                    body_fraction[row] >= 0.5 and outer_fraction[row] >= 0.75
                )
            entry[row] = opens[index + 1]
            break
    return {
        "confirmed": confirmed,
        "direction": direction,
        "confirmation_step": step,
        "strong_close": strong,
        "body_fraction": body_fraction,
        "outer_fraction": outer_fraction,
        "entry_price": entry,
    }


def oco_execution(
    tape: pd.DataFrame,
    positions: np.ndarray,
    horizon: int,
    upper: np.ndarray,
    lower: np.ndarray,
    exit_price: np.ndarray,
) -> dict[str, np.ndarray]:
    opens = tape["open"].to_numpy(float)
    highs = tape["high"].to_numpy(float)
    lows = tape["low"].to_numpy(float)
    status = np.full(len(positions), "no_trigger", dtype=object)
    direction = np.zeros(len(positions), dtype=np.int8)
    step = np.zeros(len(positions), dtype=int)
    entry = np.full(len(positions), np.nan)
    for row, position in enumerate(positions):
        for offset in range(1, horizon + 1):
            index = int(position + offset)
            if opens[index] >= upper[row] and opens[index] <= lower[row]:
                status[row] = "ambiguous_same_bar"
                break
            if opens[index] >= upper[row]:
                status[row] = "filled"
                direction[row] = 1
                step[row] = offset
                entry[row] = max(upper[row], opens[index])
                break
            if opens[index] <= lower[row]:
                status[row] = "filled"
                direction[row] = -1
                step[row] = offset
                entry[row] = min(lower[row], opens[index])
                break
            up = highs[index] >= upper[row]
            down = lows[index] <= lower[row]
            if up and down:
                status[row] = "ambiguous_same_bar"
                break
            if up:
                status[row] = "filled"
                direction[row] = 1
                step[row] = offset
                entry[row] = upper[row]
                break
            if down:
                status[row] = "filled"
                direction[row] = -1
                step[row] = offset
                entry[row] = lower[row]
                break
    filled = status == "filled"
    gross = np.full(len(positions), np.nan)
    gross[filled] = direction[filled] * (
        exit_price[filled] / entry[filled] - 1.0
    )
    return {
        "status": status,
        "direction": direction,
        "entry_step": step,
        "entry_price": entry,
        "exit_price": exit_price,
        "gross_return": gross,
        "holding_bars": np.where(filled, horizon - step + 1, 0),
    }


def attach_features(oof: pd.DataFrame, tape: pd.DataFrame) -> pd.DataFrame:
    lookup = tape[
        [
            "symbol_norm",
            "timestamp",
            "session_date",
            "bar_ordinal",
            "tape_position",
            "open",
            "high",
            "low",
            "close",
        ]
    ].rename(
        columns={
            "timestamp": "start_timestamp",
            "session_date": "tape_session_date",
            "open": "anchor_open",
            "high": "anchor_high",
            "low": "anchor_low",
            "close": "anchor_close",
        }
    )
    frame = (
        oof.reset_index(names="oof_position")
        .merge(
            lookup,
            on=["symbol_norm", "start_timestamp"],
            how="left",
            validate="one_to_one",
        )
        .sort_values("oof_position", kind="stable")
        .reset_index(drop=True)
    )
    if frame["tape_position"].isna().any() or not frame["session_date"].eq(
        frame["tape_session_date"]
    ).all():
        raise AssertionError("OOF-to-provider join failure")
    positions = frame["tape_position"].to_numpy(int)
    ordinals = frame["bar_ordinal"].to_numpy(int)
    tape_symbols = tape["symbol_norm"].to_numpy(str)
    tape_sessions = tape["session_date"].to_numpy(str)
    tape_timestamps = tape["timestamp"].to_numpy()
    for horizon in HORIZONS:
        future = positions + horizon
        exact = (
            (tape_symbols[future] == frame["symbol_norm"].to_numpy(str))
            & (tape_sessions[future] == frame["session_date"].to_numpy(str))
            & (
                tape_timestamps[future] - frame["start_timestamp"].to_numpy()
                == np.timedelta64(5 * horizon, "m")
            )
        )
        if not exact.all():
            raise AssertionError(f"inexact setup horizon h{horizon}")
        frame[f"exit_close_{horizon}"] = tape["close"].to_numpy(float)[future]
    confirmation = close_confirmation(
        tape,
        positions,
        frame["anchor_high"].to_numpy(float),
        frame["anchor_low"].to_numpy(float),
    )
    for name, values in confirmation.items():
        frame[name] = values
    range_values = tape["range_pct"].to_numpy(float)
    closes = tape["close"].to_numpy(float)
    short_mean = np.full(len(frame), np.nan)
    long_mean = np.full(len(frame), np.nan)
    trend = np.full(len(frame), np.nan)
    for row, (position, ordinal) in enumerate(zip(positions, ordinals, strict=True)):
        if ordinal >= 5:
            short_mean[row] = range_values[position - 5 : position + 1].mean()
        if ordinal >= 23:
            long_mean[row] = range_values[position - 23 : position + 1].mean()
        if ordinal >= 6:
            trend[row] = math.log(closes[position] / closes[position - 6])
    frame["compression_ratio"] = short_mean / long_mean
    frame["compression_pass"] = (
        frame["bar_ordinal"].ge(23)
        & frame["compression_ratio"].le(0.75)
    )
    frame["trend_return_6"] = trend
    frame["trend_aligned"] = (
        frame["confirmed"].astype(bool)
        & np.isfinite(trend)
        & ((frame["direction"].to_numpy(int) * trend) > 0.0)
    )
    frame["clock_quartile"] = np.minimum(
        frame["bar_ordinal"].to_numpy(int) * 4 // SESSION_BARS, 3
    ).astype(np.int8)
    return frame


def greedy_positions(
    frame: pd.DataFrame, eligible: np.ndarray, horizon: int
) -> np.ndarray:
    accepted = np.zeros(len(frame), dtype=bool)
    candidates = frame.loc[
        eligible, ["symbol_norm", "session_date", "bar_ordinal"]
    ]
    for _, group in candidates.groupby(["symbol_norm", "session_date"], sort=False):
        blocked_until = -1
        for position, ordinal in zip(group.index, group["bar_ordinal"], strict=True):
            ordinal = int(ordinal)
            if ordinal >= blocked_until:
                accepted[int(position)] = True
                blocked_until = ordinal + horizon
    return np.flatnonzero(accepted)


def build_setup_ledger(frame: pd.DataFrame, tape: pd.DataFrame) -> pd.DataFrame:
    positions_all = frame["tape_position"].to_numpy(int)
    ledger_frames: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        history_prediction = frame[
            f"prediction__history__future_range_bps__h{horizon}"
        ].to_numpy(float)
        history_gate = history_prediction >= THRESHOLDS[horizon]
        oco = oco_execution(
            tape,
            positions_all,
            horizon,
            frame["anchor_high"].to_numpy(float),
            frame["anchor_low"].to_numpy(float),
            frame[f"exit_close_{horizon}"].to_numpy(float),
        )
        for definition in SETUPS:
            if definition["exante"] == "all":
                eligible = np.ones(len(frame), dtype=bool)
            elif definition["exante"] == "history_gate":
                eligible = history_gate
            elif definition["exante"] == "history_gate_compression":
                eligible = history_gate & frame["compression_pass"].to_numpy(bool)
            else:
                raise AssertionError("unknown ex-ante setup gate")
            selected_positions = greedy_positions(frame, eligible, horizon)
            selected = frame.loc[
                selected_positions,
                [
                    "anchor_id",
                    "symbol_norm",
                    "session_date",
                    "month_key",
                    "start_timestamp",
                    "state",
                    "history_token",
                    "bar_ordinal",
                    "clock_quartile",
                    "anchor_high",
                    "anchor_low",
                    "anchor_close",
                    "compression_ratio",
                    "compression_pass",
                    "trend_return_6",
                    "confirmed",
                    "confirmation_step",
                    "strong_close",
                    "body_fraction",
                    "outer_fraction",
                    "trend_aligned",
                ],
            ].copy()
            selected["setup"] = definition["setup"]
            selected["family"] = definition["family"]
            selected["horizon"] = horizon
            selected["movement_prediction_bps"] = history_prediction[selected_positions]
            selected["movement_threshold_bps"] = (
                THRESHOLDS[horizon]
                if definition["exante"] != "all"
                else math.nan
            )
            selected["signal_exit_ordinal"] = (
                selected["bar_ordinal"].to_numpy(int) + horizon
            )
            if definition["condition"] == "oco":
                for name, values in oco.items():
                    selected[name] = values[selected_positions]
                selected["condition_pass"] = selected["status"].eq("filled")
                selected["confirmation_step"] = selected["entry_step"]
            else:
                confirmed = selected["confirmed"].to_numpy(bool)
                if definition["condition"] == "none":
                    condition_pass = confirmed
                elif definition["condition"] == "strong_close":
                    condition_pass = confirmed & selected["strong_close"].to_numpy(bool)
                elif definition["condition"] == "trend_aligned":
                    condition_pass = confirmed & selected["trend_aligned"].to_numpy(bool)
                else:
                    raise AssertionError("unknown confirmation condition")
                status = np.full(len(selected), "no_confirmation", dtype=object)
                status[confirmed & ~condition_pass] = "condition_failed"
                status[condition_pass] = "filled"
                selected["status"] = status
                selected["condition_pass"] = condition_pass
                selected["direction"] = frame.loc[
                    selected_positions, "direction"
                ].to_numpy(int)
                selected["entry_step"] = (
                    selected["confirmation_step"].to_numpy(int) + 1
                )
                selected["entry_price"] = frame.loc[
                    selected_positions, "entry_price"
                ].to_numpy(float)
                selected["exit_price"] = frame.loc[
                    selected_positions, f"exit_close_{horizon}"
                ].to_numpy(float)
                gross = np.full(len(selected), np.nan)
                gross[condition_pass] = (
                    selected.loc[condition_pass, "direction"].to_numpy(float)
                    * (
                        selected.loc[condition_pass, "exit_price"].to_numpy(float)
                        / selected.loc[condition_pass, "entry_price"].to_numpy(float)
                        - 1.0
                    )
                )
                selected["gross_return"] = gross
                selected["holding_bars"] = np.where(
                    condition_pass,
                    horizon - selected["confirmation_step"].to_numpy(int),
                    0,
                )
            selected["gross_return_bps"] = 10000.0 * selected[
                "gross_return"
            ].to_numpy(float)
            ledger_frames.append(selected)
    ledger = pd.concat(ledger_frames, ignore_index=True)
    if ledger.duplicated(["setup", "horizon", "anchor_id"]).any():
        raise AssertionError("duplicate setup signal")
    filled = ledger["status"].eq("filled")
    if not np.isfinite(
        ledger.loc[filled, ["entry_price", "exit_price", "gross_return"]].to_numpy(float)
    ).all():
        raise AssertionError("non-finite setup trade")
    return ledger


def portfolio_stats(daily: np.ndarray) -> dict[str, float]:
    values = np.asarray(daily, float)
    equity = np.cumprod(1.0 + values)
    cumulative = float(equity[-1] - 1.0)
    annualized = float((1.0 + cumulative) ** (252.0 / len(values)) - 1.0)
    volatility = float(np.std(values, ddof=1) * math.sqrt(252.0))
    sharpe = (
        float(values.mean() / np.std(values, ddof=1) * math.sqrt(252.0))
        if np.std(values, ddof=1) > 0.0
        else math.nan
    )
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


def daily_for(
    signals: pd.DataFrame,
    sessions: list[str],
    cost: int,
    deleted_symbol: str | None = None,
) -> tuple[pd.Series, np.ndarray]:
    trades = signals.loc[signals["status"].eq("filled")].copy()
    divisor = UNIVERSE_SIZE
    if deleted_symbol is not None:
        trades = trades.loc[trades["symbol_norm"].ne(deleted_symbol)].copy()
        divisor -= 1
    net = trades["gross_return"].to_numpy(float) - 2.0 * cost / 10000.0
    if (net <= -1.0).any():
        raise AssertionError("setup trade exceeded collateral")
    trades["log_growth"] = np.log1p(net)
    sleeve = np.expm1(
        trades.groupby(["session_date", "symbol_norm"], sort=False)[
            "log_growth"
        ].sum()
    )
    daily = (sleeve.groupby("session_date").sum() / divisor).reindex(
        sessions, fill_value=0.0
    )
    return daily, 10000.0 * net


def evaluate(
    ledger: pd.DataFrame, sessions: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    month_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    clock_rows: list[dict[str, Any]] = []
    symbols = sorted(ledger["symbol_norm"].unique())
    for (setup, family, horizon), signals in ledger.groupby(
        ["setup", "family", "horizon"], sort=False
    ):
        for cost in COSTS:
            daily, net_bps = daily_for(signals, sessions, cost)
            filled = signals["status"].eq("filled")
            positive = net_bps[net_bps > 0.0]
            negative = net_bps[net_bps < 0.0]
            profit_factor = (
                float(positive.sum() / -negative.sum())
                if len(negative)
                else (math.inf if len(positive) else math.nan)
            )
            metric_rows.append(
                {
                    "setup": setup,
                    "family": family,
                    "horizon": int(horizon),
                    "cost_bps_per_side": cost,
                    "armed_signals": len(signals),
                    "filled_trades": int(filled.sum()),
                    "no_confirmation_or_trigger": int(
                        signals["status"].isin(["no_confirmation", "no_trigger"]).sum()
                    ),
                    "condition_failures": int(
                        signals["status"].eq("condition_failed").sum()
                    ),
                    "ambiguous_signals": int(
                        signals["status"].eq("ambiguous_same_bar").sum()
                    ),
                    "fill_rate": float(filled.mean()),
                    "stocks_with_filled_trade": int(
                        signals.loc[filled, "symbol_norm"].nunique()
                    ),
                    "mean_net_trade_bps": float(net_bps.mean()) if len(net_bps) else math.nan,
                    "median_net_trade_bps": float(np.median(net_bps)) if len(net_bps) else math.nan,
                    "win_rate": float((net_bps > 0.0).mean()) if len(net_bps) else math.nan,
                    "profit_factor": profit_factor,
                    "exposure_fraction": float(
                        signals.loc[filled, "holding_bars"].sum()
                        / (SESSION_BARS * UNIVERSE_SIZE * len(sessions))
                    ),
                    **portfolio_stats(daily.to_numpy(float)),
                }
            )
            daily_frame = pd.DataFrame(
                {
                    "setup": setup,
                    "family": family,
                    "horizon": int(horizon),
                    "cost_bps_per_side": cost,
                    "session_date": daily.index.astype(str),
                    "daily_return": daily.to_numpy(float),
                }
            )
            daily_frames.append(daily_frame)
            daily_frame["month"] = daily_frame["session_date"].str.slice(0, 7)
            for month, selected in daily_frame.groupby("month", sort=True):
                trades_month = signals.loc[
                    signals["session_date"].str.startswith(month)
                    & signals["status"].eq("filled")
                ]
                month_rows.append(
                    {
                        "setup": setup,
                        "horizon": int(horizon),
                        "cost_bps_per_side": cost,
                        "month": month,
                        "session_dates": len(selected),
                        "filled_trades": len(trades_month),
                        **portfolio_stats(selected["daily_return"].to_numpy(float)),
                    }
                )
            for deleted_symbol in symbols:
                deleted_daily, _ = daily_for(
                    signals, sessions, cost, deleted_symbol
                )
                deletion_rows.append(
                    {
                        "setup": setup,
                        "horizon": int(horizon),
                        "cost_bps_per_side": cost,
                        "deleted_symbol": deleted_symbol,
                        **portfolio_stats(deleted_daily.to_numpy(float)),
                    }
                )
        filled_primary = signals.loc[signals["status"].eq("filled")].copy()
        filled_primary["net_bps"] = (
            10000.0 * filled_primary["gross_return"].to_numpy(float)
            - 2.0 * PRIMARY_COST
        )
        for quartile in range(4):
            selected = filled_primary.loc[
                filled_primary["clock_quartile"].eq(quartile)
            ]
            clock_rows.append(
                {
                    "setup": setup,
                    "horizon": int(horizon),
                    "clock_quartile": quartile,
                    "filled_trades": len(selected),
                    "mean_net_trade_bps": float(selected["net_bps"].mean())
                    if len(selected)
                    else math.nan,
                    "win_rate": float(selected["net_bps"].gt(0.0).mean())
                    if len(selected)
                    else math.nan,
                }
            )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(daily_frames, ignore_index=True),
        pd.DataFrame(month_rows),
        pd.DataFrame(deletion_rows),
        pd.DataFrame(clock_rows),
    )


def moving_block(
    values: np.ndarray, seed_offset: int, draws: int = 5000
) -> tuple[float, float, float]:
    data = np.asarray(values, float)
    rng = np.random.default_rng(SEED + seed_offset)
    starts = np.arange(len(data) - 5 + 1)
    block_count = math.ceil(len(data) / 5)
    selected = rng.choice(starts, size=(draws, block_count), replace=True)
    positions = (
        selected[:, :, None] + np.arange(5)[None, None, :]
    ).reshape(draws, -1)[:, : len(data)]
    sampled = data[positions].mean(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975], method="linear")
    return float(data.mean()), float(lower), float(upper)


def bootstrap_hypotheses(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hypothesis_index, (hypothesis, candidate, baseline) in enumerate(HYPOTHESES):
        for horizon_index, horizon in enumerate(HORIZONS):
            common = daily.loc[
                daily["horizon"].eq(horizon)
                & daily["cost_bps_per_side"].eq(PRIMARY_COST)
            ]
            candidate_daily = common.loc[
                common["setup"].eq(candidate)
            ].sort_values("session_date")
            baseline_daily = common.loc[
                common["setup"].eq(baseline)
            ].sort_values("session_date")
            if not candidate_daily["session_date"].reset_index(drop=True).equals(
                baseline_daily["session_date"].reset_index(drop=True)
            ):
                raise AssertionError("setup bootstrap alignment failure")
            observed, lower, upper = moving_block(
                candidate_daily["daily_return"].to_numpy(float),
                hypothesis_index * 100 + horizon_index * 2,
            )
            rows.append(
                {
                    "hypothesis": hypothesis,
                    "candidate": candidate,
                    "baseline": "zero",
                    "comparison": "candidate_absolute",
                    "horizon": horizon,
                    "session_dates": len(candidate_daily),
                    "mean_daily_return": observed,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
            difference = candidate_daily["daily_return"].to_numpy(float) - baseline_daily[
                "daily_return"
            ].to_numpy(float)
            observed, lower, upper = moving_block(
                difference,
                hypothesis_index * 100 + horizon_index * 2 + 1,
            )
            rows.append(
                {
                    "hypothesis": hypothesis,
                    "candidate": candidate,
                    "baseline": baseline,
                    "comparison": "candidate_minus_baseline",
                    "horizon": horizon,
                    "session_dates": len(candidate_daily),
                    "mean_daily_return": observed,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def decisions(
    ledger: pd.DataFrame,
    metrics: pd.DataFrame,
    months: pd.DataFrame,
    deletions: pd.DataFrame,
    bootstraps: pd.DataFrame,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for hypothesis, candidate, baseline in HYPOTHESES:
        primary = metrics.loc[
            metrics["setup"].eq(candidate)
            & metrics["cost_bps_per_side"].eq(PRIMARY_COST)
        ]
        month = months.loc[
            months["setup"].eq(candidate)
            & months["cost_bps_per_side"].eq(PRIMARY_COST)
        ]
        deletion = deletions.loc[
            deletions["setup"].eq(candidate)
            & deletions["cost_bps_per_side"].eq(PRIMARY_COST)
        ]
        bootstrap = bootstraps.loc[bootstraps["hypothesis"].eq(hypothesis)]
        absolute = bootstrap.loc[bootstrap["comparison"].eq("candidate_absolute")]
        paired = bootstrap.loc[
            bootstrap["comparison"].eq("candidate_minus_baseline")
        ]
        checks = {
            "minimum_confirmed_trades_each_horizon": bool(
                primary["filled_trades"].ge(500).all() and len(primary) == 3
            ),
            "minimum_stocks_each_horizon": bool(
                primary["stocks_with_filled_trade"].ge(15).all()
            ),
            "minimum_monthly_confirmed_trades": bool(
                month["filled_trades"].ge(50).all() and len(month) == 18
            ),
            "positive_mean_net_trade_each_horizon": bool(
                primary["mean_net_trade_bps"].gt(0.0).all()
            ),
            "positive_cumulative_return_each_horizon": bool(
                primary["cumulative_return"].gt(0.0).all()
            ),
            "absolute_bootstrap_lower_above_zero_each_horizon": bool(
                absolute["ci_lower"].gt(0.0).all() and len(absolute) == 3
            ),
            "paired_bootstrap_lower_above_zero_each_horizon": bool(
                paired["ci_lower"].gt(0.0).all() and len(paired) == 3
            ),
            "positive_each_month_and_horizon": bool(
                month["cumulative_return"].gt(0.0).all()
            ),
            "positive_every_stock_deletion_and_horizon": bool(
                deletion["cumulative_return"].gt(0.0).all()
                and len(deletion) == UNIVERSE_SIZE * 3
            ),
        }
        checks["retained"] = bool(all(checks.values()))
        output[hypothesis] = {
            "candidate": candidate,
            "baseline": baseline,
            "checks": checks,
            "decision": (
                "retain_internal_research_hypothesis"
                if checks["retained"]
                else "reject_or_descriptive_only"
            ),
        }
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "strategy_promotion": False,
        "economic_edge_claim": False,
        "hypotheses": output,
        "retained_hypotheses": [
            key for key, value in output.items() if value["checks"]["retained"]
        ],
    }


def main() -> None:
    oof = load_oof()
    symbols = sorted(oof["symbol_norm"].unique())
    if len(symbols) != UNIVERSE_SIZE:
        raise AssertionError("setup universe drift")
    contract, pre_score = load_contract_and_verify(symbols)
    validate_threshold_source()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    tape = load_tape(symbols)
    frame = attach_features(oof, tape)
    sessions = sorted(
        date
        for date in tape["session_date"].unique()
        if "2024-07" <= date[:7] <= "2024-12"
    )
    ledger = build_setup_ledger(frame, tape)
    metrics, daily, months, deletions, clock = evaluate(ledger, sessions)
    bootstraps = bootstrap_hypotheses(daily)
    decision = decisions(ledger, metrics, months, deletions, bootstraps)

    feature_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "bar_ordinal",
        "tape_position",
        "anchor_open",
        "anchor_high",
        "anchor_low",
        "anchor_close",
        "confirmed",
        "direction",
        "confirmation_step",
        "strong_close",
        "body_fraction",
        "outer_fraction",
        "entry_price",
        "compression_ratio",
        "compression_pass",
        "trend_return_6",
        "trend_aligned",
        "clock_quartile",
        *(f"exit_close_{horizon}" for horizon in HORIZONS),
        *(f"prediction__history__future_range_bps__h{horizon}" for horizon in HORIZONS),
    ]
    frame.loc[:, feature_columns].to_parquet(
        OUT / "setup_feature_ledger_2024.parquet", index=False
    )
    ledger.to_parquet(OUT / "accepted_setup_signals_2024.parquet", index=False)
    metrics.to_csv(OUT / "setup_metrics.csv", index=False)
    daily.to_parquet(OUT / "daily_setup_returns.parquet", index=False)
    months.to_csv(OUT / "monthly_setup_metrics.csv", index=False)
    deletions.to_csv(OUT / "setup_stock_deletions.csv", index=False)
    clock.to_csv(OUT / "setup_clock_slices.csv", index=False)
    bootstraps.to_csv(OUT / "setup_bootstraps.csv", index=False)
    write_json(OUT / "decision.json", decision)
    write_json(
        OUT / "source_hashes.json",
        {**pre_score, "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH)},
    )
    summary = {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "oof_anchor_rows": len(oof),
        "symbols": len(symbols),
        "session_dates": len(sessions),
        "accepted_setup_signal_rows": len(ledger),
        "primary_metrics": metrics.loc[
            metrics["cost_bps_per_side"].eq(PRIMARY_COST)
        ].to_dict(orient="records"),
        "bootstraps": bootstraps.to_dict(orient="records"),
        "decision": decision,
    }
    write_json(OUT / "summary.json", summary)
    files = sorted(path for path in OUT.iterdir() if path.is_file())
    write_json(
        OUT / "artifact_manifest.json",
        {
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "provider_volume_label": "historical_volume_not_used",
            "files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in files
            ],
        },
    )
    print(json.dumps(safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
