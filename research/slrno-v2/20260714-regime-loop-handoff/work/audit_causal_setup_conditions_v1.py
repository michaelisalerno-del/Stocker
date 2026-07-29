#!/usr/bin/env python3
"""Independent audit for causal setup-conditions V1."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WORK = Path(__file__).resolve().parent
CONTRACT = WORK / "contracts/20260712-causal-setup-conditions-v1.json"
PRE_SCORE = WORK / "contracts/20260712-causal-setup-conditions-v1-pre-score.json"
RUNNER = WORK / "run_causal_setup_conditions_v1.py"
OOF_ROOT = Path("/private/tmp/stocker_regime_utility_ablation_v1_20260711")
OOF = OOF_ROOT / "oof_predictions_2024.parquet"
OOF_AUDIT = OOF_ROOT / "independent_audit.json"
THRESHOLDS_FILE = Path("/private/tmp/stocker_frozen_regime_loop_pnl_sanity_v1_20260712/prediction_thresholds_2024.csv")
RAW_ROOT = Path("/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock")
ROOT = Path("/private/tmp/stocker_causal_setup_conditions_v1_20260712")

SEED = 20260712
HORIZONS = (6, 12, 24)
COSTS = (0, 1, 2, 5, 10)
PRIMARY_COST = 5
UNIVERSE = 22
SESSION_BARS = 78
THRESHOLDS = {6: 210.3204137212535, 12: 283.3242044166901, 24: 372.9191260003554}
SETUPS = (
    ("oco_anchor_breakout_all", "oco_baseline", "all", "oco"),
    ("close_confirmed_all", "close_confirmation", "all", "none"),
    ("history_gate_close_confirmed", "movement_gate", "history", "none"),
    ("history_gate_strong_close", "strong_close", "history", "strong"),
    ("history_gate_compression_close", "compression", "compression", "none"),
    ("history_gate_trend_aligned_close", "trend_alignment", "history", "trend"),
)
HYPOTHESES = (
    ("H1_close_confirmation", "close_confirmed_all", "oco_anchor_breakout_all"),
    ("H2_movement_gate", "history_gate_close_confirmed", "close_confirmed_all"),
    ("H3_strong_close", "history_gate_strong_close", "history_gate_close_confirmed"),
    ("H4_compression", "history_gate_compression_close", "history_gate_close_confirmed"),
    ("H5_trend_alignment", "history_gate_trend_aligned_close", "history_gate_close_confirmed"),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def provider_path(symbol: str) -> Path:
    return RAW_ROOT / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def sources(symbols: list[str]) -> dict[str, Path]:
    values = {
        "contract": CONTRACT,
        "runner": RUNNER,
        "oof_predictions": OOF,
        "oof_independent_audit": OOF_AUDIT,
        "frozen_thresholds": THRESHOLDS_FILE,
    }
    values.update({f"provider_2024_{symbol}": provider_path(symbol) for symbol in symbols})
    return values


def load_tape(symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = pd.read_parquet(provider_path(symbol), columns=["timestamp", "open", "high", "low", "close"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.loc[
            frame["timestamp"].ge(pd.Timestamp("2024-01-01", tz="UTC"))
            & frame["timestamp"].lt(pd.Timestamp("2025-01-01", tz="UTC"))
        ].dropna()
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        minute = local.dt.hour * 60 + local.dt.minute
        frame = frame.loc[minute.ge(570) & minute.lt(960)].copy().sort_values("timestamp", kind="stable")
        local = frame["timestamp"].dt.tz_convert("America/New_York")
        frame["session_date"] = local.dt.strftime("%Y-%m-%d")
        frame["symbol_norm"] = symbol
        frame["bar_ordinal"] = frame.groupby("session_date", sort=False).cumcount()
        frame["range_pct"] = (frame["high"] - frame["low"]) / frame["open"]
        frames.append(frame)
    tape = pd.concat(frames, ignore_index=True).sort_values(
        ["symbol_norm", "session_date", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    tape["tape_position"] = np.arange(len(tape))
    return tape


def independent_feature_reconstruction(
    raw_oof: pd.DataFrame, tape: pd.DataFrame
) -> pd.DataFrame:
    lookup = tape.rename(columns={"timestamp": "start_timestamp"})[
        ["symbol_norm", "start_timestamp", "session_date", "bar_ordinal", "tape_position", "open", "high", "low", "close"]
    ]
    frame = raw_oof.merge(
        lookup,
        on=["symbol_norm", "start_timestamp"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_tape"),
    ).sort_values(["symbol_norm", "session_date", "start_timestamp"], kind="stable").reset_index(drop=True)
    positions = frame["tape_position"].to_numpy(int)
    opens = tape["open"].to_numpy(float)
    highs = tape["high"].to_numpy(float)
    lows = tape["low"].to_numpy(float)
    closes = tape["close"].to_numpy(float)
    ranges = tape["range_pct"].to_numpy(float)
    count = len(frame)
    confirmed = np.zeros(count, bool)
    direction = np.zeros(count, np.int8)
    step = np.zeros(count, int)
    strong = np.zeros(count, bool)
    body = np.full(count, np.nan)
    outer = np.full(count, np.nan)
    entry = np.full(count, np.nan)
    compression = np.full(count, np.nan)
    trend = np.full(count, np.nan)
    for row, (position, ordinal) in enumerate(zip(positions, frame["bar_ordinal"].to_numpy(int), strict=True)):
        upper = highs[position]
        lower = lows[position]
        for offset in range(1, 4):
            index = position + offset
            if closes[index] > upper:
                side = 1
            elif closes[index] < lower:
                side = -1
            else:
                continue
            confirmed[row] = True
            direction[row] = side
            step[row] = offset
            spread = highs[index] - lows[index]
            if spread > 0:
                body[row] = abs(closes[index] - opens[index]) / spread
                outer[row] = (closes[index] - lows[index]) / spread if side == 1 else (highs[index] - closes[index]) / spread
                strong[row] = body[row] >= 0.5 and outer[row] >= 0.75
            entry[row] = opens[index + 1]
            break
        if ordinal >= 23:
            compression[row] = ranges[position - 5 : position + 1].mean() / ranges[position - 23 : position + 1].mean()
        if ordinal >= 6:
            trend[row] = math.log(closes[position] / closes[position - 6])
    frame["confirmed"] = confirmed
    frame["direction"] = direction
    frame["confirmation_step"] = step
    frame["strong_close"] = strong
    frame["body_fraction"] = body
    frame["outer_fraction"] = outer
    frame["entry_price"] = entry
    frame["compression_ratio"] = compression
    frame["compression_pass"] = frame["bar_ordinal"].ge(23) & frame["compression_ratio"].le(0.75)
    frame["trend_return_6"] = trend
    frame["trend_aligned"] = confirmed & np.isfinite(trend) & ((direction * trend) > 0)
    frame["clock_quartile"] = np.minimum(frame["bar_ordinal"].to_numpy(int) * 4 // 78, 3)
    for horizon in HORIZONS:
        frame[f"exit_close_{horizon}"] = closes[positions + horizon]
    return frame


def greedy(frame: pd.DataFrame, eligible: np.ndarray, horizon: int) -> np.ndarray:
    accepted: list[int] = []
    for _, group in frame.loc[eligible].groupby(["symbol_norm", "session_date"], sort=False):
        blocked = -1
        for position, ordinal in zip(group.index, group["bar_ordinal"], strict=True):
            if int(ordinal) >= blocked:
                accepted.append(int(position))
                blocked = int(ordinal) + horizon
    return np.asarray(accepted, int)


def oco(frame: pd.DataFrame, tape: pd.DataFrame, selected: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    opens = tape["open"].to_numpy(float)
    highs = tape["high"].to_numpy(float)
    lows = tape["low"].to_numpy(float)
    positions = frame.loc[selected, "tape_position"].to_numpy(int)
    upper = frame.loc[selected, "high"].to_numpy(float)
    lower = frame.loc[selected, "low"].to_numpy(float)
    exits = frame.loc[selected, f"exit_close_{horizon}"].to_numpy(float)
    status = np.full(len(selected), "no_trigger", object)
    direction = np.zeros(len(selected), int)
    step = np.zeros(len(selected), int)
    entry = np.full(len(selected), np.nan)
    for row, position in enumerate(positions):
        for offset in range(1, horizon + 1):
            index = position + offset
            if opens[index] >= upper[row] and opens[index] <= lower[row]:
                status[row] = "ambiguous_same_bar"
                break
            if opens[index] >= upper[row]:
                status[row], direction[row], step[row], entry[row] = "filled", 1, offset, max(upper[row], opens[index])
                break
            if opens[index] <= lower[row]:
                status[row], direction[row], step[row], entry[row] = "filled", -1, offset, min(lower[row], opens[index])
                break
            up, down = highs[index] >= upper[row], lows[index] <= lower[row]
            if up and down:
                status[row] = "ambiguous_same_bar"
                break
            if up or down:
                status[row], direction[row], step[row], entry[row] = (
                    "filled", 1 if up else -1, offset, upper[row] if up else lower[row]
                )
                break
    filled = status == "filled"
    gross = np.full(len(selected), np.nan)
    gross[filled] = direction[filled] * (exits[filled] / entry[filled] - 1)
    return dict(status=status, direction=direction, entry_step=step, entry_price=entry, exit_price=exits, gross_return=gross, holding_bars=np.where(filled, horizon - step + 1, 0))


def verify_signal_ledger(features: pd.DataFrame, tape: pd.DataFrame, observed: pd.DataFrame) -> int:
    checked = 0
    for horizon in HORIZONS:
        prediction = features[f"prediction__history__future_range_bps__h{horizon}"].to_numpy(float)
        history = prediction >= THRESHOLDS[horizon]
        for setup, _, exante, condition in SETUPS:
            eligible = np.ones(len(features), bool) if exante == "all" else history.copy()
            if exante == "compression":
                eligible &= features["compression_pass"].to_numpy(bool)
            positions = greedy(features, eligible, horizon)
            selected = observed.loc[observed["setup"].eq(setup) & observed["horizon"].eq(horizon)].sort_values(
                ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
            )
            if not np.array_equal(features.loc[positions, "anchor_id"].to_numpy(int), selected["anchor_id"].to_numpy(int)):
                raise AssertionError(f"accepted setup mismatch {setup} h{horizon}")
            if condition == "oco":
                expected = oco(features, tape, positions, horizon)
                for name, values in expected.items():
                    if name == "status":
                        if not np.array_equal(values.astype(str), selected[name].astype(str)):
                            raise AssertionError("OCO status mismatch")
                    elif not np.allclose(values, selected[name], rtol=1e-9, atol=1e-9, equal_nan=True):
                        raise AssertionError(f"OCO {name} mismatch")
            else:
                confirmed = features.loc[positions, "confirmed"].to_numpy(bool)
                passes = confirmed.copy()
                if condition == "strong":
                    passes &= features.loc[positions, "strong_close"].to_numpy(bool)
                if condition == "trend":
                    passes &= features.loc[positions, "trend_aligned"].to_numpy(bool)
                status = np.full(len(positions), "no_confirmation", object)
                status[confirmed & ~passes] = "condition_failed"
                status[passes] = "filled"
                if not np.array_equal(status.astype(str), selected["status"].astype(str)):
                    raise AssertionError("close-confirmation status mismatch")
                exits = features.loc[positions, f"exit_close_{horizon}"].to_numpy(float)
                entries = features.loc[positions, "entry_price"].to_numpy(float)
                directions = features.loc[positions, "direction"].to_numpy(int)
                gross = np.full(len(positions), np.nan)
                gross[passes] = directions[passes] * (exits[passes] / entries[passes] - 1)
                if not np.allclose(gross, selected["gross_return"], equal_nan=True, atol=1e-10):
                    raise AssertionError("close-confirmation return mismatch")
            checked += len(selected)
    return checked


def stats(daily: np.ndarray) -> dict[str, float]:
    values = np.asarray(daily, float)
    equity = np.cumprod(1 + values)
    cumulative = equity[-1] - 1
    path = np.r_[1.0, equity]
    return dict(
        cumulative_return=cumulative,
        annualized_return=(1 + cumulative) ** (252 / len(values)) - 1,
        annualized_volatility=np.std(values, ddof=1) * np.sqrt(252),
        descriptive_sharpe_zero_rate=(values.mean() / np.std(values, ddof=1) * np.sqrt(252)) if np.std(values, ddof=1) > 0 else np.nan,
        maximum_drawdown=(path / np.maximum.accumulate(path) - 1).min(initial=0),
        mean_daily_return=values.mean(),
    )


def daily_for(signals: pd.DataFrame, sessions: list[str], cost: int, deleted: str | None = None) -> tuple[pd.Series, np.ndarray]:
    trades = signals.loc[signals["status"].eq("filled")].copy()
    divisor = UNIVERSE
    if deleted is not None:
        trades = trades.loc[trades["symbol_norm"].ne(deleted)].copy()
        divisor -= 1
    net = trades["gross_return"].to_numpy(float) - 2 * cost / 10000
    trades["log_growth"] = np.log1p(net)
    sleeve = np.expm1(trades.groupby(["session_date", "symbol_norm"], sort=False)["log_growth"].sum())
    daily = (sleeve.groupby("session_date").sum() / divisor).reindex(sessions, fill_value=0.0)
    return daily, net * 10000


def recompute_tables(ledger: pd.DataFrame, sessions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, daily_frames, months, deletions, clocks = [], [], [], [], []
    symbols = sorted(ledger["symbol_norm"].unique())
    for (setup, family, horizon), signals in ledger.groupby(["setup", "family", "horizon"], sort=False):
        for cost in COSTS:
            daily, net = daily_for(signals, sessions, cost)
            filled = signals["status"].eq("filled")
            pos, neg = net[net > 0], net[net < 0]
            pf = pos.sum() / -neg.sum() if len(neg) else (np.inf if len(pos) else np.nan)
            metrics.append(dict(setup=setup, family=family, horizon=int(horizon), cost_bps_per_side=cost, armed_signals=len(signals), filled_trades=int(filled.sum()), no_confirmation_or_trigger=int(signals["status"].isin(["no_confirmation", "no_trigger"]).sum()), condition_failures=int(signals["status"].eq("condition_failed").sum()), ambiguous_signals=int(signals["status"].eq("ambiguous_same_bar").sum()), fill_rate=filled.mean(), stocks_with_filled_trade=signals.loc[filled, "symbol_norm"].nunique(), mean_net_trade_bps=net.mean() if len(net) else np.nan, median_net_trade_bps=np.median(net) if len(net) else np.nan, win_rate=(net > 0).mean() if len(net) else np.nan, profit_factor=pf, exposure_fraction=signals.loc[filled, "holding_bars"].sum() / (78 * 22 * len(sessions)), **stats(daily.to_numpy(float))))
            df = pd.DataFrame(dict(setup=setup, family=family, horizon=int(horizon), cost_bps_per_side=cost, session_date=daily.index.astype(str), daily_return=daily.to_numpy(float)))
            daily_frames.append(df)
            df["month"] = df["session_date"].str[:7]
            for month, selected in df.groupby("month", sort=True):
                count = len(signals.loc[signals["session_date"].str.startswith(month) & signals["status"].eq("filled")])
                months.append(dict(setup=setup, horizon=int(horizon), cost_bps_per_side=cost, month=month, session_dates=len(selected), filled_trades=count, **stats(selected["daily_return"].to_numpy(float))))
            for symbol in symbols:
                dd, _ = daily_for(signals, sessions, cost, symbol)
                deletions.append(dict(setup=setup, horizon=int(horizon), cost_bps_per_side=cost, deleted_symbol=symbol, **stats(dd.to_numpy(float))))
        filled = signals.loc[signals["status"].eq("filled")].copy()
        filled["net"] = filled["gross_return"] * 10000 - 10
        for quartile in range(4):
            selected = filled.loc[filled["clock_quartile"].eq(quartile)]
            clocks.append(dict(setup=setup, horizon=int(horizon), clock_quartile=quartile, filled_trades=len(selected), mean_net_trade_bps=selected["net"].mean() if len(selected) else np.nan, win_rate=selected["net"].gt(0).mean() if len(selected) else np.nan))
    return pd.DataFrame(metrics), pd.concat(daily_frames, ignore_index=True), pd.DataFrame(months), pd.DataFrame(deletions), pd.DataFrame(clocks)


def block(values: np.ndarray, offset: int) -> tuple[float, float, float]:
    data = np.asarray(values, float)
    rng = np.random.default_rng(SEED + offset)
    starts = np.arange(len(data) - 4)
    count = math.ceil(len(data) / 5)
    chosen = rng.choice(starts, size=(5000, count), replace=True)
    positions = (chosen[:, :, None] + np.arange(5)[None, None, :]).reshape(5000, -1)[:, : len(data)]
    sampled = data[positions].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975], method="linear")
    return data.mean(), low, high


def recompute_bootstraps(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hi, (hypothesis, candidate, baseline) in enumerate(HYPOTHESES):
        for hj, horizon in enumerate(HORIZONS):
            common = daily.loc[daily["horizon"].eq(horizon) & daily["cost_bps_per_side"].eq(5)]
            c = common.loc[common["setup"].eq(candidate)].sort_values("session_date")
            b = common.loc[common["setup"].eq(baseline)].sort_values("session_date")
            mean, low, high = block(c["daily_return"], hi * 100 + hj * 2)
            rows.append(dict(hypothesis=hypothesis, candidate=candidate, baseline="zero", comparison="candidate_absolute", horizon=horizon, session_dates=len(c), mean_daily_return=mean, ci_lower=low, ci_upper=high))
            mean, low, high = block(c["daily_return"].to_numpy(float) - b["daily_return"].to_numpy(float), hi * 100 + hj * 2 + 1)
            rows.append(dict(hypothesis=hypothesis, candidate=candidate, baseline=baseline, comparison="candidate_minus_baseline", horizon=horizon, session_dates=len(c), mean_daily_return=mean, ci_lower=low, ci_upper=high))
    return pd.DataFrame(rows)


def frame_close(observed: pd.DataFrame, expected: pd.DataFrame, name: str) -> float:
    if list(observed.columns) != list(expected.columns) or len(observed) != len(expected):
        raise AssertionError(f"{name} frame mismatch")
    maximum = 0.0
    for column in observed:
        if pd.api.types.is_numeric_dtype(observed[column]):
            left, right = observed[column].to_numpy(float), expected[column].to_numpy(float)
            if not np.allclose(left, right, rtol=1e-9, atol=1e-10, equal_nan=True):
                raise AssertionError(f"{name}:{column} mismatch")
            delta = np.abs(left - right)[np.isfinite(left - right)]
            maximum = max(maximum, float(delta.max(initial=0)))
        elif not observed[column].astype(str).equals(expected[column].astype(str)):
            raise AssertionError(f"{name}:{column} mismatch")
    return maximum


def main() -> None:
    checks: list[dict[str, Any]] = []
    def record(name: str, detail: Any) -> None:
        checks.append({"name": name, "pass": True, "detail": detail})

    oof_columns = ["anchor_id", "symbol_norm", "session_date", "start_timestamp", *(f"prediction__history__future_range_bps__h{h}" for h in HORIZONS)]
    raw_oof = pd.read_parquet(OOF, columns=oof_columns)
    raw_oof["start_timestamp"] = pd.to_datetime(raw_oof["start_timestamp"], utc=True)
    symbols = sorted(raw_oof["symbol_norm"].unique())
    pre = json.loads(PRE_SCORE.read_text())
    actual_hashes = {name: digest(path) for name, path in sources(symbols).items()}
    if actual_hashes != pre["sha256"]:
        raise AssertionError("frozen source mismatch")
    record("frozen_sources", len(actual_hashes))
    source_artifact = json.loads((ROOT / "source_hashes.json").read_text())
    if source_artifact["sha256"] != actual_hashes or source_artifact["pre_score_manifest_sha256"] != digest(PRE_SCORE):
        raise AssertionError("artifact source binding mismatch")
    record("source_binding", digest(PRE_SCORE))
    contract = json.loads(CONTRACT.read_text())
    if not (contract["research_only"] and not contract["live_ordering_enabled"] and contract["order_placement"] == "disabled"):
        raise AssertionError("safety boundary mismatch")
    record("research_only_boundary", True)
    if not json.loads(OOF_AUDIT.read_text())["all_passed"]:
        raise AssertionError("parent audit failure")
    record("parent_oof_audit", True)

    tape = load_tape(symbols)
    features = independent_feature_reconstruction(raw_oof, tape)
    observed_features = pd.read_parquet(ROOT / "setup_feature_ledger_2024.parquet").sort_values(["symbol_norm", "session_date", "start_timestamp"], kind="stable").reset_index(drop=True)
    if not features["anchor_id"].equals(observed_features["anchor_id"]):
        raise AssertionError("feature alignment mismatch")
    maximum_feature = 0.0
    mapping = {"open": "anchor_open", "high": "anchor_high", "low": "anchor_low", "close": "anchor_close"}
    for source, observed in mapping.items():
        maximum_feature = max(maximum_feature, float(np.max(np.abs(features[source].to_numpy(float) - observed_features[observed].to_numpy(float)))))
    for column in ("confirmed", "direction", "confirmation_step", "strong_close", "body_fraction", "outer_fraction", "entry_price", "compression_ratio", "compression_pass", "trend_return_6", "trend_aligned", "clock_quartile", *(f"exit_close_{h}" for h in HORIZONS)):
        if pd.api.types.is_bool_dtype(observed_features[column]) or column in ("direction", "confirmation_step", "clock_quartile"):
            if not np.array_equal(features[column], observed_features[column]):
                raise AssertionError(f"feature {column} mismatch")
        elif not np.allclose(features[column], observed_features[column], equal_nan=True, rtol=1e-9, atol=1e-10):
            raise AssertionError(f"feature {column} mismatch")
    record("exact_causal_setup_features", maximum_feature)

    ledger = pd.read_parquet(ROOT / "accepted_setup_signals_2024.parquet")
    checked = verify_signal_ledger(features, tape, ledger)
    if checked != len(ledger):
        raise AssertionError("signal row count mismatch")
    record("exact_setup_gates_overlap_confirmations_and_returns", checked)

    sessions = sorted(date for date in tape["session_date"].unique() if "2024-07" <= date[:7] <= "2024-12")
    metrics, daily, months, deletions, clocks = recompute_tables(ledger, sessions)
    errors = {
        "metrics": frame_close(pd.read_csv(ROOT / "setup_metrics.csv"), metrics, "metrics"),
        "daily": frame_close(pd.read_parquet(ROOT / "daily_setup_returns.parquet"), daily, "daily"),
        "months": frame_close(pd.read_csv(ROOT / "monthly_setup_metrics.csv"), months, "months"),
        "deletions": frame_close(pd.read_csv(ROOT / "setup_stock_deletions.csv"), deletions, "deletions"),
        "clock": frame_close(pd.read_csv(ROOT / "setup_clock_slices.csv"), clocks, "clock"),
    }
    record("exact_costs_compounding_and_metric_slices", errors)
    if not daily.groupby(["setup", "horizon", "cost_bps_per_side"]).size().eq(128).all():
        raise AssertionError("daily session grid mismatch")
    record("full_zero_fill_session_grid", 128)
    bootstraps = recompute_bootstraps(daily)
    record("exact_bootstraps", frame_close(pd.read_csv(ROOT / "setup_bootstraps.csv"), bootstraps, "bootstraps"))

    decision = json.loads((ROOT / "decision.json").read_text())
    if decision["retained_hypotheses"] != [] or any(value["checks"]["retained"] for value in decision["hypotheses"].values()):
        raise AssertionError("unexpected retained setup")
    record("decision_no_retained_hypothesis", True)
    if not (decision["research_only"] and not decision["live_ordering_enabled"] and not decision["strategy_promotion"]):
        raise AssertionError("decision safety mismatch")
    record("decision_safety", True)

    result = {
        "audit": "causal_setup_conditions_v1_independent",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "all_passed": True,
        "passed": len(checks),
        "failed": 0,
        "checks": checks,
    }
    audit_path = ROOT / "independent_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    files = sorted(path for path in ROOT.iterdir() if path.is_file() and path.name != "artifact_manifest.json")
    (ROOT / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "research_only": True,
                "live_ordering_enabled": False,
                "order_placement": "disabled",
                "provider_volume_label": "historical_volume_not_used",
                "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in files],
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
