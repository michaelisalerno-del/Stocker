"""Research-only causal long/short/neutral detector experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-long-short-neutral-detector-v1.json"
AUDITOR_PATH = HERE / "audit_long_short_neutral_detector_v1.py"
TEST_PATH = HERE / "tests/test_long_short_neutral_detector_v1.py"
RUNNER_PATH = Path(__file__).resolve()

CONTRACT_ID = "20260714-long-short-neutral-detector-v1"
SEED = 20260714
CLASSES = ("long", "neutral", "short")
MODELS = ("M0_clock_prior", "M1_price_context", "M2_price_plus_activity")
PRIMARY_COST_PER_SIDE_BPS = 5.0
ROUND_TRIP_COST_BPS = 10.0
HORIZON_BARS = 24
DECISION_ORDINALS = (12, 36)
FORBIDDEN_PRE_OUTCOME_TOKENS = (
    "label",
    "target_class",
    "future",
    "upper_hit",
    "lower_hit",
    "payoff",
    "net_bps",
    "mfe",
    "mae",
)

PRICE_NUMERIC = (
    "barrier_bps",
    "current_range_scale",
    "current_body_scale",
    "current_close_location",
    "current_upper_wick_fraction",
    "current_lower_wick_fraction",
    "return_1_scale",
    "return_3_scale",
    "return_6_scale",
    "return_12_scale",
    "mean_abs_return_6_scale",
    "compression_3_to_12",
    "session_return_scale",
    "session_mean_distance_scale",
    "opening_range_position",
    "opening_range_width_scale",
)
ACTIVITY_NUMERIC = (
    "current_activity_ratio_12",
    "activity_trend_3_to_12",
)
CATEGORICAL = ("decision_clock",)
MODEL_FEATURES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "M0_clock_prior": ((), CATEGORICAL),
    "M1_price_context": (PRICE_NUMERIC, CATEGORICAL),
    "M2_price_plus_activity": ((*PRICE_NUMERIC, *ACTIVITY_NUMERIC), CATEGORICAL),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["contract_id"] != CONTRACT_ID:
        raise AssertionError("contract id drift")
    safety = contract["safety"]
    if not safety["research_only"] or safety["live_ordering_enabled"]:
        raise AssertionError("research safety drift")
    if safety["order_placement"] != "disabled":
        raise AssertionError("order placement must remain disabled")
    return contract


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def all_symbols(contract: dict[str, Any]) -> list[str]:
    return sorted(
        set(contract["data"]["symbols_2024_2025"])
        | set(contract["data"]["symbols_2026"])
    )


def source_paths(contract: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "contract": CONTRACT_PATH,
        "runner": RUNNER_PATH,
        "auditor": AUDITOR_PATH,
        "tests": TEST_PATH,
    }
    root = Path(contract["data"]["provider_root"])
    for symbol in all_symbols(contract):
        paths[f"provider_{symbol}"] = provider_path(root, symbol)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen sources: {missing}")
    return paths


def _period_mask(local: pd.Series, period: int) -> pd.Series:
    if period in (2024, 2025):
        return local.dt.year.eq(period)
    return local.dt.year.eq(2026) & local.dt.date.lt(pd.Timestamp("2026-06-30").date())


def load_provider_rows(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(contract["data"]["provider_root"])
    frames: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for period in (2024, 2025, 2026):
        symbols = (
            contract["data"]["symbols_2026"]
            if period == 2026
            else contract["data"]["symbols_2024_2025"]
        )
        for symbol in symbols:
            path = provider_path(root, symbol)
            frame = pd.read_parquet(
                path, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
            frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
            if frame["timestamp"].duplicated().any():
                raise AssertionError(f"duplicate timestamps: {symbol}")
            local = frame["timestamp"].dt.tz_convert("America/New_York")
            period_rows = frame.loc[_period_mask(local, period)].copy()
            local_period = period_rows["timestamp"].dt.tz_convert("America/New_York")
            minute = local_period.dt.hour * 60 + local_period.dt.minute
            regular = minute.ge(570) & minute.lt(960)
            on_grid = (
                (minute - 570).mod(5).eq(0)
                & local_period.dt.second.eq(0)
                & local_period.dt.microsecond.eq(0)
            )
            prices = period_rows[["open", "high", "low", "close"]].apply(
                pd.to_numeric, errors="coerce"
            )
            volume = pd.to_numeric(period_rows["volume"], errors="coerce")
            valid_prices = (
                prices.gt(0).all(axis=1)
                & prices["high"].ge(prices[["open", "close"]].max(axis=1))
                & prices["low"].le(prices[["open", "close"]].min(axis=1))
            )
            valid_volume = volume.ge(0) & np.isfinite(volume)
            accepted = regular & on_grid & valid_prices & valid_volume
            selected = period_rows.loc[accepted].copy()
            selected[["open", "high", "low", "close"]] = prices.loc[accepted]
            selected["volume"] = volume.loc[accepted]
            selected_local = selected["timestamp"].dt.tz_convert("America/New_York")
            selected_minute = selected_local.dt.hour * 60 + selected_local.dt.minute
            selected["period"] = period
            selected["symbol_norm"] = symbol
            selected["session_date"] = selected_local.dt.strftime("%Y-%m-%d")
            selected["month"] = selected["session_date"].str.slice(0, 7)
            selected["bar_ordinal"] = ((selected_minute - 570) // 5).astype(np.int16)
            selected = selected.sort_values("timestamp", kind="stable").reset_index(drop=True)
            duplicate_ordinals = int(
                selected.duplicated(["session_date", "bar_ordinal"], keep=False).sum()
            )
            if duplicate_ordinals:
                raise AssertionError(f"duplicate session ordinals: {period} {symbol}")
            previous_close = selected["close"].shift(1)
            denominator = previous_close.where(previous_close.gt(0), selected["open"])
            true_range = np.maximum.reduce(
                [
                    (selected["high"] - selected["low"]).to_numpy(float),
                    (selected["high"] - denominator).abs().to_numpy(float),
                    (selected["low"] - denominator).abs().to_numpy(float),
                ]
            )
            selected["true_range_bps"] = 10000.0 * true_range / denominator
            selected["range_bps"] = (
                10000.0 * (selected["high"] - selected["low"]) / selected["open"]
            )
            coverage.append(
                {
                    "period": period,
                    "symbol_norm": symbol,
                    "source_rows_in_period": len(period_rows),
                    "accepted_regular_rows": len(selected),
                    "sessions": selected["session_date"].nunique(),
                    "outside_regular_rows": int((~regular).sum()),
                    "off_grid_regular_rows": int((regular & ~on_grid).sum()),
                    "invalid_ohlc_rows": int((regular & on_grid & ~valid_prices).sum()),
                    "invalid_volume_rows": int((regular & on_grid & ~valid_volume).sum()),
                    "first_timestamp": selected["timestamp"].min(),
                    "last_timestamp": selected["timestamp"].max(),
                }
            )
            frames.append(selected)
    tape = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["period", "symbol_norm", "session_date", "bar_ordinal"], kind="stable")
        .reset_index(drop=True)
    )
    return tape, pd.DataFrame(coverage)


def load_symbol_rows(
    contract: dict[str, Any], symbol: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one symbol across all periods to bound peak research memory."""
    root = Path(contract["data"]["provider_root"])
    frame = pd.read_parquet(
        provider_path(root, symbol),
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise AssertionError(f"duplicate timestamps: {symbol}")
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    years = local.dt.year
    in_scope = years.isin([2024, 2025, 2026])
    in_scope &= ~(years.eq(2026) & local.dt.date.ge(pd.Timestamp("2026-06-30").date()))
    if symbol == "AAL":
        in_scope &= ~years.eq(2026)
    scoped = frame.loc[in_scope].copy()
    scoped_local = scoped["timestamp"].dt.tz_convert("America/New_York")
    scoped_minute = scoped_local.dt.hour * 60 + scoped_local.dt.minute
    regular = scoped_minute.ge(570) & scoped_minute.lt(960)
    on_grid = (
        (scoped_minute - 570).mod(5).eq(0)
        & scoped_local.dt.second.eq(0)
        & scoped_local.dt.microsecond.eq(0)
    )
    prices = scoped[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    volume = pd.to_numeric(scoped["volume"], errors="coerce")
    valid_prices = (
        prices.gt(0).all(axis=1)
        & prices["high"].ge(prices[["open", "close"]].max(axis=1))
        & prices["low"].le(prices[["open", "close"]].min(axis=1))
    )
    valid_volume = volume.ge(0) & np.isfinite(volume)
    accepted = regular & on_grid & valid_prices & valid_volume
    selected = scoped.loc[accepted].copy()
    selected[["open", "high", "low", "close"]] = prices.loc[accepted]
    selected["volume"] = volume.loc[accepted]
    selected_local = selected["timestamp"].dt.tz_convert("America/New_York")
    selected_minute = selected_local.dt.hour * 60 + selected_local.dt.minute
    selected["period"] = selected_local.dt.year.astype(np.int16)
    selected["symbol_norm"] = symbol
    selected["session_date"] = selected_local.dt.strftime("%Y-%m-%d")
    selected["month"] = selected["session_date"].str.slice(0, 7)
    selected["bar_ordinal"] = ((selected_minute - 570) // 5).astype(np.int16)
    selected = selected.sort_values(["period", "timestamp"], kind="stable").reset_index(drop=True)
    if selected.duplicated(["session_date", "bar_ordinal"]).any():
        raise AssertionError(f"duplicate session ordinals: {symbol}")
    previous_close = selected.groupby("period", sort=False)["close"].shift(1)
    denominator = previous_close.where(previous_close.gt(0), selected["open"])
    true_range = np.maximum.reduce(
        [
            (selected["high"] - selected["low"]).to_numpy(float),
            (selected["high"] - denominator).abs().to_numpy(float),
            (selected["low"] - denominator).abs().to_numpy(float),
        ]
    )
    selected["true_range_bps"] = 10000.0 * true_range / denominator
    selected["range_bps"] = 10000.0 * (selected["high"] - selected["low"]) / selected["open"]
    coverage: list[dict[str, Any]] = []
    for period in (2024, 2025, 2026):
        if period == 2026 and symbol == "AAL":
            continue
        period_scope = scoped_local.dt.year.eq(period)
        period_selected = selected.loc[selected["period"].eq(period)]
        coverage.append(
            {
                "period": period,
                "symbol_norm": symbol,
                "source_rows_in_period": int(period_scope.sum()),
                "accepted_regular_rows": len(period_selected),
                "sessions": period_selected["session_date"].nunique(),
                "outside_regular_rows": int((period_scope & ~regular).sum()),
                "off_grid_regular_rows": int((period_scope & regular & ~on_grid).sum()),
                "invalid_ohlc_rows": int((period_scope & regular & on_grid & ~valid_prices).sum()),
                "invalid_volume_rows": int((period_scope & regular & on_grid & ~valid_volume).sum()),
                "first_timestamp": period_selected["timestamp"].min(),
                "last_timestamp": period_selected["timestamp"].max(),
            }
        )
    return selected, pd.DataFrame(coverage)


def _fraction(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator


def _geometric_mean(values: pd.Series) -> float:
    clean = values.to_numpy(float)
    if len(clean) == 0 or not np.isfinite(clean).all() or (clean <= 0).any():
        return float("nan")
    return float(np.exp(np.log(clean).mean()))


def event_features(
    period: int,
    symbol: str,
    session_date: str,
    session: pd.DataFrame,
    decision_ordinal: int,
) -> dict[str, Any] | None:
    by_ordinal = session.set_index("bar_ordinal", drop=False)
    required = list(range(0, decision_ordinal + 1))
    if any(ordinal not in by_ordinal.index for ordinal in required):
        return None
    if any(ordinal not in by_ordinal.index for ordinal in range(0, 6)):
        return None
    prior = by_ordinal.loc[list(range(decision_ordinal - 12, decision_ordinal))]
    current = by_ordinal.loc[decision_ordinal]
    scale_bps = float(prior["true_range_bps"].median())
    if not math.isfinite(scale_bps) or scale_bps <= 0:
        return None
    barrier_bps = float(np.clip(4.0 * scale_bps, 40.0, 250.0))
    current_range = float(current["high"] - current["low"])
    current_open = float(current["open"])
    current_close = float(current["close"])
    if current_range <= 0 or current_open <= 0 or current_close <= 0:
        return None
    current_body_bps = 10000.0 * (current_close / current_open - 1.0)
    close_location = (current_close - float(current["low"])) / current_range
    upper_wick = (
        float(current["high"]) - max(current_open, current_close)
    ) / current_range
    lower_wick = (
        min(current_open, current_close) - float(current["low"])
    ) / current_range

    def scaled_return(lag: int) -> float:
        lag_close = float(by_ordinal.loc[decision_ordinal - lag, "close"])
        return _fraction(10000.0 * (current_close / lag_close - 1.0), scale_bps)

    recent = by_ordinal.loc[list(range(decision_ordinal - 6, decision_ordinal + 1)), "close"]
    one_bar_returns = 10000.0 * recent.pct_change().dropna()
    mean_abs_return = float(one_bar_returns.abs().mean()) / scale_bps
    recent_three_ranges = by_ordinal.loc[
        list(range(decision_ordinal - 2, decision_ordinal + 1)), "range_bps"
    ]
    compression = _fraction(float(recent_three_ranges.median()), float(prior["range_bps"].median()))
    session_so_far = by_ordinal.loc[list(range(0, decision_ordinal + 1))]
    session_open = float(by_ordinal.loc[0, "open"])
    typical_mean = float(
        ((session_so_far["high"] + session_so_far["low"] + session_so_far["close"]) / 3.0).mean()
    )
    opening = by_ordinal.loc[list(range(0, 6))]
    opening_high = float(opening["high"].max())
    opening_low = float(opening["low"].min())
    opening_width = opening_high - opening_low
    prior_volume_gmean = _geometric_mean(prior["volume"])
    recent_volume_gmean = _geometric_mean(
        by_ordinal.loc[list(range(decision_ordinal - 2, decision_ordinal + 1)), "volume"]
    )
    event_id = f"lsn|{period}|{symbol}|{session_date}|{decision_ordinal:02d}"
    return {
        "event_id": event_id,
        "period": period,
        "symbol_norm": symbol,
        "session_date": session_date,
        "month": session_date[:7],
        "decision_timestamp": pd.Timestamp(current["timestamp"]),
        "decision_ordinal": decision_ordinal,
        "decision_clock": f"clock_{decision_ordinal:02d}",
        "prior_scale_bps": scale_bps,
        "barrier_bps": barrier_bps,
        "current_range_scale": float(current["range_bps"]) / scale_bps,
        "current_body_scale": current_body_bps / scale_bps,
        "current_close_location": close_location,
        "current_upper_wick_fraction": upper_wick,
        "current_lower_wick_fraction": lower_wick,
        "return_1_scale": scaled_return(1),
        "return_3_scale": scaled_return(3),
        "return_6_scale": scaled_return(6),
        "return_12_scale": scaled_return(12),
        "mean_abs_return_6_scale": mean_abs_return,
        "compression_3_to_12": compression,
        "session_return_scale": _fraction(
            10000.0 * (current_close / session_open - 1.0), scale_bps
        ),
        "session_mean_distance_scale": _fraction(
            10000.0 * (current_close / typical_mean - 1.0), scale_bps
        ),
        "opening_range_position": _fraction(current_close - opening_low, opening_width),
        "opening_range_width_scale": _fraction(
            10000.0 * opening_width / session_open, scale_bps
        ),
        "current_activity_ratio_12": _fraction(float(current["volume"]), prior_volume_gmean),
        "activity_trend_3_to_12": _fraction(recent_volume_gmean, prior_volume_gmean),
    }


def build_population(tape: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    grouped = tape.groupby(["period", "symbol_norm", "session_date"], sort=False)
    for (period, symbol, session_date), session in grouped:
        emitted = 0
        for ordinal in DECISION_ORDINALS:
            row = event_features(int(period), str(symbol), str(session_date), session, ordinal)
            if row is not None:
                rows.append(row)
                emitted += 1
        counts.append(
            {
                "period": int(period),
                "symbol_norm": str(symbol),
                "session_date": str(session_date),
                "eligible_anchors": emitted,
            }
        )
    events = pd.DataFrame(rows).sort_values(
        ["session_date", "symbol_norm", "decision_ordinal"], kind="stable"
    ).reset_index(drop=True)
    if events.empty or events["event_id"].duplicated().any():
        raise AssertionError("invalid event population")
    forbidden = [
        column
        for column in events.columns
        if any(token in column.lower() for token in FORBIDDEN_PRE_OUTCOME_TOKENS)
    ]
    if forbidden:
        raise AssertionError(f"forbidden pre-outcome fields: {forbidden}")
    return events, pd.DataFrame(counts)


def build_population_from_sources(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_parts: list[pd.DataFrame] = []
    count_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    for symbol in all_symbols(contract):
        symbol_tape, symbol_coverage = load_symbol_rows(contract, symbol)
        symbol_events, symbol_counts = build_population(symbol_tape)
        event_parts.append(symbol_events)
        count_parts.append(symbol_counts)
        coverage_parts.append(symbol_coverage)
    events = pd.concat(event_parts, ignore_index=True).sort_values(
        ["session_date", "symbol_norm", "decision_ordinal"], kind="stable"
    ).reset_index(drop=True)
    if events["event_id"].duplicated().any():
        raise AssertionError("streamed event identity drift")
    counts = pd.concat(count_parts, ignore_index=True).sort_values(
        ["period", "symbol_norm", "session_date"], kind="stable"
    ).reset_index(drop=True)
    coverage = pd.concat(coverage_parts, ignore_index=True).sort_values(
        ["period", "symbol_norm"], kind="stable"
    ).reset_index(drop=True)
    return events, counts, coverage


def outcome_for_event(event: Any, session: pd.DataFrame) -> dict[str, Any]:
    decision = int(event.decision_ordinal)
    by_ordinal = session.set_index("bar_ordinal", drop=False)
    path_ordinals = list(range(decision + 1, decision + HORIZON_BARS + 1))
    base = {
        "event_id": str(event.event_id),
        "period": int(event.period),
        "symbol_norm": str(event.symbol_norm),
        "session_date": str(event.session_date),
        "decision_ordinal": decision,
        "barrier_bps": float(event.barrier_bps),
    }
    if any(ordinal not in by_ordinal.index for ordinal in path_ordinals):
        return {**base, "score_status": "missing_exact_24_bar_path"}
    path = by_ordinal.loc[path_ordinals]
    entry = float(path.iloc[0]["open"])
    if not math.isfinite(entry) or entry <= 0:
        return {**base, "score_status": "invalid_entry_open"}
    width = float(event.barrier_bps)
    upper = entry * (1.0 + width / 10000.0)
    lower = entry * (1.0 - width / 10000.0)
    actual_class = "neutral"
    neutral_reason = "no_touch"
    first_touch_step: int | None = None
    for step, bar in enumerate(path.itertuples(index=False), start=1):
        bar_open = float(bar.open)
        if bar_open >= upper:
            actual_class = "long"
            neutral_reason = ""
            first_touch_step = step
            break
        if bar_open <= lower:
            actual_class = "short"
            neutral_reason = ""
            first_touch_step = step
            break
        upper_touch = float(bar.high) >= upper
        lower_touch = float(bar.low) <= lower
        if upper_touch and lower_touch:
            actual_class = "neutral"
            neutral_reason = "same_bar_dual_touch"
            first_touch_step = step
            break
        if upper_touch:
            actual_class = "long"
            neutral_reason = ""
            first_touch_step = step
            break
        if lower_touch:
            actual_class = "short"
            neutral_reason = ""
            first_touch_step = step
            break
    close_24 = float(path.iloc[-1]["close"])
    end_return_bps = 10000.0 * (close_24 / entry - 1.0)
    if actual_class == "long":
        long_gross = width
        short_gross = -width
    elif actual_class == "short":
        long_gross = -width
        short_gross = width
    elif neutral_reason == "same_bar_dual_touch":
        long_gross = -width
        short_gross = -width
    else:
        long_gross = end_return_bps
        short_gross = -end_return_bps
    return {
        **base,
        "score_status": "scored",
        "entry_timestamp": pd.Timestamp(path.iloc[0]["timestamp"]),
        "entry_open": entry,
        "upper_barrier": upper,
        "lower_barrier": lower,
        "actual_class": actual_class,
        "neutral_reason": neutral_reason,
        "first_touch_step": first_touch_step,
        "close_24": close_24,
        "end_return_bps": end_return_bps,
        "long_gross_bps": long_gross,
        "short_gross_bps": short_gross,
        "long_net_bps_5": long_gross - ROUND_TRIP_COST_BPS,
        "short_net_bps_5": short_gross - ROUND_TRIP_COST_BPS,
    }


def build_outcomes(events: pd.DataFrame, tape: pd.DataFrame) -> pd.DataFrame:
    sessions = {
        (int(period), str(symbol), str(date)): frame
        for (period, symbol, date), frame in tape.groupby(
            ["period", "symbol_norm", "session_date"], sort=False
        )
    }
    rows = []
    for event in events.itertuples(index=False):
        key = (int(event.period), str(event.symbol_norm), str(event.session_date))
        session = sessions.get(key, pd.DataFrame())
        rows.append(outcome_for_event(event, session))
    return pd.DataFrame(rows).sort_values("event_id", kind="stable").reset_index(drop=True)


def build_outcomes_from_sources(
    events: pd.DataFrame, contract: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    coverage_parts: list[pd.DataFrame] = []
    for symbol in all_symbols(contract):
        symbol_tape, symbol_coverage = load_symbol_rows(contract, symbol)
        coverage_parts.append(symbol_coverage)
        sessions = {
            (int(period), str(date)): frame
            for (period, date), frame in symbol_tape.groupby(
                ["period", "session_date"], sort=False
            )
        }
        symbol_events = events.loc[events["symbol_norm"].eq(symbol)]
        for event in symbol_events.itertuples(index=False):
            session = sessions.get((int(event.period), str(event.session_date)), pd.DataFrame())
            rows.append(outcome_for_event(event, session))
    outcomes = pd.DataFrame(rows).sort_values("event_id", kind="stable").reset_index(drop=True)
    coverage = pd.concat(coverage_parts, ignore_index=True).sort_values(
        ["period", "symbol_norm"], kind="stable"
    ).reset_index(drop=True)
    return outcomes, coverage


def build_pipeline(numeric: tuple[str, ...], categorical: tuple[str, ...]) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipe, list(numeric)))
    if categorical:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipe, list(categorical)))
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def economic_state(
    p_long: float, p_short: float, barrier_bps: float
) -> tuple[str, float, float]:
    long_ev = (p_long - p_short) * barrier_bps - ROUND_TRIP_COST_BPS
    short_ev = (p_short - p_long) * barrier_bps - ROUND_TRIP_COST_BPS
    if long_ev > 0.0 and long_ev > short_ev:
        return "long", long_ev, short_ev
    if short_ev > 0.0 and short_ev > long_ev:
        return "short", long_ev, short_ev
    return "neutral", long_ev, short_ev


def fit_prequential(joined: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    usable = joined.loc[joined["score_status"].eq("scored")].copy()
    usable = usable.replace([np.inf, -np.inf], np.nan)
    score_dates = sorted(
        usable.loc[usable["period"].isin([2025, 2026]), "session_date"].unique()
    )
    rows: list[dict[str, Any]] = []
    window = int(contract["model_fit"]["training_window_completed_sessions"])
    minimum_sessions = int(contract["model_fit"]["minimum_training_sessions"])
    minimum_rows = int(contract["model_fit"]["minimum_training_rows"])
    for score_date in score_dates:
        prior_dates = sorted(usable.loc[usable["session_date"].lt(score_date), "session_date"].unique())
        training_dates = prior_dates[-window:]
        if len(training_dates) < minimum_sessions:
            continue
        train = usable.loc[usable["session_date"].isin(training_dates)].copy()
        score = usable.loc[usable["session_date"].eq(score_date)].copy()
        if len(train) < minimum_rows or score.empty or train["actual_class"].nunique() < 3:
            continue
        for model_id in MODELS:
            numeric, categorical = MODEL_FEATURES[model_id]
            columns = [*numeric, *categorical]
            pipeline = build_pipeline(numeric, categorical)
            pipeline.fit(train[columns], train["actual_class"])
            model_classes = list(pipeline.named_steps["model"].classes_)
            raw = pipeline.predict_proba(score[columns])
            probability = {
                label: raw[:, model_classes.index(label)] for label in CLASSES
            }
            matrix = np.column_stack([probability[label] for label in CLASSES])
            class_index = np.argmax(matrix, axis=1)
            predicted_class = [CLASSES[index] for index in class_index]
            for offset, (_, item) in enumerate(score.iterrows()):
                p_long = float(probability["long"][offset])
                p_neutral = float(probability["neutral"][offset])
                p_short = float(probability["short"][offset])
                state, long_ev, short_ev = economic_state(
                    p_long, p_short, float(item["barrier_bps"])
                )
                if state == "long":
                    gross = float(item["long_gross_bps"])
                    net = float(item["long_net_bps_5"])
                elif state == "short":
                    gross = float(item["short_gross_bps"])
                    net = float(item["short_net_bps_5"])
                else:
                    gross = 0.0
                    net = 0.0
                rows.append(
                    {
                        "event_id": str(item["event_id"]),
                        "model_id": model_id,
                        "period": int(item["period"]),
                        "symbol_norm": str(item["symbol_norm"]),
                        "session_date": str(item["session_date"]),
                        "month": str(item["month"]),
                        "decision_ordinal": int(item["decision_ordinal"]),
                        "barrier_bps": float(item["barrier_bps"]),
                        "actual_class": str(item["actual_class"]),
                        "neutral_reason": str(item["neutral_reason"]),
                        "p_long": p_long,
                        "p_neutral": p_neutral,
                        "p_short": p_short,
                        "predicted_class": predicted_class[offset],
                        "economic_state": state,
                        "long_proxy_ev_bps": long_ev,
                        "short_proxy_ev_bps": short_ev,
                        "realized_gross_bps": gross,
                        "realized_net_bps_5": net,
                        "training_sessions": len(training_dates),
                        "training_rows": len(train),
                        "training_start": training_dates[0],
                        "training_end": training_dates[-1],
                    }
                )
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise AssertionError("no prequential predictions")
    return predictions.sort_values(
        ["session_date", "symbol_norm", "decision_ordinal", "model_id"], kind="stable"
    ).reset_index(drop=True)


def multiclass_brier(group: pd.DataFrame) -> float:
    actual = np.column_stack(
        [group["actual_class"].eq(label).to_numpy(float) for label in CLASSES]
    )
    predicted = group[[f"p_{label}" for label in CLASSES]].to_numpy(float)
    return float(np.mean(np.sum((predicted - actual) ** 2, axis=1)))


def predictive_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    for (period, model_id), group in predictions.groupby(["period", "model_id"], sort=True):
        y = group["actual_class"]
        predicted = group["predicted_class"]
        probability = group[[f"p_{label}" for label in CLASSES]].to_numpy(float)
        metrics.append(
            {
                "period": int(period),
                "model_id": str(model_id),
                "rows": len(group),
                "sessions": group["session_date"].nunique(),
                "stocks": group["symbol_norm"].nunique(),
                "accuracy": accuracy_score(y, predicted),
                "balanced_accuracy": balanced_accuracy_score(y, predicted),
                "macro_f1": f1_score(y, predicted, labels=list(CLASSES), average="macro", zero_division=0),
                "log_loss": log_loss(y, probability, labels=list(CLASSES)),
                "multiclass_brier": multiclass_brier(group),
                **{f"actual_{label}_rate": y.eq(label).mean() for label in CLASSES},
                **{f"predicted_{label}_rate": predicted.eq(label).mean() for label in CLASSES},
            }
        )
        precision, recall, f1_values, support = precision_recall_fscore_support(
            y, predicted, labels=list(CLASSES), zero_division=0
        )
        for index, label in enumerate(CLASSES):
            classes.append(
                {
                    "period": int(period),
                    "model_id": str(model_id),
                    "class": label,
                    "precision": precision[index],
                    "recall": recall[index],
                    "f1": f1_values[index],
                    "support": int(support[index]),
                    "predicted_rows": int(predicted.eq(label).sum()),
                }
            )
        matrix = confusion_matrix(y, predicted, labels=list(CLASSES))
        for actual_index, actual_label in enumerate(CLASSES):
            for predicted_index, predicted_label in enumerate(CLASSES):
                confusion.append(
                    {
                        "period": int(period),
                        "model_id": str(model_id),
                        "actual_class": actual_label,
                        "predicted_class": predicted_label,
                        "rows": int(matrix[actual_index, predicted_index]),
                    }
                )
    return pd.DataFrame(metrics), pd.DataFrame(classes), pd.DataFrame(confusion)


def policy_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    deletions: list[dict[str, Any]] = []
    for (period, model_id), group in predictions.groupby(["period", "model_id"], sort=True):
        selected = group.loc[~group["economic_state"].eq("neutral")]
        long_rows = selected.loc[selected["economic_state"].eq("long")]
        short_rows = selected.loc[selected["economic_state"].eq("short")]
        policy.append(
            {
                "period": int(period),
                "model_id": str(model_id),
                "opportunities": len(group),
                "directional_outputs": len(selected),
                "directional_coverage": len(selected) / len(group),
                "mean_net_per_directional_bps": selected["realized_net_bps_5"].mean(),
                "mean_net_per_opportunity_bps": group["realized_net_bps_5"].mean(),
                "directional_precision": selected["economic_state"].eq(selected["actual_class"]).mean(),
                "long_outputs": len(long_rows),
                "long_precision": long_rows["actual_class"].eq("long").mean(),
                "short_outputs": len(short_rows),
                "short_precision": short_rows["actual_class"].eq("short").mean(),
                "sessions": selected["session_date"].nunique(),
                "stocks": selected["symbol_norm"].nunique(),
                "same_bar_ambiguous_rate": group["neutral_reason"].eq("same_bar_dual_touch").mean(),
            }
        )
        for month, month_group in group.groupby("month", sort=True):
            month_selected = month_group.loc[~month_group["economic_state"].eq("neutral")]
            monthly.append(
                {
                    "period": int(period),
                    "model_id": str(model_id),
                    "month": str(month),
                    "opportunities": len(month_group),
                    "directional_outputs": len(month_selected),
                    "directional_coverage": len(month_selected) / len(month_group),
                    "mean_net_per_directional_bps": month_selected["realized_net_bps_5"].mean(),
                    "mean_net_per_opportunity_bps": month_group["realized_net_bps_5"].mean(),
                }
            )
        for cost in (2.5, 5.0, 7.5, 10.0):
            adjusted = np.where(
                group["economic_state"].eq("neutral"),
                0.0,
                group["realized_gross_bps"].to_numpy(float) - 2.0 * cost,
            )
            selected_adjusted = adjusted[~group["economic_state"].eq("neutral").to_numpy()]
            costs.append(
                {
                    "period": int(period),
                    "model_id": str(model_id),
                    "cost_bps_per_side": cost,
                    "mean_net_per_directional_bps": float(np.mean(selected_adjusted)) if len(selected_adjusted) else np.nan,
                    "mean_net_per_opportunity_bps": float(np.mean(adjusted)),
                }
            )
        for deleted_stock in sorted(group["symbol_norm"].unique()):
            remaining = group.loc[group["symbol_norm"].ne(deleted_stock)]
            remaining_selected = remaining.loc[~remaining["economic_state"].eq("neutral")]
            deletions.append(
                {
                    "period": int(period),
                    "model_id": str(model_id),
                    "deleted_stock": str(deleted_stock),
                    "opportunities": len(remaining),
                    "directional_outputs": len(remaining_selected),
                    "mean_net_per_directional_bps": remaining_selected["realized_net_bps_5"].mean(),
                    "mean_net_per_opportunity_bps": remaining["realized_net_bps_5"].mean(),
                }
            )
    return (
        pd.DataFrame(policy),
        pd.DataFrame(monthly),
        pd.DataFrame(costs),
        pd.DataFrame(deletions),
    )


def _block_draw_counts(sessions: list[str], draws: int, block: int, seed: int) -> np.ndarray:
    count = len(sessions)
    blocks = int(math.ceil(count / block))
    rng = np.random.default_rng(seed)
    output = np.zeros((draws, count), dtype=np.int16)
    offsets = np.arange(block, dtype=int)
    for draw in range(draws):
        starts = rng.integers(0, count, size=blocks)
        sampled = ((starts[:, None] + offsets[None, :]) % count).ravel()[:count]
        output[draw] = np.bincount(sampled, minlength=count)
    return output


def _macro_f1_from_confusion(matrix: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(matrix, axis1=1, axis2=2)
    predicted = matrix.sum(axis=1)
    actual = matrix.sum(axis=2)
    denominator = predicted + actual
    per_class = np.divide(
        2.0 * diagonal,
        denominator,
        out=np.zeros_like(diagonal, dtype=float),
        where=denominator > 0,
    )
    return per_class.mean(axis=1)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values, kind="stable")
    adjusted = np.zeros(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_metrics(predictions: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    draws = int(contract["evaluation"]["bootstrap_draws"])
    block = int(contract["evaluation"]["session_block_length"])
    rows: list[dict[str, Any]] = []
    endpoint_index = 0
    for period in (2025, 2026):
        period_rows = predictions.loc[predictions["period"].eq(period)].copy()
        for candidate, comparator in (
            ("M1_price_context", "M0_clock_prior"),
            ("M2_price_plus_activity", "M1_price_context"),
        ):
            candidate_rows = period_rows.loc[period_rows["model_id"].eq(candidate)].sort_values("event_id")
            comparator_rows = period_rows.loc[period_rows["model_id"].eq(comparator)].sort_values("event_id")
            if not candidate_rows["event_id"].tolist() == comparator_rows["event_id"].tolist():
                raise AssertionError("paired prediction population drift")
            sessions = sorted(candidate_rows["session_date"].unique())
            session_lookup = {session: index for index, session in enumerate(sessions)}
            weights = _block_draw_counts(sessions, draws, block, SEED + endpoint_index)
            endpoint_index += 1
            session_indices = candidate_rows["session_date"].map(session_lookup).to_numpy(int)
            actual_indices = candidate_rows["actual_class"].map(
                {label: index for index, label in enumerate(CLASSES)}
            ).to_numpy(int)
            confusion_by_session: dict[str, np.ndarray] = {}
            for model_name, model_rows in ((candidate, candidate_rows), (comparator, comparator_rows)):
                predicted_indices = model_rows["predicted_class"].map(
                    {label: index for index, label in enumerate(CLASSES)}
                ).to_numpy(int)
                matrices = np.zeros((len(sessions), len(CLASSES), len(CLASSES)), dtype=float)
                np.add.at(matrices, (session_indices, actual_indices, predicted_indices), 1.0)
                confusion_by_session[model_name] = matrices
            candidate_matrix = np.tensordot(weights, confusion_by_session[candidate], axes=(1, 0))
            comparator_matrix = np.tensordot(weights, confusion_by_session[comparator], axes=(1, 0))
            macro_draws = _macro_f1_from_confusion(candidate_matrix) - _macro_f1_from_confusion(comparator_matrix)
            observed_macro = f1_score(
                candidate_rows["actual_class"],
                candidate_rows["predicted_class"],
                labels=list(CLASSES),
                average="macro",
                zero_division=0,
            ) - f1_score(
                comparator_rows["actual_class"],
                comparator_rows["predicted_class"],
                labels=list(CLASSES),
                average="macro",
                zero_division=0,
            )

            actual_probability_index = actual_indices
            candidate_probability = candidate_rows[[f"p_{label}" for label in CLASSES]].to_numpy(float)
            comparator_probability = comparator_rows[[f"p_{label}" for label in CLASSES]].to_numpy(float)
            candidate_loss = -np.log(np.clip(candidate_probability[np.arange(len(candidate_rows)), actual_probability_index], 1e-15, 1.0))
            comparator_loss = -np.log(np.clip(comparator_probability[np.arange(len(candidate_rows)), actual_probability_index], 1e-15, 1.0))
            improvement = comparator_loss - candidate_loss
            session_sum = np.bincount(session_indices, weights=improvement, minlength=len(sessions))
            session_count = np.bincount(session_indices, minlength=len(sessions))
            loss_draws = (weights @ session_sum) / (weights @ session_count)
            observed_loss = float(improvement.mean())
            for endpoint, observed, values in (
                ("macro_f1_improvement", observed_macro, macro_draws),
                ("log_loss_improvement", observed_loss, loss_draws),
            ):
                rows.append(
                    {
                        "period": period,
                        "model_id": candidate,
                        "comparator": comparator,
                        "endpoint": endpoint,
                        "observed": observed,
                        "lower_95": float(np.quantile(values, 0.025)),
                        "upper_95": float(np.quantile(values, 0.975)),
                        "one_sided_p": (1.0 + float(np.sum(values <= 0.0))) / (draws + 1.0),
                    }
                )
        for candidate in ("M1_price_context", "M2_price_plus_activity"):
            model_rows = period_rows.loc[period_rows["model_id"].eq(candidate)].sort_values("event_id")
            sessions = sorted(model_rows["session_date"].unique())
            lookup = {session: index for index, session in enumerate(sessions)}
            session_indices = model_rows["session_date"].map(lookup).to_numpy(int)
            weights = _block_draw_counts(sessions, draws, block, SEED + endpoint_index)
            endpoint_index += 1
            net = model_rows["realized_net_bps_5"].to_numpy(float)
            full_sum = np.bincount(session_indices, weights=net, minlength=len(sessions))
            full_count = np.bincount(session_indices, minlength=len(sessions))
            full_draws = (weights @ full_sum) / (weights @ full_count)
            selected_mask = ~model_rows["economic_state"].eq("neutral").to_numpy()
            selected_sum = np.bincount(
                session_indices, weights=np.where(selected_mask, net, 0.0), minlength=len(sessions)
            )
            selected_count = np.bincount(
                session_indices, weights=selected_mask.astype(float), minlength=len(sessions)
            )
            selected_denominator = weights @ selected_count
            selected_draws = np.divide(
                weights @ selected_sum,
                selected_denominator,
                out=np.full(draws, np.nan),
                where=selected_denominator > 0,
            )
            for endpoint, observed, values in (
                ("mean_net_per_opportunity_bps", float(net.mean()), full_draws),
                (
                    "mean_net_per_directional_bps",
                    float(net[selected_mask].mean()) if selected_mask.any() else np.nan,
                    selected_draws,
                ),
            ):
                finite = values[np.isfinite(values)]
                rows.append(
                    {
                        "period": period,
                        "model_id": candidate,
                        "comparator": "zero",
                        "endpoint": endpoint,
                        "observed": observed,
                        "lower_95": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                        "upper_95": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                        "one_sided_p": (1.0 + float(np.sum(finite <= 0.0))) / (len(finite) + 1.0) if len(finite) else np.nan,
                    }
                )
    adjusted = _holm_adjust([float(row["one_sided_p"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
    return pd.DataFrame(rows)


def make_decision(
    model_metrics: pd.DataFrame,
    class_metrics: pd.DataFrame,
    policy: pd.DataFrame,
    monthly: pd.DataFrame,
    costs: pd.DataFrame,
    deletions: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    retained: list[str] = []
    for model_id, comparator in (
        ("M1_price_context", "M0_clock_prior"),
        ("M2_price_plus_activity", "M1_price_context"),
    ):
        period_checks: dict[str, Any] = {}
        for period in (2025, 2026):
            policy_row = policy.loc[
                policy["period"].eq(period) & policy["model_id"].eq(model_id)
            ].iloc[0]
            predictive_boot = bootstrap.loc[
                bootstrap["period"].eq(period)
                & bootstrap["model_id"].eq(model_id)
                & bootstrap["comparator"].eq(comparator)
            ]
            economic_boot = bootstrap.loc[
                bootstrap["period"].eq(period)
                & bootstrap["model_id"].eq(model_id)
                & bootstrap["comparator"].eq("zero")
            ]
            macro_row = predictive_boot.loc[predictive_boot["endpoint"].eq("macro_f1_improvement")].iloc[0]
            loss_row = predictive_boot.loc[predictive_boot["endpoint"].eq("log_loss_improvement")].iloc[0]
            selected_row = economic_boot.loc[economic_boot["endpoint"].eq("mean_net_per_directional_bps")].iloc[0]
            cost_row = costs.loc[
                costs["period"].eq(period)
                & costs["model_id"].eq(model_id)
                & costs["cost_bps_per_side"].eq(7.5)
            ].iloc[0]
            months = monthly.loc[
                monthly["period"].eq(period) & monthly["model_id"].eq(model_id)
            ]
            stocks = deletions.loc[
                deletions["period"].eq(period) & deletions["model_id"].eq(model_id)
            ]
            required_outputs = 300 if period == 2025 else 150
            checks = {
                "macro_f1_lower_positive": bool(macro_row["lower_95"] > 0),
                "log_loss_improvement_positive": bool(loss_row["observed"] > 0),
                "log_loss_holm": bool(loss_row["holm_adjusted_p"] < 0.05),
                "long_precision_at_least_0_55": bool(policy_row["long_precision"] >= 0.55),
                "short_precision_at_least_0_55": bool(policy_row["short_precision"] >= 0.55),
                "directional_net_lower_positive": bool(selected_row["lower_95"] > 0),
                "positive_at_7_5_bps_per_side": bool(cost_row["mean_net_per_opportunity_bps"] > 0),
                "positive_month_majority": bool(
                    months["mean_net_per_opportunity_bps"].gt(0).sum() * 2 >= len(months)
                ),
                "all_leave_one_stock_out_positive": bool(
                    stocks["mean_net_per_opportunity_bps"].gt(0).all()
                ),
                "minimum_directional_outputs": bool(policy_row["directional_outputs"] >= required_outputs),
                "minimum_directional_coverage": bool(policy_row["directional_coverage"] >= 0.05),
                "minimum_stock_breadth": bool(policy_row["stocks"] >= 15),
            }
            period_checks[str(period)] = checks
        passes = all(all(checks.values()) for checks in period_checks.values())
        candidates[model_id] = {
            "comparator": comparator,
            "checks": period_checks,
            "decision": (
                "descriptive_candidate_requires_genuinely_prospective_test"
                if passes
                else "rejected_or_descriptive_only"
            ),
        }
        if passes:
            retained.append(model_id)
    return {
        "contract_id": CONTRACT_ID,
        "scientific_status": "causal_retrospective_three_state_development_on_opened_data_not_validation",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "application_modified": False,
        "strategy_promotion": False,
        "validation_claim": False,
        "economic_edge_claim": False,
        "overall_decision": (
            "descriptive_candidate_only_requires_prospective_test"
            if retained
            else "all_three_state_detectors_rejected_or_descriptive_only"
        ),
        "retained_for_prospective_research_logging_only": retained,
        "models": candidates,
    }


def source_hash_payload(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in sorted(paths.items())
    }


def run_population(output_root: Path) -> None:
    contract = load_contract()
    paths = source_paths(contract)
    output_root.mkdir(parents=True, exist_ok=False)
    events, counts, coverage = build_population_from_sources(contract)
    events.to_parquet(output_root / "pre_outcome_events.parquet", index=False)
    coverage.to_csv(output_root / "data_coverage.csv", index=False)
    counts.to_csv(output_root / "population_counts.csv", index=False)
    sources = source_hash_payload(paths)
    write_json(output_root / "source_hashes.json", {"contract_id": CONTRACT_ID, "sources": sources})
    frozen_files = {}
    for name in (
        "pre_outcome_events.parquet",
        "data_coverage.csv",
        "population_counts.csv",
        "source_hashes.json",
    ):
        path = output_root / name
        frozen_files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(
        output_root / "pre_score_manifest.json",
        {
            "contract_id": CONTRACT_ID,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "event_rows": len(events),
            "sessions": events["session_date"].nunique(),
            "symbols": events["symbol_norm"].nunique(),
            "frozen_files": frozen_files,
            "source_hashes": sources,
        },
    )
    print(json.dumps({"phase": "population", "events": len(events), "root": str(output_root)}))


def verify_frozen(frozen_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((frozen_root / "pre_score_manifest.json").read_text(encoding="utf-8"))
    if manifest["contract_id"] != CONTRACT_ID:
        raise AssertionError("frozen contract id drift")
    for name, expected in manifest["frozen_files"].items():
        path = frozen_root / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise AssertionError(f"frozen file drift: {name}")
    current = source_hash_payload(source_paths(contract))
    if current != manifest["source_hashes"]:
        raise AssertionError("source hash drift")
    return manifest


def run_score(frozen_root: Path, output_root: Path) -> None:
    contract = load_contract()
    manifest = verify_frozen(frozen_root, contract)
    output_root.mkdir(parents=True, exist_ok=False)
    events = pd.read_parquet(frozen_root / "pre_outcome_events.parquet")
    outcomes, raw_coverage = build_outcomes_from_sources(events, contract)
    joined = events.merge(outcomes, on=["event_id", "period", "symbol_norm", "session_date", "decision_ordinal", "barrier_bps"], how="left", validate="one_to_one")
    predictions = fit_prequential(joined, contract)
    model_metrics, class_metrics, confusion = predictive_metrics(predictions)
    policy, monthly, costs, deletions = policy_metrics(predictions)
    bootstrap = bootstrap_metrics(predictions, contract)
    decision = make_decision(
        model_metrics, class_metrics, policy, monthly, costs, deletions, bootstrap
    )
    summary = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scientific_status": decision["scientific_status"],
        "historical_volume_label": contract["data"]["volume_label"],
        "quotes_or_ticks_used": False,
        "population_rows": len(events),
        "scored_outcome_rows": int(outcomes["score_status"].eq("scored").sum()),
        "missing_outcome_rows": int(outcomes["score_status"].ne("scored").sum()),
        "prediction_rows": len(predictions),
        "decision": decision,
        "model_metrics": model_metrics.to_dict(orient="records"),
        "policy_metrics": policy.to_dict(orient="records"),
    }
    events.to_parquet(output_root / "pre_outcome_events.parquet", index=False)
    outcomes.to_parquet(output_root / "outcomes.parquet", index=False)
    predictions.to_parquet(output_root / "predictions.parquet", index=False)
    pd.read_csv(frozen_root / "data_coverage.csv").to_csv(output_root / "data_coverage.csv", index=False)
    raw_coverage.to_csv(output_root / "raw_data_coverage.csv", index=False)
    pd.read_csv(frozen_root / "population_counts.csv").to_csv(output_root / "population_counts.csv", index=False)
    model_metrics.to_csv(output_root / "model_metrics.csv", index=False)
    class_metrics.to_csv(output_root / "class_metrics.csv", index=False)
    confusion.to_csv(output_root / "confusion_matrix.csv", index=False)
    policy.to_csv(output_root / "policy_metrics.csv", index=False)
    monthly.to_csv(output_root / "monthly_metrics.csv", index=False)
    costs.to_csv(output_root / "cost_sensitivity.csv", index=False)
    deletions.to_csv(output_root / "stock_deletions.csv", index=False)
    bootstrap.to_csv(output_root / "bootstrap_metrics.csv", index=False)
    write_json(output_root / "decision.json", decision)
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "source_hashes.json", {"contract_id": CONTRACT_ID, "sources": manifest["source_hashes"]})
    write_json(output_root / "pre_score_manifest.json", manifest)
    print(
        json.dumps(
            {
                "phase": "score",
                "population": len(events),
                "scored": int(outcomes["score_status"].eq("scored").sum()),
                "predictions": len(predictions),
                "decision": decision["overall_decision"],
                "root": str(output_root),
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("population", "score"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "population":
        run_population(args.output_root)
    else:
        if args.frozen_root is None:
            raise SystemExit("--frozen-root is required for score")
        run_score(args.frozen_root, args.output_root)


if __name__ == "__main__":
    main()
