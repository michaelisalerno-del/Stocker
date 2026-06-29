"""Research-only VWAP trade quality attribution diagnostics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from stocker_data.storage import DatasetKey, load_dataset
from stocker_research.hypothesis import HypothesisHoldingPolicy
from stocker_research.intraday_features import IntradayFeatureConfig, build_intraday_feature_frame
from stocker_research.position_policy import apply_holding_policy_to_positions
from stocker_research.templates import get_template

VWAP_TEMPLATE_NAME = "vwap_reclaim_rejection"
DEFAULT_OUTPUT_DIR = Path("data/reports/research/stage4_vwap_quality_attribution")
NEAR_VWAP_THRESHOLD = 0.001
NEAR_VWAP_WINDOW = 30


@dataclass(frozen=True)
class VWAPQualityReportResult:
    """Paths and headline counts from a VWAP attribution run."""

    summary_json_path: Path
    summary_markdown_path: Path
    trade_attribution_csv_path: Path
    feature_bucket_summary_csv_path: Path
    symbol_summary_csv_path: Path
    report_count_analyzed: int
    trade_count: int
    parameter_set_count: int


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _safe_median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _series_sign(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    signs = pd.Series(0, index=numeric.index, dtype=int)
    signs.loc[numeric > 0] = 1
    signs.loc[numeric < 0] = -1
    return signs


def _quality_for_session(
    group: pd.DataFrame, *, near_vwap_threshold: float, near_vwap_window: int
) -> pd.DataFrame:
    output = group.copy()
    diff = pd.to_numeric(output["close"], errors="coerce") - pd.to_numeric(
        output["session_vwap"],
        errors="coerce",
    )
    sign = _series_sign(diff)
    previous_sign = sign.shift(1).fillna(0).astype(int)
    crosses = (sign != 0) & (previous_sign != 0) & (sign != previous_sign)
    reclaims = (previous_sign < 0) & (sign > 0)
    output["vwap_cross_count_so_far"] = crosses.cumsum().astype(int)
    output["vwap_reclaim_count_so_far"] = reclaims.cumsum().astype(int)

    bars_since: list[int] = []
    last_cross_index: int | None = None
    for position, is_cross in enumerate(crosses.tolist()):
        if bool(is_cross):
            last_cross_index = position
            bars_since.append(0)
        elif last_cross_index is None:
            bars_since.append(position)
        else:
            bars_since.append(position - last_cross_index)
    output["bars_since_last_vwap_cross"] = bars_since

    distance = pd.to_numeric(
        output.get("distance_from_vwap", pd.Series(index=output.index)), errors="coerce"
    )
    near_vwap = distance.abs().lt(near_vwap_threshold).fillna(False).astype(float)
    output["near_vwap_pct_30_bars"] = near_vwap.rolling(
        near_vwap_window,
        min_periods=1,
    ).mean()
    vwap = pd.to_numeric(output["session_vwap"], errors="coerce")
    close = pd.to_numeric(output["close"], errors="coerce")
    for window in (3, 6, 12):
        output[f"vwap_slope_{window}_bars"] = vwap / vwap.shift(window) - 1.0
        output[f"close_momentum_{window}_bars"] = close / close.shift(window) - 1.0
    return output


def add_vwap_quality_features(
    frame: pd.DataFrame,
    *,
    near_vwap_threshold: float = NEAR_VWAP_THRESHOLD,
    near_vwap_window: int = NEAR_VWAP_WINDOW,
) -> pd.DataFrame:
    """Add VWAP quality features known at each row's bar close."""

    if "session_date" not in frame:
        raise ValueError("VWAP quality features require session_date.")
    required = {"close", "session_vwap"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"VWAP quality features missing columns: {missing}")
    session_frames = [
        _quality_for_session(
            group,
            near_vwap_threshold=near_vwap_threshold,
            near_vwap_window=near_vwap_window,
        )
        for _, group in frame.copy().groupby("session_date", sort=False)
    ]
    enriched = pd.concat(session_frames, ignore_index=True) if session_frames else frame.copy()
    opening_mid = pd.to_numeric(enriched.get("opening_range_mid"), errors="coerce")
    opening_high = pd.to_numeric(enriched.get("opening_range_high"), errors="coerce")
    opening_width = pd.to_numeric(enriched.get("opening_range_width"), errors="coerce")
    close = pd.to_numeric(enriched["close"], errors="coerce")
    enriched["opening_range_width_pct"] = opening_width / opening_mid
    enriched.loc[opening_mid <= 0, "opening_range_width_pct"] = math.nan
    enriched["price_above_opening_range_mid"] = (close > opening_mid).fillna(False)
    enriched["price_above_opening_range_high"] = (close > opening_high).fillna(False)
    minutes_from_open = pd.to_numeric(enriched.get("minutes_from_session_open"), errors="coerce")
    enriched["time_of_day_bucket"] = "unknown"
    enriched.loc[minutes_from_open < 120, "time_of_day_bucket"] = "opening"
    enriched.loc[(minutes_from_open >= 120) & (minutes_from_open < 300), "time_of_day_bucket"] = (
        "midday"
    )
    enriched.loc[minutes_from_open >= 300, "time_of_day_bucket"] = "late_day"
    return enriched


def _bar_returns(frame: pd.DataFrame, positions: pd.Series, *, cost_bps: float) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="coerce").reset_index(drop=True)
    position = positions.reset_index(drop=True).astype(float).reindex(close.index).fillna(0.0)
    gross_returns = position.shift(1).fillna(0.0) * close.pct_change(fill_method=None).fillna(0.0)
    position_change = position.diff().abs().fillna(position.abs())
    cost_returns = position_change * cost_bps / 10_000.0
    return pd.DataFrame(
        {
            "position": position,
            "gross_return": gross_returns,
            "cost_return": cost_returns,
            "net_return": gross_returns - cost_returns,
        }
    )


def _exit_reason(row: pd.Series, exit_mode: str) -> str:
    if bool(row.get("time_stop_exit", False)):
        return "time_stop"
    if bool(row.get("vwap_lost_exit", False)):
        return "vwap_lost"
    return exit_mode


def _timestamp_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _entry_context(row: pd.Series) -> dict[str, Any]:
    return {
        "session_date": row.get("session_date", ""),
        "bar_index_in_session": row.get("bar_index_in_session", ""),
        "minutes_from_open": row.get("minutes_from_session_open", ""),
        "minutes_to_close": row.get("minutes_to_session_close", ""),
        "entry_close": row.get("close", ""),
        "session_vwap": row.get("session_vwap", ""),
        "distance_from_vwap": row.get("distance_from_vwap", ""),
        "relative_volume_at_bar_index": row.get("relative_volume_at_bar_index", ""),
        "relative_cumulative_volume": row.get("relative_cumulative_volume", ""),
        "opening_range_high": row.get("opening_range_high", ""),
        "opening_range_low": row.get("opening_range_low", ""),
        "opening_range_mid": row.get("opening_range_mid", ""),
        "opening_range_width_pct": row.get("opening_range_width_pct", ""),
        "price_above_opening_range_mid": bool(row.get("price_above_opening_range_mid", False)),
        "price_above_opening_range_high": bool(row.get("price_above_opening_range_high", False)),
        "vwap_slope_3_bars": row.get("vwap_slope_3_bars", ""),
        "vwap_slope_6_bars": row.get("vwap_slope_6_bars", ""),
        "vwap_slope_12_bars": row.get("vwap_slope_12_bars", ""),
        "close_momentum_3_bars": row.get("close_momentum_3_bars", ""),
        "close_momentum_6_bars": row.get("close_momentum_6_bars", ""),
        "close_momentum_12_bars": row.get("close_momentum_12_bars", ""),
        "vwap_cross_count_so_far": row.get("vwap_cross_count_so_far", ""),
        "bars_since_last_vwap_cross": row.get("bars_since_last_vwap_cross", ""),
        "vwap_reclaim_count_so_far": row.get("vwap_reclaim_count_so_far", ""),
        "near_vwap_pct_30_bars": row.get("near_vwap_pct_30_bars", ""),
        "time_of_day_bucket": row.get("time_of_day_bucket", "unknown"),
        "session_quality_warning": bool(row.get("session_complete_warning", False)),
    }


def reconstruct_round_trip_trades(
    frame: pd.DataFrame,
    positions: pd.Series,
    *,
    signals: pd.DataFrame | None = None,
    cost_bps: float = 0.0,
    symbol: str = "",
    experiment_id: str = "",
    parameter_set_id: str = "",
    exit_mode: str = "",
    split_id: str = "",
    is_selected_parameter_set: bool = False,
) -> list[dict[str, Any]]:
    """Reconstruct long-only round trips from scored target positions."""

    data = frame.reset_index(drop=True).copy()
    position = positions.reset_index(drop=True).astype(float).reindex(data.index).fillna(0.0)
    signal_frame = (
        signals.reset_index(drop=True).reindex(data.index)
        if signals is not None
        else pd.DataFrame(index=data.index)
    )
    returns = _bar_returns(data, position, cost_bps=cost_bps)
    trades: list[dict[str, Any]] = []
    active = False
    entry_index = 0
    gross_return = 0.0
    net_return = 0.0
    previous_position = 0.0

    for index, current_position in enumerate(returns["position"].tolist()):
        opens_trade = not active and abs(previous_position) == 0.0 and abs(current_position) > 0.0
        closes_trade = active and abs(previous_position) > 0.0 and abs(current_position) == 0.0
        if opens_trade:
            active = True
            entry_index = index
            gross_return = 0.0
            net_return = 0.0
        if active:
            gross_return += float(returns.loc[index, "gross_return"])
            net_return += float(returns.loc[index, "net_return"])
        if closes_trade:
            entry_row = data.loc[entry_index]
            exit_row = data.loc[index]
            signal_entry = (
                signal_frame.loc[entry_index] if entry_index in signal_frame.index else pd.Series()
            )
            signal_exit = signal_frame.loc[index] if index in signal_frame.index else pd.Series()
            entry_mode = str(signal_entry.get("entry_mode", "") or "")
            trades.append(
                {
                    "symbol": symbol,
                    "experiment_id": experiment_id,
                    "parameter_set_id": parameter_set_id,
                    "split_id": split_id,
                    "is_selected_parameter_set": is_selected_parameter_set,
                    "entry_timestamp": _timestamp_text(entry_row.get("timestamp")),
                    "exit_timestamp": _timestamp_text(exit_row.get("timestamp")),
                    "holding_bars": int(index - entry_index),
                    "entry_mode": entry_mode,
                    "exit_mode": _exit_reason(signal_exit, exit_mode),
                    "trade_gross_return": float(gross_return),
                    "trade_net_return": float(net_return),
                    "win_loss": "win" if net_return > 0 else "loss",
                    **_entry_context(entry_row),
                }
            )
            active = False
            gross_return = 0.0
            net_return = 0.0
        previous_position = float(current_position)

    if active and len(data) > 0:
        entry_row = data.loc[entry_index]
        exit_row = data.loc[len(data) - 1]
        signal_entry = (
            signal_frame.loc[entry_index] if entry_index in signal_frame.index else pd.Series()
        )
        entry_mode = str(signal_entry.get("entry_mode", "") or "")
        trades.append(
            {
                "symbol": symbol,
                "experiment_id": experiment_id,
                "parameter_set_id": parameter_set_id,
                "split_id": split_id,
                "is_selected_parameter_set": is_selected_parameter_set,
                "entry_timestamp": _timestamp_text(entry_row.get("timestamp")),
                "exit_timestamp": _timestamp_text(exit_row.get("timestamp")),
                "holding_bars": int(len(data) - 1 - entry_index),
                "entry_mode": entry_mode,
                "exit_mode": "window_end",
                "trade_gross_return": float(gross_return),
                "trade_net_return": float(net_return),
                "win_loss": "win" if net_return > 0 else "loss",
                **_entry_context(entry_row),
            }
        )
    return trades


def _profit_factor(values: list[float]) -> float:
    wins = sum(value for value in values if value > 0)
    losses = sum(value for value in values if value < 0)
    if losses < 0:
        return float(wins / abs(losses))
    if wins > 0:
        return float("inf")
    return 0.0


def _top_winner_share(values: list[float], top_count: int = 5) -> float:
    positives = sorted((value for value in values if value > 0), reverse=True)
    total = sum(positives)
    return _safe_divide(sum(positives[:top_count]), total)


def _top_loser_share(values: list[float], top_count: int = 5) -> float:
    losses = sorted(value for value in values if value < 0)
    total = abs(sum(losses))
    return _safe_divide(abs(sum(losses[:top_count])), total)


def _summary_for_rows(rows: pd.DataFrame) -> dict[str, Any]:
    gross = [_finite_float(value) for value in rows.get("trade_gross_return", [])]
    net = [_finite_float(value) for value in rows.get("trade_net_return", [])]
    holding = [_finite_float(value) for value in rows.get("holding_bars", [])]
    return {
        "trade_count": int(len(rows)),
        "mean_gross_return": _safe_mean(gross),
        "median_gross_return": _safe_median(gross),
        "mean_net_return": _safe_mean(net),
        "median_net_return": _safe_median(net),
        "win_rate": float(sum(value > 0 for value in net) / len(net)) if net else 0.0,
        "profit_factor": _profit_factor(net),
        "average_holding_bars": _safe_mean(holding),
        "median_holding_bars": _safe_median(holding),
        "top_5_winner_contribution_share": _top_winner_share(net),
        "loser_concentration_share": _top_loser_share(net),
    }


def _bucket_cross_count(value: Any) -> str:
    count = int(_finite_float(value, -1))
    if count < 0:
        return "unknown"
    if count == 0:
        return "0"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    return "6+"


def _bucket_reclaim_count(value: Any) -> str:
    count = int(_finite_float(value, -1))
    if count < 0:
        return "unknown"
    if count == 0:
        return "0"
    if count <= 2:
        return "1-2"
    return "3+"


def _bucket_bars_since_cross(value: Any) -> str:
    count = int(_finite_float(value, -1))
    if count < 0:
        return "unknown"
    if count <= 1:
        return "0-1"
    if count <= 5:
        return "2-5"
    if count <= 12:
        return "6-12"
    return "13+"


def _bucket_slope(value: Any) -> str:
    slope = _finite_float(value, math.nan)
    if not math.isfinite(slope):
        return "unknown"
    if slope > 0:
        return "rising"
    if slope < 0:
        return "falling"
    return "flat"


def _bucket_opening_range(row: pd.Series) -> str:
    if bool(row.get("price_above_opening_range_high", False)):
        return "above_high"
    if bool(row.get("price_above_opening_range_mid", False)):
        return "above_mid"
    if pd.isna(row.get("opening_range_mid", math.nan)):
        return "unknown"
    return "below_mid"


def _bucket_relative_cumulative_volume(value: Any) -> str:
    ratio = _finite_float(value, math.nan)
    if not math.isfinite(ratio):
        return "unknown"
    if ratio < 0.8:
        return "<0.8"
    if ratio < 1.1:
        return "0.8-1.1"
    if ratio < 1.2:
        return "1.1-1.2"
    return ">=1.2"


def _bucket_near_vwap(value: Any) -> str:
    ratio = _finite_float(value, math.nan)
    if not math.isfinite(ratio):
        return "unknown"
    if ratio < 0.25:
        return "<25%"
    if ratio < 0.5:
        return "25-50%"
    return ">=50%"


def _bucketed_frame(trades: pd.DataFrame) -> pd.DataFrame:
    data = trades.copy()
    if data.empty:
        return data
    data["vwap_cross_count_bucket"] = data.get(
        "vwap_cross_count_so_far",
        pd.Series(math.nan, index=data.index),
    ).map(_bucket_cross_count)
    data["reclaim_count_bucket"] = data.get(
        "vwap_reclaim_count_so_far",
        pd.Series(math.nan, index=data.index),
    ).map(_bucket_reclaim_count)
    data["bars_since_last_vwap_cross_bucket"] = data.get(
        "bars_since_last_vwap_cross",
        pd.Series(math.nan, index=data.index),
    ).map(_bucket_bars_since_cross)
    data["vwap_slope_bucket"] = data.get(
        "vwap_slope_6_bars",
        pd.Series(math.nan, index=data.index),
    ).map(_bucket_slope)
    data["opening_range_position_bucket"] = data.apply(_bucket_opening_range, axis=1)
    data["relative_cumulative_volume_bucket"] = data.get(
        "relative_cumulative_volume",
        pd.Series(math.nan, index=data.index),
    ).map(_bucket_relative_cumulative_volume)
    data["near_vwap_chop_bucket"] = data.get(
        "near_vwap_pct_30_bars",
        pd.Series(math.nan, index=data.index),
    ).map(_bucket_near_vwap)
    return data


def build_bucket_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize VWAP trade outcomes by diagnostic feature buckets."""

    if trades.empty:
        return pd.DataFrame()
    data = _bucketed_frame(trades)
    bucket_columns = {
        "entry_mode": "entry_mode",
        "exit_mode": "exit_mode",
        "time_of_day_bucket": "time_of_day_bucket",
        "vwap_cross_count_bucket": "vwap_cross_count_bucket",
        "reclaim_count_bucket": "reclaim_count_bucket",
        "bars_since_last_vwap_cross_bucket": "bars_since_last_vwap_cross_bucket",
        "vwap_slope_bucket": "vwap_slope_bucket",
        "opening_range_position_bucket": "opening_range_position_bucket",
        "relative_cumulative_volume_bucket": "relative_cumulative_volume_bucket",
        "near_vwap_chop_bucket": "near_vwap_chop_bucket",
    }
    rows: list[dict[str, Any]] = []
    for feature_name, column in bucket_columns.items():
        if column not in data:
            continue
        for bucket, group in data.groupby(column, dropna=False, sort=True):
            rows.append(
                {
                    "feature": feature_name,
                    "bucket": str(bucket),
                    **_summary_for_rows(group),
                }
            )
    return pd.DataFrame(rows)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_dataframe_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_universe_symbols(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = _load_json(path)
    symbols = payload.get("qualified_symbols", [])
    output: set[str] = set()
    for item in symbols:
        if isinstance(item, dict) and item.get("symbol"):
            output.add(str(item["symbol"]).upper())
        elif isinstance(item, str):
            output.add(item.upper())
    return output


def _report_symbol(payload: dict[str, Any]) -> str:
    return str(payload.get("symbol", "")).upper()


def _is_vwap_report(payload: dict[str, Any], *, timeframe: str, source: str) -> bool:
    hypothesis = payload.get("hypothesis", {})
    return (
        isinstance(hypothesis, dict)
        and hypothesis.get("template") == VWAP_TEMPLATE_NAME
        and str(payload.get("timeframe", "")) == timeframe
        and str(payload.get("source", "")) == source
        and bool(payload.get("grid_results"))
    )


def _latest_universe_payload(reports_dir: Path, *, timeframe: str) -> dict[str, Any] | None:
    universe_dir = reports_dir / "universe"
    candidates = sorted(
        universe_dir.glob(f"*vwap_reclaim_rejection_intraday_session_flat_v1*_{timeframe}.json"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for path in candidates:
        payload = _load_json(path)
        if payload.get("symbol_results"):
            return payload
    return None


def find_vwap_report_paths(
    *,
    reports_dir: str | Path,
    timeframe: str = "5m",
    source: str = "eodhd",
    universe_path: str | Path | None = None,
    symbols: list[str] | None = None,
    explicit_report_paths: list[str | Path] | None = None,
) -> list[Path]:
    """Find the latest VWAP per-symbol experiment reports."""

    if explicit_report_paths:
        return [Path(path) for path in explicit_report_paths]

    reports_root = Path(reports_dir)
    allowed_symbols = _load_universe_symbols(Path(universe_path) if universe_path else None)
    if symbols:
        requested = {symbol.upper() for symbol in symbols}
        allowed_symbols = requested if allowed_symbols is None else allowed_symbols & requested

    universe_payload = _latest_universe_payload(reports_root, timeframe=timeframe)
    if universe_payload is not None:
        paths: list[Path] = []
        for row in universe_payload.get("symbol_results", []):
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "")).upper()
            if allowed_symbols is not None and symbol not in allowed_symbols:
                continue
            json_path = row.get("json_path")
            if json_path:
                paths.append(Path(json_path))
        if paths:
            return paths

    latest_by_symbol: dict[str, Path] = {}
    for path in reports_root.glob(
        f"*vwap_reclaim_rejection_intraday_session_flat_v1*_{timeframe}.json"
    ):
        payload = _load_json(path)
        if not _is_vwap_report(payload, timeframe=timeframe, source=source):
            continue
        symbol = _report_symbol(payload)
        if allowed_symbols is not None and symbol not in allowed_symbols:
            continue
        current = latest_by_symbol.get(symbol)
        if current is None or (path.stat().st_mtime, path.name) > (
            current.stat().st_mtime,
            current.name,
        ):
            latest_by_symbol[symbol] = path
    return [latest_by_symbol[symbol] for symbol in sorted(latest_by_symbol)]


def _cost_bps(payload: dict[str, Any]) -> float:
    costs = payload.get("cost_assumptions", {})
    if not isinstance(costs, dict):
        return 0.0
    return (
        _finite_float(costs.get("spread_bps"))
        + _finite_float(costs.get("commission_bps"))
        + _finite_float(costs.get("slippage_bps"))
    )


def _holding_policy(payload: dict[str, Any]) -> HypothesisHoldingPolicy | None:
    raw = payload.get("holding_policy")
    if not isinstance(raw, dict):
        return None
    return HypothesisHoldingPolicy.model_validate(raw)


def _split_ranges(payload: dict[str, Any]) -> list[dict[str, Any]]:
    splits = payload.get("splits", [])
    if not isinstance(splits, list):
        return []
    return [split for split in splits if isinstance(split, dict)]


def _parameter_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("grid_results", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get("params"), dict)]


def _features_for_params(
    frame: pd.DataFrame,
    params: dict[str, Any],
    *,
    timeframe: str,
    market_calendar: str | None,
    cache: dict[tuple[Any, ...], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    cache_key = (
        str(params.get("timeframe", timeframe)),
        str(params.get("market_calendar", market_calendar)),
        int(params.get("open_buffer_minutes", 0)),
        int(params.get("entry_cutoff_before_close_minutes", 30)),
        int(params.get("flatten_before_close_minutes", 10)),
        int(params.get("relative_volume_lookback_sessions", 20)),
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key].copy()
    config = IntradayFeatureConfig(
        timeframe=str(params.get("timeframe", timeframe)),
        market_calendar=params.get("market_calendar", market_calendar),
        open_buffer_minutes=int(params.get("open_buffer_minutes", 0)),
        entry_cutoff_before_close_minutes=int(params.get("entry_cutoff_before_close_minutes", 30)),
        flatten_before_close_minutes=int(params.get("flatten_before_close_minutes", 10)),
        relative_volume_lookback_sessions=int(params.get("relative_volume_lookback_sessions", 20)),
    )
    features = add_vwap_quality_features(build_intraday_feature_frame(frame, config))
    if cache is not None:
        cache[cache_key] = features.copy()
    return features


def _test_window_mask(length: int, splits: list[dict[str, Any]]) -> pd.Series:
    mask = pd.Series(False, index=range(length))
    if not splits:
        return pd.Series(True, index=range(length))
    for split in splits:
        start = max(0, int(split.get("test_start", 0)))
        end = min(length, int(split.get("test_end", 0)))
        if end > start:
            mask.iloc[start:end] = True
    return mask


def _split_ids(length: int, splits: list[dict[str, Any]]) -> pd.Series:
    values = pd.Series("", index=range(length), dtype=object)
    if not splits:
        values[:] = "full_sample"
        return values
    for index, split in enumerate(splits):
        start = max(0, int(split.get("test_start", 0)))
        end = min(length, int(split.get("test_end", 0)))
        split_id = str(split.get("split_id", f"split_{index + 1:03d}"))
        if end > start:
            values.iloc[start:end] = split_id
    return values


def _attribute_report(
    report_path: Path,
    *,
    data_dir: Path,
    timeframe: str,
    source: str,
    instrument_type: str,
    market_calendar: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _load_json(report_path)
    if not _is_vwap_report(payload, timeframe=timeframe, source=source):
        return [], {"report_path": str(report_path), "skipped": True, "reason": "not_vwap_report"}

    symbol = _report_symbol(payload)
    frame = (
        load_dataset(
            DatasetKey(
                source=source,
                instrument_type=instrument_type,
                symbol=symbol,
                timeframe=timeframe,
            ),
            data_dir=data_dir,
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    template = get_template(VWAP_TEMPLATE_NAME)
    holding_policy = _holding_policy(payload)
    splits = _split_ranges(payload)
    split_mask = _test_window_mask(len(frame), splits)
    split_ids = _split_ids(len(frame), splits)
    selected_id = str(payload.get("selected_result", {}).get("parameter_set_id", ""))
    one_way_cost_bps = _cost_bps(payload)
    report_trades: list[dict[str, Any]] = []
    feature_cache: dict[tuple[Any, ...], pd.DataFrame] = {}
    for row in _parameter_rows(payload):
        params = dict(row["params"])
        parameter_set_id = str(row.get("parameter_set_id", "unknown"))
        params["parameter_set_id"] = parameter_set_id
        params.setdefault("timeframe", timeframe)
        params.setdefault("market_calendar", market_calendar)
        features = _features_for_params(
            frame,
            params,
            timeframe=timeframe,
            market_calendar=market_calendar,
            cache=feature_cache,
        )
        signals = template.generate_signals(frame, params).reset_index(drop=True)
        positions = pd.to_numeric(signals["target_position"], errors="coerce").fillna(0.0)
        if holding_policy is not None:
            policy_result = apply_holding_policy_to_positions(
                frame,
                positions,
                policy=holding_policy,
                timeframe=timeframe,
                market_calendar=market_calendar,
            )
            positions = policy_result.adjusted_positions.reset_index(drop=True)
        attributed = features.copy()
        for column in signals.columns:
            if column not in attributed:
                attributed[column] = signals[column]
        attributed["split_id"] = split_ids
        attributed = attributed.loc[split_mask].reset_index(drop=True)
        split_positions = positions.loc[split_mask].reset_index(drop=True)
        split_signals = signals.loc[split_mask].reset_index(drop=True)
        for split_id, group_indices in attributed.groupby("split_id", sort=False).groups.items():
            indices = list(group_indices)
            trades = reconstruct_round_trip_trades(
                attributed.loc[indices].reset_index(drop=True),
                split_positions.loc[indices].reset_index(drop=True),
                signals=split_signals.loc[indices].reset_index(drop=True),
                cost_bps=one_way_cost_bps,
                symbol=symbol,
                experiment_id=str(payload.get("experiment_id", "")),
                parameter_set_id=parameter_set_id,
                exit_mode=str(params.get("exit_mode", "")),
                split_id=str(split_id),
                is_selected_parameter_set=parameter_set_id == selected_id,
            )
            for trade in trades:
                if not trade["entry_mode"]:
                    trade["entry_mode"] = str(params.get("entry_mode", ""))
                report_trades.append(trade)
    metadata = {
        "report_path": str(report_path),
        "symbol": symbol,
        "experiment_id": payload.get("experiment_id", ""),
        "parameter_set_count": len(_parameter_rows(payload)),
        "trade_count": len(report_trades),
        "robustness_flags": payload.get("robustness_diagnostics", {}).get(
            "robustness_flags",
            [],
        ),
        "classification": payload.get("classification", ""),
        "classification_reasons": payload.get("classification_reasons", []),
    }
    return report_trades, metadata


def _filter_summary(name: str, trades: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    group = trades.loc[mask.fillna(False)]
    summary = _summary_for_rows(group)
    return {"filter": name, **summary}


def _candidate_filter_summaries(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    numeric = trades.copy()
    for column in (
        "vwap_reclaim_count_so_far",
        "vwap_cross_count_so_far",
        "near_vwap_pct_30_bars",
        "vwap_slope_6_bars",
        "vwap_slope_12_bars",
        "relative_cumulative_volume",
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    filters = [
        (
            "first_second_reclaim_only",
            (numeric["entry_mode"] == "reclaim") & numeric["vwap_reclaim_count_so_far"].le(2),
        ),
        ("vwap_cross_count_le_2", numeric["vwap_cross_count_so_far"].le(2)),
        ("vwap_cross_count_le_3", numeric["vwap_cross_count_so_far"].le(3)),
        (
            "avoid_near_vwap_chop_lt_50pct",
            numeric["near_vwap_pct_30_bars"].lt(0.50),
        ),
        ("rising_vwap_6_bars", numeric["vwap_slope_6_bars"].gt(0)),
        ("rising_vwap_12_bars", numeric["vwap_slope_12_bars"].gt(0)),
        ("above_opening_range_mid", numeric["price_above_opening_range_mid"].astype(bool)),
        ("above_opening_range_high", numeric["price_above_opening_range_high"].astype(bool)),
        ("relative_cumulative_volume_gt_1_1", numeric["relative_cumulative_volume"].gt(1.1)),
        ("relative_cumulative_volume_gt_1_2", numeric["relative_cumulative_volume"].gt(1.2)),
        ("opening_time_bucket", numeric["time_of_day_bucket"].eq("opening")),
        ("avoid_midday", ~numeric["time_of_day_bucket"].eq("midday")),
        ("avoid_late_day", ~numeric["time_of_day_bucket"].eq("late_day")),
    ]
    return [_filter_summary(name, numeric, mask) for name, mask in filters]


def _rank_filters(
    filter_summaries: list[dict[str, Any]],
    baseline: dict[str, Any],
    *,
    minimum_trades: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    baseline_median = float(baseline.get("median_net_return", 0.0))
    baseline_mean = float(baseline.get("mean_net_return", 0.0))
    enough = [row for row in filter_summaries if int(row.get("trade_count", 0)) >= minimum_trades]
    dangerous = [
        row for row in filter_summaries if 0 < int(row.get("trade_count", 0)) < minimum_trades
    ]
    for row in enough:
        row["median_net_uplift_vs_all"] = float(row.get("median_net_return", 0.0)) - baseline_median
        row["mean_net_uplift_vs_all"] = float(row.get("mean_net_return", 0.0)) - baseline_mean
    promising = sorted(
        [
            row
            for row in enough
            if row["median_net_uplift_vs_all"] > 0 and row["mean_net_uplift_vs_all"] > 0
        ],
        key=lambda row: (
            float(row.get("median_net_uplift_vs_all", 0.0)),
            float(row.get("mean_net_uplift_vs_all", 0.0)),
        ),
        reverse=True,
    )[:3]
    weak = sorted(
        [
            row
            for row in enough
            if row["median_net_uplift_vs_all"] <= 0 or row["mean_net_uplift_vs_all"] <= 0
        ],
        key=lambda row: str(row.get("filter", "")),
    )
    return {
        "promising_filters": promising,
        "weak_or_no_signal_filters": weak,
        "dangerous_too_few_trades_filters": dangerous,
    }


def _symbol_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for symbol, group in trades.groupby("symbol", sort=True):
        rows.append({"symbol": symbol, **_summary_for_rows(group)})
    return pd.DataFrame(rows)


def _mode_comparison(bucket_summary: pd.DataFrame) -> dict[str, Any]:
    if bucket_summary.empty:
        return {}
    rows = bucket_summary[bucket_summary["feature"].eq("entry_mode")]
    return {
        str(row["bucket"]): {
            "trade_count": int(row["trade_count"]),
            "mean_net_return": float(row["mean_net_return"]),
            "median_net_return": float(row["median_net_return"]),
            "profit_factor": float(row["profit_factor"]),
        }
        for _, row in rows.iterrows()
    }


def _boolean_answer(bucket_summary: pd.DataFrame, feature: str, good_bucket: str) -> dict[str, Any]:
    if bucket_summary.empty:
        return {"answer": "not_available"}
    rows = bucket_summary[bucket_summary["feature"].eq(feature)]
    if rows.empty:
        return {"answer": "not_available"}
    target = rows[rows["bucket"].eq(good_bucket)]
    others = rows[~rows["bucket"].eq(good_bucket)]
    if target.empty or others.empty:
        return {"answer": "not_enough_comparison"}
    target_median = float(target.iloc[0]["median_net_return"])
    other_median = float(others["median_net_return"].median())
    return {
        "answer": "yes" if target_median > other_median else "no",
        "target_median_net_return": target_median,
        "other_bucket_median_net_return": other_median,
    }


def _markdown(summary: dict[str, Any]) -> str:
    promising = summary["filter_rankings"]["promising_filters"]
    weak = summary["filter_rankings"]["weak_or_no_signal_filters"]
    dangerous = summary["filter_rankings"]["dangerous_too_few_trades_filters"]

    def filter_lines(rows: list[dict[str, Any]], empty: str) -> str:
        if not rows:
            return empty
        return "\n".join(
            "- `{filter}`: trades={trade_count}, median_net={median_net_return:.6f}, "
            "mean_net={mean_net_return:.6f}, win_rate={win_rate:.3f}".format(**row)
            for row in rows
        )

    recommendations = summary["recommended_vwap_v2_parameter_changes"]
    recommendation_lines = (
        "\n".join(f"- {item}" for item in recommendations) if recommendations else "- None"
    )
    mode = summary["entry_mode_comparison"]
    reclaim = mode.get("reclaim", {})
    rejection = mode.get("rejection", {})
    reclaim_vs_rejection = summary["questions"]["did_reclaim_beat_rejection"]
    early_answer = summary["questions"]["were_winners_concentrated_early"]["answer"]
    rising_vwap_answer = summary["questions"]["did_rising_vwap_help"]["answer"]
    low_cross_answer = summary["questions"]["did_low_cross_count_help"]["answer"]
    opening_range_answer = summary["questions"]["did_opening_range_position_help"]["answer"]
    relative_volume_answer = summary["questions"]["did_high_relative_cumulative_volume_help"][
        "answer"
    ]
    chop_answer = summary["questions"]["did_near_vwap_chop_predict_bad_trades"]["answer"]
    enough_trades_answer = summary["questions"]["enough_trades_after_filters"]["answer"]
    reclaim_trade_count = reclaim.get("trade_count", 0)
    reclaim_median_net = reclaim.get("median_net_return", 0.0)
    rejection_trade_count = rejection.get("trade_count", 0)
    rejection_median_net = rejection.get("median_net_return", 0.0)
    return f"""# VWAP Quality Attribution

This is a research-only diagnostic. It does not create candidates, loosen gates,
change strategy logic, fetch data, or alter classifications.

## Scope

- Reports analyzed: {summary["report_count_analyzed"]}
- Symbols analyzed: {summary["symbol_count"]}
- Parameter sets attributed: {summary["parameter_set_count"]}
- Trades attributed: {summary["trade_count"]}
- Timeframe: `{summary["timeframe"]}`
- Source: `{summary["source"]}`

## Headline Answers

1. Did reclaim beat rejection? {reclaim_vs_rejection["answer"]}
   - Reclaim: trades={reclaim_trade_count}, median_net={reclaim_median_net:.6f}
   - Rejection: trades={rejection_trade_count}, median_net={rejection_median_net:.6f}
2. Were winners concentrated in early session? {early_answer}
3. Did rising VWAP help? {rising_vwap_answer}
4. Did low VWAP cross count help? {low_cross_answer}
5. Did opening range position help? {opening_range_answer}
6. Did high relative cumulative volume help? {relative_volume_answer}
7. Did near-VWAP chop predict bad trades? {chop_answer}
8. Enough trades after filters? {enough_trades_answer}

## Promising Filters

{filter_lines(promising, "- None cleared the minimum diagnostic bar.")}

## Weak / No Signal Filters

{filter_lines(weak[:8], "- None")}

## Dangerous Filters With Too Few Trades

{filter_lines(dangerous[:8], "- None")}

## Recommended VWAP v2 Parameter Changes

{recommendation_lines}

## Do Not Promote To Candidate Yet

This attribution is diagnostic only. Any VWAP v2 must be re-run through the same
walk-forward, cost-stress, concentration, benchmark, null, and session-flat gates.
"""


def _questions(
    bucket_summary: pd.DataFrame,
    filter_rankings: dict[str, list[dict[str, Any]]],
    mode_comparison: dict[str, Any],
) -> dict[str, Any]:
    reclaim = mode_comparison.get("reclaim", {})
    rejection = mode_comparison.get("rejection", {})
    reclaim_median = float(reclaim.get("median_net_return", 0.0))
    rejection_median = float(rejection.get("median_net_return", 0.0))
    reclaim_mean = float(reclaim.get("mean_net_return", 0.0))
    rejection_mean = float(rejection.get("mean_net_return", 0.0))
    reclaim_profit_factor = float(reclaim.get("profit_factor", 0.0))
    rejection_profit_factor = float(rejection.get("profit_factor", 0.0))
    if (
        reclaim_median > rejection_median
        and reclaim_mean > rejection_mean
        and reclaim_profit_factor > rejection_profit_factor
    ):
        reclaim_answer = "yes"
    elif reclaim_mean > rejection_mean and reclaim_profit_factor > rejection_profit_factor:
        reclaim_answer = "mixed_reclaim_better_on_mean_and_profit_factor"
    elif reclaim_median > rejection_median:
        reclaim_answer = "mixed_reclaim_better_on_median_only"
    else:
        reclaim_answer = "no"
    dangerous = filter_rankings["dangerous_too_few_trades_filters"]
    enough_filters = [
        row
        for group in (
            filter_rankings["promising_filters"],
            filter_rankings["weak_or_no_signal_filters"],
        )
        for row in group
    ]
    chop = _boolean_answer(bucket_summary, "near_vwap_chop_bucket", ">=50%")
    if chop["answer"] in {"yes", "no"}:
        chop["answer"] = "yes" if chop["answer"] == "no" else "no"
    return {
        "did_reclaim_beat_rejection": {
            "answer": reclaim_answer,
            "reclaim_median_net_return": reclaim_median,
            "rejection_median_net_return": rejection_median,
            "reclaim_mean_net_return": reclaim_mean,
            "rejection_mean_net_return": rejection_mean,
            "reclaim_profit_factor": reclaim_profit_factor,
            "rejection_profit_factor": rejection_profit_factor,
        },
        "were_winners_concentrated_early": _boolean_answer(
            bucket_summary,
            "time_of_day_bucket",
            "opening",
        ),
        "did_rising_vwap_help": _boolean_answer(bucket_summary, "vwap_slope_bucket", "rising"),
        "did_low_cross_count_help": _boolean_answer(
            bucket_summary,
            "vwap_cross_count_bucket",
            "1-2",
        ),
        "did_opening_range_position_help": _boolean_answer(
            bucket_summary,
            "opening_range_position_bucket",
            "above_high",
        ),
        "did_high_relative_cumulative_volume_help": _boolean_answer(
            bucket_summary,
            "relative_cumulative_volume_bucket",
            ">=1.2",
        ),
        "did_near_vwap_chop_predict_bad_trades": chop,
        "enough_trades_after_filters": {
            "answer": "yes" if enough_filters else "no",
            "minimum_trade_count": 50,
            "too_few_trade_filter_count": len(dangerous),
        },
    }


def _recommendations(filter_rankings: dict[str, list[dict[str, Any]]]) -> list[str]:
    mapping = {
        "first_second_reclaim_only": (
            "Test a VWAP v2 option limiting reclaim entries to the first or second reclaim."
        ),
        "vwap_cross_count_le_2": "Test a VWAP v2 chop filter requiring VWAP cross count <= 2.",
        "vwap_cross_count_le_3": "Test a VWAP v2 chop filter requiring VWAP cross count <= 3.",
        "avoid_near_vwap_chop_lt_50pct": (
            "Test a VWAP v2 near-VWAP chop filter using the last 30 bars."
        ),
        "rising_vwap_6_bars": "Test a VWAP v2 trend filter requiring 6-bar VWAP slope > 0.",
        "rising_vwap_12_bars": "Test a VWAP v2 trend filter requiring 12-bar VWAP slope > 0.",
        "above_opening_range_mid": (
            "Test a VWAP v2 structure filter requiring close above opening range mid."
        ),
        "above_opening_range_high": (
            "Test a stricter VWAP v2 structure filter requiring close above opening range high."
        ),
        "relative_cumulative_volume_gt_1_1": (
            "Test a VWAP v2 participation filter requiring relative cumulative volume > 1.1."
        ),
        "relative_cumulative_volume_gt_1_2": (
            "Test a stricter participation filter requiring relative cumulative volume > 1.2."
        ),
        "avoid_midday": "Test a VWAP v2 time filter excluding midday entries.",
        "avoid_late_day": "Test a VWAP v2 time filter excluding late-day entries.",
    }
    recommendations: list[str] = []
    for row in filter_rankings["promising_filters"]:
        text = mapping.get(str(row.get("filter")))
        if text and text not in recommendations:
            recommendations.append(text)
    recommendations.append(
        "Keep VWAP rejection parked unless a separate diagnostic shows durable improvement."
    )
    recommendations.append("Do not promote VWAP v2 to candidate without passing robustness gates.")
    return recommendations


def _trade_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "symbol",
        "experiment_id",
        "parameter_set_id",
        "split_id",
        "is_selected_parameter_set",
        "entry_timestamp",
        "exit_timestamp",
        "holding_bars",
        "entry_mode",
        "exit_mode",
        "trade_gross_return",
        "trade_net_return",
        "win_loss",
        "session_date",
        "bar_index_in_session",
        "minutes_from_open",
        "minutes_to_close",
        "entry_close",
        "session_vwap",
        "distance_from_vwap",
        "relative_volume_at_bar_index",
        "relative_cumulative_volume",
        "opening_range_high",
        "opening_range_low",
        "opening_range_mid",
        "opening_range_width_pct",
        "price_above_opening_range_mid",
        "price_above_opening_range_high",
        "vwap_slope_3_bars",
        "vwap_slope_6_bars",
        "vwap_slope_12_bars",
        "close_momentum_3_bars",
        "close_momentum_6_bars",
        "close_momentum_12_bars",
        "vwap_cross_count_so_far",
        "bars_since_last_vwap_cross",
        "vwap_reclaim_count_so_far",
        "near_vwap_pct_30_bars",
        "time_of_day_bucket",
        "session_quality_warning",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    return [*preferred, *extra]


def build_vwap_quality_report(
    *,
    reports_dir: str | Path = "data/reports/research",
    output_dir: str | Path | None = None,
    report_paths: list[str | Path] | None = None,
    universe_path: str | Path | None = None,
    symbols: list[str] | None = None,
    data_dir: str | Path = "data",
    timeframe: str = "5m",
    source: str = "eodhd",
    instrument_type: str = "stock",
    market_calendar: str | None = "XNYS",
) -> VWAPQualityReportResult:
    """Build VWAP quality attribution reports from existing local reports/data."""

    output_path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    paths = find_vwap_report_paths(
        reports_dir=reports_dir,
        timeframe=timeframe,
        source=source,
        universe_path=universe_path,
        symbols=symbols,
        explicit_report_paths=report_paths,
    )
    all_trades: list[dict[str, Any]] = []
    report_metadata: list[dict[str, Any]] = []
    parameter_set_count = 0
    for path in paths:
        trades, metadata = _attribute_report(
            Path(path),
            data_dir=Path(data_dir),
            timeframe=timeframe,
            source=source,
            instrument_type=instrument_type,
            market_calendar=market_calendar,
        )
        all_trades.extend(trades)
        report_metadata.append(metadata)
        parameter_set_count += int(metadata.get("parameter_set_count", 0))

    trades_frame = pd.DataFrame(all_trades)
    bucket_summary = build_bucket_summary(trades_frame)
    symbol_summary = _symbol_summary(trades_frame)
    baseline = (
        _summary_for_rows(trades_frame)
        if not trades_frame.empty
        else _summary_for_rows(pd.DataFrame())
    )
    filter_summaries = _candidate_filter_summaries(trades_frame)
    filter_rankings = _rank_filters(filter_summaries, baseline)
    mode_comparison = _mode_comparison(bucket_summary)
    questions = _questions(bucket_summary, filter_rankings, mode_comparison)
    recommendations = _recommendations(filter_rankings)
    robustness_flags = Counter()
    classifications = Counter()
    for metadata in report_metadata:
        classifications.update([str(metadata.get("classification", ""))])
        flags = metadata.get("robustness_flags", [])
        if isinstance(flags, list):
            robustness_flags.update(str(flag) for flag in flags)

    summary = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_count_analyzed": len([item for item in report_metadata if not item.get("skipped")]),
        "skipped_reports": [item for item in report_metadata if item.get("skipped")],
        "symbol_count": int(trades_frame["symbol"].nunique()) if not trades_frame.empty else 0,
        "trade_count": int(len(trades_frame)),
        "parameter_set_count": parameter_set_count,
        "timeframe": timeframe,
        "source": source,
        "market_calendar": market_calendar,
        "classification_counts": dict(classifications),
        "robustness_flag_counts": dict(robustness_flags),
        "baseline_trade_summary": baseline,
        "entry_mode_comparison": mode_comparison,
        "candidate_filter_summaries": filter_summaries,
        "filter_rankings": filter_rankings,
        "questions": questions,
        "recommended_vwap_v2_parameter_changes": recommendations,
        "report_metadata": report_metadata,
        "conclusion": "Do not promote to candidate yet.",
    }

    summary_json_path = output_path / "summary.json"
    summary_markdown_path = output_path / "summary.md"
    trade_csv_path = output_path / "trade_attribution.csv"
    bucket_csv_path = output_path / "feature_bucket_summary.csv"
    symbol_csv_path = output_path / "symbol_summary.csv"
    summary_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    summary_markdown_path.write_text(_markdown(summary), encoding="utf-8")
    _write_csv(trade_csv_path, all_trades, _trade_fieldnames(all_trades))
    _write_dataframe_csv(bucket_csv_path, bucket_summary)
    _write_dataframe_csv(symbol_csv_path, symbol_summary)
    return VWAPQualityReportResult(
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        trade_attribution_csv_path=trade_csv_path,
        feature_bucket_summary_csv_path=bucket_csv_path,
        symbol_summary_csv_path=symbol_csv_path,
        report_count_analyzed=int(summary["report_count_analyzed"]),
        trade_count=int(summary["trade_count"]),
        parameter_set_count=parameter_set_count,
    )
