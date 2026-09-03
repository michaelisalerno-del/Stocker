"""Research-only IBKR history and REALIZED_M_20 equivalence test."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as wall_time
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from stocker_core.config import load_ibkr_config
from stocker_core.runs import Environment
from stocker_execution.history import (
    HistorySemantics,
    HistoryStatus,
    IbkrHistoryCache,
    IbkrHistoryService,
)
from stocker_execution.ibkr import HistoricalBar, IbkrConnection

RESEARCH_ONLY = True
ORDER_PLACEMENT = "disabled"
NEW_YORK = ZoneInfo("America/New_York")
LOOKBACK = 20
MINIMUM_VALID = 10
ENDPOINT_OFFSET = 14
PRE_MOVE_THRESHOLD = 0.475764059845861
HISTORY_DURATION = "40 D"
BAR_SIZE = "1 min"
WHAT_TO_SHOW = "TRADES"
USE_RTH = True

SYMBOLS: tuple[tuple[str, str, str, date], ...] = (
    ("WULF", "NASDAQ", "USD", date(2025, 8, 14)),
    ("CRWD", "NASDAQ", "USD", date(2025, 1, 28)),
    ("TSLA", "NASDAQ", "USD", date(2025, 6, 25)),
    ("AAPL", "NASDAQ", "USD", date(2026, 2, 24)),
    ("OKLO", "NYSE", "USD", date(2025, 5, 23)),
)

CASE_SESSIONS = {
    "AAPL": "2026-02-24",
    "CRWD": "2025-01-28",
    "OKLO": "2025-05-23",
    "TSLA": "2025-06-25",
    "WULF": "2025-08-14",
}


@dataclass(frozen=True, slots=True)
class RetrievalMetric:
    symbol: str
    con_id: int
    duration: str
    bar_size: str
    what_to_show: str
    use_rth: bool
    first_load_requests: int
    second_load_requests: int
    bars_received: int
    completed_prior_sessions: int
    oldest_timestamp: str
    newest_timestamp: str
    duplicate_bars: int
    missing_rth_minutes: int
    first_cache_status: str
    second_cache_status: str
    request_failures: int
    retries: int
    retrieval_seconds: float


@dataclass(frozen=True, slots=True)
class RealizedResult:
    valid_prior_sessions: int
    realized_return: float
    realized_price: float
    pre_move_m: float


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _frame_from_bars(bars: tuple[HistoricalBar, ...]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": [item.timestamp for item in bars],
            "open": [item.open for item in bars],
            "high": [item.high for item in bars],
            "low": [item.low for item in bars],
            "close": [item.close for item in bars],
            "volume": [item.volume for item in bars],
        }
    )
    return normalize_prices(frame)


def normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    if not required <= set(frame.columns):
        raise ValueError(f"price input missing columns: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    result = result.drop_duplicates("timestamp", keep="last").sort_values(
        "timestamp", kind="mergesort"
    )
    numeric = result[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="raise"
    )
    if result.empty or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("price input contains missing or non-finite OHLC")
    if not bool(numeric.gt(0).all().all()):
        raise ValueError("price input contains non-positive OHLC")
    result[["open", "high", "low", "close"]] = numeric
    local = result["timestamp"].dt.tz_convert(NEW_YORK)
    result["session_date"] = local.dt.date.astype(str)
    result["minute_of_day"] = local.dt.hour * 60 + local.dt.minute
    return result.set_index("timestamp", drop=False)


def calculate_realized_m_20(
    prices: pd.DataFrame,
    *,
    signal_timestamp: pd.Timestamp,
    current_p0: float,
) -> RealizedResult:
    """Apply the frozen 20-session, same-minute, +14-bar definition exactly."""

    if signal_timestamp.tzinfo is None or signal_timestamp.utcoffset() is None:
        raise ValueError("signal timestamp must be timezone-aware")
    if not math.isfinite(current_p0) or current_p0 <= 0:
        raise ValueError("current P0 must be finite and positive")
    timestamp = signal_timestamp.tz_convert(UTC)
    local = timestamp.tz_convert(NEW_YORK)
    session = str(local.date())
    minute = local.hour * 60 + local.minute
    starts = prices.loc[
        prices["session_date"].lt(session) & prices["minute_of_day"].eq(minute),
        ["session_date", "timestamp", "open"],
    ].copy()
    endpoints = starts["timestamp"] + pd.Timedelta(minutes=ENDPOINT_OFFSET)
    starts["endpoint_close"] = prices["close"].reindex(endpoints).to_numpy()
    starts["endpoint_session"] = prices["session_date"].reindex(endpoints).to_numpy()
    starts = starts.loc[
        starts["endpoint_close"].notna()
        & starts["endpoint_session"].eq(starts["session_date"])
    ]
    starts = (
        starts.sort_index(kind="mergesort")
        .drop_duplicates("session_date", keep="last")
        .tail(LOOKBACK)
    )
    count = len(starts)
    if count < MINIMUM_VALID:
        raise ValueError(f"only {count} valid prior sessions; {MINIMUM_VALID} required")
    returns = np.abs(
        starts["endpoint_close"].to_numpy(float) / starts["open"].to_numpy(float) - 1.0
    )
    realized_return = float(np.median(returns))
    realized_price = current_p0 * realized_return

    if timestamp not in prices.index or timestamp - pd.Timedelta(minutes=3) not in prices.index:
        raise ValueError("T0 or T0-minus-3-minute OPEN is missing")
    source_p0 = float(cast(Any, prices.at[timestamp, "open"]))
    source_prior = float(
        cast(Any, prices.at[timestamp - pd.Timedelta(minutes=3), "open"])
    )
    aligned_prior = source_prior * (current_p0 / source_p0)
    pre_move_m = abs(current_p0 - aligned_prior) / realized_price
    return RealizedResult(count, realized_return, realized_price, pre_move_m)


def _expected_rth_minutes(start: datetime, end: datetime) -> tuple[datetime, ...]:
    calendar = mcal.get_calendar("XNYS")
    schedule = calendar.schedule(
        start_date=(start.astimezone(NEW_YORK).date() - timedelta(days=1)),
        end_date=end.astimezone(NEW_YORK).date(),
    )
    expected: list[datetime] = []
    for row in schedule.itertuples():
        cursor = pd.Timestamp(row.market_open).to_pydatetime().astimezone(UTC)
        close = pd.Timestamp(row.market_close).to_pydatetime().astimezone(UTC)
        while cursor < close:
            if start <= cursor <= end:
                expected.append(cursor)
            cursor += timedelta(minutes=1)
    return tuple(expected)


def _history_end(session: date) -> datetime:
    return datetime.combine(session, wall_time(16, 0), tzinfo=NEW_YORK).astimezone(UTC)


async def fetch_ibkr_history(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_ibkr_config(args.ibkr_config, Environment.PAPER).model_copy(
        update={"client_id": args.client_id}
    )
    connection = IbkrConnection(config)
    cache = IbkrHistoryCache(args.cache)
    service = IbkrHistoryService(connection, cache)
    semantics = HistorySemantics(BAR_SIZE, WHAT_TO_SHOW, USE_RTH)
    metrics: list[RetrievalMetric] = []
    try:
        await connection.connect()
        for symbol, primary_exchange, currency, session in SYMBOLS:
            instrument = await connection.resolve_stock(
                symbol,
                exchange="SMART",
                primary_exchange=primary_exchange,
                currency=currency,
            )
            end = _history_end(session)
            provisional_start = end - timedelta(days=40)
            provisional_required = _expected_rth_minutes(provisional_start, end)
            first = cache.get_required_history(
                instrument, semantics, provisional_required, as_of=end
            )
            before_requests = connection.resource_status().historical_requests_today
            started = time.perf_counter()
            bars = await service.fetch_and_store(
                instrument,
                bar_size=BAR_SIZE,
                duration=HISTORY_DURATION,
                what_to_show=WHAT_TO_SHOW,
                regular_trading_hours=USE_RTH,
                end_time=end,
                minimum_bars=MINIMUM_VALID * 15,
            )
            elapsed = time.perf_counter() - started
            after_requests = connection.resource_status().historical_requests_today
            frame = _frame_from_bars(bars)
            raw_count = len(bars)
            duplicate_count = raw_count - int(frame["timestamp"].nunique())
            oldest = pd.Timestamp(frame.index.min()).to_pydatetime().astimezone(UTC)
            newest = pd.Timestamp(frame.index.max()).to_pydatetime().astimezone(UTC)
            expected = _expected_rth_minutes(oldest, newest)
            expected_snapshot = cache.get_required_history(
                instrument, semantics, expected, as_of=end
            )
            returned_snapshot = cache.get_required_history(
                instrument,
                semantics,
                tuple(pd.Timestamp(item).to_pydatetime() for item in frame.index),
                as_of=end,
            )
            second_before = connection.resource_status().historical_requests_today
            if returned_snapshot.status is not HistoryStatus.READY:
                raise ValueError(f"cached IBKR response is incomplete for {symbol}")
            second_after = connection.resource_status().historical_requests_today
            frame.reset_index(drop=True).to_csv(
                output / f"{symbol}_ibkr_1m.csv", index=False, lineterminator="\n"
            )
            prior_sessions = int(
                frame.loc[frame["session_date"].lt(session.isoformat()), "session_date"].nunique()
            )
            metrics.append(
                RetrievalMetric(
                    symbol=symbol,
                    con_id=instrument.con_id,
                    duration=HISTORY_DURATION,
                    bar_size=BAR_SIZE,
                    what_to_show=WHAT_TO_SHOW,
                    use_rth=USE_RTH,
                    first_load_requests=after_requests - before_requests,
                    second_load_requests=second_after - second_before,
                    bars_received=raw_count,
                    completed_prior_sessions=prior_sessions,
                    oldest_timestamp=oldest.isoformat(),
                    newest_timestamp=newest.isoformat(),
                    duplicate_bars=duplicate_count,
                    missing_rth_minutes=len(expected_snapshot.missing_timestamps),
                    first_cache_status=first.status.value,
                    second_cache_status=returned_snapshot.status.value,
                    request_failures=0,
                    retries=0,
                    retrieval_seconds=elapsed,
                )
            )
    finally:
        connection.disconnect()
    _write_json(output / "retrieval_metrics.json", [asdict(item) for item in metrics])


def _provider_paths(source_root: Path, stocker_local: Path, cohort: str, symbol: str) -> list[Path]:
    directional = source_root / "research/directional-readiness"
    unseen = directional / "20260830-session-hard-unseen-portfolio-pnl-v0"
    retro = directional / "20260830-session-hard-pretouch-overlay-2026-holdout"
    broad = directional / "20260830-session-hard-broad-universe-expansion-v0"
    if cohort.startswith("ORIGINAL20_"):
        paths = [
            stocker_local
            / "data/processed/source=eodhd/instrument_type=stock"
            / f"symbol={symbol}/timeframe=1m/data.parquet"
        ]
    elif cohort == "UNSEEN49":
        paths = [unseen / f"data/unseen_1m_2025/symbol={symbol}/timeframe=1m/data.parquet"]
    elif cohort == "RETROSPECTIVE_2026":
        paths = [
            unseen / f"data/unseen_1m_2025/symbol={symbol}/timeframe=1m/data.parquet",
            retro / f"data/unseen_1m_2026/symbol={symbol}/timeframe=1m/data.parquet",
        ]
    elif cohort == "BROAD2025":
        inventory = pd.read_csv(broad / "data_inventory.csv").set_index("canonical_ticker")
        value = inventory.at[symbol, "one_minute_path"]
        generated_path = (
            broad / f"data/broad_1m_2025/symbol={symbol}/timeframe=1m/data.parquet"
        )
        paths = [generated_path if pd.isna(value) else Path(str(value))]
    else:
        raise ValueError(f"unsupported provider cohort {cohort}")
    present = [path for path in paths if path.is_file()]
    if not present:
        raise FileNotFoundError(f"provider data missing for {cohort}/{symbol}")
    return present


def _load_provider_prices(paths: list[Path]) -> pd.DataFrame:
    return normalize_prices(
        pd.concat(
            [
                pd.read_parquet(
                    path,
                    columns=["timestamp", "open", "high", "low", "close"],
                )
                for path in paths
            ],
            ignore_index=True,
        )
    )


def _summary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    ibkr = frame["ibkr_realized_return"]
    existing = frame["existing_realized_return"]
    ratio = ibkr / existing
    pct = (ibkr - existing).abs() / existing
    threshold = PRE_MOVE_THRESHOLD
    ibkr_q = frame["ibkr_pre_move_m"].gt(threshold)
    existing_q = frame["existing_pre_move_m"].gt(threshold)
    both = ibkr_q & existing_q
    neither = ~ibkr_q & ~existing_q
    union = ibkr_q | existing_q
    return {
        "n_comparisons": len(frame),
        "pearson": float(ibkr.corr(existing, method="pearson")),
        "spearman": float(ibkr.corr(existing, method="spearman")),
        "median_ratio_ibkr_existing": float(ratio.median()),
        "median_absolute_percentage_difference": float(pct.median() * 100),
        "within_5_pct": float(pct.le(0.05).mean() * 100),
        "within_10_pct": float(pct.le(0.10).mean() * 100),
        "within_20_pct": float(pct.le(0.20).mean() * 100),
        "pre_move_pearson": float(
            frame["ibkr_pre_move_m"].corr(frame["existing_pre_move_m"], method="pearson")
        ),
        "pre_move_spearman": float(
            frame["ibkr_pre_move_m"].corr(frame["existing_pre_move_m"], method="spearman")
        ),
        "pre_move_median_absolute_difference": float(
            (frame["ibkr_pre_move_m"] - frame["existing_pre_move_m"]).abs().median()
        ),
        "both_qualify": int(both.sum()),
        "both_reject": int(neither.sum()),
        "ibkr_only_qualify": int((ibkr_q & ~existing_q).sum()),
        "existing_only_qualify": int((~ibkr_q & existing_q).sum()),
        "classification_agreement_pct": float((ibkr_q == existing_q).mean() * 100),
        "jaccard_overlap": float(both.sum() / union.sum()) if union.any() else 1.0,
        "existing_recalculation_max_ledger_price_difference": float(
            frame["existing_recalculation_abs_ledger_difference"].max()
        ),
    }


def _source_difference_metrics(
    ibkr: pd.DataFrame, existing: pd.DataFrame
) -> dict[str, Any]:
    start = pd.Timestamp(ibkr.index.min())
    end = pd.Timestamp(ibkr.index.max())
    provider = existing.loc[(existing.index >= start) & (existing.index <= end)]
    common = ibkr.index.intersection(provider.index)
    ibkr_common = ibkr.loc[common]
    provider_common = provider.loc[common]
    close_ratio = provider_common["close"] / ibkr_common["close"]
    absolute_close_pct = (close_ratio - 1.0).abs()
    return {
        "ibkr_oldest": start.isoformat(),
        "ibkr_newest": end.isoformat(),
        "provider_oldest": pd.Timestamp(existing.index.min()).isoformat(),
        "provider_newest": pd.Timestamp(existing.index.max()).isoformat(),
        "ibkr_bars": len(ibkr),
        "provider_bars_in_ibkr_window": len(provider),
        "common_timestamps": len(common),
        "ibkr_only_timestamps": len(ibkr.index.difference(provider.index)),
        "provider_only_timestamps": len(provider.index.difference(ibkr.index)),
        "median_provider_ibkr_close_ratio": float(close_ratio.median()),
        "median_absolute_close_percentage_difference": float(
            absolute_close_pct.median() * 100
        ),
        "maximum_absolute_close_percentage_difference": float(
            absolute_close_pct.max() * 100
        ),
        "exact_ohlc_rows": int(
            provider_common[["open", "high", "low", "close"]]
            .eq(ibkr_common[["open", "high", "low", "close"]])
            .all(axis=1)
            .sum()
        ),
    }


def compare(args: argparse.Namespace) -> None:
    ibkr_dir = Path(args.ibkr_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(args.ledger)
    selected = pd.concat(
        [
            ledger.loc[
                ledger["stock"].eq(symbol)
                & ledger["session"].astype(str).eq(session)
                & (
                    ledger["cohort"].eq("UNSEEN49")
                    if symbol in {"OKLO", "TSLA"}
                    else pd.Series(True, index=ledger.index)
                )
            ]
            for symbol, session in CASE_SESSIONS.items()
        ],
        ignore_index=True,
    ).drop_duplicates(["stock", "signal_timestamp"], keep="first")
    rows: list[dict[str, Any]] = []
    source_root = Path(args.source_root)
    stocker_local = Path(args.stocker_local)
    provider_cache: dict[tuple[str, str], pd.DataFrame] = {}
    ibkr_cache: dict[str, pd.DataFrame] = {}
    for case in selected.itertuples(index=False):
        symbol = str(case.stock)
        cohort = str(case.cohort)
        if symbol not in ibkr_cache:
            ibkr_cache[symbol] = normalize_prices(
                pd.read_csv(ibkr_dir / f"{symbol}_ibkr_1m.csv")
            )
        key = (cohort, symbol)
        if key not in provider_cache:
            provider_cache[key] = _load_provider_prices(
                _provider_paths(source_root, stocker_local, cohort, symbol)
            )
        timestamp = pd.Timestamp(cast(Any, case.signal_timestamp))
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(UTC)
        p0 = float(cast(Any, case.P0))
        ibkr = calculate_realized_m_20(
            ibkr_cache[symbol], signal_timestamp=timestamp, current_p0=p0
        )
        existing = calculate_realized_m_20(
            provider_cache[key], signal_timestamp=timestamp, current_p0=p0
        )
        ledger_price = float(cast(Any, case.REALIZED_M_20))
        rows.append(
            {
                "row_id": case.row_id,
                "cohort": cohort,
                "symbol": symbol,
                "session": case.session,
                "t0": timestamp.isoformat(),
                "current_p0": p0,
                "ibkr_valid_prior_sessions": ibkr.valid_prior_sessions,
                "ibkr_realized_return": ibkr.realized_return,
                "ibkr_realized_price": ibkr.realized_price,
                "existing_valid_prior_sessions": existing.valid_prior_sessions,
                "existing_realized_return": existing.realized_return,
                "existing_realized_price": existing.realized_price,
                "existing_ledger_price": ledger_price,
                "existing_recalculation_abs_ledger_difference": abs(
                    existing.realized_price - ledger_price
                ),
                "ibkr_pre_move_m": ibkr.pre_move_m,
                "existing_pre_move_m": existing.pre_move_m,
                "ibkr_qualifies": ibkr.pre_move_m > PRE_MOVE_THRESHOLD,
                "existing_qualifies": existing.pre_move_m > PRE_MOVE_THRESHOLD,
            }
        )
    comparison = pd.DataFrame(rows).sort_values(["symbol", "t0"], kind="mergesort")
    comparison.to_csv(output / "comparisons.csv", index=False, lineterminator="\n")
    _write_json(output / "equivalence_summary.json", _summary_metrics(comparison))
    source_differences = {
        f"{cohort}/{symbol}": _source_difference_metrics(
            ibkr_cache[symbol], provider_prices
        )
        for (cohort, symbol), provider_prices in sorted(provider_cache.items())
    }
    _write_json(output / "source_differences.json", source_differences)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--ibkr-config", required=True)
    fetch.add_argument("--cache", required=True)
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--client-id", type=int, default=9200)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--ibkr-dir", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--ledger", required=True)
    compare_parser.add_argument("--source-root", required=True)
    compare_parser.add_argument("--stocker-local", required=True)
    return value


def main() -> None:
    if not RESEARCH_ONLY or ORDER_PLACEMENT != "disabled":
        raise RuntimeError("research/order lock failed")
    args = parser().parse_args()
    if args.command == "fetch":
        asyncio.run(fetch_ibkr_history(args))
    else:
        compare(args)


if __name__ == "__main__":
    main()
