#!/usr/bin/env python3
"""Independently audit the repaired EODHD overnight-options quick screen V0.1."""

from __future__ import annotations

# ruff: noqa: E402 -- numerical thread limits must be fixed before imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_data.calendars import get_market_calendar
from stocker_research.eodhd_fixed_options_strategy_v0 import (
    CHECKPOINT,
    MAXIMUM_MONTH_SHARE,
    ZERO_BASED_BAR_ORDINAL,
    assert_no_daily_option_high_low,
    expiry_intrinsic_values,
    option_position_pnl,
    reject_protected_dates,
    session_bootstrap_intervals,
)
from stocker_research.eodhd_fixed_options_strategy_v01 import (
    OVERALL_DECISIONS_V01,
    SAFETY_FLAGS_V01,
    assert_safety_flags_v01,
)

DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "primary"
REQUIRED_ARTIFACTS = (
    "contract.json",
    "repair_manifest.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "exact_date_filter_audit.csv",
    "cached_response_reprocessing.json",
    "late_day_stock_signal_ledger.parquet",
    "direction_mapping_audit.json",
    "required_option_dates.csv",
    "options_download_gap_after_repair.csv",
    "options_download_manifest.json",
    "contract_preselection_manifest.json",
    "selected_contracts.parquet",
    "quote_integrity.csv",
    "strategy_trade_ledger.parquet",
    "strategy_metrics.csv",
    "monthly_metrics.csv",
    "stock_metrics.csv",
    "matched_control_metrics.csv",
    "veto_metrics.csv",
    "bootstrap_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "lightweight_audit.json",
    "determinism_check.json",
    "report.md",
)
FROZEN_COHORT = {
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
}


def canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _calendar_maps() -> tuple[list[str], dict[str, str], dict[str, str]]:
    calendar = get_market_calendar("NYSE")
    schedule = calendar.schedule(start_date="2023-12-15", end_date="2025-09-10")
    dates = [timestamp.date().isoformat() for timestamp in schedule.index]
    previous = {dates[index]: dates[index - 1] for index in range(1, len(dates))}
    following = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
    in_scope = [value for value in dates if "2024-01-01" <= value <= "2025-08-22"]
    return in_scope, previous, following


def _record(
    checks: dict[str, dict[str, object]],
    name: str,
    passed: bool,
    detail: str,
    *,
    applicability: str = "applicable",
) -> None:
    checks[name] = {
        "passed": bool(passed),
        "detail": detail,
        "applicability": applicability,
    }


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _number(value: object) -> float:
    number = float(cast(Any, value))
    if not math.isfinite(number):
        raise ValueError("non-finite option value")
    return number


def _explicit_boolean(value: object, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be an explicit non-null boolean")


def _quote_reason(row: Mapping[str, object], *, require_open_interest: bool) -> str | None:
    try:
        bid = _number(row.get("bid"))
        ask = _number(row.get("ask"))
        midpoint_value = row.get("midpoint")
        midpoint = (
            (bid + ask) / 2.0
            if midpoint_value is None or pd.isna(midpoint_value)
            else _number(midpoint_value)
        )
    except (TypeError, ValueError):
        return "missing_or_nonfinite_quote"
    if bid < 0.0:
        return "bid_invalid"
    if ask < bid:
        return "ask_invalid_or_crossed"
    if midpoint <= 0.0:
        return "midpoint_not_positive"
    if (ask - bid) / midpoint > 0.75:
        return "relative_spread_above_0_75"
    if require_open_interest:
        try:
            if _number(row.get("open_interest")) < 25:
                return "open_interest_below_25"
        except (TypeError, ValueError):
            return "open_interest_below_25"
    try:
        adjusted = _explicit_boolean(row.get("adjusted_contract"), name="adjusted_contract")
        deliverable = _explicit_boolean(
            row.get("deliverable_resolved"), name="deliverable_resolved"
        )
    except ValueError:
        return "contract_safety_metadata_unknown"
    if adjusted:
        return "adjusted_contract"
    if not deliverable:
        return "unresolved_deliverable"
    return None


def _combined_relative_spread(first: Mapping[str, object], second: Mapping[str, object]) -> float:
    first_bid = _number(first.get("bid"))
    first_ask = _number(first.get("ask"))
    second_bid = _number(second.get("bid"))
    second_ask = _number(second.get("ask"))
    first_midpoint = first.get("midpoint")
    second_midpoint = second.get("midpoint")
    first_mid = (
        (first_bid + first_ask) / 2.0
        if first_midpoint is None or pd.isna(first_midpoint)
        else _number(first_midpoint)
    )
    second_mid = (
        (second_bid + second_ask) / 2.0
        if second_midpoint is None or pd.isna(second_midpoint)
        else _number(second_midpoint)
    )
    denominator = first_mid + second_mid
    return (
        math.inf
        if denominator <= 0.0
        else (first_ask - first_bid + second_ask - second_bid) / denominator
    )


def _load_complete_source_options(cache_root: Path) -> pd.DataFrame:
    required = {
        "underlying_symbol",
        "contract_id",
        "option_type",
        "expiration_date",
        "strike",
        "trade_date",
        "bid",
        "ask",
        "midpoint",
        "open_interest",
        "implied_volatility",
        "chain_complete",
    }
    optional = {
        "adjusted_contract",
        "deliverable_resolved",
        "contract_multiplier",
        "settlement_style",
    }
    frames: list[pd.DataFrame] = []
    for path in sorted(cache_root.rglob("*.parquet")):
        names = set(pq.ParquetFile(path).schema.names)
        if not required.issubset(names):
            continue
        trade_date_type = pq.ParquetFile(path).schema_arrow.field("trade_date").type
        if pa.types.is_date(trade_date_type):
            protected_filter_value: object = date(2025, 8, 23)
        elif pa.types.is_string(trade_date_type) or pa.types.is_large_string(trade_date_type):
            protected_filter_value = "2025-08-23"
        elif pa.types.is_timestamp(trade_date_type):
            protected_filter_value = pd.Timestamp("2025-08-23")
        else:
            raise ValueError(f"unsupported option trade_date type: {trade_date_type}")
        columns = sorted(required.union(optional).intersection(names))
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[("trade_date", "<", protected_filter_value)],
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("no complete canonical source-option Parquet is available")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["trade_date"] = combined["trade_date"].map(_as_date)
    combined["expiration_date"] = combined["expiration_date"].map(_as_date)
    if any(combined["trade_date"].ge(date(2025, 8, 23))):
        raise ValueError("protected source option quote loaded")
    combined["chain_complete"] = combined["chain_complete"].map(
        lambda value: _explicit_boolean(value, name="chain_complete")
    )
    combined = combined.loc[combined["chain_complete"]].copy()
    identity = ["contract_id", "trade_date"]
    comparison_columns = [
        "underlying_symbol",
        "option_type",
        "expiration_date",
        "strike",
        "bid",
        "ask",
        "midpoint",
        "open_interest",
        "implied_volatility",
    ]
    for _, group in combined.loc[combined.duplicated(identity, keep=False)].groupby(
        identity, sort=False
    ):
        if any(group[column].nunique(dropna=False) != 1 for column in comparison_columns):
            raise ValueError("conflicting source option duplicate")
    return (
        combined.sort_values(
            ["underlying_symbol", "trade_date", "expiration_date", "strike", "option_type"],
            kind="mergesort",
        )
        .drop_duplicates(identity, keep="first")
        .reset_index(drop=True)
    )


def _complete_strategy_coverage(cache_root: Path) -> set[tuple[str, str, str]]:
    coverage: set[tuple[str, str, str]] = set()
    required = {
        "underlying_symbol",
        "trade_date",
        "request_strategy",
        "chain_complete",
    }
    for path in sorted(cache_root.rglob("*.parquet")):
        parquet_file = pq.ParquetFile(path)
        names = set(parquet_file.schema.names)
        if not required.issubset(names):
            continue
        trade_date_type = parquet_file.schema_arrow.field("trade_date").type
        if pa.types.is_date(trade_date_type):
            protected_filter_value: object = date(2025, 8, 23)
        elif pa.types.is_string(trade_date_type) or pa.types.is_large_string(trade_date_type):
            protected_filter_value = "2025-08-23"
        elif pa.types.is_timestamp(trade_date_type):
            protected_filter_value = pd.Timestamp("2025-08-23")
        else:
            raise ValueError(f"unsupported option trade_date type: {trade_date_type}")
        frame = pd.read_parquet(
            path,
            columns=sorted(required),
            filters=[("trade_date", "<", protected_filter_value)],
        )
        frame["chain_complete"] = frame["chain_complete"].map(
            lambda value: _explicit_boolean(value, name="chain_complete")
        )
        for (symbol, option_date, strategy), group in frame.groupby(
            ["underlying_symbol", "trade_date", "request_strategy"],
            dropna=False,
            sort=False,
        ):
            if str(strategy) in {"S1", "S3"} and group["chain_complete"].all():
                coverage.add((str(symbol), str(option_date)[:10], str(strategy)))
    receipt_path = cache_root / "fixed-overnight-options-v0" / "bounded_download_manifest.json"
    if receipt_path.is_file():
        receipt = _json(receipt_path)
        queries = receipt.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError("bounded download receipt queries are invalid")
        for row in queries:
            if not isinstance(row, dict) or row.get("status") != "complete":
                continue
            option_date = date.fromisoformat(str(row["option_date"]))
            strategy = str(row["strategy"])
            if option_date >= date(2025, 8, 23) or strategy not in {"S1", "S3"}:
                raise ValueError("bounded download receipt escaped the frozen scope")
            coverage.add((str(row["symbol"]), option_date.isoformat(), strategy))
    return coverage


def _provider_timestamp_date(value: object) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/New_York")
    else:
        timestamp = timestamp.tz_convert("America/New_York")
    return timestamp.date()


def _bounded_raw_chronology_evidence(cache_root: Path) -> dict[str, object]:
    """Independently verify exact-date failures from raw response identities."""

    data_dir = (cache_root / "fixed-overnight-options-v0").resolve()
    manifest_dir = data_dir / "manifests" / "completed"
    manifest_paths = sorted(manifest_dir.glob("*.json")) if manifest_dir.is_dir() else []
    raw_records_checked = 0
    mismatch_records = 0
    last_trade_filter_mismatch_records = 0
    protected_observation_rows = 0
    requested_dates: set[str] = set()
    observed_dates: set[str] = set()
    for manifest_path in manifest_paths:
        payload = _json(manifest_path)
        rows = payload.get("manifest_rows", [])
        if not isinstance(rows, list):
            raise ValueError("bounded request manifest rows are invalid")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("bounded request manifest row is invalid")
            requested = date.fromisoformat(str(row["trade_date_from"]))
            cache_path = Path(str(row["cache_path"])).resolve()
            response_hash = str(row["response_hash"])
            if not cache_path.is_relative_to(data_dir) or not cache_path.is_file():
                raise ValueError("bounded raw response path is missing or outside cache")
            if sha256_file(cache_path) != response_hash:
                raise ValueError("bounded raw response hash differs")
            response = _json(cache_path)
            records = response.get("data", [])
            if not isinstance(records, list):
                raise ValueError("bounded raw response data are invalid")
            for item in records:
                if not isinstance(item, dict) or not isinstance(item.get("attributes"), dict):
                    raise ValueError("bounded raw option record is invalid")
                attributes = cast(dict[str, object], item["attributes"])
                resource_id = item.get("id")
                if not isinstance(resource_id, str):
                    raise ValueError("bounded raw option identity is invalid")
                observation = date.fromisoformat(resource_id[-10:])
                bid_date = _provider_timestamp_date(attributes.get("bid_date"))
                ask_date = _provider_timestamp_date(attributes.get("ask_date"))
                trade_date = _provider_timestamp_date(attributes.get("tradetime"))
                raw_records_checked += 1
                protected_observation_rows += int(observation >= date(2025, 8, 23))
                if observation == bid_date == ask_date and observation != requested:
                    mismatch_records += 1
                    last_trade_filter_mismatch_records += int(trade_date == requested)
                    requested_dates.add(requested.isoformat())
                    observed_dates.add(observation.isoformat())
    return {
        "manifest_rows": len(manifest_paths),
        "raw_records_checked": raw_records_checked,
        "accepted_observation_date_mismatch_records": mismatch_records,
        "last_trade_filter_mismatch_records": last_trade_filter_mismatch_records,
        "protected_observation_rows": protected_observation_rows,
        "requested_dates": sorted(requested_dates),
        "observed_dates": sorted(observed_dates),
    }


def _independent_atm_identity(
    chain: pd.DataFrame, selected_row: Mapping[str, object]
) -> tuple[str, str, date, int, float]:
    entry_date = _as_date(selected_row["signal_session"])
    strategy = str(selected_row["strategy"])
    minimum_dte, maximum_dte = (7, 14) if strategy == "S1" else (1, 1)
    working = chain.copy()
    working["entry_dte"] = working["expiration_date"].map(
        lambda expiration: (_as_date(expiration) - entry_date).days
    )
    working["option_type"] = working["option_type"].astype(str).str.casefold()
    working = working.loc[
        working["entry_dte"].between(minimum_dte, maximum_dte)
        & working["option_type"].isin(["call", "put"])
        & pd.to_numeric(working["strike"], errors="raise").gt(0.0)
    ].copy()
    eligible: list[tuple[int, date]] = []
    for expiration, group in working.groupby("expiration_date", sort=True):
        calls = set(group.loc[group["option_type"].eq("call"), "strike"].astype(float))
        puts = set(group.loc[group["option_type"].eq("put"), "strike"].astype(float))
        if calls.intersection(puts):
            eligible.append((int(group["entry_dte"].min()), _as_date(expiration)))
    if not eligible:
        raise ValueError("selected pair has no independently eligible expiry")
    selected_dte, expiration = min(eligible, key=lambda value: (value[0], value[1]))
    expiry = working.loc[working["expiration_date"].eq(expiration)]
    underlying_close = _number(selected_row["previous_close_underlying_price"])
    candidates: list[tuple[tuple[object, ...], Mapping[str, object], Mapping[str, object]]] = []
    calls_by_strike = {
        float(strike): cast(list[dict[str, object]], group.to_dict(orient="records"))
        for strike, group in expiry.loc[expiry["option_type"].eq("call")].groupby("strike")
    }
    puts_by_strike = {
        float(strike): cast(list[dict[str, object]], group.to_dict(orient="records"))
        for strike, group in expiry.loc[expiry["option_type"].eq("put")].groupby("strike")
    }
    for strike in sorted(set(calls_by_strike).intersection(puts_by_strike)):
        for call in calls_by_strike[strike]:
            for put in puts_by_strike[strike]:
                call_oi = (
                    float(call["open_interest"]) if pd.notna(call["open_interest"]) else -math.inf
                )
                put_oi = (
                    float(put["open_interest"]) if pd.notna(put["open_interest"]) else -math.inf
                )
                call_iv = (
                    float(call["implied_volatility"])
                    if pd.notna(call["implied_volatility"])
                    else math.nan
                )
                put_iv = (
                    float(put["implied_volatility"])
                    if pd.notna(put["implied_volatility"])
                    else math.nan
                )
                iv_gap = (
                    abs(call_iv - put_iv)
                    if math.isfinite(call_iv) and math.isfinite(put_iv)
                    else math.inf
                )
                rank: tuple[object, ...] = (
                    abs(math.log(strike / underlying_close)),
                    -min(call_oi, put_oi),
                    _combined_relative_spread(call, put),
                    iv_gap,
                    strike,
                    str(call["contract_id"]),
                    str(put["contract_id"]),
                )
                candidates.append((rank, call, put))
    _, call, put = min(candidates, key=lambda value: value[0])
    if _quote_reason(call, require_open_interest=True) is not None:
        raise ValueError("independently selected call fails selection liquidity")
    if _quote_reason(put, require_open_interest=True) is not None:
        raise ValueError("independently selected put fails selection liquidity")
    return (
        str(call["contract_id"]),
        str(put["contract_id"]),
        expiration,
        selected_dte,
        float(call["strike"]),
    )


def _quote_index(options: pd.DataFrame) -> dict[tuple[str, date], Mapping[str, object]]:
    return {
        (str(item.contract_id), _as_date(item.trade_date)): cast(
            Mapping[str, object], item._asdict()
        )
        for item in options.itertuples(index=False)
    }


_OCC_IDENTITY = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def _standard_occ_identity(
    contract_id: str,
    *,
    symbol: str,
    expiration: date,
    option_type: str,
    strike: float,
) -> bool:
    match = _OCC_IDENTITY.fullmatch(contract_id)
    if match is None:
        return False
    encoded_expiration = date(
        2000 + int(match.group(2)[:2]),
        int(match.group(2)[2:4]),
        int(match.group(2)[4:6]),
    )
    encoded_type = "call" if match.group(3) == "C" else "put"
    encoded_strike = int(match.group(4)) / 1000.0
    return bool(
        match.group(1) == symbol.upper()
        and encoded_expiration == expiration
        and encoded_type == option_type
        and math.isclose(encoded_strike, strike, abs_tol=1e-12)
    )


def _independent_construction_status(
    selected_row: Mapping[str, object],
    quotes: Mapping[tuple[str, date], Mapping[str, object]],
) -> bool:
    entry_date = _as_date(selected_row["signal_session"])
    exit_date = _as_date(selected_row["exit_session"])
    expiration = _as_date(selected_row["expiration_date"])
    strike = _number(selected_row["strike"])
    symbol = str(selected_row["symbol"])
    call_id = str(selected_row["call_contract_id"])
    put_id = str(selected_row["put_contract_id"])
    try:
        source = {
            "call_entry": quotes[(call_id, entry_date)],
            "put_entry": quotes[(put_id, entry_date)],
            "call_exit": quotes[(call_id, exit_date)],
            "put_exit": quotes[(put_id, exit_date)],
        }
    except KeyError:
        return False
    if any(_quote_reason(row, require_open_interest=False) is not None for row in source.values()):
        return False
    if not _standard_occ_identity(
        call_id,
        symbol=symbol,
        expiration=expiration,
        option_type="call",
        strike=strike,
    ) or not _standard_occ_identity(
        put_id,
        symbol=symbol,
        expiration=expiration,
        option_type="put",
        strike=strike,
    ):
        return False
    for row in source.values():
        multiplier = row.get("contract_multiplier")
        if multiplier is not None:
            try:
                if int(cast(Any, multiplier)) != 100:
                    return False
            except (TypeError, ValueError):
                return False
    if str(selected_row["strategy"]) == "S3":
        if expiration != exit_date:
            return False
        for leg in ("call_entry", "put_entry"):
            if str(source[leg].get("settlement_style")) not in {
                "pm",
                "standard_equity_pm",
            }:
                return False
    try:
        _independent_straddle_return(selected_row, quotes)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _independent_straddle_return(
    selected_row: Mapping[str, object],
    quotes: Mapping[tuple[str, date], Mapping[str, object]],
) -> tuple[float, float, dict[str, float]]:
    entry_date = _as_date(selected_row["signal_session"])
    exit_date = _as_date(selected_row["exit_session"])
    call_id = str(selected_row["call_contract_id"])
    put_id = str(selected_row["put_contract_id"])
    source = {
        "call_entry": quotes[(call_id, entry_date)],
        "put_entry": quotes[(put_id, entry_date)],
        "call_exit": quotes[(call_id, exit_date)],
        "put_exit": quotes[(put_id, exit_date)],
    }
    if any(_quote_reason(row, require_open_interest=False) is not None for row in source.values()):
        raise ValueError("trade references an invalid source quote")
    call_ask = _number(source["call_entry"]["ask"])
    put_ask = _number(source["put_entry"]["ask"])
    call_bid = _number(source["call_exit"]["bid"])
    put_bid = _number(source["put_exit"]["bid"])
    entry_debit = call_ask + put_ask
    exit_credit = call_bid + put_bid
    net_pnl = 100.0 * (exit_credit - entry_debit) - 3.0
    initial_cash = 100.0 * entry_debit + 1.5
    return (
        net_pnl,
        net_pnl / initial_cash,
        {
            "call_entry_ask": call_ask,
            "put_entry_ask": put_ask,
            "call_exit_bid": call_bid,
            "put_exit_bid": put_bid,
        },
    )


def _entry_dte_bin(strategy: str, entry_dte: int) -> str:
    if strategy == "S3":
        return "1"
    if entry_dte <= 9:
        return "7-9"
    if entry_dte <= 12:
        return "10-12"
    return "13-14"


def _independent_status(
    trades: pd.DataFrame,
    bootstrap: pd.DataFrame,
    strategy: str,
) -> tuple[str, bool, bool]:
    assessment = trades.loc[trades["strategy"].eq(strategy) & trades["period"].eq("assessment")]
    development = trades.loc[trades["strategy"].eq(strategy) & trades["period"].eq("development")]
    maximum_stock_share = (
        float(assessment["symbol"].value_counts(normalize=True).max())
        if not assessment.empty
        else math.inf
    )
    maximum_month_share = (
        float(assessment["calendar_month"].value_counts(normalize=True).max())
        if not assessment.empty
        else math.inf
    )
    if strategy == "S1":
        support = bool(
            len(assessment) >= 60
            and assessment["session"].nunique() >= 40
            and assessment["symbol"].nunique() >= 10
            and assessment["calendar_month"].nunique() >= 5
            and maximum_stock_share <= 0.20
        )
        statistic = "s1_mean_return_on_debit"
        stock_limit = 0.20
    else:
        support = bool(
            len(assessment) >= 40
            and assessment["session"].nunique() >= 30
            and assessment["symbol"].nunique() >= 8
            and assessment["calendar_month"].nunique() >= 4
            and maximum_stock_share <= 0.25
        )
        statistic = "s3_mean_return_on_debit"
        stock_limit = 0.25
    if not support or assessment.empty or development.empty:
        return "insufficient_support", False, False
    assessment_returns = pd.to_numeric(assessment["return_on_entry_debit"], errors="raise")
    development_returns = pd.to_numeric(development["return_on_entry_debit"], errors="raise")
    assessment_pnl = pd.to_numeric(assessment["net_pnl"], errors="raise").to_numpy(dtype=float)
    positive_pnl = np.sort(assessment_pnl[assessment_pnl > 0.0])[::-1]
    top_count = max(1, int(math.ceil(0.05 * len(assessment_pnl))))
    top_share = (
        float(positive_pnl[:top_count].sum() / positive_pnl.sum())
        if positive_pnl.size
        else math.nan
    )
    bootstrap_row = bootstrap.loc[
        bootstrap["statistic"].eq(statistic)
        & np.isclose(pd.to_numeric(bootstrap["level"], errors="raise"), 0.80)
    ]
    lower = float(bootstrap_row.iloc[0]["lower"]) if not bootstrap_row.empty else math.nan
    matched_coverage = float(assessment["matched"].fillna(False).astype(bool).mean())
    matched_excess = pd.to_numeric(assessment["matched_control_excess"], errors="coerce").mean()
    concentration_passed = bool(
        maximum_stock_share <= stock_limit
        and maximum_month_share <= MAXIMUM_MONTH_SHARE
        and math.isfinite(top_share)
        and top_share < 1.0
    )
    positive = bool(
        assessment_returns.mean() > 0.0
        and assessment_returns.median() >= -0.05
        and np.sign(assessment_returns.mean()) == np.sign(development_returns.mean())
        and assessment.groupby("calendar_month")["return_on_entry_debit"].mean().gt(0.0).sum() >= 4
        and math.isfinite(lower)
        and lower >= 0.0
        and (matched_coverage < 0.70 or (pd.notna(matched_excess) and float(matched_excess) > 0.0))
        and concentration_passed
    )
    return ("supported" if positive else "not_supported"), positive, concentration_passed


def audit(output: Path) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, dict[str, object]] = {}
    missing_artifacts = sorted(name for name in REQUIRED_ARTIFACTS if not (output / name).is_file())
    _record(
        checks,
        "required_artifacts",
        not missing_artifacts,
        f"missing={missing_artifacts}",
    )
    if missing_artifacts:
        return False, {
            **SAFETY_FLAGS_V01,
            "audit_stage": "independent",
            "checks": checks,
            "failures": ["required_artifacts"],
            "passed": False,
        }

    contract = _json(output / "contract.json")
    repair = _json(output / "repair_manifest.json")
    source = _json(output / "source_manifest.json")
    protected = _json(output / "protected_boundary_audit.json")
    cached_reprocessing = _json(output / "cached_response_reprocessing.json")
    download_manifest = _json(output / "options_download_manifest.json")
    direction = _json(output / "direction_mapping_audit.json")
    preselection = _json(output / "contract_preselection_manifest.json")
    decision = _json(output / "decision.json")
    determinism = _json(output / "determinism_check.json")
    signals = pd.read_parquet(output / "late_day_stock_signal_ledger.parquet")
    required = pd.read_csv(output / "required_option_dates.csv")
    exact_date_audit = pd.read_csv(output / "exact_date_filter_audit.csv")
    gaps = pd.read_csv(output / "options_download_gap_after_repair.csv")
    selected = pd.read_parquet(output / "selected_contracts.parquet")
    quote_integrity = pd.read_csv(output / "quote_integrity.csv")
    trades = pd.read_parquet(output / "strategy_trade_ledger.parquet")
    strategy_metrics = pd.read_csv(output / "strategy_metrics.csv")
    matched = pd.read_csv(output / "matched_control_metrics.csv")
    veto = pd.read_csv(output / "veto_metrics.csv")
    bootstrap = pd.read_csv(output / "bootstrap_metrics.csv")
    concentration = pd.read_csv(output / "concentration_metrics.csv")
    safety_passed = True
    try:
        assert_safety_flags_v01(contract)
        assert_safety_flags_v01(cast(Mapping[str, object], contract["safety"]))
        assert_safety_flags_v01(repair)
        assert_safety_flags_v01(decision)
    except (KeyError, ValueError):
        safety_passed = False
    _record(checks, "scientific_and_execution_boundary", safety_passed, "exact frozen flags")

    implementation_paths = {
        "run_screen_v01.py": EXPERIMENT_DIR / "run_screen_v01.py",
        "audit_screen_v01.py": EXPERIMENT_DIR / "audit_screen_v01.py",
        "contract.json": EXPERIMENT_DIR / "contract.json",
        "repair_manifest.json": EXPERIMENT_DIR / "repair_manifest.json",
        "eodhd_fixed_options_strategy_v01.py": (
            REPO_ROOT
            / "packages"
            / "stocker_research"
            / "src"
            / "stocker_research"
            / "eodhd_fixed_options_strategy_v01.py"
        ),
    }
    expected_hashes = cast(dict[str, str], source.get("implementation_sha256", {}))
    implementation_passed = expected_hashes == {
        name: sha256_file(path) for name, path in implementation_paths.items()
    }
    _record(
        checks,
        "implementation_identity",
        implementation_passed,
        "runner, auditor, contract, and reusable module hashes agree",
    )

    in_scope_sessions, previous, following = _calendar_maps()
    expected_rows = len(in_scope_sessions) * len(FROZEN_COHORT)
    population_passed = (
        len(signals) == expected_rows
        and not signals.duplicated(["symbol", "session"]).any()
        and set(signals["symbol"].astype(str)) == FROZEN_COHORT
        and set(signals["session"].astype(str)) == set(in_scope_sessions)
    )
    _record(
        checks,
        "dates_cohort_and_one_row_per_regular_session",
        population_passed,
        f"observed_rows={len(signals)}, expected_rows={expected_rows}",
    )

    protected_passed = True
    try:
        reject_protected_dates(signals, ["session", "contract_selection_date", "exit_session"])
        reject_protected_dates(required, ["signal_session", "option_date"])
    except ValueError:
        protected_passed = False
    protected_passed &= (
        int(source.get("protected_rows_materialised", -1)) == 0
        and int(protected.get("protected_rows_materialised", -1)) == 0
        and int(source.get("protected_market_rows_materialised", -1)) == 0
        and int(source.get("protected_option_observations_materialised", -1)) == 0
        and int(protected.get("protected_market_rows_materialised", -1)) == 0
        and int(protected.get("protected_option_observations_materialised", -1)) == 0
        and int(cached_reprocessing.get("protected_option_observations_materialised", -1)) == 0
        and bool(protected.get("passed", False))
    )
    _record(
        checks,
        "protected_boundary",
        protected_passed,
        "no market or option observation dated 2025-08-23 or later",
    )

    exact_columns = {
        "request_id",
        "symbol",
        "contract_id",
        "requested_observation_date",
        "returned_observation_dates",
        "requested_date_present",
        "exact_date_record_count",
        "discarded_other_date_record_count",
        "discarded_post_boundary_record_count",
        "pagination_complete",
        "response_hash",
        "exact_date_hash",
        "provider_dte_disagreement_record_count",
        "unquotable_exact_date_record_count",
        "status",
    }
    exact_filter_passed = exact_columns.issubset(exact_date_audit.columns)
    valid_exact_statuses = {
        "exact_date_complete",
        "exact_date_absent",
        "extra_dates_discarded",
        "ambiguous_date_mapping",
        "incomplete_pagination",
        "schema_failure",
    }
    if exact_filter_passed:
        exact_filter_passed &= bool(
            not exact_date_audit.duplicated("request_id").any()
            and set(exact_date_audit["status"].astype(str)).issubset(valid_exact_statuses)
        )
        for row in exact_date_audit.itertuples(index=False):
            try:
                returned = cast(list[str], json.loads(str(row.returned_observation_dates)))
                requested = str(row.requested_observation_date)
                present = _explicit_boolean(
                    row.requested_date_present,
                    name="requested_date_present",
                )
                pagination_complete = _explicit_boolean(
                    row.pagination_complete,
                    name="pagination_complete",
                )
                exact_count = int(row.exact_date_record_count)
                discarded_count = int(row.discarded_other_date_record_count)
                post_boundary_count = int(row.discarded_post_boundary_record_count)
                status = str(row.status)
                exact_filter_passed &= bool(
                    present == (requested in returned)
                    and exact_count >= 0
                    and discarded_count >= 0
                    and post_boundary_count >= 0
                    and post_boundary_count <= discarded_count
                    and len(str(row.response_hash)) == 64
                    and len(str(row.exact_date_hash)) == 64
                )
                if status == "exact_date_complete":
                    exact_filter_passed &= present and discarded_count == 0 and pagination_complete
                elif status == "extra_dates_discarded":
                    exact_filter_passed &= present and discarded_count > 0 and pagination_complete
                elif status == "exact_date_absent":
                    exact_filter_passed &= not present and exact_count == 0 and pagination_complete
                elif status == "incomplete_pagination":
                    exact_filter_passed &= not pagination_complete
            except (TypeError, ValueError, json.JSONDecodeError):
                exact_filter_passed = False
    exact_filter_passed &= bool(
        int(exact_date_audit["discarded_other_date_record_count"].sum())
        == int(cached_reprocessing.get("extra_date_records_discarded", -1))
        and int(exact_date_audit["discarded_post_boundary_record_count"].sum())
        == int(cached_reprocessing.get("post_boundary_records_discarded", -1))
        and int(exact_date_audit["provider_dte_disagreement_record_count"].sum())
        == int(cached_reprocessing.get("provider_dte_disagreement_records", -1))
        and int(exact_date_audit["unquotable_exact_date_record_count"].sum())
        == int(cached_reprocessing.get("unquotable_exact_date_records_discarded", -1))
        and int(cached_reprocessing.get("previously_missing_chains_recovered", -1)) >= 1
    )
    _record(
        checks,
        "exact_requested_date_filtering_and_cached_recovery",
        exact_filter_passed,
        (
            f"requests={len(exact_date_audit)}, "
            f"extra_discarded={cached_reprocessing.get('extra_date_records_discarded')}"
        ),
    )
    before_network = cast(
        Mapping[str, object],
        cached_reprocessing.get("before_network", {}),
    )
    network_outcome = cast(
        Mapping[str, object],
        download_manifest.get("network_outcome", {}),
    )
    logical_request_count = int(
        required.groupby(["symbol", "option_date", "strategy"], sort=False).ngroups
    )
    cumulative_download = cast(
        Mapping[str, object],
        download_manifest.get("cumulative_v01_download", {}),
    )
    no_redundant_download_passed = bool(
        int(before_network.get("cached_responses_examined", -1))
        + int(cached_reprocessing.get("initially_unresolved_requests", -1))
        == logical_request_count
        and int(network_outcome.get("requests_considered", -1))
        == int(cached_reprocessing.get("initially_unresolved_requests", -2))
        and source.get("option_download_attempted")
        is bool(int(network_outcome.get("network_requests_made", 0)))
        and source.get("cumulative_option_download_attempted")
        is bool(int(cumulative_download.get("new_logical_requests", 0)))
        and download_manifest.get("api_token_value_recorded") is False
    )
    _record(
        checks,
        "cached_response_recovery_and_no_redundant_download",
        no_redundant_download_passed,
        (
            f"cached_examined={before_network.get('cached_responses_examined')}, "
            f"network_considered={network_outcome.get('requests_considered')}"
        ),
    )

    expiration_boundary_passed = bool(
        protected.get("maximum_materialised_option_observation_date") is None
        or str(protected["maximum_materialised_option_observation_date"]) <= "2025-08-22"
    )
    expiration_boundary_passed &= bool(
        protected.get("maximum_contract_expiration_metadata_date") is None
        or str(protected["maximum_contract_expiration_metadata_date"])
        >= str(protected["maximum_materialised_option_observation_date"])
    )
    _record(
        checks,
        "expiration_metadata_not_observation_date",
        expiration_boundary_passed,
        (
            "post-boundary expiration metadata allowed; observation maximum="
            f"{protected.get('maximum_materialised_option_observation_date')}"
        ),
    )

    available = signals.loc[signals["ordinal_72_structural_available"].astype(bool)].copy()
    local_times = pd.to_datetime(
        available["feature_available_timestamp_utc"], utc=True, errors="raise"
    ).dt.tz_convert("America/New_York")
    checkpoint_passed = (
        available["checkpoint"].eq(CHECKPOINT).all()
        and available["checkpoint_bar_ordinal_zero_based"].eq(ZERO_BASED_BAR_ORDINAL).all()
        and local_times.dt.strftime("%H:%M").eq("15:30").all()
        and pd.to_numeric(available["bars_remaining_before_close"], errors="raise").ge(3).all()
    )
    _record(
        checks,
        "ordinal_72_signal_timing",
        bool(checkpoint_passed),
        f"available_rows={len(available)}",
    )

    thresholds_value = source["structural_manifest"]["frozen_route"]["route_quartiles"]
    thresholds = {
        str(key): tuple(float(item) for item in cast(list[object], values))
        for key, values in cast(dict[str, list[object]], thresholds_value).items()
    }
    independently_labelled = pd.Series("OTHER", index=available.index, dtype="string")
    low_support = available["active_prefix_count"].le(2)
    dominant = available["top_prefix_depth_fraction"].ge(
        thresholds["top_prefix_depth_fraction"][2]
    ) & available["top_minus_second_prefix_depth"].ge(
        thresholds["top_minus_second_prefix_depth"][2]
    )
    narrowing = available["active_prefix_count_change_last_3_bars"].lt(0) & available[
        "depth_margin_change_last_3_bars"
    ].gt(0)
    broad = available["prefix_family_entropy"].ge(
        thresholds["prefix_family_entropy"][2]
    ) & available["top_minus_second_prefix_depth"].le(
        thresholds["top_minus_second_prefix_depth"][0]
    )
    independently_labelled.loc[low_support] = "LOW_ROUTE_SUPPORT"
    independently_labelled.loc[dominant] = "DOMINANT_ROUTE"
    independently_labelled.loc[narrowing] = "NARROWING"
    independently_labelled.loc[broad] = "BROAD_CONFLICT"
    label_mismatches = int(
        independently_labelled.astype(str).ne(available["route_resolution_state"].astype(str)).sum()
    )
    route_passed = label_mismatches == 0
    for label in ("BROAD_CONFLICT", "NARROWING", "DOMINANT_ROUTE", "LOW_ROUTE_SUPPORT"):
        route_passed &= (
            available[label].astype(bool).eq(available["route_resolution_state"].eq(label)).all()
        )
    _record(
        checks,
        "frozen_route_state_reconstruction",
        bool(route_passed),
        f"label_mismatches={label_mismatches}",
    )

    direction_passed = (
        direction.get("audited_orientation_to_price_direction_mapping_available") is False
        and direction.get("audited_mapping") == {}
        and direction.get("directional_strategy_blocker") == "blocked_direction_mapping_unavailable"
        and selected.loc[selected["strategy"].astype(str).str.startswith("S2")].empty
        and trades.loc[trades["strategy"].astype(str).str.startswith("S2")].empty
        and decision.get("s2_directional_spread_status") == "blocked_direction_mapping_unavailable"
        and decision.get("s2_hidden_veto_status") == "blocked_direction_mapping_unavailable"
    )
    _record(
        checks,
        "direction_eligibility_and_mapping_rejection",
        direction_passed,
        "no audited orientation-to-price direction mapping; S2 rejected",
    )

    chronology_passed = not required.empty
    for item in required.itertuples(index=False):
        signal = str(item.signal_session)
        option_date = str(item.option_date)
        if item.role == "contract_preselection":
            chronology_passed &= option_date == previous[signal] and option_date < signal
        elif item.role == "entry_quote":
            chronology_passed &= option_date == signal
        elif item.role == "exit_quote":
            chronology_passed &= option_date == following[signal] and option_date > signal
        else:
            chronology_passed = False
    chronology_passed &= (
        preselection.get("same_session_closing_chain_used_for_identity_selection") is False
        and preselection.get("same_session_greeks_used_for_selection") is False
        and preselection.get("older_chain_forward_filled") is False
        and preselection.get("selected_contract_replacement_at_entry") is False
    )
    _record(
        checks,
        "previous_session_contract_preselection_and_no_same_close_selection",
        bool(chronology_passed),
        f"required_option_date_rows={len(required)}",
    )
    strategy_coverage_passed = True
    try:
        complete_coverage = {
            (
                str(item.symbol),
                str(item.requested_observation_date),
                str(item.strategy),
            )
            for item in exact_date_audit.itertuples(index=False)
            if str(item.status) in {"exact_date_complete", "extra_dates_discarded"}
        }
        required_coverage = {
            (str(item.symbol), str(item.option_date), str(item.strategy))
            for item in required.itertuples(index=False)
        }
        expected_missing = required_coverage.difference(complete_coverage)
        observed_missing = {
            (str(item.symbol), str(item.option_date), str(item.strategies))
            for item in gaps.itertuples(index=False)
        }
        strategy_coverage_passed = expected_missing == observed_missing
    except (KeyError, OSError, TypeError, ValueError):
        strategy_coverage_passed = False
        expected_missing = set()
        observed_missing = set()
    _record(
        checks,
        "strategy_specific_chain_coverage",
        strategy_coverage_passed,
        (f"expected_missing={len(expected_missing)}, observed_missing={len(observed_missing)}"),
    )

    option_high_low_passed = True
    try:
        assert_no_daily_option_high_low(selected)
        assert_no_daily_option_high_low(trades)
    except ValueError:
        option_high_low_passed = False
    _record(
        checks,
        "no_daily_option_high_low",
        option_high_low_passed,
        "analytical option artifacts contain no option high/low fields",
    )

    corporate_passed = (
        source.get("corporate_action_handling", {}).get("ambiguous_boundaries_rejected") is True
        and {
            "inferred_split_on_signal_date",
            "inferred_split_on_exit_date",
            "split_boundary_ambiguous",
        }.issubset(signals.columns)
        and signals["split_boundary_ambiguous"]
        .astype(bool)
        .eq(
            signals["inferred_split_on_signal_date"].astype(bool)
            | signals["inferred_split_on_exit_date"].astype(bool)
        )
        .all()
        and not signals.loc[
            signals["broad_conflict_candidate_pre_iv"].astype(bool),
            "split_boundary_ambiguous",
        ]
        .astype(bool)
        .any()
    )
    _record(
        checks,
        "corporate_action_exclusions",
        corporate_passed,
        "signal- or exit-session split-boundary rows cannot enter option requirements",
    )

    primary_blocker_value = decision.get("primary_blocker")
    primary_blocker = str(primary_blocker_value) if isinstance(primary_blocker_value, str) else None
    construction_blocked = primary_blocker is not None
    if construction_blocked:
        blocker_shape_passed = bool(
            primary_blocker in OVERALL_DECISIONS_V01
            and cast(str, primary_blocker).startswith("blocked_")
            and decision.get("overall_decision") == primary_blocker
        )
        if primary_blocker == "blocked_options_contract_reconstruction_failure":
            blocker_shape_passed &= bool(
                not gaps.empty
                and decision.get("s1_overnight_straddle_status") == "blocked_contract_coverage"
                and decision.get("s3_dte1_straddle_status") == "blocked_contract_coverage"
            )
        elif primary_blocker == "blocked_quick_options_strategy_resource_limit":
            blocker_shape_passed &= bool(
                not gaps.empty
                and decision.get("s1_overnight_straddle_status") == "blocked_resource_limit"
                and decision.get("s3_dte1_straddle_status") == "blocked_resource_limit"
            )
        elif primary_blocker == "blocked_reproducibility_or_audit_failure":
            blocker_shape_passed &= not bool(determinism.get("passed", True))
        _record(
            checks,
            "fail_closed_primary_blocker",
            blocker_shape_passed,
            f"primary_blocker={primary_blocker}",
        )
        _record(
            checks,
            "provider_observation_date_chronology_blocker",
            exact_filter_passed,
            "extra provider dates are audited discards, not a chronology blocker",
            applicability="not_applicable_repaired_exact_date_filter",
        )
        for name in (
            "dte_calculation",
            "atm_strike_selection",
            "delta_spread_leg_selection",
            "entry_exit_bid_ask_sides",
            "contract_multipliers",
            "commissions",
            "commission_sensitivity",
            "expiry_intrinsic_values",
        ):
            _record(
                checks,
                name,
                True,
                (
                    "not applicable: the run failed closed before contract construction "
                    f"with {primary_blocker}"
                ),
                applicability="not_applicable_blocked_before_contract_selection",
            )
        metrics_passed = (
            not strategy_metrics.empty
            and set(strategy_metrics["status"].astype(str)).issubset(
                {
                    "blocked_contract_coverage",
                    "blocked_resource_limit",
                    "blocked_direction_mapping_unavailable",
                }
            )
            and not matched.empty
            and set(matched["status"].astype(str)).issubset(
                {
                    "blocked_contract_coverage",
                    "blocked_resource_limit",
                    "blocked_direction_mapping_unavailable",
                }
            )
            and not veto.empty
            and veto["status"].eq("blocked_direction_mapping_unavailable").all()
        )
        _record(
            checks,
            "matched_controls_and_veto",
            bool(metrics_passed),
            "no matched-control or veto claim emitted without constructed trades",
            applicability="not_applicable_blocked_before_contract_selection",
        )
        bootstrap_passed = bootstrap.empty
        _record(
            checks,
            "whole_session_bootstrap",
            bool(bootstrap_passed),
            "exactly ten draws frozen; no interval inferred from zero trades",
            applicability="not_applicable_blocked_before_contract_selection",
        )
        expected_decision = primary_blocker
        decision_passed = bool(
            decision.get("overall_decision") == expected_decision
            and decision.get("s2_directional_spread_status")
            == "blocked_direction_mapping_unavailable"
            and decision.get("s2_hidden_veto_status") == "blocked_direction_mapping_unavailable"
        )
    else:
        expected_signal_rows = signals.loc[
            signals["ordinal_72_structural_available"].astype(bool)
            & signals["chronology_eligible"].astype(bool)
            & signals["underlying_source_available"].astype(bool)
            & ~signals["split_boundary_ambiguous"].astype(bool)
        ].copy()
        expected_selection_ids = {
            f"{strategy}|{item.symbol}|{item.session}"
            for item in expected_signal_rows.itertuples(index=False)
            for strategy in ("S1", "S3")
        }
        observed_selection_ids = set(selected["selection_id"].astype(str))
        selection_population_passed = bool(
            not selected.duplicated("selection_id").any()
            and expected_selection_ids == observed_selection_ids
        )
        signals_by_key = {
            (str(item.symbol), str(item.session)): cast(Mapping[str, object], item._asdict())
            for item in expected_signal_rows.itertuples(index=False)
        }
        for item in selected.itertuples(index=False):
            try:
                signal = signals_by_key[(str(item.symbol), str(item.signal_session))]
                selection_population_passed &= bool(
                    str(item.selection_id) == f"{item.strategy}|{item.symbol}|{item.signal_session}"
                    and str(item.strategy) in {"S1", "S3"}
                    and str(item.contract_selection_date)
                    == str(signal["contract_selection_date"])
                    == previous[str(item.signal_session)]
                    and str(item.exit_session)
                    == str(signal["exit_session"])
                    == following[str(item.signal_session)]
                    and str(item.period) == str(signal["period"])
                    and str(item.route_resolution_state) == str(signal["route_resolution_state"])
                    and bool(item.broad_conflict) == bool(signal["BROAD_CONFLICT"])
                    and bool(item.recent_registered_completion_prior_6)
                    == bool(signal["recent_registered_completion_prior_6"])
                    and bool(item.hidden_2_3_2_prior_6) == bool(signal["hidden_2_3_2_prior_6"])
                    and bool(item.any_hidden_event_prior_6)
                    == bool(signal["any_hidden_event_prior_6"])
                    and math.isclose(
                        float(item.previous_close_underlying_price),
                        float(signal["previous_close_underlying_price"]),
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        float(item.entry_underlying_close),
                        float(signal["entry_underlying_close"]),
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        float(item.exit_underlying_close),
                        float(signal["exit_underlying_close"]),
                        abs_tol=1e-12,
                    )
                )
            except (KeyError, TypeError, ValueError):
                selection_population_passed = False
        _record(
            checks,
            "selection_population_and_signal_lineage",
            selection_population_passed,
            (
                f"expected_selections={len(expected_selection_ids)}, "
                f"observed_selections={len(observed_selection_ids)}"
            ),
        )
        valid_selected = selected.loc[selected["selection_status"].eq("selected")].copy()
        entry_dates = pd.to_datetime(valid_selected["signal_session"], errors="raise").dt.date
        expirations = pd.to_datetime(valid_selected["expiration_date"], errors="raise").dt.date
        calculated_dte = pd.Series(
            [
                (expiration - entry).days
                for expiration, entry in zip(expirations, entry_dates, strict=True)
            ],
            index=valid_selected.index,
        )
        dte_passed = (
            calculated_dte.eq(
                pd.to_numeric(valid_selected["entry_dte"], errors="raise").astype(int)
            ).all()
            and calculated_dte.loc[valid_selected["strategy"].eq("S1")].between(7, 14).all()
            and calculated_dte.loc[valid_selected["strategy"].eq("S3")].eq(1).all()
        )
        _record(checks, "dte_calculation", bool(dte_passed), "recomputed from entry/expiry dates")
        source_options = pd.DataFrame()
        source_load_error = ""
        try:
            cache_root = Path(str(source["options_cache"]["canonical_cache_path"])).parent
            source_options = _load_complete_source_options(cache_root)
        except (KeyError, OSError, TypeError, ValueError) as error:
            source_load_error = f"{type(error).__name__}:{error}"
        source_load_passed = not source_options.empty
        _record(
            checks,
            "source_option_quote_lineage",
            source_load_passed,
            source_load_error or f"complete_source_option_rows={len(source_options)}",
        )
        atm_passed = source_load_passed
        for item in selected.itertuples(index=False):
            try:
                selection_date = _as_date(item.contract_selection_date)
                source_chain = source_options.loc[
                    source_options["underlying_symbol"].astype(str).eq(str(item.symbol))
                    & source_options["trade_date"].eq(selection_date)
                ]
                call_id, put_id, expiration, entry_dte, strike = _independent_atm_identity(
                    source_chain,
                    cast(Mapping[str, object], item._asdict()),
                )
                atm_passed &= bool(
                    str(item.selection_status) == "selected"
                    and call_id == str(item.call_contract_id)
                    and put_id == str(item.put_contract_id)
                    and expiration == _as_date(item.expiration_date)
                    and entry_dte == int(item.entry_dte)
                    and math.isclose(strike, float(item.strike), abs_tol=1e-12)
                )
            except (KeyError, TypeError, ValueError):
                atm_passed &= str(item.selection_status) == "rejected"
        _record(
            checks,
            "atm_strike_selection",
            atm_passed,
            "nearest common D-1 ATM pair and frozen tie-break independently rebuilt",
        )
        source_quotes = _quote_index(source_options) if source_load_passed else {}
        audited_selected = selected.copy()
        audited_selected["_audit_atm_iv"] = np.nan
        audited_selected["_audit_combined_relative_spread"] = np.nan
        audited_selected["_audit_cheap_iv"] = False
        audited_selected["_audit_iv_quartile"] = pd.Series(
            pd.NA, index=audited_selected.index, dtype="string"
        )
        audited_selected["_audit_constructed"] = False
        state_passed = source_load_passed
        for index, item in audited_selected.loc[
            audited_selected["selection_status"].eq("selected")
        ].iterrows():
            try:
                selection_date = _as_date(item["contract_selection_date"])
                call = source_quotes[(str(item["call_contract_id"]), selection_date)]
                put = source_quotes[(str(item["put_contract_id"]), selection_date)]
                if _quote_reason(call, require_open_interest=True) is not None:
                    raise ValueError("D-1 call state quote is invalid")
                if _quote_reason(put, require_open_interest=True) is not None:
                    raise ValueError("D-1 put state quote is invalid")
                call_midpoint = _number(call["midpoint"])
                put_midpoint = _number(put["midpoint"])
                independently_computed = {
                    "atm_iv": (
                        _number(call["implied_volatility"]) + _number(put["implied_volatility"])
                    )
                    / 2.0,
                    "straddle_mid_pct": (call_midpoint + put_midpoint)
                    / _number(item["previous_close_underlying_price"]),
                    "combined_relative_spread": _combined_relative_spread(call, put),
                    "combined_open_interest": (
                        _number(call["open_interest"]) + _number(put["open_interest"])
                    ),
                }
                for field, expected_value in independently_computed.items():
                    state_passed &= abs(_number(item[field]) - expected_value) <= 1e-12
                audited_selected.loc[index, "_audit_atm_iv"] = independently_computed["atm_iv"]
                audited_selected.loc[index, "_audit_combined_relative_spread"] = (
                    independently_computed["combined_relative_spread"]
                )
            except (KeyError, TypeError, ValueError):
                state_passed = False
        valid_audited = audited_selected["selection_status"].eq("selected")
        for _, indices in (
            audited_selected.loc[valid_audited]
            .groupby(["strategy", "symbol"], sort=True)
            .groups.items()
        ):
            all_index = list(indices)
            development_index = [
                index
                for index in all_index
                if audited_selected.loc[index, "period"] == "development"
            ]
            if not development_index:
                continue
            development_iv = pd.to_numeric(
                audited_selected.loc[development_index, "_audit_atm_iv"],
                errors="raise",
            )
            median_iv = float(development_iv.median())
            quartiles = development_iv.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
            spread_median = float(
                pd.to_numeric(
                    audited_selected.loc[development_index, "_audit_combined_relative_spread"],
                    errors="raise",
                ).median()
            )
            values = pd.to_numeric(
                audited_selected.loc[all_index, "_audit_atm_iv"], errors="raise"
            ).to_numpy(dtype=float)
            expected_cheap = values <= median_iv
            expected_quartiles = np.asarray(("Q1", "Q2", "Q3", "Q4"), dtype=object)[
                np.searchsorted(quartiles, values, side="left")
            ]
            expected_spread_group = np.where(
                pd.to_numeric(
                    audited_selected.loc[all_index, "_audit_combined_relative_spread"],
                    errors="raise",
                ).to_numpy(dtype=float)
                <= spread_median,
                "tight",
                "wide",
            )
            state_passed &= bool(
                np.allclose(
                    pd.to_numeric(
                        audited_selected.loc[all_index, "stock_specific_2024_median_atm_iv"],
                        errors="raise",
                    ).to_numpy(dtype=float),
                    median_iv,
                    rtol=0.0,
                    atol=1e-12,
                )
                and np.array_equal(
                    audited_selected.loc[all_index, "cheap_iv"].astype(bool).to_numpy(),
                    expected_cheap,
                )
                and np.array_equal(
                    audited_selected.loc[all_index, "previous_close_atm_iv_quartile"]
                    .astype(str)
                    .to_numpy(),
                    expected_quartiles.astype(str),
                )
                and np.array_equal(
                    audited_selected.loc[all_index, "spread_group"].astype(str).to_numpy(),
                    expected_spread_group.astype(str),
                )
            )
            audited_selected.loc[all_index, "_audit_cheap_iv"] = expected_cheap
            audited_selected.loc[all_index, "_audit_iv_quartile"] = expected_quartiles
        for index, item in audited_selected.loc[valid_audited].iterrows():
            expected_constructed = bool(
                pd.notna(item["_audit_iv_quartile"])
                and _independent_construction_status(
                    cast(Mapping[str, object], item.to_dict()), source_quotes
                )
            )
            audited_selected.loc[index, "_audit_constructed"] = expected_constructed
            state_passed &= str(item["economics_status"]) == (
                "constructed" if expected_constructed else "rejected"
            )
        expected_trade_ids = set(
            audited_selected.loc[
                audited_selected["_audit_constructed"].astype(bool)
                & audited_selected["_audit_cheap_iv"].astype(bool)
                & audited_selected["broad_conflict"].astype(bool),
                "selection_id",
            ].astype(str)
        )
        actual_trade_ids = set(trades["trade_id"].astype(str))
        complete_trade_population = expected_trade_ids == actual_trade_ids
        _record(
            checks,
            "previous_close_state_and_development_iv_filter",
            state_passed,
            "D-1 state, 2024 stock medians/quartiles, cheap-IV, and construction rebuilt",
        )
        _record(
            checks,
            "eligible_trade_population",
            complete_trade_population,
            (
                f"expected_trade_ids={len(expected_trade_ids)}, "
                f"observed_trade_ids={len(actual_trade_ids)}"
            ),
        )
        _record(
            checks,
            "delta_spread_leg_selection",
            direction_passed,
            "not applicable: S2 was rejected before contract selection by the direction map gate",
            applicability="not_applicable_direction_mapping_unavailable",
        )
        pnl_passed = True
        multiplier_passed = True
        commission_passed = True
        commission_sensitivity_passed = True
        intrinsic_passed = True
        settlement_passed = True
        selected_by_id = {
            str(item.selection_id): cast(Mapping[str, object], item._asdict())
            for item in selected.itertuples(index=False)
        }
        audited_by_id = {
            str(row["selection_id"]): cast(Mapping[str, object], row.to_dict())
            for _index, row in audited_selected.iterrows()
        }
        for item in trades.itertuples(index=False):
            try:
                selected_row = selected_by_id[str(item.trade_id)]
                source_net_pnl, source_return, source_quote_values = _independent_straddle_return(
                    selected_row, source_quotes
                )
                recomputed = option_position_pnl(
                    structure="long_straddle",
                    entry_quotes={
                        "call_ask": float(item.call_entry_ask),
                        "put_ask": float(item.put_entry_ask),
                    },
                    exit_quotes={
                        "call_bid": float(item.call_exit_bid),
                        "put_bid": float(item.put_exit_bid),
                    },
                    multiplier=int(item.contract_multiplier),
                    commission_per_contract_side=float(item.commission_per_contract_side),
                )
                for field in (
                    "entry_debit",
                    "exit_credit",
                    "commissions",
                    "net_pnl",
                    "total_initial_cash_debit",
                    "return_on_entry_debit",
                ):
                    pnl_passed &= abs(float(getattr(item, field)) - recomputed[field]) <= 1e-10
                pnl_passed &= abs(float(item.net_pnl) - source_net_pnl) <= 1e-10
                pnl_passed &= abs(float(item.return_on_entry_debit) - source_return) <= 1e-10
                for field, value in source_quote_values.items():
                    pnl_passed &= abs(float(getattr(item, field)) - value) <= 1e-12
                sensitivity = option_position_pnl(
                    structure="long_straddle",
                    entry_quotes={
                        "call_ask": float(item.call_entry_ask),
                        "put_ask": float(item.put_entry_ask),
                    },
                    exit_quotes={
                        "call_bid": float(item.call_exit_bid),
                        "put_bid": float(item.put_exit_bid),
                    },
                    multiplier=int(item.contract_multiplier),
                    commission_per_contract_side=1.0,
                )
                commission_sensitivity_passed &= bool(
                    abs(float(item.sensitivity_commission_per_contract_side) - 1.0) <= 1e-12
                    and abs(float(item.sensitivity_net_pnl) - sensitivity["net_pnl"]) <= 1e-10
                    and abs(
                        float(item.sensitivity_return_on_entry_debit)
                        - sensitivity["return_on_entry_debit"]
                    )
                    <= 1e-10
                )
                multiplier_passed &= int(item.contract_multiplier) == 100
                commission_passed &= abs(float(item.commissions) - 3.0) <= 1e-10
                if str(item.strategy) == "S3":
                    entry_date = _as_date(item.session)
                    call_source = source_quotes[(str(item.call_contract_id), entry_date)]
                    put_source = source_quotes[(str(item.put_contract_id), entry_date)]
                    settlement_passed &= bool(
                        _as_date(item.expiration_date) == _as_date(selected_row["exit_session"])
                        and str(call_source.get("settlement_style")) in {"pm", "standard_equity_pm"}
                        and str(put_source.get("settlement_style")) in {"pm", "standard_equity_pm"}
                        and not _explicit_boolean(
                            call_source.get("adjusted_contract"),
                            name="call adjusted_contract",
                        )
                        and not _explicit_boolean(
                            put_source.get("adjusted_contract"),
                            name="put adjusted_contract",
                        )
                        and _explicit_boolean(
                            call_source.get("deliverable_resolved"),
                            name="call deliverable_resolved",
                        )
                        and _explicit_boolean(
                            put_source.get("deliverable_resolved"),
                            name="put deliverable_resolved",
                        )
                    )
                    intrinsic = expiry_intrinsic_values(
                        underlying_close=float(
                            signals.loc[
                                signals["row_id"].eq(f"{item.symbol}|{item.session}|{CHECKPOINT}"),
                                "exit_underlying_close",
                            ].iloc[0]
                        ),
                        strike=float(item.strike),
                    )
                    intrinsic_passed &= (
                        abs(float(item.call_intrinsic) - intrinsic["call_intrinsic"]) <= 1e-10
                        and abs(float(item.put_intrinsic) - intrinsic["put_intrinsic"]) <= 1e-10
                    )
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                pnl_passed = False
                settlement_passed = False
                commission_sensitivity_passed = False
        _record(
            checks,
            "entry_exit_bid_ask_sides",
            pnl_passed,
            "recomputed from reloaded source entry asks and exit bids",
        )
        _record(
            checks,
            "contract_multipliers",
            multiplier_passed,
            "all constructed contracts use independently parsed standard multiplier 100",
        )
        _record(
            checks,
            "commissions",
            commission_passed,
            "four contract-sides at $0.75 independently recomputed",
        )
        _record(
            checks,
            "commission_sensitivity",
            commission_sensitivity_passed,
            "four contract-sides at $1.00 independently recomputed",
        )
        _record(
            checks,
            "expiry_intrinsic_values",
            intrinsic_passed and settlement_passed,
            "S3 call/put settlement and secondary intrinsic independently recomputed",
        )
        quote_audit_discrepancies: list[str] = []
        if not source_load_passed:
            quote_audit_discrepancies.append("source_options_unavailable")
        selection_summaries = quote_integrity.loc[
            quote_integrity["scope"].eq("selected_contract")
            & quote_integrity["quote_role"].eq("selection_summary")
        ]
        selected_for_economics = selected.loc[selected["selection_status"].eq("selected")]
        if selection_summaries.duplicated("selection_id").any():
            quote_audit_discrepancies.append("duplicate_selection_summary")
        if set(selection_summaries["selection_id"].astype(str)) != set(
            selected_for_economics["selection_id"].astype(str)
        ):
            quote_audit_discrepancies.append("selection_summary_population")
        for item in selected_for_economics.itertuples(index=False):
            summary = selection_summaries.loc[
                selection_summaries["selection_id"].astype(str).eq(str(item.selection_id))
            ]
            if len(summary) != 1:
                quote_audit_discrepancies.append(
                    f"{item.selection_id}:selection_summary_count:{len(summary)}"
                )
                continue
            expected_constructed = bool(audited_by_id[str(item.selection_id)]["_audit_constructed"])
            if bool(summary.iloc[0]["passed"]) != expected_constructed:
                quote_audit_discrepancies.append(f"{item.selection_id}:selection_summary_outcome")
            leg_rows = quote_integrity.loc[
                quote_integrity["selection_id"].astype(str).eq(str(item.selection_id))
                & quote_integrity["check"].eq("entry_exit_leg_quote")
            ]
            if expected_constructed and (
                len(leg_rows) != 4 or not leg_rows["passed"].astype(bool).all()
            ):
                quote_audit_discrepancies.append(f"{item.selection_id}:constructed_leg_population")
            for quote_row in leg_rows.itertuples(index=False):
                key = (str(quote_row.contract_id), _as_date(quote_row.quote_date))
                source_quote = source_quotes.get(key)
                if source_quote is None:
                    if not (
                        not bool(quote_row.passed)
                        and str(quote_row.rejection_reason) == "missing_quote"
                        and pd.isna(quote_row.bid)
                        and pd.isna(quote_row.ask)
                        and pd.isna(quote_row.midpoint)
                    ):
                        quote_audit_discrepancies.append(
                            f"{item.selection_id}:{quote_row.quote_role}:missing_source_provenance"
                        )
                    continue
                try:
                    values_agree = bool(
                        abs(float(quote_row.bid) - _number(source_quote["bid"])) <= 1e-12
                        and abs(float(quote_row.ask) - _number(source_quote["ask"])) <= 1e-12
                        and abs(float(quote_row.midpoint) - _number(source_quote["midpoint"]))
                        <= 1e-12
                    )
                    expected_quote_passed = (
                        _quote_reason(source_quote, require_open_interest=False) is None
                    )
                    outcome_agrees = bool(quote_row.passed) == expected_quote_passed
                    rejection_documented = bool(quote_row.passed) or bool(
                        str(quote_row.rejection_reason)
                    )
                    if not (values_agree and outcome_agrees and rejection_documented):
                        quote_audit_discrepancies.append(
                            f"{item.selection_id}:{quote_row.quote_role}:source_quote_mismatch"
                        )
                except (KeyError, TypeError, ValueError) as error:
                    quote_audit_discrepancies.append(
                        f"{item.selection_id}:{quote_row.quote_role}:{type(error).__name__}"
                    )
        _record(
            checks,
            "quote_integrity_rejection_provenance",
            not quote_audit_discrepancies,
            (
                "every selected pair has a source-linked construction or rejection summary; "
                f"discrepancies={len(quote_audit_discrepancies)}"
            ),
        )

        matched_discrepancies: list[str] = []
        if not source_load_passed:
            matched_discrepancies.append("source_options_unavailable")
        matching_executed = cast(
            Mapping[str, object],
            decision.get("matched_controls_executed", {}),
        )
        for item in trades.itertuples(index=False):
            try:
                treated = selected_by_id[str(item.trade_id)]
                control_ids = cast(list[str], json.loads(str(item.control_trade_ids)))
                if not bool(matching_executed.get(str(item.strategy), False)):
                    if not bool(
                        control_ids == []
                        and int(item.control_count) == 0
                        and not bool(item.matched)
                        and pd.isna(item.control_mean_return)
                        and pd.isna(item.matched_control_excess)
                    ):
                        matched_discrepancies.append(f"{item.trade_id}:unexpected_unexecuted_match")
                    continue
                pool = audited_selected.loc[
                    audited_selected["strategy"].astype(str).eq(str(item.strategy))
                    & audited_selected["symbol"].astype(str).eq(str(item.symbol))
                    & audited_selected["selection_status"].eq("selected")
                    & audited_selected["_audit_constructed"].astype(bool)
                    & audited_selected["_audit_cheap_iv"].astype(bool)
                    & ~(
                        audited_selected["broad_conflict"].astype(bool)
                        & audited_selected["_audit_cheap_iv"].astype(bool)
                    )
                ].copy()
                if not pool.empty:
                    pool = pool.loc[
                        pool["signal_session"].astype(str).str[:7].eq(str(item.calendar_month))
                        & pool["signal_session"]
                        .map(lambda value: _as_date(value).weekday())
                        .eq(int(item.weekday))
                        & pool.apply(
                            lambda row: _entry_dte_bin(str(row["strategy"]), int(row["entry_dte"])),
                            axis=1,
                        ).eq(str(item.entry_dte_bin))
                        & pool["_audit_iv_quartile"]
                        .astype(str)
                        .eq(str(item.previous_close_atm_iv_quartile))
                    ].copy()
                pool["_different_session"] = (
                    pool["signal_session"].astype(str).ne(str(item.session))
                )
                expected_ids = (
                    pool.sort_values(
                        ["_different_session", "signal_session", "selection_id"],
                        ascending=[False, True, True],
                        kind="mergesort",
                    )
                    .head(5)["selection_id"]
                    .astype(str)
                    .tolist()
                )
                if control_ids != expected_ids:
                    matched_discrepancies.append(f"{item.trade_id}:control_identity")
                if int(item.control_count) != len(expected_ids):
                    matched_discrepancies.append(f"{item.trade_id}:control_count")
                expected_matched = len(expected_ids) >= 3
                if bool(item.matched) != expected_matched:
                    matched_discrepancies.append(f"{item.trade_id}:matched_flag")
                control_returns = [
                    _independent_straddle_return(selected_by_id[control_id], source_quotes)[1]
                    for control_id in control_ids
                ]
                expected_mean = float(np.mean(control_returns)) if expected_matched else math.nan
                if expected_matched:
                    if not (
                        abs(float(item.control_mean_return) - expected_mean) <= 1e-10
                        and abs(
                            float(item.return_on_entry_debit)
                            - expected_mean
                            - float(item.matched_control_excess)
                        )
                        <= 1e-10
                    ):
                        matched_discrepancies.append(f"{item.trade_id}:control_return")
                elif not pd.isna(item.matched_control_excess):
                    matched_discrepancies.append(f"{item.trade_id}:unmatched_excess")
                audited_treated = audited_by_id[str(item.trade_id)]
                if not (
                    bool(treated["broad_conflict"]) and bool(audited_treated["_audit_cheap_iv"])
                ):
                    matched_discrepancies.append(f"{item.trade_id}:treated_eligibility")
            except (KeyError, TypeError, ValueError) as error:
                matched_discrepancies.append(f"{item.trade_id}:{type(error).__name__}")
        veto_passed = bool(
            decision.get("s2_hidden_veto_status") == "blocked_direction_mapping_unavailable"
            and veto["status"].eq("blocked_direction_mapping_unavailable").all()
        )
        if not veto_passed:
            matched_discrepancies.append("s2_veto_blocker")
        _record(
            checks,
            "matched_controls_and_veto",
            not matched_discrepancies,
            (
                "matching keys, no-signal pool, control identities, returns, and S2 veto "
                f"verified; discrepancies={len(matched_discrepancies)}"
            ),
        )
        empty_bootstrap = pd.DataFrame(columns=["statistic", "level", "lower", "upper"])
        support_names = {
            strategy
            for strategy in ("S1", "S3")
            if _independent_status(trades, empty_bootstrap, strategy)[0] != "insufficient_support"
        }
        expected_bootstrap = session_bootstrap_intervals(
            trades.loc[trades["strategy"].isin(support_names)]
        )
        allowed_statistics = {
            statistic
            for strategy in support_names
            for statistic in (
                (
                    "s1_mean_return_on_debit",
                    "s1_matched_control_excess",
                )
                if strategy == "S1"
                else (
                    "s3_mean_return_on_debit",
                    "s3_matched_control_excess",
                )
            )
        }
        expected_bootstrap = expected_bootstrap.loc[
            expected_bootstrap["statistic"].isin(allowed_statistics)
        ]
        comparison = bootstrap.merge(
            expected_bootstrap,
            on=["statistic", "level", "draws", "seed"],
            how="outer",
            suffixes=("_observed", "_expected"),
            indicator=True,
        )
        bootstrap_passed = comparison["_merge"].eq("both").all()
        for field in ("lower", "upper"):
            observed = pd.to_numeric(comparison[f"{field}_observed"], errors="coerce")
            expected = pd.to_numeric(comparison[f"{field}_expected"], errors="coerce")
            bootstrap_passed &= (
                observed.sub(expected).abs().le(1e-12) | (observed.isna() & expected.isna())
            ).all()
        _record(
            checks,
            "whole_session_bootstrap",
            bool(bootstrap_passed),
            "all ten fixed-seed session-bootstrap intervals independently rebuilt",
        )
        s1_status, s1_positive, _ = _independent_status(trades, bootstrap, "S1")
        s3_status, s3_positive, _ = _independent_status(trades, bootstrap, "S3")
        expected_rough = {
            "S1": s1_positive,
            "S2": False,
            "S3": s3_positive,
            "S2_hidden_veto": False,
        }
        positive = {
            name
            for name, value in (("S1", s1_positive), ("S2", False), ("S3", s3_positive))
            if value
        }
        if len(positive) >= 2:
            expected_decision = "multiple_eodhd_options_strategies_show_feasibility"
        elif positive == {"S1"}:
            expected_decision = "overnight_straddle_feasible_only"
        elif positive == {"S2"}:
            expected_decision = "directional_debit_spread_feasible_only"
        elif positive == {"S3"}:
            expected_decision = "dte1_straddle_feasible_only"
        elif s1_status == "insufficient_support" and s3_status == "insufficient_support":
            expected_decision = "all_supported_strategies_insufficient_support"
        elif s1_status in {"supported", "not_supported"} or s3_status in {
            "supported",
            "not_supported",
        }:
            expected_decision = "no_eodhd_options_strategy_feasibility"
        else:
            expected_decision = "descriptive_options_strategy_results_only"
        concentration_passed = True
        for strategy, stock_limit in (("S1", 0.20), ("S3", 0.25)):
            assessment = trades.loc[
                trades["strategy"].eq(strategy) & trades["period"].eq("assessment")
            ]
            expected_gate = False
            maximum_stock_share = math.nan
            maximum_month_share = math.nan
            top_share = math.nan
            if not assessment.empty:
                maximum_stock_share = float(assessment["symbol"].value_counts(normalize=True).max())
                maximum_month_share = float(
                    assessment["calendar_month"].value_counts(normalize=True).max()
                )
                pnl = pd.to_numeric(assessment["net_pnl"], errors="raise").to_numpy(dtype=float)
                positive_pnl = np.sort(pnl[pnl > 0.0])[::-1]
                top_count = max(1, int(math.ceil(0.05 * len(pnl))))
                top_share = (
                    float(positive_pnl[:top_count].sum() / positive_pnl.sum())
                    if positive_pnl.size
                    else math.nan
                )
                expected_gate = bool(
                    maximum_stock_share <= stock_limit
                    and maximum_month_share <= MAXIMUM_MONTH_SHARE
                    and math.isfinite(top_share)
                    and top_share < 1.0
                )
            row = concentration.loc[concentration["strategy"].eq(strategy)]
            concentration_passed &= len(row) == 1
            if len(row) == 1:
                observed = row.iloc[0]
                concentration_passed &= bool(observed["concentration_gate_passed"]) == expected_gate
                for field, expected_value in (
                    ("maximum_stock_share", maximum_stock_share),
                    ("maximum_month_share", maximum_month_share),
                    ("top_5pct_positive_pnl_contribution", top_share),
                ):
                    observed_value = float(observed[field])
                    concentration_passed &= bool(
                        (math.isnan(expected_value) and math.isnan(observed_value))
                        or abs(observed_value - expected_value) <= 1e-12
                    )
        _record(
            checks,
            "concentration_gates",
            concentration_passed,
            "stock, 30% month, and top-5%-P&L gates independently recomputed",
        )
        decision_passed = bool(
            decision.get("overall_decision") == expected_decision
            and decision.get("s1_overnight_straddle_status") == s1_status
            and decision.get("s2_directional_spread_status")
            == "blocked_direction_mapping_unavailable"
            and decision.get("s2_hidden_veto_status") == "blocked_direction_mapping_unavailable"
            and decision.get("s3_dte1_straddle_status") == s3_status
            and cast(dict[str, bool], decision.get("rough_screen_positive", {})) == expected_rough
            and strategy_metrics.loc[strategy_metrics["strategy"].eq("S1"), "status"]
            .eq(s1_status)
            .all()
            and strategy_metrics.loc[strategy_metrics["strategy"].eq("S3"), "status"]
            .eq(s3_status)
            .all()
        )
    _record(
        checks,
        "decision_logic",
        decision_passed,
        f"expected={expected_decision}, observed={decision.get('overall_decision')}",
    )

    determinism_passed = (
        int(determinism.get("exact_date_record_mismatches", -1)) == 0
        and int(determinism.get("selected_contract_mismatches", -1)) == 0
        and int(determinism.get("trade_identity_mismatches", -1)) == 0
        and int(determinism.get("late_day_signal_mismatches", -1)) == 0
        and float(determinism.get("maximum_quote_difference", -1.0)) == 0.0
        and float(determinism.get("maximum_pnl_difference", 1.0)) <= 1e-10
        and bool(determinism.get("passed", False))
    )
    _record(
        checks,
        "determinism",
        determinism_passed,
        "signals, selections, trade identities, quotes, P&L, and decision agree",
    )

    failures = sorted(name for name, value in checks.items() if not bool(value["passed"]))
    audit_result = {
        **SAFETY_FLAGS_V01,
        "audit_stage": "independent",
        "auditor_reused_runner_decision_logic": False,
        "trade_dependent_checks_explained_as_not_applicable": construction_blocked,
        "checks": checks,
        "failures": failures,
        "passed": not failures,
    }
    return not failures, audit_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output = arguments.output.resolve()
    passed, result = audit(output)
    write_json(output / "lightweight_audit.json", result)
    report_path = output / "report.md"
    if passed:
        report = report_path.read_text(encoding="utf-8")
        if "## Independent audit" not in report:
            report += "\n## Independent audit\n\nPassed with no unexplained discrepancy.\n"
            report_path.write_text(report, encoding="utf-8")
            reports_copy = EXPERIMENT_DIR / "reports" / "report.md"
            reports_copy.write_text(report, encoding="utf-8")
        print("independent_audit_passed")
        return 0
    decision_path = output / "decision.json"
    decision = _json(decision_path)
    decision["pre_audit_overall_decision"] = decision.get("overall_decision")
    decision["overall_decision"] = "blocked_reproducibility_or_audit_failure"
    decision["primary_blocker"] = "blocked_reproducibility_or_audit_failure"
    decision["audit_failures"] = result["failures"]
    write_json(decision_path, decision)
    print("blocked_reproducibility_or_audit_failure")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
