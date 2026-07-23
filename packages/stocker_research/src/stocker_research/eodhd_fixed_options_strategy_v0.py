"""Frozen mechanics for the EODHD fixed overnight options quick screen.

The module contains selection, quote, synthetic-P&L, matching, bootstrap, and
decision seams only.  It cannot download data, access a broker, size a
portfolio, or place an order.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd

from stocker_data.calendars import get_market_calendar
from stocker_research.route_competition_hazard_v0 import assign_route_resolution_state

PROTECTED_START: Final[date] = date(2025, 8, 23)
CHECKPOINT: Final[int] = 72
ZERO_BASED_BAR_ORDINAL: Final[int] = 71
MAX_RELATIVE_SPREAD: Final[float] = 0.75
MINIMUM_OPEN_INTEREST: Final[int] = 25
MAXIMUM_DELTA_ERROR: Final[float] = 0.10
MAXIMUM_MONTH_SHARE: Final[float] = 0.30
PRIMARY_COMMISSION: Final[float] = 0.75
SENSITIVITY_COMMISSION: Final[float] = 1.00
BOOTSTRAP_DRAWS: Final[int] = 10
BOOTSTRAP_SEED: Final[int] = 20260723

SAFETY_FLAGS: Final[dict[str, bool | str]] = {
    "research_only": True,
    "retrospective_options_strategy_screen": True,
    "options_data_granularity": "end_of_day",
    "stock_signal_time": "15:30 America/New_York",
    "option_entry_time_proxy": "same_session_end_of_day_quote",
    "option_exit_time_proxy": "future_end_of_day_quote_or_expiry_intrinsic",
    "intraday_option_fill_simulated": False,
    "daily_option_high_low_used": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "prospective_validation": False,
    "achieved_pnl_claimed": False,
}

DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "multiple_eodhd_options_strategies_show_feasibility",
        "overnight_straddle_feasible_only",
        "directional_debit_spread_feasible_only",
        "dte1_straddle_feasible_only",
        "hidden_diversion_veto_improves_directional_spreads",
        "descriptive_options_strategy_results_only",
        "no_eodhd_options_strategy_feasibility",
        "blocked_missing_eodhd_api_token",
        "blocked_insufficient_options_cache",
        "blocked_direction_mapping_unavailable",
        "blocked_options_contract_reconstruction_failure",
        "blocked_options_quote_integrity_failure",
        "blocked_protected_boundary_failure",
        "blocked_chronology_or_leakage_failure",
        "blocked_quick_options_strategy_resource_limit",
        "blocked_reproducibility_or_audit_failure",
    }
)

STRATEGY_STATUSES: Final[frozenset[str]] = frozenset(
    {"supported", "descriptive_only", "not_supported", "insufficient_support", "blocked"}
)


class OptionSelectionError(ValueError):
    """A frozen contract-selection or quote-integrity requirement failed."""


class DirectionMappingUnavailable(OptionSelectionError):
    """No audited orientation-to-price-direction mapping resolves the prefix."""


@dataclass(frozen=True)
class AtmStraddleSelection:
    """Immutable result of one causal ATM call/put preselection."""

    available: bool
    reason: str
    selection_date: date
    entry_date: date
    expiration_date: date | None = None
    entry_dte: int | None = None
    strike: float | None = None
    call_contract_id: str | None = None
    put_contract_id: str | None = None
    call: Mapping[str, Any] | None = None
    put: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DirectionalSpreadSelection:
    """Both directional spreads frozen before the 15:30 stock signal."""

    available: bool
    reason: str
    selection_date: date
    entry_date: date
    expiration_date: date | None = None
    entry_dte: int | None = None
    bullish_long_contract_id: str | None = None
    bullish_short_contract_id: str | None = None
    bearish_long_contract_id: str | None = None
    bearish_short_contract_id: str | None = None
    bullish_long: Mapping[str, Any] | None = None
    bullish_short: Mapping[str, Any] | None = None
    bearish_long: Mapping[str, Any] | None = None
    bearish_short: Mapping[str, Any] | None = None


def assert_safety_flags(value: Mapping[str, object]) -> None:
    """Require the exact non-execution research boundary."""

    observed = {key: value.get(key) for key in SAFETY_FLAGS}
    if observed != SAFETY_FLAGS:
        raise ValueError("research and execution safety flags differ from the frozen contract")


def _date_value(value: object, *, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as error:
        raise ValueError(f"{name} is not an ISO date") from error


def reject_protected_dates(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Fail before analysis when any materialized date reaches the protected boundary."""

    for column in columns:
        if column not in frame:
            continue
        nonmissing = frame[column].dropna()
        if any(_date_value(value, name=column) >= PROTECTED_START for value in nonmissing):
            raise ValueError("protected date 2025-08-23 or later materialised")


def assert_no_daily_option_high_low(frame: pd.DataFrame) -> None:
    """Reject vendor option high/low fields from every analytical surface."""

    forbidden = {
        "high",
        "low",
        "option_high",
        "option_low",
        "daily_high",
        "daily_low",
    }
    present = sorted(forbidden.intersection(str(column).casefold() for column in frame.columns))
    if present:
        raise ValueError(f"daily option high/low fields are prohibited: {present}")


def previous_us_session(entry_date: date) -> date:
    """Return the exact prior NYSE regular session."""

    calendar = get_market_calendar("NYSE")
    sessions = calendar.valid_days(
        start_date=entry_date - timedelta(days=15),
        end_date=entry_date - timedelta(days=1),
    )
    if sessions.empty:
        raise OptionSelectionError("no exact previous US trading session")
    return cast(pd.Timestamp, sessions[-1]).date()


def validate_preselection_date(selection_date: date, entry_date: date) -> None:
    """Prohibit same-close, forward-filled, or non-adjacent contract selection."""

    if selection_date >= entry_date or selection_date != previous_us_session(entry_date):
        raise OptionSelectionError("contract selection date must be the exact previous US session")


def validate_checkpoint_timing(
    *,
    checkpoint: int,
    zero_based_bar_ordinal: int,
    feature_available_timestamp: pd.Timestamp,
    scheduled_close_timestamp: pd.Timestamp,
) -> int:
    """Verify completed-bar count 72 is available at 15:30 ET with a close gap."""

    if checkpoint != CHECKPOINT or zero_based_bar_ordinal != ZERO_BASED_BAR_ORDINAL:
        raise ValueError("stock checkpoint must be completed-bar count 72 / zero-based ordinal 71")
    available = pd.Timestamp(feature_available_timestamp)
    close = pd.Timestamp(scheduled_close_timestamp)
    if available.tzinfo is None or close.tzinfo is None:
        raise ValueError("signal and close timestamps must be timezone-aware")
    local = available.tz_convert("America/New_York")
    if local.strftime("%H:%M") != "15:30":
        raise ValueError("ordinal-72 feature availability must be 15:30 America/New_York")
    seconds = (close.tz_convert("UTC") - available.tz_convert("UTC")).total_seconds()
    bars_left = int(seconds // 300)
    if bars_left < 3:
        raise ValueError("fewer than three complete five-minute bars remain before close")
    return bars_left


def frozen_route_state_labels(
    frame: pd.DataFrame, thresholds: Mapping[str, Sequence[float]]
) -> pd.Series:
    """Apply the repository's frozen labels without redefining their precedence."""

    return assign_route_resolution_state(frame, thresholds)


def resolve_price_direction(
    orientation_id: str, audited_mapping: Mapping[str, str]
) -> Literal["bullish", "bearish"]:
    """Resolve only an explicitly audited one-to-one price-direction mapping."""

    value = audited_mapping.get(orientation_id)
    if value not in {"bullish", "bearish"}:
        raise DirectionMappingUnavailable(
            "audited orientation-to-price-direction mapping is unavailable"
        )
    return cast(Literal["bullish", "bearish"], value)


def directional_signal_eligible(
    row: Mapping[str, object],
    *,
    audited_mapping: Mapping[str, str],
    top_depth_q75: float,
    depth_margin_q75: float,
) -> tuple[bool, str | None]:
    """Apply the five frozen directional eligibility requirements."""

    if str(row.get("route_resolution_state")) not in {"NARROWING", "DOMINANT_ROUTE"}:
        return False, None
    orientation = row.get("top_prefix_orientation")
    if orientation is None:
        return False, None
    direction = resolve_price_direction(str(orientation), audited_mapping)
    pressure = _finite_number(row.get("signed_pressure"), "signed_pressure")
    pressure_agrees = pressure > 0.0 if direction == "bullish" else pressure < 0.0
    if not pressure_agrees:
        return False, direction
    depth = _finite_number(row.get("top_prefix_depth"), "top_prefix_depth")
    margin = _finite_number(row.get("top_minus_second_depth"), "top_minus_second_depth")
    return bool(depth >= top_depth_q75 and margin >= depth_margin_q75), direction


def _finite_number(value: object, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise OptionSelectionError(f"{name} is missing")
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise OptionSelectionError(f"{name} is not numeric") from error
    if not math.isfinite(number):
        raise OptionSelectionError(f"{name} is not finite")
    return number


def _optional_number(value: object) -> float | None:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _explicit_boolean(row: Mapping[str, object], name: str) -> bool | None:
    value = row.get(name)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def relative_spread(row: Mapping[str, object]) -> float:
    """Return quote width divided by midpoint, or infinity when invalid."""

    bid = _optional_number(row.get("bid"))
    ask = _optional_number(row.get("ask"))
    midpoint = _optional_number(row.get("midpoint"))
    if midpoint is None and bid is not None and ask is not None:
        midpoint = (bid + ask) / 2.0
    if bid is None or ask is None or midpoint is None or bid < 0.0 or ask < bid or midpoint <= 0.0:
        return math.inf
    return (ask - bid) / midpoint


def quote_integrity_reason(row: Mapping[str, object], *, require_open_interest: bool) -> str | None:
    """Apply the frozen selection/entry/exit quote integrity requirements."""

    bid = _optional_number(row.get("bid"))
    ask = _optional_number(row.get("ask"))
    midpoint = _optional_number(row.get("midpoint"))
    if midpoint is None and bid is not None and ask is not None:
        midpoint = (bid + ask) / 2.0
    if bid is None or bid < 0.0:
        return "bid_invalid"
    if ask is None or ask < bid:
        return "ask_invalid_or_crossed"
    if midpoint is None or midpoint <= 0.0:
        return "midpoint_not_positive"
    if relative_spread(row) > MAX_RELATIVE_SPREAD:
        return "relative_spread_above_0_75"
    if require_open_interest:
        open_interest = _optional_number(row.get("open_interest"))
        if open_interest is None or open_interest < MINIMUM_OPEN_INTEREST:
            return "open_interest_below_25"
    adjusted = _explicit_boolean(row, "adjusted_contract")
    deliverable_resolved = _explicit_boolean(row, "deliverable_resolved")
    if adjusted is None:
        return "adjusted_contract_metadata_unknown"
    if adjusted:
        return "adjusted_contract"
    if deliverable_resolved is None:
        return "deliverable_metadata_unknown"
    if not deliverable_resolved:
        return "unresolved_deliverable"
    return None


_OCC_CONTRACT = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def standard_contract_multiplier(
    contract_id: str,
    *,
    underlying_symbol: str,
    strike: float,
    adjusted_contract: bool,
    deliverable_resolved: bool,
) -> int:
    """Return 100 only for an unadjusted, internally consistent OCC identity."""

    if adjusted_contract or not deliverable_resolved:
        raise OptionSelectionError("adjusted or unresolved option deliverable")
    match = _OCC_CONTRACT.fullmatch(contract_id)
    if match is None or match.group(1) != underlying_symbol.upper():
        raise OptionSelectionError("nonstandard or underlying-mismatched option contract")
    encoded_strike = int(match.group(4)) / 1000.0
    if not math.isclose(encoded_strike, float(strike), rel_tol=0.0, abs_tol=1e-9):
        raise OptionSelectionError("option contract strike identity mismatch")
    return 100


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def _prepare_chain(
    chain: pd.DataFrame,
    *,
    selection_date: date,
    entry_date: date,
    entry_dte_min: int,
    entry_dte_max: int,
) -> pd.DataFrame:
    validate_preselection_date(selection_date, entry_date)
    if entry_dte_min < 0 or entry_dte_max < entry_dte_min:
        raise ValueError("invalid entry-DTE bounds")
    required = {
        "trade_date",
        "contract_id",
        "expiration_date",
        "option_type",
        "strike",
        "bid",
        "ask",
        "open_interest",
    }
    if missing := sorted(required.difference(chain.columns)):
        raise OptionSelectionError(f"option chain missing columns: {missing}")
    assert_no_daily_option_high_low(chain)
    working = chain.copy()
    working["trade_date"] = working["trade_date"].map(
        lambda value: _date_value(value, name="trade_date")
    )
    working = working.loc[working["trade_date"].eq(selection_date)].copy()
    if working.empty:
        raise OptionSelectionError("exact previous-session option chain is absent")
    working["expiration_date"] = working["expiration_date"].map(
        lambda value: _date_value(value, name="expiration_date")
    )
    working["entry_dte"] = working["expiration_date"].map(lambda expiry: (expiry - entry_date).days)
    working["strike"] = pd.to_numeric(working["strike"], errors="raise")
    working["option_type"] = working["option_type"].astype(str).str.casefold()
    working = working.loc[
        working["entry_dte"].between(entry_dte_min, entry_dte_max, inclusive="both")
        & working["strike"].gt(0.0)
        & working["option_type"].isin(["call", "put"])
    ].copy()
    return working


def _combined_relative_spread(first: Mapping[str, object], second: Mapping[str, object]) -> float:
    first_bid = _optional_number(first.get("bid"))
    first_ask = _optional_number(first.get("ask"))
    second_bid = _optional_number(second.get("bid"))
    second_ask = _optional_number(second.get("ask"))
    first_mid = _optional_number(first.get("midpoint"))
    second_mid = _optional_number(second.get("midpoint"))
    if first_mid is None and first_bid is not None and first_ask is not None:
        first_mid = (first_bid + first_ask) / 2.0
    if second_mid is None and second_bid is not None and second_ask is not None:
        second_mid = (second_bid + second_ask) / 2.0
    values = (first_bid, first_ask, second_bid, second_ask, first_mid, second_mid)
    if any(value is None for value in values):
        return math.inf
    typed = cast(tuple[float, float, float, float, float, float], values)
    denominator = typed[4] + typed[5]
    if typed[1] < typed[0] or typed[3] < typed[2] or denominator <= 0.0:
        return math.inf
    return (typed[1] - typed[0] + typed[3] - typed[2]) / denominator


def select_atm_straddle(
    chain: pd.DataFrame,
    *,
    selection_date: date,
    entry_date: date,
    underlying_close: float,
    entry_dte_min: int,
    entry_dte_max: int,
) -> AtmStraddleSelection:
    """Select one nearest-expiry common-strike pair from D-1 only."""

    if not math.isfinite(underlying_close) or underlying_close <= 0.0:
        raise OptionSelectionError("underlying close must be finite and positive")
    working = _prepare_chain(
        chain,
        selection_date=selection_date,
        entry_date=entry_date,
        entry_dte_min=entry_dte_min,
        entry_dte_max=entry_dte_max,
    )
    eligible_expiries: list[tuple[int, date]] = []
    for expiry, group in working.groupby("expiration_date", sort=True):
        call_strikes = set(group.loc[group["option_type"].eq("call"), "strike"].astype(float))
        put_strikes = set(group.loc[group["option_type"].eq("put"), "strike"].astype(float))
        if call_strikes.intersection(put_strikes):
            eligible_expiries.append((int(group["entry_dte"].min()), cast(date, expiry)))
    if not eligible_expiries:
        return AtmStraddleSelection(
            False, "no_eligible_common_strike_expiry", selection_date, entry_date
        )
    selected_dte, selected_expiry = min(eligible_expiries, key=lambda value: (value[0], value[1]))
    expiry_frame = working.loc[working["expiration_date"].eq(selected_expiry)]
    calls: dict[float, list[dict[str, Any]]] = {}
    puts: dict[float, list[dict[str, Any]]] = {}
    for record in _records(expiry_frame):
        target = calls if record["option_type"] == "call" else puts
        target.setdefault(float(record["strike"]), []).append(record)
    candidates: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
    for strike in sorted(set(calls).intersection(puts)):
        for call in calls[strike]:
            for put in puts[strike]:
                call_oi = _optional_number(call.get("open_interest"))
                put_oi = _optional_number(put.get("open_interest"))
                minimum_oi = min(
                    -math.inf if call_oi is None else call_oi,
                    -math.inf if put_oi is None else put_oi,
                )
                call_iv = _optional_number(call.get("implied_volatility"))
                put_iv = _optional_number(put.get("implied_volatility"))
                iv_gap = math.inf if call_iv is None or put_iv is None else abs(call_iv - put_iv)
                rank = (
                    abs(math.log(strike / underlying_close)),
                    -minimum_oi,
                    _combined_relative_spread(call, put),
                    iv_gap,
                    strike,
                    str(call["contract_id"]),
                    str(put["contract_id"]),
                )
                candidates.append((rank, call, put))
    _rank, call, put = min(candidates, key=lambda candidate: candidate[0])
    strike = float(call["strike"])
    for side, record in (("call", call), ("put", put)):
        if reason := quote_integrity_reason(record, require_open_interest=True):
            return AtmStraddleSelection(
                False,
                f"selected_{side}_{reason}",
                selection_date,
                entry_date,
                selected_expiry,
                selected_dte,
                strike,
                str(call["contract_id"]),
                str(put["contract_id"]),
                call,
                put,
            )
        iv = _optional_number(record.get("implied_volatility"))
        if iv is None or not 0.005 <= iv <= 5.0:
            return AtmStraddleSelection(
                False,
                f"selected_{side}_iv_invalid",
                selection_date,
                entry_date,
                selected_expiry,
                selected_dte,
                strike,
                str(call["contract_id"]),
                str(put["contract_id"]),
                call,
                put,
            )
    return AtmStraddleSelection(
        True,
        "selected",
        selection_date,
        entry_date,
        selected_expiry,
        selected_dte,
        strike,
        str(call["contract_id"]),
        str(put["contract_id"]),
        call,
        put,
    )


def previous_close_option_state(
    selection: AtmStraddleSelection, *, underlying_close: float
) -> dict[str, float]:
    """Calculate the four frozen D-1 ATM state fields."""

    if not selection.available or selection.call is None or selection.put is None:
        raise OptionSelectionError("valid ATM selection is required")
    call = selection.call
    put = selection.put
    call_iv = _finite_number(call.get("implied_volatility"), "call implied volatility")
    put_iv = _finite_number(put.get("implied_volatility"), "put implied volatility")
    call_mid = _finite_number(call.get("midpoint"), "call midpoint")
    put_mid = _finite_number(put.get("midpoint"), "put midpoint")
    call_oi = _finite_number(call.get("open_interest"), "call open interest")
    put_oi = _finite_number(put.get("open_interest"), "put open interest")
    straddle_mid = call_mid + put_mid
    return {
        "atm_iv": (call_iv + put_iv) / 2.0,
        "straddle_mid_pct": straddle_mid / underlying_close,
        "combined_relative_spread": _combined_relative_spread(call, put),
        "combined_open_interest": call_oi + put_oi,
    }


def _spread_candidate(
    frame: pd.DataFrame,
    *,
    option_type: Literal["call", "put"],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    target_long = 0.50
    target_short = 0.25
    records = _records(frame.loc[frame["option_type"].eq(option_type)])
    candidates: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
    for long_leg in records:
        long_delta = _optional_number(long_leg.get("delta"))
        if long_delta is None or abs(abs(long_delta) - target_long) > MAXIMUM_DELTA_ERROR:
            continue
        if quote_integrity_reason(long_leg, require_open_interest=True) is not None:
            continue
        for short_leg in records:
            if long_leg["contract_id"] == short_leg["contract_id"]:
                continue
            long_strike = float(long_leg["strike"])
            short_strike = float(short_leg["strike"])
            strike_valid = (
                short_strike > long_strike if option_type == "call" else short_strike < long_strike
            )
            if not strike_valid:
                continue
            short_delta = _optional_number(short_leg.get("delta"))
            if short_delta is None or abs(abs(short_delta) - target_short) > MAXIMUM_DELTA_ERROR:
                continue
            if quote_integrity_reason(short_leg, require_open_interest=True) is not None:
                continue
            minimum_oi = min(
                _finite_number(long_leg.get("open_interest"), "long open interest"),
                _finite_number(short_leg.get("open_interest"), "short open interest"),
            )
            total_error = abs(abs(long_delta) - target_long) + abs(abs(short_delta) - target_short)
            rank = (
                -minimum_oi,
                _combined_relative_spread(long_leg, short_leg),
                total_error,
                str(long_leg["contract_id"]),
                str(short_leg["contract_id"]),
            )
            candidates.append((rank, long_leg, short_leg))
    if not candidates:
        return None
    _rank, long_leg, short_leg = min(candidates, key=lambda candidate: candidate[0])
    return long_leg, short_leg


def select_directional_debit_spreads(
    chain: pd.DataFrame,
    *,
    selection_date: date,
    entry_date: date,
) -> DirectionalSpreadSelection:
    """Preselect both call and put debit spreads from the exact D-1 chain."""

    working = _prepare_chain(
        chain,
        selection_date=selection_date,
        entry_date=entry_date,
        entry_dte_min=7,
        entry_dte_max=14,
    )
    if working.empty:
        return DirectionalSpreadSelection(
            False, "no_expiry_in_entry_dte_7_14", selection_date, entry_date
        )
    selected_dte = int(working["entry_dte"].min())
    selected_expiry = cast(
        date,
        working.loc[working["entry_dte"].eq(selected_dte), "expiration_date"].min(),
    )
    expiry = working.loc[working["expiration_date"].eq(selected_expiry)].copy()
    bullish = _spread_candidate(expiry, option_type="call")
    bearish = _spread_candidate(expiry, option_type="put")
    if bullish is None or bearish is None:
        reason = "bullish_spread_unavailable" if bullish is None else "bearish_spread_unavailable"
        return DirectionalSpreadSelection(
            False, reason, selection_date, entry_date, selected_expiry, selected_dte
        )
    bullish_long, bullish_short = bullish
    bearish_long, bearish_short = bearish
    return DirectionalSpreadSelection(
        True,
        "selected",
        selection_date,
        entry_date,
        selected_expiry,
        selected_dte,
        str(bullish_long["contract_id"]),
        str(bullish_short["contract_id"]),
        str(bearish_long["contract_id"]),
        str(bearish_short["contract_id"]),
        bullish_long,
        bullish_short,
        bearish_long,
        bearish_short,
    )


def option_position_pnl(
    *,
    structure: Literal["long_straddle", "debit_spread"],
    entry_quotes: Mapping[str, float],
    exit_quotes: Mapping[str, float],
    multiplier: int,
    commission_per_contract_side: float,
) -> dict[str, float]:
    """Apply frozen bid/ask sides and all-in per-contract-side commissions."""

    if multiplier <= 0:
        raise ValueError("contract multiplier must be positive")
    if commission_per_contract_side < 0.0:
        raise ValueError("commission cannot be negative")
    if structure == "long_straddle":
        entry_debit = _finite_number(
            entry_quotes.get("call_ask"), "call entry ask"
        ) + _finite_number(entry_quotes.get("put_ask"), "put entry ask")
        exit_credit = _finite_number(exit_quotes.get("call_bid"), "call exit bid") + _finite_number(
            exit_quotes.get("put_bid"), "put exit bid"
        )
    elif structure == "debit_spread":
        entry_debit = _finite_number(
            entry_quotes.get("long_ask"), "long entry ask"
        ) - _finite_number(entry_quotes.get("short_bid"), "short entry bid")
        exit_credit = _finite_number(exit_quotes.get("long_bid"), "long exit bid") - _finite_number(
            exit_quotes.get("short_ask"), "short exit ask"
        )
    else:
        raise ValueError(f"unsupported frozen structure: {structure}")
    if entry_debit <= 0.0:
        raise OptionSelectionError("entry debit must be finite and positive")
    commissions = 4.0 * commission_per_contract_side
    entry_commissions = 2.0 * commission_per_contract_side
    net_pnl = multiplier * (exit_credit - entry_debit) - commissions
    initial_cash = multiplier * entry_debit + entry_commissions
    return {
        "entry_debit": entry_debit,
        "exit_credit": exit_credit,
        "commissions": commissions,
        "net_pnl": net_pnl,
        "total_initial_cash_debit": initial_cash,
        "return_on_entry_debit": net_pnl / initial_cash,
    }


def expiry_intrinsic_values(*, underlying_close: float, strike: float) -> dict[str, float]:
    """Return the non-primary expiration intrinsic diagnostic."""

    underlying = _finite_number(underlying_close, "expiry underlying close")
    strike_value = _finite_number(strike, "strike")
    return {
        "call_intrinsic": max(underlying - strike_value, 0.0),
        "put_intrinsic": max(strike_value - underlying, 0.0),
    }


def validate_expiration_session(
    *,
    expiration_date: date,
    exit_session: date,
    settlement_style: str | None,
    scheduled_close_timestamp: pd.Timestamp,
    adjusted_contract: bool,
    deliverable_resolved: bool,
) -> Literal["regular_close", "early_close"]:
    """Reject ambiguous settlement while explicitly accepting a scheduled early close."""

    if adjusted_contract or not deliverable_resolved:
        raise OptionSelectionError("adjusted or unresolved expiration deliverable")
    if expiration_date != exit_session:
        raise OptionSelectionError("expiration date is not the exact next-session exit")
    if settlement_style not in {"pm", "standard_equity_pm"}:
        raise OptionSelectionError("ambiguous or nonstandard option settlement")
    close = pd.Timestamp(scheduled_close_timestamp)
    if close.tzinfo is None:
        raise OptionSelectionError("expiration-session close must be timezone-aware")
    local_close = close.tz_convert("America/New_York")
    close_minutes = local_close.hour * 60 + local_close.minute
    if close_minutes == 16 * 60:
        return "regular_close"
    if 13 * 60 <= close_minutes < 16 * 60:
        return "early_close"
    raise OptionSelectionError("unsupported expiration-session market close")


def apply_hidden_232_veto(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude only the frozen recent hidden 2→3→2 family."""

    if "hidden_2_3_2_prior_6" not in frame:
        raise ValueError("hidden_2_3_2_prior_6 is required")
    return frame.loc[~frame["hidden_2_3_2_prior_6"].astype(bool)].copy()


def build_matched_controls(
    candidates: pd.DataFrame, *, treated_trade_ids: Sequence[str]
) -> pd.DataFrame:
    """Construct deterministic, future-return-blind controls, capped at five."""

    required = {
        "trade_id",
        "strategy",
        "symbol",
        "session",
        "calendar_month",
        "weekday",
        "entry_dte_bin",
        "previous_close_atm_iv_quartile",
        "valid_strategy_construction",
        "qualifying_signal",
        "return_on_entry_debit",
    }
    if missing := sorted(required.difference(candidates.columns)):
        raise ValueError(f"matched-control candidates missing columns: {missing}")
    treated_set = set(treated_trade_ids)
    treated = candidates.loc[candidates["trade_id"].astype(str).isin(treated_set)]
    if set(treated["trade_id"].astype(str)) != treated_set:
        raise ValueError("treated trade identity is absent from control candidates")
    match_columns = (
        "strategy",
        "symbol",
        "calendar_month",
        "weekday",
        "entry_dte_bin",
        "previous_close_atm_iv_quartile",
    )
    valid_pool = candidates.loc[
        candidates["valid_strategy_construction"].astype(bool)
        & ~candidates["qualifying_signal"].astype(bool)
    ].copy()
    rows: list[dict[str, object]] = []
    for item in _records(treated.sort_values("trade_id", kind="stable")):
        mask = pd.Series(True, index=valid_pool.index)
        for column in match_columns:
            mask &= valid_pool[column].astype(str).eq(str(item[column]))
        pool = valid_pool.loc[mask].copy()
        pool["_different_session"] = pool["session"].astype(str).ne(str(item["session"]))
        pool = pool.sort_values(
            ["_different_session", "session", "trade_id"],
            ascending=[False, True, True],
            kind="stable",
        ).head(5)
        count = len(pool)
        matched = count >= 3
        control_mean = (
            float(pd.to_numeric(pool["return_on_entry_debit"], errors="raise").mean())
            if matched
            else math.nan
        )
        treated_return = _finite_number(item["return_on_entry_debit"], "treated return")
        rows.append(
            {
                "treated_trade_id": str(item["trade_id"]),
                "strategy": str(item["strategy"]),
                "control_count": count,
                "matched": matched,
                "control_trade_ids": json.dumps(
                    pool["trade_id"].astype(str).tolist(), separators=(",", ":")
                ),
                "control_mean_return": control_mean,
                "treated_return": treated_return,
                "matched_control_excess": treated_return - control_mean if matched else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    weight_values = pd.to_numeric(weights, errors="raise").to_numpy(dtype=float)
    finite = np.isfinite(numeric) & np.isfinite(weight_values) & (weight_values > 0.0)
    if not finite.any():
        return math.nan
    return float(np.average(numeric[finite], weights=weight_values[finite]))


def session_bootstrap_intervals(
    trades: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> pd.DataFrame:
    """Produce intervals from exactly ten complete-session bootstrap draws."""

    if draws != BOOTSTRAP_DRAWS:
        raise ValueError("session bootstrap must use exactly 10 draws")
    required = {"strategy", "session", "return_on_entry_debit", "win"}
    if missing := sorted(required.difference(trades.columns)):
        raise ValueError(f"bootstrap trades missing columns: {missing}")
    sessions = np.asarray(sorted(trades["session"].astype(str).unique()), dtype=object)
    if len(sessions) == 0:
        return pd.DataFrame(columns=["statistic", "level", "lower", "upper", "draws", "seed"])
    rng = np.random.default_rng(seed)
    statistics: dict[str, list[float]] = {
        "s1_mean_return_on_debit": [],
        "s1_matched_control_excess": [],
        "s2_all_mean_return_on_debit": [],
        "s2_veto_mean_return_on_debit": [],
        "s2_veto_minus_all_return_difference": [],
        "s2_veto_minus_all_win_rate_difference": [],
        "s3_mean_return_on_debit": [],
        "s3_matched_control_excess": [],
    }
    for _draw in range(draws):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        multiplicity = {session: int(np.sum(sampled == session)) for session in sessions}
        weights = trades["session"].astype(str).map(multiplicity).astype(float)

        def subset_mean(strategy: str, column: str, draw_weights: pd.Series = weights) -> float:
            mask = trades["strategy"].astype(str).eq(strategy)
            if not bool(mask.any()) or column not in trades:
                return math.nan
            return _weighted_mean(trades.loc[mask, column], draw_weights.loc[mask])

        s2_all_return = subset_mean("S2_ALL", "return_on_entry_debit")
        s2_veto_return = subset_mean("S2_VETO", "return_on_entry_debit")
        s2_all_win = subset_mean("S2_ALL", "win")
        s2_veto_win = subset_mean("S2_VETO", "win")
        statistics["s1_mean_return_on_debit"].append(subset_mean("S1", "return_on_entry_debit"))
        statistics["s1_matched_control_excess"].append(subset_mean("S1", "matched_control_excess"))
        statistics["s2_all_mean_return_on_debit"].append(s2_all_return)
        statistics["s2_veto_mean_return_on_debit"].append(s2_veto_return)
        statistics["s2_veto_minus_all_return_difference"].append(s2_veto_return - s2_all_return)
        statistics["s2_veto_minus_all_win_rate_difference"].append(s2_veto_win - s2_all_win)
        statistics["s3_mean_return_on_debit"].append(subset_mean("S3", "return_on_entry_debit"))
        statistics["s3_matched_control_excess"].append(subset_mean("S3", "matched_control_excess"))
    rows: list[dict[str, object]] = []
    for statistic, values in statistics.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        for level in (0.80, 0.90, 0.95):
            alpha = (1.0 - level) / 2.0
            lower = float(np.quantile(finite, alpha)) if len(finite) else math.nan
            upper = float(np.quantile(finite, 1.0 - alpha)) if len(finite) else math.nan
            rows.append(
                {
                    "statistic": statistic,
                    "level": level,
                    "lower": lower,
                    "upper": upper,
                    "draws": draws,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def strategy_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Calculate the frozen descriptive economics and concentration metrics."""

    required = {
        "trade_id",
        "session",
        "symbol",
        "calendar_month",
        "total_initial_cash_debit",
        "net_pnl",
        "return_on_entry_debit",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"strategy ledger missing metric columns: {missing}")
    if frame.empty:
        return {
            "trades": 0,
            "sessions": 0,
            "stocks": 0,
            "months": 0,
            "total_entry_debit": 0.0,
        }
    pnl = pd.to_numeric(frame["net_pnl"], errors="raise").to_numpy(dtype=float)
    returns = pd.to_numeric(frame["return_on_entry_debit"], errors="raise").to_numpy(dtype=float)
    debit = pd.to_numeric(frame["total_initial_cash_debit"], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(pnl).all() and np.isfinite(returns).all() and np.isfinite(debit).all()):
        raise ValueError("strategy metrics contain non-finite economics")
    ordered = np.sort(pnl)
    trim = int(math.floor(0.10 * len(ordered)))
    trimmed = ordered[trim : len(ordered) - trim] if trim and 2 * trim < len(ordered) else ordered
    positive = np.sort(pnl[pnl > 0.0])[::-1]
    top_count = max(1, int(math.ceil(0.05 * len(pnl))))
    positive_contribution = (
        float(positive[:top_count].sum() / positive.sum()) if positive.size else math.nan
    )
    stock_share = frame["symbol"].astype(str).value_counts(normalize=True)
    month_share = frame["calendar_month"].astype(str).value_counts(normalize=True)
    matched = (
        pd.to_numeric(frame["matched_control_excess"], errors="coerce")
        if "matched_control_excess" in frame
        else pd.Series(dtype=float)
    )
    quantiles = np.quantile(returns, [0.05, 0.25, 0.75, 0.95])
    return {
        "trades": len(frame),
        "sessions": int(frame["session"].astype(str).nunique()),
        "stocks": int(frame["symbol"].astype(str).nunique()),
        "months": int(frame["calendar_month"].astype(str).nunique()),
        "total_entry_debit": float(debit.sum()),
        "mean_net_pnl": float(pnl.mean()),
        "median_net_pnl": float(np.median(pnl)),
        "trimmed_mean_net_pnl": float(trimmed.mean()),
        "mean_return_on_debit": float(returns.mean()),
        "median_return_on_debit": float(np.median(returns)),
        "win_rate": float(np.mean(pnl > 0.0)),
        "full_loss_rate": float(np.mean(returns <= -1.0)),
        "maximum_gain": float(pnl.max()),
        "maximum_loss": float(pnl.min()),
        "return_p05": float(quantiles[0]),
        "return_p25": float(quantiles[1]),
        "return_p75": float(quantiles[2]),
        "return_p95": float(quantiles[3]),
        "matched_control_excess": float(matched.mean()) if matched.notna().any() else math.nan,
        "maximum_stock_share": float(stock_share.max()),
        "maximum_month_share": float(month_share.max()),
        "top_5pct_positive_pnl_contribution": positive_contribution,
    }


def choose_overall_decision(
    strategy_positive: Mapping[str, bool],
    *,
    hidden_veto_positive: bool,
    any_supported: bool,
    blocker: str | None = None,
) -> str:
    """Map frozen evidence to exactly one allowed overall decision."""

    if blocker is not None:
        if blocker not in DECISIONS or not blocker.startswith("blocked_"):
            raise ValueError(f"unknown blocker decision: {blocker}")
        return blocker
    positive = {name for name in ("S1", "S2", "S3") if strategy_positive.get(name, False)}
    if len(positive) >= 2:
        decision = "multiple_eodhd_options_strategies_show_feasibility"
    elif positive == {"S1"}:
        decision = "overnight_straddle_feasible_only"
    elif positive == {"S2"}:
        decision = "directional_debit_spread_feasible_only"
    elif positive == {"S3"}:
        decision = "dte1_straddle_feasible_only"
    elif hidden_veto_positive:
        decision = "hidden_diversion_veto_improves_directional_spreads"
    elif any_supported:
        decision = "no_eodhd_options_strategy_feasibility"
    else:
        decision = "descriptive_options_strategy_results_only"
    if decision not in DECISIONS:
        raise AssertionError("decision mapping escaped the frozen categories")
    return decision


__all__ = [
    "AtmStraddleSelection",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CHECKPOINT",
    "DECISIONS",
    "DirectionMappingUnavailable",
    "DirectionalSpreadSelection",
    "MAXIMUM_DELTA_ERROR",
    "MAXIMUM_MONTH_SHARE",
    "MAX_RELATIVE_SPREAD",
    "MINIMUM_OPEN_INTEREST",
    "OptionSelectionError",
    "PRIMARY_COMMISSION",
    "PROTECTED_START",
    "SAFETY_FLAGS",
    "SENSITIVITY_COMMISSION",
    "STRATEGY_STATUSES",
    "ZERO_BASED_BAR_ORDINAL",
    "apply_hidden_232_veto",
    "assert_no_daily_option_high_low",
    "assert_safety_flags",
    "build_matched_controls",
    "choose_overall_decision",
    "directional_signal_eligible",
    "expiry_intrinsic_values",
    "frozen_route_state_labels",
    "option_position_pnl",
    "previous_close_option_state",
    "previous_us_session",
    "quote_integrity_reason",
    "reject_protected_dates",
    "relative_spread",
    "resolve_price_direction",
    "select_atm_straddle",
    "select_directional_debit_spreads",
    "session_bootstrap_intervals",
    "standard_contract_multiplier",
    "strategy_metrics",
    "validate_expiration_session",
    "validate_checkpoint_timing",
    "validate_preselection_date",
]
