"""Listed Saxo identities, actual sessions and conservative executable option costs."""

import math
import re
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from typing import Any

from stocker_execution.config import MARKETS, Environment

MULTIPLIERS = {"CL": 1000, "GC": 100, "NG": 10000, "NQ": 20, "SI": 5000}


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("TIMESTAMP_TIMEZONE_MISSING")
    return result.astimezone(UTC)


def positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("MISSING_" + name) from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError("INVALID_" + name)
    return number


def future_identity(market: str, environment: Environment, raw: dict[str, Any]) -> dict[str, Any]:
    if market not in MARKETS or raw.get("AssetType") != "ContractFutures":
        raise ValueError("ONLY_APPROVED_LISTED_FUTURES")
    symbol = str(raw.get("Symbol", ""))
    match = re.fullmatch(market + r"([FGHJKMNQUVXZ])([0-9]{1,4}):[A-Za-z0-9_-]+", symbol)
    if not match:
        raise ValueError("STANDARD_FUTURE_FAMILY_NOT_VERIFIED")
    multiplier = positive(raw.get("ContractSize"), "CONTRACT_MULTIPLIER")
    if multiplier != MULTIPLIERS[market]:
        raise ValueError("STANDARD_CONTRACT_SIZE_MISMATCH")
    expiry = str(raw.get("ExpiryDate", ""))
    if not re.match(r"^\d{4}-\d{2}-\d{2}", expiry):
        raise ValueError("ACTUAL_EXPIRY_MISSING")
    expiry_year = datetime.fromisoformat(expiry.replace("Z", "+00:00")).year
    years = [y for y in range(expiry_year - 1, expiry_year + 2) if str(y).endswith(match[2])]
    if len(years) != 1:
        raise ValueError("CONTRACT_MONTH_YEAR_AMBIGUOUS")
    contract_month = f"{years[0]}-{'FGHJKMNQUVXZ'.index(match[1]) + 1:02d}"
    if (
        not isinstance(raw.get("Uic"), int)
        or raw["Uic"] <= 0
        or not raw.get("CurrencyCode")
        or not (raw.get("Exchange") or {}).get("ExchangeId")
    ):
        raise ValueError("REFERENCE_IDENTITY_INCOMPLETE")
    tick = positive(raw.get("TickSize"), "TICK_SIZE")
    factor = positive(raw.get("PriceToContractFactor"), "PRICE_TO_CONTRACT_FACTOR")
    return {
        "provider": "SAXO",
        "environment": environment,
        "asset_type": "ContractFutures",
        "uic": int(positive(raw.get("Uic"), "UIC")),
        "market": market,
        "symbol": symbol,
        "exchange": raw.get("Exchange", {}).get("ExchangeId"),
        "contract_month": contract_month,
        "expiry": expiry,
        "currency": raw.get("CurrencyCode"),
        "tick_size": tick,
        "tick_value": tick * factor,
        "multiplier": multiplier,
        "price_factor": factor,
    }


def key(identity: dict[str, Any]) -> str:
    return f"SAXO:{identity['environment']}:{identity['asset_type']}:{identity['uic']}"


def session_state(raw: dict[str, Any], at: datetime) -> str:
    for session in (raw.get("TradingSessions") or {}).get("Sessions", []):
        try:
            if utc(session["StartTime"]) <= at < utc(session["EndTime"]):
                return str(session.get("State", "UNKNOWN")).upper()
        except (KeyError, ValueError):
            continue
    return "SESSION_UNVERIFIED"


def option_identity(
    future: dict[str, Any],
    raw: dict[str, Any],
    root: int,
    space: dict[str, Any],
) -> dict[str, Any]:
    if (
        raw.get("AssetType") != "FuturesOption"
        or raw.get("UnderlyingAssetType") != "ContractFutures"
    ):
        raise ValueError("ONLY_ACTUAL_FUTURES_OPTIONS")
    if space.get("UnderlyingUic") != future["uic"] or raw.get("Uic") != space.get("Uic"):
        raise ValueError("OPTION_UNDERLYING_RELATIONSHIP_MISMATCH")
    right = raw.get("PutCall")
    if right not in {"Call", "Put"} or right != space.get("PutCall"):
        raise ValueError("OPTION_RIGHT_MISMATCH")
    if raw.get("StrikePrice") != space.get("StrikePrice"):
        raise ValueError("OPTION_STRIKE_MISMATCH")
    return {
        "provider": "SAXO",
        "environment": future["environment"],
        "market": future["market"],
        "asset_type": "FuturesOption",
        "uic": raw["Uic"],
        "underlying_uic": future["uic"],
        "option_root_id": root,
        "symbol": raw.get("Symbol"),
        "right": right,
        "strike": positive(raw.get("StrikePrice"), "STRIKE"),
        "expiry": raw.get("ExpiryDate"),
        "currency": raw.get("CurrencyCode"),
        "exchange": raw.get("Exchange", {}).get("ExchangeId"),
        "contract_month": future["contract_month"],
        "multiplier": positive(raw.get("ContractSize"), "MULTIPLIER"),
        "price_factor": positive(raw.get("PriceToContractFactor"), "PRICE_FACTOR"),
        "tick_size": positive(raw.get("TickSizeLimitOrder", raw.get("TickSize")), "TICK_SIZE"),
        "minimum_quantity": positive(raw.get("MinimumTradeSize"), "MINIMUM_QUANTITY"),
        "lot_size": positive(raw.get("LotSize"), "LOT_SIZE"),
        "amount_decimals": raw.get("AmountDecimals"),
        "exercise_cutoff": raw.get("ExerciseCutOffTime"),
        "trading_sessions": raw.get("TradingSessions"),
    }


def quote_check(value: dict[str, Any], receipt: float | None, at: datetime) -> dict[str, Any]:
    if receipt is None or not 0 <= at.timestamp() - receipt <= 5:
        raise ValueError("QUOTE_STALE_OR_UNAVAILABLE")
    quote = value.get("Quote") or {}
    if quote.get("DelayedByMinutes") != 0:
        raise ValueError("QUOTE_DELAYED_OR_DELAY_UNKNOWN")
    if quote.get("ErrorCode") not in (None, "None"):
        raise ValueError("QUOTE_PERMISSION_OR_SIZE_ERROR")
    if any(
        quote.get(k) not in {"Tradable", "Indicative"} for k in ("PriceTypeBid", "PriceTypeAsk")
    ):
        raise ValueError("QUOTE_NOT_USABLE")
    bid, ask = positive(quote.get("Bid"), "BID"), positive(quote.get("Ask"), "ASK")
    if bid > ask:
        raise ValueError("CROSSED_QUOTE")
    return quote


def budget(
    option: dict[str, Any], ask: float, fee_gbp: float, native_to_gbp: float
) -> dict[str, Any]:
    if option["asset_type"] != "FuturesOption":
        raise ValueError("DIRECT_FUTURES_ORDERS_DISABLED")
    if option["minimum_quantity"] != 1 or option["lot_size"] != 1 or option["amount_decimals"] != 0:
        raise ValueError("ONE_WHOLE_CONTRACT_NOT_EXECUTABLE")
    values = [
        positive(x, "BUDGET_INPUT") for x in (ask, fee_gbp, native_to_gbp, option["price_factor"])
    ]
    premium = Decimal(str(values[0])) * Decimal(str(values[3])) * Decimal(str(values[2]))
    fees = Decimal(str(values[1]))
    pennies = int(((premium + fees) * 100).to_integral_value(rounding=ROUND_CEILING))
    if pennies > 1000:
        raise ValueError("MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET")
    return {
        "quantity": 1,
        "premium_gbp": float(premium),
        "fees_gbp": float(fees),
        "total_gbp": pennies / 100,
        "cash_pennies": pennies,
        "currency": option["currency"],
        "multiplier": option["price_factor"],
        "price_unit_factor": 1,
        "option": option,
        "limit": ask,
        "fx": native_to_gbp,
    }


def verified_cutoff(option: dict[str, Any], exit_at: datetime) -> datetime:
    # Saxo ExpiryDate often supplies a date, not an exercise/last-trade instant.
    # Do not promote a date-at-midnight or future's expiry to a verified option cutoff.
    expiry = str(option.get("expiry", ""))
    if "T" not in expiry or utc(expiry).time().isoformat() == "00:00:00":
        raise ValueError("OPTION_SPECIFIC_EXPIRY_TIME_UNVERIFIED")
    expiry_at = utc(expiry)
    sessions = (option.get("trading_sessions") or {}).get("Sessions", [])
    ends = [
        utc(s["EndTime"])
        for s in sessions
        if utc(s["StartTime"]) <= exit_at < utc(s["EndTime"])
        and str(s.get("State", "")).lower() in {"open", "openfortrading"}
    ]
    if not ends:
        raise ValueError("OPTION_EXIT_SESSION_UNVERIFIED")
    cutoff = min(expiry_at, max(ends))
    if exit_at + timedelta(seconds=120) >= cutoff:
        raise ValueError("UNSUPPORTED_EXIT_BEFORE_CONTRACT_CUTOFF")
    return cutoff
