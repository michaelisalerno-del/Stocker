"""Listed Saxo identities, actual sessions and conservative executable option costs."""

import math
import re
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from stocker_execution.config import (
    MARKETS,
    MAX_PREMIUM_RISK_GBP,
    MAX_PREMIUM_RISK_PENNIES,
    QUOTE_MAX_AGE_SECONDS,
    Environment,
)

MULTIPLIERS = {"CL": 1000, "GC": 100, "NG": 10000, "NQ": 20, "SI": 5000}
# Saxo InstrumentSessionState for continuous trading. Saxo documents no "Open" state;
# auctions, breaks, halts, pre/post sessions and unknown values all stay blocked.
TRADING_SESSION_STATES = {"AUTOMATEDTRADING"}
# Saxo price qualities that represent a current price. Saxo marks Tradable obsolete and
# documents Indicative as its normal price ("in most cases as relevant as a Tradable price";
# the exception is FX options). OldIndicative (stale), Pending, NoMarket, NoAccess, None and
# unknown values are never usable. Real-time delivery is checked separately.
USABLE_PRICE_TYPES = {"Tradable", "Indicative"}


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("TIMESTAMP_TIMEZONE_MISSING")
    return result.astimezone(UTC)


def positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("INVALID_" + name)
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
    match = re.fullmatch(market + r"([FGHJKMNQUVXZ])([0-9]{1,4})(?::[A-Za-z0-9_-]+)?", symbol)
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
        "notice_date": raw.get("NoticeDate"),
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
    if space.get("Expiry") and str(raw.get("ExpiryDate", ""))[:10] != str(space["Expiry"])[:10]:
        raise ValueError("OPTION_EXPIRY_MISMATCH")
    return {
        "provider": "SAXO",
        "environment": future["environment"],
        "market": future["market"],
        "asset_type": "FuturesOption",
        "uic": raw["Uic"],
        "underlying_uic": future["uic"],
        "underlying_symbol": future.get("symbol"),
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
        "last_trade_at": space.get("LastTradeDate"),
        "settlement_style": raw.get("SettlementStyle"),
        "exercise_style": space.get("ExerciseStyle"),
        "notice_date": raw.get("NoticeDate"),
        "tick_size_scheme": raw.get("TickSizeScheme", space.get("TickSizeScheme")),
        "trading_sessions": raw.get("TradingSessions"),
        "is_tradable": raw.get("IsTradable"),
    }


def quote_check(value: dict[str, Any], receipt: float | None, at: datetime) -> dict[str, Any]:
    if receipt is None or not 0 <= at.timestamp() - receipt <= QUOTE_MAX_AGE_SECONDS:
        raise ValueError("QUOTE_STALE_OR_UNAVAILABLE")
    quote = value.get("Quote") or {}
    if quote.get("DelayedByMinutes") != 0:
        raise ValueError("QUOTE_DELAYED_OR_DELAY_UNKNOWN")
    if quote.get("ErrorCode") not in (None, "None"):
        raise ValueError("QUOTE_PERMISSION_OR_SIZE_ERROR")
    if any(quote.get(k) not in USABLE_PRICE_TYPES for k in ("PriceTypeBid", "PriceTypeAsk")):
        raise ValueError("QUOTE_NOT_USABLE")
    bid, ask = positive(quote.get("Bid"), "BID"), positive(quote.get("Ask"), "ASK")
    if bid > ask:
        raise ValueError("CROSSED_QUOTE")
    return quote


def executable_quote(
    option: dict[str, Any], value: dict[str, Any], receipt: float | None, at: datetime
) -> dict[str, Any]:
    if value.get("price_source") == "OPTIONS_CHAIN":
        raise ValueError("CHAIN_PRICE_NOT_EXECUTABLE")
    quote = quote_check(value, receipt, at)
    if option.get("is_tradable") is not True:
        raise ValueError("OPTION_TRADING_PERMISSION_UNVERIFIED")
    if session_state({"TradingSessions": option.get("trading_sessions")}, at) not in (
        TRADING_SESSION_STATES
    ):
        raise ValueError("OPTION_CURRENT_SESSION_NOT_OPEN_OR_UNVERIFIED")
    return quote


def nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("INVALID_" + name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("MISSING_" + name) from None
    if not math.isfinite(result) or result < 0:
        raise ValueError("INVALID_" + name)
    return result


def require_one_whole_contract(option: dict[str, Any]) -> None:
    if option["minimum_quantity"] != 1 or option["lot_size"] != 1 or option["amount_decimals"] != 0:
        raise ValueError("ONE_WHOLE_CONTRACT_NOT_EXECUTABLE")


def all_in_pennies(
    ask: float, price_factor: float, native_to_gbp: float, fees_gbp: Decimal
) -> tuple[Decimal, int]:
    """Premium in GBP and the whole-penny ceiling of premium plus fees (rounded up)."""
    premium = (
        Decimal(str(positive(ask, "ASK")))
        * Decimal(str(positive(price_factor, "PRICE_FACTOR")))
        * Decimal(str(positive(native_to_gbp, "GBP_CONVERSION")))
    )
    return premium, int(((premium + fees_gbp) * 100).to_integral_value(rounding=ROUND_CEILING))


def cost_estimate(
    option: dict[str, Any], ask: float, conditions: dict[str, Any], native_to_gbp: float
) -> dict[str, Any]:
    """One whole contract, entry plus separately reserved exit costs, as before.

    Fixed/per-lot commissions are supported. Ambiguous tiers, percentage fees,
    taxes or conversion markup require verified conventions, never a zero fill.
    Returned totals/price commissions are not added to these component costs.
    """
    if conditions.get("AssetType") != "FuturesOption" or conditions.get("Uic") != option["uic"]:
        raise ValueError("CONTRACT_OPTION_COST_IDENTITY_UNVERIFIED")
    if conditions.get("InstrumentCurrency") != option["currency"]:
        raise ValueError("CONTRACT_OPTION_COST_CURRENCY_MISMATCH")
    if conditions.get("IsTradable") is False:
        raise ValueError("CONTRACT_OPTION_TRADING_NOT_ALLOWED")
    require_one_whole_contract(option)
    if option.get("tick_size_scheme"):
        raise ValueError("VARIABLE_TICK_SCHEME_REQUIRES_VERIFIED_PRICE_TIER")
    rate = positive(native_to_gbp, "GBP_CONVERSION")

    def convert(amount: float, currency: Any) -> float:
        if currency == "GBP":
            return amount
        if currency == option["currency"]:
            return amount * rate
        raise ValueError("FEE_CURRENCY_CONVERSION_UNVERIFIED")

    for name in ("Taxes", "ScheduledContractOptionTradingConditions", "HoldingFee", "CarryingCost"):
        if conditions.get(name):
            raise ValueError("CONTRACT_OPTION_COST_RULE_UNVERIFIED_" + name.upper())
    conversion = conditions.get("CurrencyConversion") or {}
    if conditions.get("AccountCurrency") != option["currency"] and conversion.get("Markup") != 0:
        raise ValueError("BROKER_FX_MARKUP_UNVERIFIED")
    limits = [
        r for r in conditions.get("CommissionLimits", []) if r.get("OrderAction") == "ExecuteOrder"
    ]
    if len(limits) != 1:
        raise ValueError("COMMISSION_SCHEDULE_MISSING_OR_AMBIGUOUS")
    rule = limits[0]
    if any(
        rule.get(k) is not None
        for k in (
            "MinAmount",
            "MaxAmount",
            "MinNumberOfContracts",
            "MaxNumberOfContracts",
            "MinPrice",
            "MaxPrice",
            "RateOnAmount",
            "SpreadMarkup",
            "SpreadRate",
            "MinSpread",
        )
    ):
        raise ValueError("COMMISSION_TIER_OR_SCALING_UNVERIFIED")
    if "PerUnitRate" not in rule and "BaseCommission" not in rule:
        raise ValueError("COMMISSION_VALUE_MISSING")
    fee = sum(
        nonnegative(rule[k], "COMMISSION") for k in ("PerUnitRate", "BaseCommission") if k in rule
    )
    if "MinCommission" in rule:
        fee = max(fee, nonnegative(rule["MinCommission"], "MIN_COMMISSION"))
    if "MaxCommission" in rule:
        fee = min(fee, nonnegative(rule["MaxCommission"], "MAX_COMMISSION"))
    entry = convert(fee, rule.get("Currency"))
    # Schema says these rules are returned if exchange fees apply separately.
    for exchange in conditions.get("ExchangeFeeRules", []):
        if exchange.get("OrderAction") != "ExecuteOrder":
            continue
        if exchange.get("Type") not in {"Absolute", "PerAction", "PerLot"}:
            raise ValueError("EXCHANGE_FEE_SCALING_UNVERIFIED")
        fee = nonnegative(exchange.get("Value"), "EXCHANGE_FEE")
        if "Minimum" in exchange:
            fee = max(fee, nonnegative(exchange["Minimum"], "EXCHANGE_MINIMUM"))
        if "Maximum" in exchange:
            fee = min(fee, nonnegative(exchange["Maximum"], "EXCHANGE_MAXIMUM"))
        entry += convert(fee, exchange.get("Currency"))
    premium, pennies = all_in_pennies(ask, option["price_factor"], rate, Decimal(str(entry)) * 2)
    return {
        "quantity": 1,
        "premium_gbp": float(premium),
        "entry_costs_gbp": entry,
        "estimated_exit_costs_gbp": entry,
        "fees_gbp": entry * 2,
        "minimum_purchase_cost_gbp": float(premium) + entry,
        "total_gbp": pennies / 100,
        "remaining_budget_gbp": (MAX_PREMIUM_RISK_PENNIES - pennies) / 100,
        "cash_pennies": pennies,
        "budget_gbp": MAX_PREMIUM_RISK_GBP,
        "budget_result": "WITHIN_BUDGET"
        if pennies <= MAX_PREMIUM_RISK_PENNIES
        else "MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET",
        "basis": "ESTIMATE_NOT_BOOKED_CHARGES",
        "policy": "PREMIUM_PLUS_ENTRY_AND_RESERVED_EXIT",
        "fee_per_side_gbp": entry,
        "currency": option["currency"],
        "multiplier": option["price_factor"],
        "price_unit_factor": 1,
        "option": option,
        "limit": ask,
        "fx": rate,
    }


def deadline_instant(value: Any, day: str, zone: str | None = None) -> str | None:
    """Accept a dated offset, or a local clock with an explicitly supplied IANA zone.

    Ambiguous/nonexistent DST times and sentinel dates remain unknown.
    """
    if not isinstance(value, str):
        return None
    try:
        if "T" in value:
            parsed = utc(value)
            return parsed.isoformat() if parsed.year > 2000 else None
        if not zone:
            return None
        local = datetime.fromisoformat(day + "T" + value)
        tz = ZoneInfo(zone)
        first, second = local.replace(tzinfo=tz, fold=0), local.replace(tzinfo=tz, fold=1)
        if (
            first.utcoffset() != second.utcoffset()
            or first.astimezone(UTC).astimezone(tz).replace(tzinfo=None) != local
        ):
            return None
        return first.astimezone(UTC).isoformat()
    except (ValueError, ZoneInfoNotFoundError):
        return None


def budget(
    option: dict[str, Any], ask: float, fee_gbp: float, native_to_gbp: float
) -> dict[str, Any]:
    """Re-price an admitted plan at its actual fill price using the plan's own fees."""
    if option["asset_type"] != "FuturesOption":
        raise ValueError("DIRECT_FUTURES_ORDERS_DISABLED")
    require_one_whole_contract(option)
    fees = Decimal(str(nonnegative(fee_gbp, "FEES")))
    premium, pennies = all_in_pennies(ask, option["price_factor"], native_to_gbp, fees)
    if pennies > MAX_PREMIUM_RISK_PENNIES:
        raise ValueError("MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET")
    return {
        "quantity": 1,
        "premium_gbp": float(premium),
        "fees_gbp": float(fees),
        "total_gbp": pennies / 100,
        "remaining_budget_gbp": (MAX_PREMIUM_RISK_PENNIES - pennies) / 100,
        "cash_pennies": pennies,
        "currency": option["currency"],
        "multiplier": option["price_factor"],
        "price_unit_factor": 1,
        "option": option,
        "limit": ask,
        "fx": native_to_gbp,
    }


def verified_cutoff(option: dict[str, Any], exit_at: datetime) -> datetime:
    expiry_instant = deadline_instant(
        option.get("expiry_instant"), str(option.get("expiry", ""))[:10]
    )
    if not expiry_instant:
        raise ValueError("OPTION_SPECIFIC_EXPIRY_TIME_UNVERIFIED")
    last_trade = deadline_instant(option.get("last_trade_at"), str(option.get("expiry", ""))[:10])
    if not last_trade:
        raise ValueError("OPTION_LAST_TRADING_DEADLINE_UNVERIFIED")
    sessions = (option.get("trading_sessions") or {}).get("Sessions", [])
    ends = [
        utc(s["EndTime"])
        for s in sessions
        if utc(s["StartTime"]) <= exit_at < utc(s["EndTime"])
        and str(s.get("State", "")).upper() in TRADING_SESSION_STATES
    ]
    if not ends:
        raise ValueError("OPTION_EXIT_SESSION_UNVERIFIED")
    cutoff = min(utc(last_trade), utc(expiry_instant), max(ends))
    if exit_at + timedelta(seconds=120) >= cutoff:
        raise ValueError("UNSUPPORTED_EXIT_BEFORE_CONTRACT_CUTOFF")
    return cutoff
