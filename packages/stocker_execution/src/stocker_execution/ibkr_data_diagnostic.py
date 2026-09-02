"""Read-only, entitlement-aware diagnostics for Stocker's required IBKR data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum

from stocker_core.runs import Environment
from stocker_execution.ibkr import (
    CurrentQuote,
    IbkrApiError,
    IbkrConnection,
    OptionChainDefinition,
    OptionMarketSnapshot,
    QualifiedInstrument,
    QualifiedOption,
    sanitize_ibkr_message,
)
from stocker_execution.pre_context import (
    build_pre_context_option_request_groups,
    next_pre_context_target_session,
    pre_context_session_contract,
    select_nearest_pre_context_qualified_options,
)


class IbkrDataCapability(StrEnum):
    CONNECTION = "IBKR connection"
    STOCK_QUALIFICATION = "US stock qualification"
    STOCK_SNAPSHOT = "US stock snapshot"
    STOCK_HISTORY = "US stock historical bars"
    OPTION_CHAIN = "Option chain metadata"
    OPTION_QUOTE = "Diagnostic ATM option quote/OI"
    OPTION_MODEL_IV = "Option tick-13 model IV"


class IbkrDataStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    DELAYED_ONLY = "DELAYED_ONLY"
    NOT_ENTITLED = "NOT_ENTITLED"
    SESSION_CONFIGURATION_ERROR = "SESSION_CONFIGURATION_ERROR"
    CONTRACT_ERROR = "CONTRACT_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class IbkrDataCheck:
    capability: IbkrDataCapability
    status: IbkrDataStatus
    detail: str
    errors: tuple[IbkrApiError, ...] = ()
    symbol: str | None = None
    exchange: str | None = None
    delayed_available: bool = False
    required_for: str = ""
    apparent_entitlement: str = ""


@dataclass(frozen=True, slots=True)
class IbkrDataDiagnosticReport:
    environment: Environment | None
    masked_account: str | None
    symbol: str
    checks: tuple[IbkrDataCheck, ...]


_NOT_ENTITLED_CODES = frozenset({354, 10089, 10090, 10186})
_SESSION_ERROR_CODES = frozenset({501, 502, 503, 504, 1100, 1101, 1102, 1300, 10197})
_CONTRACT_ERROR_CODES = frozenset({200, 203, 321, 322})


def classify_ibkr_data_failure(
    errors: tuple[IbkrApiError, ...],
    *,
    exception: Exception | None = None,
    delayed_available: bool = False,
) -> IbkrDataStatus:
    """Classify a failed request without disguising unknown broker responses."""

    codes = {error.code for error in errors}
    text = " ".join([*(error.message.lower() for error in errors), str(exception or "").lower()])
    if codes & _SESSION_ERROR_CODES or any(
        phrase in text
        for phrase in (
            "not connected",
            "connection failed",
            "socket port",
            "competing session",
            "api connection",
        )
    ):
        return IbkrDataStatus.SESSION_CONFIGURATION_ERROR
    if codes & _CONTRACT_ERROR_CODES or any(
        phrase in text
        for phrase in (
            "no security definition",
            "contract could not be resolved",
            "resolved ambiguously",
        )
    ):
        return IbkrDataStatus.CONTRACT_ERROR
    if delayed_available:
        return IbkrDataStatus.DELAYED_ONLY
    if codes & _NOT_ENTITLED_CODES or any(
        phrase in text
        for phrase in (
            "not subscribed",
            "requires additional subscription",
            "market data permissions",
            "no market data permission",
        )
    ):
        return IbkrDataStatus.NOT_ENTITLED
    return IbkrDataStatus.UNKNOWN


async def diagnose_ibkr_data(
    connection: IbkrConnection,
    *,
    symbol: str = "AAPL",
    exchange: str = "SMART",
    primary_exchange: str | None = "NASDAQ",
    currency: str = "USD",
    clock: Callable[[], datetime] | None = None,
) -> IbkrDataDiagnosticReport:
    """Exercise Stocker's required data calls on one symbol without placing orders."""

    now = (clock or (lambda: datetime.now(tz=UTC)))()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("IBKR data diagnostic clock must be timezone-aware")
    checks: list[IbkrDataCheck] = []
    environment: Environment | None = getattr(connection, "environment", None)
    masked_account: str | None = None
    instrument: QualifiedInstrument | None = None
    quote: CurrentQuote | None = None
    reference_price: float | None = None
    option_observation_date: date | None = None
    try:
        target_session = next_pre_context_target_session(now)
        option_observation_date, _, previous_close_at, _ = pre_context_session_contract(
            target_session
        )
    except ValueError:
        previous_close_at = None
    chains: tuple[OptionChainDefinition, ...] = ()

    try:
        with connection.capture_api_errors() as errors:
            try:
                session = await connection.connect()
            except Exception as exc:
                checks.append(
                    _failure_check(
                        IbkrDataCapability.CONNECTION,
                        errors,
                        exc,
                        symbol=symbol,
                        exchange=exchange,
                        required_for="all IBKR runtime data",
                    )
                )
                _append_not_checked(checks, "IBKR connection was not available")
                return IbkrDataDiagnosticReport(environment, None, symbol, tuple(checks))
        environment = session.environment
        masked_account = session.masked_account_id
        checks.append(
            IbkrDataCheck(
                IbkrDataCapability.CONNECTION,
                IbkrDataStatus.AVAILABLE,
                f"verified {session.environment.value} session",
                tuple(errors),
                required_for="all IBKR runtime data",
            )
        )

        with connection.capture_api_errors() as errors:
            try:
                instrument = await connection.resolve_stock(
                    symbol,
                    exchange=exchange,
                    primary_exchange=primary_exchange,
                    currency=currency,
                )
            except Exception as exc:
                checks.append(
                    _failure_check(
                        IbkrDataCapability.STOCK_QUALIFICATION,
                        errors,
                        exc,
                        symbol=symbol,
                        exchange=primary_exchange or exchange,
                        required_for="Stage 2 canonical conId identity",
                    )
                )
                _append_not_checked(checks, "test stock could not be qualified")
                return IbkrDataDiagnosticReport(environment, masked_account, symbol, tuple(checks))
        checks.append(
            IbkrDataCheck(
                IbkrDataCapability.STOCK_QUALIFICATION,
                IbkrDataStatus.AVAILABLE,
                f"qualified conId={instrument.con_id}",
                tuple(errors),
                instrument.symbol,
                instrument.primary_exchange or instrument.exchange,
                required_for="Stage 2 canonical conId identity",
            )
        )

        quote, quote_check = await _diagnose_stock_snapshot(connection, instrument)
        checks.append(quote_check)
        with connection.capture_api_errors() as errors:
            try:
                bars = await connection.historical_bars(
                    instrument,
                    bar_size="5 mins",
                    duration="1 D",
                    what_to_show="TRADES",
                    regular_trading_hours=True,
                    end_time=previous_close_at,
                )
            except Exception as exc:
                checks.append(
                    _failure_check(
                        IbkrDataCapability.STOCK_HISTORY,
                        errors,
                        exc,
                        instrument=instrument,
                        required_for="Stage 4 prior session and Stage 5 PRE inputs",
                    )
                )
            else:
                reference_price = bars[-1].close
                checks.append(
                    _success_check(
                        IbkrDataCapability.STOCK_HISTORY,
                        f"received {len(bars)} validated 5-minute TRADES bars",
                        errors,
                        instrument,
                        required_for="Stage 4 prior session and Stage 5 PRE inputs",
                    )
                )

        with connection.capture_api_errors() as errors:
            try:
                chains = await connection.option_chains(instrument)
            except Exception as exc:
                checks.append(
                    _failure_check(
                        IbkrDataCapability.OPTION_CHAIN,
                        errors,
                        exc,
                        instrument=instrument,
                        required_for="Stage 4 option expiry and strike discovery",
                    )
                )
            else:
                checks.append(
                    _success_check(
                        IbkrDataCapability.OPTION_CHAIN,
                        f"received {len(chains)} security-definition chains",
                        errors,
                        instrument,
                        required_for="Stage 4 option expiry and strike discovery",
                    )
                )

        if not chains or reference_price is None or option_observation_date is None:
            reason = (
                "option-chain metadata was unavailable"
                if not chains
                else "no historical reference was available for the PRE option selector"
            )
            _append_option_not_checked(checks, reason, instrument)
        else:
            option_checks = await _diagnose_option_data(
                connection,
                instrument,
                chains,
                reference_price=reference_price,
                observation_date=option_observation_date,
            )
            checks.extend(option_checks)
        return IbkrDataDiagnosticReport(environment, masked_account, symbol, tuple(checks))
    finally:
        connection.disconnect()


async def _diagnose_stock_snapshot(
    connection: IbkrConnection, instrument: QualifiedInstrument
) -> tuple[CurrentQuote | None, IbkrDataCheck]:
    live_errors: list[IbkrApiError]
    with connection.capture_api_errors() as live_errors:
        try:
            quote = await connection.current_quote(instrument, market_data_type=1)
        except Exception as live_exc:
            with connection.capture_api_errors() as delayed_errors:
                try:
                    delayed = await connection.current_quote(instrument, market_data_type=3)
                except Exception as delayed_exc:
                    combined = tuple([*live_errors, *delayed_errors])
                    return None, _failure_check(
                        IbkrDataCapability.STOCK_SNAPSHOT,
                        combined,
                        delayed_exc,
                        instrument=instrument,
                        required_for="read-only current quote diagnostics",
                    )
            status = classify_ibkr_data_failure(
                tuple([*live_errors, *delayed_errors]),
                exception=live_exc,
                delayed_available=True,
            )
            return delayed, IbkrDataCheck(
                IbkrDataCapability.STOCK_SNAPSHOT,
                status,
                "live request failed "
                f"({sanitize_ibkr_message(live_exc)}); delayed quote is available",
                tuple([*live_errors, *delayed_errors]),
                instrument.symbol,
                instrument.primary_exchange or instrument.exchange,
                True,
                "read-only current quote diagnostics",
                _underlying_entitlement(instrument),
            )
    if quote.market_data_type in {1, 2}:
        status = IbkrDataStatus.AVAILABLE
        delayed_available = False
    elif quote.market_data_type in {3, 4}:
        status = IbkrDataStatus.DELAYED_ONLY
        delayed_available = True
    else:
        status = IbkrDataStatus.UNKNOWN
        delayed_available = False
    return quote, IbkrDataCheck(
        IbkrDataCapability.STOCK_SNAPSHOT,
        status,
        f"market data type {quote.market_data_type or 'not reported'}",
        tuple(live_errors),
        instrument.symbol,
        instrument.primary_exchange or instrument.exchange,
        delayed_available,
        "read-only current quote diagnostics",
        _underlying_entitlement(instrument),
    )


async def _diagnose_option_data(
    connection: IbkrConnection,
    instrument: QualifiedInstrument,
    chains: tuple[OptionChainDefinition, ...],
    *,
    reference_price: float,
    observation_date: date,
) -> tuple[IbkrDataCheck, IbkrDataCheck]:
    with connection.capture_api_errors() as qualification_errors:
        try:
            options = await _qualify_pre_context_option_set(
                connection,
                instrument,
                chains,
                reference_price=reference_price,
                observation_date=observation_date,
            )
        except Exception as exc:
            failure = _failure_check(
                IbkrDataCapability.OPTION_QUOTE,
                qualification_errors,
                exc,
                instrument=instrument,
                required_for="Stage 4 call/put bid, ask, and open-interest selection",
                apparent_entitlement=_option_entitlement(instrument),
            )
            iv_failure = _failure_check(
                IbkrDataCapability.OPTION_MODEL_IV,
                qualification_errors,
                exc,
                instrument=instrument,
                required_for="canonical PRE_CONTEXT_V1 ATM IV",
                apparent_entitlement=_option_model_entitlement(instrument),
            )
            return failure, iv_failure

    live_exception: Exception | None = None
    with connection.capture_api_errors() as live_errors:
        try:
            live_snapshots = await connection.option_snapshots(options, market_data_type=1)
        except Exception as exc:
            live_exception = exc
            live_snapshots = ()

    live_quote_complete = _option_quote_complete(live_snapshots, market_data_types={1, 2})
    live_iv_complete = _option_iv_complete(live_snapshots, market_data_types={1, 2})
    delayed_snapshots: tuple[OptionMarketSnapshot, ...] = ()
    delayed_errors: list[IbkrApiError] = []
    delayed_exception: Exception | None = None
    if not (live_quote_complete and live_iv_complete):
        with connection.capture_api_errors() as delayed_errors:
            try:
                delayed_snapshots = await connection.option_snapshots(
                    options, market_data_type=3
                )
            except Exception as exc:
                delayed_exception = exc

    errors = tuple([*qualification_errors, *live_errors, *delayed_errors])
    delayed_quote_available = _option_quote_complete(
        delayed_snapshots, market_data_types={3, 4}
    )
    delayed_iv_available = _option_iv_complete(
        delayed_snapshots, market_data_types={3, 4}
    )
    failure_exception = delayed_exception or live_exception

    option_exchange = options[0].exchange if options else instrument.exchange
    if live_quote_complete:
        quote_status = IbkrDataStatus.AVAILABLE
        quote_detail = "call/put bid, ask, and open interest received"
    elif delayed_quote_available:
        quote_status = classify_ibkr_data_failure(
            errors,
            exception=failure_exception,
            delayed_available=True,
        )
        quote_detail = "only delayed call/put bid, ask, and open interest received"
    else:
        quote_status = classify_ibkr_data_failure(errors, exception=failure_exception)
        quote_detail = "call/put bid, ask, or open interest missing"
    if live_iv_complete:
        iv_status = IbkrDataStatus.AVAILABLE
        iv_detail = "live/frozen tick-13 model implied volatility received for call and put"
    else:
        iv_status = classify_ibkr_data_failure(
            errors,
            exception=failure_exception,
            delayed_available=delayed_iv_available,
        )
        iv_detail = "required live/frozen tick-13 model implied volatility missing"
    return (
        IbkrDataCheck(
            IbkrDataCapability.OPTION_QUOTE,
            quote_status,
            quote_detail,
            errors,
            instrument.symbol,
            option_exchange,
            delayed_quote_available,
            "Stage 4 call/put bid, ask, and open-interest selection",
            _option_entitlement(instrument),
        ),
        IbkrDataCheck(
            IbkrDataCapability.OPTION_MODEL_IV,
            iv_status,
            iv_detail,
            errors,
            instrument.symbol,
            option_exchange,
            delayed_iv_available,
            "canonical PRE_CONTEXT_V1 ATM IV",
            _option_model_entitlement(instrument),
        ),
    )


def _option_quote_complete(
    snapshots: tuple[OptionMarketSnapshot, ...], *, market_data_types: set[int]
) -> bool:
    return bool(snapshots) and all(
        item.market_data_type in market_data_types
        and item.bid is not None
        and item.ask is not None
        and item.open_interest is not None
        for item in snapshots
    )


def _option_iv_complete(
    snapshots: tuple[OptionMarketSnapshot, ...], *, market_data_types: set[int]
) -> bool:
    return bool(snapshots) and all(
        item.market_data_type in market_data_types and item.model_iv is not None
        for item in snapshots
    )


async def _qualify_pre_context_option_set(
    connection: IbkrConnection,
    instrument: QualifiedInstrument,
    chains: tuple[OptionChainDefinition, ...],
    *,
    reference_price: float,
    observation_date: date,
) -> tuple[QualifiedOption, ...]:
    """Use the frozen PRE request and nearest-strike selectors without redefining them."""

    request_groups = build_pre_context_option_request_groups(
        instrument,
        chains,
        previous_close=reference_price,
        observation_date=observation_date,
    )
    for requests in request_groups:
        qualified = await connection.qualify_options(requests)
        try:
            return select_nearest_pre_context_qualified_options(
                qualified,
                previous_close=reference_price,
            )
        except ValueError:
            continue
    raise ValueError("no eligible qualified common-strike expiry")


def _success_check(
    capability: IbkrDataCapability,
    detail: str,
    errors: list[IbkrApiError],
    instrument: QualifiedInstrument,
    *,
    required_for: str,
) -> IbkrDataCheck:
    return IbkrDataCheck(
        capability,
        IbkrDataStatus.AVAILABLE,
        detail,
        tuple(errors),
        instrument.symbol,
        instrument.primary_exchange or instrument.exchange,
        required_for=required_for,
        apparent_entitlement=_underlying_entitlement(instrument),
    )


def _failure_check(
    capability: IbkrDataCapability,
    errors: list[IbkrApiError] | tuple[IbkrApiError, ...],
    exception: Exception,
    *,
    instrument: QualifiedInstrument | None = None,
    symbol: str | None = None,
    exchange: str | None = None,
    required_for: str,
    apparent_entitlement: str = "",
) -> IbkrDataCheck:
    normalized_errors = tuple(errors)
    resolved_symbol = instrument.symbol if instrument is not None else symbol
    resolved_exchange = (
        instrument.primary_exchange or instrument.exchange if instrument is not None else exchange
    )
    entitlement = apparent_entitlement
    if not entitlement and instrument is not None:
        entitlement = _underlying_entitlement(instrument)
    return IbkrDataCheck(
        capability,
        classify_ibkr_data_failure(normalized_errors, exception=exception),
        sanitize_ibkr_message(exception),
        normalized_errors,
        resolved_symbol,
        resolved_exchange,
        False,
        required_for,
        entitlement,
    )


def _append_not_checked(checks: list[IbkrDataCheck], reason: str) -> None:
    already = {check.capability for check in checks}
    for capability in IbkrDataCapability:
        if capability not in already:
            checks.append(IbkrDataCheck(capability, IbkrDataStatus.UNKNOWN, reason))


def _append_option_not_checked(
    checks: list[IbkrDataCheck], reason: str, instrument: QualifiedInstrument
) -> None:
    checks.extend(
        (
            IbkrDataCheck(
                IbkrDataCapability.OPTION_QUOTE,
                IbkrDataStatus.UNKNOWN,
                reason,
                symbol=instrument.symbol,
                exchange=instrument.primary_exchange or instrument.exchange,
                required_for="Stage 4 call/put bid, ask, and open-interest selection",
                apparent_entitlement=_option_entitlement(instrument),
            ),
            IbkrDataCheck(
                IbkrDataCapability.OPTION_MODEL_IV,
                IbkrDataStatus.UNKNOWN,
                reason,
                symbol=instrument.symbol,
                exchange=instrument.primary_exchange or instrument.exchange,
                required_for="canonical PRE_CONTEXT_V1 ATM IV",
                apparent_entitlement=_option_model_entitlement(instrument),
            ),
        )
    )


def _underlying_entitlement(instrument: QualifiedInstrument) -> str:
    exchange = (instrument.primary_exchange or instrument.exchange).upper()
    if exchange == "NASDAQ":
        return "NASDAQ (Network C/UTP) top-of-book"
    if exchange == "NYSE":
        return "NYSE (Network A/CTA) top-of-book"
    if exchange in {"AMEX", "ARCA", "BATS", "IEX"}:
        return "Network B US equities top-of-book"
    return "the listing exchange's US equity top-of-book service"


def _option_entitlement(instrument: QualifiedInstrument) -> str:
    del instrument
    return "OPRA (US Options Exchanges) top-of-book"


def _option_model_entitlement(instrument: QualifiedInstrument) -> str:
    return f"OPRA plus {_underlying_entitlement(instrument)}"
