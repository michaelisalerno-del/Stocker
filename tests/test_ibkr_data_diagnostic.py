import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_core.config import IbkrConfig
from stocker_core.runs import Environment
from stocker_execution.ibkr import (
    BrokerSession,
    CurrentQuote,
    HistoricalBar,
    IbkrApiError,
    OptionChainDefinition,
    OptionContractRequest,
    OptionMarketSnapshot,
    QualifiedInstrument,
    QualifiedOption,
)
from stocker_execution.ibkr_data_diagnostic import (
    IbkrDataCapability,
    IbkrDataCheck,
    IbkrDataDiagnosticReport,
    IbkrDataStatus,
    classify_ibkr_data_failure,
    diagnose_ibkr_data,
)


def api_error(code: int, message: str) -> IbkrApiError:
    return IbkrApiError(17, code, message, 265598, "AAPL", "SMART")


def test_ibkr_data_failure_classification_distinguishes_entitlement_and_session_errors() -> None:
    assert (
        classify_ibkr_data_failure((api_error(354, "Not subscribed to requested market data"),))
        is IbkrDataStatus.NOT_ENTITLED
    )
    assert (
        classify_ibkr_data_failure(
            (api_error(10089, "Requested market data requires additional subscription"),)
        )
        is IbkrDataStatus.NOT_ENTITLED
    )
    assert (
        classify_ibkr_data_failure(
            (api_error(10197, "No market data during competing live session"),)
        )
        is IbkrDataStatus.SESSION_CONFIGURATION_ERROR
    )
    assert (
        classify_ibkr_data_failure((api_error(200, "No security definition found"),))
        is IbkrDataStatus.CONTRACT_ERROR
    )
    assert (
        classify_ibkr_data_failure((api_error(99999, "Unclassified gateway response"),))
        is IbkrDataStatus.UNKNOWN
    )
    assert classify_ibkr_data_failure((), delayed_available=True) is IbkrDataStatus.DELAYED_ONLY
    assert (
        classify_ibkr_data_failure(
            (api_error(10197, "No market data during competing live session"),),
            delayed_available=True,
        )
        is IbkrDataStatus.SESSION_CONFIGURATION_ERROR
    )


class ReadyDiagnosticConnection:
    def __init__(self) -> None:
        self.disconnected = False
        self.quote_market_data_types: list[int] = []
        self.option_market_data_types: list[int] = []
        self.qualified_request_strikes: list[float] = []
        self.historical_end_time: datetime | None = None

    async def connect(self) -> BrokerSession:
        return BrokerSession(Environment.PAPER, "DU123456", True)

    def disconnect(self) -> None:
        self.disconnected = True

    @contextmanager
    def capture_api_errors(self) -> Iterator[list[IbkrApiError]]:
        yield []

    async def resolve_stock(self, *args: object, **kwargs: object) -> QualifiedInstrument:
        return QualifiedInstrument("AAPL", 265598, "SMART", "NASDAQ", "USD", "STK")

    async def current_quote(
        self, instrument: QualifiedInstrument, *, market_data_type: int = 1
    ) -> CurrentQuote:
        self.quote_market_data_types.append(market_data_type)
        return CurrentQuote(
            instrument.symbol,
            instrument.con_id,
            datetime(2026, 9, 2, 14, 31, tzinfo=UTC),
            230.8,
            230.9,
            230.85,
            229.7,
            market_data_type,
        )

    async def historical_bars(
        self, instrument: QualifiedInstrument, **kwargs: object
    ) -> tuple[HistoricalBar, ...]:
        end_time = kwargs.get("end_time")
        self.historical_end_time = end_time if isinstance(end_time, datetime) else None
        return (
            HistoricalBar(
                datetime(2026, 9, 1, 19, 55, tzinfo=UTC),
                229.5,
                230.0,
                229.0,
                229.7,
                1_000,
            ),
        )

    async def option_chains(
        self, instrument: QualifiedInstrument
    ) -> tuple[OptionChainDefinition, ...]:
        return (
            OptionChainDefinition(
                "SMART",
                instrument.con_id,
                instrument.symbol,
                "100",
                (date(2026, 9, 18),),
                (225.0, 230.0, 235.0),
            ),
        )

    async def qualify_options(
        self, requests: tuple[OptionContractRequest, ...]
    ) -> tuple[QualifiedOption, ...]:
        self.qualified_request_strikes.extend(request.strike for request in requests)
        return tuple(
            QualifiedOption(
                request.symbol,
                99001 if request.right == "C" else 99002,
                request.exchange,
                request.currency,
                "OPT",
                request.expiry,
                request.strike,
                request.right,
                request.multiplier,
                request.trading_class,
            )
            for request in requests
        )

    async def option_snapshots(
        self, options: tuple[QualifiedOption, ...], *, market_data_type: int = 1
    ) -> tuple[OptionMarketSnapshot, ...]:
        self.option_market_data_types.append(market_data_type)
        return tuple(
            OptionMarketSnapshot(
                option,
                datetime(2026, 9, 2, 14, 32, tzinfo=UTC),
                1.0,
                1.2,
                100,
                market_data_type,
                0.25,
                0.5 if option.right == "C" else -0.5,
                0.02,
            )
            for option in options
        )


def test_read_only_data_diagnostic_exercises_every_required_capability() -> None:
    connection = ReadyDiagnosticConnection()

    report = asyncio.run(
        diagnose_ibkr_data(
            connection,  # type: ignore[arg-type]
            symbol="AAPL",
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
            clock=lambda: datetime(2026, 9, 2, 14, 30, tzinfo=UTC),
        )
    )

    assert report.environment is Environment.PAPER
    assert report.masked_account == "DU***456"
    assert {check.capability: check.status for check in report.checks} == {
        IbkrDataCapability.CONNECTION: IbkrDataStatus.AVAILABLE,
        IbkrDataCapability.STOCK_QUALIFICATION: IbkrDataStatus.AVAILABLE,
        IbkrDataCapability.STOCK_SNAPSHOT: IbkrDataStatus.AVAILABLE,
        IbkrDataCapability.STOCK_HISTORY: IbkrDataStatus.AVAILABLE,
        IbkrDataCapability.OPTION_CHAIN: IbkrDataStatus.AVAILABLE,
        IbkrDataCapability.OPTION_QUOTE: IbkrDataStatus.AVAILABLE,
        IbkrDataCapability.OPTION_MODEL_IV: IbkrDataStatus.AVAILABLE,
    }
    assert connection.quote_market_data_types == [1]
    assert connection.option_market_data_types == [1]
    assert set(connection.qualified_request_strikes) == {225.0, 230.0, 235.0}
    assert connection.historical_end_time == datetime(2026, 9, 1, 20, 0, tzinfo=UTC)
    assert connection.disconnected is True


def test_diagnostic_does_not_treat_an_unknown_market_data_type_as_available() -> None:
    class UnknownQuoteTypeConnection(ReadyDiagnosticConnection):
        async def current_quote(
            self, instrument: QualifiedInstrument, *, market_data_type: int = 1
        ) -> CurrentQuote:
            return CurrentQuote(
                instrument.symbol,
                instrument.con_id,
                datetime(2026, 9, 2, 14, 31, tzinfo=UTC),
                230.8,
                230.9,
                230.85,
                229.7,
                None,
            )

    report = asyncio.run(
        diagnose_ibkr_data(
            UnknownQuoteTypeConnection(),  # type: ignore[arg-type]
            clock=lambda: datetime(2026, 9, 2, 14, 30, tzinfo=UTC),
        )
    )

    stock_quote = next(
        check for check in report.checks if check.capability is IbkrDataCapability.STOCK_SNAPSHOT
    )
    assert stock_quote.status is IbkrDataStatus.UNKNOWN


def test_diagnostic_probes_delayed_option_data_after_live_option_failure() -> None:
    class DelayedOptionConnection(ReadyDiagnosticConnection):
        async def option_snapshots(
            self, options: tuple[QualifiedOption, ...], *, market_data_type: int = 1
        ) -> tuple[OptionMarketSnapshot, ...]:
            self.option_market_data_types.append(market_data_type)
            if market_data_type == 1:
                raise RuntimeError("not subscribed to requested market data")
            return tuple(
                OptionMarketSnapshot(
                    option,
                    datetime(2026, 9, 2, 14, 32, tzinfo=UTC),
                    1.0,
                    1.2,
                    100,
                    3,
                    0.24,
                    0.5 if option.right == "C" else -0.5,
                    0.02,
                )
                for option in options
            )

    connection = DelayedOptionConnection()
    report = asyncio.run(
        diagnose_ibkr_data(
            connection,  # type: ignore[arg-type]
            clock=lambda: datetime(2026, 9, 2, 14, 30, tzinfo=UTC),
        )
    )

    option_checks = {
        check.capability: check
        for check in report.checks
        if check.capability
        in {IbkrDataCapability.OPTION_QUOTE, IbkrDataCapability.OPTION_MODEL_IV}
    }
    assert connection.option_market_data_types == [1, 3]
    assert option_checks[IbkrDataCapability.OPTION_QUOTE].status is IbkrDataStatus.DELAYED_ONLY
    assert option_checks[IbkrDataCapability.OPTION_QUOTE].delayed_available is True
    assert option_checks[IbkrDataCapability.OPTION_MODEL_IV].status is IbkrDataStatus.DELAYED_ONLY
    assert option_checks[IbkrDataCapability.OPTION_MODEL_IV].delayed_available is True


def test_ibkr_data_diagnostic_cli_reports_capability_and_universe_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    run_config = root / "run.yaml"
    run_config.write_text(
        "run_id: diagnostic\nuniverse: CUSTOM\nstrategy: SESSION_HARD\nenvironment: PAPER\n",
        encoding="utf-8",
    )
    ibkr_config = root / "ibkr.yaml"
    ibkr_config.write_text(
        """PAPER:
  environment: PAPER
  host: 127.0.0.1
  port: 4002
  client_id: 21
""",
        encoding="utf-8",
    )
    runs_config = root / "runs.yaml"
    runs_config.write_text(
        """universes:
  - universe_id: CUSTOM
    name: Custom
    members:
      - {symbol: AAPL, exchange: SMART, primary_exchange: NASDAQ, currency: USD}
runs:
  - run_id: diagnostic
    universe: CUSTOM
    strategy: SESSION_HARD
    environment: PAPER
""",
        encoding="utf-8",
    )

    class DiagnosticConnection:
        def __init__(self, config: IbkrConfig) -> None:
            self.config = config

    async def fake_diagnostic(*args: object, **kwargs: object) -> IbkrDataDiagnosticReport:
        return IbkrDataDiagnosticReport(
            Environment.PAPER,
            "DU***456",
            "AAPL",
            (
                IbkrDataCheck(
                    IbkrDataCapability.CONNECTION,
                    IbkrDataStatus.NOT_ENTITLED,
                    "subscription unavailable",
                    symbol="AAPL",
                    exchange="NASDAQ",
                ),
            ),
        )

    monkeypatch.setattr("stocker_execution.ibkr.IbkrConnection", DiagnosticConnection)
    monkeypatch.setattr(
        "stocker_execution.ibkr_data_diagnostic.diagnose_ibkr_data", fake_diagnostic
    )

    result = CliRunner().invoke(
        app,
        [
            "ibkr-data-diagnostic",
            "--run-config",
            str(run_config),
            "--ibkr-config",
            str(ibkr_config),
            "--runs-config",
            str(runs_config),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "Configured environment: PAPER" in result.output
    assert "IBKR connection" in result.output
    assert "NOT_ENTITLED" in result.output
    assert "AAPL/NASDAQ" in result.output
    assert "Universe CUSTOM: READY members=1" in result.output
    assert "No order was transmitted" in result.output
