import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_core.config import IbkrConfig, load_ibkr_config
from stocker_core.runs import Environment
from stocker_execution.ibkr import (
    BrokerSession,
    CurrentQuote,
    HistoricalBar,
    IbkrConnection,
    IbkrError,
    QualifiedInstrument,
)


class FakeIbClient:
    def __init__(
        self,
        *,
        accounts: list[str],
        qualification_result: list[object] | None = None,
        historical_result: list[object] | Exception | None = None,
        ticker_result: list[object] | Exception | None = None,
    ) -> None:
        self.accounts = accounts
        self.qualification_result = qualification_result or []
        self.historical_result = historical_result or []
        self.ticker_result = ticker_result or []
        self.connected = False
        self.connect_kwargs: dict[str, object] = {}
        self.disconnect_count = 0
        self.qualification_request: object | None = None
        self.historical_request: object | None = None
        self.historical_kwargs: dict[str, object] = {}
        self.ticker_request: object | None = None

    async def connectAsync(self, host: str, port: int, **kwargs: object) -> None:
        self.connect_kwargs = {"host": host, "port": port, **kwargs}
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1

    def isConnected(self) -> bool:
        return self.connected

    def managedAccounts(self) -> list[str]:
        return self.accounts

    async def qualifyContractsAsync(
        self, *contracts: object, returnAll: bool = False
    ) -> list[object]:
        assert returnAll is True
        self.qualification_request = contracts[0]
        return self.qualification_result

    async def reqHistoricalDataAsync(self, contract: object, **kwargs: object) -> list[object]:
        self.historical_request = contract
        self.historical_kwargs = kwargs
        if isinstance(self.historical_result, Exception):
            raise self.historical_result
        return self.historical_result

    async def reqTickersAsync(
        self, *contracts: object, regulatorySnapshot: bool = False
    ) -> list[object]:
        assert regulatorySnapshot is False
        self.ticker_request = contracts[0]
        if isinstance(self.ticker_result, Exception):
            raise self.ticker_result
        return self.ticker_result


def test_paper_and_live_ibkr_configurations_are_explicit_and_independent() -> None:
    paper = IbkrConfig(
        environment=Environment.PAPER,
        host="127.0.0.1",
        port=4002,
        client_id=21,
    )
    live = IbkrConfig(
        environment=Environment.LIVE,
        host="gateway.internal",
        port=4001,
        client_id=22,
    )

    assert paper.environment is Environment.PAPER
    assert paper.port == 4002
    assert paper.client_id == 21
    assert live.environment is Environment.LIVE
    assert live.host == "gateway.internal"
    assert live.port == 4001
    assert live.client_id == 22


def test_run_environment_selects_the_matching_explicit_ibkr_configuration(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "ibkr.yaml"
    config_path.write_text(
        """
PAPER:
  environment: PAPER
  host: 127.0.0.1
  port: 4002
  client_id: 21
LIVE:
  environment: LIVE
  host: gateway.internal
  port: 4101
  client_id: 22
""",
        encoding="utf-8",
    )

    paper = load_ibkr_config(config_path, Environment.PAPER)
    live = load_ibkr_config(config_path, Environment.LIVE)

    assert paper.environment is Environment.PAPER
    assert paper.port == 4002
    assert paper.client_id == 21
    assert live.environment is Environment.LIVE
    assert live.host == "gateway.internal"
    assert live.port == 4101
    assert live.client_id == 22


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"host": ""},
        {"host": "   "},
        {"port": 0},
        {"client_id": 0},
    ],
)
def test_ibkr_connection_configuration_rejects_invalid_required_fields(
    invalid_field: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "environment": Environment.PAPER,
        "host": "127.0.0.1",
        "port": 4002,
        "client_id": 21,
    }
    values.update(invalid_field)

    with pytest.raises(ValidationError):
        IbkrConfig.model_validate(values)


def test_connect_identifies_paper_session_and_disconnects_cleanly() -> None:
    client = FakeIbClient(accounts=["DU123456"])
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )

    session = asyncio.run(connection.connect())

    assert session.environment is Environment.PAPER
    assert session.account_id == "DU123456"
    assert session.masked_account_id == "DU***456"
    assert session.connected is True
    assert client.connect_kwargs["host"] == "127.0.0.1"
    assert client.connect_kwargs["port"] == 4002
    assert client.connect_kwargs["clientId"] == 21
    assert client.connect_kwargs["readonly"] is True
    assert connection.is_connected is True

    connection.disconnect()

    assert connection.is_connected is False
    assert client.disconnect_count == 1


def test_environment_mismatch_fails_and_disconnects() -> None:
    client = FakeIbClient(accounts=["U1234567"])
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )

    with pytest.raises(
        IbkrError,
        match="environment mismatch: configured PAPER, account appears LIVE",
    ):
        asyncio.run(connection.connect())

    assert connection.is_connected is False
    assert client.disconnect_count == 1


def test_multiple_accounts_require_an_explicit_expected_account() -> None:
    client = FakeIbClient(accounts=["DU123456", "DU654321"])
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )

    with pytest.raises(IbkrError, match="multiple accounts"):
        asyncio.run(connection.connect())

    assert connection.is_connected is False


def test_separate_broker_instances_do_not_share_connection_state() -> None:
    paper_client = FakeIbClient(accounts=["DU123456"])
    live_client = FakeIbClient(accounts=["U1234567"])
    paper = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=paper_client,
    )
    live = IbkrConnection(
        IbkrConfig(
            environment=Environment.LIVE,
            host="127.0.0.1",
            port=4001,
            client_id=22,
        ),
        client=live_client,
    )

    asyncio.run(paper.connect())
    asyncio.run(live.connect())
    paper.disconnect()

    assert paper.is_connected is False
    assert live.is_connected is True
    assert not hasattr(IbkrConnection, "place_order")
    with pytest.raises(IbkrError, match="verified active connection"):
        asyncio.run(paper.cancel_order(1))


def test_resolve_stock_returns_stable_qualified_instrument_identity() -> None:
    qualified_contract = SimpleNamespace(
        symbol="AAPL",
        conId=265598,
        exchange="SMART",
        primaryExchange="NASDAQ",
        currency="USD",
        secType="STK",
    )
    client = FakeIbClient(
        accounts=["DU123456"],
        qualification_result=[qualified_contract],
    )
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )

    async def scenario() -> object:
        await connection.connect()
        return await connection.resolve_stock(
            "aapl",
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
        )

    instrument = asyncio.run(scenario())

    assert instrument.symbol == "AAPL"
    assert instrument.con_id == 265598
    assert instrument.exchange == "SMART"
    assert instrument.primary_exchange == "NASDAQ"
    assert instrument.currency == "USD"
    assert instrument.security_type == "STK"
    assert client.qualification_request.symbol == "AAPL"
    assert client.qualification_request.primaryExchange == "NASDAQ"


@pytest.mark.parametrize(
    ("qualification_result", "message"),
    [
        ([None], "could not be resolved"),
        ([[SimpleNamespace(conId=1), SimpleNamespace(conId=2)]], "resolved ambiguously"),
    ],
)
def test_invalid_or_ambiguous_contract_resolution_fails_clearly(
    qualification_result: list[object], message: str
) -> None:
    client = FakeIbClient(
        accounts=["DU123456"],
        qualification_result=qualification_result,
    )
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )

    async def scenario() -> None:
        await connection.connect()
        await connection.resolve_stock("INVALID", exchange="SMART", currency="USD")

    with pytest.raises(IbkrError, match=message):
        asyncio.run(scenario())


def test_historical_ibkr_bars_are_converted_without_fabrication() -> None:
    timestamp = datetime(2026, 9, 1, 14, 30, tzinfo=UTC)
    client = FakeIbClient(
        accounts=["DU123456"],
        historical_result=[
            SimpleNamespace(
                date=timestamp,
                open=230.1,
                high=231.25,
                low=229.8,
                close=230.9,
                volume=12345,
            )
        ],
    )
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )
    instrument = QualifiedInstrument(
        symbol="AAPL",
        con_id=265598,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
    )

    async def scenario() -> object:
        await connection.connect()
        return await connection.historical_bars(
            instrument,
            bar_size="1 min",
            duration="1 D",
            what_to_show="TRADES",
            regular_trading_hours=True,
            end_time=timestamp,
        )

    bars = asyncio.run(scenario())

    assert len(bars) == 1
    assert bars[0].timestamp == timestamp
    assert bars[0].open == 230.1
    assert bars[0].high == 231.25
    assert bars[0].low == 229.8
    assert bars[0].close == 230.9
    assert bars[0].volume == 12345.0
    assert client.historical_request.conId == 265598
    assert client.historical_kwargs == {
        "endDateTime": timestamp,
        "durationStr": "1 D",
        "barSizeSetting": "1 min",
        "whatToShow": "TRADES",
        "useRTH": True,
        "formatDate": 2,
        "keepUpToDate": False,
        "timeout": 60.0,
    }


@pytest.mark.parametrize(
    ("historical_result", "message"),
    [
        ([], "returned no bars"),
        (RuntimeError("request rejected"), "historical data request failed"),
    ],
)
def test_historical_failure_does_not_substitute_data_or_drop_connection(
    historical_result: list[object] | Exception, message: str
) -> None:
    client = FakeIbClient(
        accounts=["DU123456"],
        historical_result=historical_result,
    )
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )
    instrument = QualifiedInstrument(
        symbol="AAPL",
        con_id=265598,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
    )

    async def scenario() -> None:
        await connection.connect()
        await connection.historical_bars(
            instrument,
            bar_size="1 min",
            duration="1 D",
            what_to_show="TRADES",
            regular_trading_hours=True,
        )

    with pytest.raises(IbkrError, match=message):
        asyncio.run(scenario())

    assert connection.is_connected is True


def test_historical_response_fails_when_shorter_than_the_callers_required_sample() -> None:
    client = FakeIbClient(
        accounts=["DU123456"],
        historical_result=[
            SimpleNamespace(
                date=datetime(2026, 9, 1, 14, 30, tzinfo=UTC),
                open=230.1,
                high=231.25,
                low=229.8,
                close=230.9,
                volume=12345,
            )
        ],
    )
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )
    instrument = QualifiedInstrument(
        symbol="AAPL",
        con_id=265598,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
    )

    async def scenario() -> None:
        await connection.connect()
        await connection.historical_bars(
            instrument,
            bar_size="1 min",
            duration="1 D",
            what_to_show="TRADES",
            regular_trading_hours=True,
            minimum_bars=2,
        )

    with pytest.raises(IbkrError, match="1 bars; at least 2 required"):
        asyncio.run(scenario())


def test_historical_response_with_non_increasing_timestamps_is_invalid() -> None:
    client = FakeIbClient(
        accounts=["DU123456"],
        historical_result=[
            SimpleNamespace(
                date=datetime(2026, 9, 1, 14, minute, tzinfo=UTC),
                open=230.1,
                high=231.25,
                low=229.8,
                close=230.9,
                volume=12345,
            )
            for minute in (31, 30)
        ],
    )
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )
    instrument = QualifiedInstrument(
        symbol="AAPL",
        con_id=265598,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
    )

    async def scenario() -> None:
        await connection.connect()
        await connection.historical_bars(
            instrument,
            bar_size="1 min",
            duration="1 D",
            what_to_show="TRADES",
            regular_trading_hours=True,
        )

    with pytest.raises(IbkrError, match="timestamps are not strictly increasing"):
        asyncio.run(scenario())


def test_current_quote_snapshot_is_converted_to_a_small_internal_result() -> None:
    timestamp = datetime(2026, 9, 1, 14, 31, tzinfo=UTC)
    client = FakeIbClient(
        accounts=["DU123456"],
        ticker_result=[
            SimpleNamespace(
                time=timestamp,
                bid=230.8,
                ask=230.9,
                last=230.85,
                close=229.7,
            )
        ],
    )
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )
    instrument = QualifiedInstrument(
        symbol="AAPL",
        con_id=265598,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
    )

    async def scenario() -> object:
        await connection.connect()
        return await connection.current_quote(instrument)

    quote = asyncio.run(scenario())

    assert quote.symbol == "AAPL"
    assert quote.con_id == 265598
    assert quote.timestamp == timestamp
    assert quote.bid == 230.8
    assert quote.ask == 230.9
    assert quote.last == 230.85
    assert quote.close == 229.7
    assert client.ticker_request.conId == 265598


def test_current_quote_with_only_a_previous_close_fails_clearly() -> None:
    client = FakeIbClient(
        accounts=["DU123456"],
        ticker_result=[
            SimpleNamespace(
                time=None,
                bid=float("nan"),
                ask=-1.0,
                last=float("nan"),
                close=229.7,
            )
        ],
    )
    connection = IbkrConnection(
        IbkrConfig(
            environment=Environment.PAPER,
            host="127.0.0.1",
            port=4002,
            client_id=21,
        ),
        client=client,
    )
    instrument = QualifiedInstrument(
        symbol="AAPL",
        con_id=265598,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
    )

    async def scenario() -> None:
        await connection.connect()
        await connection.current_quote(instrument)

    with pytest.raises(IbkrError, match="contained no current bid, ask, or last"):
        asyncio.run(scenario())

    assert connection.is_connected is True


def test_ibkr_diagnostic_runs_the_read_only_data_path_and_always_disconnects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_config = tmp_path / "run.yaml"
    run_config.write_text(
        """
run_id: stage2_check
universe: NASDAQ
strategy: SESSION_HARD
environment: PAPER
""",
        encoding="utf-8",
    )
    ibkr_config = tmp_path / "ibkr.yaml"
    ibkr_config.write_text(
        """
PAPER:
  environment: PAPER
  host: 127.0.0.1
  port: 4002
  client_id: 21
LIVE:
  environment: LIVE
  host: 127.0.0.1
  port: 4001
  client_id: 22
""",
        encoding="utf-8",
    )

    class DiagnosticConnection:
        instances: ClassVar[list["DiagnosticConnection"]] = []

        def __init__(self, config: IbkrConfig) -> None:
            self.config = config
            self.disconnected = False
            self.instances.append(self)

        async def connect(self) -> BrokerSession:
            return BrokerSession(Environment.PAPER, "DU123456", True)

        async def resolve_stock(self, *args: object, **kwargs: object) -> QualifiedInstrument:
            return QualifiedInstrument("AAPL", 265598, "SMART", "NASDAQ", "USD", "STK")

        async def historical_bars(
            self, *args: object, **kwargs: object
        ) -> tuple[HistoricalBar, ...]:
            return (
                HistoricalBar(
                    datetime(2026, 9, 1, 14, 30, tzinfo=UTC),
                    230.1,
                    231.25,
                    229.8,
                    230.9,
                    12345.0,
                ),
            )

        async def current_quote(self, *args: object, **kwargs: object) -> CurrentQuote:
            return CurrentQuote(
                "AAPL",
                265598,
                datetime(2026, 9, 1, 14, 31, tzinfo=UTC),
                230.8,
                230.9,
                230.85,
                229.7,
            )

        def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr("stocker_execution.ibkr.IbkrConnection", DiagnosticConnection)

    result = CliRunner().invoke(
        app,
        [
            "ibkr-check",
            "--run-config",
            str(run_config),
            "--ibkr-config",
            str(ibkr_config),
        ],
    )

    assert result.exit_code == 0
    assert "IBKR PAPER connection established" in result.stdout
    assert "Account: DU***456" in result.stdout
    assert "AAPL resolved: conId=265598" in result.stdout
    assert "Historical bars received: 1" in result.stdout
    assert "Current data received" in result.stdout
    assert "IBKR disconnected" in result.stdout
    assert DiagnosticConnection.instances[0].config.environment is Environment.PAPER
    assert DiagnosticConnection.instances[0].disconnected is True
