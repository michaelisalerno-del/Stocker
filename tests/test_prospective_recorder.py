from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from stocker_prospective.config import ProspectiveConfig
from stocker_prospective.database import ProspectiveRepository
from stocker_prospective.market_data import (
    BoundedRealtimeBarQueue,
    BoundedStreamQuoteCache,
    CallbackResult,
    ConnectionTracker,
    MarketDataBudget,
    MarketDataType,
    RealtimeBarUpdate,
)
from stocker_prospective.recorder import (
    IBKRDiagnosticRecorder,
    RecorderDeploymentIdentity,
)

SYMBOLS = (
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
)
START = datetime(2026, 7, 24, 14, 0, tzinfo=UTC)


def _config(tmp_path: Path) -> ProspectiveConfig:
    return ProspectiveConfig.model_validate(
        {
            "paths": {
                "database": str(tmp_path / "prospective.sqlite3"),
                "bundle_root": str(tmp_path / "bundles"),
                "feature_parity_report": str(tmp_path / "parity.json"),
            },
            "runtime": {
                "mode": "record_only",
                "source": "ibkr",
                "prospective_start_utc": START.isoformat(),
                "instance_id": "recorder-test",
                "app_version": "0.1.0-test",
                "git_commit": "deadbeef",
                "run_id": "ibkr-diagnostic-test",
            },
            "risk": {"trading_enabled": False},
            "web": {"host": "127.0.0.1"},
            "ibkr": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 71,
                "expected_environment": "paper",
                "market_data_line_budget": 100,
                "reserved_line_headroom": 10,
                "request_rate_per_second": 20,
            },
            "context": {
                "mode": "signed_import",
                "hmac_secret_env": "STOCKER_CONTEXT_SIGNING_SECRET",
            },
        }
    )


class FakeAdapter:
    def __init__(self, *, missing_symbol: str | None = None) -> None:
        self.missing_symbol = missing_symbol
        self.config = SimpleNamespace(
            allowed_market_data_types=(MarketDataType.LIVE,),
        )
        self.connection = ConnectionTracker()
        self.budget = MarketDataBudget(
            line_limit=100,
            reserved_headroom=10,
            request_rate_limit=20,
        )
        self.connection.connected(MarketDataType.LIVE)
        self.realtime_bars = BoundedRealtimeBarQueue(max_items=256)
        self.stream_quotes = BoundedStreamQuoteCache(
            max_subscriptions=64,
            max_fields_per_subscription=16,
        )
        self.qualifications = 0
        self.quote_requests: dict[str, int] = {}
        self.bar_requests: dict[str, int] = {}
        self.cancelled_quotes: list[int] = []
        self.cancelled_bars: list[int] = []
        self.reconnect_count = 0
        self._next_request = 100

    def _request_id(self) -> int:
        result = self._next_request
        self._next_request += 1
        return result

    def qualify_exact_contract(self, contract: dict[str, object]) -> CallbackResult:
        self.qualifications += 1
        symbol = str(contract["symbol"])
        items: tuple[object, ...] = ()
        if symbol != self.missing_symbol:
            items = (
                {
                    **contract,
                    "conId": 1000 + SYMBOLS.index(symbol),
                    "localSymbol": symbol,
                },
            )
        return CallbackResult(
            request_id=self._request_id(),
            kind="exact_contract_qualification",
            items=items,
            complete=True,
            error=None,
        )

    def request_market_data(
        self,
        contract: dict[str, object],
        *,
        subscription_key: str,
    ) -> int:
        request_id = self._request_id()
        self.quote_requests[subscription_key] = request_id
        self.stream_quotes.register(request_id)
        return request_id

    def request_realtime_bars(
        self,
        contract: dict[str, object],
        *,
        subscription_key: str,
    ) -> int:
        request_id = self._request_id()
        self.bar_requests[subscription_key] = request_id
        return request_id

    def cancel_market_data(self, request_id: int, *, subscription_key: str) -> None:
        self.cancelled_quotes.append(request_id)
        self.stream_quotes.remove(request_id)

    def cancel_realtime_bars(self, request_id: int, *, subscription_key: str) -> None:
        self.cancelled_bars.append(request_id)

    def reconnect(self) -> None:
        self.reconnect_count += 1
        self.connection.connected(None)


def _contract(symbol: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "secType": "STK",
        "exchange": "SMART",
        "currency": "USD",
    }


def _recorder(
    tmp_path: Path,
    *,
    missing_symbol: str | None = None,
) -> tuple[IBKRDiagnosticRecorder, ProspectiveRepository, FakeAdapter]:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    adapter = FakeAdapter(missing_symbol=missing_symbol)
    recorder = IBKRDiagnosticRecorder(
        config=_config(tmp_path),
        repository=repository,
        adapter=adapter,  # type: ignore[arg-type]
        identity=RecorderDeploymentIdentity(
            model_artifact_id="verified-frozen-test",
            universe_id="anchor-frozen-20-test",
            universe_hash="a" * 64,
            symbols=SYMBOLS,
            bundle_verified=True,
        ),
        contract_factory=_contract,
        sleep=lambda _seconds: None,
    )
    return recorder, repository, adapter


def test_recorder_creates_no_prospective_evidence_before_configured_start(
    tmp_path: Path,
) -> None:
    recorder, repository, adapter = _recorder(tmp_path)

    recorder.poll(now=START - timedelta(seconds=1))

    assert recorder.initialized is False
    assert adapter.qualifications == 0
    with repository._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM prospective_run").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM evidence_envelope").fetchone()[0] == 0


def test_recorder_registers_every_symbol_and_persists_completed_diagnostic_bar(
    tmp_path: Path,
) -> None:
    recorder, repository, adapter = _recorder(tmp_path)
    recorder.poll(now=START)

    quote_request = adapter.quote_requests["underlying:AAL:quote"]
    quote_time = datetime(2026, 7, 24, 14, 35, 1, tzinfo=UTC)
    for field, value in (
        ("bid", 12.00),
        ("ask", 12.04),
        ("bid_size", 5),
        ("ask_size", 7),
        ("last", 12.02),
        ("last_size", 2),
    ):
        adapter.stream_quotes.add(
            quote_request,
            {
                "field": field,
                "value": value,
                "market_data_type": "live",
                "receive_timestamp_utc": quote_time.isoformat(),
            },
        )
    bar_request = adapter.bar_requests["underlying:AAL:bars"]
    bar_start = datetime(2026, 7, 24, 14, 30, tzinfo=UTC)
    for index in range(61):
        timestamp = bar_start + timedelta(seconds=5 * index)
        adapter.realtime_bars.add(
            RealtimeBarUpdate(
                request_id=bar_request,
                source_timestamp_utc=timestamp,
                receive_timestamp_utc=timestamp + timedelta(seconds=1),
                open=12.0,
                high=12.1,
                low=11.9,
                close=12.02,
                volume=10.0,
                wap=12.01,
                trade_count=2,
            )
        )
    recorder.poll(now=quote_time + timedelta(seconds=1))

    with repository._connect() as connection:
        memberships = connection.execute(
            "SELECT symbol, operational_status FROM universe_membership ORDER BY symbol"
        ).fetchall()
        bar = connection.execute("SELECT * FROM underlying_bar").fetchone()
        quote = connection.execute("SELECT * FROM underlying_quote").fetchone()
        assert len(memberships) == 20
        assert {row["symbol"] for row in memberships} == set(SYMBOLS)
        assert {row["operational_status"] for row in memberships} == {"recording_diagnostics"}
        assert bar is not None
        assert bar["completeness"] == "complete"
        assert bar["eligibility"] == 0
        assert bar["rejection_reason"] == "blocked_feature_source_semantics_mismatch"
        assert quote is not None
        assert quote["symbol"] == "AAL"
        assert quote["con_id"] == 1000
        assert quote["bid"] == 12.0
        assert quote["ask"] == 12.04
        assert quote["midpoint"] == 12.02
        assert quote["capture_status"] == "recorded_live"

    recorder.shutdown(now=quote_time + timedelta(seconds=2))
    assert len(adapter.cancelled_quotes) == 20
    assert len(adapter.cancelled_bars) == 20


def test_missing_exact_contract_is_an_explicit_anchor_rejection(tmp_path: Path) -> None:
    recorder, repository, _adapter = _recorder(tmp_path, missing_symbol="AAL")

    recorder.poll(now=START)

    with repository._connect() as connection:
        member = connection.execute(
            "SELECT operational_status, rejection_reason FROM universe_membership "
            "WHERE symbol = 'AAL'"
        ).fetchone()
        contract = connection.execute(
            "SELECT qualification_status, rejection_reason FROM underlying_contract "
            "WHERE symbol = 'AAL'"
        ).fetchone()
        assert member is not None
        assert member["operational_status"] == "rejected_contract_qualification"
        assert member["rejection_reason"] == "missing_exact_underlying_contract"
        assert contract is not None
        assert contract["qualification_status"] == "rejected"
        assert contract["rejection_reason"] == "missing_exact_underlying_contract"


def test_only_lost_data_reconnect_rebuilds_subscriptions(tmp_path: Path) -> None:
    recorder, _repository, adapter = _recorder(tmp_path)
    recorder.poll(now=START)
    initial_quotes = len(adapter.quote_requests)
    initial_bars = len(adapter.bar_requests)

    adapter.connection.connection_restored(data_maintained=True, code=1102)
    recorder.poll(now=START + timedelta(seconds=1))
    assert len(adapter.quote_requests) == initial_quotes
    assert len(adapter.bar_requests) == initial_bars

    adapter.connection.connection_restored(data_maintained=False, code=1101)
    recorder.poll(now=START + timedelta(seconds=2))
    assert len(adapter.quote_requests) == initial_quotes
    assert len(adapter.bar_requests) == initial_bars
    assert adapter.connection.health().subscriptions_require_rebuild is False

    with sqlite3.connect(tmp_path / "prospective.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_event WHERE event_type = 'ibkr_subscriptions_rebuilt'"
        ).fetchone() == (1,)

    adapter.connection.connection_lost(
        code=1100,
        message="official_socket_connection_closed",
    )
    recorder.poll(now=START + timedelta(seconds=3))
    assert adapter.reconnect_count == 1
    with sqlite3.connect(tmp_path / "prospective.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_event WHERE event_type = 'ibkr_socket_reconnected'"
        ).fetchone() == (1,)
