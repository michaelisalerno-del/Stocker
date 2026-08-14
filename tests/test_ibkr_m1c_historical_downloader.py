from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tools.research.download_ibkr_m1c_microstructure import (
    DEFAULT_CANONICAL_SOURCE,
    ConservativeHistoricalPacer,
    ContractIdentity,
    DownloadClient,
    HistoricalCallbackRegistry,
    HistoricalDataOnlyFacade,
    HistoricalEvent,
    HistoricalPage,
    HistoricalRequestError,
    assign_provider_identities,
    build_event_metadata,
    classify_historical_error,
    create_historical_stock_contract,
    download_feed_pages,
    event_output_directory,
    event_window,
    load_frozen_events,
    merge_provider_pages,
    normalize_bid_ask_tick,
    normalize_trade_tick,
    qualify_exact_stock,
    resume_completed_feeds,
    run_download,
    select_validation_events,
    sha256_file,
    write_feed_parquet,
    write_json_atomic,
)


@dataclass
class _LastAttributes:
    pastLimit: bool = False
    unreported: bool = True


@dataclass
class _TradeTick:
    time: int
    price: float
    size: str
    exchange: str
    specialConditions: str
    tickAttribLast: _LastAttributes


@dataclass
class _BidAskAttributes:
    bidPastLow: bool = True
    askPastHigh: bool = False


@dataclass
class _BidAskTick:
    time: int
    priceBid: float
    priceAsk: float
    sizeBid: str
    sizeAsk: str
    tickAttribBidAsk: _BidAskAttributes


def _event() -> HistoricalEvent:
    return HistoricalEvent(
        period="development",
        event_id="AAPL|2024-01-03|6",
        symbol="AAPL",
        session_date="2024-01-03",
        family="HARD_M1C",
        probability=0.61,
        checkpoint=6,
        t0_utc=datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
    )


def _contract() -> ContractIdentity:
    return ContractIdentity(
        con_id=265598,
        symbol="AAPL",
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        local_symbol="AAPL",
        primary_exchange="NASDAQ",
    )


def test_trade_normalisation_preserves_identical_ticks_at_one_timestamp() -> None:
    tick = _TradeTick(
        time=1704294000,
        price=185.25,
        size="100",
        exchange="NASDAQ",
        specialConditions="",
        tickAttribLast=_LastAttributes(),
    )

    rows = [normalize_trade_tick(tick, _event(), _contract()) for _ in range(2)]
    identified = assign_provider_identities(rows, feed="TRADES", event_id=_event().event_id)

    assert [row["provider_timestamp_utc"] for row in identified] == [
        datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
        datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
    ]
    assert [row["provider_content_occurrence"] for row in identified] == [0, 1]
    assert identified[0]["provider_event_identity"] != identified[1]["provider_event_identity"]
    assert identified[0]["tick_attrib_unreported"] is True
    assert identified[0]["source"] == "ibkr_historical_ticks"
    assert identified[0]["feed"] == "TRADES"


def test_bid_ask_normalisation_preserves_provider_flags() -> None:
    row = normalize_bid_ask_tick(
        _BidAskTick(
            time=1704294001,
            priceBid=185.20,
            priceAsk=185.22,
            sizeBid="300",
            sizeAsk="125",
            tickAttribBidAsk=_BidAskAttributes(),
        ),
        _event(),
        _contract(),
    )

    assert row["provider_timestamp_utc"] == datetime(2024, 1, 3, 15, 0, 1, tzinfo=UTC)
    assert row["bid"] == 185.20
    assert row["ask"] == 185.22
    assert row["bid_size"] == 300.0
    assert row["ask_size"] == 125.0
    assert row["tick_attrib_bid_past_low"] is True
    assert row["tick_attrib_ask_past_high"] is False
    assert row["feed"] == "BID_ASK"


def test_pagination_removes_only_the_exact_sequence_overlap() -> None:
    def row(timestamp: int, price: float, size: float) -> dict[str, object]:
        return {
            "provider_epoch_seconds": timestamp,
            "price": price,
            "size": size,
            "feed": "TRADES",
        }

    first = [row(1, 10.0, 1.0), row(2, 10.1, 2.0), row(2, 10.1, 2.0)]
    second = [row(2, 10.1, 2.0), row(2, 10.1, 2.0), row(3, 10.2, 1.0)]

    merged, removed = merge_provider_pages(first, second)

    assert removed == 2
    assert [(item["provider_epoch_seconds"], item["size"]) for item in merged] == [
        (1, 1.0),
        (2, 2.0),
        (2, 2.0),
        (3, 1.0),
    ]


def test_frozen_event_loading_separates_periods_and_excludes_2026(tmp_path: Path) -> None:
    threshold = 0.4883337107940334
    source = tmp_path / "events.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": ["dev-hard", "dev-p01", "dev-r01", "ass-hard", "protected"],
                "stock": ["AAPL", "MSFT", "NVDA", "AMZN", "META"],
                "session": [
                    "2024-01-03",
                    "2024-02-05",
                    "2024-03-06",
                    "2025-08-22",
                    "2026-01-02",
                ],
                "partition": [
                    "development",
                    "development",
                    "development",
                    "assessment",
                    "stress",
                ],
                "checkpoint": [6, 8, 10, 6, 6],
                "signal_timestamp": [
                    datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
                    datetime(2024, 2, 5, 15, 10, tzinfo=UTC),
                    datetime(2024, 3, 6, 15, 20, tzinfo=UTC),
                    datetime(2025, 8, 22, 14, 0, tzinfo=UTC),
                    datetime(2026, 1, 2, 15, 0, tzinfo=UTC),
                ],
                "M1C_probability": [
                    threshold,
                    threshold * 0.9,
                    threshold * 0.7,
                    threshold * 1.1,
                    threshold * 1.2,
                ],
                "m1c_high_tail_threshold_v1": [threshold] * 5,
                "m1c_high_tail_v1": [True, False, False, True, True],
            }
        ),
        source,
    )

    default_events = load_frozen_events(source, period="development", include_r01=False)
    with_r01 = load_frozen_events(source, period="development", include_r01=True)
    assessment = load_frozen_events(source, period="assessment", include_r01=True)

    assert [event.family for event in default_events] == ["HARD_M1C", "P01_NEAR_M1"]
    assert [event.family for event in with_r01] == [
        "HARD_M1C",
        "P01_NEAR_M1",
        "R01",
    ]
    assert [event.event_id for event in assessment] == ["ass-hard"]
    assert all(event.t0_utc.year != 2026 for event in [*with_r01, *assessment])
    assert event_window(default_events[0]) == (
        default_events[0].t0_utc - timedelta(minutes=5),
        default_events[0].t0_utc + timedelta(minutes=20),
    )


def test_authoritative_source_hash_and_validation_sample_are_frozen() -> None:
    events = load_frozen_events(
        DEFAULT_CANONICAL_SOURCE,
        period="development",
        include_r01=False,
    )
    sample = select_validation_events(events, count=5, family="HARD_M1C")

    assert sha256_file(DEFAULT_CANONICAL_SOURCE) == (
        "8dd0ef53d9c5493b70f600a28d6f77e8ffabd5e7b48a5378cf0bb4411382cb8f"
    )
    assert len(events) == 2756
    assert len(sample) == 5
    assert {event.family for event in sample} == {"HARD_M1C"}
    assert len({event.symbol for event in sample}) == 5


@dataclass
class _OfficialContract:
    conId: int
    symbol: str
    secType: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    localSymbol: str = "AAPL"
    primaryExchange: str = "NASDAQ"


def test_contract_qualification_requires_one_exact_us_stock() -> None:
    contract = _OfficialContract(conId=265598, symbol="AAPL")
    unrelated = _OfficialContract(conId=272093, symbol="MSFT")

    identity, retained = qualify_exact_stock("AAPL", [unrelated, contract])

    assert retained is contract
    assert identity == _contract()

    for candidates in (
        [],
        [contract, contract],
        [_OfficialContract(conId=0, symbol="AAPL")],
        [_OfficialContract(conId=265598, symbol="AAPL", secType="OPT")],
        [_OfficialContract(conId=265598, symbol="MSFT")],
        [_OfficialContract(conId=265598, symbol="AAPL", currency="GBP")],
    ):
        try:
            qualify_exact_stock("AAPL", candidates)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ambiguous/invalid stock contract accepted: {candidates}")


class _PagedClient:
    def __init__(self, pages: list[HistoricalPage | Exception]) -> None:
        self.pages = list(pages)
        self.starts: list[datetime] = []
        self.request_count = 0

    def request_historical_ticks(
        self,
        *,
        contract: object,
        feed: str,
        start_utc: datetime,
        number_of_ticks: int,
        use_rth: bool,
        timeout_seconds: float,
    ) -> HistoricalPage:
        del contract, feed, number_of_ticks, use_rth, timeout_seconds
        self.request_count += 1
        self.starts.append(start_utc)
        result = self.pages.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _trade(timestamp: int, price: float) -> _TradeTick:
    return _TradeTick(
        time=timestamp,
        price=price,
        size="1",
        exchange="NASDAQ",
        specialConditions="",
        tickAttribLast=_LastAttributes(),
    )


def test_feed_pagination_advances_by_a_complete_second_and_terminates_at_window() -> None:
    start, _ = event_window(_event())
    client = _PagedClient(
        [
            HistoricalPage(
                "TRADES",
                (
                    _trade(int(start.timestamp()), 10.0),
                    _trade(int(start.timestamp()) + 1, 10.1),
                ),
                True,
            ),
            HistoricalPage("TRADES", (_trade(int(start.timestamp()) + 2, 10.2),), True),
        ]
    )

    result = download_feed_pages(
        client=client,
        provider_contract=object(),
        event=_event(),
        contract=_contract(),
        feed="TRADES",
        requested_start_utc=start,
        requested_end_utc=start + timedelta(seconds=2),
        page_size=2,
        request_timeout_seconds=1,
        max_pages=10,
        max_retries=2,
        retry_backoff_seconds=0,
        sleeper=lambda _: None,
    )

    assert client.starts == [start, start + timedelta(seconds=2)]
    assert result.completion_status == "COMPLETE"
    assert result.request_count == 2
    assert result.raw_rows == 3
    assert result.final_rows == 3
    assert [row["price"] for row in result.rows] == [10.0, 10.1, 10.2]


def test_pacing_errors_retry_but_permission_errors_fail_closed() -> None:
    start, _ = event_window(_event())
    pacing = HistoricalRequestError(
        kind="pacing",
        code=162,
        message="Historical data request pacing violation",
    )
    client = _PagedClient([pacing, HistoricalPage("TRADES", (), True)])
    sleeps: list[float] = []

    result = download_feed_pages(
        client=client,
        provider_contract=object(),
        event=_event(),
        contract=_contract(),
        feed="TRADES",
        requested_start_utc=start,
        requested_end_utc=start + timedelta(minutes=1),
        page_size=1000,
        request_timeout_seconds=1,
        max_pages=10,
        max_retries=2,
        retry_backoff_seconds=2,
        sleeper=sleeps.append,
    )

    assert result.request_count == 2
    assert sleeps == [2]
    assert classify_historical_error(354, "not subscribed").kind == "permission"
    assert (
        classify_historical_error(162, "HMDS query returned no market data permissions").kind
        == "permission"
    )

    permission_client = _PagedClient(
        [HistoricalRequestError(kind="permission", code=354, message="not subscribed")]
    )
    try:
        download_feed_pages(
            client=permission_client,
            provider_contract=object(),
            event=_event(),
            contract=_contract(),
            feed="TRADES",
            requested_start_utc=start,
            requested_end_utc=start + timedelta(minutes=1),
            page_size=1000,
            request_timeout_seconds=1,
            max_pages=10,
            max_retries=2,
            retry_backoff_seconds=0,
            sleeper=lambda _: None,
        )
    except HistoricalRequestError as error:
        assert error.kind == "permission"
        assert error.request_count == 1
    else:
        raise AssertionError("permission failure was retried or ignored")
    assert permission_client.request_count == 1


def test_parquet_output_is_deterministic_and_resume_validates_interval(tmp_path: Path) -> None:
    tick = _trade(1704294000, 185.25)
    rows = assign_provider_identities(
        [normalize_trade_tick(tick, _event(), _contract())],
        feed="TRADES",
        event_id=_event().event_id,
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"

    write_feed_parquet(first, feed="TRADES", rows=rows)
    write_feed_parquet(second, feed="TRADES", rows=rows)

    assert sha256_file(first) == sha256_file(second)
    table = pq.read_table(first)
    assert table.schema.field("provider_timestamp_utc").type == pa.timestamp("us", tz="UTC")
    assert table.column("provider_event_identity").to_pylist() == [
        rows[0]["provider_event_identity"]
    ]

    event_dir = event_output_directory(tmp_path / "dataset", _event())
    output = event_dir / "trades.parquet"
    write_feed_parquet(output, feed="TRADES", rows=rows)
    start, end = event_window(_event())
    download = download_feed_pages(
        client=_PagedClient([HistoricalPage("TRADES", (tick,), True)]),
        provider_contract=object(),
        event=_event(),
        contract=_contract(),
        feed="TRADES",
        requested_start_utc=start,
        requested_end_utc=end,
        sleeper=lambda _: None,
    )
    metadata = build_event_metadata(
        event=_event(),
        contract=_contract(),
        requested_start_utc=start,
        requested_end_utc=end,
        feeds={"TRADES": download},
        ibkr_api_version="10.49.1",
        server_version=180,
        tws_gateway_version="10.49",
        git_commit="abc123",
        downloaded_at_utc=datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
        canonical_source_sha256="source-sha",
        family_definition_sha256="family-sha",
    )
    write_json_atomic(event_dir / "metadata.json", metadata)

    assert resume_completed_feeds(
        event_dir,
        event=_event(),
        requested_start_utc=start,
        requested_end_utc=end,
        canonical_source_sha256="source-sha",
        family_definition_sha256="family-sha",
    ) == {"TRADES"}
    assert (
        resume_completed_feeds(
            event_dir,
            event=_event(),
            requested_start_utc=start - timedelta(minutes=1),
            requested_end_utc=end,
            canonical_source_sha256="source-sha",
            family_definition_sha256="family-sha",
        )
        == set()
    )
    valid_table = pq.read_table(output)
    pq.write_table(valid_table.drop_columns(["price"]), output)
    assert (
        resume_completed_feeds(
            event_dir,
            event=_event(),
            requested_start_utc=start,
            requested_end_utc=end,
            canonical_source_sha256="source-sha",
            family_definition_sha256="family-sha",
        )
        == set()
    )
    timestamp_index = valid_table.schema.get_field_index("provider_timestamp_utc")
    null_timestamps = pa.array([None] * valid_table.num_rows, type=pa.timestamp("us", tz="UTC"))
    pq.write_table(
        valid_table.set_column(timestamp_index, "provider_timestamp_utc", null_timestamps),
        output,
    )
    assert (
        resume_completed_feeds(
            event_dir,
            event=_event(),
            requested_start_utc=start,
            requested_end_utc=end,
            canonical_source_sha256="source-sha",
            family_definition_sha256="family-sha",
        )
        == set()
    )
    changed_event = replace(_event(), probability=0.72)
    assert (
        resume_completed_feeds(
            event_dir,
            event=changed_event,
            requested_start_utc=start,
            requested_end_utc=end,
            canonical_source_sha256="source-sha",
            family_definition_sha256="family-sha",
        )
        == set()
    )


def test_development_and_assessment_output_directories_are_physically_separate(
    tmp_path: Path,
) -> None:
    assessment = HistoricalEvent(
        period="assessment",
        event_id="AAPL|2025-01-03|6",
        symbol="AAPL",
        session_date="2025-01-03",
        family="HARD_M1C",
        probability=0.61,
        checkpoint=6,
        t0_utc=datetime(2025, 1, 3, 15, 0, tzinfo=UTC),
    )

    development_dir = event_output_directory(tmp_path, _event())
    assessment_dir = event_output_directory(tmp_path, assessment)

    assert development_dir == tmp_path / "development" / "AAPL" / _event().event_id
    assert assessment_dir == tmp_path / "assessment" / "AAPL" / assessment.event_id
    assert development_dir != assessment_dir


def test_output_directory_rejects_absolute_or_nested_source_identities(tmp_path: Path) -> None:
    for event_id in ("/tmp/escape", "nested/event", "..", r"nested\event"):
        with pytest.raises(ValueError, match="unsafe event ID"):
            event_output_directory(tmp_path, replace(_event(), event_id=event_id))
    with pytest.raises(ValueError, match="unsafe event symbol"):
        event_output_directory(tmp_path, replace(_event(), symbol="nested/AAPL"))


def test_verified_historical_contract_factory_matches_stocker_stock_contract() -> None:
    class Contract:
        pass

    contract = create_historical_stock_contract(Contract, "AAPL")

    assert contract.symbol == "AAPL"
    assert contract.secType == "STK"
    assert contract.exchange == "SMART"
    assert contract.currency == "USD"


def test_official_historical_callbacks_route_all_three_tick_shapes() -> None:
    registry = HistoricalCallbackRegistry(max_pending=2, max_ticks_per_request=1000)
    registry.begin(1, expected_feed="TRADES")
    trade = _trade(1704294000, 185.25)
    registry.deliver(1, callback_feed="TRADES", ticks=(trade,), done=True)
    assert registry.wait(1, timeout_seconds=0.01) == HistoricalPage("TRADES", (trade,), True)

    registry.begin(2, expected_feed="BID_ASK")
    registry.deliver(2, callback_feed="MIDPOINT", ticks=(), done=True)
    try:
        registry.wait(2, timeout_seconds=0.01)
    except HistoricalRequestError as error:
        assert error.kind == "callback"
        assert "MIDPOINT" in error.message
    else:
        raise AssertionError("unexpected MIDPOINT callback was accepted")


class _RawOfficialClient:
    def connect(self, host: str, port: int, client_id: int) -> tuple[str, int, int]:
        return host, port, client_id

    def disconnect(self) -> None:
        return None

    def run(self) -> None:
        return None

    def reqContractDetails(self, request_id: int, contract: object) -> None:
        self.contract_request = (request_id, contract)

    def reqHistoricalTicks(self, *arguments: object) -> None:
        self.historical_request = arguments

    def serverVersion(self) -> int:
        return 180


def test_narrow_historical_facade_exposes_no_order_account_or_execution_surface() -> None:
    facade = HistoricalDataOnlyFacade(_RawOfficialClient())
    public = {name for name in dir(facade) if not name.startswith("_")}

    assert public == {
        "connect",
        "disconnect",
        "reqContractDetails",
        "reqHistoricalTicks",
        "run",
        "serverVersion",
    }
    assert not {
        "placeOrder",
        "cancelOrder",
        "exerciseOptions",
        "reqGlobalCancel",
        "reqAccountSummary",
        "reqPositions",
        "reqExecutions",
        "reqPnL",
    }.intersection(public)


class _CompleteDownloadClient:
    api_version = "10.49.1"
    server_version = 180
    tws_gateway_version = "10.49"

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self.qualifications: list[str] = []
        self.closed = False

    def qualify_symbol(
        self, symbol: str, *, timeout_seconds: float
    ) -> tuple[ContractIdentity, object]:
        del timeout_seconds
        self.qualifications.append(symbol)
        return _contract(), object()

    def request_historical_ticks(
        self,
        *,
        contract: object,
        feed: str,
        start_utc: datetime,
        number_of_ticks: int,
        use_rth: bool,
        timeout_seconds: float,
    ) -> HistoricalPage:
        del contract, number_of_ticks, use_rth, timeout_seconds
        self.requests.append((feed, start_utc.isoformat()))
        if feed == "TRADES":
            tick: object = _trade(1704294000, 185.25)
        else:
            tick = _BidAskTick(
                time=1704294000,
                priceBid=185.20,
                priceAsk=185.22,
                sizeBid="300",
                sizeAsk="125",
                tickAttribBidAsk=_BidAskAttributes(),
            )
        return HistoricalPage(feed, (tick,), True)

    def close(self) -> None:
        self.closed = True


def test_complete_run_writes_manifest_summary_and_resumes_without_redownload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "events.parquet"
    threshold = 0.4883337107940334
    pq.write_table(
        pa.table(
            {
                "row_id": [_event().event_id],
                "stock": [_event().symbol],
                "session": [_event().session_date],
                "partition": [_event().period],
                "checkpoint": [_event().checkpoint],
                "signal_timestamp": [_event().t0_utc],
                "M1C_probability": [threshold * 1.1],
                "m1c_high_tail_threshold_v1": [threshold],
                "m1c_high_tail_v1": [True],
            }
        ),
        source,
    )
    output_root = tmp_path / "ibkr-m1c-microstructure-v0"
    first_client = _CompleteDownloadClient()

    first_result = run_download(
        period="development",
        source=source,
        output_root=output_root,
        include_r01=False,
        resume=False,
        validation_hard_count=None,
        event_limit=None,
        client_factory=lambda: first_client,
        request_timeout_seconds=1,
        max_retries=1,
        retry_backoff_seconds=0,
        before_request=None,
        git_commit="abc123",
        now=lambda: datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
    )

    event_dir = event_output_directory(output_root, _event())
    metadata = __import__("json").loads((event_dir / "metadata.json").read_text())
    manifest = __import__("json").loads((output_root / "download_manifest.json").read_text())

    assert first_result["complete_events"] == 1
    assert first_client.qualifications == ["AAPL"]
    assert [request[0] for request in first_client.requests] == ["TRADES", "BID_ASK"]
    assert first_client.closed is True
    assert (event_dir / "trades.parquet").is_file()
    assert (event_dir / "bid_ask.parquet").is_file()
    assert metadata["TRADES"]["final_rows"] == 1
    assert metadata["BID_ASK"]["final_rows"] == 1
    assert manifest["canonical_M1C_source_sha256"] == sha256_file(source)
    assert (
        manifest["family_definition"]["authoritative_source"]["git_blob"]
        == "b78bc00191dd878460b09b5348ffff418c656376"
    )
    assert manifest["family_definition"]["rules"]["P01_NEAR_M1"] == "0.8T < score <= T"
    assert manifest["direction_analysis_performed"] is False
    assert manifest["no_order_invariant"] == "PASS"
    summary = (output_root / "download_summary.csv").read_text()
    assert "development,AAPL|2024-01-03|6,AAPL,TRADES" in summary
    assert "development,AAPL|2024-01-03|6,AAPL,BID_ASK" in summary

    resume_client = _CompleteDownloadClient()
    resume_result = run_download(
        period="development",
        source=source,
        output_root=output_root,
        include_r01=False,
        resume=True,
        validation_hard_count=None,
        event_limit=None,
        client_factory=lambda: resume_client,
        request_timeout_seconds=1,
        max_retries=1,
        retry_backoff_seconds=0,
        before_request=None,
        git_commit="abc123",
        now=lambda: datetime(2026, 8, 14, 12, 1, tzinfo=UTC),
    )

    assert resume_result["skipped_events"] == 1
    assert resume_client.qualifications == []
    assert resume_client.requests == []
    assert resume_client.closed is False

    with pytest.raises(ValueError, match="incompatible provenance"):
        run_download(
            period="development",
            source=source,
            output_root=output_root,
            include_r01=True,
            resume=True,
            validation_hard_count=None,
            event_limit=None,
            client_factory=lambda: _CompleteDownloadClient(),
            request_timeout_seconds=1,
            max_retries=1,
            retry_backoff_seconds=0,
            before_request=None,
            git_commit="abc123",
            now=lambda: datetime(2026, 8, 14, 12, 1, tzinfo=UTC),
        )


class _PermissionFailureClient(_CompleteDownloadClient):
    def request_historical_ticks(
        self,
        *,
        contract: object,
        feed: str,
        start_utc: datetime,
        number_of_ticks: int,
        use_rth: bool,
        timeout_seconds: float,
    ) -> HistoricalPage:
        if feed == "BID_ASK":
            self.requests.append((feed, start_utc.isoformat()))
            raise HistoricalRequestError(kind="permission", code=354, message="not subscribed")
        return super().request_historical_ticks(
            contract=contract,
            feed=feed,
            start_utc=start_utc,
            number_of_ticks=number_of_ticks,
            use_rth=use_rth,
            timeout_seconds=timeout_seconds,
        )


class _ContractFailureClient(_CompleteDownloadClient):
    def qualify_symbol(
        self, symbol: str, *, timeout_seconds: float
    ) -> tuple[ContractIdentity, object]:
        del symbol, timeout_seconds
        raise HistoricalRequestError(kind="contract", code=200, message="ambiguous contract")


def test_contract_block_writes_both_required_summary_feed_rows(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": [_event().event_id],
                "stock": [_event().symbol],
                "session": [_event().session_date],
                "partition": [_event().period],
                "checkpoint": [_event().checkpoint],
                "signal_timestamp": [_event().t0_utc],
                "M1C_probability": [0.61],
                "m1c_high_tail_threshold_v1": [0.4883337107940334],
                "m1c_high_tail_v1": [True],
            }
        ),
        source,
    )
    output = tmp_path / "dataset"
    blocked = _ContractFailureClient()
    result = run_download(
        period="development",
        source=source,
        output_root=output,
        include_r01=False,
        resume=False,
        validation_hard_count=None,
        event_limit=None,
        client_factory=lambda: blocked,
        request_timeout_seconds=1,
        max_retries=1,
        retry_backoff_seconds=0,
        before_request=None,
        git_commit="abc123",
        now=lambda: datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
    )

    assert result["blocked_events"] == 1
    with (output / "download_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["feed"] for row in rows] == ["BID_ASK", "TRADES"]
    assert {row["status"] for row in rows} == {"BLOCKED_CONTRACT"}
    assert {row["contract_status"] for row in rows} == {"BLOCKED"}


def test_connection_failure_records_every_selected_event_as_blocked(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    threshold = 0.4883337107940334
    pq.write_table(
        pa.table(
            {
                "row_id": [_event().event_id],
                "stock": [_event().symbol],
                "session": [_event().session_date],
                "partition": [_event().period],
                "checkpoint": [_event().checkpoint],
                "signal_timestamp": [_event().t0_utc],
                "M1C_probability": [0.61],
                "m1c_high_tail_threshold_v1": [threshold],
                "m1c_high_tail_v1": [True],
            }
        ),
        source,
    )
    output = tmp_path / "dataset"

    def unavailable() -> DownloadClient:
        raise RuntimeError("official API unavailable")

    result = run_download(
        period="development",
        source=source,
        output_root=output,
        include_r01=False,
        resume=False,
        validation_hard_count=None,
        event_limit=None,
        client_factory=unavailable,
        request_timeout_seconds=1,
        max_retries=1,
        retry_backoff_seconds=0,
        before_request=None,
        git_commit="abc123",
        now=lambda: datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
    )

    assert result["blocked_events"] == 1
    with (output / "download_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["feed"] for row in rows] == ["BID_ASK", "TRADES"]
    assert {row["status"] for row in rows} == {"BLOCKED_CONNECTION"}
    manifest = json.loads((output / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["blocked_events"] == 1


def test_partial_event_resume_keeps_complete_feed_and_retries_only_blocked_feed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "events.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": [_event().event_id],
                "stock": [_event().symbol],
                "session": [_event().session_date],
                "partition": [_event().period],
                "checkpoint": [_event().checkpoint],
                "signal_timestamp": [_event().t0_utc],
                "M1C_probability": [0.61],
                "m1c_high_tail_threshold_v1": [0.4883337107940334],
                "m1c_high_tail_v1": [True],
            }
        ),
        source,
    )
    output = tmp_path / "dataset"
    blocked = _PermissionFailureClient()
    common = {
        "period": "development",
        "source": source,
        "output_root": output,
        "include_r01": False,
        "validation_hard_count": None,
        "event_limit": None,
        "request_timeout_seconds": 1,
        "max_retries": 1,
        "retry_backoff_seconds": 0,
        "before_request": None,
        "git_commit": "abc123",
        "now": lambda: datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
    }

    first = run_download(
        **common,
        resume=False,
        client_factory=lambda: blocked,
    )
    assert first["partial_events"] == 1
    assert first["blocked_events"] == 0
    assert [request[0] for request in blocked.requests] == ["TRADES", "BID_ASK"]

    def unavailable() -> DownloadClient:
        raise RuntimeError("official API unavailable")

    connection_args = {
        **common,
        "git_commit": "retry-commit",
        "now": lambda: datetime(2026, 8, 14, 12, 30, tzinfo=UTC),
    }
    connection_block = run_download(
        **connection_args,
        resume=True,
        client_factory=unavailable,
    )
    assert connection_block["partial_events"] == 1
    connection_metadata = json.loads(
        (event_output_directory(output, _event()) / "metadata.json").read_text(encoding="utf-8")
    )
    assert connection_metadata["TRADES"]["completion_status"] == "COMPLETE"
    assert connection_metadata["contract_status"] == "PASS"
    assert connection_metadata["contract"]["conId"] == _contract().con_id
    assert connection_metadata["ibkr_api_version"] == "10.49.1"
    assert connection_metadata["git_commit"] == "abc123"
    assert connection_metadata["download_timestamp_utc"] == "2026-08-14T12:00:00+00:00"
    assert (
        connection_metadata["connection_attempt"]["download_timestamp_utc"]
        == "2026-08-14T12:30:00+00:00"
    )
    assert connection_metadata["connection_attempt"]["status"] == "BLOCKED_CONNECTION"

    qualification_block = _ContractFailureClient()
    still_partial = run_download(
        **common,
        resume=True,
        client_factory=lambda: qualification_block,
    )
    assert still_partial["partial_events"] == 1
    event_metadata = json.loads(
        (event_output_directory(output, _event()) / "metadata.json").read_text(encoding="utf-8")
    )
    assert event_metadata["TRADES"]["completion_status"] == "COMPLETE"
    assert event_metadata["BID_ASK"]["completion_status"] == "BLOCKED_CONTRACT"

    resumed = _CompleteDownloadClient()
    second = run_download(
        **common,
        resume=True,
        client_factory=lambda: resumed,
    )
    assert second["complete_events"] == 1
    assert [request[0] for request in resumed.requests] == ["BID_ASK"]


def test_conservative_pacer_enforces_rate_and_window_bounds() -> None:
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    pacer = ConservativeHistoricalPacer(
        requests_per_window=2,
        window_seconds=10,
        request_rate_per_second=1,
        monotonic=lambda: clock[0],
        sleeper=sleep,
    )

    pacer.acquire()
    pacer.acquire()
    pacer.acquire()

    assert clock[0] == 10.0

    clock[0] = 0.0
    weighted = ConservativeHistoricalPacer(
        requests_per_window=60,
        window_seconds=600,
        request_rate_per_second=20,
        monotonic=lambda: clock[0],
        sleeper=sleep,
    )
    weighted.acquire(weight=2)
    for _ in range(4):
        weighted.acquire()
    assert clock[0] == 2.0


def test_source_contains_no_order_api_invocation_or_direction_calculation() -> None:
    source_path = (
        Path(__file__).parents[1] / "tools" / "research" / "download_ibkr_m1c_microstructure.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    error_callbacks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "error"
    ]
    assert [argument.arg for argument in error_callbacks[0].args.args[:5]] == [
        "self",
        "reqId",
        "errorTime",
        "errorCode",
        "errorString",
    ]
    invoked_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert not {
        "placeOrder",
        "cancelOrder",
        "exerciseOptions",
        "reqGlobalCancel",
        "reqAccountSummary",
        "reqAccountUpdates",
        "reqPositions",
        "reqExecutions",
        "reqPnL",
    }.intersection(invoked_attributes)
    for prohibited in (
        "trade_imbalance",
        "quote_imbalance",
        "microprice",
        "probable_trade_side",
        "future_returns",
        "direction_accuracy",
        "mfe",
        "mae",
    ):
        assert prohibited not in source.casefold()
