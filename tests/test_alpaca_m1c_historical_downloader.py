from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest
from tools.research.download_alpaca_m1c_microstructure import (
    AlpacaCredentials,
    AlpacaHistoricalClient,
    ProviderFeed,
    ProviderPaginationError,
    ProviderRequestError,
    assign_provider_identities,
    normalize_quote,
    normalize_trade,
    run_download,
    write_feed_parquet,
)
from tools.research.download_ibkr_m1c_microstructure import (
    DEFAULT_CANONICAL_SOURCE,
    HistoricalEvent,
)


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


def test_normalisation_preserves_same_timestamp_ticks_by_provider_order() -> None:
    trade = {
        "t": "2024-01-03T15:00:00.123456789Z",
        "p": 185.25,
        "s": 100,
        "x": "Q",
        "c": ["@"],
        "i": 1234,
        "z": "C",
    }
    quote = {
        "t": "2024-01-03T15:00:00.123456789Z",
        "bp": 185.24,
        "ap": 185.25,
        "bs": 3,
        "as": 4,
        "bx": "Q",
        "ax": "P",
        "c": ["R"],
        "z": "C",
    }

    trades = assign_provider_identities(
        [normalize_trade(trade, _event(), sequence=index) for index in range(2)]
    )
    quotes = assign_provider_identities(
        [normalize_quote(quote, _event(), sequence=index) for index in range(2)]
    )

    assert len({row["provider_event_identity"] for row in trades}) == 2
    assert len({row["provider_event_identity"] for row in quotes}) == 2
    assert trades[0]["provider_timestamp_ns"] == 1704294000123456789
    assert quotes[0]["bid_size"] == 3
    assert quotes[0]["bid_size_unit"] == "round_lots"


def test_client_uses_sip_and_follows_opaque_page_tokens() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        token = request.url.params.get("page_token")
        payload = (
            {
                "trades": [{"t": "2024-01-03T15:00:00Z", "p": 1, "s": 1}],
                "next_page_token": "next-token",
            }
            if token is None
            else {
                "trades": [{"t": "2024-01-03T15:00:01Z", "p": 2, "s": 1}],
                "next_page_token": None,
            }
        )
        return httpx.Response(200, json=payload)

    client = AlpacaHistoricalClient(
        AlpacaCredentials("key", "secret"),
        transport=httpx.MockTransport(handler),
        sleeper=lambda _seconds: None,
    )
    result = client.download_feed(
        symbol="AAPL",
        feed="TRADES",
        start_utc=datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
        end_utc=datetime(2024, 1, 3, 15, 1, tzinfo=UTC),
    )

    assert result.request_count == 2
    assert len(result.records) == 2
    assert requests[0].url.params["feed"] == "sip"
    assert requests[0].url.params["limit"] == "10000"
    assert requests[1].url.params["page_token"] == "next-token"
    assert requests[0].headers["APCA-API-KEY-ID"] == "key"


def test_client_rejects_repeated_page_token() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"quotes": [], "next_page_token": "repeated"},
        )

    client = AlpacaHistoricalClient(
        AlpacaCredentials("key", "secret"),
        transport=httpx.MockTransport(handler),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(ProviderPaginationError, match="repeated page token"):
        client.download_feed(
            symbol="AAPL",
            feed="BID_ASK",
            start_utc=datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
            end_utc=datetime(2024, 1, 3, 15, 1, tzinfo=UTC),
        )


def test_client_retries_rate_limit_and_classifies_permission_error() -> None:
    attempts = 0
    sleeps: list[float] = []

    def rate_limited(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, text="slow down")
        return httpx.Response(200, json={"trades": [], "next_page_token": None})

    client = AlpacaHistoricalClient(
        AlpacaCredentials("key", "secret"),
        transport=httpx.MockTransport(rate_limited),
        sleeper=sleeps.append,
    )
    result = client.download_feed(
        symbol="AAPL",
        feed="TRADES",
        start_utc=datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
        end_utc=datetime(2024, 1, 3, 15, 1, tzinfo=UTC),
    )
    assert result.request_count == 2
    assert attempts == 2
    assert 2.0 in sleeps

    blocked = AlpacaHistoricalClient(
        AlpacaCredentials("key", "secret"),
        transport=httpx.MockTransport(lambda _request: httpx.Response(403, text="forbidden")),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(ProviderRequestError) as captured:
        blocked.download_feed(
            symbol="AAPL",
            feed="BID_ASK",
            start_utc=datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
            end_utc=datetime(2024, 1, 3, 15, 1, tzinfo=UTC),
        )
    assert captured.value.status_code == 403


def test_run_download_filters_window_resumes_and_separates_periods(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def download_feed(
            self,
            *,
            symbol: str,
            feed: str,
            start_utc: datetime,
            end_utc: datetime,
            max_pages: int = 20_000,
        ) -> ProviderFeed:
            del max_pages
            self.calls.append((symbol, feed))
            timestamps = [
                start_utc.timestamp() - 1,
                start_utc.timestamp(),
                end_utc.timestamp(),
                end_utc.timestamp() + 1,
            ]
            if feed == "TRADES":
                records = tuple(
                    {
                        "t": datetime.fromtimestamp(value, tz=UTC)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "p": 10,
                        "s": 1,
                        "x": "Q",
                    }
                    for value in timestamps
                )
            else:
                records = tuple(
                    {
                        "t": datetime.fromtimestamp(value, tz=UTC)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "bp": 9,
                        "ap": 10,
                        "bs": 1,
                        "as": 2,
                    }
                    for value in timestamps
                )
            return ProviderFeed(feed, records, 1)

    root = tmp_path / "alpaca"
    client = FakeClient()
    first = run_download(
        period="development",
        source=DEFAULT_CANONICAL_SOURCE,
        output_root=root,
        include_r01=False,
        resume=True,
        validation_hard_count=None,
        event_limit=1,
        client=client,
    )
    assert first["complete_events"] == 1
    assert len(client.calls) == 2
    event_directory = next((root / "development").glob("*/*"))
    assert pq.read_table(event_directory / "trades.parquet").num_rows == 2

    second = run_download(
        period="development",
        source=DEFAULT_CANONICAL_SOURCE,
        output_root=root,
        include_r01=False,
        resume=True,
        validation_hard_count=None,
        event_limit=1,
        client=client,
    )
    assert second["skipped_events"] == 1
    assert len(client.calls) == 2

    run_download(
        period="assessment",
        source=DEFAULT_CANONICAL_SOURCE,
        output_root=root,
        include_r01=False,
        resume=True,
        validation_hard_count=None,
        event_limit=1,
        client=client,
    )
    assert any((root / "development").glob("*/*/metadata.json"))
    assert any((root / "assessment").glob("*/*/metadata.json"))
    metadata = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.glob("*/*/*/metadata.json")
    ]
    assert all(not row["session_date"].startswith("2026-") for row in metadata)
    manifest = json.loads((root / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["event_counts"] == {"development": 1, "assessment": 1, "total": 2}

def test_parquet_output_is_deterministic(tmp_path: Path) -> None:
    records = assign_provider_identities(
        [
            normalize_trade(
                {
                    "t": "2024-01-03T15:00:00.123456789Z",
                    "p": 185.25,
                    "s": 100,
                    "x": "Q",
                    "c": ["@"],
                    "i": 1234,
                    "z": "C",
                },
                _event(),
                sequence=0,
            )
        ]
    )
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"

    write_feed_parquet(left, "TRADES", records)
    write_feed_parquet(right, "TRADES", records)

    assert left.read_bytes() == right.read_bytes()
    table = pq.read_table(left)
    assert table.schema.field("provider_timestamp_utc").type.unit == "ns"
    assert table.column("source").to_pylist() == ["alpaca_historical_sip"]


def test_credentials_fail_closed_when_environment_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="credentials are absent"):
        AlpacaCredentials.from_environment()


def test_downloader_contains_no_order_or_direction_operations() -> None:
    path = (
        Path(__file__).parents[1]
        / "tools"
        / "research"
        / "download_alpaca_m1c_microstructure.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {
        "submit_order",
        "cancel_order",
        "get_account",
        "get_positions",
        "trade_imbalance",
        "quote_imbalance",
        "microprice",
        "probable_trade_side",
        "future_returns",
        "pnl",
    }
    names = {
        node.attr.casefold()
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    } | {node.id.casefold() for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert names.isdisjoint(forbidden)
    assert "direction_analysis_performed" in path.read_text(encoding="utf-8")
