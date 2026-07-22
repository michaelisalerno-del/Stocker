from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from stocker_research.eodhd_options_downloader_v0 import (
    DownloadConfig,
    EODHDOptionsDownloader,
    OptionsRequest,
    OptionsResourceLimitExceeded,
    canonicalize_response_records,
    deterministic_symbol_mapping,
    redact_secrets,
    resolve_canonical_duplicates,
    sha256_bytes,
    split_request_for_offset_limit,
    stable_request_id,
)


def test_credentials_are_redacted_from_request_identity_and_text() -> None:
    token = "secret-token-that-must-not-survive"
    params = {
        "api_token": token,
        "filter[underlying_symbol]": "AAPL",
        "page[limit]": 10,
    }

    request_id = stable_request_id("/mp/unicornbay/options/eod", params)
    redacted = redact_secrets(
        f"https://eodhd.com/api/options/eod?api_token={token}&symbol=AAPL",
        secrets=(token,),
    )

    assert token not in request_id
    assert token not in redacted
    assert "api_token=%5BREDACTED%5D" in redacted


@dataclass
class FakeResponse:
    status_code: int
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    closed: bool = False

    @property
    def content(self) -> bytes:
        return json.dumps(self.payload, sort_keys=True).encode("utf-8")

    def json(self) -> dict[str, Any]:
        return self.payload

    def iter_content(self, chunk_size: int) -> Any:
        content = self.content
        for position in range(0, len(content), chunk_size):
            yield content[position : position + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, *, params: dict[str, object], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return self.responses.pop(0)


def response_page(*, offset: int, limit: int, total: int, ids: list[str]) -> FakeResponse:
    data = [
        {
            "id": record_id,
            "type": "options-eod",
            "attributes": {"contract": record_id},
        }
        for record_id in ids
    ]
    next_link = "" if offset + len(data) >= total else "https://eodhd.com/api/next"
    return FakeResponse(
        200,
        {
            "meta": {"offset": offset, "limit": limit, "total": total, "fields": []},
            "data": data,
            "links": {"next": next_link},
        },
    )


def test_pagination_hashes_and_caches_every_complete_page(tmp_path: Path) -> None:
    token = "never-persist-this-token"
    transport = FakeTransport(
        [
            response_page(offset=0, limit=2, total=3, ids=["one", "two"]),
            response_page(offset=2, limit=2, total=3, ids=["three"]),
        ]
    )
    downloader = EODHDOptionsDownloader(
        DownloadConfig(token=token, data_dir=tmp_path, page_limit=2, max_attempts=2),
        transport=transport,
        sleep=lambda _seconds: None,
    )

    result = downloader.download(
        OptionsRequest(
            underlying_symbol="AAPL",
            trade_date_from=date(2025, 1, 2),
            trade_date_to=date(2025, 1, 2),
            strike_from=100.0,
            strike_to=300.0,
        )
    )

    assert [call["params"]["page[offset]"] for call in transport.calls] == [0, 2]  # type: ignore[index]
    assert [item["id"] for item in result.records] == ["one", "two", "three"]
    assert len(result.manifest_rows) == 2
    assert all(row.response_hash for row in result.manifest_rows)
    assert all(Path(row.cache_path).is_file() for row in result.manifest_rows)
    assert token not in json.dumps([row.to_dict() for row in result.manifest_rows])


def test_completed_download_resumes_from_verified_content_cache(tmp_path: Path) -> None:
    request = OptionsRequest(
        underlying_symbol="AAPL",
        trade_date_from=date(2025, 1, 2),
        trade_date_to=date(2025, 1, 2),
    )
    first_transport = FakeTransport([response_page(offset=0, limit=10, total=1, ids=["cached"])])
    config = DownloadConfig(token="secret", data_dir=tmp_path, page_limit=10)
    first = EODHDOptionsDownloader(
        config, transport=first_transport, sleep=lambda _seconds: None
    ).download(request)
    second_transport = FakeTransport([])

    resumed = EODHDOptionsDownloader(
        config, transport=second_transport, sleep=lambda _seconds: None
    ).download(request)

    assert resumed.records == first.records
    assert resumed.manifest_rows == first.manifest_rows
    assert second_transport.calls == []


def test_oversized_request_splits_and_retains_attempt_manifest(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            response_page(offset=0, limit=1000, total=11_001, ids=["probe"]),
            response_page(offset=0, limit=1000, total=1, ids=["left"]),
            response_page(offset=0, limit=1000, total=1, ids=["right"]),
        ]
    )
    downloader = EODHDOptionsDownloader(
        DownloadConfig(token="secret", data_dir=tmp_path),
        transport=transport,
        sleep=lambda _seconds: None,
    )
    request = OptionsRequest(
        underlying_symbol="AAPL",
        trade_date_from=date(2025, 1, 2),
        trade_date_to=date(2025, 1, 3),
        expiration_from=date(2025, 1, 9),
        expiration_to=date(2025, 4, 3),
        strike_from=100.0,
        strike_to=300.0,
    )

    result = downloader.download_with_splitting(request)

    assert [record["id"] for record in result.records] == ["left", "right"]
    assert len(result.manifest_rows) == 3
    assert result.manifest_rows[0].record_count == 1
    assert result.manifest_rows[0].superseded_by_split is True
    assert all(not row.superseded_by_split for row in result.manifest_rows[1:])


def test_retry_after_is_respected_for_transient_rate_limit(tmp_path: Path) -> None:
    rate_limited = FakeResponse(429, {"error": "slow down"}, {"Retry-After": "2"})
    transport = FakeTransport(
        [rate_limited, response_page(offset=0, limit=10, total=1, ids=["ok"])]
    )
    sleeps: list[float] = []
    downloader = EODHDOptionsDownloader(
        DownloadConfig(token="secret", data_dir=tmp_path, page_limit=10, max_attempts=3),
        transport=transport,
        sleep=sleeps.append,
    )

    result = downloader.download(OptionsRequest(underlying_symbol="AAPL"))

    assert sleeps == [2.0]
    assert result.manifest_rows[0].attempts == 2
    assert len(transport.calls) == 2


def test_raw_record_resource_limit_stops_before_another_page(tmp_path: Path) -> None:
    transport = FakeTransport([response_page(offset=0, limit=1, total=2, ids=["first"])])
    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token="secret",
            data_dir=tmp_path,
            page_limit=1,
            maximum_raw_records=1,
        ),
        transport=transport,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(OptionsResourceLimitExceeded, match="resource_limit") as captured:
        downloader.download(OptionsRequest(underlying_symbol="AAPL"))

    assert len(transport.calls) == 1
    assert len(captured.value.manifest_rows) == 1
    assert captured.value.manifest_rows[0].record_count == 1


def test_streamed_byte_limit_blocks_without_content_length(tmp_path: Path) -> None:
    response = response_page(offset=0, limit=10, total=1, ids=["too-large"])
    maximum_bytes = len(response.content) - 1
    transport = FakeTransport([response])
    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token="secret",
            data_dir=tmp_path,
            page_limit=10,
            maximum_download_bytes=maximum_bytes,
        ),
        transport=transport,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(OptionsResourceLimitExceeded, match="resource_limit") as captured:
        downloader.download(OptionsRequest(underlying_symbol="AAPL"))

    assert captured.value.manifest_rows == []
    assert len(transport.calls) == 1
    assert response.closed is True
    assert not (tmp_path / "raw").exists()


def test_streamed_body_at_exact_declared_byte_limit_is_not_double_counted(
    tmp_path: Path,
) -> None:
    response = response_page(offset=0, limit=10, total=1, ids=["exact"])
    response.headers["Content-Length"] = str(len(response.content))
    transport = FakeTransport([response])
    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token="secret",
            data_dir=tmp_path,
            page_limit=10,
            maximum_download_bytes=len(response.content),
        ),
        transport=transport,
        sleep=lambda _seconds: None,
    )

    result = downloader.download(OptionsRequest(underlying_symbol="AAPL"))

    assert [record["id"] for record in result.records] == ["exact"]
    assert response.closed is True


def test_offset_limit_chunk_split_prefers_dates_then_expiry_then_strike() -> None:
    monthly = OptionsRequest(
        underlying_symbol="AAPL",
        trade_date_from=date(2025, 1, 2),
        trade_date_to=date(2025, 1, 31),
        expiration_from=date(2025, 1, 9),
        expiration_to=date(2025, 4, 30),
        strike_from=100.0,
        strike_to=300.0,
    )
    left, right = split_request_for_offset_limit(monthly)
    assert left.trade_date_to < right.trade_date_from  # type: ignore[operator]
    assert left.expiration_from == right.expiration_from

    one_day = monthly.replace(trade_date_from=date(2025, 1, 2), trade_date_to=date(2025, 1, 2))
    left, right = split_request_for_offset_limit(one_day)
    assert left.expiration_to < right.expiration_from  # type: ignore[operator]

    one_expiry = one_day.replace(expiration_from=date(2025, 2, 21), expiration_to=date(2025, 2, 21))
    left, right = split_request_for_offset_limit(one_expiry)
    assert left.strike_to == 200.0
    assert right.strike_from > left.strike_to  # type: ignore[operator]
    assert right.strike_from == math.nextafter(200.0, math.inf)


def test_raw_response_hash_is_sha256() -> None:
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def valid_provider_record(**overrides: object) -> dict[str, Any]:
    attributes: dict[str, object] = {
        "contract": "AAPL250117C00200000",
        "underlying_symbol": "AAPL",
        "type": "call",
        "exp_date": "2025-01-17",
        "strike": 200.0,
        "tradetime": "2025-01-02",
        "last": 1.1,
        "bid": 1.0,
        "ask": 1.2,
        "bid_size": 4,
        "ask_size": 5,
        "volume": 10,
        "open_interest": 100,
        "volatility": 0.4,
        "theoretical": 1.15,
        "delta": 0.5,
        "gamma": 0.02,
        "theta": -0.03,
        "vega": 0.1,
        "rho": 0.01,
        "dte": 15,
        "moneyness": 0.0,
    }
    attributes.update(overrides)
    return {
        "id": f"{attributes['contract']}-{attributes['tradetime']}",
        "type": "options-eod",
        "attributes": attributes,
    }


def test_canonical_options_schema_maps_documented_eod_fields() -> None:
    result = canonicalize_response_records(
        [valid_provider_record()],
        request_id="request-1",
        provider_schema_version="openapi-2.0.0",
    )

    assert result.rejections == []
    record = result.records[0]
    assert record["provider"] == "EODHD/UnicornBay"
    assert record["contract_id"] == "AAPL250117C00200000"
    assert record["option_type"] == "call"
    assert record["trade_date"] == date(2025, 1, 2)
    assert record["trade_timestamp"].isoformat() == "2025-01-02T21:00:00+00:00"
    assert record["midpoint"] == 1.1
    assert record["implied_volatility"] == 0.4
    assert record["raw_record_hash"]


def test_invalid_canonical_records_are_rejected_without_coercion() -> None:
    invalid_type = valid_provider_record(type="banana")
    expired = valid_provider_record(exp_date="2024-12-31")
    bad_strike = valid_provider_record(strike=0)

    result = canonicalize_response_records(
        [invalid_type, expired, bad_strike],
        request_id="request-2",
        provider_schema_version="openapi-2.0.0",
    )

    assert result.records == []
    assert [item.reason_code for item in result.rejections] == [
        "invalid_option_type",
        "expiration_before_trade_date",
        "invalid_strike",
    ]


def test_duplicate_contract_dates_resolve_by_raw_hash() -> None:
    first = canonicalize_response_records(
        [valid_provider_record(bid=1.0)],
        request_id="request-3",
        provider_schema_version="openapi-2.0.0",
    ).records[0]
    second = canonicalize_response_records(
        [valid_provider_record(bid=0.9)],
        request_id="request-4",
        provider_schema_version="openapi-2.0.0",
    ).records[0]

    result = resolve_canonical_duplicates([first, second, dict(first)])

    assert result.duplicate_records == 2
    assert len(result.records) == 1
    assert result.records[0]["raw_record_hash"] == min(
        first["raw_record_hash"], second["raw_record_hash"]
    )


def test_symbol_mapping_uses_only_auditable_exact_transforms() -> None:
    result = deterministic_symbol_mapping(
        ["AAPL", "MSFT", "BRK.B", "UNKNOWN"],
        provider_coverage={"AAPL", "MSFT.US", "BRK-B.US"},
    )

    rows = {row.stocker_symbol: row for row in result.rows}
    assert rows["AAPL"].eodhd_underlying_symbol == "AAPL"
    assert rows["AAPL"].mapping_method == "exact_symbol"
    assert rows["MSFT"].eodhd_underlying_symbol == "MSFT.US"
    assert rows["MSFT"].mapping_method == "add_us_suffix"
    assert rows["BRK.B"].eodhd_underlying_symbol == "BRK-B.US"
    assert rows["BRK.B"].mapping_method == "dot_to_hyphen_add_us_suffix"
    assert rows["UNKNOWN"].coverage_available is False
    assert result.ambiguous_symbols == []
