import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_core.config import EODHDConfig, ResearchConfig, load_research_config
from stocker_data.storage import DatasetKey, dataset_path, load_dataset
from stocker_data.vendors.eodhd import (
    EODHDClient,
    EODHDEmptyResponseError,
    EODHDHTTPError,
    EODHDInvalidDateRangeError,
    EODHDMissingTokenError,
    EODHDSchemaError,
    EODHDUnsupportedIntervalError,
    chunk_intraday_range,
    fetch_eod_to_storage,
    fetch_intraday_to_storage,
    normalize_eod_response,
    normalize_intraday_response,
    plan_eod_fetch,
    plan_intraday_fetch,
)
from stocker_data.vendors.eodhd_qa import (
    AdjustedPricePolicy,
    create_eodhd_qa_report,
)

FIXTURES = Path(__file__).parent / "fixtures" / "eodhd"


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_research_config_loads_eodhd_vendor_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "research.yaml"
    config_path.write_text(
        """
data:
  data_dir: data
data_vendors:
  eodhd:
    enabled: true
    base_url: https://example.test/api
    api_token_env: EODHD_API_TOKEN
    default_fmt: json
    request_timeout_seconds: 15
    max_retries: 2
    save_raw_by_default: true
""",
        encoding="utf-8",
    )

    config = load_research_config(config_path)

    assert isinstance(config, ResearchConfig)
    assert config.data_vendors.eodhd.enabled is True
    assert config.data_vendors.eodhd.base_url == "https://example.test/api"
    assert config.data_vendors.eodhd.api_token_env == "EODHD_API_TOKEN"
    assert config.data_vendors.eodhd.request_timeout_seconds == 15


def test_missing_token_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
    client = EODHDClient(config=EODHDConfig(api_token_env="EODHD_API_TOKEN"))

    with pytest.raises(EODHDMissingTokenError, match="EODHD_API_TOKEN"):
        client.require_token()


def test_dry_run_plan_does_not_require_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)

    eod_plan = plan_eod_fetch(
        symbol="AAPL.US",
        from_date="2024-01-01",
        to_date="2024-02-01",
        period="d",
        instrument_type="stock",
        data_dir="data",
        save_raw=True,
    )
    intraday_plan = plan_intraday_fetch(
        symbol="AAPL.US",
        from_date="2024-01-01",
        to_date="2024-06-01",
        interval="1m",
        instrument_type="stock",
        data_dir="data",
        save_raw=False,
    )

    assert eod_plan.endpoint == "/eod/AAPL.US"
    assert eod_plan.timeframe == "1d"
    assert "api_token" not in eod_plan.params
    assert intraday_plan.endpoint == "/intraday/AAPL.US"
    assert intraday_plan.timeframe == "1m"
    assert len(intraday_plan.chunks) == 2


def test_eod_url_construction_includes_required_params() -> None:
    client = EODHDClient(config=EODHDConfig(base_url="https://example.test/api"))

    request = client.build_eod_request(
        symbol="AAPL.US",
        from_date="2024-01-01",
        to_date="2024-02-01",
        period="d",
        api_token="token",
    )

    assert str(request.url).startswith("https://example.test/api/eod/AAPL.US")
    assert dict(request.url.params) == {
        "from": "2024-01-01",
        "to": "2024-02-01",
        "period": "d",
        "api_token": "token",
        "fmt": "json",
    }


def test_intraday_url_construction_uses_utc_unix_timestamps() -> None:
    client = EODHDClient(config=EODHDConfig(base_url="https://example.test/api"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)

    request = client.build_intraday_request(
        symbol="AAPL.US",
        interval="1m",
        start=start,
        end=end,
        api_token="token",
    )

    params = dict(request.url.params)
    assert str(request.url).startswith("https://example.test/api/intraday/AAPL.US")
    assert params["interval"] == "1m"
    assert params["from"] == "1704067200"
    assert params["to"] == "1704153600"
    assert params["api_token"] == "token"
    assert params["fmt"] == "json"


def test_intraday_chunking_and_validation() -> None:
    chunks = chunk_intraday_range(
        "2024-01-01",
        "2024-06-01",
        interval="1m",
    )

    assert len(chunks) == 2
    assert chunks[0][0] == datetime(2024, 1, 1, tzinfo=UTC)
    assert chunks[0][1] < chunks[1][1]

    with pytest.raises(EODHDUnsupportedIntervalError):
        chunk_intraday_range("2024-01-01", "2024-01-02", interval="30m")

    with pytest.raises(EODHDInvalidDateRangeError):
        chunk_intraday_range("2024-01-03", "2024-01-02", interval="1m")


def test_eod_response_normalizes_to_canonical_schema() -> None:
    frame = normalize_eod_response(
        _fixture("eod_aapl_sample.json"),
        symbol="AAPL.US",
        instrument_type="stock",
        period="d",
        currency="USD",
    )

    assert frame["source"].unique().tolist() == ["eodhd"]
    assert frame["symbol"].unique().tolist() == ["AAPL.US"]
    assert frame["timeframe"].unique().tolist() == ["1d"]
    assert frame["timezone"].unique().tolist() == ["UTC"]
    assert str(frame["timestamp"].dt.tz) == "UTC"
    assert frame["adjusted_close"].tolist() == [185.4, 184.01]
    assert frame["timestamp"].is_monotonic_increasing


def test_intraday_response_normalizes_sorts_and_dedupes() -> None:
    raw = list(reversed(_fixture("intraday_aapl_1m_sample.json")))
    raw.append(raw[-1] | {"close": 187.99})

    frame = normalize_intraday_response(
        raw,
        symbol="AAPL.US",
        instrument_type="stock",
        interval="1m",
        currency="USD",
    )

    assert len(frame) == 2
    assert frame["timeframe"].unique().tolist() == ["1m"]
    assert str(frame["timestamp"].dt.tz) == "UTC"
    assert frame["close"].tolist() == [187.99, 187.25]


def test_empty_and_bad_schema_responses_raise_clear_errors() -> None:
    with pytest.raises(EODHDEmptyResponseError):
        normalize_eod_response(
            _fixture("empty_response.json"),
            symbol="AAPL.US",
            instrument_type="stock",
            period="d",
        )

    with pytest.raises(EODHDSchemaError, match="high"):
        normalize_eod_response(
            _fixture("bad_schema_response.json"),
            symbol="AAPL.US",
            instrument_type="stock",
            period="d",
        )


def test_fetch_eod_to_storage_merges_dedupes_and_writes_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EODHD_API_TOKEN", "test-token")
    responses = [_fixture("eod_aapl_sample.json"), [_fixture("eod_aapl_sample.json")[-1]]]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = responses.pop(0)
        return httpx.Response(200, json=payload, request=request)

    client = EODHDClient(
        config=EODHDConfig(api_token_env="EODHD_API_TOKEN"),
        transport=httpx.MockTransport(handler),
    )

    first = fetch_eod_to_storage(
        client=client,
        data_dir=tmp_path / "data",
        symbol="AAPL.US",
        from_date="2024-01-02",
        to_date="2024-01-03",
        period="d",
        instrument_type="stock",
        save_raw=True,
        overwrite=True,
    )
    second = fetch_eod_to_storage(
        client=client,
        data_dir=tmp_path / "data",
        symbol="AAPL.US",
        from_date="2024-01-03",
        to_date="2024-01-03",
        period="d",
        instrument_type="stock",
        merge=True,
    )

    assert first.rows_saved == 2
    assert second.rows_saved == 2
    assert first.raw_paths and first.raw_paths[0].exists()

    frame = load_dataset(
        DatasetKey(source="eodhd", instrument_type="stock", symbol="AAPL.US", timeframe="1d"),
        data_dir=tmp_path / "data",
    )
    assert len(frame) == 2
    assert frame["timestamp"].is_unique
    assert (tmp_path / "data" / "catalog.json").exists()


def test_fetch_intraday_to_storage_and_optional_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EODHD_API_TOKEN", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fixture("intraday_aapl_1m_sample.json"), request=request)

    client = EODHDClient(
        config=EODHDConfig(api_token_env="EODHD_API_TOKEN"),
        transport=httpx.MockTransport(handler),
    )

    result = fetch_intraday_to_storage(
        client=client,
        data_dir=tmp_path / "data",
        symbol="AAPL.US",
        from_date="2024-01-02",
        to_date="2024-01-03",
        interval="1m",
        instrument_type="stock",
        save_raw=True,
        overwrite=True,
        audit=True,
    )

    assert result.rows_fetched == 2
    assert result.rows_saved == 2
    assert result.validation_error_count == 0
    assert result.output_path == dataset_path(
        DatasetKey(source="eodhd", instrument_type="stock", symbol="AAPL.US", timeframe="1m"),
        data_dir=tmp_path / "data",
    )
    assert result.audit_markdown_path is not None
    assert result.audit_markdown_path.exists()


def test_http_error_response_raises_vendor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EODHD_API_TOKEN", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limit"}, request=request)

    client = EODHDClient(
        config=EODHDConfig(api_token_env="EODHD_API_TOKEN"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(EODHDHTTPError, match="429"):
        client.fetch_eod(
            symbol="AAPL.US",
            from_date="2024-01-01",
            to_date="2024-01-02",
            period="d",
        )


def test_cli_eodhd_dry_runs_do_not_require_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
    runner = CliRunner()

    eod = runner.invoke(
        app,
        [
            "data",
            "fetch-eodhd-eod",
            "--symbol",
            "AAPL.US",
            "--from",
            "2024-01-01",
            "--to",
            "2024-02-01",
            "--period",
            "d",
            "--instrument-type",
            "stock",
            "--dry-run",
        ],
    )
    intraday = runner.invoke(
        app,
        [
            "data",
            "fetch-eodhd-intraday",
            "--symbol",
            "AAPL.US",
            "--interval",
            "1m",
            "--from",
            "2024-01-01",
            "--to",
            "2024-06-01",
            "--instrument-type",
            "stock",
            "--dry-run",
        ],
    )

    assert eod.exit_code == 0, eod.output
    assert "dry_run" in eod.output
    assert "/eod/AAPL.US" in eod.output
    assert "data.parquet" in eod.output
    assert intraday.exit_code == 0, intraday.output
    assert "/intraday/AAPL.US" in intraday.output
    assert "chunks" in intraday.output


def test_cli_mocked_successful_eod_and_intraday_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EODHD_API_TOKEN", "test-token")
    data_dir = tmp_path / "data"
    runner = CliRunner()

    eod_payload = json.dumps(_fixture("eod_aapl_sample.json")).encode()
    intraday_payload = json.dumps(_fixture("intraday_aapl_1m_sample.json")).encode()

    def fake_eod_fetch(*args: Any, **kwargs: Any) -> Any:
        from stocker_data.vendors.eodhd import _fetch_to_storage_from_payload

        return _fetch_to_storage_from_payload(
            payload=json.loads(eod_payload),
            endpoint="eod",
            data_dir=kwargs["data_dir"],
            symbol=kwargs["symbol"],
            timeframe="1d",
            instrument_type=kwargs["instrument_type"],
            raw_filename="2024-01-02_2024-01-03.json",
            normalize=lambda payload: normalize_eod_response(
                payload,
                symbol=kwargs["symbol"],
                instrument_type=kwargs["instrument_type"],
                period=kwargs["period"],
            ),
            save_raw=kwargs["save_raw"],
            overwrite=kwargs["overwrite"],
            merge=kwargs["merge"],
            audit=kwargs["audit"],
        )

    def fake_intraday_fetch(*args: Any, **kwargs: Any) -> Any:
        from stocker_data.vendors.eodhd import _fetch_to_storage_from_payload

        return _fetch_to_storage_from_payload(
            payload=json.loads(intraday_payload),
            endpoint="intraday",
            data_dir=kwargs["data_dir"],
            symbol=kwargs["symbol"],
            timeframe=kwargs["interval"],
            instrument_type=kwargs["instrument_type"],
            raw_filename="2024-01-02_2024-01-03.json",
            normalize=lambda payload: normalize_intraday_response(
                payload,
                symbol=kwargs["symbol"],
                instrument_type=kwargs["instrument_type"],
                interval=kwargs["interval"],
            ),
            save_raw=kwargs["save_raw"],
            overwrite=kwargs["overwrite"],
            merge=kwargs["merge"],
            audit=kwargs["audit"],
        )

    monkeypatch.setattr("stocker_data.vendors.eodhd.fetch_eod_to_storage", fake_eod_fetch)
    monkeypatch.setattr("stocker_data.vendors.eodhd.fetch_intraday_to_storage", fake_intraday_fetch)

    eod_result = runner.invoke(
        app,
        [
            "data",
            "fetch-eodhd-eod",
            "--symbol",
            "AAPL.US",
            "--from",
            "2024-01-02",
            "--to",
            "2024-01-03",
            "--period",
            "d",
            "--instrument-type",
            "stock",
            "--overwrite",
            "--save-raw",
            "--data-dir",
            str(data_dir),
        ],
    )
    intraday_result = runner.invoke(
        app,
        [
            "data",
            "fetch-eodhd-intraday",
            "--symbol",
            "AAPL.US",
            "--interval",
            "1m",
            "--from",
            "2024-01-02",
            "--to",
            "2024-01-03",
            "--instrument-type",
            "stock",
            "--overwrite",
            "--save-raw",
            "--data-dir",
            str(data_dir),
        ],
    )

    assert eod_result.exit_code == 0, eod_result.output
    assert "rows_saved" in eod_result.output
    assert intraday_result.exit_code == 0, intraday_result.output
    assert "rows_saved" in intraday_result.output


def test_eodhd_qa_report_captures_adjustments_raw_calendar_and_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EODHD_API_TOKEN", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fixture("eod_aapl_sample.json"), request=request)

    client = EODHDClient(
        config=EODHDConfig(api_token_env="EODHD_API_TOKEN"),
        transport=httpx.MockTransport(handler),
    )
    fetch_eod_to_storage(
        client=client,
        data_dir=tmp_path / "data",
        symbol="AAPL.US",
        from_date="2024-01-02",
        to_date="2024-01-03",
        period="d",
        instrument_type="stock",
        save_raw=True,
        overwrite=True,
    )

    report = create_eodhd_qa_report(
        data_dir=tmp_path / "data",
        symbol="AAPL.US",
        timeframe="1d",
        instrument_type="stock",
        market_calendar="XNYS",
        adjusted_price_policy="adjusted_available",
        require_raw=True,
    )

    assert report.markdown_path.exists()
    assert report.json_path.exists()
    assert report.status == "warning"

    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "AAPL.US"
    assert payload["adjusted_close"]["present"] is True
    assert payload["adjusted_close"]["different_from_close_count"] == 2
    assert payload["raw_files"]["count"] == 1
    assert payload["calendar"]["market_calendar"] == "XNYS"
    assert payload["refresh_plan"]["recommended_mode"] == "merge"
    assert "--from 2024-01-03" in payload["refresh_plan"]["eod_command"]
    assert "Adjusted Close" in report.markdown_path.read_text(encoding="utf-8")


def test_eodhd_qa_fails_when_adjusted_close_required_but_missing(tmp_path: Path) -> None:
    frame = normalize_intraday_response(
        _fixture("intraday_aapl_1m_sample.json"),
        symbol="AAPL.US",
        instrument_type="stock",
        interval="1m",
    )
    from stocker_data.storage import write_parquet

    key = DatasetKey(source="eodhd", instrument_type="stock", symbol="AAPL.US", timeframe="1m")
    write_parquet(frame, dataset_path(key, data_dir=tmp_path / "data"))

    report = create_eodhd_qa_report(
        data_dir=tmp_path / "data",
        symbol="AAPL.US",
        timeframe="1m",
        instrument_type="stock",
        adjusted_price_policy="require_adjusted_close",
        require_raw=True,
    )

    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert report.status == "fail"
    assert "missing_adjusted_close" in payload["issue_codes"]
    assert "missing_raw_responses" in payload["issue_codes"]


def test_adjusted_price_policy_type_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        AdjustedPricePolicy("use_magic_prices")


def test_cli_eodhd_qa_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EODHD_API_TOKEN", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fixture("eod_aapl_sample.json"), request=request)

    client = EODHDClient(
        config=EODHDConfig(api_token_env="EODHD_API_TOKEN"),
        transport=httpx.MockTransport(handler),
    )
    fetch_eod_to_storage(
        client=client,
        data_dir=tmp_path / "data",
        symbol="AAPL.US",
        from_date="2024-01-02",
        to_date="2024-01-03",
        period="d",
        instrument_type="stock",
        save_raw=True,
        overwrite=True,
    )

    result = CliRunner().invoke(
        app,
        [
            "data",
            "qa-eodhd",
            "--symbol",
            "AAPL.US",
            "--timeframe",
            "1d",
            "--instrument-type",
            "stock",
            "--adjusted-price-policy",
            "adjusted_available",
            "--require-raw",
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "vendor_qa" in result.output
    assert "warning" in result.output
