import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_core.config import EODHDConfig
from stocker_data.storage import DatasetKey, dataset_path, write_parquet
from stocker_data.universe import (
    UniverseDefinition,
    UniverseQualificationRules,
    load_research_ready_universe,
    load_universe,
    qualify_universe,
    save_universe,
    universe_health_report,
    validate_universe,
)
from stocker_data.vendors.eodhd import (
    EODHDClient,
    EODHDSchemaError,
    EODHDUnsupportedScreenerError,
    build_screener_filters,
    fetch_screener_all,
    normalize_screener_response,
)


def _write_config(tmp_path: Path, *, data_dir: Path, enabled: bool = True) -> Path:
    config_path = tmp_path / "research.yaml"
    config_path.write_text(
        f"""
data:
  data_dir: {data_dir}
  timezone: UTC
  default_currency: USD
data_vendors:
  eodhd:
    enabled: {str(enabled).lower()}
    base_url: https://example.test/api
    api_token_env: EODHD_TEST_TOKEN
    default_fmt: json
    request_timeout_seconds: 5
    max_retries: 2
    save_raw_by_default: true
""",
        encoding="utf-8",
    )
    return config_path


def _universe_payload(symbols: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": "us_test",
        "name": "US Test",
        "description": "Small test universe",
        "source": "manual",
        "created_at": "2026-06-28T00:00:00Z",
        "filters": {
            "exchange": "US",
            "min_price": 5,
            "min_market_cap": 1_000_000_000,
            "min_avgvol_200d": 500_000,
            "sectors": [],
            "industries": [],
        },
        "symbols": symbols
        or [
            {"symbol": "MSFT.US", "name": "Microsoft", "exchange": "US", "currency": "USD"},
            {"symbol": "AAPL.US", "name": "Apple", "exchange": "US", "currency": "USD"},
        ],
    }


def _write_universe(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "universe.yaml"
    path.write_text(yaml.safe_dump(payload or _universe_payload(), sort_keys=False), "utf-8")
    return path


def _sample_frame(
    *,
    symbol: str,
    rows: int,
    close: float = 20.0,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    close_series = pd.Series([close for _ in range(rows)])
    return pd.DataFrame(
        {
            "source": "eodhd",
            "symbol": symbol,
            "instrument_type": "stock",
            "timeframe": "1d",
            "timestamp": dates,
            "open": close_series,
            "high": close_series + 1.0,
            "low": close_series - 1.0,
            "close": close_series,
            "volume": volume,
            "currency": "USD",
            "timezone": "UTC",
        }
    )


def _intraday_frame(symbol: str, sessions: int = 2, bars_per_session: int = 78) -> pd.DataFrame:
    chunks = []
    for session_start in pd.date_range("2026-06-25 13:30", periods=sessions, freq="B", tz="UTC"):
        chunks.append(pd.date_range(session_start, periods=bars_per_session, freq="5min"))
    timestamps = chunks[0]
    for chunk in chunks[1:]:
        timestamps = timestamps.append(chunk)
    close_series = pd.Series([100.0 + index * 0.01 for index in range(len(timestamps))])
    return pd.DataFrame(
        {
            "source": "eodhd",
            "symbol": symbol,
            "instrument_type": "stock",
            "timeframe": "5m",
            "timestamp": timestamps,
            "open": close_series.shift(1).fillna(close_series.iloc[0]),
            "high": close_series + 0.05,
            "low": close_series - 0.05,
            "close": close_series,
            "volume": 10_000,
            "currency": "USD",
            "timezone": "UTC",
        }
    )


def test_universe_load_save_validate_and_research_ready_export(tmp_path: Path) -> None:
    universe_path = _write_universe(
        tmp_path,
        _universe_payload(
            [
                {"symbol": "msft.us", "name": "Microsoft", "exchange": "US"},
                {"symbol": "AAPL.US", "name": "Apple", "exchange": "US"},
                {"symbol": "MSFT.US", "name": "Duplicate", "exchange": "US"},
            ]
        ),
    )

    universe = load_universe(universe_path)
    issues = validate_universe(universe)

    assert universe.symbols[0].symbol == "MSFT.US"
    assert any(issue.code == "duplicate_symbol" for issue in issues)

    clean = UniverseDefinition.model_validate(_universe_payload())
    output = tmp_path / "sorted.yaml"
    save_universe(clean, output)
    reloaded = load_universe(output)
    assert [symbol.symbol for symbol in reloaded.symbols] == ["AAPL.US", "MSFT.US"]

    research_ready = tmp_path / "ready.json"
    research_ready.write_text(
        json.dumps({"qualified_symbols": [{"symbol": "AAPL.US"}, {"symbol": "MSFT.US"}]}),
        encoding="utf-8",
    )
    assert load_research_ready_universe(research_ready) == ["AAPL.US", "MSFT.US"]


def test_eodhd_screener_request_filters_pagination_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EODHD_TEST_TOKEN", "secret-token")
    client = EODHDClient(
        config=EODHDConfig(base_url="https://example.test/api", api_token_env="EODHD_TEST_TOKEN")
    )
    filters = build_screener_filters(
        exchange="US",
        min_price=5,
        min_market_cap=1_000_000_000,
        min_avgvol_200d=500_000,
        sectors=["Technology"],
        industries=["Software"],
    )

    request = client.build_screener_request(
        filters=filters,
        signals=[],
        sort="market_capitalization.desc",
        limit=100,
        offset=0,
        api_token="secret-token",
    )

    params = dict(request.url.params)
    assert str(request.url).startswith("https://example.test/api/screener")
    assert params["limit"] == "100"
    assert params["offset"] == "0"
    assert params["sort"] == "market_capitalization.desc"
    assert json.loads(params["filters"]) == filters
    assert "secret-token" in params["api_token"]

    normalized = normalize_screener_response(
        {
            "data": [
                {
                    "code": "msft.us",
                    "name": "Microsoft",
                    "exchange": "US",
                    "currency": "USD",
                    "sector": "Technology",
                    "market_capitalization": 3_000_000_000_000,
                    "adjusted_close": 420.0,
                    "avgvol_200d": 25_000_000,
                }
            ]
        }
    )
    assert normalized[0].symbol == "MSFT.US"
    assert normalized[0].market_capitalization == 3_000_000_000_000

    with pytest.raises(EODHDUnsupportedScreenerError):
        client.build_screener_request(
            filters=[],
            signals=[],
            sort="market_capitalization.desc",
            limit=101,
            offset=0,
            api_token="secret-token",
        )

    with pytest.raises(EODHDSchemaError):
        normalize_screener_response({"data": [{"name": "Missing code"}]})


def test_fetch_screener_all_uses_pages_and_safe_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EODHD_TEST_TOKEN", "secret-token")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        offset = int(request.url.params["offset"])
        payload = {
            "data": [
                {
                    "code": f"SYM{offset}.US",
                    "name": f"Symbol {offset}",
                    "exchange": "US",
                    "currency": "USD",
                }
            ]
        }
        return httpx.Response(200, json=payload, request=request)

    client = EODHDClient(
        config=EODHDConfig(api_token_env="EODHD_TEST_TOKEN"),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    rows = fetch_screener_all(
        client=client,
        filters=[],
        signals=[],
        sort="market_capitalization.desc",
        limit=150,
        max_pages=2,
    )

    assert [row.symbol for row in rows] == ["SYM0.US", "SYM100.US"]
    assert [int(request.url.params["limit"]) for request in requests] == [100, 50]
    assert [int(request.url.params["offset"]) for request in requests] == [0, 100]

    with pytest.raises(EODHDUnsupportedScreenerError):
        fetch_screener_all(
            client=client,
            filters=[],
            signals=[],
            sort="market_capitalization.desc",
            limit=1_200,
            max_pages=12,
        )


def test_universe_build_cli_dry_run_and_mocked_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_data.vendors import eodhd as eodhd_module

    config_path = _write_config(tmp_path, data_dir=tmp_path / "data")
    runner = CliRunner()
    dry_run = runner.invoke(
        app,
        [
            "universe",
            "build-eodhd",
            "--id",
            "us_large_liquid",
            "--name",
            "US Large Liquid Stocks",
            "--exchange",
            "US",
            "--min-price",
            "5",
            "--min-market-cap",
            "1000000000",
            "--min-avgvol-200d",
            "500000",
            "--limit",
            "100",
            "--output",
            str(tmp_path / "generated.yaml"),
            "--dry-run",
            "--config",
            str(config_path),
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert "planned_requests" in dry_run.output
    assert "api_token" not in dry_run.output

    monkeypatch.setenv("EODHD_TEST_TOKEN", "secret-token")

    def fake_fetch_screener_all(**_: Any) -> list[Any]:
        return normalize_screener_response(
            {
                "data": [
                    {"code": "MSFT.US", "name": "Microsoft", "exchange": "US"},
                    {"code": "AAPL.US", "name": "Apple", "exchange": "US"},
                ]
            }
        )

    monkeypatch.setattr(eodhd_module, "fetch_screener_all", fake_fetch_screener_all)
    output = tmp_path / "generated.yaml"
    result = runner.invoke(
        app,
        [
            "universe",
            "build-eodhd",
            "--id",
            "us_large_liquid",
            "--name",
            "US Large Liquid Stocks",
            "--exchange",
            "US",
            "--limit",
            "2",
            "--output",
            str(output),
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    built = load_universe(output)
    assert [symbol.symbol for symbol in built.symbols] == ["AAPL.US", "MSFT.US"]
    assert (tmp_path / "data" / "reports" / "universes" / "us_large_liquid_build.json").exists()


def test_universe_validate_cli_passes_and_fails_duplicates(tmp_path: Path) -> None:
    runner = CliRunner()
    valid = tmp_path / "valid.yaml"
    duplicate = tmp_path / "duplicate.yaml"
    valid.write_text(yaml.safe_dump(_universe_payload(), sort_keys=False), encoding="utf-8")
    duplicate.write_text(
        yaml.safe_dump(
            _universe_payload(
                [
                    {"symbol": "AAPL.US", "exchange": "US"},
                    {"symbol": "AAPL.US", "exchange": "US"},
                ]
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    valid_result = runner.invoke(app, ["universe", "validate", "--universe", str(valid)])
    duplicate_result = runner.invoke(app, ["universe", "validate", "--universe", str(duplicate)])

    assert valid_result.exit_code == 0, valid_result.output
    assert duplicate_result.exit_code != 0
    assert "duplicate_symbol" in duplicate_result.output


def test_universe_fetch_cli_dry_run_and_batch_failure_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_data.vendors import eodhd as eodhd_module

    config_path = _write_config(tmp_path, data_dir=tmp_path / "data")
    universe_path = _write_universe(
        tmp_path,
        _universe_payload(
            [
                {"symbol": "AAPL.US", "exchange": "US"},
                {"symbol": "FAIL.US", "exchange": "US"},
                {"symbol": "MSFT.US", "exchange": "US"},
            ]
        ),
    )
    runner = CliRunner()
    dry_run = runner.invoke(
        app,
        [
            "universe",
            "fetch",
            "--universe",
            str(universe_path),
            "--from",
            "2024-01-01",
            "--to",
            "2024-02-01",
            "--timeframe",
            "1d",
            "--source",
            "eodhd",
            "--dry-run",
            "--max-symbols",
            "2",
            "--config",
            str(config_path),
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert "planned" in dry_run.output
    assert "AAPL.US" in dry_run.output
    assert "FAIL.US" in dry_run.output
    assert "MSFT.US" not in dry_run.output

    monkeypatch.setenv("EODHD_TEST_TOKEN", "secret-token")

    def fake_fetch_eod_to_storage(**kwargs: Any) -> Any:
        if kwargs["symbol"] == "FAIL.US":
            raise RuntimeError("vendor failure")
        key = DatasetKey(
            source="eodhd",
            instrument_type="stock",
            symbol=kwargs["symbol"],
            timeframe="1d",
        )
        path = dataset_path(key, data_dir=kwargs["data_dir"])
        write_parquet(_sample_frame(symbol=kwargs["symbol"], rows=4), path)
        from stocker_data.vendors.eodhd import EODHDFetchResult

        return EODHDFetchResult(
            output_path=path,
            catalog_path=Path(kwargs["data_dir"]) / "catalog.json",
            raw_paths=[],
            rows_fetched=4,
            rows_saved=4,
            min_timestamp="2024-01-01 00:00:00+00:00",
            max_timestamp="2024-01-04 00:00:00+00:00",
            issues=[],
        )

    monkeypatch.setattr(eodhd_module, "fetch_eod_to_storage", fake_fetch_eod_to_storage)
    result = runner.invoke(
        app,
        [
            "universe",
            "fetch",
            "--universe",
            str(universe_path),
            "--from",
            "2024-01-01",
            "--to",
            "2024-02-01",
            "--timeframe",
            "1d",
            "--source",
            "eodhd",
            "--merge",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(
        (tmp_path / "data" / "reports" / "universes" / "us_test_1d_fetch.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["status"] for item in report["items"]] == ["fetched", "failed", "fetched"]

    fail_fast = runner.invoke(
        app,
        [
            "universe",
            "fetch",
            "--universe",
            str(universe_path),
            "--from",
            "2024-01-01",
            "--to",
            "2024-02-01",
            "--timeframe",
            "1d",
            "--source",
            "eodhd",
            "--merge",
            "--fail-fast",
            "--config",
            str(config_path),
        ],
    )
    assert fail_fast.exit_code != 0
    assert "vendor failure" in fail_fast.output


def test_universe_fetch_resume_and_skip_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stocker_data.vendors import eodhd as eodhd_module

    monkeypatch.setenv("EODHD_TEST_TOKEN", "secret-token")
    data_dir = tmp_path / "data"
    config_path = _write_config(tmp_path, data_dir=data_dir)
    universe_path = _write_universe(tmp_path)
    key = DatasetKey(source="eodhd", instrument_type="stock", symbol="AAPL.US", timeframe="1d")
    write_parquet(_sample_frame(symbol="AAPL.US", rows=4), dataset_path(key, data_dir=data_dir))

    called: list[str] = []

    def fake_fetch_eod_to_storage(**kwargs: Any) -> Any:
        called.append(kwargs["symbol"])
        from stocker_data.vendors.eodhd import EODHDFetchResult

        return EODHDFetchResult(
            output_path=dataset_path(
                DatasetKey(
                    source="eodhd",
                    instrument_type="stock",
                    symbol=kwargs["symbol"],
                    timeframe="1d",
                ),
                data_dir=kwargs["data_dir"],
            ),
            catalog_path=Path(kwargs["data_dir"]) / "catalog.json",
            raw_paths=[],
            rows_fetched=1,
            rows_saved=1,
            min_timestamp=None,
            max_timestamp=None,
            issues=[],
        )

    monkeypatch.setattr(eodhd_module, "fetch_eod_to_storage", fake_fetch_eod_to_storage)
    result = CliRunner().invoke(
        app,
        [
            "universe",
            "fetch",
            "--universe",
            str(universe_path),
            "--from",
            "2024-01-01",
            "--to",
            "2024-02-01",
            "--timeframe",
            "1d",
            "--source",
            "eodhd",
            "--skip-existing",
            "--max-symbols",
            "2",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert called == ["MSFT.US"]


def test_universe_qualify_rejects_and_exports_research_ready(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    universe_path = _write_universe(
        tmp_path,
        _universe_payload(
            [
                {"symbol": "GOOD.US", "exchange": "US"},
                {"symbol": "SHORT.US", "exchange": "US"},
                {"symbol": "LOWPRICE.US", "exchange": "US"},
                {"symbol": "LOWVOL.US", "exchange": "US"},
            ]
        ),
    )
    for symbol, rows, close, volume in [
        ("GOOD.US", 800, 50.0, 1_000_000),
        ("SHORT.US", 20, 50.0, 1_000_000),
        ("LOWPRICE.US", 800, 2.0, 1_000_000),
        ("LOWVOL.US", 800, 50.0, 10),
    ]:
        write_parquet(
            _sample_frame(symbol=symbol, rows=rows, close=close, volume=volume),
            dataset_path(
                DatasetKey(source="eodhd", instrument_type="stock", symbol=symbol, timeframe="1d"),
                data_dir=data_dir,
            ),
        )
    output = tmp_path / "ready.json"
    result = qualify_universe(
        universe=load_universe(universe_path),
        data_dir=data_dir,
        timeframe="1d",
        source="eodhd",
        rules=UniverseQualificationRules(
            min_history_days=750,
            min_last_close=5,
            min_median_dollar_volume_60d=10_000_000,
            max_validation_errors=0,
            max_missing_session_warnings=500,
        ),
        output_path=output,
    )

    assert result.qualified_symbols == ["GOOD.US"]
    rejected = {item.symbol: item.reasons for item in result.rejected_symbols}
    assert "insufficient_history" in rejected["SHORT.US"]
    assert "low_last_close" in rejected["LOWPRICE.US"]
    assert "low_dollar_volume" in rejected["LOWVOL.US"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["qualified_symbols"] == [{"symbol": "GOOD.US"}]
    assert (data_dir / "reports" / "universes" / "us_test_1d_qualification.md").exists()


def test_universe_qualify_supports_intraday_row_and_session_rules(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    universe_path = _write_universe(
        tmp_path,
        _universe_payload(
            [
                {"symbol": "GOOD.US", "exchange": "US"},
                {"symbol": "SHORTROWS.US", "exchange": "US"},
                {"symbol": "ONESESSION.US", "exchange": "US"},
            ]
        ),
    )
    for symbol, frame in [
        ("GOOD.US", _intraday_frame("GOOD.US", sessions=2, bars_per_session=78)),
        ("SHORTROWS.US", _intraday_frame("SHORTROWS.US", sessions=2, bars_per_session=20)),
        ("ONESESSION.US", _intraday_frame("ONESESSION.US", sessions=1, bars_per_session=78)),
    ]:
        write_parquet(
            frame,
            dataset_path(
                DatasetKey(source="eodhd", instrument_type="stock", symbol=symbol, timeframe="5m"),
                data_dir=data_dir,
            ),
        )
    output = tmp_path / "ready_5m.json"

    result = qualify_universe(
        universe=load_universe(universe_path),
        data_dir=data_dir,
        timeframe="5m",
        source="eodhd",
        rules=UniverseQualificationRules(
            min_history_days=0,
            min_last_close=0,
            min_median_dollar_volume_60d=0,
            min_row_count=100,
            min_sessions=2,
            max_validation_errors=0,
            max_missing_session_warnings=0,
        ),
        output_path=output,
        market_calendar="XNYS",
    )

    assert result.qualified_symbols == ["GOOD.US"]
    rejected = {item.symbol: item.reasons for item in result.rejected_symbols}
    assert "insufficient_rows" in rejected["SHORTROWS.US"]
    assert "insufficient_sessions" in rejected["ONESESSION.US"]


def test_universe_health_counts_datasets_and_qualification(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    universe = UniverseDefinition.model_validate(_universe_payload())
    write_parquet(
        _sample_frame(symbol="AAPL.US", rows=10),
        dataset_path(
            DatasetKey(source="eodhd", instrument_type="stock", symbol="AAPL.US", timeframe="1d"),
            data_dir=data_dir,
        ),
    )
    ready_dir = data_dir / "universes" / "research_ready"
    ready_dir.mkdir(parents=True)
    (ready_dir / "us_test_1d.json").write_text(
        json.dumps(
            {
                "qualified_symbols": [{"symbol": "AAPL.US"}],
                "rejected_symbols": [{"symbol": "MSFT.US", "reasons": ["missing_dataset"]}],
            }
        ),
        encoding="utf-8",
    )

    report = universe_health_report(
        universe=universe,
        data_dir=data_dir,
        timeframe="1d",
        source="eodhd",
    )

    assert report.total_symbols == 2
    assert report.datasets_present == 1
    assert report.datasets_missing == 1
    assert report.research_ready_count == 1
    assert report.rejected_count == 1
    assert report.markdown_path.exists()
    assert report.json_path.exists()


def test_universe_cli_help_screens() -> None:
    runner = CliRunner()

    for args in [
        ["universe", "--help"],
        ["universe", "build-eodhd", "--help"],
        ["universe", "fetch", "--help"],
        ["universe", "qualify", "--help"],
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
