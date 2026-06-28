import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_data.audit import create_audit_report
from stocker_data.catalog import query_dataset, scan_catalog
from stocker_data.ingest import import_csv, parse_column_mapping
from stocker_data.storage import DatasetKey, dataset_metadata, dataset_path, load_dataset
from stocker_data.validate import validate_ohlcv
from stocker_research.baselines import create_baseline_report

FIXTURES = Path(__file__).parent / "fixtures" / "market_data"


def test_import_csv_auto_maps_columns_and_writes_partitioned_parquet(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    result = import_csv(
        file_path=FIXTURES / "clean_ohlcv.csv",
        data_dir=data_dir,
        symbol="AAPL",
        source="manual",
        timeframe="1d",
        instrument_type="stock",
        timezone="America/New_York",
        currency="USD",
    )

    expected_path = (
        data_dir
        / "processed"
        / "source=manual"
        / "instrument_type=stock"
        / "symbol=AAPL"
        / "timeframe=1d"
        / "data.parquet"
    )
    assert result.path == expected_path
    assert result.rows == 5
    assert result.error_count == 0
    assert result.warning_count == 0
    assert any(issue.code == "calendar_gap_check_skipped" for issue in result.issues)

    frame = load_dataset(
        DatasetKey(source="manual", instrument_type="stock", symbol="AAPL", timeframe="1d"),
        data_dir=data_dir,
    )
    assert frame["symbol"].unique().tolist() == ["AAPL"]
    assert frame["source"].unique().tolist() == ["manual"]
    assert str(frame["timestamp"].dt.tz) == "America/New_York"
    assert frame["timestamp"].is_monotonic_increasing
    assert (
        dataset_path(
            DatasetKey(source="manual", instrument_type="stock", symbol="AAPL", timeframe="1d"),
            data_dir=data_dir,
        )
        == expected_path
    )

    catalog = json.loads((data_dir / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["datasets"][0]["symbol"] == "AAPL"
    assert catalog["datasets"][0]["row_count"] == 5


def test_parse_explicit_column_mapping() -> None:
    assert parse_column_mapping("timestamp=Date,open=O,close=C") == {
        "timestamp": "Date",
        "open": "O",
        "close": "C",
    }


def test_validation_detects_duplicate_bad_ohlc_negative_volume_and_gaps() -> None:
    duplicate = pd.read_csv(FIXTURES / "duplicate_timestamp.csv")
    duplicate_issues = validate_ohlcv(
        duplicate, timeframe="1m", timezone="America/New_York", require_timezone=True
    )
    assert any(issue.code == "duplicate_timestamp" for issue in duplicate_issues)

    bad = pd.read_csv(FIXTURES / "bad_ohlc.csv")
    bad_issues = validate_ohlcv(
        bad, timeframe="1d", timezone="America/New_York", require_timezone=True
    )
    assert any(issue.code == "ohlc_inconsistent" for issue in bad_issues)
    assert any(issue.code == "non_positive_price" for issue in bad_issues)
    assert any(issue.code == "negative_volume" for issue in bad_issues)

    gap = pd.read_csv(FIXTURES / "missing_gap.csv")
    gap_issues = validate_ohlcv(
        gap,
        timeframe="1d",
        timezone="America/New_York",
        require_timezone=True,
        market_calendar="XNYS",
    )
    assert any(issue.code == "timestamp_gap" for issue in gap_issues)

    naive = pd.read_csv(FIXTURES / "timezone_naive.csv")
    naive_issues = validate_ohlcv(
        naive, timeframe="1m", timezone="America/New_York", require_timezone=True
    )
    assert any(issue.code == "timezone_naive" for issue in naive_issues)


def test_catalog_metadata_and_duckdb_query(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    import_csv(
        file_path=FIXTURES / "clean_ohlcv.csv",
        data_dir=data_dir,
        symbol="AAPL",
        source="manual",
        timeframe="1d",
        instrument_type="stock",
        timezone="America/New_York",
        currency="USD",
    )

    entries = scan_catalog(data_dir=data_dir)
    assert len(entries) == 1
    assert entries[0].symbol == "AAPL"
    assert entries[0].row_count == 5

    metadata = dataset_metadata(
        DatasetKey(source="manual", instrument_type="stock", symbol="AAPL", timeframe="1d"),
        data_dir=data_dir,
    )
    assert metadata.row_count == 5
    assert metadata.min_timestamp is not None
    assert metadata.max_timestamp is not None

    query_result = query_dataset(
        entries[0].path,
        "select count(*) as rows, max(close) as max_close",
    )
    assert query_result.to_dict("records") == [{"rows": 5, "max_close": 105.25}]


def test_audit_report_creation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    import_csv(
        file_path=FIXTURES / "clean_ohlcv.csv",
        data_dir=data_dir,
        symbol="AAPL",
        source="manual",
        timeframe="1d",
        instrument_type="stock",
        timezone="America/New_York",
        currency="USD",
    )

    report = create_audit_report(
        data_dir=data_dir,
        symbol="AAPL",
        timeframe="1d",
        source="manual",
        instrument_type="stock",
    )

    assert report.markdown_path.exists()
    assert report.json_path.exists()
    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "AAPL"
    assert payload["row_count"] == 5
    assert payload["passed"] is True
    assert "Return Distribution" in report.markdown_path.read_text(encoding="utf-8")


def test_baseline_report_creation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    import_csv(
        file_path=FIXTURES / "clean_ohlcv.csv",
        data_dir=data_dir,
        symbol="AAPL",
        source="manual",
        timeframe="1d",
        instrument_type="stock",
        timezone="America/New_York",
        currency="USD",
    )

    report = create_baseline_report(
        data_dir=data_dir,
        symbol="AAPL",
        timeframe="1d",
        source="manual",
        instrument_type="stock",
        spread_bps=1.0,
        commission_bps=0.5,
        slippage_bps=0.5,
    )

    assert report.markdown_path.exists()
    assert report.json_path.exists()
    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert {result["name"] for result in payload["results"]} == {
        "buy_and_hold",
        "always_flat",
        "random_entry_exit",
        "sma_momentum",
        "mean_reversion",
    }
    assert all("net_total_return" in result for result in payload["results"])


def test_data_cli_import_catalog_show_audit_and_baseline(tmp_path: Path) -> None:
    runner = CliRunner()
    data_dir = tmp_path / "data"

    import_result = runner.invoke(
        app,
        [
            "data",
            "import-csv",
            "--file",
            str(FIXTURES / "clean_ohlcv.csv"),
            "--symbol",
            "AAPL",
            "--source",
            "manual",
            "--timeframe",
            "1d",
            "--instrument-type",
            "stock",
            "--timezone",
            "America/New_York",
            "--currency",
            "USD",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert import_result.exit_code == 0, import_result.output

    catalog_result = runner.invoke(app, ["data", "catalog", "--data-dir", str(data_dir)])
    assert catalog_result.exit_code == 0, catalog_result.output
    assert "AAPL" in catalog_result.output

    show_result = runner.invoke(
        app,
        [
            "data",
            "show",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert show_result.exit_code == 0, show_result.output
    assert "row_count" in show_result.output

    audit_result = runner.invoke(
        app,
        [
            "data",
            "audit",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert audit_result.exit_code == 0, audit_result.output

    baseline_result = runner.invoke(
        app,
        [
            "research",
            "baseline",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--data-dir",
            str(data_dir),
        ],
    )
    assert baseline_result.exit_code == 0, baseline_result.output
    assert "baseline" in baseline_result.output.lower()
