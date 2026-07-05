from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_core.config import EODHDConfig
from stocker_data.storage import DatasetKey, dataset_path, read_parquet, write_parquet
from stocker_data.vendors.eodhd import EODHDFetchResult
from stocker_research.frozen_template_technique_v0 import (
    FrozenTemplateTechniqueConfig,
    run_frozen_template_technique_v0,
)


def _session_timestamps(session_date: str, bars: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{session_date} 13:30", periods=bars, freq="5min", tz="UTC")


def _controlled_pullback_frame(symbol: str = "TEST") -> pd.DataFrame:
    closes = pd.Series(
        [
            100.00,
            100.10,
            100.24,
            100.42,
            100.68,
            100.95,
            101.18,
            101.40,
            101.62,
            101.82,
            101.96,
            102.08,
            102.18,
            102.05,
            101.96,
            102.04,
            102.12,
            102.18,
            102.24,
            102.30,
            102.35,
            102.40,
            102.44,
            102.48,
            102.52,
            102.56,
            102.60,
            102.64,
            102.68,
            102.72,
            102.76,
            102.80,
            102.84,
            102.88,
            102.92,
            102.96,
        ],
        dtype="float",
    )
    open_ = closes.shift(1).fillna(closes.iloc[0])
    high = pd.concat([open_, closes], axis=1).max(axis=1) + 0.05
    low = pd.concat([open_, closes], axis=1).min(axis=1) - 0.05
    frame = pd.DataFrame(
        {
            "source": "eodhd",
            "symbol": symbol,
            "instrument_type": "stock",
            "timeframe": "5m",
            "timestamp": _session_timestamps("2026-06-23", len(closes)),
            "open": open_,
            "high": high,
            "low": low,
            "close": closes,
            "volume": 10_000.0,
            "currency": "USD",
            "timezone": "UTC",
        }
    )
    frame.loc[3:12, "volume"] = 25_000.0
    frame.loc[13:15, "volume"] = 8_000.0
    return frame


def _dirty_frame(symbol: str) -> pd.DataFrame:
    clean = _controlled_pullback_frame(symbol)
    duplicate = clean.iloc[[12]].copy()
    duplicate["close"] = pd.to_numeric(duplicate["close"]) + 0.01
    duplicate["high"] = pd.to_numeric(duplicate["high"]) + 0.01
    return (
        pd.concat([clean.iloc[5:], duplicate, clean.iloc[:5]], ignore_index=True)
        .reset_index(drop=True)
    )


def _write_dirty_local_bars(data_dir: Path, symbols: tuple[str, ...]) -> None:
    for symbol in symbols:
        key = DatasetKey(
            source="eodhd",
            instrument_type="stock",
            symbol=symbol,
            timeframe="5m",
        )
        write_parquet(_dirty_frame(symbol), dataset_path(key, data_dir=data_dir))


def test_frozen_template_technique_cleans_bars_then_replays_templates(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    symbols = ("AAA", "BBB")
    _write_dirty_local_bars(data_dir, symbols)

    result = run_frozen_template_technique_v0(
        data_dir=data_dir,
        symbols=symbols,
        output_dir=tmp_path / "reports",
        config=FrozenTemplateTechniqueConfig(
            from_date="2026-06-23",
            to_date="2026-06-24",
            download_eodhd=False,
            min_events_for_similarity=1,
        ),
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    cleaner = pd.read_csv(result.bar_cleaner_report_csv_path)
    event_rows = pd.read_csv(result.event_rows_csv_path)
    transfer_summary = json.loads(result.template_transfer_summary_json_path.read_text())

    assert summary["research_only"] is True
    assert summary["live_ordering_enabled"] is False
    assert summary["order_placement"] == "disabled"
    assert summary["edge_claimed"] is False
    assert summary["download_eodhd"] is False
    assert summary["bar_cleaner"]["symbols_cleaned"] == 2
    assert cleaner["duplicate_timestamps_removed"].sum() == 2
    assert not event_rows.empty
    assert transfer_summary["mode"] == "frozen-template-transfer-replay"

    key = DatasetKey(source="eodhd", instrument_type="stock", symbol="AAA", timeframe="5m")
    cleaned = read_parquet(dataset_path(key, data_dir=data_dir))
    timestamps = pd.to_datetime(cleaned["timestamp"], utc=True)
    assert timestamps.is_monotonic_increasing
    assert not timestamps.duplicated().any()


def test_frozen_template_technique_download_stage_uses_eodhd_fetcher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from stocker_data.vendors import eodhd

    data_dir = tmp_path / "data"
    calls: list[dict[str, object]] = []

    def fake_fetch_intraday_to_storage(**kwargs) -> EODHDFetchResult:
        symbol = str(kwargs["symbol"]).upper()
        calls.append(
            {
                "symbol": symbol,
                "interval": kwargs["interval"],
                "merge": kwargs["merge"],
                "audit": kwargs["audit"],
            }
        )
        key = DatasetKey(
            source="eodhd",
            instrument_type="stock",
            symbol=symbol,
            timeframe=str(kwargs["interval"]),
        )
        output_path = dataset_path(key, data_dir=kwargs["data_dir"])
        write_parquet(_dirty_frame(symbol), output_path)
        return EODHDFetchResult(
            output_path=output_path,
            catalog_path=Path(kwargs["data_dir"]) / "catalog.duckdb",
            raw_paths=[],
            rows_fetched=37,
            rows_saved=37,
            min_timestamp="2026-06-23 13:30:00+00:00",
            max_timestamp="2026-06-23 16:25:00+00:00",
            issues=[],
        )

    monkeypatch.setattr(eodhd, "fetch_intraday_to_storage", fake_fetch_intraday_to_storage)

    result = run_frozen_template_technique_v0(
        data_dir=data_dir,
        symbols=("AAA",),
        output_dir=tmp_path / "reports",
        eodhd_config=EODHDConfig(enabled=True),
        config=FrozenTemplateTechniqueConfig(
            from_date="2026-06-23",
            to_date="2026-06-24",
            download_eodhd=True,
            merge=True,
            audit=True,
            min_events_for_similarity=1,
        ),
    )

    fetch_report = pd.read_csv(result.fetch_report_csv_path)
    cleaner = pd.read_csv(result.bar_cleaner_report_csv_path)

    assert calls == [{"symbol": "AAA", "interval": "5m", "merge": True, "audit": True}]
    assert fetch_report.iloc[0]["status"] == "fetched"
    assert int(fetch_report.iloc[0]["rows_fetched"]) == 37
    assert int(cleaner.iloc[0]["duplicate_timestamps_removed"]) == 1


def test_frozen_template_technique_cli_smoke_without_download(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    symbols = ("AAA", "BBB")
    _write_dirty_local_bars(data_dir, symbols)
    config_path = tmp_path / "research.yaml"
    config_path.write_text(f"data:\n  data_dir: {data_dir}\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "research",
            "frozen-template-technique",
            "--symbols",
            "AAA,BBB",
            "--from",
            "2026-06-23",
            "--to",
            "2026-06-24",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "reports"),
            "--no-download-eodhd",
            "--min-events-for-similarity",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "frozen_template_technique_v0" in result.output
    assert "template_transfer_summary_json_path" in result.output
