from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_data.storage import DatasetKey, dataset_path, write_parquet
from stocker_research.behavioral_state_similarity import (
    BehavioralStateConfig,
    add_forward_response_columns,
    build_behavioral_state_frame,
    label_behavioral_states,
    run_behavioral_state_similarity_lab,
    run_nearest_neighbor_similarity,
    summarize_state_responses,
)


def _session_timestamps(session_date: str, bars: int = 79) -> pd.DatetimeIndex:
    return pd.date_range(f"{session_date} 13:30", periods=bars, freq="5min", tz="UTC")


def _frame_from_closes(
    closes: list[float],
    *,
    session_date: str = "2026-06-23",
    symbol: str = "TEST.US",
    volume: float = 10_000.0,
) -> pd.DataFrame:
    close = pd.Series(closes, dtype="float")
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.08
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.08
    return pd.DataFrame(
        {
            "source": "eodhd",
            "symbol": symbol,
            "instrument_type": "stock",
            "timeframe": "5m",
            "timestamp": _session_timestamps(session_date, len(closes)),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "currency": "USD",
            "timezone": "UTC",
        }
    )


def _config(**overrides: object) -> BehavioralStateConfig:
    values = {
        "timeframe": "5m",
        "market_calendar": "XNYS",
        "horizons": (6, 12, 24),
        "min_bars_after_open": 6,
        "entry_cutoff_before_close_minutes": 30,
        "relative_volume_lookback_sessions": 2,
        "direction_windows": (3, 6, 12),
        "nearest_neighbors": 3,
        "min_state_occurrences": 2,
        "min_symbols_per_state": 2,
    }
    values.update(overrides)
    return BehavioralStateConfig(**values)


def test_behavioral_features_do_not_use_future_rows() -> None:
    first = _frame_from_closes(
        [100 + index * 0.02 for index in range(79)],
        session_date="2026-06-23",
    )
    second = _frame_from_closes(
        [101 + index * 0.03 for index in range(79)],
        session_date="2026-06-24",
    )
    frame = pd.concat([first, second], ignore_index=True)
    eval_index = 79 + 30

    original = label_behavioral_states(
        build_behavioral_state_frame(frame, symbol="TEST.US", config=_config()),
        _config(),
    )
    mutated = frame.copy()
    mutated.loc[eval_index + 1 :, ["open", "high", "low", "close", "volume"]] = 10_000.0
    recomputed = label_behavioral_states(
        build_behavioral_state_frame(mutated, symbol="TEST.US", config=_config()),
        _config(),
    )

    compared_columns = [
        column
        for column in original.columns
        if not column.startswith("forward_") and column not in {"session_warning_reason"}
    ]
    pd.testing.assert_frame_equal(
        original.loc[:eval_index, compared_columns],
        recomputed.loc[:eval_index, compared_columns],
        check_dtype=False,
    )


def test_forward_response_columns_are_same_session_only() -> None:
    first = _frame_from_closes([100 + index for index in range(8)], session_date="2026-06-23")
    second = _frame_from_closes([200 + index for index in range(8)], session_date="2026-06-24")
    frame = build_behavioral_state_frame(
        pd.concat([first, second], ignore_index=True),
        symbol="TEST.US",
        config=_config(horizons=(6,)),
    )

    with_forward = add_forward_response_columns(frame, horizons=(6,))

    assert pd.isna(with_forward.loc[4, "forward_6_bar_return"])
    assert with_forward.loc[1, "forward_6_bar_return"] == pytest.approx(
        with_forward.loc[7, "close"] / with_forward.loc[1, "close"] - 1.0
    )


def test_liquidation_failed_low_recovery_label() -> None:
    closes = [
        100.0,
        99.5,
        99.0,
        98.5,
        98.0,
        97.5,
        97.0,
        96.5,
        96.0,
        95.5,
        95.0,
        94.5,
        94.0,
        94.4,
        94.8,
    ] + [95.0 + index * 0.02 for index in range(64)]
    frame = _frame_from_closes(closes)
    frame.loc[13, "low"] = 93.9
    frame.loc[13, "close"] = 94.6
    features = build_behavioral_state_frame(frame, symbol="TEST.US", config=_config())

    labeled = label_behavioral_states(features, _config())

    assert bool(labeled.loc[13, "state_liquidation_failed_low_recovery"])
    assert labeled.loc[13, "primary_state_label"] == "liquidation_failed_low_recovery"


def test_extension_exhaustion_label() -> None:
    closes = [100.0 + index * 0.22 for index in range(13)] + [102.55, 102.58]
    closes += [102.6 - index * 0.01 for index in range(64)]
    frame = _frame_from_closes(closes)
    frame.loc[13, "high"] = frame.loc[13, "close"] + 0.75
    frame.loc[13, "close"] = frame.loc[13, "open"] + 0.02
    features = build_behavioral_state_frame(frame, symbol="TEST.US", config=_config())

    labeled = label_behavioral_states(features, _config())

    assert bool(labeled.loc[13, "state_extension_exhaustion"])
    assert labeled.loc[13, "primary_state_label"] == "extension_exhaustion"


def test_dead_chop_label() -> None:
    closes = [100.0 + (0.03 if index % 2 else -0.03) for index in range(79)]
    frame = _frame_from_closes(closes)
    features = build_behavioral_state_frame(frame, symbol="TEST.US", config=_config())

    labeled = label_behavioral_states(features, _config())

    assert bool(labeled.loc[20, "state_dead_chop"])
    assert labeled.loc[20, "primary_state_label"] == "dead_chop"


def test_response_summary_counts_symbols_and_horizons() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "session_date": ["2026-06-23", "2026-06-23", "2026-06-24", "2026-06-24"],
            "primary_state_label": ["extension_exhaustion"] * 4,
            "forward_6_bar_return": [-0.01, -0.02, 0.005, -0.004],
            "forward_6_bar_mfe": [0.002, 0.003, 0.008, 0.001],
            "forward_6_bar_mae": [-0.012, -0.022, -0.001, -0.006],
            "forward_6_bar_abs_return": [0.01, 0.02, 0.005, 0.004],
            "forward_12_bar_return": [-0.02, -0.01, -0.003, 0.002],
            "forward_12_bar_mfe": [0.002, 0.004, 0.001, 0.003],
            "forward_12_bar_mae": [-0.025, -0.015, -0.006, -0.002],
            "forward_12_bar_abs_return": [0.02, 0.01, 0.003, 0.002],
        }
    )

    summary = summarize_state_responses(events, _config(horizons=(6, 12)))
    row = summary[(summary["state"] == "extension_exhaustion") & (summary["horizon"] == 6)].iloc[0]

    assert row["occurrence_count"] == 4
    assert row["symbol_count"] == 2
    assert row["median_return"] == pytest.approx(-0.007)
    assert row["win_rate"] == pytest.approx(0.25)


def test_nearest_neighbor_excludes_forward_columns() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "timestamp": pd.date_range("2026-06-23 14:30", periods=3, freq="5min", tz="UTC"),
            "session_date": ["2026-06-23"] * 3,
            "primary_state_label": ["dead_chop"] * 3,
            "prior_6_bar_return": [0.001, 0.002, -0.001],
            "directional_efficiency_6": [0.2, 0.25, 0.3],
            "forward_6_bar_return": [0.001, -0.001, 0.002],
        }
    )

    with pytest.raises(ValueError, match="response columns"):
        run_nearest_neighbor_similarity(
            events,
            feature_columns=["prior_6_bar_return", "forward_6_bar_return"],
            config=_config(horizons=(6,)),
        )


def test_lab_runner_writes_reports(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for offset, symbol in enumerate(["AAA.US", "BBB.US", "CCC.US"]):
        closes = [100.0 + offset + index * 0.08 for index in range(79)]
        frame = _frame_from_closes(closes, symbol=symbol)
        key = DatasetKey(source="eodhd", instrument_type="stock", symbol=symbol, timeframe="5m")
        write_parquet(frame, dataset_path(key, data_dir=data_dir))

    result = run_behavioral_state_similarity_lab(
        data_dir=data_dir,
        symbols=["AAA.US", "BBB.US", "CCC.US"],
        output_dir=tmp_path / "reports",
        config=_config(horizons=(6, 12), min_state_occurrences=1),
    )

    assert result.summary_json_path.exists()
    assert result.summary_markdown_path.exists()
    assert result.event_csv_path.exists()
    assert result.state_summary_csv_path.exists()
    assert result.match_summary_csv_path.exists()
    payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert payload["symbols_completed"] == ["AAA.US", "BBB.US", "CCC.US"]
    assert "state_response_summary" in payload
    assert "nearest_neighbor_summary" in payload


def test_behavioral_state_similarity_cli_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for symbol in ["AAA.US", "BBB.US"]:
        key = DatasetKey(source="eodhd", instrument_type="stock", symbol=symbol, timeframe="5m")
        write_parquet(
            _frame_from_closes([100 + index * 0.08 for index in range(79)], symbol=symbol),
            dataset_path(key, data_dir=data_dir),
        )
    config_path = tmp_path / "research.yaml"
    config_path.write_text(f"data:\n  data_dir: {data_dir}\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "research",
            "behavioral-state-similarity",
            "--symbol",
            "AAA.US",
            "--symbol",
            "BBB.US",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "state_counts" in result.output
    assert "summary_json_path" in result.output
