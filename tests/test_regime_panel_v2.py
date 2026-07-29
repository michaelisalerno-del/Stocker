from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocker_research.regime_panel_v2 import (
    EMISSION_FEATURES,
    add_cross_sectional_market_features,
    add_emission_features,
    bounded_source_hash,
    build_symbol_features,
    canonical_frame_hash,
    verify_source_hashes,
)


def _raw(
    symbol: str,
    ordinals: list[int],
    *,
    session: str = "2024-01-02",
    close_offset: float = 0.0,
) -> pd.DataFrame:
    start = pd.Timestamp(f"{session} 14:30:00", tz="UTC")
    rows = []
    for ordinal in ordinals:
        opening = 100.0 + ordinal + close_offset
        rows.append(
            {
                "timestamp": start + pd.Timedelta(minutes=5 * ordinal),
                "open": opening,
                "high": opening + 1.0,
                "low": opening - 1.0,
                "close": opening + 0.25,
                "volume": 1000.0 + ordinal,
            }
        )
    return pd.DataFrame(rows)


def test_symbol_builder_orders_rows_and_uses_unique_natural_keys() -> None:
    raw = _raw("TEST", [2, 0, 1])
    expected = {("TEST", "2024-01-02"): 3}

    frame, gaps = build_symbol_features(raw, symbol="TEST", expected_bars=expected)

    assert frame["bar_ordinal"].tolist() == [0, 1, 2]
    assert not frame[["symbol", "session", "bar_start_timestamp", "bar_ordinal"]].duplicated().any()
    assert (
        frame["bar_complete_timestamp"]
        .eq(frame["bar_start_timestamp"] + pd.Timedelta(minutes=5))
        .all()
    )
    assert gaps.empty


def test_rolling_features_reset_after_source_gap() -> None:
    raw = _raw("TEST", [0, 1, 3])
    raw.loc[raw["timestamp"].eq(pd.Timestamp("2024-01-02 14:45", tz="UTC")), "open"] = 200.0
    raw.loc[raw["timestamp"].eq(pd.Timestamp("2024-01-02 14:45", tz="UTC")), "close"] = 201.0

    frame, _ = build_symbol_features(
        raw,
        symbol="TEST",
        expected_bars={("TEST", "2024-01-02"): 4},
    )
    post_gap = frame.loc[frame["bar_ordinal"].eq(3)].iloc[0]

    assert post_gap["segment_bar_ordinal"] == 0
    assert post_gap["return_sum_3"] == pytest.approx(post_gap["bar_log_return"])
    assert post_gap["cumulative_historical_volume"] == pytest.approx(post_gap["volume"])


def test_off_grid_provider_row_is_ledgered_and_invalidates_session() -> None:
    raw = _raw("TEST", [0, 1])
    invalid = raw.iloc[[0]].copy()
    invalid["timestamp"] = pd.Timestamp("2024-01-02 14:33:19", tz="UTC")
    raw = pd.concat([raw, invalid], ignore_index=True)

    frame, gaps = build_symbol_features(
        raw,
        symbol="TEST",
        expected_bars={("TEST", "2024-01-02"): 2},
    )

    assert frame["bar_ordinal"].tolist() == [0, 1]
    assert frame["source_data_error_in_session"].all()
    assert not frame["session_source_complete"].any()
    assert gaps["gap_reason"].tolist() == ["invalid_non_five_minute_source_timestamp"]


def test_cross_section_uses_only_peers_available_at_timestamp() -> None:
    left, _ = build_symbol_features(
        _raw("LEFT", [0, 1]),
        symbol="LEFT",
        expected_bars={("LEFT", "2024-01-02"): 2},
    )
    right, _ = build_symbol_features(
        _raw("RIGHT", [0], close_offset=5.0),
        symbol="RIGHT",
        expected_bars={("RIGHT", "2024-01-02"): 2},
    )
    vti, _ = build_symbol_features(
        _raw("VTI", [0, 1], close_offset=10.0),
        symbol="VTI",
        expected_bars={("VTI", "2024-01-02"): 2},
    )

    panel = add_cross_sectional_market_features(pd.concat([left, right], ignore_index=True), vti)

    peer_counts = panel.set_index(["symbol", "bar_ordinal"])["market_peer_count"]
    assert peer_counts.loc[("LEFT", 0)] == 2
    assert peer_counts.loc[("LEFT", 1)] == 1


def test_all_fourteen_emissions_are_explicit_and_available_no_earlier_than_bar_completion() -> None:
    parts = []
    for index, symbol in enumerate(("A", "B")):
        frame, _ = build_symbol_features(
            _raw(symbol, list(range(12)), close_offset=float(index)),
            symbol=symbol,
            expected_bars={(symbol, "2024-01-02"): 12},
        )
        parts.append(frame)
    vti, _ = build_symbol_features(
        _raw("VTI", list(range(12)), close_offset=2.0),
        symbol="VTI",
        expected_bars={("VTI", "2024-01-02"): 12},
    )

    panel = add_emission_features(
        add_cross_sectional_market_features(pd.concat(parts, ignore_index=True), vti)
    )

    assert set(EMISSION_FEATURES).issubset(panel.columns)
    assert panel["feature_available_timestamp_max"].le(panel["bar_complete_timestamp"]).all()


def test_canonical_frame_hash_is_order_deterministic_and_full_precision() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["B", "A"],
            "session": ["2024-01-02", "2024-01-02"],
            "bar_start_timestamp": pd.to_datetime(
                ["2024-01-02 14:30Z", "2024-01-02 14:30Z"], utc=True
            ),
            "bar_ordinal": [0, 0],
            "feature": [np.nextafter(1.0, 2.0), 1.0],
        }
    )

    first = canonical_frame_hash(
        frame,
        columns=[
            "symbol",
            "session",
            "bar_start_timestamp",
            "bar_ordinal",
            "feature",
        ],
    )
    second = canonical_frame_hash(
        frame.iloc[::-1],
        columns=[
            "symbol",
            "session",
            "bar_start_timestamp",
            "bar_ordinal",
            "feature",
        ],
    )
    rounded = frame.copy()
    rounded["feature"] = rounded["feature"].round(12)
    rounded_hash = canonical_frame_hash(
        rounded,
        columns=[
            "symbol",
            "session",
            "bar_start_timestamp",
            "bar_ordinal",
            "feature",
        ],
    )

    assert first == second
    assert first != rounded_hash


def test_bounded_source_hash_and_verification(tmp_path: Path) -> None:
    source = tmp_path / "data.parquet"
    raw = _raw("TEST", [0, 1])
    raw.to_parquet(source, index=False)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")

    digest, rows = bounded_source_hash(source, start=start, end=end)
    verify_source_hashes({"TEST": source}, {"TEST": digest}, start=start, end=end)

    assert rows == 2
    assert len(digest) == hashlib.sha256().digest_size * 2
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_source_hashes(
            {"TEST": source},
            {"TEST": "0" * 64},
            start=start,
            end=end,
        )
