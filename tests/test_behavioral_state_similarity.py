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
    STATE_FINGERPRINT_FEATURE_COLUMNS,
    add_forward_response_columns,
    apply_state_gate_to_positions,
    build_behavioral_state_frame,
    build_decision_summary,
    build_horizon_events,
    build_permutation_baseline,
    extract_independent_events,
    label_behavioral_states,
    run_fingerprint_cross_symbol_similarity,
    run_same_state_cross_symbol_similarity,
    run_behavioral_state_similarity_lab,
    run_nearest_neighbor_oos_similarity,
    run_oos_state_response_test,
    run_oos_response_shape_similarity,
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


def test_horizon_specific_events_do_not_require_all_horizons() -> None:
    rows = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "timestamp": pd.date_range("2026-06-23 14:30", periods=2, freq="5min", tz="UTC"),
            "session_date": ["2026-06-23", "2026-06-23"],
            "bar_index_in_session": [12, 13],
            "can_evaluate_state": [True, True],
            "primary_state_label": ["dead_chop", "dead_chop"],
            "state_family": ["chop_compression", "chop_compression"],
            "stimulus_label": ["none", "none"],
            "forward_6_bar_return": [0.01, -0.01],
            "forward_6_bar_mfe": [0.02, 0.01],
            "forward_6_bar_mae": [-0.01, -0.02],
            "forward_24_bar_return": [pd.NA, 0.03],
            "forward_24_bar_mfe": [pd.NA, 0.04],
            "forward_24_bar_mae": [pd.NA, -0.01],
        }
    )

    events = build_horizon_events(rows, _config(horizons=(6, 24)))

    assert len(events[events["response_horizon"] == 6]) == 2
    assert len(events[events["response_horizon"] == 24]) == 1
    assert set(events.columns) >= {
        "state_subtype",
        "response_horizon",
        "response_return",
        "response_mfe",
        "response_mae",
        "path_return_1",
        "path_return_6",
        "final_return",
        "mfe",
        "mae",
        "time_to_mfe",
        "time_to_mae",
        "recoil_ratio",
        "continuation_score",
        "failure_score",
    }


def test_response_path_vectors_start_after_detection_bar() -> None:
    rows = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "timestamp": pd.date_range("2026-06-23 14:30", periods=4, freq="5min", tz="UTC"),
            "session_date": ["2026-06-23"] * 4,
            "bar_index_in_session": [10, 11, 12, 13],
            "can_evaluate_state": [True, True, True, True],
            "primary_state_label": ["dead_chop"] * 4,
            "state_family": ["chop_compression"] * 4,
            "state_subtype": ["dead_chop"] * 4,
            "stimulus_label": ["none"] * 4,
            "close": [100.0, 101.0, 103.0, 102.0],
            "high": [100.2, 101.5, 103.5, 102.5],
            "low": [99.8, 100.8, 102.8, 101.5],
            "rolling_intraday_range_pct": [0.01] * 4,
            "rolling_volatility_12": [0.005] * 4,
            "forward_2_bar_return": [0.03, pd.NA, pd.NA, pd.NA],
            "forward_2_bar_mfe": [0.035, pd.NA, pd.NA, pd.NA],
            "forward_2_bar_mae": [0.008, pd.NA, pd.NA, pd.NA],
        }
    )

    events = build_horizon_events(rows, _config(horizons=(2,)))
    row = events.iloc[0]

    assert row["path_return_1"] == pytest.approx(0.01)
    assert row["path_return_2"] == pytest.approx(0.03)
    assert row["final_return"] == pytest.approx(0.03)
    assert row["time_to_mfe"] == 2
    assert row["time_to_mae"] == 1


def test_independent_event_extraction_reduces_repeated_state_bars() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA"] * 6,
            "session_date": ["2026-06-23"] * 6,
            "bar_index_in_session": [10, 11, 12, 13, 16, 17],
            "primary_state_label": ["dead_chop"] * 6,
            "response_horizon": [6] * 6,
            "response_return": [0.0] * 6,
            "state_entry": [True, False, False, False, False, False],
            "stimulus_event": [False] * 6,
        }
    )

    independent = extract_independent_events(events, mode="non_overlapping_by_horizon")

    assert independent["raw_row_count"].iloc[0] == 6
    assert independent["independent_event_count"].iloc[0] == 2
    assert independent["bar_index_in_session"].tolist() == [10, 16]


def test_state_entry_events_enforce_horizon_embargo() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "session_date": ["2026-06-23"] * 4,
            "bar_index_in_session": [10, 12, 15, 17],
            "primary_state_label": ["extension_exhaustion"] * 4,
            "response_horizon": [6] * 4,
            "response_return": [-0.01, -0.02, -0.03, -0.04],
            "state_entry": [True, True, True, True],
            "stimulus_event": [True, True, True, True],
        }
    )

    independent = extract_independent_events(events, mode="state_entry_non_overlapping")

    assert independent["bar_index_in_session"].tolist() == [10, 17]


def test_permutation_label_baseline_preserves_state_counts() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4 + ["BBB"] * 4,
            "session_date": ["2026-06-23"] * 8,
            "bar_index_bucket": ["open"] * 4 + ["midday"] * 4,
            "time_of_day_bucket": ["open"] * 4 + ["midday"] * 4,
            "primary_state_label": ["a", "a", "b", "b", "a", "b", "b", "a"],
            "response_horizon": [6] * 8,
            "response_return": [0.01, 0.02, -0.01, -0.02, 0.03, -0.03, -0.01, 0.04],
        }
    )

    baseline = build_permutation_baseline(
        events,
        config=_config(horizons=(6,), random_seed=7),
        permutation_count=5,
    )

    assert set(baseline["state"]) == {"a", "b"}
    assert set(baseline["observed_event_count"]) == {4}
    assert set(baseline["permuted_event_count"]) == {4}
    assert baseline["permutation_p_value"].between(0.0, 1.0).all()


def test_same_state_cross_symbol_matching_is_strict_same_state_and_cross_symbol() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "CCC"],
            "timestamp": pd.to_datetime(
                [
                    "2026-06-23 14:30",
                    "2026-06-24 14:30",
                    "2026-06-25 14:30",
                    "2026-06-24 14:35",
                ],
                utc=True,
            ),
            "session_date": ["2026-06-23", "2026-06-24", "2026-06-25", "2026-06-24"],
            "time_of_day_bucket": ["morning"] * 4,
            "primary_state_label": [
                "initiative_buying_continuation",
                "initiative_buying_continuation",
                "initiative_buying_continuation",
                "dead_chop",
            ],
            "state_subtype": ["initiative_buying_controlled_pullback"] * 3 + ["dead_chop"],
            "stimulus_label": ["continuation_pressure"] * 3 + ["none"],
            "response_horizon": [6] * 4,
            "response_return": [0.03, 0.025, -0.02, 0.0],
            "response_mfe": [0.04, 0.03, 0.01, 0.002],
            "response_mae": [-0.005, -0.004, -0.03, -0.002],
            "path_return_1": [0.005, 0.004, -0.002, 0.0],
            "path_return_2": [0.010, 0.009, -0.004, 0.0],
            "path_return_3": [0.015, 0.014, -0.010, 0.0],
            "path_return_4": [0.020, 0.018, -0.015, 0.0],
            "path_return_5": [0.025, 0.022, -0.018, 0.0],
            "path_return_6": [0.030, 0.025, -0.020, 0.0],
            "bar_return": [0.01, 0.011, -0.01, 0.0],
            "prior_6_bar_return": [0.02, 0.021, -0.02, 0.0],
            "directional_efficiency_6": [0.8, 0.78, 0.2, 0.1],
        }
    )

    matches, summary, baselines = run_same_state_cross_symbol_similarity(
        events,
        feature_columns=["bar_return", "prior_6_bar_return", "directional_efficiency_6"],
        config=_config(horizons=(6,)),
        top_k=2,
    )

    assert not matches.empty
    assert (matches["source_symbol"] != matches["match_symbol"]).all()
    assert (matches["source_state"] == matches["match_state"]).all()
    assert set(matches["match_state"]) == {"initiative_buying_continuation"}
    assert (matches["match_state"] != "dead_chop").all()
    assert {"same_state_cross_symbol", "different_state_cross_symbol", "random_cross_symbol"}.issubset(
        set(baselines["baseline"])
    )
    assert summary.iloc[0]["state"] == "initiative_buying_continuation"


def test_state_fingerprint_feature_list_is_leakage_safe() -> None:
    expected = {
        "prior_3_bar_return",
        "prior_6_bar_return",
        "prior_12_bar_return",
        "directional_efficiency_6",
        "directional_efficiency_12",
        "close_location_value",
        "body_pct_of_range",
        "upper_wick_pct_of_range",
        "lower_wick_pct_of_range",
        "distance_from_vwap_pct",
        "distance_from_opening_range_mid_pct",
        "distance_from_opening_range_high_pct",
        "distance_from_opening_range_low_pct",
        "opening_range_width_pct",
        "rolling_intraday_range_pct",
        "compression_zscore",
        "relative_volume_at_bar_index",
        "relative_cumulative_volume",
        "state_age_bars",
        "time_of_day_bucket",
        "bar_index_bucket",
    }

    assert expected.issubset(set(STATE_FINGERPRINT_FEATURE_COLUMNS))
    forbidden_prefixes = ("forward_", "response_", "path_return_", "normalized_path_return_")
    assert not any(
        column.startswith(forbidden_prefixes) or column in {"final_return", "mfe", "mae"}
        for column in STATE_FINGERPRINT_FEATURE_COLUMNS
    )


def test_fingerprint_cross_symbol_matching_uses_observable_features() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "AAA"],
            "timestamp": pd.to_datetime(
                [
                    "2026-06-24 14:30",
                    "2026-06-23 14:30",
                    "2026-06-23 14:35",
                    "2026-06-22 14:30",
                ],
                utc=True,
            ),
            "session_date": ["2026-06-24", "2026-06-23", "2026-06-23", "2026-06-22"],
            "time_of_day_bucket": ["morning", "morning", "morning", "morning"],
            "bar_index_bucket": ["post_open", "post_open", "post_open", "post_open"],
            "primary_state_label": ["dead_chop", "extension_exhaustion", "dead_chop", "dead_chop"],
            "stimulus_label": ["none", "stall_or_upper_wick", "none", "none"],
            "response_horizon": [6, 6, 6, 6],
            "response_return": [0.01, 0.011, -0.03, 0.02],
            "response_mfe": [0.012, 0.013, 0.001, 0.03],
            "response_mae": [-0.002, -0.002, -0.04, -0.01],
            "path_return_1": [0.002, 0.0021, -0.010, 0.005],
            "path_return_2": [0.004, 0.0041, -0.015, 0.008],
            "path_return_3": [0.006, 0.0061, -0.020, 0.012],
            "path_return_4": [0.008, 0.0081, -0.025, 0.015],
            "path_return_5": [0.009, 0.0091, -0.028, 0.018],
            "path_return_6": [0.010, 0.0110, -0.030, 0.020],
            "prior_3_bar_return": [0.010, 0.011, -0.050, 0.090],
            "prior_6_bar_return": [0.020, 0.021, -0.080, 0.120],
            "prior_12_bar_return": [0.030, 0.031, -0.100, 0.150],
            "directional_efficiency_6": [0.70, 0.71, 0.20, 0.95],
            "directional_efficiency_12": [0.60, 0.61, 0.10, 0.90],
            "close_location_value": [0.55, 0.56, 0.10, 0.95],
            "body_pct_of_range": [0.40, 0.41, 0.05, 0.90],
            "upper_wick_pct_of_range": [0.20, 0.21, 0.80, 0.02],
            "lower_wick_pct_of_range": [0.20, 0.19, 0.05, 0.02],
            "distance_from_vwap_pct": [0.001, 0.0011, -0.020, 0.080],
            "distance_from_opening_range_mid_pct": [0.002, 0.0021, -0.030, 0.090],
            "distance_from_opening_range_high_pct": [-0.001, -0.0011, -0.040, 0.050],
            "distance_from_opening_range_low_pct": [0.004, 0.0041, -0.010, 0.110],
            "opening_range_width_pct": [0.010, 0.0101, 0.050, 0.090],
            "rolling_intraday_range_pct": [0.020, 0.0201, 0.070, 0.100],
            "compression_zscore": [-0.50, -0.49, 2.00, 3.00],
            "relative_volume_at_bar_index": [1.10, 1.11, 5.00, 6.00],
            "relative_cumulative_volume": [1.05, 1.06, 4.00, 6.00],
            "state_age_bars": [2, 2, 12, 20],
        }
    )

    matches, summary, baselines = run_fingerprint_cross_symbol_similarity(
        events,
        feature_columns=list(STATE_FINGERPRINT_FEATURE_COLUMNS),
        config=_config(horizons=(6,)),
        top_k=1,
    )

    aaa_match = matches[matches["source_symbol"].eq("AAA")].iloc[0]
    assert aaa_match["match_symbol"] == "BBB"
    assert aaa_match["match_state"] == "extension_exhaustion"
    assert (matches["source_symbol"] != matches["match_symbol"]).all()
    assert "fingerprint_cross_symbol" in set(summary["baseline"])
    assert {"random_cross_symbol", "different_state_cross_symbol", "same_symbol_random"}.issubset(
        set(baselines["baseline"])
    )


def test_oos_response_shape_similarity_fingerprint_fit_is_train_only() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "timestamp": pd.to_datetime(
                [
                    "2026-06-23 14:30",
                    "2026-06-23 14:35",
                    "2026-06-24 14:30",
                    "2026-06-24 14:35",
                ],
                utc=True,
            ),
            "session_date": ["2026-06-23", "2026-06-23", "2026-06-24", "2026-06-24"],
            "time_of_day_bucket": ["morning"] * 4,
            "bar_index_bucket": ["post_open"] * 4,
            "primary_state_label": ["dead_chop", "extension_exhaustion", "dead_chop", "dead_chop"],
            "response_horizon": [6] * 4,
            "response_return": [0.01, -0.01, 0.011, -0.02],
            "path_return_1": [0.002, -0.002, 0.0021, -0.004],
            "path_return_2": [0.004, -0.004, 0.0041, -0.008],
            "path_return_3": [0.006, -0.006, 0.0061, -0.012],
            "path_return_4": [0.008, -0.008, 0.0081, -0.016],
            "path_return_5": [0.009, -0.009, 0.0091, -0.018],
            "path_return_6": [0.010, -0.010, 0.0110, -0.020],
            "prior_6_bar_return": [0.020, -0.020, 0.021, -0.019],
            "directional_efficiency_6": [0.70, 0.20, 0.71, 0.19],
            "state_age_bars": [2, 3, 2, 3],
        }
    )

    summary = run_oos_response_shape_similarity(
        events,
        feature_columns=["prior_6_bar_return", "directional_efficiency_6", "state_age_bars"],
        config=_config(horizons=(6,), nearest_neighbors=1),
        similarity_mode="fingerprint",
    )

    assert not summary.empty
    assert set(summary["fit_scope"]) == {"train_only"}
    assert "fingerprint_cross_symbol" in set(summary["baseline"])
    assert int(summary["test_event_count"].max()) == 2


def test_oos_state_response_uses_train_rows_for_expectation() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "timestamp": pd.to_datetime(
                [
                    "2026-06-23 14:30",
                    "2026-06-23 14:35",
                    "2026-06-24 14:30",
                    "2026-06-24 14:35",
                ],
                utc=True,
            ),
            "session_date": ["2026-06-23", "2026-06-23", "2026-06-24", "2026-06-24"],
            "primary_state_label": ["extension_exhaustion"] * 4,
            "response_horizon": [6] * 4,
            "response_return": [0.02, 0.01, -0.03, -0.02],
            "response_mfe": [0.03, 0.02, 0.01, 0.01],
            "response_mae": [-0.01, -0.01, -0.04, -0.03],
        }
    )

    oos = run_oos_state_response_test(
        events,
        config=_config(horizons=(6,), min_state_occurrences=1),
        split_mode="walk_forward",
    )
    row = oos.iloc[0]

    assert row["train_median_return"] > 0
    assert row["test_median_return"] < 0
    assert row["expected_direction"] == 1
    assert row["directional_accuracy"] == 0.0


def test_nearest_neighbor_oos_reports_train_only_fit_scope() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "timestamp": pd.date_range("2026-06-23 14:30", periods=4, freq="5min", tz="UTC"),
            "session_date": ["2026-06-23"] * 2 + ["2026-06-24"] * 2,
            "primary_state_label": ["dead_chop"] * 4,
            "response_horizon": [6] * 4,
            "response_return": [0.01, 0.02, -0.01, -0.02],
            "prior_6_bar_return": [0.1, 0.2, 100.0, 101.0],
            "directional_efficiency_6": [0.2, 0.3, 10.0, 11.0],
        }
    )

    summary = run_nearest_neighbor_oos_similarity(
        events,
        feature_columns=["prior_6_bar_return", "directional_efficiency_6"],
        config=_config(horizons=(6,), nearest_neighbors=1),
    )

    assert summary.iloc[0]["fit_scope"] == "train_only"
    assert summary.iloc[0]["train_row_count"] == 2
    assert summary.iloc[0]["test_row_count"] == 2


def test_state_gated_template_does_not_alter_exit_logic() -> None:
    positions = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    allowed = pd.Series([False, True, False, False, False, True, False])

    gated = apply_state_gate_to_positions(positions, allowed)

    assert gated.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]


def test_pipeline_passed_is_not_evidence_supported() -> None:
    decision = build_decision_summary(
        oos_state_response=pd.DataFrame(),
        permutation_baseline=pd.DataFrame(),
        concentration_warnings=pd.DataFrame(),
        template_overlay_summary=pd.DataFrame(),
        pipeline_passed=True,
        config=_config(),
    )

    assert decision["pipeline_passed"] is True
    assert decision["label_similarity_supported"] is False
    assert decision["fingerprint_similarity_supported"] is False
    assert decision["state_similarity_supported"] is False
    assert decision["oos_similarity_supported"] is False
    assert decision["template_overlay_supported"] is False
    assert decision["decision"].startswith("reject_")


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
            "--event-mode",
            "state_entry_non_overlapping",
            "--permutation-count",
            "3",
            "--random-seed",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "state_counts" in result.output
    assert "summary_json_path" in result.output
    run_dirs = list((tmp_path / "reports").glob("behavioral_state_similarity_*"))
    assert len(run_dirs) == 1
    expected_files = {
        "summary.json",
        "summary.md",
        "decision.json",
        "events.csv",
        "horizon_events.csv",
        "independent_events.csv",
        "same_state_cross_symbol_matches.csv",
        "response_shape_similarity_summary.csv",
        "response_shape_baselines.csv",
        "fingerprint_cross_symbol_matches.csv",
        "fingerprint_similarity_summary.csv",
        "fingerprint_response_shape_baselines.csv",
        "oos_response_shape_similarity.csv",
        "horizon_state_summary.csv",
        "permutation_baseline.csv",
        "oos_state_response.csv",
        "nearest_neighbor_oos_summary.csv",
        "stimulus_response_matrix.csv",
        "manual_audit_examples.csv",
        "state_match_examples.csv",
    }
    assert expected_files.issubset({path.name for path in run_dirs[0].iterdir()})
    payload = json.loads((run_dirs[0] / "summary.json").read_text(encoding="utf-8"))
    assert "pipeline_passed" in payload
    assert "research_passed" not in payload
    assert payload["label_similarity_supported"] is False
    assert payload["fingerprint_similarity_supported"] is False
    assert payload["state_similarity_supported"] is False
    assert payload["oos_similarity_supported"] is False
    assert payload["decision"].startswith("reject_")
