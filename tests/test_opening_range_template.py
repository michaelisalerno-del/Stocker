import pandas as pd
import pytest

from stocker_research.hypothesis import Hypothesis
from stocker_research.templates import get_template
from stocker_research.windows import build_evaluation_window


def _bar_index(session_index: int, bar_index: int, bars_per_session: int) -> int:
    return session_index * bars_per_session + bar_index


def _intraday_frame(
    *,
    sessions: int = 1,
    bars_per_session: int = 8,
    overrides: dict[tuple[int, int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    overrides = overrides or {}
    for session_index in range(sessions):
        session_open = pd.Timestamp("2024-01-02 14:30", tz="UTC") + pd.Timedelta(
            days=session_index
        )
        for bar_index in range(bars_per_session):
            timestamp = session_open + pd.Timedelta(minutes=5 * bar_index)
            if bar_index == 0:
                values = {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.1}
            elif bar_index == 1:
                values = {"open": 100.1, "high": 101.0, "low": 99.8, "close": 100.4}
            elif bar_index == 2:
                values = {"open": 100.4, "high": 100.8, "low": 99.7, "close": 100.2}
            else:
                values = {"open": 100.2, "high": 100.9, "low": 100.0, "close": 100.6}
            values["volume"] = 1000.0
            values.update(overrides.get((session_index, bar_index), {}))
            rows.append({"timestamp": timestamp, **values})
    return pd.DataFrame(rows)


def _params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "opening_minutes": 15,
        "breakout_buffer_bps": 0,
        "min_bars_after_open": 1,
        "max_hold_bars": 6,
        "min_relative_volume": 0.0,
        "max_opening_range_width_pct": 1.0,
        "exit_mode": "time_stop",
        "timeframe": "5m",
        "market_calendar": None,
        "relative_volume_lookback_sessions": 1,
        "entry_cutoff_before_close_minutes": 0,
        "flatten_before_close_minutes": 0,
    }
    params.update(overrides)
    return params


def _opening_range_template():
    return get_template("opening_range_breakout")


def test_no_signal_before_opening_range_completion() -> None:
    frame = _intraday_frame(
        overrides={
            (0, 2): {"high": 102.0, "close": 101.8},
            (0, 3): {"high": 101.5, "close": 100.8},
        }
    )

    positions = _opening_range_template().generate_positions(frame, _params())

    assert positions.iloc[:3].eq(0.0).all()
    assert positions.iloc[3] == 0.0


def test_long_signal_after_breakout_above_completed_opening_range_high() -> None:
    frame = _intraday_frame(overrides={(0, 3): {"high": 101.8, "close": 101.4}})

    positions = _opening_range_template().generate_positions(frame, _params())

    assert positions.iloc[2] == 0.0
    assert positions.iloc[3] == 1.0


def test_breakout_buffer_blocks_small_breakout() -> None:
    frame = _intraday_frame(overrides={(0, 3): {"high": 101.2, "close": 101.02}})
    template = _opening_range_template()

    no_buffer = template.generate_positions(frame, _params(breakout_buffer_bps=0))
    with_buffer = template.generate_positions(frame, _params(breakout_buffer_bps=5))

    assert no_buffer.iloc[3] == 1.0
    assert with_buffer.iloc[3] == 0.0


def test_relative_volume_filter_blocks_and_permits_entries() -> None:
    frame = _intraday_frame(
        sessions=2,
        overrides={
            (1, 3): {"high": 101.8, "close": 101.4, "volume": 500.0},
        },
    )
    permitted = frame.copy()
    permitted.loc[_bar_index(1, 3, 8), "volume"] = 1500.0

    template = _opening_range_template()
    blocked_positions = template.generate_positions(frame, _params(min_relative_volume=1.0))
    permitted_positions = template.generate_positions(
        permitted,
        _params(min_relative_volume=1.0),
    )

    assert blocked_positions.iloc[_bar_index(1, 3, 8)] == 0.0
    assert permitted_positions.iloc[_bar_index(1, 3, 8)] == 1.0


def test_max_opening_range_width_filter_blocks_overly_wide_sessions() -> None:
    frame = _intraday_frame(
        overrides={
            (0, 0): {"high": 106.0, "low": 94.0, "close": 100.0},
            (0, 1): {"high": 107.0, "low": 93.0, "close": 100.0},
            (0, 2): {"high": 105.0, "low": 94.5, "close": 100.0},
            (0, 3): {"high": 108.0, "close": 108.0},
        }
    )

    positions = _opening_range_template().generate_positions(
        frame,
        _params(max_opening_range_width_pct=0.04),
    )

    assert positions.iloc[3] == 0.0


def test_min_bars_after_open_blocks_early_entries() -> None:
    frame = _intraday_frame(overrides={(0, 3): {"high": 101.8, "close": 101.4}})

    positions = _opening_range_template().generate_positions(
        frame,
        _params(min_bars_after_open=4),
    )

    assert positions.iloc[3] == 0.0
    assert positions.sum() == 0.0


def test_time_stop_exits_after_max_hold_bars() -> None:
    frame = _intraday_frame(
        overrides={
            (0, 3): {"high": 101.8, "close": 101.4},
            (0, 4): {"high": 101.9, "close": 101.5},
            (0, 5): {"high": 101.7, "close": 101.3},
        }
    )

    positions = _opening_range_template().generate_positions(
        frame,
        _params(max_hold_bars=2),
    )

    assert positions.iloc[3] == 1.0
    assert positions.iloc[4] == 1.0
    assert positions.iloc[5] == 0.0


def test_no_entries_in_no_entry_window() -> None:
    frame = _intraday_frame(overrides={(0, 6): {"high": 101.8, "close": 101.4}})

    positions = _opening_range_template().generate_positions(
        frame,
        _params(entry_cutoff_before_close_minutes=10),
    )

    assert positions.iloc[6] == 0.0
    assert positions.sum() == 0.0


def test_positions_are_deterministic() -> None:
    frame = _intraday_frame(overrides={(0, 3): {"high": 101.8, "close": 101.4}})
    template = _opening_range_template()

    first = template.generate_positions(frame, _params(max_hold_bars=2))
    second = template.generate_positions(frame, _params(max_hold_bars=2))

    assert first.equals(second)


def test_context_window_positions_match_full_frame_positions_for_eval_rows() -> None:
    frame = _intraday_frame(
        sessions=3,
        overrides={
            (2, 3): {"high": 101.8, "close": 101.4, "volume": 1500.0},
        },
    )
    template = _opening_range_template()
    params = _params(min_relative_volume=1.0)
    eval_start = _bar_index(2, 0, 8)
    eval_end = _bar_index(3, 0, 8)

    full_positions = template.generate_positions(frame, params).iloc[eval_start:eval_end]
    context_window = build_evaluation_window(
        frame,
        template,
        params,
        eval_start=eval_start,
        eval_end=eval_end,
    )

    assert context_window.eval_positions.equals(full_positions.reset_index(drop=True))


def test_mutating_future_rows_after_eval_end_does_not_change_eval_positions() -> None:
    frame = _intraday_frame(
        sessions=3,
        overrides={
            (1, 3): {"high": 101.8, "close": 101.4, "volume": 1500.0},
            (2, 3): {"high": 101.8, "close": 101.4, "volume": 1500.0},
        },
    )
    template = _opening_range_template()
    params = _params(min_relative_volume=1.0)
    eval_start = _bar_index(1, 0, 8)
    eval_end = _bar_index(2, 0, 8)

    original = build_evaluation_window(
        frame,
        template,
        params,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    mutated = frame.copy()
    mutated.loc[eval_end:, ["high", "low", "close", "volume"]] = 10000.0
    changed = build_evaluation_window(
        mutated,
        template,
        params,
        eval_start=eval_start,
        eval_end=eval_end,
    )

    assert original.eval_positions.equals(changed.eval_positions)


def test_generated_signals_include_opening_range_columns() -> None:
    frame = _intraday_frame(overrides={(0, 3): {"high": 101.8, "close": 101.4}})

    signals = _opening_range_template().generate_signals(frame, _params())

    assert {
        "timestamp",
        "signal",
        "target_position",
        "entry",
        "exit",
        "template_name",
        "parameter_set_id",
        "opening_range_complete",
        "opening_range_high",
        "opening_range_low",
        "opening_range_mid",
        "breakout_threshold",
        "raw_entry",
        "time_stop_exit",
        "range_reclaim_exit",
    }.issubset(signals.columns)
    assert len(signals) == len(frame)


def test_template_is_registered_and_hypothesis_validation_allows_it() -> None:
    template = _opening_range_template()
    hypothesis = Hypothesis.model_validate(
        {
            "id": "opening_range_breakout_unit",
            "name": "Opening Range Breakout Unit",
            "description": "Unit test hypothesis for opening range breakout.",
            "hypothesis_version": 1,
            "market_universe": "unit_test",
            "instrument_type": "stock",
            "timeframe": "5m",
            "data_source": "manual",
            "template": "opening_range_breakout",
            "signal_family": "opening_range_breakout",
            "direction": "long_only",
            "entry_logic": "Long after a completed opening range breaks higher.",
            "exit_logic": "Exit by time stop or range reclaim failure.",
            "holding_period": "Session-flat intraday.",
            "expected_edge_reason": "Unit test fixture.",
            "invalidation_rules": ["Unit test invalidation."],
            "parameter_space": {
                "opening_minutes": [15],
                "breakout_buffer_bps": [0],
                "min_bars_after_open": [1],
                "max_hold_bars": [6],
                "min_relative_volume": [1.0],
                "max_opening_range_width_pct": [0.04],
                "exit_mode": ["time_stop"],
            },
            "maximum_parameter_sets": 1,
            "costs": {"spread_bps": 1.0, "commission_bps": 0.2, "slippage_bps": 1.0},
            "risk": {"max_drawdown": 0.25},
            "walkforward": {
                "mode": "rolling",
                "train_bars": 20,
                "test_bars": 8,
                "embargo_bars": 0,
                "step_bars": 8,
                "minimum_rows": 28,
            },
            "created_at": "2026-06-29T00:00:00Z",
        }
    )

    assert template.name == "opening_range_breakout"
    assert hypothesis.template == "opening_range_breakout"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"opening_minutes": 0}, "opening_minutes must be positive"),
        ({"breakout_buffer_bps": -1}, "breakout_buffer_bps must be non-negative"),
        ({"min_bars_after_open": -1}, "min_bars_after_open must be non-negative"),
        ({"max_hold_bars": 0}, "max_hold_bars must be positive"),
        ({"min_relative_volume": -0.1}, "min_relative_volume must be non-negative"),
        (
            {"max_opening_range_width_pct": 0},
            "max_opening_range_width_pct must be positive",
        ),
        ({"exit_mode": "trailing_stop"}, "exit_mode must be one of"),
    ],
)
def test_invalid_params_fail_clearly(override: dict[str, object], message: str) -> None:
    frame = _intraday_frame()

    with pytest.raises(ValueError, match=message):
        _opening_range_template().generate_positions(frame, _params(**override))
