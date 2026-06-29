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
    bars_per_session: int = 10,
    overrides: dict[tuple[int, int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    overrides = overrides or {}
    default_closes = [100.0, 99.0, 98.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    for session_index in range(sessions):
        session_open = pd.Timestamp("2024-01-02 14:30", tz="UTC") + pd.Timedelta(
            days=session_index
        )
        for bar_index in range(bars_per_session):
            timestamp = session_open + pd.Timedelta(minutes=5 * bar_index)
            close = default_closes[bar_index % len(default_closes)]
            values = {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1000.0,
            }
            values.update(overrides.get((session_index, bar_index), {}))
            rows.append({"timestamp": timestamp, **values})
    return pd.DataFrame(rows)


def _params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "entry_mode": "reclaim",
        "entry_buffer_bps": 0,
        "min_reclaim_distance_bps": 0,
        "reclaim_lookback_bars": 3,
        "rejection_lookback_bars": 3,
        "max_reclaim_distance_from_vwap_pct": 0.10,
        "max_rejection_distance_from_vwap_pct": 0.02,
        "min_bounce_distance_bps": 0,
        "min_bars_after_open": 1,
        "max_hold_bars": 3,
        "min_relative_volume": 0.0,
        "exit_mode": "time_stop",
        "timeframe": "5m",
        "market_calendar": None,
        "relative_volume_lookback_sessions": 1,
        "entry_cutoff_before_close_minutes": 0,
        "flatten_before_close_minutes": 0,
    }
    params.update(overrides)
    return params


def _vwap_template():
    return get_template("vwap_reclaim_rejection")


def test_no_entry_before_required_intraday_window() -> None:
    frame = _intraday_frame()

    positions = _vwap_template().generate_positions(frame, _params(min_bars_after_open=4))

    assert positions.iloc[3] == 0.0
    assert positions.sum() == 0.0


def test_reclaim_entry_triggers_only_after_completed_close_above_vwap() -> None:
    frame = _intraday_frame()

    positions = _vwap_template().generate_positions(frame, _params())
    signals = _vwap_template().generate_signals(frame, _params())

    assert positions.iloc[2] == 0.0
    assert positions.iloc[3] == 1.0
    assert bool(signals.loc[3, "reclaim_cross_above_vwap"]) is True
    assert signals.loc[3, "entry_mode"] == "reclaim"


def test_reclaim_does_not_trigger_without_recent_below_vwap() -> None:
    frame = _intraday_frame(
        overrides={
            (0, 0): {"close": 100.0, "high": 100.0, "low": 100.0, "open": 100.0},
            (0, 1): {"close": 101.0, "high": 101.0, "low": 101.0, "open": 101.0},
            (0, 2): {"close": 102.0, "high": 102.0, "low": 102.0, "open": 102.0},
            (0, 3): {"close": 103.0, "high": 103.0, "low": 103.0, "open": 103.0},
        }
    )

    positions = _vwap_template().generate_positions(frame, _params())

    assert positions.sum() == 0.0


def test_rejection_entry_triggers_after_valid_vwap_test_and_bounce() -> None:
    frame = _intraday_frame(
        overrides={
            (0, 0): {"close": 100.0, "high": 100.0, "low": 100.0, "open": 100.0},
            (0, 1): {"close": 102.0, "high": 102.0, "low": 102.0, "open": 102.0},
            (0, 2): {"close": 101.4, "high": 101.8, "low": 100.9, "open": 101.4},
            (0, 3): {"close": 102.4, "high": 102.4, "low": 102.4, "open": 102.4},
        }
    )

    positions = _vwap_template().generate_positions(frame, _params(entry_mode="rejection"))
    signals = _vwap_template().generate_signals(frame, _params(entry_mode="rejection"))

    assert positions.iloc[2] == 0.0
    assert positions.iloc[3] == 1.0
    assert bool(signals.loc[3, "rejection_valid_test"]) is True
    assert bool(signals.loc[3, "rejection_bounce"]) is True
    assert signals.loc[3, "entry_mode"] == "rejection"


def test_rejection_does_not_trigger_when_vwap_is_lost() -> None:
    frame = _intraday_frame(
        overrides={
            (0, 0): {"close": 100.0, "high": 100.0, "low": 100.0, "open": 100.0},
            (0, 1): {"close": 102.0, "high": 102.0, "low": 102.0, "open": 102.0},
            (0, 2): {"close": 99.5, "high": 101.0, "low": 99.5, "open": 99.5},
            (0, 3): {"close": 102.4, "high": 102.4, "low": 102.4, "open": 102.4},
        }
    )

    positions = _vwap_template().generate_positions(frame, _params(entry_mode="rejection"))

    assert positions.sum() == 0.0


def test_relative_volume_filter_blocks_and_permits_entries() -> None:
    frame = _intraday_frame(
        sessions=2,
        overrides={(1, 3): {"volume": 500.0}},
    )
    permitted = frame.copy()
    permitted.loc[_bar_index(1, 3, 10), "volume"] = 1500.0

    template = _vwap_template()
    blocked = template.generate_positions(frame, _params(min_relative_volume=1.0))
    allowed = template.generate_positions(permitted, _params(min_relative_volume=1.0))

    assert blocked.iloc[_bar_index(1, 3, 10)] == 0.0
    assert allowed.iloc[_bar_index(1, 3, 10)] == 1.0


def test_max_distance_from_vwap_filter_blocks_stretched_reclaim() -> None:
    frame = _intraday_frame(
        overrides={
            (0, 1): {"close": 80.0, "high": 80.0, "low": 80.0, "open": 80.0},
            (0, 2): {"close": 78.0, "high": 78.0, "low": 78.0, "open": 78.0},
            (0, 3): {"close": 101.0, "high": 101.0, "low": 101.0, "open": 101.0},
        }
    )

    positions = _vwap_template().generate_positions(
        frame,
        _params(max_reclaim_distance_from_vwap_pct=0.05),
    )

    assert positions.iloc[3] == 0.0


def test_time_stop_exits_work() -> None:
    frame = _intraday_frame()

    positions = _vwap_template().generate_positions(frame, _params(max_hold_bars=2))

    assert positions.iloc[3] == 1.0
    assert positions.iloc[4] == 1.0
    assert positions.iloc[5] == 0.0


def test_vwap_lost_exits_work() -> None:
    frame = _intraday_frame(
        overrides={
            (0, 4): {"close": 97.0, "high": 97.0, "low": 97.0, "open": 97.0},
        }
    )

    positions = _vwap_template().generate_positions(frame, _params(exit_mode="vwap_lost"))
    signals = _vwap_template().generate_signals(frame, _params(exit_mode="vwap_lost"))

    assert positions.iloc[3] == 1.0
    assert positions.iloc[4] == 0.0
    assert bool(signals.loc[4, "vwap_lost_exit"]) is True


def test_no_entry_window_blocks_late_entries() -> None:
    frame = _intraday_frame(
        overrides={
            (0, 1): {"close": 100.0, "high": 100.0, "low": 100.0, "open": 100.0},
            (0, 2): {"close": 100.0, "high": 100.0, "low": 100.0, "open": 100.0},
            (0, 3): {"close": 100.0, "high": 100.0, "low": 100.0, "open": 100.0},
            (0, 4): {"close": 100.0, "high": 100.0, "low": 100.0, "open": 100.0},
            (0, 5): {"close": 100.0, "high": 100.0, "low": 100.0, "open": 100.0},
            (0, 6): {"close": 100.0, "high": 100.0, "low": 100.0, "open": 100.0},
            (0, 7): {"close": 101.0, "high": 101.0, "low": 100.2, "open": 101.0},
            (0, 8): {"close": 102.4, "high": 102.4, "low": 102.4, "open": 102.4},
        }
    )

    positions = _vwap_template().generate_positions(
        frame,
        _params(entry_mode="rejection", entry_cutoff_before_close_minutes=10),
    )

    assert positions.iloc[8] == 0.0
    assert positions.sum() == 0.0


def test_positions_are_deterministic() -> None:
    frame = _intraday_frame()
    template = _vwap_template()

    first = template.generate_positions(frame, _params(max_hold_bars=2))
    second = template.generate_positions(frame, _params(max_hold_bars=2))

    assert first.equals(second)


def test_context_window_positions_match_full_frame_positions_for_eval_rows() -> None:
    frame = _intraday_frame(sessions=3)
    template = _vwap_template()
    params = _params(min_relative_volume=1.0)
    eval_start = _bar_index(2, 0, 10)
    eval_end = _bar_index(3, 0, 10)

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
    frame = _intraday_frame(sessions=3)
    template = _vwap_template()
    params = _params(min_relative_volume=1.0)
    eval_start = _bar_index(1, 0, 10)
    eval_end = _bar_index(2, 0, 10)

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


def test_generated_signals_include_vwap_diagnostic_columns() -> None:
    frame = _intraday_frame()

    signals = _vwap_template().generate_signals(frame, _params(entry_mode="both"))

    assert {
        "timestamp",
        "signal",
        "target_position",
        "entry",
        "exit",
        "template_name",
        "parameter_set_id",
        "session_vwap",
        "distance_from_vwap",
        "entry_mode",
        "raw_entry",
        "time_stop_exit",
        "vwap_lost_exit",
        "reclaim_recent_below_vwap",
        "reclaim_cross_above_vwap",
        "rejection_valid_test",
        "rejection_bounce",
        "relative_volume_ok",
        "distance_filter_ok",
    }.issubset(signals.columns)
    assert len(signals) == len(frame)


def test_template_is_registered_and_hypothesis_validation_allows_it() -> None:
    template = _vwap_template()
    hypothesis = Hypothesis.model_validate(
        {
            "id": "vwap_reclaim_rejection_unit",
            "name": "VWAP Reclaim Rejection Unit",
            "description": "Unit test hypothesis for VWAP reclaim and rejection.",
            "hypothesis_version": 1,
            "market_universe": "unit_test",
            "instrument_type": "stock",
            "timeframe": "5m",
            "data_source": "manual",
            "template": "vwap_reclaim_rejection",
            "signal_family": "vwap_reclaim_rejection",
            "direction": "long_only",
            "entry_logic": "Long after VWAP reclaim or VWAP rejection continuation.",
            "exit_logic": "Exit by time stop or VWAP lost.",
            "holding_period": "Session-flat intraday.",
            "expected_edge_reason": "Unit test fixture.",
            "invalidation_rules": ["Unit test invalidation."],
            "parameter_space": {
                "entry_mode": ["reclaim", "rejection"],
                "entry_buffer_bps": [0],
                "min_reclaim_distance_bps": [0],
                "max_hold_bars": [3],
                "min_relative_volume": [0.0],
                "exit_mode": ["time_stop"],
            },
            "maximum_parameter_sets": 2,
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

    assert template.name == "vwap_reclaim_rejection"
    assert hypothesis.template == "vwap_reclaim_rejection"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"entry_mode": "short_reclaim"}, "entry_mode must be one of"),
        ({"entry_buffer_bps": -1}, "entry_buffer_bps must be non-negative"),
        ({"min_reclaim_distance_bps": -1}, "min_reclaim_distance_bps must be non-negative"),
        ({"reclaim_lookback_bars": 0}, "reclaim_lookback_bars must be positive"),
        ({"rejection_lookback_bars": 0}, "rejection_lookback_bars must be positive"),
        (
            {"max_reclaim_distance_from_vwap_pct": -0.1},
            "max_reclaim_distance_from_vwap_pct must be non-negative",
        ),
        (
            {"max_rejection_distance_from_vwap_pct": -0.1},
            "max_rejection_distance_from_vwap_pct must be non-negative",
        ),
        ({"max_hold_bars": 0}, "max_hold_bars must be positive"),
        ({"min_relative_volume": -0.1}, "min_relative_volume must be non-negative"),
        ({"exit_mode": "trailing_stop"}, "exit_mode must be one of"),
    ],
)
def test_invalid_params_fail_clearly(override: dict[str, object], message: str) -> None:
    frame = _intraday_frame()

    with pytest.raises(ValueError, match=message):
        _vwap_template().generate_positions(frame, _params(**override))
