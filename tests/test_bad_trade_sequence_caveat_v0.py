from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.bad_trade_sequence_caveat_v0 import (
    BadTradeSequenceCaveatConfig,
    attach_prior_event_context,
    run_bad_trade_sequence_caveat_lab,
)


def _trade(
    *,
    symbol: str,
    timestamp: str,
    session_date: str,
    month: str,
    personality: str,
    event_state: str,
    net_r: float,
    filter_rule: str = "range_zscore <= 0",
    stop_model: str = "fixed_75bps",
    risk_bps: float = 100.0,
    close_location_value: float = 0.60,
    return_zscore: float = 0.0,
    prior_12_bar_return: float = 0.0,
    relative_volume_at_bar_index: float = 1.0,
    same_direction_other_symbol_count_15m: int = 3,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "session_date": session_date,
        "month": month,
        "personality": personality,
        "event_state": event_state,
        "net_r": net_r,
        "exit_reason": "time_exit",
        "bar_index_in_session": 20,
        "time_x_vwap_regime": "morning|above",
        "vwap_x_range_regime": "above|high_range",
        "compression_x_efficiency_regime": "compressed|mixed_efficiency",
        "regime_field": "time_x_vwap_regime",
        "regime_value": "morning|above",
        "filter_rule": filter_rule,
        "stop_model": stop_model,
        "risk_bps": risk_bps,
        "close_location_value": close_location_value,
        "return_zscore": return_zscore,
        "prior_12_bar_return": prior_12_bar_return,
        "relative_volume_at_bar_index": relative_volume_at_bar_index,
        "same_direction_other_symbol_count_15m": same_direction_other_symbol_count_15m,
    }


def _event(
    *,
    symbol: str,
    timestamp: str,
    session_date: str,
    bar_index_in_session: int,
    event_state: str,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "session_date": session_date,
        "bar_index_in_session": bar_index_in_session,
        "event_state": event_state,
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    selected_dir = tmp_path / "walk_forward_selected_filter_exit_v0_20260701T000000Z"
    event_dir = tmp_path / "state_event_detector_v0_20260701T000000Z"
    selected_dir.mkdir()
    event_dir.mkdir()

    trades = [
        _trade(
            symbol="AAA",
            timestamp="2026-01-05T15:00:00Z",
            session_date="2026-01-05",
            month="2026-01",
            personality="pullback_continuation",
            event_state="controlled_pullback_after_bullish_impulse",
            net_r=0.60,
            close_location_value=0.30,
        ),
        _trade(
            symbol="BBB",
            timestamp="2026-02-05T15:00:00Z",
            session_date="2026-02-05",
            month="2026-02",
            personality="pullback_continuation",
            event_state="controlled_pullback_after_bullish_impulse",
            net_r=0.40,
            close_location_value=0.45,
        ),
        _trade(
            symbol="DDD",
            timestamp="2026-03-05T15:00:00Z",
            session_date="2026-03-05",
            month="2026-03",
            personality="reclaim_reversal",
            event_state="liquidation_failed_low_reclaim",
            net_r=0.30,
            close_location_value=0.55,
        ),
        _trade(
            symbol="EEE",
            timestamp="2026-04-05T15:00:00Z",
            session_date="2026-04-05",
            month="2026-04",
            personality="reclaim_reversal",
            event_state="liquidation_failed_low_reclaim",
            net_r=0.20,
            close_location_value=0.65,
        ),
        _trade(
            symbol="AAA",
            timestamp="2026-05-05T15:00:00Z",
            session_date="2026-05-05",
            month="2026-05",
            personality="pullback_continuation",
            event_state="controlled_pullback_after_bullish_impulse",
            net_r=-1.00,
            close_location_value=-99.0,
        ),
        _trade(
            symbol="BBB",
            timestamp="2026-05-06T15:00:00Z",
            session_date="2026-05-06",
            month="2026-05",
            personality="pullback_continuation",
            event_state="controlled_pullback_after_bullish_impulse",
            net_r=-1.20,
            close_location_value=-88.0,
        ),
        _trade(
            symbol="CCC",
            timestamp="2026-06-05T15:00:00Z",
            session_date="2026-06-05",
            month="2026-06",
            personality="pullback_continuation",
            event_state="controlled_pullback_after_bullish_impulse",
            net_r=-0.80,
            close_location_value=-77.0,
        ),
        _trade(
            symbol="FFF",
            timestamp="2026-06-07T15:00:00Z",
            session_date="2026-06-07",
            month="2026-06",
            personality="pullback_continuation",
            event_state="controlled_pullback_after_bullish_impulse",
            net_r=-0.70,
            close_location_value=-66.0,
        ),
        _trade(
            symbol="GGG",
            timestamp="2026-06-08T15:00:00Z",
            session_date="2026-06-08",
            month="2026-06",
            personality="pullback_continuation",
            event_state="controlled_pullback_after_bullish_impulse",
            net_r=-0.90,
            close_location_value=-55.0,
        ),
        _trade(
            symbol="HHH",
            timestamp="2026-06-09T15:00:00Z",
            session_date="2026-06-09",
            month="2026-06",
            personality="pullback_continuation",
            event_state="controlled_pullback_after_bullish_impulse",
            net_r=-1.10,
            close_location_value=-44.0,
        ),
        _trade(
            symbol="DDD",
            timestamp="2026-06-06T15:00:00Z",
            session_date="2026-06-06",
            month="2026-06",
            personality="reclaim_reversal",
            event_state="liquidation_failed_low_reclaim",
            net_r=0.20,
            close_location_value=99.0,
        ),
    ]
    events = [
        _event(
            symbol="AAA",
            timestamp="2026-01-05T14:55:00Z",
            session_date="2026-01-05",
            bar_index_in_session=19,
            event_state="failed_bullish_impulse_recoil",
        ),
        _event(
            symbol="BBB",
            timestamp="2026-02-05T14:55:00Z",
            session_date="2026-02-05",
            bar_index_in_session=19,
            event_state="failed_bullish_impulse_recoil",
        ),
        _event(
            symbol="AAA",
            timestamp="2026-05-05T14:55:00Z",
            session_date="2026-05-05",
            bar_index_in_session=19,
            event_state="failed_bullish_impulse_recoil",
        ),
        _event(
            symbol="BBB",
            timestamp="2026-05-06T14:55:00Z",
            session_date="2026-05-06",
            bar_index_in_session=19,
            event_state="failed_bullish_impulse_recoil",
        ),
        _event(
            symbol="CCC",
            timestamp="2026-06-05T14:55:00Z",
            session_date="2026-06-05",
            bar_index_in_session=19,
            event_state="failed_bullish_impulse_recoil",
        ),
        _event(
            symbol="FFF",
            timestamp="2026-06-07T14:55:00Z",
            session_date="2026-06-07",
            bar_index_in_session=19,
            event_state="failed_bullish_impulse_recoil",
        ),
        _event(
            symbol="GGG",
            timestamp="2026-06-08T14:55:00Z",
            session_date="2026-06-08",
            bar_index_in_session=19,
            event_state="failed_bullish_impulse_recoil",
        ),
        _event(
            symbol="HHH",
            timestamp="2026-06-09T14:55:00Z",
            session_date="2026-06-09",
            bar_index_in_session=19,
            event_state="failed_bullish_impulse_recoil",
        ),
    ]
    pd.DataFrame(trades).to_csv(selected_dir / "trades.csv", index=False)
    pd.DataFrame(events).to_csv(event_dir / "event_rows.csv", index=False)
    return selected_dir, event_dir


def test_prior_context_uses_only_earlier_same_symbol_session_events() -> None:
    trades = pd.DataFrame(
        [
            _trade(
                symbol="AAA",
                timestamp="2026-05-05T15:00:00Z",
                session_date="2026-05-05",
                month="2026-05",
                personality="pullback_continuation",
                event_state="controlled_pullback_after_bullish_impulse",
                net_r=-1.0,
            )
        ]
    )
    events = pd.DataFrame(
        [
            _event(
                symbol="AAA",
                timestamp="2026-05-05T14:55:00Z",
                session_date="2026-05-05",
                bar_index_in_session=19,
                event_state="failed_bullish_impulse_recoil",
            ),
            _event(
                symbol="AAA",
                timestamp="2026-05-05T15:00:00Z",
                session_date="2026-05-05",
                bar_index_in_session=20,
                event_state="liquidation_failed_low_reclaim",
            ),
            _event(
                symbol="AAA",
                timestamp="2026-05-05T15:05:00Z",
                session_date="2026-05-05",
                bar_index_in_session=21,
                event_state="failed_bounce_active_liquidation",
            ),
        ]
    )

    enriched = attach_prior_event_context(trades, events)

    row = enriched.iloc[0]
    assert row["prev_event_personality"] == "impulse_recoil"
    assert row["prev_event_state"] == "failed_bullish_impulse_recoil"
    assert row["bars_since_prev_event"] == 1
    assert pd.isna(row["prev2_event_personality"])


def test_prior_context_replaces_existing_prior_columns() -> None:
    trades = pd.DataFrame(
        [
            {
                **_trade(
                    symbol="AAA",
                    timestamp="2026-05-05T15:00:00Z",
                    session_date="2026-05-05",
                    month="2026-05",
                    personality="pullback_continuation",
                    event_state="controlled_pullback_after_bullish_impulse",
                    net_r=-1.0,
                ),
                "prev_event_personality": "stale_value",
                "prev_event_state": "stale_state",
                "bars_since_prev_event": 999,
            }
        ]
    )
    events = pd.DataFrame(
        [
            _event(
                symbol="AAA",
                timestamp="2026-05-05T14:55:00Z",
                session_date="2026-05-05",
                bar_index_in_session=19,
                event_state="failed_bullish_impulse_recoil",
            ),
        ]
    )

    enriched = attach_prior_event_context(trades, events)

    assert enriched.columns.is_unique
    row = enriched.iloc[0]
    assert row["prev_event_personality"] == "impulse_recoil"
    assert row["prev_event_state"] == "failed_bullish_impulse_recoil"
    assert row["bars_since_prev_event"] == 1


def test_bad_trade_sequence_caveat_report_writes_expected_files_and_strict_status(
    tmp_path: Path,
) -> None:
    selected_dir, event_dir = _write_inputs(tmp_path)

    result = run_bad_trade_sequence_caveat_lab(
        input_selected_report_dir=selected_dir,
        input_event_dir=event_dir,
        output_dir=tmp_path / "out",
        config=BadTradeSequenceCaveatConfig(
            train_months=("2026-01", "2026-02", "2026-03", "2026-04"),
            test_months=("2026-05", "2026-06"),
            numeric_quantiles=(0.20, 0.50, 0.80),
            random_iterations=20,
            random_seed=11,
        ),
    )

    expected_files = {
        "summary.md",
        "summary.json",
        "decision.json",
        "sequence_caveat_results.csv",
        "current_personality_caveats.csv",
        "prior_sequence_caveats.csv",
        "numeric_threshold_caveats.csv",
        "strict_validation_results.csv",
        "trade_caveat_flags.csv",
    }
    assert expected_files.issubset({path.name for path in result.output_dir.iterdir()})

    decision = json.loads(result.decision_json_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "continue_research_oos_warning_not_train_validated"
    assert decision["research_only"] is True
    assert decision["edge_claimed"] is False

    strict = pd.read_csv(result.strict_validation_results_csv_path)
    prior_rule = strict[
        strict["rule_name"].eq("prior impulse_recoil -> current pullback_continuation")
    ].iloc[0]
    assert prior_rule["strict_status"] == "oos_only_not_train_supported"
    assert bool(prior_rule["strict_oos_supported"]) is True
    assert bool(prior_rule["strict_train_supported"]) is False


def test_numeric_threshold_candidates_use_train_values_only(tmp_path: Path) -> None:
    selected_dir, event_dir = _write_inputs(tmp_path)

    result = run_bad_trade_sequence_caveat_lab(
        input_selected_report_dir=selected_dir,
        input_event_dir=event_dir,
        output_dir=tmp_path / "out",
        config=BadTradeSequenceCaveatConfig(
            train_months=("2026-01", "2026-02", "2026-03", "2026-04"),
            test_months=("2026-05", "2026-06"),
            numeric_quantiles=(0.20,),
            random_iterations=5,
            random_seed=7,
        ),
    )

    numeric = pd.read_csv(result.numeric_threshold_caveats_csv_path)
    weak_close = numeric[numeric["rule_name"].str.startswith("weak close: close_location_value <=")]

    assert not weak_close.empty
    assert weak_close["selected_threshold"].max() >= 0.30
    assert weak_close["selected_threshold"].min() > -1.0


def test_bad_trade_sequence_caveat_cli_smoke(tmp_path: Path) -> None:
    selected_dir, event_dir = _write_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "bad-trade-sequence-caveat",
            "--input-selected-report-dir",
            str(selected_dir),
            "--input-event-dir",
            str(event_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--train-months",
            "2026-01,2026-02,2026-03,2026-04",
            "--test-months",
            "2026-05,2026-06",
            "--numeric-quantiles",
            "0.2,0.5,0.8",
            "--random-iterations",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "bad_trade_sequence_caveat_v0" in result.output
