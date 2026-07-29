from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.walk_forward_personality_filter_exit_v0 import (
    WalkForwardSelectedFilterExitConfig,
)
from stocker_research.walk_forward_staged_mixed_regime_caveat_exit_v0 import (
    StagedMixedRegimeCaveatExitConfig,
    _apply_caveat_rules,
    _cap_exit_sweep_with_personality_floor,
    _decision,
    _entry_policy_diagnostics,
    _personality_acceptance_book,
    _prior_replay_personality_acceptance_book,
    _select_staged_train_caveat_book,
    run_staged_mixed_regime_caveat_exit_lab,
)


def _event_row(
    *,
    symbol: str,
    timestamp: str,
    session_date: str,
    bar_index: int,
    event_state: str = "controlled_pullback_after_bullish_impulse",
    regime: str = "compressed|mixed_efficiency",
    bar_return: float = 0.006,
    forward_return: float = 0.010,
    forward_mfe: float = 0.012,
    forward_mae: float = -0.001,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "session_date": session_date,
        "bar_index_in_session": bar_index,
        "time_of_day_bucket": "morning",
        "event_state": event_state,
        "compression_x_efficiency_regime": regime,
        "vwap_x_range_regime": "above|high_range",
        "time_x_vwap_regime": "morning|above",
        "volume_x_vwap_regime": "normal_relative_volume|below",
        "bar_return": bar_return,
        "prior_6_bar_return": -0.003,
        "return_zscore": 0.1,
        "distance_from_vwap_pct": 0.004,
        "distance_from_session_low_pct": 0.006,
        "distance_from_session_high_pct": -0.012,
        "distance_from_recent_low_pct": 0.005,
        "distance_from_recent_high_pct": -0.011,
        "distance_from_opening_range_low_pct": 0.006,
        "distance_from_opening_range_high_pct": -0.012,
        "close_location_value": 0.72,
        "upper_wick_pct_of_range": 0.1,
        "lower_wick_pct_of_range": 0.1,
        "body_pct_of_range": 0.7,
        "directional_efficiency_6": 0.6,
        "directional_efficiency_12": 0.65,
        "rolling_intraday_range_pct": 0.015,
        "compression_zscore": 0.1,
        "range_zscore": 0.2,
        "relative_volume_at_bar_index": 1.0,
        "relative_cumulative_volume": 1.0,
        "same_direction_other_symbol_count_15m": 3,
        "same_personality_other_symbol_count_15m": 2,
        "same_direction_other_symbol_count_30m": 3,
        "same_personality_other_symbol_count_30m": 2,
        "forward_12_bar_return": forward_return,
        "forward_12_bar_mfe": forward_mfe,
        "forward_12_bar_mae": forward_mae,
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    event_dir = tmp_path / "state_event_detector_v0" / "run"
    discovery_dir = tmp_path / "personality_discovery_v0" / "run"
    caveat_dir = tmp_path / "bad_trade_sequence_caveat_v0" / "run"
    event_dir.mkdir(parents=True)
    discovery_dir.mkdir(parents=True)
    caveat_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    for month in ("2026-01", "2026-02"):
        for index in range(8):
            rows.append(
                _event_row(
                    symbol=symbols[index % len(symbols)],
                    timestamp=f"{month}-{1 + index // 4:02d}T14:{30 + index % 4:02d}:00Z",
                    session_date=f"{month}-{1 + index // 4:02d}",
                    bar_index=10 + index,
                    bar_return=0.006 + index * 0.0001,
                )
            )
    rows.extend(
        [
            _event_row(
                symbol="AAA",
                timestamp="2026-03-05T14:55:00Z",
                session_date="2026-03-05",
                bar_index=19,
                event_state="failed_bullish_impulse_recoil",
                bar_return=-0.004,
                forward_return=-0.004,
                forward_mfe=0.001,
                forward_mae=-0.008,
            ),
            _event_row(
                symbol="AAA",
                timestamp="2026-03-05T15:00:00Z",
                session_date="2026-03-05",
                bar_index=20,
                forward_return=-0.010,
                forward_mfe=0.001,
                forward_mae=-0.010,
            ),
            _event_row(
                symbol="BBB",
                timestamp="2026-03-06T15:00:00Z",
                session_date="2026-03-06",
                bar_index=20,
            ),
            _event_row(
                symbol="CCC",
                timestamp="2026-03-07T15:00:00Z",
                session_date="2026-03-07",
                bar_index=20,
            ),
        ]
    )
    pd.DataFrame(rows).to_csv(event_dir / "event_rows.csv", index=False)

    pd.DataFrame(
        [
            {
                "personality": "pullback_continuation",
                "horizon": 12,
                "regime_field": "vwap_x_range_regime",
                "regime_value": "above|high_range",
                "filter_rule": "bar_return >= 0",
                "feature": "bar_return",
                "operator": ">=",
                "threshold": 0.0,
                "filtered_test_same_result_rate": 0.80,
                "test_lift_vs_personality": 0.20,
                "test_lift_vs_regime": 0.10,
                "retained_test_count": 20,
            },
            {
                "personality": "pullback_continuation",
                "horizon": 12,
                "regime_field": "compression_x_efficiency_regime",
                "regime_value": "expanded|choppy_efficiency",
                "filter_rule": "bar_return >= 0",
                "feature": "bar_return",
                "operator": ">=",
                "threshold": 0.0,
                "filtered_test_same_result_rate": 0.95,
                "test_lift_vs_personality": 0.50,
                "test_lift_vs_regime": 0.30,
                "retained_test_count": 30,
            },
        ]
    ).to_csv(discovery_dir / "passed_personality_rules.csv", index=False)

    pd.DataFrame(
        [
            {
                "rule_name": "prior impulse_recoil -> current pullback_continuation",
                "rule_family": "fixed_prior_personality_sequence",
                "strict_status": "strict_train_and_oos_supported",
                "current_personality": "pullback_continuation",
                "prior_personality": "impulse_recoil",
                "prior2_personality": "",
            }
        ]
    ).to_csv(caveat_dir / "strict_validation_results.csv", index=False)
    return event_dir, discovery_dir, caveat_dir


def test_staged_lab_builds_mixed_regime_filter_book_and_blocks_before_exit(
    tmp_path: Path,
) -> None:
    event_dir, discovery_dir, caveat_dir = _write_inputs(tmp_path)

    result = run_staged_mixed_regime_caveat_exit_lab(
        input_event_dir=event_dir,
        input_personality_discovery_dir=discovery_dir,
        input_caveat_report_dir=caveat_dir,
        output_dir=tmp_path / "out",
        config=StagedMixedRegimeCaveatExitConfig(
            replay_months=("2026-03",),
            stop_models=("fixed_50bps",),
            target_r_multiples=(1.0,),
            min_train_events=8,
            min_symbol_train_events=1,
            min_train_symbols=3,
            min_train_months=2,
            min_total_trades=1,
            max_single_symbol_share=1.0,
            max_single_session_share=1.0,
            max_single_month_share=1.0,
            random_iterations=3,
        ),
    )

    filter_book = pd.read_csv(result.mixed_regime_filter_book_csv_path)
    caveated = pd.read_csv(result.caveated_signals_csv_path)
    trades = pd.read_csv(result.trades_csv_path)

    assert "above|high_range" in set(filter_book["regime_value"])
    assert caveated["caveat_rule_name"].tolist() == [
        "prior impulse_recoil -> current pullback_continuation"
    ]
    assert set(trades["symbol"]) == {"BBB", "CCC"}
    assert result.decision == "continue_research_staged_mixed_regime_caveat_exit"


def test_staged_lab_excludes_replay_symbols_without_prior_symbol_support(tmp_path: Path) -> None:
    event_dir, discovery_dir, caveat_dir = _write_inputs(tmp_path)
    rows = pd.read_csv(event_dir / "event_rows.csv")
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    _event_row(
                        symbol="ZZZ",
                        timestamp="2026-03-08T15:00:00Z",
                        session_date="2026-03-08",
                        bar_index=20,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    rows.to_csv(event_dir / "event_rows.csv", index=False)

    result = run_staged_mixed_regime_caveat_exit_lab(
        input_event_dir=event_dir,
        input_personality_discovery_dir=discovery_dir,
        input_caveat_report_dir=caveat_dir,
        output_dir=tmp_path / "out",
        config=StagedMixedRegimeCaveatExitConfig(
            replay_months=("2026-03",),
            stop_models=("fixed_50bps",),
            target_r_multiples=(1.0,),
            min_train_events=8,
            min_symbol_train_events=1,
            min_train_symbols=3,
            min_train_months=2,
            min_total_trades=1,
            max_single_symbol_share=1.0,
            max_single_session_share=1.0,
            max_single_month_share=1.0,
            random_iterations=3,
        ),
    )

    trades = pd.read_csv(result.trades_csv_path)

    assert "ZZZ" not in set(trades["symbol"])


def test_staged_lab_uses_unscored_warmup_months_for_prior_replay_gate(
    tmp_path: Path,
) -> None:
    event_dir, discovery_dir, caveat_dir = _write_inputs(tmp_path)

    result = run_staged_mixed_regime_caveat_exit_lab(
        input_event_dir=event_dir,
        input_personality_discovery_dir=discovery_dir,
        input_caveat_report_dir=caveat_dir,
        output_dir=tmp_path / "out",
        config=StagedMixedRegimeCaveatExitConfig(
            warmup_months=("2026-02",),
            replay_months=("2026-03",),
            stop_models=("fixed_50bps",),
            target_r_multiples=(1.0,),
            min_train_events=4,
            min_symbol_train_events=1,
            min_train_symbols=3,
            min_train_months=1,
            min_total_trades=1,
            max_single_symbol_share=1.0,
            max_single_session_share=1.0,
            max_single_month_share=1.0,
            enable_prior_replay_personality_acceptance=True,
            min_prior_replay_personality_trades=2,
            min_prior_replay_personality_total_net_r=0.0,
            random_iterations=3,
        ),
    )

    monthly = pd.read_csv(result.monthly_summary_csv_path)
    trades = pd.read_csv(result.trades_csv_path)
    acceptance = pd.read_csv(result.personality_acceptance_csv_path)
    march_prior = acceptance[
        (acceptance["month"].astype(str) == "2026-03")
        & (acceptance["acceptance_source"].astype(str) == "prior_replay")
    ]

    assert monthly["month"].tolist() == ["2026-03"]
    assert set(trades["month"]) == {"2026-03"}
    assert not march_prior.empty
    assert march_prior["train_trade_count"].max() > 0


def test_staged_train_caveat_book_selects_non_overlapping_train_supported_rules() -> None:
    rows = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "session_date": "2026-01-02",
                "month": "2026-01",
                "personality": "active_liquidation",
                "event_state": "failed_bounce_active_liquidation",
                "net_r": -1.0,
                "risk_bps": 75.0,
                "close_location_value": 0.20,
                "same_direction_other_symbol_count_15m": 0,
                "filter_rule": "bar_return <= 0",
                "stop_model": "fixed_75bps",
            },
            {
                "symbol": "BBB",
                "session_date": "2026-01-03",
                "month": "2026-01",
                "personality": "active_liquidation",
                "event_state": "failed_bounce_active_liquidation",
                "net_r": -0.8,
                "risk_bps": 80.0,
                "close_location_value": 0.25,
                "same_direction_other_symbol_count_15m": 0,
                "filter_rule": "bar_return <= 0",
                "stop_model": "fixed_75bps",
            },
            {
                "symbol": "CCC",
                "session_date": "2026-01-04",
                "month": "2026-01",
                "personality": "active_liquidation",
                "event_state": "failed_bounce_active_liquidation",
                "net_r": 1.2,
                "risk_bps": 220.0,
                "close_location_value": 0.75,
                "same_direction_other_symbol_count_15m": 4,
                "filter_rule": "bar_return <= 0",
                "stop_model": "fixed_75bps",
            },
        ]
    )

    caveats = _select_staged_train_caveat_book(
        rows,
        config=StagedMixedRegimeCaveatExitConfig(
            min_staged_caveat_train_trades=3,
            min_staged_caveat_flagged_trades=2,
            max_staged_caveat_rules_per_month=1,
        ),
        month="2026-02",
    )

    assert caveats["rule_family"].tolist() == ["train_selected_numeric"]
    assert caveats["feature"].tolist() == ["risk_bps"]
    assert caveats["operator"].tolist() == ["<="]


def test_personality_acceptance_book_rejects_negative_train_personality() -> None:
    selected = pd.DataFrame(
        [
            {"personality": "active_liquidation"},
            {"personality": "slow_repair"},
        ]
    )
    train_trades = pd.DataFrame(
        [
            {"personality": "active_liquidation", "net_r": 1.2},
            {"personality": "active_liquidation", "net_r": 0.8},
            {"personality": "slow_repair", "net_r": -0.7},
            {"personality": "slow_repair", "net_r": 0.2},
        ]
    )

    acceptance = _personality_acceptance_book(
        selected,
        train_trades,
        config=StagedMixedRegimeCaveatExitConfig(
            min_personality_train_trades=2,
            min_personality_train_total_net_r=0.0,
            min_personality_train_win_rate=0.0,
        ),
        month="2026-05",
    )

    accepted = dict(zip(acceptance["personality"], acceptance["accepted"], strict=True))
    reasons = dict(zip(acceptance["personality"], acceptance["rejection_reason"], strict=True))
    assert bool(accepted["active_liquidation"])
    assert not bool(accepted["slow_repair"])
    assert reasons["slow_repair"] == "train_total_net_r_below_min"


def test_prior_replay_personality_acceptance_falls_back_on_small_prior_sample() -> None:
    selected = pd.DataFrame(
        [
            {"personality": "active_liquidation"},
            {"personality": "open_down_pressure"},
            {"personality": "slow_repair"},
        ]
    )
    prior_replay_trades = pd.DataFrame(
        [
            {"month": "2026-01", "personality": "active_liquidation", "net_r": 1.0},
            {"month": "2026-01", "personality": "active_liquidation", "net_r": 0.7},
            {"month": "2026-01", "personality": "slow_repair", "net_r": -0.6},
            {"month": "2026-01", "personality": "slow_repair", "net_r": 0.1},
        ]
    )

    acceptance = _prior_replay_personality_acceptance_book(
        selected,
        prior_replay_trades,
        config=StagedMixedRegimeCaveatExitConfig(
            min_prior_replay_personality_trades=2,
            min_prior_replay_personality_total_net_r=0.25,
            min_prior_replay_personality_win_rate=0.0,
        ),
        month="2026-02",
    )

    accepted = dict(zip(acceptance["personality"], acceptance["accepted"], strict=True))
    reasons = dict(zip(acceptance["personality"], acceptance["rejection_reason"], strict=True))
    assert bool(accepted["active_liquidation"])
    assert bool(accepted["open_down_pressure"])
    assert not bool(accepted["slow_repair"])
    assert reasons["open_down_pressure"] == "insufficient_prior_replay_sample_fallback_train"
    assert reasons["slow_repair"] == "prior_replay_total_net_r_below_min"


def test_prior_replay_default_gate_requires_material_bad_sample_to_reject() -> None:
    selected = pd.DataFrame(
        [
            {"personality": "active_liquidation"},
            {"personality": "open_down_pressure"},
            {"personality": "slow_repair"},
        ]
    )
    prior_replay_trades = pd.DataFrame(
        [
            *[
                {
                    "month": "2025-07",
                    "personality": "active_liquidation",
                    "net_r": -0.05,
                }
                for _ in range(12)
            ],
            *[
                {
                    "month": "2025-07",
                    "personality": "open_down_pressure",
                    "net_r": -0.04,
                }
                for _ in range(15)
            ],
            *[
                {
                    "month": "2025-07",
                    "personality": "slow_repair",
                    "net_r": -0.08,
                }
                for _ in range(15)
            ],
        ]
    )

    acceptance = _prior_replay_personality_acceptance_book(
        selected,
        prior_replay_trades,
        config=StagedMixedRegimeCaveatExitConfig(),
        month="2025-08",
    )

    accepted = dict(zip(acceptance["personality"], acceptance["accepted"], strict=True))
    reasons = dict(zip(acceptance["personality"], acceptance["rejection_reason"], strict=True))
    assert bool(accepted["active_liquidation"])
    assert bool(accepted["open_down_pressure"])
    assert not bool(accepted["slow_repair"])
    assert reasons["active_liquidation"] == "insufficient_prior_replay_sample_fallback_train"
    assert reasons["open_down_pressure"] == ""
    assert reasons["slow_repair"] == "prior_replay_total_net_r_below_min"


def test_exit_sweep_cap_preserves_each_personality_top_candidate() -> None:
    exit_sweep = pd.DataFrame(
        [
            {
                "personality": "active_liquidation",
                "exit_selection_score": 1.00,
                "train_exit_total_net_r": 10.0,
            },
            {
                "personality": "active_liquidation",
                "exit_selection_score": 0.99,
                "train_exit_total_net_r": 9.0,
            },
            {
                "personality": "open_down_pressure",
                "exit_selection_score": 0.98,
                "train_exit_total_net_r": 8.0,
            },
            {
                "personality": "open_down_pressure",
                "exit_selection_score": 0.97,
                "train_exit_total_net_r": 7.0,
            },
            {
                "personality": "pullback_continuation",
                "exit_selection_score": 0.20,
                "train_exit_total_net_r": 6.0,
            },
        ]
    )

    capped = _cap_exit_sweep_with_personality_floor(
        exit_sweep,
        config=WalkForwardSelectedFilterExitConfig(
            max_exit_candidates_per_month=4,
            max_selected_per_personality_month=1,
        ),
    )

    assert len(capped) == 4
    assert "pullback_continuation" in set(capped["personality"])
    assert capped.groupby("personality").head(1)["personality"].tolist() == [
        "active_liquidation",
        "open_down_pressure",
        "pullback_continuation",
    ]


def test_staged_train_caveat_book_selects_crowded_30m_same_direction_rule() -> None:
    rows = pd.DataFrame(
        [
            {
                "personality": "active_liquidation",
                "net_r": 1.0,
                "same_direction_other_symbol_count_30m": 1,
            },
            {
                "personality": "active_liquidation",
                "net_r": 0.8,
                "same_direction_other_symbol_count_30m": 2,
            },
            {
                "personality": "active_liquidation",
                "net_r": 0.6,
                "same_direction_other_symbol_count_30m": 3,
            },
            {
                "personality": "active_liquidation",
                "net_r": -1.0,
                "same_direction_other_symbol_count_30m": 5,
            },
            {
                "personality": "active_liquidation",
                "net_r": -0.8,
                "same_direction_other_symbol_count_30m": 5,
            },
            {
                "personality": "active_liquidation",
                "net_r": -0.6,
                "same_direction_other_symbol_count_30m": 5,
            },
        ]
    )

    caveats = _select_staged_train_caveat_book(
        rows,
        config=StagedMixedRegimeCaveatExitConfig(
            min_staged_caveat_train_trades=6,
            min_staged_caveat_flagged_trades=2,
            max_staged_caveat_rules_per_month=1,
            staged_caveat_numeric_quantiles=(0.67,),
        ),
        month="2026-05",
    )

    assert caveats["rule_family"].tolist() == ["train_selected_numeric"]
    assert caveats["feature"].tolist() == ["same_direction_other_symbol_count_30m"]
    assert caveats["operator"].tolist() == [">="]


def test_personality_numeric_caveat_only_blocks_matching_personality() -> None:
    feature = "same_personality_other_symbol_count_15m"
    rows = pd.DataFrame(
        [
            {"personality": "slow_repair", "net_r": -1.0, feature: 0},
            {"personality": "slow_repair", "net_r": -0.8, feature: 1},
            {"personality": "slow_repair", "net_r": 1.0, feature: 3},
            {"personality": "slow_repair", "net_r": 0.6, feature: 4},
            {"personality": "active_liquidation", "net_r": 1.0, feature: 0},
            {"personality": "active_liquidation", "net_r": 0.9, feature: 1},
            {"personality": "active_liquidation", "net_r": 0.2, feature: 3},
            {"personality": "active_liquidation", "net_r": 0.1, feature: 4},
        ]
    )

    caveats = _select_staged_train_caveat_book(
        rows,
        config=StagedMixedRegimeCaveatExitConfig(
            min_staged_caveat_train_trades=8,
            min_staged_caveat_flagged_trades=2,
            max_staged_caveat_rules_per_month=1,
            staged_caveat_numeric_quantiles=(0.50,),
        ),
        month="2026-05",
    )

    assert caveats["rule_family"].tolist() == ["train_selected_personality_numeric"]
    assert caveats["current_personality"].tolist() == ["slow_repair"]
    assert caveats["feature"].tolist() == ["same_personality_other_symbol_count_15m"]
    assert caveats["operator"].tolist() == ["<="]

    passed, blocked = _apply_caveat_rules(rows, caveats)

    assert set(blocked["personality"]) == {"slow_repair"}
    assert len(blocked) == 2
    assert len(passed) == 6


def test_conditional_context_caveat_blocks_only_when_condition_and_threshold_match() -> None:
    rows = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "personality": "active_liquidation",
                "efficiency_regime": "choppy_efficiency",
                "prev_36_opening_mid_side_regime_current_share": 0.30,
            },
            {
                "symbol": "BBB",
                "personality": "active_liquidation",
                "efficiency_regime": "choppy_efficiency",
                "prev_36_opening_mid_side_regime_current_share": 0.70,
            },
            {
                "symbol": "CCC",
                "personality": "active_liquidation",
                "efficiency_regime": "directional_efficiency",
                "prev_36_opening_mid_side_regime_current_share": 0.20,
            },
        ]
    )
    caveat_book = pd.DataFrame(
        [
            {
                "caveat_rule_id": 0,
                "rule_name": (
                    "conditional_context: IF efficiency_regime == choppy_efficiency "
                    "THEN prev_36_opening_mid_side_regime_current_share <= 0.5"
                ),
                "rule_family": "train_selected_conditional_context_numeric",
                "strict_status": "strict_train_and_oos_supported",
                "current_personality": "",
                "prior_personality": "",
                "prior2_personality": "",
                "condition_feature": "efficiency_regime",
                "condition_operator": "==",
                "condition_value": "choppy_efficiency",
                "feature": "prev_36_opening_mid_side_regime_current_share",
                "operator": "<=",
                "selected_threshold": 0.5,
            }
        ]
    )

    passed, blocked = _apply_caveat_rules(rows, caveat_book)

    assert blocked["symbol"].tolist() == ["AAA"]
    assert set(passed["symbol"]) == {"BBB", "CCC"}


def _decision_trade_rows(month_counts: dict[str, int], net_rs: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    for month, count in month_counts.items():
        for item in range(count):
            rows.append(
                {
                    "symbol": f"SYM{index % 8:02d}",
                    "session_date": f"{month}-{1 + item:02d}",
                    "month": month,
                    "net_r": net_rs[index],
                }
            )
            index += 1
    return pd.DataFrame(rows)


def test_decision_allows_sparse_high_quality_when_only_trade_count_and_month_warn() -> None:
    trades = _decision_trade_rows(
        {
            "2026-01": 12,
            "2026-02": 1,
            "2026-03": 1,
            "2026-04": 3,
            "2026-05": 1,
            "2026-06": 2,
        },
        [1.0] * 16 + [-0.4] * 4,
    )
    monthly_summary = pd.DataFrame(
        [
            {"month": "2026-01", "total_net_r": 8.0},
            {"month": "2026-02", "total_net_r": 1.0},
            {"month": "2026-03", "total_net_r": 1.0},
            {"month": "2026-04", "total_net_r": 2.0},
            {"month": "2026-05", "total_net_r": -0.4},
            {"month": "2026-06", "total_net_r": 1.2},
        ]
    )

    decision, reasons = _decision(
        trades=trades,
        monthly_summary=monthly_summary,
        random_month_sum=-5.0,
        config=StagedMixedRegimeCaveatExitConfig(
            min_total_trades=30,
            max_single_month_share=0.50,
            allow_sparse_quality_decision=True,
            min_sparse_total_trades=15,
            min_sparse_positive_months=5,
            min_sparse_win_rate=0.75,
            min_sparse_mean_net_r=0.20,
            max_sparse_single_month_share=0.75,
        ),
    )

    assert decision == "continue_research_sparse_high_quality"
    assert reasons == ["low_trade_count_sparse_allowed", "month_concentrated_sparse_allowed"]


def test_decision_rejects_sparse_candidate_when_quality_floor_fails() -> None:
    trades = _decision_trade_rows(
        {"2026-01": 8, "2026-02": 4, "2026-03": 4},
        [1.0] * 8 + [-0.7] * 8,
    )
    monthly_summary = pd.DataFrame(
        [
            {"month": "2026-01", "total_net_r": 4.0},
            {"month": "2026-02", "total_net_r": -1.4},
            {"month": "2026-03", "total_net_r": 1.4},
        ]
    )

    decision, reasons = _decision(
        trades=trades,
        monthly_summary=monthly_summary,
        random_month_sum=-2.0,
        config=StagedMixedRegimeCaveatExitConfig(
            min_total_trades=30,
            allow_sparse_quality_decision=True,
            min_sparse_total_trades=15,
            min_sparse_positive_months=3,
            min_sparse_win_rate=0.75,
            min_sparse_mean_net_r=0.20,
        ),
    )

    assert decision == "reject_low_trade_count"
    assert "sparse_positive_months_below_min" in reasons
    assert "sparse_win_rate_below_min" in reasons


def test_entry_policy_diagnostics_compare_first_signal_to_highest_prior_score() -> None:
    signals = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "session_date": "2026-03-01",
                "timestamp": "2026-03-01T14:30:00Z",
                "month": "2026-03",
                "net_r": -1.0,
                "exit_selection_score": 0.1,
            },
            {
                "symbol": "AAA",
                "session_date": "2026-03-01",
                "timestamp": "2026-03-01T15:00:00Z",
                "month": "2026-03",
                "net_r": 1.5,
                "exit_selection_score": 0.9,
            },
        ]
    )

    diagnostics = _entry_policy_diagnostics(signals)

    totals = dict(zip(diagnostics["entry_policy"], diagnostics["total_net_r"], strict=True))
    assert totals["first_signal_per_symbol_session"] == -1.0
    assert totals["highest_prior_score_per_symbol_session"] == 1.5


def test_staged_mixed_regime_caveat_exit_cli_smoke(tmp_path: Path) -> None:
    event_dir, discovery_dir, caveat_dir = _write_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "walk-forward-staged-mixed-regime-caveat-exit",
            "--input-event-dir",
            str(event_dir),
            "--input-personality-discovery-dir",
            str(discovery_dir),
            "--input-caveat-report-dir",
            str(caveat_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--replay-months",
            "2026-03",
            "--stop-models",
            "fixed_50bps",
            "--target-r-multiples",
            "1",
            "--min-train-events",
            "8",
            "--min-train-symbols",
            "3",
            "--min-train-months",
            "2",
            "--min-total-trades",
            "1",
            "--max-single-month-share",
            "1",
            "--random-iterations",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "walk_forward_staged_mixed_regime_caveat_exit_v0" in result.output
    run_dirs = sorted(
        (tmp_path / "cli-out").glob("walk_forward_staged_mixed_regime_caveat_exit_v0_*")
    )
    filter_book = pd.read_csv(run_dirs[-1] / "mixed_regime_filter_book.csv")
    assert "above|high_range" in set(filter_book["regime_value"])
