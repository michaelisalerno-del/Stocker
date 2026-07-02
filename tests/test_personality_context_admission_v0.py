from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.personality_context_admission_v0 import (
    PersonalityContextAdmissionConfig,
    run_personality_context_admission_lab,
)
from stocker_research.personality_context_rule_discovery_v0 import (
    PersonalityContextRuleDiscoveryConfig,
    ReportPair,
    run_personality_context_rule_discovery_lab,
)


def _trade_row(
    *,
    symbol: str,
    timestamp: str,
    personality: str,
    volume_x_vwap_regime: str,
    net_r: float,
    selected_filter_rank: int = 1,
    monthly_candidate_rank: int = 1,
) -> dict[str, object]:
    session_date = timestamp[:10]
    month = timestamp[:7]
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "session_date": session_date,
        "month": month,
        "personality": personality,
        "event_state": "slow_snapback_after_dip"
        if personality == "slow_repair"
        else "failed_bounce_active_liquidation",
        "volume_x_vwap_regime": volume_x_vwap_regime,
        "return_zscore": 0.25,
        "net_r": net_r,
        "stop_model": "fixed_75bps",
        "target_r": 1.5,
        "monthly_candidate_rank": monthly_candidate_rank,
        "selected_filter_rank": selected_filter_rank,
    }


def _write_report_pair(tmp_path: Path) -> tuple[Path, Path]:
    baseline_dir = tmp_path / "baseline_report"
    candidate_dir = tmp_path / "candidate_report"
    baseline_dir.mkdir()
    candidate_dir.mkdir()

    baseline_rows = [
        _trade_row(
            symbol="BASE",
            timestamp="2026-01-02T15:00:00Z",
            personality="active_liquidation",
            volume_x_vwap_regime="low_relative_volume|below",
            net_r=1.0,
        ),
    ]
    blocked_rows = [
        _trade_row(
            symbol="AAA",
            timestamp="2026-01-03T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="low_relative_volume|below",
            net_r=1.0,
        ),
        _trade_row(
            symbol="BBB",
            timestamp="2026-02-03T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="low_relative_volume|below",
            net_r=0.8,
        ),
        _trade_row(
            symbol="CCC",
            timestamp="2026-03-03T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="normal_relative_volume|near",
            net_r=0.9,
        ),
        _trade_row(
            symbol="DDD",
            timestamp="2026-04-03T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="normal_relative_volume|near",
            net_r=0.8,
        ),
        _trade_row(
            symbol="EEE",
            timestamp="2026-01-04T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="high_relative_volume|below",
            net_r=-0.5,
        ),
        _trade_row(
            symbol="FFF",
            timestamp="2026-02-04T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="high_relative_volume|below",
            net_r=-0.4,
        ),
        _trade_row(
            symbol="GGG",
            timestamp="2026-03-04T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="unknown|above",
            net_r=-0.2,
        ),
        _trade_row(
            symbol="HHH",
            timestamp="2026-04-04T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="unknown|above",
            net_r=-0.3,
        ),
        _trade_row(
            symbol="III",
            timestamp="2026-05-03T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="low_relative_volume|below",
            net_r=0.7,
        ),
        _trade_row(
            symbol="JJJ",
            timestamp="2026-06-03T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="low_relative_volume|below",
            net_r=0.6,
        ),
        _trade_row(
            symbol="KKK",
            timestamp="2026-05-04T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="normal_relative_volume|near",
            net_r=-0.4,
        ),
        _trade_row(
            symbol="LLL",
            timestamp="2026-06-04T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="normal_relative_volume|near",
            net_r=-0.5,
        ),
        _trade_row(
            symbol="MMM",
            timestamp="2026-05-05T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="high_relative_volume|below",
            net_r=1.0,
        ),
        _trade_row(
            symbol="NNN",
            timestamp="2026-06-05T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="high_relative_volume|below",
            net_r=0.8,
        ),
        _trade_row(
            symbol="OOO",
            timestamp="2026-05-06T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="unknown|above",
            net_r=-0.2,
        ),
        _trade_row(
            symbol="PPP",
            timestamp="2026-06-06T15:00:00Z",
            personality="slow_repair",
            volume_x_vwap_regime="unknown|above",
            net_r=-0.1,
        ),
        _trade_row(
            symbol="QQQ",
            timestamp="2026-05-07T15:00:00Z",
            personality="open_down_pressure",
            volume_x_vwap_regime="low_relative_volume|below",
            net_r=1.2,
            selected_filter_rank=2,
            monthly_candidate_rank=2,
        ),
    ]

    pd.DataFrame(baseline_rows).to_csv(baseline_dir / "trades.csv", index=False)
    pd.DataFrame([*baseline_rows, *blocked_rows]).to_csv(
        candidate_dir / "trades.csv",
        index=False,
    )
    return baseline_dir, candidate_dir


def test_personality_context_admission_learns_train_context_and_scores_oos(
    tmp_path: Path,
) -> None:
    baseline_dir, candidate_dir = _write_report_pair(tmp_path)

    result = run_personality_context_admission_lab(
        input_baseline_report_dir=baseline_dir,
        input_candidate_report_dir=candidate_dir,
        output_dir=tmp_path / "out",
        config=PersonalityContextAdmissionConfig(
            train_months=("2026-01", "2026-02", "2026-03", "2026-04"),
            test_months=("2026-05", "2026-06"),
            target_personalities=("slow_repair",),
            context_features=("volume_x_vwap_regime",),
            random_iterations=100,
            min_train_admitted_count=2,
            min_oos_admitted_count=2,
            max_single_symbol_share=1.0,
            max_single_session_share=1.0,
        ),
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    rules = pd.read_csv(result.admission_rule_results_csv_path)
    selected = pd.read_csv(result.selected_admissions_csv_path)
    blocked = pd.read_csv(result.blocked_candidate_trades_csv_path)

    assert result.decision == "continue_research_strict_personality_context_admission_supported"
    assert summary["research_only"] is True
    assert summary["edge_claimed"] is False
    assert len(blocked) == 17

    statuses = dict(zip(rules["context_value"], rules["strict_status"], strict=True))
    assert statuses["low_relative_volume|below"] == "strict_train_and_oos_supported"
    assert statuses["normal_relative_volume|near"] == "train_only_not_oos_supported"
    assert statuses["high_relative_volume|below"] == "oos_only_not_train_supported"
    assert statuses["unknown|above"] == "not_supported"

    low_volume = selected[
        selected["context_value"].astype(str).eq("low_relative_volume|below")
    ].iloc[0]
    assert bool(low_volume["train_selected"])
    assert low_volume["personality"] == "slow_repair"
    assert low_volume["context_feature"] == "volume_x_vwap_regime"
    assert low_volume["train_admitted_count"] == 2
    assert low_volume["test_admitted_count"] == 2
    assert low_volume["test_admitted_total_net_r"] > 0.0
    assert low_volume["test_excess_vs_random_median_r"] > 0.0
    assert low_volume["test_admitted_symbol_count"] == 2


def test_personality_context_admission_cli_smoke(tmp_path: Path) -> None:
    baseline_dir, candidate_dir = _write_report_pair(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "personality-context-admission",
            "--input-baseline-report-dir",
            str(baseline_dir),
            "--input-candidate-report-dir",
            str(candidate_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--train-months",
            "2026-01,2026-02,2026-03,2026-04",
            "--test-months",
            "2026-05,2026-06",
            "--target-personalities",
            "slow_repair",
            "--context-features",
            "volume_x_vwap_regime",
            "--random-iterations",
            "50",
            "--min-train-admitted-count",
            "2",
            "--min-oos-admitted-count",
            "2",
            "--max-single-symbol-share",
            "1",
            "--max-single-session-share",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "personality_context_admission_v0" in result.output
    run_dirs = sorted((tmp_path / "cli-out").glob("personality_context_admission_v0_*"))
    assert run_dirs
    decision = json.loads((run_dirs[-1] / "decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == (
        "continue_research_strict_personality_context_admission_supported"
    )


def _discovery_trade_row(
    *,
    symbol: str,
    timestamp: str,
    personality: str,
    net_r: float,
    prior_3_bar_return: float,
    prev_event_personality: str,
    distance_from_recent_high_pct: float,
    volume_x_vwap_regime: str,
    selected_filter_rank: int,
) -> dict[str, object]:
    row = _trade_row(
        symbol=symbol,
        timestamp=timestamp,
        personality=personality,
        volume_x_vwap_regime=volume_x_vwap_regime,
        net_r=net_r,
        selected_filter_rank=selected_filter_rank,
        monthly_candidate_rank=selected_filter_rank,
    )
    row.update(
        {
            "prior_3_bar_return": prior_3_bar_return,
            "prev_event_personality": prev_event_personality,
            "distance_from_recent_high_pct": distance_from_recent_high_pct,
        }
    )
    return row


def _write_discovery_pair(
    tmp_path: Path,
    label: str,
    month: str,
    *,
    suffix: int,
) -> tuple[Path, Path]:
    baseline_dir = tmp_path / f"{label}_baseline"
    candidate_dir = tmp_path / f"{label}_candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline_rows = [
        _discovery_trade_row(
            symbol=f"BASE{suffix}",
            timestamp=f"{month}-02T15:00:00Z",
            personality="active_liquidation",
            net_r=0.25,
            prior_3_bar_return=0.0,
            prev_event_personality="dead_chop_noise",
            distance_from_recent_high_pct=-0.04,
            volume_x_vwap_regime="low_relative_volume|below",
            selected_filter_rank=1,
        )
    ]
    candidate_only_rows = [
        _discovery_trade_row(
            symbol=f"STR{suffix}",
            timestamp=f"{month}-03T15:00:00Z",
            personality="slow_repair",
            net_r=1.0,
            prior_3_bar_return=0.010,
            prev_event_personality="active_liquidation",
            distance_from_recent_high_pct=-0.040,
            volume_x_vwap_regime="low_relative_volume|below",
            selected_filter_rank=2,
        ),
        _discovery_trade_row(
            symbol=f"REC{suffix}",
            timestamp=f"{month}-04T15:00:00Z",
            personality="slow_repair",
            net_r=0.8,
            prior_3_bar_return=0.002,
            prev_event_personality="reclaim_reversal",
            distance_from_recent_high_pct=-0.010,
            volume_x_vwap_regime="low_relative_volume|below",
            selected_filter_rank=3,
        ),
        _discovery_trade_row(
            symbol=f"LEAK{suffix}",
            timestamp=f"{month}-05T15:00:00Z",
            personality="slow_repair",
            net_r=-1.1,
            prior_3_bar_return=0.002,
            prev_event_personality="reclaim_reversal",
            distance_from_recent_high_pct=-0.010,
            volume_x_vwap_regime="high_relative_volume|above",
            selected_filter_rank=4,
        ),
        _discovery_trade_row(
            symbol=f"BAD{suffix}",
            timestamp=f"{month}-06T15:00:00Z",
            personality="slow_repair",
            net_r=-0.7,
            prior_3_bar_return=-0.002,
            prev_event_personality="dead_chop_noise",
            distance_from_recent_high_pct=-0.030,
            volume_x_vwap_regime="normal_relative_volume|below",
            selected_filter_rank=5,
        ),
    ]
    pd.DataFrame(baseline_rows).to_csv(baseline_dir / "trades.csv", index=False)
    pd.DataFrame([*baseline_rows, *candidate_only_rows]).to_csv(
        candidate_dir / "trades.csv",
        index=False,
    )
    return baseline_dir, candidate_dir


def test_personality_context_rule_discovery_finds_or_reentry_rule(
    tmp_path: Path,
) -> None:
    first_baseline, first_candidate = _write_discovery_pair(
        tmp_path,
        "w1",
        "2026-01",
        suffix=1,
    )
    second_baseline, second_candidate = _write_discovery_pair(
        tmp_path,
        "w2",
        "2026-02",
        suffix=2,
    )

    result = run_personality_context_rule_discovery_lab(
        report_pairs=(
            ReportPair("w1", first_baseline, first_candidate),
            ReportPair("w2", second_baseline, second_candidate),
        ),
        output_dir=tmp_path / "discovery-out",
        config=PersonalityContextRuleDiscoveryConfig(
            target_personalities=("slow_repair",),
            categorical_features=("prev_event_personality", "volume_x_vwap_regime"),
            numeric_features=("prior_3_bar_return", "distance_from_recent_high_pct"),
            quantiles=(0.5,),
            min_rule_trades=2,
            min_rule_windows=2,
            min_positive_windows=2,
            max_negative_windows=0,
            max_single_window_share=0.75,
            random_iterations=50,
        ),
    )

    selected = pd.read_csv(result.selected_rules_csv_path)
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

    assert result.decision == "continue_research_blank_slate_context_rule_supported"
    assert summary["research_only"] is True
    assert summary["edge_claimed"] is False
    assert not selected.empty

    best = selected.iloc[0]
    assert best["support_status"] == "supported_candidate_only_reentry"
    assert best["rule_kind"] == "or2"
    assert best["candidate_only_admitted_count"] == 4
    assert best["candidate_only_admitted_total_net_r"] == 3.6
    assert " OR " in best["rule_expression"]
    assert "high_relative_volume|above" in best["rule_expression"]


def test_personality_context_rule_discovery_cli_smoke(tmp_path: Path) -> None:
    first_baseline, first_candidate = _write_discovery_pair(
        tmp_path,
        "w1_cli",
        "2026-01",
        suffix=1,
    )
    second_baseline, second_candidate = _write_discovery_pair(
        tmp_path,
        "w2_cli",
        "2026-02",
        suffix=2,
    )

    result = CliRunner().invoke(
        app,
        [
            "research",
            "personality-context-rule-discovery",
            "--report-pair",
            f"w1={first_baseline},{first_candidate}",
            "--report-pair",
            f"w2={second_baseline},{second_candidate}",
            "--output-dir",
            str(tmp_path / "cli-discovery-out"),
            "--target-personalities",
            "slow_repair",
            "--categorical-features",
            "prev_event_personality,volume_x_vwap_regime",
            "--numeric-features",
            "prior_3_bar_return,distance_from_recent_high_pct",
            "--quantiles",
            "0.5",
            "--min-rule-trades",
            "2",
            "--min-rule-windows",
            "2",
            "--min-positive-windows",
            "2",
            "--max-negative-windows",
            "0",
            "--max-single-window-share",
            "0.75",
            "--random-iterations",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "personality_context_rule_discovery_v0" in result.output
    run_dirs = sorted(
        (tmp_path / "cli-discovery-out").glob("personality_context_rule_discovery_v0_*")
    )
    assert run_dirs
    decision = json.loads((run_dirs[-1] / "decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "continue_research_blank_slate_context_rule_supported"
