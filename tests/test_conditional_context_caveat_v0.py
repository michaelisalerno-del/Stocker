from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.conditional_context_caveat_v0 import (
    ConditionalContextCaveatConfig,
    run_conditional_context_caveat_lab,
)


def _context_row(
    *,
    split: str,
    symbol: str,
    session_date: str,
    efficiency_regime: str,
    opening_mid_stability: float,
    net_r: float,
) -> dict[str, object]:
    month = session_date[:7]
    return {
        "split": split,
        "symbol": symbol,
        "timestamp": f"{session_date}T15:00:00Z",
        "session_date": session_date,
        "month": month,
        "personality": "active_liquidation",
        "event_state": "failed_bounce_active_liquidation",
        "net_r": net_r,
        "efficiency_regime": efficiency_regime,
        "prev_36_opening_mid_side_regime_current_share": opening_mid_stability,
    }


def _write_context_report(tmp_path: Path) -> Path:
    report_dir = tmp_path / "state_lifecycle_context_lab_v0" / "run"
    report_dir.mkdir(parents=True)
    rows = [
        _context_row(
            split="train",
            symbol="AAA",
            session_date="2026-01-02",
            efficiency_regime="choppy_efficiency",
            opening_mid_stability=0.20,
            net_r=-1.0,
        ),
        _context_row(
            split="train",
            symbol="BBB",
            session_date="2026-02-03",
            efficiency_regime="choppy_efficiency",
            opening_mid_stability=0.30,
            net_r=-0.8,
        ),
        _context_row(
            split="train",
            symbol="CCC",
            session_date="2026-03-04",
            efficiency_regime="choppy_efficiency",
            opening_mid_stability=0.70,
            net_r=1.0,
        ),
        _context_row(
            split="train",
            symbol="DDD",
            session_date="2026-04-05",
            efficiency_regime="choppy_efficiency",
            opening_mid_stability=0.80,
            net_r=0.8,
        ),
        _context_row(
            split="train",
            symbol="EEE",
            session_date="2026-04-06",
            efficiency_regime="directional_efficiency",
            opening_mid_stability=2.00,
            net_r=0.5,
        ),
        _context_row(
            split="test",
            symbol="AAA",
            session_date="2026-05-02",
            efficiency_regime="choppy_efficiency",
            opening_mid_stability=0.20,
            net_r=-1.2,
        ),
        _context_row(
            split="test",
            symbol="BBB",
            session_date="2026-05-03",
            efficiency_regime="choppy_efficiency",
            opening_mid_stability=0.30,
            net_r=-0.7,
        ),
        _context_row(
            split="test",
            symbol="CCC",
            session_date="2026-06-04",
            efficiency_regime="choppy_efficiency",
            opening_mid_stability=0.70,
            net_r=1.0,
        ),
        _context_row(
            split="test",
            symbol="DDD",
            session_date="2026-06-05",
            efficiency_regime="choppy_efficiency",
            opening_mid_stability=0.80,
            net_r=0.9,
        ),
        _context_row(
            split="test",
            symbol="EEE",
            session_date="2026-06-06",
            efficiency_regime="directional_efficiency",
            opening_mid_stability=0.10,
            net_r=0.5,
        ),
    ]
    pd.DataFrame(rows).to_csv(report_dir / "trade_context_features.csv", index=False)
    return report_dir


def test_conditional_context_caveat_selects_train_threshold_and_evaluates_oos(
    tmp_path: Path,
) -> None:
    report_dir = _write_context_report(tmp_path)

    result = run_conditional_context_caveat_lab(
        input_context_report_dir=report_dir,
        output_dir=tmp_path / "out",
        config=ConditionalContextCaveatConfig(
            condition_features=("efficiency_regime",),
            numeric_features=("prev_36_opening_mid_side_regime_current_share",),
            numeric_quantiles=(0.50,),
            random_iterations=50,
            min_train_condition_count=2,
            min_train_flagged_count=1,
            min_oos_flagged_count=1,
            max_single_symbol_share=1.0,
            max_single_session_share=1.0,
        ),
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    strict = pd.read_csv(result.strict_validation_results_csv_path)
    flags = pd.read_csv(result.trade_conditional_caveat_flags_csv_path)

    assert result.decision == "continue_research_strict_conditional_caveat_supported"
    assert summary["research_only"] is True
    assert summary["edge_claimed"] is False
    assert strict["strict_status"].tolist() == ["strict_train_and_oos_supported"]
    row = strict.iloc[0]
    assert row["condition_feature"] == "efficiency_regime"
    assert row["condition_value"] == "choppy_efficiency"
    assert row["feature"] == "prev_36_opening_mid_side_regime_current_share"
    assert row["operator"] == "<="
    assert row["selected_threshold"] == 0.5
    assert row["train_flagged_count"] == 2
    assert row["test_flagged_count"] == 2
    assert row["test_kept_lift_vs_base_r"] > 0

    flag_columns = [column for column in flags.columns if column.startswith("flag_")]
    assert len(flag_columns) == 1
    flagged = flags[flags[flag_columns[0]].astype(bool)]
    assert set(flagged["symbol"]) == {"AAA", "BBB"}
    assert set(flagged["efficiency_regime"]) == {"choppy_efficiency"}


def test_conditional_context_caveat_cli_smoke(tmp_path: Path) -> None:
    report_dir = _write_context_report(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "conditional-context-caveat",
            "--input-context-report-dir",
            str(report_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--condition-features",
            "efficiency_regime",
            "--numeric-features",
            "prev_36_opening_mid_side_regime_current_share",
            "--numeric-quantiles",
            "0.50",
            "--random-iterations",
            "20",
            "--min-train-condition-count",
            "2",
            "--min-train-flagged-count",
            "1",
            "--min-oos-flagged-count",
            "1",
            "--max-single-symbol-share",
            "1",
            "--max-single-session-share",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "conditional_context_caveat_v0" in result.output
    run_dirs = sorted((tmp_path / "cli-out").glob("conditional_context_caveat_v0_*"))
    assert run_dirs
    decision = json.loads((run_dirs[-1] / "decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "continue_research_strict_conditional_caveat_supported"
