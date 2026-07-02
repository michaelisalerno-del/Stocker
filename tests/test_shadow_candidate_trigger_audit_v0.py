from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.shadow_candidate_trigger_audit_v0 import (
    ShadowCandidateTriggerConfig,
    add_shadow_candidate_trigger_features,
    run_shadow_candidate_trigger_audit,
)


def _context_row(
    *,
    idx: int,
    split: str = "test",
    month: str = "2026-05",
    net_r: float,
    anti_stale_feature: float = 1.0,
    weak: bool = True,
) -> dict[str, object]:
    timestamp = pd.Timestamp("2026-05-01T14:30:00Z") + pd.Timedelta(minutes=idx * 5)
    return {
        "symbol": f"S{idx % 3}",
        "timestamp": timestamp.isoformat(),
        "session_date": timestamp.strftime("%Y-%m-%d"),
        "month": month,
        "split": split,
        "personality": "active_liquidation",
        "net_r": net_r,
        "prev_12_time_regime_current_share": anti_stale_feature,
        "efficiency_regime": "choppy_efficiency" if weak else "directional_efficiency",
        "vwap_side_regime": "below" if weak else "above",
        "opening_mid_side_regime": "below" if weak else "above",
        "time_regime": "morning" if weak else "late_day",
        "range_regime": "low_range" if weak else "high_range",
        "prev_24_vwap_x_efficiency_regime_current_share": 0.0 if weak else 0.50,
        "prev_36_opening_mid_side_regime_current_share": 0.0 if weak else 0.75,
        "vwap_cross_count_12": 3 if weak else 0,
    }


def test_shadow_candidate_trigger_features_use_only_prior_candidates() -> None:
    rows = pd.DataFrame(
        [
            _context_row(idx=idx, net_r=-0.20, anti_stale_feature=1.0, weak=True)
            for idx in range(9)
        ]
        + [_context_row(idx=9, net_r=1.0, anti_stale_feature=1.0, weak=False)]
    )

    features = add_shadow_candidate_trigger_features(
        rows,
        config=ShadowCandidateTriggerConfig(
            shadow_window=5,
            min_prior_candidates=3,
            weak_context_share_threshold=0.75,
            shadow_net_r_threshold=-0.50,
        ),
    )

    assert features["shadow_candidate_index"].tolist() == list(range(10))
    assert features.loc[0, "prior_5_candidate_count"] == 0
    assert features.loc[0, "weak_cluster_shadow_deterioration_trigger"] is False

    first_trigger = features.loc[3]
    assert first_trigger["prior_5_candidate_count"] == 3
    assert first_trigger["prior_5_weak_context_share"] == 1.0
    assert first_trigger["prior_5_shadow_candidate_net_r"] == pytest.approx(-0.6)
    assert first_trigger["weak_cluster_shadow_deterioration_trigger"] is True

    last = features.loc[9]
    assert last["prior_5_candidate_count"] == 5
    assert last["prior_5_weak_context_share"] == 1.0
    assert last["planned_exit_shadow_net_r"] == 1.0
    assert last["candidate_block_reason"] == "none"
    assert last["would_be_taken_without_safety"] is True


def _write_context_report(tmp_path: Path) -> Path:
    report_dir = tmp_path / "context"
    report_dir.mkdir()
    rows: list[dict[str, object]] = []
    for idx in range(6):
        rows.append(
            _context_row(
                idx=idx,
                split="train",
                month="2026-01",
                net_r=0.5,
                anti_stale_feature=0.10,
                weak=False,
            )
        )
    for idx in range(6, 12):
        rows.append(
            _context_row(
                idx=idx,
                split="train",
                month="2026-02",
                net_r=-0.5,
                anti_stale_feature=0.90,
                weak=True,
            )
        )
    for idx in range(12, 20):
        rows.append(
            _context_row(
                idx=idx,
                split="test",
                month="2026-03",
                net_r=-0.4,
                anti_stale_feature=0.90,
                weak=True,
            )
        )
    for idx in range(20, 24):
        rows.append(
            _context_row(
                idx=idx,
                split="test",
                month="2026-03",
                net_r=0.7,
                anti_stale_feature=0.10,
                weak=True,
            )
        )
    pd.DataFrame(rows).to_csv(report_dir / "trade_context_features.csv", index=False)
    return report_dir


def test_shadow_candidate_trigger_audit_writes_policy_outputs(tmp_path: Path) -> None:
    report_dir = _write_context_report(tmp_path)

    result = run_shadow_candidate_trigger_audit(
        input_context_report_dir=report_dir,
        output_dir=tmp_path / "out",
        config=ShadowCandidateTriggerConfig(
            shadow_window=5,
            min_prior_candidates=3,
            weak_context_share_threshold=0.75,
            shadow_net_r_threshold=-0.50,
            anti_stale_feature_bases=("time_regime",),
            anti_stale_windows=(12,),
            anti_stale_quantiles=(0.50,),
            min_train_count=6,
            min_rule_keep_count=3,
            random_iterations=20,
        ),
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    policy_summary = pd.read_csv(result.policy_summary_csv_path)
    features = pd.read_csv(result.shadow_candidate_features_csv_path)

    assert result.decision == "continue_research_shadow_trigger_audit"
    assert summary["research_only"] is True
    assert summary["edge_claimed"] is False
    assert summary["order_placement"] == "disabled"
    assert result.shadow_candidate_features_csv_path.exists()
    assert {"planned_exit_shadow_net_r", "prior_5_shadow_candidate_net_r"}.issubset(
        set(features.columns)
    )

    test_rows = policy_summary[
        (policy_summary["split"] == "test")
        & (policy_summary["policy"] == "weak_cluster_shadow_deterioration")
    ]
    assert len(test_rows) == 1
    assert test_rows.iloc[0]["kept_net_r"] > test_rows.iloc[0]["base_net_r"]
    assert test_rows.iloc[0]["skipped_count"] > 0


def test_shadow_candidate_trigger_audit_cli_smoke(tmp_path: Path) -> None:
    report_dir = _write_context_report(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "shadow-candidate-trigger-audit",
            "--input-context-report-dir",
            str(report_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--shadow-window",
            "5",
            "--min-prior-candidates",
            "3",
            "--weak-context-share-threshold",
            "0.75",
            "--shadow-net-r-threshold",
            "-0.50",
            "--anti-stale-feature-bases",
            "time_regime",
            "--anti-stale-windows",
            "12",
            "--anti-stale-quantiles",
            "0.50",
            "--min-train-count",
            "6",
            "--min-rule-keep-count",
            "3",
            "--random-iterations",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "shadow_candidate_trigger_audit_v0" in result.output
    run_dirs = sorted((tmp_path / "cli-out").glob("shadow_candidate_trigger_audit_v0_*"))
    assert run_dirs
    decision = json.loads((run_dirs[-1] / "decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "continue_research_shadow_trigger_audit"
