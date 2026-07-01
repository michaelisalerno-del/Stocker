from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.sparse_exhaustion_extension_v0 import (
    run_sparse_exhaustion_extension_lab,
)


def _write_candidate_scan(tmp_path: Path) -> Path:
    report_dir = tmp_path / "candidate_personality_scan_v0" / "run"
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "timestamp": "2026-01-02T15:30:00Z",
                "session_date": "2026-01-02",
                "bar_index_in_session": 12,
                "candidate_personality": "exhaustion_extension",
                "expected_direction": -1,
                "candidate_role": "mean_reversion_or_no_chase",
                "trigger_reason": "upside extension stretched from vwap with rejection",
                "forward_12_bar_return": -0.01,
                "forward_12_bar_mfe": 0.002,
                "forward_12_bar_mae": -0.012,
                "split": "test",
            },
            {
                "symbol": "BBB",
                "timestamp": "2026-01-02T15:35:00Z",
                "session_date": "2026-01-02",
                "bar_index_in_session": 13,
                "candidate_personality": "trend_acceptance_drive",
                "expected_direction": 1,
                "candidate_role": "directional_continuation",
                "trigger_reason": "not exhaustion",
                "forward_12_bar_return": 0.01,
                "forward_12_bar_mfe": 0.012,
                "forward_12_bar_mae": -0.002,
                "split": "test",
            },
        ]
    ).to_csv(report_dir / "candidate_event_rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_personality": "exhaustion_extension",
                "horizon": 12,
                "test_count": 1,
                "test_same_result_rate": 1.0,
                "test_median_aligned_return": 0.01,
                "beats_random_p95": True,
                "enough_sample": True,
            }
        ]
    ).to_csv(report_dir / "passed_candidate_personality_horizons.csv", index=False)
    return report_dir


def test_sparse_exhaustion_extension_filters_candidate_rows(tmp_path: Path) -> None:
    input_dir = _write_candidate_scan(tmp_path)

    result = run_sparse_exhaustion_extension_lab(
        input_candidate_report_dir=input_dir,
        output_dir=tmp_path / "out",
    )

    events = pd.read_csv(result.exhaustion_event_rows_csv_path)
    selected = pd.read_csv(result.selected_horizons_csv_path)
    assert result.decision == "continue_research_sparse_exhaustion_extension"
    assert events["event_state"].tolist() == ["exhaustion_extension"]
    assert events["event_family"].tolist() == ["extension_exhaustion"]
    assert selected["horizon"].tolist() == [12]


def test_sparse_exhaustion_extension_cli_smoke(tmp_path: Path) -> None:
    input_dir = _write_candidate_scan(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "sparse-exhaustion-extension",
            "--input-candidate-report-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "sparse_exhaustion_extension_v0" in result.output
