from __future__ import annotations

from pathlib import Path

import pytest

from stocker_research.sequential_loop_competitor_veto import (
    ProspectiveCompetitorLedger,
)


def _forecast() -> dict[str, object]:
    return {
        "run_id": "holdout-run",
        "git_sha": "abc",
        "contract_hash": "def",
        "data_snapshot_hash": "ghi",
        "model_version": "sequential_loop_competitor_veto_v1.0.0",
        "forecast_id": "forecast-1",
        "event_lineage_id": "event-1",
        "opportunity_id": "opportunity-1",
        "anchor_id": "anchor-1",
        "stock": "AAL",
        "session": "2099-01-05",
        "decision_timestamp": "2099-01-05T14:35:00Z",
        "checkpoint_timestamp": "2099-01-05T14:35:00Z",
        "checkpoint_type": "anchor_freeze",
        "bars_since_anchor": 0,
        "bars_remaining": 24,
        "current_state": 4,
        "state_history": "1,2,4",
        "clock_phase": "open",
        "compatible_loop_set": ["cycle_04", "cycle_06"],
        "loop_posterior": {"cycle_04": 0.2, "cycle_06": 0.1, "unknown": 0.7},
        "good_loop_mass": 0.2,
        "bad_loop_mass": 0.1,
        "unknown_loop_mass": 0.7,
        "entropy": 0.73,
        "competitor_eliminations": [],
        "decision_state": "unresolved",
        "reason_codes": ["unknown_mass_excessive"],
        "freeze_timestamp": "2099-01-05T14:35:00Z",
        "feature_availability_timestamps": ["2099-01-05T14:35:00Z"],
        "training_cutoff": "2099-01-04T21:00:00Z",
    }


def test_prospective_forecast_is_create_only_and_lineage_persists(tmp_path: Path) -> None:
    ledger = ProspectiveCompetitorLedger(tmp_path, opened_periods={2023, 2024, 2025, 2026})
    path = ledger.append_forecast(_forecast(), holdout=True)

    assert path.exists()
    with pytest.raises(FileExistsError):
        ledger.append_forecast(_forecast(), holdout=True)


def test_holdout_mode_rejects_opened_historical_sessions(tmp_path: Path) -> None:
    ledger = ProspectiveCompetitorLedger(tmp_path, opened_periods={2023, 2024, 2025, 2026})
    record = _forecast()
    record["session"] = "2025-01-05"

    with pytest.raises(ValueError, match="opened period"):
        ledger.append_forecast(record, holdout=True)


def test_outcome_appends_separately_and_points_to_exact_forecast(tmp_path: Path) -> None:
    ledger = ProspectiveCompetitorLedger(tmp_path, opened_periods={2023, 2024, 2025, 2026})
    ledger.append_forecast(_forecast(), holdout=True)

    path = ledger.append_outcome(
        {
            "outcome_id": "outcome-1",
            "forecast_id": "forecast-1",
            "event_lineage_id": "event-1",
            "settlement_timestamp": "2099-01-05T16:40:00Z",
            "constant_terminal_net_bps": 12.0,
        }
    )

    assert path.exists()


def test_hindsight_labels_are_forbidden_from_prospective_features(tmp_path: Path) -> None:
    ledger = ProspectiveCompetitorLedger(tmp_path, opened_periods=set())
    record = _forecast()
    record["hindsight_episode_label"] = "winner"

    with pytest.raises(ValueError, match="hindsight"):
        ledger.append_forecast(record, holdout=True)
