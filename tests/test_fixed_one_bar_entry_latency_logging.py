from __future__ import annotations

import json
from pathlib import Path

import pytest

from stocker_research.fixed_one_bar_entry_latency.immutable_ledger import (
    ProspectiveLatencyLedger,
)


def _opportunity() -> dict[str, object]:
    return {
        "run_id": "run-new",
        "git_sha": "abc123",
        "contract_hash": "contract-hash",
        "data_snapshot_hash": "new-snapshot",
        "source_run_id": "source-run",
        "source_artifact_hash": "source-hash",
        "source_opportunity_hash": "opportunity-hash",
        "opportunity_id": "opp-1",
        "anchor_id": "anchor-1",
        "event_lineage_id": "event-1",
        "symbol": "TEST",
        "session": "2027-01-04",
        "loop_id": "cycle_04",
        "orientation": "state_4",
        "frozen_direction": 1,
        "anchor_timestamp": "2027-01-04T14:30:00+00:00",
        "t0_entry_timestamp": "2027-01-04T14:45:00+00:00",
        "t0_entry_price": 100.0,
        "expected_t1_timestamp": "2027-01-04T14:50:00+00:00",
        "original_terminal_timestamp": "2027-01-04T16:35:00+00:00",
        "provider_data_hash": "provider-hash",
        "forecast_freeze_timestamp": "2027-01-04T14:45:00+00:00",
    }


def _timing() -> dict[str, object]:
    return {
        "timing_id": "timing-1",
        "opportunity_id": "opp-1",
        "t1_entry_timestamp": "2027-01-04T14:50:00+00:00",
        "t1_entry_price": 99.5,
        "t1_availability": "available",
        "unavailability_reason": None,
        "data_availability_timestamp": "2027-01-04T14:50:00+00:00",
        "settlement_command_identity": "timing-command-1",
    }


def _outcome() -> dict[str, object]:
    return {
        "outcome_id": "outcome-1",
        "opportunity_id": "opp-1",
        "settlement_timestamp": "2027-01-04T16:35:00+00:00",
        "t0_gross_return_bps": 100.0,
        "t0_cost_bps": 10.0,
        "t0_net_return_bps": 90.0,
        "t1_gross_return_bps": 150.0,
        "t1_cost_bps": 10.0,
        "t1_net_return_bps": 140.0,
        "paired_difference_bps": 50.0,
    }


def test_prospective_ledgers_are_create_only_and_separately_settled(tmp_path: Path) -> None:
    ledger = ProspectiveLatencyLedger(tmp_path, opened_periods={2023, 2025})

    opportunity_path = ledger.append_opportunity(_opportunity(), holdout=True)
    timing_path = ledger.append_timing(_timing())
    outcome_path = ledger.append_outcome(_outcome())

    assert opportunity_path.parent.name == "opportunities"
    assert timing_path.parent.name == "timings"
    assert outcome_path.parent.name == "outcomes"
    frozen = json.loads(opportunity_path.read_text())
    assert frozen["research_only"] is True
    assert frozen["execution_enabled"] is False
    with pytest.raises(FileExistsError):
        ledger.append_opportunity(_opportunity(), holdout=True)


def test_holdout_mode_rejects_opened_periods() -> None:
    record = _opportunity()
    record["session"] = "2025-01-03"

    with pytest.raises(ValueError, match="opened period"):
        ProspectiveLatencyLedger(Path("unused"), opened_periods={2023, 2025}).validate_opportunity(
            record, holdout=True
        )


def test_timing_must_use_exact_t0_plus_five_minutes(tmp_path: Path) -> None:
    ledger = ProspectiveLatencyLedger(tmp_path, opened_periods={2023, 2025})
    ledger.append_opportunity(_opportunity(), holdout=True)
    timing = _timing()
    timing["t1_entry_timestamp"] = "2027-01-04T14:55:00+00:00"

    with pytest.raises(ValueError, match="exact expected T1"):
        ledger.append_timing(timing)


def test_hindsight_episode_or_payoff_cannot_enter_opportunity_features(tmp_path: Path) -> None:
    ledger = ProspectiveLatencyLedger(tmp_path, opened_periods={2023, 2025})
    record = _opportunity()
    record["feature_values"] = {"hindsight_episode_id": "episode-1"}

    with pytest.raises(ValueError, match="future outcome"):
        ledger.append_opportunity(record, holdout=True)
