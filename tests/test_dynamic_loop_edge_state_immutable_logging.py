from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stocker_research.dynamic_loop_edge_state_lead_lag.immutable_ledger import (
    ProspectiveResearchLedger,
)


def _forecast() -> dict[str, object]:
    return {
        "forecast_id": "forecast-1",
        "run_id": "run-1",
        "git_sha": "abc",
        "contract_hash": "def",
        "model_version": "v1",
        "data_snapshot_hash": "data",
        "feature_schema_version": "features",
        "forecast_creation_timestamp": "2025-01-03T14:30:00Z",
        "forecast_effective_session": "2025-01-03",
        "stock_id": None,
        "loop_id": "cycle_01",
        "orientation": "state_1",
        "horizon": 24,
        "model_name": "hierarchical_change_point",
        "p_next_payoff_positive": 0.7,
        "p_edge_positive": 0.65,
        "p_edge_active": 0.7,
        "p_change_now": 0.1,
        "p_on_next": 0.6,
        "p_off_next": 0.2,
        "p_survive_horizon": 0.7,
        "posterior_mean_net_bps": 8.0,
        "posterior_lower_bound_net_bps": 1.0,
        "posterior_run_length_mean": 4.0,
        "edge_state": "active",
        "reason_codes": [],
        "independent_session_support": 10,
        "independent_stock_support": 4,
        "effective_sample_size": 8.0,
        "forecast_freeze_timestamp": "2025-01-03T14:30:00Z",
        "feature_max_availability_timestamp": "2025-01-02T21:00:00Z",
        "feature_availability_timestamps": {"structural_breadth": "2025-01-02T21:00:00Z"},
        "frozen_feature_values": {"structural_breadth": 0.2},
    }


def test_forecast_and_outcome_records_are_unique_and_separate(tmp_path: Path) -> None:
    ledger = ProspectiveResearchLedger(tmp_path)
    ledger.append_forecast(_forecast())
    forecast_path = tmp_path / "forecasts" / "forecast-1.json"
    before = forecast_path.read_bytes()

    ledger.append_outcome(
        {
            "outcome_id": "outcome-1",
            "forecast_id": "forecast-1",
            "target_session": "2025-01-06",
            "target_lead_sessions": 1,
            "settlement_timestamp": "2025-01-06T18:00:00Z",
            "target_robust_net_bps": 12.0,
        }
    )

    assert forecast_path.read_bytes() == before
    assert (tmp_path / "outcomes" / "outcome-1.json").exists()
    with pytest.raises(FileExistsError):
        ledger.append_forecast(_forecast())
    with pytest.raises(FileExistsError):
        ledger.append_outcome(
            {
                "outcome_id": "outcome-1",
                "forecast_id": "forecast-1",
                "target_session": "2025-01-06",
                "target_lead_sessions": 1,
                "settlement_timestamp": "2025-01-06T18:00:00Z",
                "target_robust_net_bps": 99.0,
            }
        )


def test_outcome_must_reference_exact_existing_forecast(tmp_path: Path) -> None:
    ledger = ProspectiveResearchLedger(tmp_path)

    with pytest.raises(ValueError, match="unknown forecast"):
        ledger.append_outcome(
            {
                "outcome_id": "outcome-1",
                "forecast_id": "missing",
                "target_session": "2025-01-06",
                "target_lead_sessions": 1,
                "settlement_timestamp": "2025-01-06T18:00:00Z",
                "target_robust_net_bps": 12.0,
            }
        )


def test_forecast_rejects_future_features_and_hindsight_episode_labels(tmp_path: Path) -> None:
    ledger = ProspectiveResearchLedger(tmp_path)
    future = _forecast()
    future["feature_max_availability_timestamp"] = "2025-01-06T14:30:00Z"
    with pytest.raises(ValueError, match="feature availability"):
        ledger.append_forecast(future)

    hindsight = _forecast()
    hindsight["forecast_id"] = "forecast-2"
    hindsight["frozen_feature_values"] = {"hindsight_episode_state": "positive"}
    with pytest.raises(ValueError, match="hindsight or episode"):
        ledger.append_forecast(hindsight)


def test_serialized_record_is_canonical_and_research_only(tmp_path: Path) -> None:
    ledger = ProspectiveResearchLedger(tmp_path)
    ledger.append_forecast(_forecast())
    record = json.loads((tmp_path / "forecasts" / "forecast-1.json").read_text())

    assert record["research_only"] is True
    assert record["execution_enabled"] is False
    assert record["order_placement_enabled"] is False
    assert pd.Timestamp(record["forecast_freeze_timestamp"]) == pd.Timestamp("2025-01-03T14:30:00Z")
