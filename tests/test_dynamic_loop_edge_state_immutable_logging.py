from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from stocker_research.dynamic_loop_edge_state_lead_lag.immutable_ledger import (
    ProspectiveResearchLedger,
)

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "research/slrno-v2/20260714-regime-loop-handoff/work"
V2_PRIMARY = WORK / "artifacts/20260714-dynamic-loop-edge-state-v2/primary"


def _runner() -> ModuleType:
    path = WORK / "run_dynamic_loop_edge_state_lead_lag_v1.py"
    spec = importlib.util.spec_from_file_location("lead_lag_runner_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_prospective_mode_requires_new_source_and_appends_outcome_separately(
    tmp_path: Path,
) -> None:
    runner = _runner()
    contract, contract_hash = runner.load_contract()
    source = pd.read_parquet(V2_PRIMARY / "causal_edge_state_forecasts.parquet")
    source = source.loc[source["score_session"].eq(source["score_session"].iloc[0])].head(4)
    source = source.copy()
    source["period"] = 2099
    source["score_session"] = "2099-01-05"
    source["decision_timestamp"] = pd.Timestamp("2099-01-05T14:30:00Z")
    source["prediction_frozen_at"] = pd.Timestamp("2099-01-05T14:30:00Z")
    source["feature_max_availability_timestamp"] = pd.Timestamp("2099-01-04T21:00:00Z")
    source["training_latest_availability_timestamp"] = pd.Timestamp("2099-01-04T21:00:00Z")
    source["run_id"] = "prospective-source-run"
    source_path = tmp_path / "new_forecasts.parquet"
    source.to_parquet(source_path, index=False)
    ledger_root = tmp_path / "ledger"

    count = runner.prospective_log_session(
        "2099-01-05",
        ledger_root,
        contract,
        contract_hash,
        source_path=source_path,
        data_snapshot_hash="new-unopened-snapshot",
        current_timestamp=pd.Timestamp("2099-01-05T14:30:00Z"),
    )

    assert count == len(source)
    forecast_files = sorted((ledger_root / "forecasts").glob("*.json"))
    assert len(forecast_files) == len(source)
    forecast = json.loads(forecast_files[0].read_text())
    assert forecast["data_snapshot_hash"] == "new-unopened-snapshot"
    outcome_path = tmp_path / "outcomes.csv"
    pd.DataFrame(
        [
            {
                "outcome_id": "prospective-outcome-1",
                "forecast_id": forecast["forecast_id"],
                "target_session": "2099-01-06",
                "target_lead_sessions": 1,
                "settlement_timestamp": "2099-01-06T18:00:00Z",
                "target_robust_net_bps": 12.0,
            }
        ]
    ).to_csv(outcome_path, index=False)

    assert runner.append_prospective_outcomes(outcome_path, ledger_root) == 1
    assert (ledger_root / "outcomes" / "prospective-outcome-1.json").exists()


def test_prospective_mode_rejects_opened_v2_copy_and_empty_session(tmp_path: Path) -> None:
    runner = _runner()
    contract, contract_hash = runner.load_contract()
    opened = V2_PRIMARY / "causal_edge_state_forecasts.parquet"

    with pytest.raises(ValueError, match="opened V2 period"):
        runner.prospective_log_session(
            "2025-12-31",
            tmp_path / "opened",
            contract,
            contract_hash,
            source_path=opened,
            data_snapshot_hash="new-snapshot",
            current_timestamp=pd.Timestamp("2025-12-31T14:35:00Z"),
        )

    future = pd.read_parquet(opened).head(1).copy()
    future["period"] = 2099
    future["score_session"] = "2099-01-05"
    future["decision_timestamp"] = pd.Timestamp("2099-01-05T14:30:00Z")
    future["prediction_frozen_at"] = pd.Timestamp("2099-01-05T14:30:00Z")
    future["feature_max_availability_timestamp"] = pd.Timestamp("2099-01-04T21:00:00Z")
    future["training_latest_availability_timestamp"] = pd.Timestamp("2099-01-04T21:00:00Z")
    future["run_id"] = "prospective-source-run"
    path = tmp_path / "future.parquet"
    future.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="no prospective forecasts"):
        runner.prospective_log_session(
            "2099-01-06",
            tmp_path / "empty",
            contract,
            contract_hash,
            source_path=path,
            data_snapshot_hash="new-snapshot",
            current_timestamp=pd.Timestamp("2099-01-06T14:30:00Z"),
        )


def test_scientific_support_requires_every_registered_gate() -> None:
    runner = _runner()
    metrics = pd.DataFrame(
        {
            "scope": ["all"],
            "target_lead_sessions": [1],
            "paired_brier_improvement": [0.01],
            "paired_economic_increment_bps": [100.0],
        }
    )

    assert runner.scientific_decision(metrics) == "hypothesis_rejected"
    gates = {name: True for name in runner.SUPPORT_DECISION_GATES}
    assert (
        runner.scientific_decision(metrics, support_gates=gates)
        == "supported_prospectively_only_required"
    )
