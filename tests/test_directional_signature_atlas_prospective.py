from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stocker_research.directional_signature_atlas.io import (
    assert_research_only_paths,
    write_deterministic_parquet,
)
from stocker_research.directional_signature_atlas.prospective import (
    ProspectiveLedger,
    build_forecast_record,
    build_settlement_record,
)


def _forecast(opportunity_id: str = "atlas|2026-07-20|AAA|12") -> dict[str, object]:
    return {
        "opportunity_id": opportunity_id,
        "session": "2026-07-20",
        "forecast_freeze_timestamp": "2026-07-20T14:30:00+00:00",
        "research_only": True,
        "execution_enabled": False,
    }


def test_prospective_forecast_records_are_append_only(tmp_path: Path) -> None:
    ledger = ProspectiveLedger(tmp_path, opened_through="2026-06-26")
    ledger.append_forecast(_forecast())
    original = ledger.forecast_path.read_bytes()
    with pytest.raises(FileExistsError, match="duplicate"):
        ledger.append_forecast(_forecast())
    assert ledger.forecast_path.read_bytes() == original


def test_settlement_cannot_overwrite_forecast_records(tmp_path: Path) -> None:
    ledger = ProspectiveLedger(tmp_path, opened_through="2026-06-26")
    ledger.append_forecast(_forecast())
    original = ledger.forecast_path.read_bytes()
    ledger.append_settlement(
        {
            "opportunity_id": _forecast()["opportunity_id"],
            "terminal_timestamp": "2026-07-20T16:30:00+00:00",
            "primary_target": "NEUTRAL",
        }
    )
    assert ledger.forecast_path.read_bytes() == original
    assert ledger.settlement_path.is_file()


def test_duplicate_prospective_opportunity_ids_fail_closed(tmp_path: Path) -> None:
    ledger = ProspectiveLedger(tmp_path, opened_through="2026-06-26")
    ledger.append_forecast(_forecast())
    with pytest.raises(FileExistsError):
        ledger.append_forecast(_forecast())


def test_prospective_mode_rejects_opened_historical_snapshot(tmp_path: Path) -> None:
    ledger = ProspectiveLedger(tmp_path, opened_through="2026-06-26")
    historical = _forecast("atlas|2026-06-01|AAA|12") | {"session": "2026-06-01"}
    with pytest.raises(ValueError, match="opened historical"):
        ledger.append_forecast(historical)


def test_primary_and_exact_rerun_parquet_are_byte_identical(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "opportunity_id": ["b", "a"],
            "value": [2.0, 1.0],
            "timestamp": pd.to_datetime(["2025-01-02T15:35Z", "2025-01-02T15:30Z"]),
        }
    )
    primary = tmp_path / "primary.parquet"
    rerun = tmp_path / "rerun.parquet"
    write_deterministic_parquet(frame, primary, sort_by=["opportunity_id"])
    write_deterministic_parquet(frame, rerun, sort_by=["opportunity_id"])
    assert primary.read_bytes() == rerun.read_bytes()


def test_safety_checks_reject_execution_runtime_paths(tmp_path: Path) -> None:
    safe = [tmp_path / "research" / "directional_signature_atlas" / "report.md"]
    assert_research_only_paths(safe)
    with pytest.raises(ValueError, match="research-only boundary"):
        assert_research_only_paths([tmp_path / "packages" / "stocker_execution" / "orders.py"])


def test_forecast_builder_evaluates_frozen_votes_and_keeps_outcomes_absent() -> None:
    metadata = {
        "run_id": "run",
        "git_sha": "abc",
        "contract_sha256": "contract",
        "data_snapshot_sha256": "data",
        "feature_schema_sha256": "schema",
    }
    feature_row = {
        "opportunity_id": "atlas|2026|AAA|2026-07-20|12",
        "symbol": "AAA",
        "session": "2026-07-20",
        "decision_clock": "clock_12",
        "decision_timestamp": "2026-07-20T14:30:00+00:00",
        "movement_permission": True,
        "signal": "yes",
        "movement_permission__available_at": "2026-07-20T14:30:00+00:00",
        "signal__available_at": "2026-07-20T14:30:00+00:00",
    }
    long_entry = {
        "signature": {
            "signature_id": "long_signal",
            "direction": "LONG",
            "conditions": [
                {"feature": "signal", "operator": "==", "value": "yes", "family": "test"}
            ],
        },
        "conservative_value_bps": 2.0,
    }
    short_entry = {
        "signature": {
            "signature_id": "short_signal",
            "direction": "SHORT",
            "conditions": [
                {"feature": "signal", "operator": "==", "value": "yes", "family": "test"}
            ],
        },
        "conservative_value_bps": 3.0,
    }
    record = build_forecast_record(
        feature_row,
        metadata=metadata,
        long_library=[long_entry],
        short_library=[short_entry],
        causal_feature_names=["signal", "movement_permission"],
        forecast_freeze_timestamp="2026-07-20T14:30:00+00:00",
    )
    assert record["final_atlas_state"] == "NEUTRAL"
    assert record["conflict_state"] is True
    assert record["long_vote_count"] == record["short_vote_count"] == 1
    assert "target" not in record and "payoff" not in record


def test_settlement_builder_is_a_separate_economic_record() -> None:
    settlement = build_settlement_record(
        {
            "opportunity_id": "atlas|2026|AAA|2026-07-20|12",
            "terminal_timestamp": "2026-07-20T16:30:00+00:00",
            "gross_long_return_bps": 30.0,
            "gross_short_return_bps": -30.0,
            "round_trip_cost_bps": 10.0,
            "net_long_return_bps": 20.0,
            "net_short_return_bps": -40.0,
            "target": "LONG",
            "first_touch_target": "UPPER_FIRST",
        },
        settlement_timestamp="2026-07-20T16:31:00+00:00",
        settlement_code_version="abc",
    )
    assert settlement["primary_target"] == "LONG"
    assert settlement["net_long_payoff_bps"] == 20.0
    assert "causal_features" not in settlement
