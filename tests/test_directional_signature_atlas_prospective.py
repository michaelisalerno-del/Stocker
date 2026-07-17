from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    canonical_library_hash,
)


def _forecast(opportunity_id: str = "atlas|2026|AAA|2026-07-20|12") -> dict[str, object]:
    return {
        "run_id": "run",
        "git_sha": "abc",
        "contract_hash": "contract",
        "data_snapshot_hash": "prospective-input-data",
        "training_data_snapshot_hash": "data",
        "feature_schema_hash": "schema",
        "long_library_hash": "long",
        "short_library_hash": "short",
        "neutral_library_hash": "neutral",
        "opportunity_id": opportunity_id,
        "symbol": "AAA",
        "session": "2026-07-20",
        "decision_clock": "clock_12",
        "decision_timestamp": "2026-07-20T14:30:00+00:00",
        "entry_timestamp": "2026-07-20T14:35:00+00:00",
        "terminal_timestamp": "2026-07-20T16:30:00+00:00",
        "causal_features": {"signal": "yes"},
        "feature_availability_timestamps": {"signal": "2026-07-20T14:30:00+00:00"},
        "movement_permission": "PASS",
        "long_signature_decisions": {},
        "short_signature_decisions": {},
        "long_vote_count": 0,
        "short_vote_count": 0,
        "conflict_state": False,
        "final_atlas_state": "NEUTRAL",
        "reason_codes": ["no_directional_vote"],
        "forecast_freeze_timestamp": "2026-07-20T14:30:00+00:00",
        "research_only": True,
        "execution_enabled": False,
    }


def _settlement(opportunity_id: str = "atlas|2026|AAA|2026-07-20|12") -> dict[str, object]:
    return {
        "opportunity_id": opportunity_id,
        "terminal_timestamp": "2026-07-20T16:30:00+00:00",
        "gross_long_payoff_bps": 0.0,
        "gross_short_payoff_bps": 0.0,
        "costs_bps": 10.0,
        "net_long_payoff_bps": -10.0,
        "net_short_payoff_bps": -10.0,
        "primary_target": "NEUTRAL",
        "secondary_first_touch_target": "NEITHER",
        "settlement_timestamp": "2026-07-20T16:31:00+00:00",
        "settlement_code_version": "abc",
        "settlement_status": "SETTLED",
        "unavailable_reason": None,
        "research_only": True,
        "execution_enabled": False,
    }


def _ledger(
    root: Path,
    *,
    completion_requirements: dict[str, int] | None = None,
) -> ProspectiveLedger:
    return ProspectiveLedger(
        root,
        opened_through="2026-06-26",
        required_causal_feature_names=["signal"],
        expected_identity={
            "run_id": "run",
            "git_sha": "abc",
            "contract_hash": "contract",
            "training_data_snapshot_hash": "data",
            "feature_schema_hash": "schema",
            "long_library_hash": "long",
            "short_library_hash": "short",
            "neutral_library_hash": "neutral",
        },
        completion_requirements=completion_requirements
        or {
            "minimum_settled_opportunities": 2000,
            "minimum_independent_sessions": 100,
            "minimum_stocks": 15,
            "minimum_completed_calendar_months": 4,
            "minimum_long_outputs": 100,
            "minimum_short_outputs": 100,
            "minimum_sessions_with_long": 30,
            "minimum_sessions_with_short": 30,
        },
    )


def test_prospective_forecast_records_are_append_only(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.append_forecast(_forecast())
    original = ledger.forecast_path.read_bytes()
    with pytest.raises(FileExistsError, match="duplicate"):
        ledger.append_forecast(_forecast())
    assert ledger.forecast_path.read_bytes() == original


def test_settlement_cannot_overwrite_forecast_records(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.append_forecast(_forecast())
    original = ledger.forecast_path.read_bytes()
    ledger.append_settlement(_settlement())
    assert ledger.forecast_path.read_bytes() == original
    assert ledger.settlement_path.is_file()


def test_duplicate_prospective_opportunity_ids_fail_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.append_forecast(_forecast())
    with pytest.raises(FileExistsError):
        ledger.append_forecast(_forecast())


def test_forecast_requires_complete_outcome_free_causal_schema(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    incomplete = _forecast()
    incomplete.pop("git_sha")
    with pytest.raises(ValueError, match="missing required forecast fields"):
        ledger.append_forecast(incomplete)

    forbidden = _forecast() | {
        "causal_features": {"future_return_24": 2.0},
        "feature_availability_timestamps": {"future_return_24": "2026-07-20T14:30:00+00:00"},
    }
    with pytest.raises(ValueError, match="forbidden causal feature"):
        ledger.append_forecast(forbidden)


def test_forecast_feature_availability_cannot_exceed_freeze(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    future_available = _forecast() | {
        "feature_availability_timestamps": {"signal": "2026-07-20T14:31:00+00:00"}
    }
    with pytest.raises(ValueError, match="after forecast freeze"):
        ledger.append_forecast(future_available)


def test_forecast_freeze_is_between_decision_and_entry(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    before_decision = _forecast() | {"forecast_freeze_timestamp": "2026-07-20T14:29:00+00:00"}
    with pytest.raises(ValueError, match="completed decision"):
        ledger.append_forecast(before_decision)
    at_entry = _forecast() | {"forecast_freeze_timestamp": "2026-07-20T14:35:00+00:00"}
    with pytest.raises(ValueError, match="before entry"):
        ledger.append_forecast(at_entry)


def test_tampered_controller_state_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    tampered = _forecast() | {
        "movement_permission": "FAIL",
        "final_atlas_state": "LONG",
        "reason_codes": ["supported_long_vote"],
    }
    with pytest.raises(ValueError, match="must produce a neutral forecast"):
        ledger.append_forecast(tampered)


def test_settlement_requires_complete_mature_consistent_economics(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.append_forecast(_forecast())

    incomplete = _settlement()
    incomplete.pop("costs_bps")
    with pytest.raises(ValueError, match="missing required settlement fields"):
        ledger.append_settlement(incomplete)

    wrong_terminal = _settlement() | {
        "terminal_timestamp": "2026-07-20T16:35:00+00:00",
        "settlement_timestamp": "2026-07-20T16:36:00+00:00",
    }
    with pytest.raises(ValueError, match="frozen forecast terminal"):
        ledger.append_settlement(wrong_terminal)

    premature = _settlement() | {"settlement_timestamp": "2026-07-20T16:29:00+00:00"}
    with pytest.raises(ValueError, match="terminal matures"):
        ledger.append_settlement(premature)

    inconsistent = _settlement() | {"net_long_payoff_bps": 1.0}
    with pytest.raises(ValueError, match="net long payoff"):
        ledger.append_settlement(inconsistent)


def test_concurrent_duplicate_forecasts_have_one_winner(tmp_path: Path) -> None:
    first = _ledger(tmp_path)
    second = _ledger(tmp_path)

    def append(ledger: ProspectiveLedger) -> str:
        try:
            ledger.append_forecast(_forecast())
        except FileExistsError:
            return "duplicate"
        return "written"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(append, (first, second)))
    assert sorted(results) == ["duplicate", "written"]
    assert len(first._records(first.forecast_path)) == 1


def test_prospective_mode_rejects_opened_historical_snapshot(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    historical = _forecast("atlas|2026|AAA|2026-06-01|12") | {
        "session": "2026-06-01",
        "decision_timestamp": "2026-06-01T14:30:00+00:00",
        "entry_timestamp": "2026-06-01T14:35:00+00:00",
        "terminal_timestamp": "2026-06-01T16:30:00+00:00",
        "feature_availability_timestamps": {"signal": "2026-06-01T14:30:00+00:00"},
        "forecast_freeze_timestamp": "2026-06-01T14:30:00+00:00",
    }
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
        "long_library_sha256": "",
        "short_library_sha256": "",
        "neutral_library_sha256": "",
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
    metadata["long_library_sha256"] = canonical_library_hash([long_entry])
    metadata["short_library_sha256"] = canonical_library_hash([short_entry])
    metadata["neutral_library_sha256"] = canonical_library_hash([])
    record = build_forecast_record(
        feature_row,
        metadata=metadata,
        long_library=[long_entry],
        short_library=[short_entry],
        neutral_library=[],
        causal_feature_names=["signal", "movement_permission"],
        forecast_input_snapshot_hash="prospective-input-data",
        forecast_freeze_timestamp="2026-07-20T14:30:00+00:00",
    )
    assert record["final_atlas_state"] == "NEUTRAL"
    assert record["conflict_state"] is True
    assert record["long_vote_count"] == record["short_vote_count"] == 1
    assert "target" not in record and "payoff" not in record

    with pytest.raises(ValueError, match="signature features missing"):
        build_forecast_record(
            feature_row,
            metadata=metadata,
            long_library=[long_entry],
            short_library=[],
            neutral_library=[],
            causal_feature_names=["movement_permission"],
            forecast_input_snapshot_hash="prospective-input-data",
            forecast_freeze_timestamp="2026-07-20T14:30:00+00:00",
        )


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

    dual_touch = build_settlement_record(
        {
            "opportunity_id": "atlas|2026|AAA|2026-07-20|12",
            "terminal_timestamp": "2026-07-20T16:30:00+00:00",
            "gross_long_return_bps": 0.0,
            "gross_short_return_bps": 0.0,
            "round_trip_cost_bps": 10.0,
            "net_long_return_bps": -10.0,
            "net_short_return_bps": -10.0,
            "target": "NEUTRAL",
            "first_touch_target": "SAME_BAR_DUAL_TOUCH",
        },
        settlement_timestamp="2026-07-20T16:31:00+00:00",
        settlement_code_version="abc",
    )
    assert dual_touch["secondary_first_touch_target"] == "SAME_BAR_DUAL_TOUCH"


def test_frozen_clock_rejects_restarted_or_off_grid_horizons(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError, match="next-provider-open"):
        ledger.append_forecast(_forecast() | {"entry_timestamp": "2026-07-20T14:31:00+00:00"})
    with pytest.raises(ValueError, match="fixed 24-bar"):
        ledger.append_forecast(_forecast() | {"terminal_timestamp": "2026-07-20T16:25:00+00:00"})


def test_ledger_rejects_identity_or_library_drift(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError, match="frozen identity mismatch"):
        ledger.append_forecast(_forecast() | {"long_library_hash": "different"})


def test_prospective_completion_gate_blinds_economics_until_complete(tmp_path: Path) -> None:
    incomplete = _ledger(tmp_path / "incomplete")
    incomplete.append_forecast(_forecast())
    incomplete.append_settlement(_settlement())
    assert incomplete.completion_status()["requirements_met"] is False
    with pytest.raises(PermissionError, match="completion rule"):
        incomplete.read_settlements()

    complete = _ledger(
        tmp_path / "complete",
        completion_requirements={
            "minimum_settled_opportunities": 1,
            "minimum_independent_sessions": 1,
            "minimum_stocks": 1,
            "minimum_completed_calendar_months": 0,
            "minimum_long_outputs": 0,
            "minimum_short_outputs": 0,
            "minimum_sessions_with_long": 0,
            "minimum_sessions_with_short": 0,
        },
    )
    complete.append_forecast(_forecast())
    complete.append_settlement(_settlement())
    assert complete.completion_status()["requirements_met"] is True
    assert len(complete.read_settlements()) == 1


def test_unavailable_matured_outcome_is_retained_without_zero_economics(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    ledger.append_forecast(_forecast())
    unavailable = build_settlement_record(
        {
            "opportunity_id": "atlas|2026|AAA|2026-07-20|12",
            "terminal_timestamp": "2026-07-20T16:30:00+00:00",
            "target": "UNAVAILABLE",
            "first_touch_target": "UNAVAILABLE",
            "score_status": "missing_exact_24_bar_path",
        },
        settlement_timestamp="2026-07-20T16:31:00+00:00",
        settlement_code_version="abc",
    )
    ledger.append_settlement(unavailable)
    assert unavailable["gross_long_payoff_bps"] is None
    assert ledger.completion_status()["matured_unavailable_records"] == 1
