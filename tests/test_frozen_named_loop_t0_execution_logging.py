from __future__ import annotations

import json
from pathlib import Path

import pytest

from stocker_research.frozen_named_loop_t0_execution import (
    DuplicateRecordError,
    IntegrityError,
    ProspectiveExecutionLedger,
    append_payloads,
    collection_parameters,
)

CONTRACT_HASH = "a" * 64
COMPLETION_HASH = "b" * 64
REAL_CONTRACT = Path(
    "research/slrno-v2/20260714-regime-loop-handoff/work/contracts/"
    "20260717-frozen-named-loop-t0-execution-realism-v1.json"
)


def opportunity(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "run_id": "prospective-run-1",
        "contract_hash": CONTRACT_HASH,
        "git_sha": "c" * 40,
        "code_version": "frozen_named_loop_t0_execution_v1.0.0",
        "data_snapshot_hash": "new-snapshot",
        "provider_data_hash": "new-provider",
        "source_model_version": "dynamic_loop_edge_state_v2.1.0",
        "source_run_id": "source-run-1",
        "opportunity_id": "opp-1",
        "anchor_id": "anchor-1",
        "event_lineage_id": "lineage-1",
        "symbol": "AAL",
        "session": "2026-07-17",
        "loop_id": "cycle_04",
        "orientation": "state_4",
        "family": "cycle_04|state_4",
        "classification": "named",
        "frozen_direction": 1,
        "anchor_timestamp": "2026-07-17T13:30:00Z",
        "anchor_close": 100.0,
        "long_threshold": 102.0,
        "short_threshold": 98.0,
        "threshold_known_timestamp": "2026-07-17T13:35:00Z",
        "signal_known_timestamp": "2026-07-17T13:35:00Z",
        "trigger_type": "opening_gap_through_threshold",
        "trigger_timestamp": "2026-07-17T13:35:00Z",
        "reference_fill_convention": "frozen_open_or_threshold",
        "reference_entry_timestamp": "2026-07-17T13:35:00Z",
        "reference_entry_price": 103.0,
        "original_terminal_timestamp": "2026-07-17T15:35:00Z",
        "feature_availability_timestamp": "2026-07-17T13:35:00Z",
        "source_availability_timestamp": "2026-07-17T13:35:00Z",
        "opportunity_created_timestamp": "2026-07-17T13:35:01Z",
        "research_only": True,
        "execution_enabled": False,
    }
    record.update(overrides)
    return record


def trigger(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "trigger_id": "trigger-1",
        "opportunity_id": "opp-1",
        "trigger_observed_timestamp": "2026-07-17T13:35:00Z",
        "trigger_bar_timestamp": "2026-07-17T13:35:00Z",
        "trigger_bar_open": 103.0,
        "trigger_bar_high": 104.0,
        "trigger_bar_low": 100.0,
        "trigger_bar_close": 101.0,
        "trigger_type": "opening_gap_through_threshold",
        "reference_entry_timestamp": "2026-07-17T13:35:00Z",
        "reference_entry_price": 103.0,
        "fill_evidence_classification": "GAP_FILL_OBSERVABLE",
        "signal_known_timestamp": "2026-07-17T13:35:00Z",
        "signal_fill_time_status": "CAUSALLY_ORDERED",
        "evidence_detail": "provider open directly observed beyond long threshold",
        "market_data_availability_timestamp": "2026-07-17T13:35:00Z",
        "provider_data_hash": "new-provider",
        "append_timestamp": "2026-07-17T13:35:02Z",
    }
    record.update(overrides)
    return record


def settlement(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "settlement_id": "settlement-1",
        "opportunity_id": "opp-1",
        "terminal_timestamp": "2026-07-17T15:35:00Z",
        "terminal_price": 104.0,
        "terminal_data_hash": "terminal-data",
        "terminal_data_availability_timestamp": "2026-07-17T15:35:01Z",
        "settlement_timestamp": "2026-07-17T15:35:02Z",
        "settlement_code_version": "frozen_named_loop_t0_execution_v1.0.0",
    }
    record.update(overrides)
    return record


def ledger(root: Path, **overrides: object) -> ProspectiveExecutionLedger:
    kwargs: dict[str, object] = {
        "root": root,
        "contract_hash": CONTRACT_HASH,
        "completion_rule_hash": COMPLETION_HASH,
        "opened_periods": {2023, 2025},
        "opened_snapshot_hashes": {"opened-snapshot"},
    }
    kwargs.update(overrides)
    return ProspectiveExecutionLedger(**kwargs)  # type: ignore[arg-type]


def test_prospective_opportunity_creation_is_append_only_and_safety_stamped(tmp_path: Path) -> None:
    path = ledger(tmp_path).append_opportunity(opportunity(), prospective=True)
    stored = json.loads(path.read_text())

    assert stored["research_only"] is True
    assert stored["execution_enabled"] is False
    assert stored["broker_connection_enabled"] is False
    assert stored["order_placement_enabled"] is False
    assert len(stored["record_sha256"]) == 64


def test_duplicate_opportunity_creation_fails_without_overwrite(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    path = store.append_opportunity(opportunity(), prospective=True)
    original = path.read_bytes()

    with pytest.raises(DuplicateRecordError):
        store.append_opportunity(opportunity(), prospective=True)

    assert path.read_bytes() == original


def test_changed_data_hash_for_existing_identity_fails_closed(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.append_opportunity(opportunity(), prospective=True)

    with pytest.raises(IntegrityError, match="different payload"):
        store.append_opportunity(opportunity(data_snapshot_hash="changed"), prospective=True)


@pytest.mark.parametrize(
    "change",
    [
        {"session": "2025-12-31"},
        {"data_snapshot_hash": "opened-snapshot"},
    ],
)
def test_opened_historical_data_is_rejected_in_genuine_prospective_mode(
    tmp_path: Path, change: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="opened"):
        ledger(tmp_path).append_opportunity(opportunity(**change), prospective=True)


def test_new_prospective_data_snapshot_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="snapshot"):
        ledger(tmp_path).append_opportunity(opportunity(data_snapshot_hash=""), prospective=True)


def test_current_or_future_payoff_cannot_alter_loop_classification(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outcome or hindsight"):
        ledger(tmp_path).append_opportunity(
            opportunity(classification="control", future_payoff_bps=1000.0), prospective=True
        )


def test_historical_episode_labels_never_enter_prospective_decisions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outcome or hindsight"):
        ledger(tmp_path).append_opportunity(
            opportunity(hindsight_episode_id="best-episode"), prospective=True
        )


def test_trigger_append_is_separate_and_cannot_overwrite_opportunity(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    opportunity_path = store.append_opportunity(opportunity(), prospective=True)
    original = opportunity_path.read_bytes()
    trigger_path = store.append_trigger(trigger())

    assert trigger_path.parent.name == "triggers"
    assert opportunity_path.read_bytes() == original
    assert json.loads(trigger_path.read_text())["opportunity_record_sha256"]


def test_trigger_append_requires_known_opportunity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown opportunity"):
        ledger(tmp_path).append_trigger(trigger())


def test_signal_known_after_reference_fill_is_rejected(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.append_opportunity(opportunity(), prospective=True)

    with pytest.raises(ValueError, match="signal-known"):
        store.append_trigger(trigger(signal_known_timestamp="2026-07-17T13:35:01Z"))


def test_trigger_changed_provider_hash_fails_closed(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.append_opportunity(opportunity(), prospective=True)

    with pytest.raises(IntegrityError, match="provider"):
        store.append_trigger(trigger(provider_data_hash="changed-provider"))


def test_settlement_requires_trigger_and_never_overwrites_it(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.append_opportunity(opportunity(), prospective=True)
    with pytest.raises(ValueError, match="trigger"):
        store.append_settlement(settlement())
    trigger_path = store.append_trigger(trigger())
    original = trigger_path.read_bytes()
    store.append_settlement(settlement())

    assert trigger_path.read_bytes() == original


def test_settlement_before_terminal_maturity_fails(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.append_opportunity(opportunity(), prospective=True)
    store.append_trigger(trigger())

    with pytest.raises(ValueError, match="mature"):
        store.append_settlement(settlement(settlement_timestamp="2026-07-17T15:34:59Z"))


def test_missing_terminal_data_remains_unavailable_not_zero(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.append_opportunity(opportunity(), prospective=True)
    store.append_trigger(trigger())

    with pytest.raises(ValueError, match="terminal price"):
        store.append_settlement(settlement(terminal_price=None))


def test_settlement_calculates_all_fill_models_from_frozen_trigger(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.append_opportunity(opportunity(), prospective=True)
    store.append_trigger(trigger())
    path = store.append_settlement(settlement())
    stored = json.loads(path.read_text())

    assert stored["family"] == "cycle_04|state_4"
    assert stored["classification"] == "named"
    assert stored["direction"] == 1
    assert stored["terminal_timestamp"] == "2026-07-17T15:35:00+00:00"
    assert stored["F0_cost_bps"] == 10.0
    assert stored["F0_net_payoff_bps"] == pytest.approx(10_000.0 * (104.0 / 103.0 - 1.0) - 10.0)
    assert stored["F10_net_payoff_bps"] < stored["F5_net_payoff_bps"]
    assert stored["fill_evidence_classification"] == "GAP_FILL_OBSERVABLE"


def test_duplicate_settlement_fails_safely(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.append_opportunity(opportunity(), prospective=True)
    store.append_trigger(trigger())
    store.append_settlement(settlement())

    with pytest.raises(DuplicateRecordError):
        store.append_settlement(settlement())


def test_completion_rule_cannot_change_after_collection_begins(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.append_opportunity(opportunity(), prospective=True)

    with pytest.raises(IntegrityError, match="collection identity"):
        ledger(tmp_path, completion_rule_hash="changed")


def test_interim_administrative_reporting_never_exposes_blinded_pnl(tmp_path: Path) -> None:
    store = ledger(tmp_path)
    store.append_opportunity(opportunity(), prospective=True)
    store.append_trigger(trigger())
    store.append_settlement(settlement())
    status = store.administrative_status()

    flattened = json.dumps(status).lower()
    assert "payoff" not in flattened
    assert "profit" not in flattened
    assert "f0_" not in flattened
    assert status["prospective_decision"] == "prospective_sample_incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        store.read_settlements_for_economic_scoring()


def test_command_support_derives_the_registered_collection_identity() -> None:
    parameters = collection_parameters(REAL_CONTRACT)

    assert len(str(parameters["contract_hash"])) == 64
    assert len(str(parameters["completion_rule_hash"])) == 64
    assert parameters["opened_periods"] == {2023, 2025}
    assert len(parameters["opened_snapshot_hashes"]) == 1  # type: ignore[arg-type]


def test_opportunity_command_dry_run_validates_without_creating_collection(
    tmp_path: Path,
) -> None:
    parameters = collection_parameters(REAL_CONTRACT)
    paths = append_payloads(
        ledger_root=tmp_path / "prospective",
        contract_path=REAL_CONTRACT,
        stage="opportunity",
        records=[opportunity(contract_hash=parameters["contract_hash"])],
        dry_run=True,
    )

    assert paths == [Path("DRY_RUN/opportunities/opp-1.json")]
    assert not (tmp_path / "prospective").exists()


def test_trigger_and_settlement_commands_preserve_separate_create_only_stages(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prospective"
    parameters = collection_parameters(REAL_CONTRACT)
    append_payloads(
        ledger_root=root,
        contract_path=REAL_CONTRACT,
        stage="opportunity",
        records=[opportunity(contract_hash=parameters["contract_hash"])],
        dry_run=False,
    )
    append_payloads(
        ledger_root=root,
        contract_path=REAL_CONTRACT,
        stage="trigger",
        records=[trigger()],
        dry_run=False,
    )
    paths = append_payloads(
        ledger_root=root,
        contract_path=REAL_CONTRACT,
        stage="settlement",
        records=[settlement()],
        dry_run=True,
    )

    assert paths == [Path("DRY_RUN/settlements/opp-1.json")]
    assert (root / "opportunities/opp-1.json").is_file()
    assert (root / "triggers/opp-1.json").is_file()
    assert not (root / "settlements/opp-1.json").exists()
