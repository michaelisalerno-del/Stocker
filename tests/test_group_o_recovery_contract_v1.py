from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from stocker_prospective import group_o_recovery
from stocker_prospective.group_o_recovery import (
    GroupORecoveryIntegrityError,
    GroupORecoveryResult,
    require_group_o_recovery_ready_before_adapter_v1,
    verify_group_o_recovery_freeze_v1,
)
from stocker_prospective.opening_leader_continuation_v0 import CANONICAL_COHORT_V0
from stocker_prospective.scientific_inputs import GroupOAcquisitionPending

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "prospective" / "m1c-group-o-recovery" / "20260802-m1c-group-o-late-revision-v1"


def test_group_o_recovery_contract_is_causal_append_only_and_record_only() -> None:
    contract = json.loads((PACKAGE / "contract.json").read_text(encoding="utf-8"))
    protected = json.loads((PACKAGE / "protected_boundary_audit.json").read_text(encoding="utf-8"))
    order_audit = json.loads((PACKAGE / "order_disable_audit.json").read_text(encoding="utf-8"))

    assert contract["contract_version"] == "m1c-group-o-late-revision-v1"
    assert contract["base_package_replacement_allowed"] is False
    assert contract["acquisition_attempts_append_only"] is True
    assert contract["revision_cutoff"] == "strictly_before_exact_xnys_signal_session_open"
    assert contract["frozen_retry_delay_minutes"] == 15
    assert contract["pre_adapter_gate"].startswith("permanent_for_this_recovery_version")
    assert contract["gate_decommission_rule"].startswith("requires_a_new_signed_recorder")
    assert contract["completion_receipt"].startswith("self_binding_hash_links")
    assert contract["revision_semantics_scope"].startswith("generic_future_signal_session")
    assert contract["m1c_scoring_semantics_changed"] is False
    assert contract["opening_leader_m1c_role"] == "context_only"
    assert protected["target_observation_session"] == "2026-07-31"
    assert protected["target_signal_session"] == "2026-08-03"
    assert protected["opening_leader_historical_outcomes_accessed"] is False
    assert protected["monday_2026_08_03_market_data_consumed_before_revision"] is False
    assert order_audit["runtime_mode_required"] == "record_only"
    assert order_audit["broker_order_methods_reachable"] is False
    assert order_audit["pre_adapter_recovery_command_uses_ibkr"] is False


def test_group_o_recovery_deployment_freeze_receipt_verifies() -> None:
    receipt = verify_group_o_recovery_freeze_v1(ROOT)

    assert receipt["target_observation_session"] == "2026-07-31"
    assert receipt["target_signal_session"] == "2026-08-03"
    assert receipt["order_placement_disabled"] is True
    assert receipt["protected_outcomes_accessed"] is False
    assert receipt["audited_failed_base_sha256"] == (
        "b8ba77cfcb6ff82c7fd1f205e5b2cbcd562ea2a09d7e3e1afa703fb0cc77d75c"
    )
    assert set(receipt["verification"]) == {
        "review_findings_addressed",
        "scientific_input_tests",
        "scoped_lint",
        "scoped_type_check",
        "static_order_surface_audit",
    }


def test_group_o_recovery_blocks_before_adapter_while_exact_chain_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        group_o_recovery,
        "verify_group_o_recovery_freeze_v1",
        lambda _release: {"deployment_receipt_id": "fixture-deployment"},
    )
    monkeypatch.setattr(
        group_o_recovery,
        "load_group_o_session_package",
        lambda **_kwargs: SimpleNamespace(
            contexts=(SimpleNamespace(quality_status="missing_exact_chain"),)
        ),
    )
    monkeypatch.setattr(
        group_o_recovery,
        "_require_audited_failed_base",
        lambda **_kwargs: tmp_path / "context" / "group-o" / "2026-08-03.json",
    )

    with pytest.raises(
        GroupORecoveryIntegrityError,
        match="blocked_pre_adapter_group_o_recovery_incomplete",
    ):
        require_group_o_recovery_ready_before_adapter_v1(
            context_root=tmp_path / "context",
            release_directory=ROOT,
            now=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )


def test_recorder_invokes_recovery_gate_before_constructing_ibkr_adapter() -> None:
    source = (
        ROOT
        / "packages"
        / "stocker_prospective"
        / "src"
        / "stocker_prospective"
        / "cli.py"
    ).read_text(encoding="utf-8")
    recorder = source[source.index("def recorder_run(") :]

    assert "if recovery_receipt.is_file():" not in recorder
    assert recorder.index("require_group_o_recovery_ready_before_adapter_v1(") < recorder.index(
        "adapter = _ibkr_adapter(config)"
    )


def test_pre_adapter_recovery_automatically_waits_for_frozen_retry_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    calls = 0
    sleeps: list[float] = []

    def recover_once(**_kwargs: object) -> GroupORecoveryResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise GroupOAcquisitionPending("fixture source publication lag")
        return GroupORecoveryResult(
            status="recovered",
            signal_session=group_o_recovery.TARGET_SIGNAL_SESSION_V1,
            observation_session=group_o_recovery.TARGET_OBSERVATION_SESSION_V1,
            attempt_id="0002",
            start_receipt_path=tmp_path / "0002" / "recovery_start_receipt.json",
            canonical_option_rows=40,
        )

    def sleep(seconds: float) -> None:
        nonlocal current
        sleeps.append(seconds)
        current += timedelta(seconds=seconds)

    monkeypatch.setattr(group_o_recovery, "recover_group_o_exact_chain_v1", recover_once)
    monkeypatch.setattr(
        group_o_recovery,
        "group_o_retry_not_before",
        lambda **_kwargs: datetime(2026, 8, 2, 12, 15, tzinfo=UTC),
    )

    result = group_o_recovery.recover_group_o_exact_chain_until_ready_v1(
        context_root=tmp_path / "context",
        release_directory=ROOT,
        symbols=CANONICAL_COHORT_V0,
        clock=lambda: current,
        sleeper=sleep,
    )

    assert result.status == "recovered"
    assert calls == 2
    assert sleeps == [900.0]


def test_group_o_recovery_modules_have_no_order_capable_surface() -> None:
    sources = "\n".join(
        (
            ROOT / "packages" / "stocker_prospective" / "src" / "stocker_prospective" / name
        ).read_text(encoding="utf-8")
        for name in (
            "append_only.py",
            "group_o.py",
            "group_o_recovery.py",
            "scientific_inputs.py",
        )
    )

    for forbidden in ("placeOrder", "reqIds", "Transmit", "orderId", "totalQuantity"):
        assert forbidden not in sources
