from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from stocker_prospective import group_o_recovery
from stocker_prospective.group_o_recovery import (
    GroupORecoveryIntegrityError,
    GroupORecoveryResult,
    require_group_o_recovery_ready_before_adapter_v2,
    verify_group_o_recovery_freeze_v2,
)
from stocker_prospective.opening_leader_continuation_v0 import CANONICAL_COHORT_V0
from stocker_prospective.scientific_inputs import (
    GroupOAcquisitionPending,
    load_group_o_attempt_receipt,
    write_group_o_attempt_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "prospective" / "m1c-group-o-recovery" / "20260802-m1c-group-o-late-revision-v2"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _resign_deployment_receipt(receipt: dict[str, object]) -> dict[str, object]:
    unsigned = dict(receipt)
    unsigned.pop("signature_sha256", None)
    unsigned.pop("deployment_receipt_id", None)
    identity_hash = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    unsigned["deployment_receipt_id"] = f"group-o-recovery-deploy-{identity_hash[:24]}"
    unsigned["signature_sha256"] = hashlib.sha256(
        _canonical_json(unsigned).encode("utf-8")
    ).hexdigest()
    return unsigned


def _copy_recovery_release(tmp_path: Path) -> Path:
    release = tmp_path / "release"
    for relative in set(group_o_recovery.RECOVERY_SOURCE_FILES_V2.values()):
        source = ROOT / relative
        destination = release / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative in (
        group_o_recovery.FAILED_V1_PACKAGE_RELATIVE,
        group_o_recovery.RECOVERY_PACKAGE_RELATIVE_V2,
    ):
        shutil.copytree(ROOT / relative, release / relative)
    return release


def test_group_o_recovery_contract_is_causal_append_only_and_record_only() -> None:
    contract = json.loads((PACKAGE / "contract.json").read_text(encoding="utf-8"))
    protected = json.loads((PACKAGE / "protected_boundary_audit.json").read_text(encoding="utf-8"))
    order_audit = json.loads((PACKAGE / "order_disable_audit.json").read_text(encoding="utf-8"))
    prior = json.loads((PACKAGE / "prior_failure_audit.json").read_text(encoding="utf-8"))

    assert contract["contract_version"] == "m1c-group-o-late-revision-v2"
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
    assert contract["provider_dte_policy"] == "recompute_from_eod_identity"
    assert contract["provider_dte_used_for_admission"] is False
    assert contract["provider_dte_diagnostic_evidence"].startswith("append_only")
    assert contract["quote_observation_date_mismatch_allowed"] is False
    assert contract["failed_v1_attempt_reuse_allowed"] is False
    assert contract["failed_v1_start_receipt_required"] is True
    assert contract["freeze_chronology"].startswith("v1_deployment_freeze")
    assert protected["target_observation_session"] == "2026-07-31"
    assert protected["target_signal_session"] == "2026-08-03"
    assert protected["opening_leader_historical_outcomes_accessed"] is False
    assert protected["monday_2026_08_03_market_data_consumed_before_revision"] is False
    assert protected["causal_friday_d1_source_rows_accessed_for_v1_failure_diagnosis"] is True
    assert prior["failed_v1_attempt_receipt_identity_sha256"] == (
        "0005c52323e91f6ab23d93abd58b01e932d73cc97cff0c6e181bd206fc85d27b"
    )
    assert prior["failed_v1_start_receipt_identity_sha256"] == (
        "ebc3d8790eb8627e22cd3097a56f1781dc9ae946b8f6fa0d7c3c80e84ca17625"
    )
    assert prior["failed_v1_start_receipt_file_sha256"] == (
        "b3f740fac0ae6f7314e7f7a984d955953bcfcebfc3ee056fbdf597b98d5249db"
    )
    assert prior["canonical_option_rows"] == 0
    assert prior["revision_created"] is False
    assert prior["v1_attempt_reuse_allowed"] is False
    assert order_audit["runtime_mode_required"] == "record_only"
    assert order_audit["broker_order_methods_reachable"] is False
    assert order_audit["pre_adapter_recovery_command_uses_ibkr"] is False


def test_group_o_recovery_deployment_freeze_receipt_verifies() -> None:
    receipt = verify_group_o_recovery_freeze_v2(ROOT)

    assert receipt["target_observation_session"] == "2026-07-31"
    assert receipt["target_signal_session"] == "2026-08-03"
    assert receipt["order_placement_disabled"] is True
    assert receipt["protected_outcomes_accessed"] is False
    assert receipt["provider_dte_policy"] == "recompute_from_eod_identity"
    assert receipt["supersedes_recovery_deployment_receipt_sha256"] == (
        "557df7e4b043da54325c9453f4f0d2838d5d142189576fcf3c2a63feff8c3528"
    )
    assert receipt["failed_v1_start_receipt_file_sha256"] == (
        "b3f740fac0ae6f7314e7f7a984d955953bcfcebfc3ee056fbdf597b98d5249db"
    )
    assert receipt["failed_v1_start_receipt_identity_sha256"] == (
        "ebc3d8790eb8627e22cd3097a56f1781dc9ae946b8f6fa0d7c3c80e84ca17625"
    )
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


def test_v2_deployment_freeze_must_follow_the_signed_v1_failure(tmp_path: Path) -> None:
    release = _copy_recovery_release(tmp_path)
    receipt_path = (
        release
        / group_o_recovery.RECOVERY_PACKAGE_RELATIVE_V2
        / "deployment_freeze_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["freeze_completed_at_utc"] = "2026-08-02T16:00:00Z"
    receipt_path.write_text(
        json.dumps(_resign_deployment_receipt(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(GroupORecoveryIntegrityError, match="freeze chronology"):
        verify_group_o_recovery_freeze_v2(release)



def test_group_o_recovery_blocks_before_adapter_while_exact_chain_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        group_o_recovery,
        "verify_group_o_recovery_freeze_v2",
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
    monkeypatch.setattr(
        group_o_recovery,
        "_require_failed_v1_attempt",
        lambda _root, **_kwargs: tmp_path / "context" / "attempt_receipt.json",
    )

    with pytest.raises(
        GroupORecoveryIntegrityError,
        match="blocked_pre_adapter_group_o_recovery_incomplete",
    ):
        require_group_o_recovery_ready_before_adapter_v2(
            context_root=tmp_path / "context",
            release_directory=ROOT,
            now=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )


def test_v2_requires_exact_signed_failed_v1_start_and_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = (
        tmp_path
        / "source-cache/eodhd-group-o/2026-07-31/attempts/0001/attempt_receipt.json"
    )
    start_identity: dict[str, object] = {
        "schema_version": "m1c-group-o-recovery-start-v1",
        "recovery_version": "m1c-group-o-late-revision-v1",
        "deployment_receipt_id": "group-o-recovery-deploy-21e71f3460d0f63353cb5b2a",
        "deployment_receipt_sha256": (
            "557df7e4b043da54325c9453f4f0d2838d5d142189576fcf3c2a63feff8c3528"
        ),
        "attempt_id": "0001",
        "target_observation_session": "2026-07-31",
        "target_signal_session": "2026-08-03",
        "signal_open_utc": "2026-08-03T13:30:00+00:00",
        "started_at_utc": "2026-08-02T16:35:01.899928+00:00",
        "base_package_path": str(tmp_path / "group-o" / "2026-08-03.json"),
        "base_package_sha256": (
            "b8ba77cfcb6ff82c7fd1f205e5b2cbcd562ea2a09d7e3e1afa703fb0cc77d75c"
        ),
        "ibkr_adapter_opened": False,
        "monday_market_data_consumed": False,
        "order_construction_allowed": False,
        "order_placement_allowed": False,
        "status": "authorised_pre_signal_acquisition",
    }
    start_digest = hashlib.sha256(_canonical_json(start_identity).encode("utf-8")).hexdigest()
    start_payload = {
        **start_identity,
        "start_receipt_id": f"group-o-recovery-start-{start_digest[:24]}",
        "start_receipt_sha256": start_digest,
    }
    start = attempt.with_name("recovery_start_receipt.json")
    start.parent.mkdir(parents=True, exist_ok=True)
    start.write_text(json.dumps(start_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    start_bytes = start.read_bytes()
    write_group_o_attempt_receipt(
        attempt,
        {
            "schema_version": "group-o-acquisition-attempt-v1",
            "attempt_id": "0001",
            "signal_session": "2026-08-03",
            "observation_session": "2026-07-31",
            "started_at_utc": "2026-08-02T16:35:08.783707+00:00",
            "completed_at_utc": "2026-08-02T16:35:27.212704+00:00",
            "retry_after_utc": "2026-08-02T16:50:27.212704+00:00",
            "status": "pending_exact_chain",
            "symbol_count": 20,
            "canonical_option_rows": 0,
            "rejected_option_rows": 6364,
            "missing_exact_chain_symbols": list(CANONICAL_COHORT_V0),
        },
    )
    signed = load_group_o_attempt_receipt(attempt)
    monkeypatch.setattr(
        group_o_recovery,
        "FAILED_V1_ATTEMPT_RECEIPT_SHA256",
        hashlib.sha256(attempt.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        group_o_recovery,
        "FAILED_V1_ATTEMPT_IDENTITY_SHA256",
        signed["attempt_receipt_sha256"],
    )
    monkeypatch.setattr(
        group_o_recovery,
        "FAILED_V1_START_RECEIPT_FILE_SHA256",
        hashlib.sha256(start_bytes).hexdigest(),
        raising=False,
    )
    monkeypatch.setattr(
        group_o_recovery,
        "FAILED_V1_START_RECEIPT_IDENTITY_SHA256",
        start_digest,
        raising=False,
    )
    freeze = {
        "audited_failed_base_sha256": start_identity["base_package_sha256"],
        "freeze_completed_at_utc": "2026-08-02T16:49:09.064117+00:00",
    }

    assert (
        group_o_recovery._require_failed_v1_attempt(tmp_path, freeze_receipt=freeze)
        == attempt
    )

    start.write_text("{}\n", encoding="utf-8")
    with pytest.raises(GroupORecoveryIntegrityError, match="failed V1 start"):
        group_o_recovery._require_failed_v1_attempt(tmp_path, freeze_receipt=freeze)

    start.write_bytes(start_bytes)
    attempt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(GroupORecoveryIntegrityError, match="failed V1 attempt file"):
        group_o_recovery._require_failed_v1_attempt(tmp_path, freeze_receipt=freeze)


def test_v2_recovery_start_must_follow_its_signed_deployment_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_root = tmp_path / "context"
    base_path = context_root / "group-o" / "2026-08-03.json"
    base_path.parent.mkdir(parents=True)
    base_path.write_text("frozen-base\n", encoding="utf-8")
    release = tmp_path / "release"
    deployment_receipt = (
        release
        / group_o_recovery.RECOVERY_PACKAGE_RELATIVE_V2
        / "deployment_freeze_receipt.json"
    )
    deployment_receipt.parent.mkdir(parents=True)
    deployment_receipt.write_text("{}\n", encoding="utf-8")
    attempt_path = (
        context_root
        / "source-cache/eodhd-group-o/2026-07-31/attempts/0002"
    )
    attempt_path.mkdir(parents=True)
    freeze = {
        "deployment_receipt_id": "group-o-recovery-deploy-" + "e" * 24,
        "audited_failed_base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "freeze_completed_at_utc": "2026-08-02T12:01:00+00:00",
    }
    contexts = tuple(
        SimpleNamespace(symbol=symbol, quality_status="missing_exact_chain")
        for symbol in CANONICAL_COHORT_V0
    )
    monkeypatch.setattr(
        group_o_recovery,
        "verify_group_o_recovery_freeze_v2",
        lambda _release: freeze,
    )
    monkeypatch.setattr(
        group_o_recovery,
        "_require_audited_failed_base",
        lambda **_kwargs: base_path,
    )
    monkeypatch.setattr(
        group_o_recovery,
        "_require_failed_v1_attempt",
        lambda _root, **_kwargs: attempt_path.parent / "0001" / "attempt_receipt.json",
    )
    monkeypatch.setattr(
        group_o_recovery,
        "load_group_o_session_package",
        lambda **_kwargs: SimpleNamespace(contexts=contexts),
    )
    monkeypatch.setattr(group_o_recovery, "group_o_retry_not_before", lambda **_kwargs: None)
    monkeypatch.setattr(
        group_o_recovery,
        "allocate_group_o_attempt",
        lambda **_kwargs: ("0002", attempt_path),
    )
    monkeypatch.setattr(
        group_o_recovery,
        "acquire_eodhd_group_o_session_package",
        lambda **_kwargs: pytest.fail("EODHD request opened before the V2 freeze"),
    )

    with pytest.raises(GroupORecoveryIntegrityError, match="start must follow deployment freeze"):
        group_o_recovery.recover_group_o_exact_chain_v2(
            context_root=context_root,
            release_directory=release,
            symbols=CANONICAL_COHORT_V0,
            clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
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
    assert recorder.index("require_group_o_recovery_ready_before_adapter_v2(") < recorder.index(
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
            signal_session=group_o_recovery.TARGET_SIGNAL_SESSION_V2,
            observation_session=group_o_recovery.TARGET_OBSERVATION_SESSION_V2,
            attempt_id="0002",
            start_receipt_path=tmp_path / "0002" / "recovery_start_receipt.json",
            canonical_option_rows=40,
        )

    def sleep(seconds: float) -> None:
        nonlocal current
        sleeps.append(seconds)
        current += timedelta(seconds=seconds)

    monkeypatch.setattr(group_o_recovery, "recover_group_o_exact_chain_v2", recover_once)
    monkeypatch.setattr(
        group_o_recovery,
        "group_o_retry_not_before",
        lambda **_kwargs: datetime(2026, 8, 2, 12, 15, tzinfo=UTC),
    )

    result = group_o_recovery.recover_group_o_exact_chain_until_ready_v2(
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
