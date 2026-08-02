"""Pre-adapter recovery for the audited Friday 2026-07-31 Group O source lag."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stocker_prospective.append_only import write_immutable_json
from stocker_prospective.group_o import (
    FrozenGroupOSessionPackage,
    FrozenGroupOSessionRevision,
    load_group_o_session_package,
)
from stocker_prospective.live_bars import xnys_session_bounds
from stocker_prospective.opening_leader_continuation_v0 import CANONICAL_COHORT_V0
from stocker_prospective.scientific_inputs import (
    GroupOAcquisitionPending,
    acquire_eodhd_group_o_session_package,
    allocate_group_o_attempt,
    group_o_retry_not_before,
    load_group_o_attempt_receipt,
    write_group_o_attempt_receipt,
)

RECOVERY_VERSION_V1: Final[str] = "m1c-group-o-late-revision-v1"
TARGET_OBSERVATION_SESSION_V1: Final[date] = date(2026, 7, 31)
TARGET_SIGNAL_SESSION_V1: Final[date] = date(2026, 8, 3)
RECOVERY_PACKAGE_RELATIVE_V1: Final[Path] = Path(
    "prospective/m1c-group-o-recovery/20260802-m1c-group-o-late-revision-v1"
)
RECOVERY_ARTIFACTS_V1: Final[tuple[str, ...]] = (
    "README.md",
    "contract.json",
    "order_disable_audit.json",
    "protected_boundary_audit.json",
)
RECOVERY_SOURCE_FILES_V1: Final[Mapping[str, str]] = {
    "append_only": (
        "packages/stocker_prospective/src/stocker_prospective/append_only.py"
    ),
    "cli": "packages/stocker_prospective/src/stocker_prospective/cli.py",
    "context": "packages/stocker_prospective/src/stocker_prospective/context.py",
    "eodhd_client": "packages/stocker_data/src/stocker_data/vendors/eodhd.py",
    "eodhd_config": "packages/stocker_core/src/stocker_core/config.py",
    "eodhd_options_downloader": (
        "packages/stocker_research/src/stocker_research/eodhd_options_downloader_v0.py"
    ),
    "front_option_features": (
        "packages/stocker_research/src/stocker_research/"
        "broad_conflict_options_iv_screen_v0.py"
    ),
    "front_option_regimes": (
        "packages/stocker_research/src/stocker_research/front_options_soft_regimes_v01.py"
    ),
    "group_o": "packages/stocker_prospective/src/stocker_prospective/group_o.py",
    "group_o_recovery": (
        "packages/stocker_prospective/src/stocker_prospective/group_o_recovery.py"
    ),
    "live_bars": "packages/stocker_prospective/src/stocker_prospective/live_bars.py",
    "opening_leader_cohort_contract": (
        "packages/stocker_prospective/src/stocker_prospective/"
        "opening_leader_continuation_v0.py"
    ),
    "option_context_features": (
        "packages/stocker_research/src/stocker_research/daily_stock_options_context_v0.py"
    ),
    "option_dimension_parameters": (
        "packages/stocker_research/src/stocker_research/daily_soft_regimes_v0.py"
    ),
    "scientific_inputs": (
        "packages/stocker_prospective/src/stocker_prospective/scientific_inputs.py"
    ),
    "project_configuration": "pyproject.toml",
    "resolved_dependencies": "uv.lock",
}
RECOVERY_VERIFICATION_KEYS_V1: Final[frozenset[str]] = frozenset(
    {
        "review_findings_addressed",
        "scientific_input_tests",
        "scoped_lint",
        "scoped_type_check",
        "static_order_surface_audit",
    }
)


class GroupORecoveryIntegrityError(RuntimeError):
    """The recovery freeze or pre-adapter evidence does not verify."""


class GroupORecoveryRetryNotDue(RuntimeError):
    """The signed 15-minute source retry boundary has not arrived."""

    def __init__(self, retry_after_utc: datetime) -> None:
        super().__init__(f"Group O recovery retry is deferred until {retry_after_utc.isoformat()}")
        self.retry_after_utc = retry_after_utc


@dataclass(frozen=True)
class GroupORecoveryResult:
    status: str
    signal_session: date
    observation_session: date
    attempt_id: str | None
    start_receipt_path: Path | None
    canonical_option_rows: int | None


class GroupORecoveryStartReceiptV1(BaseModel):
    """Self-binding evidence written before the target EODHD request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["m1c-group-o-recovery-start-v1"]
    recovery_version: Literal["m1c-group-o-late-revision-v1"]
    deployment_receipt_id: str = Field(pattern=r"^group-o-recovery-deploy-[a-f0-9]{24}$")
    deployment_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    attempt_id: str = Field(pattern=r"^[0-9]{4}$")
    target_observation_session: date
    target_signal_session: date
    signal_open_utc: datetime
    started_at_utc: datetime
    base_package_path: str
    base_package_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    ibkr_adapter_opened: Literal[False]
    monday_market_data_consumed: Literal[False]
    order_construction_allowed: Literal[False]
    order_placement_allowed: Literal[False]
    status: Literal["authorised_pre_signal_acquisition"]
    start_receipt_id: str = Field(pattern=r"^group-o-recovery-start-[a-f0-9]{24}$")
    start_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("signal_open_utc", "started_at_utc")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Group O recovery timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _self_binding_causal_identity(self) -> GroupORecoveryStartReceiptV1:
        expected_open, _ = xnys_session_bounds(TARGET_SIGNAL_SESSION_V1)
        if (
            self.target_observation_session != TARGET_OBSERVATION_SESSION_V1
            or self.target_signal_session != TARGET_SIGNAL_SESSION_V1
            or self.signal_open_utc != expected_open
        ):
            raise ValueError("Group O recovery start session identity differs")
        if self.started_at_utc >= self.signal_open_utc:
            raise ValueError("Group O recovery start must precede signal open")
        identity = {
            "schema_version": self.schema_version,
            "recovery_version": self.recovery_version,
            "deployment_receipt_id": self.deployment_receipt_id,
            "deployment_receipt_sha256": self.deployment_receipt_sha256,
            "attempt_id": self.attempt_id,
            "target_observation_session": self.target_observation_session.isoformat(),
            "target_signal_session": self.target_signal_session.isoformat(),
            "signal_open_utc": self.signal_open_utc.isoformat(),
            "started_at_utc": self.started_at_utc.isoformat(),
            "base_package_path": self.base_package_path,
            "base_package_sha256": self.base_package_sha256,
            "ibkr_adapter_opened": self.ibkr_adapter_opened,
            "monday_market_data_consumed": self.monday_market_data_consumed,
            "order_construction_allowed": self.order_construction_allowed,
            "order_placement_allowed": self.order_placement_allowed,
            "status": self.status,
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        if self.start_receipt_sha256 != digest:
            raise ValueError("Group O recovery start receipt hash differs")
        if self.start_receipt_id != f"group-o-recovery-start-{digest[:24]}":
            raise ValueError("Group O recovery start receipt ID differs")
        return self


class GroupORecoveryCompletionReceiptV1(BaseModel):
    """Fully linked proof that the pre-adapter revision completed immutably."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["m1c-group-o-recovery-completion-v1"]
    recovery_version: Literal["m1c-group-o-late-revision-v1"]
    status: Literal["published_revision_verified", "published_revision_reconciled"]
    attempt_id: str = Field(pattern=r"^[0-9]{4}$")
    target_observation_session: date
    target_signal_session: date
    signal_open_utc: datetime
    published_at_utc: datetime
    completed_at_utc: datetime
    deployment_receipt_id: str = Field(pattern=r"^group-o-recovery-deploy-[a-f0-9]{24}$")
    deployment_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    start_receipt_path: str
    start_receipt_id: str = Field(pattern=r"^group-o-recovery-start-[a-f0-9]{24}$")
    start_receipt_identity_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    start_receipt_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    base_package_path: str
    base_package_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_package_path: str
    candidate_package_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_package_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    revision_path: str
    revision_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    revision_id: str = Field(pattern=r"^group-o-revision-[a-f0-9]{24}$")
    revision_identity_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    acquisition_attempt_receipt_path: str
    acquisition_attempt_receipt_file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    acquisition_attempt_receipt_identity_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    order_construction_allowed: Literal[False]
    order_placement_allowed: Literal[False]
    completion_receipt_id: str = Field(pattern=r"^group-o-recovery-complete-[a-f0-9]{24}$")
    completion_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("signal_open_utc", "published_at_utc", "completed_at_utc")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Group O recovery completion timestamps must be timezone-aware")
        return value.astimezone(UTC)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "recovery_version": self.recovery_version,
            "status": self.status,
            "attempt_id": self.attempt_id,
            "target_observation_session": self.target_observation_session.isoformat(),
            "target_signal_session": self.target_signal_session.isoformat(),
            "signal_open_utc": self.signal_open_utc.isoformat(),
            "published_at_utc": self.published_at_utc.isoformat(),
            "completed_at_utc": self.completed_at_utc.isoformat(),
            "deployment_receipt_id": self.deployment_receipt_id,
            "deployment_receipt_sha256": self.deployment_receipt_sha256,
            "start_receipt_path": self.start_receipt_path,
            "start_receipt_id": self.start_receipt_id,
            "start_receipt_identity_sha256": self.start_receipt_identity_sha256,
            "start_receipt_file_sha256": self.start_receipt_file_sha256,
            "base_package_path": self.base_package_path,
            "base_package_sha256": self.base_package_sha256,
            "candidate_package_path": self.candidate_package_path,
            "candidate_package_file_sha256": self.candidate_package_file_sha256,
            "candidate_package_hash": self.candidate_package_hash,
            "revision_path": self.revision_path,
            "revision_file_sha256": self.revision_file_sha256,
            "revision_id": self.revision_id,
            "revision_identity_sha256": self.revision_identity_sha256,
            "acquisition_attempt_receipt_path": self.acquisition_attempt_receipt_path,
            "acquisition_attempt_receipt_file_sha256": (
                self.acquisition_attempt_receipt_file_sha256
            ),
            "acquisition_attempt_receipt_identity_sha256": (
                self.acquisition_attempt_receipt_identity_sha256
            ),
            "order_construction_allowed": self.order_construction_allowed,
            "order_placement_allowed": self.order_placement_allowed,
        }

    @model_validator(mode="after")
    def _self_binding_causal_identity(self) -> GroupORecoveryCompletionReceiptV1:
        expected_open, _ = xnys_session_bounds(TARGET_SIGNAL_SESSION_V1)
        if (
            self.target_observation_session != TARGET_OBSERVATION_SESSION_V1
            or self.target_signal_session != TARGET_SIGNAL_SESSION_V1
            or self.signal_open_utc != expected_open
            or self.published_at_utc >= self.signal_open_utc
            or self.completed_at_utc < self.published_at_utc
        ):
            raise ValueError("Group O recovery completion chronology differs")
        digest = hashlib.sha256(
            _canonical_json(self.identity_payload()).encode("utf-8")
        ).hexdigest()
        if self.completion_receipt_sha256 != digest:
            raise ValueError("Group O recovery completion receipt hash differs")
        if self.completion_receipt_id != f"group-o-recovery-complete-{digest[:24]}":
            raise ValueError("Group O recovery completion receipt ID differs")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_group_o_recovery_freeze_v1(release_directory: str | Path) -> dict[str, Any]:
    """Verify every signed artifact and source before any EODHD request."""

    release = Path(release_directory)
    package_root = release / RECOVERY_PACKAGE_RELATIVE_V1
    receipt_path = package_root / "deployment_freeze_receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise GroupORecoveryIntegrityError("missing Group O recovery deployment freeze receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GroupORecoveryIntegrityError(
            "invalid Group O recovery deployment freeze receipt"
        ) from exc
    if not isinstance(receipt, dict):
        raise GroupORecoveryIntegrityError("invalid Group O recovery deployment freeze receipt")
    required_identity = {
        "schema_version": "m1c-group-o-recovery-deployment-freeze-v1",
        "recovery_version": RECOVERY_VERSION_V1,
        "target_observation_session": TARGET_OBSERVATION_SESSION_V1.isoformat(),
        "target_signal_session": TARGET_SIGNAL_SESSION_V1.isoformat(),
        "order_placement_disabled": True,
        "protected_outcomes_accessed": False,
        "source_hashes_signed": True,
    }
    if any(receipt.get(key) != value for key, value in required_identity.items()):
        raise GroupORecoveryIntegrityError("Group O recovery freeze identity differs")
    audited_base_hash = receipt.get("audited_failed_base_sha256")
    if (
        not isinstance(audited_base_hash, str)
        or len(audited_base_hash) != 64
        or any(character not in "0123456789abcdef" for character in audited_base_hash)
    ):
        raise GroupORecoveryIntegrityError("Group O recovery audited base hash is invalid")
    artifact_hashes = receipt.get("artifact_hashes")
    source_hashes = receipt.get("source_hashes")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(
        RECOVERY_ARTIFACTS_V1
    ):
        raise GroupORecoveryIntegrityError("Group O recovery artifact set differs")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(
        RECOVERY_SOURCE_FILES_V1
    ):
        raise GroupORecoveryIntegrityError("Group O recovery source set differs")
    for name in RECOVERY_ARTIFACTS_V1:
        artifact = package_root / name
        if not artifact.is_file() or artifact.is_symlink():
            raise GroupORecoveryIntegrityError(f"Group O recovery artifact is invalid: {name}")
        if artifact_hashes[name] != _sha256_path(artifact):
            raise GroupORecoveryIntegrityError(f"Group O recovery artifact hash differs: {name}")
    for name, relative_path in RECOVERY_SOURCE_FILES_V1.items():
        source = release / relative_path
        if not source.is_file() or source.is_symlink():
            raise GroupORecoveryIntegrityError(f"Group O recovery source is invalid: {name}")
        if source_hashes[name] != _sha256_path(source):
            raise GroupORecoveryIntegrityError(f"Group O recovery source hash differs: {name}")
    expected_code_hash = hashlib.sha256(
        _canonical_json(source_hashes).encode("utf-8")
    ).hexdigest()
    if receipt.get("code_hash") != expected_code_hash:
        raise GroupORecoveryIntegrityError("Group O recovery aggregate code hash differs")
    if receipt.get("contract_hash") != artifact_hashes["contract.json"]:
        raise GroupORecoveryIntegrityError("Group O recovery contract hash differs")
    unsigned = dict(receipt)
    signature = unsigned.pop("signature_sha256", None)
    receipt_identity = dict(unsigned)
    receipt_identity.pop("deployment_receipt_id", None)
    identity_hash = hashlib.sha256(_canonical_json(receipt_identity).encode("utf-8")).hexdigest()
    expected_signature = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    if signature != expected_signature:
        raise GroupORecoveryIntegrityError("Group O recovery freeze signature differs")
    expected_receipt_id = f"group-o-recovery-deploy-{identity_hash[:24]}"
    if receipt.get("deployment_receipt_id") != expected_receipt_id:
        raise GroupORecoveryIntegrityError("Group O recovery deployment receipt ID differs")
    verification = receipt.get("verification")
    if (
        not isinstance(verification, dict)
        or set(verification) != RECOVERY_VERIFICATION_KEYS_V1
        or any(value != "passed" for value in verification.values())
    ):
        raise GroupORecoveryIntegrityError("Group O recovery verification is incomplete")
    return receipt


def _target_base_path(context_root: Path) -> Path:
    return context_root / "group-o" / f"{TARGET_SIGNAL_SESSION_V1.isoformat()}.json"


def _require_audited_failed_base(
    *,
    context_root: Path,
    freeze_receipt: Mapping[str, Any],
) -> Path:
    base_path = _target_base_path(context_root)
    if not base_path.is_file() or base_path.is_symlink():
        raise GroupORecoveryIntegrityError("Group O recovery immutable base is unavailable")
    expected_hash = freeze_receipt.get("audited_failed_base_sha256")
    if _sha256_path(base_path) != expected_hash:
        raise GroupORecoveryIntegrityError("Group O recovery audited failed base hash differs")
    try:
        package = FrozenGroupOSessionPackage.model_validate_json(
            base_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise GroupORecoveryIntegrityError("Group O recovery audited base is invalid") from exc
    if (
        package.signal_session != TARGET_SIGNAL_SESSION_V1
        or tuple(context.symbol for context in package.contexts) != CANONICAL_COHORT_V0
        or not any(
            context.quality_status == "missing_exact_chain" for context in package.contexts
        )
    ):
        raise GroupORecoveryIntegrityError("Group O recovery audited base identity differs")
    return base_path


def _load_recovery_start_receipt(
    *,
    path: Path,
    context_root: Path,
    release_directory: Path,
    freeze_receipt: Mapping[str, Any],
) -> GroupORecoveryStartReceiptV1:
    if not path.is_file() or path.is_symlink():
        raise GroupORecoveryIntegrityError("Group O recovery start receipt is invalid")
    try:
        receipt = GroupORecoveryStartReceiptV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise GroupORecoveryIntegrityError("Group O recovery start receipt is invalid") from exc
    base_path = _require_audited_failed_base(
        context_root=context_root,
        freeze_receipt=freeze_receipt,
    )
    deployment_path = (
        release_directory / RECOVERY_PACKAGE_RELATIVE_V1 / "deployment_freeze_receipt.json"
    )
    if (
        path.parent.name != receipt.attempt_id
        or receipt.deployment_receipt_id != freeze_receipt.get("deployment_receipt_id")
        or receipt.deployment_receipt_sha256 != _sha256_path(deployment_path)
        or receipt.base_package_path != str(base_path)
        or receipt.base_package_sha256 != freeze_receipt.get("audited_failed_base_sha256")
    ):
        raise GroupORecoveryIntegrityError("Group O recovery start receipt linkage differs")
    return receipt


def _load_target_revision(context_root: Path) -> tuple[Path, FrozenGroupOSessionRevision]:
    revision_path = (
        context_root
        / "group-o"
        / "revisions"
        / TARGET_SIGNAL_SESSION_V1.isoformat()
        / "0001.json"
    )
    if not revision_path.is_file() or revision_path.is_symlink():
        raise GroupORecoveryIntegrityError("blocked_missing_group_o_recovery_revision")
    try:
        revision = FrozenGroupOSessionRevision.model_validate_json(
            revision_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise GroupORecoveryIntegrityError("invalid Group O recovery revision") from exc
    return revision_path, revision


def _package_hash(package: FrozenGroupOSessionPackage) -> str:
    return hashlib.sha256(
        _canonical_json(package.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def _validate_acquisition_attempt_receipt(
    *,
    path: Path,
    start: GroupORecoveryStartReceiptV1,
    revision: FrozenGroupOSessionRevision,
    base_path: Path,
) -> dict[str, object]:
    try:
        completed = load_group_o_attempt_receipt(path)
    except ValueError as exc:
        raise GroupORecoveryIntegrityError(
            "Group O recovery completion receipt is invalid"
        ) from exc
    if (
        completed.get("attempt_id") != start.attempt_id
        or completed.get("signal_session") != TARGET_SIGNAL_SESSION_V1.isoformat()
        or completed.get("observation_session") != TARGET_OBSERVATION_SESSION_V1.isoformat()
        or completed.get("status")
        not in {"published_revision", "published_revision_reconciled_after_restart"}
        or completed.get("published_revision_id") != revision.revision_id
        or completed.get("published_base_path") != str(base_path)
    ):
        raise GroupORecoveryIntegrityError("Group O recovery completion linkage differs")
    raw_published = completed.get("published_at_utc")
    if not isinstance(raw_published, str):
        raise GroupORecoveryIntegrityError("Group O recovery completion timestamp is missing")
    published = datetime.fromisoformat(raw_published)
    if published.tzinfo is None or published.utcoffset() is None:
        raise GroupORecoveryIntegrityError("Group O recovery completion timestamp is invalid")
    signal_open, _ = xnys_session_bounds(TARGET_SIGNAL_SESSION_V1)
    if published.astimezone(UTC) >= signal_open:
        raise GroupORecoveryIntegrityError("Group O recovery completion crossed signal open")
    return completed


def _validate_linked_recovery_completion_receipt(
    *,
    path: Path,
    release_directory: Path,
    start_path: Path,
    start: GroupORecoveryStartReceiptV1,
    base_path: Path,
    candidate_path: Path,
    candidate: FrozenGroupOSessionPackage,
    revision_path: Path,
    revision: FrozenGroupOSessionRevision,
    attempt_receipt_path: Path,
    attempt_receipt: Mapping[str, object],
) -> GroupORecoveryCompletionReceiptV1:
    if not path.is_file() or path.is_symlink():
        raise GroupORecoveryIntegrityError("Group O linked completion receipt is invalid")
    try:
        receipt = GroupORecoveryCompletionReceiptV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise GroupORecoveryIntegrityError(
            "Group O linked completion receipt is invalid"
        ) from exc
    deployment_path = (
        release_directory / RECOVERY_PACKAGE_RELATIVE_V1 / "deployment_freeze_receipt.json"
    )
    expected_status = (
        "published_revision_reconciled"
        if attempt_receipt.get("status") == "published_revision_reconciled_after_restart"
        else "published_revision_verified"
    )
    expected = {
        "status": expected_status,
        "attempt_id": start.attempt_id,
        "deployment_receipt_id": start.deployment_receipt_id,
        "deployment_receipt_sha256": _sha256_path(deployment_path),
        "start_receipt_path": str(start_path),
        "start_receipt_id": start.start_receipt_id,
        "start_receipt_identity_sha256": start.start_receipt_sha256,
        "start_receipt_file_sha256": _sha256_path(start_path),
        "base_package_path": str(base_path),
        "base_package_sha256": _sha256_path(base_path),
        "candidate_package_path": str(candidate_path),
        "candidate_package_file_sha256": _sha256_path(candidate_path),
        "candidate_package_hash": _package_hash(candidate),
        "revision_path": str(revision_path),
        "revision_file_sha256": _sha256_path(revision_path),
        "revision_id": revision.revision_id,
        "revision_identity_sha256": revision.revision_sha256,
        "acquisition_attempt_receipt_path": str(attempt_receipt_path),
        "acquisition_attempt_receipt_file_sha256": _sha256_path(attempt_receipt_path),
        "acquisition_attempt_receipt_identity_sha256": attempt_receipt.get(
            "attempt_receipt_sha256"
        ),
    }
    if any(getattr(receipt, key) != value for key, value in expected.items()):
        raise GroupORecoveryIntegrityError("Group O linked completion receipt differs")
    if (
        start.started_at_utc > receipt.published_at_utc
        or receipt.published_at_utc != revision.created_at_utc
        or revision.supersedes_sha256 != receipt.base_package_sha256
        or revision.revised_package_hash != receipt.candidate_package_hash
    ):
        raise GroupORecoveryIntegrityError("Group O linked completion chronology differs")
    return receipt


def _write_or_validate_linked_completion_receipt(
    *,
    attempt_path: Path,
    release_directory: Path,
    start_path: Path,
    start: GroupORecoveryStartReceiptV1,
    base_path: Path,
    candidate_path: Path,
    candidate: FrozenGroupOSessionPackage,
    revision_path: Path,
    revision: FrozenGroupOSessionRevision,
    attempt_receipt_path: Path,
    attempt_receipt: Mapping[str, object],
    completed_at_utc: datetime,
) -> GroupORecoveryCompletionReceiptV1:
    destination = attempt_path / "recovery_completion_receipt.json"
    if not destination.exists():
        signal_open, _ = xnys_session_bounds(TARGET_SIGNAL_SESSION_V1)
        status = (
            "published_revision_reconciled"
            if attempt_receipt.get("status")
            == "published_revision_reconciled_after_restart"
            else "published_revision_verified"
        )
        identity: dict[str, object] = {
            "schema_version": "m1c-group-o-recovery-completion-v1",
            "recovery_version": RECOVERY_VERSION_V1,
            "status": status,
            "attempt_id": start.attempt_id,
            "target_observation_session": TARGET_OBSERVATION_SESSION_V1.isoformat(),
            "target_signal_session": TARGET_SIGNAL_SESSION_V1.isoformat(),
            "signal_open_utc": signal_open.isoformat(),
            "published_at_utc": revision.created_at_utc.isoformat(),
            "completed_at_utc": completed_at_utc.astimezone(UTC).isoformat(),
            "deployment_receipt_id": start.deployment_receipt_id,
            "deployment_receipt_sha256": start.deployment_receipt_sha256,
            "start_receipt_path": str(start_path),
            "start_receipt_id": start.start_receipt_id,
            "start_receipt_identity_sha256": start.start_receipt_sha256,
            "start_receipt_file_sha256": _sha256_path(start_path),
            "base_package_path": str(base_path),
            "base_package_sha256": _sha256_path(base_path),
            "candidate_package_path": str(candidate_path),
            "candidate_package_file_sha256": _sha256_path(candidate_path),
            "candidate_package_hash": _package_hash(candidate),
            "revision_path": str(revision_path),
            "revision_file_sha256": _sha256_path(revision_path),
            "revision_id": revision.revision_id,
            "revision_identity_sha256": revision.revision_sha256,
            "acquisition_attempt_receipt_path": str(attempt_receipt_path),
            "acquisition_attempt_receipt_file_sha256": _sha256_path(attempt_receipt_path),
            "acquisition_attempt_receipt_identity_sha256": attempt_receipt[
                "attempt_receipt_sha256"
            ],
            "order_construction_allowed": False,
            "order_placement_allowed": False,
        }
        digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        payload = {
            **identity,
            "completion_receipt_id": f"group-o-recovery-complete-{digest[:24]}",
            "completion_receipt_sha256": digest,
        }
        try:
            validated = GroupORecoveryCompletionReceiptV1.model_validate(payload)
        except Exception as exc:
            raise GroupORecoveryIntegrityError(
                "Group O linked completion receipt is invalid"
            ) from exc
        write_immutable_json(
            destination,
            validated.model_dump(mode="json"),
            conflict_message="immutable Group O linked completion receipt differs",
        )
    return _validate_linked_recovery_completion_receipt(
        path=destination,
        release_directory=release_directory,
        start_path=start_path,
        start=start,
        base_path=base_path,
        candidate_path=candidate_path,
        candidate=candidate,
        revision_path=revision_path,
        revision=revision,
        attempt_receipt_path=attempt_receipt_path,
        attempt_receipt=attempt_receipt,
    )


def _write_recovery_start_receipt(
    *,
    attempt_path: Path,
    attempt_id: str,
    context_root: Path,
    release_directory: Path,
    freeze_receipt: Mapping[str, Any],
    started_at_utc: datetime,
) -> Path:
    signal_open, _ = xnys_session_bounds(TARGET_SIGNAL_SESSION_V1)
    if started_at_utc.tzinfo is None or started_at_utc.utcoffset() is None:
        raise GroupORecoveryIntegrityError("Group O recovery start timestamp is invalid")
    started_at_utc = started_at_utc.astimezone(UTC)
    if started_at_utc >= signal_open:
        raise GroupORecoveryIntegrityError("Group O recovery start must precede Monday open")
    expected_attempt = (
        context_root
        / "source-cache"
        / "eodhd-group-o"
        / TARGET_OBSERVATION_SESSION_V1.isoformat()
        / "attempts"
        / attempt_id
    )
    if attempt_path != expected_attempt or not attempt_path.is_dir() or attempt_path.is_symlink():
        raise GroupORecoveryIntegrityError("Group O recovery attempt identity differs")
    base_path = _require_audited_failed_base(
        context_root=context_root,
        freeze_receipt=freeze_receipt,
    )
    package_receipt = (
        release_directory / RECOVERY_PACKAGE_RELATIVE_V1 / "deployment_freeze_receipt.json"
    )
    if not package_receipt.is_file() or package_receipt.is_symlink():
        raise GroupORecoveryIntegrityError("Group O recovery deployment receipt is invalid")
    identity: dict[str, object] = {
        "schema_version": "m1c-group-o-recovery-start-v1",
        "recovery_version": RECOVERY_VERSION_V1,
        "deployment_receipt_id": str(freeze_receipt["deployment_receipt_id"]),
        "deployment_receipt_sha256": _sha256_path(package_receipt),
        "attempt_id": attempt_id,
        "target_observation_session": TARGET_OBSERVATION_SESSION_V1.isoformat(),
        "target_signal_session": TARGET_SIGNAL_SESSION_V1.isoformat(),
        "signal_open_utc": signal_open.isoformat(),
        "started_at_utc": started_at_utc.isoformat(),
        "base_package_path": str(base_path),
        "base_package_sha256": _sha256_path(base_path),
        "ibkr_adapter_opened": False,
        "monday_market_data_consumed": False,
        "order_construction_allowed": False,
        "order_placement_allowed": False,
        "status": "authorised_pre_signal_acquisition",
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    receipt = {
        **identity,
        "start_receipt_id": f"group-o-recovery-start-{digest[:24]}",
        "start_receipt_sha256": digest,
    }
    try:
        validated = GroupORecoveryStartReceiptV1.model_validate(receipt)
    except Exception as exc:
        raise GroupORecoveryIntegrityError("Group O recovery start receipt is invalid") from exc
    destination = attempt_path / "recovery_start_receipt.json"
    write_immutable_json(
        destination,
        validated.model_dump(mode="json"),
        conflict_message="immutable Group O recovery start receipt differs",
    )
    return destination


def reconcile_group_o_recovery_completion_v1(
    *,
    context_root: str | Path,
    release_directory: str | Path,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> bool:
    """Rebuild only the signed completion receipt after a post-link process crash."""

    root = Path(context_root)
    release = Path(release_directory)
    freeze = verify_group_o_recovery_freeze_v1(release)
    base_path = _require_audited_failed_base(
        context_root=root,
        freeze_receipt=freeze,
    )
    try:
        revision_path, revision = _load_target_revision(root)
    except GroupORecoveryIntegrityError as exc:
        if str(exc) == "blocked_missing_group_o_recovery_revision":
            return False
        raise
    if (
        revision.signal_session != TARGET_SIGNAL_SESSION_V1
        or revision.supersedes_sha256 != _sha256_path(base_path)
        or tuple(context.symbol for context in revision.package.contexts)
        != CANONICAL_COHORT_V0
    ):
        raise GroupORecoveryIntegrityError("Group O recovery revision linkage differs")
    attempts_root = (
        root
        / "source-cache"
        / "eodhd-group-o"
        / TARGET_OBSERVATION_SESSION_V1.isoformat()
        / "attempts"
    )
    if not attempts_root.is_dir() or attempts_root.is_symlink():
        return False
    for attempt in sorted(attempts_root.iterdir(), key=lambda path: path.name):
        if not attempt.is_dir() or attempt.is_symlink() or not attempt.name.isdigit():
            continue
        start_path = attempt / "recovery_start_receipt.json"
        if not start_path.exists():
            continue
        start = _load_recovery_start_receipt(
            path=start_path,
            context_root=root,
            release_directory=release,
            freeze_receipt=freeze,
        )
        candidate_path = attempt / "revision_candidate_package.json"
        if not candidate_path.exists():
            candidate_path = attempt / "candidate_package.json"
        if not candidate_path.is_file() or candidate_path.is_symlink():
            continue
        try:
            candidate = FrozenGroupOSessionPackage.model_validate_json(
                candidate_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise GroupORecoveryIntegrityError(
                "Group O recovery revision candidate is invalid"
            ) from exc
        if _package_hash(candidate) != revision.revised_package_hash:
            continue
        completion_path = attempt / "attempt_receipt.json"
        if completion_path.exists():
            completed = _validate_acquisition_attempt_receipt(
                path=completion_path,
                start=start,
                revision=revision,
                base_path=base_path,
            )
            observed = clock()
            if observed.tzinfo is None or observed.utcoffset() is None:
                raise GroupORecoveryIntegrityError(
                    "Group O recovery completion timestamp is invalid"
                )
            _write_or_validate_linked_completion_receipt(
                attempt_path=attempt,
                release_directory=release,
                start_path=start_path,
                start=start,
                base_path=base_path,
                candidate_path=candidate_path,
                candidate=candidate,
                revision_path=revision_path,
                revision=revision,
                attempt_receipt_path=completion_path,
                attempt_receipt=completed,
                completed_at_utc=observed.astimezone(UTC),
            )
            return True
        reconciled_at = clock()
        if reconciled_at.tzinfo is None or reconciled_at.utcoffset() is None:
            raise GroupORecoveryIntegrityError(
                "Group O recovery reconciliation timestamp is invalid"
            )
        write_group_o_attempt_receipt(
            completion_path,
            {
                "schema_version": "group-o-acquisition-attempt-v1",
                "attempt_id": start.attempt_id,
                "signal_session": TARGET_SIGNAL_SESSION_V1.isoformat(),
                "observation_session": TARGET_OBSERVATION_SESSION_V1.isoformat(),
                "started_at_utc": start.started_at_utc.isoformat(),
                "completed_at_utc": revision.created_at_utc.isoformat(),
                "reconciled_at_utc": reconciled_at.astimezone(UTC).isoformat(),
                "status": "published_revision_reconciled_after_restart",
                "published_at_utc": revision.created_at_utc.isoformat(),
                "published_base_path": str(base_path),
                "published_revision_id": revision.revision_id,
                "reconciliation_basis": {
                    "start_receipt_sha256": start.start_receipt_sha256,
                    "candidate_package_sha256": _sha256_path(candidate_path),
                    "revision_sha256": revision.revision_sha256,
                },
            },
        )
        completed = _validate_acquisition_attempt_receipt(
            path=completion_path,
            start=start,
            revision=revision,
            base_path=base_path,
        )
        _write_or_validate_linked_completion_receipt(
            attempt_path=attempt,
            release_directory=release,
            start_path=start_path,
            start=start,
            base_path=base_path,
            candidate_path=candidate_path,
            candidate=candidate,
            revision_path=revision_path,
            revision=revision,
            attempt_receipt_path=completion_path,
            attempt_receipt=completed,
            completed_at_utc=reconciled_at.astimezone(UTC),
        )
        return True
    return False


def recover_group_o_exact_chain_v1(
    *,
    context_root: str | Path,
    release_directory: str | Path,
    symbols: tuple[str, ...],
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> GroupORecoveryResult:
    """Acquire Friday's exact chain before opening any IBKR market-data adapter."""

    if symbols != CANONICAL_COHORT_V0:
        raise GroupORecoveryIntegrityError("Group O recovery canonical cohort differs")
    root = Path(context_root)
    release = Path(release_directory)
    freeze_receipt = verify_group_o_recovery_freeze_v1(release)
    base_path = _require_audited_failed_base(
        context_root=root,
        freeze_receipt=freeze_receipt,
    )
    resolved = load_group_o_session_package(
        context_root=root,
        signal_session=TARGET_SIGNAL_SESSION_V1,
    )
    if tuple(context.symbol for context in resolved.contexts) != CANONICAL_COHORT_V0:
        raise GroupORecoveryIntegrityError("Group O recovery base cohort differs")
    if not any(context.quality_status == "missing_exact_chain" for context in resolved.contexts):
        if not reconcile_group_o_recovery_completion_v1(
            context_root=root,
            release_directory=release,
            clock=clock,
        ):
            raise GroupORecoveryIntegrityError(
                "Group O recovery revision lacks linked completion evidence"
            )
        return GroupORecoveryResult(
            status="already_recovered",
            signal_session=TARGET_SIGNAL_SESSION_V1,
            observation_session=TARGET_OBSERVATION_SESSION_V1,
            attempt_id=None,
            start_receipt_path=None,
            canonical_option_rows=None,
        )
    started = clock().astimezone(UTC)
    signal_open, _ = xnys_session_bounds(TARGET_SIGNAL_SESSION_V1)
    if started >= signal_open:
        raise GroupORecoveryIntegrityError("Group O recovery cutoff passed before acquisition")
    cache_root = root / "source-cache" / "eodhd-group-o"
    retry_after = group_o_retry_not_before(
        cache_root=cache_root,
        observation_session=TARGET_OBSERVATION_SESSION_V1,
    )
    if retry_after is not None and started < retry_after:
        raise GroupORecoveryRetryNotDue(retry_after)
    attempt_id, attempt_path = allocate_group_o_attempt(
        cache_root=cache_root,
        observation_session=TARGET_OBSERVATION_SESSION_V1,
    )
    start_receipt = _write_recovery_start_receipt(
        attempt_path=attempt_path,
        attempt_id=attempt_id,
        context_root=root,
        release_directory=release,
        freeze_receipt=freeze_receipt,
        started_at_utc=started,
    )
    artifacts = (
        release
        / "research"
        / "cross-market-context"
        / "20260723-daily-stock-front-options-context-v01"
        / "artifacts"
        / "primary"
    )
    result = acquire_eodhd_group_o_session_package(
        signal_session=TARGET_SIGNAL_SESSION_V1,
        symbols=symbols,
        output_path=base_path,
        cache_root=cache_root,
        cache_attempt_id=attempt_id,
        feature_manifest_path=artifacts / "front_options_feature_manifest.json",
        regime_mapping_path=artifacts / "front_options_regime_mapping.json",
        supersedes_path=base_path,
        clock=clock,
    )
    revised = load_group_o_session_package(
        context_root=root,
        signal_session=TARGET_SIGNAL_SESSION_V1,
    )
    if any(context.quality_status == "missing_exact_chain" for context in revised.contexts):
        raise GroupORecoveryIntegrityError("Group O recovery did not resolve the exact chain")
    if not reconcile_group_o_recovery_completion_v1(
        context_root=root,
        release_directory=release,
        clock=clock,
    ):
        raise GroupORecoveryIntegrityError("Group O recovery completion evidence is unavailable")
    return GroupORecoveryResult(
        status="recovered",
        signal_session=TARGET_SIGNAL_SESSION_V1,
        observation_session=TARGET_OBSERVATION_SESSION_V1,
        attempt_id=attempt_id,
        start_receipt_path=start_receipt,
        canonical_option_rows=result.canonical_option_rows,
    )


def recover_group_o_exact_chain_until_ready_v1(
    *,
    context_root: str | Path,
    release_directory: str | Path,
    symbols: tuple[str, ...],
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleeper: Callable[[float], None] = time.sleep,
) -> GroupORecoveryResult:
    """Retry automatically on the signed 15-minute cadence until the pre-open cutoff."""

    cache_root = Path(context_root) / "source-cache" / "eodhd-group-o"
    signal_open, _ = xnys_session_bounds(TARGET_SIGNAL_SESSION_V1)
    while True:
        try:
            return recover_group_o_exact_chain_v1(
                context_root=context_root,
                release_directory=release_directory,
                symbols=symbols,
                clock=clock,
            )
        except GroupORecoveryRetryNotDue as exc:
            retry_after = exc.retry_after_utc
        except GroupOAcquisitionPending:
            pending_retry = group_o_retry_not_before(
                cache_root=cache_root,
                observation_session=TARGET_OBSERVATION_SESSION_V1,
            )
            if pending_retry is None:
                raise GroupORecoveryIntegrityError(
                    "Group O pending acquisition lacks signed retry evidence"
                ) from None
            retry_after = pending_retry
        now = clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise GroupORecoveryIntegrityError("Group O recovery retry clock is invalid")
        now_utc = now.astimezone(UTC)
        if retry_after >= signal_open:
            raise GroupORecoveryIntegrityError(
                "Group O exact chain remained unavailable before the signal open"
            )
        wait_seconds = max(0.0, (retry_after - now_utc).total_seconds())
        sleeper(wait_seconds)


def require_group_o_recovery_ready_before_adapter_v1(
    *,
    context_root: str | Path,
    release_directory: str | Path,
    now: datetime,
) -> None:
    """Block IBKR construction until the target revision and start evidence verify."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise GroupORecoveryIntegrityError("pre-adapter recovery clock is invalid")
    release = Path(release_directory)
    freeze = verify_group_o_recovery_freeze_v1(release)
    root = Path(context_root)
    _require_audited_failed_base(context_root=root, freeze_receipt=freeze)
    resolved = load_group_o_session_package(
        context_root=root,
        signal_session=TARGET_SIGNAL_SESSION_V1,
    )
    if any(context.quality_status == "missing_exact_chain" for context in resolved.contexts):
        raise GroupORecoveryIntegrityError(
            "blocked_pre_adapter_group_o_recovery_incomplete"
        )
    if not reconcile_group_o_recovery_completion_v1(
        context_root=root,
        release_directory=release,
        clock=lambda: now,
    ):
        raise GroupORecoveryIntegrityError("blocked_missing_pre_adapter_recovery_evidence")


def group_o_recovery_result_payload(result: GroupORecoveryResult) -> dict[str, object]:
    """JSON-ready record-only CLI projection."""

    return asdict(result)


__all__ = [
    "GroupORecoveryIntegrityError",
    "GroupORecoveryResult",
    "GroupORecoveryRetryNotDue",
    "GroupORecoveryStartReceiptV1",
    "RECOVERY_PACKAGE_RELATIVE_V1",
    "RECOVERY_VERSION_V1",
    "TARGET_OBSERVATION_SESSION_V1",
    "TARGET_SIGNAL_SESSION_V1",
    "group_o_recovery_result_payload",
    "recover_group_o_exact_chain_v1",
    "recover_group_o_exact_chain_until_ready_v1",
    "reconcile_group_o_recovery_completion_v1",
    "require_group_o_recovery_ready_before_adapter_v1",
    "verify_group_o_recovery_freeze_v1",
]
