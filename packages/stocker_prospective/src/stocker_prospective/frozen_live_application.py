"""Construction and lifecycle of the live frozen-M1C recorder application."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from stocker_prospective.activation import ActivationRecord, ProspectiveActivationLedger
from stocker_prospective.budget_reports import BudgetAwareDailyReportWriter
from stocker_prospective.capability import (
    CapabilityObservation,
    IBKRCapabilityManifest,
    run_capability_preflight,
)
from stocker_prospective.capacity import (
    CapacityDiscovery,
    RuntimeCapacityManifest,
    RuntimeCapacitySettings,
    WindowedRequestPacer,
    resolve_runtime_capacity,
)
from stocker_prospective.config import ProspectiveConfig, operational_thresholds
from stocker_prospective.context import previous_xnys_session
from stocker_prospective.contract import (
    BUDGET_AWARE_RECORDER_CONTRACT_VERSION,
    CONTRACT_VERSION,
    M1C_FEATURE_MANIFEST_SHA256,
    M1C_SCALING_ARTIFACT_SHA256,
    M1C_THRESHOLD_ARTIFACT_SHA256,
    SECTOR_PROXY_BY_SYMBOL,
    claims_boundary,
)
from stocker_prospective.database import (
    EvidenceMetadata,
    ProspectiveRepository,
    UnderlyingContractInput,
)
from stocker_prospective.direction import FrozenDirectionRuntime
from stocker_prospective.direction_features import FrozenDirectionFeatureBuilder
from stocker_prospective.durable_inbox import DurableCallbackInbox
from stocker_prospective.event_ingest import IBKRCallbackNormalizer
from stocker_prospective.frozen_m1c import FrozenM1CRuntime
from stocker_prospective.group_o import (
    FrozenGroupOSessionPackage,
    load_group_o_session_package,
)
from stocker_prospective.ibkr import IBKRMarketDataAdapter
from stocker_prospective.live_bars import xnys_session_bounds
from stocker_prospective.live_recorder import (
    FrozenM1CLiveRecorder,
    LivePollResult,
    ScientificReadiness,
)
from stocker_prospective.live_subscriptions import (
    LiveSubscriptionController,
    QualifiedUnderlying,
)
from stocker_prospective.m1c_features import (
    HistoricalActivityBaseline,
    M1CCausalFeatureBuilder,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningReversalCapacityCoordinatorV1,
    OpeningReversalPredictionReceiptV1,
    build_capacity_degradation_events_v1,
    load_activation_receipt_v1,
    load_frozen_experiment_config_v1,
    select_promoted_prediction_v1,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1_1 import (
    load_activation_receipt_v1_1,
    load_frozen_timing_addendum_config_v1_1,
)
from stocker_prospective.market_data import ConnectionState, MarketDataType
from stocker_prospective.opening_leader_continuation_v0 import (
    M1CContextV0,
    OpeningLeaderContinuationRecorderV0,
    OpeningLeaderEvidenceStoreV0,
    OpeningLeaderSelectionPromotionV0,
    checkpoint_timestamp_v0,
)
from stocker_prospective.opening_leader_live_v0 import (
    OpeningLeaderDeploymentReceiptV0,
    OpeningLeaderDeploymentRefreezeReceiptV1,
    OpeningLeaderDeploymentRefreezeReceiptV2,
    OpeningLeaderDeploymentRefreezeReceiptV3,
    OpeningLeaderDeploymentRefreezeReceiptV4,
    OpeningLeaderDeploymentRefreezeReceiptV5,
    OpeningLeaderDeploymentRefreezeReceiptV6,
    OpeningLeaderDeploymentRefreezeReceiptV7,
    OpeningLeaderDeploymentRefreezeReceiptV8,
    OpeningLeaderDeploymentRefreezeReceiptV9,
    OpeningLeaderDeploymentRefreezeReceiptV10,
    OpeningLeaderDeploymentRefreezeReceiptV11,
    OpeningLeaderDeploymentRefreezeReceiptV12,
    OpeningLeaderDeploymentRefreezeReceiptV13,
    OpeningLeaderIBKROptionSnapshotterV0,
    assert_opening_leader_runtime_configuration_v0,
    load_opening_leader_package_v0,
    opening_leader_runtime_source_files_v0,
)
from stocker_prospective.opening_market_transition_v1 import (
    load_opening_transition_threshold_manifest_v1,
)
from stocker_prospective.operational_state import (
    RecorderOperationalRepository,
    RuntimeArtifactVerification,
    stable_artifact_verification_id,
)
from stocker_prospective.option_budget import (
    BudgetAwareEpisodeStateMachine,
    EpisodeAllocationRecord,
    EpisodeKind,
    EpisodeState,
)
from stocker_prospective.option_discovery import BoundedOptionDiscoveryService
from stocker_prospective.option_recorder import BoundedOptionRecorder
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.phase import (
    EpisodeCompletion,
    ProspectivePhaseLedger,
    ProspectivePhaseManager,
)
from stocker_prospective.quality_report import (
    build_session_quality_report,
    write_session_quality_report,
)
from stocker_prospective.quiet_state_phase import (
    QuietObservationCompletion,
    QuietObservationKind,
    QuietStatePhaseLedger,
    QuietStatePhaseManager,
)
from stocker_prospective.recorder import RecorderDeploymentIdentity
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import (
    FrozenM1CRecorderEngine,
    RecorderCheckpointResult,
)
from stocker_prospective.signed_market_shock_v1 import (
    load_signed_market_shock_threshold_manifest_v1,
)
from stocker_prospective.source_transfer import SourceTransferCoordinator
from stocker_prospective.storage_recovery import CrossStoreReconciler
from stocker_prospective.subscriptions import (
    PromotionScheduler,
    SubscriptionBudgetManager,
    SubscriptionKind,
)
from stocker_prospective.tail_phase_v1 import load_tail_phase_frozen_config_v1

NEW_YORK = ZoneInfo("America/New_York")
MARKET_PROXY = "VTI"
SECTOR_PROXY_SYMBOLS = tuple(sorted(set(SECTOR_PROXY_BY_SYMBOL.values())))
_PROMOTION_RECOVERABLE_REJECTIONS = frozenset(
    {
        "scientific_recording_not_authorized",
        "underlying_quote_stale",
    }
)
_HARDENING_OPERATIONAL_RUNTIME_FIELDS = frozenset(
    {
        "callback_inbox_max_unacknowledged",
        "callback_inbox_batch_limit",
        "callback_inbox_lease_seconds",
        "callback_heartbeat_stale_seconds",
        "raw_storage_heartbeat_stale_seconds",
        "callback_acknowledgement_stale_seconds",
        "callback_inbox_healthy_backlog",
        "callback_inbox_oldest_healthy_seconds",
    }
)
_POST_ACTIVATION_RECORD_ONLY_IBKR_FIELDS = frozenset(
    {
        "option_commission_per_contract",
        "option_regulatory_fee_per_contract",
        "option_exchange_fee_per_contract",
    }
)
_POST_ACTIVATION_CROSS_VENDOR_CLAIMS = frozenset(
    {
        "market_data_source",
        "historical_research_source",
        "cross_vendor_validation_diagnostic_only",
        "cross_vendor_validation_required_for_science",
        "prospective_evidence_description",
    }
)
_LEGACY_WEB_PROJECTION_CACHE_SECONDS = 60.0


def _operationally_promotable(checkpoint: RecorderCheckpointResult) -> bool:
    """Allow quote acquisition only when it can resolve every current rejection."""

    return not (set(checkpoint.rejection_reasons) - _PROMOTION_RECOVERABLE_REJECTIONS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration_hash(
    config: ProspectiveConfig,
    *,
    git_commit: str | None = None,
    app_version: str | None = None,
    run_id: str | None = None,
    tws_or_gateway_version: str | None = None,
    omitted_runtime_fields: frozenset[str] = frozenset(),
    omitted_ibkr_fields: frozenset[str] = frozenset(),
    web_projection_cache_seconds: float | None = None,
) -> str:
    payload: dict[str, Any] = config.model_dump(mode="json")
    runtime_payload = cast(dict[str, Any], payload["runtime"])
    ibkr_payload = cast(dict[str, Any], payload["ibkr"])
    web_payload = cast(dict[str, Any], payload["web"])
    if git_commit is not None:
        runtime_payload["git_commit"] = git_commit
    if app_version is not None:
        runtime_payload["app_version"] = app_version
    if run_id is not None:
        runtime_payload["run_id"] = run_id
    if tws_or_gateway_version is not None:
        ibkr_payload["tws_or_gateway_version"] = tws_or_gateway_version
    for field_name in omitted_runtime_fields:
        runtime_payload.pop(field_name)
    for field_name in omitted_ibkr_fields:
        ibkr_payload.pop(field_name)
    if web_projection_cache_seconds is not None:
        web_payload["operational_projection_cache_seconds"] = web_projection_cache_seconds
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _activation_configuration_hash_candidates(
    config: ProspectiveConfig,
    *,
    activation_git_commit: str,
    activation_app_version: str | None = None,
    activation_tws_or_gateway_version: str | None = None,
    historical_run_ids: tuple[str, ...] = (),
) -> frozenset[str]:
    """Reconstruct supported activation shapes without weakening scientific fields."""

    candidates: set[str] = set()
    # A fatal run is never rewritten or cleared. A replacement run may reuse
    # the immutable activation only when a persisted historical run ID
    # reconstructs the exact original configuration hash. This makes run_id an
    # operational lineage boundary while every scientific input remains bound.
    for historical_run_id in (None, *sorted(set(historical_run_ids))):
        for web_projection_cache_seconds in (None, _LEGACY_WEB_PROJECTION_CACHE_SECONDS):
            for omitted_ibkr_fields in (
                frozenset(),
                _POST_ACTIVATION_RECORD_ONLY_IBKR_FIELDS,
            ):
                candidates.update(
                    {
                        _configuration_hash(
                            config,
                            git_commit=activation_git_commit,
                            app_version=activation_app_version,
                            run_id=historical_run_id,
                            tws_or_gateway_version=activation_tws_or_gateway_version,
                            omitted_ibkr_fields=omitted_ibkr_fields,
                            web_projection_cache_seconds=web_projection_cache_seconds,
                        ),
                        _configuration_hash(
                            config,
                            git_commit=activation_git_commit,
                            app_version=activation_app_version,
                            run_id=historical_run_id,
                            tws_or_gateway_version=activation_tws_or_gateway_version,
                            omitted_runtime_fields=_HARDENING_OPERATIONAL_RUNTIME_FIELDS,
                            omitted_ibkr_fields=omitted_ibkr_fields,
                            web_projection_cache_seconds=web_projection_cache_seconds,
                        ),
                    }
                )
    return frozenset(candidates)


def _activation_claims_boundary_candidates() -> tuple[Mapping[str, object], ...]:
    """Return exact current and superseded non-trading activation claim shapes."""

    current = claims_boundary()
    legacy = dict(current)
    for field_name in _POST_ACTIVATION_CROSS_VENDOR_CLAIMS:
        legacy.pop(field_name)
    legacy["engineering_phase_sessions"] = legacy.pop("historical_engineering_phase_sessions")
    return current, legacy


def _require_compatible_existing_activation(
    *,
    activation: ActivationRecord,
    config: ProspectiveConfig,
    artifact_hashes: Mapping[str, str],
    ibkr_api_version: str,
    tws_or_gateway_version: str,
    activation_app_version: str | None = None,
    historical_run_ids: tuple[str, ...] = (),
) -> None:
    """Preserve first activation while failing closed on scientific drift."""

    if activation.contract_version != CONTRACT_VERSION:
        raise ValueError("blocked_existing_activation_contract_version_mismatch")
    if all(
        activation.claims_boundary != candidate
        for candidate in _activation_claims_boundary_candidates()
    ):
        raise ValueError("blocked_existing_activation_claims_boundary_mismatch")
    if activation.model_artifact_hashes != dict(sorted(artifact_hashes.items())):
        raise ValueError("blocked_existing_activation_artifact_hash_mismatch")
    # The activation records the dependency baseline used at first collection.
    # Later official API/Gateway maintenance is operational: current versions
    # are independently verified and written to capability evidence. Rebuild
    # the old configuration hash with the baseline Gateway identity so only
    # that dependency field may roll without changing scientific admission.
    if not ibkr_api_version or ibkr_api_version == "unknown":
        raise ValueError("blocked_existing_activation_ibkr_api_version_unavailable")
    if not tws_or_gateway_version:
        raise ValueError("blocked_existing_activation_gateway_version_unavailable")
    if activation.configuration_hash not in _activation_configuration_hash_candidates(
        config,
        activation_git_commit=activation.git_sha,
        activation_app_version=activation_app_version,
        activation_tws_or_gateway_version=activation.tws_or_gateway_version,
        historical_run_ids=historical_run_ids,
    ):
        raise ValueError("blocked_existing_activation_configuration_mismatch")


def _attribute(value: Any, *names: str) -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _probe_required_market_data_type(
    adapter: Any,
    *,
    contract: Any,
    timeout_seconds: float,
) -> MarketDataType | None:
    """Observe IBKR's authoritative data type with one cancelled snapshot."""

    try:
        result = adapter.capture_temporary_quote(
            contract=contract,
            timeout_seconds=timeout_seconds,
        )
    except RuntimeError:
        return cast(MarketDataType | None, adapter.connection.health().market_data_type)
    observed: set[MarketDataType] = set()
    for item in result.items:
        raw = _attribute(item, "market_data_type")
        if raw is None and _attribute(item, "field") == "market_data_type":
            raw = _attribute(item, "value")
        if raw is None:
            continue
        try:
            observed.add(MarketDataType(str(raw)))
        except ValueError:
            continue
    if len(observed) != 1:
        return cast(MarketDataType | None, adapter.connection.health().market_data_type)
    market_data_type = next(iter(observed))
    adapter.connection.market_data_type_observed(market_data_type)
    return market_data_type


def _passed(path: Path, *, label: str) -> bool:
    if not path.is_file():
        raise ValueError(f"{label} report is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("passed") is True


class ScientificPrerequisiteError(ValueError):
    """Carry a stable blocker identity independently of operator-facing text."""

    def __init__(self, blocker_code: str, message: str) -> None:
        if not blocker_code.startswith("blocked_"):
            raise ValueError("scientific prerequisite blocker code is invalid")
        super().__init__(message)
        self.blocker_code = blocker_code


def _assert_frozen_m1c_artifact_hashes(artifact_files: Mapping[str, Path]) -> None:
    """Fail closed unless the exact preregistered M1C artifacts are loaded."""

    expected = {
        "m1c_feature_manifest": M1C_FEATURE_MANIFEST_SHA256,
        "m1c_threshold": M1C_THRESHOLD_ARTIFACT_SHA256,
        "m1c_scaling": M1C_SCALING_ARTIFACT_SHA256,
    }
    mismatches = sorted(
        name
        for name, expected_hash in expected.items()
        if name not in artifact_files or _sha256(artifact_files[name]) != expected_hash
    )
    if mismatches:
        raise ValueError("blocked_frozen_artifact_hash_mismatch: " + ",".join(mismatches))


def _activation_receipt_identity(record: object | None) -> str:
    if record is None or not hasattr(record, "model_dump"):
        return "activation_receipt_unavailable"
    payload = record.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _record_blocked_runtime_artifacts(
    *,
    repository: RecorderOperationalRepository | None,
    activation_ledger: ProspectiveActivationLedger,
    run_id: str,
    recorder_generation: int | None,
    artifact_files: Mapping[str, Path],
    blocker: str,
) -> None:
    """Best-effort startup evidence for a generation that cannot load artifacts."""

    if repository is None or recorder_generation is None:
        return
    try:
        activation = activation_ledger.load()
    except (OSError, ValueError):
        activation = None
    immutable_hashes = {} if activation is None else activation.model_artifact_hashes
    built_in_hashes = {
        "m1c_feature_manifest": M1C_FEATURE_MANIFEST_SHA256,
        "m1c_threshold": M1C_THRESHOLD_ARTIFACT_SHA256,
        "m1c_scaling": M1C_SCALING_ARTIFACT_SHA256,
    }
    receipt_identity = _activation_receipt_identity(activation)
    observed_at = datetime.now(UTC)
    for artifact_name, artifact_path in sorted(artifact_files.items()):
        found = artifact_path.is_file()
        observed_hash: str | None = None
        if found:
            try:
                observed_hash = _sha256(artifact_path)
            except OSError:
                found = False
        expected_hash = immutable_hashes.get(
            artifact_name,
            built_in_hashes.get(artifact_name, "expected_hash_unavailable"),
        )
        hash_verified = observed_hash is not None and observed_hash == expected_hash
        verification = RuntimeArtifactVerification(
            verification_id=stable_artifact_verification_id(
                run_id=run_id,
                recorder_generation=recorder_generation,
                artifact_name=artifact_name,
                expected_hash=expected_hash,
            ),
            run_id=run_id,
            recorder_generation=recorder_generation,
            artifact_bundle_id="artifact_bundle_unavailable",
            artifact_name=artifact_name,
            expected_hash=expected_hash,
            observed_hash=observed_hash,
            feature_contract_version=BUDGET_AWARE_RECORDER_CONTRACT_VERSION,
            activation_receipt_identity=receipt_identity,
            found=found,
            loaded=False,
            schema_validated=False,
            hash_verified=hash_verified,
            contract_compatible=False,
            used_by_active_generation=False,
            load_timestamp_utc=observed_at,
            verification_result="blocked",
            blocker=blocker,
            details={
                "startup_failed_closed": True,
                "expected_hash_source": (
                    "immutable_activation_receipt"
                    if artifact_name in immutable_hashes
                    else "built_in_frozen_contract"
                    if artifact_name in built_in_hashes
                    else "unavailable_before_first_activation"
                ),
            },
        )
        try:
            repository.record_artifact_verification(verification)
        except Exception:
            # The original artifact failure remains authoritative. Persistence
            # is best effort here because a storage failure has its own fatal
            # path and must not be disguised as an artifact parsing error.
            return


class FrozenProspectiveApplication:
    """Poll and recover one market-data-only process; no broker mutation exists."""

    def __init__(
        self,
        *,
        config: ProspectiveConfig,
        adapter: IBKRMarketDataAdapter,
        repository: ProspectiveRepository,
        metadata_factory: Callable[
            [datetime, tuple[datetime, ...]],
            EvidenceMetadata,
        ],
        live_recorder: FrozenM1CLiveRecorder,
        subscriptions: LiveSubscriptionController,
        subscription_budget: SubscriptionBudgetManager,
        opening_reversal_capacity: (OpeningReversalCapacityCoordinatorV1 | None),
        option_discovery: BoundedOptionDiscoveryService,
        phase_manager: ProspectivePhaseManager,
        quiet_phase_manager: QuietStatePhaseManager,
        promotion_scheduler: PromotionScheduler,
        runtime_capacity: RuntimeCapacityManifest,
        source_transfer: SourceTransferCoordinator,
        daily_report_writer: BudgetAwareDailyReportWriter,
        resolved_contracts: tuple[QualifiedUnderlying, ...],
        ibkr_api_version: str,
        tws_or_gateway_version: str,
        session_context_preflight: Callable[[date, datetime], None],
        opening_leader_recorder: OpeningLeaderContinuationRecorderV0 | None,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.repository = repository
        self.metadata_factory = metadata_factory
        self.live_recorder = live_recorder
        self.subscriptions = subscriptions
        self.subscription_budget = subscription_budget
        self.option_discovery = option_discovery
        self.phase_manager = phase_manager
        self.quiet_phase_manager = quiet_phase_manager
        self.promotion_scheduler = promotion_scheduler
        self.runtime_capacity = runtime_capacity
        self.source_transfer = source_transfer
        self.daily_report_writer = daily_report_writer
        self.resolved_contracts = resolved_contracts
        self.ibkr_api_version = ibkr_api_version
        self.tws_or_gateway_version = tws_or_gateway_version
        self.session_context_preflight = session_context_preflight
        self.opening_leader_recorder = opening_leader_recorder
        self._opening_leader_m1c_contexts: dict[
            tuple[date, int],
            dict[str, M1CContextV0],
        ] = {}
        self._probabilities: dict[str, float] = {}
        self._eligible_symbols: set[str] = set()
        self._active_episode_end: dict[str, tuple[str, datetime]] = {}
        self._episode_results: dict[str, RecorderCheckpointResult] = {}
        self._phase_finalised: set[str] = set()
        self._quiet_observation_results: dict[
            str,
            tuple[RecorderCheckpointResult, QuietObservationKind],
        ] = {}
        self._quiet_phase_finalised: set[str] = set()
        self._sessions_seen: set[date] = set()
        # Session reports are immutable per run/date. A replacement generation
        # must hydrate their identities before processing callbacks, otherwise
        # its first post-close poll can attempt to rewrite committed evidence.
        self._session_reports_written = (
            self.phase_manager.repository.recorded_session_quality_report_dates(
                run_id=self.config.runtime.run_id or "",
            )
        )
        self._session_context_checked: set[date] = set()
        self._session_context_failures: dict[date, str] = {}
        self._reconnect_attempts = 0
        self._next_reconnect_at: datetime | None = None
        self._last_reconciliation_monotonic = 0.0
        self._last_clock_probe_monotonic = time.monotonic()
        self.opening_reversal_capacity = opening_reversal_capacity
        self._opening_reversal_results: dict[
            tuple[date, int],
            dict[str, RecorderCheckpointResult],
        ] = {}
        self._opening_reversal_receipts: dict[
            tuple[date, int],
            dict[str, OpeningReversalPredictionReceiptV1],
        ] = {}
        self._opening_reversal_finalised_groups: set[tuple[date, int]] = set()

    def process_source_transfer(self, session: date, observed_at: datetime) -> None:
        """Rescore completed EODHD bars without changing live V0 decisions."""

        report = self.source_transfer.process_session(
            session=session,
            observed_at=observed_at,
        )
        self.daily_report_writer.write(
            session=session,
            generated_at=observed_at,
            capacity_manifest=self.runtime_capacity.to_dict(),
            budget_snapshot=self.subscription_budget.snapshot(),
            transfer_report=report,
        )

    def _persist_connection_events(self, observed_at: datetime) -> None:
        for event in self.adapter.connection.drain_events():
            if event.recorded_at < self.config.runtime.prospective_start_utc:
                continue
            metadata = self.metadata_factory(
                max(observed_at, event.recorded_at),
                (event.recorded_at,),
            )
            self.repository.record_ibkr_connection_event(
                metadata,
                state=event.state.value,
                error_code=event.code,
                message=event.message,
                data_maintained=event.data_maintained,
                reconnect_attempt=None,
                details={
                    "source": "official_ibkr_callback",
                    "event_kind": event.event_kind.value,
                },
            )

    def poll(self, *, now: datetime) -> LivePollResult:
        """Run one callback batch and fail closed across the full application."""

        observed = now.astimezone(UTC)
        try:
            return self._poll_once(now=observed)
        except Exception as exc:
            self.live_recorder.fail_inflight_durable_poll(
                exc,
                occurred_at=observed,
            )
            raise

    def _poll_once(self, *, now: datetime) -> LivePollResult:
        observed = now.astimezone(UTC)
        self._persist_connection_events(observed)
        observed_session = observed.astimezone(NEW_YORK).date()
        market_session_valid = False
        observed_market_open: datetime | None = None
        try:
            observed_market_open, _ = xnys_session_bounds(observed_session)
        except ValueError:
            pass
        else:
            market_session_valid = True
            self._sessions_seen.add(observed_session)
            if observed_session not in self._session_context_checked:
                try:
                    self.session_context_preflight(observed_session, observed)
                except (ScientificPrerequisiteError, ValueError) as exc:
                    message = str(exc)
                    self.live_recorder.set_session_context_ready(passed=False)
                    blocker_code = (
                        exc.blocker_code
                        if isinstance(exc, ScientificPrerequisiteError)
                        else "blocked_previous_session_options_context_invalid"
                    )
                    if self._session_context_failures.get(observed_session) != message:
                        metadata = self.metadata_factory(observed, (observed,))
                        self.repository.record_data_health_event(
                            metadata,
                            severity="blocker",
                            blocker_code=blocker_code,
                            component="previous_session_options_context",
                            message=message,
                            details={
                                "signal_session": observed_session.isoformat(),
                                "m1c_scoring_allowed": False,
                                "option_episode_capture_allowed": False,
                                "retrying": True,
                            },
                        )
                    self._session_context_failures[observed_session] = message
                else:
                    self.live_recorder.set_session_context_ready(passed=True)
                    self._session_context_checked.add(observed_session)
                    if observed_session in self._session_context_failures:
                        metadata = self.metadata_factory(observed, (observed,))
                        self.repository.record_data_health_event(
                            metadata,
                            severity="info",
                            blocker_code=None,
                            component="previous_session_options_context",
                            message="previous_session_options_context_ready",
                            details={
                                "signal_session": observed_session.isoformat(),
                                "m1c_scoring_allowed": (
                                    not self.live_recorder.scientific_block_latched
                                ),
                                "option_episode_capture_allowed": (
                                    not self.live_recorder.scientific_block_latched
                                ),
                                "resolved_blocker": (
                                    "blocked_missing_previous_session_options_context"
                                ),
                            },
                        )
                        self._session_context_failures.pop(observed_session, None)
        self._recover_if_required(observed)
        self._persist_connection_events(observed)
        result = self.live_recorder.poll(now=observed)
        opening_reversal_seen = False
        promoted_opening_episode_id: str | None = None
        callback_metadata = self.metadata_factory(observed, (observed,))
        for request_id, code in result.ibkr_errors:
            self.subscriptions.record_ibkr_error(
                callback_metadata,
                request_id=request_id,
                code=code,
            )
        for symbol in result.depth_reset_symbols:
            self.subscriptions.resubscribe_depth(
                callback_metadata,
                symbol=symbol,
            )
        if self.live_recorder.scientific_block_latched:
            # A replacement process may continue lossless raw admission and
            # generation-fenced acknowledgement, but no episode, promotion,
            # outcome, or checkpoint state can be reconstructed implicitly.
            self.live_recorder.finalize_durable_poll(
                result,
                acknowledged_at=observed,
            )
            return result
        checkpoints_to_complete: list[RecorderCheckpointResult] = []
        for checkpoint in result.checkpoint_results:
            symbol = checkpoint.episode_decision.symbol
            fresh_status = "NO_QUALIFIED_FRESH_EVENT"
            if checkpoint.episode_decision.fresh_episode and not checkpoint.rejection_reasons:
                fresh_status = (
                    "FIRST_ENTRY"
                    if checkpoint.tail_phase_v1.m1c_tail_phase_v1 == "FIRST_ENTRY"
                    else (
                        "QUALIFIED_RE_ENTRY"
                        if checkpoint.tail_phase_v1.m1c_tail_phase_v1 == "RE_ENTRY"
                        else "QUALIFIED_FRESH_EVENT_OTHER_PHASE"
                    )
                )
            elif checkpoint.episode_decision.fresh_episode:
                fresh_status = "UNQUALIFIED_FRESH_EVENT"
            self._opening_leader_m1c_contexts.setdefault(
                (
                    checkpoint.episode_decision.session,
                    checkpoint.episode_decision.checkpoint,
                ),
                {},
            )[symbol] = M1CContextV0(
                probability=checkpoint.score.probability,
                high_low_state=("HIGH" if checkpoint.score.threshold_passed else "LOW"),
                tail_phase=str(checkpoint.tail_phase_v1.m1c_tail_phase_v1),
                qualified_fresh_event_status=fresh_status,
                movement_consumed=(checkpoint.movement_consumed_state_v1.movement_consumed_v1),
                source_completeness=(
                    "complete"
                    if checkpoint.score.missing_feature_count == 0
                    else f"incomplete:{checkpoint.score.missing_feature_count}_missing_features"
                ),
            )
            opening_receipt = checkpoint.opening_reversal_prediction_v1
            if opening_receipt is not None:
                opening_reversal_seen = True
                group_key = (
                    checkpoint.episode_decision.session,
                    checkpoint.episode_decision.checkpoint,
                )
                self._opening_reversal_results.setdefault(group_key, {})[symbol] = checkpoint
                self._opening_reversal_receipts.setdefault(group_key, {})[symbol] = opening_receipt
            self._sessions_seen.add(checkpoint.episode_decision.session)
            self._probabilities[symbol] = checkpoint.score.probability
            if _operationally_promotable(checkpoint):
                self._eligible_symbols.add(symbol)
            else:
                self._eligible_symbols.discard(symbol)
            if checkpoint.episode_decision.fresh_episode and opening_receipt is None:
                episode_id = checkpoint.episode_decision.episode_id
                assert episode_id is not None
                self._episode_results[episode_id] = checkpoint
                metadata = self.metadata_factory(
                    observed,
                    (checkpoint.episode_decision.trigger_bar_end,),
                )
                self.subscriptions.promote_active_episode(
                    metadata,
                    symbol=symbol,
                    episode_id=episode_id,
                )
                self.option_discovery.schedule(checkpoint)
                self._active_episode_end[episode_id] = (
                    symbol,
                    checkpoint.episode_decision.prospective_entry_timestamp + timedelta(minutes=60),
                )
            if opening_receipt is None:
                self.option_discovery.schedule_quiet_state(checkpoint)
            quiet_observations: tuple[
                tuple[str | None, QuietObservationKind],
                ...,
            ] = (
                (checkpoint.quiet_observation_id, "quiet_bottom_10"),
                (checkpoint.neutral_control_id, "neutral_control"),
                (checkpoint.high_tail_control_id, "high_tail_control"),
            )
            for observation_id, kind in quiet_observations if opening_receipt is None else ():
                if observation_id is None:
                    continue
                self._quiet_observation_results[observation_id] = (checkpoint, kind)
                metadata = self.metadata_factory(
                    observed,
                    (checkpoint.quiet_episode_decision.trigger_timestamp,),
                )
                if kind == "quiet_bottom_10":
                    self.subscriptions.promote_active_episode(
                        metadata,
                        symbol=symbol,
                        episode_id=observation_id,
                    )
                self._active_episode_end[observation_id] = (
                    symbol,
                    checkpoint.quiet_episode_decision.prospective_entry_timestamp
                    + timedelta(minutes=60),
                )
            if opening_receipt is None:
                self.option_discovery.persist_checkpoint_schedules(checkpoint)
            checkpoints_to_complete.append(checkpoint)
        if (
            self.opening_leader_recorder is not None
            and observed >= self.opening_leader_recorder.boundary_utc
        ):
            for recovery_session in self.opening_leader_recorder.outstanding_sessions(now=observed):
                if recovery_session == observed_session:
                    continue
                self.opening_leader_recorder.poll(
                    session=recovery_session,
                    now=observed,
                    m1c_context_by_checkpoint={
                        checkpoint: dict(
                            self._opening_leader_m1c_contexts.get(
                                (recovery_session, checkpoint),
                                {},
                            )
                        )
                        for checkpoint in (6, 12)
                    },
                )
            if (
                market_session_valid
                and observed_market_open is not None
                and observed_market_open >= self.opening_leader_recorder.boundary_utc
            ):
                self.opening_leader_recorder.poll(
                    session=observed_session,
                    now=observed,
                    m1c_context_by_checkpoint={
                        checkpoint: dict(
                            self._opening_leader_m1c_contexts.get(
                                (observed_session, checkpoint),
                                {},
                            )
                        )
                        for checkpoint in (6, 12)
                    },
                )
        for receipt in result.opening_reversal_prediction_receipts:
            opening_reversal_seen = True
            self._opening_reversal_receipts.setdefault(
                (receipt.session, receipt.checkpoint),
                {},
            )[receipt.stock] = receipt
        barrier_audits_v1_1 = {
            audit.session: audit for audit in (result.opening_reversal_causal_barrier_audits_v1_1)
        }
        for group_key, group_receipts in tuple(self._opening_reversal_receipts.items()):
            if group_key in self._opening_reversal_finalised_groups or set(group_receipts) != set(
                self.live_recorder.universe_symbols
            ):
                continue
            receipts = tuple(
                group_receipts[symbol] for symbol in self.live_recorder.universe_symbols
            )
            if {receipt.experiment_version for receipt in receipts} == {"1.1"}:
                barrier_audit = barrier_audits_v1_1.get(group_key[0])
                if barrier_audit is None:
                    recorder_repository = self.live_recorder.repository
                    barrier_audit = (
                        recorder_repository.load_opening_reversal_causal_barrier_audit_v1_1(
                            run_id=callback_metadata.run_id,
                            session=group_key[0],
                        )
                    )
                if barrier_audit is None:
                    continue
                if barrier_audit.barrier_status != "passed":
                    self._opening_reversal_finalised_groups.add(group_key)
                    continue
            selection = select_promoted_prediction_v1(receipts)
            self._opening_reversal_finalised_groups.add(group_key)
            if selection.promoted is None:
                continue
            promoted = selection.promoted
            group_results = self._opening_reversal_results.get(group_key, {})
            if promoted.stock not in group_results:
                raise RuntimeError("eligible opening-reversal receipt lacks its causal score")
            promoted_result = group_results[promoted.stock]
            episode_id = promoted.fresh_episode_id
            assert episode_id is not None
            self.live_recorder.repository.record_opening_reversal_promotion_v1(
                self.metadata_factory(observed, (promoted.receipt_created_at_utc,)),
                selection,
            )
            metadata = self.metadata_factory(
                observed,
                (promoted.receipt_created_at_utc,),
            )
            self.subscriptions.promote_active_episode(
                metadata,
                symbol=promoted.stock,
                episode_id=episode_id,
            )
            self.option_discovery.schedule_opening_reversal(
                promoted_result,
                promoted,
            )
            self.option_discovery.persist_checkpoint_schedules(
                promoted_result,
                episode_ids=(episode_id,),
            )
            self._episode_results[episode_id] = promoted_result
            self._active_episode_end[episode_id] = (
                promoted.stock,
                promoted.entry_timestamp_utc + timedelta(minutes=30),
            )
            promoted_opening_episode_id = episode_id
        if result.checkpoint_results and not opening_reversal_seen:
            metadata = self.metadata_factory(observed, (observed,))
            active_symbols = {symbol for symbol, _ in self._active_episode_end.values()}
            decisions = self.promotion_scheduler.rank_checkpoint(
                checkpoint_time=observed,
                probabilities=self._probabilities,
                eligible_symbols=self._eligible_symbols,
                active_episode_symbols=active_symbols,
            )
            self.subscriptions.apply_checkpoint_promotions(metadata, decisions)
        self.option_discovery.poll(now=observed)
        if opening_reversal_seen:
            assert self.opening_reversal_capacity is not None
            snapshot = self.opening_reversal_capacity.snapshot(
                observed_at_utc=observed,
                promoted_episode_id=promoted_opening_episode_id,
            )
            self.live_recorder.repository.record_opening_reversal_capacity_snapshot_v1(
                self.metadata_factory(observed, (observed,)),
                snapshot,
            )
        for checkpoint in checkpoints_to_complete:
            opening_receipt = checkpoint.opening_reversal_prediction_v1
            if opening_receipt is not None:
                group_key = (
                    checkpoint.episode_decision.session,
                    checkpoint.episode_decision.checkpoint,
                )
                if group_key not in self._opening_reversal_finalised_groups:
                    continue
            completion_metadata = self.metadata_factory(
                observed,
                (checkpoint.episode_decision.trigger_bar_end,),
            )
            self.live_recorder.repository.mark_checkpoint_complete(
                completion_metadata,
                checkpoint_id=checkpoint.checkpoint_id,
                symbol=checkpoint.episode_decision.symbol,
                session=checkpoint.episode_decision.session,
                checkpoint=checkpoint.episode_decision.checkpoint,
            )
            self.live_recorder.acknowledge_checkpoint(checkpoint)
        self._reconcile_subscriptions(observed)
        self._finalise_due_phases(observed)
        self._finalise_due_quiet_phases(observed)
        self._write_due_session_reports(observed)
        metadata = self.metadata_factory(observed, (observed,))
        for episode_id, (_, end) in tuple(self._active_episode_end.items()):
            if observed >= end:
                self.subscriptions.end_active_episode(
                    metadata,
                    episode_id=episode_id,
                )
                self._active_episode_end.pop(episode_id)
        capability = self._write_capability_manifest(observed)
        self.live_recorder.set_capability_preflight(passed=capability.scientific_recording_valid)
        self.live_recorder.finalize_durable_poll(
            result,
            acknowledged_at=observed,
        )
        self._request_clock_probe_if_due(monotonic_now=time.monotonic())
        return result

    def _request_clock_probe_if_due(self, *, monotonic_now: float) -> bool:
        interval = self.config.ibkr.subscription_reconciliation_interval_seconds
        if monotonic_now - self._last_clock_probe_monotonic < interval:
            return False
        if self.adapter.connection.health().state is not ConnectionState.CONNECTED:
            return False
        # A startup probe can wait behind recovery CPU pressure before EWrapper
        # receives it. Refreshing this low-rate control sample lets a later
        # timely callback supersede stale latency evidence without relaxing the
        # frozen clock-drift tolerance.
        self.adapter.request_current_time()
        self._last_clock_probe_monotonic = monotonic_now
        return True

    def _reconcile_subscriptions(self, observed: datetime) -> None:
        monotonic_now = time.monotonic()
        if (
            monotonic_now - self._last_reconciliation_monotonic
            < self.config.ibkr.subscription_reconciliation_interval_seconds
        ):
            return
        actual_provider = getattr(
            self.adapter,
            "actual_subscription_request_ids",
            None,
        )
        if not callable(actual_provider):
            return
        actual = set(actual_provider())
        metadata = self.metadata_factory(observed, (observed,))
        self.subscriptions.reconcile(
            metadata,
            actual_request_ids=actual,
            pending_timeout_seconds=(self.config.ibkr.pending_subscription_timeout_seconds),
        )
        self.option_discovery.reconcile(
            metadata,
            actual_request_ids=actual,
        )
        self._last_reconciliation_monotonic = monotonic_now

    def _finalise_due_phases(self, observed: datetime) -> None:
        finalizations = self.option_discovery.finalizations
        for episode_id, result in sorted(
            self._episode_results.items(),
            key=lambda item: (
                item[1].episode_decision.prospective_entry_timestamp,
                item[0],
            ),
        ):
            if episode_id in self._phase_finalised:
                continue
            decision = result.episode_decision
            if observed < decision.prospective_entry_timestamp + timedelta(minutes=61):
                continue
            safety = result.episode_safety
            option_finalization = finalizations.get(episode_id)
            entry_quote = self.live_recorder.first_valid_quote_at_or_after(
                decision.symbol,
                decision.prospective_entry_timestamp,
            )
            completion = EpisodeCompletion(
                m1c_features_valid=(safety is not None and safety.scientific_recording_valid),
                directional_outputs_valid=(
                    result.direction_display_allowed
                    and set(result.directional_classifications) == {"A1", "C1", "R1"}
                ),
                underlying_level1_quality_passed=(
                    safety is not None and safety.scientific_recording_valid
                ),
                prospective_entry_observed=entry_quote is not None,
                ten_minute_outcome_observed=self.live_recorder.episode_window_completed(
                    episode_id,
                    "entry_to_+10m",
                ),
                required_quote_windows_finalised=(
                    option_finalization is not None
                    and bool(option_finalization.raw_contract_outcomes)
                ),
                data_gaps_accounted_for=not self.live_recorder.episode_has_gap(episode_id),
            )
            self.phase_manager.finalise(
                episode_id=episode_id,
                occurred_at=observed,
                completion=completion,
            )
            self._phase_finalised.add(episode_id)

    def _finalise_due_quiet_phases(self, observed: datetime) -> None:
        finalizations = self.option_discovery.finalizations
        rejections = self.option_discovery.rejections
        for observation_id, (result, kind) in sorted(
            self._quiet_observation_results.items(),
            key=lambda item: (
                item[1][0].quiet_episode_decision.prospective_entry_timestamp,
                item[0],
            ),
        ):
            if observation_id in self._quiet_phase_finalised:
                continue
            decision = result.quiet_episode_decision
            if observed < decision.prospective_entry_timestamp + timedelta(minutes=61):
                continue
            option_finalization = finalizations.get(observation_id)
            entry_quote = self.live_recorder.first_valid_quote_at_or_after(
                decision.symbol,
                decision.prospective_entry_timestamp,
            )
            completion = QuietObservationCompletion(
                frozen_m1c_prediction_valid=not result.rejection_reasons,
                underlying_entry_observed=entry_quote is not None,
                underlying_quote_window_finalised=self.live_recorder.episode_window_completed(
                    observation_id,
                    "entry_to_+60m",
                ),
                option_selection_attempted=(
                    option_finalization is not None or observation_id in rejections
                ),
                required_option_quote_windows_finalised=(
                    option_finalization is not None
                    and option_finalization.required_option_quote_windows_finalised
                ),
                data_gaps_accounted_for=not self.live_recorder.episode_has_gap(observation_id),
            )
            self.quiet_phase_manager.finalise(
                observation_id=observation_id,
                observation_kind=kind,
                occurred_at=observed,
                completion=completion,
            )
            self._quiet_phase_finalised.add(observation_id)

    def _write_due_session_reports(self, observed: datetime) -> None:
        for session in sorted(self._sessions_seen):
            if session in self._session_reports_written:
                continue
            try:
                _, market_close = xnys_session_bounds(session)
            except ValueError:
                continue
            if observed < market_close + timedelta(minutes=31):
                continue
            report = build_session_quality_report(
                database_path=self.repository.database_path,
                run_id=self.config.runtime.run_id or "",
                session_date=session,
                generated_at=observed,
            )
            metadata = self.metadata_factory(observed, (market_close,))
            self.phase_manager.repository.record_session_quality_report(metadata, report)
            write_session_quality_report(
                self.live_recorder.raw_store.root
                / "_session_reports"
                / f"{session.isoformat()}.json",
                report,
            )
            self.daily_report_writer.write(
                session=session,
                generated_at=observed,
                capacity_manifest=self.runtime_capacity.to_dict(),
                budget_snapshot=self.subscription_budget.snapshot(),
            )
            self._session_reports_written.add(session)

    def _recover_if_required(self, now: datetime) -> None:
        health = self.adapter.connection.health()
        if health.state is ConnectionState.PORT_RESET:
            raise RuntimeError("blocked_ibkr_connection: socket port reset")
        if health.state is ConnectionState.DISCONNECTED:
            if self._next_reconnect_at is not None and now < self._next_reconnect_at:
                return
            if self._reconnect_attempts >= self.config.ibkr.reconnect_max_attempts:
                raise RuntimeError("blocked_ibkr_connection: reconnect exhausted")
            self._reconnect_attempts += 1
            try:
                self.adapter.reconnect()
            except RuntimeError:
                self._next_reconnect_at = now + timedelta(
                    seconds=self.config.ibkr.reconnect_backoff_seconds
                    * (2 ** (self._reconnect_attempts - 1))
                )
                return
            health = self.adapter.connection.health()
        if health.subscriptions_require_rebuild:
            metadata = self.metadata_factory(now, (now,))
            for item in self.resolved_contracts:
                self.live_recorder.mark_gap(
                    item.symbol,
                    started_at=now,
                    cause_code="CONNECTION_DATA_LOSS_REBUILD",
                    stream_kind="connection",
                    recoverability="recoverable",
                )
            self.subscriptions.rebuild_after_data_loss(metadata)
            self.option_discovery.rebuild_after_data_loss(metadata)
            self.adapter.connection.subscriptions_rebuilt()
        if health.state is ConnectionState.CONNECTED:
            self._reconnect_attempts = 0
            self._next_reconnect_at = None

    def _write_capability_manifest(
        self,
        observed_at: datetime,
    ) -> IBKRCapabilityManifest:
        quotes = self.live_recorder.latest_quotes
        quote_types = {item.market_data_type for item in quotes.values()}
        if quote_types == {MarketDataType.LIVE}:
            market_data_type = MarketDataType.LIVE
        elif len(quote_types) == 1:
            market_data_type = next(iter(quote_types))
        else:
            market_data_type = (
                self.adapter.connection.health().market_data_type or MarketDataType.UNKNOWN
            )
        resolved = tuple(sorted(item.symbol for item in self.resolved_contracts))
        required_proxy_symbols = (MARKET_PROXY, *SECTOR_PROXY_SYMBOLS)
        stock_symbols = tuple(symbol for symbol in resolved if symbol not in required_proxy_symbols)
        clock_drift = self.live_recorder.clock_drift_seconds
        try:
            observed_session = observed_at.astimezone(NEW_YORK).date()
            calendar_valid = previous_xnys_session(observed_session) < observed_session
        except (RuntimeError, ValueError):
            calendar_valid = False
        permission_error_codes = {
            code
            for record in self.subscription_budget.records.values()
            for code in record.ibkr_error_codes
            if code in {354, 10089, 10090, 10186, 10197}
        }
        active_bar_symbols = {
            record.symbol
            for record in self.subscription_budget.records.values()
            if record.active and record.kind is SubscriptionKind.BAR
        }
        observation = CapabilityObservation(
            connected=self.adapter.connection.health().state is ConnectionState.CONNECTED,
            api_server_version=self.adapter.server_version(),
            ibkr_api_version=self.ibkr_api_version,
            tws_or_gateway_version=self.tws_or_gateway_version,
            market_data_type=market_data_type,
            underlying_level1_symbols=tuple(sorted(set(stock_symbols).intersection(quotes))),
            market_proxy_level1_symbols=tuple(
                sorted(set(required_proxy_symbols).intersection(quotes))
            ),
            underlying_bar_symbols=tuple(
                sorted(set(stock_symbols).intersection(active_bar_symbols))
            ),
            market_proxy_bar_symbols=tuple(
                sorted(set(required_proxy_symbols).intersection(active_bar_symbols))
            ),
            option_level1_available=self.option_discovery.live_option_quote_seen,
            option_computation_fields_available=(self.option_discovery.option_computation_seen),
            tick_by_tick_capacity=int(self.runtime_capacity.tick_by_tick_capacity.value),
            depth_capacity=int(self.runtime_capacity.depth_capacity.value),
            option_capacity=min(
                self.config.ibkr.max_option_subscriptions,
                self.config.ibkr.max_active_option_episodes
                * self.config.ibkr.max_option_lines_per_episode,
            ),
            depth_exchanges=self.live_recorder.depth_exchanges,
            resolved_contracts=resolved,
            unresolved_contracts=(),
            clock_drift_seconds=clock_drift,
            new_york_calendar_valid=calendar_valid,
            timestamps_valid=all(
                item.received_timestamp_utc
                >= (self.live_recorder.normalizer.prospective_collection_start)
                for item in quotes.values()
            ),
            permission_errors=tuple(
                f"ibkr_error_{code}" for code in sorted(permission_error_codes)
            ),
        )
        return run_capability_preflight(
            observation,
            required_underlyings=tuple(
                item.symbol for item in self.resolved_contracts if not item.market_proxy
            ),
            required_market_proxies=required_proxy_symbols,
            maximum_clock_drift_seconds=(self.config.ibkr.maximum_clock_drift_seconds),
            output_path=self.config.paths.ibkr_capability_manifest,
            observed_at=observed_at,
        )

    def shutdown(self, *, now: datetime) -> None:
        observed = now.astimezone(UTC)
        self._write_due_session_reports(observed)
        metadata = self.metadata_factory(observed, (observed,))
        self.option_discovery.shutdown(metadata)
        self.subscriptions.shutdown(metadata)


def build_frozen_prospective_application(
    *,
    config: ProspectiveConfig,
    adapter: IBKRMarketDataAdapter,
    repository: ProspectiveRepository,
    identity: RecorderDeploymentIdentity,
    stock_contract_factory: Callable[[str], Any],
    option_contract_factory: Callable[
        [str, date, float, str, int, str, str],
        Any,
    ],
    ibkr_api_version: str,
    heartbeat: Callable[[], object] | None = None,
    durable_inbox: DurableCallbackInbox | None = None,
    recorder_generation: int | None = None,
    recorder_owner_id: str | None = None,
) -> FrozenProspectiveApplication:
    """Build the live service only after all frozen artifact identities verify."""

    if config.runtime.run_id is None:
        raise ValueError("frozen M1C application requires runtime.run_id")
    paths = config.paths
    opening_leader_receipt: (
        OpeningLeaderDeploymentReceiptV0
        | OpeningLeaderDeploymentRefreezeReceiptV1
        | OpeningLeaderDeploymentRefreezeReceiptV2
        | OpeningLeaderDeploymentRefreezeReceiptV3
        | OpeningLeaderDeploymentRefreezeReceiptV4
        | OpeningLeaderDeploymentRefreezeReceiptV5
        | OpeningLeaderDeploymentRefreezeReceiptV6
        | OpeningLeaderDeploymentRefreezeReceiptV7
        | OpeningLeaderDeploymentRefreezeReceiptV8
        | OpeningLeaderDeploymentRefreezeReceiptV9
        | OpeningLeaderDeploymentRefreezeReceiptV10
        | OpeningLeaderDeploymentRefreezeReceiptV11
        | OpeningLeaderDeploymentRefreezeReceiptV12
        | OpeningLeaderDeploymentRefreezeReceiptV13
        | None
    ) = None
    if paths.opening_leader_continuation_v0_root is not None:
        assert_opening_leader_runtime_configuration_v0(
            mode=config.runtime.mode,
            maximum_quote_age_seconds=config.ibkr.maximum_quote_age_seconds,
            trading_enabled=config.risk.trading_enabled,
        )
        opening_leader_receipt = load_opening_leader_package_v0(
            paths.opening_leader_continuation_v0_root,
            source_files=opening_leader_runtime_source_files_v0(),
        )
    required_paths = {
        "raw_event_root": paths.raw_event_root,
        "recorder_activation": paths.recorder_activation,
        "m1c_live_parity_report": paths.m1c_live_parity_report,
        "direction_live_parity_report": paths.direction_live_parity_report,
        "frozen_m1c_artifact_root": paths.frozen_m1c_artifact_root,
        "m1c_scaling_artifact": paths.m1c_scaling_artifact,
        "direction_beta_artifact": paths.direction_beta_artifact,
        "historical_activity_bars": paths.historical_activity_bars,
        "bar_compatibility_report": paths.bar_compatibility_report,
        "context_root": paths.context_root,
        "ibkr_capability_manifest": paths.ibkr_capability_manifest,
        "prospective_phase_ledger": paths.prospective_phase_ledger,
        "prospective_report_root": paths.prospective_report_root,
        "aggregate_transfer_report": paths.aggregate_transfer_report,
    }
    missing = sorted(name for name, value in required_paths.items() if value is None)
    if missing:
        raise ValueError("frozen recorder paths absent: " + ",".join(missing))
    resolved_paths: dict[str, Path] = {}
    for name, value in required_paths.items():
        assert value is not None
        resolved_paths[name] = Path(value)
    operational_repository = (
        None
        if recorder_generation is None or recorder_owner_id is None
        else RecorderOperationalRepository(repository.database_path)
    )
    activation_ledger = ProspectiveActivationLedger(resolved_paths["recorder_activation"])
    historical_run_ids = repository.prospective_run_ids()
    first_activation_app_version = (
        repository.prospective_run_app_version(run_id=config.runtime.run_id)
        or (
            repository.prospective_run_app_version(run_id=historical_run_ids[0])
            if historical_run_ids
            else None
        )
        or config.runtime.app_version
    )
    artifact_root = resolved_paths["frozen_m1c_artifact_root"]
    if set(SECTOR_PROXY_BY_SYMBOL) != set(identity.symbols):
        raise ValueError("frozen sector-proxy map differs from the exact 20-stock cohort")
    artifact_files = {
        "m1c_feature_manifest": artifact_root / "causal_movement_feature_manifest.json",
        "m1c_threshold": artifact_root / "causal_movement_threshold.json",
        "direction_models": artifact_root / "model_configurations.json",
        "direction_normalisation": artifact_root / "stock_local_normalisation_parameters.json",
        "direction_thresholds": artifact_root / "frozen_archetype_thresholds.json",
        "direction_beta": resolved_paths["direction_beta_artifact"],
        "m1c_scaling": resolved_paths["m1c_scaling_artifact"],
    }
    tail_phase_config = None
    tail_phase_activation_status_v1 = "not_configured"
    if paths.m1c_tail_phase_v1_config is not None:
        tail_phase_path = Path(paths.m1c_tail_phase_v1_config)
        if not tail_phase_path.is_file():
            tail_phase_activation_status_v1 = "unavailable:config_absent"
        else:
            try:
                tail_phase_config = load_tail_phase_frozen_config_v1(tail_phase_path)
            except (OSError, ValueError):
                tail_phase_activation_status_v1 = "unavailable:config_invalid"
            else:
                tail_phase_activation_status_v1 = "available"
                artifact_files["m1c_tail_phase_v1_config"] = tail_phase_path
    signed_market_shock_thresholds_v1 = None
    signed_market_shock_activation_status_v1 = "not_configured"
    if paths.m1c_signed_market_shock_v1_config is not None:
        shock_path = Path(paths.m1c_signed_market_shock_v1_config)
        if not shock_path.is_file():
            signed_market_shock_activation_status_v1 = "unavailable:config_absent"
        else:
            try:
                signed_market_shock_thresholds_v1 = load_signed_market_shock_threshold_manifest_v1(
                    shock_path
                )
            except (OSError, ValueError):
                signed_market_shock_activation_status_v1 = "unavailable:config_invalid"
            else:
                signed_market_shock_activation_status_v1 = "available"
                artifact_files["m1c_signed_market_shock_v1_config"] = shock_path
    opening_transition_thresholds_v1 = None
    opening_transition_activation_status_v1 = "not_configured"
    if paths.m1c_opening_market_transition_v1_config is not None:
        opening_path = Path(paths.m1c_opening_market_transition_v1_config)
        if not opening_path.is_file():
            opening_transition_activation_status_v1 = "unavailable:config_absent"
        else:
            try:
                opening_manifest = load_opening_transition_threshold_manifest_v1(opening_path)
            except (OSError, ValueError):
                opening_transition_activation_status_v1 = "unavailable:config_invalid"
            else:
                opening_transition_thresholds_v1 = opening_manifest.thresholds
                opening_transition_activation_status_v1 = "available"
                artifact_files["m1c_opening_market_transition_v1_config"] = opening_path
    opening_reversal_activation_v1 = None
    reversal_config_path = paths.m1c_prospective_opening_reversal_v1_config
    reversal_activation_path = paths.m1c_prospective_opening_reversal_v1_activation
    if (reversal_config_path is None) != (reversal_activation_path is None):
        raise ValueError("opening reversal config and activation must be configured together")
    if reversal_config_path is not None:
        assert reversal_activation_path is not None
        artifact_files["m1c_prospective_opening_reversal_v1_config"] = Path(reversal_config_path)
        artifact_files["m1c_prospective_opening_reversal_v1_activation"] = Path(
            reversal_activation_path
        )
    opening_reversal_activation_v1_1 = None
    reversal_v1_1_config_path = paths.m1c_prospective_opening_reversal_v1_1_config
    reversal_v1_1_activation_path = paths.m1c_prospective_opening_reversal_v1_1_activation
    if (reversal_v1_1_config_path is None) != (reversal_v1_1_activation_path is None):
        raise ValueError("opening reversal V1.1 config and activation must be configured together")
    if reversal_v1_1_config_path is not None:
        assert reversal_v1_1_activation_path is not None
        artifact_files["m1c_prospective_opening_reversal_v1_1_config"] = Path(
            reversal_v1_1_config_path
        )
        artifact_files["m1c_prospective_opening_reversal_v1_1_activation"] = Path(
            reversal_v1_1_activation_path
        )
    if operational_repository is not None:
        assert recorder_generation is not None
        assert recorder_owner_id is not None
        operational_repository.start_generation(
            run_id=config.runtime.run_id,
            recorder_generation=recorder_generation,
            owner_id=recorder_owner_id,
            started_at=datetime.now(UTC),
            required_market_data_mode=(
                config.ibkr.allowed_market_data_types[0]
                if len(config.ibkr.allowed_market_data_types) == 1
                else None
            ),
            expected_artifact_count=len(artifact_files),
        )

    def block_runtime_artifacts(blocker: str) -> None:
        _record_blocked_runtime_artifacts(
            repository=operational_repository,
            activation_ledger=activation_ledger,
            run_id=config.runtime.run_id or "",
            recorder_generation=recorder_generation,
            artifact_files=artifact_files,
            blocker=blocker,
        )

    if reversal_config_path is not None:
        assert reversal_activation_path is not None
        if config.ibkr.reserved_future_trading_lines != 12:
            raise ValueError("opening reversal V1 requires exactly 12 reserved market-data lines")
        try:
            frozen_reversal_config = load_frozen_experiment_config_v1(str(reversal_config_path))
            opening_reversal_activation_v1 = load_activation_receipt_v1(
                str(reversal_activation_path)
            )
        except (OSError, ValueError):
            block_runtime_artifacts("RUNTIME_ARTIFACT_SCHEMA_INVALID")
            raise
        if (
            opening_reversal_activation_v1.configuration_hash
            != frozen_reversal_config.configuration_hash
        ):
            block_runtime_artifacts("RUNTIME_ARTIFACT_CONTRACT_INCOMPATIBLE")
            raise ValueError("opening reversal activation/configuration hash mismatch")
    if reversal_v1_1_config_path is not None:
        assert reversal_v1_1_activation_path is not None
        if opening_reversal_activation_v1 is None:
            block_runtime_artifacts("RUNTIME_ARTIFACT_CONTRACT_INCOMPATIBLE")
            raise ValueError("opening reversal V1.1 requires the exact V1 activation")
        if config.ibkr.reserved_future_trading_lines != 12:
            raise ValueError(
                "opening reversal V1.1 preserves exactly 12 reserved market-data lines"
            )
        try:
            timing_addendum_config = load_frozen_timing_addendum_config_v1_1(
                str(reversal_v1_1_config_path)
            )
            opening_reversal_activation_v1_1 = load_activation_receipt_v1_1(
                str(reversal_v1_1_activation_path)
            )
        except (OSError, ValueError):
            block_runtime_artifacts("RUNTIME_ARTIFACT_SCHEMA_INVALID")
            raise
        if (
            opening_reversal_activation_v1_1.timing_addendum_configuration_hash_v1_1
            != timing_addendum_config.configuration_hash_v1_1
            or (
                opening_reversal_activation_v1_1.superseded_activation_receipt_hash_v1
                != opening_reversal_activation_v1.activation_receipt_hash
            )
            or opening_reversal_activation_v1_1.frozen_rule_hash
            != opening_reversal_activation_v1.frozen_rule_hash
            or (
                opening_reversal_activation_v1_1.frozen_configuration_hash_v1
                != opening_reversal_activation_v1.configuration_hash
            )
        ):
            block_runtime_artifacts("RUNTIME_ARTIFACT_CONTRACT_INCOMPATIBLE")
            raise ValueError("opening reversal V1.1 activation/addendum/V1 binding mismatch")
    if any(not path.is_file() for path in artifact_files.values()):
        absent = sorted(name for name, path in artifact_files.items() if not path.is_file())
        block_runtime_artifacts("RUNTIME_ARTIFACT_NOT_FOUND")
        raise ValueError("frozen recorder artifact absent: " + ",".join(absent))
    try:
        _assert_frozen_m1c_artifact_hashes(artifact_files)
    except ValueError:
        block_runtime_artifacts("RUNTIME_ARTIFACT_HASH_MISMATCH")
        raise
    try:
        m1c_runtime = FrozenM1CRuntime.from_artifacts(
            feature_manifest_path=artifact_files["m1c_feature_manifest"],
            threshold_path=artifact_files["m1c_threshold"],
        )
        direction_runtime = FrozenDirectionRuntime.from_artifacts(
            model_configurations_path=artifact_files["direction_models"],
            normalisation_path=artifact_files["direction_normalisation"],
            thresholds_path=artifact_files["direction_thresholds"],
        )
        m1c_features = M1CCausalFeatureBuilder.from_scaling_artifact(artifact_files["m1c_scaling"])
        direction_features = FrozenDirectionFeatureBuilder.from_beta_artifact(
            artifact_files["direction_beta"]
        )
    except (OSError, ValueError):
        block_runtime_artifacts("RUNTIME_ARTIFACT_SCHEMA_INVALID")
        raise
    m1c_parity = _passed(
        resolved_paths["m1c_live_parity_report"],
        label="M1C live parity",
    )
    direction_parity = _passed(
        resolved_paths["direction_live_parity_report"],
        label="direction live parity",
    )
    bar_compatibility_path = resolved_paths["bar_compatibility_report"]
    bar_compatibility_available = bar_compatibility_path.is_file()
    bar_compatibility = (
        _passed(
            bar_compatibility_path,
            label="bar compatibility",
        )
        if bar_compatibility_available
        else False
    )
    gateway_version = config.ibkr.tws_or_gateway_version
    if not gateway_version:
        raise ValueError("IBKR_TWS_OR_GATEWAY_VERSION is required at activation")
    configuration_hash = _configuration_hash(config)

    def pace_request() -> None:
        if heartbeat is not None:
            heartbeat()
        time.sleep(1.05 / config.ibkr.request_rate_per_second)

    # Resolve every scientifically required contract before the immutable activation
    # boundary is written. Failed qualification is a startup failure, not prospective
    # evidence.
    qualified: list[QualifiedUnderlying] = []
    statuses: dict[str, tuple[str, str | None]] = {}
    proxy_symbols = (MARKET_PROXY, *SECTOR_PROXY_SYMBOLS)
    for symbol in (*identity.symbols, *proxy_symbols):
        result = adapter.qualify_exact_contract(stock_contract_factory(symbol))
        pace_request()
        matches = [
            (_attribute(item, "contract") or item)
            for item in result.items
            if str(
                _attribute(
                    _attribute(item, "contract") or item,
                    "symbol",
                )
                or ""
            )
            == symbol
            and int(
                _attribute(
                    _attribute(item, "contract") or item,
                    "conId",
                    "con_id",
                )
                or 0
            )
            > 0
        ]
        if len(matches) != 1:
            if symbol in identity.symbols:
                statuses[symbol] = (
                    "rejected_contract_qualification",
                    "exact_contract_resolution_failed",
                )
            continue
        contract = matches[0]
        qualified.append(
            QualifiedUnderlying(
                symbol=symbol,
                con_id=int(_attribute(contract, "conId", "con_id")),
                upstream_contract=contract,
                exchange=str(_attribute(contract, "exchange") or "SMART"),
                market_proxy=symbol in proxy_symbols,
            )
        )
        if symbol in identity.symbols:
            statuses[symbol] = ("recording", None)
    if {item.symbol for item in qualified} != {
        *identity.symbols,
        *proxy_symbols,
    }:
        raise ValueError("required underlying or context-proxy contract unresolved")
    adapter.require_live_market_data()
    market_data_probe_contract = next(
        item.upstream_contract for item in qualified if item.symbol == MARKET_PROXY
    )
    _probe_required_market_data_type(
        adapter,
        contract=market_data_probe_contract,
        timeout_seconds=config.ibkr.quote_capture_timeout_seconds,
    )
    pace_request()
    capacity_observer = getattr(adapter, "discover_market_data_capacity", None)
    discovered_capacity = (
        capacity_observer()
        if callable(capacity_observer)
        else CapacityDiscovery(
            market_data_status=(
                adapter.connection.health().market_data_type or MarketDataType.UNKNOWN
            ).value
        )
    )
    if not isinstance(discovered_capacity, CapacityDiscovery):
        raise TypeError("IBKR capacity discovery must return CapacityDiscovery")
    capability_manifest_path = resolved_paths["ibkr_capability_manifest"]
    capacity_manifest_path = (
        Path(paths.ibkr_runtime_capacity_manifest)
        if paths.ibkr_runtime_capacity_manifest is not None
        else capability_manifest_path.with_name("ibkr_runtime_capacity_manifest.json")
    )
    runtime_capacity = resolve_runtime_capacity(
        settings=RuntimeCapacitySettings(
            configured_total_market_data_lines=config.ibkr.market_data_line_budget,
            configured_externally_reserved_lines=config.ibkr.externally_reserved_lines,
            reserved_future_trading_lines=config.ibkr.reserved_future_trading_lines,
            safety_margin_lines=config.ibkr.safety_margin_lines,
            configured_max_tick_by_tick=config.ibkr.max_tick_by_tick_subscriptions,
            configured_max_depth=config.ibkr.max_depth_subscriptions,
            configured_max_concurrent_snapshots=config.ibkr.max_concurrent_snapshots,
            configured_max_active_option_episodes=(config.ibkr.max_active_option_episodes),
            configured_max_option_lines_per_episode=(config.ibkr.max_option_lines_per_episode),
            configured_historical_requests_per_window=(config.ibkr.historical_requests_per_window),
            configured_historical_request_window_seconds=(
                config.ibkr.historical_request_window_seconds
            ),
        ),
        discovery=discovered_capacity,
        output_path=capacity_manifest_path,
    )
    historical_request_pacer = WindowedRequestPacer(
        maximum_requests=int(runtime_capacity.historical_requests_per_window.value),
        window_seconds=float(runtime_capacity.historical_request_window_seconds.value),
        heartbeat=heartbeat,
    )

    artifact_hashes = {name: _sha256(path) for name, path in sorted(artifact_files.items())}
    existing_activation = activation_ledger.load()
    if existing_activation is None:
        activation = activation_ledger.activate(
            activation_timestamp_utc=datetime.now(UTC),
            git_sha=config.runtime.git_commit,
            model_artifact_hashes=artifact_hashes,
            configuration_hash=configuration_hash,
            ibkr_api_version=ibkr_api_version,
            tws_or_gateway_version=gateway_version,
        )
    else:
        _require_compatible_existing_activation(
            activation=existing_activation,
            config=config,
            artifact_hashes=artifact_hashes,
            ibkr_api_version=ibkr_api_version,
            tws_or_gateway_version=gateway_version,
            activation_app_version=first_activation_app_version,
            historical_run_ids=historical_run_ids,
        )
        activation = existing_activation
    prospective_start = activation.prospective_collection_start_utc

    def metadata_factory(
        observed_at: datetime,
        source_timestamps: tuple[datetime, ...],
    ) -> EvidenceMetadata:
        return EvidenceMetadata(
            run_id=config.runtime.run_id or "",
            prospective_start_utc=prospective_start,
            app_version=first_activation_app_version,
            # prospective_run is the immutable first-activation identity.
            # The current release is persisted with the active generation's
            # runtime-artifact receipts below.
            git_commit=activation.git_sha,
            model_artifact_id=m1c_runtime.model_hash,
            universe_id=identity.universe_id,
            cohort="anchor_frozen_20",
            source_timestamps=[item.astimezone(UTC).isoformat() for item in source_timestamps],
            recorded_at_utc=max(observed_at.astimezone(UTC), prospective_start),
        )

    initial_metadata = metadata_factory(prospective_start, (prospective_start,))
    repository.create_run(initial_metadata, mode="record_only")
    frozen_repository = FrozenRecorderRepository(
        repository,
        configuration_hash=activation.configuration_hash,
    )
    if opening_reversal_activation_v1 is not None:
        frozen_repository.record_opening_reversal_activation_v1(
            initial_metadata,
            opening_reversal_activation_v1,
        )
    if opening_reversal_activation_v1_1 is not None:
        frozen_repository.record_opening_reversal_activation_v1_1(
            initial_metadata,
            opening_reversal_activation_v1_1,
        )
    historical_activity_path = resolved_paths["historical_activity_bars"]
    historical_activity_available = historical_activity_path.is_file()
    if not bar_compatibility_available:
        repository.record_data_health_event(
            initial_metadata,
            severity="info",
            blocker_code=None,
            component="bar_compatibility",
            message=(
                "EODHD-to-IBKR bar compatibility is pending and will be monitored "
                "prospectively; exact vendor equality is not required"
            ),
            details={
                "path": str(bar_compatibility_path),
                "m1c_scoring_allowed": True,
                "raw_ibkr_acquisition_allowed": True,
                "source_transfer_monitoring": True,
                "exact_vendor_bar_equality_required": False,
            },
        )
    if not historical_activity_available:
        repository.record_data_health_event(
            initial_metadata,
            severity="blocker",
            blocker_code="blocked_historical_activity_baseline_absent",
            component="historical_activity_baseline",
            message=(
                "IBKR acquisition remains active; M1C scoring is blocked until "
                "the frozen prior-session activity baseline exists"
            ),
            details={
                "path": str(historical_activity_path),
                "m1c_scoring_allowed": False,
                "raw_ibkr_acquisition_allowed": True,
            },
        )

    static_scientific_prerequisites_passed = (
        m1c_parity and direction_parity and historical_activity_available
    )

    def prospective_phase_at(observed_at: datetime) -> tuple[str, bool]:
        phase, phase_allows_scientific_evidence = frozen_repository.prospective_phase_for_session(
            run_id=config.runtime.run_id or "",
            session=observed_at.astimezone(NEW_YORK).date(),
        )
        return (
            phase,
            phase_allows_scientific_evidence and static_scientific_prerequisites_passed,
        )

    frozen_repository.record_runtime_capacity(
        initial_metadata,
        manifest=runtime_capacity.to_dict(),
    )
    phase_manager = ProspectivePhaseManager(
        ledger=ProspectivePhaseLedger(resolved_paths["prospective_phase_ledger"]),
        repository=frozen_repository,
        phase_resolver=prospective_phase_at,
    )
    quiet_phase_manager = QuietStatePhaseManager(
        ledger=QuietStatePhaseLedger(
            resolved_paths["prospective_phase_ledger"].with_name("quiet_state_phases_v0.jsonl")
        ),
        repository=frozen_repository,
        phase_resolver=prospective_phase_at,
    )
    session = datetime.now(NEW_YORK).date()
    activity = (
        HistoricalActivityBaseline.from_parquet(
            historical_activity_path,
            latest_authorised_session=previous_xnys_session(session),
        )
        if historical_activity_available
        else HistoricalActivityBaseline()
    )
    normalizer = IBKRCallbackNormalizer(prospective_collection_start=prospective_start)
    raw_store = PartitionedEventStore(
        root=resolved_paths["raw_event_root"],
        prospective_collection_start=prospective_start,
        recorder_version=config.runtime.app_version,
        contract_version=BUDGET_AWARE_RECORDER_CONTRACT_VERSION,
        run_id=config.runtime.run_id,
    )
    if (
        durable_inbox is not None
        and recorder_generation is not None
        and operational_repository is not None
    ):
        recovery = CrossStoreReconciler(
            repository=repository,
            recorder_repository=frozen_repository,
            raw_store=raw_store,
            inbox=durable_inbox,
            run_id=config.runtime.run_id,
            recorder_generation=recorder_generation,
        ).reconcile(initial_metadata, observed_at=datetime.now(UTC))
        if not recovery.safe_to_score:
            raise RuntimeError("blocked_storage_integrity_reconciliation_failed")
    group_packages: dict[date, FrozenGroupOSessionPackage] = {}

    def group_o_provider(symbol: str, signal_session: date) -> Any:
        package = group_packages.get(signal_session)
        if package is None:
            package = load_group_o_session_package(
                context_root=resolved_paths["context_root"],
                signal_session=signal_session,
            )
            group_packages[signal_session] = package
        return package.for_symbol(symbol)

    def session_context_preflight(signal_session: date, observed_at: datetime) -> None:
        package = group_packages.get(signal_session)
        if package is None:
            package_path = (
                resolved_paths["context_root"] / "group-o" / f"{signal_session.isoformat()}.json"
            )
            if not package_path.is_file():
                raise ScientificPrerequisiteError(
                    "blocked_missing_previous_session_options_context",
                    "exact previous-session Group O package is absent for "
                    f"{signal_session.isoformat()}",
                )
            try:
                package = load_group_o_session_package(
                    context_root=resolved_paths["context_root"],
                    signal_session=signal_session,
                )
            except ValueError as exc:
                raise ScientificPrerequisiteError(
                    "blocked_previous_session_options_context_invalid",
                    str(exc),
                ) from exc
            group_packages[signal_session] = package
        package_symbols = {context.symbol for context in package.contexts}
        if package_symbols != set(identity.symbols):
            raise ScientificPrerequisiteError(
                "blocked_previous_session_options_context_invalid",
                "Group O package does not contain the exact frozen cohort",
            )
        metadata = metadata_factory(observed_at, (observed_at,))
        for symbol in identity.symbols:
            context = package.for_symbol(symbol)
            missing_features = m1c_runtime.missing_group_o_features(context.features)
            if missing_features:
                raise ScientificPrerequisiteError(
                    "blocked_previous_session_options_context_invalid",
                    "Group O context missing frozen feature keys for "
                    f"{symbol}: {','.join(missing_features)}",
                )
            frozen_repository.record_group_o_context(metadata, context)

    engine = FrozenM1CRecorderEngine(
        m1c_runtime=m1c_runtime,
        m1c_features=m1c_features,
        direction_runtime=direction_runtime,
        direction_features=direction_features,
        repository=frozen_repository,
        movement_consumed_median_v1=(
            None if tail_phase_config is None else tail_phase_config.movement_consumed_median_2024
        ),
        tail_phase_activation_status_v1=tail_phase_activation_status_v1,
        signed_market_shock_thresholds_v1=signed_market_shock_thresholds_v1,
        signed_market_shock_activation_status_v1=(signed_market_shock_activation_status_v1),
        opening_transition_thresholds_v1=opening_transition_thresholds_v1,
        opening_transition_activation_status_v1=(opening_transition_activation_status_v1),
        opening_reversal_activation_v1=opening_reversal_activation_v1,
        opening_reversal_activation_v1_1=(opening_reversal_activation_v1_1),
    )
    controller_budget = SubscriptionBudgetManager(
        limits={
            SubscriptionKind.LEVEL1: len(identity.symbols) + 1,
            SubscriptionKind.BAR: len(identity.symbols) + len(proxy_symbols),
            SubscriptionKind.TICK_BY_TICK: min(
                runtime_capacity.available_tick_by_tick,
                config.ibkr.tick_by_tick_active_underlyings * 2,
            ),
            SubscriptionKind.DEPTH: min(
                runtime_capacity.available_depth,
                config.ibkr.level2_active_underlyings,
            ),
            SubscriptionKind.OPTION: min(
                config.ibkr.max_option_subscriptions,
                config.ibkr.max_active_option_episodes * config.ibkr.max_option_lines_per_episode,
            ),
            SubscriptionKind.MARKET_PROXY: 0,
        },
        request_rate_limit=config.ibkr.request_rate_per_second,
        total_line_limit=int(runtime_capacity.total_level1_allowance.value),
        externally_reserved_lines=int(runtime_capacity.externally_reserved_lines.value),
        preexisting_internal_lines=runtime_capacity.current_internal_level1_lines,
        future_trading_reserve_lines=int(runtime_capacity.reserved_future_trading_lines.value),
        safety_margin_lines=int(runtime_capacity.safety_margin_lines.value),
    )
    opening_reversal_capacity = (
        None
        if opening_reversal_activation_v1 is None
        else OpeningReversalCapacityCoordinatorV1(
            budget=controller_budget,
        )
    )
    if opening_reversal_capacity is not None:

        def persist_pre_receipt_capacity_snapshot(
            metadata: EvidenceMetadata,
        ) -> str:
            snapshot = opening_reversal_capacity.snapshot(
                observed_at_utc=metadata.recorded_at_utc,
                promoted_episode_id=None,
            )
            frozen_repository.record_opening_reversal_capacity_snapshot_v1(
                metadata,
                snapshot,
            )
            return snapshot.snapshot_hash

        engine.set_opening_reversal_capacity_snapshot_provider_v1(
            persist_pre_receipt_capacity_snapshot
        )
    live = FrozenM1CLiveRecorder(
        adapter=adapter,
        normalizer=normalizer,
        raw_store=raw_store,
        repository=frozen_repository,
        engine=engine,
        activity_baseline=activity,
        group_o_provider=group_o_provider,
        metadata_factory=metadata_factory,
        run_id=config.runtime.run_id,
        universe_symbols=identity.symbols,
        market_proxy_symbol=MARKET_PROXY,
        sector_proxy_by_symbol=SECTOR_PROXY_BY_SYMBOL,
        readiness=ScientificReadiness(
            m1c_parity_passed=m1c_parity,
            direction_parity_passed=direction_parity,
            bar_compatibility_passed=bar_compatibility,
            historical_activity_baseline_available=historical_activity_available,
            clock_drift_within_tolerance=True,
        ),
        maximum_quote_age=timedelta(seconds=config.ibkr.maximum_quote_age_seconds),
        maximum_clock_drift_seconds=config.ibkr.maximum_clock_drift_seconds,
        depth_rows=config.ibkr.level2_rows,
        durable_inbox=durable_inbox,
        recorder_generation=recorder_generation,
        lease_owner=recorder_owner_id,
        inbox_lease_timeout=timedelta(seconds=config.runtime.callback_inbox_lease_seconds),
        inbox_batch_limit=config.runtime.callback_inbox_batch_limit,
        operational_repository=operational_repository,
        operational_thresholds=operational_thresholds(config),
        processing_heartbeat=heartbeat,
    )

    controller = LiveSubscriptionController(
        adapter=adapter,
        budget=controller_budget,
        normalizer=normalizer,
        repository=frozen_repository,
        depth_rows=config.ibkr.level2_rows,
        enable_depth=(config.ibkr.enable_level2 and config.ibkr.level2_active_underlyings > 0),
        depth_phase_permitted=lambda metadata: (
            prospective_phase_at(metadata.recorded_at_utc)[0] != "engineering_transfer"
        ),
        stream_registration_sink=live.register_stream,
        request_pacer=pace_request,
        historical_request_pacer=historical_request_pacer.acquire,
    )

    for qualified_item in qualified:
        contract = qualified_item.upstream_contract
        repository.record_underlying_contract(
            UnderlyingContractInput(
                metadata=initial_metadata,
                symbol=qualified_item.symbol,
                con_id=qualified_item.con_id,
                exchange=qualified_item.exchange,
                currency=str(_attribute(contract, "currency") or "USD"),
                local_symbol=(
                    None
                    if _attribute(contract, "localSymbol", "local_symbol") is None
                    else str(_attribute(contract, "localSymbol", "local_symbol"))
                ),
                qualification_status="qualified_exact",
                rejection_reason=None,
            )
        )
    repository.register_universe_membership(
        initial_metadata,
        symbols=identity.symbols,
        operational_status_by_symbol=statuses,
    )
    if operational_repository is not None:
        assert recorder_generation is not None
        activation_receipt_identity = _activation_receipt_identity(activation)
        loaded_at = datetime.now(UTC)
        for artifact_name, artifact_path in sorted(artifact_files.items()):
            expected_hash = activation.model_artifact_hashes[artifact_name]
            observed_hash = _sha256(artifact_path)
            verified = observed_hash == expected_hash
            operational_repository.record_artifact_verification(
                RuntimeArtifactVerification(
                    verification_id=stable_artifact_verification_id(
                        run_id=config.runtime.run_id,
                        recorder_generation=recorder_generation,
                        artifact_name=artifact_name,
                        expected_hash=expected_hash,
                    ),
                    run_id=config.runtime.run_id,
                    recorder_generation=recorder_generation,
                    artifact_bundle_id=m1c_runtime.model_hash,
                    artifact_name=artifact_name,
                    expected_hash=expected_hash,
                    observed_hash=observed_hash,
                    feature_contract_version=BUDGET_AWARE_RECORDER_CONTRACT_VERSION,
                    activation_receipt_identity=activation_receipt_identity,
                    found=True,
                    loaded=True,
                    schema_validated=True,
                    hash_verified=verified,
                    contract_compatible=True,
                    used_by_active_generation=True,
                    load_timestamp_utc=loaded_at,
                    verification_result="verified" if verified else "blocked",
                    blocker=(None if verified else "blocked_frozen_artifact_hash_mismatch"),
                    details={
                        "expected_hash_source": "immutable_activation_receipt",
                        "application_wiring_completed": False,
                        "static_verification_completed_before_subscription_start": True,
                        "activation_git_commit": activation.git_sha,
                        "runtime_git_commit": config.runtime.git_commit,
                        "activation_app_version": first_activation_app_version,
                        "runtime_app_version": config.runtime.app_version,
                    },
                )
            )
    controller.start_always_on(
        initial_metadata,
        tuple(qualified),
        required_level1_symbols=frozenset((*identity.symbols, MARKET_PROXY)),
    )
    # These callbacks are prospective evidence and must be requested only after
    # the immutable activation boundary and callback normalizer exist.
    adapter.request_current_time()
    adapter.request_depth_exchanges()
    option_recorder = BoundedOptionRecorder(
        adapter=adapter,
        subscriptions=controller_budget,
        repository=frozen_repository,
        raw_store=raw_store,
        maximum_quote_age=timedelta(seconds=config.ibkr.maximum_quote_age_seconds),
        stream_registration_sink=live.register_stream,
        stream_unregistration_sink=normalizer.unregister,
        request_pacer=pace_request,
        underlying_path_provider=live.underlying_price_path,
        underlying_quote_provider=live.underlying_quote_path,
        underlying_halt_provider=live.underlying_halted_in_window,
        configured_commission_per_contract=(config.ibkr.option_commission_per_contract),
        configured_regulatory_fee_per_contract=(config.ibkr.option_regulatory_fee_per_contract),
        configured_exchange_fee_per_contract=(config.ibkr.option_exchange_fee_per_contract),
    )
    live.option_quote_sink = option_recorder.record_quote

    def cancel_evicted_subscription(
        evicted_key: str,
        replacement_key: str,
        observed_at: datetime,
    ) -> bool:
        metadata = metadata_factory(observed_at, (observed_at,))
        try:
            handled = option_recorder.cancel_evicted_subscription(
                metadata,
                key=evicted_key,
                replacement_key=replacement_key,
            )
            if not handled:
                handled = controller.cancel_evicted_subscription(
                    metadata,
                    key=evicted_key,
                    replacement_key=replacement_key,
                )
            # An unhandled key was a pending reservation with no broker stream.
            return True
        except Exception as exc:
            frozen_repository.record_skipped_recording(
                metadata,
                session=metadata.recorded_at_utc.date(),
                episode_id=None,
                symbol=None,
                recording_kind="subscription_reconciliation_warning",
                reason="evicted_subscription_cancellation_failed",
                requested_payload={
                    "evicted_key": evicted_key,
                    "replacement_key": replacement_key,
                    "error_type": type(exc).__name__,
                },
            )
            return False

    option_recorder.eviction_sink = cancel_evicted_subscription

    def persist_episode_allocation(
        record: EpisodeAllocationRecord,
    ) -> None:
        allocation_metadata = metadata_factory(
            record.updated_at_utc,
            (record.updated_at_utc,),
        )
        frozen_repository.record_option_episode_allocation(
            allocation_metadata,
            record,
        )
        if (
            record.kind is not EpisodeKind.OPENING_REVERSAL
            or opening_reversal_capacity is None
            or record.state not in {EpisodeState.EPISODE_QUEUED, EpisodeState.DEGRADED}
        ):
            return
        snapshot = opening_reversal_capacity.snapshot(
            observed_at_utc=record.updated_at_utc,
            promoted_episode_id=record.episode_id,
        )
        frozen_repository.record_opening_reversal_capacity_snapshot_v1(
            allocation_metadata,
            snapshot,
        )
        for event in build_capacity_degradation_events_v1(record):
            frozen_repository.record_opening_reversal_degradation_v1(
                allocation_metadata,
                event,
                capacity_snapshot_hash_v1=snapshot.snapshot_hash,
            )

    episode_state = BudgetAwareEpisodeStateMachine(
        budget=controller_budget,
        max_active_episodes=config.ibkr.max_active_option_episodes,
        max_option_lines_per_episode=config.ibkr.max_option_lines_per_episode,
        max_concurrent_snapshots=max(
            1,
            min(
                config.ibkr.max_concurrent_snapshots,
                int(runtime_capacity.snapshot_pacing_limit.value),
            ),
        ),
        maximum_recording_duration=timedelta(minutes=config.ibkr.option_episode_maximum_minutes),
        persistence_sink=persist_episode_allocation,
        eviction_sink=cancel_evicted_subscription,
        phase_resolver=lambda task: prospective_phase_at(task.triggered_at_utc),
    )
    option_discovery = BoundedOptionDiscoveryService(
        adapter=adapter,
        option_recorder=option_recorder,
        budget=controller_budget,
        underlying_contracts={item.symbol: item for item in qualified if not item.market_proxy},
        contract_factory=lambda symbol, expiry, strike, right, multiplier, exchange, trading: (
            option_contract_factory(
                symbol,
                expiry,
                strike,
                right,
                multiplier,
                exchange,
                trading,
            )
        ),
        metadata_factory=metadata_factory,
        reference_quote_provider=live.first_valid_quote_at_or_after,
        run_id=config.runtime.run_id,
        strike_steps=config.ibkr.option_strike_steps,
        maximum_contracts_per_episode=(config.ibkr.maximum_option_contracts_per_episode),
        heartbeat=pace_request,
        episode_state_machine=episode_state,
        maximum_continuous_lines=config.ibkr.max_option_lines_per_episode,
    )

    def displace_option_episode(
        episode_id: str,
        replacement_episode_id: str,
        observed_at: datetime,
    ) -> None:
        metadata = metadata_factory(observed_at, (observed_at,))
        option_discovery.handle_displacement(
            episode_id,
            replacement_episode_id,
            observed_at,
        )
        controller.end_active_episode(
            metadata,
            episode_id=episode_id,
        )
        replacement_symbol = option_discovery.pending_symbol(replacement_episode_id)
        if replacement_symbol is not None:
            controller.promote_active_episode(
                metadata,
                symbol=replacement_symbol,
                episode_id=replacement_episode_id,
            )

    episode_state.displacement_sink = displace_option_episode
    source_transfer = SourceTransferCoordinator(
        repository=repository,
        frozen_repository=frozen_repository,
        run_id=config.runtime.run_id or "",
        model=m1c_runtime,
        features=m1c_features,
        activity_baseline=activity,
        group_o_provider=group_o_provider,
        metadata_factory=metadata_factory,
        aggregate_report_path=resolved_paths["aggregate_transfer_report"],
        runtime_parity_passed=m1c_parity,
        expected_symbols=identity.symbols,
        opening_reversal_enabled=(opening_reversal_activation_v1 is not None),
    )
    daily_report_writer = BudgetAwareDailyReportWriter(
        database_path=repository.database_path,
        run_id=config.runtime.run_id or "",
        report_root=resolved_paths["prospective_report_root"],
    )
    opening_leader_recorder: OpeningLeaderContinuationRecorderV0 | None = None
    if opening_leader_receipt is not None:
        opening_leader_store = OpeningLeaderEvidenceStoreV0(
            repository,
            deployment_receipt_id=opening_leader_receipt.deployment_receipt_id,
            contract_hash=opening_leader_receipt.contract_hash,
            code_hash=opening_leader_receipt.code_hash,
            cohort_hash=opening_leader_receipt.cohort_hash,
        )
        opening_leader_option_snapshotter = OpeningLeaderIBKROptionSnapshotterV0(
            adapter=adapter,
            underlying_contracts={item.symbol: item for item in qualified if not item.market_proxy},
            contract_factory=lambda symbol, expiry, strike, right, multiplier, exchange, trading: (
                option_contract_factory(
                    symbol,
                    expiry,
                    strike,
                    cast(Any, right),
                    multiplier,
                    exchange,
                    trading,
                )
            ),
            request_heartbeat=pace_request,
            maximum_quote_age_seconds=config.ibkr.maximum_quote_age_seconds,
        )

        def promote_opening_leader_underlying(
            symbol: str,
            session: date,
            checkpoint: int,
            observed: datetime,
        ) -> OpeningLeaderSelectionPromotionV0:
            selection_id = f"opening-leader-continuation-v0:{session.isoformat()}:C{checkpoint}"
            result = controller.promote_opening_leader_underlying(
                metadata_factory(
                    observed,
                    (checkpoint_timestamp_v0(session, checkpoint),),
                ),
                symbol=symbol,
                selection_id=selection_id,
            )
            return OpeningLeaderSelectionPromotionV0(
                selection_id=selection_id,
                symbol=result.symbol,
                level1_started=result.level1_started,
                approved_keys=result.approved_keys,
                denied_keys=result.denied_keys,
                budget_state=result.budget_state.value,
            )

        opening_leader_recorder = OpeningLeaderContinuationRecorderV0(
            store=opening_leader_store,
            freeze_identity=opening_leader_receipt,
            prospective_start_utc=opening_leader_receipt.freeze_completed_at_utc,
            metadata_factory=metadata_factory,
            bar_provider=live.opening_leader_checkpoint_bars,
            underlying_quote_provider=live.opening_leader_underlying_quote,
            option_snapshot_provider=opening_leader_option_snapshotter,
            rank_persistence_provider=live.opening_leader_rank_persistence,
            official_close_provider=live.opening_leader_official_close,
            selection_promotion_sink=promote_opening_leader_underlying,
            option_commission_per_contract=config.ibkr.option_commission_per_contract,
            option_regulatory_fee_per_contract=(config.ibkr.option_regulatory_fee_per_contract),
            option_exchange_fee_per_contract=config.ibkr.option_exchange_fee_per_contract,
        )
    application = FrozenProspectiveApplication(
        config=config,
        adapter=adapter,
        repository=repository,
        metadata_factory=metadata_factory,
        live_recorder=live,
        subscriptions=controller,
        subscription_budget=controller_budget,
        opening_reversal_capacity=opening_reversal_capacity,
        option_discovery=option_discovery,
        phase_manager=phase_manager,
        quiet_phase_manager=quiet_phase_manager,
        promotion_scheduler=PromotionScheduler(
            max_tick_by_tick=0,
            # Tick-by-tick and depth are active-episode-only enhancements.
            max_depth=0,
            max_level1=config.ibkr.max_high_resolution_underlyings,
            high_arming_threshold=config.ibkr.high_tail_approach_boundary,
        ),
        runtime_capacity=runtime_capacity,
        source_transfer=source_transfer,
        daily_report_writer=daily_report_writer,
        resolved_contracts=tuple(qualified),
        ibkr_api_version=ibkr_api_version,
        tws_or_gateway_version=gateway_version,
        session_context_preflight=session_context_preflight,
        opening_leader_recorder=opening_leader_recorder,
    )
    return application


__all__ = [
    "FrozenProspectiveApplication",
    "build_frozen_prospective_application",
]
