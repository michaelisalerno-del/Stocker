"""Construction and lifecycle of the live frozen-M1C recorder application."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from stocker_prospective.activation import ProspectiveActivationLedger
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
    resolve_runtime_capacity,
)
from stocker_prospective.config import ProspectiveConfig
from stocker_prospective.context import previous_xnys_session
from stocker_prospective.contract import (
    BUDGET_AWARE_RECORDER_CONTRACT_VERSION,
    M1C_FEATURE_MANIFEST_SHA256,
    M1C_SCALING_ARTIFACT_SHA256,
    M1C_THRESHOLD_ARTIFACT_SHA256,
    SECTOR_PROXY_BY_SYMBOL,
)
from stocker_prospective.database import (
    EvidenceMetadata,
    ProspectiveRepository,
    UnderlyingContractInput,
)
from stocker_prospective.direction import FrozenDirectionRuntime
from stocker_prospective.direction_features import FrozenDirectionFeatureBuilder
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
from stocker_prospective.market_data import ConnectionState, MarketDataType
from stocker_prospective.option_budget import BudgetAwareEpisodeStateMachine
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
from stocker_prospective.source_transfer import SourceTransferCoordinator
from stocker_prospective.subscriptions import (
    PromotionScheduler,
    SubscriptionBudgetManager,
    SubscriptionKind,
)

NEW_YORK = ZoneInfo("America/New_York")
MARKET_PROXY = "VTI"
SECTOR_PROXY_SYMBOLS = tuple(sorted(set(SECTOR_PROXY_BY_SYMBOL.values())))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _passed(path: Path, *, label: str) -> bool:
    if not path.is_file():
        raise ValueError(f"{label} report is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("passed") is True


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
        self._session_reports_written: set[date] = set()
        self._session_context_checked: set[date] = set()
        self._session_context_failures: dict[date, str] = {}
        self._reconnect_attempts = 0
        self._next_reconnect_at: datetime | None = None
        self._last_reconciliation_monotonic = 0.0

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

    def poll(self, *, now: datetime) -> LivePollResult:
        observed = now.astimezone(UTC)
        observed_session = observed.astimezone(NEW_YORK).date()
        try:
            xnys_session_bounds(observed_session)
        except ValueError:
            pass
        else:
            self._sessions_seen.add(observed_session)
            if observed_session not in self._session_context_checked:
                try:
                    self.session_context_preflight(observed_session, observed)
                except ValueError as exc:
                    self._session_context_failures[observed_session] = str(exc)
                self._session_context_checked.add(observed_session)
        self._recover_if_required(observed)
        result = self.live_recorder.poll(now=observed)
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
        for checkpoint in result.checkpoint_results:
            symbol = checkpoint.episode_decision.symbol
            self._sessions_seen.add(checkpoint.episode_decision.session)
            self._probabilities[symbol] = checkpoint.score.probability
            if not checkpoint.rejection_reasons:
                self._eligible_symbols.add(symbol)
            else:
                self._eligible_symbols.discard(symbol)
            if checkpoint.episode_decision.fresh_episode:
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
            self.option_discovery.schedule_quiet_state(checkpoint)
            quiet_observations: tuple[
                tuple[str | None, QuietObservationKind],
                ...,
            ] = (
                (checkpoint.quiet_observation_id, "quiet_bottom_10"),
                (checkpoint.neutral_control_id, "neutral_control"),
                (checkpoint.high_tail_control_id, "high_tail_control"),
            )
            for observation_id, kind in quiet_observations:
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
        if result.checkpoint_results:
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
        return result

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
                self.live_recorder.mark_gap(item.symbol, started_at=now)
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
) -> FrozenProspectiveApplication:
    """Build the live service only after all frozen artifact identities verify."""

    paths = config.paths
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
    if any(not path.is_file() for path in artifact_files.values()):
        absent = sorted(name for name, path in artifact_files.items() if not path.is_file())
        raise ValueError("frozen recorder artifact absent: " + ",".join(absent))
    _assert_frozen_m1c_artifact_hashes(artifact_files)
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
    m1c_parity = _passed(
        resolved_paths["m1c_live_parity_report"],
        label="M1C live parity",
    )
    direction_parity = _passed(
        resolved_paths["direction_live_parity_report"],
        label="direction live parity",
    )
    bar_compatibility = _passed(
        resolved_paths["bar_compatibility_report"],
        label="bar compatibility",
    )
    gateway_version = config.ibkr.tws_or_gateway_version
    if not gateway_version:
        raise ValueError("IBKR_TWS_OR_GATEWAY_VERSION is required at activation")
    configuration_hash = hashlib.sha256(
        json.dumps(
            config.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

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
    capacity_observer = getattr(adapter, "discover_market_data_capacity", None)
    discovered_capacity = (
        capacity_observer()
        if callable(capacity_observer)
        else CapacityDiscovery(market_data_status="live")
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
        ),
        discovery=discovered_capacity,
        output_path=capacity_manifest_path,
    )

    activation_ledger = ProspectiveActivationLedger(resolved_paths["recorder_activation"])
    existing_activation = activation_ledger.load()
    activation = activation_ledger.activate(
        activation_timestamp_utc=(
            datetime.now(UTC)
            if existing_activation is None
            else existing_activation.prospective_collection_start_utc
        ),
        git_sha=config.runtime.git_commit,
        model_artifact_hashes={
            name: _sha256(path) for name, path in sorted(artifact_files.items())
        },
        configuration_hash=configuration_hash,
        ibkr_api_version=ibkr_api_version,
        tws_or_gateway_version=gateway_version,
    )
    prospective_start = activation.prospective_collection_start_utc

    def metadata_factory(
        observed_at: datetime,
        source_timestamps: tuple[datetime, ...],
    ) -> EvidenceMetadata:
        return EvidenceMetadata(
            run_id=config.runtime.run_id or "",
            prospective_start_utc=prospective_start,
            app_version=config.runtime.app_version,
            git_commit=config.runtime.git_commit,
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
        configuration_hash=configuration_hash,
    )

    def prospective_phase_at(observed_at: datetime) -> tuple[str, bool]:
        return frozen_repository.prospective_phase_for_session(
            run_id=config.runtime.run_id or "",
            session=observed_at.astimezone(NEW_YORK).date(),
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
    activity = HistoricalActivityBaseline.from_parquet(
        resolved_paths["historical_activity_bars"],
        latest_authorised_session=previous_xnys_session(session),
    )
    normalizer = IBKRCallbackNormalizer(prospective_collection_start=prospective_start)
    raw_store = PartitionedEventStore(
        root=resolved_paths["raw_event_root"],
        prospective_collection_start=prospective_start,
        recorder_version=config.runtime.app_version,
        contract_version=BUDGET_AWARE_RECORDER_CONTRACT_VERSION,
    )
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
            package = load_group_o_session_package(
                context_root=resolved_paths["context_root"],
                signal_session=signal_session,
            )
            group_packages[signal_session] = package
        package_symbols = {context.symbol for context in package.contexts}
        if package_symbols != set(identity.symbols):
            raise ValueError("Group O package does not contain the exact frozen cohort")
        metadata = metadata_factory(observed_at, (observed_at,))
        for symbol in identity.symbols:
            context = package.for_symbol(symbol)
            missing_features = m1c_runtime.missing_group_o_features(context.features)
            if missing_features:
                raise ValueError(
                    "Group O context missing frozen feature keys for "
                    f"{symbol}: {','.join(missing_features)}"
                )
            frozen_repository.record_group_o_context(metadata, context)

    engine = FrozenM1CRecorderEngine(
        m1c_runtime=m1c_runtime,
        m1c_features=m1c_features,
        direction_runtime=direction_runtime,
        direction_features=direction_features,
        repository=frozen_repository,
    )
    controller_budget = SubscriptionBudgetManager(
        limits={
            SubscriptionKind.LEVEL1: config.ibkr.max_high_resolution_underlyings,
            SubscriptionKind.BAR: len(identity.symbols) + len(proxy_symbols),
            SubscriptionKind.TICK_BY_TICK: min(
                int(runtime_capacity.tick_by_tick_capacity.value),
                config.ibkr.tick_by_tick_active_underlyings * 2,
            ),
            SubscriptionKind.DEPTH: min(
                int(runtime_capacity.depth_capacity.value),
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
        future_trading_reserve_lines=int(runtime_capacity.reserved_future_trading_lines.value),
        safety_margin_lines=int(runtime_capacity.safety_margin_lines.value),
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
        universe_symbols=identity.symbols,
        market_proxy_symbol=MARKET_PROXY,
        sector_proxy_by_symbol=SECTOR_PROXY_BY_SYMBOL,
        readiness=ScientificReadiness(
            m1c_parity_passed=m1c_parity,
            direction_parity_passed=direction_parity,
            bar_compatibility_passed=bar_compatibility,
            clock_drift_within_tolerance=True,
        ),
        maximum_quote_age=timedelta(seconds=config.ibkr.maximum_quote_age_seconds),
        maximum_clock_drift_seconds=config.ibkr.maximum_clock_drift_seconds,
        depth_rows=config.ibkr.level2_rows,
    )

    controller = LiveSubscriptionController(
        adapter=adapter,
        budget=controller_budget,
        normalizer=normalizer,
        repository=frozen_repository,
        depth_rows=config.ibkr.level2_rows,
        enable_depth=(config.ibkr.enable_level2 and config.ibkr.level2_active_underlyings > 0),
        stream_registration_sink=live.register_stream,
        request_pacer=pace_request,
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
    controller.start_always_on(initial_metadata, tuple(qualified))
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
        underlying_halt_provider=live.underlying_halted_in_window,
    )
    live.option_quote_sink = option_recorder.record_quote
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
        persistence_sink=lambda record: frozen_repository.record_option_episode_allocation(
            metadata_factory(record.updated_at_utc, (record.updated_at_utc,)),
            record,
        ),
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
    )
    daily_report_writer = BudgetAwareDailyReportWriter(
        database_path=repository.database_path,
        run_id=config.runtime.run_id or "",
        report_root=resolved_paths["prospective_report_root"],
    )
    return FrozenProspectiveApplication(
        config=config,
        adapter=adapter,
        repository=repository,
        metadata_factory=metadata_factory,
        live_recorder=live,
        subscriptions=controller,
        subscription_budget=controller_budget,
        option_discovery=option_discovery,
        phase_manager=phase_manager,
        quiet_phase_manager=quiet_phase_manager,
        promotion_scheduler=PromotionScheduler(
            max_tick_by_tick=0,
            max_depth=config.ibkr.level2_active_underlyings,
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
    )


__all__ = [
    "FrozenProspectiveApplication",
    "build_frozen_prospective_application",
]
