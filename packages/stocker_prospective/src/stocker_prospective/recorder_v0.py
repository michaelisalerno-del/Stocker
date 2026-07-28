"""Frozen M1C checkpoint processor for the prospective record-only service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from stocker_prospective.database import EvidenceMetadata
from stocker_prospective.direction import DirectionClassification, FrozenDirectionRuntime
from stocker_prospective.direction_features import (
    DirectionFeatureBar,
    FrozenDirectionFeatureBuilder,
)
from stocker_prospective.frozen_m1c import (
    EpisodeDecision,
    FreshEpisodeTracker,
    FrozenM1CRuntime,
    FrozenM1CScore,
)
from stocker_prospective.group_o import FrozenGroupOContext
from stocker_prospective.m1c_features import LiveFeatureBar, M1CCausalFeatureBuilder
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.quiet_state import (
    NeutralControlDecision,
    NeutralControlSampler,
    QuietEpisodeDecision,
    QuietEpisodeTracker,
    QuietStateSnapshot,
    classify_quiet_state,
)
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.safety import (
    EpisodeSafetyDecision,
    EpisodeSafetyInputs,
    evaluate_episode_safety,
)
from stocker_prospective.tail_phase_v1 import (
    MovementConsumedBarV1,
    MovementConsumedBucketV1,
    MovementConsumedStateV1,
    TailPhaseStateV1,
    TailPhaseTrackerV1,
    assign_movement_consumed_bucket_v1,
    calculate_movement_consumed_v1,
)


@dataclass(frozen=True)
class RecorderCheckpointInput:
    metadata: EvidenceMetadata
    symbol: str
    session: date
    completed_m1c_bars: tuple[LiveFeatureBar, ...]
    completed_direction_bars: tuple[DirectionFeatureBar, ...]
    group_o_context: FrozenGroupOContext
    market_data_type: MarketDataType
    capability_preflight_passed: bool
    m1c_parity_passed: bool
    direction_parity_passed: bool
    clock_drift_within_tolerance: bool
    underlying_quote_fresh: bool
    unresolved_bar_gap: bool
    raw_event_storage_writable: bool
    feature_freshness: str = "fresh"


class RecorderCheckpointResult(BaseModel):
    """One checkpoint result; classifications remain distinct from microstructure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_id: int
    score: FrozenM1CScore
    tail_phase_v1: TailPhaseStateV1
    movement_consumed_state_v1: MovementConsumedStateV1
    movement_consumed_bucket_v1: MovementConsumedBucketV1
    episode_decision: EpisodeDecision
    quiet_state: QuietStateSnapshot
    quiet_episode_decision: QuietEpisodeDecision
    quiet_observation_id: str | None
    neutral_control_decision: NeutralControlDecision
    neutral_control_id: str | None
    high_tail_control_id: str | None
    episode_safety: EpisodeSafetyDecision | None
    directional_classifications: dict[str, DirectionClassification]
    direction_display_allowed: bool
    rejection_reasons: tuple[str, ...]


class FrozenM1CRecorderEngine:
    """Construct, score, eventise, classify, and persist one completed checkpoint."""

    def __init__(
        self,
        *,
        m1c_runtime: FrozenM1CRuntime,
        m1c_features: M1CCausalFeatureBuilder,
        direction_runtime: FrozenDirectionRuntime,
        direction_features: FrozenDirectionFeatureBuilder,
        repository: FrozenRecorderRepository,
        tracker: FreshEpisodeTracker | None = None,
        quiet_tracker: QuietEpisodeTracker | None = None,
        neutral_sampler: NeutralControlSampler | None = None,
        tail_phase_tracker_v1: TailPhaseTrackerV1 | None = None,
        movement_consumed_median_v1: float | None = None,
        tail_phase_activation_status_v1: str = "not_configured",
    ) -> None:
        self.m1c_runtime = m1c_runtime
        self.m1c_features = m1c_features
        self.direction_runtime = direction_runtime
        self.direction_features = direction_features
        self.repository = repository
        self.tracker = FreshEpisodeTracker() if tracker is None else tracker
        self.quiet_tracker = QuietEpisodeTracker() if quiet_tracker is None else quiet_tracker
        self.neutral_sampler = (
            NeutralControlSampler() if neutral_sampler is None else neutral_sampler
        )
        self.tail_phase_tracker_v1 = (
            TailPhaseTrackerV1() if tail_phase_tracker_v1 is None else tail_phase_tracker_v1
        )
        self.movement_consumed_median_v1 = movement_consumed_median_v1
        self.tail_phase_activation_status_v1 = tail_phase_activation_status_v1
        self._restored_sessions: set[tuple[str, date]] = set()
        self._high_tail_episode_timestamps: dict[
            tuple[str, date],
            list[datetime],
        ] = {}

    def _restore_session(self, item: RecorderCheckpointInput) -> None:
        key = (item.symbol, item.session)
        if key in self._restored_sessions:
            return
        previous_probability, previous_episode, count = self.repository.session_episode_state(
            run_id=item.metadata.run_id,
            symbol=item.symbol,
            session=item.session,
        )
        self.tracker.restore_session(
            symbol=item.symbol,
            session=item.session,
            previous_eligible_probability=previous_probability,
            previous_episode_timestamp=previous_episode,
            episode_count=count,
        )
        quiet_probability, quiet_episode, quiet_count = self.repository.quiet_session_state(
            run_id=item.metadata.run_id,
            symbol=item.symbol,
            session=item.session,
        )
        self.quiet_tracker.restore_session(
            symbol=item.symbol,
            session=item.session,
            previous_eligible_probability=quiet_probability,
            previous_episode_timestamp=quiet_episode,
            episode_count=quiet_count,
        )
        for (
            checkpoint,
            timestamp,
            probability,
            eligible,
            invalid_reason,
        ) in self.repository.session_tail_phase_history(
            run_id=item.metadata.run_id,
            symbol=item.symbol,
            session=item.session,
        ):
            self.tail_phase_tracker_v1.evaluate(
                symbol=item.symbol,
                session=item.session,
                checkpoint=checkpoint,
                causal_timestamp=timestamp,
                probability=probability,
                valid=eligible,
                invalid_reason=invalid_reason,
            )
        if previous_episode is not None:
            self._high_tail_episode_timestamps.setdefault(key, []).append(previous_episode)
        self._restored_sessions.add(key)

    def process_checkpoint(
        self,
        item: RecorderCheckpointInput,
    ) -> RecorderCheckpointResult:
        """Process only a final completed bar; no broker mutation is reachable."""

        self._restore_session(item)
        checkpoint = len(item.completed_m1c_bars)
        if not item.completed_m1c_bars:
            raise ValueError("M1C requires completed bars")
        trigger = item.completed_m1c_bars[-1]
        if (
            trigger.symbol != item.symbol
            or trigger.session != item.session
            or not trigger.finalised
        ):
            raise ValueError("trigger bar identity or finalisation is invalid")
        if (
            item.group_o_context.symbol != item.symbol
            or item.group_o_context.signal_session != item.session
        ):
            raise ValueError("Group O context identity differs from checkpoint")

        causal = self.m1c_features.build(
            symbol=item.symbol,
            checkpoint=checkpoint,
            completed_bars=item.completed_m1c_bars,
        )
        score = self.m1c_runtime.score(
            symbol=item.symbol,
            checkpoint=checkpoint,
            group_o_context=item.group_o_context.features,
            causal_group_i=causal.scaled_features,
        )
        missing_group_o_features = self.m1c_runtime.missing_group_o_features(
            item.group_o_context.features
        )
        rejection_reasons = tuple(
            dict.fromkeys(
                (
                    *item.group_o_context.rejection_reasons,
                    *(
                        ()
                        if not missing_group_o_features
                        else ("group_o_feature_keys_missing:" + ",".join(missing_group_o_features),)
                    ),
                    *(
                        ()
                        if item.capability_preflight_passed
                        else ("ibkr_capability_preflight_failed",)
                    ),
                    *(() if item.m1c_parity_passed else ("m1c_parity_failed",)),
                    *(
                        ()
                        if item.market_data_type is MarketDataType.LIVE
                        else ("market_data_not_live",)
                    ),
                    *(
                        ()
                        if item.clock_drift_within_tolerance
                        else ("clock_drift_outside_tolerance",)
                    ),
                    *(() if item.underlying_quote_fresh else ("underlying_quote_stale",)),
                    *(() if trigger.finalised else ("trigger_bar_incomplete",)),
                    *(() if not item.unresolved_bar_gap else ("unresolved_bar_gap",)),
                    *(
                        ()
                        if item.raw_event_storage_writable
                        else ("raw_event_storage_not_writable",)
                    ),
                )
            )
        )
        signal_inputs_eligible = (
            item.capability_preflight_passed
            and item.m1c_parity_passed
            and item.group_o_context.eligible
            and not missing_group_o_features
            and item.market_data_type is MarketDataType.LIVE
            and item.clock_drift_within_tolerance
            and item.underlying_quote_fresh
            and trigger.finalised
            and not item.unresolved_bar_gap
            and item.raw_event_storage_writable
        )
        tail_phase_v1 = self.tail_phase_tracker_v1.evaluate(
            symbol=item.symbol,
            session=item.session,
            checkpoint=checkpoint,
            causal_timestamp=trigger.bar_complete_timestamp,
            probability=score.probability,
            valid=signal_inputs_eligible,
            invalid_reason=(None if not rejection_reasons else ";".join(rejection_reasons)),
        )
        movement_consumed_state_v1 = calculate_movement_consumed_v1(
            symbol=item.symbol,
            session=item.session,
            checkpoint=checkpoint,
            completed_bars=tuple(
                MovementConsumedBarV1(
                    symbol=bar.symbol,
                    session=bar.session,
                    bar_ordinal=bar.bar_ordinal,
                    bar_start_timestamp=bar.bar_start_timestamp,
                    bar_complete_timestamp=bar.bar_complete_timestamp,
                    high=bar.high,
                    low=bar.low,
                    finalised=bar.finalised,
                )
                for bar in item.completed_m1c_bars
            ),
            previous_close_implied_movement_15m=(
                item.group_o_context.previous_close_implied_movement_15m
            ),
        )
        movement_consumed_bucket_v1 = assign_movement_consumed_bucket_v1(
            movement_consumed_state_v1.movement_consumed_v1,
            frozen_median=self.movement_consumed_median_v1,
        )
        checkpoint_id = self.repository.record_checkpoint(
            item.metadata,
            symbol=item.symbol,
            session=item.session,
            checkpoint=checkpoint,
            bar_start_utc=trigger.bar_start_timestamp,
            bar_end_utc=trigger.bar_complete_timestamp,
            score=score,
            session_context_hash=item.group_o_context.context_hash,
            feature_values={
                **dict(zip(score.feature_order, score.feature_values, strict=True)),
                "_causal_builder_feature_hash": causal.feature_hash,
                "_scaling_artifact_hash": causal.scaling_artifact_hash,
            },
            eligible=signal_inputs_eligible,
            feature_freshness=item.feature_freshness,
            rejection_reasons=rejection_reasons,
            tail_phase_v1=tail_phase_v1,
            movement_consumed_v1=movement_consumed_state_v1,
            movement_consumed_bucket_v1=movement_consumed_bucket_v1,
            movement_consumed_frozen_median_v1=self.movement_consumed_median_v1,
            tail_phase_activation_status_v1=self.tail_phase_activation_status_v1,
        )
        key = (item.symbol, item.session)
        previous_high_tail_nearby = any(
            0.0 < (trigger.bar_complete_timestamp - high_timestamp).total_seconds() / 60.0 <= 60.0
            for high_timestamp in self._high_tail_episode_timestamps.get(key, ())
        )
        quiet_episode = self.quiet_tracker.evaluate(
            symbol=item.symbol,
            session=item.session,
            checkpoint=checkpoint,
            trigger_bar_end=trigger.bar_complete_timestamp,
            probability=score.probability,
            eligible=signal_inputs_eligible,
            data_quality_flags=rejection_reasons,
            previous_high_tail_within_60_minutes=previous_high_tail_nearby,
            rejection_reason=(None if not rejection_reasons else ";".join(rejection_reasons)),
        )
        quiet_state = classify_quiet_state(
            probability=score.probability,
            previous_probability=quiet_episode.previous_probability,
            model_hash=score.model_hash,
            feature_hash=score.feature_hash,
            data_quality_status=("valid" if signal_inputs_eligible else "invalid"),
            data_quality_flags=rejection_reasons,
        )
        quiet_checkpoint_id = self.repository.record_quiet_checkpoint(
            item.metadata,
            checkpoint_id=checkpoint_id,
            symbol=item.symbol,
            session=item.session,
            checkpoint=checkpoint,
            snapshot=quiet_state,
            eligible=signal_inputs_eligible,
        )
        quiet_observation_id: str | None = None
        if quiet_episode.fresh_episode:
            quiet_observation_id = self.repository.record_quiet_episode(
                item.metadata,
                quiet_checkpoint_id=quiet_checkpoint_id,
                decision=quiet_episode,
                scientific_recording_valid=signal_inputs_eligible,
            )
        neutral_control = self.neutral_sampler.evaluate(
            session=item.session,
            symbol=item.symbol,
            checkpoint=checkpoint,
            model_hash=score.model_hash,
            probability=score.probability,
            eligible=signal_inputs_eligible,
        )
        neutral_control_id: str | None = None
        if neutral_control.selected:
            neutral_control_id = self.repository.record_neutral_control(
                item.metadata,
                quiet_checkpoint_id=quiet_checkpoint_id,
                decision=neutral_control,
                trigger_timestamp=trigger.bar_complete_timestamp,
                data_quality_flags=rejection_reasons,
            )
        episode = self.tracker.evaluate(
            symbol=item.symbol,
            session=item.session,
            checkpoint=checkpoint,
            trigger_bar_end=trigger.bar_complete_timestamp,
            probability=score.probability,
            eligible=signal_inputs_eligible,
            rejection_reason=(None if not rejection_reasons else ";".join(rejection_reasons)),
        )
        if episode.fresh_episode:
            self.repository.mark_following_high_tail_proximity(
                run_id=item.metadata.run_id,
                symbol=item.symbol,
                session=item.session,
                high_tail_timestamp=trigger.bar_complete_timestamp,
            )
            self._high_tail_episode_timestamps.setdefault(key, []).append(
                trigger.bar_complete_timestamp
            )
        safety: EpisodeSafetyDecision | None = None
        high_tail_control_id: str | None = None
        classifications: dict[str, DirectionClassification] = {}
        display_allowed = False
        if episode.fresh_episode:
            safety = evaluate_episode_safety(
                EpisodeSafetyInputs(
                    capability_preflight_passed=item.capability_preflight_passed,
                    m1c_parity_passed=item.m1c_parity_passed,
                    direction_parity_passed=item.direction_parity_passed,
                    market_data_type=item.market_data_type,
                    previous_close_group_o_valid=item.group_o_context.eligible,
                    trigger_bar_complete=trigger.finalised,
                    clock_drift_within_tolerance=item.clock_drift_within_tolerance,
                    underlying_quote_fresh=item.underlying_quote_fresh,
                    unresolved_bar_gap=item.unresolved_bar_gap,
                    deterministic_episode_identity=episode.episode_id is not None,
                    raw_event_storage_writable=item.raw_event_storage_writable,
                )
            )
            episode_id = self.repository.record_episode(
                item.metadata,
                checkpoint_id=checkpoint_id,
                decision=episode,
                safety=safety,
            )
            high_tail_control_id = self.repository.record_high_tail_control(
                item.metadata,
                quiet_checkpoint_id=quiet_checkpoint_id,
                decision=episode,
                scientific_recording_valid=safety.scientific_recording_valid,
                data_quality_flags=tuple(
                    dict.fromkeys((*rejection_reasons, *safety.rejection_reasons))
                ),
            )
            if item.direction_parity_passed and item.market_data_type is MarketDataType.LIVE:
                direction_features = self.direction_features.build(
                    symbol=item.symbol,
                    checkpoint=checkpoint,
                    completed_bars=item.completed_direction_bars,
                )
                classifications = self.direction_runtime.classify(
                    raw_features=direction_features.raw_features,
                    symbol=item.symbol,
                    checkpoint=checkpoint,
                    checkpoint_category=direction_features.checkpoint_category,
                    day_of_week=direction_features.day_of_week,
                )
                self.repository.record_directions(
                    item.metadata,
                    episode_id=episode_id,
                    features=direction_features,
                    classifications=classifications,
                    valid=safety.scientific_recording_valid,
                )
                display_allowed = True
        all_reasons = tuple(
            dict.fromkeys(
                (
                    *rejection_reasons,
                    *(() if safety is None else safety.rejection_reasons),
                )
            )
        )
        return RecorderCheckpointResult(
            checkpoint_id=checkpoint_id,
            score=score,
            tail_phase_v1=tail_phase_v1,
            movement_consumed_state_v1=movement_consumed_state_v1,
            movement_consumed_bucket_v1=movement_consumed_bucket_v1,
            episode_decision=episode,
            quiet_state=quiet_state,
            quiet_episode_decision=quiet_episode,
            quiet_observation_id=quiet_observation_id,
            neutral_control_decision=neutral_control,
            neutral_control_id=neutral_control_id,
            high_tail_control_id=high_tail_control_id,
            episode_safety=safety,
            directional_classifications=classifications,
            direction_display_allowed=display_allowed,
            rejection_reasons=all_reasons,
        )


__all__ = [
    "FrozenM1CRecorderEngine",
    "RecorderCheckpointInput",
    "RecorderCheckpointResult",
]
