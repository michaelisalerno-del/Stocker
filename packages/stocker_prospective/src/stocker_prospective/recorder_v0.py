"""Frozen M1C checkpoint processor for the prospective record-only service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.safety import (
    EpisodeSafetyDecision,
    EpisodeSafetyInputs,
    evaluate_episode_safety,
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
    episode_decision: EpisodeDecision
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
    ) -> None:
        self.m1c_runtime = m1c_runtime
        self.m1c_features = m1c_features
        self.direction_runtime = direction_runtime
        self.direction_features = direction_features
        self.repository = repository
        self.tracker = FreshEpisodeTracker() if tracker is None else tracker
        self._restored_sessions: set[tuple[str, date]] = set()

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
        safety: EpisodeSafetyDecision | None = None
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
            episode_decision=episode,
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
