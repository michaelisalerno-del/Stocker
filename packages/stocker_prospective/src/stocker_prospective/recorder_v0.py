"""Frozen M1C checkpoint processor for the prospective record-only service."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

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
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningReversalActivationReceiptV1,
    OpeningReversalPredictionInputV1,
    OpeningReversalPredictionReceiptV1,
    OpeningReversalPredictionTimingEvidenceV1_1,
    build_prediction_receipt_v1,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1_1 import (
    OpeningReversalActivationReceiptV1_1,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.opening_market_transition_v1 import (
    EXPECTED_OPENING_BAR_COUNT_V1,
    OPENING_MARKET_PROXY_V1,
    OpeningMarketTransitionStateResultV1,
    OpeningPreEntryWindowV1,
    OpeningTransitionThresholdsV1,
    StockOpeningResponseResultV1,
    calculate_opening_preentry_window_v1,
    calculate_stock_opening_response_v1,
    classify_opening_market_transition_v1,
)
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
from stocker_prospective.signed_market_shock_v1 import (
    MARKET_SHOCK_PROXY_V1,
    CheckpointShockThresholdsV1,
    MarketShockBarV1,
    MarketShockStateResultV1,
    PreentryMarketWindowsV1,
    SignedMarketShockThresholdManifestV1,
    StockShockResponseResultV1,
    calculate_preentry_windows_v1,
    calculate_stock_shock_response_v1,
    classify_market_shock_state_v1,
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


def _comparison_sign(value: float | None) -> int | None:
    if value is None:
        return None
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def _action_sign(action: object | None) -> int | None:
    if action is None:
        return None
    text = str(action)
    return 1 if text == "CALL" else -1 if text == "PUT" else 0


def _existing_clean_market_direction_baseline_sign_v1(
    bars: tuple[DirectionFeatureBar, ...],
    *,
    checkpoint: int = 6,
) -> int | None:
    """Exact inherited T-1 market return over ordinals checkpoint-3/-2."""

    expected = (checkpoint - 3, checkpoint - 2)
    selected = tuple(
        sorted(
            (bar for bar in bars if bar.bar_ordinal in expected),
            key=lambda value: value.bar_ordinal,
        )
    )
    if not (
        len(selected) == 2
        and tuple(bar.bar_ordinal for bar in selected) == expected
        and all(bar.finalised for bar in selected)
        and selected[0].session == selected[1].session
        and all(math.isfinite(bar.market_log_return) for bar in selected)
        and selected[0].bar_complete_timestamp == selected[1].bar_start_timestamp
    ):
        return None
    return _comparison_sign(sum(bar.market_log_return for bar in selected))


def _opening_reversal_baseline_comparisons_v1(
    *,
    item: RecorderCheckpointInput,
    stock_opening_response_v1: StockOpeningResponseResultV1,
    classifications: dict[str, DirectionClassification],
) -> tuple[tuple[str, str | float | int | bool | None], ...]:
    """Freeze causal baseline values without allowing them into the V1 action."""

    bars = tuple(sorted(item.completed_m1c_bars, key=lambda value: value.bar_ordinal))
    recent_sign = (
        None
        if len(bars) < 6 or not bars[-1].finalised
        else _comparison_sign(bars[-1].close - bars[-1].open)
    )
    opening_sign = _comparison_sign(stock_opening_response_v1.stock_opening_return_v1)
    clean_market_sign = _existing_clean_market_direction_baseline_sign_v1(
        item.completed_direction_bars,
    )
    a1 = classifications.get("A1")
    a1_sign = _action_sign(None if a1 is None else a1.action)
    comparisons: list[tuple[str, str | float | int | bool | None]] = []
    available = {
        "most_recent_completed_five_minute_stock_momentum": recent_sign,
        "complete_stock_opening_window_momentum": opening_sign,
        "existing_clean_market_direction_baseline": clean_market_sign,
        "frozen_a1": a1_sign,
    }
    for baseline, sign in available.items():
        comparisons.append((f"baseline_{baseline}_prediction_sign_v1", sign))
        comparisons.append(
            (
                f"baseline_{baseline}_unavailable_reason_v1",
                (None if sign is not None else f"{baseline}_not_causally_available_at_receipt"),
            )
        )
    unavailable = {
        "frozen_historical_asymmetric_downside_score": (
            "frozen_historical_asymmetric_downside_score_not_available_in_live_recorder"
        ),
        "independently_frozen_microstructure_rule": (
            "no_independently_frozen_microstructure_rule_available_at_receipt"
        ),
    }
    for baseline, reason in unavailable.items():
        comparisons.extend(
            (
                (f"baseline_{baseline}_prediction_sign_v1", None),
                (f"baseline_{baseline}_unavailable_reason_v1", reason),
            )
        )
    return tuple(comparisons)


def _unknown_signed_market_shock_logging_v1(
    *,
    session: date,
    checkpoint: int,
    signal_timestamp: datetime,
    reason: str,
) -> tuple[
    PreentryMarketWindowsV1,
    MarketShockStateResultV1,
    StockShockResponseResultV1,
]:
    """Represent an optional logging failure without touching M1C control flow."""

    windows = PreentryMarketWindowsV1(
        market_proxy_v1=MARKET_SHOCK_PROXY_V1,
        session=session,
        checkpoint=checkpoint,
        signal_timestamp=signal_timestamp,
        w0_bar_ordinals_v1=(checkpoint - 3, checkpoint - 2, checkpoint - 1),
        w1_bar_ordinals_v1=(checkpoint - 6, checkpoint - 5, checkpoint - 4),
        market_return_w0_v1=None,
        market_range_w0_v1=None,
        market_return_w1_v1=None,
        market_range_w1_v1=None,
        maximum_market_timestamp_v1=None,
        complete_v1=False,
        missing_reasons_v1=(reason,),
    )
    state = MarketShockStateResultV1(
        market_shock_state_v1="UNKNOWN_INCOMPLETE",
        market_shock_event_id_v1=None,
        shock_sign_v1=None,
        complete_v1=False,
        missing_reasons_v1=(reason,),
    )
    response = StockShockResponseResultV1(
        stock_return_w0_v1=None,
        stock_absolute_alignment_v1=None,
        shock_relative_response_v1=None,
        shock_response_class_v1="UNKNOWN_INCOMPLETE",
        resisting_subtype_v1=None,
        maximum_stock_timestamp_v1=None,
        complete_v1=False,
        missing_reasons_v1=(reason,),
    )
    return windows, state, response


def _unknown_opening_market_transition_logging_v1(
    *,
    session: date,
    previous_session: date | None,
    signal_timestamp: datetime,
    reason: str,
) -> tuple[
    OpeningPreEntryWindowV1,
    OpeningMarketTransitionStateResultV1,
    StockOpeningResponseResultV1,
]:
    """Represent optional opening logging failure without changing core flow."""

    session_open = signal_timestamp - timedelta(minutes=30)
    window = OpeningPreEntryWindowV1(
        market_proxy_v1=OPENING_MARKET_PROXY_V1,
        session=session,
        previous_session_v1=(session if previous_session is None else previous_session),
        checkpoint_v1=6,
        session_open_timestamp_v1=session_open,
        signal_timestamp_v1=signal_timestamp,
        entry_timestamp_v1=signal_timestamp,
        opening_bar_ordinals_v1=tuple(range(EXPECTED_OPENING_BAR_COUNT_V1)),
        expected_opening_bar_count_v1=EXPECTED_OPENING_BAR_COUNT_V1,
        observed_opening_bar_count_v1=0,
        final_complete_pre_entry_bar_start_v1=None,
        entry_bar_ordinal_v1=EXPECTED_OPENING_BAR_COUNT_V1,
        entry_bar_included_v1=False,
        market_session_open_v1=None,
        market_prior_regular_session_close_v1=None,
        market_last_complete_pre_entry_close_v1=None,
        market_opening_return_v1=None,
        market_opening_range_v1=None,
        market_overnight_gap_v1=None,
        market_total_transition_v1=None,
        market_gap_open_alignment_v1="UNKNOWN_INCOMPLETE",
        maximum_market_timestamp_v1=None,
        complete_v1=False,
        missing_reasons_v1=(reason,),
    )
    state = OpeningMarketTransitionStateResultV1(
        opening_market_transition_state_v1="UNKNOWN_INCOMPLETE",
        opening_transition_sign_v1=None,
        opening_transition_event_id_v1=None,
        complete_v1=False,
        missing_reasons_v1=(reason,),
    )
    response = StockOpeningResponseResultV1(
        stock_opening_return_v1=None,
        stock_opening_range_v1=None,
        stock_opening_alignment_v1=None,
        stock_relative_opening_response_v1=None,
        stock_opening_response_class_v1="UNKNOWN_INCOMPLETE",
        resisting_subtype_v1=None,
        expected_opening_bar_count_v1=EXPECTED_OPENING_BAR_COUNT_V1,
        observed_opening_bar_count_v1=0,
        maximum_stock_timestamp_v1=None,
        complete_v1=False,
        missing_reasons_v1=(reason,),
    )
    return window, state, response


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
    scientific_recording_authorized: bool
    feature_freshness: str = "fresh"
    completed_market_shock_bars_v1: tuple[MarketShockBarV1, ...] = ()
    market_previous_session_v1: date | None = None
    market_prior_regular_session_close_v1: float | None = None
    opening_reversal_receipt_created_at_utc_v1_1: datetime | None = None
    opening_reversal_first_buffered_event_received_at_utc_v1_1: datetime | None = None
    opening_reversal_entry_data_admitted_before_receipt_v1_1: bool = False


class RecorderCheckpointResult(BaseModel):
    """One checkpoint result; classifications remain distinct from microstructure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_id: int
    score: FrozenM1CScore
    tail_phase_v1: TailPhaseStateV1
    movement_consumed_state_v1: MovementConsumedStateV1
    movement_consumed_bucket_v1: MovementConsumedBucketV1
    market_windows_v1: PreentryMarketWindowsV1
    market_shock_state_v1: MarketShockStateResultV1
    stock_shock_response_v1: StockShockResponseResultV1
    opening_window_v1: OpeningPreEntryWindowV1
    opening_transition_state_v1: OpeningMarketTransitionStateResultV1
    stock_opening_response_v1: StockOpeningResponseResultV1
    opening_reversal_prediction_v1: OpeningReversalPredictionReceiptV1 | None = None
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
        signed_market_shock_thresholds_v1: (SignedMarketShockThresholdManifestV1 | None) = None,
        signed_market_shock_activation_status_v1: str = "not_configured",
        opening_transition_thresholds_v1: (OpeningTransitionThresholdsV1 | None) = None,
        opening_transition_activation_status_v1: str = "not_configured",
        opening_reversal_activation_v1: (OpeningReversalActivationReceiptV1 | None) = None,
        opening_reversal_activation_v1_1: (OpeningReversalActivationReceiptV1_1 | None) = None,
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
        self.signed_market_shock_thresholds_v1 = signed_market_shock_thresholds_v1
        self.signed_market_shock_activation_status_v1 = signed_market_shock_activation_status_v1
        self.opening_transition_thresholds_v1 = opening_transition_thresholds_v1
        self.opening_transition_activation_status_v1 = opening_transition_activation_status_v1
        self.opening_reversal_activation_v1 = opening_reversal_activation_v1
        self.opening_reversal_activation_v1_1 = opening_reversal_activation_v1_1
        if opening_reversal_activation_v1_1 is not None and (
            opening_reversal_activation_v1 is None
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
            raise ValueError("opening reversal V1.1 does not supersede the configured V1")
        self.opening_reversal_capacity_snapshot_provider_v1: (
            Callable[[EvidenceMetadata], str] | None
        ) = None
        self._restored_sessions: set[tuple[str, date]] = set()
        self._high_tail_episode_timestamps: dict[
            tuple[str, date],
            list[datetime],
        ] = {}

    def set_opening_reversal_capacity_snapshot_provider_v1(
        self,
        provider: Callable[[EvidenceMetadata], str],
    ) -> None:
        """Attach the live conservative snapshot writer before checkpoint use."""

        if self.opening_reversal_capacity_snapshot_provider_v1 is not None:
            raise ValueError("opening reversal capacity snapshot provider is immutable")
        self.opening_reversal_capacity_snapshot_provider_v1 = provider

    def _opening_reversal_capacity_snapshot_id_v1(
        self,
        metadata: EvidenceMetadata,
    ) -> str | None:
        provider = self.opening_reversal_capacity_snapshot_provider_v1
        if provider is None:
            return None
        try:
            return provider(metadata)
        except Exception:
            # Core M1C collection remains live; the V1 receipt itself fails
            # closed through its explicit missing-capacity guard.
            return None

    def _build_opening_reversal_prediction_v1(
        self,
        *,
        item: RecorderCheckpointInput,
        checkpoint: int,
        trigger_timestamp: datetime,
        score: FrozenM1CScore,
        episode: EpisodeDecision,
        tail_phase_v1: TailPhaseStateV1,
        opening_window_v1: OpeningPreEntryWindowV1,
        opening_transition_state_v1: OpeningMarketTransitionStateResultV1,
        stock_opening_response_v1: StockOpeningResponseResultV1,
        classifications: dict[str, DirectionClassification],
        signal_inputs_eligible: bool,
    ) -> OpeningReversalPredictionReceiptV1:
        assert self.opening_reversal_activation_v1 is not None
        phase, transfer_status = self.repository.opening_reversal_phase_for_session(
            run_id=item.metadata.run_id,
            session=item.session,
        )
        a1 = classifications.get("A1")
        addendum = self.opening_reversal_activation_v1_1
        receipt_created_at = (
            item.metadata.recorded_at_utc
            if addendum is None
            else item.opening_reversal_receipt_created_at_utc_v1_1
        )
        if receipt_created_at is None:
            raise ValueError("V1.1 receipt creation timestamp is unavailable")
        timing_evidence = (
            None
            if addendum is None
            else OpeningReversalPredictionTimingEvidenceV1_1(
                timing_addendum_activation_receipt_hash_v1_1=(
                    addendum.activation_receipt_hash_v1_1
                ),
                rule_committed_at_utc=addendum.activation_timestamp_utc,
                causal_barrier_armed_at_utc=addendum.activation_timestamp_utc,
                predictor_window_completed_at_utc=trigger_timestamp,
                first_entry_or_post_entry_event_buffered_at_utc=(
                    item.opening_reversal_first_buffered_event_received_at_utc_v1_1
                ),
                entry_or_post_entry_data_admitted_before_receipt=(
                    item.opening_reversal_entry_data_admitted_before_receipt_v1_1
                ),
                raw_event_archive_write_before_receipt=True,
                decision_surface_release_requires_durable_receipt=True,
                nominal_entry_actionable=False,
                receipt_latency_after_nominal_entry_seconds=(
                    receipt_created_at - episode.prospective_entry_timestamp
                ).total_seconds(),
            )
        )
        return build_prediction_receipt_v1(
            OpeningReversalPredictionInputV1(
                activation_timestamp_utc=(
                    self.opening_reversal_activation_v1.activation_timestamp_utc
                    if addendum is None
                    else addendum.activation_timestamp_utc
                ),
                experiment_version="1" if addendum is None else "1.1",
                cohort_phase=phase,
                transfer_status=transfer_status,
                session=item.session,
                stock=item.symbol,
                checkpoint=checkpoint,
                signal_timestamp_utc=trigger_timestamp,
                entry_timestamp_utc=episode.prospective_entry_timestamp,
                receipt_created_at_utc=receipt_created_at,
                m1c_probability=score.probability,
                # Signal validity describes whether the frozen predictor ran on
                # complete causal inputs. Scientific admission is a separate
                # persisted property of the parent evidence.
                m1c_probability_valid=signal_inputs_eligible,
                high_tail_membership=score.threshold_passed,
                fresh_episode_id=episode.episode_id,
                canonical_fresh_episode=episode.fresh_episode,
                tail_phase_v1=tail_phase_v1.m1c_tail_phase_v1,
                market_opening_return_v1=(opening_window_v1.market_opening_return_v1),
                market_opening_range_v1=(opening_window_v1.market_opening_range_v1),
                opening_market_transition_state_v1=(
                    opening_transition_state_v1.opening_market_transition_state_v1
                ),
                opening_transition_sign_v1=(opening_transition_state_v1.opening_transition_sign_v1),
                opening_transition_event_id_v1=(
                    opening_transition_state_v1.opening_transition_event_id_v1
                ),
                vti_opening_transition_complete=(opening_transition_state_v1.complete_v1),
                stock_causal_data_complete=(stock_opening_response_v1.complete_v1),
                previous_close_atm_iv_scale_15m=(
                    item.group_o_context.previous_close_implied_movement_15m
                ),
                previous_close_atm_iv_scale_valid=(
                    item.group_o_context.eligible
                    and item.group_o_context.previous_close_implied_movement_15m is not None
                ),
                data_source="ibkr",
                capacity_snapshot_id=(
                    self._opening_reversal_capacity_snapshot_id_v1(item.metadata)
                ),
                frozen_comparisons=(
                    *_opening_reversal_baseline_comparisons_v1(
                        item=item,
                        stock_opening_response_v1=stock_opening_response_v1,
                        classifications=classifications,
                    ),
                    (
                        "a1_action_v1",
                        None if a1 is None else a1.action,
                    ),
                    (
                        "a1_probability_up_v1",
                        None if a1 is None else a1.probability_up,
                    ),
                    (
                        "stock_opening_return_v1",
                        stock_opening_response_v1.stock_opening_return_v1,
                    ),
                    (
                        "stock_opening_range_v1",
                        stock_opening_response_v1.stock_opening_range_v1,
                    ),
                    (
                        "stock_opening_alignment_v1",
                        stock_opening_response_v1.stock_opening_alignment_v1,
                    ),
                    (
                        "stock_opening_response_class_v1",
                        stock_opening_response_v1.stock_opening_response_class_v1,
                    ),
                    (
                        "stock_relative_opening_response_v1",
                        stock_opening_response_v1.stock_relative_opening_response_v1,
                    ),
                ),
                timing_evidence_v1_1=timing_evidence,
            )
        )

    def build_incomplete_opening_reversal_prediction_v1(
        self,
        *,
        metadata: EvidenceMetadata,
        session: date,
        stock: str,
        signal_timestamp: datetime,
        opening_window_v1: OpeningPreEntryWindowV1,
        opening_transition_state_v1: OpeningMarketTransitionStateResultV1,
        group_o_context: FrozenGroupOContext | None,
        missing_reason: str,
        receipt_created_at_utc_v1_1: datetime | None = None,
        first_buffered_event_received_at_utc_v1_1: datetime | None = None,
        entry_data_admitted_before_receipt_v1_1: bool = False,
    ) -> OpeningReversalPredictionReceiptV1:
        """Emit the frozen ABSTAIN receipt when checkpoint-6 data misses deadline."""

        if self.opening_reversal_activation_v1 is None:
            raise ValueError("opening reversal is not activated")
        existing = self.repository.load_opening_reversal_prediction_v1(
            run_id=metadata.run_id,
            session=session,
            stock=stock,
            experiment_version=("1" if self.opening_reversal_activation_v1_1 is None else "1.1"),
        )
        if existing is not None:
            return existing
        phase, transfer_status = self.repository.opening_reversal_phase_for_session(
            run_id=metadata.run_id,
            session=session,
        )
        iv_scale = (
            None if group_o_context is None else group_o_context.previous_close_implied_movement_15m
        )
        addendum = self.opening_reversal_activation_v1_1
        receipt_created_at = (
            metadata.recorded_at_utc if addendum is None else receipt_created_at_utc_v1_1
        )
        if receipt_created_at is None:
            raise ValueError("V1.1 receipt creation timestamp is unavailable")
        timing_evidence = (
            None
            if addendum is None
            else OpeningReversalPredictionTimingEvidenceV1_1(
                timing_addendum_activation_receipt_hash_v1_1=(
                    addendum.activation_receipt_hash_v1_1
                ),
                rule_committed_at_utc=addendum.activation_timestamp_utc,
                causal_barrier_armed_at_utc=addendum.activation_timestamp_utc,
                predictor_window_completed_at_utc=signal_timestamp,
                first_entry_or_post_entry_event_buffered_at_utc=(
                    first_buffered_event_received_at_utc_v1_1
                ),
                entry_or_post_entry_data_admitted_before_receipt=(
                    entry_data_admitted_before_receipt_v1_1
                ),
                raw_event_archive_write_before_receipt=True,
                decision_surface_release_requires_durable_receipt=True,
                nominal_entry_actionable=False,
                receipt_latency_after_nominal_entry_seconds=(
                    receipt_created_at - signal_timestamp
                ).total_seconds(),
            )
        )
        receipt = build_prediction_receipt_v1(
            OpeningReversalPredictionInputV1(
                activation_timestamp_utc=(
                    self.opening_reversal_activation_v1.activation_timestamp_utc
                    if addendum is None
                    else addendum.activation_timestamp_utc
                ),
                experiment_version="1" if addendum is None else "1.1",
                cohort_phase=phase,
                transfer_status=transfer_status,
                session=session,
                stock=stock,
                checkpoint=6,
                signal_timestamp_utc=signal_timestamp,
                entry_timestamp_utc=signal_timestamp,
                receipt_created_at_utc=receipt_created_at,
                m1c_probability=None,
                m1c_probability_valid=False,
                high_tail_membership=False,
                fresh_episode_id=None,
                canonical_fresh_episode=False,
                tail_phase_v1="UNKNOWN_INCOMPLETE",
                market_opening_return_v1=(opening_window_v1.market_opening_return_v1),
                market_opening_range_v1=(opening_window_v1.market_opening_range_v1),
                opening_market_transition_state_v1=(
                    opening_transition_state_v1.opening_market_transition_state_v1
                ),
                opening_transition_sign_v1=(opening_transition_state_v1.opening_transition_sign_v1),
                opening_transition_event_id_v1=(
                    opening_transition_state_v1.opening_transition_event_id_v1
                ),
                vti_opening_transition_complete=(opening_transition_state_v1.complete_v1),
                stock_causal_data_complete=False,
                previous_close_atm_iv_scale_15m=iv_scale,
                previous_close_atm_iv_scale_valid=(
                    group_o_context is not None
                    and group_o_context.eligible
                    and iv_scale is not None
                ),
                data_source="ibkr",
                capacity_snapshot_id=(self._opening_reversal_capacity_snapshot_id_v1(metadata)),
                frozen_comparisons=(
                    ("incomplete_checkpoint_reason_v1", missing_reason),
                    *tuple(
                        item
                        for baseline in (
                            "most_recent_completed_five_minute_stock_momentum",
                            "complete_stock_opening_window_momentum",
                            "existing_clean_market_direction_baseline",
                            "frozen_a1",
                            "frozen_historical_asymmetric_downside_score",
                            "independently_frozen_microstructure_rule",
                        )
                        for item in (
                            (
                                f"baseline_{baseline}_prediction_sign_v1",
                                None,
                            ),
                            (
                                f"baseline_{baseline}_unavailable_reason_v1",
                                "checkpoint_receipt_incomplete",
                            ),
                        )
                    ),
                    ("stock_opening_return_v1", None),
                    ("stock_opening_range_v1", None),
                    ("stock_opening_alignment_v1", None),
                    (
                        "stock_opening_response_class_v1",
                        "UNKNOWN_INCOMPLETE",
                    ),
                ),
                timing_evidence_v1_1=timing_evidence,
            )
        )
        self.repository.record_opening_reversal_prediction_v1(
            metadata,
            receipt,
        )
        return receipt

    def _signed_market_shock_logging_v1(
        self,
        *,
        item: RecorderCheckpointInput,
        checkpoint: int,
        signal_timestamp: datetime,
    ) -> tuple[
        PreentryMarketWindowsV1,
        MarketShockStateResultV1,
        StockShockResponseResultV1,
        CheckpointShockThresholdsV1 | None,
    ]:
        """Calculate optional evidence while containing every local data failure."""

        thresholds = (
            None
            if self.signed_market_shock_thresholds_v1 is None
            else self.signed_market_shock_thresholds_v1.threshold_for_checkpoint(checkpoint)
        )
        try:
            windows = calculate_preentry_windows_v1(
                market_proxy=MARKET_SHOCK_PROXY_V1,
                session=item.session,
                checkpoint=checkpoint,
                signal_timestamp=signal_timestamp,
                completed_bars=item.completed_market_shock_bars_v1,
            )
            state = classify_market_shock_state_v1(
                windows=windows,
                thresholds=thresholds,
            )
            response = calculate_stock_shock_response_v1(
                symbol=item.symbol,
                session=item.session,
                checkpoint=checkpoint,
                signal_timestamp=signal_timestamp,
                completed_stock_bars=tuple(
                    MarketShockBarV1(
                        symbol=bar.symbol,
                        session=bar.session,
                        bar_ordinal=bar.bar_ordinal,
                        bar_start_timestamp=bar.bar_start_timestamp,
                        bar_complete_timestamp=bar.bar_complete_timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        finalised=bar.finalised,
                    )
                    for bar in item.completed_m1c_bars
                ),
                market_return_w0_v1=windows.market_return_w0_v1,
                market_shock_state_v1=state,
                threshold_15m=(item.group_o_context.previous_close_implied_movement_15m),
            )
            return windows, state, response, thresholds
        except Exception as error:
            reason = f"signed_market_shock_logging_calculation_error:{type(error).__name__}"
            windows, state, response = _unknown_signed_market_shock_logging_v1(
                session=item.session,
                checkpoint=checkpoint,
                signal_timestamp=signal_timestamp,
                reason=reason,
            )
            return windows, state, response, thresholds

    def _opening_market_transition_logging_v1(
        self,
        *,
        item: RecorderCheckpointInput,
        checkpoint: int,
        signal_timestamp: datetime,
    ) -> tuple[
        OpeningPreEntryWindowV1,
        OpeningMarketTransitionStateResultV1,
        StockOpeningResponseResultV1,
    ]:
        """Calculate optional checkpoint-6 opening evidence, failure-contained."""

        if checkpoint != 6:
            return _unknown_opening_market_transition_logging_v1(
                session=item.session,
                previous_session=item.market_previous_session_v1,
                signal_timestamp=signal_timestamp,
                reason="opening_transition_not_checkpoint_6",
            )
        try:
            market_bars = item.completed_market_shock_bars_v1
            session_open = next(
                (
                    bar.bar_start_timestamp
                    for bar in market_bars
                    if bar.symbol == OPENING_MARKET_PROXY_V1
                    and bar.session == item.session
                    and bar.bar_ordinal == 0
                ),
                signal_timestamp - timedelta(minutes=30),
            )
            window = calculate_opening_preentry_window_v1(
                market_proxy=OPENING_MARKET_PROXY_V1,
                session=item.session,
                previous_session=(
                    item.session
                    if item.market_previous_session_v1 is None
                    else item.market_previous_session_v1
                ),
                session_open_timestamp=session_open,
                signal_timestamp=signal_timestamp,
                entry_timestamp=signal_timestamp,
                completed_bars=market_bars,
                prior_regular_session_close=(item.market_prior_regular_session_close_v1),
            )
            state = classify_opening_market_transition_v1(
                window=window,
                thresholds=self.opening_transition_thresholds_v1,
            )
            response = calculate_stock_opening_response_v1(
                symbol=item.symbol,
                session=item.session,
                session_open_timestamp=session_open,
                signal_timestamp=signal_timestamp,
                completed_stock_bars=tuple(
                    MarketShockBarV1(
                        symbol=bar.symbol,
                        session=bar.session,
                        bar_ordinal=bar.bar_ordinal,
                        bar_start_timestamp=bar.bar_start_timestamp,
                        bar_complete_timestamp=bar.bar_complete_timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        finalised=bar.finalised,
                    )
                    for bar in item.completed_m1c_bars
                ),
                market_opening_return_v1=window.market_opening_return_v1,
                opening_transition_state_v1=state,
                threshold_15m=(item.group_o_context.previous_close_implied_movement_15m),
            )
            return window, state, response
        except Exception as error:
            return _unknown_opening_market_transition_logging_v1(
                session=item.session,
                previous_session=item.market_previous_session_v1,
                signal_timestamp=signal_timestamp,
                reason=(
                    f"opening_market_transition_logging_calculation_error:{type(error).__name__}"
                ),
            )

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
        _phase, phase_allows_scientific_evidence = self.repository.prospective_phase_for_session(
            run_id=item.metadata.run_id,
            session=item.session,
        )
        scientific_recording_authorized = (
            item.scientific_recording_authorized and phase_allows_scientific_evidence
        )
        scientific_inputs_eligible = signal_inputs_eligible and scientific_recording_authorized
        scientific_rejection_reasons = tuple(
            dict.fromkeys(
                (
                    *rejection_reasons,
                    *(
                        ()
                        if scientific_recording_authorized
                        else ("scientific_recording_not_authorized",)
                    ),
                )
            )
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
        (
            market_windows_v1,
            market_shock_state_v1,
            stock_shock_response_v1,
            checkpoint_thresholds_v1,
        ) = self._signed_market_shock_logging_v1(
            item=item,
            checkpoint=checkpoint,
            signal_timestamp=trigger.bar_complete_timestamp,
        )
        (
            opening_window_v1,
            opening_transition_state_v1,
            stock_opening_response_v1,
        ) = self._opening_market_transition_logging_v1(
            item=item,
            checkpoint=checkpoint,
            signal_timestamp=trigger.bar_complete_timestamp,
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
            eligible=scientific_inputs_eligible,
            feature_freshness=item.feature_freshness,
            rejection_reasons=scientific_rejection_reasons,
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
            eligible=scientific_inputs_eligible,
        )
        quiet_observation_id: str | None = None
        if quiet_episode.fresh_episode:
            quiet_observation_id = self.repository.record_quiet_episode(
                item.metadata,
                quiet_checkpoint_id=quiet_checkpoint_id,
                decision=quiet_episode,
                scientific_recording_valid=scientific_inputs_eligible,
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
                scientific_recording_valid=scientific_inputs_eligible,
                data_quality_flags=scientific_rejection_reasons,
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
                    scientific_recording_authorized=scientific_recording_authorized,
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
        try:
            self.repository.record_signed_market_shock_checkpoint_v1(
                item.metadata,
                checkpoint_id=checkpoint_id,
                symbol=item.symbol,
                session=item.session,
                checkpoint=checkpoint,
                market_windows_v1=market_windows_v1,
                market_shock_state_v1=market_shock_state_v1,
                stock_shock_response_v1=stock_shock_response_v1,
                market_shock_thresholds_v1=checkpoint_thresholds_v1,
                activation_status_v1=(self.signed_market_shock_activation_status_v1),
            )
        except Exception as error:
            (
                market_windows_v1,
                market_shock_state_v1,
                stock_shock_response_v1,
            ) = _unknown_signed_market_shock_logging_v1(
                session=item.session,
                checkpoint=checkpoint,
                signal_timestamp=trigger.bar_complete_timestamp,
                reason=(f"signed_market_shock_logging_persistence_error:{type(error).__name__}"),
            )
        try:
            self.repository.record_opening_market_transition_checkpoint_v1(
                item.metadata,
                checkpoint_id=checkpoint_id,
                symbol=item.symbol,
                session=item.session,
                checkpoint=checkpoint,
                opening_window_v1=opening_window_v1,
                opening_transition_state_v1=opening_transition_state_v1,
                stock_opening_response_v1=stock_opening_response_v1,
                opening_thresholds_v1=self.opening_transition_thresholds_v1,
                activation_status_v1=(self.opening_transition_activation_status_v1),
            )
        except Exception as error:
            (
                opening_window_v1,
                opening_transition_state_v1,
                stock_opening_response_v1,
            ) = _unknown_opening_market_transition_logging_v1(
                session=item.session,
                previous_session=item.market_previous_session_v1,
                signal_timestamp=trigger.bar_complete_timestamp,
                reason=(
                    f"opening_market_transition_logging_persistence_error:{type(error).__name__}"
                ),
            )
        all_reasons = tuple(
            dict.fromkeys(
                (
                    *scientific_rejection_reasons,
                    *(() if safety is None else safety.rejection_reasons),
                )
            )
        )
        opening_reversal_prediction_v1: OpeningReversalPredictionReceiptV1 | None = None
        if checkpoint == 6 and self.opening_reversal_activation_v1 is not None:
            existing_opening_receipt = self.repository.load_opening_reversal_prediction_v1(
                run_id=item.metadata.run_id,
                session=item.session,
                stock=item.symbol,
                experiment_version=(
                    "1" if self.opening_reversal_activation_v1_1 is None else "1.1"
                ),
            )
            if existing_opening_receipt is not None:
                opening_reversal_prediction_v1 = existing_opening_receipt
            else:
                opening_reversal_prediction_v1 = self._build_opening_reversal_prediction_v1(
                    item=item,
                    checkpoint=checkpoint,
                    trigger_timestamp=trigger.bar_complete_timestamp,
                    score=score,
                    episode=episode,
                    tail_phase_v1=tail_phase_v1,
                    opening_window_v1=opening_window_v1,
                    opening_transition_state_v1=opening_transition_state_v1,
                    stock_opening_response_v1=stock_opening_response_v1,
                    classifications=classifications,
                    signal_inputs_eligible=signal_inputs_eligible,
                )
                self.repository.record_opening_reversal_prediction_v1(
                    item.metadata,
                    opening_reversal_prediction_v1,
                )
        return RecorderCheckpointResult(
            checkpoint_id=checkpoint_id,
            score=score,
            tail_phase_v1=tail_phase_v1,
            movement_consumed_state_v1=movement_consumed_state_v1,
            movement_consumed_bucket_v1=movement_consumed_bucket_v1,
            market_windows_v1=market_windows_v1,
            market_shock_state_v1=market_shock_state_v1,
            stock_shock_response_v1=stock_shock_response_v1,
            opening_window_v1=opening_window_v1,
            opening_transition_state_v1=opening_transition_state_v1,
            stock_opening_response_v1=stock_opening_response_v1,
            opening_reversal_prediction_v1=opening_reversal_prediction_v1,
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
