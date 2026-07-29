"""Frozen prospective analysis contract for M1C Opening Reversal V1.

The functions in this module accept only post-activation, non-engineering
episodes. They contain no fitting, threshold search, or subgroup selection.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningReversalDecisionReceiptV1,
    OpeningReversalPredictionReceiptV1,
    OpeningReversalSupportGateV1,
    OpeningReversalUnderlyingOutcomeV1,
    PrimaryOptionBidAskOutcomeV1,
    PromotionSelectionV1,
    build_opening_reversal_decision_receipt_v1,
)

AnalysisPhaseV1 = Literal[
    "prospective_development",
    "untouched_confirmation",
]
DirectionDecisionV1 = Literal[
    "prospective_opening_reversal_development_supported",
    "prospective_opening_reversal_development_not_supported",
    "blocked_insufficient_prospective_development_support",
    "blocked_opening_transfer",
    "operational_failure",
    "prospective_opening_reversal_direction_supported",
    "prospective_opening_reversal_ranking_only",
    "prospective_opening_reversal_not_supported",
    "blocked_insufficient_confirmation_support",
]
OptionEconomicsDecisionV1 = Literal[
    "prospective_opening_reversal_option_economics_supported",
    "direction_supported_without_option_edge",
    "option_economics_blocked_insufficient_bid_ask_support",
    "option_economics_blocked_capacity",
    "option_economics_not_supported",
]

SESSION_BOOTSTRAP_SEED_V1 = 2026072902
EVENT_BOOTSTRAP_SEED_V1 = 2026072903
PRIMARY_NULL_SEED_V1: Literal[2026072901] = 2026072901
CLUSTER_BOOTSTRAP_REPLICATIONS_V1 = 2_000
PRIMARY_NULL_REPLICATIONS_V1 = 1_000
WINSOR_FRACTION_V1 = 0.01
FROZEN_BASELINE_IDS_V1 = (
    "follow_vti_severe_opening_sign",
    "oppose_vti_severe_opening_sign",
    "always_call",
    "always_put",
    "most_recent_completed_five_minute_stock_momentum",
    "complete_stock_opening_window_momentum",
    "existing_clean_market_direction_baseline",
    "frozen_a1",
    "frozen_historical_asymmetric_downside_score",
    "independently_frozen_microstructure_rule",
)


def _freeze_sign_map(
    value: object,
) -> tuple[tuple[str, Literal[-1, 0, 1]], ...]:
    rows: tuple[object, ...]
    if isinstance(value, Mapping):
        rows = tuple(value.items())
    elif isinstance(value, (tuple, list)):
        rows = tuple(value)
    else:
        raise TypeError("baseline signs must be a mapping or key/value sequence")
    output: list[tuple[str, Literal[-1, 0, 1]]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise TypeError("baseline sign rows must be key/value pairs")
        key, sign = row
        name = str(key)
        if (
            name in seen
            or not isinstance(sign, int)
            or isinstance(sign, bool)
            or sign not in {-1, 0, 1}
        ):
            raise ValueError("baseline sign rows must be unique and in {-1,0,1}")
        seen.add(name)
        canonical: Literal[-1, 0, 1] = (
            -1 if sign == -1 else 1 if sign == 1 else 0
        )
        output.append((name, canonical))
    return tuple(sorted(output))


def _freeze_reason_map(value: object) -> tuple[tuple[str, str], ...]:
    rows: tuple[object, ...]
    if isinstance(value, Mapping):
        rows = tuple(value.items())
    elif isinstance(value, (tuple, list)):
        rows = tuple(value)
    else:
        raise TypeError(
            "baseline reasons must be a mapping or key/value sequence"
        )
    output: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise TypeError("baseline reason rows must be key/value pairs")
        key, reason = row
        name = str(key)
        text = str(reason).strip()
        if name in seen or not text:
            raise ValueError("baseline unavailability reasons must be unique and nonempty")
        seen.add(name)
        output.append((name, text))
    return tuple(sorted(output))


def _freeze_float_map(
    value: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(key), float(item)) for key, item in value.items()))


class OpeningReversalAnalysisEpisodeV1(BaseModel):
    """One complete scientific episode; engineering rows cannot validate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_receipt_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    outcome_receipt_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    cohort_phase: AnalysisPhaseV1
    session: date
    stock: str
    opening_transition_event_id_v1: str
    opening_transition_sign_v1: Literal[-1, 1]
    prediction_sign_v1: Literal[-1, 1]
    prediction_v1: Literal["CALL", "PUT"]
    r_15m: float
    threshold_15m: float = Field(gt=0.0)
    outcome_direction_v1: Literal[-1, 0, 1]
    material_direction_correct_v1: bool | None
    opening_reversal_aligned_return_v1: float
    promoted: bool
    primary_option_evidence_complete: bool
    baseline_prediction_signs: tuple[
        tuple[str, Literal[-1, 0, 1]],
        ...,
    ] = ()
    baseline_unavailable_reasons: tuple[tuple[str, str], ...] = ()

    @field_validator("baseline_prediction_signs", mode="before")
    @classmethod
    def _baseline_signs_are_immutable(
        cls,
        value: object,
    ) -> tuple[tuple[str, Literal[-1, 0, 1]], ...]:
        return _freeze_sign_map(value)

    @field_validator("baseline_unavailable_reasons", mode="before")
    @classmethod
    def _baseline_reasons_are_immutable(
        cls,
        value: object,
    ) -> tuple[tuple[str, str], ...]:
        return _freeze_reason_map(value)

    @model_validator(mode="after")
    def _frozen_relationships(self) -> OpeningReversalAnalysisEpisodeV1:
        if not all(
            math.isfinite(value)
            for value in (
                self.r_15m,
                self.threshold_15m,
                self.opening_reversal_aligned_return_v1,
            )
        ):
            raise ValueError("analysis episode contains non-finite values")
        if self.prediction_sign_v1 != -self.opening_transition_sign_v1:
            raise ValueError("analysis episode differs from reversal rule")
        expected_prediction = "CALL" if self.prediction_sign_v1 == 1 else "PUT"
        if self.prediction_v1 != expected_prediction:
            raise ValueError("prediction label and sign differ")
        if not math.isclose(
            self.opening_reversal_aligned_return_v1,
            self.prediction_sign_v1 * self.r_15m,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("aligned return differs from frozen formula")
        expected_direction = (
            1
            if self.r_15m > self.threshold_15m
            else -1
            if self.r_15m < -self.threshold_15m
            else 0
        )
        if self.outcome_direction_v1 != expected_direction:
            raise ValueError("material direction differs from strict threshold")
        expected_correct = (
            None
            if expected_direction == 0
            else expected_direction == self.prediction_sign_v1
        )
        if self.material_direction_correct_v1 is not expected_correct:
            raise ValueError("material correctness differs")
        signs = dict(self.baseline_prediction_signs)
        unavailable = dict(self.baseline_unavailable_reasons)
        if set(signs).intersection(unavailable):
            raise ValueError("a baseline cannot be available and unavailable")
        if set(signs).union(unavailable) != set(FROZEN_BASELINE_IDS_V1):
            raise ValueError(
                "every frozen baseline must be present or explicitly unavailable"
            )
        exact_signs = {
            "follow_vti_severe_opening_sign": self.opening_transition_sign_v1,
            "oppose_vti_severe_opening_sign": self.prediction_sign_v1,
            "always_call": 1,
            "always_put": -1,
        }
        if any(signs.get(name) != sign for name, sign in exact_signs.items()):
            raise ValueError("deterministic baseline sign differs")
        return self


def build_opening_reversal_analysis_episode_v1(
    *,
    prediction: OpeningReversalPredictionReceiptV1,
    outcome: OpeningReversalUnderlyingOutcomeV1,
    promoted: bool,
    primary_option_evidence_complete: bool,
) -> OpeningReversalAnalysisEpisodeV1:
    """Join immutable receipts and carry every frozen causal baseline forward."""

    receipt = OpeningReversalPredictionReceiptV1.model_validate(
        prediction.model_dump(mode="python")
    )
    result = OpeningReversalUnderlyingOutcomeV1.model_validate(
        outcome.model_dump(mode="python")
    )
    if (
        result.outcome_completeness_v1 != "complete"
        or not receipt.scientific_outcome_eligible_v1
        or receipt.cohort_phase == "engineering_transfer"
        or result.prediction_receipt_hash_v1 != receipt.receipt_hash_v1
        or receipt.opening_transition_sign_v1 not in {-1, 1}
        or receipt.opening_transition_event_id_v1 is None
        or result.r_15m is None
        or result.opening_reversal_aligned_return_v1 is None
        or result.outcome_state_v1 is None
    ):
        raise ValueError("analysis episode requires linked complete scientific receipts")
    comparisons = dict(receipt.frozen_comparisons)
    signs: dict[str, Literal[-1, 0, 1]] = {
        "follow_vti_severe_opening_sign": receipt.opening_transition_sign_v1,
        "oppose_vti_severe_opening_sign": receipt.prediction_sign_v1,
        "always_call": 1,
        "always_put": -1,
    }
    unavailable: dict[str, str] = {}
    for baseline in FROZEN_BASELINE_IDS_V1[4:]:
        raw_sign = comparisons.get(f"baseline_{baseline}_prediction_sign_v1")
        raw_reason = comparisons.get(f"baseline_{baseline}_unavailable_reason_v1")
        if (
            isinstance(raw_sign, int)
            and not isinstance(raw_sign, bool)
            and raw_sign in {-1, 0, 1}
        ):
            signs[baseline] = cast(Literal[-1, 0, 1], raw_sign)
        elif isinstance(raw_reason, str) and raw_reason.strip():
            unavailable[baseline] = raw_reason.strip()
        else:
            raise ValueError(
                f"frozen baseline lacks causal value or reason:{baseline}"
            )
    outcome_direction: Literal[-1, 0, 1] = (
        1
        if result.outcome_state_v1 == "MATERIAL_UP"
        else -1
        if result.outcome_state_v1 == "MATERIAL_DOWN"
        else 0
    )
    return OpeningReversalAnalysisEpisodeV1(
        prediction_receipt_hash_v1=receipt.receipt_hash_v1,
        outcome_receipt_hash_v1=result.outcome_receipt_hash_v1,
        cohort_phase=receipt.cohort_phase,
        session=receipt.session,
        stock=receipt.stock,
        opening_transition_event_id_v1=receipt.opening_transition_event_id_v1,
        opening_transition_sign_v1=receipt.opening_transition_sign_v1,
        prediction_sign_v1=cast(Literal[-1, 1], receipt.prediction_sign_v1),
        prediction_v1=cast(Literal["CALL", "PUT"], receipt.prediction_v1),
        r_15m=result.r_15m,
        threshold_15m=result.threshold_15m,
        outcome_direction_v1=outcome_direction,
        material_direction_correct_v1=(
            result.correct_predicted_material_direction_v1
        ),
        opening_reversal_aligned_return_v1=(
            result.opening_reversal_aligned_return_v1
        ),
        promoted=promoted,
        primary_option_evidence_complete=primary_option_evidence_complete,
        baseline_prediction_signs=tuple(sorted(signs.items())),
        baseline_unavailable_reasons=tuple(sorted(unavailable.items())),
    )


class OpeningReversalSupportSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    complete_eligible_stock_episodes: int
    unique_severe_opening_events: int
    positive_transition_events: int
    negative_transition_events: int
    represented_stocks: int
    sessions: int
    maximum_stock_episode_fraction: float
    maximum_event_episode_fraction: float
    passes: bool
    failure_reasons: tuple[str, ...]


class IntervalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lower_95: float | None
    upper_95: float | None


class OpeningReversalDirectionSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_count: int
    event_count: int
    mean_aligned_return: float | None
    median_aligned_return: float | None
    session_cluster_interval: IntervalV1
    event_cluster_interval: IntervalV1
    material_direction_accuracy: float | None
    accuracy_counting_no_moves_as_failures: float | None
    material_up_count: int
    material_down_count: int
    no_material_move_count: int
    follow_vti_accuracy_counting_no_moves_as_failures: float | None
    mean_follow_vti_aligned_return: float | None
    difference_versus_follow_vti: float | None
    positive_transition_mean_reversal_return: float | None
    negative_transition_mean_reversal_return: float | None
    call_mean_reversal_return: float | None
    put_mean_reversal_return: float | None
    positive_session_rate: float | None
    positive_event_rate: float | None
    positive_month_rate: float | None
    winsorised_one_percent_mean: float | None
    leave_one_stock_out_minimum_mean: float | None
    leave_one_session_out_minimum_mean: float | None
    leave_one_event_out_minimum_mean: float | None
    primary_null_p_value: float | None
    temporal_placebo_mean: float | None


class OpeningReversalNullDrawV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: Literal[2026072901]
    replication: int
    mean_aligned_return: float
    material_direction_accuracy: float | None
    accuracy_counting_no_moves_as_failures: float
    difference_versus_follow_vti: float
    positive_transition_consistent: bool
    negative_transition_consistent: bool


class OpeningReversalBaselinePopulationResultV1(BaseModel):
    """One frozen baseline evaluated on one explicitly named population."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    population: Literal[
        "all_complete_eligible_episodes",
        "promoted_episodes",
        "complete_primary_option_evidence",
        "same_event_comparison",
    ]
    baseline: str
    population_episode_count: int = Field(ge=0)
    available_episode_count: int = Field(ge=0)
    unavailable_episode_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    baseline_mean_aligned_return: float | None
    reversal_mean_on_identical_episodes: float | None
    reversal_minus_baseline_mean: float | None


class OpeningReversalAnalysisResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort_phase: AnalysisPhaseV1
    support: OpeningReversalSupportSummaryV1
    summary: OpeningReversalDirectionSummaryV1
    session_bootstrap_draws: tuple[float, ...]
    event_bootstrap_draws: tuple[float, ...]
    primary_null_draws: tuple[OpeningReversalNullDrawV1, ...]
    temporal_placebo_values: tuple[float, ...]
    leave_one_stock_out: tuple[tuple[str, float], ...]
    leave_one_session_out: tuple[tuple[str, float], ...]
    leave_one_event_out: tuple[tuple[str, float], ...]
    baseline_means: tuple[tuple[str, float], ...]
    baseline_population_results: tuple[
        OpeningReversalBaselinePopulationResultV1,
        ...,
    ]
    decision: DirectionDecisionV1
    decision_reasons: tuple[str, ...]
    transfer_failure: bool
    operational_failure: bool


class OpeningReversalOptionEpisodeV1(BaseModel):
    """One promoted episode with complete actual ask-entry/bid-exit evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_receipt_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    predicted_leg_outcome_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    opposite_leg_outcome_hash_v1: str = Field(pattern=r"^[a-f0-9]{64}$")
    session: date
    stock: str
    opening_transition_event_id_v1: str
    prediction_v1: Literal["CALL", "PUT"]
    expiry: date
    predicted_leg_conservative_return_v1: float
    opposite_leg_conservative_return_v1: float
    actual_bid_ask_evidence: Literal[True]
    quote_quality_passed: Literal[True]
    staleness_passed: Literal[True]
    continuously_or_adequately_quoted: Literal[True]

    @model_validator(mode="after")
    def _finite_option_returns(self) -> OpeningReversalOptionEpisodeV1:
        if not all(
            math.isfinite(value)
            for value in (
                self.predicted_leg_conservative_return_v1,
                self.opposite_leg_conservative_return_v1,
            )
        ):
            raise ValueError("option economics contains non-finite return")
        if self.predicted_leg_outcome_hash_v1 == self.opposite_leg_outcome_hash_v1:
            raise ValueError("option legs must have distinct outcome hashes")
        return self


def build_opening_reversal_option_episode_v1(
    *,
    prediction: OpeningReversalPredictionReceiptV1,
    promotion: PromotionSelectionV1,
    predicted_leg: PrimaryOptionBidAskOutcomeV1,
    opposite_leg: PrimaryOptionBidAskOutcomeV1,
) -> OpeningReversalOptionEpisodeV1:
    """Join the promoted receipt to its exact complete primary bid/ask pair."""

    receipt = OpeningReversalPredictionReceiptV1.model_validate(
        prediction.model_dump(mode="python")
    )
    selection = PromotionSelectionV1.model_validate(
        promotion.model_dump(mode="python")
    )
    predicted = PrimaryOptionBidAskOutcomeV1.model_validate(
        predicted_leg.model_dump(mode="python")
    )
    opposite = PrimaryOptionBidAskOutcomeV1.model_validate(
        opposite_leg.model_dump(mode="python")
    )
    if (
        selection.promoted is None
        or selection.promoted.receipt_hash_v1 != receipt.receipt_hash_v1
        or not receipt.scientific_outcome_eligible_v1
        or receipt.opening_transition_event_id_v1 is None
        or receipt.prediction_v1 not in {"CALL", "PUT"}
        or predicted.prediction_receipt_hash_v1 != receipt.receipt_hash_v1
        or opposite.prediction_receipt_hash_v1 != receipt.receipt_hash_v1
        or predicted.role != "predicted_leg"
        or opposite.role != "opposite_leg"
        or not predicted.complete
        or not opposite.complete
        or predicted.conservative_return_v1 is None
        or opposite.conservative_return_v1 is None
    ):
        raise ValueError("option episode requires the linked complete promoted pair")
    expected_right = "C" if receipt.prediction_v1 == "CALL" else "P"
    if (
        predicted.contract.right != expected_right
        or opposite.contract.right == expected_right
        or predicted.contract.underlying != receipt.stock
        or opposite.contract.underlying != receipt.stock
        or predicted.contract.expiry != opposite.contract.expiry
        or predicted.contract.strike != opposite.contract.strike
        or predicted.contract.multiplier != opposite.contract.multiplier
    ):
        raise ValueError("primary option legs differ from frozen pair selection")
    return OpeningReversalOptionEpisodeV1(
        prediction_receipt_hash_v1=receipt.receipt_hash_v1,
        predicted_leg_outcome_hash_v1=predicted.outcome_hash_v1,
        opposite_leg_outcome_hash_v1=opposite.outcome_hash_v1,
        session=receipt.session,
        stock=receipt.stock,
        opening_transition_event_id_v1=receipt.opening_transition_event_id_v1,
        prediction_v1=cast(Literal["CALL", "PUT"], receipt.prediction_v1),
        expiry=predicted.contract.expiry,
        predicted_leg_conservative_return_v1=predicted.conservative_return_v1,
        opposite_leg_conservative_return_v1=opposite.conservative_return_v1,
        actual_bid_ask_evidence=True,
        quote_quality_passed=True,
        staleness_passed=True,
        continuously_or_adequately_quoted=True,
    )


class OpeningReversalOptionEconomicsResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_count: int
    call_episode_count: int
    put_episode_count: int
    unique_event_count: int
    mean_predicted_leg_conservative_return: float | None
    mean_opposite_leg_conservative_return: float | None
    predicted_minus_opposite_mean: float | None
    event_cluster_interval: IntervalV1
    maximum_stock_fraction: float
    maximum_expiry_fraction: float
    maximum_event_fraction: float
    support_passes: bool
    support_failure_reasons: tuple[str, ...]
    decision: OptionEconomicsDecisionV1
    decision_reasons: tuple[str, ...]
    event_bootstrap_draws: tuple[float, ...]


def _mean(values: Iterable[float]) -> float | None:
    rows = tuple(values)
    return statistics.fmean(rows) if rows else None


def _positive_group_rate(
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
    key: Callable[[OpeningReversalAnalysisEpisodeV1], object],
) -> float | None:
    groups: dict[object, list[float]] = defaultdict(list)
    for episode in episodes:
        groups[key(episode)].append(episode.opening_reversal_aligned_return_v1)
    if not groups:
        return None
    return statistics.fmean(
        statistics.fmean(values) > 0.0 for values in groups.values()
    )


def _validate_event_identity_v1(
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
) -> None:
    identities: dict[str, tuple[date, Literal[-1, 1]]] = {}
    for episode in episodes:
        identity = (episode.session, episode.opening_transition_sign_v1)
        existing = identities.setdefault(
            episode.opening_transition_event_id_v1,
            identity,
        )
        if existing != identity:
            raise ValueError(
                "one opening event ID maps to multiple sessions or signs"
            )


def evaluate_support_v1(
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
    *,
    gate: OpeningReversalSupportGateV1 | None = None,
) -> OpeningReversalSupportSummaryV1:
    frozen_gate = gate or OpeningReversalSupportGateV1()
    rows = tuple(episodes)
    _validate_event_identity_v1(rows)
    events = {row.opening_transition_event_id_v1 for row in rows}
    positive_events = {
        row.opening_transition_event_id_v1
        for row in rows
        if row.opening_transition_sign_v1 == 1
    }
    negative_events = {
        row.opening_transition_event_id_v1
        for row in rows
        if row.opening_transition_sign_v1 == -1
    }
    stock_counts = Counter(row.stock for row in rows)
    event_counts = Counter(row.opening_transition_event_id_v1 for row in rows)
    denominator = len(rows)
    maximum_stock_fraction = (
        max(stock_counts.values(), default=0) / denominator if denominator else 0.0
    )
    maximum_event_fraction = (
        max(event_counts.values(), default=0) / denominator if denominator else 0.0
    )
    checks = (
        (
            len(rows) >= frozen_gate.complete_eligible_stock_episodes,
            "complete_eligible_stock_episodes_below_150",
        ),
        (
            len(events) >= frozen_gate.unique_severe_opening_events,
            "unique_severe_opening_events_below_40",
        ),
        (
            len(positive_events) >= frozen_gate.positive_transition_events,
            "positive_transition_events_below_15",
        ),
        (
            len(negative_events) >= frozen_gate.negative_transition_events,
            "negative_transition_events_below_15",
        ),
        (
            len(stock_counts) >= frozen_gate.represented_stocks,
            "represented_stocks_below_12",
        ),
        (
            len({row.session for row in rows}) >= frozen_gate.sessions,
            "sessions_below_40",
        ),
        (
            maximum_stock_fraction <= frozen_gate.maximum_stock_episode_fraction,
            "stock_concentration_above_20_percent",
        ),
        (
            maximum_event_fraction <= frozen_gate.maximum_event_episode_fraction,
            "event_concentration_above_15_percent",
        ),
    )
    failures = tuple(reason for passed, reason in checks if not passed)
    return OpeningReversalSupportSummaryV1(
        complete_eligible_stock_episodes=len(rows),
        unique_severe_opening_events=len(events),
        positive_transition_events=len(positive_events),
        negative_transition_events=len(negative_events),
        represented_stocks=len(stock_counts),
        sessions=len({row.session for row in rows}),
        maximum_stock_episode_fraction=maximum_stock_fraction,
        maximum_event_episode_fraction=maximum_event_fraction,
        passes=not failures,
        failure_reasons=failures,
    )


def _cluster_bootstrap(
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
    *,
    cluster: Callable[[OpeningReversalAnalysisEpisodeV1], object],
    seed: int,
    replications: int,
) -> tuple[float, ...]:
    grouped: dict[object, list[float]] = defaultdict(list)
    for episode in episodes:
        grouped[cluster(episode)].append(
            episode.opening_reversal_aligned_return_v1
        )
    keys = tuple(sorted(grouped, key=str))
    if not keys:
        return ()
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(replications):
        selected = generator.choices(keys, k=len(keys))
        values = [value for key in selected for value in grouped[key]]
        draws.append(statistics.fmean(values))
    return tuple(draws)


def _interval(draws: Sequence[float]) -> IntervalV1:
    if not draws:
        return IntervalV1(lower_95=None, upper_95=None)
    ordered = sorted(draws)

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return IntervalV1(lower_95=quantile(0.025), upper_95=quantile(0.975))


def _winsorised_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    lower_index = int(math.floor(WINSOR_FRACTION_V1 * (len(ordered) - 1)))
    upper_index = int(math.ceil((1.0 - WINSOR_FRACTION_V1) * (len(ordered) - 1)))
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return statistics.fmean(min(max(value, lower), upper) for value in values)


def _leave_one_out(
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
    *,
    key: Callable[[OpeningReversalAnalysisEpisodeV1], str],
) -> dict[str, float]:
    values = tuple(sorted({key(row) for row in episodes}))
    return {
        omitted: statistics.fmean(
            row.opening_reversal_aligned_return_v1
            for row in episodes
            if key(row) != omitted
        )
        for omitted in values
        if any(key(row) != omitted for row in episodes)
    }


def _primary_null_draws(
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
    *,
    seed: int,
    replications: int,
) -> tuple[OpeningReversalNullDrawV1, ...]:
    by_stock: dict[str, list[OpeningReversalAnalysisEpisodeV1]] = defaultdict(list)
    for episode in episodes:
        by_stock[episode.stock].append(episode)
    if any(
        len(rows) < 2 or len({row.session for row in rows}) < 2
        for rows in by_stock.values()
    ):
        return ()
    generator = random.Random(seed)
    draws: list[OpeningReversalNullDrawV1] = []
    for replication in range(replications):
        pairs: list[
            tuple[
                OpeningReversalAnalysisEpisodeV1,
                OpeningReversalAnalysisEpisodeV1,
            ]
        ] = []
        for stock in sorted(by_stock):
            predictions = sorted(
                by_stock[stock],
                key=lambda row: (row.session, row.prediction_receipt_hash_v1),
            )
            shift = generator.randrange(1, len(predictions))
            outcomes = predictions[shift:] + predictions[:shift]
            if any(
                prediction.session == outcome.session
                for prediction, outcome in zip(
                    predictions,
                    outcomes,
                    strict=True,
                )
            ):
                raise ValueError("primary null failed its different-session contract")
            pairs.extend(
                zip(
                    predictions,
                    outcomes,
                    strict=True,
                )
            )
        aligned = tuple(
            prediction.prediction_sign_v1 * outcome.r_15m
            for prediction, outcome in pairs
        )
        follow = tuple(
            prediction.opening_transition_sign_v1 * outcome.r_15m
            for prediction, outcome in pairs
        )
        material = tuple(
            (
                prediction.prediction_sign_v1
                == outcome.outcome_direction_v1
            )
            for prediction, outcome in pairs
            if outcome.outcome_direction_v1 != 0
        )
        positive = tuple(
            value
            for value, (prediction, _outcome) in zip(
                aligned,
                pairs,
                strict=True,
            )
            if prediction.opening_transition_sign_v1 == 1
        )
        negative = tuple(
            value
            for value, (prediction, _outcome) in zip(
                aligned,
                pairs,
                strict=True,
            )
            if prediction.opening_transition_sign_v1 == -1
        )
        draws.append(
            OpeningReversalNullDrawV1(
                seed=PRIMARY_NULL_SEED_V1,
                replication=replication,
                mean_aligned_return=statistics.fmean(aligned),
                material_direction_accuracy=(
                    statistics.fmean(material) if material else None
                ),
                accuracy_counting_no_moves_as_failures=statistics.fmean(
                    prediction.prediction_sign_v1
                    == outcome.outcome_direction_v1
                    for prediction, outcome in pairs
                ),
                difference_versus_follow_vti=(
                    statistics.fmean(aligned) - statistics.fmean(follow)
                ),
                positive_transition_consistent=(
                    bool(positive) and statistics.fmean(positive) > 0.0
                ),
                negative_transition_consistent=(
                    bool(negative) and statistics.fmean(negative) > 0.0
                ),
            )
        )
    return tuple(draws)


def _temporal_placebo(
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
) -> tuple[float, ...]:
    by_stock: dict[str, list[OpeningReversalAnalysisEpisodeV1]] = defaultdict(list)
    for episode in episodes:
        by_stock[episode.stock].append(episode)
    values: list[float] = []
    for stock in sorted(by_stock):
        rows = sorted(
            by_stock[stock],
            key=lambda row: (row.session, row.prediction_receipt_hash_v1),
        )
        for prediction, next_outcome in zip(rows, rows[1:], strict=False):
            if next_outcome.session <= prediction.session:
                raise ValueError("temporal placebo chronology differs")
            values.append(prediction.prediction_sign_v1 * next_outcome.r_15m)
    return tuple(values)


def _baseline_means(
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for baseline in FROZEN_BASELINE_IDS_V1:
        values = tuple(
            dict(row.baseline_prediction_signs)[baseline] * row.r_15m
            for row in episodes
            if baseline in dict(row.baseline_prediction_signs)
        )
        if values:
            output[baseline] = statistics.fmean(values)
    return output


def _baseline_population_results(
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
) -> tuple[OpeningReversalBaselinePopulationResultV1, ...]:
    populations: tuple[
        tuple[
            Literal[
                "all_complete_eligible_episodes",
                "promoted_episodes",
                "complete_primary_option_evidence",
                "same_event_comparison",
            ],
            tuple[OpeningReversalAnalysisEpisodeV1, ...],
        ],
        ...,
    ] = (
        ("all_complete_eligible_episodes", tuple(episodes)),
        ("promoted_episodes", tuple(row for row in episodes if row.promoted)),
        (
            "complete_primary_option_evidence",
            tuple(row for row in episodes if row.primary_option_evidence_complete),
        ),
        ("same_event_comparison", tuple(episodes)),
    )
    results: list[OpeningReversalBaselinePopulationResultV1] = []
    for population, population_rows in populations:
        for baseline in FROZEN_BASELINE_IDS_V1:
            available = tuple(
                row
                for row in population_rows
                if baseline in dict(row.baseline_prediction_signs)
            )
            baseline_mean = _mean(
                dict(row.baseline_prediction_signs)[baseline] * row.r_15m
                for row in available
            )
            reversal_mean = _mean(
                row.opening_reversal_aligned_return_v1 for row in available
            )
            results.append(
                OpeningReversalBaselinePopulationResultV1(
                    population=population,
                    baseline=baseline,
                    population_episode_count=len(population_rows),
                    available_episode_count=len(available),
                    unavailable_episode_count=len(population_rows) - len(available),
                    event_count=len(
                        {
                            row.opening_transition_event_id_v1
                            for row in available
                        }
                    ),
                    baseline_mean_aligned_return=baseline_mean,
                    reversal_mean_on_identical_episodes=reversal_mean,
                    reversal_minus_baseline_mean=(
                        None
                        if baseline_mean is None or reversal_mean is None
                        else reversal_mean - baseline_mean
                    ),
                )
            )
    return tuple(results)


def build_opening_reversal_direction_decision_receipt_v1(
    *,
    result: OpeningReversalAnalysisResultV1,
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
    boundary_timestamp_utc: datetime,
) -> OpeningReversalDecisionReceiptV1:
    """Bind a development/confirmation decision to every outcome receipt."""

    rows = tuple(
        OpeningReversalAnalysisEpisodeV1.model_validate(
            episode.model_dump(mode="python")
        )
        for episode in episodes
    )
    expected_phase: AnalysisPhaseV1 = result.cohort_phase
    if not rows or any(row.cohort_phase != expected_phase for row in rows):
        raise ValueError("decision receipt rows differ from analysis cohort")
    support = evaluate_support_v1(rows)
    if support != result.support:
        raise ValueError("decision receipt support differs from analysis result")
    recalculated = analyze_direction_cohort_v1(
        rows,
        phase=expected_phase,
        transfer_failure=result.transfer_failure,
        operational_failure=result.operational_failure,
        cluster_bootstrap_replications=CLUSTER_BOOTSTRAP_REPLICATIONS_V1,
        primary_null_replications=PRIMARY_NULL_REPLICATIONS_V1,
    )
    if recalculated != result:
        raise ValueError("decision receipt statistics differ from frozen analysis")
    stock_counts = Counter(row.stock for row in rows)
    event_counts = Counter(row.opening_transition_event_id_v1 for row in rows)
    support_counts = {
        "complete_eligible_stock_episodes": support.complete_eligible_stock_episodes,
        "unique_severe_opening_events": support.unique_severe_opening_events,
        "positive_transition_events": support.positive_transition_events,
        "negative_transition_events": support.negative_transition_events,
        "represented_stocks": support.represented_stocks,
        "sessions": support.sessions,
        "maximum_stock_episode_count": max(stock_counts.values()),
        "maximum_event_episode_count": max(event_counts.values()),
    }
    return build_opening_reversal_decision_receipt_v1(
        receipt_kind=(
            "development"
            if expected_phase == "prospective_development"
            else "confirmation"
        ),
        boundary_timestamp_utc=boundary_timestamp_utc,
        decision=result.decision,
        cohort_first_session=min(row.session for row in rows),
        cohort_last_session=max(row.session for row in rows),
        source_receipt_hashes=tuple(
            sorted(row.outcome_receipt_hash_v1 for row in rows)
        ),
        support_counts=support_counts,
        protected_outcome_fields_accessed=True,
    )


def _decision(
    *,
    phase: AnalysisPhaseV1,
    support: OpeningReversalSupportSummaryV1,
    summary: OpeningReversalDirectionSummaryV1,
    transfer_failure: bool,
    operational_failure: bool,
) -> tuple[DirectionDecisionV1, tuple[str, ...]]:
    if operational_failure:
        return "operational_failure", ("critical_operational_failure",)
    if transfer_failure:
        return "blocked_opening_transfer", ("opening_transfer_not_supported",)
    if not support.passes:
        return (
            "blocked_insufficient_prospective_development_support"
            if phase == "prospective_development"
            else "blocked_insufficient_confirmation_support",
            support.failure_reasons,
        )
    assert summary.mean_aligned_return is not None
    assert summary.difference_versus_follow_vti is not None
    assert summary.positive_transition_mean_reversal_return is not None
    assert summary.negative_transition_mean_reversal_return is not None
    development_checks = {
        "mean_reversal_not_positive": summary.mean_aligned_return > 0.0,
        "material_accuracy_not_above_half": (
            summary.material_direction_accuracy is not None
            and summary.material_direction_accuracy > 0.5
        ),
        "did_not_beat_follow_vti": summary.difference_versus_follow_vti > 0.0,
        "positive_transition_reversal_not_positive": (
            summary.positive_transition_mean_reversal_return > 0.0
        ),
        "negative_transition_reversal_not_positive": (
            summary.negative_transition_mean_reversal_return > 0.0
        ),
        "stock_dependence": (
            summary.leave_one_stock_out_minimum_mean is not None
            and summary.leave_one_stock_out_minimum_mean > 0.0
        ),
        "session_dependence": (
            summary.leave_one_session_out_minimum_mean is not None
            and summary.leave_one_session_out_minimum_mean > 0.0
        ),
        "event_dependence": (
            summary.leave_one_event_out_minimum_mean is not None
            and summary.leave_one_event_out_minimum_mean > 0.0
        ),
        "session_uncertainty_clearly_inconsistent": (
            summary.session_cluster_interval.upper_95 is not None
            and summary.session_cluster_interval.upper_95 > 0.0
        ),
        "event_uncertainty_clearly_inconsistent": (
            summary.event_cluster_interval.upper_95 is not None
            and summary.event_cluster_interval.upper_95 > 0.0
        ),
    }
    failures = tuple(name for name, passed in development_checks.items() if not passed)
    if phase == "prospective_development":
        return (
            (
                "prospective_opening_reversal_development_supported"
                if not failures
                else "prospective_opening_reversal_development_not_supported"
            ),
            failures,
        )

    confirmation_checks = {
        **development_checks,
        "session_cluster_lower_bound_not_positive": (
            summary.session_cluster_interval.lower_95 is not None
            and summary.session_cluster_interval.lower_95 > 0.0
        ),
        "event_cluster_lower_bound_not_positive": (
            summary.event_cluster_interval.lower_95 is not None
            and summary.event_cluster_interval.lower_95 > 0.0
        ),
        "accuracy_including_no_moves_not_above_follow": (
            summary.accuracy_counting_no_moves_as_failures is not None
            and summary.follow_vti_accuracy_counting_no_moves_as_failures is not None
            and summary.accuracy_counting_no_moves_as_failures
            > summary.follow_vti_accuracy_counting_no_moves_as_failures
        ),
        "one_percent_winsorised_mean_not_positive": (
            summary.winsorised_one_percent_mean is not None
            and summary.winsorised_one_percent_mean > 0.0
        ),
        "primary_null_p_value_not_below_0_05": (
            summary.primary_null_p_value is not None
            and summary.primary_null_p_value < 0.05
        ),
        "temporal_placebo_reproduced_effect": (
            summary.temporal_placebo_mean is not None
            and summary.temporal_placebo_mean <= 0.0
        ),
    }
    confirmation_failures = tuple(
        name for name, passed in confirmation_checks.items() if not passed
    )
    if not confirmation_failures:
        return "prospective_opening_reversal_direction_supported", ()
    ranking_checks = (
        summary.mean_aligned_return > 0.0
        and summary.difference_versus_follow_vti > 0.0
        and summary.primary_null_p_value is not None
        and summary.primary_null_p_value < 0.05
    )
    return (
        (
            "prospective_opening_reversal_ranking_only"
            if ranking_checks
            else "prospective_opening_reversal_not_supported"
        ),
        confirmation_failures,
    )


def analyze_direction_cohort_v1(
    episodes: Sequence[OpeningReversalAnalysisEpisodeV1],
    *,
    phase: AnalysisPhaseV1,
    transfer_failure: bool = False,
    operational_failure: bool = False,
    cluster_bootstrap_replications: int = CLUSTER_BOOTSTRAP_REPLICATIONS_V1,
    primary_null_replications: int = PRIMARY_NULL_REPLICATIONS_V1,
) -> OpeningReversalAnalysisResultV1:
    """Run the frozen support, uncertainty, null, placebo, and decision contract."""

    if cluster_bootstrap_replications < 1 or primary_null_replications < 1_000:
        raise ValueError("analysis replication count below frozen contract")
    rows = tuple(
        OpeningReversalAnalysisEpisodeV1.model_validate(
            episode.model_dump(mode="python")
        )
        for episode in episodes
    )
    if any(row.cohort_phase != phase for row in rows):
        raise ValueError("analysis cannot cross development and confirmation")
    if len({(row.stock, row.session) for row in rows}) != len(rows):
        raise ValueError("analysis has duplicate stock/session episodes")
    _validate_event_identity_v1(rows)
    support = evaluate_support_v1(rows)
    aligned = tuple(row.opening_reversal_aligned_return_v1 for row in rows)
    session_draws = _cluster_bootstrap(
        rows,
        cluster=lambda row: row.session,
        seed=SESSION_BOOTSTRAP_SEED_V1,
        replications=cluster_bootstrap_replications,
    )
    event_draws = _cluster_bootstrap(
        rows,
        cluster=lambda row: row.opening_transition_event_id_v1,
        seed=EVENT_BOOTSTRAP_SEED_V1,
        replications=cluster_bootstrap_replications,
    )
    null_draws = _primary_null_draws(
        rows,
        seed=PRIMARY_NULL_SEED_V1,
        replications=primary_null_replications,
    )
    placebo = _temporal_placebo(rows)
    leave_stock = _leave_one_out(rows, key=lambda row: row.stock)
    leave_session = _leave_one_out(
        rows,
        key=lambda row: row.session.isoformat(),
    )
    leave_event = _leave_one_out(
        rows,
        key=lambda row: row.opening_transition_event_id_v1,
    )
    material = tuple(
        row for row in rows if row.outcome_direction_v1 != 0
    )
    primary_accuracy_all = _mean(
        float(row.material_direction_correct_v1 is True) for row in rows
    )
    follow_accuracy_all = _mean(
        float(row.outcome_direction_v1 == row.opening_transition_sign_v1)
        for row in rows
    )
    observed_mean = _mean(aligned)
    null_p = (
        None
        if observed_mean is None or not null_draws
        else (
            1
            + sum(draw.mean_aligned_return >= observed_mean for draw in null_draws)
        )
        / (len(null_draws) + 1)
    )
    summary = OpeningReversalDirectionSummaryV1(
        episode_count=len(rows),
        event_count=len(
            {row.opening_transition_event_id_v1 for row in rows}
        ),
        mean_aligned_return=observed_mean,
        median_aligned_return=statistics.median(aligned) if aligned else None,
        session_cluster_interval=_interval(session_draws),
        event_cluster_interval=_interval(event_draws),
        material_direction_accuracy=_mean(
            float(row.material_direction_correct_v1 is True)
            for row in material
        ),
        accuracy_counting_no_moves_as_failures=primary_accuracy_all,
        material_up_count=sum(row.outcome_direction_v1 == 1 for row in rows),
        material_down_count=sum(row.outcome_direction_v1 == -1 for row in rows),
        no_material_move_count=sum(row.outcome_direction_v1 == 0 for row in rows),
        follow_vti_accuracy_counting_no_moves_as_failures=follow_accuracy_all,
        mean_follow_vti_aligned_return=_mean(
            row.opening_transition_sign_v1 * row.r_15m for row in rows
        ),
        difference_versus_follow_vti=(
            None
            if observed_mean is None
            else observed_mean
            - statistics.fmean(
                row.opening_transition_sign_v1 * row.r_15m for row in rows
            )
        ),
        positive_transition_mean_reversal_return=_mean(
            row.opening_reversal_aligned_return_v1
            for row in rows
            if row.opening_transition_sign_v1 == 1
        ),
        negative_transition_mean_reversal_return=_mean(
            row.opening_reversal_aligned_return_v1
            for row in rows
            if row.opening_transition_sign_v1 == -1
        ),
        call_mean_reversal_return=_mean(
            row.opening_reversal_aligned_return_v1
            for row in rows
            if row.prediction_v1 == "CALL"
        ),
        put_mean_reversal_return=_mean(
            row.opening_reversal_aligned_return_v1
            for row in rows
            if row.prediction_v1 == "PUT"
        ),
        positive_session_rate=_positive_group_rate(rows, lambda row: row.session),
        positive_event_rate=_positive_group_rate(
            rows,
            lambda row: row.opening_transition_event_id_v1,
        ),
        positive_month_rate=_positive_group_rate(
            rows,
            lambda row: (row.session.year, row.session.month),
        ),
        winsorised_one_percent_mean=_winsorised_mean(aligned),
        leave_one_stock_out_minimum_mean=(
            min(leave_stock.values()) if leave_stock else None
        ),
        leave_one_session_out_minimum_mean=(
            min(leave_session.values()) if leave_session else None
        ),
        leave_one_event_out_minimum_mean=(
            min(leave_event.values()) if leave_event else None
        ),
        primary_null_p_value=null_p,
        temporal_placebo_mean=_mean(placebo),
    )
    decision, reasons = _decision(
        phase=phase,
        support=support,
        summary=summary,
        transfer_failure=transfer_failure,
        operational_failure=operational_failure,
    )
    return OpeningReversalAnalysisResultV1(
        cohort_phase=phase,
        support=support,
        summary=summary,
        session_bootstrap_draws=session_draws,
        event_bootstrap_draws=event_draws,
        primary_null_draws=null_draws,
        temporal_placebo_values=placebo,
        leave_one_stock_out=_freeze_float_map(leave_stock),
        leave_one_session_out=_freeze_float_map(leave_session),
        leave_one_event_out=_freeze_float_map(leave_event),
        baseline_means=_freeze_float_map(_baseline_means(rows)),
        baseline_population_results=_baseline_population_results(rows),
        decision=decision,
        decision_reasons=reasons,
        transfer_failure=transfer_failure,
        operational_failure=operational_failure,
    )


def analyze_option_economics_v1(
    episodes: Sequence[OpeningReversalOptionEpisodeV1],
    *,
    underlying_direction_supported: bool,
    capacity_blocked: bool = False,
    event_bootstrap_replications: int = CLUSTER_BOOTSTRAP_REPLICATIONS_V1,
) -> OpeningReversalOptionEconomicsResultV1:
    """Apply the separate frozen actual-bid/ask option decision contract."""

    if event_bootstrap_replications < 1:
        raise ValueError("option event bootstrap replication count is invalid")
    rows = tuple(
        OpeningReversalOptionEpisodeV1.model_validate(
            episode.model_dump(mode="python")
        )
        for episode in episodes
    )
    if len({row.prediction_receipt_hash_v1 for row in rows}) != len(rows):
        raise ValueError("option economics has duplicate prediction receipts")
    if (
        len(
            {
                outcome_hash
                for row in rows
                for outcome_hash in (
                    row.predicted_leg_outcome_hash_v1,
                    row.opposite_leg_outcome_hash_v1,
                )
            }
        )
        != 2 * len(rows)
    ):
        raise ValueError("option economics reuses a primary-option outcome")
    event_sessions: dict[str, date] = {}
    for row in rows:
        if (
            event_sessions.setdefault(
                row.opening_transition_event_id_v1,
                row.session,
            )
            != row.session
        ):
            raise ValueError("one option opening event maps to multiple sessions")
    events = {
        row.opening_transition_event_id_v1 for row in rows
    }
    stock_counts = Counter(row.stock for row in rows)
    expiry_counts = Counter(row.expiry for row in rows)
    event_counts = Counter(
        row.opening_transition_event_id_v1 for row in rows
    )
    denominator = len(rows)

    def fraction(counter: Counter[object]) -> float:
        return (
            max(counter.values(), default=0) / denominator
            if denominator
            else 0.0
        )

    call_count = sum(row.prediction_v1 == "CALL" for row in rows)
    put_count = sum(row.prediction_v1 == "PUT" for row in rows)
    maximum_stock_fraction = fraction(Counter(stock_counts))
    maximum_expiry_fraction = fraction(Counter(expiry_counts))
    maximum_event_fraction = fraction(Counter(event_counts))
    checks = (
        (len(rows) >= 100, "complete_promoted_option_episodes_below_100"),
        (call_count >= 30, "call_option_episodes_below_30"),
        (put_count >= 30, "put_option_episodes_below_30"),
        (len(events) >= 30, "unique_severe_opening_events_below_30"),
        (
            maximum_stock_fraction <= 0.20,
            "option_stock_concentration_above_20_percent",
        ),
        (
            maximum_expiry_fraction <= 0.20,
            "option_expiry_concentration_above_20_percent",
        ),
        (
            maximum_event_fraction <= 0.15,
            "option_event_concentration_above_15_percent",
        ),
    )
    support_failures = tuple(reason for passed, reason in checks if not passed)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.opening_transition_event_id_v1].append(
            row.predicted_leg_conservative_return_v1
        )
    keys = tuple(sorted(grouped))
    generator = random.Random(EVENT_BOOTSTRAP_SEED_V1)
    event_draws = tuple(
        statistics.fmean(
            value
            for key in generator.choices(keys, k=len(keys))
            for value in grouped[key]
        )
        for _ in range(event_bootstrap_replications)
    ) if keys else ()
    interval = _interval(event_draws)
    predicted_mean = _mean(
        row.predicted_leg_conservative_return_v1 for row in rows
    )
    opposite_mean = _mean(
        row.opposite_leg_conservative_return_v1 for row in rows
    )
    difference = (
        None
        if predicted_mean is None or opposite_mean is None
        else predicted_mean - opposite_mean
    )
    reasons: tuple[str, ...]
    if capacity_blocked:
        decision: OptionEconomicsDecisionV1 = "option_economics_blocked_capacity"
        reasons = ("primary_option_pair_capacity_blocked",)
    elif not underlying_direction_supported:
        decision = "option_economics_blocked_insufficient_bid_ask_support"
        reasons = ("underlying_direction_not_supported",)
    elif support_failures:
        decision = "option_economics_blocked_insufficient_bid_ask_support"
        reasons = support_failures
    else:
        statistical_checks = (
            (
                predicted_mean is not None and predicted_mean > 0.0,
                "predicted_leg_conservative_return_not_positive",
            ),
            (
                interval.lower_95 is not None and interval.lower_95 > 0.0,
                "event_cluster_lower_bound_not_positive",
            ),
            (
                difference is not None and difference > 0.0,
                "predicted_leg_did_not_beat_opposite_leg",
            ),
        )
        reasons = tuple(
            reason for passed, reason in statistical_checks if not passed
        )
        decision = (
            "prospective_opening_reversal_option_economics_supported"
            if not reasons
            else "direction_supported_without_option_edge"
        )
    return OpeningReversalOptionEconomicsResultV1(
        episode_count=len(rows),
        call_episode_count=call_count,
        put_episode_count=put_count,
        unique_event_count=len(events),
        mean_predicted_leg_conservative_return=predicted_mean,
        mean_opposite_leg_conservative_return=opposite_mean,
        predicted_minus_opposite_mean=difference,
        event_cluster_interval=interval,
        maximum_stock_fraction=maximum_stock_fraction,
        maximum_expiry_fraction=maximum_expiry_fraction,
        maximum_event_fraction=maximum_event_fraction,
        support_passes=not support_failures,
        support_failure_reasons=support_failures,
        decision=decision,
        decision_reasons=reasons,
        event_bootstrap_draws=event_draws,
    )


def build_opening_reversal_option_decision_receipt_v1(
    *,
    result: OpeningReversalOptionEconomicsResultV1,
    episodes: Sequence[OpeningReversalOptionEpisodeV1],
    supported_direction_receipt: OpeningReversalDecisionReceiptV1,
    boundary_timestamp_utc: datetime,
) -> OpeningReversalDecisionReceiptV1:
    """Bind the option decision to both legs and supported direction evidence."""

    rows = tuple(
        OpeningReversalOptionEpisodeV1.model_validate(
            episode.model_dump(mode="python")
        )
        for episode in episodes
    )
    direction = OpeningReversalDecisionReceiptV1.model_validate(
        supported_direction_receipt.model_dump(mode="python")
    )
    if (
        direction.receipt_kind != "confirmation"
        or direction.decision
        != "prospective_opening_reversal_direction_supported"
    ):
        raise ValueError("option decision requires supported confirmation direction")
    recalculated = analyze_option_economics_v1(
        rows,
        underlying_direction_supported=True,
        capacity_blocked=result.decision == "option_economics_blocked_capacity",
        event_bootstrap_replications=CLUSTER_BOOTSTRAP_REPLICATIONS_V1,
    )
    if recalculated != result:
        raise ValueError("option decision rows differ from analysis result")
    if not rows:
        if (
            result.decision != "option_economics_blocked_capacity"
            or direction.cohort_first_session is None
            or direction.cohort_last_session is None
        ):
            raise ValueError("empty option cohort requires recorded capacity block")
        return build_opening_reversal_decision_receipt_v1(
            receipt_kind="option_economics",
            boundary_timestamp_utc=boundary_timestamp_utc,
            decision=result.decision,
            cohort_first_session=direction.cohort_first_session,
            cohort_last_session=direction.cohort_last_session,
            source_receipt_hashes=(direction.receipt_hash_v1,),
            support_counts={
                "complete_promoted_option_episodes": 0,
                "call_option_episodes": 0,
                "put_option_episodes": 0,
                "unique_severe_opening_events": 0,
                "represented_stocks": 0,
                "represented_expiries": 0,
                "maximum_stock_episode_count": 0,
                "maximum_expiry_episode_count": 0,
                "maximum_event_episode_count": 0,
            },
            protected_outcome_fields_accessed=True,
        )
    stock_counts = Counter(row.stock for row in rows)
    expiry_counts = Counter(row.expiry for row in rows)
    event_counts = Counter(row.opening_transition_event_id_v1 for row in rows)
    return build_opening_reversal_decision_receipt_v1(
        receipt_kind="option_economics",
        boundary_timestamp_utc=boundary_timestamp_utc,
        decision=result.decision,
        cohort_first_session=min(row.session for row in rows),
        cohort_last_session=max(row.session for row in rows),
        source_receipt_hashes=(
            direction.receipt_hash_v1,
            *tuple(
                sorted(
                    outcome_hash
                    for row in rows
                    for outcome_hash in (
                        row.predicted_leg_outcome_hash_v1,
                        row.opposite_leg_outcome_hash_v1,
                    )
                )
            ),
        ),
        support_counts={
            "complete_promoted_option_episodes": len(rows),
            "call_option_episodes": sum(
                row.prediction_v1 == "CALL" for row in rows
            ),
            "put_option_episodes": sum(
                row.prediction_v1 == "PUT" for row in rows
            ),
            "unique_severe_opening_events": len(event_counts),
            "represented_stocks": len(stock_counts),
            "represented_expiries": len(expiry_counts),
            "maximum_stock_episode_count": max(stock_counts.values()),
            "maximum_expiry_episode_count": max(expiry_counts.values()),
            "maximum_event_episode_count": max(event_counts.values()),
        },
        protected_outcome_fields_accessed=True,
    )


__all__ = [
    "CLUSTER_BOOTSTRAP_REPLICATIONS_V1",
    "EVENT_BOOTSTRAP_SEED_V1",
    "OpeningReversalAnalysisEpisodeV1",
    "OpeningReversalAnalysisResultV1",
    "OpeningReversalBaselinePopulationResultV1",
    "OpeningReversalDirectionSummaryV1",
    "OpeningReversalNullDrawV1",
    "OpeningReversalOptionEconomicsResultV1",
    "OpeningReversalOptionEpisodeV1",
    "OpeningReversalSupportSummaryV1",
    "PRIMARY_NULL_REPLICATIONS_V1",
    "PRIMARY_NULL_SEED_V1",
    "SESSION_BOOTSTRAP_SEED_V1",
    "WINSOR_FRACTION_V1",
    "analyze_direction_cohort_v1",
    "analyze_option_economics_v1",
    "build_opening_reversal_analysis_episode_v1",
    "build_opening_reversal_direction_decision_receipt_v1",
    "build_opening_reversal_option_decision_receipt_v1",
    "build_opening_reversal_option_episode_v1",
    "evaluate_support_v1",
]
