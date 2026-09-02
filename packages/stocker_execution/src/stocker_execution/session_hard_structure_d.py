"""Concrete Stage 6 Session HARD / Structure D strategy."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from math import exp, isfinite

from stocker_execution.stage5 import Stage5FeatureSnapshot, Stage5Status

SESSION_HARD_THRESHOLD = 0.999361477
PRE_MOVE_THRESHOLD = 0.475764059845861
ENTRY_TRIGGER_M = 0.20
ENTRY_WINDOW_MINUTES = 5
CANDIDATE_CAPACITY = 5
STRATEGY_ID = "SESSION_HARD_HIGH_PRE_MOVE_DOWN_STRUCTURE_D"
STRATEGY_VERSION = "SESSION_HARD_STRUCTURE_D_V1"
COHORT_MINIMUM_PRIOR_OBSERVATIONS = 30
LOW_PERCENTILE_MAX = 33.33
MID_PERCENTILE_MAX = 66.67
SESSION_HARD_CHECKPOINTS = tuple(range(6, 35, 2))
SESSION_HARD_FEATURES = (
    "range_effort",
    "travel_effort",
    "absolute_efficiency",
    "close_retention",
    "directional_persistence",
    "prior_6_mean_range",
    "prior_6_price_travel",
    "prior_6_absolute_net_movement",
    "recent_vs_earlier_range_ratio",
    "current_bar_range_vs_prior_6",
    "current_bar_body_fraction",
    "current_bar_extreme_wick_fraction",
    "current_volume_vs_session_mean",
    "current_volume_vs_prior6_mean",
    "last3_vs_first3_volume",
)

_SESSION_HARD_MEANS = (
    7.520285144182087,
    6.8540313024793065,
    0.24647864315562662,
    0.12813516597006494,
    0.5562294730600414,
    87.98977427442436,
    279.4119010677174,
    111.23431762828353,
    0.9498247222802021,
    0.9165072264768855,
    0.4903089190160964,
    0.390732639199654,
    0.6369055061739176,
    0.9393887792276489,
    0.40677150957392233,
)
_SESSION_HARD_SCALES = (
    0.5735147628516906,
    0.6219857918102291,
    0.19077115947542334,
    0.10150863880092864,
    0.10229093200406345,
    52.10975964878536,
    190.3181927258607,
    115.20946398218234,
    0.35725549519834693,
    0.33322402857989875,
    0.2817875724878281,
    0.22184954023897208,
    0.43983431043899246,
    1.0456493701618994,
    0.3137951573538811,
)
_SESSION_HARD_FEATURE_COEFFICIENTS = (
    0.9799803580175513,
    -0.8944637698447322,
    -0.07524779609903433,
    -0.19654640811260027,
    -0.2182218060043288,
    -1.7454784536650403,
    0.4462862161511877,
    -0.2492856268353956,
    0.05102541038778435,
    -0.16914735966972935,
    0.309123120727562,
    0.3038384994273069,
    0.19921027619561069,
    -0.5240104229482786,
    -0.24540204020190584,
)
_SESSION_HARD_CHECKPOINT_COEFFICIENTS = (
    -0.602829915528228,
    -0.59297939657289,
    -0.7588170071651871,
    -0.5214946228076957,
    -0.8535869818880394,
    -0.8015011139327045,
    -0.6187749101755738,
    -0.760828087775292,
    -0.6880981932383786,
    0.21875971420084145,
    0.7338756683883667,
    1.0246151079514028,
    0.3996463121878181,
    1.6107018877859098,
    1.9615281816178574,
)
_SESSION_HARD_INTERCEPT = -3.5035822734365243


class PreMoveBand(StrEnum):
    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"


class SignalStatus(StrEnum):
    WAITING_FOR_ENTRY = "WAITING_FOR_ENTRY"
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    NOT_QUALIFIED = "NOT_QUALIFIED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class CohortOpportunity:
    run_id: str
    session: date
    pre_move_m: float

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("cohort opportunity run_id is required")
        if not isfinite(self.pre_move_m) or self.pre_move_m < 0.0:
            raise ValueError("cohort opportunity PRE_MOVE_M must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class StrategyContext:
    run_id: str
    session_hard_scores: Mapping[int, float]
    session_hard_checkpoints: Mapping[int, int] = field(default_factory=dict)
    cohort_history: tuple[CohortOpportunity, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("strategy run_id is required")
        if any(
            checkpoint not in SESSION_HARD_CHECKPOINTS
            for checkpoint in self.session_hard_checkpoints.values()
        ):
            raise ValueError("strategy context contains an unsupported Session HARD checkpoint")


@dataclass(frozen=True, slots=True)
class StrategySignal:
    strategy_id: str
    strategy_version: str
    signal_id: str
    run_id: str
    underlying_con_id: int | None
    symbol: str
    universe_id: str
    session: date
    t0: datetime
    status: SignalStatus
    reason: str
    pre_move_m: float | None
    cohort_percentile: float | None
    band: PreMoveBand | None
    session_hard_score: float | None
    session_hard_checkpoint: int | None
    session_hard_qualified: bool
    feature_calculation_version: str
    side: str | None = None
    direction: str | None = None
    candidate_rank: int | None = None
    selected: bool = False
    p0: float | None = None
    m_price: float | None = None
    entry_level: float | None = None
    entry_reference: float | None = None
    entry_timestamp: datetime | None = None
    signal_timestamp: datetime | None = None
    stop_distance_m: float = 0.50
    target_distance_m: float = 1.00


@dataclass(frozen=True, slots=True)
class EntryBar:
    timestamp: datetime
    open: float
    high: float
    low: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("entry bar timestamp must be timezone-aware")
        values = (self.open, self.high, self.low)
        if not all(isfinite(value) and value > 0.0 for value in values):
            raise ValueError("entry bar prices must be finite and positive")
        if self.high < max(self.open, self.low) or self.low > min(self.open, self.high):
            raise ValueError("entry bar OHLC geometry is invalid")


class SessionHardStructureDStrategy:
    """Evaluate one concrete strategy without account or broker behavior."""

    def __init__(self) -> None:
        self._signals: dict[str, StrategySignal] = {}
        self._cohort_watch_ids: set[str] = set()
        self._cohort_opportunities: dict[str, CohortOpportunity] = {}
        self._observed_entry_bars: dict[str, dict[datetime, EntryBar]] = {}

    @property
    def cohort_opportunities(self) -> tuple[CohortOpportunity, ...]:
        """Return baseline DOWN opportunities discovered by this strategy instance."""

        return tuple(self._cohort_opportunities.values())

    @property
    def signals(self) -> tuple[StrategySignal, ...]:
        """Return the latest deterministic state of each strategy opportunity."""

        return tuple(
            sorted(
                self._signals.values(),
                key=lambda item: (item.t0, item.symbol, item.signal_id),
            )
        )

    def evaluate(
        self,
        feature_rows: Sequence[Stage5FeatureSnapshot],
        context: StrategyContext,
    ) -> tuple[StrategySignal, ...]:
        signals: list[StrategySignal] = []
        for row in feature_rows:
            if context.run_id not in row.run_ids:
                continue
            signal_id = _signal_id(row, context.run_id)
            existing = self._signals.get(signal_id)
            if existing is not None:
                signals.append(existing)
                continue
            score = context.session_hard_scores.get(row.con_id) if row.con_id is not None else None
            pre_move_m = row.pre_move_m
            ready = (
                row.status is Stage5Status.READY
                and row.con_id is not None
                and pre_move_m is not None
                and row.p0 is not None
                and row.m_price is not None
            )
            percentile = None
            band = None
            if ready:
                assert pre_move_m is not None
                percentile = calculate_cohort_percentile(
                    pre_move_m,
                    _prior_cohort_values(
                        (*context.cohort_history, *self._cohort_opportunities.values()),
                        context.run_id,
                        row.session,
                    ),
                )
                band = classify_pre_move_band(percentile)
            if not ready:
                status = SignalStatus.NOT_QUALIFIED
                reason = row.exclusion_reason or "STAGE5_NOT_READY"
            elif pre_move_m is None or not isfinite(pre_move_m) or pre_move_m <= PRE_MOVE_THRESHOLD:
                status = SignalStatus.NOT_QUALIFIED
                reason = "PRE_MOVE_THRESHOLD"
            elif score is None or not isfinite(score):
                status = SignalStatus.NOT_QUALIFIED
                reason = "SESSION_HARD_SCORE_UNAVAILABLE"
            elif score < SESSION_HARD_THRESHOLD:
                status = SignalStatus.NOT_QUALIFIED
                reason = "SESSION_HARD"
            elif band is PreMoveBand.MID:
                status = SignalStatus.NOT_QUALIFIED
                reason = "COHORT_MID_VETO"
            else:
                status = SignalStatus.WAITING_FOR_ENTRY
                reason = "AWAITING_STRUCTURE_D_DOWN_FIRST_TOUCH"
            signal = StrategySignal(
                strategy_id=STRATEGY_ID,
                strategy_version=STRATEGY_VERSION,
                signal_id=signal_id,
                run_id=context.run_id,
                underlying_con_id=row.con_id,
                symbol=row.symbol,
                universe_id=row.universe_id,
                session=row.session,
                t0=row.t0,
                status=status,
                reason=reason,
                pre_move_m=pre_move_m,
                cohort_percentile=percentile,
                band=band,
                session_hard_score=score,
                session_hard_checkpoint=(
                    context.session_hard_checkpoints.get(row.con_id)
                    if row.con_id is not None
                    else None
                ),
                session_hard_qualified=(
                    score is not None and isfinite(score) and score >= SESSION_HARD_THRESHOLD
                ),
                feature_calculation_version=row.calculation_version,
                p0=row.p0,
                m_price=row.m_price,
                entry_level=(
                    row.p0 - ENTRY_TRIGGER_M * row.m_price
                    if row.p0 is not None and row.m_price is not None
                    else None
                ),
            )
            self._signals[signal_id] = signal
            if (
                ready
                and pre_move_m is not None
                and pre_move_m > PRE_MOVE_THRESHOLD
                and score is not None
                and isfinite(score)
                and score >= SESSION_HARD_THRESHOLD
            ):
                self._cohort_watch_ids.add(signal_id)
            signals.append(signal)
        return tuple(signals)

    def observe_entry_bars(
        self, bars_by_con_id: Mapping[int, Sequence[EntryBar]]
    ) -> tuple[StrategySignal, ...]:
        """Advance waiting candidates using only frozen one-minute first-touch bars."""

        changed: list[StrategySignal] = []
        for signal_id, signal in tuple(self._signals.items()):
            if (
                signal_id not in self._cohort_watch_ids
                or signal.underlying_con_id is None
                or signal.p0 is None
                or signal.m_price is None
            ):
                continue
            observed = bars_by_con_id.get(signal.underlying_con_id)
            if not observed:
                continue
            accumulated = self._observed_entry_bars.setdefault(signal_id, {})
            for bar in observed:
                accumulated[bar.timestamp.astimezone(UTC)] = bar
            observed_touch = _apply_first_touch(signal, tuple(accumulated.values()))
            if observed_touch.status is SignalStatus.ENTRY_TRIGGERED:
                assert signal.pre_move_m is not None
                self._cohort_opportunities.setdefault(
                    signal_id,
                    CohortOpportunity(signal.run_id, signal.session, signal.pre_move_m),
                )
            if signal.status is SignalStatus.NOT_QUALIFIED:
                updated = replace(
                    observed_touch,
                    status=signal.status,
                    reason=signal.reason,
                    side=None,
                    selected=False,
                )
            else:
                updated = observed_touch
            if observed_touch.status in {
                SignalStatus.ENTRY_TRIGGERED,
                SignalStatus.EXPIRED,
            }:
                self._cohort_watch_ids.discard(signal_id)
            self._signals[signal_id] = updated
            changed.append(updated)

        triggered_by_timestamp: dict[datetime, list[StrategySignal]] = {}
        for signal in changed:
            if signal.status is SignalStatus.ENTRY_TRIGGERED and signal.entry_timestamp is not None:
                triggered_by_timestamp.setdefault(signal.entry_timestamp, []).append(signal)
        for simultaneous in triggered_by_timestamp.values():
            ranked = sorted(
                simultaneous,
                key=lambda item: (
                    -(item.session_hard_score or 0.0),
                    item.symbol,
                    _ranking_row_id(item),
                ),
            )
            for rank, signal in enumerate(ranked, start=1):
                if rank <= CANDIDATE_CAPACITY:
                    updated = replace(signal, candidate_rank=rank, selected=True)
                else:
                    updated = replace(
                        signal,
                        status=SignalStatus.NOT_QUALIFIED,
                        reason="STRATEGY_CANDIDATE_CAPACITY",
                        candidate_rank=rank,
                        selected=False,
                    )
                self._signals[signal.signal_id] = updated
                changed[changed.index(signal)] = updated
        return tuple(
            sorted(
                changed,
                key=lambda item: (
                    item.entry_timestamp or item.t0,
                    item.candidate_rank or CANDIDATE_CAPACITY + 1,
                    item.symbol,
                    item.signal_id,
                ),
            )
        )


def _signal_id(row: Stage5FeatureSnapshot, run_id: str) -> str:
    identity = "|".join(
        (
            STRATEGY_ID,
            STRATEGY_VERSION,
            run_id,
            str(row.con_id),
            row.session.isoformat(),
            row.t0.isoformat(),
            row.calculation_version,
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:32]


def _ranking_row_id(signal: StrategySignal) -> str:
    checkpoint = (
        str(signal.session_hard_checkpoint)
        if signal.session_hard_checkpoint is not None
        else signal.t0.isoformat()
    )
    return f"{signal.symbol}|{signal.session.isoformat()}|{checkpoint}"


def _prior_cohort_values(
    history: Sequence[CohortOpportunity], run_id: str, current_session: date
) -> tuple[float, ...]:
    eligible = tuple(
        item for item in history if item.run_id == run_id and item.session < current_session
    )
    sessions = sorted({item.session for item in eligible})[-20:]
    selected_sessions = set(sessions)
    return tuple(item.pre_move_m for item in eligible if item.session in selected_sessions)


def _apply_first_touch(signal: StrategySignal, observed: Sequence[EntryBar]) -> StrategySignal:
    assert signal.p0 is not None and signal.m_price is not None
    t0 = signal.t0.astimezone(UTC)
    upper = signal.p0 + ENTRY_TRIGGER_M * signal.m_price
    lower = signal.p0 - ENTRY_TRIGGER_M * signal.m_price
    window_end = t0 + timedelta(minutes=ENTRY_WINDOW_MINUTES - 1)
    bars = sorted(
        (bar for bar in observed if t0 <= bar.timestamp.astimezone(UTC) <= window_end),
        key=lambda bar: bar.timestamp,
    )
    for bar in bars:
        timestamp = bar.timestamp.astimezone(UTC)
        if bar.open >= upper:
            return replace(
                signal,
                status=SignalStatus.EXPIRED,
                reason="STRUCTURE_D_UP_FIRST_TOUCH",
                direction="UP",
                entry_level=lower,
            )
        if bar.open <= lower:
            return replace(
                signal,
                status=SignalStatus.ENTRY_TRIGGERED,
                reason="STRUCTURE_D_DOWN_FIRST_TOUCH",
                side="SHORT",
                direction="DOWN",
                selected=True,
                entry_level=lower,
                entry_reference=bar.open,
                entry_timestamp=timestamp,
                signal_timestamp=timestamp,
            )
        up = bar.high >= upper
        down = bar.low <= lower
        if up and down:
            return replace(
                signal,
                status=SignalStatus.EXPIRED,
                reason="STRUCTURE_D_FIRST_TOUCH_AMBIGUOUS",
                entry_level=lower,
            )
        if up:
            return replace(
                signal,
                status=SignalStatus.EXPIRED,
                reason="STRUCTURE_D_UP_FIRST_TOUCH",
                direction="UP",
                entry_level=lower,
            )
        if down:
            return replace(
                signal,
                status=SignalStatus.ENTRY_TRIGGERED,
                reason="STRUCTURE_D_DOWN_FIRST_TOUCH",
                side="SHORT",
                direction="DOWN",
                selected=True,
                entry_level=lower,
                entry_reference=lower,
                entry_timestamp=timestamp,
                signal_timestamp=timestamp + timedelta(minutes=1),
            )
    expected = {t0 + timedelta(minutes=index) for index in range(ENTRY_WINDOW_MINUTES)}
    observed_timestamps = {bar.timestamp.astimezone(UTC) for bar in bars}
    if expected.issubset(observed_timestamps):
        return replace(
            signal,
            status=SignalStatus.EXPIRED,
            reason="STRUCTURE_D_NO_DOWN_FIRST_TOUCH",
            entry_level=lower,
        )
    return replace(signal, entry_level=lower)


def calculate_cohort_percentile(
    current_pre_move_m: float, prior_qualifying_pre_move_m: tuple[float, ...]
) -> float | None:
    """Return the frozen empirical percentile over prior baseline opportunities."""

    current = float(current_pre_move_m)
    if not isfinite(current) or current < 0.0:
        raise ValueError("current PRE_MOVE_M must be finite and nonnegative")
    history = tuple(float(value) for value in prior_qualifying_pre_move_m)
    if any(not isfinite(value) or value < 0.0 for value in history):
        raise ValueError("prior PRE_MOVE_M values must be finite and nonnegative")
    if len(history) < COHORT_MINIMUM_PRIOR_OBSERVATIONS:
        return None
    return 100.0 * sum(value <= current for value in history) / len(history)


def classify_pre_move_band(percentile: float | None) -> PreMoveBand | None:
    """Apply the frozen unrounded LOW/MID/HIGH percentile boundaries."""

    if percentile is None:
        return None
    value = float(percentile)
    if not isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError("cohort percentile must be between zero and 100")
    if value <= LOW_PERCENTILE_MAX:
        return PreMoveBand.LOW
    if value <= MID_PERCENTILE_MAX:
        return PreMoveBand.MID
    return PreMoveBand.HIGH


def calculate_session_hard_score(*, checkpoint: int, features: Mapping[str, float]) -> float:
    """Apply the frozen Session HARD Model B scorer to causal prepared features."""

    if checkpoint not in SESSION_HARD_CHECKPOINTS:
        raise ValueError(f"unsupported Session HARD checkpoint: {checkpoint}")
    if set(features) != set(SESSION_HARD_FEATURES):
        raise ValueError("Session HARD requires the exact frozen feature set")

    linear = _SESSION_HARD_INTERCEPT
    for name, mean, scale, coefficient in zip(
        SESSION_HARD_FEATURES,
        _SESSION_HARD_MEANS,
        _SESSION_HARD_SCALES,
        _SESSION_HARD_FEATURE_COEFFICIENTS,
        strict=True,
    ):
        value = float(features[name])
        if not isfinite(value):
            raise ValueError(f"Session HARD feature {name} must be finite")
        linear += ((value - mean) / scale) * coefficient
    linear += _SESSION_HARD_CHECKPOINT_COEFFICIENTS[SESSION_HARD_CHECKPOINTS.index(checkpoint)]
    quiet_probability = 1.0 / (1.0 + exp(-linear))
    return 1.0 - quiet_probability
