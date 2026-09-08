"""Current Session HARD package; legacy Structure D is calculation/provenance only."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stocker_execution.ibkr import QualifiedInstrument
    from stocker_execution.runtime import EntryBarSource, RuntimeStore

from stocker_core.markets import MarketId
from stocker_core.methods import ARTIFACTS, SESSION_HARD, content_hash, verified_q1_spec
from stocker_execution.session_hard_structure_d import (
    PRE_MOVE_THRESHOLD,
    CohortOpportunity,
    EntryBar,
    SessionHardStructureDStrategy,
    SignalStatus,
    StrategyContext,
    StrategyOpportunityKey,
    StrategySignal,
    _apply_first_touch,
)
from stocker_execution.stage5 import Stage5FeatureSnapshot


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """An actual observed print, with provider receipt order preserved."""

    timestamp: datetime
    price: float
    sequence: int

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or not isfinite(self.price) or self.price <= 0:
            raise ValueError("Trade event needs an aware timestamp and positive finite price")


class FrozenWhipsawModel:
    def __init__(self) -> None:
        import joblib

        self.spec = verified_q1_spec()  # Verify bytes before deserialising.
        self.columns = tuple(
            json.loads((ARTIFACTS / "MODEL_T0_parameters.json").read_text())["input_columns"]
        )
        self._model = joblib.load(ARTIFACTS / "MODEL_T0.joblib")

    def score(self, features: Mapping[str, float]) -> float:
        import pandas as pd

        if set(features) != set(self.columns):
            raise ValueError("MODEL_T0 requires the exact frozen feature columns")
        # Missing values use the saved FIT imputer, never a new fitted transform.
        values = pd.DataFrame([[features[c] for c in self.columns]], columns=self.columns)
        # Original research preprocessing precedes the saved pipeline.
        values = values.replace([float("inf"), float("-inf")], float("nan"))
        score = float(self._model.predict_proba(values)[0, 1])
        if not isfinite(score):
            raise ValueError("MODEL_T0 produced a nonfinite score")
        return score

    def admits(self, score: float) -> bool:
        return isfinite(score) and score <= self.spec["q1_risk_cutoff"]


class SessionHardMethod(SessionHardStructureDStrategy):
    """Reuse frozen T0 qualification; Q1 precedes arming; actual first break selects side."""

    def __init__(
        self, market: MarketId = MarketId.US_ALL, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        super().__init__(strategy_id=SESSION_HARD.method_id, strategy_version=SESSION_HARD.version)
        self.spec = SESSION_HARD.specification(market)
        self.spec_hash = content_hash(self.spec)
        self.model = FrozenWhipsawModel()
        self._clock = clock or (lambda: datetime.now(UTC))
        self.cohort_labels: dict[str, StrategySignal] = {}

    def restore_runtime_state(self, store: RuntimeStore, run_id: str) -> None:
        self.cohort_labels.update({s.signal_id: s for s in store.load_cohort_labels(run_id)})

    def save_runtime_state(self, store: RuntimeStore) -> None:
        store.save_cohort_labels(tuple(self.cohort_labels.values()))

    async def advance_runtime_state(
        self,
        source: EntryBarSource,
        instruments: Mapping[int, QualifiedInstrument],
        store: RuntimeStore,
        now: datetime,
        logger: Any,
    ) -> None:
        from stocker_execution.ibkr import IbkrError

        for signal_id, label in tuple(self.cohort_labels.items()):
            if label.status is not SignalStatus.WAITING_FOR_ENTRY:
                continue
            con_id = label.underlying_con_id
            if now < label.t0 + timedelta(minutes=5) or con_id is None or con_id not in instruments:
                continue
            try:
                bars = await source.cohort_bars(instruments[con_id], label, now)
                self.observe_cohort_bars(signal_id, bars)
                for opportunity_id, opportunity in tuple(self._cohort_opportunities.items()):
                    store.save_cohort(opportunity_id, opportunity)
                    self.acknowledge_cohort_update(opportunity_id)
                store.save_cohort_labels((self.cohort_labels[signal_id],))
            except (IbkrError, ValueError) as exc:
                logger.warning("cohort_label_unavailable", signal_id=signal_id, reason=str(exc))

    def observe_cohort_bars(self, signal_id: str, bars: Sequence[EntryBar]) -> None:
        """Original completed historical DOWN population, used only by later sessions.

        This label path never arms or enters a position, and is independent of Q1.
        It preserves the frozen cohort predictor instead of redefining it using
        only the new method's admitted trades.
        """
        label = self.cohort_labels[signal_id]
        outcome = _apply_first_touch(label, bars)
        self.cohort_labels[signal_id] = outcome
        if outcome.status is SignalStatus.ENTRY_TRIGGERED:
            assert label.pre_move_m is not None
            self._cohort_opportunities[signal_id] = CohortOpportunity(
                label.run_id, label.session, label.pre_move_m
            )

    def acknowledge_cohort_update(self, signal_id: str) -> None:
        """After persistence, score from the saved population without a duplicate copy."""
        self._cohort_opportunities.pop(signal_id, None)

    def expire_unobservable(self, con_id: int, reason: str) -> None:
        """A gap in the actual first-break prefix cannot be repaired with OHLC."""
        for signal_id, signal in tuple(self._signals.items()):
            if (
                signal.underlying_con_id == con_id
                and signal.status is SignalStatus.WAITING_FOR_ENTRY
            ):
                self._signals[signal_id] = replace(
                    signal, status=SignalStatus.EXPIRED, reason=reason
                )

    def expire_waiting_before(self, now: datetime) -> tuple[StrategySignal, ...]:
        """The half-open entry window also expires when no new print arrives."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("strategy expiry time must be timezone-aware")
        expired = []
        for signal_id, signal in tuple(self._signals.items()):
            if (
                signal.status is SignalStatus.WAITING_FOR_ENTRY
                and now >= signal.t0 + timedelta(minutes=5)
            ):
                updated = replace(
                    signal, status=SignalStatus.EXPIRED, reason="ENTRY_WINDOW_EXPIRED"
                )
                self._signals[signal_id] = updated
                self._cohort_watch_ids.discard(signal_id)
                self._strategy_candidate_ids.discard(signal_id)
                expired.append(updated)
        return tuple(expired)

    def evaluate(
        self, feature_rows: Sequence[Stage5FeatureSnapshot], context: StrategyContext
    ) -> tuple[StrategySignal, ...]:
        previous = set(self._signals)
        result = []
        for signal in super().evaluate(feature_rows, context):
            if signal.signal_id in previous:
                result.append(signal)
                continue
            if (
                signal.session_hard_qualified
                and signal.pre_move_m is not None
                and signal.pre_move_m > PRE_MOVE_THRESHOLD
                and (
                    signal.status is SignalStatus.WAITING_FOR_ENTRY
                    or signal.reason == "COHORT_MID_VETO"
                )
                and signal.p0 is not None
                and signal.m_price is not None
            ):
                self.cohort_labels[signal.signal_id] = replace(
                    signal,
                    status=SignalStatus.WAITING_FOR_ENTRY,
                    reason="HISTORICAL_COHORT_LABEL_ONLY",
                    selected=False,
                )
            signal = replace(
                signal,
                market_id=self.spec["market"],
                method_spec_hash=self.spec_hash,
                artifact_hashes=self.spec["artifact_hashes"],
                stop_distance_m=self.spec["exits"]["stop_M"],
                target_distance_m=self.spec["exits"]["target_M"],
                deadline=signal.t0 + timedelta(minutes=15),
            )
            if signal.status is SignalStatus.WAITING_FOR_ENTRY:
                assert signal.underlying_con_id is not None
                assert signal.p0 is not None and signal.m_price is not None
                key = StrategyOpportunityKey(signal.underlying_con_id, signal.session, signal.t0)
                features = context.whipsaw_features.get(key)
                if features is None:
                    signal = replace(
                        signal,
                        status=SignalStatus.NOT_QUALIFIED,
                        reason="MODEL_T0_INPUTS_UNAVAILABLE",
                    )
                else:
                    values = dict(features)
                    values["cohort_percentile"] = (
                        signal.cohort_percentile
                        if signal.cohort_percentile is not None
                        else float("nan")
                    )
                    risk = self.model.score(values)
                    admitted = self.model.admits(risk)
                    assert signal.p0 is not None and signal.m_price is not None
                    signal = replace(
                        signal,
                        whipsaw_risk_score=risk,
                        q1_eligible=admitted,
                        status=SignalStatus.WAITING_FOR_ENTRY
                        if admitted
                        else SignalStatus.NOT_QUALIFIED,
                        reason="ARMED_FIRST_BREAK" if admitted else "MODEL_T0_Q1_VETO",
                        armed_at=max(context.available_at or signal.t0, self._clock())
                        if admitted
                        else None,
                        up_trigger=signal.p0 + 0.20 * signal.m_price,
                        down_trigger=signal.p0 - 0.20 * signal.m_price,
                        entry_level=None,
                    )
            self._signals[signal.signal_id] = signal
            result.append(signal)
        # The old completed-bar entry/ranking path cannot act on current candidates.
        self._cohort_watch_ids.clear()
        self._strategy_candidate_ids.clear()
        return tuple(result)

    def observe_entry_bars(
        self, bars_by_con_id: Mapping[int, Sequence[EntryBar]]
    ) -> tuple[StrategySignal, ...]:
        raise ValueError(
            "Session HARD causal entry requires ordered TRADES events, not completed bars"
        )

    def observe_trades(
        self, events_by_con_id: Mapping[int, Sequence[TradeEvent]]
    ) -> tuple[StrategySignal, ...]:
        changed = []
        for signal_id, signal in tuple(self._signals.items()):
            if signal.status is not SignalStatus.WAITING_FOR_ENTRY or not signal.q1_eligible:
                continue
            assert signal.underlying_con_id is not None
            for event in events_by_con_id.get(signal.underlying_con_id, ()):
                if (
                    signal.last_event_sequence is not None
                    and event.sequence <= signal.last_event_sequence
                ):
                    continue
                if event.timestamp < signal.t0:
                    continue
                signal = replace(signal, last_event_sequence=event.sequence)
                if event.timestamp >= signal.t0 + timedelta(minutes=5):
                    signal = replace(
                        signal, status=SignalStatus.EXPIRED, reason="ENTRY_WINDOW_EXPIRED"
                    )
                    break
                assert signal.up_trigger is not None and signal.down_trigger is not None
                direction = (
                    1
                    if event.price >= signal.up_trigger
                    else -1
                    if event.price <= signal.down_trigger
                    else 0
                )
                if not direction:
                    continue
                if signal.armed_at is None or event.timestamp < signal.armed_at:
                    signal = replace(
                        signal, status=SignalStatus.EXPIRED, reason="FIRST_BREAK_PRECEDED_ARMING"
                    )
                    break
                level = signal.up_trigger if direction == 1 else signal.down_trigger
                assert signal.m_price is not None
                signal = replace(
                    signal,
                    status=SignalStatus.ENTRY_TRIGGERED,
                    reason="CAUSAL_FIRST_BREAK",
                    side="LONG" if direction == 1 else "SHORT",
                    direction="UP" if direction == 1 else "DOWN",
                    selected=True,
                    entry_level=level,
                    entry_reference=level,
                    entry_timestamp=event.timestamp,
                    signal_timestamp=event.timestamp,
                    stop_price=level - direction * self.spec["exits"]["stop_M"] * signal.m_price,
                    target_price=level
                    + direction * self.spec["exits"]["target_M"] * signal.m_price,
                )
                break
            if signal != self._signals[signal_id]:
                self._signals[signal_id] = signal
                changed.append(signal)
        return tuple(changed)
