"""Frozen, local-only HARD_M1C T20/D shadow state machine.

The module consumes ordinary Level-I BBO observations and never imports a
broker adapter or any microstructure-derived feature.  It models research
state only; no transition represents or requests a broker order.
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

STRATEGY_VERSION_V0: Literal["m1c_directionless_t20_d_v0"] = "m1c_directionless_t20_d_v0"
FAMILY_V0: Literal["HARD_M1C"] = "HARD_M1C"
TRIGGER_M_V0 = 0.20
STOP_M_V0 = 0.50
TARGET_M_V0 = 1.00
ENTRY_MINUTES_V0 = 5
HORIZON_MINUTES_V0 = 15
ROUND_TRIP_COST_BPS_V0 = 10.0


class DirectionlessShadowStateV0(StrEnum):
    CREATED = "CREATED"
    WAITING_FOR_ENTRY = "WAITING_FOR_ENTRY"
    LONG_ACTIVE = "LONG_ACTIVE"
    SHORT_ACTIVE = "SHORT_ACTIVE"
    TARGET_EXITED = "TARGET_EXITED"
    STOP_EXITED = "STOP_EXITED"
    TIME_EXITED = "TIME_EXITED"
    NO_ENTRY_TIMEOUT = "NO_ENTRY_TIMEOUT"
    ENTRY_UNRESOLVED = "ENTRY_UNRESOLVED"
    ACTIVE_PATH_UNRESOLVED = "ACTIVE_PATH_UNRESOLVED"


TERMINAL_STATES_V0 = frozenset(
    {
        DirectionlessShadowStateV0.TARGET_EXITED,
        DirectionlessShadowStateV0.STOP_EXITED,
        DirectionlessShadowStateV0.TIME_EXITED,
        DirectionlessShadowStateV0.NO_ENTRY_TIMEOUT,
        DirectionlessShadowStateV0.ENTRY_UNRESOLVED,
        DirectionlessShadowStateV0.ACTIVE_PATH_UNRESOLVED,
    }
)


class ShadowBBOObservationV0(BaseModel):
    """One causally ordered ordinary Level-I quote used as source evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    ordering_timestamp: datetime
    receive_timestamp: datetime
    local_sequence: int = Field(ge=0)
    connection_generation: int = Field(ge=0)
    bid: float
    ask: float

    @model_validator(mode="after")
    def _timestamps_are_aware(self) -> ShadowBBOObservationV0:
        for value in (self.ordering_timestamp, self.receive_timestamp):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("BBO timestamps must be timezone-aware")
        return self

    @property
    def valid(self) -> bool:
        return (
            math.isfinite(self.bid)
            and math.isfinite(self.ask)
            and self.bid > 0.0
            and self.ask >= self.bid
        )

    @property
    def order_key(self) -> tuple[datetime, int, datetime, str]:
        return (
            self.ordering_timestamp.astimezone(UTC),
            self.local_sequence,
            self.receive_timestamp.astimezone(UTC),
            self.event_id,
        )


class DirectionlessShadowEpisodeV0(BaseModel):
    """Durable state for one immutable HARD_M1C episode/version pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shadow_episode_id: str
    strategy_version: Literal["m1c_directionless_t20_d_v0"]
    m1c_episode_id: str
    symbol: str
    con_id: int
    session: date
    t0: datetime
    p0: float
    movement_price: float
    movement_scale_fraction: float | None = None
    family: Literal["HARD_M1C"]
    probability: float
    consumed_ratio: float | None
    upper_trigger: float
    lower_trigger: float
    entry_deadline: datetime
    horizon: datetime
    state: DirectionlessShadowStateV0
    direction: Literal["LONG", "SHORT"] | None = None
    entry_timestamp: datetime | None = None
    entry_price: float | None = None
    entry_source_event_id: str | None = None
    entry_gap: bool = False
    observed_market_price_at_detection: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    exit_timestamp: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    exit_source_event_id: str | None = None
    MFE_price: float = 0.0
    MAE_price: float = 0.0
    MFE_M: float = 0.0
    MAE_M: float = 0.0
    MFE_R: float = 0.0
    MAE_R: float = 0.0
    gross_price_return: float | None = None
    gross_M: float | None = None
    gross_R: float | None = None
    cost_price: float | None = None
    cost_M: float | None = None
    net_M: float | None = None
    net_R: float | None = None
    opposite_trigger_touched: bool = False
    time_to_opposite_trigger_seconds: float | None = None
    opposite_within_1m: bool = False
    opposite_within_5m: bool = False
    opposite_before_horizon: bool = False
    gap_id: str | None = None
    gap_start: datetime | None = None
    gap_end: datetime | None = None
    gap_connection_generation: int | None = None
    last_valid_price_before_gap: float | None = None
    first_restored_price: float | None = None
    last_source_event_id: str | None = None
    last_ordering_timestamp: datetime | None = None
    last_local_sequence: int | None = None
    last_bid: float | None = None
    last_ask: float | None = None
    code_hash: str
    config_hash: str
    m1c_artifact_hash: str
    created_at: datetime
    updated_at: datetime

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES_V0


def create_directionless_shadow_v0(
    *,
    m1c_episode_id: str,
    symbol: str,
    con_id: int,
    session: date,
    t0: datetime,
    p0: float,
    movement_price: float,
    probability: float,
    consumed_ratio: float | None,
    code_hash: str,
    config_hash: str,
    m1c_artifact_hash: str,
    family: Literal["HARD_M1C"] = FAMILY_V0,
    movement_scale_fraction: float | None = None,
    created_at: datetime | None = None,
) -> DirectionlessShadowEpisodeV0:
    """Create one V0 state using already-frozen M1C evidence."""

    if family != FAMILY_V0:
        raise ValueError("directionless shadow V0 accepts HARD_M1C only")
    if t0.tzinfo is None or t0.utcoffset() is None:
        raise ValueError("T0 must be timezone-aware")
    if not math.isfinite(p0) or p0 <= 0.0:
        raise ValueError("P0 must be finite and positive")
    if not math.isfinite(movement_price) or movement_price <= 0.0:
        raise ValueError("M must be finite and positive")
    timestamp = (created_at or t0).astimezone(UTC)
    t0_utc = t0.astimezone(UTC)
    identity = f"{STRATEGY_VERSION_V0}|{m1c_episode_id}"
    return DirectionlessShadowEpisodeV0(
        shadow_episode_id=f"dsv0-{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
        strategy_version=STRATEGY_VERSION_V0,
        m1c_episode_id=m1c_episode_id,
        symbol=symbol,
        con_id=con_id,
        session=session,
        t0=t0_utc,
        p0=p0,
        movement_price=movement_price,
        movement_scale_fraction=movement_scale_fraction,
        family=FAMILY_V0,
        probability=probability,
        consumed_ratio=consumed_ratio,
        upper_trigger=p0 + TRIGGER_M_V0 * movement_price,
        lower_trigger=p0 - TRIGGER_M_V0 * movement_price,
        entry_deadline=t0_utc + timedelta(minutes=ENTRY_MINUTES_V0),
        horizon=t0_utc + timedelta(minutes=HORIZON_MINUTES_V0),
        state=DirectionlessShadowStateV0.WAITING_FOR_ENTRY,
        code_hash=code_hash,
        config_hash=config_hash,
        m1c_artifact_hash=m1c_artifact_hash,
        created_at=timestamp,
        updated_at=timestamp,
    )


class FrozenDirectionlessShadowV0:
    """Deterministic monotonic transition engine for one shadow episode."""

    def __init__(self, episode: DirectionlessShadowEpisodeV0) -> None:
        self._episode = episode
        self._seen_event_ids: set[str] = set()
        self._last_order_key: tuple[datetime, int, datetime, str] | None = None
        self._gap_open: tuple[datetime, int, str, float | None] | None = None
        self._restored_gap: tuple[datetime, datetime, int, str, float | None] | None = None
        if (
            episode.last_source_event_id is not None
            and episode.last_ordering_timestamp is not None
            and episode.last_local_sequence is not None
        ):
            self._seen_event_ids.add(episode.last_source_event_id)
            self._last_order_key = (
                episode.last_ordering_timestamp.astimezone(UTC),
                episode.last_local_sequence,
                episode.updated_at.astimezone(UTC),
                episode.last_source_event_id,
            )

    @property
    def episode(self) -> DirectionlessShadowEpisodeV0:
        return self._episode

    def advance_time(self, now: datetime) -> DirectionlessShadowEpisodeV0:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must be timezone-aware")
        if (
            self._episode.state is DirectionlessShadowStateV0.WAITING_FOR_ENTRY
            and now.astimezone(UTC) > self._episode.entry_deadline
        ):
            self._episode = self._episode.model_copy(
                update={
                    "state": DirectionlessShadowStateV0.NO_ENTRY_TIMEOUT,
                    "exit_reason": "NO_ENTRY_TIMEOUT",
                    "updated_at": now.astimezone(UTC),
                }
            )
        return self._episode

    def connection_lost(self, at: datetime, *, generation: int, gap_id: str) -> None:
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("gap timestamp must be timezone-aware")
        last_price = self._liquidation_price_from_episode()
        self._gap_open = (at.astimezone(UTC), generation, gap_id, last_price)

    def connection_restored(self, at: datetime, *, generation: int, gap_id: str) -> None:
        if self._gap_open is None or self._gap_open[2] != gap_id:
            return
        started_at, lost_generation, _, last_price = self._gap_open
        self._restored_gap = (
            started_at,
            at.astimezone(UTC),
            lost_generation,
            gap_id,
            last_price,
        )
        self._gap_open = None

    def observe(self, observation: ShadowBBOObservationV0) -> DirectionlessShadowEpisodeV0:
        if self._episode.terminal or observation.event_id in self._seen_event_ids:
            return self._episode
        self._seen_event_ids.add(observation.event_id)
        if not observation.valid:
            return self._episode
        if self._last_order_key is not None and observation.order_key <= self._last_order_key:
            return self._episode
        self._last_order_key = observation.order_key
        if observation.ordering_timestamp < self._episode.t0:
            return self._episode

        if self._restored_gap is not None:
            return self._resolve_gap_with(observation)

        if self._episode.state is DirectionlessShadowStateV0.WAITING_FOR_ENTRY:
            return self._observe_waiting(observation)
        if self._episode.state in {
            DirectionlessShadowStateV0.LONG_ACTIVE,
            DirectionlessShadowStateV0.SHORT_ACTIVE,
        }:
            return self._observe_active(observation)
        return self._episode

    def _observe_waiting(self, observation: ShadowBBOObservationV0) -> DirectionlessShadowEpisodeV0:
        timestamp = observation.ordering_timestamp.astimezone(UTC)
        if timestamp > self._episode.entry_deadline:
            return self.advance_time(timestamp)
        long_touch = observation.ask >= self._episode.upper_trigger
        short_touch = observation.bid <= self._episode.lower_trigger
        if long_touch and short_touch:
            return self._terminal_unresolved(
                observation,
                state=DirectionlessShadowStateV0.ENTRY_UNRESOLVED,
                reason="ENTRY_PATH_UNRESOLVED",
            )
        if not long_touch and not short_touch:
            self._remember(observation)
            return self._episode
        direction: Literal["LONG", "SHORT"] = "LONG" if long_touch else "SHORT"
        entry_price = self._episode.upper_trigger if long_touch else self._episode.lower_trigger
        observed_price = observation.ask if long_touch else observation.bid
        stop = (
            entry_price - STOP_M_V0 * self._episode.movement_price
            if long_touch
            else entry_price + STOP_M_V0 * self._episode.movement_price
        )
        target = (
            entry_price + TARGET_M_V0 * self._episode.movement_price
            if long_touch
            else entry_price - TARGET_M_V0 * self._episode.movement_price
        )
        state = (
            DirectionlessShadowStateV0.LONG_ACTIVE
            if long_touch
            else DirectionlessShadowStateV0.SHORT_ACTIVE
        )
        self._episode = self._episode.model_copy(
            update={
                "state": state,
                "direction": direction,
                "entry_timestamp": timestamp,
                "entry_price": entry_price,
                "entry_source_event_id": observation.event_id,
                "entry_gap": not math.isclose(observed_price, entry_price),
                "observed_market_price_at_detection": observed_price,
                "stop_price": stop,
                "target_price": target,
                "last_source_event_id": observation.event_id,
                "last_ordering_timestamp": timestamp,
                "last_local_sequence": observation.local_sequence,
                "last_bid": observation.bid,
                "last_ask": observation.ask,
                "updated_at": observation.receive_timestamp.astimezone(UTC),
            }
        )
        return self._observe_active(observation, entry_event=True)

    def _observe_active(
        self,
        observation: ShadowBBOObservationV0,
        *,
        entry_event: bool = False,
    ) -> DirectionlessShadowEpisodeV0:
        timestamp = observation.ordering_timestamp.astimezone(UTC)
        direction = self._episode.direction
        entry = self._episode.entry_price
        assert direction is not None and entry is not None
        liquidation = observation.bid if direction == "LONG" else observation.ask
        favourable = liquidation - entry if direction == "LONG" else entry - liquidation
        adverse = entry - liquidation if direction == "LONG" else liquidation - entry
        mfe = max(self._episode.MFE_price, favourable, 0.0)
        mae = max(self._episode.MAE_price, adverse, 0.0)
        updates: dict[str, object] = {
            "MFE_price": mfe,
            "MAE_price": mae,
            "MFE_M": mfe / self._episode.movement_price,
            "MAE_M": mae / self._episode.movement_price,
            "MFE_R": mfe / (STOP_M_V0 * self._episode.movement_price),
            "MAE_R": mae / (STOP_M_V0 * self._episode.movement_price),
            "last_source_event_id": observation.event_id,
            "last_ordering_timestamp": timestamp,
            "last_local_sequence": observation.local_sequence,
            "last_bid": observation.bid,
            "last_ask": observation.ask,
            "updated_at": observation.receive_timestamp.astimezone(UTC),
        }
        if not self._episode.opposite_trigger_touched:
            opposite = (
                observation.bid <= self._episode.lower_trigger
                if direction == "LONG"
                else observation.ask >= self._episode.upper_trigger
            )
            if opposite:
                assert self._episode.entry_timestamp is not None
                elapsed = (timestamp - self._episode.entry_timestamp).total_seconds()
                updates.update(
                    {
                        "opposite_trigger_touched": True,
                        "time_to_opposite_trigger_seconds": elapsed,
                        "opposite_within_1m": elapsed <= 60.0,
                        "opposite_within_5m": elapsed <= 300.0,
                        "opposite_before_horizon": timestamp <= self._episode.horizon,
                    }
                )
        self._episode = self._episode.model_copy(update=updates)

        if timestamp >= self._episode.horizon:
            return self._exit(observation, liquidation, "SHADOW_TIME_EXIT")
        assert self._episode.stop_price is not None and self._episode.target_price is not None
        stop_touch = (
            liquidation <= self._episode.stop_price
            if direction == "LONG"
            else liquidation >= self._episode.stop_price
        )
        target_touch = (
            liquidation >= self._episode.target_price
            if direction == "LONG"
            else liquidation <= self._episode.target_price
        )
        if target_touch:
            return self._exit(observation, self._episode.target_price, "SHADOW_TARGET")
        if stop_touch:
            return self._exit(observation, self._episode.stop_price, "SHADOW_STOP")
        return self._episode

    def _exit(
        self,
        observation: ShadowBBOObservationV0,
        exit_price: float,
        reason: str,
    ) -> DirectionlessShadowEpisodeV0:
        assert self._episode.entry_price is not None and self._episode.direction is not None
        sign = 1.0 if self._episode.direction == "LONG" else -1.0
        gross_price = sign * (exit_price - self._episode.entry_price)
        gross_m = gross_price / self._episode.movement_price
        cost_price = self._episode.entry_price * ROUND_TRIP_COST_BPS_V0 / 10_000.0
        cost_m = cost_price / self._episode.movement_price
        net_m = gross_m - cost_m
        state = {
            "SHADOW_TARGET": DirectionlessShadowStateV0.TARGET_EXITED,
            "SHADOW_STOP": DirectionlessShadowStateV0.STOP_EXITED,
            "SHADOW_TIME_EXIT": DirectionlessShadowStateV0.TIME_EXITED,
        }[reason]
        self._episode = self._episode.model_copy(
            update={
                "state": state,
                "exit_timestamp": observation.ordering_timestamp.astimezone(UTC),
                "exit_price": exit_price,
                "exit_reason": reason,
                "exit_source_event_id": observation.event_id,
                "gross_price_return": gross_price,
                "gross_M": gross_m,
                "gross_R": gross_m / STOP_M_V0,
                "cost_price": cost_price,
                "cost_M": cost_m,
                "net_M": net_m,
                "net_R": net_m / STOP_M_V0,
                "updated_at": observation.receive_timestamp.astimezone(UTC),
            }
        )
        return self._episode

    def _resolve_gap_with(
        self, observation: ShadowBBOObservationV0
    ) -> DirectionlessShadowEpisodeV0:
        assert self._restored_gap is not None
        start, end, generation, gap_id, last_price = self._restored_gap
        self._restored_gap = None
        if self._episode.state in {
            DirectionlessShadowStateV0.LONG_ACTIVE,
            DirectionlessShadowStateV0.SHORT_ACTIVE,
        }:
            return self._terminal_unresolved(
                observation,
                state=DirectionlessShadowStateV0.ACTIVE_PATH_UNRESOLVED,
                reason="SHADOW_PATH_UNRESOLVED_DATA_GAP",
                gap=(start, end, generation, gap_id, last_price),
            )
        if self._episode.state is DirectionlessShadowStateV0.WAITING_FOR_ENTRY:
            long_touch = observation.ask >= self._episode.upper_trigger
            short_touch = observation.bid <= self._episode.lower_trigger
            if long_touch and short_touch:
                return self._terminal_unresolved(
                    observation,
                    state=DirectionlessShadowStateV0.ENTRY_UNRESOLVED,
                    reason="ENTRY_PATH_UNRESOLVED",
                    gap=(start, end, generation, gap_id, last_price),
                )
            # A single-side restored touch has a known side but unknown touch
            # time; V0 records it as a gap entry at the immutable boundary.
            result = self._observe_waiting(observation)
            if result.state in {
                DirectionlessShadowStateV0.LONG_ACTIVE,
                DirectionlessShadowStateV0.SHORT_ACTIVE,
            }:
                self._episode = result.model_copy(
                    update={
                        "entry_gap": True,
                        "gap_id": gap_id,
                        "gap_start": start,
                        "gap_end": end,
                        "gap_connection_generation": generation,
                        "last_valid_price_before_gap": last_price,
                        "first_restored_price": (
                            observation.ask if result.direction == "LONG" else observation.bid
                        ),
                    }
                )
            return self._episode
        return self._episode

    def _terminal_unresolved(
        self,
        observation: ShadowBBOObservationV0,
        *,
        state: DirectionlessShadowStateV0,
        reason: str,
        gap: tuple[datetime, datetime, int, str, float | None] | None = None,
    ) -> DirectionlessShadowEpisodeV0:
        updates: dict[str, object] = {
            "state": state,
            "exit_reason": reason,
            "exit_timestamp": observation.ordering_timestamp.astimezone(UTC),
            "exit_source_event_id": observation.event_id,
            "last_source_event_id": observation.event_id,
            "last_ordering_timestamp": observation.ordering_timestamp.astimezone(UTC),
            "last_local_sequence": observation.local_sequence,
            "last_bid": observation.bid,
            "last_ask": observation.ask,
            "updated_at": observation.receive_timestamp.astimezone(UTC),
        }
        if gap is not None:
            start, end, generation, gap_id, last_price = gap
            updates.update(
                {
                    "gap_id": gap_id,
                    "gap_start": start,
                    "gap_end": end,
                    "gap_connection_generation": generation,
                    "last_valid_price_before_gap": last_price,
                    "first_restored_price": (
                        observation.ask
                        if self._episode.direction == "SHORT"
                        else observation.bid
                        if self._episode.direction == "LONG"
                        else (observation.bid + observation.ask) / 2.0
                    ),
                }
            )
        self._episode = self._episode.model_copy(update=updates)
        return self._episode

    def _remember(self, observation: ShadowBBOObservationV0) -> None:
        self._episode = self._episode.model_copy(
            update={
                "last_source_event_id": observation.event_id,
                "last_ordering_timestamp": observation.ordering_timestamp.astimezone(UTC),
                "last_local_sequence": observation.local_sequence,
                "last_bid": observation.bid,
                "last_ask": observation.ask,
                "updated_at": observation.receive_timestamp.astimezone(UTC),
            }
        )

    def _liquidation_price_from_episode(self) -> float | None:
        # The exact pre-gap quote is referenced by last_source_event_id; a
        # numeric liquidation price is only retained once a position exists.
        if self._episode.last_bid is None or self._episode.last_ask is None:
            return None
        if self._episode.direction == "LONG":
            return self._episode.last_bid
        if self._episode.direction == "SHORT":
            return self._episode.last_ask
        return (self._episode.last_bid + self._episode.last_ask) / 2.0


__all__ = [
    "DirectionlessShadowEpisodeV0",
    "DirectionlessShadowStateV0",
    "FrozenDirectionlessShadowV0",
    "ShadowBBOObservationV0",
    "create_directionless_shadow_v0",
]
