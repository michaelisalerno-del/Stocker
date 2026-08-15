"""Runtime coordinator for the frozen directionless shadow state machines."""

from __future__ import annotations

from datetime import date, datetime

from stocker_prospective.directionless_shadow_repository_v0 import (
    DirectionlessShadowRepositoryV0,
)
from stocker_prospective.events import RawEvent, UnderlyingLevel1QuoteEvent
from stocker_prospective.m1c_directionless_shadow_v0 import (
    DirectionlessShadowEpisodeV0,
    FrozenDirectionlessShadowV0,
    ShadowBBOObservationV0,
    create_directionless_shadow_v0,
)


class DirectionlessShadowServiceV0:
    """Own exactly one local-only state machine per HARD episode/version."""

    def __init__(
        self,
        *,
        repository: DirectionlessShadowRepositoryV0,
        con_id_by_symbol: dict[str, int],
        code_hash: str,
        config_hash: str,
        m1c_artifact_hash: str,
        eligible_sessions: frozenset[date],
        activation_timestamp: datetime,
        contract_sha256: str,
    ) -> None:
        self.repository = repository
        self.con_id_by_symbol = dict(con_id_by_symbol)
        self.code_hash = code_hash
        self.config_hash = config_hash
        self.m1c_artifact_hash = m1c_artifact_hash
        if len(eligible_sessions) != 20:
            raise ValueError("directionless shadow V0 requires exactly 20 eligible sessions")
        self.eligible_sessions = eligible_sessions
        repository.record_run_contract(
            activation_timestamp=activation_timestamp,
            first_eligible_session=min(eligible_sessions),
            contract_sha256=contract_sha256,
            config_hash=config_hash,
            code_hash=code_hash,
            m1c_artifact_hash=m1c_artifact_hash,
        )
        self._machines: dict[str, FrozenDirectionlessShadowV0] = {}
        for episode in repository.load_active():
            machine = FrozenDirectionlessShadowV0(episode)
            # A process restart creates a real observation gap. The first
            # restored quote resolves waiting state conservatively and makes
            # an active path explicitly unresolved.
            machine.connection_lost(
                episode.updated_at,
                generation=episode.gap_connection_generation or 0,
                gap_id=f"restart-{episode.shadow_episode_id}-{episode.updated_at.isoformat()}",
            )
            machine.connection_restored(
                episode.updated_at,
                generation=episode.gap_connection_generation or 0,
                gap_id=f"restart-{episode.shadow_episode_id}-{episode.updated_at.isoformat()}",
            )
            self._machines[episode.m1c_episode_id] = machine

    def create_hard_episode(
        self,
        *,
        m1c_episode_id: str,
        symbol: str,
        session: date,
        t0: datetime,
        p0: float,
        movement_scale_fraction: float,
        probability: float,
        consumed_ratio: float | None,
        created_at: datetime,
    ) -> DirectionlessShadowEpisodeV0 | None:
        if session not in self.eligible_sessions:
            return None
        existing = self.repository.load(m1c_episode_id)
        if existing is not None:
            self._machines.setdefault(m1c_episode_id, FrozenDirectionlessShadowV0(existing))
            return existing
        movement_price = p0 * movement_scale_fraction
        episode = create_directionless_shadow_v0(
            m1c_episode_id=m1c_episode_id,
            symbol=symbol,
            con_id=self.con_id_by_symbol[symbol],
            session=session,
            t0=t0,
            p0=p0,
            movement_price=movement_price,
            movement_scale_fraction=movement_scale_fraction,
            probability=probability,
            consumed_ratio=consumed_ratio,
            code_hash=self.code_hash,
            config_hash=self.config_hash,
            m1c_artifact_hash=self.m1c_artifact_hash,
            created_at=created_at,
        )
        self.repository.save(episode)
        self._machines[m1c_episode_id] = FrozenDirectionlessShadowV0(episode)
        return episode

    def persist_raw_events(self, events: tuple[RawEvent, ...]) -> None:
        for event in events:
            if not isinstance(event, UnderlyingLevel1QuoteEvent):
                continue
            if (
                not event.quote_valid
                or event.bid is None
                or event.ask is None
                or event.symbol not in self.con_id_by_symbol
            ):
                continue
            observation = ShadowBBOObservationV0(
                event_id=event.event_id,
                ordering_timestamp=event.ordering_timestamp,
                receive_timestamp=event.received_timestamp_utc,
                local_sequence=event.source_sequence,
                connection_generation=event.connection_generation or 0,
                bid=event.bid,
                ask=event.ask,
            )
            for machine in tuple(self._machines.values()):
                before = machine.episode
                if before.symbol != event.symbol or before.terminal:
                    continue
                after = machine.observe(observation)
                if after != before:
                    self.repository.save(
                        after,
                        source_partition_reference=(
                            f"session_date={event.session.isoformat()}/symbol={event.symbol}/"
                            "event_type=underlying_level1_quote"
                        ),
                        connection_generation=event.connection_generation,
                        record_transition=(
                            after.state != before.state
                            or after.opposite_trigger_touched != before.opposite_trigger_touched
                        ),
                    )

    def advance_time(self, now: datetime) -> None:
        for machine in tuple(self._machines.values()):
            before = machine.episode
            after = machine.advance_time(now)
            if after != before:
                self.repository.save(after)

    def connection_lost(self, at: datetime, *, generation: int, gap_id: str) -> None:
        for machine in self._machines.values():
            if not machine.episode.terminal:
                machine.connection_lost(at, generation=generation, gap_id=gap_id)

    def connection_restored(self, at: datetime, *, generation: int, gap_id: str) -> None:
        for machine in self._machines.values():
            if not machine.episode.terminal:
                machine.connection_restored(at, generation=generation, gap_id=gap_id)


__all__ = ["DirectionlessShadowServiceV0"]
