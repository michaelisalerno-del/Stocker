"""Bounded IBKR option discovery and post-episode recording lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial
from typing import Any, Literal, cast

from stocker_prospective.database import EvidenceMetadata
from stocker_prospective.events import UnderlyingLevel1QuoteEvent
from stocker_prospective.ibkr import IBKRMarketDataAdapter
from stocker_prospective.live_subscriptions import QualifiedUnderlying
from stocker_prospective.option_ledger import (
    OptionContract,
    OptionContractPlan,
    build_contract_plan,
)
from stocker_prospective.option_recorder import (
    BoundedOptionRecorder,
    OptionEpisodeFinalization,
    ResolvedOptionContract,
)
from stocker_prospective.options import select_expiries
from stocker_prospective.recorder_v0 import RecorderCheckpointResult
from stocker_prospective.subscriptions import (
    SubscriptionBudgetManager,
    SubscriptionKind,
)

OptionContractFactory = Callable[
    [str, date, float, Literal["C", "P"], int, str, str],
    Any,
]
MetadataFactory = Callable[[datetime, tuple[datetime, ...]], EvidenceMetadata]
ReferenceQuoteProvider = Callable[
    [str, datetime],
    UnderlyingLevel1QuoteEvent | None,
]
QUIET_OPTION_STRIKE_STEPS = 4
QUIET_MAXIMUM_CONTRACTS_PER_OBSERVATION = 54
QUIET_WING_TARGET_FRACTIONS = (0.01, 0.03, 0.06, 0.10)


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


def _parse_expiry(value: object) -> date | None:
    raw = str(value)
    if len(raw) < 8:
        return None
    try:
        return datetime.strptime(raw[:8], "%Y%m%d").date()
    except ValueError:
        return None


@dataclass
class _PendingEpisode:
    episode_id: str
    symbol: str
    session: date
    entry_timestamp: datetime
    underlying: QualifiedUnderlying
    directional_actions: dict[str, str]
    quiet_state: bool = False
    recording_duration: timedelta = timedelta(minutes=30)
    strike_steps: int = 2
    maximum_contracts: int = 30
    started: bool = False
    finalised: bool = False


class BoundedOptionDiscoveryService:
    """Discover metadata once, qualify exact contracts, and freeze outcomes."""

    def __init__(
        self,
        *,
        adapter: IBKRMarketDataAdapter,
        option_recorder: BoundedOptionRecorder,
        budget: SubscriptionBudgetManager,
        underlying_contracts: Mapping[str, QualifiedUnderlying],
        contract_factory: OptionContractFactory,
        metadata_factory: MetadataFactory,
        reference_quote_provider: ReferenceQuoteProvider,
        strike_steps: int = 2,
        maximum_contracts_per_episode: int = 30,
        maximum_entry_delay: timedelta = timedelta(minutes=2),
        sensitivity_wait: timedelta = timedelta(seconds=5),
        common_strike_fallback_attempts: int = 3,
        heartbeat: Callable[[], object] | None = None,
    ) -> None:
        if strike_steps < 0 or maximum_contracts_per_episode < 0:
            raise ValueError("option discovery bounds must be nonnegative")
        if common_strike_fallback_attempts < 0:
            raise ValueError("common-strike fallback attempts must be nonnegative")
        self.adapter = adapter
        self.option_recorder = option_recorder
        self.budget = budget
        self.underlying_contracts = dict(underlying_contracts)
        self.contract_factory = contract_factory
        self.metadata_factory = metadata_factory
        self.reference_quote_provider = reference_quote_provider
        self.strike_steps = strike_steps
        self.maximum_contracts_per_episode = maximum_contracts_per_episode
        self.maximum_entry_delay = maximum_entry_delay
        self.sensitivity_wait = sensitivity_wait
        self.common_strike_fallback_attempts = common_strike_fallback_attempts
        self.heartbeat = heartbeat
        self._pending: dict[str, _PendingEpisode] = {}
        self._finalizations: dict[str, OptionEpisodeFinalization] = {}
        self._rejections: dict[str, str] = {}
        self._resolved_contracts: dict[str, ResolvedOptionContract] = {}

    def schedule(self, result: RecorderCheckpointResult) -> None:
        decision = result.episode_decision
        if not decision.fresh_episode or decision.episode_id is None:
            return
        underlying = self.underlying_contracts.get(decision.symbol)
        if underlying is None:
            self._rejections[decision.episode_id] = "underlying_contract_not_resolved"
            return
        actions: dict[str, str] = {
            model_id: classification.action
            for model_id, classification in result.directional_classifications.items()
        }
        self._pending.setdefault(
            decision.episode_id,
            _PendingEpisode(
                episode_id=decision.episode_id,
                symbol=decision.symbol,
                session=decision.session,
                entry_timestamp=decision.prospective_entry_timestamp,
                underlying=underlying,
                directional_actions=actions,
                quiet_state=False,
                recording_duration=timedelta(minutes=30),
                strike_steps=self.strike_steps,
                maximum_contracts=self.maximum_contracts_per_episode,
            ),
        )

    def schedule_quiet_state(self, result: RecorderCheckpointResult) -> None:
        """Schedule the same bounded 60-minute panel for all comparison cohorts."""

        decision = result.quiet_episode_decision
        observation_ids = tuple(
            value
            for value in (
                result.quiet_observation_id,
                result.neutral_control_id,
                result.high_tail_control_id,
            )
            if value is not None
        )
        underlying = self.underlying_contracts.get(decision.symbol)
        for observation_id in observation_ids:
            if underlying is None:
                self._rejections[observation_id] = "underlying_contract_not_resolved"
                continue
            existing = self._pending.get(observation_id)
            if existing is not None:
                if existing.started:
                    raise RuntimeError("quiet comparison upgrade occurred after option start")
                existing.quiet_state = True
                existing.recording_duration = timedelta(minutes=60)
                existing.strike_steps = QUIET_OPTION_STRIKE_STEPS
                existing.maximum_contracts = QUIET_MAXIMUM_CONTRACTS_PER_OBSERVATION
                continue
            self._pending[observation_id] = _PendingEpisode(
                episode_id=observation_id,
                symbol=decision.symbol,
                session=decision.session,
                entry_timestamp=decision.prospective_entry_timestamp,
                underlying=underlying,
                directional_actions={},
                quiet_state=True,
                recording_duration=timedelta(minutes=60),
                strike_steps=QUIET_OPTION_STRIKE_STEPS,
                maximum_contracts=QUIET_MAXIMUM_CONTRACTS_PER_OBSERVATION,
            )

    def poll(self, *, now: datetime) -> None:
        for episode in tuple(self._pending.values()):
            if episode.finalised:
                continue
            if not episode.started and now >= episode.entry_timestamp:
                quote = self.reference_quote_provider(
                    episode.symbol,
                    episode.entry_timestamp,
                )
                if quote is None:
                    if now - episode.entry_timestamp > self.maximum_entry_delay:
                        self._rejections[episode.episode_id] = (
                            "underlying_entry_reference_quote_unavailable"
                        )
                        episode.finalised = True
                    continue
                reference = (
                    (quote.bid + quote.ask) / 2.0
                    if quote.bid is not None and quote.ask is not None
                    else quote.last
                )
                if reference is None or reference <= 0.0:
                    continue
                metadata = self.metadata_factory(now, (quote.ordering_timestamp,))
                try:
                    plan = self._discover_plan(episode, reference)
                    self.option_recorder.start_episode(
                        metadata,
                        episode_id=episode.episode_id,
                        symbol=episode.symbol,
                        entry_timestamp=episode.entry_timestamp,
                        plan=plan,
                        resolver=partial(self._resolve, episode.symbol),
                        quiet_state=episode.quiet_state,
                        recording_duration=episode.recording_duration,
                    )
                except (RuntimeError, ValueError) as exc:
                    self._rejections[episode.episode_id] = str(exc)
                    episode.finalised = True
                    continue
                episode.started = True
            finish_at = episode.entry_timestamp + episode.recording_duration + self.sensitivity_wait
            if episode.started and now >= finish_at:
                metadata = self.metadata_factory(now, (finish_at,))
                self._finalizations[episode.episode_id] = self.option_recorder.finalise_episode(
                    metadata,
                    episode_id=episode.episode_id,
                    symbol=episode.symbol,
                    entry_timestamp=episode.entry_timestamp,
                    directional_actions=episode.directional_actions,
                )
                episode.finalised = True
        self.option_recorder.flush_pending(self.metadata_factory(now, (now,)))

    def mark_data_gap(self) -> None:
        self.option_recorder.mark_data_gap()

    def rebuild_after_data_loss(self, metadata: EvidenceMetadata) -> None:
        self.option_recorder.rebuild_after_data_loss(metadata)

    def shutdown(self, metadata: EvidenceMetadata) -> None:
        self.option_recorder.shutdown(metadata)

    def _discover_plan(
        self,
        episode: _PendingEpisode,
        underlying_reference: float,
    ) -> OptionContractPlan:
        result = self.adapter.request_option_chain_metadata(
            underlying_symbol=episode.symbol,
            exchange="",
            underlying_security_type="STK",
            underlying_contract_id=episode.underlying.con_id,
        )
        if self.heartbeat is not None:
            self.heartbeat()
        candidates = [
            item
            for item in result.items
            if int(_attribute(item, "underlying_contract_id", "underlyingConId") or 0)
            == episode.underlying.con_id
        ]
        if not candidates:
            raise ValueError("option_chain_metadata_unavailable")
        selected = min(
            candidates,
            key=lambda item: (
                0 if str(_attribute(item, "exchange") or "") == "SMART" else 1,
                str(_attribute(item, "exchange") or ""),
                str(_attribute(item, "trading_class", "tradingClass") or ""),
            ),
        )
        expiries = tuple(
            parsed
            for value in cast(list[object], _attribute(selected, "expirations") or [])
            for parsed in (_parse_expiry(value),)
            if parsed is not None
        )
        selections = select_expiries(episode.session, expiries)
        chosen_expiries = {bucket: selection.expiry for bucket, selection in selections.items()}
        strikes = tuple(
            float(cast(Any, value))
            for value in cast(list[object], _attribute(selected, "strikes") or [])
            if float(cast(Any, value)) > 0.0
        )
        active_options = int(
            cast(
                dict[str, int],
                self.budget.snapshot()["active"],
            ).get(SubscriptionKind.OPTION.value, 0)
        )
        available = max(
            0,
            self.budget.limits.get(SubscriptionKind.OPTION, 0) - active_options,
        )
        maximum = min(episode.maximum_contracts, available)
        exchange = str(_attribute(selected, "exchange") or "SMART")
        trading_class = str(_attribute(selected, "trading_class", "tradingClass") or episode.symbol)
        strikes_by_expiry_right: dict[tuple[date, str], tuple[float, ...]] = {}
        if maximum > 0:
            for expiry in chosen_expiries.values():
                if expiry is None:
                    continue
                valid_common = self._qualify_valid_common_strikes(
                    episode=episode,
                    expiry=expiry,
                    candidate_strikes=strikes,
                    underlying_reference=underlying_reference,
                    exchange=exchange,
                    trading_class=trading_class,
                )
                strikes_by_expiry_right[(expiry, "C")] = valid_common
                strikes_by_expiry_right[(expiry, "P")] = valid_common
        return build_contract_plan(
            underlying_con_id=episode.underlying.con_id,
            session_date=episode.session,
            underlying_reference=underlying_reference,
            expiries=chosen_expiries,
            strikes_by_expiry_right=strikes_by_expiry_right,
            strike_steps=episode.strike_steps,
            maximum_contracts=maximum,
            exchange=exchange,
            trading_class=trading_class,
        )

    def _qualify_valid_common_strikes(
        self,
        *,
        episode: _PendingEpisode,
        expiry: date,
        candidate_strikes: tuple[float, ...],
        underlying_reference: float,
        exchange: str,
        trading_class: str,
    ) -> tuple[float, ...]:
        """Boundedly find exact strikes that resolve for both call and put."""

        available = sorted(set(candidate_strikes))
        if not available:
            return ()
        nearest_order = sorted(
            available,
            key=lambda strike: (abs(strike - underlying_reference), strike),
        )
        atm: float | None = None
        search_limit = min(
            len(nearest_order),
            1 + self.common_strike_fallback_attempts,
        )
        for strike in nearest_order[:search_limit]:
            if self._qualify_pair(
                episode=episode,
                expiry=expiry,
                strike=strike,
                exchange=exchange,
                trading_class=trading_class,
            ):
                atm = strike
                break
        if atm is None:
            return ()

        if episode.quiet_state:
            return self._qualify_quiet_symmetric_strikes(
                episode=episode,
                expiry=expiry,
                available=tuple(available),
                atm=atm,
                underlying_reference=underlying_reference,
                exchange=exchange,
                trading_class=trading_class,
            )

        selected = [atm]
        side_limit = episode.strike_steps + self.common_strike_fallback_attempts
        for side_candidates in (
            sorted(strike for strike in available if strike > atm),
            sorted((strike for strike in available if strike < atm), reverse=True),
        ):
            found = 0
            for strike in side_candidates[:side_limit]:
                if self._qualify_pair(
                    episode=episode,
                    expiry=expiry,
                    strike=strike,
                    exchange=exchange,
                    trading_class=trading_class,
                ):
                    selected.append(strike)
                    found += 1
                    if found >= episode.strike_steps:
                        break
        return tuple(sorted(selected))

    def _qualify_quiet_symmetric_strikes(
        self,
        *,
        episode: _PendingEpisode,
        expiry: date,
        available: tuple[float, ...],
        atm: float,
        underlying_reference: float,
        exchange: str,
        trading_class: str,
    ) -> tuple[float, ...]:
        """Select frozen broad offsets while preserving exact symmetric fly wings."""

        upper = {round(strike - atm, 10): strike for strike in available if strike > atm}
        lower = {round(atm - strike, 10): strike for strike in available if strike < atm}
        remaining = set(upper).intersection(lower)
        ordered_distances: list[float] = []
        minimum_distance = QUIET_WING_TARGET_FRACTIONS[0] * underlying_reference
        eligible_wings = {
            distance for distance in remaining if distance + 1e-12 >= minimum_distance
        }
        if eligible_wings:
            nearest_valid_wing = min(eligible_wings)
            ordered_distances.append(nearest_valid_wing)
            remaining.remove(nearest_valid_wing)
        for fraction in QUIET_WING_TARGET_FRACTIONS[1:]:
            if not remaining:
                break
            target = fraction * underlying_reference
            eligible_remaining = {
                distance for distance in remaining if distance + 1e-12 >= minimum_distance
            }
            if not eligible_remaining:
                break
            distance = min(
                eligible_remaining,
                key=lambda value: (abs(value - target), value),
            )
            ordered_distances.append(distance)
            remaining.remove(distance)
        ordered_distances.extend(
            sorted(distance for distance in remaining if distance + 1e-12 >= minimum_distance)
        )
        attempt_limit = episode.strike_steps + self.common_strike_fallback_attempts
        selected = [atm]
        selected_pairs = 0
        for distance in ordered_distances[:attempt_limit]:
            upper_strike = upper[distance]
            lower_strike = lower[distance]
            upper_valid = self._qualify_pair(
                episode=episode,
                expiry=expiry,
                strike=upper_strike,
                exchange=exchange,
                trading_class=trading_class,
            )
            lower_valid = self._qualify_pair(
                episode=episode,
                expiry=expiry,
                strike=lower_strike,
                exchange=exchange,
                trading_class=trading_class,
            )
            if not upper_valid or not lower_valid:
                continue
            selected.extend((lower_strike, upper_strike))
            selected_pairs += 1
            if selected_pairs >= episode.strike_steps:
                break
        return tuple(sorted(selected))

    def _qualify_pair(
        self,
        *,
        episode: _PendingEpisode,
        expiry: date,
        strike: float,
        exchange: str,
        trading_class: str,
    ) -> bool:
        pair: list[tuple[OptionContract, ResolvedOptionContract]] = []
        for right in ("C", "P"):
            unresolved = OptionContract(
                underlying_con_id=episode.underlying.con_id,
                con_id=None,
                expiry=expiry,
                dte=(expiry - episode.session).days,
                dte_bucket=next(
                    bucket
                    for bucket, selection in select_expiries(
                        episode.session,
                        (expiry,),
                    ).items()
                    if selection.expiry == expiry
                ),
                strike=strike,
                right=right,
                multiplier=100,
                exchange=exchange,
                trading_class=trading_class,
            )
            resolved = self._resolve_uncached(episode.symbol, unresolved)
            if resolved is None:
                return False
            pair.append((unresolved, resolved))
        for unresolved, resolved in pair:
            self._resolved_contracts[unresolved.con_id_key] = resolved
        return True

    def _resolve(
        self,
        symbol: str,
        contract: OptionContract,
    ) -> ResolvedOptionContract | None:
        cached = self._resolved_contracts.get(contract.con_id_key)
        if cached is not None:
            return cached
        return self._resolve_uncached(symbol, contract)

    def _resolve_uncached(
        self,
        symbol: str,
        contract: OptionContract,
    ) -> ResolvedOptionContract | None:
        upstream = self.contract_factory(
            symbol,
            contract.expiry,
            contract.strike,
            contract.right,
            contract.multiplier,
            contract.exchange,
            contract.trading_class,
        )
        result = self.adapter.qualify_exact_contract(upstream)
        if self.heartbeat is not None:
            self.heartbeat()
        matching = []
        for detail in result.items:
            qualified = _attribute(detail, "contract") or detail
            expiry = _parse_expiry(
                _attribute(
                    qualified,
                    "lastTradeDateOrContractMonth",
                    "expiry",
                )
            )
            if (
                str(_attribute(qualified, "symbol") or "") == symbol
                and str(_attribute(qualified, "secType", "sec_type") or "") == "OPT"
                and expiry == contract.expiry
                and float(_attribute(qualified, "strike") or 0.0) == contract.strike
                and str(_attribute(qualified, "right") or "") == contract.right
                and int(_attribute(qualified, "conId", "con_id") or 0) > 0
            ):
                matching.append(qualified)
        if len(matching) != 1:
            return None
        qualified = matching[0]
        con_id = int(_attribute(qualified, "conId", "con_id"))
        return ResolvedOptionContract(
            contract=OptionContract(
                **{
                    **contract.__dict__,
                    "con_id": con_id,
                    "exchange": str(_attribute(qualified, "exchange") or contract.exchange),
                    "trading_class": str(
                        _attribute(
                            qualified,
                            "tradingClass",
                            "trading_class",
                        )
                        or contract.trading_class
                    ),
                }
            ),
            upstream_contract=qualified,
        )

    @property
    def rejections(self) -> dict[str, str]:
        return dict(self._rejections)

    @property
    def finalizations(self) -> dict[str, OptionEpisodeFinalization]:
        return dict(self._finalizations)

    @property
    def live_option_quote_seen(self) -> bool:
        return self.option_recorder.live_option_quote_seen

    @property
    def option_computation_seen(self) -> bool:
        return self.option_recorder.option_computation_seen


__all__ = ["BoundedOptionDiscoveryService"]
