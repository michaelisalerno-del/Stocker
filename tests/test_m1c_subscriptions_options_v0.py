from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.event_ingest import IBKRCallbackNormalizer
from stocker_prospective.events import OptionQuoteEvent
from stocker_prospective.live_subscriptions import (
    LiveSubscriptionController,
    QualifiedUnderlying,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.option_discovery import (
    BoundedOptionDiscoveryService,
    _PendingEpisode,
)
from stocker_prospective.option_ledger import (
    OptionContract,
    OptionContractPlan,
    build_contract_plan,
    build_shadow_outcomes,
)
from stocker_prospective.option_recorder import (
    BoundedOptionRecorder,
    ResolvedOptionContract,
)
from stocker_prospective.options import DteBucket
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.subscriptions import (
    PromotionScheduler,
    SubscriptionBudgetManager,
    SubscriptionKind,
    SubscriptionPriority,
)

ENTRY = datetime(2026, 7, 24, 14, 5, tzinfo=UTC)


class SubscriptionAdapter:
    def __init__(self) -> None:
        self.next_request_id = 1
        self.cancelled: list[tuple[str, int]] = []

    def _next(self) -> int:
        value = self.next_request_id
        self.next_request_id += 1
        return value

    def request_market_data(self, *_args: Any, **_kwargs: Any) -> int:
        return self._next()

    def request_historical_five_minute_updates(self, *_args: Any, **_kwargs: Any) -> int:
        return self._next()

    def request_tick_by_tick(self, *_args: Any, **_kwargs: Any) -> int:
        return self._next()

    def request_market_depth(self, *_args: Any, **_kwargs: Any) -> int:
        return self._next()

    def cancel_market_data(self, request_id: int, **_kwargs: Any) -> None:
        self.cancelled.append(("level1", request_id))

    def cancel_historical_updates(self, request_id: int, **_kwargs: Any) -> None:
        self.cancelled.append(("bar", request_id))

    def cancel_tick_by_tick(self, request_id: int, **_kwargs: Any) -> None:
        self.cancelled.append(("tick", request_id))

    def cancel_market_depth(self, request_id: int, **_kwargs: Any) -> None:
        self.cancelled.append(("depth", request_id))


class FailingSecondOptionAdapter(SubscriptionAdapter):
    def request_market_data(self, *_args: Any, **_kwargs: Any) -> int:
        if self.next_request_id == 2:
            raise RuntimeError("synthetic option subscription failure")
        return super().request_market_data(*_args, **_kwargs)


class OptionRepositoryStub:
    def __init__(self) -> None:
        self.next_contract_id = 1
        self.shadow_validity: list[tuple[str, bool]] = []
        self.structure_validity: list[tuple[str, bool]] = []

    def record_option_contract(self, *_args: Any, **_kwargs: Any) -> int:
        value = self.next_contract_id
        self.next_contract_id += 1
        return value

    def record_subscription(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def update_option_quote_projection(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_partition(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_shadow_outcome(self, *_args: Any, **kwargs: Any) -> None:
        self.shadow_validity.append((str(kwargs["archetype"]), bool(kwargs["valid"])))

    def record_shadow_structure(self, *_args: Any, **kwargs: Any) -> None:
        self.structure_validity.append((str(kwargs["structure_type"]), bool(kwargs["valid"])))


def controller_fixture(
    tmp_path: Path,
    *,
    tick_limit: int,
    bar_limit: int = 1,
    depth_limit: int = 1,
) -> tuple[
    LiveSubscriptionController,
    SubscriptionBudgetManager,
    SubscriptionAdapter,
    EvidenceMetadata,
]:
    database = ProspectiveRepository(tmp_path / "recorder.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="subscription-test",
        prospective_start_utc=ENTRY,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[ENTRY.isoformat()],
        recorded_at_utc=ENTRY,
    )
    database.create_run(metadata)
    budget = SubscriptionBudgetManager(
        limits={
            SubscriptionKind.LEVEL1: 1,
            SubscriptionKind.BAR: bar_limit,
            SubscriptionKind.TICK_BY_TICK: tick_limit,
            SubscriptionKind.DEPTH: depth_limit,
            SubscriptionKind.OPTION: 0,
        },
        request_rate_limit=100,
    )
    upstream = SubscriptionAdapter()
    controller = LiveSubscriptionController(
        adapter=upstream,  # type: ignore[arg-type]
        budget=budget,
        normalizer=IBKRCallbackNormalizer(prospective_collection_start=ENTRY),
        repository=FrozenRecorderRepository(database),
        depth_rows=5,
        enable_depth=True,
    )
    return controller, budget, upstream, metadata


def test_always_on_bar_capacity_and_active_tick_pair_fail_closed(tmp_path: Path) -> None:
    contract = QualifiedUnderlying(
        symbol="AAL",
        con_id=1,
        upstream_contract=object(),
        exchange="SMART",
    )
    blocked, _, _, metadata = controller_fixture(tmp_path / "blocked", tick_limit=2, bar_limit=0)
    with pytest.raises(RuntimeError, match="five_minute_bar_capacity"):
        blocked.start_always_on(metadata, (contract,))

    controller, budget, upstream, metadata = controller_fixture(
        tmp_path / "paired",
        tick_limit=1,
    )
    controller.start_always_on(metadata, (contract,))
    controller.promote_active_episode(metadata, symbol="AAL", episode_id="episode-1")

    assert budget.get("underlying:AAL:tbt:bidask") is None
    assert budget.get("underlying:AAL:tbt:last") is None
    assert any(kind == "tick" for kind, _ in upstream.cancelled)


def test_depth_reset_resubscription_replaces_the_exact_request(tmp_path: Path) -> None:
    controller, budget, upstream, metadata = controller_fixture(tmp_path, tick_limit=2)
    contract = QualifiedUnderlying(
        symbol="AAL",
        con_id=1,
        upstream_contract=object(),
        exchange="SMART",
    )
    controller.start_always_on(metadata, (contract,))
    controller.promote_active_episode(metadata, symbol="AAL", episode_id="episode-1")
    first = budget.get("underlying:AAL:depth")
    assert first is not None
    first_request_id = first.request_id

    assert controller.resubscribe_depth(metadata, symbol="AAL") is True
    replacement = budget.get("underlying:AAL:depth")
    assert replacement is not None
    assert replacement.request_id != first_request_id
    assert ("depth", first_request_id) in upstream.cancelled


def test_subscription_manager_protects_universe_level1_and_evicts_deterministically() -> None:
    manager = SubscriptionBudgetManager(
        limits={
            SubscriptionKind.LEVEL1: 25,
            SubscriptionKind.TICK_BY_TICK: 2,
            SubscriptionKind.DEPTH: 1,
            SubscriptionKind.OPTION: 4,
        },
        request_rate_limit=100,
    )
    for symbol in ("AAL", "AAOI"):
        manager.allocate(
            key=f"l1:{symbol}",
            kind=SubscriptionKind.LEVEL1,
            symbol=symbol,
            con_id=1,
            request_id=len(manager.records) + 1,
            priority=SubscriptionPriority.UNIVERSE_LEVEL1,
            protected=True,
            now_monotonic=float(len(manager.records)),
        )
    manager.allocate(
        key="tbt:AAOI",
        kind=SubscriptionKind.TICK_BY_TICK,
        symbol="AAOI",
        con_id=2,
        request_id=20,
        priority=SubscriptionPriority.ARMED_CANDIDATE,
        now_monotonic=10.0,
    )
    manager.allocate(
        key="tbt:AAL",
        kind=SubscriptionKind.TICK_BY_TICK,
        symbol="AAL",
        con_id=1,
        request_id=21,
        priority=SubscriptionPriority.ARMED_CANDIDATE,
        now_monotonic=11.0,
    )
    decision = manager.allocate(
        key="tbt:MSTR",
        kind=SubscriptionKind.TICK_BY_TICK,
        symbol="MSTR",
        con_id=3,
        request_id=22,
        priority=SubscriptionPriority.ACTIVE_EPISODE,
        owner_episode="episode-1",
        now_monotonic=12.0,
    )

    assert decision.accepted is True
    assert decision.evicted_key == "tbt:AAOI"
    assert all(manager.get(f"l1:{symbol}") is not None for symbol in ("AAL", "AAOI"))
    assert manager.get("tbt:MSTR").owner_episode == "episode-1"  # type: ignore[union-attr]


def test_checkpoint_promotion_is_ranked_with_symbol_tie_breaking() -> None:
    scheduler = PromotionScheduler(max_tick_by_tick=2, max_depth=1)
    decisions = scheduler.rank_checkpoint(
        checkpoint_time=ENTRY,
        probabilities={"MSTR": 0.7, "AAL": 0.7, "AAOI": 0.6},
        eligible_symbols={"MSTR", "AAL", "AAOI"},
        active_episode_symbols={"AAOI"},
    )

    tick = [item for item in decisions if item.subscription_type == "tick_by_tick"]
    depth = [item for item in decisions if item.subscription_type == "depth"]
    assert [item.symbol for item in tick] == ["AAL", "MSTR"]
    assert [item.symbol for item in depth] == ["AAL"]
    assert all(item.reason == "checkpoint_ranked_armed_candidate" for item in decisions)


def test_checkpoint_promotion_decision_persists_probability_rank_and_capacity(
    tmp_path: Path,
) -> None:
    controller, _, _, metadata = controller_fixture(tmp_path, tick_limit=2)
    contract = QualifiedUnderlying(
        symbol="AAL",
        con_id=1,
        upstream_contract=object(),
        exchange="SMART",
    )
    controller.start_always_on(metadata, (contract,))
    decisions = PromotionScheduler(max_tick_by_tick=1, max_depth=0).rank_checkpoint(
        checkpoint_time=ENTRY,
        probabilities={"AAL": 0.61},
        eligible_symbols={"AAL"},
        active_episode_symbols=set(),
    )

    controller.apply_checkpoint_promotions(metadata, decisions)

    database = ProspectiveRepository(tmp_path / "recorder.sqlite3")
    with database._connect() as connection:
        row = connection.execute("SELECT * FROM promotion_decision_v0").fetchone()
    assert row is not None
    assert row["symbol"] == "AAL"
    assert row["m1c_probability"] == 0.61
    assert row["rank"] == 1
    assert row["capacity_available"] == 1
    assert row["reason"] == "checkpoint_ranked_armed_candidate"


def test_option_plan_honours_buckets_common_strikes_and_global_capacity() -> None:
    expiries = {
        DteBucket.ZERO_DTE: date(2026, 7, 24),
        DteBucket.ONE_DTE: date(2026, 7, 25),
        DteBucket.THREE_TO_FIVE_DTE: date(2026, 7, 28),
    }
    plan = build_contract_plan(
        underlying_con_id=123,
        session_date=date(2026, 7, 24),
        underlying_reference=102.0,
        expiries=expiries,
        strikes_by_expiry_right={
            (expiry, right): (95.0, 100.0, 105.0, 110.0)
            for expiry in expiries.values()
            for right in ("C", "P")
        },
        strike_steps=2,
        maximum_contracts=12,
        exchange="SMART",
        trading_class="AAL",
    )

    assert len(plan.contracts) == 12
    assert len({contract.con_id_key for contract in plan.contracts}) == 12
    for bucket in DteBucket:
        bucket_contracts = [item for item in plan.contracts if item.dte_bucket is bucket]
        assert {(item.strike, item.right) for item in bucket_contracts[:2]} == {
            (100.0, "C"),
            (100.0, "P"),
        }
    assert plan.capacity_reduced is True


def test_option_discovery_replaces_invalid_near_strikes_with_valid_common_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = BoundedOptionDiscoveryService(
        adapter=cast(Any, object()),
        option_recorder=cast(Any, object()),
        budget=SubscriptionBudgetManager(
            limits={kind: 0 for kind in SubscriptionKind},
            request_rate_limit=100,
        ),
        underlying_contracts={},
        contract_factory=lambda *_args: object(),
        metadata_factory=lambda _observed, _sources: cast(Any, None),
        reference_quote_provider=lambda _symbol, _timestamp: None,
        strike_steps=1,
        common_strike_fallback_attempts=2,
    )
    monkeypatch.setattr(
        service,
        "_qualify_pair",
        lambda **kwargs: kwargs["strike"] in {95.0, 105.0, 110.0},
    )
    episode = _PendingEpisode(
        episode_id="episode-common",
        symbol="AAL",
        session=date(2026, 7, 24),
        entry_timestamp=ENTRY,
        underlying=QualifiedUnderlying(
            symbol="AAL",
            con_id=123,
            upstream_contract=object(),
            exchange="SMART",
        ),
        directional_actions={},
    )

    selected = service._qualify_valid_common_strikes(
        episode=episode,
        expiry=date(2026, 7, 24),
        candidate_strikes=(90.0, 95.0, 100.0, 105.0, 110.0, 115.0),
        underlying_reference=101.0,
        exchange="SMART",
        trading_class="AAL",
    )

    assert selected == (95.0, 105.0, 110.0)


def option_quote(
    contract: OptionContract,
    seconds: int,
    *,
    bid: float,
    ask: float,
    underlying: float,
) -> OptionQuoteEvent:
    observed = ENTRY + timedelta(seconds=seconds)
    return OptionQuoteEvent(
        event_id=f"{contract.con_id_key}-{seconds}",
        received_timestamp_utc=observed,
        received_monotonic_ns=seconds + 1_000,
        provider_timestamp_utc=observed,
        source_sequence=seconds + 1_000,
        session=date(2026, 7, 24),
        episode_id="episode-1",
        symbol="AAL",
        con_id=contract.con_id,
        request_id=50,
        expiry=contract.expiry,
        dte=contract.dte,
        dte_bucket=contract.dte_bucket,
        strike=contract.strike,
        right=contract.right,
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
        bid=bid,
        ask=ask,
        bid_size=10.0,
        ask_size=12.0,
        last=(bid + ask) / 2,
        last_size=1.0,
        market_data_type=MarketDataType.LIVE,
        underlying_reference_price=underlying,
        implied_volatility=0.4,
        delta=0.5,
        gamma=0.05,
        theta=-0.02,
        vega=0.08,
    )


def test_shadow_ledger_uses_first_ask_after_entry_and_last_bid_before_horizon() -> None:
    contract = OptionContract(
        underlying_con_id=123,
        con_id=456,
        expiry=date(2026, 7, 28),
        dte=4,
        dte_bucket=DteBucket.THREE_TO_FIVE_DTE,
        strike=100.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )
    quotes = (
        option_quote(contract, -1, bid=1.0, ask=1.2, underlying=99.9),
        option_quote(contract, 1, bid=1.1, ask=1.3, underlying=100.0),
        option_quote(contract, 299, bid=1.6, ask=1.8, underlying=101.0),
        option_quote(contract, 301, bid=1.7, ask=1.9, underlying=101.1),
    )
    outcomes = build_shadow_outcomes(
        episode_id="episode-1",
        symbol="AAL",
        entry_timestamp=ENTRY,
        contracts=(contract,),
        quotes=quotes,
        horizons=(timedelta(minutes=5),),
        maximum_quote_age=timedelta(seconds=5),
    )

    outcome = outcomes[0]
    assert outcome.entry_ask == 1.3
    assert outcome.exit_bid == 1.6
    assert outcome.first_bid_after_horizon == 1.7
    assert outcome.ask_to_bid_return == (1.6 - 1.3) / 1.3
    assert outcome.dollar_pnl_per_contract == 30.0
    assert outcome.mid_to_mid_return is not None
    assert (
        outcome.primary_return_definition
        == "first_valid_ask_after_entry_to_last_valid_bid_at_or_before_horizon"
    )


def test_option_finalisation_flushes_raw_quotes_and_invalidates_every_gap_spanning_structure(
    tmp_path: Path,
) -> None:
    metadata = EvidenceMetadata(
        run_id="option-gap-test",
        prospective_start_utc=ENTRY,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[ENTRY.isoformat()],
        recorded_at_utc=ENTRY + timedelta(minutes=31),
    )
    call = OptionContract(
        underlying_con_id=123,
        con_id=456,
        expiry=date(2026, 7, 28),
        dte=4,
        dte_bucket=DteBucket.THREE_TO_FIVE_DTE,
        strike=100.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )
    put = OptionContract(
        **{
            **call.__dict__,
            "con_id": 457,
            "right": "P",
        }
    )
    plan = OptionContractPlan(
        contracts=(call, put),
        requested_contract_count=2,
        maximum_contracts=2,
        capacity_reduced=False,
        missing_buckets=(),
    )
    repository = OptionRepositoryStub()
    budget = SubscriptionBudgetManager(
        limits={kind: (2 if kind is SubscriptionKind.OPTION else 0) for kind in SubscriptionKind},
        request_rate_limit=100,
    )
    recorder = BoundedOptionRecorder(
        adapter=SubscriptionAdapter(),  # type: ignore[arg-type]
        subscriptions=budget,
        repository=repository,  # type: ignore[arg-type]
        raw_store=PartitionedEventStore(
            root=tmp_path / "raw",
            prospective_collection_start=ENTRY,
            recorder_version="test",
            contract_version="frozen-m1c-microstructure-recorder-v0",
        ),
        maximum_quote_age=timedelta(seconds=2),
    )
    recorder.start_episode(
        metadata,
        episode_id="episode-1",
        symbol="AAL",
        entry_timestamp=ENTRY,
        plan=plan,
        resolver=lambda contract: ResolvedOptionContract(
            contract=contract,
            upstream_contract=object(),
        ),
    )
    for contract in (call, put):
        for seconds in (1, 299, 599, 899, 1799):
            recorder.record_quote(
                metadata,
                option_quote(
                    contract,
                    seconds,
                    bid=1.0 + seconds / 1_000,
                    ask=1.1 + seconds / 1_000,
                    underlying=100.0 + seconds / 10_000,
                ),
            )
    recorder.mark_data_gap()
    recorder.finalise_episode(
        metadata,
        episode_id="episode-1",
        symbol="AAL",
        entry_timestamp=ENTRY,
        directional_actions={"A1": "CALL", "C1": "PUT", "R1": "ABSTAIN"},
    )

    assert len(list((tmp_path / "raw").rglob("*.parquet"))) == 1
    partition_metadata = json.loads(
        next((tmp_path / "raw").rglob("*.metadata.json")).read_text(encoding="utf-8")
    )
    assert partition_metadata["row_count"] == 10
    assert repository.shadow_validity
    assert all(valid is False for _, valid in repository.shadow_validity)
    assert repository.structure_validity
    assert all(valid is False for _, valid in repository.structure_validity)


def test_option_start_rolls_back_every_earlier_stream_after_late_failure(
    tmp_path: Path,
) -> None:
    call = OptionContract(
        underlying_con_id=123,
        expiry=date(2026, 7, 24),
        dte=0,
        dte_bucket=DteBucket.ZERO_DTE,
        strike=100.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
        con_id=1,
    )
    put = OptionContract(
        underlying_con_id=123,
        expiry=date(2026, 7, 24),
        dte=0,
        dte_bucket=DteBucket.ZERO_DTE,
        strike=100.0,
        right="P",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
        con_id=2,
    )
    plan = OptionContractPlan(
        contracts=(call, put),
        requested_contract_count=2,
        maximum_contracts=2,
        capacity_reduced=False,
        missing_buckets=(),
    )
    repository = OptionRepositoryStub()
    budget = SubscriptionBudgetManager(
        limits={kind: (2 if kind is SubscriptionKind.OPTION else 0) for kind in SubscriptionKind},
        request_rate_limit=100,
    )
    upstream = FailingSecondOptionAdapter()
    metadata = EvidenceMetadata(
        run_id="option-rollback",
        prospective_start_utc=ENTRY,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[ENTRY.isoformat()],
        recorded_at_utc=ENTRY,
    )
    recorder = BoundedOptionRecorder(
        adapter=upstream,  # type: ignore[arg-type]
        subscriptions=budget,
        repository=repository,  # type: ignore[arg-type]
        raw_store=PartitionedEventStore(
            root=tmp_path / "raw",
            prospective_collection_start=ENTRY,
            recorder_version="test",
            contract_version="frozen-m1c-microstructure-recorder-v0",
        ),
        maximum_quote_age=timedelta(seconds=2),
    )

    with pytest.raises(RuntimeError, match="synthetic option subscription failure"):
        recorder.start_episode(
            metadata,
            episode_id="episode-rollback",
            symbol="AAL",
            entry_timestamp=ENTRY,
            plan=plan,
            resolver=lambda contract: ResolvedOptionContract(
                contract=contract,
                upstream_contract=object(),
            ),
        )

    assert recorder.active_episode_ids == frozenset()
    assert budget.snapshot()["active"][SubscriptionKind.OPTION.value] == 0
    assert upstream.cancelled == [("level1", 1)]
