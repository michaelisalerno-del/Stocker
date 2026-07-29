from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningReversalCapacityCoordinatorV1,
    OptionalOpeningReversalFeedV1,
    OptionContractCandidateV1,
    OptionTopOfBookV1,
    PrimaryOptionBidAskOutcomeV1,
    build_capacity_degradation_events_v1,
    build_primary_option_bid_ask_outcome_v1,
    select_primary_option_pair_v1,
)
from stocker_prospective.option_budget import (
    BudgetAwareEpisodeStateMachine,
    EpisodeKind,
    EpisodeState,
    OptionEpisodeTask,
    OptionSubscriptionIntent,
)
from stocker_prospective.options import DteBucket
from stocker_prospective.subscriptions import (
    SubscriptionBudgetManager,
    SubscriptionClass,
    SubscriptionKind,
    SubscriptionPriority,
)

NOW = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)


def _hash_without(payload: dict[str, object], field: str) -> str:
    value = {key: item for key, item in payload.items() if key != field}
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _budget(*, total: int = 40) -> SubscriptionBudgetManager:
    return SubscriptionBudgetManager(
        limits={
            SubscriptionKind.BAR: 30,
            SubscriptionKind.LEVEL1: 4,
            SubscriptionKind.OPTION: 12,
            SubscriptionKind.TICK_BY_TICK: 2,
            SubscriptionKind.DEPTH: 0,
            SubscriptionKind.MARKET_PROXY: 0,
        },
        request_rate_limit=100,
        total_line_limit=total,
        future_trading_reserve_lines=12,
    )


def _allocate_mandatory(
    budget: SubscriptionBudgetManager,
    *,
    count: int,
) -> None:
    for index in range(count):
        decision = budget.allocate(
            key=f"BAR|{index + 1}|5m|RTH",
            kind=SubscriptionKind.BAR,
            symbol=f"S{index:02d}",
            con_id=index + 1,
            request_id=index + 1,
            priority=SubscriptionPriority.FROZEN_UNIVERSE_SIGNAL,
            owner_id="core_m1c_universe",
            subscription_class=SubscriptionClass.FROZEN_UNIVERSE_SIGNAL,
            protected=True,
            now_monotonic=float(index),
            now_utc=NOW,
        )
        assert decision.accepted


def _intent(
    con_id: int,
    *,
    role: str,
    required: bool,
    drop_order: int | None = None,
) -> OptionSubscriptionIntent:
    return OptionSubscriptionIntent(
        key=f"OPTION_LEVEL1|{con_id}",
        con_id=con_id,
        role=role,
        subscription_class=(
            SubscriptionClass.ACTIVE_EPISODE
            if required
            else SubscriptionClass.OPTIONAL_RESEARCH
        ),
        required=required,
        dte_bucket=DteBucket.ONE_DTE,
        drop_order=drop_order,
    )


def _task(
    subscriptions: tuple[OptionSubscriptionIntent, ...],
) -> OptionEpisodeTask:
    return OptionEpisodeTask(
        episode_id="episode-1",
        symbol="AAL",
        kind=EpisodeKind.OPENING_REVERSAL,
        probability=0.75,
        triggered_at_utc=NOW,
        useful_until_utc=NOW + timedelta(minutes=30),
        requested_subscriptions=subscriptions,
    )


def test_reserved_twelve_lines_cannot_be_used_by_optional_research() -> None:
    budget = _budget(total=36)
    _allocate_mandatory(budget, count=22)
    machine = BudgetAwareEpisodeStateMachine(
        budget=budget,
        max_option_lines_per_episode=8,
    )
    record = machine.submit(
        _task(
            (
                _intent(1001, role="primary_call", required=True),
                _intent(1002, role="primary_put", required=True),
                _intent(
                    1003,
                    role="0dte_call",
                    required=False,
                    drop_order=4,
                ),
                _intent(
                    1004,
                    role="0dte_put",
                    required=False,
                    drop_order=4,
                ),
            )
        ),
        now=NOW,
    )

    assert record.state is EpisodeState.DEGRADED
    assert record.required_subscriptions == (
        "OPTION_LEVEL1|1001",
        "OPTION_LEVEL1|1002",
    )
    assert record.approved_subscriptions == (
        "OPTION_LEVEL1|1001",
        "OPTION_LEVEL1|1002",
    )
    assert record.denied_subscriptions == (
        "OPTION_LEVEL1|1003",
        "OPTION_LEVEL1|1004",
    )
    event = build_capacity_degradation_events_v1(record)
    assert len(event) == 1
    assert event[0].feed is OptionalOpeningReversalFeedV1.ZERO_DTE_COMPARISON
    assert event[0].reason == "optional_feed_not_started_capacity_reserved"
    assert event[0].subscription_ids == (
        "OPTION_LEVEL1|1003",
        "OPTION_LEVEL1|1004",
    )
    assert event[0].primary_option_evidence_remains_complete
    assert budget.snapshot()["reserved_future_trading_lines"] == 12
    assert budget.snapshot()["available_research_lines"] == 0


def test_primary_pair_capacity_failure_preserves_underlying_direction() -> None:
    budget = _budget(total=35)
    _allocate_mandatory(budget, count=22)
    machine = BudgetAwareEpisodeStateMachine(
        budget=budget,
        max_option_lines_per_episode=8,
    )
    record = machine.submit(
        _task(
            (
                _intent(2001, role="primary_call", required=True),
                _intent(2002, role="primary_put", required=True),
            )
        ),
        now=NOW,
    )

    assert record.state is EpisodeState.EPISODE_QUEUED
    assert record.required_subscriptions == (
        "OPTION_LEVEL1|2001",
        "OPTION_LEVEL1|2002",
    )
    assert record.degradation_reason == "primary_option_legs_incomplete"
    assert not any(
        record.active and record.kind is SubscriptionKind.OPTION
        for record in budget.records.values()
    )
    assert all(
        record.active
        for record in budget.records.values()
        if record.kind is SubscriptionKind.BAR
    )
    event = build_capacity_degradation_events_v1(record)
    assert len(event) == 1
    assert event[0].feed is None
    assert event[0].reason == "option_economics_blocked_capacity"
    assert event[0].subscription_ids == (
        "OPTION_LEVEL1|2001",
        "OPTION_LEVEL1|2002",
    )
    assert not event[0].primary_option_evidence_remains_complete


def test_deterministic_degradation_drops_optional_feeds_before_primary_pair() -> None:
    budget = _budget(total=60)
    _allocate_mandatory(budget, count=22)
    machine = BudgetAwareEpisodeStateMachine(
        budget=budget,
        max_option_lines_per_episode=12,
    )
    feeds = (
        OptionalOpeningReversalFeedV1.NEUTRAL_CONTROL,
        OptionalOpeningReversalFeedV1.ADDITIONAL_STRIKE,
        OptionalOpeningReversalFeedV1.THREE_TO_FIVE_DTE_COMPARISON,
        OptionalOpeningReversalFeedV1.ZERO_DTE_COMPARISON,
        OptionalOpeningReversalFeedV1.TICK_BY_TICK,
        OptionalOpeningReversalFeedV1.ADDITIONAL_UNDERLYING_DIAGNOSTIC,
    )
    optional = tuple(
        _intent(
            3000 + ordinal,
            role=feed.value,
            required=False,
            drop_order=feed.drop_order,
        )
        for ordinal, feed in enumerate(feeds, start=1)
    )
    record = machine.submit(
        _task(
            (
                _intent(2001, role="primary_call", required=True),
                _intent(2002, role="primary_put", required=True),
                *optional,
            )
        ),
        now=NOW,
    )
    assert record.state is EpisodeState.COMPARISON_LEGS_STREAMING
    dropped = []
    previous_denied: set[str] = set()
    for _ in feeds:
        degraded = machine.degrade_optional(
            "episode-1",
            now=NOW,
            reason="capacity_tightened",
        )
        new_denied = set(degraded.denied_subscriptions) - previous_denied
        assert len(new_denied) == 1
        event = build_capacity_degradation_events_v1(degraded)
        assert len(event) == 1
        assert set(event[0].subscription_ids) == new_denied
        dropped.append(next(iter(new_denied)))
        previous_denied = set(degraded.denied_subscriptions)

    assert tuple(dropped) == tuple(
        f"OPTION_LEVEL1|{3000 + ordinal}"
        for ordinal in range(1, 7)
    )
    assert (
        "OPTION_LEVEL1|2001"
        in machine.record("episode-1").approved_subscriptions
    )
    assert (
        "OPTION_LEVEL1|2002"
        in machine.record("episode-1").approved_subscriptions
    )


def test_capacity_snapshot_is_machine_readable_and_owns_active_feeds() -> None:
    budget = _budget(total=50)
    _allocate_mandatory(budget, count=22)
    machine = BudgetAwareEpisodeStateMachine(
        budget=budget,
        max_option_lines_per_episode=8,
    )
    record = machine.submit(
        _task(
            (
                _intent(2001, role="primary_call", required=True),
                _intent(2002, role="primary_put", required=True),
            )
        ),
        now=NOW,
    )
    assert record.state is EpisodeState.PRIMARY_LEGS_STREAMING
    coordinator = OpeningReversalCapacityCoordinatorV1(budget=budget)

    snapshot = coordinator.snapshot(
        observed_at_utc=NOW,
        promoted_episode_id="episode-1",
    )

    assert snapshot.schema_version == "market_data_capacity_snapshot_v1"
    assert snapshot.configured_budget == 50
    assert snapshot.reserved_lines == 12
    assert snapshot.current_promoted_episode_id == "episode-1"
    assert {feed.subscription_identifier for feed in snapshot.active_subscriptions}.issuperset(
        {"OPTION_LEVEL1|2001", "OPTION_LEVEL1|2002"}
    )
    assert all(feed.owning_subsystem for feed in snapshot.active_subscriptions)
    optional_budget = _budget(total=60)
    _allocate_mandatory(optional_budget, count=22)
    optional_machine = BudgetAwareEpisodeStateMachine(
        budget=optional_budget,
        max_option_lines_per_episode=8,
    )
    optional_machine.submit(
        _task(
            (
                _intent(2201, role="primary_call", required=True),
                _intent(2202, role="primary_put", required=True),
                _intent(
                    2203,
                    role="0dte_call",
                    required=False,
                    drop_order=4,
                ),
            )
        ),
        now=NOW,
    )
    optional_snapshot = OpeningReversalCapacityCoordinatorV1(
        budget=optional_budget
    ).snapshot(
        observed_at_utc=NOW,
        promoted_episode_id="episode-1",
    )
    optional_feed = next(
        feed
        for feed in optional_snapshot.active_subscriptions
        if feed.subscription_identifier == "OPTION_LEVEL1|2203"
    )
    assert optional_feed.drop_order == 4
    assert len(snapshot.snapshot_hash) == 64
    assert snapshot.snapshot_hash == _hash_without(
        snapshot.model_dump(mode="json"),
        "snapshot_hash",
    )
    tampered = snapshot.model_dump(mode="python")
    tampered["mandatory_lines"] = snapshot.mandatory_lines + 1
    with pytest.raises(ValidationError, match="snapshot hash mismatch"):
        type(snapshot).model_validate(tampered)


def test_completion_releases_local_primary_capacity_ownership() -> None:
    budget = _budget(total=50)
    _allocate_mandatory(budget, count=22)
    machine = BudgetAwareEpisodeStateMachine(
        budget=budget,
        max_option_lines_per_episode=8,
    )
    machine.submit(
        _task(
            (
                _intent(2101, role="primary_call", required=True),
                _intent(2102, role="primary_put", required=True),
            )
        ),
        now=NOW,
    )

    machine.complete("episode-1", now=NOW + timedelta(minutes=30))

    assert budget.get("OPTION_LEVEL1|2101") is None
    assert budget.get("OPTION_LEVEL1|2102") is None


def test_primary_contract_selection_is_metadata_only_one_dte_and_deterministic() -> None:
    candidates = tuple(
        OptionContractCandidateV1(
            con_id=con_id,
            underlying="AAL",
            expiry=expiry,
            strike=strike,
            right=right,
            multiplier=100,
            exchange="SMART",
            trading_class="AAL",
        )
        for con_id, expiry, strike, right in (
            (1, date(2026, 7, 30), 99.0, "C"),
            (2, date(2026, 7, 30), 99.0, "P"),
            (3, date(2026, 7, 31), 99.0, "C"),
            (4, date(2026, 7, 31), 99.0, "P"),
            (5, date(2026, 7, 31), 101.0, "C"),
            (6, date(2026, 7, 31), 101.0, "P"),
            (7, date(2026, 8, 3), 100.0, "C"),
            (8, date(2026, 8, 3), 100.0, "P"),
        )
    )

    selection = select_primary_option_pair_v1(
        session=date(2026, 7, 30),
        underlying_reference=100.0,
        candidates=candidates,
        discovery_timestamp_utc=NOW,
        contract_source="ibkr_secdef_metadata",
        cache_hit=False,
    )

    assert selection.call.con_id == 3
    assert selection.put.con_id == 4
    assert selection.call.expiry == selection.put.expiry == date(2026, 7, 31)
    assert selection.call.strike == selection.put.strike == 99.0
    assert selection.candidates_inspected == len(candidates)
    assert selection.live_market_data_lines_consumed == 0
    assert selection.planned_live_market_data_lines == 2
    assert selection.metadata_request_ended
    assert not selection.full_chain_live_subscription_created
    assert selection.selection_hash == _hash_without(
        selection.model_dump(mode="json"),
        "selection_hash",
    )
    tampered = selection.model_dump(mode="python")
    tampered["cache_hit"] = True
    with pytest.raises(ValidationError, match="selection hash mismatch"):
        type(selection).model_validate(tampered)


def test_duplicate_contract_definitions_choose_lowest_con_id() -> None:
    candidates = tuple(
        OptionContractCandidateV1(
            con_id=con_id,
            underlying="AAL",
            expiry=date(2026, 7, 31),
            strike=100.0,
            right=right,
            multiplier=100,
            exchange="SMART",
            trading_class="AAL",
        )
        for con_id, right in (
            (19, "C"),
            (11, "C"),
            (20, "P"),
            (12, "P"),
        )
    )

    selection = select_primary_option_pair_v1(
        session=date(2026, 7, 30),
        underlying_reference=100.0,
        candidates=candidates,
        discovery_timestamp_utc=NOW,
        contract_source="ibkr_secdef_metadata",
        cache_hit=False,
    )

    assert selection.call.con_id == 11
    assert selection.put.con_id == 12


def test_missing_one_dte_pair_is_not_silently_substituted() -> None:
    candidates = (
        OptionContractCandidateV1(
            con_id=1,
            underlying="AAL",
            expiry=date(2026, 8, 3),
            strike=100.0,
            right="C",
            multiplier=100,
            exchange="SMART",
            trading_class="AAL",
        ),
        OptionContractCandidateV1(
            con_id=2,
            underlying="AAL",
            expiry=date(2026, 8, 3),
            strike=100.0,
            right="P",
            multiplier=100,
            exchange="SMART",
            trading_class="AAL",
        ),
    )

    try:
        select_primary_option_pair_v1(
            session=date(2026, 7, 30),
            underlying_reference=100.0,
            candidates=candidates,
            discovery_timestamp_utc=NOW,
            contract_source="cache",
            cache_hit=True,
        )
    except ValueError as error:
        assert str(error) == "primary_1dte_option_pair_unavailable"
    else:
        raise AssertionError("missing 1DTE was silently substituted")


def test_primary_option_evidence_uses_actual_ask_then_bid_not_midpoints() -> None:
    contract = OptionContractCandidateV1(
        con_id=41,
        underlying="AAL",
        expiry=date(2026, 7, 31),
        strike=100.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )
    entry = OptionTopOfBookV1(
        timestamp_utc=NOW,
        bid=1.0,
        ask=2.0,
        quote_age_seconds=0.0,
        locked_or_crossed=False,
        stale=False,
        missing_reason=None,
    )
    exit_quote = OptionTopOfBookV1(
        timestamp_utc=NOW + timedelta(minutes=15, seconds=1),
        bid=1.0,
        ask=3.0,
        quote_age_seconds=1.0,
        locked_or_crossed=False,
        stale=False,
        missing_reason=None,
    )

    outcome = build_primary_option_bid_ask_outcome_v1(
        prediction_receipt_hash_v1="a" * 64,
        contract=contract,
        role="predicted_leg",
        entry_timestamp_utc=NOW,
        subscription_start_utc=NOW,
        subscription_end_utc=NOW + timedelta(minutes=30),
        capacity_line_owner="opening_reversal_primary_pair:episode-1",
        entry_quote=entry,
        exit_quote=exit_quote,
    )

    assert outcome.complete
    assert outcome.conservative_return_v1 == -0.5
    assert outcome.entry_midpoint_diagnostic == 1.5
    assert outcome.exit_midpoint_diagnostic == 2.0
    assert outcome.outcome_hash_v1 == _hash_without(
        outcome.model_dump(mode="json"),
        "outcome_hash_v1",
    )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (("entry_quote", "bid", None), "entry quote quality"),
        (("exit_quote", "stale", True), "exit quote quality"),
        (("entry_spread", None, 99.0), "entry spread is inconsistent"),
        (
            ("conservative_return_v1", None, 0.75),
            "conservative return is inconsistent",
        ),
    ],
)
def test_direct_option_outcome_load_revalidates_quote_and_derived_invariants(
    mutation: tuple[str, str | None, object],
    expected_error: str,
) -> None:
    contract = OptionContractCandidateV1(
        con_id=41,
        underlying="AAL",
        expiry=date(2026, 7, 31),
        strike=100.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )
    valid = build_primary_option_bid_ask_outcome_v1(
        prediction_receipt_hash_v1="a" * 64,
        contract=contract,
        role="predicted_leg",
        entry_timestamp_utc=NOW,
        subscription_start_utc=NOW,
        subscription_end_utc=NOW + timedelta(minutes=30),
        capacity_line_owner="opening_reversal_primary_pair:episode-1",
        entry_quote=OptionTopOfBookV1(
            timestamp_utc=NOW,
            bid=1.0,
            ask=2.0,
            quote_age_seconds=0.0,
            locked_or_crossed=False,
            stale=False,
            missing_reason=None,
        ),
        exit_quote=OptionTopOfBookV1(
            timestamp_utc=NOW + timedelta(minutes=15),
            bid=1.0,
            ask=1.5,
            quote_age_seconds=0.0,
            locked_or_crossed=False,
            stale=False,
            missing_reason=None,
        ),
    )
    payload = valid.model_dump(mode="json")
    field, nested_field, value = mutation
    if nested_field is None:
        payload[field] = value
    else:
        nested = payload[field]
        assert isinstance(nested, dict)
        nested[nested_field] = value
    payload["outcome_hash_v1"] = _hash_without(payload, "outcome_hash_v1")

    with pytest.raises(ValidationError, match=expected_error):
        PrimaryOptionBidAskOutcomeV1.model_validate_json(json.dumps(payload))


def test_stale_or_crossed_option_quote_is_explicitly_incomplete() -> None:
    contract = OptionContractCandidateV1(
        con_id=41,
        underlying="AAL",
        expiry=date(2026, 7, 31),
        strike=100.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )
    entry = OptionTopOfBookV1(
        timestamp_utc=NOW,
        bid=1.0,
        ask=2.0,
        quote_age_seconds=0.0,
        locked_or_crossed=False,
        stale=False,
        missing_reason=None,
    )
    stale_exit = OptionTopOfBookV1(
        timestamp_utc=NOW + timedelta(minutes=16),
        bid=1.0,
        ask=1.5,
        quote_age_seconds=60.0,
        locked_or_crossed=False,
        stale=True,
        missing_reason=None,
    )

    outcome = build_primary_option_bid_ask_outcome_v1(
        prediction_receipt_hash_v1="a" * 64,
        contract=contract,
        role="predicted_leg",
        entry_timestamp_utc=NOW,
        subscription_start_utc=NOW,
        subscription_end_utc=NOW + timedelta(minutes=30),
        capacity_line_owner="opening_reversal_primary_pair:episode-1",
        entry_quote=entry,
        exit_quote=stale_exit,
    )

    assert not outcome.complete
    assert outcome.conservative_return_v1 is None
    assert outcome.missing_reason == "exit_quote_stale"


def test_missing_book_side_age_or_chronology_cannot_be_complete() -> None:
    contract = OptionContractCandidateV1(
        con_id=41,
        underlying="AAL",
        expiry=date(2026, 7, 31),
        strike=100.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )
    missing_entry_bid = OptionTopOfBookV1(
        timestamp_utc=NOW,
        bid=None,
        ask=2.0,
        quote_age_seconds=None,
        locked_or_crossed=False,
        stale=False,
        missing_reason=None,
    )
    early_exit = OptionTopOfBookV1(
        timestamp_utc=NOW + timedelta(minutes=14),
        bid=1.0,
        ask=1.5,
        quote_age_seconds=0.0,
        locked_or_crossed=False,
        stale=False,
        missing_reason=None,
    )

    missing_book = build_primary_option_bid_ask_outcome_v1(
        prediction_receipt_hash_v1="a" * 64,
        contract=contract,
        role="predicted_leg",
        entry_timestamp_utc=NOW,
        subscription_start_utc=NOW,
        subscription_end_utc=NOW + timedelta(minutes=30),
        capacity_line_owner="opening_reversal_primary_pair:episode-1",
        entry_quote=missing_entry_bid,
        exit_quote=OptionTopOfBookV1(
            timestamp_utc=NOW + timedelta(minutes=15),
            bid=1.0,
            ask=1.5,
            quote_age_seconds=0.0,
            locked_or_crossed=False,
            stale=False,
            missing_reason=None,
        ),
    )
    bad_chronology = build_primary_option_bid_ask_outcome_v1(
        prediction_receipt_hash_v1="a" * 64,
        contract=contract,
        role="predicted_leg",
        entry_timestamp_utc=NOW,
        subscription_start_utc=NOW,
        subscription_end_utc=NOW + timedelta(minutes=30),
        capacity_line_owner="opening_reversal_primary_pair:episode-1",
        entry_quote=OptionTopOfBookV1(
            timestamp_utc=NOW,
            bid=1.0,
            ask=2.0,
            quote_age_seconds=0.0,
            locked_or_crossed=False,
            stale=False,
            missing_reason=None,
        ),
        exit_quote=early_exit,
    )

    assert not missing_book.complete
    assert missing_book.missing_reason == "entry_bid_ask_invalid"
    assert not bad_chronology.complete
    assert bad_chronology.missing_reason == "exit_quote_age_invalid"
