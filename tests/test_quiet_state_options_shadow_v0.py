from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from stocker_prospective.contract import (
    CLAIMS_BOUNDARY,
    M1C_FEATURE_MANIFEST_SHA256,
    M1C_SCALING_ARTIFACT_SHA256,
    M1C_THRESHOLD_ARTIFACT_SHA256,
    SECTOR_PROXY_BY_SYMBOL,
)
from stocker_prospective.events import OptionQuoteEvent
from stocker_prospective.frozen_live_application import (
    _assert_frozen_m1c_artifact_hashes,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.option_ledger import OptionContract, build_shadow_outcomes
from stocker_prospective.option_recorder import (
    BoundedOptionRecorder,
    _quiet_virtual_leg_evidence,
    _required_quote_matrix_completion,
)
from stocker_prospective.options import DteBucket
from stocker_prospective.quiet_state import (
    BOTTOM_5_THRESHOLD,
    BOTTOM_10_THRESHOLD,
    BOTTOM_20_THRESHOLD,
    HIGH_TAIL_THRESHOLD,
    NEUTRAL_CONTROL_SALT,
    NeutralControlSampler,
    QuietEpisodeTracker,
    classify_quiet_state,
    high_tail_proximity,
)
from stocker_prospective.quiet_state_phase import (
    QuietObservationCompletion,
    QuietStatePhaseLedger,
)
from stocker_prospective.short_premium_shadow import (
    MAXIMUM_DELTA_DISTANCE,
    StructureType,
    calculate_credit_shadow,
    select_delta_iron_condor,
    select_fixed_width_credit_spread,
    select_iron_butterfly,
)

ENTRY = datetime(2026, 7, 27, 14, 0, tzinfo=UTC)
SESSION = date(2026, 7, 27)
CONTRACT_ROOT = (
    Path(__file__).parents[1] / "research/prospective/frozen-m1c-microstructure-recorder-v0"
)


def _contract(strike: float, right: str, con_id: int) -> OptionContract:
    return OptionContract(
        underlying_con_id=1,
        con_id=con_id,
        expiry=date(2026, 7, 31),
        dte=4,
        dte_bucket=DteBucket.THREE_TO_FIVE_DTE,
        strike=strike,
        right=right,  # type: ignore[arg-type]
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )


def _quote(
    contract: OptionContract,
    *,
    bid: float | None,
    ask: float | None,
    delta: float | None,
    seconds: int = 0,
    underlying: float = 100.0,
    market_data_type: MarketDataType = MarketDataType.LIVE,
) -> OptionQuoteEvent:
    observed = ENTRY + timedelta(seconds=seconds)
    return OptionQuoteEvent(
        event_id=f"{contract.con_id}-{seconds}",
        received_timestamp_utc=observed,
        received_monotonic_ns=contract.con_id or 0,
        provider_timestamp_utc=observed,
        source_sequence=contract.con_id or 0,
        session=SESSION,
        symbol="AAL",
        con_id=contract.con_id or 0,
        request_id=10,
        episode_id="quiet-1",
        expiry=contract.expiry,
        dte=contract.dte,
        dte_bucket=contract.dte_bucket,
        strike=contract.strike,
        right=contract.right,
        multiplier=contract.multiplier,
        exchange=contract.exchange,
        trading_class=contract.trading_class,
        bid=bid,
        bid_size=10.0,
        ask=ask,
        ask_size=12.0,
        last=None if bid is None or ask is None else (bid + ask) / 2.0,
        last_size=1.0,
        market_data_type=market_data_type,
        implied_volatility=0.4,
        delta=delta,
        gamma=0.05,
        theta=-0.02,
        vega=0.08,
        underlying_reference_price=underlying,
    )


def _chain() -> tuple[tuple[OptionContract, ...], tuple[OptionQuoteEvent, ...]]:
    definitions = (
        (95.0, "P", -0.10, 0.35, 0.40),
        (98.0, "P", -0.25, 0.90, 1.00),
        (100.0, "P", -0.50, 1.80, 1.95),
        (100.0, "C", 0.50, 1.70, 1.85),
        (102.0, "C", 0.25, 0.85, 0.95),
        (105.0, "C", 0.10, 0.30, 0.35),
    )
    contracts = tuple(
        _contract(strike, right, index + 1)
        for index, (strike, right, _delta, _bid, _ask) in enumerate(definitions)
    )
    quotes = tuple(
        _quote(contract, bid=bid, ask=ask, delta=delta)
        for contract, (_strike, _right, delta, bid, ask) in zip(
            contracts,
            definitions,
            strict=True,
        )
    )
    return contracts, quotes


@pytest.mark.parametrize(
    ("probability", "bottom_5", "bottom_10", "bottom_20", "high_tail"),
    [
        (BOTTOM_5_THRESHOLD, True, True, True, False),
        (BOTTOM_10_THRESHOLD, False, True, True, False),
        (BOTTOM_20_THRESHOLD, False, False, True, False),
        (HIGH_TAIL_THRESHOLD, False, False, False, True),
    ],
)
def test_quiet_state_memberships_are_inclusive_and_frozen(
    probability: float,
    bottom_5: bool,
    bottom_10: bool,
    bottom_20: bool,
    high_tail: bool,
) -> None:
    state = classify_quiet_state(
        probability=probability,
        previous_probability=0.2,
        model_hash="a" * 64,
        feature_hash="b" * 64,
        data_quality_status="valid",
    )

    assert state.bottom_5 is bottom_5
    assert state.bottom_10 is bottom_10
    assert state.bottom_20 is bottom_20
    assert state.high_tail is high_tail
    assert state.distance_from_bottom_10 == pytest.approx(probability - BOTTOM_10_THRESHOLD)


def test_quiet_episode_tracker_uses_downward_crossing_and_thirty_minute_spacing() -> None:
    tracker = QuietEpisodeTracker()
    first = tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        trigger_bar_end=ENTRY,
        probability=0.13,
    )
    continued = tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=8,
        trigger_bar_end=ENTRY + timedelta(minutes=10),
        probability=0.12,
    )
    tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=10,
        trigger_bar_end=ENTRY + timedelta(minutes=20),
        probability=0.20,
    )
    suppressed = tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=12,
        trigger_bar_end=ENTRY + timedelta(minutes=25),
        probability=0.10,
    )
    tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=14,
        trigger_bar_end=ENTRY + timedelta(minutes=30),
        probability=0.20,
    )
    second = tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=16,
        trigger_bar_end=ENTRY + timedelta(minutes=35),
        probability=0.11,
    )

    assert first.fresh_episode is True
    assert first.previous_probability is None
    assert first.prospective_entry_timestamp == ENTRY
    assert continued.fresh_episode is False
    assert suppressed.fresh_episode is False
    assert suppressed.rejection_reason == "minimum_episode_spacing_not_met"
    assert second.fresh_episode is True
    assert second.episode_number == 2
    assert second.minutes_since_previous_episode == 35.0


def test_neutral_control_sampling_is_deterministic_and_outcome_independent() -> None:
    sampler = NeutralControlSampler(
        salt=NEUTRAL_CONTROL_SALT,
        sampling_fraction=0.10,
    )
    arguments = {
        "session": SESSION,
        "symbol": "AAL",
        "checkpoint": 6,
        "model_hash": "a" * 64,
        "probability": 0.25,
        "eligible": True,
    }

    first = sampler.evaluate(**arguments)
    second = sampler.evaluate(**arguments)

    assert first == second
    assert first.population_eligible is True
    assert 0.0 <= first.hash_fraction < 1.0
    assert first.selected is (first.hash_fraction < 0.10)
    assert "outcome" not in first.model_dump()


def test_neutral_control_rejects_tail_and_high_tail_populations() -> None:
    sampler = NeutralControlSampler()

    low = sampler.evaluate(
        session=SESSION,
        symbol="AAL",
        checkpoint=6,
        model_hash="a" * 64,
        probability=BOTTOM_20_THRESHOLD,
        eligible=True,
    )
    high = sampler.evaluate(
        session=SESSION,
        symbol="AAL",
        checkpoint=6,
        model_hash="a" * 64,
        probability=HIGH_TAIL_THRESHOLD,
        eligible=True,
    )

    assert low.population_eligible is False
    assert high.population_eligible is False
    assert low.selected is False
    assert high.selected is False


def test_high_tail_proximity_reports_prior_and_following_sixty_minutes() -> None:
    proximity = high_tail_proximity(
        trigger_timestamp=ENTRY,
        high_tail_timestamps=(
            ENTRY - timedelta(minutes=45),
            ENTRY + timedelta(minutes=60),
            ENTRY + timedelta(minutes=70),
        ),
    )

    assert proximity.previous_60_minutes is True
    assert proximity.following_60_minutes is True
    assert proximity.any_within_60_minutes is True


def test_quiet_phase_boundaries_are_30_150_150_and_controls_follow_chronology(
    tmp_path: Path,
) -> None:
    ledger = QuietStatePhaseLedger(tmp_path / "quiet-phases.jsonl")
    complete = QuietObservationCompletion.all_valid()
    assignments = [
        ledger.record(
            observation_id=f"quiet-{index:03d}",
            observation_kind="quiet_bottom_10",
            occurred_at=ENTRY + timedelta(minutes=index),
            completion=complete,
        )
        for index in range(181)
    ]
    neutral = ledger.record(
        observation_id="neutral-181",
        observation_kind="neutral_control",
        occurred_at=ENTRY + timedelta(minutes=181),
        completion=complete,
    )

    assert assignments[29].phase == "engineering_shakedown"
    assert assignments[30].phase == "quiet_state_development"
    assert assignments[179].phase == "quiet_state_development"
    assert assignments[180].phase == "quiet_state_confirmation"
    assert assignments[180].target_dependent_selection_opened is False
    assert neutral.phase == "quiet_state_confirmation"
    assert neutral.complete_quiet_episode_ordinal is None


def test_quiet_phase_rejects_naive_timestamps(tmp_path: Path) -> None:
    ledger = QuietStatePhaseLedger(tmp_path / "quiet-phases.jsonl")

    with pytest.raises(ValueError, match="timezone-aware"):
        ledger.record(
            observation_id="quiet-naive",
            observation_kind="quiet_bottom_10",
            occurred_at=datetime(2026, 7, 27, 14, 0),
            completion=QuietObservationCompletion.all_valid(),
        )


def test_frozen_m1c_startup_hashes_are_exact_and_fail_closed(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    primary = (
        root
        / "research/directional-readiness"
        / "20260726-stock-local-directional-archetypes-v0"
        / "artifacts/primary"
    )
    scaling = (
        root
        / "research/route-competition"
        / "20260722-broad-conflict-advance-hazard-v02"
        / "artifacts/primary/model_configurations.json"
    )
    exact = {
        "m1c_feature_manifest": primary / "causal_movement_feature_manifest.json",
        "m1c_threshold": primary / "causal_movement_threshold.json",
        "m1c_scaling": scaling,
    }

    _assert_frozen_m1c_artifact_hashes(exact)
    assert {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in exact.items()
    } == {
        "m1c_feature_manifest": M1C_FEATURE_MANIFEST_SHA256,
        "m1c_threshold": M1C_THRESHOLD_ARTIFACT_SHA256,
        "m1c_scaling": M1C_SCALING_ARTIFACT_SHA256,
    }

    changed = tmp_path / "threshold.json"
    changed.write_bytes(exact["m1c_threshold"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="blocked_frozen_artifact_hash_mismatch"):
        _assert_frozen_m1c_artifact_hashes({**exact, "m1c_threshold": changed})


def test_iron_butterfly_uses_atm_short_legs_and_symmetric_one_percent_wings() -> None:
    contracts, _ = _chain()

    structure = select_iron_butterfly(
        contracts=contracts,
        underlying_entry_price=100.0,
    )

    assert structure.available is True
    assert structure.structure_type is StructureType.ATM_IRON_BUTTERFLY
    assert [(leg.side, leg.contract.strike, leg.contract.right) for leg in structure.legs] == [
        ("short", 100.0, "C"),
        ("short", 100.0, "P"),
        ("long", 102.0, "C"),
        ("long", 98.0, "P"),
    ]


def test_iron_butterfly_marks_asymmetric_or_missing_wings_unavailable() -> None:
    contracts, _ = _chain()
    asymmetric = tuple(
        contract
        for contract in contracts
        if not (contract.right == "P" and contract.strike in {95.0, 98.0})
    )

    structure = select_iron_butterfly(
        contracts=asymmetric,
        underlying_entry_price=100.0,
    )

    assert structure.available is False
    assert "symmetric_wings_unavailable" in structure.quality_flags


def test_delta_iron_condor_uses_frozen_targets_ordering_and_tolerance() -> None:
    contracts, quotes = _chain()

    structure = select_delta_iron_condor(
        contracts=contracts,
        entry_quotes=quotes,
        maximum_delta_distance=MAXIMUM_DELTA_DISTANCE,
    )

    assert structure.available is True
    assert [(leg.side, leg.contract.strike, leg.contract.right) for leg in structure.legs] == [
        ("short", 102.0, "C"),
        ("short", 98.0, "P"),
        ("long", 105.0, "C"),
        ("long", 95.0, "P"),
    ]
    assert all(distance == pytest.approx(0.0) for distance in structure.delta_distances)


def test_quiet_virtual_leg_evidence_freezes_each_conservative_bid_ask_side() -> None:
    contracts, entry_quotes = _chain()
    structure = select_delta_iron_condor(
        contracts=contracts,
        entry_quotes=entry_quotes,
    )
    exit_quotes = tuple(
        quote.model_copy(
            update={
                "event_id": f"exit-{quote.event_id}",
                "received_timestamp_utc": ENTRY + timedelta(minutes=15),
                "provider_timestamp_utc": ENTRY + timedelta(minutes=15),
                "bid": None if quote.bid is None else quote.bid * 0.5,
                "ask": None if quote.ask is None else quote.ask * 0.5,
            }
        )
        for quote in entry_quotes
    )

    evidence = _quiet_virtual_leg_evidence(
        structure=structure,
        entry_surface=entry_quotes,
        exit_surface=exit_quotes,
    )

    assert [(leg["side"], leg["right"], leg["strike"]) for leg in evidence] == [
        ("short", "C", 102.0),
        ("short", "P", 98.0),
        ("long", "C", 105.0),
        ("long", "P", 95.0),
    ]
    assert [leg["entry_fill_price"] for leg in evidence] == pytest.approx([0.85, 0.90, 0.35, 0.40])
    assert [leg["exit_fill_price"] for leg in evidence] == pytest.approx([0.475, 0.50, 0.15, 0.175])
    assert all(leg["entry_quote_timestamp_utc"] == ENTRY.isoformat() for leg in evidence)
    assert all(
        leg["exit_quote_timestamp_utc"] == (ENTRY + timedelta(minutes=15)).isoformat()
        for leg in evidence
    )


def test_delta_iron_condor_fails_closed_outside_delta_tolerance() -> None:
    contracts, quotes = _chain()
    altered = tuple(
        quote.model_copy(update={"delta": 0.40})
        if quote.right == "C" and quote.strike == 102.0
        else quote
        for quote in quotes
    )

    structure = select_delta_iron_condor(
        contracts=contracts,
        entry_quotes=altered,
        maximum_delta_distance=0.01,
    )

    assert structure.available is False
    assert "delta_tolerance_failed" in structure.quality_flags


def test_delta_iron_condor_records_asymmetric_wing_quality() -> None:
    contracts, quotes = _chain()
    altered_contracts = tuple(
        contract
        if contract.con_id != 6
        else OptionContract(**{**contract.__dict__, "strike": 106.0})
        for contract in contracts
    )
    altered_quotes = tuple(
        quote if quote.con_id != 6 else quote.model_copy(update={"strike": 106.0})
        for quote in quotes
    )

    structure = select_delta_iron_condor(
        contracts=altered_contracts,
        entry_quotes=altered_quotes,
    )

    assert structure.available is True
    assert "asymmetric_wings" in structure.quality_flags


def test_zero_bid_is_retained_as_full_loss_exit_with_quality_flag() -> None:
    contract = _contract(100.0, "C", 101)
    quotes = (
        _quote(contract, bid=1.0, ask=1.1, delta=0.5),
        _quote(contract, bid=0.0, ask=0.05, delta=0.5, seconds=300),
    )

    outcome = build_shadow_outcomes(
        episode_id="quiet-zero-bid",
        symbol="AAL",
        entry_timestamp=ENTRY,
        contracts=(contract,),
        quotes=quotes,
        horizons=(timedelta(minutes=5),),
        maximum_quote_age=timedelta(seconds=5),
    )[0]

    assert outcome.exit_bid == 0.0
    assert outcome.dollar_pnl_per_contract == pytest.approx(-110.0)
    assert "zero_bid" in outcome.quote_quality_flags


def test_quiet_phase_requires_the_complete_contract_horizon_matrix() -> None:
    contract = _contract(100.0, "C", 102)
    partial_quotes = (
        _quote(contract, bid=1.0, ask=1.1, delta=0.5),
        _quote(contract, bid=0.8, ask=0.9, delta=0.5, seconds=300),
    )
    partial_outcomes = build_shadow_outcomes(
        episode_id="quiet-partial-matrix",
        symbol="AAL",
        entry_timestamp=ENTRY,
        contracts=(contract,),
        quotes=partial_quotes,
        horizons=(timedelta(minutes=5), timedelta(minutes=10)),
        maximum_quote_age=timedelta(seconds=5),
    )

    assert _required_quote_matrix_completion(
        planned_contracts=(contract,),
        requested_contract_count=1,
        plan_capacity_reduced=False,
        outcomes=partial_outcomes,
        horizon_minutes=(5, 10),
        subscription_gap_spans_horizon=False,
    ) == (2, 1, False)

    complete_outcomes = build_shadow_outcomes(
        episode_id="quiet-complete-matrix",
        symbol="AAL",
        entry_timestamp=ENTRY,
        contracts=(contract,),
        quotes=(
            *partial_quotes,
            _quote(contract, bid=0.7, ask=0.8, delta=0.5, seconds=600),
        ),
        horizons=(timedelta(minutes=5), timedelta(minutes=10)),
        maximum_quote_age=timedelta(seconds=5),
    )
    assert _required_quote_matrix_completion(
        planned_contracts=(contract,),
        requested_contract_count=1,
        plan_capacity_reduced=False,
        outcomes=complete_outcomes,
        horizon_minutes=(5, 10),
        subscription_gap_spans_horizon=False,
    ) == (2, 2, True)
    assert _required_quote_matrix_completion(
        planned_contracts=(contract,),
        requested_contract_count=1,
        plan_capacity_reduced=False,
        outcomes=complete_outcomes,
        horizon_minutes=(5, 10),
        subscription_gap_spans_horizon=True,
    ) == (2, 2, False)
    assert _required_quote_matrix_completion(
        planned_contracts=(contract,),
        requested_contract_count=2,
        plan_capacity_reduced=True,
        outcomes=complete_outcomes,
        horizon_minutes=(5, 10),
        subscription_gap_spans_horizon=False,
    ) == (4, 2, False)


def test_capacity_omitted_bucket_emits_every_unavailable_structure_attempt() -> None:
    class StructureRepository:
        def __init__(self) -> None:
            self.rows: list[dict[str, Any]] = []

        def record_quiet_shadow_structure(
            self,
            *_args: object,
            **kwargs: Any,
        ) -> None:
            self.rows.append(kwargs)

    repository = StructureRepository()
    recorder = BoundedOptionRecorder(
        adapter=cast(Any, object()),
        subscriptions=cast(Any, object()),
        repository=cast(Any, repository),
        raw_store=cast(Any, object()),
        maximum_quote_age=timedelta(seconds=5),
    )

    recorder._record_quiet_bucket_outcomes(
        cast(Any, object()),
        observation_id="quiet-capacity-omitted",
        symbol="AAL",
        entry_timestamp=ENTRY,
        bucket=DteBucket.ZERO_DTE,
        planned_bucket_contracts=(),
        bucket_contracts=(),
        atm_call=None,
        atm_put=None,
        quotes=(),
        outcomes=(),
        horizon_minutes=(5,),
        subscription_gap_spans_horizon=False,
        plan_capacity_reduced=True,
    )

    assert {row["structure_type"] for row in repository.rows} == {
        "LONG_CALL",
        "LONG_PUT",
        "ATM_STRADDLE",
        "ATM_IRON_BUTTERFLY",
        "DELTA_IRON_CONDOR",
        "CALL_CREDIT_SPREAD",
        "PUT_CREDIT_SPREAD",
    }
    assert all(row["attempted"] is True for row in repository.rows)
    assert all("option_plan_capacity_reduced" in row["quality_flags"] for row in repository.rows)


def test_credit_shadow_uses_conservative_fills_max_risk_and_touch_flags() -> None:
    contracts, entry_quotes = _chain()
    structure = select_delta_iron_condor(
        contracts=contracts,
        entry_quotes=entry_quotes,
    )
    exit_quotes = tuple(
        quote.model_copy(
            update={
                "event_id": f"exit-{quote.event_id}",
                "received_timestamp_utc": ENTRY + timedelta(minutes=15),
                "provider_timestamp_utc": ENTRY + timedelta(minutes=15),
                "bid": None if quote.bid is None else quote.bid * 0.5,
                "ask": None if quote.ask is None else quote.ask * 0.5,
            }
        )
        for quote in entry_quotes
    )
    adverse_mark = tuple(
        quote.model_copy(
            update={
                "event_id": f"mark-{quote.event_id}",
                "received_timestamp_utc": ENTRY + timedelta(minutes=5),
                "provider_timestamp_utc": ENTRY + timedelta(minutes=5),
                "bid": None if quote.bid is None else quote.bid * 1.2,
                "ask": None if quote.ask is None else quote.ask * 1.5,
            }
        )
        for quote in entry_quotes
    )

    result = calculate_credit_shadow(
        structure=structure,
        entry_quotes=entry_quotes,
        exit_quotes=exit_quotes,
        entry_timestamp=ENTRY,
        exit_timestamp=ENTRY + timedelta(minutes=15),
        underlying_path=(100.0, 102.5, 104.0, 99.0),
        mark_quote_surfaces=(adverse_mark,),
        configured_commission_per_contract=0.65,
    )

    expected_credit = (0.85 + 0.90 - 0.35 - 0.40) * 100
    expected_debit = (0.95 * 0.5 + 1.00 * 0.5 - 0.30 * 0.5 - 0.35 * 0.5) * 100
    assert result.opening_net_credit == pytest.approx(expected_credit)
    assert result.closing_debit == pytest.approx(expected_debit)
    assert result.commission_free_pnl == pytest.approx(expected_credit - expected_debit)
    assert result.configured_commission_pnl == pytest.approx(
        expected_credit - expected_debit - 8 * 0.65
    )
    assert result.maximum_defined_risk == pytest.approx(300.0 - expected_credit)
    assert result.return_on_maximum_risk == pytest.approx(
        result.commission_free_pnl / result.maximum_defined_risk
    )
    assert result.short_strike_touched is True
    assert result.short_strike_crossed is True
    assert result.protective_wing_touched is False
    adverse_debit = (0.95 * 1.5 + 1.00 * 1.5 - 0.30 * 1.2 - 0.35 * 1.2) * 100
    assert result.maximum_adverse_marked_pnl == pytest.approx(expected_credit - adverse_debit)
    assert result.maximum_favourable_marked_pnl == pytest.approx(expected_credit - expected_debit)

    halted = calculate_credit_shadow(
        structure=structure,
        entry_quotes=entry_quotes,
        exit_quotes=exit_quotes,
        entry_timestamp=ENTRY,
        exit_timestamp=ENTRY + timedelta(minutes=15),
        underlying_path=(100.0, 101.0),
        additional_quality_flags=("underlying_halted",),
    )
    assert halted.strict_quote_quality is False
    assert "underlying_halted" in halted.quote_quality_flags


@pytest.mark.parametrize("right", ["C", "P"])
def test_fixed_width_credit_spreads_are_defined_risk(right: str) -> None:
    contracts, _ = _chain()

    structure = select_fixed_width_credit_spread(
        contracts=contracts,
        underlying_entry_price=100.0,
        right=right,  # type: ignore[arg-type]
    )

    assert structure.available is True
    assert len(structure.legs) == 2
    assert {leg.side for leg in structure.legs} == {"short", "long"}


def test_shadow_records_missing_stale_and_non_live_quote_quality() -> None:
    contracts, quotes = _chain()
    structure = select_iron_butterfly(
        contracts=contracts,
        underlying_entry_price=100.0,
    )
    broken = tuple(
        quote.model_copy(
            update={
                "bid": None if quote.con_id == 5 else quote.bid,
                "ask": None if quote.con_id == 5 else quote.ask,
                "market_data_type": (
                    MarketDataType.DELAYED if quote.con_id == 2 else quote.market_data_type
                ),
            }
        )
        for quote in quotes
    )

    result = calculate_credit_shadow(
        structure=structure,
        entry_quotes=broken,
        exit_quotes=broken,
        entry_timestamp=ENTRY + timedelta(minutes=2),
        exit_timestamp=ENTRY + timedelta(minutes=2),
        underlying_path=(100.0,),
        additional_quality_flags=("subscription_started_late",),
    )

    assert result.complete_quote_quality is False
    assert result.strict_quote_quality is False
    assert "missing_leg_quote" in result.quote_quality_flags
    assert "exit_quote_unavailable" in result.quote_quality_flags
    assert "stale_quote" in result.quote_quality_flags
    assert "market_data_not_live" in result.quote_quality_flags
    assert "subscription_started_late" in result.quote_quality_flags


def test_all_quiet_state_contract_artifacts_embed_binding_claims() -> None:
    required_claims = {
        key: CLAIMS_BOUNDARY[key]
        for key in (
            "research_only",
            "original_low_movement_decision_preserved",
            "original_decision",
            "retrospective_gate_relaxation_allowed",
            "m1c_frozen",
            "m1c_bottom_5_threshold",
            "m1c_bottom_10_threshold",
            "m1c_bottom_20_threshold",
            "primary_quiet_state",
            "prospective_record_only",
            "option_shadow_outcomes_only",
            "defined_risk_short_premium_only",
            "naked_short_options_allowed",
            "paper_orders_allowed",
            "live_orders_allowed",
            "broker_order_methods_allowed",
            "strategy_promotion",
            "protected_historical_start",
        )
    }
    artifact_names = (
        "quiet_state_signal_contract.json",
        "quiet_state_threshold_manifest.json",
        "neutral_control_sampling_contract.json",
        "quiet_state_option_selection_contract.json",
        "defined_risk_structure_contract.json",
        "conservative_fill_contract.json",
        "quiet_state_phase_contract.json",
        "quiet_state_schema_manifest.json",
        "quiet_state_safety_contract.json",
        "quiet_state_independent_audit.json",
        "quiet_state_determinism_check.json",
    )

    for name in artifact_names:
        payload = json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))
        assert {key: payload[key] for key in required_claims} == required_claims
    signal = json.loads(
        (CONTRACT_ROOT / "quiet_state_signal_contract.json").read_text(encoding="utf-8")
    )
    assert signal["descriptive_context_proxies"]["sector_proxy_by_symbol"] == SECTOR_PROXY_BY_SYMBOL
