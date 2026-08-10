from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from stocker_ideas.plugins.frozen_m1c_v0 import COHORT
from stocker_ideas.plugins.m1c_quiet_state_options_v0 import (
    PARAMETERS,
    M1CQuietStateOptionsV0,
)
from stocker_ideas.plugins.m1c_quiet_state_v0 import (
    BOTTOM_10_THRESHOLD,
    OptionPanelContract,
    classify_quiet_state,
    select_defined_risk_structures,
)
from stocker_runtime.domain import (
    JsonValue,
    MarketEvent,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
)
from stocker_runtime.ideas.contract import (
    DiscoveryReceipt,
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
)
from stocker_runtime.ingestion.dynamic_market_data import (
    ContractCandidate,
    InstrumentResolver,
    InterestResolutionRequest,
    MarketDataCapacity,
    MarketDataDemand,
    OptionParameterSet,
    plan_market_data,
)
from stocker_runtime.ingestion.recorder import InstrumentSpec


def _hash(value: Mapping[str, JsonValue]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _activation() -> IdeaActivation:
    return IdeaActivation(
        instance_id="quiet-1",
        parameters=PARAMETERS,
        parameters_hash=_hash(PARAMETERS),
        plugin_code_hash="1" * 64,
        activated_at_us=1,
        run_id="run-1",
        protected_data_class=ProtectedDataClass.PROSPECTIVE,
        universe=COHORT,
    )


def _event(
    event_id: str,
    instrument_id: str,
    event_kind: str,
    at_us: int,
    payload: Mapping[str, JsonValue],
) -> MarketEvent:
    return MarketEvent(
        event_id=event_id,
        instrument_id=instrument_id,
        feed_kind="quotes" if event_kind in {"quote", "option_snapshot_capture"} else "bars",
        event_kind=event_kind,
        event_at_us=at_us,
        received_at_us=at_us,
        payload=payload,
    )


def _prefix_payload(*, symbol: str, session: str, checkpoint: int = 6) -> dict[str, JsonValue]:
    base = 40.0
    trailing: list[dict[str, JsonValue]] = []
    for number in range(1, checkpoint + 1):
        opening = base + number / 10.0
        trailing.append(
            {
                "bar_number": number,
                "event_at_us": 1_000_000 * number,
                "open": opening,
                "high": opening + 1.0,
                "low": opening - 1.0,
                "close": opening + 0.2,
                "volume": 1_000.0 + number,
                "historical_relative_activity": 1.2,
                "return_bps": 10.0,
                "true_range_bps": 200.0,
                "upper_wick_fraction": 0.4,
                "lower_wick_fraction": 0.4,
                "session_vwap": base + number / 20.0,
            }
        )
    return {
        "schema_version": 1,
        "session": session,
        "bar_number": checkpoint,
        "source_completeness": "complete",
        "historical_session_count": 21,
        "historical_relative_activity": 1.2,
        "activity_ready": True,
        "trailing_bars": trailing,
        "session_volumes": [1_000.0 + number for number in range(1, checkpoint + 1)],
        "accumulator": {
            "bar_count": checkpoint,
            "session_open": base,
            "session_high": base + checkpoint / 10.0 + 1.0,
            "session_low": base - 0.9,
            "last_close": base + checkpoint / 10.0 + 0.2,
            "activity_sum": checkpoint * 1.2,
            "range_sum": checkpoint * 200.0,
            "travel_sum": checkpoint * 10.0,
            "return_sum": checkpoint * 10.0,
            "width_sum": checkpoint * 2.0,
            "positive_return_count": checkpoint,
            "negative_return_count": 0,
            "zero_return_count": 0,
        },
        "fixture_symbol": symbol,
    }


def _ordinary_batch(
    events: tuple[MarketEvent, ...],
    *,
    retained: tuple[str, ...] = (),
    receipts: tuple[DiscoveryReceipt, ...] = (),
) -> IdeaBatch:
    return IdeaBatch(
        mode=RuntimeMode.SHADOW,
        events=events,
        input_watermark=events[-1].event_id,
        causal_from_at_us=min(event.event_at_us for event in events),
        causal_through_at_us=max(event.event_at_us for event in events),
        prior_state_input_event_ids=retained,
        discovery_receipts=receipts,
    )


def _prime_low_tail_context(
    plugin: M1CQuietStateOptionsV0,
    *,
    symbol: str = "RGTI",
    state: JsonValue | None = None,
    retained: tuple[str, ...] = (),
    all_cohort_low: bool = False,
) -> tuple[JsonValue, tuple[str, ...]]:
    prior_close = 22.0 if all_cohort_low else 40.0
    realised_volatility = 1.2 if all_cohort_low else 0.58
    baseline_event = _event(
        f"baseline-{symbol}",
        symbol,
        "session_volume_baseline",
        1_000_000,
        {
            "session": "2026-08-07",
            "complete_session_count": 21,
            "session_closes": [prior_close - 0.5, prior_close],
            "realised_volatility_20d": realised_volatility,
        },
    )
    baseline = plugin.evaluate(
        _ordinary_batch((baseline_event,), retained=retained),
        {} if state is None else state,
    )
    assert len(baseline.interests) == 2
    receipts = tuple(
        DiscoveryReceipt(
            receipt_id=f"receipt-{symbol}-{right}",
            interest_id=f"interest-{symbol}-{right}",
            interest_key=f"quiet:m1c:d1:2026-08-07:{symbol}:{right}",
            instance_id="quiet-1",
            status="resolved",
            instrument_id=f"{symbol}-d1-{right}",
            expiry="20260918",
            strike=prior_close,
            option_right=right,
            multiplier="100",
            candidates_inspected=20,
            completed_at_us=1_100_000,
        )
        for right in ("call", "put")
    )
    captures = tuple(
        _event(
            f"capture-{symbol}-{right}",
            f"{symbol}-d1-{right}",
            "option_snapshot_capture",
            1_200_000 + index,
            {
                "source_completeness": "complete",
                "bid": (
                    (3.2 if right == "call" else 2.1)
                    if all_cohort_low
                    else (4.0 if right == "call" else 3.2)
                ),
                "ask": (
                    (3.4 if right == "call" else 5.3)
                    if all_cohort_low
                    else (11.5 if right == "call" else 9.1)
                ),
                "model_implied_volatility": (
                    (4.9 if right == "call" else 5.0)
                    if all_cohort_low
                    else (3.4 if right == "call" else 1.1)
                ),
            },
        )
        for index, right in enumerate(("call", "put"))
    )
    captured = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=baseline.retained_input_event_ids,
            receipts=receipts,
        ),
        baseline.state,
    )
    return captured.state, captured.retained_input_event_ids


def _planned_entry_panel() -> tuple[
    M1CQuietStateOptionsV0,
    IdeaEvaluation,
    MarketEvent,
    MarketEvent,
]:
    plugin = M1CQuietStateOptionsV0()
    state, retained = _prime_low_tail_context(plugin)
    prefix = _event(
        "prefix-rgti-6",
        "RGTI",
        "bar_5m_session_prefix",
        2_000_000_000,
        _prefix_payload(symbol="RGTI", session="2026-08-10"),
    )
    triggered = plugin.evaluate(_ordinary_batch((prefix,), retained=retained), state)
    quote = _event(
        "quote-rgti-entry",
        "RGTI",
        "quote",
        prefix.event_at_us + 1,
        {"bid": 40.0, "ask": 40.2},
    )
    planned = plugin.evaluate(
        _ordinary_batch((quote,), retained=triggered.retained_input_event_ids),
        triggered.state,
    )
    return plugin, planned, prefix, quote


def _entry_panel_evidence(
    planned: IdeaEvaluation,
    *,
    incomplete_key: str | None = None,
) -> tuple[tuple[DiscoveryReceipt, ...], tuple[MarketEvent, ...]]:
    expiries = {(0, 0): "20260810", (1, 1): "20260811", (3, 5): "20260813"}
    call_deltas = (0.95, 0.85, 0.75, 0.55, 0.50, 0.40, 0.25, 0.10, 0.05)
    put_deltas = (-0.05, -0.10, -0.25, -0.40, -0.50, -0.55, -0.75, -0.85, -0.95)
    receipts: list[DiscoveryReceipt] = []
    captures: list[MarketEvent] = []
    for index, interest in enumerate(planned.interests):
        bucket = (interest.minimum_days_to_expiry, interest.maximum_days_to_expiry)
        offset = interest.strike_offset
        right = interest.option_right
        instrument = f"option-{bucket[0]}-{bucket[1]}-{offset}-{right}"
        receipts.append(
            DiscoveryReceipt(
                receipt_id=f"receipt-{index}",
                interest_id=f"interest-{index}",
                interest_key=interest.interest_key,
                instance_id="quiet-1",
                status="resolved",
                instrument_id=instrument,
                expiry=expiries[bucket],
                strike=40.0 + offset,
                option_right=right,
                multiplier="100",
                candidates_inspected=40,
                completed_at_us=2_000_000_100 + index,
            )
        )
        bid, ask = (
            (1.50, 1.60) if offset == 0 else (0.85, 0.95) if abs(offset) == 2 else (0.25, 0.30)
        )
        delta = call_deltas[offset + 4] if right == "call" else put_deltas[offset + 4]
        captures.append(
            _event(
                f"panel-capture-{index}",
                instrument,
                "option_snapshot_capture",
                2_000_000_200 + index,
                {
                    "source_completeness": (
                        "incomplete" if interest.interest_key == incomplete_key else "complete"
                    ),
                    "bid": bid,
                    "ask": ask,
                    "model_implied_volatility": 0.50,
                    "model_delta": delta,
                },
            )
        )
    return tuple(receipts), tuple(captures)


class _PanelDiscoveryBackend:
    def __init__(self) -> None:
        self.parameter_calls = 0
        self.contract_calls: list[tuple[str, float, str]] = []

    def option_parameters(
        self,
        *,
        underlying_con_id: int,
        symbol: str,
    ) -> tuple[OptionParameterSet, ...]:
        assert underlying_con_id == 1234
        assert symbol == "RGTI"
        self.parameter_calls += 1
        return (
            OptionParameterSet(
                exchange="SMART",
                trading_class="RGTI",
                multiplier="100",
                expirations=("20260810", "20260811", "20260813"),
                strikes=tuple(float(value) for value in range(36, 45)),
            ),
        )

    def option_contracts(
        self,
        *,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        multiplier: str,
        trading_class: str,
    ) -> tuple[ContractCandidate, ...]:
        self.contract_calls.append((expiry, strike, right))
        return (
            ContractCandidate(
                con_id=int(expiry[-2:]) * 10_000 + int(strike * 10) * 10 + int(right == "P"),
                symbol=symbol,
                expiry=expiry,
                strike=strike,
                right=right,
                multiplier=multiplier,
                exchange="SMART",
                currency="USD",
                trading_class=trading_class,
            ),
        )


def _bucket_contracts() -> tuple[OptionPanelContract, ...]:
    strikes = tuple(float(value) for value in range(96, 105))
    call_deltas = (0.95, 0.85, 0.75, 0.55, 0.50, 0.40, 0.25, 0.10, 0.05)
    put_deltas = (-0.05, -0.10, -0.25, -0.40, -0.50, -0.55, -0.75, -0.85, -0.95)
    contracts: list[OptionPanelContract] = []
    for offset, strike in zip(range(-4, 5), strikes, strict=True):
        bid, ask = (
            (1.50, 1.60) if offset == 0 else (0.85, 0.95) if abs(offset) == 2 else (0.25, 0.30)
        )
        for right, delta in (
            ("call", call_deltas[offset + 4]),
            ("put", put_deltas[offset + 4]),
        ):
            contracts.append(
                OptionPanelContract(
                    event_id=f"capture-{offset}-{right}",
                    interest_key=f"quiet:episode:entry:0DTE:{offset}:{right}",
                    instrument_id=f"option-{offset}-{right}",
                    bucket="0DTE",
                    offset=offset,
                    right=right,
                    expiry="20260810",
                    strike=strike,
                    multiplier="100",
                    bid=bid,
                    ask=ask,
                    delta=delta,
                )
            )
    return tuple(contracts)


def test_quiet_plugin_declares_only_the_existing_generic_market_data_seams() -> None:
    plugin = M1CQuietStateOptionsV0()
    requirements = plugin.requirements(_activation())

    assert plugin.manifest.idea_id == "m1c_quiet_state_options"
    assert plugin.manifest.modes == (RuntimeMode.PROSPECTIVE_RECORD, RuntimeMode.SHADOW)
    assert plugin.manifest.maximum_outputs_per_batch == 64
    assert plugin.manifest.maximum_interests_per_batch == 64
    assert len(requirements) == 60
    assert {item.instrument_id for item in requirements} == set(COHORT)
    assert {(item.feed_kind, item.event_kind, item.cadence) for item in requirements} == {
        ("bars", "bar_5m_session_prefix", "5s"),
        ("bars", "session_volume_baseline", "5s"),
        ("quotes", "quote", "stream"),
    }


def test_quiet_constants_remain_bound_to_the_frozen_research_contracts() -> None:
    root = Path("research/prospective/frozen-m1c-microstructure-recorder-v0")
    thresholds = json.loads((root / "quiet_state_threshold_manifest.json").read_text())
    option_selection = json.loads((root / "quiet_state_option_selection_contract.json").read_text())
    structures = json.loads((root / "defined_risk_structure_contract.json").read_text())

    assert thresholds["m1c_bottom_10_threshold"] == BOTTOM_10_THRESHOLD
    assert thresholds["primary_quiet_state"] == "bottom_10_percent"
    assert option_selection["maximum_contracts_per_quiet_observation"] == 54
    assert option_selection["maximum_neighbouring_strike_steps"] == 4
    assert option_selection["complete_chain_stream_allowed"] is False
    assert option_selection["capacity_reduced_plan_complete"] is False
    assert structures["naked_short_options_allowed"] is False
    assert structures["long_premium_structures"] == ["ATM_CALL", "ATM_PUT", "ATM_STRADDLE"]


def test_d1_snapshot_planning_uses_only_the_latest_open_window_and_never_backfills() -> None:
    def baseline(event_id: str, session: str, at_us: int) -> MarketEvent:
        return _event(
            event_id,
            "RGTI",
            "session_volume_baseline",
            at_us,
            {
                "session": session,
                "complete_session_count": 21,
                "session_closes": [39.5, 40.0],
                "realised_volatility_20d": 0.58,
            },
        )

    latest_open = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (
                baseline("baseline-1", "2026-08-05", 1_000_000),
                baseline("baseline-2", "2026-08-06", 2_000_000),
                baseline("baseline-3", "2026-08-07", 3_000_000),
                _event(
                    "heartbeat-open",
                    "RGTI",
                    "quote",
                    3_000_001,
                    {"bid": 39.9, "ask": 40.1},
                ),
            )
        ),
        {},
    )

    assert len(latest_open.interests) == 2
    assert {interest.interest_key for interest in latest_open.interests} == {
        "quiet:m1c:d1:2026-08-07:RGTI:call",
        "quiet:m1c:d1:2026-08-07:RGTI:put",
    }

    expired = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (
                baseline("baseline-expired", "2026-08-07", 1_000_000),
                _event(
                    "heartbeat-after-expiry",
                    "RGTI",
                    "quote",
                    1_801_000_001,
                    {"bid": 39.9, "ask": 40.1},
                ),
            )
        ),
        {},
    )

    assert expired.interests == ()


def test_quiet_crossing_waits_for_a_quote_then_requests_the_exact_54_contract_panel() -> None:
    plugin = M1CQuietStateOptionsV0()
    state, retained = _prime_low_tail_context(plugin)
    prefix = _event(
        "prefix-rgti-6",
        "RGTI",
        "bar_5m_session_prefix",
        2_000_000_000,
        _prefix_payload(symbol="RGTI", session="2026-08-10"),
    )

    triggered = plugin.evaluate(
        _ordinary_batch((prefix,), retained=retained),
        state,
    )

    assert triggered.interests == ()
    assert len(triggered.outputs) == 1
    assert triggered.outputs[0].kind == "observation"
    assert triggered.outputs[0].payload["status"] == "complete"
    assert triggered.outputs[0].payload["fresh_episode"] is True
    assert triggered.outputs[0].payload["bottom_10"] is True
    assert triggered.outputs[0].payload["probability"] == pytest.approx(0.08276432334461586)

    quote = _event(
        "quote-rgti-entry",
        "RGTI",
        "quote",
        prefix.event_at_us + 1,
        {"bid": 40.0, "ask": 40.2},
    )
    planned = plugin.evaluate(
        _ordinary_batch((quote,), retained=triggered.retained_input_event_ids),
        triggered.state,
    )

    assert len(planned.interests) == 54
    assert len({interest.interest_key for interest in planned.interests}) == 54
    assert {
        (
            interest.minimum_days_to_expiry,
            interest.maximum_days_to_expiry,
            interest.strike_offset,
            interest.option_right,
        )
        for interest in planned.interests
    } == {
        (minimum, maximum, offset, right)
        for minimum, maximum in ((0, 0), (1, 1), (3, 5))
        for offset in range(-4, 5)
        for right in ("call", "put")
    }
    assert all(interest.required for interest in planned.interests)
    assert all(interest.cadence == "snapshot" for interest in planned.interests)
    assert all(interest.reference_price == pytest.approx(40.1) for interest in planned.interests)
    assert all(interest.input_event_id == quote.event_id for interest in planned.interests)
    assert len(planned.state_json()) < 65_536


def test_exact_54_panel_resolves_and_plans_within_the_existing_market_data_bound() -> None:
    _plugin, planned, _prefix, _quote = _planned_entry_panel()
    as_of = int(datetime(2026, 8, 10, 14, tzinfo=UTC).timestamp() * 1_000_000)
    backend = _PanelDiscoveryBackend()
    resolver = InstrumentResolver(
        backend,
        underlyings={"RGTI": InstrumentSpec("RGTI", 1234, "stock", "RGTI", "SMART", "USD")},
        completed_at_us=lambda: as_of + 1,
    )
    resolved = tuple(
        resolver.resolve(
            InterestResolutionRequest(
                interest_id=f"panel-{index}",
                instance_id="quiet-1",
                interest=type(interest).model_validate(
                    {
                        **interest.model_dump(),
                        "as_of_at_us": as_of,
                        "expires_at_us": as_of + 3_600_000_000,
                    }
                ),
            )
        )
        for index, interest in enumerate(planned.interests)
    )

    assert all(receipt.status == "resolved" for receipt in resolved)
    assert backend.parameter_calls == 1
    assert len(backend.contract_calls) == 54
    assert {(receipt.expiry, receipt.strike, receipt.option_right) for receipt in resolved} == {
        (expiry, 40.0 + offset, right)
        for expiry in ("20260810", "20260811", "20260813")
        for offset in range(-4, 5)
        for right in ("call", "put")
    }
    demands = tuple(
        MarketDataDemand(
            source_id=receipt.interest_id,
            instrument_id=cast(str, receipt.instrument_id),
            feed_kind="quotes",
            required=True,
            priority=200,
            stale_after_us=60_000_000,
            snapshot=True,
        )
        for receipt in resolved
    )
    plan = plan_market_data((), demands, MarketDataCapacity(line_limit=54))
    assert plan.required_complete is True
    assert plan.line_count == 54
    assert plan.deferred_source_ids == ()


def test_complete_panel_emits_nine_long_observations_and_twelve_protected_attempts() -> None:
    plugin, planned, prefix, quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)

    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )

    observations = tuple(output for output in completed.outputs if output.kind == "observation")
    proposals = tuple(output for output in completed.outputs if output.kind == "proposed_trade")
    assert len(completed.outputs) == 21
    assert len(observations) == 9
    assert len(proposals) == 12
    assert {output.payload["structure_type"] for output in observations} == {
        "ATM_CALL",
        "ATM_PUT",
        "ATM_STRADDLE",
    }
    assert {output.payload["structure_type"] for output in proposals} == {
        "ATM_IRON_BUTTERFLY",
        "DELTA_IRON_CONDOR",
        "CALL_CREDIT_SPREAD",
        "PUT_CREDIT_SPREAD",
    }
    for proposal in proposals:
        assert proposal.status == "unapproved"
        assert proposal.payload["defined_risk"] is True
        assert proposal.payload["research_only"] is True
        assert proposal.payload["execution_enabled"] is False
        assert any(leg.action == "sell" and leg.target == "short" for leg in proposal.legs)
        assert any(leg.action == "buy" and leg.target == "long" for leg in proposal.legs)
        assert {leg.currency for leg in proposal.legs} == {"USD"}
    assert len(completed.interests) == 24
    assert all(interest.cadence == "stream" for interest in completed.interests)
    assert all(interest.required for interest in completed.interests)
    assert len({interest.interest_key for interest in completed.interests}) == 24
    assert len(completed.output_input_event_ids) == 21
    assert all(prefix.event_id in lineage for lineage in completed.output_input_event_ids)
    assert all(quote.event_id in lineage for lineage in completed.output_input_event_ids)
    assert all(len(lineage) == 59 for lineage in completed.output_input_event_ids)
    assert len(completed.state_json()) < 65_536
    assert len(completed.retained_input_event_ids) <= 256


def test_one_incomplete_capture_fails_the_whole_panel_closed() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    incomplete_key = planned.interests[17].interest_key
    receipts, captures = _entry_panel_evidence(planned, incomplete_key=incomplete_key)

    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )

    assert len(completed.outputs) == 1
    assert completed.outputs[0].kind == "observation"
    assert completed.outputs[0].payload["status"] == "incomplete"
    assert completed.outputs[0].payload["reason"] == "option_panel_capture_incomplete"
    assert completed.interests == ()
    assert all(output.kind != "proposed_trade" for output in completed.outputs)


def test_one_denied_required_entry_contract_fails_the_whole_panel_closed() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    denied_interest = planned.interests[0]
    denied = DiscoveryReceipt(
        receipt_id="entry-denied",
        interest_id="entry-interest-denied",
        interest_key=denied_interest.interest_key,
        instance_id="quiet-1",
        status="denied",
        reason_code="no_matching_expiry",
        candidates_inspected=20,
        completed_at_us=2_000_000_100,
    )
    heartbeat = _event(
        "quote-rgti-entry-denial",
        "RGTI",
        "quote",
        2_000_000_101,
        {"bid": 40.0, "ask": 40.2},
    )

    evaluated = plugin.evaluate(
        _ordinary_batch(
            (heartbeat,),
            retained=planned.retained_input_event_ids,
            receipts=(denied,),
        ),
        planned.state,
    )

    assert len(evaluated.outputs) == 1
    assert evaluated.outputs[0].payload["reason"] == "option_panel_discovery_denied"
    assert all(output.kind != "proposed_trade" for output in evaluated.outputs)
    assert evaluated.interests == ()


def test_partial_panel_resumes_on_a_fresh_plugin_instance_without_duplicate_interests() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    partial = plugin.evaluate(
        _ordinary_batch(
            captures[:27],
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )

    assert partial.outputs == ()
    assert partial.interests == ()
    assert len(partial.retained_input_event_ids) == 32
    assert len(partial.state_json()) < 65_536

    completed = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            captures[27:],
            retained=partial.retained_input_event_ids,
            receipts=receipts,
        ),
        partial.state,
    )

    assert len(completed.outputs) == 21
    assert len(completed.interests) == 24
    assert all(len(lineage) == 59 for lineage in completed.output_input_event_ids)


def test_all_twenty_simultaneous_crossings_service_one_and_record_the_other_nineteen() -> None:
    plugin = M1CQuietStateOptionsV0()
    state: JsonValue | None = None
    retained: tuple[str, ...] = ()
    for symbol in COHORT:
        state, retained = _prime_low_tail_context(
            plugin,
            symbol=symbol,
            state=state,
            retained=retained,
            all_cohort_low=True,
        )
    assert state is not None
    at_us = 2_000_000_000
    events = tuple(
        event
        for index, symbol in enumerate(COHORT)
        for event in (
            _event(
                f"prefix-{symbol.lower()}-6",
                symbol,
                "bar_5m_session_prefix",
                at_us + index * 2,
                _prefix_payload(symbol=symbol, session="2026-08-10"),
            ),
            _event(
                f"quote-{symbol.lower()}",
                symbol,
                "quote",
                at_us + index * 2 + 1,
                {"bid": 40.0, "ask": 40.2},
            ),
        )
    )
    candidate_batch = _ordinary_batch(events, retained=retained)
    selected_count = plugin.select_input_prefix(candidate_batch, state)
    assert selected_count == 39

    evaluated = plugin.evaluate(
        _ordinary_batch(events[:selected_count], retained=retained),
        state,
    )

    assert len(evaluated.interests) == 54
    assert {interest.underlying_instrument_id for interest in evaluated.interests} == {"RGTI"}
    incomplete = tuple(
        output
        for output in evaluated.outputs
        if output.payload.get("reason") == "complete_panel_capacity_unavailable"
    )
    assert len(incomplete) == 19
    assert {output.subject_instrument_id for output in incomplete} == set(COHORT) - {"RGTI"}
    assert len(evaluated.outputs) == 39
    receipts, captures = _entry_panel_evidence(evaluated)
    worst_case = plugin.evaluate(
        _ordinary_batch(
            captures[:53],
            retained=evaluated.retained_input_event_ids,
            receipts=receipts,
        ),
        evaluated.state,
    )
    assert worst_case.outputs == ()
    assert len(worst_case.state_json()) < 65_536
    assert len(worst_case.retained_input_event_ids) <= 256


def test_denied_selected_leg_stream_is_recorded_once_as_incomplete() -> None:
    plugin, planned, _prefix, _quote = _planned_entry_panel()
    receipts, captures = _entry_panel_evidence(planned)
    completed = plugin.evaluate(
        _ordinary_batch(
            captures,
            retained=planned.retained_input_event_ids,
            receipts=receipts,
        ),
        planned.state,
    )
    denied_interest = completed.interests[0]
    denied = DiscoveryReceipt(
        receipt_id="stream-denied",
        interest_id="stream-interest-denied",
        interest_key=denied_interest.interest_key,
        instance_id="quiet-1",
        status="denied",
        reason_code="line_capacity",
        candidates_inspected=0,
        completed_at_us=2_000_001_000,
    )
    heartbeat = _event(
        "quote-rgti-heartbeat",
        "RGTI",
        "quote",
        2_000_001_001,
        {"bid": 40.0, "ask": 40.2},
    )

    reported = plugin.evaluate(
        _ordinary_batch(
            (heartbeat,),
            retained=completed.retained_input_event_ids,
            receipts=(denied,),
        ),
        completed.state,
    )

    assert len(reported.outputs) == 1
    assert reported.outputs[0].kind == "observation"
    assert reported.outputs[0].payload["status"] == "incomplete"
    assert reported.outputs[0].payload["reason"] == "selected_leg_stream_denied"
    assert reported.outputs[0].payload["denial_reason_codes"] == ("line_capacity",)
    assert reported.interests == ()

    replayed = M1CQuietStateOptionsV0().evaluate(
        _ordinary_batch(
            (
                _event(
                    "quote-rgti-heartbeat-2",
                    "RGTI",
                    "quote",
                    2_000_001_002,
                    {"bid": 40.0, "ask": 40.2},
                ),
            ),
            retained=reported.retained_input_event_ids,
            receipts=(denied,),
        ),
        reported.state,
    )
    assert replayed.outputs == ()


def test_quiet_state_uses_the_frozen_inclusive_downward_crossing() -> None:
    exact = classify_quiet_state(
        probability=BOTTOM_10_THRESHOLD,
        previous_probability=BOTTOM_10_THRESHOLD + 0.01,
        minutes_since_previous_episode=None,
    )
    already_quiet = classify_quiet_state(
        probability=BOTTOM_10_THRESHOLD - 0.01,
        previous_probability=BOTTOM_10_THRESHOLD,
        minutes_since_previous_episode=45.0,
    )
    too_close = classify_quiet_state(
        probability=BOTTOM_10_THRESHOLD - 0.01,
        previous_probability=BOTTOM_10_THRESHOLD + 0.01,
        minutes_since_previous_episode=29.999,
    )

    assert exact.bottom_10 is True
    assert exact.fresh_episode is True
    assert already_quiet.fresh_episode is False
    assert too_close.fresh_episode is False


def test_defined_risk_selection_matches_the_frozen_protective_contracts() -> None:
    attempts = select_defined_risk_structures(
        contracts=_bucket_contracts(),
        underlying_reference_price=100.0,
    )

    assert tuple(attempt.structure_type for attempt in attempts) == (
        "ATM_IRON_BUTTERFLY",
        "DELTA_IRON_CONDOR",
        "CALL_CREDIT_SPREAD",
        "PUT_CREDIT_SPREAD",
    )
    assert all(attempt.available for attempt in attempts)
    assert tuple(
        (leg.side, leg.contract.offset, leg.contract.right) for leg in attempts[0].legs
    ) == (
        ("short", 0, "call"),
        ("short", 0, "put"),
        ("long", 1, "call"),
        ("long", -1, "put"),
    )
    assert tuple(
        (leg.side, leg.contract.offset, leg.contract.right) for leg in attempts[1].legs
    ) == (
        ("short", 2, "call"),
        ("short", -2, "put"),
        ("long", 3, "call"),
        ("long", -3, "put"),
    )
    assert all(
        any(leg.side == "short" for leg in attempt.legs)
        and any(leg.side == "long" for leg in attempt.legs)
        for attempt in attempts
    )


def test_short_premium_selection_never_emits_an_unprotected_attempt() -> None:
    contracts = tuple(contract for contract in _bucket_contracts() if contract.offset == 0)

    attempts = select_defined_risk_structures(
        contracts=contracts,
        underlying_reference_price=100.0,
    )

    assert all(not attempt.available for attempt in attempts)
    assert all(attempt.legs == () for attempt in attempts)
    assert {attempt.reason for attempt in attempts} == {
        "symmetric_wings_unavailable",
        "delta_tolerance_failed",
        "protective_wing_unavailable",
    }


def test_non_credit_entry_never_becomes_a_short_premium_proposal_attempt() -> None:
    contracts = tuple(replace(contract, bid=0.0, ask=1.0) for contract in _bucket_contracts())

    attempts = select_defined_risk_structures(
        contracts=contracts,
        underlying_reference_price=100.0,
    )

    assert all(not attempt.available for attempt in attempts)
    assert {attempt.reason for attempt in attempts} == {"non_positive_opening_credit"}
