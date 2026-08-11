from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import pytest

from stocker_ideas.plugins import m1c_opening_reversal_v1_1 as opening_reversal
from stocker_ideas.plugins.m1c_opening_reversal_v1_1 import (
    COHORT,
    MANIFEST,
    NEGATIVE_OPENING_RETURN_THRESHOLD,
    OPENING_RANGE_THRESHOLD,
    PARAMETERS,
    POSITIVE_OPENING_RETURN_THRESHOLD,
    UNIVERSE,
    M1COpeningReversalV1_1,
    create_plugin,
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
from stocker_runtime.ideas.discovery import IdeaConfig, discover_plugins, reviewed_code_hash


def _activation() -> IdeaActivation:
    return IdeaActivation(
        instance_id="opening-reversal-1",
        parameters=PARAMETERS,
        parameters_hash=hashlib.sha256(canonical_json_bytes(PARAMETERS)).hexdigest(),
        plugin_code_hash="a" * 64,
        activated_at_us=1,
        run_id="run-1",
        protected_data_class=ProtectedDataClass.SHADOW,
        universe=UNIVERSE,
    )


def _batch(
    events: tuple[MarketEvent, ...],
    *,
    prior_ids: tuple[str, ...] = (),
    receipts: tuple[DiscoveryReceipt, ...] = (),
) -> IdeaBatch:
    return IdeaBatch(
        mode=RuntimeMode.SHADOW,
        events=events,
        input_watermark=events[-1].event_id,
        causal_from_at_us=min(event.event_at_us for event in events),
        causal_through_at_us=max(max(event.event_at_us, event.received_at_us) for event in events),
        prior_state_input_event_ids=prior_ids,
        discovery_receipts=receipts,
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
        feed_kind=("quotes" if event_kind in {"option_snapshot_capture", "quote"} else "bars"),
        event_kind=event_kind,
        event_at_us=at_us,
        received_at_us=at_us,
        payload=payload,
    )


def _received(event: MarketEvent, received_at_us: int) -> MarketEvent:
    return MarketEvent(
        event_id=event.event_id,
        instrument_id=event.instrument_id,
        feed_kind=event.feed_kind,
        event_kind=event.event_kind,
        event_at_us=event.event_at_us,
        received_at_us=received_at_us,
        payload=event.payload,
    )


def _baseline(symbol: str, index: int) -> MarketEvent:
    return _event(
        f"baseline-{symbol}",
        symbol,
        "session_volume_baseline",
        1_000_000 + index,
        {
            "schema_version": 1,
            "session": "2026-08-07",
            "complete_session_count": 21,
            "session_closes": [100.0],
            "realised_volatility_20d": 0.30,
        },
    )


def _capture(
    symbol: str,
    right: str,
    index: int,
    *,
    completed_at_us: int | None = None,
) -> tuple[MarketEvent, DiscoveryReceipt]:
    instrument_id = f"option-{symbol}-{right}"
    interest_key = f"opening-reversal:m1c:d1:2026-08-07:{symbol}:{right}"
    receipt = DiscoveryReceipt(
        receipt_id=f"receipt-{symbol}-{right}",
        interest_id=f"interest-{symbol}-{right}",
        interest_key=interest_key,
        instance_id="opening-reversal-1",
        status="resolved",
        instrument_id=instrument_id,
        expiry="20260918",
        strike=100.0,
        option_right=cast(str, right),
        multiplier="100",
        candidates_inspected=2,
        completed_at_us=(2_000_000 + index if completed_at_us is None else completed_at_us),
    )
    event = _event(
        f"capture-{symbol}-{right}",
        instrument_id,
        "option_snapshot_capture",
        3_000_000 + index,
        {
            "schema_version": 1,
            "source_completeness": "complete",
            "bid": 4.0 if right == "call" else 3.8,
            "ask": 4.2 if right == "call" else 4.0,
            "model_implied_volatility": 0.45 if right == "call" else 0.46,
            "open_interest": 100,
        },
    )
    return event, receipt


def _prefix_payload(
    *,
    session: str,
    checkpoint: int = 6,
    base: float = 100.0,
) -> dict[str, JsonValue]:
    trailing: list[dict[str, JsonValue]] = []
    for number in range(1, checkpoint + 1):
        opening = base + number / 10.0
        trailing.append(
            {
                "bar_number": number,
                "event_at_us": number * 1_000_000,
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
    }


def _checkpoint_event(symbol: str, index: int) -> MarketEvent:
    payload = _prefix_payload(
        session="2026-08-10",
        base=250.0 if symbol == "VTI" else 100.0,
    )
    if symbol == "VTI":
        accumulator = cast(dict[str, JsonValue], payload["accumulator"])
        accumulator.update(
            {
                "session_open": 250.0,
                "session_high": 251.0,
                "session_low": 248.0,
                "last_close": 249.0,
            }
        )
    return _event(
        f"prefix-{symbol}-6",
        symbol,
        "bar_5m_session_prefix",
        2_000_000_000 + index,
        payload,
    )


def _advance(
    plugin: M1COpeningReversalV1_1,
    state: JsonValue,
    prior_ids: tuple[str, ...],
    events: tuple[MarketEvent, ...],
    *,
    receipts: tuple[DiscoveryReceipt, ...] = (),
) -> IdeaEvaluation:
    evaluation = plugin.evaluate(
        _batch(events, prior_ids=prior_ids, receipts=receipts),
        state,
    )
    return evaluation


def _pending_primary_episode(
    *,
    final_vti_received_at_us: int | None = None,
    checkpoint_received_at_us: Mapping[str, int] | None = None,
    checkpoint_order: tuple[str, ...] | None = None,
) -> tuple[M1COpeningReversalV1_1, IdeaEvaluation]:
    plugin = create_plugin()
    evaluation = _advance(
        plugin,
        {},
        (),
        tuple(_baseline(symbol, index) for index, symbol in enumerate(COHORT)),
    )
    capture_pairs = tuple(
        _capture(symbol, right, index * 2 + right_index)
        for index, symbol in enumerate(COHORT)
        for right_index, right in enumerate(("call", "put"))
    )
    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        tuple(pair[0] for pair in capture_pairs),
        receipts=tuple(pair[1] for pair in capture_pairs),
    )
    for symbol in checkpoint_order or UNIVERSE:
        index = UNIVERSE.index(symbol)
        checkpoint_event = _checkpoint_event(symbol, index)
        if symbol == "VTI" and final_vti_received_at_us is not None:
            checkpoint_event = _received(checkpoint_event, final_vti_received_at_us)
        if checkpoint_received_at_us is not None and symbol in checkpoint_received_at_us:
            checkpoint_event = _received(
                checkpoint_event,
                checkpoint_received_at_us[symbol],
            )
        evaluation = _advance(
            plugin,
            evaluation.state,
            evaluation.retained_input_event_ids,
            (checkpoint_event,),
        )
    return plugin, evaluation


def _active_primary_episode() -> tuple[M1COpeningReversalV1_1, IdeaEvaluation]:
    plugin, evaluation = _pending_primary_episode()
    quote = _event(
        "quote-AAL",
        "AAL",
        "quote",
        2_000_001_000,
        {"bid": 100.7, "ask": 100.9},
    )
    return plugin, _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (quote,),
    )


def _primary_receipt(
    right: Literal["call", "put"],
    *,
    completed_at_us: int,
    instrument_id: str | None = None,
    strike: float = 101.0,
    status: Literal["resolved", "denied"] = "resolved",
) -> DiscoveryReceipt:
    resolved_instrument = instrument_id or f"primary-{right}"
    return DiscoveryReceipt(
        receipt_id=f"primary-receipt-{right}-{completed_at_us}",
        interest_id=f"primary-interest-{right}",
        interest_key=f"opening-reversal:primary:2026-08-10:AAL:{right}",
        instance_id="opening-reversal-1",
        status=status,
        reason_code="OPTION_PERMISSION_DENIED" if status == "denied" else None,
        instrument_id=resolved_instrument if status == "resolved" else None,
        expiry="20260811" if status == "resolved" else None,
        strike=strike if status == "resolved" else None,
        option_right=right if status == "resolved" else None,
        multiplier="100" if status == "resolved" else None,
        candidates_inspected=2,
        completed_at_us=completed_at_us,
    )


def test_opening_reversal_declares_only_existing_generic_market_data_seams() -> None:
    plugin = create_plugin()

    assert plugin.manifest == MANIFEST
    assert plugin.manifest.idea_id == "m1c_opening_reversal"
    assert plugin.manifest.idea_version == "v1_1"
    assert plugin.manifest.modes == (
        RuntimeMode.PROSPECTIVE_RECORD,
        RuntimeMode.SHADOW,
    )
    assert tuple(kind.value for kind in plugin.manifest.output_kinds) == (
        "observation",
        "signal",
    )
    assert plugin.manifest.maximum_outputs_per_batch == 21
    assert plugin.manifest.maximum_interests_per_batch == 42

    requirements = plugin.requirements(_activation())

    assert len(COHORT) == 20
    assert (*COHORT, "VTI") == UNIVERSE
    assert len(requirements) == 61
    assert sum(item.event_kind == "bar_5m_session_prefix" for item in requirements) == 21
    assert sum(item.event_kind == "session_volume_baseline" for item in requirements) == 20
    assert sum(item.event_kind == "quote" for item in requirements) == 20
    assert sum(item.feed_kind == "bars" for item in requirements) == 41
    assert sum(item.feed_kind == "quotes" for item in requirements) == 20
    assert all(item.gaps_block and item.staleness_block for item in requirements)


def test_opening_reversal_frozen_artifacts_and_isolated_discovery_are_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts = {
        "research/prospective/20260729-m1c-prospective-opening-reversal-v1/"
        "artifacts/primary/frozen_rule_manifest_v1.json": (
            "887e1019b8706c7caca019241247477162dbae0298d43e2376bc0e1cc63cf5e5"
        ),
        "research/prospective/20260729-m1c-prospective-opening-reversal-v1/"
        "artifacts/primary/frozen_experiment_configuration_v1.json": (
            "3422e52a7601dd1eaf40cfe10118bb614d288a80b945604d083e92cb32ca8108"
        ),
        "research/prospective/20260729-m1c-prospective-opening-reversal-v1-1/"
        "artifacts/primary/frozen_rule_manifest_v1_1.json": (
            "1f69e2b47cbcecae9ec74d911c4d394d6d83e966b6fa36c122b253725f6fd18e"
        ),
        "research/prospective/20260729-m1c-prospective-opening-reversal-v1-1/"
        "artifacts/primary/frozen_timing_addendum_configuration_v1_1.json": (
            "1f4770ca05e578cecd5c9bbd0c9f942017a0ab45cad174f9a3212e281eaa5c37"
        ),
        "research/prospective/20260729-m1c-prospective-opening-reversal-v1-1/"
        "artifacts/primary/artifact_manifest_v1_1.json": (
            "473cb631d2bfeea36f338108b26841f63d2032c7360bb9d4f732bcc2cc7eb04b"
        ),
    }

    assert {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in artifacts
    } == artifacts

    module = "stocker_ideas.plugins.m1c_opening_reversal_v1_1"
    discovered = discover_plugins(
        (
            IdeaConfig(
                module=module,
                expected_code_hash=reviewed_code_hash(module),
                expected_manifest_hash=hashlib.sha256(MANIFEST.to_canonical_json()).hexdigest(),
                parameters=PARAMETERS,
                universe=UNIVERSE,
                enabled=True,
            ),
        )
    )

    assert len(discovered) == 1
    assert discovered[0].manifest.idea_id == "m1c_opening_reversal"
    assert len(discovered[0].requirements) == 61


@pytest.mark.parametrize(
    ("close", "expected_state", "expected_sign"),
    (
        (
            math.nextafter(math.exp(NEGATIVE_OPENING_RETURN_THRESHOLD), 0.0),
            "NEGATIVE_SEVERE_OPENING_TRANSITION",
            -1,
        ),
        (
            math.nextafter(math.exp(POSITIVE_OPENING_RETURN_THRESHOLD), math.inf),
            "POSITIVE_SEVERE_OPENING_TRANSITION",
            1,
        ),
        (
            math.nextafter(math.exp(NEGATIVE_OPENING_RETURN_THRESHOLD), math.inf),
            "ELEVATED_OPENING_RANGE_NONDIRECTIONAL",
            None,
        ),
        (
            math.nextafter(math.exp(POSITIVE_OPENING_RETURN_THRESHOLD), 0.0),
            "ELEVATED_OPENING_RANGE_NONDIRECTIONAL",
            None,
        ),
    ),
)
def test_opening_transition_uses_inclusive_frozen_threshold_edges(
    close: float,
    expected_state: str,
    expected_sign: int | None,
) -> None:
    payload = _prefix_payload(session="2026-08-10", base=1.0)
    accumulator = cast(dict[str, JsonValue], payload["accumulator"])
    accumulator.update(
        {
            "session_open": 1.0,
            "session_low": 1.0,
            "session_high": math.nextafter(math.exp(OPENING_RANGE_THRESHOLD), math.inf),
            "last_close": close,
        }
    )
    event = _event(
        f"edge-{expected_state}-{close}",
        "VTI",
        "bar_5m_session_prefix",
        2_000_000_000,
        payload,
    )

    evaluation = create_plugin().evaluate(_batch((event,)), {})
    state = cast(Mapping[str, JsonValue], evaluation.state)
    cohort = cast(Mapping[str, JsonValue], state["cohort"])
    market = cast(Mapping[str, JsonValue], cohort["market"])

    assert market["state"] == expected_state
    assert market["sign"] == expected_sign


def test_checkpoint_six_negative_transition_promotes_one_call_episode() -> None:
    plugin = create_plugin()
    state: JsonValue = {}
    prior_ids: tuple[str, ...] = ()
    baselines = tuple(_baseline(symbol, index) for index, symbol in enumerate(COHORT))

    evaluation = _advance(plugin, state, prior_ids, baselines)

    assert len(evaluation.interests) == 40
    assert {interest.cadence for interest in evaluation.interests} == {"snapshot"}
    assert {interest.option_right for interest in evaluation.interests} == {"call", "put"}
    state = evaluation.state
    prior_ids = evaluation.retained_input_event_ids

    capture_pairs = tuple(
        _capture(symbol, right, index * 2 + right_index)
        for index, symbol in enumerate(COHORT)
        for right_index, right in enumerate(("call", "put"))
    )
    capture_events = tuple(pair[0] for pair in capture_pairs)
    receipts = tuple(pair[1] for pair in capture_pairs)
    evaluation = _advance(
        plugin,
        state,
        prior_ids,
        capture_events,
        receipts=receipts,
    )
    state = evaluation.state
    prior_ids = evaluation.retained_input_event_ids

    combined_prefixes = tuple(
        _checkpoint_event(symbol, index) for index, symbol in enumerate(UNIVERSE)
    )
    assert (
        plugin.select_input_prefix(
            _batch(combined_prefixes, prior_ids=prior_ids),
            state,
        )
        == 1
    )

    for index, symbol in enumerate(UNIVERSE):
        evaluation = _advance(
            plugin,
            state,
            prior_ids,
            (_checkpoint_event(symbol, index),),
        )
        state = evaluation.state
        prior_ids = evaluation.retained_input_event_ids
        if index < len(UNIVERSE) - 1:
            assert not evaluation.outputs
            assert not evaluation.interests

    observations = tuple(output for output in evaluation.outputs if output.kind == "observation")
    signals = tuple(output for output in evaluation.outputs if output.kind == "signal")
    by_symbol = {output.subject_instrument_id: output for output in observations}

    assert len(observations) == 20
    assert len(signals) == 1
    assert signals[0].subject_instrument_id == "AAL"
    assert signals[0].payload["action"] == "CALL"
    assert signals[0].payload["opening_transition_state"] == ("NEGATIVE_SEVERE_OPENING_TRANSITION")
    assert signals[0].payload["probability"] == pytest.approx(0.776658525, abs=1e-9)
    assert by_symbol["APLD"].payload["action"] == "ABSTAIN"
    assert by_symbol["APLD"].payload["reason"] == "m1c_below_frozen_high_tail"
    assert all("transfer_status" not in output.payload for output in observations)
    assert all("provider" not in output.payload for output in observations)
    cohort_prefix_ids = {f"prefix-{symbol}-6" for symbol in UNIVERSE}
    assert all(
        cohort_prefix_ids.issubset(evaluation.output_input_event_ids[index])
        for index, output in enumerate(evaluation.outputs)
        if output.kind == "observation"
    )

    assert not evaluation.interests

    quote = _event(
        "quote-AAL",
        "AAL",
        "quote",
        2_000_001_000,
        {"bid": 100.7, "ask": 100.9},
    )
    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (quote,),
    )

    assert len(evaluation.interests) == 2
    assert {interest.option_right for interest in evaluation.interests} == {"call", "put"}
    assert {interest.cadence for interest in evaluation.interests} == {"stream"}
    assert {interest.underlying_instrument_id for interest in evaluation.interests} == {"AAL"}
    assert {interest.minimum_days_to_expiry for interest in evaluation.interests} == {1}
    assert {interest.maximum_days_to_expiry for interest in evaluation.interests} == {1}
    assert {interest.strike_offset for interest in evaluation.interests} == {0}
    assert {interest.input_event_id for interest in evaluation.interests} == {quote.event_id}
    assert {interest.as_of_at_us for interest in evaluation.interests} == {quote.event_at_us}
    assert {interest.expires_at_us for interest in evaluation.interests} == {
        2_000_000_020 + 15 * 60 * 1_000_000
    }


@pytest.mark.parametrize("barrier_lag_us", (0, 1))
def test_barrier_at_or_after_primary_cutoff_terminalizes_inside_the_signal(
    barrier_lag_us: int,
) -> None:
    cutoff_at_us = 2_000_000_020 + 15 * 60 * 1_000_000
    plugin, evaluation = _pending_primary_episode(
        final_vti_received_at_us=cutoff_at_us + barrier_lag_us
    )
    observations = tuple(output for output in evaluation.outputs if output.kind == "observation")
    signals = tuple(output for output in evaluation.outputs if output.kind == "signal")

    assert len(observations) == 20
    assert len(signals) == 1
    assert len(evaluation.outputs) == MANIFEST.maximum_outputs_per_batch
    assert not evaluation.interests
    signal = signals[0]
    primary = cast(Mapping[str, JsonValue], signal.payload["primary_pair"])
    assert signal.payload["action"] == "CALL"
    assert primary == {
        "status": "unavailable",
        "terminal_basis": "barrier_released_at_or_after_primary_cutoff",
        "cutoff_at_us": cutoff_at_us,
        "barrier_available_at_us": cutoff_at_us + barrier_lag_us,
        "terminal_event_id": "prefix-VTI-6",
        "expected_interest_keys": (
            "opening-reversal:primary:2026-08-10:AAL:call",
            "opening-reversal:primary:2026-08-10:AAL:put",
        ),
        "interests_issued": False,
    }
    signal_index = next(
        index for index, output in enumerate(evaluation.outputs) if output.kind == "signal"
    )
    assert primary["terminal_event_id"] in evaluation.output_input_event_ids[signal_index]
    state = cast(Mapping[str, JsonValue], evaluation.state)
    assert state["pending"] == {}
    assert cast(Mapping[str, JsonValue], state["primary_terminal"])["status"] == ("unavailable")

    late_quote = _event(
        "late-underlying-after-terminal",
        "AAL",
        "quote",
        cutoff_at_us + barrier_lag_us + 1,
        {"bid": 100.7, "ask": 100.9},
    )
    restarted = _advance(
        create_plugin(),
        evaluation.state,
        evaluation.retained_input_event_ids,
        (late_quote,),
    )

    assert not restarted.outputs
    assert not restarted.interests
    restarted_state = cast(Mapping[str, JsonValue], restarted.state)
    assert restarted_state["pending"] == {}
    assert restarted_state["active"] == {}
    assert restarted_state["primary_terminal"] == state["primary_terminal"]
    assert cast(Mapping[str, JsonValue], restarted_state["causal_horizon"])["i"] == (
        late_quote.event_id
    )


@pytest.mark.parametrize(
    ("aal_received_at_us", "aaoi_received_at_us", "expected_winner"),
    (
        (2_000_000_100, 2_000_000_001, "AAOI"),
        (2_000_000_100, 2_000_000_100, "AAL"),
    ),
)
def test_equal_probability_promotion_uses_receipt_time_then_ticker(
    monkeypatch: pytest.MonkeyPatch,
    aal_received_at_us: int,
    aaoi_received_at_us: int,
    expected_winner: str,
) -> None:
    def frozen_equal_score(
        *,
        symbol: str,
        checkpoint: int,
        group_o: Mapping[str, object],
        group_i: Mapping[str, object],
    ) -> dict[str, JsonValue]:
        assert checkpoint == 6
        assert group_o
        assert group_i
        return {
            "probability": 0.6 if symbol in {"AAL", "AAOI"} else 0.1,
            "feature_hash": "f" * 64,
            "model_hash": "m" * 64,
            "missing_feature_count": 0,
        }

    monkeypatch.setattr(opening_reversal, "score_m1c", frozen_equal_score)
    _, evaluation = _pending_primary_episode(
        checkpoint_received_at_us={
            "AAL": aal_received_at_us,
            "AAOI": aaoi_received_at_us,
        },
        checkpoint_order=("AAOI", "AAL", *UNIVERSE[2:]),
    )
    signal = next(output for output in evaluation.outputs if output.kind == "signal")

    assert signal.subject_instrument_id == expected_winner
    assert signal.payload["candidate_count"] == 2
    assert signal.payload["selection_rule"] == ("m1c_probability_desc_receipt_time_asc_ticker_asc")


def test_d1_denial_is_auditable_and_cannot_promote_that_stock() -> None:
    plugin = create_plugin()
    evaluation = _advance(
        plugin,
        {},
        (),
        tuple(_baseline(symbol, index) for index, symbol in enumerate(COHORT)),
    )
    pairs = tuple(
        _capture(symbol, right, index * 2 + right_index)
        for index, symbol in enumerate(COHORT)
        for right_index, right in enumerate(("call", "put"))
        if (symbol, right) != ("AAL", "call")
    )
    denial = DiscoveryReceipt(
        receipt_id="denied-AAL-call",
        interest_id="interest-AAL-call",
        interest_key="opening-reversal:m1c:d1:2026-08-07:AAL:call",
        instance_id="opening-reversal-1",
        status="denied",
        reason_code="OPTION_PERMISSION_DENIED",
        candidates_inspected=0,
        completed_at_us=2_000_000,
    )
    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        tuple(pair[0] for pair in pairs),
        receipts=(*tuple(pair[1] for pair in pairs), denial),
    )
    for index, symbol in enumerate(UNIVERSE):
        evaluation = _advance(
            plugin,
            evaluation.state,
            evaluation.retained_input_event_ids,
            (_checkpoint_event(symbol, index),),
        )

    aal = next(
        output
        for output in evaluation.outputs
        if output.kind == "observation" and output.subject_instrument_id == "AAL"
    )
    terminal = cast(Mapping[str, JsonValue], aal.payload["d1_terminal_evidence"])

    assert aal.payload["status"] == "unavailable"
    assert aal.payload["action"] == "ABSTAIN"
    assert aal.payload["reason"] == "prior_session_option_pair_incomplete"
    assert terminal["terminal_basis"] == "mixed_terminal_option_evidence"
    assert terminal["statuses"] == {"call": "denied", "put": "captured"}
    assert terminal["denial_reasons"] == {"call": "OPTION_PERMISSION_DENIED"}
    assert cast(Mapping[str, JsonValue], terminal["interest_keys"])["call"] == (denial.interest_key)
    lineage = evaluation.output_input_event_ids[
        next(index for index, output in enumerate(evaluation.outputs) if output is aal)
    ]
    assert "baseline-AAL" in lineage
    assert "capture-AAL-put" in lineage
    assert "prefix-AAL-6" in lineage


@pytest.mark.parametrize("delivery_lag_us", (30 * 60 * 1_000_000, 31 * 60 * 1_000_000))
def test_delayed_d1_baseline_never_opens_an_expired_snapshot_window(
    delivery_lag_us: int,
) -> None:
    plugin = create_plugin()
    original = _baseline("AAL", 0)
    delayed = _received(original, original.event_at_us + delivery_lag_us)

    evaluation = plugin.evaluate(_batch((delayed,)), {})

    assert not evaluation.interests
    state = cast(Mapping[str, JsonValue], evaluation.state)
    assert cast(Mapping[str, JsonValue], state["requested"]) == {}

    restarted = create_plugin().evaluate(
        _batch((delayed,), prior_ids=evaluation.retained_input_event_ids),
        evaluation.state,
    )

    assert not restarted.interests
    assert restarted.state == evaluation.state


def test_d1_capture_after_a_source_order_cutoff_is_late_even_if_backdated() -> None:
    plugin = create_plugin()
    baseline = _baseline("AAL", 0)
    initial = plugin.evaluate(_batch((baseline,)), {})
    capture, receipt = _capture("AAL", "call", 0)
    cutoff_at_us = baseline.event_at_us + 30 * 60 * 1_000_000
    cutoff = _event(
        "d1-cutoff-first",
        "VTI",
        "quote",
        cutoff_at_us,
        {"bid": 249.0, "ask": 249.2},
    )
    batch = _batch(
        (cutoff, capture),
        prior_ids=initial.retained_input_event_ids,
        receipts=(receipt,),
    )

    assert plugin.select_input_prefix(batch, initial.state) == 2

    evaluation = plugin.evaluate(batch, initial.state)
    state = cast(Mapping[str, JsonValue], evaluation.state)
    d1_terminal = cast(Mapping[str, Mapping[str, JsonValue]], state["d1_terminal"])

    assert d1_terminal["AAL"]["call"] == "late"
    assert d1_terminal["AAL"]["call_i"] == capture.event_id
    assert d1_terminal["AAL"]["call_cutoff_i"] == cutoff.event_id
    assert {capture.event_id, cutoff.event_id}.issubset(evaluation.retained_input_event_ids)

    cutoff_only = create_plugin().evaluate(
        _batch((cutoff,), prior_ids=initial.retained_input_event_ids),
        initial.state,
    )
    cutoff_state = cast(Mapping[str, JsonValue], cutoff_only.state)
    cutoff_terminal = cast(Mapping[str, Mapping[str, JsonValue]], cutoff_state["d1_terminal"])
    assert cutoff_terminal["AAL"]["call"] == "window_elapsed"
    assert cutoff_terminal["AAL"]["call_cutoff_i"] == cutoff.event_id
    assert cutoff.event_id in cutoff_only.retained_input_event_ids

    split = create_plugin().evaluate(
        _batch(
            (capture,),
            prior_ids=cutoff_only.retained_input_event_ids,
            receipts=(receipt,),
        ),
        cutoff_only.state,
    )
    split_state = cast(Mapping[str, JsonValue], split.state)
    split_terminal = cast(Mapping[str, Mapping[str, JsonValue]], split_state["d1_terminal"])
    assert split_terminal["AAL"]["call"] == "late"
    assert split_terminal["AAL"]["call_i"] == capture.event_id
    assert split_terminal["AAL"]["call_cutoff_i"] == cutoff.event_id
    assert split.state == evaluation.state
    assert split.retained_input_event_ids == evaluation.retained_input_event_ids

    restarted = create_plugin().evaluate(
        _batch(
            (_checkpoint_event("AAL", 0),),
            prior_ids=split.retained_input_event_ids,
        ),
        split.state,
    )
    restarted_state = cast(Mapping[str, JsonValue], restarted.state)
    cohort = cast(Mapping[str, JsonValue], restarted_state["cohort"])
    stocks = cast(Mapping[str, Mapping[str, JsonValue]], cohort["stocks"])
    selected = cast(tuple[str, ...], stocks["AAL"]["l"])

    assert {capture.event_id, cutoff.event_id}.issubset(selected)
    terminal = cast(Mapping[str, JsonValue], stocks["AAL"]["terminal"])
    assert terminal["terminal_basis"] == "mixed_terminal_option_evidence"
    assert cast(Mapping[str, JsonValue], terminal["cutoff_event_ids"])["call"] == (cutoff.event_id)


def test_cutoff_before_backdated_final_cohort_never_reopens_primary_window() -> None:
    _, before_vti = _pending_primary_episode(checkpoint_order=COHORT)
    vti = _checkpoint_event("VTI", UNIVERSE.index("VTI"))
    cutoff_at_us = vti.event_at_us + 15 * 60 * 1_000_000
    cutoff = _event(
        "primary-cutoff-before-final-cohort",
        "AAL",
        "quote",
        cutoff_at_us,
        {"bid": 100.0, "ask": 100.2},
    )

    combined = create_plugin().evaluate(
        _batch(
            (cutoff, vti),
            prior_ids=before_vti.retained_input_event_ids,
        ),
        before_vti.state,
    )
    cutoff_only = create_plugin().evaluate(
        _batch((cutoff,), prior_ids=before_vti.retained_input_event_ids),
        before_vti.state,
    )
    split = create_plugin().evaluate(
        _batch((vti,), prior_ids=cutoff_only.retained_input_event_ids),
        cutoff_only.state,
    )

    for evaluation in (combined, split):
        signal = next(output for output in evaluation.outputs if output.kind == "signal")
        primary = cast(Mapping[str, JsonValue], signal.payload["primary_pair"])
        state = cast(Mapping[str, JsonValue], evaluation.state)
        assert primary["status"] == "unavailable"
        assert primary["terminal_basis"] == ("barrier_released_at_or_after_primary_cutoff")
        assert primary["terminal_event_id"] == cutoff.event_id
        assert not primary["interests_issued"]
        assert state["pending"] == {}
        assert not evaluation.interests
        signal_index = next(
            index for index, output in enumerate(evaluation.outputs) if output.kind == "signal"
        )
        assert cutoff.event_id in evaluation.output_input_event_ids[signal_index]

    assert split.outputs == combined.outputs
    assert split.output_input_event_ids == combined.output_input_event_ids
    assert split.state == combined.state
    assert split.retained_input_event_ids == combined.retained_input_event_ids


def test_capture_resolved_after_d1_cutoff_remains_retained_late_evidence() -> None:
    baseline = _baseline("AAL", 0)
    initial = create_plugin().evaluate(_batch((baseline,)), {})
    cutoff_at_us = baseline.event_at_us + 30 * 60 * 1_000_000
    late_at_us = cutoff_at_us + 1
    capture, receipt = _capture("AAL", "call", 0, completed_at_us=late_at_us)
    capture = _received(capture, late_at_us)

    evaluation = create_plugin().evaluate(
        _batch(
            (capture,),
            prior_ids=initial.retained_input_event_ids,
            receipts=(receipt,),
        ),
        initial.state,
    )
    state = cast(Mapping[str, JsonValue], evaluation.state)
    terminal = cast(Mapping[str, Mapping[str, JsonValue]], state["d1_terminal"])["AAL"]

    assert terminal["call"] == "late"
    assert terminal["call_i"] == capture.event_id
    assert terminal["call_cutoff_i"] == capture.event_id
    assert capture.event_id in evaluation.retained_input_event_ids


def test_primary_pair_requires_coherent_one_dte_identity_and_in_window_quotes() -> None:
    plugin, evaluation = _active_primary_episode()
    call_receipt = _primary_receipt("call", completed_at_us=2_100_000_000)
    put_receipt = _primary_receipt("put", completed_at_us=2_100_000_001)
    call_quote = _event(
        "primary-call-quote",
        "primary-call",
        "quote",
        2_100_000_010,
        {"bid": 4.0, "ask": 4.2},
    )
    put_quote = _event(
        "primary-put-quote",
        "primary-put",
        "quote",
        2_100_000_011,
        {"bid": 3.8, "ask": 4.0},
    )
    cutoff_at_us = 2_000_000_020 + 15 * 60 * 1_000_000
    cutoff = _event(
        "primary-cutoff",
        "VTI",
        "quote",
        cutoff_at_us,
        {"bid": 249.0, "ask": 249.2},
    )
    combined = _batch(
        (call_quote, put_quote, cutoff),
        prior_ids=evaluation.retained_input_event_ids,
        receipts=(call_receipt, put_receipt),
    )

    assert plugin.select_input_prefix(combined, evaluation.state) == 1

    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (call_quote,),
        receipts=(call_receipt, put_receipt),
    )
    plugin = create_plugin()
    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (put_quote,),
    )
    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (cutoff,),
    )

    assert not evaluation.interests
    assert len(evaluation.outputs) == 1
    output = evaluation.outputs[0]
    assert output.kind == "observation"
    assert output.subject_instrument_id == "AAL"
    assert output.payload["status"] == "complete"
    assert output.payload["reason"] is None
    assert output.payload["action"] == "CALL"
    assert output.payload["pair_expiry"] == "20260811"
    assert output.payload["pair_strike"] == 101.0
    assert output.payload["quote_proof_count"] == 2
    assert output.payload["cutoff_crossing_event_id"] == cutoff.event_id
    assert output.payload["primary_pair_quote_event_ids"] == {
        call_receipt.interest_key: call_quote.event_id,
        put_receipt.interest_key: put_quote.event_id,
    }
    assert {
        call_quote.event_id,
        put_quote.event_id,
        cutoff.event_id,
    }.issubset(evaluation.output_input_event_ids[0])
    assert cast(Mapping[str, JsonValue], evaluation.state)["active"] == {}
    assert len(evaluation.retained_input_event_ids) < 256
    assert len(evaluation.state_json()) < 65_536


def test_primary_pair_denial_fails_closed_without_changing_the_signal() -> None:
    plugin, evaluation = _active_primary_episode()
    denial = _primary_receipt(
        "call",
        completed_at_us=2_100_000_000,
        status="denied",
    )
    terminal = _event(
        "primary-denial-clock",
        "VTI",
        "quote",
        denial.completed_at_us,
        {"bid": 249.0, "ask": 249.2},
    )

    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (terminal,),
        receipts=(denial,),
    )

    assert len(evaluation.outputs) == 1
    output = evaluation.outputs[0]
    assert output.payload["status"] == "incomplete"
    assert output.payload["reason"] == "primary_pair_discovery_denied"
    assert output.payload["action"] == "CALL"
    assert output.payload["terminal_basis"] == "explicit_discovery_denial"
    statuses = cast(
        Mapping[str, Mapping[str, JsonValue]],
        output.payload["primary_pair_interest_statuses"],
    )
    assert statuses[denial.interest_key]["reason"] == "OPTION_PERMISSION_DENIED"
    assert terminal.event_id in evaluation.output_input_event_ids[0]
    assert cast(Mapping[str, JsonValue], evaluation.state)["active"] == {}


def test_primary_pair_identity_mismatch_and_crossed_quote_stay_incomplete() -> None:
    plugin, evaluation = _active_primary_episode()
    call_receipt = _primary_receipt(
        "call",
        completed_at_us=2_100_000_000,
        instrument_id="same-contract",
    )
    put_receipt = _primary_receipt(
        "put",
        completed_at_us=2_100_000_001,
        instrument_id="same-contract",
    )
    terminal = _event(
        "primary-mismatch-clock",
        "VTI",
        "quote",
        put_receipt.completed_at_us,
        {"bid": 249.0, "ask": 249.2},
    )
    mismatch = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (terminal,),
        receipts=(call_receipt, put_receipt),
    )

    assert mismatch.outputs[0].payload["reason"] == "primary_pair_identity_mismatch"

    plugin, evaluation = _active_primary_episode()
    call_receipt = _primary_receipt("call", completed_at_us=2_100_000_000)
    put_receipt = _primary_receipt("put", completed_at_us=2_100_000_001)
    crossed = _event(
        "crossed-primary-call",
        "primary-call",
        "quote",
        2_100_000_010,
        {"bid": 4.2, "ask": 4.0},
    )
    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (crossed,),
        receipts=(call_receipt, put_receipt),
    )
    cutoff_at_us = 2_000_000_020 + 15 * 60 * 1_000_000
    cutoff = _event(
        "crossed-primary-cutoff",
        "VTI",
        "quote",
        cutoff_at_us,
        {"bid": 249.0, "ask": 249.2},
    )
    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (cutoff,),
    )

    assert evaluation.outputs[0].payload["reason"] == ("primary_pair_quote_evidence_incomplete")
    assert evaluation.outputs[0].payload["quote_proof_count"] == 0


def test_primary_receipt_at_cutoff_is_late_and_never_completes_the_pair() -> None:
    plugin, evaluation = _active_primary_episode()
    cutoff_at_us = 2_000_000_020 + 15 * 60 * 1_000_000
    late = _primary_receipt("call", completed_at_us=cutoff_at_us)
    cutoff = _event(
        "late-primary-receipt-cutoff",
        "primary-call",
        "quote",
        cutoff_at_us,
        {"bid": 4.0, "ask": 4.2},
    )

    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (cutoff,),
        receipts=(late,),
    )

    output = evaluation.outputs[0]
    statuses = cast(
        Mapping[str, Mapping[str, JsonValue]],
        output.payload["primary_pair_interest_statuses"],
    )
    assert output.payload["status"] == "incomplete"
    assert output.payload["reason"] == "primary_pair_discovery_late"
    assert output.payload["terminal_basis"] == "late_discovery_evidence"
    assert statuses[late.interest_key]["status"] == "late"
    assert statuses[late.interest_key]["completed_at_us"] == cutoff_at_us
    assert output.payload["quote_proof_count"] == 0


def test_missing_primary_reference_quote_terminalizes_with_cutoff_lineage() -> None:
    plugin, evaluation = _pending_primary_episode()
    cutoff_at_us = 2_000_000_020 + 15 * 60 * 1_000_000
    cutoff = _event(
        "missing-reference-cutoff",
        "VTI",
        "quote",
        cutoff_at_us,
        {"bid": 249.0, "ask": 249.2},
    )

    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (cutoff,),
    )

    assert not evaluation.interests
    assert len(evaluation.outputs) == 1
    output = evaluation.outputs[0]
    assert output.payload["status"] == "incomplete"
    assert output.payload["reason"] == "primary_underlying_quote_unavailable"
    assert output.payload["action"] == "CALL"
    assert output.payload["cutoff_crossing_event_id"] == cutoff.event_id
    assert cutoff.event_id in evaluation.output_input_event_ids[0]
    assert cast(Mapping[str, JsonValue], evaluation.state)["pending"] == {}


def test_later_session_closes_an_incomplete_barrier_without_silent_overwrite() -> None:
    plugin = create_plugin()
    first = _checkpoint_event("AAL", 0)
    evaluation = _advance(plugin, {}, (), (first,))
    later = _event(
        "prefix-AAL-next-session",
        "AAL",
        "bar_5m_session_prefix",
        3_000_000_000,
        _prefix_payload(session="2026-08-11"),
    )

    evaluation = _advance(
        plugin,
        evaluation.state,
        evaluation.retained_input_event_ids,
        (later,),
    )

    assert len(evaluation.outputs) == 20
    assert {output.kind for output in evaluation.outputs} == {"observation"}
    assert {output.payload["action"] for output in evaluation.outputs} == {"ABSTAIN"}
    assert {output.payload["reason"] for output in evaluation.outputs} == {
        "checkpoint_six_cohort_incomplete_at_rollover"
    }
    assert all(later.event_id in lineage for lineage in evaluation.output_input_event_ids)
    state = cast(Mapping[str, JsonValue], evaluation.state)
    cohort = cast(Mapping[str, JsonValue], state["cohort"])
    assert cohort["s"] == "2026-08-11"
    assert len(cast(Mapping[str, JsonValue], cohort["stocks"])) == 1
    assert len(evaluation.outputs) <= MANIFEST.maximum_outputs_per_batch
