"""Offline comparisons and unchanged Session HARD replay; no broker operations."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timedelta
from math import ceil
from statistics import median
from typing import Any

from stocker_core.markets import MarketId
from stocker_execution.dual_feed import ORDINARY, REFERENCE, StreamEvidence, TapeEvent
from stocker_execution.session_hard_method import SessionHardMethod, TradeEvent
from stocker_execution.session_hard_structure_d import SignalStatus, StrategySignal

TIME_DOMAINS = {
    "METHOD_TIME": "Local packet receipt time: REFERENCE_TBT keeps its production event_at; "
    "ORDINARY_TRADE_STREAM uses received_at. Current production behavior is reproduced "
    "as-is for equivalence testing, not endorsed or changed.",
    "BROKER_EVENT_TIME": "broker_at retains the feed's broker timestamp; ordinary raw "
    "event_at also retains broker milliseconds. Descriptive metadata only, not replay time.",
    "MONOTONIC_RECEIPT_TIME": "received_monotonic_ns is local callback observation time. "
    "Used descriptively for latency/order, never for method timestamps. TBT is observed "
    "at packet update dispatch and ordinary at tickString dispatch, so callback lag "
    "includes that dispatch difference.",
}

# Corrected before any market observation. Never tune using observed outcomes.
CRITERIA = {
    "version": "DUAL_FEED_V2_LOCAL_RECEIPT",
    "conversion": "REFERENCE_TBT REPLAY: existing production TradeEvents exactly as received. "
    "ORDINARY REPLAY: one TradeEvent per valid RT Trade Volume tick-77 payload, timestamped "
    "using that payload's LOCAL PACKET RECEIPT time; raw price unchanged; sequence in "
    "callback receipt order. Broker milliseconds remain descriptive metadata only. "
    "No rounding, shifting, bucketing, resampling, deduplication, expansion, interpolation "
    "or later outcome-based adjustment of replay inputs.",
    "time_domains": TIME_DOMAINS,
    "alignment": "Broker UTC second buckets, occurrence order within each bucket; zip only "
    "existing observations; excess observations remain unmatched. Price is not a match key.",
    "reference_timestamp": "Unchanged production TradeEvent timestamp is ib_async 2.1.0 "
    "local packet receive time, NOT the broker seconds argument. Both are retained separately.",
    "strict": [
        "All sufficiently observed pairs must be EXACT_METHOD_MATCH",
        "No opposite direction or missed reference signal (including receive-time expiry)",
        "Decision-relevant order preserved: same ordered price prefix through first break",
    ],
    "sufficient": "Both recorders active by T0 through T0+5m, one connection epoch, live data, "
    "valid prints on both feeds, no loss/errors, prospectively frozen method context available",
    "verdict_policy": "No production substitution claim. Incomplete pairs are always disclosed. "
    "No sufficient pairs => DIAGNOSTIC_NOT_RUN. "
    "Any sufficient strict failure => NOT_SUITABLE; all planned pairs sufficient and strict => "
    "METHOD_EQUIVALENT_IN_OBSERVED_SAMPLE; otherwise PROMISING_MORE_EVIDENCE_REQUIRED.",
}


def method_time(event: TapeEvent) -> datetime | None:
    """Select replay time without altering the recorded evidence or production TBT."""
    if event.feed == ORDINARY:
        return event.received_at
    if event.feed == REFERENCE:
        return event.event_at
    raise ValueError(f"Unknown diagnostic feed: {event.feed}")


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "median": median(ordered) if ordered else None,
        "p95": ordered[ceil(len(ordered) * 0.95) - 1] if ordered else None,
        "maximum": max(ordered) if ordered else None,
        "minimum": min(ordered) if ordered else None,
    }


def _valid(event: TapeEvent) -> bool:
    return not event.invalid_reason and event.event_at is not None and event.price is not None


def coverage(events: Sequence[TapeEvent]) -> dict[str, Any]:
    times = [e.event_at for e in events if e.event_at is not None]
    broker_times = [e.broker_at for e in events if e.broker_at is not None]
    pairs = list(zip(events, events[1:], strict=False))
    receipt_counts = Counter(e.received_at for e in events)
    method_times = [at for e in events if (at := method_time(e)) is not None]
    return {
        "event_count": len(events),
        "unique_price_levels": len({e.price for e in events}),
        "price_change_count": sum(a.price != b.price for a, b in pairs),
        "repeated_adjacent_prices": sum(a.price == b.price for a, b in pairs),
        "out_of_order_source_event_times": sum(
            a.event_at > b.event_at for a, b in pairs if a.event_at and b.event_at
        ),
        "out_of_order_broker_times": sum(
            a > b for a, b in zip(broker_times, broker_times[1:], strict=False)
        ),
        "broker_timestamp_ties": len(broker_times) - len(set(broker_times)),
        "first_broker_event_timestamp": broker_times[0] if broker_times else None,
        "last_broker_event_timestamp": broker_times[-1] if broker_times else None,
        "first_method_timestamp": method_times[0] if method_times else None,
        "last_method_timestamp": method_times[-1] if method_times else None,
        "receipt_order_inversions": sum(a.received_at > b.received_at for a, b in pairs),
        "monotonic_receipt_order_inversions": sum(
            a.received_monotonic_ns > b.received_monotonic_ns for a, b in pairs
        ),
        "packet_receipt_timestamp_ties": len(events) - len(receipt_counts),
        "max_events_per_receipt_timestamp": max(receipt_counts.values(), default=0),
        "timestamp_ties": len(times) - len(set(times)),
        "first_event_timestamp": times[0] if times else None,
        "last_event_timestamp": times[-1] if times else None,
        "duration_seconds": (max(times) - min(times)).total_seconds() if times else None,
        "not_single_trade_flags": sum(
            e.raw.get("payload", "").split(";")[-1].lower() == "false" for e in events
        ),
    }


def replay_method(
    initial: StrategySignal, events: Sequence[TapeEvent], *, end: datetime
) -> dict[str, Any]:
    """Restore identical pre-event context; deliver by receipt, expire by local clock.

    The initial signal is the real method's prospective qualification/Q1 output.
    Both replays use restore_signals/observe_trades/expire_waiting_before unchanged.
    A prefix received before arming is buffered until that same arming instant.
    """
    if initial.market_id is None:
        raise ValueError("METHOD_MARKET_UNAVAILABLE")
    method = SessionHardMethod(
        MarketId(initial.market_id),
        method_version=initial.strategy_version,
    )
    if initial.method_spec_hash != method.spec_hash:
        raise ValueError("METHOD_SPEC_MISMATCH")
    if initial.last_event_sequence is not None or initial.entry_timestamp is not None:
        raise ValueError("CONTEXT_ALREADY_CONSUMED_EVENTS")
    method.restore_signals((initial,))
    assert initial.underlying_con_id is not None
    decision_received_at = None
    trigger: TapeEvent | None = None
    first_crossing: TapeEvent | None = None
    ordered = sorted(events, key=lambda e: e.sequence)
    for event in ordered:
        assert event.event_at is not None and event.price is not None
        timestamp = method_time(event)
        assert timestamp is not None
        if not initial.t0 <= timestamp < initial.t0 + timedelta(minutes=5):
            continue
        if (
            first_crossing is None
            and initial.up_trigger is not None
            and initial.down_trigger is not None
            and (event.price >= initial.up_trigger or event.price <= initial.down_trigger)
        ):
            first_crossing = event
        now = max(event.received_at, initial.armed_at or initial.t0)
        if now >= end:
            continue
        method.expire_waiting_before(now)
        before = method.signals[0]
        method.observe_trades(
            {initial.underlying_con_id: (TradeEvent(timestamp, event.price, event.sequence),)}
        )
        after = method.signals[0]
        if before.status is SignalStatus.WAITING_FOR_ENTRY and after.status is not before.status:
            trigger, decision_received_at = event, now
    method.expire_waiting_before(end)
    result = method.signals[0]
    signal = result.status is SignalStatus.ENTRY_TRIGGERED
    return {
        "qualifying_break": signal,
        "first_break_direction": result.direction if signal else None,
        "first_qualifying_event_timestamp": result.entry_timestamp,
        "replay_timestamp_domain": "METHOD_TIME",
        "method_decision_timestamp": result.signal_timestamp,
        "method_decision_minute": result.signal_timestamp.replace(second=0, microsecond=0)
        if result.signal_timestamp
        else None,
        "decision_received_at": decision_received_at,
        "entry_side": result.side if signal else None,
        "entry_reference": result.entry_reference if signal else None,
        "status": result.status.value,
        "reason": result.reason,
        "signal": signal,
        "trigger_price": trigger.price if trigger else None,
        "trigger_sequence": trigger.sequence if trigger else None,
        "first_crossing": asdict(first_crossing) if first_crossing else None,
    }


def _classify(reference: dict[str, Any], ordinary: dict[str, Any]) -> str:
    if reference["signal"] and not ordinary["signal"]:
        return "REFERENCE_SIGNAL_MISSED"
    if ordinary["signal"] and not reference["signal"]:
        return "ALTERNATIVE_ONLY_SIGNAL"
    if reference["entry_side"] != ordinary["entry_side"]:
        return "OPPOSITE_DIRECTION"
    if any(reference[k] != ordinary[k] for k in ("entry_reference", "trigger_price", "reason")):
        return "SAME_DIRECTION_DIFFERENT_TRIGGER"
    if reference["method_decision_timestamp"] != ordinary["method_decision_timestamp"]:
        return "SAME_METHOD_DECISION_DIFFERENT_EVENT_TIMING"
    return "EXACT_METHOD_MATCH"


def compare_pair(
    con_id: int,
    events: Sequence[TapeEvent],
    streams: Sequence[StreamEvidence],
    *,
    t0: datetime,
    end: datetime,
    initial: StrategySignal | None,
    integrity_errors: Sequence[str] = (),
) -> dict[str, Any]:
    window_end = t0 + timedelta(minutes=5)
    raw = [e for e in events if e.con_id == con_id]
    tapes = {
        feed: [
            e
            for e in raw
            if e.feed == feed
            and _valid(e)
            and (timestamp := method_time(e)) is not None
            and t0 <= timestamp < window_end
        ]
        for feed in (REFERENCE, ORDINARY)
    }
    errors = list(integrity_errors)
    if end < window_end:
        errors.append("INCOMPLETE_CAUSAL_WINDOW")
    for feed, tape in tapes.items():
        stream = next((s for s in streams if s.con_id == con_id and s.feed == feed), None)
        if stream is None or stream.subscribed_at is None:
            errors.append(f"{feed}:NOT_SUBSCRIBED")
        elif (
            stream.recording_started_at is None
            or stream.recording_started_at > t0
            or stream.released_at is None
            or stream.released_at < window_end
        ):
            errors.append(f"{feed}:INCOMPLETE_RECORDING_PREFIX")
        if stream and (stream.rejection or stream.errors):
            errors.extend([f"{feed}:{v}" for v in [stream.rejection, *stream.errors] if v])
        if not tape:
            errors.append(f"{feed}:ZERO_VALID_PRINTS")
        if any(e.market_data_type != 1 for e in tape):
            errors.append(f"{feed}:NON_LIVE_OR_UNKNOWN_DATA_TYPE")
    if len({e.connection_epoch for e in raw}) > 1:
        errors.append("MULTIPLE_CONNECTION_EPOCHS")
    if any(e.invalid_reason for e in raw):
        errors.append("INVALID_RAW_OBSERVATIONS")
    if initial is None or initial.t0 != t0 or initial.underlying_con_id != con_id:
        errors.append("METHOD_CONTEXT_UNAVAILABLE_OR_MISMATCHED")
    ref, alt = tapes[REFERENCE], tapes[ORDINARY]
    buckets: dict[datetime, list[list[TapeEvent]]] = defaultdict(lambda: [[], []])
    for index, tape in enumerate((ref, alt)):
        for e in tape:
            assert e.event_at is not None
            if e.broker_at is not None:
                buckets[e.broker_at.replace(microsecond=0)][index].append(e)
            else:
                errors.append(f"{e.feed}:BROKER_TIMESTAMP_UNAVAILABLE_FOR_ALIGNMENT")
                buckets[e.event_at.replace(microsecond=0)][index].append(e)
    aligned: list[tuple[TapeEvent, TapeEvent]] = []
    only_ref, only_alt = [], []
    for left, right in buckets.values():
        count = min(len(left), len(right))
        aligned.extend(zip(left[:count], right[:count], strict=True))
        only_ref.extend(left[count:])
        only_alt.extend(right[count:])
    equal = [(a, b) for a, b in aligned if a.price == b.price]
    result: dict[str, Any] = {
        "con_id": con_id,
        "t0": t0,
        "window_end": window_end,
        "time_domains": TIME_DOMAINS,
        "coverage_window_domain": "METHOD_TIME",
        "reference": coverage(ref),
        "ordinary": coverage(alt),
        "invalid_observations": sum(bool(e.invalid_reason) for e in raw),
        "events_only_in_tbt": [e.sequence for e in only_ref],
        "events_only_in_ordinary": [e.sequence for e in only_alt],
        "alignment_pairs": [(a.sequence, b.sequence) for a, b in aligned],
        "alignment_is_trade_identity": False,
        "price_mismatched_alignment_pairs": [
            (a.sequence, b.sequence) for a, b in aligned if a.price != b.price
        ],
        "exact_price_match_proportion_aligned": len(equal) / len(aligned) if aligned else None,
        "exact_price_match_proportion_reference": len(equal) / len(ref) if ref else None,
        "absolute_price_difference": distribution(
            [
                abs(a.price - b.price)
                for a, b in aligned
                if a.price is not None and b.price is not None
            ]
        ),
        "receive_lag_seconds_equal_price_aligned": distribution(
            [(b.received_at - a.received_at).total_seconds() for a, b in equal]
        ),
        "broker_event_lag_seconds_equal_price_aligned": distribution(
            [
                (b.broker_at - a.broker_at).total_seconds()
                for a, b in equal
                if a.broker_at is not None and b.broker_at is not None
            ]
        ),
        "monotonic_receipt_lag_seconds_equal_price_aligned": distribution(
            [(b.received_monotonic_ns - a.received_monotonic_ns) / 1_000_000_000 for a, b in equal]
        ),
        "missing_reference_price_levels": sorted({e.price for e in ref} - {e.price for e in alt}),
        "unmatched_semantics": "Occurrence alignment only; unequal bucket counts can reflect "
        "aggregation or missing/extra prints. No unique broker trade IDs exist here.",
    }
    reference_replay = ordinary_replay = None
    if initial is not None and not any("METHOD_CONTEXT" in e for e in errors):
        try:
            reference_replay = replay_method(initial, ref, end=end)
            ordinary_replay = replay_method(initial, alt, end=end)
        except ValueError as exc:
            errors.append(str(exc))
    classification = "INSUFFICIENT_DATA"
    preserved = False
    late = None
    if reference_replay is not None and ordinary_replay is not None:
        classification = _classify(reference_replay, ordinary_replay)
        rseq, aseq = reference_replay["trigger_sequence"], ordinary_replay["trigger_sequence"]
        # Conservative: exact order of all prices through the decision, including
        # repeats. Extra/collapsed repeats remain visible and cannot pass strict.
        preserved = [e.price for e in ref if rseq is None or e.sequence <= rseq] == [
            e.price for e in alt if aseq is None or e.sequence <= aseq
        ]
        when = reference_replay["decision_received_at"]
        late = (
            sum(e.feed == ORDINARY and _valid(e) and when < e.received_at < end for e in raw)
            if when
            else None
        )
    result.update(
        reference_replay=reference_replay,
        ordinary_replay=ordinary_replay,
        descriptive_classification=classification,
        classification="INSUFFICIENT_DATA" if errors else classification,
        insufficient_reasons=sorted(set(errors)),
        decision_relevant_price_order_preserved=preserved,
        ordinary_events_received_after_reference_decision=late,
        ordinary_events_received_at_or_after_expiry=sum(
            e.feed == ORDINARY and _valid(e) and window_end <= e.received_at < end for e in raw
        ),
        strict_pass=not errors and classification == "EXACT_METHOD_MATCH" and preserved,
    )
    return result


def verdict(pairs: Sequence[dict[str, Any]]) -> str:
    sufficient = [p for p in pairs if p["classification"] != "INSUFFICIENT_DATA"]
    if not sufficient:
        return "DUAL_FEED_DIAGNOSTIC_NOT_RUN"
    if any(not p["strict_pass"] for p in sufficient):
        return "ORDINARY_FEED_NOT_SUITABLE"
    if sufficient and len(sufficient) == len(pairs):
        return "ORDINARY_FEED_METHOD_EQUIVALENT_IN_OBSERVED_SAMPLE"
    return "ORDINARY_FEED_PROMISING_MORE_EVIDENCE_REQUIRED"
