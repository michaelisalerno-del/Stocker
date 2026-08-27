"""Frozen M1C Opening Reversal V1.1 over generic causal receipts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Literal, cast

from stocker_ideas.plugins.frozen_m1c_v0 import (
    COHORT,
    M1C_THRESHOLD,
    build_front_options_context,
    build_group_i_for_symbol,
    score_m1c,
)
from stocker_runtime.domain import (
    IdeaOutput,
    JsonValue,
    MarketEvent,
    Observation,
    OutputKind,
    RuntimeMode,
    Signal,
    canonical_json_bytes,
)
from stocker_runtime.ideas.contract import (
    DiscoveryReceipt,
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    IdeaManifest,
    MarketDataInterest,
    MarketDataRequirement,
)

UNIVERSE = (*COHORT, "VTI")

NEGATIVE_OPENING_RETURN_THRESHOLD = -0.00288963733897
POSITIVE_OPENING_RETURN_THRESHOLD = 0.00225522676046
OPENING_RANGE_THRESHOLD = 0.00384818171835
RULE_HASH = "7a2a40170fa1dffb148cb51144c1c9bfcb029c5d797ea9680164cfd9fbce1ea7"

_D1_WINDOW_US = 30 * 60 * 1_000_000
_PRIMARY_WINDOW_US = 15 * 60 * 1_000_000
_RIGHTS: tuple[Literal["call", "put"], ...] = ("call", "put")

PARAMETERS: Mapping[str, JsonValue] = {
    "checkpoint": 6,
    "m1c_threshold": M1C_THRESHOLD,
    "negative_opening_return_threshold": NEGATIVE_OPENING_RETURN_THRESHOLD,
    "positive_opening_return_threshold": POSITIVE_OPENING_RETURN_THRESHOLD,
    "opening_range_threshold": OPENING_RANGE_THRESHOLD,
    "d1_option_minimum_days_to_expiry": 7,
    "d1_option_maximum_days_to_expiry": 45,
    "d1_snapshot_lifetime_minutes": 30,
    "primary_option_days_to_expiry": 1,
    "primary_horizon_minutes": 15,
    "maximum_promoted_underlyings": 1,
}

MANIFEST = IdeaManifest(
    api_version=1,
    idea_id="m1c_opening_reversal",
    idea_version="v1_1",
    display_name="M1C Opening Reversal V1.1",
    description="Frozen checkpoint-six M1C opening-reversal evidence and primary pair recording.",
    modes=(RuntimeMode.PROSPECTIVE_RECORD, RuntimeMode.SHADOW),
    output_kinds=(OutputKind.OBSERVATION, OutputKind.SIGNAL),
    parameter_schema_version="m1c-opening-reversal-v1.1",
    parameter_schema={
        "type": "object",
        "additionalProperties": False,
        "required": tuple(PARAMETERS),
        "properties": {
            "checkpoint": {"type": "integer", "enum": (6,)},
            "m1c_threshold": {"type": "number", "enum": (M1C_THRESHOLD,)},
            "negative_opening_return_threshold": {
                "type": "number",
                "enum": (NEGATIVE_OPENING_RETURN_THRESHOLD,),
            },
            "positive_opening_return_threshold": {
                "type": "number",
                "enum": (POSITIVE_OPENING_RETURN_THRESHOLD,),
            },
            "opening_range_threshold": {
                "type": "number",
                "enum": (OPENING_RANGE_THRESHOLD,),
            },
            "d1_option_minimum_days_to_expiry": {"type": "integer", "enum": (7,)},
            "d1_option_maximum_days_to_expiry": {"type": "integer", "enum": (45,)},
            "d1_snapshot_lifetime_minutes": {"type": "integer", "enum": (30,)},
            "primary_option_days_to_expiry": {"type": "integer", "enum": (1,)},
            "primary_horizon_minutes": {"type": "integer", "enum": (15,)},
            "maximum_promoted_underlyings": {"type": "integer", "enum": (1,)},
        },
    },
    maximum_state_bytes=65_536,
    maximum_outputs_per_batch=21,
    maximum_interests_per_batch=42,
)


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _table(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): dict(item)
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, Mapping)
    }


def _string_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _available_at(event: MarketEvent) -> int:
    return max(event.event_at_us, event.received_at_us)


def _lineage(batch: IdeaBatch, selected: set[str]) -> tuple[str, ...]:
    available = tuple(
        dict.fromkeys(
            (
                *batch.prior_state_input_event_ids,
                *(event.event_id for event in batch.events),
                *(event.event_id for event in batch.rehydrated_events),
            )
        )
    )
    return tuple(event_id for event_id in available if event_id in selected)


def _d1_interest_parts(key: str) -> tuple[str, str, Literal["call", "put"]] | None:
    parts = key.split(":")
    if len(parts) != 6 or parts[:3] != ["opening-reversal", "m1c", "d1"] or parts[5] not in _RIGHTS:
        return None
    return parts[3], parts[4], parts[5]


def _primary_interest_parts(key: str) -> tuple[str, str, Literal["call", "put"]] | None:
    parts = key.split(":")
    if len(parts) != 5 or parts[:2] != ["opening-reversal", "primary"] or parts[4] not in _RIGHTS:
        return None
    return parts[2], parts[3], parts[4]


def _valid_quote(
    event: MarketEvent,
    *,
    after_us: int,
    before_us: int,
) -> float | None:
    available = _available_at(event)
    bid = _number(event.payload.get("bid"))
    ask = _number(event.payload.get("ask"))
    if (
        event.feed_kind != "quotes"
        or event.event_kind != "quote"
        or available < after_us
        or available >= before_us
        or bid is None
        or ask is None
        or bid < 0.0
        or ask < bid
        or (bid + ask) / 2.0 <= 0.0
    ):
        return None
    return (bid + ask) / 2.0


def _first_event_at_or_after(batch: IdeaBatch, at_us: int) -> MarketEvent | None:
    return next((event for event in batch.events if _available_at(event) >= at_us), None)


def _advance_causal_horizon(
    horizon: Mapping[str, object],
    event: MarketEvent,
) -> dict[str, object]:
    available = _available_at(event)
    prior_available = _integer(horizon.get("a"))
    if prior_available is None or available > prior_available:
        return {"a": available, "i": event.event_id}
    return dict(horizon)


def _terminalize_elapsed_d1_windows(
    *,
    baselines: Mapping[str, dict[str, object]],
    d1_context: Mapping[str, dict[str, object]],
    d1_terminal: dict[str, dict[str, object]],
    causal_horizon: Mapping[str, object],
) -> None:
    horizon_at_us = _integer(causal_horizon.get("a"))
    horizon_event_id = causal_horizon.get("i")
    if horizon_at_us is None or not isinstance(horizon_event_id, str):
        return
    for symbol, baseline in baselines.items():
        baseline_session = baseline.get("s")
        cutoff_at_us = _integer(baseline.get("x"))
        if (
            not isinstance(baseline_session, str)
            or cutoff_at_us is None
            or horizon_at_us < cutoff_at_us
        ):
            continue
        context = d1_context.get(symbol)
        terminal = d1_terminal.get(symbol)
        if terminal is None or terminal.get("s") != baseline_session:
            terminal = {"s": baseline_session}
        changed = False
        for right in _RIGHTS:
            capture = (
                _mapping(context.get(right))
                if context is not None and context.get("s") == baseline_session
                else {}
            )
            if capture or terminal.get(right) in {
                "captured",
                "denied",
                "late",
                "window_elapsed",
            }:
                continue
            terminal[right] = "window_elapsed"
            terminal.setdefault(
                f"{right}_k",
                f"opening-reversal:m1c:d1:{baseline_session}:{symbol}:{right}",
            )
            terminal[f"{right}_x"] = cutoff_at_us
            terminal[f"{right}_cutoff_i"] = horizon_event_id
            terminal[f"{right}_cutoff_a"] = horizon_at_us
            changed = True
        if changed:
            d1_terminal[symbol] = terminal


def _primary_expected_keys(active: Mapping[str, object]) -> tuple[str, ...]:
    return _string_values(active.get("k"))


def _primary_pair_identity(
    active: Mapping[str, object],
) -> tuple[str | None, int | None, dict[str, JsonValue]]:
    expected = _primary_expected_keys(active)
    statuses = _table(active.get("u"))
    if len(expected) != 2:
        raise ValueError("Opening Reversal primary pair must contain two interests")
    denied = tuple(
        status
        for key in expected
        if (status := statuses.get(key)) is not None and status.get("status") == "denied"
    )
    if denied:
        completed = tuple(
            value for status in denied if (value := _integer(status.get("completed"))) is not None
        )
        return (
            "primary_pair_discovery_denied",
            min(completed) if completed else None,
            {},
        )
    late = tuple(
        status
        for key in expected
        if (status := statuses.get(key)) is not None and status.get("status") == "late"
    )
    if late:
        completed = tuple(
            value for status in late if (value := _integer(status.get("completed"))) is not None
        )
        return (
            "primary_pair_discovery_late",
            min(completed) if completed else None,
            {},
        )
    resolved = tuple(statuses.get(key) for key in expected)
    if any(status is None or status.get("status") != "resolved" for status in resolved):
        return "primary_pair_discovery_incomplete", None, {}
    complete_statuses = cast(tuple[dict[str, object], dict[str, object]], resolved)
    completion_times = tuple(
        value
        for status in complete_statuses
        if (value := _integer(status.get("completed"))) is not None
    )
    terminal_at = max(completion_times) if len(completion_times) == 2 else None
    rights: dict[str, dict[str, object]] = {}
    for key, status in zip(expected, complete_statuses, strict=True):
        parts = _primary_interest_parts(key)
        if (
            parts is None
            or status.get("right") != parts[2]
            or not isinstance(status.get("instrument_id"), str)
            or not isinstance(status.get("expiry"), str)
            or _number(status.get("strike")) is None
            or _integer(status.get("completed")) is None
        ):
            return "primary_pair_identity_invalid", terminal_at, {}
        rights[parts[2]] = status
    if set(rights) != set(_RIGHTS):
        return "primary_pair_identity_invalid", terminal_at, {}
    call = rights["call"]
    put = rights["put"]
    session = active.get("s")
    try:
        exact_expiry = (date.fromisoformat(cast(str, session)) + timedelta(days=1)).strftime(
            "%Y%m%d"
        )
    except (TypeError, ValueError):
        return "primary_pair_identity_invalid", terminal_at, {}
    call_strike = cast(float, _number(call.get("strike")))
    put_strike = cast(float, _number(put.get("strike")))
    if (
        call.get("instrument_id") == put.get("instrument_id")
        or call.get("expiry") != put.get("expiry")
        or call.get("expiry") != exact_expiry
        or call_strike != put_strike
    ):
        return "primary_pair_identity_mismatch", terminal_at, {}
    return (
        None,
        None,
        {
            "expiry": cast(str, call["expiry"]),
            "strike": call_strike,
            "call_instrument_id": cast(str, call["instrument_id"]),
            "put_instrument_id": cast(str, put["instrument_id"]),
        },
    )


def _terminal_attribution(
    statuses: Mapping[str, Mapping[str, object]],
    expected: tuple[str, ...],
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key in expected:
        status = statuses.get(key)
        result[key] = cast(
            JsonValue,
            {
                "status": None if status is None else status.get("status"),
                "reason": None if status is None else status.get("reason"),
                "instrument_id": (None if status is None else status.get("instrument_id")),
                "expiry": None if status is None else status.get("expiry"),
                "strike": None if status is None else status.get("strike"),
                "right": None if status is None else status.get("right"),
                "completed_at_us": (None if status is None else status.get("completed")),
            },
        )
    return result


def _stable_id(*parts: object) -> str:
    return hashlib.sha256(canonical_json_bytes(cast(JsonValue, tuple(parts)))).hexdigest()


def _market_record(event: MarketEvent) -> dict[str, object]:
    record: dict[str, object] = {
        "i": event.event_id,
        "t": event.event_at_us,
        "a": _available_at(event),
        "status": "unavailable",
        "reason": "market_prefix_incomplete",
        "state": "UNKNOWN_INCOMPLETE",
        "sign": None,
    }
    if event.payload.get("source_completeness") != "complete":
        return record
    accumulator = _mapping(event.payload.get("accumulator"))
    opening = _number(accumulator.get("session_open"))
    high = _number(accumulator.get("session_high"))
    low = _number(accumulator.get("session_low"))
    close = _number(accumulator.get("last_close"))
    if (
        opening is None
        or high is None
        or low is None
        or close is None
        or min(opening, high, low, close) <= 0.0
        or high < low
    ):
        record["reason"] = "market_opening_measurement_incomplete"
        return record
    opening_return = math.log(close / opening)
    opening_range = math.log(high / low)
    state = "NORMAL_OPENING"
    sign: int | None = None
    if (
        opening_return <= NEGATIVE_OPENING_RETURN_THRESHOLD
        and opening_range >= OPENING_RANGE_THRESHOLD
    ):
        state = "NEGATIVE_SEVERE_OPENING_TRANSITION"
        sign = -1
    elif (
        opening_return >= POSITIVE_OPENING_RETURN_THRESHOLD
        and opening_range >= OPENING_RANGE_THRESHOLD
    ):
        state = "POSITIVE_SEVERE_OPENING_TRANSITION"
        sign = 1
    elif opening_range >= OPENING_RANGE_THRESHOLD:
        state = "ELEVATED_OPENING_RANGE_NONDIRECTIONAL"
    record.update(
        {
            "status": "complete",
            "reason": None if sign is not None else "opening_state_not_severe",
            "state": state,
            "sign": sign,
            "opening_return": opening_return,
            "opening_range": opening_range,
        }
    )
    return record


def _stock_record(
    *,
    event: MarketEvent,
    symbol: str,
    session: str,
    baseline: Mapping[str, object] | None,
    context: Mapping[str, object] | None,
    terminal: Mapping[str, object] | None,
) -> dict[str, object]:
    selected = {event.event_id}
    record: dict[str, object] = {
        "i": event.event_id,
        "t": event.event_at_us,
        "a": _available_at(event),
        "p": None,
        "h": False,
        "status": "unavailable",
        "reason": "session_prefix_incomplete",
        "l": (),
        "reference": None,
        "iv_scale_15m": None,
    }
    accumulator = _mapping(event.payload.get("accumulator"))
    reference = _number(accumulator.get("last_close"))
    if reference is not None and reference > 0.0:
        record["reference"] = reference
    if event.payload.get("source_completeness") != "complete":
        record["l"] = tuple(sorted(selected))
        return record
    try:
        group_i = build_group_i_for_symbol(event.payload, symbol=symbol, checkpoint=6)
    except ValueError:
        record["reason"] = "activity_baseline_not_ready"
        record["l"] = tuple(sorted(selected))
        return record
    if baseline is None or not isinstance(baseline.get("i"), str):
        record["reason"] = "prior_session_baseline_missing"
        record["l"] = tuple(sorted(selected))
        return record
    selected.add(cast(str, baseline["i"]))
    baseline_session = baseline.get("s")
    if not isinstance(baseline_session, str) or baseline_session >= session:
        record["reason"] = "prior_session_baseline_not_causal"
        record["l"] = tuple(sorted(selected))
        return record
    cutoff = _integer(baseline.get("x"))
    if cutoff is None:
        raise ValueError("Opening Reversal D-1 cutoff is unavailable")
    if _available_at(event) < cutoff:
        raise ValueError("checkpoint-six cohort precedes its D-1 option cutoff")
    realised = _number(baseline.get("v"))
    if realised is None or realised <= 0.0:
        record["reason"] = "realised_volatility_20d_not_ready"
        record["l"] = tuple(sorted(selected))
        return record
    call = {} if context is None else _mapping(context.get("call"))
    put = {} if context is None else _mapping(context.get("put"))
    for capture in (call, put):
        if isinstance(capture.get("i"), str):
            selected.add(cast(str, capture["i"]))
    if not call or not put:
        record["reason"] = "prior_session_option_pair_incomplete"
        terminal_statuses: dict[str, JsonValue] = {}
        interest_keys: dict[str, JsonValue] = {}
        denial_reasons: dict[str, JsonValue] = {}
        evidence_event_ids: dict[str, JsonValue] = {}
        cutoff_event_ids: dict[str, JsonValue] = {}
        for right, capture in (("call", call), ("put", put)):
            status = "captured" if capture else None
            if isinstance(capture.get("i"), str):
                evidence_event_ids[right] = cast(str, capture["i"])
            if status is None and terminal is not None:
                raw_status = terminal.get(right)
                status = raw_status if isinstance(raw_status, str) else None
                terminal_event_id = terminal.get(f"{right}_i")
                if isinstance(terminal_event_id, str):
                    selected.add(terminal_event_id)
                    evidence_event_ids[right] = terminal_event_id
                cutoff_event_id = terminal.get(f"{right}_cutoff_i")
                if isinstance(cutoff_event_id, str):
                    selected.add(cutoff_event_id)
                    cutoff_event_ids[right] = cutoff_event_id
                interest_key = terminal.get(f"{right}_k")
                if isinstance(interest_key, str):
                    interest_keys[right] = interest_key
                reason = terminal.get(f"{right}_r")
                if isinstance(reason, str):
                    denial_reasons[right] = reason
            terminal_statuses[right] = (
                status if status in {"captured", "denied", "late"} else "window_elapsed"
            )
            interest_keys.setdefault(
                right,
                f"opening-reversal:m1c:d1:{baseline_session}:{symbol}:{right}",
            )
        distinct_statuses = set(terminal_statuses.values())
        terminal_basis = (
            "explicit_discovery_denial"
            if distinct_statuses == {"denied"}
            else "late_option_capture"
            if distinct_statuses == {"late"}
            else "causal_window_elapsed_without_capture"
            if distinct_statuses == {"window_elapsed"}
            else "mixed_terminal_option_evidence"
        )
        record["terminal"] = {
            "statuses": terminal_statuses,
            "interest_keys": interest_keys,
            "denial_reasons": denial_reasons,
            "evidence_event_ids": evidence_event_ids,
            "cutoff_event_ids": cutoff_event_ids,
            "cutoff_at_us": cutoff,
            "cohort_available_at_us": _available_at(event),
            "terminal_basis": terminal_basis,
        }
        record["l"] = tuple(sorted(selected))
        return record
    try:
        call_iv = _number(call.get("model_implied_volatility"))
        put_iv = _number(put.get("model_implied_volatility"))
        if call_iv is None or put_iv is None or min(call_iv, put_iv) <= 0.0:
            raise ValueError("D-1 IV pair is invalid")
        group_o = build_front_options_context(
            call_capture=call,
            put_capture=put,
            prior_close=cast(float, baseline["c"]),
            realised_volatility_20d=realised,
        )
        score = score_m1c(
            symbol=symbol,
            checkpoint=6,
            group_o=group_o,
            group_i=group_i,
        )
    except (KeyError, TypeError, ValueError):
        record["reason"] = "prior_session_option_pair_invalid"
        record["l"] = tuple(sorted(selected))
        return record
    probability = cast(float, score["probability"])
    high_tail = probability >= M1C_THRESHOLD
    iv_scale = (
        ((call_iv + put_iv) / 2.0) * math.sqrt(15.0 / (252.0 * 390.0)) * math.sqrt(2.0 / math.pi)
    )
    record.update(
        {
            "p": probability,
            "h": high_tail,
            "status": "complete",
            "reason": None if high_tail else "m1c_below_frozen_high_tail",
            "l": tuple(sorted(selected)),
            "feature_hash": score["feature_hash"],
            "model_hash": score["model_hash"],
            "missing_feature_count": score["missing_feature_count"],
            "iv_scale_15m": iv_scale,
        }
    )
    return record


def _emit_complete_cohort(
    *,
    batch: IdeaBatch,
    cohort: Mapping[str, object],
    causal_horizon: Mapping[str, object],
) -> tuple[
    list[IdeaOutput],
    list[tuple[str, ...]],
    dict[str, object],
    dict[str, object],
]:
    session = cohort.get("s")
    stocks = _table(cohort.get("stocks"))
    market = _mapping(cohort.get("market"))
    if not isinstance(session, str) or set(stocks) != set(COHORT) or not market:
        raise ValueError("Opening Reversal cohort is incomplete")
    market_sign = market.get("sign")
    candidates = tuple(
        sorted(
            (
                (
                    -cast(float, record["p"]),
                    cast(int, record["a"]),
                    symbol,
                )
                for symbol, record in stocks.items()
                if record.get("h") is True
                and isinstance(record.get("p"), float)
                and market_sign in {-1, 1}
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
    )
    winner = None if not candidates else candidates[0][2]
    action = "CALL" if market_sign == -1 else "PUT" if market_sign == 1 else "ABSTAIN"
    barrier_candidates = [
        (cast(int, market["a"]), cast(str, market["i"])),
        *((cast(int, stocks[symbol]["a"]), cast(str, stocks[symbol]["i"])) for symbol in COHORT),
    ]
    horizon_at_us = _integer(causal_horizon.get("a"))
    horizon_event_id = causal_horizon.get("i")
    if horizon_at_us is not None and isinstance(horizon_event_id, str):
        barrier_candidates.append((horizon_at_us, horizon_event_id))
    barrier_at_us, barrier_event_id = max(
        barrier_candidates,
        key=lambda item: (item[0], item[1]),
    )
    primary_cutoff_at_us = cast(int, market["t"]) + _PRIMARY_WINDOW_US
    transition_id = _stable_id(
        "m1c-opening-reversal-transition-v1.1",
        session,
        market.get("state"),
        market.get("i"),
    )
    outputs: list[IdeaOutput] = []
    lineages: list[tuple[str, ...]] = []
    winner_payload: Mapping[str, JsonValue] | None = None
    all_selected = {cast(str, market["i"])}
    for record in stocks.values():
        all_selected.update(_string_values(record.get("l")))
    all_selected.add(barrier_event_id)
    for symbol in sorted(COHORT):
        record = stocks[symbol]
        lineage = _lineage(batch, all_selected)
        if not lineage:
            raise ValueError("Opening Reversal output lacks causal evidence")
        stock_complete = record.get("status") == "complete"
        market_complete = market.get("status") == "complete"
        eligible = bool(record.get("h")) and market_sign in {-1, 1}
        reason = (
            None
            if eligible
            else market.get("reason")
            if market_sign not in {-1, 1}
            else record.get("reason")
        )
        payload = cast(
            Mapping[str, JsonValue],
            {
                "method_id": "m1c_opening_reversal_v1_1",
                "rule_hash": RULE_HASH,
                "session": session,
                "checkpoint": 6,
                "status": "complete" if stock_complete and market_complete else "unavailable",
                "reason": reason,
                "action": action if eligible else "ABSTAIN",
                "prediction_sign": (1 if eligible and action == "CALL" else -1 if eligible else 0),
                "eligible": eligible,
                "promoted": symbol == winner,
                "promotion_status": (
                    "selected"
                    if symbol == winner
                    else "not_promoted"
                    if eligible
                    else "not_eligible"
                ),
                "probability": record.get("p"),
                "m1c_threshold": M1C_THRESHOLD,
                "high_tail_membership": bool(record.get("h")),
                "tail_phase": "FIRST_ENTRY",
                "fresh_episode": bool(record.get("h")),
                "feature_hash": record.get("feature_hash"),
                "model_hash": record.get("model_hash"),
                "missing_feature_count": record.get("missing_feature_count"),
                "previous_close_atm_iv_scale_15m": record.get("iv_scale_15m"),
                "d1_terminal_evidence": record.get("terminal"),
                "market_proxy": "VTI",
                "opening_transition_id": transition_id,
                "opening_transition_state": market.get("state"),
                "opening_transition_sign": market_sign,
                "market_opening_return": market.get("opening_return"),
                "market_opening_range": market.get("opening_range"),
                "negative_opening_return_threshold": NEGATIVE_OPENING_RETURN_THRESHOLD,
                "positive_opening_return_threshold": POSITIVE_OPENING_RETURN_THRESHOLD,
                "opening_range_threshold": OPENING_RANGE_THRESHOLD,
                "validated_directional_evidence": False,
                "option_profitability_claim": False,
                "research_only": True,
                "shadow_only": True,
                "execution_enabled": False,
            },
        )
        outputs.append(
            Observation(
                subject_instrument_id=symbol,
                as_of_at_us=barrier_at_us,
                payload=payload,
            )
        )
        lineages.append(lineage)
        if symbol == winner:
            winner_payload = payload
    pending: dict[str, object] = {}
    primary_terminal: dict[str, object] = {}
    if winner is not None and winner_payload is not None:
        signal_lineage = _lineage(batch, all_selected)
        if not signal_lineage:
            raise ValueError("Opening Reversal promotion lacks causal evidence")
        episode_id = _stable_id(
            "m1c-opening-reversal-episode-v1.1",
            session,
            transition_id,
            winner,
            *_string_values(stocks[winner].get("l")),
        )
        primary_pair_audit: Mapping[str, JsonValue] | None = None
        if barrier_at_us >= primary_cutoff_at_us:
            expected_keys = tuple(
                f"opening-reversal:primary:{session}:{winner}:{right}" for right in _RIGHTS
            )
            primary_pair_audit = cast(
                Mapping[str, JsonValue],
                {
                    "status": "unavailable",
                    "terminal_basis": "barrier_released_at_or_after_primary_cutoff",
                    "cutoff_at_us": primary_cutoff_at_us,
                    "barrier_available_at_us": barrier_at_us,
                    "terminal_event_id": barrier_event_id,
                    "expected_interest_keys": expected_keys,
                    "interests_issued": False,
                },
            )
            primary_terminal = {
                "s": session,
                "y": winner,
                "e": episode_id,
                "status": "unavailable",
            }
        signal_payload = cast(
            Mapping[str, JsonValue],
            {
                **winner_payload,
                "opening_reversal_episode_id": episode_id,
                "candidate_count": len(candidates),
                "selection_rule": "m1c_probability_desc_receipt_time_asc_ticker_asc",
                **({"primary_pair": primary_pair_audit} if primary_pair_audit else {}),
            },
        )
        outputs.append(
            Signal(
                subject_instrument_id=winner,
                as_of_at_us=barrier_at_us,
                payload=signal_payload,
            )
        )
        lineages.append(signal_lineage)
        if primary_pair_audit is None:
            pending = {
                "s": session,
                "y": winner,
                "e": episode_id,
                "a": barrier_at_us,
                "x": primary_cutoff_at_us,
                "d": signal_lineage,
                "b": cast(str, stocks[winner]["i"]),
                "action": winner_payload["action"],
                "transition_id": transition_id,
            }
    return outputs, lineages, pending, primary_terminal


def _emit_incomplete_rollover(
    *,
    batch: IdeaBatch,
    cohort: Mapping[str, object],
    rollover_event: MarketEvent,
    rollover_session: str,
) -> tuple[list[IdeaOutput], list[tuple[str, ...]]]:
    session = cohort.get("s")
    if not isinstance(session, str):
        return [], []
    stocks = _table(cohort.get("stocks"))
    market = _mapping(cohort.get("market"))
    selected = {rollover_event.event_id}
    if isinstance(market.get("i"), str):
        selected.add(cast(str, market["i"]))
    for record in stocks.values():
        selected.update(_string_values(record.get("l")))
    lineage = _lineage(batch, selected)
    if not lineage:
        raise ValueError("Opening Reversal rollover lacks causal evidence")
    missing_symbols = tuple(sorted(set(COHORT).difference(stocks)))
    payload = cast(
        Mapping[str, JsonValue],
        {
            "method_id": "m1c_opening_reversal_v1_1",
            "rule_hash": RULE_HASH,
            "session": session,
            "checkpoint": 6,
            "status": "unavailable",
            "reason": "checkpoint_six_cohort_incomplete_at_rollover",
            "action": "ABSTAIN",
            "eligible": False,
            "promoted": False,
            "available_stock_count": len(stocks),
            "missing_symbols": missing_symbols,
            "market_prefix_available": bool(market),
            "rollover_session": rollover_session,
            "terminal_basis": "later_session_checkpoint_observed",
            "rollover_event_id": rollover_event.event_id,
            "validated_directional_evidence": False,
            "option_profitability_claim": False,
            "research_only": True,
            "shadow_only": True,
            "execution_enabled": False,
        },
    )
    outputs: list[IdeaOutput] = [
        Observation(
            subject_instrument_id=symbol,
            as_of_at_us=_available_at(rollover_event),
            payload=payload,
        )
        for symbol in sorted(COHORT)
    ]
    return outputs, [lineage] * len(outputs)


def _emit_pending_timeout(
    *,
    batch: IdeaBatch,
    pending: Mapping[str, object],
    terminal_event: MarketEvent,
) -> tuple[Observation, tuple[str, ...]]:
    selected = set(_string_values(pending.get("d")))
    selected.add(terminal_event.event_id)
    lineage = _lineage(batch, selected)
    if not lineage:
        raise ValueError("Opening Reversal quote timeout lacks causal evidence")
    cutoff = cast(int, pending["x"])
    return (
        Observation(
            subject_instrument_id=cast(str, pending["y"]),
            as_of_at_us=_available_at(terminal_event),
            payload=cast(
                Mapping[str, JsonValue],
                {
                    "method_id": "m1c_opening_reversal_v1_1",
                    "rule_hash": RULE_HASH,
                    "session": pending.get("s"),
                    "status": "incomplete",
                    "reason": "primary_underlying_quote_unavailable",
                    "action": pending.get("action"),
                    "opening_reversal_episode_id": pending.get("e"),
                    "cutoff_at_us": cutoff,
                    "cutoff_crossing_event_id": terminal_event.event_id,
                    "terminal_basis": "primary_window_elapsed_without_reference_quote",
                    "validated_directional_evidence": False,
                    "option_profitability_claim": False,
                    "research_only": True,
                    "shadow_only": True,
                    "execution_enabled": False,
                },
            ),
        ),
        lineage,
    )


def _emit_primary_pair_terminal(
    *,
    batch: IdeaBatch,
    active: Mapping[str, object],
    terminal_event: MarketEvent,
    explicit_reason: str | None,
) -> tuple[Observation, tuple[str, ...]]:
    expected = _primary_expected_keys(active)
    statuses = _table(active.get("u"))
    identity_reason, _, identity = _primary_pair_identity(active)
    proofs = _table(active.get("z"))
    proven_keys = tuple(
        key for key in expected if isinstance(_mapping(proofs.get(key)).get("i"), str)
    )
    reason = explicit_reason or identity_reason
    if reason is None and len(proven_keys) != len(expected):
        reason = "primary_pair_quote_evidence_incomplete"
    complete = reason is None and len(proven_keys) == len(expected)
    selected = set(_string_values(active.get("d")))
    if isinstance(active.get("q"), str):
        selected.add(cast(str, active["q"]))
    quote_event_ids: dict[str, JsonValue] = {}
    for key in proven_keys:
        event_id = _mapping(proofs[key]).get("i")
        if isinstance(event_id, str):
            selected.add(event_id)
            quote_event_ids[key] = event_id
    selected.add(terminal_event.event_id)
    lineage = _lineage(batch, selected)
    if not lineage:
        raise ValueError("Opening Reversal primary-pair terminal lacks causal evidence")
    payload = cast(
        Mapping[str, JsonValue],
        {
            "method_id": "m1c_opening_reversal_v1_1",
            "rule_hash": RULE_HASH,
            "session": active.get("s"),
            "status": "complete" if complete else "incomplete",
            "reason": reason,
            "action": active.get("action"),
            "opening_reversal_episode_id": active.get("e"),
            "expected_pair_count": len(expected),
            "resolved_pair_count": sum(
                statuses.get(key, {}).get("status") == "resolved" for key in expected
            ),
            "quote_proof_count": len(proven_keys),
            "primary_pair_interest_statuses": _terminal_attribution(statuses, expected),
            "primary_pair_quote_event_ids": quote_event_ids,
            "pair_expiry": identity.get("expiry"),
            "pair_strike": identity.get("strike"),
            "call_instrument_id": identity.get("call_instrument_id"),
            "put_instrument_id": identity.get("put_instrument_id"),
            "cutoff_at_us": active.get("x"),
            "cutoff_crossing_event_id": terminal_event.event_id,
            "terminal_basis": (
                "primary_window_complete"
                if complete
                else "explicit_discovery_denial"
                if reason == "primary_pair_discovery_denied"
                else "late_discovery_evidence"
                if reason == "primary_pair_discovery_late"
                else "pair_identity_rejected"
                if reason
                in {
                    "primary_pair_identity_invalid",
                    "primary_pair_identity_mismatch",
                }
                else "primary_window_elapsed_incomplete"
            ),
            "validated_directional_evidence": False,
            "option_profitability_claim": False,
            "research_only": True,
            "shadow_only": True,
            "execution_enabled": False,
        },
    )
    return (
        Observation(
            subject_instrument_id=cast(str, active["y"]),
            as_of_at_us=_available_at(terminal_event),
            payload=payload,
        ),
        lineage,
    )


class M1COpeningReversalV1_1:
    @property
    def manifest(self) -> IdeaManifest:
        return MANIFEST

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
        if activation.universe != UNIVERSE:
            raise ValueError("Opening Reversal V1.1 requires its exact stock and VTI cohort")
        if dict(activation.parameters) != dict(PARAMETERS):
            raise ValueError("Opening Reversal V1.1 parameters must match the frozen contract")
        prefixes = tuple(
            MarketDataRequirement(
                feed_kind="bars",
                event_kind="bar_5m_session_prefix",
                instrument_id=instrument_id,
                cadence="5s",
                gaps_block=True,
                staleness_block=True,
            )
            for instrument_id in UNIVERSE
        )
        baselines = tuple(
            MarketDataRequirement(
                feed_kind="bars",
                event_kind="session_volume_baseline",
                instrument_id=instrument_id,
                cadence="5s",
                gaps_block=True,
                staleness_block=True,
            )
            for instrument_id in COHORT
        )
        quotes = tuple(
            MarketDataRequirement(
                feed_kind="quotes",
                event_kind="quote",
                instrument_id=instrument_id,
                cadence="stream",
                gaps_block=True,
                staleness_block=True,
            )
            for instrument_id in COHORT
        )
        return (*prefixes, *baselines, *quotes)

    def select_input_prefix(self, batch: IdeaBatch, state: JsonValue) -> int:
        if batch.continuation_request is not None or not batch.events:
            raise ValueError("Opening Reversal input-prefix selection requires an ordinary batch")
        prior = _mapping(state)
        pending = _mapping(prior.get("pending"))
        active = _mapping(prior.get("active"))
        pending_symbol = pending.get("y")
        pending_after = _integer(pending.get("a"))
        pending_cutoff = _integer(pending.get("x"))
        active_cutoff = _integer(active.get("x"))
        active_statuses = _table(active.get("u"))
        active_proofs = _table(active.get("z"))
        expected_keys = set(_primary_expected_keys(active))
        for receipt in batch.discovery_receipts:
            parts = _primary_interest_parts(receipt.interest_key)
            if (
                active_cutoff is None
                or parts is None
                or receipt.interest_key not in expected_keys
                or parts[:2] != (active.get("s"), active.get("y"))
            ):
                continue
            receipt_status = "late" if receipt.completed_at_us >= active_cutoff else receipt.status
            active_statuses.setdefault(
                receipt.interest_key,
                {
                    "status": receipt_status,
                    "reason": (
                        "discovery_completed_at_or_after_cutoff"
                        if receipt_status == "late"
                        else receipt.reason_code
                    ),
                    "instrument_id": receipt.instrument_id,
                    "expiry": receipt.expiry,
                    "strike": receipt.strike,
                    "right": receipt.option_right,
                    "completed": receipt.completed_at_us,
                },
            )
        selector_active = {**active, "u": active_statuses}
        identity_reason, terminal_receipt_at, _ = (
            _primary_pair_identity(selector_active) if active else (None, None, {})
        )
        if identity_reason == "primary_pair_discovery_incomplete":
            terminal_receipt_at = None
        resolved_instruments = {
            cast(str, value["instrument_id"]): cast(int, value["completed"])
            for key, value in active_statuses.items()
            if key not in active_proofs
            and value.get("status") == "resolved"
            and isinstance(value.get("instrument_id"), str)
            and _integer(value.get("completed")) is not None
        }
        for index, event in enumerate(batch.events):
            if event.event_kind in {"session_volume_baseline", "option_snapshot_capture"} or (
                event.event_kind == "bar_5m_session_prefix"
                and event.instrument_id in UNIVERSE
                and event.payload.get("bar_number") == 6
            ):
                return index + 1
            if terminal_receipt_at is not None and _available_at(event) >= terminal_receipt_at:
                return index + 1
            if (
                isinstance(pending_symbol, str)
                and pending_after is not None
                and pending_cutoff is not None
                and (
                    (
                        event.instrument_id == pending_symbol
                        and _valid_quote(
                            event,
                            after_us=pending_after,
                            before_us=pending_cutoff,
                        )
                        is not None
                    )
                    or _available_at(event) >= pending_cutoff
                )
            ):
                return index + 1
            if active_cutoff is not None and (
                _available_at(event) >= active_cutoff
                or (
                    event.instrument_id in resolved_instruments
                    and _valid_quote(
                        event,
                        after_us=resolved_instruments[event.instrument_id],
                        before_us=active_cutoff,
                    )
                    is not None
                )
            ):
                return index + 1
        return len(batch.events)

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        if batch.continuation_request is not None:
            raise ValueError("Opening Reversal V1.1 does not use continuation batches")
        prior = _mapping(state)
        baselines = _table(prior.get("baselines"))
        d1_context = _table(prior.get("d1_context"))
        d1_terminal = _table(prior.get("d1_terminal"))
        requested = {
            str(key): str(value)
            for key, value in _mapping(prior.get("requested")).items()
            if isinstance(key, str) and isinstance(value, str)
        }
        cohort = _mapping(prior.get("cohort"))
        pending = _mapping(prior.get("pending"))
        active = _mapping(prior.get("active"))
        primary_terminal = _mapping(prior.get("primary_terminal"))
        causal_horizon = _mapping(prior.get("causal_horizon"))
        initial_pending = bool(pending)
        initial_active = bool(active)
        outputs: list[IdeaOutput] = []
        lineages: list[tuple[str, ...]] = []
        interests: list[MarketDataInterest] = []

        receipts_by_instrument: dict[str, list[DiscoveryReceipt]] = {}
        for receipt in batch.discovery_receipts:
            d1_parts = _d1_interest_parts(receipt.interest_key)
            if d1_parts is not None:
                receipt_baseline_session, symbol, right = d1_parts
                baseline = baselines.get(symbol)
                baseline_cutoff = (
                    _integer(baseline.get("x"))
                    if baseline is not None and baseline.get("s") == receipt_baseline_session
                    else None
                )
                receipt_status = (
                    "late"
                    if baseline_cutoff is not None and receipt.completed_at_us >= baseline_cutoff
                    else receipt.status
                )
                terminal = d1_terminal.get(symbol)
                if terminal is None or str(terminal.get("s", "")) < receipt_baseline_session:
                    terminal = {"s": receipt_baseline_session}
                if terminal.get("s") == receipt_baseline_session and terminal.get(right) not in {
                    "captured",
                    "denied",
                    "late",
                    "window_elapsed",
                }:
                    terminal[right] = receipt_status
                    terminal[f"{right}_k"] = receipt.interest_key
                    terminal[f"{right}_completed"] = receipt.completed_at_us
                    terminal_reason = (
                        "discovery_completed_at_or_after_cutoff"
                        if receipt_status == "late"
                        else receipt.reason_code
                    )
                    if terminal_reason is not None:
                        terminal[f"{right}_r"] = terminal_reason
                    d1_terminal[symbol] = terminal
                elif (
                    terminal.get("s") == receipt_baseline_session
                    and terminal.get(right) == "window_elapsed"
                ):
                    terminal[f"{right}_k"] = receipt.interest_key
                    terminal[f"{right}_completed"] = receipt.completed_at_us
                    d1_terminal[symbol] = terminal
            primary_parts = _primary_interest_parts(receipt.interest_key)
            if (
                primary_parts is not None
                and active.get("e") is not None
                and primary_parts[:2] == (active.get("s"), active.get("y"))
                and receipt.interest_key in set(_primary_expected_keys(active))
                and (active_cutoff := _integer(active.get("x"))) is not None
            ):
                statuses = _table(active.get("u"))
                receipt_status = (
                    "late" if receipt.completed_at_us >= active_cutoff else receipt.status
                )
                statuses.setdefault(
                    receipt.interest_key,
                    {
                        "status": receipt_status,
                        "reason": (
                            "discovery_completed_at_or_after_cutoff"
                            if receipt_status == "late"
                            else receipt.reason_code
                        ),
                        "instrument_id": receipt.instrument_id,
                        "expiry": receipt.expiry,
                        "strike": receipt.strike,
                        "right": receipt.option_right,
                        "completed": receipt.completed_at_us,
                    },
                )
                active["u"] = statuses
            if receipt.status == "resolved" and receipt.instrument_id is not None:
                receipts_by_instrument.setdefault(receipt.instrument_id, []).append(receipt)

        pending_cutoff = _integer(pending.get("x")) if initial_pending else None
        pending_terminal_event = (
            None if pending_cutoff is None else _first_event_at_or_after(batch, pending_cutoff)
        )
        active_cutoff = _integer(active.get("x")) if initial_active else None
        active_reason: str | None = None
        active_terminal_at = active_cutoff
        if initial_active:
            identity_reason, identity_terminal_at, _ = _primary_pair_identity(active)
            if (
                identity_reason != "primary_pair_discovery_incomplete"
                and identity_terminal_at is not None
            ):
                active_reason = identity_reason
                active_terminal_at = (
                    identity_terminal_at
                    if active_cutoff is None
                    else min(active_cutoff, identity_terminal_at)
                )
        active_terminal_event = (
            None
            if active_terminal_at is None
            else _first_event_at_or_after(batch, active_terminal_at)
        )
        active_evidence_event_ids: set[str] = set()
        for event in batch.events:
            causal_horizon = _advance_causal_horizon(causal_horizon, event)
            _terminalize_elapsed_d1_windows(
                baselines=baselines,
                d1_context=d1_context,
                d1_terminal=d1_terminal,
                causal_horizon=causal_horizon,
            )
            if (
                active_terminal_event is not None
                and event.event_id == active_terminal_event.event_id
            ):
                break
            active_evidence_event_ids.add(event.event_id)

        for event in batch.events:
            if (
                initial_pending
                and pending_terminal_event is not None
                and event.event_id == pending_terminal_event.event_id
            ):
                observation, lineage = _emit_pending_timeout(
                    batch=batch,
                    pending=pending,
                    terminal_event=event,
                )
                outputs.append(observation)
                lineages.append(lineage)
                primary_terminal = {
                    "s": pending.get("s"),
                    "y": pending.get("y"),
                    "e": pending.get("e"),
                    "status": "incomplete",
                }
                pending = {}
                initial_pending = False
            if (
                initial_active
                and active_terminal_event is not None
                and event.event_id == active_terminal_event.event_id
            ):
                observation, lineage = _emit_primary_pair_terminal(
                    batch=batch,
                    active=active,
                    terminal_event=event,
                    explicit_reason=active_reason,
                )
                outputs.append(observation)
                lineages.append(lineage)
                primary_terminal = {
                    "s": active.get("s"),
                    "y": active.get("y"),
                    "e": active.get("e"),
                    "status": observation.payload["status"],
                }
                active = {}
                initial_active = False
            if (
                event.event_kind == "quote"
                and active
                and event.event_id in active_evidence_event_ids
            ):
                statuses = _table(active.get("u"))
                proofs = _table(active.get("z"))
                cutoff = _integer(active.get("x"))
                if cutoff is not None:
                    for key in _primary_expected_keys(active):
                        status = statuses.get(key)
                        if status is None or key in proofs:
                            continue
                        completed = _integer(status.get("completed"))
                        if (
                            status.get("status") != "resolved"
                            or status.get("instrument_id") != event.instrument_id
                            or completed is None
                            or _valid_quote(
                                event,
                                after_us=completed,
                                before_us=cutoff,
                            )
                            is None
                        ):
                            continue
                        proofs[key] = {
                            "i": event.event_id,
                            "a": _available_at(event),
                            "bid": event.payload.get("bid"),
                            "ask": event.payload.get("ask"),
                        }
                    active["z"] = proofs
                continue
            if event.event_kind == "session_volume_baseline" and event.instrument_id in COHORT:
                baseline_session = event.payload.get("session")
                closes = event.payload.get("session_closes")
                count = event.payload.get("complete_session_count")
                if (
                    not isinstance(baseline_session, str)
                    or not isinstance(closes, list | tuple)
                    or not closes
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                ):
                    continue
                close = _number(closes[-1])
                if close is None or close <= 0.0:
                    continue
                identity = (baseline_session, event.event_at_us, event.event_id)
                existing = baselines.get(event.instrument_id)
                existing_identity = (
                    (
                        str(existing.get("s", "")),
                        cast(int, existing.get("t", -1)),
                        str(existing.get("i", "")),
                    )
                    if existing is not None
                    else None
                )
                if existing_identity is not None and identity <= existing_identity:
                    continue
                if existing is None or existing.get("s") != baseline_session:
                    d1_context.pop(event.instrument_id, None)
                    d1_terminal.pop(event.instrument_id, None)
                baselines[event.instrument_id] = {
                    "i": event.event_id,
                    "s": baseline_session,
                    "c": close,
                    "v": _number(event.payload.get("realised_volatility_20d")),
                    "t": event.event_at_us,
                    "a": _available_at(event),
                    "x": event.event_at_us + _D1_WINDOW_US,
                    "k": count,
                }
                _terminalize_elapsed_d1_windows(
                    baselines=baselines,
                    d1_context=d1_context,
                    d1_terminal=d1_terminal,
                    causal_horizon=causal_horizon,
                )
                cutoff = event.event_at_us + _D1_WINDOW_US
                horizon_at_us = _integer(causal_horizon.get("a"))
                if (
                    requested.get(event.instrument_id) != baseline_session
                    and _available_at(event) < cutoff
                    and (horizon_at_us is None or horizon_at_us < cutoff)
                ):
                    for right in _RIGHTS:
                        interests.append(
                            MarketDataInterest(
                                interest_key=(
                                    "opening-reversal:m1c:d1:"
                                    f"{baseline_session}:{event.instrument_id}:{right}"
                                ),
                                underlying_instrument_id=event.instrument_id,
                                minimum_days_to_expiry=7,
                                maximum_days_to_expiry=45,
                                option_right=right,
                                strike_offset=0,
                                reference_price=close,
                                cadence="snapshot",
                                as_of_at_us=event.event_at_us,
                                expires_at_us=cutoff,
                                required=True,
                                priority=100,
                                input_event_id=event.event_id,
                            )
                        )
                    requested[event.instrument_id] = baseline_session
                continue

            if event.event_kind == "option_snapshot_capture":
                available = _available_at(event)
                candidates = [
                    receipt
                    for receipt in receipts_by_instrument.get(event.instrument_id, ())
                    if receipt.completed_at_us <= available
                    and _d1_interest_parts(receipt.interest_key) is not None
                ]
                if not candidates:
                    continue
                receipt = max(candidates, key=lambda item: (item.completed_at_us, item.receipt_id))
                parts = _d1_interest_parts(receipt.interest_key)
                assert parts is not None
                baseline_session, symbol, right = parts
                baseline = baselines.get(symbol)
                if baseline is None or baseline.get("s") != baseline_session:
                    continue
                cutoff = _integer(baseline.get("x"))
                if cutoff is None:
                    continue
                terminal = d1_terminal.get(symbol)
                if terminal is None or terminal.get("s") != baseline_session:
                    terminal = {"s": baseline_session}
                terminal_status = terminal.get(right)
                if terminal_status in {"captured", "denied"}:
                    continue
                if terminal_status == "late":
                    terminal[f"{right}_i"] = event.event_id
                    terminal[f"{right}_a"] = available
                    terminal[f"{right}_x"] = cutoff
                    if available >= cutoff:
                        terminal[f"{right}_cutoff_i"] = event.event_id
                        terminal[f"{right}_cutoff_a"] = available
                    d1_terminal[symbol] = terminal
                    continue
                if available >= cutoff or terminal.get(right) == "window_elapsed":
                    cutoff_event_id = terminal.get(f"{right}_cutoff_i")
                    cutoff_event_at = _integer(terminal.get(f"{right}_cutoff_a"))
                    if not isinstance(cutoff_event_id, str) or cutoff_event_at is None:
                        cutoff_event_id = event.event_id
                        cutoff_event_at = available
                    terminal[right] = "late"
                    terminal[f"{right}_i"] = event.event_id
                    terminal[f"{right}_a"] = available
                    terminal[f"{right}_x"] = cutoff
                    terminal[f"{right}_cutoff_i"] = cutoff_event_id
                    terminal[f"{right}_cutoff_a"] = cutoff_event_at
                    d1_terminal[symbol] = terminal
                    continue
                context = d1_context.get(symbol)
                if context is None or context.get("s") != baseline_session:
                    context = {"s": baseline_session}
                context[right] = {
                    "i": event.event_id,
                    "t": event.event_at_us,
                    "a": available,
                    "source_completeness": event.payload.get("source_completeness"),
                    "bid": event.payload.get("bid"),
                    "ask": event.payload.get("ask"),
                    "model_implied_volatility": event.payload.get("model_implied_volatility"),
                    "open_interest": event.payload.get("open_interest"),
                    "option_right": right,
                    "expiry": receipt.expiry,
                    "strike": receipt.strike,
                }
                d1_context[symbol] = context
                terminal[right] = "captured"
                terminal[f"{right}_k"] = receipt.interest_key
                terminal[f"{right}_a"] = available
                terminal[f"{right}_x"] = cutoff
                d1_terminal[symbol] = terminal
                continue

            if (
                event.event_kind == "bar_5m_session_prefix"
                and event.instrument_id in UNIVERSE
                and event.payload.get("bar_number") == 6
            ):
                session = event.payload.get("session")
                if not isinstance(session, str):
                    continue
                cohort_session = cohort.get("s")
                if isinstance(cohort_session, str) and session > cohort_session:
                    if cohort.get("emitted") is not True:
                        rollover_outputs, rollover_lineages = _emit_incomplete_rollover(
                            batch=batch,
                            cohort=cohort,
                            rollover_event=event,
                            rollover_session=session,
                        )
                        outputs.extend(rollover_outputs)
                        lineages.extend(rollover_lineages)
                    cohort = {"s": session, "stocks": {}, "market": {}, "emitted": False}
                elif not isinstance(cohort_session, str):
                    cohort = {"s": session, "stocks": {}, "market": {}, "emitted": False}
                elif session < cohort_session or cohort.get("emitted") is True:
                    continue
                stocks = _table(cohort.get("stocks"))
                if event.instrument_id == "VTI":
                    existing_market = _mapping(cohort.get("market"))
                    if existing_market and existing_market.get("i") != event.event_id:
                        raise ValueError("conflicting VTI checkpoint-six receipt")
                    cohort["market"] = _market_record(event)
                else:
                    existing_stock = stocks.get(event.instrument_id)
                    if existing_stock is not None and existing_stock.get("i") != event.event_id:
                        raise ValueError("conflicting stock checkpoint-six receipt")
                    if existing_stock is None:
                        stocks[event.instrument_id] = _stock_record(
                            event=event,
                            symbol=event.instrument_id,
                            session=session,
                            baseline=baselines.get(event.instrument_id),
                            context=d1_context.get(event.instrument_id),
                            terminal=d1_terminal.get(event.instrument_id),
                        )
                    cohort["stocks"] = stocks
                continue

            if pending and event.instrument_id == pending.get("y"):
                after = _integer(pending.get("a"))
                cutoff = _integer(pending.get("x"))
                if after is not None and cutoff is not None:
                    reference = _valid_quote(event, after_us=after, before_us=cutoff)
                    if reference is not None:
                        pending_session = cast(str, pending["s"])
                        symbol = cast(str, pending["y"])
                        episode_id = cast(str, pending["e"])
                        keys: list[str] = []
                        for right in _RIGHTS:
                            key = f"opening-reversal:primary:{pending_session}:{symbol}:{right}"
                            keys.append(key)
                            interests.append(
                                MarketDataInterest(
                                    interest_key=key,
                                    underlying_instrument_id=symbol,
                                    minimum_days_to_expiry=1,
                                    maximum_days_to_expiry=1,
                                    option_right=right,
                                    strike_offset=0,
                                    reference_price=reference,
                                    cadence="stream",
                                    as_of_at_us=_available_at(event),
                                    expires_at_us=cutoff,
                                    required=True,
                                    priority=300,
                                    input_event_id=event.event_id,
                                )
                            )
                        active = {
                            **pending,
                            "g": "streams",
                            "k": tuple(keys),
                            "q": event.event_id,
                            "u": {},
                            "z": {},
                            "episode_id": episode_id,
                        }
                        pending = {}

        stocks = _table(cohort.get("stocks"))
        market = _mapping(cohort.get("market"))
        if cohort.get("emitted") is not True and set(stocks) == set(COHORT) and market:
            (
                cohort_outputs,
                cohort_lineages,
                next_pending,
                next_primary_terminal,
            ) = _emit_complete_cohort(
                batch=batch,
                cohort=cohort,
                causal_horizon=causal_horizon,
            )
            outputs.extend(cohort_outputs)
            lineages.extend(cohort_lineages)
            cohort = {"s": cohort.get("s"), "emitted": True}
            pending = next_pending
            primary_terminal = next_primary_terminal

        retained_values: set[str] = set()
        for item in baselines.values():
            if isinstance(item.get("i"), str):
                retained_values.add(cast(str, item["i"]))
        for item in d1_context.values():
            for right in _RIGHTS:
                capture = _mapping(item.get(right))
                if isinstance(capture.get("i"), str):
                    retained_values.add(cast(str, capture["i"]))
        for item in d1_terminal.values():
            for right in _RIGHTS:
                for suffix in ("i", "cutoff_i"):
                    key = f"{right}_{suffix}"
                    if isinstance(item.get(key), str):
                        retained_values.add(cast(str, item[key]))
        if cohort.get("emitted") is not True:
            market_item = _mapping(cohort.get("market"))
            if isinstance(market_item.get("i"), str):
                retained_values.add(cast(str, market_item["i"]))
            for item in _table(cohort.get("stocks")).values():
                retained_values.update(_string_values(item.get("l")))
        for lifecycle in (pending, active):
            retained_values.update(_string_values(lifecycle.get("d")))
            if isinstance(lifecycle.get("q"), str):
                retained_values.add(cast(str, lifecycle["q"]))
            for proof in _table(lifecycle.get("z")).values():
                if isinstance(proof.get("i"), str):
                    retained_values.add(cast(str, proof["i"]))
        if isinstance(causal_horizon.get("i"), str):
            retained_values.add(cast(str, causal_horizon["i"]))
        retained = _lineage(batch, retained_values)
        latest_state = cast(
            JsonValue,
            {
                "schema_version": 1,
                "baselines": baselines,
                "d1_context": d1_context,
                "d1_terminal": d1_terminal,
                "requested": requested,
                "cohort": cohort,
                "pending": pending,
                "active": active,
                "primary_terminal": primary_terminal,
                "causal_horizon": causal_horizon,
            },
        )
        return IdeaEvaluation(
            state=latest_state,
            outputs=tuple(outputs),
            retained_input_event_ids=retained,
            output_input_event_ids=tuple(lineages),
            interests=tuple(interests),
        )


def create_plugin() -> M1COpeningReversalV1_1:
    return M1COpeningReversalV1_1()


__all__ = [
    "COHORT",
    "MANIFEST",
    "PARAMETERS",
    "UNIVERSE",
    "M1COpeningReversalV1_1",
    "create_plugin",
]
