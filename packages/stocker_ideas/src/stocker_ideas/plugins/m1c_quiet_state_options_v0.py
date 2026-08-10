"""Frozen M1C Quiet-State Options V0 over generic market-data evidence."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Literal, cast

from stocker_ideas.plugins.frozen_m1c_v0 import (
    CHECKPOINTS,
    COHORT,
    build_front_options_context,
    build_group_i_for_symbol,
    score_m1c,
)
from stocker_ideas.plugins.m1c_quiet_state_v0 import (
    BOTTOM_10_THRESHOLD,
    OptionPanelContract,
    classify_quiet_state,
    select_defined_risk_structures,
)
from stocker_runtime.domain import (
    IdeaOutput,
    JsonValue,
    MarketEvent,
    Observation,
    OutputKind,
    ProposedTrade,
    ProposedTradeLeg,
    RuntimeMode,
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

_D1_WINDOW_US = 30 * 60 * 1_000_000
_PANEL_WINDOW_US = 60 * 60 * 1_000_000
_DTE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0DTE", 0, 0),
    ("1DTE", 1, 1),
    ("3_TO_5_DTE", 3, 5),
)
_RIGHTS: tuple[Literal["call", "put"], ...] = ("call", "put")
_OFFSETS = tuple(range(-4, 5))

PARAMETERS: Mapping[str, JsonValue] = {
    "checkpoints": CHECKPOINTS,
    "quiet_threshold": BOTTOM_10_THRESHOLD,
    "minimum_episode_spacing_minutes": 30,
    "d1_option_minimum_days_to_expiry": 7,
    "d1_option_maximum_days_to_expiry": 45,
    "d1_snapshot_lifetime_minutes": 30,
    "entry_dte_buckets": ((0, 0), (1, 1), (3, 5)),
    "entry_strike_offsets": _OFFSETS,
    "option_panel_lifetime_minutes": 60,
    "maximum_active_episodes": 1,
}

MANIFEST = IdeaManifest(
    api_version=1,
    idea_id="m1c_quiet_state_options",
    idea_version="v0",
    display_name="M1C Quiet State Options V0",
    description="Frozen bottom-tail M1C observations and defined-risk virtual option proposals.",
    modes=(RuntimeMode.PROSPECTIVE_RECORD, RuntimeMode.SHADOW),
    output_kinds=(OutputKind.OBSERVATION, OutputKind.PROPOSED_TRADE),
    parameter_schema_version="m1c-quiet-state-options-v0",
    parameter_schema={
        "type": "object",
        "additionalProperties": False,
        "required": tuple(PARAMETERS),
        "properties": {
            "checkpoints": {
                "type": "array",
                "items": {"type": "integer", "enum": CHECKPOINTS},
                "enum": (CHECKPOINTS,),
            },
            "quiet_threshold": {"type": "number", "enum": (BOTTOM_10_THRESHOLD,)},
            "minimum_episode_spacing_minutes": {"type": "integer", "enum": (30,)},
            "d1_option_minimum_days_to_expiry": {"type": "integer", "enum": (7,)},
            "d1_option_maximum_days_to_expiry": {"type": "integer", "enum": (45,)},
            "d1_snapshot_lifetime_minutes": {"type": "integer", "enum": (30,)},
            "entry_dte_buckets": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "enum": (((0, 0), (1, 1), (3, 5)),),
            },
            "entry_strike_offsets": {
                "type": "array",
                "items": {"type": "integer", "enum": _OFFSETS},
                "enum": (_OFFSETS,),
            },
            "option_panel_lifetime_minutes": {"type": "integer", "enum": (60,)},
            "maximum_active_episodes": {"type": "integer", "enum": (1,)},
        },
    },
    maximum_state_bytes=65_536,
    maximum_outputs_per_batch=64,
    maximum_interests_per_batch=64,
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


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def _d1_interest_parts(key: str) -> tuple[str, str, str] | None:
    parts = key.split(":")
    if len(parts) != 6 or parts[:3] != ["quiet", "m1c", "d1"] or parts[5] not in _RIGHTS:
        return None
    return parts[3], parts[4], parts[5]


def _entry_interest_parts(key: str) -> tuple[str, str, int, str] | None:
    parts = key.split(":")
    if len(parts) != 6 or parts[0] != "quiet" or parts[2] != "entry":
        return None
    if parts[3] not in {item[0] for item in _DTE_BUCKETS} or parts[5] not in _RIGHTS:
        return None
    try:
        offset = int(parts[4])
    except ValueError:
        return None
    if offset not in _OFFSETS:
        return None
    return parts[1], parts[3], offset, parts[5]


def _stream_interest_parts(key: str) -> tuple[str, str, int, str] | None:
    parts = key.split(":")
    if len(parts) != 6 or parts[0] != "quiet" or parts[2] != "stream":
        return None
    if parts[3] not in {item[0] for item in _DTE_BUCKETS} or parts[5] not in _RIGHTS:
        return None
    try:
        offset = int(parts[4])
    except ValueError:
        return None
    if offset not in _OFFSETS:
        return None
    return parts[1], parts[3], offset, parts[5]


def _episode_id(candidate: Mapping[str, object]) -> str:
    identity = "|".join(
        (
            "m1c-quiet-state-options-v0",
            str(candidate["s"]),
            str(candidate["n"]),
            str(candidate["y"]),
            str(candidate["i"]),
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def _base_payload(**values: JsonValue) -> Mapping[str, JsonValue]:
    return {
        "method_id": "m1c_quiet_state_options_v0",
        "frozen_version": "m1c-quiet-state-option-selection-v0",
        "research_only": True,
        "execution_enabled": False,
        "shadow_only": True,
        "original_decision": "blocked_insufficient_low_tail_support",
        **values,
    }


def _available_at(event: MarketEvent) -> int:
    return max(event.event_at_us, event.received_at_us)


def _first_cutoff_event(batch: IdeaBatch, cutoff_at_us: int) -> MarketEvent | None:
    return next((event for event in batch.events if _available_at(event) >= cutoff_at_us), None)


def _events_before_cutoff(batch: IdeaBatch, cutoff_at_us: int) -> tuple[MarketEvent, ...]:
    output: list[MarketEvent] = []
    for event in batch.events:
        if _available_at(event) >= cutoff_at_us:
            break
        output.append(event)
    return tuple(output)


def _terminal_attribution(
    statuses: Mapping[str, object],
    expected: tuple[str, ...],
) -> Mapping[str, JsonValue]:
    return {
        key: {
            "status": str(_mapping(statuses.get(key)).get("status", "missing")),
            "reason": (
                str(_mapping(statuses.get(key))["reason"])
                if isinstance(_mapping(statuses.get(key)).get("reason"), str)
                else None
            ),
            "completed_at_us": _integer(_mapping(statuses.get(key)).get("completed")),
        }
        for key in expected
    }


def _d1_terminal_payload(
    *,
    symbol: str,
    session: str,
    cutoff_at_us: int,
    cohort_available_at_us: int,
    terminal: Mapping[str, object],
) -> Mapping[str, JsonValue]:
    statuses: dict[str, str] = {right: str(terminal.get(right, "missing")) for right in _RIGHTS}
    reasons: dict[str, str] = {
        right: str(terminal[f"{right}_r"])
        for right in _RIGHTS
        if isinstance(terminal.get(f"{right}_r"), str)
    }
    evidence_available: dict[str, int] = {
        right: cast(int, terminal[f"{right}_a"])
        for right in _RIGHTS
        if _integer(terminal.get(f"{right}_a")) is not None
    }
    completed: dict[str, int] = {
        right: cast(int, terminal[f"{right}_completed"])
        for right in _RIGHTS
        if _integer(terminal.get(f"{right}_completed")) is not None
    }
    observed = set(statuses.values())
    if len(observed) > 1:
        basis = "mixed_terminal_option_evidence"
    elif observed == {"denied"}:
        basis = "explicit_discovery_denial"
    elif observed == {"late"}:
        basis = "late_capture"
    elif observed == {"window_elapsed"}:
        basis = "causal_window_elapsed_without_capture"
    else:
        basis = "captured_evidence_invalid"
    return {
        "interest_keys": tuple(f"quiet:m1c:d1:{session}:{symbol}:{right}" for right in _RIGHTS),
        "terminal_statuses": statuses,
        "denial_reasons": reasons,
        "interest_completed_at_us": completed,
        "evidence_available_at_us": evidence_available,
        "cutoff_at_us": cutoff_at_us,
        "cohort_available_at_us": cohort_available_at_us,
        "terminal_basis": basis,
    }


def _valid_quote(event: MarketEvent, *, after_us: int, before_us: int) -> float | None:
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


def _entry_interests(
    *,
    episode: Mapping[str, object],
    quote: MarketEvent,
    reference_price: float,
) -> tuple[MarketDataInterest, ...]:
    episode_id = str(episode["e"])
    deadline = cast(int, episode["x"])
    available = max(quote.event_at_us, quote.received_at_us)
    output: list[MarketDataInterest] = []
    for bucket, minimum, maximum in _DTE_BUCKETS:
        for offset in _OFFSETS:
            for right in _RIGHTS:
                output.append(
                    MarketDataInterest(
                        interest_key=(f"quiet:{episode_id}:entry:{bucket}:{offset:+d}:{right}"),
                        underlying_instrument_id=str(episode["y"]),
                        minimum_days_to_expiry=minimum,
                        maximum_days_to_expiry=maximum,
                        option_right=right,
                        strike_offset=offset,
                        reference_price=reference_price,
                        cadence="snapshot",
                        as_of_at_us=available,
                        expires_at_us=deadline,
                        required=True,
                        priority=200,
                        input_event_id=quote.event_id,
                    )
                )
    return tuple(output)


def _panel_contracts(
    active: Mapping[str, object],
) -> tuple[tuple[OptionPanelContract, ...], str | None]:
    raw_captures = _mapping(active.get("z"))
    expected = active.get("k")
    if not isinstance(expected, list | tuple) or len(expected) != 54:
        return (), "option_panel_identity_mismatch"
    if any(not isinstance(key, str) or key not in raw_captures for key in expected):
        return (), "option_panel_capture_missing"
    contracts: list[OptionPanelContract] = []
    quote_available = _integer(active.get("a"))
    deadline = _integer(active.get("x"))
    if quote_available is None or deadline is None:
        return (), "option_panel_identity_mismatch"
    for key in expected:
        raw = _mapping(raw_captures[cast(str, key)])
        if raw.get("complete") is not True:
            return (), "option_panel_capture_incomplete"
        parsed = _entry_interest_parts(cast(str, key))
        bid = _number(raw.get("bid"))
        ask = _number(raw.get("ask"))
        strike = _number(raw.get("strike"))
        available = _integer(raw.get("available"))
        if (
            parsed is None
            or bid is None
            or ask is None
            or strike is None
            or available is None
            or not quote_available <= available < deadline
            or bid < 0.0
            or ask < bid
            or strike <= 0.0
            or not isinstance(raw.get("event_id"), str)
            or not isinstance(raw.get("instrument_id"), str)
            or not isinstance(raw.get("expiry"), str)
            or not isinstance(raw.get("multiplier"), str)
            or not cast(str, raw["multiplier"]).isdigit()
            or int(cast(str, raw["multiplier"])) <= 0
        ):
            return (), "option_panel_quote_or_identity_invalid"
        episode_id, bucket, offset, right = parsed
        if episode_id != active.get("e") or raw.get("right") != right:
            return (), "option_panel_identity_mismatch"
        try:
            contracts.append(
                OptionPanelContract(
                    event_id=cast(str, raw["event_id"]),
                    interest_key=cast(str, key),
                    instrument_id=cast(str, raw["instrument_id"]),
                    bucket=bucket,
                    offset=offset,
                    right=cast(Literal["call", "put"], right),
                    expiry=cast(str, raw["expiry"]),
                    strike=strike,
                    multiplier=cast(str, raw["multiplier"]),
                    bid=bid,
                    ask=ask,
                    delta=_number(raw.get("delta")),
                )
            )
        except ValueError:
            return (), "option_panel_quote_or_identity_invalid"
    if len({item.instrument_id for item in contracts}) != 54:
        return (), "option_panel_identity_mismatch"
    try:
        session_date = date.fromisoformat(cast(str, active["s"]))
    except (KeyError, TypeError, ValueError):
        return (), "option_panel_identity_mismatch"
    for bucket, minimum, maximum in _DTE_BUCKETS:
        observed = tuple(item for item in contracts if item.bucket == bucket)
        if (
            len(observed) != 18
            or len({item.expiry for item in observed}) != 1
            or len({item.multiplier for item in observed}) != 1
        ):
            return (), "option_panel_identity_mismatch"
        try:
            expiry_date = datetime.strptime(observed[0].expiry, "%Y%m%d").date()
        except ValueError:
            return (), "option_panel_identity_mismatch"
        if not minimum <= (expiry_date - session_date).days <= maximum:
            return (), "option_panel_identity_mismatch"
        strikes: list[float] = []
        for offset in _OFFSETS:
            pair = tuple(item for item in observed if item.offset == offset)
            if (
                len(pair) != 2
                or {item.right for item in pair} != set(_RIGHTS)
                or len({item.strike for item in pair}) != 1
            ):
                return (), "option_panel_identity_mismatch"
            strikes.append(pair[0].strike)
        if any(left >= right for left, right in zip(strikes, strikes[1:], strict=False)):
            return (), "option_panel_identity_mismatch"
    return tuple(contracts), None


def _panel_lineage(
    batch: IdeaBatch,
    active: Mapping[str, object],
    *,
    extra_event_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    proof = active.get("d")
    selected = (
        {value for value in proof if isinstance(value, str)}
        if isinstance(proof, list | tuple)
        else set()
    )
    for name in ("i", "q"):
        value = active.get(name)
        if isinstance(value, str):
            selected.add(value)
    for raw in _mapping(active.get("z")).values():
        event_id = _mapping(raw).get("event_id")
        if isinstance(event_id, str):
            selected.add(event_id)
    selected.update(extra_event_ids)
    return _lineage(batch, selected)


def _complete_panel(
    *,
    batch: IdeaBatch,
    active: Mapping[str, object],
    contracts: tuple[OptionPanelContract, ...],
) -> tuple[list[IdeaOutput], list[tuple[str, ...]], list[MarketDataInterest], dict[str, object]]:
    lineage = _panel_lineage(batch, active)
    as_of = max(cast(int, _mapping(raw)["available"]) for raw in _mapping(active.get("z")).values())
    symbol = cast(str, active["y"])
    episode_id = cast(str, active["e"])
    reference = cast(float, _number(active["r"]))
    deadline = cast(int, active["x"])
    outputs: list[IdeaOutput] = []
    lineages: list[tuple[str, ...]] = []
    selected_streams: dict[str, OptionPanelContract] = {}
    bucket_ranges = {name: (minimum, maximum) for name, minimum, maximum in _DTE_BUCKETS}
    for bucket, _minimum, _maximum in _DTE_BUCKETS:
        observed = tuple(item for item in contracts if item.bucket == bucket)
        atm_call = next(item for item in observed if item.offset == 0 and item.right == "call")
        atm_put = next(item for item in observed if item.offset == 0 and item.right == "put")
        for structure_type, items in (
            ("ATM_CALL", (atm_call,)),
            ("ATM_PUT", (atm_put,)),
            ("ATM_STRADDLE", (atm_call, atm_put)),
        ):
            outputs.append(
                Observation(
                    subject_instrument_id=items[0].instrument_id,
                    as_of_at_us=as_of,
                    payload=_base_payload(
                        status="complete",
                        quiet_episode_id=episode_id,
                        session=cast(str, active["s"]),
                        checkpoint=cast(int, active["n"]),
                        probability=cast(float, active["p"]),
                        dte_bucket=bucket,
                        structure_type=structure_type,
                        long_premium_observation=True,
                        contract_multiplier=items[0].multiplier,
                        instrument_ids=tuple(item.instrument_id for item in items),
                    ),
                )
            )
            lineages.append(lineage)
        for attempt in select_defined_risk_structures(
            contracts=observed,
            underlying_reference_price=reference,
        ):
            payload = _base_payload(
                status="available" if attempt.available else "unavailable",
                reason=attempt.reason,
                quiet_episode_id=episode_id,
                session=cast(str, active["s"]),
                checkpoint=cast(int, active["n"]),
                probability=cast(float, active["p"]),
                dte_bucket=bucket,
                structure_type=attempt.structure_type,
                defined_risk=True,
                virtual=True,
                contract_multiplier=observed[0].multiplier,
                comparison_only=attempt.structure_type
                in {"CALL_CREDIT_SPREAD", "PUT_CREDIT_SPREAD"},
                opening_credit=attempt.opening_credit,
            )
            if not attempt.available:
                outputs.append(
                    Observation(
                        subject_instrument_id=symbol,
                        as_of_at_us=as_of,
                        payload=payload,
                    )
                )
                lineages.append(lineage)
                continue
            legs = tuple(
                ProposedTradeLeg(
                    instrument_id=leg.contract.instrument_id,
                    action="sell" if leg.side == "short" else "buy",
                    target="short" if leg.side == "short" else "long",
                    quantity_value=1.0,
                    currency="USD",
                    price_hint=(leg.contract.bid if leg.side == "short" else leg.contract.ask),
                )
                for leg in attempt.legs
            )
            outputs.append(
                ProposedTrade(
                    subject_instrument_id=legs[0].instrument_id,
                    as_of_at_us=as_of,
                    payload=payload,
                    legs=legs,
                )
            )
            lineages.append(lineage)
            for leg in attempt.legs:
                selected_streams.setdefault(leg.contract.instrument_id, leg.contract)
    interests: list[MarketDataInterest] = []
    stream_instruments: dict[str, str] = {}
    for contract in sorted(
        selected_streams.values(),
        key=lambda item: (item.bucket, item.offset, item.right, item.instrument_id),
    ):
        minimum, maximum = bucket_ranges[contract.bucket]
        interest = MarketDataInterest(
            interest_key=(
                f"quiet:{episode_id}:stream:{contract.bucket}:{contract.offset:+d}:{contract.right}"
            ),
            underlying_instrument_id=symbol,
            minimum_days_to_expiry=minimum,
            maximum_days_to_expiry=maximum,
            option_right=contract.right,
            strike_offset=contract.offset,
            reference_price=reference,
            cadence="stream",
            as_of_at_us=as_of,
            expires_at_us=deadline,
            required=True,
            priority=250,
            input_event_id=contract.event_id,
        )
        interests.append(interest)
        stream_instruments[interest.interest_key] = contract.instrument_id
    if len(interests) > 30:
        raise ValueError("Quiet selected-leg stream bound exceeded")
    stream_state = {key: value for key, value in active.items() if key not in {"k", "z", "u"}}
    stream_state.update(
        {
            "g": "streams",
            "k": tuple(item.interest_key for item in interests),
            "m": stream_instruments,
            "u": {},
            "z": {},
            "d": lineage,
        }
    )
    return outputs, lineages, interests, stream_state


def _advance_episode(
    *,
    batch: IdeaBatch,
    pending: dict[str, object],
    active: dict[str, object],
    existing_interest_count: int,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[IdeaOutput],
    list[tuple[str, ...]],
    list[MarketDataInterest],
]:
    outputs: list[IdeaOutput] = []
    lineages: list[tuple[str, ...]] = []
    interests: list[MarketDataInterest] = []
    initial_active_stage = active.get("g")
    started_pending = bool(pending and not active)

    if started_pending:
        trigger = _integer(pending.get("t"))
        deadline = _integer(pending.get("x"))
        symbol_value = pending.get("y")
        if trigger is not None and deadline is not None and isinstance(symbol_value, str):
            valid = tuple(
                (event, price)
                for event in _events_before_cutoff(batch, deadline)
                if event.instrument_id == symbol_value
                and (price := _valid_quote(event, after_us=trigger, before_us=deadline)) is not None
            )
            if valid:
                quote, reference = min(
                    valid,
                    key=lambda item: (_available_at(item[0]), item[0].event_id),
                )
                if existing_interest_count <= 10:
                    entry = _entry_interests(
                        episode=pending,
                        quote=quote,
                        reference_price=reference,
                    )
                    interests.extend(entry)
                    active = {
                        **pending,
                        "g": "entry",
                        "q": quote.event_id,
                        "r": reference,
                        "a": _available_at(quote),
                        "k": tuple(item.interest_key for item in entry),
                        "z": {},
                    }
                    pending = {}
                else:
                    lineage = _panel_lineage(
                        batch,
                        pending,
                        extra_event_ids=(quote.event_id,),
                    )
                    if lineage:
                        outputs.append(
                            Observation(
                                subject_instrument_id=symbol_value,
                                as_of_at_us=_available_at(quote),
                                payload=_base_payload(
                                    status="incomplete",
                                    reason="complete_panel_capacity_unavailable",
                                    quiet_episode_id=cast(str, pending["e"]),
                                ),
                            )
                        )
                        lineages.append(lineage)
                    pending = {}
            elif (cutoff_event := _first_cutoff_event(batch, deadline)) is not None:
                lineage = _panel_lineage(
                    batch,
                    pending,
                    extra_event_ids=(cutoff_event.event_id,),
                )
                if lineage:
                    outputs.append(
                        Observation(
                            subject_instrument_id=symbol_value,
                            as_of_at_us=_available_at(cutoff_event),
                            payload=_base_payload(
                                status="incomplete",
                                reason="underlying_reference_quote_unavailable",
                                quiet_episode_id=cast(str, pending["e"]),
                                cutoff_at_us=deadline,
                                cutoff_crossing_event_id=cutoff_event.event_id,
                            ),
                        )
                    )
                    lineages.append(lineage)
                pending = {}

    if initial_active_stage == "entry" and active.get("g") == "entry":
        expected_value = active.get("k")
        expected = (
            tuple(key for key in expected_value if isinstance(key, str))
            if isinstance(expected_value, list | tuple)
            else ()
        )
        terminal = _mapping(active.get("u"))
        captures = _mapping(active.get("z"))
        deadline = _integer(active.get("x"))
        panel_reason: str | None = None
        terminal_event: MarketEvent | None = None
        denied = tuple(
            key for key in expected if _mapping(terminal.get(key)).get("status") == "denied"
        )
        if denied:
            denied_at = max(
                (_integer(_mapping(terminal.get(key)).get("completed")) or 0 for key in denied),
                default=0,
            )
            terminal_event = _first_cutoff_event(batch, denied_at)
            if terminal_event is not None:
                panel_reason = "option_panel_discovery_denied"
        elif len(captures) == len(expected) == 54:
            contracts, panel_reason = _panel_contracts(active)
            if panel_reason is None:
                panel_outputs, panel_lineages, stream_interests, stream_state = _complete_panel(
                    batch=batch,
                    active=active,
                    contracts=contracts,
                )
                if existing_interest_count + len(interests) + len(stream_interests) > 64:
                    panel_reason = "complete_panel_capacity_unavailable"
                else:
                    outputs.extend(panel_outputs)
                    lineages.extend(panel_lineages)
                    interests.extend(stream_interests)
                    active = stream_state if stream_interests else {}
        elif deadline is not None:
            terminal_event = _first_cutoff_event(batch, deadline)
            if terminal_event is not None:
                panel_reason = "option_panel_capture_missing"
        if panel_reason is not None:
            extra = () if terminal_event is None else (terminal_event.event_id,)
            lineage = _panel_lineage(batch, active, extra_event_ids=extra)
            if lineage:
                attribution = _terminal_attribution(terminal, expected)
                payload: dict[str, JsonValue] = {
                    "status": "incomplete",
                    "reason": panel_reason,
                    "quiet_episode_id": cast(str, active["e"]),
                    "interest_statuses": attribution,
                    "resolved_interest_count": sum(
                        1
                        for value in terminal.values()
                        if isinstance(value, Mapping) and value.get("status") == "resolved"
                    ),
                    "capture_count": len(captures),
                }
                if deadline is not None and terminal_event is not None:
                    payload.update(
                        {
                            "cutoff_at_us": deadline,
                            "cutoff_crossing_event_id": terminal_event.event_id,
                        }
                    )
                outputs.append(
                    Observation(
                        subject_instrument_id=cast(str, active["y"]),
                        as_of_at_us=(
                            _available_at(terminal_event)
                            if terminal_event is not None
                            else batch.causal_through_at_us
                        ),
                        payload=_base_payload(**payload),
                    )
                )
                lineages.append(lineage)
            active = {}

    if initial_active_stage == "streams" and active.get("g") == "streams":
        deadline = _integer(active.get("x"))
        cutoff_event = None if deadline is None else _first_cutoff_event(batch, deadline)
        if deadline is not None and cutoff_event is not None:
            expected_value = active.get("k")
            expected = (
                tuple(key for key in expected_value if isinstance(key, str))
                if isinstance(expected_value, list | tuple)
                else ()
            )
            statuses = _mapping(active.get("u"))
            proofs = _mapping(active.get("z"))
            completed_keys: list[str] = []
            for key in expected:
                status = _mapping(statuses.get(key))
                completed_at_us = _integer(status.get("completed"))
                if (
                    status.get("status") == "resolved"
                    and completed_at_us is not None
                    and completed_at_us < deadline
                    and isinstance(_mapping(proofs.get(key)).get("event_id"), str)
                ):
                    completed_keys.append(key)
            complete = len(completed_keys) == len(expected) and bool(expected)
            lineage = _panel_lineage(
                batch,
                active,
                extra_event_ids=(cutoff_event.event_id,),
            )
            if lineage:
                outputs.append(
                    Observation(
                        subject_instrument_id=cast(str, active["y"]),
                        as_of_at_us=_available_at(cutoff_event),
                        payload=_base_payload(
                            status="complete" if complete else "incomplete",
                            reason=None if complete else "selected_leg_stream_window_incomplete",
                            quiet_episode_id=cast(str, active["e"]),
                            expected_stream_count=len(expected),
                            resolved_stream_count=len(completed_keys),
                            stream_interest_statuses=_terminal_attribution(statuses, expected),
                            quote_proof_event_ids={
                                key: cast(str, _mapping(proofs[key])["event_id"])
                                for key in completed_keys
                            },
                            cutoff_at_us=deadline,
                            cutoff_crossing_event_id=cutoff_event.event_id,
                        ),
                    )
                )
                lineages.append(lineage)
            active = {}

    return pending, active, outputs, lineages, interests


class M1CQuietStateOptionsV0:
    @property
    def manifest(self) -> IdeaManifest:
        return MANIFEST

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
        if activation.universe != COHORT:
            raise ValueError("Quiet Options V0 requires the exact frozen stock cohort")
        if dict(activation.parameters) != dict(PARAMETERS):
            raise ValueError("Quiet Options V0 parameters must match the frozen contract")
        prefixes = tuple(
            MarketDataRequirement(
                feed_kind="bars",
                event_kind="bar_5m_session_prefix",
                instrument_id=symbol,
                cadence="5s",
                gaps_block=True,
                staleness_block=True,
            )
            for symbol in COHORT
        )
        baselines = tuple(
            MarketDataRequirement(
                feed_kind="bars",
                event_kind="session_volume_baseline",
                instrument_id=symbol,
                cadence="5s",
                gaps_block=True,
                staleness_block=True,
            )
            for symbol in COHORT
        )
        quotes = tuple(
            MarketDataRequirement(
                feed_kind="quotes",
                event_kind="quote",
                instrument_id=symbol,
                cadence="stream",
                gaps_block=True,
                staleness_block=True,
            )
            for symbol in COHORT
        )
        return (*prefixes, *baselines, *quotes)

    def select_input_prefix(self, batch: IdeaBatch, state: JsonValue) -> int:
        if batch.continuation_request is not None or not batch.events:
            raise ValueError("Quiet input-prefix selection requires an ordinary batch")
        prior = _mapping(state)
        pending = _mapping(prior.get("pending"))
        active = _mapping(prior.get("active"))
        pending_trigger = _integer(pending.get("t"))
        pending_deadline = _integer(pending.get("x"))
        pending_symbol = pending.get("y")
        active_deadline = _integer(active.get("x"))
        active_stage = active.get("g")
        active_statuses = _mapping(active.get("u"))
        active_proofs = _mapping(active.get("z"))
        expected_value = active.get("k")
        expected_keys = (
            {key for key in expected_value if isinstance(key, str)}
            if isinstance(expected_value, list | tuple)
            else set()
        )
        active_episode = active.get("e")
        terminal_receipt_at: int | None = None
        for receipt in batch.discovery_receipts:
            if active_deadline is None or receipt.completed_at_us >= active_deadline:
                continue
            if active_stage == "entry":
                parts = _entry_interest_parts(receipt.interest_key)
                if (
                    parts is None
                    or parts[0] != active_episode
                    or receipt.interest_key not in expected_keys
                ):
                    continue
                if receipt.status == "denied":
                    terminal_receipt_at = (
                        receipt.completed_at_us
                        if terminal_receipt_at is None
                        else min(terminal_receipt_at, receipt.completed_at_us)
                    )
                elif receipt.status == "resolved" and isinstance(receipt.instrument_id, str):
                    active_statuses.setdefault(
                        receipt.interest_key,
                        {
                            "status": "resolved",
                            "instrument_id": receipt.instrument_id,
                            "completed": receipt.completed_at_us,
                        },
                    )
            elif active_stage == "streams":
                parts = _stream_interest_parts(receipt.interest_key)
                expected_instrument = _mapping(active.get("m")).get(receipt.interest_key)
                if (
                    parts is not None
                    and parts[0] == active_episode
                    and receipt.interest_key in expected_keys
                    and receipt.status == "resolved"
                    and receipt.instrument_id == expected_instrument
                ):
                    active_statuses.setdefault(
                        receipt.interest_key,
                        {
                            "status": "resolved",
                            "completed": receipt.completed_at_us,
                        },
                    )
        missing_entry_instruments = {
            str(status["instrument_id"]): cast(int, status["completed"])
            for key, value in active_statuses.items()
            if key not in active_proofs
            and (status := _mapping(value)).get("status") == "resolved"
            and isinstance(status.get("instrument_id"), str)
            and _integer(status.get("completed")) is not None
        }
        missing_stream_instruments = {
            str(instrument_id): cast(int, status["completed"])
            for key, instrument_id in _mapping(active.get("m")).items()
            if key not in active_proofs
            and (status := _mapping(active_statuses.get(key))).get("status") == "resolved"
            and _integer(status.get("completed")) is not None
            and isinstance(instrument_id, str)
        }
        for index, event in enumerate(batch.events):
            if (
                pending_trigger is not None
                and pending_deadline is not None
                and isinstance(pending_symbol, str)
                and (
                    (
                        event.instrument_id == pending_symbol
                        and _valid_quote(
                            event,
                            after_us=pending_trigger,
                            before_us=pending_deadline,
                        )
                        is not None
                    )
                    or _available_at(event) >= pending_deadline
                )
            ):
                return index + 1
            if active_deadline is not None and _available_at(event) >= active_deadline:
                return index + 1
            if terminal_receipt_at is not None and _available_at(event) >= terminal_receipt_at:
                return index + 1
            if (
                active_stage == "entry"
                and event.event_kind == "option_snapshot_capture"
                and event.instrument_id in missing_entry_instruments
                and _available_at(event) >= missing_entry_instruments[event.instrument_id]
            ):
                return index + 1
            if (
                active_stage == "streams"
                and event.event_kind == "quote"
                and event.instrument_id in missing_stream_instruments
                and active_deadline is not None
                and _valid_quote(
                    event,
                    after_us=missing_stream_instruments[event.instrument_id],
                    before_us=active_deadline,
                )
                is not None
            ):
                return index + 1
            if event.event_kind == "bar_5m_session_prefix" and event.instrument_id in COHORT:
                return index + 1
        return len(batch.events)

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        if batch.continuation_request is not None:
            raise ValueError("Quiet Options V0 does not use continuation batches")
        prior = _mapping(state)
        baselines = _table(prior.get("baselines"))
        d1_context = _table(prior.get("d1_context"))
        d1_terminal = _table(prior.get("d1_terminal"))
        episodes = _table(prior.get("episodes"))
        requested = {
            str(key): str(value)
            for key, value in _mapping(prior.get("requested")).items()
            if isinstance(key, str) and isinstance(value, str)
        }
        pending = _mapping(prior.get("pending"))
        active = _mapping(prior.get("active"))
        outputs: list[IdeaOutput] = []
        output_lineages: list[tuple[str, ...]] = []
        interests: list[MarketDataInterest] = []
        candidates: list[dict[str, object]] = []
        latest_baseline_events: dict[str, tuple[str, int, str]] = {}
        for event in batch.events:
            baseline_session = event.payload.get("session")
            if (
                event.event_kind != "session_volume_baseline"
                or event.instrument_id not in COHORT
                or not isinstance(baseline_session, str)
            ):
                continue
            identity = (baseline_session, event.event_at_us, event.event_id)
            if identity > latest_baseline_events.get(event.instrument_id, ("", -1, "")):
                latest_baseline_events[event.instrument_id] = identity

        receipts_by_instrument: dict[str, list[DiscoveryReceipt]] = {}
        for receipt in batch.discovery_receipts:
            parts = _d1_interest_parts(receipt.interest_key)
            if parts is not None:
                session, symbol, right = parts
                d1_cutoff = _integer(_mapping(baselines.get(symbol)).get("x"))
                if d1_cutoff is None or receipt.completed_at_us < d1_cutoff:
                    terminal = d1_terminal.get(symbol)
                    if terminal is None or str(terminal.get("s", "")) < session:
                        terminal = {"s": session}
                    if terminal.get("s") == session and terminal.get(right) not in {
                        "captured",
                        "denied",
                        "late",
                        "window_elapsed",
                    }:
                        terminal[right] = receipt.status
                        terminal[f"{right}_k"] = receipt.interest_key
                        terminal[f"{right}_completed"] = receipt.completed_at_us
                        if receipt.reason_code is not None:
                            terminal[f"{right}_r"] = receipt.reason_code
                        d1_terminal[symbol] = terminal
            entry_parts = _entry_interest_parts(receipt.interest_key)
            if (
                entry_parts is not None
                and active.get("g") == "entry"
                and entry_parts[0] == active.get("e")
                and (entry_deadline := _integer(active.get("x"))) is not None
                and receipt.completed_at_us < entry_deadline
            ):
                expected = active.get("k")
                if isinstance(expected, list | tuple) and receipt.interest_key in expected:
                    terminal = _mapping(active.get("u"))
                    existing = _mapping(terminal.get(receipt.interest_key))
                    if existing.get("status") not in {"resolved", "denied"}:
                        terminal[receipt.interest_key] = {
                            "status": receipt.status,
                            "reason": receipt.reason_code,
                            "instrument_id": receipt.instrument_id,
                            "expiry": receipt.expiry,
                            "strike": receipt.strike,
                            "right": receipt.option_right,
                            "multiplier": receipt.multiplier,
                            "completed": receipt.completed_at_us,
                        }
                        active["u"] = terminal
            stream_parts = _stream_interest_parts(receipt.interest_key)
            if (
                stream_parts is not None
                and active.get("g") == "streams"
                and stream_parts[0] == active.get("e")
                and (stream_deadline := _integer(active.get("x"))) is not None
                and receipt.completed_at_us < stream_deadline
            ):
                expected_streams = active.get("k")
                if (
                    isinstance(expected_streams, list | tuple)
                    and receipt.interest_key in expected_streams
                ):
                    stream_statuses = _mapping(active.get("u"))
                    existing_stream = _mapping(stream_statuses.get(receipt.interest_key))
                    if existing_stream.get("status") not in {"resolved", "denied"}:
                        expected_instrument = _mapping(active.get("m")).get(receipt.interest_key)
                        status = receipt.status
                        reason_code = receipt.reason_code
                        if status == "resolved" and receipt.instrument_id != expected_instrument:
                            status = "denied"
                            reason_code = "instrument_identity_mismatch"
                        stream_statuses[receipt.interest_key] = {
                            "status": status,
                            "reason": reason_code,
                            "completed": receipt.completed_at_us,
                        }
                        active["u"] = stream_statuses
            if receipt.status == "resolved" and receipt.instrument_id is not None:
                receipts_by_instrument.setdefault(receipt.instrument_id, []).append(receipt)

        initial_active_deadline = _integer(active.get("x"))
        active_evidence_event_ids = {
            event.event_id
            for event in (
                batch.events
                if initial_active_deadline is None
                else _events_before_cutoff(batch, initial_active_deadline)
            )
        }
        for event in batch.events:
            if event.event_kind == "session_volume_baseline" and event.instrument_id in COHORT:
                baseline_session_value = event.payload.get("session")
                closes = event.payload.get("session_closes")
                count = event.payload.get("complete_session_count")
                realised = _number(event.payload.get("realised_volatility_20d"))
                if (
                    not isinstance(baseline_session_value, str)
                    or not isinstance(closes, list | tuple)
                    or not closes
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or _number(closes[-1]) is None
                ):
                    continue
                baseline_session = baseline_session_value
                existing_baseline = baselines.get(event.instrument_id)
                if (
                    existing_baseline is not None
                    and isinstance(existing_baseline.get("s"), str)
                    and str(existing_baseline["s"]) > baseline_session
                ):
                    continue
                close = cast(float, _number(closes[-1]))
                baselines[event.instrument_id] = {
                    "i": event.event_id,
                    "s": baseline_session,
                    "c": close,
                    "v": realised,
                    "t": event.event_at_us,
                    "x": event.event_at_us + _D1_WINDOW_US,
                    "k": count,
                }
                request_identity = latest_baseline_events.get(event.instrument_id)
                if (
                    request_identity is not None
                    and request_identity[2] == event.event_id
                    and event.event_at_us + _D1_WINDOW_US > batch.causal_through_at_us
                    and requested.get(event.instrument_id) != baseline_session
                ):
                    for right in _RIGHTS:
                        interests.append(
                            MarketDataInterest(
                                interest_key=(
                                    f"quiet:m1c:d1:{baseline_session}:{event.instrument_id}:{right}"
                                ),
                                underlying_instrument_id=event.instrument_id,
                                minimum_days_to_expiry=7,
                                maximum_days_to_expiry=45,
                                option_right=right,
                                strike_offset=0,
                                reference_price=close,
                                cadence="snapshot",
                                as_of_at_us=event.event_at_us,
                                expires_at_us=event.event_at_us + _D1_WINDOW_US,
                                required=True,
                                priority=100,
                                input_event_id=event.event_id,
                            )
                        )
                    requested[event.instrument_id] = baseline_session
                continue

            if event.event_kind == "quote" and active.get("g") == "streams":
                if event.event_id not in active_evidence_event_ids:
                    continue
                expected_instruments = _mapping(active.get("m"))
                stream_statuses = _mapping(active.get("u"))
                quote_proofs = _mapping(active.get("z"))
                stream_deadline = _integer(active.get("x"))
                for key, instrument_id in expected_instruments.items():
                    stream_status = _mapping(stream_statuses.get(key))
                    completed_at_us = _integer(stream_status.get("completed"))
                    if (
                        instrument_id != event.instrument_id
                        or stream_status.get("status") != "resolved"
                        or completed_at_us is None
                        or stream_deadline is None
                        or key in quote_proofs
                        or _valid_quote(
                            event,
                            after_us=completed_at_us,
                            before_us=stream_deadline,
                        )
                        is None
                    ):
                        continue
                    quote_proofs[key] = {
                        "event_id": event.event_id,
                        "available": _available_at(event),
                    }
                active["z"] = quote_proofs
                continue

            if event.event_kind == "option_snapshot_capture":
                entry_terminal = (
                    _mapping(active.get("u"))
                    if active.get("g") == "entry" and event.event_id in active_evidence_event_ids
                    else {}
                )
                entry_matches = tuple(
                    (key, _mapping(value))
                    for key, value in entry_terminal.items()
                    if isinstance(key, str)
                    and isinstance(value, Mapping)
                    and value.get("status") == "resolved"
                    and value.get("instrument_id") == event.instrument_id
                    and _integer(value.get("completed")) is not None
                    and cast(int, value["completed"])
                    <= max(event.event_at_us, event.received_at_us)
                )
                if entry_matches:
                    captures = _mapping(active.get("z"))
                    for key, receipt_data in entry_matches:
                        entry_parts = _entry_interest_parts(key)
                        if entry_parts is None:
                            continue
                        captures[key] = {
                            "event_id": event.event_id,
                            "available": max(event.event_at_us, event.received_at_us),
                            "instrument_id": event.instrument_id,
                            "expiry": receipt_data.get("expiry"),
                            "strike": receipt_data.get("strike"),
                            "right": receipt_data.get("right"),
                            "multiplier": receipt_data.get("multiplier"),
                            "complete": event.payload.get("source_completeness") == "complete",
                            "bid": event.payload.get("bid"),
                            "ask": event.payload.get("ask"),
                            "delta": event.payload.get("model_delta"),
                        }
                    active["z"] = captures
                    continue
                receipt_candidates = tuple(
                    receipt
                    for receipt in receipts_by_instrument.get(event.instrument_id, ())
                    if receipt.completed_at_us <= max(event.event_at_us, event.received_at_us)
                )
                if not receipt_candidates:
                    continue
                receipt = max(
                    receipt_candidates,
                    key=lambda item: (item.completed_at_us, item.receipt_id),
                )
                parts = _d1_interest_parts(receipt.interest_key)
                if parts is None:
                    continue
                session, symbol, right = parts
                baseline = baselines.get(symbol)
                expiry = (
                    None
                    if baseline is None or baseline.get("s") != session
                    else _integer(baseline.get("x"))
                )
                available = max(event.event_at_us, event.received_at_us)
                if expiry is None:
                    continue
                terminal = d1_terminal.get(symbol)
                if terminal is None or str(terminal.get("s", "")) < session:
                    terminal = {"s": session}
                if terminal.get(right) in {
                    "captured",
                    "denied",
                    "late",
                    "window_elapsed",
                }:
                    continue
                if available >= expiry:
                    terminal[right] = "late"
                    terminal[f"{right}_k"] = receipt.interest_key
                    terminal[f"{right}_a"] = available
                    terminal[f"{right}_i"] = event.event_id
                    d1_terminal[symbol] = terminal
                    continue
                context = d1_context.get(symbol)
                if context is None or str(context.get("s", "")) <= session:
                    if context is None or context.get("s") != session:
                        context = {"s": session}
                    context[right] = {
                        "i": event.event_id,
                        "t": event.event_at_us,
                        "a": available,
                        "source_completeness": event.payload.get("source_completeness"),
                        "bid": event.payload.get("bid"),
                        "ask": event.payload.get("ask"),
                        "model_implied_volatility": event.payload.get("model_implied_volatility"),
                        "option_right": right,
                        "expiry": receipt.expiry,
                        "strike": receipt.strike,
                    }
                    d1_context[symbol] = context
                    terminal[right] = "captured"
                    terminal[f"{right}_k"] = receipt.interest_key
                    terminal[f"{right}_a"] = available
                    d1_terminal[symbol] = terminal
                continue

            if event.event_kind != "bar_5m_session_prefix" or event.instrument_id not in COHORT:
                continue
            prefix_session_value = event.payload.get("session")
            checkpoint_value = event.payload.get("bar_number")
            selected = {event.event_id}
            reason: str | None = None
            probability: float | None = None
            feature_hash: str | None = None
            model_hash: str | None = None
            baseline = baselines.get(event.instrument_id)
            context = d1_context.get(event.instrument_id)
            prefix_terminal: dict[str, object] = {}
            cutoff_at_us: int | None = None
            as_of = _available_at(event)
            if (
                not isinstance(prefix_session_value, str)
                or isinstance(checkpoint_value, bool)
                or not isinstance(checkpoint_value, int)
                or checkpoint_value not in CHECKPOINTS
            ):
                reason = "invalid_session_prefix_identity"
            elif event.payload.get("source_completeness") != "complete":
                reason = "session_prefix_incomplete"
            elif baseline is None or not isinstance(baseline.get("i"), str):
                reason = "prior_session_baseline_missing"
            elif not isinstance(baseline.get("s"), str) or str(baseline["s"]) >= (
                prefix_session_value
            ):
                reason = "prior_session_baseline_not_causal"
            else:
                selected.add(cast(str, baseline["i"]))
                cutoff_at_us = _integer(baseline.get("x"))
                if cutoff_at_us is None:
                    reason = "prior_session_baseline_invalid"
                elif as_of < cutoff_at_us:
                    raise ValueError("next-session Quiet cohort precedes the D-1 cutoff")
                else:
                    prefix_terminal = d1_terminal.get(event.instrument_id, {})
                    if prefix_terminal.get("s") != baseline.get("s"):
                        prefix_terminal = {"s": baseline["s"]}
                    for right in _RIGHTS:
                        if prefix_terminal.get(right) not in {
                            "captured",
                            "denied",
                            "late",
                            "window_elapsed",
                        }:
                            prefix_terminal[right] = "window_elapsed"
                            prefix_terminal[f"{right}_k"] = (
                                f"quiet:m1c:d1:{baseline['s']}:{event.instrument_id}:{right}"
                            )
                    d1_terminal[event.instrument_id] = prefix_terminal
                    selected.update(
                        cast(str, prefix_terminal[f"{right}_i"])
                        for right in _RIGHTS
                        if isinstance(prefix_terminal.get(f"{right}_i"), str)
                    )
                    if context is not None and context.get("s") == baseline.get("s"):
                        selected.update(
                            cast(str, capture["i"])
                            for right in _RIGHTS
                            if (capture := _mapping(context.get(right)))
                            and isinstance(capture.get("i"), str)
                        )
                    if _number(baseline.get("v")) is None:
                        reason = "realised_volatility_20d_not_ready"
                    elif context is None or context.get("s") != baseline.get("s"):
                        reason = "prior_session_option_pair_missing"
                    else:
                        call = _mapping(context.get("call"))
                        put = _mapping(context.get("put"))
                        if not call or not put:
                            reason = "prior_session_option_pair_incomplete"
                        else:
                            try:
                                group_i = build_group_i_for_symbol(
                                    event.payload,
                                    symbol=event.instrument_id,
                                    checkpoint=checkpoint_value,
                                )
                                group_o = build_front_options_context(
                                    call_capture=call,
                                    put_capture=put,
                                    prior_close=cast(float, _number(baseline["c"])),
                                    realised_volatility_20d=cast(float, _number(baseline["v"])),
                                )
                                score = score_m1c(
                                    symbol=event.instrument_id,
                                    checkpoint=checkpoint_value,
                                    group_o=group_o,
                                    group_i=group_i,
                                )
                                probability = cast(float, score["probability"])
                                feature_hash = cast(str, score["feature_hash"])
                                model_hash = cast(str, score["model_hash"])
                            except ValueError:
                                reason = "m1c_inputs_invalid"
            if reason is not None or probability is None:
                lineage = _lineage(batch, selected)
                if lineage:
                    evidence: dict[str, JsonValue] = {}
                    if (
                        baseline is not None
                        and isinstance(baseline.get("i"), str)
                        and isinstance(baseline.get("s"), str)
                        and cutoff_at_us is not None
                    ):
                        evidence = {
                            "baseline_event_id": cast(str, baseline["i"]),
                            "baseline_session": cast(str, baseline["s"]),
                            **_d1_terminal_payload(
                                symbol=event.instrument_id,
                                session=cast(str, baseline["s"]),
                                cutoff_at_us=cutoff_at_us,
                                cohort_available_at_us=as_of,
                                terminal=prefix_terminal,
                            ),
                        }
                    outputs.append(
                        Observation(
                            subject_instrument_id=event.instrument_id,
                            as_of_at_us=as_of,
                            payload=_base_payload(
                                status="unavailable",
                                reason=reason or "m1c_score_unavailable",
                                session=prefix_session_value,
                                checkpoint=checkpoint_value,
                                **evidence,
                            ),
                        )
                    )
                    output_lineages.append(lineage)
                continue

            assert isinstance(prefix_session_value, str)
            assert isinstance(checkpoint_value, int) and not isinstance(checkpoint_value, bool)
            prefix_session = prefix_session_value
            checkpoint = checkpoint_value
            episode = episodes.get(event.instrument_id)
            if episode is None or episode.get("s") != prefix_session:
                episode = {"s": prefix_session, "p": None, "l": None, "c": 0}
            previous = _number(episode.get("p"))
            last = _integer(episode.get("l"))
            elapsed = None if last is None else (as_of - last) / 60_000_000.0
            quiet = classify_quiet_state(
                probability=probability,
                previous_probability=previous,
                minutes_since_previous_episode=elapsed,
            )
            episode["p"] = probability
            count = _integer(episode.get("c")) or 0
            candidate: dict[str, object] | None = None
            if quiet.fresh_episode:
                count += 1
                episode["l"] = as_of
                episode["c"] = count
                candidate = {
                    "i": event.event_id,
                    "s": prefix_session,
                    "n": checkpoint,
                    "y": event.instrument_id,
                    "p": probability,
                    "t": as_of,
                    "x": as_of + _PANEL_WINDOW_US,
                    "c": count,
                    "d": _lineage(batch, selected),
                }
                candidate["e"] = _episode_id(candidate)
                candidates.append(candidate)
            episodes[event.instrument_id] = episode
            lineage = _lineage(batch, selected)
            outputs.append(
                Observation(
                    subject_instrument_id=event.instrument_id,
                    as_of_at_us=as_of,
                    payload=_base_payload(
                        status="complete",
                        session=prefix_session,
                        checkpoint=checkpoint,
                        probability=probability,
                        previous_probability=previous,
                        threshold=BOTTOM_10_THRESHOLD,
                        bottom_10=quiet.bottom_10,
                        bottom_5=quiet.bottom_5,
                        bottom_20=quiet.bottom_20,
                        fresh_episode=quiet.fresh_episode,
                        episode_number=count if quiet.fresh_episode else None,
                        feature_hash=feature_hash,
                        model_hash=model_hash,
                    ),
                )
            )
            output_lineages.append(lineage)

        pending, active, stage_outputs, stage_lineages, stage_interests = _advance_episode(
            batch=batch,
            pending=pending,
            active=active,
            existing_interest_count=len(interests),
        )
        outputs[:0] = stage_outputs
        output_lineages[:0] = stage_lineages
        interests.extend(stage_interests)

        if candidates:
            if pending or active:
                selected_candidate = None
            else:
                selected_candidate = candidates[0]
                pending = dict(selected_candidate)
            for candidate in candidates:
                if selected_candidate is candidate:
                    continue
                proof = candidate.get("d")
                selected_ids = (
                    {value for value in proof if isinstance(value, str)}
                    if isinstance(proof, list | tuple)
                    else {cast(str, candidate["i"])}
                )
                lineage = _lineage(batch, selected_ids)
                if lineage:
                    outputs.append(
                        Observation(
                            subject_instrument_id=cast(str, candidate["y"]),
                            as_of_at_us=cast(int, candidate["t"]),
                            payload=_base_payload(
                                status="incomplete",
                                reason="complete_panel_capacity_unavailable",
                                session=cast(str, candidate["s"]),
                                checkpoint=cast(int, candidate["n"]),
                                probability=cast(float, candidate["p"]),
                                quiet_episode_id=cast(str, candidate["e"]),
                            ),
                        )
                    )
                    output_lineages.append(lineage)

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
                if isinstance(item.get(f"{right}_i"), str):
                    retained_values.add(cast(str, item[f"{right}_i"]))
        for item in (pending, active):
            for name in ("i", "q"):
                if isinstance(item.get(name), str):
                    retained_values.add(cast(str, item[name]))
            proof_ids = item.get("d")
            if isinstance(proof_ids, list | tuple):
                retained_values.update(value for value in proof_ids if isinstance(value, str))
        if active.get("g") in {"entry", "streams"}:
            for raw_capture in _mapping(active.get("z")).values():
                capture = _mapping(raw_capture)
                if isinstance(capture.get("event_id"), str):
                    retained_values.add(cast(str, capture["event_id"]))
        retained = _lineage(batch, retained_values)
        latest_state = cast(
            JsonValue,
            {
                "schema_version": 2,
                "baselines": baselines,
                "d1_context": d1_context,
                "d1_terminal": d1_terminal,
                "episodes": episodes,
                "requested": requested,
                "pending": pending,
                "active": active,
            },
        )
        return IdeaEvaluation(
            state=latest_state,
            outputs=tuple(outputs),
            retained_input_event_ids=retained,
            output_input_event_ids=tuple(output_lineages),
            interests=tuple(interests),
            continuation=None,
        )


def create_plugin() -> M1CQuietStateOptionsV0:
    return M1CQuietStateOptionsV0()


__all__ = [
    "MANIFEST",
    "PARAMETERS",
    "M1CQuietStateOptionsV0",
    "create_plugin",
]
