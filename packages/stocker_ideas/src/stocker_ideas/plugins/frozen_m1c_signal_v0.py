"""Frozen M1C Signal V0 over generic causal market-data receipts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from stocker_ideas.plugins.frozen_m1c_v0 import (
    CHECKPOINTS,
    COHORT,
    M1C_THRESHOLD,
    build_direction_features,
    build_front_options_context,
    build_group_i_for_symbol,
    classify_directions,
    score_m1c,
)
from stocker_runtime.domain import (
    IdeaOutput,
    JsonValue,
    Observation,
    OutputKind,
    RuntimeMode,
    Signal,
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
_OPTION_RIGHTS: tuple[Literal["call", "put"], ...] = ("call", "put")
_PARAMETERS = {
    "checkpoints": CHECKPOINTS,
    "minimum_activity_sessions": 10,
    "minimum_episode_spacing_minutes": 30,
    "option_minimum_days_to_expiry": 7,
    "option_maximum_days_to_expiry": 45,
    "option_snapshot_lifetime_minutes": 30,
    "threshold": M1C_THRESHOLD,
}
MANIFEST = IdeaManifest(
    api_version=1,
    idea_id="frozen_m1c_signal",
    idea_version="v0",
    display_name="Frozen M1C Signal V0",
    description="Frozen causal M1C movement evidence with labelled A1/C1/R1 controls.",
    modes=(RuntimeMode.PROSPECTIVE_RECORD, RuntimeMode.SHADOW),
    output_kinds=(OutputKind.OBSERVATION, OutputKind.SIGNAL),
    parameter_schema_version="frozen-m1c-signal-v0",
    parameter_schema={
        "type": "object",
        "additionalProperties": False,
        "required": tuple(_PARAMETERS),
        "properties": {
            "checkpoints": {
                "type": "array",
                "minItems": 15,
                "maxItems": 15,
                "enum": (CHECKPOINTS,),
                "items": {"type": "integer", "enum": CHECKPOINTS},
            },
            "minimum_activity_sessions": {"type": "integer", "enum": (10,)},
            "minimum_episode_spacing_minutes": {"type": "integer", "enum": (30,)},
            "option_minimum_days_to_expiry": {"type": "integer", "enum": (7,)},
            "option_maximum_days_to_expiry": {"type": "integer", "enum": (45,)},
            "option_snapshot_lifetime_minutes": {"type": "integer", "enum": (30,)},
            "threshold": {"type": "number", "enum": (M1C_THRESHOLD,)},
        },
    },
    maximum_state_bytes=65_536,
    maximum_outputs_per_batch=128,
    maximum_interests_per_batch=40,
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
    return result if result == result and abs(result) != float("inf") else None


def _pack_prefix(
    event_payload: Mapping[str, object], event_id: str, event_at_us: int
) -> dict[str, object]:
    session = event_payload.get("session")
    bar_number = event_payload.get("bar_number")
    trailing = event_payload.get("trailing_bars")
    if (
        not isinstance(session, str)
        or isinstance(bar_number, bool)
        or not isinstance(bar_number, int)
        or not isinstance(trailing, list | tuple)
    ):
        raise ValueError("session-prefix identity is invalid")
    rows: list[list[object]] = []
    for value in trailing:
        if not isinstance(value, Mapping):
            raise ValueError("session-prefix trailing bar is invalid")
        rows.append(
            [
                value.get("bar_number"),
                value.get("return_bps"),
                value.get("open"),
                value.get("high"),
                value.get("low"),
                value.get("close"),
                value.get("historical_relative_activity"),
                value.get("session_vwap"),
            ]
        )
    accumulator_value = event_payload.get("accumulator")
    accumulator = (
        dict(cast(Mapping[str, object], accumulator_value))
        if isinstance(accumulator_value, Mapping)
        else {}
    )
    return {
        "i": event_id,
        "s": session,
        "n": bar_number,
        "t": event_at_us,
        "c": event_payload.get("source_completeness"),
        "r": rows,
        "a": accumulator,
    }


def _inflate_prefix(packed: Mapping[str, object]) -> dict[str, object]:
    rows = packed.get("r")
    trailing: list[dict[str, object]] = []
    if isinstance(rows, list | tuple):
        for row in rows:
            if isinstance(row, list | tuple) and len(row) == 8:
                trailing.append(
                    {
                        "bar_number": row[0],
                        "return_bps": row[1],
                        "open": row[2],
                        "high": row[3],
                        "low": row[4],
                        "close": row[5],
                        "historical_relative_activity": row[6],
                        "session_vwap": row[7],
                    }
                )
    return {
        "session": packed.get("s"),
        "bar_number": packed.get("n"),
        "source_completeness": packed.get("c"),
        "trailing_bars": trailing,
        "accumulator": packed.get("a", {}),
    }


def _interest_parts(key: str) -> tuple[str, str, str] | None:
    parts = key.split(":")
    if len(parts) != 5 or parts[:2] != ["m1c", "d1"] or parts[4] not in {"call", "put"}:
        return None
    return parts[2], parts[3], parts[4]


def _lineage(batch: IdeaBatch, selected: set[str]) -> tuple[str, ...]:
    available = (
        *batch.prior_state_input_event_ids,
        *(event.event_id for event in batch.events),
    )
    return tuple(event_id for event_id in available if event_id in selected)


def _status_output(
    *,
    symbol: str,
    session: str,
    checkpoint: int,
    reason: str,
    as_of_at_us: int,
) -> Observation:
    return Observation(
        subject_instrument_id=symbol,
        as_of_at_us=as_of_at_us,
        payload={
            "method_id": "frozen_m1c_signal_v0",
            "session": session,
            "checkpoint": checkpoint,
            "status": "unavailable",
            "reason": reason,
            "research_only": True,
            "execution_enabled": False,
        },
    )


class FrozenM1CSignalV0:
    @property
    def manifest(self) -> IdeaManifest:
        return MANIFEST

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
        if activation.universe != UNIVERSE:
            raise ValueError("Frozen M1C Signal V0 requires its exact stock and VTI cohort")
        if dict(activation.parameters) != _PARAMETERS:
            raise ValueError("Frozen M1C Signal V0 parameters must match the frozen contract")
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
        return (*prefixes, *baselines)

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        prior = _mapping(state)
        baselines = _table(prior.get("baselines"))
        option_context = _table(prior.get("option_context"))
        pending = _table(prior.get("pending"))
        statuses = _table(prior.get("statuses"))
        episodes = _table(prior.get("episodes"))
        requested = {
            str(key): str(value)
            for key, value in _mapping(prior.get("requested")).items()
            if isinstance(key, str) and isinstance(value, str)
        }
        market = _mapping(prior.get("market"))
        outputs: list[IdeaOutput] = []
        output_lineages: list[tuple[str, ...]] = []
        interests: list[MarketDataInterest] = []
        baseline_requests: dict[str, tuple[str, float, int, str]] = {}

        receipts_by_instrument: dict[str, list[DiscoveryReceipt]] = {}
        for receipt in batch.discovery_receipts:
            if receipt.status == "resolved" and receipt.instrument_id is not None:
                receipts_by_instrument.setdefault(receipt.instrument_id, []).append(receipt)

        for event in batch.events:
            if event.event_kind == "session_volume_baseline" and event.instrument_id in COHORT:
                session = event.payload.get("session")
                closes = event.payload.get("session_closes")
                count = event.payload.get("complete_session_count")
                if (
                    isinstance(session, str)
                    and isinstance(closes, list | tuple)
                    and closes
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and (_number(closes[-1]) is not None)
                ):
                    existing_baseline = baselines.get(event.instrument_id)
                    if (
                        existing_baseline is not None
                        and isinstance(existing_baseline.get("s"), str)
                        and str(existing_baseline["s"]) > session
                    ):
                        continue
                    close = float(cast(float, _number(closes[-1])))
                    realised = _number(event.payload.get("realised_volatility_20d"))
                    baselines[event.instrument_id] = {
                        "i": event.event_id,
                        "s": session,
                        "c": close,
                        "v": realised,
                        "t": event.event_at_us,
                        "k": count,
                    }
                    baseline_requests[event.instrument_id] = (
                        session,
                        close,
                        event.event_at_us,
                        event.event_id,
                    )
                continue
            if event.event_kind == "option_snapshot_capture":
                candidates = [
                    receipt
                    for receipt in receipts_by_instrument.get(event.instrument_id, ())
                    if receipt.completed_at_us <= max(event.event_at_us, event.received_at_us)
                ]
                if candidates:
                    receipt = max(
                        candidates, key=lambda item: (item.completed_at_us, item.receipt_id)
                    )
                    parts = _interest_parts(receipt.interest_key)
                    if parts is not None:
                        session, symbol, right = parts
                        existing = option_context.get(symbol)
                        if existing is None or str(existing.get("s", "")) <= session:
                            if existing is None or existing.get("s") != session:
                                existing = {"s": session}
                            existing[right] = {
                                "i": event.event_id,
                                "t": event.event_at_us,
                                "source_completeness": event.payload.get("source_completeness"),
                                "bid": event.payload.get("bid"),
                                "ask": event.payload.get("ask"),
                                "model_implied_volatility": event.payload.get(
                                    "model_implied_volatility"
                                ),
                                "open_interest": event.payload.get("open_interest"),
                                "option_right": right,
                                "expiry": receipt.expiry,
                                "strike": receipt.strike,
                            }
                            option_context[symbol] = existing
                continue
            if (
                event.event_kind == "bar_5m_session_prefix"
                and event.instrument_id in UNIVERSE
                and event.payload.get("bar_number") in CHECKPOINTS
            ):
                packed = _pack_prefix(event.payload, event.event_id, event.event_at_us)
                if event.instrument_id == "VTI":
                    market = packed
                else:
                    try:
                        packed["g"] = build_group_i_for_symbol(
                            event.payload,
                            symbol=event.instrument_id,
                            checkpoint=cast(int, event.payload["bar_number"]),
                        )
                    except ValueError as error:
                        packed["e"] = str(error)
                    pending[event.instrument_id] = packed

        for symbol, (session, close, as_of_at_us, input_event_id) in sorted(
            baseline_requests.items()
        ):
            if requested.get(symbol) == session:
                continue
            for right in _OPTION_RIGHTS:
                interests.append(
                    MarketDataInterest(
                        interest_key=f"m1c:d1:{session}:{symbol}:{right}",
                        underlying_instrument_id=symbol,
                        minimum_days_to_expiry=7,
                        maximum_days_to_expiry=45,
                        option_right=right,
                        strike_offset=0,
                        reference_price=close,
                        cadence="snapshot",
                        as_of_at_us=as_of_at_us,
                        expires_at_us=as_of_at_us + 30 * 60 * 1_000_000,
                        required=True,
                        priority=100,
                        input_event_id=input_event_id,
                    )
                )
            requested[symbol] = session

        for symbol in COHORT:
            current = pending.get(symbol)
            if current is None:
                continue
            session_value = current.get("s")
            checkpoint_value = current.get("n")
            event_id = current.get("i")
            current_at = current.get("t")
            if (
                not isinstance(session_value, str)
                or isinstance(checkpoint_value, bool)
                or not isinstance(checkpoint_value, int)
                or not isinstance(event_id, str)
                or isinstance(current_at, bool)
                or not isinstance(current_at, int)
            ):
                continue
            current_session = session_value
            checkpoint = checkpoint_value
            reason: str | None = None
            selected = {event_id}
            if current.get("c") != "complete":
                reason = "session_prefix_incomplete"
            elif "g" not in current:
                reason = "activity_baseline_not_ready"
            if market.get("s") != current_session or market.get("n") != checkpoint:
                reason = reason or "market_prefix_not_ready"
            market_id = market.get("i")
            if isinstance(market_id, str) and market.get("s") == current_session:
                selected.add(market_id)
            baseline = baselines.get(symbol)
            context = option_context.get(symbol)
            if baseline is None or not isinstance(baseline.get("i"), str):
                reason = reason or "prior_session_baseline_missing"
            elif not isinstance(baseline.get("s"), str) or str(baseline["s"]) >= current_session:
                reason = reason or "prior_session_baseline_not_causal"
            else:
                selected.add(cast(str, baseline["i"]))
                if _number(baseline.get("v")) is None:
                    reason = reason or "realised_volatility_20d_not_ready"
            if context is None or context.get("s") != (
                None if baseline is None else baseline.get("s")
            ):
                reason = reason or "prior_session_option_pair_missing"
            call = None if context is None else _mapping(context.get("call"))
            put = None if context is None else _mapping(context.get("put"))
            if not call or not put:
                reason = reason or "prior_session_option_pair_incomplete"
            else:
                if isinstance(call.get("i"), str):
                    selected.add(cast(str, call["i"]))
                if isinstance(put.get("i"), str):
                    selected.add(cast(str, put["i"]))
                if (
                    call.get("source_completeness") != "complete"
                    or put.get("source_completeness") != "complete"
                ):
                    reason = reason or "prior_session_option_pair_incomplete"
                call_strike = _number(call.get("strike"))
                put_strike = _number(put.get("strike"))
                if (
                    call.get("option_right") != "call"
                    or put.get("option_right") != "put"
                    or not isinstance(call.get("expiry"), str)
                    or call.get("expiry") != put.get("expiry")
                    or call_strike is None
                    or call_strike <= 0.0
                    or put_strike is None
                    or call_strike != put_strike
                ):
                    reason = reason or "prior_session_option_pair_mismatch"
            status = statuses.get(symbol)
            if reason is not None:
                if (
                    status is None
                    or status.get("s") != current_session
                    or status.get("r") != reason
                ):
                    lineage = _lineage(batch, selected)
                    if lineage:
                        outputs.append(
                            _status_output(
                                symbol=symbol,
                                session=current_session,
                                checkpoint=checkpoint,
                                reason=reason,
                                as_of_at_us=current_at,
                            )
                        )
                        output_lineages.append(lineage)
                    statuses[symbol] = {"s": current_session, "r": reason}
                continue
            assert baseline is not None and call and put and isinstance(market_id, str)
            group_o = build_front_options_context(
                call_capture=call,
                put_capture=put,
                prior_close=cast(float, _number(baseline["c"])),
                realised_volatility_20d=cast(float, _number(baseline["v"])),
            )
            group_i = cast(Mapping[str, object], current["g"])
            score = score_m1c(
                symbol=symbol,
                checkpoint=checkpoint,
                group_o=group_o,
                group_i=group_i,
            )
            raw_direction = build_direction_features(
                symbol=symbol,
                checkpoint=checkpoint,
                stock_prefix=_inflate_prefix(current),
                market_prefix=_inflate_prefix(market),
            )
            lineage = _lineage(batch, selected)
            as_of_at_us = max(
                current_at,
                cast(int, market["t"]),
                cast(int, baseline["t"]),
                cast(int, call["t"]),
                cast(int, put["t"]),
            )
            payload = cast(
                Mapping[str, JsonValue],
                {
                    "method_id": "frozen_m1c_signal_v0",
                    "session": current_session,
                    "checkpoint": checkpoint,
                    "status": "complete",
                    "eligible": True,
                    "probability": score["probability"],
                    "threshold": M1C_THRESHOLD,
                    "threshold_passed": score["threshold_passed"],
                    "missing_feature_count": score["missing_feature_count"],
                    "feature_hash": score["feature_hash"],
                    "model_hash": score["model_hash"],
                    "research_only": True,
                    "execution_enabled": False,
                },
            )
            outputs.append(
                Observation(
                    subject_instrument_id=symbol,
                    as_of_at_us=as_of_at_us,
                    payload=payload,
                )
            )
            output_lineages.append(lineage)
            episode = episodes.get(symbol)
            if episode is None or episode.get("s") != current_session:
                episode = {"s": current_session, "p": None, "l": None, "c": 0}
            probability = cast(float, score["probability"])
            previous = _number(episode.get("p"))
            last = episode.get("l")
            crossing = probability >= M1C_THRESHOLD and (
                previous is None or previous < M1C_THRESHOLD
            )
            spaced = not isinstance(last, int) or as_of_at_us - last >= 30 * 60 * 1_000_000
            fresh = crossing and spaced
            episode["p"] = probability
            if fresh:
                episode["l"] = as_of_at_us
                count_value = episode.get("c", 0)
                episode["c"] = (
                    count_value + 1
                    if isinstance(count_value, int) and not isinstance(count_value, bool)
                    else 1
                )
                signal_payload = cast(
                    Mapping[str, JsonValue],
                    {
                        **payload,
                        "fresh_episode": True,
                        "episode_number": episode["c"],
                        "previous_probability": previous,
                        "prospective_entry_timestamp_us": as_of_at_us,
                    },
                )
                outputs.append(
                    Signal(
                        subject_instrument_id=symbol,
                        as_of_at_us=as_of_at_us,
                        payload=signal_payload,
                    )
                )
                output_lineages.append(lineage)
                for classification in classify_directions(
                    symbol=symbol,
                    checkpoint=checkpoint,
                    session=current_session,
                    raw_features=raw_direction,
                ):
                    outputs.append(
                        Observation(
                            subject_instrument_id=symbol,
                            as_of_at_us=as_of_at_us,
                            payload=cast(
                                Mapping[str, JsonValue],
                                {
                                    "method_id": "frozen_m1c_signal_v0",
                                    "control": "direction_classification",
                                    "session": current_session,
                                    "checkpoint": checkpoint,
                                    **classification,
                                    "research_only": True,
                                    "execution_enabled": False,
                                },
                            ),
                        )
                    )
                    output_lineages.append(lineage)
            episodes[symbol] = episode
            statuses[symbol] = {"s": current_session, "r": "ready"}
            pending.pop(symbol, None)

        retained_values: list[str] = []
        for item in baselines.values():
            if isinstance(item.get("i"), str):
                retained_values.append(cast(str, item["i"]))
        for item in option_context.values():
            for right in ("call", "put"):
                capture = _mapping(item.get(right))
                if isinstance(capture.get("i"), str):
                    retained_values.append(cast(str, capture["i"]))
        for item in pending.values():
            if isinstance(item.get("i"), str):
                retained_values.append(cast(str, item["i"]))
        if isinstance(market.get("i"), str):
            retained_values.append(cast(str, market["i"]))
        retained = _lineage(batch, set(retained_values))
        latest_state = cast(
            JsonValue,
            {
                "schema_version": 1,
                "baselines": baselines,
                "option_context": option_context,
                "pending": pending,
                "market": market,
                "statuses": statuses,
                "episodes": episodes,
                "requested": requested,
                "input_watermark": batch.input_watermark,
            },
        )
        return IdeaEvaluation(
            state=latest_state,
            outputs=tuple(outputs),
            retained_input_event_ids=retained,
            output_input_event_ids=tuple(output_lineages),
            interests=tuple(interests),
        )


def create_plugin() -> FrozenM1CSignalV0:
    return FrozenM1CSignalV0()


__all__ = ["COHORT", "MANIFEST", "UNIVERSE", "FrozenM1CSignalV0", "create_plugin"]
