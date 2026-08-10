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
    MarketEvent,
    Observation,
    OutputKind,
    RuntimeMode,
    Signal,
)
from stocker_runtime.ideas.contract import (
    AncestorPageContinuation,
    DiscoveryReceipt,
    ExactEventsContinuation,
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    IdeaManifest,
    MarketDataInterest,
    MarketDataRequirement,
)

UNIVERSE = (*COHORT, "VTI")
_OPTION_RIGHTS: tuple[Literal["call", "put"], ...] = ("call", "put")
_OPTION_WINDOW_US = 30 * 60 * 1_000_000
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


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _baseline_expiry(baseline: Mapping[str, object]) -> int | None:
    explicit = _integer(baseline.get("x"))
    if explicit is not None:
        return explicit
    event_at_us = _integer(baseline.get("t"))
    return None if event_at_us is None else event_at_us + _OPTION_WINDOW_US


def _root_availability(root: Mapping[str, object]) -> int | None:
    explicit = _integer(root.get("a"))
    return explicit if explicit is not None else _integer(root.get("t"))


def _interest_parts(key: str) -> tuple[str, str, str] | None:
    parts = key.split(":")
    if len(parts) != 5 or parts[:2] != ["m1c", "d1"] or parts[4] not in {"call", "put"}:
        return None
    return parts[2], parts[3], parts[4]


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


def _cohort_key(session: str, checkpoint: int) -> str:
    return f"{session}|{checkpoint:02d}"


def _cohort_identity(key: str) -> tuple[str, int] | None:
    session, separator, checkpoint = key.rpartition("|")
    if not separator or not checkpoint.isdigit():
        return None
    number = int(checkpoint)
    if number not in CHECKPOINTS:
        return None
    return session, number


def _next_complete_cohort(
    prefix_index: Mapping[str, Mapping[str, object]],
    processed: Mapping[str, object],
) -> tuple[str, Mapping[str, object]] | None:
    for key in sorted(prefix_index):
        entries = prefix_index[key]
        if key not in processed and set(entries) == set(UNIVERSE):
            return key, entries
    return None


def _root_ids(prefix_roots: Mapping[str, Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(cast(str, prefix_roots[symbol]["i"]) for symbol in UNIVERSE)


def _update_prefix_root(
    prefix_roots: dict[str, dict[str, object]],
    event: MarketEvent,
) -> bool:
    prefix_session = event.payload.get("session")
    bar_number = event.payload.get("bar_number")
    if (
        not isinstance(prefix_session, str)
        or isinstance(bar_number, bool)
        or not isinstance(bar_number, int)
    ):
        raise ValueError("session-prefix identity is invalid")
    root = {
        "i": event.event_id,
        "s": prefix_session,
        "n": bar_number,
        "t": event.event_at_us,
        "a": max(event.event_at_us, event.received_at_us),
    }
    existing = prefix_roots.get(event.instrument_id)
    current_order = (
        (
            str(existing.get("s", "")),
            cast(int, existing.get("n", 0)),
            cast(int, existing.get("t", 0)),
            str(existing.get("i", "")),
        )
        if existing is not None
        else None
    )
    next_order = (
        prefix_session,
        bar_number,
        event.event_at_us,
        event.event_id,
    )
    if current_order is not None and next_order <= current_order:
        return False
    prefix_roots[event.instrument_id] = root
    return True


def _next_root_checkpoint(
    prefix_roots: Mapping[str, Mapping[str, object]],
    processed: Mapping[str, object],
) -> tuple[str, int] | None:
    if set(prefix_roots) != set(UNIVERSE):
        return None
    sessions = {str(prefix_roots[symbol].get("s", "")) for symbol in UNIVERSE}
    if len(sessions) != 1:
        return None
    session = next(iter(sessions))
    for checkpoint in CHECKPOINTS:
        if _cohort_key(session, checkpoint) in processed:
            continue
        if all(
            isinstance(prefix_roots[symbol].get("n"), int)
            and not isinstance(prefix_roots[symbol].get("n"), bool)
            and cast(int, prefix_roots[symbol]["n"]) >= checkpoint
            for symbol in UNIVERSE
        ):
            return session, checkpoint
    return None


def _ancestor_request(
    prefix_roots: Mapping[str, Mapping[str, object]],
    *,
    cursor: str | None = None,
) -> AncestorPageContinuation:
    return AncestorPageContinuation(
        root_event_ids=_root_ids(prefix_roots),
        event_kind="bar_5m_session_prefix",
        input_roles=("prior_receipt",),
        cursor=cursor,
    )


def _exact_request(
    prefix_roots: Mapping[str, Mapping[str, object]],
    entries: Mapping[str, object],
) -> ExactEventsContinuation:
    return ExactEventsContinuation(
        root_event_ids=_root_ids(prefix_roots),
        event_ids=tuple(cast(str, entries[symbol]) for symbol in sorted(UNIVERSE)),
        event_kind="bar_5m_session_prefix",
        input_roles=("prior_receipt",),
    )


def _status_output(
    *,
    symbol: str,
    session: str,
    checkpoint: int,
    reason: str,
    as_of_at_us: int,
    details: Mapping[str, JsonValue] | None = None,
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
            **({} if details is None else details),
            "research_only": True,
            "execution_enabled": False,
        },
    )


def _prerequisites_are_terminal(
    *,
    current_session: str,
    baselines: Mapping[str, Mapping[str, object]],
    option_context: Mapping[str, Mapping[str, object]],
    option_terminal: dict[str, dict[str, object]],
    prefix_roots: Mapping[str, Mapping[str, object]],
) -> bool:
    for symbol in COHORT:
        baseline = baselines.get(symbol)
        if baseline is None:
            continue
        baseline_session = None if not isinstance(baseline.get("s"), str) else str(baseline["s"])
        if baseline_session is None or baseline_session >= current_session:
            continue
        expiry = _baseline_expiry(baseline)
        stock_root = prefix_roots.get(symbol)
        market_root = prefix_roots.get("VTI")
        stock_available = None if stock_root is None else _root_availability(stock_root)
        market_available = None if market_root is None else _root_availability(market_root)
        if expiry is None or stock_available is None or market_available is None:
            raise ValueError("M1C causal option-window evidence is incomplete")
        cohort_available = max(stock_available, market_available)
        context = option_context.get(symbol)
        terminal = option_terminal.get(symbol)
        if terminal is None or str(terminal.get("s", "")) < baseline_session:
            terminal = {"s": baseline_session}
        for right in _OPTION_RIGHTS:
            capture = (
                {}
                if context is None or context.get("s") != baseline_session
                else _mapping(context.get(right))
            )
            terminal_status = None if terminal.get("s") != baseline_session else terminal.get(right)
            if capture or terminal_status in {"denied", "window_elapsed", "late"}:
                continue
            if expiry > cohort_available:
                raise ValueError("complete M1C cohort precedes its D-1 option cutoff")
            terminal[right] = "window_elapsed"
            terminal[f"{right}_x"] = expiry
            terminal[f"{right}_a"] = cohort_available
        option_terminal[symbol] = terminal
    return True


def _evaluate_exact_cohort(
    *,
    batch: IdeaBatch,
    events_by_symbol: Mapping[str, MarketEvent],
    baselines: Mapping[str, dict[str, object]],
    option_context: Mapping[str, dict[str, object]],
    option_terminal: Mapping[str, dict[str, object]],
    statuses: dict[str, dict[str, object]],
    episodes: dict[str, dict[str, object]],
) -> tuple[list[IdeaOutput], list[tuple[str, ...]], str]:
    market_event = events_by_symbol["VTI"]
    identities = {
        (event.payload.get("session"), event.payload.get("bar_number"))
        for event in events_by_symbol.values()
    }
    if len(identities) != 1:
        raise ValueError("exact continuation crosses M1C checkpoints")
    session_value, checkpoint_value = next(iter(identities))
    if (
        not isinstance(session_value, str)
        or isinstance(checkpoint_value, bool)
        or not isinstance(checkpoint_value, int)
        or checkpoint_value not in CHECKPOINTS
    ):
        raise ValueError("exact continuation checkpoint is invalid")
    current_session = session_value
    checkpoint = checkpoint_value
    outputs: list[IdeaOutput] = []
    output_lineages: list[tuple[str, ...]] = []
    for symbol in sorted(COHORT):
        current_event = events_by_symbol[symbol]
        baseline = baselines.get(symbol)
        context = option_context.get(symbol)
        selected = {current_event.event_id, market_event.event_id}
        selected_times = [current_event.event_at_us, market_event.event_at_us]
        group_i: Mapping[str, object] | None = None
        reason: str | None = None
        status_details: Mapping[str, JsonValue] | None = None
        if current_event.payload.get("source_completeness") != "complete":
            reason = "session_prefix_incomplete"
        else:
            try:
                group_i = build_group_i_for_symbol(
                    current_event.payload,
                    symbol=symbol,
                    checkpoint=checkpoint,
                )
            except ValueError:
                reason = "activity_baseline_not_ready"
        if market_event.payload.get("source_completeness") != "complete":
            reason = reason or "market_prefix_incomplete"
        if baseline is None or not isinstance(baseline.get("i"), str):
            reason = reason or "prior_session_baseline_missing"
        elif not isinstance(baseline.get("s"), str) or str(baseline["s"]) >= current_session:
            reason = reason or "prior_session_baseline_not_causal"
        else:
            selected.add(cast(str, baseline["i"]))
            if isinstance(baseline.get("t"), int):
                selected_times.append(cast(int, baseline["t"]))
            if _number(baseline.get("v")) is None:
                reason = reason or "realised_volatility_20d_not_ready"
        if context is None or context.get("s") != (None if baseline is None else baseline.get("s")):
            reason = reason or "prior_session_option_pair_missing"
        call = None if context is None else _mapping(context.get("call"))
        put = None if context is None else _mapping(context.get("put"))
        if not call or not put:
            reason = reason or "prior_session_option_pair_incomplete"
        else:
            for capture in (call, put):
                if isinstance(capture.get("i"), str):
                    selected.add(cast(str, capture["i"]))
                if isinstance(capture.get("t"), int):
                    selected_times.append(cast(int, capture["t"]))
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
        terminal = option_terminal.get(symbol)
        baseline_session = None if baseline is None else baseline.get("s")
        if terminal is not None and terminal.get("s") == baseline_session:
            elapsed_rights = tuple(
                right
                for right in _OPTION_RIGHTS
                if terminal.get(right) in {"window_elapsed", "late"}
            )
            if elapsed_rights:
                expiry = _baseline_expiry({} if baseline is None else baseline)
                cohort_available = max(
                    max(current_event.event_at_us, current_event.received_at_us),
                    max(market_event.event_at_us, market_event.received_at_us),
                )
                if expiry is not None:
                    status_details = {
                        "terminal_basis": "causal_window_elapsed_without_capture",
                        "interest_keys": tuple(
                            f"m1c:d1:{baseline_session}:{symbol}:{right}"
                            for right in elapsed_rights
                        ),
                        "interest_expires_at_us": expiry,
                        "cohort_available_at_us": cohort_available,
                    }
        as_of_at_us = max(selected_times)
        status = statuses.get(symbol)
        if reason is not None:
            if status is None or status.get("s") != current_session or status.get("r") != reason:
                lineage = _lineage(batch, selected)
                if lineage:
                    outputs.append(
                        _status_output(
                            symbol=symbol,
                            session=current_session,
                            checkpoint=checkpoint,
                            reason=reason,
                            as_of_at_us=as_of_at_us,
                            details=status_details,
                        )
                    )
                    output_lineages.append(lineage)
                statuses[symbol] = {"s": current_session, "r": reason}
            continue
        assert baseline is not None and call and put and group_i is not None
        group_o = build_front_options_context(
            call_capture=call,
            put_capture=put,
            prior_close=cast(float, _number(baseline["c"])),
            realised_volatility_20d=cast(float, _number(baseline["v"])),
        )
        score = score_m1c(
            symbol=symbol,
            checkpoint=checkpoint,
            group_o=group_o,
            group_i=group_i,
        )
        raw_direction = build_direction_features(
            symbol=symbol,
            checkpoint=checkpoint,
            stock_prefix=current_event.payload,
            market_prefix=market_event.payload,
        )
        lineage = _lineage(batch, selected)
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
        crossing = probability >= M1C_THRESHOLD and (previous is None or previous < M1C_THRESHOLD)
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
    return outputs, output_lineages, _cohort_key(current_session, checkpoint)


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

    def select_input_prefix(self, batch: IdeaBatch, state: JsonValue) -> int:
        if batch.continuation_request is not None or not batch.events:
            raise ValueError("M1C input-prefix selection requires an ordinary batch")
        prior = _mapping(state)
        prefix_roots = _table(prior.get("prefix_roots"))
        processed = _mapping(prior.get("processed"))
        for index, event in enumerate(batch.events):
            if event.event_kind != "bar_5m_session_prefix" or event.instrument_id not in UNIVERSE:
                continue
            _update_prefix_root(prefix_roots, event)
            if _next_root_checkpoint(prefix_roots, processed) is not None:
                return index + 1
        return len(batch.events)

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        prior = _mapping(state)
        baselines = _table(prior.get("baselines"))
        option_context = _table(prior.get("option_context"))
        option_terminal = _table(prior.get("option_terminal"))
        prefix_roots = _table(prior.get("prefix_roots"))
        prefix_index = _table(prior.get("prefix_index"))
        processed = _mapping(prior.get("processed"))
        statuses = _table(prior.get("statuses"))
        episodes = _table(prior.get("episodes"))
        requested = {
            str(key): str(value)
            for key, value in _mapping(prior.get("requested")).items()
            if isinstance(key, str) and isinstance(value, str)
        }
        outputs: list[IdeaOutput] = []
        output_lineages: list[tuple[str, ...]] = []
        interests: list[MarketDataInterest] = []
        continuation: AncestorPageContinuation | ExactEventsContinuation | None = None

        receipts_by_instrument: dict[str, list[DiscoveryReceipt]] = {}
        for receipt in batch.discovery_receipts:
            parts = _interest_parts(receipt.interest_key)
            if parts is not None:
                receipt_session, symbol, right = parts
                if symbol in COHORT:
                    terminal = option_terminal.get(symbol)
                    if terminal is None or str(terminal.get("s", "")) < receipt_session:
                        terminal = {"s": receipt_session}
                    if terminal.get("s") == receipt_session:
                        terminal[right] = receipt.status
                        option_terminal[symbol] = terminal
            if receipt.status == "resolved" and receipt.instrument_id is not None:
                receipts_by_instrument.setdefault(receipt.instrument_id, []).append(receipt)

        if batch.continuation_request is None:
            baseline_requests: dict[str, tuple[str, float, int, int, str]] = {}
            roots_changed = False
            for event in batch.events:
                if event.event_kind == "session_volume_baseline" and event.instrument_id in COHORT:
                    baseline_session = event.payload.get("session")
                    closes = event.payload.get("session_closes")
                    count = event.payload.get("complete_session_count")
                    if (
                        isinstance(baseline_session, str)
                        and isinstance(closes, list | tuple)
                        and closes
                        and isinstance(count, int)
                        and not isinstance(count, bool)
                        and _number(closes[-1]) is not None
                    ):
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
                            "v": _number(event.payload.get("realised_volatility_20d")),
                            "t": event.event_at_us,
                            "a": max(event.event_at_us, event.received_at_us),
                            "x": event.event_at_us + _OPTION_WINDOW_US,
                            "k": count,
                        }
                        baseline_requests[event.instrument_id] = (
                            baseline_session,
                            close,
                            event.event_at_us,
                            event.event_at_us + _OPTION_WINDOW_US,
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
                            candidates,
                            key=lambda item: (item.completed_at_us, item.receipt_id),
                        )
                        parts = _interest_parts(receipt.interest_key)
                        if parts is not None:
                            receipt_session, symbol, right = parts
                            baseline = baselines.get(symbol)
                            capture_available = max(event.event_at_us, event.received_at_us)
                            capture_expiry = (
                                None
                                if baseline is None or baseline.get("s") != receipt_session
                                else _baseline_expiry(baseline)
                            )
                            if capture_expiry is None:
                                continue
                            if capture_available > capture_expiry:
                                terminal = option_terminal.get(symbol)
                                if terminal is None or str(terminal.get("s", "")) < receipt_session:
                                    terminal = {"s": receipt_session}
                                if terminal.get("s") == receipt_session:
                                    terminal[right] = "late"
                                    terminal[f"{right}_x"] = capture_expiry
                                    terminal[f"{right}_a"] = capture_available
                                    option_terminal[symbol] = terminal
                                continue
                            existing = option_context.get(symbol)
                            if existing is None or str(existing.get("s", "")) <= receipt_session:
                                if existing is None or existing.get("s") != receipt_session:
                                    existing = {"s": receipt_session}
                                existing[right] = {
                                    "i": event.event_id,
                                    "t": event.event_at_us,
                                    "a": capture_available,
                                    "x": capture_expiry,
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
                                terminal = option_terminal.get(symbol)
                                if terminal is None or str(terminal.get("s", "")) < receipt_session:
                                    terminal = {"s": receipt_session}
                                if terminal.get("s") == receipt_session:
                                    terminal[right] = "captured"
                                    option_terminal[symbol] = terminal
                    continue
                if (
                    event.event_kind == "bar_5m_session_prefix"
                    and event.instrument_id in UNIVERSE
                    and _update_prefix_root(prefix_roots, event)
                ):
                    roots_changed = True
            for symbol, (session, close, as_of_at_us, expires_at_us, input_event_id) in sorted(
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
                            expires_at_us=expires_at_us,
                            required=True,
                            priority=100,
                            input_event_id=input_event_id,
                        )
                    )
                requested[symbol] = session
            if roots_changed:
                prefix_index = {}
                active_sessions = {
                    str(item["s"])
                    for item in prefix_roots.values()
                    if isinstance(item.get("s"), str)
                }
                processed = {
                    key: value
                    for key, value in processed.items()
                    if (_cohort_identity(str(key)) or ("", 0))[0] in active_sessions
                }
            next_root_checkpoint = _next_root_checkpoint(prefix_roots, processed)
            if next_root_checkpoint is not None and _prerequisites_are_terminal(
                current_session=next_root_checkpoint[0],
                baselines=baselines,
                option_context=option_context,
                option_terminal=option_terminal,
                prefix_roots=prefix_roots,
            ):
                continuation = _ancestor_request(prefix_roots)

        elif isinstance(batch.continuation_request, AncestorPageContinuation):
            if batch.continuation_request.root_event_ids != _root_ids(prefix_roots):
                raise ValueError("ancestor continuation roots do not match plugin state")
            for event in batch.rehydrated_events:
                prefix_session = event.payload.get("session")
                checkpoint = event.payload.get("bar_number")
                if (
                    event.event_kind != "bar_5m_session_prefix"
                    or event.instrument_id not in UNIVERSE
                    or not isinstance(prefix_session, str)
                    or isinstance(checkpoint, bool)
                    or not isinstance(checkpoint, int)
                ):
                    raise ValueError("ancestor continuation event is invalid")
                if checkpoint not in CHECKPOINTS:
                    continue
                key = _cohort_key(prefix_session, checkpoint)
                entries = prefix_index.setdefault(key, {})
                existing_id = entries.get(event.instrument_id)
                if existing_id is not None and existing_id != event.event_id:
                    raise ValueError("duplicate continuation checkpoint identity")
                entries[event.instrument_id] = event.event_id
            if batch.continuation_token is not None:
                continuation = _ancestor_request(
                    prefix_roots,
                    cursor=batch.continuation_token,
                )
            else:
                next_cohort = _next_complete_cohort(prefix_index, processed)
                if next_cohort is not None:
                    continuation = _exact_request(prefix_roots, next_cohort[1])

        else:
            request = batch.continuation_request
            if request.root_event_ids != _root_ids(prefix_roots):
                raise ValueError("exact continuation roots do not match plugin state")
            if tuple(event.event_id for event in batch.rehydrated_events) != request.event_ids:
                raise ValueError("exact continuation events do not match the request")
            events_by_symbol = {event.instrument_id: event for event in batch.rehydrated_events}
            if set(events_by_symbol) != set(UNIVERSE) or len(events_by_symbol) != len(UNIVERSE):
                raise ValueError("exact continuation requires the complete M1C cohort")
            exact_outputs, exact_lineages, key = _evaluate_exact_cohort(
                batch=batch,
                events_by_symbol=events_by_symbol,
                baselines=baselines,
                option_context=option_context,
                option_terminal=option_terminal,
                statuses=statuses,
                episodes=episodes,
            )
            indexed = prefix_index.get(key)
            if indexed is None or any(
                indexed.get(symbol) != events_by_symbol[symbol].event_id for symbol in UNIVERSE
            ):
                raise ValueError("exact continuation is absent from the scanned index")
            outputs.extend(exact_outputs)
            output_lineages.extend(exact_lineages)
            processed[key] = 1
            next_cohort = _next_complete_cohort(prefix_index, processed)
            if next_cohort is not None:
                continuation = _exact_request(prefix_roots, next_cohort[1])

        retained_values: set[str] = set()
        for item in baselines.values():
            if isinstance(item.get("i"), str):
                retained_values.add(cast(str, item["i"]))
        for item in option_context.values():
            for right in _OPTION_RIGHTS:
                capture = _mapping(item.get(right))
                if isinstance(capture.get("i"), str):
                    retained_values.add(cast(str, capture["i"]))
        for item in prefix_roots.values():
            if isinstance(item.get("i"), str):
                retained_values.add(cast(str, item["i"]))
        retained = _lineage(batch, retained_values)
        latest_state = cast(
            JsonValue,
            {
                "schema_version": 2,
                "baselines": baselines,
                "option_context": option_context,
                "option_terminal": option_terminal,
                "prefix_roots": prefix_roots,
                "prefix_index": prefix_index,
                "processed": processed,
                "statuses": statuses,
                "episodes": episodes,
                "requested": requested,
            },
        )
        return IdeaEvaluation(
            state=latest_state,
            outputs=tuple(outputs),
            retained_input_event_ids=retained,
            output_input_event_ids=tuple(output_lineages),
            interests=tuple(interests),
            continuation=continuation,
        )


def create_plugin() -> FrozenM1CSignalV0:
    return FrozenM1CSignalV0()


__all__ = ["COHORT", "MANIFEST", "UNIVERSE", "FrozenM1CSignalV0", "create_plugin"]
