"""Frozen causal Opening Leader Continuation V0 reference plugin.

The plugin consumes generic, receipt-backed five-minute bars.  Completeness is owned by
the core projector: a bar is usable only when all sixty IBKR five-second constituents
exist and no gap (including a later-resolved gap) overlaps its interval.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from stocker_runtime.domain import (
    IdeaOutput,
    JsonValue,
    Observation,
    OutputKind,
    ProposedTrade,
    ProposedTradeLeg,
    RuntimeMode,
    Signal,
)
from stocker_runtime.ideas.contract import (
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    IdeaManifest,
    MarketDataRequirement,
)

COHORT = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)

MANIFEST = IdeaManifest(
    api_version=1,
    idea_id="opening_leader_continuation",
    idea_version="v0",
    display_name="Opening Leader Continuation V0",
    description="Frozen causal opening-slate rank-one continuation proposal.",
    modes=(RuntimeMode.PROSPECTIVE_RECORD, RuntimeMode.SHADOW),
    output_kinds=(OutputKind.OBSERVATION, OutputKind.SIGNAL, OutputKind.PROPOSED_TRADE),
    parameter_schema_version="v0-frozen",
    parameter_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ("checkpoints", "minimum_complete_slate"),
        "properties": {
            "checkpoints": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "enum": ((6, 12),),
                "items": {"type": "integer", "enum": (6, 12)},
            },
            "minimum_complete_slate": {"type": "integer", "enum": (15,)},
        },
    },
    maximum_state_bytes=65_536,
    maximum_outputs_per_batch=54,
    maximum_interests_per_batch=0,
)


class OpeningLeaderContinuationV0:
    @property
    def manifest(self) -> IdeaManifest:
        return MANIFEST

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
        if activation.universe != COHORT:
            raise ValueError("Opening Leader V0 requires its exact frozen canonical cohort")
        if (
            activation.parameters.get("checkpoints") != (6, 12)
            or activation.parameters.get("minimum_complete_slate") != 15
        ):
            raise ValueError("Opening Leader V0 parameters must match its frozen semantics")
        return tuple(
            MarketDataRequirement(
                feed_kind="bars",
                event_kind="bar_5m",
                instrument_id=instrument_id,
                cadence="5s",
                gaps_block=False,
                staleness_block=False,
            )
            for instrument_id in activation.universe
        )

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        prior = state if isinstance(state, Mapping) else {}
        session_value = prior.get("session")
        session: str | None = session_value if isinstance(session_value, str) else None
        emitted_value = prior.get("emitted", ())
        emitted = (
            {
                value
                for value in emitted_value
                if isinstance(value, int) and not isinstance(value, bool) and value in (6, 12)
            }
            if isinstance(emitted_value, tuple | list)
            else set()
        )
        bars: dict[str, dict[int, tuple[str, bool, float | None, float | None, int]]] = {}
        stored_bars = prior.get("bars")
        if isinstance(stored_bars, Mapping):
            for symbol, values in stored_bars.items():
                if not isinstance(symbol, str) or not isinstance(values, Mapping):
                    continue
                restored: dict[int, tuple[str, bool, float | None, float | None, int]] = {}
                for number_text, value in values.items():
                    if (
                        not isinstance(number_text, str)
                        or not number_text.isdigit()
                        or not isinstance(value, tuple | list)
                        or len(value) != 5
                        or not isinstance(value[0], str)
                        or not isinstance(value[1], bool)
                        or not isinstance(value[4], int)
                    ):
                        continue
                    opening = value[2] if isinstance(value[2], int | float) else None
                    close = value[3] if isinstance(value[3], int | float) else None
                    restored[int(number_text)] = (
                        value[0],
                        value[1],
                        None if opening is None else float(opening),
                        None if close is None else float(close),
                        value[4],
                    )
                bars[symbol] = restored
        lineage_value = prior.get("lineage", ())
        lineage = (
            [value for value in lineage_value if isinstance(value, str)]
            if isinstance(lineage_value, tuple | list)
            else []
        )

        incoming = tuple(
            event
            for event in batch.events
            if event.instrument_id in COHORT
            and event.feed_kind == "bars"
            and event.event_kind == "bar_5m"
        )
        outputs: list[IdeaOutput] = []
        output_lineages: list[tuple[str, ...]] = []

        def reset(new_session: str) -> None:
            nonlocal session, bars, lineage, emitted
            session = new_session
            bars = {}
            lineage = []
            emitted = set()

        def emit_ready() -> None:
            nonlocal bars, lineage
            for checkpoint in (6, 12):
                if checkpoint in emitted or not all(
                    checkpoint in bars.get(symbol, {}) for symbol in COHORT
                ):
                    continue
                rows: list[tuple[float, str]] = []
                for symbol in COHORT:
                    symbol_bars = bars.get(symbol, {})
                    required = tuple(symbol_bars.get(number) for number in range(1, checkpoint + 1))
                    if (
                        any(value is None or not value[1] for value in required)
                        or required[0] is None
                        or required[-1] is None
                        or required[0][2] is None
                        or required[-1][3] is None
                        or required[0][2] <= 0
                        or required[-1][3] <= 0
                    ):
                        continue
                    rows.append((10_000.0 * (required[-1][3] / required[0][2] - 1.0), symbol))
                emitted.add(checkpoint)
                if len(rows) < 15:
                    continue
                ranking = sorted(rows, key=lambda item: (-item[0], item[1]))
                leader = ranking[0]
                runner_up = ranking[1] if len(ranking) > 1 else None
                payload = cast(
                    Mapping[str, JsonValue],
                    {
                        "checkpoint": checkpoint,
                        "checkpoint_role": "primary" if checkpoint == 6 else "secondary",
                        "session": session,
                        "eligible": True,
                        "slate_size": len(ranking),
                        "rank_1": leader[1],
                        "rank_1_return_bps": leader[0],
                        "rank_2": runner_up[1] if runner_up else None,
                        "rank_1_minus_rank_2_bps": leader[0] - runner_up[0] if runner_up else None,
                        "selected_identity": "rank_1",
                        "direction": "LONG",
                        "sizing_basis": "one_share_reference",
                        "frozen_version": "opening-leader-continuation-recorder-v0",
                    },
                )
                as_of = max(bars[symbol][checkpoint][4] for symbol in COHORT)
                generated: tuple[IdeaOutput, ...] = (
                    Observation(
                        subject_instrument_id=leader[1], as_of_at_us=as_of, payload=payload
                    ),
                    Signal(subject_instrument_id=leader[1], as_of_at_us=as_of, payload=payload),
                    ProposedTrade(
                        subject_instrument_id=leader[1],
                        as_of_at_us=as_of,
                        payload={**payload, "action": "buy", "target": "long"},
                        legs=(
                            ProposedTradeLeg(
                                instrument_id=leader[1],
                                action="buy",
                                target="long",
                                quantity_value=1.0,
                                currency="USD",
                            ),
                        ),
                    ),
                )
                required_ids = {
                    bars[symbol][number][0]
                    for symbol in COHORT
                    for number in range(1, checkpoint + 1)
                    if number in bars[symbol]
                }
                output_lineage = tuple(item for item in lineage if item in required_ids)
                outputs.extend(generated)
                output_lineages.extend((output_lineage,) * len(generated))

        for event in incoming:
            event_session = event.payload.get("session")
            bar_number = event.payload.get("bar_number")
            if (
                not isinstance(event_session, str)
                or isinstance(bar_number, bool)
                or not isinstance(bar_number, int)
                or not 1 <= bar_number <= 78
            ):
                continue
            if session is None or event_session > session:
                emit_ready()
                reset(event_session)
            elif event_session < session:
                continue
            if event.event_id not in lineage:
                lineage.append(event.event_id)
            completeness = event.payload.get("source_completeness") == "complete"
            event_open = event.payload.get("open")
            event_close = event.payload.get("close")
            value = (
                event.event_id,
                completeness,
                float(event_open)
                if isinstance(event_open, int | float) and not isinstance(event_open, bool)
                else None,
                float(event_close)
                if isinstance(event_close, int | float) and not isinstance(event_close, bool)
                else None,
                event.event_at_us,
            )
            existing = bars.setdefault(event.instrument_id, {}).get(bar_number)
            if existing is None:
                bars[event.instrument_id][bar_number] = value
            elif existing != value:
                bars[event.instrument_id][bar_number] = (
                    existing[0],
                    False,
                    None,
                    None,
                    existing[4],
                )
        emit_ready()

        retained_ids = {
            value[0]
            for values in bars.values()
            for number, value in values.items()
            if 12 not in emitted and number <= 12
        }
        if 12 in emitted:
            bars = {}
            retained_ids = set()
        retained = tuple(item for item in lineage if item in retained_ids)
        latest_state = cast(
            JsonValue,
            {
                "session": session,
                "emitted": tuple(sorted(emitted)),
                "bars": {
                    symbol: {str(number): value for number, value in sorted(values.items())}
                    for symbol, values in sorted(bars.items())
                },
                "lineage": retained,
                "input_watermark": batch.input_watermark,
            },
        )
        return IdeaEvaluation(
            state=latest_state,
            outputs=tuple(outputs),
            retained_input_event_ids=retained,
            output_input_event_ids=tuple(output_lineages),
            interests=(),
        )


def create_plugin() -> OpeningLeaderContinuationV0:
    return OpeningLeaderContinuationV0()
