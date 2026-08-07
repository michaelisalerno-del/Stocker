"""Frozen causal Opening Leader Continuation V0 reference plugin.

Retained semantics: canonical 20-stock cohort, C6/C12 checkpoints, causal complete-bar
admission, open-to-checkpoint return ranking, symbol tie-break, minimum 15-stock slate,
rank-one LONG selection, and unapproved proposal status supplied by the domain contract.

Deferred deliberately: legacy option snapshots, later observation schedules, shadow
valuation, M1C context, report/artifact graphs, repository projections, and all broker UI.
Protected outcomes are never read and therefore cannot tune these frozen parameters.
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
                feed_kind="bars_5m",
                instrument_id=instrument_id,
                cadence="5m",
                gaps_block=True,
                staleness_block=True,
            )
            for instrument_id in activation.universe
        )

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        prior = state if isinstance(state, Mapping) else {}
        session_value = prior.get("session")
        current_session = session_value if isinstance(session_value, str) else None
        emitted_value = prior.get("emitted")
        emitted = {
            int(value)
            for value in (emitted_value if isinstance(emitted_value, tuple | list) else ())
            if isinstance(value, int) and not isinstance(value, bool) and value in (6, 12)
        }
        bars: dict[int, dict[str, tuple[float, float, str, int] | None]] = {6: {}, 12: {}}
        checkpoint_lineage: dict[int, list[str]] = {6: [], 12: []}
        lineage_value = prior.get("lineage")
        lineage = [
            value
            for value in (lineage_value if isinstance(lineage_value, tuple | list) else ())
            if isinstance(value, str)
        ]
        stored_bars = prior.get("bars")
        stored_checkpoint_lineage = prior.get("checkpoint_lineage")
        if isinstance(stored_bars, Mapping):
            for checkpoint in (6, 12):
                stored_checkpoint = stored_bars.get(str(checkpoint))
                if not isinstance(stored_checkpoint, Mapping):
                    continue
                for symbol, value in stored_checkpoint.items():
                    if not isinstance(symbol, str):
                        continue
                    if value is None:
                        bars[checkpoint][symbol] = None
                    elif isinstance(value, tuple | list) and len(value) == 4:
                        stored_open, stored_close, stored_event_id, stored_event_at_us = value
                        if (
                            isinstance(stored_open, int | float)
                            and not isinstance(stored_open, bool)
                            and isinstance(stored_close, int | float)
                            and not isinstance(stored_close, bool)
                            and isinstance(stored_event_id, str)
                            and isinstance(stored_event_at_us, int)
                            and not isinstance(stored_event_at_us, bool)
                        ):
                            bars[checkpoint][symbol] = (
                                float(stored_open),
                                float(stored_close),
                                stored_event_id,
                                stored_event_at_us,
                            )
        if isinstance(stored_checkpoint_lineage, Mapping):
            for checkpoint in (6, 12):
                values = stored_checkpoint_lineage.get(str(checkpoint))
                if isinstance(values, tuple | list):
                    checkpoint_lineage[checkpoint] = [
                        value for value in values if isinstance(value, str)
                    ]

        outputs: list[IdeaOutput] = []
        output_lineages: list[tuple[str, ...]] = []

        def emit_ready() -> None:
            for checkpoint in (6, 12):
                if checkpoint in emitted:
                    continue
                rows = [
                    (instrument, values)
                    for instrument, values in bars[checkpoint].items()
                    if values is not None
                ]
                if len(rows) < 15:
                    continue
                ranking = sorted(
                    (
                        (10_000.0 * (values[1] / values[0] - 1.0), instrument)
                        for instrument, values in rows
                    ),
                    key=lambda item: (-item[0], item[1]),
                )
                leader = ranking[0]
                runner_up = ranking[1] if len(ranking) > 1 else None
                payload = cast(
                    Mapping[str, JsonValue],
                    {
                        "checkpoint": checkpoint,
                        "checkpoint_role": "primary" if checkpoint == 6 else "secondary",
                        "session": current_session,
                        "eligible": True,
                        "slate_size": len(ranking),
                        "rank_1": leader[1],
                        "rank_1_return_bps": leader[0],
                        "rank_2": runner_up[1] if runner_up else None,
                        "rank_1_minus_rank_2_bps": (
                            leader[0] - runner_up[0] if runner_up is not None else None
                        ),
                        "selected_identity": "rank_1",
                        "direction": "LONG",
                        "sizing_basis": "one_share_reference",
                        "frozen_version": "opening-leader-continuation-recorder-v0",
                    },
                )
                as_of = max(values[3] for _, values in rows)
                outputs.extend(
                    (
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
                )
                used = set(checkpoint_lineage[checkpoint])
                output_lineage = tuple(event_id for event_id in lineage if event_id in used)
                output_lineages.extend((output_lineage, output_lineage, output_lineage))
                emitted.add(checkpoint)
                bars[checkpoint] = {}
                checkpoint_lineage[checkpoint] = []

        def remember_event(event_id: str, checkpoint: int) -> None:
            if event_id not in lineage:
                lineage.append(event_id)
            if event_id not in checkpoint_lineage[checkpoint]:
                checkpoint_lineage[checkpoint].append(event_id)

        for event in batch.events:
            payload = event.payload
            if (
                event.instrument_id not in COHORT
                or event.feed_kind != "bars_5m"
                or payload.get("source_completeness") != "complete"
            ):
                continue
            session = payload.get("session")
            event_checkpoint = payload.get("checkpoint")
            opening = payload.get("regular_session_open")
            close = payload.get("checkpoint_close")
            if (
                not isinstance(session, str)
                or not isinstance(event_checkpoint, int)
                or isinstance(event_checkpoint, bool)
                or event_checkpoint not in (6, 12)
                or not isinstance(opening, int | float)
                or isinstance(opening, bool)
                or not isinstance(close, int | float)
                or isinstance(close, bool)
                or opening <= 0
                or close <= 0
            ):
                continue
            if current_session is not None and session < current_session:
                continue
            if current_session is None or session > current_session:
                emit_ready()
                current_session = session
                emitted.clear()
                bars = {6: {}, 12: {}}
                checkpoint_lineage = {6: [], 12: []}
                lineage = []
            resolution = payload.get("duplicate_resolution")
            existing = bars[event_checkpoint].get(event.instrument_id, "missing")

            if resolution == "unresolved":
                if existing is not None:
                    remember_event(event.event_id, event_checkpoint)
                bars[event_checkpoint][event.instrument_id] = None
                continue
            if existing == "missing" or resolution == "causally_resolved":
                remember_event(event.event_id, event_checkpoint)
                bars[event_checkpoint][event.instrument_id] = (
                    float(opening),
                    float(close),
                    event.event_id,
                    event.event_at_us,
                )
            elif existing is not None:
                remember_event(event.event_id, event_checkpoint)
                bars[event_checkpoint][event.instrument_id] = None

        emit_ready()
        retained_set = {event_id for values in checkpoint_lineage.values() for event_id in values}
        retained = tuple(event_id for event_id in lineage if event_id in retained_set)
        latest_state = cast(
            JsonValue,
            {
                "session": current_session,
                "emitted": tuple(sorted(emitted)),
                "bars": {
                    str(checkpoint): {
                        symbol: values for symbol, values in sorted(bars[checkpoint].items())
                    }
                    for checkpoint in (6, 12)
                },
                "checkpoint_lineage": {
                    str(checkpoint): tuple(checkpoint_lineage[checkpoint]) for checkpoint in (6, 12)
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
        )


def create_plugin() -> OpeningLeaderContinuationV0:
    """Create the sole reviewed plugin object exposed by this module."""

    return OpeningLeaderContinuationV0()
