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
from datetime import UTC, datetime, time, timedelta
from typing import cast
from zoneinfo import ZoneInfo

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
_NEW_YORK = ZoneInfo("America/New_York")
_RAW_BAR_US = 5_000_000

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
                feed_kind="bars",
                instrument_id=instrument_id,
                cadence="5s",
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
        opens: dict[str, tuple[float, str, int] | None] = {}
        closes: dict[int, dict[str, tuple[float, str, int] | None]] = {6: {}, 12: {}}
        progress: dict[int, dict[str, str]] = {6: {}, 12: {}}
        open_lineage: list[str] = []
        checkpoint_lineage: dict[int, list[str]] = {6: [], 12: []}
        lineage_value = prior.get("lineage")
        lineage = [
            value
            for value in (lineage_value if isinstance(lineage_value, tuple | list) else ())
            if isinstance(value, str)
        ]

        def restore_value(value: object) -> tuple[float, str, int] | None:
            if value is None:
                return None
            if not isinstance(value, tuple | list) or len(value) != 3:
                return None
            number, event_id, event_at_us = value
            if (
                isinstance(number, bool)
                or not isinstance(number, int | float)
                or not isinstance(event_id, str)
                or isinstance(event_at_us, bool)
                or not isinstance(event_at_us, int)
            ):
                return None
            return float(number), event_id, event_at_us

        stored_opens = prior.get("opens")
        if isinstance(stored_opens, Mapping):
            for symbol, value in stored_opens.items():
                if isinstance(symbol, str):
                    opens[symbol] = restore_value(value)
        stored_closes = prior.get("closes")
        if isinstance(stored_closes, Mapping):
            for checkpoint in (6, 12):
                values = stored_closes.get(str(checkpoint))
                if isinstance(values, Mapping):
                    for symbol, value in values.items():
                        if isinstance(symbol, str):
                            closes[checkpoint][symbol] = restore_value(value)
        stored_progress = prior.get("progress")
        if isinstance(stored_progress, Mapping):
            for checkpoint in (6, 12):
                values = stored_progress.get(str(checkpoint))
                if isinstance(values, Mapping):
                    progress[checkpoint] = {
                        str(symbol): str(event_id)
                        for symbol, event_id in values.items()
                        if isinstance(symbol, str) and isinstance(event_id, str)
                    }
        stored_open_lineage = prior.get("open_lineage")
        if isinstance(stored_open_lineage, tuple | list):
            open_lineage = [value for value in stored_open_lineage if isinstance(value, str)]
        stored_checkpoint_lineage = prior.get("checkpoint_lineage")
        if isinstance(stored_checkpoint_lineage, Mapping):
            for checkpoint in (6, 12):
                values = stored_checkpoint_lineage.get(str(checkpoint))
                if isinstance(values, tuple | list):
                    checkpoint_lineage[checkpoint] = [
                        value for value in values if isinstance(value, str)
                    ]

        outputs: list[IdeaOutput] = []
        output_lineages: list[tuple[str, ...]] = []

        def remember(event_id: str) -> None:
            if event_id not in lineage:
                lineage.append(event_id)

        def remember_open(event_id: str) -> None:
            remember(event_id)
            if event_id not in open_lineage:
                open_lineage.append(event_id)

        def remember_checkpoint(event_id: str, checkpoint: int) -> None:
            remember(event_id)
            if event_id not in checkpoint_lineage[checkpoint]:
                checkpoint_lineage[checkpoint].append(event_id)

        def boundary_us(checkpoint: int) -> int:
            if current_session is None:
                raise ValueError("checkpoint session is absent")
            session_date = datetime.fromisoformat(current_session).date()
            local_open = datetime.combine(session_date, time(9, 30), tzinfo=_NEW_YORK)
            return int((local_open + timedelta(minutes=5 * checkpoint)).timestamp() * 1_000_000)

        def emit_ready() -> None:
            for checkpoint in (6, 12):
                if checkpoint in emitted:
                    continue
                if set(progress[checkpoint]) != set(COHORT):
                    continue
                rows = [
                    (instrument, opens.get(instrument), closes[checkpoint].get(instrument))
                    for instrument in COHORT
                    if opens.get(instrument) is not None
                    and closes[checkpoint].get(instrument) is not None
                ]
                if len(rows) < 15:
                    emitted.add(checkpoint)
                    closes[checkpoint] = {}
                    progress[checkpoint] = {}
                    checkpoint_lineage[checkpoint] = []
                    continue
                ranking = sorted(
                    (
                        (10_000.0 * (close[0] / opening[0] - 1.0), instrument)
                        for instrument, opening, close in rows
                        if opening is not None and close is not None
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
                as_of = boundary_us(checkpoint)
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
                used = set(open_lineage) | set(checkpoint_lineage[checkpoint])
                output_lineage = tuple(event_id for event_id in lineage if event_id in used)
                output_lineages.extend((output_lineage, output_lineage, output_lineage))
                emitted.add(checkpoint)
                closes[checkpoint] = {}
                progress[checkpoint] = {}
                checkpoint_lineage[checkpoint] = []

        for event in batch.events:
            payload = event.payload
            if (
                event.instrument_id not in COHORT
                or event.feed_kind != "bars"
                or event.event_kind != "bar"
            ):
                continue
            event_datetime = datetime.fromtimestamp(event.event_at_us / 1_000_000, tz=UTC)
            local_event = event_datetime.astimezone(_NEW_YORK)
            session = local_event.date().isoformat()
            local_open = datetime.combine(local_event.date(), time(9, 30), tzinfo=_NEW_YORK)
            session_open_us = int(local_open.timestamp() * 1_000_000)
            opening = payload.get("open")
            close = payload.get("close")
            if (
                not isinstance(opening, int | float)
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
                opens = {}
                closes = {6: {}, 12: {}}
                progress = {6: {}, 12: {}}
                open_lineage = []
                checkpoint_lineage = {6: [], 12: []}
                lineage = []

            if event.event_at_us == session_open_us:
                existing_open = opens.get(event.instrument_id, "missing")
                if existing_open == "missing":
                    remember_open(event.event_id)
                    opens[event.instrument_id] = (
                        float(opening),
                        event.event_id,
                        event.event_at_us,
                    )
                elif existing_open is not None and existing_open[0] != float(opening):
                    remember_open(event.event_id)
                    opens[event.instrument_id] = None

            for checkpoint in (6, 12):
                checkpoint_boundary = boundary_us(checkpoint)
                close_start = checkpoint_boundary - _RAW_BAR_US
                if event.event_at_us == close_start:
                    existing_close = closes[checkpoint].get(event.instrument_id, "missing")
                    if existing_close == "missing":
                        remember_checkpoint(event.event_id, checkpoint)
                        closes[checkpoint][event.instrument_id] = (
                            float(close),
                            event.event_id,
                            event.event_at_us,
                        )
                    elif existing_close is not None and existing_close[0] != float(close):
                        remember_checkpoint(event.event_id, checkpoint)
                        closes[checkpoint][event.instrument_id] = None
                if (
                    event.event_at_us >= checkpoint_boundary
                    or (
                        event.event_at_us == close_start
                        and event.received_at_us >= checkpoint_boundary
                    )
                ) and event.instrument_id not in progress[checkpoint]:
                    remember_checkpoint(event.event_id, checkpoint)
                    progress[checkpoint][event.instrument_id] = event.event_id

        emit_ready()
        retained_set = set(open_lineage) | {
            event_id for values in checkpoint_lineage.values() for event_id in values
        }
        if 12 in emitted:
            opens = {}
            open_lineage = []
            retained_set = {
                event_id for values in checkpoint_lineage.values() for event_id in values
            }
        retained = tuple(event_id for event_id in lineage if event_id in retained_set)
        latest_state = cast(
            JsonValue,
            {
                "session": current_session,
                "emitted": tuple(sorted(emitted)),
                "opens": {symbol: value for symbol, value in sorted(opens.items())},
                "closes": {
                    str(checkpoint): dict(sorted(closes[checkpoint].items()))
                    for checkpoint in (6, 12)
                },
                "progress": {
                    str(checkpoint): dict(sorted(progress[checkpoint].items()))
                    for checkpoint in (6, 12)
                },
                "open_lineage": tuple(open_lineage),
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
