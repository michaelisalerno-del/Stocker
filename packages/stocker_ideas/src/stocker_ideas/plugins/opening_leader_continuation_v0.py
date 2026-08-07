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
    RuntimeMode,
    Signal,
)
from stocker_runtime.ideas import (
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
                "items": {"type": "integer", "enum": (6, 12)},
            },
            "minimum_complete_slate": {"type": "integer", "enum": (15,)},
        },
    },
    maximum_state_bytes=65_536,
    maximum_outputs_per_batch=3,
)


class OpeningLeaderContinuationV0:
    @property
    def manifest(self) -> IdeaManifest:
        return MANIFEST

    def requirements(self, activation: IdeaActivation) -> tuple[MarketDataRequirement, ...]:
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
        del state
        by_checkpoint: dict[tuple[str, int], tuple[str, float, float]] = {}
        duplicate_keys: set[tuple[str, int]] = set()
        for event in batch.events:
            payload = event.payload
            if (
                event.instrument_id not in COHORT
                or event.feed_kind != "bars_5m"
                or payload.get("source_completeness") != "complete"
                or payload.get("duplicate_resolution") == "unresolved"
            ):
                continue
            session = payload.get("session")
            checkpoint = payload.get("checkpoint")
            opening = payload.get("regular_session_open")
            close = payload.get("checkpoint_close")
            if (
                not isinstance(session, str)
                or not isinstance(checkpoint, int)
                or isinstance(checkpoint, bool)
                or checkpoint not in (6, 12)
                or not isinstance(opening, int | float)
                or isinstance(opening, bool)
                or not isinstance(close, int | float)
                or isinstance(close, bool)
                or opening <= 0
                or close <= 0
            ):
                continue
            key = (event.instrument_id, checkpoint)
            if key in duplicate_keys:
                continue
            if key in by_checkpoint:
                by_checkpoint.pop(key)
                duplicate_keys.add(key)
                continue
            by_checkpoint[key] = (session, float(opening), float(close))

        outputs: list[IdeaOutput] = []
        latest_state: dict[str, JsonValue] = {"input_watermark": batch.input_watermark}
        for checkpoint in (6, 12):
            rows = [
                (instrument, values)
                for (instrument, candidate), values in by_checkpoint.items()
                if candidate == checkpoint
            ]
            if not rows:
                continue
            sessions = {values[0] for _, values in rows}
            eligible = len(rows) >= 15 and len(sessions) == 1
            ranking = sorted(
                (
                    (10_000.0 * (values[2] / values[1] - 1.0), instrument)
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
                    "session": next(iter(sessions)) if len(sessions) == 1 else None,
                    "eligible": eligible,
                    "slate_size": len(ranking),
                    "rank_1": leader[1],
                    "rank_1_return_bps": leader[0],
                    "rank_2": runner_up[1] if runner_up else None,
                    "rank_1_minus_rank_2_bps": (
                        leader[0] - runner_up[0] if runner_up is not None else None
                    ),
                    "selected_identity": "rank_1",
                    "direction": "LONG",
                    "frozen_version": "opening-leader-continuation-recorder-v0",
                },
            )
            as_of = batch.causal_through_at_us
            outputs.append(
                Observation(subject_instrument_id=leader[1], as_of_at_us=as_of, payload=payload)
            )
            if eligible:
                outputs.extend(
                    (
                        Signal(subject_instrument_id=leader[1], as_of_at_us=as_of, payload=payload),
                        ProposedTrade(
                            subject_instrument_id=leader[1],
                            as_of_at_us=as_of,
                            payload={**payload, "action": "buy", "target": "long"},
                        ),
                    )
                )
            latest_state[f"checkpoint_{checkpoint}"] = cast(JsonValue, payload)
            break  # one causal checkpoint per bounded evaluation
        return IdeaEvaluation(state=cast(JsonValue, latest_state), outputs=tuple(outputs))


def create_plugin() -> OpeningLeaderContinuationV0:
    """Create the sole reviewed plugin object exposed by this module."""

    return OpeningLeaderContinuationV0()
