"""Sequential, transaction-isolated generic idea runner over normalized V2 evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from stocker_runtime.domain import (
    IdeaOutput,
    JsonValue,
    MarketEvent,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
    ensure_authority_free_json,
)
from stocker_runtime.ideas.contract import IdeaActivation, IdeaBatch, IdeaEvaluation, IdeaPlugin
from stocker_runtime.ideas.discovery import DiscoveredPlugin
from stocker_runtime.storage.connection import connect_v2

MAX_EVALUATION_NS = 50_000_000


class IdeaRunnerError(RuntimeError):
    """An activation or evaluation violates a generic Phase 4 invariant."""


@dataclass(frozen=True)
class ActivationResult:
    instance_id: str
    protected_data_class: ProtectedDataClass
    inserted: bool


@dataclass(frozen=True)
class EvaluationResult:
    instance_id: str
    advanced: bool
    output_count: int
    error_code: str | None = None


def _hash(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json(value: JsonValue) -> str:
    return canonical_json_bytes(value).decode()


class IdeaRunner:
    """Core-owned runner; plugins receive immutable DTOs and no privileged object."""

    def __init__(self, database_path: str | Path, plugins: tuple[DiscoveredPlugin, ...]) -> None:
        self.database_path = Path(database_path)
        self._discovered = plugins
        self._plugins_by_instance: dict[str, IdeaPlugin | object] = {}
        with connect_v2(self.database_path) as connection:
            rows = connection.execute(
                "SELECT instance_id, idea_id, idea_version, plugin_code_hash, parameters_hash, "
                "universe_hash FROM idea_instances WHERE deactivated_at_us IS NULL"
            ).fetchall()
        by_identity = {
            (
                plugin.manifest.idea_id,
                plugin.manifest.idea_version,
                plugin.code_hash,
                plugin.parameters_hash,
                plugin.universe_hash,
            ): plugin.plugin
            for plugin in plugins
        }
        for row in rows:
            identity = tuple(str(row[key]) for key in row.keys()[1:])
            if identity in by_identity:
                self._plugins_by_instance[str(row["instance_id"])] = by_identity[identity]

    def activate(
        self,
        *,
        run_id: str,
        plugin: DiscoveredPlugin,
        activated_at_us: int,
    ) -> ActivationResult:
        """Freeze an activation at the current causal watermark; never backfill earlier rows."""

        with connect_v2(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT mode, data_class, status FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if run is None or str(run["status"]) not in {"created", "running"}:
                raise IdeaRunnerError("activation requires an active known run")
            mode = RuntimeMode(str(run["mode"]))
            data_class = ProtectedDataClass(str(run["data_class"]))
            expected = (
                ProtectedDataClass.PROSPECTIVE
                if mode is RuntimeMode.PROSPECTIVE_RECORD
                else ProtectedDataClass.SHADOW
            )
            if data_class is not expected or mode not in plugin.manifest.modes:
                raise IdeaRunnerError(
                    "plugin activation crosses its mode or protected-data boundary"
                )
            missing = connection.execute(
                "SELECT instrument_id FROM (SELECT value AS instrument_id FROM json_each(?)) "
                "WHERE instrument_id NOT IN (SELECT instrument_id FROM instruments) LIMIT 1",
                (_json(plugin.config.universe),),
            ).fetchone()
            if missing is not None:
                raise IdeaRunnerError(
                    f"activation universe contains unknown instrument {missing[0]}"
                )
            watermark = int(
                connection.execute(
                    "SELECT coalesce(max(source_sequence), 0) FROM market_events WHERE run_id=?",
                    (run_id,),
                ).fetchone()[0]
            )
            identity_material = cast(
                JsonValue,
                {
                    "run_id": run_id,
                    "idea_id": plugin.manifest.idea_id,
                    "idea_version": plugin.manifest.idea_version,
                    "code_hash": plugin.code_hash,
                    "manifest_hash": plugin.manifest_hash,
                    "parameters_hash": plugin.parameters_hash,
                    "universe_hash": plugin.universe_hash,
                    "activated_at_us": activated_at_us,
                    "activated_after_source_sequence": watermark,
                    "data_class": data_class.value,
                },
            )
            instance_id = _hash(identity_material)
            requirements_json = _json(
                cast(JsonValue, [item.model_dump(mode="json") for item in plugin.requirements])
            )
            requirements_hash = hashlib.sha256(requirements_json.encode()).hexdigest()
            connection.execute(
                "INSERT INTO idea_plugins(idea_id, idea_version, api_version, display_name, "
                "description, manifest_hash, code_hash, manifest_json, discovered_at_us) "
                "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(idea_id, idea_version, code_hash) DO NOTHING",
                (
                    plugin.manifest.idea_id,
                    plugin.manifest.idea_version,
                    plugin.manifest.display_name,
                    plugin.manifest.description,
                    plugin.manifest_hash,
                    plugin.code_hash,
                    plugin.manifest_json,
                    activated_at_us,
                ),
            )
            persisted = connection.execute(
                "SELECT manifest_hash, code_hash FROM idea_plugins "
                "WHERE idea_id=? AND idea_version=? AND code_hash=?",
                (plugin.manifest.idea_id, plugin.manifest.idea_version, plugin.code_hash),
            ).fetchone()
            if persisted is None or tuple(persisted) != (plugin.manifest_hash, plugin.code_hash):
                raise IdeaRunnerError("configured plugin identity collides with persisted plugin")
            cursor = connection.execute(
                "INSERT INTO idea_instances(instance_id, idea_id, idea_version, run_id, mode, "
                "parameters_json, parameters_hash, plugin_code_hash, manifest_hash, universe_json, "
                "universe_hash, requirements_json, requirements_hash, "
                "activated_after_source_sequence, activated_at_us, health, data_class) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "'healthy', ?) ON CONFLICT(instance_id) DO NOTHING",
                (
                    instance_id,
                    plugin.manifest.idea_id,
                    plugin.manifest.idea_version,
                    run_id,
                    mode.value,
                    _json(plugin.config.parameters),
                    plugin.parameters_hash,
                    plugin.code_hash,
                    plugin.manifest_hash,
                    _json(plugin.config.universe),
                    plugin.universe_hash,
                    requirements_json,
                    requirements_hash,
                    watermark,
                    activated_at_us,
                    data_class.value,
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                empty_state = "{}"
                connection.execute(
                    "INSERT INTO idea_checkpoints(instance_id, last_market_event_id, "
                    "last_source_sequence, state_json, "
                    "state_hash, updated_at_us) VALUES (?, NULL, ?, ?, ?, ?)",
                    (
                        instance_id,
                        None,
                        empty_state,
                        hashlib.sha256(empty_state.encode()).hexdigest(),
                        activated_at_us,
                    ),
                )
            connection.commit()
        self._plugins_by_instance[instance_id] = plugin.plugin
        return ActivationResult(instance_id, data_class, inserted)

    def run_once(self, *, now_us: int) -> tuple[EvaluationResult, ...]:
        """Evaluate each healthy instance independently and isolate every failure."""

        with connect_v2(self.database_path) as connection:
            instance_ids = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT instance_id FROM idea_instances WHERE deactivated_at_us IS NULL "
                    "AND health != 'disabled' ORDER BY instance_id"
                )
            )
        return tuple(self._run_instance(instance_id, now_us=now_us) for instance_id in instance_ids)

    def _run_instance(self, instance_id: str, *, now_us: int) -> EvaluationResult:
        plugin = self._plugins_by_instance.get(instance_id)
        if plugin is None or not hasattr(plugin, "evaluate"):
            return self._degrade(instance_id, now_us, "PLUGIN_UNAVAILABLE")
        try:
            context = self._load_batch(instance_id)
            if context is None:
                return EvaluationResult(instance_id, False, 0)
            activation, batch, state, event_ids, starting_checkpoint = context
            started = time.monotonic_ns()
            evaluation = cast(IdeaPlugin, plugin).evaluate(batch, state)
            elapsed = time.monotonic_ns() - started
            if elapsed > MAX_EVALUATION_NS:
                raise IdeaRunnerError("plugin evaluation exceeded 50ms observation bound")
            evaluation = IdeaEvaluation.model_validate(evaluation.model_dump(mode="python"))
            self._validate_evaluation(activation, cast(IdeaPlugin, plugin), batch, evaluation)
            self._commit_evaluation(
                activation, batch, evaluation, event_ids, starting_checkpoint, now_us
            )
            return EvaluationResult(instance_id, True, len(evaluation.outputs))
        except Exception as error:
            code = f"{type(error).__name__}:{error}".upper()[:64]
            return self._degrade(instance_id, now_us, code)

    def _load_batch(
        self, instance_id: str
    ) -> tuple[IdeaActivation, IdeaBatch, JsonValue, tuple[str, ...], str | None] | None:
        with connect_v2(self.database_path) as connection:
            row = connection.execute(
                "SELECT instance.*, checkpoint.last_market_event_id, "
                "checkpoint.last_source_sequence, checkpoint.state_json "
                "FROM idea_instances instance JOIN idea_checkpoints checkpoint USING(instance_id) "
                "WHERE instance.instance_id=?",
                (instance_id,),
            ).fetchone()
            if row is None:
                raise IdeaRunnerError("instance checkpoint is missing")
            requirements = json.loads(str(row["requirements_json"]))
            for requirement in requirements:
                if requirement["gaps_block"]:
                    gap = connection.execute(
                        "SELECT 1 FROM gaps gap JOIN subscriptions subscription "
                        "ON subscription.subscription_id=gap.subscription_id "
                        "WHERE gap.run_id=? AND gap.resolved_at_us IS NULL "
                        "AND subscription.instrument_id=? AND subscription.feed_kind=? LIMIT 1",
                        (row["run_id"], requirement["instrument_id"], requirement["feed_kind"]),
                    ).fetchone()
                    if gap is not None:
                        return None
            start_sequence = (
                int(row["last_source_sequence"])
                if row["last_source_sequence"] is not None
                else int(row["activated_after_source_sequence"])
            )
            candidates = connection.execute(
                "SELECT event.* FROM market_events event JOIN json_each(?) requirement "
                "ON event.instrument_id=json_extract(requirement.value, '$.instrument_id') "
                "AND event.feed_kind=json_extract(requirement.value, '$.feed_kind') "
                "WHERE event.run_id=? AND event.source_sequence>? "
                "ORDER BY event.source_sequence LIMIT 256",
                (row["requirements_json"], row["run_id"], start_sequence),
            ).fetchall()
            rows = list(candidates)
            if not rows:
                return None
            events = tuple(
                MarketEvent(
                    event_id=str(item["event_id"]),
                    instrument_id=str(item["instrument_id"]),
                    feed_kind=str(item["feed_kind"]),
                    event_kind=str(item["event_kind"]),
                    event_at_us=int(item["event_at_us"]),
                    received_at_us=int(item["received_at_us"]),
                    payload=cast(Mapping[str, JsonValue], json.loads(str(item["payload_json"]))),
                )
                for item in rows
            )
            activation = IdeaActivation(
                instance_id=instance_id,
                parameters=cast(Mapping[str, JsonValue], json.loads(str(row["parameters_json"]))),
                parameters_hash=str(row["parameters_hash"]),
                plugin_code_hash=str(row["plugin_code_hash"]),
                activated_at_us=int(row["activated_at_us"]),
                run_id=str(row["run_id"]),
                protected_data_class=ProtectedDataClass(str(row["data_class"])),
                universe=tuple(json.loads(str(row["universe_json"]))),
            )
            batch = IdeaBatch(
                mode=RuntimeMode(str(row["mode"])),
                events=events,
                input_watermark=events[-1].event_id,
                causal_from_at_us=min(item.event_at_us for item in events),
                causal_through_at_us=max(item.event_at_us for item in events),
            )
            return (
                activation,
                batch,
                cast(JsonValue, json.loads(str(row["state_json"]))),
                tuple(item.event_id for item in events),
                None if row["last_market_event_id"] is None else str(row["last_market_event_id"]),
            )

    @staticmethod
    def _validate_evaluation(
        activation: IdeaActivation,
        plugin: IdeaPlugin,
        batch: IdeaBatch,
        evaluation: IdeaEvaluation,
    ) -> None:
        if len(evaluation.state_json()) > plugin.manifest.maximum_state_bytes:
            raise IdeaRunnerError("plugin state exceeds manifest bound")
        if len(evaluation.outputs) > min(plugin.manifest.maximum_outputs_per_batch, 256):
            raise IdeaRunnerError("plugin outputs exceed manifest bound")
        allowed = {item.value for item in plugin.manifest.output_kinds}
        for output in evaluation.outputs:
            if output.kind not in allowed:
                raise IdeaRunnerError("plugin emitted a forbidden output kind")
            if output.subject_instrument_id not in activation.universe:
                raise IdeaRunnerError("plugin output subject is outside the activation universe")
            if not batch.causal_from_at_us <= output.as_of_at_us <= batch.causal_through_at_us:
                raise IdeaRunnerError("plugin output as-of time is outside its causal input")
            if output.kind in {"proposed_position", "proposed_trade"}:
                if getattr(output, "status", None) != "unapproved":
                    raise IdeaRunnerError("proposal is not explicitly unapproved")
                ensure_authority_free_json(output.payload)

    def _commit_evaluation(
        self,
        activation: IdeaActivation,
        batch: IdeaBatch,
        evaluation: IdeaEvaluation,
        event_ids: tuple[str, ...],
        starting_checkpoint: str | None,
        now_us: int,
    ) -> None:
        state_json = evaluation.state_json().decode()
        state_hash = hashlib.sha256(state_json.encode()).hexdigest()
        input_events_hash = _hash(cast(JsonValue, event_ids))
        with connect_v2(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT last_market_event_id FROM idea_checkpoints WHERE instance_id=?",
                (activation.instance_id,),
            ).fetchone()
            current_id = None if current is None or current[0] is None else str(current[0])
            if current_id != starting_checkpoint:
                raise IdeaRunnerError("checkpoint changed during plugin evaluation")
            for ordinal, output in enumerate(evaluation.outputs):
                self._insert_output(
                    connection,
                    activation,
                    batch,
                    output,
                    ordinal,
                    event_ids,
                    input_events_hash,
                    now_us,
                )
            watermark_row = connection.execute(
                "SELECT source_sequence FROM market_events WHERE event_id=? AND run_id=?",
                (batch.input_watermark, activation.run_id),
            ).fetchone()
            if watermark_row is None:
                raise IdeaRunnerError("input watermark disappeared before checkpoint commit")
            changed = connection.execute(
                "UPDATE idea_checkpoints SET last_market_event_id=?, last_source_sequence=?, "
                "state_json=?, state_hash=?, "
                "last_success_at_us=?, updated_at_us=?, consecutive_failures=0 WHERE instance_id=? "
                "AND last_market_event_id IS ?",
                (
                    batch.input_watermark,
                    int(watermark_row[0]),
                    state_json,
                    state_hash,
                    now_us,
                    now_us,
                    activation.instance_id,
                    starting_checkpoint,
                ),
            ).rowcount
            if changed != 1:
                raise IdeaRunnerError("checkpoint compare-and-swap failed")
            connection.execute(
                "UPDATE idea_instances SET health='healthy', error_code=NULL WHERE instance_id=?",
                (activation.instance_id,),
            )
            connection.commit()

    @staticmethod
    def _insert_output(
        connection: sqlite3.Connection,
        activation: IdeaActivation,
        batch: IdeaBatch,
        output: IdeaOutput,
        ordinal: int,
        event_ids: tuple[str, ...],
        input_events_hash: str,
        now_us: int,
    ) -> None:
        payload_json = output.payload_json().decode()
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
        identity = cast(
            JsonValue,
            {
                "instance_id": activation.instance_id,
                "input_watermark": batch.input_watermark,
                "input_event_ids": event_ids,
                "kind": output.kind,
                "ordinal": ordinal,
                "as_of_at_us": output.as_of_at_us,
                "payload_hash": payload_hash,
            },
        )
        output_id = _hash(identity)
        authority = (
            "unapproved" if output.kind in {"proposed_position", "proposed_trade"} else "recorded"
        )
        content_hash = _hash(
            cast(
                JsonValue,
                {
                    "identity": identity,
                    "subject": output.subject_instrument_id,
                    "authority_status": authority,
                    "data_class": activation.protected_data_class.value,
                },
            )
        )
        try:
            connection.execute(
                "INSERT INTO idea_outputs(output_id, run_id, instance_id, output_kind, "
                "subject_instrument_id, emitted_at_us, as_of_at_us, first_input_event_id, "
                "last_input_event_id, input_watermark, input_events_hash, output_ordinal, "
                "payload_json, payload_hash, content_hash, data_class, authority_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    output_id,
                    activation.run_id,
                    activation.instance_id,
                    output.kind,
                    output.subject_instrument_id,
                    now_us,
                    output.as_of_at_us,
                    event_ids[0],
                    event_ids[-1],
                    batch.input_watermark,
                    input_events_hash,
                    ordinal,
                    payload_json,
                    payload_hash,
                    content_hash,
                    activation.protected_data_class.value,
                    authority,
                ),
            )
        except sqlite3.IntegrityError as error:
            existing = connection.execute(
                "SELECT content_hash FROM idea_outputs WHERE instance_id=? AND input_watermark=? "
                "AND output_kind=? AND output_ordinal=? AND as_of_at_us=?",
                (
                    activation.instance_id,
                    batch.input_watermark,
                    output.kind,
                    ordinal,
                    output.as_of_at_us,
                ),
            ).fetchone()
            if existing is None or str(existing[0]) != content_hash:
                raise IdeaRunnerError("deterministic output identity collision") from error
        for input_ordinal, event_id in enumerate(event_ids):
            connection.execute(
                "INSERT INTO idea_output_inputs(output_id, event_id, input_ordinal) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT DO NOTHING",
                (output_id, event_id, input_ordinal),
            )

    def _degrade(self, instance_id: str, now_us: int, code: str) -> EvaluationResult:
        with connect_v2(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE idea_instances SET health='degraded', error_code=? WHERE instance_id=?",
                (code, instance_id),
            )
            connection.execute(
                "UPDATE idea_checkpoints SET consecutive_failures=consecutive_failures+1, "
                "updated_at_us=? WHERE instance_id=?",
                (now_us, instance_id),
            )
            incident_id = _hash(
                cast(
                    JsonValue,
                    {
                        "instance_id": instance_id,
                        "code": code,
                        "opened_at_us": now_us,
                    },
                )
            )
            run = connection.execute(
                "SELECT run_id FROM idea_instances WHERE instance_id=?", (instance_id,)
            ).fetchone()
            if run is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO incidents(incident_id, run_id, scope, severity, code, "
                    "plugin_instance_id, opened_at_us, details_json) VALUES (?, ?, 'idea_plugin', "
                    "'degraded', ?, ?, ?, '{}')",
                    (incident_id, run[0], code, instance_id, now_us),
                )
            connection.commit()
        return EvaluationResult(instance_id, False, 0, code)
