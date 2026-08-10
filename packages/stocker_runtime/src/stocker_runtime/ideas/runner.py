"""Sequential, transaction-isolated generic idea runner over normalized V2 evidence."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import sqlite3
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import cast

from stocker_runtime.domain import (
    IdeaOutput,
    JsonValue,
    MarketEvent,
    ProposedTrade,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
    ensure_authority_free_json,
)
from stocker_runtime.ideas.contract import (
    DiscoveryReceipt,
    IdeaActivation,
    IdeaBatch,
    IdeaEvaluation,
    IdeaManifest,
    IdeaPlugin,
    MarketDataInterest,
    MarketDataRequirement,
)
from stocker_runtime.ideas.discovery import DiscoveredPlugin
from stocker_runtime.ideas.identity import (
    deterministic_idea_output_content_hash,
    deterministic_idea_output_id,
)
from stocker_runtime.storage.connection import connect_v2

MAX_EVALUATION_NS = 50_000_000
MAX_NONTERMINAL_INTERESTS_PER_INSTANCE = 64
MAX_NONTERMINAL_INTERESTS_PER_RUN = 256
MAX_EVALUATION_SECONDS = MAX_EVALUATION_NS / 1_000_000_000
WORKER_START_SECONDS = 5.0
RETRY_BASE_US = 1_000_000
RETRY_MAX_US = 60_000_000


class IdeaRunnerError(RuntimeError):
    """An activation or evaluation violates a generic Phase 4 invariant."""


def _merge_batch_requirements(
    requirements: tuple[MarketDataRequirement, ...],
) -> tuple[MarketDataRequirement, ...]:
    """Merge duplicate static and discovered needs without weakening a blocker."""

    merged: dict[tuple[str, str], MarketDataRequirement] = {}
    for requirement in requirements:
        key = (requirement.instrument_id, requirement.feed_kind)
        prior = merged.get(key)
        if prior is None:
            merged[key] = requirement
            continue
        if prior.cadence == requirement.cadence:
            cadence = prior.cadence
        elif prior.cadence == "snapshot":
            cadence = requirement.cadence
        elif requirement.cadence == "snapshot":
            cadence = prior.cadence
        elif prior.cadence == "stream":
            cadence = requirement.cadence
        elif requirement.cadence == "stream":
            cadence = prior.cadence
        else:
            raise IdeaRunnerError(f"conflicting requirement cadence for {key}")
        merged[key] = MarketDataRequirement(
            feed_kind=requirement.feed_kind,
            event_kind=(
                requirement.event_kind if prior.event_kind == requirement.event_kind else None
            ),
            instrument_id=requirement.instrument_id,
            cadence=cadence,
            gaps_block=prior.gaps_block or requirement.gaps_block,
            staleness_block=prior.staleness_block or requirement.staleness_block,
        )
    return tuple(sorted(merged.values(), key=lambda item: item.to_canonical_json()))


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


def _worker_main(connection: Connection, plugin: IdeaPlugin) -> None:
    """Run reviewed plugin code outside the recorder process; this is not a sandbox."""

    connection.send(("ready", None))
    while True:
        try:
            message = connection.recv()
        except EOFError:
            return
        if message is None:
            return
        batch, state = message
        try:
            evaluation = plugin.evaluate(batch, state)
            connection.send(("ok", evaluation.model_dump(mode="python")))
        except BaseException as error:
            connection.send(("error", f"{type(error).__name__}:{error}"))


class _PluginWorker:
    def __init__(self, plugin: IdeaPlugin) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_worker_main, args=(child, plugin), daemon=True)
        process.start()
        child.close()
        if not parent.poll(WORKER_START_SECONDS) or parent.recv()[0] != "ready":
            process.terminate()
            process.join(timeout=1)
            parent.close()
            raise IdeaRunnerError("plugin worker failed to start")
        self.connection = parent
        self.process = process

    def evaluate(self, batch: IdeaBatch, state: JsonValue) -> IdeaEvaluation:
        deadline = time.monotonic() + MAX_EVALUATION_SECONDS
        send_error: list[BaseException] = []

        def send() -> None:
            try:
                self.connection.send((batch, state))
            except BaseException as error:
                send_error.append(error)

        sender = threading.Thread(target=send, daemon=True, name="stocker-plugin-send")
        sender.start()
        sender.join(max(0.0, deadline - time.monotonic()))
        if sender.is_alive():
            self.terminate()
            raise IdeaRunnerError("plugin evaluation exceeded 50ms end-to-end bound during send")
        if send_error:
            self.terminate()
            raise IdeaRunnerError(f"plugin request delivery failed: {send_error[0]}")
        if not self.connection.poll(max(0.0, deadline - time.monotonic())):
            self.terminate()
            raise IdeaRunnerError("plugin evaluation exceeded 50ms end-to-end bound")
        response: list[object] = []
        receive_error: list[BaseException] = []

        def receive() -> None:
            try:
                response.append(self.connection.recv())
            except BaseException as error:
                receive_error.append(error)

        receiver = threading.Thread(target=receive, daemon=True, name="stocker-plugin-receive")
        receiver.start()
        receiver.join(max(0.0, deadline - time.monotonic()))
        if receiver.is_alive() or not response:
            self.terminate()
            detail = f": {receive_error[0]}" if receive_error else ""
            raise IdeaRunnerError(f"plugin response delivery failed within 50ms{detail}")
        status, payload = cast(tuple[object, object], response[0])
        if status != "ok":
            raise IdeaRunnerError(str(payload))
        return IdeaEvaluation.model_validate(payload)

    def terminate(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
        self.connection.close()


def _hash(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json(value: JsonValue) -> str:
    return canonical_json_bytes(value).decode()


def _verified_json(text: str, expected_hash: str, label: str) -> object:
    try:
        value = json.loads(text)
        canonical = canonical_json_bytes(cast(JsonValue, value))
    except Exception as error:
        raise IdeaRunnerError(f"{label} is not valid canonical JSON") from error
    if canonical.decode() != text:
        raise IdeaRunnerError(f"{label} is not canonical JSON")
    if hashlib.sha256(canonical).hexdigest() != expected_hash:
        raise IdeaRunnerError(f"{label} hash mismatch")
    return value


class IdeaRunner:
    """Core-owned runner; plugins receive immutable DTOs and no privileged object."""

    def __init__(
        self,
        database_path: str | Path,
        plugins: tuple[DiscoveredPlugin, ...],
        *,
        run_id: str,
    ) -> None:
        self.database_path = Path(database_path)
        self._discovered = plugins
        self._plugins_by_instance: dict[str, IdeaPlugin | object] = {}
        self._discovered_by_instance: dict[str, DiscoveredPlugin] = {}
        self._workers: dict[str, _PluginWorker] = {}
        by_identity = {
            (
                plugin.manifest.idea_id,
                plugin.manifest.idea_version,
                plugin.code_hash,
                plugin.parameters_hash,
                plugin.universe_hash,
            ): plugin
            for plugin in plugins
        }
        with connect_v2(self.database_path) as connection:
            active_runs = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT run_id FROM runs WHERE status IN ('created','running') "
                    "AND run_id=? ORDER BY run_id",
                    (run_id,),
                )
            )
            if len(active_runs) != 1:
                raise IdeaRunnerError("idea runner requires exactly one active bound run")
            self.run_id = active_runs[0]
            rows = connection.execute(
                "SELECT instance.*, checkpoint.last_market_event_id, "
                "checkpoint.last_source_sequence, checkpoint.state_json, checkpoint.state_hash, "
                "checkpoint.state_input_event_ids_json "
                "FROM idea_instances instance JOIN idea_checkpoints checkpoint USING(instance_id) "
                "WHERE instance.run_id=? AND instance.deactivated_at_us IS NULL",
                (self.run_id,),
            ).fetchall()
            for row in rows:
                identity = (
                    str(row["idea_id"]),
                    str(row["idea_version"]),
                    str(row["plugin_code_hash"]),
                    str(row["parameters_hash"]),
                    str(row["universe_hash"]),
                )
                discovered = by_identity.get(identity)
                if discovered is not None:
                    self._verify_persisted_instance(connection, row, discovered)
                    instance_id = str(row["instance_id"])
                    self._plugins_by_instance[instance_id] = discovered.plugin
                    self._discovered_by_instance[instance_id] = discovered

    @staticmethod
    def _verify_persisted_instance(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        discovered: DiscoveredPlugin,
    ) -> None:
        plugin_row = connection.execute(
            "SELECT * FROM idea_plugins WHERE idea_id=? AND idea_version=? AND code_hash=?",
            (row["idea_id"], row["idea_version"], row["plugin_code_hash"]),
        ).fetchone()
        if plugin_row is None:
            raise IdeaRunnerError("persisted plugin identity is missing")
        manifest_value = _verified_json(
            str(plugin_row["manifest_json"]), str(plugin_row["manifest_hash"]), "manifest"
        )
        del manifest_value
        manifest = IdeaManifest.model_validate_json(str(plugin_row["manifest_json"]))
        if (
            str(plugin_row["manifest_hash"]) != discovered.manifest_hash
            or str(plugin_row["code_hash"]) != discovered.code_hash
            or str(row["manifest_hash"]) != discovered.manifest_hash
            or manifest != discovered.manifest
        ):
            raise IdeaRunnerError("persisted manifest or source binding mismatch")
        parameters = _verified_json(
            str(row["parameters_json"]), str(row["parameters_hash"]), "parameters"
        )
        universe = _verified_json(str(row["universe_json"]), str(row["universe_hash"]), "universe")
        requirements = _verified_json(
            str(row["requirements_json"]), str(row["requirements_hash"]), "requirements"
        )
        state = _verified_json(str(row["state_json"]), str(row["state_hash"]), "checkpoint state")
        del state
        if (
            parameters != json.loads(_json(discovered.config.parameters))
            or universe != json.loads(_json(discovered.config.universe))
            or requirements != [item.model_dump(mode="json") for item in discovered.requirements]
        ):
            raise IdeaRunnerError("persisted activation binding mismatch")
        lineage_text = str(row["state_input_event_ids_json"])
        try:
            lineage = json.loads(lineage_text)
        except json.JSONDecodeError as error:
            raise IdeaRunnerError("checkpoint lineage is invalid JSON") from error
        if (
            canonical_json_bytes(cast(JsonValue, lineage)).decode() != lineage_text
            or not isinstance(lineage, list)
            or len(lineage) > 256
            or len(set(lineage)) != len(lineage)
            or any(not isinstance(event_id, str) for event_id in lineage)
        ):
            raise IdeaRunnerError("checkpoint lineage is not canonical and bounded")
        if lineage:
            placeholders = ",".join("?" for _ in lineage)
            count = connection.execute(
                f"SELECT count(*) FROM market_events WHERE run_id=? "  # noqa: S608
                f"AND event_id IN ({placeholders})",
                (row["run_id"], *lineage),
            ).fetchone()[0]
            if int(count) != len(lineage):
                raise IdeaRunnerError("checkpoint lineage crosses its run binding")
        if row["last_market_event_id"] is not None:
            watermark = connection.execute(
                "SELECT coalesce(source_sequence, derived_after_source_sequence) "
                "FROM market_events WHERE event_id=? AND run_id=?",
                (row["last_market_event_id"], row["run_id"]),
            ).fetchone()
            if watermark is None or int(watermark[0]) != int(row["last_source_sequence"]):
                raise IdeaRunnerError("checkpoint event watermark binding mismatch")

    def activate(
        self,
        *,
        run_id: str,
        plugin: DiscoveredPlugin,
        activated_at_us: int,
    ) -> ActivationResult:
        """Freeze an activation at the current causal watermark; never backfill earlier rows."""

        if run_id != self.run_id:
            raise IdeaRunnerError("activation does not match the idea runner bound run")
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
            existing = connection.execute(
                "SELECT instance_id, data_class FROM idea_instances WHERE run_id=? AND idea_id=? "
                "AND idea_version=? AND plugin_code_hash=? AND manifest_hash=? "
                "AND parameters_hash=? AND universe_hash=? AND deactivated_at_us IS NULL",
                (
                    run_id,
                    plugin.manifest.idea_id,
                    plugin.manifest.idea_version,
                    plugin.code_hash,
                    plugin.manifest_hash,
                    plugin.parameters_hash,
                    plugin.universe_hash,
                ),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                instance_id = str(existing["instance_id"])
                self._plugins_by_instance[instance_id] = plugin.plugin
                self._discovered_by_instance[instance_id] = plugin
                return ActivationResult(
                    instance_id, ProtectedDataClass(str(existing["data_class"])), False
                )
            watermark = int(
                connection.execute(
                    "SELECT coalesce(max(sequence), 0) FROM ("
                    "SELECT max(source_sequence) AS sequence FROM callback_inbox WHERE run_id=? "
                    "UNION ALL SELECT max(source_sequence) FROM market_events WHERE run_id=? "
                    "UNION ALL SELECT max(last_source_sequence) FROM callback_receipts "
                    "WHERE run_id=? UNION ALL SELECT max(compacted_through_sequence) "
                    "FROM callback_compaction_watermarks WHERE run_id=?)",
                    (run_id, run_id, run_id, run_id),
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
                    "last_source_sequence, state_json, state_hash, state_input_event_ids_json, "
                    "updated_at_us) VALUES (?, NULL, ?, ?, ?, '[]', ?)",
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
        self._discovered_by_instance[instance_id] = plugin
        return ActivationResult(instance_id, data_class, inserted)

    def run_once(self, *, now_us: int) -> tuple[EvaluationResult, ...]:
        """Evaluate each healthy instance independently and isolate every failure."""

        with connect_v2(self.database_path) as connection:
            instance_ids = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT instance.instance_id FROM idea_instances instance "
                    "JOIN runs run USING(run_id) WHERE instance.run_id=? "
                    "AND run.status IN ('created','running') "
                    "AND instance.deactivated_at_us IS NULL "
                    "AND instance.health != 'disabled' ORDER BY instance.instance_id",
                    (self.run_id,),
                )
            )
        return tuple(self._run_instance(instance_id, now_us=now_us) for instance_id in instance_ids)

    def deactivate_unconfigured(self, *, run_id: str, now_us: int) -> tuple[str, ...]:
        """Disable active instances absent from the explicit current configuration."""

        configured = {plugin.identity for plugin in self._discovered}
        with connect_v2(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT instance_id, idea_id, idea_version, plugin_code_hash, parameters_hash, "
                "universe_hash FROM idea_instances WHERE run_id=? AND deactivated_at_us IS NULL",
                (run_id,),
            ).fetchall()
            deactivated = tuple(
                str(row["instance_id"])
                for row in rows
                if tuple(str(row[key]) for key in row.keys()[1:]) not in configured
            )
            for instance_id in deactivated:
                connection.execute(
                    "UPDATE idea_instances SET deactivated_at_us=?, health='disabled', "
                    "error_code=NULL WHERE instance_id=? AND deactivated_at_us IS NULL",
                    (now_us, instance_id),
                )
                connection.execute(
                    "UPDATE incidents SET resolved_at_us=? WHERE plugin_instance_id=? "
                    "AND resolved_at_us IS NULL",
                    (now_us, instance_id),
                )
                connection.execute(
                    "UPDATE market_data_interests SET lifecycle='cancelled', "
                    "reason_code='PLUGIN_DEACTIVATED', updated_at_us=? WHERE instance_id=? "
                    "AND lifecycle IN ('pending','resolved','active')",
                    (now_us, instance_id),
                )
            connection.commit()
        for instance_id in deactivated:
            self._plugins_by_instance.pop(instance_id, None)
            self._discovered_by_instance.pop(instance_id, None)
            worker = self._workers.pop(instance_id, None)
            if worker is not None:
                worker.terminate()
        return deactivated

    def close(self) -> None:
        """Stop all plugin workers without affecting recorder or persisted state."""

        for worker in self._workers.values():
            worker.terminate()
        self._workers.clear()

    def _run_instance(self, instance_id: str, *, now_us: int) -> EvaluationResult:
        with connect_v2(self.database_path) as connection:
            retry = connection.execute(
                "SELECT consecutive_failures, updated_at_us FROM idea_checkpoints "
                "WHERE instance_id=?",
                (instance_id,),
            ).fetchone()
        if retry is not None and int(retry["consecutive_failures"]) > 0:
            failures = int(retry["consecutive_failures"])
            retry_delay = min(RETRY_MAX_US, RETRY_BASE_US * (2 ** min(failures - 1, 6)))
            if now_us < int(retry["updated_at_us"]) + retry_delay:
                return EvaluationResult(instance_id, False, 0, "RETRY_BACKOFF")
        plugin = self._plugins_by_instance.get(instance_id)
        if plugin is None or not hasattr(plugin, "evaluate"):
            worker = self._workers.pop(instance_id, None)
            if worker is not None:
                worker.terminate()
            return self._degrade(instance_id, now_us, "PLUGIN_UNAVAILABLE")
        try:
            context = self._load_batch(instance_id)
            if context is None:
                return EvaluationResult(instance_id, False, 0)
            activation, batch, state, event_ids, starting_checkpoint = context
            worker = self._workers.get(instance_id)
            if worker is None:
                worker = _PluginWorker(cast(IdeaPlugin, plugin))
                self._workers[instance_id] = worker
            evaluation = worker.evaluate(batch, state)
            self._validate_evaluation(activation, cast(IdeaPlugin, plugin), batch, evaluation)
            self._commit_evaluation(
                activation, batch, evaluation, event_ids, starting_checkpoint, now_us
            )
            return EvaluationResult(instance_id, True, len(evaluation.outputs))
        except Exception as error:
            worker = self._workers.pop(instance_id, None)
            if worker is not None:
                worker.terminate()
            code = f"{type(error).__name__}:{error}".upper()[:64]
            return self._degrade(instance_id, now_us, code)

    def _load_batch(
        self, instance_id: str
    ) -> tuple[IdeaActivation, IdeaBatch, JsonValue, tuple[str, ...], str | None] | None:
        with connect_v2(self.database_path) as connection:
            row = connection.execute(
                "SELECT instance.*, checkpoint.last_market_event_id, "
                "checkpoint.last_source_sequence, checkpoint.state_json, "
                "checkpoint.state_hash, checkpoint.state_input_event_ids_json "
                "FROM idea_instances instance JOIN idea_checkpoints checkpoint USING(instance_id) "
                "WHERE instance.instance_id=?",
                (instance_id,),
            ).fetchone()
            if row is None:
                raise IdeaRunnerError("instance checkpoint is missing")
            discovered = self._discovered_by_instance.get(instance_id)
            if discovered is None:
                raise IdeaRunnerError("configured plugin binding is missing")
            self._verify_persisted_instance(connection, row, discovered)
            requirements = tuple(
                MarketDataRequirement.model_validate(item)
                for item in json.loads(str(row["requirements_json"]))
            )
            receipt_rows = connection.execute(
                "SELECT receipt.*, interest.interest_key, interest.feed_kind, "
                "interest.cadence, interest.required, interest.lifecycle, "
                "interest.expires_at_us "
                "FROM instrument_discovery_receipts receipt "
                "JOIN market_data_interests interest USING(interest_id) "
                "WHERE receipt.instance_id=? "
                "ORDER BY CASE WHEN interest.lifecycle IN ('resolved','active') "
                "THEN 0 ELSE 1 END, receipt.completed_at_us DESC, receipt.receipt_id DESC "
                "LIMIT 64",
                (instance_id,),
            ).fetchall()
            batch_requirements = tuple(
                cast(
                    JsonValue,
                    {
                        "feed_kind": item.feed_kind,
                        "event_kind": item.event_kind,
                        "instrument_id": item.instrument_id,
                        "available_at_us": 0,
                    },
                )
                for item in requirements
            ) + tuple(
                cast(
                    JsonValue,
                    {
                        "feed_kind": str(item["feed_kind"]),
                        "event_kind": None,
                        "instrument_id": str(item["instrument_id"]),
                        "available_at_us": int(item["completed_at_us"]),
                        "available_until_us": int(item["expires_at_us"]),
                    },
                )
                for item in receipt_rows
                if str(item["status"]) == "resolved" and item["instrument_id"] is not None
            )
            batch_requirements_json = _json(cast(JsonValue, batch_requirements))
            # Imported at the use site to avoid the ingestion package's public recorder
            # exports creating a runner/recorder import cycle.
            from stocker_runtime.ingestion.bar_projection import (
                project_required_five_minute_bars,
            )
            from stocker_runtime.ingestion.session_projection import (
                project_required_session_receipts,
            )

            project_required_five_minute_bars(
                connection,
                run_id=str(row["run_id"]),
                requirements=requirements,
                after_source_sequence=int(row["activated_after_source_sequence"]),
            )
            project_required_session_receipts(
                connection,
                run_id=str(row["run_id"]),
                requirements=requirements,
                after_source_sequence=int(row["activated_after_source_sequence"]),
            )
            start_sequence = (
                int(row["last_source_sequence"])
                if row["last_source_sequence"] is not None
                else int(row["activated_after_source_sequence"])
            )
            candidates = connection.execute(
                "SELECT event.* FROM market_events event WHERE event.run_id=? "
                "AND EXISTS (SELECT 1 FROM json_each(?) requirement "
                "WHERE event.instrument_id=json_extract(requirement.value, '$.instrument_id') "
                "AND event.feed_kind=json_extract(requirement.value, '$.feed_kind') "
                "AND (json_extract(requirement.value, '$.event_kind') IS NULL "
                "OR event.event_kind=json_extract(requirement.value, '$.event_kind')) "
                "AND json_extract(requirement.value, '$.available_at_us') "
                "<=max(event.event_at_us, event.received_at_us) "
                "AND (json_extract(requirement.value, '$.available_until_us') IS NULL "
                "OR max(event.event_at_us, event.received_at_us)<"
                "json_extract(requirement.value, '$.available_until_us'))) "
                "AND (coalesce(event.source_sequence, event.derived_after_source_sequence)>? "
                "OR (? IS NOT NULL AND "
                "coalesce(event.source_sequence, event.derived_after_source_sequence)=? "
                "AND event.event_id>?)) "
                "AND (event.event_kind!='bar_5m' OR "
                "json_extract(event.payload_json, '$.first_source_sequence')>?) "
                "ORDER BY coalesce(event.source_sequence, event.derived_after_source_sequence), "
                "event.event_id LIMIT 256",
                (
                    row["run_id"],
                    batch_requirements_json,
                    start_sequence,
                    row["last_market_event_id"],
                    start_sequence,
                    row["last_market_event_id"],
                    int(row["activated_after_source_sequence"]),
                ),
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
            causal_through_at_us = max(
                max(item.event_at_us, item.received_at_us) for item in events
            )
            causal_receipt_rows = tuple(
                item
                for item in receipt_rows
                if int(item["completed_at_us"]) <= causal_through_at_us
            )
            discovery_receipts = tuple(
                DiscoveryReceipt.model_validate(
                    {
                        "receipt_id": str(item["receipt_id"]),
                        "interest_id": str(item["interest_id"]),
                        "interest_key": str(item["interest_key"]),
                        "instance_id": str(item["instance_id"]),
                        "status": str(item["status"]),
                        "reason_code": (
                            None if item["reason_code"] is None else str(item["reason_code"])
                        ),
                        "instrument_id": (
                            None if item["instrument_id"] is None else str(item["instrument_id"])
                        ),
                        "expiry": None if item["expiry"] is None else str(item["expiry"]),
                        "strike": None if item["strike"] is None else float(item["strike"]),
                        "option_right": (
                            None if item["option_right"] is None else str(item["option_right"])
                        ),
                        "multiplier": (
                            None if item["multiplier"] is None else str(item["multiplier"])
                        ),
                        "candidates_inspected": int(item["candidates_inspected"]),
                        "completed_at_us": int(item["completed_at_us"]),
                    }
                )
                for item in causal_receipt_rows
            )
            causal_dynamic_requirements = tuple(
                MarketDataRequirement(
                    feed_kind=str(item["feed_kind"]),
                    event_kind=None,
                    instrument_id=str(item["instrument_id"]),
                    cadence=str(item["cadence"]),
                    gaps_block=(
                        bool(item["required"]) and str(item["lifecycle"]) in {"resolved", "active"}
                    ),
                    staleness_block=(
                        bool(item["required"]) and str(item["lifecycle"]) in {"resolved", "active"}
                    ),
                )
                for item in causal_receipt_rows
                if str(item["status"]) == "resolved" and item["instrument_id"] is not None
            )
            causal_requirements = _merge_batch_requirements(
                (*requirements, *causal_dynamic_requirements)
            )
            for requirement in causal_requirements:
                gap = connection.execute(
                    "SELECT 1 FROM gaps gap JOIN subscriptions subscription "
                    "ON subscription.subscription_id=gap.subscription_id "
                    "WHERE gap.run_id=? AND gap.resolved_at_us IS NULL "
                    "AND subscription.instrument_id=? AND subscription.feed_kind=? "
                    "AND (gap.ended_at_us IS NULL OR gap.ended_at_us>=?) "
                    "AND ((gap.reason='STREAM_STALE' AND ?=1) OR "
                    "(gap.reason!='STREAM_STALE' AND ?=1)) LIMIT 1",
                    (
                        row["run_id"],
                        requirement.instrument_id,
                        requirement.feed_kind,
                        int(row["activated_at_us"]),
                        int(requirement.staleness_block),
                        int(requirement.gaps_block),
                    ),
                ).fetchone()
                if gap is not None:
                    return None
            prior_state_input_event_ids = tuple(
                str(value) for value in json.loads(str(row["state_input_event_ids_json"]))
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
                causal_through_at_us=causal_through_at_us,
                prior_state_input_event_ids=prior_state_input_event_ids,
                discovery_receipts=discovery_receipts,
            )
            return (
                activation,
                batch,
                cast(JsonValue, json.loads(str(row["state_json"]))),
                tuple(item.event_id for item in events),
                None if row["last_market_event_id"] is None else str(row["last_market_event_id"]),
            )

    def _validate_evaluation(
        self,
        activation: IdeaActivation,
        plugin: IdeaPlugin,
        batch: IdeaBatch,
        evaluation: IdeaEvaluation,
    ) -> None:
        if len(evaluation.state_json()) > plugin.manifest.maximum_state_bytes:
            raise IdeaRunnerError("plugin state exceeds manifest bound")
        if len(evaluation.outputs) > min(plugin.manifest.maximum_outputs_per_batch, 256):
            raise IdeaRunnerError("plugin outputs exceed manifest bound")
        if len(evaluation.interests) > plugin.manifest.maximum_interests_per_batch:
            raise IdeaRunnerError("plugin interests exceed manifest bound")
        available = (
            *batch.prior_state_input_event_ids,
            *(event.event_id for event in batch.events),
        )
        available = tuple(dict.fromkeys(available))
        available_set = set(available)
        if len(evaluation.output_input_event_ids) != len(evaluation.outputs):
            raise IdeaRunnerError("plugin must declare one causal lineage per output")
        declared = (
            *evaluation.output_input_event_ids,
            evaluation.retained_input_event_ids,
            *((interest.input_event_id,) for interest in evaluation.interests),
        )
        for lineage in declared:
            if not set(lineage).issubset(available_set) or tuple(
                event_id for event_id in available if event_id in set(lineage)
            ) != tuple(lineage):
                raise IdeaRunnerError("plugin declared invalid or unordered causal lineage")
        event_times: dict[str, tuple[int, int]] = {
            event.event_id: (event.event_at_us, event.received_at_us) for event in batch.events
        }
        allowed_instrument_ids = {
            *activation.universe,
            *(
                receipt.instrument_id
                for receipt in batch.discovery_receipts
                if receipt.status == "resolved" and receipt.instrument_id is not None
            ),
        }
        missing_prior = tuple(
            event_id
            for event_id in batch.prior_state_input_event_ids
            if event_id not in event_times
        )
        if missing_prior:
            placeholders = ",".join("?" for _ in missing_prior)
            with connect_v2(self.database_path) as connection:
                rows = connection.execute(
                    f"SELECT event_id, event_at_us, received_at_us FROM market_events "  # noqa: S608
                    f"WHERE run_id=? AND event_id IN ({placeholders})",
                    (activation.run_id, *missing_prior),
                ).fetchall()
            event_times.update(
                {
                    str(row["event_id"]): (
                        int(row["event_at_us"]),
                        int(row["received_at_us"]),
                    )
                    for row in rows
                }
            )
            if any(event_id not in event_times for event_id in missing_prior):
                raise IdeaRunnerError("retained causal evidence is missing")
        allowed = {item.value for item in plugin.manifest.output_kinds}
        for interest in evaluation.interests:
            if interest.underlying_instrument_id not in activation.universe:
                raise IdeaRunnerError("market-data interest underlying is outside the universe")
            if (
                event_times[interest.input_event_id][0] > interest.as_of_at_us
                or interest.as_of_at_us > batch.causal_through_at_us
            ):
                raise IdeaRunnerError("market-data interest has noncausal input lineage")
        for output, lineage in zip(
            evaluation.outputs, evaluation.output_input_event_ids, strict=True
        ):
            if output.kind not in allowed:
                raise IdeaRunnerError("plugin emitted a forbidden output kind")
            if output.subject_instrument_id not in allowed_instrument_ids:
                raise IdeaRunnerError("plugin output subject is outside the activation universe")
            if not (
                min(event_times[event_id][0] for event_id in lineage)
                <= output.as_of_at_us
                <= batch.causal_through_at_us
            ):
                raise IdeaRunnerError("plugin output as-of time is outside its causal input")
            if any(event_times[event_id][0] > output.as_of_at_us for event_id in lineage):
                raise IdeaRunnerError("plugin output declares evidence later than its as-of time")
            if output.kind in {"proposed_position", "proposed_trade"}:
                if getattr(output, "status", None) != "unapproved":
                    raise IdeaRunnerError("proposal is not explicitly unapproved")
                ensure_authority_free_json(output.payload)
            if isinstance(output, ProposedTrade):
                for leg in output.legs:
                    if leg.instrument_id not in allowed_instrument_ids:
                        raise IdeaRunnerError("proposed trade leg is outside activation universe")

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
        retained_json = _json(cast(JsonValue, evaluation.retained_input_event_ids))
        with connect_v2(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT last_market_event_id FROM idea_checkpoints WHERE instance_id=?",
                (activation.instance_id,),
            ).fetchone()
            current_id = None if current is None or current[0] is None else str(current[0])
            if current_id != starting_checkpoint:
                raise IdeaRunnerError("checkpoint changed during plugin evaluation")
            for interest in evaluation.interests:
                self._insert_interest(connection, activation, interest, now_us)
            for ordinal, output in enumerate(evaluation.outputs):
                output_event_ids = evaluation.output_input_event_ids[ordinal]
                self._insert_output(
                    connection,
                    activation,
                    batch,
                    output,
                    ordinal,
                    output_event_ids,
                    _hash(cast(JsonValue, output_event_ids)),
                    now_us,
                )
            watermark_row = connection.execute(
                "SELECT coalesce(source_sequence, derived_after_source_sequence) "
                "FROM market_events WHERE event_id=? AND run_id=?",
                (batch.input_watermark, activation.run_id),
            ).fetchone()
            if watermark_row is None:
                raise IdeaRunnerError("input watermark disappeared before checkpoint commit")
            changed = connection.execute(
                "UPDATE idea_checkpoints SET last_market_event_id=?, last_source_sequence=?, "
                "state_json=?, state_hash=?, state_input_event_ids_json=?, "
                "last_success_at_us=?, updated_at_us=?, consecutive_failures=0 WHERE instance_id=? "
                "AND last_market_event_id IS ?",
                (
                    batch.input_watermark,
                    int(watermark_row[0]),
                    state_json,
                    state_hash,
                    retained_json,
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
            connection.execute(
                "UPDATE incidents SET resolved_at_us=? WHERE plugin_instance_id=? "
                "AND resolved_at_us IS NULL",
                (now_us, activation.instance_id),
            )
            connection.commit()

    @staticmethod
    def _insert_interest(
        connection: sqlite3.Connection,
        activation: IdeaActivation,
        interest: MarketDataInterest,
        now_us: int,
    ) -> None:
        content_hash = hashlib.sha256(interest.to_canonical_json()).hexdigest()
        interest_id = _hash(
            cast(
                JsonValue,
                {
                    "instance_id": activation.instance_id,
                    "interest": interest.model_dump(mode="json"),
                },
            )
        )
        existing = connection.execute(
            "SELECT content_hash FROM market_data_interests WHERE interest_id=?",
            (interest_id,),
        ).fetchone()
        if existing is not None:
            if str(existing["content_hash"]) != content_hash:
                raise IdeaRunnerError("deterministic market-data interest collision")
            return
        instance_count = int(
            connection.execute(
                "SELECT count(*) FROM market_data_interests WHERE instance_id=? "
                "AND lifecycle IN ('pending','resolved','active')",
                (activation.instance_id,),
            ).fetchone()[0]
        )
        run_count = int(
            connection.execute(
                "SELECT count(*) FROM market_data_interests WHERE run_id=? "
                "AND lifecycle IN ('pending','resolved','active')",
                (activation.run_id,),
            ).fetchone()[0]
        )
        if (
            instance_count >= MAX_NONTERMINAL_INTERESTS_PER_INSTANCE
            or run_count >= MAX_NONTERMINAL_INTERESTS_PER_RUN
        ):
            raise IdeaRunnerError("nonterminal interest cap is reached")
        cursor = connection.execute(
            "INSERT INTO market_data_interests(interest_id, run_id, instance_id, interest_key, "
            "underlying_instrument_id, asset_kind, minimum_days_to_expiry, "
            "maximum_days_to_expiry, option_right, strike_offset, reference_price, feed_kind, "
            "cadence, as_of_at_us, expires_at_us, required, priority, maximum_contracts, "
            "input_event_id, content_hash, lifecycle, next_attempt_at_us, "
            "created_at_us, updated_at_us) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "'pending', ?, ?, ?) ON CONFLICT(interest_id) DO NOTHING",
            (
                interest_id,
                activation.run_id,
                activation.instance_id,
                interest.interest_key,
                interest.underlying_instrument_id,
                interest.asset_kind,
                interest.minimum_days_to_expiry,
                interest.maximum_days_to_expiry,
                interest.option_right,
                interest.strike_offset,
                interest.reference_price,
                interest.feed_kind,
                interest.cadence,
                interest.as_of_at_us,
                interest.expires_at_us,
                int(interest.required),
                interest.priority,
                interest.maximum_contracts,
                interest.input_event_id,
                content_hash,
                min(now_us, interest.expires_at_us),
                now_us,
                now_us,
            ),
        )
        if cursor.rowcount == 0:
            existing = connection.execute(
                "SELECT content_hash FROM market_data_interests WHERE interest_id=?",
                (interest_id,),
            ).fetchone()
            if existing is None or str(existing[0]) != content_hash:
                raise IdeaRunnerError("deterministic market-data interest collision")

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
        typed_legs = output.legs if isinstance(output, ProposedTrade) else ()
        output_id = deterministic_idea_output_id(
            instance_id=activation.instance_id,
            input_event_ids=event_ids,
            output_kind=output.kind,
            output_ordinal=ordinal,
            as_of_at_us=output.as_of_at_us,
            payload=output.payload,
            legs=typed_legs,
        )
        authority = (
            "unapproved" if output.kind in {"proposed_position", "proposed_trade"} else "recorded"
        )
        content_hash = deterministic_idea_output_content_hash(
            run_id=activation.run_id,
            instance_id=activation.instance_id,
            output_kind=output.kind,
            subject_instrument_id=output.subject_instrument_id,
            emitted_at_us=now_us,
            as_of_at_us=output.as_of_at_us,
            valid_until_at_us=None,
            direction=None,
            strength=None,
            confidence=None,
            horizon_us=None,
            input_event_ids=event_ids,
            output_ordinal=ordinal,
            payload=output.payload,
            data_class=activation.protected_data_class.value,
            authority_status=authority,
            legs=typed_legs,
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
                    event_ids[-1],
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
                "SELECT output.content_hash, seal.output_id AS sealed_output_id "
                "FROM idea_outputs output "
                "LEFT JOIN idea_output_seals seal ON seal.output_id=output.output_id "
                "WHERE output.instance_id=? AND output.input_watermark=? "
                "AND output.output_kind=? AND output.output_ordinal=? AND output.as_of_at_us=?",
                (
                    activation.instance_id,
                    event_ids[-1],
                    output.kind,
                    ordinal,
                    output.as_of_at_us,
                ),
            ).fetchone()
            if existing is None or str(existing[0]) != content_hash:
                raise IdeaRunnerError("deterministic output identity collision") from error
            if existing["sealed_output_id"] is None:
                raise IdeaRunnerError("deterministic output is not durably sealed") from error
            stored_inputs = tuple(
                str(row["event_id"])
                for row in connection.execute(
                    "SELECT event_id FROM idea_output_inputs WHERE output_id=? "
                    "ORDER BY input_ordinal",
                    (output_id,),
                )
            )
            stored_legs = tuple(
                tuple(row)
                for row in connection.execute(
                    "SELECT instrument_id, action, target, quantity_value, notional_value, "
                    "currency, price_hint FROM idea_output_legs WHERE output_id=? "
                    "ORDER BY leg_number",
                    (output_id,),
                )
            )
            expected_legs = tuple(
                (
                    leg.instrument_id,
                    leg.action,
                    leg.target,
                    leg.quantity_value,
                    leg.notional_value,
                    leg.currency,
                    leg.price_hint,
                )
                for leg in typed_legs
            )
            if stored_inputs != event_ids or stored_legs != expected_legs:
                raise IdeaRunnerError("deterministic output sealed provenance mismatch") from error
            return
        for input_ordinal, event_id in enumerate(event_ids):
            connection.execute(
                "INSERT INTO idea_output_inputs(output_id, event_id, input_ordinal) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT DO NOTHING",
                (output_id, event_id, input_ordinal),
            )
        if isinstance(output, ProposedTrade):
            for leg_number, leg in enumerate(output.legs):
                connection.execute(
                    "INSERT INTO idea_output_legs(output_id, leg_number, instrument_id, action, "
                    "target, quantity_value, notional_value, currency, price_hint) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                    (
                        output_id,
                        leg_number,
                        leg.instrument_id,
                        leg.action,
                        leg.target,
                        leg.quantity_value,
                        leg.notional_value,
                        leg.currency,
                        leg.price_hint,
                    ),
                )
        connection.execute(
            "INSERT INTO idea_output_seals(output_id) VALUES (?) ON CONFLICT DO NOTHING",
            (output_id,),
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
            incident_id = _hash(cast(JsonValue, {"instance_id": instance_id, "open": True}))
            run = connection.execute(
                "SELECT run_id FROM idea_instances WHERE instance_id=?", (instance_id,)
            ).fetchone()
            if run is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO incidents(incident_id, run_id, scope, severity, code, "
                    "plugin_instance_id, opened_at_us, details_json) VALUES (?, ?, 'idea_plugin', "
                    "'degraded', ?, ?, ?, '{}') ON CONFLICT(incident_id) DO UPDATE SET "
                    "code=excluded.code, opened_at_us=excluded.opened_at_us, "
                    "resolved_at_us=NULL, details_json=excluded.details_json",
                    (incident_id, run[0], code, instance_id, now_us),
                )
            connection.commit()
        return EvaluationResult(instance_id, False, 0, code)
