"""Small bounded query layer for the generic Stocker V2 read model."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from stocker_runtime.market_session import market_data_expected_since_us
from stocker_runtime.storage import RetentionPolicy, read_backup_manifests
from stocker_runtime.web.config import WebConfig
from stocker_runtime.web.readiness import calculate_readiness, select_operational_run

API_ROUTES = (
    "/api/v2/meta",
    "/api/v2/live",
    "/api/v2/ideas",
    "/api/v2/ideas/{instance_id}",
    "/api/v2/results",
    "/api/v2/results/{position_id}",
    "/api/v2/diagnostics",
    "/api/v2/ready",
)
BANNER = "PROSPECTIVE / SHADOW ONLY — NO APPROVAL OR EXECUTION"
IDEA_WINDOW_US = 604_800_000_000
RESULT_WINDOW_US = 2_592_000_000_000
SQLITE_INTEGER_MAX = 9_223_372_036_854_775_807
OUTPUT_COLUMNS = (
    "output.output_id, output.output_kind, output.subject_instrument_id, "
    "output.emitted_at_us, output.as_of_at_us, output.valid_until_at_us, "
    "output.direction, output.strength, output.confidence, output.horizon_us, "
    "output.first_input_event_id, output.last_input_event_id, output.input_watermark, "
    "output.input_events_hash, output.payload_json, output.content_hash, "
    "output.data_class, output.authority_status"
)


class QueryTimeoutError(RuntimeError):
    """SQLite interrupted a read that exceeded the fixed query budget."""


class CursorError(ValueError):
    """A pagination cursor is malformed or belongs to another query."""


class WindowError(ValueError):
    """A requested API history window exceeds its fixed bound."""


@dataclass(frozen=True)
class CursorPosition:
    timestamp: int
    identity: str
    window: tuple[int, int] | None = None


def _filter_hash(filters: dict[str, Any]) -> str:
    encoded = json.dumps(
        filters,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _encode_cursor(
    *,
    route: str,
    filters: dict[str, Any],
    timestamp: int,
    identity: str,
    window: tuple[int, int] | None = None,
) -> str:
    payload = {
        "f": _filter_hash(filters),
        "i": identity,
        "r": route,
        "t": timestamp,
        "v": 1,
        "w": None if window is None else list(window),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(encoded).decode().rstrip("=")


def _decode_cursor(
    cursor: str,
    *,
    route: str,
    filters: dict[str, Any],
) -> CursorPosition:
    try:
        decoded = base64.b64decode(
            cursor + "=" * (-len(cursor) % 4),
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(decoded)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CursorError("invalid_cursor") from error
    if not isinstance(payload, dict) or set(payload) != {"f", "i", "r", "t", "v", "w"}:
        raise CursorError("invalid_cursor")
    if (
        payload["v"] != 1
        or payload["r"] != route
        or payload["f"] != _filter_hash(filters)
        or isinstance(payload["t"], bool)
        or not isinstance(payload["t"], int)
        or payload["t"] < 0
        or payload["t"] > SQLITE_INTEGER_MAX
        or not isinstance(payload["i"], str)
        or not payload["i"]
        or len(payload["i"]) > 512
    ):
        raise CursorError("invalid_cursor")
    raw_window = payload["w"]
    if raw_window is None:
        window = None
    elif (
        isinstance(raw_window, list)
        and len(raw_window) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value <= SQLITE_INTEGER_MAX
            for value in raw_window
        )
        and 0 <= raw_window[0] <= raw_window[1]
    ):
        window = (raw_window[0], raw_window[1])
    else:
        raise CursorError("invalid_cursor")
    return CursorPosition(payload["t"], payload["i"], window)


def _validate_cursor_window(
    position: CursorPosition | None,
    *,
    maximum_us: int | None,
) -> None:
    if position is None:
        return
    if maximum_us is None:
        if position.window is not None:
            raise CursorError("invalid_cursor")
        return
    if (
        position.window is None
        or position.window[1] - position.window[0] > maximum_us
        or not position.window[0] <= position.timestamp <= position.window[1]
    ):
        raise CursorError("invalid_cursor")


class ReadModel:
    """Open one query-only SQLite connection per bounded API projection."""

    def __init__(self, config: WebConfig) -> None:
        self.config = config

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        deadline = time.perf_counter() + self.config.query_budget_ms / 1_000
        uri = self.config.database.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(
            uri,
            uri=True,
            isolation_level=None,
            timeout=self.config.query_budget_ms / 1_000,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.config.query_budget_ms}")
        connection.set_progress_handler(lambda: int(time.perf_counter() >= deadline), 1_000)
        try:
            yield connection
        except sqlite3.OperationalError as error:
            message = str(error).lower()
            if any(marker in message for marker in ("interrupted", "locked", "busy")):
                raise QueryTimeoutError("query_timeout") from error
            raise
        finally:
            connection.set_progress_handler(None, 0)
            connection.close()

    @staticmethod
    def _dictionary(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return None if row is None else dict(row)

    def _latest_run(self, connection: sqlite3.Connection) -> dict[str, Any] | None:
        selected, _reason = select_operational_run(
            connection,
            pinned_run_id=self.config.run_id,
            now_us=time.time_ns() // 1_000,
        )
        if selected is None:
            return None
        return {key: selected[key] for key in ("run_id", "mode", "status", "started_at_us")}

    def ready(self, *, now_us: int | None = None) -> dict[str, Any]:
        """Return current recorder readiness without touching writer state."""

        evaluated_at_us = time.time_ns() // 1_000 if now_us is None else now_us
        expected_since_us = market_data_expected_since_us(evaluated_at_us)
        with self._connection() as connection:
            return calculate_readiness(
                connection,
                pinned_run_id=self.config.run_id,
                now_us=evaluated_at_us,
                expected_since_us=expected_since_us,
            )

    def _backup_projection(self, *, limit: int) -> dict[str, Any]:
        root = self.config.backup_directory
        if root is None:
            return {
                "available": False,
                "items": [],
                "status": {
                    "checked_at_us": None,
                    "code": None,
                    "format_version": 1,
                    "latest_manifest_filename": None,
                    "state": "unavailable",
                },
                "truncated": False,
            }
        return cast(dict[str, Any], read_backup_manifests(root, limit=limit).to_dict())

    def _backup_summary(self) -> dict[str, Any]:
        projection = self._backup_projection(limit=200)
        items = projection["items"]
        assert isinstance(items, list)
        return {
            "available": projection["available"],
            "entries": len(items),
            "latest": None if not items else items[0]["archive_filename"],
        }

    def live(
        self,
        *,
        feed_kind: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Read current recorder/latest projections without scanning event history."""

        normalized_feed_kind = None if feed_kind is None else feed_kind.strip().lower()
        filters = {"feed_kind": normalized_feed_kind}
        route = "/api/v2/live"
        position = None if cursor is None else _decode_cursor(cursor, route=route, filters=filters)
        _validate_cursor_window(position, maximum_us=None)
        with self._connection() as connection:
            run = self._latest_run(connection)
            if run is None:
                return {
                    "run": None,
                    "recorder": None,
                    "ibkr": None,
                    "feeds": {"active": 0, "by_kind": {}},
                    "instruments": [],
                    "callback_inbox": {"nonterminal": 0, "bytes": 0},
                    "storage": self._storage(None),
                    "gaps": {"unresolved": 0, "data_loss_possible": 0},
                    "backup": self._backup_summary(),
                    "limit": limit,
                    "next_cursor": None,
                }
            run_id = str(run["run_id"])
            state = connection.execute(
                "SELECT lifecycle, reason, process_heartbeat_at_us, callback_heartbeat_at_us, "
                "admission_heartbeat_at_us, projection_heartbeat_at_us, connection_state, "
                "connection_generation, inbox_nonterminal_count, inbox_bytes, database_bytes, "
                "wal_bytes FROM runtime_state WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            feed_rows = connection.execute(
                "SELECT feed_kind, COUNT(*) AS item_count FROM subscriptions "
                "WHERE run_id = ? AND lifecycle = 'active' GROUP BY feed_kind ORDER BY feed_kind",
                (run_id,),
            ).fetchall()
            latest_clauses = ["latest.run_id = ?"]
            latest_parameters: list[object] = [run_id]
            if normalized_feed_kind:
                latest_clauses.append("latest.feed_kind = ?")
                latest_parameters.append(normalized_feed_kind)
            if position is not None:
                latest_clauses.append(
                    "(latest.event_at_us < ? OR (latest.event_at_us = ? AND latest.event_id > ?))"
                )
                latest_parameters.extend(
                    (position.timestamp, position.timestamp, position.identity)
                )
            latest_rows = connection.execute(
                "SELECT latest.instrument_id, instrument.kind, instrument.symbol, "
                "instrument.exchange, instrument.currency, latest.feed_kind, latest.event_id, "
                "latest.event_at_us, latest.received_at_us, latest.quality_bits, "
                "latest.bid_value AS bid, latest.ask_value AS ask, latest.last_value AS last, "
                "latest.close_value AS close, latest.size_value AS size "
                "FROM market_latest AS latest JOIN instruments AS instrument "
                "ON instrument.instrument_id = latest.instrument_id "
                "WHERE "
                + " AND ".join(latest_clauses)
                + " ORDER BY latest.event_at_us DESC, latest.event_id LIMIT ?",
                (*latest_parameters, min(limit, 250) + 1),
            ).fetchall()
            gap = connection.execute(
                "SELECT COUNT(*) AS unresolved, "
                "COALESCE(SUM(data_loss_possible), 0) AS data_loss_possible FROM gaps "
                "WHERE run_id = ? AND resolved_at_us IS NULL",
                (run_id,),
            ).fetchone()

        page = latest_rows[:limit]
        next_cursor = None
        if len(latest_rows) > limit and page:
            tail = page[-1]
            next_cursor = _encode_cursor(
                route=route,
                filters=filters,
                timestamp=int(tail["event_at_us"]),
                identity=str(tail["event_id"]),
            )
        state_value = self._dictionary(state)
        feed_counts = {str(row["feed_kind"]): int(row["item_count"]) for row in feed_rows}
        freshness = None
        if state_value is not None:
            freshness = max(
                (
                    value
                    for value in (
                        state_value["callback_heartbeat_at_us"],
                        state_value["admission_heartbeat_at_us"],
                        state_value["projection_heartbeat_at_us"],
                    )
                    if value is not None
                ),
                default=None,
            )
        return {
            "run": run,
            "recorder": None
            if state_value is None
            else {
                "lifecycle": state_value["lifecycle"],
                "reason": state_value["reason"],
                "process_heartbeat_at_us": state_value["process_heartbeat_at_us"],
            },
            "ibkr": None
            if state_value is None
            else {
                "connection_state": state_value["connection_state"],
                "connection_generation": state_value["connection_generation"],
                "freshness_at_us": freshness,
            },
            "feeds": {"active": sum(feed_counts.values()), "by_kind": feed_counts},
            "instruments": [dict(row) for row in page],
            "callback_inbox": {
                "nonterminal": 0 if state_value is None else state_value["inbox_nonterminal_count"],
                "bytes": 0 if state_value is None else state_value["inbox_bytes"],
            },
            "storage": self._storage(state_value),
            "gaps": {
                "unresolved": 0 if gap is None else int(gap["unresolved"]),
                "data_loss_possible": 0 if gap is None else int(gap["data_loss_possible"]),
            },
            "backup": self._backup_summary(),
            "limit": limit,
            "next_cursor": next_cursor,
        }

    def meta(self) -> dict[str, Any]:
        """Return bounded build, schema, mode, and authority metadata."""

        with self._connection() as connection:
            run = self._latest_run(connection)
            schema_version = int(
                connection.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
                ).fetchone()[0]
            )
        return {
            "application": {"name": "Stocker V2", "version": self.config.app_version},
            "build": {"git_commit": self.config.git_commit},
            "schema_version": schema_version,
            "run": run,
            "modes": ["prospective_record", "shadow"],
            "banner": BANNER,
            "api_routes": list(API_ROUTES),
            "authority": {
                "approval": False,
                "execution": False,
                "broker_orders": False,
                "broker_fills": False,
                "broker_positions": False,
            },
        }

    def diagnostics(self, *, limit: int) -> dict[str, Any]:
        """Read bounded system detail for the non-primary diagnostics drawer."""

        with self._connection() as connection:
            query_only = bool(connection.execute("PRAGMA query_only").fetchone()[0])
            run = self._latest_run(connection)
            run_id = None if run is None else str(run["run_id"])
            if run_id is None:
                run_hashes = None
                state = None
                incidents: list[sqlite3.Row] = []
                gaps: list[sqlite3.Row] = []
                subscriptions: list[sqlite3.Row] = []
            else:
                run_hashes = connection.execute(
                    "SELECT config_hash, git_commit FROM runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                state = connection.execute(
                    "SELECT run_id, recorder_generation, lifecycle, reason, "
                    "process_heartbeat_at_us, callback_heartbeat_at_us, "
                    "admission_heartbeat_at_us, projection_heartbeat_at_us, "
                    "connection_state, connection_generation, inbox_nonterminal_count, "
                    "inbox_bytes, database_bytes, wal_bytes FROM runtime_state WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                incidents = connection.execute(
                    "SELECT incident_id, scope, severity, code, plugin_instance_id, "
                    "subscription_id, recorder_generation, opened_at_us, resolved_at_us "
                    "FROM incidents WHERE run_id = ? "
                    "ORDER BY opened_at_us DESC, incident_id DESC LIMIT ?",
                    (run_id, limit),
                ).fetchall()
                gaps = connection.execute(
                    "SELECT gap_id, subscription_id, started_at_us, ended_at_us, reason, "
                    "data_loss_possible, continuity_required, resolved_at_us FROM gaps "
                    "WHERE run_id = ? ORDER BY started_at_us DESC, gap_id DESC LIMIT ?",
                    (run_id, limit),
                ).fetchall()
                subscriptions = connection.execute(
                    "SELECT subscription_id, instrument_id, feed_kind, request_id, lifecycle, "
                    "continuity_required, optional, opened_at_us, closed_at_us, latest_event_id "
                    "FROM subscriptions WHERE run_id = ? "
                    "ORDER BY opened_at_us DESC, subscription_id DESC LIMIT ?",
                    (run_id, limit),
                ).fetchall()
            plugin_hashes = connection.execute(
                "SELECT idea_id, idea_version, manifest_hash, code_hash, discovered_at_us "
                "FROM idea_plugins ORDER BY discovered_at_us DESC, idea_id DESC LIMIT 100"
            ).fetchall()
        state_value = self._dictionary(state)
        storage = self._storage(state_value)
        measured_database_bytes = max(
            int(storage["reported_database_bytes"]),
            0 if storage["database_bytes"] is None else int(storage["database_bytes"]),
        )
        measured_wal_bytes = max(int(storage["reported_wal_bytes"]), int(storage["wal_bytes"]))
        policy = RetentionPolicy()
        if measured_database_bytes >= policy.database_cap_bytes:
            retention_status = "fatal"
        elif measured_database_bytes >= int(policy.database_cap_bytes * 0.95):
            retention_status = "critical"
        elif measured_database_bytes >= int(policy.database_cap_bytes * 0.85):
            retention_status = "maintenance_required"
        else:
            retention_status = "healthy"
        return {
            "run": run,
            "runtime": state_value,
            "incidents": [dict(row) for row in incidents],
            "gaps": [dict(row) for row in gaps],
            "subscriptions": [dict(row) for row in subscriptions],
            "backups": self._backup_projection(limit=limit),
            "database": {**storage, "query_only": query_only},
            "hashes": {
                "build": self.config.git_commit,
                "configuration": (
                    self.config.config_hash
                    if run_hashes is None
                    else str(run_hashes["config_hash"])
                ),
                "run_build": None if run_hashes is None else run_hashes["git_commit"],
                "plugins": [dict(row) for row in plugin_hashes],
            },
            "retention": {
                "status": retention_status,
                "database_cap_bytes": policy.database_cap_bytes,
                "wal_cap_bytes": policy.wal_cap_bytes,
                "database_bytes": measured_database_bytes,
                "wal_bytes": measured_wal_bytes,
                "terminal_callback_payload_hours": policy.callback_payload_us // 3_600_000_000,
                "callback_tombstone_hours": policy.tombstone_us // 3_600_000_000,
                "raw_market_event_hours": policy.raw_market_event_us // 3_600_000_000,
                "idea_and_shadow_result_years": 7,
            },
            "limit": limit,
        }

    def ideas(
        self,
        *,
        health: str | None,
        mode: str | None,
        active: bool | None,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        """List bounded generic plugin and instance summaries."""

        route = "/api/v2/ideas"
        filters = {"active": active, "health": health, "mode": mode}
        position = None if cursor is None else _decode_cursor(cursor, route=route, filters=filters)
        _validate_cursor_window(position, maximum_us=None)
        clauses: list[str] = []
        parameters: list[object] = []
        if self.config.run_id is not None:
            clauses.append("instance.run_id = ?")
            parameters.append(self.config.run_id)
        if health is not None:
            clauses.append("instance.health = ?")
            parameters.append(health)
        if mode is not None:
            clauses.append("instance.mode = ?")
            parameters.append(mode)
        if active is True:
            clauses.append("instance.deactivated_at_us IS NULL")
        elif active is False:
            clauses.append("instance.deactivated_at_us IS NOT NULL")
        if position is not None:
            clauses.append(
                "(instance.activated_at_us < ? OR "
                "(instance.activated_at_us = ? AND instance.instance_id > ?))"
            )
            parameters.extend((position.timestamp, position.timestamp, position.identity))
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT instance.instance_id, instance.idea_id, instance.idea_version, "
                "instance.run_id, instance.mode, instance.activated_at_us, "
                "instance.deactivated_at_us, instance.health, instance.error_code, "
                "instance.data_class, plugin.display_name, plugin.description, "
                "instance.manifest_hash, instance.plugin_code_hash, instance.parameters_hash, "
                "instance.universe_hash, instance.requirements_hash "
                "FROM idea_instances AS instance JOIN idea_plugins AS plugin "
                "ON plugin.idea_id = instance.idea_id "
                "AND plugin.idea_version = instance.idea_version "
                "AND plugin.code_hash = instance.plugin_code_hash "
                f"{where} ORDER BY instance.activated_at_us DESC, instance.instance_id "
                "LIMIT ?",
                (*parameters, limit + 1),
            ).fetchall()
            plugins = connection.execute(
                "SELECT idea_id, idea_version, api_version, display_name, description, "
                "manifest_hash, code_hash, discovered_at_us FROM idea_plugins "
                "ORDER BY discovered_at_us DESC, idea_id DESC, idea_version DESC LIMIT 100"
            ).fetchall()
        page = rows[:limit]
        next_cursor = None
        if len(rows) > limit and page:
            tail = page[-1]
            next_cursor = _encode_cursor(
                route=route,
                filters=filters,
                timestamp=int(tail["activated_at_us"]),
                identity=str(tail["instance_id"]),
            )
        return {
            "items": [self._idea_summary(row) for row in page],
            "plugins": [self._plugin_summary(row) for row in plugins],
            "limit": limit,
            "next_cursor": next_cursor,
        }

    def idea_detail(
        self,
        instance_id: str,
        *,
        kind: str | None,
        start_us: int | None,
        end_us: int | None,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any] | None:
        """Read one plugin instance and its sealed generic output stream."""

        route = "/api/v2/ideas/{instance_id}"
        filters = {
            "end_us": end_us,
            "instance_id": instance_id,
            "kind": kind,
            "start_us": start_us,
        }
        requested_window = (
            None
            if start_us is None and end_us is None
            else self._history_window(
                start_us=start_us,
                end_us=end_us,
                maximum_us=IDEA_WINDOW_US,
                latest_us=None,
            )
        )
        position = None if cursor is None else _decode_cursor(cursor, route=route, filters=filters)
        _validate_cursor_window(position, maximum_us=IDEA_WINDOW_US)
        if (
            position is not None
            and requested_window is not None
            and position.window != requested_window
        ):
            raise CursorError("invalid_cursor")
        with self._connection() as connection:
            instance_clauses = ["instance.instance_id = ?"]
            instance_parameters: list[object] = [instance_id]
            if self.config.run_id is not None:
                instance_clauses.append("instance.run_id = ?")
                instance_parameters.append(self.config.run_id)
            instance = connection.execute(
                "SELECT instance.instance_id, instance.idea_id, instance.idea_version, "
                "instance.run_id, instance.mode, instance.parameters_json, "
                "instance.parameters_hash, instance.plugin_code_hash, "
                "instance.manifest_hash, instance.universe_json, instance.universe_hash, "
                "instance.requirements_json, instance.requirements_hash, "
                "instance.activated_at_us, instance.deactivated_at_us, instance.health, "
                "instance.error_code, instance.data_class, plugin.display_name, "
                "plugin.description, plugin.api_version, plugin.manifest_json, "
                "plugin.discovered_at_us "
                "FROM idea_instances AS instance JOIN idea_plugins AS plugin "
                "ON plugin.idea_id = instance.idea_id "
                "AND plugin.idea_version = instance.idea_version "
                "AND plugin.code_hash = instance.plugin_code_hash "
                "WHERE " + " AND ".join(instance_clauses),
                tuple(instance_parameters),
            ).fetchone()
            if instance is None:
                return None
            interests = connection.execute(
                "SELECT interest_id, interest_key, underlying_instrument_id, asset_kind, "
                "minimum_days_to_expiry, maximum_days_to_expiry, option_right, strike_offset, "
                "reference_price, feed_kind, cadence, as_of_at_us, expires_at_us, required, "
                "priority, maximum_contracts, input_event_id, bound_subscription_id, "
                "content_hash, lifecycle, reason_code, attempts, next_attempt_at_us, "
                "created_at_us, updated_at_us "
                "FROM market_data_interests WHERE instance_id=? "
                "ORDER BY updated_at_us DESC, interest_id LIMIT 64",
                (instance_id,),
            ).fetchall()
            receipts = connection.execute(
                "SELECT receipt_id, interest_id, status, reason_code, instrument_id, expiry, "
                "strike, option_right, multiplier, candidates_inspected, completed_at_us "
                "FROM instrument_discovery_receipts WHERE instance_id=? "
                "ORDER BY completed_at_us DESC, receipt_id LIMIT 64",
                (instance_id,),
            ).fetchall()
            if position is not None:
                assert position.window is not None
                window = position.window
            elif requested_window is not None:
                window = requested_window
            else:
                latest_clauses = ["output.instance_id = ?"]
                latest_parameters: list[object] = [instance_id]
                if kind is not None:
                    latest_clauses.append("output.output_kind = ?")
                    latest_parameters.append(kind)
                latest = connection.execute(
                    "SELECT MAX(output.as_of_at_us) FROM idea_outputs AS output "
                    "JOIN idea_output_seals AS seal ON seal.output_id = output.output_id "
                    "WHERE " + " AND ".join(latest_clauses),
                    tuple(latest_parameters),
                ).fetchone()[0]
                window = self._history_window(
                    start_us=start_us,
                    end_us=end_us,
                    maximum_us=IDEA_WINDOW_US,
                    latest_us=None if latest is None else int(latest),
                )
            clauses = [
                "output.instance_id = ?",
                "output.as_of_at_us >= ?",
                "output.as_of_at_us <= ?",
            ]
            parameters: list[object] = [instance_id, window[0], window[1]]
            if kind is not None:
                clauses.append("output.output_kind = ?")
                parameters.append(kind)
            if position is not None:
                clauses.append(
                    "(output.as_of_at_us < ? OR (output.as_of_at_us = ? AND output.output_id > ?))"
                )
                parameters.extend((position.timestamp, position.timestamp, position.identity))
            rows = connection.execute(
                f"SELECT {OUTPUT_COLUMNS} FROM idea_outputs AS output "
                "JOIN idea_output_seals AS seal ON seal.output_id = output.output_id "
                "WHERE "
                + " AND ".join(clauses)
                + " ORDER BY output.as_of_at_us DESC, output.output_id LIMIT ?",
                (*parameters, limit + 1),
            ).fetchall()
            page = rows[:limit]
            legs_by_output: dict[str, list[dict[str, Any]]] = {
                str(row["output_id"]): [] for row in page
            }
            if page:
                placeholders = ",".join("?" for _ in page)
                legs = connection.execute(
                    "SELECT output_id, leg_number, instrument_id, action, target, "
                    "quantity_value, notional_value, currency, price_hint "
                    f"FROM idea_output_legs WHERE output_id IN ({placeholders}) "
                    "ORDER BY output_id, leg_number",
                    tuple(row["output_id"] for row in page),
                ).fetchall()
                for leg in legs:
                    legs_by_output[str(leg["output_id"])].append(
                        {key: value for key, value in dict(leg).items() if key != "output_id"}
                    )
        next_cursor = None
        if len(rows) > limit and page:
            tail = page[-1]
            next_cursor = _encode_cursor(
                route=route,
                filters=filters,
                timestamp=int(tail["as_of_at_us"]),
                identity=str(tail["output_id"]),
                window=window,
            )
        return {
            "instance": self._idea_detail_projection(instance),
            "market_data": {
                "interests": [self._interest_projection(row) for row in interests],
                "receipts": [dict(row) for row in receipts],
            },
            "outputs": [
                self._output_projection(row, legs_by_output[str(row["output_id"])]) for row in page
            ],
            "window": {"start_us": window[0], "end_us": window[1]},
            "limit": limit,
            "next_cursor": next_cursor,
        }

    @staticmethod
    def _result_status_sql() -> str:
        return (
            "CASE "
            "WHEN position.lifecycle = 'invalid' OR outcome.completeness = 'invalid' "
            "THEN 'invalid' "
            "WHEN position.lifecycle = 'pending' OR outcome.completeness = 'incomplete' "
            "THEN 'incomplete' "
            "WHEN position.lifecycle = 'open' THEN 'open' "
            "ELSE 'closed' END"
        )

    def _shadow_run_id(self, connection: sqlite3.Connection) -> str | None:
        if self.config.run_id is not None:
            row = connection.execute(
                "SELECT run_id FROM runs WHERE run_id = ? AND mode = 'shadow'",
                (self.config.run_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT run_id FROM runs WHERE mode = 'shadow' "
                "ORDER BY started_at_us DESC, run_id DESC LIMIT 1"
            ).fetchone()
        return None if row is None else str(row["run_id"])

    def results(
        self,
        *,
        status: str | None,
        instance_id: str | None,
        start_us: int | None,
        end_us: int | None,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        """List generic virtual positions from at most one 30-day window."""

        route = "/api/v2/results"
        filters = {
            "end_us": end_us,
            "instance_id": instance_id,
            "start_us": start_us,
            "status": status,
        }
        position_cursor = (
            None if cursor is None else _decode_cursor(cursor, route=route, filters=filters)
        )
        _validate_cursor_window(position_cursor, maximum_us=RESULT_WINDOW_US)
        requested_window = (
            None
            if start_us is None and end_us is None
            else self._history_window(
                start_us=start_us,
                end_us=end_us,
                maximum_us=RESULT_WINDOW_US,
                latest_us=None,
            )
        )
        if (
            position_cursor is not None
            and requested_window is not None
            and position_cursor.window != requested_window
        ):
            raise CursorError("invalid_cursor")
        status_sql = self._result_status_sql()
        with self._connection() as connection:
            run_id = self._shadow_run_id(connection)
            if run_id is None:
                return {
                    "items": [],
                    "aggregates": {"window_items": 0, "by_status": {}},
                    "window": (
                        None
                        if requested_window is None
                        else {"start_us": requested_window[0], "end_us": requested_window[1]}
                    ),
                    "limit": limit,
                    "next_cursor": None,
                    "language": "virtual only; no broker orders, fills, or positions",
                }
            joins = (
                "FROM idea_outputs AS output "
                "JOIN idea_output_seals AS seal ON seal.output_id = output.output_id "
                "JOIN shadow_positions AS position "
                "ON position.proposed_trade_output_id = output.output_id "
                "LEFT JOIN shadow_outcomes AS outcome "
                "ON outcome.position_id = position.position_id "
            )
            if position_cursor is not None:
                assert position_cursor.window is not None
                window = position_cursor.window
            elif requested_window is not None:
                window = requested_window
            else:
                latest_clauses = [
                    "output.run_id = ?",
                    "output.output_kind = 'proposed_trade'",
                ]
                latest_parameters: list[object] = [run_id]
                if instance_id is not None:
                    latest_clauses.append("position.instance_id = ?")
                    latest_parameters.append(instance_id)
                if status is not None:
                    latest_clauses.append(f"({status_sql}) = ?")
                    latest_parameters.append(status)
                latest = connection.execute(
                    "SELECT MAX(output.as_of_at_us) "
                    + joins
                    + "WHERE "
                    + " AND ".join(latest_clauses),
                    tuple(latest_parameters),
                ).fetchone()[0]
                window = self._history_window(
                    start_us=start_us,
                    end_us=end_us,
                    maximum_us=RESULT_WINDOW_US,
                    latest_us=None if latest is None else int(latest),
                )
            clauses = [
                "output.run_id = ?",
                "output.output_kind = 'proposed_trade'",
                "output.as_of_at_us >= ?",
                "output.as_of_at_us <= ?",
            ]
            parameters: list[object] = [run_id, window[0], window[1]]
            if instance_id is not None:
                clauses.append("position.instance_id = ?")
                parameters.append(instance_id)
            if status is not None:
                clauses.append(f"({status_sql}) = ?")
                parameters.append(status)
            aggregate_clauses = list(clauses)
            aggregate_parameters = list(parameters)
            if position_cursor is not None:
                clauses.append(
                    "(output.as_of_at_us < ? OR "
                    "(output.as_of_at_us = ? AND position.position_id > ?))"
                )
                parameters.extend(
                    (
                        position_cursor.timestamp,
                        position_cursor.timestamp,
                        position_cursor.identity,
                    )
                )
            rows = connection.execute(
                "SELECT position.position_id, position.proposed_trade_output_id, "
                "position.run_id, position.instance_id, position.opened_at_us, "
                "position.closed_at_us, position.lifecycle, position.cost_model_id, "
                "position.fill_model_id, position.currency, position.invalid_reason, "
                "position.data_class, output.as_of_at_us AS result_at_us, "
                "outcome.outcome_at_us, outcome.net_pnl, outcome.return_value, "
                f"outcome.completeness, {status_sql} AS status {joins} WHERE "
                + " AND ".join(clauses)
                + " ORDER BY output.as_of_at_us DESC, position.position_id LIMIT ?",
                (*parameters, limit + 1),
            ).fetchall()
            aggregates = connection.execute(
                f"SELECT {status_sql} AS status, COUNT(*) AS item_count {joins} WHERE "
                + " AND ".join(aggregate_clauses)
                + " GROUP BY status ORDER BY status",
                tuple(aggregate_parameters),
            ).fetchall()
        page = rows[:limit]
        next_cursor = None
        if len(rows) > limit and page:
            tail = page[-1]
            next_cursor = _encode_cursor(
                route=route,
                filters=filters,
                timestamp=int(tail["result_at_us"]),
                identity=str(tail["position_id"]),
                window=window,
            )
        counts = {str(row["status"]): int(row["item_count"]) for row in aggregates}
        return {
            "items": [self._result_summary(row) for row in page],
            "aggregates": {"window_items": sum(counts.values()), "by_status": counts},
            "window": {"start_us": window[0], "end_us": window[1]},
            "limit": limit,
            "next_cursor": next_cursor,
            "language": "virtual only; no broker orders, fills, or positions",
        }

    @staticmethod
    def _result_summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "position_id": row["position_id"],
            "source_proposal_id": row["proposed_trade_output_id"],
            "run_id": row["run_id"],
            "instance_id": row["instance_id"],
            "status": row["status"],
            "lifecycle": row["lifecycle"],
            "result_at_us": row["result_at_us"],
            "opened_at_us": row["opened_at_us"],
            "closed_at_us": row["closed_at_us"],
            "outcome_at_us": row["outcome_at_us"],
            "net_pnl": row["net_pnl"],
            "return_value": row["return_value"],
            "completeness": row["completeness"],
            "invalid_reason": row["invalid_reason"],
            "currency": row["currency"],
            "cost_model_id": row["cost_model_id"],
            "fill_model_id": row["fill_model_id"],
            "data_class": row["data_class"],
            "shadow": True,
            "broker_position": False,
            "fill": False,
        }

    def result_detail(
        self,
        position_id: str,
        *,
        start_us: int | None,
        end_us: int | None,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any] | None:
        """Read one shadow position with bounded marks and exact provenance."""

        route = "/api/v2/results/{position_id}"
        filters = {"end_us": end_us, "position_id": position_id, "start_us": start_us}
        requested_window = (
            None
            if start_us is None and end_us is None
            else self._history_window(
                start_us=start_us,
                end_us=end_us,
                maximum_us=RESULT_WINDOW_US,
                latest_us=None,
            )
        )
        mark_cursor = (
            None if cursor is None else _decode_cursor(cursor, route=route, filters=filters)
        )
        _validate_cursor_window(mark_cursor, maximum_us=RESULT_WINDOW_US)
        if (
            mark_cursor is not None
            and requested_window is not None
            and mark_cursor.window != requested_window
        ):
            raise CursorError("invalid_cursor")
        status_sql = self._result_status_sql()
        with self._connection() as connection:
            run_id = self._shadow_run_id(connection)
            if run_id is None:
                return None
            position = connection.execute(
                "SELECT position.position_id, position.proposed_trade_output_id, "
                "position.run_id, position.instance_id, position.opened_at_us, "
                "position.closed_at_us, position.lifecycle, position.cost_model_id, "
                "position.fill_model_id, position.currency, position.invalid_reason, "
                "position.data_class, position.policy_hash, "
                "output.as_of_at_us AS result_at_us, "
                f"{status_sql} AS status FROM shadow_positions AS position "
                "JOIN idea_outputs AS output "
                "ON output.output_id = position.proposed_trade_output_id "
                "LEFT JOIN shadow_outcomes AS outcome "
                "ON outcome.position_id = position.position_id "
                "WHERE position.position_id = ? AND position.run_id = ?",
                (position_id, run_id),
            ).fetchone()
            if position is None:
                return None
            proposal = connection.execute(
                f"SELECT {OUTPUT_COLUMNS} FROM idea_outputs AS output "
                "JOIN idea_output_seals AS seal ON seal.output_id = output.output_id "
                "WHERE output.output_id = ?",
                (position["proposed_trade_output_id"],),
            ).fetchone()
            if proposal is None:
                return None
            proposal_legs = [
                {key: value for key, value in dict(row).items() if key != "output_id"}
                for row in connection.execute(
                    "SELECT output_id, leg_number, instrument_id, action, target, "
                    "quantity_value, notional_value, currency, price_hint "
                    "FROM idea_output_legs WHERE output_id = ? ORDER BY leg_number",
                    (proposal["output_id"],),
                )
            ]
            legs = [
                dict(row)
                for row in connection.execute(
                    "SELECT leg_number, instrument_id, side, quantity, entry_market_event_id, "
                    "entry_price, exit_market_event_id, exit_price, "
                    "entry_bid_market_event_id, entry_ask_market_event_id, "
                    "exit_bid_market_event_id, exit_ask_market_event_id "
                    "FROM shadow_legs WHERE position_id = ? ORDER BY leg_number",
                    (position_id,),
                )
            ]
            outcome_row = connection.execute(
                "SELECT position_id, outcome_at_us, reason, gross_pnl, net_pnl, "
                "return_value, mfe, mae, completeness, payload_json "
                "FROM shadow_outcomes WHERE position_id = ?",
                (position_id,),
            ).fetchone()
            latest = connection.execute(
                "SELECT MAX(marked_at_us) FROM shadow_marks WHERE position_id = ?",
                (position_id,),
            ).fetchone()[0]
            latest_us = int(position["result_at_us"]) if latest is None else int(latest)
            if mark_cursor is not None:
                assert mark_cursor.window is not None
                window = mark_cursor.window
            elif requested_window is not None:
                window = requested_window
            else:
                window = self._history_window(
                    start_us=start_us,
                    end_us=end_us,
                    maximum_us=RESULT_WINDOW_US,
                    latest_us=latest_us,
                )
            mark_clauses = [
                "position_id = ?",
                "marked_at_us >= ?",
                "marked_at_us <= ?",
            ]
            mark_parameters: list[object] = [position_id, window[0], window[1]]
            if mark_cursor is not None:
                mark_clauses.append("(marked_at_us < ? OR (marked_at_us = ? AND position_id > ?))")
                mark_parameters.extend(
                    (mark_cursor.timestamp, mark_cursor.timestamp, mark_cursor.identity)
                )
            mark_rows = connection.execute(
                "SELECT marked_at_us, gross_value, gross_pnl, net_pnl, return_value, "
                "quality_bits, payload_json FROM shadow_marks WHERE "
                + " AND ".join(mark_clauses)
                + " ORDER BY marked_at_us DESC, position_id LIMIT ?",
                (*mark_parameters, limit + 1),
            ).fetchall()
        page = mark_rows[:limit]
        next_cursor = None
        if len(mark_rows) > limit and page:
            tail = page[-1]
            next_cursor = _encode_cursor(
                route=route,
                filters=filters,
                timestamp=int(tail["marked_at_us"]),
                identity=position_id,
                window=window,
            )
        outcome = None if outcome_row is None else dict(outcome_row)
        if outcome is not None:
            outcome["payload"] = json.loads(outcome.pop("payload_json"))
        position_projection = {
            **dict(position),
            "shadow": True,
            "broker_position": False,
            "fill": False,
        }
        return {
            "language": "virtual only; no broker orders, fills, or positions",
            "position": position_projection,
            "source_proposal": self._output_projection(proposal, proposal_legs),
            "legs": legs,
            "marks": [
                {
                    **{key: value for key, value in dict(row).items() if key != "payload_json"},
                    "payload": json.loads(row["payload_json"]),
                }
                for row in page
            ],
            "outcome": outcome,
            "window": {"start_us": window[0], "end_us": window[1]},
            "limit": limit,
            "next_cursor": next_cursor,
        }

    @staticmethod
    def _history_window(
        *,
        start_us: int | None,
        end_us: int | None,
        maximum_us: int,
        latest_us: int | None,
    ) -> tuple[int, int]:
        if start_us is None and end_us is None:
            resolved_end = time.time_ns() // 1_000 if latest_us is None else latest_us
            resolved_start = max(0, resolved_end - maximum_us)
        elif start_us is None:
            assert end_us is not None
            resolved_end = end_us
            resolved_start = max(0, resolved_end - maximum_us)
        elif end_us is None:
            resolved_start = start_us
            resolved_end = start_us + maximum_us
        else:
            resolved_start = start_us
            resolved_end = end_us
        if (
            resolved_start < 0
            or resolved_end < resolved_start
            or resolved_end > SQLITE_INTEGER_MAX
            or resolved_end - resolved_start > maximum_us
        ):
            raise WindowError("invalid_window")
        return resolved_start, resolved_end

    def _idea_detail_projection(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            **self._idea_summary(row),
            "api_version": row["api_version"],
            "discovered_at_us": row["discovered_at_us"],
            "manifest": json.loads(row["manifest_json"]),
            "parameters": json.loads(row["parameters_json"]),
            "universe": json.loads(row["universe_json"]),
            "requirements": json.loads(row["requirements_json"]),
        }

    @staticmethod
    def _interest_projection(row: sqlite3.Row) -> dict[str, Any]:
        return {
            **dict(row),
            "required": bool(row["required"]),
        }

    @staticmethod
    def _output_projection(
        row: sqlite3.Row,
        legs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "output_id": row["output_id"],
            "kind": row["output_kind"],
            "subject_instrument_id": row["subject_instrument_id"],
            "emitted_at_us": row["emitted_at_us"],
            "as_of_at_us": row["as_of_at_us"],
            "valid_until_at_us": row["valid_until_at_us"],
            "direction": row["direction"],
            "strength": row["strength"],
            "confidence": row["confidence"],
            "horizon_us": row["horizon_us"],
            "authority": row["authority_status"],
            "data_class": row["data_class"],
            "input": {
                "first_event_id": row["first_input_event_id"],
                "last_event_id": row["last_input_event_id"],
                "watermark": row["input_watermark"],
                "events_hash": row["input_events_hash"],
            },
            "payload": json.loads(row["payload_json"]),
            "content_hash": row["content_hash"],
            "legs": legs,
        }

    @staticmethod
    def _bounded_text(value: object, maximum: int) -> str:
        text = str(value)
        return text if len(text) <= maximum else text[: maximum - 1] + "…"

    def _plugin_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "idea_id": row["idea_id"],
            "idea_version": row["idea_version"],
            "api_version": row["api_version"],
            "display_name": self._bounded_text(row["display_name"], 256),
            "description": self._bounded_text(row["description"], 1_024),
            "manifest_hash": row["manifest_hash"],
            "code_hash": row["code_hash"],
            "discovered_at_us": row["discovered_at_us"],
        }

    def _idea_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "instance_id": row["instance_id"],
            "idea_id": row["idea_id"],
            "idea_version": row["idea_version"],
            "display_name": self._bounded_text(row["display_name"], 256),
            "description": self._bounded_text(row["description"], 1_024),
            "run_id": row["run_id"],
            "mode": row["mode"],
            "activated_at_us": row["activated_at_us"],
            "deactivated_at_us": row["deactivated_at_us"],
            "health": row["health"],
            "error_code": row["error_code"],
            "data_class": row["data_class"],
            "hashes": {
                "manifest": row["manifest_hash"],
                "plugin": row["plugin_code_hash"],
                "parameters": row["parameters_hash"],
                "universe": row["universe_hash"],
                "requirements": row["requirements_hash"],
            },
        }

    def _storage(self, state: dict[str, Any] | None) -> dict[str, Any]:
        database = self.config.database
        wal = Path(f"{database}-wal")
        try:
            database_bytes = database.stat().st_size
        except OSError:
            database_bytes = None
        try:
            wal_bytes = wal.stat().st_size
        except OSError:
            wal_bytes = 0
        return {
            "database_bytes": database_bytes,
            "wal_bytes": wal_bytes,
            "reported_database_bytes": 0 if state is None else state["database_bytes"],
            "reported_wal_bytes": 0 if state is None else state["wal_bytes"],
        }
