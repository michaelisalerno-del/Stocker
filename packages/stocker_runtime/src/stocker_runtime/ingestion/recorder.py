"""Fail-closed configuration and lifecycle for the V2 prospective recorder."""

from __future__ import annotations

import hashlib
import ipaddress
import sqlite3
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stocker_runtime.domain import JsonValue, canonical_json_bytes
from stocker_runtime.ingestion.ibkr_market_data import MarketDataAdapter
from stocker_runtime.ingestion.inbox import (
    AdmissionResult,
    CallbackFence,
    CallbackIdentityCollision,
    CallbackInbox,
    InboxAdmissionError,
    MarketDataCallback,
    NormalizationError,
)
from stocker_runtime.storage import RetentionManager, StorageCapState, connect_v2


class DuplicateWriterError(RuntimeError):
    """Another recorder owns a fresh authoritative-writer lease."""


class RecorderFatalError(RuntimeError):
    """The recorder entered a persisted fail-stop state."""


@dataclass(frozen=True)
class InstrumentSpec:
    instrument_id: str
    ibkr_con_id: int | None
    kind: str
    symbol: str
    exchange: str
    currency: str


@dataclass(frozen=True)
class SubscriptionSpec:
    name: str
    instrument_id: str
    feed_kind: str
    request_id: int
    continuity_required: bool
    optional: bool
    stale_after_us: int

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.instrument_id
            or not self.feed_kind
            or self.request_id < 0
            or self.stale_after_us <= 0
        ):
            raise ValueError("subscription identity and staleness bound are required")


@dataclass(frozen=True)
class RecorderState:
    run_id: str
    recorder_generation: int
    connection_generation: int
    fences: tuple[CallbackFence, ...]


class RecorderConfig(BaseModel):
    """Strict recorder configuration containing no credentials or broker identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    database: Path
    run_id: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    mode: Literal["prospective_record"]
    host: str
    port: int = Field(ge=1, le=65_535)
    client_id: int = Field(ge=0)
    read_only: Literal[True]
    external_read_only_verified: Literal[True]
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    git_commit: str = Field(pattern=r"^[a-f0-9]{7,64}$")
    writer_lease_stale_us: int = Field(default=60_000_000, ge=5_000_000)
    callback_lease_us: int = Field(default=30_000_000, ge=5_000_000)

    @model_validator(mode="after")
    def loopback_only(self) -> Self:
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError as error:
            raise ValueError("IBKR host must be a literal loopback address") from error
        if not address.is_loopback:
            raise ValueError("IBKR host must be a literal loopback address")
        return self


def load_recorder_config(path: str | Path) -> RecorderConfig:
    """Load a strict JSON configuration without inspecting environment variables."""

    return RecorderConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _safe_adapter(adapter: object) -> None:
    forbidden = ("order", "account", "position", "execution", "pnl", "fill", "portfolio")
    unsafe = sorted(
        name
        for name in dir(adapter)
        if not name.startswith("_")
        and callable(getattr(adapter, name, None))
        and any(term in name.lower() for term in forbidden)
    )
    if unsafe:
        raise RecorderFatalError(f"unsafe broker capability is public: {','.join(unsafe)}")
    required = {
        "set_callback",
        "set_disconnect_callback",
        "connect",
        "disconnect",
        "subscribe",
        "cancel",
    }
    missing = sorted(name for name in required if not callable(getattr(adapter, name, None)))
    if missing:
        raise RecorderFatalError(f"market-data adapter surface is incomplete: {','.join(missing)}")


class Recorder:
    """Sole-writer lifecycle over a market-data-only adapter and durable inbox."""

    def __init__(self, config: RecorderConfig, adapter: MarketDataAdapter) -> None:
        _safe_adapter(adapter)
        self.config = config
        self.adapter = adapter
        self.inbox = CallbackInbox(config.database)
        self.state: RecorderState | None = None
        self._instruments: tuple[InstrumentSpec, ...] = ()
        self._subscriptions: tuple[SubscriptionSpec, ...] = ()

    def start(
        self,
        *,
        now_us: int,
        instruments: tuple[InstrumentSpec, ...],
        subscriptions: tuple[SubscriptionSpec, ...],
    ) -> RecorderState:
        """Acquire the writer generation, recover the inbox, then connect."""

        if self.state is not None:
            raise DuplicateWriterError("this recorder is already started")
        self._instruments = instruments
        self._subscriptions = subscriptions
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT state.run_id, state.process_heartbeat_at_us, run.status "
                "FROM runtime_state state JOIN runs run ON run.run_id = state.run_id "
                "WHERE state.lifecycle IN ('starting','recovering','running','degraded') "
                "ORDER BY state.run_id LIMIT 1"
            ).fetchone()
            if active is not None:
                heartbeat = active["process_heartbeat_at_us"]
                fresh = (
                    heartbeat is None
                    or now_us - int(heartbeat) <= self.config.writer_lease_stale_us
                )
                if fresh:
                    raise DuplicateWriterError(
                        f"authoritative writer lease is held by run {active['run_id']}"
                    )
                gap_start = (
                    now_us
                    if active["process_heartbeat_at_us"] is None
                    else int(active["process_heartbeat_at_us"])
                )
                self._close_stale_writer(
                    connection,
                    str(active["run_id"]),
                    now_us=now_us,
                    gap_start_us=min(gap_start, now_us),
                    continuing_same_run=str(active["run_id"]) == self.config.run_id,
                )
            run = connection.execute(
                "SELECT mode, config_hash, status FROM runs WHERE run_id = ?",
                (self.config.run_id,),
            ).fetchone()
            if run is None:
                connection.execute(
                    "INSERT INTO runs(run_id, mode, source, started_at_us, config_hash, "
                    "git_commit, data_class, status) VALUES (?, 'prospective_record', 'ibkr', "
                    "?, ?, ?, 'prospective_protected', 'running')",
                    (
                        self.config.run_id,
                        now_us,
                        self.config.config_hash,
                        self.config.git_commit,
                    ),
                )
            elif (
                str(run["mode"]) != "prospective_record"
                or str(run["config_hash"]) != self.config.config_hash
            ):
                raise RecorderFatalError("run mode or frozen configuration changed")
            elif str(run["status"]) == "fatal":
                raise RecorderFatalError("persisted fatal recorder state requires operator action")
            elif str(run["status"]) == "stopped":
                raise RecorderFatalError(
                    "a cleanly stopped run cannot be restarted; use a new run_id"
                )
            previous_state = connection.execute(
                "SELECT recorder_generation, connection_generation FROM runtime_state "
                "WHERE run_id = ?",
                (self.config.run_id,),
            ).fetchone()
            generation = int(
                connection.execute(
                    "SELECT COALESCE(MAX(generation), 0) + 1 FROM recorder_generations "
                    "WHERE run_id = ?",
                    (self.config.run_id,),
                ).fetchone()[0]
            )
            connection_generation = (
                1 if previous_state is None else int(previous_state["connection_generation"]) + 1
            )
            connection.execute(
                "INSERT INTO recorder_generations(run_id, generation, owner_id, started_at_us) "
                "VALUES (?, ?, ?, ?)",
                (self.config.run_id, generation, self.config.owner_id, now_us),
            )
            connection.execute(
                "INSERT INTO runtime_state(run_id, recorder_generation, lifecycle, reason, "
                "process_heartbeat_at_us, connection_generation, inbox_nonterminal_count) "
                "VALUES (?, ?, 'recovering', NULL, ?, ?, "
                "(SELECT count(*) FROM callback_inbox WHERE lifecycle IN ('pending','leased'))) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "recorder_generation=excluded.recorder_generation, "
                "lifecycle=excluded.lifecycle, reason=NULL, "
                "process_heartbeat_at_us=excluded.process_heartbeat_at_us, "
                "connection_generation=excluded.connection_generation, "
                "inbox_nonterminal_count=excluded.inbox_nonterminal_count",
                (self.config.run_id, generation, now_us, connection_generation),
            )
            self._upsert_instruments(connection, instruments)
            fences = self._install_subscriptions(
                connection,
                generation,
                connection_generation,
                subscriptions,
                now_us,
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        self.state = RecorderState(self.config.run_id, generation, connection_generation, fences)
        self.inbox.reclaim_expired_leases(now_us=now_us)
        self.drain(now_us=now_us)
        cap = RetentionManager(self.config.database).run(now_us=now_us)
        with connect_v2(self.config.database) as size_connection:
            size_connection.execute(
                "UPDATE runtime_state SET database_bytes=?, wal_bytes=? WHERE run_id=?",
                (cap.database_bytes, cap.wal_bytes, self.config.run_id),
            )
        if not cap.admission_allowed:
            self._fatal(cap.required_action or "STORAGE_CAP_FATAL", now_us)
            raise RecorderFatalError(cap.required_action or "storage cap closed admission")
        if not cap.optional_feeds_allowed:
            self._pause_optional(now_us)
        cast(
            Callable[[Callable[[CallbackFence, MarketDataCallback], AdmissionResult]], None],
            self.adapter.set_callback,
        )(self.receive)
        self.adapter.set_disconnect_callback(
            lambda disconnected_at_us: self.disconnected(now_us=disconnected_at_us)
        )
        self.adapter.connect()
        optional_requests = {spec.request_id for spec in self._subscriptions if spec.optional}
        for fence in self.state.fences:
            if not cap.optional_feeds_allowed and fence.request_id in optional_requests:
                continue
            self.adapter.subscribe(fence)
        self._set_lifecycle("running", None, now_us)
        return self.state

    def _close_stale_writer(
        self,
        connection: sqlite3.Connection,
        stale_run_id: str,
        *,
        now_us: int,
        gap_start_us: int,
        continuing_same_run: bool,
    ) -> None:
        rows = tuple(
            connection.execute(
                "SELECT subscription_id FROM subscriptions WHERE run_id = ? "
                "AND lifecycle = 'active'",
                (stale_run_id,),
            )
        )
        for row in rows:
            self._open_gap_for_run(
                connection,
                stale_run_id,
                str(row["subscription_id"]),
                gap_start_us,
                "UNCLEAN_RECORDER_RESTART",
                True,
            )
        connection.execute(
            "UPDATE subscriptions SET lifecycle='closed', closed_at_us=? WHERE run_id=? "
            "AND lifecycle='active'",
            (now_us, stale_run_id),
        )
        connection.execute(
            "UPDATE recorder_generations SET ended_at_us=?, clean_stop=0, "
            "termination_code='UNCLEAN_RESTART' WHERE run_id=? AND ended_at_us IS NULL",
            (now_us, stale_run_id),
        )
        connection.execute(
            "UPDATE runtime_state SET lifecycle='stopped', reason='UNCLEAN_RESTART', "
            "process_heartbeat_at_us=? WHERE run_id=?",
            (now_us, stale_run_id),
        )
        if not continuing_same_run:
            connection.execute(
                "UPDATE runs SET status='stopped', ended_at_us=? WHERE run_id=?",
                (now_us, stale_run_id),
            )

    @staticmethod
    def _identity_hash(spec: InstrumentSpec) -> str:
        material = cast(
            JsonValue,
            {
                "instrument_id": spec.instrument_id,
                "ibkr_con_id": spec.ibkr_con_id,
                "kind": spec.kind,
                "symbol": spec.symbol,
                "exchange": spec.exchange,
                "currency": spec.currency,
            },
        )
        return hashlib.sha256(canonical_json_bytes(material)).hexdigest()

    def _upsert_instruments(
        self, connection: sqlite3.Connection, instruments: tuple[InstrumentSpec, ...]
    ) -> None:
        for spec in instruments:
            identity_hash = self._identity_hash(spec)
            existing = connection.execute(
                "SELECT identity_hash FROM instruments WHERE instrument_id = ?",
                (spec.instrument_id,),
            ).fetchone()
            if existing is not None and str(existing[0]) != identity_hash:
                raise RecorderFatalError("instrument identity changed within the operational store")
            connection.execute(
                "INSERT OR IGNORE INTO instruments(instrument_id, identity_hash, ibkr_con_id, "
                "kind, symbol, exchange, currency) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    spec.instrument_id,
                    identity_hash,
                    spec.ibkr_con_id,
                    spec.kind,
                    spec.symbol,
                    spec.exchange,
                    spec.currency,
                ),
            )

    def _install_subscriptions(
        self,
        connection: sqlite3.Connection,
        generation: int,
        connection_generation: int,
        subscriptions: tuple[SubscriptionSpec, ...],
        now_us: int,
    ) -> tuple[CallbackFence, ...]:
        fences: list[CallbackFence] = []
        for spec in subscriptions:
            material = cast(
                JsonValue,
                {
                    "name": spec.name,
                    "instrument_id": spec.instrument_id,
                    "feed_kind": spec.feed_kind,
                    "continuity_required": spec.continuity_required,
                    "optional": spec.optional,
                    "stale_after_us": spec.stale_after_us,
                },
            )
            requirements_hash = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
            subscription_id = hashlib.sha256(
                f"{self.config.run_id}|{connection_generation}|{spec.name}".encode()
            ).hexdigest()
            connection.execute(
                "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
                "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
                "requirements_hash, opened_at_us) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                (
                    subscription_id,
                    self.config.run_id,
                    generation,
                    connection_generation,
                    spec.instrument_id,
                    spec.feed_kind,
                    spec.request_id,
                    requirements_hash,
                    now_us,
                ),
            )
            fences.append(
                CallbackFence(
                    self.config.run_id,
                    generation,
                    connection_generation,
                    spec.request_id,
                    subscription_id,
                )
            )
        return tuple(fences)

    def receive(self, fence: CallbackFence, callback: MarketDataCallback) -> AdmissionResult:
        """External callback boundary: durable admission completes before return."""

        if self.state is None:
            raise RecorderFatalError("recorder is not started")
        try:
            return self.inbox.admit(fence, callback)
        except InboxAdmissionError:
            self._fatal("CALLBACK_ADMISSION_FAILED", callback.received_at_us)
            raise

    def drain(self, *, now_us: int, limit: int = 256) -> int:
        """Recover and process one bounded callback batch without blocking on poison."""

        if self.state is None:
            raise RecorderFatalError("recorder is not started")
        processed = 0
        for leased in self.inbox.lease_pending(
            self.config.owner_id,
            now_us=now_us,
            lease_us=self.config.callback_lease_us,
            limit=limit,
        ):
            try:
                result = self.inbox.project(leased)
            except NormalizationError:
                self.inbox.fail(leased, "MALFORMED_CALLBACK", failed_at_us=now_us)
                continue
            except CallbackIdentityCollision:
                self._fatal("EVENT_IDENTITY_COLLISION", now_us)
                raise
            self.inbox.acknowledge(leased, result.event_id, acknowledged_at_us=now_us)
            processed += 1
        self.inbox.create_pending_receipts(created_at_us=now_us, limit=limit)
        self._heartbeat(now_us)
        return processed

    def _heartbeat(self, now_us: int) -> None:
        with connect_v2(self.config.database) as connection:
            connection.execute(
                "UPDATE runtime_state SET process_heartbeat_at_us=? WHERE run_id=? "
                "AND recorder_generation=?",
                (now_us, self.config.run_id, self.state.recorder_generation if self.state else -1),
            )

    def mark_stale(self, *, now_us: int, market_data_expected: bool = True) -> int:
        """Open per-subscription gaps; optional staleness never blocks required feeds."""

        if self.state is None:
            raise RecorderFatalError("recorder is not started")
        if not market_data_expected:
            return 0
        opened = 0
        by_request = {spec.request_id: spec for spec in self._subscriptions}
        with connect_v2(self.config.database) as connection:
            rows = tuple(
                connection.execute(
                    "SELECT subscription.subscription_id, subscription.request_id, "
                    "subscription.opened_at_us, event.received_at_us "
                    "FROM subscriptions subscription LEFT JOIN market_events event "
                    "ON event.event_id=subscription.latest_event_id WHERE subscription.run_id=? "
                    "AND subscription.connection_generation=? AND subscription.lifecycle='active'",
                    (self.config.run_id, self.state.connection_generation),
                )
            )
            for row in rows:
                spec = by_request[int(row["request_id"])]
                reference = int(
                    row["opened_at_us"] if row["received_at_us"] is None else row["received_at_us"]
                )
                if now_us - reference < spec.stale_after_us:
                    continue
                unresolved = connection.execute(
                    "SELECT 1 FROM gaps WHERE run_id=? AND subscription_id=? "
                    "AND reason='STREAM_STALE' AND resolved_at_us IS NULL",
                    (self.config.run_id, str(row["subscription_id"])),
                ).fetchone()
                if unresolved is None:
                    self._open_gap(
                        connection,
                        str(row["subscription_id"]),
                        reference + spec.stale_after_us,
                        "STREAM_STALE",
                        spec.continuity_required,
                    )
                    opened += 1
        return opened

    def disconnected(self, *, now_us: int) -> None:
        """Treat a temporary socket loss as recoverable degraded state."""

        if self.state is None:
            raise RecorderFatalError("recorder is not started")
        required = {spec.request_id: spec.continuity_required for spec in self._subscriptions}
        with connect_v2(self.config.database) as connection:
            for fence in self.state.fences:
                self._open_gap(
                    connection,
                    cast(str, fence.subscription_id),
                    now_us,
                    "IBKR_DISCONNECT",
                    required[cast(int, fence.request_id)],
                )
        self.adapter.disconnect()
        self._set_lifecycle("degraded", "IBKR_DISCONNECT", now_us)

    def reconnect(self, *, now_us: int) -> RecorderState:
        """Fence old requests and reconnect with a new durable socket generation."""

        if self.state is None:
            raise RecorderFatalError("recorder is not started")
        old = self.state
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE subscriptions SET lifecycle='closed', closed_at_us=? WHERE run_id=? "
                "AND connection_generation=? AND lifecycle='active'",
                (now_us, self.config.run_id, old.connection_generation),
            )
            connection.execute(
                "UPDATE gaps SET ended_at_us=?, resolved_at_us=? WHERE run_id=? "
                "AND reason='IBKR_DISCONNECT' AND resolved_at_us IS NULL",
                (now_us, now_us, self.config.run_id),
            )
            generation = old.connection_generation + 1
            fences = self._install_subscriptions(
                connection,
                old.recorder_generation,
                generation,
                self._subscriptions,
                now_us,
            )
            for fence, spec in zip(fences, self._subscriptions, strict=True):
                self._open_gap(
                    connection,
                    cast(str, fence.subscription_id),
                    now_us,
                    "RECONNECT_UNCERTAINTY",
                    spec.continuity_required,
                )
            connection.execute(
                "UPDATE runtime_state SET connection_generation=?, lifecycle='running', "
                "reason=NULL, process_heartbeat_at_us=? WHERE run_id=?",
                (generation, now_us, self.config.run_id),
            )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        self.state = RecorderState(self.config.run_id, old.recorder_generation, generation, fences)
        self.adapter.connect()
        for fence in fences:
            self.adapter.subscribe(fence)
        return self.state

    def stop(self, *, now_us: int) -> None:
        """Close subscriptions and the writer generation without changing mode."""

        if self.state is None:
            return
        self.adapter.disconnect()
        with connect_v2(self.config.database) as connection:
            connection.execute(
                "UPDATE subscriptions SET lifecycle='closed', closed_at_us=? WHERE run_id=? "
                "AND lifecycle IN ('active','paused')",
                (now_us, self.config.run_id),
            )
            connection.execute(
                "UPDATE recorder_generations SET ended_at_us=?, clean_stop=1, "
                "termination_code='CLEAN_STOP' WHERE run_id=? AND generation=?",
                (
                    now_us,
                    self.config.run_id,
                    self.state.recorder_generation,
                ),
            )
            connection.execute(
                "UPDATE runtime_state SET lifecycle='stopped', reason=NULL, "
                "process_heartbeat_at_us=? WHERE run_id=?",
                (now_us, self.config.run_id),
            )
            connection.execute(
                "UPDATE runs SET status='stopped', ended_at_us=? WHERE run_id=?",
                (now_us, self.config.run_id),
            )
        self.state = None

    def _pause_optional(self, now_us: int) -> None:
        if self.state is None:
            return
        optional = {spec.request_id: spec for spec in self._subscriptions if spec.optional}
        with connect_v2(self.config.database) as connection:
            for fence in self.state.fences:
                if fence.request_id is None:
                    continue
                spec = optional.get(fence.request_id)
                if spec is None:
                    continue
                self.adapter.cancel(fence.request_id)
                connection.execute(
                    "UPDATE subscriptions SET lifecycle='paused', closed_at_us=? "
                    "WHERE subscription_id=?",
                    (now_us, fence.subscription_id),
                )
                self._open_gap(
                    connection,
                    cast(str, fence.subscription_id),
                    now_us,
                    "STORAGE_DEGRADED_OPTIONAL_PAUSED",
                    spec.continuity_required,
                )

    def _fatal(self, code: str, now_us: int) -> None:
        if self.state is not None:
            for fence in self.state.fences:
                if fence.request_id is not None:
                    with suppress(Exception):
                        self.adapter.cancel(fence.request_id)
        try:
            connection = connect_v2(self.config.database)
            try:
                connection.execute("BEGIN IMMEDIATE")
                CallbackInbox._record_fatal(connection, self.config.run_id, now_us, code)
                connection.commit()
            finally:
                connection.close()
        except (OSError, sqlite3.Error):
            pass
        finally:
            self.adapter.disconnect()

    def maintain(self, *, now_us: int) -> StorageCapState:
        """Consume one bounded Phase 2 retention result and apply recorder reactions."""

        if self.state is None:
            raise RecorderFatalError("recorder is not started")
        result = RetentionManager(self.config.database).run(now_us=now_us)
        with connect_v2(self.config.database) as connection:
            connection.execute(
                "UPDATE runtime_state SET database_bytes=?, wal_bytes=? WHERE run_id=?",
                (result.database_bytes, result.wal_bytes, self.config.run_id),
            )
        if not result.admission_allowed:
            self._fatal(result.required_action or "STORAGE_CAP_FATAL", now_us)
            raise RecorderFatalError(result.required_action or "storage cap closed admission")
        if not result.optional_feeds_allowed:
            self._pause_optional(now_us)
            self._set_lifecycle("degraded", result.required_action, now_us)
        return result.cap_state

    def _set_lifecycle(self, lifecycle: str, reason: str | None, now_us: int) -> None:
        with connect_v2(self.config.database) as connection:
            connection.execute(
                "UPDATE runtime_state SET lifecycle=?, reason=?, process_heartbeat_at_us=? "
                "WHERE run_id=?",
                (lifecycle, reason, now_us, self.config.run_id),
            )

    def _open_gap(
        self,
        connection: sqlite3.Connection,
        subscription_id: str,
        started_at_us: int,
        reason: str,
        continuity_required: bool,
    ) -> None:
        self._open_gap_for_run(
            connection,
            self.config.run_id,
            subscription_id,
            started_at_us,
            reason,
            continuity_required,
        )

    @staticmethod
    def _open_gap_for_run(
        connection: sqlite3.Connection,
        run_id: str,
        subscription_id: str,
        started_at_us: int,
        reason: str,
        continuity_required: bool,
    ) -> None:
        gap_id = hashlib.sha256(
            f"{run_id}|{subscription_id}|{reason}|{started_at_us}".encode()
        ).hexdigest()
        connection.execute(
            "INSERT OR IGNORE INTO gaps(gap_id, run_id, subscription_id, started_at_us, "
            "reason, data_loss_possible, continuity_required) VALUES (?, ?, ?, ?, ?, 1, ?)",
            (
                gap_id,
                run_id,
                subscription_id,
                started_at_us,
                reason,
                int(continuity_required),
            ),
        )
