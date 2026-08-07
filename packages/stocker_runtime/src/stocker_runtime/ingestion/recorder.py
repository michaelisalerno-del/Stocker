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
from stocker_runtime.ideas.discovery import (
    DiscoveredPlugin,
    aggregate_requirements,
    discover_plugins,
    load_idea_configs,
)
from stocker_runtime.ideas.runner import IdeaRunner
from stocker_runtime.ingestion.ibkr_market_data import MarketDataAdapter, MarketDataStatus
from stocker_runtime.ingestion.inbox import (
    AdmissionResult,
    CallbackFence,
    CallbackInbox,
    CallbackTimestampOrderingLoss,
    InboxAdmissionError,
    MarketDataCallback,
    NormalizationError,
    WriterAuthority,
)
from stocker_runtime.storage import RetentionManager, StorageCapState, connect_v2


class DuplicateWriterError(RuntimeError):
    """Another recorder owns a fresh authoritative-writer lease."""


class RecorderFatalError(RuntimeError):
    """The recorder entered a persisted fail-stop state."""


class AuthoritativeLeaseLost(RecorderFatalError):
    """This recorder object no longer owns the persisted authoritative generation."""


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
    idea_config: Path | None = None

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
        "set_status_callback",
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
        self._ideas: tuple[DiscoveredPlugin, ...] = (
            ()
            if config.idea_config is None
            else discover_plugins(load_idea_configs(config.idea_config))
        )
        self.idea_requirements = aggregate_requirements(self._ideas)
        self._idea_runner: IdeaRunner | None = None

    def _authority(self) -> WriterAuthority:
        state = self._authority_state()
        return WriterAuthority(
            state.run_id,
            state.recorder_generation,
            self.config.owner_id,
        )

    def _verify_owned(self, connection: sqlite3.Connection) -> None:
        try:
            CallbackInbox.verify_writer(connection, self._authority())
        except InboxAdmissionError as error:
            raise AuthoritativeLeaseLost(str(error)) from error

    def _check_owned(self) -> None:
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            connection.commit()
        finally:
            connection.close()

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
        available = {(item.instrument_id, item.feed_kind) for item in subscriptions}
        missing_idea_requirements = tuple(
            (item.instrument_id, item.feed_kind)
            for item in self.idea_requirements
            if (item.instrument_id, item.feed_kind) not in available
        )
        if missing_idea_requirements:
            raise RecorderFatalError(
                f"configured idea requirements lack subscriptions: {missing_idea_requirements}"
            )
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute("SELECT 1 FROM runs WHERE status='fatal' LIMIT 1").fetchone()
                is not None
            ):
                raise RecorderFatalError(
                    "persisted global fatal state requires explicit future operator recovery"
                )
            active_rows = tuple(
                connection.execute(
                    "SELECT state.run_id, state.recorder_generation, "
                    "state.process_heartbeat_at_us, run.status "
                    "FROM runtime_state state JOIN runs run ON run.run_id = state.run_id "
                    "WHERE state.lifecycle IN "
                    "('starting','recovering','connecting','running','degraded') "
                    "ORDER BY state.run_id"
                )
            )
            if len(active_rows) > 1:
                raise DuplicateWriterError("multiple authoritative recorder states are active")
            active = None if not active_rows else active_rows[0]
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
                    int(active["recorder_generation"]),
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
                "process_heartbeat_at_us, connection_state, connection_generation, "
                "inbox_nonterminal_count) VALUES (?, ?, 'recovering', NULL, ?, "
                "'disconnected', ?, "
                "(SELECT count(*) FROM callback_inbox WHERE lifecycle IN ('pending','leased'))) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "recorder_generation=excluded.recorder_generation, "
                "lifecycle=excluded.lifecycle, reason=NULL, connection_state='disconnected', "
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
        self._idea_runner = IdeaRunner(self.config.database, self._ideas)
        self._idea_runner.deactivate_unconfigured(run_id=self.config.run_id, now_us=now_us)
        for plugin in self._ideas:
            self._idea_runner.activate(
                run_id=self.config.run_id,
                plugin=plugin,
                activated_at_us=now_us,
            )
        self.inbox.reclaim_expired_leases(now_us=now_us, authority=self._authority())
        self.drain(now_us=now_us)
        self.maintain(now_us=now_us)
        cast(
            Callable[[Callable[[CallbackFence, MarketDataCallback], AdmissionResult]], None],
            self.adapter.set_callback,
        )(self.receive)
        self.adapter.set_disconnect_callback(
            lambda disconnected_at_us: self.disconnected(now_us=disconnected_at_us)
        )
        self.adapter.set_status_callback(self.market_data_status)
        self._connect_subscriptions(now_us=now_us)
        return self.state

    def _close_stale_writer(
        self,
        connection: sqlite3.Connection,
        stale_run_id: str,
        stale_generation: int,
        *,
        now_us: int,
        gap_start_us: int,
        continuing_same_run: bool,
    ) -> None:
        rows = tuple(
            connection.execute(
                "SELECT subscription_id, lifecycle, continuity_required, optional "
                "FROM subscriptions WHERE run_id=? AND recorder_generation=? "
                "AND lifecycle!='closed'",
                (stale_run_id, stale_generation),
            )
        )
        for row in rows:
            # An optional paused feed is intentionally out of service and already has
            # its causal gap; every other nonterminal state has restart uncertainty.
            if str(row["lifecycle"]) == "paused" and bool(row["optional"]):
                continue
            self._open_gap_for_run(
                connection,
                stale_run_id,
                str(row["subscription_id"]),
                gap_start_us,
                "UNCLEAN_RECORDER_RESTART",
                bool(row["continuity_required"]),
            )
        connection.execute(
            "UPDATE subscriptions SET lifecycle='closed', closed_at_us=? WHERE run_id=? "
            "AND recorder_generation=? AND lifecycle!='closed'",
            (now_us, stale_run_id, stale_generation),
        )
        connection.execute(
            "UPDATE recorder_generations SET ended_at_us=?, clean_stop=0, "
            "termination_code='UNCLEAN_RESTART' WHERE run_id=? AND generation=? "
            "AND ended_at_us IS NULL",
            (now_us, stale_run_id, stale_generation),
        )
        connection.execute(
            "UPDATE runtime_state SET lifecycle='stopped', reason='UNCLEAN_RESTART', "
            "connection_state='disconnected', process_heartbeat_at_us=? WHERE run_id=? "
            "AND recorder_generation=?",
            (now_us, stale_run_id, stale_generation),
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
                "continuity_required, optional, requirements_hash, opened_at_us) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'connecting', ?, ?, ?, ?)",
                (
                    subscription_id,
                    self.config.run_id,
                    generation,
                    connection_generation,
                    spec.instrument_id,
                    spec.feed_kind,
                    spec.request_id,
                    int(spec.continuity_required),
                    int(spec.optional),
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

    def _connect_subscriptions(self, *, now_us: int) -> None:
        """Expose connected/active state only after each external action succeeds."""

        self._check_owned()
        self._begin_connecting(now_us)
        try:
            self.adapter.connect()
        except Exception as error:
            try:
                self._persist_connection_failure(
                    now_us=now_us,
                    code="IBKR_CONNECT_FAILED",
                    details=type(error).__name__,
                )
            finally:
                with suppress(Exception):
                    self.adapter.disconnect()
            return
        try:
            self._check_owned()
        except AuthoritativeLeaseLost:
            with suppress(Exception):
                self.adapter.disconnect()
            raise
        by_request = {spec.request_id: spec for spec in self._subscriptions}
        for fence in self._authority_state().fences:
            if fence.request_id is None:
                continue
            with connect_v2(self.config.database) as connection:
                lifecycle = connection.execute(
                    "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
                    (fence.subscription_id,),
                ).fetchone()
            if lifecycle is None or str(lifecycle[0]) == "paused":
                continue
            spec = by_request[fence.request_id]
            try:
                self._check_owned()
                self.adapter.subscribe(fence)
                self._check_owned()
            except AuthoritativeLeaseLost:
                with suppress(Exception):
                    self.adapter.disconnect()
                raise
            except Exception as error:
                try:
                    self._persist_subscription_failure(
                        fence,
                        spec,
                        now_us=now_us,
                        code="IBKR_SUBSCRIBE_FAILED",
                        details=type(error).__name__,
                    )
                except Exception:
                    with suppress(Exception):
                        self.adapter.disconnect()
                    raise
                if not spec.optional:
                    try:
                        self._persist_connection_failure(
                            now_us=now_us,
                            code="IBKR_SUBSCRIBE_FAILED",
                            details="required subscription aborted remaining requests",
                        )
                    finally:
                        with suppress(Exception):
                            self.adapter.disconnect()
                    return
                continue
            connection = connect_v2(self.config.database)
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._verify_owned(connection)
                unresolved_farm = connection.execute(
                    "SELECT 1 FROM gaps WHERE run_id=? AND subscription_id=? "
                    "AND reason LIKE 'IBKR_FARM_%' AND resolved_at_us IS NULL LIMIT 1",
                    (self.config.run_id, fence.subscription_id),
                ).fetchone()
                target_lifecycle = "degraded" if unresolved_farm is not None else "active"
                cursor = connection.execute(
                    "UPDATE subscriptions SET lifecycle=?, closed_at_us=NULL "
                    "WHERE subscription_id=? AND lifecycle='connecting'",
                    (target_lifecycle, fence.subscription_id),
                )
                if cursor.rowcount != 1:
                    current = connection.execute(
                        "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
                        (fence.subscription_id,),
                    ).fetchone()
                    if current is None or str(current["lifecycle"]) not in {
                        "active",
                        "degraded",
                        "disconnected",
                        "paused",
                    }:
                        raise RecorderFatalError("subscription activation state changed")
                connection.commit()
            finally:
                connection.close()
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            state = self._authority_state()
            runtime = connection.execute(
                "SELECT lifecycle, reason, connection_state FROM runtime_state "
                "WHERE run_id=? AND recorder_generation=?",
                (self.config.run_id, state.recorder_generation),
            ).fetchone()
            if runtime is not None and str(runtime["connection_state"]) == "connecting":
                connection.execute(
                    "UPDATE subscriptions SET lifecycle='degraded' WHERE run_id=? "
                    "AND recorder_generation=? AND connection_generation=? "
                    "AND lifecycle IN ('connecting','active') AND EXISTS "
                    "(SELECT 1 FROM gaps gap WHERE gap.run_id=subscriptions.run_id "
                    "AND gap.subscription_id=subscriptions.subscription_id "
                    "AND gap.reason LIKE 'IBKR_FARM_%' AND gap.resolved_at_us IS NULL)",
                    (
                        self.config.run_id,
                        state.recorder_generation,
                        state.connection_generation,
                    ),
                )
                connection.execute(
                    "UPDATE subscriptions SET lifecycle='active', closed_at_us=NULL "
                    "WHERE run_id=? AND recorder_generation=? AND connection_generation=? "
                    "AND (lifecycle='connecting' OR (lifecycle='degraded' AND EXISTS "
                    "(SELECT 1 FROM gaps gap WHERE gap.run_id=subscriptions.run_id "
                    "AND gap.subscription_id=subscriptions.subscription_id "
                    "AND gap.reason LIKE 'IBKR_FARM_%') AND NOT EXISTS "
                    "(SELECT 1 FROM gaps gap WHERE gap.run_id=subscriptions.run_id "
                    "AND gap.subscription_id=subscriptions.subscription_id "
                    "AND gap.reason LIKE 'IBKR_FARM_%' AND gap.resolved_at_us IS NULL)))",
                    (
                        self.config.run_id,
                        state.recorder_generation,
                        state.connection_generation,
                    ),
                )
                required_incomplete = connection.execute(
                    "SELECT 1 FROM subscriptions WHERE run_id=? AND recorder_generation=? "
                    "AND connection_generation=? AND optional=0 AND lifecycle!='active' LIMIT 1",
                    (
                        self.config.run_id,
                        state.recorder_generation,
                        state.connection_generation,
                    ),
                ).fetchone()
                farm_reason = connection.execute(
                    "SELECT gap.reason FROM gaps gap JOIN subscriptions subscription "
                    "ON subscription.subscription_id=gap.subscription_id "
                    "WHERE gap.run_id=? AND subscription.recorder_generation=? "
                    "AND subscription.connection_generation=? AND subscription.optional=0 "
                    "AND gap.reason LIKE 'IBKR_FARM_%' AND gap.resolved_at_us IS NULL "
                    "ORDER BY gap.started_at_us DESC, gap.gap_id LIMIT 1",
                    (
                        self.config.run_id,
                        state.recorder_generation,
                        state.connection_generation,
                    ),
                ).fetchone()
                lifecycle = "running" if required_incomplete is None else "degraded"
                reason = (
                    None
                    if required_incomplete is None
                    else (
                        str(farm_reason["reason"])
                        if farm_reason is not None
                        else "REQUIRED_SUBSCRIPTION_UNAVAILABLE"
                    )
                )
                connection.execute(
                    "UPDATE runtime_state SET connection_state='connected', "
                    "lifecycle=CASE WHEN lifecycle IN ('recovering','connecting','running') "
                    "THEN ? ELSE lifecycle END, reason=CASE WHEN lifecycle IN "
                    "('recovering','connecting','running') THEN ? ELSE reason END, "
                    "process_heartbeat_at_us=? WHERE run_id=? AND recorder_generation=? "
                    "AND connection_state='connecting'",
                    (
                        lifecycle,
                        reason,
                        now_us,
                        self.config.run_id,
                        state.recorder_generation,
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    def _authority_state(self) -> RecorderState:
        if self.state is None:
            raise RecorderFatalError("recorder is not started")
        return self.state

    def _persist_connection_failure(self, *, now_us: int, code: str, details: str) -> None:
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            state = self._authority_state()
            specs = {spec.request_id: spec for spec in self._subscriptions}
            for fence in state.fences:
                if fence.request_id is None:
                    continue
                lifecycle = connection.execute(
                    "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
                    (fence.subscription_id,),
                ).fetchone()
                if lifecycle is None or str(lifecycle[0]) not in {"connecting", "active"}:
                    continue
                spec = specs[fence.request_id]
                self._open_gap(
                    connection,
                    cast(str, fence.subscription_id),
                    now_us,
                    code,
                    spec.continuity_required,
                )
            connection.execute(
                "UPDATE subscriptions SET lifecycle='disconnected' WHERE run_id=? "
                "AND recorder_generation=? AND connection_generation=? "
                "AND lifecycle IN ('connecting','active')",
                (
                    self.config.run_id,
                    state.recorder_generation,
                    state.connection_generation,
                ),
            )
            connection.execute(
                "UPDATE runtime_state SET connection_state='disconnected', "
                "lifecycle=CASE WHEN lifecycle IN ('recovering','connecting','running') "
                "OR (lifecycle='degraded' AND (reason='IBKR_DISCONNECT' "
                "OR reason LIKE 'IBKR_FARM_%')) THEN 'degraded' ELSE lifecycle END, "
                "reason=CASE WHEN lifecycle IN ('recovering','connecting','running') "
                "OR (lifecycle='degraded' AND (reason='IBKR_DISCONNECT' "
                "OR reason LIKE 'IBKR_FARM_%')) THEN ? ELSE reason END WHERE run_id=? "
                "AND recorder_generation=?",
                (code, self.config.run_id, self._authority_state().recorder_generation),
            )
            self._record_incident(connection, None, now_us, code, details)
            connection.commit()
        finally:
            connection.close()

    def _persist_subscription_failure(
        self,
        fence: CallbackFence,
        spec: SubscriptionSpec,
        *,
        now_us: int,
        code: str,
        details: str,
    ) -> None:
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            affected: tuple[tuple[str, bool], ...]
            if spec.optional:
                connection.execute(
                    "UPDATE subscriptions SET lifecycle='paused' WHERE subscription_id=?",
                    (fence.subscription_id,),
                )
                affected = ((cast(str, fence.subscription_id), spec.continuity_required),)
            else:
                state = self._authority_state()
                continuity = {
                    item.request_id: item.continuity_required for item in self._subscriptions
                }
                affected_by_id = {
                    str(row["subscription_id"]): continuity[int(row["request_id"])]
                    for row in connection.execute(
                        "SELECT subscription_id, request_id FROM subscriptions WHERE run_id=? "
                        "AND recorder_generation=? AND connection_generation=? "
                        "AND lifecycle IN ('connecting','active','degraded')",
                        (
                            self.config.run_id,
                            state.recorder_generation,
                            state.connection_generation,
                        ),
                    )
                }
                affected_by_id.setdefault(
                    cast(str, fence.subscription_id), spec.continuity_required
                )
                affected = tuple(affected_by_id.items())
                connection.execute(
                    "UPDATE subscriptions SET lifecycle='disconnected' WHERE run_id=? "
                    "AND recorder_generation=? AND connection_generation=? "
                    "AND lifecycle IN ('connecting','active','degraded')",
                    (
                        self.config.run_id,
                        state.recorder_generation,
                        state.connection_generation,
                    ),
                )
            for subscription_id, continuity_required in affected:
                self._open_gap(
                    connection,
                    subscription_id,
                    now_us,
                    code,
                    continuity_required,
                )
            self._record_incident(connection, fence.subscription_id, now_us, code, details)
            if not spec.optional:
                connection.execute(
                    "UPDATE runtime_state SET connection_state='disconnected', "
                    "lifecycle=CASE WHEN lifecycle IN ('recovering','connecting','running') "
                    "OR (lifecycle='degraded' AND reason LIKE 'IBKR_FARM_%') "
                    "THEN 'degraded' ELSE lifecycle END, reason=CASE WHEN lifecycle IN "
                    "('recovering','connecting','running') OR (lifecycle='degraded' "
                    "AND reason LIKE 'IBKR_FARM_%') THEN ? ELSE reason END WHERE run_id=?",
                    (code, self.config.run_id),
                )
            connection.commit()
        finally:
            connection.close()

    def _record_incident(
        self,
        connection: sqlite3.Connection,
        subscription_id: str | None,
        now_us: int,
        code: str,
        details: str,
    ) -> None:
        incident_id = hashlib.sha256(
            f"{self.config.run_id}|{subscription_id}|{code}|{now_us}".encode()
        ).hexdigest()
        details_json = canonical_json_bytes(cast(JsonValue, {"error": details})).decode()
        connection.execute(
            "INSERT OR IGNORE INTO incidents(incident_id, run_id, scope, severity, code, "
            "subscription_id, opened_at_us, details_json) "
            "VALUES (?, ?, 'market_data', 'degraded', ?, ?, ?, ?)",
            (
                incident_id,
                self.config.run_id,
                code,
                subscription_id,
                now_us,
                details_json,
            ),
        )

    def receive(self, fence: CallbackFence, callback: MarketDataCallback) -> AdmissionResult:
        """External callback boundary: durable admission completes before return."""

        try:
            self._check_owned()
            return self.inbox.admit(fence, callback)
        except AuthoritativeLeaseLost:
            self._cleanup_private_adapter()
            raise
        except InboxAdmissionError:
            try:
                self._fatal("CALLBACK_ADMISSION_FAILED", callback.received_at_us)
            except AuthoritativeLeaseLost:
                self._cleanup_private_adapter()
            raise

    def drain(self, *, now_us: int, limit: int = 256) -> int:
        """Recover and process one bounded callback batch without blocking on poison."""

        authority = self._authority()
        try:
            processed = 0
            for leased in self.inbox.lease_pending(
                self.config.owner_id,
                now_us=now_us,
                lease_us=self.config.callback_lease_us,
                limit=limit,
                authority=authority,
            ):
                try:
                    result = self.inbox.project(leased, authority=authority)
                except NormalizationError:
                    self.inbox.fail(
                        leased,
                        "MALFORMED_CALLBACK",
                        failed_at_us=now_us,
                        authority=authority,
                    )
                    continue
                self.inbox.acknowledge(
                    leased,
                    result.event_id,
                    acknowledged_at_us=now_us,
                    authority=authority,
                )
                processed += 1
            self.inbox.create_pending_receipts(
                created_at_us=now_us,
                limit=limit,
                authority=authority,
            )
            self._heartbeat(now_us)
            if self._idea_runner is not None:
                self._idea_runner.run_once(now_us=now_us)
            return processed
        except CallbackTimestampOrderingLoss as error:
            self._fatal("CALLBACK_TIMESTAMP_ORDERING_LOSS", now_us)
            raise RecorderFatalError("callback timestamp ordering loss") from error
        except AuthoritativeLeaseLost:
            raise
        except Exception as error:
            self._fatal("POST_ADMISSION_PRESERVATION_FAILED", now_us)
            raise RecorderFatalError("post-admission preservation failed") from error

    def _heartbeat(self, now_us: int) -> None:
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            connection.execute(
                "UPDATE runtime_state SET process_heartbeat_at_us=? WHERE run_id=? "
                "AND recorder_generation=?",
                (now_us, self.config.run_id, self.state.recorder_generation if self.state else -1),
            )
            connection.commit()
        finally:
            connection.close()

    def mark_stale(self, *, now_us: int, market_data_expected: bool = True) -> int:
        """Open per-subscription gaps; optional staleness never blocks required feeds."""

        if self.state is None:
            raise RecorderFatalError("recorder is not started")
        if not market_data_expected:
            return 0
        opened = 0
        by_request = {spec.request_id: spec for spec in self._subscriptions}
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
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
            connection.commit()
            return opened
        finally:
            connection.close()

    def disconnected(self, *, now_us: int) -> None:
        """Treat a temporary socket loss as recoverable degraded state."""

        self._check_owned()
        state = self._authority_state()
        required = {spec.request_id: spec.continuity_required for spec in self._subscriptions}
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            disconnected = tuple(
                connection.execute(
                    "SELECT subscription_id, request_id FROM subscriptions WHERE run_id=? "
                    "AND recorder_generation=? AND connection_generation=? "
                    "AND lifecycle IN ('connecting','active','degraded')",
                    (
                        self.config.run_id,
                        state.recorder_generation,
                        state.connection_generation,
                    ),
                )
            )
            for row in disconnected:
                self._open_gap(
                    connection,
                    str(row["subscription_id"]),
                    now_us,
                    "IBKR_DISCONNECT",
                    required[int(row["request_id"])],
                )
            connection.execute(
                "UPDATE subscriptions SET lifecycle='disconnected' WHERE run_id=? "
                "AND recorder_generation=? AND connection_generation=? "
                "AND lifecycle IN ('connecting','active','degraded')",
                (
                    self.config.run_id,
                    state.recorder_generation,
                    state.connection_generation,
                ),
            )
            connection.execute(
                "UPDATE runtime_state SET connection_state='disconnected', "
                "lifecycle=CASE WHEN lifecycle IN ('recovering','connecting','running') "
                "OR (lifecycle='degraded' AND reason LIKE 'IBKR_FARM_%') "
                "THEN 'degraded' ELSE lifecycle END, reason=CASE WHEN lifecycle IN "
                "('recovering','connecting','running') OR (lifecycle='degraded' "
                "AND reason LIKE 'IBKR_FARM_%') THEN 'IBKR_DISCONNECT' ELSE reason END, "
                "process_heartbeat_at_us=? WHERE run_id=? AND recorder_generation=?",
                (now_us, self.config.run_id, state.recorder_generation),
            )
            connection.commit()
        finally:
            connection.close()
        self.adapter.disconnect()

    def market_data_status(self, status: MarketDataStatus) -> None:
        """Persist a typed official status without broadening the broker surface."""

        if status.kind == "temporary_disconnect" and status.request_id is None:
            self.disconnected(now_us=status.received_at_us)
            return
        if status.kind in {"farm_degraded", "farm_recovered"}:
            self._market_data_farm_status(status)
            return
        self._check_owned()
        state = self._authority_state()
        by_request = {fence.request_id: fence for fence in state.fences}
        specs = {spec.request_id: spec for spec in self._subscriptions}
        fence = None if status.request_id is None else by_request.get(status.request_id)
        spec = None if status.request_id is None else specs.get(status.request_id)
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            if status.kind == "recovered":
                if fence is not None:
                    connection.execute(
                        "UPDATE gaps SET ended_at_us=?, resolved_at_us=? WHERE run_id=? "
                        "AND subscription_id=? AND started_at_us<=? "
                        "AND resolved_at_us IS NULL",
                        (
                            status.received_at_us,
                            status.received_at_us,
                            self.config.run_id,
                            fence.subscription_id,
                            status.received_at_us,
                        ),
                    )
                self._record_incident(
                    connection,
                    None if fence is None else cast(str, fence.subscription_id),
                    status.received_at_us,
                    f"IBKR_STATUS_{status.code}_RECOVERED",
                    status.message,
                )
            else:
                code = f"IBKR_STATUS_{status.code}_{status.kind.upper()}"
                if fence is not None and spec is not None:
                    lifecycle = "paused" if spec.optional else "disconnected"
                    connection.execute(
                        "UPDATE subscriptions SET lifecycle=? WHERE subscription_id=?",
                        (lifecycle, fence.subscription_id),
                    )
                    self._open_gap(
                        connection,
                        cast(str, fence.subscription_id),
                        status.received_at_us,
                        code,
                        spec.continuity_required,
                    )
                    if not spec.optional:
                        connection.execute(
                            "UPDATE runtime_state SET lifecycle=CASE WHEN lifecycle IN "
                            "('recovering','connecting','running') OR (lifecycle='degraded' "
                            "AND reason LIKE 'IBKR_FARM_%') THEN 'degraded' ELSE lifecycle END, "
                            "reason=CASE WHEN lifecycle IN ('recovering','connecting','running') "
                            "OR (lifecycle='degraded' AND reason LIKE 'IBKR_FARM_%') "
                            "THEN ? ELSE reason END WHERE run_id=? AND recorder_generation=?",
                            (code, self.config.run_id, state.recorder_generation),
                        )
                self._record_incident(
                    connection,
                    None if fence is None else cast(str, fence.subscription_id),
                    status.received_at_us,
                    code,
                    status.message,
                )
            connection.commit()
        finally:
            connection.close()

    def _market_data_farm_status(self, status: MarketDataStatus) -> None:
        """Scope farm health to affected feeds while the shared socket remains connected."""

        self._check_owned()
        state = self._authority_state()
        specs = {spec.request_id: spec for spec in self._subscriptions}
        targets = tuple(
            fence
            for fence in state.fences
            if fence.request_id is not None
            and (
                fence.request_id == status.request_id
                if status.request_id is not None
                else specs[fence.request_id].feed_kind in status.affected_feed_kinds
            )
        )
        if not targets:
            return
        recovery_source_code = {2104: 2103, 2106: 2105}.get(status.code)
        if status.kind == "farm_recovered" and recovery_source_code is None:
            return
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            if status.kind == "farm_degraded":
                code = f"IBKR_FARM_{status.code}_DEGRADED"
                runtime = connection.execute(
                    "SELECT lifecycle, reason, connection_state FROM runtime_state WHERE run_id=? "
                    "AND recorder_generation=?",
                    (self.config.run_id, state.recorder_generation),
                ).fetchone()
                state_mutation_allowed = (
                    runtime is not None and str(runtime["connection_state"]) == "connected"
                )
                affected_connection_state = False
                for fence in targets:
                    spec = specs[cast(int, fence.request_id)]
                    lifecycle = connection.execute(
                        "SELECT lifecycle FROM subscriptions WHERE subscription_id=?",
                        (fence.subscription_id,),
                    ).fetchone()
                    if (
                        state_mutation_allowed
                        and lifecycle is not None
                        and str(lifecycle["lifecycle"]) in {"active", "degraded"}
                    ):
                        affected_connection_state = True
                        connection.execute(
                            "UPDATE subscriptions SET lifecycle='degraded' "
                            "WHERE subscription_id=? AND lifecycle='active'",
                            (fence.subscription_id,),
                        )
                    self._open_gap(
                        connection,
                        cast(str, fence.subscription_id),
                        status.received_at_us,
                        code,
                        spec.continuity_required,
                    )
                    self._record_incident(
                        connection,
                        cast(str, fence.subscription_id),
                        status.received_at_us,
                        code,
                        status.message,
                    )
                if affected_connection_state:
                    connection.execute(
                        "UPDATE runtime_state SET lifecycle=CASE WHEN lifecycle IN "
                        "('recovering','connecting','running') OR (lifecycle='degraded' "
                        "AND reason LIKE 'IBKR_FARM_%') THEN 'degraded' ELSE lifecycle END, "
                        "reason=CASE WHEN lifecycle IN ('recovering','connecting','running') "
                        "OR (lifecycle='degraded' AND reason LIKE 'IBKR_FARM_%') "
                        "THEN ? ELSE reason END WHERE run_id=? AND recorder_generation=?",
                        (code, self.config.run_id, state.recorder_generation),
                    )
            else:
                matching_reason = f"IBKR_FARM_{recovery_source_code}_DEGRADED"
                runtime = connection.execute(
                    "SELECT lifecycle, reason, connection_state FROM runtime_state WHERE run_id=? "
                    "AND recorder_generation=?",
                    (self.config.run_id, state.recorder_generation),
                ).fetchone()
                activation_allowed = (
                    runtime is not None and str(runtime["connection_state"]) == "connected"
                )
                for fence in targets:
                    connection.execute(
                        "UPDATE gaps SET ended_at_us=?, resolved_at_us=? WHERE run_id=? "
                        "AND subscription_id=? AND reason=? "
                        "AND started_at_us<=? "
                        "AND resolved_at_us IS NULL",
                        (
                            status.received_at_us,
                            status.received_at_us,
                            self.config.run_id,
                            fence.subscription_id,
                            matching_reason,
                            status.received_at_us,
                        ),
                    )
                    connection.execute(
                        "UPDATE incidents SET resolved_at_us=? WHERE run_id=? "
                        "AND subscription_id=? AND code=? "
                        "AND opened_at_us<=? "
                        "AND resolved_at_us IS NULL",
                        (
                            status.received_at_us,
                            self.config.run_id,
                            fence.subscription_id,
                            matching_reason,
                            status.received_at_us,
                        ),
                    )
                    unresolved_for_subscription = connection.execute(
                        "SELECT 1 FROM gaps WHERE run_id=? AND subscription_id=? "
                        "AND reason LIKE 'IBKR_FARM_%' AND resolved_at_us IS NULL LIMIT 1",
                        (self.config.run_id, fence.subscription_id),
                    ).fetchone()
                    if activation_allowed and unresolved_for_subscription is None:
                        connection.execute(
                            "UPDATE subscriptions SET lifecycle='active' "
                            "WHERE subscription_id=? AND lifecycle='degraded'",
                            (fence.subscription_id,),
                        )
                required_degraded = connection.execute(
                    "SELECT 1 FROM subscriptions WHERE run_id=? AND recorder_generation=? "
                    "AND connection_generation=? AND optional=0 AND lifecycle!='active' LIMIT 1",
                    (
                        self.config.run_id,
                        state.recorder_generation,
                        state.connection_generation,
                    ),
                ).fetchone()
                if (
                    activation_allowed
                    and runtime is not None
                    and str(runtime["reason"]) == matching_reason
                ):
                    unresolved_farm = connection.execute(
                        "SELECT gap.reason FROM gaps gap JOIN subscriptions subscription "
                        "ON subscription.subscription_id=gap.subscription_id "
                        "WHERE gap.run_id=? AND subscription.recorder_generation=? "
                        "AND subscription.connection_generation=? "
                        "AND gap.reason LIKE 'IBKR_FARM_%' AND gap.resolved_at_us IS NULL "
                        "ORDER BY gap.started_at_us DESC, gap.gap_id LIMIT 1",
                        (
                            self.config.run_id,
                            state.recorder_generation,
                            state.connection_generation,
                        ),
                    ).fetchone()
                    if unresolved_farm is not None:
                        connection.execute(
                            "UPDATE runtime_state SET lifecycle='degraded', reason=? "
                            "WHERE run_id=? AND recorder_generation=?",
                            (
                                str(unresolved_farm["reason"]),
                                self.config.run_id,
                                state.recorder_generation,
                            ),
                        )
                    elif required_degraded is None:
                        connection.execute(
                            "UPDATE runtime_state SET lifecycle='running', reason=NULL "
                            "WHERE run_id=? AND recorder_generation=?",
                            (self.config.run_id, state.recorder_generation),
                        )
            connection.commit()
        finally:
            connection.close()

    def reconnect(self, *, now_us: int) -> RecorderState:
        """Fence old requests and reconnect with a new durable socket generation."""

        self._check_owned()
        old = self._authority_state()
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            paused_request_ids = {
                int(row[0])
                for row in connection.execute(
                    "SELECT request_id FROM subscriptions WHERE run_id=? "
                    "AND recorder_generation=? AND lifecycle='paused'",
                    (self.config.run_id, old.recorder_generation),
                )
            }
            connection.execute(
                "UPDATE subscriptions SET lifecycle='closed', closed_at_us=? WHERE run_id=? "
                "AND connection_generation=? AND lifecycle!='closed'",
                (now_us, self.config.run_id, old.connection_generation),
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
                if fence.request_id in paused_request_ids:
                    connection.execute(
                        "UPDATE subscriptions SET lifecycle='paused' WHERE subscription_id=?",
                        (fence.subscription_id,),
                    )
                self._open_gap(
                    connection,
                    cast(str, fence.subscription_id),
                    now_us,
                    "RECONNECT_UNCERTAINTY",
                    spec.continuity_required,
                )
            connection.execute(
                "UPDATE runtime_state SET connection_generation=?, "
                "connection_state='disconnected', lifecycle=CASE WHEN lifecycle='degraded' "
                "AND reason='IBKR_DISCONNECT' THEN 'recovering' ELSE lifecycle END, "
                "reason=CASE WHEN lifecycle='degraded' AND reason='IBKR_DISCONNECT' "
                "THEN NULL ELSE reason END, process_heartbeat_at_us=? WHERE run_id=?",
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
        self._connect_subscriptions(now_us=now_us)
        return self.state

    def stop(self, *, now_us: int) -> None:
        """Close subscriptions and the writer generation without changing mode."""

        if self.state is None:
            if self._idea_runner is not None:
                self._idea_runner.close()
                self._idea_runner = None
            return
        self._check_owned()
        with suppress(Exception):
            self.adapter.disconnect()
        self._check_owned()
        state = self._authority_state()
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            connection.execute(
                "UPDATE subscriptions SET lifecycle='closed', closed_at_us=? WHERE run_id=? "
                "AND recorder_generation=? AND lifecycle!='closed'",
                (now_us, self.config.run_id, state.recorder_generation),
            )
            connection.execute(
                "UPDATE recorder_generations SET ended_at_us=?, clean_stop=1, "
                "termination_code='CLEAN_STOP' WHERE run_id=? AND generation=?",
                (
                    now_us,
                    self.config.run_id,
                    state.recorder_generation,
                ),
            )
            connection.execute(
                "UPDATE runtime_state SET lifecycle='stopped', reason=NULL, "
                "connection_state='disconnected', process_heartbeat_at_us=? WHERE run_id=?",
                (now_us, self.config.run_id),
            )
            connection.execute(
                "UPDATE runs SET status='stopped', ended_at_us=? WHERE run_id=?",
                (now_us, self.config.run_id),
            )
            connection.commit()
        finally:
            connection.close()
        self.state = None
        if self._idea_runner is not None:
            self._idea_runner.close()
            self._idea_runner = None

    def _pause_optional(self, now_us: int) -> None:
        if self.state is None:
            return
        self._check_owned()
        optional = {spec.request_id: spec for spec in self._subscriptions if spec.optional}
        for fence in self._authority_state().fences:
            if fence.request_id is None:
                continue
            spec = optional.get(fence.request_id)
            if spec is None:
                continue
            self._check_owned()
            self.adapter.cancel(fence.request_id)
            self._check_owned()
            connection = connect_v2(self.config.database)
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._verify_owned(connection)
                connection.execute(
                    "UPDATE subscriptions SET lifecycle='paused', closed_at_us=? "
                    "WHERE subscription_id=? AND lifecycle!='closed'",
                    (now_us, fence.subscription_id),
                )
                self._open_gap(
                    connection,
                    cast(str, fence.subscription_id),
                    now_us,
                    "STORAGE_DEGRADED_OPTIONAL_PAUSED",
                    spec.continuity_required,
                )
                connection.commit()
            finally:
                connection.close()

    def _fatal(self, code: str, now_us: int) -> None:
        try:
            connection = connect_v2(self.config.database)
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._verify_owned(connection)
                CallbackInbox._record_fatal(
                    connection,
                    self.config.run_id,
                    now_us,
                    code,
                    authority=self._authority(),
                )
                connection.commit()
            finally:
                connection.close()
        except AuthoritativeLeaseLost:
            raise
        except (OSError, sqlite3.Error) as error:
            self._cleanup_private_adapter()
            raise RecorderFatalError("fatal state could not be persisted") from error
        self._cleanup_private_adapter()

    def _cleanup_private_adapter(self) -> None:
        state = self.state
        if state is None:
            with suppress(Exception):
                self.adapter.disconnect()
            return
        for fence in state.fences:
            if fence.request_id is not None:
                with suppress(Exception):
                    self.adapter.cancel(fence.request_id)
        with suppress(Exception):
            self.adapter.disconnect()

    def maintain(self, *, now_us: int) -> StorageCapState:
        """Consume one bounded Phase 2 retention result and apply recorder reactions."""

        authority = self._authority()
        self._check_owned()
        try:
            result = RetentionManager(self.config.database).run(
                now_us=now_us,
                precondition=lambda connection: CallbackInbox.verify_writer(connection, authority),
            )
        except AuthoritativeLeaseLost:
            raise
        except InboxAdmissionError as error:
            raise AuthoritativeLeaseLost(str(error)) from error
        except Exception as error:
            self._fatal("RETENTION_INVARIANT_FAILED", now_us)
            raise RecorderFatalError("retention invariant failed") from error
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            connection.execute(
                "UPDATE runtime_state SET database_bytes=?, wal_bytes=? WHERE run_id=? "
                "AND recorder_generation=?",
                (
                    result.database_bytes,
                    result.wal_bytes,
                    self.config.run_id,
                    authority.recorder_generation,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        if not result.admission_allowed:
            self._fatal(result.required_action or "STORAGE_CAP_FATAL", now_us)
            raise RecorderFatalError(result.required_action or "storage cap closed admission")
        if not result.optional_feeds_allowed:
            self._pause_optional(now_us)
            self._set_lifecycle("degraded", result.required_action, now_us)
        return result.cap_state

    def _set_lifecycle(self, lifecycle: str, reason: str | None, now_us: int) -> None:
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            connection.execute(
                "UPDATE runtime_state SET lifecycle=?, reason=?, process_heartbeat_at_us=? "
                "WHERE run_id=? AND recorder_generation=?",
                (
                    lifecycle,
                    reason,
                    now_us,
                    self.config.run_id,
                    self._authority_state().recorder_generation,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def _begin_connecting(self, now_us: int) -> None:
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            connection.execute(
                "UPDATE runtime_state SET connection_state='connecting', "
                "lifecycle=CASE WHEN lifecycle='recovering' THEN 'connecting' ELSE lifecycle END, "
                "reason=CASE WHEN lifecycle='recovering' THEN NULL ELSE reason END, "
                "process_heartbeat_at_us=? WHERE run_id=? AND recorder_generation=?",
                (
                    now_us,
                    self.config.run_id,
                    self._authority_state().recorder_generation,
                ),
            )
            connection.commit()
        finally:
            connection.close()

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
