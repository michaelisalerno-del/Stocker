"""Fail-closed configuration and lifecycle for the V2 prospective recorder."""

from __future__ import annotations

import hashlib
import ipaddress
import math
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stocker_runtime.domain import JsonValue, canonical_json_bytes
from stocker_runtime.ideas.contract import DiscoveryReceipt, MarketDataInterest
from stocker_runtime.ideas.discovery import (
    DiscoveredPlugin,
    IdeaDiscoveryError,
    aggregate_instruments,
    aggregate_requirements,
    discover_plugins,
    load_idea_configs,
)
from stocker_runtime.ideas.runner import IdeaRunner
from stocker_runtime.ingestion.dynamic_market_data import (
    InstrumentResolver,
    InterestResolutionRequest,
    MarketDataCapacity,
    MarketDataDemand,
    MarketDataPlan,
    OptionDiscoveryBackend,
    SubscriptionApplyPlan,
    SubscriptionBackend,
    SubscriptionController,
    plan_market_data,
)
from stocker_runtime.ingestion.ibkr_market_data import (
    IBKRSubscription,
    MarketDataAdapter,
    MarketDataStatus,
)
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
from stocker_runtime.shadow import ShadowEngine
from stocker_runtime.storage import (
    RetentionManager,
    StorageCapState,
    connect_v2,
    read_backup_manifests,
)


class DuplicateWriterError(RuntimeError):
    """Another recorder owns a fresh authoritative-writer lease."""


class RecorderFatalError(RuntimeError):
    """The recorder entered a persisted fail-stop state."""


class AuthoritativeLeaseLost(RecorderFatalError):
    """This recorder object no longer owns the persisted authoritative generation."""


_IDEA_REQUEST_ID_BASE = 1_000_000
_DYNAMIC_REQUEST_ID_BASE = 2_000_000
_DYNAMIC_REQUEST_ID_SPAN = 900_000_000
_MAX_INTERESTS_PER_RECONCILIATION = 4
_CADENCE = re.compile(r"^(?P<count>[1-9][0-9]*)(?P<unit>us|ms|s|m|h)$")
_SECURITY_TYPES = {
    "stock": "STK",
    "option": "OPT",
    "future": "FUT",
    "forex": "CASH",
    "index": "IND",
}


@dataclass(frozen=True)
class InstrumentSpec:
    instrument_id: str
    ibkr_con_id: int | None
    kind: str
    symbol: str
    exchange: str
    currency: str
    option_expiry: str | None = None
    option_strike: str | None = None
    option_right: str | None = None
    option_multiplier: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.instrument_id
            or self.ibkr_con_id is not None
            and self.ibkr_con_id <= 0
            or self.kind not in _SECURITY_TYPES
            or not self.symbol
            or not self.exchange
            or not self.currency
        ):
            raise ValueError("instrument identity is invalid")
        option_fields = (
            self.option_expiry,
            self.option_strike,
            self.option_right,
            self.option_multiplier,
        )
        if self.kind == "option":
            if (
                any(value is None for value in option_fields)
                or self.option_expiry is None
                or len(self.option_expiry) != 8
                or not self.option_expiry.isdigit()
                or self.option_strike is None
                or not math.isfinite(float(self.option_strike))
                or float(self.option_strike) <= 0
                or self.option_right not in {"call", "put"}
                or not self.option_multiplier
            ):
                raise ValueError("option instrument identity is incomplete")
        elif any(value is not None for value in option_fields):
            raise ValueError("non-option instrument contains option identity")


@dataclass(frozen=True)
class SubscriptionSpec:
    name: str
    instrument_id: str
    feed_kind: str
    request_id: int
    continuity_required: bool
    optional: bool
    stale_after_us: int
    snapshot: bool = False

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.instrument_id
            or not self.feed_kind
            or self.request_id < 0
            or self.request_id > 1_499_999_999
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
    mode: Literal["prospective_record", "shadow"]
    host: str
    port: int = Field(ge=1, le=65_535)
    client_id: int = Field(ge=0)
    read_only: Literal[True]
    external_read_only_verified: Literal[True]
    config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    git_commit: str = Field(pattern=r"^[a-f0-9]{7,64}$")
    writer_lease_stale_us: int = Field(default=60_000_000, ge=15_000_000)
    callback_lease_us: int = Field(default=30_000_000, ge=5_000_000)
    market_data_line_limit: int = Field(default=100, ge=1, le=100)
    idea_config: Path | None = None
    backup_directory: Path | None = None

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
        self._base_subscriptions: tuple[SubscriptionSpec, ...] = ()
        self._subscriptions: tuple[SubscriptionSpec, ...] = ()
        self._ideas: tuple[DiscoveredPlugin, ...] = (
            ()
            if config.idea_config is None
            else discover_plugins(load_idea_configs(config.idea_config))
        )
        self.idea_requirements = aggregate_requirements(self._ideas)
        self._idea_runner: IdeaRunner | None = None
        self._shadow_engine: ShadowEngine | None = None
        self._snapshot_handshake_lock = threading.Lock()
        self._adapter_transition_lock = threading.Lock()
        self._subscription_lifecycle_lock = threading.RLock()
        self._starting_dynamic_request_ids: set[int] = set()
        self._pending_dynamic_statuses: list[MarketDataStatus] = []

    @contextmanager
    def _adapter_reset_transition(self) -> Iterator[None]:
        with self._adapter_transition_lock:
            self._check_owned()
            self.adapter.disconnect()
            self._check_owned()
            yield

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
        instruments, subscriptions, generated_idea_subscriptions = self._prepare_inputs(
            instruments, subscriptions
        )
        self._instruments = instruments
        self._base_subscriptions = subscriptions
        self._subscriptions = subscriptions
        self._configure_adapter(
            instruments,
            subscriptions,
            required=generated_idea_subscriptions,
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
                    "git_commit, data_class, status) VALUES (?, ?, 'ibkr', ?, ?, ?, ?, 'running')",
                    (
                        self.config.run_id,
                        self.config.mode,
                        now_us,
                        self.config.config_hash,
                        self.config.git_commit,
                        (
                            "prospective_protected"
                            if self.config.mode == "prospective_record"
                            else "shadow_protected"
                        ),
                    ),
                )
            elif (
                str(run["mode"]) != self.config.mode
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
        self._idea_runner = IdeaRunner(self.config.database, self._ideas, run_id=self.config.run_id)
        self._shadow_engine = (
            ShadowEngine(self.config.database, run_id=self.config.run_id)
            if self.config.mode == "shadow"
            else None
        )
        self._idea_runner.deactivate_unconfigured(run_id=self.config.run_id, now_us=now_us)
        for plugin in self._ideas:
            self._idea_runner.activate(
                run_id=self.config.run_id,
                plugin=plugin,
                activated_at_us=now_us,
            )
        self._restore_dynamic_subscriptions(now_us=now_us)
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

    @staticmethod
    def _cadence_us(cadence: str) -> int:
        match = _CADENCE.fullmatch(cadence)
        if match is None:
            raise RecorderFatalError(f"unsupported idea requirement cadence: {cadence}")
        factor = {
            "us": 1,
            "ms": 1_000,
            "s": 1_000_000,
            "m": 60_000_000,
            "h": 3_600_000_000,
        }[match.group("unit")]
        return int(match.group("count")) * factor

    def _prepare_inputs(
        self,
        instruments: tuple[InstrumentSpec, ...],
        subscriptions: tuple[SubscriptionSpec, ...],
    ) -> tuple[tuple[InstrumentSpec, ...], tuple[SubscriptionSpec, ...], bool]:
        """Merge core-owned idea requirements into the recorder's exact request set."""

        instrument_by_id = {item.instrument_id: item for item in instruments}
        if len(instrument_by_id) != len(instruments):
            raise RecorderFatalError("instrument identities must be unique")
        caller_instrument_ids = frozenset(instrument_by_id)
        for plugin in self._ideas:
            own_instrument_ids = {item.instrument_id for item in plugin.config.instruments}
            missing = sorted(
                {
                    requirement.instrument_id
                    for requirement in plugin.requirements
                    if requirement.instrument_id not in own_instrument_ids
                    and requirement.instrument_id not in caller_instrument_ids
                }
            )
            if missing:
                raise RecorderFatalError(
                    f"configured idea requirement lacks same-entry instrument metadata: {missing}"
                )
        try:
            configured_instruments = aggregate_instruments(self._ideas)
        except IdeaDiscoveryError as error:
            raise RecorderFatalError(str(error)) from error
        for configured in configured_instruments:
            candidate = InstrumentSpec(
                instrument_id=configured.instrument_id,
                ibkr_con_id=configured.ibkr_con_id,
                kind=configured.kind,
                symbol=configured.symbol,
                exchange=configured.exchange,
                currency=configured.currency,
            )
            prior = instrument_by_id.get(candidate.instrument_id)
            if prior is not None and prior != candidate:
                raise RecorderFatalError(
                    f"conflicting instrument metadata for {candidate.instrument_id}"
                )
            instrument_by_id[candidate.instrument_id] = candidate

        subscription_by_key: dict[tuple[str, str], SubscriptionSpec] = {}
        request_ids: set[int] = set()
        names: set[str] = set()
        for subscription in subscriptions:
            key = (subscription.instrument_id, subscription.feed_kind)
            if key in subscription_by_key:
                raise RecorderFatalError(f"duplicate subscription requirement for {key}")
            if subscription.request_id in request_ids or subscription.name in names:
                raise RecorderFatalError(
                    "subscription request identifiers and names must be unique"
                )
            if subscription.instrument_id not in instrument_by_id:
                raise RecorderFatalError(
                    f"subscription lacks instrument metadata: {subscription.instrument_id}"
                )
            subscription_by_key[key] = subscription
            request_ids.add(subscription.request_id)
            names.add(subscription.name)

        generated = False
        next_request_id = _IDEA_REQUEST_ID_BASE
        for requirement in self.idea_requirements:
            key = (requirement.instrument_id, requirement.feed_kind)
            if key in subscription_by_key:
                continue
            if requirement.instrument_id not in instrument_by_id:
                raise RecorderFatalError(
                    "configured idea requirement lacks instrument metadata: "
                    f"{requirement.instrument_id}"
                )
            while next_request_id in request_ids:
                next_request_id += 1
            subscription = SubscriptionSpec(
                name=f"idea:{requirement.instrument_id}:{requirement.feed_kind}",
                instrument_id=requirement.instrument_id,
                feed_kind=requirement.feed_kind,
                request_id=next_request_id,
                continuity_required=requirement.gaps_block,
                optional=not (requirement.gaps_block or requirement.staleness_block),
                stale_after_us=max(15_000_000, 3 * self._cadence_us(requirement.cadence)),
            )
            subscription_by_key[key] = subscription
            request_ids.add(next_request_id)
            names.add(subscription.name)
            next_request_id += 1
            generated = True
        merged_instruments = tuple(instrument_by_id[key] for key in sorted(instrument_by_id))
        # Existing recorder subscriptions retain their caller-owned fence order;
        # generated requirements are appended in aggregate_requirements order.
        merged_subscriptions = tuple(subscription_by_key.values())
        if len(merged_subscriptions) > self.config.market_data_line_limit:
            raise RecorderFatalError(
                "configured market-data subscriptions exceed the explicit line limit"
            )
        return merged_instruments, merged_subscriptions, generated

    def _configure_adapter(
        self,
        instruments: tuple[InstrumentSpec, ...],
        subscriptions: tuple[SubscriptionSpec, ...],
        *,
        required: bool,
    ) -> None:
        configure = getattr(self.adapter, "configure_subscriptions", None)
        if not callable(configure):
            if required:
                raise RecorderFatalError(
                    "market-data adapter cannot accept core-owned idea subscriptions"
                )
            return
        exact = self._exact_subscriptions(instruments, subscriptions)
        try:
            configure(exact)
        except Exception as error:
            raise RecorderFatalError(
                f"market-data adapter rejected core-owned subscriptions: {type(error).__name__}"
            ) from error

    @staticmethod
    def _exact_subscriptions(
        instruments: tuple[InstrumentSpec, ...],
        subscriptions: tuple[SubscriptionSpec, ...],
    ) -> tuple[IBKRSubscription, ...]:
        instrument_by_id = {item.instrument_id: item for item in instruments}
        exact: list[IBKRSubscription] = []
        for subscription in subscriptions:
            instrument = instrument_by_id[subscription.instrument_id]
            if instrument.ibkr_con_id is None:
                raise RecorderFatalError(
                    f"IBKR contract identity is missing for {instrument.instrument_id}"
                )
            security_type = _SECURITY_TYPES.get(instrument.kind.lower())
            if security_type is None:
                raise RecorderFatalError(f"unsupported IBKR instrument kind: {instrument.kind}")
            exact.append(
                IBKRSubscription(
                    request_id=subscription.request_id,
                    con_id=instrument.ibkr_con_id,
                    symbol=instrument.symbol,
                    security_type=security_type,
                    exchange=instrument.exchange,
                    currency=instrument.currency,
                    feed_kind=subscription.feed_kind,
                    snapshot=subscription.snapshot,
                )
            )
        return tuple(exact)

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
                "option_expiry": spec.option_expiry,
                "option_strike": spec.option_strike,
                "option_right": spec.option_right,
                "option_multiplier": spec.option_multiplier,
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
                "kind, symbol, exchange, currency, option_expiry, option_strike, "
                "option_right, option_multiplier) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    spec.instrument_id,
                    identity_hash,
                    spec.ibkr_con_id,
                    spec.kind,
                    spec.symbol,
                    spec.exchange,
                    spec.currency,
                    spec.option_expiry,
                    spec.option_strike,
                    spec.option_right,
                    spec.option_multiplier,
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
            requirements_hash = self._subscription_requirements_hash(spec)
            subscription_id = hashlib.sha256(
                f"{self.config.run_id}|{connection_generation}|{spec.name}".encode()
            ).hexdigest()
            connection.execute(
                "INSERT INTO subscriptions(subscription_id, run_id, recorder_generation, "
                "connection_generation, instrument_id, feed_kind, request_id, lifecycle, "
                "continuity_required, optional, requirements_hash, opened_at_us, snapshot) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'connecting', ?, ?, ?, ?, ?)",
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
                    int(spec.snapshot),
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

    @staticmethod
    def _subscription_requirements_hash(spec: SubscriptionSpec) -> str:
        material = cast(
            JsonValue,
            {
                "name": spec.name,
                "instrument_id": spec.instrument_id,
                "feed_kind": spec.feed_kind,
                "continuity_required": spec.continuity_required,
                "optional": spec.optional,
                "stale_after_us": spec.stale_after_us,
                "snapshot": spec.snapshot,
            },
        )
        return hashlib.sha256(canonical_json_bytes(material)).hexdigest()

    @staticmethod
    def _interest_from_row(row: sqlite3.Row) -> MarketDataInterest:
        return MarketDataInterest.model_validate(
            {
                "interest_key": str(row["interest_key"]),
                "underlying_instrument_id": str(row["underlying_instrument_id"]),
                "asset_kind": str(row["asset_kind"]),
                "minimum_days_to_expiry": int(row["minimum_days_to_expiry"]),
                "maximum_days_to_expiry": int(row["maximum_days_to_expiry"]),
                "option_right": str(row["option_right"]),
                "strike_offset": int(row["strike_offset"]),
                "reference_price": float(row["reference_price"]),
                "feed_kind": str(row["feed_kind"]),
                "cadence": str(row["cadence"]),
                "as_of_at_us": int(row["as_of_at_us"]),
                "expires_at_us": int(row["expires_at_us"]),
                "required": bool(row["required"]),
                "priority": int(row["priority"]),
                "maximum_contracts": int(row["maximum_contracts"]),
                "input_event_id": str(row["input_event_id"]),
            }
        )

    @staticmethod
    def _denied_receipt(
        *,
        interest_id: str,
        interest_key: str,
        instance_id: str,
        reason_code: str,
        completed_at_us: int,
    ) -> DiscoveryReceipt:
        material = cast(
            JsonValue,
            {
                "interest_id": interest_id,
                "interest_key": interest_key,
                "instance_id": instance_id,
                "status": "denied",
                "reason_code": reason_code,
                "completed_at_us": completed_at_us,
            },
        )
        return DiscoveryReceipt(
            receipt_id=hashlib.sha256(canonical_json_bytes(material)).hexdigest(),
            interest_id=interest_id,
            interest_key=interest_key,
            instance_id=instance_id,
            status="denied",
            reason_code=reason_code,
            candidates_inspected=0,
            completed_at_us=completed_at_us,
        )

    @staticmethod
    def _resolved_option_spec(
        receipt: DiscoveryReceipt, underlying: InstrumentSpec
    ) -> InstrumentSpec:
        prefix = "ibkr-option-"
        if (
            receipt.status != "resolved"
            or receipt.instrument_id is None
            or not receipt.instrument_id.startswith(prefix)
            or receipt.expiry is None
            or receipt.strike is None
            or receipt.option_right is None
            or receipt.multiplier is None
        ):
            raise RecorderFatalError("resolved option receipt identity is incomplete")
        con_id_text = receipt.instrument_id.removeprefix(prefix)
        if not con_id_text.isdigit() or int(con_id_text) <= 0:
            raise RecorderFatalError("resolved option contract id is invalid")
        return InstrumentSpec(
            instrument_id=receipt.instrument_id,
            ibkr_con_id=int(con_id_text),
            kind="option",
            symbol=underlying.symbol,
            exchange="SMART",
            currency=underlying.currency,
            option_expiry=receipt.expiry,
            option_strike=format(receipt.strike, ".15g"),
            option_right=receipt.option_right,
            option_multiplier=receipt.multiplier,
        )

    @staticmethod
    def _insert_discovery_receipt(
        connection: sqlite3.Connection,
        receipt: DiscoveryReceipt,
        run_id: str,
    ) -> None:
        connection.execute(
            "INSERT INTO instrument_discovery_receipts(receipt_id, interest_id, run_id, "
            "instance_id, status, reason_code, instrument_id, expiry, strike, option_right, "
            "multiplier, candidates_inspected, completed_at_us) VALUES (?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?)",
            (
                receipt.receipt_id,
                receipt.interest_id,
                run_id,
                receipt.instance_id,
                receipt.status,
                receipt.reason_code,
                receipt.instrument_id,
                receipt.expiry,
                receipt.strike,
                receipt.option_right,
                receipt.multiplier,
                receipt.candidates_inspected,
                receipt.completed_at_us,
            ),
        )

    def _persist_discovery_receipt(
        self,
        *,
        receipt: DiscoveryReceipt,
        option_spec: InstrumentSpec | None,
        now_us: int,
    ) -> None:
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            current = connection.execute(
                "SELECT lifecycle FROM market_data_interests WHERE interest_id=?",
                (receipt.interest_id,),
            ).fetchone()
            if current is None or str(current["lifecycle"]) != "pending":
                connection.rollback()
                return
            if option_spec is not None:
                self._upsert_instruments(connection, (option_spec,))
            self._insert_discovery_receipt(connection, receipt, self.config.run_id)
            lifecycle = "resolved" if receipt.status == "resolved" else "denied"
            connection.execute(
                "UPDATE market_data_interests SET lifecycle=?, reason_code=?, "
                "attempts=CASE WHEN ?='resolved' THEN 0 ELSE attempts END, "
                "next_attempt_at_us=CASE WHEN ?='resolved' THEN 0 ELSE next_attempt_at_us END, "
                "updated_at_us=? "
                "WHERE interest_id=? AND lifecycle='pending'",
                (
                    lifecycle,
                    receipt.reason_code,
                    receipt.status,
                    receipt.status,
                    now_us,
                    receipt.interest_id,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        if option_spec is not None:
            by_id = {item.instrument_id: item for item in self._instruments}
            by_id[option_spec.instrument_id] = option_spec
            self._instruments = tuple(by_id[key] for key in sorted(by_id))

    def _defer_discovery(self, row: sqlite3.Row, *, now_us: int, error: Exception) -> None:
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            current = connection.execute(
                "SELECT lifecycle, attempts, expires_at_us FROM market_data_interests "
                "WHERE interest_id=?",
                (row["interest_id"],),
            ).fetchone()
            if current is None or str(current["lifecycle"]) != "pending":
                connection.rollback()
                return
            if now_us >= int(current["expires_at_us"]):
                connection.execute(
                    "UPDATE market_data_interests SET lifecycle='expired', "
                    "reason_code='INTEREST_EXPIRED_DURING_DISCOVERY', updated_at_us=? "
                    "WHERE interest_id=? AND lifecycle='pending'",
                    (now_us, row["interest_id"]),
                )
                connection.commit()
                return
            attempts = int(current["attempts"]) + 1
            if attempts < 5:
                next_attempt = min(
                    int(current["expires_at_us"]),
                    now_us + min(60_000_000, 1_000_000 * (2 ** (attempts - 1))),
                )
                connection.execute(
                    "UPDATE market_data_interests SET attempts=?, next_attempt_at_us=?, "
                    "reason_code='DISCOVERY_RETRY', updated_at_us=? WHERE interest_id=?",
                    (attempts, next_attempt, now_us, row["interest_id"]),
                )
                self._record_incident(
                    connection,
                    None,
                    now_us,
                    "INSTRUMENT_DISCOVERY_RETRY",
                    type(error).__name__,
                )
                connection.commit()
                return
            receipt = self._denied_receipt(
                interest_id=str(row["interest_id"]),
                interest_key=str(row["interest_key"]),
                instance_id=str(row["instance_id"]),
                reason_code="DISCOVERY_RETRY_EXHAUSTED",
                completed_at_us=now_us,
            )
            self._insert_discovery_receipt(connection, receipt, self.config.run_id)
            connection.execute(
                "UPDATE market_data_interests SET lifecycle='denied', attempts=5, "
                "next_attempt_at_us=?, reason_code='DISCOVERY_RETRY_EXHAUSTED', "
                "updated_at_us=? WHERE interest_id=? AND lifecycle='pending'",
                (now_us, now_us, row["interest_id"]),
            )
            connection.commit()
        finally:
            connection.close()

    def _resolve_pending_interests(self, *, now_us: int) -> None:
        option_parameters = getattr(self.adapter, "option_parameters", None)
        option_contracts = getattr(self.adapter, "option_contracts", None)
        with connect_v2(self.config.database) as connection:
            rows = connection.execute(
                "SELECT * FROM market_data_interests WHERE run_id=? AND lifecycle='pending' "
                "AND expires_at_us>? AND next_attempt_at_us<=? ORDER BY required DESC, "
                "priority DESC, interest_id LIMIT ?",
                (
                    self.config.run_id,
                    now_us,
                    now_us,
                    _MAX_INTERESTS_PER_RECONCILIATION,
                ),
            ).fetchall()
        if not rows:
            return
        if not callable(option_parameters) or not callable(option_contracts):
            for row in rows:
                receipt = self._denied_receipt(
                    interest_id=str(row["interest_id"]),
                    interest_key=str(row["interest_key"]),
                    instance_id=str(row["instance_id"]),
                    reason_code="DISCOVERY_ADAPTER_UNAVAILABLE",
                    completed_at_us=now_us,
                )
                self._persist_discovery_receipt(receipt=receipt, option_spec=None, now_us=now_us)
            return
        resolution_started_ns = monotonic_ns()

        def completed_at_us() -> int:
            return now_us + max(0, (monotonic_ns() - resolution_started_ns) // 1_000)

        resolver = InstrumentResolver(
            cast(OptionDiscoveryBackend, self.adapter),
            underlyings={item.instrument_id: item for item in self._instruments},
            completed_at_us=completed_at_us,
            lease_heartbeat=lambda: self._heartbeat(completed_at_us()),
        )
        underlying_by_id = {item.instrument_id: item for item in self._instruments}
        for row in rows:
            self._check_owned()
            interest = self._interest_from_row(row)
            try:
                receipt = resolver.resolve(
                    InterestResolutionRequest(
                        str(row["interest_id"]),
                        str(row["instance_id"]),
                        interest,
                    )
                )
            except Exception as error:
                self._defer_discovery(row, now_us=completed_at_us(), error=error)
                continue
            if receipt.completed_at_us >= interest.expires_at_us:
                self._defer_discovery(
                    row,
                    now_us=receipt.completed_at_us,
                    error=TimeoutError("interest expired during instrument discovery"),
                )
                continue
            underlying = underlying_by_id.get(interest.underlying_instrument_id)
            option_spec = (
                None
                if receipt.status == "denied" or underlying is None
                else self._resolved_option_spec(receipt, underlying)
            )
            self._persist_discovery_receipt(
                receipt=receipt,
                option_spec=option_spec,
                now_us=receipt.completed_at_us,
            )

    def _base_request_ids(self) -> frozenset[int]:
        return frozenset(spec.request_id for spec in self._base_subscriptions)

    def _dynamic_subscriptions(self) -> tuple[SubscriptionSpec, ...]:
        base_request_ids = self._base_request_ids()
        return tuple(
            spec for spec in self._subscriptions if spec.request_id not in base_request_ids
        )

    def _next_dynamic_request_ids(
        self,
        count: int,
        *,
        reserved: tuple[int, ...] = (),
    ) -> tuple[int, ...]:
        """Allocate never-reused request identities under the sole-writer lease."""

        if not 0 <= count <= self.config.market_data_line_limit:
            raise RecorderFatalError("dynamic request allocation bound exceeded")
        if count == 0:
            return ()
        used = {
            *reserved,
            *(spec.request_id for spec in self._base_subscriptions),
            *(spec.request_id for spec in self._subscriptions),
        }
        with connect_v2(self.config.database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            row = connection.execute(
                "SELECT dynamic_request_high_water FROM runtime_state WHERE run_id=?",
                (self.config.run_id,),
            ).fetchone()
            if row is None:
                raise RecorderFatalError("dynamic request high-water state is missing")
            candidate = max(_DYNAMIC_REQUEST_ID_BASE, int(row[0]) + 1)
            allocated: list[int] = []
            while len(allocated) < count:
                while candidate in used:
                    candidate += 1
                if candidate >= _DYNAMIC_REQUEST_ID_BASE + _DYNAMIC_REQUEST_ID_SPAN:
                    raise RecorderFatalError("dynamic request id range exhausted")
                allocated.append(candidate)
                used.add(candidate)
                candidate += 1
            connection.execute(
                "UPDATE runtime_state SET dynamic_request_high_water=? WHERE run_id=? "
                "AND recorder_generation=?",
                (
                    allocated[-1],
                    self.config.run_id,
                    self._authority_state().recorder_generation,
                ),
            )
            connection.commit()
        return tuple(allocated)

    @staticmethod
    def _instrument_spec_from_row(row: sqlite3.Row) -> InstrumentSpec:
        return InstrumentSpec(
            instrument_id=str(row["instrument_id"]),
            ibkr_con_id=None if row["ibkr_con_id"] is None else int(row["ibkr_con_id"]),
            kind=str(row["kind"]),
            symbol=str(row["symbol"]),
            exchange=str(row["exchange"]),
            currency=str(row["currency"]),
            option_expiry=(None if row["option_expiry"] is None else str(row["option_expiry"])),
            option_strike=(None if row["option_strike"] is None else str(row["option_strike"])),
            option_right=None if row["option_right"] is None else str(row["option_right"]),
            option_multiplier=(
                None if row["option_multiplier"] is None else str(row["option_multiplier"])
            ),
        )

    def _dynamic_plan(self, *, now_us: int) -> tuple[MarketDataPlan, tuple[SubscriptionSpec, ...]]:
        static_demands = tuple(
            MarketDataDemand(
                source_id=f"static:{spec.name}",
                instrument_id=spec.instrument_id,
                feed_kind=spec.feed_kind,
                required=not spec.optional,
                priority=1_000,
                stale_after_us=spec.stale_after_us,
                snapshot=spec.snapshot,
            )
            for spec in self._base_subscriptions
        )
        with connect_v2(self.config.database) as connection:
            rows = connection.execute(
                "SELECT interest.*, receipt.instrument_id FROM market_data_interests interest "
                "JOIN instrument_discovery_receipts receipt USING(interest_id) "
                "WHERE interest.run_id=? AND receipt.status='resolved' "
                "AND interest.lifecycle IN ('resolved','active') AND interest.expires_at_us>? "
                "ORDER BY interest.required DESC, interest.priority DESC, interest.interest_id",
                (self.config.run_id, now_us),
            ).fetchall()
            instrument_ids = tuple(
                sorted({str(row["instrument_id"]) for row in rows if row["instrument_id"]})
            )
            instrument_rows = (
                ()
                if not instrument_ids
                else connection.execute(
                    "SELECT instrument.* FROM instruments instrument JOIN json_each(?) selected "
                    "ON selected.value=instrument.instrument_id ORDER BY instrument.instrument_id",
                    (canonical_json_bytes(cast(JsonValue, instrument_ids)).decode(),),
                ).fetchall()
            )
        dynamic_instruments = {
            str(row["instrument_id"]): self._instrument_spec_from_row(row)
            for row in instrument_rows
        }
        instrument_by_id = {item.instrument_id: item for item in self._instruments}
        instrument_by_id.update(dynamic_instruments)
        self._instruments = tuple(instrument_by_id[key] for key in sorted(instrument_by_id))
        interest_demands = tuple(
            MarketDataDemand(
                source_id=f"interest:{row['interest_id']}",
                instrument_id=str(row["instrument_id"]),
                feed_kind=str(row["feed_kind"]),
                required=bool(row["required"]),
                priority=int(row["priority"]),
                stale_after_us=(60_000_000 if str(row["cadence"]) == "snapshot" else 15_000_000),
                snapshot=str(row["cadence"]) == "snapshot",
            )
            for row in rows
        )
        retryable_interest_ids = {
            str(row["interest_id"]) for row in rows if int(row["attempts"]) < 5
        }
        plan = plan_market_data(
            static_demands,
            interest_demands,
            MarketDataCapacity(self.config.market_data_line_limit),
        )
        static_keys = {(spec.instrument_id, spec.feed_kind) for spec in self._base_subscriptions}
        current_by_key: dict[tuple[str, str, bool], SubscriptionSpec] = {}
        for spec in self._dynamic_subscriptions():
            key = (spec.instrument_id, spec.feed_kind, spec.snapshot)
            if key in current_by_key:
                raise RecorderFatalError("duplicate dynamic subscription key")
            current_by_key[key] = spec
        new_keys = tuple(
            (planned.instrument_id, planned.feed_kind, planned.snapshot)
            for planned in plan.subscriptions
            if (planned.instrument_id, planned.feed_kind) not in static_keys
            and (planned.instrument_id, planned.feed_kind, planned.snapshot) not in current_by_key
            and any(
                source_id.removeprefix("interest:") in retryable_interest_ids
                for source_id in planned.source_ids
                if source_id.startswith("interest:")
            )
        )
        allocated_ids = iter(self._next_dynamic_request_ids(len(new_keys)))
        dynamic_specs: list[SubscriptionSpec] = []
        for planned in plan.subscriptions:
            market_key = (planned.instrument_id, planned.feed_kind)
            if market_key in static_keys:
                continue
            interest_source_ids = tuple(
                source_id.removeprefix("interest:")
                for source_id in planned.source_ids
                if source_id.startswith("interest:")
            )
            if not any(
                interest_id in retryable_interest_ids for interest_id in interest_source_ids
            ):
                continue
            request_key = (*market_key, planned.snapshot)
            current = current_by_key.get(request_key)
            request_id = current.request_id if current is not None else next(allocated_ids)
            dynamic_specs.append(
                SubscriptionSpec(
                    name=(
                        f"dynamic:{planned.instrument_id}:{planned.feed_kind}:"
                        f"{'snapshot' if planned.snapshot else 'stream'}:{request_id}"
                    ),
                    instrument_id=planned.instrument_id,
                    feed_kind=planned.feed_kind,
                    request_id=request_id,
                    continuity_required=planned.required,
                    optional=not planned.required,
                    stale_after_us=planned.stale_after_us,
                    snapshot=planned.snapshot,
                )
            )
        return plan, tuple(dynamic_specs)

    @staticmethod
    def _planned_interest_ids(
        plan: MarketDataPlan,
    ) -> dict[tuple[str, str], tuple[str, ...]]:
        return {
            (subscription.instrument_id, subscription.feed_kind): tuple(
                source_id.removeprefix("interest:")
                for source_id in subscription.source_ids
                if source_id.startswith("interest:")
            )
            for subscription in plan.subscriptions
        }

    def _due_dynamic_keys(self, plan: MarketDataPlan, *, now_us: int) -> set[tuple[str, str]]:
        interest_ids_by_key = self._planned_interest_ids(plan)
        interest_ids = tuple(
            sorted(
                {
                    interest_id
                    for interest_ids in interest_ids_by_key.values()
                    for interest_id in interest_ids
                }
            )
        )
        if not interest_ids:
            return set()
        with connect_v2(self.config.database) as connection:
            due_ids = {
                str(row["interest_id"])
                for row in connection.execute(
                    "SELECT interest.interest_id FROM market_data_interests interest "
                    "JOIN json_each(?) selected ON selected.value=interest.interest_id "
                    "WHERE interest.attempts<5 AND interest.next_attempt_at_us<=?",
                    (
                        canonical_json_bytes(cast(JsonValue, interest_ids)).decode(),
                        now_us,
                    ),
                )
            }
        return {
            key
            for key, source_ids in interest_ids_by_key.items()
            if any(interest_id in due_ids for interest_id in source_ids)
        }

    @staticmethod
    def _dynamic_spec_with_request_id(
        spec: SubscriptionSpec,
        request_id: int,
    ) -> SubscriptionSpec:
        return SubscriptionSpec(
            name=(
                f"dynamic:{spec.instrument_id}:{spec.feed_kind}:"
                f"{'snapshot' if spec.snapshot else 'stream'}:{request_id}"
            ),
            instrument_id=spec.instrument_id,
            feed_kind=spec.feed_kind,
            request_id=request_id,
            continuity_required=spec.continuity_required,
            optional=spec.optional,
            stale_after_us=spec.stale_after_us,
            snapshot=spec.snapshot,
        )

    def _bind_planned_interests(
        self,
        connection: sqlite3.Connection,
        plan: MarketDataPlan,
        instrument_id: str,
        feed_kind: str,
        subscription_id: str,
    ) -> None:
        interest_ids = self._planned_interest_ids(plan).get((instrument_id, feed_kind), ())
        if not interest_ids:
            return
        connection.execute(
            "UPDATE market_data_interests SET bound_subscription_id=? "
            "WHERE interest_id IN (SELECT value FROM json_each(?)) "
            "AND lifecycle IN ('resolved','active')",
            (
                subscription_id,
                canonical_json_bytes(cast(JsonValue, interest_ids)).decode(),
            ),
        )

    def _record_subscription_attempt(
        self,
        connection: sqlite3.Connection,
        plan: MarketDataPlan,
        spec: SubscriptionSpec,
        *,
        succeeded: bool,
        now_us: int,
    ) -> None:
        interest_ids = self._planned_interest_ids(plan).get(
            (spec.instrument_id, spec.feed_kind), ()
        )
        if not interest_ids:
            return
        encoded_ids = canonical_json_bytes(cast(JsonValue, interest_ids)).decode()
        if succeeded:
            connection.execute(
                "UPDATE market_data_interests SET attempts=0, next_attempt_at_us=0, "
                "reason_code=NULL, updated_at_us=? WHERE interest_id IN "
                "(SELECT value FROM json_each(?)) AND lifecycle IN ('resolved','active')",
                (now_us, encoded_ids),
            )
            return
        rows = connection.execute(
            "SELECT interest_id, attempts, expires_at_us FROM market_data_interests "
            "WHERE interest_id IN (SELECT value FROM json_each(?)) "
            "AND lifecycle IN ('resolved','active')",
            (encoded_ids,),
        ).fetchall()
        for row in rows:
            attempts = min(5, int(row["attempts"]) + 1)
            retry_at_us = min(
                int(row["expires_at_us"]),
                now_us + min(60_000_000, 1_000_000 * (2 ** (attempts - 1))),
            )
            connection.execute(
                "UPDATE market_data_interests SET attempts=?, next_attempt_at_us=?, "
                "reason_code=?, updated_at_us=? WHERE interest_id=?",
                (
                    attempts,
                    retry_at_us,
                    ("SUBSCRIPTION_RETRY" if attempts < 5 else "SUBSCRIPTION_RETRY_EXHAUSTED"),
                    now_us,
                    row["interest_id"],
                ),
            )

    def _expire_interests(self, *, now_us: int) -> None:
        with connect_v2(self.config.database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            connection.execute(
                "UPDATE market_data_interests SET lifecycle='expired', "
                "reason_code='INTEREST_EXPIRED', updated_at_us=? WHERE run_id=? "
                "AND expires_at_us<=? AND lifecycle IN ('pending','resolved','active')",
                (now_us, self.config.run_id, now_us),
            )
            connection.commit()

    def _sync_interest_lifecycles(self, plan: MarketDataPlan, *, now_us: int) -> None:
        state = self._authority_state()
        with connect_v2(self.config.database) as connection:
            active_rows = tuple(
                connection.execute(
                    "SELECT subscription_id, instrument_id, feed_kind, snapshot "
                    "FROM subscriptions WHERE run_id=? "
                    "AND recorder_generation=? AND connection_generation=? "
                    "AND lifecycle IN ('active','degraded')",
                    (
                        self.config.run_id,
                        state.recorder_generation,
                        state.connection_generation,
                    ),
                )
            )
        active_by_key: dict[tuple[str, str], sqlite3.Row] = {}
        for row in active_rows:
            key = (str(row["instrument_id"]), str(row["feed_kind"]))
            if key in active_by_key:
                raise RecorderFatalError("multiple active subscriptions share one market-data key")
            active_by_key[key] = row
        active_keys = set(active_by_key)
        incident_id = hashlib.sha256(
            f"{self.config.run_id}|DYNAMIC_MARKET_DATA_CAPACITY".encode()
        ).hexdigest()
        with connect_v2(self.config.database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            connection.execute(
                "UPDATE market_data_interests SET lifecycle='resolved', "
                "reason_code=CASE WHEN attempts>0 THEN reason_code "
                "ELSE 'CAPACITY_DEFERRED' END, updated_at_us=? WHERE run_id=? "
                "AND lifecycle IN ('resolved','active')",
                (now_us, self.config.run_id),
            )
            for subscription in plan.subscriptions:
                key = (subscription.instrument_id, subscription.feed_kind)
                active = active_by_key.get(key)
                if active is None or bool(active["snapshot"]):
                    continue
                self._bind_planned_interests(
                    connection,
                    plan,
                    subscription.instrument_id,
                    subscription.feed_kind,
                    str(active["subscription_id"]),
                )
            active_subscription_ids = tuple(str(row["subscription_id"]) for row in active_rows)
            if active_subscription_ids:
                connection.execute(
                    "UPDATE market_data_interests SET lifecycle='active', reason_code=NULL, "
                    "updated_at_us=? WHERE run_id=? AND bound_subscription_id IN "
                    "(SELECT value FROM json_each(?)) AND lifecycle IN ('resolved','active')",
                    (
                        now_us,
                        self.config.run_id,
                        canonical_json_bytes(cast(JsonValue, active_subscription_ids)).decode(),
                    ),
                )
            for subscription in plan.subscriptions:
                active = active_by_key.get((subscription.instrument_id, subscription.feed_kind))
                if active is None or not bool(active["snapshot"]):
                    continue
                interest_ids = self._planned_interest_ids(plan).get(
                    (subscription.instrument_id, subscription.feed_kind), ()
                )
                if interest_ids:
                    connection.execute(
                        "UPDATE market_data_interests SET reason_code='SNAPSHOT_QUEUED', "
                        "updated_at_us=? WHERE interest_id IN (SELECT value FROM json_each(?)) "
                        "AND lifecycle='resolved' AND bound_subscription_id IS NULL "
                        "AND attempts=0",
                        (
                            now_us,
                            canonical_json_bytes(cast(JsonValue, interest_ids)).decode(),
                        ),
                    )
            required_active = all(
                not subscription.required
                or (subscription.instrument_id, subscription.feed_kind) in active_keys
                for subscription in plan.subscriptions
            )
            if plan.required_complete and required_active:
                connection.execute(
                    "UPDATE incidents SET resolved_at_us=? WHERE incident_id=? "
                    "AND resolved_at_us IS NULL",
                    (now_us, incident_id),
                )
            else:
                connection.execute(
                    "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
                    "opened_at_us, details_json) VALUES (?, ?, 'market_data', 'degraded', "
                    "'DYNAMIC_MARKET_DATA_CAPACITY', ?, '{}') ON CONFLICT(incident_id) "
                    "DO UPDATE SET resolved_at_us=NULL",
                    (incident_id, self.config.run_id, now_us),
                )
            connection.commit()

    def _connection_is_connected(self) -> bool:
        with connect_v2(self.config.database) as connection:
            row = connection.execute(
                "SELECT connection_state FROM runtime_state WHERE run_id=?",
                (self.config.run_id,),
            ).fetchone()
        return row is not None and str(row["connection_state"]) == "connected"

    def _restore_dynamic_subscriptions(self, *, now_us: int) -> None:
        """Recreate resolved desired subscriptions before a restarted socket connects."""

        self._expire_interests(now_us=now_us)
        plan, dynamic_specs = self._dynamic_plan(now_us=now_us)
        due_keys = self._due_dynamic_keys(plan, now_us=now_us)
        dynamic_specs = tuple(
            spec for spec in dynamic_specs if (spec.instrument_id, spec.feed_kind) in due_keys
        )
        if not dynamic_specs:
            return
        state = self._authority_state()
        with connect_v2(self.config.database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            fences = self._install_subscriptions(
                connection,
                state.recorder_generation,
                state.connection_generation,
                dynamic_specs,
                now_us,
            )
            for spec, fence in zip(dynamic_specs, fences, strict=True):
                self._bind_planned_interests(
                    connection,
                    plan,
                    spec.instrument_id,
                    spec.feed_kind,
                    cast(str, fence.subscription_id),
                )
            connection.commit()
        self._subscriptions = (*self._base_subscriptions, *dynamic_specs)
        self.state = RecorderState(
            state.run_id,
            state.recorder_generation,
            state.connection_generation,
            (*state.fences, *fences),
        )
        self._configure_adapter(
            self._instruments,
            self._subscriptions,
            required=True,
        )

    def _reconcile_dynamic_subscriptions(self, *, now_us: int) -> None:
        with self._adapter_transition_lock, self._subscription_lifecycle_lock:
            if not self._connection_is_connected():
                return
            self._reconcile_dynamic_subscriptions_locked(now_us=now_us)

    def _reconcile_dynamic_subscriptions_locked(self, *, now_us: int) -> None:
        plan, desired = self._dynamic_plan(now_us=now_us)
        current = self._dynamic_subscriptions()
        if not current and not desired:
            self._sync_interest_lifecycles(plan, now_us=now_us)
            return
        current_by_key: dict[tuple[str, str, bool], SubscriptionSpec] = {}
        for item in current:
            key = (item.instrument_id, item.feed_kind, item.snapshot)
            if key in current_by_key:
                raise RecorderFatalError("duplicate current dynamic subscription key")
            current_by_key[key] = item
        desired_by_key = {
            (item.instrument_id, item.feed_kind, item.snapshot): item for item in desired
        }
        due_keys = self._due_dynamic_keys(plan, now_us=now_us)
        state = self._authority_state()
        current_fences = {
            fence.request_id: fence for fence in state.fences if fence.request_id is not None
        }
        if any(spec.request_id not in current_fences for spec in current):
            raise RecorderFatalError("dynamic subscription fence is missing")
        current_request_ids_json = canonical_json_bytes(
            cast(JsonValue, tuple(spec.request_id for spec in current))
        ).decode()
        with connect_v2(self.config.database) as connection:
            lifecycle_by_request = {
                int(row["request_id"]): str(row["lifecycle"])
                for row in connection.execute(
                    "SELECT subscription.request_id, subscription.lifecycle "
                    "FROM subscriptions subscription JOIN json_each(?) selected "
                    "ON selected.value=subscription.request_id WHERE subscription.run_id=? "
                    "AND subscription.recorder_generation=? "
                    "AND subscription.connection_generation=?",
                    (
                        current_request_ids_json,
                        self.config.run_id,
                        state.recorder_generation,
                        state.connection_generation,
                    ),
                )
            }
        shared_keys = desired_by_key.keys() & current_by_key.keys()
        replacement_keys = tuple(
            sorted(
                key
                for key in shared_keys
                if key[:2] in due_keys
                and lifecycle_by_request.get(current_by_key[key].request_id)
                not in {"active", "degraded"}
            )
        )
        reserved_ids = tuple(spec.request_id for spec in (*current, *desired))
        fresh_ids = iter(
            self._next_dynamic_request_ids(len(replacement_keys), reserved=reserved_ids)
        )
        replacement_pairs: dict[
            tuple[str, str, bool], tuple[SubscriptionSpec, SubscriptionSpec]
        ] = {}
        for key in replacement_keys:
            old = current_by_key[key]
            fresh = self._dynamic_spec_with_request_id(desired_by_key[key], next(fresh_ids))
            desired_by_key[key] = fresh
            replacement_pairs[key] = (old, fresh)
        new_starts = tuple(
            desired_by_key[key]
            for key in sorted(desired_by_key.keys() - current_by_key.keys())
            if key[:2] in due_keys
        )
        retry_starts = tuple(replacement_pairs[key][1] for key in replacement_keys)
        starts = (*new_starts, *retry_starts)
        removed = tuple(
            current_by_key[key] for key in sorted(current_by_key.keys() - desired_by_key.keys())
        )
        replacement_old = tuple(replacement_pairs[key][0] for key in replacement_keys)
        stops = tuple(
            spec
            for spec in (*removed, *replacement_old)
            if lifecycle_by_request.get(spec.request_id) != "closed"
        )
        kept = tuple(
            current_by_key[key]
            for key in sorted(shared_keys)
            if lifecycle_by_request.get(current_by_key[key].request_id) in {"active", "degraded"}
        )
        waiting = tuple(
            current_by_key[key]
            for key in sorted(shared_keys)
            if key[:2] not in due_keys
            and lifecycle_by_request.get(current_by_key[key].request_id)
            not in {"active", "degraded"}
        )
        for key in shared_keys - set(replacement_keys):
            current_spec = current_by_key[key]
            desired_spec = desired_by_key[key]
            if (
                current_spec.name,
                current_spec.instrument_id,
                current_spec.feed_kind,
                current_spec.request_id,
            ) != (
                desired_spec.name,
                desired_spec.instrument_id,
                desired_spec.feed_kind,
                desired_spec.request_id,
            ):
                raise RecorderFatalError("dynamic subscription identity changed while active")
        with connect_v2(self.config.database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            for spec in stops:
                fence = current_fences.get(spec.request_id)
                if fence is not None:
                    connection.execute(
                        "UPDATE subscriptions SET lifecycle='cancelling', closed_at_us=NULL "
                        "WHERE subscription_id=?",
                        (fence.subscription_id,),
                    )
            new_start_fences = self._install_subscriptions(
                connection,
                state.recorder_generation,
                state.connection_generation,
                starts,
                now_us,
            )
            for spec, fence in zip(starts, new_start_fences, strict=True):
                self._bind_planned_interests(
                    connection,
                    plan,
                    spec.instrument_id,
                    spec.feed_kind,
                    cast(str, fence.subscription_id),
                )
            for spec in (*kept, *waiting):
                fence = current_fences.get(spec.request_id)
                if fence is None:
                    raise RecorderFatalError("dynamic subscription fence is missing")
                connection.execute(
                    "UPDATE subscriptions SET continuity_required=?, optional=?, snapshot=?, "
                    "requirements_hash=? WHERE subscription_id=?",
                    (
                        int(spec.continuity_required),
                        int(spec.optional),
                        int(spec.snapshot),
                        self._subscription_requirements_hash(spec),
                        fence.subscription_id,
                    ),
                )
            connection.commit()
        new_fence_by_request = {cast(int, fence.request_id): fence for fence in new_start_fences}
        start_fences = tuple(new_fence_by_request.get(spec.request_id) for spec in starts)
        if any(fence is None for fence in start_fences):
            raise RecorderFatalError("dynamic subscription start fence is missing")
        typed_start_fences = cast(tuple[CallbackFence, ...], start_fences)
        desired_subscriptions = (*self._base_subscriptions, *desired_by_key.values())
        exact = self._exact_subscriptions(self._instruments, desired_subscriptions)
        exact_by_request = {item.request_id: item for item in exact}
        starting_request_ids = {item.request_id for item in starts}
        with self._snapshot_handshake_lock:
            if self._pending_dynamic_statuses:
                raise RecorderFatalError("dynamic request status buffer was not drained")
            self._starting_dynamic_request_ids = starting_request_ids
        try:
            result = SubscriptionController(cast(SubscriptionBackend, self.adapter)).apply(
                SubscriptionApplyPlan(
                    configured=cast(tuple[object, ...], exact),
                    starts=tuple(
                        (exact_by_request[cast(int, fence.request_id)], fence)
                        for fence in typed_start_fences
                    ),
                    stops=tuple(item.request_id for item in stops),
                )
            )
        except Exception:
            with self._snapshot_handshake_lock:
                self._starting_dynamic_request_ids = set()
                self._pending_dynamic_statuses.clear()
            raise
        started = set(result.started_request_ids)
        stopped = set(result.stopped_request_ids)
        failures = {(action, request_id): code for action, request_id, code in result.failures}
        with connect_v2(self.config.database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            replacement_old_by_new = {
                fresh.request_id: old for old, fresh in replacement_pairs.values()
            }
            for spec, fence in zip(starts, typed_start_fences, strict=True):
                if spec.request_id in started:
                    connection.execute(
                        "UPDATE subscriptions SET lifecycle='active', closed_at_us=NULL "
                        "WHERE subscription_id=? AND lifecycle!='closed'",
                        (fence.subscription_id,),
                    )
                    self._record_subscription_attempt(
                        connection, plan, spec, succeeded=True, now_us=now_us
                    )
                    self._resolve_dynamic_incident(
                        connection,
                        cast(str, fence.subscription_id),
                        now_us,
                        "DYNAMIC_SUBSCRIBE_FAILED",
                    )
                    self._resolve_subscription_status_failures(
                        connection,
                        spec.instrument_id,
                        spec.feed_kind,
                        now_us,
                    )
                else:
                    replaced = replacement_old_by_new.get(spec.request_id)
                    old_stop_failed = (
                        replaced is not None
                        and lifecycle_by_request.get(replaced.request_id) != "closed"
                        and replaced.request_id not in stopped
                    )
                    lifecycle = "closed" if old_stop_failed else "paused"
                    connection.execute(
                        "UPDATE subscriptions SET lifecycle=?, closed_at_us=? "
                        "WHERE subscription_id=?",
                        (
                            lifecycle,
                            now_us if lifecycle == "closed" else None,
                            fence.subscription_id,
                        ),
                    )
                    if old_stop_failed and replaced is not None:
                        old_fence = current_fences[replaced.request_id]
                        self._bind_planned_interests(
                            connection,
                            plan,
                            replaced.instrument_id,
                            replaced.feed_kind,
                            cast(str, old_fence.subscription_id),
                        )
                    self._record_subscription_attempt(
                        connection, plan, spec, succeeded=False, now_us=now_us
                    )
                    self._record_dynamic_incident(
                        connection,
                        cast(str, fence.subscription_id),
                        now_us,
                        "DYNAMIC_SUBSCRIBE_FAILED",
                        failures.get(("subscribe", spec.request_id), "not_started"),
                    )
            for spec in stops:
                fence = current_fences[spec.request_id]
                if spec.request_id in stopped:
                    connection.execute(
                        "UPDATE subscriptions SET lifecycle='closed', closed_at_us=? "
                        "WHERE subscription_id=?",
                        (now_us, fence.subscription_id),
                    )
                    self._resolve_dynamic_incident(
                        connection,
                        cast(str, fence.subscription_id),
                        now_us,
                        "DYNAMIC_CANCEL_FAILED",
                    )
                else:
                    self._record_dynamic_incident(
                        connection,
                        cast(str, fence.subscription_id),
                        now_us,
                        "DYNAMIC_CANCEL_FAILED",
                        failures.get(("cancel", spec.request_id), "not_cancelled"),
                    )
            connection.commit()
        failed_stop_ids = {item.request_id for item in stops if item.request_id not in stopped}
        retained_removed = tuple(item for item in removed if item.request_id in failed_stop_ids)
        next_replacements = tuple(
            old if old.request_id in failed_stop_ids else fresh
            for old, fresh in replacement_pairs.values()
        )
        next_dynamic = (*kept, *waiting, *retained_removed, *new_starts, *next_replacements)
        if len(
            {(item.instrument_id, item.feed_kind, item.snapshot) for item in next_dynamic}
        ) != len(next_dynamic):
            raise RecorderFatalError("next dynamic subscription set is ambiguous")
        next_request_ids = {item.request_id for item in next_dynamic}
        base_request_ids = self._base_request_ids()
        retained_fences = tuple(
            fence
            for fence in state.fences
            if fence.request_id is None
            or fence.request_id in base_request_ids
            or fence.request_id in next_request_ids
        )
        fence_by_request = {
            fence.request_id: fence for fence in (*retained_fences, *typed_start_fences)
        }
        next_fences = tuple(
            fence
            for fence in state.fences
            if fence.request_id is None or fence.request_id in base_request_ids
        ) + tuple(fence_by_request[spec.request_id] for spec in next_dynamic)
        self._subscriptions = (*self._base_subscriptions, *next_dynamic)
        self.state = RecorderState(
            state.run_id,
            state.recorder_generation,
            state.connection_generation,
            next_fences,
        )
        self._sync_interest_lifecycles(plan, now_us=now_us)
        with self._snapshot_handshake_lock:
            self._starting_dynamic_request_ids = set()
            pending_statuses = tuple(self._pending_dynamic_statuses)
            self._pending_dynamic_statuses.clear()
        for status in pending_statuses:
            self.market_data_status(status)

    def _reconcile_dynamic_market_data(self, *, now_us: int) -> None:
        self._expire_interests(now_us=now_us)
        self._resolve_pending_interests(now_us=now_us)
        self._reconcile_dynamic_subscriptions(now_us=now_us)

    def _fulfill_snapshot_interests_from_streams(self, *, now_us: int) -> None:
        """Fulfill snapshot demand only after its bound stream records a causal event."""

        with connect_v2(self.config.database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            connection.execute(
                "UPDATE market_data_interests AS interest SET lifecycle='fulfilled', "
                "reason_code=NULL, updated_at_us=? WHERE interest.run_id=? "
                "AND interest.cadence='snapshot' "
                "AND interest.lifecycle IN ('resolved','active') "
                "AND interest.expires_at_us>? AND EXISTS ("
                "SELECT 1 FROM subscriptions subscription JOIN market_events event "
                "ON event.event_id=subscription.latest_event_id "
                "WHERE subscription.subscription_id=interest.bound_subscription_id "
                "AND subscription.run_id=interest.run_id AND subscription.snapshot=0 "
                "AND subscription.lifecycle IN ('active','degraded') "
                "AND event.run_id=interest.run_id "
                "AND event.instrument_id=subscription.instrument_id "
                "AND event.feed_kind=subscription.feed_kind "
                "AND event.event_at_us>=interest.as_of_at_us "
                "AND max(event.event_at_us, event.received_at_us)<interest.expires_at_us "
                "AND event.received_at_us<=?)",
                (now_us, self.config.run_id, now_us, now_us),
            )
            connection.commit()

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
                        *(("closed",) if spec.snapshot else ()),
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
                    "AND connection_generation=? AND optional=0 AND lifecycle!='active' "
                    "AND NOT (snapshot=1 AND lifecycle='closed') LIMIT 1",
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
        plan, _dynamic_specs = self._dynamic_plan(now_us=now_us)
        self._sync_interest_lifecycles(plan, now_us=now_us)

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

    def _record_dynamic_incident(
        self,
        connection: sqlite3.Connection,
        subscription_id: str,
        now_us: int,
        code: str,
        details: str,
    ) -> None:
        incident_id = hashlib.sha256(
            f"{self.config.run_id}|{subscription_id}|{code}".encode()
        ).hexdigest()
        details_json = canonical_json_bytes(cast(JsonValue, {"error": details})).decode()
        connection.execute(
            "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
            "subscription_id, opened_at_us, details_json) VALUES (?, ?, 'market_data', "
            "'degraded', ?, ?, ?, ?) ON CONFLICT(incident_id) DO UPDATE SET "
            "details_json=excluded.details_json, resolved_at_us=NULL",
            (
                incident_id,
                self.config.run_id,
                code,
                subscription_id,
                now_us,
                details_json,
            ),
        )

    def _resolve_dynamic_incident(
        self,
        connection: sqlite3.Connection,
        subscription_id: str,
        now_us: int,
        code: str,
    ) -> None:
        incident_id = hashlib.sha256(
            f"{self.config.run_id}|{subscription_id}|{code}".encode()
        ).hexdigest()
        connection.execute(
            "UPDATE incidents SET resolved_at_us=? WHERE incident_id=? AND resolved_at_us IS NULL",
            (now_us, incident_id),
        )

    def _resolve_subscription_status_failures(
        self,
        connection: sqlite3.Connection,
        instrument_id: str,
        feed_kind: str,
        now_us: int,
    ) -> None:
        """Resolve prior-incarnation request failures after a fresh start succeeds."""

        parameters = (
            now_us,
            self.config.run_id,
            self.config.run_id,
            instrument_id,
            feed_kind,
        )
        connection.execute(
            "UPDATE gaps SET ended_at_us=?, resolved_at_us=? WHERE run_id=? "
            "AND reason LIKE 'IBKR_STATUS_%' AND resolved_at_us IS NULL "
            "AND subscription_id IN (SELECT subscription_id FROM subscriptions "
            "WHERE run_id=? AND instrument_id=? AND feed_kind=?)",
            (
                now_us,
                now_us,
                self.config.run_id,
                self.config.run_id,
                instrument_id,
                feed_kind,
            ),
        )
        connection.execute(
            "UPDATE incidents SET resolved_at_us=? WHERE run_id=? "
            "AND code LIKE 'IBKR_STATUS_%' AND resolved_at_us IS NULL "
            "AND subscription_id IN (SELECT subscription_id FROM subscriptions "
            "WHERE run_id=? AND instrument_id=? AND feed_kind=?)",
            parameters,
        )
        connection.execute(
            "UPDATE runtime_state SET lifecycle='running', reason=NULL WHERE run_id=? "
            "AND connection_state='connected' AND lifecycle='degraded' "
            "AND reason LIKE 'IBKR_STATUS_%' AND NOT EXISTS (SELECT 1 FROM gaps "
            "WHERE run_id=? AND resolved_at_us IS NULL AND continuity_required=1)",
            (self.config.run_id, self.config.run_id),
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
            self._fulfill_snapshot_interests_from_streams(now_us=now_us)
            self._heartbeat(now_us)
            if self._idea_runner is not None:
                self._idea_runner.run_once(now_us=now_us)
            if self._connection_is_connected():
                self._reconcile_dynamic_market_data(now_us=now_us)
            if self._shadow_engine is not None:
                self._shadow_engine.run_once(now_us=now_us)
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

        with self._subscription_lifecycle_lock:
            self._disconnected_locked(now_us=now_us)

    def _disconnected_locked(self, *, now_us: int) -> None:
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

        with self._snapshot_handshake_lock:
            if self._starting_dynamic_request_ids and (
                status.request_id is None or status.request_id in self._starting_dynamic_request_ids
            ):
                status_limit = 16 * len(self._starting_dynamic_request_ids)
                if len(self._pending_dynamic_statuses) >= status_limit:
                    raise RecorderFatalError("dynamic request status buffer exceeded")
                self._pending_dynamic_statuses.append(status)
                return
        if status.kind == "snapshot_end":
            self._complete_dynamic_snapshot(status)
            return
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
        dynamic_plan = (
            self._dynamic_plan(now_us=status.received_at_us)[0]
            if spec is not None and spec.request_id not in self._base_request_ids()
            else None
        )
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
                    if dynamic_plan is not None:
                        self._record_subscription_attempt(
                            connection,
                            dynamic_plan,
                            spec,
                            succeeded=False,
                            now_us=status.received_at_us,
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
        if dynamic_plan is not None and status.kind != "recovered":
            self._sync_interest_lifecycles(
                dynamic_plan,
                now_us=status.received_at_us,
            )

    def _retire_dynamic_snapshot_state(
        self,
        *,
        request_id: int,
        fence: CallbackFence,
        state: RecorderState,
    ) -> None:
        self._subscriptions = tuple(
            item for item in self._subscriptions if item.request_id != request_id
        )
        self.state = RecorderState(
            state.run_id,
            state.recorder_generation,
            state.connection_generation,
            tuple(item for item in state.fences if item != fence),
        )

    def _complete_dynamic_snapshot(self, status: MarketDataStatus) -> None:
        if status.request_id is None:
            return
        with self._subscription_lifecycle_lock:
            self._check_owned()
            state = self._authority_state()
            fence = next(
                (item for item in state.fences if item.request_id == status.request_id), None
            )
            spec = next(
                (item for item in self._subscriptions if item.request_id == status.request_id),
                None,
            )
            if (
                fence is None
                or spec is None
                or spec.request_id in self._base_request_ids()
                or not spec.snapshot
            ):
                return
            with connect_v2(self.config.database) as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._verify_owned(connection)
                connection.execute(
                    "UPDATE subscriptions SET lifecycle='closed', closed_at_us=? "
                    "WHERE subscription_id=? AND lifecycle!='closed'",
                    (status.received_at_us, fence.subscription_id),
                )
                connection.execute(
                    "UPDATE market_data_interests SET "
                    "lifecycle=CASE WHEN expires_at_us<=? THEN 'expired' ELSE 'fulfilled' END, "
                    "reason_code=CASE WHEN expires_at_us<=? "
                    "THEN 'INTEREST_EXPIRED' ELSE NULL END, "
                    "updated_at_us=? WHERE bound_subscription_id=? "
                    "AND lifecycle IN ('resolved','active')",
                    (
                        status.received_at_us,
                        status.received_at_us,
                        status.received_at_us,
                        fence.subscription_id,
                    ),
                )
                connection.commit()
            self._retire_dynamic_snapshot_state(
                request_id=status.request_id,
                fence=fence,
                state=state,
            )
        updated_plan, _dynamic_specs = self._dynamic_plan(now_us=status.received_at_us)
        self._sync_interest_lifecycles(updated_plan, now_us=status.received_at_us)

    def _market_data_farm_status(self, status: MarketDataStatus) -> None:
        """Scope farm health to affected feeds while the shared socket remains connected."""

        with self._subscription_lifecycle_lock:
            self._market_data_farm_status_locked(status)

    def _market_data_farm_status_locked(self, status: MarketDataStatus) -> None:
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
                    "AND connection_generation=? AND optional=0 AND lifecycle!='active' "
                    "AND NOT (snapshot=1 AND lifecycle='closed') LIMIT 1",
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
        with self._adapter_reset_transition(), self._subscription_lifecycle_lock:
            old = self._authority_state()
            self._expire_interests(now_us=now_us)
            reconnect_plan, dynamic = self._dynamic_plan(now_us=now_us)
            fresh_request_ids = iter(
                self._next_dynamic_request_ids(
                    len(dynamic),
                    reserved=tuple(spec.request_id for spec in self._subscriptions),
                )
            )
            replacement_by_request = {
                spec.request_id: self._dynamic_spec_with_request_id(spec, next(fresh_request_ids))
                for spec in dynamic
            }
            reconnect_subscriptions = tuple(
                (*self._base_subscriptions, *replacement_by_request.values())
            )
            prior_request_by_request = {
                replacement.request_id: request_id
                for request_id, replacement in replacement_by_request.items()
            }
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
                    "UPDATE subscriptions SET lifecycle='closed', closed_at_us=? "
                    "WHERE run_id=? AND connection_generation=? AND lifecycle!='closed'",
                    (now_us, self.config.run_id, old.connection_generation),
                )
                generation = old.connection_generation + 1
                fences = self._install_subscriptions(
                    connection,
                    old.recorder_generation,
                    generation,
                    reconnect_subscriptions,
                    now_us,
                )
                connection.execute(
                    "UPDATE runtime_state SET connection_generation=?, "
                    "connection_state='disconnected', lifecycle=CASE WHEN "
                    "lifecycle='degraded' AND reason='IBKR_DISCONNECT' "
                    "THEN 'recovering' ELSE lifecycle END, reason=CASE WHEN "
                    "lifecycle='degraded' AND reason='IBKR_DISCONNECT' "
                    "THEN NULL ELSE reason END, process_heartbeat_at_us=? WHERE run_id=?",
                    (generation, now_us, self.config.run_id),
                )
                for fence, spec in zip(fences, reconnect_subscriptions, strict=True):
                    prior_request_id = prior_request_by_request.get(
                        spec.request_id, spec.request_id
                    )
                    remains_paused = prior_request_id in paused_request_ids
                    if remains_paused:
                        connection.execute(
                            "UPDATE subscriptions SET lifecycle='paused' WHERE subscription_id=?",
                            (fence.subscription_id,),
                        )
                    if not remains_paused:
                        self._bind_planned_interests(
                            connection,
                            reconnect_plan,
                            spec.instrument_id,
                            spec.feed_kind,
                            cast(str, fence.subscription_id),
                        )
                    self._open_gap(
                        connection,
                        cast(str, fence.subscription_id),
                        now_us,
                        "RECONNECT_UNCERTAINTY",
                        spec.continuity_required,
                    )
                connection.commit()
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                connection.close()
            self._subscriptions = reconnect_subscriptions
            self.state = RecorderState(
                self.config.run_id,
                old.recorder_generation,
                generation,
                fences,
            )
            self._configure_adapter(
                self._instruments,
                reconnect_subscriptions,
                required=bool(self.idea_requirements or dynamic),
            )
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

    def abandon_unclean(self) -> None:
        """Release process-local resources while leaving durable restart evidence unclean."""

        self.state = None
        with suppress(Exception):
            self.adapter.disconnect()
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
        self._sync_backup_status(now_us=now_us)
        if not result.admission_allowed:
            self._fatal(result.required_action or "STORAGE_CAP_FATAL", now_us)
            raise RecorderFatalError(result.required_action or "storage cap closed admission")
        if not result.optional_feeds_allowed:
            self._pause_optional(now_us)
            self._set_lifecycle("degraded", result.required_action, now_us)
        return result.cap_state

    def _sync_backup_status(self, *, now_us: int) -> None:
        """Persist backup health through the sole authoritative database writer."""

        if self.config.backup_directory is None:
            return
        status = read_backup_manifests(self.config.backup_directory, limit=1).status
        connection = connect_v2(self.config.database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_owned(connection)
            if status.state == "healthy":
                connection.execute(
                    "UPDATE incidents SET resolved_at_us=? WHERE run_id=? AND scope='storage' "
                    "AND code='BACKUP_DEGRADED' AND resolved_at_us IS NULL",
                    (now_us, self.config.run_id),
                )
            else:
                existing = connection.execute(
                    "SELECT 1 FROM incidents WHERE run_id=? AND scope='storage' "
                    "AND code='BACKUP_DEGRADED' AND resolved_at_us IS NULL LIMIT 1",
                    (self.config.run_id,),
                ).fetchone()
                if existing is None:
                    status_code = status.code or "BACKUP_STATUS_UNAVAILABLE"
                    opened_at_us = min(status.checked_at_us or now_us, now_us)
                    incident_id = hashlib.sha256(
                        f"{self.config.run_id}|backup|{opened_at_us}|{status_code}".encode()
                    ).hexdigest()
                    details_json = canonical_json_bytes(
                        cast(JsonValue, {"backup_status_code": status_code})
                    ).decode()
                    connection.execute(
                        "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
                        "opened_at_us, details_json) VALUES (?, ?, 'storage', 'degraded', "
                        "'BACKUP_DEGRADED', ?, ?)",
                        (incident_id, self.config.run_id, opened_at_us, details_json),
                    )
            connection.commit()
        finally:
            connection.close()

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
