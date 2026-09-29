"""Existing SLRNO runtime, now Saxo-only and unarmed on every startup."""

import asyncio
import fcntl
import hashlib
import json
import logging
import re
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from stocker_execution import option_context, views
from stocker_execution.alerts import Alerts
from stocker_execution.broker import PaperBroker, now
from stocker_execution.config import (
    MAX_OPEN_POSITIONS,
    MAX_PREMIUM_RISK_GBP,
    MAX_SIMULTANEOUS_ENTRY_RISK_GBP,
    PAGE_SIZE,
    QUOTE_MAX_AGE_SECONDS,
    ROLLING_WINDOW_SECONDS,
    RULE_VERSION,
    SUBSCRIPTION_LIMIT,
    FuturesConfig,
)
from stocker_execution.contracts import key, quote_check, session_state
from stocker_execution.event_calendar import EventCalendar, load_calendar
from stocker_execution.recorder import Recorder
from stocker_execution.rules import NY, clocks, eligibility, next_clock, opportunity, prior_rv
from stocker_execution.saxo_auth import OAuth
from stocker_execution.saxo_client import REST_QUEUE_LIMIT, SaxoClient
from stocker_execution.saxo_data import DataService, MarketState
from stocker_execution.store import Store, encode

log = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def failure_code(exc: BaseException) -> str:
    """Coded ValueErrors are the application's own reasons; other text may hold secrets."""
    if isinstance(exc, ValueError) and re.fullmatch(r"[A-Za-z0-9_]{1,120}", str(exc)):
        return str(exc)
    return type(exc).__name__


def skip_reason(exc: ValueError | KeyError) -> str:
    """Trading gates raise coded ValueErrors; a KeyError is a data/code gap, not a reason."""
    if isinstance(exc, KeyError):
        log.error("decision input missing field %r", exc.args[0] if exc.args else None)
        return "UNEXPECTED_MISSING_FIELD"
    return str(exc)


class Runtime:
    def __init__(self, config: FuturesConfig, store: Store, data: DataService | None = None):
        self.config, self.store = config, store
        store.bind(config.data_environment, config.execution_mode)
        self.root = Path(store.db.execute("PRAGMA database_list").fetchone()[2]).parent
        if data is None:
            recorder = Recorder(config.recorder, self.root / config.data_environment / "events")
            client = SaxoClient(OAuth(config.data_environment, config.saxo, self.root))
            data = DataService(config, client, recorder)
        self.data = data
        self.recorder = data.recorder
        self.broker = PaperBroker(config, store, self.data)
        self.data.owned_options = {json.loads(r["plan"])["option"]["uic"] for r in store.active()}
        self.markets = self.data.markets
        self.worker_health = self.manager_health = self.web_health = "STARTING"
        self.stopping = False
        self.history_needed = asyncio.Event()
        self.tasks: set[asyncio.Task[Any]] = set()
        self.owner: Any = None
        self.config_hash = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
        self.started_at = now()
        self.depth = self.recorder  # existing read-only dashboard recording access
        self.alerts = Alerts(config.alerts)
        # Optional context: an unreadable calendar is reported, never fatal.
        self.calendar: EventCalendar | None = None
        self.calendar_problem = ""
        if config.event_calendar_file:
            try:
                self.calendar = load_calendar(config.event_calendar_file)
            except (OSError, ValueError):
                self.calendar_problem = "EVENT_CALENDAR_UNREADABLE"
        directory = self.root / config.data_environment
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        handler = RotatingFileHandler(directory / "slrno.log", maxBytes=2 * 1024**2, backupCount=3)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        self.log_handler = handler
        for name in ("httpx", "httpcore", "websockets"):
            logging.getLogger(name).setLevel(logging.WARNING)
        selections = {k: v.model_dump(mode="json") for k, v in config.contracts.items()}
        if store.get_meta("display_contracts") != selections:
            with store.db:
                store.audit(
                    "configuration",
                    "EXPLICIT_DISPLAY_CONTRACT_SELECTION",
                    {
                        "before": store.get_meta("display_contracts"),
                        "after": selections,
                        "configuration_hash": self.config_hash,
                    },
                )
            store.set_meta("display_contracts", selections)

    @property
    def pause(self) -> bool:
        return bool(self.store.get_meta("paused", False))

    @pause.setter
    def pause(self, value: bool) -> None:
        self.store.set_meta("paused", value)

    async def decision_pass(self) -> None:
        try:
            await self.decisions()
        except Exception as exc:
            # Sticky and alerted: reconciliation must not clear it; an operator restart does.
            self.broker.armed = False
            self.broker.fatal_error = "DECISION_WORKER_ERROR_REVIEW_REQUIRED"
            self.report_failure("decisions", exc)

    def report_failure(self, worker: str, exc: BaseException) -> None:
        # Only coded reasons are logged; other exception text can contain URLs or credentials.
        log.error("%s failed: %s", worker, failure_code(exc))

    async def cancel_tasks(self, tasks: set[asyncio.Task[Any]]) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def status(self) -> dict[str, Any]:
        at = now()
        return {
            "application": "SLRNO",
            # Clients derive countdowns from server time, so a skewed browser clock is harmless.
            "server_time": at.timestamp(),
            "next_clock": next_clock(at).isoformat(),
            "setup": views.setup(self),
            "alerts": self.alerts.status(),
            "activity_events": self.data.activity_problem
            or (
                "SUBSCRIBED"
                if any(s["kind"] == "ACTIVITIES" for s in self.data.subscriptions.values())
                else "NOT_SUBSCRIBED"
            ),
            "closed_positions_problem": self.broker.closed_problem,
            "calendar_problem": self.calendar_problem,
            "limits": {
                "per_trade_gbp": MAX_PREMIUM_RISK_GBP,
                "slots": MAX_OPEN_POSITIONS,
                "allocation_gbp": MAX_SIMULTANEOUS_ENTRY_RISK_GBP,
                "quote_max_age_seconds": QUOTE_MAX_AGE_SECONDS,
            },
            "data_environment": self.config.data_environment,
            "execution_mode": self.config.execution_mode,
            "account": "Verified configured Saxo account"
            if self.data.account_verified
            else "Unverified",
            "connected": self.data.connected,
            "reconciled": self.broker.reconciled,
            "armed": self.broker.armed,
            "paused": self.pause,
            "entry_block_reason": self.broker.entry_reason(),
            "problem": self.data.problem,
            "worker_health": self.worker_health,
            "manager_health": self.manager_health,
            "web_health": self.web_health,
            "rule_version": RULE_VERSION,
            "configuration_hash": self.config_hash,
            "live_available": False,
            "live_orders_disabled": True,
            "oauth": self.data.client.oauth.status,
            "oauth_problem": self.data.client.oauth.failure_reason,
            "session": self.data.session,
            "stream": {
                "connected": self.data.connected,
                "reconnects": self.data.reconnects,
                "last_message_id": self.data.last_message_id,
            },
            "market_data": {
                "owned_lines": len(self.data.subscriptions),
                "app_budget": SUBSCRIPTION_LIMIT,
                "option_budget": self.config.option_subscription_budget,
                "option_lines": len(self.data.options),
                "candidate_window": self.config.option_candidate_window,
                "rate_limits": self.data.client.rate_headers,
                "rest_queue": self.data.client.waiters,
                "rest_queue_limit": REST_QUEUE_LIMIT,
            },
            "l2_recording": self.recorder.status(),
            "bar_storage": {
                "bytes": self.data.bar_cache.size,
                "limit": self.config.recorder.bar_max_bytes,
                "problem": self.data.bar_cache.problem,
            },
            "management_problems": self.broker.management_problems,
            **self.store.capacity(),
        }

    def block_reason(self, state: MarketState, paused: bool) -> str:
        """The first display gate a clock decision would hit; the decision re-checks itself."""
        reason = state.problem or state.history_problem
        if not reason and state.market not in self.config.mappings:
            reason = "LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED"
        return reason or ("ENTRIES_PAUSED" if paused else self.broker.entry_reason())

    def market_card(
        self, state: MarketState, at: datetime, trades: list[dict[str, Any]], paused: bool
    ) -> dict[str, Any]:
        """The fields the overview and market pages must agree on."""
        market, reason = state.market, self.block_reason(state, paused)
        return {
            "market": market,
            "contract": (state.identity or {}).get("symbol"),
            "strategy_state": trades[0]["state"]
            if trades
            else "MONITOR_ONLY"
            if market == "GC" and reason
            else "BLOCKED"
            if reason
            else "MONITORING",
            "block_reason": reason,
            "entry_enabled": not reason,
            "data_status": "CURRENT"
            if views.quote_current(state, at.timestamp())
            else "STALE_OR_MISSING",
            "last_receipt": state.price.receipt,
            "candidate_uic": state.candidate_uic,
            "trades": trades,
            "next_time": (state.boundary_clock or next_clock(at)).isoformat(),
            "pending_data": state.boundary_clock is not None,
            "gates": views.gates(self, market, state, at.timestamp(), paused),
        }

    def overview(self) -> dict[str, Any]:
        trades = [self.trade_view(t) for t in self.store.active()]
        at, paused = now(), self.pause
        clock = next_clock(at)
        cards = [
            {
                **self.market_card(state, at, [t for t in trades if t["market"] == market], paused),
                "events": self.calendar.near(market, clock) if self.calendar else [],
            }
            for market, state in self.markets.items()
        ]
        pnl = self.store.economics()
        pnl = {
            **pnl,
            "broker_reported": {**pnl["broker_reported"], "currency": self.data.account_currency},
        }
        return {
            "system": self.status(),
            "account": self.data.balance_view(at.timestamp()),
            "markets": cards,
            "pnl": pnl,
        }

    def execution_view(self) -> dict[str, Any]:
        rows = self.store.active()
        return {
            "system": self.status(),
            "trades": [self.trade_view(r) for r in rows],
            "orders": [o for r in rows for o in self.store.orders(r["id"])],
            "fills": [
                dict(r)
                for r in self.store.db.execute(
                    "SELECT f.*,o.event_id,o.role FROM fills f JOIN orders o USING(reference) "
                    "ORDER BY at DESC,exec_id LIMIT ?",
                    (PAGE_SIZE,),
                )
            ],
            "positions": [
                dict(r)
                for r in self.store.db.execute(
                    "SELECT * FROM positions WHERE quantity<>0 ORDER BY con_id LIMIT ?",
                    (PAGE_SIZE,),
                )
            ],
        }

    def market_detail(self, selected: str, *, diagnostics: bool = False) -> dict[str, Any]:
        active = self.store.active()
        recent = self.store.recent_signals(selected)
        market, state = selected, self.markets[selected]
        last_event = recent[0] if recent else None
        event_context = (
            json.loads(
                self.store.db.execute(
                    "SELECT detail FROM signals WHERE id=?", (last_event["id"],)
                ).fetchone()[0]
            ).get("option_context")
            if last_event
            else None
        )
        at = now()
        card = self.market_card(
            state, at, [self.trade_view(t) for t in active if t["market"] == market], self.pause
        )
        quote = (state.price.value or {}).get("Quote") or {}
        identity = state.identity
        depth = state.price.depth(at.timestamp())
        depth["last_receipt"] = (
            datetime.fromtimestamp(state.price.depth_receipt, UTC).isoformat()
            if state.price.depth_receipt
            else None
        )
        recording = (
            self.recorder.view(key(identity), at.timestamp())
            if identity
            else {"state": "UNAVAILABLE", "prehistory_seconds": 0}
        )
        depth.update(
            pre_seconds=recording["prehistory_seconds"],
            target_pre_seconds=ROLLING_WINDOW_SECONDS,
        )
        features = {}
        try:
            if state.bars:
                features["rv15"] = prior_rv(
                    state.bars,
                    state.bars[-1].at.replace(second=0) + timedelta(minutes=1),
                )
        except ValueError:
            pass
        day = at.astimezone(NY).date()
        card.update(
            {
                "identity": identity,
                "market_status": session_state(state.reference, at),
                "direction": "BUY CALL" if market == "CL" else "BUY PUT",
                "conditions": features,
                "l1": {
                    "status": card["data_status"],
                    "quote": quote,
                    "sizes": {side.lower(): size for side, size in state.price.sizes().items()},
                    "spread": quote["Ask"] - quote["Bid"]
                    if isinstance(quote.get("Ask"), (int, float))
                    and isinstance(quote.get("Bid"), (int, float))
                    else None,
                    "delay_minutes": quote.get("DelayedByMinutes"),
                    "last_receipt": datetime.fromtimestamp(state.price.receipt, UTC).isoformat()
                    if state.price.receipt
                    else None,
                },
                "l2": depth,
                "book_flow": self.recorder.book_flow_view(key(identity), at.timestamp())
                if identity
                else {"status": "UNAVAILABLE"},
                "recorder": recording,
                "capabilities": dict(state.capabilities) if diagnostics else None,
                "underlying_context": option_context.view(state.price.analytics, at.timestamp()),
                "option_context": {
                    "latest_event": {
                        "id": last_event["id"],
                        "signal_at": last_event["signal_at"],
                        "context": event_context,
                    }
                    if last_event and event_context
                    else None,
                    "candidate_uic": state.candidate_uic,
                    "problem": state.candidate_problem,
                    "candidate_changes": list(state.candidate_changes) if diagnostics else None,
                    "contracts": [
                        self.data.option_view(uic, at.timestamp())
                        for uic, (i, _) in self.data.options.items()
                        if i["market"] == market
                    ],
                },
                "chart": [{"at": b.at.isoformat(), "close": b.close} for b in state.bars[-90:]],
                "chart_context": views.chart_context(state, at),
                "sessions_today": views.sessions_today(state, at),
                "events_today": [
                    {"at": when.isoformat(), "name": name}
                    for when, name in self.calendar.occurrences(market, day)
                ]
                if self.calendar
                else [],
                "candidate_deltas": {str(k): v for k, v in state.candidate_deltas.items()},
                "price_context": views.price_context(state),
                "smile": views.smile(
                    state.option_board,
                    day.isoformat(),
                    self.config.provider_volatility_scale,
                    views.model_sigma(features.get("rv15")),
                ),
                "model_sigma": views.model_sigma(features.get("rv15")),
                "target_delta": 0.2 if market == "SI" else 0.1,
                "signals": [
                    {k: s[k] for k in ("id", "signal_at", "decision", "reason")} for s in recent
                ],
                "details": {
                    "signal_contract": identity,
                    "reference_sessions": len(state.references),
                    "entry_clocks": "09:00–16:00 America/New_York weekdays; NG 13:00 veto",
                    "exit_anchor": "Original opportunity + 60 minutes",
                    "options_block": "Verify actual 0DTE expiry, product and delta tolerance",
                    "recording": recording,
                    "option_board": state.option_board,
                }
                if diagnostics
                else None,
            }
        )
        return {"system": self.status(), "markets": [card]}

    def trade_view(self, row: dict[str, Any]) -> dict[str, Any]:
        plan = json.loads(row["plan"])
        fills = self.store.fills(row["id"])
        quantity = self.store.exposure(row["id"])
        valuation: dict[str, Any] = {
            "value_gbp": None,
            "fresh": False,
            "method": "CONSERVATIVE_BID_ESTIMATE",
        }
        try:
            _, price = self.data.options[plan["option"]["uic"]]
            q = quote_check(price.value or {}, price.receipt, now())
            fx = quote_check(self.data.fx.value or {}, self.data.fx.receipt, now())
            entries = [f for f in fills if f["side"] == "BOT"]
            if quantity and entries and all(f["fx"] is not None for f in entries):
                cost = sum(f["quantity"] * f["price"] * f["fx"] for f in entries)
                valuation.update(
                    value_gbp=(quantity * float(q["Bid"]) / float(fx["Ask"]) - cost)
                    * plan["multiplier"],
                    fresh=True,
                    quote_at=datetime.fromtimestamp(price.receipt or 0, UTC).isoformat(),
                    fx_at=self.data.fx.receipt,
                )
        except (KeyError, ValueError):
            pass
        return {
            "id": row["id"],
            "market": row["market"],
            "allocation_pennies": row["allocation_pennies"],
            "policy_pennies": row["policy_pennies"],
            "state": row["state"],
            "quantity": quantity,
            "entry_at": next((f["at"] for f in fills if f["side"] == "BOT"), None),
            "exit_at": row["exit_at"],
            "option": plan["option"],
            "premium_gbp": plan["premium_gbp"],
            "fees_gbp": plan["fees_gbp"],
            "total_gbp": plan["total_gbp"],
            "valuation": valuation,
            "basis": self.config.execution_mode,
        }

    async def decisions(self) -> None:
        at = now()
        for state in self.markets.values():
            if not state.identity:
                continue
            for clock in clocks(at.astimezone(NY).date()):
                if (
                    clock < self.started_at
                    or clock > at
                    or (state.last_clock and clock <= state.last_clock)
                ):
                    continue
                event = opportunity(state.market, state.identity["uic"], clock)
                event["id"] = self.config.data_environment + "|" + str(event["id"])
                event.update(
                    provider="SAXO",
                    environment=self.config.data_environment,
                    contract=state.identity,
                    configuration_hash=self.config_hash,
                )
                if self.calendar:
                    # Observation only: recorded with the event, never an entry rule.
                    event["scheduled_events"] = self.calendar.near(state.market, clock)
                reason = str(event["veto"])
                inputs: dict[str, float] = {}
                # Earlier markets await broker I/O: gates need the time now, not at loop entry,
                # or a quote refreshed meanwhile has a negative age and reads as stale.
                at = now()
                try:
                    if not reason:
                        if (at - clock).total_seconds() >= self.config.entry_deadline_seconds:
                            raise ValueError("STALE_SIGNAL_NO_REPLAY")
                        if state.problem or state.history_problem:
                            raise ValueError(state.problem or state.history_problem)
                        quote_check(state.price.value or {}, state.price.receipt, at)
                        inputs = eligibility(
                            state.bars,
                            clock,
                            datetime.fromisoformat(state.identity["expiry"][:10]).date(),
                            state.references,
                        )
                except ValueError as exc:
                    reason = str(exc)
                # Only the final completed-minute boundary may wait. Earlier gaps,
                # invalid bars, reference failures and other gates remain final skips.
                by_time = {b.at: b for b in state.bars}
                if (
                    reason == "INCOMPLETE_COMPLETED_HISTORY"
                    and (at - clock).total_seconds() < self.config.entry_deadline_seconds
                    and clock - timedelta(minutes=1) not in by_time
                    and all(
                        (b := by_time.get(clock - timedelta(minutes=i))) is not None and b.valid()
                        for i in range(2, 32)
                    )
                ):
                    if state.boundary_clock != clock:
                        state.boundary_clock = clock
                        state.boundary_checked = float("-inf")
                        self.history_needed.set()
                    continue
                state.boundary_clock = None
                state.last_clock = clock
                # Capture an observed frozen clock even for veto, capacity, option or data skips.
                event["skip_reason"] = reason
                observed = self.store.observe(event, reason, inputs)
                if not observed:
                    continue
                option_keys = [
                    key(i)
                    for i, _ in self.data.options.values()
                    if i["market"] == state.market
                    and i["underlying_uic"] == state.identity["uic"]
                    and i["uic"] in state.warm_uics
                ]
                capture = self.recorder.trigger(
                    key(state.identity), event, time.time(), option_keys
                )
                self.store.depth_capture(str(event["id"]), capture)
                context: dict[str, Any] = {"selection_status": "NOT_SELECTED", "reason": reason}
                if not reason:
                    try:
                        selected, _ = await self.broker.select_option(event, state, inputs)
                        context = self.data.option_view(selected["uic"], time.time())
                        context.update(
                            **views.iv_spread(context, inputs.get("rv15")),
                            selection_status="SELECTED",
                            strategy_exit_at=event["exit_at"],
                            pre_trigger_seconds=max(
                                0,
                                context["coverage_seconds"]
                                - max(0, time.time() - clock.timestamp()),
                            ),
                        )
                    except (ValueError, KeyError) as exc:
                        context["reason"] = skip_reason(exc)
                self.recorder.annotate(str(event["id"]), {"option_context": context})
                with self.store.db:
                    row = self.store.db.execute(
                        "SELECT detail FROM signals WHERE id=?", (event["id"],)
                    ).fetchone()
                    detail = json.loads(row[0])
                    detail["option_context"] = context
                    # Observation only: the chain as seen at this clock (provider units).
                    detail["option_chain"] = views.smile(
                        state.option_board,
                        clock.astimezone(NY).date().isoformat(),
                        self.config.provider_volatility_scale,
                        views.model_sigma(inputs.get("rv15")),
                    )
                    self.store.db.execute(
                        "UPDATE signals SET detail=? WHERE id=?", (encode(detail), event["id"])
                    )
                if not reason:
                    reason = "ENTRIES_PAUSED" if self.pause else self.broker.entry_reason()
                if not reason:
                    try:
                        plan = await self.broker.prepare(event, state, inputs)
                        reason = await self.broker.enter(event, plan)
                    except (ValueError, KeyError) as exc:
                        reason = skip_reason(exc)
                if reason:
                    self.store.decision(str(event["id"]), "SKIPPED", reason)
                self.recorder.annotate(str(event["id"]), {"skip_reason": reason, "inputs": inputs})

    async def manager(self) -> None:
        self.manager_health = "RUNNING"
        last_failure = ""
        while not self.stopping:
            try:
                await self.broker.manage()
                last_failure = ""
            except Exception as exc:
                self.broker.reconciled = False
                # Log transitions, not every two-second retry of the same failure.
                if failure_code(exc) != last_failure:
                    self.report_failure("management", exc)
                last_failure = failure_code(exc)
            await asyncio.sleep(2)

    async def refresh_histories(self) -> None:
        # A small priority pass at the normal five-second worker cadence, woken once
        # by a newly pending clock. Do not put optional option work ahead of this pass.
        for state in self.markets.values():
            clock = state.boundary_clock
            if (
                self.data.connected
                and clock is not None
                and 0 <= (now() - clock).total_seconds() < self.config.entry_deadline_seconds
                and time.monotonic() - state.boundary_checked >= 5
            ):
                state.boundary_checked = time.monotonic()
                try:
                    await self.data.history(state, boundary=True)
                except Exception as exc:
                    self.report_failure("boundary-history", exc)
        for state in self.markets.values():
            if self.history_needed.is_set():
                return
            if (
                self.data.connected
                and state.boundary_clock is None
                and time.monotonic() - state.history_checked >= 60
            ):
                try:
                    await self.data.history(state)
                    await self.data.warm_candidates(state)
                    await self.data.daily_context(state)
                except Exception as exc:
                    state.history_problem = "SAXO_HISTORY_UNAVAILABLE"
                    state.history_checked = time.monotonic()
                    self.report_failure("history", exc)

    async def alert_worker(self) -> None:
        # Isolated: an alert failure is recorded in alerts.status() and never stops trading.
        while not self.stopping:
            try:
                await self.alerts.check(self)
            except Exception as exc:
                self.report_failure("alerts", exc)
            await asyncio.sleep(30)

    async def history_worker(self) -> None:
        while not self.stopping:
            self.history_needed.clear()
            await self.refresh_histories()
            if not self.history_needed.is_set():
                try:
                    await self.data.release_unused_options(
                        {json.loads(r["plan"])["option"]["uic"] for r in self.store.active()}
                    )
                    if self.data.connected:
                        await self.data.refresh_option_metadata()
                        await self.data.ensure_balance_subscription()
                        await self.data.ensure_activity_subscription()
                        await self.broker.refresh_closed_positions()
                except Exception as exc:
                    self.report_failure("option-subscription-maintenance", exc)
            with suppress(TimeoutError):
                await asyncio.wait_for(self.history_needed.wait(), timeout=5)

    async def run(self) -> None:
        # Environment lock prevents another browser/server instance duplicating subscriptions.
        lock = self.root / self.config.data_environment / "data.owner.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        self.owner = lock.open("a")
        fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.worker_health = "RUNNING"
        try:
            await self.recorder.start()
        except (OSError, ValueError):
            self.recorder.problem = "RECORDER_STORAGE_UNAVAILABLE"
        for row in self.store.db.execute(
            "SELECT s.id,c.summary FROM signals s LEFT JOIN depth_captures c USING(id) "
            "WHERE c.id IS NULL OR json_extract(c.summary,'$.state')='CAPTURING'"
        ).fetchall():
            summary = json.loads(row["summary"]) if row["summary"] else {}
            self.store.depth_capture(
                row["id"],
                {**summary, "state": "INCOMPLETE", "reason": "INTERRUPTED_RESTART_CHECK_MANIFEST"},
            )
        self.tasks.update(
            {
                asyncio.create_task(self.data.run()),
                asyncio.create_task(self.manager()),
                asyncio.create_task(self.history_worker()),
                asyncio.create_task(self.alert_worker()),
            }
        )
        try:
            while not self.stopping:
                try:
                    self.recorder.tick(time.time())
                except Exception as exc:
                    self.recorder.problem = "RECORDER_FAILED"
                    self.report_failure("recorder", exc)
                await self.decision_pass()
                for task in self.tasks:
                    if task.done() and not task.cancelled():
                        raise RuntimeError("BACKGROUND_WORKER_STOPPED") from task.exception()
                await asyncio.sleep(0.25)
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self.stopping:
            return
        self.stopping = self.data.stopping = True
        self.broker.armed = False
        await self.cancel_tasks(self.tasks)
        await self.recorder.close()
        await self.data.client.close()
        await self.alerts.close()
        log.removeHandler(self.log_handler)
        self.log_handler.close()
        if self.owner:
            self.owner.close()
            self.owner = None
