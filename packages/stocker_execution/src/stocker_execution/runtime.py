"""Existing SLRNO runtime, now Saxo-only and unarmed on every startup."""

import asyncio
import fcntl
import hashlib
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from stocker_execution.broker import PaperBroker, now
from stocker_execution.config import RULE_VERSION, FuturesConfig
from stocker_execution.contracts import key, quote_check, session_state
from stocker_execution.recorder import Recorder
from stocker_execution.rules import NY, clocks, eligibility, next_clock, opportunity, prior_rv
from stocker_execution.saxo_auth import OAuth
from stocker_execution.saxo_client import SaxoClient
from stocker_execution.saxo_data import DataService
from stocker_execution.store import Store

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, config: FuturesConfig, store: Store, data: DataService | None = None):
        self.config, self.store = config, store
        store.bind(config.data_environment, config.execution_mode)
        self.root = Path(store.db.execute("PRAGMA database_list").fetchone()[2]).parent
        self.recorder = Recorder(config.recorder, self.root / config.data_environment / "events")
        self.data = data or DataService(
            config,
            SaxoClient(OAuth(config.data_environment, config.saxo, self.root)),
            self.recorder,
        )
        if data:
            self.recorder = data.recorder
        self.broker = PaperBroker(config, store, self.data)
        self.markets = self.data.markets
        self.worker_health = self.manager_health = self.web_health = "STARTING"
        self.stopping = False
        self.tasks: set[asyncio.Task[Any]] = set()
        self.owner: Any = None
        self.config_hash = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
        self.started_at = now()
        self.depth = self.recorder  # existing read-only dashboard recording access
        directory = self.root / config.data_environment
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        handler = RotatingFileHandler(directory / "slrno.log", maxBytes=2 * 1024**2, backupCount=3)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
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

    def report_failure(self, worker: str, exc: BaseException) -> None:
        # Exception strings can contain request URLs or authorization details.
        log.error("%s failed (%s)", worker, type(exc).__name__)

    async def cancel_tasks(self, tasks: set[asyncio.Task[Any]]) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def status(self) -> dict[str, Any]:
        return {
            "application": "SLRNO",
            "data_environment": self.config.data_environment,
            "execution_mode": self.config.execution_mode,
            "environment": self.config.data_environment,
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
            "providers": {
                "SAXO": "ACTIVE",
                "IBKR": "PARKED",
                "FMP": "INACTIVE",
                "EODHD": "INACTIVE",
            },
            "oauth": self.data.client.oauth.status,
            "session": self.data.session,
            "stream": {
                "connected": self.data.connected,
                "reconnects": self.data.reconnects,
                "last_message_id": self.data.last_message_id,
            },
            "market_data": {
                "owned_lines": len(self.data.subscriptions),
                "app_budget": 32,
                "rate_limits": self.data.client.rate_headers,
                "rest_queue": self.data.client.waiters,
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

    def overview(self) -> dict[str, Any]:
        cards = []
        active = self.store.active()
        recent = self.store.history(None, None, None)
        for market, state in self.markets.items():
            reason = state.problem or state.history_problem
            if not reason and market not in self.config.mappings:
                reason = "LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED"
            reason = reason or self.broker.entry_reason()
            value = state.price.value or {}
            quote = value.get("Quote") or {}
            current = (
                state.price.receipt is not None
                and 0 <= time.time() - state.price.receipt <= 5
                and not state.price.problem
            )
            identity = state.identity
            depth = state.price.depth(time.time())
            depth["last_receipt"] = (
                datetime.fromtimestamp(state.price.depth_receipt, UTC).isoformat()
                if state.price.depth_receipt
                else None
            )
            recording = (
                self.recorder.view(key(identity), time.time())
                if identity
                else {"state": "UNAVAILABLE", "prehistory_seconds": 0}
            )
            depth.update(pre_seconds=recording["prehistory_seconds"], target_pre_seconds=900)
            features = {}
            try:
                if state.bars:
                    features["rv15"] = prior_rv(
                        state.bars,
                        state.bars[-1].at.replace(second=0) + timedelta(minutes=1),
                    )
            except ValueError:
                pass
            cards.append(
                {
                    "market": market,
                    "contract": identity["symbol"] if identity else None,
                    "identity": identity,
                    "market_status": session_state(state.reference, now()),
                    "data_status": "CURRENT" if current else "STALE_OR_MISSING",
                    "strategy_state": "MONITOR_ONLY"
                    if market == "GC" and reason
                    else "BLOCKED"
                    if reason
                    else "MONITORING",
                    "entry_enabled": not reason and self.broker.armed and not self.pause,
                    "block_reason": reason,
                    "direction": "BUY CALL" if market == "CL" else "BUY PUT",
                    "conditions": features,
                    "trades": [self.trade_view(t) for t in active if t["market"] == market],
                    "l1": {
                        "status": "CURRENT" if current else "STALE_OR_MISSING",
                        "quote": quote,
                        "sizes": {side.lower(): size for side, size in state.price.sizes().items()},
                        "spread": quote["Ask"] - quote["Bid"]
                        if isinstance(quote.get("Ask"), (int, float))
                        and isinstance(quote.get("Bid"), (int, float))
                        else None,
                        "delay_minutes": quote.get("DelayedByMinutes"),
                        "provider_timestamp": value.get("LastUpdated"),
                        "last_receipt": datetime.fromtimestamp(state.price.receipt, UTC).isoformat()
                        if state.price.receipt
                        else None,
                    },
                    "l2": depth,
                    "book_flow": self.recorder.book_flow_view(key(identity), time.time())
                    if identity
                    else {"status": "UNAVAILABLE"},
                    "recorder": recording,
                    "capabilities": self.data.capability_view(state),
                    "option": {
                        "root": state.option_root,
                        "discovered": len(state.option_space),
                        "board": state.option_board,
                        "quotes": [
                            {"identity": ident, "price": price.value, "receipt": price.receipt}
                            for ident, price in self.data.options.values()
                            if ident["market"] == market
                        ],
                    },
                    "chart": [{"at": b.at.isoformat(), "close": b.close} for b in state.bars[-90:]],
                    "signals": [
                        {k: s[k] for k in ("id", "signal_at", "decision", "reason")}
                        for s in recent
                        if s["market"] == market
                    ][:8],
                    "next_time": next_clock(now()).isoformat(),
                    "next_market_time": None,
                    "exchange_trade_date": None,
                    "rule_version": RULE_VERSION,
                    "diagnostic": "L2 observation only",
                    "details": {
                        "signal_contract": identity,
                        "reference_sessions": len(state.references),
                        "entry_clocks": "09:00–16:00 America/New_York weekdays; NG 13:00 veto",
                        "exit_anchor": "Original opportunity + 60 minutes",
                        "options_block": "Verify actual 0DTE expiry, product and delta tolerance",
                        "recording": recording,
                    },
                }
            )
        return {"system": self.status(), "markets": cards, "pnl": self.store.economics()}

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
                reason = str(event["veto"])
                inputs: dict[str, float] = {}
                try:
                    if not reason:
                        if (at - clock).total_seconds() > 20:
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
                state.last_clock = clock
                # Capture an observed frozen clock even for veto, capacity, option or data skips.
                event["skip_reason"] = reason
                observed = self.store.observe(event, reason, inputs)
                if not observed:
                    continue
                option_keys = [
                    key(i)
                    for i, _ in self.data.options.values()
                    if i["market"] == state.market and i["underlying_uic"] == state.identity["uic"]
                ]
                capture = self.recorder.trigger(
                    key(state.identity), event, time.time(), option_keys
                )
                self.store.depth_capture(str(event["id"]), capture)
                if not reason:
                    reason = "ENTRIES_PAUSED" if self.pause else self.broker.entry_reason()
                if not reason:
                    try:
                        plan = await self.broker.prepare(event, state, inputs)
                        reason = await self.broker.enter(event, plan)
                    except (ValueError, KeyError) as exc:
                        reason = str(exc)
                if reason:
                    self.store.decision(str(event["id"]), "SKIPPED", reason)
                self.recorder.annotate(str(event["id"]), {"skip_reason": reason, "inputs": inputs})

    async def manager(self) -> None:
        self.manager_health = "RUNNING"
        while not self.stopping:
            try:
                await self.broker.manage()
            except Exception as exc:
                self.broker.reconciled = False
                self.report_failure("management", exc)
            await asyncio.sleep(2)

    async def history_worker(self) -> None:
        while not self.stopping:
            try:
                await self.data.release_unused_options(
                    {json.loads(r["plan"])["option"]["uic"] for r in self.store.active()}
                )
            except Exception as exc:
                self.report_failure("option-subscription-cleanup", exc)
            for state in self.markets.values():
                if self.data.connected and time.monotonic() - state.history_checked >= 60:
                    try:
                        await self.data.history(state)
                    except Exception as exc:
                        state.history_problem = "SAXO_HISTORY_UNAVAILABLE"
                        state.history_checked = time.monotonic()
                        self.report_failure("history", exc)
            await asyncio.sleep(5)

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
            }
        )
        try:
            while not self.stopping:
                try:
                    self.recorder.tick(time.time())
                except Exception as exc:
                    self.recorder.problem = "RECORDER_FAILED"
                    self.report_failure("recorder", exc)
                try:
                    await self.decisions()
                except Exception as exc:
                    self.broker.armed = False
                    self.broker.problem = "DECISION_WORKER_ERROR_REVIEW_REQUIRED"
                    self.report_failure("decisions", exc)
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
        log.removeHandler(self.log_handler)
        self.log_handler.close()
        if self.owner:
            self.owner.close()
            self.owner = None
