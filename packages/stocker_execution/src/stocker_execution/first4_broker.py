"""Listed contracts and real IBKR PAPER orders. No valuation or simulated fills."""

import asyncio
import json
import logging
import math
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from typing import Any
from zoneinfo import ZoneInfo

from ib_async import (
    ComboLeg,
    Contract,
    ExecutionFilter,
    LimitOrder,
    MarketOrder,
    Option,
    RequestError,
)

from stocker_execution.first4_config import PAPER_ACCOUNT, First4Config
from stocker_execution.first4_requests import First4IB
from stocker_execution.first4_store import Store

log = logging.getLogger(__name__)


def now() -> datetime:
    return datetime.now(UTC)


def listed_strike(strikes: list[float], anchor: float, right: str) -> float:
    reference = Decimal(str(anchor))
    target = reference * Decimal(".98" if right == "P" else "1.02")
    values = [Decimal(str(x)) for x in strikes if math.isfinite(x) and x > 0]
    values = [x for x in values if (x < reference if right == "P" else x > reference)]
    values = [x for x in values if abs(x - target) <= reference * Decimal(".01")]
    if not values:
        raise ValueError("No listed strike for configured mapping")
    return float(min(values, key=lambda x: (abs(x - target), x if right == "P" else -x)))


def listed_expiry(expiries: dict[str, datetime], baseline: datetime) -> str:
    local_day = baseline.astimezone(ZoneInfo("America/New_York")).date()
    target = baseline + timedelta(minutes=2880)
    eligible = [
        e
        for e, stamp in expiries.items()
        if stamp.astimezone(ZoneInfo("America/New_York")).date() > local_day
        and abs(stamp - target) <= timedelta(days=1)
    ]
    if not eligible:
        raise ValueError("EXPIRY_UNAVAILABLE_WITHIN_ONE_CALENDAR_DAY")
    return min(eligible, key=lambda e: (abs(expiries[e] - target), -expiries[e].timestamp()))


def expiration_at(details: Any) -> datetime:
    # Use the broker's actual contract time; never substitute the stock close.
    if not details.realExpirationDate or not details.lastTradeTime or not details.timeZoneId:
        raise ValueError("OPTION_EXPIRY_TIME_UNAVAILABLE")
    return (
        datetime.strptime(
            details.realExpirationDate + " " + details.lastTradeTime, "%Y%m%d %H:%M:%S"
        )
        .replace(tzinfo=ZoneInfo(details.timeZoneId))
        .astimezone(UTC)
    )


def limit_price(price: float, tick: float) -> float:
    if not all(math.isfinite(x) and x > 0 for x in (price, tick)):
        raise ValueError("INVALID_EXECUTABLE_PRICE_INCREMENT")
    return float(
        (Decimal(str(price)) / Decimal(str(tick))).to_integral_value(rounding=ROUND_FLOOR)
        * Decimal(str(tick))
    )


class PaperBroker:
    def __init__(self, config: First4Config, store: Store, ib: Any = None):
        self.config, self.store = config, store
        self.ib: Any = ib if ib is not None else First4IB()
        self.ib.RaiseRequestErrors = True
        self.reconciled = False
        self.problem = "NOT_CONNECTED"
        self.entry_blocker = "NOT_RECONCILED"
        self.chains: dict[int, Any] = {}
        self.opening_verified_session: str | None = None
        self.reconciliation_lock = asyncio.Lock()
        self.upstream_available = True
        self.upstream_status = "UNVERIFIED"
        self.last_quote_at: datetime | None = None
        self.data_problem = ""
        self.unavailable_farms: set[int] = set()
        self.data_generation = 0
        self.management_block = ""
        self.fatal_error = ""
        self.persistence_failed = False
        self.operator_exceptions: dict[str, str] = {}
        self.ib.disconnectedEvent += self.disconnected
        self.ib.execDetailsEvent += self.persist_event(self.fill)
        self.ib.commissionReportEvent += self.persist_event(self.commission)
        self.ib.orderStatusEvent += self.persist_event(self.order_status)
        self.ib.errorEvent += self.error
        self.ib.positionEvent += self.persist_event(self.position)

    def persist_event(self, callback: Callable[..., None]) -> Callable[..., None]:
        def receive(*args: Any) -> None:
            try:
                callback(*args)
            except sqlite3.Error as exc:
                self.fatal_error = "BROKER_EVENT_PERSISTENCE_FAILED: " + str(exc)
                self.persistence_failed = True
                self.reconciled = False
                log.exception("FIRST4 broker callback %s could not persist", callback.__name__)

        return receive

    def disconnected(self, *args: Any) -> None:
        self.opening_verified_session = None
        self.reconciled = False
        self.problem = "DISCONNECTED_RECONCILIATION_REQUIRED"
        self.data_generation += 1
        self.last_quote_at = None

    def entries_armed(self) -> bool:
        return not self.entry_reason()

    def entry_reason(self, session: str | None = None) -> str:
        day = now().astimezone(ZoneInfo("America/New_York")).date()
        if self.fatal_error or self.management_block:
            return self.fatal_error or self.management_block
        if not self.ib.isConnected() or not self.upstream_available or self.data_problem:
            return self.data_problem or "UPSTREAM_OR_API_UNAVAILABLE"
        if not self.reconciled:
            return "NOT_RECONCILED"
        try:
            self.guard()
        except ValueError as exc:
            return str(exc)
        if self.entry_blocker:
            return self.entry_blocker
        if self.store.get_meta("paused", False):
            return "ENTRIES_PAUSED"
        state = self.store.db.execute(
            "SELECT blocked FROM first4_sessions WHERE session=?", (session or day.isoformat(),)
        ).fetchone()
        if state and state["blocked"]:
            return str(state["blocked"])
        if not self.config.armed and (
            self.config.arm_after_quote_check_on != day
            or self.opening_verified_session != day.isoformat()
            or not self.reconciled
        ):
            return "PAPER entries are unarmed"
        return ""

    def require_entries(self) -> None:
        self.config.require_settings()
        if reason := self.entry_reason():
            raise ValueError(reason)

    def position(self, position: Any) -> None:
        if position.account != PAPER_ACCOUNT:
            return
        con_id = position.contract.conId
        owned = self.owned_quantities()
        if con_id in owned:
            with self.store.db:
                self.store.db.execute(
                    "INSERT OR REPLACE INTO first4_positions VALUES (?,?,?)",
                    (
                        con_id,
                        float(position.position),
                        json.dumps(
                            {
                                "owned_quantity": owned[con_id],
                                "average_cost": position.avgCost,
                            }
                        ),
                    ),
                )

    def guard(self) -> None:
        if self.config.environment != "PAPER" or self.config.expected_account != PAPER_ACCOUNT:
            raise ValueError("PAPER account identity mismatch")
        if not self.ib.isConnected() or self.ib.managedAccounts() != [PAPER_ACCOUNT]:
            self.reconciled = False
            raise ValueError("Verified PAPER account is not connected")
        if self.ib.client.clientId != self.config.client_id:
            raise ValueError("Unexpected FIRST4 execution client")
        if not self.upstream_available:
            raise ValueError("UPSTREAM_UNAVAILABLE")
        if self.fatal_error:
            raise ValueError(self.fatal_error)
        if not self.reconciled:
            raise ValueError("Broker reconciliation required: " + self.problem)

    async def connect(self) -> None:
        self.reconciled = False
        self.upstream_available = True
        self.upstream_status = "UNVERIFIED"
        self.data_problem = ""
        self.unavailable_farms.clear()
        await self.ib.connectAsync(
            self.config.host,
            self.config.port,
            clientId=self.config.client_id,
            timeout=10,
            readonly=False,
            account=PAPER_ACCOUNT,
            raiseSyncErrors=True,
        )
        if self.ib.managedAccounts() != [PAPER_ACCOUNT]:
            self.ib.disconnect()
            raise ValueError("Connected account differs from verified PAPER account")
        self.ib.reqMarketDataType(1)
        await self.reconcile()

    def fill(self, trade: Any, fill: Any) -> None:
        e, c = fill.execution, fill.contract
        if e.acctNumber != PAPER_ACCOUNT or c.secType != "OPT":
            return  # BAG summaries are not leg executions.
        reference = e.orderRef
        if not self.store.db.execute(
            "SELECT 1 FROM first4_orders WHERE reference=?", (reference,)
        ).fetchone():
            return
        with self.store.db:
            self.store.db.execute(
                "INSERT OR IGNORE INTO first4_fills "
                "(exec_id,reference,con_id,quantity,price,side,multiplier,time,commission) "
                "VALUES (?,?,?,?,?,?,?,?,NULL)",
                (
                    e.execId,
                    reference,
                    c.conId,
                    float(e.shares),
                    float(e.price),
                    e.side,
                    float(c.multiplier),
                    e.time.isoformat(),
                ),
            )
            # Correction reports share a stem and increase the final numeric
            # component. Retain all reports, but account only the latest revision.
            base, separator, suffix = e.execId.rpartition(".")
            if separator and suffix.isdigit():
                revisions = self.store.db.execute(
                    "SELECT exec_id,reference,con_id FROM first4_fills "
                    "WHERE exec_id>=? AND exec_id<?",
                    (base + ".", base + ".\uffff"),
                ).fetchall()
                versions = []
                for row in revisions:
                    tail = row["exec_id"].removeprefix(base + ".")
                    if not tail.isdigit():
                        continue
                    if row["reference"] != reference or row["con_id"] != c.conId:
                        self.fatal_error = "EXECUTION_CORRECTION_OWNERSHIP_MISMATCH"
                        raise ValueError(self.fatal_error)
                    versions.append((int(tail), row["exec_id"]))
                latest = max(versions)[0]
                for version, exec_id in versions:
                    if version < latest:
                        self.store.db.execute(
                            "UPDATE first4_fills SET superseded=1 WHERE exec_id=? AND superseded=0",
                            (exec_id,),
                        )

    def commission(self, trade: Any, fill: Any, report: Any) -> None:
        if (
            report.currency != "USD"
            or not math.isfinite(report.commission)
            or abs(report.commission) >= 1e100
        ):
            return
        with self.store.db:
            self.store.db.execute(
                "UPDATE first4_fills SET commission=? WHERE exec_id=? AND commission IS NOT ?",
                (report.commission, report.execId, report.commission),
            )

    def order_status(self, trade: Any) -> None:
        with self.store.db:
            self.store.db.execute(
                "UPDATE first4_orders SET status=?,perm_id=? WHERE reference=? AND order_id=?",
                (
                    trade.orderStatus.status,
                    trade.orderStatus.permId or trade.order.permId,
                    trade.order.orderRef,
                    trade.order.orderId,
                ),
            )

    def error(self, request_id: int, code: int, message: str, contract: Any = None) -> None:
        if code in {1100, 1300}:
            self.upstream_available = False
            self.upstream_status = "LOST"
            self.disconnected()
        elif code in {1101, 1102}:
            self.upstream_available = True
            self.upstream_status = "RESTORED_RECONCILIATION_REQUIRED"
            self.reconciled = False
            self.opening_verified_session = None
            if code == 1101:
                # All active request consumers reject this generation. Their
                # finally blocks cancel subscriptions; the manager reconnects
                # once to restore IB's account subscriptions. No anchor replay.
                self.data_generation += 1
                self.ib.disconnect()
        elif code in {2103, 2105, 2157}:
            self.unavailable_farms.add(code)
            self.data_problem = f"DATA_FARM_UNAVAILABLE:{sorted(self.unavailable_farms)}"
            if code == 2103:
                self.data_generation += 1
        elif code in {2104, 2106, 2158}:
            self.unavailable_farms.discard({2104: 2103, 2106: 2105, 2158: 2157}[code])
            self.data_problem = (
                f"DATA_FARM_UNAVAILABLE:{sorted(self.unavailable_farms)}"
                if self.unavailable_farms
                else ""
            )
            return
        if code in {2107, 2108, 2158}:
            return
        self.report_error(
            "broker_error",
            {"time": now().isoformat(), "request_id": request_id, "code": code, "message": message},
        )

    def report_error(self, key: str, detail: dict[str, Any]) -> None:
        log.error("FIRST4 %s: %s", key, detail)
        try:
            self.store.set_meta(key, detail)
        except sqlite3.Error as exc:
            self.fatal_error = "LEDGER_UNAVAILABLE: " + str(exc)
            self.persistence_failed = True
            self.reconciled = False
            log.exception("FIRST4 could not persist %s", key)

    def owned_quantities(self, allocation: str | None = None) -> dict[int, float]:
        args: tuple[str, ...]
        if allocation is not None:
            query = (
                "SELECT con_id,SUM(quantity*CASE side WHEN 'BOT' THEN 1 ELSE -1 END) AS q "
                "FROM first4_effective_fills WHERE reference>=? AND reference<? GROUP BY con_id"
            )
            args = (allocation, allocation + "\uffff")
        else:
            query = (
                "SELECT f.con_id,"
                "SUM(f.quantity*CASE f.side WHEN 'BOT' THEN 1 ELSE -1 END) AS q "
                # CROSS JOIN fixes the driving table to outstanding obligations;
                # SQLite otherwise may choose to scan every historical fill.
                "FROM first4_orders e CROSS JOIN first4_orders o "
                "ON o.session=e.session AND o.symbol=e.symbol "
                "CROSS JOIN first4_effective_fills f ON f.reference=o.reference "
                "WHERE e.role='ENTRY' AND e.obligation_done=0 GROUP BY f.con_id"
            )
            args = ()
        return {r["con_id"]: r["q"] for r in self.store.db.execute(query, args)}

    def deadline_blocker(self) -> str:
        for order in self.store.pending_deadlines():
            payload = json.loads(order["payload"])
            deadline = payload.get("entry_deadline_at")
            if (
                deadline
                and now() >= datetime.fromisoformat(deadline)
                and not payload.get("deadline_reconciled")
            ):
                return "ENTRY_DEADLINE_AWAITING_CANCEL_FILL_RECONCILIATION"
        return ""

    async def reconcile(self, require_flat: bool = False) -> None:
        # IB uses fixed request keys for positions/open/completed orders.
        # Two consumers on this connection must never overwrite those futures.
        async with self.reconciliation_lock:
            try:
                await self._reconcile(require_flat)
            except (asyncio.CancelledError, TimeoutError):
                # Fixed request keys cannot be reused while an old response is
                # in flight. Disconnect resets the wrapper before reconnect.
                self.disconnected()
                self.ib.disconnect()
                raise

    async def _reconcile(self, require_flat: bool) -> None:
        self.reconciled = False
        if not self.ib.isConnected() or self.ib.managedAccounts() != [PAPER_ACCOUNT]:
            raise ValueError("PAPER identity mismatch during reconciliation")
        opens = await self.ib.reqAllOpenOrdersAsync()
        completed = await self.ib.reqCompletedOrdersAsync(apiOnly=False)
        fills = await self.ib.reqExecutionsAsync(ExecutionFilter(acctCode=PAPER_ACCOUNT))
        for f in fills:
            self.fill(None, f)
            if f.commissionReport.execId:
                self.commission(None, f, f.commissionReport)
        positions = await self.ib.reqPositionsAsync()
        matched = set()
        unknown_orders = []
        for t in opens:
            row = self.store.db.execute(
                "SELECT * FROM first4_orders WHERE reference=?", (t.order.orderRef,)
            ).fetchone()
            if row:
                if t.order.account != PAPER_ACCOUNT or t.order.clientId != self.config.client_id:
                    raise ValueError("Owned order account/client mismatch")
                self.order_status(t)
                matched.add(row["reference"])
            elif t.order.account == PAPER_ACCOUNT and not t.isDone():
                unknown_orders.append(t.order.orderId)
        for t in completed:
            row = self.store.db.execute(
                "SELECT * FROM first4_orders WHERE reference=?", (t.order.orderRef,)
            ).fetchone()
            if row:
                # TWS completedOrder has no clientId/orderId. Its permId is on Order,
                # not OrderStatus; do not overwrite the durable API order identifier.
                permanent_id = t.order.permId
                if (
                    t.order.account != PAPER_ACCOUNT
                    or not permanent_id
                    or (row["perm_id"] and row["perm_id"] != permanent_id)
                ):
                    raise ValueError("Completed order identity mismatch")
                with self.store.db:
                    self.store.db.execute(
                        "UPDATE first4_orders SET status=?,perm_id=? WHERE reference=?",
                        (t.orderStatus.status, permanent_id, row["reference"]),
                    )
                matched.add(row["reference"])
        terminal = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
        uncertain = [
            o["reference"]
            for o in self.store.rows("orders")
            if o["status"] not in terminal and o["reference"] not in matched
        ]
        for order in self.store.rows("orders"):
            if order["status"] != "Filled":
                continue
            payload = json.loads(order["payload"])
            ids = (
                [payload[key]["conId"] for key in ("put", "call")]
                if order["role"] == "ENTRY"
                else payload["exit_con_ids"]
            )
            expected = payload["quantity"] if order["role"] == "ENTRY" else payload["exit_quantity"]
            executions = self.store.order_fills(order["reference"])
            if any(
                sum(f["quantity"] for f in executions if f["con_id"] == con_id) != expected
                for con_id in ids
            ):
                uncertain.append("FILLED_ORDER_MISSING_LEG_EXECUTIONS:" + order["reference"])
        actual = {
            p.contract.conId: p for p in positions if p.account == PAPER_ACCOUNT and p.position
        }
        owned = self.owned_quantities()
        unknown = sorted(set(actual) - set(owned))
        self.entry_blocker = (
            f"UNOWNED_BROKER_POSITIONS:{unknown}"
            if unknown
            else f"UNOWNED_PENDING_ORDERS:{unknown_orders}"
            if unknown_orders
            else self.deadline_blocker()
        )
        with self.store.db:
            self.store.db.execute("DELETE FROM first4_positions")
            for con_id, quantity in owned.items():
                p = actual.get(con_id)
                broker_quantity = float(p.position) if p else 0
                self.store.db.execute(
                    "INSERT INTO first4_positions VALUES (?,?,?)",
                    (
                        con_id,
                        broker_quantity,
                        json.dumps(
                            {"owned_quantity": quantity, "average_cost": p.avgCost if p else None}
                        ),
                    ),
                )
                if broker_quantity != quantity:
                    uncertain.append(
                        f"POSITION_MISMATCH:{con_id}:owned={quantity}:broker={broker_quantity}"
                    )
        if uncertain:
            self.problem = "; ".join(uncertain)
            raise ValueError(self.problem)
        if require_flat and (opens or actual):
            raise ValueError("OPENING_CHECK_REQUIRES_FLAT_ACCOUNT_AND_NO_OPEN_ORDERS")
        # Unrelated contracts/orders are never cancelled or managed by FIRST4.
        self.reconciled, self.problem = True, ""
        if self.upstream_available:
            self.upstream_status = "RECONCILED_DATA_CHECKS_PER_REQUEST"
        self.store.set_meta("reconciled_at", now().isoformat())

    async def chain(self, underlying: Any) -> Any:
        if underlying.conId not in self.chains:
            self.chains[underlying.conId] = await self.ib.reqSecDefOptParamsAsync(
                underlying.symbol, "", "STK", underlying.conId
            )
        return self.chains[underlying.conId]

    async def contracts(
        self, underlying: Any, anchor: float, baseline: datetime
    ) -> tuple[Any, Any, Any]:
        chains = [
            x
            for x in await self.chain(underlying)
            if x.exchange == "SMART"
            and x.tradingClass == underlying.symbol
            and x.multiplier == "100"
        ]
        if len(chains) != 1:
            raise ValueError("AMBIGUOUS_OPTION_TRADING_CLASS_OR_MULTIPLIER")
        chain = chains[0]

        async def qualify(expiry: str, right: str) -> Any:
            strike = listed_strike(list(chain.strikes), anchor, right)
            details = await self.ib.reqContractDetailsAsync(
                Option(
                    underlying.symbol,
                    expiry,
                    strike,
                    right,
                    "SMART",
                    multiplier=chain.multiplier,
                    currency="USD",
                    tradingClass=chain.tradingClass,
                )
            )
            if not details:
                if right == "P":
                    return None
                raise ValueError("OPTION_CONTRACT_UNAVAILABLE_OR_AMBIGUOUS")
            if len(details) != 1 or details[0].underConId != underlying.conId:
                raise ValueError("OPTION_CONTRACT_UNAVAILABLE_OR_AMBIGUOUS")
            d = details[0]
            actual = d.contract
            osi = f"{underlying.symbol:<6}{expiry[2:]}{right}{round(strike * 1000):08d}"
            if (
                actual.secType != "OPT"
                or actual.currency != "USD"
                or actual.conId <= 0
                or actual.multiplier != "100"
                or actual.tradingClass != underlying.symbol
                or actual.localSymbol != osi
                or actual.strike != strike
                or actual.right != right
                or actual.lastTradeDateOrContractMonth != expiry
                or d.realExpirationDate != expiry
                or not {"LMT", "GTD"}.issubset(set(d.orderTypes.split(",")))
            ):
                raise ValueError("NONSTANDARD_OR_UNVERIFIED_OPTION_CONTRACT")
            expiration_at(d)
            return d

        target_day = (
            (baseline + timedelta(minutes=2880)).astimezone(ZoneInfo("America/New_York")).date()
        )
        local_day = baseline.astimezone(ZoneInfo("America/New_York")).date()
        dates = sorted(
            e
            for e in chain.expirations
            if datetime.strptime(e, "%Y%m%d").date() > local_day
            and abs((datetime.strptime(e, "%Y%m%d").date() - target_day).days) <= 1
        )

        # At most three specific contracts, not full option-chain downloads. Unknown
        # expiry metadata rejects the candidate rather than widening the search.
        async def permitted_put(expiry: str) -> Any:
            try:
                return await qualify(expiry, "P")
            except RequestError as exc:
                if exc.code != 200 or not exc.message.startswith("No security definition"):
                    raise
                return None

        puts = await asyncio.gather(*(permitted_put(e) for e in dates), return_exceptions=True)
        for result in puts:
            if isinstance(result, BaseException):
                raise result
        by_expiry = {e: d for e, d in zip(dates, puts, strict=True) if d is not None}
        expiry = listed_expiry({e: expiration_at(d) for e, d in by_expiry.items()}, baseline)
        if len(by_expiry) != len(dates) and expiration_at(by_expiry[expiry]) != (
            baseline + timedelta(minutes=2880)
        ):
            # An unavailable strike does not prove its listed expiry was farther
            # away. Without its actual timestamp only an exact target match is
            # provably unbeatable. Never infer an expiry time from a date alone.
            raise ValueError("EXPIRY_SELECTION_AMBIGUOUS_UNAVAILABLE_ALTERNATIVE")
        legs = [by_expiry[expiry], await qualify(expiry, "C")]
        if expiration_at(legs[0]) != expiration_at(legs[1]):
            raise ValueError("OPTION_EXPIRY_TIME_MISMATCH")
        combo = Contract(
            secType="BAG",
            symbol=underlying.symbol,
            currency="USD",
            exchange="SMART",
            comboLegs=[
                ComboLeg(conId=d.contract.conId, ratio=1, action="BUY", exchange="SMART")
                for d in legs
            ],
        )
        return legs[0], legs[1], combo

    async def quotes(
        self, contracts: list[Any], deadline: datetime, quantity: float = 1, side: str = "BUY"
    ) -> list[dict[str, Any]]:
        generation = self.data_generation
        subscriptions = []
        states: list[dict[str, Any]] = [{"bid_at": None, "ask_at": None} for _ in contracts]
        try:
            for contract, state in zip(contracts, states, strict=True):
                t = self.ib.reqMktData(contract, "", False, False)

                def update(ticker: Any, state: dict[str, Any] = state) -> None:
                    for tick in ticker.ticks:
                        if tick.tickType == 1:
                            state["bid_at"] = tick.time
                        elif tick.tickType == 2:
                            state["ask_at"] = tick.time

                t.updateEvent += update
                subscriptions.append((contract, t, update))
            while now() < deadline:
                self.guard()
                if generation != self.data_generation or 2103 in self.unavailable_farms:
                    raise ValueError("QUOTE_DATA_INTERRUPTED")
                result = []
                for (_, t, _), s in zip(subscriptions, states, strict=True):
                    stamps = [s["bid_at"], s["ask_at"]]
                    if t.marketDataType != 1 or any(x is None for x in stamps):
                        break
                    age = max((now() - x).total_seconds() for x in stamps)
                    if (
                        not 0 <= age <= self.config.number("quote_max_age_seconds")
                        or not all(math.isfinite(x) for x in (t.bid, t.ask, t.bidSize, t.askSize))
                        or not 0 < t.bid <= t.ask
                        or min(t.bidSize, t.askSize) <= 0
                        or (t.askSize if side == "BUY" else t.bidSize) < quantity
                    ):
                        break
                    result.append(
                        {
                            "bid": t.bid,
                            "ask": t.ask,
                            "bid_size": t.bidSize,
                            "ask_size": t.askSize,
                            "bid_at": s["bid_at"].isoformat(),
                            "ask_at": s["ask_at"].isoformat(),
                        }
                    )
                if len(result) == len(contracts):
                    self.last_quote_at = now()
                    return result
                await asyncio.sleep(0.02)
            raise ValueError("OPTION_QUOTES_INVALID_STALE_OR_UNAVAILABLE")
        finally:
            for contract, t, callback in subscriptions:
                t.updateEvent -= callback
                self.ib.cancelMktData(contract)

    def submit(
        self,
        event: dict[str, Any],
        contract: Any,
        order: Any,
        payload: dict[str, Any],
        role: str,
        suffix: str = "",
    ) -> Any:
        self.guard()
        if role == "ENTRY":
            self.require_entries()
            if reason := self.entry_reason(event["session"]):
                raise ValueError(reason)
            for trade in self.ib.openTrades():
                if trade.order.account == PAPER_ACCOUNT and not trade.isDone():
                    row = self.store.db.execute(
                        "SELECT order_id FROM first4_orders WHERE reference=?",
                        (trade.order.orderRef,),
                    ).fetchone()
                    if (
                        not row
                        or row[0] != trade.order.orderId
                        or trade.order.clientId != self.config.client_id
                    ):
                        raise ValueError("UNOWNED_PENDING_ORDERS")
            blocker = self.entry_blocker or self.deadline_blocker()
            if self.store.get_meta("paused", False) or blocker:
                raise ValueError(blocker or "ENTRIES_PAUSED")
            actual = {
                p.contract.conId: float(p.position)
                for p in self.ib.positions(PAPER_ACCOUNT)
                if p.position
            }
            owned = self.owned_quantities()
            if any(actual.get(k, 0) != owned.get(k, 0) for k in actual.keys() | owned.keys()):
                raise ValueError("ENTRY_POSITION_RECONCILIATION_REQUIRED")
            entry = datetime.fromisoformat(event["entry_at"])
            close = datetime.fromisoformat(event["close_at"])
            exit_at = close - timedelta(seconds=self.config.number("exit_seconds_before_close"))
            if entry >= exit_at or now() >= exit_at:
                raise ValueError("BASELINE_OUTSIDE_TRADABLE_HOLDING_WINDOW")
            if (
                not entry
                <= now()
                < entry + timedelta(seconds=self.config.number("entry_deadline_seconds"))
            ):
                raise ValueError("ENTRY_OUTSIDE_BASELINE_EXECUTION_WINDOW")
            admission = self.store.db.execute(
                "SELECT slot,entry_at FROM first4_events WHERE session=? AND symbol=?",
                (event["session"], event["symbol"]),
            ).fetchone()
            if (
                not admission
                or admission["slot"] not in (1, 2, 3, 4)
                or admission["slot"] != event["slot"]
                or admission["entry_at"] != event["entry_at"]
            ):
                raise ValueError("ENTRY_REQUIRES_PERSISTED_FIRST4_ADMISSION")
            if contract.secType != "BAG" or order.action != "BUY" or order.totalQuantity != 1:
                raise ValueError("ENTRY_REQUIRES_ONE_DEBIT_COMBINATION")
            if not math.isfinite(order.lmtPrice) or not 0 < Decimal(
                str(order.lmtPrice)
            ) * 100 <= self.config.number("premium_budget_usd"):
                raise ValueError("PREMIUM_BUDGET_EXCEEDED")
            payload = {**payload, "allocation_usd": 260, "fee_reserve_usd": 10}
        elif role == "EXIT":
            if now() >= datetime.fromisoformat(event["close_at"]):
                raise ValueError("MISSED_SESSION_CLOSE_EXIT_REQUIRES_OPERATOR")
            # Quote retrieval awaited broker events. Recheck the actual remaining legs
            # at the socket boundary so a stale close can never open a short option.
            actual = {p.contract.conId: float(p.position) for p in self.ib.positions(PAPER_ACCOUNT)}
            owned = self.owned_quantities()
            allocated = self.owned_quantities(f"F4:{event['session']}:{event['slot']}:")
            ids = (
                [leg.conId for leg in contract.comboLegs]
                if contract.secType == "BAG"
                else [contract.conId]
            )
            if order.action != "SELL" or any(
                actual.get(con_id, 0) != owned.get(con_id, 0)
                or not 0 < order.totalQuantity <= allocated.get(con_id, 0)
                for con_id in ids
            ):
                self.reconciled = False
                raise ValueError("EXIT_POSITION_CHANGED_DURING_PREPARATION")
        order.account = PAPER_ACCOUNT
        order.orderId = self.ib.client.getReqId()
        payload = {**payload, "submitted_at": now().isoformat()}
        order.orderRef = self.store.reserve_order(event, role, order.orderId, payload, suffix)
        # Durable reservation precedes the socket write. Ambiguous writes never retry entry.
        self.guard()
        trade = self.ib.placeOrder(contract, order)
        self.order_status(trade)
        return trade

    async def enter(self, event: dict[str, Any], underlying: Any, anchor: float) -> None:
        self.require_entries()
        baseline = datetime.fromisoformat(event["entry_at"])
        deadline = baseline + timedelta(seconds=self.config.number("entry_deadline_seconds"))
        p, c, combo = await self.contracts(underlying, anchor, baseline)
        async with asyncio.TaskGroup() as group:
            quote_task = group.create_task(self.quotes([p.contract, c.contract], deadline))
            tick_task = group.create_task(self.combo_tick(combo, deadline))
        quotes, tick = quote_task.result(), tick_task.result()
        # reqContractDetails does not encode BAG legs. Use actual qualified leg
        # size constraints and the BAG market-data tickReqParams price increment.
        increments = [Decimal(str(d.sizeIncrement)) for d in (p, c)]
        if any(not i.is_finite() or i <= 0 for i in increments):
            raise ValueError("OPTION_QUANTITY_RULES_UNAVAILABLE")
        scale = 10 ** max(0, *(-int(i.as_tuple().exponent) for i in increments))
        increment = math.lcm(*(int(i * scale) for i in increments)) / scale
        minimum = max(float(p.minSize), float(c.minSize))
        if not all(math.isfinite(x) and x > 0 for x in (tick, increment, minimum)):
            raise ValueError("COMBO_EXECUTION_RULES_INVALID")
        # Exactly one pair, with both qualified contracts permitting a quantity of one.
        if minimum > 1 or Decimal(1) % Decimal(str(increment)):
            raise ValueError("ONE_PACKAGE_QUANTITY_UNSUPPORTED")
        ask = float(sum(Decimal(str(q["ask"])) for q in quotes))
        limit = limit_price(ask, tick)
        multiplier = float(p.contract.multiplier)
        if multiplier != 100 or c.contract.multiplier != p.contract.multiplier:
            raise ValueError("STANDARD_MULTIPLIER_REQUIRED")
        quantity = 1
        if (
            not 0
            < Decimal(str(limit)) * Decimal(str(multiplier))
            <= self.config.number("premium_budget_usd")
        ):
            raise ValueError("PREMIUM_BUDGET_EXCEEDED")
        if any(
            not 0
            <= (now() - datetime.fromisoformat(q[k])).total_seconds()
            <= self.config.number("quote_max_age_seconds")
            for q in quotes
            for k in ("bid_at", "ask_at")
        ):
            raise ValueError("QUOTES_EXPIRED_DURING_PREPARATION")
        payload = {
            "anchor": anchor,
            "anchor_at": baseline.isoformat(),
            "put": p.contract.dict(),
            "call": c.contract.dict(),
            "quantity": quantity,
            "multiplier": multiplier,
            "limit": limit,
            "quotes": quotes,
            "entry_deadline_at": deadline.isoformat(),
            "target_expiry_at": (baseline + timedelta(minutes=2880)).isoformat(),
            "actual_expiry_at": expiration_at(p).isoformat(),
            "expiry_target_difference_seconds": (
                expiration_at(p) - baseline - timedelta(minutes=2880)
            ).total_seconds(),
            "remaining_expiry_seconds_at_submission": (expiration_at(p) - now()).total_seconds(),
            "target_put_strike": anchor * 0.98,
            "target_call_strike": anchor * 1.02,
            "put_strike_difference": p.contract.strike - anchor * 0.98,
            "call_strike_difference": c.contract.strike - anchor * 1.02,
            "quoted_entry_ask_usd": sum(q["ask"] for q in quotes) * multiplier,
            "exit_at": (
                datetime.fromisoformat(event["close_at"])
                - timedelta(seconds=self.config.number("exit_seconds_before_close"))
            ).isoformat(),
        }
        order = LimitOrder(
            "BUY",
            quantity,
            limit,
            tif="GTD",
            goodTillDate=deadline.strftime("%Y%m%d-%H:%M:%S"),
            outsideRth=False,
        )
        self.submit(event, combo, order, payload, "ENTRY")
        self.store.outcome(event, "ORDER_SUBMITTED", payload)

    async def cancel_due_entries(self) -> None:
        """GTD is backed by explicit owned-order cancellation and broker reconciliation."""
        terminal = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
        refreshed = False
        for entry in self.store.pending_deadlines():
            payload = json.loads(entry["payload"])
            deadline = datetime.fromisoformat(payload["entry_deadline_at"])
            if now() < deadline or payload.get("deadline_reconciled"):
                continue
            self.guard()
            if entry["status"] not in terminal:
                trades = [t for t in self.ib.openTrades() if t.order.orderRef == entry["reference"]]
                for trade in trades:
                    if (
                        trade.order.account != PAPER_ACCOUNT
                        or trade.order.clientId != self.config.client_id
                    ):
                        raise ValueError("Cancellation account/client mismatch")
                    if not trade.isDone() and trade.orderStatus.status != "PendingCancel":
                        self.ib.cancelOrder(trade.order)
                self.problem = "ENTRY_DEADLINE_AWAITING_CANCEL_FILL_RECONCILIATION"
                self.entry_blocker = self.problem
                if not trades:
                    self.reconciled = False
                continue
            if not refreshed:
                await self.reconcile()
                refreshed = True
            current = self.store.db.execute(
                "SELECT status FROM first4_orders WHERE reference=?", (entry["reference"],)
            ).fetchone()
            if current[0] not in terminal:
                continue
            entry["status"] = current[0]
            payload["deadline_reconciled"] = True
            with self.store.db:
                self.store.db.execute(
                    "UPDATE first4_orders SET payload=? WHERE reference=?",
                    (json.dumps(payload), entry["reference"]),
                )
            event = self.store.event(entry["session"], entry["symbol"])
            allocation = entry["reference"].rsplit(":", 1)[0] + ":"
            quantity = self.owned_quantities(allocation)
            self.store.outcome(
                event,
                "ENTRY_RECONCILED_HELD" if any(quantity.values()) else "ENTRY_UNFILLED",
                {"entry_status": entry["status"]},
            )
        if not self.entry_blocker.startswith("UNOWNED"):
            self.entry_blocker = self.deadline_blocker()

    async def combo_tick(self, combo: Any, deadline: datetime) -> float:
        ticker = self.ib.reqMktData(combo, "", False, False)
        try:
            while now() < deadline:
                if math.isfinite(ticker.minTick) and ticker.minTick > 0:
                    return float(ticker.minTick)
                await asyncio.sleep(0.02)
            raise ValueError("COMBO_PRICE_INCREMENT_UNAVAILABLE")
        finally:
            self.ib.cancelMktData(combo)

    def exit_exception(self, reference: str, reason: str) -> None:
        self.problem = reason
        if self.operator_exceptions.get(reference) == reason:
            return
        self.operator_exceptions[reference] = reason
        self.report_error(
            "exit_exception:" + reference,
            {"time": now().isoformat(), "error": reason, "requires_operator": True},
        )

    async def close_due(self) -> None:
        for entry in self.store.active_entries():
            try:
                await self.close_one(entry)
            except sqlite3.Error:
                raise
            except Exception as exc:
                self.exit_exception(entry["reference"], str(exc) or type(exc).__name__)

    async def close_one(self, entry: dict[str, Any]) -> None:
        payload = json.loads(entry["payload"])
        if now() < datetime.fromisoformat(payload["exit_at"]):
            return
        self.guard()
        # Cancel only this manager's pending entry; require terminal acknowledgement.
        for trade in self.ib.openTrades():
            if trade.order.orderRef == entry["reference"] and not trade.isDone():
                if (
                    trade.order.account != PAPER_ACCOUNT
                    or trade.order.clientId != self.config.client_id
                ):
                    raise ValueError("Cancellation account/client mismatch")
                if trade.orderStatus.status != "PendingCancel":
                    self.ib.cancelOrder(trade.order)
                return
        if entry["status"] not in {"Filled", "Cancelled", "ApiCancelled", "Inactive"}:
            self.problem = "EXIT_WAITING_FOR_ENTRY_CANCELLATION_OR_RECONCILIATION"
            return
        event = self.store.event(entry["session"], entry["symbol"])
        # Always reconcile actual individual legs; a BAG acknowledgement is not two fills.
        quantities = self.owned_quantities()
        legs = [Contract.create(**payload[key]) for key in ("put", "call")]
        actual = {p.contract.conId: float(p.position) for p in self.ib.positions(PAPER_ACCOUNT)}
        for leg in legs:
            if actual.get(leg.conId, 0) != quantities.get(leg.conId, 0):
                self.reconciled = False
                self.problem = "EXIT_POSITION_MISMATCH_RECONCILIATION_REQUIRED"
                return
        allocation = entry["reference"].rsplit(":", 1)[0] + ":"
        allocated_quantities = self.owned_quantities(allocation)
        q = [allocated_quantities.get(x.conId, 0) for x in legs]
        if min(q) < 0:
            raise ValueError("UNEXPECTED_SHORT_OPTION_POSITION")
        if not any(q):
            entry_fills = self.store.order_fills(entry["reference"])
            if entry["status"] == "Filled" and not entry_fills:
                self.problem = "FILLED_ENTRY_MISSING_LEG_EXECUTIONS"
                self.reconciled = False
                return
            orders = self.store.allocation_orders(entry["session"], entry["symbol"])
            if not payload.get("deadline_reconciled") or any(
                o["status"] not in {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
                for o in orders
            ):
                self.problem = "ZERO_QUANTITY_AWAITING_ENTRY_EXIT_RECONCILIATION"
                return
            for order in orders:
                if order["status"] == "Filled":
                    data = json.loads(order["payload"])
                    ids = (
                        [data[k]["conId"] for k in ("put", "call")]
                        if order["role"] == "ENTRY"
                        else data["exit_con_ids"]
                    )
                    expected = (
                        data.get("quantity", 1)
                        if order["role"] == "ENTRY"
                        else data["exit_quantity"]
                    )
                    fills = self.store.order_fills(order["reference"])
                    if any(
                        sum(f["quantity"] for f in fills if f["con_id"] == i) != expected
                        for i in ids
                    ):
                        self.reconciled = False
                        self.problem = "FILLED_ORDER_MISSING_LEG_EXECUTIONS"
                        return
            self.store.outcome(event, "CLOSED" if entry_fills else "ENTRY_UNFILLED")
            with self.store.db:
                self.store.db.execute(
                    "UPDATE first4_orders SET obligation_done=1 "
                    "WHERE reference=? AND obligation_done=0",
                    (entry["reference"],),
                )
            self.operator_exceptions.pop(entry["reference"], None)
            return
        prior_exits = [
            o
            for o in self.store.allocation_orders(entry["session"], entry["symbol"])
            if o["role"] == "EXIT"
        ]
        combo_exit = next((o for o in prior_exits if o["reference"] == allocation + "EXIT"), None)
        if combo_exit:
            raise ValueError("EXIT_PENDING_OR_INCOMPLETE_REQUIRES_OPERATOR")
        if now() >= datetime.fromisoformat(event["close_at"]):
            self.exit_exception(entry["reference"], "MISSED_SESSION_CLOSE_EXIT_REQUIRES_OPERATOR")
            self.store.outcome(event, "EXIT_OVERDUE", {"remaining_legs": q})
            return
        if q[0] == q[1] and not prior_exits:
            combo = Contract(
                secType="BAG",
                symbol=event["symbol"],
                currency="USD",
                exchange="SMART",
                comboLegs=[
                    ComboLeg(conId=x.conId, ratio=1, action="BUY", exchange="SMART") for x in legs
                ],
            )
            await self.close_order(event, combo, legs, q[0], payload)
        else:
            # A non-atomic partial execution leaves explicit individual close obligations.
            for leg, quantity in zip(legs, q, strict=True):
                if quantity:
                    if any(
                        o["reference"] == allocation + "EXIT" + str(leg.conId) for o in prior_exits
                    ):
                        self.exit_exception(
                            entry["reference"], "LEG_EXIT_PENDING_OR_INCOMPLETE_REQUIRES_OPERATOR"
                        )
                        continue
                    await self.close_order(event, leg, [leg], quantity, payload, str(leg.conId))

    async def close_order(
        self,
        event: dict[str, Any],
        contract: Any,
        legs: list[Any],
        quantity: float,
        payload: dict[str, Any],
        suffix: str = "",
    ) -> None:
        close = datetime.fromisoformat(event["close_at"])
        deadline = min(close, now() + timedelta(seconds=2))
        async with asyncio.timeout(max(0, (deadline - now()).total_seconds())):
            quotes = await self.quotes(legs, deadline, quantity=quantity, side="SELL")
        if now() >= close:
            raise ValueError("MISSED_SESSION_CLOSE_EXIT_REQUIRES_OPERATOR")
        if any(
            not 0 <= (now() - datetime.fromisoformat(q[k])).total_seconds() <= 5
            for q in quotes
            for k in ("bid_at", "ask_at")
        ):
            raise ValueError("EXIT_QUOTES_EXPIRED_DURING_PREPARATION")
        original_asks = {
            payload[key]["conId"]: payload["quotes"][i]["ask"]
            for i, key in enumerate(("put", "call"))
        }
        quoted_exit = sum(
            q["bid"] * float(leg.multiplier) * quantity for leg, q in zip(legs, quotes, strict=True)
        )
        quoted_entry = sum(
            original_asks[leg.conId] * float(leg.multiplier) * quantity for leg in legs
        )
        closing = {
            **payload,
            "exit_quotes": quotes,
            "exit_con_ids": [leg.conId for leg in legs],
            "exit_quantity": quantity,
            "exit_order_type": "MKT",
            "quoted_exit_bid_usd": quoted_exit,
            "quoted_entry_ask_for_exit_legs_usd": quoted_entry,
            "quoted_ask_to_bid_gross_usd": quoted_exit - quoted_entry,
        }
        self.submit(
            event,
            contract,
            MarketOrder(
                "SELL",
                quantity,
                tif="DAY",
                outsideRth=False,
            ),
            closing,
            "EXIT",
            suffix,
        )
        self.store.outcome(event, "EXIT_SUBMITTED", {"exit_quote_comparison": closing})
