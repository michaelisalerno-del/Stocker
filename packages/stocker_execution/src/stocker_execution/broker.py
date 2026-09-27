"""One IB connection and durable owner for actual long FOP PAPER orders."""

import asyncio
import json
import logging
import math
from bisect import bisect_left
from datetime import UTC, datetime, timedelta
from typing import Any

from ib_async import Contract, ExecutionFilter, Forex, FuturesOption, LimitOrder

from stocker_execution.config import PAPER_ACCOUNT, FuturesConfig
from stocker_execution.contracts import Quote, budget, tick_price, utc, verify_option
from stocker_execution.requests import BrokerConnection
from stocker_execution.rules import frozen_strike, model_delta
from stocker_execution.store import TERMINAL, Store

log = logging.getLogger(__name__)


def now() -> datetime:
    return datetime.now(UTC)


class PaperBroker:
    def __init__(self, config: FuturesConfig, store: Store, ib: Any = None):
        self.config, self.store = config, store
        self.ib: Any = ib if ib is not None else BrokerConnection()
        self.reconciled = False
        self.synchronized = False
        self.problem = "NOT_CONNECTED"
        self.fatal_error = ""
        self.upstream = True
        self.generation = 0
        self.reconciled_at: datetime | None = None
        self.lock = asyncio.Lock()
        self.quote_lock = asyncio.Lock()
        self.chains: dict[tuple[int, str], Any] = {}
        self.fx: tuple[float, datetime] | None = None
        self.fx_contract: Any = None
        self.fx_ticker: Any = None
        self.fx_bid_at: datetime | None = None
        self.fx_ask_at: datetime | None = None
        self.option_quotes: dict[int, Quote] = {}
        self.management_problems: dict[str, str] = {}
        for event, callback in (
            ("execDetailsEvent", self.fill),
            ("commissionReportEvent", self.commission),
            ("orderStatusEvent", self.order_status),
            ("positionEvent", self.position),
        ):
            getattr(self.ib, event).__iadd__(self.persist(callback))
        self.ib.disconnectedEvent += self.disconnected
        self.ib.errorEvent += self.error

    def persist(self, callback: Any) -> Any:
        def receive(*args: Any) -> None:
            try:
                callback(*args)
            except Exception as exc:
                self.fatal_error = f"BROKER_EVIDENCE_FAILED:{type(exc).__name__}:{exc}"
                self.reconciled = False
                log.exception("Broker evidence could not be recorded")

        return receive

    def disconnected(self, *args: Any) -> None:
        self.reconciled = False
        self.synchronized = False
        self.problem = "DISCONNECTED_RECONCILIATION_REQUIRED"
        self.generation += 1
        self.chains.clear()
        self.fx = None
        self.fx_bid_at = self.fx_ask_at = None
        self.option_quotes.clear()

    def error(self, request_id: int, code: int, message: str, contract: Any = None) -> None:
        if code in {1100, 1300, 10197}:
            self.upstream = False
            self.disconnected()
            self.problem = f"BROKER_{code}:{message}"
        elif code in {1101, 1102}:
            # All data must be resubscribed after restoration, even when IB reports retained data.
            self.ib.disconnect()

    def guard(self) -> None:
        if (
            self.config.environment != "PAPER"
            or self.config.expected_account != PAPER_ACCOUNT
            or not self.ib.isConnected()
            or self.ib.managedAccounts() != [PAPER_ACCOUNT]
            or self.ib.client.clientId != self.config.client_id
        ):
            raise ValueError("VERIFIED_ALLOWLISTED_PAPER_ACCOUNT_REQUIRED")
        if not self.upstream or self.fatal_error:
            raise ValueError(self.fatal_error or "BROKER_UPSTREAM_UNAVAILABLE")

    def entry_reason(self) -> str:
        try:
            self.guard()
        except ValueError as exc:
            return str(exc)
        if not self.reconciled:
            return self.problem or "RECONCILIATION_REQUIRED"
        if not self.reconciled_at or (now() - self.reconciled_at).total_seconds() > 30:
            return "RECONCILIATION_STALE"
        if self.store.get_meta("paused", False):
            return "ENTRIES_PAUSED"
        if not self.config.armed:
            return "EXECUTION_UNARMED"
        if (
            not self.config.available_market_data_lines
            or not self.config.market_data_allocation_source
        ):
            return "MARKET_DATA_ALLOCATION_UNVERIFIED"
        return ""

    async def connect(self) -> None:
        async with self.lock:
            try:
                await self._connect()
            except BaseException:
                self.ib.disconnect()
                self.synchronized = False
                raise

    async def _connect(self) -> None:
        self.reconciled = False
        self.synchronized = False
        self.upstream = True
        await self.ib.connectAsync(
            self.config.host,
            self.config.port,
            clientId=self.config.client_id,
            account=PAPER_ACCOUNT,
            timeout=10,
            readonly=False,
            raiseSyncErrors=True,
        )
        try:
            self.guard()
        except ValueError:
            self.ib.disconnect()
            raise
        self.ib.reqMarketDataType(1)
        await self._reconcile()
        self.fx_contract = (await self.ib.qualifyContractsAsync(Forex("GBPUSD")))[0]
        self.fx_ticker = self.ib.reqMktData(self.fx_contract, "", False, False)
        self.fx_ticker.marketDataType = 0
        self.fx_ticker.updateEvent += self.update_fx
        self.synchronized = True

    def update_fx(self, ticker: Any) -> None:
        for tick in ticker.ticks:
            if tick.tickType == 1:
                self.fx_bid_at = utc(tick.time)
            if tick.tickType == 2:
                self.fx_ask_at = utc(tick.time)
        if self.fx_bid_at and self.fx_ask_at:
            quote = Quote(
                ticker.bid, ticker.ask, self.fx_bid_at, self.fx_ask_at, ticker.marketDataType
            )
            try:
                quote.validate(now())
            except ValueError:
                self.fx = None
                return
            # GBPUSD bid: conservative GBP cash cost of purchasing USD.
            self.fx = (1 / ticker.bid, min(self.fx_bid_at, self.fx_ask_at))

    def fill(self, trade: Any, fill: Any) -> None:
        e, c = fill.execution, fill.contract
        row = self.store.db.execute(
            "SELECT * FROM orders WHERE reference=?", (e.orderRef,)
        ).fetchone()
        if row is None:
            return
        if (
            e.acctNumber != PAPER_ACCOUNT
            or c.secType != "FOP"
            or e.clientId != self.config.client_id
            or e.orderId != row["order_id"]
            or not e.permId
            or (row["perm_id"] and row["perm_id"] != e.permId)
        ):
            raise ValueError("EXECUTION_OWNERSHIP_MISMATCH")
        payload = json.loads(row["payload"])
        if c.conId != payload["con_id"] or e.side != ("BOT" if row["role"] == "ENTRY" else "SLD"):
            raise ValueError("EXECUTION_CONTRACT_OR_SIDE_MISMATCH")
        if not math.isfinite(e.price) or e.price < 0 or not 0 < float(e.shares) <= 1:
            raise ValueError("INVALID_EXECUTION")
        at = utc(e.time)
        fx = self.fx if self.fx and 0 <= (at - self.fx[1]).total_seconds() <= 30 else None
        if fx is None and payload.get("fx") and payload.get("fx_at"):
            saved_at = datetime.fromisoformat(payload["fx_at"])
            if 0 <= (at - saved_at).total_seconds() <= 30:
                fx = (payload["fx"], saved_at)
        with self.store.db:
            self.store.db.execute(
                "UPDATE orders SET perm_id=? WHERE reference=? AND perm_id IS NULL",
                (e.permId, e.orderRef),
            )
        self.store.record_fill(
            {
                "exec_id": e.execId,
                "reference": e.orderRef,
                "con_id": c.conId,
                "quantity": float(e.shares),
                "price": e.price,
                "side": e.side,
                "at": at.isoformat(),
                "fx": fx[0] if fx else None,
                "fx_at": fx[1].isoformat() if fx else None,
            }
        )
        self.reconciled = False
        self.generation += 1
        if fill.commissionReport and fill.commissionReport.execId:
            self.commission(trade, fill, fill.commissionReport)

    def commission(self, trade: Any, fill: Any, report: Any) -> None:
        if not math.isfinite(report.commission) or abs(report.commission) >= 1e100:
            return
        previous = self.store.db.execute(
            "SELECT commission,commission_currency FROM fills WHERE exec_id=?", (report.execId,)
        ).fetchone()
        if previous is None or tuple(previous) == (report.commission, report.currency):
            return
        with self.store.db:
            self.store.db.execute(
                "UPDATE fills SET commission=?,commission_currency=? WHERE exec_id=?",
                (report.commission, report.currency, report.execId),
            )
            self.store.audit(
                fill.execution.orderRef,
                "COMMISSION",
                {
                    "exec_id": report.execId,
                    "amount": report.commission,
                    "currency": report.currency,
                },
            )

    def order_status(self, trade: Any) -> None:
        o, s = trade.order, trade.orderStatus
        row = self.store.db.execute(
            "SELECT * FROM orders WHERE reference=?", (o.orderRef,)
        ).fetchone()
        if row is None:
            return
        perm = s.permId or o.permId
        if (
            o.account != PAPER_ACCOUNT
            or o.clientId != self.config.client_id
            or o.orderId != row["order_id"]
            or (row["perm_id"] and row["perm_id"] != perm)
            or trade.contract.conId != json.loads(row["payload"])["con_id"]
        ):
            raise ValueError("ORDER_OWNERSHIP_MISMATCH")
        if s.status in {"", "PendingSubmit"} and row["status"] in TERMINAL:
            return
        if float(s.filled) < row["filled"]:
            raise ValueError("ORDER_FILL_TOTAL_REGRESSED")
        if (row["status"], row["filled"], row["remaining"], row["perm_id"]) == (
            s.status,
            float(s.filled),
            float(s.remaining),
            perm or None,
        ):
            return
        self.generation += 1
        with self.store.db:
            self.store.db.execute(
                "UPDATE orders SET status=?,filled=?,remaining=?,perm_id=? WHERE reference=?",
                (s.status, float(s.filled), float(s.remaining), perm or None, o.orderRef),
            )
            self.store.audit(
                o.orderRef,
                "ORDER_STATUS",
                {
                    "status": s.status,
                    "filled": float(s.filled),
                    "remaining": float(s.remaining),
                    "perm_id": perm,
                },
            )
        self.reconciled = False

    def position(self, value: Any) -> None:
        if value.account != PAPER_ACCOUNT:
            return
        previous = self.store.db.execute(
            "SELECT quantity FROM positions WHERE con_id=?", (value.contract.conId,)
        ).fetchone()
        if previous is None or previous[0] != float(value.position):
            self.generation += 1
        with self.store.db:
            self.store.db.execute(
                "INSERT OR REPLACE INTO positions VALUES(?,?,?)",
                (value.contract.conId, float(value.position), json.dumps(value.contract.dict())),
            )
        self.reconciled = False

    async def reconcile(self) -> None:
        async with self.lock:
            await self._reconcile()

    async def _reconcile(self) -> None:
        self.reconciled = False
        self.guard()
        async with asyncio.timeout(20):
            opened = await self.ib.reqAllOpenOrdersAsync()
            completed = await self.ib.reqCompletedOrdersAsync(apiOnly=False)
            fills = await self.ib.reqExecutionsAsync(ExecutionFilter(acctCode=PAPER_ACCOUNT))
            for fill in fills:
                self.fill(None, fill)
            for trade in opened:
                self.order_status(trade)
            for trade in completed:
                self.completed_order(trade)
            generation = self.generation
            positions = await self.ib.reqPositionsAsync()
        if generation != self.generation:
            self.problem = "BROKER_CHANGED_DURING_RECONCILIATION"
            return
        self.guard()
        by_contract = {
            p.contract.conId: float(p.position)
            for p in positions
            if p.account == PAPER_ACCOUNT and p.position
        }
        with self.store.db:
            self.store.db.execute("DELETE FROM positions")
            for p in positions:
                if p.account == PAPER_ACCOUNT:
                    self.store.db.execute(
                        "INSERT INTO positions VALUES(?,?,?)",
                        (p.contract.conId, float(p.position), json.dumps(p.contract.dict())),
                    )
        known: dict[int, float] = {}
        owned_refs = {r[0] for r in self.store.db.execute("SELECT reference FROM orders")}
        reasons = []
        for trade in opened:
            if trade.order.account == PAPER_ACCOUNT and trade.order.orderRef not in owned_refs:
                reasons.append("UNRELATED_OPEN_ORDERS_REQUIRE_REVIEW")
        for row in self.store.active():
            plan = json.loads(row["plan"])
            qty = self.store.exposure(row["id"])
            cid = plan["option"]["conId"]
            known[cid] = known.get(cid, 0) + qty
            orders = self.store.orders(row["id"])
            if not orders or any(o["status"] in {"SUBMITTING", "UNKNOWN", ""} for o in orders):
                reasons.append("SUBMISSION_UNCERTAIN_RECONCILE_BEFORE_RETRY")
            for order in orders:
                actual = sum(
                    f["quantity"]
                    for f in self.store.fills(row["id"])
                    if f["reference"] == order["reference"]
                )
                if actual != order["filled"]:
                    reasons.append("EXECUTION_REPORTS_INCOMPLETE")
            if qty < 0 or qty > 1:
                reasons.append("UNEXPECTED_STRATEGY_EXPOSURE")
            if qty > 0:
                with self.store.db:
                    self.store.db.execute(
                        "UPDATE reservations SET state='POSITION_OPEN' WHERE id=?", (row["id"],)
                    )
        if {k: v for k, v in known.items() if v} != by_contract:
            reasons.append("UNEXPLAINED_POSITION_OR_EXERCISE_EXPOSURE")
        else:
            # Aggregate broker exposure agrees with every owned execution. A closed trade
            # can release its slot while another owned trade retains the same contract.
            for row in self.store.active():
                self.store.confirm_closed(row["id"], 0)
        capacity = self.store.capacity()
        if capacity["reserved_open_trades"] > 4 or capacity["allocation_pennies"] > 4000:
            reasons.append("RISK_INVARIANT_VIOLATION")
        self.problem = ";".join(sorted(set(reasons)))
        self.reconciled = not reasons
        self.reconciled_at = now()
        self.store.set_meta(
            "reconciliation",
            {
                "at": self.reconciled_at.isoformat(),
                "account": PAPER_ACCOUNT,
                "problem": self.problem,
                "positions": by_contract,
                "open_orders": len(opened),
            },
        )

    def completed_order(self, trade: Any) -> None:
        """IB completedOrder has no clientId/orderId and a default-zero OrderStatus total."""
        order = trade.order
        row = self.store.db.execute(
            "SELECT * FROM orders WHERE reference=?", (order.orderRef,)
        ).fetchone()
        if row is None:
            return
        payload = json.loads(row["payload"])
        if (
            order.account != PAPER_ACCOUNT
            or not order.permId
            or (row["perm_id"] and row["perm_id"] != order.permId)
            or trade.contract.conId != payload["con_id"]
            or order.action != ("BUY" if row["role"] == "ENTRY" else "SELL")
            or float(order.totalQuantity) != payload["quantity"]
            or order.orderType != "LMT"
            or order.lmtPrice != payload["limit"]
        ):
            raise ValueError("COMPLETED_ORDER_OWNERSHIP_MISMATCH")
        filled = float(order.filledQuantity)
        if not math.isfinite(filled) or not 0 <= filled <= payload["quantity"]:
            raise ValueError("COMPLETED_ORDER_FILL_TOTAL_UNAVAILABLE")
        status = trade.orderStatus.status
        if status not in TERMINAL:
            raise ValueError("COMPLETED_ORDER_TERMINAL_STATE_UNAVAILABLE")
        if filled < row["filled"]:
            raise ValueError("COMPLETED_ORDER_FILL_TOTAL_REGRESSED")
        with self.store.db:
            self.store.db.execute(
                "UPDATE orders SET status=?,filled=?,remaining=0,perm_id=? WHERE reference=?",
                (status, filled, order.permId, order.orderRef),
            )
            if row["status"] != status or row["filled"] != filled:
                self.store.audit(
                    order.orderRef,
                    "COMPLETED_ORDER",
                    {"status": status, "filled": filled, "perm_id": order.permId},
                )

    async def quote(self, contract: Any) -> Quote:
        async with self.quote_lock:
            return await self._quote(contract)

    async def _quote(self, contract: Any) -> Quote:
        generation = self.generation
        bid_at = ask_at = None
        ready = asyncio.Event()

        def update(ticker: Any) -> None:
            nonlocal bid_at, ask_at
            for tick in ticker.ticks:
                if tick.tickType == 1:
                    bid_at = utc(tick.time)
                elif tick.tickType == 2:
                    ask_at = utc(tick.time)
            if bid_at and ask_at:
                ready.set()

        with self.ib.market_data(contract, update) as (ticker, request):
            waiter = asyncio.create_task(ready.wait())
            try:
                done, _ = await asyncio.wait(
                    {waiter, request}, timeout=5, return_when=asyncio.FIRST_COMPLETED
                )
                if request in done:
                    request.result()
                if not ready.is_set() or bid_at is None or ask_at is None:
                    raise ValueError("OPTION_QUOTE_NOT_RECEIVED")
                if generation != self.generation:
                    raise ValueError("BROKER_STATE_CHANGED_DURING_QUOTE")
                result = Quote(ticker.bid, ticker.ask, bid_at, ask_at, ticker.marketDataType)
                result.validate(now())
                self.option_quotes[contract.conId] = result
                return result
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)

    async def increments(self, details: Any) -> list[tuple[float, float]]:
        exchanges = details.validExchanges.split(",")
        identifiers = details.marketRuleIds.split(",")
        if details.contract.exchange not in exchanges or len(exchanges) != len(identifiers):
            raise ValueError("PRICE_INCREMENT_UNVERIFIED")
        identifier = identifiers[exchanges.index(details.contract.exchange)]
        if not identifier.isdigit() or int(identifier) <= 0:
            raise ValueError("PRICE_INCREMENT_UNVERIFIED")
        rules = await self.ib.reqMarketRuleAsync(int(identifier))
        return [(float(r.lowEdge), float(r.increment)) for r in rules or []]

    async def prepare(
        self, event: dict[str, Any], future: Any, inputs: dict[str, float]
    ) -> dict[str, Any]:
        market = event["market"]
        mapping = self.config.mappings.get(market)
        if not mapping:
            raise ValueError("LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED")
        at, exit_at = (
            datetime.fromisoformat(event["signal_at"]),
            datetime.fromisoformat(event["exit_at"]),
        )
        key = (future.conId, at.date().isoformat())
        if key not in self.chains:
            self.chains.clear() if len(self.chains) >= 6 else None
            self.chains[key] = await self.ib.reqSecDefOptParamsAsync(
                future.symbol, mapping.exchange, "FUT", future.conId
            )
        chains = [
            c
            for c in self.chains[key]
            if c.exchange == mapping.exchange
            and c.tradingClass == mapping.trading_class
            and c.underlyingConId == future.conId
            and float(c.multiplier) == mapping.multiplier
        ]
        if len(chains) != 1:
            raise ValueError("NO_UNAMBIGUOUS_APPROVED_FOP_CHAIN")
        from zoneinfo import ZoneInfo

        day = at.astimezone(ZoneInfo(mapping.expiry_timezone)).strftime("%Y%m%d")
        if day not in chains[0].expirations:
            raise ValueError("NO_REAL_0DTE_MATCH")
        expiry = (
            datetime.strptime(day + " " + mapping.termination_time, "%Y%m%d %H:%M:%S")
            .replace(tzinfo=ZoneInfo(mapping.expiry_timezone))
            .astimezone(UTC)
        )
        if expiry <= at:
            raise ValueError("NO_REAL_0DTE_MATCH")
        target = frozen_strike(
            inputs["futures_price"],
            inputs["rv15"],
            at,
            expiry,
            event["right"],
            event["target_delta"],
        )
        strikes = sorted(s for s in chains[0].strikes if math.isfinite(s) and s > 0)
        index = bisect_left(strikes, target)
        candidates = strikes[max(0, index - 1) : index + 1]
        if not candidates:
            raise ValueError("NO_LISTED_STRIKE")
        strike = min(
            candidates,
            key=lambda s: (
                abs(
                    model_delta(
                        inputs["futures_price"], s, inputs["rv15"], at, expiry, event["right"]
                    )
                    - event["target_delta"]
                ),
                s,
            ),
        )
        delta = model_delta(
            inputs["futures_price"], strike, inputs["rv15"], at, expiry, event["right"]
        )
        if abs(delta - event["target_delta"]) > mapping.delta_tolerance:
            raise ValueError("LISTED_STRIKE_OUTSIDE_APPROVED_DELTA_TOLERANCE")
        contract = FuturesOption(
            symbol=mapping.symbol,
            lastTradeDateOrContractMonth=day,
            strike=strike,
            right=event["right"],
            exchange=mapping.exchange,
            multiplier=str(mapping.multiplier),
            currency=mapping.currency,
            tradingClass=mapping.trading_class,
        )
        details = await self.ib.reqContractDetailsAsync(contract)
        if len(details) != 1:
            raise ValueError("OPTION_IDENTITY_AMBIGUOUS")
        detail = details[0]
        expiry, _ = verify_option(
            detail, future.conId, market, mapping, now(), exit_at, event["right"]
        )
        quote = await self.quote(detail.contract)
        increments = await self.increments(detail)
        price = tick_price(quote.ask, increments, True)
        if self.fx is None:
            raise ValueError("FX_STALE_OR_UNAVAILABLE")
        cash = budget(
            price,
            mapping.multiplier,
            mapping.price_unit_factor,
            *self.fx,
            now(),
            mapping.fee_reserve_gbp,
        )
        return {
            **cash,
            "option": detail.contract.dict(),
            "signal_future": future.dict(),
            "expiry_at": expiry.isoformat(),
            "exit_at": event["exit_at"],
            "signal_at": event["signal_at"],
            "currency": mapping.currency,
            "multiplier": mapping.multiplier,
            "price_unit_factor": mapping.price_unit_factor,
            "limit": price,
            "increments": increments,
            "selected_delta": delta,
            "delta_basis": "FROZEN_PRIOR_RV_MODEL_WITH_APPROVED_REAL_EXPIRY_ADAPTATION",
            "bid": quote.bid,
            "ask": quote.ask,
            "bid_at": quote.bid_at.isoformat(),
            "ask_at": quote.ask_at.isoformat(),
            "mapping": mapping.model_dump(),
            "inputs": inputs,
        }

    async def enter(self, event: dict[str, Any], plan: dict[str, Any]) -> str:
        async with self.lock:
            reason = self.entry_reason()
            if reason:
                return reason
            at = datetime.fromisoformat(event["signal_at"])
            if not 0 <= (now() - at).total_seconds() < self.config.entry_deadline_seconds:
                return "STALE_SIGNAL_NO_REPLAY"
            Quote(
                plan["bid"],
                plan["ask"],
                datetime.fromisoformat(plan["bid_at"]),
                datetime.fromisoformat(plan["ask_at"]),
                1,
            ).validate(now())
            mapping = self.config.mappings.get(event["market"])
            if not mapping or mapping.model_dump() != plan["mapping"]:
                return "MAPPING_AUTHORITY_CHANGED"
            if self.fx is None:
                return "FX_STALE_OR_UNAVAILABLE"
            plan = {
                **plan,
                **budget(
                    plan["limit"],
                    mapping.multiplier,
                    mapping.price_unit_factor,
                    *self.fx,
                    now(),
                    mapping.fee_reserve_gbp,
                ),
            }
            if reason := self.store.reserve(event["id"], plan):
                return reason
            deadline = at + timedelta(seconds=self.config.entry_deadline_seconds)
            await self.submit(event["id"], "ENTRY", plan, deadline, plan["limit"], 1)
            self.store.decision(event["id"], "ORDER_SUBMITTED")
            return ""

    async def submit(
        self,
        identity: str,
        role: str,
        plan: dict[str, Any],
        deadline: datetime,
        price: float,
        quantity: float,
    ) -> None:
        self.guard()
        if not 0 < quantity <= 1 or (role == "ENTRY" and quantity != 1):
            raise ValueError("INVALID_ORDER_QUANTITY")
        if not math.isfinite(price) or price <= 0:
            raise ValueError("INVALID_ORDER_LIMIT")
        oid = self.ib.client.getReqId()
        payload = {
            "con_id": plan["option"]["conId"],
            "quantity": quantity,
            "limit": price,
            "submitted_at": now().isoformat(),
            "fx": self.fx[0] if self.fx else None,
            "fx_at": self.fx[1].isoformat() if self.fx else None,
        }
        ref = self.store.prepare_order(identity, role, oid, deadline.isoformat(), payload)
        order = LimitOrder(
            "BUY" if role == "ENTRY" else "SELL",
            quantity,
            price,
            orderId=oid,
            account=PAPER_ACCOUNT,
            orderRef=ref,
            tif="GTD",
            goodTillDate=deadline.strftime("%Y%m%d %H:%M:%S UTC"),
            outsideRth=True,
            transmit=True,
        )
        try:
            # No await between final account guard and submission; timeout never triggers resend.
            self.guard()
            self.ib.placeOrder(Contract.create(**plan["option"]), order)
        except Exception:
            with self.store.db:
                self.store.db.execute(
                    "UPDATE orders SET status='UNKNOWN' WHERE reference=?", (ref,)
                )
            raise
        finally:
            self.reconciled = False

    async def manage(self) -> None:
        """Always runs, including unarmed/paused/full/missing-entry-data states."""
        self.guard()
        if (
            not self.reconciled
            or not self.reconciled_at
            or (now() - self.reconciled_at).total_seconds() >= 15
        ):
            await self.reconcile()
        for row in self.store.active():
            try:
                await self.manage_one(row)
            except Exception as exc:
                self.management_problems[row["id"]] = str(exc)
                log.warning("Position management %s: %s", row["market"], exc)

    async def manage_one(self, row: dict[str, Any]) -> None:
        plan = json.loads(row["plan"])
        orders = self.store.orders(row["id"])
        pending = [o for o in orders if o["status"] not in TERMINAL]
        for order in pending:
            if now() < datetime.fromisoformat(order["deadline"]):
                continue
            trade = next(
                (
                    t
                    for t in self.ib.openTrades()
                    if t.order.orderRef == order["reference"]
                    and t.order.orderId == order["order_id"]
                    and t.order.account == PAPER_ACCOUNT
                ),
                None,
            )
            if trade and order["status"] != "PendingCancel":
                self.guard()
                self.ib.cancelOrder(trade.order)
        if pending:
            return
        quantity = self.store.exposure(row["id"])
        if quantity <= 0 or now() < datetime.fromisoformat(row["exit_at"]):
            if quantity > 0:
                previous = self.option_quotes.get(plan["option"]["conId"])
                if previous is None or (now() - previous.bid_at).total_seconds() >= 30:
                    await self.quote(Contract.create(**plan["option"]))
            return
        # Fresh account snapshot and exact aggregate ownership are required to avoid overselling.
        if not self.reconciled_at or (now() - self.reconciled_at).total_seconds() > 20:
            raise ValueError("EXIT_POSITION_SNAPSHOT_STALE")
        cid = plan["option"]["conId"]
        observed = self.store.db.execute(
            "SELECT quantity FROM positions WHERE con_id=?", (cid,)
        ).fetchone()
        aggregate = sum(
            self.store.exposure(r["id"])
            for r in self.store.active()
            if json.loads(r["plan"])["option"]["conId"] == cid
        )
        if not observed or observed[0] != aggregate or aggregate < quantity:
            raise ValueError("EXIT_EXPOSURE_MISMATCH")
        if any(
            o["filled"]
            != sum(
                f["quantity"]
                for f in self.store.fills(row["id"])
                if f["reference"] == o["reference"]
            )
            for o in orders
        ):
            raise ValueError("EXIT_EXECUTION_REPORTS_INCOMPLETE")
        if len([o for o in orders if o["role"] == "EXIT"]) >= self.config.exit_attempts:
            raise ValueError("EXIT_RETRIES_EXHAUSTED_EXPOSURE_REQUIRES_OPERATOR")
        if now() >= datetime.fromisoformat(plan["expiry_at"]):
            raise ValueError("EXPIRY_EXERCISE_EXPOSURE_REQUIRES_OPERATOR")
        generation = self.generation
        quote = await self.quote(Contract.create(**plan["option"]))
        price = tick_price(quote.bid, plan["increments"], False)
        async with self.lock:
            # Position/order callbacks may have changed exposure while the quote was awaited.
            if generation != self.generation:
                raise ValueError("EXIT_EXPOSURE_CHANGED_DURING_QUOTE")
            latest = self.store.db.execute(
                "SELECT quantity FROM positions WHERE con_id=?", (cid,)
            ).fetchone()
            if (
                latest is None
                or latest[0] != aggregate
                or self.store.exposure(row["id"]) != quantity
                or any(o["status"] not in TERMINAL for o in self.store.orders(row["id"]))
            ):
                raise ValueError("EXIT_EXPOSURE_CHANGED_BEFORE_SUBMISSION")
            await self.submit(
                row["id"],
                "EXIT",
                plan,
                min(now() + timedelta(seconds=20), datetime.fromisoformat(plan["expiry_at"])),
                price,
                quantity,
            )
        self.management_problems.pop(row["id"], None)
