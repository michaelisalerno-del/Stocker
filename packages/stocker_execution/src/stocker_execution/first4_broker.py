"""Listed contracts and real IBKR PAPER orders. No valuation or simulated fills."""

import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from typing import Any

from ib_async import IB, ComboLeg, Contract, ExecutionFilter, LimitOrder, MarketOrder, Option

from stocker_execution.first4_config import PAPER_ACCOUNT, First4Config
from stocker_execution.first4_store import Store


def now() -> datetime:
    return datetime.now(UTC)


def size(budget: float, debit: float, multiplier: float, fee: float, increment: float) -> float:
    if (
        not all(math.isfinite(x) for x in (budget, debit, multiplier, fee, increment))
        or min(budget, debit, multiplier, increment) <= 0
        or fee < 0
    ):
        raise ValueError("Invalid sizing inputs")
    b, d, m, f, i = map(lambda x: Decimal(str(x)), (budget, debit, multiplier, fee, increment))
    return float((b / ((d * m + f) * i)).to_integral_value(rounding=ROUND_FLOOR) * i)


def listed_strike(strikes: list[float], target: float, right: str, rule: str) -> float:
    values = sorted(x for x in strikes if math.isfinite(x) and x > 0)
    if rule == "OUTWARD":
        values = (
            [x for x in values if x <= target]
            if right == "P"
            else [x for x in values if x >= target]
        )
    if not values:
        raise ValueError("No listed strike for configured mapping")
    return min(values, key=lambda x: (abs(x - target), x if right == "P" else -x))


class PaperBroker:
    def __init__(self, config: First4Config, store: Store, ib: Any = None):
        self.config, self.store = config, store
        self.ib: Any = ib if ib is not None else IB()
        self.reconciled = False
        self.problem = "NOT_CONNECTED"
        self.entry_blocker = "NOT_RECONCILED"
        self.chains: dict[int, Any] = {}
        self.ib.disconnectedEvent += self.disconnected
        self.ib.execDetailsEvent += self.fill
        self.ib.commissionReportEvent += self.commission
        self.ib.orderStatusEvent += self.order_status
        self.ib.errorEvent += self.error
        self.ib.positionEvent += self.position

    def disconnected(self, *args: Any) -> None:
        self.reconciled = False
        self.problem = "DISCONNECTED_RECONCILIATION_REQUIRED"

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
        if not self.reconciled:
            raise ValueError("Broker reconciliation required: " + self.problem)

    async def connect(self) -> None:
        self.reconciled = False
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
                "INSERT OR IGNORE INTO first4_fills VALUES (?,?,?,?,?,?,?,?,NULL)",
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

    def commission(self, trade: Any, fill: Any, report: Any) -> None:
        if report.currency != "USD" or not math.isfinite(report.commission):
            return
        with self.store.db:
            self.store.db.execute(
                "UPDATE first4_fills SET commission=? WHERE exec_id=?",
                (report.commission, report.execId),
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
        if code in {2104, 2106, 2158, 2108}:
            return
        self.store.set_meta(
            "broker_error",
            {"time": now().isoformat(), "request_id": request_id, "code": code, "message": message},
        )

    def owned_quantities(self, allocation: str | None = None) -> dict[int, float]:
        result: dict[int, float] = {}
        for f in self.store.rows("fills"):
            if allocation is not None and not f["reference"].startswith(allocation):
                continue
            result[f["con_id"]] = result.get(f["con_id"], 0) + f["quantity"] * (
                1 if f["side"] == "BOT" else -1
            )
        return result

    async def reconcile(self) -> None:
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
        for t in opens:
            row = self.store.db.execute(
                "SELECT * FROM first4_orders WHERE reference=?", (t.order.orderRef,)
            ).fetchone()
            if row:
                if t.order.account != PAPER_ACCOUNT or t.order.clientId != self.config.client_id:
                    raise ValueError("Owned order account/client mismatch")
                self.order_status(t)
                matched.add(row["reference"])
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
        actual = {
            p.contract.conId: p for p in positions if p.account == PAPER_ACCOUNT and p.position
        }
        owned = self.owned_quantities()
        unknown = sorted(set(actual) - set(owned))
        self.entry_blocker = f"UNOWNED_BROKER_POSITIONS:{unknown}" if unknown else ""
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
        # Unrelated contracts/orders are never cancelled or managed by FIRST4.
        self.reconciled, self.problem = True, ""
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
        c = self.config
        target = (baseline + timedelta(minutes=2880)).date().strftime("%Y%m%d")
        chains = [x for x in await self.chain(underlying) if x.exchange == "SMART"]
        choices = []
        for x in chains:
            expiries = sorted(
                e
                for e in x.expirations
                if e == target or (c.expiry_rule == "FIRST_ON_OR_AFTER" and e >= target)
            )
            if expiries:
                choices.append((expiries[0], x))
        if not choices:
            raise ValueError("EXPIRY_UNAVAILABLE")
        earliest = min(e for e, _ in choices)
        choices = [(e, x) for e, x in choices if e == earliest]
        if len(choices) != 1:
            raise ValueError("AMBIGUOUS_OPTION_TRADING_CLASS_OR_MULTIPLIER")
        expiry, chain = choices[0]
        legs = []
        for right, ratio in (("P", 0.98), ("C", 1.02)):
            strike = listed_strike(list(chain.strikes), anchor * ratio, right, str(c.strike_rule))
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
            if len(details) != 1 or details[0].underConId != underlying.conId:
                raise ValueError("OPTION_CONTRACT_UNAVAILABLE_OR_AMBIGUOUS")
            legs.append(details[0])
        if legs[0].contract.multiplier != legs[1].contract.multiplier:
            raise ValueError("OPTION_MULTIPLIER_MISMATCH")
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

    async def quotes(self, contracts: list[Any], deadline: datetime) -> list[dict[str, Any]]:
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
                result = []
                for (_, t, _), s in zip(subscriptions, states, strict=True):
                    stamps = [s["bid_at"], s["ask_at"]]
                    if t.marketDataType != 1 or any(x is None for x in stamps):
                        break
                    age = max((now() - x).total_seconds() for x in stamps)
                    if (
                        not 0 <= age <= self.config.number("quote_max_age_seconds")
                        or not all(math.isfinite(x) for x in (t.bid, t.ask, t.bidSize, t.askSize))
                        or not 0 <= t.bid <= t.ask
                        or t.ask <= 0
                        or min(t.bidSize, t.askSize) <= 0
                    ):
                        break
                    result.append(
                        {
                            "bid": t.bid,
                            "ask": t.ask,
                            "bid_at": s["bid_at"].isoformat(),
                            "ask_at": s["ask_at"].isoformat(),
                        }
                    )
                if len(result) == len(contracts):
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
            self.config.require_execution()
            if self.store.get_meta("paused", False) or self.entry_blocker:
                raise ValueError(self.entry_blocker or "ENTRIES_PAUSED")
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
        self.config.require_execution()
        baseline = datetime.fromisoformat(event["entry_at"])
        deadline = baseline + timedelta(seconds=self.config.number("entry_deadline_seconds"))
        p, c, combo = await self.contracts(underlying, anchor, baseline)
        quotes, tick = await asyncio.gather(
            self.quotes([p.contract, c.contract], deadline),
            self.combo_tick(combo, deadline),
        )
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
        # Floor protects the configured debit. Broker determines whether it can fill.
        ask = sum(q["ask"] for q in quotes)
        limit = float(
            (Decimal(str(ask)) / Decimal(str(tick))).to_integral_value(rounding=ROUND_FLOOR)
            * Decimal(str(tick))
        )
        multiplier = float(p.contract.multiplier)
        quantity = size(
            self.config.number("premium_budget_usd"),
            limit,
            multiplier,
            self.config.number("fee_reserve_per_package_usd"),
            increment,
        )
        if quantity < minimum:
            raise ValueError("PREMIUM_BUDGET_BELOW_ONE_PACKAGE")
        if any(
            (now() - datetime.fromisoformat(q[k])).total_seconds()
            > self.config.number("quote_max_age_seconds")
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

    async def close_due(self) -> None:
        entries = [o for o in self.store.rows("orders") if o["role"] == "ENTRY"]
        for entry in entries:
            payload = json.loads(entry["payload"])
            if now() < datetime.fromisoformat(payload["exit_at"]):
                continue
            self.guard()
            # Cancel only this manager's pending entry; require terminal acknowledgement.
            for trade in self.ib.openTrades():
                if trade.order.orderRef == entry["reference"] and not trade.isDone():
                    self.ib.cancelOrder(trade.order)
                    return
            if entry["status"] not in {"Filled", "Cancelled", "ApiCancelled", "Inactive"}:
                self.problem = "EXIT_WAITING_FOR_ENTRY_CANCELLATION_OR_RECONCILIATION"
                return
            event = next(
                e
                for e in self.store.rows("events")
                if e["session"] == entry["session"] and e["symbol"] == entry["symbol"]
            )
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
                self.store.outcome(event, "CLOSED")
                continue
            prior_exits = [
                o
                for o in self.store.rows("orders")
                if o["session"] == entry["session"]
                and o["symbol"] == entry["symbol"]
                and o["role"] == "EXIT"
            ]
            combo_exit = next(
                (o for o in prior_exits if o["reference"] == allocation + "EXIT"), None
            )
            if combo_exit:
                self.problem = (
                    "EXIT_PENDING_OR_INCOMPLETE_REQUIRES_OPERATOR"
                    if not all(o["status"] == "Filled" for o in prior_exits)
                    else "EXIT_FILL_POSITION_RECONCILIATION_REQUIRED"
                )
                continue
            if now() >= datetime.fromisoformat(event["close_at"]):
                self.problem = "MISSED_SESSION_CLOSE_EXIT_REQUIRES_OPERATOR"
                self.store.outcome(event, "EXIT_OVERDUE", {"remaining_legs": q})
                continue
            if q[0] == q[1] and not prior_exits:
                combo = Contract(
                    secType="BAG",
                    symbol=event["symbol"],
                    currency="USD",
                    exchange="SMART",
                    comboLegs=[
                        ComboLeg(conId=x.conId, ratio=1, action="BUY", exchange="SMART")
                        for x in legs
                    ],
                )
                self.submit(
                    event,
                    combo,
                    MarketOrder("SELL", q[0], tif="DAY", outsideRth=False),
                    payload,
                    "EXIT",
                )
            else:
                # A non-atomic partial execution leaves explicit individual close obligations.
                for leg, quantity in zip(legs, q, strict=True):
                    if quantity:
                        if any(
                            o["reference"] == allocation + "EXIT" + str(leg.conId)
                            for o in prior_exits
                        ):
                            self.problem = "LEG_EXIT_PENDING_OR_INCOMPLETE_REQUIRES_OPERATOR"
                            continue
                        self.submit(
                            event,
                            leg,
                            MarketOrder("SELL", quantity, tif="DAY", outsideRth=False),
                            payload,
                            "EXIT",
                            str(leg.conId),
                        )
