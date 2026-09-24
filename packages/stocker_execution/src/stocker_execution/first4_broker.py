"""Listed contracts and real IBKR PAPER orders. No valuation or simulated fills."""

import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from typing import Any
from zoneinfo import ZoneInfo

from ib_async import IB, ComboLeg, Contract, ExecutionFilter, LimitOrder, MarketOrder, Option
from ib_async.wrapper import RequestError

from stocker_execution.first4_config import PAPER_ACCOUNT, First4Config
from stocker_execution.first4_store import Store


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
        self.ib: Any = ib if ib is not None else IB()
        self.reconciled = False
        self.problem = "NOT_CONNECTED"
        self.entry_blocker = "NOT_RECONCILED"
        self.chains: dict[int, Any] = {}
        self.opening_verified_session: str | None = None
        self.reconciliation_lock = asyncio.Lock()
        self.connection_generation = 0
        self.upstream_lost = False
        self.active_session: str | None = None
        self.order_observation_generation = 0
        self.ib.disconnectedEvent += self.disconnected
        self.ib.execDetailsEvent += self.fill
        self.ib.commissionReportEvent += self.commission
        self.ib.orderStatusEvent += self.order_status
        self.ib.errorEvent += self.error
        self.ib.positionEvent += self.position

    def disconnected(self, *args: Any) -> None:
        self.invalidate_connection("DISCONNECTED_RECONCILIATION_REQUIRED")

    def invalidate_connection(self, reason: str) -> None:
        self.connection_generation += 1
        self.opening_verified_session = None
        self.reconciled = False
        self.problem = reason
        self.chains.clear()
        day = (
            self.active_session or now().astimezone(ZoneInfo("America/New_York")).date().isoformat()
        )
        state = self.store.db.execute(
            "SELECT last_clock FROM first4_sessions WHERE session=?", (day,)
        ).fetchone()
        if self.active_session or (state and state["last_clock"]):
            self.store.block(day, "SCANNER_CONTINUITY_LOST_AFTER_DISCONNECT")

    def entries_armed(self) -> bool:
        day = now().astimezone(ZoneInfo("America/New_York")).date()
        state = self.store.db.execute(
            "SELECT blocked FROM first4_sessions WHERE session=?", (day.isoformat(),)
        ).fetchone()
        if self.upstream_lost or not self.reconciled or (state and state["blocked"]):
            return False
        if self.config.armed:
            return True
        if (
            self.config.arm_after_quote_check_on != day
            or self.opening_verified_session != day.isoformat()
            or not self.reconciled
        ):
            return False
        state = self.store.db.execute(
            "SELECT blocked FROM first4_sessions WHERE session=?", (day.isoformat(),)
        ).fetchone()
        return not (state and state["blocked"])

    def require_entries(self) -> None:
        self.config.require_settings()
        if not self.entries_armed():
            raise ValueError("PAPER entries are unarmed")

    def position(self, position: Any) -> None:
        if position.account != PAPER_ACCOUNT:
            return
        con_id = position.contract.conId
        owned = self.owned_quantities()
        if con_id in owned:
            with self.store.db:
                if not position.position and not owned[con_id]:
                    self.store.db.execute("DELETE FROM first4_positions WHERE con_id=?", (con_id,))
                    return
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
        if self.upstream_lost:
            raise ValueError("UPSTREAM_CONNECTION_LOST")
        if not self.reconciled:
            raise ValueError("Broker reconciliation required: " + self.problem)

    async def connect(self) -> None:
        self.reconciled = False
        generation = self.connection_generation
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
        if generation != self.connection_generation:
            raise ValueError("CONNECTION_CHANGED_DURING_CONNECT")
        self.upstream_lost = False
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
            inserted = self.store.db.execute(
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
            if inserted.rowcount:
                self.store.reopen(reference)

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
            row = self.store.order(trade.order.orderRef)
            if not row or row["order_id"] != trade.order.orderId:
                return
            status = trade.orderStatus.status
            terminal = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
            if row["status"] in terminal and status not in terminal:
                self.order_observation_generation += 1
                self.reconciled = False
                self.problem = "OUT_OF_ORDER_STATUS_RECONCILIATION_REQUIRED"
                self.store.reopen(row["reference"])
                return
            perm_id = trade.orderStatus.permId or trade.order.permId
            changed = self.store.db.execute(
                "UPDATE first4_orders SET status=?,perm_id=? WHERE reference=? AND order_id=? "
                "AND (status IS NOT ? OR perm_id IS NOT ?)",
                (
                    status,
                    perm_id,
                    trade.order.orderRef,
                    trade.order.orderId,
                    status,
                    perm_id,
                ),
            )
            if changed.rowcount:
                self.store.reopen(row["reference"])

    def error(self, request_id: int, code: int, message: str, contract: Any = None) -> None:
        if code in {1100, 2110}:
            self.upstream_lost = True
            self.invalidate_connection("UPSTREAM_CONNECTION_LOST_RECONCILIATION_REQUIRED")
        elif code in {1101, 1102}:
            # 1101: subscriptions lost; 1102: maintained. All our market-data
            # subscriptions are scoped to a request and are recreated after recovery.
            self.invalidate_connection("UPSTREAM_RESTORED_RECONCILIATION_REQUIRED")
            self.upstream_lost = False
        if code in {2104, 2106, 2158, 2108}:
            return
        self.store.set_meta(
            "broker_error",
            {"time": now().isoformat(), "request_id": request_id, "code": code, "message": message},
        )

    def owned_quantities(
        self, allocation: str | None = None, *, all_history: bool = False
    ) -> dict[int, float]:
        query = "SELECT f.con_id, sum(f.quantity * CASE f.side WHEN 'BOT' THEN 1 ELSE -1 END) "
        if all_history:
            query += "FROM first4_fills f WHERE 1=1 "
            params: tuple[Any, ...] = ()
        elif allocation is not None:
            query += (
                "FROM first4_orders e JOIN first4_orders o USING(session,symbol) "
                "JOIN first4_fills f ON f.reference=o.reference WHERE e.reference=? "
            )
            params = (allocation + "ENTRY",)
        else:
            query += (
                "FROM first4_orders e INDEXED BY first4_unresolved "
                "CROSS JOIN first4_orders o USING(session,symbol) "
                "CROSS JOIN first4_fills f ON f.reference=o.reference WHERE e.role='ENTRY' "
                "AND coalesce(json_extract(e.payload,'$.management_resolved'),0)=0 "
            )
            params = ()
        return {r[0]: r[1] for r in self.store.db.execute(query + "GROUP BY f.con_id", params)}

    def deadline_blocker(self) -> str:
        for order in self.store.unresolved():
            if order["role"] != "ENTRY":
                continue
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
            await self._reconcile(require_flat)

    async def _reconcile(self, require_flat: bool) -> None:
        self.reconciled = False
        generation = self.connection_generation
        order_generation = self.order_observation_generation
        if (
            self.upstream_lost
            or not self.ib.isConnected()
            or self.ib.managedAccounts() != [PAPER_ACCOUNT]
        ):
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
                    changed = self.store.db.execute(
                        "UPDATE first4_orders SET status=?,perm_id=? WHERE reference=? "
                        "AND (status IS NOT ? OR perm_id IS NOT ?)",
                        (
                            t.orderStatus.status,
                            permanent_id,
                            row["reference"],
                            t.orderStatus.status,
                            permanent_id,
                        ),
                    )
                    if changed.rowcount:
                        self.store.reopen(row["reference"])
                matched.add(row["reference"])
        terminal = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
        orders = self.store.rows("orders")
        uncertain = [
            o["reference"]
            for o in orders
            if o["status"] not in terminal and o["reference"] not in matched
        ]
        for order in orders:
            if order["status"] != "Filled":
                continue
            payload = json.loads(order["payload"])
            ids = (
                [payload[key]["conId"] for key in ("put", "call")]
                if order["role"] == "ENTRY"
                else payload["exit_con_ids"]
            )
            expected = payload["quantity"] if order["role"] == "ENTRY" else payload["exit_quantity"]
            executions = self.store.executions(order["reference"])
            if any(
                sum(f["quantity"] for f in executions if f["con_id"] == con_id) != expected
                for con_id in ids
            ):
                uncertain.append("FILLED_ORDER_MISSING_LEG_EXECUTIONS:" + order["reference"])
        actual = {
            p.contract.conId: p for p in positions if p.account == PAPER_ACCOUNT and p.position
        }
        owned = self.owned_quantities(all_history=True)
        unknown = sorted(set(actual) - set(owned))
        self.entry_blocker = (
            f"UNOWNED_BROKER_POSITIONS:{unknown}" if unknown else self.deadline_blocker()
        )
        with self.store.db:
            self.store.db.execute("DELETE FROM first4_positions")
            for con_id, quantity in owned.items():
                p = actual.get(con_id)
                broker_quantity = float(p.position) if p else 0
                if not quantity and not broker_quantity:
                    continue  # Positions is a current snapshot; executions retain history.
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
        if (
            generation != self.connection_generation
            or order_generation != self.order_observation_generation
            or self.upstream_lost
            or not self.ib.isConnected()
        ):
            raise ValueError("CONNECTION_CHANGED_DURING_RECONCILIATION")
        # Unrelated contracts/orders are never cancelled or managed by FIRST4.
        self.reconciled, self.problem = True, ""
        self.store.set_meta("reconciled_at", now().isoformat())

    async def chain(self, underlying: Any) -> Any:
        generation = self.connection_generation
        if underlying.conId not in self.chains:
            result = await self.ib.reqSecDefOptParamsAsync(
                underlying.symbol, "", "STK", underlying.conId
            )
            if generation != self.connection_generation:
                raise ValueError("CONNECTION_CHANGED_DURING_OPTION_CHAIN")
            self.chains[underlying.conId] = result
        return self.chains[underlying.conId]

    async def contract_details(self, contract: Any) -> list[Any]:
        """Distinguish request failure from a completed empty contract lookup."""
        request_id = self.ib.client.getReqId()
        future = self.ib.wrapper.startReq(request_id, contract)
        failure = None

        def failed(req_id: int, code: int, message: str, *args: Any) -> None:
            nonlocal failure
            if req_id == request_id:
                failure = RequestError(req_id, code, message)
                if not future.done():
                    future.set_result([])

        self.ib.errorEvent += failed
        try:
            self.ib.client.reqContractDetails(request_id, contract)
            result = await future
            if failure:
                raise failure
            return list(result)
        finally:
            self.ib.errorEvent -= failed
            future.cancel()
            # The API offers no contract-details cancellation. Late callbacks are
            # ignored once these request-specific containers have been removed.
            self.ib.wrapper._futures.pop(request_id, None)
            self.ib.wrapper._results.pop(request_id, None)
            self.ib.wrapper._reqId2Contract.pop(request_id, None)

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
            details = await self.contract_details(
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
                return None
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
        puts = await asyncio.gather(*(qualify(e, "P") for e in dates), return_exceptions=True)
        by_expiry = {}
        for e, result in zip(dates, puts, strict=True):
            # Code 200 can also mean ambiguous metadata. Only this explicit
            # no-security-definition response proves nonexistence.
            if (
                isinstance(result, RequestError)
                and result.code == 200
                and "No security definition has been found" in result.message
            ):
                continue
            if isinstance(result, BaseException):
                raise result
            if result is not None:
                by_expiry[e] = result
        expiry = listed_expiry({e: expiration_at(d) for e, d in by_expiry.items()}, baseline)
        legs = [by_expiry[expiry], await qualify(expiry, "C")]
        if legs[1] is None:
            raise ValueError("OPTION_CONTRACT_UNAVAILABLE_OR_AMBIGUOUS")
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
        generation = self.connection_generation
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
                if generation != self.connection_generation or self.upstream_lost:
                    raise ValueError("CONNECTION_CHANGED_DURING_QUOTES")
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
            for working in self.ib.openTrades():
                working_ids = (
                    [leg.conId for leg in working.contract.comboLegs]
                    if working.contract.secType == "BAG"
                    else [working.contract.conId]
                )
                if not working.isDone() and set(ids).intersection(working_ids):
                    raise ValueError("CONFLICTING_WORKING_ORDER_ON_EXIT_LEGS")
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
        for entry in self.store.unresolved():
            if entry["role"] != "ENTRY":
                continue
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
            await self.reconcile()
            payload["deadline_reconciled"] = True
            with self.store.db:
                self.store.db.execute(
                    "UPDATE first4_orders SET payload=? WHERE reference=?",
                    (json.dumps(payload), entry["reference"]),
                )
            event = self.store.event(entry)
            allocation = entry["reference"].rsplit(":", 1)[0] + ":"
            quantity = self.owned_quantities(allocation)
            self.store.outcome(
                event,
                "ENTRY_RECONCILED_HELD" if any(quantity.values()) else "ENTRY_UNFILLED",
                {"entry_status": entry["status"]},
            )
        if not self.entry_blocker.startswith("UNOWNED_BROKER_POSITIONS"):
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

    async def close_due(self) -> None:
        for entry in self.store.unresolved():
            if entry["role"] != "ENTRY":
                continue
            try:
                await self.close_one(entry["reference"])
            except Exception as exc:
                self.problem = str(exc) or type(exc).__name__
                self.store.set_meta(
                    "exit_exception:" + entry["reference"],
                    {"error": self.problem, "requires_operator": True},
                )

    async def close_one(self, reference: str) -> None:
        found = self.store.order(reference)
        entries = [found] if found else []
        for entry in entries:
            payload = json.loads(entry["payload"])
            if now() < datetime.fromisoformat(payload["exit_at"]):
                continue
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
            event = self.store.event(entry)
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
            if any(k not in {x.conId for x in legs} and v for k, v in allocated_quantities.items()):
                self.reconciled = False
                raise ValueError("UNEXPECTED_ALLOCATION_LEG_RECONCILIATION_REQUIRED")
            prior_exits = [o for o in self.store.allocation_orders(entry) if o["role"] == "EXIT"]
            terminal = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
            working = [
                t
                for t in self.ib.openTrades()
                if t.order.orderRef.startswith(allocation) and not t.isDone()
            ]
            pending = working or any(o["status"] not in terminal for o in prior_exits)
            if pending and (
                not any(q)
                or q[0] == q[1]
                or any(o["reference"] == allocation + "EXIT" for o in prior_exits)
            ):
                status = "EXIT_WORKING" if working else "EXIT_ACKNOWLEDGEMENT_UNRESOLVED"
                if now() >= datetime.fromisoformat(event["close_at"]):
                    status = "EXIT_OVERDUE"
                self.exit_obligation(event, status, q)
                continue
            if min(q) < 0:
                raise ValueError("UNEXPECTED_SHORT_OPTION_POSITION")
            if not any(q):
                entry_fills = self.store.executions(entry["reference"])
                if entry["status"] == "Filled" and not entry_fills:
                    self.problem = "FILLED_ENTRY_MISSING_LEG_EXECUTIONS"
                    self.reconciled = False
                    continue
                if prior_exits:
                    # Submission/status callbacks alone cannot establish closure.
                    # This fresh snapshot also validates all Filled leg executions.
                    await self.reconcile()
                    if any(self.owned_quantities(allocation).values()):
                        continue
                    refreshed = self.store.allocation_orders(entry)
                    if any(o["status"] not in terminal for o in refreshed) or any(
                        t.order.orderRef.startswith(allocation) and not t.isDone()
                        for t in self.ib.openTrades()
                    ):
                        self.exit_obligation(event, "EXIT_ACKNOWLEDGEMENT_UNRESOLVED", q)
                        continue
                self.store.outcome(event, "CLOSED" if entry_fills else "ENTRY_UNFILLED")
                with self.store.db:
                    self.store.db.execute(
                        "UPDATE first4_orders SET payload=json_set(payload,"
                        "'$.management_resolved',1,'$.deadline_reconciled',1) WHERE reference=?",
                        (reference,),
                    )
                continue
            combo_exit = next(
                (o for o in prior_exits if o["reference"] == allocation + "EXIT"), None
            )
            if combo_exit:
                self.exit_obligation(
                    event,
                    "EXIT_OVERDUE"
                    if now() >= datetime.fromisoformat(event["close_at"])
                    else "EXIT_TERMINAL_RESIDUAL_REQUIRES_OPERATOR",
                    q,
                )
                continue
            if now() >= datetime.fromisoformat(event["close_at"]):
                self.exit_obligation(event, "EXIT_OVERDUE", q)
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
                await self.close_order(event, combo, legs, q[0], payload)
            else:
                # A non-atomic partial execution leaves explicit individual close obligations.
                for leg, quantity in zip(legs, q, strict=True):
                    if quantity:
                        if any(
                            o["reference"] == allocation + "EXIT" + str(leg.conId)
                            for o in prior_exits
                        ):
                            self.exit_obligation(event, "LEG_EXIT_RESIDUAL_REQUIRES_OPERATOR", q)
                            continue
                        await self.close_order(event, leg, [leg], quantity, payload, str(leg.conId))

    def exit_obligation(self, event: dict[str, Any], status: str, quantities: list[float]) -> None:
        self.problem = status
        self.store.outcome(event, status, {"remaining_legs": quantities})

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
