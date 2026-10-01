"""Long-option paper admission, durable intent and independent closure/reconciliation."""

import asyncio
import json
import math
import time
from datetime import UTC, datetime, timedelta
from decimal import ROUND_FLOOR
from typing import Any

from stocker_execution.config import (
    MAX_PREMIUM_RISK_PENNIES,
    QUOTE_MAX_AGE_SECONDS,
    FuturesConfig,
)
from stocker_execution.contracts import (
    budget,
    executable_quote,
    fill_conversion,
    key,
    option_price,
    positive,
    quote_check,
    tick_at,
    utc,
    verified_cutoff,
)
from stocker_execution.saxo_data import DataService, MarketState, roots_verified
from stocker_execution.store import TERMINAL, Store, encode

# Idle broker state is re-read on this cadence; pending orders or held exposure
# are reconciled every management cycle. SIM entries require a recent pass.
RECONCILE_INTERVAL_SECONDS = 30
RECONCILE_MAX_AGE_SECONDS = 60
CLOSED_POSITION_FIELDS = (
    "Amount",
    "AssetType",
    "Uic",
    "BuyOrSell",
    "OpenPrice",
    "ClosingPrice",
    "ExecutionTimeOpen",
    "ExecutionTimeClose",
    "OpeningExternalReferenceId",
    "ClosingExternalReferenceId",
    "ProfitLossOnTrade",
    "ProfitLossOnTradeInBaseCurrency",
    "ClosedProfitLoss",
    "ClosedProfitLossInBaseCurrency",
    "CostOpening",
    "CostOpeningInBaseCurrency",
    "CostClosing",
    "CostClosingInBaseCurrency",
)
ORDER_DEADLINE_SECONDS = 20  # a working order unconfirmed after this is reconciled/cancelled
NO_BID_GRACE_SECONDS = 60  # a current quote with no sellable bid this long after the exit: zero


def now() -> datetime:
    return datetime.now(UTC)


class PaperBroker:
    def __init__(self, config: FuturesConfig, store: Store, data: DataService):
        self.config, self.store, self.data = config, store, data
        self.armed = False
        self.reconciled = False
        self.fatal_error = ""
        self.problem = "PREFLIGHT_REQUIRED"
        self.management_problems: dict[str, str] = {}
        self.lock = asyncio.Lock()
        self.last_reconcile = 0.0
        self.preflight_at = 0.0
        self.size_used: dict[tuple[int, float, str], float] = {}
        self.closed_checked = float("-inf")
        self.closed_problem = ""

    def entry_reason(self) -> str:
        if self.fatal_error:
            return self.fatal_error
        if self.config.execution_mode == "DISABLED":
            return "EXECUTION_DISABLED"
        if not self.armed:
            return "PAPER_DISARMED"
        if not self.reconciled:
            return "RECONCILIATION_REQUIRED"
        if not self.data.connected or self.data.session.get("TradeLevel") != "FullTradingAndChat":
            return "DATA_OR_SESSION_UNAVAILABLE"
        if (
            self.config.execution_mode == "SAXO_SIM"
            and time.monotonic() - self.last_reconcile > RECONCILE_MAX_AGE_SECONDS
        ):
            return "RECONCILIATION_STALE"
        return self.problem

    async def preflight(self) -> dict[str, Any]:
        # Only reference/portfolio reads. Never precheck or send a test order here.
        # Same lock as manage(): a reconcile must not interleave with position management.
        async with self.lock:
            await self.data.verify_account()
            await self.reconcile()
            self.preflight_at = time.monotonic() if self.reconciled else 0
        return {
            "non_transmitting": True,
            "execution_mode": self.config.execution_mode,
            "data_environment": self.config.data_environment,
            "reconciled": self.reconciled,
            "problem": self.problem,
            "armed": self.armed,
            "live_orders_disabled": True,
        }

    def arm(self, acknowledgement: str) -> None:
        if acknowledgement != "ENABLE PAPER ONLY" or self.config.execution_mode == "DISABLED":
            raise ValueError("EXPLICIT_PAPER_ARM_REQUIRED")
        if self.fatal_error:
            raise ValueError(self.fatal_error)
        if (
            not self.preflight_at
            or time.monotonic() - self.preflight_at > 60
            or not self.reconciled
        ):
            raise ValueError("FRESH_NON_TRANSMITTING_PREFLIGHT_REQUIRED")
        self.armed = True

    async def prepare(
        self, event: dict[str, Any], state: MarketState, inputs: dict[str, float]
    ) -> dict[str, Any]:
        mapping = self.config.mappings.get(state.market)
        if mapping is None:
            raise ValueError("LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED")
        if not roots_verified(state, mapping.option_root_ids):
            raise ValueError("OPTION_ROOT_NOT_VERIFIED")
        option, distance = await self.select_option(event, state, inputs)
        exit_at = utc(event["exit_at"])
        cutoff = verified_cutoff(option, exit_at)
        context = self.data.option_view(option["uic"], time.time())
        plan: dict[str, Any] = context["costs"]
        if plan["budget_result"] != "WITHIN_BUDGET":
            raise ValueError(plan.get("reason", plan["budget_result"]))
        plan.update(
            option=option,
            cutoff=cutoff.isoformat(),
            exit_at=event["exit_at"],
            fee_evidence=self.data.option_references[option["uic"]]["version"],
            delta_distance=distance,
            expiry_rule=mapping.expiry_rule,
            underlying=state.identity,
            simulated=self.config.execution_mode == "INTERNAL_PAPER",
        )
        return plan

    async def select_option(
        self, event: dict[str, Any], state: MarketState, inputs: dict[str, float]
    ) -> tuple[dict[str, Any], float]:
        # One event pins one UIC, even if the running candidate later changes.
        if event.get("selected_option"):
            selected = event["selected_option"]
            return selected, float(event["delta_distance"])
        distance, _, selected = self.data.rank_candidates(state, event, inputs)[0]
        option = await self.data.option_subscribe(state, selected)
        event.update(selected_option=option, delta_distance=distance)
        self.data.option_required_at[option["uic"]] = max(
            self.data.option_required_at.get(option["uic"], 0),
            utc(event["signal_at"]).timestamp() + 3600,
        )
        self.data.recorder.attach(str(event["id"]), key(option), time.time())
        return option, distance

    def validate_order(self, plan: dict[str, Any], role: str, identity: str) -> None:
        option = plan["option"]
        if (
            option["asset_type"] != "FuturesOption"
            or plan["quantity"] != 1
            or option["environment"] != self.config.data_environment
        ):
            raise ValueError("ONLY_OWNED_LONG_FUTURES_OPTIONS")
        if role == "ENTRY":
            if self.entry_reason():
                raise ValueError(self.entry_reason())
            if self.store.get_meta("paused", False):
                raise ValueError("ENTRIES_PAUSED")
            reservation = self.store.db.execute(
                "SELECT policy_pennies FROM reservations WHERE id=?", (identity,)
            ).fetchone()
            ceiling = (
                min(MAX_PREMIUM_RISK_PENNIES, reservation[0])
                if reservation
                else MAX_PREMIUM_RISK_PENNIES
            )
            if not 0 < plan["cash_pennies"] <= ceiling:
                raise ValueError("MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET")
        elif role == "EXIT":
            if self.store.exposure(identity) != 1:
                raise ValueError("SELL_ONLY_EXISTING_OWNED_LONG_OPTION")
            if self.config.execution_mode == "SAXO_SIM":
                row = self.store.db.execute(
                    "SELECT quantity FROM positions WHERE con_id=?", (option["uic"],)
                ).fetchone()
                if not self.reconciled or not row or row[0] < 1:
                    raise ValueError("BROKER_OWNERSHIP_NOT_RECONCILED")
        else:
            raise ValueError("INVALID_ORDER_ROLE")

    def payload(
        self, plan: dict[str, Any], role: str, reference: str, price: float
    ) -> dict[str, Any]:
        return {
            "AccountKey": self.data.client.oauth.account_key,
            "AssetType": "FuturesOption",
            "Uic": plan["option"]["uic"],
            "Amount": 1,
            "BuySell": "Buy" if role == "ENTRY" else "Sell",
            "ToOpenClose": "ToOpen" if role == "ENTRY" else "ToClose",
            "OrderType": "Limit",
            "OrderPrice": price,
            "OrderDuration": {"DurationType": "DayOrder"},
            "ExternalReference": reference,
            "ManualOrder": False,
        }

    async def enter(self, event: dict[str, Any], plan: dict[str, Any]) -> str:
        async with self.lock:
            self.validate_order(plan, "ENTRY", event["id"])
            if now() >= utc(event["signal_at"]) + timedelta(
                seconds=self.config.entry_deadline_seconds
            ):
                return "STALE_SIGNAL_NO_REPLAY"
            reason = self.store.reserve(event["id"], plan)
            if reason:
                return reason
            self.data.recorder.link_trade(event["id"], True, time.time())
            await self.submit(event["id"], plan, "ENTRY", plan["limit"])
            return ""

    async def submit(self, identity: str, plan: dict[str, Any], role: str, price: float) -> None:
        self.validate_order(plan, role, identity)
        order_id = int(
            self.store.db.execute("SELECT COALESCE(MAX(order_id),0)+1 FROM orders").fetchone()[0]
        )
        deadline = now() + timedelta(seconds=ORDER_DEADLINE_SECONDS)
        if role == "ENTRY":
            signal = self.store.db.execute(
                "SELECT signal_at FROM signals WHERE id=?", (identity,)
            ).fetchone()
            deadline = min(
                deadline, utc(signal[0]) + timedelta(seconds=self.config.entry_deadline_seconds)
            )
        reference = self.store.prepare_order(
            identity,
            role,
            order_id,
            deadline.isoformat(),
            {
                "option": plan["option"],
                "role": role,
                "limit": price,
                "mode": self.config.execution_mode,
            },
        )
        if self.config.execution_mode == "INTERNAL_PAPER":
            self.internal_fill(identity, reference, plan, role, price)
            return
        if self.config.execution_mode != "SAXO_SIM" or self.config.data_environment != "SAXO_SIM":
            raise ValueError("LIVE_ORDERS_DISABLED")
        payload = self.payload(plan, role, reference, price)
        transmitted = False
        try:
            precheck = await self.data.client.request(
                "POST",
                "/trade/v2/orders/precheck",
                body={**payload, "FieldGroups": ["Costs"]},
                execution=True,
            )
            if (
                precheck.get("ErrorInfo")
                or precheck.get("PreTradeDisclaimers")
                or precheck.get("PreCheckResult") not in {"Ok", "Success"}
            ):
                with self.store.db:
                    self.store.db.execute(
                        "UPDATE orders SET status='Inactive' WHERE reference=?", (reference,)
                    )
                raise ValueError("PRETRADE_CHECK_OR_DISCLAIMER_REQUIRES_USER_ACTION")
            if role == "ENTRY":
                self.check_broker_cost(plan, precheck)
            # Revalidate after awaited I/O; disconnect/disarm/stale data cannot race admission.
            self.validate_order(plan, role, identity)
            current_option, price_state = self.data.options[plan["option"]["uic"]]
            executable_quote(
                current_option,
                price_state.value or {},
                self.data.quote_receipt(price_state),
                now(),
            )
            if now() >= deadline:
                raise ValueError("ENTRY_PREFLIGHT_EXPIRED")
            transmitted = True  # durable SUBMITTING intent already exists
            result = await self.data.client.request(
                "POST",
                "/trade/v2/orders",
                body=payload,
                execution=True,
            )
            if not result.get("OrderId") or result.get("ErrorInfo"):
                raise ValueError("AMBIGUOUS_ORDER_RESPONSE_RECONCILE_REQUIRED")
            self.store.set_meta("broker_order:" + reference, str(result["OrderId"]))
            with self.store.db:
                self.store.db.execute(
                    "UPDATE orders SET status='Submitted' WHERE reference=?", (reference,)
                )
        except Exception as exc:
            if transmitted:
                self.reconciled = False
                self.problem = "ORDER_STATUS_UNCERTAIN_RECONCILE_REQUIRED"
                self.store.decision(identity, "ORDER_STATUS_UNCERTAIN", self.problem)
            else:
                with self.store.db:
                    self.store.db.execute(
                        "UPDATE orders SET status='Inactive' WHERE reference=?", (reference,)
                    )
                reason = str(exc) if isinstance(exc, ValueError) else "PRECHECK_UNAVAILABLE"
                self.store.decision(
                    identity, "SKIPPED" if role == "ENTRY" else "EXIT_EXCEPTION", reason
                )
                self.data.recorder.annotate(identity, {"order_skip_reason": reason})
            # Never resubmit after a timeout, even when no broker order ID arrived.

    def check_broker_cost(self, plan: dict[str, Any], result: dict[str, Any]) -> None:
        currency = result.get("EstimatedCashRequiredCurrency")
        if currency not in {"GBP", "USD"} or self.data.account_currency not in {"GBP", "USD"}:
            raise ValueError("BROKER_COST_CURRENCY_UNVERIFIED")
        fx = quote_check(self.data.fx.value or {}, self.data.quote_receipt(self.data.fx), now())
        rate = 1 / float(fx["Bid"])
        cash = positive(result.get("EstimatedCashRequired"), "BROKER_REQUIRED_CASH")
        cash *= rate if currency == "USD" else 1
        fee = positive(result.get("EstimatedTotalCostInAccountCurrency"), "BROKER_FEES")
        fee *= rate if self.data.account_currency == "USD" else 1
        premium = plan["limit"] * plan["option"]["price_factor"] * rate
        total = max(cash, premium + fee) + plan["fee_per_side_gbp"]
        if (
            math.ceil(total * 100) > MAX_PREMIUM_RISK_PENNIES
            or math.ceil(total * 100) > plan["cash_pennies"]
        ):
            raise ValueError("MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET")

    def internal_fill(
        self, identity: str, reference: str, plan: dict[str, Any], role: str, price: float
    ) -> None:
        option, at = plan["option"], now()
        state = self.data.options[option["uic"]][1]
        try:
            fx = quote_check(self.data.fx.value or {}, self.data.quote_receipt(self.data.fx), at)
            conversion = fill_conversion(
                1 / float(fx["Bid" if role == "ENTRY" else "Ask"]), plan["fx_markup"], role
            )
            if role == "ENTRY":
                revised = budget(option, price, plan["fee_per_side_gbp"] * 2, conversion)
                plan.update(revised)
                self.validate_order(plan, role, identity)
            plan.update(fx=conversion, fx_at=self.data.fx.receipt)
            quote = executable_quote(
                self.data.options[option["uic"]][0],
                state.value or {},
                self.data.quote_receipt(state),
                at,
            )
            side = "Ask" if role == "ENTRY" else "Bid"
            touch = float(quote[side])
            if (role == "ENTRY" and price < touch + tick_at(option, touch, 1) - 1e-10) or (
                role == "EXIT" and price > touch - tick_at(option, touch, -1) + 1e-10
            ):
                raise ValueError("PRICE_MOVED_NO_ASSUMED_FILL")
            size_at = state.size_times.get(side)
            standing = self.data.size_receipt(state, side)
            if (
                size_at is None
                or standing is None
                or not 0 <= at.timestamp() - standing <= QUOTE_MAX_AGE_SECONDS
            ):
                raise ValueError("AVAILABLE_OPTION_SIZE_STALE")
            size = positive(state.sizes().get(side), "AVAILABLE_OPTION_SIZE")
            receipt = size_at
            usage_key = (option["uic"], receipt, side)
            if size - self.size_used.get(usage_key, 0) < 1:
                raise ValueError("INSUFFICIENT_DISPLAYED_OPTION_SIZE")
            self.size_used = {
                k: v for k, v in self.size_used.items() if k[1] >= at.timestamp() - 60
            }
            self.size_used[usage_key] = self.size_used.get(usage_key, 0) + 1
        except ValueError as exc:
            with self.store.db:
                self.store.db.execute(
                    "UPDATE orders SET status='Cancelled' WHERE reference=?", (reference,)
                )
            self.store.decision(
                identity, "SKIPPED" if role == "ENTRY" else "EXIT_EXCEPTION", str(exc)
            )
            return
        self.book_internal_fill(
            identity, reference, plan, role, price, at, plan["fee_per_side_gbp"]
        )

    def book_internal_fill(
        self,
        identity: str,
        reference: str,
        plan: dict[str, Any],
        role: str,
        price: float,
        at: datetime,
        fees_gbp: float,
        reason: str = "",
    ) -> None:
        option = plan["option"]
        with self.store.db:
            self.store.record_fill(
                {
                    "exec_id": "INTERNAL-" + reference,
                    "reference": reference,
                    "con_id": option["uic"],
                    "quantity": 1,
                    "price": price,
                    "side": "BOT" if role == "ENTRY" else "SLD",
                    "at": at.isoformat(),
                    "fx": plan["fx"],
                    "fx_at": datetime.fromtimestamp(plan["fx_at"], UTC).isoformat(),
                },
                commit=False,
            )
            self.store.db.execute(
                "UPDATE fills SET commission=?,commission_currency='GBP' WHERE reference=?",
                (fees_gbp, reference),
            )
            self.store.db.execute(
                "UPDATE orders SET status='Filled',filled=1,remaining=0 WHERE reference=?",
                (reference,),
            )
            quantity = sum(
                self.store.exposure(r["id"])
                for r in self.store.active()
                if json.loads(r["plan"])["option"]["uic"] == option["uic"]
            )
            self.store.db.execute(
                "INSERT OR REPLACE INTO positions VALUES(?,?,?)",
                (option["uic"], quantity, encode({"internally_simulated": True})),
            )
            self.store.db.execute(
                "UPDATE reservations SET plan=? WHERE id=?", (encode(plan), identity)
            )
        self.store.decision(identity, "INTERNALLY_SIMULATED_FILL", reason)
        self.data.recorder.annotate(
            identity,
            {
                "last_fill": {
                    "role": role,
                    "quantity": 1,
                    "price": price,
                    "fees_gbp": fees_gbp,
                    "at": at.isoformat(),
                    "basis": "INTERNALLY_SIMULATED",
                    **({"write_off": reason} if reason else {}),
                },
            },
        )

    def write_off_reason(self, plan: dict[str, Any], exit_at: datetime) -> str:
        """INTERNAL_PAPER only: why an owed exit cannot be sold and is booked at zero, or "".

        Far-out same-day options often lose their bid before expiry; selling one tick below a
        one-tick bid is zero too. Unknown is never "no bid": a missing, stale, delayed or paused
        quote keeps the exit retrying. Past the last trading time nothing can be sold.
        """
        if self.config.execution_mode != "INTERNAL_PAPER":
            return ""
        at = now()
        if at >= utc(plan["cutoff"]):
            return "WRITTEN_OFF_AFTER_LAST_TRADE"
        if at < exit_at + timedelta(seconds=NO_BID_GRACE_SECONDS):
            return ""
        state = self.data.options.get(plan["option"]["uic"])
        if not state:
            return ""
        price = state[1]
        quote = (price.value or {}).get("Quote") or {}
        receipt = self.data.quote_receipt(price)
        if (
            receipt is None
            or not 0 <= at.timestamp() - receipt <= QUOTE_MAX_AGE_SECONDS
            or price.problem
            or quote.get("DelayedByMinutes") != 0
        ):
            return ""
        bid = quote.get("Bid")
        if quote.get("PriceTypeBid") == "NoMarket" or not isinstance(bid, (int, float)) or bid <= 0:
            return "WRITTEN_OFF_NO_BID"
        if option_price(plan["option"], float(bid), ROUND_FLOOR, -1) <= 0:
            return "WRITTEN_OFF_ONE_TICK_BID"
        return ""

    def write_off(self, identity: str, plan: dict[str, Any], reason: str) -> None:
        """Close an unsellable paper option at zero; no exit fee, as no trade takes place."""
        order_id = int(
            self.store.db.execute("SELECT COALESCE(MAX(order_id),0)+1 FROM orders").fetchone()[0]
        )
        reference = self.store.prepare_order(
            identity,
            "EXIT",
            order_id,
            (now() + timedelta(seconds=ORDER_DEADLINE_SECONDS)).isoformat(),
            {"option": plan["option"], "role": "EXIT", "limit": 0.0, "mode": "INTERNAL_PAPER"},
        )
        self.book_internal_fill(identity, reference, plan, "EXIT", 0.0, now(), 0.0, reason)

    async def portfolio(self, path: str) -> list[dict[str, Any]]:
        rows = []
        for page in range(10):
            result = await self.data.client.request(
                "GET",
                path,
                params={
                    "$top": 100,
                    "$skip": page * 100,
                    "AccountKey": self.data.client.oauth.account_key,
                },
            )
            rows.extend(result.get("Data", []))
            if not result.get("__next"):
                return rows
        raise ValueError("PORTFOLIO_PAGINATION_LIMIT_RECONCILIATION_INCOMPLETE")

    async def refresh_closed_positions(self) -> None:
        """SAXO_SIM: keep Saxo's own closed-position figures for SLRNO orders.

        Evidence and display only. Saxo removes intraday closed positions after
        settlement, so this runs every ten minutes while SIM execution is configured.
        """
        if (
            self.config.execution_mode != "SAXO_SIM"
            or not self.data.account_verified
            or time.monotonic() - self.closed_checked < 600
        ):
            return
        self.closed_checked = time.monotonic()
        references = {r[0] for r in self.store.db.execute("SELECT reference FROM orders")}
        if not references:
            return
        try:
            rows = await self.portfolio("/port/v1/closedpositions")
        except ValueError as exc:
            self.closed_problem = str(exc)
            return
        self.closed_problem = ""
        for row in rows:
            closed = row.get("ClosedPosition") or {}
            ids = (
                closed.get("OpeningExternalReferenceId"),
                closed.get("ClosingExternalReferenceId"),
            )
            matched = sorted(str(r) for r in ids if r in references)
            unique = str(row.get("ClosedPositionUniqueId") or "")
            if not matched or not unique or self.store.get_meta("broker_closed:" + unique):
                continue
            record = {k: closed.get(k) for k in CLOSED_POSITION_FIELDS}
            with self.store.db:
                self.store.db.execute(
                    "INSERT OR IGNORE INTO futures_meta VALUES(?,?)",
                    ("broker_closed:" + unique, encode(record)),
                )
                self.store.audit(matched[-1], "SAXO_SIM_CLOSED_POSITION", record)

    async def reconcile(self) -> None:
        self.reconciled = False
        if self.config.execution_mode != "SAXO_SIM":
            # Internal obligations derive solely from the durable internal ledger.
            held: dict[int, float] = {}
            with self.store.db:
                for reservation in self.store.active():
                    uic = json.loads(reservation["plan"])["option"]["uic"]
                    held[uic] = held.get(uic, 0) + self.store.exposure(reservation["id"])
                    for order in self.store.orders(reservation["id"]):
                        if order["status"] not in TERMINAL:
                            filled = sum(
                                f["quantity"]
                                for f in self.store.fills(reservation["id"])
                                if f["reference"] == order["reference"]
                            )
                            if filled not in (0, 1):
                                raise ValueError("INTERNAL_LEDGER_EXPOSURE_EXCEPTION")
                            self.store.db.execute(
                                "UPDATE orders SET status=?,filled=?,remaining=? WHERE reference=?",
                                (
                                    "Filled" if filled else "Cancelled",
                                    filled,
                                    1 - filled,
                                    order["reference"],
                                ),
                            )
                current = {
                    r["con_id"]: r["quantity"]
                    for r in self.store.db.execute("SELECT con_id,quantity FROM positions")
                }
                if current != held:
                    self.store.db.execute("DELETE FROM positions")
                    for uic, quantity in held.items():
                        self.store.db.execute(
                            "INSERT INTO positions VALUES(?,?,?)",
                            (uic, quantity, encode({"internally_simulated": True})),
                        )
                self.store.mark_reconciled()
            self.reconciled = True
            self.problem = ""
            self.last_reconcile = time.monotonic()
            return
        if not self.data.account_verified or not self.data.client.sim_account_verified:
            raise ValueError("SIM_ACCOUNT_NOT_VERIFIED")
        positions = await self.portfolio("/port/v1/positions/me")
        open_orders = await self.portfolio("/port/v1/orders/me")
        local = {
            r["reference"]: dict(r)
            for r in self.store.db.execute(
                "SELECT o.* FROM orders o JOIN reservations r ON r.id=o.event_id WHERE r.active=1"
            )
        }
        unknown = [o for o in open_orders if o.get("ExternalReference") not in local]
        held = {}
        for position in positions:
            base = position.get("PositionBase", {})
            if base.get("Amount", 0):
                if (
                    base.get("AccountId") != self.data.account_id
                    or base.get("AssetType") != "FuturesOption"
                ):
                    self.problem = "UNACCOUNTED_BROKER_POSITION_OR_ACCOUNT"
                    return
                held[int(base["Uic"])] = held.get(int(base["Uic"]), 0) + float(base["Amount"])
        with self.store.db:
            self.store.db.execute("DELETE FROM positions")
            for uic, quantity in held.items():
                self.store.db.execute(
                    "INSERT INTO positions VALUES(?,?,?)",
                    (uic, quantity, encode({"provider": "SAXO", "environment": "SAXO_SIM"})),
                )
        for reference, order in local.items():
            if order["status"] in TERMINAL:
                continue
            remote = next((o for o in open_orders if o.get("ExternalReference") == reference), None)
            broker_id = self.store.get_meta("broker_order:" + reference)
            if remote:
                payload = json.loads(order["payload"])
                if (
                    remote.get("AccountId") != self.data.account_id
                    or remote.get("Uic") != payload["option"]["uic"]
                    or remote.get("AssetType") != "FuturesOption"
                    or remote.get("BuySell") != ("Buy" if order["role"] == "ENTRY" else "Sell")
                ):
                    raise ValueError("BROKER_ORDER_IDENTITY_MISMATCH")
                if broker_id != str(remote["OrderId"]):
                    broker_id = str(remote["OrderId"])
                    self.store.set_meta("broker_order:" + reference, broker_id)
            if not broker_id:
                self.problem = "AMBIGUOUS_ORDER_REQUIRES_BROKER_AUDIT"
                return
            result = await self.data.client.request(
                "GET",
                "/cs/v1/audit/orderactivities",
                params={
                    "OrderId": broker_id,
                    "EntryType": "Last",
                    "$top": 100,
                },
            )
            if result.get("__next") or len(result.get("Data", [])) != 1:
                self.problem = "ORDER_AUDIT_INCOMPLETE"
                return
            evidence = result["Data"][0]
            if str(evidence.get("OrderId")) != broker_id:
                raise ValueError("BROKER_ORDER_AUDIT_ID_MISMATCH")
            self.apply_order_evidence(order, evidence)
        expected: dict[int, float] = {}
        for reservation in self.store.active():
            uic = json.loads(reservation["plan"])["option"]["uic"]
            expected[uic] = expected.get(uic, 0) + self.store.exposure(reservation["id"])
        if unknown or any(
            held.get(uic, 0) != expected.get(uic, 0) for uic in set(held) | set(expected)
        ):
            self.problem = "UNACCOUNTED_BROKER_ORDERS_OR_EXPOSURE"
            return
        with self.store.db:
            self.store.mark_reconciled()
        self.problem = ""
        self.reconciled = True
        self.last_reconcile = time.monotonic()

    def apply_order_evidence(self, order: dict[str, Any], evidence: dict[str, Any]) -> None:
        with self.store.db:
            self._apply_order_evidence(order, evidence)

    def _apply_order_evidence(self, order: dict[str, Any], evidence: dict[str, Any]) -> None:
        if evidence.get("SubStatus") != "Confirmed":
            raise ValueError("ORDER_TERMINAL_STATE_NOT_CONFIRMED")
        filled = float(evidence.get("FilledAmount", 0))
        seen = {k: evidence.get(k) for k in ("LogId", "Status", "FilledAmount", "AveragePrice")}
        if (
            float(order["filled"]) == filled
            and self.store.get_meta("evidence:" + order["reference"]) == seen
        ):
            return  # the same audit entry as the last pass: nothing new to record
        known = [
            f for f in self.store.fills(order["event_id"]) if f["reference"] == order["reference"]
        ]
        previous = sum(float(f["quantity"]) for f in known)
        if not 0 <= previous <= filled <= 1:
            raise ValueError("FILL_CORRECTION_REQUIRES_RECONCILIATION")
        if filled > previous:
            plan_row = self.store.db.execute(
                "SELECT plan FROM reservations WHERE id=?", (order["event_id"],)
            ).fetchone()
            plan = json.loads(plan_row[0])
            prior_cash = sum(f["quantity"] * f["price"] for f in known)
            average = positive(evidence.get("AveragePrice"), "BROKER_AVERAGE_FILL")
            price = (average * filled - prior_cash) / (filled - previous)
            self.store.record_fill(
                {
                    "exec_id": "SAXO-LOG-" + str(evidence["LogId"]),
                    "reference": order["reference"],
                    "con_id": plan["option"]["uic"],
                    "quantity": filled - previous,
                    "price": price,
                    "side": "BOT" if order["role"] == "ENTRY" else "SLD",
                    "at": evidence["ActivityTime"],
                    "fx": None,
                    "fx_at": None,
                },
                commit=False,
            )
            self.data.recorder.annotate(
                order["event_id"],
                {
                    "last_broker_fill": {
                        k: evidence.get(k)
                        for k in (
                            "OrderId",
                            "LogId",
                            "ActivityTime",
                            "FilledAmount",
                            "AveragePrice",
                        )
                    },
                    "basis": "SAXO_SIM_BROKER_EVIDENCE_COSTS_PROVISIONAL",
                },
            )
        status = {"FinalFill": "Filled", "Cancelled": "Cancelled", "Expired": "Cancelled"}.get(
            evidence["Status"], "Submitted"
        )
        self.store.db.execute(
            "UPDATE orders SET status=?,filled=?,remaining=? WHERE reference=?",
            (status, filled, 1 - filled, order["reference"]),
        )
        self.store.audit(order["reference"], "SAXO_SIM_ORDER_EVIDENCE", evidence)
        # Inside the caller's transaction: Store.set_meta() would commit the fill early.
        self.store.db.execute(
            "INSERT OR REPLACE INTO futures_meta VALUES (?,?)",
            ("evidence:" + order["reference"], encode(seen)),
        )

    def reconcile_due(self) -> bool:
        if not self.reconciled:
            return True
        if time.monotonic() - self.last_reconcile >= RECONCILE_INTERVAL_SECONDS:
            return True
        # A Saxo order/position event since the last pass: re-read broker state now.
        if self.data.last_activity > self.last_reconcile:
            return True
        return any(
            self.store.exposure(r["id"])
            or any(o["status"] not in TERMINAL for o in self.store.orders(r["id"]))
            for r in self.store.active()
        )

    async def manage(self) -> None:
        async with self.lock:
            if self.reconcile_due():
                await self.reconcile()
            for reservation in self.store.active():
                identity, plan = reservation["id"], json.loads(reservation["plan"])
                try:
                    if not self.reconciled:
                        raise ValueError(self.problem or "RECONCILIATION_REQUIRED")
                    if self.data.connected:
                        await self.data.restore_option(plan)
                    for order in self.store.orders(identity):
                        if order["status"] not in TERMINAL and now() >= utc(order["deadline"]):
                            broker_id = self.store.get_meta("broker_order:" + order["reference"])
                            if (
                                self.config.execution_mode == "SAXO_SIM"
                                and broker_id
                                and not self.store.get_meta("cancel_attempt:" + order["reference"])
                            ):
                                # Record before sending: a lost response never causes blind retries.
                                self.store.set_meta(
                                    "cancel_attempt:" + order["reference"], now().isoformat()
                                )
                                await self.data.client.request(
                                    "DELETE",
                                    f"/trade/v2/orders/{broker_id}",
                                    params={"AccountKey": self.data.client.oauth.account_key},
                                    execution=True,
                                )
                            raise ValueError("PENDING_ORDER_RECONCILIATION_REQUIRED")
                    exit_at = utc(reservation["exit_at"])
                    reason = (
                        self.write_off_reason(plan, exit_at)
                        if self.store.exposure(identity) > 0 and now() >= exit_at
                        else ""
                    )
                    if reason:
                        self.write_off(identity, plan, reason)
                    elif self.store.exposure(identity) > 0 and now() >= exit_at:
                        if now() >= utc(plan["cutoff"]):
                            raise ValueError("FAILED_CLOSURE_EXPIRY_EXPOSURE_EXCEPTION")
                        state = self.data.options.get(plan["option"]["uic"])
                        if not state:
                            raise ValueError("OWNED_OPTION_QUOTE_UNAVAILABLE")
                        q = quote_check(
                            state[1].value or {}, self.data.quote_receipt(state[1]), now()
                        )
                        fx = quote_check(
                            self.data.fx.value or {}, self.data.quote_receipt(self.data.fx), now()
                        )
                        plan["fx"] = fill_conversion(
                            1 / float(fx["Ask"]), plan["fx_markup"], "EXIT"
                        )
                        plan["fx_at"] = self.data.fx.receipt
                        # Internal sale includes one tick adverse slippage.
                        price = option_price(
                            plan["option"],
                            float(q["Bid"]),
                            ROUND_FLOOR,
                            -1 if self.config.execution_mode == "INTERNAL_PAPER" else 0,
                        )
                        positive(price, "EXIT_BID")
                        exits = [o for o in self.store.orders(identity) if o["role"] == "EXIT"]
                        if len(exits) >= 3:
                            raise ValueError("FAILED_CLOSURE_REQUIRES_OPERATOR")
                        await self.submit(identity, plan, "EXIT", price)
                    position = self.store.db.execute(
                        "SELECT quantity FROM positions WHERE con_id=?", (plan["option"]["uic"],)
                    ).fetchone()
                    others = sum(
                        self.store.exposure(r["id"])
                        for r in self.store.active()
                        if r["id"] != identity
                        and json.loads(r["plan"])["option"]["uic"] == plan["option"]["uic"]
                    )
                    attributable = (position[0] if position else 0) - others
                    if self.store.confirm_closed(identity, attributable):
                        self.data.recorder.link_trade(identity, False, time.time())
                        self.data.recorder.annotate(
                            identity,
                            {"outcome": "VERIFIED_FLAT", "mode": self.config.execution_mode},
                        )
                        self.management_problems.pop(identity, None)
                    elif identity in self.management_problems:
                        # Resolved without closure; a recurrence is audited again.
                        self.management_problems.pop(identity)
                except Exception as exc:
                    reason = str(exc) if isinstance(exc, ValueError) else "MANAGEMENT_EXCEPTION"
                    # Audit transitions, not every two-second retry of the same exception.
                    if self.management_problems.get(identity) != reason:
                        self.management_problems[identity] = reason
                        self.store.decision(identity, "EXPOSURE_EXCEPTION", reason)
