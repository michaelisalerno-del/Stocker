"""Long-option paper admission, durable intent and independent closure/reconciliation."""

import asyncio
import json
import math
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from stocker_execution.config import FuturesConfig
from stocker_execution.contracts import budget, key, positive, quote_check, utc, verified_cutoff
from stocker_execution.rules import NY, model_delta
from stocker_execution.saxo_data import DataService, MarketState
from stocker_execution.store import TERMINAL, Store, encode


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
        return self.problem

    async def preflight(self) -> dict[str, Any]:
        # Only reference/portfolio reads. Never precheck or send a test order here.
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
        if state.identity is None or state.option_root != mapping.option_root_id:
            raise ValueError("OPTION_ROOT_NOT_VERIFIED")
        at, exit_at = utc(event["signal_at"]), utc(event["exit_at"])
        candidates = []
        for selected in state.option_space:
            if selected.get("PutCall") != ("Call" if event["right"] == "C" else "Put"):
                continue
            expiry_text = str(selected.get("Expiry", ""))
            if expiry_text[:10] != at.astimezone(NY).date().isoformat():
                continue
            # Date-only expiry is insufficient for a model using time-to-expiry.
            if "T" not in expiry_text or utc(expiry_text).time().isoformat() == "00:00:00":
                continue
            expiry = utc(expiry_text)
            delta = model_delta(
                inputs["futures_price"],
                float(selected["StrikePrice"]),
                inputs["rv15"],
                at,
                expiry,
                str(event["right"]),
            )
            candidates.append(
                (
                    abs(delta - float(event["target_delta"])),
                    float(selected["StrikePrice"]),
                    selected,
                )
            )
        if not candidates:
            raise ValueError("NO_VERIFIED_REAL_0DTE_EXPIRY_TIME")
        distance, _, selected = min(candidates, key=lambda x: (x[0], x[1]))
        if distance > mapping.delta_tolerance:
            raise ValueError("FROZEN_DELTA_OUTSIDE_APPROVED_TOLERANCE")
        option = await self.data.option_subscribe(state, selected)
        self.data.recorder.attach(str(event["id"]), key(option), time.time())
        cutoff = verified_cutoff(option, exit_at)
        price = self.data.options[option["uic"]][1]
        q = quote_check(price.value or {}, price.receipt, now())
        fx = quote_check(self.data.fx.value or {}, self.data.fx.receipt, now())
        if option["currency"] != "USD":
            raise ValueError("CURRENCY_CONVERSION_PAIR_UNVERIFIED")
        # Buy at ask + one tick internally; broker limit is the same conservative ceiling.
        limit = (math.ceil(float(q["Ask"]) / option["tick_size"]) + 1) * option["tick_size"]
        plan = budget(option, limit, mapping.fee_per_side_gbp * 2, 1 / float(fx["Bid"]))
        plan.update(
            cutoff=cutoff.isoformat(),
            exit_at=event["exit_at"],
            quote_at=price.receipt,
            fx_at=self.data.fx.receipt,
            fee_per_side_gbp=mapping.fee_per_side_gbp,
            fee_evidence=mapping.fee_evidence,
            delta_distance=distance,
            underlying=state.identity,
            simulated=self.config.execution_mode == "INTERNAL_PAPER",
        )
        return plan

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
            if not 0 < plan["cash_pennies"] <= 1000:
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
            if now() >= utc(event["signal_at"]) + timedelta(seconds=20):
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
        deadline = now() + timedelta(seconds=20)
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
            price_state = self.data.options[plan["option"]["uic"]][1]
            quote_check(price_state.value or {}, price_state.receipt, now())
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
        fx = quote_check(self.data.fx.value or {}, self.data.fx.receipt, now())
        rate = 1 / float(fx["Bid"])
        cash = positive(result.get("EstimatedCashRequired"), "BROKER_REQUIRED_CASH")
        cash *= rate if currency == "USD" else 1
        fee = positive(result.get("EstimatedTotalCostInAccountCurrency"), "BROKER_FEES")
        fee *= rate if self.data.account_currency == "USD" else 1
        premium = plan["limit"] * plan["option"]["price_factor"] * rate
        total = max(cash, premium + fee) + plan["fee_per_side_gbp"]
        if math.ceil(total * 100) > 1000 or math.ceil(total * 100) > plan["cash_pennies"]:
            raise ValueError("MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET")

    def internal_fill(
        self, identity: str, reference: str, plan: dict[str, Any], role: str, price: float
    ) -> None:
        option, at = plan["option"], now()
        state = self.data.options[option["uic"]][1]
        try:
            quote = quote_check(state.value or {}, state.receipt, at)
            side = "Ask" if role == "ENTRY" else "Bid"
            if (role == "ENTRY" and price < float(quote[side]) + option["tick_size"] - 1e-10) or (
                role == "EXIT" and price > float(quote[side]) - option["tick_size"] + 1e-10
            ):
                raise ValueError("PRICE_MOVED_NO_ASSUMED_FILL")
            if state.size_receipt is None or not 0 <= at.timestamp() - state.size_receipt <= 5:
                raise ValueError("AVAILABLE_OPTION_SIZE_STALE")
            details = (state.value or {}).get("PriceInfoDetails") or {}
            size = positive(details.get(side + "Size"), "AVAILABLE_OPTION_SIZE")
            receipt = state.size_receipt
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
            }
        )
        with self.store.db:
            self.store.db.execute(
                "UPDATE fills SET commission=?,commission_currency='GBP' WHERE reference=?",
                (plan["fee_per_side_gbp"], reference),
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
        self.store.decision(identity, "INTERNALLY_SIMULATED_FILL")
        self.data.recorder.annotate(
            identity,
            {
                "last_fill": {
                    "role": role,
                    "quantity": 1,
                    "price": price,
                    "fees_gbp": plan["fee_per_side_gbp"],
                    "at": at.isoformat(),
                    "basis": "INTERNALLY_SIMULATED",
                },
            },
        )

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
                self.store.db.execute("DELETE FROM positions")
                for uic, quantity in held.items():
                    self.store.db.execute(
                        "INSERT INTO positions VALUES(?,?,?)",
                        (uic, quantity, encode({"internally_simulated": True})),
                    )
            self.reconciled = True
            self.problem = ""
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
        self.problem = ""
        self.reconciled = True
        self.last_reconcile = time.monotonic()

    def apply_order_evidence(self, order: dict[str, Any], evidence: dict[str, Any]) -> None:
        if evidence.get("SubStatus") != "Confirmed":
            raise ValueError("ORDER_TERMINAL_STATE_NOT_CONFIRMED")
        filled = float(evidence.get("FilledAmount", 0))
        previous = float(order["filled"])
        if not 0 <= previous <= filled <= 1:
            raise ValueError("FILL_CORRECTION_REQUIRES_RECONCILIATION")
        if filled > previous:
            plan_row = self.store.db.execute(
                "SELECT plan FROM reservations WHERE id=?", (order["event_id"],)
            ).fetchone()
            plan = json.loads(plan_row[0])
            prior_cash = sum(
                f["quantity"] * f["price"]
                for f in self.store.fills(order["event_id"])
                if f["reference"] == order["reference"]
            )
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
                }
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
        with self.store.db:
            self.store.db.execute(
                "UPDATE orders SET status=?,filled=?,remaining=? WHERE reference=?",
                (status, filled, 1 - filled, order["reference"]),
            )
            self.store.audit(order["reference"], "SAXO_SIM_ORDER_EVIDENCE", evidence)

    async def manage(self) -> None:
        async with self.lock:
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
                    if self.store.exposure(identity) > 0 and now() >= utc(reservation["exit_at"]):
                        if now() >= utc(plan["cutoff"]):
                            raise ValueError("FAILED_CLOSURE_EXPIRY_EXPOSURE_EXCEPTION")
                        state = self.data.options.get(plan["option"]["uic"])
                        if not state:
                            raise ValueError("OWNED_OPTION_QUOTE_UNAVAILABLE")
                        q = quote_check(state[1].value or {}, state[1].receipt, now())
                        fx = quote_check(self.data.fx.value or {}, self.data.fx.receipt, now())
                        plan["fx"], plan["fx_at"] = 1 / float(fx["Ask"]), self.data.fx.receipt
                        # Internal sale includes one tick adverse slippage.
                        price = (
                            math.floor(float(q["Bid"]) / plan["option"]["tick_size"])
                            * plan["option"]["tick_size"]
                        )
                        if self.config.execution_mode == "INTERNAL_PAPER":
                            price -= plan["option"]["tick_size"]
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
                except Exception as exc:
                    self.management_problems[identity] = (
                        str(exc) if isinstance(exc, ValueError) else "MANAGEMENT_EXCEPTION"
                    )
                    self.store.decision(
                        identity, "EXPOSURE_EXCEPTION", self.management_problems[identity]
                    )
