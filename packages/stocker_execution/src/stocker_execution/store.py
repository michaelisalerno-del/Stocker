"""Fresh execution namespace, durable intent, reservations and broker evidence."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def stamp() -> str:
    return datetime.now(UTC).isoformat()


def encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=5)
        self.db.row_factory = sqlite3.Row
        tables = {
            x[0] for x in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if tables and "futures_meta" not in tables:
            self.db.close()
            raise ValueError("FRESH_FUTURES_LEDGER_REQUIRED")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS futures_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS signals (
          id TEXT PRIMARY KEY, market TEXT NOT NULL, rule_version TEXT NOT NULL,
          signal_at TEXT NOT NULL, exit_at TEXT NOT NULL, decision TEXT NOT NULL,
          reason TEXT NOT NULL, detail TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS signal_history ON signals(signal_at,market,rule_version);
        CREATE INDEX IF NOT EXISTS signal_decision ON signals(decision);
        CREATE INDEX IF NOT EXISTS signal_reason ON signals(reason);
        CREATE TABLE IF NOT EXISTS reservations (
          id TEXT PRIMARY KEY REFERENCES signals(id), allocation_pennies INTEGER NOT NULL
          CHECK(allocation_pennies=1000), active INTEGER NOT NULL CHECK(active IN (0,1)),
          state TEXT NOT NULL, plan TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS orders (
          reference TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES reservations(id),
          role TEXT NOT NULL CHECK(role IN ('ENTRY','EXIT')), order_id INTEGER NOT NULL UNIQUE,
          perm_id INTEGER, status TEXT NOT NULL, filled REAL NOT NULL DEFAULT 0,
          remaining REAL NOT NULL DEFAULT 1, deadline TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS order_event ON orders(event_id,role);
        CREATE TABLE IF NOT EXISTS fills (
          exec_id TEXT PRIMARY KEY, reference TEXT NOT NULL REFERENCES orders(reference),
          con_id INTEGER NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL,
          side TEXT NOT NULL, at TEXT NOT NULL, commission REAL, commission_currency TEXT,
          fx REAL, fx_at TEXT, superseded INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS fill_order ON fills(reference,superseded);
        CREATE TABLE IF NOT EXISTS positions (
          con_id INTEGER PRIMARY KEY, quantity REAL NOT NULL, detail TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS lifecycle (
          sequence INTEGER PRIMARY KEY, at TEXT NOT NULL, reference TEXT NOT NULL,
          kind TEXT NOT NULL, detail TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS lifecycle_reference ON lifecycle(reference,sequence);
        CREATE TABLE IF NOT EXISTS bars (
          con_id INTEGER NOT NULL, at TEXT NOT NULL, market TEXT NOT NULL, detail TEXT NOT NULL,
          PRIMARY KEY(con_id,at));
        CREATE INDEX IF NOT EXISTS bar_market ON bars(market,at);
        """)
        if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("LEDGER_INTEGRITY_FAILURE")

    def set_meta(self, key: str, value: Any) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO futures_meta VALUES (?,?)", (key, encode(value))
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM futures_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def audit(self, reference: str, kind: str, detail: Any) -> None:
        self.db.execute(
            "INSERT INTO lifecycle(at,reference,kind,detail) VALUES(?,?,?,?)",
            (stamp(), reference, kind, encode(detail)),
        )

    def observe(self, event: dict[str, Any], reason: str, detail: Any) -> bool:
        with self.db:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO signals VALUES(?,?,?,?,?,?,?,?)",
                (
                    event["id"],
                    event["market"],
                    event["rule_version"],
                    event["signal_at"],
                    event["exit_at"],
                    "SKIPPED" if reason else "SIGNAL_OBSERVED",
                    reason,
                    encode({**event, "inputs": detail, "capacity": self.capacity()}),
                ),
            )
            return cur.rowcount == 1

    def decision(self, identity: str, decision: str, reason: str = "") -> None:
        with self.db:
            self.db.execute(
                "UPDATE signals SET decision=?,reason=? WHERE id=?", (decision, reason, identity)
            )
            self.audit(
                identity,
                "DECISION",
                {"decision": decision, "reason": reason, "capacity": self.capacity()},
            )

    def capacity(self) -> dict[str, int]:
        row = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(allocation_pennies),0) FROM reservations WHERE active=1"
        ).fetchone()
        return {"reserved_open_trades": row[0], "allocation_pennies": row[1]}

    def reserve(self, identity: str, plan: dict[str, Any]) -> str:
        """BEGIN IMMEDIATE serializes admission across tasks/processes; reserve full £10."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.db.execute("SELECT 1 FROM reservations WHERE id=?", (identity,)).fetchone():
                return "DUPLICATE_OPPORTUNITY"
            capacity = self.capacity()
            if plan["quantity"] != 1 or not 0 < plan["cash_pennies"] <= 1000:
                reason = "SKIP_BUDGET_TOO_SMALL"
            elif capacity["reserved_open_trades"] >= 4:
                reason = "SKIP_CAPACITY_FULL"
            elif capacity["allocation_pennies"] + 1000 > 4000:
                reason = "SKIP_ALLOCATION_LIMIT"
            else:
                reason = ""
            self.audit(identity, "ADMISSION", {**capacity, "reason": reason, "plan": plan})
            if reason:
                self.db.execute(
                    "UPDATE signals SET decision='SKIPPED',reason=? WHERE id=?", (reason, identity)
                )
                return reason
            self.db.execute(
                "INSERT INTO reservations VALUES(?,1000,1,'RESERVED',?,?)",
                (identity, encode(plan), stamp()),
            )
            self.db.execute("UPDATE signals SET decision='ORDER_ELIGIBLE' WHERE id=?", (identity,))
        return ""

    def prepare_order(
        self, identity: str, role: str, order_id: int, deadline: str, payload: dict[str, Any]
    ) -> str:
        import hashlib

        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            reservation = self.db.execute(
                "SELECT active FROM reservations WHERE id=?", (identity,)
            ).fetchone()
            if not reservation or not reservation[0]:
                raise ValueError("NO_ACTIVE_RESERVATION")
            previous = list(
                self.db.execute(
                    "SELECT * FROM orders WHERE event_id=? AND role=?", (identity, role)
                )
            )
            if role == "ENTRY" and previous:
                raise ValueError("ENTRY_ALREADY_ATTEMPTED")
            if any(o["status"] not in TERMINAL for o in previous):
                raise ValueError("RECONCILE_BEFORE_RETRY")
            reference = (
                "SLRNOF-"
                + hashlib.sha256(identity.encode()).hexdigest()[:20]
                + f"-{'B' if role == 'ENTRY' else 'S'}{len(previous)}"
            )
            self.db.execute(
                "INSERT INTO orders(reference,event_id,role,order_id,status,deadline,payload) "
                "VALUES(?,?,?,?,'SUBMITTING',?,?)",
                (reference, identity, role, order_id, deadline, encode(payload)),
            )
            self.audit(reference, "DURABLE_SUBMISSION_INTENT", payload)
        return reference

    def active(self) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT r.*,s.market,s.exit_at,s.signal_at "
                "FROM reservations r JOIN signals s USING(id) "
                "WHERE active=1 ORDER BY s.signal_at,s.market"
            )
        ]

    def orders(self, identity: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM orders WHERE event_id=? ORDER BY order_id", (identity,)
            )
        ]

    def fills(self, identity: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT f.*,o.role FROM fills f JOIN orders o USING(reference) "
                "WHERE event_id=? AND superseded=0 ORDER BY at,exec_id",
                (identity,),
            )
        ]

    def exposure(self, identity: str) -> float:
        return float(
            sum(f["quantity"] * (1 if f["side"] == "BOT" else -1) for f in self.fills(identity))
        )

    def record_fill(self, values: dict[str, Any]) -> None:
        with self.db:
            existing = self.db.execute(
                "SELECT * FROM fills WHERE exec_id=?", (values["exec_id"],)
            ).fetchone()
            if existing:
                for key in ("reference", "con_id", "quantity", "price", "side", "at"):
                    if existing[key] != values[key]:
                        raise ValueError("EXECUTION_ID_REUSED_WITH_DIFFERENT_EVIDENCE")
                return
            self.db.execute(
                "INSERT OR IGNORE INTO fills"
                "(exec_id,reference,con_id,quantity,price,side,at,fx,fx_at) "
                "VALUES(:exec_id,:reference,:con_id,:quantity,:price,:side,:at,:fx,:fx_at)",
                values,
            )
            base, sep, suffix = values["exec_id"].rpartition(".")
            if sep and suffix.isdigit():
                rows = list(
                    self.db.execute(
                        "SELECT exec_id,reference,con_id FROM fills WHERE exec_id LIKE ?",
                        (base + ".%",),
                    )
                )
                revisions = [
                    (int(r["exec_id"].rpartition(".")[2]), r)
                    for r in rows
                    if r["exec_id"].rpartition(".")[2].isdigit()
                ]
                latest = max(n for n, _ in revisions)
                for number, row in revisions:
                    if row["reference"] != values["reference"] or row["con_id"] != values["con_id"]:
                        raise ValueError("EXECUTION_CORRECTION_OWNERSHIP_MISMATCH")
                    self.db.execute(
                        "UPDATE fills SET superseded=? WHERE exec_id=?",
                        (int(number < latest), row["exec_id"]),
                    )
            self.audit(values["reference"], "BROKER_FILL", values)
            event = self.db.execute(
                "SELECT event_id FROM orders WHERE reference=?", (values["reference"],)
            ).fetchone()[0]
            # A late fill reopens its obligation even if a prior snapshot looked flat.
            self.db.execute(
                "UPDATE reservations SET active=1,state='EXPOSURE_REQUIRES_RECONCILIATION' "
                "WHERE id=?",
                (event,),
            )
            self.db.execute("UPDATE signals SET decision='BROKER_PAPER_FILL' WHERE id=?", (event,))

    def confirm_closed(self, identity: str, position: float) -> bool:
        orders = self.orders(identity)
        fills = self.fills(identity)
        if not orders or any(o["status"] not in TERMINAL for o in orders):
            return False
        for order in orders:
            actual = sum(f["quantity"] for f in fills if f["reference"] == order["reference"])
            if actual != order["filled"]:
                return False
        if self.exposure(identity) != 0 or position != 0:
            return False
        with self.db:
            self.db.execute(
                "UPDATE reservations SET active=0,state=? WHERE id=?",
                ("CLOSED" if fills else "UNFILLED_CONFIRMED", identity),
            )
            self.db.execute(
                "UPDATE signals SET decision=? WHERE id=?",
                ("CLOSED_PAPER_TRADE" if fills else "UNFILLED_CONFIRMED", identity),
            )
            self.audit(identity, "CLOSURE_CONFIRMED", {"position": position})
        return True

    def history(
        self, market: str | None, day: str | None, version: str | None, offset: int = 0
    ) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT s.*,r.state,r.plan FROM signals s "
                "LEFT JOIN reservations r USING(id) WHERE (? IS NULL OR market=?) "
                "AND (? IS NULL OR substr(signal_at,1,10)=?) AND (? IS NULL OR rule_version=?) "
                "ORDER BY signal_at DESC,market LIMIT 100 OFFSET ?",
                (market, market, day, day, version, version, offset),
            )
        ]

    def economics(self) -> dict[str, Any]:
        net, closed, wins, provisional = 0.0, 0, 0, 0
        # One indexed join, not one fills query per historical trade on every UI refresh.
        rows = self.db.execute("""
            SELECT r.id,COUNT(f.exec_id) AS executions,
              SUM(CASE WHEN f.exec_id IS NOT NULL AND (f.commission IS NULL OR f.fx IS NULL
                OR f.commission_currency IS NULL
                OR f.commission_currency NOT IN (json_extract(r.plan,'$.currency'),'GBP'))
                THEN 1 ELSE 0 END) AS incomplete,
              SUM((CASE WHEN f.side='SLD' THEN 1 ELSE -1 END)*f.quantity*f.price
                *json_extract(r.plan,'$.multiplier')*json_extract(r.plan,'$.price_unit_factor')
                *f.fx-f.commission*(CASE WHEN f.commission_currency='GBP' THEN 1 ELSE f.fx END)
              ) AS pnl
            FROM reservations r LEFT JOIN orders o ON o.event_id=r.id
            LEFT JOIN fills f ON f.reference=o.reference AND f.superseded=0
            WHERE r.state='CLOSED' GROUP BY r.id
        """)
        for row in rows:
            if not row["executions"] or row["incomplete"]:
                provisional += 1
                continue
            pnl = row["pnl"]
            net += pnl
            closed += 1
            wins += pnl > 0
        counts = {
            r[0]: r[1]
            for r in self.db.execute("SELECT decision,COUNT(*) FROM signals GROUP BY decision")
        }
        reasons = {
            r[0]: r[1]
            for r in self.db.execute("SELECT reason,COUNT(*) FROM signals GROUP BY reason")
        }
        return {
            "realised_net_gbp": net if closed or not provisional else None,
            "closed_with_complete_costs": closed,
            "wins": wins,
            "win_rate": wins / closed if closed else None,
            "provisional_closed": provisional,
            "opportunities": sum(counts.values()),
            "decisions": counts,
            "skip_reasons": reasons,
            "eligible_trades": self.db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0],
            "fills": self.db.execute("SELECT COUNT(*) FROM fills WHERE superseded=0").fetchone()[0],
            "basis": "BROKER_SIMULATED_PAPER_EXECUTIONS",
        }
