"""Small durable FIRST4 ledger; a selected slot is never recycled."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from stocker_execution.first4 import eligible, times


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS first4_sessions (
            session TEXT PRIMARY KEY, last_clock TEXT, blocked TEXT);
        CREATE TABLE IF NOT EXISTS first4_events (
            session TEXT, symbol TEXT, con_id INTEGER, information_at TEXT, rank INTEGER,
            decision TEXT, prior15 REAL, slot INTEGER, entry_at TEXT, expiry_at TEXT,
            close_at TEXT, outcome TEXT, detail TEXT, PRIMARY KEY(session,symbol),
            UNIQUE(session,slot));
        CREATE TABLE IF NOT EXISTS first4_orders (
            reference TEXT PRIMARY KEY, session TEXT, symbol TEXT, role TEXT,
            order_id INTEGER, perm_id INTEGER, status TEXT, payload TEXT);
        CREATE TABLE IF NOT EXISTS first4_fills (
            exec_id TEXT PRIMARY KEY, reference TEXT, con_id INTEGER, quantity REAL,
            price REAL, side TEXT, multiplier REAL, time TEXT, commission REAL);
        CREATE TABLE IF NOT EXISTS first4_positions (
            con_id INTEGER PRIMARY KEY, quantity REAL, payload TEXT);
        CREATE TABLE IF NOT EXISTS first4_meta (key TEXT PRIMARY KEY, value TEXT);
        """)

    def rows(self, table: str) -> list[dict[str, Any]]:
        if table not in {"sessions", "events", "orders", "fills", "positions", "meta"}:
            raise ValueError("Unknown ledger table")
        return [dict(r) for r in self.db.execute(f"SELECT * FROM first4_{table}")]

    def set_meta(self, key: str, value: Any) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO first4_meta VALUES (?,?)", (key, json.dumps(value))
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM first4_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def block(self, session: str, reason: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO first4_sessions VALUES (?,NULL,NULL)", (session,)
            )
            self.db.execute(
                "UPDATE first4_sessions SET blocked=? WHERE session=?", (reason, session)
            )

    def observe(
        self, session: str, clock: datetime, close: datetime, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Commit an entire scanner minute in native rank order, before execution work."""
        selected = []
        ordered = sorted(candidates, key=lambda c: c["rank"])
        ranks = [c["rank"] for c in ordered]
        if len(ranks) != len(set(ranks)) or any(r < 1 or r > 25 for r in ranks):
            raise ValueError("Ambiguous scanner rank")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO first4_sessions VALUES (?,NULL,NULL)", (session,)
            )
            state = self.db.execute(
                "SELECT * FROM first4_sessions WHERE session=?", (session,)
            ).fetchone()
            if state["blocked"]:
                raise ValueError(state["blocked"])
            if state["last_clock"] and clock.isoformat() <= state["last_clock"]:
                raise ValueError("OUT_OF_ORDER_SCANNER_MINUTE")
            used = self.db.execute(
                "SELECT count(*) FROM first4_events WHERE session=? AND slot IS NOT NULL",
                (session,),
            ).fetchone()[0]
            for c in ordered:
                if self.db.execute(
                    "SELECT 1 FROM first4_events WHERE session=? AND symbol=?",
                    (session, c["symbol"]),
                ).fetchone():
                    continue
                reason = eligible(c["price"], c["change_pct"], c["prior15"])
                slot = None
                if reason == "Q5":
                    if used < 4:
                        used += 1
                        slot, reason = used, "SELECTED"
                    else:
                        reason = "DAILY_CAP"
                entry, expiry, exit_at = times(clock, close)
                outcome = "SELECTED" if slot else "REJECTED"
                self.db.execute(
                    "INSERT INTO first4_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        session,
                        c["symbol"],
                        c["con_id"],
                        clock.isoformat(),
                        c["rank"],
                        reason,
                        c["prior15"],
                        slot,
                        entry.isoformat(),
                        expiry.isoformat(),
                        exit_at.isoformat(),
                        outcome,
                        json.dumps(c.get("detail", {})),
                    ),
                )
                if slot:
                    selected.append(
                        dict(
                            self.db.execute(
                                "SELECT * FROM first4_events WHERE session=? AND symbol=?",
                                (session, c["symbol"]),
                            ).fetchone()
                        )
                    )
            self.db.execute(
                "UPDATE first4_sessions SET last_clock=? WHERE session=?",
                (clock.isoformat(), session),
            )
        return selected

    def outcome(self, event: dict[str, Any], outcome: str, detail: Any = None) -> None:
        with self.db:
            self.db.execute(
                "UPDATE first4_events SET outcome=?,detail=? WHERE session=? AND symbol=?",
                (
                    outcome,
                    json.dumps(detail),
                    event["session"],
                    event["symbol"],
                ),
            )

    def reserve_order(
        self,
        event: dict[str, Any],
        role: str,
        order_id: int,
        payload: dict[str, Any],
        suffix: str = "",
    ) -> str:
        reference = f"F4:{event['session']}:{event['slot']}:{role}{suffix}"
        with self.db:
            self.db.execute(
                "INSERT INTO first4_orders VALUES (?,?,?,?,?,NULL,'RESERVED',?)",
                (
                    reference,
                    event["session"],
                    event["symbol"],
                    role,
                    order_id,
                    json.dumps(payload),
                ),
            )
        return reference
