"""The paper record: one SQLite file of signals, paper trades, event states and saved boards."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocker_execution.config import PAGE_SIZE

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  day TEXT NOT NULL, name TEXT NOT NULL, due TEXT NOT NULL, effective TEXT,
  status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', updated TEXT NOT NULL,
  PRIMARY KEY (day, name));
CREATE TABLE IF NOT EXISTS signals (
  day TEXT NOT NULL, method TEXT NOT NULL, market TEXT NOT NULL,
  side INTEGER NOT NULL CHECK (side IN (-1, 0, 1)), reason TEXT NOT NULL,
  detail TEXT NOT NULL, created TEXT NOT NULL,
  PRIMARY KEY (day, method, market));
CREATE TABLE IF NOT EXISTS cfd_trades (
  id INTEGER PRIMARY KEY, day TEXT NOT NULL, market TEXT NOT NULL, uic INTEGER NOT NULL,
  side INTEGER NOT NULL CHECK (side IN (-1, 1)), amount REAL NOT NULL CHECK (amount > 0),
  factor REAL NOT NULL, stop REAL NOT NULL,
  entry_at TEXT NOT NULL, entry_price REAL NOT NULL, entry_bid REAL NOT NULL,
  entry_ask REAL NOT NULL,
  exit_at TEXT, exit_price REAL, exit_bid REAL, exit_ask REAL, exit_reason TEXT, pnl_usd REAL,
  UNIQUE (day, market));
CREATE TABLE IF NOT EXISTS option_trades (
  id INTEGER PRIMARY KEY, day TEXT NOT NULL, method TEXT NOT NULL,
  side INTEGER NOT NULL CHECK (side IN (-1, 1)), uic INTEGER NOT NULL, expiry TEXT NOT NULL,
  put_call TEXT NOT NULL, strike REAL NOT NULL, exit_day TEXT NOT NULL,
  entry_for TEXT NOT NULL, entry_read TEXT NOT NULL, entry_delay INTEGER,
  entry_bid REAL NOT NULL, entry_ask REAL NOT NULL, entry_delta REAL, entry_vol REAL,
  entry_underlying REAL, cost_buy REAL,
  exit_for TEXT, exit_read TEXT, exit_delay INTEGER, exit_bid REAL, exit_ask REAL,
  exit_delta REAL, exit_underlying REAL, cost_sell REAL, exit_reason TEXT, pnl_usd REAL,
  late_for TEXT, late_bid REAL, late_ask REAL, late_underlying REAL,
  UNIQUE (day, method));
CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY, day TEXT NOT NULL, event TEXT NOT NULL, taken TEXT NOT NULL,
  payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS snapshot_day ON snapshots (day, event);
"""


def stamp(at: datetime | None = None) -> str:
    return (at or datetime.now(UTC)).astimezone(UTC).isoformat()


def encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, default=str)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=5)
        self.db.row_factory = sqlite3.Row
        tables = {
            x[0] for x in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if tables and "paper_meta" not in tables:
            # Never write paper rows into the retired futures ledger or any other database.
            self.db.close()
            raise ValueError("PAPER_RECORD_DATABASE_REQUIRED")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)
        if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("PAPER_RECORD_INTEGRITY_FAILURE")

    def bind(self, environment: str) -> None:
        """Rows from one Saxo environment never share a file with another's."""
        expected = encode({"record": "PAPER", "data_environment": environment})
        row = self.db.execute("SELECT value FROM paper_meta WHERE key='provenance'").fetchone()
        if row is None:
            with self.db:
                self.db.execute("INSERT INTO paper_meta VALUES ('provenance', ?)", (expected,))
        elif row[0] != expected:
            raise ValueError("PAPER_RECORD_PROVENANCE_MISMATCH")

    # events ---------------------------------------------------------------------------------
    def event(self, day: str, name: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.db.execute(
            "SELECT * FROM events WHERE day=? AND name=?", (day, name)
        ).fetchone()
        return row

    def set_event(
        self,
        day: str,
        name: str,
        due: datetime,
        status: str,
        *,
        effective: datetime | None = None,
        note: str = "",
    ) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?) ON CONFLICT (day, name) DO UPDATE SET "
                "due=excluded.due, effective=excluded.effective, status=excluded.status, "
                "note=excluded.note, updated=excluded.updated",
                (day, name, stamp(due), effective and stamp(effective), status, note, stamp()),
            )

    # signals and trades -----------------------------------------------------------------------
    def add_signal(
        self, day: str, method: str, market: str, side: int, reason: str, detail: dict[str, Any]
    ) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO signals VALUES (?,?,?,?,?,?,?)",
                (day, method, market, side, reason, encode(detail), stamp()),
            )

    def signal(self, day: str, method: str, market: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.db.execute(
            "SELECT * FROM signals WHERE day=? AND method=? AND market=?", (day, method, market)
        ).fetchone()
        return row

    def insert(self, table: str, values: dict[str, Any]) -> int:
        assert table in {"cfd_trades", "option_trades"}
        columns = ",".join(values)
        with self.db:
            cursor = self.db.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({','.join('?' * len(values))})",
                tuple(values.values()),
            )
        return int(cursor.lastrowid or 0)

    def update(self, table: str, row_id: int, values: dict[str, Any]) -> None:
        assert table in {"cfd_trades", "option_trades"}
        assignments = ",".join(f"{k}=?" for k in values)
        with self.db:
            self.db.execute(
                f"UPDATE {table} SET {assignments} WHERE id=?", (*values.values(), row_id)
            )

    def cfd_trade(self, day: str, market: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.db.execute(
            "SELECT * FROM cfd_trades WHERE day=? AND market=?", (day, market)
        ).fetchone()
        return row

    def open_cfd(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM cfd_trades WHERE exit_at IS NULL ORDER BY id"))

    def open_options(self) -> list[sqlite3.Row]:
        return list(
            self.db.execute("SELECT * FROM option_trades WHERE exit_read IS NULL ORDER BY id")
        )

    def option(self, day: str, method: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.db.execute(
            "SELECT * FROM option_trades WHERE day=? AND method=?", (day, method)
        ).fetchone()
        return row

    def snapshot(self, day: str, event: str, payload: dict[str, Any]) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO snapshots (day, event, taken, payload) VALUES (?,?,?,?)",
                (day, event, stamp(), encode(payload)),
            )

    # dashboard reads ------------------------------------------------------------------------
    def day_view(self, day: str) -> dict[str, Any]:
        def rows(sql: str) -> list[dict[str, Any]]:
            return [dict(r) for r in self.db.execute(sql, {"d": day})]

        signals = rows("SELECT * FROM signals WHERE day=:d ORDER BY method, market")
        for s in signals:
            s["detail"] = json.loads(s["detail"])
        return {
            "events": rows("SELECT * FROM events WHERE day=:d ORDER BY due, name"),
            "signals": signals,
            "cfd": rows("SELECT * FROM cfd_trades WHERE day=:d ORDER BY id"),
            # Today's option trades and any earlier one that closes today (Connors).
            "options": rows("SELECT * FROM option_trades WHERE day=:d OR exit_day=:d ORDER BY id"),
        }

    def record(self, offset: int) -> dict[str, Any]:
        trades = [
            dict(r)
            for r in self.db.execute(
                """SELECT 'CFD ' || market AS method, day, side, entry_at AS opened,
                          exit_at AS closed, exit_reason, pnl_usd FROM cfd_trades
                   UNION ALL
                   SELECT method, day, side, entry_for, exit_for, exit_reason, pnl_usd
                     FROM option_trades
                   ORDER BY day DESC, method LIMIT ? OFFSET ?""",
                (PAGE_SIZE, offset),
            )
        ]
        summary = [
            dict(r)
            for r in self.db.execute(
                """SELECT method, COUNT(*) AS trades, COUNT(pnl_usd) AS closed,
                          SUM(pnl_usd > 0) AS wins, ROUND(SUM(pnl_usd), 2) AS total_usd,
                          ROUND(AVG(pnl_usd), 2) AS mean_usd, MIN(day) AS first_day
                     FROM (SELECT 'CFD ' || market AS method, day, pnl_usd FROM cfd_trades
                           UNION ALL SELECT method, day, pnl_usd FROM option_trades)
                    GROUP BY method ORDER BY method"""
            )
        ]
        return {"trades": trades, "summary": summary, "has_more": len(trades) == PAGE_SIZE}
