"""The paper record: one SQLite file of signals, paper trades, event states and saved boards."""

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from stocker_execution import methods
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
  worst REAL,
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
CREATE TABLE IF NOT EXISTS cross_trades (
  id INTEGER PRIMARY KEY, day TEXT NOT NULL, signal_candle TEXT NOT NULL, market TEXT NOT NULL,
  uic INTEGER NOT NULL, qqq_close REAL NOT NULL, qqq_vwap REAL NOT NULL, qqq_ema REAL NOT NULL,
  amount REAL, factor REAL, entry_at TEXT, entry_price REAL, entry_bid REAL, entry_ask REAL,
  exit_candle TEXT, qqq_exit_close REAL,
  exit_at TEXT, exit_price REAL, exit_bid REAL, exit_ask REAL, exit_reason TEXT, pnl_usd REAL,
  note TEXT NOT NULL DEFAULT '',
  UNIQUE (day, signal_candle));
CREATE TABLE IF NOT EXISTS dip_trades (
  id INTEGER PRIMARY KEY, day TEXT NOT NULL, rule TEXT NOT NULL, market TEXT NOT NULL,
  uic INTEGER NOT NULL, book INTEGER NOT NULL CHECK (book IN (0, 1)), qqq_price REAL NOT NULL,
  amount REAL NOT NULL CHECK (amount > 0), factor REAL NOT NULL,
  entry_at TEXT NOT NULL, entry_price REAL NOT NULL, entry_bid REAL NOT NULL,
  entry_ask REAL NOT NULL,
  exit_day TEXT, exit_at TEXT, exit_price REAL, exit_bid REAL, exit_ask REAL, exit_reason TEXT,
  qqq_exit_price REAL, pnl_usd REAL,
  UNIQUE (day, rule));
CREATE TABLE IF NOT EXISTS stock_bars (
  code TEXT NOT NULL, day TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
  low REAL NOT NULL, close REAL NOT NULL, adj REAL NOT NULL, volume REAL NOT NULL,
  PRIMARY KEY (code, day));
CREATE TABLE IF NOT EXISTS stock_orders (
  id INTEGER PRIMARY KEY, signal_day TEXT NOT NULL, code TEXT NOT NULL, close REAL NOT NULL,
  limit_price REAL NOT NULL, sell_off REAL NOT NULL, closing_range REAL NOT NULL,
  adx REAL NOT NULL, crsi REAL NOT NULL, rank INTEGER NOT NULL,
  tier TEXT NOT NULL CHECK (tier IN ('A', 'B')), status TEXT NOT NULL,
  fill_day TEXT, fill_price REAL, note TEXT NOT NULL DEFAULT '',
  UNIQUE (signal_day, code));
CREATE TABLE IF NOT EXISTS stock_trades (
  id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL UNIQUE, code TEXT NOT NULL,
  tier TEXT NOT NULL, signal_day TEXT NOT NULL, entry_day TEXT NOT NULL,
  entry_price REAL NOT NULL, notional_usd REAL NOT NULL,
  exit_day TEXT, exit_price REAL, exit_reason TEXT, crsi_exit REAL, pnl_usd REAL);
CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY, day TEXT NOT NULL, event TEXT NOT NULL, taken TEXT NOT NULL,
  payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS snapshot_day ON snapshots (day, event);
"""


TRADE_TABLES = {
    "cfd_trades",
    "option_trades",
    "cross_trades",
    "dip_trades",
    "stock_orders",
    "stock_trades",
}


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
        # Amendment 4 addendum: the worst adverse price seen while a CFD trade was open.
        if "worst" not in {r[1] for r in self.db.execute("PRAGMA table_info(cfd_trades)")}:
            with self.db:
                self.db.execute("ALTER TABLE cfd_trades ADD COLUMN worst REAL")
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
        assert table in TRADE_TABLES
        columns = ",".join(values)
        with self.db:
            cursor = self.db.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({','.join('?' * len(values))})",
                tuple(values.values()),
            )
        return int(cursor.lastrowid or 0)

    def update(self, table: str, row_id: int, values: dict[str, Any]) -> None:
        assert table in TRADE_TABLES
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

    def cross_trade(self, day: str, signal_candle: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.db.execute(
            "SELECT * FROM cross_trades WHERE day=? AND signal_candle=?", (day, signal_candle)
        ).fetchone()
        return row

    def open_crosses(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM cross_trades WHERE exit_at IS NULL ORDER BY id"))

    def open_dips(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM dip_trades WHERE exit_at IS NULL ORDER BY id"))

    def dip(self, day: str, rule: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.db.execute(
            "SELECT * FROM dip_trades WHERE day=? AND rule=?", (day, rule)
        ).fetchone()
        return row

    def open_options(self) -> list[sqlite3.Row]:
        return list(
            self.db.execute("SELECT * FROM option_trades WHERE exit_read IS NULL ORDER BY id")
        )

    def option(self, day: str, method: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.db.execute(
            "SELECT * FROM option_trades WHERE day=? AND method=?", (day, method)
        ).fetchone()
        return row

    def pending_events(self, name: str) -> list[sqlite3.Row]:
        """Events of `name` still being retried, oldest first."""
        return list(
            self.db.execute(
                "SELECT * FROM events WHERE name=? AND status IN ('RETRYING', 'WAITING') "
                "ORDER BY day",
                (name,),
            )
        )

    # stock dips (amendment 5) -------------------------------------------------------------
    def put_stock_bar(self, code: str, b: methods.StockBar) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO stock_bars VALUES (?,?,?,?,?,?,?,?)",
                (code, b.day, b.open, b.high, b.low, b.close, b.adj, b.volume),
            )

    def replace_stock_bars(self, code: str, bars: list[methods.StockBar]) -> None:
        with self.db:
            self.db.execute("DELETE FROM stock_bars WHERE code=?", (code,))
            self.db.executemany(
                "INSERT OR REPLACE INTO stock_bars VALUES (?,?,?,?,?,?,?,?)",
                [(code, b.day, b.open, b.high, b.low, b.close, b.adj, b.volume) for b in bars],
            )

    def prune_stock_bars(self, before: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM stock_bars WHERE day<?", (before,))

    def stock_bar_count(self, code: str) -> int:
        return int(
            self.db.execute("SELECT COUNT(*) FROM stock_bars WHERE code=?", (code,)).fetchone()[0]
        )

    def latest_stock_bar(self, code: str) -> methods.StockBar | None:
        row = self.db.execute(
            "SELECT * FROM stock_bars WHERE code=? ORDER BY day DESC LIMIT 1", (code,)
        ).fetchone()
        return None if row is None else methods.StockBar(*row[1:])

    def stock_bars(self, code: str) -> list[methods.StockBar]:
        return [
            methods.StockBar(*r[1:])
            for r in self.db.execute("SELECT * FROM stock_bars WHERE code=? ORDER BY day", (code,))
        ]

    def all_stock_bars(self) -> dict[str, list[methods.StockBar]]:
        out: dict[str, list[methods.StockBar]] = {}
        for r in self.db.execute("SELECT * FROM stock_bars ORDER BY code, day"):
            out.setdefault(r[0], []).append(methods.StockBar(*r[1:]))
        return out

    def sessions_after(self, day: str) -> int:
        """Sessions since `day`, counted on SPY's bars."""
        return int(
            self.db.execute(
                "SELECT COUNT(*) FROM stock_bars WHERE code='SPY' AND day>?", (day,)
            ).fetchone()[0]
        )

    def pending_stock_orders(self) -> list[sqlite3.Row]:
        return list(
            self.db.execute("SELECT * FROM stock_orders WHERE status='PENDING' ORDER BY id")
        )

    def open_stock_trades(self) -> list[sqlite3.Row]:
        return list(
            self.db.execute("SELECT * FROM stock_trades WHERE exit_day IS NULL ORDER BY id")
        )

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
            "crosses": rows("SELECT * FROM cross_trades WHERE day=:d ORDER BY signal_candle"),
            # Dip trades bought today, closed today or still held from an earlier day.
            "dips": rows(
                "SELECT * FROM dip_trades WHERE day=:d OR exit_day=:d OR exit_at IS NULL "
                "ORDER BY id"
            ),
            # Stock orders placed tonight, still resting, or settled today; positions open or
            # opened or closed today.
            "stock_orders": rows(
                "SELECT * FROM stock_orders WHERE signal_day=:d OR status='PENDING' "
                "OR fill_day=:d ORDER BY crsi, code"
            ),
            "stock_trades": rows(
                "SELECT * FROM stock_trades WHERE exit_day IS NULL OR entry_day=:d OR exit_day=:d "
                "ORDER BY id"
            ),
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
                   UNION ALL
                   SELECT 'CROSS ' || market, day, 1, entry_at, exit_at, exit_reason, pnl_usd
                     FROM cross_trades
                   UNION ALL
                   SELECT 'DIP ' || rule, day, 1, entry_at, exit_at, exit_reason, pnl_usd
                     FROM dip_trades
                   UNION ALL
                   SELECT 'STOCK DIP ' || code, entry_day, 1, entry_day, exit_day, exit_reason,
                          pnl_usd
                     FROM stock_trades
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
                           UNION ALL SELECT method, day, pnl_usd FROM option_trades
                           UNION ALL SELECT 'CROSS ' || market, day, pnl_usd FROM cross_trades
                           UNION ALL SELECT 'DIP ' || rule, day, pnl_usd FROM dip_trades
                           UNION ALL SELECT 'STOCK DIP', entry_day, pnl_usd FROM stock_trades
                           -- the one-at-a-time book: the dip trades it would have held
                           UNION ALL SELECT 'DIP BOOK', day, pnl_usd FROM dip_trades WHERE book=1)
                    GROUP BY method ORDER BY method"""
            )
        ]
        # The equity curve: every closed result booked on the day it closed.
        by_day = [
            dict(r)
            for r in self.db.execute(
                """SELECT day, ROUND(SUM(pnl_usd), 2) AS pnl_usd, COUNT(*) AS closed
                     FROM (SELECT day, pnl_usd FROM cfd_trades
                           UNION ALL SELECT exit_day, pnl_usd FROM option_trades
                           UNION ALL SELECT day, pnl_usd FROM cross_trades
                           UNION ALL SELECT exit_day, pnl_usd FROM dip_trades
                           UNION ALL SELECT exit_day, pnl_usd FROM stock_trades)
                    WHERE pnl_usd IS NOT NULL GROUP BY day ORDER BY day"""
            )
        ]
        return {
            "trades": trades,
            "summary": summary,
            "books": self.books(),
            "by_day": by_day,
            "has_more": len(trades) == PAGE_SIZE,
        }

    def books(self) -> list[dict[str, Any]]:
        """Amendment 4: the frozen subsets of the first-candle trades (methods.BOOKS) and of the
        VWAP crossings (methods.CROSS_BOOKS), summed by instrument. A day recorded without its
        context belongs only to the books that need none (weekday, agreement)."""
        signals = {
            (r["day"], r["market"]): (r["side"], json.loads(r["detail"]))
            for r in self.db.execute(
                "SELECT day, market, side, detail FROM signals WHERE method='FIRST_CANDLE'"
            )
        }

        def day_of(day: str, market: str) -> methods.Day | None:
            own = signals.get((day, market))
            if own is None:
                return None
            other = signals.get((day, "US500" if market == "NAS100" else "NAS100"))
            agree = None if other is None or not other[0] or not own[0] else other[0] == own[0]
            return methods.Day(own[0], date.fromisoformat(day).weekday(), agree, own[1])

        rows: list[tuple[str, methods.Day | None, float | None, bool]] = []
        what_if: list[float] = []  # the Nasdaq CFD trades re-run with the what-if stop
        for r in self.db.execute(
            "SELECT day, market, side, entry_price, worst, pnl_usd FROM cfd_trades"
        ):
            rows.append((r["market"], day_of(r["day"], r["market"]), r["pnl_usd"], False))
            if r["market"] == methods.SIGNAL_MARKET:
                live = next(c.stop_bp for c in methods.CFDS if c.market == r["market"])
                v = methods.what_if_stop(
                    r["side"],
                    r["entry_price"],
                    r["worst"],
                    r["pnl_usd"],
                    methods.WHAT_IF_STOP_BP,
                    live,
                )
                if v is not None:
                    what_if.append(v)
        for r in self.db.execute(
            "SELECT day, pnl_usd FROM option_trades WHERE method='FIRST_CANDLE_QQQ'"
        ):
            rows.append(("QQQ option", day_of(r["day"], "NAS100"), r["pnl_usd"], False))
        for r in self.db.execute(
            "SELECT day, signal_candle, pnl_usd FROM cross_trades WHERE entry_at IS NOT NULL"
        ):
            closes = datetime.fromisoformat(r["signal_candle"]) + timedelta(minutes=5)
            early = closes.astimezone(methods.NY).hour < methods.CROSS_LATE_HOUR
            rows.append(("CROSS", day_of(r["day"], "NAS100"), r["pnl_usd"], early))
        out = []
        for book in methods.BOOKS + methods.SIZED_BOOKS + methods.CROSS_BOOKS:
            cells: dict[str, dict[str, Any]] = {}
            for instrument, d, pnl, early in rows:
                if d is None or (instrument == "CROSS") != (book in methods.CROSS_BOOKS):
                    continue
                size = 1
                if book == "CROSS PULLBACK":
                    member = d.known and d.pullback
                elif book == "CROSS BEFORE 14:00":
                    member = early
                elif book == "SIZED V2":
                    size = methods.size_v2(d, instrument)
                    member = size > 0
                elif book == "SIZED V3":
                    size = methods.size_v3(d) if instrument != "QQQ option" or d.pullback else 0
                    member = size > 0
                else:
                    member = methods.in_book(book, d, instrument)
                if not member:
                    continue
                c = cells.setdefault(instrument, {"trades": 0, "closed": 0, "total_usd": 0.0})
                c["trades"] += 1
                if pnl is not None:
                    c["closed"] += 1
                    c["total_usd"] = round(c["total_usd"] + pnl * size, 2)
            for c in cells.values():
                c["mean_usd"] = round(c["total_usd"] / c["closed"], 2) if c["closed"] else None
            out.append({"book": book, "cells": cells})
        if what_if:
            total = round(sum(what_if), 2)
            out.append(
                {
                    "book": "STOP 75 BP",
                    "cells": {
                        methods.SIGNAL_MARKET: {
                            "trades": len(what_if),
                            "closed": len(what_if),
                            "total_usd": total,
                            "mean_usd": round(total / len(what_if), 2),
                        }
                    },
                }
            )
        return out
