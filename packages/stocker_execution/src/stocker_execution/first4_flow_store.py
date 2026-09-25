"""One bounded writer thread, append-only JSONL evidence and a separate SQLite index.

The execution ledger is never opened here. Raw batches are fsynced before summary
commits. An interrupted tail is disclosed by replay; existing evidence is never
rewritten, pruned or silently repaired.
"""

import json
import os
import queue
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stocker_execution.first4_config import OrderFlowConfig
from stocker_execution.first4_flow import VERSION, FlowEvent, FlowReducer


@dataclass(frozen=True)
class CaptureStart:
    capture_id: str
    metadata: str


class FlowWriter:
    def __init__(self, config: OrderFlowConfig):
        self.config = config
        self.queue: queue.Queue[CaptureStart | FlowEvent] = queue.Queue(config.queue_events)
        self.stop_requested = threading.Event()
        self.ready = threading.Event()
        self.error = ""
        self.dropped = 0
        self.uncommitted_events = 0
        self.high_water = 0
        self.max_queue_latency_ms = 0.0
        self.thread = threading.Thread(target=self.run, name="first4-flow-writer", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def put(self, item: CaptureStart | FlowEvent) -> bool:
        if self.error or self.stop_requested.is_set():
            self.dropped += 1
            return False
        try:
            self.queue.put_nowait(item)
            self.high_water = max(self.high_water, self.queue.qsize())
            return True
        except queue.Full:
            self.dropped += 1
            self.error = "QUEUE_OVERFLOW_CAPTURE_STOPPED"
            return False

    def close(self) -> None:
        self.stop_requested.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            self.error = "WRITER_SHUTDOWN_TIMEOUT"

    def run(self) -> None:
        db: sqlite3.Connection | None = None
        active: dict[str, tuple[dict[str, Any], FlowReducer]] = {}
        raw_bytes = 0
        try:
            root = self.config.raw_path
            root.mkdir(parents=True, exist_ok=True)
            # Once at startup, off the event loop. No scans during refresh.
            for index, path in enumerate(root.iterdir()):
                if index >= 10000:
                    raise OSError("CAPTURE_DIRECTORY_FILE_LIMIT")
                if path.is_file() and path.name != "summaries.sqlite3":
                    raw_bytes += path.stat().st_size
            db = sqlite3.connect(root / "summaries.sqlite3", timeout=0.5)
            db.executescript("""
                PRAGMA journal_mode=DELETE;
                CREATE TABLE IF NOT EXISTS captures (
                    capture_id TEXT PRIMARY KEY, session TEXT, con_id INTEGER,
                    requested_at TEXT, payload TEXT);
                CREATE INDEX IF NOT EXISTS flow_identity ON captures(session,con_id,requested_at);
                CREATE TABLE IF NOT EXISTS minutes (
                    capture_id TEXT, minute TEXT, payload TEXT,
                    PRIMARY KEY(capture_id,minute));
            """)
            self.ready.set()
            while not self.stop_requested.is_set() or not self.queue.empty():
                batch: list[CaptureStart | FlowEvent] = []
                try:
                    batch.append(self.queue.get(timeout=self.config.flush_seconds))
                except queue.Empty:
                    if self.error:
                        break
                    continue
                deadline = time.monotonic() + self.config.flush_seconds
                while len(batch) < self.config.batch_events:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(self.queue.get(timeout=remaining))
                    except queue.Empty:
                        break
                lines: dict[str, list[str]] = {}
                touched: set[str] = set()
                changed: set[tuple[str, str]] = set()
                ended: set[str] = set()
                for item in batch:
                    cid = item.capture_id
                    touched.add(cid)
                    if isinstance(item, CaptureStart):
                        if len(set(active) - ended) >= 4:
                            raise OSError("ACTIVE_CAPTURE_LIMIT")
                        metadata = json.loads(item.metadata)
                        active[cid] = (metadata, FlowReducer(self.config.quote_age_ms))
                        record = {"capture": metadata}
                    else:
                        self.max_queue_latency_ms = max(
                            self.max_queue_latency_ms,
                            (time.monotonic_ns() - item.monotonic_ns) / 1e6,
                        )
                        metadata, reducer = active[cid]
                        derived = reducer.apply(item)
                        record = {"event": item.record(), "classification": derived}
                        minute = (
                            datetime.fromisoformat(item.received_at)
                            .replace(second=0, microsecond=0)
                            .isoformat()
                        )
                        if minute in reducer.bars:
                            changed.add((cid, minute))
                        if item.kind == "end":
                            metadata.update(ended_at=item.received_at, end_reason=item.reason)
                            ended.add(cid)
                    line = json.dumps(record, separators=(",", ":"), allow_nan=True) + "\n"
                    if len(line.encode()) > 16384:
                        raise OSError("RAW_EVENT_SIZE_LIMIT")
                    lines.setdefault(cid, []).append(line)
                encoded = {cid: "".join(rows).encode() for cid, rows in lines.items()}
                size = sum(len(data) for data in encoded.values())
                index_size = (root / "summaries.sqlite3").stat().st_size
                if raw_bytes + index_size + size + 1_000_000 > self.config.max_storage_bytes:
                    raise OSError("STORAGE_LIMIT_CAPTURE_STOPPED")
                if shutil.disk_usage(root).free - size < self.config.min_free_bytes:
                    raise OSError("FREE_DISK_RESERVE_CAPTURE_STOPPED")
                for cid, data in encoded.items():
                    with (root / (cid + ".jsonl")).open("ab") as tape:
                        tape.write(data)
                        tape.flush()
                        os.fsync(tape.fileno())
                raw_bytes += size
                with db:
                    for cid in touched:
                        metadata, reducer = active[cid]
                        snapshot = reducer.snapshot()
                        bars = snapshot.pop("bars")
                        metadata["writer_health"] = {
                            "dropped_events": self.dropped,
                            "queue_high_water": self.high_water,
                            "max_queue_latency_ms": self.max_queue_latency_ms,
                            "error": self.error,
                        }
                        db.execute(
                            "INSERT INTO captures VALUES (?,?,?,?,?) ON CONFLICT(capture_id) "
                            "DO UPDATE SET payload=excluded.payload",
                            (
                                cid,
                                metadata["session"],
                                metadata["con_id"],
                                metadata.get("segment_requested_at", metadata["requested_at"]),
                                json.dumps({**metadata, **snapshot}),
                            ),
                        )
                        for bar in bars:
                            if (cid, bar["minute"]) in changed:
                                db.execute(
                                    "INSERT INTO minutes VALUES (?,?,?) "
                                    "ON CONFLICT(capture_id,minute) "
                                    "DO UPDATE SET payload=excluded.payload",
                                    (cid, bar["minute"], json.dumps(bar)),
                                )
                for cid in ended:
                    active.pop(cid)
                if self.error and self.queue.empty():
                    break
        except Exception as exc:
            self.uncommitted_events = len(batch) + self.queue.qsize() if "batch" in locals() else 0
            self.error = "STORAGE_ERROR: " + str(exc)
        finally:
            self.ready.set()
            if db is not None:
                # Best effort only, in the observer DB. A full disk may prevent it;
                # unclosed captures are always shown as interrupted on restart.
                try:
                    with db:
                        for cid in active:
                            row = db.execute(
                                "SELECT payload FROM captures WHERE capture_id=?", (cid,)
                            ).fetchone()
                            if row:
                                persisted = json.loads(row[0])
                                persisted.update(
                                    end_reason=self.error or "WRITER_STOPPED",
                                    interrupted=True,
                                    dropped_events=self.dropped,
                                    uncommitted_events=self.uncommitted_events,
                                )
                                db.execute(
                                    "UPDATE captures SET payload=? WHERE capture_id=?",
                                    (json.dumps(persisted), cid),
                                )
                except sqlite3.Error:
                    pass
                db.close()


def read_flow(root: Path, session: str, con_id: int) -> dict[str, Any]:
    path = root / "summaries.sqlite3"
    if not path.exists():
        return {"captures": [], "bars": []}
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.1) as db:
        captures = [
            json.loads(row[0])
            for row in db.execute(
                "SELECT payload FROM captures WHERE session=? AND con_id=? "
                "ORDER BY requested_at DESC,rowid DESC LIMIT 32",
                (session, con_id),
            )
        ]
        bars = [
            dict(json.loads(row[1]), capture_id=row[0])
            for row in db.execute(
                "SELECT m.capture_id,m.payload FROM captures c JOIN minutes m USING(capture_id) "
                "WHERE c.session=? AND c.con_id=? "
                "ORDER BY m.minute DESC,c.requested_at DESC,c.rowid DESC LIMIT 400",
                (session, con_id),
            )
        ]
    return {"captures": list(reversed(captures)), "bars": list(reversed(bars))}


def replay(root: Path, session: str, con_id: int) -> dict[str, Any]:
    """Offline selected-identity export, raw evidence authoritative; no broker imports."""
    retained = read_flow(root.resolve(), session, con_id)
    saved = {c["capture_id"]: c for c in retained["captures"]}
    output = []
    # Offline export may read all rows for one bounded capture, not the UI's 400-row tail.
    saved_bars: dict[str, list[dict[str, Any]]] = {}
    index = root.resolve() / "summaries.sqlite3"
    if index.exists():
        with sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as db:
            for cid in saved:
                saved_bars[cid] = [
                    json.loads(row[0])
                    for row in db.execute(
                        "SELECT payload FROM minutes WHERE capture_id=? ORDER BY minute", (cid,)
                    )
                ]
    # Offline only: inspect raw headers so a crash before the index commit does
    # not hide a retained tape. Online refresh never scans raw files.
    for path in sorted(root.glob("*.jsonl")):
        with path.open() as tape:
            metadata = json.loads(next(tape))["capture"]
            if metadata["session"] != session or metadata["con_id"] != con_id:
                continue
            if metadata["classification_version"] != VERSION:
                raise ValueError("UNSUPPORTED_CLASSIFICATION_VERSION")
            persisted = saved.get(metadata["capture_id"])
            committed_totals = None
            committed_bars = None
            reducer = FlowReducer(metadata["config"]["quote_age_ms"])
            if persisted and persisted.get("last_sequence", 0) == 0:
                committed_totals = reducer.snapshot()["totals"]
                committed_bars = reducer.snapshot()["bars"]
            count = 0
            for line in tape:
                row = json.loads(line)  # A corrupt/partial tail fails explicitly.
                event = FlowEvent(**row["event"])
                if event.capture_id != metadata["capture_id"]:
                    raise ValueError("CAPTURE_ID_MISMATCH")
                result = reducer.apply(event)
                if result != row["classification"]:
                    raise ValueError("REPLAY_CLASSIFICATION_MISMATCH")
                count += int(event.kind == "trade")
                if persisted and event.sequence == persisted.get("last_sequence"):
                    committed_totals = reducer.snapshot()["totals"]
                    committed_bars = reducer.snapshot()["bars"]
        persisted = saved.get(metadata["capture_id"])
        output.append(
            {
                "capture_id": metadata["capture_id"],
                "trade_records": count,
                "replayed": reducer.snapshot(),
                "saved_totals_match": committed_totals == persisted["totals"]
                if persisted
                else None,
                "saved_minutes_match": committed_bars == saved_bars.get(metadata["capture_id"])
                if persisted
                else None,
                "summary_missing": persisted is None,
                "raw_tail_after_summary": bool(
                    persisted and reducer.last_sequence > persisted.get("last_sequence", 0)
                ),
            }
        )
    return {
        "session": session,
        "con_id": con_id,
        "classification_version": VERSION,
        "captures": output,
        "continuity": "SEPARATE_CAPTURE_SEGMENTS_NO_BACKFILL",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Offline FIRST4 order-flow replay/export")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--con-id", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.root, args.session, args.con_id), indent=2))


if __name__ == "__main__":
    main()
