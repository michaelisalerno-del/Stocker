"""Fair in-process coordination for recorder-side SQLite writers."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast


class _FairReentrantGate:
    """Serve database writers FIFO while allowing same-thread nesting."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._owner: int | None = None
        self._depth = 0

    def acquire(self) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            if self._owner == thread_id:
                self._depth += 1
                return
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving_ticket or self._owner is not None:
                self._condition.wait()
            self._owner = thread_id
            self._depth = 1

    def release(self) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            if self._owner != thread_id or self._depth <= 0:
                raise RuntimeError("SQLITE_WRITE_COORDINATION_NOT_OWNED")
            self._depth -= 1
            if self._depth:
                return
            self._owner = None
            self._serving_ticket += 1
            self._condition.notify_all()


_GATES_LOCK = threading.Lock()
_GATES: dict[str, _FairReentrantGate] = {}


DatabasePath = str | bytes | os.PathLike[str] | os.PathLike[bytes]


def _database_key(database: DatabasePath) -> str:
    value = os.fsdecode(os.fspath(database))
    if value.startswith("file:"):
        return value
    return str(Path(value).resolve())


def _gate_for(database: DatabasePath) -> _FairReentrantGate:
    key = _database_key(database)
    with _GATES_LOCK:
        gate = _GATES.get(key)
        if gate is None:
            gate = _FairReentrantGate()
            _GATES[key] = gate
        return gate


def _statement_may_write(statement: str) -> bool:
    stripped = statement.lstrip()
    while stripped.startswith("--"):
        _, separator, stripped = stripped.partition("\n")
        if not separator:
            return False
        stripped = stripped.lstrip()
    if not stripped:
        return False
    operation = stripped.split(None, maxsplit=1)[0].upper()
    return operation not in {"SELECT", "EXPLAIN"}


class CoordinatedSQLiteConnection(sqlite3.Connection):
    """Hold one fair per-database gate for the life of each write transaction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        database = cast(DatabasePath, args[0] if args else kwargs["database"])
        super().__init__(*args, **kwargs)
        self._write_gate = _gate_for(database)
        self._write_gate_owned = False

    def _acquire_write_gate(self) -> bool:
        if self._write_gate_owned:
            return False
        self._write_gate.acquire()
        self._write_gate_owned = True
        return True

    def _release_write_gate(self) -> None:
        if not self._write_gate_owned:
            return
        self._write_gate_owned = False
        self._write_gate.release()

    def _release_if_autocommit(self, *, acquired: bool) -> None:
        if acquired and not self.in_transaction:
            self._release_write_gate()

    def execute(
        self,
        sql: str,
        parameters: Any = (),
        /,
    ) -> sqlite3.Cursor:
        acquired = self._acquire_write_gate() if _statement_may_write(sql) else False
        try:
            cursor = super().execute(sql, parameters)
        except BaseException:
            if acquired and not self.in_transaction:
                self._release_write_gate()
            raise
        self._release_if_autocommit(acquired=acquired)
        return cursor

    def executemany(
        self,
        sql: str,
        parameters: Any,
        /,
    ) -> sqlite3.Cursor:
        acquired = self._acquire_write_gate() if _statement_may_write(sql) else False
        try:
            cursor = super().executemany(sql, parameters)
        except BaseException:
            if acquired and not self.in_transaction:
                self._release_write_gate()
            raise
        self._release_if_autocommit(acquired=acquired)
        return cursor

    def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
        acquired = self._acquire_write_gate()
        try:
            cursor = super().executescript(sql_script)
        except BaseException:
            if acquired and not self.in_transaction:
                self._release_write_gate()
            raise
        self._release_if_autocommit(acquired=acquired)
        return cursor

    def commit(self) -> None:
        try:
            super().commit()
        finally:
            if not self.in_transaction:
                self._release_write_gate()

    def rollback(self) -> None:
        try:
            super().rollback()
        finally:
            if not self.in_transaction:
                self._release_write_gate()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> Literal[False]:
        try:
            return super().__exit__(exception_type, exception, traceback)
        finally:
            if not self.in_transaction:
                self._release_write_gate()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._release_write_gate()
