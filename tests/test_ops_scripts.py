"""Operator scripts: consistent ledger backups and a preflight that cannot touch a live ledger."""

import asyncio
import fcntl
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from saxo_support import plan, signal
from stocker_execution.store import Store


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ledger_backup_is_consistent_verifiable_and_never_overwrites(tmp_path):
    backup = script("ledger_backup")
    database, config = tmp_path / "live" / "ledger.sqlite3", tmp_path / "saxo.sim.yaml"
    config.write_text("execution_mode: DISABLED\n")
    store = Store(database)
    store.bind("SAXO_SIM", "INTERNAL_PAPER")
    for i in range(3):
        event = signal(i)
        store.observe(event, "", {})
    assert store.reserve("fixture-0", plan()) == ""
    # The live connection stays open in WAL mode while the backup runs.
    manifest = backup.backup(database, config, tmp_path / "bundle")
    assert manifest["rows"]["signals"] == 3 and manifest["rows"]["reservations"] == 1
    assert manifest["provenance"]["execution_mode"] == "INTERNAL_PAPER"
    assert backup.verify(tmp_path / "bundle")["files"] == manifest["files"]
    copy = sqlite3.connect(tmp_path / "bundle" / "ledger.sqlite3")
    assert copy.execute("SELECT allocation_pennies FROM reservations").fetchall() == [(5000,)]
    copy.close()
    with pytest.raises(FileExistsError):
        backup.backup(database, config, tmp_path / "bundle")
    (tmp_path / "bundle" / "saxo.sim.yaml").write_text("execution_mode: SAXO_SIM\n")
    with pytest.raises(ValueError, match="CHECKSUM"):
        backup.verify(tmp_path / "bundle")
    store.db.close()


def test_preflight_refuses_a_held_owner_lock_before_opening_the_ledger(tmp_path):
    preflight = script("saxo_preflight")
    config = tmp_path / "saxo.sim.yaml"
    config.write_text(json.dumps({"data_environment": "SAXO_SIM"}))
    database = tmp_path / "ledger.sqlite3"
    lock = tmp_path / "SAXO_SIM" / "data.owner.lock"
    lock.parent.mkdir()
    with lock.open("a") as service:
        fcntl.flock(service, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="DATA_OWNER_LOCK_HELD"):
            asyncio.run(preflight.preflight(config, database, 5))
    assert not database.exists()
