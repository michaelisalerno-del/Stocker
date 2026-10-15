"""Operator scripts: the shipped config loads, and paper-record backups are consistent."""

import importlib.util
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stocker_execution.store import Store


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_tracked_config_still_loads():
    from stocker_execution.config import load

    paths = sorted(Path("configs").glob("*.yaml"))
    assert paths
    for path in paths:
        load(path)


def test_ledger_backup_is_consistent_verifiable_and_never_overwrites(tmp_path):
    backup = script("ledger_backup")
    database, config = tmp_path / "live" / "paper.sqlite3", tmp_path / "paper.yaml"
    config.write_text("data_environment: SAXO_SIM\n")
    store = Store(database)
    store.bind("SAXO_SIM")
    now = datetime(2026, 10, 19, 13, 35, tzinfo=UTC)
    for day in ("2026-10-19", "2026-10-20"):
        store.add_signal(day, "FIRST_CANDLE", "NAS100", 1, "", {"close": 1})
        store.set_event(day, "FIRST_CANDLE", now, "DONE")
    store.insert(
        "cfd_trades",
        {
            "day": "2026-10-19",
            "market": "NAS100",
            "uic": 4912,
            "side": 1,
            "amount": 0.65,
            "factor": 1.0,
            "stop": 29700.0,
            "entry_at": now.isoformat(),
            "entry_price": 30000.0,
            "entry_bid": 29999.0,
            "entry_ask": 30000.0,
        },
    )
    # The live connection stays open in WAL mode while the backup runs.
    manifest = backup.backup(database, config, tmp_path / "bundle")
    assert manifest["rows"]["signals"] == 2 and manifest["rows"]["cfd_trades"] == 1
    assert manifest["provenance"] == {"data_environment": "SAXO_SIM", "record": "PAPER"}
    assert backup.verify(tmp_path / "bundle")["files"] == manifest["files"]
    copy = sqlite3.connect(tmp_path / "bundle" / "ledger.sqlite3")
    assert copy.execute("SELECT amount FROM cfd_trades").fetchall() == [(0.65,)]
    copy.close()
    with pytest.raises(FileExistsError):
        backup.backup(database, config, tmp_path / "bundle")
    (tmp_path / "bundle" / "paper.yaml").write_text("data_environment: SAXO_LIVE\n")
    with pytest.raises(ValueError, match="CHECKSUM"):
        backup.verify(tmp_path / "bundle")
    store.db.close()
