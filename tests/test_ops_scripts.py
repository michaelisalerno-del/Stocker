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
    assert copy.execute("SELECT allocation_pennies FROM reservations").fetchall() == [(100000,)]
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


def test_reference_audit_picks_prior_session_volume_leader_and_loads_as_the_runtime_does(tmp_path):
    from datetime import date

    from stocker_execution.config import ContractSelection
    from stocker_execution.reference_sessions import load_selections

    audit = script("reference_audit")
    pinned = ContractSelection(
        environment="SAXO_LIVE",
        uic=1,
        symbol="CLX6",
        exchange="NYMEX",
        contract_month="2026-11",
        approval="fixture pin approval",
    )
    nearby = [
        {
            "uic": 1,
            "symbol": "CLX6",
            "exchange": "NYMEX",
            "contract_month": "2026-11",
            "expiry": "2026-10-20",
        },
        {
            "uic": 2,
            "symbol": "CLZ6",
            "exchange": "NYMEX",
            "contract_month": "2026-12",
            "expiry": "2026-11-20",
        },
    ]
    days = [
        "2026-09-22",
        "2026-09-23",
        "2026-09-24",
        "2026-09-25",
        "2026-09-28",
        "2026-09-29",
        "2026-09-30",
    ]
    volumes = {1: {d: 300000.0 for d in days}, 2: {d: 200000.0 for d in days}}
    volumes[2]["2026-09-24"] = 400000.0  # December led on the 24th, so it is the 25th's reference
    book = audit.build_audit(
        "CL", "SAXO_LIVE", pinned, nearby, volumes, "2026-09-30", "fixture standing approval"
    )
    assert [s.day.isoformat() for s in book.sessions] == days[
        1:6
    ]  # today's partial sample excluded
    assert [s.contract.symbol for s in book.sessions] == ["CLX6", "CLX6", "CLZ6", "CLX6", "CLX6"]
    assert "CLZ6 400,000" in book.sessions[2].prior_session_volume_evidence
    path = tmp_path / "selections.json"
    path.write_text(json.dumps([json.loads(book.model_dump_json())]))
    loaded, _ = load_selections(path, "SAXO_LIVE", "CL", date(2026, 9, 30), 1)
    assert loaded.current_uic == 1
    with pytest.raises(ValueError, match="FEWER_THAN_SIX"):
        audit.build_audit(
            "CL",
            "SAXO_LIVE",
            pinned,
            nearby,
            {1: {d: 1.0 for d in days[:5]}},
            "2026-09-30",
            "x" * 20,
        )


def test_reference_audit_targets_the_next_session_after_the_close():
    from datetime import datetime

    audit = script("reference_audit")
    day = lambda *a: audit.audit_day(datetime(*a, tzinfo=audit.NY))  # noqa: E731
    assert day(2026, 10, 5, 7, 30) == "2026-10-05"  # Monday morning: today's session
    assert day(2026, 10, 8, 17, 10) == "2026-10-09"  # Thursday after the close: Friday's
    assert day(2026, 10, 9, 17, 10) == "2026-10-12"  # Friday after the close: Monday's
