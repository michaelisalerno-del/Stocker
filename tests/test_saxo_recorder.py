import asyncio
import gzip
import json

from stocker_execution.config import RecorderConfig
from stocker_execution.recorder import Recorder

IDENTITY = {
    "provider": "SAXO",
    "environment": "SAXO_SIM",
    "uic": 100,
    "market": "CL",
    "asset_type": "ContractFutures",
}


def recorder(tmp_path, **overrides):
    config = RecorderConfig(
        persistent_capture=True,
        recording_permission_evidence="OFFLINE FIXTURE ONLY",
        disk_reserve_bytes=0,
        **overrides,
    )
    result = Recorder(config, tmp_path / "events")
    result.register("CL-contract", IDENTITY)
    return result


def trigger(i, **kwargs):
    return {
        "id": f"event-{i}",
        "rule_version": "fixture",
        "skip_reason": "MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET",
        **kwargs,
    }


def test_window_expiry_keeps_valid_advancing_checkpoint(tmp_path):
    r = recorder(tmp_path)
    r.ingest("CL-contract", "SNAPSHOT", {"Quote": {"Bid": 10, "Ask": 11}}, 0)
    for t in range(1, 1002):
        r.ingest("CL-contract", "UPDATE", {"Quote": {"Ask": t}}, t, message_id=str(t))
    w = r.windows["CL-contract"]
    assert w.rows[0][0] == 101
    assert w.checkpoint == {"Quote": {"Bid": 10, "Ask": 100}}
    assert w.current == {"Quote": {"Bid": 10, "Ask": 1001}}
    assert w.coverage(1001) == 900
    assert not list(r.directory.glob("*.gz"))


def test_memory_cap_and_gap_coverage(tmp_path):
    r = recorder(tmp_path, rolling_max_bytes=65536)
    r.ingest("CL-contract", "SNAPSHOT", {"Quote": {"Bid": 10}}, 0)
    for t in range(1000):
        r.ingest("CL-contract", "UPDATE", {"Quote": {"Bid": 10}, "padding": "x" * 500}, t)
    assert r.memory() <= 65536
    assert r.windows["CL-contract"].coverage(999) < 900
    r.ingest("CL-contract", "GAP", {"reason": "RESET"}, 1000)
    assert r.windows["CL-contract"].coverage(1000) == 0
    r.ingest("CL-contract", "SNAPSHOT", {"Quote": {"Bid": 11}}, 1001)
    assert r.windows["CL-contract"].coverage(1005) == 4


def test_atomic_overlap_skips_and_incremental_crash_evidence(tmp_path):
    async def scenario():
        r = recorder(tmp_path)
        await r.start()
        r.ingest("CL-contract", "SNAPSHOT", {"Quote": {"Bid": 10}}, 0)
        one = r.trigger("CL-contract", trigger(1), 10)
        two = r.trigger("CL-contract", trigger(2), 20)
        duplicate = r.trigger("CL-contract", trigger(1), 21)
        assert one["segment"] == two["segment"]
        assert duplicate["state"] == "DUPLICATE_SUPPRESSED"
        r.ingest("CL-contract", "UPDATE", {"Quote": {"Bid": 11}}, 30)
        await r.queue.join()
        path = r.directory / (one["segment"] + ".jsonl.gz")
        rows = [json.loads(x) for x in gzip.decompress(path.read_bytes()).splitlines()]
        assert sum(x["kind"] == "SNAPSHOT" for x in rows) == 1
        assert sum(x["kind"] == "TRIGGER" for x in rows) == 2
        assert next(x for x in rows if x["kind"] == "TRIGGER")["payload"]["skip_reason"]
        # Simulate crash after durable writes; append a partial next gzip member.
        r.closed = True
        await r.worker
        with path.open("ab") as f:
            f.write(b"\x1f\x8b\x08\x00truncated")
        restored = recorder(tmp_path)
        restored.recover()
        manifest = json.loads(path.with_name(one["segment"] + ".manifest.json").read_text())
        assert manifest["state"] == "INCOMPLETE"
        assert manifest["reason"] == "INTERRUPTED_RESTART"
        assert gzip.decompress(path.read_bytes())

    asyncio.run(scenario())


def test_capture_holds_through_close_plus_five_minutes(tmp_path):
    r = recorder(tmp_path)
    r.ingest("CL-contract", "SNAPSHOT", {}, 0)
    seg = r.trigger("CL-contract", trigger(1), 1)["segment"]
    r.link_trade("event-1", True, 2)
    r.tick(5000)
    assert r.active[seg]["state"] == "CAPTURING"
    r.link_trade("event-1", False, 5000)
    r.tick(5299)
    assert r.active[seg]["state"] == "CAPTURING"
    r.tick(5300)
    assert r.active[seg]["state"] == "COMPLETE"


def test_storage_and_queue_pressure_do_not_block_ingest(tmp_path):
    async def scenario():
        r = recorder(tmp_path, archive_max_bytes=65536)
        await r.start()
        r.ingest("CL-contract", "SNAPSHOT", {}, 0)
        r.trigger("CL-contract", trigger(1), 1)
        await r.queue.join()
        assert r.problem == "STORAGE_LIMIT_REACHED"
        assert not list(r.directory.glob("*.gz"))
        r.ingest("CL-contract", "UPDATE", {"Quote": {"Bid": 1}}, 2)
        assert r.windows["CL-contract"].current["Quote"]["Bid"] == 1
        await r.close()

    asyncio.run(scenario())
    q = recorder(tmp_path / "queue", queue_max_items=2)
    q.ingest("CL-contract", "SNAPSHOT", {}, 0)
    seg = q.trigger("CL-contract", trigger(1), 1)["segment"]
    for t in range(2, 10):
        q.ingest("CL-contract", "UPDATE", {}, t)
    assert q.problem == "" and q.queue.qsize() == 2
    assert q.active[seg]["state"] == "INCOMPLETE"
    assert q.active[seg]["reason"] == "WRITE_QUEUE_LIMIT_REACHED"


def test_queue_pressure_ends_one_capture_and_later_captures_still_record(tmp_path):
    async def scenario():
        r = recorder(tmp_path, queue_max_items=2)
        r.ingest("CL-contract", "SNAPSHOT", {}, 0)
        first = r.trigger("CL-contract", trigger(1), 1)["segment"]
        for t in range(2, 6):  # the writer is not running yet: the queue fills
            r.ingest("CL-contract", "UPDATE", {"Quote": {"Bid": t}}, t)
        assert r.active[first]["state"] == "INCOMPLETE" and first in r.markers_pending
        await r.start()
        await r.queue.join()
        r.tick(6)  # the drained queue takes the marker
        await r.queue.join()
        manifest = json.loads((r.directory / (first + ".manifest.json")).read_text())
        assert manifest["state"] == "INCOMPLETE"
        assert manifest["reason"] == "WRITE_QUEUE_LIMIT_REACHED"
        assert not r.markers_pending and r.problem == ""
        second = r.trigger("CL-contract", trigger(2), 7)
        assert second["state"] == "CAPTURING" and second["segment"] != first
        r.ingest("CL-contract", "UPDATE", {"Quote": {"Bid": 7}}, 8)
        await r.queue.join()
        assert (r.directory / (second["segment"] + ".jsonl.gz")).exists()
        await r.close()

    asyncio.run(scenario())


def test_prefix_bridges_by_sequence_and_rows_only_writes_keep_the_archive_complete(tmp_path):
    async def scenario():
        r = recorder(tmp_path)
        await r.start()
        r.ingest("CL-contract", "SNAPSHOT", {"Quote": {"Bid": 1}}, 0)
        for t in range(1, 5):
            r.ingest("CL-contract", "UPDATE", {"Quote": {"Bid": t}}, t)
        w = r.windows["CL-contract"]
        assert [json.loads(b)["local_sequence"] for b in w.prefix(2)] == [3, 4, 5]
        full = w.prefix()
        assert json.loads(full[0])["kind"] == "CHECKPOINT"
        assert [json.loads(b)["local_sequence"] for b in full[1:]] == [1, 2, 3, 4, 5]
        segment = r.trigger("CL-contract", trigger(1), 5)["segment"]
        for t in range(6, 9):
            r.ingest("CL-contract", "UPDATE", {"Quote": {"Bid": t}}, t)
        await r.queue.join()
        path = r.directory / (segment + ".jsonl.gz")
        rows = [json.loads(x) for x in gzip.decompress(path.read_bytes()).splitlines()]
        assert [x["local_sequence"] for x in rows if "local_sequence" in x] == list(range(1, 9))
        manifest = json.loads(path.with_name(segment + ".manifest.json").read_text())
        assert manifest["state"] == "CAPTURING" and manifest["committed_bytes"] > 0
        await r.close()

    asyncio.run(scenario())


def test_permissions_do_not_disable_rolling_buffer(tmp_path):
    r = Recorder(RecorderConfig(), tmp_path)
    r.register("CL-contract", IDENTITY)
    r.ingest("CL-contract", "SNAPSHOT", {}, 0)
    assert r.view("CL-contract", 10)["state"] == "BUFFERING"
    assert r.trigger("CL-contract", trigger(1), 10)["reason"] == "RECORDING_PERMISSION_NOT_VERIFIED"


def test_completed_overlap_bridges_without_duplicate_or_full_session_retention(tmp_path):
    async def scenario():
        r = recorder(tmp_path)
        await r.start()
        r.ingest("CL-contract", "SNAPSHOT", {"Quote": {"Bid": 1}}, 0)
        first = r.trigger("CL-contract", trigger(1), 1)
        r.tick(3601)
        await r.queue.join()
        path = r.directory / (first["segment"] + ".jsonl.gz")
        before = path.read_bytes()
        r.ingest("CL-contract", "UPDATE", {"Quote": {"Bid": 2}}, 3650)
        await r.queue.join()
        assert path.read_bytes() == before  # no continuous post-window archive
        second = r.trigger("CL-contract", trigger(2), 3700)
        assert first["segment"] == second["segment"] and second["state"] == "CAPTURING"
        r.ingest("CL-contract", "UPDATE", {"Quote": {"Bid": 3}}, 3701)
        await r.queue.join()
        rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
        assert [x["local_sequence"] for x in rows if "local_sequence" in x] == [1, 2, 3]
        await r.close()

    asyncio.run(scenario())


def test_option_attachment_reports_only_collected_history_and_prune_is_protected(tmp_path):
    async def scenario():
        r = recorder(tmp_path)
        await r.start()
        r.ingest("CL-contract", "SNAPSHOT", {}, 0)
        first = r.trigger("CL-contract", trigger(1), 1000)
        r.register("option", {**IDENTITY, "asset_type": "FuturesOption", "uic": 101})
        r.ingest("option", "SNAPSHOT", {"Quote": {"Ask": 0.01}}, 1001)
        r.attach("event-1", "option", 1001)
        await r.queue.join()
        path = r.directory / (first["segment"] + ".jsonl.gz")
        rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
        assert (
            next(x for x in rows if x["kind"] == "OPTION_ATTACHED")["actual_prehistory_seconds"]
            == 0
        )
        r.tick(5601)
        await r.queue.join()
        import pytest

        with pytest.raises(ValueError, match="REFERENCED"):
            r.prune(first["segment"], {first["segment"]})
        assert path.exists()
        r.prune(first["segment"], set())
        assert not path.exists()
        await r.close()

    asyncio.run(scenario())
