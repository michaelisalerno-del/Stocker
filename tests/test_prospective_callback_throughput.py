from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.durable_inbox import DurableCallbackInbox
from stocker_prospective.ibkr import IBKRConnectionConfig, IBKRMarketDataAdapter
from stocker_prospective.market_data import MarketDataBudget, MarketDataType

NOW = datetime(2026, 8, 5, 19, 27, tzinfo=UTC)
RUN_ID = "run-callback-throughput"
QUOTE_STREAMS = 21
BAR_STREAMS = 28
BATCHES = 4


def _adapter(tmp_path: Path) -> tuple[IBKRMarketDataAdapter, DurableCallbackInbox]:
    database_path = tmp_path / "prospective.sqlite3"
    repository = ProspectiveRepository(database_path)
    repository.migrate()
    repository.create_run(
        EvidenceMetadata(
            run_id=RUN_ID,
            prospective_start_utc=NOW - timedelta(days=1),
            app_version="test",
            git_commit="a" * 40,
            model_artifact_id="frozen-m1c",
            universe_id="anchor-frozen-20",
            cohort="anchor_frozen_20",
            source_timestamps=[NOW.isoformat()],
            recorded_at_utc=NOW,
        )
    )
    inbox = DurableCallbackInbox(
        database_path,
        max_unacknowledged=10_000,
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
    )
    adapter = IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=4002,
            client_id=91,
            expected_environment="read_only",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=1,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=MarketDataBudget(
            line_limit=100,
            reserved_headroom=1,
            request_rate_limit=1_000,
        ),
        durable_inbox=inbox,
        require_durable_inbox_on_start=True,
    )
    adapter._connection_generation = 1
    for index in range(QUOTE_STREAMS):
        request_id = 100 + index
        symbol = f"Q{index:02d}"
        adapter._track_request(request_id, f"{symbol}:level1")
        adapter._subscription_kinds[request_id] = "market_data"
        adapter.stream_quotes.register(request_id)
        adapter.register_stream_owner(
            request_id,
            {
                "request_id": request_id,
                "kind": "underlying_level1",
                "symbol": symbol,
                "con_id": 1_000 + index,
                "exchange": "SMART",
                "episode_id": None,
                "option_contract": None,
            },
        )
    for index in range(BAR_STREAMS):
        request_id = 200 + index
        symbol = f"B{index:02d}"
        adapter._track_request(request_id, f"{symbol}:bars")
        adapter._subscription_kinds[request_id] = "historical_bars"
        adapter.register_stream_owner(
            request_id,
            {
                "request_id": request_id,
                "kind": "historical_bars",
                "symbol": symbol,
                "con_id": 2_000 + index,
                "exchange": "SMART",
                "episode_id": None,
                "option_contract": None,
            },
        )
    return adapter, inbox


def _database_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.exists()
    )


def test_mixed_live_callback_load_retains_one_row_per_provider_delivery(
    tmp_path: Path,
) -> None:
    adapter, inbox = _adapter(tmp_path)
    connection = inbox._connect()
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    bytes_before = _database_bytes(inbox.database_path)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    batch_metrics: list[dict[str, float | int]] = []
    provider_pending_before_normalization: list[int] = []

    for batch in range(BATCHES):
        for index in range(QUOTE_STREAMS):
            request_id = 100 + index
            payload = {
                "field": "bid" if batch % 2 == 0 else "ask",
                "value": 10.0 + index + batch / 100,
                "market_data_type": "live",
                "receive_timestamp_utc": NOW.isoformat(),
            }
            def normalize_quote(
                request_id: int = request_id,
                payload: dict[str, object] = payload,
                index: int = index,
            ) -> None:
                if index == 0:
                    provider_pending_before_normalization.append(
                        int(
                            connection.execute(
                                """
                                SELECT COUNT(*)
                                FROM callback_inbox_v1
                                WHERE status = 'provider_pending'
                                """
                            ).fetchone()[0]
                        )
                    )
                adapter.on_quote_update(request_id, payload)

            adapter.contain_official_callback(
                "tick_price",
                request_id,
                normalize_quote,
                provider_arguments=(request_id, 1, payload["value"], {}),
            )
        provider_bar_at = NOW - timedelta(hours=2) + timedelta(minutes=5 * batch)
        for index in range(BAR_STREAMS):
            request_id = 200 + index
            payload = {
                "provider_bar_timestamp": str(int(provider_bar_at.timestamp())),
                "bar_start_utc": provider_bar_at.isoformat(),
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.05,
                "volume": 1_000.0,
                "wap": 10.02,
                "trade_count": 20,
                "source": "ibkr_historical_keep_up_to_date",
            }
            adapter.contain_official_callback(
                "historical_data_update",
                request_id,
                lambda request_id=request_id, payload=payload: adapter.on_historical_bar(
                    request_id,
                    payload,
                    update=True,
                ),
                provider_arguments=(request_id, payload),
            )
        adapter.flush_pending_callback_failure()
        accounting = inbox.accounting()
        batch_wall_at = NOW + timedelta(minutes=10 * batch)
        batch_metrics.append(
            {
                "batch": batch + 1,
                "cpu_seconds": round(time.process_time() - cpu_started, 6),
                "wall_seconds": round(time.perf_counter() - wall_started, 6),
                "sqlite_begin_operations": sum(
                    statement.startswith("BEGIN") for statement in statements
                ),
                "database_growth_bytes": (_database_bytes(inbox.database_path) - bytes_before),
                "durable_backlog": (
                    accounting.pending + accounting.leased + accounting.quarantined
                ),
                "provider_bar_lag_seconds": int((batch_wall_at - provider_bar_at).total_seconds()),
                "local_inbox_age_seconds": 0,
            }
        )

    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    connection.set_trace_callback(None)
    expected_callbacks = BATCHES * (QUOTE_STREAMS + BAR_STREAMS)
    rows = connection.execute(
        """
        SELECT callback_kind, original_payload_json
        FROM callback_inbox_v1
        ORDER BY source_sequence
        """
    ).fetchall()
    accounting = inbox.accounting()
    metrics = {
        "callbacks": expected_callbacks,
        "batches": BATCHES,
        "cpu_seconds": round(cpu_seconds, 6),
        "wall_seconds": round(wall_seconds, 6),
        "callbacks_per_wall_second": round(expected_callbacks / wall_seconds, 3),
        "sqlite_begin_operations": sum(statement.startswith("BEGIN") for statement in statements),
        "sqlite_write_operations": sum(
            statement.lstrip().startswith(("INSERT", "UPDATE", "DELETE"))
            for statement in statements
        ),
        "database_growth_bytes": _database_bytes(inbox.database_path) - bytes_before,
        "durable_backlog": (accounting.pending + accounting.leased + accounting.quarantined),
        "provider_bar_lag_seconds": int(batch_metrics[-1]["provider_bar_lag_seconds"]),
        "repeated_batches": batch_metrics,
    }
    print(json.dumps(metrics, sort_keys=True))

    assert adapter.fatal_callback_code is None
    assert len(rows) == expected_callbacks
    assert metrics["durable_backlog"] == expected_callbacks
    assert metrics["sqlite_begin_operations"] <= BATCHES * 2
    assert provider_pending_before_normalization == [49, 49, 49, 49]
    assert all(
        "original_provider_callback" not in quote
        for quote in adapter.stream_quotes.snapshot(100)
    )
    assert [item["durable_backlog"] for item in batch_metrics] == [49, 98, 147, 196]
    provider_lags = [int(item["provider_bar_lag_seconds"]) for item in batch_metrics]
    assert provider_lags == sorted(provider_lags)
    assert len(set(provider_lags)) == BATCHES
    assert all(item["local_inbox_age_seconds"] == 0 for item in batch_metrics)
    assert all(not str(row["callback_kind"]).startswith("official_provider_") for row in rows)
    for row in rows:
        payload = json.loads(str(row["original_payload_json"]))
        provider = payload["original_provider_callback"]
        encoded = json.dumps(provider, sort_keys=True, separators=(",", ":"))
        assert (
            payload["original_provider_callback_sha256"]
            == hashlib.sha256(encoded.encode()).hexdigest()
        )
