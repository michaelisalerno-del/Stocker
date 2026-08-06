"""Synthetic, temporary-only measurements for the prospective web repair."""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from stocker_prospective.config import ProspectiveConfig
from stocker_prospective.replay import ReplaySettings, run_deterministic_replay
from stocker_prospective.web import create_web_app

ROOT = Path(__file__).parents[1]
RUN_ID = "synthetic-web-measurement"
HISTORY_ROWS = 10_000
SAMPLES = 30


class _RequestMetricHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[dict[str, object]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = json.loads(record.getMessage())
        except json.JSONDecodeError:
            return
        if (
            payload.get("event") == "request_completed"
            and payload.get("route") == "/api/dashboard/summary"
        ):
            self.items.append(payload)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def _config(root: Path) -> ProspectiveConfig:
    return ProspectiveConfig.model_validate(
        {
            "paths": {
                "database": str(root / "prospective.sqlite3"),
                "bundle_root": str(root / "bundles"),
                "prospective_report_root": str(root / "reports"),
                "feature_parity_report": str(root / "absent-feature-parity.json"),
            },
            "runtime": {
                "mode": "shadow",
                "source": "replay",
                "prospective_start_utc": "2025-01-02T13:00:00Z",
                "instance_id": "synthetic-measurement",
                "app_version": "synthetic",
                "git_commit": "deadbeef",
                "run_id": RUN_ID,
            },
            "risk": {"trading_enabled": False},
            "web": {
                "host": "127.0.0.1",
                "production": True,
                "requests_per_minute": 10_000,
                "allowed_hosts": ["testserver", "127.0.0.1", "localhost"],
            },
            "ibkr": {},
            "context": {
                "mode": "signed_import",
                "hmac_secret_env": "SYNTHETIC_UNUSED_SECRET",
            },
            "parallel_validation": {"enabled": False},
        }
    )


def _seed(config: ProspectiveConfig) -> None:
    run_deterministic_replay(
        ReplaySettings(
            database_path=config.paths.database,
            run_id=RUN_ID,
            prospective_start_utc=datetime(2025, 1, 2, 13, 0, tzinfo=UTC),
            app_version="synthetic",
            git_commit="deadbeef",
            universe_path=ROOT / "configs/prospective/anchor-frozen-20.json",
            owner_id="synthetic-measurement",
            recorder_lease_stale_seconds=60,
        )
    )


def _enlarge_history(config: ProspectiveConfig) -> None:
    start = datetime(2025, 1, 2, 14, 0, tzinfo=UTC)
    rows = [
        (
            RUN_ID,
            "2025-01-02",
            "AAPL",
            f"/synthetic/not-opened/{index}.parquet",
            (start + timedelta(seconds=index)).isoformat(),
            f"{index:064x}",
        )
        for index in range(HISTORY_ROWS)
    ]
    with sqlite3.connect(config.paths.database) as connection:
        connection.executemany(
            """
            INSERT INTO raw_partition_manifest_v0(
                run_id, data_source, session_date, symbol, event_type,
                file_path, row_count, minimum_timestamp_utc,
                maximum_timestamp_utc, schema_version, content_hash,
                complete, gap_count, recorder_version, contract_version,
                recorded_at_utc, claims_json
            ) VALUES (?, 'synthetic', ?, ?,
                      'underlying_level1_quote_event', ?, 1, ?, ?,
                      'synthetic', ?, 1, 0, 'synthetic', 'synthetic', ?, '{}')
            """,
            [
                (
                    run_id,
                    session_date,
                    symbol,
                    file_path,
                    observed_at,
                    observed_at,
                    content_hash,
                    observed_at,
                )
                for (
                    run_id,
                    session_date,
                    symbol,
                    file_path,
                    observed_at,
                    content_hash,
                ) in rows
            ],
        )


def _measure(config: ProspectiveConfig) -> dict[str, float | int]:
    client = TestClient(create_web_app(config))
    for _ in range(3):
        response = client.get("/api/dashboard/summary")
        response.raise_for_status()
    durations: list[float] = []
    response_size = 0
    logger = logging.getLogger("uvicorn.error.stocker_prospective.web")
    prior_level = logger.level
    handler = _RequestMetricHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        for _ in range(SAMPLES):
            started = time.perf_counter()
            response = client.get("/api/dashboard/summary")
            durations.append((time.perf_counter() - started) * 1_000.0)
            response.raise_for_status()
            response_size = len(response.content)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
    ordered = sorted(durations)
    server_elapsed = [float(item["elapsed_ms"]) for item in handler.items]
    sqlite_durations = [float(item["sqlite_duration_ms"]) for item in handler.items]
    sqlite_operations = [int(item["sqlite_operations"]) for item in handler.items]
    return {
        "samples": SAMPLES,
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(_percentile(ordered, 0.95), 3),
        "maximum_ms": round(max(ordered), 3),
        "server_median_ms": round(statistics.median(server_elapsed), 3),
        "server_p95_ms": round(_percentile(server_elapsed, 0.95), 3),
        "server_maximum_ms": round(max(server_elapsed), 3),
        "sqlite_operations_median": int(statistics.median(sqlite_operations)),
        "sqlite_operations_maximum": max(sqlite_operations),
        "sqlite_duration_median_ms": round(statistics.median(sqlite_durations), 3),
        "parquet_files_examined_maximum": max(
            int(item["parquet_files_examined"]) for item in handler.items
        ),
        "parquet_input_rows_maximum": max(
            int(item["parquet_input_rows"]) for item in handler.items
        ),
        "response_bytes": response_size,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stocker-web-measurement-") as temporary:
        root = Path(temporary)
        baseline_config = _config(root / "baseline")
        enlarged_config = _config(root / "enlarged")
        _seed(baseline_config)
        _seed(enlarged_config)
        _enlarge_history(enlarged_config)
        result = {
            "method": "FastAPI TestClient, warmed, sequential, temporary synthetic SQLite",
            "historical_manifest_rows_added": HISTORY_ROWS,
            "baseline": _measure(baseline_config),
            "enlarged": _measure(enlarged_config),
            "polling_requests_per_minute_busiest_screen": round(
                (60_000 / 15_000) + (3 * 60_000 / 90_000) + (5 * 60_000 / 300_000),
                3,
            ),
            "replay_default_maximum_records": 250_000,
            "replay_default_maximum_materialized_bytes": 64 * 1024 * 1024,
        }
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
