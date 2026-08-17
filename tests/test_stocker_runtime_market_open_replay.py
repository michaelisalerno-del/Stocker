from __future__ import annotations

from pathlib import Path

from stocker_runtime.opening_burst import (
    CI_OPENING_BURST_SECONDS,
    OPENING_BURST_FEEDS,
    opening_burst_callback_count,
    opening_burst_failures,
    run_opening_burst,
)
from stocker_runtime.storage import RetentionPolicy


def test_deterministic_ten_second_opening_burst_preserves_durable_evidence(
    tmp_path: Path,
) -> None:
    result = run_opening_burst(
        tmp_path / "opening-burst.sqlite3",
        seconds=CI_OPENING_BURST_SECONDS,
    )

    assert opening_burst_callback_count(CI_OPENING_BURST_SECONDS) == 2_022
    assert result["required_feeds_expected"] == OPENING_BURST_FEEDS
    assert result["wal_bytes_after_session_checkpoint"] < RetentionPolicy().wal_cap_bytes
    assert opening_burst_failures(result, include_performance=False) == ()
