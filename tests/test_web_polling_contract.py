from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
STATIC_ROOT = (
    ROOT / "packages" / "stocker_prospective" / "src" / "stocker_prospective" / "web_static"
)
POLLING = (STATIC_ROOT / "polling.js").read_text(encoding="utf-8")
APPLICATION = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
INDEX = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
SERVER_CONFIG = (ROOT / "configs/prospective/server.example.yaml").read_text(encoding="utf-8")


def interval_milliseconds(name: str) -> int:
    match = re.search(rf"{name}:\s*([\d_]+)", POLLING)
    assert match is not None
    return int(match.group(1).replace("_", ""))


def test_normal_visible_tab_polling_budget_is_below_forty_requests_per_minute() -> None:
    fast_per_minute = 60_000 / interval_milliseconds("fastIntervalMs")
    busiest_slow_screen = 2 * 60_000 / interval_milliseconds("slowIntervalMs")
    busiest_manual_screen = 2 * 60_000 / interval_milliseconds("manualIntervalMs")
    requests_per_minute = fast_per_minute + busiest_slow_screen + busiest_manual_screen

    assert requests_per_minute == 5.733333333333333
    assert requests_per_minute < 40
    configured_limit = int(
        re.search(r"web:.*?requests_per_minute:\s*(\d+)", SERVER_CONFIG, re.DOTALL).group(1)  # type: ignore[union-attr]
    )
    assert requests_per_minute * 2 < configured_limit


def test_fast_poll_only_requests_the_compact_summary() -> None:
    fast_body = APPLICATION.split("function refreshFast", 1)[1].split(
        "function refreshSlow",
        1,
    )[0]

    assert 'fastEndpoints: Object.freeze(["/api/dashboard/summary"])' in POLLING
    assert 'api("/api/dashboard/summary"' in fast_body
    for heavy_path in (
        "/api/audit/events",
        "/api/reports/daily",
        "/api/source-transfer",
        "/api/shadow-outcomes",
        "/api/episodes/",
        "/api/quiet-state/concentration-audit",
    ):
        assert heavy_path not in fast_body


def test_polling_supersedes_manual_refresh_and_pauses_while_hidden() -> None:
    refresh_all_body = APPLICATION.split("function refreshAll", 1)[1].split(
        "async function controlReplay",
        1,
    )[0]

    assert "new AbortController()" in POLLING
    assert "this.inFlight" in POLLING
    assert "if (existing && !supersede) return existing.promise" in POLLING
    assert "polling.cancelAll()" in refresh_all_body
    assert "refreshAllPromise" in refresh_all_body
    assert 'document.visibilityState === "visible"' in APPLICATION


def test_heavy_data_is_routed_by_visible_screen_and_polling_loads_first() -> None:
    assert POLLING.index('"/api/audit/events?limit=100"') > POLLING.index("manualEndpointsByScreen")
    assert POLLING.index('"/api/reports/daily"') > POLLING.index("manualEndpointsByScreen")
    assert INDEX.index("/assets/polling.js") < INDEX.index("/assets/app.js")
