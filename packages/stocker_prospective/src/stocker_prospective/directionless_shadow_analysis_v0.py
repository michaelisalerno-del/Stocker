"""Post-period-only report for the one frozen directionless V0 configuration."""

from __future__ import annotations

import json
import random
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from stocker_prospective.m1c_directionless_shadow_v0 import DirectionlessShadowEpisodeV0


def _mean(values: list[float]) -> float | None:
    return None if not values else statistics.fmean(values)


def _median(values: list[float]) -> float | None:
    return None if not values else statistics.median(values)


def _clustered_interval(
    episodes: list[DirectionlessShadowEpisodeV0],
    field: str,
) -> list[float] | None:
    by_session: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        value = getattr(episode, field)
        if value is not None:
            by_session[episode.session.isoformat()].append(float(value))
    sessions = sorted(by_session)
    if len(sessions) < 2:
        return None
    generator = random.Random(20260815)
    draws: list[float] = []
    for _ in range(5_000):
        sampled = [generator.choice(sessions) for _ in sessions]
        values = [value for session in sampled for value in by_session[session]]
        draws.append(statistics.fmean(values))
    draws.sort()
    return [draws[124], draws[4_874]]


def analyse_directionless_shadow_v0(
    database_path: Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Read one completed V0 run; callers must enforce the opening receipt."""

    uri = f"file:{database_path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        rows = connection.execute(
            "SELECT payload_json FROM m1c_directionless_shadow_v0 "
            "WHERE run_id = ? AND strategy_version = 'm1c_directionless_t20_d_v0' "
            "ORDER BY session_date, t0_utc, symbol",
            (run_id,),
        ).fetchall()
    finally:
        connection.close()
    episodes = [
        DirectionlessShadowEpisodeV0.model_validate_json(row["payload_json"]) for row in rows
    ]
    entered = [episode for episode in episodes if episode.direction is not None]
    unresolved = [episode for episode in episodes if "UNRESOLVED" in episode.state.value]
    clean = [
        episode for episode in entered if episode not in unresolved and episode.net_M is not None
    ]
    net_m = [float(episode.net_M) for episode in clean if episode.net_M is not None]
    net_r = [float(episode.net_R) for episode in clean if episode.net_R is not None]

    def stability(key: str) -> dict[str, dict[str, float | int | None]]:
        grouped: dict[str, list[DirectionlessShadowEpisodeV0]] = defaultdict(list)
        for episode in clean:
            grouped[str(getattr(episode, key))].append(episode)
        return {
            name: {
                "entries": len(items),
                "mean_net_M": _mean(
                    [float(item.net_M) for item in items if item.net_M is not None]
                ),
                "mean_net_R": _mean(
                    [float(item.net_R) for item in items if item.net_R is not None]
                ),
            }
            for name, items in sorted(grouped.items())
        }

    return {
        "strategy_version": "m1c_directionless_t20_d_v0",
        "eligible_hard_events": len(episodes),
        "valid_path_events": len(episodes) - len(unresolved),
        "entry_count": len(entered),
        "entry_coverage": None if not episodes else len(entered) / len(episodes),
        "no_entry_count": sum(episode.state.value == "NO_ENTRY_TIMEOUT" for episode in episodes),
        "unresolved_count": len(unresolved),
        "long_count": sum(episode.direction == "LONG" for episode in entered),
        "short_count": sum(episode.direction == "SHORT" for episode in entered),
        "median_entry_delay_seconds": _median(
            [
                (episode.entry_timestamp - episode.t0).total_seconds()
                for episode in entered
                if episode.entry_timestamp is not None
            ]
        ),
        "target_count": sum(episode.state.value == "TARGET_EXITED" for episode in clean),
        "stop_count": sum(episode.state.value == "STOP_EXITED" for episode in clean),
        "time_exit_count": sum(episode.state.value == "TIME_EXITED" for episode in clean),
        "mean_net_M": _mean(net_m),
        "median_net_M": _median(net_m),
        "mean_net_R": _mean(net_r),
        "median_net_R": _median(net_r),
        "total_net_R": sum(net_r),
        "positive_net_rate": None
        if not net_m
        else sum(value > 0.0 for value in net_m) / len(net_m),
        "median_MFE_M": _median([episode.MFE_M for episode in clean]),
        "median_MAE_M": _median([episode.MAE_M for episode in clean]),
        "whipsaw_1m": sum(episode.opposite_within_1m for episode in clean),
        "whipsaw_5m": sum(episode.opposite_within_5m for episode in clean),
        "whipsaw_horizon": sum(episode.opposite_before_horizon for episode in clean),
        "session_clustered_95ci_mean_net_M": _clustered_interval(clean, "net_M"),
        "session_clustered_95ci_mean_net_R": _clustered_interval(clean, "net_R"),
        "by_session": stability("session"),
        "by_symbol": stability("symbol"),
        "parameter_comparisons": False,
    }


def require_analysis_open_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("directionless shadow analysis receipt must be a JSON object")
    if (
        payload.get("analysis_opened") is not True
        or payload.get("strategy_version") != "m1c_directionless_t20_d_v0"
        or int(payload.get("complete_eligible_sessions", 0)) != 20
    ):
        raise ValueError("directionless shadow V0 analysis period is not formally open")
    return cast(dict[str, Any], payload)


__all__ = ["analyse_directionless_shadow_v0", "require_analysis_open_receipt"]
