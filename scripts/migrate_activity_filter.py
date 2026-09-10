"""Create current activity-filter PAPER runs and archive the previous version.

Writes a separate config file; activation and database backup remain deployment steps.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from stocker_core.config import RunsConfig
from stocker_core.markets import get_market
from stocker_core.methods import SESSION_HARD
from stocker_dashboard.universe_runs import UniverseRunBuilder

PREVIOUS_VERSION = "SESSION_HARD_CAUSAL_Q1_DISCOVERY_V6"
PREVIOUS_VERSIONS = {
    PREVIOUS_VERSION, "SESSION_HARD_CAUSAL_Q1_ACTIVITY_V5", "SESSION_HARD_CAUSAL_Q1_ACTIVITY_V4",
}


def migrate(payload: dict[str, Any]) -> tuple[RunsConfig, dict[str, str]]:
    updated = copy.deepcopy(payload)
    enabled: dict[str, bool] = {}
    for row in updated.get("runs", []):
        if (
            row.get("strategy_id") == SESSION_HARD.method_id
            and row.get("strategy_version") in PREVIOUS_VERSIONS
            and not row.get("archived", False)
        ):
            if row.get("environment") != "PAPER":
                raise ValueError("Activity filter migration is PAPER-only")
            enabled[row["run_id"]] = row.get("enabled", True)
            row.update(enabled=False, archived=True)
    config = RunsConfig.model_validate(updated)
    mapping = {}
    for previous in tuple(config.runs):
        if previous.run_id not in enabled:
            continue
        assert previous.market_id is not None
        original_universes = config.universes
        listing_id = get_market(previous.market_id).listing_membership
        if listing_id is not None and SESSION_HARD.discovery_profile(previous.market_id) is None:
            snapshot = previous.universe_snapshot
            if snapshot is None or not snapshot.members:
                raise ValueError("Migration requires the previous run's saved listing membership")
            # Reuse the exact saved population, even if the catalogue was removed or refreshed.
            config = config.model_copy(
                update={
                    "universes": (
                        *(u for u in config.universes if u.universe_id != listing_id),
                        snapshot.model_copy(update={"universe_id": listing_id}),
                    )
                }
            )
        config, current = UniverseRunBuilder().add(
            config,
            market_id=previous.market_id,
            strategy_id=SESSION_HARD.method_id,
            strategy_version=SESSION_HARD.version,
            environment=previous.environment,
            risk=previous.risk,
        )
        if previous.discovery_profile is not None and current.discovery_profile is not None:
            # Carry forward operational budgets/minima, keeping the new method-owned policy.
            profile = current.discovery_profile.model_copy(update={
                field: getattr(previous.discovery_profile, field)
                for field in (
                    "results_per_band", "merged_candidate_limit", "monitoring_limit",
                    "scan_concurrency", "minimum_price", "minimum_volume",
                    "minimum_average_volume",
                )
            })
            current = current.model_copy(update={"discovery_profile": profile})
            config = config.model_copy(update={
                "runs": tuple(current if r.run_id == current.run_id else r for r in config.runs),
            })
        if listing_id is not None and SESSION_HARD.discovery_profile(previous.market_id) is None:
            config = config.model_copy(
                update={
                    "universes": (
                        *(u for u in config.universes if u.universe_id != listing_id),
                        *(u for u in original_universes if u.universe_id == listing_id),
                    )
                }
            )
        if not enabled[previous.run_id]:
            config = UniverseRunBuilder().disable(config, current.run_id)
        mapping[previous.run_id] = current.run_id
    return RunsConfig.model_validate(config.model_dump()), mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.runs_config.resolve() == args.output.resolve():
        parser.error("Output must be separate from the active config")
    config, mapping = migrate(yaml.safe_load(args.runs_config.read_text()))
    with args.output.open("x") as output:
        yaml.safe_dump(config.model_dump(mode="json"), output, sort_keys=False)
    print(json.dumps({"output": str(args.output), "runs": mapping}, indent=2))


if __name__ == "__main__":
    main()
