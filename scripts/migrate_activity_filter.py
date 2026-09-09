"""Create V2 PAPER runs while retaining immutable V1 runs as archived history.

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
from stocker_core.methods import SESSION_HARD
from stocker_dashboard.universe_runs import UniverseRunBuilder

PREVIOUS_VERSION = "SESSION_HARD_CAUSAL_Q1_FIT_V1"


def migrate(payload: dict[str, Any]) -> tuple[RunsConfig, dict[str, str]]:
    updated = copy.deepcopy(payload)
    enabled: dict[str, bool] = {}
    for row in updated.get("runs", []):
        if (
            row.get("strategy_id") == SESSION_HARD.method_id
            and row.get("strategy_version") == PREVIOUS_VERSION
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
        config, current = UniverseRunBuilder().add(
            config,
            market_id=previous.market_id,
            strategy_id=SESSION_HARD.method_id,
            strategy_version=SESSION_HARD.version,
            environment=previous.environment,
            risk=previous.risk,
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
