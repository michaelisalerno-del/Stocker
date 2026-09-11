"""Create disabled current acquisition / Range/RV PAPER replacements in a separate file.

No database changes, broker calls, activation, deployment or history reinterpretation.
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
from stocker_core.universes import NAMED_US_UNIVERSES, load_us_universe_snapshot
from stocker_dashboard.universe_runs import UniverseRunBuilder


def migrate(payload: dict[str, Any]) -> tuple[RunsConfig, dict[str, str]]:
    updated = copy.deepcopy(payload)
    snapshot = updated.pop("named_universe_snapshot", None)
    if snapshot is not None:
        raise ValueError("Resolve named_universe_snapshot with load_payload before migration")
    previous_ids = []
    for row in updated.get("runs", []):
        if (
            row.get("strategy_id") == SESSION_HARD.method_id
            and row.get("strategy_version") != SESSION_HARD.version
            and not row.get("archived", False)
        ):
            if row.get("environment") != "PAPER":
                raise ValueError("Candidate migration is PAPER-only")
            previous_ids.append(row["run_id"])
            row.update(enabled=False, archived=True)
    config = RunsConfig.model_validate(updated)
    mapping = {}
    builder = UniverseRunBuilder()
    for old in tuple(config.runs):
        if old.run_id not in previous_ids:
            continue
        assert old.market_id is not None
        current = next(
            (
                r
                for r in config.runs
                if r.market_id == old.market_id and r.strategy_version == SESSION_HARD.version
            ),
            None,
        )
        if current is not None:
            raise ValueError(
                "Migration would overwrite an existing market run; migrate one run per market"
            )
        config, new = builder.add(
            config,
            market_id=old.market_id,
            strategy_id=SESSION_HARD.method_id,
            strategy_version=SESSION_HARD.version,
            environment=old.environment,
            risk=old.risk,
        )
        config = builder.disable(config, new.run_id)
        mapping[old.run_id] = new.run_id
    return RunsConfig.model_validate(config.model_dump()), mapping


def load_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = yaml.safe_load(path.read_text())
    snapshot = payload.pop("named_universe_snapshot", None)
    if snapshot:
        snapshot_path = Path(snapshot)
        if not snapshot_path.is_absolute():
            snapshot_path = path.parent / snapshot_path
        catalog = load_us_universe_snapshot(snapshot_path)
        existing = payload.setdefault("universes", [])
        by_id = {u["universe_id"]: u for u in existing}
        for name in NAMED_US_UNIVERSES:
            universe = catalog.get_universe(name)
            if not by_id.get(universe.universe_id, {}).get("members"):
                by_id[universe.universe_id] = universe.model_dump(mode="json")
        payload["universes"] = list(by_id.values())
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs_config.resolve() == args.output.resolve():
        parser.error("Output must be separate from the active configuration")
    config, mapping = migrate(load_payload(args.runs_config))
    with args.output.open("x") as stream:
        yaml.safe_dump(config.model_dump(mode="json"), stream, sort_keys=False)
    print(json.dumps({"output": str(args.output), "runs": mapping, "new_runs_enabled": False}))
