"""Opt-in read-only PAPER Gateway acquisition benchmark; never creates an execution service."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

from stocker_core.acquisition import ACQUISITION_EXPERIMENT_V1, AcquisitionRecipe
from stocker_core.config import load_ibkr_config, load_runs_config
from stocker_core.markets import get_market
from stocker_core.methods import SESSION_HARD, content_hash
from stocker_core.runs import Environment, RunInstance, RunState
from stocker_execution.acquired_candidates import AcquiredCandidates
from stocker_execution.candidate_pipeline import CandidateStore
from stocker_execution.history import IbkrHistoryCache
from stocker_execution.ibkr import IbkrConnection
from stocker_execution.runtime import ExchangeSessionResolver
from stocker_execution.scanner_acquisition import acquisition_scans


async def benchmark(args: argparse.Namespace) -> None:
    if not args.confirm_dedicated_paper_gateway:
        raise ValueError(
            "Use a dedicated PAPER Gateway; external engines cannot yield to this benchmark"
        )
    config = load_ibkr_config(args.ibkr_config, Environment.PAPER)
    config = type(config).model_validate(config.model_dump() | {"client_id": args.client_id})
    broker = IbkrConnection(config, execution_enabled=False)
    runs = load_runs_config(args.runs_config)
    saved = next(r for r in runs.runs if r.run_id == args.run_id)
    if saved.environment is not Environment.PAPER or saved.strategy_version != SESSION_HARD.version:
        raise ValueError("Benchmark requires the current PAPER acquisition method configuration")
    recipe = (
        AcquisitionRecipe.model_validate_json(args.recipe.read_text())
        if args.recipe
        else ACQUISITION_EXPERIMENT_V1
    )
    if (
        args.recipe
        and recipe != ACQUISITION_EXPERIMENT_V1
        and recipe.recipe_id == ACQUISITION_EXPERIMENT_V1.recipe_id
    ):
        raise ValueError("Changed experimental matrix requires a new recipe ID")
    # A separate run ID and state file preserve the configured strategy run untouched.
    run = saved.model_copy(
        update={
            "run_id": "acquisition-benchmark-"
            + content_hash({"run": saved.run_id, "recipe": recipe.model_dump(mode="json")})[:20],
            "enabled": True,
        }
    )
    assert run.universe_snapshot is not None and run.market_id is not None
    instance = RunInstance(run, run.universe_snapshot, RunState.ACTIVE)

    def clock() -> datetime:
        return datetime.now(UTC)

    candidates = AcquiredCandidates(
        broker, IbkrHistoryCache(args.history_cache), CandidateStore(args.state), clock
    )
    candidates.provider.experiment = recipe
    session = ExchangeSessionResolver().resolve(run, clock())
    selected_date = args.session or session.session
    try:
        await broker.connect()
        capabilities = await broker.scanner_capabilities()
        digest = candidates.store.save_capabilities(asdict(capabilities))
        print(
            json.dumps(
                {
                    "capabilities_digest": digest,
                    "codes": sorted(capabilities.scan_codes),
                    "locations": sorted(capabilities.locations),
                    "plans": [
                        asdict(p)
                        for p in acquisition_scans(recipe, get_market(run.market_id), capabilities)
                    ],
                },
                default=str,
            )
        )
        if args.capabilities_only:
            return
        candidates.pipeline(instance)
        if args.resume_audit:
            candidates.store.resume_audit(run.run_id, selected_date)
        if not args.audit_only:
            while True:
                await candidates.advance(instance, session, clock())
                summary = candidates.candidate_store.summary(run.run_id, session.session)
                if summary and (summary["ready"] or summary["state"] == "DEGRADED"):
                    print(json.dumps({"candidate_selection": summary}, default=str))
                    break
                await asyncio.sleep(0.25)
        if args.oracle:
            while True:
                await candidates.background(instance, True)
                summary = candidates.store.summary(run.run_id, selected_date)
                if summary is None:
                    raise ValueError("No saved acquisition session to audit")
                if summary["oracle_state"] in {"COMPLETE", "INCOMPLETE"}:
                    break
                await asyncio.sleep(1)
        print(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "summary": candidates.store.summary(run.run_id, selected_date),
                    "benchmark": candidates.store.benchmark(run.run_id, selected_date),
                    "recall_history": candidates.store.recall_history(run.run_id),
                },
                default=str,
            )
        )
    finally:
        await candidates.stop()
        broker.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibkr-config", type=Path, required=True)
    parser.add_argument("--runs-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True, help="Dedicated benchmark database")
    parser.add_argument(
        "--history-cache", type=Path, required=True, help="IBKR-only historical cache"
    )
    parser.add_argument(
        "--recipe", type=Path, help="Predeclared versioned experimental matrix JSON"
    )
    parser.add_argument("--capabilities-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume-audit", action="store_true")
    parser.add_argument("--session", type=date.fromisoformat)
    parser.add_argument(
        "--oracle", action="store_true", help="Wait until session close, then run delayed oracle"
    )
    parser.add_argument("--confirm-dedicated-paper-gateway", action="store_true")
    asyncio.run(benchmark(parser.parse_args()))


if __name__ == "__main__":
    main()
