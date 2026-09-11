"""Opt-in read-only PAPER Gateway acquisition benchmark; never creates an execution service."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from stocker_core.acquisition import ACQUISITION_EXPERIMENT_V1, AcquisitionRecipe, AcquisitionScan
from stocker_core.config import load_ibkr_config, load_runs_config
from stocker_core.markets import MARKET_CATALOGUE, MarketId, get_market
from stocker_core.methods import SESSION_HARD, content_hash
from stocker_core.runs import Environment, RunInstance, RunState
from stocker_execution.acquired_candidates import AcquiredCandidates
from stocker_execution.acquisition_store import AcquisitionStore, encoded
from stocker_execution.candidate_pipeline import CandidateStore
from stocker_execution.history import IbkrHistoryCache
from stocker_execution.ibkr import IbkrConnection
from stocker_execution.runtime import ExchangeSessionResolver
from stocker_execution.scanner_acquisition import acquisition_scans


async def inspect_scanners(args: argparse.Namespace) -> None:
    """Finite read-only diagnostics; no run, historical queue, or strategy is constructed."""
    config = load_ibkr_config(args.ibkr_config, Environment.PAPER)
    config = type(config).model_validate(config.model_dump() | {"client_id": args.client_id})
    broker = IbkrConnection(config, execution_enabled=False)
    store = AcquisitionStore(args.state)
    markets = [get_market(m) for m in args.markets] if args.markets else list(MARKET_CATALOGUE)
    report: dict[str, Any] = {
        "mode": "SCANNER_ACCESS_CHECK" if args.scanner_check else "CAPABILITIES_ONLY",
        "evidence": "ACCESS_DIAGNOSTIC_ONLY_NOT_OPENING_RECALL",
        "subscription_status": "Not inferred from advertised scanner capabilities",
        "started_at": datetime.now(UTC).isoformat(),
        "markets": [],
    }
    recipe = (AcquisitionRecipe(
        recipe_id="SCANNER_ACCESS_DIAGNOSTIC_EXPLICIT_CODES",
        evaluation_period="ACCESS_CHECK_ONLY", families=tuple(args.scan_codes),
    ) if args.scan_codes else ACQUISITION_EXPERIMENT_V1)
    report["recipe"] = recipe.model_dump(mode="json")
    output = args.state.with_suffix(".scanner-check.json")
    try:
        await broker.connect()
        capabilities = await broker.scanner_capabilities()
        report.update(
            capabilities_digest=store.save_capabilities(asdict(capabilities)),
            gateway_api_version=capabilities.server_version,
            codes=sorted(capabilities.scan_codes), locations=sorted(capabilities.locations),
            filters=sorted(capabilities.filters),
            descriptions=capabilities.scan_descriptions,
        )
        output.write_text(encoded(report))
        print(json.dumps({k:report[k] for k in
              ("mode", "capabilities_digest", "gateway_api_version", "codes")}), flush=True)
        semaphore = asyncio.Semaphore(2)
        shared: dict[tuple[Any, ...], dict[str, Any]] = {}
        fx_cache: dict[str, tuple[float | None, str]] = {}
        for market in markets:
            fx, fx_error = None, ""
            if args.scanner_check and market.currency != "USD":
                if market.currency not in fx_cache:
                    try:
                        quote = await broker.discovery_fx(market.currency)
                        fx_cache[market.currency] = (quote.local_per_usd, "")
                    except Exception as exc:
                        fx_cache[market.currency] = (None, str(exc))
                fx, fx_error = fx_cache[market.currency]
            plans = acquisition_scans(recipe, market, capabilities, fx)
            result: dict[str, Any] = {
                "market": market.market_id.value, "location": market.scanner_location,
                "fx_local_per_usd": fx, "fx_error": fx_error,
                "components": [asdict(plan) for plan in plans],
            }
            report["markets"].append(result)
            if args.scanner_check:
                async def check(plan: AcquisitionScan, row: dict[str, Any]) -> None:
                    if plan.unsupported_reason:
                        row["status"] = "UNSUPPORTED"
                        return
                    key = (plan.location, plan.instrument, plan.scan_code, plan.rows, plan.filters)
                    async with semaphore:
                        if key in shared:
                            row.update(shared[key], shared_request=True)
                            return
                        audit: dict[str, Any] = {}
                        try:
                            rows = await broker.acquisition_scan(plan, audit)
                            observed = {"status": "COMPLETE", "audit": audit,
                                        "hits": [asdict(hit) for hit in rows]}
                        except Exception as exc:
                            observed = {"status": "FAILED", "audit": audit, "error": str(exc)}
                        shared[key] = observed
                        row.update(observed)
                await asyncio.gather(*(
                    check(p, r) for p, r in zip(plans, result["components"], strict=True)
                ))
            output.write_text(encoded(report))
            print(json.dumps({
                "market": result["market"], "location": result["location"],
                "supported_components": sum(not p.unsupported_reason for p in plans),
                "complete": sum(r.get("status") == "COMPLETE" for r in result["components"]),
                "failed": sum(r.get("status") == "FAILED" for r in result["components"]),
                "unsupported": sorted({p.unsupported_reason for p in plans
                                       if p.unsupported_reason}),
                "unique_raw_conids": len({h["con_id"] for r in result["components"]
                                          for h in r.get("hits", [])}),
            }), flush=True)
    finally:
        broker.disconnect()
        report["finished_at"] = datetime.now(UTC).isoformat()
        output.write_text(encoded(report))
        print(json.dumps({"report": str(output), "state": str(args.state)}), flush=True)


async def benchmark(args: argparse.Namespace) -> None:
    if args.capabilities_only or args.scanner_check:
        await inspect_scanners(args)
        return
    if args.scan_codes or args.markets:
        raise ValueError("--scan-codes and --markets are inspection-only options")
    if not all((args.runs_config, args.run_id, args.history_cache)):
        raise ValueError("Opening benchmark requires --runs-config, --run-id and --history-cache")
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
    parser.add_argument("--runs-config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True, help="Dedicated benchmark database")
    parser.add_argument(
        "--history-cache", type=Path, help="IBKR-only historical cache"
    )
    parser.add_argument(
        "--recipe", type=Path, help="Predeclared versioned experimental matrix JSON"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--capabilities-only", action="store_true")
    modes.add_argument("--scanner-check", action="store_true",
                       help="One bounded snapshot of advertised components; no history or strategy")
    parser.add_argument("--markets", nargs="+", choices=[m.value for m in MarketId],
                        help="Inspection markets; defaults to the supported market catalogue")
    parser.add_argument("--scan-codes", nargs="+",
                        help="Explicit advertised codes for diagnostics only; never changes V1")
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
