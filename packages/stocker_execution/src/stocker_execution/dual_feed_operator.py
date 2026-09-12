"""Finite PAPER/data-session operator. No trading runtime or execution service."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import sqlite3
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from stocker_core.candidate_selection import CandidateIdentity
from stocker_core.config import load_ibkr_config, load_runs_config
from stocker_core.methods import SESSION_HARD
from stocker_core.runs import Environment, RunConfig
from stocker_execution.acquisition_store import encoded
from stocker_execution.candidate_pipeline import instrument
from stocker_execution.dual_feed import ORDINARY, REFERENCE, DualFeedRecorder, StreamEvidence
from stocker_execution.dual_feed_comparison import CRITERIA, compare_pair, verdict
from stocker_execution.history import IbkrHistoryCache
from stocker_execution.ibkr import IbkrConnection, QualifiedInstrument, mask_ibkr_account
from stocker_execution.runtime import ExchangeSessionResolver, _signal_payload
from stocker_execution.session_hard_data import (
    IbkrSessionDataSource,
    PriorSessionExpectedMoveService,
)
from stocker_execution.session_hard_method import SessionHardMethod
from stocker_execution.session_hard_structure_d import CohortOpportunity, StrategySignal
from stocker_execution.stage5 import (
    STAGE5_HV_CALCULATION_VERSION,
    Stage5Analyzer,
    Stage5CurrentDataService,
    Stage5Membership,
    Stage5QualifiedRequest,
    Stage5Status,
)


def context_json(value: Any) -> Any:
    """Retain nonfinite model inputs explicitly in diagnostic JSON, never impute."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {k: context_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [context_json(v) for v in value]
    return value


def block_order_methods(broker: IbkrConnection, attempts: list[str]) -> None:
    """Additional diagnostic-only tripwire on the exclusively owned SDK instance."""

    def blocked(*args: Any, **kwargs: Any) -> Any:
        attempts.append("ORDER_METHOD_BLOCKED")
        raise RuntimeError("Order/what-if method forbidden in dual-feed diagnostic")

    client: Any = broker._client
    client.placeOrder = blocked
    client.whatIfOrderAsync = blocked
    client.client.placeOrder = blocked


def frozen_selection(
    database: Path, run: RunConfig, session: date, *, now: datetime
) -> tuple[tuple[QualifiedInstrument, ...], tuple[CohortOpportunity, ...]]:
    """Read one frozen RV15 selection and cohort in a SQLite read-only transaction."""
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        row = db.execute(
            "SELECT state,spec_hash FROM opening_candidate_sessions WHERE run_id=? AND session=?",
            (run.run_id, session.isoformat()),
        ).fetchone()
        if row is None or row[0] != "SESSION_HARD_ACTIVE" or row[1] != run.method_spec_hash:
            raise ValueError("Frozen current-session selection unavailable or spec mismatch")
        rows = db.execute(
            "SELECT p.identity,s.selected_at FROM opening_candidate_population p "
            "JOIN opening_candidate_stages s ON p.run_id=s.run_id AND p.session=s.session "
            "AND p.con_id=s.con_id WHERE p.run_id=? AND p.session=? AND s.stage=2 "
            "AND s.selected=1 ORDER BY p.con_id",
            (run.run_id, session.isoformat()),
        ).fetchall()
        if not 1 <= len(rows) <= 30:
            raise ValueError("Expected the unchanged frozen population of at most 30")
        if any(datetime.fromisoformat(r[1]) > now for r in rows):
            raise ValueError("Selection was not available prospectively")
        selected = tuple(instrument(CandidateIdentity(**json.loads(r[0]))) for r in rows)
        cohort = tuple(
            CohortOpportunity(str(r[0]), date.fromisoformat(r[1]), float(r[2]))
            for r in db.execute(
                "SELECT run_id,session,pre_move_m FROM runtime_cohort_opportunities "
                "WHERE run_id=? AND session<? ORDER BY session,identity",
                (run.run_id, session.isoformat()),
            ).fetchall()
        )
    return selected, cohort


async def _until(when: datetime) -> None:
    while (remaining := (when - datetime.now(UTC)).total_seconds()) > 0:
        await asyncio.sleep(min(0.25, remaining))


async def prepare_contexts(
    broker: IbkrConnection,
    run: RunConfig,
    selected: Sequence[QualifiedInstrument],
    cohort: Sequence[CohortOpportunity],
    t0: datetime,
    session: date,
    checkpoint: int,
    output: Path,
    contexts: dict[int, StrategySignal],
) -> None:
    """Reuse exact IBKR feature/context producers without starting or enabling a run."""
    cache = IbkrHistoryCache(output / "diagnostic-history.sqlite3")
    source = IbkrSessionDataSource(broker, cache)
    features = Stage5Analyzer(
        Stage5CurrentDataService(
            broker,
            cache,
            PriorSessionExpectedMoveService(broker, cache),
            calculation_version=STAGE5_HV_CALCULATION_VERSION,
        ),
        calculation_version=STAGE5_HV_CALCULATION_VERSION,
    )
    requests = tuple(
        Stage5QualifiedRequest(i, (Stage5Membership(run.run_id, run.universe),)) for i in selected
    )
    await features.prepare_expected_moves(requests, session=session, t0=t0)
    await _until(t0)
    # Same production four-stock batch seam. Capture once before comparing tapes.
    for offset in range(0, len(requests), 4):
        rows = await features.analyze(requests[offset : offset + 4], session=session, t0=t0)
        context = await source.context_for(
            run,
            rows,
            checkpoint,
            {i.con_id: i for i in selected},
            cohort,
        )
        assert run.market_id is not None and run.strategy_version is not None
        method = SessionHardMethod(run.market_id, method_version=run.strategy_version)
        for row in rows:
            if row.status is not Stage5Status.READY or row.con_id not in (
                context.required_history_ready or ()
            ):
                continue
            initial = method.evaluate((row,), context)[0]
            if datetime.now(UTC) >= t0 + timedelta(minutes=5):
                return
            assert row.con_id is not None
            contexts[row.con_id] = initial
            # Retain feature inputs, cohort and post-qualification state. Never
            # save diagnostic signals into RuntimeStore or a production DB.
            (output / f"context-{row.con_id}.json").write_text(
                encoded(
                    context_json(
                        {
                            "research_only": True,
                            "frozen_at": datetime.now(UTC),
                            "feature": asdict(row),
                            "cohort": [asdict(c) for c in cohort],
                            "assessment": [
                                {"key": asdict(k), "value": asdict(v)}
                                for k, v in context.session_hard.items()
                            ],
                            "whipsaw_features": [
                                {"key": asdict(k), "value": v}
                                for k, v in context.whipsaw_features.items()
                            ],
                            "available_at": context.available_at,
                            "initial": _signal_payload(initial),
                        }
                    )
                )
            )


def export_report(output: Path, report: dict[str, Any], recorder: DualFeedRecorder | None) -> None:
    (output / "comparison.json").write_text(encoded(report))
    if recorder:
        with (output / "events.csv").open("w", newline="") as handle:
            names = (
                list(asdict(recorder.events[0]))
                if recorder.events
                else [
                    "con_id",
                    "symbol",
                    "feed",
                    "event_at",
                    "received_at",
                    "price",
                    "size",
                    "sequence",
                ]
            )
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            for event in recorder.events:
                row = asdict(event)
                row["raw"] = encoded(row["raw"])
                writer.writerow(row)
    lines = [
        "# Stocker dual-feed diagnostic",
        "",
        "Research only. Order placement disabled. "
        f"Order method attempts: {report.get('order_methods_invoked', 0)}.",
        "",
        f"Release: `{report['release_sha']}`. "
        f"Environment: PAPER / {report.get('account', 'unconnected')}.",
        f"Session: {report['session']}; T0: {report['t0']}.",
        "Ordinary TOP30 simultaneously observed: "
        f"**{report.get('ordinary_top30_observed', False)}**.",
        "",
        "| conId | TBT prints | ordinary prints | replay | strict |",
        "|---|---:|---:|---|---|",
    ]
    for pair in report.get("pairs", []):
        lines.append(
            f"| {pair['con_id']} | {pair['reference']['event_count']} | "
            f"{pair['ordinary']['event_count']} | "
            f"{pair['classification']} | {pair['strict_pass']} |"
        )
    lines += [
        "",
        "Full resource counters, errors, timing/price/order metrics and replay details "
        "are in comparison.json.",
        "CSV contains raw provenance; repeated observations have not been removed.",
        "",
        "One small session cannot validate production substitution. Keep the true TBT source.",
        "",
        report["verdict"],
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def validate_operator(args: argparse.Namespace) -> tuple[RunConfig, Any, datetime, int]:
    if not args.confirm_dedicated_paper_gateway:
        raise ValueError("A dedicated PAPER/data Gateway with all trading runs paused is required")
    runs = load_runs_config(args.runs_config)
    if any(r.enabled for r in runs.runs):
        raise ValueError(
            "All saved runs must remain paused; diagnostic cannot enable or pause them"
        )
    run = next(r for r in runs.runs if r.run_id == args.run_id)
    if run.environment is not Environment.PAPER or run.strategy_id != SESSION_HARD.method_id:
        raise ValueError("PAPER Session HARD run required")
    config = load_ibkr_config(args.ibkr_config, Environment.PAPER)
    if config.port in {4001, 7496}:
        raise ValueError("Refusing standard LIVE API ports")
    config = type(config).model_validate(config.model_dump() | {"client_id": args.client_id})
    t0 = datetime.fromisoformat(args.t0)
    now = datetime.now(UTC)
    if t0.tzinfo is None or not timedelta(0) < t0 - now <= timedelta(minutes=30):
        raise ValueError("T0 must be an aware, future checkpoint within 30 minutes")
    market = ExchangeSessionResolver().resolve(run, t0)
    assert run.method_spec is not None
    schedule = market.checkpoint_times(tuple(run.method_spec["qualification"]["checkpoints"]))
    checkpoint = next((n for n, at in schedule if at == t0), None)
    if checkpoint is None:
        raise ValueError("T0 is not an unchanged method checkpoint in an open session")
    return run, config, t0, checkpoint


async def observe(args: argparse.Namespace) -> Path:
    run, config, t0, checkpoint = validate_operator(args)
    market = ExchangeSessionResolver().resolve(run, t0)
    selected, cohort = frozen_selection(args.database, run, market.session, now=datetime.now(UTC))
    # Fixed conId order is sample assignment only. Do not change the TOP30, rotate
    # on rejection, or choose a replacement when an attempted reference fails.
    paired = selected[: min(5, max(1, config.market_data_line_budget // 20))]
    ordinary = selected if args.ordinary_top30 else paired
    output = Path(args.output) / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    criteria_bytes = encoded(CRITERIA).encode()
    (output / "acceptance-criteria.json").write_bytes(criteria_bytes)
    release = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    report: dict[str, Any] = {
        "research_only": True,
        "order_placement": "disabled",
        "order_methods_invoked": 0,
        "release_sha": release,
        "dirty": dirty,
        "session": str(market.session),
        "t0": t0,
        "criteria_sha256": hashlib.sha256(criteria_bytes).hexdigest(),
        "reference_instruments": [asdict(i) for i in paired],
        "frozen_selected": [asdict(i) for i in selected],
        "ordinary_top30_requested": len(ordinary),
        "reference_requested": len(paired),
        "errors": [],
        "context_errors": [],
        "pairs": [],
        "verdict": "DUAL_FEED_DIAGNOSTIC_NOT_RUN",
    }
    broker = IbkrConnection(config, execution_enabled=False)
    order_attempts: list[str] = []
    block_order_methods(broker, order_attempts)
    recorder = None
    contexts: dict[int, StrategySignal] = {}
    preparation = None
    started = datetime.now(UTC)
    try:
        await broker.connect()
        report["account"] = mask_ibkr_account(broker.account)
        report["resources_before"] = asdict(broker.resource_status())
        broker._client.reqMarketDataType(1)
        recorder = DualFeedRecorder(broker, max_events=args.max_events)
        for stock in paired:
            try:
                broker.prepare_trade_events(stock)
                recorder.attach_reference(stock)
            except Exception as exc:
                recorder.streams[(stock.con_id, REFERENCE)] = StreamEvidence(
                    stock.con_id,
                    stock.symbol,
                    REFERENCE,
                    datetime.now(UTC),
                    rejection=str(exc),
                )
        for stock in ordinary:
            recorder.acquire_ordinary(stock)
        report["resources_peak"] = asdict(broker.resource_status())
        preparation = asyncio.create_task(
            prepare_contexts(
                broker,
                run,
                paired,
                cohort,
                t0,
                market.session,
                checkpoint,
                output,
                contexts,
            )
        )
        # Fixed half-open method window; grace captures late arrivals descriptively.
        finish = t0 + timedelta(minutes=5, seconds=2)
        while datetime.now(UTC) < finish:
            if not broker.is_connected or broker.connection_epoch != recorder.epoch:
                raise ValueError("CONNECTION_LOST_NO_RECONNECT_OR_PREFIX_REPAIR")
            if recorder.dropped_events:
                raise ValueError("BOUNDED_TAPE_LIMIT_REACHED")
            if any(r.enabled for r in load_runs_config(args.runs_config).runs):
                raise ValueError("RUN_ENABLED_EXTERNALLY_DIAGNOSTIC_STOPPED")
            await asyncio.sleep(0.25)
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}:{exc}")
    finally:
        if preparation:
            if not preparation.done():
                preparation.cancel()
            try:
                await preparation
            except (Exception, asyncio.CancelledError) as exc:
                report["context_errors"].append(f"CONTEXT_PREPARATION:{type(exc).__name__}")
        ended = datetime.now(UTC)
        if recorder:
            report["reference_readiness_before_release"] = {
                str(i.con_id): broker.trade_stream_status(i, t0=t0) for i in paired
            }
            recorder.close()
            report["streams"] = [asdict(s) for s in recorder.streams.values()]
            report["dropped_events"] = recorder.dropped_events
            report["integrity_errors"] = recorder.integrity_errors
            report["resources_before_disconnect"] = asdict(broker.resource_status())
        # This operator alone owns the dedicated connection/TBT. Even a broker
        # cancellation error must leave a report and close the socket.
        try:
            broker.disconnect()
        except Exception as exc:
            report["errors"].append(f"DISCONNECT_FAILED:{type(exc).__name__}")
            broker._client.disconnect()
        report["resources_after_release"] = asdict(broker.resource_status())
        report.update(started_at=started, ended_at=ended, reference_physical_release_at=ended)
        report["order_methods_invoked"] = len(order_attempts)
        if order_attempts:
            report["errors"].append("ORDER_METHOD_ATTEMPT_BLOCKED")
        if recorder:
            failures = [*report["errors"], *recorder.integrity_errors]
            if recorder.dropped_events:
                failures.append("TAPE_OVERFLOW")
            report["pairs"] = [
                compare_pair(
                    i.con_id,
                    recorder.events,
                    tuple(recorder.streams.values()),
                    t0=t0,
                    end=ended,
                    initial=contexts.get(i.con_id),
                    integrity_errors=failures,
                )
                for i in paired
            ]
            for feed in (REFERENCE, ORDINARY):
                streams = [s for s in recorder.streams.values() if s.feed == feed]
                report[feed] = {
                    "requested": len(streams),
                    "subscribed": sum(s.subscribed_at is not None for s in streams),
                    "rejected": sum(bool(s.rejection or s.errors) for s in streams),
                    "released": sum(s.released_at is not None for s in streams),
                    "valid_print_counts": {
                        str(s.con_id): sum(
                            e.feed == feed
                            and e.con_id == s.con_id
                            and not e.invalid_reason
                            and e.event_at is not None
                            and t0 <= e.event_at < t0 + timedelta(minutes=5)
                            for e in recorder.events
                        )
                        for s in streams
                    },
                }
                report[feed]["succeeded"] = sum(
                    bool(report[feed]["valid_print_counts"][str(s.con_id)])
                    and not s.rejection
                    and not s.errors
                    for s in streams
                )
            ordinary_streams = [s for s in recorder.streams.values() if s.feed == ORDINARY]
            report["ordinary_top30_observed"] = (
                len(ordinary_streams) == 30
                and not failures
                and not recorder.dropped_events
                and all(
                    s.subscribed_at
                    and s.recording_started_at is not None
                    and s.recording_started_at <= t0
                    and not s.errors
                    and any(
                        e.con_id == s.con_id
                        and e.feed == ORDINARY
                        and not e.invalid_reason
                        and e.market_data_type == 1
                        and e.event_at is not None
                        and t0 <= e.event_at < t0 + timedelta(minutes=5)
                        for e in recorder.events
                    )
                    for s in ordinary_streams
                )
            )
            report["verdict"] = verdict(report["pairs"])
        export_report(output, report, recorder)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-config", type=Path, required=True)
    parser.add_argument("--ibkr-config", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--t0", required=True, help="Exact future method checkpoint with timezone")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--ordinary-top30", action="store_true")
    parser.add_argument("--max-events", type=int, default=250_000)
    parser.add_argument("--output", type=Path, default=Path(".stocker/dual-feed"))
    parser.add_argument("--confirm-dedicated-paper-gateway", action="store_true")
    print(asyncio.run(observe(parser.parse_args())))
