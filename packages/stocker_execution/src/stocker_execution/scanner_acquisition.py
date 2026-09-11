"""Causal scanner-union acquisition. Scanner rank never enters candidate mathematics."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from stocker_core.acquisition import SHADOW_RECIPES, AcquisitionRecipe, AcquisitionScan, cap_bounds
from stocker_core.candidate_selection import CandidateIdentity
from stocker_core.markets import MarketDefinition, get_market
from stocker_core.runs import RunInstance
from stocker_execution.acquisition_store import AcquisitionStore
from stocker_execution.activity_shortlist import ScannerCapabilities
from stocker_execution.discovery import DiscoveryRow
from stocker_execution.ibkr import IbkrConnection, IbkrInstrumentUnavailable


def acquisition_scans(
    recipe: AcquisitionRecipe,
    market: MarketDefinition,
    capabilities: ScannerCapabilities,
    local_per_usd: float | None = None,
) -> tuple[AcquisitionScan, ...]:
    available = capabilities.scan_codes_for(market.scanner_location)
    filters = capabilities.filters_for(market.scanner_location)
    requests = []
    for family in recipe.families:
        # Exact since-open codes observed in Gateway 178. Validate against this
        # connection/location before requesting; overnight OPEN_GAP is not equivalent.
        requested_code = {
            "OPENING_PERCENT_GAIN": "TOP_OPEN_PERC_GAIN",
            "OPENING_PERCENT_LOSS": "TOP_OPEN_PERC_LOSE",
        }.get(family, family)
        code = requested_code if requested_code in available else ""
        for cap in recipe.cap_slices:
            reason = ""
            configured_filters = []
            if not code:
                reason = f"SCANNER_CODE_UNAVAILABLE_OR_AMBIGUOUS: {family}"
            if (
                market.scanner_location not in capabilities.locations
                or not capabilities.supports_instrument(
                    market.scanner_location, market.scanner_instrument
                )
            ):
                reason = f"SCANNER_LOCATION_OR_INSTRUMENT_UNSUPPORTED: {market.scanner_location}"
            lower, upper = cap_bounds(cap)
            for boundary, names in (
                (lower, ("marketCapAbove", "marketCapAbove1e6")),
                (upper, ("marketCapBelow", "marketCapBelow1e6")),
            ):
                if boundary is None:
                    continue
                field = next((name for name in names if name in filters), None)
                if field is None:
                    reason = f"SCANNER_FILTER_UNSUPPORTED: {'/'.join(names)}"
                elif market.currency != "USD" and local_per_usd is None:
                    reason = "SCANNER_CAP_FX_UNAVAILABLE"
                else:
                    configured_filters.append((field, str(boundary * (local_per_usd or 1) / 1e6)))
            requests.append(
                AcquisitionScan(
                    family + ":" + cap,
                    family,
                    cap,
                    market.scanner_location,
                    market.scanner_instrument,
                    code,
                    recipe.rows_per_component,
                    tuple(configured_filters),
                    reason,
                )
            )
    return tuple(requests)


class ScannerAcquisition:
    def __init__(
        self,
        broker: IbkrConnection,
        store: AcquisitionStore,
        clock: Callable[[], datetime],
        wait_until: Callable[[datetime], Awaitable[None]] | None = None,
        experiment: AcquisitionRecipe | None = None,
    ):
        self.broker, self.store, self.clock = broker, store, clock
        self.wait_until = wait_until or self._wait_until
        self.experiment = experiment
        self._sweep_results: dict[tuple[Any, ...], tuple[DiscoveryRow, ...]] = {}
        self._sweep_locks: dict[tuple[Any, ...], asyncio.Lock] = {}

    async def _wait_until(self, due: datetime) -> None:
        await asyncio.sleep(max(0, (due - self.clock()).total_seconds()))

    async def acquire(
        self, instance: RunInstance
    ) -> tuple[tuple[CandidateIdentity, ...], list[dict[str, Any]]]:
        from stocker_execution.runtime import ExchangeSessionResolver

        run = instance.config
        assert run.market_id is not None and run.method_spec is not None
        if run.environment.value != "PAPER":
            raise ValueError("Scanner acquisition is PAPER-only")
        recipe = self.experiment or AcquisitionRecipe.model_validate(
            run.method_spec["universe_acquisition"]
        )
        market = get_market(run.market_id)
        session = ExchangeSessionResolver().resolve(run, self.clock())
        prefix = session.minute_prefix(15)
        cutoff = prefix[4] + timedelta(minutes=1)
        metadata = {
            "recipe": recipe.model_dump(mode="json"),
            "shadow_recipes": SHADOW_RECIPES,
            "market": run.market_id.value,
            "prefix": [t.isoformat() for t in prefix],
            "oracle_after": session.closes_at.isoformat() if session.closes_at else None,
            "cutoff": cutoff.isoformat(),
            "method_spec_hash": run.method_spec_hash,
            "upstream_evidence": (
                "PROSPECTIVE_IBKR_TEST"
                if market.country == "US"
                else "UNVALIDATED_CROSS_MARKET_ACQUISITION_TRANSFER"
            ),
        }
        key = run.run_id, session.session
        await asyncio.to_thread(self.store.begin, *key, metadata, instance.universe.members)
        saved = self.store.session(*key)
        assert saved is not None
        if saved["sealed"]:
            return self._result(*key, recipe=recipe)
        fatal = ""
        qualification_tasks: list[asyncio.Task[None]] = []
        try:
            capabilities = await self.broker.scanner_capabilities()
            capability_id = await asyncio.to_thread(
                self.store.save_capabilities, asdict(capabilities)
            )
            fx = None
            fx_error = ""
            if market.currency != "USD":
                try:
                    observed_fx = await self.broker.discovery_fx(market.currency)
                    fx = observed_fx.local_per_usd
                except Exception as exc:
                    fx_error = str(exc)
            plans = acquisition_scans(recipe, market, capabilities, fx)
            refs: dict[tuple[str, str], list[tuple[int, Any]]] = {}
            for i, reference in enumerate(instance.universe.members):
                refs.setdefault((reference.symbol, reference.currency), []).append((i, reference))
            qualified: dict[int, tuple[CandidateIdentity, int] | None] = {}
            qualification_locks: dict[int, asyncio.Lock] = {}

            async def qualify_rows(
                plan: AcquisitionScan,
                sweep: int,
                rows: tuple[DiscoveryRow, ...],
                received: datetime,
                audit: dict[str, Any],
            ) -> None:
                try:
                    async with asyncio.timeout(max(0, (cutoff - self.clock()).total_seconds())):
                        audit["qualification_started_at"] = self.clock().isoformat()
                        # Qualification allocates no price/history streams.
                        rejections = []
                        for row in rows:
                            async with qualification_locks.setdefault(row.con_id, asyncio.Lock()):
                                if row.con_id not in qualified:
                                    qualified[row.con_id] = await self._eligible(
                                        row, refs, run.market_id
                                    )
                            member = qualified[row.con_id]
                            if member is None:
                                rejections.append(row.con_id)
                                continue
                            identity, index = member
                            if self.clock() >= cutoff:
                                raise ValueError("SCANNER_QUALIFICATION_DEADLINE")
                            await asyncio.to_thread(
                                self.store.add_pool,
                                *key,
                                identity,
                                index,
                                sweep,
                                received,
                                row.raw_rank,
                            )
                        audit["rejected_conids"] = rejections
                        if self.clock() >= cutoff:
                            raise ValueError("SCANNER_QUALIFICATION_DEADLINE")
                        audit["qualification_completed_at"] = self.clock().isoformat()
                        await asyncio.to_thread(
                            self.store.component, *key, sweep, plan.component_id, "COMPLETE", audit
                        )
                except asyncio.CancelledError:
                    await asyncio.to_thread(
                        self.store.component,
                        *key,
                        sweep,
                        plan.component_id,
                        "FAILED",
                        audit | {"error": "SCANNER_QUALIFICATION_INTERRUPTED"},
                    )
                    raise
                except Exception as exc:
                    await asyncio.to_thread(
                        self.store.component,
                        *key,
                        sweep,
                        plan.component_id,
                        "FAILED",
                        audit | {"error": str(exc) or type(exc).__name__},
                    )

            self._sweep_results = {
                k: v for k, v in self._sweep_results.items() if k[0] >= session.session
            }
            self._sweep_locks = {
                k: v for k, v in self._sweep_locks.items() if k[0] >= session.session
            }
            for planned_sweep in range(len(recipe.sweep_active_seconds)):
                for plan in plans:
                    await asyncio.to_thread(self.store.plan, *key, planned_sweep, plan)
            for sweep, seconds in enumerate(recipe.sweep_active_seconds):
                due = prefix[seconds // 60] + timedelta(seconds=seconds % 60)
                await self.wait_until(due)
                semaphore = asyncio.Semaphore(recipe.scanner_concurrency)

                async def component(
                    plan: AcquisitionScan,
                    sweep: int = sweep,
                    due: datetime = due,
                    semaphore: asyncio.Semaphore = semaphore,
                ) -> None:
                    record = self.store.plan(*key, sweep, plan)
                    if record["status"] == "COMPLETE":
                        return
                    # Never repeat an interrupted or missed causal scanner observation.
                    audit: dict[str, Any] = {
                        "capability_id": capability_id,
                        "fx_local_per_usd": fx,
                        "fx_error": fx_error,
                        "scheduled_at": due.isoformat(),
                    }
                    if record["status"] != "PENDING":
                        await asyncio.to_thread(
                            self.store.component,
                            *key,
                            sweep,
                            plan.component_id,
                            "FAILED",
                            audit | {"error": "SCANNER_SWEEP_INTERRUPTED"},
                        )
                        return
                    if plan.unsupported_reason:
                        await asyncio.to_thread(
                            self.store.component,
                            *key,
                            sweep,
                            plan.component_id,
                            "FAILED",
                            audit | {"error": plan.unsupported_reason},
                        )
                        return
                    missed = self.clock() > due + timedelta(seconds=recipe.sweep_lateness_seconds)
                    async with semaphore:
                        if self.clock() >= cutoff or missed:
                            await asyncio.to_thread(
                                self.store.component,
                                *key,
                                sweep,
                                plan.component_id,
                                "FAILED",
                                audit | {"error": "SCANNER_SWEEP_WINDOW_MISSED"},
                            )
                            return
                        await asyncio.to_thread(
                            self.store.component, *key, sweep, plan.component_id, "RUNNING", audit
                        )
                        try:
                            async with asyncio.timeout(
                                max(0, (cutoff - self.clock()).total_seconds())
                            ):
                                request_key = (
                                    session.session,
                                    due,
                                    plan.location,
                                    plan.instrument,
                                    plan.scan_code,
                                    plan.rows,
                                    plan.filters,
                                )
                                async with self._sweep_locks.setdefault(
                                    request_key, asyncio.Lock()
                                ):
                                    rows = self._sweep_results.get(request_key)
                                    if rows is None:
                                        rows = await self.broker.acquisition_scan(plan, audit)
                                        self._sweep_results[request_key] = rows
                                    else:
                                        audit["shared_scanner_result"] = True
                                received = self.clock()
                                audit["received_at"] = received.isoformat()
                                await asyncio.to_thread(
                                    self.store.component,
                                    *key,
                                    sweep,
                                    plan.component_id,
                                    "DATA_RECEIVED",
                                    audit,
                                    rows,
                                )
                                if received >= cutoff:
                                    raise ValueError("SCANNER_ACQUISITION_DEADLINE")
                                # Release the scanner slot after scannerDataEnd/cancellation.
                                # Contract checks use the broker's existing bounded ingress;
                                # they must not block remaining scans or later sweep times.
                                qualification_tasks.append(
                                    asyncio.create_task(
                                        qualify_rows(plan, sweep, rows, received, audit)
                                    )
                                )
                        except asyncio.CancelledError:
                            await asyncio.to_thread(
                                self.store.component,
                                *key,
                                sweep,
                                plan.component_id,
                                "FAILED",
                                audit | {"error": "SCANNER_SWEEP_INTERRUPTED"},
                            )
                            raise
                        except Exception as exc:
                            await asyncio.to_thread(
                                self.store.component,
                                *key,
                                sweep,
                                plan.component_id,
                                "FAILED",
                                audit | {"error": str(exc)},
                            )

                await asyncio.gather(*(component(plan) for plan in plans))
            await asyncio.gather(*qualification_tasks)
        except asyncio.CancelledError:
            fatal = "SCANNER_ACQUISITION_INTERRUPTED"
            raise
        except Exception as exc:
            fatal = str(exc)
        finally:
            # No eligibility worker may write into the pool after it is sealed.
            for task in qualification_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*qualification_tasks, return_exceptions=True)
            await asyncio.to_thread(
                self.store.seal,
                *key,
                capacity=recipe.pool_capacity,
                allow_partial=recipe.allow_partial_components,
                reason=fatal,
            )
        return self._result(*key, recipe=recipe)

    def _result(
        self, run_id: str, session: Any, *, recipe: AcquisitionRecipe
    ) -> tuple[tuple[CandidateIdentity, ...], list[dict[str, Any]]]:
        saved = self.store.session(run_id, session)
        assert saved is not None
        failed = saved["state"] == "SCANNER_ACQUISITION_FAILED" or (
            saved["state"] == "SCANNER_ACQUISITION_PARTIAL"
            and not recipe.allow_partial_components
            and saved["reason"] != "ACQUISITION_POOL_CAP_APPLIED"
        )
        return self.store.pool(run_id, session), (
            [{"acquisition_failure": True, "reason": saved["reason"] or saved["state"]}]
            if failed
            else []
        )

    async def _eligible(
        self,
        row: DiscoveryRow,
        references: dict[tuple[str, str], list[tuple[int, Any]]],
        market_id: Any,
    ) -> tuple[CandidateIdentity, int] | None:
        if row.con_id <= 0 or row.raw_rank < 0 or row.security_type != "STK":
            return None
        candidates = references.get((row.symbol, row.currency), ())
        if not candidates:
            return None
        try:
            stock, stock_type = await self.broker.qualify_discovery_candidate(row)
            if (
                stock.con_id != row.con_id
                or stock.security_type != "STK"
                or stock_type not in {"COMMON", "CORP", "ADR", "REIT"}
            ):
                return None
            for index, ref in candidates:
                resolved = await self.broker.resolve_stock(
                    ref.symbol,
                    exchange=ref.exchange,
                    primary_exchange=ref.primary_exchange,
                    currency=ref.currency,
                )
                if resolved.con_id == stock.con_id:
                    # The selected run market is supplied by membership, not inferred from ticker.
                    return CandidateIdentity(
                        stock.con_id,
                        stock.symbol,
                        stock.primary_exchange,
                        stock.primary_exchange or stock.exchange,
                        stock.currency,
                        market_id,
                        stock.security_type,
                        stock.exchange,
                    ), index
        except IbkrInstrumentUnavailable:
            return None
        return None
