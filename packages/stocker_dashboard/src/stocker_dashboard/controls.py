"""Explicit persisted and hot-applied Stage 10 control commands."""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import yaml

from stocker_core.config import IbkrConfig, RunsConfig, load_ibkr_config, load_runs_config
from stocker_core.markets import CapBucket, MarketId
from stocker_core.runs import Environment, RunConfig, RunRiskConfig
from stocker_core.universes import InstrumentReference, UniverseDefinition
from stocker_dashboard.universe_runs import UniverseRunBuilder
from stocker_execution.runtime import RuntimeStatus


class ActiveRuntimeControl(Protocol):
    """Narrow command seam implemented by the Stage 8/9 runtime."""

    def status(self) -> RuntimeStatus: ...

    async def apply_runs_config(
        self, config: RunsConfig, *, changed_run_ids: frozenset[str]
    ) -> RuntimeStatus: ...

    async def replace_broker_config(self, config: IbkrConfig) -> RuntimeStatus: ...


@dataclass(frozen=True, slots=True)
class LiveConfirmation:
    confirmed: bool
    target_account: str


class ApplyMode(StrEnum):
    HOT_APPLY = "HOT_APPLY"
    RESTART_RUN = "RESTART_RUN"
    RECONNECT_ENVIRONMENT = "RECONNECT_ENVIRONMENT"


@dataclass(frozen=True, slots=True)
class ControlResult:
    """Authoritative result of one explicit configuration command."""

    persisted: bool
    runtime_applied: bool
    apply_mode: ApplyMode
    detail: str
    runtime: RuntimeStatus | None = None
    run: RunConfig | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "persisted": self.persisted,
            "runtime_applied": self.runtime_applied,
            "apply_mode": self.apply_mode.value,
            "detail": self.detail,
            "run": self.run.model_dump(mode="json") if self.run else None,
            "runtime": self.runtime.as_dict() if self.runtime else None,
        }


class RunControlService:
    """Serialize, validate, persist, and apply deliberate control operations."""

    def __init__(
        self,
        runs_config_path: str | Path,
        ibkr_config_path: str | Path,
        *,
        runtime: ActiveRuntimeControl | None = None,
    ) -> None:
        self.runs_config_path = Path(runs_config_path)
        self.ibkr_config_path = Path(ibkr_config_path)
        self.runtime = runtime
        self._lock = asyncio.Lock()
        self._builder = UniverseRunBuilder()

    async def universe_builder_options(self) -> dict[str, object]:
        """Return the finite backend-owned builder catalogue."""

        options = self._builder.options(load_runs_config(self.runs_config_path))
        readiness: dict[str, str] = {}
        if self.runtime is not None:
            inspect_readiness = getattr(self.runtime, "activity_scanner_readiness", None)
            if inspect_readiness is not None:
                readiness = await inspect_readiness()
        markets = options["markets"]
        assert isinstance(markets, list)
        for market in markets:
            assert isinstance(market, dict)
            market_id = str(market["market_id"])
            market["scanner_readiness"] = readiness.get(
                market_id,
                "BROKER_NOT_CONNECTED",
            )
        return options

    async def add_universe_run(
        self,
        *,
        market_id: MarketId,
        cap_bucket: CapBucket,
        strategy_id: str,
        strategy_version: str,
        environment: Environment,
        risk_per_trade: float,
        max_concurrent_positions: int | None,
        confirmation: LiveConfirmation | None = None,
    ) -> ControlResult:
        """Create or re-enable one exact market/cap/method/environment lineage."""

        if risk_per_trade <= 0 or risk_per_trade > 1:
            raise ValueError("risk_per_trade must be greater than zero and no more than one")
        if max_concurrent_positions is not None and max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")
        risk = RunRiskConfig(
            risk_per_trade=risk_per_trade,
            max_concurrent_positions=max_concurrent_positions,
        )
        async with self._lock:
            current = load_runs_config(self.runs_config_path)
            updated, run = self._builder.add(
                current,
                market_id=market_id,
                cap_bucket=cap_bucket,
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                environment=environment,
                risk=risk,
            )
            if environment is Environment.LIVE:
                self._confirm_live(run, confirmation, risk=risk)
            if updated == current:
                return ControlResult(
                    True,
                    self.runtime is not None,
                    ApplyMode.HOT_APPLY,
                    "Run already active",
                    self.runtime.status() if self.runtime else None,
                    run,
                )
            if self.runtime is None:
                self._write_runs(updated)
                return ControlResult(
                    True, False, ApplyMode.RESTART_RUN, "Saved; active runtime unavailable", run=run
                )
            runtime_environment = next(
                (
                    item
                    for item in self.runtime.status().execution_environments
                    if item.environment is environment
                ),
                None,
            )
            if runtime_environment is None or not runtime_environment.ready:
                try:
                    await self.runtime.replace_broker_config(
                        load_ibkr_config(self.ibkr_config_path, environment)
                    )
                except Exception as exc:
                    prefix = (
                        "LIVE_NOT_READY" if environment is Environment.LIVE else "BROKER_NOT_READY"
                    )
                    return ControlResult(
                        False,
                        False,
                        ApplyMode.RESTART_RUN,
                        f"{prefix}: {exc}",
                        self.runtime.status(),
                        run,
                    )
            result = await self._apply_runs(updated, {run.run_id}, ApplyMode.RESTART_RUN, run=run)
            if not result.runtime_applied:
                return result
            return await self._persist_runs_after_apply(updated, current, {run.run_id}, result)

    async def enable_run(
        self, run_id: str, *, confirmation: LiveConfirmation | None = None
    ) -> ControlResult:
        async with self._lock:
            config, current = self._config_and_run(run_id)
            if current.environment is Environment.LIVE:
                self._confirm_live(current, confirmation)
            return await self._replace_run(
                config,
                current,
                ApplyMode.HOT_APPLY,
                target_environment=current.environment,
                enabled=True,
            )

    async def disable_run(self, run_id: str) -> ControlResult:
        async with self._lock:
            config, current = self._config_and_run(run_id)
            return await self._replace_run(config, current, ApplyMode.HOT_APPLY, enabled=False)

    async def update_run_config(
        self,
        run_id: str,
        *,
        risk_per_trade: float,
        max_concurrent_positions: int | None,
        universe: str,
        strategy: str,
        confirmation: LiveConfirmation | None = None,
    ) -> ControlResult:
        async with self._lock:
            config, current = self._config_and_run(run_id)
            if risk_per_trade <= 0 or risk_per_trade > 1:
                raise ValueError("risk_per_trade must be greater than zero and no more than one")
            if max_concurrent_positions is not None and max_concurrent_positions <= 0:
                raise ValueError("max_concurrent_positions must be positive")
            if strategy != "SESSION_HARD":
                raise ValueError("active runtime supports only SESSION_HARD strategy config")
            if current.market_id is not None and (
                universe != current.universe or strategy != current.strategy
            ):
                raise ValueError(
                    "market, capitalisation, method, and screen identity must be changed "
                    "by creating a separate run"
                )
            risk = RunRiskConfig(
                risk_per_trade=risk_per_trade,
                max_concurrent_positions=max_concurrent_positions,
            )
            if current.environment is Environment.LIVE:
                self._confirm_live(current, confirmation, risk=risk)
            mode = (
                ApplyMode.RESTART_RUN
                if universe != current.universe or strategy != current.strategy
                else ApplyMode.HOT_APPLY
            )
            return await self._replace_run(
                config,
                current,
                mode,
                universe=universe,
                strategy=strategy,
                risk=risk.model_dump(mode="json"),
            )

    async def change_execution_environment(
        self,
        run_id: str,
        environment: Environment,
        *,
        confirmation: LiveConfirmation | None = None,
    ) -> ControlResult:
        async with self._lock:
            config, current = self._config_and_run(run_id)
            if current.market_id is not None and environment is not current.environment:
                raise ValueError(
                    "execution environment is run identity; create the separate PAPER or LIVE run"
                )
            if environment is Environment.LIVE:
                self._confirm_live(current, confirmation, target=environment)
            return await self._replace_run(
                config,
                current,
                ApplyMode.RESTART_RUN,
                target_environment=environment,
                environment=environment.value,
            )

    async def update_broker_config(self, config: IbkrConfig) -> ControlResult:
        """Persist and reconnect only one explicit broker environment."""

        async with self._lock:
            other_environment = (
                Environment.LIVE if config.environment is Environment.PAPER else Environment.PAPER
            )
            try:
                other = load_ibkr_config(self.ibkr_config_path, other_environment)
            except ValueError:
                other = None
            if other and (config.host, config.port, config.client_id) == (
                other.host,
                other.port,
                other.client_id,
            ):
                raise ValueError("PAPER and LIVE cannot share the same IBKR session identity")
            if not config.expected_account:
                raise ValueError(f"{config.environment.value} expected_account is required")
            raw = self._read_broker_yaml()
            raw[config.environment.value] = config.model_dump(mode="json")
            if self.runtime is None:
                self._write_yaml(self.ibkr_config_path, raw)
                return ControlResult(
                    True,
                    False,
                    ApplyMode.RECONNECT_ENVIRONMENT,
                    "Saved; active runtime unavailable",
                )
            result = await self._apply_broker(config)
            if not result.runtime_applied:
                return result
            previous = load_ibkr_config(self.ibkr_config_path, config.environment)
            return await self._persist_broker_after_apply(raw, previous, result)

    async def replace_custom_universe_symbols(
        self,
        universe_id: str,
        symbols: list[str],
        *,
        name: str | None = None,
        exchange: str = "SMART",
        primary_exchange: str | None = None,
        currency: str = "USD",
        security_type: str = "STK",
    ) -> ControlResult:
        """Replace one CUSTOM universe through canonical core normalization."""

        async with self._lock:
            normalized_id = universe_id.strip().upper()
            if not normalized_id.startswith("CUSTOM"):
                raise ValueError("Only CUSTOM universes may be edited")
            config = load_runs_config(self.runs_config_path)
            existing = next(
                (item for item in config.universes if item.universe_id == normalized_id), None
            )
            existing_by_symbol = (
                {member.symbol: member for member in existing.members} if existing else {}
            )
            members: list[InstrumentReference] = []
            for raw_symbol in symbols:
                symbol = raw_symbol.strip().upper()
                if not symbol:
                    continue
                members.append(
                    existing_by_symbol.get(symbol)
                    or InstrumentReference(
                        symbol=symbol,
                        exchange=exchange,
                        primary_exchange=primary_exchange,
                        currency=currency,
                        security_type=security_type,
                    )
                )
            universe = UniverseDefinition(
                universe_id=normalized_id,
                name=name or (existing.name if existing else normalized_id),
                members=tuple(members),
            )
            universes = tuple(
                universe if item.universe_id == normalized_id else item for item in config.universes
            )
            if existing is None:
                universes = (*universes, universe)
            updated = RunsConfig(universes=universes, runs=config.runs)
            affected = {
                run.run_id for run in updated.runs if run.universe == normalized_id and run.enabled
            }
            if self.runtime is None:
                self._write_runs(updated)
                return ControlResult(
                    True,
                    False,
                    ApplyMode.RESTART_RUN,
                    "Saved; active runtime unavailable",
                )
            result = await self._apply_runs(
                updated,
                affected,
                ApplyMode.RESTART_RUN,
                run=None,
            )
            if not result.runtime_applied:
                return result
            return await self._persist_runs_after_apply(updated, config, affected, result)

    def live_confirmation_context(self, run_id: str) -> dict[str, object]:
        _, run = self._config_and_run(run_id)
        account = load_ibkr_config(self.ibkr_config_path, Environment.LIVE).expected_account
        if account is None:
            raise ValueError("LIVE expected_account is required")
        return {
            "run_id": run.run_id,
            "universe": run.universe,
            "strategy": run.strategy,
            "target_environment": "LIVE",
            "target_account": account,
            "risk": run.risk.model_dump(mode="json") if run.risk else None,
        }

    def configuration(self) -> dict[str, object]:
        """Return editable config fields without broker credentials or runtime state copies."""

        runs = load_runs_config(self.runs_config_path)
        brokers = []
        for environment in Environment:
            try:
                broker = load_ibkr_config(self.ibkr_config_path, environment)
            except ValueError:
                continue
            brokers.append(broker.model_dump(mode="json"))
        return {
            "broker_configuration": brokers,
            "custom_universes": [
                item.model_dump(mode="json")
                for item in runs.universes
                if item.universe_id.upper().startswith("CUSTOM")
            ],
        }

    def _config_and_run(self, run_id: str) -> tuple[RunsConfig, RunConfig]:
        config = load_runs_config(self.runs_config_path)
        run = next((item for item in config.runs if item.run_id == run_id), None)
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        return config, run

    async def _replace_run(
        self,
        config: RunsConfig,
        current: RunConfig,
        mode: ApplyMode,
        *,
        target_environment: Environment | None = None,
        **updates: object,
    ) -> ControlResult:
        payload = current.model_dump(mode="json")
        payload.update(updates)
        updated = RunConfig.model_validate(payload)
        runs = tuple(updated if item.run_id == current.run_id else item for item in config.runs)
        validated = RunsConfig(universes=config.universes, runs=runs)
        if self.runtime is None:
            self._write_runs(validated)
            return ControlResult(
                True,
                False,
                mode,
                "Saved; runtime restart required",
                run=updated,
            )
        if target_environment is not None:
            runtime_environment = next(
                (
                    item
                    for item in self.runtime.status().execution_environments
                    if item.environment is target_environment
                ),
                None,
            )
            if runtime_environment is None or not runtime_environment.ready:
                try:
                    await self.runtime.replace_broker_config(
                        load_ibkr_config(self.ibkr_config_path, target_environment)
                    )
                except Exception as exc:
                    prefix = (
                        "LIVE_NOT_READY"
                        if target_environment is Environment.LIVE
                        else "BROKER_NOT_READY"
                    )
                    return ControlResult(
                        False,
                        False,
                        mode,
                        f"{prefix}: {exc}",
                        self.runtime.status(),
                        updated,
                    )
        result = await self._apply_runs(validated, {current.run_id}, mode, run=updated)
        if not result.runtime_applied:
            return result
        return await self._persist_runs_after_apply(validated, config, {current.run_id}, result)

    async def _apply_runs(
        self,
        config: RunsConfig,
        changed_run_ids: set[str],
        mode: ApplyMode,
        *,
        run: RunConfig | None,
    ) -> ControlResult:
        if self.runtime is None:
            raise RuntimeError("active runtime is required")
        try:
            status = await self.runtime.apply_runs_config(
                config, changed_run_ids=frozenset(changed_run_ids)
            )
        except Exception as exc:
            status = self.runtime.status()
            prefix = (
                "LIVE_NOT_READY"
                if run and run.environment is Environment.LIVE
                else "RUNTIME_NOT_APPLIED"
            )
            return ControlResult(False, False, mode, f"{prefix}: {exc}", status, run)
        selected = [
            item
            for item in status.runs
            if item.run_id in changed_run_ids and item.state.value == "DEGRADED"
        ]
        detail = "Applied to runtime"
        if selected:
            reasons = "; ".join(f"{item.run_id}: {item.reason}" for item in selected)
            detail = f"Applied; runtime degraded: {reasons}"
        return ControlResult(False, True, mode, detail, status, run)

    async def _apply_broker(self, config: IbkrConfig) -> ControlResult:
        if self.runtime is None:
            raise RuntimeError("active runtime is required")
        try:
            status = await self.runtime.replace_broker_config(config)
        except Exception as exc:
            status = self.runtime.status()
            return ControlResult(
                False,
                False,
                ApplyMode.RECONNECT_ENVIRONMENT,
                f"BROKER_NOT_APPLIED: {exc}",
                status,
            )
        environment = next(
            item for item in status.execution_environments if item.environment is config.environment
        )
        detail = (
            "Reconnected and reconciled"
            if environment.ready
            else "Applied; broker environment not ready"
        )
        return ControlResult(False, True, ApplyMode.RECONNECT_ENVIRONMENT, detail, status)

    @staticmethod
    def _persisted(result: ControlResult) -> ControlResult:
        detail = result.detail
        if detail.startswith("Applied"):
            detail = f"Saved and {detail[0].lower()}{detail[1:]}"
        elif detail.startswith("Reconnected"):
            detail = f"Saved, {detail[0].lower()}{detail[1:]}"
        else:
            detail = f"Saved; {detail}"
        return ControlResult(
            True,
            result.runtime_applied,
            result.apply_mode,
            detail,
            result.runtime,
            result.run,
        )

    async def _persist_runs_after_apply(
        self,
        updated: RunsConfig,
        previous: RunsConfig,
        changed_run_ids: set[str],
        result: ControlResult,
    ) -> ControlResult:
        try:
            self._write_runs(updated)
        except Exception as persistence_error:
            if self.runtime is None:
                raise
            try:
                await self.runtime.apply_runs_config(
                    previous, changed_run_ids=frozenset(changed_run_ids)
                )
            except Exception as rollback_error:
                raise RuntimeError(
                    "configuration persistence failed and runtime rollback failed: "
                    f"{rollback_error}"
                ) from persistence_error
            raise
        return self._persisted(result)

    async def _persist_broker_after_apply(
        self,
        payload: dict[str, object],
        previous: IbkrConfig,
        result: ControlResult,
    ) -> ControlResult:
        try:
            self._write_yaml(self.ibkr_config_path, payload)
        except Exception as persistence_error:
            if self.runtime is None:
                raise
            try:
                await self.runtime.replace_broker_config(previous)
            except Exception as rollback_error:
                raise RuntimeError(
                    f"broker persistence failed and runtime rollback failed: {rollback_error}"
                ) from persistence_error
            raise
        return self._persisted(result)

    def _confirm_live(
        self,
        run: RunConfig,
        confirmation: LiveConfirmation | None,
        *,
        target: Environment = Environment.LIVE,
        risk: RunRiskConfig | None = None,
    ) -> None:
        broker = load_ibkr_config(self.ibkr_config_path, target)
        if broker.expected_account is None:
            raise ValueError("LIVE expected_account is required")
        if confirmation is None or not confirmation.confirmed:
            raise ValueError("LIVE confirmation is required")
        if confirmation.target_account != broker.expected_account:
            raise ValueError("LIVE target account does not match configured expected_account")
        if risk is None and run.risk is None:
            raise ValueError("LIVE run requires explicit risk configuration")

    def _read_broker_yaml(self) -> dict[str, object]:
        with self.ibkr_config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("IBKR config must contain a YAML mapping")
        return raw

    def _write_runs(self, config: RunsConfig) -> None:
        self._write_yaml(self.runs_config_path, config.model_dump(mode="json"))

    @staticmethod
    def _write_yaml(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
