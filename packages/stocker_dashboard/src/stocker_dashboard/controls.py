"""Explicit run-configuration commands for the Stage 10 control surface."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from stocker_core.config import RunsConfig, load_ibkr_config, load_runs_config
from stocker_core.runs import Environment, RunConfig, RunRiskConfig


@dataclass(frozen=True, slots=True)
class LiveConfirmation:
    confirmed: bool
    target_account: str


class RunControlService:
    """Validate and persist deliberate operations through Stocker's config models."""

    def __init__(
        self,
        runs_config_path: str | Path,
        ibkr_config_path: str | Path,
        *,
        on_change: Callable[[str, RunConfig], None] | None = None,
    ) -> None:
        self.runs_config_path = Path(runs_config_path)
        self.ibkr_config_path = Path(ibkr_config_path)
        self.on_change = on_change

    def enable_run(self, run_id: str, *, confirmation: LiveConfirmation | None = None) -> RunConfig:
        current = self._run(run_id)
        if current.environment is Environment.LIVE:
            self._confirm_live(current, confirmation)
        return self._replace("enable_run", current, enabled=True)

    def disable_run(self, run_id: str) -> RunConfig:
        return self._replace("disable_run", self._run(run_id), enabled=False)

    def update_run_config(
        self,
        run_id: str,
        *,
        risk_per_trade: float,
        max_concurrent_positions: int | None,
        universe: str,
        strategy: str,
        confirmation: LiveConfirmation | None = None,
    ) -> RunConfig:
        current = self._run(run_id)
        if risk_per_trade <= 0 or risk_per_trade > 1:
            raise ValueError("risk_per_trade must be greater than zero and no more than one")
        if max_concurrent_positions is not None and max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")
        if current.environment is Environment.LIVE:
            self._confirm_live(current, confirmation)
        risk = RunRiskConfig(
            risk_per_trade=risk_per_trade,
            max_concurrent_positions=max_concurrent_positions,
        )
        return self._replace(
            "update_run_config",
            current,
            universe=universe,
            strategy=strategy,
            risk=risk.model_dump(mode="json"),
        )

    def change_execution_environment(
        self,
        run_id: str,
        environment: Environment,
        *,
        confirmation: LiveConfirmation | None = None,
    ) -> RunConfig:
        current = self._run(run_id)
        if environment is Environment.LIVE:
            self._confirm_live(current, confirmation, target=environment)
        return self._replace("change_execution_environment", current, environment=environment.value)

    def live_confirmation_context(self, run_id: str) -> dict[str, object]:
        run = self._run(run_id)
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

    def _confirm_live(
        self,
        run: RunConfig,
        confirmation: LiveConfirmation | None,
        *,
        target: Environment = Environment.LIVE,
    ) -> None:
        broker = load_ibkr_config(self.ibkr_config_path, target)
        if broker.expected_account is None:
            raise ValueError("LIVE expected_account is required")
        if confirmation is None or not confirmation.confirmed:
            raise ValueError("LIVE confirmation is required")
        if confirmation.target_account != broker.expected_account:
            raise ValueError("LIVE target account does not match configured expected_account")
        if run.risk is None:
            raise ValueError("LIVE run requires explicit risk configuration")

    def _run(self, run_id: str) -> RunConfig:
        config = load_runs_config(self.runs_config_path)
        run = next((item for item in config.runs if item.run_id == run_id), None)
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        return run

    def _replace(self, operation: str, current: RunConfig, **updates: object) -> RunConfig:
        config = load_runs_config(self.runs_config_path)
        payload = current.model_dump(mode="json")
        payload.update(updates)
        updated = RunConfig.model_validate(payload)
        runs = tuple(updated if item.run_id == current.run_id else item for item in config.runs)
        validated = RunsConfig(universes=config.universes, runs=runs)
        self._write(validated)
        if self.on_change is not None:
            self.on_change(operation, updated)
        return updated

    def _write(self, config: RunsConfig) -> None:
        self.runs_config_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=self.runs_config_path.parent,
            prefix=f".{self.runs_config_path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.runs_config_path)
        except Exception:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
