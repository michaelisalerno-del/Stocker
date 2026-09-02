"""Explicit, opt-in Stage 7 IBKR PAPER execution diagnostic."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from stocker_core.config import IbkrConfig
from stocker_core.runs import RunConfig
from stocker_execution.execution_ledger import ExecutionLedger
from stocker_execution.ibkr import IbkrConnection, IbkrError
from stocker_execution.session_hard_structure_d import SignalStatus, StrategySignal
from stocker_execution.stage7 import ExecutionResultCode, Stage7ExecutionService


@dataclass(frozen=True, slots=True)
class PaperDiagnosticReport:
    run_id: str
    account: str
    signal_id: str
    symbol: str
    con_id: int
    account_equity: float | None
    risk_fraction: float
    risk_budget: float
    entry: float
    stop: float
    target: float
    quantity: int
    parent_order_id: int
    stop_order_id: int
    target_order_id: int
    entry_status: str
    filled_quantity: float
    average_fill_price: float | None
    position_quantity: float
    ledger_status: str


async def run_paper_diagnostic(
    *,
    run: RunConfig,
    broker_config: IbkrConfig,
    signal_id: str,
    symbol: str,
    exchange: str,
    primary_exchange: str | None,
    currency: str,
    entry_reference: float,
    m_price: float,
    ledger_path: Path,
    wait_seconds: float,
) -> PaperDiagnosticReport:
    """Submit and observe one deliberately requested protected PAPER order."""

    if broker_config.expected_account is None or run.risk is None:
        raise IbkrError("Stage 7 diagnostic requires explicit account and run risk")
    connection = IbkrConnection(broker_config, execution_enabled=True)
    try:
        session = await connection.connect()
        instrument = await connection.resolve_stock(
            symbol,
            exchange=exchange,
            primary_exchange=primary_exchange,
            currency=currency,
        )
        observed_at = datetime.now(tz=UTC)
        intent = StrategySignal(
            strategy_id="STAGE7_PAPER_DIAGNOSTIC",
            strategy_version="STAGE7_DIAGNOSTIC_V1",
            signal_id=signal_id,
            run_id=run.run_id,
            underlying_con_id=instrument.con_id,
            symbol=instrument.symbol,
            universe_id=run.universe,
            session=observed_at.date(),
            t0=observed_at,
            status=SignalStatus.ENTRY_TRIGGERED,
            reason="EXPLICIT_STAGE7_PAPER_DIAGNOSTIC",
            pre_move_m=None,
            cohort_percentile=None,
            band=None,
            session_hard_score=None,
            session_hard_checkpoint=None,
            session_hard_qualified=False,
            feature_calculation_version="DIAGNOSTIC",
            side="SHORT",
            direction="DOWN",
            candidate_rank=1,
            selected=True,
            p0=entry_reference,
            m_price=m_price,
            entry_level=entry_reference,
            entry_reference=entry_reference,
            entry_timestamp=observed_at,
            signal_timestamp=observed_at,
        )
        ledger = ExecutionLedger(ledger_path)
        service = Stage7ExecutionService(
            run=run,
            expected_account=broker_config.expected_account,
            broker=connection,
            ledger=ledger,
        )
        reconciliation = await service.reconcile()
        if not reconciliation.ok:
            raise IbkrError(reconciliation.detail)
        account_state = await connection.account_state()
        result = await service.execute(intent, instrument, diagnostic=True)
        if result.code is not ExecutionResultCode.SUBMITTED:
            raise IbkrError(f"{result.code.value}: {result.detail}")
        if result.order_plan is None or result.order_ids is None:
            raise IbkrError("IBKR submission returned no protected order identity")

        deadline = asyncio.get_running_loop().time() + wait_seconds
        while True:
            await service.refresh_broker_state()
            record = ledger.get(result.order_plan.order_plan_id)
            if wait_seconds == 0.0 or (
                record is not None
                and record.status.value in {"FILLED", "CLOSED", "CANCELLED", "REJECTED"}
            ):
                break
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0.0:
                break
            await asyncio.sleep(min(1.0, remaining))

        statuses = {status.order_id: status for status in await connection.read_order_statuses()}
        positions = await connection.read_positions()
        record = ledger.get(result.order_plan.order_plan_id)
        parent_status = statuses.get(result.order_ids.parent)
        matching = [position for position in positions if position.con_id == instrument.con_id]
        return PaperDiagnosticReport(
            run_id=run.run_id,
            account=session.masked_account_id,
            signal_id=signal_id,
            symbol=instrument.symbol,
            con_id=instrument.con_id,
            account_equity=account_state.equity,
            risk_fraction=run.risk.risk_per_trade,
            risk_budget=(account_state.equity or 0.0) * run.risk.risk_per_trade,
            entry=result.order_plan.entry_reference,
            stop=result.order_plan.stop_price,
            target=result.order_plan.target_price,
            quantity=result.order_plan.quantity,
            parent_order_id=result.order_ids.parent,
            stop_order_id=result.order_ids.stop,
            target_order_id=result.order_ids.target,
            entry_status=(parent_status.status.value if parent_status else "PENDING_CALLBACK"),
            filled_quantity=record.filled_quantity if record else 0.0,
            average_fill_price=record.average_fill_price if record else None,
            position_quantity=matching[0].quantity if matching else 0.0,
            ledger_status=record.status.value if record else "UNKNOWN",
        )
    finally:
        connection.disconnect()
