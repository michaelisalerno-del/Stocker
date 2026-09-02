"""Frozen-fixture diagnostic for the non-trading Stage 6 strategy."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from stocker_execution.session_hard_structure_d import (
    CohortOpportunity,
    EntryBar,
    SessionHardStructureDStrategy,
    StrategyContext,
    StrategySignal,
)
from stocker_execution.stage5 import Stage5FeatureSnapshot, Stage5Status


class _HistoryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: date
    pre_move_m: float = Field(ge=0.0)


class _EntryBarRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)


class _CandidateRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    con_id: int = Field(gt=0)
    symbol: str = Field(min_length=1)
    universe_id: str = Field(min_length=1)
    session: date
    t0: datetime
    p0: float = Field(gt=0.0)
    m_price: float = Field(gt=0.0)
    pre_move_m: float = Field(ge=0.0)
    session_hard_score: float
    session_hard_checkpoint: int
    entry_bars: tuple[_EntryBarRow, ...] = ()


class _StrategyFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    cohort_history: tuple[_HistoryRow, ...]
    candidates: tuple[_CandidateRow, ...]


def evaluate_strategy_fixture(path: Path) -> tuple[StrategySignal, ...]:
    """Evaluate a local frozen fixture without market data, account access, or orders."""

    fixture = _StrategyFixture.model_validate_json(path.read_text(encoding="utf-8"))
    history = tuple(
        CohortOpportunity(fixture.run_id, row.session, row.pre_move_m)
        for row in fixture.cohort_history
    )
    snapshots = tuple(_snapshot(fixture.run_id, row) for row in fixture.candidates)
    scores = {row.con_id: row.session_hard_score for row in fixture.candidates}
    checkpoints = {row.con_id: row.session_hard_checkpoint for row in fixture.candidates}
    entry_bars = {
        row.con_id: tuple(
            EntryBar(bar.timestamp, bar.open, bar.high, bar.low) for bar in row.entry_bars
        )
        for row in fixture.candidates
        if row.entry_bars
    }

    strategy = SessionHardStructureDStrategy()
    strategy.evaluate(
        snapshots,
        StrategyContext(
            run_id=fixture.run_id,
            session_hard_scores=scores,
            session_hard_checkpoints=checkpoints,
            cohort_history=history,
        ),
    )
    strategy.observe_entry_bars(entry_bars)
    return strategy.signals


def format_strategy_diagnostic(signals: tuple[StrategySignal, ...]) -> str:
    """Render the concise strategy fields needed to inspect a fixture run."""

    header = (
        "symbol PRE_MOVE_M percentile band session_hard direction rank selected "
        "entry_level entry_status signal_status"
    )
    rows = [header]
    for signal in signals:
        rows.append(
            " ".join(
                (
                    signal.symbol,
                    _number(signal.pre_move_m),
                    _number(signal.cohort_percentile),
                    signal.band.value if signal.band is not None else "NA",
                    "YES" if signal.session_hard_qualified else "NO",
                    signal.direction or "NA",
                    str(signal.candidate_rank) if signal.candidate_rank is not None else "NA",
                    "YES" if signal.selected else "NO",
                    _number(signal.entry_level),
                    signal.reason,
                    signal.status.value,
                )
            )
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local frozen-fixture diagnostic as a small standalone command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/stage6_glw_golden.json"),
        help="Frozen Stage 6 JSON fixture; no broker or account access is used.",
    )
    arguments = parser.parse_args(argv)
    print(format_strategy_diagnostic(evaluate_strategy_fixture(arguments.fixture)))
    return 0


def _snapshot(run_id: str, row: _CandidateRow) -> Stage5FeatureSnapshot:
    expected_return = row.m_price / row.p0
    return Stage5FeatureSnapshot(
        run_ids=(run_id,),
        universe_id=row.universe_id,
        con_id=row.con_id,
        symbol=row.symbol,
        session=row.session,
        t0=row.t0,
        status=Stage5Status.READY,
        exclusion_reason="",
        p0=row.p0,
        expected_absolute_return_15m=expected_return,
        m_price=row.m_price,
        raw_open_t0_minus_3m=row.p0 - row.pre_move_m * row.m_price,
        raw_open_t0=row.p0,
        alignment_factor=1.0,
        aligned_pre_open=row.p0 - row.pre_move_m * row.m_price,
        raw_pre_move_price=row.pre_move_m * row.m_price,
        pre_move_m=row.pre_move_m,
        calculation_version="STAGE5_PRE_MOVE_V1",
    )


def _number(value: float | None) -> str:
    return "NA" if value is None else f"{value:.12g}"


if __name__ == "__main__":
    raise SystemExit(main())
