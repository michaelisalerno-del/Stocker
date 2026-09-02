from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from stocker_core.runs import Environment, RunConfig, RunInstance, RunState
from stocker_core.universes import InstrumentReference, UniverseDefinition
from stocker_execution.ibkr import IbkrConnection, IbkrError, QualifiedInstrument
from stocker_execution.stage5 import (
    CandidateStatus,
    Stage5Analyzer,
    Stage5CandidateSnapshot,
    Stage5FeatureResult,
    Stage5Membership,
    Stage5QualifiedRequest,
    Stage5SnapshotStore,
    calculate_cohort_percentile,
    qualify_active_runs,
)

T0 = datetime(2025, 2, 20, 15, 0, tzinfo=UTC)


def qualified(symbol: str, con_id: int) -> QualifiedInstrument:
    return QualifiedInstrument(symbol, con_id, "SMART", "NASDAQ", "USD", "STK")


def ready_feature(instrument: QualifiedInstrument) -> Stage5FeatureResult:
    return Stage5FeatureResult(
        con_id=instrument.con_id,
        symbol=instrument.symbol,
        session=date(2025, 2, 20),
        t0=T0,
        status=CandidateStatus.READY,
        exclusion_reason="",
        p0=100.0,
        expected_absolute_return_15m=0.01,
        m_price=1.0,
        raw_open_t0_minus_3m=99.0,
        raw_open_t0=100.0,
        alignment_factor=1.0,
        aligned_pre_open=99.0,
        raw_pre_move_price=1.0,
        pre_move_m=1.0,
        passes_pre_move_threshold=True,
    )


class FeatureService:
    def __init__(self, invalid_con_id: int | None = None) -> None:
        self.calls: list[int] = []
        self.invalid_con_id = invalid_con_id

    async def get_feature(
        self, instrument: QualifiedInstrument, *, session: date, t0: datetime
    ) -> Stage5FeatureResult:
        self.calls.append(instrument.con_id)
        if instrument.con_id == self.invalid_con_id:
            return Stage5FeatureResult(
                con_id=instrument.con_id,
                symbol=instrument.symbol,
                session=session,
                t0=t0,
                status=CandidateStatus.PRE_MOVE_NOT_READY,
                exclusion_reason="PRE_MOVE_NOT_READY: missing exact T0 open",
            )
        return replace(ready_feature(instrument), session=session, t0=t0)


def test_overlapping_universes_reuse_one_canonical_conid_feature_calculation() -> None:
    service = FeatureService()
    analyzer = Stage5Analyzer(service)
    instrument = qualified("HOOD", 123)

    rows = asyncio.run(
        analyzer.analyze(
            (
                Stage5QualifiedRequest(
                    instrument,
                    (Stage5Membership("RUN_NASDAQ", "NASDAQ"),),
                ),
                Stage5QualifiedRequest(
                    instrument,
                    (Stage5Membership("RUN_ALL", "US_ALL"),),
                ),
            ),
            session=date(2025, 2, 20),
            t0=T0,
        )
    )

    assert service.calls == [123]
    assert [(row.universe_id, row.run_ids) for row in rows] == [
        ("NASDAQ", ("RUN_NASDAQ",)),
        ("US_ALL", ("RUN_ALL",)),
    ]
    assert {row.con_id for row in rows} == {123}
    assert all(row.cohort_id is None for row in rows)
    assert all(row.cohort_pre_move_percentile is None for row in rows)
    assert all(row.pre_move_band is None for row in rows)
    assert all(
        row.cohort_reason == "CANONICAL_COHORT_MEMBERSHIP_UNRESOLVED" for row in rows
    )


def test_one_invalid_instrument_does_not_kill_the_rest_of_the_batch() -> None:
    service = FeatureService(invalid_con_id=456)
    analyzer = Stage5Analyzer(service)

    rows = asyncio.run(
        analyzer.analyze(
            (
                Stage5QualifiedRequest(
                    qualified("HOOD", 123),
                    (Stage5Membership("RUN", "US_ALL"),),
                ),
                Stage5QualifiedRequest(
                    qualified("BAD", 456),
                    (Stage5Membership("RUN", "US_ALL"),),
                ),
            ),
            session=date(2025, 2, 20),
            t0=T0,
        )
    )

    assert [row.symbol for row in rows] == ["HOOD", "BAD"]
    assert [row.status for row in rows] == [
        CandidateStatus.READY,
        CandidateStatus.PRE_MOVE_NOT_READY,
    ]


def stored_row(session: date, con_id: int, pre_move_m: float) -> Stage5CandidateSnapshot:
    return Stage5CandidateSnapshot(
        run_ids=("RUN",),
        universe_id="US_ALL",
        cohort_id="US_ALL",
        con_id=con_id,
        symbol=f"S{con_id}",
        session=session,
        t0=datetime.combine(session, T0.timetz()),
        status=CandidateStatus.READY,
        exclusion_reason="",
        p0=100.0,
        expected_absolute_return_15m=0.01,
        m_price=1.0,
        raw_open_t0_minus_3m=99.0,
        raw_open_t0=100.0,
        alignment_factor=1.0,
        aligned_pre_open=99.0,
        raw_pre_move_price=pre_move_m,
        pre_move_m=pre_move_m,
        passes_pre_move_threshold=True,
        cohort_size=0,
        cohort_pre_move_percentile=None,
        pre_move_band=None,
        cohort_reason="",
        calculation_version="STAGE5_PRE_MOVE_V1",
    )


def test_store_uses_only_preceding_20_distinct_explicit_cohort_sessions(
    tmp_path: Path,
) -> None:
    store = Stage5SnapshotStore(tmp_path / "stage5.sqlite3")
    current_session = date(2025, 2, 22)
    for offset in range(21, 0, -1):
        prior_session = current_session - timedelta(days=offset)
        value_pair = (1.0, 1.0) if offset == 21 else (0.5, 2.0)
        for index, value in enumerate(value_pair):
            store.save(stored_row(prior_session, offset * 10 + index, value))
    store.save(stored_row(current_session, 999, 100.0))

    history = store.prior_qualifying_pre_move_m(
        "US_ALL",
        before_session=current_session,
    )

    assert len(history) == 40
    assert calculate_cohort_percentile(1.0, history) == 50.0


def test_transient_not_ready_rerun_cannot_replace_ready_snapshot(tmp_path: Path) -> None:
    store = Stage5SnapshotStore(tmp_path / "stage5.sqlite3")
    prior_session = date(2025, 2, 19)
    ready = stored_row(prior_session, 123, 1.25)
    store.save(ready)
    store.save(
        replace(
            ready,
            status=CandidateStatus.PRE_MOVE_NOT_READY,
            exclusion_reason="transient missing T0 open",
            pre_move_m=None,
            passes_pre_move_threshold=None,
        )
    )

    assert store.prior_qualifying_pre_move_m(
        "US_ALL", before_session=date(2025, 2, 20)
    ) == (1.25,)


class QualificationBoundary(IbkrConnection):
    def __init__(self) -> None:
        self.config = SimpleNamespace(environment=Environment.PAPER)
        self.calls: list[str] = []
        self.order_calls = 0

    async def resolve_stock(
        self,
        symbol: str,
        *,
        exchange: str,
        currency: str,
        primary_exchange: str | None = None,
    ) -> QualifiedInstrument:
        del exchange, currency, primary_exchange
        self.calls.append(symbol)
        if symbol == "BAD":
            raise IbkrError("unresolved")
        return qualified(symbol, 123 if symbol == "HOOD" else 456)

    async def place_order(self, *_args: object, **_kwargs: object) -> None:
        self.order_calls += 1
        raise AssertionError("Stage 5 must never place an order")


def active_run(run_id: str, universe_id: str, *symbols: str) -> RunInstance:
    universe = UniverseDefinition(
        universe_id=universe_id,
        name=universe_id,
        members=tuple(
            InstrumentReference(symbol=symbol, exchange="SMART", currency="USD")
            for symbol in symbols
        ),
    )
    return RunInstance(
        RunConfig(
            run_id=run_id,
            universe=universe_id,
            strategy="LATER_STAGE",
            environment=Environment.PAPER,
        ),
        universe,
        RunState.ACTIVE,
    )


def test_active_run_qualification_reuses_reference_and_isolates_one_failure() -> None:
    boundary = QualificationBoundary()

    result = asyncio.run(
        qualify_active_runs(
            boundary,
            (
                active_run("RUN_A", "NASDAQ", "HOOD", "BAD"),
                active_run("RUN_B", "US_ALL", "HOOD"),
            ),
        )
    )

    assert boundary.calls == ["BAD", "HOOD"]
    assert len(result.requests) == 1
    assert result.requests[0].instrument.con_id == 123
    assert set(result.requests[0].memberships) == {
        Stage5Membership("RUN_A", "NASDAQ"),
        Stage5Membership("RUN_B", "US_ALL"),
    }
    assert len(result.ineligible) == 1
    assert result.ineligible[0].symbol == "BAD"


def test_active_run_analysis_emits_ineligible_rows_and_places_no_orders() -> None:
    boundary = QualificationBoundary()
    analyzer = Stage5Analyzer(FeatureService())

    rows = asyncio.run(
        analyzer.analyze_active_runs(
            boundary,
            (active_run("RUN", "US_ALL", "HOOD", "BAD"),),
            session=date(2025, 2, 20),
            t0=T0,
        )
    )

    assert [(row.symbol, row.con_id, row.status) for row in rows] == [
        ("HOOD", 123, CandidateStatus.READY),
        ("BAD", None, CandidateStatus.INELIGIBLE),
    ]
    assert boundary.order_calls == 0
