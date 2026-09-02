from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

from stocker_core.runs import Environment, RunConfig, RunInstance, RunState
from stocker_core.universes import InstrumentReference, UniverseDefinition
from stocker_execution.ibkr import IbkrConnection, IbkrError, QualifiedInstrument
from stocker_execution.stage5 import (
    Stage5Analyzer,
    Stage5FeatureResult,
    Stage5FeatureSnapshot,
    Stage5Membership,
    Stage5QualifiedRequest,
    Stage5SnapshotStore,
    Stage5Status,
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
        status=Stage5Status.READY,
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
                status=Stage5Status.PRE_MOVE_NOT_READY,
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
    assert all(
        not hasattr(row, field)
        for row in rows
        for field in (
            "passes_pre_move_threshold",
            "cohort_id",
            "cohort_size",
            "cohort_pre_move_percentile",
            "pre_move_band",
            "cohort_reason",
        )
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
        Stage5Status.READY,
        Stage5Status.PRE_MOVE_NOT_READY,
    ]


def test_snapshot_store_preserves_ready_feature_across_transient_rerun(
    tmp_path: Path,
) -> None:
    store = Stage5SnapshotStore(tmp_path / "stage5.sqlite3")
    ready = Stage5FeatureSnapshot(
        run_ids=("RUN",),
        universe_id="US_ALL",
        con_id=123,
        symbol="HOOD",
        session=date(2025, 2, 20),
        t0=T0,
        status=Stage5Status.READY,
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
        calculation_version="STAGE5_PRE_MOVE_V1",
    )
    store.save(ready)
    store.save(
        replace(
            ready,
            status=Stage5Status.PRE_MOVE_NOT_READY,
            exclusion_reason="transient missing T0 open",
            p0=None,
            m_price=None,
            pre_move_m=None,
        )
    )

    assert store.get("US_ALL", T0, 123) == ready


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


def test_active_run_qualification_rejects_unsupported_security_type_locally() -> None:
    boundary = QualificationBoundary()
    universe = UniverseDefinition(
        universe_id="OPTIONS",
        name="OPTIONS",
        members=(
            InstrumentReference(
                symbol="HOOD",
                exchange="SMART",
                currency="USD",
                security_type="OPT",
            ),
        ),
    )
    run = RunInstance(
        RunConfig(
            run_id="RUN",
            universe="OPTIONS",
            strategy="LATER_STAGE",
            environment=Environment.PAPER,
        ),
        universe,
        RunState.ACTIVE,
    )

    result = asyncio.run(qualify_active_runs(boundary, (run,)))

    assert boundary.calls == []
    assert result.requests == ()
    assert len(result.ineligible) == 1
    assert result.ineligible[0].reason == "unsupported security type: OPT"


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
        ("HOOD", 123, Stage5Status.READY),
        ("BAD", None, Stage5Status.INELIGIBLE),
    ]
    assert boundary.order_calls == 0
