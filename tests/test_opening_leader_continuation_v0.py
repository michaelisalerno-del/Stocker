from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections import deque
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from stocker_prospective import opening_leader_live_v0 as opening_leader_live
from stocker_prospective.contract import assert_no_broker_mutation_surface
from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.events import UnderlyingLevel1QuoteEvent
from stocker_prospective.live_bars import (
    AuditedLiveBar,
    KeepUpToDateBarFinalizer,
    xnys_session_bounds,
)
from stocker_prospective.live_recorder import FrozenM1CLiveRecorder
from stocker_prospective.live_subscriptions import QualifiedUnderlying
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.opening_leader_continuation_v0 import (
    CANONICAL_COHORT_HASH_V0,
    CANONICAL_COHORT_V0,
    CausalCheckpointBarV0,
    M1CContextV0,
    OpeningLeaderContinuationRecorderV0,
    OpeningLeaderEvidenceStoreV0,
    OpeningLeaderFreezeIdentityV0,
    OptionQuoteV0,
    OptionSnapshotCaptureV0,
    ProspectiveBoundaryErrorV0,
    RankPersistenceV0,
    build_observation_schedule_v0,
    calculate_rank_persistence_v0,
    calculate_underlying_shadow_return_v0,
    checkpoint_timestamp_v0,
    normalize_underlying_quote_v0,
    rank_opening_leader_v0,
    select_option_chain_requests_v0,
    select_option_diagnostics_v0,
    stable_evidence_id_v0,
)
from stocker_prospective.opening_leader_live_v0 import (
    OpeningLeaderDeploymentRefreezeReceiptV1,
    OpeningLeaderDeploymentRefreezeReceiptV2,
    OpeningLeaderIBKROptionSnapshotterV0,
    assert_opening_leader_runtime_configuration_v0,
    freeze_opening_leader_package_v0,
    load_opening_leader_package_v0,
    opening_leader_repository_root_v0,
    opening_leader_runtime_source_files_v0,
)
from stocker_prospective.read_store import ProspectiveReadStore

SESSION = date(2026, 8, 3)


def test_release_root_resolution_supports_no_editable_production_layout(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    installed_package = (
        release / ".venv" / "lib" / "python3.12" / "site-packages" / "stocker_prospective"
    )
    installed_package.mkdir(parents=True)
    (release / "pyproject.toml").write_text("[project]\nname = 'stocker'\n", encoding="utf-8")
    (release / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    assert opening_leader_repository_root_v0(installed_package) == release


def _bar(
    symbol: str,
    *,
    session: date = SESSION,
    checkpoint: int = 6,
    session_open: float = 100.0,
    checkpoint_close: float = 100.0,
) -> CausalCheckpointBarV0:
    signal_at = checkpoint_timestamp_v0(session, checkpoint)
    return CausalCheckpointBarV0(
        symbol=symbol,
        session=session,
        checkpoint=checkpoint,
        bar_start_utc=signal_at - timedelta(minutes=5),
        bar_end_utc=signal_at,
        available_at_utc=signal_at + timedelta(seconds=1),
        regular_session_open=session_open,
        checkpoint_open=min(session_open, checkpoint_close),
        checkpoint_high=max(session_open, checkpoint_close) + 0.1,
        checkpoint_low=min(session_open, checkpoint_close) - 0.1,
        checkpoint_close=checkpoint_close,
        session_open_source_id=f"open:{symbol}:{session.isoformat()}",
        checkpoint_source_id=f"bar:{symbol}:{checkpoint}",
        source_timestamp_utc=signal_at,
        received_timestamp_utc=signal_at + timedelta(seconds=1),
        source_completeness="complete",
        duplicate_resolution="unique",
    )


def test_canonical_cohort_checkpoint_mapping_and_rank_selection_are_frozen() -> None:
    assert CANONICAL_COHORT_V0 == (
        "AAL",
        "AAOI",
        "APLD",
        "ASTS",
        "CIFR",
        "HIMS",
        "IONQ",
        "IREN",
        "MARA",
        "MP",
        "MRNA",
        "MSTR",
        "NVTS",
        "QBTS",
        "RGTI",
        "RIOT",
        "RIVN",
        "SMCI",
        "SOFI",
        "WULF",
    )
    assert CANONICAL_COHORT_HASH_V0 == (
        "ab2a55b55329b83e410ebdf46fa5d47f5a75a44798d01374a4d7f072da57b634"
    )
    assert checkpoint_timestamp_v0(SESSION, 6) == datetime(2026, 8, 3, 14, 0, tzinfo=UTC)
    assert checkpoint_timestamp_v0(SESSION, 12) == datetime(2026, 8, 3, 14, 30, tzinfo=UTC)

    bars = tuple(
        _bar(
            symbol,
            checkpoint_close=(
                102.0 if symbol in {"AAL", "AAOI"} else 101.0 if symbol == "APLD" else 100.0
            ),
        )
        for symbol in CANONICAL_COHORT_V0
    )
    contexts = {
        symbol: M1CContextV0(
            probability=0.99 if symbol == "WULF" else 0.01,
            high_low_state="high" if symbol == "WULF" else "low",
            tail_phase="OUTSIDE_TAIL",
            qualified_fresh_event_status="none",
            movement_consumed=None,
            source_completeness="complete",
        )
        for symbol in CANONICAL_COHORT_V0
    }

    ranked = rank_opening_leader_v0(
        session=SESSION,
        checkpoint=6,
        bars=bars,
        m1c_context_by_symbol=contexts,
        cohort_hash=CANONICAL_COHORT_HASH_V0,
        evaluated_at_utc=checkpoint_timestamp_v0(SESSION, 6) + timedelta(seconds=2),
    )

    assert ranked.eligible is True
    assert ranked.rank_1.symbol == "AAL"
    assert ranked.rank_2.symbol == "AAOI"
    assert ranked.rank_1.open_to_checkpoint_return_bps == pytest.approx(200.0)
    assert ranked.rank_1_minus_rank_2_bps == pytest.approx(0.0)
    assert tuple(item.symbol for item in ranked.ranking[:3]) == ("AAL", "AAOI", "APLD")
    assert ranked.selected_identity == "rank_1"
    assert ranked.direction == "LONG"
    assert ranked.slate_size == 20
    assert ranked.exclusions == {}


def test_causal_slate_fails_closed_and_m1c_cannot_change_the_leader() -> None:
    evaluated = checkpoint_timestamp_v0(SESSION, 6) + timedelta(seconds=2)
    bars = tuple(
        _bar(symbol, checkpoint_close=104.0 if symbol == "AAL" else 100.0)
        for symbol in CANONICAL_COHORT_V0
    )
    low_context = {
        symbol: M1CContextV0(
            probability=0.0,
            high_low_state="LOW",
            tail_phase="OUTSIDE_TAIL",
            qualified_fresh_event_status="NO_QUALIFIED_FRESH_EVENT",
            movement_consumed=0.0,
            source_completeness="complete",
        )
        for symbol in CANONICAL_COHORT_V0
    }
    inverted_context = {
        symbol: context.model_copy(
            update={
                "probability": 1.0 if symbol == "WULF" else 0.0,
                "high_low_state": "HIGH" if symbol == "WULF" else "LOW",
                "tail_phase": "FIRST_ENTRY" if symbol == "WULF" else "OUTSIDE_TAIL",
                "qualified_fresh_event_status": (
                    "FIRST_ENTRY" if symbol == "WULF" else "NO_QUALIFIED_FRESH_EVENT"
                ),
                "movement_consumed": 99.0 if symbol == "WULF" else 0.0,
            }
        )
        for symbol, context in low_context.items()
    }
    baseline = rank_opening_leader_v0(
        session=SESSION,
        checkpoint=6,
        bars=bars,
        m1c_context_by_symbol=low_context,
        cohort_hash=CANONICAL_COHORT_HASH_V0,
        evaluated_at_utc=evaluated,
    )
    changed_context = rank_opening_leader_v0(
        session=SESSION,
        checkpoint=6,
        bars=bars,
        m1c_context_by_symbol=inverted_context,
        cohort_hash=CANONICAL_COHORT_HASH_V0,
        evaluated_at_utc=evaluated,
    )
    assert baseline.rank_1 is not None
    assert changed_context.rank_1 is not None
    assert baseline.rank_1.symbol == changed_context.rank_1.symbol == "AAL"
    assert baseline.slate_hash == changed_context.slate_hash
    assert [item.symbol for item in baseline.ranking] == [
        item.symbol for item in changed_context.ranking
    ]

    incomplete = rank_opening_leader_v0(
        session=SESSION,
        checkpoint=6,
        bars=bars[:14],
        m1c_context_by_symbol=inverted_context,
        cohort_hash=CANONICAL_COHORT_HASH_V0,
        evaluated_at_utc=evaluated,
    )
    assert incomplete.eligible is False
    assert "fewer_than_15_complete_stocks" in incomplete.failure_reasons
    assert len(incomplete.exclusions) == 6
    assert all(reason == "missing_checkpoint_bar" for reason in incomplete.exclusions.values())

    future_bar = bars[0].model_copy(update={"available_at_utc": evaluated + timedelta(seconds=1)})
    future_contaminated = rank_opening_leader_v0(
        session=SESSION,
        checkpoint=6,
        bars=(future_bar, *bars[1:]),
        m1c_context_by_symbol=low_context,
        cohort_hash="0" * 64,
        evaluated_at_utc=evaluated,
    )
    assert future_contaminated.eligible is False
    assert future_contaminated.exclusions["AAL"] == "future_data_not_causally_available"
    assert "canonical_cohort_hash_mismatch" in future_contaminated.failure_reasons


def test_observation_and_option_schedules_are_fixed_from_e0_and_xnys_close() -> None:
    signal_at = checkpoint_timestamp_v0(SESSION, 6)
    e0_at = signal_at + timedelta(seconds=7)

    schedule = build_observation_schedule_v0(
        session=SESSION,
        signal_timestamp_utc=signal_at,
        e0_timestamp_utc=e0_at,
    )

    targets = {item.name: item for item in schedule.underlying_targets}
    assert targets["SIGNAL"].target_timestamp_utc == signal_at
    assert targets["E0"].target_timestamp_utc == e0_at
    assert targets["E1"].target_timestamp_utc == e0_at + timedelta(minutes=5)
    assert targets["E2"].target_timestamp_utc == e0_at + timedelta(minutes=10)
    assert targets["H30"].target_timestamp_utc == e0_at + timedelta(minutes=30)
    assert targets["H60"].target_timestamp_utc == e0_at + timedelta(minutes=60)
    assert targets["H120"].target_timestamp_utc == e0_at + timedelta(minutes=120)
    assert targets["PRE_CLOSE_30"].target_timestamp_utc == datetime(2026, 8, 3, 19, 30, tzinfo=UTC)
    assert targets["PRE_CLOSE_15"].target_timestamp_utc == datetime(2026, 8, 3, 19, 45, tzinfo=UTC)
    assert targets["PRE_CLOSE_5"].target_timestamp_utc == datetime(2026, 8, 3, 19, 55, tzinfo=UTC)
    assert targets["PRE_CLOSE_1"].target_timestamp_utc == datetime(2026, 8, 3, 19, 59, tzinfo=UTC)
    assert targets["FINAL_CONTINUOUS"].target_timestamp_utc == datetime(
        2026, 8, 3, 19, 59, 59, tzinfo=UTC
    )
    assert targets["FINAL_CONTINUOUS"].executable is True
    assert targets["OFFICIAL_CLOSE"].target_timestamp_utc == datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    assert targets["OFFICIAL_CLOSE"].executable is False
    assert targets["OFFICIAL_CLOSE"].source_kind == "official_bar_close_reference"
    assert schedule.option_snapshot_names == (
        "SIGNAL",
        "E0",
        "H60",
        "H120",
        "PRE_CLOSE_30",
        "FINAL_CONTINUOUS",
    )


def test_underlying_quotes_preserve_staleness_and_conservative_shadow_sides() -> None:
    entry_at = checkpoint_timestamp_v0(SESSION, 6) + timedelta(seconds=7)
    entry = normalize_underlying_quote_v0(
        quote_id="entry-quote",
        symbol="AAL",
        target_timestamp_utc=entry_at,
        captured_at_utc=entry_at,
        provider_timestamp_utc=entry_at - timedelta(milliseconds=250),
        values={
            "last": 100.75,
            "bid": 100.0,
            "ask": 101.0,
            "bid_size": 50.0,
            "ask_size": 40.0,
            "market_data_type": "live",
            "halted": False,
        },
        source="ibkr_snapshot",
        maximum_quote_age_seconds=2.0,
    )
    exit_quote = normalize_underlying_quote_v0(
        quote_id="exit-quote",
        symbol="AAL",
        target_timestamp_utc=entry_at + timedelta(minutes=30),
        captured_at_utc=entry_at + timedelta(minutes=30),
        provider_timestamp_utc=entry_at + timedelta(minutes=30, milliseconds=-500),
        values={
            "last": 103.25,
            "bid": 103.0,
            "ask": 104.0,
            "bid_size": 60.0,
            "ask_size": 30.0,
            "market_data_type": "live",
            "halted": False,
        },
        source="ibkr_snapshot",
        maximum_quote_age_seconds=2.0,
    )

    assert entry.midpoint == pytest.approx(100.5)
    assert entry.spread_dollars == pytest.approx(1.0)
    assert entry.spread_bps == pytest.approx(99.5024875622)
    assert entry.quote_age_seconds == pytest.approx(0.25)
    assert entry.valid_for_signal is True

    result = calculate_underlying_shadow_return_v0(
        entry=entry,
        exit_quote=exit_quote,
        configured_fee_bps=1.0,
        friction_diagnostics_bps=(0.0, 1.0, 5.0, 10.0),
        official_close_reference=104.5,
    )

    assert result.conservative_ask_to_bid_gross_bps == pytest.approx(
        (103.0 / 101.0 - 1.0) * 10_000.0
    )
    assert result.conservative_ask_to_bid_net_bps == pytest.approx(
        (103.0 / 101.0 - 1.0) * 10_000.0 - 1.0
    )
    assert result.midpoint_to_midpoint_diagnostic_bps == pytest.approx(
        (103.5 / 100.5 - 1.0) * 10_000.0
    )
    assert result.last_to_last_diagnostic_bps == pytest.approx((103.25 / 100.75 - 1.0) * 10_000.0)
    assert result.official_close_reference_bps == pytest.approx((104.5 / 101.0 - 1.0) * 10_000.0)
    assert result.official_close_executable is False
    assert result.friction_diagnostics_bps["10"] == pytest.approx(
        result.conservative_ask_to_bid_gross_bps - 10.0
    )

    stale = normalize_underlying_quote_v0(
        quote_id="stale-quote",
        symbol="AAL",
        target_timestamp_utc=entry_at,
        captured_at_utc=entry_at,
        provider_timestamp_utc=entry_at - timedelta(seconds=3),
        values={"bid": 100.0, "ask": 101.0, "market_data_type": "live"},
        source="ibkr_snapshot",
        maximum_quote_age_seconds=2.0,
    )
    assert stale.valid_for_signal is False
    assert "quote_stale" in stale.data_quality_flags


def test_live_quote_projection_uses_exclusive_e0_and_last_continuous_quote() -> None:
    signal = checkpoint_timestamp_v0(SESSION, 6)
    _, market_close = xnys_session_bounds(SESSION)

    def event(event_id: str, timestamp: datetime, sequence: int) -> UnderlyingLevel1QuoteEvent:
        receive_only = event_id == "signal-receive-only"
        return UnderlyingLevel1QuoteEvent(
            event_id=event_id,
            received_timestamp_utc=(
                timestamp if receive_only else timestamp + timedelta(milliseconds=100)
            ),
            received_monotonic_ns=sequence,
            provider_timestamp_utc=None if receive_only else timestamp,
            source_sequence=sequence,
            session=SESSION,
            symbol="AAL",
            con_id=101,
            request_id=1,
            bid=100.0 + sequence,
            bid_size=10.0,
            ask=100.1 + sequence,
            ask_size=11.0,
            last=100.05 + sequence,
            last_size=1.0,
            market_data_type=MarketDataType.LIVE,
            source="synthetic_ibkr_level1",
            quote_valid=True,
            tick_type="level1",
            exchange="SMART",
            halted=False,
        )

    live = object.__new__(FrozenM1CLiveRecorder)
    live.maximum_quote_age = timedelta(seconds=2)
    live._quotes = {
        "AAL": deque(
            (
                event("signal-receive-only", signal, 1),
                event("next", signal + timedelta(milliseconds=1), 2),
                event("last-continuous", market_close - timedelta(seconds=2), 3),
                event("actual-final", market_close - timedelta(milliseconds=200), 4),
            )
        )
    }
    signal_quote = live.opening_leader_underlying_quote("AAL", 6, "SIGNAL", signal, signal)
    e0 = live.opening_leader_underlying_quote("AAL", 6, "E0", signal, signal)
    final = live.opening_leader_underlying_quote(
        "AAL",
        6,
        "FINAL_CONTINUOUS",
        market_close - timedelta(seconds=1),
        market_close,
    )

    assert signal_quote is not None and signal_quote.valid_for_signal is True
    assert signal_quote.provider_timestamp_utc is None
    assert signal_quote.timestamp_provenance == "receive"
    assert "provider_timestamp_unavailable_receive_fallback" in (signal_quote.data_quality_flags)
    assert e0 is not None and e0.quote_id == "next"
    assert final is not None and final.quote_id == "actual-final"
    assert final.actual_quote_timestamp_utc < market_close


def test_option_chain_and_diagnostics_are_bounded_deterministic_and_unavailable_safe() -> None:
    plan = select_option_chain_requests_v0(
        session=SESSION,
        underlying="AAL",
        underlying_con_id=101,
        spot=100.0,
        available_expiries=(
            SESSION - timedelta(days=1),
            SESSION + timedelta(days=9),
            SESSION,
            SESSION + timedelta(days=2),
        ),
        available_strikes=tuple(float(value) for value in range(80, 121)),
        exchange="SMART",
        trading_class="AAL",
    )

    assert plan.status == "AVAILABLE"
    assert plan.selected_expiries == (SESSION, SESSION + timedelta(days=2))
    assert plan.selected_strikes_by_expiry == {
        SESSION.isoformat(): (85.0, 92.0, 100.0, 107.0, 115.0),
        (SESSION + timedelta(days=2)).isoformat(): (
            85.0,
            92.0,
            100.0,
            107.0,
            115.0,
        ),
    }
    assert len(plan.requests) == 20
    assert {request.right for request in plan.requests} == {"C", "P"}
    assert all(85.0 <= request.strike <= 115.0 for request in plan.requests)

    observed = checkpoint_timestamp_v0(SESSION, 6)

    def option_quote(
        con_id: int,
        strike: float,
        delta: float,
        *,
        bid: float = 1.0,
        ask: float = 1.1,
        age_seconds: float = 0.5,
    ) -> OptionQuoteV0:
        return OptionQuoteV0.from_snapshot(
            snapshot_id="option-snapshot",
            underlying="AAL",
            con_id=con_id,
            right="P",
            strike=strike,
            expiry=SESSION + timedelta(days=2),
            multiplier=100,
            trading_class="AAL",
            exchange="SMART",
            captured_at_utc=observed,
            provider_timestamp_utc=observed - timedelta(seconds=age_seconds),
            values={
                "bid": bid,
                "ask": ask,
                "last": 1.05,
                "bid_size": 10.0,
                "ask_size": 12.0,
                "volume": 100.0,
                "open_interest": 1000.0,
                "implied_volatility": 0.6,
                "delta": delta,
                "gamma": 0.03,
                "theta": -0.02,
                "vega": 0.04,
                "underlying_reference_price": 100.0,
                "market_data_type": "live",
            },
            maximum_quote_age_seconds=2.0,
        )

    quotes = (
        option_quote(201, 96.0, -0.10),
        option_quote(202, 97.0, -0.20),
        option_quote(203, 98.0, -0.30),
        option_quote(204, 99.0, -0.40),
    )
    diagnostics = select_option_diagnostics_v0(quotes)
    assert diagnostics["P20"].status == "AVAILABLE"
    assert diagnostics["P20"].short_con_id == 202
    assert diagnostics["P30"].status == "AVAILABLE"
    assert diagnostics["P30"].short_con_id == 203
    assert diagnostics["BPS20"].status == "AVAILABLE"
    assert diagnostics["BPS20"].short_con_id == 202
    assert diagnostics["BPS20"].long_con_id == 201
    assert diagnostics["BPS20"].entry_credit == pytest.approx(1.0 - 1.1)

    stale = option_quote(205, 95.0, -0.05, age_seconds=3.0)
    assert stale.stale is True
    assert "quote_stale" in stale.data_quality_flags

    unavailable = select_option_diagnostics_v0((option_quote(301, 97.0, -0.20),))
    assert unavailable["BPS20"].status == "UNAVAILABLE"
    assert unavailable["BPS20"].reason == "next_lower_strike_quote_unavailable"
    assert unavailable["BPS20"].long_con_id is None


def _metadata(recorded_at: datetime) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id="opening-leader-test-run",
        prospective_start_utc=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="opening-leader-test-model",
        universe_id="anchor-frozen-20-20260724",
        cohort="anchor_frozen_20",
        source_timestamps=[recorded_at.isoformat()],
        recorded_at_utc=recorded_at,
    )


def test_evidence_store_is_append_only_idempotent_restartable_and_links_late_data(
    tmp_path: Path,
) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    observed = checkpoint_timestamp_v0(SESSION, 6) + timedelta(seconds=5)
    metadata = _metadata(observed)
    repository.create_run(metadata, mode="record_only")
    store = OpeningLeaderEvidenceStoreV0(
        repository,
        deployment_receipt_id="olc-deployment-test",
        contract_hash="b" * 64,
        code_hash="c" * 64,
        cohort_hash=CANONICAL_COHORT_HASH_V0,
    )
    stable_id = stable_evidence_id_v0(
        deployment_receipt_id="olc-deployment-test",
        session=SESSION,
        checkpoint=6,
        record_type="signal_receipt",
        observation_name="SIGNAL",
    )

    first = store.append(
        metadata,
        stable_id=stable_id,
        session=SESSION,
        checkpoint=6,
        signal_timestamp_utc=checkpoint_timestamp_v0(SESSION, 6),
        selected_symbol="AAL",
        record_type="signal_receipt",
        observation_name="SIGNAL",
        payload={"selected_identity": "rank_1", "direction": "LONG"},
        data_quality_flags=(),
    )
    retry = store.append(
        metadata,
        stable_id=stable_id,
        session=SESSION,
        checkpoint=6,
        signal_timestamp_utc=checkpoint_timestamp_v0(SESSION, 6),
        selected_symbol="AAL",
        record_type="signal_receipt",
        observation_name="SIGNAL",
        payload={"selected_identity": "rank_1", "direction": "LONG"},
        data_quality_flags=(),
    )
    assert retry == first
    assert store.records_for_run("opening-leader-test-run") == (first,)

    with pytest.raises(ValueError, match="immutable opening-leader evidence differs"):
        store.append(
            metadata,
            stable_id=stable_id,
            session=SESSION,
            checkpoint=6,
            signal_timestamp_utc=checkpoint_timestamp_v0(SESSION, 6),
            selected_symbol="AAOI",
            record_type="signal_receipt",
            observation_name="SIGNAL",
            payload={"selected_identity": "rank_1", "direction": "LONG"},
            data_quality_flags=(),
        )

    late = store.append_late_or_correction(
        _metadata(observed + timedelta(minutes=1)),
        original_stable_id=stable_id,
        correction_kind="late_observation",
        payload={"late_source_timestamp": (observed - timedelta(seconds=1)).isoformat()},
        data_quality_flags=("late_data_not_rewriting_original",),
    )
    assert late.original_stable_id == stable_id
    assert late.stable_id != stable_id

    restarted = OpeningLeaderEvidenceStoreV0(
        repository,
        deployment_receipt_id="olc-deployment-test",
        contract_hash="b" * 64,
        code_hash="c" * 64,
        cohort_hash=CANONICAL_COHORT_HASH_V0,
    )
    assert restarted.recorded_identities("opening-leader-test-run") == {
        stable_id,
        late.stable_id,
    }

    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE opening_leader_evidence_v0 SET selected_symbol = 'WULF' "
                "WHERE stable_id = ?",
                (stable_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM opening_leader_evidence_v0 WHERE stable_id = ?",
                (stable_id,),
            )


def test_dashboard_support_excludes_receipts_without_primary_outcome_quotes(
    tmp_path: Path,
) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    start = datetime(2026, 8, 1, 19, 30, tzinfo=UTC)
    repository.create_run(_metadata(start), mode="record_only")
    store = OpeningLeaderEvidenceStoreV0(
        repository,
        deployment_receipt_id="olc-deployment-test",
        contract_hash="b" * 64,
        code_hash="c" * 64,
        cohort_hash=CANONICAL_COHORT_HASH_V0,
    )
    sessions: list[date] = []
    candidate = SESSION
    while len(sessions) < 60:
        try:
            checkpoint_timestamp_v0(candidate, 6)
        except ValueError:
            pass
        else:
            sessions.append(candidate)
        candidate += timedelta(days=1)
    for index, session in enumerate(sessions):
        signal_at = checkpoint_timestamp_v0(session, 6)
        store.append(
            _metadata(signal_at + timedelta(minutes=6)),
            stable_id=stable_evidence_id_v0(
                deployment_receipt_id="olc-deployment-test",
                session=session,
                checkpoint=6,
                record_type="signal_receipt",
                observation_name="SIGNAL",
            ),
            session=session,
            checkpoint=6,
            signal_timestamp_utc=signal_at,
            selected_symbol=CANONICAL_COHORT_V0[index % 15],
            record_type="signal_receipt",
            observation_name="SIGNAL",
            payload={"valid": True},
            data_quality_flags=(),
        )

    dashboard = ProspectiveReadStore(
        repository.database_path,
        run_id="opening-leader-test-run",
    ).opening_leader_continuation_v0()

    assert dashboard["sample_status"] == "PROSPECTIVE SAMPLE INCOMPLETE"
    assert dashboard["checkpoints"]["C6"]["support"]["valid_sessions"] == 0
    assert dashboard["checkpoints"]["C6"]["support"]["complete"] is False


def test_rank_persistence_is_diagnostic_and_tracks_original_leader_excursions() -> None:
    result = calculate_rank_persistence_v0(
        original_leader="AAL",
        signal_price=100.0,
        current_price=101.0,
        current_return_bps_by_symbol={
            "AAOI": 250.0,
            "AAL": 200.0,
            "APLD": 150.0,
            "ASTS": 50.0,
            "CIFR": -25.0,
        },
        observed_path_prices=(100.0, 102.0, 99.0, 101.0),
    )

    assert result.original_leader == "AAL"
    assert result.current_rank == 2
    assert result.remains_rank_1 is False
    assert result.remains_top_2 is True
    assert result.remains_above_cohort_median is True
    assert result.return_since_signal_bps == pytest.approx(100.0)
    assert result.drawdown_from_signal_bps == pytest.approx(0.0)
    assert result.maximum_favourable_excursion_bps == pytest.approx(200.0)
    assert result.maximum_adverse_excursion_bps == pytest.approx(-100.0)
    assert result.signal_admission_changed is False


def test_live_rank_persistence_uses_latest_causally_finalised_common_bar() -> None:
    market_open, _ = xnys_session_bounds(SESSION)
    target = checkpoint_timestamp_v0(SESSION, 6) + timedelta(minutes=11)

    def audited_bar(symbol: str, checkpoint: int) -> AuditedLiveBar:
        start = market_open + timedelta(minutes=5 * (checkpoint - 1))
        return AuditedLiveBar(
            symbol=symbol,
            session=SESSION,
            bar_start_utc=start,
            bar_end_utc=start + timedelta(minutes=5),
            checkpoint=checkpoint,
            open=100.0,
            high=101.0 + checkpoint / 10,
            low=99.0,
            close=100.0 + checkpoint / 10,
            volume_or_activity_field=1000.0,
            wap_where_available=100.0,
            trade_count_where_available=10,
            source="synthetic_finalised_bar",
            source_completeness="complete",
            finalised=True,
            provider_timestamp_utc=start + timedelta(minutes=5),
            received_timestamp_utc=start + timedelta(minutes=5, milliseconds=100),
        )

    quote = UnderlyingLevel1QuoteEvent(
        event_id="persistence-target",
        received_timestamp_utc=target,
        received_monotonic_ns=1,
        provider_timestamp_utc=target,
        source_sequence=1,
        session=SESSION,
        symbol="AAL",
        con_id=101,
        request_id=1,
        bid=104.9,
        bid_size=10.0,
        ask=105.1,
        ask_size=10.0,
        last=105.0,
        last_size=1.0,
        market_data_type=MarketDataType.LIVE,
        source="synthetic_ibkr_level1",
        quote_valid=True,
        tick_type="level1",
        exchange="SMART",
        halted=False,
    )
    live = object.__new__(FrozenM1CLiveRecorder)
    live.universe_symbols = CANONICAL_COHORT_V0
    live._bars = {
        (symbol, SESSION): {
            checkpoint: audited_bar(symbol, checkpoint) for checkpoint in range(1, 9)
        }
        for symbol in CANONICAL_COHORT_V0
    }
    live._bar_finalised_at = {
        (symbol, SESSION, checkpoint): (
            target + timedelta(seconds=1) if checkpoint == 8 else target - timedelta(seconds=1)
        )
        for symbol in CANONICAL_COHORT_V0
        for checkpoint in range(1, 9)
    }
    live._quotes = {"AAL": deque((quote,))}
    live._trades = {}

    persistence = live.opening_leader_rank_persistence(
        "AAL",
        SESSION,
        6,
        target,
        100.0,
    )

    assert persistence is not None
    assert persistence.return_since_signal_bps == pytest.approx(500.0)
    assert persistence.current_rank == 1


def test_official_close_uses_post_close_boundary_snapshot_and_stays_non_executable() -> None:
    _, market_close = xnys_session_bounds(SESSION)
    finalizer = KeepUpToDateBarFinalizer(
        prospective_collection_start=datetime(2026, 8, 1, tzinfo=UTC)
    )
    finalizer.register(1, symbol="AAL", con_id=101)
    assert (
        finalizer.add(
            request_id=1,
            bar_start_utc=market_close - timedelta(minutes=5),
            provider_timestamp_utc=market_close,
            received_timestamp_utc=market_close + timedelta(milliseconds=100),
            open=103.0,
            high=104.0,
            low=102.0,
            close=103.75,
            volume=1000.0,
            wap=103.5,
            trade_count=10,
        )
        == ()
    )
    live = object.__new__(FrozenM1CLiveRecorder)
    live._bars = {}
    live._finalizer = finalizer

    assert (
        live.opening_leader_official_close("AAL", SESSION, market_close + timedelta(seconds=4))
        is None
    )
    official = live.opening_leader_official_close(
        "AAL", SESSION, market_close + timedelta(seconds=5)
    )
    assert official is not None
    assert official[0] == pytest.approx(103.75)
    assert official[2] == market_close


def test_synthetic_recorder_creates_independent_receipts_and_recovers_due_observations(
    tmp_path: Path,
) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    start = datetime(2026, 8, 1, 19, 30, tzinfo=UTC)
    repository.create_run(_metadata(start), mode="record_only")
    store = OpeningLeaderEvidenceStoreV0(
        repository,
        deployment_receipt_id="olc-deployment-test",
        contract_hash="b" * 64,
        code_hash="c" * 64,
        cohort_hash=CANONICAL_COHORT_HASH_V0,
    )
    freeze = OpeningLeaderFreezeIdentityV0(
        deployment_receipt_id="olc-deployment-test",
        freeze_completed_at_utc=start,
        contract_hash="b" * 64,
        code_hash="c" * 64,
        cohort_hash=CANONICAL_COHORT_HASH_V0,
        source_hashes_signed=True,
        order_routing_disabled=True,
        protected_historical_outcomes_accessed=False,
    )
    quote_calls: list[tuple[str, int, str, datetime]] = []
    persistence_signal_prices: list[float] = []

    def bars_for(bar_session: date, checkpoint: int) -> tuple[CausalCheckpointBarV0, ...]:
        return tuple(
            _bar(
                symbol,
                session=bar_session,
                checkpoint=checkpoint,
                checkpoint_close=(
                    103.0 if symbol == "AAL" else 102.0 if symbol == "AAOI" else 100.0
                ),
            )
            for symbol in CANONICAL_COHORT_V0
        )

    def quote_for(
        symbol: str,
        checkpoint: int,
        observation_name: str,
        target: datetime,
        now: datetime,
    ) -> RankPersistenceV0:
        quote_calls.append((symbol, checkpoint, observation_name, target))
        _, market_close = xnys_session_bounds(target.astimezone(UTC).date())
        provider_timestamp = (
            min(
                market_close - timedelta(milliseconds=200),
                now - timedelta(milliseconds=100),
            )
            if observation_name == "FINAL_CONTINUOUS"
            else now - timedelta(milliseconds=100)
        )
        price = {
            "SIGNAL": 103.0,
            "E0": 110.0,
            "E1": 111.0,
        }.get(observation_name, 104.0)
        return normalize_underlying_quote_v0(
            quote_id=f"{checkpoint}-{observation_name}-{len(quote_calls)}",
            symbol=symbol,
            target_timestamp_utc=target,
            captured_at_utc=now,
            provider_timestamp_utc=provider_timestamp,
            values={
                "bid": price - 0.1,
                "ask": price + 0.1,
                "last": price,
                "bid_size": 20.0,
                "ask_size": 21.0,
                "market_data_type": "live",
                "halted": False,
            },
            source="synthetic_ibkr_snapshot",
            maximum_quote_age_seconds=2.0,
        )

    def option_for(
        _symbol: str,
        _checkpoint: int,
        observation_name: str,
        _spot: float,
        now: datetime,
    ) -> OptionSnapshotCaptureV0:
        return OptionSnapshotCaptureV0(
            snapshot_id=f"options-{observation_name}-{now.timestamp()}",
            observation_name=observation_name,
            captured_at_utc=now,
            status="UNAVAILABLE",
            reason="synthetic_fixture_no_chain",
            selection=None,
            quotes=(),
        )

    contexts = {
        symbol: M1CContextV0(
            probability=0.99 if symbol == "WULF" else 0.01,
            high_low_state="high" if symbol == "WULF" else "low",
            tail_phase="OUTSIDE_TAIL",
            qualified_fresh_event_status="none",
            movement_consumed=None,
            source_completeness="complete",
        )
        for symbol in CANONICAL_COHORT_V0
    }

    def persistence_for(
        symbol: str,
        _session: date,
        _checkpoint: int,
        _target: datetime,
        signal_price: float,
    ) -> object:
        persistence_signal_prices.append(signal_price)
        return calculate_rank_persistence_v0(
            original_leader=symbol,
            signal_price=signal_price,
            current_price=signal_price + 1.0,
            current_return_bps_by_symbol={symbol: 200.0, "AAOI": 100.0},
            observed_path_prices=(signal_price, signal_price + 1.0),
        )

    def build_recorder() -> OpeningLeaderContinuationRecorderV0:
        return OpeningLeaderContinuationRecorderV0(
            store=store,
            freeze_identity=freeze,
            prospective_start_utc=start,
            metadata_factory=lambda observed, _sources: _metadata(observed),
            bar_provider=bars_for,
            underlying_quote_provider=quote_for,
            option_snapshot_provider=option_for,
            rank_persistence_provider=persistence_for,
            official_close_provider=lambda *_args: None,
        )

    signal = checkpoint_timestamp_v0(SESSION, 6)
    first = build_recorder().poll(
        session=SESSION,
        now=signal + timedelta(minutes=6),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert first.created_signal_receipts == ("C6",)
    assert first.created_observations == ("C6:SIGNAL",)
    assert first.created_option_snapshots == ("C6:SIGNAL",)

    second = build_recorder().poll(
        session=SESSION,
        now=signal + timedelta(minutes=6, seconds=1),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert second.created_signal_receipts == ()
    assert second.created_observations == ("C6:E0",)
    assert quote_calls[-1][3] == signal + timedelta(minutes=6)

    e1 = build_recorder().poll(
        session=SESSION,
        now=signal + timedelta(minutes=11, seconds=1),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert e1.created_observations == ("C6:E1",)
    assert persistence_signal_prices == [103.0, 103.0]
    records = store.records_for_run("opening-leader-test-run")
    assert len([record for record in records if record.record_type == "signal_receipt"]) == 1
    assert {
        record.observation_name
        for record in records
        if record.record_type == "underlying_observation"
    } == {"SIGNAL", "E0", "E1"}

    c12_signal = checkpoint_timestamp_v0(SESSION, 12)
    c12 = build_recorder().poll(
        session=SESSION,
        now=c12_signal + timedelta(minutes=6),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert c12.created_signal_receipts == ("C12",)
    assert (
        len(
            [
                record
                for record in store.records_for_run("opening-leader-test-run")
                if record.record_type == "signal_receipt"
            ]
        )
        == 2
    )
    projection = ProspectiveReadStore(
        repository.database_path,
        run_id="opening-leader-test-run",
    ).opening_leader_continuation_v0()
    assert projection["banner"] == "RECORD ONLY — ORDERS DISABLED"
    assert projection["sample_status"] == "PROSPECTIVE SAMPLE INCOMPLETE"
    assert projection["checkpoints"]["C6"]["rank_1"] == "AAL"
    assert projection["checkpoints"]["C12"]["role"] == "secondary"
    assert projection["checkpoint_pooling_allowed"] is False
    assert projection["option_policy_authorized"] is False

    _, market_close = xnys_session_bounds(SESSION)
    before_close = build_recorder().poll(
        session=SESSION,
        now=market_close - timedelta(milliseconds=500),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert "C6:FINAL_CONTINUOUS" not in before_close.created_observations
    assert "C6:FINAL_CONTINUOUS" in before_close.created_option_snapshots
    after_close = build_recorder().poll(
        session=SESSION,
        now=market_close + timedelta(seconds=1),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert "C6:FINAL_CONTINUOUS" in after_close.created_observations
    assert "C6:FINAL_CONTINUOUS" not in after_close.created_option_snapshots
    final_record = next(
        record
        for record in store.records_for_run("opening-leader-test-run")
        if record.session == SESSION
        and record.checkpoint == 6
        and record.record_type == "underlying_observation"
        and record.observation_name == "FINAL_CONTINUOUS"
    )
    assert datetime.fromisoformat(
        str(final_record.payload["quote"]["actual_quote_timestamp_utc"]).replace("Z", "+00:00")
    ) == market_close - timedelta(milliseconds=200)

    second_session = date(2026, 8, 4)
    second_session_signal = checkpoint_timestamp_v0(second_session, 6)
    next_day = build_recorder().poll(
        session=second_session,
        now=second_session_signal + timedelta(minutes=6),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert next_day.created_signal_receipts == ("C6",)
    assert (
        len(
            [
                record
                for record in store.records_for_run("opening-leader-test-run")
                if record.record_type == "signal_receipt"
            ]
        )
        == 3
    )

    receipt_only_session = date(2026, 8, 5)
    receipt_only_signal = checkpoint_timestamp_v0(receipt_only_session, 6)
    receipt_only_recorder = build_recorder()

    def crash_before_signal_child(**_kwargs: object) -> object:
        raise AssertionError("synthetic_crash_after_signal_receipt")

    receipt_only_recorder._record_quote_observation = crash_before_signal_child  # type: ignore[method-assign]
    with pytest.raises(AssertionError, match="synthetic_crash_after_signal_receipt"):
        receipt_only_recorder.poll(
            session=receipt_only_session,
            now=receipt_only_signal + timedelta(minutes=6),
            m1c_context_by_checkpoint={6: contexts, 12: contexts},
        )
    receipt_only_rows = [
        record
        for record in store.records_for_run("opening-leader-test-run")
        if record.session == receipt_only_session and record.checkpoint == 6
    ]
    assert [record.record_type for record in receipt_only_rows] == ["signal_receipt"]
    receipt_recovery = build_recorder().poll(
        session=receipt_only_session,
        now=receipt_only_signal + timedelta(minutes=6, seconds=1),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert receipt_recovery.created_observations == ("C6:SIGNAL", "C6:E0")
    assert receipt_recovery.created_option_snapshots == ("C6:SIGNAL", "C6:E0")

    option_crash_session = date(2026, 8, 6)
    option_crash_signal = checkpoint_timestamp_v0(option_crash_session, 6)
    option_crash_recorder = build_recorder()

    def crash_before_option_child(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("synthetic_crash_before_option_snapshot")

    option_crash_recorder.option_snapshot_provider = crash_before_option_child
    with pytest.raises(AssertionError, match="synthetic_crash_before_option_snapshot"):
        option_crash_recorder.poll(
            session=option_crash_session,
            now=option_crash_signal + timedelta(minutes=6),
            m1c_context_by_checkpoint={6: contexts, 12: contexts},
        )
    option_crash_rows = [
        record
        for record in store.records_for_run("opening-leader-test-run")
        if record.session == option_crash_session and record.checkpoint == 6
    ]
    assert [record.record_type for record in option_crash_rows] == [
        "signal_receipt",
        "underlying_observation",
    ]
    option_recovery = build_recorder().poll(
        session=option_crash_session,
        now=option_crash_signal + timedelta(minutes=6, seconds=1),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert option_recovery.created_option_snapshots == ("C6:SIGNAL", "C6:E0")

    late_session = date(2026, 8, 7)
    late_signal = checkpoint_timestamp_v0(late_session, 6)
    late_available = False

    def late_quote_for(
        symbol: str,
        checkpoint: int,
        observation_name: str,
        target: datetime,
        now: datetime,
    ) -> object | None:
        if observation_name != "E0":
            return quote_for(symbol, checkpoint, observation_name, target, now)
        if not late_available:
            return None
        actual = target + timedelta(seconds=1)
        return normalize_underlying_quote_v0(
            quote_id=f"late-{checkpoint}-E0",
            symbol=symbol,
            target_timestamp_utc=target,
            captured_at_utc=actual,
            provider_timestamp_utc=actual,
            values={
                "bid": 109.9,
                "ask": 110.1,
                "last": 110.0,
                "market_data_type": "live",
                "halted": False,
            },
            source="synthetic_retained_late_quote",
            maximum_quote_age_seconds=2.0,
        )

    late_recorder = build_recorder()
    late_recorder.underlying_quote_provider = late_quote_for
    late_recorder.poll(
        session=late_session,
        now=late_signal + timedelta(minutes=6),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    missed = late_recorder.poll(
        session=late_session,
        now=late_signal + timedelta(minutes=7, seconds=31),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert "C6:E0" in missed.created_observations
    late_available = True
    appended_late = late_recorder.poll(
        session=late_session,
        now=late_signal + timedelta(minutes=8),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert "C6:E0" in appended_late.created_observations
    late_rows = [
        record
        for record in store.records_for_run("opening-leader-test-run")
        if record.session == late_session
        and record.checkpoint == 6
        and record.observation_name == "E0"
    ]
    original_e0 = next(record for record in late_rows if record.original_stable_id is None)
    linked_e0 = next(record for record in late_rows if record.original_stable_id is not None)
    assert original_e0.payload["status"] == "UNAVAILABLE"
    assert linked_e0.original_stable_id == original_e0.stable_id
    assert linked_e0.record_type == "late_observation"

    no_backfill_session = date(2026, 8, 10)
    no_backfill = build_recorder().poll(
        session=no_backfill_session,
        now=checkpoint_timestamp_v0(date(2026, 8, 11), 6),
        m1c_context_by_checkpoint={6: contexts, 12: contexts},
    )
    assert no_backfill.created_signal_receipts == ()
    assert no_backfill.created_failures == ()
    assert not any(
        record.session == no_backfill_session
        for record in store.records_for_run("opening-leader-test-run")
    )

    with pytest.raises(ProspectiveBoundaryErrorV0, match="historical backfill is forbidden"):
        build_recorder().poll(
            session=date(2026, 7, 31),
            now=start,
            m1c_context_by_checkpoint={6: contexts, 12: contexts},
        )


def test_deployment_freeze_receipt_binds_artifacts_sources_and_boundary(
    tmp_path: Path,
) -> None:
    source_package = (
        Path(__file__).parents[1]
        / "prospective"
        / "opening-leader-continuation"
        / "20260801-opening-leader-continuation-recorder-v0"
    )
    package = tmp_path / "frozen-package"
    shutil.copytree(
        source_package,
        package,
        ignore=shutil.ignore_patterns(
            "deployment_freeze_receipt.json",
            "deployment_freeze_receipt_v1.json",
            "deployment_freeze_receipt_v2.json",
        ),
    )
    source = tmp_path / "opening_leader_source.py"
    source.write_text("ORDER_ROUTING_ENABLED = False\n", encoding="utf-8")
    frozen_at = datetime(2026, 8, 1, 20, 0, tzinfo=UTC)
    receipt = freeze_opening_leader_package_v0(
        package,
        freeze_completed_at_utc=frozen_at,
        source_files={"opening_leader_source": source},
        verification={
            "focused_tests": "passed",
            "prospective_recorder_tests": "passed",
            "ibkr_shadow_tests": "passed",
            "dashboard_tests": "passed",
            "lint": "passed",
            "type_check": "passed",
            "synthetic_dry_run": "passed",
            "restart_recovery": "passed",
        },
    )
    loaded = load_opening_leader_package_v0(
        package,
        prospective_start_utc=frozen_at + timedelta(seconds=1),
        source_files={"opening_leader_source": source},
    )
    assert loaded == receipt
    assert (
        load_opening_leader_package_v0(
            package,
            source_files={"opening_leader_source": source},
        )
        == receipt
    )
    assert loaded.source_hashes_signed is True
    assert loaded.order_routing_disabled is True
    assert loaded.protected_historical_outcomes_accessed is False
    assert_opening_leader_runtime_configuration_v0(
        mode="record_only",
        maximum_quote_age_seconds=2.0,
        trading_enabled=False,
    )
    with pytest.raises(ValueError, match="order placement"):
        assert_opening_leader_runtime_configuration_v0(
            mode="record_only",
            maximum_quote_age_seconds=2.0,
            trading_enabled=True,
        )

    (package / "rank_manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_opening_leader_package_v0(
            package,
            prospective_start_utc=frozen_at + timedelta(seconds=1),
            source_files={"opening_leader_source": source},
        )


def test_committed_opening_leader_refreeze_preserves_original_and_binds_current_sources() -> None:
    package = (
        Path(__file__).parents[1]
        / "prospective"
        / "opening-leader-continuation"
        / "20260801-opening-leader-continuation-recorder-v0"
    )
    original = package / "deployment_freeze_receipt.json"

    receipt = load_opening_leader_package_v0(
        package,
        source_files=opening_leader_runtime_source_files_v0(),
    )

    assert hashlib.sha256(original.read_bytes()).hexdigest() == (
        "22c205fe043d7ce3a9f427d0de997de2a0170be2022ec39db3ae661d7534ef7d"
    )
    assert isinstance(receipt, OpeningLeaderDeploymentRefreezeReceiptV2)
    assert receipt.recorder_version == "opening-leader-continuation-recorder-v0"
    assert receipt.supersedes_receipt_sha256 == (
        "65c3f9aa2bd788850d5beacd968992503ced47849b1634ed068f032e0910a306"
    )
    assert receipt.supersedes_deployment_receipt_id == (
        "olc-deploy-7d2019790b5d19f6982cb835"
    )
    assert receipt.frozen_semantics_changed is False


def test_opening_leader_refreeze_cannot_move_freeze_boundary_backward(
    tmp_path: Path,
) -> None:
    source_package = (
        Path(__file__).parents[1]
        / "prospective"
        / "opening-leader-continuation"
        / "20260801-opening-leader-continuation-recorder-v0"
    )
    package = tmp_path / "opening-leader-package"
    shutil.copytree(source_package, package)
    (package / "deployment_freeze_receipt_v2.json").unlink(missing_ok=True)
    original = opening_leader_live.OpeningLeaderDeploymentReceiptV0.model_validate_json(
        (package / "deployment_freeze_receipt.json").read_text(encoding="utf-8")
    )
    refreeze_path = package / "deployment_freeze_receipt_v1.json"
    payload = json.loads(refreeze_path.read_text(encoding="utf-8"))
    payload["freeze_completed_at_utc"] = (
        original.freeze_completed_at_utc - timedelta(seconds=1)
    ).isoformat()
    payload["deployment_receipt_id"] = "olc-deploy-placeholder"
    payload["signature_sha256"] = "0" * 64
    provisional = OpeningLeaderDeploymentRefreezeReceiptV1.model_validate(payload)
    with_id = provisional.model_copy(
        update={
            "deployment_receipt_id": opening_leader_live._expected_deployment_receipt_id(
                provisional
            )
        }
    )
    signed = with_id.model_copy(
        update={
            "signature_sha256": hashlib.sha256(
                opening_leader_live._canonical_json(
                    opening_leader_live._signature_payload(with_id)
                ).encode("utf-8")
            ).hexdigest()
        }
    )
    refreeze_path.write_text(
        json.dumps(signed.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refreeze lineage mismatch"):
        load_opening_leader_package_v0(
            package,
            source_files=opening_leader_runtime_source_files_v0(),
        )


def test_opening_leader_v2_refreeze_cannot_move_freeze_boundary_backward(
    tmp_path: Path,
) -> None:
    source_package = (
        Path(__file__).parents[1]
        / "prospective"
        / "opening-leader-continuation"
        / "20260801-opening-leader-continuation-recorder-v0"
    )
    package = tmp_path / "opening-leader-package"
    shutil.copytree(source_package, package)
    prior = OpeningLeaderDeploymentRefreezeReceiptV1.model_validate_json(
        (package / "deployment_freeze_receipt_v1.json").read_text(encoding="utf-8")
    )
    refreeze_path = package / "deployment_freeze_receipt_v2.json"
    payload = json.loads(refreeze_path.read_text(encoding="utf-8"))
    payload["freeze_completed_at_utc"] = (
        prior.freeze_completed_at_utc - timedelta(seconds=1)
    ).isoformat()
    payload["deployment_receipt_id"] = "olc-deploy-placeholder"
    payload["signature_sha256"] = "0" * 64
    provisional = OpeningLeaderDeploymentRefreezeReceiptV2.model_validate(payload)
    with_id = provisional.model_copy(
        update={
            "deployment_receipt_id": opening_leader_live._expected_deployment_receipt_id(
                provisional
            )
        }
    )
    signed = with_id.model_copy(
        update={
            "signature_sha256": hashlib.sha256(
                opening_leader_live._canonical_json(
                    opening_leader_live._signature_payload(with_id)
                ).encode("utf-8")
            ).hexdigest()
        }
    )
    refreeze_path.write_text(
        json.dumps(signed.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="V2 deployment refreeze lineage mismatch"):
        load_opening_leader_package_v0(
            package,
            source_files=opening_leader_runtime_source_files_v0(),
        )


def test_option_snapshotter_uses_two_expiries_twenty_exact_contracts_and_pacing() -> None:
    now = checkpoint_timestamp_v0(SESSION, 6) + timedelta(minutes=5)
    pacing: list[str] = []

    class Adapter:
        def request_option_chain_metadata(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                items=(
                    SimpleNamespace(
                        underlyingConId=101,
                        exchange="SMART",
                        tradingClass="AAL",
                        expirations={"20260807", "20260814", "20260821"},
                        strikes={80.0, 90.0, 98.0, 99.0, 100.0, 101.0, 102.0, 110.0},
                    ),
                )
            )

        def qualify_exact_contract(self, contract: object) -> SimpleNamespace:
            return SimpleNamespace(
                items=(
                    SimpleNamespace(
                        contract=SimpleNamespace(
                            **vars(contract),
                            secType="OPT",
                            conId=(
                                int(contract.strike * 100)
                                + (1 if contract.right == "C" else 2)
                                + int(str(contract.expiry)[-2:]) * 10_000
                            ),
                        )
                    ),
                )
            )

        def capture_temporary_quote(self, **_kwargs: object) -> SimpleNamespace:
            contract = _kwargs["contract"]
            received = (now - timedelta(milliseconds=100)).isoformat()
            return SimpleNamespace(
                items=(
                    {
                        "field": "bid",
                        "value": 1.0,
                        "receive_timestamp_utc": received,
                        "market_data_type": "live",
                    },
                    {
                        "field": "ask",
                        "value": 1.1,
                        "receive_timestamp_utc": received,
                        "market_data_type": "live",
                    },
                    {
                        "field": "bid_size",
                        "value": 10.0,
                        "receive_timestamp_utc": received,
                        "market_data_type": "live",
                    },
                    {
                        "field": "ask_size",
                        "value": 12.0,
                        "receive_timestamp_utc": received,
                        "market_data_type": "live",
                    },
                    {
                        "field": (
                            "put_open_interest" if contract.right == "P" else "call_open_interest"
                        ),
                        "value": 321.0,
                        "receive_timestamp_utc": received,
                        "market_data_type": "live",
                    },
                    {
                        "field": "option_computation",
                        "computation_source": "ask",
                        "delta": -0.91 if contract.right == "P" else 0.91,
                        "implied_volatility": 0.9,
                        "receive_timestamp_utc": received,
                        "market_data_type": "live",
                    },
                    {
                        "field": "option_computation",
                        "computation_source": "model",
                        "implied_volatility": 0.4,
                        "delta": -0.2 if contract.right == "P" else 0.2,
                        "gamma": 0.03,
                        "theta": -0.04,
                        "vega": 0.05,
                        "underlying_reference_price": 100.0,
                        "receive_timestamp_utc": received,
                        "market_data_type": "live",
                    },
                    {
                        "field": "option_computation",
                        "computation_source": "bid",
                        "delta": -0.81 if contract.right == "P" else 0.81,
                        "implied_volatility": 0.8,
                        "receive_timestamp_utc": received,
                        "market_data_type": "live",
                    },
                )
            )

    snapshotter = OpeningLeaderIBKROptionSnapshotterV0(
        adapter=Adapter(),
        underlying_contracts={
            "AAL": QualifiedUnderlying(
                symbol="AAL",
                con_id=101,
                upstream_contract=object(),
                exchange="SMART",
            )
        },
        contract_factory=lambda symbol, expiry, strike, right, multiplier, exchange, trading: (
            SimpleNamespace(
                symbol=symbol,
                expiry=expiry.strftime("%Y%m%d"),
                strike=strike,
                right=right,
                multiplier=multiplier,
                exchange=exchange,
                tradingClass=trading,
            )
        ),
        request_heartbeat=lambda: pacing.append("paced"),
        maximum_quote_age_seconds=2.0,
        clock=lambda: now,
    )
    capture = snapshotter("AAL", 6, "SIGNAL", 100.0, now)

    assert capture.status == "AVAILABLE", capture.reason
    assert capture.selection is not None
    assert capture.selection.selected_expiries == (date(2026, 8, 7), date(2026, 8, 14))
    assert len(capture.selection.requests) == 20
    assert len(capture.quotes) == 20
    assert all(quote.available for quote in capture.quotes)
    assert all(quote.provider_timestamp_utc is None for quote in capture.quotes)
    assert all(quote.timestamp_provenance == "receive" for quote in capture.quotes)
    assert all(quote.open_interest == 321.0 for quote in capture.quotes)
    assert all(abs(quote.delta or 0.0) == 0.2 for quote in capture.quotes)
    assert all(quote.greeks_source == "model" for quote in capture.quotes)
    assert all(
        set(quote.option_computation_by_source) == {"ask", "model", "bid"}
        for quote in capture.quotes
    )
    assert len(pacing) == 41
    assert_no_broker_mutation_surface(snapshotter)
