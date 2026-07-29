from __future__ import annotations

import json
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from stocker_prospective.budget_reports import (
    DAILY_REPORT_FILENAMES,
    BudgetAwareDailyReportWriter,
)
from stocker_prospective.capacity import (
    CapacityDiscovery,
    RuntimeCapacitySettings,
    WindowedRequestPacer,
    resolve_runtime_capacity,
)
from stocker_prospective.contract import claims_boundary
from stocker_prospective.database import (
    EvidenceMetadata,
    ProspectiveRepository,
)
from stocker_prospective.fake_ibkr import FakeIBKRAdapter
from stocker_prospective.frozen_live_application import (
    _probe_required_market_data_type,
)
from stocker_prospective.ibkr_official import OfficialMarketDataOnlyClient
from stocker_prospective.live_subscriptions import (
    LiveSubscriptionController,
    QualifiedUnderlying,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.option_budget import (
    BudgetAwareEpisodeStateMachine,
    DteAllocator,
    EpisodeKind,
    EpisodeState,
    OptionEpisodeTask,
    OptionSubscriptionIntent,
    SnapshotConcurrencyGate,
)
from stocker_prospective.option_discovery import merge_snapshot_items
from stocker_prospective.option_ledger import OptionContract
from stocker_prospective.option_recorder import _planned_atm_contract
from stocker_prospective.options import DteBucket
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.subscriptions import (
    BudgetState,
    SubscriptionBudgetManager,
    SubscriptionClass,
    SubscriptionKind,
    SubscriptionPriority,
    canonical_subscription_key,
)
from stocker_prospective.transfer import (
    M1CTransferMonitor,
    ProviderM1CObservation,
    TransferBar,
    create_ibkr_calibration_candidate,
)

ROOT = Path(__file__).parents[1]
BUDGET_FAKE_FIXTURE = (
    ROOT
    / "packages/stocker_prospective/src/stocker_prospective/fixtures"
    / "ibkr-budget-aware-shadow-v0.json"
)


def test_runtime_capacity_prefers_discovery_and_preserves_future_trading_reserve(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "ibkr_runtime_capacity_manifest.json"
    manifest = resolve_runtime_capacity(
        settings=RuntimeCapacitySettings(
            configured_total_market_data_lines=100,
            configured_externally_reserved_lines=4,
            reserved_future_trading_lines=12,
            safety_margin_lines=2,
            configured_max_tick_by_tick=5,
            configured_max_depth=3,
            configured_max_concurrent_snapshots=2,
            configured_historical_requests_per_window=60,
            configured_historical_request_window_seconds=600,
        ),
        discovery=CapacityDiscovery(
            total_level1_allowance=84,
            externally_consumed_lines=7,
            tws_watchlist_lines=3,
            other_api_client_lines=4,
            current_internal_level1_lines=21,
            tick_by_tick_capacity=4,
            tick_by_tick_in_use=1,
            depth_capacity=2,
            depth_in_use=0,
            snapshot_pacing_limit=3,
            historical_requests_per_window=55,
            historical_request_window_seconds=540,
            option_computation_available=True,
            market_data_status="live",
        ),
        environment={},
        observed_at=datetime(2026, 7, 27, 14, 0, tzinfo=UTC),
        output_path=manifest_path,
    )

    assert manifest.total_level1_allowance.value == 84
    assert manifest.total_level1_allowance.source == "ibkr_discovery"
    assert manifest.externally_reserved_lines.value == 7
    assert manifest.reserved_future_trading_lines.value == 12
    assert manifest.available_research_level1_lines == 42
    assert manifest.available_tick_by_tick == 3
    assert manifest.available_depth == 2
    assert manifest.snapshot_pacing_limit.value == 3
    assert manifest.max_active_option_episodes.value == 1
    assert manifest.max_option_lines_per_episode.value == 8
    assert manifest.historical_requests_per_window.value == 55
    assert manifest.historical_request_window_seconds.value == 540
    assert manifest.option_computation_available.value is True

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["claims_boundary"] == claims_boundary()
    assert payload["reserved_future_trading_lines"]["value"] == 12
    assert payload["available_research_level1_lines"] == 42


def test_runtime_capacity_uses_exact_environment_fallback_names() -> None:
    manifest = resolve_runtime_capacity(
        settings=RuntimeCapacitySettings(),
        discovery=CapacityDiscovery(),
        environment={
            "IBKR_TOTAL_MARKET_DATA_LINES": "70",
            "IBKR_EXTERNALLY_RESERVED_LINES": "8",
            "IBKR_RESERVED_FUTURE_TRADING_LINES": "12",
            "IBKR_MAX_TICK_BY_TICK": "2",
            "IBKR_MAX_DEPTH": "1",
            "IBKR_MAX_CONCURRENT_SNAPSHOTS": "2",
            "IBKR_MAX_ACTIVE_OPTION_EPISODES": "2",
            "IBKR_MAX_OPTION_LINES_PER_EPISODE": "7",
            "IBKR_HISTORICAL_REQUESTS_PER_WINDOW": "45",
            "IBKR_HISTORICAL_REQUEST_WINDOW_SECONDS": "480",
        },
        observed_at=datetime(2026, 7, 27, 14, 0, tzinfo=UTC),
    )

    assert manifest.total_level1_allowance.value == 70
    assert manifest.total_level1_allowance.source == "configured_environment"
    assert manifest.externally_reserved_lines.value == 8
    assert manifest.reserved_future_trading_lines.value == 12
    assert manifest.tick_by_tick_capacity.value == 2
    assert manifest.depth_capacity.value == 1
    assert manifest.snapshot_pacing_limit.value == 2
    assert manifest.max_active_option_episodes.value == 2
    assert manifest.max_active_option_episodes.source == "configured_environment"
    assert manifest.max_option_lines_per_episode.value == 7
    assert manifest.max_option_lines_per_episode.source == "configured_environment"
    assert manifest.historical_requests_per_window.value == 45
    assert manifest.historical_request_window_seconds.value == 480
    assert manifest.available_research_level1_lines == 48


def test_bounded_startup_probe_observes_live_market_data_type() -> None:
    adapter = FakeIBKRAdapter(
        fixture_id="live-market-data-type-probe",
        events=(),
    )

    observed = _probe_required_market_data_type(
        adapter,
        contract=SimpleNamespace(symbol="VTI"),
        timeout_seconds=1,
    )

    assert observed is MarketDataType.LIVE
    assert adapter.connection.health().market_data_type is MarketDataType.LIVE


def test_preexisting_internal_usage_is_reserved_from_new_research_allocations() -> None:
    manager = SubscriptionBudgetManager(
        limits={SubscriptionKind.LEVEL1: 20},
        request_rate_limit=20,
        total_line_limit=40,
        externally_reserved_lines=5,
        preexisting_internal_lines=10,
        future_trading_reserve_lines=12,
        safety_margin_lines=2,
    )

    assert manager.usable_research_lines == 11
    assert manager.snapshot()["preexisting_internal_lines"] == 10
    assert manager.snapshot()["total_current_internal_usage"] == 10


def test_discovered_available_lines_raise_effective_external_reservation() -> None:
    manifest = resolve_runtime_capacity(
        settings=RuntimeCapacitySettings(),
        discovery=CapacityDiscovery(
            total_level1_allowance=100,
            available_level1_capacity=70,
            externally_consumed_lines=5,
        ),
        environment={},
        observed_at=datetime(2026, 7, 27, 14, 0, tzinfo=UTC),
    )

    assert manifest.externally_reserved_lines.value == 30
    assert manifest.externally_reserved_lines.source == ("ibkr_discovery_available_capacity")
    assert manifest.available_research_level1_lines == 56


def test_discovered_available_lines_do_not_double_count_current_internal_usage() -> None:
    manifest = resolve_runtime_capacity(
        settings=RuntimeCapacitySettings(),
        discovery=CapacityDiscovery(
            total_level1_allowance=100,
            available_level1_capacity=70,
            externally_consumed_lines=5,
            current_internal_level1_lines=10,
        ),
        environment={},
        observed_at=datetime(2026, 7, 27, 14, 0, tzinfo=UTC),
    )

    assert manifest.externally_reserved_lines.value == 20
    assert manifest.available_ordinary_level1_lines == 70
    assert manifest.available_research_level1_lines == 56
    manager = SubscriptionBudgetManager(
        limits={SubscriptionKind.BAR: 30},
        request_rate_limit=20,
        total_line_limit=int(manifest.total_level1_allowance.value),
        externally_reserved_lines=int(manifest.externally_reserved_lines.value),
        preexisting_internal_lines=manifest.current_internal_level1_lines,
        future_trading_reserve_lines=int(manifest.reserved_future_trading_lines.value),
        safety_margin_lines=int(manifest.safety_margin_lines.value),
    )
    assert manager.usable_research_lines == manifest.available_research_level1_lines


def test_windowed_request_pacer_enforces_discovered_rolling_window() -> None:
    now = [0.0]
    sleep_steps: list[float] = []
    heartbeats: list[float] = []

    def advance(seconds: float) -> None:
        sleep_steps.append(seconds)
        now[0] += seconds

    pacer = WindowedRequestPacer(
        maximum_requests=2,
        window_seconds=10,
        clock=lambda: now[0],
        sleeper=advance,
        heartbeat=lambda: heartbeats.append(now[0]),
        maximum_sleep_step_seconds=5,
    )

    pacer.acquire()
    pacer.acquire()
    pacer.acquire()

    assert sleep_steps == [5, 5]
    assert heartbeats == [0.0, 5.0]
    assert now[0] == 10.0
    assert pacer.current_window_usage == 1


def test_binding_claims_include_transfer_budget_and_no_order_boundary() -> None:
    claims = claims_boundary()
    required = {
        "research_only": True,
        "record_only": True,
        "frozen_m1c": True,
        "source_transfer_monitoring": True,
        "exact_vendor_bar_equality_required": False,
        "option_shadow_outcomes_only": True,
        "engineering_phase_sessions": 20,
        "market_data_budget_enforced": True,
        "market_data_limits_runtime_discovered": True,
        "full_option_chain_streaming_allowed": False,
        "tick_by_tick_universe_streaming_allowed": False,
        "level2_universe_streaming_allowed": False,
        "reserved_future_trading_capacity": True,
        "paper_orders_allowed": False,
        "live_orders_allowed": False,
        "order_methods_available": False,
        "account_access_required": False,
        "position_access_required": False,
        "strategy_promotion": False,
    }
    assert claims | required == claims


def test_twenty_valid_sessions_open_option_development_without_outcomes(
    tmp_path,
) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    frozen = FrozenRecorderRepository(repository)
    start = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)
    metadata = EvidenceMetadata(
        run_id="phase-gate-run",
        prospective_start_utc=start,
        app_version="test",
        git_commit="deadbeef",
        model_artifact_id="frozen-m1c",
        universe_id="anchor-frozen-20-v1",
        cohort="anchor_frozen_20",
        source_timestamps=[start.isoformat()],
        recorded_at_utc=start,
    )
    repository.create_run(metadata, mode="record_only")
    for ordinal in range(20):
        session = date(2026, 6, 1) + timedelta(days=ordinal)
        observed = start + timedelta(days=ordinal)
        session_metadata = metadata.model_copy(
            update={
                "recorded_at_utc": observed,
                "source_timestamps": [observed.isoformat()],
            }
        )
        frozen.record_source_transfer_session(
            session_metadata,
            session=session,
            valid=True,
            decision="ibkr_transfer_supported_without_recalibration",
            report={
                "ordinal": ordinal + 1,
                "claims_boundary": claims_boundary(),
            },
        )

    phase = frozen.prospective_phase_for_session(
        run_id=metadata.run_id,
        session=date(2026, 6, 21),
    )
    assert phase == ("option_development", True)
    engineering = frozen.prospective_phase_for_session(
        run_id=metadata.run_id,
        session=date(2026, 6, 20),
    )
    assert engineering == ("engineering_transfer", False)


def test_skipped_recording_idempotency_preserves_distinct_subscription_payloads(
    tmp_path,
) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    frozen = FrozenRecorderRepository(repository)
    observed = datetime(2026, 7, 27, 14, 0, tzinfo=UTC)
    metadata = EvidenceMetadata(
        run_id="skipped-recording-run",
        prospective_start_utc=observed,
        app_version="test",
        git_commit="deadbeef",
        model_artifact_id="frozen-m1c",
        universe_id="anchor-frozen-20-v1",
        cohort="anchor_frozen_20",
        source_timestamps=[observed.isoformat()],
        recorded_at_utc=observed,
    )
    repository.create_run(metadata, mode="record_only")
    shared = {
        "session": observed.date(),
        "episode_id": "episode-1",
        "symbol": "AAL",
        "recording_kind": "optional_option_stream_evicted",
        "reason": "preempted_by:OPTION_LEVEL1|99",
    }

    first_id = frozen.record_skipped_recording(
        metadata,
        requested_payload={"subscription_key": "OPTION_LEVEL1|10", "request_id": 10},
        **shared,
    )
    repeated_id = frozen.record_skipped_recording(
        metadata,
        requested_payload={"request_id": 10, "subscription_key": "OPTION_LEVEL1|10"},
        **shared,
    )
    second_id = frozen.record_skipped_recording(
        metadata,
        requested_payload={"subscription_key": "OPTION_LEVEL1|11", "request_id": 11},
        **shared,
    )

    assert repeated_id == first_id
    assert second_id != first_id
    with repository._connect() as connection:
        rows = connection.execute(
            """
            SELECT requested_payload_json
            FROM skipped_recording_v0
            WHERE run_id = ?
            ORDER BY id
            """,
            (metadata.run_id,),
        ).fetchall()
    assert [json.loads(row["requested_payload_json"]) for row in rows] == [
        {"subscription_key": "OPTION_LEVEL1|10", "request_id": 10},
        {"subscription_key": "OPTION_LEVEL1|11", "request_id": 11},
    ]


def test_daily_report_package_contains_exact_required_files(tmp_path) -> None:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    writer = BudgetAwareDailyReportWriter(
        database_path=repository.database_path,
        run_id="report-run",
        report_root=tmp_path / "reports",
    )
    package = writer.write(
        session=date(2026, 7, 27),
        generated_at=datetime(2026, 7, 27, 22, 0, tzinfo=UTC),
        capacity_manifest={
            "total_level1_allowance": {"value": 100, "source": "configured_fallback"},
            "claims_boundary": claims_boundary(),
        },
        budget_snapshot={
            "budget_state": "budget_healthy",
            "current_internal_usage": 21,
            "reserved_future_trading_lines": 12,
        },
    )

    assert tuple(path.name for path in package.files) == DAILY_REPORT_FILENAMES
    assert all(path.is_file() for path in package.files)
    with zipfile.ZipFile(package.archive_path) as archive:
        assert tuple(archive.namelist()) == DAILY_REPORT_FILENAMES
        summary = json.loads(archive.read("session_summary.json"))
    assert summary["cohort_phase"] == "engineering_transfer"
    assert summary["scientific_option_evidence"] is False
    assert summary["claims_boundary"] == claims_boundary()


def test_fake_adapter_contains_all_budget_and_episode_scenarios() -> None:
    adapter = FakeIBKRAdapter.from_fixture(BUDGET_FAKE_FIXTURE)
    required = {
        "normal_universe_operation",
        "one_quiet_episode",
        "two_simultaneous_quiet_episodes",
        "quiet_and_high_tail_conflict",
        "capacity_nearly_full_before_episode",
        "optional_lines_exhausted",
        "critical_lines_unavailable",
        "duplicate_contract_requested_twice",
        "subscription_not_released",
        "reconnect_with_active_option_episode",
        "delayed_data",
        "no_1dte_expiry",
        "missing_option_leg",
        "one_queued_neutral_control",
        "tws_external_line_use_reducing_available_budget",
    }
    assert adapter.scenarios == required
    capacity = adapter.discover_market_data_capacity()
    assert capacity.total_level1_allowance == 72
    assert capacity.tws_watchlist_lines == 6
    assert capacity.other_api_client_lines == 5

    first = FakeIBKRAdapter.from_fixture(BUDGET_FAKE_FIXTURE)
    second = FakeIBKRAdapter.from_fixture(BUDGET_FAKE_FIXTURE)
    first.connect()
    second.connect()
    first_replay = tuple(item.model_dump(mode="json") for item in first.replay())
    second_replay = tuple(item.model_dump(mode="json") for item in second.replay())
    assert first_replay == second_replay


def test_canonical_subscription_keys_are_broker_stream_identities() -> None:
    assert canonical_subscription_key(SubscriptionKind.LEVEL1, con_id=123) == "LEVEL1|123"
    assert (
        canonical_subscription_key(
            SubscriptionKind.BAR,
            con_id=123,
            bar_size="5m",
            use_rth=True,
        )
        == "BAR|123|5m|RTH"
    )
    assert (
        canonical_subscription_key(
            SubscriptionKind.TICK_BY_TICK,
            con_id=123,
            tick_type="BidAsk",
        )
        == "TBT_BIDASK|123"
    )
    assert (
        canonical_subscription_key(
            SubscriptionKind.DEPTH,
            con_id=123,
            depth_rows=5,
            smart_depth=True,
        )
        == "DEPTH|123|5|1"
    )
    assert canonical_subscription_key(SubscriptionKind.OPTION, con_id=456) == ("OPTION_LEVEL1|456")


def test_subscription_registry_deduplicates_multi_owner_streams_until_last_release() -> None:
    manager = SubscriptionBudgetManager(
        limits={SubscriptionKind.BAR: 2},
        request_rate_limit=100,
        total_line_limit=20,
        externally_reserved_lines=2,
        future_trading_reserve_lines=12,
        safety_margin_lines=1,
    )
    key = "BAR|123|5m|RTH"
    first = manager.allocate(
        key=key,
        kind=SubscriptionKind.BAR,
        symbol="AAL",
        con_id=123,
        request_id=10,
        priority=SubscriptionPriority.FROZEN_UNIVERSE_SIGNAL,
        subscription_class=SubscriptionClass.FROZEN_UNIVERSE_SIGNAL,
        owner_id="universe:AAL",
        protected=True,
        now_monotonic=1.0,
    )
    second = manager.allocate(
        key=key,
        kind=SubscriptionKind.BAR,
        symbol="AAL",
        con_id=123,
        request_id=99,
        priority=SubscriptionPriority.ACTIVE_EPISODE,
        subscription_class=SubscriptionClass.ACTIVE_EPISODE,
        owner_id="episode:quiet-1",
        protected=True,
        now_monotonic=2.0,
    )

    assert first.accepted is True
    assert second.reason == "already_active_owner_added"
    assert manager.snapshot()["current_internal_usage"] == 1
    assert manager.get(key) is not None
    assert manager.get(key).owner_count == 2  # type: ignore[union-attr]

    first_release = manager.release(
        key,
        owner_id="universe:AAL",
        reason="owner_complete",
    )
    assert first_release.cancel_upstream is False
    assert first_release.remaining_owner_count == 1
    last_release = manager.release(
        key,
        owner_id="episode:quiet-1",
        reason="episode_complete",
    )
    assert last_release.cancel_upstream is True
    assert manager.get(key) is None


def test_optional_capacity_sheds_class_five_before_class_four_and_never_uses_reserve() -> None:
    manager = SubscriptionBudgetManager(
        limits={
            SubscriptionKind.BAR: 3,
            SubscriptionKind.OPTION: 3,
            SubscriptionKind.TICK_BY_TICK: 2,
        },
        request_rate_limit=100,
        total_line_limit=12,
        externally_reserved_lines=1,
        future_trading_reserve_lines=4,
        safety_margin_lines=1,
    )
    for index, subscription_class in enumerate(
        (
            SubscriptionClass.OPTIONAL_RESEARCH,
            SubscriptionClass.OPTIONAL_RESEARCH,
            SubscriptionClass.MICROSTRUCTURE_ENHANCEMENT,
        )
    ):
        manager.allocate(
            key=f"OPTION_LEVEL1|{index + 1}",
            kind=SubscriptionKind.OPTION,
            symbol="AAL",
            con_id=index + 1,
            request_id=index + 1,
            priority=SubscriptionPriority.from_class(subscription_class),
            subscription_class=subscription_class,
            owner_id=f"optional:{index}",
            now_monotonic=float(index),
        )
    for index in range(3):
        manager.allocate(
            key=f"BAR|{100 + index}|5m|RTH",
            kind=SubscriptionKind.BAR,
            symbol=f"S{index}",
            con_id=100 + index,
            request_id=100 + index,
            priority=SubscriptionPriority.FROZEN_UNIVERSE_SIGNAL,
            subscription_class=SubscriptionClass.FROZEN_UNIVERSE_SIGNAL,
            owner_id=f"universe:{index}",
            protected=True,
            now_monotonic=10.0 + index,
        )

    decision = manager.allocate(
        key="OPTION_LEVEL1|999",
        kind=SubscriptionKind.OPTION,
        symbol="MSTR",
        con_id=999,
        request_id=999,
        priority=SubscriptionPriority.ACTIVE_EPISODE,
        subscription_class=SubscriptionClass.ACTIVE_EPISODE,
        owner_id="episode:quiet-primary",
        now_monotonic=20.0,
    )

    assert decision.accepted is True
    assert decision.evicted_keys == ("OPTION_LEVEL1|1",)
    snapshot = manager.snapshot()
    assert snapshot["current_internal_usage"] == 6
    assert snapshot["reserved_future_trading_lines"] == 4
    assert snapshot["available_research_lines"] == 0
    assert manager.get("BAR|100|5m|RTH") is not None


def test_optional_exhaustion_degrades_but_critical_exhaustion_is_explicit() -> None:
    manager = SubscriptionBudgetManager(
        limits={SubscriptionKind.BAR: 1, SubscriptionKind.OPTION: 1},
        request_rate_limit=100,
        total_line_limit=15,
        externally_reserved_lines=0,
        future_trading_reserve_lines=12,
        safety_margin_lines=2,
    )
    manager.allocate(
        key="BAR|1|5m|RTH",
        kind=SubscriptionKind.BAR,
        symbol="AAL",
        con_id=1,
        request_id=1,
        priority=SubscriptionPriority.FROZEN_UNIVERSE_SIGNAL,
        subscription_class=SubscriptionClass.FROZEN_UNIVERSE_SIGNAL,
        owner_id="universe:AAL",
        protected=True,
        now_monotonic=1.0,
    )
    optional = manager.allocate(
        key="OPTION_LEVEL1|2",
        kind=SubscriptionKind.OPTION,
        symbol="AAL",
        con_id=2,
        request_id=2,
        priority=SubscriptionPriority.OPTIONAL_RESEARCH,
        subscription_class=SubscriptionClass.OPTIONAL_RESEARCH,
        owner_id="neutral:1",
        now_monotonic=2.0,
    )
    critical = manager.allocate(
        key="BAR|3|5m|RTH",
        kind=SubscriptionKind.BAR,
        symbol="AAOI",
        con_id=3,
        request_id=3,
        priority=SubscriptionPriority.FROZEN_UNIVERSE_SIGNAL,
        subscription_class=SubscriptionClass.FROZEN_UNIVERSE_SIGNAL,
        owner_id="universe:AAOI",
        protected=True,
        now_monotonic=3.0,
    )

    assert optional.accepted is False
    assert optional.budget_state is BudgetState.OPTIONAL_FEEDS_DEGRADED
    assert optional.reason != "blocked_market_data_budget_exhausted"
    assert critical.accepted is False
    assert critical.budget_state is BudgetState.CRITICAL_BUDGET_UNAVAILABLE


def test_reconciliation_releases_stale_pending_and_reports_unknown_actual_requests() -> None:
    manager = SubscriptionBudgetManager(
        limits={SubscriptionKind.OPTION: 3},
        request_rate_limit=100,
    )
    manager.allocate(
        key="OPTION_LEVEL1|10",
        kind=SubscriptionKind.OPTION,
        symbol="AAL",
        con_id=10,
        request_id=-1,
        priority=SubscriptionPriority.EPISODE_ENGINEERING,
        subscription_class=SubscriptionClass.EPISODE_ENGINEERING,
        owner_id="episode:1",
        now_monotonic=1.0,
    )
    manager.allocate(
        key="OPTION_LEVEL1|11",
        kind=SubscriptionKind.OPTION,
        symbol="AAL",
        con_id=11,
        request_id=11,
        priority=SubscriptionPriority.ACTIVE_EPISODE,
        subscription_class=SubscriptionClass.ACTIVE_EPISODE,
        owner_id="episode:1",
        now_monotonic=2.0,
    )
    manager.mark_active("OPTION_LEVEL1|11", request_id=11)

    result = manager.reconcile(
        actual_request_ids={99},
        now_monotonic=20.0,
        pending_timeout_seconds=5.0,
    )

    assert result.released_keys == ("OPTION_LEVEL1|10", "OPTION_LEVEL1|11")
    assert result.orphan_request_ids == (99,)
    assert {warning.code for warning in result.warnings} == {
        "actual_request_without_internal_owner",
        "active_internal_request_missing_upstream",
        "stale_pending_reservation",
    }


def test_reconnect_restore_order_is_critical_then_universe_then_active_episode() -> None:
    manager = SubscriptionBudgetManager(
        limits={
            SubscriptionKind.MARKET_PROXY: 1,
            SubscriptionKind.BAR: 1,
            SubscriptionKind.OPTION: 2,
        },
        request_rate_limit=100,
        total_line_limit=20,
        future_trading_reserve_lines=12,
        safety_margin_lines=2,
    )
    rows = (
        (
            "OPTION_LEVEL1|4",
            SubscriptionKind.OPTION,
            SubscriptionClass.OPTIONAL_RESEARCH,
            None,
        ),
        (
            "OPTION_LEVEL1|3",
            SubscriptionKind.OPTION,
            SubscriptionClass.ACTIVE_EPISODE,
            "active-episode",
        ),
        (
            "BAR|2|5m|RTH",
            SubscriptionKind.BAR,
            SubscriptionClass.FROZEN_UNIVERSE_SIGNAL,
            None,
        ),
        (
            "LEVEL1|1",
            SubscriptionKind.MARKET_PROXY,
            SubscriptionClass.CRITICAL_SYSTEM,
            None,
        ),
    )
    for index, (key, kind, subscription_class, episode_id) in enumerate(rows):
        manager.allocate(
            key=key,
            kind=kind,
            symbol="AAL",
            con_id=4 - index,
            request_id=4 - index,
            priority=SubscriptionPriority.from_class(subscription_class),
            subscription_class=subscription_class,
            owner_id=f"owner:{index}",
            owner_episode=episode_id,
            protected=subscription_class <= SubscriptionClass.ACTIVE_EPISODE,
            now_monotonic=float(index),
        )

    assert manager.restore_order(active_episode_ids={"active-episode"}) == (
        "LEVEL1|1",
        "BAR|2|5m|RTH",
        "OPTION_LEVEL1|3",
        "OPTION_LEVEL1|4",
    )
    assert manager.restore_order(active_episode_ids=set()) == (
        "LEVEL1|1",
        "BAR|2|5m|RTH",
        "OPTION_LEVEL1|4",
    )


class _SubscriptionAdapter:
    def __init__(self, *, fail_tick_budget: bool = False) -> None:
        self.next_request_id = 1
        self.fail_tick_budget = fail_tick_budget
        self.requests: list[tuple[str, int]] = []
        self.cancelled: list[tuple[str, int]] = []

    def _request(self, kind: str, contract) -> int:
        if self.fail_tick_budget and kind.startswith("tick"):
            raise RuntimeError("blocked_market_data_budget_exhausted: market_data_lines")
        request_id = self.next_request_id
        self.next_request_id += 1
        self.requests.append((kind, int(contract.conId)))
        return request_id

    def request_historical_five_minute_updates(self, contract, **_kwargs) -> int:
        return self._request("bar", contract)

    def request_market_data(self, contract, **_kwargs) -> int:
        return self._request("level1", contract)

    def request_tick_by_tick(self, contract, *, tick_type, **_kwargs) -> int:
        return self._request(f"tick:{tick_type}", contract)

    def request_market_depth(self, contract, **_kwargs) -> int:
        return self._request("depth", contract)

    def cancel_historical_updates(self, request_id: int, **_kwargs) -> None:
        self.cancelled.append(("bar", request_id))

    def cancel_market_data(self, request_id: int, **_kwargs) -> None:
        self.cancelled.append(("level1", request_id))

    def cancel_tick_by_tick(self, request_id: int, **_kwargs) -> None:
        self.cancelled.append(("tick", request_id))

    def cancel_market_depth(self, request_id: int, **_kwargs) -> None:
        self.cancelled.append(("depth", request_id))


class _Normalizer:
    def __init__(self) -> None:
        self.registered: dict[int, object] = {}

    def register(self, owner) -> None:
        self.registered[owner.request_id] = owner

    def unregister(self, request_id: int) -> None:
        self.registered.pop(request_id, None)


class _SubscriptionRepository:
    def __init__(self) -> None:
        self.records: list[object] = []

    def record_subscription(self, _metadata, record) -> int:
        self.records.append(record)
        return len(self.records)

    def record_promotion_decision(self, _metadata, _decision) -> int:
        return 1


def _metadata() -> EvidenceMetadata:
    observed = datetime(2026, 7, 27, 14, 0, tzinfo=UTC)
    return EvidenceMetadata(
        run_id="budget-aware-test",
        prospective_start_utc=observed,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[observed.isoformat()],
        recorded_at_utc=observed,
    )


def test_live_controller_uses_one_always_on_bar_stream_and_optional_promotion_degrades() -> None:
    adapter = _SubscriptionAdapter(fail_tick_budget=True)
    historical_request_observations: list[int] = []
    budget = SubscriptionBudgetManager(
        limits={
            SubscriptionKind.BAR: 3,
            SubscriptionKind.LEVEL1: 1,
            SubscriptionKind.TICK_BY_TICK: 2,
            SubscriptionKind.DEPTH: 0,
        },
        request_rate_limit=100,
        total_line_limit=20,
        future_trading_reserve_lines=12,
        safety_margin_lines=2,
    )
    controller = LiveSubscriptionController(
        adapter=adapter,  # type: ignore[arg-type]
        budget=budget,
        normalizer=_Normalizer(),  # type: ignore[arg-type]
        repository=_SubscriptionRepository(),  # type: ignore[arg-type]
        depth_rows=5,
        enable_depth=False,
        historical_request_pacer=lambda: historical_request_observations.append(
            len(adapter.requests)
        ),
    )
    contracts = tuple(
        QualifiedUnderlying(
            symbol=symbol,
            con_id=con_id,
            upstream_contract=SimpleNamespace(conId=con_id, symbol=symbol),
            exchange="SMART",
            market_proxy=symbol == "VTI",
        )
        for symbol, con_id in (("AAL", 1), ("AAOI", 2), ("VTI", 3))
    )

    controller.start_always_on(_metadata(), contracts)
    controller.start_always_on(_metadata(), contracts)
    assert adapter.requests == [("bar", 1), ("bar", 2), ("bar", 3)]
    assert historical_request_observations == [0, 1, 2]
    assert budget.snapshot()["active"]["bar"] == 3  # type: ignore[index]
    assert budget.snapshot()["active"]["level1"] == 0  # type: ignore[index]

    controller.rebuild_after_data_loss(_metadata())
    assert adapter.requests == [
        ("bar", 1),
        ("bar", 2),
        ("bar", 3),
        ("bar", 3),
        ("bar", 1),
        ("bar", 2),
    ]
    assert historical_request_observations == [0, 1, 2, 3, 4, 5]
    assert budget.snapshot()["active"]["bar"] == 3  # type: ignore[index]

    promotion = controller.promote_active_episode(
        _metadata(),
        symbol="AAL",
        episode_id="quiet-1",
    )
    assert promotion.level1_started is True
    assert promotion.budget_state is BudgetState.OPTIONAL_FEEDS_DEGRADED
    assert adapter.requests.count(("level1", 1)) == 1
    assert all(budget.get(f"BAR|{con_id}|5m|RTH") is not None for con_id in (1, 2, 3))


def test_level2_is_not_admitted_during_engineering_transfer_phase() -> None:
    adapter = _SubscriptionAdapter()
    budget = SubscriptionBudgetManager(
        limits={
            SubscriptionKind.BAR: 1,
            SubscriptionKind.LEVEL1: 1,
            SubscriptionKind.TICK_BY_TICK: 2,
            SubscriptionKind.DEPTH: 1,
        },
        request_rate_limit=100,
        total_line_limit=20,
        future_trading_reserve_lines=12,
        safety_margin_lines=2,
    )
    controller = LiveSubscriptionController(
        adapter=adapter,  # type: ignore[arg-type]
        budget=budget,
        normalizer=_Normalizer(),  # type: ignore[arg-type]
        repository=_SubscriptionRepository(),  # type: ignore[arg-type]
        depth_rows=5,
        enable_depth=True,
        depth_phase_permitted=lambda _metadata: False,
    )
    contract = QualifiedUnderlying(
        symbol="AAL",
        con_id=1,
        upstream_contract=SimpleNamespace(conId=1, symbol="AAL"),
        exchange="SMART",
    )

    controller.start_always_on(_metadata(), (contract,))
    result = controller.promote_active_episode(
        _metadata(),
        symbol="AAL",
        episode_id="quiet-engineering",
    )

    assert ("depth", 1) not in adapter.requests
    assert result.budget_state is BudgetState.OPTIONAL_FEEDS_DEGRADED
    assert result.denied_keys == ("DEPTH|1|5|1",)


def test_dte_allocation_is_one_dte_first_and_never_substitutes_a_missing_primary() -> None:
    allocator = DteAllocator()
    quiet = allocator.allocate(
        episode_id="quiet-1",
        kind=EpisodeKind.QUIET,
        available=(
            DteBucket.ZERO_DTE,
            DteBucket.ONE_DTE,
            DteBucket.THREE_TO_FIVE_DTE,
        ),
        allow_secondary=True,
    )
    missing = allocator.allocate(
        episode_id="quiet-no-1dte",
        kind=EpisodeKind.QUIET,
        available=(DteBucket.ZERO_DTE, DteBucket.THREE_TO_FIVE_DTE),
        allow_secondary=True,
    )
    neutral = tuple(
        allocator.allocate(
            episode_id=f"neutral-{ordinal}",
            kind=EpisodeKind.NEUTRAL_CONTROL,
            available=(
                DteBucket.ZERO_DTE,
                DteBucket.ONE_DTE,
                DteBucket.THREE_TO_FIVE_DTE,
            ),
            allow_secondary=False,
            neutral_control_ordinal=ordinal,
        ).primary
        for ordinal in range(3)
    )

    assert quiet.primary is DteBucket.ONE_DTE
    assert quiet.secondary == (DteBucket.ZERO_DTE,)
    assert missing.primary is None
    assert missing.skipped[DteBucket.ONE_DTE.value] == "no_1dte_expiry"
    assert neutral == (
        DteBucket.ONE_DTE,
        DteBucket.ZERO_DTE,
        DteBucket.THREE_TO_FIVE_DTE,
    )


def test_snapshot_gate_never_exceeds_two_concurrent_discovery_requests() -> None:
    gate = SnapshotConcurrencyGate(max_concurrent=2)

    assert gate.reserve("snapshot-1") is True
    assert gate.reserve("snapshot-2") is True
    assert gate.reserve("snapshot-2") is True
    assert gate.reserve("snapshot-3") is False
    assert gate.active_count == 2
    assert gate.release("snapshot-1") is True
    assert gate.reserve("snapshot-3") is True
    assert gate.maximum_observed == 2


def _option_intents(prefix: int = 100) -> tuple[OptionSubscriptionIntent, ...]:
    roles_and_classes = (
        ("primary_short_call", SubscriptionClass.ACTIVE_EPISODE),
        ("primary_short_put", SubscriptionClass.ACTIVE_EPISODE),
        ("primary_long_call", SubscriptionClass.ACTIVE_EPISODE),
        ("primary_long_put", SubscriptionClass.ACTIVE_EPISODE),
        ("atm_call", SubscriptionClass.EPISODE_ENGINEERING),
        ("atm_put", SubscriptionClass.EPISODE_ENGINEERING),
        ("alternate_short_call", SubscriptionClass.EPISODE_ENGINEERING),
        ("alternate_short_put", SubscriptionClass.EPISODE_ENGINEERING),
        ("outer_call", SubscriptionClass.OPTIONAL_RESEARCH),
        ("outer_put", SubscriptionClass.OPTIONAL_RESEARCH),
    )
    return tuple(
        OptionSubscriptionIntent(
            key=f"OPTION_LEVEL1|{prefix + index}",
            con_id=prefix + index,
            role=role,
            subscription_class=subscription_class,
            required=role.startswith("primary_"),
            dte_bucket=DteBucket.ONE_DTE,
        )
        for index, (role, subscription_class) in enumerate(roles_and_classes)
    )


def test_option_state_machine_limits_lines_queues_overlap_and_persists_degradation() -> None:
    budget = SubscriptionBudgetManager(
        limits={SubscriptionKind.OPTION: 12, SubscriptionKind.BAR: 1},
        request_rate_limit=100,
        total_line_limit=30,
        future_trading_reserve_lines=12,
        safety_margin_lines=2,
    )
    budget.allocate(
        key="BAR|1|5m|RTH",
        kind=SubscriptionKind.BAR,
        symbol="AAL",
        con_id=1,
        request_id=1,
        priority=SubscriptionPriority.FROZEN_UNIVERSE_SIGNAL,
        subscription_class=SubscriptionClass.FROZEN_UNIVERSE_SIGNAL,
        owner_id="universe:AAL",
        protected=True,
        now_monotonic=0.0,
    )
    persisted = []
    machine = BudgetAwareEpisodeStateMachine(
        budget=budget,
        max_active_episodes=1,
        max_option_lines_per_episode=8,
        max_concurrent_snapshots=2,
        persistence_sink=persisted.append,
    )
    now = datetime(2026, 7, 27, 14, 0, tzinfo=UTC)
    quiet = machine.submit(
        OptionEpisodeTask(
            episode_id="quiet-1",
            symbol="AAL",
            kind=EpisodeKind.QUIET,
            probability=0.10,
            triggered_at_utc=now,
            useful_until_utc=now.replace(hour=15),
            requested_subscriptions=_option_intents(),
        ),
        now=now,
    )
    high = machine.submit(
        OptionEpisodeTask(
            episode_id="high-1",
            symbol="AAOI",
            kind=EpisodeKind.HIGH_TAIL,
            probability=0.75,
            triggered_at_utc=now,
            useful_until_utc=now.replace(hour=15),
            requested_subscriptions=_option_intents(200)[:2],
        ),
        now=now,
    )

    assert quiet.state is EpisodeState.DEGRADED
    assert len(quiet.approved_subscriptions) == 8
    assert quiet.denied_subscriptions == ("OPTION_LEVEL1|108", "OPTION_LEVEL1|109")
    assert quiet.cohort_phase == "engineering_transfer"
    assert quiet.scientific_option_evidence is False
    assert high.state is EpisodeState.EPISODE_QUEUED
    assert high.queued_subscriptions == ("OPTION_LEVEL1|200", "OPTION_LEVEL1|201")
    assert budget.get("BAR|1|5m|RTH") is not None
    machine.fail_request(
        "quiet-1",
        key="OPTION_LEVEL1|100",
        now=now,
        reason="simulated_required_leg_failure",
    )
    assert machine.snapshot()["budget_state"] == (
        BudgetState.OPTION_EPISODE_PARTIALLY_RECORDED.value
    )

    machine.complete("quiet-1", now=now)
    machine.poll(now=now)
    assert machine.record("high-1").state is EpisodeState.PRIMARY_LEGS_STREAMING
    assert machine.active_episode_ids == ("high-1",)
    assert persisted
    assert all(record.capacity_before and record.capacity_after for record in persisted)


def test_official_client_facade_hides_inseparable_order_methods() -> None:
    class RawOfficialClient:
        def __init__(self) -> None:
            self.requests: list[tuple[object, ...]] = []

        def reqMktData(self, *arguments: object) -> None:  # noqa: N802
            self.requests.append(arguments)

        def placeOrder(self, *_arguments: object) -> None:  # noqa: N802
            raise AssertionError("order method must never be reachable")

    raw = RawOfficialClient()
    facade = OfficialMarketDataOnlyClient(raw)

    facade.reqMktData(17, "bounded-contract")

    assert raw.requests == [(17, "bounded-contract")]
    assert not hasattr(facade, "placeOrder")
    assert not hasattr(facade, "cancelOrder")
    assert not hasattr(facade, "reqAccountSummary")
    assert not hasattr(facade, "reqPositions")


def test_official_field_value_callbacks_merge_into_valid_option_snapshot() -> None:
    merged = merge_snapshot_items(
        (
            {"field": "bid", "value": 1.20, "market_data_type": "live"},
            {"field": "ask", "value": 1.30, "market_data_type": "live"},
            {
                "field": "option_computation",
                "delta": 0.25,
                "implied_volatility": 0.31,
                "market_data_type": "live",
            },
        )
    )

    assert merged["bid"] == 1.20
    assert merged["ask"] == 1.30
    assert merged["delta"] == 0.25
    assert merged["implied_volatility"] == 0.31
    assert "field" not in merged
    assert "value" not in merged


def test_quiet_atm_selection_uses_roles_instead_of_first_condor_legs() -> None:
    expiry = date(2026, 7, 28)

    def option(
        con_id: int,
        strike: float,
        right: Literal["C", "P"],
    ) -> OptionContract:
        return OptionContract(
            underlying_con_id=1,
            con_id=con_id,
            expiry=expiry,
            dte=1,
            dte_bucket=DteBucket.ONE_DTE,
            strike=strike,
            right=right,
            multiplier=100,
            exchange="SMART",
            trading_class="AAL",
        )

    short_call = option(101, 105.0, "C")
    short_put = option(102, 95.0, "P")
    atm_call = option(103, 100.0, "C")
    atm_put = option(104, 100.0, "P")
    planned = (short_call, short_put, atm_call, atm_put)
    roles = {
        short_call.con_id_key: ("primary_short_call_025_delta",),
        short_put.con_id_key: ("primary_short_put_minus_025_delta",),
        atm_call.con_id_key: ("comparison_atm_call",),
        atm_put.con_id_key: ("comparison_atm_put",),
    }

    assert _planned_atm_contract(planned, selection_roles=roles, right="C") is atm_call
    assert _planned_atm_contract(planned, selection_roles=roles, right="P") is atm_put


def test_episode_allocation_cancels_every_evicted_broker_stream() -> None:
    now = datetime(2026, 7, 27, 14, 0, tzinfo=UTC)
    manager = SubscriptionBudgetManager(
        limits={SubscriptionKind.OPTION: 2},
        request_rate_limit=20,
        total_line_limit=20,
        future_trading_reserve_lines=12,
    )
    existing_key = canonical_subscription_key(SubscriptionKind.OPTION, con_id=900)
    assert manager.allocate(
        key=existing_key,
        kind=SubscriptionKind.OPTION,
        symbol="CONTROL",
        con_id=900,
        request_id=900,
        priority=SubscriptionPriority.OPTIONAL_RESEARCH,
        owner_id="control:900",
        subscription_class=SubscriptionClass.OPTIONAL_RESEARCH,
        protected=False,
        now_monotonic=1.0,
        now_utc=now,
    ).accepted
    evictions: list[tuple[str, str]] = []
    machine = BudgetAwareEpisodeStateMachine(
        budget=manager,
        max_active_episodes=1,
        max_option_lines_per_episode=8,
        eviction_sink=lambda evicted, replacement, _observed: (
            evictions.append((evicted, replacement)) or True
        ),
    )
    requested = tuple(
        OptionSubscriptionIntent(
            key=canonical_subscription_key(SubscriptionKind.OPTION, con_id=con_id),
            con_id=con_id,
            role=f"primary_leg_{con_id}",
            subscription_class=SubscriptionClass.ACTIVE_EPISODE,
            required=True,
            dte_bucket=DteBucket.ONE_DTE,
        )
        for con_id in (901, 902)
    )
    record = machine.submit(
        OptionEpisodeTask(
            episode_id="quiet-eviction",
            symbol="AAL",
            kind=EpisodeKind.QUIET,
            probability=0.10,
            triggered_at_utc=now,
            useful_until_utc=now + timedelta(minutes=60),
            requested_subscriptions=requested,
        ),
        now=now,
    )

    assert record.state is EpisodeState.PRIMARY_LEGS_STREAMING
    assert evictions == [(existing_key, requested[1].key)]
    assert manager.get(existing_key) is None


def _transfer_observations(
    *,
    ibkr_transform=lambda value: value,
) -> tuple[tuple[ProviderM1CObservation, ...], tuple[ProviderM1CObservation, ...]]:
    ibkr = []
    eodhd = []
    probabilities = (0.10, 0.13, 0.20, 0.50, 0.70)
    for session_index in range(20):
        session = date(2026, 1, 1) + timedelta(days=session_index)
        previous_ibkr = None
        previous_eodhd = None
        for checkpoint_index, probability in enumerate(probabilities):
            checkpoint = 6 + checkpoint_index * 2
            bar_start = datetime.combine(
                session,
                datetime.min.time(),
                tzinfo=UTC,
            ) + timedelta(hours=14, minutes=checkpoint * 5)
            ibkr_probability = float(ibkr_transform(probability))
            for destination, provider, observed_probability, previous in (
                (eodhd, "eodhd", probability, previous_eodhd),
                (ibkr, "ibkr", ibkr_probability, previous_ibkr),
            ):
                quiet = observed_probability <= 0.135896965695626
                high = observed_probability >= 0.488333710794033
                destination.append(
                    ProviderM1CObservation(
                        provider=provider,
                        symbol="AAL",
                        session=session,
                        checkpoint=checkpoint,
                        bar=TransferBar(
                            identity=f"{provider}:{session}:{checkpoint}",
                            start_utc=bar_start,
                            end_utc=bar_start + timedelta(minutes=5),
                            open=100.0,
                            high=101.0 if provider == "eodhd" else 101.01,
                            low=99.0,
                            close=100.5 if provider == "eodhd" else 100.51,
                            complete=True,
                        ),
                        features={"f1": observed_probability, "f2": checkpoint / 10.0},
                        probability=observed_probability,
                        quiet_episode=quiet and previous is not True,
                        high_tail_episode=high and previous is not True,
                    )
                )
            previous_eodhd = probability <= 0.135896965695626
            previous_ibkr = ibkr_probability <= 0.135896965695626
    return tuple(ibkr), tuple(eodhd)


def test_transfer_monitor_accepts_nonidentical_bars_when_ranking_and_tails_transfer() -> None:
    ibkr, eodhd = _transfer_observations(ibkr_transform=lambda value: value + 0.001)
    report = M1CTransferMonitor(
        robust_feature_scales={"f1": 0.1, "f2": 1.0},
        feature_coefficients={"f1": 2.0, "f2": 0.1},
    ).evaluate(
        ibkr=ibkr,
        eodhd=eodhd,
        runtime_parity_passed=True,
    )

    assert report.valid_session_count == 20
    assert report.bar_semantics_passed is True
    assert report.exact_vendor_bar_equality_required is False
    assert report.probability_metrics.spearman == 1.0
    assert report.tail_metrics.bottom_10_agreement == 1.0
    assert report.decision == "ibkr_transfer_supported_without_recalibration"
    assert report.bar_comparisons[0].high_absolute_difference == pytest.approx(0.01)


def test_transfer_monitor_detects_rank_preserving_scale_shift_and_v1_uses_no_outcomes() -> None:
    ibkr, eodhd = _transfer_observations(ibkr_transform=lambda value: 0.08 + 0.75 * value)
    report = M1CTransferMonitor().evaluate(
        ibkr=ibkr,
        eodhd=eodhd,
        runtime_parity_passed=True,
    )
    candidate = create_ibkr_calibration_candidate(report=report, ibkr=ibkr)

    assert report.probability_metrics.spearman == 1.0
    assert report.decision == "ibkr_ranking_supported_probability_scale_shifted"
    assert candidate.candidate_id == "M1C_IBKR_CALIBRATION_V1_CANDIDATE"
    assert candidate.source == "ibkr_probability_distribution_only"
    assert len(candidate.source_valid_sessions) == 20
    assert candidate.source_observation_count == len(ibkr)
    assert candidate.outcome_fields_used == ()
    assert candidate.option_pnl_used is False
    assert candidate.thresholds["bottom_10"] < candidate.thresholds["bottom_20"]

    invalid_extra_session = (
        *ibkr,
        ProviderM1CObservation(
            **{
                **ibkr[0].__dict__,
                "session": date(2026, 2, 1),
            }
        ),
    )
    with pytest.raises(ValueError, match="exactly twenty valid sessions"):
        create_ibkr_calibration_candidate(
            report=report,
            ibkr=invalid_extra_session,
        )


def test_transfer_monitor_blocks_before_twenty_valid_sessions_and_on_bar_semantics() -> None:
    ibkr, eodhd = _transfer_observations()
    short = M1CTransferMonitor().evaluate(
        ibkr=ibkr[:25],
        eodhd=eodhd[:25],
        runtime_parity_passed=True,
    )
    broken_row = ProviderM1CObservation(
        **{
            **ibkr[0].__dict__,
            "bar": TransferBar(
                **{
                    **ibkr[0].bar.__dict__,
                    "end_utc": ibkr[0].bar.start_utc + timedelta(minutes=4),
                }
            ),
        }
    )
    broken = M1CTransferMonitor().evaluate(
        ibkr=(broken_row, *ibkr[1:]),
        eodhd=eodhd,
        runtime_parity_passed=True,
    )

    assert short.decision == "blocked_insufficient_valid_sessions"
    assert broken.decision == "blocked_bar_semantics_failure"
