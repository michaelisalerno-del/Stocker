from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_core.strategies import (
    SESSION_HARD_HV_METHOD,
    StrategyDefinition,
)
from stocker_execution.session_hard_structure_d import (
    SESSION_HARD_THRESHOLD,
    CohortOpportunity,
    EntryBar,
    PreMoveBand,
    SessionHardAssessment,
    SessionHardStructureDStrategy,
    SignalStatus,
    StrategyContext,
    StrategyOpportunityKey,
    calculate_cohort_percentile,
    calculate_session_hard_score,
    classify_pre_move_band,
)
from stocker_execution.stage5 import Stage5FeatureSnapshot, Stage5Status
from stocker_execution.stage6_diagnostic import (
    evaluate_strategy_fixture,
    format_strategy_diagnostic,
)
from stocker_execution.stage6_diagnostic import (
    main as diagnostic_main,
)
from stocker_execution.strategy_factory import create_strategy


def ready_snapshot(*, pre_move_m: float, con_id: int = 101) -> Stage5FeatureSnapshot:
    return Stage5FeatureSnapshot(
        run_ids=("RUN_A",),
        universe_id="BROAD",
        con_id=con_id,
        symbol=f"S{con_id}",
        session=date(2025, 2, 20),
        t0=datetime(2025, 2, 20, 15, 0, tzinfo=UTC),
        status=Stage5Status.READY,
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
        calculation_version="PRE_MOVE_M_V1",
    )


def strategy_context(
    feature_rows: tuple[Stage5FeatureSnapshot, ...],
    scores: dict[int, float],
    *,
    run_id: str = "RUN_A",
    cohort_history: tuple[CohortOpportunity, ...] = (),
    checkpoints: dict[int, int] | None = None,
) -> StrategyContext:
    checkpoint_by_con_id = checkpoints or {}
    return StrategyContext(
        run_id=run_id,
        session_hard={
            StrategyOpportunityKey(row.con_id, row.session, row.t0): SessionHardAssessment(
                score=scores[row.con_id],
                checkpoint=checkpoint_by_con_id.get(row.con_id, 6),
            )
            for row in feature_rows
            if row.con_id is not None and row.con_id in scores
        },
        cohort_history=cohort_history,
    )


def test_frozen_session_hard_score_matches_accepted_row() -> None:
    features = {
        "range_effort": 7.143007947913424,
        "travel_effort": 6.644027797475038,
        "absolute_efficiency": 0.381144089051591,
        "close_retention": 0.223804375985926,
        "directional_persistence": 0.666666666666667,
        "prior_6_mean_range": 210.704735376659528,
        "prior_6_price_travel": 767.182855168199239,
        "prior_6_absolute_net_movement": 292.407210469082258,
        "recent_vs_earlier_range_ratio": 0.678818073766329,
        "current_bar_range_vs_prior_6": 0.857381500683692,
        "current_bar_body_fraction": 0.714418303577804,
        "current_bar_extreme_wick_fraction": 0.285581696422196,
        "current_volume_vs_session_mean": 0.417960863585287,
        "current_volume_vs_prior6_mean": 0.374380090654236,
        "last3_vs_first3_volume": 0.297377541611083,
    }

    score = calculate_session_hard_score(checkpoint=6, features=features)

    assert score == pytest.approx(0.999636387501572, abs=5e-15)


def test_frozen_unseen49_hood_percentile_and_band_match_accepted_ledger() -> None:
    history = (
        1.13307437950998,
        1.01638600994221,
        0.520139320403003,
        1.05290217408396,
        0.653603476926764,
        1.11373459162757,
        1.19819684673582,
        1.02624385971943,
        0.714653207954931,
        0.6466291343547,
        0.559767962323959,
        1.64639951326801,
        1.46564328277018,
        1.0116985052533,
        0.867104005963024,
        0.733777192523018,
        0.501895931066927,
        1.21719454641158,
        1.35691004079688,
        1.25810973323133,
        0.521349556474786,
        0.564017288969056,
        0.955582054581471,
        1.12212665421029,
        0.994529262286479,
        0.821932221300254,
        0.547724099764726,
        2.08020245737097,
        1.19497626428145,
        0.571842409121946,
        0.681553698006495,
        0.517062567837534,
        0.812194056764132,
        0.81896515840008,
        0.865079403128148,
        1.04477944085885,
        0.924755484420779,
        0.913129871699216,
        0.558624762706318,
        1.04999576955118,
        0.933354802446138,
        0.982243889734966,
        0.983948377863543,
        0.596856827982767,
        0.485588060171765,
        1.95962662660837,
        1.18532362930428,
        0.82524187326742,
    )

    percentile = calculate_cohort_percentile(0.784827092404623, history)

    assert percentile == pytest.approx(33.3333333333333, abs=5e-14)
    assert classify_pre_move_band(percentile) is PreMoveBand.MID


def test_exact_pre_move_threshold_is_not_qualified() -> None:
    snapshot = ready_snapshot(pre_move_m=0.475764059845861)
    strategy = SessionHardStructureDStrategy()

    signals = strategy.evaluate(
        (snapshot,),
        strategy_context((snapshot,), {101: SESSION_HARD_THRESHOLD}),
    )

    assert len(signals) == 1
    assert signals[0].status is SignalStatus.NOT_QUALIFIED
    assert signals[0].reason == "PRE_MOVE_THRESHOLD"
    assert snapshot.pre_move_m == 0.475764059845861


@pytest.mark.parametrize("method", (SESSION_HARD_HV_METHOD,))
def test_current_method_reuses_frozen_qualification_and_keeps_current_identity(
    method: StrategyDefinition,
) -> None:
    snapshot = ready_snapshot(pre_move_m=0.475764059845861)
    strategy = create_strategy(method.strategy_id, method.strategy_version)

    signal = strategy.evaluate(
        (snapshot,), strategy_context((snapshot,), {101: SESSION_HARD_THRESHOLD})
    )[0]

    assert signal.strategy_id == method.strategy_id
    assert signal.strategy_version == method.strategy_version
    assert signal.reason == "PRE_MOVE_THRESHOLD"
    assert signal.stop_distance_m == 0.50
    assert signal.target_distance_m == 1.00

    restored = create_strategy(method.strategy_id, method.strategy_version)
    restored.restore_signals((signal,))
    assert restored.signals == (signal,)


@pytest.mark.parametrize("method", (SESSION_HARD_HV_METHOD,))
def test_known_mid_band_is_vetoed_from_strategy_entry(method: StrategyDefinition) -> None:
    start = date(2025, 1, 1)
    history = tuple(
        CohortOpportunity(
            run_id="RUN_A",
            session=start + timedelta(days=index % 20),
            pre_move_m=value,
        )
        for index, value in enumerate((1.0,) * 15 + (2.0,) * 15)
    )
    snapshot = ready_snapshot(pre_move_m=1.0)

    signal = create_strategy(method.strategy_id, method.strategy_version).evaluate(
        (snapshot,),
        strategy_context(
            (snapshot,),
            {101: SESSION_HARD_THRESHOLD},
            cohort_history=history,
        ),
    )[0]

    assert signal.cohort_percentile == 50.0
    assert signal.band is PreMoveBand.MID
    assert signal.status is SignalStatus.NOT_QUALIFIED
    assert signal.reason == "COHORT_MID_VETO"


@pytest.mark.parametrize("method", (SESSION_HARD_HV_METHOD,))
def test_frozen_glw_down_first_touch_emits_short_intention_at_level(
    method: StrategyDefinition,
) -> None:
    p0 = 52.150001
    m_price = 0.275429305778967
    snapshot = replace(
        ready_snapshot(pre_move_m=1.74273397613477, con_id=202),
        symbol="GLW",
        p0=p0,
        m_price=m_price,
    )
    history = tuple(
        CohortOpportunity("RUN_A", date(2025, 1, 1) + timedelta(days=index % 20), 0.6)
        for index in range(30)
    )
    strategy = SessionHardStructureDStrategy()
    waiting = strategy.evaluate(
        (snapshot,),
        strategy_context(
            (snapshot,),
            {202: 0.999976342006068},
            cohort_history=history,
        ),
    )[0]

    triggered = strategy.observe_entry_bars(
        {
            202: (
                EntryBar(
                    timestamp=snapshot.t0,
                    open=52.150001000000003,
                    high=52.200001000958778,
                    low=51.860000994439119,
                ),
            )
        }
    )[0]

    assert waiting.status is SignalStatus.WAITING_FOR_ENTRY
    assert triggered.status is SignalStatus.ENTRY_TRIGGERED
    assert triggered.direction == "DOWN"
    assert triggered.side == "SHORT"
    assert triggered.entry_level == pytest.approx(p0 - 0.20 * m_price)
    assert triggered.entry_reference == pytest.approx(p0 - 0.20 * m_price)
    assert triggered.entry_timestamp == snapshot.t0
    assert triggered.signal_timestamp == snapshot.t0 + timedelta(minutes=1)
    assert triggered.stop_distance_m == 0.50
    assert triggered.target_distance_m == 1.00


@pytest.mark.parametrize("method", (SESSION_HARD_HV_METHOD,))
def test_simultaneous_candidates_rank_by_session_hard_score_and_cap_at_five(
    method: StrategyDefinition,
) -> None:
    strategy = SessionHardStructureDStrategy()
    snapshots = tuple(
        replace(
            ready_snapshot(pre_move_m=0.8 + index / 100, con_id=300 + index),
            symbol=chr(ord("A") + index),
        )
        for index in range(7)
    )
    scores = {300 + index: 0.9994 + index / 100_000 for index in range(7)}
    strategy.evaluate(
        snapshots,
        strategy_context(snapshots, scores),
    )
    bars = {
        300 + index: (
            EntryBar(
                timestamp=snapshots[index].t0,
                open=100.0,
                high=100.0,
                low=99.0,
            ),
        )
        for index in range(7)
    }

    results = strategy.observe_entry_bars(bars)

    selected = [signal for signal in results if signal.status is SignalStatus.ENTRY_TRIGGERED]
    excluded = [signal for signal in results if signal.reason == "STRATEGY_CANDIDATE_CAPACITY"]
    assert [(signal.symbol, signal.candidate_rank) for signal in selected] == [
        ("G", 1),
        ("F", 2),
        ("E", 3),
        ("D", 4),
        ("C", 5),
    ]
    assert [(signal.symbol, signal.candidate_rank) for signal in excluded] == [
        ("B", 6),
        ("A", 7),
    ]


def test_split_simultaneous_observations_wait_then_apply_one_global_cap() -> None:
    snapshots = tuple(
        replace(
            ready_snapshot(pre_move_m=0.8, con_id=700 + index),
            symbol=chr(ord("A") + index),
        )
        for index in range(7)
    )
    scores = {700 + index: 0.9994 + index / 100_000 for index in range(7)}
    bars = {
        snapshot.con_id: (EntryBar(snapshot.t0, 100.0, 100.0, 99.0),)
        for snapshot in snapshots
        if snapshot.con_id is not None
    }
    strategy = SessionHardStructureDStrategy()
    strategy.evaluate(snapshots, strategy_context(snapshots, scores))

    first = strategy.observe_entry_bars(dict(tuple(bars.items())[:3]))
    second = strategy.observe_entry_bars(dict(tuple(bars.items())[3:]))

    assert not any(signal.selected for signal in first)
    assert sum(signal.selected for signal in second) == 5
    selected = [signal for signal in strategy.signals if signal.selected]
    assert [(signal.symbol, signal.candidate_rank) for signal in selected] == [
        ("C", 5),
        ("D", 4),
        ("E", 3),
        ("F", 2),
        ("G", 1),
    ]


def test_pending_ranking_finalizes_when_missing_candidate_reports_no_touch() -> None:
    triggered = replace(ready_snapshot(pre_move_m=0.8, con_id=750), symbol="TRIGGERED")
    no_touch = replace(ready_snapshot(pre_move_m=0.8, con_id=751), symbol="NO_TOUCH")
    snapshots = (triggered, no_touch)
    strategy = SessionHardStructureDStrategy()
    strategy.evaluate(
        snapshots,
        strategy_context(
            snapshots,
            {750: SESSION_HARD_THRESHOLD, 751: SESSION_HARD_THRESHOLD},
        ),
    )

    incomplete = strategy.observe_entry_bars({750: (EntryBar(triggered.t0, 100.0, 100.0, 99.0),)})
    finalized = strategy.observe_entry_bars({751: (EntryBar(no_touch.t0, 100.0, 100.1, 99.9),)})

    assert not any(signal.selected for signal in incomplete)
    selected = [signal for signal in finalized if signal.selected]
    assert [(signal.symbol, signal.candidate_rank) for signal in selected] == [("TRIGGERED", 1)]


def test_candidate_capacity_is_independent_for_distinct_runs() -> None:
    run_a = tuple(
        replace(
            ready_snapshot(pre_move_m=0.8, con_id=800 + index),
            run_ids=("RUN_A",),
            symbol=f"A{index}",
        )
        for index in range(6)
    )
    run_b = tuple(
        replace(
            ready_snapshot(pre_move_m=0.8, con_id=900 + index),
            run_ids=("RUN_B",),
            symbol=f"B{index}",
        )
        for index in range(6)
    )
    strategy = SessionHardStructureDStrategy()
    strategy.evaluate(
        run_a,
        strategy_context(run_a, {800 + index: 0.9995 + index / 100_000 for index in range(6)}),
    )
    strategy.evaluate(
        run_b,
        strategy_context(
            run_b,
            {900 + index: 0.9995 + index / 100_000 for index in range(6)},
            run_id="RUN_B",
        ),
    )
    all_snapshots = (*run_a, *run_b)

    strategy.observe_entry_bars(
        {
            snapshot.con_id: (EntryBar(snapshot.t0, 100.0, 100.0, 99.0),)
            for snapshot in all_snapshots
            if snapshot.con_id is not None
        }
    )

    for run_id in ("RUN_A", "RUN_B"):
        run_signals = [signal for signal in strategy.signals if signal.run_id == run_id]
        assert sum(signal.selected for signal in run_signals) == 5
        assert sorted(signal.candidate_rank for signal in run_signals if signal.selected) == [
            1,
            2,
            3,
            4,
            5,
        ]


def test_frozen_broad2025_competition_reproduces_top_session_hard_five() -> None:
    # Frozen 2025-02-20 15:00 UTC group from enriched_context_ledger.csv.
    rows = (
        ("RCL", 2.33931879964265, 0.999994799489345, 239.880004, 0.741993505076624),
        ("HOOD", 1.64639951326801, 0.999952450245927, 55.549999, 0.321914569451631),
        ("CRWD", 3.3804147858686, 0.999896972702262, 420.945007, 1.50496373617038),
        ("VST", 0.579240829402579, 0.999858531698062, 159.770004, 1.44275054261026),
        ("UAL", 0.966793627813105, 0.999770810267955, 102.214996, 0.47062782490986),
        ("NRG", 0.698801697520803, 0.9997261026453, 108.61, 0.493702292401384),
        ("WST", 0.654770787364182, 0.999679797227071, 208.0, 0.717808443916704),
        ("PANW", 2.77554363451459, 0.9996407016144, 195.925003, 0.59627959177474),
        ("ADSK", 0.877205199014426, 0.99938456540693, 292.679992, 1.57910596482309),
    )
    snapshots = tuple(
        replace(
            ready_snapshot(pre_move_m=pre_move_m, con_id=600 + index),
            symbol=symbol,
            p0=p0,
            m_price=m_price,
        )
        for index, (symbol, pre_move_m, _, p0, m_price) in enumerate(rows)
    )
    strategy = SessionHardStructureDStrategy()
    strategy.evaluate(
        snapshots,
        strategy_context(
            snapshots,
            {600 + index: score for index, (_, _, score, _, _) in enumerate(rows)},
        ),
    )

    signals = strategy.observe_entry_bars(
        {
            snapshot.con_id: (
                EntryBar(
                    snapshot.t0,
                    snapshot.p0,
                    snapshot.p0,
                    snapshot.p0 - snapshot.m_price,
                ),
            )
            for snapshot in snapshots
            if snapshot.con_id is not None
            and snapshot.p0 is not None
            and snapshot.m_price is not None
        }
    )

    assert [signal.symbol for signal in signals if signal.selected] == [
        "RCL",
        "HOOD",
        "CRWD",
        "VST",
        "UAL",
    ]
    assert [signal.symbol for signal in signals if not signal.selected] == [
        "NRG",
        "WST",
        "PANW",
        "ADSK",
    ]


def test_mid_veto_down_touch_still_enters_future_percentile_cohort() -> None:
    start = date(2025, 1, 1)
    seed = tuple(
        CohortOpportunity(
            "RUN_A",
            start + timedelta(days=index % 20),
            1.0 if index < 15 else 2.0,
        )
        for index in range(30)
    )
    strategy = SessionHardStructureDStrategy()
    snapshot = ready_snapshot(pre_move_m=1.0)
    signal = strategy.evaluate(
        (snapshot,),
        strategy_context(
            (snapshot,),
            {101: SESSION_HARD_THRESHOLD},
            cohort_history=seed,
        ),
    )[0]

    strategy.observe_entry_bars(
        {
            101: (
                EntryBar(
                    timestamp=snapshot.t0,
                    open=100.0,
                    high=100.0,
                    low=99.0,
                ),
            )
        }
    )

    assert signal.reason == "COHORT_MID_VETO"
    assert strategy.cohort_opportunities == (CohortOpportunity("RUN_A", snapshot.session, 1.0),)


def test_frozen_glw_fixture_reproduces_percentile_band_and_entry() -> None:
    fixture = Path("tests/fixtures/stage6_glw_golden.json")

    signals = evaluate_strategy_fixture(fixture)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.symbol == "GLW"
    assert signal.cohort_percentile == pytest.approx(91.4285714285714)
    assert signal.band is PreMoveBand.HIGH
    assert signal.candidate_rank == 1
    assert signal.status is SignalStatus.ENTRY_TRIGGERED
    assert signal.entry_reference == pytest.approx(52.150001 - 0.20 * 0.275429305778967)
    diagnostic = format_strategy_diagnostic(signals)
    assert "GLW 1.74273397613 91.4285714286 HIGH YES DOWN 1 YES" in diagnostic
    assert "ENTRY_TRIGGERED" in diagnostic


@pytest.mark.parametrize(
    ("percentile", "expected"),
    [
        (33.33, PreMoveBand.LOW),
        (33.3300000001, PreMoveBand.MID),
        (66.67, PreMoveBand.MID),
        (66.6700000001, PreMoveBand.HIGH),
    ],
)
def test_production_band_boundaries_are_exact(percentile: float, expected: PreMoveBand) -> None:
    assert classify_pre_move_band(percentile) is expected


def test_cohort_uses_same_run_prior_twenty_distinct_sessions_only() -> None:
    start = date(2025, 1, 1)
    history = [
        CohortOpportunity("RUN_A", start, 0.5),
        CohortOpportunity("RUN_A", start, 0.5),
    ]
    for index in range(1, 21):
        history.extend(
            (
                CohortOpportunity("RUN_A", start + timedelta(days=index), 0.5),
                CohortOpportunity("RUN_A", start + timedelta(days=index), 2.0),
            )
        )
    history.extend(
        (
            CohortOpportunity("RUN_B", start + timedelta(days=20), 0.5),
            CohortOpportunity("RUN_A", date(2025, 2, 20), 0.5),
        )
    )

    snapshot = ready_snapshot(pre_move_m=1.0)
    signal = SessionHardStructureDStrategy().evaluate(
        (snapshot,),
        strategy_context(
            (snapshot,),
            {101: SESSION_HARD_THRESHOLD},
            cohort_history=tuple(history),
        ),
    )[0]

    assert signal.cohort_percentile == 50.0
    assert signal.band is PreMoveBand.MID


def test_session_hard_threshold_is_inclusive_but_lower_score_fails() -> None:
    snapshots = (
        ready_snapshot(pre_move_m=0.8, con_id=401),
        ready_snapshot(pre_move_m=0.8, con_id=402),
    )

    signals = SessionHardStructureDStrategy().evaluate(
        snapshots,
        strategy_context(
            snapshots,
            {
                401: SESSION_HARD_THRESHOLD,
                402: SESSION_HARD_THRESHOLD - 1e-12,
            },
        ),
    )

    assert signals[0].status is SignalStatus.WAITING_FOR_ENTRY
    assert signals[1].status is SignalStatus.NOT_QUALIFIED
    assert signals[1].reason == "SESSION_HARD"


def test_session_hard_assessment_is_keyed_by_con_id_session_and_t0() -> None:
    first = ready_snapshot(pre_move_m=0.8, con_id=450)
    second = replace(first, t0=first.t0 + timedelta(minutes=10))
    context = StrategyContext(
        run_id="RUN_A",
        session_hard={
            StrategyOpportunityKey(first.con_id, first.session, first.t0): SessionHardAssessment(
                SESSION_HARD_THRESHOLD,
                6,
            ),
            StrategyOpportunityKey(second.con_id, second.session, second.t0): SessionHardAssessment(
                SESSION_HARD_THRESHOLD - 1e-12,
                8,
            ),
        },
    )

    signals = SessionHardStructureDStrategy().evaluate((first, second), context)

    assert signals[0].session_hard_checkpoint == 6
    assert signals[0].status is SignalStatus.WAITING_FOR_ENTRY
    assert signals[1].session_hard_checkpoint == 8
    assert signals[1].reason == "SESSION_HARD"


def test_invalid_checkpoint_and_score_are_isolated_to_their_candidates() -> None:
    invalid_checkpoint = ready_snapshot(pre_move_m=0.8, con_id=460)
    invalid_score = ready_snapshot(pre_move_m=0.8, con_id=461)
    valid = ready_snapshot(pre_move_m=0.8, con_id=462)
    snapshots = (invalid_checkpoint, invalid_score, valid)

    signals = SessionHardStructureDStrategy().evaluate(
        snapshots,
        strategy_context(
            snapshots,
            {
                460: SESSION_HARD_THRESHOLD,
                461: 1.01,
                462: SESSION_HARD_THRESHOLD,
            },
            checkpoints={460: 7},
        ),
    )

    assert signals[0].reason == "SESSION_HARD_CHECKPOINT"
    assert signals[1].reason == "SESSION_HARD_SCORE_UNAVAILABLE"
    assert signals[2].status is SignalStatus.WAITING_FOR_ENTRY


def test_invalid_stage5_candidate_does_not_stop_valid_candidate() -> None:
    invalid = replace(
        ready_snapshot(pre_move_m=0.8, con_id=501),
        status=Stage5Status.PRE_MOVE_NOT_READY,
        exclusion_reason="PRE_MOVE_NOT_READY",
        pre_move_m=None,
        p0=None,
        m_price=None,
    )
    valid = ready_snapshot(pre_move_m=0.8, con_id=502)

    signals = SessionHardStructureDStrategy().evaluate(
        (invalid, valid),
        strategy_context((invalid, valid), {502: SESSION_HARD_THRESHOLD}),
    )

    assert [(signal.underlying_con_id, signal.status) for signal in signals] == [
        (501, SignalStatus.NOT_QUALIFIED),
        (502, SignalStatus.WAITING_FOR_ENTRY),
    ]


def test_malformed_ready_values_are_rejected_without_stopping_batch() -> None:
    nan_pre = replace(ready_snapshot(pre_move_m=0.8, con_id=510), pre_move_m=float("nan"))
    infinite_p0 = replace(ready_snapshot(pre_move_m=0.8, con_id=511), p0=float("inf"))
    zero_m = replace(ready_snapshot(pre_move_m=0.8, con_id=512), m_price=0.0)
    valid = ready_snapshot(pre_move_m=0.8, con_id=513)
    snapshots = (nan_pre, infinite_p0, zero_m, valid)

    signals = SessionHardStructureDStrategy().evaluate(
        snapshots,
        strategy_context(
            snapshots,
            {con_id: SESSION_HARD_THRESHOLD for con_id in range(510, 514)},
        ),
    )

    assert [signal.reason for signal in signals[:3]] == [
        "STAGE5_INVALID_FEATURES",
        "STAGE5_INVALID_FEATURES",
        "STAGE5_INVALID_FEATURES",
    ]
    assert signals[3].status is SignalStatus.WAITING_FOR_ENTRY


@pytest.mark.parametrize(
    ("bar", "reason", "direction"),
    [
        (
            EntryBar(datetime(2025, 2, 20, 15, 0, tzinfo=UTC), 100.0, 101.0, 100.0),
            "STRUCTURE_D_UP_FIRST_TOUCH",
            "UP",
        ),
        (
            EntryBar(datetime(2025, 2, 20, 15, 0, tzinfo=UTC), 100.0, 101.0, 99.0),
            "STRUCTURE_D_FIRST_TOUCH_AMBIGUOUS",
            None,
        ),
    ],
)
def test_up_or_ambiguous_first_touch_does_not_emit_short_intention(
    bar: EntryBar, reason: str, direction: str | None
) -> None:
    strategy = SessionHardStructureDStrategy()
    snapshot = ready_snapshot(pre_move_m=0.8)
    strategy.evaluate(
        (snapshot,),
        strategy_context((snapshot,), {101: SESSION_HARD_THRESHOLD}),
    )

    signal = strategy.observe_entry_bars({101: (bar,)})[0]

    assert signal.status is SignalStatus.EXPIRED
    assert signal.reason == reason
    assert signal.direction == direction
    assert signal.side is None


def test_no_touch_waits_then_expires_after_complete_five_bar_window() -> None:
    strategy = SessionHardStructureDStrategy()
    snapshot = ready_snapshot(pre_move_m=0.8)
    strategy.evaluate(
        (snapshot,),
        strategy_context((snapshot,), {101: SESSION_HARD_THRESHOLD}),
    )
    bars = tuple(
        EntryBar(snapshot.t0 + timedelta(minutes=index), 100.0, 100.1, 99.9) for index in range(5)
    )

    partial = strategy.observe_entry_bars({101: bars[:4]})[0]
    expired = strategy.observe_entry_bars({101: bars[4:]})[0]

    assert partial.status is SignalStatus.WAITING_FOR_ENTRY
    assert expired.status is SignalStatus.EXPIRED
    assert expired.reason == "STRUCTURE_D_NO_DOWN_FIRST_TOUCH"


def test_later_touch_waits_for_complete_earlier_bar_prefix() -> None:
    strategy = SessionHardStructureDStrategy()
    snapshot = ready_snapshot(pre_move_m=0.8)
    strategy.evaluate(
        (snapshot,),
        strategy_context((snapshot,), {101: SESSION_HARD_THRESHOLD}),
    )
    later_down = EntryBar(
        snapshot.t0 + timedelta(minutes=1),
        100.0,
        100.0,
        99.0,
    )

    incomplete = strategy.observe_entry_bars({101: (later_down,)})[0]
    complete = strategy.observe_entry_bars({101: (EntryBar(snapshot.t0, 100.0, 100.1, 99.9),)})[0]

    assert incomplete.status is SignalStatus.WAITING_FOR_ENTRY
    assert incomplete.direction is None
    assert not incomplete.selected
    assert complete.status is SignalStatus.ENTRY_TRIGGERED
    assert complete.direction == "DOWN"
    assert complete.selected


def test_prior_timestamp_entry_does_not_block_later_ranking_group() -> None:
    first = replace(ready_snapshot(pre_move_m=0.8, con_id=520), symbol="FIRST")
    later = replace(ready_snapshot(pre_move_m=0.8, con_id=521), symbol="LATER")
    snapshots = (first, later)
    strategy = SessionHardStructureDStrategy()
    strategy.evaluate(
        snapshots,
        strategy_context(
            snapshots,
            {520: SESSION_HARD_THRESHOLD, 521: SESSION_HARD_THRESHOLD},
        ),
    )

    first_group = strategy.observe_entry_bars(
        {
            520: (EntryBar(first.t0, 100.0, 100.0, 99.0),),
            521: (EntryBar(later.t0, 100.0, 100.1, 99.9),),
        }
    )
    later_group = strategy.observe_entry_bars(
        {
            521: (
                EntryBar(
                    later.t0 + timedelta(minutes=1),
                    100.0,
                    100.0,
                    99.0,
                ),
            )
        }
    )

    assert [signal.symbol for signal in first_group if signal.selected] == ["FIRST"]
    assert [signal.symbol for signal in later_group if signal.selected] == ["LATER"]


def test_gap_through_lower_level_uses_open_and_emits_only_once() -> None:
    strategy = SessionHardStructureDStrategy()
    snapshot = ready_snapshot(pre_move_m=0.8)
    original = replace(snapshot)
    evaluated = strategy.evaluate(
        (snapshot,),
        strategy_context((snapshot,), {101: SESSION_HARD_THRESHOLD}),
    )
    bar = EntryBar(snapshot.t0, 99.5, 99.7, 99.4)

    first = strategy.observe_entry_bars({101: (bar,)})
    duplicate = strategy.observe_entry_bars({101: (bar,)})

    assert len(evaluated) == 1
    assert len(first) == 1
    assert duplicate == ()
    assert first[0].entry_reference == 99.5
    assert first[0].signal_timestamp == snapshot.t0
    assert snapshot == original
    assert not hasattr(first[0], "quantity")
    assert not hasattr(first[0], "account_equity")
    assert not hasattr(strategy, "broker")


def test_run_identity_keeps_same_physical_opportunity_separate() -> None:
    snapshot = replace(ready_snapshot(pre_move_m=0.8), run_ids=("RUN_A", "RUN_B"))
    strategy = SessionHardStructureDStrategy()

    first = strategy.evaluate(
        (snapshot,),
        strategy_context(
            (snapshot,),
            {101: SESSION_HARD_THRESHOLD},
            run_id="RUN_A",
        ),
    )[0]
    second = strategy.evaluate(
        (snapshot,),
        strategy_context(
            (snapshot,),
            {101: SESSION_HARD_THRESHOLD},
            run_id="RUN_B",
        ),
    )[0]

    assert first.signal_id != second.signal_id
    assert first.run_id == "RUN_A"
    assert second.run_id == "RUN_B"


def test_stage6_diagnostic_uses_frozen_fixture_without_broker(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = diagnostic_main(["--fixture", "tests/fixtures/stage6_glw_golden.json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "GLW" in output
    assert "ENTRY_TRIGGERED" in output
