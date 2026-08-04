from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from stocker_prospective.contract import CLAIMS_BOUNDARY, M1C_FROZEN_THRESHOLD
from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.direction import FrozenDirectionRuntime
from stocker_prospective.direction_features import FrozenDirectionFeatureBuilder
from stocker_prospective.frozen_m1c import (
    FreshEpisodeTracker,
    FrozenM1CRuntime,
    FrozenM1CScore,
)
from stocker_prospective.group_o import build_group_o_context
from stocker_prospective.m1c_features import FROZEN_CHECKPOINTS, LiveFeatureBar
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import (
    FrozenM1CRecorderEngine,
    RecorderCheckpointInput,
)
from stocker_prospective.safety import EpisodeSafetyInputs, evaluate_episode_safety
from stocker_prospective.tail_phase_v1 import (
    FORBIDDEN_TAIL_PHASE_FEATURES,
    MovementConsumedBarV1,
    TailPhaseTrackerV1,
    assert_tail_phase_unprotected_sessions,
    assign_movement_consumed_bucket_v1,
    calculate_movement_consumed_v1,
    load_tail_phase_frozen_config_v1,
)
from stocker_research.m1c_tail_phase_v1 import (
    apply_frozen_consumed_bucket_v1,
    attach_canonical_tail_outcomes_v1,
    attach_frozen_a1_and_regime_v1,
    build_tail_phase_checkpoint_rows_v1,
    construct_fresh_tail_episodes_v1,
    freeze_movement_consumed_median_v1,
    score_frozen_m1c_checkpoint_rows_v1,
)

SESSION = date(2025, 1, 2)
START = datetime(2025, 1, 2, 15, 0, tzinfo=UTC)
BELOW = 0.20
ABOVE = 0.60
ROOT = Path(__file__).resolve().parents[1]
ARCHETYPE_PRIMARY = (
    ROOT
    / "research"
    / "directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
    / "artifacts"
    / "primary"
)
TAIL_PHASE_PRIMARY = (
    ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-tail-phase-v1"
    / "artifacts"
    / "primary"
)


def _observe(
    tracker: TailPhaseTrackerV1,
    *,
    checkpoint: int,
    probability: float,
    session: date = SESSION,
    valid: bool = True,
):
    return tracker.evaluate(
        symbol="AAL",
        session=session,
        checkpoint=checkpoint,
        causal_timestamp=START + timedelta(minutes=(checkpoint - FROZEN_CHECKPOINTS[0]) * 5),
        probability=probability,
        valid=valid,
        invalid_reason=None if valid else "synthetic_invalid_checkpoint",
    )


def _bar(
    ordinal: int,
    *,
    symbol: str = "AAL",
    session: date = SESSION,
    high: float | None = None,
    low: float | None = None,
) -> MovementConsumedBarV1:
    start = START + timedelta(minutes=ordinal * 5)
    return MovementConsumedBarV1(
        symbol=symbol,
        session=session,
        bar_ordinal=ordinal,
        bar_start_timestamp=start,
        bar_complete_timestamp=start + timedelta(minutes=5),
        high=float(101 + ordinal if high is None else high),
        low=float(100 + ordinal if low is None else low),
        finalised=True,
    )


def test_below_below_above_is_first_entry() -> None:
    tracker = TailPhaseTrackerV1()
    _observe(tracker, checkpoint=6, probability=BELOW)
    _observe(tracker, checkpoint=8, probability=BELOW)

    result = _observe(tracker, checkpoint=10, probability=ABOVE)

    assert result.m1c_tail_phase_v1 == "FIRST_ENTRY"
    assert result.m1c_high_tail_v1 is True
    assert result.tail_entry_number_v1 == 1
    assert result.prior_tail_entries_v1 == 0
    assert result.tail_run_length_checkpoints_v1 == 1
    assert result.tail_run_age_minutes_v1 == 0.0
    assert result.previous_checkpoint_above_tail_v1 is False
    assert result.phase_history_complete_v1 is True
    assert result.phase_missing_reason_v1 is None


def test_second_consecutive_above_is_persistent() -> None:
    tracker = TailPhaseTrackerV1()
    _observe(tracker, checkpoint=6, probability=BELOW)
    first = _observe(tracker, checkpoint=8, probability=ABOVE)

    result = _observe(tracker, checkpoint=10, probability=ABOVE)

    assert first.m1c_tail_phase_v1 == "FIRST_ENTRY"
    assert result.m1c_tail_phase_v1 == "PERSISTENT"
    assert result.tail_entry_number_v1 == 1
    assert result.tail_run_length_checkpoints_v1 == 2
    assert result.tail_run_age_minutes_v1 == 10.0
    assert result.previous_checkpoint_above_tail_v1 is True


def test_above_below_above_is_re_entry() -> None:
    tracker = TailPhaseTrackerV1()
    _observe(tracker, checkpoint=6, probability=BELOW)
    _observe(tracker, checkpoint=8, probability=ABOVE)
    outside = _observe(tracker, checkpoint=10, probability=BELOW)

    result = _observe(tracker, checkpoint=12, probability=ABOVE)

    assert outside.m1c_tail_phase_v1 == "OUTSIDE_TAIL"
    assert outside.tail_run_length_checkpoints_v1 == 0
    assert result.m1c_tail_phase_v1 == "RE_ENTRY"
    assert result.tail_entry_number_v1 == 2
    assert result.prior_tail_entries_v1 == 1
    assert result.minutes_since_previous_tail_exit_v1 == 10.0


def test_first_frozen_checkpoint_above_is_first_entry() -> None:
    result = _observe(TailPhaseTrackerV1(), checkpoint=6, probability=ABOVE)

    assert result.m1c_tail_phase_v1 == "FIRST_ENTRY"
    assert result.previous_checkpoint_above_tail_v1 is False
    assert result.phase_history_complete_v1 is True


def test_missing_immediate_prior_checkpoint_is_unknown() -> None:
    tracker = TailPhaseTrackerV1()
    _observe(tracker, checkpoint=6, probability=BELOW)

    result = _observe(tracker, checkpoint=10, probability=ABOVE)

    assert result.m1c_tail_phase_v1 == "UNKNOWN_INCOMPLETE"
    assert result.tail_entry_number_v1 is None
    assert result.previous_checkpoint_above_tail_v1 is None
    assert result.phase_history_complete_v1 is False
    assert result.phase_missing_reason_v1 == "missing_immediately_preceding_checkpoint:8"


def test_missing_earlier_history_that_could_hide_entry_is_unknown() -> None:
    tracker = TailPhaseTrackerV1()
    _observe(tracker, checkpoint=8, probability=BELOW)

    result = _observe(tracker, checkpoint=10, probability=ABOVE)

    assert result.m1c_tail_phase_v1 == "UNKNOWN_INCOMPLETE"
    assert result.previous_checkpoint_above_tail_v1 is False
    assert result.phase_missing_reason_v1 == "earlier_checkpoint_history_incomplete:6"


def test_below_after_entry_is_outside_tail() -> None:
    tracker = TailPhaseTrackerV1()
    _observe(tracker, checkpoint=6, probability=ABOVE)

    result = _observe(tracker, checkpoint=8, probability=BELOW)

    assert result.m1c_tail_phase_v1 == "OUTSIDE_TAIL"
    assert result.m1c_high_tail_v1 is False
    assert result.tail_run_length_checkpoints_v1 == 0
    assert result.tail_run_age_minutes_v1 == 0.0
    assert result.minutes_since_previous_tail_exit_v1 == 0.0


def test_threshold_equality_is_in_tail() -> None:
    result = _observe(
        TailPhaseTrackerV1(),
        checkpoint=6,
        probability=M1C_FROZEN_THRESHOLD,
    )

    assert result.m1c_high_tail_v1 is True
    assert result.m1c_tail_phase_v1 == "FIRST_ENTRY"


def test_invalid_previous_checkpoint_is_not_bridged() -> None:
    tracker = TailPhaseTrackerV1()
    _observe(tracker, checkpoint=6, probability=BELOW)
    _observe(tracker, checkpoint=8, probability=ABOVE, valid=False)

    result = _observe(tracker, checkpoint=10, probability=ABOVE)

    assert result.m1c_tail_phase_v1 == "UNKNOWN_INCOMPLETE"
    assert result.phase_missing_reason_v1 == "invalid_immediately_preceding_checkpoint:8"


def test_tail_history_resets_between_sessions() -> None:
    tracker = TailPhaseTrackerV1()
    _observe(tracker, checkpoint=6, probability=ABOVE)

    next_session = _observe(
        tracker,
        checkpoint=6,
        probability=ABOVE,
        session=date(2025, 1, 3),
    )

    assert next_session.m1c_tail_phase_v1 == "FIRST_ENTRY"
    assert next_session.tail_entry_number_v1 == 1


def test_movement_consumed_uses_only_three_pretrigger_bars() -> None:
    bars = tuple(_bar(index) for index in range(9))
    denominator = 0.01

    result = calculate_movement_consumed_v1(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        completed_bars=bars,
        previous_close_implied_movement_15m=denominator,
    )

    expected_numerator = math.log(106.0 / 103.0)
    assert result.movement_consumed_complete_v1 is True
    assert result.movement_consumed_numerator_v1 == pytest.approx(expected_numerator)
    assert result.movement_consumed_denominator_v1 == denominator
    assert result.movement_consumed_v1 == pytest.approx(expected_numerator / denominator)
    assert result.movement_consumed_missing_reason_v1 is None


def test_future_bars_cannot_change_movement_consumed() -> None:
    original = tuple(_bar(index) for index in range(9))
    changed = tuple(
        _bar(index, high=10_000.0, low=1.0) if index >= 6 else _bar(index) for index in range(9)
    )

    first = calculate_movement_consumed_v1(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        completed_bars=original,
        previous_close_implied_movement_15m=0.01,
    )
    second = calculate_movement_consumed_v1(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        completed_bars=changed,
        previous_close_implied_movement_15m=0.01,
    )

    assert first == second


def test_incomplete_pretrigger_window_is_explicit_and_does_not_cross_session() -> None:
    bars = (
        _bar(3),
        _bar(5),
        _bar(4, session=date(2025, 1, 1)),
    )

    result = calculate_movement_consumed_v1(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        completed_bars=bars,
        previous_close_implied_movement_15m=0.01,
    )

    assert result.movement_consumed_complete_v1 is False
    assert result.movement_consumed_v1 is None
    assert result.movement_consumed_numerator_v1 is None
    assert result.movement_consumed_denominator_v1 == 0.01
    assert result.movement_consumed_missing_reason_v1 == "incomplete_pretrigger_window:4"


def test_another_stock_cannot_change_stock_local_consumed_value() -> None:
    own = tuple(_bar(index) for index in range(6))
    with_peer = (
        *own,
        *tuple(_bar(index, symbol="SOFI", high=999.0, low=1.0) for index in range(6)),
    )

    first = calculate_movement_consumed_v1(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        completed_bars=own,
        previous_close_implied_movement_15m=0.01,
    )
    second = calculate_movement_consumed_v1(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        completed_bars=with_peer,
        previous_close_implied_movement_15m=0.01,
    )

    assert first == second


def test_phase_is_unchanged_by_future_checkpoints_or_peer_availability() -> None:
    tracker = TailPhaseTrackerV1()
    tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        causal_timestamp=START,
        probability=BELOW,
    )
    current = tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=8,
        causal_timestamp=START + timedelta(minutes=10),
        probability=ABOVE,
    )
    tracker.evaluate(
        symbol="AAOI",
        session=SESSION,
        checkpoint=6,
        causal_timestamp=START,
        probability=ABOVE,
    )
    tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=10,
        causal_timestamp=START + timedelta(minutes=20),
        probability=ABOVE,
    )

    repeated = tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=8,
        causal_timestamp=START + timedelta(minutes=10),
        probability=ABOVE,
    )

    assert repeated == current
    assert current.m1c_tail_phase_v1 == "FIRST_ENTRY"


def test_out_of_order_checkpoint_is_explicitly_unknown() -> None:
    tracker = TailPhaseTrackerV1()
    tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=8,
        causal_timestamp=START + timedelta(minutes=10),
        probability=ABOVE,
    )

    result = tracker.evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        causal_timestamp=START,
        probability=BELOW,
    )

    assert result.m1c_tail_phase_v1 == "UNKNOWN_INCOMPLETE"
    assert result.phase_missing_reason_v1 == "checkpoint_out_of_order"


def test_consumed_bucket_uses_frozen_median_with_equality_low() -> None:
    assert assign_movement_consumed_bucket_v1(1.0, frozen_median=1.0) == "LOW_OR_EQUAL"
    assert assign_movement_consumed_bucket_v1(1.00001, frozen_median=1.0) == "HIGH"
    assert assign_movement_consumed_bucket_v1(None, frozen_median=1.0) == "UNKNOWN_INCOMPLETE"


def test_protected_session_guard_fails_before_outcome_work() -> None:
    assert_tail_phase_unprotected_sessions([date(2025, 12, 31)])
    with pytest.raises(ValueError, match="protected"):
        assert_tail_phase_unprotected_sessions([date(2026, 1, 1)])


def test_tail_phase_contract_excludes_contaminated_and_order_capable_inputs() -> None:
    assert {"signed_pressure", "tension"}.issubset(FORBIDDEN_TAIL_PHASE_FEATURES)
    assert CLAIMS_BOUNDARY["execution_enabled"] is False
    assert CLAIMS_BOUNDARY["place_order_method_available"] is False


def test_group_o_preserves_explicit_previous_close_implied_movement() -> None:
    context = build_group_o_context(
        symbol="AAL",
        signal_session=date(2026, 7, 24),
        actual_option_observation_session=date(2026, 7, 23),
        front_expiry=date(2026, 7, 24),
        dte=1,
        atm_strike=12.0,
        previous_close_implied_movement_15m=0.0125,
        features={"options_missing": 0.0},
        missing_indicators={"options_missing": False},
        quality_status="valid",
        source_receipt_hashes=("a" * 64,),
    )

    assert context.eligible is True
    assert context.previous_close_implied_movement_15m == 0.0125


def test_invalid_optional_implied_movement_does_not_invalidate_group_o() -> None:
    context = build_group_o_context(
        symbol="AAL",
        signal_session=date(2026, 7, 24),
        actual_option_observation_session=date(2026, 7, 23),
        front_expiry=date(2026, 7, 24),
        dte=1,
        atm_strike=12.0,
        previous_close_implied_movement_15m=0.0,
        features={"options_missing": 0.0},
        missing_indicators={"options_missing": False},
        quality_status="valid",
        source_receipt_hashes=("a" * 64,),
    )
    consumed = calculate_movement_consumed_v1(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        completed_bars=tuple(_bar(index) for index in range(6)),
        previous_close_implied_movement_15m=(context.previous_close_implied_movement_15m),
    )

    assert context.eligible is True
    assert consumed.movement_consumed_complete_v1 is False
    assert (
        consumed.movement_consumed_missing_reason_v1
        == "previous_close_implied_movement_15m_invalid"
    )
    assert consumed.movement_consumed_numerator_v1 is not None


def test_checkpoint_repository_persists_flat_tail_phase_fields(tmp_path: Path) -> None:
    database = ProspectiveRepository(tmp_path / "tail-phase.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="tail-phase-v1",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    phase = _observe(TailPhaseTrackerV1(), checkpoint=6, probability=ABOVE)
    consumed = calculate_movement_consumed_v1(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        completed_bars=tuple(_bar(index) for index in range(6)),
        previous_close_implied_movement_15m=0.01,
    )
    assert consumed.movement_consumed_v1 is not None
    repository = FrozenRecorderRepository(database)
    score = FrozenM1CScore(
        model_hash="b" * 64,
        probability=ABOVE,
        threshold=M1C_FROZEN_THRESHOLD,
        threshold_passed=True,
        feature_order=("x",),
        feature_values=(1.0,),
        transformed_values=(1.0,),
        feature_hash="c" * 64,
        missing_feature_count=0,
    )
    checkpoint_id = repository.record_checkpoint(
        metadata,
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        bar_start_utc=START - timedelta(minutes=5),
        bar_end_utc=START,
        score=score,
        session_context_hash="d" * 64,
        feature_values={"x": 1.0},
        eligible=True,
        feature_freshness="fresh",
        rejection_reasons=(),
        tail_phase_v1=phase,
        movement_consumed_v1=consumed,
        movement_consumed_bucket_v1="HIGH",
        movement_consumed_frozen_median_v1=1.3986941389121161,
    )

    with pytest.raises(ValueError, match="membership differs"):
        repository.record_checkpoint(
            metadata,
            symbol="AAL",
            session=SESSION,
            checkpoint=6,
            bar_start_utc=START - timedelta(minutes=5),
            bar_end_utc=START,
            score=score,
            session_context_hash="d" * 64,
            feature_values={"x": 1.0},
            eligible=True,
            feature_freshness="fresh",
            rejection_reasons=(),
            tail_phase_v1=phase.model_copy(
                update={
                    "m1c_high_tail_v1": False,
                    "m1c_tail_phase_v1": "OUTSIDE_TAIL",
                }
            ),
            movement_consumed_v1=consumed,
            movement_consumed_bucket_v1="HIGH",
            movement_consumed_frozen_median_v1=1.3986941389121161,
        )

    with pytest.raises(ValueError, match="immutable movement-consumed"):
        repository.record_checkpoint(
            metadata,
            symbol="AAL",
            session=SESSION,
            checkpoint=6,
            bar_start_utc=START - timedelta(minutes=5),
            bar_end_utc=START,
            score=score,
            session_context_hash="d" * 64,
            feature_values={"x": 1.0},
            eligible=True,
            feature_freshness="fresh",
            rejection_reasons=(),
            tail_phase_v1=phase,
            movement_consumed_v1=consumed.model_copy(
                update={"movement_consumed_v1": (consumed.movement_consumed_v1 + 0.1)}
            ),
            movement_consumed_bucket_v1="HIGH",
            movement_consumed_frozen_median_v1=1.3986941389121161,
        )

    with sqlite3.connect(database.database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM m1c_checkpoint_v0 WHERE id = ?",
            (checkpoint_id,),
        ).fetchone()

    assert row is not None
    assert row["m1c_high_tail_v1"] == 1
    assert row["m1c_tail_phase_v1"] == "FIRST_ENTRY"
    assert row["tail_entry_number_v1"] == 1
    assert row["movement_consumed_complete_v1"] == 1
    assert row["movement_consumed_bucket_v1"] == "HIGH"
    checkpoint_source = json.loads(str(row["tail_phase_source_v1_json"]))
    assert checkpoint_source["movement_consumed_frozen_median_v1"] == 1.3986941389121161
    assert checkpoint_source["movement_consumed_median_provenance"] == {
        "start": "2024-01-01",
        "end": "2024-12-31",
        "predictor_values_only": True,
    }
    decision = FreshEpisodeTracker().evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        trigger_bar_end=START,
        probability=ABOVE,
    )
    safety = evaluate_episode_safety(
        EpisodeSafetyInputs(
            capability_preflight_passed=True,
            m1c_parity_passed=True,
            direction_parity_passed=True,
            market_data_type=MarketDataType.LIVE,
            previous_close_group_o_valid=True,
            trigger_bar_complete=True,
            clock_drift_within_tolerance=True,
            underlying_quote_fresh=True,
            unresolved_bar_gap=False,
            deterministic_episode_identity=True,
            raw_event_storage_writable=True,
            scientific_recording_authorized=True,
        )
    )
    episode_id = repository.record_episode(
        metadata,
        checkpoint_id=checkpoint_id,
        decision=decision,
        safety=safety,
    )
    with sqlite3.connect(database.database_path) as connection:
        connection.row_factory = sqlite3.Row
        episode = connection.execute(
            "SELECT * FROM m1c_episode_v0 WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()

    assert episode is not None
    assert episode["phase_at_trigger_v1"] == "FIRST_ENTRY"
    assert episode["tail_run_age_at_trigger_v1"] == 0.0
    assert episode["movement_consumed_at_trigger_v1"] == pytest.approx(
        consumed.movement_consumed_v1
    )
    assert episode["m1c_high_tail_threshold_v1"] == M1C_FROZEN_THRESHOLD
    assert json.loads(str(episode["tail_phase_source_v1_json"])) == checkpoint_source


def test_checkpoint_replay_accepts_legacy_stale_quote_rejection_as_diagnostic(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "legacy-stale-quote.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="legacy-stale-quote-v0",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    score = FrozenM1CScore(
        model_hash="b" * 64,
        probability=ABOVE,
        threshold=M1C_FROZEN_THRESHOLD,
        threshold_passed=True,
        feature_order=("x",),
        feature_values=(1.0,),
        transformed_values=(1.0,),
        feature_hash="c" * 64,
        missing_feature_count=0,
    )
    checkpoint_id = repository.record_checkpoint(
        metadata,
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        bar_start_utc=START - timedelta(minutes=5),
        bar_end_utc=START,
        score=score,
        session_context_hash="d" * 64,
        feature_values={"x": 1.0},
        eligible=False,
        feature_freshness="fresh",
        rejection_reasons=("underlying_quote_stale",),
    )

    replayed_checkpoint_id = repository.record_checkpoint(
        metadata,
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        bar_start_utc=START - timedelta(minutes=5),
        bar_end_utc=START,
        score=score,
        session_context_hash="d" * 64,
        feature_values={"x": 1.0},
        eligible=True,
        feature_freshness="fresh",
        rejection_reasons=(),
        diagnostic_quality_flags=("underlying_quote_stale",),
    )

    assert replayed_checkpoint_id == checkpoint_id
    with sqlite3.connect(database.database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT eligible, rejection_reasons_json, diagnostic_quality_flags_json
            FROM m1c_checkpoint_v0
            WHERE id = ?
            """,
            (checkpoint_id,),
        ).fetchone()
    assert row is not None
    assert row["eligible"] == 0
    assert json.loads(str(row["rejection_reasons_json"])) == ["underlying_quote_stale"]
    assert json.loads(str(row["diagnostic_quality_flags_json"])) == []


def test_live_engine_logs_tail_phase_but_stale_quote_blocks_episode(
    tmp_path: Path,
) -> None:
    class FakeFeatureBuilder:
        def build(
            self,
            *,
            symbol: str,
            checkpoint: int,
            completed_bars: tuple[LiveFeatureBar, ...],
        ) -> SimpleNamespace:
            return SimpleNamespace(
                scaled_features={"x": 1.0},
                feature_hash="e" * 64,
                scaling_artifact_hash="f" * 64,
            )

    class FakeRuntime:
        def missing_group_o_features(self, context: dict[str, object]) -> tuple[str, ...]:
            return ()

        def score(self, **_: object) -> FrozenM1CScore:
            return FrozenM1CScore(
                model_hash="b" * 64,
                probability=ABOVE,
                threshold=M1C_FROZEN_THRESHOLD,
                threshold_passed=True,
                feature_order=("x",),
                feature_values=(1.0,),
                transformed_values=(1.0,),
                feature_hash="c" * 64,
                missing_feature_count=0,
            )

    database = ProspectiveRepository(tmp_path / "engine-tail-phase.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="engine-tail-phase-v1",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    context = build_group_o_context(
        symbol="AAL",
        signal_session=SESSION,
        actual_option_observation_session=date(2024, 12, 31),
        front_expiry=date(2025, 1, 3),
        dte=1,
        atm_strike=12.0,
        previous_close_implied_movement_15m=0.01,
        features={"x": 1.0},
        missing_indicators={"x": False},
        quality_status="valid",
        source_receipt_hashes=("a" * 64,),
    )
    repository.record_group_o_context(metadata, context)
    bar_start = START - timedelta(minutes=30)
    bars = tuple(
        LiveFeatureBar(
            symbol="AAL",
            session=SESSION,
            bar_ordinal=ordinal,
            bar_start_timestamp=bar_start + timedelta(minutes=5 * ordinal),
            bar_complete_timestamp=bar_start + timedelta(minutes=5 * (ordinal + 1)),
            open=100.0 + ordinal * 0.1,
            high=100.3 + ordinal * 0.1,
            low=99.8 + ordinal * 0.1,
            close=100.1 + ordinal * 0.1,
            volume=1_000.0,
            historical_relative_activity=1.0,
            finalised=True,
            source="fixture",
        )
        for ordinal in range(6)
    )
    engine = FrozenM1CRecorderEngine(
        m1c_runtime=cast(Any, FakeRuntime()),
        m1c_features=cast(Any, FakeFeatureBuilder()),
        direction_runtime=cast(Any, object()),
        direction_features=cast(Any, object()),
        repository=repository,
        movement_consumed_median_v1=1.3986941389121161,
        tail_phase_activation_status_v1="available",
    )
    result = engine.process_checkpoint(
        RecorderCheckpointInput(
            metadata=metadata,
            symbol="AAL",
            session=SESSION,
            completed_m1c_bars=bars,
            completed_direction_bars=(),
            group_o_context=context,
            market_data_type=MarketDataType.LIVE,
            capability_preflight_passed=True,
            m1c_parity_passed=True,
            direction_parity_passed=False,
            clock_drift_within_tolerance=True,
            underlying_quote_fresh=False,
            unresolved_bar_gap=False,
            raw_event_storage_writable=True,
            scientific_recording_authorized=True,
        )
    )

    assert result.episode_decision.fresh_episode is False
    assert "underlying_quote_stale" in result.rejection_reasons
    assert "underlying_quote_stale" in result.diagnostic_quality_flags
    assert result.episode_safety is None
    assert result.tail_phase_v1.m1c_tail_phase_v1 == "UNKNOWN_INCOMPLETE"
    assert result.movement_consumed_state_v1.movement_consumed_complete_v1
    assert result.movement_consumed_bucket_v1 == "LOW_OR_EQUAL"
    with database._connect() as connection:
        checkpoint = connection.execute(
            """
            SELECT eligible, rejection_reasons_json, m1c_tail_phase_v1,
                   phase_missing_reason_v1, movement_consumed_bucket_v1,
                   tail_phase_source_v1_json
            FROM m1c_checkpoint_v0
            """
        ).fetchone()
        episode = connection.execute("SELECT phase_at_trigger_v1 FROM m1c_episode_v0").fetchone()
    assert checkpoint is not None
    assert episode is None
    assert checkpoint["eligible"] == 0
    assert json.loads(str(checkpoint["rejection_reasons_json"])) == ["underlying_quote_stale"]
    assert checkpoint["m1c_tail_phase_v1"] == "UNKNOWN_INCOMPLETE"
    assert checkpoint["phase_missing_reason_v1"] == (
        "current_checkpoint_invalid:underlying_quote_stale"
    )
    source = json.loads(str(checkpoint["tail_phase_source_v1_json"]))
    assert source["tail_phase_activation_status_v1"] == "available"
    assert source["previous_close_implied_movement_15m_status"] == "available"

    next_session = date(2025, 1, 3)
    missing_context = build_group_o_context(
        symbol="AAL",
        signal_session=next_session,
        actual_option_observation_session=SESSION,
        front_expiry=next_session,
        dte=0,
        atm_strike=12.0,
        previous_close_implied_movement_15m=None,
        features={"x": 1.0},
        missing_indicators={"x": False},
        quality_status="valid",
        source_receipt_hashes=("a" * 64,),
    )
    repository.record_group_o_context(metadata, missing_context)
    missing_bars = tuple(
        bar.model_copy(
            update={
                "session": next_session,
                "bar_start_timestamp": bar.bar_start_timestamp + timedelta(days=1),
                "bar_complete_timestamp": (bar.bar_complete_timestamp + timedelta(days=1)),
            }
        )
        for bar in bars
    )
    missing_result = engine.process_checkpoint(
        RecorderCheckpointInput(
            metadata=metadata,
            symbol="AAL",
            session=next_session,
            completed_m1c_bars=missing_bars,
            completed_direction_bars=(),
            group_o_context=missing_context,
            market_data_type=MarketDataType.LIVE,
            capability_preflight_passed=True,
            m1c_parity_passed=True,
            direction_parity_passed=False,
            clock_drift_within_tolerance=True,
            underlying_quote_fresh=True,
            unresolved_bar_gap=False,
            raw_event_storage_writable=True,
            scientific_recording_authorized=True,
        )
    )

    assert missing_result.score.threshold_passed is True
    assert missing_result.episode_decision.fresh_episode is True
    assert not missing_result.movement_consumed_state_v1.movement_consumed_complete_v1
    assert missing_result.movement_consumed_bucket_v1 == "UNKNOWN_INCOMPLETE"


def test_frozen_config_loader_rejects_threshold_or_chronology_drift(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "m1c-tail-phase-v1",
        "research_id": "M1C Tail Phase V1",
        "m1c_threshold": M1C_FROZEN_THRESHOLD,
        "frozen_checkpoints": list(FROZEN_CHECKPOINTS),
        "underlying_bar_minutes": 5,
        "movement_consumed_lookback_minutes": 15,
        "movement_consumed_median_2024": 1.3986941389121161,
        "movement_consumed_median_provenance": {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "complete_observations": 18596,
            "predictor_values_only": True,
        },
        "chronology": {
            "development": {"start": "2024-01-01", "end": "2024-12-31"},
            "assessment": {"start": "2025-01-01", "end": "2025-08-22"},
            "stress": {"start": "2025-09-01", "end": "2025-12-31"},
            "protected": {"start": "2026-01-01", "end": None},
        },
        "model_identifiers": {
            "movement": "M1C/frozen-m1c-v0",
            "direction": "A1/frozen-comparison-unchanged",
        },
        "bootstrap": {"seed": 20260728, "draws": 1000, "confidence_level": 0.95},
    }
    path = tmp_path / "frozen-config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_tail_phase_frozen_config_v1(path)

    assert loaded.movement_consumed_median_2024 == 1.3986941389121161
    changed = dict(payload)
    changed["movement_consumed_median_2024"] = 1.25
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="median differs"):
        load_tail_phase_frozen_config_v1(path)

    payload["m1c_threshold"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        load_tail_phase_frozen_config_v1(path)


def _batch_bars(*, session: str, bars: int = 12) -> pd.DataFrame:
    start = pd.Timestamp(f"{session}T14:30:00Z")
    return pd.DataFrame(
        [
            {
                "stock": "AAL",
                "session": session,
                "bar_ordinal": ordinal,
                "bar_start_timestamp": start + pd.Timedelta(minutes=ordinal * 5),
                "bar_complete_timestamp": start + pd.Timedelta(minutes=(ordinal + 1) * 5),
                "open": 100.0 + ordinal,
                "high": 101.0 + ordinal,
                "low": 99.0 + ordinal,
                "close": 100.5 + ordinal,
                "volume": 1_000.0,
                "historical_relative_activity": 1.0,
                "bar_log_return": 0.001,
                "vti__bar_log_return": 0.0005,
            }
            for ordinal in range(bars)
        ]
    )


def test_batch_checkpoint_builder_and_fresh_episode_reuse_frozen_definition() -> None:
    bars = _batch_bars(session="2024-01-02")
    checkpoints = pd.DataFrame(
        {
            "stock": ["AAL", "AAL", "AAL", "AAL"],
            "session": ["2024-01-02"] * 4,
            "checkpoint": [6, 8, 10, 12],
            "feature_available_timestamp_utc": [
                bars.loc[5, "bar_complete_timestamp"],
                bars.loc[7, "bar_complete_timestamp"],
                bars.loc[9, "bar_complete_timestamp"],
                bars.loc[11, "bar_complete_timestamp"],
            ],
            "M1C_probability": [BELOW, ABOVE, ABOVE, BELOW],
            "atm_iv": [0.50] * 4,
            "partition": ["development"] * 4,
        }
    )

    panel = build_tail_phase_checkpoint_rows_v1(checkpoints, bars)
    frozen = freeze_movement_consumed_median_v1(panel)
    panel = apply_frozen_consumed_bucket_v1(panel, frozen_median=frozen.value)
    episodes = construct_fresh_tail_episodes_v1(panel)

    assert panel["m1c_tail_phase_v1"].tolist() == [
        "OUTSIDE_TAIL",
        "FIRST_ENTRY",
        "PERSISTENT",
        "OUTSIDE_TAIL",
    ]
    assert frozen.complete_observations == 4
    assert set(panel["movement_consumed_bucket_v1"]) <= {"LOW_OR_EQUAL", "HIGH"}
    assert episodes[["stock", "session", "checkpoint"]].to_dict("records") == [
        {"stock": "AAL", "session": "2024-01-02", "checkpoint": 8}
    ]
    assert episodes.iloc[0]["episode_id"].startswith("m1c-")


def test_batch_phase_ignores_future_outcome_columns_and_handles_null_inputs() -> None:
    bars = _batch_bars(session="2024-01-02")
    base = pd.DataFrame(
        {
            "stock": ["AAL", "AAL"],
            "session": ["2024-01-02", "2024-01-02"],
            "checkpoint": [6, 8],
            "feature_available_timestamp_utc": [
                bars.loc[5, "bar_complete_timestamp"],
                bars.loc[7, "bar_complete_timestamp"],
            ],
            "M1C_probability": [BELOW, ABOVE],
            "atm_iv": [0.50, 0.50],
            "synthetic_future_episode_outcome": [1.0, -1.0],
        }
    )
    changed = base.copy()
    changed["synthetic_future_episode_outcome"] = [999.0, -999.0]

    first = build_tail_phase_checkpoint_rows_v1(base, bars)
    second = build_tail_phase_checkpoint_rows_v1(changed, bars)
    causal_columns = [
        "m1c_tail_phase_v1",
        "tail_entry_number_v1",
        "tail_run_length_checkpoints_v1",
        "tail_run_age_minutes_v1",
        "movement_consumed_v1",
        "movement_consumed_numerator_v1",
        "movement_consumed_denominator_v1",
        "movement_consumed_complete_v1",
    ]
    pd.testing.assert_frame_equal(first[causal_columns], second[causal_columns])

    missing = base.iloc[[0]].copy()
    missing["M1C_probability"] = None
    missing["atm_iv"] = None
    incomplete = build_tail_phase_checkpoint_rows_v1(missing, bars)

    assert incomplete.iloc[0]["m1c_tail_phase_v1"] == "UNKNOWN_INCOMPLETE"
    assert not bool(incomplete.iloc[0]["movement_consumed_complete_v1"])
    assert (
        incomplete.iloc[0]["movement_consumed_missing_reason_v1"]
        == "previous_close_implied_movement_15m_missing"
    )
    assert pd.notna(incomplete.iloc[0]["movement_consumed_numerator_v1"])


def test_batch_outcomes_use_non_overlapping_post_range_and_are_future_only() -> None:
    bars = _batch_bars(session="2025-01-02")
    checkpoints = pd.DataFrame(
        {
            "stock": ["AAL"],
            "session": ["2025-01-02"],
            "checkpoint": [6],
            "feature_available_timestamp_utc": [bars.loc[5, "bar_complete_timestamp"]],
            "M1C_probability": [ABOVE],
            "atm_iv": [0.50],
            "partition": ["assessment"],
        }
    )
    panel = build_tail_phase_checkpoint_rows_v1(checkpoints, bars)

    outcomes = attach_canonical_tail_outcomes_v1(panel, bars)

    pre = float(outcomes.iloc[0]["movement_consumed_numerator_v1"])
    post = math.log(float(bars.loc[6:7, "high"].max()) / float(bars.loc[6:7, "low"].min()))
    assert outcomes.iloc[0]["post_share_of_local_range_complete_v1"]
    assert outcomes.iloc[0]["post_share_of_local_range_v1"] == pytest.approx(post / (pre + post))
    assert outcomes.iloc[0]["absolute_return_10m"] >= 0.0
    assert outcomes.iloc[0]["absolute_return_15m"] >= 0.0


def test_post_share_marks_missing_pre_or_post_window_without_raising() -> None:
    full_bars = _batch_bars(session="2025-01-02")
    checkpoints = pd.DataFrame(
        {
            "stock": ["AAL"],
            "session": ["2025-01-02"],
            "checkpoint": [6],
            "feature_available_timestamp_utc": [full_bars.loc[5, "bar_complete_timestamp"]],
            "M1C_probability": [ABOVE],
            "atm_iv": [0.50],
            "partition": ["assessment"],
        }
    )
    panel = build_tail_phase_checkpoint_rows_v1(checkpoints, full_bars)

    missing_pre = panel.copy()
    missing_pre["movement_consumed_numerator_v1"] = None
    pre_result = attach_canonical_tail_outcomes_v1(missing_pre, full_bars)
    post_result = attach_canonical_tail_outcomes_v1(panel, full_bars.iloc[:7])

    assert not bool(pre_result.iloc[0]["post_share_of_local_range_complete_v1"])
    assert pd.isna(pre_result.iloc[0]["post_share_of_local_range_v1"])
    assert not bool(post_result.iloc[0]["post_share_of_local_range_complete_v1"])
    assert pd.isna(post_result.iloc[0]["post_share_of_local_range_v1"])


def test_consumed_median_ignores_2025_predictor_values() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-02-01", "2025-01-02"],
            "movement_consumed_v1": [1.0, 3.0, 10_000.0],
            "movement_consumed_complete_v1": [True, True, True],
        }
    )

    frozen = freeze_movement_consumed_median_v1(frame)

    assert frozen.value == 2.0
    assert frozen.complete_observations == 2


def test_frozen_a1_and_market_context_ignore_post_checkpoint_bars() -> None:
    bars = _batch_bars(session="2025-01-02")
    checkpoints = pd.DataFrame(
        {
            "stock": ["AAL"],
            "session": ["2025-01-02"],
            "checkpoint": [8],
        }
    )
    builder = FrozenDirectionFeatureBuilder.from_beta_artifact(
        ARCHETYPE_PRIMARY / "stock_market_beta_parameters.csv"
    )
    runtime = FrozenDirectionRuntime.from_artifacts(
        model_configurations_path=ARCHETYPE_PRIMARY / "model_configurations.json",
        normalisation_path=ARCHETYPE_PRIMARY / "stock_local_normalisation_parameters.json",
        thresholds_path=ARCHETYPE_PRIMARY / "frozen_archetype_thresholds.json",
    )

    original = attach_frozen_a1_and_regime_v1(
        checkpoints,
        bars,
        feature_builder=builder,
        direction_runtime=runtime,
    )
    changed = bars.copy()
    changed.loc[changed["bar_ordinal"].ge(8), ["close", "bar_log_return"]] = [
        9_999.0,
        -0.25,
    ]
    future_changed = attach_frozen_a1_and_regime_v1(
        checkpoints,
        changed,
        feature_builder=builder,
        direction_runtime=runtime,
    )

    pinned = [
        "A1_probability_up_v1",
        "A1_action_v1",
        "A1_feature_hash_v1",
        "pre_entry_stock_signed_return_10m_v1",
        "pre_entry_broad_market_signed_return_10m_v1",
        "stock_market_alignment_v1",
    ]
    pd.testing.assert_frame_equal(original[pinned], future_changed[pinned])
    assert original.iloc[0]["A1_complete_v1"]
    assert original.iloc[0]["sector_context_status_v1"] == "out_of_scope_not_available"

    incomplete_bars = bars.copy()
    incomplete_bars.loc[0, "historical_relative_activity"] = math.nan
    incomplete = attach_frozen_a1_and_regime_v1(
        checkpoints,
        incomplete_bars,
        feature_builder=builder,
        direction_runtime=runtime,
    )
    assert not bool(incomplete.iloc[0]["A1_complete_v1"])
    assert str(incomplete.iloc[0]["A1_missing_reason_v1"]).startswith("frozen_a1_input_incomplete:")


def test_batch_m1c_scoring_uses_frozen_runtime_without_refitting() -> None:
    runtime = FrozenM1CRuntime.from_artifacts(
        feature_manifest_path=ARCHETYPE_PRIMARY / "causal_movement_feature_manifest.json",
        threshold_path=ARCHETYPE_PRIMARY / "causal_movement_threshold.json",
    )
    row: dict[str, object] = {
        "stock": "AAL",
        "session": "2025-01-02",
        "checkpoint": 6,
    }
    row.update({name: 0.0 for name in runtime.required_group_o_features})
    row.update({name: 0.0 for name in runtime.causal_group_i_features})

    scored = score_frozen_m1c_checkpoint_rows_v1(pd.DataFrame([row]), runtime=runtime)

    assert scored.iloc[0]["M1C_probability"] == pytest.approx(
        0.3791098724444006,
        abs=1e-15,
    )
    assert not bool(scored.iloc[0]["m1c_high_tail_v1"])
    assert scored.iloc[0]["m1c_model_hash_v1"] == runtime.model_hash


def test_generated_v1_artifacts_preserve_boundaries_and_episode_identity() -> None:
    config = load_tail_phase_frozen_config_v1(TAIL_PHASE_PRIMARY / "frozen_config_v1.json")
    checkpoints = pd.read_parquet(TAIL_PHASE_PRIMARY / "checkpoint_results_v1.parquet")
    episodes = pd.read_parquet(TAIL_PHASE_PRIMARY / "fresh_episode_results_v1.parquet")

    assert config.m1c_threshold == M1C_FROZEN_THRESHOLD
    assert str(checkpoints["session"].max()) <= "2025-12-31"
    assert str(episodes["session"].max()) <= "2025-12-31"
    assert not checkpoints.duplicated(["stock", "session", "checkpoint"]).any()
    assert not episodes["episode_id"].duplicated().any()
    assert (
        checkpoints["m1c_high_tail_v1"]
        .astype(bool)
        .equals(checkpoints["M1C_probability"].ge(M1C_FROZEN_THRESHOLD))
    )
    complete_share = checkpoints.loc[
        checkpoints["post_share_of_local_range_complete_v1"].astype("boolean").fillna(False)
    ]
    assert complete_share["post_share_of_local_range_v1"].between(0.0, 1.0).all()
    assert episodes["m1c_tail_phase_v1"].isin(["FIRST_ENTRY", "RE_ENTRY"]).all()
    assert episodes["phase_at_trigger_v1"].equals(episodes["m1c_tail_phase_v1"])
    assert not bool(episodes["protected_outcomes_accessed_v1"].any())
    directional = pd.read_csv(TAIL_PHASE_PRIMARY / "directional_diagnostics_v1.csv")
    blocked_interactions = directional.loc[
        directional["subgroup_type"].eq("phase_x_movement_consumed_bucket")
        & directional["status"].eq("blocked_insufficient_support")
    ]
    assert (
        blocked_interactions[
            [
                "A1_accuracy",
                "A1_mean_aligned_10m_return",
                "A1_median_aligned_10m_return",
                "A1_CALL_accuracy",
                "A1_PUT_accuracy",
            ]
        ]
        .isna()
        .all(axis=None)
    )
