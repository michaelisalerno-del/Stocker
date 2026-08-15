from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from stocker_prospective.config import DirectionlessShadowV0Config, ProspectiveConfig
from stocker_prospective.database import ProspectiveRepository
from stocker_prospective.directionless_shadow_repository_v0 import (
    DirectionlessShadowRepositoryV0,
)
from stocker_prospective.m1c_directionless_shadow_v0 import (
    DirectionlessShadowStateV0,
    FrozenDirectionlessShadowV0,
    ShadowBBOObservationV0,
    create_directionless_shadow_v0,
)

T0 = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)


def _episode() -> FrozenDirectionlessShadowV0:
    return create_directionless_shadow_v0(
        m1c_episode_id="m1c-hard-1",
        symbol="AAPL",
        con_id=265598,
        session=date(2026, 8, 17),
        t0=T0,
        p0=100.0,
        movement_price=2.0,
        probability=0.91,
        consumed_ratio=0.7,
        code_hash="a" * 40,
        config_hash="b" * 64,
        m1c_artifact_hash="c" * 64,
    )


def _quote(seconds: int, bid: float, ask: float, event_id: str) -> ShadowBBOObservationV0:
    return ShadowBBOObservationV0(
        event_id=event_id,
        ordering_timestamp=T0 + timedelta(seconds=seconds),
        receive_timestamp=T0 + timedelta(seconds=seconds, milliseconds=1),
        local_sequence=seconds,
        connection_generation=1,
        bid=bid,
        ask=ask,
    )


def test_frozen_levels_and_non_hard_rejected() -> None:
    episode = _episode()
    assert episode.family == "HARD_M1C"
    assert episode.upper_trigger == pytest.approx(100.4)
    assert episode.lower_trigger == pytest.approx(99.6)
    assert episode.entry_deadline == T0 + timedelta(minutes=5)
    assert episode.horizon == T0 + timedelta(minutes=15)
    with pytest.raises(ValueError, match="HARD_M1C"):
        create_directionless_shadow_v0(
            m1c_episode_id="p01",
            symbol="AAPL",
            con_id=265598,
            session=date(2026, 8, 17),
            t0=T0,
            p0=100.0,
            movement_price=2.0,
            probability=0.8,
            consumed_ratio=None,
            family="P01_NEAR_M1",  # type: ignore[arg-type]
            code_hash="a" * 40,
            config_hash="b" * 64,
            m1c_artifact_hash="c" * 64,
        )


def test_long_entry_target_metrics_and_whipsaw() -> None:
    machine = FrozenDirectionlessShadowV0(_episode())
    machine.observe(_quote(1, 100.0, 100.2, "q0"))
    entered = machine.observe(_quote(2, 100.3, 100.4, "entry"))
    assert entered.state is DirectionlessShadowStateV0.LONG_ACTIVE
    assert entered.direction == "LONG"
    assert entered.entry_price == pytest.approx(100.4)
    assert entered.stop_price == pytest.approx(99.4)
    assert entered.target_price == pytest.approx(102.4)
    assert entered.entry_source_event_id == "entry"
    assert entered.observed_market_price_at_detection == pytest.approx(100.4)

    machine.observe(_quote(3, 99.5, 99.6, "whipsaw"))
    exited = machine.observe(_quote(4, 102.4, 102.5, "target"))
    assert exited.state is DirectionlessShadowStateV0.TARGET_EXITED
    assert exited.exit_reason == "SHADOW_TARGET"
    assert exited.gross_price_return == pytest.approx(2.0)
    assert exited.gross_M == pytest.approx(1.0)
    assert exited.cost_price == pytest.approx(0.1004)
    assert exited.net_M == pytest.approx(0.9498)
    assert exited.net_R == pytest.approx(1.8996)
    assert exited.MFE_price == pytest.approx(2.0)
    assert exited.MAE_price == pytest.approx(0.9)
    assert exited.opposite_trigger_touched is True
    assert exited.opposite_within_1m is True


def test_short_entry_stop_and_no_reentry() -> None:
    machine = FrozenDirectionlessShadowV0(_episode())
    entered = machine.observe(_quote(2, 99.6, 99.7, "entry"))
    assert entered.state is DirectionlessShadowStateV0.SHORT_ACTIVE
    assert entered.entry_price == pytest.approx(99.6)
    assert entered.stop_price == pytest.approx(100.6)
    assert entered.target_price == pytest.approx(97.6)
    exited = machine.observe(_quote(3, 100.5, 100.6, "stop"))
    assert exited.state is DirectionlessShadowStateV0.STOP_EXITED
    assert exited.exit_reason == "SHADOW_STOP"
    unchanged = machine.observe(_quote(4, 97.5, 97.6, "late-target"))
    assert unchanged == exited


def test_time_exit_uses_first_causal_liquidation_quote() -> None:
    machine = FrozenDirectionlessShadowV0(_episode())
    machine.observe(_quote(1, 100.3, 100.4, "entry"))
    machine.observe(_quote(899, 100.8, 100.9, "before"))
    exited = machine.observe(_quote(900, 100.7, 100.8, "horizon"))
    assert exited.state is DirectionlessShadowStateV0.TIME_EXITED
    assert exited.exit_reason == "SHADOW_TIME_EXIT"
    assert exited.exit_price == pytest.approx(100.7)
    assert exited.exit_source_event_id == "horizon"


def test_no_entry_timeout_and_same_timestamp_sequence_order() -> None:
    machine = FrozenDirectionlessShadowV0(_episode())
    machine.observe(_quote(300, 100.0, 100.1, "deadline"))
    timed_out = machine.advance_time(T0 + timedelta(minutes=5, microseconds=1))
    assert timed_out.state is DirectionlessShadowStateV0.NO_ENTRY_TIMEOUT
    assert machine.observe(_quote(301, 100.4, 100.5, "late")) == timed_out

    same_time = FrozenDirectionlessShadowV0(_episode())
    q1 = _quote(1, 100.0, 100.1, "first")
    q2 = q1.model_copy(update={"event_id": "second", "local_sequence": 2, "ask": 100.4})
    same_time.observe(q1)
    assert same_time.observe(q2).entry_source_event_id == "second"


def test_gap_does_not_infer_path() -> None:
    waiting = FrozenDirectionlessShadowV0(_episode())
    waiting.observe(_quote(1, 100.0, 100.1, "before-gap"))
    waiting.connection_lost(T0 + timedelta(seconds=2), generation=1, gap_id="gap-1")
    waiting.connection_restored(T0 + timedelta(seconds=3), generation=2, gap_id="gap-1")
    unresolved = waiting.observe(_quote(4, 99.5, 100.5, "after-gap"))
    assert unresolved.state is DirectionlessShadowStateV0.ENTRY_UNRESOLVED
    assert unresolved.exit_reason == "ENTRY_PATH_UNRESOLVED"

    active = FrozenDirectionlessShadowV0(_episode())
    active.observe(_quote(1, 100.3, 100.4, "entry"))
    active.connection_lost(T0 + timedelta(seconds=2), generation=1, gap_id="gap-2")
    active.connection_restored(T0 + timedelta(seconds=3), generation=2, gap_id="gap-2")
    unresolved_active = active.observe(_quote(4, 99.0, 99.1, "after-gap"))
    assert unresolved_active.state is DirectionlessShadowStateV0.ACTIVE_PATH_UNRESOLVED
    assert unresolved_active.exit_reason == "SHADOW_PATH_UNRESOLVED_DATA_GAP"
    assert unresolved_active.gap_start == T0 + timedelta(seconds=2)
    assert unresolved_active.gap_end == T0 + timedelta(seconds=3)


def test_invalid_quote_is_ignored_and_event_replay_is_idempotent() -> None:
    machine = FrozenDirectionlessShadowV0(_episode())
    invalid = _quote(1, 100.5, 100.4, "bad")
    assert machine.observe(invalid).state is DirectionlessShadowStateV0.WAITING_FOR_ENTRY
    valid = _quote(2, 100.3, 100.4, "entry")
    entered = machine.observe(valid)
    assert machine.observe(valid) == entered


def test_config_is_disabled_and_frozen_by_default(tmp_path: Path) -> None:
    config = ProspectiveConfig.model_validate(
        {
            "paths": {
                "database": str(tmp_path / "db.sqlite3"),
                "bundle_root": str(tmp_path / "bundles"),
                "feature_parity_report": str(tmp_path / "parity.json"),
            },
            "runtime": {
                "mode": "shadow",
                "source": "replay",
                "prospective_start_utc": T0.isoformat(),
                "instance_id": "test",
                "app_version": "0.1.0",
                "git_commit": "abcdef1",
            },
            "context": {"mode": "signed_import", "hmac_secret_env": "TEST_SECRET"},
        }
    )
    assert config.directionless_shadow_v0.enabled is False
    assert config.directionless_shadow_v0.orders_enabled is False
    assert config.directionless_shadow_v0.execution_enabled is False
    assert config.directionless_shadow_v0.analysis_enabled is False
    with pytest.raises(ValidationError):
        DirectionlessShadowV0Config.model_validate({"trigger_M": 0.25})
    with pytest.raises(ValidationError):
        DirectionlessShadowV0Config.model_validate(
            {"enabled": True, "activation_timestamp_utc": T0.isoformat()}
        )


def test_repository_restart_and_exactly_once_transition(tmp_path: Path) -> None:
    database = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    database.migrate()
    repository = DirectionlessShadowRepositoryV0(database, run_id="run-v0")
    episode = _episode()
    repository.save(episode)
    repository.save(episode)
    restarted = DirectionlessShadowRepositoryV0(database, run_id="run-v0")
    loaded = restarted.load("m1c-hard-1")
    assert loaded == episode
    assert restarted.load_active() == (episode,)
    with database._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM m1c_directionless_shadow_transition_v0"
            ).fetchone()[0]
            == 1
        )
