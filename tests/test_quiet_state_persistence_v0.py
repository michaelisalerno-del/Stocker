from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_prospective.contract import CLAIMS_BOUNDARY
from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.frozen_m1c import FrozenM1CScore
from stocker_prospective.option_ledger import OptionContract, OptionContractPlan
from stocker_prospective.options import DteBucket
from stocker_prospective.quiet_state import (
    NeutralControlSampler,
    QuietEpisodeTracker,
    classify_quiet_state,
)
from stocker_prospective.read_store import ProspectiveReadStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository

START = datetime(2026, 7, 27, 14, 0, tzinfo=UTC)
SESSION = date(2026, 7, 27)


def _metadata(run_id: str) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=run_id,
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="b" * 64,
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )


def _score(probability: float) -> FrozenM1CScore:
    return FrozenM1CScore(
        model_hash="b" * 64,
        probability=probability,
        threshold=0.488333710794033,
        threshold_passed=probability >= 0.488333710794033,
        feature_order=("x",),
        feature_values=(1.0,),
        transformed_values=(1.0,),
        feature_hash="c" * 64,
        missing_feature_count=0,
    )


def test_quiet_checkpoint_episode_and_control_are_immutable_and_claim_bounded(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "quiet.sqlite3")
    database.migrate()
    metadata = _metadata("quiet-persistence")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    score = _score(0.13)
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
    )
    snapshot = classify_quiet_state(
        probability=score.probability,
        previous_probability=None,
        model_hash=score.model_hash,
        feature_hash=score.feature_hash,
        data_quality_status="valid",
    )
    quiet_checkpoint_id = repository.record_quiet_checkpoint(
        metadata,
        checkpoint_id=checkpoint_id,
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        snapshot=snapshot,
        eligible=True,
    )
    decision = QuietEpisodeTracker().evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        trigger_bar_end=START,
        probability=score.probability,
    )
    quiet_id = repository.record_quiet_episode(
        metadata,
        quiet_checkpoint_id=quiet_checkpoint_id,
        decision=decision,
        scientific_recording_valid=True,
    )

    assert (
        repository.record_quiet_episode(
            metadata,
            quiet_checkpoint_id=quiet_checkpoint_id,
            decision=decision,
            scientific_recording_valid=True,
        )
        == quiet_id
    )
    with database._connect() as connection:
        checkpoint = connection.execute("SELECT * FROM quiet_state_checkpoint_v0").fetchone()
        episode = connection.execute("SELECT * FROM quiet_state_observation_v0").fetchone()
    assert checkpoint["bottom_5"] == 0
    assert checkpoint["bottom_10"] == 1
    assert checkpoint["bottom_20"] == 1
    assert checkpoint["high_tail"] == 0
    assert episode["observation_kind"] == "quiet_bottom_10"
    assert json.loads(episode["claims_json"]) == CLAIMS_BOUNDARY

    other_metadata = _metadata("quiet-persistence-other-run")
    database.create_run(other_metadata)
    other_checkpoint_id = repository.record_checkpoint(
        other_metadata,
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
    )
    other_quiet_checkpoint_id = repository.record_quiet_checkpoint(
        other_metadata,
        checkpoint_id=other_checkpoint_id,
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        snapshot=snapshot,
        eligible=True,
    )
    with pytest.raises(ValueError, match="immutable quiet episode differs"):
        repository.record_quiet_episode(
            other_metadata,
            quiet_checkpoint_id=other_quiet_checkpoint_id,
            decision=decision,
            scientific_recording_valid=True,
        )

    recording_contract = OptionContract(
        underlying_con_id=1,
        con_id=101,
        expiry=date(2026, 7, 31),
        dte=4,
        dte_bucket=DteBucket.THREE_TO_FIVE_DTE,
        strike=100.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )
    unresolved_contract = OptionContract(
        **{
            **recording_contract.__dict__,
            "con_id": None,
            "strike": 105.0,
        }
    )
    later_recording_contract = OptionContract(
        **{
            **recording_contract.__dict__,
            "con_id": 102,
            "strike": 95.0,
            "right": "P",
        }
    )
    for rank, contract, status in (
        (1, recording_contract, "recording"),
        (2, unresolved_contract, "contract_not_resolved"),
        (3, later_recording_contract, "recording"),
    ):
        repository.record_quiet_option_contract(
            metadata,
            observation_id=quiet_id,
            contract=contract,
            selection_rank=rank,
            selection_roles=(),
            resolution_status=status,
            rejection_reason=(None if status == "recording" else status),
            recording_started_at_utc=(START if status == "recording" else None),
            recording_ends_at_utc=START + timedelta(minutes=60),
        )
    with database._connect() as connection:
        option_context_valid = connection.execute(
            """
            SELECT option_context_valid FROM quiet_state_observation_v0
            WHERE observation_id = ?
            """,
            (quiet_id,),
        ).fetchone()["option_context_valid"]
    assert option_context_valid == 0

    path_id = repository.record_quiet_underlying_path(
        metadata,
        observation_id=quiet_id,
        horizon_label="15m",
        target_timestamp_utc=START + timedelta(minutes=15),
        payload={
            "entry_reference_price": 100.0,
            "maximum_absolute_excursion": 1.25,
            "terminal_return": -0.002,
        },
        quality_flags=(),
    )
    assert (
        repository.record_quiet_underlying_path(
            metadata,
            observation_id=quiet_id,
            horizon_label="15m",
            target_timestamp_utc=START + timedelta(minutes=15),
            payload={
                "entry_reference_price": 100.0,
                "maximum_absolute_excursion": 1.25,
                "terminal_return": -0.002,
            },
            quality_flags=(),
        )
        == path_id
    )
    with database._connect() as connection:
        path = connection.execute("SELECT * FROM quiet_state_underlying_path_v0").fetchone()
    assert json.loads(path["payload_json"])["maximum_absolute_excursion"] == 1.25
    assert json.loads(path["claims_json"]) == CLAIMS_BOUNDARY

    neutral_score = _score(0.25)
    neutral_checkpoint_id = repository.record_checkpoint(
        metadata,
        symbol="AAL",
        session=SESSION,
        checkpoint=10,
        bar_start_utc=START + timedelta(minutes=5),
        bar_end_utc=START + timedelta(minutes=10),
        score=neutral_score,
        session_context_hash="d" * 64,
        feature_values={"x": 1.0},
        eligible=True,
        feature_freshness="fresh",
        rejection_reasons=(),
    )
    neutral_quiet_checkpoint_id = repository.record_quiet_checkpoint(
        metadata,
        checkpoint_id=neutral_checkpoint_id,
        symbol="AAL",
        session=SESSION,
        checkpoint=10,
        snapshot=classify_quiet_state(
            probability=neutral_score.probability,
            previous_probability=score.probability,
            model_hash=neutral_score.model_hash,
            feature_hash=neutral_score.feature_hash,
            data_quality_status="valid",
        ),
        eligible=True,
    )
    neutral = NeutralControlSampler().evaluate(
        session=SESSION,
        symbol="AAL",
        checkpoint=10,
        model_hash=neutral_score.model_hash,
        probability=0.25,
        eligible=True,
    )
    control_id = repository.record_neutral_control(
        metadata,
        quiet_checkpoint_id=neutral_quiet_checkpoint_id,
        decision=neutral,
        trigger_timestamp=START + timedelta(minutes=10),
        data_quality_flags=(),
    )
    assert control_id.startswith("m1c-neutral-")
    capacity_contract = OptionContract(
        underlying_con_id=1,
        con_id=201,
        expiry=date(2026, 7, 31),
        dte=4,
        dte_bucket=DteBucket.THREE_TO_FIVE_DTE,
        strike=100.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )
    repository.record_quiet_option_plan(
        metadata,
        observation_id=control_id,
        plan=OptionContractPlan(
            contracts=(capacity_contract,),
            requested_contract_count=2,
            maximum_contracts=1,
            capacity_reduced=True,
            missing_buckets=(),
        ),
    )
    repository.record_quiet_option_contract(
        metadata,
        observation_id=control_id,
        contract=capacity_contract,
        selection_rank=1,
        selection_roles=(),
        resolution_status="recording",
        rejection_reason=None,
        recording_started_at_utc=START,
        recording_ends_at_utc=START + timedelta(minutes=60),
    )
    with database._connect() as connection:
        capacity_context = connection.execute(
            """
            SELECT option_plan_capacity_reduced, option_context_valid
            FROM quiet_state_observation_v0 WHERE observation_id = ?
            """,
            (control_id,),
        ).fetchone()
    assert capacity_context["option_plan_capacity_reduced"] == 1
    assert capacity_context["option_context_valid"] == 0


def test_quiet_session_state_restores_last_eligible_and_episode(tmp_path: Path) -> None:
    database = ProspectiveRepository(tmp_path / "quiet.sqlite3")
    database.migrate()
    metadata = _metadata("quiet-restore")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    score = _score(0.13)
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
    )
    snapshot = classify_quiet_state(
        probability=score.probability,
        previous_probability=None,
        model_hash=score.model_hash,
        feature_hash=score.feature_hash,
        data_quality_status="valid",
    )
    quiet_checkpoint_id = repository.record_quiet_checkpoint(
        metadata,
        checkpoint_id=checkpoint_id,
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        snapshot=snapshot,
        eligible=True,
    )
    decision = QuietEpisodeTracker().evaluate(
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        trigger_bar_end=START,
        probability=score.probability,
    )
    repository.record_quiet_episode(
        metadata,
        quiet_checkpoint_id=quiet_checkpoint_id,
        decision=decision,
        scientific_recording_valid=True,
    )
    repository.mark_checkpoint_complete(
        metadata,
        checkpoint_id=checkpoint_id,
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
    )

    probability, timestamp, count = repository.quiet_session_state(
        run_id=metadata.run_id,
        symbol="AAL",
        session=SESSION,
    )

    assert probability == 0.13
    assert timestamp == START
    assert count == 1


def test_quiet_virtual_ledger_contains_only_quiet_short_premium_structures(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "quiet-virtual-ledger.sqlite3")
    database.migrate()
    metadata = _metadata("quiet-virtual-ledger")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    score = _score(0.13)
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
    )
    quiet_checkpoint_id = repository.record_quiet_checkpoint(
        metadata,
        checkpoint_id=checkpoint_id,
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        snapshot=classify_quiet_state(
            probability=score.probability,
            previous_probability=None,
            model_hash=score.model_hash,
            feature_hash=score.feature_hash,
            data_quality_status="valid",
        ),
        eligible=True,
    )
    observation_id = repository.record_quiet_episode(
        metadata,
        quiet_checkpoint_id=quiet_checkpoint_id,
        decision=QuietEpisodeTracker().evaluate(
            symbol="AAL",
            session=SESSION,
            checkpoint=6,
            trigger_bar_end=START,
            probability=score.probability,
        ),
        scientific_recording_valid=True,
    )
    payload = {
        "closing_debit": 40.0,
        "configured_commission_pnl": 7.4,
        "legs": [
            {"side": "short", "con_id": 101, "strike": 100.0, "right": "C"},
            {"side": "short", "con_id": 102, "strike": 100.0, "right": "P"},
            {"side": "long", "con_id": 103, "strike": 105.0, "right": "C"},
            {"side": "long", "con_id": 104, "strike": 95.0, "right": "P"},
        ],
    }
    for structure_type in ("DELTA_IRON_CONDOR", "LONG_CALL"):
        repository.record_quiet_shadow_structure(
            metadata,
            observation_id=observation_id,
            structure_type=structure_type,
            dte_bucket="1DTE",
            horizon_label="15m",
            horizon_minutes=15,
            payload=payload,
            opening_credit_or_debit=50.0,
            maximum_defined_risk=450.0,
            conservative_pnl=10.0,
            return_on_maximum_risk=10.0 / 450.0,
            short_strike_touched=False,
            protective_wing_touched=False,
            attempted=True,
            complete_quote_quality=True,
            strict_quote_quality=True,
            quality_status="strict_quality",
            quality_flags=(),
        )

    with database._connect() as connection:
        rows = connection.execute("SELECT * FROM quiet_state_virtual_position_v1").fetchall()

    assert len(rows) == 1
    row = rows[0]
    assert row["ledger_scope"] == "quiet_state_short_premium"
    assert row["observation_id"] == observation_id
    assert row["observation_kind"] == "quiet_bottom_10"
    assert row["structure_type"] == "DELTA_IRON_CONDOR"
    assert row["lifecycle_state"] == "CLOSED"
    assert row["opening_net_credit"] == pytest.approx(50.0)
    assert row["closing_net_debit"] == pytest.approx(40.0)
    assert row["conservative_pnl"] == pytest.approx(10.0)
    assert row["leg_count"] == 4
    assert row["execution_claimed"] == 0
    assert row["paper_fill_claimed"] == 0

    projected = ProspectiveReadStore(
        database.database_path,
        run_id=metadata.run_id,
    ).quiet_state_virtual_positions_v1()
    assert len(projected) == 1
    assert projected[0]["structure_type"] == "DELTA_IRON_CONDOR"
    assert projected[0]["legs"][0]["side"] == "short"
    assert projected[0]["conservative_pnl"] == pytest.approx(10.0)
