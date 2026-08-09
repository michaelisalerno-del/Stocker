from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

import stocker_runtime.storage.legacy_import as legacy_import_module
from stocker_runtime.cli import app
from stocker_runtime.storage import (
    LEGACY_SCHEMA_DIGESTS,
    LegacyImportError,
    import_legacy_database,
    verify_database,
)

ROOT = Path(__file__).parents[1]
LEGACY_MIGRATIONS = ROOT / "tests/fixtures/legacy_prospective_migrations"
MIGRATION_NAMES = (
    "0001_prospective.sql",
    "0002_runtime_state.sql",
    "0003_option_computations.sql",
    "0004_underlying_quote_identity.sql",
    "0005_market_data_budget_events.sql",
    "0006_parallel_source_bars.sql",
    "0007_frozen_m1c_recorder_v0.sql",
    "0008_promotion_decisions_and_option_model_sources.sql",
    "0009_quiet_state_options_shadow_v0.sql",
    "0010_ibkr_budget_transfer_v0.sql",
    "0011_m1c_checkpoint_completion_v0.sql",
    "0011_m1c_tail_phase_v1.sql",
    "0012_m1c_signed_market_shock_v1.sql",
    "0012_option_schedule_degradation_v0.sql",
    "0013_m1c_opening_market_transition_v1.sql",
    "0014_m1c_prospective_opening_reversal_v1.sql",
    "0015_m1c_prospective_opening_reversal_v1_1.sql",
    "0016_prospective_recorder_hardening_v1.sql",
    "0017_callback_raw_only_recovery_v1.sql",
    "0018_virtual_position_ledgers_v1.sql",
    "0019_virtual_position_ledger_evidence_v1.sql",
    "0020_opening_reversal_shadow_capture_v1.sql",
    "0021_opening_reversal_activation_run_binding_v1.sql",
    "0022_web_read_projections_v0.sql",
    "0023_web_latest_state_v0.sql",
    "0024_m1c_validity_separation_v1.sql",
    "0025_parallel_source_capture_recovery_v1.sql",
    "0026_opening_leader_continuation_v0.sql",
    "0027_option_risk_accounting_v0.sql",
    "0028_web_latest_subscription_state_v0.sql",
    "0029_m1c_diagnostic_quality_flags_v0.sql",
    "0030_quiet_checkpoint_quote_audit_v0.sql",
)
PRODUCTION_0030_LEDGER_NAMES = (
    *MIGRATION_NAMES[:11],
    "0012_option_schedule_degradation_v0.sql",
    "0011_m1c_tail_phase_v1.sql",
    "0012_m1c_signed_market_shock_v1.sql",
    *MIGRATION_NAMES[14:],
)
LEGACY_SCIENTIFIC_CLASSIFICATION = (
    "Previous-close front-options context + current intraday H0 stock condition -> "
    "improved prediction that near-term underlying movement exceeds previous-close "
    "option-implied movement."
)


def _legacy_database(path: Path, prefix: int) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE schema_migrations (version TEXT PRIMARY KEY, applied_at_utc TEXT NOT NULL)"
    )
    for name in MIGRATION_NAMES[:prefix]:
        connection.executescript((LEGACY_MIGRATIONS / name).read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
            (name, "2026-01-01T00:00:00+00:00"),
        )
    connection.commit()
    return connection


def _replace_migration_ledger(
    connection: sqlite3.Connection,
    names: tuple[str, ...],
) -> None:
    connection.execute("DELETE FROM schema_migrations")
    connection.executemany(
        "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
        ((name, "2026-01-01T00:00:00+00:00") for name in names),
    )
    connection.commit()


def _seed_representative_legacy_rows(connection: sqlite3.Connection) -> None:
    timestamp = "2026-01-02T14:30:00+00:00"
    later = "2026-01-02T14:35:00+00:00"
    connection.execute(
        "INSERT INTO prospective_run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-shadow",
            timestamp,
            "1.0",
            "abc123",
            "model-v1",
            "universe-v1",
            "cohort-a",
            timestamp,
            "shadow",
            "stopped",
            LEGACY_SCIENTIFIC_CLASSIFICATION,
        ),
    )
    envelope_id = connection.execute(
        """
        INSERT INTO evidence_envelope(
            run_id, prospective_start_utc, app_version, git_commit, model_artifact_id,
            universe_id, cohort, source_timestamps_json, recorded_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-shadow",
            timestamp,
            "1.0",
            "abc123",
            "model-v1",
            "universe-v1",
            "cohort-a",
            "[]",
            timestamp,
        ),
    ).lastrowid
    assert envelope_id is not None
    underlying_id = connection.execute(
        """
        INSERT INTO underlying_contract(
            envelope_id, run_id, symbol, con_id, exchange, currency, local_symbol,
            qualification_status, rejection_reason
        ) VALUES (?, 'legacy-shadow', 'ABC', 101, 'SMART', 'USD', 'ABC', 'qualified', NULL)
        """,
        (envelope_id,),
    ).lastrowid
    assert underlying_id is not None
    connection.execute(
        """
        INSERT INTO underlying_bar(
            envelope_id, run_id, symbol, con_id, bar_start_utc, bar_end_utc, session_date,
            open, high, low, close, activity_value, activity_semantic_label, bar_source,
            source_timestamp_utc, receive_timestamp_utc, completeness, feature_as_of_utc,
            m0_probability, m1_probability, frozen_threshold, model_bundle_id,
            feature_schema_hash, eligibility, rejection_reason
        ) VALUES (?, 'legacy-shadow', 'ABC', 101, ?, ?, '2026-01-02', 10, 12, 9, 11,
                  1000, 'volume', 'ibkr_realtime_bar_5_second_aggregation', ?, ?,
                  'complete', ?, 0.1, 0.8, 0.7,
                  'model-v1', ?, 1, NULL)
        """,
        (envelope_id, timestamp, later, later, later, later, "f" * 64),
    )
    score_id = connection.execute(
        """
        INSERT INTO model_score(
            envelope_id, run_id, cohort, symbol, bar_end_utc, session_date, feature_as_of_utc,
            m0_probability, m1_probability, frozen_threshold, model_bundle_id,
            feature_schema_hash, eligibility, rejection_reason, score_label
        ) VALUES (?, 'legacy-shadow', 'cohort-a', 'ABC', ?, '2026-01-02', ?, 0.1, 0.8,
                  0.7, 'model-v1', ?, 1, NULL, 'above_threshold')
        """,
        (envelope_id, later, later, "f" * 64),
    ).lastrowid
    assert score_id is not None
    connection.execute(
        """
        INSERT INTO signal_episode(
            id, envelope_id, run_id, cohort, symbol, model_bundle_id,
            crossing_timestamp_utc, idempotency_key, startup_above_threshold, status
        ) VALUES ('episode-1', ?, 'legacy-shadow', 'cohort-a', 'ABC', 'model-v1', ?,
                  'episode-key-1', 0, 'complete')
        """,
        (envelope_id, later),
    )
    connection.execute(
        """
        INSERT INTO signal_checkpoint(
            envelope_id, run_id, signal_episode_id, model_score_id, checkpoint_timestamp_utc,
            m1_probability, frozen_threshold
        ) VALUES (?, 'legacy-shadow', 'episode-1', ?, ?, 0.8, 0.7)
        """,
        (envelope_id, score_id, later),
    )
    option_id = connection.execute(
        """
        INSERT INTO option_contract(
            envelope_id, run_id, underlying_con_id, con_id, local_symbol, expiry, strike,
            right, multiplier, exchange, trading_class, dte_bucket, qualification_status,
            rejection_reason
        ) VALUES (?, 'legacy-shadow', 101, 202, 'ABC  C', '2026-01-03', 11, 'C', '100',
                  'SMART', 'ABC', '1DTE', 'qualified', NULL)
        """,
        (envelope_id,),
    ).lastrowid
    assert option_id is not None
    capture_id = connection.execute(
        """
        INSERT INTO option_surface_capture(
            envelope_id, run_id, signal_episode_id, dte_bucket, target_timestamp_utc,
            actual_quote_timestamp_utc, capture_lag_seconds, market_data_type,
            quote_freshness, completeness, connection_status, budget_status,
            missing_contract_reason, missing_quote_reason, subscription_error, capture_status
        ) VALUES (?, 'legacy-shadow', 'episode-1', '1DTE', ?, ?, 0, 'real_time', 'fresh',
                  'complete', 'connected', 'within_budget', NULL, NULL, NULL, 'captured')
        """,
        (envelope_id, later, later),
    ).lastrowid
    assert capture_id is not None
    connection.execute(
        """
        INSERT INTO option_quote(
            envelope_id, run_id, surface_capture_id, option_contract_id, bid, ask, bid_size,
            ask_size, last, last_size, volume, open_interest, computation_source,
            provider_timestamp_utc, receive_timestamp_utc, market_data_type,
            staleness_seconds, completeness, permission_error
        ) VALUES (?, 'legacy-shadow', ?, ?, 1.0, 1.2, 5, 6, 1.1, 2, 100, 200, NULL,
                  ?, ?, 'real_time', 0, 'complete', NULL)
        """,
        (envelope_id, capture_id, option_id, later, later),
    )
    connection.execute(
        """
        INSERT INTO shadow_structure(
            id, envelope_id, run_id, signal_episode_id, cohort, symbol, dte_bucket,
            structure_type, entry_debit, multiplier, estimated_fees, spread_quality,
            completeness, rejection_reason, quoted_research_ledger
        ) VALUES ('shadow-1', ?, 'legacy-shadow', 'episode-1', 'cohort-a', 'ABC', '1DTE',
                  'LONG_CALL', 1.2, 100, 1.0, 'complete', 'complete', NULL, 1)
        """,
        (envelope_id,),
    )
    connection.execute(
        """
        INSERT INTO shadow_leg(
            envelope_id, run_id, shadow_structure_id, option_contract_id, leg_role,
            quantity, entry_side, entry_price, quote_timestamp_utc
        ) VALUES (?, 'legacy-shadow', 'shadow-1', ?, 'long', 1, 'ask', 1.2, ?)
        """,
        (envelope_id, option_id, later),
    )
    connection.execute(
        """
        INSERT INTO shadow_horizon_valuation(
            envelope_id, run_id, shadow_structure_id, horizon_minutes, target_timestamp_utc,
            actual_quote_timestamp_utc, capture_lag_seconds, exit_credit,
            gross_return_on_debit, gross_pnl, estimated_fees, market_data_type,
            completeness, rejection_reason
        ) VALUES (?, 'legacy-shadow', 'shadow-1', 15, ?, ?, 0, 1.5, 0.25, 30, 1,
                  'real_time', 'complete', NULL)
        """,
        (envelope_id, later, later),
    )
    connection.execute(
        "INSERT INTO audit_event(envelope_id, run_id, sequence, event_type, actor, message, "
        "payload_json) VALUES (?, 'legacy-shadow', 1, 'legacy-only', 'test', 'omitted', '{}')",
        (envelope_id,),
    )
    connection.commit()


@pytest.mark.parametrize(
    ("migration_name", "expected_digest"),
    (
        (
            "0027_option_risk_accounting_v0.sql",
            "1326438ca6ac196c0a71df64ded75212e3b8ef4d8b36306bb7d5afbdc7894290",
        ),
        (
            "0028_web_latest_subscription_state_v0.sql",
            "152959a5cf4cd751d085ccec23cb7ccac930d1d536972740fa87af639e2e91f7",
        ),
        (
            "0029_m1c_diagnostic_quality_flags_v0.sql",
            "40121e046376b49d1a5c35c2d295b5c5f390527f9662cf06861af573d74ac7f4",
        ),
        (
            "0030_quiet_checkpoint_quote_audit_v0.sql",
            "afb0e2d62ddd671daaaa9fbab7b2100be7efa7649443b47f358acb2f36319d59",
        ),
    ),
)
def test_import_fixtures_match_the_deployed_schema_prefix_digests(
    tmp_path: Path,
    migration_name: str,
    expected_digest: str,
) -> None:
    source = tmp_path / f"{migration_name}.sqlite3"
    prefix = MIGRATION_NAMES.index(migration_name) + 1
    with _legacy_database(source, prefix) as legacy:
        assert legacy_import_module._schema_digest(legacy) == expected_digest


@pytest.mark.parametrize("prefix", range(1, len(MIGRATION_NAMES) + 1))
def test_import_accepts_every_frozen_legacy_schema_and_is_deterministic(
    tmp_path: Path, prefix: int
) -> None:
    source = tmp_path / f"source-{prefix}.sqlite3"
    with _legacy_database(source, prefix):
        pass
    source_bytes = source.read_bytes()

    first = import_legacy_database(
        source,
        tmp_path / f"target-{prefix}-a.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )
    second = import_legacy_database(
        source,
        tmp_path / f"target-{prefix}-b.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )

    assert source.read_bytes() == source_bytes
    assert first.source_database_hash == hashlib.sha256(source_bytes).hexdigest()
    assert first.source_schema_digest == second.source_schema_digest
    assert first.target_digest == second.target_digest
    assert first.source_row_count == first.imported_row_count + first.omitted_row_count
    assert first.verification_status == "verified"
    verify_database(first.target_path)


def test_import_accepts_exact_deployed_schema_0030_ledger_variant(tmp_path: Path) -> None:
    source = tmp_path / "production-ledger.sqlite3"
    with _legacy_database(source, len(MIGRATION_NAMES)) as legacy:
        _replace_migration_ledger(legacy, PRODUCTION_0030_LEDGER_NAMES)
        assert legacy_import_module._schema_digest(legacy) == LEGACY_SCHEMA_DIGESTS[-1][1]
    source_bytes = source.read_bytes()

    result = import_legacy_database(source, tmp_path / "target.sqlite3", started_at_us=1)

    assert source.read_bytes() == source_bytes
    assert result.source_schema_digest == LEGACY_SCHEMA_DIGESTS[-1][1]
    verify_database(result.target_path)


def test_import_rejects_every_other_schema_0030_ledger_reordering(tmp_path: Path) -> None:
    source = tmp_path / "reordered-ledger.sqlite3"
    unsupported = list(PRODUCTION_0030_LEDGER_NAMES)
    unsupported[-2], unsupported[-1] = unsupported[-1], unsupported[-2]
    with _legacy_database(source, len(MIGRATION_NAMES)) as legacy:
        _replace_migration_ledger(legacy, tuple(unsupported))
        assert legacy_import_module._schema_digest(legacy) == LEGACY_SCHEMA_DIGESTS[-1][1]

    with pytest.raises(LegacyImportError, match="not an accepted prefix"):
        import_legacy_database(source, tmp_path / "target.sqlite3", started_at_us=1)


def test_import_digest_does_not_depend_on_migration_clock(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    with _legacy_database(source, 1):
        pass

    first = import_legacy_database(
        source,
        tmp_path / "target-a.sqlite3",
        started_at_us=1,
    )
    second = import_legacy_database(
        source,
        tmp_path / "target-b.sqlite3",
        started_at_us=2,
    )

    assert first.target_digest == second.target_digest


def test_import_maps_generic_v2_rows_and_reconciles_every_source_row(tmp_path: Path) -> None:
    source = tmp_path / "legacy.sqlite3"
    with _legacy_database(source, 1) as legacy:
        _seed_representative_legacy_rows(legacy)

    result = import_legacy_database(
        source,
        tmp_path / "stocker-v2.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )
    report = json.loads(result.reconciliation_path.read_text(encoding="utf-8"))

    assert report["source_row_count"] == result.source_row_count
    assert report["imported_row_count"] == result.imported_row_count
    assert report["omitted_row_count"] == result.omitted_row_count
    assert all(
        item["source_rows"] == item["imported_rows"] + item["omitted_rows"]
        for item in report["tables"]
    )
    assert len({item["row_classification_hash"] for item in report["tables"]}) == len(
        report["tables"]
    )
    assert next(item for item in report["tables"] if item["table"] == "audit_event")[
        "omission_reasons"
    ] == {"legacy_runtime_only": 1}

    with sqlite3.connect(result.target_path) as target:
        target.row_factory = sqlite3.Row
        run = target.execute("SELECT mode, data_class, status, config_hash FROM runs").fetchone()
        assert tuple(run[:3]) == ("shadow", "shadow_protected", "stopped")
        assert run[3] == legacy_import_module._sha(
            {
                "app_version": "1.0",
                "cohort": "cohort-a",
                "model_artifact_id": "model-v1",
                "scientific_classification": LEGACY_SCIENTIFIC_CLASSIFICATION,
                "universe_id": "universe-v1",
            }
        )
        parameters = json.loads(
            target.execute("SELECT parameters_json FROM idea_instances").fetchone()[0]
        )
        assert parameters == {"legacy_scientific_classification": LEGACY_SCIENTIFIC_CLASSIFICATION}
        assert target.execute("SELECT count(*) FROM instruments").fetchone()[0] == 2
        assert (
            target.execute(
                "SELECT count(*) FROM market_events WHERE feed_kind != 'migration'"
            ).fetchone()[0]
            == 2
        )
        kinds = {row[0] for row in target.execute("SELECT output_kind FROM idea_outputs")}
        assert {"observation", "signal", "proposed_trade"}.issubset(kinds)
        inputs = target.execute(
            "SELECT output.output_kind, event.feed_kind, event.event_id "
            "FROM idea_outputs output "
            "JOIN idea_output_inputs input ON input.output_id=output.output_id "
            "JOIN market_events event ON event.event_id=input.event_id "
            "ORDER BY output.output_id, input.input_ordinal"
        ).fetchall()
        assert inputs
        assert all(row[1] != "migration" for row in inputs)
        position = target.execute("SELECT lifecycle, data_class FROM shadow_positions").fetchone()
        assert tuple(position) == ("closed", "shadow_protected")
        leg = target.execute(
            "SELECT side, entry_market_event_id, exit_market_event_id, exit_price FROM shadow_legs"
        ).fetchone()
        assert leg[0] == "buy"
        assert leg[1] is not None
        assert leg[2] == leg[1]
        assert leg[3] == 1.0
        assert target.execute("SELECT count(*) FROM shadow_marks").fetchone()[0] == 1
        mark_payload = json.loads(
            target.execute("SELECT payload_json FROM shadow_marks").fetchone()[0]
        )
        assert mark_payload["source_market_event_ids"] == [leg[2]]
        outcome = target.execute(
            "SELECT completeness, gross_pnl, net_pnl, payload_json FROM shadow_outcomes"
        ).fetchone()
        assert tuple(outcome[:3]) == ("complete", 30.0, 29.0)
        assert json.loads(outcome[3])["source_market_event_ids"] == [leg[2]]
        manifest = target.execute(
            "SELECT source_row_count, imported_row_count, omitted_row_count, "
            "verification_status, target_digest, importer_version FROM migration_manifests"
        ).fetchone()
        assert manifest[0] == manifest[1] + manifest[2]
        assert manifest[3] == "verified"
        assert manifest[4] == result.target_digest
        assert "+reconciliation." in manifest[5]
        table_names = {
            row[0]
            for row in target.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert not any(
            name.startswith(("orders", "fills", "broker_", "accounts", "portfolio_", "risk_"))
            for name in table_names
        )
    assert b"eodhd" not in result.target_path.read_bytes().lower()
    assert result.target_path.stat().st_mode & 0o777 == 0o640


def test_import_maps_production_short_bid_leg_to_sell(tmp_path: Path) -> None:
    source = tmp_path / "legacy-short-leg.sqlite3"
    with _legacy_database(source, 1) as legacy:
        _seed_representative_legacy_rows(legacy)
        legacy.execute("UPDATE shadow_leg SET leg_role='short', entry_side='bid', entry_price=1.0")
        legacy.commit()

    result = import_legacy_database(
        source,
        tmp_path / "stocker-v2.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )

    with sqlite3.connect(result.target_path) as target:
        assert target.execute(
            "SELECT side, entry_price, exit_price FROM shadow_legs"
        ).fetchone() == ("sell", 1.0, 1.2)


def test_import_omits_shadow_rows_from_a_record_only_run(tmp_path: Path) -> None:
    source = tmp_path / "legacy-record-only.sqlite3"
    with _legacy_database(source, 1) as legacy:
        _seed_representative_legacy_rows(legacy)
        legacy.execute("UPDATE prospective_run SET mode='record_only'")
        legacy.commit()

    result = import_legacy_database(
        source,
        tmp_path / "stocker-v2.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )
    report = json.loads(result.reconciliation_path.read_text(encoding="utf-8"))
    tables = {item["table"]: item for item in report["tables"]}

    assert tables["shadow_structure"]["omission_reasons"] == {"shadow_run_not_imported": 1}
    assert tables["shadow_leg"]["omission_reasons"] == {"shadow_parent_omitted": 1}
    assert tables["shadow_horizon_valuation"]["omission_reasons"] == {"shadow_parent_omitted": 1}
    with sqlite3.connect(result.target_path) as target:
        assert target.execute("SELECT mode, data_class FROM runs").fetchone() == (
            "prospective_record",
            "prospective_protected",
        )
        assert target.execute("SELECT count(*) FROM shadow_positions").fetchone()[0] == 0
        assert (
            target.execute(
                "SELECT count(*) FROM idea_outputs WHERE output_kind='proposed_trade'"
            ).fetchone()[0]
            == 0
        )


def test_import_omits_shadow_position_without_exact_entry_market_event(tmp_path: Path) -> None:
    source = tmp_path / "legacy-missing-entry-event.sqlite3"
    with _legacy_database(source, 1) as legacy:
        _seed_representative_legacy_rows(legacy)
        legacy.execute("DELETE FROM option_quote")
        legacy.commit()

    result = import_legacy_database(
        source,
        tmp_path / "stocker-v2.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )
    report = json.loads(result.reconciliation_path.read_text(encoding="utf-8"))
    tables = {item["table"]: item for item in report["tables"]}

    assert tables["shadow_structure"]["omission_reasons"] == {"shadow_leg_evidence_incomplete": 1}
    with sqlite3.connect(result.target_path) as target:
        assert target.execute("SELECT count(*) FROM shadow_positions").fetchone()[0] == 0
        assert (
            target.execute(
                "SELECT count(*) FROM idea_outputs WHERE output_kind='proposed_trade'"
            ).fetchone()[0]
            == 0
        )


def test_import_omits_outputs_without_exact_market_event_provenance(tmp_path: Path) -> None:
    source = tmp_path / "legacy-output-provenance.sqlite3"
    with _legacy_database(source, 1) as legacy:
        _seed_representative_legacy_rows(legacy)
        legacy.execute("DELETE FROM underlying_bar")
        legacy.commit()

    result = import_legacy_database(
        source,
        tmp_path / "stocker-v2.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )
    report = json.loads(result.reconciliation_path.read_text(encoding="utf-8"))
    tables = {item["table"]: item for item in report["tables"]}

    for table in ("model_score", "signal_episode", "signal_checkpoint"):
        assert tables[table]["imported_rows"] == 0
        assert tables[table]["omission_reasons"] == {"output_provenance_incomplete": 1}
    with sqlite3.connect(result.target_path) as target:
        assert (
            target.execute(
                "SELECT count(*) FROM idea_output_inputs input "
                "JOIN market_events event ON event.event_id=input.event_id "
                "WHERE event.feed_kind='migration'"
            ).fetchone()[0]
            == 0
        )


def test_import_does_not_close_shadow_without_exact_exit_market_events(tmp_path: Path) -> None:
    source = tmp_path / "legacy-missing-exit-event.sqlite3"
    with _legacy_database(source, 1) as legacy:
        _seed_representative_legacy_rows(legacy)
        legacy.execute(
            "UPDATE shadow_horizon_valuation "
            "SET target_timestamp_utc=?, actual_quote_timestamp_utc=?",
            ("2026-01-02T14:40:00+00:00", "2026-01-02T14:40:00+00:00"),
        )
        legacy.commit()

    result = import_legacy_database(
        source,
        tmp_path / "stocker-v2.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )
    report = json.loads(result.reconciliation_path.read_text(encoding="utf-8"))
    tables = {item["table"]: item for item in report["tables"]}

    assert tables["shadow_horizon_valuation"]["omission_reasons"] == {
        "shadow_mark_evidence_incomplete": 1
    }
    with sqlite3.connect(result.target_path) as target:
        assert target.execute(
            "SELECT lifecycle, closed_at_us FROM shadow_positions"
        ).fetchone() == ("open", None)
        assert target.execute(
            "SELECT exit_market_event_id, exit_price FROM shadow_legs"
        ).fetchone() == (None, None)
        assert target.execute("SELECT count(*) FROM shadow_marks").fetchone()[0] == 0
        assert target.execute("SELECT count(*) FROM shadow_outcomes").fetchone()[0] == 0


def test_import_closed_shadow_outcome_uses_the_same_evidence_as_leg_exits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-mixed-horizons.sqlite3"
    with _legacy_database(source, 1) as legacy:
        _seed_representative_legacy_rows(legacy)
        envelope_id = legacy.execute("SELECT id FROM evidence_envelope").fetchone()[0]
        option_id = legacy.execute("SELECT id FROM option_contract").fetchone()[0]
        later = "2026-01-02T14:40:00+00:00"
        capture_id = legacy.execute(
            """
            INSERT INTO option_surface_capture(
                envelope_id, run_id, signal_episode_id, dte_bucket, target_timestamp_utc,
                actual_quote_timestamp_utc, capture_lag_seconds, market_data_type,
                quote_freshness, completeness, connection_status, budget_status,
                missing_contract_reason, missing_quote_reason, subscription_error,
                capture_status
            ) VALUES (?, 'legacy-shadow', 'episode-1', '1DTE', ?, ?, 0, 'real_time',
                      'fresh', 'incomplete', 'connected', 'within_budget', NULL, NULL,
                      NULL, 'captured')
            """,
            (envelope_id, later, later),
        ).lastrowid
        assert capture_id is not None
        legacy.execute(
            """
            INSERT INTO option_quote(
                envelope_id, run_id, surface_capture_id, option_contract_id, bid, ask,
                bid_size, ask_size, last, last_size, volume, open_interest,
                computation_source, provider_timestamp_utc, receive_timestamp_utc,
                market_data_type, staleness_seconds, completeness, permission_error
            ) VALUES (?, 'legacy-shadow', ?, ?, 0.8, 1.0, 5, 6, 0.9, 2, 100, 200,
                      NULL, ?, ?, 'real_time', 0, 'incomplete', NULL)
            """,
            (envelope_id, capture_id, option_id, later, later),
        )
        legacy.execute(
            """
            INSERT INTO shadow_horizon_valuation(
                envelope_id, run_id, shadow_structure_id, horizon_minutes,
                target_timestamp_utc, actual_quote_timestamp_utc, capture_lag_seconds,
                exit_credit, gross_return_on_debit, gross_pnl, estimated_fees,
                market_data_type, completeness, rejection_reason
            ) VALUES (?, 'legacy-shadow', 'shadow-1', 30, ?, ?, 0, 0.8, -0.33, -40, 1,
                      'real_time', 'incomplete', 'PARTIAL_QUOTE')
            """,
            (envelope_id, later, later),
        )
        legacy.commit()

    result = import_legacy_database(
        source,
        tmp_path / "stocker-v2.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )

    with sqlite3.connect(result.target_path) as target:
        target.row_factory = sqlite3.Row
        position = target.execute("SELECT lifecycle, closed_at_us FROM shadow_positions").fetchone()
        leg = target.execute("SELECT exit_market_event_id FROM shadow_legs").fetchone()
        outcome = target.execute(
            "SELECT outcome_at_us, completeness, payload_json FROM shadow_outcomes"
        ).fetchone()
        assert position["lifecycle"] == "closed"
        assert position["closed_at_us"] == outcome["outcome_at_us"]
        assert outcome["completeness"] == "complete"
        assert json.loads(outcome["payload_json"])["source_market_event_ids"] == [
            leg["exit_market_event_id"]
        ]


def test_import_rejects_replay_run_and_unverifiable_underlying_quote(tmp_path: Path) -> None:
    source = tmp_path / "legacy-replay.sqlite3"
    with _legacy_database(source, 1) as legacy:
        _seed_representative_legacy_rows(legacy)
        envelope_id = legacy.execute("SELECT id FROM evidence_envelope").fetchone()[0]
        legacy.execute("UPDATE underlying_bar SET bar_source='deterministic_replay'")
        legacy.execute(
            """
            INSERT INTO underlying_quote(
                envelope_id, run_id, signal_episode_id, target_timestamp_utc,
                actual_quote_timestamp_utc, capture_lag_seconds, bid, ask, bid_size,
                ask_size, last, last_size, midpoint, spread, provider_timestamp_utc,
                receive_timestamp_utc, market_data_type, freshness, completeness,
                capture_status, missing_quote_reason
            ) VALUES (?, 'legacy-shadow', 'episode-1', ?, ?, 0, 10, 11, 1, 1, 10.5, 1,
                      10.5, 1, ?, ?, 'real_time', 'fresh', 'complete', 'captured', NULL)
            """,
            (
                envelope_id,
                "2026-01-02T14:35:00+00:00",
                "2026-01-02T14:35:00+00:00",
                "2026-01-02T14:35:00+00:00",
                "2026-01-02T14:35:00+00:00",
            ),
        )
        legacy.commit()

    result = import_legacy_database(
        source,
        tmp_path / "stocker-v2.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )
    report = json.loads(result.reconciliation_path.read_text(encoding="utf-8"))
    tables = {item["table"]: item for item in report["tables"]}

    assert tables["prospective_run"]["omission_reasons"] == {"source_provenance_not_ibkr": 1}
    assert tables["underlying_quote"]["omission_reasons"] == {"source_provenance_unverifiable": 1}
    with sqlite3.connect(result.target_path) as target:
        assert target.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
        assert target.execute("SELECT count(*) FROM market_events").fetchone()[0] == 0


def test_import_preserves_closed_diagnostics_and_archives_active_rows(tmp_path: Path) -> None:
    source = tmp_path / "legacy-diagnostics.sqlite3"
    prefix = MIGRATION_NAMES.index("0016_prospective_recorder_hardening_v1.sql") + 1
    with _legacy_database(source, prefix) as legacy:
        _seed_representative_legacy_rows(legacy)
        legacy.executemany(
            """
            INSERT INTO operational_incident_v1(
                incident_id, run_id, occurred_at_utc, component, severity,
                stable_error_code, error_class, evidence_loss_possible, details_json,
                resolved_at_utc
            ) VALUES (?, 'legacy-shadow', ?, 'recorder', ?, ?, 'RuntimeError', 0, '{}', ?)
            """,
            (
                (
                    "closed-incident",
                    "2026-01-02T14:31:00+00:00",
                    "fatal",
                    "LEGACY_CLOSED",
                    "2026-01-02T14:32:00+00:00",
                ),
                (
                    "active-incident",
                    "2026-01-02T14:33:00+00:00",
                    "degraded",
                    "LEGACY_ACTIVE",
                    None,
                ),
            ),
        )
        legacy.executemany(
            """
            INSERT INTO gap_incident_v1(
                gap_id, run_id, recorder_generation, symbol, stream_kind,
                connection_generation, start_timestamp_utc, end_timestamp_utc,
                detection_timestamp_utc, cause_code, severity, recoverability,
                affected_episode_ids_json, resolution_timestamp_utc
            ) VALUES (?, 'legacy-shadow', 1, 'ABC', 'bars', 1, ?, ?, ?, ?, 'degraded',
                      ?, '[]', ?)
            """,
            (
                (
                    "closed-gap",
                    "2026-01-02T14:31:00+00:00",
                    "2026-01-02T14:32:00+00:00",
                    "2026-01-02T14:32:00+00:00",
                    "LEGACY_GAP",
                    "recovered",
                    "2026-01-02T14:32:00+00:00",
                ),
                (
                    "active-gap",
                    "2026-01-02T14:33:00+00:00",
                    None,
                    "2026-01-02T14:33:00+00:00",
                    "LEGACY_ACTIVE_GAP",
                    "unknown",
                    None,
                ),
            ),
        )
        legacy.commit()

    result = import_legacy_database(
        source,
        tmp_path / "stocker-v2.sqlite3",
        started_at_us=1_800_000_000_000_000,
    )

    with sqlite3.connect(result.target_path) as target:
        target.row_factory = sqlite3.Row
        gaps = target.execute("SELECT reason, resolved_at_us FROM gaps ORDER BY gap_id").fetchall()
        assert [(row["reason"], row["resolved_at_us"] is not None) for row in gaps] == [
            ("LEGACY_GAP", True)
        ]
        incidents = target.execute(
            "SELECT code, resolved_at_us, details_json FROM incidents ORDER BY code"
        ).fetchall()
        assert [row["code"] for row in incidents] == [
            "LEGACY_ACTIVE_DIAGNOSTICS_ARCHIVED",
            "LEGACY_CLOSED",
        ]
        assert all(row["resolved_at_us"] is not None for row in incidents)
        archive = next(
            row for row in incidents if row["code"] == "LEGACY_ACTIVE_DIAGNOSTICS_ARCHIVED"
        )
        assert json.loads(archive["details_json"])["active_legacy_rows"] == {
            "gaps": 1,
            "incidents": 1,
        }


def test_import_fails_closed_for_active_or_ambiguous_source_and_existing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.sqlite3"
    with _legacy_database(source, 1) as legacy:
        legacy.execute(
            "INSERT INTO recorder_lease VALUES ('recorder', 'run', 'owner', 'x', 'x', 1, 0)"
        )
        legacy.commit()
    with pytest.raises(LegacyImportError, match="active recorder lease"):
        import_legacy_database(source, tmp_path / "target.sqlite3", started_at_us=1)

    with sqlite3.connect(source) as legacy:
        legacy.execute("DELETE FROM recorder_lease")
        legacy.commit()
    Path(f"{source}-wal").write_bytes(b"not-checkpointed")
    with pytest.raises(LegacyImportError, match="WAL"):
        import_legacy_database(source, tmp_path / "target.sqlite3", started_at_us=1)
    Path(f"{source}-wal").unlink()

    Path(f"{source}-journal").write_bytes(b"unfinished-rollback")
    with pytest.raises(LegacyImportError, match="journal"):
        import_legacy_database(source, tmp_path / "target.sqlite3", started_at_us=1)
    Path(f"{source}-journal").unlink()

    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"occupied")
    with pytest.raises(LegacyImportError, match="new target"):
        import_legacy_database(source, target, started_at_us=1)
    assert target.read_bytes() == b"occupied"


def test_import_rejects_an_active_legacy_recorder_generation(tmp_path: Path) -> None:
    source = tmp_path / "legacy-active-generation.sqlite3"
    prefix = MIGRATION_NAMES.index("0016_prospective_recorder_hardening_v1.sql") + 1
    with _legacy_database(source, prefix) as legacy:
        legacy.execute(
            "INSERT INTO recorder_generation_v1("
            "run_id, recorder_generation, owner_id, started_at_utc"
            ") VALUES ('unbound-startup', 1, 'legacy-owner', '2026-01-01T00:00:00+00:00')"
        )
        legacy.commit()

    with pytest.raises(LegacyImportError, match="active recorder generation"):
        import_legacy_database(source, tmp_path / "target.sqlite3", started_at_us=1)


def test_attended_import_archives_quiescent_unclean_generations_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-quiescent-unclean.sqlite3"
    prefix = MIGRATION_NAMES.index("0030_quiet_checkpoint_quote_audit_v0.sql") + 1
    with _legacy_database(source, prefix) as legacy:
        _seed_representative_legacy_rows(legacy)
        legacy.execute(
            "INSERT INTO recorder_generation_v1("
            "run_id, recorder_generation, owner_id, started_at_utc"
            ") VALUES ('legacy-shadow', 7, 'legacy-owner', "
            "'2026-01-02T14:30:00+00:00')"
        )
        legacy.executemany(
            "INSERT INTO recorder_generation_v1("
            "run_id, recorder_generation, owner_id, started_at_utc"
            ") VALUES (?, 1, 'legacy-owner', '2026-01-01T00:00:00+00:00')",
            ((f"unbound-{number:03d}",) for number in range(55)),
        )
        legacy.commit()
    with pytest.raises(LegacyImportError, match="requires a read-only source"):
        import_legacy_database(
            source,
            tmp_path / "writable-target.sqlite3",
            started_at_us=1,
            accept_quiescent_unclean_generations=True,
        )
    source.chmod(0o440)
    source_bytes = source.read_bytes()

    result = import_legacy_database(
        source,
        tmp_path / "target.sqlite3",
        started_at_us=1_800_000_000_000_000,
        accept_quiescent_unclean_generations=True,
    )
    second = import_legacy_database(
        source,
        tmp_path / "second-target.sqlite3",
        started_at_us=1_800_000_000_000_001,
        accept_quiescent_unclean_generations=True,
    )

    assert source.read_bytes() == source_bytes
    assert source.stat().st_mode & 0o777 == 0o440
    assert result.target_digest == second.target_digest
    report = json.loads(result.reconciliation_path.read_text(encoding="utf-8"))
    archive = report["quiescent_unclean_generations"]
    assert archive["disposition"] == "archived_quiescent_unclean"
    assert archive["row_count"] == 56
    assert archive["run_count"] == 56
    assert len(archive["run_summary"]) == 50
    assert archive["run_summary_truncated"] is True
    assert archive["source_rows_modified"] is False
    assert len(archive["row_content_digest"]) == 64
    with sqlite3.connect(result.target_path) as target:
        target.row_factory = sqlite3.Row
        generation = target.execute(
            "SELECT clean_stop, termination_code FROM recorder_generations "
            "WHERE run_id='legacy-shadow'"
        ).fetchone()
        assert tuple(generation) == (0, "MIGRATED_QUIESCENT_UNCLEAN_ARCHIVED")
        incident = target.execute(
            "SELECT resolved_at_us, details_json FROM incidents "
            "WHERE code='LEGACY_QUIESCENT_UNCLEAN_GENERATIONS_ARCHIVED'"
        ).fetchone()
        assert incident["resolved_at_us"] == legacy_import_module._timestamp_us(
            "2026-01-02T14:30:00+00:00",
            label="test",
        )
        details = json.loads(incident["details_json"])
        assert details["row_content_digest"] == archive["row_content_digest"]
        assert details["attended_assertion"] == "accept_quiescent_unclean_generations"


def test_attended_unclean_assertion_never_bypasses_a_lease_or_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "legacy-unsafe.sqlite3"
    prefix = MIGRATION_NAMES.index("0016_prospective_recorder_hardening_v1.sql") + 1
    with _legacy_database(source, prefix) as legacy:
        legacy.execute(
            "INSERT INTO recorder_generation_v1("
            "run_id, recorder_generation, owner_id, started_at_utc"
            ") VALUES ('unbound-startup', 1, 'legacy-owner', "
            "'2026-01-01T00:00:00+00:00')"
        )
        legacy.execute(
            "INSERT INTO recorder_lease VALUES ('recorder', 'run', 'owner', 'x', 'x', 1, 0)"
        )
        legacy.commit()
    source.chmod(0o440)

    with pytest.raises(LegacyImportError, match="active recorder lease"):
        import_legacy_database(
            source,
            tmp_path / "lease-target.sqlite3",
            started_at_us=1,
            accept_quiescent_unclean_generations=True,
        )

    source.chmod(0o640)
    with sqlite3.connect(source) as legacy:
        legacy.execute("DELETE FROM recorder_lease")
        legacy.commit()
    source.chmod(0o440)
    Path(f"{source}-shm").write_bytes(b"ambiguous")
    with pytest.raises(LegacyImportError, match="SHM"):
        import_legacy_database(
            source,
            tmp_path / "sidecar-target.sqlite3",
            started_at_us=1,
            accept_quiescent_unclean_generations=True,
        )


def test_wholly_omitted_tables_are_reconciled_without_loading_payload_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "legacy-omitted.sqlite3"
    with _legacy_database(source, 1) as legacy:
        _seed_representative_legacy_rows(legacy)
    original_rows = legacy_import_module._LegacyReader.rows

    def guarded_rows(
        reader: legacy_import_module._LegacyReader, table: str
    ) -> Iterator[tuple[tuple[object, ...], sqlite3.Row]]:
        if table == "audit_event":
            raise AssertionError("omitted payload table must use primary-key-only iteration")
        yield from original_rows(reader, table)

    monkeypatch.setattr(legacy_import_module._LegacyReader, "rows", guarded_rows)

    result = import_legacy_database(source, tmp_path / "target.sqlite3", started_at_us=1)
    report = json.loads(result.reconciliation_path.read_text(encoding="utf-8"))
    audit = next(item for item in report["tables"] if item["table"] == "audit_event")
    assert audit["source_rows"] == 1
    assert audit["omission_reasons"] == {"legacy_runtime_only": 1}


def test_import_rejects_schema_tampering_and_publishes_atomically(tmp_path: Path) -> None:
    source = tmp_path / "legacy.sqlite3"
    with _legacy_database(source, 1) as legacy:
        legacy.execute("CREATE TABLE injected_secret(value TEXT)")
        legacy.commit()
    target = tmp_path / "target.sqlite3"

    with pytest.raises(LegacyImportError, match="schema digest"):
        import_legacy_database(source, target, started_at_us=1)

    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".migration-reconciliation.json").exists()
    assert not tuple(tmp_path.glob(".stocker-v2-import-*"))


def test_import_rejects_a_tampered_migration_ledger(tmp_path: Path) -> None:
    source = tmp_path / "legacy.sqlite3"
    with _legacy_database(source, 1) as legacy:
        legacy.execute("ALTER TABLE schema_migrations RENAME TO old_schema_migrations")
        legacy.execute(
            "CREATE TABLE schema_migrations ("
            "version TEXT PRIMARY KEY, applied_at_utc TEXT NOT NULL, forged TEXT)"
        )
        legacy.execute(
            "INSERT INTO schema_migrations(version, applied_at_utc) "
            "SELECT version, applied_at_utc FROM old_schema_migrations"
        )
        legacy.execute("DROP TABLE old_schema_migrations")
        legacy.commit()

    with pytest.raises(LegacyImportError, match="ledger has unexpected columns"):
        import_legacy_database(source, tmp_path / "target.sqlite3", started_at_us=1)


def test_import_detects_a_wal_created_during_the_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "legacy.sqlite3"
    with _legacy_database(source, 1):
        pass
    original_hash = legacy_import_module._file_sha256
    calls = 0

    def hash_and_start_writer(path: Path) -> str:
        nonlocal calls
        calls += 1
        value = original_hash(path)
        if calls == 2:
            Path(f"{source}-wal").write_bytes(b"writer-race")
        return value

    monkeypatch.setattr(legacy_import_module, "_file_sha256", hash_and_start_writer)
    target = tmp_path / "target.sqlite3"

    with pytest.raises(LegacyImportError, match="WAL"):
        import_legacy_database(source, target, started_at_us=1)

    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".migration-reconciliation.json").exists()


def test_import_bounds_the_reconciliation_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "legacy.sqlite3"
    with _legacy_database(source, 1):
        pass
    monkeypatch.setattr(legacy_import_module, "MAX_RECONCILIATION_BYTES", 1)
    target = tmp_path / "target.sqlite3"

    with pytest.raises(LegacyImportError, match="report exceeds"):
        import_legacy_database(source, target, started_at_us=1)

    assert not target.exists()
    assert not target.with_suffix(target.suffix + ".migration-reconciliation.json").exists()


def test_legacy_import_cli_emits_verified_manifest(tmp_path: Path) -> None:
    source = tmp_path / "legacy.sqlite3"
    with _legacy_database(source, 1):
        pass
    target = tmp_path / "target.sqlite3"

    result = CliRunner().invoke(
        app,
        [
            "legacy",
            "import",
            "--source",
            str(source),
            "--target",
            str(target),
            "--started-at-us",
            "1800000000000000",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["verification_status"] == "verified"
    assert payload["target_path"] == str(target)


def test_legacy_import_cli_requires_the_explicit_quiescent_unclean_assertion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-unclean.sqlite3"
    prefix = MIGRATION_NAMES.index("0016_prospective_recorder_hardening_v1.sql") + 1
    with _legacy_database(source, prefix) as legacy:
        _seed_representative_legacy_rows(legacy)
        legacy.execute(
            "INSERT INTO recorder_generation_v1("
            "run_id, recorder_generation, owner_id, started_at_utc"
            ") VALUES ('legacy-shadow', 1, 'legacy-owner', "
            "'2026-01-02T14:30:00+00:00')"
        )
        legacy.commit()
    source.chmod(0o440)

    rejected = CliRunner().invoke(
        app,
        [
            "legacy",
            "import",
            "--source",
            str(source),
            "--target",
            str(tmp_path / "rejected.sqlite3"),
            "--started-at-us",
            "1800000000000000",
        ],
    )
    assert rejected.exit_code == 1
    assert "active recorder generation" in rejected.stdout

    target = tmp_path / "accepted.sqlite3"
    accepted = CliRunner().invoke(
        app,
        [
            "legacy",
            "import",
            "--source",
            str(source),
            "--target",
            str(target),
            "--started-at-us",
            "1800000000000000",
            "--accept-quiescent-unclean-generations",
        ],
    )
    assert accepted.exit_code == 0, accepted.output
    assert json.loads(accepted.stdout)["verification_status"] == "verified"
