"""Daily download packages for the budget-aware IBKR research recorder."""

from __future__ import annotations

import csv
import json
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

from stocker_prospective.contract import (
    ORIGINAL_LOW_MOVEMENT_DECISION,
    claims_boundary,
)
from stocker_prospective.transfer import (
    TransferReport,
    classify_cross_vendor_validation_status,
)

DAILY_REPORT_FILENAMES = (
    "session_summary.json",
    "ibkr_bar_quality.csv",
    "m1c_ibkr_predictions.csv",
    "eodhd_ibkr_bar_comparison.csv",
    "eodhd_ibkr_feature_comparison.csv",
    "eodhd_ibkr_probability_comparison.csv",
    "tail_membership_comparison.csv",
    "episode_comparison.csv",
    "market_data_budget_report.json",
    "subscription_lifecycle.csv",
    "option_episode_quality.csv",
    "shadow_outcomes.csv",
    "skipped_recordings.csv",
    "report.md",
)


@dataclass(frozen=True)
class DailyReportPackage:
    session: date
    generated_at_utc: datetime
    report_directory: Path
    archive_path: Path
    files: tuple[Path, ...]


def _json_safe(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        )
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["status"]
        rows = [{"status": "no_rows"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _rows(
    connection: sqlite3.Connection,
    statement: str,
    parameters: tuple[object, ...],
) -> list[dict[str, object]]:
    return [dict(row) for row in connection.execute(statement, parameters).fetchall()]


class BudgetAwareDailyReportWriter:
    """Create append-only report generations; never replace an earlier package."""

    def __init__(
        self,
        *,
        database_path: Path,
        run_id: str,
        report_root: Path,
    ) -> None:
        self.database_path = database_path
        self.run_id = run_id
        self.report_root = report_root

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def write(
        self,
        *,
        session: date,
        generated_at: datetime,
        capacity_manifest: dict[str, object],
        budget_snapshot: dict[str, object],
        transfer_report: TransferReport | None = None,
    ) -> DailyReportPackage:
        generated = generated_at.astimezone(UTC)
        generation = generated.strftime("%Y%m%dT%H%M%S.%fZ")
        session_root = self.report_root / session.isoformat()
        destination = session_root / generation
        destination.mkdir(parents=True, exist_ok=False)
        claims = claims_boundary()
        with self._connect() as connection:
            checkpoints = _rows(
                connection,
                """
                SELECT c.*, q.bottom_5, q.bottom_10, q.bottom_20, q.high_tail,
                       q.data_quality_status, q.data_quality_flags_json
                FROM m1c_checkpoint_v0 c
                LEFT JOIN quiet_state_checkpoint_v0 q ON q.checkpoint_id = c.id
                WHERE c.run_id = ? AND c.session_date = ?
                ORDER BY c.symbol, c.checkpoint
                """,
                (self.run_id, session.isoformat()),
            )
            lifecycle = _rows(
                connection,
                """
                SELECT * FROM subscription_lifecycle_event_v0
                WHERE run_id = ? AND substr(occurred_at_utc, 1, 10) = ?
                ORDER BY occurred_at_utc, id
                """,
                (self.run_id, session.isoformat()),
            )
            option_quality = _rows(
                connection,
                """
                SELECT * FROM option_episode_allocation_v0
                WHERE run_id = ? AND substr(updated_at_utc, 1, 10) = ?
                ORDER BY updated_at_utc, id
                """,
                (self.run_id, session.isoformat()),
            )
            quiet_shadow = _rows(
                connection,
                """
                SELECT s.*, q.symbol, q.session_date, q.observation_kind
                FROM quiet_state_shadow_outcome_v0 s
                JOIN quiet_state_observation_v0 q
                  ON q.observation_id = s.observation_id
                WHERE s.run_id = ? AND q.session_date = ?
                ORDER BY s.observation_id, s.dte_bucket, s.horizon_minutes, s.id
                """,
                (self.run_id, session.isoformat()),
            )
            high_shadow = _rows(
                connection,
                """
                SELECT s.*, e.symbol, e.session_date, 'high_tail' AS observation_kind
                FROM shadow_quote_outcome_v0 s
                JOIN m1c_episode_v0 e ON e.episode_id = s.episode_id
                WHERE s.run_id = ? AND e.session_date = ?
                ORDER BY s.episode_id, s.dte_bucket, s.horizon_minutes, s.id
                """,
                (self.run_id, session.isoformat()),
            )
            skipped = _rows(
                connection,
                """
                SELECT * FROM skipped_recording_v0
                WHERE run_id = ? AND session_date = ?
                ORDER BY occurred_at_utc, id
                """,
                (self.run_id, session.isoformat()),
            )
            phase_row = connection.execute(
                """
                SELECT phase
                FROM prospective_session_phase_v0
                WHERE run_id = ? AND session_date = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (self.run_id, session.isoformat()),
            ).fetchone()
        prediction_rows = [
            {
                "symbol": row["symbol"],
                "session": row["session_date"],
                "checkpoint": row["checkpoint"],
                "ibkr_bar_identity": row["bar_identity"],
                "feature_vector_json": row["feature_values_json"],
                "m1c_probability": row["probability"],
                "bottom_5": row["bottom_5"],
                "bottom_10": row["bottom_10"],
                "bottom_20": row["bottom_20"],
                "high_tail": row["high_tail"],
                "data_quality_status": row["data_quality_status"],
                "model_hash": row["model_hash"],
                "configuration_hash": row["configuration_hash"],
                "claims_json": row["claims_json"],
            }
            for row in checkpoints
        ]
        bar_quality = [
            {
                "symbol": row["symbol"],
                "session": row["session_date"],
                "checkpoint": row["checkpoint"],
                "bar_identity": row["bar_identity"],
                "bar_start_utc": row["bar_start_utc"],
                "bar_end_utc": row["bar_end_utc"],
                "feature_freshness": row["feature_freshness"],
                "eligible": row["eligible"],
                "data_quality_status": row["data_quality_status"],
                "rejection_reasons_json": row["rejection_reasons_json"],
            }
            for row in checkpoints
        ]
        bar_comparison: list[dict[str, object]] = []
        feature_comparison: list[dict[str, object]] = []
        probability_comparison: list[dict[str, object]] = []
        tail_comparison: list[dict[str, object]] = []
        episode_comparison: list[dict[str, object]] = []
        transfer_decision = "cross_vendor_validation_not_configured"
        valid_sessions = 0
        if transfer_report is not None:
            transfer_decision = transfer_report.decision
            valid_sessions = transfer_report.valid_session_count
            semantics = cast(
                dict[str, object],
                _json_safe(asdict(transfer_report.bar_semantics_metrics)),
            )
            bar_comparison = [
                {
                    **cast(dict[str, object], _json_safe(asdict(row))),
                    "bar_semantics_summary": semantics,
                }
                for row in transfer_report.bar_comparisons
            ]
            feature_comparison = [
                cast(dict[str, object], _json_safe(asdict(row)))
                for row in transfer_report.feature_comparisons
            ]
            probability_comparison = [
                cast(dict[str, object], _json_safe(asdict(row)))
                for row in transfer_report.probability_comparisons
            ]
            tail_comparison = [
                {
                    **asdict(transfer_report.tail_metrics),
                    "session": session.isoformat(),
                }
            ]
            episode_comparison = [
                {
                    **asdict(transfer_report.episode_metrics),
                    "session": session.isoformat(),
                }
            ]
        cohort_phase = "option_development" if phase_row is None else str(phase_row["phase"])
        cross_vendor_status = classify_cross_vendor_validation_status(
            enabled=transfer_report is not None,
            credential_configured=transfer_report is not None,
            valid_session_count=valid_sessions,
            decision=transfer_decision,
        )
        scientific_option_evidence = any(
            bool(row.get("scientific_option_evidence")) for row in option_quality
        )
        summary = {
            "session": session.isoformat(),
            "generated_at_utc": generated.isoformat(),
            "cohort_phase": cohort_phase,
            "scientific_option_evidence": scientific_option_evidence,
            "scientific_option_evidence_allowed": cohort_phase != "engineering_transfer",
            "valid_transfer_sessions": valid_sessions,
            "source_transfer_decision": transfer_decision,
            "cross_vendor_validation_status": cross_vendor_status,
            "cross_vendor_validation_diagnostic_only": True,
            "market_data_source": "ibkr",
            "historical_research_source": "eodhd",
            "m1c_prediction_count": len(prediction_rows),
            "queued_episode_transition_count": sum(
                row.get("state") == "EPISODE_QUEUED" for row in option_quality
            ),
            "degraded_episode_transition_count": sum(
                row.get("state") == "DEGRADED" for row in option_quality
            ),
            "skipped_recording_count": len(skipped),
            "historical_decision": ORIGINAL_LOW_MOVEMENT_DECISION,
            "historical_validation_gate_passed": False,
            "exact_vendor_bar_equality_required": False,
            "strategy_profitability_decision_allowed": False,
            "claims_boundary": claims,
        }
        budget_report = {
            "session": session.isoformat(),
            "generated_at_utc": generated.isoformat(),
            "runtime_capacity": capacity_manifest,
            "current_budget": budget_snapshot,
            "fatal_state": "critical_budget_unavailable",
            "optional_exhaustion_is_fatal": False,
            "claims_boundary": claims,
        }
        _write_json(destination / "session_summary.json", summary)
        _write_csv(destination / "ibkr_bar_quality.csv", bar_quality)
        _write_csv(destination / "m1c_ibkr_predictions.csv", prediction_rows)
        _write_csv(destination / "eodhd_ibkr_bar_comparison.csv", bar_comparison)
        _write_csv(destination / "eodhd_ibkr_feature_comparison.csv", feature_comparison)
        _write_csv(
            destination / "eodhd_ibkr_probability_comparison.csv",
            probability_comparison,
        )
        _write_csv(destination / "tail_membership_comparison.csv", tail_comparison)
        _write_csv(destination / "episode_comparison.csv", episode_comparison)
        _write_json(destination / "market_data_budget_report.json", budget_report)
        _write_csv(destination / "subscription_lifecycle.csv", lifecycle)
        _write_csv(destination / "option_episode_quality.csv", option_quality)
        _write_csv(
            destination / "shadow_outcomes.csv",
            [*quiet_shadow, *high_shadow],
        )
        _write_csv(destination / "skipped_recordings.csv", skipped)
        (destination / "report.md").write_text(
            self._markdown(summary=summary, budget=budget_report),
            encoding="utf-8",
        )
        files = tuple(destination / name for name in DAILY_REPORT_FILENAMES)
        if any(not path.is_file() for path in files):
            raise RuntimeError("daily report package is incomplete")
        archive = session_root / f"chatgpt-report-package-{generation}.zip"
        with zipfile.ZipFile(
            archive,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as bundle:
            for path in files:
                bundle.write(path, arcname=path.name)
        _write_json(
            session_root / f"package-{generation}.json",
            {
                "session": session.isoformat(),
                "generated_at_utc": generated.isoformat(),
                "archive": archive.name,
                "report_directory": destination.name,
                "files": list(DAILY_REPORT_FILENAMES),
                "claims_boundary": claims,
            },
        )
        return DailyReportPackage(
            session=session,
            generated_at_utc=generated,
            report_directory=destination,
            archive_path=archive,
            files=files,
        )

    @staticmethod
    def _markdown(
        *,
        summary: dict[str, object],
        budget: dict[str, object],
    ) -> str:
        current = budget["current_budget"]
        return "\n".join(
            (
                "# IBKR Budget-Aware M1C Session Report",
                "",
                "**RECORD ONLY — NO ORDERS**",
                "",
                f"- Session: {summary['session']}",
                f"- Cohort phase: {summary['cohort_phase']}",
                "- Cross-vendor diagnostic status (non-blocking): "
                f"{summary['cross_vendor_validation_status']}",
                "- Cross-vendor diagnostic decision (legacy vocabulary; non-blocking): "
                f"{summary['source_transfer_decision']}",
                f"- Frozen M1C predictions: {summary['m1c_prediction_count']}",
                f"- Queued episode transitions: {summary['queued_episode_transition_count']}",
                f"- Degraded episode transitions: {summary['degraded_episode_transition_count']}",
                f"- Historical decision: `{ORIGINAL_LOW_MOVEMENT_DECISION}`",
                "- Exact vendor bar equality required: false",
                "- Prospective IBKR scientific option evidence recorded: "
                f"{str(summary['scientific_option_evidence']).lower()}",
                "- Strategy profitability decision allowed: false",
                "",
                "## Capacity",
                "",
                f"```json\n{json.dumps(current, sort_keys=True, indent=2)}\n```",
                "",
            )
        )


__all__ = [
    "BudgetAwareDailyReportWriter",
    "DAILY_REPORT_FILENAMES",
    "DailyReportPackage",
]
