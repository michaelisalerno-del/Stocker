"""Rescore later EODHD bars and persist provider-transfer evidence for M1C V0."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from stocker_prospective.contract import claims_boundary
from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.frozen_m1c import (
    FreshEpisodeTracker,
    FrozenM1CRuntime,
)
from stocker_prospective.group_o import FrozenGroupOContext
from stocker_prospective.m1c_features import (
    FROZEN_CHECKPOINTS,
    HistoricalActivityBaseline,
    LiveFeatureBar,
    M1CCausalFeatureBuilder,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningTransferBarV1,
    evaluate_opening_transfer_session_v1,
)
from stocker_prospective.quiet_state import QuietEpisodeTracker
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.transfer import (
    IBKRCalibrationCandidate,
    M1CTransferMonitor,
    ProviderM1CObservation,
    TransferBar,
    TransferDecision,
    TransferReport,
    create_ibkr_calibration_candidate,
)

GroupOProvider = Callable[[str, date], FrozenGroupOContext]
MetadataFactory = Callable[[datetime, tuple[datetime, ...]], EvidenceMetadata]


def transfer_report_payload(report: TransferReport) -> dict[str, object]:
    """Return a JSON-safe report while retaining the binding claims boundary."""

    return cast(
        dict[str, object],
        json.loads(
            json.dumps(
                asdict(report),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
                allow_nan=False,
            )
        ),
    )


def _finite_features(payload: Mapping[str, object]) -> dict[str, float]:
    features: dict[str, float] = {}
    for name, value in payload.items():
        if name.startswith("_") or value is None or isinstance(value, bool):
            continue
        try:
            number = float(cast(Any, value))
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            features[name] = number
    return features


def _aware(value: object) -> datetime:
    observed = datetime.fromisoformat(str(value))
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("provider bar timestamp is not timezone-aware")
    return observed.astimezone(UTC)


class SourceTransferCoordinator:
    """Keep IBKR scoring live and evaluate EODHD equivalence only after capture."""

    def __init__(
        self,
        *,
        repository: ProspectiveRepository,
        frozen_repository: FrozenRecorderRepository,
        run_id: str,
        model: FrozenM1CRuntime,
        features: M1CCausalFeatureBuilder,
        activity_baseline: HistoricalActivityBaseline,
        group_o_provider: GroupOProvider,
        metadata_factory: MetadataFactory,
        aggregate_report_path: Path,
        runtime_parity_passed: bool,
        expected_symbols: tuple[str, ...],
        opening_reversal_enabled: bool = False,
    ) -> None:
        if len(expected_symbols) != 20 or len(set(expected_symbols)) != 20:
            raise ValueError("source transfer requires the exact frozen 20-stock universe")
        self.repository = repository
        self.frozen_repository = frozen_repository
        self.run_id = run_id
        self.model = model
        self.features = features
        self.activity_baseline = activity_baseline
        self.group_o_provider = group_o_provider
        self.metadata_factory = metadata_factory
        self.aggregate_report_path = aggregate_report_path
        self.runtime_parity_passed = runtime_parity_passed
        self.expected_symbols = expected_symbols
        self.opening_reversal_enabled = opening_reversal_enabled
        numeric_width = len(model.numeric_features)
        self.monitor = M1CTransferMonitor(
            robust_feature_scales={
                name: float(model.numeric_scales[index])
                for index, name in enumerate(model.numeric_features)
            },
            feature_coefficients={
                name: float(model.coefficients[index])
                for index, name in enumerate(model.numeric_features)
                if index < numeric_width
            },
        )

    def process_session(
        self,
        *,
        session: date,
        observed_at: datetime,
    ) -> TransferReport:
        """Persist both providers, recompute the cumulative decision, and never fit."""

        observed = observed_at.astimezone(UTC)
        metadata = self.metadata_factory(observed, (observed,))
        eodhd_rows = self._eodhd_rows(session)
        ibkr_bars = self._ibkr_bars(session)
        self._persist_ibkr_observations(
            metadata=metadata,
            session=session,
            bars=ibkr_bars,
        )
        self._persist_eodhd_observations(
            metadata=metadata,
            session=session,
            bars=eodhd_rows,
        )
        ibkr, eodhd = self._load_provider_observations()
        ibkr_session = tuple(row for row in ibkr if row.session == session)
        eodhd_session = tuple(row for row in eodhd if row.session == session)
        current_session_report = self.monitor.evaluate(
            ibkr=ibkr_session,
            eodhd=eodhd_session,
            runtime_parity_passed=self.runtime_parity_passed,
        )
        session_valid = self._session_is_valid(
            ibkr=ibkr_session,
            eodhd=eodhd_session,
            report=current_session_report,
        )
        opening_transfer = None
        if self.opening_reversal_enabled:
            ibkr_high = {
                (row.symbol, row.checkpoint)
                for row in ibkr_session
                if row.checkpoint == 6 and row.high_tail_episode
            }
            eodhd_high = {
                (row.symbol, row.checkpoint)
                for row in eodhd_session
                if row.checkpoint == 6 and row.high_tail_episode
            }
            operational_evidence = (
                self.frozen_repository
                .opening_reversal_engineering_operational_evidence_v1(
                    run_id=self.run_id,
                    session=session,
                )
            )
            opening_transfer = evaluate_opening_transfer_session_v1(
                session=session,
                ibkr_bars=self._opening_ibkr_bars(ibkr_bars),
                eodhd_bars=self._opening_eodhd_bars(
                    eodhd_rows.get("VTI", ())
                ),
                checkpoint_6_episode_identity_agreement=(
                    ibkr_high == eodhd_high
                ),
                stock_probability_rank_comparison_available=(
                    current_session_report.probability_metrics.count > 0
                ),
                operational_evidence=operational_evidence,
            )
            self.frozen_repository.record_opening_reversal_transfer_session_v1(
                metadata,
                opening_transfer,
            )
            transfer_decision_receipt = (
                self.frozen_repository.maybe_record_opening_transfer_decision_v1(
                    metadata
                )
            )
        else:
            transfer_decision_receipt = None
        valid_sessions = self._prior_valid_sessions()
        if session_valid:
            valid_sessions.add(session)
        decision_sessions = set(sorted(valid_sessions)[:20])
        aggregate_ibkr = tuple(row for row in ibkr if row.session in decision_sessions)
        aggregate_eodhd = tuple(row for row in eodhd if row.session in decision_sessions)
        report = (
            current_session_report
            if not aggregate_ibkr and not aggregate_eodhd
            else self.monitor.evaluate(
                ibkr=aggregate_ibkr,
                eodhd=aggregate_eodhd,
                runtime_parity_passed=self.runtime_parity_passed,
            )
        )
        payload = transfer_report_payload(report)
        payload["generated_at_utc"] = observed.isoformat()
        payload["recommendation"] = self._recommendation(report.decision)
        payload["strategy_profitability_decision_allowed"] = False
        payload["current_session"] = session.isoformat()
        payload["current_session_valid"] = session_valid
        payload["current_session_quality_decision"] = current_session_report.decision
        payload["engineering_recording_summary"] = self._engineering_summary()
        payload["opening_reversal_transfer_v1"] = (
            None
            if opening_transfer is None
            else opening_transfer.model_dump(mode="json")
        )
        payload["opening_reversal_transfer_decision_receipt_v1"] = (
            None
            if transfer_decision_receipt is None
            else transfer_decision_receipt.model_dump(mode="json")
        )
        payload["claims_boundary"] = claims_boundary()
        self.frozen_repository.record_source_transfer_session(
            metadata,
            session=session,
            valid=session_valid,
            decision=report.decision,
            report=payload,
        )
        valid_ordinal = self._valid_session_ordinal(session) if session_valid else None
        phase = (
            "engineering_transfer"
            if valid_ordinal is not None and valid_ordinal <= 20
            else self.frozen_repository.prospective_phase_for_session(
                run_id=self.run_id,
                session=session,
            )[0]
        )
        self.frozen_repository.record_prospective_session_phase(
            metadata,
            session=session,
            valid=session_valid,
            valid_session_ordinal=valid_ordinal,
            phase=phase,
            source_transfer_decision=report.decision,
        )
        self.aggregate_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.aggregate_report_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if (
            report.valid_session_count >= 20
            and report.decision is TransferDecision.RANKING_SUPPORTED_SCALE_SHIFTED
        ):
            self._freeze_calibration_candidate(
                create_ibkr_calibration_candidate(
                    report=report,
                    ibkr=aggregate_ibkr,
                )
            )
        return report

    @staticmethod
    def _opening_ibkr_bars(
        bars: Mapping[tuple[str, int], Mapping[str, object]],
    ) -> tuple[OpeningTransferBarV1, ...]:
        output: list[OpeningTransferBarV1] = []
        for checkpoint in range(1, 7):
            row = bars.get(("VTI", checkpoint))
            if row is None:
                continue
            output.append(
                OpeningTransferBarV1(
                    ordinal=checkpoint - 1,
                    bar_start_timestamp_utc=_aware(row["bar_start_utc"]),
                    bar_complete_timestamp_utc=_aware(row["bar_end_utc"]),
                    open=float(cast(Any, row["open"])),
                    high=float(cast(Any, row["high"])),
                    low=float(cast(Any, row["low"])),
                    close=float(cast(Any, row["close"])),
                    complete=bool(row.get("finalised", True)),
                )
            )
        return tuple(output)

    @staticmethod
    def _opening_eodhd_bars(
        bars: tuple[Mapping[str, object], ...],
    ) -> tuple[OpeningTransferBarV1, ...]:
        return tuple(
            OpeningTransferBarV1(
                ordinal=ordinal,
                bar_start_timestamp_utc=_aware(row["bar_start_utc"]),
                bar_complete_timestamp_utc=_aware(row["bar_end_utc"]),
                open=float(cast(Any, row["open"])),
                high=float(cast(Any, row["high"])),
                low=float(cast(Any, row["low"])),
                close=float(cast(Any, row["close"])),
                complete=str(row["completeness"]) == "complete",
            )
            for ordinal, row in enumerate(bars[:6])
        )

    def _freeze_calibration_candidate(
        self,
        candidate: IBKRCalibrationCandidate,
    ) -> None:
        """Create V1 once from probabilities only; never update it with later rows."""

        path = self.aggregate_report_path.with_name("M1C_IBKR_CALIBRATION_V1_CANDIDATE.json")
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("candidate_id") != candidate.candidate_id
                or payload.get("source") != "ibkr_probability_distribution_only"
                or payload.get("source_valid_sessions") != list(candidate.source_valid_sessions)
                or payload.get("source_observation_count") != candidate.source_observation_count
                or payload.get("outcome_fields_used") != []
                or payload.get("option_pnl_used") is not False
            ):
                raise ValueError("frozen IBKR calibration candidate is invalid")
            return
        path.write_text(
            json.dumps(asdict(candidate), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _engineering_summary(self) -> dict[str, object]:
        """Summarise recorder engineering quality without judging profitability."""

        with self.repository._connect() as connection:
            state_rows = connection.execute(
                """
                SELECT state, COUNT(*) AS count
                FROM option_episode_allocation_v0
                WHERE run_id = ?
                GROUP BY state
                """,
                (self.run_id,),
            ).fetchall()
            skipped_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM skipped_recording_v0 WHERE run_id = ?
                    """,
                    (self.run_id,),
                ).fetchone()[0]
            )
            resolved_count = int(
                connection.execute(
                    """
                    SELECT (
                        SELECT COUNT(*) FROM episode_option_contract_v0
                        WHERE run_id = ? AND resolution_status = 'recording'
                    ) + (
                        SELECT COUNT(*) FROM quiet_state_option_contract_v0
                        WHERE run_id = ? AND resolution_status = 'recording'
                    )
                    """,
                    (self.run_id, self.run_id),
                ).fetchone()[0]
            )
            contract_count = int(
                connection.execute(
                    """
                    SELECT (
                        SELECT COUNT(*) FROM episode_option_contract_v0
                        WHERE run_id = ?
                    ) + (
                        SELECT COUNT(*) FROM quiet_state_option_contract_v0
                        WHERE run_id = ?
                    )
                    """,
                    (self.run_id, self.run_id),
                ).fetchone()[0]
            )
            shadow_count = int(
                connection.execute(
                    """
                    SELECT (
                        SELECT COUNT(*) FROM shadow_quote_outcome_v0
                        WHERE run_id = ?
                    ) + (
                        SELECT COUNT(*) FROM shadow_structure_outcome_v0
                        WHERE run_id = ?
                    ) + (
                        SELECT COUNT(*) FROM quiet_state_shadow_outcome_v0
                        WHERE run_id = ?
                    )
                    """,
                    (self.run_id, self.run_id, self.run_id),
                ).fetchone()[0]
            )
            complete_shadow_count = int(
                connection.execute(
                    """
                    SELECT (
                        SELECT COUNT(*) FROM shadow_quote_outcome_v0
                        WHERE run_id = ? AND valid = 1
                    ) + (
                        SELECT COUNT(*) FROM shadow_structure_outcome_v0
                        WHERE run_id = ? AND valid = 1
                    ) + (
                        SELECT COUNT(*) FROM quiet_state_shadow_outcome_v0
                        WHERE run_id = ? AND complete_quote_quality = 1
                    )
                    """,
                    (self.run_id, self.run_id, self.run_id),
                ).fetchone()[0]
            )
            latest_capacity = connection.execute(
                """
                SELECT manifest_json FROM ibkr_runtime_capacity_v0
                WHERE run_id = ? ORDER BY observed_at_utc DESC LIMIT 1
                """,
                (self.run_id,),
            ).fetchone()
        states = {str(row["state"]): int(row["count"]) for row in state_rows}
        return {
            "capacity_utilisation": (
                {} if latest_capacity is None else json.loads(str(latest_capacity["manifest_json"]))
            ),
            "queued_episode_transition_count": states.get("EPISODE_QUEUED", 0),
            "degraded_episode_transition_count": states.get("DEGRADED", 0),
            "skipped_recording_count": skipped_count,
            "contract_resolution_attempt_count": contract_count,
            "contract_resolution_success_count": resolved_count,
            "contract_resolution_success_rate": (
                None if contract_count == 0 else resolved_count / contract_count
            ),
            "shadow_engineering_outcome_count": shadow_count,
            "complete_quote_shadow_outcome_count": complete_shadow_count,
            "option_quote_coverage": (
                None if shadow_count == 0 else complete_shadow_count / shadow_count
            ),
            "profitability_decision_allowed": False,
        }

    @staticmethod
    def _recommendation(decision: TransferDecision) -> str:
        return {
            TransferDecision.SUPPORTED_WITHOUT_RECALIBRATION: "continue_v0",
            TransferDecision.RANKING_SUPPORTED_SCALE_SHIFTED: (
                "create_distribution_only_ibkr_threshold_calibration_candidate"
            ),
            TransferDecision.MIXED_STOCK_OR_CHECKPOINT_FAILURES: "repair_pipeline",
            TransferDecision.NOT_SUPPORTED: "stop",
            TransferDecision.BLOCKED_INSUFFICIENT_VALID_SESSIONS: "continue_v0",
            TransferDecision.BLOCKED_BAR_SEMANTICS_FAILURE: "repair_pipeline",
            TransferDecision.BLOCKED_M1C_RUNTIME_PARITY_FAILURE: "stop",
        }[decision]

    def _eodhd_rows(self, session: date) -> dict[str, tuple[dict[str, object], ...]]:
        with self.repository._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM source_bar_observation
                WHERE run_id = ? AND provider = 'eodhd' AND session_date = ?
                ORDER BY symbol, bar_start_utc, id
                """,
                (self.run_id, session.isoformat()),
            ).fetchall()
        by_symbol: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_symbol.setdefault(str(row["symbol"]), []).append(dict(row))
        return {symbol: tuple(items) for symbol, items in sorted(by_symbol.items())}

    def _ibkr_bars(self, session: date) -> dict[tuple[str, int], dict[str, object]]:
        with self.repository._connect() as connection:
            paths = connection.execute(
                """
                SELECT file_path FROM raw_partition_manifest_v0
                WHERE run_id = ? AND data_source = 'ibkr' AND session_date = ?
                  AND event_type = 'five_minute_bar_event' AND complete = 1
                ORDER BY file_path
                """,
                (self.run_id, session.isoformat()),
            ).fetchall()
        frames: list[pd.DataFrame] = []
        for row in paths:
            path = Path(str(row["file_path"]))
            if not path.is_absolute():
                path = self.repository.database_path.parent / path
            if path.is_file():
                frames.append(pd.read_parquet(path))
        if not frames:
            return {}
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.loc[
            frame["session"].astype(str).eq(session.isoformat()) & frame["finalised"].astype(bool)
        ].copy()
        frame = frame.sort_values(
            ["symbol", "checkpoint", "received_timestamp_utc", "event_id"],
            kind="mergesort",
        ).drop_duplicates(["symbol", "checkpoint"], keep="last")
        output: dict[tuple[str, int], dict[str, object]] = {}
        for raw_row in frame.itertuples(index=False):
            row = cast(Any, raw_row)
            output[(str(row.symbol), int(row.checkpoint))] = cast(
                dict[str, object],
                row._asdict(),
            )
        return output

    def _persist_ibkr_observations(
        self,
        *,
        metadata: EvidenceMetadata,
        session: date,
        bars: Mapping[tuple[str, int], Mapping[str, object]],
    ) -> None:
        with self.repository._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*,
                       EXISTS(
                           SELECT 1 FROM quiet_state_observation_v0 q
                           WHERE q.run_id = c.run_id AND q.symbol = c.symbol
                             AND q.session_date = c.session_date
                             AND q.trigger_checkpoint = c.checkpoint
                             AND q.observation_kind = 'quiet_bottom_10'
                       ) AS quiet_episode,
                       EXISTS(
                           SELECT 1 FROM m1c_episode_v0 e
                           WHERE e.run_id = c.run_id AND e.symbol = c.symbol
                             AND e.session_date = c.session_date
                             AND e.trigger_checkpoint = c.checkpoint
                       ) AS high_tail_episode
                FROM m1c_checkpoint_v0 c
                WHERE c.run_id = ? AND c.session_date = ?
                ORDER BY c.symbol, c.checkpoint
                """,
                (self.run_id, session.isoformat()),
            ).fetchall()
        for row in rows:
            key = (str(row["symbol"]), int(row["checkpoint"]))
            source = bars.get(key)
            if source is None:
                continue
            bar = {
                "identity": str(
                    row["bar_identity"]
                    or (
                        f"IBKR|{key[0]}|{session.isoformat()}|"
                        f"{source['bar_start_utc']}|{source['bar_end_utc']}"
                    )
                ),
                "start_utc": _aware(source["bar_start_utc"]).isoformat(),
                "end_utc": _aware(source["bar_end_utc"]).isoformat(),
                "open": float(cast(Any, source["open"])),
                "high": float(cast(Any, source["high"])),
                "low": float(cast(Any, source["low"])),
                "close": float(cast(Any, source["close"])),
            }
            self.frozen_repository.record_provider_m1c_observation(
                metadata,
                provider="ibkr",
                symbol=key[0],
                session=session,
                checkpoint=key[1],
                bar=bar,
                feature_values=_finite_features(
                    cast(
                        Mapping[str, object],
                        json.loads(str(row["feature_values_json"])),
                    )
                ),
                probability=float(row["probability"]),
                quiet_episode=bool(row["quiet_episode"]),
                high_tail_episode=bool(row["high_tail_episode"]),
                data_quality_status=("valid" if bool(row["eligible"]) else "invalid"),
                model_hash=str(row["model_hash"]),
            )

    def _persist_eodhd_observations(
        self,
        *,
        metadata: EvidenceMetadata,
        session: date,
        bars: Mapping[str, tuple[Mapping[str, object], ...]],
    ) -> None:
        quiet_tracker = QuietEpisodeTracker()
        high_tracker = FreshEpisodeTracker()
        for symbol in self.expected_symbols:
            rows = bars.get(symbol, ())
            if len(rows) != 78:
                continue
            feature_bars: list[LiveFeatureBar] = []
            valid = True
            for ordinal, row in enumerate(rows):
                if str(row["completeness"]) != "complete":
                    valid = False
                    break
                volume = float(cast(Any, row["activity_value"]))
                activity = self.activity_baseline.relative_activity(
                    symbol=symbol,
                    session=session,
                    bar_ordinal=ordinal,
                    volume=volume,
                )
                if activity is None:
                    valid = False
                    break
                feature_bars.append(
                    LiveFeatureBar(
                        symbol=symbol,
                        session=session,
                        bar_ordinal=ordinal,
                        bar_start_timestamp=_aware(row["bar_start_utc"]),
                        bar_complete_timestamp=_aware(row["bar_end_utc"]),
                        open=float(cast(Any, row["open"])),
                        high=float(cast(Any, row["high"])),
                        low=float(cast(Any, row["low"])),
                        close=float(cast(Any, row["close"])),
                        volume=volume,
                        historical_relative_activity=activity,
                        finalised=True,
                        source="eodhd_parallel_transfer",
                    )
                )
            if not valid:
                continue
            context = self.group_o_provider(symbol, session)
            for checkpoint in FROZEN_CHECKPOINTS:
                causal = self.features.build(
                    symbol=symbol,
                    checkpoint=checkpoint,
                    completed_bars=tuple(feature_bars[:checkpoint]),
                )
                score = self.model.score(
                    symbol=symbol,
                    checkpoint=checkpoint,
                    group_o_context=context.features,
                    causal_group_i=causal.scaled_features,
                )
                trigger = feature_bars[checkpoint - 1]
                quiet = quiet_tracker.evaluate(
                    symbol=symbol,
                    session=session,
                    checkpoint=checkpoint,
                    trigger_bar_end=trigger.bar_complete_timestamp,
                    probability=score.probability,
                    eligible=True,
                )
                high = high_tracker.evaluate(
                    symbol=symbol,
                    session=session,
                    checkpoint=checkpoint,
                    trigger_bar_end=trigger.bar_complete_timestamp,
                    probability=score.probability,
                    eligible=True,
                )
                self.frozen_repository.record_provider_m1c_observation(
                    metadata,
                    provider="eodhd",
                    symbol=symbol,
                    session=session,
                    checkpoint=checkpoint,
                    bar={
                        "identity": (
                            f"EODHD|{symbol}|{session.isoformat()}|"
                            f"{trigger.bar_start_timestamp.isoformat()}|"
                            f"{trigger.bar_complete_timestamp.isoformat()}"
                        ),
                        "start_utc": trigger.bar_start_timestamp.isoformat(),
                        "end_utc": trigger.bar_complete_timestamp.isoformat(),
                        "open": trigger.open,
                        "high": trigger.high,
                        "low": trigger.low,
                        "close": trigger.close,
                    },
                    feature_values=_finite_features(
                        dict(zip(score.feature_order, score.feature_values, strict=True))
                    ),
                    probability=score.probability,
                    quiet_episode=quiet.fresh_episode,
                    high_tail_episode=high.fresh_episode,
                    data_quality_status="valid",
                    model_hash=score.model_hash,
                )

    def _load_provider_observations(
        self,
    ) -> tuple[
        tuple[ProviderM1CObservation, ...],
        tuple[ProviderM1CObservation, ...],
    ]:
        with self.repository._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM provider_m1c_observation_v0
                WHERE run_id = ?
                ORDER BY session_date, symbol, checkpoint, provider
                """,
                (self.run_id,),
            ).fetchall()
        providers: dict[str, list[ProviderM1CObservation]] = {
            "ibkr": [],
            "eodhd": [],
        }
        for row in rows:
            provider = str(row["provider"])
            providers[provider].append(
                ProviderM1CObservation(
                    provider=cast(Any, provider),
                    symbol=str(row["symbol"]),
                    session=date.fromisoformat(str(row["session_date"])),
                    checkpoint=int(row["checkpoint"]),
                    bar=TransferBar(
                        identity=str(row["bar_identity"]),
                        start_utc=_aware(row["bar_start_utc"]),
                        end_utc=_aware(row["bar_end_utc"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        complete=str(row["data_quality_status"]) == "valid",
                    ),
                    features=_finite_features(
                        cast(
                            Mapping[str, object],
                            json.loads(str(row["feature_values_json"])),
                        )
                    ),
                    probability=float(row["probability"]),
                    quiet_episode=bool(row["quiet_episode"]),
                    high_tail_episode=bool(row["high_tail_episode"]),
                )
            )
        return tuple(providers["ibkr"]), tuple(providers["eodhd"])

    def _session_is_valid(
        self,
        *,
        ibkr: tuple[ProviderM1CObservation, ...],
        eodhd: tuple[ProviderM1CObservation, ...],
        report: TransferReport,
    ) -> bool:
        expected = len(self.expected_symbols) * len(FROZEN_CHECKPOINTS)
        if len(ibkr) != expected or len(eodhd) != expected:
            return False
        return report.bar_semantics_passed and report.runtime_parity_passed

    def _prior_valid_sessions(self) -> set[date]:
        with self.repository._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_date FROM source_transfer_session_v0
                WHERE run_id = ? AND valid = 1
                ORDER BY session_date
                """,
                (self.run_id,),
            ).fetchall()
        return {date.fromisoformat(str(row["session_date"])) for row in rows}

    def _valid_session_ordinal(self, session: date) -> int:
        with self.repository._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM source_transfer_session_v0
                WHERE run_id = ? AND valid = 1 AND session_date <= ?
                """,
                (self.run_id, session.isoformat()),
            ).fetchone()
        assert isinstance(row, sqlite3.Row)
        return int(row["count"])


__all__ = ["SourceTransferCoordinator", "transfer_report_payload"]
