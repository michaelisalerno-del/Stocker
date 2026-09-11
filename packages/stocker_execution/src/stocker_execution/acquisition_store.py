"""Normalized scanner/oracle audit persistence. No strategy-state writes."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from stocker_core.candidate_selection import CandidateIdentity, CandidateRank
from stocker_core.methods import content_hash


def encoded(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        allow_nan=False,
        default=lambda v: sorted(v) if isinstance(v, (set, frozenset)) else v.isoformat(),
    )


class AcquisitionStore:
    def __init__(self, path: Path, *, initialize: bool = True):
        self.path = path
        if initialize:
            with self.connect() as db:
                db.executescript("""
                CREATE TABLE IF NOT EXISTS acquisition_sessions (
                  run_id TEXT, session TEXT, recipe_hash TEXT, metadata TEXT,
                  state TEXT, reason TEXT DEFAULT '', sealed INTEGER DEFAULT 0,
                  audit_state TEXT DEFAULT 'PENDING', oracle_metrics TEXT DEFAULT '{}',
                  PRIMARY KEY(run_id,session));
                CREATE TABLE IF NOT EXISTS acquisition_capabilities (
                  digest TEXT PRIMARY KEY, document TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS acquisition_broad (
                  run_id TEXT, session TEXT, source_index INTEGER, reference TEXT,
                  identity TEXT, eligibility TEXT DEFAULT 'UNRESOLVED',
                  bars TEXT, five_minute_bar TEXT, error TEXT DEFAULT '',
                  PRIMARY KEY(run_id,session,source_index));
                CREATE TABLE IF NOT EXISTS acquisition_components (
                  run_id TEXT, session TEXT, sweep INTEGER, component TEXT,
                  request TEXT, status TEXT, audit TEXT DEFAULT '{}',
                  PRIMARY KEY(run_id,session,sweep,component));
                CREATE TABLE IF NOT EXISTS acquisition_hits (
                  run_id TEXT, session TEXT, sweep INTEGER, component TEXT,
                  hit_index INTEGER, con_id INTEGER, scanner_rank INTEGER,
                  observed_at TEXT, payload TEXT,
                  PRIMARY KEY(run_id,session,sweep,component,hit_index));
                CREATE INDEX IF NOT EXISTS acquisition_hits_identity
                  ON acquisition_hits(run_id,session,con_id,component);
                CREATE TABLE IF NOT EXISTS acquisition_pool (
                  run_id TEXT, session TEXT, con_id INTEGER, identity TEXT,
                  source_index INTEGER, first_sweep INTEGER, first_seen TEXT,
                  best_scanner_rank INTEGER, selected INTEGER DEFAULT 1,
                  PRIMARY KEY(run_id,session,con_id));
                CREATE TABLE IF NOT EXISTS acquisition_history_requests (
                  run_id TEXT, session TEXT, phase TEXT, stage INTEGER, con_id INTEGER,
                  payload TEXT, PRIMARY KEY(run_id,session,phase,stage,con_id));
                CREATE TABLE IF NOT EXISTS acquisition_oracle_ranks (
                  run_id TEXT, session TEXT, stage INTEGER, con_id INTEGER,
                  score_name TEXT, value REAL, rank INTEGER, selected INTEGER,
                  missing_reason TEXT, PRIMARY KEY(run_id,session,stage,con_id));
                """)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def session(self, run_id: str, session: date) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with self.connect() as db:
            if not db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='acquisition_sessions'"
            ).fetchone():
                return None
            row = db.execute(
                "SELECT run_id,session,recipe_hash,metadata,state,reason,sealed,audit_state,"
                "json_object('label','AUDIT_ONLY','recipes',json_extract(oracle_metrics,'$.recipes'),"
                "'range_rank_buckets',json_extract(oracle_metrics,'$.range_rank_buckets'),"
                "'transport_parity',json_extract(oracle_metrics,'$.transport_parity')) AS"
                " oracle_metrics "
                "FROM acquisition_sessions WHERE run_id=? AND session=?",
                (run_id, str(session)),
            ).fetchone()
        return dict(row) if row else None

    def begin(
        self, run_id: str, session: date, metadata: dict[str, Any], references: Sequence[Any]
    ) -> None:
        digest = content_hash(metadata["recipe"])
        with self.connect() as db:
            collision = db.execute(
                "SELECT 1 FROM acquisition_sessions WHERE "
                "json_extract(metadata,'$.recipe.recipe_id')=? AND recipe_hash!=? LIMIT 1",
                (metadata["recipe"]["recipe_id"], digest),
            ).fetchone()
            if collision:
                raise ValueError("Changed acquisition recipes require a new recipe ID")
            existing = db.execute(
                "SELECT recipe_hash,metadata FROM acquisition_sessions "
                "WHERE run_id=? AND session=?",
                (run_id, str(session)),
            ).fetchone()
            if existing:
                if existing[0] != digest or json.loads(existing[1]) != metadata:
                    raise ValueError("Acquisition session market/specification is immutable")
                return
            db.execute(
                "INSERT INTO acquisition_sessions(run_id,session,recipe_hash,metadata,state) "
                "VALUES (?,?,?,?,'SCANNER_ACQUISITION_PENDING')",
                (run_id, str(session), digest, encoded(metadata)),
            )
            db.executemany(
                "INSERT INTO acquisition_broad(run_id,session,source_index,reference) "
                "VALUES (?,?,?,?)",
                ((run_id, str(session), i, r.model_dump_json()) for i, r in enumerate(references)),
            )

    def save_capabilities(self, document: dict[str, Any]) -> str:
        serialized = encoded(document)
        import hashlib

        digest = hashlib.sha256(serialized.encode()).hexdigest()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO acquisition_capabilities VALUES (?,?)", (digest, serialized)
            )
        return digest

    def plan(self, run_id: str, session: date, sweep: int, component: Any) -> dict[str, Any]:
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO acquisition_components VALUES (?,?,?,?,?,'PENDING','{}')",
                (run_id, str(session), sweep, component.component_id, encoded(asdict(component))),
            )
            row = db.execute(
                "SELECT * FROM acquisition_components WHERE run_id=? AND session=? AND "
                "sweep=? AND component=?",
                (run_id, str(session), sweep, component.component_id),
            ).fetchone()
        assert row is not None
        if row["request"] != encoded(asdict(component)):
            raise ValueError("Resolved acquisition component changed during the saved session")
        return dict(row)

    def component(
        self,
        run_id: str,
        session: date,
        sweep: int,
        component: str,
        status: str,
        audit: dict[str, Any],
        rows: Sequence[Any] = (),
    ) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE acquisition_components SET status=?,audit=? "
                "WHERE run_id=? AND session=? AND sweep=? AND component=?",
                (status, encoded(audit), run_id, str(session), sweep, component),
            )
            db.executemany(
                "INSERT INTO acquisition_hits VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    (
                        run_id,
                        str(session),
                        sweep,
                        component,
                        i,
                        r.con_id,
                        r.raw_rank,
                        audit["received_at"],
                        encoded(asdict(r)),
                    )
                    for i, r in enumerate(rows)
                ),
            )

    def add_pool(
        self,
        run_id: str,
        session: date,
        identity: CandidateIdentity,
        source_index: int,
        sweep: int,
        observed: datetime,
        scanner_rank: int,
    ) -> None:
        with self.connect() as db:
            sealed = db.execute(
                "SELECT sealed FROM acquisition_sessions WHERE run_id=? AND session=?",
                (run_id, str(session)),
            ).fetchone()
            if sealed is None or sealed[0]:
                raise ValueError("Acquisition union is sealed")
            db.execute(
                "INSERT INTO acquisition_pool VALUES (?,?,?,?,?,?,?,?,1) "
                "ON CONFLICT(run_id,session,con_id) DO UPDATE SET "
                "best_scanner_rank=MIN(best_scanner_rank,excluded.best_scanner_rank), "
                "first_sweep=MIN(first_sweep,excluded.first_sweep), "
                "first_seen=MIN(first_seen,excluded.first_seen)",
                (
                    run_id,
                    str(session),
                    identity.con_id,
                    encoded(asdict(identity)),
                    source_index,
                    sweep,
                    observed.isoformat(),
                    scanner_rank,
                ),
            )
            db.execute(
                "UPDATE acquisition_broad SET identity=?,eligibility='ELIGIBLE' "
                "WHERE run_id=? AND session=? AND source_index=?",
                (encoded(asdict(identity)), run_id, str(session), source_index),
            )

    def seal(
        self,
        run_id: str,
        session: date,
        *,
        capacity: int | None,
        allow_partial: bool,
        reason: str = "",
    ) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            sealed = db.execute(
                "SELECT sealed FROM acquisition_sessions WHERE run_id=? AND session=?",
                (run_id, str(session)),
            ).fetchone()
            if sealed and sealed[0]:
                return
            failures = db.execute(
                "SELECT COUNT(*) FROM acquisition_components WHERE run_id=? "
                "AND session=? AND status!='COMPLETE'",
                (run_id, str(session)),
            ).fetchone()[0]
            count = db.execute(
                "SELECT COUNT(*) FROM acquisition_pool WHERE run_id=? AND session=?",
                (run_id, str(session)),
            ).fetchone()[0]
            state = (
                "SCANNER_ACQUISITION_FAILED"
                if reason or not count
                else "SCANNER_ACQUISITION_PARTIAL"
                if failures
                else "SCANNER_ACQUISITION_READY"
            )
            if failures and not allow_partial:
                reason = reason or "SCANNER_COMPONENT_INCOMPLETE"
            if capacity is not None and count > capacity:
                db.execute(
                    "UPDATE acquisition_pool SET selected=0 WHERE run_id=? AND session=? "
                    "AND con_id NOT IN "
                    "(SELECT con_id FROM acquisition_pool WHERE run_id=? AND session=? "
                    "ORDER BY best_scanner_rank,first_seen,con_id LIMIT ?)",
                    (run_id, str(session), run_id, str(session), capacity),
                )
                state = "SCANNER_ACQUISITION_PARTIAL"
                reason = reason or "ACQUISITION_POOL_CAP_APPLIED"
            db.execute(
                "UPDATE acquisition_sessions SET sealed=1,state=?,reason=? WHERE run_id=?"
                " AND session=?",
                (state, reason, run_id, str(session)),
            )

    def pool(self, run_id: str, session: date) -> tuple[CandidateIdentity, ...]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT identity FROM acquisition_pool WHERE run_id=? AND session=? AND selected=1",
                (run_id, str(session)),
            ).fetchall()
        return tuple(CandidateIdentity(**json.loads(r[0])) for r in rows)

    def history_request(
        self,
        run_id: str,
        session: date,
        phase: str,
        stage: int,
        con_id: int,
        payload: dict[str, Any],
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO acquisition_history_requests VALUES (?,?,?,?,?,?)",
                (run_id, str(session), phase, stage, con_id, encoded(payload)),
            )

    def summary(self, run_id: str, session: date) -> dict[str, Any] | None:
        row = self.session(run_id, session)
        if row is None:
            return None
        metadata = json.loads(row["metadata"])
        key = (run_id, str(session))
        with self.connect() as db:
            broad = db.execute(
                "SELECT COUNT(*), SUM(bars IS NOT NULL OR eligibility='INELIGIBLE') "
                "FROM acquisition_broad WHERE run_id=? AND session=?",
                key,
            ).fetchone()
            hits = db.execute(
                "SELECT COUNT(*),COUNT(DISTINCT con_id) FROM acquisition_hits WHERE "
                "run_id=? AND session=?",
                key,
            ).fetchone()
            pool = db.execute(
                "SELECT COUNT(*),SUM(selected) FROM acquisition_pool WHERE run_id=? AND session=?",
                key,
            ).fetchone()
            scans = db.execute(
                "SELECT COUNT(*), SUM(status='COMPLETE'), SUM(status='FAILED') "
                "FROM acquisition_components WHERE run_id=? AND session=?",
                key,
            ).fetchone()
            history = db.execute(
                "SELECT COUNT(*),SUM(json_extract(payload,'$.prefix_ready')), "
                "SUM(json_extract(payload,'$.broker_request')),SUM(json_extract(payload,'$.cache_hit'))"
                " "
                "FROM acquisition_history_requests WHERE run_id=? AND session=? AND "
                "phase='LIVE' AND stage=0",
                key,
            ).fetchone()
        return {
            "state": row["state"],
            "reason": row["reason"],
            "recipe_id": metadata["recipe"]["recipe_id"],
            "recipe_hash": row["recipe_hash"],
            "upstream_evidence": metadata["upstream_evidence"],
            "source_evidence": "UNVALIDATED_UPSTREAM_ACQUISITION",
            "broad_membership": broad[0],
            "raw_hits": hits[0],
            "duplicate_hits": hits[0] - hits[1],
            "scanner_unique_conids": hits[1],
            "eligible_union": pool[0],
            "acquisition_count": pool[1] or 0,
            "scanner_components": scans[0],
            "components_complete": scans[1] or 0,
            "components_failed": scans[2] or 0,
            "range5_requests_completed": history[0],
            "range5_prefixes_ready": history[1] or 0,
            "range5_broker_requests": history[2] or 0,
            "range5_cache_hits": history[3] or 0,
            "all_membership_request_baseline": broad[0],
            "range5_data_status": (
                "RANGE5_DATA_READY" if pool[1] and history[1] == pool[1] else "RANGE5_DATA_PENDING"
            ),
            "oracle_state": row["audit_state"],
            "oracle_completed": broad[1] or 0,
            "oracle_metrics": json.loads(row["oracle_metrics"]),
        }

    def details(
        self, run_id: str, session: date, kind: str, limit: int, offset: int
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        if kind in {"targets", "misses", "contributions"}:
            field = "component_contributions" if kind == "contributions" else "targets"
            predicate = " AND json_extract(j.value,'$.captured')=0" if kind == "misses" else ""
            with self.connect() as db:
                rows = db.execute(
                    "SELECT j.key,j.value FROM acquisition_sessions s, "
                    "json_each(s.oracle_metrics, ?) j WHERE s.run_id=? AND s.session=?"
                    + predicate
                    + " LIMIT ? OFFSET ?",
                    ("$." + field, run_id, str(session), limit, offset),
                ).fetchall()
            return [{"key": r[0], **json.loads(r[1])} for r in rows]
        tables = {
            "hits": "acquisition_hits",
            "components": "acquisition_components",
            "pool": "acquisition_pool",
            "broad": "acquisition_broad",
            "requests": "acquisition_history_requests",
            "oracle": "acquisition_oracle_ranks",
        }
        table = tables[kind]
        with self.connect() as db:
            rows = db.execute(
                f"SELECT * FROM {table} WHERE run_id=? AND session=? "
                "ORDER BY rowid LIMIT ? OFFSET ?",
                (run_id, str(session), limit, offset),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for field in (
                "reference",
                "identity",
                "payload",
                "request",
                "audit",
                "bars",
                "five_minute_bar",
            ):
                if item.get(field) is not None:
                    item[field] = json.loads(item[field])
            result.append(item)
        return result

    def next_oracle_row(self, run_id: str, session: date) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM acquisition_broad WHERE run_id=? AND session=? "
                "AND eligibility NOT IN ('ERROR','INELIGIBLE') AND bars IS NULL "
                "ORDER BY source_index LIMIT 1",
                (run_id, str(session)),
            ).fetchone()
        return dict(row) if row else None

    def resume_audit(self, run_id: str, session: date) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE acquisition_broad SET eligibility='UNRESOLVED' "
                "WHERE run_id=? AND session=? AND eligibility='ERROR'",
                (run_id, str(session)),
            )
            db.execute(
                "UPDATE acquisition_sessions SET audit_state='PENDING' "
                "WHERE run_id=? AND session=? AND audit_state='INCOMPLETE'",
                (run_id, str(session)),
            )

    def oracle_rows(self, run_id: str, session: date) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM acquisition_broad WHERE run_id=? AND session=? ORDER "
                    "BY source_index",
                    (run_id, str(session)),
                )
            ]

    def audit_progress(
        self,
        run_id: str,
        session: date,
        source_index: int,
        identity: CandidateIdentity | None,
        eligibility: str,
        bars: Sequence[Any] | None,
        error: str = "",
        five_bar: Sequence[Any] | None = None,
    ) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE acquisition_broad SET "
                "identity=?,eligibility=?,bars=?,error=?,five_minute_bar=? "
                "WHERE run_id=? AND session=? AND source_index=?",
                (
                    encoded(asdict(identity)) if identity else None,
                    eligibility,
                    encoded([asdict(b) for b in bars]) if bars is not None else None,
                    error,
                    encoded([asdict(b) for b in five_bar]) if five_bar is not None else None,
                    run_id,
                    str(session),
                    source_index,
                ),
            )

    def complete_oracle(
        self,
        run_id: str,
        session: date,
        rankings: Sequence[Sequence[CandidateRank]],
        metrics: dict[str, Any],
    ) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT audit_state FROM acquisition_sessions WHERE run_id=? AND session=?",
                (run_id, str(session)),
            ).fetchone()
            if row and row[0] == "COMPLETE":
                return
            db.executemany(
                "INSERT INTO acquisition_oracle_ranks VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    (
                        run_id,
                        str(session),
                        stage,
                        r.identity.con_id,
                        r.score_name,
                        r.value,
                        r.rank,
                        int(r.selected),
                        r.missing_reason,
                    )
                    for stage, ranked in enumerate(rankings)
                    for r in ranked
                ),
            )
            db.execute(
                "UPDATE acquisition_sessions SET audit_state='COMPLETE',oracle_metrics=? "
                "WHERE run_id=? AND session=?",
                (encoded(metrics), run_id, str(session)),
            )

    def audit_state(self, run_id: str, session: date, state: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE acquisition_sessions SET audit_state=? WHERE run_id=? AND "
                "session=? AND audit_state!='COMPLETE'",
                (state, run_id, str(session)),
            )

    def benchmark(self, run_id: str, session: date) -> dict[str, Any]:
        """On-demand instrumentation; overview only uses aggregate counts."""
        with self.connect() as db:
            requests = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM acquisition_history_requests WHERE run_id=? AND session=? "
                    "AND phase='LIVE' AND stage=0",
                    (run_id, str(session)),
                )
            ]
            components = [
                dict(r)
                for r in db.execute(
                    "SELECT sweep,component,audit FROM acquisition_components WHERE "
                    "run_id=? AND session=?",
                    (run_id, str(session)),
                )
            ]
            stage_timing = []
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='opening_candidate_stages'"
            ).fetchone():
                stage_timing = [
                    dict(r)
                    for r in db.execute(
                        "SELECT stage,MIN(due_at) AS cutoff,MAX(selected_at) AS calculated_at,"
                        "MAX((julianday(selected_at)-julianday(due_at))*86400000) AS "
                        "completion_delay_ms "
                        "FROM opening_candidate_stages WHERE run_id=? AND session=? GROUP BY stage",
                        (run_id, str(session)),
                    )
                ]

        def percentiles(values: list[float]) -> dict[str, float | None]:
            values.sort()
            return {
                f"p{p}": values[min(len(values) - 1, int((len(values) - 1) * p / 100))]
                if values
                else None
                for p in (50, 90, 95, 99)
            }

        scans = [json.loads(c["audit"]) for c in components]
        failures = [r.get("error", "") for r in requests + scans if r.get("error")]
        return {
            "stage_timing": stage_timing,
            "actual_scanner_requests": sum("request_start" in r for r in scans),
            "shared_scanner_results": sum(bool(r.get("shared_scanner_result")) for r in scans),
            "history_latency_ms": percentiles(
                [r["latency_ms"] for r in requests if "latency_ms" in r]
            ),
            "scanner_latency_ms": percentiles(
                [r["latency_ms"] for r in scans if "latency_ms" in r]
            ),
            "first_bar_availability_delay_ms": min(
                (r["availability_delay_ms"] for r in requests if r.get("prefix_ready")),
                default=None,
            ),
            "completed_by_next_stage": sum(
                bool(r.get("completed_before_next_stage")) for r in requests
            ),
            "missing_prefixes": sum(bool(r.get("missing_prefix")) for r in requests),
            "failures": failures,
            "pacing_errors": sum("pacing" in e.lower() or "420:" in e for e in failures),
            "entitlement_errors": sum("permission" in e.lower() or "354:" in e for e in failures),
            "sweeps": [
                {
                    "sweep": sweep,
                    "raw_hits": sum(
                        r.get("row_count", 0)
                        for c, r in zip(components, scans, strict=True)
                        if c["sweep"] == sweep
                    ),
                }
                for sweep in sorted({c["sweep"] for c in components})
            ],
        }

    def recall_history(self, run_id: str) -> dict[str, Any]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT session,recipe_hash,oracle_metrics FROM acquisition_sessions "
                "WHERE run_id=? AND audit_state='COMPLETE' ORDER BY session",
                (run_id,),
            ).fetchall()
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            metrics = json.loads(row["oracle_metrics"])
            for recipe, stages in metrics.get("recipes", {}).items():
                for stage, metric in stages.items():
                    if metric["recall"] is not None:
                        key = f"{row['recipe_hash']}:{recipe}:{stage}"
                        groups.setdefault(key, []).append({"session": row["session"], **metric})
        return {
            key: {
                "sessions": len(days),
                "mean": mean(d["recall"] for d in days),
                "median": median(d["recall"] for d in days),
                "worst_day": min(days, key=lambda d: d["recall"]),
                "exact_recall_days": sum(d["recall"] == 1 for d in days),
            }
            for key, days in groups.items()
        }
