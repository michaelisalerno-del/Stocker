"""Bounded deterministic valuation of proposals as virtual shadow evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from stocker_runtime.domain import ShadowCostPolicy, ShadowFillPolicy
from stocker_runtime.storage import connect_v2

MAX_PROPOSALS_PER_CALL = 32
MAX_EVIDENCE_SEQUENCES_PER_CALL = 256


@dataclass(frozen=True)
class ShadowPolicy:
    """Frozen per-position valuation policy with bounded outcome horizons."""

    fill: ShadowFillPolicy
    cost: ShadowCostPolicy
    horizons_us: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.horizons_us or len(self.horizons_us) > 8:
            raise ValueError("shadow policy requires 1..8 horizons")
        if tuple(sorted(set(self.horizons_us))) != self.horizons_us:
            raise ValueError("shadow horizons must be unique and ascending")
        if self.horizons_us[0] <= 0:
            raise ValueError("shadow horizons must be positive")
        if self.horizons_us[-1] > 30 * 24 * 60 * 60 * 1_000_000:
            raise ValueError("shadow horizon exceeds 30 days")

    @classmethod
    def default(cls) -> ShadowPolicy:
        return cls(
            fill=ShadowFillPolicy(model_id="conservative_quote_v1"),
            cost=ShadowCostPolicy(model_id="zero_bps_v1"),
            horizons_us=(30 * 60 * 1_000_000,),
        )


@dataclass(frozen=True)
class _Snapshot:
    source_sequence: int
    actual_at_us: int
    quotes: tuple[sqlite3.Row, ...]


class _PolicyMismatch(ValueError):
    pass


class ShadowEngine:
    """One-writer, restart-safe virtual evidence projector for one shadow run."""

    def __init__(
        self, database_path: str | Path, *, run_id: str, policy: ShadowPolicy | None = None
    ) -> None:
        self.database_path = Path(database_path)
        self.run_id = run_id
        self.policy = policy or ShadowPolicy.default()

    @staticmethod
    def _position_id(output_id: str) -> str:
        return hashlib.sha256(f"shadow-position-v2:{output_id}".encode()).hexdigest()

    def _policy_json(self) -> str:
        return json.dumps(
            {
                "cost": self.policy.cost.model_dump(mode="json"),
                "fill": self.policy.fill.model_dump(mode="json"),
                "horizons_us": self.policy.horizons_us,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def _policy_hash(self) -> str:
        return hashlib.sha256(self._policy_json().encode()).hexdigest()

    def run_once(self, *, now_us: int) -> int:
        """Advance bounded causal evidence; isolate one virtual position's failure."""

        with connect_v2(self.database_path) as connection:
            run = connection.execute(
                "SELECT mode, data_class FROM runs WHERE run_id=?", (self.run_id,)
            ).fetchone()
            if (
                run is None
                or str(run["mode"]) != "shadow"
                or str(run["data_class"]) != "shadow_protected"
            ):
                raise ValueError("shadow engine requires a shadow-protected shadow run")
            connection.execute("BEGIN IMMEDIATE")
            proposals = tuple(
                connection.execute(
                    "SELECT output.output_id, output.instance_id, output.last_input_event_id "
                    "FROM idea_outputs output LEFT JOIN shadow_positions position "
                    "ON position.proposed_trade_output_id=output.output_id "
                    "WHERE output.run_id=? AND output.output_kind='proposed_trade' "
                    "AND output.authority_status='unapproved' "
                    "AND output.data_class='shadow_protected' "
                    "AND (position.position_id IS NULL "
                    "OR position.lifecycle IN ('pending','open')) "
                    "ORDER BY output.emitted_at_us, output.output_id LIMIT ?",
                    (self.run_id, MAX_PROPOSALS_PER_CALL),
                )
            )
            changed = 0
            remaining = MAX_EVIDENCE_SEQUENCES_PER_CALL
            for ordinal, proposal in enumerate(proposals):
                if remaining <= 0:
                    break
                savepoint = f"shadow_position_{ordinal}"
                connection.execute(f"SAVEPOINT {savepoint}")  # noqa: S608
                try:
                    position_changed, consumed = self._advance(
                        connection, proposal, now_us=now_us, sequence_limit=remaining
                    )
                except (sqlite3.Error, _PolicyMismatch):
                    connection.execute(f"ROLLBACK TO {savepoint}")  # noqa: S608
                    connection.execute(f"RELEASE {savepoint}")  # noqa: S608
                    raise
                except Exception as error:
                    connection.execute(f"ROLLBACK TO {savepoint}")  # noqa: S608
                    connection.execute(f"RELEASE {savepoint}")  # noqa: S608
                    self._record_incident(connection, proposal, now_us, error)
                    continue
                connection.execute(f"RELEASE {savepoint}")  # noqa: S608
                changed += position_changed
                remaining -= consumed
            connection.commit()
            return changed

    def _advance(
        self,
        connection: sqlite3.Connection,
        proposal: sqlite3.Row,
        *,
        now_us: int,
        sequence_limit: int,
    ) -> tuple[int, int]:
        position = self._ensure_position(connection, proposal, now_us)
        if str(position["lifecycle"]) == "invalid":
            return 1, 0
        if str(position["policy_hash"]) != self._policy_hash():
            raise _PolicyMismatch("shadow policy differs from the persisted position policy")
        progress = connection.execute(
            "SELECT * FROM shadow_progress WHERE position_id=?", (position["position_id"],)
        ).fetchone()
        if progress is None:
            raise RuntimeError("shadow position is missing durable progress")
        legs = tuple(
            connection.execute(
                "SELECT proposal.leg_number, proposal.instrument_id, proposal.action, "
                "proposal.quantity_value, shadow.side, shadow.quantity, shadow.entry_price "
                "FROM idea_output_legs proposal LEFT JOIN shadow_legs shadow "
                "ON shadow.position_id=? AND shadow.leg_number=proposal.leg_number "
                "WHERE proposal.output_id=? ORDER BY proposal.leg_number",
                (position["position_id"], proposal["output_id"]),
            )
        )
        sequences = self._sequences(
            connection,
            legs,
            after_sequence=int(progress["next_source_sequence"]) - 1,
            now_us=now_us,
            limit=sequence_limit,
        )
        changed = 0
        consumed = 0
        for sequence in sequences:
            consumed += 1
            snapshot = self._snapshot(
                connection,
                legs,
                after_sequence=int(progress["entry_after_source_sequence"]),
                at_sequence=sequence,
            )
            self._set_cursor(connection, str(position["position_id"]), sequence + 1, now_us)
            if snapshot is None:
                continue
            if str(position["lifecycle"]) == "pending":
                self._open(connection, position, legs, snapshot, now_us)
                changed += 1
                position = connection.execute(
                    "SELECT * FROM shadow_positions WHERE position_id=?", (position["position_id"],)
                ).fetchone()
                progress = connection.execute(
                    "SELECT * FROM shadow_progress WHERE position_id=?", (position["position_id"],)
                ).fetchone()
                legs = tuple(
                    connection.execute(
                        "SELECT proposal.leg_number, proposal.instrument_id, proposal.action, "
                        "proposal.quantity_value, shadow.side, shadow.quantity, shadow.entry_price "
                        "FROM idea_output_legs proposal JOIN shadow_legs shadow "
                        "ON shadow.position_id=? AND shadow.leg_number=proposal.leg_number "
                        "WHERE proposal.output_id=? ORDER BY proposal.leg_number",
                        (position["position_id"], proposal["output_id"]),
                    )
                )
                continue
            if self._observe(connection, position, legs, progress, snapshot, now_us):
                changed = 1
            position = connection.execute(
                "SELECT * FROM shadow_positions WHERE position_id=?", (position["position_id"],)
            ).fetchone()
            if str(position["lifecycle"]) == "closed":
                break
            progress = connection.execute(
                "SELECT * FROM shadow_progress WHERE position_id=?", (position["position_id"],)
            ).fetchone()
        return changed, consumed

    def _ensure_position(
        self, connection: sqlite3.Connection, proposal: sqlite3.Row, now_us: int
    ) -> sqlite3.Row:
        existing = connection.execute(
            "SELECT * FROM shadow_positions WHERE proposed_trade_output_id=?",
            (proposal["output_id"],),
        ).fetchone()
        if existing is not None:
            return cast(sqlite3.Row, existing)
        proposed_legs = tuple(
            connection.execute(
                "SELECT leg_number, instrument_id, action, quantity_value, currency "
                "FROM idea_output_legs WHERE output_id=? ORDER BY leg_number",
                (proposal["output_id"],),
            )
        )
        position_id = self._position_id(str(proposal["output_id"]))
        currencies = {str(leg["currency"] or "") for leg in proposed_legs}
        if not proposed_legs or any(leg["quantity_value"] is None for leg in proposed_legs):
            self._invalidate(connection, position_id, proposal, "quantity_unavailable", now_us)
            return cast(
                sqlite3.Row,
                connection.execute(
                    "SELECT * FROM shadow_positions WHERE position_id=?", (position_id,)
                ).fetchone(),
            )
        if len(currencies) != 1 or not next(iter(currencies)):
            self._invalidate(connection, position_id, proposal, "currency_unavailable", now_us)
            return cast(
                sqlite3.Row,
                connection.execute(
                    "SELECT * FROM shadow_positions WHERE position_id=?", (position_id,)
                ).fetchone(),
            )
        boundary = connection.execute(
            "SELECT coalesce(source_sequence, derived_after_source_sequence) "
            "FROM market_events WHERE event_id=? AND run_id=?",
            (proposal["last_input_event_id"], self.run_id),
        ).fetchone()
        if boundary is None or boundary[0] is None:
            raise RuntimeError("proposal entry boundary evidence is unavailable")
        connection.execute(
            "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
            "instance_id, lifecycle, cost_model_id, fill_model_id, currency, data_class, "
            "policy_json, policy_hash) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, "
            "'shadow_protected', ?, ?)",
            (
                position_id,
                proposal["output_id"],
                self.run_id,
                proposal["instance_id"],
                self.policy.cost.model_id,
                self.policy.fill.model_id,
                next(iter(currencies)),
                self._policy_json(),
                self._policy_hash(),
            ),
        )
        connection.execute(
            "INSERT INTO shadow_progress(position_id, entry_after_source_sequence, "
            "next_source_sequence, next_horizon_index, updated_at_us) VALUES (?, ?, ?, 0, ?)",
            (position_id, int(boundary[0]), int(boundary[0]) + 1, now_us),
        )
        return cast(
            sqlite3.Row,
            connection.execute(
                "SELECT * FROM shadow_positions WHERE position_id=?", (position_id,)
            ).fetchone(),
        )

    def _sequences(
        self,
        connection: sqlite3.Connection,
        legs: tuple[sqlite3.Row, ...],
        *,
        after_sequence: int,
        now_us: int,
        limit: int,
    ) -> tuple[int, ...]:
        instruments = tuple(str(leg["instrument_id"]) for leg in legs)
        placeholders = ",".join("?" for _ in instruments)
        rows = connection.execute(
            f"SELECT DISTINCT source_sequence FROM market_events WHERE run_id=? "  # noqa: S608
            f"AND instrument_id IN ({placeholders}) AND source_sequence>? "
            "AND source_sequence IS NOT NULL AND event_at_us<=? "
            "ORDER BY source_sequence LIMIT ?",
            (self.run_id, *instruments, after_sequence, now_us, limit),
        )
        return tuple(int(row[0]) for row in rows)

    def _snapshot(
        self,
        connection: sqlite3.Connection,
        legs: tuple[sqlite3.Row, ...],
        *,
        after_sequence: int,
        at_sequence: int,
    ) -> _Snapshot | None:
        quotes: list[sqlite3.Row] = []
        for leg in legs:
            quote = connection.execute(
                "SELECT event_id, source_sequence, event_at_us, bid_value, ask_value "
                "FROM market_events WHERE run_id=? AND instrument_id=? "
                "AND source_sequence>? AND source_sequence<=? "
                "ORDER BY source_sequence DESC, event_id DESC LIMIT 1",
                (self.run_id, leg["instrument_id"], after_sequence, at_sequence),
            ).fetchone()
            if quote is None:
                return None
            quotes.append(quote)
        actual_at_us = max(int(quote["event_at_us"]) for quote in quotes)
        if any(
            actual_at_us - int(quote["event_at_us"]) > self.policy.fill.max_quote_age_us
            or float(quote["bid_value"] or 0) <= 0
            or float(quote["ask_value"] or 0) <= 0
            or float(quote["bid_value"]) > float(quote["ask_value"])
            for quote in quotes
        ):
            return None
        return _Snapshot(at_sequence, actual_at_us, tuple(quotes))

    def _open(
        self,
        connection: sqlite3.Connection,
        position: sqlite3.Row,
        legs: tuple[sqlite3.Row, ...],
        snapshot: _Snapshot,
        now_us: int,
    ) -> None:
        for leg, quote in zip(legs, snapshot.quotes, strict=True):
            entry_price = float(
                quote["ask_value"] if str(leg["action"]) == "buy" else quote["bid_value"]
            )
            connection.execute(
                "INSERT INTO shadow_legs(position_id, leg_number, instrument_id, side, quantity, "
                "entry_market_event_id, entry_price) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    position["position_id"],
                    leg["leg_number"],
                    leg["instrument_id"],
                    leg["action"],
                    leg["quantity_value"],
                    quote["event_id"],
                    entry_price,
                ),
            )
        final_target = snapshot.actual_at_us + self.policy.horizons_us[-1]
        connection.execute(
            "UPDATE shadow_positions SET lifecycle='open', opened_at_us=? WHERE position_id=?",
            (snapshot.actual_at_us, position["position_id"]),
        )
        connection.execute(
            "UPDATE shadow_progress SET entry_source_sequence=?, next_source_sequence=?, "
            "final_target_at_us=?, updated_at_us=? WHERE position_id=?",
            (
                snapshot.source_sequence,
                snapshot.source_sequence + 1,
                final_target,
                now_us,
                position["position_id"],
            ),
        )

    def _observe(
        self,
        connection: sqlite3.Connection,
        position: sqlite3.Row,
        legs: tuple[sqlite3.Row, ...],
        progress: sqlite3.Row,
        snapshot: _Snapshot,
        now_us: int,
    ) -> bool:
        gross, net, return_value, gross_value, event_ids = self._value(legs, snapshot)
        mfe = progress["mfe"]
        mae = progress["mae"]
        mfe_value = gross if mfe is None else max(float(mfe), gross)
        mae_value = gross if mae is None else min(float(mae), gross)
        mfe_ids = (
            event_ids
            if mfe is None or gross > float(mfe)
            else json.loads(progress["mfe_event_ids_json"])
        )
        mae_ids = (
            event_ids
            if mae is None or gross < float(mae)
            else json.loads(progress["mae_event_ids_json"])
        )
        next_horizon = int(progress["next_horizon_index"])
        prior_horizon = next_horizon
        while next_horizon < len(self.policy.horizons_us):
            target_at = int(position["opened_at_us"]) + self.policy.horizons_us[next_horizon]
            if snapshot.actual_at_us < target_at:
                break
            payload = self._mark_payload(next_horizon, target_at, snapshot, event_ids)
            connection.execute(
                "INSERT INTO shadow_marks(position_id, marked_at_us, gross_value, gross_pnl, "
                "net_pnl, return_value, quality_bits, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, ?) ON CONFLICT DO NOTHING",
                (
                    position["position_id"],
                    target_at,
                    gross_value,
                    gross,
                    net,
                    return_value,
                    payload,
                ),
            )
            next_horizon += 1
        connection.execute(
            "UPDATE shadow_progress SET next_source_sequence=?, next_horizon_index=?, "
            "mfe=?, mae=?, "
            "mfe_event_ids_json=?, mae_event_ids_json=?, updated_at_us=? WHERE position_id=?",
            (
                snapshot.source_sequence + 1,
                next_horizon,
                mfe_value,
                mae_value,
                json.dumps(mfe_ids, separators=(",", ":")),
                json.dumps(mae_ids, separators=(",", ":")),
                now_us,
                position["position_id"],
            ),
        )
        if next_horizon != len(self.policy.horizons_us):
            return next_horizon != prior_horizon
        for leg, quote in zip(legs, snapshot.quotes, strict=True):
            exit_price = float(
                quote["bid_value"] if str(leg["side"]) == "buy" else quote["ask_value"]
            )
            connection.execute(
                "UPDATE shadow_legs SET exit_market_event_id=?, exit_price=? "
                "WHERE position_id=? AND leg_number=?",
                (quote["event_id"], exit_price, position["position_id"], leg["leg_number"]),
            )
        outcome_payload = json.dumps(
            {
                "exit_market_event_ids": event_ids,
                "mae_market_event_ids": mae_ids,
                "mfe_market_event_ids": mfe_ids,
                "policy_hash": self._policy_hash(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        connection.execute(
            "UPDATE shadow_positions SET lifecycle='closed', closed_at_us=? WHERE position_id=?",
            (snapshot.actual_at_us, position["position_id"]),
        )
        connection.execute(
            "INSERT INTO shadow_outcomes(position_id, outcome_at_us, reason, gross_pnl, net_pnl, "
            "return_value, mfe, mae, completeness, payload_json) "
            "VALUES (?, ?, 'horizon', ?, ?, ?, ?, ?, 'complete', ?)",
            (
                position["position_id"],
                snapshot.actual_at_us,
                gross,
                net,
                return_value,
                mfe_value,
                mae_value,
                outcome_payload,
            ),
        )
        return True

    def _value(
        self, legs: tuple[sqlite3.Row, ...], snapshot: _Snapshot
    ) -> tuple[float, float, float | None, float, list[str]]:
        gross = 0.0
        entry_notional = 0.0
        event_ids: list[str] = []
        for leg, quote in zip(legs, snapshot.quotes, strict=True):
            entry = float(leg["entry_price"])
            quantity = float(leg["quantity"])
            mark = float(quote["bid_value"] if str(leg["side"]) == "buy" else quote["ask_value"])
            gross += (1.0 if str(leg["side"]) == "buy" else -1.0) * (mark - entry) * quantity
            entry_notional += entry * quantity
            event_ids.append(str(quote["event_id"]))
        cost = entry_notional * self.policy.cost.per_side_bps / 10_000.0
        return (
            gross,
            gross - 2 * cost,
            gross / entry_notional if entry_notional else None,
            entry_notional + gross,
            event_ids,
        )

    def _mark_payload(
        self,
        horizon_index: int,
        target_at_us: int,
        snapshot: _Snapshot,
        event_ids: list[str],
    ) -> str:
        return json.dumps(
            {
                "actual_quote_at_us": snapshot.actual_at_us,
                "horizon_index": horizon_index,
                "market_event_ids": event_ids,
                "policy_hash": self._policy_hash(),
                "target_at_us": target_at_us,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def _set_cursor(
        self, connection: sqlite3.Connection, position_id: str, sequence: int, now_us: int
    ) -> None:
        connection.execute(
            "UPDATE shadow_progress SET next_source_sequence=?, updated_at_us=? "
            "WHERE position_id=?",
            (sequence, now_us, position_id),
        )

    def _record_incident(
        self, connection: sqlite3.Connection, proposal: sqlite3.Row, now_us: int, error: Exception
    ) -> None:
        incident_id = hashlib.sha256(f"shadow:{proposal['output_id']}".encode()).hexdigest()
        details = json.dumps(
            {"error_type": type(error).__name__, "proposal_output_id": proposal["output_id"]},
            separators=(",", ":"),
            sort_keys=True,
        )
        connection.execute(
            "INSERT INTO incidents(incident_id, run_id, scope, severity, code, opened_at_us, "
            "details_json) VALUES (?, ?, 'shadow_position', 'degraded', "
            "'SHADOW_POSITION_FAILED', ?, ?) ON CONFLICT(incident_id) DO UPDATE SET "
            "opened_at_us=excluded.opened_at_us, resolved_at_us=NULL, "
            "details_json=excluded.details_json",
            (incident_id, self.run_id, now_us, details),
        )

    def _invalidate(
        self,
        connection: sqlite3.Connection,
        position_id: str,
        proposal: sqlite3.Row,
        reason: str,
        now_us: int,
    ) -> None:
        connection.execute(
            "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
            "instance_id, closed_at_us, lifecycle, cost_model_id, fill_model_id, currency, "
            "invalid_reason, data_class, policy_json, policy_hash) "
            "VALUES (?, ?, ?, ?, ?, 'invalid', ?, ?, 'UNKNOWN', ?, 'shadow_protected', ?, ?)",
            (
                position_id,
                proposal["output_id"],
                self.run_id,
                proposal["instance_id"],
                now_us,
                self.policy.cost.model_id,
                self.policy.fill.model_id,
                reason,
                self._policy_json(),
                self._policy_hash(),
            ),
        )
