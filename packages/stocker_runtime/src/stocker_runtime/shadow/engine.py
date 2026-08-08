"""Deterministic valuation of unapproved proposed trades as virtual shadow evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from stocker_runtime.domain import ShadowCostPolicy, ShadowFillPolicy
from stocker_runtime.storage import connect_v2


@dataclass(frozen=True)
class ShadowPolicy:
    """Frozen per-engine valuation policy with bounded outcome horizons."""

    fill: ShadowFillPolicy
    cost: ShadowCostPolicy
    horizons_us: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.horizons_us or len(self.horizons_us) > 8:
            raise ValueError("shadow policy requires 1..8 horizons")
        if tuple(sorted(set(self.horizons_us))) != self.horizons_us:
            raise ValueError("shadow horizons must be unique and ascending")
        if self.horizons_us[-1] > 30 * 24 * 60 * 60 * 1_000_000:
            raise ValueError("shadow horizon exceeds 30 days")

    @classmethod
    def default(cls) -> ShadowPolicy:
        return cls(
            fill=ShadowFillPolicy(model_id="conservative_quote_v1"),
            cost=ShadowCostPolicy(model_id="zero_bps_v1"),
            horizons_us=(30 * 60 * 1_000_000,),
        )


class ShadowEngine:
    """One-writer, restart-safe virtual evidence projector for an existing shadow run."""

    def __init__(
        self, database_path: str | Path, *, run_id: str, policy: ShadowPolicy | None = None
    ):
        self.database_path = Path(database_path)
        self.run_id = run_id
        self.policy = policy or ShadowPolicy.default()

    @staticmethod
    def _position_id(output_id: str) -> str:
        return hashlib.sha256(f"shadow-position-v1:{output_id}".encode()).hexdigest()

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
        """Advance every proposal once; exact retries are harmless and causal only."""

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
                    "SELECT output.output_id, output.instance_id, output.emitted_at_us, "
                    "output.last_input_event_id FROM idea_outputs output "
                    "LEFT JOIN shadow_positions position "
                    "ON position.proposed_trade_output_id=output.output_id "
                    "WHERE output.run_id=? AND output.output_kind='proposed_trade' "
                    "AND output.authority_status='unapproved' "
                    "AND output.data_class='shadow_protected' "
                    "AND (position.position_id IS NULL "
                    "OR position.lifecycle IN ('pending','open')) "
                    "ORDER BY output.emitted_at_us, output.output_id",
                    (self.run_id,),
                )
            )
            changed = 0
            for proposal in proposals:
                changed += self._advance(connection, proposal, now_us)
            connection.commit()
            return changed

    def _advance(self, connection: sqlite3.Connection, proposal: sqlite3.Row, now_us: int) -> int:
        output_id = str(proposal["output_id"])
        position_id = self._position_id(output_id)
        position = connection.execute(
            "SELECT * FROM shadow_positions WHERE proposed_trade_output_id=?", (output_id,)
        ).fetchone()
        if position is None:
            legs = tuple(
                connection.execute(
                    "SELECT leg_number, instrument_id, action, quantity_value, currency "
                    "FROM idea_output_legs WHERE output_id=? ORDER BY leg_number",
                    (output_id,),
                )
            )
            currencies = {str(leg["currency"] or "") for leg in legs}
            if not legs or any(leg["quantity_value"] is None for leg in legs):
                return self._invalidate(
                    connection, position_id, proposal, "quantity_unavailable", now_us
                )
            if len(currencies) != 1 or not next(iter(currencies)):
                return self._invalidate(
                    connection, position_id, proposal, "currency_unavailable", now_us
                )
            connection.execute(
                "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
                "instance_id, "
                "lifecycle, cost_model_id, fill_model_id, currency, data_class, policy_json, "
                "policy_hash) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, 'shadow_protected', ?, ?)",
                (
                    position_id,
                    output_id,
                    self.run_id,
                    str(proposal["instance_id"]),
                    self.policy.cost.model_id,
                    self.policy.fill.model_id,
                    next(iter(currencies)),
                    self._policy_json(),
                    self._policy_hash(),
                ),
            )
            position = connection.execute(
                "SELECT * FROM shadow_positions WHERE position_id=?", (position_id,)
            ).fetchone()
        if str(position["policy_hash"]) != self._policy_hash():
            raise ValueError("shadow policy differs from the persisted position policy")
        if str(position["lifecycle"]) == "pending":
            return self._open_if_ready(connection, position, proposal, now_us)
        return self._mark_or_close(connection, position, now_us)

    def _quote(
        self,
        connection: sqlite3.Connection,
        instrument_id: str,
        after_sequence: int,
        now_us: int,
        *,
        latest: bool = False,
        not_before_us: int = 0,
        before_sequence: int | None = None,
        reference_at_us: int | None = None,
    ) -> sqlite3.Row | None:
        upper_sequence = (
            before_sequence if before_sequence is not None else 9_223_372_036_854_775_807
        )
        rows = tuple(
            connection.execute(
                "SELECT event_id, source_sequence, event_at_us, bid_value, ask_value "
                "FROM market_events WHERE run_id=? AND instrument_id=? AND source_sequence>? "
                "AND source_sequence<=? "
                "AND event_at_us BETWEEN ? AND ? ORDER BY source_sequence, event_id",
                (self.run_id, instrument_id, after_sequence, upper_sequence, not_before_us, now_us),
            )
        )
        reference_at = now_us if reference_at_us is None else reference_at_us
        valid = tuple(
            row
            for row in rows
            if reference_at - int(row["event_at_us"]) <= self.policy.fill.max_quote_age_us
            and float(row["bid_value"] or 0) > 0
            and float(row["ask_value"] or 0) > 0
            and float(row["bid_value"]) <= float(row["ask_value"])
        )
        if not valid:
            return None
        return cast(sqlite3.Row, valid[-1] if latest else valid[0])

    def _open_if_ready(
        self,
        connection: sqlite3.Connection,
        position: sqlite3.Row,
        proposal: sqlite3.Row,
        now_us: int,
    ) -> int:
        boundary = connection.execute(
            "SELECT source_sequence FROM market_events WHERE event_id=?",
            (proposal["last_input_event_id"],),
        ).fetchone()
        if boundary is None or boundary[0] is None:
            return 0
        legs = tuple(
            connection.execute(
                "SELECT leg_number, instrument_id, action, quantity_value FROM idea_output_legs "
                "WHERE output_id=? ORDER BY leg_number",
                (proposal["output_id"],),
            )
        )
        quotes = tuple(
            self._quote(connection, str(leg["instrument_id"]), int(boundary[0]), now_us)
            for leg in legs
        )
        if any(quote is None for quote in quotes):
            return 0
        if any(
            float(quote["bid_value"]) <= 0
            or float(quote["ask_value"]) <= 0
            or float(quote["bid_value"]) > float(quote["ask_value"])
            for quote in quotes
            if quote is not None
        ):
            return self._invalidate(
                connection,
                str(position["position_id"]),
                proposal,
                "crossed_or_invalid_quote",
                now_us,
            )
        opened_at = max(int(quote["event_at_us"]) for quote in quotes if quote is not None)
        for leg, quote in zip(legs, quotes, strict=True):
            assert quote is not None
            price = float(quote["ask_value"] if str(leg["action"]) == "buy" else quote["bid_value"])
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
                    price,
                ),
            )
        connection.execute(
            "UPDATE shadow_positions SET lifecycle='open', opened_at_us=? WHERE position_id=?",
            (opened_at, position["position_id"]),
        )
        return 1

    def _mark_or_close(
        self, connection: sqlite3.Connection, position: sqlite3.Row, now_us: int
    ) -> int:
        legs = tuple(
            connection.execute(
                "SELECT * FROM shadow_legs WHERE position_id=? ORDER BY leg_number",
                (position["position_id"],),
            )
        )
        entry_boundary = max(
            int(
                connection.execute(
                    "SELECT source_sequence FROM market_events WHERE event_id=?",
                    (leg["entry_market_event_id"],),
                ).fetchone()[0]
            )
            for leg in legs
        )
        self._record_causal_marks(connection, position, legs, entry_boundary, now_us)
        horizon_at_us = int(position["opened_at_us"]) + self.policy.horizons_us[-1]
        quotes = tuple(
            self._quote(
                connection,
                str(leg["instrument_id"]),
                entry_boundary,
                now_us,
                latest=True,
                not_before_us=horizon_at_us if now_us >= horizon_at_us else 0,
            )
            for leg in legs
        )
        if any(quote is None for quote in quotes):
            return 0
        if any(
            now_us - int(quote["event_at_us"]) > self.policy.fill.max_quote_age_us
            for quote in quotes
            if quote is not None
        ):
            return 0
        if any(
            float(quote["bid_value"]) <= 0
            or float(quote["ask_value"]) <= 0
            or float(quote["bid_value"]) > float(quote["ask_value"])
            for quote in quotes
            if quote is not None
        ):
            return 0
        gross = 0.0
        entry_notional = 0.0
        mark_ids: list[str] = []
        for leg, quote in zip(legs, quotes, strict=True):
            assert quote is not None
            entry = float(leg["entry_price"])
            quantity = float(leg["quantity"])
            mark = float(quote["bid_value"] if str(leg["side"]) == "buy" else quote["ask_value"])
            sign = 1.0 if str(leg["side"]) == "buy" else -1.0
            gross += sign * (mark - entry) * quantity
            entry_notional += entry * quantity
            mark_ids.append(str(quote["event_id"]))
        cost = entry_notional * self.policy.cost.per_side_bps / 10_000.0
        marked_at = max(int(quote["event_at_us"]) for quote in quotes if quote is not None)
        payload = json.dumps(
            {"market_event_ids": mark_ids, "cost_model_id": self.policy.cost.model_id},
            separators=(",", ":"),
            sort_keys=True,
        )
        connection.execute(
            "INSERT INTO shadow_marks(position_id, marked_at_us, gross_value, gross_pnl, net_pnl, "
            "return_value, quality_bits, payload_json) VALUES (?, ?, ?, ?, ?, ?, 0, ?) "
            "ON CONFLICT(position_id, marked_at_us) DO NOTHING",
            (
                position["position_id"],
                marked_at,
                entry_notional + gross,
                gross,
                gross - 2 * cost,
                gross / entry_notional if entry_notional else None,
                payload,
            ),
        )
        if now_us < horizon_at_us:
            return 1
        for leg, quote in zip(legs, quotes, strict=True):
            assert quote is not None
            exit_price = float(
                quote["bid_value"] if str(leg["side"]) == "buy" else quote["ask_value"]
            )
            connection.execute(
                "UPDATE shadow_legs SET exit_market_event_id=?, exit_price=? "
                "WHERE position_id=? AND leg_number=?",
                (quote["event_id"], exit_price, position["position_id"], leg["leg_number"]),
            )
        extremes = tuple(
            connection.execute(
                "SELECT gross_pnl FROM shadow_marks WHERE position_id=?", (position["position_id"],)
            )
        )
        connection.execute(
            "UPDATE shadow_positions SET lifecycle='closed', closed_at_us=? WHERE position_id=?",
            (marked_at, position["position_id"]),
        )
        connection.execute(
            "INSERT INTO shadow_outcomes(position_id, outcome_at_us, reason, gross_pnl, net_pnl, "
            "return_value, mfe, mae, completeness, payload_json) "
            "VALUES (?, ?, 'horizon', ?, ?, ?, ?, ?, 'complete', ?) "
            "ON CONFLICT(position_id) DO NOTHING",
            (
                position["position_id"],
                marked_at,
                gross,
                gross - 2 * cost,
                gross / entry_notional if entry_notional else None,
                max(float(row[0]) for row in extremes),
                min(float(row[0]) for row in extremes),
                payload,
            ),
        )
        return 1

    def _record_causal_marks(
        self,
        connection: sqlite3.Connection,
        position: sqlite3.Row,
        legs: tuple[sqlite3.Row, ...],
        entry_boundary: int,
        now_us: int,
    ) -> None:
        """Persist each complete causal quote snapshot, including excursions while offline."""

        instrument_ids = tuple(str(leg["instrument_id"]) for leg in legs)
        placeholders = ",".join("?" for _ in instrument_ids)
        sequences = tuple(
            connection.execute(
                f"SELECT source_sequence, max(event_at_us) AS reference_at_us "  # noqa: S608
                "FROM market_events WHERE run_id=? "
                f"AND instrument_id IN ({placeholders}) AND source_sequence>? AND event_at_us<=? "
                "GROUP BY source_sequence ORDER BY source_sequence",
                (self.run_id, *instrument_ids, entry_boundary, now_us),
            )
        )
        for sequence_row in sequences:
            sequence = int(sequence_row[0])
            reference_at_us = int(sequence_row["reference_at_us"])
            quotes = tuple(
                self._quote(
                    connection,
                    str(leg["instrument_id"]),
                    entry_boundary,
                    now_us,
                    latest=True,
                    before_sequence=sequence,
                    reference_at_us=reference_at_us,
                )
                for leg in legs
            )
            if any(quote is None for quote in quotes):
                continue
            gross = 0.0
            entry_notional = 0.0
            event_ids: list[str] = []
            for leg, quote in zip(legs, quotes, strict=True):
                assert quote is not None
                entry = float(leg["entry_price"])
                quantity = float(leg["quantity"])
                value = float(
                    quote["bid_value"] if str(leg["side"]) == "buy" else quote["ask_value"]
                )
                gross += (1.0 if str(leg["side"]) == "buy" else -1.0) * (value - entry) * quantity
                entry_notional += entry * quantity
                event_ids.append(str(quote["event_id"]))
            marked_at = max(int(quote["event_at_us"]) for quote in quotes if quote is not None)
            cost = entry_notional * self.policy.cost.per_side_bps / 10_000.0
            payload = json.dumps(
                {"market_event_ids": event_ids}, separators=(",", ":"), sort_keys=True
            )
            connection.execute(
                "INSERT INTO shadow_marks(position_id, marked_at_us, gross_value, gross_pnl, "
                "net_pnl, "
                "return_value, quality_bits, payload_json) VALUES (?, ?, ?, ?, ?, ?, 0, ?) "
                "ON CONFLICT(position_id, marked_at_us) DO NOTHING",
                (
                    position["position_id"],
                    marked_at,
                    entry_notional + gross,
                    gross,
                    gross - 2 * cost,
                    gross / entry_notional if entry_notional else None,
                    payload,
                ),
            )

    def _invalidate(
        self,
        connection: sqlite3.Connection,
        position_id: str,
        proposal: sqlite3.Row,
        reason: str,
        now_us: int,
    ) -> int:
        connection.execute(
            "INSERT INTO shadow_positions(position_id, proposed_trade_output_id, run_id, "
            "instance_id, closed_at_us, lifecycle, cost_model_id, fill_model_id, currency, "
            "invalid_reason, data_class) "
            "VALUES (?, ?, ?, ?, ?, 'invalid', ?, ?, 'UNKNOWN', ?, 'shadow_protected') "
            "ON CONFLICT(proposed_trade_output_id) DO NOTHING",
            (
                position_id,
                proposal["output_id"],
                self.run_id,
                proposal["instance_id"],
                now_us,
                self.policy.cost.model_id,
                self.policy.fill.model_id,
                reason,
            ),
        )
        return 1
