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
from stocker_runtime.storage.shadow import terminalize_expired_pending_positions

MAX_PROPOSALS_PER_CALL = 32
MAX_EVIDENCE_SEQUENCES_PER_CALL = 256
EVIDENCE_SEQUENCES_PER_POSITION = MAX_EVIDENCE_SEQUENCES_PER_CALL // MAX_PROPOSALS_PER_CALL
MAX_TRADE_LEGS = 8
MAX_EVENTS_PER_INSTRUMENT_FETCH = EVIDENCE_SEQUENCES_PER_POSITION
MAX_PENDING_RETENTION_US = 30 * 24 * 60 * 60 * 1_000_000


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
class _Quote:
    instrument_id: str
    bid_event_id: str
    bid_source_sequence: int
    bid_at_us: int
    bid_value: float
    ask_event_id: str
    ask_source_sequence: int
    ask_at_us: int
    ask_value: float


@dataclass
class _QuoteState:
    leg_number: int
    instrument_id: str
    bid_event_id: str | None
    bid_source_sequence: int | None
    bid_at_us: int | None
    bid_value: float | None
    ask_event_id: str | None
    ask_source_sequence: int | None
    ask_at_us: int | None
    ask_value: float | None


@dataclass(frozen=True)
class _Snapshot:
    source_sequence: int
    actual_at_us: int
    quotes: tuple[_Quote, ...]


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
            scheduled = tuple(
                connection.execute(
                    "SELECT output_id FROM shadow_schedule "
                    "INDEXED BY shadow_schedule_run_count_idx WHERE run_id=? "
                    "ORDER BY schedule_count, output_id LIMIT ?",
                    (self.run_id, MAX_PROPOSALS_PER_CALL),
                )
            )
            changed = 0
            remaining = MAX_EVIDENCE_SEQUENCES_PER_CALL
            for ordinal, schedule in enumerate(scheduled):
                if remaining <= 0:
                    break
                connection.execute(
                    "UPDATE shadow_schedule SET schedule_count=schedule_count + 1 "
                    "WHERE output_id=? AND run_id=?",
                    (schedule["output_id"], self.run_id),
                )
                proposal = cast(
                    sqlite3.Row,
                    connection.execute(
                        "SELECT output_id, instance_id, emitted_at_us, last_input_event_id "
                        "FROM idea_outputs WHERE output_id=? AND run_id=? "
                        "AND output_kind='proposed_trade' AND authority_status='unapproved' "
                        "AND data_class='shadow_protected'",
                        (schedule["output_id"], self.run_id),
                    ).fetchone(),
                )
                savepoint = f"shadow_position_{ordinal}"
                connection.execute(f"SAVEPOINT {savepoint}")  # noqa: S608
                try:
                    position_changed, consumed = self._advance(
                        connection,
                        proposal,
                        now_us=now_us,
                        sequence_limit=min(remaining, EVIDENCE_SEQUENCES_PER_POSITION),
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
        if terminalize_expired_pending_positions(
            connection,
            now_us=now_us,
            limit=1,
            position_id=str(position["position_id"]),
        ):
            return 1, 0
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
        states = self._quote_states(connection, str(position["position_id"]))
        if len(states) != len(legs) or any(
            state.leg_number != int(leg["leg_number"])
            or state.instrument_id != str(leg["instrument_id"])
            for state, leg in zip(states, legs, strict=True)
        ):
            raise RuntimeError("shadow position quote state does not match its proposal legs")
        events = self._events(
            connection,
            legs,
            next_sequence=int(progress["next_source_sequence"]),
            limit=sequence_limit,
        )
        changed = 0
        consumed = 0
        for event in events:
            if int(event["received_at_us"]) > now_us:
                break
            consumed += 1
            sequence = int(event["source_sequence"])
            self._apply_event(
                connection,
                str(position["position_id"]),
                states,
                event,
            )
            snapshot = self._snapshot(states, at_sequence=sequence)
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
        if len(proposed_legs) > MAX_TRADE_LEGS:
            self._invalidate(connection, position_id, proposal, "leg_count_exceeds_limit", now_us)
            return cast(
                sqlite3.Row,
                connection.execute(
                    "SELECT * FROM shadow_positions WHERE position_id=?", (position_id,)
                ).fetchone(),
            )
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
            "next_source_sequence, next_horizon_index, updated_at_us, "
            "pending_retention_deadline_us) VALUES (?, ?, ?, 0, ?, ?)",
            (
                position_id,
                int(boundary[0]),
                int(boundary[0]) + 1,
                now_us,
                int(proposal["emitted_at_us"]) + MAX_PENDING_RETENTION_US,
            ),
        )
        connection.executemany(
            "INSERT INTO shadow_quote_state(position_id, leg_number, instrument_id) "
            "VALUES (?, ?, ?)",
            (
                (position_id, int(leg["leg_number"]), str(leg["instrument_id"]))
                for leg in proposed_legs
            ),
        )
        return cast(
            sqlite3.Row,
            connection.execute(
                "SELECT * FROM shadow_positions WHERE position_id=?", (position_id,)
            ).fetchone(),
        )

    def _quote_states(self, connection: sqlite3.Connection, position_id: str) -> list[_QuoteState]:
        rows = connection.execute(
            "SELECT leg_number, instrument_id, bid_event_id, bid_source_sequence, bid_at_us, "
            "bid_value, ask_event_id, ask_source_sequence, ask_at_us, ask_value "
            "FROM shadow_quote_state WHERE position_id=? ORDER BY leg_number",
            (position_id,),
        )
        return [
            _QuoteState(
                leg_number=int(row["leg_number"]),
                instrument_id=str(row["instrument_id"]),
                bid_event_id=None if row["bid_event_id"] is None else str(row["bid_event_id"]),
                bid_source_sequence=(
                    None if row["bid_source_sequence"] is None else int(row["bid_source_sequence"])
                ),
                bid_at_us=None if row["bid_at_us"] is None else int(row["bid_at_us"]),
                bid_value=None if row["bid_value"] is None else float(row["bid_value"]),
                ask_event_id=None if row["ask_event_id"] is None else str(row["ask_event_id"]),
                ask_source_sequence=(
                    None if row["ask_source_sequence"] is None else int(row["ask_source_sequence"])
                ),
                ask_at_us=None if row["ask_at_us"] is None else int(row["ask_at_us"]),
                ask_value=None if row["ask_value"] is None else float(row["ask_value"]),
            )
            for row in rows
        ]

    def _events(
        self,
        connection: sqlite3.Connection,
        legs: tuple[sqlite3.Row, ...],
        *,
        next_sequence: int,
        limit: int,
    ) -> tuple[sqlite3.Row, ...]:
        fetched: list[sqlite3.Row] = []
        instruments = tuple(dict.fromkeys(str(leg["instrument_id"]) for leg in legs))
        for instrument_id in instruments:
            fetched.extend(
                connection.execute(
                    "SELECT source_sequence, event_id, instrument_id, event_at_us, "
                    "received_at_us, quality_bits, bid_value, ask_value FROM market_events "
                    "INDEXED BY market_events_shadow_raw_idx "
                    "WHERE run_id=? AND instrument_id=? AND event_kind='quote' "
                    "AND source_sequence>=? AND source_sequence IS NOT NULL "
                    "ORDER BY source_sequence, event_id LIMIT ?",
                    (
                        self.run_id,
                        instrument_id,
                        next_sequence,
                        min(limit, MAX_EVENTS_PER_INSTRUMENT_FETCH),
                    ),
                )
            )
        fetched.sort(key=lambda row: (int(row["source_sequence"]), str(row["event_id"])))
        return tuple(fetched[:limit])

    def _apply_event(
        self,
        connection: sqlite3.Connection,
        position_id: str,
        states: list[_QuoteState],
        event: sqlite3.Row,
    ) -> None:
        if int(event["quality_bits"]) != 0 or int(event["event_at_us"]) > int(
            event["received_at_us"]
        ):
            return
        instrument_id = str(event["instrument_id"])
        event_id = str(event["event_id"])
        source_sequence = int(event["source_sequence"])
        event_at_us = int(event["event_at_us"])
        if event["bid_value"] is not None:
            bid_value = float(event["bid_value"])
            for state in states:
                if state.instrument_id == instrument_id:
                    state.bid_event_id = event_id
                    state.bid_source_sequence = source_sequence
                    state.bid_at_us = event_at_us
                    state.bid_value = bid_value
            connection.execute(
                "UPDATE shadow_quote_state SET bid_event_id=?, bid_source_sequence=?, "
                "bid_at_us=?, bid_value=? WHERE position_id=? AND instrument_id=?",
                (event_id, source_sequence, event_at_us, bid_value, position_id, instrument_id),
            )
        if event["ask_value"] is not None:
            ask_value = float(event["ask_value"])
            for state in states:
                if state.instrument_id == instrument_id:
                    state.ask_event_id = event_id
                    state.ask_source_sequence = source_sequence
                    state.ask_at_us = event_at_us
                    state.ask_value = ask_value
            connection.execute(
                "UPDATE shadow_quote_state SET ask_event_id=?, ask_source_sequence=?, "
                "ask_at_us=?, ask_value=? WHERE position_id=? AND instrument_id=?",
                (event_id, source_sequence, event_at_us, ask_value, position_id, instrument_id),
            )

    def _snapshot(self, states: list[_QuoteState], *, at_sequence: int) -> _Snapshot | None:
        if len(states) == 0 or any(
            state.bid_event_id is None
            or state.bid_source_sequence is None
            or state.bid_at_us is None
            or state.bid_value is None
            or state.ask_event_id is None
            or state.ask_source_sequence is None
            or state.ask_at_us is None
            or state.ask_value is None
            for state in states
        ):
            return None
        quotes = tuple(
            _Quote(
                instrument_id=state.instrument_id,
                bid_event_id=cast(str, state.bid_event_id),
                bid_source_sequence=cast(int, state.bid_source_sequence),
                bid_at_us=cast(int, state.bid_at_us),
                bid_value=cast(float, state.bid_value),
                ask_event_id=cast(str, state.ask_event_id),
                ask_source_sequence=cast(int, state.ask_source_sequence),
                ask_at_us=cast(int, state.ask_at_us),
                ask_value=cast(float, state.ask_value),
            )
            for state in states
        )
        actual_at_us = max(
            side_at_us for quote in quotes for side_at_us in (quote.bid_at_us, quote.ask_at_us)
        )
        if any(
            actual_at_us - quote.bid_at_us > self.policy.fill.max_quote_age_us
            or actual_at_us - quote.ask_at_us > self.policy.fill.max_quote_age_us
            or quote.bid_value <= 0
            or quote.ask_value <= 0
            or quote.bid_value > quote.ask_value
            for quote in quotes
        ):
            return None
        return _Snapshot(at_sequence, actual_at_us, quotes)

    def _open(
        self,
        connection: sqlite3.Connection,
        position: sqlite3.Row,
        legs: tuple[sqlite3.Row, ...],
        snapshot: _Snapshot,
        now_us: int,
    ) -> None:
        for leg, quote in zip(legs, snapshot.quotes, strict=True):
            is_buy = str(leg["action"]) == "buy"
            entry_price = quote.ask_value if is_buy else quote.bid_value
            entry_event_id = quote.ask_event_id if is_buy else quote.bid_event_id
            connection.execute(
                "INSERT INTO shadow_legs(position_id, leg_number, instrument_id, side, quantity, "
                "entry_market_event_id, entry_price, entry_bid_market_event_id, "
                "entry_ask_market_event_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    position["position_id"],
                    leg["leg_number"],
                    leg["instrument_id"],
                    leg["action"],
                    leg["quantity_value"],
                    entry_event_id,
                    entry_price,
                    quote.bid_event_id,
                    quote.ask_event_id,
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
        gross, net, return_value, gross_value, quote_event_ids = self._value(legs, snapshot)
        mfe = progress["mfe"]
        mae = progress["mae"]
        mfe_value = gross if mfe is None else max(float(mfe), gross)
        mae_value = gross if mae is None else min(float(mae), gross)
        mfe_ids = (
            quote_event_ids
            if mfe is None or gross > float(mfe)
            else json.loads(progress["mfe_event_ids_json"])
        )
        mae_ids = (
            quote_event_ids
            if mae is None or gross < float(mae)
            else json.loads(progress["mae_event_ids_json"])
        )
        next_horizon = int(progress["next_horizon_index"])
        prior_horizon = next_horizon
        while next_horizon < len(self.policy.horizons_us):
            target_at = int(position["opened_at_us"]) + self.policy.horizons_us[next_horizon]
            if snapshot.actual_at_us < target_at:
                break
            payload = self._mark_payload(next_horizon, target_at, snapshot, quote_event_ids)
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
            is_buy = str(leg["side"]) == "buy"
            exit_price = quote.bid_value if is_buy else quote.ask_value
            exit_event_id = quote.bid_event_id if is_buy else quote.ask_event_id
            connection.execute(
                "UPDATE shadow_legs SET exit_market_event_id=?, exit_price=?, "
                "exit_bid_market_event_id=?, exit_ask_market_event_id=? "
                "WHERE position_id=? AND leg_number=?",
                (
                    exit_event_id,
                    exit_price,
                    quote.bid_event_id,
                    quote.ask_event_id,
                    position["position_id"],
                    leg["leg_number"],
                ),
            )
        outcome_payload = json.dumps(
            {
                "exit_quote_event_ids": quote_event_ids,
                "mae_quote_event_ids": mae_ids,
                "mfe_quote_event_ids": mfe_ids,
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
    ) -> tuple[float, float, float | None, float, list[dict[str, int | str]]]:
        gross = 0.0
        entry_notional = 0.0
        quote_event_ids: list[dict[str, int | str]] = []
        for leg, quote in zip(legs, snapshot.quotes, strict=True):
            entry = float(leg["entry_price"])
            quantity = float(leg["quantity"])
            mark = quote.bid_value if str(leg["side"]) == "buy" else quote.ask_value
            gross += (1.0 if str(leg["side"]) == "buy" else -1.0) * (mark - entry) * quantity
            entry_notional += entry * quantity
            quote_event_ids.append(
                {
                    "ask_event_id": quote.ask_event_id,
                    "bid_event_id": quote.bid_event_id,
                    "instrument_id": quote.instrument_id,
                    "leg_number": int(leg["leg_number"]),
                }
            )
        cost = entry_notional * self.policy.cost.per_side_bps / 10_000.0
        return (
            gross,
            gross - 2 * cost,
            gross / entry_notional if entry_notional else None,
            entry_notional + gross,
            quote_event_ids,
        )

    def _mark_payload(
        self,
        horizon_index: int,
        target_at_us: int,
        snapshot: _Snapshot,
        quote_event_ids: list[dict[str, int | str]],
    ) -> str:
        return json.dumps(
            {
                "actual_quote_at_us": snapshot.actual_at_us,
                "horizon_index": horizon_index,
                "policy_hash": self._policy_hash(),
                "quote_event_ids": quote_event_ids,
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
