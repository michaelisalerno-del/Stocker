"""One-way, read-only import from the frozen Stocker V1 SQLite schemas."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from stocker_runtime.storage.connection import connect_v2, initialize_database, verify_database

IMPORTER_VERSION = "stocker-v2-legacy-import/1"
MAX_TARGET_BYTES = 8 * 1024**3
MAX_JSON_BYTES = 16 * 1024
MAX_RECONCILIATION_BYTES = 256 * 1024
_IBKR_BAR_SOURCE_NAMES = frozenset({"ibkr_realtime_bar_5_second_aggregation"})

# Schema structure, not mutable data, is frozen for every historically deployed prefix.
# Duplicate 0011/0012 prefixes retain their exact original full-name ordering.
LEGACY_SCHEMA_DIGESTS: tuple[tuple[str, str], ...] = (
    ("0001_prospective.sql", "4411dca958ad02b9474d23dc6406e27b9b3ffdf19c9005e7f59153e121dda85f"),
    ("0002_runtime_state.sql", "47c8642193145b27291900dbacbaf85ae083d41f47852e8ae0bcf31d793f0c2d"),
    (
        "0003_option_computations.sql",
        "158d17c664c6d47ece0b3c8ee0530e6df59373bf08f196afd3798631a9127762",
    ),
    (
        "0004_underlying_quote_identity.sql",
        "7747aa544d26fd2b15ef02ed6f8260296c02bd214e310cf6560f41f3c98c3bd6",
    ),
    (
        "0005_market_data_budget_events.sql",
        "c5900a8f2faafc47439245be32945864fd3acc1bbac37f7372e73641d7492f51",
    ),
    (
        "0006_parallel_source_bars.sql",
        "5c7ce59b3e829393c3e0bf69679526b909b2e2c066b23c9141e9b8924585d192",
    ),
    (
        "0007_frozen_m1c_recorder_v0.sql",
        "01abb2c0865e2adfcd4d6ce0c4592d19ec2e48ca4c68937c04d04075ab33a4ef",
    ),
    (
        "0008_promotion_decisions_and_option_model_sources.sql",
        "70903b3e455cbb5c0fb6d19d4194d54c38d726b0ca94baa219af4931b839d93d",
    ),
    (
        "0009_quiet_state_options_shadow_v0.sql",
        "35e311cd78ddeb99a73978f150803a6efd29b1ea2a9eed2c469c1e5ad4445516",
    ),
    (
        "0010_ibkr_budget_transfer_v0.sql",
        "fcc5dd918c2732bbf56b67b581dcde0e87d8473f96effc3e568c5cec9cb8cc91",
    ),
    (
        "0011_m1c_checkpoint_completion_v0.sql",
        "11574f56669555d2f9b568f95836ccc6bb03bcb7ca0e5793115b3a3b7407e290",
    ),
    (
        "0011_m1c_tail_phase_v1.sql",
        "3e27a5c7735b57fa26bc2ffc77ceb84215165aa4a1fdf0053e6b6d937ca0e07f",
    ),
    (
        "0012_m1c_signed_market_shock_v1.sql",
        "176146c921fbea769577c43f3667aef00e5ac95f24a35d41447e0e4c068040e2",
    ),
    (
        "0012_option_schedule_degradation_v0.sql",
        "62be781d96fda7782db03e33f253abff0e28838599556803d6bf22d1625f3316",
    ),
    (
        "0013_m1c_opening_market_transition_v1.sql",
        "63d41c0fb1d39d02438607f5c68ad98a87b0fab3e6df387c167c2b7883a70d76",
    ),
    (
        "0014_m1c_prospective_opening_reversal_v1.sql",
        "91abb46df8b1c4ba461c8d69cac52fb3e9614ce089bce69086d6bc27eb30dc3b",
    ),
    (
        "0015_m1c_prospective_opening_reversal_v1_1.sql",
        "e6de557f14e975bc2852486ef9f884cb880e36c6768b82a46dce3cfccabbba13",
    ),
    (
        "0016_prospective_recorder_hardening_v1.sql",
        "52667e848d6fa8f984754472278fe4955ba1e05ddc7c621af62b6f65bb70793c",
    ),
    (
        "0017_callback_raw_only_recovery_v1.sql",
        "ac291ad702ab20179fe8058c5a80c83cf404556d00d633f9f24cb9d16e4002b9",
    ),
    (
        "0018_virtual_position_ledgers_v1.sql",
        "11de9fc132b7b3917f2478644494ff09895c58bde5a2a9a702a9d804ace5e94c",
    ),
    (
        "0019_virtual_position_ledger_evidence_v1.sql",
        "70b23ea5429b266495b5a21227f058967534f2ca4b2c9ef52ba59c4cc8cf40f7",
    ),
    (
        "0020_opening_reversal_shadow_capture_v1.sql",
        "6b24ea524292c9d678b0c7032fd4ad3bc4589974fd7ab8ca28ef36a4d07555e6",
    ),
    (
        "0021_opening_reversal_activation_run_binding_v1.sql",
        "dcd67afed98f52b3d6e4c3b0f4894845232a1a6b342bbef0785214818889b8dd",
    ),
    (
        "0022_web_read_projections_v0.sql",
        "c91079841b3685eecbde247f4a5939a77af1a67d965f53670e363d06b016f5ea",
    ),
    (
        "0023_web_latest_state_v0.sql",
        "ff8582f938cef8b09276057142a1c74c3de182b6a5d7e15ec90188b04ebc6052",
    ),
    (
        "0024_m1c_validity_separation_v1.sql",
        "c21ba1294db40ffabe7e88c72fa01608bd5edfda610eeb5e0ebb5e15421475b6",
    ),
    (
        "0025_parallel_source_capture_recovery_v1.sql",
        "0200004e823e77c702cd3b94b2c18f9c402ef68e858dc21c96e8d91ff1fd3c47",
    ),
    (
        "0026_opening_leader_continuation_v0.sql",
        "ba8ef7830b5a6fafa2b297250a3d9206b7d756c30ea15f10e408eb54c58e3497",
    ),
)


class LegacyImportError(RuntimeError):
    """The stopped legacy source cannot be safely and completely reconciled."""


@dataclass(frozen=True)
class LegacyImportResult:
    target_path: Path
    reconciliation_path: Path
    migration_id: str
    source_database_hash: str
    source_schema_digest: str
    source_row_count: int
    imported_row_count: int
    omitted_row_count: int
    target_digest: str
    verification_status: Literal["verified"] = "verified"


class _Digest(Protocol):
    def update(self, value: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass
class _TableTracker:
    table: str
    source_rows: int = 0
    imported_rows: int = 0
    omitted_rows: int = 0
    omission_reasons: Counter[str] = field(default_factory=Counter)
    _digest: _Digest = field(default_factory=hashlib.sha256)

    def __post_init__(self) -> None:
        self._digest.update(_canonical_bytes({"table": self.table}))
        self._digest.update(b"\n")

    def record(
        self,
        key: Sequence[object],
        *,
        imported: bool,
        reason: str | None = None,
        target_refs: Sequence[str] = (),
    ) -> None:
        if imported == (reason is not None):
            raise LegacyImportError("row disposition must be imported or carry one omission reason")
        self.source_rows += 1
        if imported:
            self.imported_rows += 1
        else:
            assert reason is not None
            self.omitted_rows += 1
            self.omission_reasons[reason] += 1
        payload = _canonical_bytes(
            {
                "disposition": "imported" if imported else "omitted",
                "key": list(key),
                "reason": reason,
                "target_refs": list(target_refs),
            }
        )
        self._digest.update(payload)
        self._digest.update(b"\n")

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "source_rows": self.source_rows,
            "imported_rows": self.imported_rows,
            "omitted_rows": self.omitted_rows,
            "omission_reasons": dict(sorted(self.omission_reasons.items())),
            "row_classification_hash": self._digest.hexdigest(),
        }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_json(value: object, *, maximum_bytes: int = MAX_JSON_BYTES) -> str:
    encoded = _canonical_bytes(value)
    if len(encoded) > maximum_bytes:
        raise LegacyImportError("bounded migration JSON exceeds its target limit")
    return encoded.decode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stable_id(kind: str, *parts: object) -> str:
    return f"legacy-{kind}-{_sha(list(parts))[:32]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_digest(connection: sqlite3.Connection) -> str:
    definitions = [
        tuple(str(value) for value in row)
        for row in connection.execute(
            "SELECT type, name, tbl_name, COALESCE(sql, '') FROM sqlite_schema "
            "WHERE name NOT LIKE 'sqlite_%' AND name != 'schema_migrations' "
            "ORDER BY type, name"
        )
    ]
    return hashlib.sha256(_canonical_bytes(definitions)).hexdigest()


def _timestamp_us(value: object, *, label: str) -> int:
    if not isinstance(value, str) or not value:
        raise LegacyImportError(f"legacy {label} is not an ISO timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise LegacyImportError(f"legacy {label} is not an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LegacyImportError(f"legacy {label} is not timezone-aware")
    return int(parsed.astimezone(UTC).timestamp() * 1_000_000)


def _finite_float(value: object) -> float | None:
    if not isinstance(value, (str, bytes, int, float)):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (str, bytes, int)):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _row_value(row: sqlite3.Row, name: str, default: object = None) -> object:
    return row[name] if name in tuple(row.keys()) else default


class _LegacyReader:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.tables = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        self._columns = {
            table: tuple(str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")'))
            for table in self.tables
        }
        self._primary_keys = {
            table: tuple(
                str(row[1])
                for row in sorted(
                    connection.execute(f'PRAGMA table_info("{table}")'),
                    key=lambda item: int(item[5]) if int(item[5]) else 1_000_000,
                )
                if int(row[5])
            )
            for table in self.tables
        }

    def has_table(self, table: str) -> bool:
        return table in self._columns

    def has_column(self, table: str, column: str) -> bool:
        return column in self._columns.get(table, ())

    def rows(self, table: str) -> Iterator[tuple[tuple[object, ...], sqlite3.Row]]:
        primary_keys = self._primary_keys[table]
        if primary_keys:
            order = ", ".join(f'"{name}"' for name in primary_keys)
            cursor = self.connection.execute(f'SELECT * FROM "{table}" ORDER BY {order}')
            for row in cursor:
                yield tuple(row[name] for name in primary_keys), row
            return
        cursor = self.connection.execute(
            f'SELECT rowid AS __rowid__, * FROM "{table}" ORDER BY rowid'
        )
        for row in cursor:
            yield (row["__rowid__"],), row

    def count(self, table: str) -> int:
        return int(self.connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])


@dataclass
class _ImportContext:
    source: _LegacyReader
    target: sqlite3.Connection
    trackers: dict[str, _TableTracker]
    run_modes: dict[str, tuple[str, str, int]] = field(default_factory=dict)
    run_scientific_classifications: dict[str, str] = field(default_factory=dict)
    instrument_by_underlying: dict[int, str] = field(default_factory=dict)
    instrument_by_option: dict[int, str] = field(default_factory=dict)
    symbol_instruments: dict[tuple[str, str], str] = field(default_factory=dict)
    run_instruments: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    underlying_bar_events: dict[tuple[str, str, int], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    option_events: dict[tuple[int, int], set[str]] = field(default_factory=dict)
    output_ordinals: dict[tuple[str, str, int], int] = field(default_factory=dict)
    handled_tables: set[str] = field(default_factory=set)
    active_diagnostics: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def tracker(self, table: str) -> _TableTracker:
        self.handled_tables.add(table)
        return self.trackers[table]


def _insert_instrument(
    context: _ImportContext,
    *,
    instrument_id: str,
    identity: Mapping[str, object],
    con_id: int | None,
    kind: str,
    symbol: str,
    exchange: str,
    currency: str,
    expiry: str | None = None,
    strike: str | None = None,
    right: str | None = None,
    multiplier: str | None = None,
) -> None:
    context.target.execute(
        """
        INSERT OR IGNORE INTO instruments(
            instrument_id, identity_hash, ibkr_con_id, kind, symbol, exchange, currency,
            option_expiry, option_strike, option_right, option_multiplier
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            instrument_id,
            _sha(identity),
            con_id,
            kind,
            symbol[:128],
            exchange[:64],
            currency[:16],
            expiry,
            strike,
            right,
            multiplier,
        ),
    )


def _ensure_symbol_instrument(context: _ImportContext, run_id: str, symbol: str) -> str:
    key = (run_id, symbol)
    existing = context.symbol_instruments.get(key)
    if existing is not None:
        return existing
    instrument_id = _stable_id("instrument", "symbol", run_id, symbol)
    _insert_instrument(
        context,
        instrument_id=instrument_id,
        identity={"kind": "stock", "run_id": run_id, "symbol": symbol},
        con_id=None,
        kind="stock",
        symbol=symbol or "UNKNOWN",
        exchange="UNKNOWN",
        currency="USD",
    )
    context.symbol_instruments[key] = instrument_id
    context.run_instruments[run_id].add(instrument_id)
    return instrument_id


def _insert_callback_event(
    context: _ImportContext,
    *,
    run_id: str,
    instrument_id: str,
    feed_kind: str,
    event_kind: str,
    event_at_us: int,
    received_at_us: int,
    provider_at_us: int | None,
    source_table: str,
    source_key: Sequence[object],
    values: Mapping[str, float | None],
) -> tuple[str, int]:
    event_id = _stable_id("event", source_table, *source_key)
    evidence = {
        "migration_source": {"key": list(source_key), "table": source_table},
        "normalized": {key: value for key, value in sorted(values.items()) if value is not None},
    }
    payload_json = _canonical_json(evidence, maximum_bytes=64 * 1024)
    cursor = context.target.execute(
        """
        INSERT INTO callback_inbox(
            event_uid, run_id, recorder_generation, connection_generation, request_id,
            callback_kind, received_at_us, provider_at_us, payload_json, payload_sha256,
            lifecycle, attempts
        ) VALUES (?, ?, 0, 0, NULL, 'legacy_import', ?, ?, NULL, ?, 'pending', 0)
        """,
        (
            _stable_id("callback", source_table, *source_key),
            run_id,
            received_at_us,
            provider_at_us,
            hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        ),
    )
    if cursor.lastrowid is None:
        raise LegacyImportError("V2 callback insertion did not allocate a source sequence")
    source_sequence = cursor.lastrowid
    context.target.execute(
        """
        INSERT INTO market_events(
            event_id, run_id, source_sequence, derived_after_source_sequence,
            instrument_id, feed_kind, event_kind, event_at_us, received_at_us,
            connection_generation, quality_bits, open_value, high_value, low_value,
            close_value, volume_value, bid_value, ask_value, bid_size_value,
            ask_size_value, last_value, size_value, payload_json, payload_sha256
        ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            run_id,
            source_sequence,
            instrument_id,
            feed_kind,
            event_kind,
            event_at_us,
            received_at_us,
            values.get("open"),
            values.get("high"),
            values.get("low"),
            values.get("close"),
            values.get("volume"),
            values.get("bid"),
            values.get("ask"),
            values.get("bid_size"),
            values.get("ask_size"),
            values.get("last"),
            values.get("size"),
            payload_json,
            hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        ),
    )
    context.target.execute(
        """
        UPDATE callback_inbox
        SET lifecycle='acknowledged', normalized_event_id=?, acknowledged_at_us=?
        WHERE source_sequence=?
        """,
        (event_id, received_at_us, source_sequence),
    )
    return event_id, source_sequence


def _import_runs(context: _ImportContext) -> None:
    tracker = context.tracker("prospective_run")
    for key, row in context.source.rows("prospective_run"):
        run_id = str(row["run_id"])
        try:
            started_at_us = _timestamp_us(row["prospective_start_utc"], label="run start")
        except LegacyImportError:
            tracker.record(key, imported=False, reason="invalid_run_timestamp")
            continue
        legacy_mode = str(row["mode"]).lower()
        if legacy_mode not in {"record_only", "shadow"}:
            tracker.record(key, imported=False, reason="unsupported_legacy_mode")
            continue
        if not _run_has_ibkr_market_evidence(context, run_id):
            tracker.record(key, imported=False, reason="source_provenance_not_ibkr")
            continue
        scientific_classification = row["scientific_classification"]
        if not isinstance(scientific_classification, str) or not scientific_classification:
            tracker.record(key, imported=False, reason="scientific_classification_invalid")
            continue
        try:
            _canonical_json({"legacy_scientific_classification": scientific_classification})
        except LegacyImportError:
            tracker.record(key, imported=False, reason="scientific_classification_unbounded")
            continue
        mode = "shadow" if legacy_mode == "shadow" else "prospective_record"
        data_class = "shadow_protected" if mode == "shadow" else "prospective_protected"
        config_hash = _sha(
            {
                "app_version": row["app_version"],
                "cohort": row["cohort"],
                "model_artifact_id": row["model_artifact_id"],
                "scientific_classification": scientific_classification,
                "universe_id": row["universe_id"],
            }
        )
        context.target.execute(
            """
            INSERT INTO runs(
                run_id, mode, source, started_at_us, ended_at_us, config_hash, git_commit,
                data_class, status, prior_run_id
            ) VALUES (?, ?, 'ibkr', ?, ?, ?, ?, ?, 'stopped', NULL)
            """,
            (
                run_id,
                mode,
                started_at_us,
                started_at_us,
                config_hash,
                str(row["git_commit"]),
                data_class,
            ),
        )
        context.target.execute(
            """
            INSERT INTO recorder_generations(
                run_id, generation, owner_id, started_at_us, ended_at_us, clean_stop,
                termination_code
            ) VALUES (?, 0, 'legacy-import', ?, ?, 1, 'MIGRATED_STOPPED')
            """,
            (run_id, started_at_us, started_at_us),
        )
        context.run_modes[run_id] = (mode, data_class, started_at_us)
        context.run_scientific_classifications[run_id] = scientific_classification
        tracker.record(key, imported=True, target_refs=(f"runs:{run_id}",))


def _run_has_ibkr_market_evidence(context: _ImportContext, run_id: str) -> bool:
    if not context.source.has_column("underlying_bar", "bar_source"):
        return False
    sources = tuple(sorted(_IBKR_BAR_SOURCE_NAMES))
    placeholders = ",".join("?" for _ in sources)
    found = context.source.connection.execute(
        "SELECT 1 FROM underlying_bar WHERE run_id=? "
        f"AND lower(trim(bar_source)) IN ({placeholders}) LIMIT 1",
        (run_id, *sources),
    ).fetchone()
    return found is not None


def _import_instruments(context: _ImportContext) -> None:
    tracker = context.tracker("underlying_contract")
    for key, row in context.source.rows("underlying_contract"):
        run_id = str(row["run_id"])
        con_id_value = _row_value(row, "con_id")
        con_id = _integer(con_id_value)
        if run_id not in context.run_modes or con_id is None:
            tracker.record(key, imported=False, reason="incomplete_instrument_identity")
            continue
        symbol = str(row["symbol"])
        instrument_id = _stable_id("instrument", "ibkr", con_id)
        _insert_instrument(
            context,
            instrument_id=instrument_id,
            identity={"con_id": con_id, "kind": "stock"},
            con_id=con_id,
            kind="stock",
            symbol=symbol,
            exchange=str(_row_value(row, "exchange") or "SMART"),
            currency=str(_row_value(row, "currency") or "USD"),
        )
        context.instrument_by_underlying[int(row["id"])] = instrument_id
        context.symbol_instruments[(run_id, symbol)] = instrument_id
        context.run_instruments[run_id].add(instrument_id)
        tracker.record(key, imported=True, target_refs=(f"instruments:{instrument_id}",))


def _import_option_instruments(context: _ImportContext) -> None:
    tracker = context.tracker("option_contract")
    for key, row in context.source.rows("option_contract"):
        run_id = str(row["run_id"])
        con_id_value = _row_value(row, "con_id")
        con_id = _integer(con_id_value)
        if run_id not in context.run_modes or con_id is None:
            tracker.record(key, imported=False, reason="incomplete_instrument_identity")
            continue
        right_raw = str(row["right"]).lower()
        right = {"c": "call", "call": "call", "p": "put", "put": "put"}.get(right_raw)
        if right is None:
            tracker.record(key, imported=False, reason="invalid_option_right")
            continue
        instrument_id = _stable_id("instrument", "ibkr", con_id)
        local_symbol = str(row["local_symbol"])
        base_symbol = local_symbol.split()[0] if local_symbol else "UNKNOWN"
        _insert_instrument(
            context,
            instrument_id=instrument_id,
            identity={"con_id": con_id, "kind": "option"},
            con_id=con_id,
            kind="option",
            symbol=base_symbol,
            exchange=str(_row_value(row, "exchange") or "SMART"),
            currency="USD",
            expiry=str(row["expiry"]),
            strike=str(row["strike"]),
            right=right,
            multiplier=str(_row_value(row, "multiplier") or "100"),
        )
        context.instrument_by_option[int(row["id"])] = instrument_id
        context.run_instruments[run_id].add(instrument_id)
        tracker.record(key, imported=True, target_refs=(f"instruments:{instrument_id}",))


def _import_market_events(context: _ImportContext) -> None:
    tracker = context.tracker("underlying_bar")
    for key, row in context.source.rows("underlying_bar"):
        run_id = str(row["run_id"])
        source_name = str(_row_value(row, "bar_source") or "").strip().lower()
        if source_name not in _IBKR_BAR_SOURCE_NAMES:
            tracker.record(key, imported=False, reason="non_ibkr_market_data")
            continue
        if run_id not in context.run_modes:
            tracker.record(key, imported=False, reason="run_not_imported")
            continue
        try:
            event_at_us = _timestamp_us(row["bar_end_utc"], label="bar end")
            received_at_us = _timestamp_us(row["receive_timestamp_utc"], label="bar receive")
            provider_at_us = (
                _timestamp_us(row["source_timestamp_utc"], label="bar provider")
                if row["source_timestamp_utc"] is not None
                else None
            )
        except LegacyImportError:
            tracker.record(key, imported=False, reason="invalid_market_timestamp")
            continue
        instrument_id = _ensure_symbol_instrument(context, run_id, str(row["symbol"]))
        event_id, _sequence = _insert_callback_event(
            context,
            run_id=run_id,
            instrument_id=instrument_id,
            feed_kind="bars",
            event_kind="bar",
            event_at_us=event_at_us,
            received_at_us=received_at_us,
            provider_at_us=provider_at_us,
            source_table="underlying_bar",
            source_key=key,
            values={
                "open": _finite_float(row["open"]),
                "high": _finite_float(row["high"]),
                "low": _finite_float(row["low"]),
                "close": _finite_float(row["close"]),
                "volume": _finite_float(row["activity_value"]),
            },
        )
        context.underlying_bar_events[(run_id, str(row["symbol"]), event_at_us)].add(event_id)
        tracker.record(key, imported=True, target_refs=(f"market_events:{event_id}",))

    tracker = context.tracker("underlying_quote")
    for key, _row in context.source.rows("underlying_quote"):
        tracker.record(
            key,
            imported=False,
            reason="source_provenance_unverifiable",
        )

    tracker = context.tracker("option_quote")
    for key, row in context.source.rows("option_quote"):
        run_id = str(row["run_id"])
        option_id = int(row["option_contract_id"])
        option_instrument_id = context.instrument_by_option.get(option_id)
        # V1's computation_source is a quote role (bid/ask/last/model), not a vendor.
        # Provider admission is therefore fixed once per run by the exact active IBKR
        # bar-source label; deterministic replay runs cannot reach this mapping.
        if run_id not in context.run_modes or option_instrument_id is None:
            tracker.record(key, imported=False, reason="instrument_not_imported")
            continue
        timestamp_value = _row_value(row, "provider_timestamp_utc") or _row_value(
            row, "receive_timestamp_utc"
        )
        receive_value = _row_value(row, "receive_timestamp_utc") or timestamp_value
        try:
            event_at_us = _timestamp_us(timestamp_value, label="option quote event")
            received_at_us = _timestamp_us(receive_value, label="option quote receive")
        except LegacyImportError:
            tracker.record(key, imported=False, reason="invalid_market_timestamp")
            continue
        event_id, _sequence = _insert_callback_event(
            context,
            run_id=run_id,
            instrument_id=option_instrument_id,
            feed_kind="quotes",
            event_kind="quote",
            event_at_us=event_at_us,
            received_at_us=received_at_us,
            provider_at_us=event_at_us,
            source_table="option_quote",
            source_key=key,
            values={
                "bid": _finite_float(row["bid"]),
                "ask": _finite_float(row["ask"]),
                "bid_size": _finite_float(row["bid_size"]),
                "ask_size": _finite_float(row["ask_size"]),
                "last": _finite_float(row["last"]),
                "size": _finite_float(row["last_size"]),
            },
        )
        for event_timestamp_us in {event_at_us, received_at_us}:
            context.option_events.setdefault((option_id, event_timestamp_us), set()).add(event_id)
        tracker.record(key, imported=True, target_refs=(f"market_events:{event_id}",))


def _project_market_latest(context: _ImportContext) -> None:
    rows = context.target.execute(
        """
        SELECT event.* FROM market_events event
        JOIN (
            SELECT instrument_id, feed_kind, max(event_at_us) AS latest_at
            FROM market_events WHERE feed_kind != 'migration'
            GROUP BY instrument_id, feed_kind
        ) latest ON latest.instrument_id=event.instrument_id
            AND latest.feed_kind=event.feed_kind AND latest.latest_at=event.event_at_us
        WHERE event.event_id=(
            SELECT max(tie.event_id) FROM market_events tie
            WHERE tie.instrument_id=event.instrument_id AND tie.feed_kind=event.feed_kind
              AND tie.event_at_us=event.event_at_us
        )
        ORDER BY event.instrument_id, event.feed_kind
        """
    )
    for row in rows:
        context.target.execute(
            """
            INSERT INTO market_latest(
                run_id, instrument_id, feed_kind, event_id, event_at_us, received_at_us,
                event_kind, quality_bits, bid_value, bid_source_event_id, ask_value,
                ask_source_event_id, bid_size_value, bid_size_source_event_id,
                ask_size_value, ask_size_source_event_id, last_value, last_source_event_id,
                size_value, size_source_event_id, close_value, close_source_event_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["run_id"],
                row["instrument_id"],
                row["feed_kind"],
                row["event_id"],
                row["event_at_us"],
                row["received_at_us"],
                row["event_kind"],
                row["bid_value"],
                row["event_id"] if row["bid_value"] is not None else None,
                row["ask_value"],
                row["event_id"] if row["ask_value"] is not None else None,
                row["bid_size_value"],
                row["event_id"] if row["bid_size_value"] is not None else None,
                row["ask_size_value"],
                row["event_id"] if row["ask_size_value"] is not None else None,
                row["last_value"],
                row["event_id"] if row["last_value"] is not None else None,
                row["size_value"],
                row["event_id"] if row["size_value"] is not None else None,
                row["close_value"],
                row["event_id"] if row["close_value"] is not None else None,
            ),
        )


def _install_legacy_idea_instances(context: _ImportContext) -> None:
    manifest = {
        "api_version": 1,
        "description": "Generic read-only projection of selected immutable V1 evidence.",
        "display_name": "Legacy imported evidence",
        "idea_id": "legacy.generic.import",
        "idea_version": "1",
    }
    manifest_json = _canonical_json(manifest, maximum_bytes=64 * 1024)
    manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    code_hash = _sha({"importer": IMPORTER_VERSION, "projection": 1})
    discovered_at = min((value[2] for value in context.run_modes.values()), default=0)
    context.target.execute(
        """
        INSERT INTO idea_plugins(
            idea_id, idea_version, api_version, display_name, description, manifest_hash,
            code_hash, manifest_json, discovered_at_us
        ) VALUES ('legacy.generic.import', '1', 1, 'Legacy imported evidence',
                  'Generic read-only projection of selected immutable V1 evidence.', ?, ?, ?, ?)
        """,
        (manifest_hash, code_hash, manifest_json, discovered_at),
    )
    for run_id, (mode, data_class, started_at_us) in sorted(context.run_modes.items()):
        universe = sorted(context.run_instruments[run_id])
        universe_json = _canonical_json(universe, maximum_bytes=64 * 1024)
        parameters = {
            "legacy_scientific_classification": context.run_scientific_classifications[run_id]
        }
        parameters_json = _canonical_json(parameters)
        requirements_json = "[]"
        context.target.execute(
            """
            INSERT INTO idea_instances(
                instance_id, idea_id, idea_version, run_id, mode, parameters_json,
                parameters_hash, plugin_code_hash, manifest_hash, universe_json,
                universe_hash, requirements_json, requirements_hash,
                activated_after_source_sequence, activated_at_us, deactivated_at_us,
                health, error_code, data_class
            ) VALUES (?, 'legacy.generic.import', '1', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?,
                      'disabled', 'MIGRATED_LEGACY_ARCHIVE', ?)
            """,
            (
                _stable_id("instance", run_id),
                run_id,
                mode,
                parameters_json,
                _sha(parameters),
                code_hash,
                manifest_hash,
                universe_json,
                hashlib.sha256(universe_json.encode("utf-8")).hexdigest(),
                requirements_json,
                hashlib.sha256(requirements_json.encode("utf-8")).hexdigest(),
                started_at_us,
                started_at_us,
                data_class,
            ),
        )
        if mode == "shadow":
            policy_json = _canonical_json({"source": "legacy_import", "virtual_only": True})
            context.target.execute(
                "INSERT INTO shadow_run_policies(run_id, policy_hash, policy_json) "
                "VALUES (?, ?, ?)",
                (run_id, hashlib.sha256(policy_json.encode()).hexdigest(), policy_json),
            )


def _insert_output(
    context: _ImportContext,
    *,
    source_table: str,
    source_key: Sequence[object],
    run_id: str,
    output_kind: Literal["observation", "signal", "proposed_trade"],
    symbol: str,
    emitted_at_us: int,
    payload: Mapping[str, object],
    input_event_ids: Sequence[str],
    strength: float | None = None,
    direction: str | None = None,
    legs: Sequence[tuple[str, str, float, float | None]] = (),
) -> str:
    mode_data = context.run_modes.get(run_id)
    if mode_data is None:
        raise LegacyImportError("idea output references a run that was not imported")
    _mode, data_class, _started = mode_data
    unique_input_ids = tuple(dict.fromkeys(input_event_ids))
    if not unique_input_ids or len(unique_input_ids) > 256:
        raise LegacyImportError("idea output has incomplete or unbounded market provenance")
    placeholders = ",".join("?" for _ in unique_input_ids)
    input_rows = context.target.execute(
        "SELECT event_id, run_id, "
        "coalesce(source_sequence, derived_after_source_sequence) AS sequence "
        f"FROM market_events WHERE event_id IN ({placeholders})",
        unique_input_ids,
    ).fetchall()
    if len(input_rows) != len(unique_input_ids) or any(
        row["run_id"] != run_id or row["sequence"] is None for row in input_rows
    ):
        raise LegacyImportError("idea output market provenance does not match its run")
    ordered_input_ids = tuple(
        row["event_id"]
        for row in sorted(input_rows, key=lambda row: (int(row["sequence"]), row["event_id"]))
    )
    first_input_id = ordered_input_ids[0]
    last_input_id = ordered_input_ids[-1]
    subject = _ensure_symbol_instrument(context, run_id, symbol)
    payload_json = _canonical_json(
        {
            "legacy_fields": dict(sorted(payload.items())),
            "migration_source": {"key": list(source_key), "table": source_table},
            "source_market_event_ids": list(ordered_input_ids),
            "virtual_only": True,
        }
    )
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    ordinal_key = (run_id, output_kind, emitted_at_us)
    ordinal = context.output_ordinals.get(ordinal_key, 0)
    context.output_ordinals[ordinal_key] = ordinal + 1
    output_id = _stable_id("output", source_table, *source_key)
    context.target.execute(
        """
        INSERT INTO idea_outputs(
            output_id, run_id, instance_id, output_kind, subject_instrument_id,
            emitted_at_us, as_of_at_us, valid_until_at_us, direction, strength, confidence,
            horizon_us, first_input_event_id, last_input_event_id, input_watermark,
            input_events_hash, output_ordinal, payload_json, payload_hash, content_hash,
            data_class, authority_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            output_id,
            run_id,
            _stable_id("instance", run_id),
            output_kind,
            subject,
            emitted_at_us,
            emitted_at_us,
            direction,
            strength,
            first_input_id,
            last_input_id,
            last_input_id,
            _sha(ordered_input_ids),
            ordinal,
            payload_json,
            payload_hash,
            _sha(
                {
                    "kind": output_kind,
                    "payload_hash": payload_hash,
                    "run_id": run_id,
                    "source": [source_table, list(source_key)],
                }
            ),
            data_class,
            "unapproved" if output_kind == "proposed_trade" else "recorded",
        ),
    )
    context.target.executemany(
        "INSERT INTO idea_output_inputs(output_id, event_id, input_ordinal) VALUES (?, ?, ?)",
        ((output_id, event_id, ordinal) for ordinal, event_id in enumerate(ordered_input_ids)),
    )
    for leg_number, (instrument_id, action, quantity, price_hint) in enumerate(legs):
        context.target.execute(
            """
            INSERT INTO idea_output_legs(
                output_id, leg_number, instrument_id, action, target, quantity_value,
                notional_value, currency, price_hint
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, 'USD', ?)
            """,
            (
                output_id,
                leg_number,
                instrument_id,
                action,
                "long" if action == "buy" else "short",
                quantity,
                price_hint if price_hint is not None and price_hint > 0 else None,
            ),
        )
    context.target.execute("INSERT INTO idea_output_seals(output_id) VALUES (?)", (output_id,))
    return output_id


def _generic_output_input_events(
    context: _ImportContext,
    *,
    table: str,
    row: sqlite3.Row,
    run_id: str,
    symbol: str,
    emitted_at_us: int,
) -> tuple[str, ...]:
    input_symbol = symbol
    input_at_us = emitted_at_us
    if table == "signal_checkpoint":
        score = context.source.connection.execute(
            "SELECT run_id, symbol, bar_end_utc FROM model_score WHERE id=?",
            (row["model_score_id"],),
        ).fetchone()
        if score is None or str(score["run_id"]) != run_id:
            return ()
        input_symbol = str(score["symbol"])
        try:
            input_at_us = _timestamp_us(score["bar_end_utc"], label="model score input")
        except LegacyImportError:
            return ()
    events = context.underlying_bar_events.get((run_id, input_symbol, input_at_us), set())
    return tuple(events) if len(events) == 1 else ()


def _import_generic_outputs(context: _ImportContext) -> None:
    specifications = (
        ("model_score", "observation", "symbol", "bar_end_utc", "m1_probability"),
        ("signal_episode", "signal", "symbol", "crossing_timestamp_utc", None),
        ("signal_checkpoint", "observation", None, "checkpoint_timestamp_utc", "m1_probability"),
        ("m1c_checkpoint_v0", "observation", "symbol", "bar_end_utc", "probability"),
        (
            "opening_reversal_prediction_v1",
            "signal",
            "stock",
            "signal_timestamp_utc",
            "m1c_probability",
        ),
    )
    for table, output_kind, symbol_column, timestamp_column, strength_column in specifications:
        if not context.source.has_table(table):
            continue
        tracker = context.tracker(table)
        for key, row in context.source.rows(table):
            run_id = str(row["run_id"])
            symbol = str(_row_value(row, symbol_column) or "") if symbol_column else ""
            if not symbol and table == "signal_checkpoint":
                episode = context.source.connection.execute(
                    "SELECT symbol FROM signal_episode WHERE id=?",
                    (row["signal_episode_id"],),
                ).fetchone()
                symbol = str(episode[0]) if episode else ""
            if run_id not in context.run_modes or not symbol:
                tracker.record(key, imported=False, reason="output_provenance_incomplete")
                continue
            try:
                emitted_at_us = _timestamp_us(row[timestamp_column], label=f"{table} output")
            except LegacyImportError:
                tracker.record(key, imported=False, reason="invalid_output_timestamp")
                continue
            strength = _finite_float(_row_value(row, strength_column)) if strength_column else None
            input_event_ids = _generic_output_input_events(
                context,
                table=table,
                row=row,
                run_id=run_id,
                symbol=symbol,
                emitted_at_us=emitted_at_us,
            )
            if not input_event_ids:
                tracker.record(key, imported=False, reason="output_provenance_incomplete")
                continue
            payload = {
                name: _row_value(row, name)
                for name in (
                    "cohort",
                    "eligibility",
                    "frozen_threshold",
                    "m1_probability",
                    "probability",
                    "rejection_reason",
                    "score_label",
                    "status",
                )
                if context.source.has_column(table, name)
            }
            output_id = _insert_output(
                context,
                source_table=table,
                source_key=key,
                run_id=run_id,
                output_kind=cast(Literal["observation", "signal"], output_kind),
                symbol=symbol,
                emitted_at_us=emitted_at_us,
                payload=payload,
                input_event_ids=input_event_ids,
                strength=strength,
            )
            tracker.record(key, imported=True, target_refs=(f"idea_outputs:{output_id}",))


def _shadow_exit_references(
    context: _ImportContext,
    legs: Sequence[tuple[sqlite3.Row, str, str, float, float | None, str, int]],
    marked_at_us: int,
) -> tuple[tuple[str, float], ...]:
    references: list[tuple[str, float]] = []
    for source_leg, _instrument_id, action, _quantity, _entry_price, _event_id, _entry_at in legs:
        events = context.option_events.get(
            (int(source_leg["option_contract_id"]), marked_at_us), set()
        )
        if len(events) != 1:
            return ()
        event_id = next(iter(events))
        event = context.target.execute(
            "SELECT bid_value, ask_value FROM market_events WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if event is None:
            return ()
        exit_price = _finite_float(event["bid_value"] if action == "buy" else event["ask_value"])
        if exit_price is None or exit_price < 0:
            return ()
        references.append((event_id, exit_price))
    return tuple(references)


def _import_shadow(context: _ImportContext) -> None:
    structure_tracker = context.tracker("shadow_structure")
    leg_tracker = context.tracker("shadow_leg")
    valuation_tracker = context.tracker("shadow_horizon_valuation")
    for structure_key, structure in context.source.rows("shadow_structure"):
        run_id = str(structure["run_id"])
        mode_data = context.run_modes.get(run_id)
        source_legs = tuple(
            context.source.connection.execute(
                "SELECT * FROM shadow_leg WHERE shadow_structure_id=? ORDER BY id",
                (structure["id"],),
            )
        )
        source_valuations = tuple(
            context.source.connection.execute(
                "SELECT * FROM shadow_horizon_valuation "
                "WHERE shadow_structure_id=? ORDER BY horizon_minutes, id",
                (structure["id"],),
            )
        )
        valid_legs: list[tuple[sqlite3.Row, str, str, float, float | None, str, int]] = []
        for leg in source_legs:
            instrument_id = context.instrument_by_option.get(int(leg["option_contract_id"]))
            quantity = _finite_float(leg["quantity"])
            leg_role = str(leg["leg_role"]).strip().lower()
            entry_side = str(leg["entry_side"]).strip().lower()
            action = {
                ("long", "ask"): "buy",
                ("short", "bid"): "sell",
            }.get((leg_role, entry_side))
            try:
                quote_at_us = _timestamp_us(leg["quote_timestamp_utc"], label="shadow entry")
            except LegacyImportError:
                continue
            entry_events = context.option_events.get(
                (int(leg["option_contract_id"]), quote_at_us), set()
            )
            if (
                instrument_id is not None
                and quantity is not None
                and quantity > 0
                and len(entry_events) == 1
                and action is not None
            ):
                valid_legs.append(
                    (
                        leg,
                        instrument_id,
                        action,
                        quantity,
                        _finite_float(leg["entry_price"]),
                        next(iter(entry_events)),
                        quote_at_us,
                    )
                )
        if mode_data is None or mode_data[0] != "shadow":
            structure_tracker.record(
                structure_key, imported=False, reason="shadow_run_not_imported"
            )
            for leg in source_legs:
                leg_tracker.record((leg["id"],), imported=False, reason="shadow_parent_omitted")
            for valuation in source_valuations:
                valuation_tracker.record(
                    (valuation["id"],), imported=False, reason="shadow_parent_omitted"
                )
            continue
        if not valid_legs or len(valid_legs) != len(source_legs) or len(valid_legs) > 16:
            structure_tracker.record(
                structure_key, imported=False, reason="shadow_leg_evidence_incomplete"
            )
            for leg in source_legs:
                leg_tracker.record((leg["id"],), imported=False, reason="shadow_parent_omitted")
            for valuation in source_valuations:
                valuation_tracker.record(
                    (valuation["id"],), imported=False, reason="shadow_parent_omitted"
                )
            continue
        opened_at_us = valid_legs[0][6]
        importable_valuations: list[tuple[sqlite3.Row, int, tuple[tuple[str, float], ...]]] = []
        for valuation in source_valuations:
            timestamp_value = (
                valuation["actual_quote_timestamp_utc"] or valuation["target_timestamp_utc"]
            )
            try:
                marked_at_us = _timestamp_us(timestamp_value, label="shadow mark")
            except LegacyImportError:
                valuation_tracker.record(
                    (valuation["id"],), imported=False, reason="invalid_shadow_timestamp"
                )
                continue
            references = _shadow_exit_references(context, valid_legs, marked_at_us)
            if not references:
                valuation_tracker.record(
                    (valuation["id"],),
                    imported=False,
                    reason="shadow_mark_evidence_incomplete",
                )
                continue
            importable_valuations.append((valuation, marked_at_us, references))
        completed = [
            item
            for item in importable_valuations
            if str(item[0]["completeness"]).lower() == "complete"
            and item[0]["actual_quote_timestamp_utc"] is not None
        ]
        invalid = bool(structure["rejection_reason"]) or str(
            structure["completeness"]
        ).lower() not in {
            "complete",
            "completed",
        }
        lifecycle = "invalid" if invalid else ("closed" if completed else "open")
        closing_valuation = (
            max(completed, key=lambda item: (item[1], int(item[0]["id"])))
            if (lifecycle == "closed")
            else None
        )
        closed_at_us = closing_valuation[1] if closing_valuation is not None else None
        output_id = _insert_output(
            context,
            source_table="shadow_structure",
            source_key=structure_key,
            run_id=run_id,
            output_kind="proposed_trade",
            symbol=str(structure["symbol"]),
            emitted_at_us=opened_at_us,
            payload={
                "completeness": structure["completeness"],
                "dte_bucket": structure["dte_bucket"],
                "structure_type": structure["structure_type"],
            },
            input_event_ids=tuple(item[5] for item in valid_legs),
            legs=tuple((item[1], item[2], item[3], item[4]) for item in valid_legs),
        )
        position_id = _stable_id("position", "shadow_structure", *structure_key)
        policy_json = _canonical_json({"source": "legacy_import", "virtual_only": True})
        context.target.execute(
            """
            INSERT INTO shadow_positions(
                position_id, proposed_trade_output_id, run_id, instance_id, opened_at_us,
                closed_at_us, lifecycle, cost_model_id, fill_model_id, currency,
                invalid_reason, data_class, policy_json, policy_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'legacy-recorded-costs', 'legacy-virtual-quotes',
                      'USD', ?, 'shadow_protected', ?, ?)
            """,
            (
                position_id,
                output_id,
                run_id,
                _stable_id("instance", run_id),
                opened_at_us,
                closed_at_us,
                lifecycle,
                str(structure["rejection_reason"]) if invalid else None,
                policy_json,
                hashlib.sha256(policy_json.encode()).hexdigest(),
            ),
        )
        for leg_number, (
            source_leg,
            instrument_id,
            action,
            quantity,
            entry_price,
            event_id,
            _quote_at_us,
        ) in enumerate(valid_legs):
            exit_event_id = (
                closing_valuation[2][leg_number][0] if closing_valuation is not None else None
            )
            exit_price = (
                closing_valuation[2][leg_number][1] if closing_valuation is not None else None
            )
            context.target.execute(
                """
                INSERT INTO shadow_legs(
                    position_id, leg_number, instrument_id, side, quantity,
                    entry_market_event_id, entry_price, exit_market_event_id, exit_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position_id,
                    leg_number,
                    instrument_id,
                    action,
                    quantity,
                    event_id,
                    entry_price,
                    exit_event_id,
                    exit_price,
                ),
            )
            leg_tracker.record(
                (source_leg["id"],),
                imported=True,
                target_refs=(f"shadow_legs:{position_id}:{leg_number}",),
            )
        for valuation, marked_at_us, references in importable_valuations:
            gross_pnl = _finite_float(valuation["gross_pnl"])
            fees = _finite_float(valuation["estimated_fees"]) or 0.0
            payload_json = _canonical_json(
                {
                    "horizon_minutes": valuation["horizon_minutes"],
                    "migration_source": {
                        "key": [valuation["id"]],
                        "table": "shadow_horizon_valuation",
                    },
                    "source_market_event_ids": [item[0] for item in references],
                    "virtual_only": True,
                }
            )
            context.target.execute(
                """
                INSERT INTO shadow_marks(
                    position_id, marked_at_us, gross_value, gross_pnl, net_pnl,
                    return_value, quality_bits, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    position_id,
                    marked_at_us,
                    _finite_float(valuation["exit_credit"]),
                    gross_pnl,
                    gross_pnl - fees if gross_pnl is not None else None,
                    _finite_float(valuation["gross_return_on_debit"]),
                    payload_json,
                ),
            )
            valuation_tracker.record(
                (valuation["id"],),
                imported=True,
                target_refs=(f"shadow_marks:{position_id}:{marked_at_us}",),
            )
        outcome_valuation = (
            closing_valuation
            if closing_valuation is not None
            else (importable_valuations[-1] if importable_valuations else None)
        )
        if outcome_valuation is not None:
            final, outcome_at_us, final_references = outcome_valuation
            gross_pnl = _finite_float(final["gross_pnl"])
            fees = _finite_float(final["estimated_fees"]) or 0.0
            completeness = (
                "complete"
                if lifecycle == "closed" and str(final["completeness"]).lower() == "complete"
                else ("invalid" if lifecycle == "invalid" else "incomplete")
            )
            context.target.execute(
                """
                INSERT INTO shadow_outcomes(
                    position_id, outcome_at_us, reason, gross_pnl, net_pnl, return_value,
                    mfe, mae, completeness, payload_json
                ) VALUES (?, ?, 'legacy_horizon', ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    position_id,
                    outcome_at_us,
                    gross_pnl,
                    gross_pnl - fees if gross_pnl is not None else None,
                    _finite_float(final["gross_return_on_debit"]),
                    completeness,
                    _canonical_json(
                        {
                            "migration_source": {
                                "key": [final["id"]],
                                "table": "shadow_horizon_valuation",
                            },
                            "source_market_event_ids": [item[0] for item in final_references],
                            "virtual_only": True,
                        }
                    ),
                ),
            )
        structure_tracker.record(
            structure_key,
            imported=True,
            target_refs=(f"idea_outputs:{output_id}", f"shadow_positions:{position_id}"),
        )


def _import_diagnostics(context: _ImportContext) -> None:
    if context.source.has_table("operational_incident_v1"):
        tracker = context.tracker("operational_incident_v1")
        for key, row in context.source.rows("operational_incident_v1"):
            run_id = str(row["run_id"])
            if run_id not in context.run_modes:
                tracker.record(key, imported=False, reason="run_not_imported")
                continue
            opened_at_us = _timestamp_us(row["occurred_at_utc"], label="incident")
            resolved = row["resolved_at_utc"]
            if resolved is None:
                incident_id = _stable_id("incident", "active-legacy", run_id)
                context.active_diagnostics[run_id]["incidents"] += 1
                tracker.record(key, imported=True, target_refs=(f"incidents:{incident_id}",))
                continue
            resolved_at_us = _timestamp_us(resolved, label="incident resolution")
            incident_id = _stable_id("incident", "operational_incident_v1", *key)
            severity_raw = str(row["severity"]).lower()
            severity = severity_raw if severity_raw in {"info", "degraded", "fatal"} else "degraded"
            details = _canonical_json(
                {
                    "migration_source": {"key": list(key), "table": "operational_incident_v1"},
                    "source_component": row["component"],
                }
            )
            context.target.execute(
                """
                INSERT INTO incidents(
                    incident_id, run_id, scope, severity, code, plugin_instance_id,
                    subscription_id, opened_at_us, resolved_at_us, details_json
                ) VALUES (?, ?, 'migration', ?, ?, NULL, NULL, ?, ?, ?)
                """,
                (
                    incident_id,
                    run_id,
                    severity,
                    str(row["stable_error_code"])[:128],
                    opened_at_us,
                    resolved_at_us,
                    details,
                ),
            )
            tracker.record(key, imported=True, target_refs=(f"incidents:{incident_id}",))
    if context.source.has_table("gap_incident_v1"):
        tracker = context.tracker("gap_incident_v1")
        for key, row in context.source.rows("gap_incident_v1"):
            run_id = str(row["run_id"])
            if run_id not in context.run_modes:
                tracker.record(key, imported=False, reason="run_not_imported")
                continue
            if row["resolution_timestamp_utc"] is None:
                incident_id = _stable_id("incident", "active-legacy", run_id)
                context.active_diagnostics[run_id]["gaps"] += 1
                tracker.record(key, imported=True, target_refs=(f"incidents:{incident_id}",))
                continue
            gap_id = _stable_id("gap", "gap_incident_v1", *key)
            started_at_us = _timestamp_us(row["start_timestamp_utc"], label="gap start")
            ended_at_us = _timestamp_us(
                row["end_timestamp_utc"] or row["resolution_timestamp_utc"],
                label="gap end",
            )
            resolved_at_us = _timestamp_us(row["resolution_timestamp_utc"], label="gap resolution")
            context.target.execute(
                """
                INSERT INTO gaps(
                    gap_id, run_id, subscription_id, started_at_us, ended_at_us, reason,
                    data_loss_possible, continuity_required, resolved_at_us
                ) VALUES (?, ?, NULL, ?, ?, ?, ?, 1, ?)
                """,
                (
                    gap_id,
                    run_id,
                    started_at_us,
                    ended_at_us,
                    str(row["cause_code"])[:128],
                    1 if str(row["recoverability"]).lower() not in {"recovered", "complete"} else 0,
                    resolved_at_us,
                ),
            )
            tracker.record(key, imported=True, target_refs=(f"gaps:{gap_id}",))
    for run_id, counts in sorted(context.active_diagnostics.items()):
        if not counts:
            continue
        incident_id = _stable_id("incident", "active-legacy", run_id)
        at_us = context.run_modes[run_id][2]
        context.target.execute(
            """
            INSERT INTO incidents(
                incident_id, run_id, scope, severity, code, plugin_instance_id,
                subscription_id, opened_at_us, resolved_at_us, details_json
            ) VALUES (?, ?, 'migration', 'degraded', 'LEGACY_ACTIVE_DIAGNOSTICS_ARCHIVED',
                      NULL, NULL, ?, ?, ?)
            """,
            (
                incident_id,
                run_id,
                at_us,
                at_us,
                _canonical_json({"active_legacy_rows": dict(sorted(counts.items()))}),
            ),
        )


def _reconcile_omitted_tables(context: _ImportContext) -> None:
    special_reasons = {
        "schema_migrations": "schema_metadata_only",
        "evidence_envelope": "superseded_envelope",
        "option_surface_capture": "superseded_capture_wrapper",
    }
    immutable_archive_tokens = (
        "callback_",
        "partition",
        "source_bar",
        "source_capture",
        "transfer",
    )
    for table in context.source.tables:
        if table in context.handled_tables:
            continue
        tracker = context.tracker(table)
        reason = special_reasons.get(table)
        if reason is None:
            reason = (
                "immutable_legacy_archive"
                if any(token in table for token in immutable_archive_tokens)
                else "legacy_runtime_only"
            )
        for key, _row in context.source.rows(table):
            tracker.record(key, imported=False, reason=reason)


def _verify_reconciliation(context: _ImportContext) -> tuple[int, int, int]:
    source_rows = imported_rows = omitted_rows = 0
    for table in context.source.tables:
        tracker = context.trackers[table]
        expected = context.source.count(table)
        if tracker.source_rows != expected:
            raise LegacyImportError(
                f"source table {table} reconciliation mismatch: {tracker.source_rows} != {expected}"
            )
        if tracker.source_rows != tracker.imported_rows + tracker.omitted_rows:
            raise LegacyImportError(f"source table {table} has an unclassified row")
        source_rows += tracker.source_rows
        imported_rows += tracker.imported_rows
        omitted_rows += tracker.omitted_rows
    if source_rows != imported_rows + omitted_rows:
        raise LegacyImportError("source reconciliation totals do not balance")
    return source_rows, imported_rows, omitted_rows


def _logical_database_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    tables = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name != 'migration_manifests' ORDER BY name"
        )
    )
    for table in tables:
        columns = tuple(str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")'))
        row_columns = (
            tuple(column for column in columns if column != "applied_at_us")
            if table == "schema_migrations"
            else columns
        )
        primary = tuple(
            str(row[1])
            for row in sorted(
                connection.execute(f'PRAGMA table_info("{table}")'),
                key=lambda item: int(item[5]) if int(item[5]) else 1_000_000,
            )
            if int(row[5])
        )
        order = primary or columns
        order_sql = ", ".join(f'"{name}"' for name in order)
        selection_sql = ", ".join(f'"{name}"' for name in row_columns)
        digest.update(
            _canonical_bytes(
                {
                    "columns": list(columns),
                    "logical_row_columns": list(row_columns),
                    "table": table,
                }
            )
        )
        digest.update(b"\n")
        for row in connection.execute(
            f'SELECT {selection_sql} FROM "{table}" ORDER BY {order_sql}'
        ):
            digest.update(_canonical_bytes(list(row)))
            digest.update(b"\n")
    return digest.hexdigest()


def _read_only_legacy(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _validate_paths(source: Path, target: Path, report: Path) -> os.stat_result:
    try:
        metadata = source.lstat()
    except FileNotFoundError as error:
        raise LegacyImportError("legacy source database does not exist") from error
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise LegacyImportError("legacy source must be one regular, non-symlink database")
    if source.resolve() in {target.resolve(), report.resolve()}:
        raise LegacyImportError("legacy source and V2 target paths must be distinct")
    if target.exists() or target.is_symlink() or report.exists() or report.is_symlink():
        raise LegacyImportError("legacy import requires a new target and reconciliation path")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise LegacyImportError("V2 target parent must be a real directory")
    _require_no_legacy_sidecars(source)
    return metadata


def _require_no_legacy_sidecars(source: Path) -> None:
    for suffix in ("-journal", "-wal", "-shm"):
        if Path(f"{source}{suffix}").exists():
            raise LegacyImportError(
                "legacy source journal/WAL/SHM remains; stop and checkpoint V1 first"
            )


def _verify_legacy_source(connection: sqlite3.Connection) -> tuple[tuple[str, ...], str]:
    if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
        raise LegacyImportError("legacy source connection is not query-only")
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise LegacyImportError("legacy source quick_check failed")
    if tuple(connection.execute("PRAGMA foreign_key_check")):
        raise LegacyImportError("legacy source contains foreign-key violations")
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if "schema_migrations" not in tables or "prospective_run" not in tables:
        raise LegacyImportError("legacy source is not a frozen prospective database")
    migration_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(schema_migrations)")
    )
    if migration_columns != (
        ("version", "TEXT", 0, 1),
        ("applied_at_utc", "TEXT", 1, 0),
    ):
        raise LegacyImportError("legacy schema migration ledger has unexpected columns")
    names = tuple(
        str(row[0])
        for row in connection.execute("SELECT version FROM schema_migrations ORDER BY rowid")
    )
    expected_names = tuple(name for name, _digest in LEGACY_SCHEMA_DIGESTS[: len(names)])
    if not names or names != expected_names:
        raise LegacyImportError("legacy schema migration history is not an accepted prefix")
    digest = _schema_digest(connection)
    if digest != LEGACY_SCHEMA_DIGESTS[len(names) - 1][1]:
        raise LegacyImportError("legacy schema digest does not match the frozen migration prefix")
    if (
        "recorder_lease" in tables
        and connection.execute("SELECT count(*) FROM recorder_lease").fetchone()[0]
    ):
        raise LegacyImportError("legacy source still has an active recorder lease")
    if (
        "recorder_generation_v1" in tables
        and connection.execute(
            "SELECT count(*) FROM recorder_generation_v1 WHERE stopped_at_utc IS NULL"
        ).fetchone()[0]
    ):
        raise LegacyImportError("legacy source still has an active recorder generation")
    return names, digest


def _write_exclusive(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def import_legacy_database(
    source_database: str | Path,
    target_database: str | Path,
    *,
    started_at_us: int | None = None,
) -> LegacyImportResult:
    """Import one stopped V1 database into a new V2 file; never alter the source."""

    source = Path(source_database)
    target = Path(target_database)
    report = target.with_suffix(target.suffix + ".migration-reconciliation.json")
    before_stat = _validate_paths(source, target, report)
    timestamp = (
        int(datetime.now(UTC).timestamp() * 1_000_000) if started_at_us is None else started_at_us
    )
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise LegacyImportError("migration start time must be a nonnegative integer")
    source_hash = _file_sha256(source)
    database_descriptor, database_name = tempfile.mkstemp(
        prefix=".stocker-v2-import-", suffix=".sqlite3", dir=target.parent
    )
    os.close(database_descriptor)
    temporary_database = Path(database_name)
    temporary_database.unlink()
    report_descriptor, report_name = tempfile.mkstemp(
        prefix=".stocker-v2-import-", suffix=".json", dir=target.parent
    )
    os.close(report_descriptor)
    temporary_report = Path(report_name)
    temporary_report.unlink()
    published_report = False
    try:
        with _read_only_legacy(source) as legacy:
            _names, schema_digest = _verify_legacy_source(legacy)
            initialize_database(temporary_database, applied_at_us=timestamp)
            with connect_v2(temporary_database) as target_connection:
                page_size = int(target_connection.execute("PRAGMA page_size").fetchone()[0])
                maximum_pages = max(1, MAX_TARGET_BYTES // page_size)
                target_connection.execute(f"PRAGMA max_page_count = {maximum_pages}")
                target_connection.execute("BEGIN IMMEDIATE")
                reader = _LegacyReader(legacy)
                context = _ImportContext(
                    source=reader,
                    target=target_connection,
                    trackers={table: _TableTracker(table) for table in reader.tables},
                )
                _import_runs(context)
                _import_instruments(context)
                _import_option_instruments(context)
                _import_market_events(context)
                _project_market_latest(context)
                _install_legacy_idea_instances(context)
                _import_generic_outputs(context)
                _import_shadow(context)
                _import_diagnostics(context)
                _reconcile_omitted_tables(context)
                source_rows, imported_rows, omitted_rows = _verify_reconciliation(context)
                database_digest = _logical_database_digest(target_connection)
                report_core = {
                    "format_version": 1,
                    "importer_version": IMPORTER_VERSION,
                    "source_database_hash": source_hash,
                    "source_schema_digest": schema_digest,
                    "source_row_count": source_rows,
                    "imported_row_count": imported_rows,
                    "omitted_row_count": omitted_rows,
                    "tables": [context.trackers[name].to_dict() for name in reader.tables],
                }
                reconciliation_digest = hashlib.sha256(_canonical_bytes(report_core)).hexdigest()
                target_digest = _sha(
                    {
                        "database_digest": database_digest,
                        "reconciliation_digest": reconciliation_digest,
                    }
                )
                migration_id = _stable_id("migration", source_hash, schema_digest)
                importer_identity = f"{IMPORTER_VERSION}+reconciliation.{reconciliation_digest}"
                target_connection.execute(
                    """
                    INSERT INTO migration_manifests(
                        migration_id, source_database_hash, source_schema_digest,
                        importer_version, started_at_us, completed_at_us, source_row_count,
                        imported_row_count, omitted_row_count, target_digest,
                        verification_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'verified')
                    """,
                    (
                        migration_id,
                        source_hash,
                        schema_digest,
                        importer_identity,
                        timestamp,
                        timestamp,
                        source_rows,
                        imported_rows,
                        omitted_rows,
                        target_digest,
                    ),
                )
                if tuple(target_connection.execute("PRAGMA foreign_key_check")):
                    raise LegacyImportError("V2 import contains foreign-key violations")
                if target_connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise LegacyImportError("V2 import quick_check failed")
                target_connection.commit()
                target_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        verify_database(temporary_database)
        if temporary_database.stat().st_size > MAX_TARGET_BYTES:
            raise LegacyImportError("V2 import exceeds the 8 GiB operational database cap")
        os.chmod(temporary_database, 0o640)
        after_stat = source.lstat()
        after_hash = _file_sha256(source)
        _require_no_legacy_sidecars(source)
        if (
            after_stat.st_dev,
            after_stat.st_ino,
            after_stat.st_size,
            after_stat.st_mtime_ns,
        ) != (
            before_stat.st_dev,
            before_stat.st_ino,
            before_stat.st_size,
            before_stat.st_mtime_ns,
        ) or after_hash != source_hash:
            raise LegacyImportError("legacy source changed during import")
        report_payload = {
            **report_core,
            "database_digest": database_digest,
            "reconciliation_digest": reconciliation_digest,
            "target_digest": target_digest,
            "verification_status": "verified",
        }
        report_bytes = _canonical_bytes(report_payload) + b"\n"
        if len(report_bytes) > MAX_RECONCILIATION_BYTES:
            raise LegacyImportError("migration reconciliation report exceeds its size limit")
        _write_exclusive(temporary_report, report_bytes)
        os.link(temporary_report, report)
        published_report = True
        os.link(temporary_database, target)
        temporary_report.unlink()
        temporary_database.unlink()
        _fsync_directory(target.parent)
        return LegacyImportResult(
            target_path=target,
            reconciliation_path=report,
            migration_id=migration_id,
            source_database_hash=source_hash,
            source_schema_digest=schema_digest,
            source_row_count=source_rows,
            imported_row_count=imported_rows,
            omitted_row_count=omitted_rows,
            target_digest=target_digest,
        )
    except (LegacyImportError, OSError, sqlite3.Error, ValueError, TypeError) as error:
        if published_report and not target.exists():
            report.unlink(missing_ok=True)
        if isinstance(error, LegacyImportError):
            raise
        raise LegacyImportError(str(error)) from error
    finally:
        temporary_database.unlink(missing_ok=True)
        Path(f"{temporary_database}-wal").unlink(missing_ok=True)
        Path(f"{temporary_database}-shm").unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)


__all__ = [
    "IMPORTER_VERSION",
    "LEGACY_SCHEMA_DIGESTS",
    "LegacyImportError",
    "LegacyImportResult",
    "import_legacy_database",
]
