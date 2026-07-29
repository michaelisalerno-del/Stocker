#!/usr/bin/env python3
"""Independent lightweight audit of the cached contract-history probe."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas_market_calendars as market_calendars

PROBE_DIR = Path(__file__).resolve().parent
PRIMARY = PROBE_DIR / "artifacts" / "primary"
NEW_YORK = ZoneInfo("America/New_York")
EXPECTED_SAFETY: dict[str, bool | str] = {
    "research_only": True,
    "options_feasibility_screen": True,
    "options_data_granularity": "end_of_day",
    "options_information_time": "previous_trading_day_close",
    "intraday_option_fill_simulated": False,
    "option_pnl_calculated": False,
    "underlying_movement_outcomes_opened": True,
    "directional_outcomes_primary": False,
    "economic_strategy_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, default=str)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ny_date(value: object) -> date:
    if not isinstance(value, str) or not value:
        raise AssertionError("missing quote observation timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=NEW_YORK)
    return parsed.astimezone(NEW_YORK).date()


def _number(attributes: Mapping[str, Any], name: str) -> float:
    value = attributes.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"selected record lacks numeric {name}")
    number = float(value)
    if not math.isfinite(number):
        raise AssertionError(f"selected record has non-finite {name}")
    return number


def _midpoint(attributes: Mapping[str, Any]) -> float:
    value = attributes.get("midpoint")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        midpoint = float(value)
        if math.isfinite(midpoint):
            return midpoint
    return (_number(attributes, "bid") + _number(attributes, "ask")) / 2.0


def _relative_spread(attributes: Mapping[str, Any]) -> float:
    return (_number(attributes, "ask") - _number(attributes, "bid")) / _midpoint(attributes)


def _assert_feature_close(left: float, right: str, *, tolerance: float = 1e-12) -> float:
    difference = abs(left - float(right))
    if difference > tolerance:
        raise AssertionError(f"feature mismatch: {left} versus {right}")
    return difference


def _previous_nyse_session(signal_date: date) -> date:
    calendar = market_calendars.get_calendar("NYSE")
    sessions = calendar.valid_days(
        start_date=signal_date - timedelta(days=30),
        end_date=signal_date - timedelta(days=1),
        tz="America/New_York",
    )
    if len(sessions) == 0:
        raise AssertionError("independent NYSE calendar found no prior session")
    return cast(date, sessions[-1].date())


def _exact_raw_attributes(
    raw_by_contract: Mapping[str, list[dict[str, Any]]],
    *,
    contract_id: str,
    required: date,
) -> Mapping[str, Any] | None:
    if contract_id not in raw_by_contract:
        raise AssertionError("candidate history required for ranking was not downloaded")
    matches = [
        item
        for item in raw_by_contract[contract_id]
        if str(item.get("id", "")).endswith(required.isoformat())
    ]
    if not matches:
        return None
    if len(matches) != 1 or not isinstance(matches[0].get("attributes"), Mapping):
        raise AssertionError("candidate history has ambiguous exact observation")
    return cast(Mapping[str, Any], matches[0]["attributes"])


def _reconstruct_primary_pair(
    contract_items: list[dict[str, Any]],
    raw_by_contract: Mapping[str, list[dict[str, Any]]],
    *,
    required: date,
    previous_close: float,
) -> tuple[str, str, date, float, tuple[str, ...]]:
    grouped: dict[tuple[date, float], dict[str, list[str]]] = {}
    for item in contract_items:
        attributes = item.get("attributes")
        if not isinstance(attributes, Mapping):
            raise AssertionError("contract discovery row lacks immutable attributes")
        expiration = date.fromisoformat(str(attributes["exp_date"])[:10])
        strike = float(attributes["strike"])
        option_type = str(attributes["type"])
        contract_id = str(attributes["contract"])
        grouped.setdefault((expiration, strike), {"call": [], "put": []})[option_type].append(
            contract_id
        )
    common_groups = [
        (expiration, strike, sides)
        for (expiration, strike), sides in grouped.items()
        if sides["call"] and sides["put"]
    ]
    requested_ids: list[str] = []
    for expiration in sorted({group[0] for group in common_groups}):
        expiry_groups = sorted(
            (group for group in common_groups if group[0] == expiration),
            key=lambda group: (abs(math.log(group[1] / previous_close)), group[1]),
        )
        position = 0
        while position < len(expiry_groups):
            distance = abs(math.log(expiry_groups[position][1] / previous_close))
            tied_groups: list[tuple[date, float, dict[str, list[str]]]] = []
            while position < len(expiry_groups):
                group = expiry_groups[position]
                candidate_distance = abs(math.log(group[1] / previous_close))
                if candidate_distance != distance:
                    break
                tied_groups.append(group)
                position += 1
            candidates: list[tuple[tuple[float | str, ...], str, str, date, float]] = []
            for _, strike, sides in tied_groups:
                calls: dict[str, Mapping[str, Any] | None] = {}
                puts: dict[str, Mapping[str, Any] | None] = {}
                for call_id in sorted(sides["call"]):
                    requested_ids.append(call_id)
                    calls[call_id] = _exact_raw_attributes(
                        raw_by_contract,
                        contract_id=call_id,
                        required=required,
                    )
                for put_id in sorted(sides["put"]):
                    requested_ids.append(put_id)
                    puts[put_id] = _exact_raw_attributes(
                        raw_by_contract,
                        contract_id=put_id,
                        required=required,
                    )
                for call_id, call in calls.items():
                    for put_id, put in puts.items():
                        if call is None or put is None:
                            continue
                        minimum_oi = min(
                            _number(call, "open_interest"), _number(put, "open_interest")
                        )
                        combined_midpoint = _midpoint(call) + _midpoint(put)
                        combined_spread = (
                            _number(call, "ask")
                            - _number(call, "bid")
                            + _number(put, "ask")
                            - _number(put, "bid")
                        ) / combined_midpoint
                        iv_gap = abs(_number(call, "volatility") - _number(put, "volatility"))
                        rank: tuple[float | str, ...] = (
                            -minimum_oi,
                            combined_spread,
                            iv_gap,
                            strike,
                            call_id,
                            put_id,
                        )
                        candidates.append((rank, call_id, put_id, expiration, strike))
            if candidates:
                _, call_id, put_id, selected_expiration, selected_strike = min(
                    candidates, key=lambda candidate: candidate[0]
                )
                return (
                    call_id,
                    put_id,
                    selected_expiration,
                    selected_strike,
                    tuple(requested_ids),
                )
    raise AssertionError("no exact primary pair reconstructs from cached histories")


def main() -> int:
    plan = json.loads((PRIMARY / "contract_history_probe_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (PRIMARY / "contract_history_probe_manifest.json").read_text(encoding="utf-8")
    )
    with (PRIMARY / "contract_history_probe_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        results = list(csv.DictReader(handle))
    if any(plan.get(key) != value for key, value in EXPECTED_SAFETY.items()):
        raise AssertionError("plan safety flags changed")
    if any(manifest.get(key) != value for key, value in EXPECTED_SAFETY.items()):
        raise AssertionError("manifest safety flags changed")
    if plan["symbols"] != ["AAL", "MSTR", "WULF"] or len(results) != 9:
        raise AssertionError("probe scope changed")
    if manifest["status"] != "contract_history_probe_complete":
        raise AssertionError("probe did not complete")
    serialized = json.dumps([plan, manifest], sort_keys=True).casefold()
    if "api_token" in serialized or "authorization" in serialized:
        raise AssertionError("credential-bearing key reached tracked artifacts")

    cache_root = Path(plan["cache_root"]).resolve()
    raw_by_contract: dict[str, list[dict[str, Any]]] = {}
    response_hash_failures = 0
    pagination_failures = 0
    counted_records = 0
    counted_bytes = 0
    payload_by_cache_path: dict[Path, dict[str, Any]] = {}
    for row in manifest["manifest_rows"]:
        cache_path = Path(row["cache_path"]).resolve()
        if not cache_path.is_relative_to(cache_root):
            raise AssertionError("manifest cache path escaped the ignored probe root")
        content = cache_path.read_bytes()
        counted_bytes += len(content)
        if hashlib.sha256(content).hexdigest() != row["response_hash"]:
            response_hash_failures += 1
        payload = json.loads(content)
        payload_by_cache_path[cache_path] = payload
        data = payload.get("data")
        meta = payload.get("meta")
        links = payload.get("links")
        if not isinstance(data, list) or not isinstance(meta, dict) or not isinstance(links, dict):
            raise AssertionError("cached response lacks meta/data/links")
        counted_records += len(data)
        if len(data) != int(row["record_count"]) or int(meta["offset"]) != int(row["offset"]):
            pagination_failures += 1
        next_value = links.get("next")
        if next_value not in {None, ""}:
            values = parse_qs(urlparse(str(next_value)).query).get("page[offset]")
            expected = int(row["offset"]) + len(data)
            if values is None or len(values) != 1 or int(values[0]) != expected:
                pagination_failures += 1
        contract_id = row.get("contract_id")
        if contract_id:
            raw_by_contract.setdefault(str(contract_id), []).extend(data)
    pagination_groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for row in manifest["manifest_rows"]:
        key = (
            row["endpoint"],
            row.get("contract_id"),
            row.get("underlying_symbol"),
            row.get("expiration_from"),
            row.get("expiration_to"),
            row.get("strike_from"),
            row.get("strike_to"),
        )
        pagination_groups.setdefault(key, []).append(row)
    for rows in pagination_groups.values():
        expected_offset = 0
        expected_total: int | None = None
        ordered = sorted(rows, key=lambda row: int(row["offset"]))
        for index, row in enumerate(ordered):
            payload = payload_by_cache_path[Path(row["cache_path"]).resolve()]
            data = payload["data"]
            meta = payload["meta"]
            links = payload["links"]
            if int(row["offset"]) != expected_offset or int(meta["offset"]) != expected_offset:
                pagination_failures += 1
            if meta.get("total") is not None:
                total = int(meta["total"])
                if expected_total is None:
                    expected_total = total
                elif expected_total != total:
                    pagination_failures += 1
            expected_offset += len(data)
            next_value = links.get("next")
            next_offset: int | None = None
            if next_value not in {None, ""}:
                values = parse_qs(urlparse(str(next_value)).query).get("page[offset]")
                if values is not None and len(values) == 1:
                    next_offset = int(values[0])
            if index < len(ordered) - 1:
                if next_offset != expected_offset:
                    pagination_failures += 1
            elif next_offset is not None:
                pagination_failures += 1
        if expected_total is not None and expected_offset != expected_total:
            pagination_failures += 1
    if response_hash_failures or pagination_failures:
        raise AssertionError("cached-response integrity or pagination failed")
    if counted_records != int(manifest["raw_records"]):
        raise AssertionError("raw record count does not reconstruct")
    if counted_bytes != int(manifest["download_bytes"]):
        raise AssertionError("download byte count does not reconstruct")

    maximum_feature_difference = 0.0
    selected_contract_mismatches = 0
    exact_date_targets = 0
    valid_pairs = 0
    previous_session_mismatches = 0
    pair_ranking_mismatches = 0
    history_request_sequence_mismatches = 0
    attributed_history_contracts: set[str] = set()
    discovery_rows = [
        row for row in manifest["manifest_rows"] if row["endpoint"].endswith("/contracts")
    ]
    for result in results:
        required = date.fromisoformat(result["required_options_date"])
        signal = date.fromisoformat(result["signal_date"])
        if required >= signal or required > date(2025, 8, 21):
            raise AssertionError("chronology or protected boundary failed")
        if _previous_nyse_session(signal) != required:
            previous_session_mismatches += 1
        expected_expiration_from = (required + timedelta(days=7)).isoformat()
        discovery = [
            row
            for row in discovery_rows
            if row["underlying_symbol"] == result["symbol"]
            and row["expiration_from"] == expected_expiration_from
        ]
        if sum(int(row["record_count"]) for row in discovery) != int(
            result["contracts_discovered"]
        ):
            raise AssertionError("contract discovery count does not reconstruct")
        contract_items = [
            item
            for row in discovery
            for item in payload_by_cache_path[Path(row["cache_path"]).resolve()]["data"]
        ]
        (
            expected_call,
            expected_put,
            expected_expiry,
            expected_strike,
            expected_history_sequence,
        ) = _reconstruct_primary_pair(
            contract_items,
            raw_by_contract,
            required=required,
            previous_close=float(result["previous_close_underlying_price"]),
        )
        recorded_history_sequence = tuple(
            value for value in result["history_contract_ids_requested"].split(";") if value
        )
        attributed_history_contracts.update(recorded_history_sequence)
        if recorded_history_sequence != expected_history_sequence or len(
            recorded_history_sequence
        ) != int(result["history_contracts_requested"]):
            history_request_sequence_mismatches += 1
        if (
            expected_call != result["call_contract_id"]
            or expected_put != result["put_contract_id"]
            or expected_expiry != date.fromisoformat(result["selected_expiry"])
            or abs(expected_strike - float(result["selected_strike"])) > 1e-12
        ):
            pair_ranking_mismatches += 1
        selected_records: dict[str, Mapping[str, Any]] = {}
        for side in ("call", "put"):
            contract_id = result[f"{side}_contract_id"]
            candidates = [
                item
                for item in raw_by_contract.get(contract_id, [])
                if str(item.get("id", "")).endswith(required.isoformat())
            ]
            if len(candidates) != 1:
                selected_contract_mismatches += 1
                continue
            item = candidates[0]
            attributes = item.get("attributes")
            if not isinstance(attributes, Mapping):
                raise AssertionError("selected raw row lacks attributes")
            if attributes.get("contract") != contract_id or attributes.get("type") != side:
                selected_contract_mismatches += 1
            if _ny_date(attributes.get("bid_date")) != required:
                raise AssertionError("bid observation date is not exact")
            if _ny_date(attributes.get("ask_date")) != required:
                raise AssertionError("ask observation date is not exact")
            selected_records[side] = attributes
        if len(selected_records) != 2:
            continue
        exact_date_targets += 1
        call = selected_records["call"]
        put = selected_records["put"]
        expiration = date.fromisoformat(str(call["exp_date"])[:10])
        if expiration != date.fromisoformat(str(put["exp_date"])[:10]):
            raise AssertionError("selected call and put expiries differ")
        if float(call["strike"]) != float(put["strike"]):
            raise AssertionError("selected call and put strikes differ")
        call_spread = _relative_spread(call)
        put_spread = _relative_spread(put)
        if result["pair_available"] == "True":
            valid_pairs += 1
            if result["reason"] != "selected":
                raise AssertionError("available pair reason changed")
            dte = (expiration - required).days
            atm_iv = (_number(call, "volatility") + _number(put, "volatility")) / 2.0
            combined_oi = int(_number(call, "open_interest") + _number(put, "open_interest"))
            maximum_feature_difference = max(
                maximum_feature_difference,
                _assert_feature_close(float(dte), result["front_dte"]),
                _assert_feature_close(atm_iv, result["atm_iv"]),
                _assert_feature_close(call_spread, result["call_relative_spread"]),
                _assert_feature_close(put_spread, result["put_relative_spread"]),
                _assert_feature_close(float(combined_oi), result["combined_open_interest"]),
            )
            if not 7 <= dte <= 45 or call_spread > 1.0 or put_spread > 1.0:
                raise AssertionError("available pair violated frozen DTE/spread quality")
            for attributes in (call, put):
                if _number(attributes, "open_interest") < 10:
                    raise AssertionError("available pair violated open-interest quality")
                if not 0.005 <= _number(attributes, "volatility") <= 5.0:
                    raise AssertionError("available pair violated IV quality")
        elif result["reason"] == "selected_pair_call_relative_spread_above_1":
            if call_spread <= 1.0:
                raise AssertionError("unavailable spread reason does not reconstruct")
        else:
            raise AssertionError("unexpected unavailable-pair reason")
    if selected_contract_mismatches:
        raise AssertionError("selected contract reconstruction failed")
    unattributed_history_contracts = set(raw_by_contract) - attributed_history_contracts
    if (
        previous_session_mismatches
        or pair_ranking_mismatches
        or history_request_sequence_mismatches
        or unattributed_history_contracts
    ):
        raise AssertionError("previous-session or frozen pair ranking reconstruction failed")
    if exact_date_targets != int(manifest["exact_date_targets"]):
        raise AssertionError("exact-date target count changed")
    if valid_pairs != int(manifest["valid_primary_pairs"]):
        raise AssertionError("valid-pair count changed")

    audit = {
        "passed": True,
        "status": "contract_history_probe_audited",
        "safety_flags_verified": True,
        "credential_redaction_verified": True,
        "response_hash_failures": response_hash_failures,
        "pagination_failures": pagination_failures,
        "pagination_groups_verified": len(pagination_groups),
        "raw_records_reconstructed": counted_records,
        "download_bytes_reconstructed": counted_bytes,
        "exact_date_targets_reconstructed": exact_date_targets,
        "valid_primary_pairs_reconstructed": valid_pairs,
        "selected_contract_mismatches": selected_contract_mismatches,
        "previous_session_mismatches": previous_session_mismatches,
        "pair_ranking_mismatches": pair_ranking_mismatches,
        "history_request_sequence_mismatches": history_request_sequence_mismatches,
        "unattributed_history_contracts": len(unattributed_history_contracts),
        "nearest_expiry_verified": True,
        "atm_common_strike_verified": True,
        "tie_break_order_verified": True,
        "no_quality_fallback_verified": (
            history_request_sequence_mismatches == 0 and not unattributed_history_contracts
        ),
        "maximum_option_feature_difference": maximum_feature_difference,
        "same_day_or_future_join_mismatches": 0,
        "protected_boundary_mismatches": 0,
        "network_requests_made": 0,
    }
    determinism = {
        "passed": True,
        "status": "cached_probe_reconstruction_match",
        "redownloaded": False,
        "selected_contract_mismatches": selected_contract_mismatches,
        "joined_row_mismatches": 0,
        "maximum_option_feature_difference": maximum_feature_difference,
        "maximum_probability_difference": None,
        "maximum_movement_difference": None,
        "models_refit": False,
        "reason_models_not_refit": "retrieval_feasibility_probe_only",
    }
    _write_json(PRIMARY / "lightweight_audit.json", audit)
    _write_json(PRIMARY / "determinism_check.json", determinism)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
