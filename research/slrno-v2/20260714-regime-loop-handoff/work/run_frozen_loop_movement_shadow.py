"""Offline prospective shadow harness for the frozen loop-movement model.

Commands are deliberately separated:

* ``init`` archives and verifies the frozen lineage without reading new bars.
* ``issue`` emits prediction-only, hash-chained batches before outcomes exist.
* ``status`` reads prediction metadata only and never calculates performance.
* ``evaluate`` refuses provider access until the frozen support prefix exists.

Research only.  There is no broker, order, position, P&L, strategy, runtime, or
deployment integration in this module.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import frozen_loop_movement_shadow_core as core


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CONTRACT_PATH = HERE / "contracts/20260710-frozen-loop-movement-shadow-v1.json"
MANIFEST_PATH = HERE / "contracts/20260710-frozen-loop-movement-shadow-v1-manifest.json"
DEFAULT_RUNTIME = HERE / "shadow_validation/frozen_loop_movement_shadow_v1"
DEFAULT_PROVIDER_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/"
    "instrument_type=stock"
)
DEFAULT_PRICE_ARTIFACT_ROOT = Path(
    "/private/tmp/stocker_frozen_loop_price_consequence_20260710"
)
DEFAULT_STATE_ARTIFACT_ROOT = Path(
    "/private/tmp/stocker_causal_semimarkov_regime_loops_20260710"
)
SEED = 20260710
PREDICTION_FORBIDDEN_FRAGMENTS = (
    "direction",
    "signed_return",
    "actual",
    "observed",
    "outcome",
    "pnl",
    "position",
    "order",
    "cost",
    "spread",
    "slippage",
)


class SupportNotMet(RuntimeError):
    """Raised before any provider read when the prediction ledger is too small."""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(core.safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return timestamp.tz_convert("UTC")


def runtime_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "bundle": root / "frozen_bundle",
        "contract": root / "contract.json",
        "manifest": root / "freeze_manifest.json",
        "metadata": root / "runtime_metadata.json",
        "ledger": root / "prediction_ledger.jsonl",
        "predictions": root / "prediction_batches",
        "evaluation": root / "evaluation",
    }


def safety_payload() -> dict[str, Any]:
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "economic_edge_claim": False,
        "deployment_enabled": False,
    }


def root_overrides(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Path]:
    defaults = {name: Path(value) for name, value in manifest["roots"].items()}
    mapping = {
        "lineage": getattr(args, "lineage_root", None),
        "state_artifact": getattr(args, "state_artifact_root", None),
        "path_artifact": getattr(args, "path_artifact_root", None),
        "price_artifact": getattr(args, "price_artifact_root", None),
        "shadow_workspace": getattr(args, "shadow_workspace_root", None),
    }
    for name, value in mapping.items():
        if value is not None:
            defaults[name] = Path(value)
    return defaults


def verify_manifest_inputs(
    manifest: dict[str, Any], roots: dict[str, Path]
) -> list[dict[str, Any]]:
    expected_contract = manifest["contract"]
    if core.sha256_file(CONTRACT_PATH) != expected_contract["sha256"]:
        raise AssertionError("prospective contract hash drifted")
    verified = []
    for item in manifest["files"]:
        source = roots[item["root_role"]] / item["relative_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        observed = core.sha256_file(source)
        if observed != item["sha256"]:
            raise AssertionError(f"frozen source hash drifted: {item['name']}")
        verified.append(
            {
                "name": item["name"],
                "source": source,
                "sha256": observed,
                "bundle_path": Path(item["bundle_path"]),
            }
        )
    return verified


def verify_frozen_semantics(bundle: Path) -> dict[str, Any]:
    path_gates = read_json(bundle / "artifacts/path/gates.json")
    price_gates = read_json(bundle / "artifacts/price/gates.json")
    feature_manifest = read_json(bundle / "artifacts/price/feature_manifest.json")
    independent_audit = read_json(
        bundle / "artifacts/price/independent_artifact_audit.json"
    )
    checks = {
        "history_path_retained": path_gates.get("history_retained") is True,
        "raw_context_path_rejected": path_gates.get("context_retained") is False,
        "movement_retained": price_gates.get("movement_consequence_retained") is True,
        "absolute_movement_retained": price_gates.get("absolute_movement_retained") is True,
        "range_movement_retained": price_gates.get("range_movement_retained") is True,
        "direction_rejected": price_gates.get("directional_consequence_retained") is False,
        "economic_edge_not_claimed": price_gates.get("economic_edge_claim") is False,
        "independent_audit_passed": independent_audit.get("all_passed") is True,
        "research_only": feature_manifest.get("research_only") is True,
        "live_ordering_disabled": feature_manifest.get("live_ordering_enabled") is False,
        "order_placement_disabled": feature_manifest.get("order_placement") == "disabled",
        "volume_not_direct_movement_input": feature_manifest.get("volume_label")
        == "historical_volume_not_used",
        "movement_horizons_frozen": feature_manifest.get("horizons") == [6, 12, 24],
        "movement_representations_frozen": feature_manifest.get("representations")
        == ["state_context", "raw_history", "loop_scores"],
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError(f"frozen semantic checks failed: {failed}")
    return checks


def init_runtime(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    roots = root_overrides(args, manifest)
    verified = verify_manifest_inputs(manifest, roots)
    paths = runtime_paths(Path(args.runtime_root))
    if paths["metadata"].is_file():
        return verify_runtime(paths["root"])
    if paths["root"].exists() and any(paths["root"].iterdir()):
        raise AssertionError(f"runtime root is non-empty: {paths['root']}")
    paths["bundle"].mkdir(parents=True, exist_ok=True)
    paths["predictions"].mkdir(parents=True, exist_ok=True)
    paths["evaluation"].mkdir(parents=True, exist_ok=True)
    for item in verified:
        destination = paths["bundle"] / item["bundle_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], destination)
        if core.sha256_file(destination) != item["sha256"]:
            raise AssertionError(f"bundle copy failed: {item['name']}")
    shutil.copy2(CONTRACT_PATH, paths["contract"])
    shutil.copy2(MANIFEST_PATH, paths["manifest"])
    paths["ledger"].write_text("")
    semantic_checks = verify_frozen_semantics(paths["bundle"])
    metadata = {
        "contract_id": read_json(CONTRACT_PATH)["contract_id"],
        "initialized_at_utc": pd.Timestamp.now(tz="UTC"),
        "contract_sha256": core.sha256_file(paths["contract"]),
        "freeze_manifest_sha256": core.sha256_file(paths["manifest"]),
        "bundle_file_count": len(verified),
        "bundle_files": {
            item["name"]: {
                "path": str(item["bundle_path"]),
                "sha256": item["sha256"],
            }
            for item in verified
        },
        "semantic_checks": semantic_checks,
        "outcomes_opened": False,
        **safety_payload(),
    }
    write_json_atomic(paths["metadata"], metadata)
    return verify_runtime(paths["root"])


def verify_runtime(root: Path) -> dict[str, Any]:
    paths = runtime_paths(root)
    if not paths["metadata"].is_file():
        raise FileNotFoundError(f"run init first: {paths['metadata']}")
    metadata = read_json(paths["metadata"])
    if core.sha256_file(paths["contract"]) != metadata["contract_sha256"]:
        raise AssertionError("runtime contract drifted")
    if core.sha256_file(paths["manifest"]) != metadata["freeze_manifest_sha256"]:
        raise AssertionError("runtime freeze manifest drifted")
    manifest = read_json(paths["manifest"])
    for item in manifest["files"]:
        bundled = paths["bundle"] / item["bundle_path"]
        if not bundled.is_file() or core.sha256_file(bundled) != item["sha256"]:
            raise AssertionError(f"runtime bundle drifted: {item['name']}")
    semantic_checks = verify_frozen_semantics(paths["bundle"])
    return {
        "runtime_root": str(root),
        "contract_id": metadata["contract_id"],
        "contract_sha256": metadata["contract_sha256"],
        "freeze_manifest_sha256": metadata["freeze_manifest_sha256"],
        "bundle_file_count": metadata["bundle_file_count"],
        "semantic_checks": semantic_checks,
        **safety_payload(),
    }


def load_frozen(root: Path) -> dict[str, Any]:
    verify_runtime(root)
    bundle = runtime_paths(root)["bundle"]
    preprocessing = pd.read_csv(
        bundle / "artifacts/state/frozen_emission_preprocessing.csv"
    )
    state_parameters = dict(
        np.load(bundle / "artifacts/state/frozen_semimarkov_parameters.npz")
    )
    cycles = core.load_cycles(
        bundle / "artifacts/state/fixed_cycle_shuffled_nulls.csv"
    )
    path_parameters = dict(np.load(bundle / "artifacts/path/model_parameters.npz"))
    feature_manifest = read_json(bundle / "artifacts/price/feature_manifest.json")
    outcome_parameters = dict(
        np.load(bundle / "artifacts/price/outcome_model_parameters.npz")
    )
    return {
        "preprocessing": preprocessing,
        "state_parameters": state_parameters,
        "cycles": cycles,
        "path_parameters": path_parameters,
        "feature_manifest": feature_manifest,
        "outcome_parameters": outcome_parameters,
    }


def read_ledger(root: Path, verify_batches: bool = False) -> list[dict[str, Any]]:
    paths = runtime_paths(root)
    if not paths["ledger"].is_file():
        raise FileNotFoundError(paths["ledger"])
    records = []
    previous = None
    for line_number, line in enumerate(paths["ledger"].read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        stored_hash = record.pop("record_sha256")
        expected_hash = core.sha256_bytes(core.canonical_json_bytes(record))
        if stored_hash != expected_hash:
            raise AssertionError(f"ledger record hash mismatch at line {line_number}")
        if record.get("previous_record_sha256") != previous:
            raise AssertionError(f"ledger chain mismatch at line {line_number}")
        record["record_sha256"] = stored_hash
        if int(record["sequence"]) != len(records) + 1:
            raise AssertionError("ledger sequence is not contiguous")
        if verify_batches:
            batch = paths["root"] / record["batch_file"]
            if batch.parent.resolve() != paths["predictions"].resolve():
                raise AssertionError("ledger batch escaped prediction directory")
            if not batch.is_file() or core.sha256_file(batch) != record["batch_sha256"]:
                raise AssertionError(f"prediction batch drifted: {batch}")
        records.append(record)
        previous = stored_hash
    return records


def load_prediction_batches(
    root: Path, through_sequence: int | None = None
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    records = read_ledger(root, verify_batches=True)
    if through_sequence is not None:
        records = [record for record in records if record["sequence"] <= through_sequence]
    frames = []
    for record in records:
        batch = pd.read_parquet(root / record["batch_file"])
        if len(batch) != int(record["anchor_count"]):
            raise AssertionError("ledger/batch anchor count mismatch")
        if batch["prediction_id"].duplicated().any():
            raise AssertionError("duplicate prediction ID inside a batch")
        if set(batch["contract_id"].astype(str)) != {record["contract_id"]}:
            raise AssertionError("batch contract ID mismatch")
        frames.append(batch)
    if not frames:
        return records, pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True)
    if predictions["prediction_id"].duplicated().any():
        raise AssertionError("duplicate prediction ID across batches")
    cumulative = records[-1]["cumulative_support"]
    observed = {
        "issued_anchors": len(predictions),
        "session_dates": sorted(predictions["session_date"].astype(str).unique()),
        "symbols": sorted(predictions["symbol_norm"].astype(str).unique()),
        "calendar_quarters": sorted(predictions["quarter"].astype(str).unique()),
        "states": sorted(predictions["state"].astype(int).unique()),
    }
    if observed != cumulative:
        raise AssertionError("ledger cumulative support disagrees with prediction batches")
    return records, predictions


def support_from_cumulative(
    cumulative: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    required = contract["cohort_close_rule"]
    summary = {
        "issued_anchors": int(cumulative["issued_anchors"]),
        "distinct_session_dates": len(cumulative["session_dates"]),
        "distinct_symbols": len(cumulative["symbols"]),
        "distinct_calendar_quarters": len(cumulative["calendar_quarters"]),
        "states": [int(value) for value in cumulative["states"]],
    }
    gates = {
        "issued_anchors": summary["issued_anchors"]
        >= int(required["minimum_issued_anchors"]),
        "session_dates": summary["distinct_session_dates"]
        >= int(required["minimum_distinct_session_dates"]),
        "symbols": summary["distinct_symbols"]
        >= int(required["minimum_distinct_symbols"]),
        "calendar_quarters": summary["distinct_calendar_quarters"]
        >= int(required["minimum_distinct_calendar_quarters"]),
        "states": summary["states"] == list(required["required_states"]),
    }
    summary["gates"] = gates
    summary["pass"] = bool(all(gates.values()))
    return summary


def support_summary(predictions: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    required = contract["cohort_close_rule"]
    states = sorted(predictions["state"].astype(int).unique()) if len(predictions) else []
    summary = {
        "issued_anchors": int(len(predictions)),
        "distinct_session_dates": int(predictions["session_date"].nunique())
        if len(predictions)
        else 0,
        "distinct_symbols": int(predictions["symbol_norm"].nunique())
        if len(predictions)
        else 0,
        "distinct_calendar_quarters": int(predictions["quarter"].nunique())
        if len(predictions)
        else 0,
        "states": states,
    }
    gates = {
        "issued_anchors": summary["issued_anchors"]
        >= int(required["minimum_issued_anchors"]),
        "session_dates": summary["distinct_session_dates"]
        >= int(required["minimum_distinct_session_dates"]),
        "symbols": summary["distinct_symbols"]
        >= int(required["minimum_distinct_symbols"]),
        "calendar_quarters": summary["distinct_calendar_quarters"]
        >= int(required["minimum_distinct_calendar_quarters"]),
        "states": states == list(required["required_states"]),
    }
    summary["gates"] = gates
    summary["pass"] = bool(all(gates.values()))
    return summary


def first_closing_prefix(
    root: Path, contract: dict[str, Any]
) -> tuple[int | None, dict[str, Any]]:
    records = read_ledger(root, verify_batches=False)
    if not records:
        return None, support_summary(pd.DataFrame(), contract)
    last_summary: dict[str, Any] | None = None
    for record in records:
        last_summary = support_from_cumulative(record["cumulative_support"], contract)
        if last_summary["pass"]:
            return int(record["sequence"]), last_summary
    assert last_summary is not None
    return None, last_summary


def status(root: Path) -> dict[str, Any]:
    verify_runtime(root)
    paths = runtime_paths(root)
    contract = read_json(paths["contract"])
    records = read_ledger(root, verify_batches=False)
    closing_sequence, support = first_closing_prefix(root, contract)
    closing_record = (
        next(record for record in records if record["sequence"] == closing_sequence)
        if closing_sequence is not None
        else None
    )
    return {
        "contract_id": contract["contract_id"],
        "eligible_session_rule": contract["eligible_session_rule"],
        "ledger_batches": len(records),
        "support": support,
        "cohort_closed": closing_sequence is not None,
        "closing_sequence": closing_sequence,
        "closing_record_sha256": closing_record["record_sha256"]
        if closing_record
        else None,
        "outcome_data_read": False,
        "performance_metrics_calculated": False,
        "ledger_chain_verified": True,
        "all_batch_hashes_verified": False,
        **safety_payload(),
    }


def append_record(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    records = read_ledger(root, verify_batches=False)
    record = dict(record)
    batch = root / record["batch_file"]
    if batch.parent.resolve() != runtime_paths(root)["predictions"].resolve():
        raise AssertionError("new ledger batch escaped prediction directory")
    if not batch.is_file() or core.sha256_file(batch) != record["batch_sha256"]:
        raise AssertionError("new prediction batch hash mismatch")
    record["sequence"] = len(records) + 1
    record["previous_record_sha256"] = (
        records[-1]["record_sha256"] if records else None
    )
    prior = records[-1]["cumulative_support"] if records else {
        "issued_anchors": 0,
        "session_dates": [],
        "symbols": [],
        "calendar_quarters": [],
        "states": [],
    }
    session_dates = set(str(value) for value in prior["session_dates"])
    session_dates.add(str(record["session_date"]))
    symbols = set(str(value) for value in prior["symbols"])
    symbols.update(str(value) for value in record["symbols"])
    states = set(int(value) for value in prior["states"])
    states.update(int(value) for value in record["states"])
    date = pd.Timestamp(record["session_date"])
    quarters = set(str(value) for value in prior["calendar_quarters"])
    quarters.add(f"{date.year}_q{date.quarter}")
    record["cumulative_support"] = {
        "issued_anchors": int(prior["issued_anchors"]) + int(record["anchor_count"]),
        "session_dates": sorted(session_dates),
        "symbols": sorted(symbols),
        "calendar_quarters": sorted(quarters),
        "states": sorted(states),
    }
    record["record_sha256"] = core.sha256_bytes(core.canonical_json_bytes(record))
    with runtime_paths(root)["ledger"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(core.safe(record), sort_keys=True) + "\n")
        handle.flush()
    read_ledger(root, verify_batches=False)
    return record


def validate_issue_time(as_of: pd.Timestamp, now: pd.Timestamp, contract: dict[str, Any]) -> str:
    if as_of.second or as_of.microsecond or as_of.minute % 5:
        raise AssertionError("as-of must be an exact five-minute provider timestamp")
    local = as_of.tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    if minute < 570 or minute >= 960:
        raise AssertionError("as-of is outside the New York regular session")
    completed_at = as_of + pd.Timedelta(
        minutes=int(contract["prediction_issue_protocol"]["completed_bar_delay_minutes"])
    )
    first_outcome_at = as_of + pd.Timedelta(minutes=30)
    if now < completed_at:
        raise AssertionError("anchor bar is not complete yet")
    if now >= first_outcome_at:
        raise AssertionError("six-bar outcome may already be available; prediction rejected")
    session_date = local.strftime("%Y-%m-%d")
    cutoff = contract["eligible_session_rule"].split(">", 1)[1].strip()
    if session_date <= cutoff:
        raise AssertionError("session is not genuinely post-freeze")
    return session_date


def issue_predictions(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.runtime_root)
    verify_runtime(root)
    current_status = status(root)
    if current_status["cohort_closed"]:
        raise AssertionError("primary cohort is already closed; further issuance is excluded")
    contract = read_json(runtime_paths(root)["contract"])
    as_of = parse_utc(args.as_of)
    command_started_at = pd.Timestamp.now(tz="UTC")
    session_date = validate_issue_time(as_of, command_started_at, contract)
    provider_root = Path(args.provider_root)
    frozen = load_frozen(root)
    symbols = list(contract["data"]["frozen_stock_universe"])
    panel, panel_audit = core.prepare_causal_panel(
        symbols,
        provider_root,
        as_of,
        int(contract["data"]["minimum_symbols_present_at_issue"]),
    )
    runs = core.make_session_runs(
        panel,
        frozen["preprocessing"],
        frozen["state_parameters"],
        frozen["cycles"],
        frozen["path_parameters"],
    )
    active = set(panel_audit["active_symbols"])
    anchors = runs.loc[
        runs["start_timestamp"].eq(as_of)
        & runs["symbol_norm"].isin(active)
        & runs["bar_index_in_session"].le(
            int(contract["frozen_model"]["maximum_anchor_bar_ordinal_zero_based"])
        )
    ].copy()
    if anchors.empty:
        return {
            "contract_id": contract["contract_id"],
            "provider_as_of_utc": as_of,
            "command_started_at_utc": command_started_at,
            "session_date": session_date,
            "anchor_count": 0,
            "reason": "no frozen-state run began at the exact as-of bar",
            "outcomes_opened": False,
            **safety_payload(),
        }
    anchors = core.movement_predictions(
        anchors.reset_index(drop=True),
        frozen["feature_manifest"],
        frozen["outcome_parameters"],
    )
    paths = runtime_paths(root)
    issued_at = pd.Timestamp.now(tz="UTC")
    validate_issue_time(as_of, issued_at, contract)
    contract_sha = core.sha256_file(paths["contract"])
    manifest_sha = core.sha256_file(paths["manifest"])
    anchors["contract_id"] = contract["contract_id"]
    anchors["issued_at_utc"] = issued_at
    anchors["provider_as_of_utc"] = as_of
    anchors["causal_input_sha256"] = panel_audit["causal_input_sha256"]
    anchors["contract_sha256"] = contract_sha
    anchors["freeze_manifest_sha256"] = manifest_sha
    anchors["research_only"] = True
    anchors["live_ordering_enabled"] = False
    anchors["order_placement"] = "disabled"
    anchors["prediction_id"] = [
        core.sha256_bytes(
            f"{contract['contract_id']}|{symbol}|{timestamp.isoformat()}".encode("utf-8")
        )
        for symbol, timestamp in zip(
            anchors["symbol_norm"].astype(str),
            pd.to_datetime(anchors["start_timestamp"], utc=True),
            strict=True,
        )
    ]
    prediction_columns = [
        column
        for column in anchors.columns
        if "_prediction_" in column
    ]
    if len(prediction_columns) != 12:
        raise AssertionError("expected twelve frozen movement prediction columns")
    forbidden = [
        column
        for column in anchors.columns
        if any(fragment in column.lower() for fragment in PREDICTION_FORBIDDEN_FRAGMENTS)
        and column not in {"order_placement", "live_ordering_enabled"}
    ]
    if forbidden:
        raise AssertionError(f"forbidden prediction-batch columns: {forbidden}")
    prior_records = read_ledger(root, verify_batches=False)
    if any(
        parse_utc(record["provider_as_of_utc"]) == as_of for record in prior_records
    ):
        raise AssertionError("the exact provider as-of timestamp is already sealed")
    sequence = len(prior_records) + 1
    stamp = as_of.strftime("%Y%m%dT%H%M%SZ")
    relative = Path("prediction_batches") / f"batch_{sequence:06d}_{stamp}.parquet"
    destination = root / relative
    temporary = destination.with_name(f".{destination.name}.tmp")
    anchors = anchors.sort_values("symbol_norm", kind="mergesort").reset_index(drop=True)
    anchors.to_parquet(temporary, index=False)
    temporary.replace(destination)
    batch_sha = core.sha256_file(destination)
    record = append_record(
        root,
        {
            "contract_id": contract["contract_id"],
            "contract_sha256": contract_sha,
            "freeze_manifest_sha256": manifest_sha,
            "issued_at_utc": issued_at,
            "provider_as_of_utc": as_of,
            "session_date": session_date,
            "batch_file": str(relative),
            "batch_sha256": batch_sha,
            "anchor_count": len(anchors),
            "symbols": sorted(anchors["symbol_norm"].astype(str).unique()),
            "states": sorted(anchors["state"].astype(int).unique()),
            "causal_input_sha256": panel_audit["causal_input_sha256"],
            "outcomes_opened": False,
            **safety_payload(),
        },
    )
    return {
        "record": record,
        "status_after_issue": status(root),
        "outcomes_opened": False,
        **safety_payload(),
    }


def verify_frozen_inference(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.runtime_root)
    frozen = load_frozen(root)
    price_root = Path(args.price_artifact_root)
    loop_columns = list(frozen["feature_manifest"]["loop_score_columns"])
    anchor_columns = [
        "anchor_id",
        "state",
        "previous_state_1",
        "previous_state_2",
        *core.NUMERIC_CONTROLS,
        *loop_columns,
    ]
    anchors = pd.read_parquet(
        price_root / "anchor_panel_2025.parquet", columns=anchor_columns
    )
    expected_columns = ["anchor_id"] + [
        f"{representation}__{target}_prediction_{horizon}"
        for representation in core.REPRESENTATIONS
        for target in core.MOVEMENT_TARGETS
        for horizon in core.HORIZONS
    ]
    expected = pd.read_parquet(
        price_root / "price_predictions_2025.parquet", columns=expected_columns
    )
    recomputed_loops = core.add_loop_scores(
        anchors.drop(columns=loop_columns), frozen["cycles"], frozen["path_parameters"]
    )
    loop_error = float(
        np.max(
            np.abs(
                recomputed_loops[loop_columns].to_numpy(dtype=float)
                - anchors[loop_columns].to_numpy(dtype=float)
            )
        )
    )
    reproduced = core.movement_predictions(
        recomputed_loops, frozen["feature_manifest"], frozen["outcome_parameters"]
    )
    merged = reproduced.merge(expected, on="anchor_id", suffixes=("__new", "__stored"))
    prediction_errors = {}
    for column in expected_columns[1:]:
        prediction_errors[column] = float(
            np.max(
                np.abs(
                    merged[f"{column}__new"].to_numpy(dtype=float)
                    - merged[f"{column}__stored"].to_numpy(dtype=float)
                )
            )
        )
    maximum_prediction_error = max(prediction_errors.values())
    result = {
        "rows": len(anchors),
        "maximum_loop_score_error": loop_error,
        "maximum_movement_prediction_error": maximum_prediction_error,
        "pass": bool(loop_error <= 1e-12 and maximum_prediction_error <= 1e-5),
        "outcome_columns_read": False,
        "direction_columns_read": False,
        "signed_return_columns_read": False,
        **safety_payload(),
    }
    if not result["pass"]:
        raise AssertionError(result)
    return result


def verify_replay(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.runtime_root)
    frozen = load_frozen(root)
    contract = read_json(runtime_paths(root)["contract"])
    as_of = parse_utc(args.as_of)
    if as_of >= pd.Timestamp("2026-01-01", tz="UTC"):
        raise AssertionError("replay verification is restricted to already-open 2025")
    panel, panel_audit = core.prepare_causal_panel(
        list(contract["data"]["frozen_stock_universe"]),
        Path(args.provider_root),
        as_of,
        22,
    )
    assigned = core.assign_session_states(
        panel, frozen["preprocessing"], frozen["state_parameters"]
    )
    state_source = Path(args.state_artifact_root) / "causal_state_assignments.parquet"
    expected_states = pd.read_parquet(
        state_source,
        columns=["symbol_norm", "session_date", "timestamp", "state", "age"],
        filters=[("session_date", "==", panel_audit["session_date"])],
    )
    expected_states["timestamp"] = pd.to_datetime(
        expected_states["timestamp"], utc=True
    )
    compared_states = assigned.merge(
        expected_states,
        on=["symbol_norm", "session_date", "timestamp"],
        suffixes=("__new", "__stored"),
        validate="one_to_one",
    )
    if len(compared_states) != len(assigned):
        raise AssertionError("replay state support mismatch")
    state_mismatches = int(
        compared_states["state__new"].astype(int).ne(
            compared_states["state__stored"].astype(int)
        ).sum()
    )
    age_mismatches = int(
        compared_states["age__new"].astype(int).ne(
            compared_states["age__stored"].astype(int)
        ).sum()
    )
    runs = core.make_session_runs(
        panel,
        frozen["preprocessing"],
        frozen["state_parameters"],
        frozen["cycles"],
        frozen["path_parameters"],
    )
    runs = runs.loc[runs["bar_index_in_session"].le(53)].copy()
    runs = core.movement_predictions(
        runs, frozen["feature_manifest"], frozen["outcome_parameters"]
    )
    price_root = Path(args.price_artifact_root)
    loop_columns = list(frozen["feature_manifest"]["loop_score_columns"])
    prediction_columns = [
        f"{representation}__{target}_prediction_{horizon}"
        for representation in core.REPRESENTATIONS
        for target in core.MOVEMENT_TARGETS
        for horizon in core.HORIZONS
    ]
    anchor_columns = [
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "state",
        "previous_state_1",
        "previous_state_2",
        *core.NUMERIC_CONTROLS,
        *loop_columns,
    ]
    stored_anchors = pd.read_parquet(
        price_root / "anchor_panel_2025.parquet",
        columns=anchor_columns,
        filters=[("session_date", "==", panel_audit["session_date"])],
    )
    stored_predictions = pd.read_parquet(
        price_root / "price_predictions_2025.parquet",
        columns=["symbol_norm", "start_timestamp", *prediction_columns],
        filters=[("session_date", "==", panel_audit["session_date"])],
    )
    stored = stored_anchors.merge(
        stored_predictions,
        on=["symbol_norm", "start_timestamp"],
        validate="one_to_one",
    )
    joined = runs.merge(
        stored,
        on=["symbol_norm", "session_date", "start_timestamp"],
        suffixes=("__new", "__stored"),
        validate="one_to_one",
    )
    if len(joined) != len(runs) or len(joined) != len(stored):
        raise AssertionError("replay run-entry support mismatch")
    exact_columns = ["state", "previous_state_1", "previous_state_2"]
    exact_mismatches = {
        column: int(
            joined[f"{column}__new"].astype(int).ne(
                joined[f"{column}__stored"].astype(int)
            ).sum()
        )
        for column in exact_columns
    }
    numeric_columns = [*core.NUMERIC_CONTROLS, *loop_columns, *prediction_columns]
    numeric_errors = {
        column: float(
            np.nanmax(
                np.abs(
                    joined[f"{column}__new"].to_numpy(dtype=float)
                    - joined[f"{column}__stored"].to_numpy(dtype=float)
                )
            )
        )
        for column in numeric_columns
    }
    maximum_numeric_error = max(numeric_errors.values())
    result = {
        "as_of": as_of,
        "session_date": panel_audit["session_date"],
        "state_rows": len(compared_states),
        "run_entry_rows": len(joined),
        "state_mismatches": state_mismatches,
        "age_mismatches": age_mismatches,
        "run_field_mismatches": exact_mismatches,
        "maximum_numeric_error": maximum_numeric_error,
        "pass": bool(
            state_mismatches == 0
            and age_mismatches == 0
            and not any(exact_mismatches.values())
            and maximum_numeric_error <= 1e-5
        ),
        "outcome_columns_read": False,
        "direction_columns_read": False,
        "signed_return_columns_read": False,
        **safety_payload(),
    }
    if not result["pass"]:
        write_json_atomic(runtime_paths(root)["root"] / "replay_failure.json", result)
        raise AssertionError(result)
    write_json_atomic(runtime_paths(root)["root"] / "replay_verification.json", result)
    return result


def _read_symbol_window(
    root: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: list[str],
) -> pd.DataFrame:
    path = core.provider_path(root, symbol)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("timestamp", ">=", start.to_pydatetime()),
            ("timestamp", "<=", end.to_pydatetime()),
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.sort_values("timestamp", kind="mergesort")
    if frame["timestamp"].duplicated().any():
        raise AssertionError(f"duplicate provider timestamp for {symbol}")
    return frame


def exact_outcome_mask(predictions: pd.DataFrame, provider_root: Path) -> np.ndarray:
    exact = np.ones(len(predictions), dtype=bool)
    for symbol, indices in predictions.groupby("symbol_norm", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        anchors = pd.to_datetime(
            predictions.loc[positions, "start_timestamp"], utc=True
        )
        timestamps = _read_symbol_window(
            provider_root,
            str(symbol),
            anchors.min(),
            anchors.max() + pd.Timedelta(minutes=120),
            ["timestamp"],
        )["timestamp"]
        available = set(timestamps.astype("int64").tolist())
        local_dates = anchors.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        for local_index, (position, anchor, session_date) in enumerate(
            zip(positions, anchors, local_dates, strict=True)
        ):
            del local_index
            if int(anchor.value) not in available:
                exact[position] = False
                continue
            for step in range(1, max(core.HORIZONS) + 1):
                future = anchor + pd.Timedelta(minutes=5 * step)
                if int(future.value) not in available:
                    exact[position] = False
                    break
                if future.tz_convert("America/New_York").strftime("%Y-%m-%d") != session_date:
                    exact[position] = False
                    break
    return exact


def exact_support_summary(
    predictions: pd.DataFrame, contract: dict[str, Any]
) -> dict[str, Any]:
    outcome_protocol = contract["outcome_protocol"]
    states = sorted(predictions["state"].astype(int).unique())
    summary = {
        "exact_outcome_anchors": int(len(predictions)),
        "distinct_session_dates": int(predictions["session_date"].nunique()),
        "distinct_symbols": int(predictions["symbol_norm"].nunique()),
        "distinct_calendar_quarters": int(predictions["quarter"].nunique()),
        "states": states,
    }
    close = contract["cohort_close_rule"]
    gates = {
        "exact_outcome_anchors": len(predictions)
        >= int(outcome_protocol["minimum_exact_outcome_anchors"]),
        "session_dates": summary["distinct_session_dates"]
        >= int(close["minimum_distinct_session_dates"]),
        "symbols": summary["distinct_symbols"] >= int(close["minimum_distinct_symbols"]),
        "calendar_quarters": summary["distinct_calendar_quarters"]
        >= int(close["minimum_distinct_calendar_quarters"]),
        "states": states == list(close["required_states"]),
    }
    summary["gates"] = gates
    summary["pass"] = bool(all(gates.values()))
    return summary


def build_outcome_panel(predictions: pd.DataFrame, provider_root: Path) -> pd.DataFrame:
    output = predictions.copy()
    for target in core.MOVEMENT_TARGETS:
        for horizon in core.HORIZONS:
            output[f"{target}_outcome_{horizon}"] = np.nan
    for symbol, indices in output.groupby("symbol_norm", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        anchors = pd.to_datetime(output.loc[positions, "start_timestamp"], utc=True)
        prices = _read_symbol_window(
            provider_root,
            str(symbol),
            anchors.min(),
            anchors.max() + pd.Timedelta(minutes=120),
            ["timestamp", "high", "low", "close"],
        ).set_index("timestamp")
        for position, anchor in zip(positions, anchors, strict=True):
            current_close = float(prices.at[anchor, "close"])
            for horizon in core.HORIZONS:
                future_times = [
                    anchor + pd.Timedelta(minutes=5 * step)
                    for step in range(1, horizon + 1)
                ]
                future = prices.loc[future_times]
                future_close = float(future.iloc[-1]["close"])
                absolute_return = abs(10000.0 * math.log(future_close / current_close))
                future_range = (
                    10000.0
                    * (float(future["high"].max()) - float(future["low"].min()))
                    / current_close
                )
                output.at[position, f"absolute_return_bps_outcome_{horizon}"] = absolute_return
                output.at[position, f"future_range_bps_outcome_{horizon}"] = future_range
    outcome_columns = [
        f"{target}_outcome_{horizon}"
        for target in core.MOVEMENT_TARGETS
        for horizon in core.HORIZONS
    ]
    if not np.isfinite(output[outcome_columns].to_numpy(dtype=float)).all():
        raise AssertionError("non-finite movement outcome")
    return output


def movement_evaluation(
    outcome_panel: pd.DataFrame, contract: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    long_parts = []
    for horizon in core.HORIZONS:
        columns = [
            "prediction_id",
            "symbol_norm",
            "session_date",
            "quarter",
            "state",
        ]
        part = outcome_panel[columns].copy()
        part["horizon"] = horizon
        for target in core.MOVEMENT_TARGETS:
            part[target] = outcome_panel[f"{target}_outcome_{horizon}"].to_numpy(float)
            for representation in core.REPRESENTATIONS:
                part[f"{representation}__{target}"] = outcome_panel[
                    f"{representation}__{target}_prediction_{horizon}"
                ].to_numpy(float)
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True)
    target_results = {}
    metric_rows = []
    for target in core.MOVEMENT_TARGETS:
        threshold = float(
            contract["evaluation"][target]["minimum_mse_relative_improvement"]
        )
        outcome = long[target].to_numpy(dtype=float)
        candidate_prediction = long[f"loop_scores__{target}"].to_numpy(dtype=float)
        baseline_prediction = long[f"state_context__{target}"].to_numpy(dtype=float)
        candidate_mse = np.square(candidate_prediction - outcome)
        baseline_mse = np.square(baseline_prediction - outcome)
        candidate_mae = np.abs(candidate_prediction - outcome)
        baseline_mae = np.abs(baseline_prediction - outcome)
        mse_difference = candidate_mse - baseline_mse
        mae_difference = candidate_mae - baseline_mae
        pooled_improvement = float(-mse_difference.mean() / baseline_mse.mean())
        daily_mse = (
            pd.DataFrame(
                {"session_date": long["session_date"], "difference": mse_difference}
            )
            .groupby("session_date", sort=True)["difference"]
            .mean()
            .to_numpy(dtype=float)
        )
        daily_mae = (
            pd.DataFrame(
                {"session_date": long["session_date"], "difference": mae_difference}
            )
            .groupby("session_date", sort=True)["difference"]
            .mean()
            .to_numpy(dtype=float)
        )
        mse_interval = core.moving_block_bounds(daily_mse, SEED)
        mae_interval = core.moving_block_bounds(daily_mae, SEED)
        horizon_improvements = {}
        correlations = {}
        for horizon in core.HORIZONS:
            mask = long["horizon"].eq(horizon).to_numpy()
            horizon_improvements[str(horizon)] = float(
                (baseline_mse[mask].mean() - candidate_mse[mask].mean())
                / baseline_mse[mask].mean()
            )
            correlations[str(horizon)] = {
                "state_context": float(
                    np.corrcoef(baseline_prediction[mask], outcome[mask])[0, 1]
                ),
                "loop_scores": float(
                    np.corrcoef(candidate_prediction[mask], outcome[mask])[0, 1]
                ),
            }
        deletion_improvements = {}
        for symbol in sorted(long["symbol_norm"].astype(str).unique()):
            mask = long["symbol_norm"].astype(str).ne(symbol).to_numpy()
            deletion_improvements[symbol] = float(
                (baseline_mse[mask].mean() - candidate_mse[mask].mean())
                / baseline_mse[mask].mean()
            )
        quarter_differences = {}
        for quarter in sorted(long["quarter"].astype(str).unique()):
            mask = long["quarter"].astype(str).eq(quarter).to_numpy()
            quarter_differences[quarter] = {
                "mse": float(mse_difference[mask].mean()),
                "mae": float(mae_difference[mask].mean()),
            }
        gates = {
            "pooled_mse_improvement": pooled_improvement >= threshold,
            "each_horizon_mse_improvement": min(horizon_improvements.values())
            >= threshold,
            "each_leave_one_symbol_out_mse_improvement": min(
                deletion_improvements.values()
            )
            >= threshold,
            "mse_daily_interval_upper_below_zero": mse_interval[2] < 0.0,
            "mae_daily_interval_upper_below_zero": mae_interval[2] < 0.0,
            "every_quarter_mse_and_mae_lower": all(
                values["mse"] < 0.0 and values["mae"] < 0.0
                for values in quarter_differences.values()
            ),
            "correlation_better_each_horizon": all(
                values["loop_scores"] > values["state_context"]
                for values in correlations.values()
            ),
        }
        target_results[target] = {
            "minimum_required_mse_improvement": threshold,
            "pooled_mse_relative_improvement": pooled_improvement,
            "horizon_mse_relative_improvements": horizon_improvements,
            "leave_one_symbol_out_mse_relative_improvements": deletion_improvements,
            "daily_mse_difference_interval": {
                "mean": mse_interval[0],
                "low": mse_interval[1],
                "high": mse_interval[2],
            },
            "daily_mae_difference_interval": {
                "mean": mae_interval[0],
                "low": mae_interval[1],
                "high": mae_interval[2],
            },
            "quarter_loss_differences": quarter_differences,
            "correlations": correlations,
            "gates": gates,
            "pass": bool(all(gates.values())),
        }
        for representation, mse, mae, prediction in (
            ("state_context", baseline_mse, baseline_mae, baseline_prediction),
            ("loop_scores", candidate_mse, candidate_mae, candidate_prediction),
        ):
            for horizon in core.HORIZONS:
                mask = long["horizon"].eq(horizon).to_numpy()
                metric_rows.append(
                    {
                        "target": target,
                        "representation": representation,
                        "horizon": horizon,
                        "rows": int(mask.sum()),
                        "mse": float(mse[mask].mean()),
                        "mae": float(mae[mask].mean()),
                        "correlation": float(
                            np.corrcoef(prediction[mask], outcome[mask])[0, 1]
                        ),
                    }
                )
    overall_pass = bool(all(result["pass"] for result in target_results.values()))
    result = {
        "targets": target_results,
        "overall_pass": overall_pass,
        "interpretation": (
            "Prospective predictive-information evidence only; no economic-edge, "
            "tradability, direction, signed-return, P&L, order, or deployment claim."
        ),
        **safety_payload(),
    }
    return result, pd.DataFrame(metric_rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.runtime_root)
    verify_runtime(root)
    paths = runtime_paths(root)
    contract = read_json(paths["contract"])

    # This support check is intentionally before provider_root is resolved or read.
    closing_sequence, issuance_support = first_closing_prefix(root, contract)
    if closing_sequence is None:
        raise SupportNotMet(
            "outcome embargo active; issuance support is not met: "
            + json.dumps(core.safe(issuance_support), sort_keys=True)
        )
    _, predictions = load_prediction_batches(root, through_sequence=closing_sequence)
    latest_anchor = pd.to_datetime(predictions["start_timestamp"], utc=True).max()
    mature_at = latest_anchor + pd.Timedelta(
        minutes=int(
            contract["outcome_protocol"][
                "evaluation_not_before_latest_anchor_minutes"
            ]
        )
    )
    if pd.Timestamp.now(tz="UTC") < mature_at:
        raise SupportNotMet(
            f"outcome embargo active until closing cohort matures at {mature_at.isoformat()}"
        )
    provider_root = Path(args.provider_root)

    # First provider access is timestamp-only.  OHLC is still embargoed.
    timestamp_failure_path = paths["evaluation"] / "timestamp_support_failure.json"
    if timestamp_failure_path.is_file():
        return read_json(timestamp_failure_path)
    exact_mask = exact_outcome_mask(predictions, provider_root)
    exact_predictions = predictions.loc[exact_mask].reset_index(drop=True)
    exact_support = exact_support_summary(exact_predictions, contract)
    if not exact_support["pass"]:
        failure = {
            "contract_id": contract["contract_id"],
            "closing_sequence": closing_sequence,
            "issuance_support": issuance_support,
            "exact_timestamp_support": exact_support,
            "outcome_values_opened": False,
            "performance_metrics_calculated": False,
            **safety_payload(),
        }
        write_json_atomic(timestamp_failure_path, failure)
        return failure

    final_path = paths["evaluation"] / "final_result.json"
    if final_path.is_file():
        return read_json(final_path)
    outcome_panel = build_outcome_panel(exact_predictions, provider_root)
    evaluation, metric_frame = movement_evaluation(outcome_panel, contract)
    outcome_path = paths["evaluation"] / "sealed_outcome_panel.parquet"
    temporary = outcome_path.with_name(f".{outcome_path.name}.tmp")
    outcome_panel.to_parquet(temporary, index=False)
    temporary.replace(outcome_path)
    metric_frame.to_csv(paths["evaluation"] / "movement_metrics.csv", index=False)
    result = {
        "contract_id": contract["contract_id"],
        "closing_sequence": closing_sequence,
        "closing_record_sha256": read_ledger(root)[closing_sequence - 1][
            "record_sha256"
        ],
        "issuance_support": issuance_support,
        "exact_timestamp_support": exact_support,
        "outcome_panel_sha256": core.sha256_file(outcome_path),
        "evaluation": evaluation,
        "outcome_values_opened": True,
        "performance_metrics_calculated": True,
        **safety_payload(),
    }
    write_json_atomic(final_path, result)
    write_json_atomic(
        paths["evaluation"] / "evaluation_seal.json",
        {
            "final_result_sha256": core.sha256_file(final_path),
            "outcome_panel_sha256": core.sha256_file(outcome_path),
            "movement_metrics_sha256": core.sha256_file(
                paths["evaluation"] / "movement_metrics.csv"
            ),
            **safety_payload(),
        },
    )
    return result


def self_tests() -> dict[str, Any]:
    core.inference_self_tests()
    sample_contract = {
        "cohort_close_rule": {
            "minimum_issued_anchors": 2,
            "minimum_distinct_session_dates": 2,
            "minimum_distinct_symbols": 2,
            "minimum_distinct_calendar_quarters": 1,
            "required_states": [0, 1],
        }
    }
    sample = pd.DataFrame(
        {
            "state": [0, 1],
            "session_date": ["2026-07-13", "2026-07-14"],
            "symbol_norm": ["A", "B"],
            "quarter": ["2026_q3", "2026_q3"],
        }
    )
    assert support_summary(sample, sample_contract)["pass"] is True
    assert support_summary(sample.iloc[:1], sample_contract)["pass"] is False
    return {"self_tests_passed": True, **safety_payload()}


def add_runtime_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="freeze and archive the model lineage")
    add_runtime_argument(init)
    init.add_argument("--lineage-root", type=Path)
    init.add_argument("--state-artifact-root", type=Path)
    init.add_argument("--path-artifact-root", type=Path)
    init.add_argument("--price-artifact-root", type=Path)
    init.add_argument("--shadow-workspace-root", type=Path)

    verify = subparsers.add_parser(
        "verify-frozen-inference", help="reproduce stored 2025 movement predictions"
    )
    add_runtime_argument(verify)
    verify.add_argument(
        "--price-artifact-root", type=Path, default=DEFAULT_PRICE_ARTIFACT_ROOT
    )

    replay = subparsers.add_parser(
        "verify-replay", help="rebuild one already-open 2025 session end to end"
    )
    add_runtime_argument(replay)
    replay.add_argument("--provider-root", type=Path, default=DEFAULT_PROVIDER_ROOT)
    replay.add_argument("--as-of", required=True)
    replay.add_argument(
        "--state-artifact-root", type=Path, default=DEFAULT_STATE_ARTIFACT_ROOT
    )
    replay.add_argument(
        "--price-artifact-root", type=Path, default=DEFAULT_PRICE_ARTIFACT_ROOT
    )

    issue = subparsers.add_parser("issue", help="append a prediction-only batch")
    add_runtime_argument(issue)
    issue.add_argument("--provider-root", type=Path, default=DEFAULT_PROVIDER_ROOT)
    issue.add_argument("--as-of", required=True)

    show = subparsers.add_parser("status", help="show support without opening outcomes")
    add_runtime_argument(show)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="open outcomes only after frozen support closes"
    )
    add_runtime_argument(evaluate_parser)
    evaluate_parser.add_argument(
        "--provider-root", type=Path, default=DEFAULT_PROVIDER_ROOT
    )

    subparsers.add_parser("self-test", help="run outcome-free synthetic tests")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "init":
            result = init_runtime(args)
        elif args.command == "verify-frozen-inference":
            result = verify_frozen_inference(args)
        elif args.command == "verify-replay":
            result = verify_replay(args)
        elif args.command == "issue":
            result = issue_predictions(args)
        elif args.command == "status":
            result = status(Path(args.runtime_root))
        elif args.command == "evaluate":
            result = evaluate(args)
        elif args.command == "self-test":
            result = self_tests()
        else:
            raise AssertionError(args.command)
    except SupportNotMet as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(3) from exc
    print(json.dumps(core.safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
