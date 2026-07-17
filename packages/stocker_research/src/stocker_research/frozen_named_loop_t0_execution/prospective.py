"""Command support for the frozen prospective collection identity."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from .historical import load_and_verify_contract, stable_hash
from .immutable_ledger import ProspectiveExecutionLedger


def load_payloads(path: Path) -> list[dict[str, object]]:
    """Load one JSON object, a JSON array, or newline-delimited JSON objects."""

    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        values = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not values or not all(isinstance(value, Mapping) for value in values):
        raise ValueError("record input must contain one or more JSON objects")
    return [dict(cast(Mapping[str, object], value)) for value in values]


def collection_parameters(contract_path: Path) -> dict[str, object]:
    """Derive the immutable collection identity from the registered contract."""

    contract, contract_hash, input_hashes = load_and_verify_contract(contract_path)
    manifest_path = (
        Path(contract_path).resolve().parent
        / str(contract["inputs"]["provider_2025_hash_manifest"]["path"])
    ).resolve()
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    provider_hashes = {
        str(key).removeprefix("provider_2025_"): str(value)
        for key, value in cast(Mapping[str, object], manifest["sha256"]).items()
        if str(key).startswith("provider_2025_")
    }
    opened_snapshot = stable_hash(
        {
            "inputs": dict(sorted(input_hashes.items())),
            "providers": dict(sorted(provider_hashes.items())),
        }
    )
    periods = {
        int(period)
        for period in cast(
            Mapping[str, object],
            contract["populations"]["expected_historical_source_counts"],
        )
    }
    return {
        "contract_hash": contract_hash,
        "completion_rule_hash": stable_hash(contract["prospective_completion_rule"]),
        "opened_periods": periods,
        "opened_snapshot_hashes": {opened_snapshot},
    }


def open_collection_ledger(root: Path, *, contract_path: Path) -> ProspectiveExecutionLedger:
    parameters = collection_parameters(contract_path)
    return ProspectiveExecutionLedger(root=Path(root), **parameters)  # type: ignore[arg-type]


def append_payloads(
    *,
    ledger_root: Path,
    contract_path: Path,
    stage: str,
    records: Iterable[Mapping[str, object]],
    dry_run: bool,
) -> list[Path]:
    """Validate or create records without ever calling an execution interface."""

    payloads = [dict(record) for record in records]
    if stage not in {"opportunity", "trigger", "settlement"}:
        raise ValueError(f"unknown immutable stage: {stage}")
    if not payloads:
        raise ValueError("at least one record is required")
    if not dry_run:
        ledger = open_collection_ledger(ledger_root, contract_path=contract_path)
        return [_append(ledger, stage, payload) for payload in payloads]

    with tempfile.TemporaryDirectory(prefix="frozen-t0-ledger-dry-run-") as directory:
        temporary_root = Path(directory)
        ledger = open_collection_ledger(temporary_root, contract_path=contract_path)
        if stage in {"trigger", "settlement"}:
            _copy_precursors(
                source_root=Path(ledger_root),
                destination_root=temporary_root,
                stage=stage,
                opportunity_ids=[str(row["opportunity_id"]) for row in payloads],
            )
        paths = [_append(ledger, stage, payload) for payload in payloads]
        return [Path("DRY_RUN") / path.relative_to(temporary_root) for path in paths]


def _append(
    ledger: ProspectiveExecutionLedger,
    stage: str,
    payload: Mapping[str, object],
) -> Path:
    if stage == "opportunity":
        return ledger.append_opportunity(payload, prospective=True)
    if stage == "trigger":
        return ledger.append_trigger(payload)
    return ledger.append_settlement(payload)


def _copy_precursors(
    *,
    source_root: Path,
    destination_root: Path,
    stage: str,
    opportunity_ids: Iterable[str],
) -> None:
    stages = ["opportunities"]
    if stage == "settlement":
        stages.append("triggers")
    for opportunity_id in opportunity_ids:
        for precursor in stages:
            source = source_root / precursor / f"{opportunity_id}.json"
            if not source.is_file():
                raise ValueError(f"missing immutable precursor for dry run: {source}")
            destination = destination_root / precursor / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
