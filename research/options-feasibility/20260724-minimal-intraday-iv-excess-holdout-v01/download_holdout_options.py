#!/usr/bin/env python3
"""Resume only missing exact-D-1 option requests for holdout V0.1."""

from __future__ import annotations

# ruff: noqa: E402 -- numerical/network limits are fixed before imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
import requests

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
V0_EXPERIMENT_DIR = (
    REPO_ROOT / "research/options-feasibility/20260723-minimal-intraday-iv-excess-holdout-v0"
)
V0_PRIMARY = V0_EXPERIMENT_DIR / "artifacts" / "primary"
V0_DOWNLOADER = V0_EXPERIMENT_DIR / "download_holdout_options.py"
DEFAULT_PROVIDER_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock"
)
DEFAULT_V0_CACHE_ROOT = (
    REPO_ROOT / "data/vendor/eodhd/options/minimal-intraday-iv-excess-holdout-v0"
)
DEFAULT_RESUME_CACHE_ROOT = (
    REPO_ROOT / "data/vendor/eodhd/options/minimal-intraday-iv-excess-holdout-v01"
)

for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.eodhd_options_downloader_v0 import (
    OPTIONS_EOD_ENDPOINT,
    DownloadConfig,
    EODHDOptionsDownloader,
    OptionsAuthenticationError,
    OptionsDownloadError,
    OptionsResourceLimitExceeded,
    canonicalize_response_records,
    provider_eod_observation_date,
    resolve_canonical_duplicates,
)
from stocker_research.minimal_intraday_iv_excess_holdout_v01 import (
    EXPECTED_FROZEN_REQUESTS,
    EXPECTED_V0_COMPLETE_RECEIPTS,
    MAXIMUM_ADDITIONAL_BYTES,
    MAXIMUM_ADDITIONAL_RECORDS,
    MAXIMUM_CUMULATIVE_RECORDS,
    SAFETY_FLAGS,
    ResumeResourceLimitError,
    assert_v01_safety_flags,
    identify_interrupted_request,
    inventory_complete_receipts,
    load_verified_receipt_records,
    remaining_resume_requests,
    request_identity,
    validate_additional_resource_usage,
)

V0_RECORDS_ENCOUNTERED = 350_000
V0_COMPLETE_RECEIPT_RECORDS = 349_802
V0_INCOMPLETE_PAGE_RECORDS = 198
V0_BYTES_ENCOUNTERED = 317_272_704


class AcquisitionBlocked(RuntimeError):
    """A fail-closed V0.1 acquisition decision."""

    def __init__(self, decision: str, detail: str) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, Path, date)):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise AcquisitionBlocked(
            "blocked_reproducibility_or_audit_failure",
            f"cannot load frozen V0 source: {path}",
        )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def rebuild_frozen_plan(provider_root: Path, v0: ModuleType) -> list[dict[str, object]]:
    """Rebuild and exactly match the immutable V0 request plan."""

    planned = cast(Sequence[Any], v0.build_request_plan(provider_root))
    rows = [cast(dict[str, object], asdict(row)) for row in planned]
    stored = cast(
        dict[str, Any],
        json.loads((V0_PRIMARY / "holdout_options_request_plan.json").read_text(encoding="utf-8")),
    )
    frozen = cast(list[dict[str, object]], stored["requests"])
    if rows != frozen or len(rows) != EXPECTED_FROZEN_REQUESTS:
        raise AcquisitionBlocked(
            "blocked_reproducibility_or_audit_failure",
            "rebuilt holdout request plan differs from the frozen V0 plan",
        )
    identities = [request_identity(row) for row in rows]
    if len(set(identities)) != EXPECTED_FROZEN_REQUESTS:
        raise AcquisitionBlocked(
            "blocked_reproducibility_or_audit_failure",
            "frozen request plan contains duplicate logical identities",
        )
    return rows


def referenced_raw_paths(cache_root: Path) -> set[Path]:
    """Return raw bodies referenced by any parseable completed receipt."""

    output: set[Path] = set()
    completed = cache_root / "manifests" / "completed"
    for receipt in sorted(completed.glob("*.json")):
        try:
            stored = json.loads(receipt.read_text(encoding="utf-8"))
            rows = stored["manifest_rows"]
        except (KeyError, json.JSONDecodeError, OSError, TypeError):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping) and isinstance(row.get("cache_path"), str):
                output.add(Path(cast(str, row["cache_path"])).resolve())
    return output


def unreceipted_raw_paths(cache_root: Path) -> list[Path]:
    all_raw = {path.resolve() for path in (cache_root / "raw").glob("*.json")}
    return sorted(all_raw.difference(referenced_raw_paths(cache_root)))


def raw_page_accounting(paths: Sequence[Path]) -> dict[str, int]:
    records = 0
    exact = 0
    extra = 0
    protected = 0
    raw_bytes = 0
    for path in paths:
        content = path.read_bytes()
        raw_bytes += len(content)
        payload = cast(dict[str, Any], json.loads(content))
        data = payload.get("data")
        if not isinstance(data, list):
            raise AcquisitionBlocked(
                "blocked_holdout_options_download_failure",
                "an unreceipted V0.1 raw response lacks provider data rows",
            )
        fields = cast(dict[str, Any], payload.get("meta", {})).get("fields")
        decoded: list[Mapping[str, Any]] = []
        for item in data:
            if isinstance(item, Mapping):
                decoded.append(cast(Mapping[str, Any], item))
            elif isinstance(item, list) and isinstance(fields, list):
                decoded.append(
                    {
                        "type": "options-eod",
                        "attributes": dict(
                            zip((str(field) for field in fields), item, strict=True)
                        ),
                    }
                )
            else:
                raise AcquisitionBlocked(
                    "blocked_holdout_options_download_failure",
                    "an unreceipted V0.1 response row differs from the audited schema",
                )
        records += len(decoded)
        for record in decoded:
            observed = provider_eod_observation_date(record)
            protected += int(observed >= date(2026, 1, 1))
            # The requested date is not trusted without a completed receipt.
            extra += 1
    return {
        "records": records,
        "exact": exact,
        "extra": extra,
        "protected": protected,
        "bytes": raw_bytes,
    }


def cache_root_for_receipt(receipt_path: Path) -> Path:
    if receipt_path.parent.name != "completed" or receipt_path.parent.parent.name != "manifests":
        raise AcquisitionBlocked(
            "blocked_reproducibility_or_audit_failure",
            "verified receipt path is outside the expected cache layout",
        )
    return receipt_path.parents[2]


def materialize_complete_canonical_cache(
    plan_rows: Sequence[Mapping[str, object]],
    *,
    receipt_paths: Mapping[str, Path],
    canonical_path: Path,
    v0: ModuleType,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Materialize exact-date canonical rows from complete receipts only."""

    canonical_records: list[dict[str, Any]] = []
    request_audit: list[dict[str, object]] = []
    totals = {
        "provider_records": 0,
        "exact_date_records": 0,
        "extra_date_records_rejected": 0,
        "protected_date_records_rejected": 0,
        "nonstandard_contract_records_rejected": 0,
        "canonical_records_accepted": 0,
        "canonical_records_rejected": 0,
    }
    for planned in sorted(
        plan_rows,
        key=lambda row: (
            str(row["symbol"]),
            str(row["required_options_date"]),
            str(row["holdout_session"]),
        ),
    ):
        identity = request_identity(planned)
        receipt_path = receipt_paths.get(identity)
        if receipt_path is None:
            raise AcquisitionBlocked(
                "blocked_holdout_options_download_failure",
                f"complete canonical materialization lacks request {identity}",
            )
        cache_root = cache_root_for_receipt(receipt_path)
        records = load_verified_receipt_records(
            planned,
            receipt_path=receipt_path,
            cache_root=cache_root,
            canonical_cache_path=canonical_path,
        )
        required = date.fromisoformat(str(planned["required_options_date"]))
        exact: list[Mapping[str, Any]] = []
        protected = 0
        for record in records:
            observed = provider_eod_observation_date(record)
            protected += int(observed >= date(2026, 1, 1))
            if observed == required and observed < date(2026, 1, 1):
                exact.append(record)
        standard = [
            record
            for record in exact
            if bool(v0.is_standard_occ(record, symbol=str(planned["symbol"])))
        ]
        canonical = canonicalize_response_records(
            standard,
            request_id=identity,
            provider_schema_version="eodhd_options_eod_noncompact_exact_date_v1",
        )
        canonical_records.extend(canonical.records)
        values = {
            "provider_records": len(records),
            "exact_date_records": len(exact),
            "extra_date_records_rejected": len(records) - len(exact),
            "protected_date_records_rejected": protected,
            "nonstandard_contract_records_rejected": len(exact) - len(standard),
            "canonical_records_accepted": len(canonical.records),
            "canonical_records_rejected": len(canonical.rejections),
        }
        for name, value in values.items():
            totals[name] += value
        request_audit.append(
            {
                **dict(planned),
                "request_id": identity,
                "receipt_source": (
                    "V0_reused"
                    if cache_root.resolve() == DEFAULT_V0_CACHE_ROOT.resolve()
                    else "V0.1_resume"
                ),
                **values,
            }
        )
    deduplicated = resolve_canonical_duplicates(canonical_records)
    canonical_frame = pd.DataFrame(deduplicated.records)
    request_frame = pd.DataFrame(request_audit)
    write_parquet(canonical_path, canonical_frame)
    write_parquet(canonical_path.with_name("request_audit.parquet"), request_frame)
    totals["duplicate_records"] = deduplicated.duplicate_records
    totals["conflicting_duplicate_groups"] = deduplicated.conflicting_duplicate_groups
    totals["canonical_cache_rows"] = len(canonical_frame)
    return canonical_frame, request_frame, totals


def request_coverage(request_audit: pd.DataFrame) -> pd.DataFrame:
    frame = request_audit.copy()
    frame["month"] = frame["holdout_session"].astype(str).str[:7]
    return (
        frame.groupby(["symbol", "month"], sort=True, observed=True)
        .agg(
            requests_planned=("request_id", "size"),
            requests_completed=("request_id", "size"),
            requests_with_exact_records=(
                "exact_date_records",
                lambda values: int(pd.Series(values).gt(0).sum()),
            ),
            exact_date_records=("exact_date_records", "sum"),
            canonical_records=("canonical_records_accepted", "sum"),
        )
        .reset_index()
    )


def missing_gap_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    columns = [
        "request_id",
        "symbol",
        "holdout_session",
        "required_options_date",
        "gap_reason",
    ]
    output = [
        {
            "request_id": request_identity(row),
            "symbol": row["symbol"],
            "holdout_session": row["holdout_session"],
            "required_options_date": row["required_options_date"],
            "gap_reason": "no_verified_complete_receipt",
        }
        for row in rows
    ]
    return pd.DataFrame(output, columns=columns)


def download_remaining(
    rows: Sequence[Mapping[str, object]],
    *,
    token: str,
    resume_cache_root: Path,
    v0: ModuleType,
    existing_records: int,
    existing_bytes: int,
) -> tuple[int, int]:
    """Download only the current missing scope in one deterministic process."""

    remaining_record_budget = MAXIMUM_ADDITIONAL_RECORDS - existing_records
    remaining_byte_budget = MAXIMUM_ADDITIONAL_BYTES - existing_bytes
    if rows and (remaining_record_budget <= 0 or remaining_byte_budget <= 0):
        raise AcquisitionBlocked(
            "blocked_resume_resource_limit",
            "no V0.1 resource allowance remains for missing logical requests",
        )
    if not rows:
        return 0, 0
    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token=token,
            data_dir=resume_cache_root,
            page_limit=1_000,
            maximum_offset=10_000,
            request_timeout_seconds=30.0,
            max_attempts=4,
            exponential_backoff_seconds=1.0,
            requests_per_minute=300,
            maximum_raw_records=remaining_record_budget,
            maximum_download_bytes=remaining_byte_budget,
        ),
        transport=requests.Session(),
    )
    completed = 0
    records = 0
    for index, planned in enumerate(rows, start=1):
        request = v0.request_from_plan(v0.PlannedRequest(**dict(planned)))
        result = downloader.download_with_splitting(request)
        completed += 1
        records += len(result.records)
        if index % 25 == 0 or index == len(rows):
            print(f"completed {index}/{len(rows)} resume-only logical requests", flush=True)
    return completed, records


def run(args: argparse.Namespace) -> dict[str, Any]:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    v0 = load_module(V0_DOWNLOADER, "minimal_holdout_v0_downloader_for_resume")
    plan = rebuild_frozen_plan(args.provider_root, v0)
    canonical_path = args.resume_cache_root / "canonical/exact_holdout_options.parquet"

    v0_inventory = inventory_complete_receipts(
        plan,
        cache_roots=[args.v0_cache_root],
        canonical_cache_path=canonical_path,
    )
    v0_inventory.audit["redundant_network_request_prevented"] = v0_inventory.audit["reused"].fillna(
        False
    )
    write_csv(PRIMARY / "complete_receipt_reuse_audit.csv", v0_inventory.audit)
    if v0_inventory.complete_receipts_found != EXPECTED_V0_COMPLETE_RECEIPTS:
        raise AcquisitionBlocked(
            "blocked_reproducibility_or_audit_failure",
            "V0 completed-receipt count differs from the frozen blocker state",
        )

    original_missing = remaining_resume_requests(
        plan,
        v0_inventory.verified_request_ids,
    )
    v0_orphans = unreceipted_raw_paths(args.v0_cache_root)
    if len(v0_orphans) != 1:
        raise AcquisitionBlocked(
            "blocked_reproducibility_or_audit_failure",
            f"expected one interrupted V0 raw page, found {len(v0_orphans)}",
        )
    interrupted = identify_interrupted_request(v0_orphans[0], original_missing)
    repair: dict[str, Any] = {
        **SAFETY_FLAGS,
        **asdict(interrupted),
        "original_raw_cache_path": str(v0_orphans[0]),
        "original_raw_sha256": sha256_file(v0_orphans[0]),
        "last_complete_page_verified": False,
        "deterministic_page_continuation_used": False,
        "new_pages_downloaded": 0,
        "deduplication_result": {
            "incomplete_page_rows_admitted": 0,
            "duplicate_rows_materialised": 0,
        },
        "final_receipt_hash": None,
        "final_exact_date_count": None,
        "status": "pending",
    }
    write_json(PRIMARY / "interrupted_request_repair.json", repair)

    resume_inventory_before = inventory_complete_receipts(
        plan,
        cache_roots=[args.resume_cache_root],
        canonical_cache_path=canonical_path,
    )
    overlap = v0_inventory.verified_request_ids.intersection(
        resume_inventory_before.verified_request_ids
    )
    if overlap:
        raise AcquisitionBlocked(
            "blocked_reproducibility_or_audit_failure",
            "a V0-complete logical request was redownloaded into the V0.1 cache",
        )
    existing_orphans = unreceipted_raw_paths(args.resume_cache_root)
    orphan_accounting = raw_page_accounting(existing_orphans)
    existing_resume_records = (
        int(
            pd.to_numeric(
                resume_inventory_before.audit.get("records_returned", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        + orphan_accounting["records"]
    )
    existing_resume_bytes = (
        int(
            pd.to_numeric(
                resume_inventory_before.audit.get("response_bytes", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        + orphan_accounting["bytes"]
    )
    validate_additional_resource_usage(
        provider_records=existing_resume_records,
        raw_bytes=existing_resume_bytes,
        cumulative_records=V0_RECORDS_ENCOUNTERED + existing_resume_records,
    )
    verified_before = frozenset(
        set(v0_inventory.verified_request_ids) | set(resume_inventory_before.verified_request_ids)
    )
    missing_before_network = remaining_resume_requests(plan, verified_before)
    if args.plan_only:
        write_csv(PRIMARY / "remaining_options_gap.csv", missing_gap_frame(missing_before_network))
        print(
            f"verified {len(verified_before)} complete requests; "
            f"{len(missing_before_network)} remain"
        )
        return {
            "status": "planned",
            "complete_receipts_reused": v0_inventory.complete_receipts_reused,
            "requests_remaining": len(missing_before_network),
        }

    token = cast(str, v0.load_token(args.env_file))
    try:
        new_completed, _new_records_current_run = download_remaining(
            missing_before_network,
            token=token,
            resume_cache_root=args.resume_cache_root,
            v0=v0,
            existing_records=existing_resume_records,
            existing_bytes=existing_resume_bytes,
        )
    except OptionsResourceLimitExceeded as error:
        raise AcquisitionBlocked("blocked_resume_resource_limit", str(error)) from error
    except OptionsAuthenticationError as error:
        raise AcquisitionBlocked(
            "blocked_holdout_options_download_failure",
            "EODHD authentication or entitlement failed",
        ) from error
    except OptionsDownloadError as error:
        raise AcquisitionBlocked(
            "blocked_holdout_options_download_failure",
            str(error),
        ) from error

    resume_inventory = inventory_complete_receipts(
        plan,
        cache_roots=[args.resume_cache_root],
        canonical_cache_path=canonical_path,
    )
    overlap = v0_inventory.verified_request_ids.intersection(resume_inventory.verified_request_ids)
    if overlap:
        raise AcquisitionBlocked(
            "blocked_reproducibility_or_audit_failure",
            "a verified V0 request was redownloaded during V0.1",
        )
    verified = frozenset(
        set(v0_inventory.verified_request_ids) | set(resume_inventory.verified_request_ids)
    )
    remaining = remaining_resume_requests(plan, verified)
    write_csv(PRIMARY / "remaining_options_gap.csv", missing_gap_frame(remaining))

    resume_orphans = unreceipted_raw_paths(args.resume_cache_root)
    resume_orphan_accounting = raw_page_accounting(resume_orphans)
    complete_new_records = int(
        pd.to_numeric(
            resume_inventory.audit.get("records_returned", pd.Series(dtype=float)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    complete_new_bytes = int(
        pd.to_numeric(
            resume_inventory.audit.get("response_bytes", pd.Series(dtype=float)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    additional_records = complete_new_records + resume_orphan_accounting["records"]
    additional_bytes = complete_new_bytes + resume_orphan_accounting["bytes"]
    validate_additional_resource_usage(
        provider_records=additional_records,
        raw_bytes=additional_bytes,
        cumulative_records=V0_RECORDS_ENCOUNTERED + additional_records,
    )
    if remaining:
        raise AcquisitionBlocked(
            "blocked_holdout_options_download_failure",
            f"{len(remaining)} frozen requests still lack verified complete receipts",
        )

    receipt_paths = {
        **dict(v0_inventory.receipt_paths),
        **dict(resume_inventory.receipt_paths),
    }
    canonical, request_audit, canonical_totals = materialize_complete_canonical_cache(
        plan,
        receipt_paths=receipt_paths,
        canonical_path=canonical_path,
        v0=v0,
    )
    coverage = request_coverage(request_audit)
    write_csv(PRIMARY / "holdout_options_coverage.csv", coverage)

    repaired_receipt = resume_inventory.receipt_paths.get(interrupted.request_id)
    if repaired_receipt is None:
        raise AcquisitionBlocked(
            "blocked_holdout_options_download_failure",
            "interrupted logical request did not obtain a complete V0.1 receipt",
        )
    repaired_row = resume_inventory.audit.loc[
        resume_inventory.audit["request_id"].astype(str).eq(interrupted.request_id)
    ]
    if len(repaired_row) != 1:
        raise AcquisitionBlocked(
            "blocked_reproducibility_or_audit_failure",
            "repaired interrupted request has ambiguous receipt audit evidence",
        )
    incomplete_payload = cast(dict[str, Any], json.loads(v0_orphans[0].read_text()))
    incomplete_ids = {
        str(item.get("id"))
        for item in cast(list[Any], incomplete_payload["data"])
        if isinstance(item, Mapping)
    }
    planned_by_id = {request_identity(row): row for row in plan}
    repaired_records = load_verified_receipt_records(
        planned_by_id[interrupted.request_id],
        receipt_path=repaired_receipt,
        cache_root=cache_root_for_receipt(repaired_receipt),
        canonical_cache_path=canonical_path,
    )
    repaired_ids = {str(item.get("id")) for item in repaired_records if isinstance(item, Mapping)}
    repair.update(
        {
            "new_pages_downloaded": int(repaired_row.iloc[0]["page_count"]),
            "deduplication_result": {
                "incomplete_page_rows_admitted": 0,
                "complete_request_records": len(repaired_records),
                "provider_identity_overlap_with_excluded_page": len(
                    incomplete_ids.intersection(repaired_ids)
                ),
                "duplicate_rows_materialised": 0,
            },
            "final_receipt_hash": sha256_file(repaired_receipt),
            "final_exact_date_count": int(repaired_row.iloc[0]["exact_date_record_count"]),
            "status": "complete",
        }
    )
    write_json(PRIMARY / "interrupted_request_repair.json", repair)

    new_exact = int(
        pd.to_numeric(resume_inventory.audit["exact_date_record_count"], errors="raise").sum()
    )
    new_extra = (
        int(
            pd.to_numeric(
                resume_inventory.audit["extra_date_rejection_count"], errors="raise"
            ).sum()
        )
        + resume_orphan_accounting["extra"]
    )
    new_protected = (
        int(
            pd.to_numeric(
                resume_inventory.audit["protected_date_rejection_count"], errors="raise"
            ).sum()
        )
        + resume_orphan_accounting["protected"]
    )
    manifest: dict[str, Any] = {
        **SAFETY_FLAGS,
        "status": "complete",
        "endpoint": OPTIONS_EOD_ENDPOINT,
        "compact": False,
        "complete_receipts_found": v0_inventory.complete_receipts_found,
        "complete_receipts_reused": v0_inventory.complete_receipts_reused,
        "complete_receipts_rejected_corrupt": v0_inventory.corrupt_receipts,
        "redundant_network_requests_prevented": v0_inventory.complete_receipts_reused,
        "interrupted_request_repaired": True,
        "interrupted_request_id": interrupted.request_id,
        "incomplete_page_records_admitted": 0,
        "missing_requests_at_resume_start": len(original_missing),
        "new_requests_attempted": len(missing_before_network),
        "new_requests_completed": new_completed,
        "new_requests_failed": 0,
        "new_provider_records": additional_records,
        "new_exact_date_records": new_exact,
        "new_extra_date_records_rejected": new_extra,
        "new_2026_or_later_records_rejected": new_protected,
        "new_bytes_downloaded": additional_bytes,
        "maximum_additional_provider_records": MAXIMUM_ADDITIONAL_RECORDS,
        "maximum_additional_raw_bytes": MAXIMUM_ADDITIONAL_BYTES,
        "maximum_cumulative_records": MAXIMUM_CUMULATIVE_RECORDS,
        "cumulative_complete_requests": len(verified),
        "requests_remaining": len(remaining),
        "cumulative_provider_records": V0_RECORDS_ENCOUNTERED + additional_records,
        "cumulative_complete_receipt_records": (V0_COMPLETE_RECEIPT_RECORDS + complete_new_records),
        "cumulative_unreceipted_records_excluded": (
            V0_INCOMPLETE_PAGE_RECORDS + resume_orphan_accounting["records"]
        ),
        "cumulative_raw_bytes": V0_BYTES_ENCOUNTERED + additional_bytes,
        "exact_date_cache_rows": len(canonical),
        "canonical_cache_path": str(canonical_path),
        "canonical_cache_sha256": sha256_file(canonical_path),
        "request_audit_cache_path": str(canonical_path.with_name("request_audit.parquet")),
        "request_audit_sha256": sha256_file(canonical_path.with_name("request_audit.parquet")),
        "complete_request_redownload_intersections": 0,
        "protected_or_unauthorised_records_materialised": 0,
        "raw_vendor_data_committed": False,
        "canonical_vendor_data_committed": False,
        "provider_records_from_all_complete_receipts": canonical_totals["provider_records"],
        "exact_date_records_from_all_complete_receipts": canonical_totals["exact_date_records"],
        "extra_date_records_rejected_from_all_complete_receipts": canonical_totals[
            "extra_date_records_rejected"
        ],
        "protected_date_records_rejected_from_all_complete_receipts": canonical_totals[
            "protected_date_records_rejected"
        ],
        "canonical_records_accepted_before_deduplication": canonical_totals[
            "canonical_records_accepted"
        ],
        "canonical_records_rejected": canonical_totals["canonical_records_rejected"],
        "nonstandard_contract_records_rejected": canonical_totals[
            "nonstandard_contract_records_rejected"
        ],
        "canonical_duplicate_records": canonical_totals["duplicate_records"],
        "canonical_conflicting_duplicate_groups": canonical_totals["conflicting_duplicate_groups"],
    }
    assert_v01_safety_flags(manifest)
    write_json(PRIMARY / "resume_download_manifest.json", manifest)
    return manifest


def write_blocked_manifest(
    *,
    decision: str,
    detail: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    existing: dict[str, Any] = {}
    path = PRIMARY / "resume_download_manifest.json"
    if path.is_file():
        existing = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    manifest = {
        **SAFETY_FLAGS,
        **existing,
        "status": "blocked",
        "overall_decision": decision,
        "blocker_detail": detail,
        "maximum_additional_provider_records": MAXIMUM_ADDITIONAL_RECORDS,
        "maximum_additional_raw_bytes": MAXIMUM_ADDITIONAL_BYTES,
        "maximum_cumulative_records": MAXIMUM_CUMULATIVE_RECORDS,
        "protected_or_unauthorised_records_materialised": 0,
        "v0_cache_root": str(args.v0_cache_root),
        "resume_cache_root": str(args.resume_cache_root),
    }
    write_json(path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-root", type=Path, default=DEFAULT_PROVIDER_ROOT)
    parser.add_argument("--v0-cache-root", type=Path, default=DEFAULT_V0_CACHE_ROOT)
    parser.add_argument("--resume-cache-root", type=Path, default=DEFAULT_RESUME_CACHE_ROOT)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = run(args)
    except ResumeResourceLimitError as error:
        manifest = write_blocked_manifest(
            decision="blocked_resume_resource_limit",
            detail=str(error),
            args=args,
        )
    except AcquisitionBlocked as error:
        manifest = write_blocked_manifest(
            decision=error.decision,
            detail=error.detail,
            args=args,
        )
    except Exception as error:
        manifest = write_blocked_manifest(
            decision="blocked_holdout_options_download_failure",
            detail=f"{type(error).__name__}: {error}",
            args=args,
        )
    if manifest.get("status") == "complete":
        print(
            f"resume complete: {manifest['cumulative_complete_requests']} requests; "
            f"{manifest['new_provider_records']} additional provider records"
        )
        return 0
    if manifest.get("status") == "planned":
        return 0
    print(str(manifest.get("overall_decision", "blocked")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
