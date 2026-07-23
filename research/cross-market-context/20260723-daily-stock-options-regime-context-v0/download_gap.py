#!/usr/bin/env python3
"""Fill only the frozen exact-date option-surface gaps within the quick-run caps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
GAP_PATH = PRIMARY / "daily_options_coverage_gap.csv"
RAW_FEATURE_PATH = PRIMARY / "daily_options_raw_features.parquet"
SOURCE_MANIFEST_PATH = PRIMARY / "source_manifest.json"
DOWNLOAD_ROOT = (
    REPO_ROOT / "data" / "vendor" / "eodhd" / "options" / "daily-stock-options-context-v0"
)
RECEIPT_PATH = DOWNLOAD_ROOT / "bounded_download_receipt.json"
OUTPUT_CACHE = DOWNLOAD_ROOT / "canonical" / "exact-date-cache.parquet"
MAXIMUM_RECORDS = 500_000
MAXIMUM_BYTES = 5 * 1024**3
PROTECTED_START = date(2025, 8, 23)

for package in ("stocker_research", "stocker_data"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))

from stocker_research.daily_stock_options_context_v0 import SAFETY_FLAGS  # noqa: E402
from stocker_research.eodhd_options_downloader_v0 import (  # noqa: E402
    DownloadConfig,
    EODHDOptionsDownloader,
    OptionsDownloadError,
    OptionsRequest,
    OptionsResourceLimitExceeded,
    canonicalize_response_records,
)


class RequestsTransport:
    """Small requests adapter for the bounded downloader."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def get(self, url: str, *, params: dict[str, object], timeout: float) -> requests.Response:
        return self.session.get(url, params=params, timeout=timeout, stream=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_plan_id(symbol: str, observation_date: date, components: tuple[str, ...]) -> str:
    value = f"{symbol}|{observation_date.isoformat()}|{'|'.join(components)}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_plans(gaps: pd.DataFrame) -> list[dict[str, Any]]:
    """Collapse component gaps into one exact-day 7–90 DTE request per stock-date."""

    required = gaps.loc[gaps["bounded_download_required"].astype(bool)].copy()
    plans: list[dict[str, Any]] = []
    for (symbol, observation_value), group in required.groupby(
        ["symbol", "required_options_date"], sort=True, observed=True
    ):
        observation_date = date.fromisoformat(str(observation_value))
        if observation_date >= PROTECTED_START:
            raise ValueError("protected option observation entered the gap plan")
        components = tuple(sorted(set(group["gap_component"].astype(str))))
        needs_front = any(
            component in {"front_atm_pair", "front_25_delta_skew", "front_expiry_open_interest"}
            for component in components
        )
        minimum_dte = 7 if needs_front else 46
        plans.append(
            {
                "plan_id": stable_plan_id(str(symbol), observation_date, components),
                "symbol": str(symbol),
                "observation_date": observation_date,
                "components": components,
                "request": OptionsRequest(
                    underlying_symbol=str(symbol),
                    trade_date_from=observation_date,
                    trade_date_to=observation_date,
                    expiration_from=observation_date + timedelta(days=minimum_dte),
                    expiration_to=observation_date + timedelta(days=90),
                    compact=True,
                ),
            }
        )
    return plans


def record_receipt(receipt: Mapping[str, Any]) -> None:
    """Persist a credential-free receipt beside both ignored cache and artifacts."""

    write_json(RECEIPT_PATH, receipt)
    write_json(PRIMARY / "download_gap_receipt.json", receipt)
    if SOURCE_MANIFEST_PATH.is_file():
        source = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
        if isinstance(source, dict):
            source["bounded_download"] = {
                "status": receipt["status"],
                "planned_exact_stock_date_requests": receipt["planned_exact_stock_date_requests"],
                "planned_gap_rows": receipt["planned_gap_rows"],
                "network_requests_made": receipt["network_requests_made"],
                "newly_downloaded_records": receipt["newly_downloaded_records"],
                "newly_downloaded_bytes": receipt["newly_downloaded_bytes"],
                "credential_recorded": False,
            }
            write_json(SOURCE_MANIFEST_PATH, source)


def _normalise_exact_records(
    records: list[dict[str, Any]],
    *,
    observation_date: date,
    request_id: str,
) -> list[dict[str, Any]]:
    exact: list[dict[str, Any]] = []
    for record in records:
        resource_id = record.get("id")
        if not isinstance(resource_id, str) or resource_id[-10:] != observation_date.isoformat():
            continue
        attributes_value = record.get("attributes")
        if not isinstance(attributes_value, Mapping):
            continue
        attributes = dict(attributes_value)
        if any(attributes.get(field) is None for field in ("bid", "ask", "bid_date", "ask_date")):
            continue
        expiration_value = attributes.get("exp_date")
        if isinstance(expiration_value, str):
            try:
                expiration_date = date.fromisoformat(expiration_value[:10])
            except ValueError:
                expiration_date = observation_date
            provider_dte = attributes.get("dte")
            if (
                isinstance(provider_dte, (int, float))
                and not isinstance(provider_dte, bool)
                and float(provider_dte).is_integer()
                and int(provider_dte) != (expiration_date - observation_date).days
            ):
                attributes["dte"] = None
        exact.append({**record, "attributes": attributes})
    canonical = canonicalize_response_records(
        exact,
        request_id=request_id,
        provider_schema_version="openapi-2.0.0-daily-context-v0",
    )
    if canonical.rejections:
        raise ValueError(
            f"canonical option records rejected for plan {request_id}: {len(canonical.rejections)}"
        )
    output: list[dict[str, Any]] = []
    for row in canonical.records:
        if cast(date, row["trade_date"]) >= PROTECTED_START:
            raise ValueError("protected option observation survived exact-date filtering")
        output.append(
            {
                **row,
                "adjusted_contract": False,
                "deliverable_resolved": True,
                "contract_multiplier": 100,
                "settlement_style": "physical",
                "chain_complete": True,
                "cache_source": "daily_stock_options_context_v0",
                "request_strategy": "daily_context_surface",
            }
        )
    return output


def run(*, token: str) -> dict[str, Any]:
    if not GAP_PATH.is_file() or not RAW_FEATURE_PATH.is_file():
        raise FileNotFoundError("run_screen_v0.py must create the exact gap manifest first")
    gaps = pd.read_csv(GAP_PATH)
    plans = build_plans(gaps)
    base = {
        **SAFETY_FLAGS,
        "maximum_additional_records": MAXIMUM_RECORDS,
        "maximum_additional_bytes": MAXIMUM_BYTES,
        "planned_exact_stock_date_requests": len(plans),
        "planned_gap_rows": int(gaps["bounded_download_required"].astype(bool).sum()),
        "credential_recorded": False,
        "protected_option_observations_materialised": 0,
    }
    if not plans:
        receipt = {
            **base,
            "status": "not_required",
            "network_requests_made": 0,
            "newly_downloaded_records": 0,
            "newly_downloaded_bytes": 0,
        }
        record_receipt(receipt)
        return receipt
    if not token:
        receipt = {
            **base,
            "status": "token_unavailable",
            "network_requests_made": 0,
            "newly_downloaded_records": 0,
            "newly_downloaded_bytes": 0,
            "output_cache": None,
        }
        record_receipt(receipt)
        return receipt

    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token=token,
            data_dir=DOWNLOAD_ROOT,
            maximum_raw_records=MAXIMUM_RECORDS,
            maximum_download_bytes=MAXIMUM_BYTES,
            requests_per_minute=600,
        ),
        transport=RequestsTransport(),
    )
    canonical_rows: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    status = "completed"
    error_detail: str | None = None
    for plan in plans:
        request = cast(OptionsRequest, plan["request"])
        try:
            result = downloader.download(request)
            exact = _normalise_exact_records(
                result.records,
                observation_date=cast(date, plan["observation_date"]),
                request_id=str(plan["plan_id"]),
            )
        except OptionsResourceLimitExceeded as error:
            status = "blocked_quick_resource_limit"
            error_detail = str(error).replace(token, "[REDACTED]")
            break
        except (OptionsDownloadError, ValueError) as error:
            status = "download_failure"
            error_detail = str(error).replace(token, "[REDACTED]")
            break
        canonical_rows.extend(exact)
        completed.append(
            {
                "plan_id": plan["plan_id"],
                "symbol": plan["symbol"],
                "observation_date": plan["observation_date"],
                "components": plan["components"],
                "provider_records": len(result.records),
                "exact_canonical_records": len(exact),
            }
        )

    downloaded_records = sum(row["provider_records"] for row in completed)
    downloaded_bytes = sum(path.stat().st_size for path in (DOWNLOAD_ROOT / "raw").glob("*.json"))
    if canonical_rows:
        inherited_path = Path(
            str(
                cast(
                    Mapping[str, Any],
                    json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))["sources"],
                )["repaired_exact_date_options_cache"]
            )
        )
        inherited = pd.read_parquet(inherited_path)
        combined = pd.concat([inherited, pd.DataFrame(canonical_rows)], ignore_index=True)
        combined = combined.sort_values(
            ["underlying_symbol", "trade_date", "expiration_date", "strike", "option_type"],
            kind="mergesort",
        ).drop_duplicates(["underlying_symbol", "contract_id", "trade_date"], keep="last")
        OUTPUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(OUTPUT_CACHE, index=False)
    receipt = {
        **base,
        "status": status,
        "network_requests_made": len(completed),
        "newly_downloaded_records": downloaded_records,
        "newly_downloaded_bytes": downloaded_bytes,
        "completed_requests": completed,
        "error_detail": error_detail,
        "output_cache": str(OUTPUT_CACHE) if OUTPUT_CACHE.is_file() else None,
    }
    record_receipt(receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-env",
        default="EODHD_API_TOKEN",
        help="Environment variable containing the EODHD token; its value is never recorded.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    receipt = run(token=os.environ.get(arguments.token_env, ""))
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "planned_exact_stock_date_requests": receipt["planned_exact_stock_date_requests"],
                "newly_downloaded_records": receipt["newly_downloaded_records"],
                "newly_downloaded_bytes": receipt["newly_downloaded_bytes"],
                "credential_recorded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
