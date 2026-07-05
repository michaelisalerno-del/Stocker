"""Packaged research-only frozen-template technique pipeline.

This module is an orchestration layer around existing Stocker research pieces:
EODHD normalized storage, bar cleaning/validation, state-event detection, and
frozen-template transfer replay. It does not touch broker, paper/live trading,
order placement, deployment, or runtime trading paths.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocker_core.config import EODHDConfig
from stocker_data.catalog import write_catalog
from stocker_data.storage import DatasetKey, dataset_path, read_parquet, write_parquet
from stocker_data.validate import ValidationIssue, validate_ohlcv
from stocker_research.state_event_detector_v0 import (
    StateEventDetectorConfig,
    run_state_event_detector_lab,
)
from stocker_research.template_discovery_system_v0 import (
    TemplateDiscoveryEventInput,
    TemplateDiscoverySystemConfig,
    run_template_discovery_system_lab,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/frozen_template_technique_v0")


@dataclass(frozen=True)
class FrozenTemplateTechniqueConfig:
    """Config for the packaged local research technique."""

    from_date: str
    to_date: str
    timeframe: str = "5m"
    source: str = "eodhd"
    instrument_type: str = "stock"
    currency: str = "USD"
    market_calendar: str | None = "XNYS"
    download_eodhd: bool = False
    merge: bool = True
    overwrite: bool = False
    audit: bool = True
    qa: bool = True
    event_mode: str = "state_entry_non_overlapping"
    random_seed: int = 1337
    min_events_for_similarity: int = 1
    template_universe_profile: str = "liquid_midcap"


@dataclass(frozen=True)
class FrozenTemplateTechniqueResult:
    """Paths and headline result for one packaged technique run."""

    run_id: str
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    fetch_report_csv_path: Path
    bar_cleaner_report_csv_path: Path
    state_event_report_dir: Path
    event_rows_csv_path: Path
    template_transfer_report_dir: Path
    template_transfer_summary_json_path: Path
    decision_json_path: Path
    decision: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _validation_counts(issues: Sequence[ValidationIssue]) -> dict[str, int]:
    counts = {"info": 0, "warning": 0, "error": 0}
    for issue in issues:
        counts[issue.severity] += 1
    return counts


def _clean_bar_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True)
    before = int(len(cleaned))
    cleaned = (
        cleaned.sort_values("timestamp", kind="mergesort")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    return cleaned, before - int(len(cleaned))


def _bar_cleaner_report_row(
    *,
    data_dir: Path,
    symbol: str,
    config: FrozenTemplateTechniqueConfig,
) -> dict[str, Any]:
    key = DatasetKey(
        source=config.source,
        instrument_type=config.instrument_type,
        symbol=symbol.upper(),
        timeframe=config.timeframe,
    )
    path = dataset_path(key, data_dir=data_dir)
    frame = read_parquet(path)
    rows_before = int(len(frame))
    cleaned, duplicate_count = _clean_bar_frame(frame)
    issues = validate_ohlcv(
        cleaned,
        timeframe=config.timeframe,
        timezone="UTC",
        require_timezone=True,
        market_calendar=config.market_calendar,
    )
    counts = _validation_counts(issues)
    if counts["error"]:
        messages = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise ValueError(f"Cleaned bars failed validation for {symbol}: {messages}")
    write_parquet(cleaned, path)
    write_catalog(data_dir=data_dir)
    timestamps = pd.to_datetime(cleaned["timestamp"], utc=True)
    return {
        "symbol": symbol.upper(),
        "dataset_path": str(path),
        "rows_before": rows_before,
        "rows_after": int(len(cleaned)),
        "duplicate_timestamps_removed": duplicate_count,
        "validation_info": counts["info"],
        "validation_warnings": counts["warning"],
        "validation_errors": counts["error"],
        "min_timestamp": str(timestamps.min()) if not timestamps.empty else None,
        "max_timestamp": str(timestamps.max()) if not timestamps.empty else None,
    }


def _fetch_eodhd_rows(
    *,
    data_dir: Path,
    symbols: Sequence[str],
    config: FrozenTemplateTechniqueConfig,
    eodhd_config: EODHDConfig,
) -> pd.DataFrame:
    from stocker_data.vendors import eodhd

    if not config.download_eodhd:
        return pd.DataFrame(
            columns=[
                "symbol",
                "status",
                "rows_fetched",
                "rows_saved",
                "output_path",
                "audit_markdown_path",
                "audit_json_path",
            ]
        )
    if config.timeframe not in {"1m", "5m", "1h"}:
        raise ValueError(
            "frozen template technique EODHD download supports intraday "
            "timeframes 1m, 5m, and 1h"
        )

    client = eodhd.EODHDClient(config=eodhd_config)
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        result = eodhd.fetch_intraday_to_storage(
            client=client,
            data_dir=data_dir,
            symbol=symbol,
            from_date=config.from_date,
            to_date=config.to_date,
            interval=config.timeframe,
            instrument_type=config.instrument_type,
            currency=config.currency,
            save_raw=True,
            overwrite=config.overwrite,
            merge=config.merge,
            audit=config.audit,
            market_calendar=config.market_calendar,
        )
        rows.append(
            {
                "symbol": symbol.upper(),
                "status": "fetched",
                "rows_fetched": result.rows_fetched,
                "rows_saved": result.rows_saved,
                "output_path": str(result.output_path),
                "audit_markdown_path": str(result.audit_markdown_path)
                if result.audit_markdown_path is not None
                else None,
                "audit_json_path": str(result.audit_json_path)
                if result.audit_json_path is not None
                else None,
                "validation_warnings": result.validation_warning_count,
                "validation_errors": result.validation_error_count,
            }
        )
    return pd.DataFrame(rows)


def _run_bar_cleaner(
    *,
    data_dir: Path,
    symbols: Sequence[str],
    config: FrozenTemplateTechniqueConfig,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _bar_cleaner_report_row(
                data_dir=data_dir,
                symbol=symbol,
                config=config,
            )
            for symbol in symbols
        ]
    )


def _write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Frozen Template Technique V0",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Decision: `{payload['decision']}`",
        "- Research-only: true",
        "- Edge claimed: false",
        "- Order placement: disabled",
        "",
        "## Stages",
        "",
        "- EODHD download: "
        + ("enabled" if payload["download_eodhd"] else "skipped"),
        f"- Bar cleaner symbols: {payload['bar_cleaner']['symbols_cleaned']}",
        f"- State-event rows: {payload['state_event_detector']['total_event_rows']}",
        "- Frozen-template transfer rows: "
        f"{payload['template_transfer']['trade_count']}",
        "",
        "## Files",
        "",
        f"- `summary.json`: `{payload['reports']['summary_json']}`",
        f"- `bar_cleaner_report.csv`: `{payload['reports']['bar_cleaner_report']}`",
        f"- `state_event_event_rows.csv`: `{payload['reports']['event_rows']}`",
        "- `template_transfer_summary.json`: "
        f"`{payload['reports']['template_transfer_summary_json']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_frozen_template_technique_v0(
    *,
    data_dir: Path,
    symbols: Sequence[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: FrozenTemplateTechniqueConfig,
    eodhd_config: EODHDConfig | None = None,
) -> FrozenTemplateTechniqueResult:
    """Run the packaged local research technique from EODHD bars to replay."""

    if not symbols:
        raise ValueError("Supply at least one symbol.")
    normalized_symbols = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    if config.overwrite and config.merge:
        raise ValueError("Use either overwrite or merge, not both.")

    run_id = "frozen_template_technique_v0_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fetch_report = _fetch_eodhd_rows(
        data_dir=data_dir,
        symbols=normalized_symbols,
        config=config,
        eodhd_config=eodhd_config or EODHDConfig(),
    )
    bar_cleaner = _run_bar_cleaner(
        data_dir=data_dir,
        symbols=normalized_symbols,
        config=config,
    )

    state_result = run_state_event_detector_lab(
        data_dir=data_dir,
        symbols=normalized_symbols,
        source=config.source,
        instrument_type=config.instrument_type,
        timeframe=config.timeframe,
        market_calendar=config.market_calendar,
        output_dir=run_dir / "state_event_detector_v0",
        config=StateEventDetectorConfig(
            timeframe=config.timeframe,
            market_calendar=config.market_calendar,
            event_mode=config.event_mode,
            random_seed=config.random_seed,
            min_events_for_similarity=config.min_events_for_similarity,
        ),
        manual_examples=[],
    )

    transfer_result = run_template_discovery_system_lab(
        input_event_rows=(
            TemplateDiscoveryEventInput(
                f"{config.template_universe_profile}_event_rows",
                state_result.event_rows_csv_path,
            ),
        ),
        output_dir=run_dir / "template_transfer_replay",
        config=TemplateDiscoverySystemConfig(
            mode="frozen-template-transfer-replay",
            universe_profile=config.template_universe_profile,
        ),
    )

    transfer_summary = json.loads(
        transfer_result.summary_json_path.read_text(encoding="utf-8")
    )
    decision = "continue_research_packaged_frozen_template_technique"
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_markdown": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "fetch_report": run_dir / "fetch_report.csv",
        "bar_cleaner_report": run_dir / "bar_cleaner_report.csv",
    }
    _write_csv(paths["fetch_report"], fetch_report)
    _write_csv(paths["bar_cleaner_report"], bar_cleaner)

    payload = {
        "run_id": run_id,
        "mode": "frozen_template_technique_v0",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "yaml_rules_saved": False,
        "download_eodhd": config.download_eodhd,
        "symbols": list(normalized_symbols),
        "data_dir": str(data_dir),
        "decision": decision,
        "stages": [
            "optional_eodhd_intraday_download",
            "canonical_bar_clean_and_validate",
            "state_event_detector_v0",
            "frozen_template_transfer_replay",
        ],
        "bar_cleaner": {
            "symbols_cleaned": int(len(bar_cleaner)),
            "rows_before": int(bar_cleaner["rows_before"].sum()),
            "rows_after": int(bar_cleaner["rows_after"].sum()),
            "duplicate_timestamps_removed": int(
                bar_cleaner["duplicate_timestamps_removed"].sum()
            ),
            "validation_errors": int(bar_cleaner["validation_errors"].sum()),
            "validation_warnings": int(bar_cleaner["validation_warnings"].sum()),
        },
        "state_event_detector": {
            "run_id": state_result.run_id,
            "output_dir": str(state_result.output_dir),
            "event_rows_csv_path": str(state_result.event_rows_csv_path),
            "total_event_rows": int(state_result.total_event_rows),
            "symbols_completed": state_result.symbols_completed,
            "symbols_failed": state_result.symbols_failed,
            "decision": state_result.decision,
        },
        "template_transfer": {
            "run_id": transfer_result.run_id,
            "output_dir": str(transfer_result.output_dir),
            "summary_json_path": str(transfer_result.summary_json_path),
            "decision": transfer_result.decision,
            "all_row_count": int(
                transfer_summary.get("frozen_template_transfer_all_row_count", 0)
            ),
            "trade_count": int(
                transfer_summary.get("frozen_template_transfer_trade_count", 0)
            ),
            "total_r": transfer_summary.get("frozen_template_transfer_total_r"),
        },
        "reports": {
            "summary_json": str(paths["summary_json"]),
            "summary_markdown": str(paths["summary_markdown"]),
            "decision_json": str(paths["decision_json"]),
            "fetch_report": str(paths["fetch_report"]),
            "bar_cleaner_report": str(paths["bar_cleaner_report"]),
            "state_event_summary_json": str(state_result.summary_json_path),
            "event_rows": str(state_result.event_rows_csv_path),
            "template_transfer_summary_json": str(transfer_result.summary_json_path),
        },
        "config": config.__dict__,
    }
    _write_json(paths["summary_json"], payload)
    _write_json(
        paths["decision_json"],
        {
            "decision": decision,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "edge_claimed": False,
            "yaml_rules_saved": False,
        },
    )
    _write_summary_md(paths["summary_markdown"], payload)

    return FrozenTemplateTechniqueResult(
        run_id=run_id,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_markdown"],
        fetch_report_csv_path=paths["fetch_report"],
        bar_cleaner_report_csv_path=paths["bar_cleaner_report"],
        state_event_report_dir=state_result.output_dir,
        event_rows_csv_path=state_result.event_rows_csv_path,
        template_transfer_report_dir=transfer_result.output_dir,
        template_transfer_summary_json_path=transfer_result.summary_json_path,
        decision_json_path=paths["decision_json"],
        decision=decision,
    )


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "FrozenTemplateTechniqueConfig",
    "FrozenTemplateTechniqueResult",
    "run_frozen_template_technique_v0",
]
