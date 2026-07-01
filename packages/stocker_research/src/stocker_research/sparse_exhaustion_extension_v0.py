"""Formalize sparse exhaustion-extension candidate events.

This research-only report consumes an existing local candidate-personality scan
and extracts the ``exhaustion_extension`` rows into a standalone sparse event
artifact. It does not fetch data, touch broker/execution paths, or place orders.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path("data/reports/research/sparse_exhaustion_extension_v0")


@dataclass(frozen=True)
class SparseExhaustionExtensionResult:
    """Paths and headline decision for the sparse exhaustion-extension report."""

    run_id: str
    input_candidate_report_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    exhaustion_event_rows_csv_path: Path
    exhaustion_horizon_summary_csv_path: Path
    selected_horizons_csv_path: Path
    decision: str
    event_count: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "No rows."
    shown = frame.head(max_rows)
    lines = [
        "| " + " | ".join(str(column) for column in shown.columns) + " |",
        "| " + " | ".join("---" for _ in shown.columns) + " |",
    ]
    for _, row in shown.iterrows():
        values: list[str] = []
        for column in shown.columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _load_candidate_events(input_candidate_report_dir: Path) -> pd.DataFrame:
    path = input_candidate_report_dir / "candidate_event_rows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate event rows: {path}")
    rows = pd.read_csv(path)
    if rows.empty:
        return rows
    required = {"symbol", "timestamp", "session_date", "candidate_personality"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"candidate_event_rows.csv missing required columns: {missing}")
    return rows


def _extract_exhaustion_events(candidate_events: pd.DataFrame) -> pd.DataFrame:
    if candidate_events.empty:
        return pd.DataFrame()
    events = candidate_events[
        candidate_events["candidate_personality"].astype(str).eq("exhaustion_extension")
    ].copy()
    if events.empty:
        return events
    events["event_state"] = "exhaustion_extension"
    events["event_family"] = "extension_exhaustion"
    expected_direction = pd.to_numeric(
        events.get("expected_direction", pd.Series(0, index=events.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    events["event_direction"] = np.select(
        [expected_direction > 0, expected_direction < 0],
        ["long_reversal_or_no_chase", "short_reversal_or_no_chase"],
        default="no_chase",
    )
    events["event_role"] = events.get("candidate_role", "mean_reversion_or_no_chase")
    events["source_candidate_personality"] = events["candidate_personality"]
    return events.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)


def _horizon_summary(exhaustion_events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if exhaustion_events.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                "event_count",
                "symbol_count",
                "session_count",
                "median_aligned_return",
                "same_result_rate",
            ]
        )
    direction = pd.to_numeric(exhaustion_events.get("expected_direction", 0), errors="coerce")
    for horizon in (6, 9, 12, 24):
        ret_col = f"forward_{horizon}_bar_return"
        if ret_col not in exhaustion_events:
            continue
        returns = pd.to_numeric(exhaustion_events[ret_col], errors="coerce")
        valid = exhaustion_events[returns.notna()].copy()
        aligned = returns[returns.notna()] * direction[returns.notna()]
        rows.append(
            {
                "horizon": horizon,
                "event_count": int(len(valid)),
                "symbol_count": int(valid["symbol"].astype(str).nunique()),
                "session_count": int(valid["session_date"].astype(str).nunique()),
                "median_aligned_return": float(aligned.median()) if not aligned.empty else math.nan,
                "same_result_rate": (
                    float((aligned > 0.0).mean()) if not aligned.empty else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_selected_horizons(input_candidate_report_dir: Path) -> pd.DataFrame:
    path = input_candidate_report_dir / "passed_candidate_personality_horizons.csv"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    if rows.empty or "candidate_personality" not in rows:
        return pd.DataFrame()
    return rows[rows["candidate_personality"].astype(str).eq("exhaustion_extension")].copy()


def _write_summary_md(
    path: Path,
    *,
    summary: dict[str, Any],
    horizon_summary: pd.DataFrame,
    selected_horizons: pd.DataFrame,
) -> None:
    lines = [
        "# Sparse Exhaustion Extension V0",
        "",
        f"Decision: `{summary['decision']}`",
        f"Input candidate report: `{summary['input_candidate_report_dir']}`",
        f"Event count: `{summary['event_count']}`",
        "Research-only: `True`",
        "Live ordering enabled: `False`",
        "Order placement: `disabled`",
        "Edge claimed: `False`",
        "",
        "## Horizon Summary",
        "",
        _markdown_table(horizon_summary),
        "",
        "## Selected Candidate Horizons",
        "",
        _markdown_table(selected_horizons),
        "",
        "## Interpretation",
        "",
        (
            "This report formalizes the existing sparse candidate scan. It does not "
            "promote exhaustion into execution or the main selected-filter book. The "
            "next step is to rebuild this detector directly inside the event-detector "
            "pipeline if the candidate remains useful after role-aware replay."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sparse_exhaustion_extension_lab(
    *,
    input_candidate_report_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> SparseExhaustionExtensionResult:
    """Create a standalone sparse exhaustion-extension candidate report."""

    candidate_events = _load_candidate_events(input_candidate_report_dir)
    exhaustion_events = _extract_exhaustion_events(candidate_events)
    horizon_summary = _horizon_summary(exhaustion_events)
    selected_horizons = _load_selected_horizons(input_candidate_report_dir)
    decision = (
        "continue_research_sparse_exhaustion_extension"
        if not exhaustion_events.empty and not selected_horizons.empty
        else "reject_sparse_exhaustion_extension_not_ready"
    )
    run_id = "sparse_exhaustion_extension_v0_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision": run_dir / "decision.json",
        "events": run_dir / "exhaustion_event_rows.csv",
        "horizon": run_dir / "exhaustion_horizon_summary.csv",
        "selected": run_dir / "selected_horizons.csv",
    }
    summary = {
        "run_id": run_id,
        "input_candidate_report_dir": str(input_candidate_report_dir),
        "output_dir": str(run_dir),
        "decision": decision,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "volume_label": (
            "historical_volume from existing local candidate personality scan; "
            "no new vendor fetch"
        ),
        "event_count": int(len(exhaustion_events)),
        "selected_horizon_count": int(len(selected_horizons)),
    }
    _write_csv(paths["events"], exhaustion_events)
    _write_csv(paths["horizon"], horizon_summary)
    _write_csv(paths["selected"], selected_horizons)
    _write_json(paths["summary_json"], summary)
    _write_json(paths["decision"], summary)
    _write_summary_md(
        paths["summary_md"],
        summary=summary,
        horizon_summary=horizon_summary,
        selected_horizons=selected_horizons,
    )
    return SparseExhaustionExtensionResult(
        run_id=run_id,
        input_candidate_report_dir=input_candidate_report_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision"],
        exhaustion_event_rows_csv_path=paths["events"],
        exhaustion_horizon_summary_csv_path=paths["horizon"],
        selected_horizons_csv_path=paths["selected"],
        decision=decision,
        event_count=int(len(exhaustion_events)),
    )


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "SparseExhaustionExtensionResult",
    "run_sparse_exhaustion_extension_lab",
]
