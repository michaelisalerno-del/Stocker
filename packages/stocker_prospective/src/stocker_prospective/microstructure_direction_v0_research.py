"""Reproducible research evaluation for Microstructure Direction V0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from stocker_prospective.microstructure_direction_v0 import (
    RESEARCH_LABEL_V0,
    DirectionActionV0,
    DirectionMethodV0,
    MicrostructureDirectionResultV0,
)

HORIZONS_MINUTES_V0 = (5, 10, 15)
PRIMARY_HORIZON_MINUTES_V0 = 15
PRIMARY_TIMING_WINDOW_V0 = "T0_to_T+15s"
DECISION_D = "D_INSUFFICIENT_PROSPECTIVE_MICROSTRUCTURE_DATA"

WINDOW_NAMES_V0 = (
    "T-60s_to_T0",
    "T-30s_to_T0",
    "T-15s_to_T0",
    "T-5s_to_T0",
    "T0_to_T+1s",
    "T0_to_T+5s",
    "T0_to_T+15s",
    "T0_to_T+30s",
    "T0_to_T+60s",
)
BASELINES_V0 = (
    "FIRST_BAR_SIGN",
    "REFERENCE_RANGE_BREAK",
    "A1",
    "C1",
    "R1",
    "UNCONDITIONAL_MAJORITY_DIRECTION",
)


class RealizedDirectionV0(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


class PricePointV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    received_timestamp_utc: datetime
    midpoint: float
    event_id: str


class ForwardDirectionOutcomeV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    run_id: str
    symbol: str
    direction_method: DirectionMethodV0
    window_name: str
    action: DirectionActionV0
    information_cutoff_utc: datetime
    entry_event_id: str
    entry_timestamp_utc: datetime
    entry_delay_seconds: float
    entry_midpoint: float
    horizon_minutes: int
    target_timestamp_utc: datetime
    terminal_event_id: str
    terminal_timestamp_utc: datetime
    terminal_midpoint: float
    forward_log_return: float
    realized_direction: RealizedDirectionV0
    direction_correct: bool | None
    signed_forward_log_return: float | None
    signed_mfe: float | None
    signed_mae: float | None
    confirmation_delay_seconds: float
    m1c_genuine_forward_move: bool | None


def evaluate_forward_outcomes_v0(
    *,
    direction: MicrostructureDirectionResultV0,
    price_points: tuple[PricePointV0, ...],
    m1c_genuine_forward_move: bool | None = None,
) -> tuple[ForwardDirectionOutcomeV0, ...]:
    """Evaluate gross underlying direction from the first post-cutoff midpoint."""

    ordered = tuple(
        sorted(
            (
                point
                for point in price_points
                if math.isfinite(point.midpoint) and point.midpoint > 0.0
            ),
            key=lambda point: (point.received_timestamp_utc, point.event_id),
        )
    )
    entry = next(
        (
            point
            for point in ordered
            if point.received_timestamp_utc > direction.information_cutoff_utc
        ),
        None,
    )
    if entry is None:
        return ()
    outcomes: list[ForwardDirectionOutcomeV0] = []
    for horizon in HORIZONS_MINUTES_V0:
        target = entry.received_timestamp_utc + timedelta(minutes=horizon)
        terminal = next(
            (point for point in ordered if point.received_timestamp_utc >= target),
            None,
        )
        if terminal is None:
            continue
        path = tuple(
            point
            for point in ordered
            if entry.received_timestamp_utc
            <= point.received_timestamp_utc
            <= terminal.received_timestamp_utc
        )
        forward_return = math.log(terminal.midpoint / entry.midpoint)
        realized = (
            RealizedDirectionV0.UP
            if forward_return > 0.0
            else RealizedDirectionV0.DOWN
            if forward_return < 0.0
            else RealizedDirectionV0.FLAT
        )
        direction_sign = (
            1.0
            if direction.action is DirectionActionV0.UP
            else -1.0
            if direction.action is DirectionActionV0.DOWN
            else None
        )
        signed_path = (
            tuple(direction_sign * math.log(point.midpoint / entry.midpoint) for point in path)
            if direction_sign is not None
            else ()
        )
        outcomes.append(
            ForwardDirectionOutcomeV0(
                episode_id=direction.episode_id,
                run_id=direction.run_id,
                symbol=direction.symbol,
                direction_method=direction.direction_method,
                window_name=direction.window_name,
                action=direction.action,
                information_cutoff_utc=direction.information_cutoff_utc,
                entry_event_id=entry.event_id,
                entry_timestamp_utc=entry.received_timestamp_utc,
                entry_delay_seconds=(
                    entry.received_timestamp_utc - direction.information_cutoff_utc
                ).total_seconds(),
                entry_midpoint=entry.midpoint,
                horizon_minutes=horizon,
                target_timestamp_utc=target,
                terminal_event_id=terminal.event_id,
                terminal_timestamp_utc=terminal.received_timestamp_utc,
                terminal_midpoint=terminal.midpoint,
                forward_log_return=forward_return,
                realized_direction=realized,
                direction_correct=(
                    None
                    if direction_sign is None
                    else (
                        (
                            direction.action is DirectionActionV0.UP
                            and realized is RealizedDirectionV0.UP
                        )
                        or (
                            direction.action is DirectionActionV0.DOWN
                            and realized is RealizedDirectionV0.DOWN
                        )
                    )
                ),
                signed_forward_log_return=(
                    None if direction_sign is None else direction_sign * forward_return
                ),
                signed_mfe=None if not signed_path else max(signed_path),
                signed_mae=None if not signed_path else min(signed_path),
                confirmation_delay_seconds=direction.confirmation_delay_seconds,
                m1c_genuine_forward_move=m1c_genuine_forward_move,
            )
        )
    return tuple(outcomes)


def _mean(values: Iterable[float]) -> float | None:
    observed = tuple(values)
    return None if not observed else statistics.fmean(observed)


def _median(values: Iterable[float]) -> float | None:
    observed = tuple(values)
    return None if not observed else statistics.median(observed)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return centre - half, centre + half


def summarise_method_results_v0(
    *,
    rows: tuple[ForwardDirectionOutcomeV0, ...],
    eligible_episode_count: int,
    abstain_episode_count: int,
) -> dict[str, Any]:
    """Summarise one period × method × timing window × horizon cell."""

    resolved = tuple(row for row in rows if row.action is not DirectionActionV0.ABSTAIN)
    correct = sum(row.direction_correct is True for row in resolved)
    predictions = (RealizedDirectionV0.UP, RealizedDirectionV0.DOWN)
    precision: dict[RealizedDirectionV0, float | None] = {}
    recall: dict[RealizedDirectionV0, float | None] = {}
    for label in predictions:
        predicted = sum(row.action.value == label.value for row in resolved)
        actual = sum(row.realized_direction is label for row in resolved)
        true_positive = sum(
            row.action.value == label.value and row.realized_direction is label for row in resolved
        )
        precision[label] = _ratio(true_positive, predicted)
        recall[label] = _ratio(true_positive, actual)
    balanced_values = tuple(value for value in recall.values() if value is not None)
    interval_low, interval_high = _wilson_interval(correct, len(resolved))
    signed_returns = tuple(
        row.signed_forward_log_return
        for row in resolved
        if row.signed_forward_log_return is not None
    )
    joint_labels_complete = bool(resolved) and all(
        row.m1c_genuine_forward_move is not None for row in resolved
    )
    useful_joint_count = (
        sum(
            row.m1c_genuine_forward_move is True and row.direction_correct is True
            for row in resolved
        )
        if joint_labels_complete
        else None
    )
    return {
        "eligible_m1c_episodes": eligible_episode_count,
        "direction_resolved_n": len(resolved),
        "abstain_n": abstain_episode_count,
        "coverage": _ratio(len(resolved), eligible_episode_count),
        "median_confirmation_delay_seconds": _median(
            row.confirmation_delay_seconds for row in resolved
        ),
        "direction_accuracy": _ratio(correct, len(resolved)),
        "balanced_accuracy": _mean(balanced_values),
        "up_precision": precision[RealizedDirectionV0.UP],
        "down_precision": precision[RealizedDirectionV0.DOWN],
        "up_recall": recall[RealizedDirectionV0.UP],
        "down_recall": recall[RealizedDirectionV0.DOWN],
        "mean_signed_forward_log_return": _mean(signed_returns),
        "median_signed_forward_log_return": _median(signed_returns),
        "mean_signed_mfe": _mean(row.signed_mfe for row in resolved if row.signed_mfe is not None),
        "mean_signed_mae": _mean(row.signed_mae for row in resolved if row.signed_mae is not None),
        "proportion_positive_signed_returns": _ratio(
            sum(value > 0.0 for value in signed_returns), len(signed_returns)
        ),
        "accuracy_ci95_low": interval_low,
        "accuracy_ci95_high": interval_high,
        "useful_joint_rate": (
            None if useful_joint_count is None else _ratio(useful_joint_count, len(resolved))
        ),
        "joint_rate_over_all_m1c_episodes": (
            None
            if useful_joint_count is None
            else _ratio(useful_joint_count, eligible_episode_count)
        ),
    }


METHOD_SUMMARY_FIELDS = (
    "period",
    "method",
    "timing_window",
    "horizon_minutes",
    "eligible_m1c_episodes",
    "direction_resolved_n",
    "abstain_n",
    "coverage",
    "median_confirmation_delay_seconds",
    "direction_accuracy",
    "balanced_accuracy",
    "up_precision",
    "down_precision",
    "up_recall",
    "down_recall",
    "mean_signed_forward_log_return",
    "median_signed_forward_log_return",
    "mean_signed_mfe",
    "mean_signed_mae",
    "proportion_positive_signed_returns",
    "accuracy_ci95_low",
    "accuracy_ci95_high",
    "useful_joint_rate",
    "joint_rate_over_all_m1c_episodes",
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_census_only_research_v0(
    *,
    census_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Emit the honest insufficient-data artifact without inventing episode rows."""

    census_bytes = census_path.read_bytes()
    census = json.loads(census_bytes)
    databases = tuple(census.get("databases", ()))
    runtime_capacity = census.get("runtime_capacity", {})
    eligible = sum(int(item.get("eligible_m1c_episodes", 0)) for item in databases)
    micro_episodes = sum(int(item.get("microstructure_episodes", 0)) for item in databases)
    micro_rows = sum(int(item.get("episode_microstructure_rows", 0)) for item in databases)
    first_dates = sorted(
        str(item["first_episode_date"]) for item in databases if item.get("first_episode_date")
    )
    last_dates = sorted(
        str(item["last_episode_date"]) for item in databases if item.get("last_episode_date")
    )
    decision = {
        "decision": DECISION_D,
        "reason": "No recorded valid M1C episode has episode-linked microstructure evidence.",
        "eligible_m1c_episodes": eligible,
        "microstructure_episodes": micro_episodes,
        "episode_microstructure_rows": micro_rows,
        "development_start_date": first_dates[0] if first_dates else None,
        "development_end_date": last_dates[-1] if last_dates else None,
        "assessment_start_date": None,
        "assessment_end_date": None,
        "assessment_claimed": False,
        "research_only": True,
        "validated": False,
        "recommendation": False,
        "research_label": RESEARCH_LABEL_V0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    empty_metrics = summarise_method_results_v0(
        rows=(),
        eligible_episode_count=eligible,
        abstain_episode_count=eligible,
    )
    method_rows = [
        {
            "period": "descriptive_development_only",
            "method": method.value,
            "timing_window": window,
            "horizon_minutes": horizon,
            **empty_metrics,
        }
        for method in DirectionMethodV0
        for window in WINDOW_NAMES_V0
        for horizon in HORIZONS_MINUTES_V0
    ]
    _write_csv(
        output_root / "episode_level_results.csv",
        (
            "period",
            "episode_id",
            "run_id",
            "symbol",
            "session_date",
            "method",
            "timing_window",
            "information_cutoff_utc",
            "entry_timestamp_utc",
            "entry_delay_seconds",
            "horizon_minutes",
            "action",
            "realized_direction",
            "forward_log_return",
            "signed_forward_log_return",
            "signed_mfe",
            "signed_mae",
            "direction_correct",
            "m1c_genuine_forward_move",
            "useful_joint",
        ),
        (),
    )
    _write_csv(output_root / "method_summary.csv", METHOD_SUMMARY_FIELDS, method_rows)
    _write_csv(
        output_root / "timing_window_comparison.csv",
        METHOD_SUMMARY_FIELDS,
        method_rows,
    )
    baseline_fields = (
        "period",
        "baseline",
        "horizon_minutes",
        "eligible_m1c_episodes",
        "direction_resolved_n",
        "coverage",
        "direction_accuracy",
        "status",
    )
    _write_csv(
        output_root / "baseline_comparison.csv",
        baseline_fields,
        (
            {
                "period": "descriptive_development_only",
                "baseline": baseline,
                "horizon_minutes": PRIMARY_HORIZON_MINUTES_V0,
                "eligible_m1c_episodes": eligible,
                "direction_resolved_n": 0,
                "coverage": None,
                "direction_accuracy": None,
                "status": "INSUFFICIENT_DATA",
            }
            for baseline in BASELINES_V0
        ),
    )
    _write_csv(
        output_root / "first_bar_agreement_analysis.csv",
        (
            "period",
            "method",
            "timing_window",
            "relationship",
            "episodes",
            "direction_accuracy",
            "mean_signed_forward_log_return",
        ),
        (),
    )
    _write_csv(
        output_root / "data_quality_summary.csv",
        (
            "database_identity",
            "eligible_m1c_episodes",
            "microstructure_episodes",
            "episode_microstructure_rows",
            "first_episode_date",
            "last_episode_date",
            "tick_stream_status",
            "depth_status",
        ),
        databases,
    )
    _write_csv(output_root / "assessment_results.csv", METHOD_SUMMARY_FIELDS, ())
    manifest = {
        "experiment": "M1C MICROSTRUCTURE DIRECTION V0",
        "generated_at_utc": census.get("observed_at_utc"),
        "source_label": census.get("source_label"),
        "source_census_sha256": hashlib.sha256(census_bytes).hexdigest(),
        "methods": [method.value for method in DirectionMethodV0],
        "windows": list(WINDOW_NAMES_V0),
        "horizons_minutes": list(HORIZONS_MINUTES_V0),
        "primary_horizon_minutes_from_causal_entry": PRIMARY_HORIZON_MINUTES_V0,
        "primary_display_window": PRIMARY_TIMING_WINDOW_V0,
        "entry_rule": "first valid underlying midpoint strictly after information cutoff",
        "return_definition": "gross log return from causal entry midpoint",
        "split_rule": (
            "chronological assessment requires at least 20 independent episode dates, "
            "100 episodes overall, 5 assessment dates, and 30 assessment episodes"
        ),
        "assessment_claimed": False,
        "level_ii_primary": False,
        "runtime_capacity": runtime_capacity,
        "research_label": RESEARCH_LABEL_V0,
    }
    _write_json(output_root / "run_manifest.json", manifest)
    _write_json(output_root / "decision.json", decision)
    reproduce_command = (
        "PYTHONPATH=packages/stocker_prospective/src python3 -m "
        "stocker_prospective.microstructure_direction_v0_research --census-json "
        "research/directional-readiness/20260813-m1c-microstructure-direction-v0/"
        "source_census.json --output research/directional-readiness/"
        "20260813-m1c-microstructure-direction-v0/artifacts/primary"
    )
    report = f"""# M1C Microstructure Direction V0

{RESEARCH_LABEL_V0}

## Result

`{DECISION_D}`

The read-only census found **{eligible}** recorded M1C episodes and
**{micro_episodes}** episodes with linked microstructure summaries. No
development/assessment split or directional performance claim is possible.

All M01–M06 definitions, nine causal timing windows, and +5/+10/+15 minute
underlying horizons are emitted in the CSV schemas, but metric cells remain
unestimated rather than being imputed.

The runtime capacity manifest reports {runtime_capacity.get("available_tick_by_tick", "unknown")}
available tick-by-tick subscriptions, supporting
{runtime_capacity.get("paired_bidask_last_underlyings_supported", "unknown")} paired BidAsk + Last
high-resolution underlying. No capacity or Level II configuration change is recommended.

## Reproduce this zero-data assessment

```bash
{reproduce_command}
```
"""
    (output_root / "README.md").write_text(report, encoding="utf-8")
    (output_root / "report.md").write_text(report, encoding="utf-8")
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=RESEARCH_LABEL_V0)
    parser.add_argument("--census-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    decision = run_census_only_research_v0(
        census_path=args.census_json,
        output_root=args.output,
    )
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ForwardDirectionOutcomeV0",
    "PricePointV0",
    "RealizedDirectionV0",
    "evaluate_forward_outcomes_v0",
    "run_census_only_research_v0",
    "summarise_method_results_v0",
]
