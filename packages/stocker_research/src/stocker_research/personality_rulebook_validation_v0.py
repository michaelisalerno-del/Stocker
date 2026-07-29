"""Validate a compact personality rulebook on holdout symbols.

This research-only layer consumes a personality discovery report, collapses
near-duplicate rule candidates into a smaller rulebook, then applies that
rulebook to a separate state-event-detector report. It does not fetch data or
touch execution, broker, paper trading, live trading, or order placement paths.
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

from stocker_research.personality_discovery_v0 import (
    EVENT_STATE_PERSONALITY,
    _candidate_mask,
    _return_column,
    _score_rows,
    add_discovery_features,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/personality_rulebook_validation_v0")


@dataclass(frozen=True)
class RulebookValidationConfig:
    """Configuration for compact personality rulebook validation."""

    top_per_personality: int = 8
    random_iterations: int = 100
    random_seed: int = 1337
    min_validation_events: int = 12
    min_validation_symbols: int = 5
    min_validation_months: int = 3
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50


@dataclass(frozen=True)
class RulebookValidationResult:
    """Paths and headline result for a rulebook validation run."""

    run_id: str
    source_personality_dir: Path
    validation_event_dir: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    collapsed_rulebook_csv_path: Path
    validation_results_csv_path: Path
    passed_rules_csv_path: Path
    rejected_rules_csv_path: Path
    random_baseline_csv_path: Path
    concentration_warnings_csv_path: Path
    decision: str
    passed_rule_count: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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


def _personality_direction() -> dict[str, int]:
    directions: dict[str, int] = {}
    for personality, _role, direction in EVENT_STATE_PERSONALITY.values():
        directions.setdefault(personality, direction)
    return directions


def _rule_signature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        "personality",
        "role",
        "horizon",
        "regime_field",
        "regime_value",
        "rule_kind",
        "feature",
        "operator",
        "feature_b",
        "operator_b",
    ]
    return [column for column in columns if column in frame.columns]


def collapse_personality_rules(
    passed_rules: pd.DataFrame,
    *,
    top_per_personality: int = 8,
) -> pd.DataFrame:
    """Collapse near-duplicate passing rules into a compact rulebook.

    Thresholds are not part of the duplicate key. If several rules have the
    same structure but slightly different thresholds, the source rule with the
    strongest held-out same-result rate is kept.
    """

    if passed_rules.empty:
        return passed_rules.copy()
    data = passed_rules.copy()
    for column in ["feature_b", "operator_b"]:
        if column not in data:
            data[column] = ""
    if "threshold_b" not in data:
        data["threshold_b"] = math.nan
    sort_columns = [
        column
        for column in [
            "filtered_test_same_result_rate",
            "test_lift_vs_personality",
            "retained_test_count",
            "random_same_count_p95_rate",
        ]
        if column in data.columns
    ]
    ascending = [False] * len(sort_columns)
    signature_columns = _rule_signature_columns(data)
    duplicate_counts = data.groupby(signature_columns).size().rename("source_duplicate_count")
    ranked = data.sort_values(sort_columns, ascending=ascending) if sort_columns else data
    collapsed = ranked.drop_duplicates(signature_columns, keep="first").merge(
        duplicate_counts.reset_index(),
        on=signature_columns,
        how="left",
    )
    score_columns = [
        column
        for column in ["filtered_test_same_result_rate", "test_lift_vs_personality"]
        if column in collapsed.columns
    ]
    collapsed = (
        collapsed.sort_values(
            ["personality", *score_columns],
            ascending=[True, *([False] * len(score_columns))],
        )
        .groupby("personality", group_keys=False)
        .head(top_per_personality)
        .reset_index(drop=True)
    )
    collapsed["rulebook_rank"] = collapsed.groupby("personality").cumcount() + 1
    return collapsed


def _read_source_symbols(source_personality_dir: Path) -> set[str]:
    direct = source_personality_dir / "event_rows.csv"
    if direct.exists():
        return set(pd.read_csv(direct, usecols=["symbol"])["symbol"].dropna().astype(str))
    summary = source_personality_dir / "summary.json"
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        input_dir = Path(str(payload.get("input_dir", "")))
        event_rows = input_dir / "event_rows.csv"
        if event_rows.exists():
            return set(pd.read_csv(event_rows, usecols=["symbol"])["symbol"].dropna().astype(str))
    return set()


def _random_baseline(
    pool: pd.DataFrame,
    *,
    count: int,
    seed: int,
    iterations: int,
) -> dict[str, float]:
    if count <= 0 or len(pool) < count:
        return {
            "random_same_count_mean_rate": math.nan,
            "random_same_count_p95_rate": math.nan,
            "random_same_count_median_score": math.nan,
        }
    rng = np.random.default_rng(seed)
    same = pool["same_result"].astype(float).to_numpy()
    score = pool["role_score"].astype(float).to_numpy()
    rates: list[float] = []
    medians: list[float] = []
    for _ in range(iterations):
        sample = rng.choice(len(pool), size=count, replace=False)
        rates.append(float(np.nanmean(same[sample])))
        medians.append(float(np.nanmedian(score[sample])))
    return {
        "random_same_count_mean_rate": float(np.nanmean(rates)),
        "random_same_count_p95_rate": float(np.nanquantile(rates, 0.95)),
        "random_same_count_median_score": float(np.nanmedian(medians)),
    }


def _concentration(rows: pd.DataFrame) -> dict[str, float | int]:
    if rows.empty:
        return {
            "validation_symbol_count": 0,
            "single_symbol_share": math.nan,
            "validation_session_count": 0,
            "single_session_share": math.nan,
            "validation_month_count": 0,
            "single_month_share": math.nan,
        }
    symbol_counts = rows["symbol"].value_counts()
    session_counts = (
        rows[["symbol", "session_date"]].astype(str).agg("|".join, axis=1).value_counts()
    )
    month_counts = rows["month"].value_counts()
    return {
        "validation_symbol_count": int(symbol_counts.size),
        "single_symbol_share": float(symbol_counts.iloc[0] / len(rows)),
        "validation_session_count": int(session_counts.size),
        "single_session_share": float(session_counts.iloc[0] / len(rows)),
        "validation_month_count": int(month_counts.size),
        "single_month_share": float(month_counts.iloc[0] / len(rows)),
    }


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.1f}%"


def run_personality_rulebook_validation(
    *,
    source_personality_dir: Path,
    validation_event_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: RulebookValidationConfig = RulebookValidationConfig(),
) -> RulebookValidationResult:
    """Collapse source rules and validate them on holdout symbols."""

    passed_path = source_personality_dir / "passed_personality_rules.csv"
    validation_events_path = validation_event_dir / "event_rows.csv"
    if not passed_path.exists():
        raise FileNotFoundError(f"Missing source passed rules: {passed_path}")
    if not validation_events_path.exists():
        raise FileNotFoundError(f"Missing validation event rows: {validation_events_path}")

    source_rules = pd.read_csv(passed_path)
    rulebook = collapse_personality_rules(
        source_rules,
        top_per_personality=config.top_per_personality,
    )
    source_symbols = _read_source_symbols(source_personality_dir)
    validation_events = add_discovery_features(pd.read_csv(validation_events_path))
    if source_symbols:
        validation_events = validation_events[
            ~validation_events["symbol"].astype(str).isin(source_symbols)
        ].copy()

    directions = _personality_direction()
    validation_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for rule_index, rule in rulebook.iterrows():
        personality = str(rule["personality"])
        horizon = int(rule["horizon"])
        ret_col = _return_column(horizon)
        if ret_col not in validation_events:
            continue
        pool = validation_events[
            validation_events["personality"].astype(str).eq(personality)
        ].copy()
        pool = pool.dropna(subset=[ret_col])
        if pool.empty:
            continue
        pool = _score_rows(
            pool,
            horizon=horizon,
            expected_direction=directions.get(personality, 0),
        )
        base_count = len(pool)
        base_rate = float(pool["same_result"].mean())
        base_median = float(pool["role_score"].median())
        regime_field = str(rule["regime_field"])
        regime_value = str(rule["regime_value"])
        if regime_field in pool:
            regime_pool = pool[pool[regime_field].astype(str).eq(regime_value)].copy()
        else:
            regime_pool = pool.iloc[0:0].copy()
        filtered = regime_pool[_candidate_mask(regime_pool, rule)].copy()
        random_result = _random_baseline(
            regime_pool if not regime_pool.empty else pool,
            count=len(filtered),
            seed=config.random_seed + int(rule_index) + horizon,
            iterations=config.random_iterations,
        )
        concentration = _concentration(filtered)
        same_rate = float(filtered["same_result"].mean()) if not filtered.empty else math.nan
        median_score = float(filtered["role_score"].median()) if not filtered.empty else math.nan
        reasons: list[str] = []
        if len(filtered) < config.min_validation_events:
            reasons.append("low_validation_count")
        if concentration["validation_symbol_count"] < config.min_validation_symbols:
            reasons.append("low_validation_symbol_count")
        if concentration["validation_month_count"] < config.min_validation_months:
            reasons.append("low_validation_month_count")
        if (
            not math.isnan(float(concentration["single_symbol_share"]))
            and concentration["single_symbol_share"] > config.max_single_symbol_share
        ):
            reasons.append("single_symbol_dominated")
        if (
            not math.isnan(float(concentration["single_session_share"]))
            and concentration["single_session_share"] > config.max_single_session_share
        ):
            reasons.append("single_session_dominated")
        if (
            not math.isnan(float(concentration["single_month_share"]))
            and concentration["single_month_share"] > config.max_single_month_share
        ):
            reasons.append("single_month_dominated")
        if math.isnan(same_rate) or same_rate <= base_rate:
            reasons.append("no_transfer_lift_vs_personality")
        if (
            not math.isnan(random_result["random_same_count_p95_rate"])
            and same_rate <= random_result["random_same_count_p95_rate"]
        ):
            reasons.append("random_p95_not_beaten")
        if (
            not math.isnan(random_result["random_same_count_median_score"])
            and median_score <= random_result["random_same_count_median_score"]
        ):
            reasons.append("random_median_not_beaten")
        row = {
            **rule.to_dict(),
            **concentration,
            **random_result,
            "source_symbol_count": len(source_symbols),
            "validation_base_count": int(base_count),
            "validation_regime_count": int(len(regime_pool)),
            "validation_retained_count": int(len(filtered)),
            "validation_base_same_result_rate": base_rate,
            "validation_filtered_same_result_rate": same_rate,
            "validation_same_result_lift": same_rate - base_rate
            if not math.isnan(same_rate)
            else math.nan,
            "validation_base_median_score": base_median,
            "validation_filtered_median_score": median_score,
            "validation_verdict": "pass_rulebook_transfer" if not reasons else "reject",
            "validation_reject_reasons": ";".join(reasons),
        }
        validation_rows.append(row)
        random_rows.append(
            {
                "personality": personality,
                "horizon": horizon,
                "rulebook_rank": rule.get("rulebook_rank"),
                "filter_rule": rule.get("filter_rule"),
                "validation_retained_count": int(len(filtered)),
                **random_result,
            }
        )

    validation = pd.DataFrame(validation_rows)
    random_baseline = pd.DataFrame(random_rows)
    passed = (
        validation[validation["validation_verdict"].eq("pass_rulebook_transfer")].copy()
        if not validation.empty
        else pd.DataFrame()
    )
    rejected = (
        validation[~validation["validation_verdict"].eq("pass_rulebook_transfer")].copy()
        if not validation.empty
        else pd.DataFrame()
    )
    concentration_warnings = (
        rejected[
            rejected["validation_reject_reasons"].astype(str).str.contains("dominated", na=False)
        ].copy()
        if not rejected.empty
        else pd.DataFrame()
    )
    decision = (
        "continue_research_rulebook_transfer"
        if not passed.empty
        else "reject_no_rulebook_transfer"
    )

    run_id = f"personality_rulebook_validation_v0_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "rulebook": run_dir / "collapsed_rulebook.csv",
        "validation": run_dir / "rulebook_validation_results.csv",
        "passed": run_dir / "passed_rulebook_rules.csv",
        "rejected": run_dir / "rejected_rulebook_rules.csv",
        "random": run_dir / "random_rulebook_baseline.csv",
        "concentration": run_dir / "concentration_warnings.csv",
    }
    for path, frame in [
        (paths["rulebook"], rulebook),
        (paths["validation"], validation),
        (paths["passed"], passed),
        (paths["rejected"], rejected),
        (paths["random"], random_baseline),
        (paths["concentration"], concentration_warnings),
    ]:
        _write_csv(path, frame)

    passed_personalities = (
        sorted(passed["personality"].dropna().astype(str).unique().tolist())
        if not passed.empty
        else []
    )
    summary_payload = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "source_personality_dir": str(source_personality_dir),
        "validation_event_dir": str(validation_event_dir),
        "run_id": run_id,
        "output_dir": str(run_dir),
        "decision": decision,
        "source_symbol_count": len(source_symbols),
        "holdout_validation_event_rows": int(len(validation_events)),
        "collapsed_rule_count": int(len(rulebook)),
        "validated_rule_count": int(len(validation)),
        "passed_rule_count": int(len(passed)),
        "passed_personalities": passed_personalities,
        "volume_label": (
            "state_event_detector_v0 event-row features from existing local 5m OHLCV "
            "reports; no vendor fetch"
        ),
    }
    _write_json(paths["summary_json"], summary_payload)
    _write_json(paths["decision_json"], summary_payload)

    lines = [
        "# Personality Rulebook Validation V0",
        "",
        (
            "Research-only diagnostic. No broker, IG, live trading, paper trading, "
            "vendor fetching, or order placement touched. No edge is claimed."
        ),
        "",
        f"Source personality discovery: `{source_personality_dir}`",
        f"Validation event report: `{validation_event_dir}`",
        "",
        f"Decision: `{decision}`",
        "",
        "## Counts",
        "",
        f"- Source symbols excluded from transfer validation: `{len(source_symbols)}`",
        f"- Holdout validation event rows: `{len(validation_events)}`",
        f"- Collapsed rulebook rules: `{len(rulebook)}`",
        f"- Validated rules: `{len(validation)}`",
        f"- Passed transfer rules: `{len(passed)}`",
        "",
        "## Passed Rulebook Rules",
        "",
    ]
    if passed.empty:
        lines.append("No collapsed rulebook rule survived holdout-symbol validation.")
    else:
        display = passed.sort_values(
            ["validation_same_result_lift", "validation_filtered_same_result_rate"],
            ascending=False,
        ).head(20)
        lines.extend(
            [
                "| personality | h | rule | n | symbols | same_result | lift | rand_p95 |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in display.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["personality"]),
                        str(int(row["horizon"])),
                        str(row["filter_rule"]),
                        str(int(row["validation_retained_count"])),
                        str(int(row["validation_symbol_count"])),
                        _format_pct(row["validation_filtered_same_result_rate"]),
                        _format_pct(row["validation_same_result_lift"]),
                        _format_pct(row["random_same_count_p95_rate"]),
                    ]
                )
                + " |"
            )
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    return RulebookValidationResult(
        run_id=run_id,
        source_personality_dir=source_personality_dir,
        validation_event_dir=validation_event_dir,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        collapsed_rulebook_csv_path=paths["rulebook"],
        validation_results_csv_path=paths["validation"],
        passed_rules_csv_path=paths["passed"],
        rejected_rules_csv_path=paths["rejected"],
        random_baseline_csv_path=paths["random"],
        concentration_warnings_csv_path=paths["concentration"],
        decision=decision,
        passed_rule_count=int(len(passed)),
    )
