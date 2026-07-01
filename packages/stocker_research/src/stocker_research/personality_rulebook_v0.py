"""Apply a fixed personality rulebook to state-event rows.

This is a research-only diagnostic layer. It does not search for thresholds,
fit filters, fetch vendor data, or touch execution paths. Rules come from an
explicit YAML file and are applied as written.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from stocker_research.personality_discovery_v0 import (
    _candidate_mask,
    _return_column,
    _score_rows,
    add_discovery_features,
)

DEFAULT_RULEBOOK_PATH = Path("configs/research/personality_rulebook_v0/rules.yaml")
DEFAULT_OUTPUT_DIR = Path("data/reports/research/personality_rulebook_v0")


@dataclass(frozen=True)
class FixedPersonalityRule:
    """One literal rule from the fixed personality rulebook."""

    rule_id: str
    personality: str
    role: str
    horizon: int
    expected_direction: int
    regime_field: str
    regime_value: str
    rule_kind: str
    feature: str
    operator: str
    threshold: float
    filter_rule: str
    feature_b: str = ""
    operator_b: str = ""
    threshold_b: float = math.nan
    source_note: str = ""


@dataclass(frozen=True)
class FixedPersonalityRulebook:
    """Versioned fixed rulebook loaded from YAML."""

    version: str
    rules: tuple[FixedPersonalityRule, ...]
    research_only: bool = True


@dataclass(frozen=True)
class PersonalityRulebookConfig:
    """Configuration for fixed rulebook evaluation."""

    random_iterations: int = 100
    random_seed: int = 1337
    min_events: int = 12
    min_symbols: int = 5
    min_months: int = 3
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50


@dataclass(frozen=True)
class PersonalityRulebookResult:
    """Paths and headline result for a fixed rulebook run."""

    run_id: str
    input_event_dir: Path
    rulebook_path: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    fixed_rulebook_csv_path: Path
    rule_matches_csv_path: Path
    rule_summary_csv_path: Path
    personality_summary_csv_path: Path
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


def load_fixed_rulebook(rulebook_path: Path) -> FixedPersonalityRulebook:
    """Load a fixed personality rulebook from YAML."""

    payload = yaml.safe_load(rulebook_path.read_text(encoding="utf-8")) or {}
    rules = []
    for raw in payload.get("rules", []):
        rules.append(
            FixedPersonalityRule(
                rule_id=str(raw["rule_id"]),
                personality=str(raw["personality"]),
                role=str(raw["role"]),
                horizon=int(raw["horizon"]),
                expected_direction=int(raw.get("expected_direction", 0)),
                regime_field=str(raw["regime_field"]),
                regime_value=str(raw["regime_value"]),
                rule_kind=str(raw.get("rule_kind", "single")),
                feature=str(raw["feature"]),
                operator=str(raw["operator"]),
                threshold=float(raw["threshold"]),
                filter_rule=str(raw.get("filter_rule", "")),
                feature_b=str(raw.get("feature_b", "")),
                operator_b=str(raw.get("operator_b", "")),
                threshold_b=float(raw.get("threshold_b", math.nan)),
                source_note=str(raw.get("source_note", "")),
            )
        )
    if not rules:
        raise ValueError(f"Rulebook has no rules: {rulebook_path}")
    return FixedPersonalityRulebook(
        version=str(payload.get("version", "v0")),
        research_only=bool(payload.get("research_only", True)),
        rules=tuple(rules),
    )


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
    rates = []
    medians = []
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
            "symbol_count": 0,
            "single_symbol_share": math.nan,
            "session_count": 0,
            "single_session_share": math.nan,
            "month_count": 0,
            "single_month_share": math.nan,
        }
    symbol_counts = rows["symbol"].value_counts()
    session_counts = (
        rows[["symbol", "session_date"]].astype(str).agg("|".join, axis=1).value_counts()
    )
    month_counts = rows["month"].value_counts()
    return {
        "symbol_count": int(symbol_counts.size),
        "single_symbol_share": float(symbol_counts.iloc[0] / len(rows)),
        "session_count": int(session_counts.size),
        "single_session_share": float(session_counts.iloc[0] / len(rows)),
        "month_count": int(month_counts.size),
        "single_month_share": float(month_counts.iloc[0] / len(rows)),
    }


def _rule_series(rule: FixedPersonalityRule) -> pd.Series:
    return pd.Series(asdict(rule))


def _prepare_events(event_rows: pd.DataFrame) -> pd.DataFrame:
    if {"personality", "month", "vwap_side_regime"}.issubset(event_rows.columns):
        return event_rows.copy()
    return add_discovery_features(event_rows)


def apply_fixed_rulebook(
    event_rows: pd.DataFrame,
    rulebook: FixedPersonalityRulebook,
    *,
    config: PersonalityRulebookConfig = PersonalityRulebookConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply fixed rules to event rows without fitting thresholds."""

    events = _prepare_events(event_rows)
    match_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []

    for index, rule in enumerate(rulebook.rules):
        ret_col = _return_column(rule.horizon)
        if ret_col not in events:
            continue
        pool = events[events["personality"].astype(str).eq(rule.personality)].copy()
        pool = pool.dropna(subset=[ret_col])
        if pool.empty:
            continue
        pool = _score_rows(
            pool,
            horizon=rule.horizon,
            expected_direction=rule.expected_direction,
        )
        base_rate = float(pool["same_result"].mean())
        base_median = float(pool["role_score"].median())
        if rule.regime_field in pool:
            regime_pool = pool[pool[rule.regime_field].astype(str).eq(rule.regime_value)]
        else:
            regime_pool = pool.iloc[0:0]
        filtered = regime_pool[_candidate_mask(regime_pool, _rule_series(rule))].copy()
        if not filtered.empty:
            for field, value in asdict(rule).items():
                filtered[field] = value
            match_rows.append(filtered)
        random_result = _random_baseline(
            regime_pool if not regime_pool.empty else pool,
            count=len(filtered),
            seed=config.random_seed + index + rule.horizon,
            iterations=config.random_iterations,
        )
        concentration = _concentration(filtered)
        same_rate = float(filtered["same_result"].mean()) if not filtered.empty else math.nan
        median_score = float(filtered["role_score"].median()) if not filtered.empty else math.nan
        reasons: list[str] = []
        if len(filtered) < config.min_events:
            reasons.append("low_event_count")
        if concentration["symbol_count"] < config.min_symbols:
            reasons.append("low_symbol_count")
        if concentration["month_count"] < config.min_months:
            reasons.append("low_month_count")
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
            reasons.append("no_lift_vs_personality")
        if (
            not math.isnan(random_result["random_same_count_p95_rate"])
            and same_rate <= random_result["random_same_count_p95_rate"]
        ):
            reasons.append("random_p95_not_beaten")
        summary_rows.append(
            {
                **asdict(rule),
                **concentration,
                **random_result,
                "personality_base_count": int(len(pool)),
                "regime_count": int(len(regime_pool)),
                "event_count": int(len(filtered)),
                "personality_base_same_result_rate": base_rate,
                "same_result_rate": same_rate,
                "same_result_lift_vs_personality": same_rate - base_rate
                if not math.isnan(same_rate)
                else math.nan,
                "personality_base_median_score": base_median,
                "median_score": median_score,
                "median_score_lift_vs_random": (
                    median_score - random_result["random_same_count_median_score"]
                    if not math.isnan(median_score)
                    and not math.isnan(random_result["random_same_count_median_score"])
                    else math.nan
                ),
                "verdict": "pass_fixed_rulebook" if not reasons else "reject",
                "reject_reasons": ";".join(reasons),
            }
        )
        random_rows.append(
            {
                "rule_id": rule.rule_id,
                "personality": rule.personality,
                "horizon": rule.horizon,
                "event_count": int(len(filtered)),
                **random_result,
            }
        )

    matches = pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame()
    return matches, pd.DataFrame(summary_rows), pd.DataFrame(random_rows)


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.1f}%"


def _decision(rule_summary: pd.DataFrame) -> str:
    if rule_summary.empty:
        return "reject_no_fixed_rule_matches"
    passed = rule_summary[rule_summary["verdict"].eq("pass_fixed_rulebook")]
    if passed.empty:
        return "reject_fixed_rulebook_no_transfer"
    personalities = set(passed["personality"].astype(str))
    if "dead_chop_noise" in personalities and len(personalities) >= 2:
        return "continue_research_fixed_rulebook"
    if personalities == {"dead_chop_noise"}:
        return "continue_research_no_trade_filter_only"
    return "continue_research_narrow_fixed_rulebook"


def run_personality_rulebook_lab(
    *,
    input_event_dir: Path,
    rulebook_path: Path = DEFAULT_RULEBOOK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: PersonalityRulebookConfig = PersonalityRulebookConfig(),
) -> PersonalityRulebookResult:
    """Run a fixed personality rulebook over one state-event report."""

    event_rows_path = input_event_dir / "event_rows.csv"
    if not event_rows_path.exists():
        raise FileNotFoundError(f"Missing event rows: {event_rows_path}")
    rulebook = load_fixed_rulebook(rulebook_path)
    fixed_rulebook = pd.DataFrame([asdict(rule) for rule in rulebook.rules])
    matches, rule_summary, random_baseline = apply_fixed_rulebook(
        pd.read_csv(event_rows_path),
        rulebook,
        config=config,
    )
    passed = (
        rule_summary[rule_summary["verdict"].eq("pass_fixed_rulebook")].copy()
        if not rule_summary.empty
        else pd.DataFrame()
    )
    concentration_warnings = (
        rule_summary[
            rule_summary["reject_reasons"].astype(str).str.contains("dominated", na=False)
        ].copy()
        if not rule_summary.empty
        else pd.DataFrame()
    )
    personality_summary = (
        rule_summary.groupby(["personality", "role"], as_index=False)
        .agg(
            rule_count=("rule_id", "count"),
            passed_rule_count=(
                "verdict",
                lambda series: int(series.eq("pass_fixed_rulebook").sum()),
            ),
            total_event_count=("event_count", "sum"),
            median_same_result_rate=("same_result_rate", "median"),
            median_lift_vs_personality=("same_result_lift_vs_personality", "median"),
        )
        if not rule_summary.empty
        else pd.DataFrame()
    )
    decision = _decision(rule_summary)

    run_id = f"personality_rulebook_v0_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "fixed_rulebook": run_dir / "fixed_rulebook.csv",
        "matches": run_dir / "rule_matches.csv",
        "rule_summary": run_dir / "rule_summary.csv",
        "personality_summary": run_dir / "personality_summary.csv",
        "random": run_dir / "random_baseline.csv",
        "concentration": run_dir / "concentration_warnings.csv",
    }
    for path, frame in [
        (paths["fixed_rulebook"], fixed_rulebook),
        (paths["matches"], matches),
        (paths["rule_summary"], rule_summary),
        (paths["personality_summary"], personality_summary),
        (paths["random"], random_baseline),
        (paths["concentration"], concentration_warnings),
    ]:
        _write_csv(path, frame)

    summary_payload = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "input_event_dir": str(input_event_dir),
        "rulebook_path": str(rulebook_path),
        "run_id": run_id,
        "output_dir": str(run_dir),
        "decision": decision,
        "rule_count": int(len(fixed_rulebook)),
        "matched_event_rows": int(len(matches)),
        "passed_rule_count": int(len(passed)),
        "passed_personalities": sorted(passed["personality"].unique().tolist())
        if not passed.empty
        else [],
        "volume_label": (
            "state_event_detector_v0 event-row features from existing local 5m OHLCV "
            "reports; no vendor fetch"
        ),
    }
    _write_json(paths["summary_json"], summary_payload)
    _write_json(paths["decision_json"], summary_payload)

    lines = [
        "# Personality Rulebook V0",
        "",
        (
            "Fixed-rulebook research diagnostic. No threshold search, no broker, no IG, "
            "no live trading, no paper trading, no vendor fetching, and no order placement. "
            "No edge is claimed."
        ),
        "",
        f"Input event report: `{input_event_dir}`",
        f"Rulebook: `{rulebook_path}`",
        f"Decision: `{decision}`",
        "",
        "## Counts",
        "",
        f"- Fixed rules: `{len(fixed_rulebook)}`",
        f"- Matched event rows: `{len(matches)}`",
        f"- Passed rules: `{len(passed)}`",
        "",
        "## Rule Results",
        "",
        "| rule | personality | h | n | symbols | same_result | lift | random_p95 | verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not rule_summary.empty:
        for _, row in rule_summary.sort_values(
            ["verdict", "same_result_lift_vs_personality"],
            ascending=[True, False],
        ).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["rule_id"]),
                        str(row["personality"]),
                        str(int(row["horizon"])),
                        str(int(row["event_count"])),
                        str(int(row["symbol_count"])),
                        _format_pct(row["same_result_rate"]),
                        _format_pct(row["same_result_lift_vs_personality"]),
                        _format_pct(row["random_same_count_p95_rate"]),
                        str(row["verdict"]),
                    ]
                )
                + " |"
            )
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    return PersonalityRulebookResult(
        run_id=run_id,
        input_event_dir=input_event_dir,
        rulebook_path=rulebook_path,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        fixed_rulebook_csv_path=paths["fixed_rulebook"],
        rule_matches_csv_path=paths["matches"],
        rule_summary_csv_path=paths["rule_summary"],
        personality_summary_csv_path=paths["personality_summary"],
        random_baseline_csv_path=paths["random"],
        concentration_warnings_csv_path=paths["concentration"],
        decision=decision,
        passed_rule_count=int(len(passed)),
    )
