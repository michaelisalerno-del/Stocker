"""Role-aware personality template discovery and validation.

This research-only layer tests explicit personality templates of the form:

    parent event state + base conditions + one regime + one filter

It consumes existing state_event_detector_v0 event rows only. It does not fetch
vendor data, touch broker/execution paths, or place orders.
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

from stocker_research.event_failure_cutter_v0 import find_latest_state_event_detector_run
from stocker_research.personality_discovery_v0 import (
    _abs_return_column,
    _candidate_mask,
    _return_column,
    add_discovery_features,
)

DEFAULT_INPUT_BASE_DIR = Path("data/reports/research/state_event_detector_v0")
DEFAULT_TEMPLATE_PATH = Path("configs/research/personality_template_v0/templates.yaml")
DEFAULT_OUTPUT_DIR = Path("data/reports/research/personality_template_v0")


@dataclass(frozen=True)
class TemplateCondition:
    """One leakage-safe condition applied at the event bar."""

    feature: str
    operator: str
    threshold: float | str | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class PersonalityTemplate:
    """One explicit personality template loaded from YAML."""

    template_id: str
    personality: str
    parent_event_state: str
    role: str
    expected_direction: int
    horizon: int
    base_conditions: tuple[TemplateCondition, ...]
    regime_fields: tuple[str, ...]
    filter_features: tuple[str, ...]
    min_base_events: int = 30
    min_retained_events: int = 8
    min_symbols: int = 3
    min_months: int = 3
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50
    source_note: str = ""


@dataclass(frozen=True)
class PersonalityTemplateBook:
    """Versioned personality template book."""

    version: str
    templates: tuple[PersonalityTemplate, ...]
    research_only: bool = True


@dataclass(frozen=True)
class PersonalityTemplateConfig:
    """Configuration for personality template evaluation."""

    random_iterations: int = 100
    random_seed: int = 1337
    max_candidates_per_template: int = 96
    max_selected_per_template: int = 12
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)
    stop_loss_bps: tuple[float, ...] = (25.0, 50.0, 100.0)


@dataclass(frozen=True)
class PersonalityTemplateResult:
    """Paths and headline result for a personality template run."""

    run_id: str
    input_event_dir: Path
    template_path: Path
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    decision_json_path: Path
    template_csv_path: Path
    base_summary_csv_path: Path
    candidate_rules_csv_path: Path
    selected_rules_csv_path: Path
    rejected_rules_csv_path: Path
    template_matches_csv_path: Path
    random_baseline_csv_path: Path
    concentration_warnings_csv_path: Path
    decision: str
    selected_rule_count: int


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


def _as_tuple(raw: Any) -> tuple[Any, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    return (raw,)


def _condition_from_payload(raw: dict[str, Any]) -> TemplateCondition:
    return TemplateCondition(
        feature=str(raw["feature"]),
        operator=str(raw["operator"]),
        threshold=raw.get("threshold"),
        lower=float(raw["lower"]) if "lower" in raw else None,
        upper=float(raw["upper"]) if "upper" in raw else None,
    )


def _template_from_payload(raw: dict[str, Any]) -> PersonalityTemplate:
    minimums = raw.get("minimums") or {}
    return PersonalityTemplate(
        template_id=str(raw["template_id"]),
        personality=str(raw["personality"]),
        parent_event_state=str(raw["parent_event_state"]),
        role=str(raw["role"]),
        expected_direction=int(raw.get("expected_direction", 0)),
        horizon=int(raw["horizon"]),
        base_conditions=tuple(
            _condition_from_payload(item) for item in raw.get("base_conditions", [])
        ),
        regime_fields=tuple(str(item) for item in _as_tuple(raw.get("regime_fields"))),
        filter_features=tuple(str(item) for item in _as_tuple(raw.get("filter_features"))),
        min_base_events=int(minimums.get("base_events", 30)),
        min_retained_events=int(minimums.get("retained_events", 8)),
        min_symbols=int(minimums.get("symbols", 3)),
        min_months=int(minimums.get("months", 3)),
        max_single_symbol_share=float(raw.get("max_single_symbol_share", 0.50)),
        max_single_session_share=float(raw.get("max_single_session_share", 0.20)),
        max_single_month_share=float(raw.get("max_single_month_share", 0.50)),
        source_note=str(raw.get("source_note", "")),
    )


def load_personality_templates(template_path: Path) -> PersonalityTemplateBook:
    """Load a YAML template book from a file or directory."""

    paths: list[Path]
    if template_path.is_dir():
        paths = sorted(template_path.glob("*.yaml")) + sorted(template_path.glob("*.yml"))
        if not paths:
            raise FileNotFoundError(f"No template YAML files found in {template_path}")
    else:
        paths = [template_path]

    templates: list[PersonalityTemplate] = []
    version = "personality_template_v0"
    research_only = True
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        version = str(payload.get("version", version))
        research_only = bool(payload.get("research_only", research_only))
        raw_templates = payload.get("templates", [])
        if "template_id" in payload:
            raw_templates = [payload]
        for raw in raw_templates:
            templates.append(_template_from_payload(raw))
    if not templates:
        raise ValueError(f"No templates found in {template_path}")
    return PersonalityTemplateBook(
        version=version,
        templates=tuple(templates),
        research_only=research_only,
    )


def _op_mask(series: pd.Series, condition: TemplateCondition) -> pd.Series:
    operator = condition.operator
    if operator == "between":
        if condition.lower is None or condition.upper is None:
            raise ValueError(f"between requires lower and upper: {condition.feature}")
        return series.between(condition.lower, condition.upper).fillna(False)
    if operator == "==":
        return series.astype(str).eq(str(condition.threshold)).fillna(False)
    if condition.threshold is None:
        raise ValueError(f"{operator} requires threshold: {condition.feature}")
    threshold = float(condition.threshold)
    if operator == "<=":
        return series.le(threshold).fillna(False)
    if operator == ">=":
        return series.ge(threshold).fillna(False)
    if operator == "<":
        return series.lt(threshold).fillna(False)
    if operator == ">":
        return series.gt(threshold).fillna(False)
    raise ValueError(f"Unsupported condition operator: {operator}")


def _base_mask(rows: pd.DataFrame, template: PersonalityTemplate) -> pd.Series:
    mask = rows["event_state"].astype(str).eq(template.parent_event_state)
    for condition in template.base_conditions:
        if condition.feature not in rows:
            return pd.Series(False, index=rows.index)
        if condition.feature.startswith("forward_"):
            raise ValueError(f"Template condition cannot use target column: {condition.feature}")
        mask &= _op_mask(rows[condition.feature], condition)
    return mask.fillna(False)


def _score_against_parent(
    rows: pd.DataFrame,
    parent_rows: pd.DataFrame,
    *,
    horizon: int,
    expected_direction: int,
) -> pd.DataFrame:
    data = rows.copy()
    ret_col = _return_column(horizon)
    if expected_direction == 0:
        abs_col = _abs_return_column(horizon)
        if abs_col in parent_rows:
            parent_threshold = float(parent_rows[abs_col].median())
            data["role_score"] = parent_threshold - data[abs_col]
            data["score_mode"] = "no_trade_low_abs_move"
        else:
            parent_threshold = float(parent_rows[ret_col].abs().median())
            data["role_score"] = parent_threshold - data[ret_col].abs()
            data["score_mode"] = "no_trade_low_abs_move"
    else:
        data["role_score"] = expected_direction * data[ret_col]
        data["score_mode"] = "directional"
    data["same_result"] = data["role_score"] > 0
    return data


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


def _stop_loss_metrics(
    rows: pd.DataFrame,
    *,
    horizon: int,
    expected_direction: int,
    stop_loss_bps: tuple[float, ...],
) -> dict[str, float]:
    if expected_direction == 0 or rows.empty:
        return {
            f"stop_{int(stop)}bps_hit_rate": math.nan for stop in stop_loss_bps
        } | {
            f"stop_{int(stop)}bps_survival_rate": math.nan for stop in stop_loss_bps
        }
    if expected_direction > 0:
        mae_col = f"forward_{horizon}_bar_mae"
        if mae_col not in rows:
            adverse_bps = pd.Series(np.nan, index=rows.index)
        else:
            adverse_bps = (-rows[mae_col].astype(float)) * 10000.0
    else:
        mfe_col = f"forward_{horizon}_bar_mfe"
        if mfe_col not in rows:
            adverse_bps = pd.Series(np.nan, index=rows.index)
        else:
            adverse_bps = rows[mfe_col].astype(float) * 10000.0
    metrics: dict[str, float] = {}
    for stop in stop_loss_bps:
        hit = adverse_bps >= stop
        has_values = hit.notna().any()
        metrics[f"stop_{int(stop)}bps_hit_rate"] = (
            float(hit.mean()) if has_values else math.nan
        )
        metrics[f"stop_{int(stop)}bps_survival_rate"] = (
            float((~hit).mean()) if has_values else math.nan
        )
    return metrics


def _condition_text(condition: TemplateCondition) -> str:
    if condition.operator == "between":
        return f"{condition.feature} between {condition.lower:.6g} and {condition.upper:.6g}"
    return f"{condition.feature} {condition.operator} {condition.threshold}"


def _template_to_row(template: PersonalityTemplate) -> dict[str, Any]:
    payload = asdict(template)
    payload["base_conditions"] = " AND ".join(
        _condition_text(condition) for condition in template.base_conditions
    )
    payload["regime_fields"] = ",".join(template.regime_fields)
    payload["filter_features"] = ",".join(template.filter_features)
    return payload


def _build_filter_rows(
    rows: pd.DataFrame,
    template: PersonalityTemplate,
    *,
    config: PersonalityTemplateConfig,
) -> pd.DataFrame:
    available = set(rows.columns)
    safe_features = [
        feature
        for feature in template.filter_features
        if feature in available
        and not feature.startswith("forward_")
        and pd.api.types.is_numeric_dtype(rows[feature])
    ]
    filter_rows: list[dict[str, Any]] = []
    for feature in safe_features:
        values = rows[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if values.nunique() < 2:
            continue
        for quantile in config.quantiles:
            threshold = float(values.quantile(quantile))
            for operator in ("<=", ">="):
                filter_rows.append(
                    {
                        "rule_kind": "single",
                        "feature": feature,
                        "operator": operator,
                        "threshold": threshold,
                        "feature_b": "",
                        "operator_b": "",
                        "threshold_b": math.nan,
                        "filter_rule": f"{feature} {operator} {threshold:.6g}",
                    }
                )
    if not filter_rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(filter_rows)
        .drop_duplicates(["feature", "operator", "threshold"])
        .head(config.max_candidates_per_template)
    )


def evaluate_personality_templates(
    event_rows: pd.DataFrame,
    template_book: PersonalityTemplateBook,
    *,
    config: PersonalityTemplateConfig = PersonalityTemplateConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate templates and discover one-regime/one-filter caveats."""

    events = add_discovery_features(event_rows)
    base_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []

    for template_index, template in enumerate(template_book.templates):
        horizon = template.horizon
        ret_col = _return_column(horizon)
        if ret_col not in events:
            continue
        parent = events[events["event_state"].astype(str).eq(template.parent_event_state)].copy()
        parent = parent.dropna(subset=[ret_col])
        if parent.empty:
            continue
        parent_scored = _score_against_parent(
            parent,
            parent,
            horizon=horizon,
            expected_direction=template.expected_direction,
        )
        base = events[_base_mask(events, template)].copy().dropna(subset=[ret_col])
        base_scored = _score_against_parent(
            base,
            parent,
            horizon=horizon,
            expected_direction=template.expected_direction,
        )
        base_concentration = _concentration(base_scored)
        base_rows.append(
            {
                "template_id": template.template_id,
                "personality": template.personality,
                "parent_event_state": template.parent_event_state,
                "role": template.role,
                "horizon": horizon,
                "expected_direction": template.expected_direction,
                "parent_event_count": int(len(parent_scored)),
                "base_event_count": int(len(base_scored)),
                "base_same_result_rate": float(base_scored["same_result"].mean())
                if not base_scored.empty
                else math.nan,
                "base_median_score": float(base_scored["role_score"].median())
                if not base_scored.empty
                else math.nan,
                "parent_same_result_rate": float(parent_scored["same_result"].mean()),
                "parent_median_score": float(parent_scored["role_score"].median()),
                "base_score_mode": str(base_scored["score_mode"].iloc[0])
                if not base_scored.empty
                else "",
                **_stop_loss_metrics(
                    base_scored,
                    horizon=horizon,
                    expected_direction=template.expected_direction,
                    stop_loss_bps=config.stop_loss_bps,
                ),
                **base_concentration,
            }
        )
        if len(base_scored) < template.min_base_events:
            continue
        filter_rows = _build_filter_rows(base_scored, template, config=config)
        if filter_rows.empty:
            continue
        for regime_field in template.regime_fields:
            if regime_field not in base_scored:
                continue
            for regime_value in sorted(base_scored[regime_field].dropna().astype(str).unique()):
                regime_pool = base_scored[
                    base_scored[regime_field].astype(str).eq(regime_value)
                ].copy()
                if len(regime_pool) < template.min_retained_events:
                    continue
                for filter_index, filter_row in filter_rows.iterrows():
                    retained = regime_pool[_candidate_mask(regime_pool, filter_row)].copy()
                    if retained.empty:
                        continue
                    random_result = _random_baseline(
                        regime_pool,
                        count=len(retained),
                        seed=(
                            config.random_seed
                            + template_index * 1009
                            + filter_index
                            + horizon
                        ),
                        iterations=config.random_iterations,
                    )
                    concentration = _concentration(retained)
                    same_rate = float(retained["same_result"].mean())
                    median_score = float(retained["role_score"].median())
                    reasons: list[str] = []
                    if len(retained) < template.min_retained_events:
                        reasons.append("low_event_count")
                    if concentration["symbol_count"] < template.min_symbols:
                        reasons.append("low_symbol_count")
                    if concentration["month_count"] < template.min_months:
                        reasons.append("low_month_count")
                    if (
                        not math.isnan(float(concentration["single_symbol_share"]))
                        and concentration["single_symbol_share"]
                        > template.max_single_symbol_share
                    ):
                        reasons.append("single_symbol_dominated")
                    if (
                        not math.isnan(float(concentration["single_session_share"]))
                        and concentration["single_session_share"]
                        > template.max_single_session_share
                    ):
                        reasons.append("single_session_dominated")
                    if (
                        not math.isnan(float(concentration["single_month_share"]))
                        and concentration["single_month_share"] > template.max_single_month_share
                    ):
                        reasons.append("single_month_dominated")
                    base_rate = float(base_scored["same_result"].mean())
                    base_median = float(base_scored["role_score"].median())
                    rate_lifted = same_rate > base_rate
                    median_lifted = median_score > base_median
                    random_rate_beaten = (
                        not math.isnan(random_result["random_same_count_p95_rate"])
                        and same_rate > random_result["random_same_count_p95_rate"]
                    )
                    random_median_beaten = (
                        not math.isnan(random_result["random_same_count_median_score"])
                        and median_score > random_result["random_same_count_median_score"]
                    )
                    if not rate_lifted and not median_lifted:
                        reasons.append("no_lift_vs_base_template")
                    if not random_rate_beaten and not random_median_beaten:
                        reasons.append("random_same_count_not_beaten")
                    row = {
                        "template_id": template.template_id,
                        "personality": template.personality,
                        "parent_event_state": template.parent_event_state,
                        "role": template.role,
                        "horizon": horizon,
                        "expected_direction": template.expected_direction,
                        "regime_field": regime_field,
                        "regime_value": regime_value,
                        **filter_row.to_dict(),
                        **concentration,
                        **random_result,
                        "base_event_count": int(len(base_scored)),
                        "regime_count": int(len(regime_pool)),
                        "retained_event_count": int(len(retained)),
                        "base_same_result_rate": float(base_scored["same_result"].mean()),
                        "retained_same_result_rate": same_rate,
                        "same_result_lift_vs_base": same_rate
                        - float(base_scored["same_result"].mean()),
                        "base_median_score": float(base_scored["role_score"].median()),
                        "retained_median_score": median_score,
                        "median_score_lift_vs_base": median_score
                        - float(base_scored["role_score"].median()),
                        **_stop_loss_metrics(
                            retained,
                            horizon=horizon,
                            expected_direction=template.expected_direction,
                            stop_loss_bps=config.stop_loss_bps,
                        ),
                        "verdict": "pass_template_caveat" if not reasons else "reject",
                        "reject_reasons": ";".join(reasons),
                    }
                    candidate_rows.append(row)
                    random_rows.append(
                        {
                            "template_id": template.template_id,
                            "regime_field": regime_field,
                            "regime_value": regime_value,
                            "filter_rule": row["filter_rule"],
                            "retained_event_count": int(len(retained)),
                            **random_result,
                        }
                    )
                    if not reasons:
                        selected_rows.append(row)

    candidates = pd.DataFrame(candidate_rows)
    selected = pd.DataFrame(selected_rows)
    if not selected.empty:
        selected = (
            selected.sort_values(
                ["template_id", "median_score_lift_vs_base", "same_result_lift_vs_base"],
                ascending=[True, False, False],
            )
            .groupby("template_id", group_keys=False)
            .head(config.max_selected_per_template)
            .reset_index(drop=True)
        )
    return pd.DataFrame(base_rows), candidates, selected, pd.DataFrame(random_rows)


def _latest_input(input_event_dir: Path | None, input_base_dir: Path) -> Path:
    return (
        input_event_dir
        if input_event_dir is not None
        else find_latest_state_event_detector_run(input_base_dir)
    )


def _decision(selected: pd.DataFrame) -> str:
    if selected.empty:
        return "reject_no_personality_template_caveat"
    roles = set(selected["role"].astype(str))
    if any("no_trade" in role for role in roles) and len(roles) == 1:
        return "continue_research_no_trade_template_only"
    if len(roles) >= 2:
        return "continue_research_personality_templates_mixed_roles"
    if any("short" in role or "blocker" in role for role in roles):
        return "continue_research_personality_templates_blocker"
    return "continue_research_personality_templates_long"


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.1f}%"


def _format_score_bps(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{10000 * float(value):.2f}"


def run_personality_template_lab(
    *,
    input_event_dir: Path | None = None,
    input_base_dir: Path = DEFAULT_INPUT_BASE_DIR,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: PersonalityTemplateConfig = PersonalityTemplateConfig(),
) -> PersonalityTemplateResult:
    """Run role-aware personality template caveat discovery."""

    resolved_input = _latest_input(input_event_dir, input_base_dir)
    event_rows_path = resolved_input / "event_rows.csv"
    if not event_rows_path.exists():
        raise FileNotFoundError(f"Missing event rows: {event_rows_path}")
    template_book = load_personality_templates(template_path)
    event_rows = pd.read_csv(event_rows_path)
    base_summary, candidates, selected, random_baseline = evaluate_personality_templates(
        event_rows,
        template_book,
        config=config,
    )
    rejected = (
        candidates[candidates["verdict"].ne("pass_template_caveat")].copy()
        if not candidates.empty
        else pd.DataFrame()
    )
    concentration_warnings = (
        candidates[candidates["reject_reasons"].astype(str).str.contains("dominated", na=False)]
        if not candidates.empty
        else pd.DataFrame()
    )
    templates = pd.DataFrame([_template_to_row(template) for template in template_book.templates])

    # Match rows for selected rules are intentionally compact: row-level output
    # is available by applying selected_template_rules back to event_rows.
    template_matches = selected.copy()

    decision = _decision(selected)
    run_id = f"personality_template_v0_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_dir / run_id
    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "decision_json": run_dir / "decision.json",
        "templates": run_dir / "personality_templates.csv",
        "base": run_dir / "template_base_summary.csv",
        "candidates": run_dir / "candidate_template_rules.csv",
        "selected": run_dir / "selected_template_rules.csv",
        "rejected": run_dir / "rejected_template_rules.csv",
        "matches": run_dir / "template_matches.csv",
        "random": run_dir / "random_baseline.csv",
        "concentration": run_dir / "concentration_warnings.csv",
    }
    for path, frame in [
        (paths["templates"], templates),
        (paths["base"], base_summary),
        (paths["candidates"], candidates),
        (paths["selected"], selected),
        (paths["rejected"], rejected),
        (paths["matches"], template_matches),
        (paths["random"], random_baseline),
        (paths["concentration"], concentration_warnings),
    ]:
        _write_csv(path, frame)

    summary_payload = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "input_event_dir": str(resolved_input),
        "template_path": str(template_path),
        "run_id": run_id,
        "output_dir": str(run_dir),
        "decision": decision,
        "template_count": int(len(templates)),
        "candidate_rule_count": int(len(candidates)),
        "selected_rule_count": int(len(selected)),
        "selected_templates": sorted(selected["template_id"].unique().tolist())
        if not selected.empty
        else [],
        "stop_loss_bps": [float(value) for value in config.stop_loss_bps],
        "volume_label": (
            "state_event_detector_v0 event-row features from existing local 5m OHLCV "
            "reports; no vendor fetch"
        ),
    }
    _write_json(paths["summary_json"], summary_payload)
    _write_json(paths["decision_json"], summary_payload)

    lines = [
        "# Personality Template V0",
        "",
        (
            "Role-aware research diagnostic for personality + regime + filter templates. "
            "No broker, no IG, no live trading, no paper trading, no vendor fetching, and "
            "no order placement. No edge is claimed."
        ),
        "",
        f"Input event report: `{resolved_input}`",
        f"Template book: `{template_path}`",
        f"Decision: `{decision}`",
        "",
        "## Counts",
        "",
        f"- Templates: `{len(templates)}`",
        f"- Candidate caveat rules: `{len(candidates)}`",
        f"- Selected caveat rules: `{len(selected)}`",
        "",
        "## Base Templates",
        "",
        "| template | role | h | base n | symbols | base rate | parent rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not base_summary.empty:
        for _, row in base_summary.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["template_id"]),
                        str(row["role"]),
                        str(int(row["horizon"])),
                        str(int(row["base_event_count"])),
                        str(int(row["symbol_count"])),
                        _format_pct(row["base_same_result_rate"]),
                        _format_pct(row["parent_same_result_rate"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Selected Caveats",
            "",
            (
                "| template | regime | filter | n | symbols | rate | lift | "
                "median_score_bps | verdict |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if not selected.empty:
        for _, row in selected.sort_values(
            ["template_id", "median_score_lift_vs_base"],
            ascending=[True, False],
        ).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["template_id"]),
                        f"{row['regime_field']}={row['regime_value']}",
                        str(row["filter_rule"]),
                        str(int(row["retained_event_count"])),
                        str(int(row["symbol_count"])),
                        _format_pct(row["retained_same_result_rate"]),
                        _format_pct(row["same_result_lift_vs_base"]),
                        _format_score_bps(row["retained_median_score"]),
                        str(row["verdict"]),
                    ]
                )
                + " |"
            )
    stop_hit_columns = sorted(
        [
            column
            for column in selected.columns
            if column.startswith("stop_") and column.endswith("bps_hit_rate")
        ],
        key=lambda column: float(column.split("_")[1].replace("bps", "")),
    )
    if stop_hit_columns:
        preferred_hit_column = (
            "stop_50bps_hit_rate"
            if "stop_50bps_hit_rate" in stop_hit_columns
            else stop_hit_columns[0]
        )
        stop_label = preferred_hit_column.removeprefix("stop_").removesuffix("_hit_rate")
        preferred_survival_column = f"stop_{stop_label}_survival_rate"
        stop_rows = selected[
            (selected["expected_direction"] != 0)
            & selected[preferred_hit_column].notna()
            & selected[preferred_survival_column].notna()
        ]
        lines.extend(
            [
                "",
                "## Stop-Loss Diagnostics",
                "",
                (
                    "Target-side adverse-excursion diagnostics only. These use forward "
                    "MFE/MAE columns after the event and do not imply executable stop "
                    "placement."
                ),
                "",
                (
                    f"| template | regime | filter | n | {stop_label} hit | "
                    f"{stop_label} survived | median_score_bps |"
                ),
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in stop_rows.sort_values(
            ["retained_median_score", "retained_event_count"],
            ascending=[False, False],
        ).head(20).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["template_id"]),
                        f"{row['regime_field']}={row['regime_value']}",
                        str(row["filter_rule"]),
                        str(int(row["retained_event_count"])),
                        _format_pct(row[preferred_hit_column]),
                        _format_pct(row[preferred_survival_column]),
                        _format_score_bps(row["retained_median_score"]),
                    ]
                )
                + " |"
            )
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    return PersonalityTemplateResult(
        run_id=run_id,
        input_event_dir=resolved_input,
        template_path=template_path,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        decision_json_path=paths["decision_json"],
        template_csv_path=paths["templates"],
        base_summary_csv_path=paths["base"],
        candidate_rules_csv_path=paths["candidates"],
        selected_rules_csv_path=paths["selected"],
        rejected_rules_csv_path=paths["rejected"],
        template_matches_csv_path=paths["matches"],
        random_baseline_csv_path=paths["random"],
        concentration_warnings_csv_path=paths["concentration"],
        decision=decision,
        selected_rule_count=int(len(selected)),
    )
