"""Discovery, multiplicity, chronology, and economic metrics for frozen rules."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from stocker_research.directional_signature_atlas.signatures import (
    Condition,
    Direction,
    Signature,
    SupportRules,
    apply_multiple_testing,
    apply_signature,
    passes_support,
)


def signature_from_dict(payload: dict[str, Any]) -> Signature:
    return Signature(
        signature_id=str(payload["signature_id"]),
        direction=str(payload["direction"]),  # type: ignore[arg-type]
        conditions=tuple(Condition(**condition) for condition in payload["conditions"]),
        source=str(payload.get("source", "bounded_census")),
    )


def _payoff_columns(direction: Direction) -> tuple[str, str]:
    return (
        ("long_net_bps", "short_net_bps")
        if direction == "LONG"
        else ("short_net_bps", "long_net_bps")
    )


def _profit_factor(payoff: pd.Series) -> float:
    positive = float(payoff.loc[payoff > 0.0].sum())
    negative = abs(float(payoff.loc[payoff < 0.0].sum()))
    if negative == 0.0:
        return math.inf if positive > 0.0 else math.nan
    return positive / negative


def _maximum_drawdown(frame: pd.DataFrame, payoff_column: str) -> float:
    """Measure drawdown after netting simultaneous fixed-clock opportunities.

    Rows sharing a decision timestamp are concurrent, so their ordering by symbol
    must not manufacture an intratimestamp peak or trough.
    """

    if frame.empty:
        return math.nan
    if "decision_timestamp" in frame:
        payoff = frame.groupby("decision_timestamp", sort=True)[payoff_column].sum()
    elif {"session", "decision_clock"} <= set(frame):
        payoff = frame.groupby(["session", "decision_clock"], sort=True)[payoff_column].sum()
    else:
        payoff = pd.to_numeric(frame[payoff_column], errors="coerce")
    cumulative = payoff.fillna(0.0).cumsum().to_numpy(float)
    if not len(cumulative):
        return math.nan
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[:-1]
    return float(np.max(peaks - cumulative))


def _chronological_sort(frame: pd.DataFrame) -> pd.DataFrame:
    if "decision_timestamp" in frame:
        columns = ["decision_timestamp", "symbol", "opportunity_id"]
    else:
        columns = ["session", "decision_clock", "symbol", "opportunity_id"]
    available = [column for column in columns if column in frame]
    return frame.sort_values(available, kind="mergesort") if available else frame


def _concentration(
    selected: pd.DataFrame,
    payoff_column: str,
    group_column: str,
) -> tuple[float, float, float]:
    if selected.empty or group_column not in selected:
        return math.nan, math.nan, math.nan
    contribution = selected.groupby(group_column, sort=False)[payoff_column].sum().abs()
    total = float(contribution.sum())
    if total <= 0.0:
        return math.nan, math.nan, math.nan
    shares = contribution.sort_values(ascending=False) / total
    return float(shares.iloc[0]), float(shares.iloc[:5].sum()), float(np.square(shares).sum())


def signature_metrics(frame: pd.DataFrame, signature: Signature) -> dict[str, Any]:
    """Calculate classification, economic, breadth, and concentration metrics."""

    selected = _chronological_sort(frame.loc[apply_signature(frame, signature)].copy())
    direction = signature.direction
    payoff_column, opposite_payoff_column = _payoff_columns(direction)
    payoff = pd.to_numeric(selected[payoff_column], errors="coerce")
    opposite_payoff = pd.to_numeric(selected[opposite_payoff_column], errors="coerce")
    base_long = float(frame["target"].eq("LONG").mean()) if len(frame) else math.nan
    base_short = float(frame["target"].eq("SHORT").mean()) if len(frame) else math.nan
    base_neutral = float(frame["target"].eq("NEUTRAL").mean()) if len(frame) else math.nan
    long_rate = float(selected["target"].eq("LONG").mean()) if len(selected) else math.nan
    short_rate = float(selected["target"].eq("SHORT").mean()) if len(selected) else math.nan
    neutral_rate = float(selected["target"].eq("NEUTRAL").mean()) if len(selected) else math.nan
    direction_rate = long_rate if direction == "LONG" else short_rate
    base_direction = base_long if direction == "LONG" else base_short
    opposite_rate = short_rate if direction == "LONG" else long_rate
    base_opposite = base_short if direction == "LONG" else base_long
    top_stock, top_five_stock, stock_hhi = _concentration(selected, payoff_column, "symbol")
    top_month, top_five_month, month_hhi = _concentration(
        selected.assign(month=selected["session"].astype(str).str[:7]),
        payoff_column,
        "month",
    )
    monthly = (
        selected.assign(month=selected["session"].astype(str).str[:7])
        .groupby("month", sort=True)[payoff_column]
        .mean()
    )
    stock_means = selected.groupby("symbol", sort=True)[payoff_column].mean()
    maximum_stock_row_fraction = (
        float(selected["symbol"].value_counts(normalize=True).max())
        if len(selected)
        else math.nan
    )
    long_payoff = pd.to_numeric(selected["long_net_bps"], errors="coerce")
    short_payoff = pd.to_numeric(selected["short_net_bps"], errors="coerce")
    return {
        "signature_id": signature.signature_id,
        "direction": direction,
        "source": signature.source,
        "condition_count": len(signature.conditions),
        "conditions_json": json.dumps(
            [condition.__dict__ for condition in signature.conditions],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "rows": int(len(selected)),
        "sessions": int(selected["session"].nunique()),
        "stocks": int(selected["symbol"].nunique()),
        "months": int(selected["session"].astype(str).str[:7].nunique()),
        "long_count": int(selected["target"].eq("LONG").sum()),
        "short_count": int(selected["target"].eq("SHORT").sum()),
        "neutral_count": int(selected["target"].eq("NEUTRAL").sum()),
        "long_rate": long_rate,
        "short_rate": short_rate,
        "neutral_rate": neutral_rate,
        "base_long_rate": base_long,
        "base_short_rate": base_short,
        "base_neutral_rate": base_neutral,
        "long_lift": long_rate - base_long if len(selected) else math.nan,
        "short_lift": short_rate - base_short if len(selected) else math.nan,
        "directional_lift": direction_rate - base_direction if len(selected) else math.nan,
        "opposite_direction_excess": opposite_rate - base_opposite if len(selected) else math.nan,
        "mean_directional_net_bps": float(payoff.mean()) if len(payoff) else math.nan,
        "mean_opposite_net_bps": float(opposite_payoff.mean())
        if len(opposite_payoff)
        else math.nan,
        "mean_long_net_bps": float(long_payoff.mean()) if len(long_payoff) else math.nan,
        "mean_short_net_bps": float(short_payoff.mean()) if len(short_payoff) else math.nan,
        "median_long_net_bps": float(long_payoff.median()) if len(long_payoff) else math.nan,
        "median_short_net_bps": float(short_payoff.median()) if len(short_payoff) else math.nan,
        "median_directional_net_bps": float(payoff.median()) if len(payoff) else math.nan,
        "directional_economic_advantage_bps": float(payoff.mean() - opposite_payoff.mean())
        if len(payoff)
        else math.nan,
        "total_net_bps": float(payoff.sum()),
        "total_cost_bps": float(selected["round_trip_cost_bps"].sum()),
        "positive_payoff_rate": float(payoff.gt(0.0).mean()) if len(payoff) else math.nan,
        "profit_factor": _profit_factor(payoff),
        "maximum_drawdown_bps": _maximum_drawdown(selected, payoff_column),
        "coverage": float(len(selected) / len(frame)) if len(frame) else math.nan,
        "abstention": float(1.0 - len(selected) / len(frame)) if len(frame) else math.nan,
        "positive_month_fraction": float(monthly.gt(0.0).mean()) if len(monthly) else math.nan,
        "positive_stock_count": int(stock_means.gt(0.0).sum()),
        "positive_stock_fraction": float(stock_means.gt(0.0).mean())
        if len(stock_means)
        else math.nan,
        "maximum_single_stock_row_fraction": maximum_stock_row_fraction,
        "top_stock_absolute_contribution_share": top_stock,
        "top_five_stock_absolute_contribution_share": top_five_stock,
        "stock_contribution_hhi": stock_hhi,
        "top_month_absolute_contribution_share": top_month,
        "top_five_month_absolute_contribution_share": top_five_month,
        "month_contribution_hhi": month_hhi,
        "hindsight_episode_attribution_status": "unavailable_no_exact_episode_identity",
        "top_episode_absolute_contribution_share": math.nan,
        "top_five_episode_absolute_contribution_share": math.nan,
        "episode_contribution_hhi": math.nan,
        "twice_cost_mean_net_bps": float((payoff - selected["round_trip_cost_bps"]).mean())
        if len(payoff)
        else math.nan,
    }


def one_sided_session_p_value(frame: pd.DataFrame, signature: Signature) -> float:
    """Conservative one-sided session-level payoff/lift p-value."""

    selected = frame.loc[apply_signature(frame, signature)].copy()
    if selected.empty:
        return 1.0
    payoff_column, _ = _payoff_columns(signature.direction)
    payoff_by_session = selected.groupby("session", sort=True)[payoff_column].mean()
    base_rate = float(frame["target"].eq(signature.direction).mean())
    lift_by_session = (
        selected.assign(
            centered=selected["target"].eq(signature.direction).astype(float) - base_rate
        )
        .groupby("session", sort=True)["centered"]
        .mean()
    )

    def p_value(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
        if len(numeric) < 2:
            return 1.0
        if np.std(numeric) == 0.0:
            return 0.0 if float(np.mean(numeric)) > 0.0 else 1.0
        result = ttest_1samp(numeric, popmean=0.0, alternative="greater")
        return float(result.pvalue) if math.isfinite(float(result.pvalue)) else 1.0

    return max(p_value(payoff_by_session), p_value(lift_by_session))


def session_block_bootstrap_interval(
    frame: pd.DataFrame,
    signature: Signature,
    *,
    draws: int,
    seed: int,
    block_length: int = 5,
) -> dict[str, float]:
    """Moving-session-block interval for retained directional payoff."""

    selected = frame.loc[apply_signature(frame, signature)].copy()
    payoff_column, _ = _payoff_columns(signature.direction)
    session_values = selected.groupby("session", sort=True)[payoff_column].mean().to_numpy(float)
    if not len(session_values):
        return {
            "bootstrap_mean": math.nan,
            "bootstrap_lower": math.nan,
            "bootstrap_upper": math.nan,
        }
    rng = np.random.default_rng(seed)
    count = len(session_values)
    samples = np.empty(draws, dtype=float)
    max_start = max(1, count - block_length + 1)
    for draw in range(draws):
        positions: list[int] = []
        while len(positions) < count:
            start = int(rng.integers(0, max_start))
            positions.extend(range(start, min(start + block_length, count)))
        samples[draw] = float(np.mean(session_values[np.asarray(positions[:count])]))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "bootstrap_mean": float(np.mean(samples)),
        "bootstrap_lower": float(lower),
        "bootstrap_upper": float(upper),
    }


def _neighbour_signatures(
    signature: Signature,
    ordered_bins: dict[str, list[Any]],
) -> list[Signature]:
    neighbours: list[Signature] = []
    for condition_index, condition in enumerate(signature.conditions):
        values = ordered_bins.get(condition.feature)
        if not values or condition.operator != "==" or condition.value not in values:
            continue
        index = values.index(condition.value)
        for neighbour_index in (index - 1, index + 1):
            if not 0 <= neighbour_index < len(values):
                continue
            conditions = list(signature.conditions)
            conditions[condition_index] = replace(condition, value=values[neighbour_index])
            neighbours.append(
                Signature(
                    f"{signature.signature_id}__neighbour_{condition_index}_{neighbour_index}",
                    signature.direction,
                    tuple(conditions),
                    "threshold_neighbour",
                )
            )
    return neighbours


def neighbourhood_stability(
    frame: pd.DataFrame,
    signature: Signature,
    ordered_bins: dict[str, list[Any]],
) -> float:
    neighbours = _neighbour_signatures(signature, ordered_bins)
    if not neighbours:
        return 1.0
    effects = [
        signature_metrics(frame, neighbour)["mean_directional_net_bps"] for neighbour in neighbours
    ]
    finite = [float(effect) for effect in effects if math.isfinite(float(effect))]
    return float(np.mean(np.asarray(finite) > 0.0)) if finite else 0.0


def discovery_score(metrics: dict[str, Any], neighbourhood: float) -> float:
    effect = float(metrics["mean_directional_net_bps"]) / 10.0 + 2.0 * float(
        metrics["directional_lift"]
    )
    consistency = float(metrics["positive_month_fraction"])
    breadth = min(1.0, float(metrics["stocks"]) / 12.0) * min(
        1.0, float(metrics["sessions"]) / 60.0
    )
    support = min(1.0, math.sqrt(float(metrics["rows"]) / 200.0))
    if float(metrics["twice_cost_mean_net_bps"]) > 0.0:
        cost_survival = 1.0
    elif float(metrics["mean_directional_net_bps"]) > 0.0:
        cost_survival = 0.5
    else:
        cost_survival = 0.0
    stock_share = float(metrics["top_stock_absolute_contribution_share"])
    month_share = float(metrics["top_month_absolute_contribution_share"])
    stock_penalty = 2.0 * max(0.0, stock_share - 0.25) if math.isfinite(stock_share) else 2.0
    month_penalty = max(0.0, month_share - 0.35) if math.isfinite(month_share) else 1.0
    complexity = 0.15 * float(metrics["condition_count"])
    conflict = max(0.0, float(metrics["opposite_direction_excess"]))
    return (
        effect * consistency * breadth * support * cost_survival * neighbourhood
        - stock_penalty
        - month_penalty
        - complexity
        - conflict
    )


def evaluate_candidate_census(
    discovery: pd.DataFrame,
    candidates: list[Signature],
    registry: list[dict[str, Any]],
    *,
    support_rules: SupportRules,
    ordered_bins: dict[str, list[Any]],
    fdr_q: float,
) -> pd.DataFrame:
    """Score every examined candidate and retain every rejection reason."""

    registry_by_id = {str(row["signature_id"]): row for row in registry}
    rows: list[dict[str, Any]] = []
    supported_positions: list[int] = []
    p_values: list[float] = []
    for candidate in candidates:
        selected = discovery.loc[apply_signature(discovery, candidate)]
        supported, reasons = passes_support(selected, candidate.direction, support_rules)
        metrics = signature_metrics(discovery, candidate)
        p_value = one_sided_session_p_value(discovery, candidate) if supported else 1.0
        neighbourhood = neighbourhood_stability(discovery, candidate, ordered_bins)
        score = discovery_score(metrics, neighbourhood)
        base = registry_by_id.get(candidate.signature_id, {})
        row = {
            **base,
            **metrics,
            "raw_p_value": p_value,
            "neighbourhood_stability": neighbourhood,
            "discovery_score": score,
            "rejection_reasons": list(base.get("rejection_reasons", [])) + reasons,
        }
        rows.append(row)
        if supported:
            supported_positions.append(len(rows) - 1)
            p_values.append(p_value)
    adjusted = apply_multiple_testing(p_values, method="fdr_bh")
    for position, q_value in zip(supported_positions, adjusted, strict=True):
        rows[position]["fdr_q_value"] = q_value
    for row in rows:
        row.setdefault("fdr_q_value", 1.0)
        reasons = list(row["rejection_reasons"])
        if float(row["mean_directional_net_bps"]) <= 0.0:
            reasons.append("non_positive_discovery_payoff")
        if float(row["directional_lift"]) <= 0.0:
            reasons.append("non_positive_discovery_lift")
        if float(row["positive_month_fraction"]) <= 0.5:
            reasons.append("chronology_not_consistent")
        if float(row["positive_stock_fraction"]) <= 0.5:
            reasons.append("stock_effect_not_consistent")
        if float(row["top_stock_absolute_contribution_share"]) > 0.25:
            reasons.append("discovery_stock_payoff_concentration")
        if float(row["top_month_absolute_contribution_share"]) > 0.35:
            reasons.append("discovery_month_payoff_concentration")
        if float(row["opposite_direction_excess"]) > 0.0:
            reasons.append("opposite_direction_not_controlled")
        row["discovery_supported_effect"] = not reasons
        if float(row["fdr_q_value"]) > fdr_q:
            reasons.append("fdr_not_passed")
        row["rejection_reasons"] = sorted(set(reasons))
        row["discovery_eligible"] = not row["rejection_reasons"]
        row["rejection_reasons_json"] = json.dumps(row["rejection_reasons"], separators=(",", ":"))
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["discovery_eligible", "discovery_score", "rows", "condition_count", "signature_id"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def freeze_discovery_library(
    census: pd.DataFrame,
    candidates: list[Signature],
    discovery: pd.DataFrame,
    *,
    retained_stage_cap: int,
    per_direction_cap: int,
) -> list[dict[str, Any]]:
    by_id = {candidate.signature_id: candidate for candidate in candidates}
    eligible = census.loc[census["discovery_eligible"]].head(retained_stage_cap)
    library: list[dict[str, Any]] = []
    for direction in ("LONG", "SHORT"):
        selected = eligible.loc[eligible["direction"].eq(direction)].head(per_direction_cap)
        for row in selected.to_dict(orient="records"):
            signature = by_id[str(row["signature_id"])]
            fired = discovery.loc[apply_signature(discovery, signature)]
            counts = fired["target"].value_counts()
            denominator = len(fired) + 3.0
            library.append(
                {
                    "signature": signature.to_dict(),
                    "discovery_score": float(row["discovery_score"]),
                    "discovery_metrics": row,
                    "exploratory_due_to_multiplicity": not bool(row["discovery_eligible"]),
                    "frozen_class_probabilities": {
                        label: float((counts.get(label, 0) + 1.0) / denominator)
                        for label in ("LONG", "SHORT", "NEUTRAL")
                    },
                }
            )
    return library


def validate_discovery_library(
    validation: pd.DataFrame,
    library: list[dict[str, Any]],
    *,
    support_rules: SupportRules,
    holm_alpha: float,
    per_direction_cap: int,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for index, entry in enumerate(library):
        signature = signature_from_dict(entry["signature"])
        selected = validation.loc[apply_signature(validation, signature)]
        supported, reasons = passes_support(selected, signature.direction, support_rules)
        metrics = signature_metrics(validation, signature)
        p_value = one_sided_session_p_value(validation, signature) if supported else 1.0
        rows.append(
            {
                **metrics,
                "discovery_score": float(entry["discovery_score"]),
                "raw_p_value": p_value,
                "support_reasons": reasons,
                **session_block_bootstrap_interval(
                    validation,
                    signature,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed + index,
                ),
            }
        )
        p_values.append(p_value)
    holm = apply_multiple_testing(p_values, method="holm")
    survivors: list[dict[str, Any]] = []
    for entry, row, adjusted in zip(library, rows, holm, strict=True):
        discovery_metrics = entry["discovery_metrics"]
        reasons = list(row["support_reasons"])
        if not bool(discovery_metrics["discovery_eligible"]):
            reasons.append("discovery_multiplicity_not_passed")
        if float(row["mean_directional_net_bps"]) <= 0.0:
            reasons.append("validation_payoff_not_positive")
        if float(row["directional_lift"]) <= 0.0:
            reasons.append("validation_lift_not_positive")
        if float(row["positive_month_fraction"]) <= 0.5:
            reasons.append("validation_month_consistency_failed")
        if float(row["positive_stock_fraction"]) <= 0.5:
            reasons.append("validation_stock_consistency_failed")
        if float(row["twice_cost_mean_net_bps"]) <= 0.0:
            reasons.append("validation_twice_cost_failed")
        if float(row["top_stock_absolute_contribution_share"]) > 0.25:
            reasons.append("validation_stock_concentration")
        if float(row["top_month_absolute_contribution_share"]) > 0.35:
            reasons.append("validation_month_concentration")
        if float(row["opposite_direction_excess"]) > 0.0:
            reasons.append("validation_opposite_direction_not_controlled")
        if adjusted > holm_alpha:
            reasons.append("holm_not_passed")
        if np.sign(float(discovery_metrics["mean_directional_net_bps"])) != np.sign(
            float(row["mean_directional_net_bps"])
        ):
            reasons.append("effect_sign_changed")
        row["holm_adjusted_p_value"] = adjusted
        row["validation_rejection_reasons"] = sorted(set(reasons))
        row["validation_survived"] = not reasons
        row["validation_rejection_reasons_json"] = json.dumps(
            row["validation_rejection_reasons"], separators=(",", ":")
        )
        if not reasons:
            survivors.append(
                {
                    **entry,
                    "validation_metrics": row,
                    "conservative_value_bps": min(
                        float(discovery_metrics["mean_directional_net_bps"]),
                        float(row["mean_directional_net_bps"]),
                        float(row["twice_cost_mean_net_bps"]),
                    ),
                }
            )
    limited: list[dict[str, Any]] = []
    for direction in ("LONG", "SHORT"):
        directional = [entry for entry in survivors if entry["signature"]["direction"] == direction]
        directional.sort(
            key=lambda entry: (
                -float(entry["discovery_score"]),
                len(entry["signature"]["conditions"]),
                entry["signature"]["signature_id"],
            )
        )
        limited.extend(directional[:per_direction_cap])
    return pd.DataFrame(rows), limited


def score_frozen_library(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows = []
    for index, entry in enumerate(library):
        signature = signature_from_dict(entry["signature"])
        rows.append(
            {
                **signature_metrics(frame, signature),
                **session_block_bootstrap_interval(
                    frame,
                    signature,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed + index,
                ),
                "discovery_score": float(entry["discovery_score"]),
                "conservative_value_bps": float(entry["conservative_value_bps"]),
            }
        )
    return pd.DataFrame(rows)


def signature_probability_metrics(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
) -> pd.DataFrame:
    """Evaluate each discovery-frozen probability vector unchanged."""

    rows: list[dict[str, Any]] = []
    for entry in library:
        signature = signature_from_dict(entry["signature"])
        selected = frame.loc[apply_signature(frame, signature)]
        frozen = entry["frozen_class_probabilities"]
        if selected.empty:
            continue
        probability = np.asarray(
            [float(frozen[label]) for label in ("LONG", "SHORT", "NEUTRAL")], dtype=float
        )
        probability /= probability.sum()
        actual = np.column_stack(
            [selected["target"].eq(label).to_numpy(float) for label in ("LONG", "SHORT", "NEUTRAL")]
        )
        base = np.asarray(
            [float(frame["target"].eq(label).mean()) for label in ("LONG", "SHORT", "NEUTRAL")]
        )
        brier = np.mean(np.square(actual - probability), axis=0)
        base_brier = np.mean(np.square(actual - base), axis=0)
        observed = actual.mean(axis=0)
        rows.append(
            {
                "signature_id": signature.signature_id,
                "direction": signature.direction,
                "chronology_stage": str(frame["chronology_stage"].iloc[0])
                if "chronology_stage" in frame
                else str(frame["period"].iloc[0]),
                "rows": len(selected),
                "brier_long": float(brier[0]),
                "brier_short": float(brier[1]),
                "brier_neutral": float(brier[2]),
                "macro_brier": float(np.mean(brier)),
                "directional_macro_brier": float(np.mean(brier[:2])),
                "base_macro_brier": float(np.mean(base_brier)),
                "log_loss": float(
                    -np.mean(np.log(np.clip((actual * probability).sum(axis=1), 1e-12, 1.0)))
                ),
                "ece_long": abs(float(probability[0] - observed[0])),
                "ece_short": abs(float(probability[1] - observed[1])),
                "ece_neutral": abs(float(probability[2] - observed[2])),
                "calibration_slope_long": math.nan,
                "calibration_slope_short": math.nan,
                "calibration_slope_status": "unidentifiable_constant_signature_probability",
                "reasonably_calibrated": bool(
                    float(np.mean(brier)) <= float(np.mean(base_brier))
                    and abs(float(probability[0] - observed[0])) <= 0.10
                    and abs(float(probability[1] - observed[1])) <= 0.10
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_simple_baseline_metrics(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
) -> pd.DataFrame:
    """Compare a rule with momentum/reversal on the identical fired rows."""

    rows: list[dict[str, Any]] = []
    one_bar = pd.to_numeric(frame["return_1_scale"], errors="coerce")
    momentum = np.where(one_bar > 0.0, "LONG", np.where(one_bar < 0.0, "SHORT", "NEUTRAL"))
    reversal = np.where(one_bar > 0.0, "SHORT", np.where(one_bar < 0.0, "LONG", "NEUTRAL"))
    for entry in library:
        signature = signature_from_dict(entry["signature"])
        mask = apply_signature(frame, signature).to_numpy(bool)
        selected = frame.loc[mask]
        if selected.empty:
            continue
        signature_payoff = selected[
            "long_net_bps" if signature.direction == "LONG" else "short_net_bps"
        ].to_numpy(float)
        selected_momentum = momentum[mask]
        selected_reversal = reversal[mask]
        selected_long = selected["long_net_bps"].to_numpy(float)
        selected_short = selected["short_net_bps"].to_numpy(float)

        def payoff(
            states: np.ndarray,
            long_values: np.ndarray = selected_long,
            short_values: np.ndarray = selected_short,
        ) -> np.ndarray:
            return np.where(
                states == "LONG",
                long_values,
                np.where(states == "SHORT", short_values, 0.0),
            )

        momentum_payoff = payoff(selected_momentum)
        reversal_payoff = payoff(selected_reversal)
        rows.append(
            {
                "signature_id": signature.signature_id,
                "direction": signature.direction,
                "chronology_stage": str(frame["chronology_stage"].iloc[0])
                if "chronology_stage" in frame
                else str(frame["period"].iloc[0]),
                "rows": len(selected),
                "signature_mean_net_bps": float(np.mean(signature_payoff)),
                "momentum_mean_net_bps_same_rows": float(np.mean(momentum_payoff)),
                "reversal_mean_net_bps_same_rows": float(np.mean(reversal_payoff)),
                "stronger_than_momentum": bool(
                    float(np.mean(signature_payoff)) > float(np.mean(momentum_payoff))
                ),
                "stronger_than_reversal": bool(
                    float(np.mean(signature_payoff)) > float(np.mean(reversal_payoff))
                ),
            }
        )
    return pd.DataFrame(rows)


def signature_breakdowns(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = [
        "period",
        "symbol",
        "decision_clock",
        "current_state",
        "state_motif_3",
        "parent_loop_family",
        "top_parent_loop",
        "top_loop_orientation",
    ]
    for entry in library:
        signature = signature_from_dict(entry["signature"])
        selected = frame.loc[apply_signature(frame, signature)].copy()
        payoff_column, _ = _payoff_columns(signature.direction)
        selected["month"] = selected["session"].astype(str).str[:7]
        for dimension in [*dimensions, "month"]:
            if dimension not in selected:
                continue
            contribution = selected.groupby(dimension, dropna=False, sort=True)[payoff_column].sum()
            absolute = contribution.abs().sort_values(ascending=False)
            total = float(absolute.sum())
            shares = absolute / total if total > 0.0 else absolute * math.nan
            rows.append(
                {
                    "scope": "individual_signature",
                    "signature_id": signature.signature_id,
                    "direction": signature.direction,
                    "dimension": dimension,
                    "status": "available",
                    "groups": int(len(contribution)),
                    "top_one_absolute_contribution_share": float(shares.iloc[0])
                    if len(shares)
                    else math.nan,
                    "top_five_absolute_contribution_share": float(shares.iloc[:5].sum())
                    if len(shares)
                    else math.nan,
                    "herfindahl_concentration": float(np.square(shares).sum())
                    if len(shares)
                    else math.nan,
                    "total_net_bps": float(selected[payoff_column].sum()),
                }
            )
        for dimension, status in (
            ("sector", "unavailable_no_frozen_sector_membership"),
            ("hindsight_episode", "unavailable_no_exact_episode_identity"),
        ):
            rows.append(
                {
                    "scope": "individual_signature",
                    "signature_id": signature.signature_id,
                    "direction": signature.direction,
                    "dimension": dimension,
                    "status": status,
                    "groups": 0,
                    "top_one_absolute_contribution_share": math.nan,
                    "top_five_absolute_contribution_share": math.nan,
                    "herfindahl_concentration": math.nan,
                    "total_net_bps": math.nan,
                }
            )
    return pd.DataFrame(rows)


def signature_attribution_rows(
    frame: pd.DataFrame,
    library: list[dict[str, Any]],
) -> pd.DataFrame:
    """Emit auditable per-value effects for every frozen exploratory rule."""

    rows: list[dict[str, Any]] = []
    dimensions = (
        "symbol",
        "decision_clock",
        "current_state",
        "state_motif_3",
        "parent_loop_family",
        "top_parent_loop",
        "top_loop_orientation",
    )
    for entry in library:
        signature = signature_from_dict(entry["signature"])
        selected = frame.loc[apply_signature(frame, signature)].copy()
        payoff_column, _ = _payoff_columns(signature.direction)
        selected["month"] = selected["session"].astype(str).str[:7]
        for stage, stage_rows in selected.groupby("chronology_stage", sort=True):
            base_frame = frame.loc[frame["chronology_stage"].eq(stage)]
            base_rate = float(base_frame["target"].eq(signature.direction).mean())
            for dimension in (*dimensions, "month"):
                for value, group in stage_rows.groupby(dimension, dropna=False, sort=True):
                    rows.append(
                        {
                            "signature_id": signature.signature_id,
                            "direction": signature.direction,
                            "chronology_stage": str(stage),
                            "dimension": dimension,
                            "value": str(value),
                            "rows": len(group),
                            "sessions": group["session"].nunique(),
                            "stocks": group["symbol"].nunique(),
                            "mean_net_bps": float(group[payoff_column].mean()),
                            "total_net_bps": float(group[payoff_column].sum()),
                            "directional_rate": float(
                                group["target"].eq(signature.direction).mean()
                            ),
                            "directional_lift": float(
                                group["target"].eq(signature.direction).mean() - base_rate
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def atlas_concentration(
    frame: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Concentrate realised controller payoff; overlaps receive equal attribution."""

    joined = frame.merge(
        decisions[
            [
                "opportunity_id",
                "predicted_state",
                "firing_signature_ids_json",
            ]
        ],
        on="opportunity_id",
        how="inner",
        validate="one_to_one",
    )
    joined["atlas_payoff_bps"] = np.where(
        joined["predicted_state"].eq("LONG"),
        joined["long_net_bps"],
        np.where(joined["predicted_state"].eq("SHORT"), joined["short_net_bps"], 0.0),
    )
    joined["month"] = joined["session"].astype(str).str[:7]
    dimensions = {
        "direction": "predicted_state",
        "stock": "symbol",
        "period": "chronology_stage",
        "month": "month",
        "clock": "decision_clock",
        "state": "current_state",
        "state_history_motif": "state_motif_3",
        "loop_family": "parent_loop_family",
    }
    rows: list[dict[str, Any]] = []

    def summarize(dimension: str, contribution: pd.Series, status: str = "available") -> None:
        absolute = contribution.abs().sort_values(ascending=False)
        total = float(absolute.sum())
        shares = absolute / total if total > 0.0 else absolute * math.nan
        rows.append(
            {
                "scope": "atlas_controller",
                "signature_id": "__ATLAS__",
                "direction": "ALL",
                "dimension": dimension,
                "status": status,
                "groups": len(contribution),
                "top_one_absolute_contribution_share": float(shares.iloc[0])
                if len(shares)
                else math.nan,
                "top_five_absolute_contribution_share": float(shares.iloc[:5].sum())
                if len(shares)
                else math.nan,
                "herfindahl_concentration": float(np.square(shares).sum())
                if len(shares)
                else math.nan,
                "total_net_bps": float(joined["atlas_payoff_bps"].sum()),
                "overlap_attribution": "equal_split_across_firing_signatures",
            }
        )

    for label, column in dimensions.items():
        summarize(
            label,
            joined.groupby(column, dropna=False, sort=True)["atlas_payoff_bps"].sum(),
        )
    signature_contributions: dict[str, float] = {}
    for row in joined.itertuples(index=False):
        ids = json.loads(str(row.firing_signature_ids_json))
        if not ids or float(cast(Any, row.atlas_payoff_bps)) == 0.0:
            continue
        share = float(cast(Any, row.atlas_payoff_bps)) / len(ids)
        for signature_id in ids:
            signature_contributions[str(signature_id)] = (
                signature_contributions.get(str(signature_id), 0.0) + share
            )
    summarize("signature", pd.Series(signature_contributions, dtype=float))
    for dimension, status in (
        ("sector", "unavailable_no_frozen_sector_membership"),
        ("hindsight_episode", "unavailable_no_exact_episode_identity"),
    ):
        summarize(dimension, pd.Series(dtype=float), status)
    return pd.DataFrame(rows)


def neutral_veto_metrics(
    frame: pd.DataFrame,
    signature: Signature,
) -> dict[str, Any]:
    """Measure a veto as excess neutral incidence, never as a trade."""

    selected = frame.loc[apply_signature(frame, signature)].copy()
    base_neutral = float(frame["target"].eq("NEUTRAL").mean()) if len(frame) else math.nan
    selected["neutral_excess"] = selected["target"].eq("NEUTRAL").astype(float) - base_neutral
    selected["month"] = selected["session"].astype(str).str[:7]
    monthly = selected.groupby("month", sort=True)["neutral_excess"].mean()
    top_stock, top_five_stock, stock_hhi = _concentration(selected, "neutral_excess", "symbol")
    top_month, top_five_month, month_hhi = _concentration(selected, "neutral_excess", "month")
    return {
        "rows": int(len(selected)),
        "sessions": int(selected["session"].nunique()),
        "stocks": int(selected["symbol"].nunique()),
        "months": int(selected["month"].nunique()),
        "neutral_count": int(selected["target"].eq("NEUTRAL").sum()),
        "neutral_rate": float(selected["target"].eq("NEUTRAL").mean())
        if len(selected)
        else math.nan,
        "base_neutral_rate": base_neutral,
        "neutral_lift": float(selected["target"].eq("NEUTRAL").mean() - base_neutral)
        if len(selected)
        else math.nan,
        "mean_long_net_bps": float(selected["long_net_bps"].mean()) if len(selected) else math.nan,
        "mean_short_net_bps": float(selected["short_net_bps"].mean())
        if len(selected)
        else math.nan,
        "positive_month_fraction": float(monthly.gt(0.0).mean()) if len(monthly) else math.nan,
        "top_stock_absolute_contribution_share": top_stock,
        "top_five_stock_absolute_contribution_share": top_five_stock,
        "stock_contribution_hhi": stock_hhi,
        "top_month_absolute_contribution_share": top_month,
        "top_five_month_absolute_contribution_share": top_five_month,
        "month_contribution_hhi": month_hhi,
        "hindsight_episode_attribution_status": "unavailable_no_exact_episode_identity",
    }


def evaluate_neutral_veto_census(
    discovery: pd.DataFrame,
    candidates: list[Signature],
    *,
    support_rules: SupportRules,
    fdr_q: float,
    ordered_bins: dict[str, list[Any]],
    cap: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Discover first-class neutral vetoes from the already-bounded condition census."""

    unique: dict[str, Signature] = {}
    for candidate in candidates:
        condition_key = json.dumps(
            [condition.__dict__ for condition in candidate.conditions],
            sort_keys=True,
            separators=(",", ":"),
        )
        unique.setdefault(condition_key, candidate)
    rows: list[dict[str, Any]] = []
    supported_positions: list[int] = []
    p_values: list[float] = []
    base_neutral = float(discovery["target"].eq("NEUTRAL").mean())
    for key, candidate in unique.items():
        selected = discovery.loc[apply_signature(discovery, candidate)].copy()
        support_target = selected.assign(
            target=np.where(selected["target"].eq("NEUTRAL"), "LONG", "SHORT")
        )
        supported, reasons = passes_support(support_target, "LONG", support_rules)
        veto_metrics = neutral_veto_metrics(discovery, candidate)
        neutral_rate = float(veto_metrics["neutral_rate"])
        monthly = (
            selected.assign(month=selected["session"].astype(str).str[:7])
            .groupby("month", sort=True)["target"]
            .apply(lambda values: float(values.eq("NEUTRAL").mean() - base_neutral))
        )
        centered = (
            selected.assign(
                neutral_centered=selected["target"].eq("NEUTRAL").astype(float) - base_neutral
            )
            .groupby("session", sort=True)["neutral_centered"]
            .mean()
        )
        if supported and len(centered) >= 2 and float(centered.std()) > 0.0:
            p_value = float(
                ttest_1samp(centered.to_numpy(float), popmean=0.0, alternative="greater").pvalue
            )
        else:
            p_value = 1.0
        neighbours = _neighbour_signatures(candidate, ordered_bins)
        neighbour_lifts = [
            float(neutral_veto_metrics(discovery, neighbour)["neutral_lift"])
            for neighbour in neighbours
        ]
        neighbourhood = (
            float(np.mean(np.asarray(neighbour_lifts) > 0.0)) if neighbour_lifts else 1.0
        )
        score = (neutral_rate - base_neutral) * min(1.0, len(selected) / 200.0) * min(
            1.0, selected["symbol"].nunique() / 12.0
        ) * float(monthly.gt(0.0).mean() if len(monthly) else 0.0) * neighbourhood - 0.15 * len(
            candidate.conditions
        )
        row = {
            "neutral_veto_id": f"neutral_veto__{hashlib_sha(key)[:12]}",
            "conditions_json": key,
            "condition_count": len(candidate.conditions),
            **veto_metrics,
            "raw_p_value": p_value,
            "neutral_score": score,
            "neighbourhood_stability": neighbourhood,
            "support_reasons": reasons,
            "signature": candidate.to_dict(),
        }
        rows.append(row)
        if supported:
            supported_positions.append(len(rows) - 1)
            p_values.append(p_value)
    adjusted = apply_multiple_testing(p_values, method="fdr_bh")
    for position, q_value in zip(supported_positions, adjusted, strict=True):
        rows[position]["fdr_q_value"] = q_value
    for row in rows:
        row.setdefault("fdr_q_value", 1.0)
        reasons = list(cast(list[str], row["support_reasons"]))
        if float(cast(Any, row["neutral_lift"])) <= 0.0:
            reasons.append("neutral_lift_not_positive")
        if (
            float(cast(Any, row["mean_long_net_bps"])) >= 0.0
            or float(cast(Any, row["mean_short_net_bps"])) >= 0.0
        ):
            reasons.append("directional_payoff_not_suppressed")
        if float(cast(Any, row["positive_month_fraction"])) <= 0.5:
            reasons.append("neutral_chronology_not_consistent")
        if float(cast(Any, row["neighbourhood_stability"])) < 0.5:
            reasons.append("neutral_threshold_isolated")
        if float(cast(Any, row["top_stock_absolute_contribution_share"])) > 0.25:
            reasons.append("neutral_stock_concentration")
        if float(cast(Any, row["top_month_absolute_contribution_share"])) > 0.35:
            reasons.append("neutral_month_concentration")
        if float(cast(Any, row["fdr_q_value"])) > fdr_q:
            reasons.append("fdr_not_passed")
        row["rejection_reasons"] = sorted(set(reasons))
        row["neutral_discovery_eligible"] = not reasons
        row["rejection_reasons_json"] = json.dumps(reasons, separators=(",", ":"))
    census = pd.DataFrame(rows).sort_values(
        ["neutral_discovery_eligible", "neutral_score", "rows", "neutral_veto_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    library = [
        {
            "neutral_veto_id": row["neutral_veto_id"],
            "signature": row["signature"],
            "discovery_metrics": row,
        }
        for row in census.loc[census["neutral_discovery_eligible"]]
        .head(cap)
        .to_dict(orient="records")
    ]
    return census.reset_index(drop=True), library


def validate_neutral_veto_library(
    validation: pd.DataFrame,
    library: list[dict[str, Any]],
    *,
    support_rules: SupportRules,
    holm_alpha: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    base_neutral = float(validation["target"].eq("NEUTRAL").mean())
    for entry in library:
        signature = signature_from_dict(entry["signature"])
        selected = validation.loc[apply_signature(validation, signature)].copy()
        support_target = selected.assign(
            target=np.where(selected["target"].eq("NEUTRAL"), "LONG", "SHORT")
        )
        supported, reasons = passes_support(support_target, "LONG", support_rules)
        centered = (
            selected.assign(
                neutral_centered=selected["target"].eq("NEUTRAL").astype(float) - base_neutral
            )
            .groupby("session", sort=True)["neutral_centered"]
            .mean()
        )
        p_value = (
            float(ttest_1samp(centered.to_numpy(float), popmean=0.0, alternative="greater").pvalue)
            if supported and len(centered) >= 2 and float(centered.std()) > 0.0
            else 1.0
        )
        veto_metrics = neutral_veto_metrics(validation, signature)
        rows.append(
            {
                "neutral_veto_id": entry["neutral_veto_id"],
                **veto_metrics,
                "raw_p_value": p_value,
                "support_reasons": reasons,
            }
        )
    holm = apply_multiple_testing([float(row["raw_p_value"]) for row in rows], method="holm")
    survivors: list[dict[str, Any]] = []
    for entry, row, adjusted in zip(library, rows, holm, strict=True):
        reasons = list(row["support_reasons"])
        if float(row["neutral_lift"]) <= 0.0:
            reasons.append("validation_neutral_lift_not_positive")
        if float(row["mean_long_net_bps"]) >= 0.0 or float(row["mean_short_net_bps"]) >= 0.0:
            reasons.append("validation_directional_payoff_not_suppressed")
        if float(row["positive_month_fraction"]) <= 0.5:
            reasons.append("validation_neutral_month_consistency_failed")
        if float(row["top_stock_absolute_contribution_share"]) > 0.25:
            reasons.append("validation_neutral_stock_concentration")
        if float(row["top_month_absolute_contribution_share"]) > 0.35:
            reasons.append("validation_neutral_month_concentration")
        if adjusted > holm_alpha:
            reasons.append("validation_holm_not_passed")
        row["holm_adjusted_p_value"] = adjusted
        row["validation_rejection_reasons"] = sorted(set(reasons))
        row["neutral_validation_survived"] = not reasons
        if not reasons:
            survivors.append({**entry, "validation_metrics": row})
    return pd.DataFrame(rows), survivors


def hashlib_sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
