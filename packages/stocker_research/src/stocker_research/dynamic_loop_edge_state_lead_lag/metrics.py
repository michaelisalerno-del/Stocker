"""Predeclared lead-lag metrics for frozen V2 forecast pairs."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PAIR_KEYS = (
    "period",
    "score_session",
    "loop_id",
    "orientation",
    "horizon",
    "target_lead_sessions",
    "target_session",
)
CONTROL_MODEL = "hierarchical_payoff_history_change_point"
FULL_MODEL = "hierarchical_change_point"


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _equal(left: pd.Series, right: pd.Series) -> pd.Series:
    numeric_left = pd.to_numeric(left, errors="coerce")
    numeric_right = pd.to_numeric(right, errors="coerce")
    numeric = left.notna() & right.notna() & numeric_left.notna() & numeric_right.notna()
    result = left.astype("string").eq(right.astype("string")) | (left.isna() & right.isna())
    result.loc[numeric] = np.isclose(
        numeric_left.loc[numeric], numeric_right.loc[numeric], rtol=0.0, atol=1e-12
    )
    return result.fillna(False)


def validate_paired_training_identity(
    forecasts: pd.DataFrame,
    training_fields: Sequence[str],
    *,
    control_model: str = CONTROL_MODEL,
    full_model: str = FULL_MODEL,
) -> None:
    """Require identical causal training populations before feature attribution."""

    keys = ["period", "score_session", "loop_id", "orientation", "horizon"]
    _require(forecasts, {*keys, "model_name", *training_fields}, "paired forecast")
    control = forecasts.loc[forecasts["model_name"].eq(control_model), [*keys, *training_fields]]
    full = forecasts.loc[forecasts["model_name"].eq(full_model), [*keys, *training_fields]]
    paired = control.merge(
        full, on=keys, how="outer", suffixes=("__control", "__full"), indicator=True
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("training-state mismatch: full/control forecast populations differ")
    mismatches: list[str] = []
    for field in training_fields:
        if not _equal(paired[f"{field}__control"], paired[f"{field}__full"]).all():
            mismatches.append(field)
    if mismatches:
        raise ValueError(f"training-state mismatch in fields: {mismatches}")


def build_paired_prediction_table(
    lead_joins: pd.DataFrame,
    *,
    control_model: str = CONTROL_MODEL,
    full_model: str = FULL_MODEL,
) -> pd.DataFrame:
    """Create the exact shared target population for the frozen hierarchy pair."""

    predictions = (
        "forecast_id",
        "p_next_payoff_positive",
        "p_edge_positive",
        "p_edge_active",
        "p_on_next",
        "p_off_next",
        "p_survive_horizon",
        "posterior_mean_net_bps",
        "posterior_lower_bound_net_bps",
        "edge_state",
    )
    targets = (
        "target_outcome_id",
        "target_status",
        "target_payoff_available",
        "target_payoff_positive",
        "target_robust_net_bps",
        "target_robust_gross_bps",
        "target_cost_contribution_bps",
        "target_independent_stocks",
        "target_independent_stock_ids",
        "target_effective_sample_size",
        "target_episode_state",
        "target_episode_id",
        "target_episode_onset_within_lead",
        "target_episode_survival",
    )
    _require(lead_joins, {*PAIR_KEYS, "model_name", *predictions, *targets}, "lead join")
    control = lead_joins.loc[
        lead_joins["model_name"].eq(control_model), [*PAIR_KEYS, *predictions, *targets]
    ]
    full = lead_joins.loc[
        lead_joins["model_name"].eq(full_model), [*PAIR_KEYS, *predictions, *targets]
    ]
    paired = control.merge(
        full,
        on=list(PAIR_KEYS),
        how="inner",
        validate="one_to_one",
        suffixes=("__control", "__full"),
    )
    for target in targets:
        control_field = f"{target}__control"
        full_field = f"{target}__full"
        if not _equal(paired[control_field], paired[full_field]).all():
            raise ValueError(f"paired target mismatch: {target}")
        paired[target] = paired[control_field]
    paired["feature_contribution_p_next"] = (
        paired["p_next_payoff_positive__full"] - paired["p_next_payoff_positive__control"]
    )
    paired["feature_contribution_posterior_mean_bps"] = (
        paired["posterior_mean_net_bps__full"] - paired["posterior_mean_net_bps__control"]
    )
    paired["feature_contribution_p_active"] = (
        paired["p_edge_active__full"] - paired["p_edge_active__control"]
    )
    paired["feature_contribution_p_on_next"] = (
        paired["p_on_next__full"] - paired["p_on_next__control"]
    )
    paired["feature_contribution_p_survive"] = (
        paired["p_survive_horizon__full"] - paired["p_survive_horizon__control"]
    )
    return paired.sort_values(list(PAIR_KEYS), kind="stable").reset_index(drop=True)


def _clip_probability(values: pd.Series) -> np.ndarray:
    return cast(
        np.ndarray,
        np.clip(
            pd.to_numeric(values, errors="raise").to_numpy(dtype=float),
            1e-12,
            1 - 1e-12,
        ),
    )


def _binary_log_loss(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return cast(
        np.ndarray,
        -(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)),
    )


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, *, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    total = len(y)
    if total == 0:
        return float("nan")
    error = 0.0
    for bin_index in range(bins):
        selected = assignments == bin_index
        if selected.any():
            error += selected.sum() / total * abs(y[selected].mean() - probability[selected].mean())
    return float(error)


def _calibration_fit(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2 or np.isclose(probability.std(), 0.0):
        return float("nan"), float("nan")
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=2_000)
    model.fit(logit, y.astype(int))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _attach_operational_onset_predictions(lead_joins: pd.DataFrame) -> pd.DataFrame:
    cell = ["model_name", "period", "loop_id", "orientation", "horizon"]
    state = (
        lead_joins.loc[:, [*cell, "score_session", "edge_state"]]
        .drop_duplicates([*cell, "score_session"])
        .sort_values([*cell, "score_session"], kind="stable")
    )
    previous = state.groupby(cell, observed=True)["edge_state"].shift(1)
    state["onset_operational_prediction"] = (
        state["edge_state"].eq("active") & previous.notna() & previous.ne("active")
    )
    return lead_joins.merge(
        state.loc[:, [*cell, "score_session", "onset_operational_prediction"]],
        on=[*cell, "score_session"],
        how="left",
        validate="many_to_one",
    )


def lead_calibration_metrics(lead_joins: pd.DataFrame) -> pd.DataFrame:
    """Compute prequential calibration and frozen-active classification by lead."""

    _require(
        lead_joins,
        {
            "model_name",
            "target_lead_sessions",
            "target_payoff_available",
            "target_payoff_positive",
            "target_robust_net_bps",
            "p_next_payoff_positive",
            "p_on_next",
            "p_survive_horizon",
            "edge_state",
            "target_episode_onset_within_lead",
            "target_episode_survival",
            "period",
            "score_session",
            "loop_id",
            "orientation",
            "horizon",
        },
        "calibration",
    )
    scored = _attach_operational_onset_predictions(lead_joins)
    records: list[dict[str, object]] = []
    for (model_name, lead), group in scored.groupby(
        ["model_name", "target_lead_sessions"], observed=True, sort=True
    ):
        observed = group.loc[group["target_payoff_available"].eq(True)].copy()  # noqa: E712
        record: dict[str, object] = {
            "model_name": model_name,
            "target_lead_sessions": int(str(lead)),
            "eligible_forecasts": len(group),
            "observable_targets": len(observed),
            "coverage": len(observed) / len(group) if len(group) else np.nan,
        }
        if observed.empty:
            records.append(record)
            continue
        y = observed["target_payoff_positive"].astype(int).to_numpy(dtype=float)
        probability = _clip_probability(observed["p_next_payoff_positive"])
        active = observed["edge_state"].eq("active").to_numpy()
        slope, intercept = _calibration_fit(y, probability)
        true_positive = int(np.sum(active & (y == 1)))
        predicted_positive = int(active.sum())
        actual_positive = int(y.sum())
        onset_target = (
            observed["target_episode_onset_within_lead"].fillna(False).astype(bool).to_numpy()
        )
        onset_prediction = (
            observed["onset_operational_prediction"].fillna(False).to_numpy(dtype=bool)
        )
        onset_probability = _clip_probability(observed["p_on_next"])
        survival_target = (
            observed["target_episode_survival"].fillna(False).astype(bool).to_numpy(dtype=float)
        )
        survival_probability = _clip_probability(observed["p_survive_horizon"])
        onset_slope, onset_intercept = _calibration_fit(
            onset_target.astype(float), onset_probability
        )
        survival_slope, survival_intercept = _calibration_fit(survival_target, survival_probability)
        record.update(
            {
                "brier_score": float(np.mean((probability - y) ** 2)),
                "log_loss": float(np.mean(_binary_log_loss(y, probability))),
                "ece": expected_calibration_error(y, probability),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "auc": (float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan),
                "active_count": predicted_positive,
                "precision_active": true_positive / predicted_positive
                if predicted_positive
                else np.nan,
                "recall_active": true_positive / actual_positive if actual_positive else np.nan,
                "mean_target_payoff_when_active_bps": float(
                    observed.loc[active, "target_robust_net_bps"].mean()
                ),
                "median_target_payoff_when_active_bps": float(
                    observed.loc[active, "target_robust_net_bps"].median()
                ),
                "positive_payoff_rate": float(y.mean()),
                "abstention_rate": float(observed["edge_state"].eq("unknown").mean()),
                "onset_operational_predictions": int(onset_prediction.sum()),
                "onset_precision": (
                    float(np.sum(onset_prediction & onset_target) / onset_prediction.sum())
                    if onset_prediction.sum()
                    else np.nan
                ),
                "onset_recall": (
                    float(np.sum(onset_prediction & onset_target) / onset_target.sum())
                    if onset_target.sum()
                    else np.nan
                ),
                "false_onset_rate": (
                    float(np.sum(onset_prediction & ~onset_target) / onset_prediction.sum())
                    if onset_prediction.sum()
                    else np.nan
                ),
                "onset_probability_brier_score": float(
                    np.mean((onset_probability - onset_target.astype(float)) ** 2)
                ),
                "onset_probability_log_loss": float(
                    np.mean(_binary_log_loss(onset_target.astype(float), onset_probability))
                ),
                "onset_probability_ece": expected_calibration_error(
                    onset_target.astype(float), onset_probability
                ),
                "onset_probability_calibration_slope": onset_slope,
                "onset_probability_calibration_intercept": onset_intercept,
                "survival_observations": len(survival_target),
                "survival_brier_score": float(
                    np.mean((survival_probability - survival_target) ** 2)
                ),
                "survival_log_loss": float(
                    np.mean(_binary_log_loss(survival_target, survival_probability))
                ),
                "survival_ece": expected_calibration_error(survival_target, survival_probability),
                "survival_calibration_slope": survival_slope,
                "survival_calibration_intercept": survival_intercept,
                "survival_auc": (
                    float(roc_auc_score(survival_target, survival_probability))
                    if len(np.unique(survival_target)) == 2
                    else np.nan
                ),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _paired_components(group: pd.DataFrame) -> pd.DataFrame:
    observed = group.loc[group["target_payoff_available"].eq(True)].copy()  # noqa: E712
    y = observed["target_payoff_positive"].astype(int).to_numpy(dtype=float)
    full = _clip_probability(observed["p_next_payoff_positive__full"])
    control = _clip_probability(observed["p_next_payoff_positive__control"])
    payoff = observed["target_robust_net_bps"].to_numpy(dtype=float)
    full_mean = observed["posterior_mean_net_bps__full"].to_numpy(dtype=float)
    control_mean = observed["posterior_mean_net_bps__control"].to_numpy(dtype=float)
    full_active = observed["edge_state__full"].eq("active").to_numpy(dtype=int)
    control_active = observed["edge_state__control"].eq("active").to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "period": observed["period"].to_numpy(),
            "score_session": observed["score_session"].to_numpy(),
            "brier": (control - y) ** 2 - (full - y) ** 2,
            "log_loss": _binary_log_loss(y, control) - _binary_log_loss(y, full),
            "absolute_error": np.abs(control_mean - payoff) - np.abs(full_mean - payoff),
            "economic": payoff * (full_active - control_active),
        }
    )


def _session_block_bootstrap(
    components: pd.DataFrame, *, resamples: int, seed: int
) -> dict[str, float]:
    if components.empty or resamples <= 0:
        return {}
    blocks = (
        components.groupby(["period", "score_session"], observed=True, sort=True)
        .agg(
            brier_sum=("brier", "sum"),
            log_loss_sum=("log_loss", "sum"),
            absolute_error_sum=("absolute_error", "sum"),
            economic_sum=("economic", "sum"),
            row_count=("brier", "size"),
        )
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), size=(resamples, len(blocks)))
    counts = blocks["row_count"].to_numpy()[draws].sum(axis=1)
    samples: dict[str, np.ndarray] = {}
    for metric in ("brier", "log_loss", "absolute_error"):
        samples[metric] = blocks[f"{metric}_sum"].to_numpy()[draws].sum(axis=1) / counts
    samples["economic"] = blocks["economic_sum"].to_numpy()[draws].sum(axis=1)
    output: dict[str, float] = {}
    for metric, values in samples.items():
        output[f"{metric}_ci_lower"] = float(np.quantile(values, 0.025))
        output[f"{metric}_ci_upper"] = float(np.quantile(values, 0.975))
    brier = samples["brier"]
    if np.isclose(brier, 0.0).all():
        output["brier_bootstrap_p_value"] = 1.0
    else:
        negative_tail = float(np.mean(brier <= 0.0))
        positive_tail = float(np.mean(brier >= 0.0))
        output["brier_bootstrap_p_value"] = min(1.0, 2.0 * min(negative_tail, positive_tail))
    return output


def _holm_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    available = values.dropna().sort_values(kind="stable")
    running = 0.0
    count = len(available)
    adjusted_values: dict[Hashable, float] = {}
    for rank, (index, value) in enumerate(available.items()):
        adjusted = min(1.0, (count - rank) * float(str(value)))
        running = max(running, adjusted)
        adjusted_values[index] = running
    return pd.Series(adjusted_values, dtype=float).reindex(result.index)


def paired_lead_metrics(
    paired: pd.DataFrame,
    *,
    bootstrap_resamples: int = 2_000,
    seed: int = 20_260_715,
) -> pd.DataFrame:
    """Compute full-minus-control paired endpoints for every registered lead."""

    _require(
        paired,
        {
            "target_lead_sessions",
            "target_payoff_available",
            "target_payoff_positive",
            "target_robust_net_bps",
            "p_next_payoff_positive__full",
            "p_next_payoff_positive__control",
            "posterior_mean_net_bps__full",
            "posterior_mean_net_bps__control",
            "edge_state__full",
            "edge_state__control",
            "period",
            "score_session",
        },
        "paired metric",
    )
    records: list[dict[str, object]] = []
    for lead, group in paired.groupby("target_lead_sessions", observed=True, sort=True):
        components = _paired_components(group)
        record: dict[str, object] = {
            "target_lead_sessions": int(str(lead)),
            "paired_forecasts": len(group),
            "paired_observable_targets": len(components),
        }
        if not components.empty:
            record.update(
                {
                    "paired_brier_improvement": float(components["brier"].mean()),
                    "paired_log_loss_improvement": float(components["log_loss"].mean()),
                    "paired_absolute_error_improvement_bps": float(
                        components["absolute_error"].mean()
                    ),
                    "paired_economic_increment_bps": float(components["economic"].sum()),
                }
            )
            record.update(
                _session_block_bootstrap(
                    components,
                    resamples=bootstrap_resamples,
                    seed=seed + int(str(lead)),
                )
            )
        records.append(record)
    result = pd.DataFrame.from_records(records)
    if "brier_bootstrap_p_value" in result:
        result["brier_holm_adjusted_p_value"] = _holm_adjust(result["brier_bootstrap_p_value"])
    result["primary_registered_endpoint"] = result["target_lead_sessions"].eq(1)
    return result


def build_feature_contribution_bins(paired: pd.DataFrame, *, bins: int = 5) -> pd.DataFrame:
    """Assign target-blind equal-frequency contribution bins within each lead."""

    _require(
        paired,
        {"target_lead_sessions", "feature_contribution_p_next"},
        "feature contribution",
    )
    result = paired.copy()
    result["contribution_bin"] = pd.Series(pd.NA, index=result.index, dtype="string")
    for _, index in result.groupby("target_lead_sessions", observed=True, sort=True).groups.items():
        values = result.loc[index, "feature_contribution_p_next"]
        try:
            categories = pd.qcut(values, q=bins, duplicates="drop")
            codes = categories.cat.codes
            result.loc[index, "contribution_bin"] = [
                f"bin_{code + 1}" if code >= 0 else pd.NA for code in codes
            ]
        except ValueError:
            result.loc[index, "contribution_bin"] = "bin_1"
    return result


def summarize_feature_contributions(binned: pd.DataFrame) -> pd.DataFrame:
    """Summarize target response without changing target-blind bin membership."""

    observed = binned.loc[binned["target_payoff_available"].eq(True)].copy()  # noqa: E712
    grouped = observed.groupby(
        ["target_lead_sessions", "contribution_bin"], observed=True, sort=True
    )
    summary = grouped.agg(
        forecasts=("feature_contribution_p_next", "size"),
        mean_feature_contribution=("feature_contribution_p_next", "mean"),
        mean_future_payoff_bps=("target_robust_net_bps", "mean"),
        positive_payoff_rate=("target_payoff_positive", "mean"),
        independent_stock_support=("target_independent_stocks", "sum"),
    ).reset_index()
    diagnostics: list[dict[str, object]] = []
    for lead, group in observed.groupby("target_lead_sessions", observed=True, sort=True):
        payoff_correlation = spearmanr(
            group["feature_contribution_p_next"], group["target_robust_net_bps"], nan_policy="omit"
        )
        event_correlation = spearmanr(
            group["feature_contribution_p_next"],
            group["target_payoff_positive"].astype(float),
            nan_policy="omit",
        )
        diagnostics.append(
            {
                "target_lead_sessions": int(str(lead)),
                "contribution_bin": "continuous_rank_diagnostic",
                "forecasts": len(group),
                "spearman_future_payoff": float(payoff_correlation.statistic),
                "spearman_future_payoff_p_value": float(payoff_correlation.pvalue),
                "spearman_positive_event": float(event_correlation.statistic),
                "spearman_positive_event_p_value": float(event_correlation.pvalue),
            }
        )
    return pd.concat(
        [summary, pd.DataFrame.from_records(diagnostics)], ignore_index=True, sort=False
    )


__all__ = [
    "build_feature_contribution_bins",
    "build_paired_prediction_table",
    "expected_calibration_error",
    "lead_calibration_metrics",
    "paired_lead_metrics",
    "summarize_feature_contributions",
    "validate_paired_training_identity",
]
