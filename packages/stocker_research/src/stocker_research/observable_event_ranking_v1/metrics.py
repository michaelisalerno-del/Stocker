"""Slate-level ranking and raw structural scale metrics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", ascending=True).to_numpy(dtype="float64")


def spearman_ic(outcomes: np.ndarray, scores: np.ndarray) -> float:
    """Return rank correlation with deterministic average ties."""

    outcome_ranks = _average_ranks(np.asarray(outcomes, dtype="float64"))
    score_ranks = _average_ranks(np.asarray(scores, dtype="float64"))
    if np.std(outcome_ranks) == 0.0 or np.std(score_ranks) == 0.0:
        return float("nan")
    return float(np.corrcoef(outcome_ranks, score_ranks)[0, 1])


def ndcg_score(relevance: np.ndarray, scores: np.ndarray) -> float:
    """Return NDCG using non-negative relevance and descending predicted score."""

    rel = np.asarray(relevance, dtype="float64")
    rel = rel - min(0.0, float(np.min(rel)))
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))

    def dcg(order: np.ndarray) -> float:
        return float(np.sum((np.power(2.0, rel[order]) - 1.0) * discounts))

    predicted = np.argsort(-np.asarray(scores, dtype="float64"), kind="mergesort")
    ideal = np.argsort(-rel, kind="mergesort")
    ideal_value = dcg(ideal)
    return float("nan") if ideal_value == 0.0 else dcg(predicted) / ideal_value


def pairwise_ranking_accuracy(outcomes: np.ndarray, scores: np.ndarray) -> float:
    """Return concordant fraction over outcome-unequal pairs."""

    correct = 0.0
    count = 0
    for left, right in combinations(range(len(outcomes)), 2):
        outcome_difference = float(outcomes[left] - outcomes[right])
        if outcome_difference == 0.0:
            continue
        score_difference = float(scores[left] - scores[right])
        count += 1
        if score_difference == 0.0:
            correct += 0.5
        elif np.sign(outcome_difference) == np.sign(score_difference):
            correct += 1.0
    return float("nan") if count == 0 else correct / count


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype="float64"), kind="mergesort")[:count]


def top_two_minus_median(outcomes: np.ndarray, scores: np.ndarray) -> float:
    """Return top-two average raw outcome minus simultaneous slate median."""

    values = np.asarray(outcomes, dtype="float64")
    top = values[_top_indices(scores, min(2, len(values)))]
    return float(np.mean(top) - np.median(values))


def top_decile_hit(outcomes: np.ndarray, scores: np.ndarray) -> float | None:
    """Return whether top score is in actual top decile; unavailable below ten members."""

    if len(outcomes) < 10:
        return None
    actual_count = max(1, int(np.ceil(len(outcomes) * 0.10)))
    actual = set(_top_indices(outcomes, actual_count))
    return float(int(int(_top_indices(scores, 1)[0]) in actual))


def per_slate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute equal-weight-ready candidate and baseline metrics per evaluable slate."""

    rows: list[dict[str, object]] = []
    evaluable = predictions.loc[predictions["slate_evaluable"].astype(bool)].copy()
    evaluable = evaluable.loc[evaluable["future_return_60m"].notna()]
    for slate_id, group in evaluable.groupby("slate_id", sort=True):
        outcomes = group["future_return_60m"].to_numpy(dtype="float64")
        target = group["target_rank_60m"].to_numpy(dtype="float64")
        candidate = group["candidate_score"].to_numpy(dtype="float64")
        baseline = group["baseline_score"].to_numpy(dtype="float64")
        candidate_ic = spearman_ic(target, candidate)
        baseline_ic = spearman_ic(target, baseline)
        candidate_scale = top_two_minus_median(outcomes, candidate)
        baseline_scale = top_two_minus_median(outcomes, baseline)
        candidate_top = outcomes[_top_indices(candidate, min(2, len(group)))]
        baseline_top = outcomes[_top_indices(baseline, min(2, len(group)))]
        ordered_scores = np.sort(candidate)[::-1]
        rows.append(
            {
                "slate_id": slate_id,
                "session": group["session"].iloc[0],
                "candidate_ic": candidate_ic,
                "baseline_ic": baseline_ic,
                "candidate_minus_baseline_ic": candidate_ic - baseline_ic,
                "candidate_ndcg": ndcg_score(target, candidate),
                "baseline_ndcg": ndcg_score(target, baseline),
                "candidate_pairwise_accuracy": pairwise_ranking_accuracy(target, candidate),
                "baseline_pairwise_accuracy": pairwise_ranking_accuracy(target, baseline),
                "candidate_top_one_outcome": float(candidate_top[0]),
                "candidate_top_two_average_outcome": float(np.mean(candidate_top)),
                "baseline_top_one_outcome": float(baseline_top[0]),
                "baseline_top_two_average_outcome": float(np.mean(baseline_top)),
                "candidate_top_one_hit": float(candidate_top[0] > 0.0),
                "candidate_top_two_hit_rate": float(np.mean(candidate_top > 0.0)),
                "candidate_top_decile_hit": top_decile_hit(outcomes, candidate),
                "candidate_top_two_minus_median": candidate_scale,
                "baseline_top_two_minus_median": baseline_scale,
                "candidate_minus_baseline_top_two": candidate_scale - baseline_scale,
                "candidate_score_separation_rank_one_two": float(
                    ordered_scores[0] - ordered_scores[1]
                    if len(ordered_scores) >= 2
                    else float("nan")
                ),
                "outcome_dispersion": float(np.std(outcomes, ddof=0)),
                "valid_target_count": len(group),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "slate_id",
                "session",
                "candidate_ic",
                "baseline_ic",
                "candidate_minus_baseline_ic",
                "candidate_ndcg",
                "baseline_ndcg",
                "candidate_pairwise_accuracy",
                "baseline_pairwise_accuracy",
                "candidate_top_one_outcome",
                "candidate_top_two_average_outcome",
                "baseline_top_one_outcome",
                "baseline_top_two_average_outcome",
                "candidate_top_one_hit",
                "candidate_top_two_hit_rate",
                "candidate_top_decile_hit",
                "candidate_top_two_minus_median",
                "baseline_top_two_minus_median",
                "candidate_minus_baseline_top_two",
                "candidate_score_separation_rank_one_two",
                "outcome_dispersion",
                "valid_target_count",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["session", "slate_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def leave_one_stock_out_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Remove each stock everywhere, recompute slate ranks, and re-evaluate paired metrics."""

    output: list[dict[str, object]] = []
    for symbol in sorted(predictions["symbol"].astype(str).unique()):
        reduced = predictions.loc[~predictions["symbol"].eq(symbol)].copy()
        reduced["slate_evaluable"] = (
            reduced.groupby("slate_id", sort=True)["symbol"].transform("size").ge(8)
        )
        reduced["target_rank_60m"] = np.nan
        for _, group in reduced.loc[reduced["slate_evaluable"]].groupby("slate_id", sort=True):
            count = len(group)
            ranks = group["future_return_60m"].rank(method="average", ascending=True)
            reduced.loc[group.index, "target_rank_60m"] = (
                (ranks - 1.0) / (count - 1.0) if count > 1 else 0.5
            )
        metrics = per_slate_metrics(reduced)
        output.append(
            {
                "removed_symbol": symbol,
                "remaining_rows": len(reduced),
                "evaluable_slates": len(metrics),
                "candidate_mean_ic": float(metrics["candidate_ic"].mean())
                if not metrics.empty
                else np.nan,
                "baseline_mean_ic": float(metrics["baseline_ic"].mean())
                if not metrics.empty
                else np.nan,
                "candidate_minus_baseline_ic": float(metrics["candidate_minus_baseline_ic"].mean())
                if not metrics.empty
                else np.nan,
                "candidate_minus_baseline_top_two": float(
                    metrics["candidate_minus_baseline_top_two"].mean()
                )
                if not metrics.empty
                else np.nan,
            }
        )
    return (
        pd.DataFrame(output).sort_values("removed_symbol", kind="mergesort").reset_index(drop=True)
    )
