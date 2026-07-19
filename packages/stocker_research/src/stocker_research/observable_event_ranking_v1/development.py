"""Frozen historical out-of-fold baseline and M1 development workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stocker_research.observable_event_ranking_v1.baselines import (
    TrainingOnlyStockClockPrior,
    deterministic_baseline_scores,
)
from stocker_research.observable_event_ranking_v1.contract import BASELINES, PRIMARY_FEATURES
from stocker_research.observable_event_ranking_v1.folds import (
    ChronologicalFold,
    build_expanding_folds,
)
from stocker_research.observable_event_ranking_v1.linear_ranker import fit_linear_ranker
from stocker_research.observable_event_ranking_v1.metrics import (
    per_slate_metrics,
    spearman_ic,
)
from stocker_research.observable_event_ranking_v1.uncertainty import (
    BootstrapResult,
    paired_session_block_bootstrap,
)

NEW_YORK = ZoneInfo("America/New_York")
_DETERMINISTIC_BASELINES = tuple(
    baseline for baseline in BASELINES if baseline != "B0_RANDOM_ELIGIBLE"
)
_SIMPLE_BASELINE_FEATURES = {
    "B1_EVENT_STRENGTH": "event_strength",
    "B2_PREVIOUS_5M_MARKET_RELATIVE_RETURN": "market_relative_return_5m",
    "B3_15M_MARKET_RELATIVE_STRENGTH": "market_relative_return_15m",
    "B4_30M_MARKET_RELATIVE_STRENGTH": "market_relative_return_30m",
    "B5_15M_SECTOR_RELATIVE_STRENGTH": "sector_relative_return_15m",
    "B6_ACTIVITY_SHOCK": "activity_shock_z",
    "B7_REALIZED_VOLATILITY": "realized_volatility_30m",
}


@dataclass(frozen=True)
class DevelopmentOOFResult:
    """All historical development outputs prior to the scientific decision."""

    folds: tuple[ChronologicalFold, ...]
    baseline_predictions: pd.DataFrame
    baseline_metrics: pd.DataFrame
    strongest_baseline: str
    candidate_predictions: pd.DataFrame
    slate_metrics: pd.DataFrame
    model_effective_configuration: dict[str, Any]
    model_parameters: dict[str, Any]
    ic_bootstrap: BootstrapResult
    top_two_bootstrap: BootstrapResult
    leave_one_stock_out: pd.DataFrame
    concentration_results: pd.DataFrame
    turnover_results: pd.DataFrame
    month_metrics: pd.DataFrame
    quarter_metrics: pd.DataFrame


def _decision_clock(frame: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(frame["assigned_decision_time"], utc=True)
    return timestamps.dt.tz_convert(NEW_YORK).dt.strftime("%H:%M")


def _random_repeated_scores(
    evaluation: pd.DataFrame,
    *,
    seed: int,
    repeats: int = 100,
) -> pd.Series:
    scores = pd.Series(0.0, index=evaluation.index, dtype="float64")
    generator = np.random.default_rng(seed)
    for _, group in evaluation.groupby("slate_id", sort=True):
        accumulated = np.zeros(len(group), dtype="float64")
        for _ in range(repeats):
            permutation = generator.permutation(len(group))
            accumulated += permutation / max(1, len(group) - 1)
        scores.loc[group.index] = accumulated / repeats
    return scores


def _baseline_oof(frame: pd.DataFrame, folds: list[ChronologicalFold]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    prepared = frame.copy()
    prepared["decision_clock"] = _decision_clock(prepared)
    for fold_number, fold in enumerate(folds, start=1):
        training = prepared.loc[list(fold.train_indices)].copy()
        evaluation = prepared.loc[list(fold.evaluation_indices)].copy()
        output = evaluation.loc[
            :,
            ["event_id", "slate_id", "symbol", "session", "decision_clock"],
        ].copy()
        output["fold_id"] = fold.fold_id
        output["B0_RANDOM_ELIGIBLE"] = _random_repeated_scores(
            evaluation, seed=20260719 + fold_number
        )
        for baseline in _DETERMINISTIC_BASELINES[:7]:
            output[baseline] = deterministic_baseline_scores(evaluation, baseline).to_numpy()
        frequency = TrainingOnlyStockClockPrior.fit(training, kind="event_frequency")
        mean_target = TrainingOnlyStockClockPrior.fit(training, kind="mean_target")
        output["B8_TRAINING_ONLY_STOCK_CLOCK_EVENT_FREQUENCY"] = [
            frequency.score(str(row.symbol), str(row.decision_clock)) for row in output.itertuples()
        ]
        output["B9_TRAINING_ONLY_STOCK_CLOCK_MEAN_TARGET_PRIOR"] = [
            mean_target.score(str(row.symbol), str(row.decision_clock))
            for row in output.itertuples()
        ]
        pieces.append(output)
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["session", "slate_id", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def _baseline_metrics_and_selection(
    frame: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    targets = frame.loc[
        :,
        ["event_id", "slate_id", "target_rank_60m", "future_return_60m"],
    ]
    joined = predictions.merge(targets, on=["event_id", "slate_id"], validate="one_to_one")
    rows: list[dict[str, object]] = []
    for baseline in _DETERMINISTIC_BASELINES:
        slate_ics = [
            spearman_ic(
                group["target_rank_60m"].to_numpy(dtype="float64"),
                group[baseline].to_numpy(dtype="float64"),
            )
            for _, group in joined.groupby("slate_id", sort=True)
        ]
        finite = [value for value in slate_ics if np.isfinite(value)]
        rows.append(
            {
                "baseline_id": baseline,
                "mean_spearman_ic": float(np.mean(finite)) if finite else np.nan,
                "median_spearman_ic": float(np.median(finite)) if finite else np.nan,
                "evaluable_slates": len(finite),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("baseline_id", kind="mergesort").reset_index(drop=True)
    eligible = metrics.dropna(subset=["mean_spearman_ic"]).sort_values(
        ["mean_spearman_ic", "baseline_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    if eligible.empty:
        raise ValueError("no deterministic baseline has evaluable out-of-fold IC")
    return metrics, str(eligible.iloc[0]["baseline_id"])


def _serialize_model(model: Any) -> dict[str, Any]:
    return {
        "alpha": model.alpha,
        "feature_names": list(model.feature_names),
        "preprocessor": asdict(model.preprocessor),
        "intercept": model.intercept,
        "coefficients": list(model.coefficients),
    }


def fit_final_frozen_components(
    target_feature_ledger: pd.DataFrame,
    strongest_baseline: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the one final historical model and selected baseline for prospective freezing."""

    frame = target_feature_ledger.copy()
    frame["session"] = pd.to_datetime(frame["session"], utc=True)
    frame["assigned_decision_time"] = pd.to_datetime(frame["assigned_decision_time"], utc=True)
    frame = frame.loc[
        frame["slate_evaluable"].astype(bool)
        & frame["target_rank_60m"].notna()
        & frame["future_return_60m"].notna()
    ].reset_index(drop=True)
    if frame.empty:
        raise ValueError("final freeze fit requires evaluable development rows")
    frame["decision_clock"] = _decision_clock(frame)
    model = fit_linear_ranker(
        frame.loc[:, list(PRIMARY_FEATURES)],
        frame["target_rank_60m"].to_numpy(dtype="float64"),
        frame["slate_id"],
    )
    model_parameters = {
        "model_id": "M1_POOLED_LINEAR_RANKER",
        "fit_scope": "all_evaluable_historical_development_rows_after_oof_selection",
        "training_rows": len(frame),
        "training_slates": frame["slate_id"].nunique(),
        "training_first_session": frame["session"].min().isoformat(),
        "training_last_session": frame["session"].max().isoformat(),
        **_serialize_model(model),
    }
    if strongest_baseline in _SIMPLE_BASELINE_FEATURES:
        baseline_parameters: dict[str, Any] = {
            "baseline_id": strongest_baseline,
            "kind": "direct_observable_feature",
            "source_feature": _SIMPLE_BASELINE_FEATURES[strongest_baseline],
            "fit_uses_outcomes": False,
        }
    elif strongest_baseline in {
        "B8_TRAINING_ONLY_STOCK_CLOCK_EVENT_FREQUENCY",
        "B9_TRAINING_ONLY_STOCK_CLOCK_MEAN_TARGET_PRIOR",
    }:
        kind: Literal["event_frequency", "mean_target"] = (
            "event_frequency"
            if strongest_baseline == "B8_TRAINING_ONLY_STOCK_CLOCK_EVENT_FREQUENCY"
            else "mean_target"
        )
        prior = TrainingOnlyStockClockPrior.fit(frame, kind=kind)
        baseline_parameters = {
            "baseline_id": strongest_baseline,
            "kind": kind,
            "global_prior": prior.global_prior,
            "shrinkage": prior.shrinkage,
            "unseen_cell_fallback": "global_prior",
            "cell_scores": [
                {
                    "symbol": symbol,
                    "decision_clock": clock,
                    "score": score,
                }
                for (symbol, clock), score in sorted(prior.cell_scores.items())
            ],
        }
    else:
        raise ValueError(f"baseline cannot be frozen: {strongest_baseline}")
    return model_parameters, baseline_parameters


def _candidate_oof(
    frame: pd.DataFrame, folds: list[ChronologicalFold]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pieces: list[pd.DataFrame] = []
    parameters: dict[str, Any] = {}
    for fold in folds:
        training = frame.loc[list(fold.train_indices)].copy()
        evaluation = frame.loc[list(fold.evaluation_indices)].copy()
        model = fit_linear_ranker(
            training.loc[:, list(PRIMARY_FEATURES)],
            training["target_rank_60m"].to_numpy(dtype="float64"),
            training["slate_id"],
        )
        output = evaluation.loc[:, ["event_id", "slate_id", "symbol", "session"]].copy()
        output["fold_id"] = fold.fold_id
        output["candidate_score"] = model.predict(evaluation.loc[:, list(PRIMARY_FEATURES)])
        pieces.append(output)
        parameters[fold.fold_id] = _serialize_model(model)
    predictions = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["session", "slate_id", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    return predictions, parameters


def _paired_predictions(
    frame: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    strongest_baseline: str,
) -> pd.DataFrame:
    metadata_columns = [
        "event_id",
        "slate_id",
        "symbol",
        "sector",
        "session",
        "assigned_decision_time",
        "future_return_60m",
        "target_rank_60m",
        "slate_evaluable",
        "decision_clock",
    ]
    paired = frame.loc[:, metadata_columns].merge(
        candidate_predictions[["event_id", "candidate_score"]],
        on="event_id",
        validate="one_to_one",
    )
    return paired.merge(
        baseline_predictions[["event_id", strongest_baseline]].rename(
            columns={strongest_baseline: "baseline_score"}
        ),
        on="event_id",
        validate="one_to_one",
    )


def _rerank_remaining_outcomes(frame: pd.DataFrame) -> pd.Series:
    ranks = frame.groupby("slate_id", sort=True)["future_return_60m"].rank(
        method="average",
        ascending=True,
    )
    sizes = frame.groupby("slate_id", sort=True)["future_return_60m"].transform("count")
    return (ranks - 1.0) / (sizes - 1.0).clip(lower=1.0)


def _leave_one_stock_out_retrained(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove each stock, then refit preprocessing, baselines, and M1 from scratch."""

    rows: list[dict[str, object]] = []
    for removed_symbol in sorted(frame["symbol"].astype(str).unique()):
        reduced = frame.loc[frame["symbol"].astype(str).ne(removed_symbol)].copy()
        reduced = reduced.reset_index(drop=True)
        reduced["target_rank_60m"] = _rerank_remaining_outcomes(reduced)
        folds = build_expanding_folds(reduced)
        if not folds:
            rows.append(
                {
                    "removed_symbol": removed_symbol,
                    "remaining_rows": len(reduced),
                    "remaining_slates": reduced["slate_id"].nunique(),
                    "strongest_baseline": "unavailable",
                    "candidate_mean_ic": np.nan,
                    "baseline_mean_ic": np.nan,
                    "candidate_minus_baseline_ic": np.nan,
                    "candidate_minus_baseline_top_two": np.nan,
                    "full_pipeline_refitted": False,
                }
            )
            continue
        baseline_predictions = _baseline_oof(reduced, folds)
        _, strongest = _baseline_metrics_and_selection(reduced, baseline_predictions)
        candidate_predictions, _ = _candidate_oof(reduced, folds)
        paired = _paired_predictions(
            reduced,
            candidate_predictions,
            baseline_predictions,
            strongest,
        )
        metrics = per_slate_metrics(paired)
        rows.append(
            {
                "removed_symbol": removed_symbol,
                "remaining_rows": len(reduced),
                "remaining_slates": metrics["slate_id"].nunique(),
                "strongest_baseline": strongest,
                "candidate_mean_ic": metrics["candidate_ic"].mean(),
                "baseline_mean_ic": metrics["baseline_ic"].mean(),
                "candidate_minus_baseline_ic": metrics["candidate_minus_baseline_ic"].mean(),
                "candidate_minus_baseline_top_two": metrics[
                    "candidate_minus_baseline_top_two"
                ].mean(),
                "full_pipeline_refitted": True,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "removed_symbol",
            "remaining_rows",
            "remaining_slates",
            "strongest_baseline",
            "candidate_mean_ic",
            "baseline_mean_ic",
            "candidate_minus_baseline_ic",
            "candidate_minus_baseline_top_two",
            "full_pipeline_refitted",
        ),
    )


def _concentration(predictions: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for _, group in predictions.groupby("slate_id", sort=True):
        selected.append(
            group.sort_values("candidate_score", ascending=False, kind="mergesort").head(2)
        )
    chosen = pd.concat(selected, ignore_index=True) if selected else predictions.iloc[0:0]
    rows: list[dict[str, object]] = []
    for dimension in ("symbol", "sector", "decision_clock"):
        for value, count in chosen[dimension].value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "selection_count": int(count),
                    "selection_fraction": float(count / len(chosen)) if len(chosen) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _turnover(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session, day in predictions.groupby("session", sort=True):
        previous: set[str] | None = None
        daily_selected: set[str] = set()
        daily_rows: list[dict[str, object]] = []
        for _, slate in day.sort_values("assigned_decision_time").groupby("slate_id", sort=False):
            selected = set(
                slate.sort_values("candidate_score", ascending=False, kind="mergesort")
                .head(2)["symbol"]
                .astype(str)
            )
            daily_selected.update(selected)
            daily_rows.append(
                {
                    "session": session,
                    "slate_id": slate["slate_id"].iloc[0],
                    "turnover": (
                        np.nan
                        if previous is None
                        else 1.0 - len(previous & selected) / max(1, len(previous))
                    ),
                    "unique_selections_that_day": 0,
                }
            )
            previous = selected
        for row in daily_rows:
            row["unique_selections_that_day"] = len(daily_selected)
        rows.extend(daily_rows)
    return pd.DataFrame(
        rows,
        columns=("session", "slate_id", "turnover", "unique_selections_that_day"),
    )


def _period_metrics(slate_metrics: pd.DataFrame, frequency: str, column: str) -> pd.DataFrame:
    frame = slate_metrics.copy()
    sessions = pd.to_datetime(frame["session"], utc=True).dt.tz_localize(None)
    frame[column] = sessions.dt.to_period(frequency).astype(str)
    numeric = [
        "candidate_ic",
        "baseline_ic",
        "candidate_minus_baseline_ic",
        "candidate_top_two_minus_median",
        "baseline_top_two_minus_median",
        "candidate_minus_baseline_top_two",
    ]
    grouped = frame.groupby(column, sort=True)[numeric].mean().reset_index()
    grouped["slate_count"] = frame.groupby(column, sort=True).size().to_numpy()
    return grouped


def run_development_oof(
    target_feature_ledger: pd.DataFrame,
    *,
    bootstrap_draws: int = 2_000,
) -> DevelopmentOOFResult:
    """Run exactly the frozen baselines and M1 across expanding chronological folds."""

    frame = target_feature_ledger.copy()
    frame["session"] = pd.to_datetime(frame["session"], utc=True)
    frame["assigned_decision_time"] = pd.to_datetime(frame["assigned_decision_time"], utc=True)
    frame = frame.loc[
        frame["slate_evaluable"].astype(bool)
        & frame["target_rank_60m"].notna()
        & frame["future_return_60m"].notna()
    ].reset_index(drop=True)
    frame["decision_clock"] = _decision_clock(frame)
    folds = build_expanding_folds(frame)
    if not folds:
        raise ValueError("development requires at least one complete chronological fold")
    baseline_predictions = _baseline_oof(frame, folds)
    baseline_metrics, strongest = _baseline_metrics_and_selection(frame, baseline_predictions)
    candidate_predictions, model_parameters = _candidate_oof(frame, folds)
    paired = _paired_predictions(
        frame,
        candidate_predictions,
        baseline_predictions,
        strongest,
    )
    slate_metrics = per_slate_metrics(paired)
    ic_bootstrap = paired_session_block_bootstrap(
        slate_metrics,
        candidate_column="candidate_ic",
        baseline_column="baseline_ic",
        draws=bootstrap_draws,
        seed=20260719,
    )
    top_two_bootstrap = paired_session_block_bootstrap(
        slate_metrics,
        candidate_column="candidate_top_two_minus_median",
        baseline_column="baseline_top_two_minus_median",
        draws=bootstrap_draws,
        seed=20260719,
    )
    return DevelopmentOOFResult(
        folds=tuple(folds),
        baseline_predictions=baseline_predictions,
        baseline_metrics=baseline_metrics,
        strongest_baseline=strongest,
        candidate_predictions=candidate_predictions,
        slate_metrics=slate_metrics,
        model_effective_configuration={
            "model_id": "M1_POOLED_LINEAR_RANKER",
            "alpha": 1.0,
            "features": list(PRIMARY_FEATURES),
            "stock_identifier_input": False,
            "sector_identifier_input": False,
            "hyperparameter_search": False,
            "feature_selection": False,
            "interaction_search": False,
            "sample_weight": "1_over_slate_size",
        },
        model_parameters=model_parameters,
        ic_bootstrap=ic_bootstrap,
        top_two_bootstrap=top_two_bootstrap,
        leave_one_stock_out=_leave_one_stock_out_retrained(frame),
        concentration_results=_concentration(paired),
        turnover_results=_turnover(paired),
        month_metrics=_period_metrics(slate_metrics, "M", "month"),
        quarter_metrics=_period_metrics(slate_metrics, "Q", "quarter"),
    )
