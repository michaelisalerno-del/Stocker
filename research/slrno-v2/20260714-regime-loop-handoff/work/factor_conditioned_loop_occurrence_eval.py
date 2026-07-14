"""Pure evaluation utilities for factor-conditioned loop occurrence V1.

This module performs no file, source-data, shadow, broker, or runtime access.
It evaluates already-constructed research predictions under frozen contract
SHA-256 ef8b61bdd4f6671fa64713551a9991f6e4591c3c96bc1ccc324c81b7195bfe7d.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


CONTRACT_SHA256 = (
    "ef8b61bdd4f6671fa64713551a9991f6e4591c3c96bc1ccc324c81b7195bfe7d"
)
RESEARCH_ONLY = True
LIVE_ORDERING_ENABLED = False
ORDER_PLACEMENT = "disabled"

EPSILON = 1e-12
PRIMARY_CANDIDATE = "qfull9"
PRIMARY_BASELINES = ("qhistory", "qpattern", "qlimited4")
LINEAGE_BASELINE = "qold_limited_path"
MODEL_COLUMNS = (*PRIMARY_BASELINES, LINEAGE_BASELINE, PRIMARY_CANDIDATE)
LOSS_NAMES = ("log_loss", "brier")
BOOTSTRAP_DRAWS = 999
BOOTSTRAP_BLOCK_LENGTH = 5
BOOTSTRAP_UPPER_QUANTILE = 0.9916666666666667
PERIOD_SEED_OFFSETS = {"2024": 0, "2025": 10_000, "2023": 20_000}
FALSIFICATION_SEED = 20260711
IRREGULAR_DATE = "2025-04-10"
IRREGULAR_SYMBOLS = ("CIFR", "IREN", "NVTS", "RIVN", "WULF")
NEW_FACTOR_COLUMNS = (
    "current_bar_log_return",
    "return_sum_6",
    "mean_abs_return_12",
    "session_return",
    "bar_range_pct",
)
FACTOR_QUARTILE_COLUMNS = tuple(
    f"factor_quartile__{column}" for column in NEW_FACTOR_COLUMNS
)
REPORT_SLICE_COLUMNS = (
    "b0_entry_high_stress",
    "b0_unknown",
    "entry_clock_quartile",
)

PRIMARY_REQUIREMENTS = {
    "qhistory": {"relative_log_loss": 0.01, "recall_gain": 0.005},
    "qpattern": {"relative_log_loss": 0.005, "recall_gain": 0.002},
    "qlimited4": {"relative_log_loss": 0.0025, "recall_gain": 0.002},
}


def _one_dimensional(values: Any, name: str, dtype: Any = float) -> np.ndarray:
    output = np.asarray(values, dtype=dtype)
    if output.ndim != 1:
        raise ValueError(f"{name} must be one dimensional")
    return output


def _validate_target_probability(
    target: Any, probability: Any
) -> tuple[np.ndarray, np.ndarray]:
    observed = _one_dimensional(target, "target", float)
    forecast = _one_dimensional(probability, "probability", float)
    if len(observed) != len(forecast) or not len(observed):
        raise ValueError("target and probability must have equal nonzero length")
    if not np.isfinite(observed).all() or not np.isfinite(forecast).all():
        raise ValueError("target and probability must be finite")
    if not np.isin(observed, (0.0, 1.0)).all():
        raise ValueError("target must be binary")
    if (forecast < 0.0).any() or (forecast > 1.0).any():
        raise ValueError("probability must be in [0, 1]")
    return observed, forecast


def _validated_weights(weights: Any, length: int) -> np.ndarray:
    output = _one_dimensional(weights, "weights", float)
    if len(output) != length or not np.isfinite(output).all():
        raise ValueError("weights must be finite and match the row count")
    if (output < 0.0).any() or float(output.sum()) <= 0.0:
        raise ValueError("weights must be nonnegative with positive total")
    return output


def inverse_compatible_weights(compatible_counts: Any) -> np.ndarray:
    counts = _one_dimensional(compatible_counts, "compatible_counts", float)
    if not len(counts) or not np.isfinite(counts).all():
        raise ValueError("compatible counts must be finite and nonempty")
    if (counts <= 0.0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("compatible counts must be positive integers")
    return 1.0 / counts


def binary_loss_arrays(target: Any, probability: Any) -> dict[str, np.ndarray]:
    observed, forecast = _validate_target_probability(target, probability)
    clipped = np.clip(forecast, EPSILON, 1.0 - EPSILON)
    return {
        "log_loss": -(
            observed * np.log(clipped)
            + (1.0 - observed) * np.log(1.0 - clipped)
        ),
        "brier": np.square(forecast - observed),
    }


def weighted_mean(values: Any, weights: Any) -> float:
    array = _one_dimensional(values, "values", float)
    weight = _validated_weights(weights, len(array))
    if not np.isfinite(array).all():
        raise ValueError("weighted values must be finite")
    return float(np.dot(array, weight) / weight.sum())


def loss_metrics(
    target: Any,
    probability: Any,
    *,
    compatible_counts: Any | None = None,
) -> dict[str, float | int]:
    observed, forecast = _validate_target_probability(target, probability)
    weights = (
        np.ones(len(observed), dtype=float)
        if compatible_counts is None
        else inverse_compatible_weights(compatible_counts)
    )
    losses = binary_loss_arrays(observed, forecast)
    return {
        "rows": int(len(observed)),
        "positives": int(observed.sum()),
        "weight_sum": float(weights.sum()),
        "log_loss": weighted_mean(losses["log_loss"], weights),
        "brier": weighted_mean(losses["brier"], weights),
    }


def loss_comparison(
    target: Any,
    candidate_probability: Any,
    baseline_probability: Any,
    *,
    compatible_counts: Any | None = None,
) -> dict[str, float]:
    observed, candidate = _validate_target_probability(
        target, candidate_probability
    )
    _, baseline = _validate_target_probability(target, baseline_probability)
    weights = (
        np.ones(len(observed), dtype=float)
        if compatible_counts is None
        else inverse_compatible_weights(compatible_counts)
    )
    candidate_loss = binary_loss_arrays(observed, candidate)
    baseline_loss = binary_loss_arrays(observed, baseline)
    output: dict[str, float] = {}
    for loss_name in LOSS_NAMES:
        candidate_mean = weighted_mean(candidate_loss[loss_name], weights)
        baseline_mean = weighted_mean(baseline_loss[loss_name], weights)
        difference = candidate_mean - baseline_mean
        output[f"candidate_{loss_name}"] = candidate_mean
        output[f"baseline_{loss_name}"] = baseline_mean
        output[f"{loss_name}_difference"] = difference
        if loss_name == "log_loss":
            if baseline_mean <= 0.0:
                raise ValueError("baseline log loss must be positive")
            output["relative_log_loss_improvement"] = (
                -difference / baseline_mean
            )
    return output


def top_three_metrics(
    frame: pd.DataFrame,
    probability_column: str,
    *,
    target_column: str = "target",
    anchor_column: str = "anchor_id",
    cycle_column: str = "cycle_id",
) -> dict[str, float | int]:
    required = {probability_column, target_column, anchor_column, cycle_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing ranking columns: {missing}")
    selected = frame.loc[:, list(required)].copy()
    if selected.empty or selected.duplicated([anchor_column, cycle_column]).any():
        raise ValueError("ranking rows must be nonempty unique anchor-cycle pairs")
    observed, probability = _validate_target_probability(
        selected[target_column], selected[probability_column]
    )
    selected[target_column] = observed.astype(np.int8)
    selected[probability_column] = probability
    ranked = selected.sort_values(
        [anchor_column, probability_column, cycle_column],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked["rank"] = ranked.groupby(anchor_column, sort=False).cumcount() + 1
    top_three = ranked["rank"].le(3)
    top_one = ranked["rank"].eq(1)
    positive = ranked[target_column].eq(1)
    positive_labels = int(positive.sum())
    selected_labels = int(top_three.sum())
    hits = int((top_three & positive).sum())
    top_one_hits = int((top_one & positive).sum())
    per_anchor = ranked.assign(
        top_three_hit=(top_three & positive).astype(np.int8),
        reciprocal=np.where(positive, 1.0 / ranked["rank"], 0.0),
    ).groupby(anchor_column, sort=False).agg(
        positive=(target_column, "max"),
        top_three_hit=("top_three_hit", "max"),
        reciprocal=("reciprocal", "max"),
    )
    positive_anchors = per_anchor["positive"].eq(1)
    positive_anchor_count = int(positive_anchors.sum())
    return {
        "anchors": int(ranked[anchor_column].nunique()),
        "positive_labels": positive_labels,
        "selected_labels": selected_labels,
        "hits": hits,
        "recall": float(hits / positive_labels) if positive_labels else math.nan,
        "precision": float(hits / selected_labels) if selected_labels else math.nan,
        "positive_anchors": positive_anchor_count,
        "positive_anchor_hit_rate": (
            float(per_anchor.loc[positive_anchors, "top_three_hit"].mean())
            if positive_anchor_count
            else math.nan
        ),
        "top_one_recall": (
            float(top_one_hits / positive_labels) if positive_labels else math.nan
        ),
        "mean_reciprocal_rank": (
            float(per_anchor.loc[positive_anchors, "reciprocal"].mean())
            if positive_anchor_count
            else math.nan
        ),
    }


def fixed_bin_calibration(
    target: Any,
    probability: Any,
    *,
    minimum_rows: int = 500,
) -> dict[str, Any]:
    observed, forecast = _validate_target_probability(target, probability)
    if minimum_rows <= 0:
        raise ValueError("minimum_rows must be positive")
    bin_index = np.minimum(np.floor(10.0 * forecast).astype(int), 9)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    supported_errors: list[float] = []
    for index in range(10):
        mask = bin_index == index
        count = int(mask.sum())
        mean_probability = float(forecast[mask].mean()) if count else math.nan
        event_rate = float(observed[mask].mean()) if count else math.nan
        error = abs(mean_probability - event_rate) if count else math.nan
        supported = count >= minimum_rows
        if count:
            ece += count / len(observed) * error
        if supported:
            supported_errors.append(error)
        rows.append(
            {
                "bin": index,
                "lower": index / 10.0,
                "upper": (index + 1) / 10.0,
                "rows": count,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "absolute_error": error,
                "supported": supported,
            }
        )
    has_supported = bool(supported_errors)
    return {
        "rows": pd.DataFrame(rows),
        "ece": float(ece),
        "maximum_supported_bin_error": (
            float(max(supported_errors)) if has_supported else math.nan
        ),
        "has_supported_bin": has_supported,
        "passable": has_supported,
    }


def lambda_month_log_losses(
    predictions: pd.DataFrame,
    *,
    month_column: str = "validation_month",
    target_column: str = "target",
    probability_column: str = "probability",
) -> dict[str, float]:
    required = {month_column, target_column, probability_column}
    missing = sorted(required.difference(predictions.columns))
    if missing or predictions.empty:
        raise ValueError(f"missing or empty lambda predictions: {missing}")
    losses = binary_loss_arrays(
        predictions[target_column], predictions[probability_column]
    )["log_loss"]
    scratch = pd.DataFrame(
        {month_column: predictions[month_column].astype(str), "loss": losses}
    )
    monthly = scratch.groupby(month_column, sort=True)["loss"].mean()
    return {str(key): float(value) for key, value in monthly.items()}


def support_summary(
    frame: pd.DataFrame,
    rule: Mapping[str, int],
) -> dict[str, Any]:
    required = {
        "target",
        "cycle_id",
        "symbol_norm",
        "quarter",
        "state",
    }
    missing = sorted(required.difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"missing or empty support frame: {missing}")
    cycle_positive = frame.groupby("cycle_id", sort=True)["target"].sum()
    summary = {
        "compatible_rows": int(len(frame)),
        "positive_rows": int(frame["target"].sum()),
        "cycles": int(frame["cycle_id"].nunique()),
        "minimum_positive_rows_per_cycle": int(cycle_positive.min()),
        "stocks": int(frame["symbol_norm"].nunique()),
        "quarters": int(frame["quarter"].nunique()),
        "current_states": int(frame["state"].nunique()),
    }
    key_map = {
        "minimum_compatible_rows": "compatible_rows",
        "minimum_positive_rows": "positive_rows",
        "cycles": "cycles",
        "minimum_positive_rows_per_cycle": "minimum_positive_rows_per_cycle",
        "minimum_stocks": "stocks",
        "quarters": "quarters",
        "current_states": "current_states",
    }
    summary["pass"] = bool(
        all(
            summary[observed] >= int(required_value)
            if key.startswith("minimum_")
            else summary[observed] == int(required_value)
            for key, required_value in rule.items()
            for observed in (key_map[key],)
        )
    )
    return summary


def common_block_positions(
    date_count: int,
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
) -> np.ndarray:
    if date_count <= 0 or draws <= 0 or block_length <= 0:
        raise ValueError("bootstrap sizes must be positive")
    needed = int(math.ceil(date_count / block_length))
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, date_count, size=(draws, needed))
    offsets = np.arange(block_length, dtype=int)
    positions = (starts[:, :, None] + offsets[None, None, :]) % date_count
    return positions.reshape(draws, -1)[:, :date_count]


def familywise_bootstrap(
    frame: pd.DataFrame,
    *,
    candidate_column: str = PRIMARY_CANDIDATE,
    baseline_columns: Sequence[str] = PRIMARY_BASELINES,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    required = {"target", "session_date", candidate_column, *baseline_columns}
    missing = sorted(required.difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"missing or empty bootstrap frame: {missing}")
    dates = np.asarray(sorted(frame["session_date"].astype(str).unique()))
    endpoints: list[tuple[str, str]] = [
        (baseline, loss_name)
        for baseline in baseline_columns
        for loss_name in LOSS_NAMES
    ]
    daily = np.empty((len(dates), len(endpoints)), dtype=float)
    target = frame["target"].to_numpy(dtype=int)
    candidate_loss = binary_loss_arrays(target, frame[candidate_column])
    for column_index, (baseline, loss_name) in enumerate(endpoints):
        baseline_loss = binary_loss_arrays(target, frame[baseline])[loss_name]
        difference = candidate_loss[loss_name] - baseline_loss
        scratch = pd.DataFrame(
            {
                "session_date": frame["session_date"].astype(str).to_numpy(),
                "difference": difference,
            }
        )
        grouped = scratch.groupby("session_date", sort=True)["difference"].mean()
        daily[:, column_index] = grouped.reindex(dates).to_numpy(dtype=float)
    if not np.isfinite(daily).all():
        raise ValueError("bootstrap daily endpoints must be finite")
    positions = common_block_positions(len(dates), seed=seed, draws=draws)
    samples = daily[positions].mean(axis=1)
    upper = np.quantile(
        samples, BOOTSTRAP_UPPER_QUANTILE, axis=0, method="linear"
    )
    rows = []
    for index, (baseline, loss_name) in enumerate(endpoints):
        rows.append(
            {
                "baseline": baseline,
                "loss": loss_name,
                "daily_mean_difference": float(daily[:, index].mean()),
                "upper_bound": float(upper[index]),
                "pass": bool(upper[index] < 0.0),
            }
        )
    return {
        "rows": pd.DataFrame(rows),
        "block_positions": positions,
        "bootstrap_means": samples,
        "upper_quantile": BOOTSTRAP_UPPER_QUANTILE,
        "pass": bool(all(row["pass"] for row in rows)),
    }


def validate_prediction_panel(frame: pd.DataFrame) -> None:
    required = {
        "anchor_id",
        "session_date",
        "symbol_norm",
        "quarter",
        "state",
        "cycle_id",
        "transition_length",
        "target",
        "n_compatible",
        "bar_ordinal",
        "terminal",
        *FACTOR_QUARTILE_COLUMNS,
        *REPORT_SLICE_COLUMNS,
        *MODEL_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"missing or empty prediction panel: {missing}")
    if frame.duplicated(["anchor_id", "cycle_id"]).any():
        raise ValueError("prediction panel has duplicate anchor-cycle rows")
    target = _one_dimensional(frame["target"], "target", float)
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("prediction target must be binary")
    for column in MODEL_COLUMNS:
        _validate_target_probability(target, frame[column])
    inverse_compatible_weights(frame["n_compatible"])
    compatible = frame.groupby("anchor_id", sort=False)["cycle_id"].transform(
        "size"
    ).to_numpy(dtype=int)
    stored = frame["n_compatible"].to_numpy(dtype=float)
    if not np.array_equal(compatible.astype(float), stored):
        raise ValueError("n_compatible does not match expanded anchor rows")
    metadata = [
        "session_date",
        "symbol_norm",
        "quarter",
        "state",
        "n_compatible",
        "bar_ordinal",
        "terminal",
        *REPORT_SLICE_COLUMNS,
        *FACTOR_QUARTILE_COLUMNS,
    ]
    if (frame.groupby("anchor_id", sort=False)[metadata].nunique(dropna=False) > 1).any().any():
        raise ValueError("anchor metadata differs across compatible cycles")
    terminal = frame["terminal"].astype(bool).to_numpy()
    if target[terminal].sum() != 0.0:
        raise ValueError("terminal anchors must have zero loop labels")
    state = frame["state"].to_numpy(dtype=float)
    if not np.equal(state, np.floor(state)).all() or not np.isin(
        state, np.arange(8)
    ).all():
        raise ValueError("state must be an integer in [0, 7]")
    transition_length = frame["transition_length"].to_numpy(dtype=float)
    if not np.isin(transition_length, (2.0, 3.0, 4.0)).all():
        raise ValueError("transition length must be 2, 3, or 4")
    bar_ordinal = frame["bar_ordinal"].to_numpy(dtype=float)
    if (
        not np.isfinite(bar_ordinal).all()
        or not np.equal(bar_ordinal, np.floor(bar_ordinal)).all()
        or (bar_ordinal < 0.0).any()
    ):
        raise ValueError("bar ordinal must be nonnegative")


def validate_population(
    anchors: pd.DataFrame,
    expanded: pd.DataFrame,
    cycles: pd.DataFrame | Sequence[Any],
    period: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact frozen population counts without reading a source."""

    year = str(period)
    if year not in {"2024", "2025", "2023"}:
        raise ValueError(f"unsupported period: {period}")
    if anchors.empty or expanded.empty:
        raise ValueError("population frames must be nonempty")
    if "anchor_id" not in anchors or anchors["anchor_id"].duplicated().any():
        raise ValueError("anchors must have unique anchor_id values")
    natural_key = ["symbol_norm", "session_date", "start_timestamp"]
    if set(natural_key).issubset(anchors.columns) and anchors.duplicated(
        natural_key
    ).any():
        raise ValueError("anchors must have a unique natural key")
    if expanded.duplicated(["anchor_id", "cycle_id"]).any():
        raise ValueError("expanded rows must be unique anchor-cycle pairs")
    if set(expanded["anchor_id"]) != set(anchors["anchor_id"]):
        raise ValueError("expanded and anchor populations differ")
    target = _one_dimensional(expanded["target"], "target", float)
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("expanded target must be binary")
    expected_cycle_ids = (
        set(cycles["cycle_id"].astype(str))
        if isinstance(cycles, pd.DataFrame) and "cycle_id" in cycles
        else {str(value) for value in cycles}
    )
    if set(expanded["cycle_id"].astype(str)) != expected_cycle_ids:
        raise ValueError("expanded cycle identities differ from frozen cycles")
    cycle_count = len(cycles)
    expected_runs = int(contract["frozen_sources"][f"runs_{year}"]["rows"])
    expected_rows = int(
        contract["population_and_target"]["compatible_anchor_cycle_rows_expected"][
            year
        ]
    )
    expected_positives = int(
        contract["population_and_target"]["positive_rows_expected"][year]
    )
    observed = {
        "period": year,
        "anchors": int(len(anchors)),
        "compatible_rows": int(len(expanded)),
        "positive_rows": int(expanded["target"].sum()),
        "cycles": int(cycle_count),
    }
    observed["pass"] = bool(
        observed["anchors"] == expected_runs
        and observed["compatible_rows"] == expected_rows
        and observed["positive_rows"] == expected_positives
        and observed["cycles"] == int(contract["frozen_sources"]["cycles"]["count"])
    )
    if not observed["pass"]:
        raise ValueError(f"population count mismatch: {observed}")
    grouped = expanded.groupby("anchor_id", sort=False)["cycle_id"].transform(
        "size"
    ).to_numpy(dtype=int)
    if "n_compatible" in expanded and not np.array_equal(
        grouped, expanded["n_compatible"].to_numpy(dtype=int)
    ):
        raise ValueError("expanded n_compatible mismatch")
    if "terminal" in expanded:
        terminal = expanded["terminal"].astype(bool).to_numpy()
        if expanded.loc[terminal, "target"].sum() != 0:
            raise ValueError("terminal population has a positive loop label")
    return observed


def _model_loss_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for surface in ("unweighted", "inverse_compatible"):
        compatible = (
            None if surface == "unweighted" else frame["n_compatible"]
        )
        for model in MODEL_COLUMNS:
            row = {
                "surface": surface,
                "model": model,
                **loss_metrics(
                    frame["target"], frame[model], compatible_counts=compatible
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for surface in ("unweighted", "inverse_compatible"):
        compatible = (
            None if surface == "unweighted" else frame["n_compatible"]
        )
        for baseline in (*PRIMARY_BASELINES, LINEAGE_BASELINE):
            rows.append(
                {
                    "surface": surface,
                    "candidate": PRIMARY_CANDIDATE,
                    "baseline": baseline,
                    **loss_comparison(
                        frame["target"],
                        frame[PRIMARY_CANDIDATE],
                        frame[baseline],
                        compatible_counts=compatible,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _ranking_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": model, **top_three_metrics(frame, model)}
            for model in MODEL_COLUMNS
        ]
    )


def _calibration_payload(frame: pd.DataFrame) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    tables: list[pd.DataFrame] = []
    for model in (*PRIMARY_BASELINES, PRIMARY_CANDIDATE):
        result = fixed_bin_calibration(frame["target"], frame[model])
        summaries.append(
            {
                "model": model,
                "ece": result["ece"],
                "maximum_supported_bin_error": result[
                    "maximum_supported_bin_error"
                ],
                "has_supported_bin": result["has_supported_bin"],
            }
        )
        table = result["rows"].copy()
        table.insert(0, "model", model)
        tables.append(table)
    return {
        "summary": pd.DataFrame(summaries),
        "bins": pd.concat(tables, ignore_index=True),
    }


def _slice_loss_rows(
    frame: pd.DataFrame,
    *,
    family: str,
    groups: pd.Series,
    baselines: Sequence[str],
    minimum_rows: int,
    minimum_positives: int,
    losses: Sequence[str] = LOSS_NAMES,
    allow_zero: bool = False,
    gate_required: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = pd.Series(groups, index=frame.index, dtype="object")
    for value in sorted(grouped.dropna().astype(str).unique()):
        mask = grouped.astype(str).eq(value).to_numpy()
        subset = frame.loc[mask].reset_index(drop=True)
        supported = bool(
            len(subset) >= minimum_rows
            and int(subset["target"].sum()) >= minimum_positives
        )
        for baseline in baselines:
            comparison = loss_comparison(
                subset["target"], subset[PRIMARY_CANDIDATE], subset[baseline]
            )
            for loss_name in losses:
                difference = float(comparison[f"{loss_name}_difference"])
                rows.append(
                    {
                        "family": family,
                        "value": value,
                        "baseline": baseline,
                        "loss": loss_name,
                        "rows": int(len(subset)),
                        "positives": int(subset["target"].sum()),
                        "supported": supported,
                        "difference": difference,
                        "gate_required": gate_required,
                        "pass": bool(
                            (not supported or difference <= 0.0)
                            if allow_zero
                            else (not supported or difference < 0.0)
                        ),
                    }
                )
    return rows


def _subset_loss_rows(
    frame: pd.DataFrame,
    *,
    family: str,
    mask: Any,
    baselines: Sequence[str],
    minimum_rows: int,
    minimum_positives: int,
    gate_required: bool = True,
) -> list[dict[str, Any]]:
    selected = _one_dimensional(mask, "slice mask", bool)
    if len(selected) != len(frame):
        raise ValueError("slice mask length mismatch")
    groups = pd.Series(np.where(selected, "selected", None), index=frame.index)
    return _slice_loss_rows(
        frame,
        family=family,
        groups=groups,
        baselines=baselines,
        minimum_rows=minimum_rows,
        minimum_positives=minimum_positives,
        gate_required=gate_required,
    )


def slice_diagnostics(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    """Evaluate every frozen required gate slice on raw probabilities."""

    validate_prediction_panel(frame)
    rows: list[dict[str, Any]] = []
    time_group = (
        frame["session_date"].astype(str).str[:7]
        if str(period) == "2024"
        else frame["quarter"].astype(str)
    )
    for family, groups in (
        ("time", time_group),
        ("current_state", frame["state"]),
        ("transition_length", frame["transition_length"]),
    ):
        rows.extend(
            _slice_loss_rows(
                frame,
                family=family,
                groups=groups,
                baselines=PRIMARY_BASELINES,
                minimum_rows=5_000,
                minimum_positives=100,
            )
        )
    rows.extend(
        _subset_loss_rows(
            frame,
            family="nonterminal",
            mask=~frame["terminal"].astype(bool).to_numpy(),
            baselines=PRIMARY_BASELINES,
            minimum_rows=5_000,
            minimum_positives=100,
        )
    )
    rows.extend(
        _subset_loss_rows(
            frame,
            family="early_entry",
            mask=frame["bar_ordinal"].to_numpy(dtype=int) <= 53,
            baselines=PRIMARY_BASELINES,
            minimum_rows=5_000,
            minimum_positives=100,
        )
    )
    rows.extend(
        _slice_loss_rows(
            frame,
            family="cycle",
            groups=frame["cycle_id"],
            baselines=PRIMARY_BASELINES,
            minimum_rows=500,
            minimum_positives=40,
            losses=("log_loss",),
        )
    )
    orientation = (
        frame["cycle_id"].astype(str) + "__s" + frame["state"].astype(str)
    )
    rows.extend(
        _slice_loss_rows(
            frame,
            family="cycle_current_state_orientation",
            groups=orientation,
            baselines=("qlimited4",),
            minimum_rows=500,
            minimum_positives=40,
            allow_zero=True,
        )
    )
    for column in sorted(
        name for name in frame.columns if name.startswith("factor_quartile__")
    ):
        rows.extend(
            _slice_loss_rows(
                frame,
                family=column,
                groups=frame[column],
                baselines=("qlimited4",),
                minimum_rows=5_000,
                minimum_positives=100,
                allow_zero=True,
            )
        )
    for column in REPORT_SLICE_COLUMNS:
        rows.extend(
            _slice_loss_rows(
                frame,
                family=column,
                groups=frame[column],
                baselines=PRIMARY_BASELINES,
                minimum_rows=5_000,
                minimum_positives=100,
                gate_required=False,
            )
        )
    rows.extend(
        _subset_loss_rows(
            frame,
            family="terminal",
            mask=frame["terminal"].astype(bool).to_numpy(),
            baselines=PRIMARY_BASELINES,
            minimum_rows=5_000,
            minimum_positives=0,
            gate_required=False,
        )
    )
    rows.extend(
        _subset_loss_rows(
            frame,
            family="late_entry",
            mask=frame["bar_ordinal"].to_numpy(dtype=int) > 53,
            baselines=PRIMARY_BASELINES,
            minimum_rows=5_000,
            minimum_positives=100,
            gate_required=False,
        )
    )
    # Leave-one-stock-out deletions are large complementary cohorts and do not
    # use the ordinary per-slice positive support floor.
    for symbol in sorted(frame["symbol_norm"].astype(str).unique()):
        subset = frame.loc[frame["symbol_norm"].astype(str).ne(symbol)]
        if subset.empty:
            for baseline in PRIMARY_BASELINES:
                for loss_name in LOSS_NAMES:
                    rows.append(
                        {
                            "family": "leave_one_stock_out",
                            "value": symbol,
                            "baseline": baseline,
                            "loss": loss_name,
                            "rows": 0,
                            "positives": 0,
                            "supported": False,
                            "difference": math.nan,
                            "gate_required": True,
                            "pass": False,
                        }
                    )
            continue
        for baseline in PRIMARY_BASELINES:
            comparison = loss_comparison(
                subset["target"], subset[PRIMARY_CANDIDATE], subset[baseline]
            )
            for loss_name in LOSS_NAMES:
                difference = float(comparison[f"{loss_name}_difference"])
                rows.append(
                    {
                        "family": "leave_one_stock_out",
                        "value": symbol,
                        "baseline": baseline,
                        "loss": loss_name,
                        "rows": int(len(subset)),
                        "positives": int(subset["target"].sum()),
                        "supported": True,
                        "difference": difference,
                        "gate_required": True,
                        "pass": bool(difference < 0.0),
                    }
                )
    return pd.DataFrame(rows)


def holm_step_down(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.01,
    order: Sequence[str] = (
        "unweighted_log_loss",
        "inverse_compatible_weighted_log_loss",
        "top_three_recall",
    ),
) -> dict[str, Any]:
    if set(p_values) != set(order):
        raise ValueError("Holm p-value family differs from declared order")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Holm alpha must be in (0, 1)")
    order_index = {name: index for index, name in enumerate(order)}
    ranked = sorted(order, key=lambda name: (float(p_values[name]), order_index[name]))
    rows = []
    family_pass = True
    count = len(ranked)
    for rank, name in enumerate(ranked, start=1):
        p_value = float(p_values[name])
        if not 0.0 <= p_value <= 1.0:
            raise ValueError("Holm p-values must be in [0, 1]")
        threshold = alpha / (count - rank + 1)
        passed = p_value <= threshold
        family_pass &= passed
        rows.append(
            {
                "name": name,
                "rank": rank,
                "p_value": p_value,
                "threshold": threshold,
                "pass": passed,
            }
        )
    return {"rows": pd.DataFrame(rows), "pass": bool(family_pass)}


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    values = np.asarray(logit, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _falsification_statistics(
    frame: pd.DataFrame,
    candidate_probability: np.ndarray,
) -> np.ndarray:
    target = frame["target"].to_numpy(dtype=int)
    limited = frame["qlimited4"].to_numpy(dtype=float)
    unweighted = loss_comparison(target, candidate_probability, limited)[
        "relative_log_loss_improvement"
    ]
    weighted = loss_comparison(
        target,
        candidate_probability,
        limited,
        compatible_counts=frame["n_compatible"],
    )["relative_log_loss_improvement"]
    scratch = frame.copy()
    scratch["__candidate"] = candidate_probability
    candidate_recall = float(top_three_metrics(scratch, "__candidate")["recall"])
    limited_recall = float(top_three_metrics(scratch, "qlimited4")["recall"])
    return np.asarray(
        [unweighted, weighted, candidate_recall - limited_recall], dtype=float
    )


def _eligible_falsification_strata(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    required = {"start_timestamp", "anchor_id", "session_date", "symbol_norm", "state"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing falsification columns: {missing}")
    metadata = frame.loc[
        :, ["anchor_id", "symbol_norm", "session_date", "start_timestamp", "state"]
    ]
    if (
        metadata.groupby("anchor_id", sort=False)[
            ["symbol_norm", "session_date", "start_timestamp", "state"]
        ].nunique(dropna=False)
        > 1
    ).any().any():
        raise ValueError("falsification anchor metadata is not unique")
    anchor = metadata.drop_duplicates("anchor_id")
    anchor = anchor.sort_values(
        ["symbol_norm", "session_date", "start_timestamp", "anchor_id"],
        kind="stable",
    ).reset_index(drop=True)
    anchor["month"] = anchor["session_date"].astype(str).str[:7]
    strata_anchor_ids: list[np.ndarray] = []
    for _, indices in anchor.groupby(
        ["symbol_norm", "month", "state"], sort=True
    ).groups.items():
        positions = np.asarray(indices, dtype=int)
        selected = anchor.loc[positions]
        if selected["session_date"].nunique() >= 2:
            strata_anchor_ids.append(selected["anchor_id"].to_numpy())
    eligible_ids = {
        value for values in strata_anchor_ids for value in values.tolist()
    }
    eligible = frame.loc[frame["anchor_id"].isin(eligible_ids)].copy()
    eligible = eligible.sort_values(["anchor_id", "cycle_id"], kind="stable").reset_index(
        drop=True
    )
    if eligible.empty or not strata_anchor_ids:
        raise ValueError("falsification has no eligible multi-session stratum")
    return eligible, strata_anchor_ids


def left_rotate_session_blocks(
    session_blocks: Sequence[Any], boundary: int
) -> np.ndarray:
    """Return whole anchor blocks ordered k+1..S,1..k for boundary k."""

    blocks = [_one_dimensional(block, "session block", int) for block in session_blocks]
    if len(blocks) < 2 or any(not len(block) for block in blocks):
        raise ValueError("session rotation requires at least two nonempty blocks")
    if boundary < 1 or boundary >= len(blocks):
        raise ValueError("session boundary must be in [1, S-1]")
    joined = np.concatenate(blocks)
    if len(np.unique(joined)) != len(joined):
        raise ValueError("session blocks contain a duplicate anchor position")
    return np.concatenate(blocks[boundary:] + blocks[:boundary])


def falsification_diagnostics(
    predictions: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = FALSIFICATION_SEED,
) -> dict[str, Any]:
    """Run the coherent three-statistic whole-session residual-logit null."""

    validate_prediction_panel(predictions)
    if draws <= 0:
        raise ValueError("falsification draws must be positive")
    frame, strata_anchor_ids = _eligible_falsification_strata(predictions)
    cycle_order = sorted(frame["cycle_id"].astype(str).unique())
    anchors = sorted(frame["anchor_id"].unique())
    anchor_position = {anchor: index for index, anchor in enumerate(anchors)}
    cycle_position = {cycle: index for index, cycle in enumerate(cycle_order)}
    row_anchor = frame["anchor_id"].map(anchor_position).to_numpy(dtype=int)
    row_cycle = frame["cycle_id"].astype(str).map(cycle_position).to_numpy(dtype=int)
    limited_eta = _logit(frame["qlimited4"].to_numpy(dtype=float))
    full_eta = _logit(frame["qfull9"].to_numpy(dtype=float))
    residual = np.full((len(anchors), len(cycle_order)), np.nan, dtype=float)
    residual[row_anchor, row_cycle] = full_eta - limited_eta
    limited_matrix = np.full_like(residual, np.nan)
    limited_matrix[row_anchor, row_cycle] = limited_eta
    target_matrix = np.zeros_like(residual, dtype=np.int8)
    target_matrix[row_anchor, row_cycle] = frame["target"].to_numpy(dtype=np.int8)
    replayed_full = _sigmoid(limited_eta + (full_eta - limited_eta))
    replay_error = float(
        np.max(np.abs(replayed_full - frame["qfull9"].to_numpy(dtype=float)))
    )
    if replay_error > 1e-12:
        raise ValueError("falsification full-logit replay exceeds tolerance")
    target = frame["target"].to_numpy(dtype=float)
    weights = inverse_compatible_weights(frame["n_compatible"])
    limited_loss = np.logaddexp(0.0, limited_eta) - target * limited_eta
    limited_mean = float(limited_loss.mean())
    limited_weighted_mean = weighted_mean(limited_loss, weights)
    positive_anchor, positive_cycle = np.nonzero(target_matrix)
    if not len(positive_anchor):
        raise ValueError("falsification requires at least one positive label")

    def recall_from_logits(logits: np.ndarray) -> float:
        positive_scores = logits[positive_anchor]
        own_score = logits[positive_anchor, positive_cycle]
        higher = np.sum(positive_scores > own_score[:, None], axis=1)
        cycle_indices = np.arange(logits.shape[1], dtype=int)
        earlier_ties = np.sum(
            (positive_scores == own_score[:, None])
            & (cycle_indices[None, :] < positive_cycle[:, None]),
            axis=1,
        )
        rank = 1 + higher + earlier_ties
        return float(np.mean(rank <= 3))

    limited_recall = recall_from_logits(limited_matrix)

    def statistics_from_logits(logits: np.ndarray) -> np.ndarray:
        long_logits = logits[row_anchor, row_cycle]
        if not np.isfinite(long_logits).all():
            raise ValueError("falsification produced a nonfinite compatible logit")
        candidate_loss = np.logaddexp(0.0, long_logits) - target * long_logits
        unweighted_improvement = (
            limited_mean - float(candidate_loss.mean())
        ) / limited_mean
        weighted_improvement = (
            limited_weighted_mean - weighted_mean(candidate_loss, weights)
        ) / limited_weighted_mean
        recall_improvement = recall_from_logits(logits) - limited_recall
        return np.asarray(
            [unweighted_improvement, weighted_improvement, recall_improvement],
            dtype=float,
        )

    full_matrix = limited_matrix + residual
    observed = statistics_from_logits(full_matrix)
    direct = _falsification_statistics(
        frame, frame["qfull9"].to_numpy(dtype=float)
    )
    statistic_replay_error = float(np.max(np.abs(observed - direct)))
    if statistic_replay_error > 1e-12:
        raise ValueError("optimized falsification statistic failed direct replay")

    # Convert canonical anchor IDs to positions and then to whole-session blocks.
    strata: list[list[np.ndarray]] = []
    anchor_meta = frame.loc[
        :, ["anchor_id", "session_date", "start_timestamp"]
    ].drop_duplicates("anchor_id").set_index("anchor_id")
    for anchor_ids in strata_anchor_ids:
        ordered = anchor_meta.loc[list(anchor_ids)].reset_index().sort_values(
            ["session_date", "start_timestamp", "anchor_id"], kind="stable"
        )
        cycle_signatures = (
            frame.loc[frame["anchor_id"].isin(anchor_ids)]
            .groupby("anchor_id", sort=False)["cycle_id"]
            .agg(lambda values: tuple(sorted(values.astype(str))))
        )
        if cycle_signatures.nunique() != 1:
            raise ValueError(
                "anchors in a falsification state stratum have different cycles"
            )
        sessions = [
            np.asarray([anchor_position[value] for value in group["anchor_id"]], dtype=int)
            for _, group in ordered.groupby("session_date", sort=True)
        ]
        if len(sessions) < 2:
            raise AssertionError("eligible falsification stratum lost a session")
        strata.append(sessions)

    rng = np.random.Generator(np.random.PCG64(seed))
    null = np.empty((draws, 3), dtype=float)
    for draw in range(draws):
        shifted = residual.copy()
        for sessions in strata:
            boundary = int(rng.integers(1, len(sessions)))
            target_positions = np.concatenate(sessions)
            donor_positions = left_rotate_session_blocks(sessions, boundary)
            shifted[target_positions] = residual[donor_positions]
        null[draw] = statistics_from_logits(limited_matrix + shifted)
    p_values_array = (1 + (null >= observed[None, :]).sum(axis=0)) / (draws + 1)
    names = (
        "unweighted_log_loss",
        "inverse_compatible_weighted_log_loss",
        "top_three_recall",
    )
    p_values = {
        name: float(p_values_array[index]) for index, name in enumerate(names)
    }
    holm = holm_step_down(p_values, alpha=0.01, order=names)
    return {
        "draws": int(draws),
        "seed": int(seed),
        "eligible_rows": int(len(frame)),
        "eligible_anchors": int(frame["anchor_id"].nunique()),
        "strata": int(len(strata)),
        "full_logit_replay_max_error": replay_error,
        "statistic_replay_max_error": statistic_replay_error,
        "statistics": pd.DataFrame(
            {
                "name": names,
                "observed": observed,
                "null_mean": null.mean(axis=0),
                "null_q99": np.quantile(null, 0.99, axis=0, method="linear"),
                "p_value": p_values_array,
            }
        ),
        "null_statistics": null,
        "holm": holm,
        "pass": bool(holm["pass"]),
    }


def _default_support_rule(period: str) -> dict[str, int]:
    if str(period) == "2024":
        return {
            "minimum_compatible_rows": 300_000,
            "minimum_positive_rows": 10_000,
            "cycles": 20,
            "minimum_positive_rows_per_cycle": 100,
            "minimum_stocks": 20,
            "quarters": 2,
            "current_states": 8,
        }
    if str(period) in {"2025", "2023"}:
        return {
            "minimum_compatible_rows": 300_000,
            "minimum_positive_rows": 10_000,
            "cycles": 20,
            "minimum_positive_rows_per_cycle": 100,
            "minimum_stocks": 18,
            "quarters": 4,
            "current_states": 8,
        }
    raise ValueError(f"unsupported period: {period}")


def _support_rule(
    period: str, contract: Mapping[str, Any] | None
) -> Mapping[str, int]:
    if contract is None:
        return _default_support_rule(period)
    key = "2024_outer_oof" if str(period) == "2024" else "later_each_period"
    return contract["support_gates"][key]


def _row_by_model(frame: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["model"]): row
        for row in frame.to_dict(orient="records")
    }


def _comparison_by_baseline(
    comparisons: pd.DataFrame, surface: str
) -> dict[str, Mapping[str, Any]]:
    selected = comparisons.loc[comparisons["surface"].eq(surface)]
    return {
        str(row["baseline"]): row
        for row in selected.to_dict(orient="records")
    }


def _slice_stability_gates(slices: pd.DataFrame, period: str) -> dict[str, Any]:
    if slices.empty:
        return {"pass": False, "reason": "empty_slice_table"}
    supported = slices.loc[slices["supported"].astype(bool)].copy()
    required_all = (
        "time",
        "leave_one_stock_out",
        "current_state",
        "transition_length",
        "nonterminal",
        "early_entry",
    )
    family_results: dict[str, bool] = {}
    for family in required_all:
        family_rows = supported.loc[supported["family"].eq(family)]
        family_results[family] = bool(
            not family_rows.empty and family_rows["pass"].all()
        )
    observed_times = set(
        supported.loc[supported["family"].eq("time"), "value"].astype(str)
    )
    expected_times = (
        {f"2024-{month:02d}" for month in range(7, 13)}
        if str(period) == "2024"
        else {f"{period}_q{quarter}" for quarter in range(1, 5)}
    )
    family_results["time_expected_values"] = observed_times == expected_times

    cycle = supported.loc[
        supported["family"].eq("cycle") & supported["loss"].eq("log_loss")
    ]
    cycle_counts = {
        baseline: int(
            cycle.loc[
                cycle["baseline"].eq(baseline) & cycle["pass"].astype(bool),
                "value",
            ].nunique()
        )
        for baseline in PRIMARY_BASELINES
    }
    cycle_pass = all(value >= 15 for value in cycle_counts.values())

    orientation = supported.loc[
        supported["family"].eq("cycle_current_state_orientation")
    ]
    orientation_pass = bool(not orientation.empty and orientation["pass"].all())
    quartile_results: dict[str, bool] = {}
    for column in FACTOR_QUARTILE_COLUMNS:
        selected = supported.loc[supported["family"].eq(column)]
        quartile_results[column] = bool(not selected.empty and selected["pass"].all())
    passed = bool(
        all(family_results.values())
        and cycle_pass
        and orientation_pass
        and all(quartile_results.values())
    )
    return {
        "required_families": family_results,
        "cycle_improvement_counts": cycle_counts,
        "cycle_pass": cycle_pass,
        "orientation_pass": orientation_pass,
        "factor_quartile_pass": quartile_results,
        "pass": passed,
    }


def evaluate_primary_gates(
    frame: pd.DataFrame,
    *,
    period: str,
    support: Mapping[str, Any],
    comparisons: pd.DataFrame,
    ranking: pd.DataFrame,
    calibration: Mapping[str, pd.DataFrame],
    bootstrap: Mapping[str, Any],
    slices: pd.DataFrame,
    falsification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply every frozen raw-probability gate to prepared artifacts."""

    unweighted = _comparison_by_baseline(comparisons, "unweighted")
    inverse = _comparison_by_baseline(comparisons, "inverse_compatible")
    ranking_map = _row_by_model(ranking)
    calibration_map = _row_by_model(calibration["summary"])
    bootstrap_rows = {
        (str(row["baseline"]), str(row["loss"])): row
        for row in bootstrap["rows"].to_dict(orient="records")
    }
    comparison_gates: dict[str, Any] = {}
    for baseline, requirement in PRIMARY_REQUIREMENTS.items():
        pooled = unweighted[baseline]
        candidate_rank = ranking_map[PRIMARY_CANDIDATE]
        baseline_rank = ranking_map[baseline]
        checks = {
            "relative_log_loss": float(pooled["relative_log_loss_improvement"])
            >= requirement["relative_log_loss"],
            "brier": float(pooled["brier_difference"]) < 0.0,
            "bootstrap_log_loss": bool(
                bootstrap_rows[(baseline, "log_loss")]["pass"]
            ),
            "bootstrap_brier": bool(
                bootstrap_rows[(baseline, "brier")]["pass"]
            ),
            "top_three_recall": (
                float(candidate_rank["recall"]) - float(baseline_rank["recall"])
            )
            >= requirement["recall_gain"],
            "top_three_precision": float(candidate_rank["precision"])
            - float(baseline_rank["precision"])
            >= 0.0,
            "positive_anchor_hit_rate": float(
                candidate_rank["positive_anchor_hit_rate"]
            )
            - float(baseline_rank["positive_anchor_hit_rate"])
            >= -0.002,
            "inverse_log_loss": float(inverse[baseline]["log_loss_difference"])
            <= 0.0,
            "inverse_brier": float(inverse[baseline]["brier_difference"]) <= 0.0,
        }
        comparison_gates[baseline] = {
            "checks": checks,
            "pass": bool(all(checks.values())),
        }

    lineage = unweighted[LINEAGE_BASELINE]
    lineage_checks = {
        "log_loss": float(lineage["log_loss_difference"]) <= 0.0,
        "brier": float(lineage["brier_difference"]) <= 0.0,
        "top_three_recall": float(ranking_map[PRIMARY_CANDIDATE]["recall"])
        - float(ranking_map[LINEAGE_BASELINE]["recall"])
        >= 0.0,
    }
    candidate_calibration = calibration_map[PRIMARY_CANDIDATE]
    history_calibration = calibration_map["qhistory"]
    calibration_checks = {
        "supported": bool(candidate_calibration["has_supported_bin"]),
        "ece": all(
            float(candidate_calibration["ece"])
            <= float(calibration_map[baseline]["ece"])
            for baseline in PRIMARY_BASELINES
        ),
        "absolute_maximum": float(
            candidate_calibration["maximum_supported_bin_error"]
        )
        <= 0.02,
        "history_margin": float(
            candidate_calibration["maximum_supported_bin_error"]
        )
        <= float(history_calibration["maximum_supported_bin_error"]) + 0.005,
    }
    stability = _slice_stability_gates(slices, period)
    pooled_primary_pass = bool(
        support["pass"]
        and all(value["pass"] for value in comparison_gates.values())
        and all(lineage_checks.values())
        and all(calibration_checks.values())
        and bootstrap["pass"]
    )
    falsification_pass = bool(
        falsification is not None and falsification.get("pass", False)
    )
    return {
        "support_pass": bool(support["pass"]),
        "comparisons": comparison_gates,
        "lineage_baseline": {
            "checks": lineage_checks,
            "pass": bool(all(lineage_checks.values())),
        },
        "calibration": {
            "checks": calibration_checks,
            "pass": bool(all(calibration_checks.values())),
        },
        "stability": stability,
        "bootstrap_pass": bool(bootstrap["pass"]),
        "falsification_pass": falsification_pass,
        "pooled_primary_pass": pooled_primary_pass,
        "primary_pass": bool(
            pooled_primary_pass and stability["pass"] and falsification_pass
        ),
    }


def evaluate_period(
    predictions: pd.DataFrame,
    period: str,
    contract: Mapping[str, Any] | None = None,
    *,
    seed_offset: int | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    falsification_draws: int = BOOTSTRAP_DRAWS,
    include_falsification: bool = True,
) -> dict[str, Any]:
    """Evaluate one immutable prediction panel and return runner artifacts."""

    year = str(period)
    validate_prediction_panel(predictions)
    rule = _support_rule(year, contract)
    support = support_summary(predictions, rule)
    overall = _model_loss_table(predictions)
    comparisons = _comparison_table(predictions)
    ranking = _ranking_table(predictions)
    calibration = _calibration_payload(predictions)
    offset = PERIOD_SEED_OFFSETS[year] if seed_offset is None else int(seed_offset)
    bootstrap = familywise_bootstrap(
        predictions,
        seed=FALSIFICATION_SEED + offset,
        draws=bootstrap_draws,
    )
    slices = slice_diagnostics(predictions, year)
    falsification = (
        falsification_diagnostics(
            predictions, draws=falsification_draws, seed=FALSIFICATION_SEED
        )
        if include_falsification
        else None
    )
    gates = evaluate_primary_gates(
        predictions,
        period=year,
        support=support,
        comparisons=comparisons,
        ranking=ranking,
        calibration=calibration,
        bootstrap=bootstrap,
        slices=slices,
        falsification=falsification,
    )
    artifacts = {
        "support": support,
        "overall": overall,
        "ranking": ranking,
        "calibration": calibration,
        "comparisons": comparisons,
        "bootstrap": bootstrap,
        "slices": slices,
        "falsification": falsification,
        "gates": gates,
    }
    return {
        "period": year,
        "artifacts": artifacts,
        "primary_pass": bool(gates["primary_pass"]),
    }


def evaluate_irregular_deletion(
    predictions_2025: pd.DataFrame,
    *,
    original_evaluation: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Evaluate the frozen five-cell 2025 deletion without fitting anything."""

    validate_prediction_panel(predictions_2025)
    symbol = predictions_2025["symbol_norm"].astype(str)
    date = predictions_2025["session_date"].astype(str)
    affected = symbol.isin(IRREGULAR_SYMBOLS) & date.eq(IRREGULAR_DATE)
    affected_frame = predictions_2025.loc[affected]
    deleted_runs = int(affected_frame["anchor_id"].nunique())
    deleted_rows = int(len(affected_frame))
    observed_symbols = tuple(sorted(affected_frame["symbol_norm"].astype(str).unique()))
    count_pass = bool(
        deleted_runs == 93
        and deleted_rows == 547
        and observed_symbols == IRREGULAR_SYMBOLS
    )
    if not count_pass:
        raise ValueError(
            "irregular deletion cohort mismatch: "
            f"runs={deleted_runs}, rows={deleted_rows}, symbols={observed_symbols}"
        )
    reduced = predictions_2025.loc[~affected].copy().reset_index(drop=True)
    evaluation = evaluate_period(
        reduced,
        "2025",
        contract,
        bootstrap_draws=bootstrap_draws,
        include_falsification=False,
    )
    original_pooled = bool(
        original_evaluation["artifacts"]["gates"]["pooled_primary_pass"]
    )
    reduced_pooled = bool(
        evaluation["artifacts"]["gates"]["pooled_primary_pass"]
    )
    return {
        "deleted_runs": deleted_runs,
        "deleted_compatible_rows": deleted_rows,
        "symbols": list(observed_symbols),
        "no_refit": True,
        "count_pass": count_pass,
        "original_pooled_primary_pass": original_pooled,
        "reduced_pooled_primary_pass": reduced_pooled,
        "pass": bool(original_pooled and reduced_pooled),
        "evaluation": evaluation,
    }


def derive_decision(
    evaluation_2024: Mapping[str, Any],
    evaluation_2025: Mapping[str, Any] | None = None,
    evaluation_2023: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive the frozen development/demotion-only decision."""

    pass_2024 = bool(evaluation_2024.get("primary_pass", False))
    if not pass_2024:
        label = (
            "factor_conditioned_loop_occurrence_rejected_2024_"
            "and_do_not_score_later_periods"
        )
        later_scoring_eligible = False
    elif evaluation_2025 is None and evaluation_2023 is None:
        label = "factor_conditioned_loop_occurrence_development_candidate"
        later_scoring_eligible = True
    elif evaluation_2025 is None or evaluation_2023 is None:
        raise ValueError("later transfer decision requires both 2025 and 2023")
    elif bool(evaluation_2025.get("primary_pass", False)) and bool(
        evaluation_2023.get("primary_pass", False)
    ):
        label = "development_candidate_retained_pending_prospective"
        later_scoring_eligible = False
    else:
        label = "development_algorithm_unconfirmed"
        later_scoring_eligible = False
    return {
        "label": label,
        "2024_pass": pass_2024,
        "2025_pass": (
            None
            if evaluation_2025 is None
            else bool(evaluation_2025.get("primary_pass", False))
        ),
        "2023_pass": (
            None
            if evaluation_2023 is None
            else bool(evaluation_2023.get("primary_pass", False))
        ),
        "later_scoring_eligible_after_independent_audit": later_scoring_eligible,
        "later_scoring_authorized": False,
        "later_periods_can_promote": False,
        "parent_loop_identity_model_changed": False,
        "good_or_high_movement_quality_grade_changed": False,
        "research_only": RESEARCH_ONLY,
        "live_ordering_enabled": LIVE_ORDERING_ENABLED,
        "order_placement": ORDER_PLACEMENT,
    }


def self_tests() -> None:
    losses = binary_loss_arrays(np.asarray([0, 1]), np.asarray([0.25, 0.75]))
    assert np.allclose(losses["log_loss"], -np.log(0.75))
    assert np.allclose(losses["brier"], 0.0625)
    assert np.array_equal(
        inverse_compatible_weights(np.asarray([1, 2, 4])),
        np.asarray([1.0, 0.5, 0.25]),
    )
    calibration = fixed_bin_calibration(
        np.asarray([0, 1]), np.asarray([0.0, 1.0]), minimum_rows=1
    )
    assert calibration["rows"].loc[9, "rows"] == 1
    positions = common_block_positions(7, seed=20260711, draws=3)
    assert positions.shape == (3, 7)
    holm = holm_step_down(
        {
            "unweighted_log_loss": 0.001,
            "inverse_compatible_weighted_log_loss": 0.002,
            "top_three_recall": 0.003,
        }
    )
    assert holm["pass"]
