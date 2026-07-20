"""Independent audit for High-Movement Pressure-Onset Screen V0.

This file deliberately does not import the experiment runner or its reusable module.
"""

# ruff: noqa: E402 -- numerical thread limits must be fixed before imports.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PREDECESSOR_PANEL = (
    REPO_ROOT
    / "research"
    / "opening-regime-path"
    / "20260720-opening-regime-path-direction-screen-v0"
    / "artifacts"
    / "primary"
    / "opening_decision_panel.parquet"
)
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
START = pd.Timestamp("2024-01-01T00:00:00Z")
BOOTSTRAP_SEED = 20260720
NULL_SEED = 20260721
RANDOM_SEED = 20260722
FORBIDDEN = (
    "regime",
    "state",
    "loop",
    "closure",
    "excursion",
    "transition",
    "posterior",
    "structural",
)
SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "feasibility_screen": True,
    "observable_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "loops_regimes_states_and_structural_paths_forbidden": True,
}
REQUIRED = (
    "contract.json",
    "source_manifest.json",
    "input_artifact_hashes.json",
    "protected_boundary_audit.json",
    "predecessor_reconstruction.json",
    "feature_manifest.json",
    "forbidden_feature_audit.json",
    "movement_oof_fold_manifest.json",
    "movement_admission_thresholds.json",
    "onset_barriers.json",
    "compact_decision_panel.parquet",
    "onset_path_ledger.parquet",
    "development_oof_predictions.parquet",
    "assessment_predictions.parquet",
    "model_configurations.json",
    "model_coefficients.json",
    "onset_metrics.csv",
    "direction_metrics.csv",
    "monthly_metrics.csv",
    "checkpoint_metrics.csv",
    "calibration_bins.csv",
    "bootstrap_metrics.csv",
    "null_metrics.csv",
    "economic_reference_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "exact_rerun_manifest.json",
    "report.md",
)


class AuditFailure(RuntimeError):
    """A failed independent integrity assertion."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrow_hash(frame: pd.DataFrame) -> str:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AuditFailure(detail)


def maximum_error(actual: Sequence[float], expected: Sequence[float]) -> float:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    require(left.shape == right.shape, "numeric comparison shapes differ")
    if left.size == 0:
        return 0.0
    both_nan = np.isnan(left) & np.isnan(right)
    differences = np.where(both_nan, 0.0, np.abs(left - right))
    return float(np.nanmax(differences))


def manual_probability(model: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    features = [str(value) for value in model["feature_names"]]
    values = frame.loc[:, features].to_numpy(dtype=float)
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    linear = float(model["intercept"]) + ((values - means) / scales) @ coefficients
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))


def equal_slate_weights(frame: pd.DataFrame) -> np.ndarray:
    sizes = frame.groupby("slate_id", sort=True)["symbol"].transform("size")
    return 1.0 / sizes.to_numpy(dtype=float)


def fixed_fit(
    frame: pd.DataFrame, labels: pd.Series, features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    values = frame.loc[:, list(features)].to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=20260720,
        n_jobs=1,
    )
    estimator.fit(
        (values - means) / scales,
        labels.to_numpy(dtype=int),
        sample_weight=equal_slate_weights(frame),
    )
    require(int(np.max(estimator.n_iter_)) < 250, "audited null fit did not converge")
    return means, scales, estimator


def fixed_predict(
    frame: pd.DataFrame,
    features: Sequence[str],
    means: np.ndarray,
    scales: np.ndarray,
    estimator: LogisticRegression,
) -> np.ndarray:
    values = frame.loc[:, list(features)].to_numpy(dtype=float)
    return estimator.predict_proba((values - means) / scales)[:, 1]


def permute_bundle(frame: pd.DataFrame, features: Sequence[str], *, seed: int) -> pd.DataFrame:
    output = frame.copy()
    source = frame.copy()
    rng = np.random.default_rng(seed)
    for _, index in output.groupby("slate_id", sort=True).groups.items():
        positions = np.asarray(list(index))
        permutation = rng.permutation(len(positions))
        output.loc[positions, list(features)] = source.loc[positions, list(features)].to_numpy(
            copy=True
        )[permutation]
    return output


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    outcomes = np.asarray(labels, dtype=int)
    prediction = np.asarray(probabilities, dtype=float)
    clipped = np.clip(prediction, 1e-15, 1.0 - 1e-15)
    return {
        "brier_score": float(np.mean((outcomes - prediction) ** 2)),
        "log_loss": float(
            -np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1.0 - clipped))
        ),
        "auc": (
            float(roc_auc_score(outcomes, prediction))
            if len(np.unique(outcomes)) == 2
            else math.nan
        ),
        "base_rate": float(np.mean(outcomes)),
    }


def calibration_parameters(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    outcomes = np.asarray(labels, dtype=float)
    if len(np.unique(outcomes)) < 2:
        return math.nan, math.nan
    prediction = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    logits = np.log(prediction / (1.0 - prediction))

    def objective(parameters: np.ndarray) -> float:
        linear = parameters[0] + parameters[1] * logits
        fitted = 1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0)))
        return float(
            -np.sum(
                outcomes * np.log(np.clip(fitted, 1e-15, 1.0))
                + (1.0 - outcomes) * np.log(np.clip(1.0 - fitted, 1e-15, 1.0))
            )
        )

    result = minimize(
        objective,
        np.asarray([0.0, 1.0]),
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    return float(result.x[0]), float(result.x[1])


def stable_random_choice(frame: pd.DataFrame, slate_id: str) -> tuple[pd.Series, float]:
    ordered = frame.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256(f"{RANDOM_SEED}:{slate_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return ordered.iloc[value % len(ordered)], 1.0 if digest[8] % 2 == 0 else -1.0


def economic_selections(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = (
        "readiness",
        "pressure",
        "confirmed",
        "highest_relative_momentum",
        "strongest_reversal",
        "random_within_slate",
    )
    for slate_id, slate in primary.groupby("slate_id", sort=True):
        ordered = slate.sort_values("symbol", kind="mergesort").reset_index(drop=True)
        require(int(ordered["source_slate_size"].min()) >= 15, "economic source slate <15")
        for candidate in candidates:
            if candidate in {"readiness", "pressure", "confirmed"}:
                column = f"signed_pressure_score__{candidate}"
                selected = (
                    ordered.assign(_absolute=ordered[column].abs())
                    .sort_values(
                        ["_absolute", "symbol"],
                        ascending=[False, True],
                        kind="mergesort",
                    )
                    .iloc[0]
                )
                score = float(selected[column])
                direction = 1.0 if score >= 0.0 else -1.0
            elif candidate in {"highest_relative_momentum", "strongest_reversal"}:
                selected = (
                    ordered.assign(
                        _absolute=ordered["open_to_decision_cohort_relative_return_bps"].abs()
                    )
                    .sort_values(
                        ["_absolute", "symbol"],
                        ascending=[False, True],
                        kind="mergesort",
                    )
                    .iloc[0]
                )
                momentum = float(selected["open_to_decision_cohort_relative_return_bps"])
                direction = 1.0 if momentum >= 0.0 else -1.0
                if candidate == "strongest_reversal":
                    direction *= -1.0
                score = direction * abs(momentum)
            else:
                selected, direction = stable_random_choice(ordered, str(slate_id))
                score = direction
            rows.append(
                {
                    "candidate": candidate,
                    "slate_id": str(slate_id),
                    "session": str(selected["session"]),
                    "decision_ordinal": int(selected["decision_ordinal"]),
                    "symbol": str(selected["symbol"]),
                    "score": score,
                    "direction_sign": direction,
                    "signed_gross_return_bps_30m": direction
                    * float(selected["raw_continuation_return_bps"]),
                    "signed_cohort_relative_return_bps_30m": direction
                    * float(selected["cohort_relative_continuation_return_bps"]),
                    "signed_gross_return_bps_remaining_session": direction
                    * float(selected["raw_remaining_session_return_bps"]),
                    "signed_cohort_relative_return_bps_remaining_session": direction
                    * float(selected["cohort_relative_remaining_session_return_bps"]),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["candidate", "session", "decision_ordinal"], kind="mergesort")
        .reset_index(drop=True)
    )


def economic_aggregate(selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    horizons = {
        "primary_30m_close_t_plus_8": (
            "signed_gross_return_bps_30m",
            "signed_cohort_relative_return_bps_30m",
        ),
        "secondary_remaining_session": (
            "signed_gross_return_bps_remaining_session",
            "signed_cohort_relative_return_bps_remaining_session",
        ),
    }
    for candidate, part in selections.groupby("candidate", sort=True):
        for horizon, (gross_column, relative_column) in horizons.items():
            gross = part[gross_column].to_numpy(dtype=float)
            relative = part[relative_column].to_numpy(dtype=float)
            for friction in (0.0, 10.0, 20.0):
                net = gross - friction
                rows.append(
                    {
                        "candidate": candidate,
                        "horizon": horizon,
                        "friction_bps": friction,
                        "mean_signed_gross_return_bps": float(np.mean(gross)),
                        "mean_signed_return_after_friction_bps": float(np.mean(net)),
                        "median_signed_return_after_friction_bps": float(np.median(net)),
                        "positive_after_friction_rate": float(np.mean(net > 0.0)),
                        "mean_signed_cohort_relative_return_bps": float(np.mean(relative)),
                        "selected_rows": len(part),
                        "sessions": int(part["session"].nunique()),
                        "stocks": int(part["symbol"].nunique()),
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["horizon", "friction_bps", "candidate"], kind="mergesort")
        .reset_index(drop=True)
    )


def bootstrap_loss(
    frame: pd.DataFrame,
    target: str,
    baseline: str,
    candidate: str,
    counts: Counter[str],
    kind: str,
) -> float:
    weights = frame["session"].astype(str).map(counts).fillna(0.0).to_numpy(dtype=float)
    mask = weights > 0.0
    outcomes = frame[target].to_numpy(dtype=float)[mask]
    before = frame[baseline].to_numpy(dtype=float)[mask]
    after = frame[candidate].to_numpy(dtype=float)[mask]
    selected_weights = weights[mask]
    if kind == "brier":
        before_loss = (outcomes - before) ** 2
        after_loss = (outcomes - after) ** 2
    else:
        before = np.clip(before, 1e-15, 1.0 - 1e-15)
        after = np.clip(after, 1e-15, 1.0 - 1e-15)
        before_loss = -(outcomes * np.log(before) + (1.0 - outcomes) * np.log(1.0 - before))
        after_loss = -(outcomes * np.log(after) + (1.0 - outcomes) * np.log(1.0 - after))
    return float(
        np.average(before_loss, weights=selected_weights)
        - np.average(after_loss, weights=selected_weights)
    )


def bootstrap_economic(
    selections: pd.DataFrame,
    baseline: str,
    candidate: str,
    counts: Counter[str],
) -> float:
    means: dict[str, float] = {}
    for name in (baseline, candidate):
        part = selections.loc[selections["candidate"].eq(name)]
        weights = part["session"].astype(str).map(counts).fillna(0.0).to_numpy(dtype=float)
        mask = weights > 0.0
        means[name] = float(
            np.average(
                part["signed_gross_return_bps_30m"].to_numpy(dtype=float)[mask] - 20.0,
                weights=weights[mask],
            )
        )
    return means[candidate] - means[baseline]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sources(source_manifest: Mapping[str, Any], provider_root: Path) -> dict[str, Any]:
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    monthly_parts: list[pd.DataFrame] = []
    for record in source_manifest["sources"]:
        symbol = str(record["symbol"])
        path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        frame = pd.read_parquet(
            path,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
            filters=[
                ("timestamp", ">=", START.to_pydatetime()),
                ("timestamp", "<", PROTECTED_START.to_pydatetime()),
            ],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        require(frame["timestamp"].lt(PROTECTED_START).all(), "protected source row")
        require(len(frame) == int(record["bounded_safe_rows"]), f"source rows differ: {symbol}")
        require(arrow_hash(frame) == record["bounded_safe_hash"], f"source hash differs: {symbol}")
        current_min = pd.Timestamp(frame["timestamp"].min())
        current_max = pd.Timestamp(frame["timestamp"].max())
        minimum = current_min if minimum is None else min(minimum, current_min)
        maximum = current_max if maximum is None else max(maximum, current_max)
        months = frame[["timestamp"]].copy()
        months["year_month"] = months["timestamp"].dt.strftime("%Y-%m")
        monthly_parts.append(months)
    counts = (
        pd.concat(monthly_parts, ignore_index=True)
        .groupby("year_month", sort=True)
        .size()
        .to_dict()
    )
    expected_counts = {
        str(row["year_month"]): int(row["row_count"])
        for row in source_manifest["source_rows_by_year_month"]
    }
    require(counts == expected_counts, "source year-month counts differ")
    require(str(minimum) == source_manifest["minimum_timestamp_read"], "source minimum differs")
    require(str(maximum) == source_manifest["maximum_timestamp_read"], "source maximum differs")
    return {
        "sources_verified": len(source_manifest["sources"]),
        "minimum_timestamp_read": str(minimum),
        "maximum_timestamp_read": str(maximum),
        "year_months_verified": len(counts),
    }


def verify_windows_and_paths(panel: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, Any]:
    require(set(panel["decision_ordinal"].astype(int)) == {6, 12}, "checkpoint set differs")
    require(
        np.array_equal(
            panel["repo_bar_start_ordinal"].to_numpy(dtype=int),
            panel["decision_ordinal"].to_numpy(dtype=int) - 1,
        ),
        "decision bar ordinal differs",
    )
    decision = pd.to_datetime(panel["decision_available_timestamp"], utc=True)
    predecessor_decision = pd.to_datetime(panel["feature_available_timestamp_utc"], utc=True)
    confirmation = pd.to_datetime(panel["confirmation_available_timestamp"], utc=True)
    entry = pd.to_datetime(panel["entry_timestamp"], utc=True)
    require(decision.equals(predecessor_decision), "decision timestamps differ")
    require((confirmation == decision + pd.Timedelta(minutes=5)).all(), "t+1 timing differs")
    require((entry == confirmation).all(), "t+2 entry timing differs")
    local = decision.dt.tz_convert("America/New_York")
    expected_minutes = panel["decision_ordinal"].map({6: 600, 12: 630}).to_numpy(dtype=int)
    actual_minutes = (local.dt.hour * 60 + local.dt.minute).to_numpy(dtype=int)
    require(np.array_equal(actual_minutes, expected_minutes), "decision local clock differs")
    require(
        np.array_equal(
            panel["entry_bar_ordinal"].to_numpy(dtype=int),
            panel["decision_ordinal"].to_numpy(dtype=int) + 1,
        ),
        "delayed t+2 entry ordinal differs",
    )
    for step, increment in ((2, 1), (3, 2), (4, 3)):
        require(
            np.array_equal(
                panel[f"onset_t_plus_{step}_bar_ordinal"].to_numpy(dtype=int),
                panel["decision_ordinal"].to_numpy(dtype=int) + increment,
            ),
            f"t+{step} path ordinal differs",
        )
        closes = pd.to_datetime(panel[f"onset_t_plus_{step}_close_timestamp"], utc=True)
        require(
            (closes == entry + pd.Timedelta(minutes=5 * (step - 1))).all(),
            f"t+{step} close timestamp differs",
        )
    require(
        np.array_equal(
            panel["continuation_exit_bar_ordinal"].to_numpy(dtype=int),
            panel["decision_ordinal"].to_numpy(dtype=int) + 7,
        ),
        "t+8 continuation exit differs",
    )
    ledger_sorted = ledger.sort_values(
        ["session", "decision_ordinal", "symbol", "path_step"], kind="mergesort"
    )
    require(len(ledger_sorted) == 3 * len(panel), "onset path ledger row count differs")
    max_path_error = 0.0
    for _, slate in panel.groupby("slate_id", sort=True):
        values = slate[
            [
                "raw_onset_t_plus_2_bps",
                "raw_onset_t_plus_3_bps",
                "raw_onset_t_plus_4_bps",
            ]
        ].to_numpy(dtype=float)
        medians = np.asarray(
            [np.median(np.delete(values, index, axis=0), axis=0) for index in range(len(values))]
        )
        expected = values - medians
        actual = slate[
            [
                "residual_t_plus_2_bps",
                "residual_t_plus_3_bps",
                "residual_t_plus_4_bps",
            ]
        ].to_numpy(dtype=float)
        max_path_error = max(max_path_error, maximum_error(actual, expected))
    require(max_path_error <= 1e-10, "cohort-relative onset paths differ")
    return {
        "rows_verified": len(panel),
        "ledger_rows_verified": len(ledger),
        "maximum_cohort_path_error": max_path_error,
        "checkpoints": [6, 12],
    }


def prepare_readiness_audit_bars(path: Path) -> pd.DataFrame:
    """Independently prepare bounded complete regular sessions for feature checks."""

    raw = pd.read_parquet(
        path,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        filters=[
            ("timestamp", ">=", START.to_pydatetime()),
            ("timestamp", "<", PROTECTED_START.to_pydatetime()),
        ],
    )
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "volume"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "open", "high", "low", "close"])
    timestamps = raw["timestamp"]
    local = timestamps.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    in_regular = minute.ge(570) & minute.lt(960)
    on_grid = ((minute - 570) % 5).eq(0) & local.dt.second.eq(0) & local.dt.microsecond.eq(0)
    invalid_sessions = set(local.loc[in_regular & ~on_grid].dt.strftime("%Y-%m-%d"))
    frame = raw.loc[in_regular & on_grid].copy()
    local_regular = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("America/New_York")
    regular_minute = local_regular.dt.hour * 60 + local_regular.dt.minute
    frame["session"] = local_regular.dt.strftime("%Y-%m-%d")
    frame["bar_ordinal"] = ((regular_minute - 570) // 5).astype(int)
    frame = frame.sort_values(
        ["session", "bar_ordinal", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)

    # The inherited readiness fields were frozen by the predecessor before its
    # complete-session eligibility filter. Reconstruct that ordering directly:
    # incomplete historical sessions may supply a prior close or a same-clock
    # activity-baseline observation, but can never become experiment rows.
    grouped_all = frame.groupby("session", sort=False)
    previous_ordinal = grouped_all["bar_ordinal"].shift(1)
    previous_timestamp = grouped_all["timestamp"].shift(1)
    segment_start = (
        previous_ordinal.isna()
        | frame["bar_ordinal"].sub(previous_ordinal).ne(1)
        | pd.to_datetime(frame["timestamp"], utc=True)
        .sub(pd.to_datetime(previous_timestamp, utc=True))
        .ne(pd.Timedelta(minutes=5))
    )
    frame["audit_segment_id"] = segment_start.cumsum()
    frame["predecessor_cumulative_activity"] = frame.groupby("audit_segment_id", sort=False)[
        "volume"
    ].cumsum()
    daily = (
        frame.groupby("session", sort=False)
        .agg(predecessor_session_close=("close", "last"))
        .reset_index()
    )
    daily["predecessor_prior_session_close"] = daily["predecessor_session_close"].shift(1)
    frame = frame.merge(
        daily[["session", "predecessor_prior_session_close"]],
        on="session",
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values(["session", "bar_ordinal", "timestamp"], kind="mergesort")
    frame["predecessor_cumulative_activity_baseline"] = frame.groupby("bar_ordinal", sort=False)[
        "predecessor_cumulative_activity"
    ].transform(lambda values: values.expanding(min_periods=10).mean().shift(1))
    frame["predecessor_activity_shock"] = np.log1p(
        frame["predecessor_cumulative_activity"]
        / frame["predecessor_cumulative_activity_baseline"].replace(0.0, np.nan)
    )
    valid: list[pd.DataFrame] = []
    for _, session in frame.groupby("session", sort=True):
        ordered = session.sort_values("bar_ordinal", kind="mergesort").copy()
        prices = ordered[["open", "high", "low", "close"]].to_numpy(dtype=float)
        activity = ordered["volume"].to_numpy(dtype=float)
        if (
            str(ordered.iloc[0]["session"]) not in invalid_sessions
            and len(ordered) == 78
            and ordered["bar_ordinal"].tolist() == list(range(78))
            and np.isfinite(prices).all()
            and bool((prices > 0.0).all())
            and np.isfinite(activity).all()
            and bool((activity >= 0.0).all())
        ):
            valid.append(ordered)
    require(bool(valid), f"no readiness audit sessions in {path.name}")
    output = pd.concat(valid, ignore_index=True).sort_values(
        ["session", "bar_ordinal"], kind="mergesort"
    )
    grouped = output.groupby("session", sort=False)
    prior_bar_close = grouped["close"].shift(1)
    denominator = prior_bar_close.where(output["bar_ordinal"].ne(0), output["open"])
    output["bar_return"] = output["close"] / denominator - 1.0
    output["true_range_bps"] = (
        10_000.0
        * np.maximum.reduce(
            [
                output["high"].to_numpy(dtype=float) - output["low"].to_numpy(dtype=float),
                np.abs(output["high"].to_numpy(dtype=float) - denominator.to_numpy(dtype=float)),
                np.abs(output["low"].to_numpy(dtype=float) - denominator.to_numpy(dtype=float)),
            ]
        )
        / denominator.to_numpy(dtype=float)
    )
    return output.reset_index(drop=True)


def verify_readiness_from_bounded_bars(panel: pd.DataFrame, provider_root: Path) -> dict[str, Any]:
    """Reconstruct all 18 readiness inputs independently from bounded causal bars."""

    predecessor = pd.read_parquet(
        PREDECESSOR_PANEL,
        columns=["symbol", "session", "decision_ordinal", "slate_id"],
    )
    predecessor["session"] = predecessor["session"].astype(str)
    records: list[dict[str, Any]] = []
    for symbol in sorted(panel["symbol"].astype(str).unique()):
        bars = prepare_readiness_audit_bars(
            provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        )
        sessions = {
            str(session): part.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
            for session, part in bars.groupby("session", sort=True)
        }
        requested = predecessor.loc[predecessor["symbol"].eq(symbol)]
        for row in requested.itertuples(index=False):
            session_name = str(row.session)
            if session_name not in sessions:
                continue
            session = sessions[session_name]
            predecessor_prior_close = float(session.iloc[0]["predecessor_prior_session_close"])
            if not np.isfinite(predecessor_prior_close):
                continue
            origin = int(row.decision_ordinal) - 1
            returns = session["bar_return"].to_numpy(dtype=float)
            ranges = session["true_range_bps"].to_numpy(dtype=float)
            last_three = returns[origin - 2 : origin + 1]
            last_six = returns[origin - 5 : origin + 1]
            opening = session.iloc[: origin + 1]
            session_open = float(session.iloc[0]["open"])
            current_close = float(session.iloc[origin]["close"])
            opening_high = float(opening["high"].max())
            opening_low = float(opening["low"].min())
            records.append(
                {
                    "symbol": symbol,
                    "session": session_name,
                    "decision_ordinal": int(row.decision_ordinal),
                    "slate_id": str(row.slate_id),
                    "opening_gap_bps": 10_000.0 * (session_open / predecessor_prior_close - 1.0),
                    "open_to_decision_raw_return_bps": 10_000.0
                    * (current_close / session_open - 1.0),
                    "latest_one_bar_return_bps": 10_000.0 * returns[origin],
                    "latest_three_bar_return_bps": 10_000.0
                    * (float(np.prod(1.0 + last_three)) - 1.0),
                    "latest_six_bar_return_bps": 10_000.0 * (float(np.prod(1.0 + last_six)) - 1.0),
                    "realized_volatility_3_bps": 10_000.0 * float(np.std(last_three, ddof=0)),
                    "realized_volatility_6_bps": 10_000.0 * float(np.std(last_six, ddof=0)),
                    "opening_high_low_range_bps": 10_000.0
                    * (opening_high - opening_low)
                    / session_open,
                    "current_true_range_bps": float(ranges[origin]),
                    "short_true_range_to_longer_true_range": float(
                        np.mean(ranges[origin - 1 : origin + 1])
                        / np.mean(ranges[origin - 5 : origin + 1])
                    ),
                    "distance_from_opening_high_bps": 10_000.0
                    * (opening_high - current_close)
                    / opening_high,
                    "distance_from_opening_low_bps": 10_000.0
                    * (current_close - opening_low)
                    / opening_low,
                    "historical_activity_proxy_shock": float(
                        session.iloc[origin]["predecessor_activity_shock"]
                    ),
                }
            )
    reconstructed = pd.DataFrame(records).sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    reconstructed["source_slate_size"] = reconstructed.groupby("slate_id", sort=True)[
        "symbol"
    ].transform("size")
    reconstructed = reconstructed.loc[reconstructed["source_slate_size"].ge(15)].copy()
    ordered = reconstructed.sort_values(["symbol", "decision_ordinal", "session"], kind="mergesort")
    ordered["trailing_volatility"] = ordered.groupby(["symbol", "decision_ordinal"], sort=False)[
        "realized_volatility_3_bps"
    ].transform(lambda values: values.rolling(20, min_periods=10).median().shift(1))
    ordered["trailing_range"] = ordered.groupby(["symbol", "decision_ordinal"], sort=False)[
        "opening_high_low_range_bps"
    ].transform(lambda values: values.rolling(20, min_periods=10).median().shift(1))
    ordered["short_realized_volatility_ratio"] = (
        ordered["realized_volatility_3_bps"] / ordered["trailing_volatility"]
    )
    ordered["opening_range_to_trailing_same_checkpoint_median"] = (
        ordered["opening_high_low_range_bps"] / ordered["trailing_range"]
    )
    reconstructed.loc[
        ordered.index,
        [
            "short_realized_volatility_ratio",
            "opening_range_to_trailing_same_checkpoint_median",
        ],
    ] = ordered[
        [
            "short_realized_volatility_ratio",
            "opening_range_to_trailing_same_checkpoint_median",
        ]
    ]
    keys = ["symbol", "session", "decision_ordinal"]
    comparison = panel[keys].merge(
        reconstructed,
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    require(len(comparison) == len(panel), "readiness reconstruction row count differs")
    base_features = (
        "opening_gap_bps",
        "open_to_decision_raw_return_bps",
        "latest_one_bar_return_bps",
        "latest_three_bar_return_bps",
        "latest_six_bar_return_bps",
        "realized_volatility_3_bps",
        "realized_volatility_6_bps",
        "short_realized_volatility_ratio",
        "opening_high_low_range_bps",
        "opening_range_to_trailing_same_checkpoint_median",
        "current_true_range_bps",
        "short_true_range_to_longer_true_range",
        "distance_from_opening_high_bps",
        "distance_from_opening_low_bps",
        "historical_activity_proxy_shock",
    )
    errors = {
        feature: maximum_error(panel[feature], comparison[feature]) for feature in base_features
    }
    cohort_errors: dict[str, float] = {
        "open_to_decision_cohort_relative_return_bps": 0.0,
        "cross_sectional_dispersion_bps": 0.0,
    }
    for _, slate in panel.groupby("slate_id", sort=True):
        raw_values = slate["open_to_decision_raw_return_bps"].to_numpy(dtype=float)
        expected_relative = np.asarray(
            [
                value - np.median(np.delete(raw_values, index))
                for index, value in enumerate(raw_values)
            ]
        )
        cohort_errors["open_to_decision_cohort_relative_return_bps"] = max(
            cohort_errors["open_to_decision_cohort_relative_return_bps"],
            maximum_error(
                slate["open_to_decision_cohort_relative_return_bps"],
                expected_relative,
            ),
        )
        expected_dispersion = np.full(len(slate), np.std(raw_values, ddof=1))
        cohort_errors["cross_sectional_dispersion_bps"] = max(
            cohort_errors["cross_sectional_dispersion_bps"],
            maximum_error(slate["cross_sectional_dispersion_bps"], expected_dispersion),
        )
    errors.update(cohort_errors)
    require(max(errors.values()) <= 1e-8, f"readiness reconstruction differs: {errors}")
    return {
        "readiness_features_reconstructed": 18,
        "movement_probability_verified_separately": True,
        "maximum_readiness_error": max(errors.values()),
        "feature_errors": errors,
    }


def verify_features(panel: pd.DataFrame, manifest: Mapping[str, Any]) -> dict[str, Any]:
    readiness = [str(value) for value in manifest["readiness_features"]]
    pressure = [str(value) for value in manifest["pressure_onset_additions"]]
    confirmation = [str(value) for value in manifest["confirmation_features"]]
    require(len(readiness) <= 18, "readiness feature maximum exceeded")
    require(len(pressure) <= 28, "pressure feature maximum exceeded")
    require(manifest["vwap_status"] == "vwap_features_unavailable", "VWAP status differs")
    for name in [*readiness, *pressure, *confirmation]:
        require(not any(value in name.lower() for value in FORBIDDEN), f"forbidden feature: {name}")
        require(name in panel.columns, f"feature missing from compact panel: {name}")
    finite_population = panel.loc[panel["p_large_remaining_move"].notna()]
    require(
        np.isfinite(finite_population[readiness].to_numpy(dtype=float)).all(),
        "readiness feature is non-finite in scored population",
    )
    require(
        np.isfinite(finite_population[pressure].to_numpy(dtype=float)).all(),
        "pressure feature is non-finite in scored population",
    )
    errors: dict[str, float] = {}
    errors["relative_strength_acceleration"] = maximum_error(
        panel["relative_strength_acceleration_bps"],
        panel["relative_return_last_3_bps"] - panel["relative_return_previous_3_bps"],
    )
    errors["activity_acceleration"] = maximum_error(
        panel["activity_acceleration"],
        np.log1p(panel["activity_last_2_mean"]) - np.log1p(panel["activity_previous_4_mean"]),
    )
    errors["range_acceleration"] = maximum_error(
        panel["range_acceleration"],
        panel["range_last_2_mean_bps"] / panel["range_previous_4_mean_bps"],
    )
    for window in (3, 6):
        signed = panel[f"calc__return_sum_{window}"] / panel[f"calc__absolute_return_sum_{window}"]
        errors[f"signed_efficiency_{window}"] = maximum_error(
            panel[f"signed_efficiency_{window}"], signed
        )
        errors[f"absolute_efficiency_{window}"] = maximum_error(
            panel[f"absolute_efficiency_{window}"], signed.abs()
        )
    raw_progress = panel["relative_return_last_3_bps"] / np.maximum(
        panel["calc__relative_activity_last_3_mean"], 1e-12
    )
    bounds = manifest["progress_per_activity_winsor_bounds"]
    errors["progress_per_activity_unwinsorized"] = maximum_error(
        panel["signed_progress_per_activity_unwinsorized"], raw_progress
    )
    errors["progress_per_activity_winsorized"] = maximum_error(
        panel["signed_progress_per_activity"],
        raw_progress.clip(bounds["lower_q01"], bounds["upper_q99"]),
    )
    delta_specs = {
        "change_cohort_relative_return_bps": (
            "open_to_decision_cohort_relative_return_bps",
            "t1__open_to_decision_cohort_relative_return_bps",
        ),
        "change_relative_strength_acceleration": (
            "relative_strength_acceleration_bps",
            "t1__relative_strength_acceleration_bps",
        ),
        "change_activity_shock": (
            "historical_activity_proxy_shock",
            "t1__historical_activity_proxy_shock",
        ),
        "change_range_acceleration": ("range_acceleration", "t1__range_acceleration"),
        "change_signed_efficiency_3": (
            "signed_efficiency_3",
            "t1__signed_efficiency_3",
        ),
        "change_close_location": (
            "current_close_location",
            "t1__current_close_location",
        ),
    }
    confirmation_finalized = (
        manifest.get("confirmation_status") != "not_finalized_due_to_support_gate"
    )
    confirmed = (
        panel.loc[panel["predicted_direction_remained_same"].notna()]
        if confirmation_finalized
        else panel
    )
    require(not confirmed.empty, "confirmation source population is empty")
    for target, (before, after) in delta_specs.items():
        errors[target] = maximum_error(confirmed[target], confirmed[after] - confirmed[before])
    errors["new_high_at_t_plus_1"] = maximum_error(
        confirmed["new_high_at_t_plus_1"],
        (confirmed["t1__current_high"] > confirmed["calc__opening_high"]).astype(float),
    )
    errors["new_low_at_t_plus_1"] = maximum_error(
        confirmed["new_low_at_t_plus_1"],
        (confirmed["t1__current_low"] < confirmed["calc__opening_low"]).astype(float),
    )
    errors["opening_range_acceptance_persisted"] = maximum_error(
        confirmed["opening_range_acceptance_persisted"],
        confirmed["t1__opening_range_acceptance_code"].eq(
            confirmed["calc__opening_range_acceptance_code"]
        ),
    )
    require(max(errors.values()) <= 1e-10, f"feature formula mismatch: {errors}")
    return {
        "readiness_features_verified": len(readiness),
        "pressure_features_verified": len(pressure),
        "confirmation_features_verified": (len(confirmation) if confirmation_finalized else 9),
        "confirmation_finalized": confirmation_finalized,
        "maximum_formula_error": max(errors.values()),
        "formula_errors": errors,
    }


def verify_thresholds_and_labels(
    panel: pd.DataFrame,
    oof: pd.DataFrame,
    threshold_artifact: Mapping[str, Any],
    barrier_artifact: Mapping[str, Any],
    fold_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        [str(row["score_month"]) for row in fold_manifest["folds"]]
        == [f"2024-{month:02d}" for month in range(7, 13)],
        "OOF score months differ",
    )
    require(
        all(
            str(row["training_end_month"]) < str(row["score_month"])
            for row in fold_manifest["folds"]
        ),
        "OOF chronology differs",
    )
    predecessor = pd.read_parquet(PREDECESSOR_PANEL)
    predecessor["session"] = predecessor["session"].astype(str)
    oof_copy = oof.copy()
    oof_copy["session"] = oof_copy["session"].astype(str)
    keys = ["symbol", "session", "decision_ordinal"]
    oof_prediction_errors: dict[str, float] = {}
    reconstructed_rows = 0
    for fold in fold_manifest["folds"]:
        score_month = str(fold["score_month"])
        scoring = predecessor.loc[
            predecessor["year_month"].astype(str).eq(score_month)
        ].sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
        stored = oof_copy.loc[oof_copy["year_month"].astype(str).eq(score_month)].sort_values(
            ["session", "decision_ordinal", "symbol"], kind="mergesort"
        )
        require(
            scoring[keys].reset_index(drop=True).equals(stored[keys].reset_index(drop=True)),
            f"OOF score keys differ for {score_month}",
        )
        reconstructed = manual_probability(fold["model"], scoring)
        oof_prediction_errors[score_month] = maximum_error(
            reconstructed, stored["p_large_remaining_move"]
        )
        reconstructed_rows += len(scoring)
    require(reconstructed_rows == len(oof), "OOF reconstructed row count differs")
    require(
        max(oof_prediction_errors.values()) <= 1e-12,
        "OOF movement probability reconstruction differs",
    )
    threshold_errors: dict[str, float] = {}
    barrier_errors: dict[str, float] = {}
    for ordinal in (6, 12):
        scores = oof.loc[oof["decision_ordinal"].eq(ordinal), "p_large_remaining_move"]
        expected_threshold = float(scores.quantile(0.75, interpolation="linear"))
        stored_threshold = float(threshold_artifact["thresholds"][str(ordinal)])
        threshold_errors[str(ordinal)] = abs(expected_threshold - stored_threshold)
        scored_panel = panel.loc[
            panel["decision_ordinal"].eq(ordinal) & panel["p_large_remaining_move"].notna()
        ]
        expected_admission = scored_panel["p_large_remaining_move"].ge(stored_threshold)
        require(
            expected_admission.equals(scored_panel["high_movement_admitted"]),
            f"admission differs at checkpoint {ordinal}",
        )
        development = panel.loc[panel["year"].eq(2024) & panel["decision_ordinal"].eq(ordinal)]
        paths = development[
            [
                "residual_t_plus_2_bps",
                "residual_t_plus_3_bps",
                "residual_t_plus_4_bps",
            ]
        ].to_numpy(dtype=float)
        expected_barrier = float(
            pd.Series(np.max(np.abs(paths), axis=1)).quantile(0.75, interpolation="linear")
        )
        stored_barrier = float(barrier_artifact["barriers_bps"][str(ordinal)])
        barrier_errors[str(ordinal)] = abs(expected_barrier - stored_barrier)
        require(
            maximum_error(
                panel.loc[panel["decision_ordinal"].eq(ordinal), "onset_barrier_bps"],
                np.full(int(panel["decision_ordinal"].eq(ordinal).sum()), stored_barrier),
            )
            <= 1e-10,
            f"row barrier differs at checkpoint {ordinal}",
        )
    expected_labels: list[str] = []
    for row in panel.itertuples(index=False):
        label = "NO_ONSET"
        for value in (
            row.residual_t_plus_2_bps,
            row.residual_t_plus_3_bps,
            row.residual_t_plus_4_bps,
        ):
            if value >= row.onset_barrier_bps:
                label = "UP_ONSET"
                break
            if value <= -row.onset_barrier_bps:
                label = "DOWN_ONSET"
                break
        expected_labels.append(label)
    require(expected_labels == panel["onset_label"].astype(str).tolist(), "onset labels differ")
    require(max(threshold_errors.values()) <= 1e-12, "movement thresholds differ")
    require(max(barrier_errors.values()) <= 1e-12, "onset barriers differ")
    return {
        "oof_prediction_errors": oof_prediction_errors,
        "oof_rows_reconstructed": reconstructed_rows,
        "threshold_errors": threshold_errors,
        "barrier_errors": barrier_errors,
        "onset_label_counts": panel["onset_label"].value_counts().sort_index().to_dict(),
    }


def verify_models(assessment: pd.DataFrame, coefficients: Mapping[str, Any]) -> dict[str, Any]:
    errors: dict[str, float] = {}
    for model_name, model in coefficients["models"].items():
        column = (
            f"p_onset__{model_name}"
            if str(model_name).startswith("A")
            else f"p_up_given_onset__{model_name}"
        )
        errors[str(model_name)] = maximum_error(
            assessment[column], manual_probability(model, assessment)
        )
        require(bool(model["converged"]), f"model did not converge: {model_name}")
        require(int(model["iterations"]) < 250, f"model hit max iterations: {model_name}")
    m1 = coefficients["frozen_predecessor_M1"]
    errors["frozen_predecessor_M1"] = maximum_error(
        assessment["p_large_remaining_move"], manual_probability(m1, assessment)
    )
    d2 = coefficients["models"]["D2"]
    t1 = assessment.copy()
    for feature in d2["feature_names"]:
        if feature in {"checkpoint_60m", "p_large_remaining_move"}:
            continue
        t1[feature] = assessment[f"t1__{feature}"]
    direction_t = manual_probability(d2, assessment) >= 0.5
    direction_t1 = manual_probability(d2, t1) >= 0.5
    errors["predicted_direction_remained_same"] = maximum_error(
        assessment["predicted_direction_remained_same"],
        (direction_t == direction_t1).astype(float),
    )
    require(max(errors.values()) <= 1e-12, f"manual prediction mismatch: {errors}")
    return {"manual_prediction_maximum_errors": errors, "models_verified": 8}


def metric_scope_frame(assessment: pd.DataFrame, row: Any) -> pd.DataFrame:
    frame = (
        assessment.loc[assessment["high_movement_admitted"].astype(bool)]
        if row.population == "primary_high_movement"
        else assessment
    )
    if row.scope_type == "month":
        frame = frame.loc[frame["year_month"].astype(str).eq(str(row.scope_value))]
    elif row.scope_type == "checkpoint":
        frame = frame.loc[frame["decision_ordinal"].eq(int(row.scope_value))]
    if row.target == "up_given_onset":
        frame = frame.loc[frame["directional_onset"].eq(1)]
    return frame


def verify_metrics(
    assessment: pd.DataFrame,
    metric_frames: Sequence[pd.DataFrame],
    calibration: pd.DataFrame,
) -> dict[str, Any]:
    errors: list[float] = []
    rows_verified = 0
    for metrics in metric_frames:
        for row in metrics.itertuples(index=False):
            frame = metric_scope_frame(assessment, row)
            values = metric_values(
                frame[row.target].to_numpy(dtype=int),
                frame[row.probability_column].to_numpy(dtype=float),
            )
            for key in ("brier_score", "log_loss", "auc", "base_rate"):
                actual = float(getattr(row, key))
                expected = values[key]
                error = (
                    0.0 if math.isnan(actual) and math.isnan(expected) else abs(actual - expected)
                )
                errors.append(error)
            intercept, slope = calibration_parameters(
                frame[row.target].to_numpy(dtype=int),
                frame[row.probability_column].to_numpy(dtype=float),
            )
            errors.extend(
                [
                    abs(float(row.calibration_intercept) - intercept),
                    abs(float(row.calibration_slope) - slope),
                ]
            )
            require(int(row.rows) == len(frame), "metric row count differs")
            require(int(row.sessions) == frame["session"].nunique(), "metric sessions differ")
            require(int(row.stocks) == frame["symbol"].nunique(), "metric stocks differ")
            rows_verified += 1
    for keys, bins in calibration.groupby(
        ["population", "scope_type", "scope_value", "target", "model"], sort=True
    ):
        population, scope_type, scope_value, target, model = keys
        synthetic = type(
            "MetricSlice",
            (),
            {
                "population": population,
                "scope_type": scope_type,
                "scope_value": scope_value,
                "target": target,
            },
        )()
        frame = metric_scope_frame(assessment, synthetic)
        probability_column = (
            f"p_onset__{model}" if str(model).startswith("A") else f"p_up_given_onset__{model}"
        )
        predictions = frame[probability_column].to_numpy(dtype=float)
        outcomes = frame[target].to_numpy(dtype=float)
        bin_numbers = np.minimum((predictions * 10).astype(int), 9)
        for bin_row in bins.itertuples(index=False):
            mask = bin_numbers == int(bin_row.bin) - 1
            require(int(bin_row.rows) == int(mask.sum()), "calibration bin count differs")
            if mask.any():
                errors.extend(
                    [
                        abs(float(bin_row.mean_probability) - float(predictions[mask].mean())),
                        abs(float(bin_row.observed_rate) - float(outcomes[mask].mean())),
                    ]
                )
    require(max(errors) <= 1e-8, f"metric reconstruction differs: {max(errors)}")
    return {"metric_rows_verified": rows_verified, "maximum_metric_error": max(errors)}


def verify_economic(
    assessment: pd.DataFrame,
    economic_artifact: pd.DataFrame,
    concentration_artifact: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary = assessment.loc[assessment["high_movement_admitted"].astype(bool)]
    selections = economic_selections(primary)
    expected = economic_aggregate(selections)
    require(
        expected[["candidate", "horizon", "friction_bps"]].equals(
            economic_artifact[["candidate", "horizon", "friction_bps"]]
        ),
        "economic metric keys differ",
    )
    numeric = [
        "mean_signed_gross_return_bps",
        "mean_signed_return_after_friction_bps",
        "median_signed_return_after_friction_bps",
        "positive_after_friction_rate",
        "mean_signed_cohort_relative_return_bps",
    ]
    economic_error = max(
        maximum_error(expected[column], economic_artifact[column]) for column in numeric
    )
    require(economic_error <= 1e-8, "economic reference calculation differs")
    row_shares = primary["symbol"].value_counts(normalize=True)
    require(float(row_shares.max()) <= 0.10 + 1e-15, "primary row concentration fails")
    selected_maximum = float(
        selections.groupby("candidate", sort=True)["symbol"].value_counts(normalize=True).max()
    )
    require(selected_maximum <= 0.20 + 1e-15, "selected concentration fails")
    require(concentration_artifact["passes"].astype(bool).all(), "stored concentration failure")
    return selections, {
        "maximum_economic_metric_error": economic_error,
        "maximum_primary_row_stock_share": float(row_shares.max()),
        "maximum_selected_stock_share": selected_maximum,
    }


def verify_bootstrap(
    primary: pd.DataFrame,
    selections: pd.DataFrame,
    artifact: pd.DataFrame,
) -> dict[str, Any]:
    sessions = np.asarray(sorted(primary["session"].astype(str).unique()), dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    direction = primary.loc[primary["directional_onset"].eq(1)]
    specs = (
        (
            "A2_minus_A1_brier_improvement",
            primary,
            "directional_onset",
            "p_onset__A1",
            "p_onset__A2",
            "brier",
        ),
        (
            "A2_minus_A1_log_loss_improvement",
            primary,
            "directional_onset",
            "p_onset__A1",
            "p_onset__A2",
            "log_loss",
        ),
        (
            "D2_minus_D1_brier_improvement",
            direction,
            "up_given_onset",
            "p_up_given_onset__D1",
            "p_up_given_onset__D2",
            "brier",
        ),
        (
            "D2_minus_D1_log_loss_improvement",
            direction,
            "up_given_onset",
            "p_up_given_onset__D1",
            "p_up_given_onset__D2",
            "log_loss",
        ),
        (
            "A3_minus_A2_brier_improvement",
            primary,
            "directional_onset",
            "p_onset__A2",
            "p_onset__A3",
            "brier",
        ),
        (
            "D3_minus_D2_brier_improvement",
            direction,
            "up_given_onset",
            "p_up_given_onset__D2",
            "p_up_given_onset__D3",
            "brier",
        ),
    )
    expected: dict[str, list[float]] = {name: [] for name, *_ in specs}
    expected["pressure_minus_readiness_return_after_20bps"] = []
    expected["confirmation_minus_pressure_return_after_20bps"] = []
    for _draw in range(200):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        counts: Counter[str] = Counter(str(value) for value in sampled)
        for name, frame, target, baseline, candidate, kind in specs:
            expected[name].append(bootstrap_loss(frame, target, baseline, candidate, counts, kind))
        expected["pressure_minus_readiness_return_after_20bps"].append(
            bootstrap_economic(selections, "readiness", "pressure", counts)
        )
        expected["confirmation_minus_pressure_return_after_20bps"].append(
            bootstrap_economic(selections, "pressure", "confirmed", counts)
        )
    errors: list[float] = []
    for metric, values in expected.items():
        draws = artifact.loc[
            artifact["record_type"].eq("draw") & artifact["metric"].eq(metric)
        ].sort_values("draw")
        errors.append(maximum_error(draws["value"], values))
        summary = artifact.loc[
            artifact["record_type"].eq("summary") & artifact["metric"].eq(metric)
        ].iloc[0]
        array = np.asarray(values, dtype=float)
        errors.extend(
            [
                abs(float(summary.lower_90) - float(np.quantile(array, 0.05))),
                abs(float(summary.upper_90) - float(np.quantile(array, 0.95))),
                abs(float(summary.lower_95) - float(np.quantile(array, 0.025))),
                abs(float(summary.upper_95) - float(np.quantile(array, 0.975))),
            ]
        )
    require(max(errors) <= 1e-8, "session-block bootstrap differs")
    return {"draws_verified": 200, "maximum_bootstrap_error": max(errors)}


def brier_improvement(labels: pd.Series, baseline: np.ndarray, candidate: np.ndarray) -> float:
    outcomes = labels.to_numpy(dtype=float)
    return float(np.mean((outcomes - baseline) ** 2) - np.mean((outcomes - candidate) ** 2))


def pressure_only_selection(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _slate_id, slate in frame.groupby("slate_id", sort=True):
        selected = (
            slate.assign(_absolute=slate["signed_pressure_score__pressure"].abs())
            .sort_values(["_absolute", "symbol"], ascending=[False, True], kind="mergesort")
            .iloc[0]
        )
        score = float(selected["signed_pressure_score__pressure"])
        direction = 1.0 if score >= 0.0 else -1.0
        rows.append(
            {
                "session": str(selected["session"]),
                "signed_gross_return_bps_30m": direction
                * float(selected["raw_continuation_return_bps"]),
            }
        )
    return pd.DataFrame(rows)


def verify_null(
    compact: pd.DataFrame,
    assessment: pd.DataFrame,
    configurations: Mapping[str, Any],
    artifact: pd.DataFrame,
    selections: pd.DataFrame,
) -> dict[str, Any]:
    development = (
        compact.loc[compact["year"].eq(2024) & compact["high_movement_admitted"].astype(bool)]
        .copy()
        .reset_index(drop=True)
    )
    primary = (
        assessment.loc[assessment["high_movement_admitted"].astype(bool)]
        .copy()
        .reset_index(drop=True)
    )
    pressure_features = [
        value
        for value in read_json(EXPERIMENT_DIR / "contract.json")["features"].get(
            "pressure_additions", []
        )
    ]
    if not pressure_features:
        pressure_features = (
            read_json(Path(configurations["feature_manifest_path"]))["pressure_onset_additions"]
            if "feature_manifest_path" in configurations
            else []
        )
    if not pressure_features:
        pressure_features = [
            name
            for name in configurations["models"]["A2"]
            if name not in configurations["models"]["A1"]
        ]
    a2_features = configurations["models"]["A2"]
    d2_features = configurations["models"]["D2"]
    readiness_mean = float(
        selections.loc[
            selections["candidate"].eq("readiness"), "signed_gross_return_bps_30m"
        ].mean()
    )
    expected: dict[str, list[float]] = {
        "A2_minus_A1_brier_improvement": [],
        "D2_minus_D1_brier_improvement": [],
        "pressure_minus_readiness_economic_30m": [],
    }
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    for draw in range(50):
        null_development = permute_bundle(development, pressure_features, seed=NULL_SEED + draw)
        null_assessment = permute_bundle(
            primary, pressure_features, seed=NULL_SEED + draw + 100_000
        )
        means_a, scales_a, model_a = fixed_fit(
            null_development,
            null_development["directional_onset"],
            a2_features,
        )
        direction_development = null_development.loc[null_development["directional_onset"].eq(1)]
        means_d, scales_d, model_d = fixed_fit(
            direction_development,
            direction_development["up_given_onset"],
            d2_features,
        )
        p_a = fixed_predict(null_assessment, a2_features, means_a, scales_a, model_a)
        p_d = fixed_predict(null_assessment, d2_features, means_d, scales_d, model_d)
        expected["A2_minus_A1_brier_improvement"].append(
            brier_improvement(
                null_assessment["directional_onset"],
                null_assessment["p_onset__A1"].to_numpy(dtype=float),
                p_a,
            )
        )
        null_direction = null_assessment["directional_onset"].eq(1)
        expected["D2_minus_D1_brier_improvement"].append(
            brier_improvement(
                null_assessment.loc[null_direction, "up_given_onset"],
                null_assessment.loc[null_direction, "p_up_given_onset__D1"].to_numpy(dtype=float),
                p_d[null_direction.to_numpy()],
            )
        )
        null_assessment["signed_pressure_score__pressure"] = (
            p_a
            * (2.0 * p_d - 1.0)
            * null_assessment["p_large_remaining_move"].to_numpy(dtype=float)
        )
        null_selection = pressure_only_selection(null_assessment)
        expected["pressure_minus_readiness_economic_30m"].append(
            float(null_selection["signed_gross_return_bps_30m"].mean()) - readiness_mean
        )
    errors: list[float] = []
    for metric, values in expected.items():
        draws = artifact.loc[
            artifact["record_type"].eq("draw") & artifact["metric"].eq(metric)
        ].sort_values("draw")
        errors.append(maximum_error(draws["null_value"], values))
        summary = artifact.loc[
            artifact["record_type"].eq("summary") & artifact["metric"].eq(metric)
        ].iloc[0]
        array = np.asarray(values, dtype=float)
        errors.extend(
            [
                abs(float(summary.null_q90) - float(np.quantile(array, 0.90))),
                abs(
                    float(summary.real_percentile)
                    - float(np.mean(array < float(summary.real_value)))
                ),
            ]
        )
    require(max(errors) <= 1e-8, "within-slate permutation null differs")
    return {
        "draws_verified": 50,
        "bundle_feature_count": len(pressure_features),
        "maximum_null_error": max(errors),
    }


def verify_decision(
    decision: Mapping[str, Any], primary: pd.DataFrame, selections: pd.DataFrame
) -> dict[str, Any]:
    support = decision["support"]
    labels = primary["onset_label"].value_counts().to_dict()
    require(int(support["rows"]) == len(primary), "decision support rows differ")
    require(int(support["sessions"]) == primary["session"].nunique(), "support sessions differ")
    require(int(support["stocks"]) == primary["symbol"].nunique(), "support stocks differ")
    require(int(support["up_onsets"]) == int(labels.get("UP_ONSET", 0)), "UP support differs")
    require(int(support["down_onsets"]) == int(labels.get("DOWN_ONSET", 0)), "DOWN support differs")
    require(int(support["no_onsets"]) == int(labels.get("NO_ONSET", 0)), "NO support differs")
    evidence = decision["evidence"]
    require(
        bool(evidence["occurrence_passes"]) == all(decision["occurrence_gates"].values()),
        "occurrence gate aggregation differs",
    )
    expected_direction = bool(
        support["conditional_direction_support_passes"]
        and all(decision["direction_gates"].values())
    )
    require(
        bool(evidence["direction_passes"]) == expected_direction,
        "direction gate aggregation differs",
    )
    expected_confirmation_occurrence = bool(
        not evidence["occurrence_passes"]
        and all(decision["confirmation_occurrence_gates"].values())
    )
    expected_confirmation_direction = bool(
        not evidence["direction_passes"]
        and support["conditional_direction_support_passes"]
        and all(decision["confirmation_direction_gates"].values())
    )
    require(
        bool(evidence["confirmation_occurrence_passes"]) == expected_confirmation_occurrence,
        "confirmation occurrence aggregation differs",
    )
    require(
        bool(evidence["confirmation_direction_passes"]) == expected_confirmation_direction,
        "confirmation direction aggregation differs",
    )
    if evidence["occurrence_passes"] and evidence["direction_passes"]:
        expected = "pressure_onset_and_direction_increment_observed"
    elif evidence["occurrence_passes"]:
        expected = "pressure_onset_occurrence_only"
    elif evidence["direction_passes"]:
        expected = "directional_pressure_only"
    elif evidence["confirmation_occurrence_passes"] or evidence["confirmation_direction_passes"]:
        expected = "one_bar_confirmation_required"
    elif evidence["readiness_useful"]:
        expected = "movement_readiness_but_direction_unresolved"
    else:
        expected = "no_pressure_onset_increment"
    require(decision["decision"] == expected, "final decision precedence differs")
    selected_shares = (
        selections.groupby("candidate", sort=True)["symbol"]
        .value_counts(normalize=True)
        .groupby(level=0)
        .max()
    )
    return {
        "decision_reconstructed": expected,
        "support_reconstructed": True,
        "maximum_selected_share": float(selected_shares.max()),
    }


def recursive_strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [item for nested in value.values() for item in recursive_strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in recursive_strings(nested)]
    return [value] if isinstance(value, str) else []


def verify_blocked_result(
    artifacts: Path,
    panel: pd.DataFrame,
    assessment: pd.DataFrame,
    decision: Mapping[str, Any],
    coefficients: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently verify a support-gated stop without running forbidden later stages."""

    blocker = "blocked_insufficient_pressure_onset_support"
    require(decision["decision"] == blocker, "blocked decision category differs")
    require(coefficients["status"] == "not_fitted_due_to_support_gate", "model status differs")
    require(coefficients["models"] == {}, "pressure models were fit after support failure")
    require(
        not any(column.startswith("p_onset__") for column in assessment.columns),
        "onset predictions exist after support failure",
    )
    require(
        assessment["screen_status"].astype(str).eq(blocker).all(),
        "assessment blocked status differs",
    )
    require(panel["screen_status"].astype(str).eq(blocker).all(), "panel blocked status differs")
    m1_error = maximum_error(
        assessment["p_large_remaining_move"],
        manual_probability(coefficients["frozen_predecessor_M1"], assessment),
    )
    require(m1_error <= 1e-12, "frozen M1 manual reconstruction differs")
    primary = assessment.loc[assessment["high_movement_admitted"].astype(bool)]
    labels = primary["onset_label"].value_counts().to_dict()
    row_shares = primary["symbol"].value_counts(normalize=True).sort_index()
    support = decision["support"]
    reconstructed = {
        "rows": len(primary),
        "sessions": int(primary["session"].nunique()),
        "stocks": int(primary["symbol"].nunique()),
        "months": int(primary["year_month"].nunique()),
        "slates": int(primary["slate_id"].nunique()),
        "directional_onset_rows": int(primary["directional_onset"].sum()),
        "up_onsets": int(labels.get("UP_ONSET", 0)),
        "down_onsets": int(labels.get("DOWN_ONSET", 0)),
        "no_onsets": int(labels.get("NO_ONSET", 0)),
        "maximum_stock_row_share": float(row_shares.max()),
        "minimum_high_movement_candidates_per_slate": int(
            primary.groupby("slate_id", sort=True)["symbol"].size().min()
        ),
        "minimum_valid_source_stocks_per_evaluated_slate": int(primary["source_slate_size"].min()),
    }
    for key, value in reconstructed.items():
        stored = support[key]
        if isinstance(value, float):
            require(abs(float(stored) - value) <= 1e-15, f"support differs: {key}")
        else:
            require(int(stored) == value, f"support differs: {key}")
    require(reconstructed["rows"] >= 1_200, "row support unexpectedly fails")
    require(reconstructed["sessions"] >= 100, "session support unexpectedly fails")
    require(reconstructed["stocks"] >= 15, "stock support unexpectedly fails")
    require(reconstructed["directional_onset_rows"] >= 250, "onset support unexpectedly fails")
    require(reconstructed["up_onsets"] >= 100, "UP support unexpectedly fails")
    require(reconstructed["down_onsets"] >= 100, "DOWN support unexpectedly fails")
    require(reconstructed["months"] >= 6, "month support unexpectedly fails")
    require(
        reconstructed["minimum_valid_source_stocks_per_evaluated_slate"] >= 15,
        "source-valid slate support unexpectedly fails",
    )
    require(
        reconstructed["minimum_high_movement_candidates_per_slate"] < 10,
        "declared high-movement per-slate support blocker is absent",
    )
    require(
        reconstructed["maximum_stock_row_share"] > 0.10,
        "declared concentration blocker is absent",
    )
    expected_failed_gates = [
        "high_movement_candidates_per_slate_at_least_10",
        "maximum_stock_row_share_at_most_0_10",
    ]
    require(
        support["failed_primary_support_gates"] == expected_failed_gates,
        "stored failed support-gate list differs",
    )
    concentration = pd.read_csv(artifacts / "concentration_metrics.csv")
    stored_shares = concentration.set_index("symbol")["share"].sort_index()
    require(
        maximum_error(stored_shares, row_shares) <= 1e-12,
        "blocked concentration ledger differs",
    )
    require(not concentration["passes"].astype(bool).all(), "concentration ledger passes")
    for name in (
        "onset_metrics.csv",
        "direction_metrics.csv",
        "monthly_metrics.csv",
        "checkpoint_metrics.csv",
        "calibration_bins.csv",
        "bootstrap_metrics.csv",
        "null_metrics.csv",
        "economic_reference_metrics.csv",
    ):
        status = pd.read_csv(artifacts / name)
        require(len(status) == 1, f"blocked status row count differs: {name}")
        require(
            status.iloc[0]["status"] == "not_run_due_to_support_gate",
            f"blocked status differs: {name}",
        )
        require(status.iloc[0]["decision"] == blocker, f"blocked code differs: {name}")
    configurations = read_json(artifacts / "model_configurations.json")
    require(
        configurations["status"] == "not_fitted_due_to_support_gate",
        "model configuration blocked status differs",
    )
    require(
        int(configurations["model_specification_count"]) == 8, "model specification count differs"
    )
    return {
        "decision_reconstructed": blocker,
        "failed_primary_support_gates": expected_failed_gates,
        "minimum_high_movement_candidates_per_slate": reconstructed[
            "minimum_high_movement_candidates_per_slate"
        ],
        "maximum_stock_row_share": reconstructed["maximum_stock_row_share"],
        "frozen_M1_maximum_prediction_error": m1_error,
        "later_stages_not_run": True,
        "support": reconstructed,
    }


def run_audit(artifacts: Path, provider_root: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED if not (artifacts / name).is_file()]
    require(not missing, f"required artifacts missing: {missing}")
    contract = read_json(artifacts / "contract.json")
    decision = read_json(artifacts / "decision.json")
    for key, expected in SAFETY_FLAGS.items():
        require(contract.get(key) == expected, f"contract safety differs: {key}")
        require(contract.get("safety", {}).get(key) == expected, f"nested safety differs: {key}")
        require(decision.get(key) == expected, f"decision safety differs: {key}")
    rerun = read_json(artifacts / "exact_rerun_manifest.json")
    require(bool(rerun["passed"]), "exact rerun was not successful")
    require(bool(rerun["comparisons"]), "exact rerun comparison list is empty")
    require(
        all(bool(row["passed"]) for row in rerun["comparisons"]),
        "an exact rerun artifact comparison failed",
    )
    input_hashes = read_json(artifacts / "input_artifact_hashes.json")
    for record in input_hashes["artifacts"]:
        path = REPO_ROOT / str(record["logical_path"])
        require(path.is_file(), f"input artifact missing: {record['logical_path']}")
        require(sha256_file(path) == record["sha256"], "input artifact hash differs")
    protected = read_json(artifacts / "protected_boundary_audit.json")
    require(int(protected["protected_rows_materialised"]) == 0, "protected rows materialised")
    require(protected["protected_files_touched"] == [], "protected files touched")
    source_manifest = read_json(artifacts / "source_manifest.json")
    source_result = verify_sources(source_manifest, provider_root)
    require(
        pd.Timestamp(source_result["maximum_timestamp_read"]) < PROTECTED_START,
        "source maximum reaches protected boundary",
    )
    panel = pd.read_parquet(artifacts / "compact_decision_panel.parquet")
    ledger = pd.read_parquet(artifacts / "onset_path_ledger.parquet")
    oof = pd.read_parquet(artifacts / "development_oof_predictions.parquet")
    assessment = pd.read_parquet(artifacts / "assessment_predictions.parquet")
    require(len(panel) <= 20_000, "compact row limit exceeded")
    forbidden_columns = sorted(
        name for name in panel.columns if any(fragment in name.lower() for fragment in FORBIDDEN)
    )
    require(not forbidden_columns, f"forbidden compact columns: {forbidden_columns}")
    require(
        pd.to_datetime(panel["decision_available_timestamp"], utc=True).lt(PROTECTED_START).all(),
        "protected compact timestamp",
    )
    windows = verify_windows_and_paths(panel, ledger)
    feature_manifest = read_json(artifacts / "feature_manifest.json")
    features = verify_features(panel, feature_manifest)
    readiness = verify_readiness_from_bounded_bars(panel, provider_root)
    thresholds = verify_thresholds_and_labels(
        panel,
        oof,
        read_json(artifacts / "movement_admission_thresholds.json"),
        read_json(artifacts / "onset_barriers.json"),
        read_json(artifacts / "movement_oof_fold_manifest.json"),
    )
    predecessor = read_json(artifacts / "predecessor_reconstruction.json")
    require(bool(predecessor["passed"]), "predecessor reconstruction did not pass")
    require(
        float(predecessor["maximum_prediction_absolute_error"]) <= 1e-12,
        "predecessor probability mismatch",
    )
    coefficients = read_json(artifacts / "model_coefficients.json")
    if decision["decision"] == "blocked_insufficient_pressure_onset_support":
        blocked = verify_blocked_result(artifacts, panel, assessment, decision, coefficients)
        json_artifacts = [
            read_json(path)
            for path in artifacts.glob("*.json")
            if path.name != "independent_audit.json"
        ]
        absolute_strings = [
            value
            for payload in json_artifacts
            for value in recursive_strings(payload)
            if value.startswith("/Users/") or value.startswith(str(REPO_ROOT))
        ]
        require(not absolute_strings, "local absolute path found in JSON artifact")
        return {
            **SAFETY_FLAGS,
            "auditor_imported_experiment_runner": False,
            "passed": True,
            "checks": {
                "safety_flags": True,
                "source_hashes_and_counts": source_result,
                "protected_boundary": True,
                "absence_of_forbidden_columns": True,
                "fixed_windows_and_cohort_paths": windows,
                "chronology_thresholds_barriers_and_labels": thresholds,
                "readiness_from_bounded_causal_bars": readiness,
                "feature_formulas_before_support_stop": features,
                "predecessor_reconstruction": True,
                "support_stop_and_non_execution": blocked,
                "exact_rerun_manifest": True,
                "local_absolute_paths_absent": True,
            },
        }
    models = verify_models(assessment, coefficients)
    metric_frames = [
        pd.read_csv(artifacts / "onset_metrics.csv", dtype={"scope_value": str}),
        pd.read_csv(artifacts / "direction_metrics.csv", dtype={"scope_value": str}),
        pd.read_csv(artifacts / "monthly_metrics.csv", dtype={"scope_value": str}),
        pd.read_csv(artifacts / "checkpoint_metrics.csv", dtype={"scope_value": str}),
    ]
    calibration = pd.read_csv(artifacts / "calibration_bins.csv", dtype={"scope_value": str})
    metrics = verify_metrics(assessment, metric_frames, calibration)
    economic_artifact = pd.read_csv(artifacts / "economic_reference_metrics.csv")
    concentration_artifact = pd.read_csv(artifacts / "concentration_metrics.csv")
    selections, economic = verify_economic(assessment, economic_artifact, concentration_artifact)
    primary = assessment.loc[assessment["high_movement_admitted"].astype(bool)]
    bootstrap = verify_bootstrap(
        primary,
        selections,
        pd.read_csv(artifacts / "bootstrap_metrics.csv"),
    )
    configurations = read_json(artifacts / "model_configurations.json")
    null = verify_null(
        panel,
        assessment,
        configurations,
        pd.read_csv(artifacts / "null_metrics.csv"),
        selections,
    )
    decision_result = verify_decision(decision, primary, selections)
    json_artifacts = [
        read_json(path)
        for path in artifacts.glob("*.json")
        if path.name != "independent_audit.json"
    ]
    absolute_strings = [
        value
        for payload in json_artifacts
        for value in recursive_strings(payload)
        if value.startswith("/Users/") or value.startswith(str(REPO_ROOT))
    ]
    require(not absolute_strings, "local absolute path found in JSON artifact")
    return {
        **SAFETY_FLAGS,
        "auditor_imported_experiment_runner": False,
        "passed": True,
        "checks": {
            "safety_flags": True,
            "source_hashes_and_counts": source_result,
            "protected_boundary": True,
            "absence_of_forbidden_columns": True,
            "fixed_windows_and_cohort_paths": windows,
            "chronology_thresholds_barriers_and_labels": thresholds,
            "readiness_from_bounded_causal_bars": readiness,
            "feature_formulas_and_confirmation": features,
            "manual_model_predictions": models,
            "probability_metrics_and_calibration": metrics,
            "session_block_bootstrap": bootstrap,
            "within_slate_bundle_permutation": null,
            "economic_reference_and_concentration": economic,
            "support_and_decision_logic": decision_result,
            "exact_rerun_manifest": True,
            "local_absolute_paths_absent": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=(
            Path.home()
            / "StockerLocal"
            / "data"
            / "processed"
            / "source=eodhd"
            / "instrument_type=stock"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.artifacts / "independent_audit.json"
    try:
        result = run_audit(args.artifacts, args.provider_root)
        output_path.write_text(canonical_json(result), encoding="utf-8")
        return 0
    except Exception as exc:
        failure = {
            **SAFETY_FLAGS,
            "auditor_imported_experiment_runner": False,
            "passed": False,
            "failure": f"{type(exc).__name__}: {exc}",
        }
        output_path.write_text(canonical_json(failure), encoding="utf-8")
        print(failure["failure"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
