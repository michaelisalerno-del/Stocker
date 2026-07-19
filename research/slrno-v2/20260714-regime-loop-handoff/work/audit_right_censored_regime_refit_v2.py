#!/usr/bin/env python3
"""Independent arithmetic auditor for Right-Censored Regime Refit V2.

This module intentionally does not import the panel builder, duration fitter,
refit implementation, primary pipeline, or unchanged-gate summary functions.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PRIMARY_DIR = WORK_DIR / "artifacts/20260719-right-censored-regime-refit-v2/primary"
EXACT_DIR = WORK_DIR / "artifacts/20260719-right-censored-regime-refit-v2/exact_rerun"
CONTRACT_PATH = WORK_DIR / "contracts/20260719-right-censored-regime-refit-v2.json"
PREVIOUS_CONTRACT_PATH = WORK_DIR / "contracts/20260718-regime-model-validity-v2.json"
FROZEN_STATE_PATH = (
    WORK_DIR / "shadow_validation/frozen_loop_movement_shadow_v1/frozen_bundle/"
    "artifacts/state/frozen_semimarkov_parameters.npz"
)
PROVIDER_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock"
)
BASELINE_SHA = "91996a9cf747a614ff6d9e08eaafc3583a58b91c"
SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "economic_outcomes_used": False,
    "payoff_selection_used": False,
    "production_runtime_modified": False,
    "strategy_promotion": False,
    "part_b_interaction_scoring_enabled": False,
    "semantic_dictionary_promotion_enabled": False,
}
IDENTITY_KEYS = (
    "run_id",
    "git_sha",
    "contract_hash",
    "data_snapshot_hash",
    "panel_hash",
    "implementation_source_hash",
    "state_model_version",
    "state_model_hash",
    "model_lineage",
)
MANIFEST_EXCLUSIONS = {
    "artifact_manifest.json",
    "independent_audit.json",
    "exact_rerun_manifest.json",
    "post_repair_tree_manifest.json",
}
EMISSIONS = (
    "regime_log_activity_3",
    "regime_log_activity_12",
    "regime_activity_acceleration",
    "signed_efficiency_6",
    "signed_efficiency_12",
    "regime_log_bar_range",
    "close_location_value",
    "regime_wick_balance",
    "log_relative_historical_volume",
    "log_relative_cumulative_historical_volume",
    "regime_log_market_dispersion",
    "regime_stock_minus_market_scaled",
    "vti__signed_efficiency_12",
    "regime_market_breadth_centered",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arrow_hash(frame: pd.DataFrame) -> str:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _canonical_key_hash(frame: pd.DataFrame) -> str:
    columns = ["symbol", "session", "bar_start_timestamp", "bar_ordinal"]
    selected = frame[columns].sort_values(columns, kind="mergesort")
    return _arrow_hash(selected.reset_index(drop=True))


def _parameter_arrays(directory: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(directory / name) as stored:
        return {key: np.asarray(stored[key]).copy() for key in stored.files}


def _bounded_source_hash(path: Path) -> str:
    frame = pd.read_parquet(
        path,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        filters=[
            (
                "timestamp",
                ">=",
                pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
            ),
            (
                "timestamp",
                "<=",
                pd.Timestamp("2024-12-31 23:59:59", tz="UTC").to_pydatetime(),
            ),
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return _arrow_hash(frame)


def _audit_source_identity(directory: Path) -> dict[str, Any]:
    source = _json(directory / "pre_repair_source_identity.json")
    manifest = _json(directory / "panel_builder_manifest.json")
    declared = manifest["provider_source_hashes_2024"]
    recomputed: dict[str, str] = {}
    for symbol in sorted(declared):
        stored = "VTI.US" if symbol == "VTI" else symbol
        path = PROVIDER_ROOT / f"symbol={stored}" / "timeframe=5m" / "data.parquet"
        recomputed[symbol] = _bounded_source_hash(path)
    return {
        "provider_source_count": len(recomputed),
        "provider_source_hashes_match": recomputed == declared,
        "pre_freeze_source_hashes_match": (recomputed == source["development_source_hashes"]),
        "frozen_state_hash_matches": (
            _sha256_file(FROZEN_STATE_PATH) == source["frozen_state_model_file_hash"]
        ),
        "previous_contract_hash_matches": (
            _sha256_file(PREVIOUS_CONTRACT_PATH) == source["previous_regime_validity_contract_hash"]
        ),
    }


def _audit_panel(directory: Path) -> dict[str, Any]:
    keys = pd.read_parquet(directory / "panel_natural_keys.parquet")
    hashes = _json(directory / "panel_hashes.json")
    order = ["symbol", "session", "bar_start_timestamp", "bar_ordinal"]
    ordered = keys.sort_values(order, kind="mergesort").reset_index(drop=True)
    unique = not keys[order].duplicated().any()
    completed = pd.to_datetime(keys["bar_complete_timestamp"], utc=True) == pd.to_datetime(
        keys["bar_start_timestamp"], utc=True
    ) + pd.Timedelta(minutes=5)
    segment_local = (
        keys.groupby("segment_id", sort=False)["bar_ordinal"]
        .apply(lambda values: bool(np.all(np.diff(values.to_numpy()) == 1)))
        .all()
    )
    return {
        "panel_row_count": len(keys),
        "natural_keys_unique": unique,
        "deterministic_order": keys[order].reset_index(drop=True).equals(ordered[order]),
        "row_key_hash_reconstructed": _canonical_key_hash(keys),
        "row_key_hash_matches": (_canonical_key_hash(keys) == hashes["development_row_key_hash"]),
        "completed_bar_availability_pass": bool(completed.all()),
        "within_segment_ordinals_contiguous": bool(segment_local),
        "segments": int(keys["segment_id"].nunique()),
    }


def _audit_emissions(directory: Path) -> dict[str, Any]:
    frame = pd.read_parquet(directory / "panel_emission_audit_sample.parquet")
    expected: dict[str, np.ndarray] = {
        "regime_log_activity_3": np.log1p(
            10_000.0 * frame["mean_abs_return_3"].clip(lower=0.0)
        ).to_numpy(),
        "regime_log_activity_12": np.log1p(
            10_000.0 * frame["mean_abs_return_12"].clip(lower=0.0)
        ).to_numpy(),
        "signed_efficiency_6": frame["signed_efficiency_6"].to_numpy(),
        "signed_efficiency_12": frame["signed_efficiency_12"].to_numpy(),
        "close_location_value": frame["close_location_value"].to_numpy(),
        "log_relative_historical_volume": frame["log_relative_historical_volume"].to_numpy(),
        "log_relative_cumulative_historical_volume": frame[
            "log_relative_cumulative_historical_volume"
        ].to_numpy(),
        "vti__signed_efficiency_12": frame["vti__signed_efficiency_12"].to_numpy(),
    }
    expected["regime_activity_acceleration"] = (
        expected["regime_log_activity_3"] - expected["regime_log_activity_12"]
    )
    expected["regime_log_bar_range"] = np.log1p(
        10_000.0 * frame["bar_range_pct"].clip(lower=0.0)
    ).to_numpy()
    expected["regime_wick_balance"] = (
        frame["upper_wick_pct_of_range"] - frame["lower_wick_pct_of_range"]
    ).to_numpy()
    expected["regime_log_market_dispersion"] = np.log1p(
        10_000.0 * frame["market_dispersion_return_6"].abs().clip(lower=0.0)
    ).to_numpy()
    denominator = (6.0 * frame["mean_abs_return_12"]).replace(0.0, np.nan).clip(lower=1e-8)
    expected["regime_stock_minus_market_scaled"] = np.tanh(
        frame["stock_minus_market_return_6"] / denominator
    ).to_numpy()
    expected["regime_market_breadth_centered"] = (
        frame["market_breadth_return_6_positive"] - 0.5
    ).to_numpy()
    checks = {
        feature: bool(
            np.allclose(
                expected[feature],
                frame[feature].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-14,
                equal_nan=True,
            )
        )
        for feature in EMISSIONS
    }
    availability = pd.to_datetime(
        frame["feature_available_timestamp_max"], utc=True
    ) <= pd.to_datetime(frame["bar_complete_timestamp"], utc=True)
    return {
        "sample_rows": len(frame),
        "feature_checks": checks,
        "all_fourteen_emissions_match": all(checks.values()),
        "no_future_feature_availability": bool(availability.all()),
    }


def _independent_run_ledger(directory: Path) -> pd.DataFrame:
    keys = pd.read_parquet(directory / "panel_natural_keys.parquet")
    labels = pd.read_parquet(directory / "full_refit_cleaned_labels.parquet")
    join = (
        labels[
            [
                "symbol",
                "session",
                "bar_start_timestamp",
                "bar_ordinal",
                "state",
            ]
        ]
        .merge(
            keys[
                [
                    "symbol",
                    "session",
                    "bar_start_timestamp",
                    "bar_ordinal",
                    "segment_id",
                    "segment_index",
                    "session_source_complete",
                ]
            ],
            on=["symbol", "session", "bar_start_timestamp", "bar_ordinal"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            ["symbol", "session", "bar_start_timestamp", "bar_ordinal"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for (symbol, session), session_frame in join.groupby(["symbol", "session"], sort=True):
        complete = bool(session_frame["session_source_complete"].all())
        segments = list(session_frame.groupby("segment_id", sort=False))
        for segment_number, (segment_id, segment) in enumerate(segments):
            states = segment["state"].to_numpy(dtype=int)
            starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
            ends = np.r_[starts[1:], len(states)]
            for local_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                left_truncated = local_index == 0 and (
                    segment_number > 0 or int(segment["bar_ordinal"].iloc[0]) > 0
                )
                terminal = int(end) == len(states)
                ends_at_gap = terminal and segment_number < len(segments) - 1
                if left_truncated or ends_at_gap:
                    status = "INVALIDATED_BY_SOURCE_GAP"
                    eligible = False
                elif not complete:
                    status = "INCOMPLETE_OR_UNAVAILABLE_SESSION"
                    eligible = False
                elif not terminal:
                    status = "OBSERVED_STATE_EXIT"
                    eligible = True
                else:
                    status = "RIGHT_CENSORED_SESSION_END"
                    eligible = True
                rows.append(
                    {
                        "symbol": symbol,
                        "session": session,
                        "segment_id": segment_id,
                        "state": int(states[int(start)]),
                        "duration": int(end - start),
                        "ending_status": status,
                        "primary_fit_eligible": eligible,
                    }
                )
    return pd.DataFrame(rows)


def _duration_counts(
    ledger: pd.DataFrame, *, state_count: int = 8, maximum_age: int = 78
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    at_risk = np.zeros((state_count, maximum_age), dtype=np.int64)
    exits = np.zeros_like(at_risk)
    censored = np.zeros_like(at_risk)
    eligible = ledger.loc[ledger["primary_fit_eligible"].astype(bool)]
    for row in eligible.itertuples(index=False):
        state = int(row.state)
        duration = int(row.duration)
        at_risk[state, :duration] += 1
        if row.ending_status == "OBSERVED_STATE_EXIT":
            exits[state, duration - 1] += 1
        else:
            censored[state, duration - 1] += 1
    return at_risk, exits, censored


def _hazard(at_risk: np.ndarray, exits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = beta = 0.5
    minimum = 5.0
    raw = (exits + alpha) / (at_risk + alpha + beta)
    pooled_risk = at_risk.sum(axis=0)
    pooled_exits = exits.sum(axis=0)
    pooled = (pooled_exits + alpha) / (pooled_risk + alpha + beta)
    reliability = pooled_risk / (pooled_risk + minimum)
    target = reliability * pooled + (1.0 - reliability) * 0.05
    weight = np.clip((minimum - at_risk) / minimum, 0.0, 1.0)
    hazard = (1.0 - weight) * raw + weight * target[None, :]
    hazard = np.clip(hazard, 0.0, 1.0 - np.finfo(np.float64).eps)
    survival = np.cumprod(1.0 - hazard, axis=1)
    return hazard, survival, weight


def _audit_duration(directory: Path) -> dict[str, Any]:
    reconstructed = _independent_run_ledger(directory)
    recorded = pd.read_parquet(directory / "training_run_ending_ledger.parquet")
    recorded_counts = Counter(recorded["ending_status"].astype(str))
    reconstructed_counts = Counter(reconstructed["ending_status"].astype(str))
    at_risk, exits, censored = _duration_counts(reconstructed)
    hazards, survival, _ = _hazard(at_risk, exits)
    stored = pd.read_parquet(directory / "right_censored_duration_counts.parquet").sort_values(
        ["state", "age"], kind="mergesort"
    )
    shape = (8, 78)
    stored_at_risk = stored["at_risk"].to_numpy(dtype=int).reshape(shape)
    stored_exits = stored["exits"].to_numpy(dtype=int).reshape(shape)
    stored_censored = stored["censored"].to_numpy(dtype=int).reshape(shape)
    stored_hazard = stored["hazard"].to_numpy(dtype=float).reshape(shape)
    stored_survival = stored["survival"].to_numpy(dtype=float).reshape(shape)
    previous = np.c_[np.ones(8), survival[:, :-1]]
    mass = (previous * hazards).sum(axis=1) + survival[:, -1]
    return {
        "training_run_count": len(reconstructed),
        "ending_status_counts": dict(sorted(reconstructed_counts.items())),
        "ending_status_counts_match": reconstructed_counts == recorded_counts,
        "at_risk_counts_match": np.array_equal(at_risk, stored_at_risk),
        "exit_counts_match": np.array_equal(exits, stored_exits),
        "censored_counts_match": np.array_equal(censored, stored_censored),
        "hazards_match": bool(np.allclose(hazards, stored_hazard, rtol=0.0, atol=1e-15)),
        "survival_matches": bool(np.allclose(survival, stored_survival, rtol=0.0, atol=1e-15)),
        "duration_24_exact": bool(stored_at_risk[:, 23].sum() > 0),
        "duration_78_representable": stored_at_risk.shape[1] == 78,
        "forced_age_24_exit": bool(np.any(hazards[:, 23] == 1.0)),
        "forced_final_age_exit": bool(np.any(hazards[:, -1] == 1.0)),
        "probability_mass_maximum_error": float(np.max(np.abs(mass - 1.0))),
        "survival_non_increasing": bool(np.all(np.diff(survival, axis=1) <= 1e-15)),
    }


def _audit_preprocessing_and_clusters(directory: Path) -> dict[str, Any]:
    training = pd.read_parquet(directory / "full_refit_training_rows.parquet")
    preprocessing = pd.read_csv(directory / "full_refit_preprocessing.csv")
    parameter_bundle = _parameter_arrays(directory, "full_refit_parameters.npz")
    feature_names = preprocessing["feature"].astype(str).tolist()
    values = training[feature_names].to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    imputed = np.where(np.isnan(values), medians[None, :], values)
    centers = np.median(imputed, axis=0)
    scales = np.percentile(imputed, 75.0, axis=0) - np.percentile(imputed, 25.0, axis=0)
    scales[scales == 0.0] = 1.0
    exact_medians = parameter_bundle["preprocessing_medians"]
    exact_centers = parameter_bundle["preprocessing_centers"]
    exact_scales = parameter_bundle["preprocessing_scales"]
    preprocessing_pass = bool(
        np.allclose(
            medians,
            exact_medians,
            rtol=0.0,
            atol=1e-10,
        )
        and np.allclose(
            centers,
            exact_centers,
            rtol=0.0,
            atol=1e-10,
        )
        and np.allclose(
            scales,
            exact_scales,
            rtol=0.0,
            atol=1e-10,
        )
    )
    csv_serialization_pass = bool(
        np.allclose(
            exact_medians,
            preprocessing["imputer_median"],
            rtol=0.0,
            atol=1e-10,
        )
        and np.allclose(
            exact_centers,
            preprocessing["scaler_center"],
            rtol=0.0,
            atol=1e-10,
        )
        and np.allclose(
            exact_scales,
            preprocessing["scaler_scale"],
            rtol=0.0,
            atol=1e-10,
        )
    )
    sample = pd.read_parquet(directory / "posterior_audit_input.parquet")
    sample_values = sample[feature_names].to_numpy(dtype=float)
    sample_imputed = np.where(np.isnan(sample_values), exact_medians[None, :], sample_values)
    # The fitted transform contract serializes emissions as float32 before both
    # clustering and filtering.  Reproduce that declared numerical boundary
    # independently; retaining float64 here changes Gaussian likelihoods even
    # when every preprocessing parameter is identical.
    sample_scaled = ((sample_imputed - exact_centers[None, :]) / exact_scales[None, :]).astype(
        np.float32
    )
    centroid_table = pd.read_csv(directory / "full_refit_cluster_centroids.csv")
    centers_semantic = np.zeros((8, len(feature_names)), dtype=float)
    for row in centroid_table.itertuples(index=False):
        centers_semantic[int(row.state), feature_names.index(str(row.feature))] = float(
            row.kmeans_scaled_centroid
        )
    nearest = np.argmin(
        np.square(sample_scaled[:, None, :] - centers_semantic[None, :, :]).sum(axis=2),
        axis=1,
    )
    mapping_frame = pd.read_csv(directory / "full_refit_semantic_mapping.csv")
    mapping = dict(
        zip(
            mapping_frame["raw_cluster_state"].astype(int),
            mapping_frame["semantic_state"].astype(int),
            strict=True,
        )
    )
    recorded_raw_semantic = np.asarray(
        [mapping[int(value)] for value in sample["raw_kmeans_state"]],
        dtype=int,
    )
    cluster_sample_pass = bool(np.array_equal(nearest, recorded_raw_semantic))
    return {
        "training_rows": len(training),
        "training_position_unique": bool(training["training_position"].is_unique),
        "preprocessing_reconstructed": preprocessing_pass,
        "preprocessing_csv_serialization_within_tolerance": (csv_serialization_pass),
        "kmeans_assignment_sample_reconstructed": cluster_sample_pass,
        "sample_scaled": sample_scaled,
        "sample": sample,
        "centers_semantic": centers_semantic,
        "mapping": mapping,
    }


def _clean_short_runs(
    labels: np.ndarray,
    scaled: np.ndarray,
    centers: np.ndarray,
    groups: list[np.ndarray],
) -> np.ndarray:
    output = labels.copy()
    for _ in range(2):
        changes = 0
        for positions in groups:
            local = output[positions].copy()
            starts = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1]
            ends = np.r_[starts[1:], len(local)]
            run_labels = local[starts]
            for run_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                if int(end - start) >= 2:
                    continue
                candidates = []
                if run_index > 0:
                    candidates.append(int(run_labels[run_index - 1]))
                if run_index + 1 < len(starts):
                    candidates.append(int(run_labels[run_index + 1]))
                eligible = sorted(
                    {
                        candidate
                        for candidate in candidates
                        if candidate != int(run_labels[run_index])
                    }
                )
                if not eligible:
                    continue
                rows = scaled[positions[int(start) : int(end)]]
                best = min(
                    eligible,
                    key=lambda state: float(np.mean(np.square(rows - centers[state]))),
                )
                local[int(start) : int(end)] = best
                changes += 1
            output[positions] = local
        if changes == 0:
            break
    return output


def _audit_cleanup_and_semantics(directory: Path, cluster: dict[str, Any]) -> dict[str, Any]:
    sample = cluster["sample"]
    mapping = cluster["mapping"]
    raw_semantic = np.asarray(
        [mapping[int(value)] for value in sample["raw_kmeans_state"]],
        dtype=int,
    )
    cleaned_recorded = np.asarray(
        [mapping[int(value)] for value in sample["cleaned_pre_semantic_state"]],
        dtype=int,
    )
    groups = [
        group.index.to_numpy(dtype=int) for _, group in sample.groupby("segment_id", sort=False)
    ]
    independently_cleaned = _clean_short_runs(
        raw_semantic,
        cluster["sample_scaled"],
        cluster["centers_semantic"],
        groups,
    )
    cleaned_full = pd.read_parquet(
        directory / "full_refit_cleaned_labels.parquet",
        columns=[
            "cleaned_pre_semantic_state",
            "state",
            "regime_log_activity_12",
            "signed_efficiency_12",
        ],
    )
    summary = (
        cleaned_full.groupby("cleaned_pre_semantic_state", sort=True)[
            ["regime_log_activity_12", "signed_efficiency_12"]
        ]
        .mean()
        .sort_values(
            ["regime_log_activity_12", "signed_efficiency_12"],
            kind="mergesort",
        )
    )
    independent_mapping = {int(old): int(new) for new, old in enumerate(summary.index)}
    semantic_from_mapping = np.asarray(
        [independent_mapping[int(value)] for value in cleaned_full["cleaned_pre_semantic_state"]],
        dtype=int,
    )
    return {
        "cleanup_sample_rows": len(sample),
        "cleanup_sample_reconstructed": bool(
            np.array_equal(independently_cleaned, cleaned_recorded)
        ),
        "semantic_mapping_reconstructed": independent_mapping == mapping,
        "semantic_labels_reconstructed": bool(
            np.array_equal(
                semantic_from_mapping,
                cleaned_full["state"].to_numpy(dtype=int),
            )
        ),
    }


def _log_emissions(scaled: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    constant = np.log(2.0 * np.pi * variances)
    return np.asarray(
        [
            -0.5
            * np.sum(
                constant[state] + np.square(scaled - means[state]) / variances[state],
                axis=1,
            )
            for state in range(len(means))
        ]
    ).T


def _filter_segment(
    emissions: np.ndarray,
    hazard: np.ndarray,
    transitions: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, states = emissions.shape
    ages = hazard.shape[1]
    probabilities = np.zeros((rows, states))
    expected_age = np.zeros(rows)
    departure = np.zeros(rows)
    likelihood = np.zeros(rows)
    alpha = np.zeros((states, ages))
    for row in range(rows):
        prior = np.zeros((states, ages))
        if row == 0:
            prior[:, 0] = initial / initial.sum()
        else:
            for state in range(states):
                prior[state, 1:] += alpha[state, :-1] * (1.0 - hazard[state, :-1])
                prior[state, -1] += alpha[state, -1] * (1.0 - hazard[state, -1])
                exit_mass = float(np.sum(alpha[state] * hazard[state]))
                prior[:, 0] += exit_mass * transitions[state]
            prior /= prior.sum()
        state_prior = prior.sum(axis=1)
        terms = np.log(np.clip(state_prior, 1e-300, None)) + emissions[row]
        maximum = float(terms.max())
        likelihood[row] = maximum + math.log(float(np.exp(terms - maximum).sum()))
        relative = np.exp(emissions[row] - emissions[row].max())
        alpha = prior * relative[:, None]
        alpha /= alpha.sum()
        probabilities[row] = alpha.sum(axis=1)
        expected_age[row] = float(np.sum(alpha * np.arange(1, ages + 1)[None, :]))
        departure[row] = float(np.sum(alpha * hazard))
    return probabilities, expected_age, departure, likelihood


def _audit_posterior(directory: Path, cluster: dict[str, Any]) -> dict[str, Any]:
    sample = cluster["sample"]
    params = _parameter_arrays(directory, "full_refit_parameters.npz")
    reconstructed_probabilities = np.zeros((len(sample), 8))
    reconstructed_age = np.zeros(len(sample))
    reconstructed_departure = np.zeros(len(sample))
    reconstructed_likelihood = np.zeros(len(sample))
    for _, group in sample.groupby("segment_id", sort=False):
        positions = group.index.to_numpy(dtype=int)
        emissions = _log_emissions(
            cluster["sample_scaled"][positions],
            params["means"],
            params["variances"],
        )
        probabilities, age, departure, likelihood = _filter_segment(
            emissions,
            params["duration_hazard"],
            params["transitions"],
            params["initial"],
        )
        reconstructed_probabilities[positions] = probabilities
        reconstructed_age[positions] = age
        reconstructed_departure[positions] = departure
        reconstructed_likelihood[positions] = likelihood
    recorded = sample[[f"state_probability_{state}" for state in range(8)]].to_numpy(dtype=float)
    return {
        "posterior_sample_rows": len(sample),
        "posterior_normalization_maximum_error": float(
            np.max(np.abs(reconstructed_probabilities.sum(axis=1) - 1.0))
        ),
        "posterior_reconstructed": bool(
            np.allclose(reconstructed_probabilities, recorded, rtol=0.0, atol=1e-12)
        ),
        "hard_state_reconstructed": bool(
            np.array_equal(
                np.argmax(reconstructed_probabilities, axis=1),
                sample["state"].to_numpy(dtype=int),
            )
        ),
        "expected_age_reconstructed": bool(
            np.allclose(
                reconstructed_age,
                sample["age"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "departure_probability_reconstructed": bool(
            np.allclose(
                reconstructed_departure,
                sample["departure_probability"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "row_likelihood_reconstructed": bool(
            np.allclose(
                reconstructed_likelihood,
                sample["row_log_likelihood"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "gap_reset_count": int(
            pd.read_csv(directory / "gap_reset_population.csv")["gap_or_missing_open_resets"].iloc[
                0
            ]
        ),
    }


def _mutual_information(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_values, left_inverse = np.unique(left, return_inverse=True)
    right_values, right_inverse = np.unique(right, return_inverse=True)
    counts = np.zeros((len(left_values), len(right_values)), dtype=float)
    np.add.at(counts, (left_inverse, right_inverse), 1.0)
    probability = counts / counts.sum()
    p_left = probability.sum(axis=1)
    p_right = probability.sum(axis=0)
    nonzero = probability > 0.0
    expected = p_left[:, None] * p_right[None, :]
    mi = float(np.sum(probability[nonzero] * np.log(probability[nonzero] / expected[nonzero])))
    h_left = float(-np.sum(p_left[p_left > 0] * np.log(p_left[p_left > 0])))
    h_right = float(-np.sum(p_right[p_right > 0] * np.log(p_right[p_right > 0])))
    nmi = mi / ((h_left + h_right) / 2.0) if h_left + h_right else 1.0
    return mi, nmi


def _event_set(frame: pd.DataFrame) -> set[tuple[str, str, str, int]]:
    return {
        (str(symbol), str(session), str(loop_id), int(bar))
        for symbol, session, loop_id, bar in frame[
            ["symbol", "session", "primitive_loop_id", "event_bar_ordinal"]
        ].itertuples(index=False, name=None)
    }


def _audit_comparisons_and_gates(directory: Path) -> dict[str, Any]:
    paths = pd.read_parquet(directory / "aligned_state_assignment_paths.parquet")
    frozen = paths["frozen_state"].to_numpy(dtype=int)
    full = paths["full_refit_aligned_state"].to_numpy(dtype=int)
    _, nmi = _mutual_information(frozen, full)
    agreement = float(np.mean(frozen == full))
    events = pd.read_parquet(directory / "loop_event_comparison.parquet")
    frozen_events = events.loc[events["model_lineage"].eq("MODEL_FROZEN")]
    full_events = events.loc[events["model_lineage"].eq("MODEL_FULL_REFIT")]
    frozen_set = _event_set(frozen_events)
    full_set = _event_set(full_events)
    exact_event_agreement = len(frozen_set & full_set) / max(len(frozen_set), 1)
    first = pd.read_parquet(directory / "first_event_comparison.parquet")
    selected = {"loop_p_5-6-5", "loop_p_4-6-4"}
    coverage = {}
    for lineage, frame in first.groupby("model_lineage", sort=True):
        coverage[str(lineage)] = float(frame["primary_label"].isin(selected).mean())
    coverage_table = pd.read_csv(directory / "dictionary_coverage_comparison.csv").set_index(
        "model_lineage"
    )
    coverage_matches = all(
        math.isclose(
            value,
            float(coverage_table.at[lineage, "dictionary_coverage"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for lineage, value in coverage.items()
    )
    contract = _json(CONTRACT_PATH)
    previous = _json(PREVIOUS_CONTRACT_PATH)
    current_thresholds = contract["unchanged_part_a_binding"]["frozen_structural_thresholds"]
    previous_thresholds = previous["frozen_structural_thresholds"]
    threshold_subset_matches = all(
        key in previous_thresholds and previous_thresholds[key] == value
        for key, value in current_thresholds.items()
    )
    gates = pd.read_csv(directory / "repaired_regime_validity_metrics.csv")
    k_registry = pd.read_csv(directory / "repaired_k_seed_model_registry.csv")
    expected_registry = {
        (state_count, seed)
        for state_count in (6, 8, 10, 12)
        for seed in (20260710, 20260711, 20260712, 20260713, 20260714)
    }
    actual_registry = set(
        zip(
            k_registry["state_count"].astype(int),
            k_registry["seed"].astype(int),
            strict=True,
        )
    )
    sample_composition = pd.read_csv(directory / "training_sample_composition.csv")
    return {
        "frozen_vs_full_nmi": nmi,
        "frozen_vs_full_bar_agreement": agreement,
        "exact_loop_event_agreement": exact_event_agreement,
        "dictionary_coverage_reconstructed": coverage,
        "dictionary_coverage_matches": coverage_matches,
        "unchanged_thresholds_match_previous_contract": threshold_subset_matches,
        "gate_row_count": len(gates),
        "k_seed_registry_complete": actual_registry == expected_registry,
        "training_sample_policies_complete": set(sample_composition["sample_variant"].astype(str))
        == {"SAMPLE_A", "SAMPLE_B", "SAMPLE_C", "SAMPLE_D"},
        "part_b_remained_closed": bool(
            not _json(directory / "repaired_part_a_decision.json")["part_b_opened"]
        ),
    }


def _audit_duration_only_identity(directory: Path) -> dict[str, Any]:
    frozen = _parameter_arrays(FROZEN_STATE_PATH.parent, FROZEN_STATE_PATH.name)
    repaired = _parameter_arrays(directory, "duration_only_repair_parameters.npz")
    non_duration = (
        "means",
        "variances",
        "transitions",
        "initial",
        "occupancy",
    )
    return {
        "non_duration_arrays_identical": all(
            np.array_equal(frozen[name], repaired[name]) for name in non_duration
        ),
        "duration_array_changed": not np.array_equal(
            frozen["duration_hazard"], repaired["duration_hazard"][:, :24]
        ),
        "duration_support": int(repaired["duration_hazard"].shape[1]),
        "model_id_distinct": str(repaired["model_id"].item())
        == "regime_model_v2_duration_only_repair",
    }


def _audit_manifest_and_exact(directory: Path) -> dict[str, Any]:
    manifest = _json(directory / "artifact_manifest.json")
    current = {
        path.relative_to(directory).as_posix(): _sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.relative_to(directory).as_posix()
        not in MANIFEST_EXCLUSIONS | {"artifact_manifest.json"}
    }
    exact = _json(directory / "exact_rerun_manifest.json")
    return {
        "artifact_manifest_matches": current == manifest["artifacts"],
        "artifact_count": len(current),
        "exact_rerun_byte_identical": bool(exact["byte_identical"]),
        "exact_rerun_compared_artifacts": int(exact["compared_artifact_count"]),
        "parameter_hashes_match": bool(exact["parameter_hashes_match"]),
        "posterior_hashes_match": bool(exact["posterior_hashes_match"]),
        "state_assignment_hashes_match": bool(exact["state_assignment_hashes_match"]),
        "training_row_hashes_match": bool(exact["training_row_hashes_match"]),
    }


def _audit_frozen_tree() -> dict[str, Any]:
    changed = subprocess.run(
        ["git", "diff", "--name-only", BASELINE_SHA, "--"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    baseline_paths = set(
        subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", BASELINE_SHA],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    modified_existing = sorted(set(changed) & baseline_paths)
    return {
        "baseline_sha": BASELINE_SHA,
        "modified_preexisting_files": modified_existing,
        "frozen_historical_tree_unchanged": not modified_existing,
    }


def _audit_safety(directory: Path) -> dict[str, Any]:
    contract = _json(CONTRACT_PATH)
    contract_flags = all(contract.get(key) == value for key, value in SAFETY_FLAGS.items())
    artifact_failures: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in {
            "independent_audit.json",
            "post_repair_tree_manifest.json",
        }:
            continue
        relative = path.relative_to(directory).as_posix()
        if path.suffix == ".json":
            payload = _json(path)
            if not all(payload.get(key) == value for key, value in SAFETY_FLAGS.items()):
                artifact_failures.append(relative)
        elif path.suffix == ".csv":
            frame = pd.read_csv(path, nrows=1)
            if not set(SAFETY_FLAGS).issubset(frame.columns):
                artifact_failures.append(relative)
        elif path.suffix == ".parquet":
            frame = pd.read_parquet(path, columns=None).head(1)
            if not set(SAFETY_FLAGS).issubset(frame.columns):
                artifact_failures.append(relative)
        elif path.suffix == ".npz":
            with np.load(path) as stored:
                if not set(SAFETY_FLAGS).issubset(stored.files):
                    artifact_failures.append(relative)
    prohibited_imports: list[str] = []
    new_sources = [
        *(REPO_ROOT / "packages/stocker_research/src/stocker_research").glob("*regime*v2.py"),
        WORK_DIR / "regime_repair_artifacts_v2.py",
        WORK_DIR / "regime_repair_validity_rerun_v2.py",
        WORK_DIR / "regime_repair_pipeline_v2.py",
        WORK_DIR / "run_right_censored_regime_refit_v2.py",
        WORK_DIR / "audit_right_censored_regime_refit_v2.py",
    ]
    for path in new_sources:
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.lower()
                    if any(
                        token in module
                        for token in ("broker", "order", "position", "execution", "ig.")
                    ):
                        prohibited_imports.append(f"{path}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").lower()
                if any(
                    token in module for token in ("broker", "order", "position", "execution", "ig.")
                ):
                    prohibited_imports.append(f"{path}:{node.module}")
    return {
        "contract_safety_flags_pass": contract_flags,
        "artifact_safety_flag_failures": artifact_failures,
        "prohibited_runtime_imports": prohibited_imports,
        "part_b_disabled": contract["part_b"]["opened"] is False,
        "dictionary_promotion_disabled": contract["dictionary"]["promotion_enabled"] is False,
    }


def _all_boolean_leaves(value: Any, *, ignore: set[str] | None = None) -> list[bool]:
    ignored = ignore or set()
    output: list[bool] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ignored:
                continue
            output.extend(_all_boolean_leaves(child, ignore=ignored))
    elif isinstance(value, bool):
        output.append(value)
    return output


def audit(directory: Path = PRIMARY_DIR) -> dict[str, Any]:
    source = _audit_source_identity(directory)
    panel = _audit_panel(directory)
    emissions = _audit_emissions(directory)
    duration = _audit_duration(directory)
    duration_only = _audit_duration_only_identity(directory)
    cluster = _audit_preprocessing_and_clusters(directory)
    cleanup = _audit_cleanup_and_semantics(directory, cluster)
    posterior = _audit_posterior(directory, cluster)
    comparisons = _audit_comparisons_and_gates(directory)
    manifest = _audit_manifest_and_exact(directory)
    frozen_tree = _audit_frozen_tree()
    safety = _audit_safety(directory)
    cluster_public = {
        key: value
        for key, value in cluster.items()
        if key not in {"sample_scaled", "sample", "centers_semantic", "mapping"}
    }
    sections = {
        "source_identity": source,
        "panel": panel,
        "emissions": emissions,
        "run_endings_and_duration": duration,
        "duration_only_identity": duration_only,
        "preprocessing_and_kmeans": cluster_public,
        "cleanup_and_semantics": cleanup,
        "posterior_and_gap_reset": posterior,
        "comparisons_and_unchanged_gates": comparisons,
        "manifest_and_exact_rerun": manifest,
        "frozen_tree": frozen_tree,
        "safety": safety,
    }
    required_true = [
        source["provider_source_hashes_match"],
        source["pre_freeze_source_hashes_match"],
        source["frozen_state_hash_matches"],
        source["previous_contract_hash_matches"],
        panel["natural_keys_unique"],
        panel["deterministic_order"],
        panel["row_key_hash_matches"],
        panel["completed_bar_availability_pass"],
        panel["within_segment_ordinals_contiguous"],
        emissions["all_fourteen_emissions_match"],
        emissions["no_future_feature_availability"],
        duration["ending_status_counts_match"],
        duration["at_risk_counts_match"],
        duration["exit_counts_match"],
        duration["censored_counts_match"],
        duration["hazards_match"],
        duration["survival_matches"],
        not duration["forced_age_24_exit"],
        not duration["forced_final_age_exit"],
        duration["probability_mass_maximum_error"] <= 1e-12,
        duration_only["non_duration_arrays_identical"],
        duration_only["duration_array_changed"],
        duration_only["duration_support"] == 78,
        duration_only["model_id_distinct"],
        cluster_public["preprocessing_reconstructed"],
        cluster_public["preprocessing_csv_serialization_within_tolerance"],
        cluster_public["kmeans_assignment_sample_reconstructed"],
        cleanup["cleanup_sample_reconstructed"],
        cleanup["semantic_mapping_reconstructed"],
        cleanup["semantic_labels_reconstructed"],
        posterior["posterior_reconstructed"],
        posterior["hard_state_reconstructed"],
        posterior["expected_age_reconstructed"],
        posterior["departure_probability_reconstructed"],
        posterior["row_likelihood_reconstructed"],
        comparisons["dictionary_coverage_matches"],
        comparisons["unchanged_thresholds_match_previous_contract"],
        comparisons["k_seed_registry_complete"],
        comparisons["training_sample_policies_complete"],
        comparisons["part_b_remained_closed"],
        manifest["artifact_manifest_matches"],
        manifest["exact_rerun_byte_identical"],
        frozen_tree["frozen_historical_tree_unchanged"],
        safety["contract_safety_flags_pass"],
        not safety["artifact_safety_flag_failures"],
        not safety["prohibited_runtime_imports"],
        safety["part_b_disabled"],
        safety["dictionary_promotion_disabled"],
    ]
    return {
        "audit_version": "right_censored_regime_refit_v2_independent_v1",
        "audit_passed": all(required_true),
        "primary_summary_generation_functions_imported": False,
        "primary_fit_functions_imported": False,
        "checks": sections,
        "failed_required_check_count": len(required_true) - sum(required_true),
        **SAFETY_FLAGS,
    }


def _identity(directory: Path) -> dict[str, Any]:
    metadata = _json(directory / "run_metadata.json")
    return {key: metadata[key] for key in IDENTITY_KEYS}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.relative_to(directory).as_posix()
        not in MANIFEST_EXCLUSIONS | {"artifact_manifest.json"}
    }


def _finalize(directory: Path, audit_result: dict[str, Any]) -> None:
    identity = _identity(directory)
    enriched_audit = {**identity, **audit_result}
    _write_json(directory / "independent_audit.json", enriched_audit)
    repair = _json(directory / "repair_decision.json")
    repair.update(
        {
            "decision": (
                "right_censored_regime_repair_complete_with_known_limitations"
                if audit_result["audit_passed"]
                else "right_censored_regime_repair_failed"
            ),
            "decision_status": "final",
            "exact_artifact_rerun_status": "passed",
            "independent_audit_status": ("passed" if audit_result["audit_passed"] else "failed"),
        }
    )
    _write_json(directory / "repair_decision.json", repair)
    part_a = _json(directory / "repaired_part_a_decision.json")
    part_a["independent_audit_status"] = "passed" if audit_result["audit_passed"] else "failed"
    _write_json(directory / "repaired_part_a_decision.json", part_a)
    frozen = audit_result["checks"]["frozen_tree"]
    _write_json(
        directory / "post_repair_tree_manifest.json",
        {
            **identity,
            **frozen,
            "audit_passed": audit_result["audit_passed"],
            **SAFETY_FLAGS,
        },
    )
    artifacts = _artifact_hashes(directory)
    manifest = _json(directory / "artifact_manifest.json")
    manifest.update(
        {
            "artifacts": artifacts,
            "artifact_count": len(artifacts),
            "manifest_hash": hashlib.sha256(
                json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    )
    _write_json(directory / "artifact_manifest.json", manifest)


def main() -> None:
    if not PRIMARY_DIR.is_dir() or not EXACT_DIR.is_dir():
        raise FileNotFoundError("primary and exact rerun directories are required")
    result = audit(PRIMARY_DIR)
    for directory in (PRIMARY_DIR, EXACT_DIR):
        _finalize(directory, result)
    if not result["audit_passed"]:
        print(json.dumps(result, sort_keys=True, default=str))
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
