"""Independent audit for the historical-volume entry-onset ablation V2.

The auditor deliberately does not import the V2 runner.  It rebuilds provider
historical-volume features, uses mathematically equivalent but independently
ordered nested weights, refits every HGB fold, replays thresholds/onsets/price
controls, and verifies probability and paired event evidence.  A pass is an
artifact-integrity result only; 2024 was already post-outcome development.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import run_raw_ohlc_entry_onset_discovery_v1 as base
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier


WORK = Path(__file__).resolve().parent
RUNNER_PATH = WORK / "run_historical_volume_entry_onset_ablation_v2.py"
CONTRACT_PATH = WORK / "contracts/20260712-historical-volume-entry-onset-ablation-v2.json"
PRE_SCORE_PATH = (
    WORK / "contracts/20260712-historical-volume-entry-onset-ablation-v2-pre-score.json"
)
ROOT = Path("/private/tmp/stocker_historical_volume_entry_onset_ablation_v2_20260712")
ENVIRONMENT_ROOT = Path("/Users/michaelsalerno/StockerLocal")
CONTRACT_ID = "historical_volume_entry_onset_ablation_v2"
BASELINE = "price_hgb"
CANDIDATE = "price_historical_volume_hgb"
ALGORITHMS = (BASELINE, CANDIDATE)
PRICE_FEATURES = base.FULL_FEATURES
CLASS_COLUMNS = base.CLASS_COLUMNS
HORIZONS = base.HORIZONS
PREDICTION_MONTHS = base.PREDICTION_MONTHS
VALIDATION_MONTHS = base.VALIDATION_MONTHS
SYMBOLS = base.SYMBOLS
FIRE_QUANTILE = 0.95
REARM_QUANTILE = 0.75
SEED = 20260712


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def independent_weights(frame: pd.DataFrame) -> np.ndarray:
    pairs = frame[["symbol_norm", "session_date"]].astype(str)
    sessions = (
        pairs.groupby("symbol_norm", sort=False)["session_date"]
        .transform("nunique")
        .to_numpy(float)
    )
    rows = (
        pairs.groupby(["symbol_norm", "session_date"], sort=False)["session_date"]
        .transform("size")
        .to_numpy(float)
    )
    # Intentionally independent operation order from the runner's literal
    # 1 / (sessions * rows), to expose numerical fragility.
    raw = (1.0 / sessions) / rows
    return raw / raw.mean()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    x = np.asarray(values, float)[order]
    w = np.asarray(weights, float)[order]
    index = int(np.searchsorted(np.cumsum(w), q * w.sum(), side="left"))
    return float(x[min(index, len(x) - 1)])


def group_transform(values: pd.Series, tape: pd.DataFrame, function: Any) -> pd.Series:
    return values.groupby(
        [tape["symbol_norm"], tape["session_date"], tape["segment_index"]], sort=False
    ).transform(function)


def load_independent_surface(
    volume_features: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tape, tape_diagnostics = base.load_tape(return_diagnostics=True)
    parts: list[pd.DataFrame] = []
    for symbol in SYMBOLS:
        frame = pd.read_parquet(
            base.clean_slate.provider_path(symbol),
            columns=["timestamp", "volume"],
            filters=base.clean_slate._year_filter(),
        ).rename(columns={"volume": "historical_volume"})
        frame["symbol_norm"] = symbol
        parts.append(frame)
    volume = pd.concat(parts, ignore_index=True)
    tape = tape.merge(
        volume,
        on=["symbol_norm", "timestamp"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    raw = tape["historical_volume"].to_numpy(float)
    valid = np.isfinite(raw) & (raw >= 0.0)
    tape.loc[~valid, "historical_volume"] = np.nan
    coverage = {
        "accepted_price_valid_regular_rows": len(tape),
        "finite_positive_historical_volume_rows": int((valid & (raw > 0.0)).sum()),
        "missing_or_nonfinite_historical_volume_rows": int((~np.isfinite(raw)).sum()),
        "zero_historical_volume_rows": int((valid & (raw == 0.0)).sum()),
        "negative_historical_volume_rows": int((np.isfinite(raw) & (raw < 0.0)).sum()),
    }
    surface = base.build_feature_surface(tape)
    lv = pd.Series(np.where(valid, np.log1p(raw), np.nan), index=tape.index, dtype=float)
    prior_1 = group_transform(lv, tape, lambda s: s.shift(1))
    prior_3 = group_transform(
        lv, tape, lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )
    prior_6 = group_transform(
        lv, tape, lambda s: s.shift(1).rolling(6, min_periods=6).mean()
    )
    prior_12 = group_transform(
        lv, tape, lambda s: s.shift(1).rolling(12, min_periods=12).mean()
    )
    segment_prior = group_transform(
        lv, tape, lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    recent_3 = group_transform(lv, tape, lambda s: s.rolling(3, min_periods=3).mean())
    older_12 = group_transform(
        lv, tape, lambda s: s.shift(3).rolling(12, min_periods=12).mean()
    )
    prior_12_std = group_transform(
        lv, tape, lambda s: s.shift(1).rolling(12, min_periods=12).std(ddof=0)
    )
    availability = group_transform(
        lv, tape, lambda s: s.rolling(12, min_periods=1).count() / 12.0
    )
    surface["historical_volume_missing_current"] = (~valid).astype(float)
    surface["historical_volume_log_change_1"] = lv - prior_1
    surface["historical_volume_log_ratio_prior_3"] = lv - prior_3
    surface["historical_volume_log_ratio_prior_6"] = lv - prior_6
    surface["historical_volume_log_ratio_prior_12"] = lv - prior_12
    surface["historical_volume_log_ratio_segment_prior_mean"] = lv - segment_prior
    surface["historical_volume_recent_3_minus_older_12"] = recent_3 - older_12
    surface["historical_volume_prior_12_log_std"] = prior_12_std
    surface["historical_volume_availability_12"] = availability
    surface["historical_volume_range_interaction_6"] = (
        surface["historical_volume_log_ratio_prior_6"] * surface["log_range_ratio_6"]
    )
    surface["historical_volume_body_interaction_6"] = (
        surface["historical_volume_log_ratio_prior_6"] * surface["signed_body_fraction"]
    )
    if tuple(feature for feature in volume_features if feature not in surface) != ():
        raise AssertionError("independent volume feature absent")
    return tape, surface, {"coverage": coverage, "price_tape": tape_diagnostics}


def build_anchor(surface: pd.DataFrame, horizon: int, volume_features: tuple[str, ...]) -> pd.DataFrame:
    anchor = base.build_anchor_surface(surface, horizon)
    return anchor.merge(
        surface[["source_position", *volume_features]],
        on="source_position",
        how="left",
        sort=False,
        validate="one_to_one",
    )


def replay_probabilities(
    tape: pd.DataFrame,
    surface: pd.DataFrame,
    volume_features: tuple[str, ...],
    hgb_parameters: dict[str, Any],
) -> tuple[pd.DataFrame, dict[tuple[str, int, str], dict[str, Any]]]:
    features_by_algorithm = {
        BASELINE: PRICE_FEATURES,
        CANDIDATE: (*PRICE_FEATURES, *volume_features),
    }
    parts: list[pd.DataFrame] = []
    bundles: dict[tuple[str, int, str], dict[str, Any]] = {}
    for horizon in HORIZONS:
        anchors = build_anchor(surface, horizon, volume_features)
        cached = pd.DataFrame()
        cached_ids: set[str] = set()
        for month in PREDICTION_MONTHS:
            train_mask, score_mask = base.fold_masks(anchors, month)
            train = anchors.loc[train_mask].copy()
            score = anchors.loc[score_mask].copy()
            missing = train.loc[~train["anchor_id"].isin(cached_ids)]
            if not missing.empty:
                added = base.attach_path_labels(tape, missing, horizon)
                cached = pd.concat([cached, added], ignore_index=True)
                cached_ids.update(added["anchor_id"].astype(str))
            target = train["anchor_id"].map(cached.set_index("anchor_id")["target_class"])
            y = target.to_numpy(np.int8)
            weights = independent_weights(train)
            for algorithm in ALGORITHMS:
                features = features_by_algorithm[algorithm]
                train_raw = train.loc[:, features].to_numpy(float)
                score_raw = score.loc[:, features].to_numpy(float)
                medians = np.nanmedian(train_raw, axis=0)
                train_values = np.where(np.isnan(train_raw), medians, train_raw)
                score_values = np.where(np.isnan(score_raw), medians, score_raw)
                estimator = HistGradientBoostingClassifier(**hgb_parameters)
                estimator.fit(train_values, y, sample_weight=weights)
                probabilities = estimator.predict_proba(score_values)
                output = score[
                    [
                        "anchor_id",
                        "symbol_norm",
                        "session_date",
                        "decision_timestamp",
                        "bar_ordinal",
                        "segment_index",
                        "segment_position",
                        "causal_scale_bps",
                        "availability_12",
                        "clock_bin_15",
                        "clock_bin_30",
                        "availability_bucket",
                    ]
                ].copy()
                output["fold_month"] = month
                output["algorithm"] = algorithm
                output["horizon"] = np.int16(horizon)
                for index, column in enumerate(CLASS_COLUMNS):
                    output[column] = probabilities[:, index]
                parts.append(output.loc[:, base.PRE_OUTCOME_LEDGER_COLUMNS])
                bundles[(month, horizon, algorithm)] = {
                    "estimator": estimator,
                    "features": features,
                    "medians": medians,
                }
    ledger = (
        pd.concat(parts, ignore_index=True)
        .sort_values(
            ["algorithm", "horizon", "symbol_norm", "session_date", "decision_timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return ledger, bundles


def replay_thresholds(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for score_month in VALIDATION_MONTHS:
        source_month = f"2024-{int(score_month[-2:]) - 1:02d}"
        for algorithm in ALGORITHMS:
            for horizon in HORIZONS:
                source = ledger.loc[
                    ledger["fold_month"].eq(source_month)
                    & ledger["algorithm"].eq(algorithm)
                    & ledger["horizon"].eq(horizon)
                ]
                weights = independent_weights(source)
                for side, column in (("long", "p_long_first"), ("short", "p_short_first")):
                    values = source[column].to_numpy(float)
                    rows.append(
                        {
                            "score_month": score_month,
                            "threshold_source_month": source_month,
                            "algorithm": algorithm,
                            "horizon": horizon,
                            "side": side,
                            "fire_threshold": weighted_quantile(values, weights, FIRE_QUANTILE),
                            "rearm_threshold": weighted_quantile(values, weights, REARM_QUANTILE),
                            "source_rows": len(source),
                            "fire_quantile": FIRE_QUANTILE,
                            "rearm_quantile": REARM_QUANTILE,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["algorithm", "horizon", "score_month", "side"], kind="stable"
    ).reset_index(drop=True)


def replay_onset_ids(ledger: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    candidate = ledger.loc[
        ledger["algorithm"].eq(CANDIDATE) & ledger["fold_month"].isin(VALIDATION_MONTHS)
    ]
    index = thresholds.set_index(["score_month", "algorithm", "horizon", "side"])
    rows: list[dict[str, Any]] = []
    keys = ["algorithm", "horizon", "symbol_norm", "session_date", "segment_index"]
    for key, group in candidate.groupby(keys, sort=False):
        algorithm, horizon, *_ = key
        long_armed = short_armed = True
        for row in group.sort_values("decision_timestamp", kind="stable").itertuples(index=False):
            long_threshold = index.loc[(row.fold_month, algorithm, horizon, "long")]
            short_threshold = index.loc[(row.fold_month, algorithm, horizon, "short")]
            if not long_armed and row.p_long_first < long_threshold["rearm_threshold"]:
                long_armed = True
            if not short_armed and row.p_short_first < short_threshold["rearm_threshold"]:
                short_armed = True
            long_fire = (
                long_armed
                and row.p_long_first >= long_threshold["fire_threshold"]
                and row.p_long_first > row.p_short_first
            )
            short_fire = (
                short_armed
                and row.p_short_first >= short_threshold["fire_threshold"]
                and row.p_short_first > row.p_long_first
            )
            side = None
            if long_fire and not short_fire:
                side = "long"
                long_armed = False
            elif short_fire and not long_fire:
                side = "short"
                short_armed = False
            if side:
                rows.append(
                    {
                        "onset_id": f"{algorithm}|{row.anchor_id}|{side}",
                        "anchor_id": row.anchor_id,
                        "horizon": int(horizon),
                        "side": side,
                        "fold_month": row.fold_month,
                        "symbol_norm": row.symbol_norm,
                        "session_date": row.session_date,
                        "decision_timestamp": row.decision_timestamp,
                        "clock_bin_15": int(row.clock_bin_15),
                        "clock_bin_30": int(row.clock_bin_30),
                        "availability_bucket": int(row.availability_bucket),
                    }
                )
    return pd.DataFrame(rows).sort_values("onset_id", kind="stable").reset_index(drop=True)


def replay_control_ids(onsets: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    price = ledger.loc[
        ledger["algorithm"].eq(BASELINE) & ledger["fold_month"].isin(VALIDATION_MONTHS)
    ]
    rows: list[dict[str, Any]] = []
    for (horizon, side), candidates in onsets.groupby(["horizon", "side"], sort=False):
        column = "p_long_first" if side == "long" else "p_short_first"
        pools = {
            key: group.sort_values([column, "anchor_id"], ascending=[False, True], kind="stable")
            for key, group in price.loc[price["horizon"].eq(horizon)].groupby(
                ["symbol_norm", "fold_month"], sort=False
            )
        }
        used: set[str] = set()
        for candidate in candidates.sort_values("onset_id", kind="stable").itertuples(index=False):
            pool = pools[(candidate.symbol_norm, candidate.fold_month)]
            base_mask = ~pool["anchor_id"].eq(candidate.anchor_id) & ~pool["anchor_id"].isin(used)
            masks = [
                base_mask
                & ~pool["session_date"].eq(candidate.session_date)
                & pool["clock_bin_15"].eq(candidate.clock_bin_15)
                & pool["availability_bucket"].eq(candidate.availability_bucket),
                base_mask
                & pool["clock_bin_30"].eq(candidate.clock_bin_30)
                & pool["availability_bucket"].eq(candidate.availability_bucket),
                base_mask & pool["clock_bin_30"].eq(candidate.clock_bin_30),
                base_mask,
            ]
            chosen = None
            tier = -1
            for tier_index, mask in enumerate(masks):
                eligible = pool.loc[mask]
                if not eligible.empty:
                    chosen = eligible.iloc[0]
                    tier = tier_index
                    break
            if chosen is None:
                rows.append({"onset_id": candidate.onset_id, "control_anchor_id": None, "match_tier": -1})
            else:
                control_id = str(chosen["anchor_id"])
                used.add(control_id)
                rows.append(
                    {
                        "onset_id": candidate.onset_id,
                        "control_anchor_id": control_id,
                        "match_tier": tier,
                    }
                )
    return pd.DataFrame(rows).sort_values("onset_id", kind="stable").reset_index(drop=True)


def compare_probability_metrics(
    predictions: pd.DataFrame, paths: pd.DataFrame, stored: pd.DataFrame
) -> float:
    validation = predictions.loc[predictions["fold_month"].isin(VALIDATION_MONTHS)].merge(
        paths[["anchor_id", "target_class"]], on="anchor_id", validate="many_to_one"
    )
    rows: list[dict[str, Any]] = []
    for (algorithm, horizon), group in validation.groupby(["algorithm", "horizon"], sort=False):
        weights = independent_weights(group)
        probability = group.loc[:, CLASS_COLUMNS].to_numpy(float)
        target = group["target_class"].to_numpy(int)
        chosen = probability[np.arange(len(group)), target]
        one_hot = np.eye(3)[target]
        rows.append(
            {
                "algorithm": algorithm,
                "horizon": horizon,
                "multiclass_log_loss": float(np.average(-np.log(chosen), weights=weights)),
                "multiclass_brier": float(
                    np.average(np.sum((probability - one_hot) ** 2, axis=1), weights=weights)
                ),
            }
        )
    observed = pd.DataFrame(rows).sort_values(["algorithm", "horizon"]).reset_index(drop=True)
    expected = stored.loc[stored["slice_type"].eq("pooled"), observed.columns].sort_values(
        ["algorithm", "horizon"]
    ).reset_index(drop=True)
    return float(
        np.max(
            np.abs(
                observed[["multiclass_log_loss", "multiclass_brier"]].to_numpy(float)
                - expected[["multiclass_log_loss", "multiclass_brier"]].to_numpy(float)
            )
        )
    )


def main() -> None:
    checks: list[dict[str, Any]] = []

    def record(name: str, evidence: Any) -> None:
        checks.append({"name": name, "passed": True, "evidence": safe(evidence)})

    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    assert contract["contract_id"] == CONTRACT_ID
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert contract["sources"]["provider_volume_label"] == "historical_volume"
    assert contract["interpretation"]["no_order_flow_claim"] is True
    record("contract_safety_and_volume_provenance", "historical_volume_not_order_flow")

    tree = ast.parse(RUNNER_PATH.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("broker" in name or "order" in name for name in imported)
    assert "run_historical_volume_entry_onset_ablation_v2" not in imported
    record("static_research_boundary", sorted(imported))

    source_hashes = dict(pre_score["sha256"])
    actual = {
        "contract": sha256(CONTRACT_PATH),
        "runner": sha256(RUNNER_PATH),
        "frozen_entry_onset_v1_runner": sha256(
            WORK / "run_raw_ohlc_entry_onset_discovery_v1.py"
        ),
        "frozen_clean_slate_runner": sha256(
            WORK / "run_clean_slate_causal_ohlc_entries_v1.py"
        ),
        "environment_pyproject": sha256(ENVIRONMENT_ROOT / "pyproject.toml"),
        "environment_uv_lock": sha256(ENVIRONMENT_ROOT / "uv.lock"),
    }
    for symbol in SYMBOLS:
        actual[f"provider_full_file_{symbol}"] = sha256(base.clean_slate.provider_path(symbol))
    assert actual == source_hashes
    assert pre_score["environment_versions"] == {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    record("frozen_source_and_environment_hashes", len(actual))

    manifest_path = ROOT / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["stage"] == "pre_independent_audit_complete_artifact_manifest"
    for name, expected_hash in manifest["files_excluding_this_manifest"].items():
        assert sha256(ROOT / name) == expected_hash
    record("pre_audit_artifact_manifest", len(manifest["files_excluding_this_manifest"]))

    volume_features = tuple(contract["historical_volume_features_in_order"])
    tape, surface, diagnostics = load_independent_surface(volume_features)
    expected_coverage = contract["target_blind_volume_coverage"]
    for key in (
        "accepted_price_valid_regular_rows",
        "finite_positive_historical_volume_rows",
        "missing_or_nonfinite_historical_volume_rows",
        "zero_historical_volume_rows",
        "negative_historical_volume_rows",
    ):
        assert diagnostics["coverage"][key] == expected_coverage[key]
    assert not np.isinf(surface.loc[:, volume_features].to_numpy(float)).any()
    record("independent_volume_coverage_and_features", diagnostics["coverage"])

    replay, _bundles = replay_probabilities(
        tape, surface, volume_features, contract["shared_estimator_parameters"]
    )
    stored_predictions = pd.read_parquet(ROOT / "probabilities_pre_evaluation.parquet")
    assert replay[["anchor_id", "algorithm"]].equals(
        stored_predictions[["anchor_id", "algorithm"]]
    )
    probability_delta = float(
        np.max(
            np.abs(
                replay.loc[:, CLASS_COLUMNS].to_numpy(float)
                - stored_predictions.loc[:, CLASS_COLUMNS].to_numpy(float)
            )
        )
    )
    assert probability_delta <= 1e-12
    record("all_fold_probability_replay_with_equivalent_weights", probability_delta)

    replayed_thresholds = replay_thresholds(replay)
    stored_thresholds = pd.read_csv(ROOT / "thresholds_pre_evaluation.csv")
    threshold_delta = float(
        np.max(
            np.abs(
                replayed_thresholds[["fire_threshold", "rearm_threshold"]].to_numpy(float)
                - stored_thresholds[["fire_threshold", "rearm_threshold"]].to_numpy(float)
            )
        )
    )
    assert threshold_delta <= 1e-12
    record("prior_month_threshold_replay", threshold_delta)

    onsets = replay_onset_ids(replay, replayed_thresholds)
    stored_onsets = pd.read_parquet(ROOT / "candidate_onsets_pre_evaluation.parquet")
    assert onsets["onset_id"].tolist() == stored_onsets["onset_id"].tolist()
    record("hysteresis_onset_replay", len(onsets))

    replayed_controls = replay_control_ids(onsets, replay)
    stored_controls = pd.read_parquet(ROOT / "matched_price_controls_pre_evaluation.parquet")
    expected_controls = stored_controls[
        ["onset_id", "control_anchor_id", "match_tier"]
    ].sort_values("onset_id", kind="stable").reset_index(drop=True)
    assert replayed_controls.equals(expected_controls)
    record("outcome_blind_matched_price_control_replay", len(replayed_controls))

    paths = base.build_validation_paths(tape, surface)
    stored_paths = pd.read_parquet(ROOT / "validation_paths.parquet")
    assert paths["anchor_id"].tolist() == stored_paths["anchor_id"].tolist()
    assert paths["target_class"].equals(stored_paths["target_class"])
    record("exact_validation_path_replay", len(paths))

    metric_delta = compare_probability_metrics(
        replay,
        paths,
        pd.read_csv(ROOT / "probability_metrics.csv"),
    )
    assert metric_delta <= 1e-12
    record("independent_pooled_probability_metrics", metric_delta)

    decision = json.loads((ROOT / "decision.json").read_text())
    assert decision["probability_hypothesis"]["retained"] is True
    assert decision["any_incremental_historical_volume_entry_sign_retained"] is False
    assert decision["volume_is_order_flow"] is False
    record(
        "decision_consistency",
        {
            "probability_retained": True,
            "entry_sign_retained": False,
        },
    )

    sensitivity = pd.read_parquet(
        ROOT / "historical_volume_sensitivities_pre_evaluation.parquet"
    )
    forbidden = ("target", "status", "confirmation", "mfe", "return", "pnl", "profit")
    assert not any(any(token in column.lower() for token in forbidden) for column in sensitivity)
    assert len(sensitivity) == len(stored_onsets)
    record("pre_evaluation_volume_sensitivity_boundary", len(sensitivity))

    audit = {
        "contract_id": CONTRACT_ID,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume",
        "volume_is_order_flow": False,
        "prior_price_only_outcomes_known_before_volume_contract": True,
        "audit_passed": True,
        "checks_passed": len(checks),
        "checks": checks,
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "interpretation": "artifact integrity passed; 2024 remains post-outcome development and no entry sign was retained",
    }
    write_json(ROOT / "independent_audit.json", audit)
    files = sorted(path for path in ROOT.iterdir() if path.is_file() and path.name != "artifact_manifest.json")
    write_json(
        manifest_path,
        {
            **{key: value for key, value in manifest.items() if key != "files_excluding_this_manifest"},
            "stage": "independent_audit_complete_artifact_manifest",
            "files_excluding_this_manifest": {path.name: sha256(path) for path in files},
        },
    )
    print(json.dumps(safe(audit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
