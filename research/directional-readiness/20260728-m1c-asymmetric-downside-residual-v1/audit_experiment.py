#!/usr/bin/env python3
"""Independently audit M1C Asymmetric Downside Residual V1 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXPERIMENT_DIR: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = EXPERIMENT_DIR.parents[2]
PRIMARY: Final[Path] = EXPERIMENT_DIR / "artifacts" / "primary"
TAIL_EPISODES: Final[Path] = (
    REPO_ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-tail-phase-v1"
    / "artifacts"
    / "primary"
    / "fresh_episode_results_v1.parquet"
)
STATE_PATH: Final[Path] = Path(
    "/Users/michaelsalerno/Documents/Codex/"
    "2026-07-23-you-are-working-in-the-github-5/data/cache/"
    "minimal-intraday-iv-excess-holdout-v0/frozen_state_surface.parquet"
)
HISTORICAL_OPTIONS_PATH: Final[Path] = Path(
    "/Users/michaelsalerno/Documents/Codex/"
    "2026-07-23-you-are-working-in-the-github-3/research/cross-market-context/"
    "20260723-daily-stock-front-options-context-v01/artifacts/primary/"
    "front_options_dimensions.parquet"
)
STRESS_OPTIONS_PATH: Final[Path] = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260724-minimal-intraday-iv-excess-holdout-v01"
    / "artifacts"
    / "primary"
    / "holdout_selected_option_pairs.parquet"
)
M1C_THRESHOLD: Final[float] = 0.488333710794033
PROTECTED_START: Final[str] = "2026-01-01"
ANNUAL_TRADING_MINUTES: Final[float] = 252.0 * 390.0
IDENTITY: Final[list[str]] = ["stock", "session", "checkpoint"]
FEATURES: Final[list[str]] = [
    "D1_signed_return_5m",
    "D2_signed_return_15m",
    "D3_close_location_15m",
    "D4_distance_from_session_vwap_iv",
]


class AuditFailure(AssertionError):
    """An artifact failed an independent reproducibility or safety check."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def read_opened(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[("session", "<", PROTECTED_START)])
    require(
        len(frame) == pq.ParquetFile(path).metadata.num_rows,
        f"protected-boundary filter excluded source rows: {path}",
    )
    require(
        bool(frame["session"].astype(str).lt(PROTECTED_START).all()),
        f"protected session materialised: {path}",
    )
    return frame


def reconstruct_targets(predictions: pd.DataFrame) -> dict[str, Any]:
    sessions = predictions["session"].astype(str)
    start = str(sessions.min())
    end = str(sessions.max())
    state = pd.read_parquet(
        STATE_PATH,
        columns=["symbol", "session", "bar_ordinal", "open", "close"],
        filters=[("session", ">=", start), ("session", "<=", end)],
    ).rename(columns={"symbol": "stock"})
    require(
        bool(state["session"].astype(str).lt(PROTECTED_START).all()),
        "protected completed bar materialised",
    )
    entry = state[["stock", "session", "bar_ordinal", "open"]].rename(
        columns={"bar_ordinal": "checkpoint", "open": "_audit_entry"}
    )
    terminal = state[["stock", "session", "bar_ordinal", "close"]].copy()
    terminal["checkpoint"] = terminal["bar_ordinal"].astype(int) - 2
    terminal = terminal.rename(columns={"close": "_audit_terminal"}).drop(columns="bar_ordinal")
    audit = predictions.merge(entry, on=IDENTITY, how="left", validate="one_to_one").merge(
        terminal,
        on=IDENTITY,
        how="left",
        validate="one_to_one",
    )
    signed = np.log(
        audit["_audit_terminal"].to_numpy(float) / audit["_audit_entry"].to_numpy(float)
    )
    signed_difference = float(
        np.max(np.abs(signed - audit["signed_endpoint_return_15m_v1"].to_numpy(float)))
    )
    require(signed_difference <= 1e-12, "independent endpoint-return audit failed")

    option_columns = ["symbol", "session", "atm_iv"]
    historical = pd.read_parquet(
        HISTORICAL_OPTIONS_PATH,
        columns=option_columns,
        filters=[
            ("session", ">=", start),
            ("session", "<=", min(end, "2025-08-22")),
        ],
    )
    option_frames = [historical]
    if end >= "2025-09-01":
        stress = pd.read_parquet(
            STRESS_OPTIONS_PATH,
            columns=[*option_columns, "pair_available"],
            filters=[("session", ">=", max(start, "2025-09-01")), ("session", "<=", end)],
        )
        stress = stress.loc[stress["pair_available"].astype(bool), option_columns]
        option_frames.append(stress)
    options = pd.concat(option_frames, ignore_index=True).rename(columns={"symbol": "stock"})
    require(
        bool(options["session"].astype(str).lt(PROTECTED_START).all()),
        "protected option context materialised",
    )
    require(
        not options.duplicated(["stock", "session"]).any(),
        "option context is not unique",
    )
    audit = audit.merge(options, on=["stock", "session"], how="left", validate="many_to_one")
    implied = (
        audit["atm_iv"].to_numpy(float)
        * math.sqrt(15.0 / ANNUAL_TRADING_MINUTES)
        * math.sqrt(2.0 / math.pi)
    )
    implied_difference = float(
        np.max(np.abs(implied - audit["iv_expected_absolute_15m"].to_numpy(float)))
    )
    require(implied_difference <= 1e-12, "independent IV denominator audit failed")
    reconstructed_state = np.where(
        signed >= implied,
        "UP_MOVE",
        np.where(signed <= -implied, "DOWN_MOVE", "NO_MOVE"),
    )
    require(
        np.array_equal(
            reconstructed_state,
            audit["primary_outcome_state_v1"].astype(str).to_numpy(),
        ),
        "independent endpoint partition audit failed",
    )
    strict_move = np.abs(signed) > implied
    require(
        np.array_equal(
            strict_move,
            audit["future_15m_exceed_iv_v1"].astype(bool).to_numpy(),
        ),
        "independent strict M1C target audit failed",
    )
    return {
        "rows": int(len(audit)),
        "maximum_signed_return_difference": signed_difference,
        "maximum_implied_denominator_difference": implied_difference,
        "up": int(np.sum(reconstructed_state == "UP_MOVE")),
        "down": int(np.sum(reconstructed_state == "DOWN_MOVE")),
        "no_move": int(np.sum(reconstructed_state == "NO_MOVE")),
        "exact_equality": int(np.sum(np.abs(signed) == implied)),
        "passed": True,
    }


def audit_scores_actions(predictions: pd.DataFrame) -> dict[str, Any]:
    model = cast(
        dict[str, Any],
        json.loads((PRIMARY / "final_model_parameters_v1.json").read_text(encoding="utf-8")),
    )
    scaling = cast(
        dict[str, Any],
        json.loads((PRIMARY / "standardisation_parameters_v1.json").read_text(encoding="utf-8")),
    )
    thresholds = cast(
        dict[str, Any],
        json.loads((PRIMARY / "frozen_action_thresholds_v1.json").read_text(encoding="utf-8")),
    )
    means = np.asarray([scaling["means"][feature] for feature in FEATURES], dtype=float)
    scales = np.asarray([scaling["scales"][feature] for feature in FEATURES], dtype=float)
    coefficients = np.asarray(
        [model["coefficients_standardised"][feature] for feature in FEATURES],
        dtype=float,
    )
    values = predictions.loc[:, FEATURES].to_numpy(float)
    logits = ((values - means) / scales) @ coefficients + float(model["intercept"])
    score = 1.0 / (1.0 + np.exp(-logits))
    score_difference = float(np.max(np.abs(score - predictions["q_down_v1"].to_numpy(float))))
    require(score_difference <= 1e-12, "stored model does not reproduce q_down")
    expected_action = np.where(
        score <= float(thresholds["low"]),
        "CALL",
        np.where(score >= float(thresholds["high"]), "PUT", "ABSTAIN"),
    )
    require(
        np.array_equal(
            expected_action,
            predictions["asymmetric_action_v1"].astype(str).to_numpy(),
        ),
        "stored thresholds do not reproduce actions",
    )
    oof = pd.read_parquet(PRIMARY / "development_oof_predictions_v1.parquet")
    low, high = np.quantile(
        oof["q_down_oof"].to_numpy(float),
        [0.2, 0.8],
        method="linear",
    )
    require(float(low) == float(thresholds["low"]), "stored low threshold is not OOF q20")
    require(float(high) == float(thresholds["high"]), "stored high threshold is not OOF q80")
    require(
        bool(pd.to_datetime(oof["session"], utc=True).dt.year.eq(2024).all()),
        "OOF threshold source contains a non-2024 row",
    )
    return {
        "rows": int(len(predictions)),
        "maximum_probability_difference": score_difference,
        "low_threshold": float(low),
        "high_threshold": float(high),
        "oof_rows": int(len(oof)),
        "passed": True,
    }


def audit_frozen_fields(predictions: pd.DataFrame) -> dict[str, Any]:
    source = pd.read_parquet(
        TAIL_EPISODES,
        filters=[
            ("session", ">=", str(predictions["session"].min())),
            ("session", "<=", str(predictions["session"].max())),
        ],
    )
    require(
        bool(source["session"].astype(str).lt(PROTECTED_START).all()),
        "protected inherited episode materialised",
    )
    fields = [
        "episode_id",
        "existing_fresh_episode_identifier",
        "M1C_probability",
        "m1c_high_tail_v1",
        "m1c_tail_phase_v1",
        "movement_consumed_v1",
        "A1_probability_up_v1",
        "A1_action_v1",
    ]
    merged = source[IDENTITY + fields].merge(
        predictions[IDENTITY + fields],
        on=IDENTITY,
        suffixes=("_source", "_experiment"),
        validate="one_to_one",
    )
    require(len(merged) == len(predictions), "fresh episode identities changed")
    for field in fields:
        left = merged[f"{field}_source"]
        right = merged[f"{field}_experiment"]
        if pd.api.types.is_numeric_dtype(left):
            equal = np.allclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            equal = left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str))
        require(bool(equal), f"frozen inherited field changed: {field}")
    require(
        bool(predictions["M1C_probability"].ge(M1C_THRESHOLD).all()),
        "prediction includes a row below the exact frozen M1C gate",
    )
    require(
        bool(predictions["phase_at_trigger_v1"].isin(["FIRST_ENTRY", "RE_ENTRY"]).all()),
        "persistent checkpoint entered primary fresh-episode support",
    )
    return {"rows": int(len(merged)), "fields": fields, "passed": True}


def audit_safety_and_artifacts(predictions: pd.DataFrame) -> dict[str, Any]:
    summary = cast(
        dict[str, Any],
        json.loads((PRIMARY / "summary_v1.json").read_text(encoding="utf-8")),
    )
    provenance = cast(
        dict[str, Any],
        json.loads((PRIMARY / "provenance_manifest_v1.json").read_text(encoding="utf-8")),
    )
    require(not summary["exact_probability_decomposition_supported"], "mismatch hidden")
    require(not summary["joint_probabilities_constructed"], "joint probabilities claimed")
    require(
        summary["primary_decision"] == "target_mismatch_prevents_exact_probability_decomposition",
        "target mismatch decision drifted",
    )
    require(
        summary["directional_diagnostic_decision"] == "blocked_insufficient_support",
        "formal endpoint decision ignored the frozen action-support minimum",
    )
    require(
        summary["descriptive_directional_finding"] == "low_downside_does_not_imply_upside",
        "descriptive endpoint finding drifted",
    )
    require(
        not any("joint" in column.lower() for column in predictions),
        "joint probability column emitted despite mismatch",
    )
    forbidden = ("signed_pressure", "tension", "peer_slate")
    require(
        not any(any(token in column.lower() for token in forbidden) for column in predictions),
        "contaminated field present in predictions",
    )
    feature_manifest = cast(
        dict[str, Any],
        json.loads((PRIMARY / "feature_manifest_v1.json").read_text(encoding="utf-8")),
    )
    require(
        feature_manifest["ordered_model_columns"] == FEATURES,
        "model feature manifest is not exactly D1-D4",
    )
    protected = cast(dict[str, Any], provenance["protected_data_confirmation"])
    execution = cast(dict[str, Any], provenance["execution_confirmation"])
    require(
        not any(
            (
                protected["protected_2026_outcome_read"],
                protected["protected_2026_outcome_calculated"],
                protected["protected_2026_outcome_displayed"],
                protected["protected_2026_outcome_inspected"],
                execution["broker_accessed"],
                execution["order_routing_path_imported"],
                execution["order_routing_enabled"],
                execution["orders_placed"],
            )
        ),
        "protected-data or execution safety claim failed",
    )
    for relative, expected_hash in provenance["output_hashes"].items():
        path = REPO_ROOT / relative
        require(path.is_file(), f"provenance output is missing: {relative}")
        require(sha256_file(path) == expected_hash, f"provenance output hash drifted: {relative}")
    permutation = pd.read_csv(PRIMARY / "label_permutation_summary_v1.csv")
    require(
        bool(permutation["draws_valid"].eq(1000).all()),
        "a label-permutation result has fewer than 1000 valid draws",
    )
    bootstrap = pd.read_csv(PRIMARY / "session_cluster_bootstrap_v1.csv")
    require(
        bool(bootstrap["draws_valid"].eq(1000).all()),
        "a bootstrap result has fewer than 1000 valid draws",
    )
    require(
        bool(
            pd.to_datetime(predictions["maximum_predictor_timestamp"], utc=True)
            .le(pd.to_datetime(predictions["feature_available_timestamp_utc"], utc=True))
            .all()
        ),
        "causal predictor timestamp exceeds the entry-available timestamp",
    )
    require(
        bool(
            predictions["maximum_predictor_bar_ordinal"]
            .eq(predictions["checkpoint"].astype(int) - 1)
            .all()
        ),
        "a predictor used a current or future bar",
    )
    return {
        "output_hashes_verified": int(len(provenance["output_hashes"])),
        "permutation_draws_per_result": 1000,
        "bootstrap_draws_per_result": 1000,
        "protected_2026_outcomes_accessed": False,
        "order_routing_enabled": False,
        "orders_placed": False,
        "passed": True,
    }


def run() -> dict[str, Any]:
    assessment = read_opened(PRIMARY / "assessment_episode_predictions_v1.parquet")
    stress = read_opened(PRIMARY / "stress_episode_predictions_v1.parquet")
    require(len(assessment) == 417, "assessment episode count drifted")
    require(len(stress) == 525, "stress episode count drifted")
    predictions = pd.concat([assessment, stress], ignore_index=True)
    result = {
        "schema_version": "m1c-asymmetric-downside-residual-v1-audit",
        "status": "passed",
        "assessment_rows": int(len(assessment)),
        "stress_rows": int(len(stress)),
        "target_reconstruction": reconstruct_targets(predictions),
        "score_threshold_action_reconstruction": audit_scores_actions(predictions),
        "frozen_system_regression": audit_frozen_fields(predictions),
        "safety_and_artifacts": audit_safety_and_artifacts(predictions),
    }
    write_json(PRIMARY / "independent_audit_v1.json", result)
    markdown = "\n".join(
        [
            "# Independent audit — M1C Asymmetric Downside Residual V1",
            "",
            "`passed`",
            "",
            f"- Assessment/stress rows: {len(assessment)}/{len(stress)}.",
            "- Endpoint returns, previous-close IV thresholds, inclusive states, and the "
            "canonical strict M1C movement label were reconstructed from primitive bars and "
            "prior-close ATM IV without importing the experiment target helper.",
            "- Persisted scaler, coefficients, intercept, OOF quantiles, scores, and actions "
            "reproduced within 1e-12.",
            "- M1C probability/tail membership, fresh episode IDs, A1 outputs, Tail Phase, and "
            "movement-consumed fields exactly match the frozen Tail Phase source artifact.",
            "- All recorded output hashes, 1,000-draw bootstrap cells, and 1,000-draw "
            "permutation cells passed.",
            "- No joint-probability or contaminated feature column was emitted.",
            "- No protected 2026 outcome or order path was accessed.",
            "",
        ]
    )
    (PRIMARY / "independent_audit_v1.md").write_text(markdown, encoding="utf-8")
    return result


def main() -> int:
    print(json.dumps(json_safe(run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
