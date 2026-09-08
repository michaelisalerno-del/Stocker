"""Freeze the user-specified FIT-only quantile, then reconcile saved development outcomes.

No fitting, threshold search, market-data acquisition, or broker access.
Run freeze before reconcile; reconcile cannot write or change the admission specification.
"""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

MODEL_HASH = "68c0f5ebf4a23744a171e32a32a8b336e8d832013692913ca034f744a88e8bc2"
PARAMETERS_HASH = "396e6a3134b101b50b3abfc6855cb414dbcf60721729ee748d2dc17484a31b94"
SOURCE_SPEC_HASH = "fe9e198cc8c31b67bf7ac7da1a12c0264e8512d3741c49609369f156ba9d56f5"
PREDICTIONS_HASH = "a09bdf76247ac64ef2297c46f580449abc20e4a1f4796fd4209bae4901801966"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "packages/stocker_core/src/stocker_core/method_artifacts/session_hard"
REPORT = ROOT / "docs/session_hard_q1_reconciliation.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def freeze(source):
    if (DEST / "prospective_q1.json").exists():
        raise ValueError("Specification is already frozen; refusing to overwrite")
    expected = {
        "MODEL_T0.joblib": MODEL_HASH,
        "MODEL_T0_parameters.json": PARAMETERS_HASH,
        "fixed_spec.json": SOURCE_SPEC_HASH,
        "MODEL_T0_predictions.csv": PREDICTIONS_HASH,
    }
    for name, digest in expected.items():
        if sha(source / name) != digest:
            raise ValueError(f"Frozen source hash mismatch: {name}")
    # Only FIT score/id/date fields enter the quantile. No labels or economics are read.
    fit = sorted(
        (
            {
                "signal_id": r["signal_id"],
                "session": r["session"],
                "risk_score": float(r["probability"]),
            }
            for r in rows(source / "MODEL_T0_predictions.csv")
            if r["split"] == "FIT"
        ),
        key=lambda r: r["signal_id"],
    )
    if not fit or any(r["session"] >= "2025-07-01" for r in fit):
        raise ValueError("Invalid original FIT partition")
    scores = np.asarray([r["risk_score"] for r in fit], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite FIT score")
    cutoff = float(np.quantile(scores, 0.20, method="linear"))
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("MODEL_T0.joblib", "MODEL_T0_parameters.json", "fixed_spec.json"):
        (DEST / name).write_bytes((source / name).read_bytes())
    write(DEST / "fit_score_distribution.json", fit)
    spec = {
        "spec_id": "SESSION_HARD_MODEL_T0_FIT_Q1_V1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "q1_risk_cutoff": cutoff,
        "q1_risk_cutoff_float64_hex": cutoff.hex(),
        "admission_rule": "whipsaw_risk_score <= q1_risk_cutoff",
        "equality_policy": "ADMIT",
        "signal_id_tie_break": "Historical assessment equal-count reproduction only",
        "quantile": {
            "library": "numpy",
            "version": np.__version__,
            "function": "numpy.quantile",
            "q": 0.20,
            "method": "linear",
            "dtype": "float64",
        },
        "fit_population": {
            "count": len(fit),
            "first_session": min(r["session"] for r in fit),
            "last_session": max(r["session"] for r in fit),
            "selection": "Original saved MODEL_T0_predictions.csv split == FIT",
        },
        "model_sha256": MODEL_HASH,
        "preprocessing_parameters_sha256": PARAMETERS_HASH,
        "research_spec_sha256": SOURCE_SPEC_HASH,
        "source_predictions_sha256": PREDICTIONS_HASH,
        "score_distribution_sha256": sha(DEST / "fit_score_distribution.json"),
        "model_refit": False,
        "assessment_used_to_choose_cutoff": False,
        "protected_data_accessed": False,
        "order_placement": "disabled",
    }
    write(DEST / "prospective_q1.json", spec)
    (DEST / "prospective_q1.sha256").write_text(sha(DEST / "prospective_q1.json") + "\n")
    print(json.dumps(spec, indent=2))


def reconcile(source, economics):
    spec_path = DEST / "prospective_q1.json"
    digest = sha(spec_path)
    if digest != (DEST / "prospective_q1.sha256").read_text().strip():
        raise ValueError("Prospective spec hash mismatch")
    spec = json.loads(spec_path.read_text())
    if sha(source / "MODEL_T0_predictions.csv") != PREDICTIONS_HASH:
        raise ValueError("Prediction hash mismatch")
    manifest = json.loads((economics / "economics_frozen_before_tick_overlay.json").read_text())
    for name in (
        "known_order_outcomes.csv",
        "unknown_dual_order_outcomes.csv",
        "assessment_scope.csv",
        "fixed_spec.json",
    ):
        if sha(economics / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Economic source hash mismatch: {name}")
    pred = rows(source / "MODEL_T0_predictions.csv")
    fit = [r for r in pred if r["split"] == "FIT"]
    assessment = [r for r in pred if r["split"] == "ASSESSMENT"]
    cutoff = spec["q1_risk_cutoff"]
    admitted = [r for r in assessment if float(r["probability"]) <= cutoff]
    ids = {r["signal_id"] for r in admitted}
    old_ids = {r["signal_id"] for r in assessment if float(r["bucket"]) == 1}
    known = [r for r in rows(economics / "known_order_outcomes.csv") if r["signal_id"] in ids]
    unknown = [
        r for r in rows(economics / "unknown_dual_order_outcomes.csv") if r["signal_id"] in ids
    ]
    if len(known) + len(unknown) != len(ids):
        raise ValueError("Economic population does not reconcile")
    if any(r["scenario_status"] != "BOTH_SCENARIOS_SCORED" for r in unknown):
        raise ValueError("Saved economic bounds incomplete")
    total = sum(float(r["net_r"]) for r in known)
    low = total + sum(float(r["min_net_R"]) for r in unknown)
    high = total + sum(float(r["max_net_R"]) for r in unknown)
    fit_n = sum(float(r["probability"]) <= cutoff for r in fit)
    result = {
        "evidence": "DEVELOPMENT_OPERATIONALISATION_NOT_UNTOUCHED_VALIDATION",
        "prospective_spec_sha256": digest,
        "q1_risk_cutoff": cutoff,
        "fit": {"total": len(fit), "admitted": fit_n, "admitted_pct": 100 * fit_n / len(fit)},
        "assessment": {
            "total": len(assessment),
            "admitted": len(ids),
            "admitted_pct": 100 * len(ids) / len(assessment),
            "whipsaw_count": sum(r["target"] == "TWO_SIDED_WHIPSAW" for r in admitted),
            "whipsaw_rate": sum(r["target"] == "TWO_SIDED_WHIPSAW" for r in admitted) / len(ids),
        },
        "old_assessment_q1": {
            "count": len(old_ids),
            "overlap": len(ids & old_ids),
            "old_only": len(old_ids - ids),
            "prospective_only": len(ids - old_ids),
        },
        "economics": {
            "known_count": len(known),
            "unknown_order_count": len(unknown),
            "known_total_net_R": total,
            "worst_total_net_R": low,
            "best_total_net_R": high,
            "worst_net_R_per_candidate": low / len(ids),
            "best_net_R_per_candidate": high / len(ids),
            "source": "Frozen pre-tick-overlay bounds; no recomputation or new orders",
            "entry_M": 0.20,
            "stop_M": 0.50,
            "target_M": 1.00,
            "deadline": "original T0+15 minutes",
            "round_trip_cost_bps": 10,
        },
        "source_hashes": {
            name: manifest["files_sha256"][name]
            for name in (
                "known_order_outcomes.csv",
                "unknown_dual_order_outcomes.csv",
                "assessment_scope.csv",
                "fixed_spec.json",
            )
        },
        "cutoff_changed_after_reconciliation": False,
        "protected_data_accessed": False,
        "order_placement": "disabled",
    }
    write(REPORT, result)
    assert sha(spec_path) == digest
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "reconcile"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--economics", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.source)
    else:
        reconcile(args.source, args.economics)
