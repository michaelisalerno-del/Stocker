"""Outcome-blind population-clock correction for the frozen V1 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "contracts/20260714-causal-payoff-model-paths-v1.json"
PRE_SCORE_PATH = HERE / "contracts/20260714-causal-payoff-model-paths-v1-pre-score.json"
WARMUP_SESSIONS = 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    output = args.artifact / "independent_audit_population_addendum.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    original_audit = json.loads((args.artifact / "independent_audit.json").read_text())
    source_hashes = json.loads((args.artifact / "source_hashes.json").read_text())
    surface = pd.read_parquet(args.artifact / "research_surface.parquet")
    recorded_population = pd.read_csv(args.artifact / "population.csv")

    inputs = contract["inputs"]
    population = contract["population"]
    signals = pd.read_parquet(Path(inputs["accepted_setup_signals_2024"]))
    base = signals.loc[
        signals["setup"].eq(population["setup"])
        & signals["family"].eq(population["family"])
        & signals["horizon"].eq(population["horizon_bars_from_anchor"])
        & signals["status"].eq(population["status"])
        & signals["symbol_norm"].isin(population["symbols"])
    ].copy()
    base["session_date"] = base["session_date"].astype(str)
    calendar = sorted(base["session_date"].unique())
    calendar_index = {date: index for index, date in enumerate(calendar)}
    score_dates = set(calendar[WARMUP_SESSIONS:])

    expected_counts = {
        candidate["candidate"]: candidate["full_surface_expected_rows"]
        for candidate in population["primary_candidates"]
    }
    observed_counts = surface.groupby("candidate", sort=True).size().to_dict()
    observed_index = surface["session_date"].map(calendar_index)
    expected_score_flag = observed_index.ge(WARMUP_SESSIONS)

    recorded_pooled = recorded_population.loc[recorded_population["group"].eq("pooled")].iloc[0]
    checks = {
        "original_audit_only_failed_population_check": bool(
            original_audit["passed"] == original_audit["total"] - 1
            and original_audit["checks"]["frozen_population_and_calendar_match"] is False
            and original_audit["errors"]["population"] == 1
            and all(
                value
                for name, value in original_audit["checks"].items()
                if name != "frozen_population_and_calendar_match"
            )
        ),
        "base_surface_calendar_matches_contract": bool(
            len(calendar) == population["surface_sessions"]
            and calendar[0] == population["surface_first_session"]
            and calendar[-1] == population["surface_last_session"]
            and len(score_dates) == population["score_completed_sessions"]
        ),
        "candidate_population_counts_match_contract": observed_counts == expected_counts,
        "candidate_dates_are_subset_of_base_calendar": bool(observed_index.notna().all()),
        "candidate_calendar_indices_replay": bool(
            observed_index.astype(int).eq(surface["calendar_index"].astype(int)).all()
        ),
        "score_flags_replay_from_base_calendar": bool(
            expected_score_flag.eq(surface["score_eligible"].astype(bool)).all()
        ),
        "recorded_candidate_population_summary_replays": bool(
            int(recorded_pooled["full_rows"]) == len(surface)
            and int(recorded_pooled["full_sessions"]) == surface["session_date"].nunique()
            and int(recorded_pooled["score_rows"]) == int(expected_score_flag.sum())
            and int(recorded_pooled["score_sessions"])
            == surface.loc[expected_score_flag, "session_date"].nunique()
        ),
        "frozen_sources_remain_unchanged": bool(
            sha256(CONTRACT_PATH) == pre_score["sha256"]["contract"]
            and source_hashes["sha256"] == pre_score["sha256"]
        ),
    }
    passed = int(sum(checks.values()))
    payload = {
        "contract_id": contract["contract_id"],
        "audit_kind": "post-score outcome-blind population-clock correction",
        "scientific_effect": "none; no payoff, route outcome, prediction, or policy field is read",
        "supersedes_check": "frozen_population_and_calendar_match",
        "correction": (
            "The frozen V1 auditor incorrectly demanded 128 distinct sessions from the "
            "candidate-only research surface. The contract assigns 128 sessions to the "
            "underlying accepted-setup surface; candidate opportunities occur on 126 of them."
        ),
        "base_surface_sessions": len(calendar),
        "base_score_sessions": len(score_dates),
        "candidate_surface_sessions": int(surface["session_date"].nunique()),
        "candidate_score_sessions": int(surface.loc[expected_score_flag, "session_date"].nunique()),
        "candidate_counts": observed_counts,
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "addendum_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(output, payload)
    print(
        json.dumps(
            {"artifact": str(args.artifact), "passed": passed, "total": len(checks)},
            indent=2,
        )
    )
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
