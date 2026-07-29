"""Outcome-blind state-pattern discovery and later-2024 qualification.

The discovery phase reads state-sequence columns only.  It hash-locks both its
candidate manifest and this runner before the qualification phase is allowed
to read movement outcomes.  All output is development research.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-causal-state-pattern-discovery-v1.json"
CONTRACT_SHA256 = "cb3c217da9bcbac1606ca0ef69b13bad16ae54307084c839b092edba4f7d5759"
PARENT_CONTRACT = HERE / "contracts/20260710-per-loop-movement-quality-v1.json"
ANCHOR = Path(
    "/private/tmp/stocker_frozen_loop_price_consequence_20260710/anchor_panel_train_2024.parquet"
)
PARAMETERS = Path(
    "/private/tmp/stocker_causal_semimarkov_regime_loops_20260710/frozen_semimarkov_parameters.npz"
)
PREPROCESSING = Path(
    "/private/tmp/stocker_causal_semimarkov_regime_loops_20260710/frozen_emission_preprocessing.csv"
)
CYCLES = Path("/private/tmp/stocker_per_loop_movement_quality_20260710/fixed_cycles.csv")
OUT = Path("/private/tmp/stocker_causal_state_pattern_discovery_v1_20260711")

EXPECTED_HASHES = {
    "anchor": "788fd81909d1c5d3e6ee20e3e36e3ebb74199188e41052ea1b04f61c96fa9932",
    "parameters": "909858ed7c9c02c1c113661202cb5d7c6bfabd243f1cc428b8a5fb1a3c022251",
    "preprocessing": "453988be81e1dd54dd892316ab0da41423197c5aec4afb691e9010e97e05fc67",
    "cycles": "bf9292fa51de1e545e5a319fa2e2faf2088926acd5315b9106597b1da318b253",
    "parent_contract": "67d64c463df52f01f360561ef0a69d5772b7eec0409468c93d6eb5a630dee02e",
}

DISCOVERY_MONTHS = tuple(f"2024-{month:02d}" for month in range(1, 7))
QUALIFICATION_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
K = 8
DESTINATIONS = 9
SEED = 20260711
EPSILON = 1e-12
BOOTSTRAP_DRAWS = 1999
SIGN_FLIP_DRAWS = 4999

STATE_COLUMNS = (
    "anchor_id",
    "symbol_norm",
    "session_date",
    "month",
    "state",
    "previous_state_1",
    "previous_state_2",
    "future_state_1",
    "future_state_2",
    "future_state_3",
    "future_state_4",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def load_contract() -> dict[str, Any]:
    observed = sha256(CONTRACT)
    if observed != CONTRACT_SHA256:
        raise AssertionError(f"contract changed: {observed}")
    contract = json.loads(CONTRACT.read_text())
    checks = {
        "id": contract.get("contract_id") == "causal_state_pattern_discovery_v1",
        "research": contract.get("research_only") is True,
        "live": contract.get("live_ordering_enabled") is False,
        "orders": contract.get("order_placement") == "disabled",
        "discovery_blind": contract["candidate_discovery"].get("outcome_blind") is True,
        "later": contract["period_and_phase_lock"].get("2023_or_2025_paths_permitted") is False,
        "shadow": contract["period_and_phase_lock"].get(
            "prospective_shadow_read_or_write_permitted"
        )
        is False,
        "trading": contract["decision_and_stop_rules"].get(
            "trading_rule_or_PnL_model_permitted"
        )
        is False,
    }
    if not all(checks.values()):
        raise AssertionError(f"safety/semantic contract failure: {checks}")
    if tuple(contract["period_and_phase_lock"]["phase_1_allowed_anchor_columns"]) != STATE_COLUMNS:
        raise AssertionError("phase-1 state-only column whitelist changed")
    return contract


def verify_sources() -> dict[str, str]:
    observed = {
        "anchor": sha256(ANCHOR),
        "parameters": sha256(PARAMETERS),
        "preprocessing": sha256(PREPROCESSING),
        "cycles": sha256(CYCLES),
        "parent_contract": sha256(PARENT_CONTRACT),
    }
    if observed != EXPECTED_HASHES:
        raise AssertionError(f"frozen source drift: expected={EXPECTED_HASHES}, actual={observed}")
    return observed


def parse_path(value: str) -> tuple[int, ...]:
    path = tuple(int(item) for item in str(value).split("->"))
    if not path:
        raise AssertionError("empty state path")
    return path


def path_string(path: Sequence[int]) -> str:
    return "->".join(str(int(value)) for value in path)


def exact_oriented_frozen_paths(cycles: pd.DataFrame) -> set[tuple[int, ...]]:
    required = {"cycle_id", "cycle", "transition_length"}
    if len(cycles) != 20 or not required.issubset(cycles.columns):
        raise AssertionError("frozen cycle dictionary changed")
    result: set[tuple[int, ...]] = set()
    for row in cycles.itertuples(index=False):
        closed = parse_path(row.cycle)
        if closed[0] != closed[-1] or len(closed) - 1 != int(row.transition_length):
            raise AssertionError(f"invalid frozen cycle: {row.cycle}")
        core = closed[:-1]
        for index in range(len(core)):
            result.add(core[index:] + core[:index] + (core[index],))
    return result


def load_state_only(contract: dict[str, Any]) -> pd.DataFrame:
    columns = tuple(contract["period_and_phase_lock"]["phase_1_allowed_anchor_columns"])
    forbidden = tuple(contract["period_and_phase_lock"]["phase_1_forbidden_column_prefixes"])
    if columns != STATE_COLUMNS or any(column.startswith(forbidden) for column in columns):
        raise AssertionError("phase-1 column guard failed")
    frame = pd.read_parquet(ANCHOR, columns=list(columns))
    if len(frame) != 70374 or frame["anchor_id"].duplicated().any():
        raise AssertionError("anchor identity drift")
    if set(frame["month"].astype(str).unique()) != {
        *(DISCOVERY_MONTHS),
        *(QUALIFICATION_MONTHS),
    }:
        raise AssertionError("2024 month surface drift")
    if not frame["state"].between(0, 7).all():
        raise AssertionError("state outside frozen range")
    for step in range(1, 5):
        if not frame[f"future_state_{step}"].between(0, 8).all():
            raise AssertionError("future destination outside frozen range")
    return frame


def candidate_id(family: str, path: Sequence[int]) -> str:
    token = "_".join(str(int(value)) for value in path)
    if family == "closed_loop":
        return f"closed_loop__L{len(path) - 1}__{token}"
    if family == "upward_excursion":
        return f"upward_excursion__{token}"
    raise AssertionError(f"unknown family: {family}")


def occurrence(frame: pd.DataFrame, path: Sequence[int]) -> np.ndarray:
    path = tuple(int(value) for value in path)
    if len(path) < 2 or len(path) > 5:
        raise AssertionError("unsupported path length")
    output = frame["state"].to_numpy(int) == path[0]
    for step, destination in enumerate(path[1:], start=1):
        output &= frame[f"future_state_{step}"].to_numpy(int) == destination
    return output


def _support_descriptor(
    selected: pd.DataFrame, discovery_months: Sequence[str]
) -> dict[str, Any]:
    month_counts = selected["month"].astype(str).value_counts()
    return {
        "discovery_occurrences": int(len(selected)),
        "discovery_stocks": int(selected["symbol_norm"].nunique()),
        "minimum_discovery_month_occurrences": int(
            min((month_counts.get(month, 0) for month in discovery_months), default=0)
        ),
        **{
            f"discovery_occurrences_{month}": int(month_counts.get(month, 0))
            for month in discovery_months
        },
    }


def discover_closed_loops(
    discovery: pd.DataFrame, cycles: pd.DataFrame, contract: dict[str, Any]
) -> pd.DataFrame:
    rule = contract["candidate_discovery"]["family_exact_closed_loop"]
    frozen = exact_oriented_frozen_paths(cycles)
    rows: list[dict[str, Any]] = []
    for length in rule["transition_lengths"]:
        columns = ["state", *[f"future_state_{step}" for step in range(1, int(length) + 1)]]
        closed = discovery.loc[
            discovery[f"future_state_{int(length)}"].to_numpy(int)
            == discovery["state"].to_numpy(int)
        ]
        if closed.empty:
            continue
        for values, positions in closed.groupby(columns, sort=True).groups.items():
            path = tuple(int(value) for value in values)
            selected = discovery.loc[list(positions)]
            support = _support_descriptor(selected, DISCOVERY_MONTHS)
            eligible = bool(
                support["discovery_occurrences"] >= int(rule["minimum_discovery_occurrences"])
                and support["discovery_stocks"] >= int(rule["minimum_discovery_stocks"])
                and support["minimum_discovery_month_occurrences"]
                >= int(rule["minimum_occurrences_each_discovery_month"])
            )
            novel = path not in frozen
            rows.append(
                {
                    "candidate_id": candidate_id("closed_loop", path),
                    "family": "closed_loop",
                    "start_state": path[0],
                    "destination_state": path[-1],
                    "transition_length": len(path) - 1,
                    "exact_path": path_string(path),
                    "novel": novel,
                    "existing_control": not novel,
                    "decision_eligible": novel,
                    "discovery_eligible": eligible,
                    **support,
                }
            )
    catalog = pd.DataFrame(rows)
    if catalog.empty or catalog["candidate_id"].duplicated().any():
        raise AssertionError("closed-loop discovery catalog invalid")
    catalog["selected"] = False
    sort_columns = [
        "minimum_discovery_month_occurrences",
        "discovery_occurrences",
        "discovery_stocks",
        "exact_path",
    ]
    ascending = [False, False, False, True]
    for novel, cap in (
        (True, int(rule["selection_cap_novel"])),
        (False, int(rule["selection_cap_existing_controls"])),
    ):
        eligible = catalog.loc[
            catalog["discovery_eligible"] & catalog["novel"].eq(novel)
        ].sort_values(sort_columns, ascending=ascending, kind="stable")
        catalog.loc[eligible.head(cap).index, "selected"] = True
    return catalog.sort_values(
        ["selected", "discovery_eligible", "novel", *sort_columns],
        ascending=[False, False, False, *ascending],
        kind="stable",
    ).reset_index(drop=True)


def load_bar_range_centroids(contract: dict[str, Any]) -> np.ndarray:
    with np.load(PARAMETERS) as payload:
        means = np.asarray(payload["means"], dtype=float)
        semantic = np.asarray(payload["semantic_new_state"], dtype=int)
    if means.shape != (8, 14) or not np.array_equal(semantic, np.arange(8)):
        raise AssertionError("frozen centroid surface changed")
    centroids = means[:, int(contract["frozen_lineage"]["semimarkov_parameters"]["bar_range_feature_index"])]
    expected = np.asarray(
        contract["frozen_lineage"]["semimarkov_parameters"]["bar_range_centroids_expected"],
        dtype=float,
    )
    if not np.allclose(centroids, expected, atol=5e-7, rtol=0):
        raise AssertionError(f"bar-range centroids changed: {centroids}")
    return centroids


def discover_upward_excursions(
    discovery: pd.DataFrame, centroids: np.ndarray, contract: dict[str, Any]
) -> pd.DataFrame:
    rule = contract["candidate_discovery"]["family_directed_excursion"]
    rows: list[dict[str, Any]] = []
    for source in range(K):
        for destination in range(K):
            delta = float(centroids[destination] - centroids[source])
            allowed = bool(
                centroids[source] < 0
                and centroids[destination] > 0
                and delta >= float(rule["minimum_centroid_increase"])
            )
            if not allowed:
                continue
            selected = discovery.loc[
                discovery["state"].eq(source)
                & discovery["future_state_1"].eq(destination)
            ]
            support = _support_descriptor(selected, DISCOVERY_MONTHS)
            eligible = bool(
                support["discovery_occurrences"] >= int(rule["minimum_discovery_occurrences"])
                and support["discovery_stocks"] >= int(rule["minimum_discovery_stocks"])
                and support["minimum_discovery_month_occurrences"]
                >= int(rule["minimum_occurrences_each_discovery_month"])
            )
            path = (source, destination)
            rows.append(
                {
                    "candidate_id": candidate_id("upward_excursion", path),
                    "family": "upward_excursion",
                    "start_state": source,
                    "destination_state": destination,
                    "transition_length": 1,
                    "exact_path": path_string(path),
                    "novel": True,
                    "existing_control": False,
                    "decision_eligible": True,
                    "source_bar_range_centroid": float(centroids[source]),
                    "destination_bar_range_centroid": float(centroids[destination]),
                    "centroid_increase": delta,
                    "discovery_eligible": eligible,
                    **support,
                }
            )
    catalog = pd.DataFrame(rows)
    if catalog.empty or catalog["candidate_id"].duplicated().any():
        raise AssertionError("upward-excursion catalog invalid")
    catalog["selected"] = False
    eligible = catalog.loc[catalog["discovery_eligible"]].sort_values(
        [
            "minimum_discovery_month_occurrences",
            "discovery_occurrences",
            "discovery_stocks",
            "start_state",
            "destination_state",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    )
    catalog.loc[eligible.head(int(rule["selection_cap"])).index, "selected"] = True
    return catalog.sort_values(
        [
            "selected",
            "discovery_eligible",
            "minimum_discovery_month_occurrences",
            "discovery_occurrences",
            "start_state",
            "destination_state",
        ],
        ascending=[False, False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def build_candidate_manifest(catalog: pd.DataFrame) -> pd.DataFrame:
    selected = catalog.loc[catalog["selected"]].copy()
    if selected.empty or selected["candidate_id"].duplicated().any():
        raise AssertionError("empty or duplicate candidate manifest")
    selected = selected.sort_values(["family", "candidate_id"], kind="stable")
    selected["candidate_index"] = selected.groupby("family", sort=True).cumcount()
    required = [
        "candidate_id",
        "family",
        "candidate_index",
        "start_state",
        "destination_state",
        "transition_length",
        "exact_path",
        "novel",
        "existing_control",
        "decision_eligible",
        "discovery_occurrences",
        "discovery_stocks",
        "minimum_discovery_month_occurrences",
    ]
    return selected.loc[:, required].reset_index(drop=True)


def run_discovery() -> None:
    contract = load_contract()
    source_hashes = verify_sources()
    if OUT.exists():
        raise AssertionError(f"artifact root already exists: {OUT}")
    state = load_state_only(contract)
    discovery = state.loc[state["month"].astype(str).isin(DISCOVERY_MONTHS)].copy()
    cycles = pd.read_csv(CYCLES)
    centroids = load_bar_range_centroids(contract)
    loops = discover_closed_loops(discovery, cycles, contract)
    excursions = discover_upward_excursions(discovery, centroids, contract)
    catalog = pd.concat([loops, excursions], ignore_index=True, sort=False)
    manifest = build_candidate_manifest(catalog)

    OUT.mkdir(parents=True, exist_ok=False)
    catalog.to_csv(OUT / "discovery_catalog.csv", index=False)
    manifest.to_csv(OUT / "candidate_manifest.csv", index=False)
    discovery_summary = {
        "phase": "outcome_blind_state_only_discovery",
        "phase_1_columns_read": list(STATE_COLUMNS),
        "forbidden_outcome_columns_read": [],
        "discovery_months": list(DISCOVERY_MONTHS),
        "discovery_anchor_rows": len(discovery),
        "catalog_rows": int(len(catalog)),
        "eligible_rows": int(catalog["discovery_eligible"].sum()),
        "selected_rows": int(catalog["selected"].sum()),
        "selected_by_family": {
            str(key): int(value)
            for key, value in manifest["family"].value_counts().sort_index().items()
        },
        "selected_novel_closed_loops": int(
            ((manifest["family"] == "closed_loop") & manifest["decision_eligible"]).sum()
        ),
        "selected_existing_controls": int(manifest["existing_control"].sum()),
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "historical_volume": "not_read_directly",
        "movement_outcomes_opened": False,
    }
    write_json(OUT / "discovery_summary.json", discovery_summary)
    lock = {
        "contract_sha256": sha256(CONTRACT),
        "runner_sha256": sha256(Path(__file__)),
        "input_sha256": source_hashes["anchor"],
        "source_hashes": source_hashes,
        "manifest_sha256": sha256(OUT / "candidate_manifest.csv"),
        "catalog_sha256": sha256(OUT / "discovery_catalog.csv"),
        "manifest_rows": len(manifest),
        "candidate_ids": manifest["candidate_id"].tolist(),
        "phase_1_columns_read": list(STATE_COLUMNS),
        "movement_outcomes_opened": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(OUT / "discovery_lock.json", lock)
    print(json.dumps(discovery_summary, indent=2, sort_keys=True))


def load_locked_manifest(contract: dict[str, Any]) -> pd.DataFrame:
    lock_path = OUT / "discovery_lock.json"
    manifest_path = OUT / "candidate_manifest.csv"
    if not lock_path.exists() or not manifest_path.exists():
        raise AssertionError("qualification requires completed discovery lock")
    lock = json.loads(lock_path.read_text())
    checks = {
        "contract": lock.get("contract_sha256") == sha256(CONTRACT) == CONTRACT_SHA256,
        "runner": lock.get("runner_sha256") == sha256(Path(__file__)),
        "input": lock.get("input_sha256") == sha256(ANCHOR),
        "manifest": lock.get("manifest_sha256") == sha256(manifest_path),
        "outcome_blind": lock.get("movement_outcomes_opened") is False,
        "research": lock.get("research_only") is True,
        "live": lock.get("live_ordering_enabled") is False,
        "orders": lock.get("order_placement") == "disabled",
    }
    if not all(checks.values()):
        raise AssertionError(f"discovery lock failed: {checks}")
    manifest = pd.read_csv(manifest_path)
    for column in ("novel", "existing_control", "decision_eligible"):
        manifest[column] = manifest[column].astype(str).str.lower().map(
            {"true": True, "false": False}
        )
        if manifest[column].isna().any():
            raise AssertionError(f"boolean manifest drift: {column}")
    if len(manifest) != int(lock["manifest_rows"]):
        raise AssertionError("manifest row count changed")
    if manifest["candidate_id"].tolist() != lock["candidate_ids"]:
        raise AssertionError("manifest candidate order changed")
    if manifest["candidate_id"].duplicated().any():
        raise AssertionError("duplicate locked candidate")
    return manifest


def qualification_columns(contract: dict[str, Any]) -> list[str]:
    numeric = contract["movement_models"]["qcontext_features"]["causal_numeric_controls"]
    columns = [
        *STATE_COLUMNS,
        "quarter",
        "bar_index_in_session",
        *numeric,
        *[f"exact_{horizon}" for horizon in HORIZONS],
        *[f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS],
    ]
    return list(dict.fromkeys(columns))


def load_qualification_anchors(contract: dict[str, Any]) -> pd.DataFrame:
    columns = qualification_columns(contract)
    if any("volume" in column.lower() for column in columns):
        raise AssertionError("qualification attempted to read direct volume")
    if any("direction" in column or "signed_return" in column for column in columns):
        raise AssertionError("qualification attempted to read directional outcome")
    frame = pd.read_parquet(ANCHOR, columns=columns)
    if len(frame) != 70374 or frame["anchor_id"].duplicated().any():
        raise AssertionError("qualification anchor drift")
    frame["month"] = frame["month"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["quarter"] = frame["quarter"].astype(str)
    bar = pd.to_numeric(frame["bar_index_in_session"], errors="raise").to_numpy(int)
    if (bar < 0).any() or (bar > 53).any():
        raise AssertionError("qualification anchor clock surface changed")
    if not all(frame[f"exact_{horizon}"].astype(bool).all() for horizon in HORIZONS):
        raise AssertionError("qualification contains an inexact future price path")
    frame["session_bucket"] = np.where(bar <= 25, "open", np.where(bar <= 50, "middle", "late"))
    thresholds = contract["outcomes"]["thresholds"]
    for target in TARGETS:
        for horizon in HORIZONS:
            values = pd.to_numeric(frame[f"{target}_{horizon}"], errors="raise").to_numpy(float)
            if not np.isfinite(values).all():
                raise AssertionError(f"non-finite outcome: {target} h{horizon}")
            p75 = float(thresholds[target][str(horizon)]["p75"])
            p90 = float(thresholds[target][str(horizon)]["p90"])
            frame[f"quality_class__{target}__h{horizon}"] = np.where(
                values > p90, 2, np.where(values > p75, 1, 0)
            ).astype(np.int8)
    return frame


def expand_family(anchors: pd.DataFrame, family_manifest: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for candidate in family_manifest.itertuples(index=False):
        path = parse_path(candidate.exact_path)
        compatible = anchors.loc[anchors["state"].eq(int(candidate.start_state))].copy()
        compatible["candidate_id"] = str(candidate.candidate_id)
        compatible["candidate_index"] = int(candidate.candidate_index)
        compatible["family"] = str(candidate.family)
        compatible["decision_eligible"] = bool(candidate.decision_eligible)
        compatible["candidate_occurs"] = occurrence(compatible, path).astype(np.int8)
        frames.append(compatible)
    expanded = pd.concat(frames, ignore_index=True, sort=False)
    counts = (
        expanded.loc[expanded["candidate_occurs"].eq(1)]
        .groupby("anchor_id", sort=False)["candidate_id"]
        .transform("count")
    )
    expanded["conditional_weight"] = 0.0
    realized_index = expanded.index[expanded["candidate_occurs"].eq(1)]
    expanded.loc[realized_index, "conditional_weight"] = 1.0 / counts.to_numpy(float)
    if not np.allclose(
        expanded.loc[expanded["candidate_occurs"].eq(1)]
        .groupby("anchor_id")["conditional_weight"]
        .sum()
        .to_numpy(float),
        1.0,
    ):
        raise AssertionError("conditional overlap weights do not sum to one")
    return expanded


def fit_destination_kernel(training: pd.DataFrame, contract: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    first_counts = np.zeros((K, DESTINATIONS), dtype=float)
    history_counts = np.zeros((DESTINATIONS, DESTINATIONS, K, DESTINATIONS), dtype=float)
    state = training["state"].to_numpy(int)
    destination = training["future_state_1"].to_numpy(int)
    previous_1 = training["previous_state_1"].to_numpy(int)
    previous_2 = training["previous_state_2"].to_numpy(int)
    np.add.at(first_counts, (state, destination), 1.0)
    np.add.at(history_counts, (previous_2, previous_1, state, destination), 1.0)
    alpha = float(contract["structural_models"]["first_order_uniform_pseudocount_each_destination"])
    first = (first_counts + alpha) / (first_counts.sum(axis=1, keepdims=True) + alpha * DESTINATIONS)
    strength = float(contract["structural_models"]["history_backoff_strength"])
    history_total = history_counts.sum(axis=3, keepdims=True)
    prior = first[np.newaxis, np.newaxis, :, :]
    history = (history_counts + strength * prior) / (history_total + strength)
    if not np.allclose(first.sum(axis=1), 1.0) or not np.allclose(history.sum(axis=3), 1.0):
        raise AssertionError("destination probabilities do not normalize")
    return first, history


def path_probability_lookup(
    first: np.ndarray,
    history: np.ndarray,
    path: Sequence[int],
    tokens: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], tuple[float, float]]:
    path = tuple(int(value) for value in path)
    result: dict[tuple[int, int], tuple[float, float]] = {}
    for previous_2, previous_1 in tokens:
        p2, p1, current = int(previous_2), int(previous_1), path[0]
        history_probability = 1.0
        first_probability = 1.0
        for destination in path[1:]:
            history_probability *= float(history[p2, p1, current, destination])
            first_probability *= float(first[current, destination])
            p2, p1, current = p1, current, destination
        result[(int(previous_2), int(previous_1))] = (
            float(np.clip(history_probability, EPSILON, 1.0 - EPSILON)),
            float(np.clip(first_probability, EPSILON, 1.0 - EPSILON)),
        )
    return result


def add_structural_oof(
    expanded: pd.DataFrame,
    anchors: pd.DataFrame,
    family_manifest: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    validation = expanded.loc[expanded["month"].isin(QUALIFICATION_MONTHS)].copy()
    validation["history_probability"] = np.nan
    validation["first_order_probability"] = np.nan
    audits: list[dict[str, Any]] = []
    manifest = family_manifest.set_index("candidate_id")
    for month in QUALIFICATION_MONTHS:
        training = anchors.loc[anchors["month"].lt(month)]
        first, history = fit_destination_kernel(training, contract)
        month_positions = validation.index[validation["month"].eq(month)]
        for candidate_id, positions in validation.loc[month_positions].groupby(
            "candidate_id", sort=True
        ).groups.items():
            path = parse_path(manifest.loc[candidate_id, "exact_path"])
            selected = validation.loc[list(positions)]
            tokens = sorted(
                set(
                    zip(
                        selected["previous_state_2"].to_numpy(int),
                        selected["previous_state_1"].to_numpy(int),
                    )
                )
            )
            lookup = path_probability_lookup(first, history, path, tokens)
            values = [
                lookup[(int(p2), int(p1))]
                for p2, p1 in zip(selected["previous_state_2"], selected["previous_state_1"])
            ]
            validation.loc[list(positions), "history_probability"] = [value[0] for value in values]
            validation.loc[list(positions), "first_order_probability"] = [value[1] for value in values]
        audits.append(
            {
                "family": str(family_manifest["family"].iloc[0]),
                "validation_month": month,
                "training_anchor_rows": len(training),
                "validation_compatible_rows": len(month_positions),
                "first_order_min": float(first.min()),
                "first_order_max": float(first.max()),
                "history_min": float(history.min()),
                "history_max": float(history.max()),
            }
        )
    for column in ("history_probability", "first_order_probability"):
        values = validation[column].to_numpy(float)
        if not np.isfinite(values).all() or values.min() <= 0 or values.max() >= 1:
            raise AssertionError(f"invalid structural OOF: {column}")
    return validation.reset_index(drop=True), audits


def training_medians(frame: pd.DataFrame, numeric: Sequence[str]) -> dict[str, float]:
    anchors = frame.drop_duplicates("anchor_id", keep="first")
    result: dict[str, float] = {}
    for column in numeric:
        value = float(pd.to_numeric(anchors[column], errors="coerce").median())
        if not np.isfinite(value):
            raise AssertionError(f"non-finite training median: {column}")
        result[column] = value
    return result


def raw_context(
    frame: pd.DataFrame, numeric: Sequence[str], medians: dict[str, float]
) -> sparse.csr_matrix:
    values = frame.loc[:, list(numeric)].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(pd.Series(medians)).to_numpy(float)
    if not np.isfinite(values).all():
        raise AssertionError("non-finite causal context")
    states = frame["state"].to_numpy(int)
    state = sparse.csr_matrix(np.eye(K, dtype=float)[states])
    return sparse.hstack((state, sparse.csr_matrix(values)), format="csr")


def append_candidate_features(
    context: sparse.csr_matrix, candidate_index: np.ndarray, width: int, scale: float
) -> sparse.csr_matrix:
    indices = np.asarray(candidate_index, dtype=int)
    if len(indices) and (indices.min() < 0 or indices.max() >= width):
        raise AssertionError("candidate index outside family width")
    identity = sparse.csr_matrix(
        (
            np.full(len(indices), float(scale)),
            (np.arange(len(indices)), indices),
        ),
        shape=(len(indices), width),
    )
    return sparse.hstack((context, identity), format="csr")


def fit_logistic(
    features: sparse.csr_matrix, target: np.ndarray, weights: np.ndarray
) -> LogisticRegression:
    if not np.array_equal(np.unique(target), np.asarray([0, 1, 2])):
        raise AssertionError(f"movement training target lacks a class: {np.unique(target)}")
    model = LogisticRegression(
        C=0.2,
        solver="lbfgs",
        max_iter=1000,
        tol=0.0001,
        random_state=SEED,
    )
    model.fit(features, target, sample_weight=weights)
    if not np.array_equal(model.classes_, np.asarray([0, 1, 2])):
        raise AssertionError("movement class order changed")
    if int(model.n_iter_[0]) >= 1000:
        raise AssertionError("movement model did not converge")
    return model


def fit_movement_oof(
    expanded_all: pd.DataFrame,
    structural_oof: pd.DataFrame,
    family_manifest: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, np.ndarray]]:
    output = structural_oof.copy()
    for model in ("qcontext", "qcandidate"):
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    output[f"{model}__{target}__h{horizon}__{tier}"] = np.nan
    family = str(family_manifest["family"].iloc[0])
    width = len(family_manifest)
    numeric = tuple(contract["movement_models"]["qcontext_features"]["causal_numeric_controls"])
    scale = float(contract["movement_models"]["candidate_one_hot_scale"])
    folds: list[dict[str, Any]] = []
    parameters: dict[str, np.ndarray] = {}
    for month in QUALIFICATION_MONTHS:
        training = expanded_all.loc[
            expanded_all["month"].lt(month) & expanded_all["candidate_occurs"].eq(1)
        ].copy()
        validation_positions = output.index[output["month"].eq(month)]
        validation = output.loc[validation_positions]
        if training.empty or validation.empty:
            raise AssertionError(f"empty movement fold: {family} {month}")
        weights = training["conditional_weight"].to_numpy(float)
        if weights.sum() <= 0:
            raise AssertionError("zero conditional training weight")
        medians = training_medians(training, numeric)
        train_raw = raw_context(training, numeric, medians)
        validation_raw = raw_context(validation, numeric, medians)
        scaler = StandardScaler(with_mean=False)
        scaler.fit(train_raw, sample_weight=weights)
        train_context = scaler.transform(train_raw).tocsr()
        validation_context = scaler.transform(validation_raw).tocsr()
        train_candidate = append_candidate_features(
            train_context, training["candidate_index"].to_numpy(int), width, scale
        )
        validation_candidate = append_candidate_features(
            validation_context, validation["candidate_index"].to_numpy(int), width, scale
        )
        prefix = f"{family}__{month}"
        parameters[f"{prefix}__medians"] = np.asarray([medians[name] for name in numeric])
        parameters[f"{prefix}__scaler_scale"] = np.asarray(scaler.scale_, dtype=float)
        for target in TARGETS:
            for horizon in HORIZONS:
                target_column = f"quality_class__{target}__h{horizon}"
                y = training[target_column].to_numpy(int)
                context_model = fit_logistic(train_context, y, weights)
                candidate_model = fit_logistic(train_candidate, y, weights)
                for label, model, features in (
                    ("qcontext", context_model, validation_context),
                    ("qcandidate", candidate_model, validation_candidate),
                ):
                    probability = model.predict_proba(features)
                    if not np.allclose(probability.sum(axis=1), 1.0):
                        raise AssertionError("movement probability did not normalize")
                    output.loc[
                        validation_positions, f"{label}__{target}__h{horizon}__p75"
                    ] = probability[:, 1] + probability[:, 2]
                    output.loc[
                        validation_positions, f"{label}__{target}__h{horizon}__p90"
                    ] = probability[:, 2]
                    model_key = f"{prefix}__{target}__h{horizon}__{label}"
                    parameters[f"{model_key}__coef"] = np.asarray(model.coef_, dtype=float)
                    parameters[f"{model_key}__intercept"] = np.asarray(model.intercept_, dtype=float)
                    parameters[f"{model_key}__n_iter"] = np.asarray(model.n_iter_, dtype=int)
                folds.append(
                    {
                        "family": family,
                        "validation_month": month,
                        "target": target,
                        "horizon": horizon,
                        "training_realized_rows": len(training),
                        "training_conditional_weight": float(weights.sum()),
                        "validation_compatible_rows": len(validation),
                        "validation_realized_rows": int(validation["candidate_occurs"].sum()),
                        "family_candidate_width": width,
                        "context_feature_width": train_context.shape[1],
                        "candidate_feature_width": train_candidate.shape[1],
                        "qcontext_n_iter": int(context_model.n_iter_[0]),
                        "qcandidate_n_iter": int(candidate_model.n_iter_[0]),
                        "numeric_medians": json.dumps(medians, sort_keys=True),
                    }
                )
    probability_columns = [
        column for column in output.columns if column.startswith(("qcontext__", "qcandidate__"))
    ]
    probabilities = output[probability_columns].to_numpy(float)
    if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
        raise AssertionError("invalid movement OOF probability")
    for model in ("qcontext", "qcandidate"):
        for target in TARGETS:
            for horizon in HORIZONS:
                if (
                    output[f"{model}__{target}__h{horizon}__p90"]
                    > output[f"{model}__{target}__h{horizon}__p75"] + 1e-12
                ).any():
                    raise AssertionError("movement probability nesting failed")
    return output, folds, parameters


def clip_probability(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)


def binary_losses(observed: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    probability = clip_probability(probability)
    log_loss = -(observed * np.log(probability) + (1.0 - observed) * np.log(1.0 - probability))
    brier = (observed - probability) ** 2
    return log_loss, brier


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0 or weights.sum() <= 0:
        return math.nan
    return float(np.average(values, weights=weights))


def calibration_summary(
    observed: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    minimum_rows: int,
) -> tuple[float, float, int]:
    observed = np.asarray(observed, dtype=float)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights, dtype=float)
    bins = np.minimum((np.clip(probability, 0, 1) * 10).astype(int), 9)
    errors: list[tuple[float, float]] = []
    for index in range(10):
        selected = bins == index
        if int(selected.sum()) < int(minimum_rows) or weights[selected].sum() <= 0:
            continue
        error = abs(weighted_mean(observed[selected], weights[selected]) - weighted_mean(probability[selected], weights[selected]))
        errors.append((float(weights[selected].sum()), float(error)))
    if not errors:
        return math.inf, math.inf, 0
    total = sum(weight for weight, _ in errors)
    ece = sum(weight * error for weight, error in errors) / total
    return float(ece), float(max(error for _, error in errors)), len(errors)


def daily_values(
    frame: pd.DataFrame, values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    daily = pd.DataFrame(
        {
            "session_date": frame["session_date"].astype(str).to_numpy(),
            "weighted": np.asarray(values, dtype=float) * np.asarray(weights, dtype=float),
            "weight": np.asarray(weights, dtype=float),
        }
    ).groupby("session_date", sort=True).sum()
    daily = daily.loc[daily["weight"] > 0]
    return (daily["weighted"] / daily["weight"]).to_numpy(float)


def five_session_blocks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.asarray(
        [float(values[index : index + 5].mean()) for index in range(0, len(values), 5) if len(values[index : index + 5]) == 5],
        dtype=float,
    )


def bootstrap_interval(values: np.ndarray, seed: int, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float]:
    blocks = five_session_blocks(values)
    if len(blocks) < 5:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(blocks, size=(draws, len(blocks)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def sign_flip_p_value(values: np.ndarray, seed: int, draws: int = SIGN_FLIP_DRAWS) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return math.nan
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    null = (signs @ values) / len(values)
    return float((1 + np.sum(null <= observed)) / (draws + 1))


def holm_adjust(frame: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        for column, value in (
            ("holm_adjusted_p", np.nan),
            ("holm_pass", False),
            ("holm_rank", 0),
            ("family_size", 0),
        ):
            output[column] = value
        return output
    output["holm_adjusted_p"] = 1.0
    output["holm_pass"] = False
    output["holm_rank"] = 0
    output["family_size"] = 0
    for (_, _), positions in output.groupby(["family", "tier"], sort=True).groups.items():
        positions = list(positions)
        ordered = sorted(
            positions,
            key=lambda position: (
                float(output.loc[position, "p_value"])
                if np.isfinite(output.loc[position, "p_value"])
                else 1.0
            ),
        )
        running = 0.0
        size = len(ordered)
        for rank, position in enumerate(ordered, start=1):
            raw = float(output.loc[position, "p_value"])
            if not np.isfinite(raw):
                raw = 1.0
            adjusted = min(1.0, max(running, (size - rank + 1) * raw))
            running = adjusted
            output.loc[position, "holm_adjusted_p"] = adjusted
            output.loc[position, "holm_pass"] = adjusted <= alpha
            output.loc[position, "holm_rank"] = rank
            output.loc[position, "family_size"] = size
    return output


def support_gate(frame: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    rule = contract["qualification_support_each_candidate"]
    realized = frame.loc[frame["candidate_occurs"].eq(1)]
    quarter_counts = realized["quarter"].value_counts()
    checks = {
        "compatible_rows": len(frame) >= int(rule["compatible_rows_minimum"]),
        "realized_rows": len(realized) >= int(rule["realized_rows_minimum"]),
        "realized_stocks": realized["symbol_norm"].nunique() >= int(rule["realized_stocks_minimum"]),
        "quarters": all(
            int(quarter_counts.get(quarter, 0)) >= int(rule["realized_rows_each_quarter_minimum"])
            for quarter in rule["required_quarters"]
        ),
    }
    return {
        "compatible_rows": len(frame),
        "realized_rows": len(realized),
        "realized_stocks": int(realized["symbol_norm"].nunique()),
        "q3_realized_rows": int(quarter_counts.get("2024_q3", 0)),
        "q4_realized_rows": int(quarter_counts.get("2024_q4", 0)),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def structural_gate(frame: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    observed = frame["candidate_occurs"].to_numpy(int)
    weights = np.ones(len(frame), dtype=float)
    history = frame["history_probability"].to_numpy(float)
    first = frame["first_order_probability"].to_numpy(float)
    history_ll, history_brier = binary_losses(observed, history)
    first_ll, first_brier = binary_losses(observed, first)
    minimum = int(contract["structural_gate_each_candidate"]["minimum_supported_bin_rows"])
    history_ece, history_max, history_bins = calibration_summary(observed, history, weights, minimum)
    first_ece, first_max, first_bins = calibration_summary(observed, first, weights, minimum)
    checks = {
        "history_log_loss_below_first_order": float(history_ll.mean()) < float(first_ll.mean()),
        "history_brier_below_first_order": float(history_brier.mean()) < float(first_brier.mean()),
        "history_ece": history_ece <= first_ece + 0.01,
        "history_max_bin_error": history_max <= first_max + 0.02,
    }
    return {
        "history_log_loss": float(history_ll.mean()),
        "first_order_log_loss": float(first_ll.mean()),
        "relative_log_loss_improvement": float((first_ll.mean() - history_ll.mean()) / first_ll.mean()),
        "history_brier": float(history_brier.mean()),
        "first_order_brier": float(first_brier.mean()),
        "history_ece": history_ece,
        "first_order_ece": first_ece,
        "history_maximum_supported_bin_error": history_max,
        "first_order_maximum_supported_bin_error": first_max,
        "history_supported_bins": history_bins,
        "first_order_supported_bins": first_bins,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def tier_support(
    observed: np.ndarray, tier: str, contract: dict[str, Any]
) -> tuple[bool, int, int]:
    rule = contract["qualification_support_each_candidate"]
    positive = int(np.asarray(observed, dtype=int).sum())
    negative = int(len(observed) - positive)
    if tier == "p75":
        passed = positive >= int(rule["p75_positive_and_negative_rows_each_target_horizon_minimum"]) and negative >= int(rule["p75_positive_and_negative_rows_each_target_horizon_minimum"])
    else:
        passed = positive >= int(rule["p90_positive_rows_each_target_horizon_minimum"]) and negative >= int(rule["p90_negative_rows_each_target_horizon_minimum"])
    return bool(passed), positive, negative


def evaluate_cell(
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    tier: str,
    contract: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    threshold = 1 if tier == "p75" else 2
    realized = frame.loc[frame["candidate_occurs"].eq(1)].reset_index(drop=True)
    observed = (realized[f"quality_class__{target}__h{horizon}"].to_numpy(int) >= threshold).astype(int)
    weights = realized["conditional_weight"].to_numpy(float)
    context_probability = realized[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float)
    candidate_probability = realized[f"qcandidate__{target}__h{horizon}__{tier}"].to_numpy(float)
    support_pass, positive_rows, negative_rows = tier_support(observed, tier, contract)
    context_ll, context_brier = binary_losses(observed, context_probability)
    candidate_ll, candidate_brier = binary_losses(observed, candidate_probability)
    conditional_context_ll = weighted_mean(context_ll, weights)
    conditional_candidate_ll = weighted_mean(candidate_ll, weights)
    conditional_improvement = (conditional_context_ll - conditional_candidate_ll) / conditional_context_ll
    conditional_brier_difference = weighted_mean(candidate_brier - context_brier, weights)
    difference = candidate_ll - context_ll
    daily_difference = daily_values(realized, difference, weights)
    bootstrap_lower, bootstrap_upper = bootstrap_interval(daily_difference, seed)
    quarter_differences = {
        quarter: weighted_mean(
            difference[realized["quarter"].eq(quarter).to_numpy()],
            weights[realized["quarter"].eq(quarter).to_numpy()],
        )
        for quarter in ("2024_q3", "2024_q4")
    }
    loso: dict[str, float] = {}
    for symbol in sorted(realized["symbol_norm"].astype(str).unique()):
        keep = realized["symbol_norm"].astype(str).ne(symbol).to_numpy()
        loso[symbol] = weighted_mean(difference[keep], weights[keep])
    conditional_context_ece, conditional_context_max, conditional_context_bins = calibration_summary(
        observed, context_probability, weights, 25
    )
    conditional_candidate_ece, conditional_candidate_max, conditional_candidate_bins = calibration_summary(
        observed, candidate_probability, weights, 25
    )
    lift = observed.astype(float) - context_probability
    daily_lift = daily_values(realized, lift, weights)
    lift_lower, lift_upper = bootstrap_interval(daily_lift, seed + 1)

    all_observed = (
        frame["candidate_occurs"].to_numpy(bool)
        & (frame[f"quality_class__{target}__h{horizon}"].to_numpy(int) >= threshold)
    ).astype(int)
    structural = frame["history_probability"].to_numpy(float)
    joint_context_probability = structural * frame[f"qcontext__{target}__h{horizon}__{tier}"].to_numpy(float)
    joint_candidate_probability = structural * frame[f"qcandidate__{target}__h{horizon}__{tier}"].to_numpy(float)
    joint_context_ll, joint_context_brier = binary_losses(all_observed, joint_context_probability)
    joint_candidate_ll, joint_candidate_brier = binary_losses(all_observed, joint_candidate_probability)
    joint_context_mean = float(joint_context_ll.mean())
    joint_candidate_mean = float(joint_candidate_ll.mean())
    joint_improvement = (joint_context_mean - joint_candidate_mean) / joint_context_mean
    joint_brier_difference = float((joint_candidate_brier - joint_context_brier).mean())
    joint_weights = np.ones(len(frame), dtype=float)
    joint_context_ece, joint_context_max, joint_context_bins = calibration_summary(
        all_observed, joint_context_probability, joint_weights, 100
    )
    joint_candidate_ece, joint_candidate_max, joint_candidate_bins = calibration_summary(
        all_observed, joint_candidate_probability, joint_weights, 100
    )

    rule = contract["quality_gates_each_candidate_target_horizon_tier"]
    checks = {
        "tier_support": support_pass,
        "conditional_relative_log_loss": conditional_improvement
        >= float(rule["conditional_qcandidate_vs_qcontext"]["minimum_relative_log_loss_improvement"]),
        "conditional_brier": conditional_brier_difference < 0,
        "both_quarters": all(np.isfinite(value) and value < 0 for value in quarter_differences.values()),
        "every_stock_deletion": bool(loso) and all(np.isfinite(value) and value <= 0 for value in loso.values()),
        "conditional_bootstrap": np.isfinite(bootstrap_upper) and bootstrap_upper < 0,
        "joint_relative_log_loss": joint_improvement
        >= float(rule["joint_same_structural_probability"]["minimum_relative_log_loss_improvement"]),
        "joint_brier": joint_brier_difference < 0,
        "conditional_ece": conditional_candidate_ece <= conditional_context_ece + 0.01,
        "conditional_max_bin_error": conditional_candidate_max <= 0.08,
        "joint_ece": joint_candidate_ece <= joint_context_ece + 0.01,
        "joint_max_bin_error": joint_candidate_max <= 0.04,
        "lift_bootstrap": np.isfinite(lift_lower) and lift_lower > 0,
    }
    return {
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "observed_rate": weighted_mean(observed, weights),
        "mean_qcontext": weighted_mean(context_probability, weights),
        "mean_qcandidate": weighted_mean(candidate_probability, weights),
        "observed_rate_over_qcontext": weighted_mean(observed, weights)
        / max(weighted_mean(context_probability, weights), EPSILON),
        "conditional_context_log_loss": conditional_context_ll,
        "conditional_candidate_log_loss": conditional_candidate_ll,
        "conditional_relative_log_loss_improvement": conditional_improvement,
        "conditional_brier_difference": conditional_brier_difference,
        "conditional_bootstrap_lower": bootstrap_lower,
        "conditional_bootstrap_upper": bootstrap_upper,
        "quarter_log_loss_differences": quarter_differences,
        "maximum_leave_one_stock_out_log_loss_difference": max(loso.values(), default=math.inf),
        "leave_one_stock_out_log_loss_differences": loso,
        "conditional_context_ece": conditional_context_ece,
        "conditional_candidate_ece": conditional_candidate_ece,
        "conditional_context_maximum_supported_bin_error": conditional_context_max,
        "conditional_candidate_maximum_supported_bin_error": conditional_candidate_max,
        "conditional_context_supported_bins": conditional_context_bins,
        "conditional_candidate_supported_bins": conditional_candidate_bins,
        "joint_context_log_loss": joint_context_mean,
        "joint_candidate_log_loss": joint_candidate_mean,
        "joint_relative_log_loss_improvement": joint_improvement,
        "joint_brier_difference": joint_brier_difference,
        "joint_context_ece": joint_context_ece,
        "joint_candidate_ece": joint_candidate_ece,
        "joint_context_maximum_supported_bin_error": joint_context_max,
        "joint_candidate_maximum_supported_bin_error": joint_candidate_max,
        "joint_context_supported_bins": joint_context_bins,
        "joint_candidate_supported_bins": joint_candidate_bins,
        "lift_bootstrap_lower": lift_lower,
        "lift_bootstrap_upper": lift_upper,
        "daily_sessions": len(daily_difference),
        "sign_flip_p_value": sign_flip_p_value(daily_difference, seed + 2),
        "checks": checks,
        "pass_without_holm": bool(all(checks.values())),
    }


def dependency_profiles(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    realized = frame.loc[frame["candidate_occurs"].eq(1)]
    for (month, bucket), selected in realized.groupby(["month", "session_bucket"], sort=True):
        row: dict[str, Any] = {
            "candidate_id": str(frame["candidate_id"].iloc[0]),
            "family": str(frame["family"].iloc[0]),
            "month": month,
            "session_bucket": bucket,
            "realized_rows": len(selected),
            "stocks": int(selected["symbol_norm"].nunique()),
        }
        for target in TARGETS:
            for horizon in HORIZONS:
                quality = selected[f"quality_class__{target}__h{horizon}"].to_numpy(int)
                row[f"{target}__h{horizon}__p75_rate"] = float((quality >= 1).mean())
                row[f"{target}__h{horizon}__p90_rate"] = float((quality >= 2).mean())
        rows.append(row)
    return rows


def evaluate_family(
    oof: pd.DataFrame,
    family_manifest: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    support_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    multiplicity_rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for candidate_position, candidate in enumerate(family_manifest.itertuples(index=False)):
        selected = oof.loc[oof["candidate_id"].eq(candidate.candidate_id)].reset_index(drop=True)
        support = support_gate(selected, contract)
        structural = structural_gate(selected, contract)
        support_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "decision_eligible": bool(candidate.decision_eligible),
                **{key: value for key, value in support.items() if key != "checks"},
                "checks": json.dumps(support["checks"], sort_keys=True),
            }
        )
        structural_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "decision_eligible": bool(candidate.decision_eligible),
                **{key: value for key, value in structural.items() if key != "checks"},
                "checks": json.dumps(structural["checks"], sort_keys=True),
            }
        )
        profiles.extend(dependency_profiles(selected))
        for target_index, target in enumerate(TARGETS):
            for horizon in HORIZONS:
                for tier_index, tier in enumerate(TIERS):
                    detail = evaluate_cell(
                        selected,
                        target,
                        horizon,
                        tier,
                        contract,
                        SEED + candidate_position * 10000 + target_index * 1000 + horizon * 10 + tier_index,
                    )
                    cell_rows.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "family": candidate.family,
                            "decision_eligible": bool(candidate.decision_eligible),
                            "target": target,
                            "horizon": horizon,
                            "tier": tier,
                            **{key: value for key, value in detail.items() if key not in {"checks", "quarter_log_loss_differences", "leave_one_stock_out_log_loss_differences"}},
                            "checks": json.dumps(detail["checks"], sort_keys=True),
                            "quarter_log_loss_differences": json.dumps(safe(detail["quarter_log_loss_differences"]), sort_keys=True),
                            "leave_one_stock_out_log_loss_differences": json.dumps(safe(detail["leave_one_stock_out_log_loss_differences"]), sort_keys=True),
                        }
                    )
                    if bool(candidate.decision_eligible) and bool(support["pass"]):
                        multiplicity_rows.append(
                            {
                                "candidate_id": candidate.candidate_id,
                                "family": candidate.family,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "daily_sessions": detail["daily_sessions"],
                                "p_value": detail["sign_flip_p_value"],
                            }
                        )
    return {
        "support": pd.DataFrame(support_rows),
        "structural": pd.DataFrame(structural_rows),
        "cells": pd.DataFrame(cell_rows),
        "multiplicity_inputs": pd.DataFrame(multiplicity_rows),
        "profiles": pd.DataFrame(profiles),
    }


def grade_candidates(
    manifest: pd.DataFrame,
    support: pd.DataFrame,
    structural: pd.DataFrame,
    cells: pd.DataFrame,
    multiplicity: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    support_lookup = support.set_index("candidate_id")["pass"].to_dict()
    structural_lookup = structural.set_index("candidate_id")["pass"].to_dict()
    holm_lookup = multiplicity.set_index(["candidate_id", "target", "horizon", "tier"])[
        "holm_pass"
    ].to_dict()
    cell_lookup = cells.set_index(["candidate_id", "target", "horizon", "tier"]).to_dict("index")
    grades: list[dict[str, Any]] = []
    for candidate in manifest.itertuples(index=False):
        for horizon in HORIZONS:
            p75 = [cell_lookup[(candidate.candidate_id, target, horizon, "p75")] for target in TARGETS]
            p90 = [cell_lookup[(candidate.candidate_id, target, horizon, "p90")] for target in TARGETS]
            p75_pass = bool(
                all(bool(cell["pass_without_holm"]) for cell in p75)
                and all(holm_lookup.get((candidate.candidate_id, target, horizon, "p75"), False) for target in TARGETS)
            )
            good_rate = all(
                float(cell["observed_rate"]) >= 0.30
                and float(cell["mean_qcandidate"]) >= 0.30
                and float(cell["observed_rate_over_qcontext"]) >= 1.10
                for cell in p75
            )
            good = bool(
                bool(candidate.decision_eligible)
                and support_lookup.get(candidate.candidate_id, False)
                and structural_lookup.get(candidate.candidate_id, False)
                and p75_pass
                and good_rate
            )
            p90_pass = bool(
                all(bool(cell["pass_without_holm"]) for cell in p90)
                and all(holm_lookup.get((candidate.candidate_id, target, horizon, "p90"), False) for target in TARGETS)
            )
            high_rate = bool(
                all(
                    float(cell["observed_rate"]) >= 0.35
                    and float(cell["mean_qcandidate"]) >= 0.35
                    for cell in p75
                )
                and all(
                    float(cell["observed_rate"]) >= 0.15
                    and float(cell["mean_qcandidate"]) >= 0.15
                    and float(cell["observed_rate_over_qcontext"]) >= 1.20
                    for cell in p90
                )
            )
            high = bool(good and p90_pass and high_rate)
            grade = (
                "development_high_candidate"
                if high
                else "development_good_candidate"
                if good
                else "development_unqualified"
            )
            grades.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "exact_path": candidate.exact_path,
                    "start_state": int(candidate.start_state),
                    "horizon": horizon,
                    "decision_eligible": bool(candidate.decision_eligible),
                    "support_pass": bool(support_lookup.get(candidate.candidate_id, False)),
                    "structural_pass": bool(structural_lookup.get(candidate.candidate_id, False)),
                    "both_targets_p75_pass": p75_pass,
                    "good_rate_pass": good_rate,
                    "both_targets_p90_pass": p90_pass,
                    "high_rate_pass": high_rate,
                    "grade": grade,
                    "prospective_validated": False,
                    "certified_good_or_high": False,
                }
            )
    grade_frame = pd.DataFrame(grades)
    candidates = grade_frame.loc[
        grade_frame["grade"].isin(["development_good_candidate", "development_high_candidate"])
    ].copy()
    capped: list[pd.DataFrame] = []
    decisions: dict[str, Any] = {}
    cap = int(contract["decision_and_stop_rules"]["candidate_cap_each_family"])
    for family in sorted(manifest["family"].unique()):
        family_candidates = candidates.loc[candidates["family"].eq(family)]
        if len(family_candidates) == 0:
            label = "no_qualified_development_candidate"
        elif len(family_candidates) <= cap:
            label = "development_candidates_frozen_pending_separate_future_contract"
            capped.append(family_candidates)
        else:
            label = "candidate_cap_exceeded_fail_closed"
        decisions[family] = {
            "label": label,
            "candidate_count_before_cap": len(family_candidates),
            "candidate_cap": cap,
            "candidate_ids": [
                f"{row.candidate_id}@h{int(row.horizon)}"
                for row in family_candidates.itertuples(index=False)
            ]
            if len(family_candidates) <= cap
            else [],
        }
    frozen = pd.concat(capped, ignore_index=True) if capped else candidates.iloc[0:0].copy()
    decision = {
        "families": decisions,
        "total_frozen_development_candidates": len(frozen),
        "later_period_scoring_performed": False,
        "same_experiment_refinement_performed": False,
        "prospective_validated": False,
        "certified_good_or_high": False,
        "economic_edge_claim": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    return grade_frame, frozen, decision


def artifact_manifest(root: Path, names: Iterable[str]) -> dict[str, Any]:
    return {
        "files": {
            name: {"size": (root / name).stat().st_size, "sha256": sha256(root / name)}
            for name in sorted(names)
        },
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def run_qualification() -> None:
    contract = load_contract()
    source_hashes = verify_sources()
    manifest = load_locked_manifest(contract)
    if (OUT / "summary.json").exists():
        raise AssertionError("qualification artifacts already exist")
    anchors = load_qualification_anchors(contract)
    all_oof: list[pd.DataFrame] = []
    all_support: list[pd.DataFrame] = []
    all_structural: list[pd.DataFrame] = []
    all_cells: list[pd.DataFrame] = []
    all_multiplicity: list[pd.DataFrame] = []
    all_profiles: list[pd.DataFrame] = []
    all_fold_rows: list[dict[str, Any]] = []
    all_structural_audit: list[dict[str, Any]] = []
    all_parameters: dict[str, np.ndarray] = {}

    for family in sorted(manifest["family"].unique()):
        family_manifest = manifest.loc[manifest["family"].eq(family)].copy()
        expanded = expand_family(anchors, family_manifest)
        structural_oof, structural_audit = add_structural_oof(
            expanded, anchors, family_manifest, contract
        )
        movement_oof, fold_rows, parameters = fit_movement_oof(
            expanded, structural_oof, family_manifest, contract
        )
        evaluated = evaluate_family(movement_oof, family_manifest, contract)
        all_oof.append(movement_oof)
        all_support.append(evaluated["support"])
        all_structural.append(evaluated["structural"])
        all_cells.append(evaluated["cells"])
        all_multiplicity.append(evaluated["multiplicity_inputs"])
        all_profiles.append(evaluated["profiles"])
        all_fold_rows.extend(fold_rows)
        all_structural_audit.extend(structural_audit)
        overlap = set(all_parameters).intersection(parameters)
        if overlap:
            raise AssertionError(f"duplicate parameter keys: {sorted(overlap)[:3]}")
        all_parameters.update(parameters)

    oof = pd.concat(all_oof, ignore_index=True, sort=False)
    support = pd.concat(all_support, ignore_index=True)
    structural = pd.concat(all_structural, ignore_index=True)
    cells = pd.concat(all_cells, ignore_index=True)
    multiplicity = holm_adjust(pd.concat(all_multiplicity, ignore_index=True))
    profiles = pd.concat(all_profiles, ignore_index=True)
    grades, candidates, decision = grade_candidates(
        manifest, support, structural, cells, multiplicity, contract
    )
    holm = multiplicity.set_index(["candidate_id", "target", "horizon", "tier"])[
        ["holm_adjusted_p", "holm_pass"]
    ]
    cells = cells.join(holm, on=["candidate_id", "target", "horizon", "tier"])
    cells["holm_adjusted_p"] = cells["holm_adjusted_p"].fillna(1.0)
    cells["holm_pass"] = cells["holm_pass"].fillna(False).astype(bool)

    prediction_columns = [
        "anchor_id",
        "candidate_id",
        "candidate_index",
        "family",
        "decision_eligible",
        "symbol_norm",
        "session_date",
        "month",
        "quarter",
        "session_bucket",
        "state",
        "previous_state_1",
        "previous_state_2",
        "future_state_1",
        "future_state_2",
        "future_state_3",
        "future_state_4",
        "candidate_occurs",
        "conditional_weight",
        "history_probability",
        "first_order_probability",
        *[f"quality_class__{target}__h{horizon}" for target in TARGETS for horizon in HORIZONS],
        *[
            f"{model}__{target}__h{horizon}__{tier}"
            for model in ("qcontext", "qcandidate")
            for target in TARGETS
            for horizon in HORIZONS
            for tier in TIERS
        ],
    ]
    oof.loc[:, prediction_columns].to_parquet(OUT / "oof_predictions_2024_h2.parquet", index=False)
    support.to_csv(OUT / "qualification_support.csv", index=False)
    structural.to_csv(OUT / "structural_metrics.csv", index=False)
    cells.to_csv(OUT / "quality_cells.csv", index=False)
    multiplicity.to_csv(OUT / "multiplicity.csv", index=False)
    grades.to_csv(OUT / "horizon_grades.csv", index=False)
    candidates.to_csv(OUT / "development_candidates.csv", index=False)
    profiles.to_csv(OUT / "dependency_profiles.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(OUT / "movement_fold_audit.csv", index=False)
    pd.DataFrame(all_structural_audit).to_csv(OUT / "structural_fold_audit.csv", index=False)
    np.savez_compressed(OUT / "movement_model_parameters.npz", **all_parameters)
    write_json(OUT / "decision.json", decision)
    summary = {
        "contract_id": contract["contract_id"],
        "scientific_status": contract["scientific_status"],
        "source_hashes": source_hashes,
        "contract_sha256": sha256(CONTRACT),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_manifest_sha256": sha256(OUT / "candidate_manifest.csv"),
        "qualification_months": list(QUALIFICATION_MONTHS),
        "manifest_candidates": len(manifest),
        "manifest_by_family": {
            str(key): int(value) for key, value in manifest["family"].value_counts().sort_index().items()
        },
        "qualification_oof_rows": len(oof),
        "support_pass_candidates": int(support["pass"].sum()),
        "structural_pass_candidates": int(structural["pass"].sum()),
        "quality_cells_pass_before_holm": int(cells["pass_without_holm"].sum()),
        "holm_pass_tests": int(multiplicity["holm_pass"].sum()),
        "grade_counts": {
            str(key): int(value) for key, value in grades["grade"].value_counts().sort_index().items()
        },
        "decision": decision,
        "direct_volume_fields_used": [],
        "volume_label": "historical_volume_not_used_directly",
        "direction_or_signed_return_used": False,
        "later_period_scoring_performed": False,
        "prospective_shadow_read_or_write_performed": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(OUT / "summary.json", summary)
    names = [
        "candidate_manifest.csv",
        "decision.json",
        "dependency_profiles.csv",
        "development_candidates.csv",
        "discovery_catalog.csv",
        "discovery_lock.json",
        "discovery_summary.json",
        "horizon_grades.csv",
        "movement_fold_audit.csv",
        "movement_model_parameters.npz",
        "multiplicity.csv",
        "oof_predictions_2024_h2.parquet",
        "quality_cells.csv",
        "qualification_support.csv",
        "structural_fold_audit.csv",
        "structural_metrics.csv",
        "summary.json",
    ]
    write_json(OUT / "artifact_manifest.json", artifact_manifest(OUT, names))
    print(json.dumps(summary, indent=2, sort_keys=True))


def self_test() -> None:
    contract = load_contract()
    assert candidate_id("closed_loop", (3, 6, 3)) == "closed_loop__L2__3_6_3"
    toy = pd.DataFrame(
        {
            "state": [3, 3],
            "future_state_1": [6, 5],
            "future_state_2": [3, 3],
            "future_state_3": [8, 8],
            "future_state_4": [8, 8],
        }
    )
    assert occurrence(toy, (3, 6, 3)).tolist() == [True, False]
    first = np.full((8, 9), 1 / 9)
    history = np.full((9, 9, 8, 9), 1 / 9)
    lookup = path_probability_lookup(first, history, (3, 6, 3), [(1, 2)])
    assert np.allclose(lookup[(1, 2)], (1 / 81, 1 / 81))
    adjusted = holm_adjust(
        pd.DataFrame(
            {
                "family": ["a"] * 3,
                "tier": ["p75"] * 3,
                "p_value": [0.001, 0.02, 0.04],
            }
        )
    ).sort_values("p_value")
    assert np.all(np.diff(adjusted["holm_adjusted_p"]) >= -1e-15)
    assert contract["research_only"] is True
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("discover", "qualify"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.phase == "discover":
        run_discovery()
    elif args.phase == "qualify":
        run_qualification()
    else:
        parser.error("choose --phase discover or --phase qualify")


if __name__ == "__main__":
    main()
