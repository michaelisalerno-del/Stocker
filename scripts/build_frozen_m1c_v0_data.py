#!/usr/bin/env python3
"""Build the reviewed, runtime-only frozen M1C data module.

The source artifacts stay research evidence.  This script verifies every complete
artifact before extracting the exact immutable subset needed by the server plugin.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COHORT = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)
CHECKPOINTS = tuple(range(6, 35, 2))
SOURCES = {
    "m1c_feature": (
        "research/directional-readiness/20260726-stock-local-directional-archetypes-v0/"
        "artifacts/primary/causal_movement_feature_manifest.json",
        "6f59177a58973d33a24741e3c265e1831bfb6dc07afac17ae371501019bdc5cc",
    ),
    "m1c_threshold": (
        "research/directional-readiness/20260726-stock-local-directional-archetypes-v0/"
        "artifacts/primary/causal_movement_threshold.json",
        "1aae6b7b28bf0f51b914d069bb31ac2e209b43ddaaa938fd373c55a2e65cbabe",
    ),
    "group_i_scaling": (
        "research/route-competition/20260722-broad-conflict-advance-hazard-v02/"
        "artifacts/primary/model_configurations.json",
        "9521b093f01313a4993a9e101ef0e214ab32933809585ef267551747762b49c2",
    ),
    "direction_models": (
        "research/directional-readiness/20260726-stock-local-directional-archetypes-v0/"
        "artifacts/primary/model_configurations.json",
        "857a8907159a4e8025f8417f829597c9b6338ad1eeabea646bc08307543bfdf8",
    ),
    "direction_normalisation": (
        "research/directional-readiness/20260726-stock-local-directional-archetypes-v0/"
        "artifacts/primary/stock_local_normalisation_parameters.json",
        "6b8e4f058e4e89a405c9ec33ddba1c6dd84ceaa5f19a4504f93bc937fa5a317d",
    ),
    "direction_thresholds": (
        "research/directional-readiness/20260726-stock-local-directional-archetypes-v0/"
        "artifacts/primary/frozen_archetype_thresholds.json",
        "54651f1fc4c74de8fecc49cfd88e317a41211201a2966128f4f0f0e23f293a3b",
    ),
    "direction_beta": (
        "research/directional-readiness/20260726-stock-local-directional-archetypes-v0/"
        "artifacts/primary/stock_market_beta_parameters.csv",
        "86c2c63def43262b758f5f4df59113d2f8038ad151fcfd0c173f7a4d043cc7be",
    ),
    "front_options_features": (
        "research/cross-market-context/20260723-daily-stock-front-options-context-v01/"
        "artifacts/primary/front_options_feature_manifest.json",
        "fb2b734ce84e545d6839dc6d537aa73532d733f0e2206e0e0a402f96786f3499",
    ),
    "front_options_regime": (
        "research/cross-market-context/20260723-daily-stock-front-options-context-v01/"
        "artifacts/primary/front_options_regime_mapping.json",
        "a73c7e2c0b9220ac598c7051e7ced77ea0e0cf0a71b769e4a4b42ae7885d2985",
    ),
}
TARGET = ROOT / "packages/stocker_ideas/src/stocker_ideas/plugins/frozen_m1c_v0_data.py"


def _verified_bytes(name: str) -> bytes:
    relative, expected = SOURCES[name]
    content = (ROOT / relative).read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise SystemExit(f"frozen source hash mismatch: {name}")
    return content


def _json(name: str) -> dict[str, Any]:
    value = json.loads(_verified_bytes(name))
    if not isinstance(value, dict):
        raise SystemExit(f"frozen source is not an object: {name}")
    return value


def _payload() -> dict[str, Any]:
    feature = _json("m1c_feature")
    threshold = _json("m1c_threshold")
    scaling = _json("group_i_scaling")
    direction = _json("direction_models")
    normalisation = _json("direction_normalisation")
    direction_thresholds = _json("direction_thresholds")
    front_features = _json("front_options_features")
    front_regime = _json("front_options_regime")
    full_models = {name: direction["full_models"][name] for name in ("A1", "C1", "R1")}
    direction_features = {
        feature_name for model in full_models.values() for feature_name in model["numeric_features"]
    }
    parameters = [
        item
        for item in normalisation["parameters"]
        if item["feature"] in direction_features
        and (
            item["stock"] == "__POOLED__"
            or (item["stock"] in COHORT and item["checkpoint"] in CHECKPOINTS)
        )
    ]
    if len(parameters) != len(direction_features) * (len(COHORT) * len(CHECKPOINTS) + 1):
        raise SystemExit("frozen direction normalisation subset is incomplete")
    beta_rows = list(csv.DictReader(_verified_bytes("direction_beta").decode().splitlines()))
    beta_rows = [item for item in beta_rows if item["fit_scope"] == "full_2024"]
    if len(beta_rows) != 60:
        raise SystemExit("frozen direction beta subset is incomplete")
    local_scaling = {
        key: value
        for key, value in scaling["local_development_scaling"].items()
        if key.split("|", 1)[0] in COHORT
    }
    if len(local_scaling) != len(COHORT) * len(CHECKPOINTS):
        raise SystemExit("frozen local M1C scaling subset is incomplete")
    return {
        "artifact_hashes": {name: expected for name, (_path, expected) in SOURCES.items()},
        "cohort": COHORT,
        "checkpoints": CHECKPOINTS,
        "m1c_feature": feature,
        "m1c_threshold": threshold,
        "group_i_component_scaling": scaling["component_development_scaling"],
        "group_i_local_scaling": local_scaling,
        "direction_models": full_models,
        "direction_normalisation": {
            "fit_period": normalisation["fit_period"],
            "minimum_support": normalisation["minimum_support"],
            "parameters": parameters,
        },
        "direction_thresholds": {name: direction_thresholds[name] for name in ("A1", "C1", "R1")},
        "direction_beta": beta_rows,
        "front_options_features": {
            "imputation_medians": front_features["imputation_medians"],
            "scales": front_features["scales"],
        },
        "front_options_regime": {
            name: front_regime[name]
            for name in (
                "input_columns",
                "input_medians",
                "canonical_input_means",
                "canonical_weights",
                "canonical_covariances",
            )
        },
    }


def main() -> None:
    serialized = json.dumps(_payload(), sort_keys=True, separators=(",", ":")).encode()
    encoded = base64.b85encode(zlib.compress(serialized, level=9)).decode()
    lines = [encoded[index : index + 100] for index in range(0, len(encoded), 100)]
    module = '''"""Generated reviewed data for Frozen M1C Signal V0; do not edit."""\n\n'''
    module += "from __future__ import annotations\n\nimport base64\nimport json\nimport zlib\n\n"
    module += "_ENCODED = (\n" + "".join(f'    "{line}"\n' for line in lines) + ")\n\n"
    module += "DATA = json.loads(zlib.decompress(base64.b85decode(_ENCODED)).decode())\n"
    module += "del _ENCODED\n"
    TARGET.write_text(module, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(serialized)} bytes -> {len(module)} chars)")


if __name__ == "__main__":
    main()
