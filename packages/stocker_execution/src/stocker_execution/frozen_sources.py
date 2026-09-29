"""Immutable source provenance; thresholds are research outcomes, not new exit rules."""

from typing import Any

FROZEN: dict[str, dict[str, Any]] = {
    "CL": {
        "primary": "call_10d_0DTE",
        "research_gain_threshold_percent": 500,
        "selection_horizon_minutes": 60,
        "sha256": "f43b3a633e1438bbbbc1042440b2f6201482206499f7c8d1016e55a1d096fc16",
    },
    "GC": {
        "primary": "put_10d_0DTE",
        "research_gain_threshold_percent": 500,
        "selection_horizon_minutes": 60,
        "sha256": "24030157dc377d2a308782f72e32fd69776d26d60f06e417b8c07d6db5157e26",
    },
    "NG": {
        "primary": "put_10d_0DTE",
        "research_gain_threshold_percent": 500,
        "selection_horizon_minutes": 60,
        "sha256": "5a79e753ab0b20d08ac6bc451b1ce174c615c8ad060d087cf5835ac48298f2d9",
    },
    "NQ": {
        "primary": "put_10d_0DTE",
        "research_gain_threshold_percent": 500,
        "selection_horizon_minutes": 60,
        "sha256": "3fa8adb9c2a40e2cca860b5e75cb83cfd44507f336fde3c1669ab1cbfa7b61d6",
    },
    "SI": {
        "primary": "put_20d_0DTE",
        "research_gain_threshold_percent": 300,
        "selection_horizon_minutes": 60,
        "sha256": "54ce25d91854d79ee03c2508a78df5e5655762e4be1e738700061e8b5ee1eb66",
    },
}

SOURCE_HASHES: dict[str, str] = {
    "CL_TAIL_FROZEN.json": "f43b3a633e1438bbbbc1042440b2f6201482206499f7c8d1016e55a1d096fc16",
    "GC_TAIL_FROZEN.json": "24030157dc377d2a308782f72e32fd69776d26d60f06e417b8c07d6db5157e26",
    "NG_TAIL_FROZEN.json": "5a79e753ab0b20d08ac6bc451b1ce174c615c8ad060d087cf5835ac48298f2d9",
    "NQ_TAIL_FROZEN.json": "3fa8adb9c2a40e2cca860b5e75cb83cfd44507f336fde3c1669ab1cbfa7b61d6",
    "SI_TAIL_FROZEN.json": "54ce25d91854d79ee03c2508a78df5e5655762e4be1e738700061e8b5ee1eb66",
    "build_features.py.txt": "411e534eecafe898b7e4a53982ae2d72bbbb5af2e2def4387884cdef83fe3263",
    "fixed_spec.json": "f2857420e4b6b80686e80d906a27a2a413c94c3d7fff9696340cfafb1b895755",
    "tail.py.txt": "5c6356d9d8038307e93934a2141aca4ba5d7675da1c89ba464bbffeb7776cd98",
}
