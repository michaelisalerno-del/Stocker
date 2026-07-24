from __future__ import annotations

import json
from pathlib import Path

import pytest

from stocker_prospective.parity import FeatureParityError, load_feature_parity_report

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "configs/prospective/feature-parity-m1.json"
MODEL = (
    ROOT / "research/options-feasibility/"
    "20260724-minimal-intraday-iv-excess-holdout-v01/"
    "artifacts/primary/model_coefficients.json"
)


def test_report_covers_every_frozen_m1_feature_in_order() -> None:
    source = json.loads(MODEL.read_text(encoding="utf-8"))
    expected = source["M1"]["numeric_features"]
    report = load_feature_parity_report(REPORT)

    assert [item.feature_name for item in report.features] == expected
    assert report.overall_scoring_allowed is False
    assert report.overall_blocker == "blocked_feature_source_semantics_mismatch"


def test_activity_proxy_is_not_relabelled_as_exchange_volume() -> None:
    report = load_feature_parity_report(REPORT)
    activity = [item for item in report.features if "activity" in item.feature_name]

    assert activity
    assert any(item.parity_status == "incompatible" for item in activity)
    combined = " ".join(
        f"{item.historical_definition} {item.runtime_definition} {item.explanation}"
        for item in activity
    ).lower()
    assert "eodhd" in combined
    assert "exchange-wide volume" not in combined


def test_real_scoring_gate_fails_closed_for_current_report() -> None:
    report = load_feature_parity_report(REPORT)

    with pytest.raises(
        FeatureParityError,
        match="blocked_feature_source_semantics_mismatch",
    ):
        report.require_scoring_allowed()


def test_invalid_status_or_missing_feature_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    payload["features"].pop()
    payload["features"][0]["parity_status"] = "probably_close"
    broken = tmp_path / "parity.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FeatureParityError, match="blocked_feature_schema_mismatch"):
        load_feature_parity_report(broken)
