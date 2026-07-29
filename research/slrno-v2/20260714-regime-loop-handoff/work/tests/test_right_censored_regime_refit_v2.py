from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORK_DIR.parents[3]
PACKAGE_ROOT = REPO_ROOT / "packages/stocker_research/src"
for path in (WORK_DIR, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_right_censored_regime_refit_v2 import (  # noqa: E402
    _filter_segment,
    _hazard,
    _mutual_information,
)
from regime_repair_artifacts_v2 import SAFETY_FLAGS  # noqa: E402
from regime_repair_pipeline_v2 import (  # noqa: E402
    REQUIRED_ARTIFACTS,
    _attach_first_event_identity,
)

from stocker_research.regime_gap_segmentation_v2 import (  # noqa: E402
    annotate_causal_segments,
)
from stocker_research.regime_refit_v2 import (  # noqa: E402
    DURATION_ONLY_MODEL_ID,
    FULL_REFIT_MODEL_ID,
)
from stocker_research.right_censored_duration_v2 import (  # noqa: E402
    DurationFitConfig,
    RunEndingStatus,
    classify_training_run_endings,
    estimate_right_censored_durations,
)

CONTRACT = WORK_DIR / "contracts/20260719-right-censored-regime-refit-v2.json"
PREVIOUS = WORK_DIR / "contracts/20260718-regime-model-validity-v2.json"


def _bars(
    *,
    states: list[int],
    ordinals: list[int] | None = None,
    expected: int | None = None,
) -> pd.DataFrame:
    ordinals = ordinals or list(range(len(states)))
    start = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": "TEST",
            "session": "2024-01-02",
            "bar_ordinal": ordinals,
            "bar_start_timestamp": [
                start + pd.Timedelta(minutes=5 * ordinal) for ordinal in ordinals
            ],
            "state": states,
        }
    )
    annotated, _ = annotate_causal_segments(
        frame,
        expected_bars={("TEST", "2024-01-02"): expected or len(states)},
    )
    return annotated


@pytest.mark.parametrize("key,value", SAFETY_FLAGS.items())
def test_contract_contains_every_safety_flag(key: str, value: object) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract[key] == value


def test_contract_keeps_part_b_and_dictionary_promotion_closed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["part_b"]["opened"] is False
    assert contract["part_b"]["scored"] is False
    assert contract["dictionary"]["promotion_enabled"] is False
    assert contract["dictionary"]["finalization_enabled"] is False


def test_contract_binds_unchanged_part_a_thresholds() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    assert (
        contract["unchanged_part_a_binding"]["frozen_structural_thresholds"]
        == previous["frozen_structural_thresholds"]
    )
    assert contract["unchanged_part_a_binding"]["gate_threshold_override_allowed"] is False


def test_repaired_model_ids_are_new_and_distinct() -> None:
    assert DURATION_ONLY_MODEL_ID == "regime_model_v2_duration_only_repair"
    assert FULL_REFIT_MODEL_ID == "regime_model_v2_full_right_censored_refit"
    assert DURATION_ONLY_MODEL_ID != FULL_REFIT_MODEL_ID


@pytest.mark.parametrize("duration", [1, 24, 78])
def test_synthetic_exact_exit_at_declared_age(duration: int) -> None:
    if duration == 78:
        ledger = pd.DataFrame(
            {
                "state": [0],
                "duration": [78],
                "ending_status": [RunEndingStatus.OBSERVED_STATE_EXIT.value],
                "primary_fit_eligible": [True],
            }
        )
    else:
        states = [0] * duration + [1]
        ledger = classify_training_run_endings(_bars(states=states, expected=len(states)))
        run = ledger.iloc[0]
        assert run["ending_status"] == RunEndingStatus.OBSERVED_STATE_EXIT.value
        assert int(run["duration"]) == duration
    fit = estimate_right_censored_durations(
        ledger,
        state_count=2,
        config=DurationFitConfig(maximum_age=78),
    )
    assert fit.exits[0, duration - 1] == 1
    assert fit.at_risk[0, :duration].tolist() == [1] * duration


@pytest.mark.parametrize("duration", [1, 24, 78])
def test_synthetic_right_censor_at_declared_age(duration: int) -> None:
    ledger = classify_training_run_endings(_bars(states=[0] * duration, expected=duration))
    run = ledger.iloc[0]
    assert run["ending_status"] == RunEndingStatus.RIGHT_CENSORED_SESSION_END.value
    fit = estimate_right_censored_durations(
        ledger,
        state_count=1,
        config=DurationFitConfig(maximum_age=78),
    )
    assert fit.censored[0, duration - 1] == 1
    assert fit.exits.sum() == 0
    assert fit.at_risk[0, :duration].tolist() == [1] * duration


def test_internal_gap_is_neither_exit_nor_session_censor() -> None:
    ledger = classify_training_run_endings(
        _bars(states=[0, 0, 0, 0], ordinals=[0, 1, 4, 5], expected=6)
    )
    assert set(ledger["ending_status"]) == {RunEndingStatus.INVALIDATED_BY_SOURCE_GAP.value}
    invalidated = ledger.loc[
        ledger["ending_status"].eq(RunEndingStatus.INVALIDATED_BY_SOURCE_GAP.value)
    ]
    assert not invalidated["primary_fit_eligible"].any()


def test_incomplete_session_fails_closed() -> None:
    ledger = classify_training_run_endings(_bars(states=[0, 0, 0], expected=78))
    assert ledger.iloc[-1]["ending_status"] == (
        RunEndingStatus.INCOMPLETE_OR_UNAVAILABLE_SESSION.value
    )
    assert not bool(ledger.iloc[-1]["primary_fit_eligible"])


def test_sparse_tail_backoff_is_deterministic_and_not_forced() -> None:
    at_risk = np.zeros((2, 78), dtype=int)
    exits = np.zeros_like(at_risk)
    at_risk[0, :24] = 1
    exits[0, 23] = 1
    first_hazard, first_survival, first_weight = _hazard(at_risk, exits)
    second_hazard, second_survival, second_weight = _hazard(at_risk, exits)
    np.testing.assert_array_equal(first_hazard, second_hazard)
    np.testing.assert_array_equal(first_survival, second_survival)
    np.testing.assert_array_equal(first_weight, second_weight)
    assert np.all(first_hazard >= 0.0)
    assert np.all(first_hazard < 1.0)
    assert np.all(np.diff(first_survival, axis=1) <= 0.0)


def test_independent_filter_resets_at_each_call() -> None:
    emissions = np.asarray([[0.0, -2.0], [-2.0, 0.0]])
    hazard = np.full((2, 4), 0.2)
    transitions = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    initial = np.asarray([0.75, 0.25])
    full = _filter_segment(emissions, hazard, transitions, initial)
    reset = _filter_segment(emissions[1:], hazard, transitions, initial)
    assert not np.allclose(full[0][1], reset[0][0])
    assert np.isclose(reset[0][0].sum(), 1.0)
    assert reset[1][0] == 1.0


def test_independent_nmi_is_numeric_label_invariant() -> None:
    left = np.asarray([0, 0, 1, 1, 2, 2])
    right = np.asarray([8, 8, 4, 4, 6, 6])
    _, nmi = _mutual_information(left, right)
    assert nmi == pytest.approx(1.0)


def test_required_artifacts_cover_both_decisions_and_no_part_b_outputs() -> None:
    assert "repair_decision.json" in REQUIRED_ARTIFACTS
    assert "repaired_part_a_decision.json" in REQUIRED_ARTIFACTS
    assert not any("interaction_model" in name for name in REQUIRED_ARTIFACTS)


def test_compact_first_event_rows_receive_panel_identity() -> None:
    events = pd.DataFrame(
        {
            "decision_id": ["d0"],
            "primary_label": ["NO_LOOP_WITHIN_HORIZON"],
            "bars_until_completion": [None],
        }
    )
    panel = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "session": ["2024-01-02"],
            "segment_id": ["TEST::2024-01-02::segment_00"],
            "bar_ordinal": [0],
            "bar_complete_timestamp": [pd.Timestamp("2024-01-02 14:35:00", tz="UTC")],
        }
    )

    enriched = _attach_first_event_identity(events, panel, model_lineage="MODEL_TEST")

    assert enriched.loc[0, "session"] == "2024-01-02"
    assert enriched.loc[0, "segment_id"].endswith("segment_00")
    assert enriched.loc[0, "model_lineage"] == "MODEL_TEST"


def test_primary_runner_does_not_import_runtime_modules() -> None:
    runner = WORK_DIR / "run_right_censored_regime_refit_v2.py"
    spec = importlib.util.spec_from_file_location("repair_runner_test", runner)
    assert spec is not None
    source = runner.read_text(encoding="utf-8").lower()
    assert "from stocker.integrations" not in source
    assert "import broker" not in source
    assert "place_order" not in source


def test_no_protected_2026_period_is_enabled() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["chronology"]["protected_2026_enabled"] is False
    assert contract["chronology"]["development_fit"] == "2024"
    assert contract["chronology"]["unchanged_retrospective_assessment"] == "2025"
