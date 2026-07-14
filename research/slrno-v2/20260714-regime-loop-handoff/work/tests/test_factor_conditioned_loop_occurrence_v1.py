from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "run_factor_conditioned_loop_occurrence_v1.py"
)
SPEC = importlib.util.spec_from_file_location("factor_loop_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_contract_hash_schedule_and_safety_are_exact() -> None:
    contract = runner.load_contract()
    assert runner.CONTRACT_SHA256 == (
        "ef8b61bdd4f6671fa64713551a9991f6e4591c3c96bc1ccc324c81b7195bfe7d"
    )
    assert runner.sha256(runner.CONTRACT) == runner.CONTRACT_SHA256
    assert contract["research_only"] is True
    assert contract["live_ordering_enabled"] is False
    assert contract["order_placement"] == "disabled"
    assert tuple(contract["models"]["ridge_lambda_grid"]) == runner.LAMBDA_GRID
    assert tuple(runner.INNER_SCHEDULE) == runner.OUTER_MONTHS
    assert runner.INNER_SCHEDULE["2024-07"] == (
        "2024-04",
        "2024-05",
        "2024-06",
    )
    assert runner.INNER_SCHEDULE["2024-12"][-1] == "2024-11"
    cleanup = contract["feature_construction"]["provider_scan_and_factor_table"][
        "fit_2024_placeholder_audit"
    ]
    assert cleanup["discarded_rows"] == 5539
    assert cleanup["partial_null_rows"] == 0
    assert cleanup["run_entry_natural_key_intersection"] == 0


def test_parser_requires_exactly_one_phase() -> None:
    parser = runner.build_parser()
    for flag in (
        "--self-test-only",
        "--validate-only",
        "--fit-only",
        "--score-only",
    ):
        parsed = parser.parse_args([flag])
        assert sum(
            bool(getattr(parsed, name))
            for name in ("self_test_only", "validate_only", "fit_only", "score_only")
        ) == 1
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--fit-only", "--score-only"])


def test_lambda_selection_uses_equal_month_mean_and_largest_tie() -> None:
    months = ("2024-04", "2024-05")
    scores = {
        value: {month: 0.4 + value for month in months}
        for value in runner.LAMBDA_GRID
    }
    scores[0.0001] = {month: 0.2 + 5e-13 for month in months}
    scores[0.0003] = {month: 0.2 for month in months}
    selected, objectives = runner.select_lambda(scores, months)
    assert selected == 0.0003
    assert set(objectives) == set(runner.LAMBDA_GRID)


def test_each_head_selects_lambda_independently() -> None:
    rows = []
    preferred = {"qpattern": 0.0001, "qlimited4": 0.001, "qfull9": 0.003}
    for month in runner.FULL_SELECTION_MONTHS:
        for head in runner.HEADS:
            for value in runner.LAMBDA_GRID:
                rows.append(
                    {
                        "validation_month": month,
                        "head": head,
                        "lambda": value,
                        "log_loss": 0.1 + abs(value - preferred[head]),
                    }
                )
    selected, audit = runner.select_head_lambdas(
        pd.DataFrame(rows), runner.FULL_SELECTION_MONTHS
    )
    assert selected == preferred
    assert len(audit) == len(runner.HEADS) * len(runner.LAMBDA_GRID)
    assert audit.groupby("head")["selected"].sum().eq(1).all()


def test_safe_json_recurses_numpy_and_nonfinite() -> None:
    payload = {"a": np.asarray([np.int64(2), np.nan]), "b": Path("x")}
    assert runner.safe(payload) == {"a": [2, None], "b": "x"}


def _synthetic_replay(rows: int) -> dict[str, np.ndarray]:
    replay = {
        "target": np.zeros(rows, dtype=np.int8),
        "offset": np.zeros(rows),
        "anchor_id": np.arange(rows),
        "cycle_index": np.zeros(rows, dtype=np.int16),
    }
    for head in runner.HEADS:
        replay[f"{head}__data"] = np.asarray([], dtype=float)
        replay[f"{head}__indices"] = np.asarray([], dtype=int)
        replay[f"{head}__indptr"] = np.zeros(rows + 1, dtype=int)
        replay[f"{head}__shape"] = np.asarray([rows, 1], dtype=int)
    return replay


def test_fold_normalization_rejects_silent_drop_wrong_month_and_target_change() -> None:
    expected = pd.DataFrame(
        {
            "anchor_id": [10, 11],
            "cycle_id": [1, 1],
            "target": [0, 1],
        }
    )
    predictions = expected.assign(
        validation_month="2024-07",
        qhistory=0.1,
        qold_limited_path=0.1,
        qpattern=0.1,
        qlimited4=0.1,
        qfull9=0.1,
    )
    result = {
        "predictions": predictions,
        "audit": {"session_overlap": False},
        "replay_inputs": _synthetic_replay(2),
    }
    runner._normalize_fold_result(result, "2024-07", expected)

    dropped = dict(result)
    dropped["predictions"] = predictions.iloc[:1].copy()
    with pytest.raises(AssertionError, match="exact expected anchor-cycle"):
        runner._normalize_fold_result(dropped, "2024-07", expected)

    changed = dict(result)
    changed_predictions = predictions.copy()
    changed_predictions.loc[1, "target"] = 0
    changed["predictions"] = changed_predictions
    with pytest.raises(AssertionError, match="target changed"):
        runner._normalize_fold_result(changed, "2024-07", expected)

    wrong_month = dict(result)
    wrong_month_predictions = predictions.copy()
    wrong_month_predictions["validation_month"] = "2024-08"
    wrong_month["predictions"] = wrong_month_predictions
    with pytest.raises(AssertionError, match="month mismatch"):
        runner._normalize_fold_result(wrong_month, "2024-07", expected)


def test_empty_replay_and_incomplete_evaluation_fail_closed() -> None:
    with pytest.raises(AssertionError, match="nonempty mapping"):
        runner._validate_replay_inputs({}, "synthetic")
    with pytest.raises(AssertionError, match="mandatory artifacts"):
        runner._evaluation_payload(
            {"primary_pass": True, "artifacts": {"gates": {"primary_pass": True}}}
        )


def test_pinned_later_replay_is_an_explicit_gate() -> None:
    predictions = pd.DataFrame({"anchor_id": [0], "cycle_id": [1], "target": [0]})
    predictions.attrs["pinned_path_replay"] = {
        "period": "2025",
        "rows": 1,
        "tolerance": 1e-12,
        "qhistory_max_abs_error": 0.0,
        "qold_limited_path_max_abs_error": 1e-13,
        "pinned_probabilities_substituted_before_candidate_scoring": True,
    }
    assert runner._validated_pinned_replay(predictions, "2025")["rows"] == 1
    predictions.attrs["pinned_path_replay"]["qhistory_max_abs_error"] = 1e-6
    with pytest.raises(AssertionError, match="pinned path replay failed"):
        runner._validated_pinned_replay(predictions, "2025")


def test_actual_production_modules_pass_data_free_self_tests() -> None:
    result = runner.run_self_test_only()
    assert result["status"] == "self_tests_passed"
    assert result["core"]["passed"] == result["core"]["total"]
    assert result["evaluation"]["status"] == "passed_without_data"
    assert result["later_period_paths_resolved"] is False


def test_artifact_manifest_binds_hash_and_size(tmp_path: Path) -> None:
    artifact = tmp_path / "example.json"
    runner.write_json(artifact, {"value": 1})
    manifest = runner.build_artifact_manifest(tmp_path, [artifact.name])
    runner.verify_artifact_manifest(tmp_path, manifest)
    artifact.write_text("{}\n")
    with pytest.raises(AssertionError, match="fit artifact changed"):
        runner.verify_artifact_manifest(tmp_path, manifest)


def test_manifest_rejects_traversal_and_symlink_before_artifact_read(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "fit.json"
    artifact.write_text("{}\n")
    manifest = runner.build_artifact_manifest(tmp_path, [artifact.name])
    manifest["artifacts"] = {
        "../outside.json": {
            "sha256": "0" * 64,
            "size": 1,
        }
    }
    with pytest.raises(AssertionError, match="invalid artifact name"):
        runner.verify_artifact_manifest(tmp_path, manifest)

    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("{}\n")
    link = tmp_path / "linked.json"
    link.symlink_to(outside)
    manifest["artifacts"] = {
        link.name: {
            "sha256": runner.sha256(outside),
            "size": outside.stat().st_size,
        }
    }
    with pytest.raises(AssertionError, match="may not be a symlink"):
        runner.verify_artifact_manifest(tmp_path, manifest)


def _write_valid_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path.parent / f"{tmp_path.name}-sources"
    source_root.mkdir()
    core_source = source_root / "core.py"
    evaluation_source = source_root / "evaluation.py"
    auditor_source = source_root / "auditor.py"
    for path in (core_source, evaluation_source, auditor_source):
        path.write_text("# synthetic source\n")
    monkeypatch.setattr(runner, "CORE_SOURCE", core_source)
    monkeypatch.setattr(runner, "EVALUATION_SOURCE", evaluation_source)
    monkeypatch.setattr(runner, "AUDITOR_SOURCE", auditor_source)

    for name in runner.REQUIRED_SCORE_INPUT_ARTIFACTS:
        (tmp_path / name).write_bytes(f"synthetic:{name}".encode())
    manifest = runner.build_artifact_manifest(
        tmp_path, runner.REQUIRED_SCORE_INPUT_ARTIFACTS
    )
    runner.write_json(tmp_path / runner.FIT_MANIFEST_NAME, manifest)
    fit = {
        "status": "fit_frozen_pending_independent_pre_score_audit",
        "contract_sha256": runner.CONTRACT_SHA256,
        "runner_sha256": runner.sha256(MODULE_PATH),
        "production_source_hashes": {
            core_source.name: runner.sha256(core_source),
            evaluation_source.name: runner.sha256(evaluation_source),
        },
        "complete_fit_artifact_manifest_sha256": runner.sha256(
            tmp_path / runner.FIT_MANIFEST_NAME
        ),
        "development_2024_primary_pass": True,
        "scoring_authorized": False,
        "later_period_paths_resolved": False,
        "later_period_rows_read": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    runner.write_json(tmp_path / runner.FIT_COMPLETE_NAME, fit)
    audit = {"all_passed": True, "scoring_authorized": True}
    runner.write_json(tmp_path / runner.AUDIT_RESULT_NAME, audit)
    authorization = {
        "contract_sha256": runner.CONTRACT_SHA256,
        "runner_sha256": runner.sha256(MODULE_PATH),
        "fit_complete_sha256": runner.sha256(tmp_path / runner.FIT_COMPLETE_NAME),
        "complete_fit_artifact_manifest_sha256": runner.sha256(
            tmp_path / runner.FIT_MANIFEST_NAME
        ),
        "auditor_source_sha256": runner.sha256(auditor_source),
        "auditor_result_sha256": runner.sha256(tmp_path / runner.AUDIT_RESULT_NAME),
        "auditor_all_passed": True,
        "development_2024_primary_pass": True,
        "scoring_authorized": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "shadow_tree_read": False,
        "shadow_tree_written": False,
    }
    runner.write_json(tmp_path / runner.AUTHORIZATION_NAME, authorization)


def test_score_lock_fails_before_any_module_or_later_path_resolution(
    tmp_path: Path,
) -> None:
    calls = {"later": 0}

    def later(*args: object, **kwargs: object) -> None:
        calls["later"] += 1

    fake_core = SimpleNamespace(phase_source_paths=later)
    with pytest.raises(FileNotFoundError):
        runner.run_score_only(
            root=tmp_path,
            core_module=fake_core,
            evaluation_module=SimpleNamespace(),
        )
    assert calls["later"] == 0


def test_invalid_authorization_fails_before_manifest_artifacts_are_followed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_valid_lock(tmp_path, monkeypatch)
    authorization_path = tmp_path / runner.AUTHORIZATION_NAME
    authorization = json.loads(authorization_path.read_text())
    authorization["scoring_authorized"] = False
    runner.write_json(authorization_path, authorization)

    def must_not_verify(*args: object, **kwargs: object) -> None:
        raise AssertionError("manifest entries followed before marker validation")

    monkeypatch.setattr(runner, "verify_artifact_manifest", must_not_verify)
    with pytest.raises(AssertionError, match="invalid pre-score authorization"):
        runner.validate_pre_score_authorization(tmp_path)


def test_authorization_booleans_are_type_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_valid_lock(tmp_path, monkeypatch)
    authorization_path = tmp_path / runner.AUTHORIZATION_NAME
    authorization = json.loads(authorization_path.read_text())
    authorization["auditor_all_passed"] = 1
    runner.write_json(authorization_path, authorization)
    with pytest.raises(AssertionError, match="invalid pre-score authorization"):
        runner.validate_pre_score_authorization(tmp_path)


def test_stale_scoring_namespace_stops_before_later_path_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_valid_lock(tmp_path, monkeypatch)
    (tmp_path / "predictions_2025.parquet").write_bytes(b"stale")
    calls = {"later": 0}

    def later(*args: object, **kwargs: object) -> None:
        calls["later"] += 1

    fake_core = SimpleNamespace(phase_source_paths=later)
    with pytest.raises(FileExistsError, match="pristine scoring namespace"):
        runner.run_score_only(
            root=tmp_path,
            core_module=fake_core,
            evaluation_module=SimpleNamespace(),
        )
    assert calls["later"] == 0


def test_pristine_namespace_rejects_alternate_names_and_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_valid_lock(tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / runner.FIT_MANIFEST_NAME).read_text())
    (tmp_path / "2025_predictions.parquet").write_bytes(b"stale")
    (tmp_path / "nested").mkdir()
    with pytest.raises(FileExistsError, match="pristine scoring namespace"):
        runner.assert_pristine_scoring_namespace(tmp_path, manifest)


def test_valid_lock_is_checked_before_later_phase_is_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_valid_lock(tmp_path, monkeypatch)
    calls: list[str] = []

    def authorize(**kwargs: object) -> str:
        calls.append("authorize")
        return "guard"

    def later(phase: str, *, guard_token: str) -> object:
        assert phase == "score" and guard_token == "guard"
        calls.append("later")
        raise RuntimeError("synthetic stop after later resolution")

    fake_core = SimpleNamespace(
        authorize_later_phase=authorize,
        phase_source_paths=later,
    )
    with pytest.raises(RuntimeError, match="synthetic stop"):
        runner.run_score_only(
            root=tmp_path,
            core_module=fake_core,
            evaluation_module=SimpleNamespace(),
        )
    assert calls == ["authorize", "later"]


def test_2024_gate_failure_writes_stop_marker_and_never_fits_full_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    placeholder_cleanup = runner.load_contract()["feature_construction"][
        "provider_scan_and_factor_table"
    ]["fit_2024_placeholder_audit"]
    prepared = {
        "anchors": pd.DataFrame({"anchor_id": [0]}),
        "source_manifest": {
            "period": "2024",
            "year": 2024,
            "run_rows": 110949,
            "compatible_rows": 759212,
            "positive_rows": 46630,
            "later_period_paths_resolved": False,
            "later_period_rows_read": False,
            "shadow_tree_read": False,
            "shadow_tree_written": False,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
        },
        "audit": {"factor_table": {"placeholder_cleanup": placeholder_cleanup}},
    }
    cycles = pd.DataFrame({"cycle_id": ["cycle_01"]})
    expanded = pd.DataFrame({"anchor_id": [0], "cycle_id": ["cycle_01"], "target": [0]})
    monkeypatch.setattr(runner, "_prepare_2024", lambda core: (prepared, cycles, expanded))

    grid_rows = [
        {
            "validation_month": month,
            "head": head,
            "lambda": value,
            "log_loss": 0.1 + value,
        }
        for month in runner.FULL_SELECTION_MONTHS
        for head in runner.HEADS
        for value in runner.LAMBDA_GRID
    ]
    grid = pd.DataFrame(grid_rows)
    monkeypatch.setattr(runner, "_grid_scores_2024", lambda *args: (grid, {}))
    oof = pd.DataFrame(
        {
            "anchor_id": [0],
            "cycle_id": ["cycle_01"],
            "target": [0],
            "qhistory": [0.1],
            "qold_limited_path": [0.1],
            "qpattern": [0.1],
            "qlimited4": [0.1],
            "qfull9": [0.1],
        }
    )
    monkeypatch.setattr(
        runner,
        "_outer_oof_2024",
        lambda *args: (oof, pd.DataFrame({"selected": [True]}), [], {}),
    )
    monkeypatch.setattr(
        runner,
        "_source_hashes",
        lambda: {"core.py": "a", "evaluation.py": "b"},
    )
    calls = {"full": 0}

    def fit_full_bundle(**kwargs: object) -> object:
        calls["full"] += 1
        raise AssertionError("must not fit after failed 2024 gates")

    core = SimpleNamespace(fit_full_bundle=fit_full_bundle)
    evaluation = SimpleNamespace(
        validate_population=lambda *args, **kwargs: {"pass": True},
        evaluate_period=lambda **kwargs: {
            "primary_pass": False,
            "artifacts": {
                "support": {},
                "overall": pd.DataFrame(),
                "ranking": pd.DataFrame(),
                "calibration": {},
                "comparisons": pd.DataFrame(),
                "bootstrap": {},
                "slices": pd.DataFrame(),
                "falsification": {},
                "gates": {"primary_pass": False},
            },
        }
    )
    result = runner.run_fit_only(
        root=tmp_path,
        core_module=core,
        evaluation_module=evaluation,
    )
    assert result["status"] == "stopped_2024_primary_gates_failed"
    assert result["scoring_authorized"] is False
    assert result["later_period_paths_resolved"] is False
    assert calls["full"] == 0
    assert not (tmp_path / runner.MODEL_BUNDLE_NAME).exists()


def test_fit_source_manifest_rejects_masked_phase_claims() -> None:
    cleanup = runner.load_contract()["feature_construction"][
        "provider_scan_and_factor_table"
    ]["fit_2024_placeholder_audit"]
    source_manifest = {
        "period": "2024",
        "year": 2024,
        "run_rows": 110949,
        "compatible_rows": 759212,
        "positive_rows": 46630,
        "later_period_paths_resolved": False,
        "later_period_rows_read": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    prepared = {
        "source_manifest": source_manifest,
        "audit": {"factor_table": {"placeholder_cleanup": cleanup}},
    }
    assert runner._fit_source_manifest(prepared)["later_period_rows_read"] is False
    source_manifest["later_period_rows_read"] = True
    with pytest.raises(AssertionError, match="source phase claim changed"):
        runner._fit_source_manifest(prepared)


def test_fit_source_cannot_resolve_later_paths() -> None:
    source = inspect.getsource(runner.run_fit_only)
    assert "phase_source_paths" not in source
    assert "prepare_period_anchors" not in source
    score_source = inspect.getsource(runner.run_score_only)
    assert score_source.index("validate_pre_score_authorization") < score_source.index(
        "phase_source_paths"
    )
    assert score_source.index("assert_pristine_scoring_namespace") < score_source.index(
        "phase_source_paths"
    )


def test_source_has_no_execution_or_shadow_surface() -> None:
    source = MODULE_PATH.read_text().lower()
    assert "place_order" not in source
    assert "broker" not in source
    assert "shadow_validation" not in source
