from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stocker_research.frozen_named_loop_t0_execution.historical import (
    build_payoff_ledger,
    build_source_populations,
    load_and_verify_contract,
    load_provider_frames,
    reconstruct_historical_2025,
    verify_2023_archive,
)

WORK = Path("research/slrno-v2/20260714-regime-loop-handoff/work")
CONTRACT = WORK / "contracts/20260717-frozen-named-loop-t0-execution-realism-v1.json"


@pytest.fixture(scope="module")
def historical() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contract, _, _ = load_and_verify_contract(CONTRACT)
    source = build_source_populations(contract, contract_path=CONTRACT)
    source_2025 = source.loc[source["period"].eq(2025)].copy()
    providers = load_provider_frames(contract, source_2025["symbol"], contract_path=CONTRACT)
    reconstructed = reconstruct_historical_2025(source_2025, providers)
    payoff = build_payoff_ledger(reconstructed)
    return contract, source, reconstructed, payoff


def test_versioned_contract_and_every_frozen_input_hash_verify(
    historical: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    contract, _, _, _ = historical

    assert contract["contract_id"] == "20260717-frozen-named-loop-t0-execution-realism-v1"
    assert contract["adverse_entry_envelope"]["primary"] == "F10"  # type: ignore[index]
    assert contract["safety"]["research_only"] is True  # type: ignore[index]
    assert contract["safety"]["execution_enabled"] is False  # type: ignore[index]


def test_historical_named_and_control_source_counts_are_exact(
    historical: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, source, _, _ = historical
    counts = source.groupby(["period", "family"]).size().to_dict()

    assert counts == {
        (2023, "cycle_04|state_2"): 8,
        (2023, "cycle_04|state_4"): 132,
        (2023, "cycle_07|state_5"): 722,
        (2023, "cycle_07|state_6"): 331,
        (2025, "cycle_04|state_2"): 6,
        (2025, "cycle_04|state_4"): 96,
        (2025, "cycle_07|state_5"): 713,
        (2025, "cycle_07|state_6"): 296,
    }
    assert (
        source.loc[source["classification"].eq("named"), "family"]
        .isin(["cycle_04|state_4", "cycle_07|state_5"])
        .all()
    )
    assert (
        source.loc[source["classification"].eq("control"), "family"]
        .isin(["cycle_04|state_2", "cycle_07|state_6"])
        .all()
    )


def test_original_t0_rule_reconstructs_every_2025_reference_exactly(
    historical: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, reconstructed, _ = historical

    assert len(reconstructed) == 1111
    assert reconstructed["reference_reconstruction_exact"].all()
    assert reconstructed["terminal_reconstruction_exact"].all()
    assert reconstructed["direction"].isin([-1, 1]).all()
    assert (
        reconstructed["original_terminal_timestamp"]
        .eq(reconstructed["anchor_timestamp"] + pd.Timedelta(minutes=125))
        .all()
    )


def test_fill_achievability_is_separated_without_silent_execution_assumption(
    historical: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, reconstructed, _ = historical
    counts = reconstructed.groupby(["classification", "fill_evidence_classification"]).size()

    assert counts.to_dict() == {
        ("control", "BOUNDED_BUT_NOT_EXACT"): 269,
        ("control", "GAP_FILL_OBSERVABLE"): 33,
        ("named", "BOUNDED_BUT_NOT_EXACT"): 755,
        ("named", "GAP_FILL_OBSERVABLE"): 54,
    }
    bounded = reconstructed["fill_evidence_classification"].eq("BOUNDED_BUT_NOT_EXACT")
    assert reconstructed.loc[bounded, "signal_known_timestamp"].isna().all()
    assert (
        reconstructed.loc[bounded, "signal_fill_time_status"]
        .eq("SIGNAL_OR_FILL_TIME_AMBIGUOUS")
        .all()
    )


@pytest.mark.parametrize(
    ("family", "fill_model", "count", "expected_total"),
    [
        ("cycle_04|state_4", "F0", 96, 2951.93135764),
        ("cycle_04|state_4", "F10", 96, 1989.999289),
        ("cycle_07|state_5", "F0", 713, 12147.8051536),
        ("cycle_07|state_5", "F10", 713, 5007.522016),
        ("cycle_04|state_2", "F0", 6, -196.758203506),
        ("cycle_07|state_6", "F0", 296, -9977.34700015),
    ],
)
def test_historical_family_payoffs_reconcile_exactly(
    historical: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
    family: str,
    fill_model: str,
    count: int,
    expected_total: float,
) -> None:
    _, _, _, payoff = historical
    selected = payoff.loc[payoff["family"].eq(family) & payoff["fill_model"].eq(fill_model)]

    assert len(selected) == count
    assert selected["net_payoff_bps"].sum() == pytest.approx(expected_total, abs=1e-6)


def test_full_2025_named_reference_and_808_latency_common_reference_both_reconcile(
    historical: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, _, payoff = historical
    named_f0 = payoff.loc[payoff["classification"].eq("named") & payoff["fill_model"].eq("F0")]
    latency = pd.read_parquet(
        WORK / "artifacts/20260716-fixed-one-bar-entry-latency-v1/primary/"
        "exact_paired_t0_t1_ledger.parquet"
    )

    assert len(named_f0) == 809
    assert named_f0["net_payoff_bps"].sum() == pytest.approx(15099.7365112, abs=1e-7)
    assert len(latency) == 808
    assert latency["t0_net_return_bps"].sum() == pytest.approx(15087.316481277237, abs=1e-8)


def test_all_fill_models_keep_the_same_opportunity_direction_and_terminal(
    historical: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, _, payoff = historical
    grouped = payoff.groupby("opportunity_id")

    assert grouped.size().eq(5).all()
    assert grouped["direction"].nunique().eq(1).all()
    assert grouped["original_terminal_timestamp"].nunique().eq(1).all()
    assert grouped["terminal_price"].nunique().eq(1).all()
    assert grouped["reference_entry_price"].nunique().eq(1).all()


def test_restored_2023_archive_requires_every_registered_hash(tmp_path: Path) -> None:
    manifest = json.loads(
        (
            WORK / "artifacts/20260713-loop-payoff-phase-path-v1/exact_rerun/source_hashes.json"
        ).read_text()
    )
    candidate = tmp_path / "symbol=AAL" / "timeframe=5m"
    candidate.mkdir(parents=True)
    (candidate / "data.parquet").write_bytes(b"fresh-but-nonmatching")

    result = verify_2023_archive(tmp_path, manifest)

    assert result["status"] == "unavailable_hash_mismatch_or_missing"
    assert result["matched_symbols"] == []
    assert result["all_registered_hashes_match"] is False
