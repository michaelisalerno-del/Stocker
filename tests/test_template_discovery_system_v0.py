from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.personality_discovery_v0 import add_discovery_features
from stocker_research.template_discovery_system_v0 import (
    DiscoveryAtom,
    TemplateDiscoveryEventInput,
    TemplateDiscoverySystemConfig,
    _build_family_replay_transitions,
    _build_transitions,
    _candidate_masks,
    _generate_atoms,
    _load_event_rows,
    _loop_regime_occupancy_reports,
    _run_family_r_replay,
    run_template_discovery_system_lab,
)


def _event_row(
    *,
    symbol: str,
    timestamp: str,
    event_state: str,
    session_open_distance: float,
    relative_cumulative_volume: float,
    forward_return: float,
    bar_index: int = 10,
) -> dict[str, object]:
    abs_forward = abs(forward_return)
    if forward_return < 0.0:
        mfe = abs_forward * 0.3
        mae = -abs_forward * 1.2
    else:
        mfe = abs_forward * 1.2
        mae = -abs_forward * 0.3
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "session_date": timestamp[:10],
        "event_state": event_state,
        "distance_from_vwap_pct": session_open_distance / 2.0,
        "distance_from_session_open_pct": session_open_distance,
        "distance_from_opening_range_mid_pct": session_open_distance / 2.0,
        "distance_from_opening_range_low_pct": abs(session_open_distance) / 2.0,
        "distance_from_opening_range_high_pct": -abs(session_open_distance),
        "distance_from_session_low_pct": 0.004,
        "distance_from_session_high_pct": -0.018,
        "distance_from_recent_low_pct": 0.006,
        "distance_from_recent_high_pct": -0.016,
        "rolling_intraday_range_pct": 0.012,
        "compression_zscore": -0.65,
        "range_zscore": -0.55,
        "return_zscore": 0.1,
        "directional_efficiency_3": 0.30,
        "directional_efficiency_6": 0.35,
        "directional_efficiency_12": 0.40,
        "relative_volume_at_bar_index": relative_cumulative_volume,
        "relative_cumulative_volume": relative_cumulative_volume,
        "bar_index_in_session": bar_index,
        "close_location_value": 0.35,
        "upper_wick_pct_of_range": 0.25,
        "lower_wick_pct_of_range": 0.20,
        "bar_return": forward_return / 4.0,
        "prior_3_bar_return": -0.002,
        "prior_6_bar_return": -0.004,
        "prior_12_bar_return": -0.006,
        "vwap_cross_count_12": 1,
        "range_cross_count_12": 1,
        "pullback_depth_from_recent_high": 0.01,
        "reclaim_from_recent_low": 0.004,
        "forward_6_bar_return": forward_return,
        "forward_9_bar_return": forward_return,
        "forward_12_bar_return": forward_return,
        "forward_24_bar_return": forward_return,
        "forward_6_bar_mfe": mfe,
        "forward_6_bar_mae": mae,
        "forward_9_bar_mfe": mfe,
        "forward_9_bar_mae": mae,
        "forward_12_bar_mfe": mfe,
        "forward_12_bar_mae": mae,
        "forward_24_bar_mfe": mfe,
        "forward_24_bar_mae": mae,
    }


def _add_transition(
    rows: list[dict[str, object]],
    *,
    symbol: str,
    source_timestamp: str,
    source_state: str,
    next_state: str,
    inside_container: bool,
    next_forward_return: float,
) -> None:
    source_time = pd.Timestamp(source_timestamp, tz="UTC")
    next_time = source_time + pd.Timedelta(minutes=5)
    source_distance = -0.010 if inside_container else 0.010
    source_volume = 0.40 if inside_container else 1.20
    rows.append(
        _event_row(
            symbol=symbol,
            timestamp=source_time.isoformat(),
            event_state=source_state,
            session_open_distance=source_distance,
            relative_cumulative_volume=source_volume,
            forward_return=0.0,
        )
    )
    rows.append(
        _event_row(
            symbol=symbol,
            timestamp=next_time.isoformat(),
            event_state=next_state,
            session_open_distance=source_distance,
            relative_cumulative_volume=source_volume,
            forward_return=next_forward_return,
            bar_index=11,
        )
    )


def _write_event_surface(tmp_path: Path, name: str, *, residual: bool = False) -> Path:
    rows: list[dict[str, object]] = []
    periods = [
        ("2024-08-05T14:30:00Z", "fresh"),
        ("2025-08-05T14:30:00Z", "saved"),
    ]
    if residual:
        periods = [("2025-09-05T14:30:00Z", "residual")]

    for index, (timestamp, _label) in enumerate(periods, start=1):
        _add_transition(
            rows,
            symbol=f"{name.upper()}A{index}",
            source_timestamp=timestamp,
            source_state="controlled_pullback_after_bullish_impulse",
            next_state="failed_bullish_impulse_recoil",
            inside_container=True,
            next_forward_return=-0.012,
        )
        _add_transition(
            rows,
            symbol=f"{name.upper()}B{index}",
            source_timestamp=timestamp,
            source_state="controlled_pullback_after_bullish_impulse",
            next_state="failed_bullish_impulse_recoil",
            inside_container=False,
            next_forward_return=0.012,
        )
        _add_transition(
            rows,
            symbol=f"{name.upper()}C{index}",
            source_timestamp=timestamp,
            source_state="failed_bounce_active_liquidation",
            next_state="liquidation_failed_low_reclaim",
            inside_container=True,
            next_forward_return=0.010,
        )
        _add_transition(
            rows,
            symbol=f"{name.upper()}D{index}",
            source_timestamp=timestamp,
            source_state="failed_bounce_active_liquidation",
            next_state="liquidation_failed_low_reclaim",
            inside_container=False,
            next_forward_return=-0.010,
        )

    path = tmp_path / f"{name}_event_rows.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_residual_ex_surface_excludes_primary_smid_symbols(tmp_path: Path) -> None:
    smid_rows = [
        _event_row(
            symbol="SHARED",
            timestamp="2025-08-05T14:30:00Z",
            event_state="failed_bounce_active_liquidation",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.4,
            forward_return=0.0,
        )
    ]
    residual_rows = [
        _event_row(
            symbol="SHARED",
            timestamp="2025-08-05T14:30:00Z",
            event_state="failed_bounce_active_liquidation",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.4,
            forward_return=0.0,
        ),
        _event_row(
            symbol="UNIQUE",
            timestamp="2025-08-05T14:30:00Z",
            event_state="failed_bounce_active_liquidation",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.4,
            forward_return=0.0,
        ),
    ]
    smid_path = tmp_path / "smid.csv"
    residual_path = tmp_path / "residual.csv"
    pd.DataFrame(smid_rows).to_csv(smid_path, index=False)
    pd.DataFrame(residual_rows).to_csv(residual_path, index=False)

    loaded = _load_event_rows(
        (
            TemplateDiscoveryEventInput("smid24", smid_path),
            TemplateDiscoveryEventInput("residual_ex_smid24", residual_path),
        )
    )

    residual_symbols = set(
        loaded.loc[loaded["surface"].eq("residual_ex_smid24"), "symbol"].astype(str)
    )

    assert residual_symbols == {"UNIQUE"}


def test_atom_discovery_keeps_cumulative_volume_numeric_atom() -> None:
    rows = []
    for index in range(40):
        rows.append(
            {
                "surface": "smid24",
                "symbol": f"S{index:02d}",
                "month": "2025-08",
                "loop_id": f"loop_{index:02d}",
                "source_auction_session_open_location": f"session_loc_{index}",
                "source_auction_opening_mid_location": f"opening_loc_{index}",
                "source_vwap_side_regime": f"vwap_{index}",
                "source_auction_current_location": f"current_loc_{index}",
                "source_range_regime": f"range_{index}",
                "source_relative_volume_regime": "low_relative_volume",
                "source_volume_x_vwap_regime": "low_relative_volume|below",
                "source_cross_stock_same_direction_bucket": "same_direction_elsewhere",
                "source_relative_cumulative_volume": 0.40 if index < 20 else 1.20,
            }
        )
    transitions = pd.DataFrame(rows)

    atoms, _scorecard = _generate_atoms(
        transitions,
        TemplateDiscoverySystemConfig(
            min_atom_rows=1,
            max_atoms=32,
        ),
    )

    assert any(
        atom.feature == "source_relative_cumulative_volume" and atom.operator == "<="
        for atom in atoms
    )


def test_atom_discovery_skips_overbroad_numeric_atoms() -> None:
    rows = pd.DataFrame(
        [
            {
                "surface": "smid24",
                "symbol": f"S{index:02d}",
                "month": "2025-08",
                "loop_id": f"loop_{index % 4}",
                "source_distance_from_session_high_pct": -0.01,
                "source_auction_session_open_location": "below"
                if index < 20
                else "above",
            }
            for index in range(40)
        ]
    )

    atoms, _scorecard = _generate_atoms(
        rows,
        TemplateDiscoverySystemConfig(
            min_atom_rows=1,
            max_atoms=32,
        ),
    )

    overbroad_atom = "near_session_high:source_distance_from_session_high_pct<=0.0016"
    assert all(atom.atom_id != overbroad_atom for atom in atoms)


def test_routing_admission_prefers_participation_context_combinations() -> None:
    rows = []
    for index in range(40):
        rows.append(
            {
                "surface": "smid24",
                "symbol": f"S{index:02d}",
                "month": "2025-08",
                "loop_id": f"loop_{index % 4}",
                "source_auction_session_open_location": "below",
                "source_bar_index_bucket": "morning",
                "source_relative_cumulative_volume": 0.40 if index < 12 else 1.20,
            }
        )
    atoms = [
        DiscoveryAtom(
            atom_id="source_auction_session_open_location==below",
            axis="location",
            feature="source_auction_session_open_location",
            operator="==",
            value="below",
            expression="source_auction_session_open_location == below",
        ),
        DiscoveryAtom(
            atom_id="source_bar_index_bucket==morning",
            axis="tempo",
            feature="source_bar_index_bucket",
            operator="==",
            value="morning",
            expression="source_bar_index_bucket == morning",
        ),
        DiscoveryAtom(
            atom_id="low_cumulative_volume:source_relative_cumulative_volume<=0.5631",
            axis="participation",
            feature="source_relative_cumulative_volume",
            operator="<=",
            value=0.5631,
            expression="low_cumulative_volume (source_relative_cumulative_volume <= 0.5631)",
        ),
    ]

    _routed, scorecard = _candidate_masks(
        pd.DataFrame(rows),
        atoms,
        TemplateDiscoverySystemConfig(
            max_containers_to_route=2,
            min_container_rows=1,
            max_single_symbol_share=1.0,
        ),
    )
    selected = scorecard[scorecard["selected_for_routing"]]

    assert selected["expression"].str.contains("low_cumulative_volume").any()


def test_routing_admission_diversifies_participation_families() -> None:
    rows = []
    for index in range(40):
        rows.append(
            {
                "surface": "smid24",
                "symbol": f"S{index:02d}",
                "month": "2025-08",
                "loop_id": f"loop_{index % 4}",
                "source_auction_session_open_location": "below",
                "source_auction_opening_mid_location": "below",
                "source_relative_cumulative_volume": 0.40
                if index < 12
                else 0.60
                if index < 24
                else 1.20,
            }
        )
    atoms = [
        DiscoveryAtom(
            atom_id="source_auction_session_open_location==below",
            axis="location",
            feature="source_auction_session_open_location",
            operator="==",
            value="below",
            expression="source_auction_session_open_location == below",
        ),
        DiscoveryAtom(
            atom_id="source_auction_opening_mid_location==below",
            axis="location",
            feature="source_auction_opening_mid_location",
            operator="==",
            value="below",
            expression="source_auction_opening_mid_location == below",
        ),
        DiscoveryAtom(
            atom_id="normal_or_low_cumulative_volume:source_relative_cumulative_volume<=0.6648",
            axis="participation",
            feature="source_relative_cumulative_volume",
            operator="<=",
            value=0.6648,
            expression=(
                "normal_or_low_cumulative_volume "
                "(source_relative_cumulative_volume <= 0.6648)"
            ),
        ),
        DiscoveryAtom(
            atom_id="low_cumulative_volume:source_relative_cumulative_volume<=0.5631",
            axis="participation",
            feature="source_relative_cumulative_volume",
            operator="<=",
            value=0.5631,
            expression="low_cumulative_volume (source_relative_cumulative_volume <= 0.5631)",
        ),
    ]

    _routed, scorecard = _candidate_masks(
        pd.DataFrame(rows),
        atoms,
        TemplateDiscoverySystemConfig(
            max_containers_to_route=2,
            min_container_rows=1,
            max_single_symbol_share=1.0,
        ),
    )
    selected_expressions = scorecard.loc[
        scorecard["selected_for_routing"], "expression"
    ].tolist()

    assert any("normal_or_low_cumulative_volume" in item for item in selected_expressions)
    assert any(" AND low_cumulative_volume (" in item for item in selected_expressions)


def test_loop_regime_reports_describe_source_mixed_and_transition_regimes() -> None:
    transitions = pd.DataFrame(
        [
            {
                "surface": "smid24",
                "symbol": "AAA",
                "month": "2025-08",
                "loop_id": "loop_a",
                "source_event_quality_regime": "mixed_event_quality",
                "event_quality_regime": "strong_event_quality",
                "source_compression_x_efficiency_regime": "compressed|mixed_efficiency",
                "compression_x_efficiency_regime": "expanded|directional_efficiency",
            },
            {
                "surface": "smid24",
                "symbol": "BBB",
                "month": "2025-08",
                "loop_id": "loop_a",
                "source_event_quality_regime": "mixed_event_quality",
                "event_quality_regime": "strong_event_quality",
                "source_compression_x_efficiency_regime": "compressed|mixed_efficiency",
                "compression_x_efficiency_regime": "expanded|directional_efficiency",
            },
            {
                "surface": "smid24",
                "symbol": "CCC",
                "month": "2025-08",
                "loop_id": "loop_b",
                "source_event_quality_regime": "weak_event_quality",
                "event_quality_regime": "mixed_event_quality",
                "source_compression_x_efficiency_regime": "expanded|directional_efficiency",
                "compression_x_efficiency_regime": "compressed|mixed_efficiency",
            },
        ]
    )

    regimes, mixed, transitions_report = _loop_regime_occupancy_reports(
        transitions,
        TemplateDiscoverySystemConfig(min_loop_regime_rows=1),
    )

    assert (
        "source_event_quality_regime",
        "mixed_event_quality",
    ) in set(zip(regimes["regime_feature"], regimes["regime_value"], strict=False))
    assert (
        "source_compression_x_efficiency_regime",
        "compressed|mixed_efficiency",
    ) in set(zip(mixed["regime_feature"], mixed["regime_value"], strict=False))
    assert (
        "source_event_quality_regime->event_quality_regime",
        "mixed_event_quality -> strong_event_quality",
    ) in set(
        zip(
            transitions_report["regime_feature"],
            transitions_report["regime_value"],
            strict=False,
        )
    )


def _config(mode: str) -> TemplateDiscoverySystemConfig:
    return TemplateDiscoverySystemConfig(
        mode=mode,
        min_behavior_loop_rows=1,
        min_behavior_loop_transition_rate=0.0,
        min_behavior_loop_symbols=1,
        min_behavior_loop_months=1,
        min_behavior_loop_split_rows=1,
        min_loop_regime_rows=1,
        min_loop_refinement_rows=1,
        min_atom_rows=1,
        min_container_rows=1,
        min_loop_inside_rows=1,
        min_loop_outside_rows=1,
        route_lift_bar=0.10,
        horizons=(6,),
        stop_models=("fixed_50bps",),
        target_r_multiples=(1.0,),
        cost_bps_values=(0.0, 5.0),
    )


def test_clean_slate_discovers_behavior_loops_before_routing(tmp_path: Path) -> None:
    smid = _write_event_surface(tmp_path, "smid")
    residual = _write_event_surface(tmp_path, "residual", residual=True)

    result = run_template_discovery_system_lab(
        input_event_rows=(
            TemplateDiscoveryEventInput("smid24", smid),
            TemplateDiscoveryEventInput("residual_ex_smid24", residual),
        ),
        output_dir=tmp_path / "out",
        config=_config("container-routing"),
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    loops = pd.read_csv(result.behavior_loop_scorecard_csv_path)
    routes = pd.read_csv(result.loop_routing_detail_csv_path)
    candidate_loop_ids = set(loops.loc[loops["candidate_behavior_loop"], "loop_id"])

    assert summary["discovery_ladder"][0] == "behavior_loop_discovery"
    assert summary["saved_rules_used"] is False
    assert summary["behavior_loop_count"] >= 2
    assert summary["candidate_behavior_loop_count"] == len(candidate_loop_ids)
    assert candidate_loop_ids == {
        "controlled_pullback_after_bullish_impulse__to__failed_bullish_impulse_recoil",
        "failed_bounce_active_liquidation__to__liquidation_failed_low_reclaim",
    }
    assert set(routes["loop_id"]).issubset(candidate_loop_ids)


def test_clean_slate_container_routing_generates_reports_without_saved_inputs(
    tmp_path: Path,
) -> None:
    smid = _write_event_surface(tmp_path, "smid")
    residual = _write_event_surface(tmp_path, "residual", residual=True)

    result = run_template_discovery_system_lab(
        input_event_rows=(
            TemplateDiscoveryEventInput("smid24", smid),
            TemplateDiscoveryEventInput("residual_ex_smid24", residual),
        ),
        output_dir=tmp_path / "out",
        config=_config("container-routing"),
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    atoms = pd.read_csv(result.atom_scorecard_csv_path)
    containers = pd.read_csv(result.container_scorecard_csv_path)
    routes = pd.read_csv(result.loop_routing_detail_csv_path)
    regimes = pd.read_csv(result.loop_regime_occupancy_csv_path)
    mixed_regimes = pd.read_csv(result.loop_mixed_regime_occupancy_csv_path)
    transition_regimes = pd.read_csv(result.loop_transition_regime_occupancy_csv_path)
    c0_parent = pd.read_csv(result.c0_parent_readout_csv_path)
    b0_summary = pd.read_csv(result.b0_state_summary_csv_path)
    loop_refinement = pd.read_csv(result.loop_context_refinement_csv_path)
    loop_context_admissions = pd.read_csv(result.loop_context_admissions_csv_path)
    loop_context_blockers = pd.read_csv(result.loop_context_blockers_csv_path)

    assert summary["clean_slate"] is True
    assert summary["saved_rules_used"] is False
    assert summary["seed_report_used"] is False
    assert summary["research_only"] is True
    assert summary["live_ordering_enabled"] is False
    assert summary["order_placement"] == "disabled"
    assert summary["edge_claimed"] is False
    assert set(atoms["source"]) == {"generated"}
    assert set(containers["source"]) == {"generated"}
    assert "source_auction_session_open_location == below" in set(containers["expression"])
    assert "stable_route_count" in containers.columns
    assert "routing_score" in containers.columns
    assert not routes.empty
    assert set(routes["directional_side"]) == {"long", "short"}
    assert not c0_parent.empty
    assert set(c0_parent["stage"]) == {"fixed_c0_parent"}
    assert "b0_state" in b0_summary.columns or b0_summary.empty
    assert not loop_refinement.empty
    assert {"source_visible", "next_event_start"}.intersection(
        set(loop_refinement["visibility"])
    )
    assert set(loop_context_admissions["candidate_kind"]).issubset(
        {"admission_refinement"}
    )
    assert set(loop_context_blockers["candidate_kind"]).issubset(
        {"blocker_refinement"}
    )
    assert not regimes.empty
    assert not mixed_regimes.empty
    assert not transition_regimes.empty
    assert summary["reports"]["loop_regime_occupancy"] == "loop_regime_occupancy.csv"
    assert summary["reports"]["c0_parent_readout"] == "c0_parent_readout.csv"
    assert summary["reports"]["b0_state_summary"] == "b0_state_summary.csv"
    assert summary["reports"]["loop_context_refinement"] == "loop_context_refinement.csv"


def test_clean_slate_keeps_admissions_and_blockers_separate(tmp_path: Path) -> None:
    smid = _write_event_surface(tmp_path, "smid")
    residual = _write_event_surface(tmp_path, "residual", residual=True)

    result = run_template_discovery_system_lab(
        input_event_rows=(
            TemplateDiscoveryEventInput("smid24", smid),
            TemplateDiscoveryEventInput("residual_ex_smid24", residual),
        ),
        output_dir=tmp_path / "out",
        config=_config("family-directional-readout"),
    )

    admissions = pd.read_csv(result.admission_candidates_csv_path)
    blockers = pd.read_csv(result.blocker_candidates_csv_path)

    assert not admissions.empty
    assert not blockers.empty
    assert set(admissions["candidate_kind"]) == {"admission_candidate"}
    assert set(blockers["candidate_kind"]) == {"blocker_candidate"}
    assert set(admissions["candidate_id"]).isdisjoint(set(blockers["candidate_id"]))


def test_container_routing_can_be_bounded_for_large_reports(tmp_path: Path) -> None:
    smid = _write_event_surface(tmp_path, "smid")
    residual = _write_event_surface(tmp_path, "residual", residual=True)
    config = _config("container-routing")
    config = TemplateDiscoverySystemConfig(
        **{**config.__dict__, "max_containers_to_route": 1}
    )

    result = run_template_discovery_system_lab(
        input_event_rows=(
            TemplateDiscoveryEventInput("smid24", smid),
            TemplateDiscoveryEventInput("residual_ex_smid24", residual),
        ),
        output_dir=tmp_path / "out",
        config=config,
    )

    routes = pd.read_csv(result.loop_routing_detail_csv_path)

    assert routes["container_id"].nunique() == 1


def test_transition_builder_keeps_multiple_source_states_per_timestamp(
    tmp_path: Path,
) -> None:
    rows = [
        _event_row(
            symbol="SMIDA",
            timestamp="2025-08-05T14:30:00Z",
            event_state="controlled_pullback_after_bullish_impulse",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.4,
            forward_return=0.0,
        ),
        _event_row(
            symbol="SMIDA",
            timestamp="2025-08-05T14:30:00Z",
            event_state="failed_bounce_active_liquidation",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.4,
            forward_return=0.0,
        ),
        _event_row(
            symbol="SMIDA",
            timestamp="2025-08-05T14:35:00Z",
            event_state="failed_bullish_impulse_recoil",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.4,
            forward_return=-0.01,
        ),
        _event_row(
            symbol="SMIDA",
            timestamp="2025-08-05T14:35:00Z",
            event_state="liquidation_failed_low_reclaim",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.4,
            forward_return=0.01,
        ),
    ]
    path = tmp_path / "events.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    events = _load_event_rows((TemplateDiscoveryEventInput("smid24", path),))
    transitions = _build_transitions(events)

    assert len(transitions) == 2
    assert set(transitions["source_current_event_state"]) == {
        "controlled_pullback_after_bullish_impulse",
        "failed_bounce_active_liquidation",
    }


def test_r_replay_outputs_cost_stop_target_metrics_after_directional_support(
    tmp_path: Path,
) -> None:
    smid = _write_event_surface(tmp_path, "smid")
    residual = _write_event_surface(tmp_path, "residual", residual=True)

    result = run_template_discovery_system_lab(
        input_event_rows=(
            TemplateDiscoveryEventInput("smid24", smid),
            TemplateDiscoveryEventInput("residual_ex_smid24", residual),
        ),
        output_dir=tmp_path / "out",
        config=_config("r-replay"),
    )

    replay = pd.read_csv(result.replay_results_csv_path)
    decision = json.loads(result.decision_json_path.read_text(encoding="utf-8"))

    assert decision["decision"] == "continue_research_directional_supported"
    assert not replay.empty
    assert {"stop_model", "target_r", "cost_bps", "total_net_r", "mean_r", "median_r"}.issubset(
        replay.columns
    )
    assert set(replay["cost_bps"]) == {0.0, 5.0}


def test_family_r_replay_uses_fixed_c0_loop_family_and_final_close_scorecard(
    tmp_path: Path,
) -> None:
    smid = _write_event_surface(tmp_path, "smid")
    residual = _write_event_surface(tmp_path, "residual", residual=True)

    result = run_template_discovery_system_lab(
        input_event_rows=(
            TemplateDiscoveryEventInput("smid24", smid),
            TemplateDiscoveryEventInput("residual_ex_smid24", residual),
        ),
        output_dir=tmp_path / "out",
        config=_config("family-r-replay"),
    )

    scorecard = pd.read_csv(result.output_dir / "family_r_replay_scorecard.csv")
    sweep = pd.read_csv(result.output_dir / "family_r_replay_summary.csv")
    selected = pd.read_csv(result.output_dir / "family_r_replay_selected_events.csv")

    assert "l1_container_h6" in set(scorecard["candidate_id"])
    assert "selected_discovered_container" in set(selected["candidate_expression"])
    assert "final_close_total_r" in set(sweep.columns)
    assert "target_capped_total_r" in set(sweep.columns)
    assert scorecard["smid_fresh_final_close_total_r"].notna().any()


def test_family_r_replay_uses_source_excursions_for_old_transition_parity() -> None:
    rows = pd.DataFrame(
        [
            {
                "surface": "smid24",
                "symbol": "SMID",
                "session_date": "2024-08-05",
                "timestamp": pd.Timestamp("2024-08-05T14:35:00Z"),
                "month": "2024-08",
                "loop_id": (
                    "controlled_pullback_after_bullish_impulse"
                    "__to__"
                    "failed_bullish_impulse_recoil"
                ),
                "source_auction_session_open_location": "below",
                "source_relative_cumulative_volume": 0.40,
                "forward_6_bar_return": -0.002,
                "forward_6_bar_mfe": 0.0,
                "forward_6_bar_mae": 0.0,
                "source_forward_6_bar_mfe": 0.003,
                "source_forward_6_bar_mae": -0.003,
            }
        ]
    )

    summary, _scorecard, _cost_focus, _selected = _run_family_r_replay(
        rows,
        TemplateDiscoverySystemConfig(
            mode="family-r-replay",
            stop_models=("fixed_25bps",),
            target_r_multiples=(1.0,),
            cost_bps_values=(0.0,),
        ),
    )

    replay_row = summary[
        summary["candidate_id"].eq("l1_container_h6")
        & summary["surface"].eq("smid24")
        & summary["period"].eq("fresh_year")
        & summary["stop_model"].eq("fixed_25bps")
        & summary["target_r"].eq(1.0)
    ].iloc[0]
    assert replay_row["final_close_total_r"] == -1.0


def test_family_replay_transitions_filter_family_loops_before_target_priority() -> None:
    rows = pd.DataFrame(
        [
            _event_row(
                symbol="SMID",
                timestamp="2025-08-05T14:30:00Z",
                event_state="controlled_pullback_after_bullish_impulse",
                session_open_distance=-0.01,
                relative_cumulative_volume=0.4,
                forward_return=0.0,
            ),
            _event_row(
                symbol="SMID",
                timestamp="2025-08-05T14:35:00Z",
                event_state="liquidation_failed_low_reclaim",
                session_open_distance=-0.01,
                relative_cumulative_volume=0.4,
                forward_return=0.01,
            ),
            _event_row(
                symbol="SMID",
                timestamp="2025-08-05T14:35:00Z",
                event_state="failed_bullish_impulse_recoil",
                session_open_distance=-0.01,
                relative_cumulative_volume=0.4,
                forward_return=-0.01,
            ),
        ]
    )
    rows["surface"] = "smid24"
    rows["input_event_rows_path"] = "synthetic"

    transitions = _build_family_replay_transitions(
        add_discovery_features(rows),
        TemplateDiscoverySystemConfig(mode="family-r-replay"),
    )

    assert set(transitions["loop_id"]) == {
        "controlled_pullback_after_bullish_impulse"
        "__to__"
        "failed_bullish_impulse_recoil"
    }
    assert transitions["forward_6_bar_return"].iloc[0] == -0.01


def _write_frozen_combo_dir(tmp_path: Path) -> Path:
    combo_dir = tmp_path / "frozen_combo"
    combo_dir.mkdir()
    rows_by_file = {
        "fixed_next_consistent_trades.csv": [
            {
                "component": "current_fixed_next",
                "rule_id": "fixed_next_confirmation_choppy_open_down_source_h6",
                "symbol": "AAA",
                "timestamp": "2025-08-05T14:30:00Z",
                "month": "2025-08",
                "net_r": 1.5,
                "stop_hit": False,
                "target_hit": True,
                "ambiguous": False,
            },
            {
                "component": "current_fixed_next",
                "rule_id": "fixed_next_confirmation_fast_failed_reclaim_short_source_h6",
                "symbol": "CCC",
                "timestamp": "2025-08-06T14:30:00Z",
                "month": "2025-08",
                "net_r": -0.5,
                "stop_hit": True,
                "target_hit": False,
                "ambiguous": False,
            },
        ],
        "omitted_saved_loop_consistent_trades.csv": [
            {
                "component": "omitted_saved_loop",
                "rule_id": "extended_directional_impulse_pullback_confirmation_note_v0",
                "symbol": "AAA",
                "timestamp": "2025-08-05T14:30:00Z",
                "month": "2025-08",
                "net_r": 99.0,
                "stop_hit": False,
                "target_hit": True,
                "ambiguous": False,
            },
            {
                "component": "omitted_saved_loop",
                "rule_id": "low_volume_reclaim_repair_behavior_loop_split_v0",
                "symbol": "BBB",
                "timestamp": "2025-08-05T14:35:00Z",
                "month": "2025-08",
                "net_r": 2.0,
                "stop_hit": False,
                "target_hit": True,
                "ambiguous": False,
            },
        ],
        "strict_exact_frozen_addon_consistent_trades.csv": [
            {
                "component": "strict_other_loop_addon",
                "rule_id": (
                    "controlled_pullback_after_bullish_impulse"
                    "__to__failed_bounce_active_liquidation"
                ),
                "symbol": "BBB",
                "timestamp": "2025-08-05T14:35:00Z",
                "month": "2025-08",
                "net_r": 77.0,
                "stop_hit": False,
                "target_hit": True,
                "ambiguous": False,
            },
            {
                "component": "strict_other_loop_addon",
                "rule_id": "failed_bullish_impulse_recoil__to__failed_open_down_continuation",
                "symbol": "DDD",
                "timestamp": "2025-09-05T14:30:00Z",
                "month": "2025-09",
                "net_r": -1.0,
                "stop_hit": True,
                "target_hit": False,
                "ambiguous": False,
            },
        ],
    }
    for filename, rows in rows_by_file.items():
        pd.DataFrame(rows).to_csv(combo_dir / filename, index=False)
    pd.DataFrame(
        [
            {
                "component": "current_fixed_next",
                "rule_id": "fixed_next_confirmation_choppy_open_down_source_h6",
                "condition": "template == choppy_open_down",
            }
        ]
    ).to_csv(combo_dir / "frozen_candidate_book.csv", index=False)
    return combo_dir


def test_frozen_combo_replay_preserves_frozen_priority_without_clean_slate_claim(
    tmp_path: Path,
) -> None:
    combo_dir = _write_frozen_combo_dir(tmp_path)

    result = run_template_discovery_system_lab(
        input_event_rows=(),
        output_dir=tmp_path / "out",
        config=TemplateDiscoverySystemConfig(
            mode="frozen-combo-replay",
            frozen_combo_dir=combo_dir,
        ),
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    trades = pd.read_csv(result.output_dir / "frozen_combo_exact_dedupe_trades.csv")
    combo_summary = pd.read_csv(result.output_dir / "frozen_combo_summary.csv")

    assert summary["clean_slate"] is False
    assert summary["saved_rules_used"] is True
    assert summary["seed_report_used"] is True
    assert summary["frozen_combo_component_row_count"] == 6
    assert summary["frozen_combo_trade_count"] == 4
    assert summary["frozen_combo_exact_overlap_count"] == 2
    assert combo_summary.iloc[0]["total_r"] == 2.0
    assert set(trades["rule_id"]) == {
        "fixed_next_confirmation_choppy_open_down_source_h6",
        "fixed_next_confirmation_fast_failed_reclaim_short_source_h6",
        "low_volume_reclaim_repair_behavior_loop_split_v0",
        "failed_bullish_impulse_recoil__to__failed_open_down_continuation",
    }


def test_template_component_selection_scores_and_selects_components_by_code(
    tmp_path: Path,
) -> None:
    combo_dir = _write_frozen_combo_dir(tmp_path)

    result = run_template_discovery_system_lab(
        input_event_rows=(),
        output_dir=tmp_path / "out",
        config=TemplateDiscoverySystemConfig(
            mode="template-component-selection",
            component_candidate_dir=combo_dir,
            min_component_candidate_rows=1,
            min_component_total_r=0.0,
            max_component_negative_months=0,
            max_component_single_symbol_share=1.0,
        ),
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    scorecard = pd.read_csv(result.output_dir / "component_candidate_scorecard.csv")
    selected_book = pd.read_csv(result.output_dir / "selected_candidate_book.csv")
    rejected = pd.read_csv(result.output_dir / "rejected_component_candidates.csv")
    trades = pd.read_csv(result.output_dir / "selected_combo_exact_dedupe_trades.csv")
    combo_summary = pd.read_csv(result.output_dir / "selected_combo_summary.csv")

    assert summary["automated_component_selection"] is True
    assert summary["selected_component_candidate_count"] == 4
    assert summary["rejected_component_candidate_count"] == 2
    assert summary["selected_component_row_count_before_dedupe"] == 4
    assert summary["selected_combo_trade_count"] == 2
    assert summary["selected_combo_exact_overlap_count"] == 2
    assert combo_summary.iloc[0]["total_r"] == 3.5
    assert set(scorecard["selected"].astype(bool)) == {True, False}
    assert set(selected_book["rule_id"]) == {
        "fixed_next_confirmation_choppy_open_down_source_h6",
        "extended_directional_impulse_pullback_confirmation_note_v0",
        "low_volume_reclaim_repair_behavior_loop_split_v0",
        (
            "controlled_pullback_after_bullish_impulse"
            "__to__failed_bounce_active_liquidation"
        ),
    }
    assert set(rejected["rule_id"]) == {
        "fixed_next_confirmation_fast_failed_reclaim_short_source_h6",
        "failed_bullish_impulse_recoil__to__failed_open_down_continuation",
    }
    assert set(trades["rule_id"]) == {
        "fixed_next_confirmation_choppy_open_down_source_h6",
        "low_volume_reclaim_repair_behavior_loop_split_v0",
    }


def _write_frozen_transfer_event_surface(tmp_path: Path) -> Path:
    rows = [
        _event_row(
            symbol="PRIOR1",
            timestamp="2025-08-04T14:30:00Z",
            event_state="failed_bullish_impulse_recoil",
            session_open_distance=-0.02,
            relative_cumulative_volume=0.5,
            forward_return=-0.01,
            bar_index=12,
        ),
        _event_row(
            symbol="PRIOR2",
            timestamp="2025-08-04T14:30:00Z",
            event_state="liquidation_failed_low_reclaim",
            session_open_distance=-0.02,
            relative_cumulative_volume=0.5,
            forward_return=-0.01,
            bar_index=12,
        ),
        _event_row(
            symbol="FAST",
            timestamp="2025-08-05T14:30:00Z",
            event_state="failed_open_down_continuation",
            session_open_distance=-0.04,
            relative_cumulative_volume=0.5,
            forward_return=-0.02,
            bar_index=12,
        ),
        _event_row(
            symbol="FAST",
            timestamp="2025-08-05T14:30:00Z",
            event_state="slow_snapback_after_dip",
            session_open_distance=-0.04,
            relative_cumulative_volume=0.5,
            forward_return=0.02,
            bar_index=12,
        ),
        _event_row(
            symbol="FAST",
            timestamp="2025-08-05T14:35:00Z",
            event_state="liquidation_failed_low_reclaim",
            session_open_distance=-0.04,
            relative_cumulative_volume=0.5,
            forward_return=-0.01,
            bar_index=13,
        ),
        _event_row(
            symbol="CHOP",
            timestamp="2025-08-05T15:00:00Z",
            event_state="failed_open_down_continuation",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.5,
            forward_return=-0.015,
            bar_index=18,
        ),
        _event_row(
            symbol="CHOP",
            timestamp="2025-08-05T15:05:00Z",
            event_state="failed_bounce_active_liquidation",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.5,
            forward_return=-0.01,
            bar_index=19,
        ),
        _event_row(
            symbol="NOCHOP",
            timestamp="2025-08-05T15:00:00Z",
            event_state="failed_open_down_continuation",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.5,
            forward_return=-0.015,
            bar_index=18,
        ),
        _event_row(
            symbol="NOCHOP",
            timestamp="2025-08-05T15:05:00Z",
            event_state="failed_bounce_active_liquidation",
            session_open_distance=-0.01,
            relative_cumulative_volume=0.5,
            forward_return=-0.01,
            bar_index=19,
        ),
    ]
    for row in rows:
        if row["symbol"] == "CHOP":
            row["compression_zscore"] = -0.10
        if row["symbol"] == "NOCHOP":
            row["compression_zscore"] = -0.65
    path = tmp_path / "frozen_transfer_events.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_frozen_template_transfer_replay_rematerializes_source_context(
    tmp_path: Path,
) -> None:
    event_rows = _write_frozen_transfer_event_surface(tmp_path)

    result = run_template_discovery_system_lab(
        input_event_rows=(TemplateDiscoveryEventInput("synthetic_smid", event_rows),),
        output_dir=tmp_path / "out",
        config=TemplateDiscoverySystemConfig(mode="frozen-template-transfer-replay"),
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    all_rows = pd.read_csv(result.output_dir / "frozen_template_transfer_all_rows.csv")
    audit = pd.read_csv(result.output_dir / "frozen_template_transfer_template_audit.csv")

    assert summary["clean_slate"] is False
    assert summary["saved_rules_used"] is True
    assert summary["frozen_template_transfer_all_row_count"] == 2
    assert summary["frozen_template_transfer_trade_count"] == 2
    assert set(all_rows["rule_id"]) == {
        "fixed_next_confirmation_choppy_open_down_source_h6",
        "fixed_next_confirmation_fast_failed_reclaim_short_source_h6",
    }
    fast = all_rows[
        all_rows["rule_id"].eq(
            "fixed_next_confirmation_fast_failed_reclaim_short_source_h6"
        )
    ].iloc[0]
    assert fast["symbol"] == "FAST"
    assert fast["event_state"] == "failed_open_down_continuation"
    assert fast["fixed_next_target_event_state"] == "liquidation_failed_low_reclaim"
    assert fast["fixed_next_confirmation_timestamp"].startswith("2025-08-05 14:35")
    assert "broad_failed_recoil_event_share_prior" in all_rows.columns
    assert "b0_raw_state" in all_rows.columns
    assert audit["missing_features"].fillna("").eq("").all()


def test_template_discovery_system_cli_frozen_template_transfer_replay_smoke(
    tmp_path: Path,
) -> None:
    event_rows = _write_frozen_transfer_event_surface(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "template-discovery-system",
            "--input-event-rows",
            f"synthetic_smid={event_rows}",
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--mode",
            "frozen-template-transfer-replay",
        ],
    )

    assert result.exit_code == 0, result.output


def test_template_discovery_system_cli_template_component_selection_smoke(
    tmp_path: Path,
) -> None:
    combo_dir = _write_frozen_combo_dir(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "template-discovery-system",
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--mode",
            "template-component-selection",
            "--component-candidate-dir",
            str(combo_dir),
            "--min-component-candidate-rows",
            "1",
            "--min-component-total-r",
            "0",
            "--max-component-negative-months",
            "0",
            "--max-component-single-symbol-share",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output


def test_template_discovery_system_cli_frozen_combo_replay_smoke(tmp_path: Path) -> None:
    combo_dir = _write_frozen_combo_dir(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "template-discovery-system",
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--mode",
            "frozen-combo-replay",
            "--frozen-combo-dir",
            str(combo_dir),
        ],
    )

    assert result.exit_code == 0, result.output


def test_template_discovery_system_cli_smoke(tmp_path: Path) -> None:
    smid = _write_event_surface(tmp_path, "smid")
    residual = _write_event_surface(tmp_path, "residual", residual=True)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "template-discovery-system",
            "--input-event-rows",
            f"smid24={smid}",
            "--input-event-rows",
            f"residual_ex_smid24={residual}",
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--mode",
            "container-routing",
            "--min-behavior-loop-rows",
            "1",
            "--min-behavior-loop-transition-rate",
            "0",
            "--min-behavior-loop-symbols",
            "1",
            "--min-behavior-loop-months",
            "1",
            "--min-behavior-loop-split-rows",
            "1",
            "--min-loop-regime-rows",
            "1",
            "--min-atom-rows",
            "1",
            "--min-container-rows",
            "1",
            "--min-loop-inside-rows",
            "1",
            "--min-loop-outside-rows",
            "1",
            "--route-lift-bar",
            "0.10",
            "--horizons",
            "6",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "template_discovery_system_v0" in result.output
    run_dirs = sorted((tmp_path / "cli-out").glob("template_discovery_system_v0_*"))
    assert run_dirs
    summary = json.loads((run_dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    assert summary["clean_slate"] is True
    assert summary["reports"]["loop_routing_detail"] == "loop_routing_detail.csv"
    assert summary["reports"]["loop_regime_occupancy"] == "loop_regime_occupancy.csv"
    assert summary["reports"]["c0_parent_readout"] == "c0_parent_readout.csv"
    assert summary["reports"]["loop_context_blockers"] == "loop_context_blockers.csv"
