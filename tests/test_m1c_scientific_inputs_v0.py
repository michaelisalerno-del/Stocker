from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Event
from time import monotonic

import pandas as pd
import pytest

from stocker_prospective import scientific_inputs
from stocker_prospective.group_o import load_group_o_session_package
from stocker_prospective.scientific_inputs import (
    EODHDGroupOPreparationService,
    GroupOAcquisitionResult,
    build_group_o_session_package,
    build_historical_activity_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
FRONT_OPTIONS_ROOT = (
    ROOT
    / "research"
    / "cross-market-context"
    / "20260723-daily-stock-front-options-context-v01"
    / "artifacts"
    / "primary"
)


def test_historical_activity_builder_keeps_one_regular_session_stream_per_symbol(
    tmp_path: Path,
) -> None:
    session = date(2026, 7, 24)
    opened = datetime(2026, 7, 24, 13, 30, tzinfo=UTC)
    source = pd.DataFrame(
        [
            {
                "timestamp": opened + timedelta(minutes=5 * ordinal),
                "volume": 1_000.0 + ordinal,
            }
            for ordinal in range(78)
        ]
        + [
            {
                "timestamp": opened + timedelta(hours=6, minutes=30),
                "volume": float("nan"),
            }
        ]
    )
    output_path = tmp_path / "prior-session-five-minute-bars.parquet"

    result = build_historical_activity_baseline(
        source_frames={"AAL": source},
        latest_authorised_session=session,
        output_path=output_path,
        minimum_sessions=1,
    )

    persisted = pd.read_parquet(output_path)
    assert result.row_count == 78
    assert result.session_count == 1
    assert persisted["bar_ordinal"].tolist() == list(range(78))
    assert persisted["session"].unique().tolist() == [session.isoformat()]
    assert (
        build_historical_activity_baseline(
            source_frames={"AAL": source},
            latest_authorised_session=session,
            output_path=output_path,
            minimum_sessions=1,
        ).row_count
        == 78
    )
    changed = source.copy()
    changed.loc[0, "volume"] = 2_000.0
    with pytest.raises(ValueError, match="immutable historical activity baseline differs"):
        build_historical_activity_baseline(
            source_frames={"AAL": changed},
            latest_authorised_session=session,
            output_path=output_path,
            minimum_sessions=1,
        )


def _option_row(
    *,
    option_type: str,
    strike: float,
    delta: float,
    implied_volatility: float,
    observation: date,
    expiration: date,
) -> dict[str, object]:
    return {
        "request_id": "fixture-request",
        "underlying_symbol": "AAL",
        "contract_id": f"AAL-{expiration}-{option_type}-{strike}",
        "option_type": option_type,
        "expiration_date": expiration,
        "strike": strike,
        "trade_date": observation,
        "bid": 4.8 if strike == 100.0 else 1.0,
        "ask": 5.2 if strike == 100.0 else 1.2,
        "midpoint": 5.0 if strike == 100.0 else 1.1,
        "open_interest": 100,
        "implied_volatility": implied_volatility,
        "delta": delta,
    }


def test_group_o_builder_applies_frozen_dimensions_and_regime_without_outcomes(
    tmp_path: Path,
) -> None:
    signal_session = date(2026, 7, 27)
    observation = date(2026, 7, 24)
    expiration = date(2026, 7, 31)
    daily_sessions = pd.bdate_range(end=observation, periods=25)
    daily_bars = pd.DataFrame(
        [
            {
                "symbol": "AAL",
                "session": item.date(),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 + index / 100.0,
                "activity": 1_000_000.0,
            }
            for index, item in enumerate(daily_sessions)
        ]
    )
    chain = pd.DataFrame(
        [
            _option_row(
                option_type="call",
                strike=100.0,
                delta=0.51,
                implied_volatility=0.80,
                observation=observation,
                expiration=expiration,
            ),
            _option_row(
                option_type="put",
                strike=100.0,
                delta=-0.49,
                implied_volatility=0.82,
                observation=observation,
                expiration=expiration,
            ),
            _option_row(
                option_type="call",
                strike=110.0,
                delta=0.25,
                implied_volatility=0.77,
                observation=observation,
                expiration=expiration,
            ),
            _option_row(
                option_type="put",
                strike=90.0,
                delta=-0.25,
                implied_volatility=0.86,
                observation=observation,
                expiration=expiration,
            ),
        ]
    )
    output_path = tmp_path / "context" / "group-o" / f"{signal_session}.json"

    package = build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={"AAL": chain},
        daily_bars=daily_bars,
        source_receipt_hashes_by_symbol={"AAL": ("a" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=output_path,
    )

    context = package.for_symbol("AAL")
    assert context.eligible is True
    assert context.actual_option_observation_session == observation
    assert context.front_expiry == expiration
    assert context.dte == 7
    assert context.atm_strike == 100.0
    assert set(context.features) == {
        "front_options_implied_tension",
        "front_options_premium_richness",
        "front_options_downside_asymmetry",
        "front_options_liquidity_stress",
        "front_options_positioning_concentration",
        "front_options_directional_positioning",
        "front_options_surface_disagreement",
        "front_options_regime_p_0",
        "front_options_regime_p_1",
        "front_options_regime_p_2",
        "front_options_regime_p_3",
        "front_options_regime_entropy",
        "front_options_regime_margin",
        "skew_25d_missing",
        "near_spot_oi_concentration_missing",
        "call_put_oi_imbalance_missing",
    }
    assert (
        load_group_o_session_package(
            context_root=output_path.parents[1],
            signal_session=signal_session,
        )
        == package
    )

    assert (
        build_group_o_session_package(
            signal_session=signal_session,
            symbols=("AAL",),
            canonical_options_by_symbol={"AAL": chain},
            daily_bars=daily_bars,
            source_receipt_hashes_by_symbol={"AAL": ("a" * 64,)},
            feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
            regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
            output_path=output_path,
        )
        == package
    )
    with pytest.raises(ValueError, match="immutable Group O package differs"):
        build_group_o_session_package(
            signal_session=signal_session,
            symbols=("AAL",),
            canonical_options_by_symbol={"AAL": chain},
            daily_bars=daily_bars,
            source_receipt_hashes_by_symbol={"AAL": ("b" * 64,)},
            feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
            regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
            output_path=output_path,
        )


def test_group_o_builder_rejects_unregistered_frozen_transform(
    tmp_path: Path,
) -> None:
    changed_manifest = tmp_path / "front_options_feature_manifest.json"
    changed_manifest.write_text(
        (FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json").read_text(encoding="utf-8")
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen Group O feature manifest hash differs"):
        build_group_o_session_package(
            signal_session=date(2026, 7, 27),
            symbols=("AAL",),
            canonical_options_by_symbol={"AAL": pd.DataFrame()},
            daily_bars=pd.DataFrame(
                columns=["symbol", "session", "open", "high", "low", "close", "activity"]
            ),
            source_receipt_hashes_by_symbol={"AAL": ("a" * 64,)},
            feature_manifest_path=changed_manifest,
            regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
            output_path=tmp_path / "group-o.json",
        )


def test_group_o_preparation_never_blocks_critical_recorder_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()

    def delayed_acquisition(**kwargs: object) -> GroupOAcquisitionResult:
        started.set()
        assert release.wait(timeout=5)
        return GroupOAcquisitionResult(
            output_path=Path(str(kwargs["output_path"])),
            signal_session=date(2026, 7, 28),
            observation_session=date(2026, 7, 27),
            symbol_count=1,
            canonical_option_rows=10,
            rejected_option_rows=0,
        )

    monkeypatch.setattr(
        scientific_inputs,
        "acquire_eodhd_group_o_session_package",
        delayed_acquisition,
    )
    service = EODHDGroupOPreparationService(
        symbols=("AAL",),
        context_root=tmp_path / "context",
        cache_root=tmp_path / "cache",
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        capture_delay_seconds=7_200,
    )
    began = monotonic()
    try:
        assert service.poll(now=datetime(2026, 7, 28, 15, 0, tzinfo=UTC)) is None
        assert monotonic() - began < 0.5
        assert started.wait(timeout=1)
        release.set()
        result = None
        deadline = monotonic() + 2
        while result is None and monotonic() < deadline:
            result = service.poll(now=datetime(2026, 7, 28, 15, 1, tzinfo=UTC))
        assert result is not None
        assert result.signal_session == date(2026, 7, 28)
    finally:
        release.set()
        service.shutdown()
