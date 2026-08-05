from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Event
from time import monotonic

import pandas as pd
import pytest

from stocker_prospective import group_o_recovery as recovery
from stocker_prospective import scientific_inputs
from stocker_prospective.group_o import (
    GROUP_O_FEATURE_MANIFEST_SHA256,
    GROUP_O_REGIME_MAPPING_SHA256,
    FrozenGroupOSessionPackage,
    append_group_o_session_revision,
    build_group_o_context,
    load_group_o_session_package,
)
from stocker_prospective.scientific_inputs import (
    EODHDGroupOPreparationService,
    GroupOAcquisitionPending,
    GroupOAcquisitionResult,
    acquire_eodhd_group_o_session_package,
    build_group_o_session_package,
    build_historical_activity_baseline,
)
from stocker_research.eodhd_options_downloader_v0 import (
    CanonicalizationResult,
    DownloadResult,
    ProviderDTEDiagnostic,
    RequestManifestRow,
    canonicalize_response_records,
)
from stocker_research.m1c_low_movement_v0 import iv_expected_absolute

ROOT = Path(__file__).resolve().parents[1]
FRONT_OPTIONS_ROOT = (
    ROOT
    / "research"
    / "cross-market-context"
    / "20260723-daily-stock-front-options-context-v01"
    / "artifacts"
    / "primary"
)


def _provider_option_record(
    *, dte: object, ask_date: str = "2026-07-31 19:59:59"
) -> dict[str, object]:
    return {
        "id": "AAL260911C00012000-2026-07-31",
        "type": "options-eod",
        "attributes": {
            "contract": "AAL260911C00012000",
            "underlying_symbol": "AAL",
            "type": "call",
            "strike": 12,
            "exp_date": "2026-09-11",
            "bid_date": "2026-08-01T03:59:59.000000Z",
            "ask_date": ask_date,
            "tradetime": "2026-07-31",
            "dte": dte,
            "bid": 1.0,
            "ask": 1.2,
        },
    }


def test_v2_provider_dte_policy_recomputes_from_exact_eod_identity() -> None:
    strict = canonicalize_response_records(
        [_provider_option_record(dte=41)],
        request_id="strict-fixture",
        provider_schema_version="eodhd-options-eod-v1",
    )
    recomputed = canonicalize_response_records(
        [_provider_option_record(dte=41)],
        request_id="v2-fixture",
        provider_schema_version="eodhd-options-eod-v1",
        provider_dte_policy="recompute_from_eod_identity",
    )
    stale_quote = canonicalize_response_records(
        [_provider_option_record(dte=41, ask_date="2026-07-30 19:59:59")],
        request_id="stale-quote-fixture",
        provider_schema_version="eodhd-options-eod-v1",
        provider_dte_policy="recompute_from_eod_identity",
    )

    assert [item.reason_code for item in strict.rejections] == ["contract_date_dte_inconsistency"]
    assert recomputed.rejections == []
    assert recomputed.records[0]["trade_date"] == date(2026, 7, 31)
    assert recomputed.records[0]["dte"] == 42
    assert [item.reason_code for item in stale_quote.rejections] == [
        "eod_observation_date_mismatch"
    ]
    assert stale_quote.provider_dte_diagnostics[0].status == "mismatch"


@pytest.mark.parametrize(
    ("provider_dte", "diagnostic_status"),
    (
        ("malformed", "invalid"),
        (41.5, "fractional"),
        (-1, "negative"),
        (10**400, "mismatch"),
    ),
)
def test_v2_provider_dte_is_diagnostic_only_for_every_provider_value(
    provider_dte: object,
    diagnostic_status: str,
) -> None:
    result = canonicalize_response_records(
        [_provider_option_record(dte=provider_dte)],
        request_id="v2-diagnostic-fixture",
        provider_schema_version="eodhd-options-eod-v1",
        provider_dte_policy="recompute_from_eod_identity",
    )

    assert result.rejections == []
    assert result.records[0]["dte"] == 42
    assert len(result.provider_dte_diagnostics) == 1
    diagnostic = result.provider_dte_diagnostics[0]
    assert diagnostic.status == diagnostic_status
    assert diagnostic.provider_dte_value == provider_dte
    assert diagnostic.calculated_dte == 42
    assert diagnostic.used_for_admission is False


def test_v2_acquisition_persists_provider_dte_as_separate_diagnostic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    _install_fake_group_o_sources(
        monkeypatch,
        observation=observation,
        canonical_records=[],
    )
    diagnostic = ProviderDTEDiagnostic(
        request_id="group-o|AAL|2026-07-31",
        record_index=0,
        provider_record_id="AAL260911C00012000-2026-07-31",
        raw_record_hash="d" * 64,
        provider_dte_value="malformed",
        status="invalid",
        calculated_dte=42,
        used_for_admission=False,
    )
    monkeypatch.setattr(
        scientific_inputs,
        "canonicalize_response_records",
        lambda *_args, **_kwargs: CanonicalizationResult(
            records=[],
            rejections=[],
            provider_dte_diagnostics=[diagnostic],
        ),
    )
    cache_root = tmp_path / "cache"

    with pytest.raises(GroupOAcquisitionPending, match="exact chain is not yet available"):
        acquire_eodhd_group_o_session_package(
            signal_session=signal_session,
            symbols=("AAL",),
            output_path=tmp_path / "context/group-o/2026-08-03.json",
            cache_root=cache_root,
            cache_attempt_id="0001",
            feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
            regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
            provider_dte_policy="recompute_from_eod_identity",
            clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )

    attempt_root = cache_root / observation.isoformat() / "attempts/0001"
    diagnostics_path = attempt_root / "provider_dte_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    receipt = scientific_inputs.load_group_o_attempt_receipt(attempt_root / "attempt_receipt.json")
    assert diagnostics["provider_dte_used_for_admission"] is False
    assert diagnostics["status_counts"] == {"invalid": 1}
    assert diagnostics["rows"] == [diagnostic.to_dict()]
    assert receipt["provider_dte_diagnostics_path"] == str(diagnostics_path)
    assert (
        receipt["provider_dte_diagnostics_file_sha256"]
        == hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()
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


def _daily_bars(*, symbol: str, observation: date) -> pd.DataFrame:
    sessions = pd.bdate_range(end=observation, periods=25)
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "session": item.date(),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 + index / 100.0,
                "activity": 1_000_000.0,
            }
            for index, item in enumerate(sessions)
        ]
    )


def _complete_option_chain(*, observation: date, expiration: date) -> pd.DataFrame:
    return pd.DataFrame(
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


def _install_fake_group_o_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    observation: date,
    canonical_records: list[dict[str, object]],
) -> None:
    class FakeEODHDClient:
        def __init__(self, *, config: object) -> None:
            del config

        def require_token(self) -> str:
            return "fixture-token"

        def fetch_eod(self, **kwargs: object) -> list[object]:
            del kwargs
            return []

    class FakeOptionsDownloader:
        def __init__(self, config: object, *, transport: object) -> None:
            del config, transport

        def download_with_splitting(self, request: object) -> DownloadResult:
            del request
            return DownloadResult(
                records=[{}],
                manifest_rows=[
                    RequestManifestRow(
                        request_id="fixture-chain",
                        underlying_symbol="AAL",
                        trade_date_from=observation.isoformat(),
                        trade_date_to=observation.isoformat(),
                        strike_from=None,
                        strike_to=None,
                        expiration_from="2026-08-07",
                        expiration_to="2026-09-14",
                        offset=0,
                        limit=1000,
                        response_status=200,
                        record_count=len(canonical_records),
                        response_hash="c" * 64,
                        attempts=1,
                        started_at="2026-08-02T12:00:00+00:00",
                        completed_at="2026-08-02T12:00:01+00:00",
                        cache_path="fixture-chain.json",
                    )
                ],
            )

    monkeypatch.setattr(scientific_inputs, "EODHDClient", FakeEODHDClient)
    monkeypatch.setattr(
        scientific_inputs,
        "EODHDOptionsDownloader",
        FakeOptionsDownloader,
    )
    monkeypatch.setattr(
        scientific_inputs,
        "normalize_eod_response",
        lambda *_args, **_kwargs: (
            _daily_bars(symbol="AAL", observation=observation)
            .rename(columns={"session": "timestamp", "activity": "volume"})
            .assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"], utc=True))
        ),
    )
    monkeypatch.setattr(
        scientific_inputs,
        "canonicalize_response_records",
        lambda *_args, **_kwargs: CanonicalizationResult(
            records=canonical_records,
            rejections=[],
        ),
    )


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
    assert context.previous_close_implied_movement_15m == pytest.approx(0.007984323703839152)
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


def test_group_o_late_revision_preserves_failed_package_and_is_selected_before_open(
    tmp_path: Path,
) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    expiration = date(2026, 8, 7)
    context_root = tmp_path / "context"
    base_path = context_root / "group-o" / f"{signal_session}.json"
    base = build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={"AAL": pd.DataFrame()},
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("a" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=base_path,
    )
    assert base.for_symbol("AAL").quality_status == "missing_exact_chain"
    original_bytes = base_path.read_bytes()
    revised = build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={
            "AAL": _complete_option_chain(
                observation=observation,
                expiration=expiration,
            )
        },
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("b" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=tmp_path / "staging" / "revised.json",
    )

    revision = append_group_o_session_revision(
        context_root=context_root,
        revised_package=revised,
        clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )

    assert revision.revision_number == 1
    assert revision.package == revised
    assert base_path.read_bytes() == original_bytes
    assert (
        load_group_o_session_package(
            context_root=context_root,
            signal_session=signal_session,
        )
        == revised
    )
    revision_files = tuple(
        (context_root / "group-o" / "revisions" / signal_session.isoformat()).glob("*.json")
    )
    assert len(revision_files) == 1
    assert revision_files[0].name == "0001.json"
    assert (
        append_group_o_session_revision(
            context_root=context_root,
            revised_package=revised,
            clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )
        == revision
    )


def test_group_o_late_revision_is_rejected_at_or_after_signal_open(tmp_path: Path) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    expiration = date(2026, 8, 7)
    context_root = tmp_path / "context"
    build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={"AAL": pd.DataFrame()},
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("a" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=context_root / "group-o" / f"{signal_session}.json",
    )
    revised = build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={
            "AAL": _complete_option_chain(
                observation=observation,
                expiration=expiration,
            )
        },
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("b" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=tmp_path / "staging" / "revised.json",
    )

    with pytest.raises(ValueError, match="revision must precede the signal session open"):
        append_group_o_session_revision(
            context_root=context_root,
            revised_package=revised,
            clock=lambda: datetime(2026, 8, 3, 13, 30, tzinfo=UTC),
        )

    assert not tuple((context_root / "group-o" / "revisions").glob("**/*.json"))


def test_group_o_revision_rechecks_cutoff_immediately_before_immutable_link(
    tmp_path: Path,
) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    expiration = date(2026, 8, 7)
    context_root = tmp_path / "context"
    build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={"AAL": pd.DataFrame()},
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("a" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=context_root / "group-o" / f"{signal_session}.json",
    )
    revised = build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={
            "AAL": _complete_option_chain(observation=observation, expiration=expiration)
        },
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("b" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=tmp_path / "staging" / "revised.json",
    )
    times = iter(
        (
            datetime(2026, 8, 3, 13, 29, 59, tzinfo=UTC),
            datetime(2026, 8, 3, 13, 30, 0, tzinfo=UTC),
        )
    )

    with pytest.raises(ValueError, match="revision must precede the signal session open"):
        append_group_o_session_revision(
            context_root=context_root,
            revised_package=revised,
            clock=lambda: next(times),
        )

    assert not tuple((context_root / "group-o" / "revisions").glob("**/*.json"))


def test_group_o_loader_rejects_a_tampered_late_revision(tmp_path: Path) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    context_root = tmp_path / "context"
    build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={"AAL": pd.DataFrame()},
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("a" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=context_root / "group-o" / f"{signal_session}.json",
    )
    revised = build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={
            "AAL": _complete_option_chain(
                observation=observation,
                expiration=date(2026, 8, 7),
            )
        },
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("b" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=tmp_path / "staging" / "revised.json",
    )
    append_group_o_session_revision(
        context_root=context_root,
        revised_package=revised,
        clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )
    revision_path = (
        context_root / "group-o" / "revisions" / signal_session.isoformat() / "0001.json"
    )
    assert revision_path.is_file()
    payload = json.loads(revision_path.read_text(encoding="utf-8"))
    payload["supersedes_sha256"] = "f" * 64
    revision_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Group O revision is invalid"):
        load_group_o_session_package(
            context_root=context_root,
            signal_session=signal_session,
        )


def test_group_o_revision_cannot_change_an_already_resolved_context(tmp_path: Path) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    expiration = date(2026, 8, 7)
    context_root = tmp_path / "context"
    base_path = context_root / "group-o" / f"{signal_session}.json"
    build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL", "AAOI"),
        canonical_options_by_symbol={
            "AAL": _complete_option_chain(observation=observation, expiration=expiration),
            "AAOI": pd.DataFrame(),
        },
        daily_bars=pd.concat(
            (
                _daily_bars(symbol="AAL", observation=observation),
                _daily_bars(symbol="AAOI", observation=observation),
            ),
            ignore_index=True,
        ),
        source_receipt_hashes_by_symbol={"AAL": ("a" * 64,), "AAOI": ("b" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=base_path,
    )
    candidate = build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL", "AAOI"),
        canonical_options_by_symbol={
            "AAL": _complete_option_chain(observation=observation, expiration=expiration),
            "AAOI": _complete_option_chain(observation=observation, expiration=expiration),
        },
        daily_bars=pd.concat(
            (
                _daily_bars(symbol="AAL", observation=observation),
                _daily_bars(symbol="AAOI", observation=observation),
            ),
            ignore_index=True,
        ),
        source_receipt_hashes_by_symbol={"AAL": ("c" * 64,), "AAOI": ("d" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=tmp_path / "staging" / "candidate.json",
    )

    with pytest.raises(ValueError, match="already-resolved context differs"):
        append_group_o_session_revision(
            context_root=context_root,
            revised_package=candidate,
            clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )

    assert not (context_root / "group-o" / "revisions").exists()


def test_group_o_revision_can_fill_only_missing_implied_movement(tmp_path: Path) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    context_root = tmp_path / "context"
    base_path = context_root / "group-o" / f"{signal_session}.json"
    base_path.parent.mkdir(parents=True)
    context_fields = {
        "symbol": "AAL",
        "signal_session": signal_session,
        "actual_option_observation_session": observation,
        "front_expiry": date(2026, 8, 14),
        "dte": 14,
        "atm_strike": 12.0,
        "features": {"front_options_implied_tension": 0.0},
        "missing_indicators": {},
        "quality_status": "valid",
        "source_receipt_hashes": ("a" * 64,),
    }
    base = FrozenGroupOSessionPackage(
        contract_version="frozen-m1c-microstructure-recorder-v0/group-o-session-v0",
        signal_session=signal_session,
        generated_from_authorised_cache=True,
        feature_manifest_hash=GROUP_O_FEATURE_MANIFEST_SHA256,
        regime_mapping_hash=GROUP_O_REGIME_MAPPING_SHA256,
        contexts=(build_group_o_context(**context_fields),),
    )
    base_path.write_text(base.model_dump_json(indent=2) + "\n", encoding="utf-8")
    original_bytes = base_path.read_bytes()
    atm_iv = 0.81
    revised = FrozenGroupOSessionPackage(
        **{
            **base.model_dump(mode="python"),
            "contexts": (
                build_group_o_context(
                    **{
                        **context_fields,
                        "previous_close_implied_movement_15m": iv_expected_absolute(atm_iv, 15),
                        "source_receipt_hashes": ("a" * 64,),
                    }
                ),
            ),
        }
    )
    wrong_pair = FrozenGroupOSessionPackage(
        **{
            **revised.model_dump(mode="python"),
            "contexts": (
                build_group_o_context(
                    **{
                        **context_fields,
                        "atm_strike": 13.0,
                        "previous_close_implied_movement_15m": (iv_expected_absolute(atm_iv, 15)),
                        "source_receipt_hashes": ("a" * 64,),
                    }
                ),
            ),
        }
    )

    with pytest.raises(ValueError, match="exact ATM pair identity differs"):
        scientific_inputs._preserve_resolved_group_o_contexts(
            previous=base,
            candidate=wrong_pair,
            include_implied_movement_corrections=True,
        )

    with pytest.raises(ValueError, match="ATM IV source"):
        append_group_o_session_revision(
            context_root=context_root,
            revised_package=revised,
            clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )

    revision = append_group_o_session_revision(
        context_root=context_root,
        revised_package=revised,
        implied_movement_atm_iv_by_symbol={"AAL": atm_iv},
        clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )

    assert revision.reason == "missing_implied_movement_source_correction"
    assert revision.implied_movement_atm_iv_by_symbol == {"AAL": atm_iv}
    assert revision.package == revised
    assert base_path.read_bytes() == original_bytes
    assert (
        load_group_o_session_package(
            context_root=context_root,
            signal_session=signal_session,
        )
        == revised
    )
    revision_path = (
        context_root / "group-o" / "revisions" / signal_session.isoformat() / "0001.json"
    )
    persisted_revision = json.loads(revision_path.read_text(encoding="utf-8"))
    assert persisted_revision["implied_movement_atm_iv_by_symbol"] == {"AAL": atm_iv}
    persisted_revision["implied_movement_atm_iv_by_symbol"]["AAL"] = 0.82
    revision_path.write_text(json.dumps(persisted_revision) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="revision is invalid"):
        load_group_o_session_package(
            context_root=context_root,
            signal_session=signal_session,
        )


def test_group_o_exact_chain_revision_cannot_hide_implied_movement_correction(
    tmp_path: Path,
) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    context_root = tmp_path / "context"
    base_path = context_root / "group-o" / f"{signal_session}.json"
    base_path.parent.mkdir(parents=True)
    valid_fields = {
        "signal_session": signal_session,
        "actual_option_observation_session": observation,
        "front_expiry": date(2026, 8, 14),
        "dte": 14,
        "atm_strike": 12.0,
        "features": {"front_options_implied_tension": 0.0},
        "missing_indicators": {},
        "quality_status": "valid",
    }
    aal_before = build_group_o_context(
        symbol="AAL",
        source_receipt_hashes=("a" * 64,),
        **valid_fields,
    )
    aaoi_before = build_group_o_context(
        symbol="AAOI",
        signal_session=signal_session,
        actual_option_observation_session=None,
        front_expiry=None,
        dte=None,
        atm_strike=None,
        features={},
        missing_indicators={},
        quality_status="missing_exact_chain",
        source_receipt_hashes=("b" * 64,),
    )
    base = FrozenGroupOSessionPackage(
        contract_version="frozen-m1c-microstructure-recorder-v0/group-o-session-v0",
        signal_session=signal_session,
        generated_from_authorised_cache=True,
        feature_manifest_hash=GROUP_O_FEATURE_MANIFEST_SHA256,
        regime_mapping_hash=GROUP_O_REGIME_MAPPING_SHA256,
        contexts=(aal_before, aaoi_before),
    )
    base_path.write_text(base.model_dump_json(indent=2) + "\n", encoding="utf-8")
    aal_atm_iv = 0.81
    mixed_candidate = FrozenGroupOSessionPackage(
        **{
            **base.model_dump(mode="python"),
            "contexts": (
                build_group_o_context(
                    symbol="AAL",
                    source_receipt_hashes=("a" * 64, "c" * 64),
                    previous_close_implied_movement_15m=iv_expected_absolute(aal_atm_iv, 15),
                    **valid_fields,
                ),
                build_group_o_context(
                    symbol="AAOI",
                    source_receipt_hashes=("d" * 64,),
                    previous_close_implied_movement_15m=0.009,
                    **valid_fields,
                ),
            ),
        }
    )

    with pytest.raises(ValueError, match="separate signed revision"):
        append_group_o_session_revision(
            context_root=context_root,
            revised_package=mixed_candidate,
            clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )

    assert not (context_root / "group-o" / "revisions").exists()
    exact_chain_candidate = FrozenGroupOSessionPackage(
        **{
            **mixed_candidate.model_dump(mode="python"),
            "contexts": (aal_before, mixed_candidate.for_symbol("AAOI")),
        }
    )
    exact_revision = append_group_o_session_revision(
        context_root=context_root,
        revised_package=exact_chain_candidate,
        clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )
    movement_revision = append_group_o_session_revision(
        context_root=context_root,
        revised_package=mixed_candidate,
        implied_movement_atm_iv_by_symbol={"AAL": aal_atm_iv},
        clock=lambda: datetime(2026, 8, 2, 12, 1, tzinfo=UTC),
    )

    assert exact_revision.reason == "late_exact_chain_source_correction"
    assert exact_revision.package.for_symbol("AAL") == aal_before
    assert movement_revision.reason == "missing_implied_movement_source_correction"
    assert movement_revision.revision_number == 2
    assert (
        load_group_o_session_package(
            context_root=context_root,
            signal_session=signal_session,
        )
        == mixed_candidate
    )


def test_group_o_recovery_reconciles_crash_after_revision_before_completion_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    expiration = date(2026, 8, 7)
    context_root = tmp_path / "context"
    base_path = context_root / "group-o" / f"{signal_session}.json"
    build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={"AAL": pd.DataFrame()},
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("a" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=base_path,
    )
    attempts_root = (
        context_root / "source-cache" / "eodhd-group-o" / observation.isoformat() / "attempts"
    )
    failed_v1_attempt_path = attempts_root / "0001"
    failed_v1_attempt_path.mkdir(parents=True)
    (failed_v1_attempt_path / "recovery_start_receipt.json").write_text(
        '{"schema_version":"m1c-group-o-recovery-start-v1"}\n',
        encoding="utf-8",
    )
    attempt_path = attempts_root / "0002"
    candidate = build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={
            "AAL": _complete_option_chain(observation=observation, expiration=expiration)
        },
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("b" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=attempt_path / "candidate_package.json",
    )
    diagnostics_path = attempt_path / "provider_dte_diagnostics.json"
    diagnostics_payload = {
        "schema_version": "group-o-provider-dte-diagnostics-v1",
        "attempt_id": "0002",
        "signal_session": signal_session.isoformat(),
        "observation_session": observation.isoformat(),
        "provider_dte_policy": "recompute_from_eod_identity",
        "provider_dte_used_for_admission": False,
        "diagnostic_count": 1,
        "status_counts": {"mismatch": 1},
        "rows": [
            {
                "request_id": "group-o|AAL|2026-07-31",
                "record_index": 0,
                "provider_record_id": "AAL260807P00010000-2026-07-31",
                "raw_record_hash": "d" * 64,
                "provider_dte_value": 6,
                "status": "mismatch",
                "calculated_dte": 7,
                "used_for_admission": False,
            }
        ],
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics_bytes = diagnostics_path.read_bytes()
    freeze = {
        "deployment_receipt_id": "group-o-recovery-deploy-" + "e" * 24,
        "audited_failed_base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "freeze_completed_at_utc": "2026-08-02T11:59:00+00:00",
    }
    release = tmp_path / "release"
    deployment_receipt = (
        release / recovery.RECOVERY_PACKAGE_RELATIVE_V2 / "deployment_freeze_receipt.json"
    )
    deployment_receipt.parent.mkdir(parents=True)
    deployment_receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(recovery, "CANONICAL_COHORT_V0", ("AAL",))
    monkeypatch.setattr(
        recovery,
        "verify_group_o_recovery_freeze_v2",
        lambda _release: freeze,
    )
    monkeypatch.setattr(
        recovery,
        "_require_failed_v1_attempt",
        lambda _root, **_kwargs: failed_v1_attempt_path / "attempt_receipt.json",
    )
    recovery._write_recovery_start_receipt(
        attempt_path=attempt_path,
        attempt_id="0002",
        context_root=context_root,
        release_directory=release,
        freeze_receipt=freeze,
        started_at_utc=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )
    start_receipt = recovery.GroupORecoveryStartReceiptV2.model_validate_json(
        (attempt_path / "recovery_start_receipt.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(recovery, "CANONICAL_COHORT_V0", ("AAL", "MSFT"))
    with pytest.raises(
        recovery.GroupORecoveryIntegrityError,
        match="provider DTE diagnostics cohort coverage",
    ):
        recovery._validate_provider_dte_diagnostics(
            attempt_path=attempt_path,
            start=start_receipt,
        )
    empty_diagnostics = {
        **diagnostics_payload,
        "diagnostic_count": 0,
        "status_counts": {},
        "rows": [],
    }
    diagnostics_path.write_text(
        json.dumps(empty_diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        recovery.GroupORecoveryIntegrityError,
        match="provider DTE diagnostics cohort coverage",
    ):
        recovery._validate_provider_dte_diagnostics(
            attempt_path=attempt_path,
            start=start_receipt,
        )
    diagnostics_path.write_bytes(diagnostics_bytes)
    monkeypatch.setattr(recovery, "CANONICAL_COHORT_V0", ("AAL",))
    append_group_o_session_revision(
        context_root=context_root,
        revised_package=candidate,
        clock=lambda: datetime(2026, 8, 2, 12, 1, tzinfo=UTC),
    )
    completion = attempt_path / "attempt_receipt.json"
    assert not completion.exists()

    missing_identity = json.loads(json.dumps(diagnostics_payload))
    del missing_identity["rows"][0]["request_id"]
    forged_classification = json.loads(json.dumps(diagnostics_payload))
    forged_classification["rows"][0]["status"] = "match"
    forged_classification["status_counts"] = {"match": 1}
    for invalid_diagnostics in (missing_identity, forged_classification):
        diagnostics_path.write_text(
            json.dumps(invalid_diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(
            recovery.GroupORecoveryIntegrityError,
            match="provider DTE diagnostics",
        ):
            recovery.reconcile_group_o_recovery_completion_v2(
                context_root=context_root,
                release_directory=release,
                clock=lambda: datetime(2026, 8, 2, 12, 2, tzinfo=UTC),
            )
        assert not completion.exists()
    diagnostics_path.write_bytes(diagnostics_bytes)

    assert recovery.reconcile_group_o_recovery_completion_v2(
        context_root=context_root,
        release_directory=release,
        clock=lambda: datetime(2026, 8, 2, 12, 2, tzinfo=UTC),
    )

    payload = json.loads(completion.read_text(encoding="utf-8"))
    assert payload["status"] == "published_revision_reconciled_after_restart"
    assert payload["published_revision_id"].startswith("group-o-revision-")
    assert len(payload["attempt_receipt_sha256"]) == 64
    assert payload["provider_dte_diagnostics_path"] == str(diagnostics_path)
    assert (
        payload["provider_dte_diagnostics_file_sha256"]
        == hashlib.sha256(diagnostics_bytes).hexdigest()
    )
    assert payload["provider_dte_diagnostic_counts"] == {"mismatch": 1}
    linked_path = attempt_path / "recovery_completion_receipt.json"
    linked = json.loads(linked_path.read_text(encoding="utf-8"))
    assert linked["status"] == "published_revision_reconciled"
    assert linked["start_receipt_identity_sha256"]
    assert linked["base_package_sha256"] == freeze["audited_failed_base_sha256"]
    assert linked["revision_identity_sha256"]
    assert (
        linked["acquisition_attempt_receipt_identity_sha256"] == payload["attempt_receipt_sha256"]
    )
    assert len(linked["completion_receipt_sha256"]) == 64

    diagnostics_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        recovery.GroupORecoveryIntegrityError,
        match="provider DTE diagnostics",
    ):
        recovery.require_group_o_recovery_ready_before_adapter_v2(
            context_root=context_root,
            release_directory=release,
            now=datetime(2026, 8, 2, 12, 3, tzinfo=UTC),
        )
    diagnostics_path.write_bytes(diagnostics_bytes)

    start_path = attempt_path / "recovery_start_receipt.json"
    tampered = json.loads(start_path.read_text(encoding="utf-8"))
    tampered["monday_market_data_consumed"] = True
    start_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        recovery.GroupORecoveryIntegrityError,
        match="recovery start receipt",
    ):
        recovery.require_group_o_recovery_ready_before_adapter_v2(
            context_root=context_root,
            release_directory=release,
            now=datetime(2026, 8, 2, 12, 3, tzinfo=UTC),
        )


def test_group_o_acquisition_rejects_wrong_provider_underlying_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    _install_fake_group_o_sources(
        monkeypatch,
        observation=observation,
        canonical_records=[
            {
                "provider": "EODHD/UnicornBay",
                "underlying_symbol": "MSFT",
                "contract_id": "MSFT-WRONG",
                "trade_date": observation,
                "raw_record_hash": "e" * 64,
            }
        ],
    )
    output_path = tmp_path / "context" / "group-o" / f"{signal_session}.json"

    with pytest.raises(ValueError, match="underlying identity differs for AAL"):
        acquire_eodhd_group_o_session_package(
            signal_session=signal_session,
            symbols=("AAL",),
            output_path=output_path,
            cache_root=tmp_path / "cache",
            cache_attempt_id="0001",
            feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
            regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
            clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )

    assert not output_path.exists()


def test_group_o_revision_rechecks_cutoff_after_candidate_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    expiration = date(2026, 8, 7)
    context_root = tmp_path / "context"
    base_path = context_root / "group-o" / f"{signal_session}.json"
    build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={"AAL": pd.DataFrame()},
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("a" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=base_path,
    )
    revised = build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={
            "AAL": _complete_option_chain(observation=observation, expiration=expiration)
        },
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("b" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=tmp_path / "prepared" / "revised.json",
    )
    canonical_records = []
    for row in _complete_option_chain(
        observation=observation,
        expiration=expiration,
    ).to_dict(orient="records"):
        canonical_records.append(
            {
                **row,
                "provider": "EODHD/UnicornBay",
                "raw_record_hash": "f" * 64,
            }
        )
    _install_fake_group_o_sources(
        monkeypatch,
        observation=observation,
        canonical_records=canonical_records,
    )
    monkeypatch.setattr(
        scientific_inputs,
        "build_group_o_session_package",
        lambda **_kwargs: revised,
    )
    times = iter(
        (
            datetime(2026, 8, 3, 13, 29, 0, tzinfo=UTC),
            datetime(2026, 8, 3, 13, 29, 30, tzinfo=UTC),
            datetime(2026, 8, 3, 13, 30, 0, tzinfo=UTC),
        )
    )

    with pytest.raises(GroupOAcquisitionPending, match="completed after the signal session open"):
        acquire_eodhd_group_o_session_package(
            signal_session=signal_session,
            symbols=("AAL",),
            output_path=base_path,
            cache_root=tmp_path / "cache",
            cache_attempt_id="0001",
            feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
            regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
            supersedes_path=base_path,
            clock=lambda: next(times),
        )

    assert not (context_root / "group-o" / "revisions").exists()


def test_group_o_acquisition_does_not_publish_when_any_exact_chain_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)

    class FakeEODHDClient:
        def __init__(self, *, config: object) -> None:
            del config

        def require_token(self) -> str:
            return "fixture-token"

        def fetch_eod(self, **kwargs: object) -> list[object]:
            del kwargs
            return []

    class EmptyOptionsDownloader:
        def __init__(self, config: object, *, transport: object) -> None:
            del config, transport

        def download_with_splitting(self, request: object) -> DownloadResult:
            del request
            return DownloadResult(
                records=[],
                manifest_rows=[
                    RequestManifestRow(
                        request_id="empty-chain",
                        underlying_symbol="AAL",
                        trade_date_from=observation.isoformat(),
                        trade_date_to=observation.isoformat(),
                        strike_from=None,
                        strike_to=None,
                        expiration_from="2026-08-07",
                        expiration_to="2026-09-14",
                        offset=0,
                        limit=1000,
                        response_status=200,
                        record_count=0,
                        response_hash="a" * 64,
                        attempts=1,
                        started_at="2026-08-02T12:00:00+00:00",
                        completed_at="2026-08-02T12:00:01+00:00",
                        cache_path=str(tmp_path / "empty.json"),
                    )
                ],
            )

    monkeypatch.setattr(scientific_inputs, "EODHDClient", FakeEODHDClient)
    monkeypatch.setattr(
        scientific_inputs,
        "EODHDOptionsDownloader",
        EmptyOptionsDownloader,
    )
    monkeypatch.setattr(
        scientific_inputs,
        "normalize_eod_response",
        lambda *_args, **_kwargs: (
            _daily_bars(symbol="AAL", observation=observation)
            .rename(columns={"session": "timestamp", "activity": "volume"})
            .assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"], utc=True))
        ),
    )
    output_path = tmp_path / "context" / "group-o" / f"{signal_session}.json"

    with pytest.raises(GroupOAcquisitionPending, match="exact chain is not yet available"):
        acquire_eodhd_group_o_session_package(
            signal_session=signal_session,
            symbols=("AAL",),
            output_path=output_path,
            cache_root=tmp_path / "cache",
            cache_attempt_id="0001",
            feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
            regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
            clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        )

    assert not output_path.exists()
    attempt_receipt = (
        tmp_path / "cache" / observation.isoformat() / "attempts" / "0001" / "attempt_receipt.json"
    )
    assert attempt_receipt.is_file()
    receipt = json.loads(attempt_receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "pending_exact_chain"
    assert receipt["retry_after_utc"] == "2026-08-02T12:15:00+00:00"
    assert len(receipt["attempt_receipt_sha256"]) == 64


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
        assert service.poll(now=datetime(2026, 7, 27, 22, 30, tzinfo=UTC)) is None
        assert monotonic() - began < 0.5
        assert started.wait(timeout=1)
        release.set()
        result = None
        deadline = monotonic() + 2
        while result is None and monotonic() < deadline:
            result = service.poll(now=datetime(2026, 7, 27, 22, 31, tzinfo=UTC))
        assert result is not None
        assert result.signal_session == date(2026, 7, 28)
    finally:
        release.set()
        service.shutdown()


def test_group_o_preparation_retry_interval_is_frozen(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="retry interval is frozen at 15 minutes"):
        EODHDGroupOPreparationService(
            symbols=("AAL",),
            context_root=tmp_path / "context",
            cache_root=tmp_path / "cache",
            feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
            regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
            capture_delay_seconds=7_200,
            retry_delay=timedelta(minutes=5),
        )


def test_group_o_preparation_restart_honours_persisted_retry_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = date(2026, 7, 31)
    attempt_path = (
        tmp_path / "cache" / observation.isoformat() / "attempts" / "0001" / "attempt_receipt.json"
    )
    attempt_path.parent.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema_version": "group-o-acquisition-attempt-v1",
        "attempt_id": "0001",
        "signal_session": "2026-08-03",
        "observation_session": observation.isoformat(),
        "completed_at_utc": "2026-08-02T12:00:00+00:00",
        "status": "pending_exact_chain",
        "retry_after_utc": "2026-08-02T12:15:00+00:00",
    }
    payload["attempt_receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    attempt_path.write_text(json.dumps(payload), encoding="utf-8")
    started = Event()
    monkeypatch.setattr(
        scientific_inputs,
        "acquire_eodhd_group_o_session_package",
        lambda **_kwargs: started.set(),
    )
    service = EODHDGroupOPreparationService(
        symbols=("AAL",),
        context_root=tmp_path / "context",
        cache_root=tmp_path / "cache",
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        capture_delay_seconds=7_200,
    )
    try:
        assert service.poll(now=datetime(2026, 8, 2, 12, 5, tzinfo=UTC)) is None
        assert not started.is_set()
        assert service.last_error == (
            "deferred_group_o_retry_not_before: 2026-08-02T12:15:00+00:00"
        )
    finally:
        service.shutdown()


def test_group_o_persisted_retry_rejects_a_signed_five_minute_interval(
    tmp_path: Path,
) -> None:
    observation = date(2026, 7, 31)
    attempt_path = (
        tmp_path / "cache" / observation.isoformat() / "attempts" / "0001" / "attempt_receipt.json"
    )
    attempt_path.parent.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema_version": "group-o-acquisition-attempt-v1",
        "attempt_id": "0001",
        "signal_session": "2026-08-03",
        "observation_session": observation.isoformat(),
        "completed_at_utc": "2026-08-02T12:00:00+00:00",
        "status": "pending_exact_chain",
        "retry_after_utc": "2026-08-02T12:05:00+00:00",
    }
    payload["attempt_receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    attempt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="retry interval differs from 15 minutes"):
        scientific_inputs.group_o_retry_not_before(
            cache_root=tmp_path / "cache",
            observation_session=observation,
        )


def test_group_o_preparation_retries_an_incomplete_immutable_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    context_root = tmp_path / "context"
    base_path = context_root / "group-o" / f"{signal_session}.json"
    build_group_o_session_package(
        signal_session=signal_session,
        symbols=("AAL",),
        canonical_options_by_symbol={"AAL": pd.DataFrame()},
        daily_bars=_daily_bars(symbol="AAL", observation=observation),
        source_receipt_hashes_by_symbol={"AAL": ("a" * 64,)},
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        output_path=base_path,
    )
    started = Event()
    release = Event()
    observed_kwargs: dict[str, object] = {}
    prior_attempt = tmp_path / "cache" / observation.isoformat() / "attempts" / "0001"
    prior_attempt.mkdir(parents=True)

    def recovery_acquisition(**kwargs: object) -> GroupOAcquisitionResult:
        observed_kwargs.update(kwargs)
        started.set()
        assert release.wait(timeout=5)
        return GroupOAcquisitionResult(
            output_path=Path(str(kwargs["output_path"])),
            signal_session=signal_session,
            observation_session=observation,
            symbol_count=1,
            canonical_option_rows=10,
            rejected_option_rows=0,
        )

    monkeypatch.setattr(
        scientific_inputs,
        "acquire_eodhd_group_o_session_package",
        recovery_acquisition,
    )
    service = EODHDGroupOPreparationService(
        symbols=("AAL",),
        context_root=context_root,
        cache_root=tmp_path / "cache",
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        capture_delay_seconds=7_200,
    )
    try:
        assert service.poll(now=datetime(2026, 8, 2, 12, 0, tzinfo=UTC)) is None
        assert started.wait(timeout=1)
        assert observed_kwargs["supersedes_path"] == base_path
        assert observed_kwargs["cache_attempt_id"] == "0002"
    finally:
        release.set()
        service.shutdown()


def test_group_o_preparation_revises_valid_package_missing_implied_movement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_session = date(2026, 8, 3)
    observation = date(2026, 7, 31)
    context_root = tmp_path / "context"
    base_path = context_root / "group-o" / f"{signal_session}.json"
    base_path.parent.mkdir(parents=True)
    package = FrozenGroupOSessionPackage(
        contract_version="frozen-m1c-microstructure-recorder-v0/group-o-session-v0",
        signal_session=signal_session,
        generated_from_authorised_cache=True,
        feature_manifest_hash=GROUP_O_FEATURE_MANIFEST_SHA256,
        regime_mapping_hash=GROUP_O_REGIME_MAPPING_SHA256,
        contexts=(
            build_group_o_context(
                symbol="AAL",
                signal_session=signal_session,
                actual_option_observation_session=observation,
                front_expiry=date(2026, 8, 14),
                dte=14,
                atm_strike=12.0,
                features={"front_options_implied_tension": 0.0},
                missing_indicators={},
                quality_status="valid",
                source_receipt_hashes=("a" * 64,),
            ),
        ),
    )
    base_path.write_text(package.model_dump_json(indent=2), encoding="utf-8")
    started = Event()
    release = Event()
    observed_kwargs: dict[str, object] = {}

    def recovery_acquisition(**kwargs: object) -> GroupOAcquisitionResult:
        observed_kwargs.update(kwargs)
        started.set()
        assert release.wait(timeout=5)
        return GroupOAcquisitionResult(
            output_path=Path(str(kwargs["output_path"])),
            signal_session=signal_session,
            observation_session=observation,
            symbol_count=1,
            canonical_option_rows=2,
            rejected_option_rows=0,
        )

    monkeypatch.setattr(
        scientific_inputs,
        "acquire_eodhd_group_o_session_package",
        recovery_acquisition,
    )
    service = EODHDGroupOPreparationService(
        symbols=("AAL",),
        context_root=context_root,
        cache_root=tmp_path / "cache",
        feature_manifest_path=FRONT_OPTIONS_ROOT / "front_options_feature_manifest.json",
        regime_mapping_path=FRONT_OPTIONS_ROOT / "front_options_regime_mapping.json",
        capture_delay_seconds=7_200,
    )
    try:
        assert service.poll(now=datetime(2026, 8, 2, 12, 0, tzinfo=UTC)) is None
        assert started.wait(timeout=1)
        assert observed_kwargs["supersedes_path"] == base_path
    finally:
        release.set()
        service.shutdown()
