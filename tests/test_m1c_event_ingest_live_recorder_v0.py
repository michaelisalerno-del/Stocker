from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.event_ingest import (
    IBKRCallbackNormalizer,
    StreamKind,
    StreamOwner,
)
from stocker_prospective.events import (
    OptionQuoteEvent,
    UnderlyingDepthEvent,
    UnderlyingLevel1QuoteEvent,
)
from stocker_prospective.frozen_m1c import FrozenM1CScore
from stocker_prospective.ibkr import IBKRConnectionConfig, IBKRMarketDataAdapter
from stocker_prospective.live_bars import AuditedLiveBar
from stocker_prospective.live_recorder import (
    FrozenM1CLiveRecorder,
    ScientificReadiness,
)
from stocker_prospective.m1c_features import HistoricalActivityBaseline
from stocker_prospective.market_data import MarketDataBudget, MarketDataType
from stocker_prospective.option_ledger import OptionContract
from stocker_prospective.options import DteBucket
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import FrozenM1CRecorderEngine

START = datetime(2026, 7, 24, 13, 30, tzinfo=UTC)
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


def callback(
    sequence: int,
    *,
    field: str,
    value: object,
    market_data_type: str | None = "live",
) -> dict[str, Any]:
    observed = START + timedelta(seconds=sequence)
    return {
        "kind": "level1_quote_update",
        "request_id": 7,
        "field": field,
        "value": value,
        "market_data_type": market_data_type,
        "received_timestamp_utc": observed.isoformat(),
        "received_monotonic_ns": 1000 + sequence,
        "source_sequence": sequence,
    }


def test_callback_normalizer_preserves_every_level1_state_change() -> None:
    normalizer = IBKRCallbackNormalizer(prospective_collection_start=START)
    normalizer.register(
        StreamOwner(
            request_id=7,
            kind=StreamKind.UNDERLYING_LEVEL1,
            symbol="AAL",
            con_id=123,
            exchange="SMART",
        )
    )

    bid = normalizer.normalize(callback(1, field="bid", value=10.0))
    ask = normalizer.normalize(callback(2, field="ask", value=10.02))
    volume = normalizer.normalize(callback(3, field="volume", value=12_345.0))
    assert bid is not None and ask is not None and volume is not None
    assert isinstance(bid.raw_event, UnderlyingLevel1QuoteEvent)
    assert isinstance(ask.raw_event, UnderlyingLevel1QuoteEvent)
    assert isinstance(volume.raw_event, UnderlyingLevel1QuoteEvent)
    assert bid.raw_event.quote_valid is False
    assert ask.raw_event.quote_valid is True
    assert ask.raw_event.bid == 10.0
    assert ask.raw_event.ask == 10.02
    assert volume.raw_event.volume == 12_345.0
    assert bid.raw_event.event_id != ask.raw_event.event_id


def test_callback_normalizer_marks_unconfirmed_market_data_unknown() -> None:
    normalizer = IBKRCallbackNormalizer(prospective_collection_start=START)
    normalizer.register(
        StreamOwner(
            request_id=7,
            kind=StreamKind.UNDERLYING_LEVEL1,
            symbol="AAL",
            con_id=123,
        )
    )
    result = normalizer.normalize(callback(1, field="bid", value=10.0, market_data_type=None))
    assert result is not None
    assert isinstance(result.raw_event, UnderlyingLevel1QuoteEvent)
    assert result.raw_event.market_data_type is MarketDataType.UNKNOWN
    assert result.raw_event.market_data_type.primary_eligible is False


def test_option_computations_preserve_source_and_only_model_populates_model_fields() -> None:
    normalizer = IBKRCallbackNormalizer(prospective_collection_start=START)
    normalizer.register(
        StreamOwner(
            request_id=9,
            kind=StreamKind.OPTION_LEVEL1,
            symbol="AAL",
            con_id=999,
            episode_id="episode-option-source",
            option_contract=OptionContract(
                underlying_con_id=123,
                con_id=999,
                expiry=START.date(),
                dte=0,
                dte_bucket=DteBucket.ZERO_DTE,
                strike=100.0,
                right="C",
                multiplier=100,
                exchange="SMART",
                trading_class="AAL",
            ),
        )
    )
    base = {
        "kind": "level1_quote_update",
        "request_id": 9,
        "field": "option_computation",
        "market_data_type": "live",
        "received_timestamp_utc": START.isoformat(),
        "provider_timestamp_utc": START.isoformat(),
        "received_monotonic_ns": 1,
        "source_sequence": 1,
        "option_price": 1.25,
        "implied_volatility": 0.5,
        "delta": 0.4,
    }
    bid = normalizer.normalize({**base, "computation_source": "bid"})
    model = normalizer.normalize(
        {
            **base,
            "computation_source": "model",
            "received_monotonic_ns": 2,
            "source_sequence": 2,
            "option_price": 1.3,
        }
    )

    assert bid is not None and isinstance(bid.raw_event, OptionQuoteEvent)
    assert model is not None and isinstance(model.raw_event, OptionQuoteEvent)
    assert bid.raw_event.option_model_price is None
    assert bid.raw_event.option_computation_by_source["bid"]["option_price"] == 1.25
    assert model.raw_event.option_model_price == 1.3
    assert set(model.raw_event.option_computation_by_source) == {"bid", "model"}


def test_depth_reset_is_preserved_as_a_raw_invalidating_event() -> None:
    normalizer = IBKRCallbackNormalizer(prospective_collection_start=START)
    normalizer.register(
        StreamOwner(
            request_id=8,
            kind=StreamKind.UNDERLYING_DEPTH,
            symbol="AAL",
            con_id=123,
        )
    )
    result = normalizer.normalize(
        {
            "kind": "depth_reset",
            "request_id": 8,
            "received_timestamp_utc": START.isoformat(),
            "received_monotonic_ns": 100,
            "source_sequence": 1,
            "smart_depth": True,
        }
    )

    assert result is not None
    assert isinstance(result.raw_event, UnderlyingDepthEvent)
    assert result.raw_event.reset is True
    assert result.control_kind == "depth_reset"


def adapter() -> IBKRMarketDataAdapter:
    return IBKRMarketDataAdapter(
        config=IBKRConnectionConfig(
            host="127.0.0.1",
            port=4002,
            client_id=91,
            expected_environment="read_only",
            connect_timeout_seconds=1,
            request_timeout_seconds=1,
            quote_capture_timeout_seconds=1,
            allowed_market_data_types=(MarketDataType.LIVE,),
        ),
        budget=MarketDataBudget(
            line_limit=50,
            reserved_headroom=1,
            request_rate_limit=100,
        ),
    )


def metadata_factory(
    observed_at: datetime,
    source_timestamps: tuple[datetime, ...],
) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id="live-v0",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[item.isoformat() for item in source_timestamps],
        recorded_at_utc=max(observed_at, START),
    )


def test_live_recorder_persists_raw_stream_and_only_bounded_projection(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    database.migrate()
    database.create_run(metadata_factory(START, (START,)))
    repository = FrozenRecorderRepository(database)
    market_adapter = adapter()
    normalizer = IBKRCallbackNormalizer(prospective_collection_start=START)
    recorder = FrozenM1CLiveRecorder(
        adapter=market_adapter,
        normalizer=normalizer,
        raw_store=PartitionedEventStore(
            root=tmp_path / "raw",
            prospective_collection_start=START,
            recorder_version="test",
            contract_version="frozen-m1c-microstructure-recorder-v0",
        ),
        repository=repository,
        engine=cast(FrozenM1CRecorderEngine, object()),
        activity_baseline=HistoricalActivityBaseline(minimum_sessions=1),
        group_o_provider=lambda _symbol, _session: (_ for _ in ()).throw(
            AssertionError("Group O must not be requested without a completed checkpoint")
        ),
        metadata_factory=metadata_factory,
        run_id="live-v0",
        universe_symbols=COHORT,
        market_proxy_symbol="VTI",
        readiness=ScientificReadiness(
            m1c_parity_passed=True,
            direction_parity_passed=True,
            bar_compatibility_passed=True,
            clock_drift_within_tolerance=True,
        ),
        maximum_quote_age=timedelta(seconds=2),
    )
    recorder.register_stream(
        StreamOwner(
            request_id=7,
            kind=StreamKind.UNDERLYING_LEVEL1,
            symbol="AAL",
            con_id=123,
            exchange="SMART",
        )
    )
    for event in (
        callback(1, field="bid", value=10.0),
        callback(2, field="ask", value=10.02),
        callback(3, field="bid_size", value=200.0),
        callback(4, field="ask_size", value=100.0),
    ):
        market_adapter._append_stream_event(
            "level1_quote_update",
            7,
            {
                key: value
                for key, value in event.items()
                if key
                not in {
                    "kind",
                    "request_id",
                    "received_timestamp_utc",
                    "received_monotonic_ns",
                    "source_sequence",
                }
            },
        )
    result = recorder.poll(now=START + timedelta(seconds=5))

    assert result.raw_event_count == 4
    assert len(result.partition_hashes) == 1
    assert list((tmp_path / "raw").rglob("*.parquet"))
    with database._connect() as connection:
        projection = connection.execute(
            "SELECT * FROM underlying_live_state_v0 WHERE symbol = 'AAL'"
        ).fetchone()
        assert projection is not None
        assert projection["bid"] == 10.0
        assert projection["ask"] == 10.02
        assert projection["quote_size_imbalance"] > 0.0
        assert (
            connection.execute("SELECT COUNT(*) AS n FROM underlying_live_state_v0").fetchone()["n"]
            == 1
        )

    gap_start = START + timedelta(seconds=10)
    recorder.mark_gap("AAL", started_at=gap_start)
    recorder.clear_gap_after_complete_bar(
        "AAL",
        completed_at=gap_start + timedelta(minutes=5),
    )
    assert recorder.gap_overlaps(
        "AAL",
        window_start=gap_start - timedelta(seconds=1),
        window_end=gap_start + timedelta(minutes=1),
    )
    assert not recorder.gap_overlaps(
        "AAL",
        window_start=gap_start + timedelta(minutes=6),
        window_end=gap_start + timedelta(minutes=7),
    )


def test_live_recorder_hydrates_processed_checkpoints_before_restart_replay(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    database.migrate()
    metadata = metadata_factory(START, (START,))
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    checkpoint_id = repository.record_checkpoint(
        metadata,
        symbol="AAL",
        session=date(2026, 7, 24),
        checkpoint=6,
        bar_start_utc=START,
        bar_end_utc=START + timedelta(minutes=30),
        score=FrozenM1CScore(
            model_hash="b" * 64,
            probability=0.25,
            threshold=0.488333710794033,
            threshold_passed=False,
            feature_order=("feature",),
            feature_values=(0.0,),
            transformed_values=(0.0,),
            feature_hash="c" * 64,
            missing_feature_count=0,
        ),
        session_context_hash="d" * 64,
        feature_values={"feature": 0.0},
        eligible=True,
        feature_freshness="exact_previous_session",
        rejection_reasons=(),
    )
    assert repository.recorded_checkpoint_identities(run_id="live-v0") == set()
    repository.mark_checkpoint_complete(
        metadata,
        checkpoint_id=checkpoint_id,
        symbol="AAL",
        session=date(2026, 7, 24),
        checkpoint=6,
    )

    recorder = FrozenM1CLiveRecorder(
        adapter=adapter(),
        normalizer=IBKRCallbackNormalizer(prospective_collection_start=START),
        raw_store=PartitionedEventStore(
            root=tmp_path / "raw",
            prospective_collection_start=START,
            recorder_version="test",
            contract_version="frozen-m1c-microstructure-recorder-v0",
        ),
        repository=repository,
        engine=cast(FrozenM1CRecorderEngine, object()),
        activity_baseline=HistoricalActivityBaseline(minimum_sessions=1),
        group_o_provider=lambda _symbol, _session: (_ for _ in ()).throw(
            AssertionError("persisted checkpoint must not be rescored")
        ),
        metadata_factory=metadata_factory,
        run_id="live-v0",
        universe_symbols=COHORT,
        market_proxy_symbol="VTI",
        readiness=ScientificReadiness(
            m1c_parity_passed=True,
            direction_parity_passed=True,
            bar_compatibility_passed=True,
            clock_drift_within_tolerance=True,
        ),
        maximum_quote_age=timedelta(seconds=2),
    )

    session = date(2026, 7, 24)
    for symbol in ("AAL", "VTI"):
        recorder._bars[(symbol, session)] = {
            checkpoint: AuditedLiveBar(
                symbol=symbol,
                session=session,
                bar_start_utc=START + timedelta(minutes=5 * (checkpoint - 1)),
                bar_end_utc=START + timedelta(minutes=5 * checkpoint),
                checkpoint=checkpoint,
                open=10.0,
                high=10.1,
                low=9.9,
                close=10.0,
                volume_or_activity_field=1_000.0,
                wap_where_available=10.0,
                trade_count_where_available=10,
                source="ibkr_historical_data_update",
                source_completeness="completed",
                finalised=True,
                provider_timestamp_utc=START + timedelta(minutes=5 * checkpoint),
                received_timestamp_utc=START + timedelta(minutes=5 * checkpoint),
            )
            for checkpoint in range(1, 7)
        }

    assert recorder._score_ready() == ()
