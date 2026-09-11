from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from stocker_core.candidate_selection import (
    SESSION_HARD_CANDIDATE_RECIPE,
    CandidateEvidenceStatus,
    CandidateIdentity,
    CandidateValue,
    candidate_evidence_status,
    candidate_value,
    opening_prefix,
    rank_candidates,
)
from stocker_core.markets import MARKET_CATALOGUE, MarketId
from stocker_execution.ibkr import HistoricalBar
from stocker_execution.runtime import ExchangeSessionResolver
from test_stage10_extension_sessions import run_for

FIXTURE = Path(__file__).parent / "fixtures" / "session_hard_candidates"
STAGES = SESSION_HARD_CANDIDATE_RECIPE.stages
OPEN = datetime(2025, 7, 17, 13, 30, tzinfo=UTC)


def identity(number: int, market: MarketId = MarketId.US_ALL) -> CandidateIdentity:
    return CandidateIdentity(number, f"S{number}", "NYSE", "NYSE", "USD", market, "STK")


def bars(count: int = 15) -> tuple[HistoricalBar, ...]:
    return tuple(
        HistoricalBar(OPEN + timedelta(minutes=i), 1, 1.2, 0.8, 1.1, 0) for i in range(count)
    )


def calculate(stage, supplied):
    return candidate_value(
        stage,
        supplied,
        expected_prefix=opening_prefix(OPEN, OPEN + timedelta(hours=6), stage.minutes),
        as_of=OPEN + timedelta(minutes=stage.minutes),
    )


def test_frozen_recipe_and_separate_score_names():
    assert [(s.minutes, s.capacity, s.direction, s.enabled) for s in STAGES] == [
        (5, 250, "HIGH", True),
        (10, 50, "HIGH", True),
        (15, 30, "HIGH", True),
    ]
    assert [s.score_name for s in STAGES] == [
        "initial_range_5m_score",
        "preselector_rv_10m_score",
        "watchlist_rv_15m_score",
    ]


@pytest.mark.parametrize("stage", STAGES)
def test_complete_prefix_future_and_extended_bars_do_not_enter_score(stage):
    original = calculate(stage, bars())
    extra = [
        HistoricalBar(OPEN - timedelta(minutes=1), 1000, 2000, 0.001, 1, 1),
        *bars(),
        HistoricalBar(OPEN + timedelta(minutes=15), 1000, 2000, 0.001, 1, 1),
    ]
    assert calculate(stage, extra) == original
    with pytest.raises(ValueError, match="not completed"):
        candidate_value(
            stage,
            bars(),
            expected_prefix=opening_prefix(OPEN, OPEN + timedelta(hours=6), stage.minutes),
            as_of=OPEN + timedelta(minutes=stage.minutes) - timedelta(microseconds=1),
        )


def test_missing_invalid_duplicate_and_nonconsecutive_prefix():
    assert calculate(STAGES[0], bars()[1:]).missing_reason == "MISSING_BAR"
    assert calculate(STAGES[0], (*bars()[:3], *bars()[4:])).missing_reason == "MISSING_BAR"
    assert calculate(STAGES[0], (*bars()[:1], *bars())).missing_reason == "NON_CONSECUTIVE_PREFIX"
    assert calculate(STAGES[0], tuple(reversed(bars()))).missing_reason == "NON_CONSECUTIVE_PREFIX"
    shifted = (replace(bars()[0], timestamp=OPEN + timedelta(seconds=1)), *bars()[1:])
    assert calculate(STAGES[0], shifted).missing_reason == "NON_CONSECUTIVE_PREFIX"
    for value in [0, -1, float("nan"), float("inf")]:
        invalid = (replace(bars()[0], open=value), *bars()[1:])
        assert calculate(STAGES[0], invalid).missing_reason == "INVALID_OPEN"
    assert calculate(STAGES[1], (replace(bars()[0], close=-1), *bars()[1:])).value is None


def test_range_uses_first_open_and_ignores_volume():
    sample = tuple(
        replace(b, open=1 if i == 0 else 1.1, volume=i * 10**9) for i, b in enumerate(bars())
    )
    assert calculate(STAGES[0], sample).value == (1.2 - 0.8) / 1


def test_global_ties_missing_last_and_no_resurrection():
    broad = tuple(identity(i) for i in range(1, 534))
    values = {
        i.con_id: CandidateValue(1.0 if i.con_id <= 240 else None, "MISSING_BAR") for i in broad
    }
    first = rank_candidates(STAGES[0], broad, values, market=MarketId.US_ALL)
    assert len(first) == 533 and sum(r.selected for r in first) == 250
    assert all(r.value is not None for r in first[:240])
    assert all(r.value is None for r in first[240:])
    assert [r.identity.symbol for r in first[:240]] == sorted(
        [i.symbol for i in broad[:240]], key=lambda s: hashlib.sha256(s.encode()).hexdigest()
    )
    assert first == rank_candidates(
        STAGES[0], tuple(reversed(broad)), values, market=MarketId.US_ALL
    )
    active = tuple(r.identity for r in first if r.selected)
    for stage in STAGES[1:]:
        selected = rank_candidates(
            stage,
            active,
            {i.con_id: CandidateValue(float(i.con_id)) for i in broad},
            market=MarketId.US_ALL,
        )
        assert sum(r.selected for r in selected) == stage.capacity
        assert {r.identity.con_id for r in selected} <= {i.con_id for i in active}
        active = tuple(r.identity for r in selected if r.selected)
    assert len(active) == 30
    assert all(i.con_id in {r.identity.con_id for r in first if r.selected} for i in active)


def test_identity_scope_nonfinite_missing_and_under_capacity():
    with pytest.raises(ValueError, match="another market"):
        rank_candidates(STAGES[0], [identity(1, MarketId.UK_LSE)], {}, market=MarketId.US_ALL)
    with pytest.raises(ValueError, match="Duplicate"):
        rank_candidates(STAGES[0], [identity(1), identity(1)], {}, market=MarketId.US_ALL)
    ranked = rank_candidates(
        STAGES[0], [identity(1)], {1: CandidateValue(float("inf"))}, market=MarketId.US_ALL
    )
    assert len(ranked) == 1 and ranked[0].selected
    assert ranked[0].value is None and ranked[0].missing_reason == "NONFINITE_SCORE"


@pytest.mark.parametrize(
    "market,day,utc_open",
    [
        (MarketId.US_ALL, "2026-03-06", "14:30"),
        (MarketId.US_ALL, "2026-03-09", "13:30"),
        (MarketId.UK_LSE, "2026-03-09", "08:00"),
        (MarketId.UK_LSE, "2026-03-30", "07:00"),
        (MarketId.US_ALL, "2026-10-26", "13:30"),
        (MarketId.UK_LSE, "2026-10-26", "08:00"),
        (MarketId.US_ALL, "2026-11-02", "14:30"),
        (MarketId.GERMANY_XETRA, "2026-03-30", "07:00"),
        (MarketId.AUSTRALIA_ASX, "2026-04-02", "23:00"),
        (MarketId.AUSTRALIA_ASX, "2026-04-07", "00:00"),
        (MarketId.US_ALL, "2025-07-03", "13:30"),
        (MarketId.UK_LSE, "2025-12-24", "08:00"),
    ],
)
def test_market_relative_boundaries_dst_and_shortened_days(market, day, utc_open):
    now = datetime.fromisoformat(day + "T12:00:00+00:00")
    session = ExchangeSessionResolver().resolve(run_for(market), now)
    expected_open = datetime.fromisoformat(day + "T" + utc_open + ":00+00:00")
    if market == MarketId.AUSTRALIA_ASX and utc_open == "23:00":
        expected_open -= timedelta(days=1)
    assert session.opens_at == expected_open
    for stage in STAGES:
        prefix = opening_prefix(session.opens_at, session.closes_at, stage.minutes)
        assert prefix[-1] + timedelta(minutes=1) == session.opens_at + timedelta(
            minutes=stage.minutes
        )
    if day == "2025-07-03":
        assert session.closes_at.hour == 17
    if day == "2025-12-24":
        assert session.closes_at.hour == 12


@pytest.mark.parametrize("market", MARKET_CATALOGUE)
def test_existing_market_registry_and_evidence_only(market):
    now = datetime(2026, 9, 8, 12, tzinfo=UTC)
    session = ExchangeSessionResolver().resolve(run_for(market.market_id), now)
    assert session.opens_at is not None
    assert len(opening_prefix(session.opens_at, session.closes_at, 15)) == 15
    expected = (
        CandidateEvidenceStatus.VALIDATED_EXISTING_RESEARCH
        if market.country == "US"
        else CandidateEvidenceStatus.UNVALIDATED_TRANSFER
    )
    assert candidate_evidence_status(market.market_id) == expected


def test_holidays_and_naive_timestamps():
    for market in [MarketId.US_ALL, MarketId.UK_LSE]:
        session = ExchangeSessionResolver().resolve(
            run_for(market), datetime(2025, 12, 25, 12, tzinfo=UTC)
        )
        assert session.opens_at is None
    with pytest.raises(ValueError, match="timezone-aware"):
        opening_prefix(OPEN.replace(tzinfo=None), OPEN + timedelta(hours=6), 5)


@pytest.mark.parametrize("day", ["2025-06-02", "2025-07-03", "2025-07-17"])
def test_saved_research_exact_scores_ordering_and_all_three_watchlists(day):
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    for name, expected_hash in manifest["fixtures_sha256"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == expected_hash
    raw = pd.read_parquet(FIXTURE / "opening_bars.parquet", filters=[("session", "==", day)])
    expected = pd.read_parquet(
        FIXTURE / "expected_scores.parquet", filters=[("session", "==", day)]
    )
    lists = json.loads((FIXTURE / "expected_watchlists.json").read_text())
    rows = {
        symbol: tuple(
            HistoricalBar(r.bar_start_utc.to_pydatetime(), r.open, r.high, r.low, r.close, r.volume)
            for r in frame.sort_values("bar_start_utc").itertuples()
        )
        for symbol, frame in raw.groupby("symbol")
    }
    population = expected[expected.snapshot.eq(5) & expected.candidate_available]
    identities = tuple(
        replace(identity(i), symbol=symbol) for i, symbol in enumerate(sorted(population.symbol), 1)
    )
    opening = datetime.fromisoformat(
        next(s["opening_utc"] for s in manifest["sources"] if s["session"] == day)
    )
    assert population.cap_bucket.nunique() >= 4
    assert raw.open.min() < 5
    active = identities
    for stage in STAGES:
        reference = expected[expected.snapshot.eq(stage.minutes)].set_index("symbol")
        values = {}
        for candidate in identities:
            value = candidate_value(
                stage,
                rows.get(candidate.symbol, ()),
                expected_prefix=opening_prefix(
                    opening, opening + timedelta(hours=6), stage.minutes
                ),
                as_of=opening + timedelta(minutes=stage.minutes),
            )
            expected_score = reference.loc[
                candidate.symbol, "range_pct" if stage.feature == "RANGE" else "rv"
            ]
            if pd.isna(expected_score):
                assert value.value is None, (day, stage.stage_id, candidate.symbol)
            else:
                assert value.value == expected_score, (
                    day,
                    stage.stage_id,
                    candidate.symbol,
                    value.value,
                    expected_score,
                )
            values[candidate.con_id] = value
        assert any(v.value is None for v in values.values())
        ranked = rank_candidates(stage, active, values, market=MarketId.US_ALL)
        feature = "range_pct" if stage.feature == "RANGE" else "rv"
        reference_order = (
            reference.loc[[i.symbol for i in active]]
            .sort_values([feature, "tie"], ascending=[False, True], na_position="last")
            .index.tolist()
        )
        assert [r.identity.symbol for r in ranked] == reference_order
        active = tuple(r.identity for r in ranked if r.selected)
        assert [i.symbol for i in active] == lists[stage.stage_id][day]


@pytest.mark.parametrize("stage", STAGES)
def test_known_formula_flat_and_low_priced_stock(stage):
    from math import log, sqrt

    flat = tuple(replace(b, open=0.01, high=0.01, low=0.01, close=0.01) for b in bars())
    assert calculate(stage, flat).value == 0
    sample = tuple(replace(b, open=0.01, high=0.02, low=0.005, close=0.02) for b in bars())
    expected = (0.02 - 0.005) / 0.01 if stage.feature == "RANGE" else sqrt(log(2) ** 2)
    assert calculate(stage, sample).value == expected
