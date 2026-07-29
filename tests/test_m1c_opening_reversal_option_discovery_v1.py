from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from stocker_prospective.live_subscriptions import QualifiedUnderlying
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    PrimaryOptionPairSelectionV1,
)
from stocker_prospective.option_budget import EpisodeKind
from stocker_prospective.option_discovery import (
    BoundedOptionDiscoveryService,
    _PendingEpisode,
)
from stocker_prospective.option_ledger import OptionContract
from stocker_prospective.options import DteBucket
from stocker_prospective.subscriptions import (
    SubscriptionBudgetManager,
    SubscriptionKind,
)

NOW = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)


class _Adapter:
    def __init__(self) -> None:
        self.metadata_requests = 0
        self.qualification_requests = 0
        self.temporary_quote_requests = 0

    def request_option_chain_metadata(self, **_kwargs: object) -> SimpleNamespace:
        self.metadata_requests += 1
        return SimpleNamespace(
            items=(
                {
                    "underlying_contract_id": 10,
                    "exchange": "SMART",
                    "trading_class": "AAL",
                    "expirations": ["20260731"],
                    "strikes": [99.0, 100.0, 101.0],
                },
            )
        )

    def qualify_exact_contract(self, contract: dict[str, object]) -> SimpleNamespace:
        self.qualification_requests += 1
        right = str(contract["right"])
        return SimpleNamespace(
            items=(
                {
                    "symbol": "AAL",
                    "secType": "OPT",
                    "lastTradeDateOrContractMonth": "20260731",
                    "strike": float(contract["strike"]),
                    "right": right,
                    "conId": 1001 if right == "C" else 1002,
                    "exchange": "SMART",
                    "tradingClass": "AAL",
                },
            )
        )

    def capture_temporary_quote(self, **_kwargs: object) -> None:
        self.temporary_quote_requests += 1
        raise AssertionError("opening reversal must not quote discovery candidates")


class _AuditRepository:
    def __init__(self) -> None:
        self.audits: list[
            tuple[str, PrimaryOptionPairSelectionV1]
        ] = []

    def record_opening_reversal_contract_discovery_v1(
        self,
        _metadata: object,
        *,
        episode_id: str,
        selection: PrimaryOptionPairSelectionV1,
    ) -> None:
        self.audits.append((episode_id, selection))


def _contract_factory(
    symbol: str,
    expiry: date,
    strike: float,
    right: str,
    multiplier: int,
    exchange: str,
    trading_class: str,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "expiry": expiry,
        "strike": strike,
        "right": right,
        "multiplier": multiplier,
        "exchange": exchange,
        "trading_class": trading_class,
    }


def test_opening_reversal_selects_metadata_pair_before_any_quote_request() -> None:
    adapter = _Adapter()
    audit_repository = _AuditRepository()
    budget = SubscriptionBudgetManager(
        limits={SubscriptionKind.OPTION: 8},
        request_rate_limit=100,
        total_line_limit=30,
        future_trading_reserve_lines=12,
    )
    service = BoundedOptionDiscoveryService(
        adapter=adapter,  # type: ignore[arg-type]
        option_recorder=SimpleNamespace(repository=audit_repository),  # type: ignore[arg-type]
        budget=budget,
        underlying_contracts={},
        contract_factory=_contract_factory,  # type: ignore[arg-type]
        metadata_factory=lambda _observed, _sources: object(),  # type: ignore[arg-type]
        reference_quote_provider=lambda _symbol, _timestamp: None,
        maximum_continuous_lines=4,
    )
    episode = _PendingEpisode(
        episode_id="opening-1",
        symbol="AAL",
        session=date(2026, 7, 30),
        entry_timestamp=NOW,
        underlying=QualifiedUnderlying(
            symbol="AAL",
            con_id=10,
            upstream_contract=object(),
            exchange="SMART",
            market_proxy=False,
        ),
        directional_actions={"OPENING_REVERSAL_V1": "PUT"},
        episode_kind=EpisodeKind.OPENING_REVERSAL,
        probability=0.8,
        recording_duration=timedelta(minutes=30),
    )

    plan = service._discover_plan(
        episode,
        100.0,
        discovered_at=NOW,
    )

    assert len(plan.contracts) == 2
    assert {contract.strike for contract in plan.contracts} == {100.0}
    assert {contract.right for contract in plan.contracts} == {"C", "P"}
    assert plan.selection_rule == (
        "opening_reversal_metadata_only_primary_1dte_atm_pair_v1"
    )
    assert adapter.metadata_requests == 1
    assert adapter.qualification_requests == 2
    assert adapter.temporary_quote_requests == 0
    assert len(audit_repository.audits) == 1
    _, audit = audit_repository.audits[0]
    assert audit.candidates_inspected == 2
    assert audit.live_market_data_lines_consumed == 0
    assert audit.planned_live_market_data_lines == 2
    assert not audit.full_chain_live_subscription_created


def test_existing_atm_helper_preserves_independent_call_put_selection() -> None:
    contracts = (
        OptionContract(
            underlying_con_id=10,
            con_id=1,
            expiry=date(2026, 7, 31),
            dte=1,
            dte_bucket=DteBucket.ONE_DTE,
            strike=99.0,
            right="C",
            multiplier=100,
            exchange="SMART",
            trading_class="AAL",
        ),
        OptionContract(
            underlying_con_id=10,
            con_id=2,
            expiry=date(2026, 7, 31),
            dte=1,
            dte_bucket=DteBucket.ONE_DTE,
            strike=101.0,
            right="P",
            multiplier=100,
            exchange="SMART",
            trading_class="AAL",
        ),
    )
    snapshots: dict[str, dict[str, Any]] = {
        contract.con_id_key: {"bid": 1.0, "ask": 1.1}
        for contract in contracts
    }

    pair = BoundedOptionDiscoveryService._atm_pair(
        contracts,
        snapshots=snapshots,
        underlying_reference=100.0,
    )

    assert pair is not None
    assert pair[0].strike == 99.0
    assert pair[1].strike == 101.0
