from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from stocker_research.eodhd_options_downloader_v0 import sha256_bytes, stable_request_id

PROBE_DIR = Path(__file__).resolve().parents[1]


def provider_record(
    *,
    contract: str,
    option_type: str,
    observation_date: str,
    tradetime: str,
    expiration_date: str = "2025-01-17",
    strike: float = 13.0,
    open_interest: int = 100,
) -> dict[str, Any]:
    return {
        "id": f"{contract}-{observation_date}",
        "type": "options-eod",
        "attributes": {
            "contract": contract,
            "underlying_symbol": "AAL",
            "type": option_type,
            "exp_date": expiration_date,
            "strike": strike,
            "tradetime": tradetime,
            "bid_date": f"{observation_date} 15:59:58-05:00",
            "ask_date": f"{observation_date} 15:59:59-05:00",
            "bid": 0.9,
            "ask": 1.1,
            "midpoint": 1.0,
            "open_interest": open_interest,
            "volatility": 0.4,
            "dte": (
                date.fromisoformat(expiration_date) - date.fromisoformat(observation_date)
            ).days,
        },
    }


def test_exact_contract_observations_use_resource_date_not_tradetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(PROBE_DIR))
    from contract_history_probe import exact_contract_observations

    exact = provider_record(
        contract="AAL250117C00013000",
        option_type="call",
        observation_date="2025-01-02",
        tradetime="2024-12-20",
    )
    future_with_matching_activity = provider_record(
        contract="AAL250117C00013000",
        option_type="call",
        observation_date="2025-01-03",
        tradetime="2025-01-02",
    )

    selected = exact_contract_observations(
        [exact, future_with_matching_activity],
        required_date=date(2025, 1, 2),
        request_id="history-request",
    )

    assert [row["trade_date"] for row in selected.records] == [date(2025, 1, 2)]
    assert selected.available_observation_dates == (date(2025, 1, 2), date(2025, 1, 3))
    assert selected.rejections == ()


def test_probe_targets_are_three_frozen_symbols_at_three_frozen_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(PROBE_DIR))
    from contract_history_probe import build_probe_targets

    audit = (
        PROBE_DIR.parent
        / "20260722-broad-conflict-prior-close-iv-v0"
        / "artifacts"
        / "primary"
        / "option_underlying_price_audit.csv"
    )
    targets = build_probe_targets(audit)

    assert len(targets) == 9
    assert {target.symbol for target in targets} == {"AAL", "MSTR", "WULF"}
    assert {target.required_options_date for target in targets} == {
        date(2024, 1, 16),
        date(2024, 10, 31),
        date(2025, 8, 21),
    }
    assert all(target.required_options_date < target.signal_date for target in targets)
    assert all(target.previous_close > 0.0 for target in targets)


def test_contract_discovery_is_bounded_and_completely_paginated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(PROBE_DIR))
    from contract_history_probe import ProbeTarget, discover_contracts

    class FakeRequester:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def get_json(self, endpoint: str, *, params: dict[str, object], timeout: float) -> object:
            self.calls.append({"endpoint": endpoint, "params": dict(params), "timeout": timeout})
            offset = int(params["page[offset]"])
            contracts = [
                ("AAL250110C00013000", "call", 13.0),
                ("AAL250110P00013000", "put", 13.0),
                ("AAL250117C00014000", "call", 14.0),
            ]
            selected = contracts[offset : offset + 2]
            next_offset = offset + len(selected)
            return {
                "meta": {"offset": offset, "limit": 2},
                "data": [
                    {
                        "id": contract,
                        "type": "options-contracts",
                        "attributes": {
                            "contract": contract,
                            "underlying_symbol": "AAL",
                            "exp_date": "2025-01-10" if strike == 13.0 else "2025-01-17",
                            "type": option_type,
                            "strike": strike,
                        },
                    }
                    for contract, option_type, strike in selected
                ],
                "links": {
                    "next": (
                        None
                        if offset == 2
                        else f"https://eodhd.com/api/next?page%5Boffset%5D={next_offset}"
                    )
                },
            }

    requester = FakeRequester()
    target = ProbeTarget(
        symbol="AAL",
        signal_date=date(2025, 1, 3),
        required_options_date=date(2025, 1, 2),
        previous_close=13.0,
    )

    result = discover_contracts(requester, target=target, page_limit=2, maximum_records=10)

    assert len(result.contracts) == 3
    assert result.requests_completed == 2
    assert [call["params"]["page[offset]"] for call in requester.calls] == [0, 2]  # type: ignore[index]
    first = requester.calls[0]["params"]
    assert first["filter[underlying_symbol]"] == "AAL"  # type: ignore[index]
    assert first["filter[exp_date_from]"] == "2025-01-09"  # type: ignore[index]
    assert first["filter[exp_date_to]"] == "2025-02-16"  # type: ignore[index]
    assert first["filter[strike_from]"] == pytest.approx(9.1)  # type: ignore[index]
    assert first["filter[strike_to]"] == pytest.approx(16.9)  # type: ignore[index]
    assert not any("tradetime" in str(key) for key in first)  # type: ignore[union-attr]


def test_contract_discovery_request_is_resumed_from_verified_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(PROBE_DIR))
    from contract_history_probe import ResumableDiscoveryRequester

    endpoint = "/mp/unicornbay/options/contracts"
    params: dict[str, object] = {
        "filter[underlying_symbol]": "AAL",
        "page[offset]": 0,
        "page[limit]": 1,
    }
    payload = {
        "meta": {"offset": 0, "limit": 1},
        "data": [],
        "links": {"next": None},
    }

    class FakeLiveRequester:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.http_requests_attempted = 0
            self.cached_bytes_accounted = 0
            self.manifest_rows: list[dict[str, object]] = []

        def account_cached_bytes(self, response_bytes: int) -> None:
            self.cached_bytes_accounted += response_bytes

        def get_json(
            self, requested_endpoint: str, *, params: dict[str, object], timeout: float
        ) -> object:
            if self.fail:
                raise AssertionError("resumed discovery must not call the provider")
            self.http_requests_attempted += 1
            content = json.dumps(payload, sort_keys=True).encode()
            cache_path = tmp_path / "raw" / f"{sha256_bytes(content)}.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(content)
            self.manifest_rows.append(
                {
                    "request_id": stable_request_id(requested_endpoint, params),
                    "endpoint": requested_endpoint,
                    "response_hash": sha256_bytes(content),
                    "cache_path": str(cache_path),
                    "record_count": 0,
                    "response_bytes": len(content),
                    "attempts": 1,
                }
            )
            return payload

    first = ResumableDiscoveryRequester(
        FakeLiveRequester(),  # type: ignore[arg-type]
        completion_dir=tmp_path / "completed",
        token="secret",
    )
    assert first.get_json(endpoint, params=params, timeout=30.0) == payload

    resumed_live = FakeLiveRequester(fail=True)
    resumed = ResumableDiscoveryRequester(
        resumed_live,  # type: ignore[arg-type]
        completion_dir=tmp_path / "completed",
        token="secret",
    )
    assert resumed.get_json(endpoint, params=params, timeout=30.0) == payload
    assert resumed_live.http_requests_attempted == 0
    assert resumed_live.cached_bytes_accounted > 0
    assert resumed.manifest_rows[0]["resumed_from_cache"] is True


def test_probe_cli_records_setup_runtime_failure_as_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(PROBE_DIR))
    import contract_history_probe

    audit = (
        PROBE_DIR.parent
        / "20260722-broad-conflict-prior-close-iv-v0"
        / "artifacts"
        / "primary"
        / "option_underlying_price_audit.csv"
    )

    def fail_probe(**_kwargs: object) -> dict[str, Any]:
        raise RuntimeError("redacted setup transport failure")

    monkeypatch.setenv("EODHD_API_TOKEN", "test-only-secret")
    monkeypatch.setattr(contract_history_probe, "run_live_probe", fail_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "contract_history_probe.py",
            "--output",
            str(tmp_path / "output"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--price-audit",
            str(audit),
        ],
    )

    assert contract_history_probe.main() == 2
    manifest = json.loads(
        (tmp_path / "output" / "contract_history_probe_manifest.json").read_text()
    )
    assert manifest["status"] == "blocked_options_download_incomplete"
    assert manifest["credential_exposures"] == 0


def test_candidate_groups_follow_frozen_expiry_and_atm_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(PROBE_DIR))
    from contract_history_probe import ContractDescriptor, ProbeTarget, candidate_groups

    target = ProbeTarget("AAL", date(2025, 1, 3), date(2025, 1, 2), 13.0)

    def contract(contract_id: str, expiry: date, kind: str, strike: float) -> ContractDescriptor:
        return ContractDescriptor(contract_id, "AAL", expiry, kind, strike)

    contracts = [
        contract("late-call", date(2025, 1, 17), "call", 13.0),
        contract("late-put", date(2025, 1, 17), "put", 13.0),
        contract("near-12-call", date(2025, 1, 10), "call", 12.0),
        contract("near-12-put", date(2025, 1, 10), "put", 12.0),
        contract("near-13-call-only", date(2025, 1, 10), "call", 13.0),
        contract("near-14-call", date(2025, 1, 10), "call", 14.0),
        contract("near-14-put", date(2025, 1, 10), "put", 14.0),
    ]

    groups = candidate_groups(contracts, target=target)

    assert [(group.expiration_date, group.strike) for group in groups] == [
        (date(2025, 1, 10), 14.0),
        (date(2025, 1, 10), 12.0),
        (date(2025, 1, 17), 13.0),
    ]
    assert groups[0].call_contract_ids == ("near-14-call",)
    assert groups[0].put_contract_ids == ("near-14-put",)


def test_probe_pair_does_not_fallback_after_nearest_pair_quality_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(PROBE_DIR))
    from contract_history_probe import (
        ContractDescriptor,
        ContractHistory,
        ProbeTarget,
        select_probe_pair,
    )

    target = ProbeTarget("AAL", date(2025, 1, 3), date(2025, 1, 2), 13.0)
    contracts = (
        ContractDescriptor("near-call", "AAL", date(2025, 1, 10), "call", 13.0),
        ContractDescriptor("near-put", "AAL", date(2025, 1, 10), "put", 13.0),
        ContractDescriptor("late-call", "AAL", date(2025, 1, 17), "call", 13.0),
        ContractDescriptor("late-put", "AAL", date(2025, 1, 17), "put", 13.0),
    )
    requested: list[str] = []

    def load_history(contract_id: str) -> ContractHistory:
        requested.append(contract_id)
        option_type = "call" if contract_id.endswith("call") else "put"
        near = contract_id.startswith("near")
        expiration = "2025-01-10" if near else "2025-01-17"
        record = provider_record(
            contract=contract_id,
            option_type=option_type,
            observation_date="2025-01-02",
            tradetime="2024-12-20",
            expiration_date=expiration,
            open_interest=5 if contract_id == "near-call" else 100,
        )
        return ContractHistory(contract_id, f"request-{contract_id}", (record,))

    result = select_probe_pair(target=target, contracts=contracts, load_history=load_history)

    assert result.available is False
    assert result.reason == "selected_pair_open_interest_below_10"
    assert result.selected_expiration_date == date(2025, 1, 10)
    assert requested == ["near-call", "near-put"]


def test_probe_pair_fetches_equal_atm_strikes_before_frozen_tie_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(PROBE_DIR))
    from contract_history_probe import (
        ContractDescriptor,
        ContractHistory,
        ProbeTarget,
        select_probe_pair,
    )

    target = ProbeTarget("AAL", date(2025, 1, 3), date(2025, 1, 2), 100.0)
    contracts = tuple(
        ContractDescriptor(f"{label}-{side}", "AAL", date(2025, 1, 10), side, strike)
        for label, strike in (("lower", 50.0), ("upper", 200.0))
        for side in ("call", "put")
    )
    requested: list[str] = []

    def load_history(contract_id: str) -> ContractHistory:
        requested.append(contract_id)
        label, option_type = contract_id.split("-")
        strike = 50.0 if label == "lower" else 200.0
        record = provider_record(
            contract=contract_id,
            option_type=option_type,
            observation_date="2025-01-02",
            tradetime="2024-12-20",
            expiration_date="2025-01-10",
            strike=strike,
            open_interest=20 if label == "lower" else 100,
        )
        return ContractHistory(contract_id, f"request-{contract_id}", (record,))

    result = select_probe_pair(target=target, contracts=contracts, load_history=load_history)

    assert result.available is True
    assert result.selected_strike == 200.0
    assert requested == ["lower-call", "lower-put", "upper-call", "upper-put"]
