"""The following expiry's chain window and the following contract month: recorded, never traded."""

import asyncio

import pytest

from saxo_support import AT, FUTURE, setup
from stocker_execution.config import ContractSelection, FuturesConfig
from stocker_execution.contracts import key
from stocker_execution.saxo_data import board_target, chain_identity


def test_next_chain_is_a_separate_recorder_instrument():
    assert chain_identity(FUTURE)["asset_type"] == "OptionsChain"
    assert chain_identity(FUTURE, "next")["asset_type"] == "OptionsChainNext"
    assert key(chain_identity(FUTURE, "next")) != key(chain_identity(FUTURE))
    assert board_target("CL") == ("CL", "") and board_target("CL:next") == ("CL", "next")


def test_next_contract_configuration_rules():
    pinned = ContractSelection(
        environment="SAXO_SIM",
        uic=100,
        symbol="CLX6",
        exchange="NYMEX",
        contract_month="2026-11",
        approval="fixture approval text",
    )
    following = pinned.model_copy(
        update={"uic": 777, "symbol": "CLZ6", "contract_month": "2026-12"}
    )
    config = FuturesConfig(contracts={"CL": pinned}, next_contracts={"CL": following})
    assert config.next_contracts["CL"].symbol == "CLZ6"
    with pytest.raises(ValueError, match="NEXT_CONTRACT_MUST_DIFFER"):
        FuturesConfig(contracts={"CL": pinned}, next_contracts={"CL": pinned})
    with pytest.raises(ValueError, match="CROSS_ENVIRONMENT"):
        FuturesConfig(
            next_contracts={"CL": following.model_copy(update={"environment": "SAXO_LIVE"})}
        )


def test_next_chain_subscribes_the_following_root_and_records_separately(tmp_path):
    async def scenario():
        _, data, _ = setup(tmp_path)
        data.subscriptions["option"]["kind"] = "PRICE"  # the fixture entry carries no kind
        state = data.markets["CL"]
        state.option_root, state.next_option_root = 4953, 4960
        calls = []

        async def request(method, path, **kwargs):
            calls.append((method, path, kwargs))
            if method == "POST":
                return {
                    "Snapshot": {
                        "Expiries": [
                            {
                                "Index": 0,
                                "Expiry": "2026-10-06",
                                "MidStrikePrice": 70,
                                "Strikes": [{"Index": 3, "Strike": 70, "Call": {"Bid": 1}}],
                            }
                        ]
                    },
                    "RefreshRate": 2000,
                    "InactivityTimeout": 30,
                }
            return {}

        data.client.request = request
        await data.subscribe_board(state, "next")
        ref, sub = next((r, s) for r, s in data.subscriptions.items() if s["target"] == "CL:next")
        assert sub["kind"] == "BOARD" and sub["arguments"]["Identifier"] == 4960
        assert state.next_option_board["Expiries"][0]["Index"] == 0 and state.option_board == {}
        assert key(chain_identity(FUTURE, "next")) in data.recorder.windows
        assert key(chain_identity(FUTURE)) not in data.recorder.windows
        assert state.capabilities["chain_recording_problem_next"] == ""
        # Updates route to the following expiry's board only.
        await data.receive(
            {
                "reference": ref,
                "message_id": "1",
                "payload": {"Expiries": [{"Index": 0, "MidStrikePrice": 71}]},
            },
            AT.timestamp(),
        )
        assert (
            state.next_option_board["Expiries"][0]["MidStrikePrice"] == 71
            and state.option_board == {}
        )
        # The window is centred on that expiry's own money.
        await data.focus_next_board(state)
        patch = next(c for c in calls if c[0] == "PATCH")
        assert patch[2]["body"] == {
            "Expiries": [{"Index": 0, "StrikeStartIndex": 0}],
            "MaxStrikesPerExpiry": data.config.option_chain_strikes,
        }
        assert state.capabilities["next_chain_problem"] == ""
        # A second call with the same root is a no-op; a missing root subscribes nothing.
        await data.subscribe_board(state, "next")
        assert sum(1 for c in calls if c[0] == "POST") == 1
        state.next_option_root = None
        await data.subscribe_board(data.markets["GC"], "next")
        assert not any(s["target"] == "GC:next" for s in data.subscriptions.values())

    asyncio.run(scenario())


def test_next_contract_is_streamed_recorded_and_idempotent(tmp_path):
    async def scenario():
        _, data, _ = setup(tmp_path)
        data.subscriptions["option"]["kind"] = "PRICE"  # the fixture entry carries no kind
        following = ContractSelection(
            environment="SAXO_SIM",
            uic=777,
            symbol="CLZ6",
            exchange="NYMEX",
            contract_month="2026-12",
            approval="pre-roll recording approved in the fixture",
        )
        data.config = data.config.model_copy(update={"next_contracts": {"CL": following}})
        state = data.markets["CL"]
        raw = {
            "AssetType": "ContractFutures",
            "Uic": 777,
            "Symbol": "CLZ6",
            "ContractSize": 1000,
            "ExpiryDate": "2026-11-20",
            "CurrencyCode": "USD",
            "Exchange": {"ExchangeId": "NYMEX"},
            "TickSize": 0.01,
            "PriceToContractFactor": 1000,
        }
        calls = []
        original = data.client.request

        async def request(method, path, **kwargs):
            calls.append((method, path))
            if path.endswith("/777/ContractFutures"):
                return raw
            if method == "POST":
                return {
                    "Snapshot": {"Quote": {"Bid": 70, "Ask": 70.01}},
                    "RefreshRate": 1000,
                    "InactivityTimeout": 30,
                }
            return await original(method, path, **kwargs)

        data.client.request = request
        await data.subscribe_next_contract(state)
        identity, price = data.next_contracts["CL"]
        assert identity["symbol"] == "CLZ6" and identity["role"] == "NEXT_CONTRACT"
        assert price.value["Quote"]["Ask"] == 70.01
        assert key(identity) in data.recorder.windows
        assert state.capabilities["next_contract"]["status"] == "SUBSCRIBED"
        assert data.price_target("CL:next")[1] is identity
        sub = next(s for s in data.subscriptions.values() if s["target"] == "CL:next")
        assert "MarketDepth" in sub["arguments"]["FieldGroups"]
        await data.subscribe_next_contract(state)  # already subscribed: nothing happens
        assert sum(1 for m, _ in calls if m == "POST") == 1
        # A contract outside the market's family is refused and recorded as a problem.
        data.config = data.config.model_copy(
            update={"next_contracts": {"GC": following.model_copy(update={"symbol": "GCG7"})}}
        )
        data.markets["GC"].identity = {**FUTURE, "market": "GC", "uic": 5}
        await data.subscribe_next_contract(data.markets["GC"])
        assert "GC" not in data.next_contracts
        assert data.markets["GC"].capabilities["next_contract"]["problem"] == (
            "STANDARD_FUTURE_FAMILY_NOT_VERIFIED"
        )

    asyncio.run(scenario())
