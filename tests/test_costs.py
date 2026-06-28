from stocker_backtest.costs import CostModel


def test_round_trip_cost_adds_two_sides_of_all_bps_costs() -> None:
    model = CostModel(spread_bps=1.5, commission_bps=0.5, slippage_bps=0.25)

    assert model.one_way_bps() == 2.25
    assert model.round_trip_bps() == 4.5


def test_zero_cost_model_has_no_round_trip_cost() -> None:
    model = CostModel()

    assert model.round_trip_bps() == 0.0
