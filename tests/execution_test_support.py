"""Synthetic execution identity used only by routing/control tests."""

from datetime import timedelta

import pytest

import stocker_core.strategies as original_registry
from stocker_core.markets import MarketId
from stocker_core.methods import session_hard_universe
from stocker_core.strategies import StrategyDefinition
from stocker_execution.session_hard_method import TradeEvent
from stocker_execution.session_hard_structure_d import SessionHardStructureDStrategy

TEST_METHOD = StrategyDefinition(
    "TEST_EXECUTION",
    "TEST_EXECUTION_V1",
    "TEST_EXECUTION",
    "Test execution",
    tuple(MarketId),
    ("PAPER", "LIVE"),
    lambda market: {"market": market.value, "method_id": "TEST_EXECUTION"},
    session_hard_universe,
)


class SyntheticExecutionMethod(SessionHardStructureDStrategy):
    """Deterministic intent producer for account tests, never an installed method."""

    def __init__(self):
        super().__init__()
        self.cohort_labels = {}

    def observe_trades(self, events):
        from stocker_execution.session_hard_structure_d import EntryBar

        return self.observe_entry_bars(
            {
                con_id: tuple(
                    EntryBar(e.timestamp.replace(second=0), e.price, e.price, e.price) for e in rows
                )
                for con_id, rows in events.items()
            }
        )

    def expire_unobservable(self, con_id, reason):
        raise AssertionError(reason)

    def restore_runtime_state(self, store, run_id):
        pass

    def save_runtime_state(self, store):
        pass

    async def advance_runtime_state(self, source, instruments, store, now, logger):
        pass


def synthetic_run(run):
    if run.strategy == TEST_METHOD.config_name:
        return run.model_copy(update={"method_spec": {"test_only_intents": True, "execution": {}}})
    return run


@pytest.fixture
def execution_method(monkeypatch):
    import stocker_core.strategies as registry
    import stocker_dashboard.controls as controls_module
    import stocker_execution.runtime as runtime_module
    from stocker_core.methods import validate_run_method
    from stocker_execution.runtime import StockerRuntime
    from stocker_execution.strategy_factory import create_strategy

    def validate(run):
        if run.strategy != TEST_METHOD.config_name:
            validate_run_method(run)

    monkeypatch.setattr(runtime_module, "validate_run_method", validate)
    monkeypatch.setattr(controls_module, "validate_run_method", validate)

    original_init = StockerRuntime.__init__
    original_apply = StockerRuntime.apply_runs_config

    async def synthetic_apply(self, config, *args, **kwargs):
        config = config.model_copy(update={"runs": tuple(synthetic_run(r) for r in config.runs)})
        return await original_apply(self, config, *args, **kwargs)

    def synthetic_init(self, *args, **kwargs):
        config = kwargs["config"]
        kwargs["config"] = config.model_copy(
            update={"runs": tuple(synthetic_run(run) for run in config.runs)}
        )
        original_init(self, *args, **kwargs)
        self._stage5_by_strategy["SESSION_HARD_HV_V1"] = kwargs["stage5"]
        source = self._entry_source
        if not hasattr(source, "trades_for"):

            async def trades_for(instruments, signals):
                bars = await source.bars_for(
                    None,
                    instruments,
                    session=signals[0].session,
                    now=self._clock(),
                    signals=signals,
                )
                return {
                    con_id: tuple(
                        TradeEvent(
                            signals[0].t0 + timedelta(seconds=60)
                            if signals[0].method_spec_hash is not None
                            else b.timestamp + timedelta(seconds=45),
                            b.open,
                            i + 1,
                        )
                        for i, b in enumerate(rows)
                    )
                    for con_id, rows in bars.items()
                }

            source.trades_for = trades_for
            source.trade_errors = {}
            source.release_trades = lambda con_id: None
            source.release_unused_trades = lambda retained: None

    monkeypatch.setattr(StockerRuntime, "__init__", synthetic_init)
    monkeypatch.setattr(StockerRuntime, "apply_runs_config", synthetic_apply)

    # Legacy account fixtures have no method identity. The production method tests
    # carry an explicit specification and use the real package.
    def construct(identity, version, market, *, clock=None):
        if version == TEST_METHOD.strategy_version:
            return SyntheticExecutionMethod()
        return create_strategy(identity, version, market, clock=clock)

    monkeypatch.setattr(runtime_module, "create_strategy", construct)
    original_ensure = StockerRuntime._ensure_strategy

    def ensure(self, run, execution, now):
        if run.strategy == TEST_METHOD.config_name:
            run = run.model_copy(
                update={
                    "strategy_id": TEST_METHOD.strategy_id,
                    "strategy_version": TEST_METHOD.strategy_version,
                }
            )
        return original_ensure(self, run, execution, now)

    monkeypatch.setattr(StockerRuntime, "_ensure_strategy", ensure)

    # CLI reload tests may leave already-collected test modules holding the old class.
    for item in {registry, original_registry}:
        installed = (*item.installed_strategies(), TEST_METHOD)
        monkeypatch.setattr(item, "installed_strategies", lambda installed=installed: installed)
        monkeypatch.setattr(
            controls_module, "installed_strategies", lambda installed=installed: installed
        )
        monkeypatch.setattr(
            item,
            "get_strategy",
            lambda identity, version, installed=installed: next(
                method
                for method in installed
                if (method.strategy_id, method.strategy_version) == (identity, version)
            ),
        )
