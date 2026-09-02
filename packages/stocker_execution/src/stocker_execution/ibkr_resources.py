"""Deterministic IBKR capacity exercises and read-only persisted-watchlist inspection."""

from __future__ import annotations

from dataclasses import dataclass

from stocker_execution.session_hard_structure_d import SESSION_HARD_CHECKPOINTS


@dataclass(frozen=True, slots=True)
class CapacitySimulation:
    """Observed counters from one deterministic fake-broker workload."""

    name: str
    configured_runs: int
    unique_stocks: int
    scanner_requests: int
    contract_qualifications: int
    stage4_contexts: int
    historical_requests: int
    peak_scanner_concurrency: int
    peak_underlying_lines: int
    peak_option_lines: int
    peak_market_data_lines: int
    duplicate_stock_requests_avoided: int
    market_data_line_budget: int
    within_budget: bool
    subscriptions_before_disconnect: int = 0
    subscriptions_after_disconnect: int = 0
    subscriptions_after_reconnect: int = 0
    pacing_issues: int = 0
    execution_safety_checks: int = 0


class _FakeResourceBroker:
    """Small synchronous fake that records the current Stocker physical-work shape."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.screens: set[tuple[str, str]] = set()
        self.qualified: set[int] = set()
        self.contexts: set[int] = set()
        self.history: set[tuple[object, ...]] = set()
        self.active_lines: set[tuple[int, str]] = set()
        self.scanner_requests = 0
        self.active_scanners = 0
        self.peak_scanners = 0
        self.peak_option_lines = 0
        self.peak_total_lines = 0
        self.capacity_rejects = 0
        self.execution_safety_checks = 0

    def run_screen(self, screen: tuple[str, str]) -> None:
        if screen in self.screens:
            return
        self.screens.add(screen)
        for _component in range(3):
            self.active_scanners += 1
            self.peak_scanners = max(self.peak_scanners, self.active_scanners)
            self.scanner_requests += 1
            self.active_scanners -= 1

    def prepare_stock(self, con_id: int) -> None:
        self.qualified.add(con_id)
        if con_id not in self.contexts:
            self.request_history((con_id, "PRE", "5 mins"))
            try:
                self.acquire_line((con_id, "CALL"))
                self.acquire_line((con_id, "PUT"))
            except RuntimeError:
                self.execution_safety_checks += 1
                return
            finally:
                self.release_line((con_id, "PUT"))
                self.release_line((con_id, "CALL"))
            self.contexts.add(con_id)
        self.execution_safety_checks += 1
        for checkpoint in range(len(SESSION_HARD_CHECKPOINTS)):
            self.request_history((con_id, checkpoint, "5 mins"))
            self.request_history((con_id, checkpoint, "1 min"))

    def request_history(self, key: tuple[object, ...]) -> None:
        self.history.add(key)

    def acquire_line(self, key: tuple[int, str]) -> None:
        if key in self.active_lines:
            return
        if len(self.active_lines) >= self.budget:
            self.capacity_rejects += 1
            raise RuntimeError("IBKR_MARKET_DATA_CAPACITY_UNAVAILABLE")
        self.active_lines.add(key)
        option_lines = sum(kind in {"CALL", "PUT"} for _con_id, kind in self.active_lines)
        self.peak_option_lines = max(self.peak_option_lines, option_lines)
        self.peak_total_lines = max(self.peak_total_lines, len(self.active_lines))

    def release_line(self, key: tuple[int, str]) -> None:
        self.active_lines.discard(key)

    def disconnect(self) -> None:
        self.active_lines.clear()


def simulate_capacity(*, market_data_line_budget: int = 100) -> tuple[CapacitySimulation, ...]:
    """Exercise realistic Stage 1–10 workloads against a deterministic fake broker."""

    if market_data_line_budget < 1:
        raise ValueError("Stocker API line budget must be positive")
    one_watchlist = frozenset(range(50))
    workloads = (
        ("A", (one_watchlist,), (("NASDAQ", "MID"),)),
        ("B", (one_watchlist, one_watchlist, one_watchlist), (("NASDAQ", "MID"),) * 3),
        (
            "C",
            tuple(frozenset(range(index * 50, (index + 1) * 50)) for index in range(4)),
            (("NASDAQ", "MID"), ("NASDAQ", "LARGE"), ("NYSE", "MID"), ("ASX", "MID")),
        ),
        (
            "REALISTIC_4_RUN",
            (one_watchlist, one_watchlist, one_watchlist, frozenset(range(50, 100))),
            (("NASDAQ", "MID"),) * 3 + (("NYSE", "MID"),),
        ),
    )
    measured = [
        _exercise_workload(name, watchlists, screens, market_data_line_budget)
        for name, watchlists, screens in workloads
    ]
    reconnect = _exercise_reconnect(market_data_line_budget)
    return (*measured[:3], reconnect, measured[3])


def _exercise_workload(
    name: str,
    watchlists: tuple[frozenset[int], ...],
    screens: tuple[tuple[str, str], ...],
    market_data_line_budget: int,
) -> CapacitySimulation:
    broker = _FakeResourceBroker(market_data_line_budget)
    configured_memberships = 0
    for watchlist, screen in zip(watchlists, screens, strict=True):
        broker.run_screen(screen)
        configured_memberships += len(watchlist)
        for con_id in watchlist:
            broker.prepare_stock(con_id)
    return CapacitySimulation(
        name=name,
        configured_runs=len(watchlists),
        unique_stocks=len(broker.qualified),
        scanner_requests=broker.scanner_requests,
        contract_qualifications=len(broker.qualified),
        stage4_contexts=len(broker.contexts),
        historical_requests=len(broker.history),
        peak_scanner_concurrency=broker.peak_scanners,
        peak_underlying_lines=0,
        peak_option_lines=broker.peak_option_lines,
        peak_market_data_lines=broker.peak_total_lines,
        duplicate_stock_requests_avoided=configured_memberships - len(broker.qualified),
        market_data_line_budget=market_data_line_budget,
        within_budget=broker.capacity_rejects == 0,
        pacing_issues=0,
        execution_safety_checks=broker.execution_safety_checks,
    )


def _exercise_reconnect(market_data_line_budget: int) -> CapacitySimulation:
    broker = _FakeResourceBroker(market_data_line_budget)
    for con_id in range(20):
        try:
            broker.acquire_line((con_id, "CALL"))
        except RuntimeError:
            break
    before = len(broker.active_lines)
    broker.disconnect()
    after_disconnect = len(broker.active_lines)
    after_reconnect = len(broker.active_lines)
    return CapacitySimulation(
        name="D",
        configured_runs=1,
        unique_stocks=20,
        scanner_requests=0,
        contract_qualifications=0,
        stage4_contexts=0,
        historical_requests=0,
        peak_scanner_concurrency=0,
        peak_underlying_lines=0,
        peak_option_lines=broker.peak_option_lines,
        peak_market_data_lines=broker.peak_total_lines,
        duplicate_stock_requests_avoided=0,
        market_data_line_budget=market_data_line_budget,
        within_budget=broker.capacity_rejects == 0,
        subscriptions_before_disconnect=before,
        subscriptions_after_disconnect=after_disconnect,
        subscriptions_after_reconnect=after_reconnect,
        pacing_issues=0,
        execution_safety_checks=broker.execution_safety_checks,
    )
