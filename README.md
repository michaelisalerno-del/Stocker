# Stocker

![CI](https://github.com/michaelisalerno-del/Stocker/actions/workflows/ci.yml/badge.svg)

Stocker is one trading research and execution application. Its operational workflow is
**Market → Method → Run**: choose a market, select **Session HARD**, then save a PAPER run.
A run retains its identity, method version, frozen specification, universe and audit history.
The current method, `SESSION_HARD_CAUSAL_Q1_ACQUISITION_V9`, is **PAPER-only**.

The existing market catalogue includes US All/NASDAQ/NYSE, Canada TSX, UK LSE,
Germany Xetra, France Paris, Netherlands Amsterdam, Switzerland SIX, Australia ASX,
Hong Kong HKEX, Japan TSE, South Korea KRX and South Africa JSE. Availability depends
on listing data, calendars and observed IBKR permissions. Non-US method use is an
**unvalidated cross-market PAPER test**. This release adds no markets or trading hours.
US sessions continue to use exchange calendars and America/New_York, including DST.

Session HARD preserves Range5 HIGH250 → RV10 HIGH50 → RV15 HIGH30, its frozen model,
scores, causal tick entries and trade geometry. Production and PAPER PRE history is
exclusively IBKR-originated. Research vendor data is separate. Shared execution
admission applies account exposure and broker capacity checks; changed admission
does **not** establish historical trade-count or performance parity.

## Start locally

Prerequisites: Git, Python **3.12**, `uv`; Node **22** and npm for browser tests.
IB Gateway/TWS and verified account permissions are needed only for integrated
execution, not normal tests or the standalone dashboard.

```bash
uv sync --locked --all-groups
npm ci
npx playwright install chromium
uv run --no-sync stocker stage10-dashboard \
  --runs-config configs/runs.example.yaml \
  --ibkr-config configs/ibkr.example.yaml \
  --database .stocker/stage8-runtime.sqlite3
```

Open [the local dashboard](http://127.0.0.1:8000). Standalone mode reads state and
saves configuration; it does **not** connect IBKR or activate execution. Use copies
of the example configuration for actual work. Local configs and databases belong
outside version control.

The integrated `stage10-run` command connects the existing engine and dashboard:
enabled runs retain their algorithmic PAPER authority. Read the
[dashboard guide](docs/STAGE10_DASHBOARD.md) and
[deployment/recovery runbook](docs/UNATTENDED_RECOVERY.md) before using it.
“Pause new entries” preserves current positions, history and broker-held protection.

## Server installation

Prepare a separate release directory; never test installation in the running environment.

```bash
uv sync --locked --no-default-groups --group server
uv run --no-sync stocker stage10-dashboard \
  --runs-config configs/runs.example.yaml \
  --ibkr-config configs/ibkr.example.yaml \
  --database .stocker/standalone.sqlite3
```

`bash scripts/bootstrap_server.sh` uses the same locked server selection.
Service launch uses the prepared `.venv/bin/stocker` directly or `uv run --no-sync`;
plain `uv run` can add the default research/dev groups. The server retains
scikit-learn 1.8.0 and joblib because the frozen model requires them.

The installed `stocker` and `stocker-mcp` launchers select the verified NumPy x86
V2/V3 numerical path before imports; tests and server smoke use the same profile.
NumPy's AVX-512 logarithm path produced one-bit differences in frozen scores on some
CI runners. Exact fixture assertions and formulas remain unchanged. ARM is unchanged.
Direct library integrations must call `stocker_launcher.configure_numeric_runtime()`
before importing NumPy or Stocker numerical modules. After editing the force-included
launcher locally, rebuild it with `uv sync --locked --all-groups --reinstall-package stocker`.

Unauthenticated access is local-only. Remote use requires an explicitly protected
HTTPS deployment. The [dashboard security modes](docs/STAGE10_DASHBOARD.md#security)
cover every API and stream, including protection against backend/proxy bypass.

## Configuration and admission

- `configs/runs.example.yaml`: listing snapshot reference and saved runs.
- `configs/ibkr.example.yaml`: separate PAPER/LIVE routing, expected accounts,
  bounded request timeouts and configured data budget.
- Each run's `risk.risk_per_trade` is a fraction; the dashboard displays percent
  (0.001 is 0.1%). `risk.max_concurrent_positions` counts account/environment
  exposure across runs, including unresolved entries.
- `risk.max_gross_notional` is an explicit positive ceiling in the verified account
  currency. Existing configurations still load without it, but new entry is blocked
  with `EXPOSURE_POLICY_REQUIRED` until an operator sets it. No leverage default is invented.

See [execution admission](docs/execution_safety.md#shared-entry-admission) for
currency, pending exposure, broker credit preview and migration details. The IBKR account
may keep its default base currency: shared execution converts foreign stock risk and
notional using fresh broker FX, with verified quotation units and permitted order quantities
for every selected market. Quantities round down within risk/capacity limits. Missing conversion
evidence blocks entry; this does not convert cash or change method prices.
A configured market-data budget is not proof of IBKR entitlement or complete tick coverage.

An admission prepared under an older run configuration is rejected before submission.
Terminal broker status can precede execution details: reported fills remain reserved
until reconciled, including across restart. See the
[completed local acceptance evidence](docs/robustness-implementation.md#continuation-acceptance).

## Checks

The opt-in [dual-feed equivalence diagnostic](docs/DUAL_FEED_DIAGNOSTIC.md) compares
the existing TBT tape with ordinary IBKR trade-volume observations on a dedicated
read-only PAPER session. It cannot enable runs or replace the production feed.

```bash
bash scripts/check.sh
```

This independently reports format, lint, typing, Python, frontend and isolated
server-install smoke results, then fails if any failed. Run one with
`bash scripts/check.sh format` (or `lint`, `typing`, `python`, `frontend`, `server`).
CI uses those same checks as separate required jobs; a failure does not hide the others.
Normal tests use fake brokers, temporary databases and isolated configuration.

Recommended main-branch protection: require all six CI checks and review before merge,
block force pushes and deletion, and require checks against the proposed main revision.
These are operator recommendations; repository administration settings are not changed by code.

## Repository map

- `packages/stocker_core`: market/method catalogue, configuration, CLI and run identities.
- `packages/stocker_execution`: method composition, IBKR adapter, admission, ledger and runtime.
- `packages/stocker_dashboard`: HTTP controls, reads, security and static frontend.
- `packages/stocker_data`: history/cache, datasets and calendars.
- `packages/stocker_research`, `packages/stocker_backtest`: separate research and replay tools.
- `packages/stocker_mcp`: read-only research/diagnostic integration.
- `apps/desktop`, `apps/server`: workspace instructions and launch helpers.
- `configs`, `universes`: configuration examples and saved listing inputs.
- `research`: retained research records and frozen source evidence.
- `tests`, `scripts`, `docs`: regressions, release/restore tools and operational contracts.

Read [architecture](docs/ARCHITECTURE.md), [candidate discovery](docs/candidate-discovery.md),
[entry protection](docs/ENTRY_EXECUTION_PROTECTION.md) and
[release evidence](docs/robustness-implementation.md).
Earlier stage walkthroughs remain in the
[research guide](docs/research_harness.md#historical-readme-walkthroughs-preserved-2026-09-11);
historical deployment records remain in the recovery runbook.
