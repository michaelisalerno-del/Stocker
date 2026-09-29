# Dashboard redesign and additional Saxo data — 29 September 2026

Branch `codex/slrno-dashboard-data`, based on `codex/slrno-review-fixes` at `b52bc39`. Nothing was
deployed or pushed and no broker request was made. Everything added here is display, evidence or
observation context: none of it feeds admission, sizing or management.
[Fixture screenshots](dashboard-screenshots/) show illustrative values under an offline banner.

## Dashboard

- **Status bar.** One row replaces the repeated banners: data environment, execution mode, armed
  state, connection, and a countdown to the next frozen clock computed from server time, so a
  skewed browser clock cannot mislead. LIVE ORDERS DISABLED stays in the header.
- **Setup checklist** (Overview and System): Saxo login, account, stream, futures contracts n/5,
  reference sessions n/5, option approvals (GC optional), execution mode, armed; recording and
  alerts are marked optional. It hides on Overview once every required step is done.
- **Readiness gates** on every market: Saxo · Contract · Quote · History · Option approval ·
  Strike · Cost ≤ £50 · Execution, in decision order, with the first block explained. They
  summarise the last refresh; the clock decision still re-checks everything itself.
- **Markets page:** a trade ticket (quote and sizes, spread as % of mid, frozen-model |Δ| against
  the target, provider delta, last trading time, all-in cost bar against £50); a chart with price
  and time axes, today's clock marks (closed sessions in red), the 15-minute RV window and the
  open-trade band; a market-context panel (session high/low/change, open interest, today's
  sessions, scheduled releases, the options-chain smile); a bar-style depth ladder and a
  five-level imbalance meter.
- **15-minute book history** (opened on demand): spread, top-of-book size and imbalance
  sparklines plus a sampled-depth heatmap. It is decoded in a worker thread from the recorder's
  existing rolling rows (last observation per 5-second bucket) via `/api/market/{m}/book`.
- **Opportunities:** Inspect shows a step timeline (clock, checks, option selection, admission,
  order intent, fills, decisions, closure, current state); raw JSON remains one click away.
- The Playwright fixture now covers each of these with production-shaped data.

## Additional Saxo data

| Feature | Source | Use |
|---|---|---|
| Chart v3 | `/chart/v3/charts` (v1 deprecated February 2025; contract identical) | All bar history and reference sessions |
| Session context | `PriceInfo` field group (High, Low, NetChange, PercentChange) with existing PriceInfoDetails / InstrumentPriceDetails | Display |
| Daily range | Chart v3, Horizon 1440, 20 completed sessions, once per New York day | Display |
| Chain smile | Options chain, window `option_chain_strikes` (default 11; Saxo caps a chain at 100 strikes) | Recorded with each observed clock; displayed |
| Order/position events | ENS `/ens/v1/activities/subscriptions` (Orders, Positions), SAXO_SIM only | Triggers an immediate reconcile; never trusted as state |
| Closed positions | `/port/v1/closedpositions`, SAXO_SIM only, every 10 minutes | Audited evidence; Saxo's figure shown in account base currency beside, never mixed with, the internal GBP P&L |

Provider volatility scaling is not documented by Saxo, so chain values stay in provider units
and no implied-minus-realised spread is computed. The frozen model's annualised RV15 volatility
is shown alongside for comparison. Saxo removes intraday closed positions after settlement,
which is why that fetch runs through the trading day while SIM execution is configured.

Not available from Saxo OpenAPI: a full trade tape, order-by-order depth, historical L2, news or
an economic calendar, and exchange holidays (sessions come from instrument TradingSessions).

## Optional configuration

```yaml
event_calendar_file: /etc/stocker/v1/event-calendar.yaml   # see configs/event-calendar.example.yaml
alerts:
  url_file: /etc/stocker/v1/alert-url.json   # mode 0600, owned by the service user
  stream_down_seconds: 120
  login_warning_hours: 24
option_chain_strikes: 11
```

The alert file is `{"url": "https://ntfy.sh/<private-topic>"}` (any HTTPS endpoint accepting a
plain-text POST). Alerts cover exposure exceptions, stopped entries, failed workers, a stream
down past the threshold and Saxo login expiry. They send only coded reasons, repeat only on
change, report resolution, and a delivery failure never affects trading. The calendar ships
with the two fixed weekly EIA releases; dated releases (CPI, FOMC) must be copied from the
official calendars, and holiday shifts must be entered by hand.

## Trading-path findings

1. **Fixed — option session gate.** The option quote and exit-cutoff checks accepted only
   `Open`/`OpenForTrading`, which Saxo's `InstrumentSessionState` does not contain. They now
   accept `AutomatedTrading`; auctions, breaks, halts, pre/post sessions and unknown values stay
   blocked. Without this every entry would have been rejected once approvals existed.
2. **Decision needed — price quality.** Internal paper fills require `PriceTypeBid`/`Ask` equal
   to `Tradable`, per `SAXO-SETUP.md` ("an indicative quote … cannot generate an internally
   simulated fill"). Saxo now marks `Tradable` obsolete and its pricing guide describes
   `Indicative` as "in most cases … as relevant as a Tradable price" (FX options excepted). If
   live SIM futures-option quotes arrive as `Indicative`, no paper fill can occur. This policy
   was left unchanged; check a live option quote once contracts are selected, then decide.

## Rollout notes

No schema migration. New optional config keys default to off. The chain window widens from 3 to
11 strikes on the existing single chain subscription per market (Saxo's minimum chain refresh is
2 s). The ENS subscription and closed-position fetch are dormant unless `execution_mode` is
`SAXO_SIM`. Chart v3 is exercised for the first time when futures contracts are selected; watch
`history_problem` on the Markets gates after that change.

## Checks executed here

- Python suite **536 passed** (7 existing warnings); Ruff format and lint; mypy 120 files.
- All three Playwright scripts passed (Node 24, Playwright 1.62.1), including new assertions for
  gates, setup, countdown, ticket, sessions, events, price context, chart marks, imbalance,
  ladder bars, smile, book history heatmap, timeline and Saxo-reported P&L.
- Runtime-only locked install smoke passed with network connections refused.
- Dashboard benchmark (five clients, 400 closed trades): 0 broker requests; Overview response
  3.6 → 8.5 KB with gates and setup; page p95 latency stayed at or below about 1 ms.
