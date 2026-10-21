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
  Strike · Cost ≤ £1,000 · Execution, in decision order, with the first block explained. They
  summarise the last refresh; the clock decision still re-checks everything itself.
- **Markets page:** a trade ticket (quote and sizes, spread as % of mid, frozen-model |Δ| against
  the target, provider delta, last trading time, all-in cost bar against the £1,000 guard); a chart with price
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

### Phone-first update — 1 October 2026 (user's choice of priorities)
Display only; no trading rule or broker request changed.
- **Status:** phones show one line (a dot, "All normal" or the problems, and the next clock); a
  tap opens the full chips and Pause. Desktop keeps the chip row. The countdown also gives the
  London time.
- **Overview:** Today (net after costs, trades, wins, open), the open-trade estimate, realised
  since start and slots; today's trades one per line (London time, option, bought → sold, result
  or the live bid estimate); each card shows the future's mid and change, any open trade with its
  estimate and exit, the first failing check in plain English (muted while the next clock is more
  than an hour away), today's trades in that market and releases, with the gates folded into
  "n of m checks pass".
- **Opportunities:** each row has its option, bought → sold and result after costs; a "Trades
  only" filter; on phones each row is a labelled card without empty values.
- **Execution:** recent trades one row each (paid and result); fills name the market and option
  with Entry/Exit; empty tables say "None."
- **Markets:** tabs instead of the dropdown; open trade, block and folded checks first, then the
  ticket and context, then the chart and the book. **System:** real-time first; API counters
  folded into "Saxo API usage".
- Per-trade figures come from `Store.trade_results`, the same sums as the realised total.

### Visual refresh — 4 October 2026 (user: "it's got to look incredible")
Display only; every element id, control and test hook is unchanged, so the Playwright suite and
the server are untouched.
- **Design system in `dashboard.css`:** colour, radius, shadow and type tokens on `:root`; layered
  surfaces with hairlines instead of uniform boxes; semantic green / amber / red kept separate from
  the lime brand accent; a colour per market (CL amber, ES blue, GC gold, NQ violet) on card
  stripes, tabs and today's trade tags; system fonts (SF Pro / SF Mono on Apple devices) because
  the CSP forbids CDN fonts and nothing needs downloading.
- **Header** is sticky and translucent with the London clock; on phones the nav becomes a fixed
  bottom tab bar (safe-area aware) and the header keeps only the brand and the PAPER badge.
- **Status bar** carries its health in a coloured dot on the bar itself (`data-tone` ok / warn /
  bad set beside the existing `#status-dot`); the next-clock chip is the prominent element.
- **Overview cards** are four across on wide screens, equal height, with a 28px tabular mid price
  that flashes briefly when it changes (`data-flash`), a short state pill (full explanation in its
  tooltip), and the folded checks summary now shows one dot per gate.
- **Markets page** leads with the same price hero; the chart has an area fill, a lighter grid and,
  on phones, a narrower coordinate space (viewBox 400) so axis text stays legible; the ladder,
  imbalance meter and sparklines are restyled, not changed.
- Tables: uppercase sticky heads, right-aligned results, hover rows; phones keep the card-per-row
  layout and use 16px inputs so iOS does not zoom.
- Favicon `static/favicon.svg`. Screenshots regenerated from the same fixture.
- Setup checklist (user's follow-up): it starts folded on every page — the summary line carries
  "n of m required steps done" — and the markup no longer ships it `open`. Overview still shows
  the (folded) section only while a required step is missing.
- Archive runway (same day, from the review of what Saxo delivers against what is kept): the
  recorder samples its archive size once a minute (`archive_growth` in `/api/system`), the
  System tile shows the stored size in GiB once the cap is above 1 GiB, the growth in GiB/day and
  the days left before `archive_max_bytes` stops captures, and the status bar lists
  "Archive n days left" under 14 days. The same change widened the chain window to 61 strikes and
  the cap to 70 GiB on the server.
- Chart stream (same day; MARKET-DATA.md): System's per-market lines show the chart stream's
  state, how many boundary bars decisions took from it and how many REST confirmed; the status
  bar says "Chart stream fell back to REST" once the tripwire has fired.

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

## Live verification

`scripts/saxo_field_probe.py` checks these fields against the live Saxo environment without
disturbing the service: it uses the service's current access token without refreshing it, stays
inside the SaxoClient allow-list, deletes its single options-chain subscription and prints no
keys. It reports Chart v3 minute/daily samples (including minute alignment and realised
volatility), futures quote price types, market state, PriceInfo and depth, session states, an
at-the-money futures option's quote and Greeks, and a chain snapshot. Run it as the service
user from a copy of this branch, then:

- set `provider_volatility_scale` to `FRACTION` or `PERCENT` if the option volatilities line up
  with realised volatility on that basis (it stays `UNVERIFIED`, with no spread, otherwise);
- use the option quote's `PriceTypeBid`/`PriceTypeAsk` to settle the Tradable/Indicative policy.

## Live findings — Saxo SIM, 29 September 2026

Two read-only probe runs ([evidence](saxo-field-probe-20260929.json)) established:

- **The SIM feed is delayed by 10 minutes** for every futures and option quote and for charts
  (`DelayedByMinutes: 10`); prices arrive as `OldIndicative` or `Pending`, and no market depth is
  provided. The frozen method needs real-time completed bars and quotes, so every clock would be
  skipped. This is now named explicitly: the Quote gate runs the same check a decision uses
  (`QUOTE_DELAYED_OR_DELAY_UNKNOWN`), history reports `SAXO_CHART_DATA_DELAYED`, and the setup
  checklist has a **Real-time market data** step. Paper trading needs real-time exchange data
  entitlements, or LIVE data feeding INTERNAL_PAPER.
- **Chart v3 works:** fields Open/High/Low/Close/Volume/Interest/MarketTradingState/Time, minute
  aligned, with DataVersion; ChartInfo carries the delay.
- **Session states** observed: AutomatedTrading, Closed, PreTrading (no "Open"), confirming the
  session-gate fix. Futures `Quote.MarketState` separately reports `Open`.
- **Volatility scale:** price-subscription `Greeks.MidVol` and chain per-strike
  `Greeks.MidVolatility` are annual fractions (CL at the money 0.56–0.59, NG 0.523; one NG chain
  strike 0.651 between bid/ask volatilities 0.645/0.657); the chain summary `ImpliedVolatility`
  is a percentage (50.5, 55.3). `MidVolatilityPct` is not populated for futures options. The
  ticket now shows implied volatility against the frozen model's sigma, and each selected
  option's recorded context includes `iv_minus_model_sigma`. `provider_volatility_scale` now
  defaults to FRACTION.
- **Price quality:** a live chain side reported `Indicative`; delayed quotes are `OldIndicative`.
  The Tradable-only paper-fill rule would have rejected normal real-time prices (resolved below).
- **NQ is not offered** on this account: no ContractFutures match "NQ", "Nasdaq", "Nasdaq 100" or
  "E-mini Nasdaq".
- **0DTE expiries exist** for CL (last trade 18:30 UTC = 14:30 New York) and GC (17:30 UTC =
  13:30 New York) on the probe date. With the existing rule that the exit must precede the cutoff
  by two minutes, only CL clocks to 13:00 and GC clocks to 12:00 New York could complete a
  60-minute hold; the frozen research assumed a 17:00 New York expiry.

## Optional configuration

```yaml
event_calendar_file: /etc/stocker/v1/event-calendar.yaml   # see configs/event-calendar.example.yaml
alerts:
  url_file: /etc/stocker/v1/alert-url.json   # mode 0600, owned by the service user
  stream_down_seconds: 120
  login_warning_minutes: 15   # healthy Saxo sessions keep 40-60 minutes
option_chain_strikes: 11
provider_volatility_scale: FRACTION     # verified live; UNVERIFIED hides the IV spread
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
2. **Resolved (owner-approved) — price quality.** Internal paper fills required
   `PriceTypeBid`/`Ask` equal to `Tradable`, which Saxo marks obsolete; its pricing guide calls
   `Indicative` "in most cases … as relevant as a Tradable price" (FX options excepted), and a
   live chain quote used it. Execution now accepts `Indicative` or `Tradable` on both sides, and
   still requires real-time delivery, a quote under five seconds old, no error code, an uncrossed
   market, a tradable instrument in an `AutomatedTrading` session and fresh displayed size of at
   least one contract; `OldIndicative`, `Pending`, `NoMarket`, `NoAccess` and unknown values are
   rejected. Internal fills remain ask plus one tick and bid minus one tick. The real-time CME
   subscription takes effect on 1 October 2026; rerun the probe then to confirm price quality,
   depth and NQ availability.

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

## Cockpit (20 October 2026)

The Today page opens with four tiles (day P&L closed and open, open positions, the next event with a countdown,
feeds and login), then a playbook panel (the research leads' action per market with the first-candle signal and
its trend context under it, and the 15:50 dip readings) beside the day's events as a timeline; the trade tables
follow. The Record page opens with net P&L, closed trades, win rate and the first recorded day, then the
by-method table, the Books table (amendment 4) and the trades. Every id the browser test reads is unchanged.
Phone width stacks the tiles two across and the playbook rows vertically; checked at a true 390 px viewport
(headless Chrome's window will not go below ~485 px, so the check renders the page inside a 390 px iframe).

## Chart (20 October 2026)

The Today page shows two chart panels side by side (stacked on a phone): QQQ and the S&P (the US 500 CFD),
5-minute candles from `GET /api/charts`. The runtime keeps each one's latest 200 candles in memory
(`Runtime.watch_charts`, `market.chart_rows`; CFD prices at the bid/ask midpoint), refreshed every minute from
three hours before the open to ten minutes after the close and every fifteen minutes otherwise — two Saxo chart
reads a minute at most, display only; a failed read keeps the last candles and notes the reason. The page draws
inline SVG with CSS classes (the CSP allows no inline styles and no chart library). QQQ: the regular session's
candles, EMA12 of the candles served, the session VWAP and EMA10 of regular-session closes (method 4's lines),
the day's option strike, the first-candle side at its 09:35 candle and the crossing entries and exits. S&P:
candles from 04:00 New York, EMA12, and the first-candle trade's entry, stop and exit levels. Levels and markers
come from `/api/today`; the 09:30 open is a dotted line. The browser test's fixture carries synthetic candles.

## Audit and redesign (21 October 2026)

Display only: no trading rule, Saxo read or record write changed. The dashboard was audited against the
paper record's protocol and redesigned so the three pages answer, in order, "is the app healthy", "what
happened and what is next today", and "how is each method doing against its verdict point".

- **Findings fixed.** Open VWAP-crossing and dip trades had no open mark (`today()` marked the first-candle
  trades only, and from whatever quote the stop watcher last held); every open CFD position is now marked
  from that quote only while it is under ten seconds old (`MARK_FRESH_SECONDS`), else shown open without a
  figure. The status chip read "Logged in · 52 min left", which is the refresh token's renewal, not the
  login's end; it now says "Saxo connected · renews in 52 min". The next-event countdown ran on the browser's
  clock; it now runs on the server's (`server_time` offset). Both charts were rebuilt on every 5-second
  refresh; they redraw only when the trades that place their levels change, or each minute with new candles.
  Chart level labels collided ("exit" on "stop") and sat on candles; they are staggered and backed. The
  heading order skipped from h1 to h3; sections are h2. A stale refresh only changed a caption; the status
  bar now turns amber (`data-stale`).
- **Today.** Four tiles, the two charts with a hover/touch crosshair (`#chart-readout`), the price and change
  since the open in each chart head, the playbook, the events timeline (the next event highlighted with its
  countdown, "n of m done"), then one card per method with a one-line summary (`#cfd-sum` etc.), the table,
  a one-line empty state and the rule under "How it works". Phones get a bottom tab bar, a scrolling chip row
  and one line per event.
- **Record.** The tiles, then the equity curve (closed results booked on their closing day, from the new
  `by_day` field of `GET /api/record`; best and worst day and the maximum drawdown in its caption) beside the
  verdict-progress meters: the protocol's checkpoints (A at 40 closed CFD trades a market, B at 40 and C at
  60 closed first-candle options, D at 250 Nasdaq days, E and the dip rules at 40, the crossings and the
  stock dips at 100, the fear rule at 10, the pullback book at 40), each as "n of N" from the summary counts.
  The by-method table carries a signed bar per total; book cells show the average and closed count beneath.
- **System.** The login panel beside a problems panel (one line per area and code); the eight facts and the
  diagnostics as before.
- **Tests.** The fixture carries context on the first-candle signals, the playbook, stock orders and trades,
  the books and `by_day`; the browser test reads the new elements. The Python suite checks `by_day` after a
  full day. Checked locally in headless Chrome at 1440 px and in a 390 px frame (no console errors, no
  horizontal overflow) with the browser test's assertions ported to the rendered DOM; the Playwright suite
  itself runs in CI.

## Stock dips prices (21 October 2026)

Display only: no rule, fill or exit changed; the evening scan still settles every order and position on the
daily bars. In the session (09:30-16:00 New York) the runtime reads Saxo's price once a minute for the stocks
with an order resting today or an open position (`Runtime.watch_stocks`, `market.stock_prices`: one
`/trade/v1/infoprices/list` call per 50 stocks with `Quote,PriceInfo,PriceInfoDetails`, none when nothing
rests or is held; after a restart Saxo's US stock list is read once for the Uics, which the scan keeps
otherwise). `GET /api/today` adds to those rows the price, the session's open and low so far, the read time
and any delay; on a position also the result if sold at that price and ConnorsRSI with that price as today's
close, which is the exit rule's value if the stock closes there. It also carries `stock_scan`, the latest
scan's counts. The Stock dips card shows a scan line (charts read, universe, setups, fills, exits, the next
scan) and a "Price now" column: on an order, the low so far and the fall still needed to reach the limit, or
the fill to expect once the low is under it; on a position, the open result and whether ConnorsRSI would
sell it at the close. A failed read is the `stock prices` problem; the last figures stay, with their time.
Two-line table cells are now one box, so a phone stacks them beside their label.
