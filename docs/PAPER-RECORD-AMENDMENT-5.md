# Paper record — amendment 5: a sixth method, the Connors pullback on US stocks

Written on 21 October 2026 (about 12:30 UTC) on the user's instruction ("We need this in my app to
paper trade alongside my other methods", after the research below). The hash of this file is in
`PAPER-RECORD-AMENDMENT-5.sha256`. The protocol (`PAPER-RECORD-PROTOCOL.md`, sha256 c218693f…) and
amendments 1-4 are unchanged: methods 1-5, their code paths and their verdicts stay as frozen. This
adds method 6 beside them; it is recorded from the first evening the release runs.

## Where the rule comes from

- **The published rule.** Connors Research, "An Introduction to ConnorsRSI" (2012), Section 3, the
  worked example: a stock above $5 trading 250,000 shares a day with ADX(10) above 30, whose low fell
  at least 4% below the prior close and which closed in the bottom quarter of its range with
  ConnorsRSI(3,2,100) below 10; bought the next session on a limit 8% below that close; sold at the
  first close with ConnorsRSI above 70. No stop.
- **The test.** `~/Codex/2026-10-21-stock-rare-setups/` PREREG131 (sha256 e59cea49…, amendment 1
  9d36ea03…; RESULT131.md): every US common stock EODHD lists, active and delisted, 2000-2026. The
  book's own window replicates (451 trades, +15.2% a trade, 78% winners against its 472, +15.0%,
  79%). After publication the book's best versions faded. The worked example on the 500 most-traded
  stocks made +3.66% a trade (t 2.16, 70% winners) in 2013-19 and +3.42% (t 1.53, 68%) in 2020-26,
  about +3% beyond SPY, positive in 6 of 7 recent years; the descriptive split by liquidity gave
  ranks 501-1,500 +3.4% (66%) in 2020-26 and everything less liquid nothing. 2023's trades include
  Signature Bank and First Republic bought on their last day (-100%, -80%). It failed PREREG131's
  bar only on the 70% win rate the user asked for. As options (PREREG134, real quotes 2023-26) it
  made nothing: the stock is the vehicle.
- **Why a paper test.** 12-30 trades a year in the top 500 and about 80 more in ranks 501-1,500 are
  too few and too tail-heavy to trust a backtest alone; a forward record from here settles whether
  it still works.

## Method 6

- **Data.** EODHD's official daily bars (open, high, low, close, adjusted close, volume) for every
  US stock, read once a session after the close from the bulk end-of-day file, with each stock's
  history fetched once (and again after a split or dividend changes its adjustment). This is the one
  data source outside Saxo in the record: Saxo cannot supply 1,500 stocks' daily bars a night. It
  never stands in for a Saxo price in methods 1-5. The token lives in a mode-0600 file named by
  `eodhd_token_file`, never in the repository or the dashboard.
- **Time.** Event `STOCK_DIPS`, due three hours after the close (19:00 New York on a regular day),
  retried every fifteen minutes until the bars are published, up to fourteen hours (09:00 the next
  morning); later the night is recorded `FAILED` with its reason and nothing is scanned afterwards.
  The scan also runs on early-close days (the rule knows nothing of them).
- **Universe (rules 1-2 and the research's cohorts).** At each close, among stocks with 260
  sessions of bars: price $5-$5,000, 21-session average volume at least 250,000 shares, 21-session
  dollar volume at most ten times SPY's, daily-return correlation with SPY of at least 0.05 over the
  trailing 250 sessions; ranked by 21-session dollar volume; ranks 1-500 are tier A, 501-1,500 tier
  B, the rest ignored.
- **Setup (rules 3-6, `methods.stock_setup`).** On split-adjusted prices: ADX(10) above 30, the
  low at least 4% below the prior close, closing range at most 25%, ConnorsRSI(3,2,100) below 10.
  Each setup is an order: a buy limit 8% below the close, good for the next session only, one per
  stock, none while the stock is held or has an order resting.
- **Fill (`methods.stock_fill`).** From the next session's official bar: filled if the low traded
  below the limit, at the open when the stock opened below the limit, else at the limit. Not
  filled otherwise; cancelled with `NO_BAR` if the stock has no bar that day.
- **Exit.** At the first close, the entry day's included, with ConnorsRSI above 70, at that close
  (`CRSI_ABOVE_70`). A stock with no bar for ten sessions (delisted or halted) closes at its last
  close (`NO_BAR_10_SESSIONS`), as the research exited delisted stocks.
- **Size and cost.** $2,000 a trade (the rule has no stop; that is the money at risk). The result
  is $2,000 x (exit / entry - 1) less the research's round-trip cost, 0.20% in tier A and 0.40% in
  tier B. No financing or conversion (reports can deduct them from the calendar nights held).
- **Recorded.** `stock_orders` (signal day, close, limit, sell-off, closing range, ADX, ConnorsRSI,
  rank, tier, status, fill day and price), `stock_trades` (entry and exit days and prices, reason,
  ConnorsRSI at exit, result), `stock_bars` (the bars used), and a nightly signal `STOCK` / `US`
  with counts (bars published, stocks kept, histories fetched or failed, universe, orders, fills,
  exits). Today page: "Stock dips" (tonight's orders, resting orders, positions). Record page:
  "US stocks · Connors pullback" per trade and in the summary.

## Differences from the research, said beforehand

- Same vendor, same bars, same formulas: the app's ADX and ConnorsRSI matched the research's on 696
  real stock-days to 1e-10, and the setup decision on all 696 (7 setups each). The research rebuilt
  delisted stocks' prices from split lists; the live feed has no delisted stocks to rebuild.
- The research's limit order is a research fill: a low below the limit. Real resting orders at
  8% below the close fill in the same way, but a stock halted or re-opened with a gap could fill
  worse. Bars published late (after 09:00) leave a night unscanned.
- Costs are the research's constants, not Saxo's stock commissions, which depend on the venue and
  the user's plan. Report them when the vehicle is chosen.

## Reporting

Recorded and reported only. First look at the 100th filled trade: mean result per trade, win rate,
worst trade and the share from tier A, against PREREG131's +3.4% and 68% winners; no verdict
changes the app.

## Code

Branch claude/stock-dips: `methods.py` sha256 7d3c8bc348fb21a983731334b922fda0e9b9208de028775ce4223e6728c68c7e, `runtime.py`
472f5499e95d73dcb1e792cf7ae525b7688ae96796c4b0332f545c6a36d6bd61, `store.py` b2ea7719bbb9bc09ee60535bf2e3432dbd8273cfa0523d7953592509fbf2cf89, `eodhd.py` (new)
383d5e7b3155e5abe6e1a1f629501ecb8c93c3e2df794c5c34f948f67c56cebc, `config.py` 6aadb2f496a75696c18a35270d4dddb5503282ae3fc0af6725028440f0399076. Methods 1-5 are unchanged in
behaviour: their tests pass unchanged; three tests are added for method 6 (the indicators, setup,
fill and exit on a synthetic series; a scan, fill and exit on three nights; a scan that waits past
midnight). Written before any stock dip could be recorded.
