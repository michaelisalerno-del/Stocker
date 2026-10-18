# Paper record — amendment 3: a fifth method, the dip rules on the Nasdaq CFD

Written on 18 October 2026 at about 22:00 UTC on the user's instruction ("Add them to the app but as
paper trading", after the research below). The hash of this file is in
`PAPER-RECORD-AMENDMENT-3.sha256`. The protocol (`PAPER-RECORD-PROTOCOL.md`, sha256 c218693f…),
amendment 1 (sample from Friday 16 October) and amendment 2 (method 4) are unchanged: methods 1-4,
their code paths and their verdicts stay as frozen. This adds method 5 alongside them; it is recorded
from the day it is deployed.

## Where the rules come from

- **Connors held as a position.** Method 3's rule (Connors & Alvarez 2008) with its published exit:
  out at the first close with RSI2 above 65. On data after publication it held for ES/NQ (t 3.6/3.8)
  and pointed the right way in all five European indices (PREREG69, t 1.8).
- **Turnaround Tuesday.** `~/Codex/2026-10-18-ssrn/` PREREG68 (sha256 f6024c90…, registered before its
  results): a Monday close at least 1% below the last close -> long to Tuesday's close. SPY+QQQ
  2020-26 +61.5 bp a trade (t 2.03), 2013-19 +41, 15 of 16 other ETFs positive, positive in Europe and
  in SPY/QQQ 2003-12; bought at 15:55 instead of the close, QQQ +66 bp (t 3.7).
- **Why next to the first candle.** `~/Codex/2026-10-18-book-mix/` PREREG72 (sha256 09cedc68…,
  descriptive): on QQQ 2013-26 the dip rules' daily results are uncorrelated with the first-candle
  trade (-0.03 to +0.03); together the Sharpe ratio rose from 1.28 / 1.31 to 1.45 / 1.55 (2013-19 /
  2020-26) and the worst month shrank, but the worst week doubled (March 2020).
- **As recorded here** (`~/Codex/2026-10-18-book-mix/app_dip_check.py`, `app_dip_check.txt`): this
  method's calculation replayed on QQQ's unadjusted daily closes 2013-26, the CFD bought and sold at
  the close, net of a 0.7 bp spread and long financing of 1.97 bp a night: Connors 8.9 trades a year,
  5.5 nights, +44 bp a trade (t 2.6); Turnaround Tuesday 6.5 a year, +58 bp (t 3.2); the
  one-at-a-time book 13.1 a year, +50 bp (t 3.6), about +$1,300 a year at $20,000 a trade.
- **Left out: 3 down closes with VIX above 30** (PREREG69's confirmed rule). This account has no VIX
  price from Saxo (read-only check, 18 October: no index, future or CFD; VIX ETNs 15 minutes late),
  and a stand-in would be a different rule.

## Method 5

- Time: each session's close minus 10 minutes (15:50 New York on a regular day). Exits run first
  (event `DIP_EXIT`), then entries (`DIP_ENTRY`). Entries only on regular 09:30-16:00 days; exits
  also on early-close days.
- Readings (`methods.dip_rules`): Saxo's QQQ daily chart (official closes, not dividend-adjusted; the
  newest at most four days old) with QQQ's midpoint at 15:50 as today's close. That price must be live
  (DelayedByMinutes 0), because the CFD trades at its live price at the same moment; otherwise
  `QQQ_PRICE_DELAYED`. Connors in: method 3's signal. Connors out: RSI2(today) above 65. Turnaround
  Tuesday: today is a Monday and the price is at least 1% below the last daily close. Saved once a day
  as signal `DIP` / `QQQ` with every reading.
- Exits: a Turnaround Tuesday trade at the first session after its entry (`NEXT_SESSION`); a Connors
  trade at the first 15:50 after its entry day with RSI2 above 65 (`RSI2_ABOVE_65`); at the live US
  Tech 100 bid, `_LATE` when read more than 60 seconds after 15:50, and never at or after the session
  close. The Turnaround Tuesday exit needs no QQQ price. A trade with a session since its entry whose
  exit the app did not complete (the app was not running, or the Connors check could not run) is
  recorded `MISSED_EXIT` with no price and no result.
- Entries: each rule with an entry reading and no open trade of its own buys a paper long US Tech 100
  CFD (Uic 4912) at the first live ask read within 60 seconds of 15:50; otherwise the event is recorded
  failed or missed, never entered later. Size as the first-candle Nasdaq trade: $200 / (entry x 100 bp
  x the price-to-contract factor), rounded down to Saxo's amount decimals. No stop (the researched
  rules have none). No commission.
- Book: a trade opened while no trade flagged for the book is open is flagged `book = 1` (Connors
  before Turnaround Tuesday on the same day). The Record page's "dip book" line totals those: the
  one-at-a-time version the research preferred. Every rule's trades are recorded either way.
- Recorded per trade (`dip_trades`): rule, book flag, QQQ's price at entry and exit, the CFD bid and
  ask at entry and exit, the exit reason and the result in US dollars.

## Results exclude financing and conversion

`pnl_usd` is the CFD's price change only. Saxo charges a long index CFD overnight financing at the
interbank rate plus 3% (on 16 October 4.19% + 3% = 7.19% a year: about 2.0 bp, roughly $3.90, a night
on a $20,000 trade) and converts US-dollar results to pounds at a 0.6% markup. Reports deduct
financing from the calendar nights between the entry and exit days.

## Differences from the research, said beforehand

- The research used dividend-adjusted 16:00 closes; this uses Saxo's unadjusted closes and the 15:50
  price, with the CFD at its bid and ask. In the replay above the calculation matched the research on
  118 of its 121 Connors trades (same entry and exit day; the app found 122) and 87 of 87 Turnaround
  Tuesday trades (the app found 90: it exits at the next session even when Tuesday is a holiday,
  where the research needed a Tuesday).
- Method 3 and method 5 compute the Connors signal separately from the same Saxo reads seconds apart;
  on a knife-edge day they can differ, and each records what it read.
- A missed entry is not replayed. The research would then be in a trade; here the rule may enter on a
  later day if its reading still holds.
- While SaxoTraderGO holds Saxo's real-time slot at 15:50 (QQQ 15 minutes late), the entries and the
  Connors exit check are missed; Turnaround Tuesday exits still happen.

## Reporting

Recorded and reported only (Record page: "Nasdaq CFD · Connors (days)", "Nasdaq CFD · Turnaround
Tuesday", "Nasdaq CFD · dip book, one at a time"); no verdict changes the app. First look at the 40th
dip trade with an exit price (about three years): mean result per trade after financing, and the
CFD spread paid against the replay's 0.7 bp.

## Code

Commit 7e05083 (branch claude/dip-rules-paper): `methods.py` sha256
f51d931027ff3d644037d23f81295e4e3cce62b60a436b68d96942ada829d6cd, `runtime.py`
f3a5ea6e9cf82c9a3a2715463a9a924f945116457939ed87974a91c4542ee265, `market.py`
97952332cec41fb090c26fda7d9a57e7812c984f7c3770e0285de9ac9e831c0e (unchanged), `store.py`
8d163e35816e3d63dcd0f37c4c39b20f35a43f5578083f0ccacef0a496ce6b1f. Methods 1-4 are unchanged in
behaviour: their tests pass unchanged; four tests are added for method 5.

Written before any dip trade could be recorded: the deployed release (ece4e08) has no method 5, and
the first possible entry is Monday 19 October 2026 at 15:50 New York (19:50 UTC).
