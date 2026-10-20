# Paper record — amendment 4: context, books, the playbook and the fear rule

Written on 20 October 2026 (about 15:00 UTC) on the user's instruction ("implementing all the other
little clues, doing something with the 50 d thing"), after the research below. The hash of this file
is in `PAPER-RECORD-AMENDMENT-4.sha256`. The protocol (`PAPER-RECORD-PROTOCOL.md`, sha256 c218693f…)
and amendments 1-3 are unchanged: methods 1-4, the Connors and Turnaround Tuesday rules, their code
paths and their verdicts stay as frozen. Nothing this amendment adds changes a trade those methods
make; it saves readings beside them, sums subsets of them, shows what the readings imply, and adds one
more dip rule next to the two of amendment 3.

## Where it comes from

`~/Codex/2026-10-20-fc-pullback/` (synth1/synth2: exploratory on data the research had already used;
PREREG114, sha256 f9521899…, sealed before its run). The whole research ledger (113 registered tests
and about 2.4 million grid strategies) holds two direction effects on QQQ and the S&P: the opening
continuation the first candle trades, and the short-term reversal the dip rules trade. The leads
below are where those two meet, read off the first-candle trade split by things known at 09:35:

- **The pullback cell.** A long first-candle signal on a day the index closed below its own 50-day
  average the day before: QQQ +18.1 bp (t 3.2) 2013-19 / +19.4 (t 2.2) 2020-26, SPY +15.7 (t 2.9) /
  +24.1 (t 3.1), 12-15% of days, 13 of 14 QQQ years positive; the other three side x trend cells
  +1..+9 bp. The S&P's whole first-candle edge sits in this cell. On the 16 other ETFs, where the
  first-candle rule has no edge (+0.7 bp), the cell is +5.4 bp, 14 of 16 positive. Sealed replication
  on Europe 2011-26 (PREREG114): +5.4 bp, 3 of 4 markets, both halves positive, t 1.57 — **a FAIL by
  its bar (t 2), the sign confirmed on every unseen set**. A lead, not a result.
- **Mondays.** The first-candle trade averages about zero on Mondays on QQQ and SPY in both periods
  (-0.6 / +2.1 and -0.3 / -4.8 bp) against +5.4..+10.8 on Tuesday to Friday (t 2.9-3.9). Post hoc.
- **Shorts after an oversold close** (three or more down closes, or RSI2 below 10): QQQ -40.6 bp
  (t -3.1) and SPY -23.4 in 2020-26, mixed before. Post hoc.
- **Agreement** (PREREG28, passed on six fresh ETFs): a signal that disagrees with the other index's
  is worth less (QQQ +9.0 / +9.8 agreeing vs -0.7 / +6.3 disagreeing; SPY +5.2 / +7.1 vs +1.5 / -1.8).
- **After a +1% day:** both sides stronger on QQQ (+17.4 t 3.1 / +21.3 t 4.0). Post hoc.
- **Puts.** On real quotes (EODHD, July 2024 - October 2026) QQQ's 0.40-delta put is priced at 1.07x
  the matching call's implied volatility on 99% of days (SPY 1.073), on the side with half the edge:
  the research's one-vol model overstates every put trade by about 7% of premium.
- **The fear rule.** PREREG69's confirmed dip rule (three down closes with the VIX above 30: Europe
  +129 bp an episode net, 5 of 5 markets, t 2.65) was left out of amendment 3 because Saxo gives this
  account no VIX. On QQQ 2009-26 the same rule on VXN's level (three down closes, VXN at 30 or more)
  made +156 bp a trade over five sessions above the unconditional (t 2.7), 73% up, 2.9 trades a year,
  worst -12.5% (March 2020); 2009-19 +245 (t 3.6), 2020-26 +96 (t 1.1). The percentile versions were
  weaker. Saxo quotes an implied volatility (MidVol) on every QQQ option, so a 30-day at-the-money
  reading is one board read away; it sits a point or two under VXN, which averages the whole strip.

## Context (methods 1 and 2)

- At 09:35, with each CFD's first-candle signal, the app reads the reference ETF's daily chart (QQQ,
  Uic 4328771, for the Nasdaq CFD; SPY, Uic 36590, for the S&P CFD; Saxo's official closes, not
  dividend-adjusted, the newest at most four days old) and saves with the signal: the prior close, its
  50- and 100-day averages, whether it sits below each, the prior day's change, the run of down closes
  ending at it, and Wilder RSI(2) at it (`methods.trend_context`).
- Reported only. A read that fails (or finds too few closes) saves its reason (`context_reason`) and
  the trade goes ahead exactly as before; the context is not an input to any trade.
- Signals recorded before this amendment (16, 19 and 20 October) have no context.

## Books (Record page)

Each book is a sum over the same recorded trades (`store.books`, rules in `methods.in_book`), by
instrument: the Nasdaq CFD, the S&P CFD and the QQQ option of methods 1 and 2; the VWAP crossings
of method 4. A day whose signal has no context belongs only to the books that need none.

- **PULLBACK:** long signal, prior close below the 50-day (any weekday).
- **WEEKDAYS:** Tuesday to Friday.
- **AGREE:** the Nasdaq and S&P signals point the same way (needs both signals).
- **NO OVERSOLD SHORTS:** every long; a short only when the run of down closes is under three and
  RSI2 at the prior close is 10 or more.
- **AFTER A +1% DAY:** the prior session closed at least 1% up.
- **PLAYBOOK:** what the leads together say — in the pullback cell the QQQ option (a call) and the
  S&P CFD; on other long days the Nasdaq CFD; on short days the Nasdaq CFD unless it is a Monday or
  the close was oversold; the S&P never outside the cell.
- **CROSS PULLBACK** and **CROSS BEFORE 14:00:** method 4's crossings on pullback days, and those
  entered at a candle closing before 14:00 New York (the 14:00-15:00 crossings continued less in
  the research, post hoc).

Each cell shows trades, the total and the average per closed trade. No book changes the app or any
verdict; the first look at the pullback book is at its 40th Nasdaq CFD trade with an exit price.

## Playbook (Today page)

Once the 09:35 signals are saved, the Today page states, per market, the action the leads imply
(`methods.playbook`: the QQQ call in the pullback cell; the Nasdaq CFD long otherwise; the Nasdaq CFD
short, never a put; skip the short on a Monday or after an oversold close; the S&P CFD in the cell
only) with the leads that apply. It is labelled as research leads, not verdicts; the record records
every day regardless.

## Method 5: the fear rule

- At 15:50, after QQQ's live price is read, the app reads the QQQ option expiry nearest 30 calendar
  days out: the strike nearest the price, call and put, and saves the mean of Saxo's MidVol for the two
  (`atm_vol_30d`, with the expiry and the quotes' delay) in the day's dip signal. A read that fails
  saves its reason; Connors and Turnaround Tuesday go ahead without it, and the fear rule cannot fire.
- Entry (`fear`): the run of down closes counting today's 15:50 price as today's close is three or
  more, and `atm_vol_30d` is 0.30 or more. A paper long US Tech 100 CFD at the first live ask within
  60 seconds of 15:50, sized as the other dip trades ($200 over 100 bp), no stop, no commission. One
  fear trade at a time; it may open beside a Connors or Turnaround Tuesday trade, and the one-at-a-time
  book takes it only when nothing is in the book (order: Connors, Turnaround Tuesday, fear).
- Exit (`FIFTH_SESSION`): at the fifth session's 15:50 after the entry day, at the live US Tech 100
  bid, `_LATE` when read more than 60 seconds after 15:50, never at or after the close. If that
  session passes without the exit (the app was down), the trade is recorded `MISSED_EXIT` with no
  price, as the other dip trades are.
- Results exclude financing and conversion, as amendment 3 states; reports deduct financing from
  the calendar nights held.

## Differences from the research, said beforehand

- The research's trend flags used dividend-adjusted ETF closes; this uses Saxo's unadjusted closes.
  The sign of close minus average differs only on knife-edge days.
- The fear rule was tested on VXN's close (the Nasdaq-100's 30-day index); this reads one strike pair
  on one expiry of QQQ options at 15:50, which runs a little under VXN. The threshold is kept at 30%
  so the record can compare the reading with VXN; the rule's first look is at its 10th trade (about
  three years).
- Days before this amendment carry no context and belong to no context book.

## Reporting

Recorded and reported only: the Record page's Books table and the "Nasdaq CFD · fear rule" line, the
dip book including fear trades, the Today page's playbook. No verdict changes the app.

## Code

Commit a5855a8 (branch claude/amendment-4-context-books): `methods.py` sha256
6f2f1ff32281c0faaff4d79581cbe6e2dbba0b4c496707b3214fa0900c68e532, `runtime.py`
bfa8c1132089158217d36b3e03d5916083734f3122afe1c5d3a3d3bc6763a202, `store.py`
2354238dfecfe3694c0ff0aec167da8b88ed74285454124c2704ac2b11363941, `market.py`
97952332cec41fb090c26fda7d9a57e7812c984f7c3770e0285de9ac9e831c0e (unchanged). Methods 1-4 and the two
earlier dip rules are unchanged in behaviour: their 60 tests pass unchanged; four tests are added.

Written before any trade with context or any fear trade could be recorded: the deployed release
(1f792f3) saves no context and has no fear rule; the first possible context is the first 09:35 after
this is deployed, and the first possible fear entry the first 15:50.

Dashboard text order fixed in commit c2b74a9 (dashboard.js and the browser test only; the hashes above are unchanged).

## Addendum (20 October 2026, 09:15 New York, before the first signal with context)

Report-only additions from the sizing study (`~/Codex/2026-10-20-fc-pullback/sizing/SIZING.md`, `m4/M4.md`):
- `cfd_trades.worst`: the worst adverse price seen while a CFD trade was open (the bid for a long, the ask
  for a short, read at the stop watch's 2-second poll). Recorded, never used by a rule.
- Books `SIZED V2` (the playbook as units: pullback cell 2, other longs 1, shorts 1 unless Monday or
  oversold, the option and the S&P in the cell only) and `SIZED V3` (score: +1 cell, +1 after a +1% day,
  +1 the other index agrees, −1 Monday, −1 oversold short; 1 + score kept within 0..3): the same recorded
  trades multiplied by their units. V3 raised the Sharpe ratio in all four market-periods of the research
  (QQQ 2013-19 1.14 → 1.56, 2020-26 1.26 → 1.47) at equal average size; V2 only in 2020-26.
- Book `STOP 75 BP`: each Nasdaq CFD trade re-run with a 75 bp stop at the same $200 risk — stopped for
  −$200 when `worst` crossed it, else the recorded result scaled by 100/75. The research found 75 bp
  about $900 a year better than 100 bp (t 2.5 / 1.3). The live 100 bp stop is unchanged.
