# Paper record — frozen protocol (written 15-16 October 2026, before the first recorded day)

The user approved the design on 15 October 2026 ("Go"): paper record only, the first-candle rule on
the Nasdaq and S&P CFDs and as a QQQ option, and Connors as a QQQ option. The hash of this file is in
`PAPER-RECORD-PROTOCOL.sha256`. Any later change is a dated amendment made before the results it
could affect are looked at. The app never places an order; this record is evidence for decisions the
user makes, and a verdict here does not change the app.

## Where the rules come from

- **First candle.** A TikTok video (quant_lab.de): if the first 5-minute candle after the New York
  open closes above the 12 EMA go long, below go short. Registered tests in
  `~/Codex/2026-10-13-first-candle-ema/` (FC_PREREG sha 42e35fdd and later steps): QQQ 09:35 -> 16:00
  +5.8 bp/day (t 2.4) 2013-2018 and +5.5 bp/day (t 2.9) 2019-2026; SPY weaker; stops and profit
  targets remove most of it; a 5-minute first candle beats 10-30 minutes. With $200 at risk
  (`~/Codex/2026-10-15-budget-200/cfd.py`), the US Tech 100 CFD with a 100 bp stop made +$9 to +$21 a
  day (2020-26, last two years) and the US 500 CFD with a 50 bp stop +$8 to +$19.
- **The QQQ option version.** Model only (Black-Scholes at VIX1D, September 2023 - October 2026): a
  ~$200 same-day QQQ option of delta 0.35 / 0.40 / 0.50 bought at 09:35 and sold at 13:00 with 1.2%
  cost a side made +$64 / +$55 / +$41 a trade, 56-61% of trades losing. Never checked on real quotes;
  this record is that check. Delta 0.40 was chosen for the user's ~$200 budget, not from quotes.
- **Connors.** The published cumulative RSI(2) rule (Connors & Alvarez 2008), unchanged, on data after
  its publication (2009-2026): NQ stand-in +0.55% a trade (t 3.8). As a next-day QQQ call on signal
  days (`~/Codex/2026-10-11-peak-timing/pullback_1dte.py`): real quotes July 2024 - October 2026,
  +25.0% of premium (t 1.8, 43 trades; ask in, intrinsic value at the next close out); model 2010-2026
  +23.5% (t 2.7).
- **The code computes the research's signals.** On QQQ history the app's `methods.first_candle`
  matched the research computation on 769 of 769 days (31 days refused for too few pre-market
  candles, which cannot happen on the CFDs' near-24-hour charts) and `methods.connors` on 4,470 of
  4,470 days 2009-2026 (324 signal days).

## Code

`packages/stocker_execution/src/stocker_execution/`: `methods.py` sha256 673f02feaea8456d61ea8f891c11f3ba9350e2ec5ace00cc7fc6eb2c14b99039,
`runtime.py` sha256 9fdc345c5fd16e6c841631bedb4f6c9e596db3c8c5d8c7362ee8eb0f4aa089c4, `market.py` sha256 be661d323381cde29b152cb7f1a379557859c101bb0623c022f947afde48f899. Commit fa15f233adfc678706fccca92fddb6a443db5433.

## Definitions

**Calendar.** Saxo's NASDAQ exchange sessions (GET /ref/v1/exchanges/NASDAQ). A recorded day has a
continuous-trading session from 09:30 to 16:00 New York. Holidays (no session) record nothing; on an
early-close day no entries are made (open Connors calls are still sold five minutes before that
close). New York times follow US daylight saving; UK times are display only.

**1. First candle on the CFDs.**
- Instruments: Saxo's US Tech 100 CFD (Uic 4912) and US 500 CFD (Uic 4913); $1 per index point per
  contract; no commission.
- Candles: Saxo's 5-minute chart of each CFD, each candle's close the midpoint of its bid and ask
  closes, overnight candles included. The candle that starts at 09:30 must be final (Saxo has started
  the next one), and at least
  30 of the 36 candles starting 06:30-09:25 present. EMA(12): smoothing 2/13 over the candles read
  (Saxo's latest 200), seeded with the oldest.
- Signal at 09:35: that candle's close above the EMA -> long; below -> short; equal -> none. The
  signal is saved as soon as it is computed, so the QQQ option follows it even if the CFD trade
  cannot open.
- Entry: the first quote read from 09:35:00 that is two-sided, undelayed, Open and Tradable; the ask
  for a long, the bid for a short. Not entered within 60 seconds -> the day is recorded failed or
  missed, never entered later.
- Size: $200 divided by the stop's cost per contract (entry x stop / 10,000 points x Saxo's
  price-to-contract factor, 1 for these CFDs), rounded down to Saxo's amount decimals. Stop: 100 bp
  from entry (Nasdaq), 50 bp (S&P).
- Stop: quotes read every 2 seconds during that day's session only; a long closes at the first bid at
  or below its stop, at that bid; a short at the first ask at or above its stop, at that ask.
- Exit: 16:00 New York at the bid (long) or ask (short). Result in US dollars.

**2. The first candle as a QQQ option.**
- Follows the Nasdaq CFD's signal; no Nasdaq signal -> no option.
- Today's QQQ expiry only; none listed -> no trade.
- Choice: of the 31 strikes nearest QQQ's midpoint (calls and puts, all saved with their quotes,
  Saxo's deltas and commissions), the call (long) or put (short) with a two-sided quote and |delta|
  between 0.25 and 0.55 nearest 0.40; ties go to the tighter spread, then the lower strike.
- One contract, bought at its ask as of 09:35 and sold at its bid as of 13:00 (no bid -> sold at 0).
  Saxo's quoted commission on each side ($2.04 on 15 October).
- Reported only: the same option's bid and ask as of 15:55.

**3. Connors on QQQ.**
- Daily closes: Saxo's QQQ daily chart (official closes, not dividend-adjusted), then QQQ's midpoint
  as of 15:50 as today's close. The newest daily close must be at most four days old (a long
  weekend); otherwise no signal (`DAILY_CLOSES_OUT_OF_DATE`).
- Signal: today's price above the average of the last 200 closes (today's included) and RSI2(today)
  + RSI2(yesterday) below 35 (Wilder RSI(2), smoothing 0.5).
- Option: the first listed expiry after today; the call with a two-sided quote and delta between 0.42
  and 0.58 nearest 0.50 (ties as above); one contract at its ask as of 15:50; sold at its bid as of
  15:55 on its expiry day (QQQ options settle into shares, so the call is sold before the close).
- Differences from the research: it bought at the 16:00 close and took intrinsic value at the next
  close; this record buys at 15:50 and sells at 15:55.

**Delayed QQQ prices.** While the user's SaxoTraderGO holds Saxo's single real-time slot, QQQ prices
reach the app late (DelayedByMinutes, 15 on 15 October). A QQQ step due at T measures the delay d
from QQQ's quote at T, is read at T + d and recorded as of T; every quote it uses must carry the same
delay. If the delay changes before the read, an entry is recorded failed; an exit measures again and
reads at the new time, recording the time its prices apply to.

**Late and missed exits.** An exit whose prices apply more than a minute after its time is labelled
late (`_LATE`) with the time they apply to. More than 30 minutes after its time, or at or after the
session close (an option then has expired or settled), the exit is recorded `MISSED_EXIT` with no
price and no result, and it counts in no total. A CFD trade still open on a later day (its 16:00
exit was missed) is recorded the same way; it is never stopped or closed overnight at a later price.
A restart repeats nothing: events and trades are in the record.

## Sample and verdicts

The sample starts on Monday 19 October 2026. Every trade is reported on the dashboard's Record page;
the verdicts below are each judged once, at the stated point, and before it only counts are read.

**A. CFD costs** — at each market's 40th CFD trade with an exit price (about mid-December 2026).
`cost_bp` = ((entry ask - entry bid) + (exit ask - exit bid)) / 2 / entry midpoint x 10,000 (one full
spread over the round trip). AS ASSUMED if the market's mean `cost_bp` is at most 1.5 x the research's
assumption (Nasdaq 0.62 bp, S&P 1.20 bp), otherwise COSTLIER. Reported: stop slippage
(|fill - stop| / stop x 10,000) of stopped trades.

**B. QQQ option spread** — at the 40th first-candle option trade with an exit price.
Cost a side = (entry ask - entry midpoint) / entry midpoint and (exit midpoint - exit bid) / exit
midpoint. AS ASSUMED if the mean of both is at most 1.8% (1.5 x the model's 1.2%), otherwise COSTLIER.

**C. QQQ option carry** — at the 60th first-candle option trade with an exit price (about
mid-January 2027).
`carry_bp` = ((exit bid - entry ask) / |entry delta| - side x (QQQ at exit - QQQ at entry)) / QQQ at
entry x 10,000: what carrying the direction through the option costs or earns against holding QQQ,
per unit of delta. VIABLE if the mean is above -5 bp and (mean + 10) / se > 2; NOT VIABLE if
(mean + 5) / se < -2; otherwise UNDECIDED (the thresholds frozen in the EMA12 first-candle protocol
on 13 October).

**D. Direction** — not judged on this sample: a drift of 5-10 bp a day against swings of about 100 bp
needs years. Checked once, at the 250th Nasdaq CFD day, for breakage: BROKEN if the mean result per
day is below zero with t < -2, otherwise NOT BROKEN.

**E. Connors** — about 18 signal days a year: recorded and reported only; first look at 40 signals.

## Disclosure

- The research behind these rules searched widely (hundreds of thousands of rule variants across
  exits, conditions, markets and times, each step registered first); the first-candle rule and the
  Connors rule are the two that held, and Nasdaq/QQQ came out of that search as the stronger market.
- Before this file was frozen the code ran read-only against live Saxo prices on 15 October 2026 at
  22:45 UTC. Seen: both CFD first-candle signals short on 15 October; a Connors signal at the
  15 October close (RSI2 3.0 + 3.3, QQQ 728.00 above its 200-day average 674.46); QQQ quotes then live
  (the app's session held the real-time slot). Nothing from 19 October on has been seen.
