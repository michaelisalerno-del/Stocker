# Paper record — amendment 2: a fourth method, VWAP crossings on long first-candle days

Written on 16 October 2026 at 04:10 UTC on the user's instruction ("Build it now and merge and
deploy", after the research below). The hash of this file is in `PAPER-RECORD-AMENDMENT-2.sha256`.
The protocol (`PAPER-RECORD-PROTOCOL.md`, sha256 c218693f…) and amendment 1 (sample from Friday
16 October) are unchanged: methods 1-3, their code paths and their verdicts A-E stay as frozen.
This adds method 4 alongside them; it is recorded from the day it is deployed, 16 October 2026.

## Where the rule comes from

`~/Codex/2026-10-16-fc-cross-filter/` PREREG37 (sha256 5f5873a2…, registered before its results):
on days the morning first-candle signal is long, a long each time EMA(10) of 5-minute closes crosses
up through the session VWAP (closes 10:00-15:00), out at the first close back below VWAP, else
16:00. SPY+QQQ, each market's own first-candle signal, 1 bp a trade: 2013-19 +3.3 bp a trade (t 2.7,
used to choose it), 2020-26 +4.65 (t 2.65; SPY +4.3, QQQ +5.0). Reported there, not judged: eight
other ETFs +2.3 (t 3.1, 8/8 positive), eight more +1.4 (t 2.1, 8/8). Not proven: those ETFs had been
used before for the unfiltered crossings, and no untouched market remains; the research expected
part of the effect to be the first-candle drift itself (it is additive: plain crossing ≈ +2.5 bp,
plus the first-candle drift over the same window). That is why it is recorded, not traded.

## Method 4

- Day: a regular session (09:30-16:00 New York) whose Nasdaq CFD first-candle signal (method 1,
  as saved at 09:35) is long. Short, no-signal and early-close days: nothing.
- Data: Saxo's QQQ 5-minute chart (GET /chart/v3/charts, Etf, Horizon 5, the latest 400 candles),
  high, low, close and volume, regular session. The newest candle is still forming and never used.
  EMA(10): smoothing 2/11 over every final regular-session close read (sessions joined), seeded with
  the oldest. VWAP: typical price (high + low + close) / 3 x volume, summed from today's 09:30
  candle. No 09:30 candle today: nothing.
- Entry: at a candle closing 10:00-15:00 where EMA - VWAP turns from <= 0 to > 0, a paper long US
  Tech 100 CFD (Uic 4912) at the first live ask read within 60 seconds of that close; otherwise the
  crossing is recorded `MISSED_ENTRY` with the reason (app not running, QQQ's chart delayed, or the
  failure that blocked it) and no price, and the replay continues as if it had been taken.
- Exit: at the first later close below VWAP (closes up to 15:55), the live bid read within 60
  seconds (`BACK_BELOW_VWAP`), up to 30 minutes late (`BACK_BELOW_VWAP_LATE`), later `MISSED_EXIT`
  with no price; otherwise at 16:00 at the bid (`CLOSE` / `CLOSE_LATE` / `MISSED_EXIT`, as the
  CFD exit of method 1). One position at a time; the next entry only after the exit candle.
- Size: as the first-candle Nasdaq trade, $200 / (entry x 100 bp x the price-to-contract factor),
  rounded down to Saxo's amount decimals. No stop (the researched rule has none). No commission.
- Recorded per trade (`cross_trades`): the signal candle, QQQ's close, VWAP and EMA there, the CFD
  bid and ask at entry and exit, the exit candle and QQQ's close there, the result in US dollars.

Differences from the research, said beforehand: the signal is the Nasdaq CFD's first candle, not
QQQ's own; VWAP comes from 5-minute candles, not 1-minute ones; QQQ volume is Saxo's. On QQQ
2013-2026 the app's calculation (`methods.vwap_crosses`, fed EODHD 5-minute candles) picks the same
entry and exit candle as the research on 63% of trades (the rest mostly one candle earlier or later,
since crossings are close calls) and its results are the same: long first-candle days 2013-19 +3.2
bp vs +3.0, 2020-26 +5.6 vs +5.0 (`~/Codex/2026-10-16-fc-cross-filter/app_check2.txt`). While
SaxoTraderGO holds Saxo's real-time slot, QQQ's chart is 15 minutes late and every crossing is
recorded missed.

## Reporting

Recorded and reported only (Record page: "Nasdaq CFD · VWAP crossings"); no verdict changes the app.
First look at the 100th crossing with an exit price: mean result per trade and the CFD spread paid
(entry ask - entry bid + exit ask - exit bid, half of it per side) against the research's 1 bp.

## Code

Commit 55fc75c (branch claude/vwap-cross-watch): `methods.py` sha256
b8cfe6ef35ebb2758283a66438e4a2b46cd6b130eee02ee5ba83559a7151bd4d, `runtime.py`
7a7c73901510af9dcd12e472ffe3b621f0c8b27911d4ccbfdf9403b42ae2ce9f, `market.py`
97952332cec41fb090c26fda7d9a57e7812c984f7c3770e0285de9ac9e831c0e, `store.py`
6224433c2f00c059fed79f838a8b539cf0511d38e84044bfbafc68e95a58214d. Methods 1-3 are unchanged in
behaviour (their tests pass unchanged); the protocol's hashes for them now describe commit fa15f23.

Written before any crossing could be recorded: the deployed release (aa0a730) has no method 4, and
Friday 16 October's first candle (09:35 New York, 13:35 UTC) is about nine and a half hours away.
