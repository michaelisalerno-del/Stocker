# DELTA-TARGET-PROTOCOL — amendment 1: a delta chosen by the session (2026-10-05, before the data it is judged on)

Amends `DELTA-TARGET-PROTOCOL.md` (sha256 dd7947d76daf4587cafaf7b807db1cddf4c460327dcccfbe6893fa5f4efe88e3) by adding one
comparison, DC; D30, D50 and every other line of that protocol are unchanged. The user asked whether the delta could be
chosen when the trade is made rather than fixed, and chose this form: near the money overnight, the frozen 0.10 delta in
the US session, never a strike with a wide spread. It is a paper comparison on the hourly ledger; the app keeps its 0.10
delta, nothing changes the runtime and no order is sent. The hash of this file is in `DELTA-TARGET-AMENDMENT-1.sha256`.

## Why this form, and what had been seen
- Overnight (the 4–5 October chain paths, 44 clocks): the strike nearest 0.50 delta lost least at the bell and also had
  the highest ceiling with perfect timing (+28% against +16% for the strike nearest 0.10): nearer the money was better at
  every level of timing skill.
- US session: no strike-against-strike outcome had been examined. What had been seen is the app's own option's timing
  ceiling on the 1–2 October daytime clocks (perfect entry and exit: median +116%, against +23% overnight), which is why
  the session keeps the 0.10 delta there. At the bell those same clocks lost more (−23%). The ledger's conditional table
  held 6 daytime clocks when this was written; their outcomes were not looked at.
- Spread: across every look, the spread paid was the one thing that separated outcomes; 15% of the mid is about the
  median spread of the 0.10-delta options on the look days.

## Definition (no free parameter)
- DC, at each clock: among the chain's out-of-the-money strikes of the app's right and expiry (the clock's recorded chain,
  as `strikes.csv` defines them) whose spread at the clock is at most 15% of the mid, the strike whose Saxo delta is
  nearest the session's target — 0.10 on the 09:00–16:00 New York clocks, 0.50 on all others — within 0.12 of the target.
  When none qualifies, DC does not trade that clock and its return is 0.
- Bought at its ask at the clock, sold at its bid at the hour's end (`dc_ret_ask_to_bid`); the paired difference
  `dc_delta_vs_app` = DC's return − the app's own option's (`opt_ret_ask_to_bid`), per pound of premium. A chosen strike
  without an hour-end quote is excluded and counted. Clocks without a chain for the app's right have no row (E-mini S&P has
  no clock profile and so no strike rows).

## Data
`cond.csv`, written by `delta_cond.py` in the hourly ledger (sha256
412ae777b8f5f5729e5d59d2ee3d34bf1dbcc2a5a8e448a94adc45aa1ce3fc98 at freezing) after `edges.py` (sha256 unchanged from the
protocol). Judged clocks, halves and the look exactly as in the protocol: 6 October–27 November, halves 6–30 October and
2–27 November, 1–5 October not judged.

## Verdict
DC is one further comparison, judged once after 2026-11-27, and PASSES only if: (a) the mean `dc_delta_vs_app` is positive
with t > 2.5 across days (the same bar as D30 and D50, so the three comparisons share one standard); (b) at least 40 clocks
on at least 10 distinct days; (c) the mean is positive in both halves. Reported regardless: the same breakdowns as the
protocol, and also by session (US and other separately, with their own t), the no-trade clocks and what the app's option
did on them (the value of the spread cap alone), and DC against D50 on the clocks both have (whether choosing by session
beats always buying near the money).

As in the protocol, a pass amends the running method only by a dated rulebook amendment and a new rule version after
2026-11-27.
