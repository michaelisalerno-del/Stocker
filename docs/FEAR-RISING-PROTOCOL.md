# Rising fear and the 0DTE price — frozen protocol (2026-10-13, on the user's "Yes", before the data it is judged on)

A paper test on the hourly ledger; nothing here changes the runtime. The hash of this file is in `FEAR-RISING-PROTOCOL.sha256`;
any later change is a dated amendment made before the results are looked at.

## Where the idea came from
`~/Codex/2026-10-13-vix-level/vix_level.py` (2023-09 → 2026-10, model prices from the VIX1D gauge): on days when the market's
fear gauge closed the previous day more than 1.1× its 20-day average, the move from 10:00 to 16:00 beat the 0DTE price more
than usual (ES 1.14, NQ 1.12 of the usual ratio; falling-fear days 0.92 / 0.91; rising minus the rest +0.17 / +0.15, t 2.5 /
2.2 by month). Not real option quotes, so it is tested here on Saxo's.

## Definitions
- Gauge per market (Cboe daily closes, downloaded each run): ES VIX, NQ VXN, GC GVZ, CL OVX.
- Fear ratio for a trading day = the gauge's last close before that day / the mean of the 20 closes ending there. Rising > 1.10,
  falling < 0.90, otherwise middle. Known before the open.
- Outcome per market and day: the ledger's 10:00 New York clock, its chain's same-day at-the-money straddle (`chain_atm_strike`,
  `straddle_ask`, `straddle_mid`, chain expiry on the clock's New York date), held to expiry and settled at |futures − strike| at
  the futures' close of the expiry minute (as in LEDGER-STRADDLE-EXPIRY-AMENDMENT-20261013). `s_ask` = value / straddle_ask − 1
  (a buyer at the ask), `s_mid` = value / straddle_mid − 1 (the pricing question without the spread). No straddle quote, no
  expiry on that date, or no futures bar at the expiry minute: no outcome.
- Day value (ES/NQ, judged): the mean of ES and NQ outcomes that day (one if only one exists); the state is ES's (VIX) state, or
  NQ's when ES has no outcome.

## Sample and timing
Weekdays from 14 October 2026. Judged once, at the later of 27 November 2026 and the day the 15th rising day with an ES/NQ
outcome occurs; the judged sample is every day from 14 October to that date. Before that `fear.py` prints counts only.

## Verdicts
- PRICING passes if the rising days' mean `s_mid` exceeds the other days' (middle + falling) with Welch t > 2, and the
  difference is positive in both halves of the judged sample (split at its middle day).
- MONEY passes if the rising days' mean `s_ask` is above zero with t > 2 (across rising days).
- Reported, not judged: GC and CL with their own gauges, and all hourly clocks 09–15 New York (straddle ask→bid and mid,
  realised/priced) by state.
A pass is evidence for a later rule; it does not change the app.

## Scripts (hourly ledger folder)
- `fear.py` sha256 3ca1626d6c155628284372b9d17fabc5695d36478bc50a7dfa61fa23d08e871d (reads `ledger.parquet`, uses `build.bar_grid`
  and `build.price_at` unchanged; writes `fear.csv`). Run by hand after `sync.sh`.

## Disclosure
While writing the script Claude ran it on 1–12 October (before the sample). Every day was "middle" (no rising or falling day);
the ES/NQ/GC/CL straddle outcomes of those days were seen. Nothing from 14 October on has been computed.
