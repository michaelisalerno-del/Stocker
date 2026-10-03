# "Priced below what follows" — five frozen tests on the hourly ledger (2026-10-03, before any of their data exists)

The user's conclusion after a month of tests: the only buyer's edge left is an option priced below the movement that
follows, by more than the spread. These five conditions are fixed now and judged once on clocks the app records from
5 October; nothing here changes the runtime and no order is sent. The hash of this file is in
`PRICED-BELOW-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the results are
looked at.

## Data
The hourly ledger (`~/Documents/Codex/2026-10-03-hourly-ledger`; README.md defines every column; `build.py` sha256
cfa9323e0a7b2a0ffdc04e2be348563f743a837db4d4ad97a4492e0054503536 at freezing): one row per market (CL, GC, NG, NQ, SI) per
New York clock 09:00–16:00, built from the app's clock signals, capture segments and Saxo minute bars. Judged clocks:
2026-10-05 to 2026-11-27. Halves: 5–30 October and 2–27 November. A row counts only if its condition columns exist at
the clock and its outcome exists; rows whose hour-end quote is older than 120 s are excluded and counted.

## Conditions (columns known at the clock)
- H1 Just after bursts: `burst_mem30` at or above the market's top-fifth level at clocks on the 80 IBKR days (June–
  September 2026): CL 0.641, GC 0.558, NG 0.825, NQ 0.435, SI 0.380.
- H2 Forecast above the price: `forecast_over_priced_var` >= 1.25 (look14's variance at least 25% above the variance the
  app's option prices in; about the spread break-even).
- H3 Dealers short gamma: `gex_net_share` < 0 (open-interest gamma, dealers long calls and short puts by convention).
- H4 The hours the bar research found cheapest: CL at the 13:00 and 14:00 clocks, GC at the 14:00 and 15:00 clocks.
- H5 Priced below yesterday: `priced_level` <= 0.8 x the same market's mean realised level over the previous trading
  day's clocks, where an hour's realised level = `realised_over_priced`^2 x `priced_level` (realised variance over the
  clock profile's normal for that hour; needs at least 4 of yesterday's clocks).

## Outcome and verdict
- Primary: `straddle_ret_ask_to_bid`, the at-the-money straddle from the clock's chain bought at the asks and sold at the
  bids an hour later (direction-free, spreads included); one straddle per qualifying clock; the day's figure is the mean
  over its qualifying clocks. If fewer than 40 qualifying clocks of a condition have a straddle outcome, that
  condition's primary is instead `opt_ret_ask_to_bid` (the app's own option, ask to bid).
- A condition PASSES only if, judged once after 2026-11-27: the mean is positive with t > 2.5 across days (five tests:
  about a 3% chance that any passes by luck), at least 40 qualifying clocks on at least 10 days, and the mean is positive
  in both halves.
- Reported regardless, for every condition: qualifying and other clocks side by side for the primary, `opt_ret_ask_to_bid`,
  `straddle_ret_mid`, and the pricing measure `realised_over_priced`^2 (1 = priced fair; the buyer needs it above the
  spread break-even), with the difference's t; counts by market and half; the five combined (any condition true).
- A pass earns a paper trading rule frozen for a further test, not money. These tests do not replace the ledger's
  monthly search (README), which may not use these five as its own discoveries.
