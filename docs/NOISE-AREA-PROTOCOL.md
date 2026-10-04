# NQ "noise area" intraday momentum — frozen forward test (2026-10-03, before the data it is judged on)

The user chose to fix this now and judge it on new trading days only. It is a futures rule recorded on paper from the
app's own minute bars; nothing below changes the runtime and no order is sent. The hash of this file is in
`NOISE-AREA-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the results are
looked at.

## Where the idea came from
Zarattini, Aziz & Barbon, "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)" (2024; data to
early 2024), popularised on YouTube (WaveLabs; Quant Radio for ES/NQ). Tested with the paper's rule
(`~/Documents/Codex/2026-10-02-deep-analysis`, section 32): QQQ Oct 2020 – Sep 2026 +4.33 bp a day after 1 bp costs
(t 2.81), +3.01 after publication (May 2024 on, t 1.21); SPY, the paper's instrument, +3.64 before and −1.51 after
(t −0.89); NQ futures on the 80 IBKR days +3.6 bp a day (t 0.4, 31 days). It was frozen because it is the least faded
of the published intraday rules tested, not because it passed. None of those days is used below.

## Rule
For each New York trading day d with a full NQ regular session, in New York time:
- price(d, hh:mm) = the close of the completed one-minute bar that ends at hh:mm, from the app's Saxo bar cache
  (`/var/lib/stocker/v1/SAXO_LIVE/bars`, source SAXO_CHART_COMPLETED_1M) for the pinned NQ contract; open(d) = the open
  of the bar that starts at 09:30; close(d) = price(d, 16:00).
- Checks c = 10:00, 10:30, ..., 15:30 (12). move(d, c) = |price(d, c) / open(d) − 1|. sigma(d, c) = the mean of move at c
  over the 14 most recent earlier trading days on which that price exists (bar cache only; no other source).
- upper(d, c) = max(open(d), close(d−1)) × (1 + sigma(d, c)); lower(d, c) = min(open(d), close(d−1)) × (1 − sigma(d, c)).
- At each check: position = +1 if price(d, c) > upper, −1 if price(d, c) < lower, otherwise 0; held to the next check;
  everything closed at 16:00. One NQ future per unit of position.
- Gain in ticks = sum of position × (next price − price) / 0.25. Cost = 2.1 ticks (half a 3-tick spread plus $3 fees at
  $5 a tick) per unit of change in position, including the close. Net = gain − cost, one figure per day; days with no
  position count as 0.

## Data
- Trading days from 2026-10-05; a day is scored once 14 earlier complete sessions exist in the bar cache (expected from
  about 21 Oct). Excluded and counted: holidays, early closes, days missing any of the 14 needed prices, and days whose
  price(d−1, 16:00) is from a different contract (the roll).

## Verdict (a one-year horizon is needed: at the historical strength, t > 2 takes years of days)
- Abandon on 2027-03-31 if the mean net per scored day is at or below zero.
- Otherwise judge on 2027-09-30: PASSES if the mean net per day is positive with t > 2 across days and at least 150 days
  are scored. If it has not passed but the mean is still positive, one further look on 2028-09-30 on all scored days,
  same bar. No other looks decide anything.
- Reported at each date regardless: days scored and traded, mean gross and net ticks, t, by month, long and short legs,
  the VWAP-stop variant (long only above max(upper, VWAP), short only below min(lower, VWAP)), the same rule on QQQ
  5-minute bars for the same dates (EODHD, 1 bp a round trip), and micro NQ costs (1 tick spread plus $1 a side at
  $0.50 a tick). Quarter-end interim reports give counts and the running mean only.
