# NQ intraday momentum — frozen forward test (2026-10-03, before the data it is judged on)

The user chose to fix this now and judge it on new trading days only. It is a futures rule recorded on paper from the
app's own minute bars; nothing below changes the runtime and no order is sent. The hash of this file is in
`INTRADAY-MOMENTUM-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the
results are looked at.

## Where the idea came from
Gao, Han, Li & Zhou, "Market Intraday Momentum" (SSRN 2440866): on the S&P 500 ETF in 1993–2013 the first half hour of
the day, measured from the previous close, predicted the last half hour (out-of-sample R² about 1.4%). On our 80 IBKR
days (June–September 2026; `~/Documents/Codex/2026-10-02-deep-analysis`, section 25) NQ was right 60% of 63 days, +63.5
ticks a bet after costs, t 1.74; the four commodity markets showed nothing. Those days are not used.

## Rule
For each New York trading day d with a full NQ session:
- r1 = price(d, 10:00) − price(previous trading day, 16:00); r13 = price(d, 16:00) − price(d, 15:30); New York time.
- price(day, hh:mm) = the close of the completed one-minute bar that ends at hh:mm, from the app's Saxo bar cache
  (`/var/lib/stocker/v1/SAXO_LIVE/bars`, source SAXO_CHART_COMPLETED_1M) for the pinned NQ contract; both days must be the
  same contract.
- Bet: one NQ future, long if r1 > 0 and short if r1 < 0, bought or sold at 15:30 and closed at 16:00. No bet if r1 = 0.
- Gain in ticks = sign(r1) x r13 / 0.25. Cost in ticks = the median quoted NQ spread in the capture between 15:30 and
  16:00 that day (3 ticks if the capture has none) plus 1.2 ticks of fees ($3 a side at $5 a tick). Net = gain − cost.

## Data
- Trading days from 2026-10-05 to 2027-01-15. Excluded and counted: exchange holidays, early-close sessions, days with a
  contract change between the two prices, and days missing any of the four bars.
- The halves for the consistency check are the first and second half of the included days by date.

## Verdict
PASSES only if, judged once after 2027-01-15: (a) the mean net gain per bet is positive with t > 2 across days; (b) at
least 50 bets; and (c) the hit rate (sign of r1 equal to the sign of r13) is above 50% in both halves. Reported
regardless: hit rate, mean gross and net ticks, by month, R² of r13 on r1, the days with the third-largest |r1| and the
third-busiest first half hours, the result at micro NQ costs (1 tick spread plus $1 a side at $0.50 a tick), and the 63
historical days alongside (not part of the verdict). An interim report on 2026-11-27 gives counts only, no results.
