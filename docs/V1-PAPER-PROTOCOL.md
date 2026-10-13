# V1 trailing stop on real futures quotes — frozen protocol (2026-10-13, on the user's "Yes", before the data it is judged on)

A paper test on the hourly ledger's recorded futures; nothing here changes the runtime. The hash of this file is in
`V1-PAPER-PROTOCOL.sha256`; any later change is a dated amendment made before the results are looked at.

## Where the idea came from
The exit search (`~/Codex/2026-10-12-exit-search`, 1,100+ rules, TRAIN 2013–2021, TEST 2022 → 2026-10) found one rule that
passed: V1, a stop at the best price since entry minus the remaining expected move E(t) × R, no target. TEST gain over holding
to 16:00: +0.015R a trade. It was tested on 1-minute bars with fills AT the stop level, so its gain disappears if each stop
fills worse by ES 0.0159R, NQ 0.0099R, GC 0.0148R, CL 0.0176R (`checks_out.txt`). This measures that slippage on real quotes.

## Definitions
- Markets ES, NQ, GC, CL: the app's recorded front futures (per-second weighted mid and spread in ticks from the capture
  segments' book flow, as cached by `build.py`).
- R for a session = mean of |log(15:59 bar close / 09:30 bar close)| over the 20 previous full sessions of the ETF stand-in
  (SPY, QQQ, GLD, USO 1-minute from EODHD), the exit search's R.
- E(m), m = 0..359 minutes after 10:00: `v1_etable.json` (written by `make_v1_etable.py` from the exit search's data and code,
  TRAIN 2013–2021).
- Each weekday, each market: enter at 10:00:03 New York both long and short, at the mid ± half the spread.
- V1 (judged): the stop trails the best mid since entry second by second at E(minute) × R behind it; it triggers at the first
  record at or beyond the level; the exit is the mid of the last record at or before trigger + 3 s, less half the spread.
  V1m (reported): the same with the best as of the last completed minute (the 1-minute version that was tested).
- Slippage per stop = the stop level − the exit mid, in R (positive = worse than the level), so both rules pay the same half
  spread as holding.
- HOLD: exit at 16:00:00 New York at the mid less half the spread.
- Holes: a version is scored only if the series has no hole over 120 s from entry to its exit (the recorder's segments end
  early at times); HOLD needs only the 16:00 quote. Excluded trades are counted.

## Sample, timing and verdict
Weekdays 14 October → 27 November 2026, judged once after 17:00 New York on 27 November. Per market: V1 SURVIVES REAL FILLS if
the mean slippage over its V1 stop exits is below that market's break-even above, with at least 20 stop exits; fewer is "no
verdict". Reported, not judged: V1m slippage, V1 − HOLD per trade (t across days), worst 10% of V1 and HOLD. A pass means the
exit-search gain is not eaten by real fills in that market; it does not change the app.

## Scripts (hourly ledger folder)
- `v1_paper.py` sha256 40841a720db3d53e069c7fe3be62791892f7f2bacb15892cbbf3806655b97545 (reads `cache/*.pkl`; fetches the ETF
  1-minute bars into `etf1m/`; writes `v1_paper.csv`). Run by hand after `sync.sh`.
- `v1_etable.json` sha256 c19d1dd4b558ee820b09a630ecd74e842ebeec55b0312a7e4bd11b1e81d76c3d; `make_v1_etable.py` (exit-search
  folder) sha256 c934bd0f26df17fd6808534ad5021c369f008345d12bc740e1d1cfc3b380b7ed.

## Disclosure
While writing the script Claude ran it on 1–12 October (before the sample) to check the mechanics. The minute version's
first draft triggered mostly at minute boundaries, which a live stop would not do; the operating version trails each second
(on those days the two stopped at the same second on every trade checked). Seen on those days: 7–11 stop exits per market,
mean slippage ES 2.1, NQ 3.7, GC 4.4, CL 10.4 ticks (break-evens at that week's R ~1.6 / 6.4 / 2.9 / 2.1). Coverage: 7–11 of
10–14 trades per market scorable (recorder holes). Nothing from 14 October on has been computed.
