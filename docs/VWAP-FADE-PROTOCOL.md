# The morning high-volume VWAP fade on ES and NQ — frozen protocol (2026-10-10, on the user's "Yes", before any of the data it is judged on)

One time-of-day idea survived the 2026-10-09 "if it was elevated, does it return?" round (Claude + Codex, folder
`2026-10-09-claude-codex-entries-exits`, TIMES_RESULT.md): Codex's volume-conditioned VWAP reversion (Campbell, Grossman and
Wang: high-volume moves revert, low-volume moves do not), which Claude replicated on SPY/QQQ 5-minute bars as a morning-only
effect. Its best grid cell is matched by chance in 9.2% of permutations, so it is suggestive, not established. This protocol
fixes one cell and scores it on the app's own 1-minute ES and NQ bars, which are untouched by the development work. It is a
futures-only paper comparison: the app keeps trading its frozen method, nothing changes the runtime and no order is sent. It is
NOT a DAY-leg rule — the fade's direction is toward VWAP (only the above-VWAP half matches the ES/NQ puts) and its trade is
10:35 → 11:35, not 10:00 → 11:00. The hash of this file is in `VWAP-FADE-PROTOCOL.sha256`; any later change is an amendment,
dated and stated as such, made before the results are looked at.

## Data
- The hourly ledger's `raw/bars`: Saxo completed 1-minute bars (open, high, low, close, volume; keyed by the bar's start
  minute) of the ES and NQ contracts the app records. One contract per market-day: the uic with the most 09:30–16:00 New
  York volume. `vwap_fade.py` in the hourly ledger (sha256 26a0ba70c0f477cf4fa8f781484efa09a7c1c7331c161d7ff92b95a18ad46279
  at freezing) reads the bars and writes `vwap_fade.csv`, one row per market per session day, and prints the report below.
  It is run by hand after `sync.sh`; it does not depend on `build.py`.
- Judged sessions: ES and NQ US sessions from 2026-11-02 onward (October's sessions 2–30 Oct are the first baseline and are
  never judged), accruing until the verdict count is reached. Halves: the signal-days sorted by date, first half and second.

## Rule (times New York; a session is complete for a window if its 09:30 bar exists and at most 5 minutes of the window are missing)
- Window: the completed bars 09:30–10:34 (start minutes 570–634); signal time 10:35.
- VWAP = Σ(typical price × volume) / Σ volume over the window, typical price = (high + low + close) / 3.
- CV = Σ volume over the window. RVOL = CV / the mean CV over the previous 20 session days that are complete for the window
  (at least 15 of them, else no signal that day).
- R20 = the mean over the previous 20 session days complete for 09:30–16:00 (at least 15) of (max − min of the closes at the
  5-minute marks 09:30, 09:35, …, 15:55) / the 09:30 bar's open.
- DISP = (close of the last bar at or before 10:34, at most 5 minutes old − VWAP) / VWAP / R20.
- **PRIMARY signal**: |DISP| ≥ 0.25 and RVOL ≥ 1.25. **CONTROL**: |DISP| ≥ 0.25 and RVOL < 1.0 (the same displacement without
  the volume; the mechanism says it should not revert). Secondaries, reported only: the primary with RVOL ≥ 1.5 and ≥ 2.0.
- Trade: toward VWAP (short when above, long when below), entry at the open of the first bar starting 10:35–10:39, exit at the
  open of the first bar starting 11:35–11:39; missing either = void, counted. Outcome: the return in basis points signed by the
  direction (gross) and net of one tick each side (ES and NQ 0.25, about 0.6 and 0.2 bp). One trade per market per day.
- **Placebo**: the same rule at 10:05 (window 09:30–10:04, trade 10:05 → 11:05), the DAY leg's own time, where the development
  data shows nothing (t 0.3–0.8).
- Descriptive, no claim: the ledger's 10:00 clock option outcome (`opt_ret_ask_to_bid`, the ES/NQ puts) on signal days, by side.

## Verdict
- Judged once, at the first month-end judging at which at least 20 signal-days (days on which ES or NQ gave a primary signal)
  have accrued; monthly reports before that carry no verdict; no threshold, time or window is retuned at any point.
- PASSES only if: the net mean over signal-days is positive with t > 2 across days (a day's figure is the mean over its
  qualifying markets); positive in both halves; and the control's mean is below half the primary's mean (the effect must come
  with the volume). Reported regardless: by market, by side (above VWAP = the puts' direction), gross and net, the two
  secondaries, the placebo, and the descriptive 10:00 column.
- Expectation stated now: on the development data about 14% of days give a signal in at least one of SPY/QQQ (52 of 383), so
  ES + NQ should give roughly 2–3 signal-days a month and reach 20 around mid-2027. If purchased CME 1-minute history with
  volume arrives (DEFINITIVE_PLAN.md), the same rule is also judged once on the untouched block, its dates committed before
  opening, reported separately. A pass earns a frozen paper rule for a further test, not money and not a DAY-leg change.

## Development evidence (SPY/QQQ 5-minute bars, March 2025 – September 2026, 383 days each; prereg/vwap_fade.py, vwap_fade2.py)
Toward-VWAP return 10:35 → 11:35 in bp, |DISP| ≥ 0.25, signal at the close of the 10:30 bar:
- RVOL ≥ 1.25: SPY n 40 +19.9 (t 3.38, positive 72%, halves +19.9 / +19.9); QQQ n 35 +25.1 (t 2.83, halves +23.9 / +26.2);
  day-clustered both markets 52 days +19.8, t 3.56. Above and below VWAP both positive (SPY +18.0 / +23.0; QQQ +22.5 / +29.0).
- RVOL ≥ 1.5: SPY n 20 +24.6 (t 2.48); QQQ n 19 +34.5 (t 3.13; the grid's best cell); 29 days +22.6, t 2.80. RVOL ≥ 2.0:
  8 days +32.6, t 3.13.
- Control (RVOL < 1.0): SPY n 40 −4.6; QQQ n 59 +3.4. RVOL 1.0–1.5 alone: −1.1 (n 49).
- Time profile: positive only 10:15–11:00 at elevated volume; 12:00–14:00 zero or negative; 10:00 nothing. GLD/USO 10:30:
  +22 / +40 (t 1.9 / 1.6). Permutation null (returns shuffled across days within each time, the 40-cell morning grid, cells
  with n ≥ 8): the best cell's |t| 3.13 is matched in 9.2% of 500 shuffles.
The registered primary is the wider RVOL ≥ 1.25 cell, chosen for its count, not its t. Known differences from the judged data:
futures rather than ETFs (ES tracks SPY within a few bp over an hour), 1-minute rather than 5-minute bars in VWAP and R20, and
a 2025–26 sample with an upward drift. Nothing here changes the DAY method.

## What was seen before this was written
The script was run on the recorded October bars with the baseline minimum lowered to 2 sessions (6–8 October, five
market-days): no primary signal; 8 October was a control day in both markets (above VWAP on low volume: ES +28 bp, NQ +37 bp
toward VWAP). Those days are the baseline, not judged, and are stated here so they cannot be claimed later.
