# PRESSURE-PROGRESS-PROTOCOL — amendment 1: the column script (2026-10-05, before any judged result is computed)

Amends `PRESSURE-PROGRESS-PROTOCOL.md` (sha256 22151a21819470dce52fdc6d68af9d613d2158486876066820c7ad920cbcff65) as that
protocol required: it records the script that writes the judged columns. No rule, threshold or verdict changes. The hash
of this file is in `PRESSURE-PROGRESS-AMENDMENT-1.sha256`.

## The script
`pressure_progress.py` in the hourly ledger (sha256 077ea4f45fe083c3f0873e9c863f51387ee9bebb509ee903540af83f1165d1e8),
run after `build.py`, writes `pressure_progress.csv`: one row per clock with the app's selected option, with the
baseline (`baseline_ret`) and, for the specification (`pp_`) and its no-Level-2 control (`np_`), the entry, exit and
direction columns and their paired differences against the baseline (`*_delta`).

## How the frozen definitions were made computable [definitions added]
- Book: CURRENT book-flow messages of the clock's market (futures only, not the following contract month); level prices
  in ticks and sizes from `book_flow.basis`; midpoint = (best bid + best ask) / 2 in price units.
- "The median imbalance across the latest 15 seconds": the median of the three 5-second samples at t − 10, t − 5 and t.
- Progress and minimum movement at a sample t use the seven 5-second samples from t − 30 to t; all must be healthy.
  V needs all 15 completed one-minute bars (Saxo bars from the bar cache); otherwise the sample is unhealthy.
- "All conditions hold for 10 seconds": they hold at the three samples t − 10, t − 5 and t.
- Entry: the first sample at clock + 5, 10, 15 or 20 s that qualifies; the option is bought at the ask of its last
  recorded quote at or before that time (at most 30 s old). Cost drag: buy cost = (ask + one option tick) x (1 + the
  clock's recorded FX markup) + entry costs; immediate net = (bid − one tick) x (1 − markup) − exit costs; the costs are
  the clock's recorded per-side costs (`option_context.costs`), converted to option price units by premium_gbp / limit;
  the option tick from the option's recorded tick scheme.
- Exit: the boundary starts from the 5-second samples of the minute before entry; once two whole calendar minutes have
  completed after the clock, at each new minute the boundary is recomputed from the samples of the two latest completed
  minutes (raised only for a call, lowered only for a put). The exit fires at the first sample at which the midpoint has
  been beyond the boundary at every sample of the last 10 seconds and the direction state opposes the position.
- Direction: the state at the clock; the contract from the chain's minute samples at the clock's minute (ask) and at
  clock + 60 minutes (bid), the app's expiry.

## Coverage on the look days (counts only; no returns were computed)
On the 48 clocks of 1–5 October with the app's option recorded through the hour: entries 0 (control 1), early exits 8
(control 46), non-UNCLEAR direction states at the clock 0 (control 4). Across those hours, the five-level pressure was
beyond ±0.20 on 10–14% of healthy samples and all three conditions aligned on 1–2%. The specification will therefore act
rarely. The verdict stands as frozen; an entry or direction comparison with too few trades for condition (d) is reported
as "insufficient trades to judge", which is a failure to pass, not a pass.
