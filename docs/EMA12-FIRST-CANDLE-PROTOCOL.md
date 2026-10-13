# The first 5-minute candle against EMA(12) — frozen protocol (2026-10-13, on the user's "Yes", before the data it is judged on)

A paper test on the hourly ledger; nothing here changes the runtime. The hash of this file is in `EMA12-FIRST-CANDLE-PROTOCOL.sha256`;
any later change is a dated amendment made before the results are looked at.

## Where the idea came from
A TikTok video (quant_lab.de, @thequantbuilder, video 7692393486640811296): if the first 5-minute candle after the New York open
closes above the 12 EMA go long, below go short; exit undisclosed. Tested in `~/Codex/2026-10-13-first-candle-ema/`, each step
written down and hashed before it ran:
- `FC_PREREG.md` (sha 42e35fdd): QQQ 1-minute 2013-2026, EMA over every 5-minute bar including extended hours, entry 09:35, held to
  16:00, 1 bp: +5.8 bp/day (t 2.4) 2013-2018, +5.5 bp/day (t 2.9) 2019-2026, short side not negative. Passed. SPY weaker (+2.9 to
  +4.6 bp). Gap sign, pre-market drift and first-candle colour alone are ~0 after 2018.
- `STOPS_PREREG.md` (sha 333f26c7): chandelier 2/3 x ATR and break-even-then-trail stops cut it to +0.5..+1.8 bp; the same stops on
  coin-flip entries make nothing. Take-profits of +25/+50 bp remove it. The gain is a slow drift across the whole day.
- `OPEN_COND_PREREG.md` (sha 3105f044): the strongest third of signals (distance from the EMA scaled by the day's typical move)
  carries it: QQQ +11.3 / +9.7 bp/day (t 2.1 / 2.9), the weakest third ~0. Fear-rising days add no direction.
- `EMA_GRID_PREREG.md` (sha d5e77a05), `OWN_OPEN_PREREG.md` (sha c7f84a5e), `NCANDLE_PREREG.md` (sha 9641c4f4): NQ positive in
  both halves for EMA 5-50 (fades at 100-200); ES about half; GC tiny at its own 08:20 open; CL flips sign; deciding after 2-12
  candles or waiting for confirming candles does not beat the first candle in both halves.

## Definitions
- Bars: the ledger's Saxo 1-minute futures bars (`build.bar_grid`, closes carried up to 5 minutes). The 5-minute closes are the grid
  values on 5-minute boundaries, overnight included; missing bars are skipped, as on a chart. EMA(12) with alpha 2/13 over them.
- Signal per market and New York weekday: the 5-minute close at 09:35 (the 09:30-09:35 candle) above the EMA(12) at that bar ->
  long (+1), below -> short (-1). Needs the 09:35 close and at least 30 of the 36 five-minute closes of the prior 3 hours.
- Strength = |close / EMA - 1| / (the gauge's last close before the day / 100); gauge NQ VXN, ES VIX (Cboe daily closes,
  downloaded each run as `fear.py` does). STRONG = strength >= the top-third cut of 2013-2018: NQ 0.00691 (from QQQ + VXN),
  ES 0.00577 (SPY + VIX). This gauge scale replaces the research's trailing 20-day range, which the ledger cannot supply before
  late October; it was checked before freezing (QQQ strong +15.2 bp t 3.0 in 2013-2018, +11.3 bp t 3.4 in 2019-2026 on 41% of
  days, the rest +1.2 / +1.3; SPY +10.4 / +6.1; the two definitions agree on 88-89% of days).
- Futures outcome: sign x (16:00 close / 09:35 close - 1) - 1 bp.
- Option outcome: from the chain board sampled at 09:35 (`build.chain_at`), the same-day expiry's call (long) or put (short) with
  Saxo's delta nearest 0.65 among two-sided quotes with |delta| 0.55-0.75 (ties: tighter spread). Bought at the ask, settled at
  its intrinsic value at the futures' close of the expiry minute (as `fear.py` and LEDGER-STRADDLE-EXPIRY-AMENDMENT-20261013).
  `o_ask` = value / ask - 1, `o_mid` = value / mid - 1.
  `carry_bp` = ((value - ask) / delta - sign x (futures at expiry - 09:35 close)) / 09:35 close x 10,000: what carrying the direction
  through the option instead of the future costs (or earns), per unit of delta, in bp of the future; `carry_mid_bp` at mid.
  No board at 09:35, no same-day expiry, no qualifying strike, or no futures bar at the expiry minute: no option outcome.

## Sample and timing
Weekdays from 14 October 2026.
- OPTION COST is judged once, at the later of 27 November 2026 and the day of the 20th NQ day with an option outcome; the judged
  sample is every NQ signal day (strong or not: the price of carrying the direction does not depend on the signal's strength)
  from 14 October to that date.
- DIRECTION is not judged on this sample (too small: a 5-10 bp drift against a ~100 bp daily swing needs years). It is checked once,
  on the first 250 NQ days with a futures outcome (about October 2027), for breakage only.
Before the cost judging point `emafc.py` prints counts only.

## Verdicts
- OPTION COST: mean `carry_bp` over the judged sample, standard error se.
  VIABLE if the mean is above -5 bp and (mean + 10) / se > 2 (an option costs at most about half of the strong-day drift of the
  research); NOT VIABLE if (mean + 5) / se < -2; otherwise UNDECIDED.
- DIRECTION CHECK (first 250 NQ futures days): BROKEN if the mean futures outcome is below zero with t < -2; otherwise NOT BROKEN.
- Reported, not judged: strong vs other days (futures and carry), `o_ask`, `o_mid`, `carry_mid_bp`, and ES throughout.
A pass is evidence for a later rule; it does not change the app.

## Scripts (hourly ledger folder)
- `emafc.py` sha256 b8cd1369a4f1268be14f4f5bee3835fdd8b942a5d7202d6f869944ba13a89eb8 (uses `build.bar_grid`, `build.price_at`,
  `build.segments`, `build.chain_at`, `fear.gauges`, `fear.tstat` unchanged; writes `emafc.csv`). Run by hand after `sync.sh`.

## Disclosure
- The research data (2013-2026 ETF stand-ins) was searched across exits, conditions, EMA lengths, markets, open times and candle
  counts, each step registered first; the strength filter and NQ come out of that search.
- While writing the script Claude ran it on 1-13 October (before the sample). Seen: NQ signals on 1, 2, 6, 7, 8, 9 and 12 October
  (strong on 8 and 12 October: futures outcome -95 bp and +17 bp); ES on 2, 6, 7, 8, 9 and 12 October; option outcomes on 6-9 October
  (NQ carry -11.8, -1.4, +32.4, -17.2 bp; ES -2.7, -22.7, -22.6, -2.7 bp). No board at 09:35 on 12 October; no regular-session bars
  on 5 October. Nothing from 14 October on has been computed.
