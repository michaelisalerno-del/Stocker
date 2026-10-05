# Exit set 3: an armed wide trail, and selling when the option's volatility is falling — frozen (2026-10-05, before the data they are judged on)

The user decided to settle exits before any entry rule and chose to freeze the two exits that the app's own trades pointed
to and no earlier protocol covers. They are paper tests on the hourly ledger: the app keeps holding every trade for the
hour, nothing changes the runtime and no order is sent. The hash of this file is in `EXIT-SET-3-PROTOCOL.sha256`; any
later change is an amendment, dated and stated as such, made before the results are looked at.

## Where the ideas came from, and what was seen
- Armed trail. On the app ledger's 72 closed paper trades of 1–5 October with a recorded option path (actual fills),
  selling once a trade had been up 20% and then gave back 25% of its highest bid moved the mean from −24% to −19%, the
  median from −30% to −10% and the winners from 14 to 21, and cut the largest winner from +378% to +67%. Every variant
  tried (leg memory, volatility classes, quote lean) traded the typical trade against that one runner.
- Volatility falling. On the same trades from 4 October, the chain's MidVolatility at the option's strike falling over the
  first 10 minutes went with a median −27% from holding on, against 0% when it rose (15 and 29 trades). The measure frozen
  here is the option's own implied volatility (its mid and the futures' mid, Black-76), because it exists on every clock;
  on the 31 look clocks where both exist, the two agreed on the direction 23 times.
- On this protocol's own definitions (48 look clocks, entry at the ask at the clock): selling at the bell −15.5% per pound;
  the armed trail −14.0% (+1.5 points; better on 12 clocks, worse on 5); the volatility exit −2.8% (+12.7 points; better
  on 15, worse on 4; on its 19 firing clocks the bell averaged −43%, against +3% on the others). Both were chosen by
  looking at these days, which are not used to judge them.
This sits beside the frozen exits of `L2-VETO-EXIT-PROTOCOL.md` (X1–X6), `MINUTE-EARLY-EXIT-PROTOCOL.md`,
`EXIT-SET-2-PROTOCOL.md`, `RELEASE-EXIT-PROTOCOL.md` and `LEAN-EXIT-PROTOCOL.md` (L, T, R).

## Definitions (parameters fixed here)
- Option and quotes: the app's selected option at each clock and its recorded two-sided quotes (not gaps).
- Entry: the ask of the last quote at or before clock + 3 s (at most 30 s old). Baseline exit: the bid of the last quote at
  or before clock + 60 min + 3 s (at most 30 s old; no bid = 0). `x3_base_ret` = exit bid / entry ask − 1.
- AT, armed trail: once a bid has reached 1.20 x the entry ask, sell at the first later bid at or below 0.75 x the highest
  bid since entry (`at_ret`); otherwise the baseline. `at_armed` marks the clocks where it armed.
- IV, volatility falling: implied volatility = the Black-76 implied move of the option's mid on the futures' weighted mid
  (the last CURRENT book-flow message at most 30 s old), divided by the square root of the hours to the option's expiry
  instant; at entry and at clock + 10 min + 3 s. If it is lower at minute 10, sell at the bid of the minute-10 quote
  (`iv_ret`); otherwise the baseline. `iv_known` marks the clocks where both values exist; unknown means hold.
- Paired differences per clock: `at_delta` = `at_ret` − `x3_base_ret`; `iv_delta` = `iv_ret` − `x3_base_ret`.
- Column: `exits3.csv`, written by `exits2.py` in the hourly ledger (sha256
  5739edd2a6f1a361dd87bbb5ef3d8a1ec3b54e1c6fe0e0d562237beae73009e1 at freezing) after `build.py`; it also records the
  chain's MidVolatility at the strike (`iv_chain_entry`, `iv_chain_10`) for the report.

## Data
Every clock with the app's selected option, traded or skipped, from 2026-10-06 to 2026-11-27. Halves: 6–30 October and
2–27 November. 1–5 October is the look and is not judged.

## Verdict
Each exit is judged once, on its own, after 2026-11-27, on the clocks where it can act (AT: `at_armed`; IV: `iv_known`).
It PASSES only if: (a) its mean paired difference is positive with t > 2.5, t across days of the daily mean (two tests,
both chosen by looking); (b) at least 40 such clocks on at least 10 distinct days; (c) the mean is positive in both
halves. The mean decides: an exit that helps most trades while selling the few large winners has not paid.

Reported regardless, for each: the median difference, clocks helped and hurt, the fire minute, the clocks whose option at
least doubled from the entry ask and what the exit did to them, the largest single contribution either way and the verdict
without it; by market; the original 09:00–16:00 clocks against the others; the frozen-scope clocks on their own; the
app's actual trades (its fills, commission and FX) on the same clocks; for IV, the same split by the chain's MidVolatility;
and the two together (AT after IV did not fire).

A pass earns an exit a place in the frozen method only by a dated rulebook amendment and a new rule version after
2026-11-27.
