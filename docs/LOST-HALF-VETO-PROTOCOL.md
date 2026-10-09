# Lost-half veto — frozen entry test (2026-10-09, on the user's "Freeze", before the data it is judged on)

The idea came from outside the data: a trader's "50% retracement rule" (a pullback that gives back more than half of the
move is a downtrend you showed up early to; leave it alone), shown to the user on 2026-10-09. Claude wrote one definition
down before scoring it (the move over the 45 minutes before the last 15, and the last 15 minutes), and Codex reproduced
it (`~/Codex/2026-10-09-claude-codex-entries-exits/`, `claude_revisit_round1.md`, `codex_revisit_round2.txt`,
`REVISED_PLAN.md`). It is a paper test on the hourly ledger: the app keeps buying at every clock, nothing changes the
runtime and no order is sent. The hash of this file is in `LOST-HALF-VETO-PROTOCOL.sha256`; any later change is an
amendment, dated and stated as such, made before the results are looked at.

## Definition (parameters fixed here)
- At each eligible clock (every clock with the app's selected option, traded or skipped, that has both quotes as the
  frozen pullback entry defines them), from the ledger's futures minute closes (`price_at`: the close of the minute
  containing the instant, carried up to 5 minutes): C0 at the clock, C15 15 minutes earlier, C60 60 minutes earlier.
  These are the inputs of the ledger's `fut_ret_prev15` and `fut_ret_prev60` (`build.py`, `math.log(p0 / p)`).
- s = +1 for a call, −1 for a put. e = s × ln(C15 / C60) (the earlier move, signed in the trade's favour);
  m = s × ln(C0 / C15) (the last 15 minutes).
- Veto exactly when e > 0 and m < −0.5 × e. Any missing close: no veto.
- Baseline per clock: the app's own option bought at the ask at the clock and sold at the bid at clock + 60 min, as the
  ledger's `opt_ret_ask_to_bid` (no bid = 0). Vetoed clocks earn 0. `lost_half_delta` = (0 if vetoed else baseline)
  − baseline, i.e. −baseline on vetoed clocks and 0 elsewhere.
- All four markets (CL, ES, GC, NQ), every clock; no market, session, spread or threshold tuning. The thresholds (45/15
  minutes, one half) are the ones scored; the extrema-based version (low-to-high of the prior hour, midpoint) was
  discussed and is NOT this test.

## Data
Judged clocks: from the first clock after the freezing commit to 2026-11-27 16:00 New York. Halves: to 2026-10-31 and
from 2026-11-02. Clocks of 5–8 October were the look and are not judged.

## Verdict (once, after 2026-11-27)
PASSES only if: (a) the mean `lost_half_delta` over all judged clocks is positive with t > 2.5, t across days of the
daily mean; (b) at least 40 vetoed clocks on at least 10 distinct days; (c) positive in both halves; (d) the portfolio
that remains (kept clocks at the baseline, vetoed clocks at 0) is positive after the app's own paper fill and fees;
(e) one-sided day-level p < 0.025, the bar shared with the next-day substitute (two hypotheses frozen together).
Reported regardless: vetoed and kept returns by market and by session (09:00–16:00 New York and the rest); overlap with
the frozen momentum-against and break-even vetoes and the pullback entry; the vetoed clocks whose option at least
doubled (what the veto cost); how often the veto fired.

## Script
The scoring script is written after this file and recorded with its sha256 in a dated amendment before any judged
result is computed.

## What was seen before this was written (5–8 October, 218 clocks with both prior closes)
Vetoed 15: mean −48%, median −41%; kept 203: −11% / −19%. Shuffling returns within market × day (100,000 permutations):
one-sided p = 0.011 for this one test. 3 of the 15 overlap a momentum-against proxy; entry spreads equal (12.0% vs
10.9%). Vetoed worse than kept on 5, 6, 7 and 8 October (n 3, 2, 7, 3). Pullbacks that held half (n 27) did NOT do
better (−27%); entries against the earlier move (n 116) were not worse (−19%).
Disclosure: the same day's rounds scored about 2,400 rule evaluations in all (option and straddle rules, single-column
and paired vetoes, direction rules); this rule was not among them and was stated once before scoring, but 8 October had
been inspected in earlier rounds before it was used as the check here. n = 15 is small; this freeze is what lets it be
judged on unseen clocks, not evidence that it works.
