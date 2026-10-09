# Next-day substitute at three hours to expiry — frozen selection test (2026-10-09, on the user's "Freeze", before the data it is judged on)

Same-day options bought with three hours or less to expiry were the worst group in the recordings (below): mostly time
value that decays inside the hold, a spread about three times wider, and often no bid at all to sell to at the hour's
end. Skipping them would also skip some winners; the test instead buys the NEXT exchange trading day's expiry at those
clocks — same market, same direction, same delta target — and asks whether that beats both the same-day option and
staying out. Agreed by Claude and Codex on 2026-10-09 (`~/Codex/2026-10-09-claude-codex-entries-exits/REVISED_PLAN.md`,
`codex_revisit_round2.txt`). It is a paper test on the hourly ledger: the app's selection, orders and runtime do not
change. The hash of this file is in `NEXT-DAY-SUBSTITUTE-PROTOCOL.sha256`; any later change is an amendment, dated and
stated as such, made before the results are looked at.

## Definition (parameters fixed here)
- Affected clocks: clocks in CL, ES, GC or NQ where the app's selected option is a same-day expiry with
  0 < `opt_hours_left` <= 3 (hours from the clock to the option's own expiry instant, `opt_expiry` − clock; NOT the
  chain's `chain_hours_left`). The cut-off is three hours and is not widened.
- Substitute: from the ledger's chain board at the clock minute (`chain_at`, the board state at that minute; every
  strike of every listed expiry with bid and ask), the next exchange trading day's expiry, the same right as the app's
  option, the strike chosen by the app's own selector rule (the ~0.10 absolute-delta target and quote-validity rules in
  force at the clock, applied to that expiry's strikes using the ledger's smile/normal-variance pricing for delta). If
  no strike qualifies (no two-sided quote, or the selector's validity rules fail), the policy keeps the original option
  and the clock is recorded as a fallback (counted, not judged).
- Entry: the substitute's ask on the board at the clock minute. Exit: the same contract's bid on the board at the
  minute of clock + 60 min (an explicit no-bid = 0; a missing board at either minute = the clock is unscored and
  counted). Equal premium budget: returns per pound of premium, so the comparison is per clock, not per contract.
- Per affected clock: `sub_ret` = substitute bid at exit / ask at entry − 1; `same_ret` = the app's own option's
  `opt_ret_ask_to_bid`; `sub_delta` = `sub_ret` − `same_ret`.
- Costs: the app's own paper fill (one tick worse on each side) and fees applied to both arms in a reported variant.

## Data
Judged clocks: from the first clock after the freezing commit to 2026-11-27 16:00 New York. Halves: to 2026-10-31 and
from 2026-11-02. 5–8 October was the look and is not judged.

## Verdict (once, after 2026-11-27)
PASSES only if BOTH hold: against the same-day option, the mean `sub_delta` is positive with t > 2.5 (t across days of
the daily mean), at least 40 affected clocks on at least 10 distinct days, positive in both halves; AND against cash,
the mean `sub_ret` after the paper fill and fees is positive with the same t, count and halves bars. Also one-sided
day-level p < 0.025 (the bar shared with the lost-half veto; two hypotheses frozen together). Losing less than the
same-day option is not a pass.
Reported regardless: by market (the no-bid rate is mostly CL) and by New York hour; the substitute's delta, spread and
hours left at entry; fallback and unscored counts; the same-day winners the substitute would have missed (the same-day
options that at least doubled on affected clocks) and what the substitute made on those clocks; the 3–4 h group for
information only (not judged, the cut-off is not widened).

## Script
The scoring script is written after this file and recorded with its sha256 in a dated amendment before any judged
result is computed.

## What was seen before this was written (5–8 October, 269 completed trades)
By the option's own hours left: <= 3 h: 23 trades, −66% mean, −80% median, 7 with no closing bid (6 CL, 1 GC), 12 ended
at 20% or less of the entry premium, median immediate ask-to-bid loss 25%; negative in both periods (−70% on 5–7 Oct,
−55% on 8 Oct). 3–4 h: 16 trades, median −50% but mean +60% (one NQ put, +1,283%). 4–6 h −33%, 6–8 h −32%, 8–12 h −11%,
12–24 h −9%. All trades with <= 8 h were same-day. The <= 3 h cut skips two winners (CL +25%, GC +89%) and keeps the
four largest (NQ +1,283% at 4.0 h, GC +273% at 5.5 h, ES +220% at 4.0 h, CL +154% at 3.5 h). Replacing the 23 with cash
moves the sample mean from −15.9% to −10.3% (still a loss). Next-day substitutes were NOT measured: the 80 actual
next-day trades (−12.9%, no no-bids) were at other clocks. 29 affected positions were recorded but 6 have no closing
return. 8 October had been inspected in earlier rounds before it was used as the check.
