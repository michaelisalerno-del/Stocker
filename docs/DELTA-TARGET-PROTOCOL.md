# The strike: nearer the money against the frozen 0.10 delta — frozen protocol (2026-10-05, before the data it is judged on)

The user asked what "the cheapest option" means after the strike-level look found no underpriced strikes, only a cost
gradient from the money outwards: the option that is cheapest in premium is the dearest in spread. This is a paper
comparison on the hourly ledger of the same trade at a different strike. The app keeps trading the frozen 0.10-delta
method unchanged, nothing below changes the runtime and no order is sent. The hash of this file is in
`DELTA-TARGET-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the results
are looked at.

## Where the idea came from, and what the look says
On 1–5 October (1,083 out-of-the-money strike-sides on 47 clocks), every strike's mid-to-mid change over the hour was
within −2% of zero: the options themselves barely decayed inside the hour, and the whole loss was the spread, which runs
from about 8% of the mid at the money to about 28% at the lowest-premium strikes. Picking the "cheapest" strike by
priced level or by spread returned the at-the-money strike every time: there is no separate underpricing, only the
gradient. Paired on the same 25–26 clocks (two days, the chain windows began on 4 October), the strike nearest 0.50
delta beat the app's option by 11 points per clock and was better on 96% of clocks; the strike nearest 0.30 delta by 8
points and 88%. Two days cannot judge this; those days are not used to judge it.

## Definitions (no free parameter)
- Clock, right, expiry, hold and fills are the app's: the clock's selected option (`option_context.identity`) fixes the
  right (call or put) and the expiry; the hold is the clock to clock + 60 min; entry is the ask at the clock and exit the
  bid at the hour's end, as `opt_ret_ask_to_bid` and the ledger's chain columns define them.
- Chain: the clock's recorded chain window for that expiry (`option_chain` in the clock signal; the hour-end quotes from
  the chain sample at the end minute, `build.chain_at`). Only strikes with two-sided quotes at both ends count.
- D30 and D50: among the chain's strikes of the app's right that are out of the money or at it, the strike whose Saxo
  delta at the clock is nearest 0.30, and the one nearest 0.50, each within 0.12 of its target (else no comparison for
  that clock). Returns are ask at the clock to bid at the hour's end, in % of the ask (`d30_ret_ask_to_bid`,
  `d50_ret_ask_to_bid`), beside the app's own (`app_ret_ask_to_bid`).
- Paired difference per clock: `d30_delta_vs_app` and `d50_delta_vs_app` (the alternative's return minus the app's).
  Returns are per pound of premium, which is the comparison that answers "cheapest"; the pound outcome of one contract
  (a D50 contract costs several times a 0.10-delta one) is reported, not judged.

## Data
The hourly ledger (`~/Documents/Codex/2026-10-03-hourly-ledger`, README.md), `picks.csv` written by `edges.py` (sha256
d5a7ec79140b6d407e7bba01cce41824d748cf10a33801e36993318871cf3b66 at freezing) after `build.py`: one row per clock with
the app's option outcome and a chain alternative. Judged clocks: 2026-10-06 to 2026-11-27, every clock the app observes,
traded or skipped. Halves: 6–30 October and 2–27 November. 1–5 October is the look and is not judged.

## Verdict
Two comparisons, D30 and D50, each judged once on its own after 2026-11-27. A target PASSES only if: (a) the mean paired
difference is positive with t > 2.5 (t across days of the daily mean; two tests); (b) at least 40 clocks on at least 10
distinct days; (c) the mean is positive in both halves. Reported regardless: the median difference and the share of
clocks the alternative won; the spread paid and the mid-to-mid change for the app's option and the alternative (how much
of the difference is the spread); one contract's pound outcome; by market; the original 09:00–16:00 clocks against the
others; the largest single difference either way and the verdict without it; the frozen-scope clocks (CL, GC, NQ,
09:00–16:00) on their own, since a pass there is what would amend the running method.

A pass earns the delta target a place in the frozen method only by a dated amendment of the rulebook and a new rule
version, after 2026-11-27; until then the app's 0.10-delta selection runs unchanged.
