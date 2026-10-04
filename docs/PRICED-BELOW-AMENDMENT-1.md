# PRICED-BELOW-PROTOCOL.md, amendment 1: per-market verdicts (2026-10-04, before any judged clock)

Dated 2026-10-04, before the first clock these tests judge (2026-10-05). The hash of this file is in
`PRICED-BELOW-AMENDMENT-1.sha256`. It adds verdicts; it removes or changes none.

## Why
The user's point: an edge may belong to one instrument, and pooling the markets can hide it. The research already
shows instruments behaving differently (look14's fit NQ 0.43 against 0.12–0.22 elsewhere; the open-to-close pattern
NQ only; spreads, implied premia and timetables differ by market).

## Added verdicts
- For each condition H1–H5 and each market judged under MARKET-CHANGE-AMENDMENT-20261004 (CL, GC and NQ at the
  09:00–16:00 New York clocks; H4 is already CL- and GC-specific, so it gives one verdict for each), the same outcome
  as the pooled verdict (`straddle_ret_ask_to_bid`, or `opt_ret_ask_to_bid` under the pooled rule's fallback, applied
  per market).
- A market-specific verdict PASSES only if, judged once after 2026-11-27: the mean is positive with t > 3 across that
  market's days (14 market-specific tests: about a 2% chance that any passes by luck), at least 40 qualifying clocks
  on at least 10 days in that market, and the mean is positive in both halves.
- The pooled verdicts are unchanged. A market-specific pass, like a pooled one, earns a paper rule frozen for a
  further test on that market alone, not money.
- ES is reported per condition where its columns exist (its look14 profile and burst threshold do not yet), never in
  a verdict.
