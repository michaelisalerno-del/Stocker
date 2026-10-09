# Keeping peaks without cutting the big winners: a trend-aware trail and a widening ratchet — frozen protocol (2026-10-09, on the user's "Freeze", before the data it is judged on)

The user: "I often reach positive before it goes negative ... I really need a method to keep the peaks", keeping the big
winners. Proposed by Codex and agreed with Claude on 2026-10-09 (`~/Codex/2026-10-09-claude-codex-entries-exits/`,
`PEAK_BRIEF.md`, `codex_peak_round1.txt`). Both rules are replayed on the recorded quotes of the app's option at every
clock; nothing here changes the runtime (the app's own exit is the armed trail, PR #18, if deployed). Neither is a
direction signal: an early sale replaces the scheduled sale and reshapes the payoff; the test is whether that reshaping
pays after the spread. The hash of this file is in `PEAK-TRAIL-PROTOCOL.sha256`; any later change is an amendment, dated
and stated as such, made before the results are looked at.

## Common definitions (as `EXIT-SET-3-PROTOCOL.md`)
- Option and quotes: the app's selected option at each clock and its recorded two-sided quotes.
- E = entry: the ask of the last quote at or before clock + 3 s (at most 30 s old).
- Baseline exit: the bid of the last quote at or before clock + 60 min + 3 s (at most 30 s old; no bid = 0).
- M = the highest recorded bid from entry to the quote being tested. "Sell" = the bid of the first quote, after
  arming, that meets the rule's condition; if none does before clock + 60 min + 3 s, the baseline exit.
- Return = exit bid / E − 1.

## Rule T — trend-aware trail (judged on CL and GC; ES and NQ reported only)
- Arms at the first quote with bid >= 1.20 x E (the frozen armed trail's arming point). At that moment, once, from the
  app's Saxo 1-minute futures bars of the option's underlying (completed minutes strictly before the arming minute):
  - trend score = the t-statistic of the least-squares slope of the last 60 completed one-minute closes on time,
    signed positive in the option's favour (calls: rising; puts: falling). A score, not a significance test.
  - volume ratio = the last completed minute's volume / the mean volume of the 60 completed minutes before it.
  - class: STRONG if trend score >= 1.4 and volume ratio >= 1.5; WEAK if trend score <= 1.1; otherwise MIDDLE.
- Sells at the first later bid <= k x M, with k = 0.65 (STRONG), 0.85 (WEAK), 0.75 (MIDDLE).
- If fewer than 60 completed minutes or any missing volume: MIDDLE (k = 0.75, identical to the frozen armed trail).

## Rule R — widening ratchet (all markets)
- Arms at the first quote with bid >= 1.50 x E.
- Sells at the first later bid <= 0.30 x M + 0.85 x E. (Retains about +30%, +45%, +105% at peaks of +50%, +100%, +300%;
  allowed fall from the high about 13%, 28%, 49%.)

## Verdict (each rule separately, once, after 2026-11-27)
- Judged clocks: from the first clock after the freezing commit to 2026-11-27 16:00 New York; halves split at
  2026-10-31. Affected clocks = clocks where the rule armed. T: CL and GC only; R: all four markets.
- Each rule is compared, on the same affected clocks, with (a) the baseline exit and (b) the frozen armed trail (AT,
  `EXIT-SET-3-PROTOCOL.md`). Difference per clock = rule return − comparator return.
- A rule PASSES only if, against BOTH comparators: mean difference positive with t > 2.5 across days (the daily mean over
  affected clocks), at least 40 affected clocks on at least 10 days, positive in both halves.
- Tail check (reported, and a pass is void if it fails): over the clocks whose baseline return is in the top 5%, the
  rule's mean return must be at least 50% of the baseline's — keeping the big winners is the point.
- Reported regardless: by market and session (09:00–16:00 New York clocks and the rest); T by class (STRONG / MIDDLE /
  WEAK counts and returns); the eventual +100% winners (baseline >= +100%) and what each rule kept of them; arming rates.
- Paper caveat: exits at a recorded bid are estimates; a real sale may get less.

## Script
The scoring script is written after this file and recorded with its sha256 in a dated amendment before any judged
result is computed. Until then no figure for either rule on clocks after the freezing commit is computed.

## What was seen before this was written
Nothing for these rules: no trigger, class or return was computed on any data. The thresholds (1.20, 1.4, 1.1, 1.5,
0.65/0.75/0.85; 1.50, 0.30, 0.85) were set in the discussion from earlier descriptive findings (to 6 Oct: CL/GC touches
that kept rising had 60-minute trend t >= 1.39 and loud volume; touched +50%/+100% ended +27%/+59% on average).
