# WALL-PROTOCOL: the scoring script, recorded before the judged run (2026-10-13)

Records the script that scores `WALL-PROTOCOL.md` and the points the protocol leaves open, fixed before any figure from the
judged days (5–16 October 2026) is computed. The protocol's rules are unchanged. The hash of this file is in
`WALL-PROTOCOL-SCORING-20261013.sha256`.

## Script
`wall_verdict.py` in the hourly ledger folder, sha256 b8cf1a70abb60c1856a3e6d5d22cca87c908d710101260b11e35a08cc4af5c5a. Run once
after the close of 16 October (`sync.sh` first): `python3 wall_verdict.py`. It writes `wall_pairs.csv` and prints the verdict
and every report the protocol lists.

## Points the protocol leaves open (as fixed in the script)
- Day = the New York calendar date of the sample, weekdays only. Book prices are `book_flow.basis` ticks.
- A CURRENT book is a record of the segment's own future (the manifest key's contract, so a pre-roll next contract is not mixed
  in) with `book_flow.status` CURRENT; the sampled book must have ten levels on both sides (samples without are counted).
- Coverage: the 300 s after a sample must lie inside one stretch of continuously valid CURRENT books (each book valid to its
  `valid_until`; a stretch ends at a GAP record or a non-CURRENT book); stretches from overlapping segments of the same market
  are joined. "The next 300 seconds" = books received in (t, t + 300].
- Verdict (a) and (c) use the mean of the daily mean scores; the pooled pair mean is shown too.
- Pulled walls: at each covered sample, every level 1–9 ticks out; the first CURRENT book in the next 300 s whose opposite best
  price is within one tick; "still at least half there" = its size then >= half its size at the sample (absent = 0).

## Validation, on the protocol's excluded look days only (1–2 October)
The ledger holds CL, GC and NQ captures for those days (the original look also had NG and SI). The original method on all five
markets reproduces the original 798 pairs exactly. On the three markets: original method 455 pairs, wall side reached 64.4% vs
other side 77.1%; this script 538 pairs (33 excluded for coverage), 69.1% vs 77.1%. The gap is narrower because the protocol
checks reach on every book message rather than every 5 s. No judged-day data has been read (`cache_wall/` holds only the nine
1–2 October segments).
