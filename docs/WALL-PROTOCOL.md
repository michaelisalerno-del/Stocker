# Big resting orders slow the price — frozen protocol (2026-10-03, before the data it is judged on)

The user chose to fix this now and judge it on the next two weeks of LIVE recordings. It is a property of the futures'
order book, not a trading rule; nothing below changes the runtime. The hash of this file is in `WALL-PROTOCOL.sha256`;
any later change is an amendment, dated and stated as such, made before the results are looked at.

## Where the idea came from
On 2026-10-01 and 2026-10-02 (`~/Documents/Codex/2026-10-02-deep-analysis`, section 24): at the same moment and the same
distance from the best price, with a wall on one side only, the wall's price was reached within 5 minutes 51% and 52% of
the time against 65% and 61% on the other side (798 pairs), in all five markets. Those days are not used.

## Definitions
- Book: the futures' 10-level ladder in the capture segments (`book_flow.basis`, price in ticks and size per level), from
  messages whose `book_flow.status` is CURRENT. Best bid and ask are the first levels.
- Samples: every 30 seconds of each capture (on the minute and half minute), the last CURRENT book at or before it, if it
  is at most 5 seconds old.
- Wall: on one side, a level other than the best (levels 2–10) whose size is at least 4 x the median size of that side's
  ten levels (a median below 1 counts as 1).
- Pair: at a sample, a distance d from 1 to 9 ticks at which one side has a level that is a wall and the other side has a
  level at the same distance from its own best price that is not. Of several such distances, only the nearest counts.
- Reached: an ask-side level at price p is reached if any CURRENT book within the next 300 seconds has its best bid at or
  above p; a bid-side level if any has its best ask at or below p. A pair without books covering those 300 seconds is
  excluded and counted.
- Score per pair: (other side reached) − (wall side reached): +1, 0 or −1.

## Data
- Every capture of every market from 2026-10-05 to 2026-10-16, all weekdays. 2026-10-01 and 2026-10-02 are not used.

## Verdict
The effect PASSES only if: (a) the mean score is positive with t > 2, t across days of the daily mean score (all markets
pooled per day); (b) there are at least 1,000 pairs on at least 7 distinct days; and (c) the mean score is positive in
both weeks (5–9 and 12–16 October). Reported regardless: reach rates for each side; by market, by distance, by New York
hour; the size multiple of the wall (4–8x, 8x and more); and, from every message, the share of walls still at least half
there when the opposite best price first comes within one tick of them, against ordinary levels. Analysis runs once,
after 2026-10-16. A pass describes the book; using it in an exit needs its own rule, frozen before the data it is judged on.
