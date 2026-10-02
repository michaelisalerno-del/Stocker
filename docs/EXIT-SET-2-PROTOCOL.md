# Two exits from the futures' own behaviour — frozen protocol (2026-10-02, before the data it is judged on)

The user chose to fix these exits now and judge them on the next two weeks of LIVE paper recordings. The app keeps
trading the frozen method unchanged (hold 60 minutes); nothing below changes the runtime. The hash of this file is in
`EXIT-SET-2-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the results are
looked at.

## Where the ideas came from
On the 24 paper trades of 2026-10-01 and 2026-10-02 (`~/Documents/Codex/2026-10-02-deep-analysis`, sections 10 and 12):
a trailing exit on the future gave −£199 against −£549 for holding (10 better, 4 worse) and kept the 2 Oct 15:00 silver
put (+£1,139); selling when buyers and sellers turned against the trade improved 13 of 17 trades but sold that silver
put five minutes in (−£917 in all; £796 better than holding without it). Around the tops of 137 big moves, buyers
minus sellers peaked at the top and turned a median 39 s after it. Selling into the first ten minutes after the 09:30
open or the 10:00 releases helped the two trades it touched. Those days suggested the rules, so they are not used.

## Common definitions
- Opportunities: every recorded clock from 2026-10-05 to 2026-10-16 with a verified candidate option (the option
  recorded at the clock in `signals.detail.option_context`), traded or not, whose option has two-sided quotes in the
  capture at the clock and at the exit E = C + 60 minutes, and whose future has quotes from C to E. Others are excluded
  and counted.
- Baseline and costs exactly as in `L2-VETO-EXIT-PROTOCOL.md` ("Data"): buy one contract at the ask plus one tick (the
  first quote within 20 seconds of C), sell at the bid less one tick at E (the last quote at or before E); fees per side
  and the conversion mark-up; as a share of the money paid in. Traded clocks use the actual fills for the baseline; an
  exit below always uses the capture, with the same entry.
- Future's price at t: the mid of its last two-sided quote (DelayedByMinutes 0) at or before t. Direction s = +1 for a
  call, −1 for a put. Favourable move fav(t) = s x (price(t) − price(C)); best(t) = the largest fav so far from C.
- One expected 15-minute move e15: the look14 level recorded at the clock (`signals.detail.forecast.level`, status
  OBSERVED) x the square root of the clock profile's variance for the 15 minutes from C (`forecast.normal_variance`)
  x price(C). Without an OBSERVED forecast the opportunity is excluded from E1 and E2 and counted.
- Checks every 10 seconds, at C + 10 s, C + 20 s, ... up to E − 10 s. A rule that fires at check t sells at the bid less
  one tick of the option's last two-sided quote at or before t + 3 s (Saxo's option prices arrive about 2 s after the
  future's). If that bid is missing or zero the rule does not sell at that check and goes on checking.
- Armed: from the first check at which best(t) >= e15.

## Exits
- **E1 Futures trailing.** Once armed, sell at the first check where fav(t) <= 0.5 x best(t).
- **E2 Buyers and sellers turn.** Once armed, sell at the first check where pressure(t) <= −0.2. pressure(t) = s x
  (buyer volume − seller volume) / total volume over the 60 seconds ending at t. Volume is the change in Saxo's
  cumulative `Volume` field between consecutive futures messages; it counts as buyer volume when the message's last
  traded price is at or above the ask of the quote before it, seller volume when at or below that bid, otherwise by the
  tick rule (above the previous last traded price buyer, below it seller, unchanged neither). With no volume in the 60
  seconds there is no pressure and no sale.
- **E3 Sell into a scheduled burst (reported now, judged later).** Sell at the first check inside the ten minutes from
  09:30:00 or from 10:00:00 New York time on which the option's bid is at least 1.2 x the price paid. No arming needed.

## Verdict
- E1 and E2 are each judged once, after 2026-10-16, on the opportunities they change (sell before E). One PASSES only if:
  (a) its mean improvement over the baseline (per opportunity, after costs, share of money paid in) is positive with
  t > 2, t across days of the daily mean improvement (days with none left out); (b) it changes at least 40
  opportunities on at least 7 distinct days; (c) the mean improvement is positive in both weeks (5–9 and 12–16 October).
- E3 is judged once, after 2026-11-27, on clocks 2026-10-05 to 2026-11-27 by (a) and (c, October against November) with
  (b) at least 40 opportunities on at least 10 days. After 2026-10-16 its two-week results are reported, not judged.
- Reported regardless, for each: the improvement in GBP per contract; by market and by clock hour; the share improved;
  the median; the result without the single best and the single worst opportunity; the whole strategy's result with
  the exit in place; E1 against E2 on the opportunities both change; and every exclusion.
- These are separate single tests. They do not change the other protocols or their counts.
