# Session policy in the runtime, alongside the hourly trades (2026-10-09; the user's decision: "use the best method we've discovered so far in my app and continue recording")

## What changes
- The first clock of each CME session (19:00 New York; there is no 18:00 clock) becomes the SESSION trade: the app buys its
  normally selected option (the overnight rule selects the next day's expiry) and holds it, instead of 60 minutes, until
  09:25 New York on the next weekday — or earlier, on an armed trail that arms when a current bid reaches 1.50 x the entry
  price and sells at the first later current bid at or below 0.60 x the highest bid since entry. An empty bid never
  triggers a sale (the scheduled exit still applies). The option subscription is kept until the exit.
- Every other clock trades exactly as before (60-minute hold; the armed trail of EXIT-SET-3 rule AT, 1.20 / 0.75), so the
  hourly incumbent keeps producing its fills and the frozen hourly tests keep their data. Selection, recording and the
  chain views are unchanged at every clock. The event carries `policy` ("SESSION" or "HOURLY") and `trail` (arm, keep).
- Paper only (INTERNAL_PAPER); no configuration change; RULE_VERSION unchanged (the clock set and the frozen definitions
  are unchanged; the 19:00 clock's exit and trail are the only differences, and they are recorded on the event).

## Why (the day's evidence, `~/Codex/2026-10-09-claude-codex-entries-exits/DEFINITIVE_PLAN.md`, `STUDY_LOOK.md`)
The loss is the toll (entry −7.9, mid-to-mid −0.6, exit −7.4 points per £ of premium, 269 trades); the direction never
changes per market; one option per session beat the sum of the hourly trades in 84–90% of sessions; an open-bought option
is worth most around 09:25 and bleeds through the day as its exit toll climbs; bigger moves give back less, so the
60-minute trail (0.75) is too tight for a session. None of this is a proven edge; it is the lowest-toll version of the
same bet, run live in paper so the study (`STUDY_DESIGN.md`) has its own prospective data.

## Effect on the frozen tests (stated as an amendment, dated 2026-10-09, before any judged result)
- Protocols judged on recorded quotes at every clock: unaffected (every clock still selects and records its option).
- Variants that use the app's own fill at a clock: the 19:00 clock's hourly fill no longer exists from the first session
  after deployment; those variants are judged on the remaining 20 clocks a day, and the 19:00 clock is reported separately.
- The session trade's own results are judged by STUDY_DESIGN.md (G1 x X4 against the hourly incumbent, first-trade-then-
  cash and the full-session trail), not by any hourly protocol.
