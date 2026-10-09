# Pre-registered pattern vetoes (2026-10-09): FVG, volume profile, liquidity sweep, GEX regime

Eight veto rules were written and hashed BEFORE scoring (Claude's four: `claude_prereg.md`, sha256
0a543c38ca0544dcf416ec6d7d2adaef9d8c4e8dee8b8409c500c09e2a441a37; Codex's four: `codex_prereg_round1.txt`, sha256
3abf9cc22d143d41f7f20e3c3b73986be88f861eb7c071a381a0639b38b64a5d; both in
`~/Codex/2026-10-09-claude-codex-entries-exits/`, code in `prereg/`), then scored once on the 1–8 October clocks
(309 with a hold result; GEX from 5 October). This is NOT a frozen protocol: it records the definitions so that the
30 October search on 5–30 October cannot retune them, and what was seen.

## The rules (vetoes; parameters as in the hashed files)
- C1 FVG against the trade, gap ≥ 0.25 × expected move, 120-min lookback, unfilled, clock close inside the gap.
- C2 Nearest high-volume node (top 10% of 1-tick bins over 240 min) ahead within 0.5 × expected move.
- C3 Sweep of the prior hour's high/low by ≥ 2 ticks with a close back inside within 5 bars, last 30 min, against the trade.
- C4 Dealers long gamma (gex_net_share > 0.2) and the trade's wall within 1.0 expected move.
- K1 FVG against the trade, gap ≥ 5 ticks, 60-min lookback, unfilled, clock close inside the zone.
- K2 Clock close inside the POC bin (10-tick bins, 240 min).
- K3 Sweep of the prior hour's high/low by ≥ 5 ticks, preceding bar inside, close back inside within 5 bars, last 30 min.
- K4 gex_net_share ≥ 0.5.

## Seen on 1–8 October (baseline −16.2% mean; null = returns shuffled within market×day)
C1 never fires. C2, K2: no difference. C3: vetoed trades did BETTER (+4% vs −20%; it vetoed the +1,283% NQ put).
C4: 15 vetoes, −24% vs −16%, p 0.33. K1: 10 vetoes, −39% vs −15%, p 0.31. K3: 29 vetoes, −36% vs −14%, p 0.059 alone,
0.42 as the best of eight; worse than kept on 3 of 6 days (the effect is 7–8 Oct). K4: 38 vetoes (ES/NQ), −24% vs −15%,
p 0.040 alone, 0.93 best of eight; 2 of 4 days. Nothing passes.

## What follows
At the 30 October search, K1 and K3 are scored with these exact definitions on the 5–30 October clocks; C1, C2, C3, K2
are dropped; C4 and K4 are reported only. A rule that then shows t > 2.5 across days may be frozen for November with
the full history (this file) disclosed. No parameter is changed on the way.
