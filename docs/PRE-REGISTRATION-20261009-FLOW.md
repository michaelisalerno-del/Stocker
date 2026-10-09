# Pre-registered options-flow vetoes (2026-10-09): addendum to PRE-REGISTRATION-20261009.md

Two rules written and hashed before scoring (Claude: `claude_flow_prereg.md`, sha256
cc7b8406d2cbb6f4a63916bd11dfc6fbf0f14816415dc89007d2fe37f27af56c; Codex: `codex_flow_prereg.txt`, sha256
10f8a667078a2a63920076bf5af75ae65ab8f689048d9f45d6682ec4818ed9f8; both in
`~/Codex/2026-10-09-claude-codex-entries-exits/`, code in `prereg/`), scored once on the 5–8 October clocks (269; the
chain board with per-strike open interest and session volume is recorded every minute from 5 October).

- FC "crowded strike": veto when the trade side's session volume / open interest at the app's strike ±2 strikes on the
  app's expiry is >= 0.5 (>= 50 contracts). Fired on 58% of clocks (the threshold is routine for same-day chains);
  vetoed −15.5% vs kept −16.5%; would have vetoed the three largest winners. Dropped.
- FK "opposite-side flow": over the 30 minutes before the clock, strikes within one expected move of the future on the
  app's expiry, call and put volume increments summed; live at >= 100 contracts; veto a call when puts are >= 75% of
  the volume, a put when calls are. 11 vetoes (4%), −44% mean / −27% median vs kept −15%, one-sided p 0.062 (returns
  shuffled within market × day), worse than kept on 3 of 4 days, best vetoed trade +12%. Not a pass. Scored again,
  unchanged, at the 30 October search on 5–30 October alongside K1 and K3 of the main pre-registration.
