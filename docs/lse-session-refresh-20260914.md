# LSE scanner permission resolved after Gateway login

At 11:14–11:15 UTC on 14 September 2026, after the user completed a fresh PAPER Gateway login from Android, all five production LSE scanner families returned 50 rows with no precision warning 492. Both CORP and empty stock-type filters passed (10 requests). A separate forced real-time TOP_TRADE_RATE request also returned 50 rows with no warnings or API errors.

[Full scanner check summary](lse-scanner-after-login-20260914.csv). Cancellation acknowledgements for completed scanner requests and a farm-OK event were preserved in the API-event column; they are not permission failures.

Before this login, a fresh API client on the previous authenticated Gateway session returned 50 rows with warning 492 at 09:27 UTC. The warning disappearing after full broker reauthentication is consistent with stale session entitlements after subscription activation. No additional subscription was purchased, and no IOB or Level II subscription was added during this resolution. This establishes working access for the tested scanner scope, not a guarantee about future sessions or every exchange product.

MSLH and RS1 each returned real-time type-1 quotes and five exact opening bars via both direct LSE and SMART routing. These retrospective history reads do not establish timely availability at today's original selection deadline.

Stocker remains deployed at c8a3753. PAPER is connected, reconciled and ready, with zero positions and open orders. US is enabled and READY. Today's original LSE opening remains DEGRADED for its saved missing-prefix failure, Korea retains its missed-selection-window result, and ASX remains disabled. The prepared V10 availability-policy changes are still not deployed or applied to replacement runs.

The Gateway session was refreshed; the application service and saved run configuration were not changed. Android access was configured with the supplied ECDSA-521 public key on a restricted stocker-mobile SSH account. Only local VNC port 5901 is permitted; shell execution and other forwards were verified denied. Temporary verification credentials were removed.
