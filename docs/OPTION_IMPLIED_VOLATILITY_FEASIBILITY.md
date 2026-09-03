# IBKR historical option-implied-volatility feasibility

## Decision

IBKR historical `OPTION_IMPLIED_VOLATILITY` is technically usable after the current-session
activity shortlist is known, but it is **not behaviorally interchangeable** with Session HARD's
frozen prior-close ATM call/put tick-13 model-IV source at the existing PRE_MOVE threshold.

No production strategy or runtime behavior was changed by this investigation.

## Source contract

IBKR's historical-data API accepts an end time, duration, bar size, `whatToShow`, and RTH flag.
Its historical-data matrix exposes `OPTION_IMPLIED_VOLATILITY` for underlying products such as
stocks and ETFs, but not as contract-specific historical model-IV bars for options:

- [IBKR historical bar data](https://interactivebrokers.github.io/tws-api/historical_bars.html)
- [IBKR historical-bar request](https://ibkrcampus.com/docs/tws-api/doc/market-data-historical/historical-bars/requesting-historical-bars)
- [IBKR tick types](https://interactivebrokers.github.io/tws-api/tick_types.html)

Stocker's existing Session HARD lineage is different. It selects a prior-session ATM call/put pair,
captures each contract's tick-13 `modelGreeks.impliedVol`, averages the two values, and persists the
result. Generic tick 106 is explicitly excluded. See `docs/M_PRE_MOVE_AUDIT.md` and
`packages/stocker_execution/src/stocker_execution/pre_context.py`.

## Method

The live probes used the deployed PAPER IBKR connection boundary with a separate, read-only client
ID. They submitted no orders and wrote nothing to Stocker's runtime database.

Each request used:

```text
bar_size = 5 mins
duration = 1 D
whatToShow = OPTION_IMPLIED_VOLATILITY
useRTH = true
end_time = exact prior exchange-session close
```

The existing US Session HARD conversion was retained for comparison:

```text
EXPECTED_ABSOLUTE_RETURN_15M = IV * sqrt(15 / (252 * 390)) * sqrt(2 / pi)
M_PRICE = P0 * EXPECTED_ABSOLUTE_RETURN_15M
PRE_MOVE_M = RAW_PRE_MOVE_PRICE / M_PRICE
qualified = PRE_MOVE_M > 0.475764059845861
```

The accepted source ledger was verified before use:

```text
rows: 1,073
valid PRE_MOVE rows: 1,071
unique symbol/session identities: 760
symbols: 243
SHA-256: d253a516be7f3dc1e65a6048679d7d711650706c0df846ef1f6ef29255ecd42e
```

That hash matches the frozen lineage recorded in `docs/M_PRE_MOVE_AUDIT.md`.

## Empirical results

### Current operational availability

AAL was selected by the real `US_ALL + MID` activity shortlist on 2026-09-03. A request made after
selection returned all 78 five-minute RTH bars for 2026-09-02. The final 19:55 UTC bar closed at
`0.41591211`. This proves that the prior day's underlying-level IV can be downloaded after today's
shortlist is known for at least this entitled US stock.

### Frozen representative rows

The 20 representative ATM-IV rows already published in `docs/M_PRE_MOVE_AUDIT.md` were replayed
against IBKR historical `OPTION_IMPLIED_VOLATILITY` for their exact prior sessions.

| Measure | Result |
|---|---:|
| Historical IV available | 20/20 |
| Pearson correlation with reference ATM-pair IV | 0.893605 |
| Median historical/reference IV ratio | 0.986011 |
| Mean historical/reference IV ratio | 0.947025 |
| Median absolute relative IV error | 5.636% |
| Mean absolute relative IV error | 11.000% |
| Ratio range | 0.681089–1.270338 |
| Within 10% of reference IV | 13/20 |
| Within 20% of reference IV | 14/20 |
| Same PRE_MOVE qualification | 20/20 |

This sample supports broad directional similarity, but it was not designed around the decision
boundary.

### Threshold-local stress test

The second test selected the 20 unique symbol/session rows nearest the frozen PRE_MOVE threshold
from the accepted ledger. Historical IV remained available for every row, including the expected
42-bar US half-day and 78 bars for each full session.

| Row | Reference IV | Historical IV | Reference PRE_MOVE | Historical PRE_MOVE | Result |
|---|---:|---:|---:|---:|---|
| EXPE\|2025-04-11\|6 | 0.905800 | 0.768326 | 0.475491 | 0.560569 | N → Y |
| OKLO\|2025-02-07\|18 | 1.344050 | 1.249324 | 0.474799 | 0.510800 | N → Y |
| RGTI\|2025-07-07\|6 | 0.840600 | 0.879448 | 0.473837 | 0.452906 | N → N |
| OTIS\|2025-02-07\|34 | 0.156250 | 0.174620 | 0.477707 | 0.427453 | Y → N |
| PRU\|2025-02-12\|12 | 0.232450 | 0.214306 | 0.473141 | 0.513199 | N → Y |
| APLD\|2024-04-12\|6 | 1.876400 | 1.195350 | 0.472602 | 0.741866 | N → Y |
| IREN\|2024-05-03\|6 | 1.273500 | 1.177888 | 0.478926 | 0.517802 | Y → Y |
| CNP\|2025-04-10\|22 | 0.291950 | 0.273042 | 0.480884 | 0.514186 | Y → Y |
| OKLO\|2025-01-28\|12 | 1.731800 | 1.369970 | 0.468918 | 0.592766 | N → Y |
| MRVL\|2025-02-27\|6 | 1.060500 | 0.728640 | 0.468520 | 0.681907 | N → Y |
| WULF\|2024-07-19\|6 | 1.345450 | 1.058830 | 0.468332 | 0.595107 | N → Y |
| CIFR\|2025-01-03\|6 | 1.079650 | 1.049305 | 0.467951 | 0.481484 | N → Y |
| PLTR\|2025-02-12\|6 | 0.624200 | 0.620693 | 0.466055 | 0.468688 | N → N |
| OKLO\|2025-04-04\|6 | 1.189500 | 1.066767 | 0.485588 | 0.541456 | Y → Y |
| APLD\|2024-02-29\|6 | 1.132600 | 1.095341 | 0.465710 | 0.481552 | N → Y |
| HRL\|2025-02-25\|8 | 0.424550 | 0.296853 | 0.486837 | 0.696259 | Y → Y |
| AXON\|2025-05-08\|6 | 1.008500 | 0.630218 | 0.463322 | 0.741427 | N → Y |
| NBIS\|2025-02-07\|6 | 0.956950 | 1.017556 | 0.459178 | 0.431829 | N → N |
| LUV\|2025-01-03\|6 | 0.348650 | 0.363526 | 0.493865 | 0.473655 | Y → N |
| MARA\|2024-02-12\|6 | 1.500450 | 1.320759 | 0.457369 | 0.519595 | N → Y |

Summary:

- Same qualification: 7/20.
- Qualification flips: 13/20.
- Of those flips, 11 changed from reject to qualify and 2 from qualify to reject.

## Conclusion and remaining validation

The proposed source solves the dynamic-shortlist acquisition problem: it can be requested after the
stocks are known and does not need a prior-close option-chain capture. It also broadly tracks the
previous ATM-pair IV in the representative sample.

It does **not** preserve the existing strategy at its decision boundary. A 65% flip rate in the
threshold-nearest sample is concrete evidence that substituting it under the existing
`SESSION_HARD_STRUCTURE_D_V1` identity would change behavior.

Any production use therefore needs a distinct strategy/calculation version and separate validation.
A full ledger comparison would require 760 paced historical requests; that was intentionally not
performed in this quick probe. International entitlement, coverage, session-minute scaling, and a
new threshold would also require explicit validation. No calibration or production implementation
is justified by this feasibility result alone.
