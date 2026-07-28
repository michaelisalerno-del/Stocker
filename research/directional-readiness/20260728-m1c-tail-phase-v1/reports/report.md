# M1C Tail Phase V1

Decision: `tail_phase_structural_separation_observed`.

This fixed retrospective study used the unchanged frozen M1C probability threshold `0.488333710794033` and a 15-minute stock-local causal movement-consumed split frozen from 2024 predictors at `1.3986941389121161` (n=18596). No 2024 outcome selected that split.

## Structural predictability

Assessment checkpoint composition: FIRST_ENTRY 371 (41.3%), PERSISTENT 462 (51.4%), RE_ENTRY 65 (7.2%).

Stress checkpoint composition: FIRST_ENTRY 455 (39.7%), PERSISTENT 594 (51.8%), RE_ENTRY 97 (8.5%).

Assessment fresh-episode composition: FIRST_ENTRY 371 (89.0%), RE_ENTRY 46 (11.0%). Persistence rows are not treated as independent episode support.

Stress fresh-episode composition: FIRST_ENTRY 455 (86.7%), RE_ENTRY 70 (13.3%).

The phase mix was similar across assessment and stress: persistence was about half of checkpoint observations, first entries about two-fifths, and re-entries the smaller remainder.

## Absolute-movement predictability

FIRST_ENTRY minus PERSISTENT future absolute 10-minute movement:

- Assessment: 0.000089 (95% session-cluster CI [-0.001463, 0.001472], n=371 vs 462, descriptive_support_available).
- Stress: 0.000144 (95% session-cluster CI [-0.001872, 0.002251], n=455 vs 594, descriptive_support_available).

FIRST_ENTRY minus RE_ENTRY future absolute 15-minute movement:

- Assessment: 0.004496 (95% session-cluster CI [0.002489, 0.006663], n=371 vs 46, descriptive_support_available).
- Stress: 0.003232 (95% session-cluster CI [0.000787, 0.006003], n=455 vs 70, descriptive_support_available).

The FIRST_ENTRY-versus-RE_ENTRY 15-minute difference kept the same positive sign in 52/52 leave-one-month and leave-one-stock estimates. Re-entry support was nevertheless concentrated by checkpoint (maximum checkpoint share 0.6522), so this is structural timing evidence rather than a direction or trading claim.

LOW_OR_EQUAL minus HIGH movement-consumed future absolute 10-minute movement:

- Assessment: -0.003231 (95% session-cluster CI [NA, NA], n=8 vs 409, blocked_insufficient_support).
- Stress: -0.004240 (95% session-cluster CI [NA, NA], n=9 vs 516, blocked_insufficient_support).

## Timing and remaining movement

`post_share_of_local_range_v1` uses a non-overlapping pre-trigger 15-minute range and future 10-minute range. It is descriptive, bounded, and is not an option-profitability measure. Session-cluster intervals, leave-one-month-out, leave-one-stock-out, and month/stock/checkpoint concentration tables are included as machine-readable artifacts.

FIRST_ENTRY minus RE_ENTRY `post_share_of_local_range_v1`:

- Assessment: 0.017765 (95% session-cluster CI [-0.012736, 0.058745], n=371 vs 46, descriptive_support_available).
- Stress: 0.009313 (95% session-cluster CI [-0.020808, 0.041500], n=455 vs 70, descriptive_support_available). Both intervals span zero, so V1 did not establish a phase difference in this bounded share diagnostic.

The frozen consumed split was highly imbalanced inside later high-M1C episodes (assessment n=8 LOW_OR_EQUAL versus 409 HIGH; stress n=9 versus 516), so the consumed-bucket remaining-movement comparison is `blocked_insufficient_support` rather than negative evidence.

## Directional predictability

Direction remains secondary. Frozen A1 was applied unchanged; no A1 threshold or coefficient was fitted or selected here.

- FIRST_ENTRY: actions 118, accuracy 0.5254, mean aligned 10-minute return 0.001657, `descriptive_support_available`.
- PERSISTENT: actions 180, accuracy 0.4611, mean aligned 10-minute return -0.002388, `descriptive_support_available`.
- RE_ENTRY: actions 26, accuracy 0.4615, mean aligned 10-minute return 0.001075, `blocked_insufficient_support`.

Assessment continuation 0.4972 and reversal 0.5028; Stress continuation 0.4626 and reversal 0.5374. The near-even continuation/reversal mix is consistent with, but does not prove, a phase-mixture explanation for weak direction.

Any apparent phase-specific A1 difference is retrospective and exploratory. It is not evidence of a directional edge and does not define a combined A1-plus-phase rule.

## Option profitability

Not tested. Previous-close IV is used only as the canonical movement scale. No option P&L, bid/ask fill, contract-selection return, or tradeability claim is produced.

## Execution realism

This is five-minute underlying-bar research. It cannot answer spread, queue, fill probability, slippage, or trade-impact questions. Prospective Tail Phase fields are logging-only and do not alter recorder priority, promotion, subscriptions, direction, contracts, capacity, or episode inclusion.

## Operational blockers and scope limits

- The canonical historical causal checkpoint surface had 52952 development and 33933 assessment rows without exact previous-close option context. M1C is undefined for those rows, so they were not scored or silently labelled outside-tail; their exclusion is reported in source coverage and missingness artifacts. The observed structural result is conditional on the valid frozen-M1C surface.
- The external Group-O package producer is outside this repository. It must supply and receipt-hash the canonical previous-close implied 15-minute movement; until the engineering-transfer checks verify that handoff, prospective consumed buckets may correctly remain `UNKNOWN_INCOMPLETE` without interrupting M1C recording.
- Sector context was not present causally in the existing historical surface, so it is explicitly out of scope; no external data was acquired.
- No existing frozen market-volatility state was present on this surface.
- Re-entry or interaction cells below 30 rows or 10 sessions are labelled `blocked_insufficient_support`; thresholds were not relaxed.
- The first 20 IBKR/EODHD transfer sessions remain `engineering_transfer` and may verify logging mechanics only.

## Answers to the ten preregistered questions

1. The exact phase composition is reported above and in `structural_counts_v1.csv`.
2. Assessment and stress use identical definitions; their observed composition is reported separately without post-hoc subgroup selection.
3. FIRST_ENTRY comparisons against PERSISTENT and RE_ENTRY are reported with session-cluster support and intervals above.
4. The fixed LOW_OR_EQUAL versus HIGH comparison reports whether greater pre-trigger consumption corresponds to less remaining movement.
5. Breadth is reported through leave-one-month, leave-one-stock, and concentration artifacts; outliers were retained.
6. Frozen A1 action counts, accuracy, and aligned returns are reported by phase and consumed bucket.
7. No apparent A1 improvement is confirmatory in this retrospective V1; small interactions are explicitly blocked.
8. Continuation and reversal rates are descriptive checks for a mixture explanation, not a directional claim.
9. Phase may be retained as a preregistered, logging-only prospective interaction according to the summary recommendation; it is not a gate.
10. Prospective bid/ask, fills, and impact remain unknown.

Protected 2026 historical outcomes were not opened, calculated, displayed, or inspected. No order-routing path was enabled and no order was placed.
