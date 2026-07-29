# Causal state-pattern discovery and temporal qualification V1

Date: 2026-07-11

Scientific status: post-inspection 2024 development discovery and temporal qualification. This is not prospective validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no direction, signed-return, P&L, costs, broker, order, position, or deployment path was used
- direct volume label: `historical_volume_not_used`
- upstream caveat: the already-frozen state detector used fourteen causal emissions, including two provider `historical_volume` transforms; this experiment read no raw volume field and makes no volume-confirmation claim

## Question

Test two new, bounded state-pattern families without selecting them from their price outcomes:

1. exact directed closed state paths of two to four transitions;
2. one-way transitions from a below-zero frozen bar-range centroid state to an above-zero centroid state, without requiring a return.

The test asks both whether the pattern occurrence is causally forecastable and whether a realized pattern has elevated absolute return and future range beyond current-state and causal entry context.

## Temporal protocol

The semantic contract was frozen first. Candidate discovery then read only January-June 2024 dates, symbols, state history, and four future state destinations. It did not read a price or movement outcome. The resulting candidate manifest and runner were hash-locked before movement outcomes were opened.

July-December 2024 was then scored with six expanding monthly folds. Every fold trained only on earlier 2024 months. Because 2024 is already an opened development year, this separation prevents outcome-based candidate selection but does not turn the result into untouched validation.

Structural occurrence probabilities used an order-3 destination kernel
`P(next_state | previous_state_2, previous_state_1, current_state)`, shrunk to a first-order current-state transition baseline. Multi-step loop probabilities were causal products with the state-history token updated at each destination.

Conditional movement models used:

- `qcontext`: current-state one-hot plus nine causal entry controls;
- `qcandidate`: the same context plus exact candidate identity.

The targets were frozen 2024 P75 and P90 exceedance probabilities for 6-, 12-, and 24-bar absolute return and future range. A good horizon required both P75 targets to pass support, structural reliability, proper loss, quarter, stock-deletion, five-session bootstrap, calibration, lift, and familywise Holm gates. High additionally required both P90 targets.

## Outcome-blind discovery

The state-only catalog contained 949 observed exact paths or allowed directed transitions. Fifty-six met their predeclared January-June recurrence/support filters. The deterministic caps selected 24 patterns:

| Group | Selected | July-December support pass | Structural pass | Both support and structural |
|---|---:|---:|---:|---:|
| Novel exact closed loops | 9 | 5 | 6 | 4 |
| Existing exact-loop controls | 8 | 8 | 7 | 7 |
| Directed low-to-high excursions | 7 | 7 | 1 | 1 |

The nine novel closed paths were:

- `4→5→4`
- `1→2→0→1`
- `1→4→2→1`
- `4→2→1→4`
- `4→2→3→4`
- `4→6→3→4`
- `1→2→1→0→1`
- `1→3→1→0→1`
- `4→6→4→6→4`

Thus the supported new recurrence dictionary was strongly entry-regime-specific: every selected novel loop began in state 1 or state 4. The seven directed excursions were `0→4`, `1→4`, `2→4`, `2→5`, `2→6`, `3→4`, and `3→6`.

## Frozen decision

No candidate-horizon qualified as development good or high.

- 72/72 candidate-horizon grades were `development_unqualified`.
- 20/24 patterns passed later-period support.
- 14/24 passed structural occurrence reliability.
- 18 conditional proper-loss tests survived their Holm families.
- Only 2/288 target/horizon/tier cells passed every non-Holm quality gate.
- No horizon passed both absolute-return and future-range requirements.
- The development survivor set is empty for both families.

No threshold was relaxed and no same-experiment second filter was run.

## What was informative despite the rejection

### Novel loop `4→5→4`

This was the strongest post-inspection lead, but it remains unqualified.

- January-June discovery occurrences: 236 across 22 stocks.
- July-December occurrences: 234 across 22 stocks, split 116 in Q3 and 118 in Q4.
- The history occurrence model improved log loss by 14.05% versus first order and passed structural calibration.
- P75 observed rates ranged from 50.85% to 63.68% across the six target/horizon cells.
- `qcandidate` conditional log-loss improvements were 6.98%-11.76%; joint improvements were 2.13%-2.82%.
- Five of its six P75 Holm tests passed; all P75 robustness, quarter, stock-deletion, lift, and relative-calibration comparisons passed.
- It failed the predeclared absolute supported-bin calibration-error ceiling in every P75 cell. Candidate maximum bin errors were 8.26%-23.99% against an 8% ceiling.

The important distinction is that its ranking/discrimination evidence was strong while its raw probability levels were not reliable enough. It cannot be labelled good or high. It is a legitimate post-inspection candidate for a separately frozen calibration experiment or unseen future test.

### Directed excursion `3→6`

This was the only low-to-high excursion that passed the structural gate.

- July-December occurrences: 786 across all 22 stocks.
- Structural log-loss improvement: 6.83% versus first order.
- All six P75 conditional movement losses improved for each target/horizon, generally by 4.32%-5.85%, with joint improvements of 1.96%-2.80%.
- Future range at 12 bars passed every non-Holm gate: observed P75 rate 36.77%, `qcontext` 25.25%, `qcandidate` 34.52%, conditional log-loss improvement 5.13%, joint improvement 2.79%.
- Its Holm-adjusted value was 0.126, so that cell did not survive the familywise gate; the other cells also failed at least bootstrap stability or calibration.

This is weaker than `4→5→4` as a complete movement candidate and does not qualify.

### What failed structurally

Only one of seven one-way low-to-high transitions passed the complete structural reliability gate, despite all seven having ample support. Exact closed recurrences were materially more forecastable: seven of eight existing controls and six of nine novel loops passed structurally. This supports retaining history-conditioned closed-loop modelling and rejects the broader idea that any low-to-high state jump is predictably timed.

## Interpretation

Yes, loop identity is regime-specific in this detector. The state-only discovery did not produce a generic loop collection spread evenly across entry states; the new supported recurrences clustered in states 1 and 4, and the clearest movement lead was the moderate/high-activity `4→5→4` return.

The stricter conclusion is not that a high-performance loop has been found. The correct conclusion is:

- exact recurrence occurrence can often be forecast from state history;
- most broad low-to-high transitions cannot be forecast reliably enough;
- one new exact recurrence, `4→5→4`, carries strong later movement information but its raw probabilities are miscalibrated;
- no tested loop is currently certified good/high, prospective, economic, or tradable.

Further narrowing inside these already-opened July-December outcomes would be outcome-driven selection. The clean next action is to freeze `4→5→4` as a post-inspection calibration candidate (and optionally `3→6` as a secondary excursion candidate), predeclare the calibration method, and evaluate it only on genuinely unseen post-freeze sessions. A development-only alternative is nested causal calibration using earlier 2024 folds, but it cannot validate the candidate.

## Integrity and reproducibility

- Contract: `work/contracts/20260711-causal-state-pattern-discovery-v1.json`
  - SHA-256 `cb3c217da9bcbac1606ca0ef69b13bad16ae54307084c839b092edba4f7d5759`
- Runner: `work/run_causal_state_pattern_discovery_v1.py`
  - SHA-256 `69bd696c13a8ae52c49d371f4849902e6a1c3ef285ffe5de42e5203f5a3ce3a1`
- Auditor: `work/audit_causal_state_pattern_discovery_v1.py`
  - SHA-256 `549f6a3ca82af277db5cb6f42faa6a74bb7f8a057f7f75e0be5dd5ee6f020144`
- Candidate manifest SHA-256: `e82b354cad060e272465661e657c300714be8fbad85b8ba9944a63153cd13b3e`
- Artifact root: `/private/tmp/stocker_causal_state_pattern_discovery_v1_20260711`
- Independent audit: 22/22 checks passed.
  - It independently rebuilt the outcome-blind manifest, exact labels, overlap weights, structural probabilities, all movement coefficients and probabilities, all 288 metric cells, Holm corrections, and the zero-candidate stop.
  - Maximum movement coefficient and prediction replay error: `0.0`.
- Tests: 220 passed.

Artifacts under `/private/tmp` are ephemeral and should be archived before a reboot if recomputation is not desired.
