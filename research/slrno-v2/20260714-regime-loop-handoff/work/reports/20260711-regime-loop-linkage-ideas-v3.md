# Regime-to-loop movement linkage ideas V3

Date: 2026-07-11

Decision: `linkage_idea_rejected_or_unconfirmed`

Scientific status: post-inspection 2024 causal-OOF development test. This is not prospective validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no direction, signed return, P&L, trading rule, cost/exit model, broker, order, position, deployment, or strategy-promotion path was used
- direct volume label: `historical_volume_not_used`

## Question

Can the retained regime/history loop-occurrence forecast and a loop-conditioned movement forecast be linked into a reliable probability that:

1. a compatible state loop occurs; and
2. its subsequent absolute return or future range exceeds a frozen 2024 P75 or P90 threshold at 6, 12, or 24 five-minute bars?

The baseline was the independence product:

`qhistory × qcontext_movement`.

Five predeclared candidate linkage ideas were tested:

1. `minimal_time_topology`: limited B0/stress/session-clock occurrence probability × semantic route-topology movement probability;
2. `raw_full_link`: full nine-factor occurrence probability × hierarchical loop-quality probability;
3. `partial_full_link`: a fixed 50% log-odds shrinkage of both full heads back toward their retained baselines, followed by multiplication;
4. `calibrated_raw_product`: a causal expanding-month logistic recalibration of the raw full product;
5. `dependency_stack`: a causal expanding-month model using occurrence baseline/residual, quality baseline/residual, and their interaction.

Two attribution diagnostics were also calculated:

- `occurrence_only`: full nine-factor occurrence × context movement;
- `topology_only`: retained history occurrence × route-topology movement.

## Data and causal split

Two already-audited July-December 2024 OOF ledgers were joined one-to-one by stock, session, run-entry timestamp, and cycle:

- factor-conditioned loop occurrence: 361,220 compatible rows;
- hierarchical movement quality: 216,438 exact-price-cohort compatible rows.

All 216,438 quality rows joined exactly. Cycle, state, current state, loop label, and month matched with zero discrepancies. The primary September-December linkage evaluation contained:

- 130,672 compatible cycle rows;
- 21,341 state-run-entry anchors;
- 20 cycles;
- 8 current states;
- 22 stocks;
- 84 sessions.

The learned meta-models used strictly earlier OOF months. September trained on July-August, October on July-September, and so on. Ninety-six binary fits were made: two meta-models × twelve target/horizon/tier cells × four validation months.

The occurrence ledger's fold-local `qhistory` was used for every composition. The quality ledger's parent `loop_probability` was excluded from every link because it is a differently constructed anchor-panel score. Their diagnostic correlation was 0.98907, mean absolute difference 0.003654, and maximum difference 0.196915; treating them as identical would have been incorrect.

## Pooled result

Positive improvement means lower joint-event log loss versus `qhistory × qcontext`.

| Variant | Equal-cell log-loss improvement | Brier difference | 95% session-block LL interval | Formal pass |
|---|---:|---:|---:|---|
| Occurrence only, diagnostic | 3.9339% | -0.00101850 | [-0.005955, -0.002217] | Not selectable |
| Topology only, diagnostic | 1.0122% | -0.00010626 | [-0.001207, -0.000602] | Not selectable |
| Minimal time × topology | 3.5502% | -0.00082876 | [-0.005354, -0.001965] | No |
| Raw full product | **4.8100%** | **-0.00112002** | [-0.006855, -0.002863] | No |
| Fixed half-shrunk product | 3.2832% | -0.00070894 | [-0.004646, -0.002131] | No |
| Causally calibrated raw product | 4.4419% | -0.00103663 | [-0.006661, -0.002609] | No |
| Learned dependency stack | 3.4800% | -0.00085791 | [-0.005953, -0.001829] | No |

Every candidate variant:

- improved log loss and Brier in all 12 target/horizon/tier cells;
- improved in every evaluation month;
- improved under every leave-one-stock-out deletion;
- passed both familywise Holm endpoints;
- had no top-three ranking degradation in any cell.

The association is therefore real at the pooled level. The rejection is about local probability reliability, not absence of information.

The attribution is also clear: factor-conditioned occurrence contributes most of the average gain. Occurrence-only linkage improved pooled log loss by 3.93%; semantic topology alone contributed 1.01%. Replacing context movement with the hierarchical quality head while retaining full occurrence lifted the raw product to 4.81%. The learned interaction did not improve on the direct product.

## Why every candidate failed

### Worst-bin calibration

Average ECE became small, but the predeclared absolute maximum supported-bin error was 0.02. Every candidate failed that ceiling in at least 11 of 12 cells.

| Variant | Maximum cell ECE | Maximum supported-bin error | Failed calibration cells |
|---|---:|---:|---:|
| Baseline | 0.013345 | 0.179855 | — |
| Minimal time × topology | 0.006451 | 0.118203 | 12/12 |
| Raw full product | **0.005459** | **0.088513** | 12/12 |
| Half-shrunk product | 0.009908 | 0.137948 | 12/12 |
| Calibrated raw product | 0.005467 | 0.113616 | 12/12 |
| Dependency stack | 0.009037 | 0.107622 | 11/12 |

The raw full product was both the best pooled model and the best candidate on maximum calibration error, but 8.85% remained far above the 2% ceiling. Global logistic recalibration reduced pooled loss less than the raw product and made the worst supported bin worse.

### Regime/loop orientation reversals

Thirty-two cycle-current-state orientations met the support rule. The contract required zero candidate-minus-baseline reversals.

| Variant | Log-loss reversals | Brier reversals |
|---|---:|---:|
| Minimal time × topology | 8/32 | 7/32 |
| Raw full product | 7/32 | 7/32 |
| Half-shrunk product | **4/32** | **2/32** |
| Calibrated raw product | 13/32 | 16/32 |
| Dependency stack | 17/32 | 23/32 |

Fixed half-shrinkage was the safest local model, reducing the raw product's reversals substantially. It did not eliminate them and sacrificed pooled improvement. The remaining half-shrunk log-loss reversals were small but predeclared as disqualifying; the worst relative degradation was 0.215%.

The learned dependency stack was not the solution. It was worse than both the raw product and its one-dimensional calibrator on pooled log loss and Brier, and it produced the most local Brier reversals.

## Regime × loop × time diagnostics

No time-attraction slice qualified because the global dependency stack failed. The following are post-inspection diagnostics only.

The clearest concentration was cycle 13, `5→7→5`, entered from state 5:

| Entry-time band | Compatible rows | P75 joint positives across six cells | LL improvement | Both halves and all six cells | Max bin error | Holm adjusted |
|---|---:|---:|---:|---|---:|---:|
| 0-100 minutes after open | 1,084 | 626 | 10.22% | Yes | 0.0628 | 0.0205 |
| 105-195 minutes after open | 840 | 602 | **16.56%** | Yes | 0.0644 | 0.0044 |
| 200-265 minutes after open | 567 | 214 | 9.62% | Yes | **0.0071** | 0.4070 |

“Joint positives across six cells” sums the two P75 targets across three horizons; it is not a count of unique trades or independent observations.

This is meaningful evidence for the user's intuition that a loop can be attracted to a specific regime and time band. It is not a qualified finding:

- the first two time bands failed calibration;
- the last band passed calibration but not multiplicity;
- the dependency stack failed globally;
- all periods are already-opened development evidence.

Cycle 09, `3→6→3`, entered from state 3, also showed 8.6%-9.0% diagnostic P75 linkage improvements in early and later clock bands, but failed all-cell and/or temporal-half stability and multiplicity requirements.

## Interpretation

The experiment supports a narrower version of the proposed linkage:

- yes, predicted loop attraction and movement-quality probabilities contain complementary joint-event information;
- most of the gain comes from improving **which loop is likely to occur now** using causal regime/time/price context;
- the simple direct product was better than a learned dependency interaction;
- fixed partial pooling reduced local harm but did not remove it;
- probability calibration and cycle-current-state heterogeneity remain the binding problems;
- no linkage model, time slice, or named loop is currently certified good/high or prospectively reliable.

The next model should not be a larger stack. The evidence favors a separately frozen orientation-aware shrinkage/calibration model:

1. retain the raw full product as the high-information reference;
2. estimate cycle-current-state-specific reliability with hierarchical backoff toward the retained baseline;
3. calibrate supported bins with partial pooling rather than one global logistic map;
4. predeclare cycle 13/state 5 time bands as post-inspection hypotheses, not discoveries;
5. make the final decision only on genuinely unseen post-freeze sessions.

No more shrinkage weights, clock cuts, or cycle selections should be tried on the same September-December results.

## Fail-closed execution record

V1 stopped before scoring because it incorrectly required the two parent structural probability columns to be identical. No V1 artifact root was created.

V2 explicitly selected factor `qhistory` and excluded the quality parent loop score, leaving all statistical rules unchanged. It completed the calculations in memory but stopped before artifact creation on an empty-qualified-slice pandas serialization error. No V2 artifact root was created and no scores were printed or inspected.

V3 changed only that empty-list serialization to an explicit list comprehension. It wrote the complete artifact bundle and manifest. Its final console pretty-print then encountered a NumPy-boolean serialization error after all files were safely written; the bundle was not rerun or altered. The independent auditor subsequently reproduced the complete result.

## Integrity and reproducibility

- V3 contract SHA-256: `88a60956857e6ccb4fb5e74beb9085e46765e55b31763b26927dc496822ce947`
- V3 runner SHA-256: `c0e8786670fd51e3d93290ecd56ba51322ebe6ace0fd7e521803f2fd8c1ce72e`
- Immutable V1 algorithm-body SHA-256: `e134f01f4d6da58581205fe8070f90a2f17d0fc0945dea0b42a2ca1c96bfa51a`
- V2 identity-adapter SHA-256: `b38a17b5e5023951e992004fac51e4c264af2c65e7f19c4b35ecea14cbd5e6ba`
- Independent auditor SHA-256: `40b8def60fc41910ced59427781a1a52315c130aac0c08e4749cc5f5e5ba0373`
- Independent audit-result SHA-256: `92343c008f0cd585c3e02c1e3c60905aa4f9cde7e9c9b99111076a2cd8be300f`
- Independent audit: 19/19 checks passed.
- All 96 meta-model parameters and all 130,672 primary predictions replayed with maximum error `0.0`.
- Cell, pooled, bootstrap, multiplicity, temporal, stock, orientation, ranking, and time-attraction metrics replayed within `1.12e-16`.
- Full workspace research suite: 240 tests passed.
- `git diff --check`: passed.
- Artifact root: `/private/tmp/stocker_regime_loop_linkage_ideas_v3_20260711`

The artifact root is ephemeral and should be archived before reboot if exact replay without recomputation is required.
