# Regime Model Validity V2

`research_only=true` · `execution_enabled=false` · `order_placement=disabled` · `broker_connected=false` · `economic_outcomes_used=false` · `payoff_selection_used=false` · `production_runtime_modified=false` · `strategy_promotion=false`

## 1. Exact scope

Structural Part A audit only. No next-loop predictor, payoff selector, economic target,
trading runtime, or protected 2026 data was opened.

## 2. Source identity

Baseline `66cd706fa727ac5873b299d5c22388221203f451` on `agent/slrno-research-handoff`; 2024 snapshot
`48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661` and frozen state model
`909858ed7c9c02c1c113661202cb5d7c6bfabd243f1cc428b8a5fb1a3c022251`. The exact V2 implementation-source
bundle is hash-bound as `eb12d895260f52d0b8e83a7aa90f5f86d9d27612f9b853237d4db645a89c4ec9`; the opened
2024 and 2025 snapshots are jointly bound in every artifact identity.

## 3. Current state implementation

The active model is an eight-state diagonal-Gaussian causal semi-Markov filter over
14 combined stock/market emissions. It reconstructed 424583
2024 bars across 22 stocks. The current posterior and
legacy hard-state path reproduced. The active posterior source is therefore
reconstructible from the frozen inputs; the historical KMeans *refit* is not byte exact.

## 4. Mathematical audit

Forward recursion pass: `True`; posterior
normalization pass: `True`; state×age
normalization pass: `True`; hard MAP argmax
pass: `True`.

## 5. Causality audit

Completed-bar availability pass: `True`; session resets pass:
`True`; critical future leakage found:
`False`.

## 6. Duration and censoring status

Full 78-bar support is causal and normalized, but the frozen state-model fit treated
terminal training runs as exact exits before the V2 tail expansion. Parameter
censoring pass: `False`. This is the
local causal-model defect that prevents validation.

## 7. Offline-cleaning findings

The neighbor-aware historical cleanup relabelled 17.7699% of reconstructed
training bars and uses both neighbouring runs. It is preserved as offline fit lineage
and is not described as causal.

## 8. Raw versus cleaned labels

`CLEANING_0` preserves raw labels. `CLEANING_1` changed 17.7699%; the
past/current-only `CLEANING_CAUSAL` changed 37.1018%. State, loop,
and dictionary differences are frozen in the four cleaning artifacts.

## 9. Hard-state churn

One-bar reversal rate: 16.8080%; two-bar reversal
rate: 24.9763%; hard/hysteretic transition agreement:
81.4236%.

## 10. Posterior-confidence results

Low-margin transition share (<0.05): 5.0257%;
margin <0.02: 2.0112%; new-state posterior <0.50:
11.5079%. These are diagnostics,
not selected thresholds.

## 11. K sensitivity

The full K={6,8,10,12} surface contains 20 deterministic fits. No K was
selected. Occupancy, duration, likelihood, transition entropy, alignment, and loop
surfaces are in the five K/seed artifacts.

## 12. Seed sensitivity

At K=8 the minimum aligned NMI was 0.454147, below the
frozen 0.50 stability threshold. Structural excess survived the preregistered seed
count gate, but numeric state identity and event timing were seed-sensitive.

## 13. Training-sample sensitivity

All 4 samples used the same 200,000-row bound. Minimum aligned
bar agreement was 49.4490%;
minimum dictionary coverage ratio was
12.8826%; minimum
selected-event agreement was 1.0758%.

## 14. State alignment

Labels were aligned with Hungarian matching over centroid, transition, and duration
profiles rather than numeric IDs. K=8 was fully matched; K=6/10/12 retain explicit
unmatched-state counts in `state_stability_by_k_seed.csv`.

## 15. Semantic drift

Maximum 2024→2025 centroid drift was
0.22609. State 4 drift was
0.0511422, occupancy moved from
19.0747% to
22.2073%, and its transition-profile cosine was
0.998638. Period drift was modest, but
the all-component semantic gate failed because identity was not seed-independent.

## 16. Stock heterogeneity

Maximum single-stock share within any state was
20.2393%, below the frozen 25%
concentration ceiling. Detailed 2024/2025 shares are in
`state_stock_heterogeneity.csv`.

## 17. Clock heterogeneity

`state_clock_heterogeneity.csv` records opening, middle, and late shares for every
state and period. Clock concentration is descriptive; it was not used to select K,
features, or a loop threshold.

## 18. Combined versus stock-only representation

The likelihood levels below are not directly comparable across different emission
dimensions. Combined states had higher minimum occupancy and lower stock concentration
than stock-only states, so the combined representation was not structurally dominated
on those frozen stability diagnostics.

| representation | causal_negative_log_likelihood | minimum_state_occupancy | maximum_stock_share |
| --- | --- | --- | --- |
| MODEL_COMBINED | 13.3507 | 0.0453527 | 0.202393 |
| MODEL_STOCK_ONLY | 9.37078 | 0.0417963 | 0.201099 |
| MODEL_HIERARCHICAL | 11.8347 | 0.00618725 | 0.438767 |

## 19. Hierarchical market × stock representation

The 32-cell market×stock representation had minimum occupancy
0.6187%
and maximum stock share
43.8767%.
It did not meet the preregistered non-degeneracy or concentration gates and was not
preferred.

## 20. Hard, hysteretic, and soft loop robustness

Minimum selected-loop hard→hysteretic same-primitive agreement with bounded shifts was
85.7420%. Of
5247 independently attached soft-support rows,
54.9647% met the frozen robust-support band. Soft mass never created
a hard event.

## 21. Primitive-loop stability

Both selected primitives retained positive semi-Markov structural excess in at least
four of five K=8 seeds. Their aligned timestamps and identities nevertheless varied
materially across seeds and training samples, so structural excess alone does not
freeze a stable state language.

## 22. Dictionary stability

The training-sample coverage gate failed at
12.8826%. The
semantic dictionary must remain paused despite passing hysteretic and K=8 excess gates.

## 23. Failure cases

- Frozen training durations counted terminal runs as observed exits.
- The preserved decision exporter resets hysteresis at nominal sessions but not
  causal source-gap resets; this V2 surface applies the stricter reset explicitly.
- Three of five K=8 seeds fell below the frozen NMI threshold.
- Sample-conditioned selected-loop coverage and event agreement collapsed.
- The hierarchical alternative produced sparse, concentrated cells.

## 24. Missing evidence

The archived `run_sealed_2025_sec_raw_activity_validation.py` panel-base dependency is
absent. Consequently, the historical KMeans refit differs by up to
0.458333; the frozen current
posterior still reproduces exactly. No 2023 portability or protected 2026 data was
opened. Historical reports retain their original meaning, with duration conclusions
requiring the narrower terminal-censoring interpretation.

## 25. Part A scientific decision

**regime_representation_requires_targeted_repair**

The independent-audit status in this primary report is
`pending`; Part B remains closed regardless because this
decision is not in the authorized set.

## 26. Whether dictionary work may proceed

Dictionary work may proceed: `False`. Part B
interaction scoring authorized: `False`.

## 27. Exact next step

Implement an audit-only, right-censored state-duration refit with a fully archived
panel builder and deterministic training-row order, then rerun this unchanged Part A
contract. If the local repair passes, the seed and training-sample instability gates
must still be re-evaluated before dictionary or interaction work opens.
