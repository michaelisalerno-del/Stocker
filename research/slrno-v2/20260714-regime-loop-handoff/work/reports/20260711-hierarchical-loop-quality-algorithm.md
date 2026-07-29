# Hierarchical loop-quality algorithm — development/portability result

Status: **complete; independent pre-score and post-score audits passed**  
Scientific status: research on already opened 2024 development, 2025 development, and backward-2023 portability evidence; not prospective validation

`research_only: true`  
`live_ordering_enabled: false`  
`order_placement: disabled`

## Result

The causal hierarchical algorithm found a real pooled improvement over both the frozen state/context baseline and the V3 route-topology representation. It did **not** pass the frozen reliability contract. The provisional 2024 label is therefore:

> `development_algorithm_unconfirmed`

The primary failures were strict cell calibration and supported causal-slice sign reversals. The joint comparison against route topology also missed its Brier uncertainty gate. No loop or loop-horizon received a good/high development-candidate label.

This does not change any frozen parent loop grade and does not support a direction, signed-return, economic-edge, tradability, or deployment claim.

## Frozen lineage

- Contract SHA-256: `f6956b6ab0495a49669f714df834d1fd0fdaa13b0ecf4b123d6c54c0fc9b5936`
- Runner SHA-256: `860cffb46e2537bf8126d3c5852cab508f88d0a4ac82f06fecf90cabfa86beba`
- Fit-complete SHA-256: `060d5ec5d9c05c2291f40d10a83104382b38cb6c5933f5d5f503cdd4aaab70e5`
- Passing pre-score audit SHA-256: `36b48db2fdc47dabf15b487ec9798e20eccb2c511dc34c68cda392d276fd50a7`
- Scoring-complete SHA-256: `a30990657f08472fe7cc00c78fd43f617f4e37f9f2f0e2dc419b844083f4ef65`
- Summary SHA-256: `c8c10e454062e303e6b44179dad5df0d44dd2c16392ee338f80fa3e5d8159187`
- Independent auditor SHA-256: `e3b20ee2c74887f81c5bd720460220fa57dbd64d43e7a5a9a9e7e0f4fd929ff2`
- Passing post-score audit SHA-256: `dba20b5692cca01eb70b60a1c0cb44230af0c0565ba54d035e8d61bb90b3c755`
- Fit period: 2024 only
- Causal OOF months: July–December 2024
- OOF compatible rows: 216,438
- OOF realized rows: 15,584
- Unique effective inverse-overlap weight: 14,167
- Frozen cycles: 20
- Compatible cycle-current-state units: 44
- No 2025, 2023, 2026, prospective-shadow, or live path was read by the fit phase.

## Algorithm fitted

The model keeps the exact V3 causal context and semantic route-topology blocks, then adds two jointly regularized hierarchical blocks:

1. a 20-column cycle block, centered by its conditional-weighted training-fold frequency;
2. a 44-column compatible cycle-current-state block, centered **within cycle**, with all route coordinates outside the row's cycle fixed to zero.

One scale pair was selected across all six ordered movement models from the frozen 15-pair grid. Scale selection was nested and causal: each validation month used only strictly earlier 2024 rows. The zero endpoint was the exact sealed V3 route-topology model.

The full-2024 selected pair was:

- `a_cycle = 1.0`
- `a_route = 0.125`
- grid index `11`

Outer-fold selections varied:

| Outer month | a_cycle | a_route |
|---|---:|---:|
| 2024-07 | 0.5 | 0.125 |
| 2024-08 | 0.5 | 0.125 |
| 2024-09 | 1.0 | 0.25 |
| 2024-10 | 0.5 | 0.25 |
| 2024-11 | 0.5 | 0.5 |
| 2024-12 | 0.5 | 0.25 |

This variation is descriptive evidence that the preferred degree of cycle/orientation detail was not stable across outer folds.

## Pooled 2024 OOF results

Positive relative log-loss improvement means qhier was better.

| Baseline | Surface | Relative log-loss improvement | Brier difference | LL 99.375% upper | Brier 99.375% upper | Formal pass |
|---|---|---:|---:|---:|---:|---|
| qcontext | Conditional | 4.0505% | -0.00322679 | -0.0103633 | -0.00240807 | No |
| qcontext | Joint | 1.8542% | -0.00005295 | -0.00065834 | -0.00003534 | No |
| qroute_topology | Conditional | 0.3764% | -0.00033931 | -0.00041525 | -0.00014667 | No |
| qroute_topology | Joint | 0.2164% | -0.00000203 | -0.00005566 | +0.00000144 | No |

The secondary qfull non-inferiority gate passed on conditional/joint log loss and Brier. This means the hierarchy retained the pooled sealed full-model signal within the frozen margins. Secondary non-inferiority cannot rescue a failed primary reliability gate.

## Why the primary gate failed

### Calibration

All 12 target × horizon × tier cells failed the complete calibration requirement on each surface.

- Conditional maximum supported-bin error reached `0.18082`, versus the frozen absolute limit of `0.02`.
- Joint maximum supported-bin error reached `0.09550`, versus the frozen absolute limit of `0.01`.
- Conditional qhier ECE was worse than qcontext in 3/12 cells and worse than route topology in 6/12.
- Joint qhier ECE was worse than qcontext in 5/12 cells and worse than route topology in 12/12.

### Supported causal-slice reversals

The contract required no positive qhier-minus-baseline log-loss or Brier difference in any supported causal slice.

| Baseline | Surface | Supported slices | Sign reversals |
|---|---|---:|---:|
| qcontext | Conditional | 43 | 8 |
| qcontext | Joint | 43 | 11 |
| qroute_topology | Conditional | 43 | 15 |
| qroute_topology | Joint | 43 | 21 |

Most reversals occurred in the cycle-current-state family. The pooled improvement therefore did not transfer uniformly across the orientations that the named loop-quality claim requires.

### Joint Brier uncertainty versus topology

The pooled joint Brier difference versus route topology was slightly favorable (`-0.00000203`), but its familywise-adjusted one-sided upper bound was positive (`+0.00000144`). That endpoint was not confirmed.

## Falsification

The 999-draw, anchor-vector-preserving stratified falsification passed all four predeclared statistics:

| Comparison | Surface | Observed relative LL improvement | Empirical p-value |
|---|---|---:|---:|
| qhier vs qcontext | Conditional | 4.0505% | 0.001 |
| qhier vs qcontext | Joint | 1.8542% | 0.001 |
| qhier vs qroute_topology | Conditional | 0.3764% | 0.001 |
| qhier vs qroute_topology | Joint | 0.2164% | 0.001 |

The independently linearized falsification statistic replayed the direct proper-loss calculation with maximum error `1.08e-15`.

This supports that the pooled prediction/outcome alignment is not explained by the frozen null transformations. It does not overcome the calibration and slice-robustness failures.

## Named loop candidates

- 60 cycle-horizon development units reported: all `development_unqualified`.
- 20 global cycle labels: all `development_unqualified`.
- Development-good candidates: 0.
- Development-high candidates: 0.
- Frozen parent grades changed: no.

Because the global primary algorithm precondition failed, the named-component bootstrap/Holm families were not run; their labels failed closed before multiplicity testing, as predeclared.

The result therefore does not establish that any loop is of a reliable good or high-performance type. “Performance” here remains movement-quality reliability only, never trading performance.

## 2025 development and backward-2023 portability

The independent pre-score audit passed 45/45 checks with maximum OOF probability replay error `0.0`, then authorized the already-opened later-period scoring. Neither period was treated as prospective validation.

Pooled relative log-loss improvements remained positive:

| Period | Baseline | Conditional | Joint |
|---|---|---:|---:|
| 2025 | qcontext | 3.7079% | 1.6710% |
| 2025 | qroute_topology | 0.2738% | 0.1482% |
| backward-2023 | qcontext | 4.2727% | 1.5459% |
| backward-2023 | qroute_topology | 0.3091% | 0.2271% |

The same reliability failures persisted.

| Period | Surface | Failed calibration cells | Maximum supported-bin error |
|---|---|---:|---:|
| 2025 | Conditional | 12/12 | 0.18569 |
| 2025 | Joint | 12/12 | 0.06226 |
| backward-2023 | Conditional | 12/12 | 0.17080 |
| backward-2023 | Joint | 12/12 | 0.06629 |

Supported-slice reversals were:

| Period | Baseline | Conditional | Joint |
|---|---|---:|---:|
| 2025 | qcontext | 5/38 | 6/38 |
| 2025 | qroute_topology | 11/38 | 11/38 |
| backward-2023 | qcontext | 8/43 | 14/43 |
| backward-2023 | qroute_topology | 13/43 | 18/43 |

All four falsification statistics passed at empirical `p = 0.001` in each later period. Secondary qfull non-inferiority also passed in both periods. These pooled confirmations cannot override the failed primary calibration and slice gates.

Exact demotion-only transfer:

- 2025: `development_algorithm_unconfirmed`
- backward-2023: `development_algorithm_unconfirmed`
- `algorithm_development_portable: false`
- portable cycle-horizon units: 0/60
- portable global cycles: 0/20
- later promotion performed: false
- parent grade changed: false

## Interpretation

The hierarchy adds measurable information beyond semantic topology in pooled 2024 OOF loss, and the selected full model uses substantially more cycle identity than route-orientation deviation. However:

- the selected scales varied across outer months;
- calibration failed broadly;
- several supported orientations became worse than both required baselines;
- no named cycle survived the complete contract.

The appropriate conclusion is that cycle/orientation detail contains a small additional movement signal, but the current hierarchical logistic algorithm does not make that signal reliable enough to label loops as good or high quality.

## Audit and next state

The fit bundle is frozen under:

`/private/tmp/stocker_hierarchical_loop_quality_algorithm_v1_20260711`

The independent pre-score replay passed 45/45 checks and bound the exact fit-complete hash, runner hash, contract hash, and complete fit artifact manifest before later rows were opened. Production scoring then completed and exited. All 16 scoring artifacts and all 20 frozen fit artifacts replay their recorded hashes exactly after completion.

The independent post-score replay passed 40/40 checks. Maximum independently reconstructed prediction error was `0.0` in both 2025 and 2023. Metrics, calibration, supported slices, falsification, support, named Holm labels, transfer labels, completion markers, and the final summary all replayed exactly.

The first post-score audit attempt failed closed on an auditor-only numerical assertion: a raw softmax tier sum could exceed one by one machine epsilon, while the frozen production validity rule allows `1e-12`. The assertion was aligned to that already-frozen bound without clipping or changing any probability, model, scientific gate, or tolerance; a one-ULP regression test and independent code review passed before the complete audit was rerun. The final auditor test file passed 21/21 tests, and the full `work/tests` suite passed 126/126.

Production results are frozen; scoring cannot be rerun because the scoring namespace is no longer pristine.

The artifact root is under `/private/tmp` and should be archived before reboot if exact replay without recomputation is required.
