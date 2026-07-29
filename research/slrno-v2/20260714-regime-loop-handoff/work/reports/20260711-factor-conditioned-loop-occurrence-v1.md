# Factor-conditioned loop-occurrence likelihood — 2024 development result

Status: **complete; rejected by the frozen 2024 reliability contract; later periods remain sealed**

`research_only: true`  
`live_ordering_enabled: false`  
`order_placement: disabled`

## Decision

The experiment did test loop occurrence from both the retained state-history pattern and additional causal entry-time factors. The nine-factor head contains meaningful pooled 2024 development-period information, but it is not reliable in every supported loop/current-state orientation. The frozen decision is therefore:

> `factor_conditioned_loop_occurrence_rejected_2024_and_do_not_score_later_periods`

The model passed the pooled loss, ranking, calibration, moving-block bootstrap, cycle-count, and coherent falsification gates. It failed the predeclared orientation-stability gate because the full model was slightly worse than the limited four-factor model in four of 44 supported cycle/current-state orientations. The contract required zero reversals.

No 2025, backward-2023, partial-2026, or prospective-shadow path was resolved or read. No later scoring was authorized.

This result concerns the probability that a compatible fixed state loop occurs. It does not qualify a loop as good or high movement quality and does not support direction, signed return, profitability, P&L, economic edge, tradability, strategy, order, position, or deployment claims.

## Algorithm tested

Each state-run entry generated an overlapping binary target for every compatible member of the frozen 20-cycle dictionary. Several cycles can be positive at one anchor, so this is not a mutually exclusive 20-class classifier.

The retained causal last-three-state probability was used as a fixed log-odds offset. Nested ridge-logistic residual heads then tested three increasingly rich representations:

1. `qpattern`: the retained history probability plus the compatible cycle/current-state pattern;
2. `qlimited4`: pattern plus B0 numeric state, B0 stress, and entry-clock sine/cosine;
3. `qfull9`: the limited head plus current-bar log return, trailing six-bar return, trailing twelve-bar mean absolute return, session-to-entry return, and current range/open.

All factors were computed from bars at or before the run entry and were used once at the anchor. They were not repeated along hypothetical future paths. The folds, preprocessing, history kernels, factor quartiles, and ridge penalties were fitted causally from strictly earlier 2024 months. July through December 2024 were the outer out-of-fold evaluation months.

Provider volume was not used: `historical_volume_not_used`.

## Population

The complete 2024 reconstruction passed its frozen population checks:

| Cohort | Run-entry anchors | Compatible anchor-cycle rows | Positive loop labels |
| --- | ---: | ---: | ---: |
| Full 2024 reconstruction | 110,949 | 759,212 | 46,630 |
| July–December causal OOF evaluation | 54,186 | 361,220 | 22,262 |

Terminal run entries were retained with zero loop labels. The OOF evaluation covered 22 stocks, all eight states, all 20 cycles, and 128 session dates.

## Pooled result

Positive relative log-loss improvement means `qfull9` was better. Brier differences are candidate minus baseline, so negative is better.

| Baseline | Unweighted relative LL improvement | Unweighted Brier difference | Inverse-compatible relative LL improvement | Inverse-compatible Brier difference |
| --- | ---: | ---: | ---: | ---: |
| Retained history (`qhistory`) | 2.2940% | -0.00096222 | 2.9187% | -0.00182221 |
| Pattern head (`qpattern`) | 2.0093% | -0.00081072 | 2.0411% | -0.00117775 |
| Limited four-factor head (`qlimited4`) | 1.2008% | -0.00049523 | 1.2810% | -0.00072376 |
| Rejected old limited-path lineage | 0.5392% | -0.00032560 | 0.9424% | -0.00065056 |

The full head's unweighted log loss was `0.195787383` and Brier score was `0.052290429`. On the inverse-compatible surface they were `0.223305275` and `0.062008543`.

All six predeclared familywise moving-block endpoints passed. The one-sided 99.1667% upper bounds for the full head's log-loss/Brier differences were:

- versus history: `-0.003852 / -0.000767`;
- versus pattern: `-0.003456 / -0.000683`;
- versus limited four-factor: `-0.002135 / -0.000445`.

## Ranking and calibration

| Metric | History | Limited four-factor | Full nine-factor |
| --- | ---: | ---: | ---: |
| Top-three recall | 0.795885 | 0.797368 | 0.803881 |
| Top-three precision | 0.110923 | 0.111130 | 0.112038 |
| Positive-anchor hit rate | 0.841158 | 0.841702 | 0.848528 |
| Top-one recall | 0.466310 | 0.467478 | 0.475923 |
| Mean reciprocal rank | 0.689534 | 0.690039 | 0.697041 |

The roughly 80.4% top-three recall is not 80.4% exact-loop accuracy. Labels overlap, and top-three precision remained approximately 11.2%.

Raw-probability calibration passed without post-hoc recalibration:

| Model | ECE | Maximum supported-bin error |
| --- | ---: | ---: |
| History | 0.004429 | 0.059933 |
| Pattern | 0.001596 | 0.033148 |
| Limited four-factor | 0.001595 | 0.021906 |
| Full nine-factor | 0.001299 | 0.014299 |

The full model stayed below the frozen `0.02` supported-bin limit.

## Exact reliability failure

The full head improved on the limited head in every outer month, all five new-factor quartile families, every supported leave-one-stock-out slice, every required state, every transition length, the nonterminal cohort, and the early-entry cohort. It improved cycle log loss in 19 of 20 cycles; the contract required at least 15.

The stricter orientation rule required both log loss and Brier score to be no worse in all 44 supported compatible cycle/current-state units. Seven of 88 checks failed across four units:

| Orientation | Rows / positives | Log-loss difference | Brier difference |
| --- | ---: | ---: | ---: |
| `cycle_09__s6` | 6,095 / 416 | +0.000402 | +0.000104 |
| `cycle_15__s3` | 9,936 / 131 | +0.000132 | +0.000007 |
| `cycle_16__s1` | 9,844 / 241 | +0.000101 | +0.000010 |
| `cycle_20__s3` | 9,936 / 295 | +0.000199 | -0.000006 (passed) |

These are small reversals, but they were supported and predeclared as disqualifying. The rule was not weakened after seeing them.

## Falsification

The coherent 999-draw whole-session left-rotation null passed all three Holm-adjusted tests at empirical `p = 0.001`:

| Statistic | Observed improvement | Null 99th percentile |
| --- | ---: | ---: |
| Unweighted log loss | 0.012010 | 0.000865 |
| Inverse-compatible log loss | 0.012817 | 0.000796 |
| Top-three recall | 0.006515 | 0.001349 |

This supports genuine pooled prediction/label alignment under the declared null. It cannot override the local orientation failures.

## Interpretation and relationship to loop quality

The answer to the research question is nuanced:

- **Yes:** causal factors beyond the state pattern improve average loop-occurrence likelihood in 2024 OOF data. The five added price-context factors improved pooled log loss by about 1.20% over the B0/stress/clock head.
- **No:** this particular nine-factor specification is not reliable enough to replace the retained history forecaster. Its gain reverses in four supported loop/current-state orientations.
- **No quality promotion:** an occurrence probability answers “which compatible loop is more likely to appear?” It does not answer “will that loop be a good/high movement loop?” The separate frozen quality lineage still has zero qualified good/high loops among the current 20-cycle dictionary.

The current model is therefore a useful diagnostic about where extra occurrence information exists, not a retained forecasting component.

## Integrity, audit, and reproducibility

- Contract SHA-256: `ef8b61bdd4f6671fa64713551a9991f6e4591c3c96bc1ccc324c81b7195bfe7d`.
- Runner SHA-256: `aafb89c6046b752335c7da664c0e8f35062eb66014b86148931fbc92180fa9ff`.
- Core SHA-256: `905faa2c6ca612888ba6dc65656e7a0148400b319b810754bc7a4be65365b36d`.
- Evaluator SHA-256: `7dd441964e6906e9c6c95dea119eabeccfc50e0ddecc3535c51ff5a70594a8e1`.
- Complete fit-artifact manifest SHA-256: `1bf16fd5fb4a102c9f9f59a7cbc7b0b44a78be3137941701b08586336bea63b6`.
- Independent read-only result verification matched all 20 declared artifact sizes and hashes, replayed pooled metrics to maximum absolute error `5.61e-15`, replayed ranking to `9.71e-17`, and confirmed the exact four orientation reversals.
- Independent rejection auditor SHA-256: `c22171bec8a9d19ff0736d36ef7e4f54b6a8df36c642a0cedec5d47aaccbb2ac`.
- Independent audit-result SHA-256: `18d4290c50f749ce6ec5434324afa82cd7bebafcd8be198ed8b1c6c7361eedb1`.
- Independent rejection audit: 47/47 checks passed. It reconstructed all 110,949 anchors, 759,212 compatible rows, 46,630 positives, all 15 serialized grid/outer folds, 108 lambda-grid rows, 72 outer selections, 361,220 OOF predictions, and every loss, ranking, calibration, slice, bootstrap, falsification, and gate result without importing the production runner, core, or evaluator.
- Final full workspace test suite: 191/191 passed. Auditor-specific test SHA-256: `689f4eab639f20b9f950cf71d13282203ff73ceffa3f0966a5b1933da68fdcf2`.
- Fit status: `stopped_2024_primary_gates_failed`.
- Scoring authorization: false.

The first full independent-audit attempt failed closed on an auditor-only omission of the runner-added `validation_month` metadata field. It had already matched the rows, probabilities, metrics, gates, and rejection. No audit or authorization marker was written. The auditor expectation was corrected, its focused tests passed, and the complete clean replay then passed 47/47. This did not change the frozen model, contract, artifacts, probabilities, scientific gates, or decision.

The passing result is stored as `pre_score_audit.json`. It explicitly records `development_2024_primary_pass: false`, `rejection_verified: true`, `scoring_authorized: false`, and `authorization_marker_written: false`. No `pre_score_authorization.json` exists.

The row-level fit artifacts are under:

`/private/tmp/stocker_factor_conditioned_loop_occurrence_v1_20260711`

That directory is ephemeral and should be archived before reboot if exact replay without recomputation is required.

## Next research boundary

Do not tune away these four failures on the same opened OOF results. A separate frozen experiment may examine partial pooling or targeted interactions that explain why the added factors reverse in specific cycle/current-state orientations. Any such model remains an occurrence model and must stay separate from movement-quality qualification.
