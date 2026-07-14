# Admission-conditioned exact cycle identity V1

Date: 2026-07-13

`research_only: true`  
`live_ordering_enabled: false`  
`order_placement: disabled`

## Decision

The proposed parent exact-cycle test revealed that the problem was framed one layer too late.

For every successful quick or persistent parent loop, the frozen `orientation_key` already determines the parent cycle ID exactly. The key is encoded as `candidate_parent_cycle|A->B`, and the successful parent path is `A->B->A`. Across Q3 2024, 2023, and 2025, the orientation decoder matched the completed parent identity 100%.

Therefore the unsolved parent problem is not “which parent cycle ID?” It is:

1. will the observed or predicted `A->B` transition return `B->A` at all;
2. if it returns, will it be quick or persistent;
3. when will it close?

The six strong parent admissions from the preceding test help with questions 1 and 2. They cannot add parent identity information because that identity is already a dictionary lookup once `A` and `B` are fixed.

Exact identity is genuinely ambiguous only for three frozen branches:

- morph `3->1`: cycle 16 versus cycle 18;
- child `1->3`: cycle 14 versus cycle 15;
- child `1->2`: cycle 15 versus cycle 17.

The frozen morph admission failed. The frozen child admission produced a stable and statistically robust exact-ID probability improvement, but failed the predeclared support gate. It is a provisional child clue, not a solved identity layer.

No P&L, execution, or tradability calculation was performed.

## Why the parent test is structurally degenerate

The frozen parent dictionary contains one cycle for every eligible reversible edge. Examples include:

- `1->3->1` = cycle 01;
- `1->2->1` = cycle 02;
- `4->6->4` = cycle 06;
- `5->6->5` = cycle 07.

The decision row at the first completed transition already contains the directed edge. Its orientation key also contains the corresponding candidate parent cycle name. If the event subsequently closes back to the source state, no second identity classifier is required.

The earlier condition-to-cycle mixtures were aggregated across many orientations. For example, a small-body admission could contain cycles 01, 05, and 09 because that same market condition occurred on several different `A->B` edges. It did not mean those cycle IDs remained ambiguous after the edge was fixed.

This corrects the previous interpretation: admissions narrow **completion route and persistence**, while orientation supplies parent identity.

## Structural audit

The audit checked every successful event against the frozen twenty-cycle dictionary.

| Family | Period | Successful loops | Structurally determined | Ambiguous |
|---|---:|---:|---:|---:|
| Quick parent | 2023 | 14,857 | 100.0% | 0 |
| Quick parent | 2025 | 13,577 | 100.0% | 0 |
| Persistent parent | 2023 | 19,186 | 100.0% | 0 |
| Persistent parent | 2025 | 19,314 | 100.0% | 0 |
| Morph | 2023 | 2,900 | 78.4% | 625 |
| Morph | 2025 | 2,514 | 73.8% | 658 |
| Child | 2023 | 4,408 | 54.0% | 2,029 |
| Child | 2025 | 3,484 | 45.3% | 1,905 |

Every observed cycle belonged to the dictionary set permitted by its orientation. Every structurally determined success decoded correctly.

## Frozen ambiguous-branch test

The experiment conditioned on eventual morph or child success to isolate exact identity. This is an oracle-family diagnostic, not a deployable joint forecast.

Q3 2024 trained the models. The unchanged 2023 and 2025 rows were scored outside the fit period.

Four models were evaluated:

1. orientation only;
2. orientation plus admission and a nonredundant orientation interaction;
3. orientation, stock, session, and previous-state context;
4. the same context plus admission and interaction.

The candidate had to beat both the simple orientation model and the stricter context model on admission-positive rows. Metrics used paired five-session block bootstrap with 5,000 draws.

The fixed occurrence admissions were:

- morph: downward onset and absolute transition-leg return no more than 14.9105 bps;
- child: trailing-three pre-transition absolute body no more than 11.7663 bps and trailing-three range no more than 27.7724 bps.

No raw-feature expansion or parameter sweep was allowed.

## Child result: promising but under-supported

Among admitted ambiguous child successes, the admission-conditioned model improved multiclass log loss against both baselines in both outside years.

| Comparator | 2023 relative log-loss improvement | 2025 relative log-loss improvement | Robust both years? |
|---|---:|---:|---:|
| Orientation only | +2.92% | +3.28% | Yes |
| Stock/session context | +2.05% | +1.96% | Yes |

Against orientation only, Brier score improved by 3.61% in 2023 and 4.03% in 2025. Top-1 accuracy stayed unchanged at 68.86% and 69.41%; the gain was better probability calibration rather than a different most-likely cycle.

Against the less stable stock/session context model, top-1 accuracy improved from 58.82% to 62.28% in 2023 and from 60.59% to 70.00% in 2025.

The admitted child identity mixtures were unusually stable:

| Parent edge | Future child alternatives | 2023 admitted mixture | 2025 admitted mixture |
|---|---|---|---|
| `1->3` | cycle 14 versus 15 | 62.1% / 37.9% | 61.5% / 38.5% |
| `1->2` | cycle 15 versus 17 | 29.1% / 70.9% | 29.2% / 70.8% |

However, the frozen support gate failed:

- Q3 trained from only 57 admitted ambiguous child successes;
- one Q3 child class had only 4 examples;
- 2023 had 289 admitted successes with a minimum class count of 41;
- 2025 had 170 admitted successes, but its smallest class had 16 cases rather than the required 20.

The correct label is **provisional child identity clue**. It is strong enough to justify prospective logging, but not enough to claim the child branch is solved.

The admission model also did not improve the entire ambiguous-child population. Its value was confined to admission-positive rows, which supports a gated interpretation rather than a universal child matrix.

## Morph result: rejected

The morph admission was too sparse and unstable:

- Q3 contained only 9 admitted ambiguous morph successes, all cycle 18;
- 2023 contained 30 admitted cases: 5 cycle 16 and 25 cycle 18;
- 2025 contained 17: 7 cycle 16 and 10 cycle 18.

It beat orientation-only log loss in both outside years, but the 2025 effect was small and it failed against the stock/session context model. Context-conditioned top-1 accuracy fell by 17.65 percentage points in 2025.

It failed support, class-balance, context-consistency, and top-1 gates. The morph admission does not solve exact morph identity.

## What this means for the loop problem

The parent problem is now cleaner than previously stated:

`current regime A + predicted/observed next regime B -> candidate parent cycle ID`

The learned layer should then answer:

`Will B return to A? If yes, quick or persistent?`

That is where the pre-loop and onset admissions help. A separate parent cycle-ID model would duplicate information already contained in the regime transition.

For morphs and children, orientation sometimes leaves more than one future cycle possible. Those require branch-specific probabilities. The low-body/low-range child admission may be the first useful exact-identity modifier, but current support is not sufficient for retention.

This result also explains why a single “next loop N” classifier was difficult to reason about: it mixed a deterministic dictionary lookup, a probabilistic completion decision, a persistence decision, and a smaller set of genuinely ambiguous future branches.

## Recommended next research step

Do not build another parent identity classifier.

Use the following research decomposition:

1. retain the existing next-regime forecast;
2. convert current `A` and predicted `B` to the candidate parent cycle by frozen dictionary lookup;
3. predict parent completion and quick-versus-persistent route with the strong admissions;
4. abstain when no admission is present;
5. prospectively log the child low-body/low-range admission until every ambiguous child class exceeds the frozen support requirement on genuinely new sessions;
6. keep morph identity unresolved.

The next fresh child test should be frozen before attaching new outcomes. Existing 2023–2025 data are development evidence and have already been inspected.

## Integrity and artifacts

Data: EODHD-provider five-minute regular-session OHLCV for twenty US stocks. Provider volume is labelled `historical_volume`; quotes, ticks, spread, order book, and news were unavailable.

MFE, MAE, future transitions, confirmed-route state, and run duration were not model features.

No app/runtime source was changed. No order, deployment, or P&L path was used.

Research contract and runner:

- `/private/tmp/stocker_admission_identity_test_20260713.md`
- `/private/tmp/run_admission_identity_test.py`

Artifact root:

- `/private/tmp/stocker_admission_identity_test_20260713`

The exact rerun reproduced summary SHA-256:

`9045d6c62587787cb7c7481343cc7117b38e3ebbd700c2558f9d109e70b67194`

The `/private/tmp` artifacts are ephemeral and should be archived before reboot if exact ledgers are required without recomputation.
