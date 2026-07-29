# Selective payoff equations V1

Date: 2026-07-14

Decision: **`all_equations_rejected_or_unknown`**

Scientific status: post-inspection causal retrospective equation development on already-opened 2024 data. This is not validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no trading app, broker, paper/demo account, deployment, position, or order functionality was changed or used
- provider volume is labelled `historical_volume_activity_proxy_not_quote_flow_or_order_book_volume`

## Direct answer

Combining the available information in a payoff equation did not produce a qualified profitable-move detector.

The result answers the prediction-versus-information question:

- predictions alone contained essentially no target-before-invalidation ranking information;
- causal local context contained a small, insecure amount;
- adding all 20 frozen loop scores made context prediction worse;
- one-to-three completed bars contained the strongest information, but only enough to rank paths weakly—not enough to emit a causally calibrated positive-expectancy admission.

Every conservative equation and every calibrated point-mean diagnostic abstained on every primary opportunity. This was not caused solely by wide uncertainty: even the maximum calibrated point expected net payoff remained negative.

The mountains of information are therefore currently more useful for identifying **bad or unknown occasions** than for proving a profitable occasion.

## Frozen experiment

The existing 2024 `causal_setup_conditions_v1` OCO-breakout surface was used because it was the only opened source with:

- all 20 frozen causal loop compatibility scores;
- exact five-minute provider paths;
- causal bar, session, range, VWAP, activity, and structural-risk features;
- enough observations for prior-session-only model fitting and calibration.

The raw surface had 6,025 filled rows across 128 sessions and 22 symbols. The frozen 20-stock universe, exact timestamp alignment, 20–250 bps risk bounds, latest-decision clock, and 24-bar per-symbol cooldown produced:

- 3,339 outcome-free base events;
- 10,017 one-, two-, and three-bar checkpoint rows;
- 6,780 checkpoints causally eligible for delayed admission;
- 1,645 primary base opportunities over the final 68 sessions;
- 3,300 eligible primary sequential predictions.

The first 40 sessions supplied the minimum rolling model history. Primary scoring began only at session index 60, after prior prequential predictions existed for empirical calibration. All calibration used prior prediction outcomes only.

The shared base population averaged **-16.97 bps per event** after 5 bps per side, with a 43.16% target-first rate.

## Equations

### Prediction-only comparator

`logit(p_target) = state + direction + clock + all 20 loop scores + loop mass + entropy + margin`

The source loop values are compatibility scores, not exhaustive probabilities; they do not sum to one. Total mass and normalised-mixture diagnostics were therefore kept separate.

### E1: context only

Context included causal entry step, session position, structural risk, anchor and decision range/ATR, body and wick structure, close location, directional displacement, VWAP distance, historical-volume activity proxy, compression, trend, current/rolling returns, state, previous state, direction, and clock.

### E2: context plus loop mixture

E2 added all 20 loop scores, total compatibility mass, entropy, top score, margin, and top-loop label to E1.

### E3: sequential confirmation

E3 added one, two, and three completed-bar snapshots:

- directional close return;
- causal running MFE and MAE;
- retracement;
- current range, body, wicks, VWAP distance, and activity;
- favourable-close fraction;
- structural-invalidation status and a recomputed next-open risk.

It could select only the earliest checkpoint whose structural invalidation had not touched and whose later-entry risk remained inside the frozen bounds.

All models were fixed L2 logistic equations with prior 60-session training only. Raw probabilities were calibrated from the nearest prior prequential predictions with Jeffreys beta-binomial uncertainty. The primary equation was:

`conservative EV = (2 × calibrated target probability lower bound − 1) × risk bps − 10 bps`

Only positive conservative EV could emit an admission.

## Predictive results

| Equation | Rows | Target rate | Raw AUC | Raw log-loss | Raw Brier |
|---|---:|---:|---:|---:|---:|
| Prediction only | 1,645 | 43.16% | 0.495 | 0.6918 | 0.2491 |
| E1 context only | 1,645 | 43.16% | 0.534 | 0.6878 | 0.2474 |
| E2 context + loop mixture | 1,645 | 43.16% | 0.530 | 0.6929 | 0.2496 |
| E3 sequential confirmation | 3,300 | 41.58% | 0.588 | 0.6746 | 0.2405 |

E3 has a different later-entry snapshot label, so its raw scores should be compared with its own target rate rather than treated as a direct AUC contest with the base equations.

### Context did not securely beat predictions

E1 improved per-row log-loss over prediction-only by +0.00397 on average, but its 95% five-session-block interval was **-0.00552 to +0.01413**. The Holm-adjusted p-value was 0.445. This is weak descriptive information, not a secure increment.

### The full loop mixture hurt context

E2's log-loss improvement over E1 was **-0.00509**, with a 95% interval of **-0.00969 to -0.00048**. Its AUC also declined from 0.534 to 0.530.

This does not say the loop forecasts are structurally wrong. It says the entire loop-score vector did not help identify target-before-invalidation payoff on this setup surface. The loop map and profitable-occurrence target remain different variables.

### Completed path state ranked best, but remained below break-even

E3 reached AUC 0.588. Its checkpoint AUCs were:

- checkpoint 1: 0.584;
- checkpoint 2: 0.584;
- checkpoint 3: 0.596.

The direct equations' median risk-specific break-even probability was 55.09%. Their maximum causally calibrated means were only 48.00%–48.40%.

For E3:

- median break-even probability: 54.72%;
- maximum calibrated mean: 51.99%;
- maximum calibrated lower bound: 47.26%;
- best point expected net: -4.52 bps;
- best conservative expected net: -11.84 bps.

Therefore, E3's zero coverage was not merely an overly cautious confidence interval. Its causally calibrated point estimate never reached economic break-even.

## Selective payoff result

| Equation | Conservative selections | Point diagnostic selections | Qualified |
|---|---:|---:|---:|
| Prediction only | 0 | 0 | No |
| E1 context only | 0 | 0 | No |
| E2 context + loop mixture | 0 | 0 | No |
| E3 sequential confirmation | 0 | 0 | No |

Consequently:

- coverage was zero;
- no selected-net, cost, month, or stock gate could pass;
- paired selector-return increments were zero because every model correctly abstained under its frozen rule;
- no equation is retained for promotion or activation.

An abstaining model is unknown, not profitable. It is preferable to forcing a negative-expectancy ranking into a research signal.

## Post-inspection path diagnostics

These diagnostics were opened only after the primary rejection and cannot define a rule.

The highest 5% of E3's **raw**, uncalibrated snapshot probabilities had:

- 165 rows;
- 55.15% target-first precision;
- +8.43 bps mean net.

This is not a candidate threshold:

- 5% was inspected after scoring;
- it used a global hindsight percentile unavailable causally;
- the highest 1% fell back to 51.52% precision;
- prior-only empirical calibration did not confirm the tail;
- multiple checkpoints from the same base event can appear in the snapshot tail;
- no predeclared time, stock, multiplicity, or paired economic test exists for it.

The lowest 10% of E3 raw probabilities had 24.85% target-first precision and -22.74 bps mean net. This suggests that completed path state may be better at identifying unfavorable occasions than profitable ones. That is a **prospective negative-veto hypothesis**, not a retrospective filter.

## What this means

The useful decomposition is now:

1. **Predictions remain a structural map.** They should not be treated as payoff probabilities.
2. **Static context is weak.** It may provide a prior, but it did not securely improve payoff prediction.
3. **The complete loop vector adds no admission value on this surface.** More loop-score interactions would be post-score mining.
4. **Completed price path contains the strongest occurrence-level information.** At present it supports rejection/abstention more than positive admission.
5. **Positive detection remains unresolved.** No causal equation produced a supported probability above risk-specific break-even after costs.

The next honest question is not “which larger combination should be tried on 2024?” It is whether a frozen sequential model can prospectively emit stable negative, positive, and unknown states on genuinely new observations. `work/contracts/20260714-selective-payoff-prospective-log-v1.json` specifies that experiment without activating or implementing anything.

## Retired opened-data paths

Do not continue tuning on opened 2024:

- the frozen prediction-only equation;
- the frozen context feature set;
- the full 20-loop-score mixture or new loop-score interactions;
- the one-, two-, and three-bar path feature equation;
- probability percentile, checkpoint, support, calibration-neighbourhood, target, stop, risk, or horizon searches;
- the post-inspection top or bottom raw-probability tails.

## Integrity and reproducibility

- Pre-score manifest SHA-256: `49ca78c91a0b7092c906ec37a880eebb1b3dcebadb4dee09adc6b7dd1596efa5`.
- Frozen contract SHA-256: `a11f7702b2766c0dfe91f9c47b87a38051244be357665e759ce4445bf03970b8`.
- Frozen runner SHA-256: `983b421f194f8af51efbf2171158c473dedf1b03bbf8de111ff4e9d18610fe02`.
- Frozen auditor SHA-256: `63dc949f0e29260dc098f4d62806b01443d5bb078a1086be2543ef9122220998`.
- Focused tests SHA-256: `54ae8f68833026c02c8cbd07a80f851e1f8183481793b792876ca9194196f072`.
- Artifact manifest SHA-256: `d7741f5cb622125dc3e5c9f65e9a1cdb8c86aaff7d61e2a21623794d2e7eb613`.
- Focused tests: 12/12 passed; lint passed.
- Independent audit: 14/14 checks passed.
- Independently reconstructed base population, causal base features, causal sequential features, all payoff paths, all direct and sequential probability refits, and calibration had maximum error 0.
- Exact rerun: all 23 files were byte-identical.
- The pre-existing dirty `StockerLocal` worktree was not modified.

Primary artifacts:

`work/artifacts/20260714-selective-payoff-equations-v1/primary`

Exact rerun:

`work/artifacts/20260714-selective-payoff-equations-v1/exact_rerun`
