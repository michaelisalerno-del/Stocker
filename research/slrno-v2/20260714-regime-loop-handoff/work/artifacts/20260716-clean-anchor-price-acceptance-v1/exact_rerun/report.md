# Clean Anchor Price Acceptance V1

## Scientific decision

**experiment_blocked_by_missing_anchor_or_bar_identity**

The exact 2025 attribution ran and is reported below, but the registered two-period hypothesis cannot be decided: the hash-pinned 2023 five-minute provider tape no longer exists. All 854 frozen 2023 named candidates remain explicit missing evidence, never zero and never reconstructed from a similar field. On the available 2025 surface, the registered interaction is **not supported** and should be retired rather than tuned. These opened retrospective surfaces cannot establish a tradable edge in any event.

## 1. Exact hypothesis and prior boundary

The registered primary comparison asks whether the exact frozen Sequential Loop Competitor Veto Track-A anchor decision plus one completed-bar directional acceptance rule improves **remaining** net payoff relative to the same named candidates at the same delayed entry and original terminal clock. This exact paired interaction had not been tested. Selective Payoff Equations V1 used a different 2024 OCO population, fitted many one-to-three-bar variables, and restarted horizons; Sequential Veto tested the anchor veto and evolving structural exclusion separately; payoff-state, lead-lag, and rotation work tested different state questions.

## 2. Frozen populations and inputs

- `cycle_04|state_4`: 2023=132, 2025=96.
- `cycle_07|state_5`: 2023=722, 2025=713.
- Frozen controls are `cycle_04|state_2` (neutral) and `cycle_07|state_6` (negative); they remain separately labelled and never enter the primary result.
- Source opportunity identity is the frozen V2 `opportunity_id`, joined one-to-one to the frozen Sequential Veto `event_lineage_id` and policy row.
- The anchor reference is V2 `anchor_close`, independently required to equal the hash-pinned provider close at `start_timestamp`.

## 3. Frozen anchor veto

The anchor score is `anchor_good_mass / anchor_bad_mass`, infinity when bad mass is zero. The exact frozen threshold is **1.0**: pass strictly above 1.0, reject at or below 1.0. Good/bad/unknown classifications, smoothing, support, and policy decisions come unchanged from Sequential Loop Competitor Veto V1. The veto is not updated after price is observed.

## 4. Checkpoint, price sign, and economic clock

The only checkpoint bar starts exactly five minutes after the anchor and freezes at its close ten minutes after anchor. Entry is the exact provider open at that freeze; a later row cannot substitute for a missing bar. For a long (short), returns and excursions are direction-adjusted exactly as registered. Admission requires signed close return > 0 and favourable excursion > adverse excursion. All A--D variants enter on the same clock and exit at the original `anchor + 125 minutes` terminal close, with 5 bps charged at entry and exit. Restarted h24 outcomes are separate diagnostics.

The retained causal range ledger is empty, so Variant E is unavailable. No replacement range model was fit.

## 5. 2025 variant results (constant terminal)

| Variant | admitted | coverage | net bps | mean/admitted bps |
|---|---:|---:|---:|---:|
| A same-clock base | 809 | 100.0% | 27936.20 | 34.53 |
| B anchor veto | 808 | 99.9% | 27974.57 | 34.62 |
| C price acceptance | 513 | 63.4% | 12250.81 | 23.88 |
| D anchor + acceptance | 513 | 63.4% | 12250.81 | 23.88 |
| E + range | 0 | unavailable | unavailable | unavailable |

## 6. Primary paired result and interaction

Variant D minus A is **-15685.39 bps**, or -19.39 bps per paired opportunity. The five-session-block 95% interval for the session-mean increment is [-27.71, -1.66] bps. Price acceptance after the anchor veto contributes -15723.76 bps; the anchor veto after price acceptance contributes 0.00 bps.

Both named loops reject the interaction on 2025: `cycle_04|state_4` D-minus-A is -1544.24 bps, and `cycle_07|state_5` is -14141.15 bps.

## 7. Four-cell interaction

| named loop | frozen cell | opportunities | mean net bps | total net bps | positive rate |
|---|---|---:|---:|---:|---:|
| cycle_04 | anchor_pass|acceptance_fail | 37 | 41.74 | 1544.24 | 59.5% |
| cycle_04 | anchor_pass|acceptance_pass | 59 | 44.92 | 2650.06 | 52.5% |
| cycle_07 | anchor_fail|acceptance_fail | 1 | -38.37 | -38.37 | 0.0% |
| cycle_07 | anchor_pass|acceptance_fail | 258 | 54.96 | 14179.52 | 51.6% |
| cycle_07 | anchor_pass|acceptance_pass | 454 | 21.15 | 9600.75 | 53.5% |

For `cycle_07`, anchor-pass/acceptance-pass averages 21.15 bps versus 54.96 bps when acceptance fails. The sole 2025 anchor-fail row is a −38.37 bps loss, so the frozen anchor veto itself is effectively a one-row filter. The 2023 cells are explicitly unavailable because no causal checkpoint OHLC survives; they are not inferred.

## 8. Veto value, continuous relationship, and nulls

Variant D avoided 26172.09 bps of losses while rejecting 41857.47 bps of winners, for veto value -15685.39 bps. The predeclared continuous acceptance diagnostic has Spearman rho 0.008 (p=0.826); neither named loop has a meaningful monotone relationship. At matched coverage, random admission averages 17607.67 bps, with the registered D result at only the 11.8% percentile. Time-shifted, flipped-direction, and close-sign-only controls are exported without selecting a replacement rule.

## 9. Stress, concentration, and failure cases

- Twice costs: D remains raw-positive at 7120.81 bps, but A is 19846.20 bps and the paired D-minus-A stress remains -12725.39 bps.
- One additional execution bar, with the original acceptance frozen and terminal unchanged: D is 8192.55 bps, A is 19219.10 bps, and D-minus-A is -11026.55 bps.
- D after removing the best stock is 4669.64 bps, but removing the top five stocks produces -4135.46 bps. The top stock supplies 30.2% of absolute D contribution and the top five supply 67.9%.
- D after removing the best hindsight episode is 7680.04 bps; removing the top five produces -4099.00 bps. Episode labels are outcome diagnostics only.
- Fully rebuilt leave-one-stock-out is blocked because the immutable 2023 V1/V2 rebuild inputs and provider tape are absent; no aggregate deletion is mislabeled as a rebuild.
- Stock, loop, direction, period, month, regime, clock, and hindsight-episode concentration are in `concentration_results.csv`. Hindsight episodes are outcome diagnostics only.
- A favourable first bar can still fail later; acceptance is a deterministic sign, not a route-completion prediction.

## 10. Interpretation and exact recommendation

The 2025 result does **not** support the proposed story that the loop supplies the candidate, the anchor removes material contamination, and one completed bar supplies a useful sign. The anchor removes only one row, price acceptance discards more winner payoff than loss payoff, and its continuous balance is uninformative. The primary registered two-period endpoint remains formally blocked by missing 2023 bars, but the available evidence points against the interaction rather than toward prospective promotion.

Retire this exact price-acceptance interaction and do not add bar features or tune its thresholds on 2025. The single useful next action is to restore and hash-pin the original 2023 provider tape solely to close the registered archival result; if it cannot be restored, leave the experiment blocked and do not launch a prospective acceptance gate.

## Reproducibility

- Run ID: `clean-anchor-ee4b0f206c7412cba0fd4a4b`
- Git SHA at execution: `1b7a415b6c2ca7419047f7209617f845650abce9`
- Contract SHA-256: `807e9d6b72e3ea59d9ca960f8d03f1e2b482ceb50086844bef298b1e8e06e42d`
- Data snapshot SHA-256: `ecff4d50651fb210e93f2ddb1d2bd736f864ed83d6812e7a830abb51f58cfe05`
- Exact command: `PYTHONPATH=packages/stocker_research/src .venv/bin/python research/slrno-v2/20260714-regime-loop-handoff/work/run_clean_anchor_price_acceptance_v1.py --output <OUTPUT> --report <REPORT>`
