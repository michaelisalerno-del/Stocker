# Causal loop state paths V1

Date: 2026-07-14

Decision: **`path_event_family_not_supported`**

Scientific status: post-inspection causal retrospective mechanism development for a prospective logging contract. This is not validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no trading app, broker, paper/demo account, deployment, position, or order path was changed or used
- provider volume remains labelled `historical_volume_activity_proxy`; it was not used

## Direct answer

Knowing the parent-loop identity does not yet solve the profitable-occurrence problem. More sharply, “100% loop prediction” must remain conditional: parent identity may be structurally determined **if the parent cycle completes**, but that is not 100% opportunity-level occurrence, completion within the payoff window, direction, or profitability.

Within the frozen 24-bar window, exact parent completion represented only 10.6%–29.2% of the four primary candidate-period populations. An incompatible first transition represented 42.9%–73.5%. Some paths could complete later, but the result confirms that conditional structural identity cannot be substituted for payoff-window occurrence.

The most important payoff finding is that **route correctness and economic correctness are different variables**:

- For `cycle_04|state4`, exact completion was profitable at the frozen close in both opened periods, but an incompatible first transition was also profitable; only the expected first leg followed by a diversion was negative in both.
- For `cycle_07|state5`, exact completion was negative at the frozen close in both periods, while an incompatible first transition was positive in both.
- Therefore, “exit when the expected route completes” and “exit when the route becomes incompatible” are not general payoff rules.

The predeclared causal next-open policies confirmed this. The primary first-terminal-route-event policy underperformed the frozen close in all four cells by -7.35 to -55.85 bps per opportunity. No primary or secondary endpoint passed its Holm-controlled family, and the route-specific policy did not consistently beat the generic first-transition control.

## Frozen experiment

The contract, runner, independent auditor, tests, V2 input, state-run files, fixed cycle dictionary, reports, and all 40 provider files were SHA-256 frozen before any path-conditioned payoff was opened.

Population:

| Candidate | 2023 | 2025 |
|---|---:|---:|
| `cycle_04|state4` | 132 | 96 |
| `cycle_07|state5` | 722 | 713 |

The population exactly reproduces the V2 190-session score surface after the first 60 provider sessions in each period were excluded.

Five exhaustive outcome-only path topologies were predeclared from the anchor state through anchor+24:

1. no transition;
2. expected first leg, no return yet;
3. exact parent completion;
4. incompatible first transition;
5. expected first leg followed by a diversion rather than return.

Four untuned hypothetical policies were predeclared:

- next open after the first terminal route event, meaning completion or invalidation;
- next open after exact completion only;
- next open after route invalidation only;
- next open after the first post-admission transition as a holding-time control.

Every state event was available only after its assigned 5-minute bar closed. Every hypothetical event exit used the following provider open. Events known before entry, same-bar exits, MFE/MAE thresholds, future duration, realised child/morph identity, and future state information were forbidden.

## Realised route topology and fixed-close payoff

### `cycle_04|state4`

| Path through anchor+24 | 2023 n / share / mean net | 2025 n / share / mean net |
|---|---:|---:|
| Exact parent completion `4→2→4` | 14 / 10.6% / +25.84 bps | 20 / 20.8% / +116.37 bps |
| Expected leg then diversion `4→2→X` | 21 / 15.9% / -39.22 bps | 15 / 15.6% / -51.73 bps |
| Incompatible first transition `4→X`, `X≠2` | 97 / 73.5% / +40.88 bps | 60 / 62.5% / +16.80 bps |
| No transition | 0 | 1 / 1.0% / +392.65 bps |

### `cycle_07|state5`

| Path through anchor+24 | 2023 n / share / mean net | 2025 n / share / mean net |
|---|---:|---:|
| Exact parent completion `5→6→5` | 184 / 25.5% / -84.36 bps | 208 / 29.2% / -14.62 bps |
| Expected first leg only `5→6` | 30 / 4.2% / +62.50 bps | 47 / 6.6% / -40.80 bps |
| Expected leg then diversion `5→6→X` | 118 / 16.3% / -65.88 bps | 131 / 18.4% / -4.21 bps |
| Incompatible first transition `5→X`, `X≠6` | 375 / 51.9% / +51.48 bps | 306 / 42.9% / +24.53 bps |
| No transition | 15 / 2.1% / +479.07 bps | 21 / 2.9% / +483.37 bps |

These are outcome-only descriptions, not causal admission features. The rare no-transition rows are especially unsuitable for rule promotion: absence can only be fully labelled at the horizon, support is small, and V2 already found that causal age/hazard did not identify payoff reliably.

Two post-inspection hypotheses are worth prospectively logging, but not retesting or tuning on 2023/2025:

- `expected_leg_then_diversion` was negative in all four opened cells;
- `incompatible_first_transition` was positive in all four, so it must not be called economic invalidation merely because it is structurally incompatible with the expected parent route.

The completion response was loop-specific rather than universal: positive for `cycle_04`, negative for `cycle_07`.

## Causal next-open policy results

The primary policy exited at the next open after the first post-entry exact completion or route invalidation, otherwise retaining the frozen close.

| Candidate | Period | Action coverage | Frozen mean net | Event-policy mean net | Paired difference | 95% session-block interval |
|---|---:|---:|---:|---:|---:|---:|
| `cycle_04|state4` | 2023 | 89.4% | +26.54 | -29.31 | -55.85 | -95.22 to -17.20 |
| `cycle_04|state4` | 2025 | 90.6% | +30.75 | +15.61 | -15.14 | -64.34 to +29.83 |
| `cycle_07|state5` | 2023 | 88.1% | +7.02 | -0.33 | -7.35 | -23.78 to +9.09 |
| `cycle_07|state5` | 2025 | 82.0% | +17.04 | +5.39 | -11.64 | -28.11 to +5.09 |

All four Holm-adjusted one-sided p-values were 1.0. Positive-quarter counts were 1/4, 0/3, 0/4, and 1/3 in the same table order; every cell had 0/20 positive leave-one-stock-out deletions for the primary paired difference. Absolute mean net payoff at 5 bps per side was also nonpositive in two cells.

Secondary policies did not rescue the mechanism:

| Policy paired difference vs frozen close | `cycle_04` 2023 | `cycle_04` 2025 | `cycle_07` 2023 | `cycle_07` 2025 |
|---|---:|---:|---:|---:|
| Exact completion only | -3.79 | -9.78 | +7.69 | -7.57 |
| Route invalidation only | -52.06 | -5.37 | -15.04 | -4.07 |
| First transition control | -50.09 | -24.14 | -7.01 | -7.84 |

No secondary endpoint passed Holm. The isolated positive completion result for `cycle_07` in 2023 had a -1.36 to +16.93 bps interval, failed multiplicity control, and reversed sign in 2025.

The terminal-route policy minus the generic first-transition control was -5.76, +8.99, -0.34, and -3.80 bps across the four cells; all intervals crossed zero. Thus there is no secure evidence that route structure adds exit value beyond generic exposure shortening. Generic shortening itself underperformed the frozen close in every cell.

Matched controls do not change the conclusion. The `cycle_04` controls had only 8 and 6 rows. For `cycle_07|state6`, the terminal event improved the already-negative frozen control by +19.75 bps in 2023 and +15.47 bps in 2025, but remained negative after costs at -15.76 and -18.23 bps. That is loss reduction on the opened control orientation, not a qualified payoff rule.

## Why the state-event exits failed

The state route and the trade payoff are related but not aligned clocks:

- A parent cycle describes a sequence of regime labels; it does not specify which leg carries the favorable price move for the fixed signal direction.
- A structurally incompatible transition can occur while the price move continues to pay. Exiting it as “invalid” therefore removed profitable exposure.
- Exact completion can arrive after the profitable leg has already reversed, or before a profitable continuation that the frozen close captures. Its economic meaning changes by loop.
- The V2 MFE/MAE split remains unresolved: state transitions alone did not distinguish the early peak that will be surrendered from the slower winner that needs most of the window.

This is why loop-occurrence accuracy and profitability accuracy must remain separate even if the loop identity is known perfectly conditional on completion.

## Most useful paths forward

### 1. Direct causal payoff-state modelling — highest priority

Predict the cost-adjusted payoff distribution conditional on the frozen loop forecast and the information available now. Output positive, negative, or unknown/abstain with support and uncertainty. Use hierarchical partial pooling with a loop-specific route-state effect; do not force every observation into a rank.

The primary target remains net payoff after 5 bps per side. Route-topology probabilities should be auxiliary forecasts, not substitutes for payoff prediction.

### 2. Prospectively forecast the next route branch

The opened topology separates payoff anatomies, but realised topology is oracle information. A causal forecast must emit probabilities for:

- orientation intact;
- expected leg active;
- exact completion next;
- incompatible first transition next;
- expected-leg diversion next;
- unknown.

This forecast can be tested as an input to the direct payoff model on new sealed observations. It must not be constructed by relabelling realised children or morphs at admission.

### 3. Join route state to causal price-path state

After each completed bar, log running favorable/adverse excursion, prior-ATR normalisation, running peak, causal retracement, volatility support, and the next-open counterfactual. The question becomes: “Given this route event and the price path already observed, is expected remaining payoff positive, negative, or unknown?”

No retracement, ATR, bar-count, or price threshold may be selected from the opened V2/V3 paths.

### 4. Test the diversion hypothesis only prospectively

The consistent negative sign of `expected_leg_then_diversion` is the clearest new mechanism lead, but it was found after opening the topology table. Freeze it as a named hypothesis and evaluate its next-open and remaining-horizon payoff only on genuinely new observations. Do not run another 2023/2025 search for exceptions or thresholds.

### 5. Add child/morph or broader context only through a causal forecast

Child/morph coarseness may explain why completion response differs between loops, but realised identities are oracle-only. Broader market or sector context is a lower-priority compact mechanism family and should enter only with timestamp-audited sources. Neither should become another large sparse lookup table.

Paths retired by this experiment:

- scalar state age/exit hazard as the payoff discriminator;
- automatic next-open exit on exact parent completion;
- automatic next-open exit on route incompatibility;
- automatic next-open exit on the first state transition;
- treating structural completion as synonymous with a profitable move.

## Prospective logging contract

`work/contracts/20260714-loop-payoff-prospective-log-v2.json` extends the prior contract with:

- a completed-bar route state machine;
- route-class probabilities and uncertainty;
- direct fixed-horizon and event-horizon payoff predictions;
- separate next-open counterfactuals for completion, incompatible first transition, and expected-leg diversion;
- positive/negative/unknown payoff state;
- explicit prohibition on reusing 2023/2025 to tune the new branch hypotheses.

The contract is a specification only and is not activated or implemented in the trading application.

## Integrity and reproducibility

- Pre-score manifest SHA-256: `c9433aa9fd08153361b38482dd594a5de35993379822357b6b403d2701e18d38`.
- Frozen contract SHA-256: `89db750c0037e64c66e76f29a4a598aacacad933d0e212eb9100f169db4eb0ab`.
- Frozen runner SHA-256: `c2a409abeba1bf8e9b66809e61ff6ebc05ca4aced160dca8fdf777ed35e2d81b`.
- Independent auditor SHA-256: `bf8199c9388808245b3467a592a66046cde8dd3ba42b4a6cbb4bfd6088276510`.
- Artifact-manifest SHA-256: `cd18877f6a54bb63fd8470a54e335396bea09dd264640daeacae79d8a109a9b6`.
- Focused tests: 7/7 passed; lint passed.
- Independent audit: 14/14 checks passed.
- Population, topology, event action, next-open clock, timestamp, pre-entry, and manifest errors: 0.
- Maximum fixed, event-price, and event-return replay errors: 0.
- Maximum aggregate error: `7.11e-15`; bootstrap replay error: `1.42e-14`; Holm replay error: `8.88e-16`.
- Exact rerun: all 17 files were byte-identical.
- The pre-existing dirty `StockerLocal` worktree was not modified.

Primary artifacts:

`work/artifacts/20260714-causal-loop-state-path-v1/primary`

Exact rerun:

`work/artifacts/20260714-causal-loop-state-path-v1/exact_rerun`
