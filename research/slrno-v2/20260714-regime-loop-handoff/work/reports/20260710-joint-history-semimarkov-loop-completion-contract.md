# Joint history-conditioned semi-Markov loop-completion contract

## Scope

This development contract was written before calculating the new 2025 or
backward-2023 joint-kernel completion scores. It tests the proposed replacement
for the rejected independence product between loop path and state-only dwell
time. It does not modify the frozen detector or the prospective movement
shadow ledger.

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`. No 2026 row, provider price, provider volume,
direction, return, range, P&L, cost, order, broker, position, strategy, or
deployment field is permitted.

2025 remains development data and 2023 remains a future-fitted backward
portability period. Neither can support a prospective-validation claim.

## Frozen joint kernel

The next-state marginal remains the retained 2024-fitted last-three-state
multinomial model. Dwell time is made conditional on the same history and the
hypothesised destination:

`P(next, dwell | prev2, prev1, current)`

`= P_history(next | prev2, prev1, current)`

`× P_backoff(dwell | prev2, prev1, current, next)`.

This is a normalized joint distribution by the probability chain rule. It
preserves the retained loop-identity probabilities exactly when dwell time is
marginalized, while allowing path identity and dwell time to interact.

Dwell buckets are exact durations 1–23 plus one overflow bucket for durations
of at least 24 bars. Because every fixed loop contains at least two
transitions, an overflow constituent cannot contribute to a completion within
the maximum 24-bar horizon. Terminal session-end runs are excluded from dwell
fitting because their duration is boundary-truncated rather than a natural
state departure. Session end remains in the destination probabilities.

All dwell estimates use complete non-terminal 2024 causal online runs only. The
fixed hierarchical prior is:

1. frozen 2024 semi-Markov current-state PMF;
2. current state plus next state, prior strength 256;
3. previous state 1 plus current and next state, prior strength 256;
4. full previous-state-2/previous-state-1/current/next context, prior strength
   1024.

Those strengths must reproduce as the unique/tie-broken minimum of a
predeclared 2024-only expanding-month screen: histories before each July–
December month score that month, the Cartesian grid is `{64, 256, 1024}` at
each level, and pooled dwell log loss selects `(256, 256, 1024)`. Failure to
reproduce stops the experiment before 2025/2023 are read.

No smoothing value may change after scoring-period results are calculated.

## Dynamic rollout

At every causal run entry with zero-based New York clock bar ordinal at most
53, enumerate every deduplicated frozen-cycle rotation beginning at the current
state. For each required transition, multiply the retained destination
probability by the applicable conditional dwell PMF, update the hypothetical
last-three-state context, and convolve the route-duration distribution.

For 6-, 12-, and 24-bar horizons, sum route mass whose constituent dwell times
total no more than the horizon. Session end remains destination 8 and makes an
unfinished route negative. Compatible cycle probabilities remain overlapping;
no mutually exclusive `no loop` class is manufactured.

The target is one only when the observed future run states exactly close a
compatible route within-session and their constituent durations sum to no more
than the horizon.

## Frozen baselines

- `history_path_only`: retained path probability without timing.
- `history_frozen_state_timed`: retained path transitions with the previously
  tested frozen semi-Markov state-only duration PMFs.
- `history_destination_timed`: retained path transitions with
  `q1(dwell | current, next)`.
- `history_order2_timed`: diagnostic retained path transitions with
  `q2(dwell | prev1, current, next)`.
- `history_joint_timed`: the candidate history-and-destination-conditioned
  `q3(dwell | prev2, prev1, current, next)` kernel.

The destination-conditioned baseline separates the value of knowing which
state comes next from the additional value of deeper history.

## Hypotheses

| Hypothesis | Expected benefit | Safety/statistical risk | Stop condition |
| --- | --- | --- | --- |
| H1: hierarchical conditional dwell PMFs are valid | stable estimates in sparse last-three-state/destination cells | sparse-context memorization or non-normalized mass | any source, normalization, overflow, or backoff reconstruction fails |
| H2: joint conditioning fixes the failed independence model | better 6/12/24 completion probabilities than frozen state-only and destination-only dwell | prevalence correction without route discrimination | either primary comparison fails in either period |
| H3: the joint model predicts the correct near-term loop more often | horizon-specific top-three ranking improvement | pooled proper-score gain with no useful ranking change | recall gain below 0.005 versus state-only or any loss versus destination-only in either period |
| H4: timing does not damage the retained path model | destination marginals remain exactly the retained history model | silently replacing the successful identity forecaster | marginal mismatch or failure versus path-only at any horizon |

## Gates

Each period/horizon requires at least 300,000 compatible anchor-cycle rows,
8,000 positive completions, all twenty cycles, at least forty positives per
cycle, eighteen stocks, four quarters, and all eight current states.

For `history_joint_timed` versus frozen state-only timing and
destination-conditioned timing, separately in 2025 and 2023:

- pooled log-loss relative improvement at least 0.5% versus frozen state-only
  timing and 0.25% versus destination-conditioned timing;
- five-session moving-block 95% upper bounds below zero for log-loss and Brier
  differences;
- both losses lower at all three horizons, all four quarters, and every
  leave-one-stock-out deletion;
- ECE no worse at every horizon and maximum supported-bin error no more than
  0.01 above baseline, using ten fixed bins and 500-row support;
- at least fifteen of twenty cycles have lower pooled log loss;
- top-three completion-label recall improves by at least 0.005 versus frozen
  state-only timing and is non-lower than destination-conditioned timing.

The candidate must additionally have lower log loss and Brier than path-only
at every horizon and improve top-three recall by at least 0.005 in each period.

Retain the joint completion model only if every support, comparison, sanity,
and independent-integrity gate passes in both periods. A pass concerns loop
completion identity/timing only and is not evidence of price direction,
economic edge, or tradability.
