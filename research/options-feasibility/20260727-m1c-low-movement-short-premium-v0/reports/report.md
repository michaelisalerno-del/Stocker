# Frozen Causal M1C Low-Movement Veto and Short-Premium Readiness Screen V0

Decision: `blocked_insufficient_low_tail_support`

This is retrospective underlying-stock movement and range-containment research. It does not calculate option P&L, model intraday option quotes, or establish short-option profitability, execution realism, paper/live readiness, or a deployable strategy.

## Frozen low-tail thresholds

- M1C: bottom 5% `0.115697407847643`, bottom 10% `0.135896965695626`, bottom 20% `0.167095528962669`.
- M0: bottom 5% `0.141444713455780`, bottom 10% `0.157941884144402`, bottom 20% `0.183996606681021`.

## Binding checkpoint results

- assessment: 740 rows, 123 sessions, 20 stocks; remains-below-IV 90.14%; NPV lift 16.89%; mean/median IV residual -0.005593/-0.005576; 1.5σ/2.0σ excursion breach 2.97%/1.22%.
- stress: 1426 rows, 63 sessions, 20 stocks; remains-below-IV 91.51%; NPV lift 18.01%; mean/median IV residual -0.004942/-0.004707; 1.5σ/2.0σ excursion breach 1.19%/0.28%.

## Fresh quiet episodes

- assessment: 308 episodes, support gate `pass`; remains-below-IV 87.01%; mean/median residual -0.004526/-0.004689; 15-minute 1σ/1.5σ/2σ containment 87.99%/96.75%/98.70%.
- stress: 541 episodes, support gate `pass`; remains-below-IV 89.83%; mean/median residual -0.004570/-0.004666; 15-minute 1σ/1.5σ/2σ containment 90.76%/97.97%/99.63%.

## Nulls and bootstrap

- assessment: matched-null wins on NPV lift/mean residual 20/20 and 20/20; permutation wins 10/10 and 10/10; 80% NPV-lift interval [0.150089, 0.185606]; 80% mean-residual interval [-0.006012, -0.005198].
- stress: matched-null wins on NPV lift/mean residual 20/20 and 20/20; permutation wins 10/10 and 10/10; 80% NPV-lift interval [0.166292, 0.192396]; 80% mean-residual interval [-0.005267, -0.004613].

## Binding gates

- Long-premium veto gate: `fail`.
- Short-premium range-containment readiness gate: `fail`.
- Prospective defined-risk short-premium shadow recording: `not prioritised`.
- Naked short options, paper orders, live orders, and strategy deployment remain unauthorised.

## Plots

- `research/options-feasibility/20260727-m1c-low-movement-short-premium-v0/reports/m1c_decile_movement_exceeds_iv.png`
- `research/options-feasibility/20260727-m1c-low-movement-short-premium-v0/reports/bottom_tail_vs_population_iv_residual.png`
- `research/options-feasibility/20260727-m1c-low-movement-short-premium-v0/reports/m1c_vs_m0_containment_surprise.png`
- `research/options-feasibility/20260727-m1c-low-movement-short-premium-v0/reports/fresh_episode_maximum_excursion.png`
