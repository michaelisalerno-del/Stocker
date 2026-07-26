# Movement-Qualified Directional Readiness Quick Screen V0

## Claims boundary

This is retrospective directional candidate evidence on underlying-stock returns. The frozen M1 movement model is unchanged and used only as an eligibility gate. No option P&L, intraday option quotes, broker access, execution claim, prospective validation, or deployable strategy claim is made.

## Frozen movement gate and episodes

- M1 reconstruction passed: `True`.
- Frozen threshold: `0.49588519865576763`.
- Raw above-threshold checkpoint rows: 1,266.
- Fresh 30-minute-spaced episodes: 538.
- Episodes per session: 2.1520.
- Development support passed: `True`; 285 episodes, 140 UP / 142 DOWN.
- Assessment support passed: `True`; 253 episodes, 141 UP / 109 DOWN.

## Direction models — 2025-01-01 through 2025-08-22

| Model | Log loss | Brier | AUC | AP | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.737800 | 0.269808 | 0.478691 | 0.549470 | 0.508000 | 0.490956 |
| D1 | 0.736848 | 0.268815 | 0.502245 | 0.573236 | 0.532000 | 0.515356 |
| D2 | 0.742957 | 0.270224 | 0.500358 | 0.587979 | 0.504000 | 0.491574 |

## Layer attribution

- D1_minus_D0: log-loss improvement 0.000952, Brier improvement 0.000993, AUC increment 0.023554, selective-accuracy increment 0.007691, aligned-return increment 0.00002069; adds value `False`.
- D2_minus_D1: log-loss improvement -0.006109, Brier improvement -0.001409, AUC increment -0.001887, selective-accuracy increment -0.004052, aligned-return increment -0.00078050; adds value `False`.

## Frozen CALL / PUT / ABSTAIN policy

- Primary candidate frozen before assessment scoring: `D2`.
- OOF confidence boundary: 0.09484478.
- Actions: 134 (83 CALL / 51 PUT); coverage 52.96%.
- Ten-minute directional accuracy: 50.76%; balanced accuracy 47.89%.
- Mean / median aligned ten-minute return: 0.00014715 / 0.00020412.
- Positive aligned-return rate: 50.00%.
- Selective support passed: `True`.

Secondary horizon results remain descriptive:

| Horizon | Accuracy | Mean aligned return | Median aligned return |
|---:|---:|---:|---:|
| 5 | 0.5489 | 0.00092743 | 0.00144766 |
| 10 | 0.5076 | 0.00014715 | 0.00020412 |
| 15 | 0.5113 | 0.00015537 | 0.00043016 |
| 30 | 0.5224 | 0.00159110 | 0.00181219 |

## Economic-readiness diagnostics on the underlying

- Realised IV-excess ten-minute selective accuracy / mean aligned return: 0.4925 / -0.00006332.
- Largest-movement-quartile selective accuracy / mean aligned return: 0.5652 / 0.00182511.
- Mean absolute remaining-movement fraction at ten minutes: 0.2102; late-direction problem `True`.
- Positive mean aligned return in 5 of eight assessment months.
- Recent-ten-minute momentum accuracy: 0.5120; market-direction accuracy: 0.5560.

## Uncertainty and nulls

- Selective accuracy 80% interval: [0.4504, 0.5617].
- Mean aligned return 80% interval: [-0.00120482, 0.00160900].
- Real candidate exceeded nulls on log loss 1/5 and AUC 2/5.

## Decision

Overall decision: `no_incremental_directional_signal`.

- movement_gate_status: `supported`.
- episode_construction_status: `supported`.
- price_direction_status: `not_supported`.
- signed_behaviour_status: `promising`.
- route_orientation_status: `not_supported`.
- selective_policy_status: `promising`.
- remaining_movement_status: `not_supported`.
- forward_readiness_status: `not_supported`.

Clean confirmation must occur prospectively through the live recorder. This screen does not establish option profitability or live/paper readiness.
