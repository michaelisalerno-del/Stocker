# Stock-Local Directional Archetype Screen V0

## Decision

`archetype_agreement_descriptive_only`

This is retrospective directional candidate research on underlying-stock returns. It is not option P&L, direct order-flow measurement, prospective validation, or a deployable strategy.

## Causal movement gate

Archived M1 was numerically affected by the future-filtered peer-slate lineage. M1C therefore removed signed pressure, tension, and all other peer-normalised Group I inputs without replacement.

- Frozen 2024 weighted 95th-percentile M1C threshold: `0.488333710794033`
- Development episodes: `474`
- Assessment episodes: `417`

## Assessment proper scores

| Model | Log loss | Brier | AUC | Average precision | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.704227 | 0.255342 | 0.497982 | 0.546904 | 0.5061 | 0.5057 |
| C1 | 0.705692 | 0.255897 | 0.512773 | 0.545766 | 0.5231 | 0.5221 |
| A1 | 0.719208 | 0.262041 | 0.501211 | 0.558256 | 0.4842 | 0.4843 |
| R1 | 0.707906 | 0.256902 | 0.509141 | 0.545045 | 0.4964 | 0.4962 |

## Frozen selective policies (ten minutes)

| Archetype | Actions | Coverage | CALL | PUT | Accuracy | Mean aligned return | Median aligned return |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 121 | 0.290 | 54 | 67 | 0.5124 | 0.000709 | 0.000303 |
| A1 | 140 | 0.336 | 66 | 74 | 0.5252 | 0.001761 | 0.000508 |
| R1 | 103 | 0.247 | 47 | 56 | 0.5098 | -0.000145 | 0.000287 |

## Falsification

- C1: beat 8/10 label nulls on log loss or AUC and 7/10 on mean aligned return; temporal-placebo pass = `False`.
- A1: beat 5/10 label nulls on log loss or AUC and 9/10 on mean aligned return; temporal-placebo pass = `False`.
- R1: beat 6/10 label nulls on log loss or AUC and 4/10 on mean aligned return; temporal-placebo pass = `True`.

## M1C gate checks

| Period | Model | Log loss | Brier | AUC | Average precision |
|---|---|---:|---:|---:|---:|
| assessment | M0 | 0.566415 | 0.190300 | 0.611423 | 0.354785 |
| assessment | M1C | 0.548986 | 0.183409 | 0.661952 | 0.414962 |
| opened_retrospective_stress | M0 | 0.564914 | 0.189450 | 0.608198 | 0.353187 |
| opened_retrospective_stress | M1C | 0.545451 | 0.182268 | 0.666107 | 0.405364 |

M1C improved log loss, Brier, AUC, and average precision versus M0 in both the assessment period and the explicitly opened movement-gate stress period.

## Secondary selective horizons

| Archetype | Horizon | Accuracy | Mean aligned return | Median aligned return |
|---|---:|---:|---:|---:|
| C1 | 5m | 0.4583 | -0.001126 | -0.000789 |
| C1 | 10m | 0.5124 | 0.000709 | 0.000303 |
| C1 | 15m | 0.5333 | 0.000395 | 0.000744 |
| C1 | 30m | 0.4380 | 0.000142 | -0.001371 |
| A1 | 5m | 0.4892 | 0.000954 | -0.000271 |
| A1 | 10m | 0.5252 | 0.001761 | 0.000508 |
| A1 | 15m | 0.4929 | 0.001212 | -0.000397 |
| A1 | 30m | 0.4857 | 0.001192 | -0.000714 |
| R1 | 5m | 0.4118 | -0.001260 | -0.001713 |
| R1 | 10m | 0.5098 | -0.000145 | 0.000287 |
| R1 | 15m | 0.5490 | 0.000394 | 0.000578 |
| R1 | 30m | 0.4466 | -0.001824 | -0.002035 |

## Material movement and remaining movement

| Archetype | Subgroup | Actions | Accuracy | Mean aligned return | Remaining fraction |
|---|---|---:|---:|---:|---:|
| C1 | ten_minute_iv_excess | 52 | 0.4808 | 0.001663 | 0.7050 |
| C1 | non_iv_excess | 69 | 0.5362 | -0.000011 | 0.4230 |
| C1 | largest_absolute_movement_quartile | 25 | 0.5600 | 0.005513 | 0.7477 |
| A1 | ten_minute_iv_excess | 75 | 0.5600 | 0.003577 | 0.6730 |
| A1 | non_iv_excess | 65 | 0.4844 | -0.000335 | 0.4114 |
| A1 | largest_absolute_movement_quartile | 36 | 0.6667 | 0.008523 | 0.7122 |
| R1 | ten_minute_iv_excess | 45 | 0.5111 | 0.000242 | 0.6874 |
| R1 | non_iv_excess | 58 | 0.5088 | -0.000445 | 0.4335 |
| R1 | largest_absolute_movement_quartile | 18 | 0.5556 | 0.001769 | 0.7375 |

Binding remaining-movement results:
- C1: mean `0.5442`, median `0.5578`, late-direction problem = `False`.
- A1: mean `0.5515`, median `0.5724`, late-direction problem = `False`.
- R1: mean `0.5445`, median `0.5895`, late-direction problem = `False`.

## Baselines and stability

Assessment-wide directional accuracy: always UP `0.5255`, ten-minute momentum `0.5085`, market direction `0.4745`, simple relative strength `0.5158`, and beta-adjusted residual `0.4988`.
- C1: positive mean aligned return in `5/8` month groups.
- A1: positive mean aligned return in `6/8` month groups.
- R1: positive mean aligned return in `4/8` month groups.

Action concentration maxima:
- C1: stock `0.1405`, month `0.1488`, session `0.0496`.
- A1: stock `0.0786`, month `0.1714`, session `0.0429`.
- R1: stock `0.1165`, month `0.1553`, session `0.0583`.

## Agreement and conflict (descriptive only)

| Category | Episodes | Accuracy | Mean aligned return |
|---|---:|---:|---:|
| Continuation only | 41 | 0.5122 | 0.001783 |
| Absorption/reversal only | 65 | 0.4923 | 0.002555 |
| Relative strength only | 29 | 0.5172 | -0.000634 |
| Two archetypes agree | 60 | 0.4915 | 0.001717 |
| All three agree | 34 | 0.5588 | -0.001065 |
| Archetypes conflict | 3 | NA | NA |
| All abstain | 185 | NA | NA |

## Bootstrap and candidate gates

- C1: 80% accuracy interval `[0.4543, 0.5903]`; 80% mean aligned-return interval `[-0.001330, 0.003253]`; status `not_supported`.
- A1: 80% accuracy interval `[0.4683, 0.5888]`; 80% mean aligned-return interval `[0.000316, 0.003436]`; status `not_supported`.
- R1: 80% accuracy interval `[0.4444, 0.5833]`; 80% mean aligned-return interval `[-0.002170, 0.002065]`; status `not_supported`.

All direction features ended at T-1. Trigger bar T was excluded. No peer-slate normalisation or archived signed-pressure value was used.

The independent audit reconstructed 100 feature rows, probabilities, actions, targets, and causal-gate probabilities within the `1e-12` tolerance. Determinism reported zero episode or action mismatches.
