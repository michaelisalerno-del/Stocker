# M1C Asymmetric Downside Residual V1

This preregistered retrospective experiment asks whether the unchanged frozen
M1C movement gate can be combined with one stock-local downside model:

- high `q_down` proposes `PUT`;
- low `q_down`, only behind the high-M1C gate, proposes `CALL`;
- the middle score region abstains.

Low downside probability is a hypothesis to test, not an upside label. The
policy is evaluated over all eligible fresh episodes, including M1C false
positives and no-move outcomes.

The primary horizon is the frozen M1C 15-minute endpoint horizon. The fixed
predictors are the latest complete five-minute return, trailing 15-minute
return, trailing 15-minute close location, and causal session-VWAP distance
scaled by the previous-close implied 15-minute movement. No direction feature
selection or hyperparameter search is permitted.

The canonical M1C target uses a strict `absolute_return > implied_movement`
comparison. The requested directional partition assigns exact equality to
`UP_MOVE` or `DOWN_MOVE`. Therefore the literal probability complement is not
claimed; conditional `q_down` and frozen M1C probability are reported
separately.

Run:

```bash
rtk uv run python \
  research/directional-readiness/20260728-m1c-asymmetric-downside-residual-v1/run_experiment.py
```

The runner reuses the hash-verified frozen M1C/A1/Tail Phase episode identities
and reads only the registered 2024–2025 primitive bars and previous-close
option context needed to reconstruct the fixed target and stock-local
predictors. It does not fit or modify M1C, A1, or Tail Phase V1, import an
execution application, or expose an order path.

Independently audit emitted targets, scores, thresholds, actions, frozen-system
parity, output hashes, protected-data guards, and execution guards:

```bash
rtk uv run python \
  research/directional-readiness/20260728-m1c-asymmetric-downside-residual-v1/audit_experiment.py
```
