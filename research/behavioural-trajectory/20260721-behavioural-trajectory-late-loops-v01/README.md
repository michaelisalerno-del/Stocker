# Behavioural-Trajectory Funnel V0.1 — Corrected Anchors and Later Loops

This is a retrospective, observable-only, structural feasibility screen. It repairs the
invalid odd middle anchor in the blocked V0 and adds fixed later-session checkpoints.
It does not inspect economic outcomes, enable execution, or support strategy promotion.

The fixed completed-bar anchors are `2/4/6`, `4/8/12`, `8/16/24`, and `12/24/36` for
decision ordinals 6, 12, 24, and 36. The target is the frozen three-class first event in
the next six completed five-minute bars. T0 is the combined-clock levels/regime baseline,
T1 adds 18 fixed trajectory predictors, and T2 adds six preregistered trajectory/regime
interactions.

Run the bounded screen with:

```bash
uv run python research/behavioural-trajectory/20260721-behavioural-trajectory-late-loops-v01/run_screen_v01.py
```

Primary artifacts are written beneath `artifacts/primary`; the readable report is also
copied to `reports/report.md`.
