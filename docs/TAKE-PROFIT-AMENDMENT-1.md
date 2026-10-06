# TAKE-PROFIT-PROTOCOL — amendment 1: the optimistic maker variant was mis-specified (2026-10-06, before any judged result)

Amends `TAKE-PROFIT-PROTOCOL.md` (sha256 18017a9ffb32cebc52d8be67aad1fa12137937a9c0baa556e2987d52262238b4). Only a reported,
unjudged column changes; rule TP, its fill, its verdict and every judged column are unchanged. No judged clock (09:00 New
York, 2026-10-06 onward) had been recorded when this was written. The hash of this file is in
`TAKE-PROFIT-AMENDMENT-1.sha256`.

## What was wrong
The protocol reported an optimistic maker estimate that also filled the resting sell "when the ask is at or below the
price on 3 consecutive quotes with volume rising". For a sell resting at 1.5–2 times the entry ask, the market's ask is
below that price almost whenever the option has not risen, so the condition fired on falling options and sold them at
the target they never reached. On the look (1–6 October, not judged) it showed a mean saving of +47%, which is an
artefact. The condition was copied from Report 4 without checking its direction.

## Correction
`tp_maker_saving`: the strict fill of the protocol, or else a new trade printed at the resting price itself (the strict
rule counts only prints strictly above it), as if the order stood first in the queue at that price; half of such
at-price fills are kept by a coin seeded on the clock id. It remains an optimistic bound, reported and not judged.

`tp.py` sha256 61de82b367c55490894a6d22e3fcbebac9b4aaf08fcbc3d774268bbbaeccc6f3 (was
e09b684e8d08679fc60acbb83b1ae159f9944b29a666a470ba907e10c5fb8eb8); the change is the function `maker_fill` and its line
in the module docstring. On the look the corrected column differs from the strict one on 1 of 97 clocks.
