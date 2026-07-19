# Prospective ledgers

This directory is reserved for hash-bound, append-only prospective predictions and their
separate settlements. Historical data cannot satisfy the prospective structural gate.

No prospective freeze exists for the current blocked run. The scoring command refuses to
write without an authorised freeze, and settlement refuses to write before the frozen
outcome-availability timestamp. Prediction records reject future-return, target, P&L,
cost, spread, and slippage fields at prediction time.

IBKR observations, if a later authorised external scheduler captures them, remain quote
observations rather than fills. The V1 observer cannot place, preview, stage, cancel, or
query orders and cannot request account, position, portfolio, balance, or execution data.
