# Operational dashboard

Start standalone mode to inspect state and edit configuration without connecting or trading:

```bash
uv run stocker stage10-dashboard \
  --runs-config configs/runs.example.yaml \
  --ibkr-config configs/ibkr.example.yaml \
  --database .stocker/stage8-runtime.sqlite3
```

Open http://127.0.0.1:8000. The main workflow is **Market → Method → Run**.
Select US All, NASDAQ or NYSE, then **Session HARD**. There is no cap selector or
independent activity-screen/exit setting. Account risk and capacity are expandable.

**Start PAPER run** saves the method specification, version/hash and authoritative listing
snapshot. In standalone mode this persists configuration; it does not start execution.
The integrated `stage10-run` command attaches these controls to the runtime and enabled runs
retain their normal algorithmic PAPER authority. The current method does not support LIVE.

Run detail exposes universe/data/search status, qualification/veto/arming/entry counts,
active/completed positions and performance. Expand method specification and provenance for the
full saved configuration. Candidate details include P0, M, score, PRE_MOVE, whipsaw risk,
Q1 eligibility, both triggers, armed time, first-break direction, entry, exits and deadline.

Disabling stops future entries and preserves fills, history and broker protection.
Re-enabling an unchanged current run preserves its identity. Historical methods are readable
but cannot be enabled. Settings retain broker/account controls; they do not define method logic.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the method contract and prospective Q1 reconciliation.
