# CI formatting scope and inherited debt

The prospective recorder gate formats every changed Python file under
`packages/stocker_prospective/` and every changed `test_prospective_*` or
`test_m1c_*` test. Ruff lint, strict mypy, the prospective/M1C test suite, and
the import/CLI smoke test are separate blocking steps, so a formatting failure
cannot prevent those results from being observed.

The repository-wide lint, type-check, and test commands remain intact in a
separate explicit debt-report job. Each command runs even if an earlier command
fails, and each outcome is written to the GitHub Actions summary. That job is
non-blocking while inherited failures remain, and its summary says `NOT
PASSING`; it is not represented as a successful repository-wide gate. The
full-repository Ruff formatting check likewise remains as a separate
non-blocking report whose output is uploaded as an artifact.

At the inspected base commit
`3442744cf183ad7669ee85de47e2a8c5e70f0bdd`, `ruff format --check .` reported
178 files that would be reformatted. Those files include archival research
material outside the prospective recorder. This hardening change does not
mass-format that inherited archive merely to make CI appear green.

The same base also has repository-wide lint, typing, and fixture-availability
debt that was previously hidden when formatting stopped the workflow. On
2026-07-29, the exact commands now retained in the report produced:

- `ruff check .`: 1,153 findings;
- `mypy packages apps`: 135 errors in 26 files;
- `pytest`: 1,607 passed, 1 skipped, 13 failed, and 19 errored. Every listed
  failure/error depended on absent frozen archival artifact files.

Those counts are an inspected-base inventory, not an allow-list. The workflow
always reruns the commands at the exact revision and reports their current
outcomes. Only the focused prospective-recorder gate is required to pass in
this change.
