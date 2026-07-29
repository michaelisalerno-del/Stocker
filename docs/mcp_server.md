# Stocker MCP Server

Stocker includes a local read-only MCP server for AI clients that need narrow access to
Stocker code, StockerLocal research reports, and local Stocker database summaries.
It is infrastructure only. It does not place orders, connect to brokers, fetch vendor
data, run research scans, or expose arbitrary filesystem access.

The server supports:

- Local stdio MCP for Codex and other local agents
- Local authenticated Streamable HTTP MCP at `/mcp` for ChatGPT connector tunnels
- ChatGPT-friendly `search` and `fetch` tools
- Read-only tool annotations for every exposed tool

## Roots

The server resolves two allowed roots:

- Repo root: the current Stocker project directory containing `pyproject.toml` and
  `packages/`.
- Stocker workspace: `STOCKER_HOME`, or `~/StockerLocal` when the variable is unset.

Report tools search these roots:

- `$STOCKER_HOME/data/reports/research`
- `data/reports/research` inside the repo, when present

Database tools are scoped to:

- `$STOCKER_HOME/db`

`db_get_symbol_bars` can also read bounded rows from Stocker's canonical bar
partitions under
`$STOCKER_HOME/data/processed/source=*/instrument_type=*/symbol=*/timeframe=*/data.parquet`.
This is limited to the requested symbol and timeframe; arbitrary Parquet file
reads remain blocked.

Exports are written only under:

- `$STOCKER_HOME/exports`

## Launch

Stdio remains the default:

```bash
export STOCKER_HOME="$HOME/StockerLocal"
uv run stocker-mcp --help
uv run stocker-mcp doctor
uv run stocker-mcp
```

Start local HTTP mode for a connector tunnel:

```bash
export STOCKER_HOME="$HOME/StockerLocal"
export STOCKER_MCP_TOKEN="$(openssl rand -hex 32)"

uv run stocker-mcp \
  --transport http \
  --host 127.0.0.1 \
  --port 8765 \
  --auth-mode oauth \
  --auth-token-env STOCKER_MCP_TOKEN
```

HTTP mode binds to `127.0.0.1` by default. For ChatGPT connectors, use
`--auth-mode oauth`. `STOCKER_MCP_TOKEN` acts as a local setup code on the OAuth
approval page. The MCP endpoint is:

```text
http://127.0.0.1:8765/mcp
```

Do not bind to `0.0.0.0` unless you have a protected private network reason.

If ChatGPT reaches the server through an HTTPS tunnel, allowlist the exact tunnel
host while keeping the bind address on loopback:

```bash
uv run stocker-mcp \
  --transport http \
  --host 127.0.0.1 \
  --port 8765 \
  --auth-mode oauth \
  --auth-token-env STOCKER_MCP_TOKEN \
  --allowed-host YOUR-TUNNEL-HOST
```

You can also set `STOCKER_MCP_ALLOWED_HOSTS="YOUR-TUNNEL-HOST"`. Do not use
wildcards.

Bearer-token mode remains available for local clients:

```bash
uv run stocker-mcp \
  --transport http \
  --host 127.0.0.1 \
  --port 8765 \
  --auth-mode bearer \
  --auth-token-env STOCKER_MCP_TOKEN
```

## Helper Commands

```bash
uv run stocker-mcp --help
uv run stocker-mcp doctor
uv run stocker-mcp connector-info
uv run stocker-mcp tools
```

`connector-info` prints:

- Local MCP URL
- Expected HTTPS tunnel URL placeholder
- Connector name and description
- OAuth metadata locations or bearer auth header format without secret values
- Tool names
- Security mode
- Docs path

`doctor` checks:

- Repo root
- `STOCKER_HOME`
- Report, DB, and export roots
- Auth env var presence without printing the value
- HTTP dependency availability
- Unsafe bind-address warning
- Tool count
- Stdio and HTTP initialization

## OAuth Endpoints

OAuth mode exposes local metadata and auth endpoints for ChatGPT:

- `/.well-known/oauth-protected-resource`
- `/.well-known/oauth-authorization-server`
- `/oauth/register`
- `/oauth/authorize`
- `/oauth/token`
- `/oauth/revoke`

The OAuth implementation is local and in-memory:

- Dynamic client registration is supported.
- Authorization code with PKCE `S256` is supported.
- Refresh tokens are supported for the current server process.
- Access and refresh tokens are not written to disk.
- The setup code is read from an environment variable and is never printed by helper
  commands.

## Codex Registration

This machine uses Codex TOML MCP config at:

```text
~/.codex/config.toml
```

Use this local stdio entry:

```toml
[mcp_servers.stocker]
command = "uv"
args = ["run", "stocker-mcp"]
cwd = "/Users/michaelsalerno/Documents/Codex/2026-06-29-we-are-working-in-my-stocker"
startup_timeout_sec = 120

[mcp_servers.stocker.env]
STOCKER_HOME = "/Users/michaelsalerno/StockerLocal"
```

Do not add API keys or broker settings to the MCP server environment.

## Tools

Discovery:

- `search`
- `fetch`

Code and git:

- `get_repo_info`
- `git_status`
- `git_log`
- `git_current_commit`
- `list_files`
- `search_code`
- `read_code_file`
- `git_diff`

Workspace and reports:

- `workspace_doctor`
- `list_recent_research_runs`
- `get_latest_universe_run`
- `read_universe_run`
- `summarise_universe_run`
- `list_symbol_reports`
- `read_symbol_report`
- `compare_universe_runs`
- `find_candidate_symbols`
- `find_interesting_symbols`
- `filter_symbol_results`

Discovery workflow summaries:

- `summarise_latest_research_state`
- `find_positive_rejected_symbols`
- `find_null_pass_benchmark_fail_symbols`
- `find_benchmark_pass_rejected_symbols`
- `find_common_rejection_reasons`
- `get_symbol_bar_summary`
- `get_symbol_recent_sessions`
- `get_trade_feature_buckets`
- `compare_template_runs`
- `suggest_research_questions`

Database:

- `db_list_databases`
- `db_list_tables`
- `db_describe_table`
- `db_preview_table`
- `db_get_symbol_bars`
- `db_get_latest_catalysts`
- `db_get_trade_attribution`
- `db_select`

Diagnostics:

- `export_diagnostics_zip`

Resources:

- `stocker://workspace/doctor`
- `stocker://reports/latest`
- `stocker://repo/status`

Prompts:

- `summarise_latest_stocker_scan`
- `compare_two_stocker_universe_runs`
- `investigate_symbol_gate_failure`

## Search And Fetch

`search` returns results shaped for ChatGPT:

```json
{
  "results": [
    {
      "id": "stocker://reports/home/universe%2Frun.json",
      "title": "run.json",
      "url": ""
    }
  ]
}
```

`fetch` accepts only validated `stocker://` IDs:

- `stocker://reports/...`
- `stocker://runs/...`
- `stocker://symbols/...`
- `stocker://code/...`
- `stocker://hypotheses/...`
- `stocker://db/...`
- `stocker://workspace/...`

It returns safe redacted text plus metadata. It does not accept raw filesystem paths,
`file://` URLs, path traversal, `.env`, credential files, or arbitrary database files.

## Security

The server is read-only except for a local redacted audit log and diagnostics zip
exports under `$STOCKER_HOME/exports`.

Every MCP tool is registered with `readOnlyHint: true`, `destructiveHint: false`, and
`openWorldHint: false`.

It blocks:

- Paths outside the allowed roots
- Path traversal
- Symlink escapes
- `.env`, `.env.*`, private keys, SSH keys, credential, token, and API-key-like files
- `.git`, virtualenvs, caches, and local tool runtime directories
- Arbitrary shell commands
- File write, patch, delete, rename, and move operations
- Broker, IBKR, order, live execution, paper execution, and vendor-fetch tools
- Raw database file download
- Raw vendor payload dumps
- Huge parquet, zip, and database dumps
- Arbitrary Parquet reads outside the `db_get_symbol_bars` canonical bar path

Text output is bounded and redacted for secret-looking values such as API keys,
tokens, bearer headers, passwords, high-entropy strings, and credential-like database
columns.

`db_select` is restricted:

- SELECT only
- No semicolons
- No `ATTACH`, `DETACH`, `COPY`, `EXPORT`, `INSTALL`, `LOAD`, `PRAGMA`, `CREATE`,
  `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, or similar statements
- No file-reading functions such as `read_csv`, `read_parquet`, or `parquet_scan`
- Forced row limit
- Read-only SQLite and DuckDB connections where supported

Disable DB querying with:

```bash
export STOCKER_MCP_DISABLE_DB=1
uv run stocker-mcp
```

## Diagnostics Export

From an MCP client, call:

```text
export_diagnostics_zip()
```

The zip includes workspace diagnostics, git status, recent git log, latest universe
run summaries, selected report JSON/Markdown files, DB schema summaries, and safe
config examples. It excludes `.env`, API keys, database files, parquet/raw vendor
payloads, broker state, and huge data files.

## MCP Inspector

If the MCP Inspector is installed, use it against stdio:

```bash
npx @modelcontextprotocol/inspector uv run stocker-mcp
```

For HTTP mode, start the server first and point the inspector at:

```text
http://127.0.0.1:8765/mcp
```

Include:

```text
Authorization: Bearer $STOCKER_MCP_TOKEN
```

## Future Tools

Add future MCP tools only through the central security context. New tools should have
focused tests first, avoid broker/order/vendor-fetch behavior unless explicitly
approved, and return summaries or bounded slices instead of raw dumps.
