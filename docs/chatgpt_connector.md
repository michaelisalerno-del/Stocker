# ChatGPT Stocker Connector

This guide connects the local read-only Stocker MCP server to ChatGPT as a custom
connector/app. It lets ChatGPT inspect Stocker code, historical bars, database
summaries, research results, reports, and hypotheses through narrow MCP tools.

It does not provide trading, broker access, IBKR access, order placement, live
execution, paper execution, arbitrary shell access, arbitrary writes, file deletion,
or secret access.

## Safety Model

The connector exposes only read-only MCP tools. All filesystem access is scoped to:

- The Stocker repo root
- `STOCKER_HOME`, normally `$HOME/StockerLocal`

The server blocks:

- `.env` and `.env.*`
- Private keys, SSH keys, token files, credential files, and API-key-like filenames
- Path traversal and symlink escapes
- Reads outside the repo root and `STOCKER_HOME`
- Raw DB file downloads
- Raw vendor payload dumps
- Huge parquet, zip, and database dumps
- SQL writes and unsafe SQL features

Responses are bounded and secret-redacted. Tool calls are logged locally without token
values.

## Start Local HTTP MCP

```bash
cd ~/Code/Stocker
export STOCKER_HOME="$HOME/StockerLocal"
export STOCKER_MCP_TOKEN="$(openssl rand -hex 32)"

uv run stocker-mcp \
  --transport http \
  --host 127.0.0.1 \
  --port 8765 \
  --auth-mode oauth \
  --auth-token-env STOCKER_MCP_TOKEN
```

The local MCP endpoint is:

```text
http://127.0.0.1:8765/mcp
```

For ChatGPT, HTTP mode uses OAuth. `STOCKER_MCP_TOKEN` is used as the local setup
code on the OAuth approval page. It is not printed by the server and should not be
committed or pasted into docs.

Bearer-token mode remains available for local clients with `--auth-mode bearer`, but
ChatGPT custom connectors should use OAuth.

## Connector Info

Print setup metadata without exposing secrets:

```bash
uv run stocker-mcp connector-info
```

This prints:

- Local URL: `http://127.0.0.1:8765/mcp`
- Tunnel URL placeholder: `https://YOUR-TUNNEL-URL/mcp`
- Connector name: `Stocker Research`
- Connector description
- OAuth metadata locations without setup-code values
- Exposed tool names
- Security mode
- Docs path

Run diagnostics:

```bash
uv run stocker-mcp doctor
```

## Tunnel Options

ChatGPT needs a reachable HTTPS MCP endpoint. Keep the Stocker server bound to
`127.0.0.1`; tunnel the local `/mcp` endpoint instead of binding publicly.

### Preferred: OpenAI Secure MCP Tunnel

Use OpenAI Secure MCP Tunnel if it is available in your environment.

This repo check found no local `tunnel-client` binary on `PATH`, so no Secure MCP
Tunnel command was installed or started here.

When available, start the tunnel to:

```text
http://127.0.0.1:8765/mcp
```

Use the returned HTTPS tunnel URL as:

```text
https://YOUR-TUNNEL-URL/mcp
```

### Fallback: ngrok

If `ngrok` is already installed and approved for your machine:

```bash
ngrok http 8765
```

Use the HTTPS forwarding URL and append `/mcp`:

```text
https://YOUR-NGROK-URL/mcp
```

### Fallback: Cloudflare Tunnel

If `cloudflared` is already installed and approved for your machine:

```bash
cloudflared tunnel --url http://127.0.0.1:8765
```

Use the HTTPS forwarding URL and append `/mcp`:

```text
https://YOUR-CLOUDFLARE-URL/mcp
```

Do not commit tunnel URLs or auth tokens. Stop temporary tunnels when finished.

## Create The ChatGPT Connector

In ChatGPT web:

1. Open Settings.
2. Open Apps & Connectors.
3. Open Advanced settings.
4. Enable Developer mode.
5. Go to Settings -> Connectors -> Create.
6. Use these fields:

```text
Connector name: Stocker Research
Description: Read-only connector for Stocker historical research, code, reports, bars, database summaries, and hypothesis analysis. No trading or execution.
Connector URL: https://YOUR-TUNNEL-URL/mcp
Authentication: OAuth
```

If ChatGPT shows your callback URL, it can be registered dynamically by Stocker MCP.
For example:

```text
Callback URL:
https://chatgpt.com/connector/oauth/LcDotGjs8ocx
```

If the UI asks you to enter OAuth settings manually, use:

```text
Authorization URL:
https://YOUR-TUNNEL-URL/oauth/authorize

Token URL:
https://YOUR-TUNNEL-URL/oauth/token

Registration URL:
https://YOUR-TUNNEL-URL/oauth/register

Client setup:
Dynamic Client Registration, if available

Scopes:
stocker.read

Token endpoint authentication:
None / public client

PKCE:
S256
```

When ChatGPT starts the OAuth flow, the Stocker approval page asks for a setup code.
Paste the value of `STOCKER_MCP_TOKEN` there. Do not paste it into the connector
description, docs, screenshots, or chat messages.

If ChatGPT offers a client setup method, use Dynamic Client Registration or discovered
OAuth settings from the server metadata.

## First Prompts

After connecting:

1. Open a new ChatGPT chat.
2. Click `+`.
3. Click `More`.
4. Select `Stocker Research`.
5. Try:

```text
Use Stocker Research to summarise the latest research state.
```

Other useful prompts:

```text
Use Stocker Research to list available tools.
Use Stocker Research to run workspace_doctor.
Use Stocker Research to summarise the latest universe run.
Use Stocker Research to find positive rejected symbols.
Use Stocker Research to search for VWAP.
Use Stocker Research to fetch the latest VWAP run summary.
Use Stocker Research to list DB tables.
Use Stocker Research to summarise NVDA 5m bars from the latest available range.
```

## Mobile

After linking the connector on ChatGPT web, it should be available in ChatGPT mobile
as a connected app/tool if your account, plan, region, and mobile client version
support custom connectors. If it does not appear on mobile, verify it works on web
first and refresh the mobile app session.

## Refresh Tool Metadata

If tools do not show up after changing the server:

1. Stop the Stocker MCP server.
2. Stop the tunnel.
3. Restart the Stocker MCP server.
4. Restart the tunnel.
5. Refresh or recreate the ChatGPT connector.
6. Start a new ChatGPT chat and reselect `Stocker Research`.

## Stop The Server And Tunnel

Stop the local MCP server with `Ctrl-C` in the terminal where it is running.
Stop your tunnel with `Ctrl-C` in its terminal, or use the tunnel provider's stop
command if it runs in the background.

Unset the token when finished:

```bash
unset STOCKER_MCP_TOKEN
```

## Troubleshooting

ChatGPT cannot connect:

- Confirm the local server is running.
- Confirm the tunnel forwards to `http://127.0.0.1:8765`.
- Confirm the connector URL ends with `/mcp`.
- Confirm the URL is HTTPS, not HTTP.

Tunnel URL wrong:

- Use the public HTTPS forwarding URL from the tunnel output.
- Append `/mcp`.
- Do not use `localhost` in the ChatGPT connector URL.

`/mcp` missing:

- Start Stocker with `--transport http`.
- Confirm `uv run stocker-mcp doctor` reports `http.local_url`.
- Confirm the tunnel forwards port `8765`.

Auth failure:

- Confirm `STOCKER_MCP_TOKEN` was set before starting the server.
- Confirm the server was started with `--auth-mode oauth`.
- Confirm ChatGPT discovered the OAuth metadata.
- Paste `STOCKER_MCP_TOKEN` only into the Stocker OAuth approval page.
- Restart the server after changing the setup code.

Server bound only to localhost:

- This is intentional. Keep it bound to `127.0.0.1`.
- Use a protected HTTPS tunnel for ChatGPT.

Tunnel not forwarding auth headers:

- Prefer a tunnel that preserves request headers.
- Recreate the connector after changing auth settings.

Tools not showing:

- Run `uv run stocker-mcp tools`.
- Run `uv run stocker-mcp connector-info`.
- Refresh connector metadata in ChatGPT.
- Recreate the connector if stale metadata persists.

Mobile not showing connector:

- Complete setup on ChatGPT web first.
- Confirm the same account is used on mobile.
- Refresh or reinstall the mobile app if the connector list is stale.

Connector UI unavailable:

- Check ChatGPT plan, region, workspace admin settings, and Developer mode availability.
- Some connector features may be limited by account, workspace, region, or client rollout.

## What Not To Expose

Do not expose:

- `.env`
- API keys
- Broker credentials
- IBKR credentials
- Public unauthenticated endpoints
- Raw local database files
- Raw vendor payloads
- Order placement tools
- Broker tools
- Live or paper execution tools
- Arbitrary shell or filesystem tools
