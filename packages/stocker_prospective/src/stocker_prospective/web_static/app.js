"use strict";

const state = {
  health: null,
  runtime: null,
  universe: null,
  signals: [],
  shadow: [],
  audit: [],
};

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function clean(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "YES" : "NO";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function short(value, length = 18) {
  const text = clean(value);
  return text.length > length ? `${text.slice(0, length)}…` : text;
}

function clock(value) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return clean(value);
  return parsed.toISOString().replace(".000Z", "Z");
}

function replace(targetId, child) {
  const target = document.getElementById(targetId);
  if (child.classList && child.classList.length === 1 && target.classList.contains(child.classList[0])) {
    target.replaceChildren(...child.childNodes);
  } else {
    target.replaceChildren(child);
  }
}

function metric(label, value, note, tone = "") {
  const card = node("article", "metric");
  card.append(node("span", "micro-label", label));
  const strong = node("strong", tone ? `value-${tone}` : "", clean(value));
  card.append(strong);
  if (note) card.append(node("small", "", note));
  return card;
}

function kvGrid(items) {
  const grid = node("div", "kv-grid");
  items.forEach(([label, value]) => {
    const item = node("div", "kv");
    item.append(node("span", "", label), node("strong", "", clean(value)));
    grid.append(item);
  });
  return grid;
}

function table(columns, rows, action) {
  if (!rows.length) return node("div", "empty-state", "No evidence rows recorded.");
  const wrapper = node("div", "table-wrap");
  const element = node("table");
  const head = node("thead");
  const headerRow = node("tr");
  columns.forEach((column) => headerRow.append(node("th", "", column.label)));
  if (action) headerRow.append(node("th", "", "Inspect"));
  head.append(headerRow);
  const body = node("tbody");
  rows.forEach((row) => {
    const tr = node("tr");
    columns.forEach((column) => {
      const raw = typeof column.value === "function" ? column.value(row) : row[column.value];
      tr.append(node("td", "", column.format ? column.format(raw) : clean(raw)));
    });
    if (action) {
      const cell = node("td");
      const button = node("button", "row-action", "Open");
      button.type = "button";
      button.addEventListener("click", () => action(row));
      cell.append(button);
      tr.append(cell);
    }
    body.append(tr);
  });
  element.append(head, body);
  wrapper.append(element);
  return wrapper;
}

function subsection(title, child) {
  const section = node("section", "subsection");
  section.append(node("h4", "", title), child);
  return section;
}

function jsonBlock(value) {
  return node("pre", "json-block", JSON.stringify(value, null, 2));
}

async function api(path) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    const error = new Error(`${response.status} ${path}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function showScreen(screenId) {
  document.querySelectorAll(".screen").forEach((screen) => {
    screen.classList.toggle("is-active", screen.id === screenId);
  });
  document.querySelectorAll(".nav-item").forEach((item) => {
    const active = item.dataset.screen === screenId;
    item.classList.toggle("is-active", active);
    item.setAttribute("aria-current", active ? "page" : "false");
  });
  document.getElementById(screenId).focus({ preventScroll: true });
}

function renderHealth() {
  const health = state.health;
  const runtime = state.runtime;
  const blockers = node("div", "blocker-strip");
  (health.blockers || []).forEach((code) => blockers.append(node("span", "blocker", code)));
  if (!(health.blockers || []).length) {
    blockers.append(node("span", "status-tag value-ok", "NO ACTIVE BLOCKERS"));
  }
  replace("blocker-strip", blockers);

  const grid = node("div", "metric-grid");
  const score = health.latest_score || {};
  const lease = health.recorder.lease || {};
  const context = health.previous_session_context || {};
  const capture = health.market_data.latest || {};
  const ibkrApi = health.ibkr_api || {};
  let ibkrApiState = `API ${clean(ibkrApi.api_version)} VERIFIED`;
  if (!ibkrApi.verified) {
    ibkrApiState = "BLOCKED";
  } else if (ibkrApi.blocker) {
    ibkrApiState = "UPDATE CHECK BLOCKED";
  } else if (ibkrApi.update_available === true) {
    ibkrApiState = "UPDATE REVIEW";
  } else if (ibkrApi.update_available === false) {
    ibkrApiState = `API ${clean(ibkrApi.api_version)} CURRENT`;
  }
  const ibkrApiNote = ibkrApi.blocker
    || (ibkrApi.update_available === null || ibkrApi.update_available === undefined
      ? "Official source verified; update check pending"
      : `Latest ${clean(ibkrApi.latest_api_version)}; automatic installation disabled`);
  [
    metric("Runtime state", health.status, "Fail-closed health", health.status === "blocked" ? "danger" : "ok"),
    metric("Instance", health.instance_identity, health.application.git_commit),
    metric("Recorder mode", health.recorder.mode, health.recorder.run_id),
    metric(
      "Recorder readiness",
      health.recorder.operational_status,
      "Operational state; scoring gates remain separate",
      ["active", "waiting_for_prospective_start"].includes(health.recorder.operational_status)
        ? "ok"
        : "danger",
    ),
    metric("Lease heartbeat", clock(lease.heartbeat_at_utc), lease.owner_id || "No owner", lease.owner_id ? "ok" : "danger"),
    metric("Active bundle", health.active_bundle.bundle_id || "UNAVAILABLE", health.active_bundle.verified ? "Verified" : "Verification blocked", health.active_bundle.verified ? "ok" : "danger"),
    metric("IBKR state", (health.ibkr || {}).state || "REPLAY / OFFLINE", (health.ibkr || {}).message || "No live broker dependency"),
    metric("Official IBKR API", ibkrApiState, ibkrApiNote, ibkrApi.verified && ibkrApi.update_status_fresh && !ibkrApi.blocker && ibkrApi.update_available !== true ? "ok" : "danger"),
    metric("Market data", capture.market_data_type || "synthetic", `${health.market_data.line_budget - health.market_data.reserved_headroom} usable lines`),
    metric("Database", health.database.status, health.database.mode, health.database.status === "healthy" ? "ok" : "danger"),
    metric("Anchor cohort", (state.universe || {}).anchor_count || 0, "Frozen membership; no pooling"),
    metric("Last bar", clock((health.last_completed_bar || {}).bar_end_utc), (health.last_completed_bar || {}).completeness || "No bar"),
    metric("Previous session", context.observation_date || "MISSING", context.eligibility ? "Exact context eligible" : clean(context.rejection_reason), context.eligibility ? "ok" : "danger"),
    metric("Latest episode", (health.latest_signal_episode || {}).id || "NONE", clock((health.latest_signal_episode || {}).crossing_timestamp_utc)),
  ].forEach((item) => grid.append(item));
  replace("runtime-grid", grid);

  const scorePanel = kvGrid([
    ["Score label", score.score_label],
    ["Symbol / cohort", score.symbol ? `${score.symbol} / ${score.cohort}` : null],
    ["M0 probability", score.m0_probability],
    ["M1 probability", score.m1_probability],
    ["Frozen threshold", score.frozen_threshold],
    ["Eligibility", score.eligibility],
    ["Feature as-of", clock(score.feature_as_of_utc)],
    ["Rejection", score.rejection_reason],
  ]);
  replace("score-panel", scorePanel);

  const anchor = ((state.universe || {}).cohorts || {}).anchor_frozen_20 || [];
  const universePanel = table(
    [
      { label: "Symbol", value: "symbol" },
      { label: "Operational state", value: "operational_status" },
      { label: "Rejection", value: "rejection_reason" },
    ],
    anchor,
  );
  replace("universe-panel", universePanel);
}

function renderSignals() {
  replace(
    "signal-list",
    table(
      [
        { label: "Symbol", value: "symbol" },
        { label: "Crossing", value: "crossing_timestamp_utc", format: clock },
        { label: "M1", value: "m1_probability" },
        { label: "Gate", value: "frozen_threshold" },
        { label: "Checkpoints", value: "checkpoint_count" },
        { label: "Label", value: "score_label", format: (value) => short(value, 22) },
      ],
      state.signals,
      (row) => loadSignal(row.id),
    ),
  );
}

async function loadSignal(signalId) {
  replace("signal-evidence", node("div", "empty-state", "Reading episode evidence…"));
  try {
    const detail = await api(`/api/signals/${encodeURIComponent(signalId)}`);
    const stack = node("div", "detail-stack");
    stack.append(
      subsection(
        "Episode identity",
        kvGrid([
          ["Signal ID", detail.episode.id],
          ["Cohort", detail.episode.cohort],
          ["Crossing", clock(detail.episode.crossing_timestamp_utc)],
          ["Bundle/model", detail.episode.model_bundle_id],
          ["Git commit", detail.episode.git_commit],
          ["Universe", detail.episode.universe_id],
          ["Recorded", clock(detail.episode.recorded_at_utc)],
          ["Source timestamps", detail.episode.source_timestamps_json],
        ]),
      ),
      subsection(
        "Above-threshold checkpoints",
        table(
          [
            { label: "Timestamp", value: "checkpoint_timestamp_utc", format: clock },
            { label: "M0", value: "m0_probability" },
            { label: "M1", value: "m1_probability" },
            { label: "Gate", value: "frozen_threshold" },
            { label: "Eligible", value: "eligibility" },
            { label: "Label", value: "score_label" },
          ],
          detail.checkpoints,
        ),
      ),
      subsection("Frozen H0 feature values / replay fixture", jsonBlock(detail.feature_snapshot || {})),
      subsection("Previous-session context", jsonBlock(detail.previous_session_context || {})),
      subsection(
        "Underlying captures",
        table(
          [
            { label: "Target", value: "target_timestamp_utc", format: clock },
            { label: "Actual", value: "actual_quote_timestamp_utc", format: clock },
            { label: "Lag s", value: "capture_lag_seconds" },
            { label: "Bid", value: "bid" },
            { label: "Ask", value: "ask" },
            { label: "Mid", value: "midpoint" },
            { label: "Data", value: "market_data_type" },
          ],
          detail.underlying_quotes,
        ),
      ),
      subsection(
        "DTE capture schedule",
        table(
          [
            { label: "Bucket", value: "dte_bucket" },
            { label: "Target", value: "target_timestamp_utc", format: clock },
            { label: "Actual", value: "actual_quote_timestamp_utc", format: clock },
            { label: "Lag s", value: "capture_lag_seconds" },
            { label: "Type", value: "market_data_type" },
            { label: "Freshness", value: "quote_freshness" },
            { label: "Status", value: "capture_status" },
            { label: "Missing", value: "missing_contract_reason" },
          ],
          detail.captures,
        ),
      ),
      subsection(
        "Bounded option surface",
        table(
          [
            { label: "Target", value: "target_timestamp_utc", format: clock },
            { label: "DTE", value: "dte_bucket" },
            { label: "Contract", value: "local_symbol" },
            { label: "Strike", value: "strike" },
            { label: "Right", value: "right" },
            { label: "Bid", value: "bid" },
            { label: "Ask", value: "ask" },
            { label: "Bid size", value: "bid_size" },
            { label: "Ask size", value: "ask_size" },
            { label: "IV bid", value: "bid_implied_volatility" },
            { label: "IV ask", value: "ask_implied_volatility" },
            { label: "Model Δ", value: "model_delta" },
            { label: "Data", value: "market_data_type" },
          ],
          detail.option_quotes,
        ),
      ),
      subsection(
        "Source-separated option computations",
        table(
          [
            { label: "Target", value: "target_timestamp_utc", format: clock },
            { label: "DTE", value: "dte_bucket" },
            { label: "Contract", value: "local_symbol" },
            { label: "Source", value: "computation_source" },
            { label: "IV", value: "implied_volatility" },
            { label: "Delta", value: "delta" },
            { label: "Gamma", value: "gamma" },
            { label: "Theta", value: "theta" },
            { label: "Vega", value: "vega" },
          ],
          detail.option_computations,
        ),
      ),
      subsection("Feature parity", jsonBlock(detail.feature_parity)),
    );
    replace("signal-evidence", stack);
  } catch (error) {
    replace("signal-evidence", node("div", "empty-state value-danger", `Unable to load ${signalId}.`));
  }
}

function renderShadow() {
  replace(
    "shadow-list",
    table(
      [
        { label: "Symbol", value: "symbol" },
        { label: "Signal", value: "signal_episode_id", format: (value) => short(value) },
        { label: "Crossing", value: "crossing_timestamp_utc", format: clock },
        { label: "Cohort", value: "cohort" },
        { label: "DTE", value: "dte_bucket" },
        { label: "Structure", value: "structure_type" },
        { label: "Entry debit", value: "entry_debit" },
        { label: "Multiplier", value: "multiplier" },
        { label: "Fees", value: "estimated_fees" },
        { label: "Complete", value: "completeness" },
        { label: "Rejection", value: "rejection_reason" },
      ],
      state.shadow,
      (row) => loadShadow(row.id),
    ),
  );
}

async function loadShadow(structureId) {
  replace("shadow-evidence", node("div", "empty-state", "Reading quoted structure…"));
  try {
    const detail = await api(`/api/shadow/${encodeURIComponent(structureId)}`);
    const stack = node("div", "detail-stack");
    stack.append(
      subsection(
        "Structure identity",
        kvGrid([
          ["Structure ID", detail.structure.id],
          ["Signal", detail.structure.signal_episode_id],
          ["Symbol / cohort", `${detail.structure.symbol} / ${detail.structure.cohort}`],
          ["DTE / type", `${detail.structure.dte_bucket} / ${detail.structure.structure_type}`],
          ["Entry debit", detail.structure.entry_debit],
          ["Estimated fees", detail.structure.estimated_fees],
          ["Ledger", detail.ledger],
          ["Rejection", detail.structure.rejection_reason],
        ]),
      ),
      subsection(
        "Every leg",
        table(
          [
            { label: "Role", value: "leg_role" },
            { label: "Entry side", value: "entry_side" },
            { label: "Contract", value: "local_symbol" },
            { label: "Right", value: "right" },
            { label: "Strike", value: "strike" },
            { label: "Multiplier", value: "multiplier" },
            { label: "Entry px", value: "entry_price" },
            { label: "Quote time", value: "quote_timestamp_utc", format: clock },
          ],
          detail.legs,
        ),
      ),
      subsection(
        "5 / 10 / 15 / 30 minute valuations",
        table(
          [
            { label: "Horizon", value: "horizon_minutes" },
            { label: "Target", value: "target_timestamp_utc", format: clock },
            { label: "Actual", value: "actual_quote_timestamp_utc", format: clock },
            { label: "Lag s", value: "capture_lag_seconds" },
            { label: "Exit credit", value: "exit_credit" },
            { label: "Gross return", value: "gross_return_on_debit" },
            { label: "Gross P&L", value: "gross_pnl" },
            { label: "Fees", value: "estimated_fees" },
            { label: "Data", value: "market_data_type" },
            { label: "Complete", value: "completeness" },
            { label: "Rejection", value: "rejection_reason" },
          ],
          detail.horizons,
        ),
      ),
    );
    replace("shadow-evidence", stack);
  } catch (error) {
    replace("shadow-evidence", node("div", "empty-state value-danger", `Unable to load ${structureId}.`));
  }
}

function safetyCard(label, value, description, tone = "") {
  const card = node("article", "safety-card");
  card.append(
    node("span", "micro-label", label),
    node("strong", tone ? `value-${tone}` : "", clean(value)),
    node("p", "", description),
  );
  return card;
}

function renderSafety() {
  const health = state.health;
  const runtime = state.runtime;
  const grid = node("div", "safety-grid");
  const bundle = health.active_bundle;
  const parity = health.feature_parity;
  const context = health.previous_session_context || {};
  const lease = health.recorder.lease || {};
  const budget = health.market_data.current_budget || {};
  const ibkrApi = health.ibkr_api || {};
  let ibkrApiSafetyState = "VERIFIED / CHECK PENDING";
  if (!ibkrApi.verified) {
    ibkrApiSafetyState = "BLOCKED";
  } else if (ibkrApi.blocker) {
    ibkrApiSafetyState = "UPDATE CHECK BLOCKED";
  } else if (ibkrApi.update_available === true) {
    ibkrApiSafetyState = "MANUAL UPDATE REVIEW";
  } else if (ibkrApi.update_available === false) {
    ibkrApiSafetyState = "VERIFIED + CURRENT";
  }
  const cards = [
    safetyCard("No-order-path verification", health.no_order_path_verified ? "VERIFIED ABSENT" : "FAILED", "The web process receives no broker object and exposes GET routes only.", health.no_order_path_verified ? "ok" : "danger"),
    safetyCard("risk.trading_enabled", "FALSE", "Startup rejects any true value. No paper or live order path exists.", "ok"),
    safetyCard("Official IBKR API", ibkrApiSafetyState, `${clean(ibkrApi.blocker)} // installed ${clean(ibkrApi.api_version)} // latest ${clean(ibkrApi.latest_api_version)} // update check fresh ${clean(ibkrApi.update_status_fresh)} // automatic installation FALSE.`, ibkrApi.verified && ibkrApi.update_status_fresh && !ibkrApi.blocker && ibkrApi.update_available !== true ? "ok" : "danger"),
    safetyCard("Active bundle", bundle.verified ? "VERIFIED" : "BLOCKED", bundle.bundle_id || clean(bundle.blockers), bundle.verified ? "ok" : "danger"),
    safetyCard("Feature parity", parity.scoring_allowed ? "ALLOWED" : "BLOCKED", `${parity.blocker} // ${JSON.stringify(parity.counts)}`, parity.scoring_allowed ? "ok" : "danger"),
    safetyCard("Previous-session context", context.eligibility ? "EXACT + ELIGIBLE" : "BLOCKED", `${clean(context.required_previous_session)} required // ${clean(context.observation_date)} observed`, context.eligibility ? "ok" : "danger"),
    safetyCard("Recorder lease", lease.owner_id ? "HELD" : "MISSING", `${clean(lease.owner_id)} // heartbeat ${clock(lease.heartbeat_at_utc)}`, lease.owner_id ? "ok" : "danger"),
    safetyCard("Market-data budget", `${budget.active_lines ?? 0} / ${budget.usable_lines ?? (health.market_data.line_budget - health.market_data.reserved_headroom)} ACTIVE`, `${budget.pending_requests ?? 0} pending; ${budget.awaiting_cancellation ?? 0} cancelling; ${budget.current_request_rate ?? 0} requests in window; ${budget.waiting_signals ?? 0} waiting; ${budget.rejected_signals ?? 0} rejected.`),
    safetyCard("Reconnect state", clean((health.ibkr || {}).message || "REPLAY DIAGNOSTIC"), "1102 retained-data and 1101 lost-data recovery remain distinct."),
    safetyCard("Application identity", `${health.application.version} / ${short(health.application.git_commit, 12)}`, `${health.instance_identity} // persistent database outside release.`),
  ];
  cards.forEach((card) => grid.append(card));
  replace("safety-grid", grid);

  const audit = node("div");
  if (!state.audit.length) {
    audit.append(node("div", "empty-state", "No audit events recorded."));
  }
  state.audit.forEach((item) => {
    const row = node("div", "audit-item");
    row.append(
      node("div", "audit-sequence", String(item.sequence).padStart(3, "0")),
      node("div", "audit-type", item.event_type),
    );
    const message = node("div", "audit-message", item.message);
    message.append(node("small", "", `${clock(item.recorded_at_utc)} // ${item.actor} // ${short(item.git_commit, 12)}`));
    row.append(message);
    audit.append(row);
  });
  replace("audit-log", audit);
}

async function refreshAll() {
  const button = document.getElementById("refresh");
  button.disabled = true;
  button.textContent = "Reading…";
  try {
    const [health, runtime, universe, signals, shadow, audit] = await Promise.all([
      api("/api/health"),
      api("/api/runtime"),
      api("/api/universe"),
      api("/api/signals"),
      api("/api/shadow"),
      api("/api/audit"),
    ]);
    state.health = health;
    state.runtime = runtime;
    state.universe = universe;
    state.signals = signals.items || [];
    state.shadow = shadow.items || [];
    state.audit = audit.items || [];
    renderHealth();
    renderSignals();
    renderShadow();
    renderSafety();
    document.getElementById("last-sync").textContent = `SYNCHRONIZED ${new Date().toISOString()}`;
  } catch (error) {
    const message = error.status === 401 ? "Authentication required." : "Persistent state is unavailable.";
    replace("blocker-strip", node("span", "blocker", message));
    document.getElementById("last-sync").textContent = `SYNC FAILED ${new Date().toISOString()}`;
  } finally {
    button.disabled = false;
    button.textContent = "Refresh evidence";
  }
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => showScreen(button.dataset.screen));
});
document.getElementById("refresh").addEventListener("click", refreshAll);
refreshAll();
setInterval(() => {
  if (document.visibilityState === "visible") refreshAll();
}, 15000);
