"use strict";

const state = {
  health: null,
  status: null,
  capabilities: null,
  universe: [],
  episodes: [],
  shadow: [],
  audit: [],
  sessionReports: [],
  selectedEpisode: null,
};

function node(tag, className = "", text = null) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== null && text !== undefined) element.textContent = String(text);
  return element;
}

function clean(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "YES" : "NO";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  }
  return String(value);
}

function clock(value) {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf())
    ? clean(value)
    : parsed.toISOString().replace(".000Z", "Z");
}

function age(value) {
  if (!value) return "—";
  const milliseconds = Date.now() - new Date(value).valueOf();
  if (!Number.isFinite(milliseconds)) return "—";
  return `${Math.max(0, milliseconds / 1000).toFixed(1)}s`;
}

function short(value, length = 22) {
  const text = clean(value);
  return text.length > length ? `${text.slice(0, length)}…` : text;
}

function replace(targetId, child) {
  document.getElementById(targetId).replaceChildren(child);
}

function metric(label, value, note = "", tone = "") {
  const card = node("article", "metric");
  card.append(node("span", "micro-label", label));
  card.append(node("strong", tone ? `value-${tone}` : "", clean(value)));
  if (note) card.append(node("small", "", note));
  return card;
}

function kvGrid(items) {
  const grid = node("div", "kv-grid");
  items.forEach(([label, value]) => {
    const cell = node("div", "kv");
    cell.append(node("span", "", label), node("strong", "", clean(value)));
    grid.append(cell);
  });
  return grid;
}

function table(columns, rows, action = null) {
  if (!rows.length) return node("div", "empty-state", "No evidence rows recorded.");
  const wrapper = node("div", "table-wrap");
  const element = node("table");
  const head = node("thead");
  const heading = node("tr");
  columns.forEach((column) => heading.append(node("th", "", column.label)));
  if (action) heading.append(node("th", "", "Inspect"));
  head.append(heading);
  const body = node("tbody");
  rows.forEach((row) => {
    const line = node("tr");
    columns.forEach((column) => {
      const raw = typeof column.value === "function"
        ? column.value(row)
        : row[column.value];
      line.append(node("td", "", column.format ? column.format(raw) : clean(raw)));
    });
    if (action) {
      const cell = node("td");
      const button = node("button", "row-action", "Open");
      button.type = "button";
      button.addEventListener("click", () => action(row));
      cell.append(button);
      line.append(cell);
    }
    body.append(line);
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
  return node("pre", "json-block", JSON.stringify(value || {}, null, 2));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    ...options,
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

function subscriptionCapacity(kind) {
  const capacity = state.status.capacity?.[kind] || {};
  return `${capacity.used ?? 0} / ${capacity.available ?? 0}`;
}

function renderStatus() {
  const status = state.status;
  const capabilities = state.capabilities;
  const manifest = capabilities.manifest || {};
  const blockers = [
    ...(state.health.blockers || []),
    ...(manifest.blockers || []),
  ];
  const strip = document.createDocumentFragment();
  if (!blockers.length) {
    strip.append(node("span", "status-tag value-ok", "NO ACTIVE BLOCKERS"));
  } else {
    [...new Set(blockers)].forEach((item) => {
      strip.append(node("span", "blocker", item));
    });
  }
  replace("blocker-strip", strip);

  const connection = status.ibkr_connection || {};
  const latest = status.latest_checkpoint || {};
  const grid = document.createDocumentFragment();
  [
    metric("Recorder readiness", status.state, status.banner, status.state === "recording" ? "ok" : "danger"),
    metric("IBKR connection", connection.state || "OFFLINE", connection.message || "No current socket event"),
    metric("Market-data type", manifest.market_data_type || "UNOBSERVED", "Scientific recording requires LIVE", capabilities.scientific_recording_valid ? "ok" : "danger"),
    metric("Gateway / TWS", manifest.tws_or_gateway_version, `API server ${clean(manifest.api_server_version)}`),
    metric("Official IBKR API", manifest.ibkr_api_version || "UNOBSERVED", "First-party socket API; no order surface"),
    metric("Clock drift", manifest.clock_drift_seconds, "local clock minus IBKR clock"),
    metric("Level I capacity", subscriptionCapacity("level1"), "Frozen cohort + required proxies"),
    metric("Tick-by-tick capacity", subscriptionCapacity("tick_by_tick"), "BidAsk + Last on promoted symbols"),
    metric("Depth capacity", subscriptionCapacity("depth"), "Optional SMART depth"),
    metric("Option capacity", subscriptionCapacity("option"), "Bounded post-episode contracts"),
    metric("M1C parity", status.model_parity?.m1c, "Tolerance ≤ 1e-12", status.model_parity?.m1c === "passed" ? "ok" : "danger"),
    metric("Direction parity", status.model_parity?.direction, "A1 / C1 / R1 withheld until passed", status.model_parity?.direction === "passed" ? "ok" : "danger"),
    metric("Bar source", status.bar?.source, status.bar?.compatibility_status),
    metric("Last completed bar", clock(status.bar?.last_completed), "Partial bars are never scored"),
    metric("Next bar completion", clock(status.bar?.next_expected_completion), `${clean(status.bar?.freshness_seconds, 1)}s since last completed bar`),
    metric("Last raw event", clock(status.last_event_timestamp), `${status.data_gaps || 0} recorded gaps`),
    metric("No-order state", status.execution_enabled ? "FAILED" : "VERIFIED", "Execution disabled; broker mutation unavailable", status.execution_enabled ? "danger" : "ok"),
  ].forEach((item) => grid.append(item));
  replace("runtime-grid", grid);

  replace("score-panel", kvGrid([
    ["Model", latest.model_id],
    ["Symbol / checkpoint", latest.symbol ? `${latest.symbol} / ${latest.checkpoint}` : null],
    ["Probability", latest.probability],
    ["Frozen threshold", latest.threshold],
    ["Above threshold", latest.threshold_passed],
    ["Eligible", latest.eligible],
    ["Feature freshness", latest.feature_freshness],
    ["Feature hash", short(latest.feature_hash)],
    ["Model hash", short(latest.model_hash)],
    ["Context hash", short(latest.session_context_hash)],
  ]));

  replace("capability-panel", kvGrid([
    ["Scientific recording valid", capabilities.scientific_recording_valid],
    ["Diagnostic display allowed", capabilities.diagnostic_display_allowed],
    ["Required data type", capabilities.required_market_data_type],
    ["Underlying Level I", (manifest.underlying_level1_symbols || []).length],
    ["Proxy Level I", (manifest.market_proxy_level1_symbols || []).length],
    ["Option Level I", manifest.option_level1_available],
    ["Option computations", manifest.option_computation_fields_available],
    ["Depth exchanges", (manifest.depth_exchanges || []).join(", ")],
    ["Resolved contracts", (manifest.resolved_contracts || []).length],
    ["Permission errors", (manifest.permission_errors || []).join(", ")],
  ]));
}

function micropriceEdge(row) {
  if ([row.bid, row.ask, row.bid_size, row.ask_size].some((value) => value === null || value === undefined)) {
    return null;
  }
  const total = Number(row.bid_size) + Number(row.ask_size);
  if (total <= 0) return null;
  const microprice = (Number(row.ask) * Number(row.bid_size)
    + Number(row.bid) * Number(row.ask_size)) / total;
  return microprice - (Number(row.bid) + Number(row.ask)) / 2;
}

function renderUniverse() {
  replace("universe-panel", table(
    [
      { label: "Symbol", value: "symbol" },
      { label: "Last completed bar", value: "last_completed_bar", format: clock },
      { label: "M1C p", value: "m1c_probability" },
      { label: "Gate", value: "m1c_threshold" },
      { label: "Distance", value: "distance_from_threshold" },
      { label: "Fresh episode", value: "fresh_episode" },
      { label: "A1", value: "a1_classification" },
      { label: "C1", value: "c1_classification" },
      { label: "R1", value: "r1_classification" },
      { label: "Bid", value: "bid" },
      { label: "Ask", value: "ask" },
      { label: "Spread", value: "spread" },
      { label: "Quote imbalance", value: "quote_imbalance" },
      { label: "Microprice edge", value: micropriceEdge },
      { label: "Tick-by-tick", value: "tick_by_tick_status" },
      { label: "Depth", value: "depth_status" },
      { label: "Quote received", value: "quote_timestamp_utc", format: clock },
      { label: "Freshness", value: "quote_timestamp_utc", format: age },
    ],
    state.universe,
  ));
}

function renderEpisodeIndex() {
  replace("signal-list", table(
    [
      { label: "Symbol", value: "symbol" },
      { label: "Trigger", value: "trigger_bar_end_utc", format: clock },
      { label: "Entry", value: "prospective_entry_timestamp_utc", format: clock },
      { label: "M1C", value: "m1c_probability" },
      { label: "A1", value: "a1_action" },
      { label: "C1", value: "c1_action" },
      { label: "R1", value: "r1_action" },
      { label: "Phase", value: "phase" },
      { label: "Valid", value: "scientific_recording_valid" },
    ],
    state.episodes,
    (row) => loadEpisode(row.episode_id),
  ));

  const picker = document.getElementById("option-episode");
  const selected = picker.value;
  picker.replaceChildren();
  if (!state.episodes.length) {
    const option = node("option", "", "No episodes");
    option.value = "";
    picker.append(option);
  } else {
    state.episodes.forEach((episode) => {
      const option = node("option", "", `${episode.symbol} // ${clock(episode.trigger_bar_end_utc)}`);
      option.value = episode.episode_id;
      picker.append(option);
    });
    picker.value = state.episodes.some((episode) => episode.episode_id === selected)
      ? selected
      : state.episodes[0].episode_id;
  }
}

function evidenceTimeline(episode) {
  const rail = node("div", "evidence-timeline");
  [
    ["T−1 direction cutoff", episode.maximum_feature_timestamp_utc],
    ["T trigger complete", episode.trigger_bar_end_utc],
    ["Prospective entry", episode.prospective_entry_timestamp_utc],
    ["Recording horizon", episode.prospective_entry_timestamp_utc
      ? new Date(new Date(episode.prospective_entry_timestamp_utc).valueOf() + 30 * 60 * 1000).toISOString()
      : null],
  ].forEach(([label, value]) => {
    const cell = node("div", "timeline-cell");
    cell.append(node("span", "", label), node("strong", "", clock(value)));
    rail.append(cell);
  });
  return rail;
}

function renderDirectionCards(directions) {
  const matrix = node("div", "score-matrix");
  directions.forEach((direction) => {
    const payload = direction.payload || {};
    const cell = node("article", "score-cell research-classification");
    cell.append(
      node("span", "", direction.classification_label),
      node("strong", "", direction.action),
      node("small", "", `p(up) ${clean(direction.probability_up, 6)} // confidence ${clean(direction.confidence, 6)} // boundary ${clean(direction.confidence_boundary, 6)}`),
    );
    cell.title = `Directional research classification only. ${clean(payload.model_hash)}`;
    matrix.append(cell);
  });
  return matrix;
}

function quoteChart(points) {
  if (points.length < 2) {
    return node("div", "empty-state", "Bid/ask chart awaits two valid quote observations.");
  }
  const width = 900;
  const height = 240;
  const padding = 22;
  const times = points.map((item) => new Date(item.timestamp_utc).valueOf());
  const prices = points.flatMap((item) => [item.bid, item.ask, item.microprice])
    .filter((value) => Number.isFinite(Number(value)))
    .map(Number);
  const minimumTime = Math.min(...times);
  const maximumTime = Math.max(...times);
  const minimumPrice = Math.min(...prices);
  const maximumPrice = Math.max(...prices);
  const timeSpan = Math.max(1, maximumTime - minimumTime);
  const priceSpan = Math.max(1e-9, maximumPrice - minimumPrice);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "quote-chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Episode bid, ask and microprice chart");
  [["bid", "quote-bid"], ["ask", "quote-ask"], ["microprice", "quote-microprice"]]
    .forEach(([field, className]) => {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      const coordinates = points
        .filter((item) => Number.isFinite(Number(item[field])))
        .map((item) => {
          const x = padding + ((new Date(item.timestamp_utc).valueOf() - minimumTime) / timeSpan)
            * (width - padding * 2);
          const y = height - padding - ((Number(item[field]) - minimumPrice) / priceSpan)
            * (height - padding * 2);
          return `${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(" ");
      line.setAttribute("points", coordinates);
      line.setAttribute("class", className);
      svg.append(line);
    });
  const region = node("div", "quote-chart-region");
  const legend = node("div", "chart-legend");
  legend.append(
    node("span", "legend-bid", "BID"),
    node("span", "legend-ask", "ASK"),
    node("span", "legend-microprice", "MICROPRICE"),
    node("span", "", `${clock(points[0].timestamp_utc)} → ${clock(points.at(-1).timestamp_utc)}`),
  );
  region.append(svg, legend);
  return region;
}

function renderMicrostructure(items, quoteSeries = [], depth = null) {
  if (!items.length) return node("div", "empty-state", "No completed microstructure windows.");
  const latest = items[items.length - 1];
  const scores = latest.summary?.scores || {};
  const matrix = node("div", "score-matrix");
  ["MC", "MD", "MA", "MB"].forEach((name) => {
    const score = scores[name] || {};
    const cell = node("article", "score-cell");
    cell.append(
      node("span", "", `${name} // microstructure descriptive score`),
      node("strong", "", clean(score.composite, 6)),
      node("small", "", `${score.valid_component_count || 0} valid equal-weight components // fitted weights FALSE`),
    );
    matrix.append(cell);
  });
  const stack = node("div", "detail-stack");
  stack.append(
    quoteChart(quoteSeries),
    matrix,
    subsection("Frozen archetype / microstructure relationship", jsonBlock(
      latest.archetype_relationship || {},
    )),
    subsection("Latest bounded SMART-depth state", depth
      ? kvGrid([
        ["Book valid", depth.book_valid],
        ["Depth imbalance", depth.depth_imbalance],
        ["Weighted imbalance", depth.weighted_depth_imbalance],
        ["Displayed bid / ask", `${clean(depth.total_bid_size)} / ${clean(depth.total_ask_size)}`],
        ["Bid / ask replenishment", `${clean(depth.bid_side_replenishment)} / ${clean(depth.ask_side_replenishment)}`],
        ["Book slope", depth.book_slope],
        ["Active venues", depth.active_venues],
        ["Reset count", depth.reset_count],
      ])
      : node("div", "empty-state", "No valid bounded depth snapshot is available.")),
    table(
      [
        { label: "Window", value: "window_name" },
        { label: "End", value: "window_end_utc", format: clock },
        { label: "Level I", value: "level1_valid" },
        { label: "Tick", value: "tick_valid" },
        { label: "Depth", value: "depth_valid" },
        { label: "Quality flags", value: (row) => (row.quality_flags || []).join(", ") },
      ],
      items,
    ),
  );
  return stack;
}

async function loadEpisode(episodeId) {
  replace("signal-evidence", node("div", "empty-state", "Reading causal evidence…"));
  try {
    const [detail, microstructure, options] = await Promise.all([
      api(`/api/episodes/${encodeURIComponent(episodeId)}`),
      api(`/api/episodes/${encodeURIComponent(episodeId)}/microstructure`),
      api(`/api/episodes/${encodeURIComponent(episodeId)}/options`),
    ]);
    state.selectedEpisode = episodeId;
    const episode = detail.episode;
    const directions = detail.directional_research_classifications || [];
    const maximumDirectionTime = directions.length
      ? directions[0].maximum_feature_timestamp_utc
      : null;
    const stack = node("div", "detail-stack");
    stack.append(
      evidenceTimeline({ ...episode, maximum_feature_timestamp_utc: maximumDirectionTime }),
      subsection("Frozen movement identity", kvGrid([
        ["Episode ID", episode.episode_id],
        ["Symbol / session", `${episode.symbol} / ${episode.session_date}`],
        ["Trigger checkpoint", episode.trigger_checkpoint],
        ["M1C probability", episode.m1c_probability],
        ["Previous probability", episode.previous_m1c_probability],
        ["Episode number", episode.episode_number],
        ["Minutes since prior", episode.minutes_since_previous_episode],
        ["Scientific recording valid", episode.scientific_recording_valid],
        ["Phase", episode.phase],
        ["Model hash", short(episode.model_hash)],
        ["Feature hash", short(episode.feature_hash)],
      ])),
      subsection("Directional research classifications", directions.length
        ? renderDirectionCards(directions)
        : node("div", "empty-state", "Classifications withheld until direction parity and live-data gates pass.")),
      subsection(
        "Microstructure windows",
        renderMicrostructure(
          microstructure.items || [],
          microstructure.quote_series || [],
          microstructure.latest_depth_snapshot || null,
        ),
      ),
      subsection("Frozen model input", jsonBlock(episode.feature_values || {})),
    );
    replace("signal-evidence", stack);
    const picker = document.getElementById("option-episode");
    if ([...picker.options].some((item) => item.value === episodeId)) picker.value = episodeId;
    renderOptions(options.items || []);
  } catch (error) {
    replace("signal-evidence", node("div", "empty-state value-danger", "Episode evidence is unavailable."));
  }
}

function renderOptions(items) {
  const region = node("div", "detail-stack");
  const buckets = ["0DTE", "1DTE", "3_TO_5_DTE"];
  buckets.forEach((bucket) => {
    const rows = items.filter((item) => item.dte_bucket === bucket);
    region.append(node("h3", "bucket-heading", `${bucket} // ${rows.length} CONTRACTS`));
    region.append(table(
      [
        { label: "Expiry", value: "expiry" },
        { label: "DTE", value: "dte" },
        { label: "Strike", value: "strike" },
        { label: "Right", value: "right" },
        { label: "Bid", value: "bid" },
        { label: "Ask", value: "ask" },
        { label: "Spread", value: (row) => row.bid === null || row.ask === null ? null : row.ask - row.bid },
        { label: "IV", value: "implied_volatility" },
        { label: "Delta", value: "delta" },
        { label: "Gamma", value: "gamma" },
        { label: "Theta", value: "theta" },
        { label: "Vega", value: "vega" },
        { label: "Quote received", value: "received_timestamp_utc", format: clock },
        { label: "Recording", value: "recording_status" },
      ],
      rows,
    ));
  });
  replace("options-panel", region);
}

async function loadSelectedOptions() {
  const episodeId = document.getElementById("option-episode").value;
  if (!episodeId) {
    renderOptions([]);
    return;
  }
  const payload = await api(`/api/episodes/${encodeURIComponent(episodeId)}/options`);
  renderOptions(payload.items || []);
}

function renderShadow() {
  replace("shadow-list", table(
    [
      { label: "Episode", value: "episode_id", format: short },
      { label: "Symbol", value: "symbol" },
      { label: "Archetype", value: "archetype" },
      { label: "Direction", value: "direction" },
      { label: "DTE", value: "dte_bucket" },
      { label: "Contract", value: "contract_identity", format: short },
      { label: "Horizon", value: (row) => `${row.horizon_minutes}m` },
      { label: "Entry ask", value: "entry_ask" },
      { label: "Exit bid", value: "exit_bid" },
      { label: "Ask→bid return", value: "ask_to_bid_return" },
      { label: "$ P&L / contract", value: "dollar_pnl_per_contract" },
      { label: "Valid", value: "valid" },
      { label: "Quality", value: (row) => (row.quality_flags || []).join(", ") },
    ],
    state.shadow,
  ));
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

function renderAudit() {
  const grid = node("div", "safety-grid");
  [
    safetyCard("No-order path", state.health.no_order_path_verified ? "VERIFIED ABSENT" : "FAILED", "No order, account or position resource. Replay POST controls only.", state.health.no_order_path_verified ? "ok" : "danger"),
    safetyCard("M1C parity", state.status.model_parity?.m1c, "250-row live inference parity gate at 1e-12."),
    safetyCard("Direction parity", state.status.model_parity?.direction, "A1, C1 and R1 output remains hidden until exact parity."),
    safetyCard("Raw evidence", state.status.last_event_timestamp ? "APPEND-ONLY" : "WAITING", "Partitioned Parquet with content hashes and explicit incomplete state."),
    safetyCard("Microstructure model", "NOT FITTED", "MC, MD, MA and MB are equal-weight descriptive scores only."),
    safetyCard("Option P&L", "SHADOW QUOTE P&L", "First valid ask entry and bid exit; no fills or execution."),
  ].forEach((card) => grid.append(card));
  replace("safety-grid", grid);

  const audit = node("div");
  if (!state.audit.length) {
    audit.append(node("div", "empty-state", "No recorder audit events."));
  }
  state.audit.forEach((item, index) => {
    const row = node("div", "audit-item");
    row.append(
      node("div", "audit-sequence", String(index + 1).padStart(3, "0")),
      node("div", "audit-type", item.audit_type),
    );
    const message = node("div", "audit-message", short(item.identity, 72));
    message.append(node("small", "", `${clock(item.recorded_at_utc)} // ${clean(item.details)}`));
    row.append(message);
    audit.append(row);
  });
  replace("audit-log", audit);
  replace("session-reports-panel", table(
    [
      { label: "Session", value: "session_date" },
      { label: "Complete", value: "complete" },
      { label: "Generated", value: "generated_at_utc", format: clock },
      { label: "Level I coverage", value: (row) => row.report?.level1_coverage },
      { label: "M1C coverage", value: (row) => row.report?.m1c_checkpoint_coverage },
      { label: "Episodes", value: (row) => row.report?.fresh_episodes },
      { label: "Option coverage", value: (row) => row.report?.option_quote_coverage },
      { label: "Gaps", value: (row) => row.report?.data_gaps },
      { label: "Capacity denials", value: (row) => row.report?.capacity_denials },
      { label: "Shadow horizons", value: (row) => row.report?.complete_shadow_horizons },
    ],
    state.sessionReports,
  ));
  const replay = state.status.replay || {};
  document.getElementById("replay-state").textContent =
    `${clean(replay.state).toUpperCase()} // IBKR CONNECTIONS ${replay.ibkr_connections_attempted || 0}`;
}

async function refreshAll() {
  const button = document.getElementById("refresh");
  button.disabled = true;
  button.textContent = "Reading…";
  try {
    const [health, status, capabilities, universe, episodes, shadow, audit, reports] = await Promise.all([
      api("/api/health"),
      api("/api/recorder/status"),
      api("/api/recorder/capabilities"),
      api("/api/universe/live"),
      api("/api/episodes"),
      api("/api/shadow-outcomes"),
      api("/api/audit/events"),
      api("/api/recorder/session-reports"),
    ]);
    state.health = health;
    state.status = status;
    state.capabilities = capabilities;
    state.universe = universe.items || [];
    state.episodes = episodes.items || [];
    state.shadow = shadow.items || [];
    state.audit = audit.items || [];
    state.sessionReports = reports.items || [];
    renderStatus();
    renderUniverse();
    renderEpisodeIndex();
    renderShadow();
    renderAudit();
    if (!state.selectedEpisode && state.episodes.length) {
      await loadEpisode(state.episodes[0].episode_id);
    } else {
      await loadSelectedOptions();
    }
    document.getElementById("last-sync").textContent =
      `SYNCHRONIZED ${new Date().toISOString()}`;
  } catch (error) {
    const message = error.status === 401
      ? "Authentication required."
      : "Persistent recorder state is unavailable.";
    replace("blocker-strip", node("span", "blocker", message));
    document.getElementById("last-sync").textContent =
      `SYNC FAILED ${new Date().toISOString()}`;
  } finally {
    button.disabled = false;
    button.textContent = "Refresh evidence";
  }
}

async function controlReplay(action) {
  const mode = document.getElementById("replay-mode").value;
  const payload = action === "start"
    ? { mode, speed: mode === "real_time" ? 1 : 10, episode_id: mode === "episode_only" ? state.selectedEpisode : null }
    : null;
  try {
    const result = await api(`/api/replay/${action}`, {
      method: "POST",
      body: payload ? JSON.stringify(payload) : undefined,
    });
    state.status.replay = result;
    renderAudit();
  } catch (error) {
    document.getElementById("replay-state").textContent = "REPLAY CONTROL REJECTED";
  }
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => showScreen(button.dataset.screen));
});
document.getElementById("refresh").addEventListener("click", refreshAll);
document.getElementById("option-episode").addEventListener("change", loadSelectedOptions);
document.getElementById("replay-start").addEventListener("click", () => controlReplay("start"));
document.getElementById("replay-stop").addEventListener("click", () => controlReplay("stop"));
refreshAll();
setInterval(() => {
  if (document.visibilityState === "visible") refreshAll();
}, 15000);
