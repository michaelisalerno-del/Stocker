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
  quietStatus: null,
  quietUniverse: [],
  quietEpisodes: [],
  quietShadow: [],
  quietSessionQuality: [],
  concentrationAudit: null,
  budget: null,
  transfer: null,
  reportPackages: [],
  selectedQuietEpisode: null,
  activeScreen: "live-monitor",
};

const polling = new window.StockerPolling.RequestCoordinator();
const pollingPolicy = window.StockerPolling.POLLING_POLICY;
let refreshAllPromise = null;
let refreshGeneration = 0;

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

function percent(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "—";
  const observed = Number(value);
  return Number.isFinite(observed) ? `${(observed * 100).toFixed(digits)}%` : clean(value);
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
  return polling.request(path, options);
}

function showScreen(screenId) {
  state.activeScreen = screenId;
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
  const capacity = state.status?.capacity?.[kind] || {};
  return `${capacity.used ?? 0} / ${capacity.available ?? 0}`;
}

function renderStatus() {
  const status = state.status || {};
  const capabilities = state.capabilities || {};
  const manifest = capabilities.manifest || {};
  const blockers = [
    ...(state.health?.blockers || []),
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

function capacityValue(value) {
  return value && typeof value === "object" ? value.value : value;
}

function renderBudgetTransfer() {
  const budget = state.budget || {};
  const capacity = budget.runtime_capacity || {};
  replace("budget-panel", kvGrid([
    ["Configured / discovered total lines", capacityValue(capacity.total_level1_allowance)],
    ["Available ordinary Level I", capacity.available_ordinary_level1_lines],
    ["Externally reserved", capacityValue(capacity.externally_reserved_lines)],
    ["Future-trading reserve", capacityValue(capacity.reserved_future_trading_lines)],
    ["Safety margin", capacityValue(capacity.safety_margin_lines)],
    ["Current internal usage", budget.current_internal_usage],
    ["Bar streams", budget.current_usage?.bar || 0],
    ["Level I streams", budget.current_usage?.level1 || 0],
    ["Option streams", budget.current_usage?.option || 0],
    ["Tick-by-tick streams", budget.current_usage?.tick_by_tick || 0],
    ["Depth streams", budget.current_usage?.depth || 0],
    ["Pending requests", budget.pending_requests],
    ["Queued episodes", budget.queued_episodes],
    ["Degraded episodes", budget.degraded_episodes],
    ["Oldest optional stream", clock(budget.oldest_active_optional_subscription)],
    ["Reconciliation warnings", (budget.reconciliation_warnings || []).length],
    ["Fatal state", budget.fatal_budget_state],
  ]));

  const transfer = state.transfer || {};
  const aggregate = transfer.aggregate || {};
  const probability = aggregate.probability_metrics || {};
  const tails = aggregate.tail_metrics || {};
  const episodes = aggregate.episode_metrics || {};
  replace("transfer-panel", kvGrid([
    ["Decision", transfer.decision || "blocked_insufficient_valid_sessions"],
    ["Valid sessions", transfer.valid_session_count || 0],
    ["Exact vendor equality required", false],
    ["Pearson / Spearman", `${clean(probability.pearson)} / ${clean(probability.spearman)}`],
    ["Mean probability bias", probability.mean_signed_bias],
    ["Bottom-10 agreement", tails.bottom_10_agreement],
    ["High-tail agreement", tails.high_tail_agreement],
    ["Quiet exact / ±1 checkpoint", `${clean(episodes.quiet_exact_checkpoint_matches)} / ${clean(episodes.quiet_matches_within_one_checkpoint)}`],
    ["High exact / ±1 checkpoint", `${clean(episodes.high_exact_checkpoint_matches)} / ${clean(episodes.high_matches_within_one_checkpoint)}`],
    ["Historical decision", transfer.historical_decision],
    ["Profitability decision", "NOT ALLOWED"],
  ]));

  const packageRegion = node("div", "report-package-list");
  if (!state.reportPackages.length) {
    packageRegion.append(node("div", "empty-state", "No completed daily report package yet."));
  }
  state.reportPackages.forEach((item) => {
    const row = node("div", "audit-item");
    row.append(
      node("div", "audit-sequence", item.session),
      node("div", "audit-type", clock(item.generated_at_utc)),
    );
    const link = node("a", "refresh", "Download package");
    link.href = item.download_path;
    link.setAttribute("download", "");
    row.append(link);
    packageRegion.append(row);
  });
  replace("report-packages-panel", packageRegion);
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
    (row) => polling.run(
      "detail:episode",
      (signal) => loadEpisode(row.episode_id, signal),
      { supersede: true },
    ),
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

async function loadEpisode(episodeId, signal = undefined) {
  replace("signal-evidence", node("div", "empty-state", "Reading causal evidence…"));
  try {
    const [detail, microstructure, options] = await Promise.all([
      api(`/api/episodes/${encodeURIComponent(episodeId)}`, { signal }),
      api(`/api/episodes/${encodeURIComponent(episodeId)}/microstructure`, { signal }),
      api(`/api/episodes/${encodeURIComponent(episodeId)}/options`, { signal }),
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
    if (isAbort(error)) return;
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

async function loadSelectedOptions(signal = undefined) {
  const episodeId = document.getElementById("option-episode").value;
  if (!episodeId) {
    renderOptions([]);
    return;
  }
  const payload = await api(
    `/api/episodes/${encodeURIComponent(episodeId)}/options`,
    { signal },
  );
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

function renderQuietUniverse() {
  const status = state.quietStatus;
  const thresholds = status.thresholds || {};
  const counts = status.observation_counts || {};
  const statusGrid = document.createDocumentFragment();
  [
    metric("Bottom 5%", thresholds.bottom_5, "Inclusive frozen threshold"),
    metric("Bottom 10%", thresholds.bottom_10, "Primary frozen quiet state", "warn"),
    metric("Bottom 20%", thresholds.bottom_20, "Inclusive secondary threshold"),
    metric("Fresh quiet episodes", counts.quiet_bottom_10 || 0, `${status.complete_quiet_episodes || 0} complete`),
    metric("Neutral controls", counts.neutral_control || 0, "Frozen deterministic 10% hash sample"),
    metric("High-tail controls", counts.high_tail_control || 0, `Every fresh p ≥ ${clean(thresholds.high_tail, 6)}`),
    metric("Phase boundary", "20 sessions / 150 / 150", "Engineering transfer / development / untouched confirmation"),
    metric("Order path", String(status.order_path || "absent").toUpperCase(), status.banner, "ok"),
  ].forEach((item) => statusGrid.append(item));
  replace("quiet-status-grid", statusGrid);

  replace("quiet-universe-panel", table(
    [
      { label: "Symbol", value: "symbol" },
      { label: "M1C p", value: "m1c_probability" },
      { label: "B5", value: "bottom_5" },
      { label: "B10", value: "bottom_10" },
      { label: "B20", value: "bottom_20" },
      { label: "Distance B10", value: "distance_from_bottom_10" },
      { label: "Fresh quiet", value: "fresh_quiet_episode" },
      { label: "Previous p", value: "previous_m1c_probability" },
      { label: "Option context", value: "option_context_valid" },
      { label: "Bid", value: "bid" },
      { label: "Ask", value: "ask" },
      { label: "Spread", value: "spread" },
      { label: "Microprice edge", value: "microprice_edge_bps" },
      { label: "Quote received", value: "received_timestamp_utc", format: clock },
      { label: "Freshness", value: "received_timestamp_utc", format: age },
      { label: "Quality", value: "data_quality_status" },
    ],
    state.quietUniverse,
  ));
}

function renderQuietEpisodeIndex() {
  replace("quiet-episode-list", table(
    [
      { label: "Symbol", value: "symbol" },
      { label: "Kind", value: "observation_kind" },
      { label: "Trigger", value: "trigger_timestamp_utc", format: clock },
      { label: "M1C", value: "m1c_probability" },
      { label: "B5", value: "bottom_5" },
      { label: "B10", value: "bottom_10" },
      { label: "B20", value: "bottom_20" },
      { label: "Phase", value: "phase" },
      { label: "Complete", value: "completion_status" },
    ],
    state.quietEpisodes,
    (row) => polling.run(
      "detail:quiet-episode",
      (signal) => loadQuietEpisode(row.observation_id, signal),
      { supersede: true },
    ),
  ));
}

function quietEvidenceTimeline(episode) {
  const rail = node("div", "evidence-timeline");
  const sixtyMinuteExit = episode.prospective_entry_timestamp_utc
    ? new Date(
      new Date(episode.prospective_entry_timestamp_utc).valueOf() + 60 * 60 * 1000,
    ).toISOString()
    : null;
  [
    ["Previous eligible p", episode.previous_m1c_probability],
    ["Quiet trigger", episode.trigger_timestamp_utc],
    ["Research entry", episode.prospective_entry_timestamp_utc],
    ["Final bounded horizon", sixtyMinuteExit],
  ].forEach(([label, value], index) => {
    const cell = node("div", "timeline-cell");
    cell.append(
      node("span", "", label),
      node("strong", "", index === 0 ? clean(value, 8) : clock(value)),
    );
    rail.append(cell);
  });
  return rail;
}

function quietOptionTable(items) {
  return table(
    [
      { label: "Bucket", value: "dte_bucket" },
      { label: "Expiry", value: "expiry" },
      { label: "DTE", value: "dte" },
      { label: "Strike", value: "strike" },
      { label: "Right", value: "right" },
      { label: "Bid / ask", value: (row) => `${clean(row.bid)} / ${clean(row.ask)}` },
      { label: "Sizes", value: (row) => `${clean(row.bid_size)} / ${clean(row.ask_size)}` },
      { label: "IV", value: "implied_volatility" },
      { label: "Delta", value: "delta" },
      { label: "Γ / Θ / V", value: (row) => `${clean(row.gamma)} / ${clean(row.theta)} / ${clean(row.vega)}` },
      { label: "Volume / OI", value: (row) => `${clean(row.volume)} / ${clean(row.open_interest)}` },
      { label: "Received", value: "received_timestamp_utc", format: clock },
      { label: "Quality", value: (row) => (row.quote_quality_flags || []).join(", ") },
    ],
    items,
  );
}

async function loadQuietEpisode(observationId, signal = undefined) {
  replace("quiet-episode-evidence", node("div", "empty-state", "Reading quiet-state evidence…"));
  try {
    const [detail, options] = await Promise.all([
      api(`/api/quiet-state/episodes/${encodeURIComponent(observationId)}`, { signal }),
      api(`/api/quiet-state/episodes/${encodeURIComponent(observationId)}/options`, {
        signal,
      }),
    ]);
    state.selectedQuietEpisode = observationId;
    const episode = detail.episode;
    const stack = node("div", "detail-stack");
    stack.append(
      quietEvidenceTimeline(episode),
      subsection("Frozen quiet identity", kvGrid([
        ["Observation ID", episode.observation_id],
        ["Kind", episode.observation_kind],
        ["Symbol / session", `${episode.symbol} / ${episode.session_date}`],
        ["Trigger checkpoint", episode.trigger_checkpoint],
        ["M1C probability", episode.m1c_probability],
        ["Previous probability", episode.previous_m1c_probability],
        ["Bottom 5 / 10 / 20", `${clean(episode.bottom_5)} / ${clean(episode.bottom_10)} / ${clean(episode.bottom_20)}`],
        ["Episode number", episode.episode_number],
        ["Minutes since previous", episode.minutes_since_previous_quiet_episode],
        ["High tail ±60m", `${clean(episode.previous_high_tail_within_60_minutes)} / ${clean(episode.following_high_tail_within_60_minutes)}`],
        ["Option context valid", episode.option_context_valid],
        ["Cohort phase", episode.phase],
        ["Completion", episode.completion_status],
        ["Model hash", short(episode.model_hash)],
        ["Feature hash", short(episode.feature_hash)],
        ["Quality flags", (episode.data_quality_flags || []).join(", ")],
      ])),
      subsection("Underlying path and maximum excursion", table(
        [
          { label: "Horizon", value: "horizon_label" },
          { label: "Target", value: "target_timestamp_utc", format: clock },
          { label: "Path evidence", value: (row) => short(JSON.stringify(row.payload || {}), 84) },
          { label: "Quality", value: (row) => (row.quality_flags || []).join(", ") },
        ],
        detail.underlying_path || [],
      )),
      subsection("Microstructure windows", table(
        [
          { label: "Window", value: "window_name" },
          { label: "Start", value: "window_start_utc", format: clock },
          { label: "End", value: "window_end_utc", format: clock },
          { label: "Evidence", value: (row) => short(JSON.stringify(row.summary || {}), 84) },
          { label: "Quality", value: (row) => (row.quality_flags || []).join(", ") },
        ],
        detail.microstructure || [],
      )),
      subsection("Bounded option contracts", quietOptionTable(options.items || [])),
      subsection("Conservative shadow structures", table(
        [
          { label: "Bucket", value: "dte_bucket" },
          { label: "Structure", value: "structure_type" },
          { label: "Horizon", value: "horizon_label" },
          { label: "Opening", value: "opening_credit_or_debit" },
          { label: "Max risk", value: "maximum_defined_risk" },
          { label: "P&L", value: "conservative_pnl" },
          { label: "Return / risk", value: "return_on_maximum_risk" },
          { label: "Short touched", value: "short_strike_touched" },
          { label: "Wing touched", value: "protective_wing_touched" },
          { label: "Quality", value: "quality_status" },
        ],
        detail.shadow_structures || [],
      )),
      subsection("Frozen thresholds", jsonBlock(detail.frozen_thresholds || {})),
    );
    replace("quiet-episode-evidence", stack);
  } catch (error) {
    if (isAbort(error)) return;
    replace(
      "quiet-episode-evidence",
      node("div", "empty-state value-danger", "Quiet-state evidence is unavailable."),
    );
  }
}

function renderQuietShadow() {
  replace("quiet-shadow-list", table(
    [
      { label: "Episode", value: "observation_id", format: short },
      { label: "Cohort phase", value: "phase" },
      { label: "Symbol", value: "symbol" },
      { label: "M1C p", value: "m1c_probability" },
      { label: "DTE bucket", value: "dte_bucket" },
      { label: "Structure", value: "structure_type" },
      { label: "Opening credit/debit", value: "opening_credit_or_debit" },
      { label: "Maximum risk", value: "maximum_defined_risk" },
      { label: "Exit", value: "horizon_label" },
      { label: "Conservative P&L", value: "conservative_pnl" },
      { label: "Return / risk", value: "return_on_maximum_risk" },
      { label: "Short touched", value: "short_strike_touched" },
      { label: "Wing touched", value: "protective_wing_touched" },
      { label: "Quote quality", value: "quality_status" },
    ],
    state.quietShadow,
  ));
  replace("quiet-session-quality", table(
    [
      { label: "Session", value: "session_date" },
      { label: "Observations", value: "observations" },
      { label: "Quiet", value: "quiet_episodes" },
      { label: "Neutral", value: "neutral_controls" },
      { label: "High tail", value: "high_tail_controls" },
      { label: "Complete observations", value: "complete_observations" },
      { label: "Attempted structures", value: "attempted_structures" },
      { label: "Complete quote quality", value: "complete_quote_quality_structures" },
      { label: "Strict quality", value: "strict_quote_quality_structures" },
    ],
    state.quietSessionQuality,
  ));
}

function renderConcentrationAudit() {
  const audit = state.concentrationAudit || {};
  const explanation = audit.month_explanation || {};
  const surprise = audit.surprise_explanation || {};
  const failedMonth = explanation.failed_stress_month || "2025-10";
  const monthTail = (audit.stress_month_tail_incidence || [])
    .find((row) => row.month === failedMonth) || {};
  const monthExposure = (audit.stress_month_exposure || [])
    .find((row) => row.month === failedMonth) || {};
  const summary = document.createDocumentFragment();
  [
    metric("Frozen decision", audit.original_decision, "Unchanged", "danger"),
    metric("Failed stress month", failedMonth, "October 2025"),
    metric("Source sessions", monthExposure.trading_sessions_in_source_calendar, `${clean(monthExposure.sessions_represented_in_joined_panel)} represented`),
    metric("Eligible-row share", percent(monthExposure.source_exposure_share), `${clean(monthExposure.eligible_checkpoint_rows)} eligible rows`),
    metric("Bottom-tail incidence", percent(monthTail.bottom_tail_incidence), "Within-month probability"),
    metric("Tail composition", percent(monthTail.bottom_tail_composition_share, 6), "529 / 1,426 frozen rows", "danger"),
    metric("Month explanation", explanation.month_concentration_explanation || explanation.explanation || "multiple causes", "Descriptive only"),
    metric("Surprise explanation", surprise.surprise_concentration_explanation || surprise.explanation || "small-count fragile", "Gate not waived"),
  ].forEach((item) => summary.append(item));
  replace("concentration-summary", summary);

  replace("month-audit-panel", table(
    [
      { label: "Month", value: "month" },
      { label: "Source sessions", value: "trading_sessions_in_source_calendar" },
      { label: "Joined sessions", value: "sessions_represented_in_joined_panel" },
      { label: "Planned stock-sessions", value: "planned_stock_sessions" },
      { label: "Valid option pairs", value: "valid_option_pairs" },
      { label: "Option coverage", value: "option_pair_coverage_rate", format: percent },
      { label: "Eligible rows", value: "eligible_checkpoint_rows" },
      { label: "Exposure share", value: "source_exposure_share", format: percent },
      { label: "Tail incidence", value: (row) => {
        const tail = (audit.stress_month_tail_incidence || [])
          .find((item) => item.month === row.month);
        return percent(tail?.bottom_tail_incidence);
      } },
      { label: "Tail composition", value: (row) => {
        const tail = (audit.stress_month_tail_incidence || [])
          .find((item) => item.month === row.month);
        return percent(tail?.bottom_tail_composition_share, 6);
      } },
      { label: "Fresh share", value: (row) => {
        const tail = (audit.stress_month_tail_incidence || [])
          .find((item) => item.month === row.month);
        return percent(tail?.fresh_episode_share);
      } },
    ],
    audit.stress_month_exposure || [],
  ));

  replace("representation-audit-panel", table(
    [
      { label: "Representation", value: "representation" },
      { label: "Month", value: "entity" },
      { label: "Rows", value: "rows" },
      { label: "Share", value: "share", format: percent },
      { label: "Below IV", value: "remains_below_iv_rate", format: percent },
      { label: "Mean residual", value: "mean_iv_residual" },
      { label: "1.5σ breach", value: "breach_1_5_sigma_rate", format: percent },
      { label: "2.0σ breach", value: "breach_2_0_sigma_rate", format: percent },
      { label: "1.5σ surprises", value: "surprise_1_5_count" },
      { label: "2.0σ surprises", value: "surprise_2_0_count" },
    ],
    audit.representation_month_concentration || [],
  ));

  replace("leave-month-panel", table(
    [
      { label: "Omitted month", value: "omitted_month" },
      { label: "Rows", value: "rows" },
      { label: "Sessions", value: "sessions" },
      { label: "Below IV", value: "remains_below_iv_rate", format: percent },
      { label: "NPV lift", value: "npv_lift", format: percent },
      { label: "Mean residual", value: "mean_iv_residual" },
      { label: "1.5σ breach", value: "breach_1_5_sigma_rate", format: percent },
      { label: "2.0σ breach", value: "breach_2_0_sigma_rate", format: percent },
      { label: "Fresh containment", value: "fresh_episode_1_5_sigma_containment", format: percent },
      { label: "Δ below IV", value: "difference_remains_below_iv_rate", format: percent },
      { label: "Status", value: "status" },
    ],
    audit.leave_one_month_out || [],
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
    `REPLAY ${clean(replay.state).toUpperCase()} // LIVE RECORDER ${clean(state.status.state).toUpperCase()} // IBKR CONNECTIONS ${replay.ibkr_connections_attempted || 0}`;
}

function applyDashboardSummary(summary) {
  state.health = summary.health;
  state.status = summary.recorder;
  state.universe = summary.current_universe?.items || [];
  renderStatus();
  renderUniverse();
  if (state.activeScreen === "safety-audit") renderAudit();
}

function isAbort(error) {
  return error?.name === "AbortError";
}

function showRefreshError(error) {
  if (isAbort(error)) return;
  const requestSuffix = error.requestId ? ` Request ${error.requestId}.` : "";
  const message = error.status === 401
    ? "Authentication required."
    : `Persistent recorder state is unavailable.${requestSuffix}`;
  replace("blocker-strip", node("span", "blocker", message));
  document.getElementById("last-sync").textContent =
    `SYNC FAILED ${new Date().toISOString()}`;
}

function refreshFast({ supersede = false } = {}) {
  return polling.run("fast", async (signal) => {
    const summary = await api("/api/dashboard/summary", { signal });
    applyDashboardSummary(summary);
    document.getElementById("last-sync").textContent =
      `SYNCHRONIZED ${new Date().toISOString()}`;
  }, { supersede });
}

function refreshSlow(screenId, { supersede = false } = {}) {
  const key = `slow:${screenId}`;
  if (screenId === "live-monitor") {
    return polling.run(key, async (signal) => {
      const [capabilities, budget] = await Promise.all([
        api("/api/recorder/capabilities", { signal }),
        api("/api/market-data-budget", { signal }),
      ]);
      state.capabilities = capabilities;
      state.budget = budget;
      renderStatus();
      renderBudgetTransfer();
    }, { supersede });
  }
  if (screenId === "signal-detail" || screenId === "options-recorder") {
    return polling.run(key, async (signal) => {
      const episodes = await api("/api/episodes", { signal });
      state.episodes = episodes.items || [];
      renderEpisodeIndex();
    }, { supersede });
  }
  if (screenId === "quiet-universe") {
    return polling.run(key, async (signal) => {
      const [quietStatus, quietUniverse] = await Promise.all([
        api("/api/quiet-state/status", { signal }),
        api("/api/quiet-state/universe", { signal }),
      ]);
      state.quietStatus = quietStatus;
      state.quietUniverse = quietUniverse.items || [];
      renderQuietUniverse();
    }, { supersede });
  }
  if (screenId === "quiet-episode") {
    return polling.run(key, async (signal) => {
      const [quietStatus, quietEpisodes] = await Promise.all([
        api("/api/quiet-state/status", { signal }),
        api("/api/quiet-state/episodes", { signal }),
      ]);
      state.quietStatus = quietStatus;
      state.quietEpisodes = quietEpisodes.items || [];
      renderQuietEpisodeIndex();
    }, { supersede });
  }
  if (screenId === "quiet-shadow") {
    return polling.run(key, async (signal) => {
      const quality = await api("/api/quiet-state/session-quality", { signal });
      state.quietSessionQuality = quality.items || [];
      renderQuietShadow();
    }, { supersede });
  }
  return Promise.resolve();
}

function refreshManualTier(screenId, { supersede = false } = {}) {
  const key = `manual:${screenId}`;
  if (screenId === "live-monitor") {
    return polling.run(key, async (signal) => {
      const [transfer, reportPackages] = await Promise.all([
        api("/api/source-transfer", { signal }),
        api("/api/reports/daily", { signal }),
      ]);
      state.transfer = transfer;
      state.reportPackages = reportPackages.items || [];
      renderBudgetTransfer();
    }, { supersede });
  }
  if (screenId === "shadow-blotter") {
    return polling.run(key, async (signal) => {
      const shadow = await api("/api/shadow-outcomes", { signal });
      state.shadow = shadow.items || [];
      renderShadow();
    }, { supersede });
  }
  if (screenId === "safety-audit") {
    return polling.run(key, async (signal) => {
      const [audit, reports] = await Promise.all([
        api("/api/audit/events?limit=100", { signal }),
        api("/api/recorder/session-reports", { signal }),
      ]);
      state.audit = audit.items || [];
      state.sessionReports = reports.items || [];
      renderAudit();
    }, { supersede });
  }
  if (screenId === "quiet-shadow") {
    return polling.run(key, async (signal) => {
      const quietShadow = await api("/api/quiet-state/shadow-structures", { signal });
      state.quietShadow = quietShadow.items || [];
      renderQuietShadow();
    }, { supersede });
  }
  if (screenId === "concentration-audit") {
    return polling.run(key, async (signal) => {
      state.concentrationAudit = await api(
        "/api/quiet-state/concentration-audit",
        { signal },
      );
      renderConcentrationAudit();
    }, { supersede });
  }
  if (screenId === "signal-detail" && state.episodes.length) {
    return polling.run(
      key,
      (signal) => loadEpisode(
        state.selectedEpisode || state.episodes[0].episode_id,
        signal,
      ),
      { supersede },
    );
  }
  if (screenId === "options-recorder" && state.episodes.length) {
    return polling.run(key, (signal) => loadSelectedOptions(signal), { supersede });
  }
  if (screenId === "quiet-episode" && state.quietEpisodes.length) {
    return polling.run(
      key,
      (signal) => loadQuietEpisode(
        state.selectedQuietEpisode || state.quietEpisodes[0].observation_id,
        signal,
      ),
      { supersede },
    );
  }
  return Promise.resolve();
}

function refreshAll({ manual = false, screenActivation = false } = {}) {
  if (refreshAllPromise && !manual && !screenActivation) return refreshAllPromise;
  const supersede = manual || screenActivation;
  if (supersede) polling.cancelAll();
  const generation = ++refreshGeneration;
  const screenId = state.activeScreen;
  const button = document.getElementById("refresh");
  if (manual) {
    button.disabled = true;
    button.textContent = "Reading…";
  }
  const promise = (async () => {
    try {
      await refreshFast({ supersede });
      await refreshSlow(screenId, { supersede });
      if (manual || screenActivation) {
        await refreshManualTier(screenId, { supersede });
      }
    } catch (error) {
      showRefreshError(error);
    } finally {
      if (generation === refreshGeneration) {
        refreshAllPromise = null;
        button.disabled = false;
        button.textContent = "Refresh evidence";
      }
    }
  })();
  refreshAllPromise = promise;
  return promise;
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
    if (!state.status) state.status = {};
    state.status.replay = result;
    if (state.health) renderAudit();
  } catch (error) {
    document.getElementById("replay-state").textContent = "REPLAY CONTROL REJECTED";
  }
}

function runAutomatic(promise) {
  promise.catch(showRefreshError);
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    showScreen(button.dataset.screen);
    runAutomatic(refreshAll({ screenActivation: true }));
  });
});
document.getElementById("refresh").addEventListener(
  "click",
  () => refreshAll({ manual: true }),
);
document.getElementById("option-episode").addEventListener("change", () => {
  runAutomatic(polling.run(
    "detail:options",
    (signal) => loadSelectedOptions(signal),
    { supersede: true },
  ));
});
document.getElementById("replay-start").addEventListener("click", () => controlReplay("start"));
document.getElementById("replay-stop").addEventListener("click", () => controlReplay("stop"));
runAutomatic(refreshAll({ screenActivation: true }));
setInterval(() => {
  if (document.visibilityState === "visible") runAutomatic(refreshFast());
}, pollingPolicy.fastIntervalMs);
setInterval(() => {
  if (document.visibilityState === "visible") {
    runAutomatic(refreshSlow(state.activeScreen));
  }
}, pollingPolicy.slowIntervalMs);
setInterval(() => {
  if (document.visibilityState === "visible") {
    runAutomatic(refreshManualTier(state.activeScreen));
  }
}, pollingPolicy.manualIntervalMs);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") runAutomatic(refreshFast());
});
