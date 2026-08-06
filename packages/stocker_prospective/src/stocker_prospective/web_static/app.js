"use strict";

import {
  DashboardPollCoordinator,
  POLLING_POLICY,
  detailRequestPlan,
  endpointsForScreen,
} from "/assets/polling.mjs";

const state = {
  health: null,
  status: null,
  capabilities: null,
  universe: [],
  episodes: [],
  shadow: [],
  virtualLedgers: {
    opening_reversal: { items: [] },
    opening_leader: { items: [] },
    quiet_state: { items: [] },
  },
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
  openingLeader: null,
  selectedQuietEpisode: null,
  sectionErrors: {},
  lastSnapshotAt: null,
};

let episodeController = null;
let quietEpisodeController = null;

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
  const used = state.status.subscriptions?.[kind] ?? capacity.used ?? 0;
  return `${used} / ${capacity.available ?? "—"}`;
}

function renderStatus() {
  const status = state.status;
  const capabilities = state.capabilities || {
    manifest: null,
    scientific_recording_valid: false,
    diagnostic_display_allowed: true,
    required_market_data_type: "live",
  };
  const manifest = capabilities.manifest || {};
  const noOrderChecks = state.health.no_order_checks || {};
  const noOrderVerified = noOrderChecks.aggregate_no_order_verdict === true;
  const blockers = [
    ...(state.health.blockers || []),
    ...(manifest.blockers || []),
    ...Object.values(state.sectionErrors || {}).map((item) => item.error_code),
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
  const operational = status.operational || {};
  const operationalTimestamps = operational.timestamps || {};
  const latest = status.latest_checkpoint || {};
  const grid = document.createDocumentFragment();
  [
    metric(
      "Recorder readiness",
      status.state,
      operational.reason_code || status.banner,
      status.state === "RECORDING_HEALTHY" ? "ok" : "danger",
    ),
    metric("Process heartbeat", clock(operationalTimestamps.process_heartbeat_at_utc), "Recorder loop liveness"),
    metric("Latest callback received", clock(operationalTimestamps.latest_callback_received_at_utc), "External callback boundary"),
    metric("Latest callback admitted", clock(operationalTimestamps.latest_callback_durably_admitted_at_utc), "SQLite WAL inbox commit"),
    metric("Latest raw partition", clock(operationalTimestamps.latest_raw_partition_committed_at_utc), "Immutable Parquet + manifest"),
    metric("Latest inbox acknowledgement", clock(operationalTimestamps.latest_inbox_acknowledgement_at_utc), "Processing commit precedes acknowledgement"),
    metric("Latest completed 5m bar", clock(operationalTimestamps.latest_completed_five_minute_bar_at_utc), "Scientific bar clock"),
    metric("Latest checkpoint", clock(operationalTimestamps.latest_successful_checkpoint_at_utc), "Successful frozen checkpoint"),
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
    metric(
      "No-order state",
      noOrderVerified ? "VERIFIED" : "UNVERIFIED",
      "Derived from named adapter, route, database, configuration, and mutation checks",
      noOrderVerified ? "ok" : "danger",
    ),
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
    ["Diagnostic status", transfer.cross_vendor_validation_status || "not_configured"],
    ["Diagnostic decision", transfer.decision],
    ["Valid sessions", transfer.valid_session_count || 0],
    ["Recorder blocking", transfer.recorder_blocking === true],
    ["Diagnostic only", transfer.cross_vendor_validation_diagnostic_only !== false],
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

function m1cEvidenceStatus(row) {
  if (row.m1c_probability === null || row.m1c_probability === undefined) {
    return "awaiting checkpoint";
  }
  if (row.m1c_scientific_eligible === true) return "scientific-eligible";
  const reasons = Array.isArray(row.m1c_rejection_reasons)
    ? row.m1c_rejection_reasons.filter((reason) => typeof reason === "string")
    : [];
  return reasons.length
    ? `engineering shadow: ${reasons.join(", ")}`
    : "engineering shadow";
}

function renderUniverse() {
  replace("universe-panel", table(
    [
      { label: "Symbol", value: "symbol" },
      { label: "Last completed bar", value: "last_completed_bar", format: clock },
      { label: "M1C p", value: "m1c_probability" },
      { label: "Gate", value: "m1c_threshold" },
      { label: "Distance", value: "distance_from_threshold" },
      { label: "Evidence status", value: m1cEvidenceStatus },
      {
        label: "Quote diagnostics",
        value: (row) => (row.m1c_diagnostic_quality_flags || []).join(", ") || "clear",
      },
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
  if (episodeController) episodeController.abort();
  const controller = new AbortController();
  episodeController = controller;
  state.selectedEpisode = episodeId;
  replace("signal-evidence", node("div", "empty-state", "Reading causal evidence…"));
  try {
    const [detail, microstructure, options] = await Promise.all([
      api(`/api/episodes/${encodeURIComponent(episodeId)}`, { signal: controller.signal }),
      api(`/api/episodes/${encodeURIComponent(episodeId)}/microstructure`, {
        signal: controller.signal,
      }),
      api(`/api/episodes/${encodeURIComponent(episodeId)}/options`, {
        signal: controller.signal,
      }),
    ]);
    if (episodeController !== controller) return;
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
    if (error.name === "AbortError") return;
    replace("signal-evidence", node("div", "empty-state value-danger", "Episode evidence is unavailable."));
  } finally {
    if (episodeController === controller) episodeController = null;
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

function renderVirtualLedgers() {
  const opening = state.virtualLedgers.opening_reversal?.items || [];
  const openingLeader = (state.virtualLedgers.opening_leader?.items || []).map((item) => ({
    ...item,
    ...(item.accounting || {}),
    accounting_status: item.status,
  }));
  const quietCaptures = state.virtualLedgers.quiet_state?.capture_items || [];
  const quiet = state.virtualLedgers.quiet_state?.items || [];
  const quietCaptureRows = quietCaptures.flatMap((capture) => {
    const contracts = capture.contracts || [];
    const summary = {
      lifecycle_state: capture.lifecycle_state,
      session_date: capture.session_date,
      symbol: capture.symbol,
      observation_id: capture.observation_id,
      status_reason: capture.status_reason,
    };
    if (!contracts.length) return [summary];
    return contracts.map((contract) => ({ ...summary, ...contract }));
  });
  const quietFinalRows = quiet.flatMap((position) => {
    const legs = position.legs || [];
    if (!legs.length) return [position];
    return legs.map((leg) => ({ ...position, ...leg }));
  });
  replace("opening-reversal-virtual-ledger", table(
    [
      { label: "State", value: "lifecycle_state" },
      { label: "Session", value: "session_date" },
      { label: "Symbol", value: "symbol" },
      { label: "Direction", value: "predicted_direction" },
      { label: "Right", value: "right" },
      { label: "1DTE contract", value: "con_id" },
      { label: "Strike", value: "strike" },
      { label: "Latest bid", value: "latest_observed_bid" },
      { label: "Latest ask", value: "latest_observed_ask" },
      { label: "Latest quote", value: "latest_quote_received_at_utc", format: clock },
      { label: "Entry ask", value: "entry_ask" },
      { label: "Exit bid", value: "exit_bid" },
      { label: "Gross quote P&L", value: "gross_quote_pnl" },
      { label: "Pair outcomes", value: (row) => `${row.pair_complete_count} / 2` },
      { label: "Blocker / wait", value: "status_reason" },
    ],
    opening,
  ));
  replace("opening-leader-option-ledger", table(
    [
      { label: "Session", value: "session_date" },
      { label: "Checkpoint", value: "checkpoint" },
      { label: "Symbol", value: "selected_symbol" },
      { label: "Observation", value: "observation_name" },
      { label: "Strategy", value: "strategy_name" },
      { label: "Status", value: "accounting_status" },
      { label: "Gross executable P&L", value: "gross_executable_pnl" },
      { label: "Net executable P&L", value: "net_option_pnl" },
      { label: "Primary capital basis", value: "primary_capital_basis" },
      { label: "Primary capital", value: "primary_capital_amount" },
      { label: "Primary ROI", value: "primary_roi", format: percent },
      { label: "Margin ROI", value: "entry_margin_roi", format: percent },
      { label: "MAE", value: "maximum_adverse_excursion" },
      { label: "Max drawdown", value: "maximum_drawdown" },
      { label: "Greek status", value: "theta_attribution_status" },
      { label: "Reason", value: "reason" },
    ],
    openingLeader,
  ));
  replace("quiet-state-capture-ledger", table(
    [
      { label: "State", value: "lifecycle_state" },
      { label: "Session", value: "session_date" },
      { label: "Symbol", value: "symbol" },
      { label: "Observation", value: "observation_id", format: short },
      { label: "Contract", value: "con_id" },
      { label: "Right", value: "right" },
      { label: "Strike", value: "strike" },
      { label: "DTE", value: "dte_bucket" },
      { label: "Latest bid", value: "latest_bid" },
      { label: "Latest ask", value: "latest_ask" },
      { label: "Quote received", value: "latest_quote_received_at_utc", format: clock },
      { label: "Market data", value: "latest_market_data_type" },
      { label: "Recording", value: "latest_recording_status" },
      { label: "Blocker / wait", value: "status_reason" },
    ],
    quietCaptureRows,
  ));
  replace("quiet-state-virtual-ledger", table(
    [
      { label: "State", value: "lifecycle_state" },
      { label: "Session", value: "session_date" },
      { label: "Symbol", value: "symbol" },
      { label: "Structure", value: "structure_type" },
      { label: "DTE", value: "dte_bucket" },
      { label: "Horizon", value: "horizon_label" },
      { label: "Leg", value: "side" },
      { label: "Contract", value: "con_id" },
      { label: "Right", value: "right" },
      { label: "Strike", value: "strike" },
      { label: "Entry bid", value: "entry_bid" },
      { label: "Entry ask", value: "entry_ask" },
      { label: "Entry side", value: "entry_fill_price" },
      { label: "Exit bid", value: "exit_bid" },
      { label: "Exit ask", value: "exit_ask" },
      { label: "Exit side", value: "exit_fill_price" },
      { label: "Structure P&L", value: "conservative_pnl" },
      { label: "Quality", value: "quality_status" },
      { label: "Scientific", value: "scientific_option_evidence" },
      { label: "Blocker", value: "status_reason" },
    ],
    quietFinalRows,
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
    (row) => loadQuietEpisode(row.observation_id),
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

async function loadQuietEpisode(observationId) {
  if (quietEpisodeController) quietEpisodeController.abort();
  const controller = new AbortController();
  quietEpisodeController = controller;
  state.selectedQuietEpisode = observationId;
  replace("quiet-episode-evidence", node("div", "empty-state", "Reading quiet-state evidence…"));
  try {
    const [detail, options] = await Promise.all([
      api(`/api/quiet-state/episodes/${encodeURIComponent(observationId)}`, {
        signal: controller.signal,
      }),
      api(`/api/quiet-state/episodes/${encodeURIComponent(observationId)}/options`, {
        signal: controller.signal,
      }),
    ]);
    if (quietEpisodeController !== controller) return;
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
    if (error.name === "AbortError") return;
    replace(
      "quiet-episode-evidence",
      node("div", "empty-state value-danger", "Quiet-state evidence is unavailable."),
    );
  } finally {
    if (quietEpisodeController === controller) quietEpisodeController = null;
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
  const noOrderChecks = state.health.no_order_checks || {};
  const noOrderVerified = noOrderChecks.aggregate_no_order_verdict === true;
  [
    safetyCard("No-order path", noOrderVerified ? "VERIFIED ABSENT" : "UNVERIFIED", "Named checks cover order, account, position, execution, read-only database, and broker mutation evidence.", noOrderVerified ? "ok" : "danger"),
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
      { label: "Recorder state", value: (row) => row.report?.recorder_operational_state },
      { label: "Scientific valid", value: (row) => row.report?.scientific_recording_valid },
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

function openingLeaderCheckpointPanel(checkpoint) {
  const wrapper = node("div");
  const support = checkpoint.support || {};
  const shadow = checkpoint.latest_hypothetical_underlying_return || {};
  const optionSnapshots = Object.values(checkpoint.option_snapshots || {});
  const optionAccountingRows = Object.entries(checkpoint.option_strategy_accounting || {})
    .flatMap(([observation, strategies]) => Object.values(strategies || {}).map((mark) => {
      const accounting = mark.accounting || {};
      return {
        observation,
        strategy: mark.strategy_name,
        status: mark.status,
        gross_pnl: accounting.gross_executable_pnl,
        net_pnl: accounting.net_option_pnl,
        capital_basis: accounting.primary_capital_basis,
        capital_amount: accounting.primary_capital_amount,
        primary_roi: accounting.primary_roi,
        margin_roi: accounting.entry_margin_roi,
        greek_status: accounting.theta_attribution_status,
      };
    }));
  wrapper.append(
    kvGrid([
      ["Eligibility", checkpoint.eligibility],
      ["Slate size", checkpoint.slate_size],
      ["Rank 1", checkpoint.rank_1],
      ["Rank 2", checkpoint.rank_2],
      ["Rank-1 return (bps)", checkpoint.rank_1_return_from_open_bps],
      ["Leader separation (bps)", checkpoint.leader_separation_bps],
      ["Feed", checkpoint.source_feed_status],
      ["Signal receipt", short(checkpoint.signal_receipt, 18)],
      ["Ask→bid net (bps)", shadow.conservative_ask_to_bid_net_bps],
      ["Option snapshots", `${optionSnapshots.filter((item) => item?.status === "AVAILABLE").length} / ${optionSnapshots.length}`],
    ]),
    subsection("Independent prospective support", kvGrid([
      ["Valid sessions", `${support.valid_sessions || 0} / 60`],
      ["Calendar months", `${support.calendar_months || 0} / 3`],
      ["Selected stocks", `${support.distinct_selected_stocks || 0} / 15`],
      ["Complete", support.complete],
    ])),
    subsection("Rank persistence (diagnostic only)", jsonBlock(checkpoint.rank_persistence)),
    subsection("Executable option accounting", table(
      [
        { label: "Observation", value: "observation" },
        { label: "Strategy", value: "strategy" },
        { label: "Status", value: "status" },
        { label: "Gross P&L", value: "gross_pnl" },
        { label: "Net P&L", value: "net_pnl" },
        { label: "Primary basis", value: "capital_basis" },
        { label: "Capital", value: "capital_amount" },
        { label: "Primary ROI", value: "primary_roi", format: percent },
        { label: "Margin ROI", value: "margin_roi", format: percent },
        { label: "Greek status", value: "greek_status" },
      ],
      optionAccountingRows,
    )),
    subsection("M1C context only", jsonBlock(checkpoint.m1c_context)),
  );
  return wrapper;
}

function renderOpeningLeader() {
  const recorder = state.openingLeader;
  if (!recorder) return;
  const c6 = recorder.checkpoints?.C6 || {};
  const c12 = recorder.checkpoints?.C12 || {};
  const grid = node("div", "metric-grid");
  [
    metric("Recorder", recorder.recorder_status, recorder.banner),
    metric("Sample", recorder.sample_status, "C6 and C12 never pooled", recorder.sample_status === "PROSPECTIVE SAMPLE INCOMPLETE" ? "warn" : "ok"),
    metric("C6 / primary", c6.eligibility, `${clean(c6.rank_1)} over ${clean(c6.rank_2)}`),
    metric("C12 / secondary", c12.eligibility, `${clean(c12.rank_1)} over ${clean(c12.rank_2)}`),
    metric("M1C role", recorder.m1c_role, "Cannot admit, reject or rerank"),
    metric("Option policy", recorder.option_policy_authorized ? "AUTHORIZED" : "NOT AUTHORIZED", "Quotes are evidence only"),
  ].forEach((item) => grid.append(item));
  replace("opening-leader-status-grid", grid);
  replace("opening-leader-c6", openingLeaderCheckpointPanel(c6));
  replace("opening-leader-c12", openingLeaderCheckpointPanel(c12));

  const observationRows = [];
  [["C6", c6], ["C12", c12]].forEach(([label, checkpoint]) => {
    const merged = {
      ...(checkpoint.observations || {}),
      ...(checkpoint.pre_close_observations || {}),
      FINAL_CONTINUOUS: checkpoint.final_continuous_observation,
    };
    Object.entries(merged).forEach(([name, payload]) => {
      if (!payload) return;
      const quote = payload.quote || {};
      observationRows.push({
        checkpoint: label,
        observation: name,
        timestamp: quote.actual_quote_timestamp_utc,
        bid: quote.bid,
        ask: quote.ask,
        midpoint: quote.midpoint,
        quote_age_seconds: quote.quote_age_seconds,
        rank: payload.rank_persistence?.current_rank,
        conservative_net_bps: payload.shadow_return?.conservative_ask_to_bid_net_bps,
        status: payload.status || quote.market_data_status,
      });
    });
  });
  replace("opening-leader-observations", table(
    [
      { label: "Checkpoint", value: "checkpoint" },
      { label: "Observation", value: "observation" },
      { label: "Timestamp", value: "timestamp", format: clock },
      { label: "Bid", value: "bid" },
      { label: "Ask", value: "ask" },
      { label: "Mid", value: "midpoint" },
      { label: "Age (s)", value: "quote_age_seconds" },
      { label: "Current rank", value: "rank" },
      { label: "Ask→bid net (bps)", value: "conservative_net_bps" },
      { label: "Feed/status", value: "status" },
    ],
    observationRows,
  ));
  const warnings = recorder.data_quality_warnings || [];
  replace(
    "opening-leader-warnings",
    warnings.length
      ? table([{ label: "Warning", value: "warning" }], warnings.map((warning) => ({ warning })))
      : node("div", "empty-state", "No Opening Leader data-quality warnings recorded."),
  );
}

function activeScreenId() {
  return document.querySelector(".screen.is-active")?.id || "live-monitor";
}

function sectionErrorKey(path) {
  return path
    .split("?", 1)[0]
    .replace(/^\/api\//, "")
    .replaceAll("/", "_")
    .replaceAll("-", "_");
}

function applySummary(summary) {
  const recorder = summary.recorder || {};
  const previousStatus = state.status || {};
  const previousTimestamps = previousStatus.operational?.timestamps || {};
  const completedBar = recorder.latest_completed_five_minute_bar || null;
  const checkpoint = recorder.latest_successful_checkpoint || null;
  state.health = {
    blockers: (summary.alerts || []).map((item) => item.code),
    no_order_checks: summary.no_order || {},
  };
  state.status = {
    ...previousStatus,
    run_id: summary.run_id,
    state: recorder.state,
    operational: {
      ...(previousStatus.operational || {}),
      reason_code: recorder.reason_code,
      timestamps: {
        ...previousTimestamps,
        process_heartbeat_at_utc: recorder.heartbeat_at_utc,
        latest_callback_received_at_utc: recorder.latest_callback_received_at_utc,
        latest_callback_durably_admitted_at_utc:
          recorder.latest_callback_durably_admitted_at_utc,
        latest_inbox_acknowledgement_at_utc:
          recorder.latest_inbox_acknowledgement_at_utc,
        latest_completed_five_minute_bar_at_utc: completedBar?.bar_end_utc || null,
        latest_successful_checkpoint_at_utc: checkpoint?.bar_end_utc || null,
      },
      inbox: recorder.callback_inbox || {},
    },
    latest_checkpoint: checkpoint,
    latest_completed_bar: completedBar,
    latest_episode: recorder.latest_episode || null,
    ibkr_connection: summary.ibkr?.connection || null,
    subscriptions: summary.ibkr?.subscriptions?.by_kind || {},
    replay: summary.replay,
  };
  state.lastSnapshotAt = summary.summary_at_utc || null;
  delete state.sectionErrors.dashboard_summary;
}

function applyEndpoint(path, payload) {
  const endpoint = path.split("?", 1)[0];
  if (endpoint === "/api/recorder/status") {
    state.status = {
      ...(state.status || {}),
      ...payload,
      operational: {
        ...(state.status?.operational || {}),
        ...(payload.operational || {}),
      },
    };
  } else if (endpoint === "/api/recorder/capabilities") state.capabilities = payload;
  else if (endpoint === "/api/market-data-budget") state.budget = payload;
  else if (endpoint === "/api/universe/live") state.universe = payload.items || [];
  else if (endpoint === "/api/episodes") state.episodes = payload.items || [];
  else if (endpoint === "/api/shadow-outcomes") state.shadow = payload.items || [];
  else if (endpoint === "/api/virtual-ledgers") state.virtualLedgers = payload;
  else if (endpoint === "/api/audit/events") state.audit = payload.items || [];
  else if (endpoint === "/api/recorder/session-reports") {
    state.sessionReports = payload.items || [];
  } else if (endpoint === "/api/quiet-state/status") state.quietStatus = payload;
  else if (endpoint === "/api/quiet-state/universe") {
    state.quietUniverse = payload.items || [];
  } else if (endpoint === "/api/quiet-state/episodes") {
    state.quietEpisodes = payload.items || [];
  } else if (endpoint === "/api/quiet-state/shadow-structures") {
    state.quietShadow = payload.items || [];
  } else if (endpoint === "/api/quiet-state/session-quality") {
    state.quietSessionQuality = payload.items || [];
  } else if (endpoint === "/api/quiet-state/concentration-audit") {
    state.concentrationAudit = payload;
  } else if (endpoint === "/api/source-transfer") state.transfer = payload;
  else if (endpoint === "/api/reports/daily") {
    state.reportPackages = payload.items || [];
  } else if (endpoint === "/api/opening-leader-continuation-v0") {
    state.openingLeader = payload;
  }
  delete state.sectionErrors[sectionErrorKey(path)];
}

async function refreshFast(signal) {
  const summary = await api("/api/dashboard/summary", { signal });
  applySummary(summary);
}

async function refreshScreenTier(tier, screenId, signal) {
  const paths = endpointsForScreen(tier, screenId);
  const results = await Promise.allSettled(
    paths.map(async (path) => [path, await api(path, { signal })]),
  );
  for (const result of results) {
    if (result.status === "fulfilled") {
      applyEndpoint(result.value[0], result.value[1]);
      continue;
    }
    if (result.reason?.name === "AbortError" || result.reason?.status === 401) {
      throw result.reason;
    }
    const path = paths[results.indexOf(result)];
    const key = sectionErrorKey(path);
    state.sectionErrors[key] = {
      error_code: `DASHBOARD_SECTION_${key.toUpperCase()}_UNAVAILABLE`,
    };
  }
}

function renderDashboard() {
  if (!state.health || !state.status) return;
  renderStatus();
  renderUniverse();
  renderEpisodeIndex();
  renderShadow();
  renderVirtualLedgers();
  renderAudit();
  if (state.quietStatus) {
    renderQuietUniverse();
    renderQuietEpisodeIndex();
  }
  renderQuietShadow();
  renderConcentrationAudit();
  renderBudgetTransfer();
  renderOpeningLeader();
}

async function performRefresh({
  tier = "fast",
  refreshDetails = false,
  signal,
} = {}) {
  const button = document.getElementById("refresh");
  const screenId = activeScreenId();
  const showButtonBusy = refreshDetails;
  if (showButtonBusy) {
    button.disabled = true;
    button.textContent = "Reading…";
  }
  try {
    if (tier === "fast" || refreshDetails) {
      await refreshFast(signal);
      if (!state.health || !state.status) {
        throw new Error("dashboard_core_sections_unavailable");
      }
      renderDashboard();
    }
    const tasks = [];
    if (tier === "slow" || tier === "manual") {
      tasks.push(refreshScreenTier("slow", screenId, signal));
    }
    if (tier === "manual") {
      tasks.push(refreshScreenTier("manual", screenId, signal));
    }
    await Promise.all(tasks);
    if (!state.health || !state.status) {
      throw new Error("dashboard_core_sections_unavailable");
    }
    renderDashboard();

    if (
      state.selectedEpisode
      && !state.episodes.some((item) => item.episode_id === state.selectedEpisode)
    ) {
      if (episodeController) episodeController.abort();
      state.selectedEpisode = null;
    }
    if (
      state.selectedQuietEpisode
      && !state.quietEpisodes.some(
        (item) => item.observation_id === state.selectedQuietEpisode,
      )
    ) {
      if (quietEpisodeController) quietEpisodeController.abort();
      state.selectedQuietEpisode = null;
    }
    if (tier !== "fast" && ["signal-detail", "options-recorder"].includes(screenId)) {
      const legacyEpisodeToLoad = detailRequestPlan({
        selectedId: state.selectedEpisode,
        availableIds: state.episodes.map((item) => item.episode_id),
        explicitRefresh: refreshDetails,
      });
      if (legacyEpisodeToLoad) await loadEpisode(legacyEpisodeToLoad);
    }
    if (tier !== "fast" && screenId === "quiet-episode") {
      const quietEpisodeToLoad = detailRequestPlan({
        selectedId: state.selectedQuietEpisode,
        availableIds: state.quietEpisodes.map((item) => item.observation_id),
        explicitRefresh: refreshDetails,
      });
      if (quietEpisodeToLoad) await loadQuietEpisode(quietEpisodeToLoad);
    }
    const partial = Object.keys(state.sectionErrors).length > 0;
    document.getElementById("last-sync").textContent =
      `${partial ? "PARTIAL SNAPSHOT" : "SYNCHRONIZED"} ${clock(state.lastSnapshotAt)}`;
  } catch (error) {
    if (error.name === "AbortError") return;
    const message = error.status === 401
      ? "Authentication required."
      : state.lastSnapshotAt
        ? "Recorder snapshot is stale; the latest refresh is unavailable."
        : "Persistent recorder state is unavailable.";
    replace("blocker-strip", node("span", "blocker", message));
    document.getElementById("last-sync").textContent =
      state.lastSnapshotAt
        ? `STALE SINCE ${clock(state.lastSnapshotAt)}`
        : `UNAVAILABLE ${new Date().toISOString()}`;
  } finally {
    if (showButtonBusy) {
      button.disabled = false;
      button.textContent = "Refresh evidence";
    }
  }
}

const polling = new DashboardPollCoordinator({
  run: performRefresh,
  isVisible: () => document.visibilityState === "visible",
});

let refreshAllPromise = null;

function refreshAll({
  tier = "fast",
  refreshDetails = false,
  supersede = false,
} = {}) {
  if (refreshDetails) {
    if (episodeController) episodeController.abort();
    if (quietEpisodeController) quietEpisodeController.abort();
  }
  if (supersede) polling.cancelAll();
  const pending = polling.refresh({
    tier,
    refreshDetails,
    supersede,
  });
  refreshAllPromise = pending;
  void pending.then(
    () => {
      if (refreshAllPromise === pending) refreshAllPromise = null;
    },
    () => {
      if (refreshAllPromise === pending) refreshAllPromise = null;
    },
  );
  return pending;
}

function refreshForScreenActivation(screenId) {
  const tier = endpointsForScreen("manual", screenId).length ? "manual" : "slow";
  return refreshAll({ tier, supersede: true });
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
  button.addEventListener("click", () => {
    showScreen(button.dataset.screen);
    refreshForScreenActivation(button.dataset.screen);
  });
});
document.getElementById("refresh").addEventListener(
  "click",
  () => refreshAll({
    tier: "manual",
    refreshDetails: true,
    supersede: true,
  }),
);
document.getElementById("option-episode").addEventListener("change", (event) => {
  const episodeId = event.target.value;
  if (episodeId && episodeId !== state.selectedEpisode) loadEpisode(episodeId);
});
document.getElementById("replay-start").addEventListener("click", () => controlReplay("start"));
document.getElementById("replay-stop").addEventListener("click", () => controlReplay("stop"));
void refreshAll({ tier: "fast", supersede: true }).then(() => {
  refreshAll({ tier: "manual" });
});
setInterval(() => {
  if (document.visibilityState === "visible") refreshAll({ tier: "fast" });
}, POLLING_POLICY.fastIntervalMs);
setInterval(() => {
  if (document.visibilityState === "visible") refreshAll({ tier: "slow" });
}, POLLING_POLICY.slowIntervalMs);
setInterval(() => {
  if (document.visibilityState === "visible") refreshAll({ tier: "manual" });
}, POLLING_POLICY.manualIntervalMs);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    const screenId = activeScreenId();
    if (endpointsForScreen("manual", screenId).length) {
      refreshAll({ tier: "manual", supersede: true });
    } else {
      refreshAll({ tier: "fast", supersede: true });
      refreshAll({ tier: "slow" });
    }
  }
  else {
    polling.hide();
    if (episodeController) episodeController.abort();
    if (quietEpisodeController) quietEpisodeController.abort();
  }
});
