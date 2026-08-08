export const ENDPOINTS = Object.freeze({
  meta: "/api/v2/meta",
  live: "/api/v2/live?limit=50",
  ideas: "/api/v2/ideas?limit=50",
  results: "/api/v2/results?limit=50",
  diagnostics: "/api/v2/diagnostics?limit=50",
});

const VIRTUAL_LANGUAGE = "virtual only; no broker orders, fills, or positions";

function primitiveText(value) {
  if (value === null) return "null";
  if (value === undefined) return "—";
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

export function genericEntries(value, prefix = "", maximum = 64) {
  const entries = [];
  const seen = new WeakSet();

  function visit(node, path) {
    if (entries.length >= maximum) return;
    if (node === null || typeof node !== "object") {
      entries.push({ label: path || "value", value: primitiveText(node) });
      return;
    }
    if (seen.has(node)) {
      entries.push({ label: path || "value", value: "[circular]" });
      return;
    }
    seen.add(node);
    const children = Array.isArray(node)
      ? node.map((item, index) => [String(index), item])
      : Object.entries(node).sort(([left], [right]) => left.localeCompare(right));
    if (children.length === 0) {
      entries.push({ label: path || "value", value: Array.isArray(node) ? "[]" : "{}" });
      return;
    }
    for (const [key, child] of children) {
      visit(child, path ? `${path}.${key}` : key);
      if (entries.length >= maximum) break;
    }
  }

  visit(value, prefix);
  return entries;
}

export class PollCoordinator {
  constructor(loaders, { intervalMs = 10_000, onError = () => {}, onState = () => {} } = {}) {
    this.loaders = loaders;
    this.intervalMs = intervalMs;
    this.onError = onError;
    this.onState = onState;
    this.activeView = null;
    this.paused = false;
    this.inFlight = null;
    this.controller = null;
    this.timer = null;
    this.queued = false;
  }

  activate(view) {
    if (!Object.hasOwn(this.loaders, view)) throw new Error(`unknown view: ${view}`);
    const changed = this.activeView !== view;
    this.activeView = view;
    if (changed && this.controller) this.controller.abort();
    return this.refresh();
  }

  refresh() {
    if (this.paused || this.activeView === null) return Promise.resolve();
    if (this.inFlight) {
      this.queued = true;
      return this.inFlight;
    }
    clearTimeout(this.timer);
    const view = this.activeView;
    this.controller = new AbortController();
    this.onState({ phase: "loading", view });
    this.inFlight = Promise.resolve(this.loaders[view](this.controller.signal))
      .then(() => this.onState({ phase: "ready", view }))
      .catch((error) => {
        if (error?.name !== "AbortError") this.onError(error, view);
      })
      .finally(() => {
        this.inFlight = null;
        this.controller = null;
        const immediately = this.queued;
        this.queued = false;
        if (this.paused) return;
        if (immediately) {
          void this.refresh();
        } else {
          this.timer = setTimeout(() => void this.refresh(), this.intervalMs);
        }
      });
    return this.inFlight;
  }

  setVisible(visible) {
    this.paused = !visible;
    if (!visible) {
      clearTimeout(this.timer);
      this.queued = false;
      if (this.controller) this.controller.abort();
      return;
    }
    void this.refresh();
  }

  stop() {
    this.paused = true;
    clearTimeout(this.timer);
    if (this.controller) this.controller.abort();
  }
}

function element(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(options)) {
    if (key === "className") node.className = value;
    else if (key === "text") node.textContent = primitiveText(value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "onClick") node.addEventListener("click", value);
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

function replace(target, children) {
  target.replaceChildren(...children);
}

function formatTime(microseconds) {
  if (microseconds === null || microseconds === undefined) return "—";
  const date = new Date(Number(microseconds) / 1_000);
  return Number.isNaN(date.valueOf()) ? primitiveText(microseconds) : date.toLocaleString();
}

function formatBytes(value) {
  if (value === null || value === undefined) return "—";
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return primitiveText(value);
  const units = ["B", "KiB", "MiB", "GiB"];
  let amount = bytes;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function shortId(value) {
  const text = primitiveText(value);
  return text.length > 18 ? `${text.slice(0, 10)}…${text.slice(-6)}` : text;
}

function metric(label, value) {
  return element("dl", { className: "metric" }, [
    element("dt", { text: label }),
    element("dd", { text: value }),
  ]);
}

function fields(value, prefix = "", maximum = 64) {
  const list = element("dl", { className: "generic-fields" });
  for (const entry of genericEntries(value, prefix, maximum)) {
    list.append(element("dt", { text: entry.label }), element("dd", { text: entry.value }));
  }
  return list;
}

function empty(message) {
  return element("p", { className: "empty-state", text: message });
}

function table(columns, rows) {
  if (rows.length === 0) return empty("No rows in the current bounded projection.");
  const head = element("tr");
  for (const column of columns) head.append(element("th", { scope: "col", text: column.label }));
  const body = element("tbody");
  for (const row of rows) {
    const line = element("tr");
    for (const column of columns) {
      line.append(
        element("td", {
          className: column.numeric ? "numeric" : "",
          text: column.value(row),
        }),
      );
    }
    body.append(line);
  }
  return element("table", {}, [element("thead", {}, [head]), body]);
}

async function fetchJSON(path, signal) {
  const response = await fetch(path, {
    method: "GET",
    credentials: "same-origin",
    headers: { Accept: "application/json" },
    signal,
  });
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      // A non-JSON proxy response still produces a stable visible error.
    }
    throw new Error(`Read failed: ${detail}`);
  }
  return response.json();
}

const state = {
  selectedIdea: null,
  selectedResult: null,
  metaLoaded: false,
  diagnosticsOpener: null,
};

async function loadMeta(signal) {
  if (state.metaLoaded) return;
  const payload = await fetchJSON(ENDPOINTS.meta, signal);
  document.querySelector("#build-label").textContent =
    `Build ${payload.build.git_commit} · schema ${payload.schema_version}`;
  state.metaLoaded = true;
}

async function loadLive(signal) {
  const [payload] = await Promise.all([fetchJSON(ENDPOINTS.live, signal), loadMeta(signal)]);
  replace(document.querySelector("#live-summary"), [
    metric("Recorder", payload.recorder?.lifecycle ?? "not running"),
    metric(
      "Recorder health",
      payload.recorder ? (payload.recorder.reason ?? "healthy") : "unavailable",
    ),
    metric("IBKR", payload.ibkr?.connection_state ?? "unavailable"),
    metric("Freshness", formatTime(payload.ibkr?.freshness_at_us)),
    metric("Unresolved gaps", payload.gaps.unresolved),
  ]);
  replace(document.querySelector("#live-feeds"), [
    metric("Active feed count", payload.feeds.active),
    fields(payload.feeds.by_kind, "feed", 24),
  ]);
  replace(document.querySelector("#live-storage"), [
    fields(
      {
        database: formatBytes(payload.storage.database_bytes),
        wal: formatBytes(payload.storage.wal_bytes),
        callback_inbox_count: payload.callback_inbox.nonterminal,
        callback_inbox_bytes: formatBytes(payload.callback_inbox.bytes),
        backup_available: payload.backup.available,
        backup_entries: payload.backup.entries,
        latest_backup: payload.backup.latest,
      },
      "",
      12,
    ),
  ]);
  replace(document.querySelector("#live-instruments"), [
    table(
      [
        { label: "Instrument", value: (row) => row.symbol || row.instrument_id },
        { label: "Kind / feed", value: (row) => `${row.kind} / ${row.feed_kind}` },
        { label: "Bid", value: (row) => row.bid, numeric: true },
        { label: "Ask", value: (row) => row.ask, numeric: true },
        { label: "Last", value: (row) => row.last, numeric: true },
        { label: "Event time", value: (row) => formatTime(row.event_at_us) },
        { label: "Event", value: (row) => shortId(row.event_id) },
      ],
      payload.instruments,
    ),
  ]);
}

function pluginCard(plugin) {
  return element("article", { className: "plugin-card" }, [
    element("p", { className: "coordinate", text: `${plugin.idea_id} / ${plugin.idea_version}` }),
    element("h3", { text: plugin.display_name }),
    element("p", { text: plugin.description }),
    element("span", { className: "mono", text: `code ${shortId(plugin.code_hash)}` }),
  ]);
}

function ideaButton(item) {
  const button = element(
    "button",
    {
      className: `select-card${state.selectedIdea === item.instance_id ? " is-selected" : ""}`,
      type: "button",
      "aria-pressed": state.selectedIdea === item.instance_id ? "true" : "false",
      onClick: () => void selectIdea(item.instance_id, button),
    },
    [
      element("strong", { text: item.display_name }),
      element("span", {
        className: "health-badge",
        text: item.health,
        dataset: { health: item.health },
      }),
      element("small", { text: `${item.idea_id} · ${item.mode} · ${shortId(item.instance_id)}` }),
    ],
  );
  return button;
}

function outputEntry(output) {
  const common = {
    output_id: output.output_id,
    subject_instrument_id: output.subject_instrument_id,
    as_of_at_us: output.as_of_at_us,
    direction: output.direction,
    strength: output.strength,
    confidence: output.confidence,
    authority: output.authority,
    input: output.input,
    legs: output.legs,
  };
  return element("article", { className: "stream-entry" }, [
    element("header", {}, [
      element("h4", { text: formatTime(output.as_of_at_us) }),
      element("span", { className: "kind-badge", text: output.kind }),
    ]),
    fields(common, "common", 36),
    fields(output.payload, "payload", 64),
  ]);
}

async function selectIdea(instanceId, button) {
  state.selectedIdea = instanceId;
  for (const item of document.querySelectorAll("#idea-instances .select-card")) {
    const selected = item === button;
    item.classList.toggle("is-selected", selected);
    item.setAttribute("aria-pressed", selected ? "true" : "false");
  }
  const target = document.querySelector("#idea-detail");
  replace(target, [empty("Reading bounded generic output stream…")]);
  try {
    const payload = await fetchJSON(
      `/api/v2/ideas/${encodeURIComponent(instanceId)}?limit=50`,
      undefined,
    );
    replace(target, [
      element("div", { className: "detail-lead" }, [
        element("h4", { text: payload.instance.display_name }),
        element("p", { text: payload.instance.description }),
        fields(
          {
            health: payload.instance.health,
            mode: payload.instance.mode,
            instance_id: payload.instance.instance_id,
            manifest: payload.instance.manifest,
            parameters: payload.instance.parameters,
            universe: payload.instance.universe,
            requirements: payload.instance.requirements,
          },
          "instance",
          64,
        ),
      ]),
      ...(payload.outputs.length
        ? payload.outputs.map(outputEntry)
        : [empty("No sealed outputs in the requested seven-day window.")]),
    ]);
  } catch (error) {
    replace(target, [element("p", { className: "error-state", text: error.message })]);
  }
}

async function loadIdeas(signal) {
  const [payload] = await Promise.all([fetchJSON(ENDPOINTS.ideas, signal), loadMeta(signal)]);
  replace(document.querySelector("#plugin-catalog"),
    payload.plugins.length ? payload.plugins.map(pluginCard) : [empty("No discovered plugins.")],
  );
  replace(document.querySelector("#idea-instances"),
    payload.items.length ? payload.items.map(ideaButton) : [empty("No activated instances.")],
  );
}

function resultButton(item) {
  const button = element(
    "button",
    {
      className: `select-card${state.selectedResult === item.position_id ? " is-selected" : ""}`,
      type: "button",
      "aria-pressed": state.selectedResult === item.position_id ? "true" : "false",
      onClick: () => void selectResult(item.position_id, button),
    },
    [
      element("strong", { text: shortId(item.position_id) }),
      element("span", {
        className: "health-badge",
        text: item.status,
        dataset: { health: item.status },
      }),
      element("small", {
        text: `${item.instance_id} · ${formatTime(item.result_at_us)} · net ${primitiveText(item.net_pnl)}`,
      }),
    ],
  );
  return button;
}

async function selectResult(positionId, button) {
  state.selectedResult = positionId;
  for (const item of document.querySelectorAll("#result-list .select-card")) {
    const selected = item === button;
    item.classList.toggle("is-selected", selected);
    item.setAttribute("aria-pressed", selected ? "true" : "false");
  }
  const target = document.querySelector("#result-detail");
  replace(target, [empty("Reading bounded virtual position detail…")]);
  try {
    const payload = await fetchJSON(
      `/api/v2/results/${encodeURIComponent(positionId)}?limit=50`,
      undefined,
    );
    if (payload.language !== VIRTUAL_LANGUAGE) throw new Error("Virtual-only marker is absent");
    replace(target, [
      element("div", { className: "virtual-warning", text: payload.language }),
      fields(payload.position, "position", 48),
      element("article", { className: "stream-entry" }, [
        element("h4", { text: "Exact source proposal" }),
        fields(payload.source_proposal, "proposal", 48),
      ]),
      element("article", { className: "stream-entry" }, [
        element("h4", { text: "Market-event references and virtual legs" }),
        fields(payload.legs, "legs", 48),
      ]),
      element("article", { className: "stream-entry" }, [
        element("h4", { text: "Virtual marks and outcome" }),
        fields({ marks: payload.marks, outcome: payload.outcome }, "result", 64),
      ]),
    ]);
  } catch (error) {
    replace(target, [element("p", { className: "error-state", text: error.message })]);
  }
}

async function loadResults(signal) {
  const [payload] = await Promise.all([fetchJSON(ENDPOINTS.results, signal), loadMeta(signal)]);
  replace(document.querySelector("#result-aggregates"), [
    metric("Window positions", payload.aggregates.window_items),
    ...["open", "closed", "incomplete", "invalid"].map((status) =>
      metric(status, payload.aggregates.by_status[status] ?? 0),
    ),
  ]);
  replace(document.querySelector("#result-list"),
    payload.items.length ? payload.items.map(resultButton) : [empty("No shadow positions.")],
  );
}

function compactList(items, preferredKeys) {
  if (!items.length) return empty("None in this bounded projection.");
  const list = element("ul", { className: "compact-list" });
  for (const item of items) {
    const pairs = preferredKeys
      .filter((key) => item[key] !== undefined && item[key] !== null)
      .map((key) => `${key}: ${primitiveText(item[key])}`);
    list.append(element("li", { text: pairs.join(" · ") || JSON.stringify(item) }));
  }
  return list;
}

async function loadDiagnostics() {
  const payload = await fetchJSON(ENDPOINTS.diagnostics, undefined);
  replace(document.querySelector("#diagnostics-summary"), [
    metric("Retention", payload.retention.status),
    metric("Database", formatBytes(payload.database.database_bytes)),
    metric("WAL", formatBytes(payload.database.wal_bytes)),
    metric("Query only", payload.database.query_only),
  ]);
  replace(document.querySelector("#diagnostic-incidents"), [
    compactList(payload.incidents, ["severity", "code", "scope", "opened_at_us"]),
  ]);
  replace(document.querySelector("#diagnostic-gaps"), [
    compactList(payload.gaps, ["reason", "subscription_id", "data_loss_possible"]),
  ]);
  replace(document.querySelector("#diagnostic-subscriptions"), [
    compactList(payload.subscriptions, ["instrument_id", "feed_kind", "lifecycle"]),
  ]);
  replace(document.querySelector("#diagnostic-backups"), [
    metric("Backup state", payload.backups.status.state),
    compactList(payload.backups.items, ["tier", "created_at_us", "archive_filename", "compressed_bytes"]),
  ]);
  replace(document.querySelector("#diagnostic-hashes"), [fields(payload.hashes, "", 48)]);
  replace(document.querySelector("#diagnostic-retention"), [fields(payload.retention, "", 24)]);
}

function showView(view, coordinator, { focus = false } = {}) {
  for (const tab of document.querySelectorAll(".view-tab")) {
    const selected = tab.dataset.view === view;
    tab.classList.toggle("is-active", selected);
    tab.setAttribute("aria-selected", selected ? "true" : "false");
    tab.tabIndex = selected ? 0 : -1;
    if (selected && focus) tab.focus();
  }
  for (const panel of document.querySelectorAll(".view-panel")) {
    const selected = panel.id === `view-${view}`;
    panel.hidden = !selected;
    panel.classList.toggle("is-active", selected);
  }
  history.replaceState(null, "", `#${view}`);
  void coordinator.activate(view);
}

function startApplication() {
  const sync = document.querySelector(".read-state");
  const syncStatus = document.querySelector("#sync-status");
  const loaders = { live: loadLive, ideas: loadIdeas, results: loadResults };
  const coordinator = new PollCoordinator(loaders, {
    onState: ({ phase, view }) => {
      sync.classList.remove("is-error");
      syncStatus.textContent = phase === "loading" ? `Reading ${view}…` : `${view} current`;
    },
    onError: (error, view) => {
      sync.classList.add("is-error");
      syncStatus.textContent = `${view}: ${error.message}`;
    },
  });

  const tabs = [...document.querySelectorAll(".view-tab")];
  for (const tab of tabs) {
    tab.addEventListener("click", () => showView(tab.dataset.view, coordinator));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const current = tabs.indexOf(tab);
      const next = event.key === "Home"
        ? 0
        : event.key === "End"
          ? tabs.length - 1
          : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      showView(tabs[next].dataset.view, coordinator, { focus: true });
    });
  }

  document.querySelector("#refresh").addEventListener("click", () => void coordinator.refresh());
  document.addEventListener("visibilitychange", () => coordinator.setVisible(!document.hidden));

  const drawer = document.querySelector("#diagnostics-drawer");
  document.querySelector("#diagnostics-open").addEventListener("click", async (event) => {
    state.diagnosticsOpener = event.currentTarget;
    drawer.showModal();
    replace(document.querySelector("#diagnostics-summary"), [empty("Reading diagnostics…")]);
    try {
      await loadDiagnostics();
    } catch (error) {
      replace(document.querySelector("#diagnostics-summary"), [
        element("p", { className: "error-state", text: error.message }),
      ]);
    }
  });
  document.querySelector("#diagnostics-close").addEventListener("click", () => drawer.close());
  drawer.addEventListener("click", (event) => {
    if (event.target === drawer) drawer.close();
  });
  drawer.addEventListener("close", () => state.diagnosticsOpener?.focus());

  const requestedView = location.hash.slice(1);
  showView(Object.hasOwn(loaders, requestedView) ? requestedView : "live", coordinator);
  window.addEventListener("pagehide", () => coordinator.stop(), { once: true });
}

if (typeof document !== "undefined") startApplication();
