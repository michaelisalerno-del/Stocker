/* Real dashboard browser checks with explicitly labelled offline API fixtures. No broker. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const { chromium } = require("playwright");

const at = "2026-09-28T14:20:00Z";
const markets = ["BTC", "CL", "GC", "NG", "NQ", "SI"];
const staticRoot = path.resolve(
  "packages/stocker_dashboard/src/stocker_dashboard/static",
);
const output = path.resolve("docs/futures-screenshots");
const state = {
  system: {
    account: "OFFLINE FIXTURE",
    connected: true,
    reconciled: true,
    armed: false,
    paused: false,
    reserved_open_trades: 2,
    allocation_pennies: 2000,
    market_data: {
      owned_lines: 24,
      app_budget: 60,
      total_account_allowance: 100,
      allowance_status: "ASSUMED",
      external_headroom: 40,
      external_usage: null,
      depth_used: 3,
      depth_limit: 3,
      temporary_quotes: 5,
      temporary_quote_limit: 15,
      pacing: {
        outbound_last_second: 4,
        outbound_cap: 40,
        urgent_reserve: 10,
        queued: 0,
      },
      errors: [],
    },
    l2_recording: {
      assigned_markets: ["BTC", "GC", "NG"],
      disk_bytes: 12582912,
      disk_limit: 268435456,
      memory_bytes: 2097152,
      memory_limit: 33554432,
      recording_gaps: 1,
      writer_queue: 0,
    },
  },
  pnl: {
    realised_net_gbp: null,
    provisional_closed: 1,
    opportunities: 12,
    eligible_trades: 3,
    skip_reasons: { SKIP_BUDGET_TOO_SMALL: 5, SKIP_CAPACITY_FULL: 1 },
    fills: 3,
    win_rate: null,
    wins: 0,
    closed_with_complete_costs: 0,
  },
  markets: markets.map((market, i) => ({
    market,
    contract: `${market} · fixture contract`,
    market_status: "OPEN",
    data_status: "CURRENT",
    updated_at: at,
    l1: { status: "ACTIVE", data_type: 1, last_receipt: at, last_change: at },
    l2: {
      status: [
        "COLLECTING",
        "WAITING_FOR_SLOT",
        "COLLECTING",
        "INCOMPLETE",
        "UNAVAILABLE",
        "DISABLED",
      ][i],
      target_pre_seconds: 120,
      pre_seconds: [120, 0, 42, 15, 0, 0][i],
      fresh: i === 0 || i === 2,
      last_receipt: at,
      asks: Array.from({ length: 5 }, (_, n) => ({
        price: 2651 + n,
        size: 10 + n,
      })),
      bids: Array.from({ length: 5 }, (_, n) => ({
        price: 2650 - n,
        size: 12 + n,
      })),
      reason: i === 4 ? "IBKR_354:NO_DEPTH_PERMISSION" : "",
    },
    entry_enabled: false,
    strategy_state: [
      "BLOCKED",
      "BLOCKED",
      "POSITION_OPEN",
      "MONITORING",
      "ORDER_PENDING",
      "BLOCKED",
    ][i],
    direction: i < 2 ? "CALL / LONG" : "PUT / SHORT DIRECTION",
    block_reason: [
      "LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED",
      "NO_REAL_0DTE_MATCH",
      "ENTRIES_PAUSED · existing position managed",
      "EXECUTION_UNARMED",
      "ORDER_STATUS_UNCERTAIN",
      "SKIP_BUDGET_TOO_SMALL",
    ][i],
    conditions: { rv15: 0.0008 + i * 0.0002 },
    next_time: "2026-09-28T15:00:00Z",
    next_market_time: "2026-09-28T21:00:00Z",
    exchange_trade_date: null,
    diagnostic: market === "GC" ? "Experimental management disabled" : "",
    details: {
      fixture: true,
      mapping: null,
      entry_timezone: "America/New_York",
      entry_clocks: "09:00–16:00 weekdays; NG 13:00 veto",
      volume: "Observation only",
      exit_anchor: "Original opportunity + 60 minutes",
    },
    chart: Array.from({ length: 90 }, (_, n) => ({
      at: new Date(Date.parse(at) - (90 - n) * 60000).toISOString(),
      close:
        [67000, 72, 2650, 3, 23000, 32][i] *
        (1 + n * 0.00001 + Math.sin(n * 0.25 + i) * 0.0008),
    })),
    signals: [
      {
        id: `${market}-signal`,
        signal_at: "2026-09-28T14:00:00Z",
        decision: "SIGNAL_OBSERVED",
        reason: "",
      },
    ],
    trades:
      market === "GC"
        ? [
            {
              id: "gc-open",
              state: "POSITION_OPEN",
              quantity: 1,
              entry_at: "2026-09-28T14:00:04Z",
              exit_at: "2026-09-28T15:00:00Z",
              valuation: {
                value_gbp: -0.8,
                fresh: true,
                quote_at: at,
                method: "CONSERVATIVE_BID_ESTIMATE",
              },
            },
          ]
        : market === "NQ"
          ? [
              {
                id: "nq-pending",
                state: "EXPOSURE_REQUIRES_RECONCILIATION",
                quantity: 0,
                exit_at: "2026-09-28T15:00:00Z",
                valuation: { value_gbp: null, fresh: false },
              },
            ]
          : [],
  })),
};
const rows = Array.from({ length: 80 }, (_, i) => ({
  id: `fixture-${i}`,
  market: markets[i % 6],
  signal_at: new Date(Date.parse(at) - i * 60000).toISOString(),
  rule_version: "CLOCK60_NG13_20260927",
  decision: i % 3 ? "SKIPPED" : "BROKER_PAPER_FILL",
  reason: i % 3 ? "SKIP_BUDGET_TOO_SMALL" : "",
  state: null,
}));
const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  let data;
  if (url.pathname === "/api/overview") data = state;
  else if (url.pathname === "/api/history")
    data = {
      rows: rows.filter(
        (r) =>
          !url.searchParams.get("market") ||
          r.market === url.searchParams.get("market"),
      ),
      has_more: false,
    };
  else if (url.pathname === "/api/system")
    data = {
      fixture: true,
      ...state.system,
      live_available: false,
      mappings: {},
    };
  else if (url.pathname === "/api/detail")
    data = {
      fixture: true,
      orders: [],
      fills: [],
      note: "Offline browser fixture",
    };
  else if (url.pathname.startsWith("/api/entries/")) {
    state.system.paused = url.pathname.endsWith("pause");
    data = { paused: state.system.paused };
  }
  if (data) {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(data));
    return;
  }
  const file = url.pathname.startsWith("/static/")
    ? path.join(staticRoot, path.basename(url.pathname))
    : path.join(staticRoot, "index.html");
  res.setHeader(
    "Content-Type",
    file.endsWith(".css")
      ? "text/css"
      : file.endsWith(".js")
        ? "text/javascript"
        : "text/html",
  );
  res.end(fs.readFileSync(file));
});
async function label(page) {
  await page.evaluate(() => {
    const el = document.createElement("p");
    el.id = "fixture-label";
    el.textContent =
      "OFFLINE TEST FIXTURE · illustrative prices and states · no broker orders";
    el.style.cssText =
      "margin:0;padding:9px 3.2vw;background:#efc184;color:#10191e;font:12px monospace";
    document.body.prepend(el);
  });
}
(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const browser = await chromium.launch({ headless: true });
  try {
    fs.mkdirSync(output, { recursive: true });
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1080 },
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.clock.install({ time: new Date(at) });
    const base = `http://127.0.0.1:${server.address().port}`;
    await page.goto(base);
    await page.waitForFunction(
      () =>
        document.querySelector("#account").textContent === "OFFLINE FIXTURE",
    );
    assert.deepEqual(
      await page
        .locator(".market")
        .evaluateAll((nodes) => nodes.map((n) => n.id)),
      markets.map((m) => `card-${m}`),
    );
    assert.equal(
      await page.locator("#diagnostic-GC").textContent(),
      "Experimental management disabled",
    );
    assert.match(await page.locator("#pnl-note").textContent(), /provisional/);
    assert.equal(await page.locator("#entries").textContent(), "Unarmed");
    await label(page);
    await page.screenshot({
      path: path.join(output, "overview-desktop-fixture.png"),
      fullPage: true,
    });
    await page.locator("#card-GC summary").click();
    await page.locator("#card-GC summary").focus();
    assert.match(await page.locator("#l2-GC").textContent(), /42 \/ 120s/);
    assert.equal(await page.locator("#depth-GC-0").isVisible(), true);
    await page.screenshot({
      path: path.join(output, "depth-expanded-fixture.png"),
      fullPage: true,
    });
    await page.evaluate(() => {
      window.savedCard = document.querySelector("#card-GC");
      window.savedFocus = document.activeElement;
      window.scrollTo(0, 250);
    });
    const y = await page.evaluate(() => scrollY);
    state.markets[2].chart.at(-1).close += 1;
    await page.evaluate(() => refresh());
    assert(
      await page.evaluate(
        () =>
          savedCard === document.querySelector("#card-GC") &&
          savedFocus === document.activeElement,
      ),
    );
    assert(await page.locator("#card-GC details").evaluate((n) => n.open));
    assert.equal(await page.evaluate(() => scrollY), y);
    state.markets[2].l2.fresh = false;
    await page.evaluate(() => refresh());
    assert.equal(await page.locator("#depth-GC-0").isVisible(), false);
    assert(await page.locator("#card-GC details").evaluate((n) => n.open));
    await page.goto(`${base}/trades`);
    await page.waitForFunction(
      () => document.querySelectorAll("#history tr").length === 80,
    );
    await label(page);
    await page.locator("#sort-filter").selectOption("asc");
    await page.locator("#version-filter").fill("CLOCK60_NG13_20260927");
    await page.locator("#version-filter").focus();
    await page.evaluate(() => {
      window.savedRow = document.querySelector("#history tr");
      window.savedFocus = document.activeElement;
      document.querySelector(".table-scroll").scrollTop = 240;
    });
    await page.evaluate(() => refresh());
    assert(
      await page.evaluate(
        () =>
          savedRow === document.querySelector("#history tr") &&
          savedFocus === document.activeElement,
      ),
    );
    assert.equal(await page.locator("#sort-filter").inputValue(), "asc");
    assert.equal(
      await page.locator("#version-filter").inputValue(),
      "CLOCK60_NG13_20260927",
    );
    assert.equal(
      await page.locator(".table-scroll").evaluate((n) => n.scrollTop),
      240,
    );
    await page.screenshot({
      path: path.join(output, "history-desktop-fixture.png"),
      fullPage: true,
    });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator(".table-scroll").evaluate((n) => {
      n.scrollLeft = 220;
      n.scrollTop = 140;
    });
    await page.evaluate(() => refresh());
    assert.equal(
      await page.locator(".table-scroll").evaluate((n) => n.scrollLeft),
      220,
    );
    await page.goto(base);
    await page.waitForFunction(
      () =>
        document.querySelector("#account").textContent === "OFFLINE FIXTURE",
    );
    await label(page);
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    const boxes = await page
      .locator(".market")
      .evaluateAll((nodes) => nodes.map((n) => n.getBoundingClientRect().x));
    assert(boxes.every((x) => x === boxes[0]));
    await page.screenshot({
      path: path.join(output, "overview-mobile-fixture.png"),
      fullPage: true,
    });
    await page.goto(`${base}/system`);
    await page.waitForFunction(() =>
      document
        .querySelector("#system-json")
        .textContent.includes('"live_available": false'),
    );
    await page.setViewportSize({ width: 1440, height: 1080 });
    await label(page);
    assert.match(await page.locator("#api-lines").textContent(), /24 \/ 60/);
    assert.match(
      await page.locator("#api-depth").textContent(),
      /3 \/ 3 research books/,
    );
    assert.match(await page.locator("#api-external").textContent(), /unknown/);
    await page.screenshot({
      path: path.join(output, "system-api-fixture.png"),
      fullPage: true,
    });
    assert.deepEqual(errors, []);
    console.log(
      "PASS: six fixed cards, signal/fill markers, provisional P&L, refresh identity/focus/scroll/filter retention, mobile layout and System",
    );
  } finally {
    await browser.close();
    server.close();
  }
})().catch((error) => {
  console.error(error);
  server.close();
  process.exitCode = 1;
});
