const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
const root =
  process.env.STOCKER_DASHBOARD_ASSETS ||
  path.join(
    __dirname,
    "../packages/stocker_dashboard/src/stocker_dashboard/static",
  );
const q5 = 4.459368321659181;
const session = "2026-09-24";
const system = {
  connected: true,
  reconciled: true,
  armed: true,
  paused: false,
  session,
  ledger_available: true,
  worker_health: "RUNNING",
  manager_health: "RUNNING",
  problem: "",
  missing_settings: [],
  settings: { armed: false },
  last_scanner_observation: "2026-09-24T14:00:00Z",
  option_quote_state: "STALE",
  outstanding_obligations: 0,
};
const flowFixture = {
  state: "COLLECTING_ESTIMATED_FLOW", feed_mode: "TBT_TRADES_TBT_QUOTES",
  first_received_at: "2026-09-24T14:01:02Z", quote_age_ms: 230, trade_age_ms: 410,
  pre_capture_gap_seconds: 2, dropped_events: 0, observation_state: "OBSERVED_PRINTS",
  coverage_warning: "Partial capture: quote pacing gap; one interrupted segment.",
  totals: {classified_volume_fraction: 0.8},
  rolling_5m: {buy_est_volume: 7200, sell_est_volume: 4800, unknown_volume: 3000,
    excluded_volume: 100, eligible_observed_volume: 15000, volume_delta: 2400,
    classified_volume_fraction: 0.8, partial_coverage: true},
  last_completed_minute: {volume_delta: -420},
  captures: [{gaps:[{at:"2026-09-24T14:04:00Z", reason:"DISCONNECT"}]}],
  bars: Array.from({length:30}, (_,i) => ({minute:new Date(Date.parse("2026-09-24T14:01:00Z") + i*60000).toISOString(),
    capture_id: i < 4 ? "fixture-a" : "fixture-b", price:10 + Math.sin(i/3)*0.4,
    volume_delta:Math.round(Math.sin(i/2)*800), unknown_volume:120+(i%3)*80,
    cumulative_volume_delta:Math.round(Math.sin(i/7)*1800), classified_volume_fraction:0.8,
    has_gap:i===4})),
};
const allocation = (slot, state) => ({
  slot,
  order_flow: flowFixture,
  symbol: ["ALFA", "BRAV", "CHAR", "DELT"][slot - 1],
  session,
  state,
  prior15: 5.27,
  rank: slot,
  information_at: "2026-09-24T14:00:00Z",
  entry_at: "2026-09-24T14:01:00Z",
  actual_entry_at: state === "SELECTED" ? null : "2026-09-24T14:01:02Z",
  intended_exit_at: "2026-09-24T19:59:00Z",
  actual_exit_at: state === "CLOSED" ? "2026-09-24T19:59:04Z" : null,
  has_exposure: ["OPEN", "PARTIALLY FILLED"].includes(state),
  closed: state === "CLOSED",
  fees_complete: true,
  fees: 1.3,
  pending_fees: 0,
  actual_premium_paid: state === "SELECTED" ? null : 180,
  realised: state === "CLOSED" ? 42.7 : null,
  return_pct: state === "CLOSED" ? 23.72 : null,
  legs: [{ con_id: 101, remaining: state === "CLOSED" ? 0 : 1 }],
  next_step:
    state === "CLOSED"
      ? "Completion verified"
      : state === "SELECTED"
        ? "No filled exposure"
        : "Unrealised P&L unavailable",
});
let slots = [
  allocation(1, "SELECTED"),
  allocation(2, "PARTIALLY FILLED"),
  allocation(3, "OPEN"),
  allocation(4, "CLOSED"),
];
const snapshots = [
  {
    symbol: "NEAR",
    prior15: 4.21,
    rank: 7,
    decision: "NOT_Q5",
    outcome: "REJECTED",
  },
  {
    symbol: "EQUAL",
    prior15: q5,
    rank: 8,
    decision: "NOT_Q5",
    outcome: "REJECTED",
  },
  {
    symbol: "MISSING",
    prior15: null,
    rank: 9,
    decision: "PRIOR15_UNAVAILABLE",
    outcome: "REJECTED",
  },
  {
    symbol: "NEGATIVE",
    prior15: -1,
    rank: 10,
    decision: "NOT_Q5",
    outcome: "REJECTED",
  },
  {
    symbol: "<img src=x onerror=alert(1)>",
    prior15: 5,
    rank: 11,
    decision: "DAILY_CAP",
    outcome: "REJECTED",
  },
].map((r) => ({ ...r, session, information_at: "2026-09-24T14:00:00Z" }));
const rows = Array.from({ length: 50 }, (_, i) => ({
  ...snapshots[0],
  symbol: `SYN${String(i).padStart(3, "0")}`,
  slot: i === 0 ? 1 : null,
}));
const pnl = {
  realised: 42.7,
  net_cash_flow: -317.3,
  actual_fees_usd: 3.9,
  exposed_allocations: 2,
  allocations_used: 4,
  reserved_fee_allowance_usd: 40,
  session_allocation_usd: 1040,
};
const overview = () => ({
  system,
  session,
  q5,
  slots,
  candidates: snapshots,
  pnl,
});
const detail = () => ({
  order_flow: flowFixture,
  event: { symbol: "SYN000", detail: "<svg onload=alert(1)>" },
  orders: [
    {
      reference: "F4:synthetic:1:ENTRY",
      role: "ENTRY",
      order_id: 123,
      status: "Filled",
      payload: { quotes: [{ ask: 1 }] },
    },
  ],
  fills: [
    {
      exec_id: "SYNTHETIC-1",
      con_id: 101,
      quantity: 1,
      price: 1,
      side: "BOT",
      multiplier: 100,
      commission: null,
      time: "2026-09-24T14:01:02Z",
    },
  ],
  has_more: false,
});
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const calls = {};
  await page.clock.install({ time: new Date("2026-09-24T14:10:00Z") });
  await page.route("http://slrno.test/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/api/")) {
      calls[url.pathname] = (calls[url.pathname] || 0) + 1;
      const responses = {
        "/api/overview": overview(),
        "/api/opportunities": { system, session, q5, rows, has_more: true },
        "/api/execution": { system, session, allocations: slots, pnl },
        "/api/detail": detail(),
        "/api/system": system,
        "/api/status": system,
      };
      if (url.pathname === "/api/first4/pause") {
        system.paused = true;
        system.armed = false;
        responses[url.pathname] = system;
      }
      return route.fulfill({ json: responses[url.pathname] });
    }
    const file = url.pathname.startsWith("/static/")
      ? url.pathname.slice(8)
      : "index.html";
    return route.fulfill({
      body: fs.readFileSync(path.join(root, file)),
      contentType: file.endsWith(".js")
        ? "application/javascript"
        : file.endsWith(".css")
          ? "text/css"
          : "text/html",
    });
  });
  async function ready(url) {
    await page.goto(`http://slrno.test${url}`);
    await page.waitForFunction(() => lastGood !== undefined);
  }
  await ready("/");
  assert.equal(await page.locator(".slot-card").count(), 4);
  assert.equal(
    await page.locator('#sidebar a[aria-current="page"]').textContent(),
    "Overview",
  );
  assert.equal(await page.locator("#main img, #main svg:not([data-chart])").count(), 0);
  assert.match(
    await page.locator("#snapshots").textContent(),
    /Equal to Q5 — does not pass/,
  );
  assert.match(
    await page.locator("#snapshots").textContent(),
    /not re-evaluated/,
  );
  assert.equal(await page.locator("#snapshots progress").count(), 0);
  assert.match(await page.locator("#snapshots").textContent(), /Rejected for this session — not reconsidered/);
  assert.match(
    await page.locator('[data-slot="3"]').textContent(),
    /Unrealised P&L unavailable/i,
  );
  assert.match(
    await page.locator('[data-slot="4"]').textContent(),
    /Realised result/,
  );
  // Lifecycle changes preserve card identities and allocation bindings.
  await page.evaluate(
    () => (window.cardNodes = [...document.querySelectorAll(".slot-card")]),
  );
  slots[0].state = "ENTRY PENDING";
  slots[0].next_step = "Awaiting fill";
  await page.clock.fastForward(5001);
  await page.waitForFunction(
    () =>
      document.querySelector('[data-slot="1"] .state').textContent ===
      "ENTRY PENDING",
  );
  assert.equal(
    await page.evaluate(() =>
      cardNodes.every(
        (node, i) => node === document.querySelectorAll(".slot-card")[i],
      ),
    ),
    true,
  );
  for (const state of ["UNFILLED", "BLOCKED / FAILED", "CLOSED"]) {
    slots[0].state = state;
    await page.evaluate(() => refresh());
    assert.equal(
      await page.locator('[data-slot="1"] h3').textContent(),
      "ALFA",
    );
    assert.equal(
      await page.locator('[data-slot="1"] .state').textContent(),
      state,
    );
  }
  slots[0] = allocation(1, "SELECTED");
  await page.evaluate(() => refresh());
  const screenshotDir = process.env.STOCKER_SCREENSHOT_DIR || path.join(__dirname, "../docs/slrno-screenshots");
  fs.mkdirSync(screenshotDir, { recursive: true });
  async function screenshot(name) {
    await page.evaluate(() => {
      const p = document.createElement("p");
      p.id = "fixture-label";
      p.textContent = "SYNTHETIC FIXTURE — NOT A BROKER SESSION";
      p.style =
        "margin:0;padding:8px;text-align:center;background:#fff0d4;color:#683b0a";
      document.body.prepend(p);
    });
    await page.screenshot({
      path: path.join(screenshotDir, name),
      fullPage: true,
    });
    await page.locator("#fixture-label").evaluate((el) => el.remove());
  }
  await screenshot("overview-desktop.png");
  await page.locator('.slot-card [class="secondary allocation-detail"]').first().click();
  await page.waitForFunction(() => !document.querySelector("#reload-detail").disabled);
  assert.equal(await page.locator(".flow-charts svg").count(), 3);
  await page.locator("#flow-zoom").fill("3");
  await page.locator("#flow-zoom").dispatchEvent("input");
  await page.evaluate(() => {
    document.querySelector(".flow-chart-scroll").scrollLeft = 240;
    document.querySelector(".slot-card details").open = true;
    window.savedFlowContainer = document.querySelector(".flow-chart-scroll");
    window.savedFlowSelection = selectedDetail;
    render(lastData);
  });
  await page.locator("#reload-detail").click();
  await page.waitForFunction(() => !document.querySelector("#reload-detail").disabled);
  assert.equal(await page.locator("#flow-zoom").inputValue(), "3");
  assert.equal(await page.evaluate(() => document.querySelector(".flow-chart-scroll").scrollLeft), 240);
  assert.equal(await page.evaluate(() => savedFlowContainer === document.querySelector(".flow-chart-scroll") && savedFlowSelection === selectedDetail && document.querySelector(".slot-card details").open), true);
  assert.match(await page.locator("#flow-detail").textContent(), /Classified coverage 80.0%/);
  await page.locator("#flow-zoom").fill("1");
  await page.locator("#flow-zoom").dispatchEvent("input");
  await screenshot("order-flow-fixture.png");
  await page.locator("#close-detail").click();

  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
    true,
    "no horizontal dashboard overflow",
  );
  for (const card of await page.locator(".slot-card").all())
    assert.ok((await card.boundingBox()).width > 300);
  const menu = page.getByRole("button", { name: "Toggle navigation" });
  await menu.click();
  assert.equal(await menu.getAttribute("aria-expanded"), "true");
  await page.keyboard.press("Escape");
  assert.equal(await menu.getAttribute("aria-expanded"), "false");
  assert.equal(
    await menu.evaluate((el) => el === document.activeElement),
    true,
  );
  await screenshot("overview-mobile.png");
  await menu.click();
  await page.locator('#sidebar a[href="/opportunities"]').click();
  await page.waitForFunction(() => lastGood !== undefined);
  assert.equal(await menu.getAttribute("aria-expanded"), "false");
  await page.setViewportSize({ width: 960, height: 800 });
  await page.locator("#decisions tbody button").first().click();
  await page.waitForFunction(() =>
    document.querySelector("#detail-json").textContent.includes("SYNTHETIC"),
  );
  await page.locator("#evidence details").evaluate((el) => (el.open = true));
  await page.locator('select[name="decision"]').selectOption("rejected");
  await page.locator('input[name="session"]').focus();
  const before = await page.evaluate(() => {
    const wrap = document.querySelector("#decisions").parentElement;
    wrap.scrollLeft = 260;
    window.scrollTo(0, 650);
    window.savedWrap = wrap;
    window.savedRow = document.querySelector("#decisions tbody tr");
    window.mutations = [];
    window.observer = new MutationObserver((records) =>
      mutations.push(...records),
    );
    observer.observe(document.querySelector("#main"), {
      subtree: true,
      childList: true,
      characterData: true,
    });
    return { left: wrap.scrollLeft, top: scrollY };
  });
  assert.ok(
    before.left >= 200,
    "deliberately wide table has substantial scroll",
  );
  for (let i = 0; i < 3; i++) {
    await page.clock.fastForward(15001);
    await page.waitForFunction(() => !refreshing);
  }
  let after = await page.evaluate(() => ({
    left: savedWrap.scrollLeft,
    top: scrollY,
    stable: savedWrap === document.querySelector("#decisions").parentElement,
    focus: document.activeElement.name,
    open: document.querySelector("#evidence details").open,
    filter: document.querySelector("select").value,
    selected: selectedDetail,
    mutations: mutations.length,
  }));
  assert.equal(after.left, before.left);
  assert.equal(after.top, before.top);
  assert.ok(after.stable);
  assert.equal(after.focus, "session");
  assert.equal(after.filter, "rejected");
  assert.ok(after.open);
  assert.equal(after.selected, `${session}/SYN000`);
  assert.equal(
    after.mutations,
    0,
    "unchanged polls do not mutate main content",
  );
  await page.evaluate(() => {
    const range = document.createRange();
    range.selectNodeContents(savedRow.children[0]);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
  });
  const selectedText = await page.evaluate(() => getSelection().toString());
  await page.clock.fastForward(15001);
  await page.waitForFunction(() => !refreshing);
  assert.equal(await page.evaluate(() => getSelection().toString()), selectedText);
  rows[0].outcome = "UPDATED_DIAGNOSTIC";
  await page.clock.fastForward(15001);
  await page.waitForFunction(() =>
    document
      .querySelector("#decisions tbody tr")
      .textContent.includes("UPDATED"),
  );
  assert.equal(await page.evaluate(() => savedWrap.scrollLeft), before.left);
  assert.equal(
    await page.evaluate(
      () => savedRow === document.querySelector("#decisions tbody tr"),
    ),
    true,
  );
  assert.equal(
    await page.evaluate(() => document.activeElement.name),
    "session",
  );
  assert.equal(await page.locator("#evidence svg:not([data-chart])").count(), 0);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await screenshot("opportunities-desktop.png");
  // Controlled transport: hung/slow reads, mutation races and timeout recovery.
  await ready("/");
  await page.evaluate(() => {
    clearTimeout(timer);
    window.transport = {
      calls: 0,
      active: 0,
      max: 0,
      mode: "slow",
      postCount: 0,
      postMode: "ok",
      statusFails: false,
    };
    window.savedData = structuredClone(lastData);
    window.fetch = (url, options) => {
      if (url === "/api/first4/pause") {
        transport.postCount++;
        return new Promise((resolve, reject) => {
          window.releasePause = () =>
            transport.postMode === "ok"
              ? resolve(
                  new Response(
                    JSON.stringify({
                      ...savedData.system,
                      paused: true,
                      armed: false,
                    }),
                  ),
                )
              : resolve(new Response("{}", { status: 503 }));
          options.signal.addEventListener("abort", () =>
            reject(new DOMException("Timeout", "AbortError")),
          );
        });
      }
      if (url.startsWith("/api/status"))
        return Promise.resolve(
          new Response(JSON.stringify({ ...savedData.system, paused: false }), {
            status: transport.statusFails ? 503 : 200,
          }),
        );
      transport.calls++;
      transport.active++;
      transport.max = Math.max(transport.max, transport.active);
      return new Promise((resolve, reject) => {
        let done = false;
        const finish = (fn, value) => {
          if (done) return;
          done = true;
          transport.active--;
          fn(value);
        };
        window.releaseRead = () =>
          finish(resolve, new Response(JSON.stringify(savedData)));
        if (transport.mode === "ok") releaseRead();
        if (transport.mode === "error")
          finish(resolve, new Response("{}", { status: 503 }));
        options.signal.addEventListener("abort", () =>
          finish(reject, new DOMException("Timeout", "AbortError")),
        );
      });
    };
    void refresh();
    void refresh();
    void refresh();
  });
  assert.equal(await page.evaluate(() => transport.calls), 1);
  await page.clock.fastForward(7000);
  assert.equal(
    await page.evaluate(() => transport.calls),
    1,
    "slow reads do not overlap",
  );
  await page.evaluate(() => releaseRead());
  await page.waitForFunction(() => !refreshing);
  assert.equal(await page.evaluate(() => transport.max), 1);
  await page.evaluate(() => {
    void refresh();
  });
  await page.clock.fastForward(8001);
  await page.waitForFunction(() => !refreshing);
  assert.match(
    await page.locator("#transport-error").textContent(),
    /timed out/,
  );
  assert.equal(
    await page.locator(".slot-card").count(),
    4,
    "timeout retains last good cards",
  );
  await page.evaluate(() => {
    transport.mode = "ok";
    return refresh();
  });
  assert.equal(
    await page.locator("#transport-error").isHidden(),
    true,
    "recovery clears stale warning",
  );
  await page.evaluate(() => {
    transport.mode = "error";
    return refresh();
  });
  assert.match(await page.locator("#last-refresh").textContent(), /STALE/);
  await page.evaluate(() => {
    transport.mode = "ok";
    return refresh();
  });
  assert.match(await page.locator("#last-refresh").textContent(), /CURRENT/);
  const hiddenCount = await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
    return transport.calls;
  });
  await page.clock.fastForward(60000);
  assert.equal(await page.evaluate(() => transport.calls), hiddenCount);
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: false,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await page.clock.fastForward(1);
  await page.waitForFunction(() => !refreshing);
  assert.equal(
    await page.evaluate(() => transport.calls),
    hiddenCount + 1,
    "one visibility recovery refresh",
  );
  // Pause uses an intentional dialog and suppresses duplicate mutations.
  await page.locator("#pause").click();
  await page.locator("#cancel-pause").click();
  assert.equal(await page.evaluate(() => transport.postCount), 0);
  await page.locator("#pause").click();
  await page.locator("#confirm-pause").click();
  await page.evaluate(() => {
    void pauseEntries();
    void pauseEntries();
  });
  assert.equal(await page.evaluate(() => transport.postCount), 1);
  assert.equal(await page.locator("#confirm-pause").isDisabled(), true);
  await page.evaluate(() => {
    savedData.system.paused = true;
    savedData.system.armed = false;
    releasePause();
  });
  await page.waitForFunction(() => !controlling);
  assert.match(
    await page.locator("#control-result").textContent(),
    /Entries PAUSED/,
  );
  assert.equal(await page.locator("#pause").isDisabled(), true);
  // Ambiguous mutation: fail state read, keep retry disabled until recovery.
  await page.evaluate(() => {
    savedData.system.paused = false;
    transport.statusFails = true;
    return refresh();
  });
  await page.locator("#pause").click();
  await page.locator("#confirm-pause").click();
  await page.evaluate(() => {
    transport.mode = "error";
  });
  await page.clock.fastForward(8001);
  await page.waitForFunction(() => !controlling);
  assert.match(
    await page.locator("#control-result").textContent(),
    /outcome uncertain/,
  );
  assert.equal(await page.locator("#pause").isDisabled(), true);
  assert.equal(
    await page.evaluate(() => transport.postCount),
    2,
    "no mutation retry on timeout",
  );
  await page.evaluate(() => {
    transport.statusFails = false;
    transport.mode = "ok";
    return refresh();
  });
  assert.equal(await page.locator("#pause").isDisabled(), false);
  assert.match(
    await page.locator("#control-result").textContent(),
    /Authoritative read: NOT PAUSED/,
  );
  // Hide while a read is pending, then immediately restore visibility.
  const visibilityBefore = await page.evaluate(() => {
    transport.mode = "slow";
    void refresh();
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: false,
    });
    transport.mode = "ok";
    document.dispatchEvent(new Event("visibilitychange"));
    return transport.calls;
  });
  await page.clock.fastForward(1);
  await page.waitForFunction(() => !refreshing);
  assert.equal(
    await page.evaluate(() => transport.calls),
    visibilityBefore + 1,
  );
  // A response already completing when pause is issued must not restore ARMED.
  await page.evaluate(() => {
    clearTimeout(timer);
    window.oldState = structuredClone(savedData);
    oldState.system.armed = true;
    oldState.system.paused = false;
    window.fetch = (url) =>
      url.startsWith("/api/overview")
        ? new Promise((resolve) => {
            window.obsoleteResponse = () =>
              resolve(new Response(JSON.stringify(oldState)));
          })
        : Promise.resolve(
            new Response(
              JSON.stringify({
                ...savedData.system,
                paused: true,
                armed: false,
              }),
            ),
          );
    void refresh();
  });
  await page.locator("#pause").click();
  await page.locator("#confirm-pause").click();
  await page.waitForFunction(() => !controlling);
  await page.evaluate(() => obsoleteResponse());
  await page.waitForFunction(() => !refreshing);
  assert.equal(await page.locator("#system-status").textContent(), "PAUSED");
  // Historical pages fetch evidence once; polling only refreshes operational status.
  const oldCalls = calls["/api/opportunities"];
  await ready("/opportunities?session=2026-09-23&decision=rejected");
  await page.clock.fastForward(30001);
  await page.waitForFunction(() => !refreshing);
  assert.equal(calls["/api/opportunities"], oldCalls + 1);
  assert.ok(calls["/api/status"] > 0);
  await ready("/execution");
  await screenshot("execution-desktop.png");
  await ready("/system");
  await page.evaluate(() => {
    window.savedData = structuredClone(lastData);
    savedData.armed = false;
    savedData.opening_check_active = false;
    savedData.opening_check = {status: "ARMED", attempt: 5};
    render(savedData);
  });
  assert.match(await page.locator("#opening").textContent(), /Current readiness: Unarmed/);
  assert.match(await page.locator("#opening").textContent(), /Historical opening result: ARMED · 5 attempts/);
  await page.evaluate(() => {
    savedData.opening_check_active = true;
    savedData.opening_remaining_seconds = 740;
    savedData.opening_check = {status: "CHECKING", attempt: 4};
    render(savedData);
  });
  assert.match(await page.locator("#opening").textContent(), /Verification active · 4 attempts · 740s remaining/);
  await screenshot("system-desktop.png");
  const requestsPerMinute = {};
  for (const [url, endpoint, period] of [
    ["/", "/api/overview", 5000],
    ["/opportunities", "/api/opportunities", 15000],
    ["/execution", "/api/execution", 5000],
    ["/system", "/api/system", 30000],
  ]) {
    await ready(url);
    const start = calls[endpoint];
    for (let elapsed = 0; elapsed < 60000; elapsed += period) {
      await page.clock.fastForward(period + 1);
      await page.waitForFunction(() => !refreshing);
    }
    requestsPerMinute[url] = calls[endpoint] - start;
  }
  assert.deepEqual(requestsPerMinute, {
    "/": 12,
    "/opportunities": 4,
    "/execution": 12,
    "/system": 2,
  });
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify(
      {
        passed: "SLRNO browser acceptance",
        maximum_concurrent_refreshes: 1,
        hidden_60s_requests: 0,
        unchanged_main_child_text_mutations: after.mutations,
        scroll_left_before: before.left,
        scroll_left_after: after.left,
        visible_schedule_seconds: {
          overview: 5,
          opportunities: 15,
          execution: 5,
          system: 30,
        },
        requestsPerMinute,
        calls,
      },
      null,
      2,
    ),
  );
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
