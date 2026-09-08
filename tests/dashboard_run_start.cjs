// Run with Node and Playwright installed. All HTTP and broker activity is mocked.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");

(async () => {
  const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    let state = { status: "IDLE" };
    let posts = 0;
    let release;
    const pending = new Promise(resolve => { release = resolve; });
    const staticPath = path.join(__dirname, "../packages/stocker_dashboard/src/stocker_dashboard/static");
    await page.route("http://stocker.test/**", async route => {
      const url = new URL(route.request().url());
      const json = value => route.fulfill({ json: value });
      if (url.pathname === "/api/overview") return json({ environments: {}, system: "READY", active_runs: 0, open_positions: 0, as_of: "2026-09-08T12:00:00Z" });
      if (url.pathname === "/api/universe-builder/options") return json({
        markets: [{ market_id: "UK_LSE", label: "UK / LSE", experimental: true, session: "08:00–16:30", timezone: "Europe/London", search_policy: "Existing activity shortlist", validation: "Unvalidated cross-market PAPER test" }],
        strategies: [{ strategy_id: "SESSION_HARD", strategy_version: "FROZEN", label: "Session HARD", environments: ["PAPER"] }],
        candidate_screen: { label: "Method-owned discovery" },
      });
      if (url.pathname === "/api/universe-runs") return json({ PAPER: [], LIVE: [] });
      if (url.pathname === "/api/universe-runs/start-status") return json(state);
      if (url.pathname === "/api/universe-runs/paper") {
        assert.equal(url.searchParams.get("background"), "true");
        assert.equal(route.request().postDataJSON().market_id, "UK_LSE");
        posts++;
        state = { status: "STARTING", operation_id: "test", detail: "Qualifying method universe" };
        await pending;
        return route.fulfill({ status: 202, json: state });
      }
      const file = url.pathname.startsWith("/static/") ? path.basename(url.pathname) : "index.html";
      return route.fulfill({ body: await fs.readFile(path.join(staticPath, file)), contentType: file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html" });
    });
    await page.goto("http://stocker.test/universes");
    await page.getByRole("button", { name: "Start PAPER run" }).click();
    await page.getByRole("button", { name: "Starting…" }).waitFor();
    assert(await page.getByRole("button", { name: "Starting…" }).isDisabled());
    assert.match(await page.locator("#run-start-status").innerText(), /Sending start request/);
    release();
    await page.waitForFunction(() => document.querySelector("#run-start-status").textContent.includes("Qualifying"));
    await page.reload();
    await page.getByRole("button", { name: "Starting…" }).waitFor();
    assert(await page.getByRole("button", { name: "Starting…" }).isDisabled());
    assert.equal(posts, 1);
    state = { status: "FAILED", detail: "IBKR test failure" };
    await page.evaluate(() => refreshCurrentPage());
    await page.getByRole("button", { name: "Start PAPER run" }).waitFor();
    assert.match(await page.locator("#run-start-status").innerText(), /FAILED: IBKR test failure/);
    assert.equal(await page.getByRole("button", { name: "Start PAPER run" }).isDisabled(), false);
    assert(await page.getByRole("button", { name: "Add to LIVE" }).isDisabled());
    assert.deepEqual(errors, []);
    console.log("PASS: start feedback, duplicate prevention, reload, visible failure, PAPER-only");
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
