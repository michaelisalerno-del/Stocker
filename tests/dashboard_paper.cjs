/* The three pages against fixture JSON dumped from the offline paper day (tests/fixtures). No Saxo calls. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const {chromium} = require("playwright");
const root = path.resolve("packages/stocker_dashboard/src/stocker_dashboard/static");
const fixture = JSON.parse(fs.readFileSync(path.resolve("tests/fixtures/paper_dashboard.json"), "utf8"));
const calls = [];
const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (url.pathname.startsWith("/api/")) {
    calls.push(url.pathname);
    res.setHeader("Content-Type", "application/json");
    return res.end(JSON.stringify(fixture[url.pathname] || {}));
  }
  const file = url.pathname.startsWith("/static/") ? path.basename(url.pathname) : "index.html";
  res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : file.endsWith(".svg") ? "image/svg+xml" : "text/html");
  res.end(fs.readFileSync(path.join(root, file)));
});
(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const browser = await chromium.launch({headless: true});
  try {
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const viewport of [{width: 1280, height: 900}, {width: 390, height: 844}]) {
      const page = await browser.newPage({viewport, timezoneId: "Europe/London"});
      const errors = [];
      page.on("pageerror", (e) => errors.push(e.message));

      await page.goto(base + "/");
      await page.waitForFunction(() => document.querySelectorAll("#cfd tr").length === 2);
      assert.equal(await page.locator("nav a[aria-current]").textContent(), "Today");
      assert.match(await page.locator("#sig-NAS100").textContent(), /^Long · candle 30018\.00 vs EMA12 30007\.00 · prior close below the 50-day$/);
      assert.match(await page.locator("#sig-US500").textContent(), /^Short/);
      assert.equal(await page.locator("#plan tr").count(), 10);
      assert.match(await page.locator("#plan tr").first().textContent(), /First candle · CFD trades.*09:35.*14:35.*Done/);
      assert.equal(await page.locator("#plan tr.next").count(), 1);
      assert.equal(await page.locator("#plan-progress").textContent(), "3 of 10 done");
      assert.match(await page.locator("#cfd tr").nth(1).textContent(), /Stop 09:40 NY.*-\$204\.00/);
      assert.equal(await page.locator("#cfd-sum").textContent(), "2 trades · closed -$204.00 · open +$79.20");
      assert.match(await page.locator("#options tr").first().textContent(), /Call 701.*\+\$59\.92/);
      assert.match(await page.locator("#crosses tr").first().textContent(), /10:30 NY · QQQ 701\.00 vs VWAP 700\.08.*Back below vwap.*\+\$65\.34/);
      assert.match(await page.locator("#crosses tr").nth(1).textContent(), /Missed entry · Qqq chart delayed/);
      assert.match(await page.locator("#sig-DIP").textContent(), /^Connors: buy · Turnaround Tuesday: buy · QQQ 710\.00 \(-1\.11% on the day\) · Fear rule: no reading$/);
      assert.equal(await page.locator("#play-NAS100").textContent(), "QQQ call · pullback cell: long below the 50-day · disagrees with the other index");
      assert.equal(await page.locator("#play-NAS100").getAttribute("data-tone"), "go");
      assert.equal(await page.locator("#play-US500").getAttribute("data-tone"), "skip");
      await page.waitForFunction(() => document.querySelector("#chart svg") && document.querySelector("#chart-US500 svg"));
      assert.match(await page.locator("#chart-price").textContent(), /^7\d\d\.\d\d [-+]\d+ bp since 09:30$/);
      assert.match(await page.locator("#chart-stat").textContent(), /^last candle 15:55 NY · 15 min late · 78 candles$/);
      assert.equal(await page.locator("#chart path.c-vwap").count(), 1);
      assert.equal(await page.locator("#chart line.c-strike").count(), 1);
      assert.equal(await page.locator("#chart rect.c-hit").count(), 1);
      assert.match(await page.locator("#chart-price-US500").textContent(), /^7\d{3} [-+]\d+ bp since 09:30$/);
      assert.match(await page.locator("#chart-stat-US500").textContent(), /^last candle 16:00 NY · 145 candles$/);
      assert.equal(await page.locator("#chart-US500 line.c-entry").count(), 1);
      assert.match(await page.locator("#dips tr").first().textContent(), /Connors · in the book.*Rsi2 above 65 2026-10-19.*\+\$197\.34/);
      assert.match(await page.locator("#dips tr").nth(1).textContent(), /Turnaround Tuesday.*Open · sells on its rule/);
      assert.equal(await page.locator("#stock-orders tr").count(), 5);
      assert.match(await page.locator("#stock-orders tr").nth(1).textContent(), /SMCI.*Filled 2026-10-19 at 35\.42/);
      assert.match(await page.locator("#stock-orders tr").nth(3).textContent(), /COIN.*Resting today · signal 2026-10-16199\.50low 196\.10 · needs a 3\.2% fall · 13:41 NY$/);
      assert.match(await page.locator("#stock-orders tr").nth(4).textContent(), /AFRM.*low 47\.10, under the limit: fills at 47\.84 · 13:41 NY$/);
      assert.equal(await page.locator("#stock-trades tr").count(), 2);
      assert.match(await page.locator("#stock-trades tr").first().textContent(), /SMCI · tier A.*36\.1013:41 NYOpen · ConnorsRSI 41\.3 now, sells at a close above 70\+\$34\.40 open$/);
      assert.equal(await page.locator("#stocks-sum").textContent(), "3 resting orders · 1 fill today · 1 open position · closed today +$122.50 · open +$34.40");
      assert.equal(await page.locator("#stocks-scan").textContent(), "Last scan Fri 16 Oct · 2,514 charts read · 1,500 in the universe · 4 setups · 0 filled · 0 sold");
      assert.equal(await page.locator("#status").getAttribute("data-tone"), "ok");
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth), 0);

      await page.goto(base + "/record");
      await page.waitForFunction(() => document.querySelectorAll("#summary tr").length === 3);
      assert.equal(await page.locator("nav a[aria-current]").textContent(), "Record");
      assert.equal(await page.locator("#trades tr").count(), 4);
      assert.equal(await page.locator("#rec-total").textContent(), "-$102.58");
      assert.equal(await page.locator("#equity svg").count(), 1);
      assert.match(await page.locator("#equity-stat").textContent(), /^2 days with closed trades · best \+\$41\.50 \(Fri 16 Oct\) · worst -\$144\.08 \(Mon 19 Oct\) · max drawdown -\$144\.08$/);
      assert.equal(await page.locator("#verdicts li").count(), 11);
      assert.match(await page.locator("#verdicts li").first().textContent(), /^A · CFD costs · Nasdaq0 of 40/);
      assert.equal(await page.locator("#books tr").count(), 4);
      assert.match(await page.locator("#books tr").first().textContent(), /Pullback cell.*1 · \$0\.00none closed.*1 · \+\$59\.92avg \+\$59\.92 · 1 closed/);
      assert.ok(await page.locator("#prev").isDisabled());
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth), 0);

      await page.goto(base + "/system");
      await page.waitForFunction(() => document.querySelector("#auth-state").textContent.includes("AUTHENTICATED"));
      assert.equal(await page.locator("#facts div").count(), 8);
      assert.ok(await page.locator("#problems-empty").isVisible());
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth), 0);
      assert.deepEqual(errors, []);
      await page.close();
    }
    // Each page asks only for its own data.
    assert.deepEqual([...new Set(calls)].sort(), ["/api/charts", "/api/record", "/api/system", "/api/today"]);
    console.log("PASS: Today, Record and System render the paper day at desktop and phone width");
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
