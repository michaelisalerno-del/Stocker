import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL(
    "../../packages/stocker_runtime/src/stocker_runtime/web/static/app.js",
    import.meta.url,
  ),
  "utf8",
);
const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const { ENDPOINTS, PollCoordinator, genericEntries } = await import(moduleUrl);
const turn = () => new Promise((resolve) => setImmediate(resolve));

test("unknown plugin payloads flatten through the generic renderer", () => {
  const entries = genericEntries({
    novel_metric: { confidence_band: [0.2, 0.7] },
    opaque_state: "watch",
  });

  assert.deepEqual(entries, [
    { label: "novel_metric.confidence_band.0", value: "0.2" },
    { label: "novel_metric.confidence_band.1", value: "0.7" },
    { label: "opaque_state", value: "watch" },
  ]);
  assert.equal(genericEntries({ a: 1, b: 2, c: 3 }, "payload", 2).length, 2);
});

test("the primary polling plan uses only generic V2 reads", () => {
  assert.deepEqual(Object.keys(ENDPOINTS), [
    "meta",
    "live",
    "ideas",
    "results",
    "diagnostics",
  ]);
  assert.ok(Object.values(ENDPOINTS).every((path) => path.startsWith("/api/v2/")));
});

test("overlapping polls are serialized and one refresh is queued", async () => {
  let active = 0;
  let maximumActive = 0;
  let runs = 0;
  const completions = [];
  const coordinator = new PollCoordinator(
    {
      live: (signal) => {
        runs += 1;
        active += 1;
        maximumActive = Math.max(maximumActive, active);
        return new Promise((resolve) => {
          let finished = false;
          const finish = () => {
            if (finished) return;
            finished = true;
            active -= 1;
            resolve();
          };
          signal.addEventListener("abort", finish, { once: true });
          completions.push(finish);
        });
      },
    },
    { intervalMs: 60_000 },
  );

  const first = coordinator.activate("live");
  await turn();
  assert.equal(coordinator.refresh(), first);
  completions[0]();
  await first;
  await turn();

  assert.equal(runs, 2);
  assert.equal(maximumActive, 1);
  completions[1]();
  await turn();
  coordinator.stop();
});

test("hidden pages pause work and visibility resumes the active view", async () => {
  let runs = 0;
  const coordinator = new PollCoordinator(
    { live: async () => { runs += 1; } },
    { intervalMs: 60_000 },
  );

  coordinator.setVisible(false);
  await coordinator.activate("live");
  assert.equal(runs, 0);
  coordinator.setVisible(true);
  await turn();
  assert.equal(runs, 1);
  coordinator.stop();
});
