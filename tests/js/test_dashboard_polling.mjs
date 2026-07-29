import assert from "node:assert/strict";
import test from "node:test";

import {
  DashboardPollCoordinator,
  detailRequestPlan,
} from "../../packages/stocker_prospective/src/stocker_prospective/web_static/polling.mjs";

const turn = () => new Promise((resolve) => setImmediate(resolve));

test("overlapping snapshot requests abort stale work without concurrent runs", async () => {
  let active = 0;
  let maximumActive = 0;
  let runCount = 0;
  const completions = [];
  const coordinator = new DashboardPollCoordinator({
    isVisible: () => true,
    run: ({ signal }) => {
      runCount += 1;
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
  });

  const first = coordinator.refresh();
  await turn();
  const overlapping = coordinator.refresh({ refreshDetails: true });

  assert.equal(overlapping, first);
  await first;
  await turn();
  assert.equal(runCount, 2);
  assert.equal(maximumActive, 1);
  completions.at(-1)();
  await turn();
  assert.equal(coordinator.inFlight, false);
});

test("hidden pages pause polling and visibility resumes immediately", async () => {
  let visible = false;
  let runs = 0;
  const coordinator = new DashboardPollCoordinator({
    isVisible: () => visible,
    run: async () => {
      runs += 1;
    },
  });

  assert.deepEqual(await coordinator.refresh(), { skipped: "hidden" });
  assert.equal(runs, 0);
  visible = true;
  await coordinator.show();
  assert.equal(runs, 1);
  visible = false;
  coordinator.hide();
  await coordinator.refresh();
  assert.equal(runs, 1);
});

test("episode details load only on selection or explicit refresh", () => {
  assert.equal(
    detailRequestPlan({
      selectedId: null,
      availableIds: ["episode-a", "episode-b"],
    }),
    "episode-a",
  );
  assert.equal(
    detailRequestPlan({
      selectedId: "episode-a",
      availableIds: ["episode-a", "episode-b"],
    }),
    null,
  );
  assert.equal(
    detailRequestPlan({
      selectedId: "episode-a",
      availableIds: ["episode-a", "episode-b"],
      explicitRefresh: true,
    }),
    "episode-a",
  );
  assert.equal(
    detailRequestPlan({
      selectedId: "removed",
      availableIds: ["episode-a"],
      explicitRefresh: true,
    }),
    null,
  );
});
