"use strict";

export const POLLING_POLICY = Object.freeze({
  fastIntervalMs: 15_000,
  slowIntervalMs: 90_000,
  manualIntervalMs: 300_000,
  fastEndpoints: Object.freeze(["/api/dashboard/summary"]),
  slowEndpointsByScreen: Object.freeze({
    "live-monitor": Object.freeze([
      "/api/recorder/status",
      "/api/recorder/capabilities",
      "/api/market-data-budget",
    ]),
    "universe-monitor": Object.freeze(["/api/universe/live"]),
    "signal-detail": Object.freeze(["/api/episodes"]),
    "options-recorder": Object.freeze(["/api/episodes"]),
    "shadow-blotter": Object.freeze([]),
    "safety-audit": Object.freeze([]),
    "quiet-universe": Object.freeze([
      "/api/quiet-state/status",
      "/api/quiet-state/universe",
    ]),
    "quiet-episode": Object.freeze([
      "/api/quiet-state/status",
      "/api/quiet-state/episodes",
    ]),
    "quiet-shadow": Object.freeze(["/api/quiet-state/session-quality"]),
    "concentration-audit": Object.freeze([]),
    "opening-leader-continuation": Object.freeze([
      "/api/opening-leader-continuation-v0",
    ]),
  }),
  manualEndpointsByScreen: Object.freeze({
    "live-monitor": Object.freeze([
      "/api/source-transfer",
      "/api/reports/daily",
    ]),
    "universe-monitor": Object.freeze([]),
    "signal-detail": Object.freeze([]),
    "options-recorder": Object.freeze([]),
    "shadow-blotter": Object.freeze([
      "/api/shadow-outcomes",
      "/api/virtual-ledgers",
    ]),
    "safety-audit": Object.freeze([
      "/api/audit/events?limit=100",
      "/api/recorder/session-reports",
    ]),
    "quiet-universe": Object.freeze([]),
    "quiet-episode": Object.freeze([]),
    "quiet-shadow": Object.freeze([
      "/api/quiet-state/shadow-structures",
      "/api/virtual-ledgers",
    ]),
    "concentration-audit": Object.freeze([
      "/api/quiet-state/concentration-audit",
    ]),
    "opening-leader-continuation": Object.freeze([]),
  }),
});

const TIER_PRIORITY = Object.freeze({
  fast: 0,
  slow: 1,
  manual: 2,
});

function higherTier(left, right) {
  return TIER_PRIORITY[left] >= TIER_PRIORITY[right] ? left : right;
}

export function endpointsForScreen(tier, screenId) {
  if (tier === "slow") {
    return POLLING_POLICY.slowEndpointsByScreen[screenId] || [];
  }
  if (tier === "manual") {
    return POLLING_POLICY.manualEndpointsByScreen[screenId] || [];
  }
  return POLLING_POLICY.fastEndpoints;
}

export class DashboardPollCoordinator {
  constructor({ run, isVisible, makeAbortController = () => new AbortController() }) {
    this.run = run;
    this.isVisible = isVisible;
    this.makeAbortController = makeAbortController;
    this.active = null;
    this.controller = null;
    this.queuedTier = null;
    this.queuedDetails = false;
  }

  refresh({
    tier = "fast",
    refreshDetails = false,
    supersede = false,
  } = {}) {
    const requestedTier = refreshDetails ? "manual" : tier;
    if (!(requestedTier in TIER_PRIORITY)) {
      return Promise.reject(new Error(`unknown polling tier: ${requestedTier}`));
    }
    if (!this.isVisible()) {
      return Promise.resolve({ skipped: "hidden" });
    }
    if (this.active) {
      this.queuedTier = this.queuedTier === null
        ? requestedTier
        : higherTier(this.queuedTier, requestedTier);
      this.queuedDetails ||= refreshDetails;
      if (supersede || refreshDetails) this.controller.abort();
      return this.active;
    }

    const controller = this.makeAbortController();
    this.controller = controller;
    const active = Promise.resolve()
      .then(() => this.run({
        tier: requestedTier,
        refreshDetails,
        signal: controller.signal,
      }))
      .finally(() => {
        if (this.active !== active) return;
        this.active = null;
        this.controller = null;
        if (this.queuedTier !== null && this.isVisible()) {
          const queuedTier = this.queuedTier;
          const queuedDetails = this.queuedDetails;
          this.queuedTier = null;
          this.queuedDetails = false;
          this.refresh({
            tier: queuedTier,
            refreshDetails: queuedDetails,
          });
        }
      });
    this.active = active;
    return active;
  }

  cancelAll() {
    this.queuedTier = null;
    this.queuedDetails = false;
    if (this.controller) this.controller.abort();
  }

  hide() {
    this.cancelAll();
  }

  show() {
    return this.refresh({ tier: "fast", supersede: true });
  }

  get inFlight() {
    return this.active !== null;
  }
}

export function detailRequestPlan({
  selectedId,
  availableIds,
  explicitRefresh = false,
}) {
  if (!selectedId) return availableIds[0] || null;
  if (!availableIds.includes(selectedId)) return null;
  return explicitRefresh ? selectedId : null;
}
