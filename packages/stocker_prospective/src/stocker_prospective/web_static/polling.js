"use strict";

(() => {
  const POLLING_POLICY = Object.freeze({
    fastIntervalMs: 15_000,
    slowIntervalMs: 90_000,
    manualIntervalMs: 300_000,
    fastEndpoints: Object.freeze(["/api/dashboard/summary"]),
    slowEndpointsByScreen: Object.freeze({
      "live-monitor": Object.freeze([
        "/api/recorder/capabilities",
        "/api/market-data-budget",
      ]),
      "signal-detail": Object.freeze(["/api/episodes"]),
      "options-recorder": Object.freeze(["/api/episodes"]),
      "quiet-universe": Object.freeze([
        "/api/quiet-state/status",
        "/api/quiet-state/universe",
      ]),
      "quiet-episode": Object.freeze([
        "/api/quiet-state/status",
        "/api/quiet-state/episodes",
      ]),
      "quiet-shadow": Object.freeze(["/api/quiet-state/session-quality"]),
    }),
    manualEndpointsByScreen: Object.freeze({
      "live-monitor": Object.freeze([
        "/api/source-transfer",
        "/api/reports/daily",
      ]),
      "shadow-blotter": Object.freeze(["/api/shadow-outcomes"]),
      "safety-audit": Object.freeze([
        "/api/audit/events?limit=100",
        "/api/recorder/session-reports",
      ]),
      "quiet-shadow": Object.freeze(["/api/quiet-state/shadow-structures"]),
      "concentration-audit": Object.freeze([
        "/api/quiet-state/concentration-audit",
      ]),
    }),
  });

  class RequestCoordinator {
    constructor(fetchImplementation = window.fetch.bind(window)) {
      this.fetchImplementation = fetchImplementation;
      this.inFlight = new Map();
    }

    async request(path, options = {}) {
      const response = await this.fetchImplementation(path, {
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
        error.requestId = response.headers.get("X-Request-ID");
        throw error;
      }
      return response.json();
    }

    run(key, task, { supersede = false } = {}) {
      const existing = this.inFlight.get(key);
      if (existing && !supersede) return existing.promise;
      if (existing) existing.controller.abort();

      const controller = new AbortController();
      const entry = {
        controller,
        promise: null,
      };
      const promise = Promise.resolve()
        .then(() => task(controller.signal))
        .finally(() => {
          if (this.inFlight.get(key) === entry) this.inFlight.delete(key);
        });
      entry.promise = promise;
      this.inFlight.set(key, entry);
      return promise;
    }

    cancel(key) {
      const existing = this.inFlight.get(key);
      if (existing) existing.controller.abort();
    }

    cancelAll() {
      this.inFlight.forEach((entry) => entry.controller.abort());
    }
  }

  window.StockerPolling = Object.freeze({
    POLLING_POLICY,
    RequestCoordinator,
  });
})();
