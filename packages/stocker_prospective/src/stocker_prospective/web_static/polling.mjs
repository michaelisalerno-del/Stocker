export class DashboardPollCoordinator {
  constructor({ run, isVisible, makeAbortController = () => new AbortController() }) {
    this.run = run;
    this.isVisible = isVisible;
    this.makeAbortController = makeAbortController;
    this.active = null;
    this.controller = null;
    this.queued = false;
    this.queuedDetails = false;
  }

  refresh({ refreshDetails = false } = {}) {
    if (!this.isVisible()) {
      return Promise.resolve({ skipped: "hidden" });
    }
    if (this.active) {
      this.queued = true;
      this.queuedDetails ||= refreshDetails;
      this.controller.abort();
      return this.active;
    }

    const controller = this.makeAbortController();
    this.controller = controller;
    const active = Promise.resolve()
      .then(() => this.run({ refreshDetails, signal: controller.signal }))
      .finally(() => {
        if (this.active !== active) return;
        this.active = null;
        this.controller = null;
        if (this.queued && this.isVisible()) {
          const queuedDetails = this.queuedDetails;
          this.queued = false;
          this.queuedDetails = false;
          this.refresh({ refreshDetails: queuedDetails });
        }
      });
    this.active = active;
    return active;
  }

  hide() {
    this.queued = false;
    this.queuedDetails = false;
    if (this.controller) this.controller.abort();
  }

  show() {
    return this.refresh();
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
