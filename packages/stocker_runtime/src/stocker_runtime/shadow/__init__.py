"""Shadow-only virtual valuation; deliberately independent of broker and execution code."""

from stocker_runtime.shadow.engine import ShadowEngine, ShadowPolicy

__all__ = ["ShadowEngine", "ShadowPolicy"]
