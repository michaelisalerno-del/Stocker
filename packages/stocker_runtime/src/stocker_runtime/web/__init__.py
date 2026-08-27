"""Bounded read-only web interfaces for Stocker V2."""

from stocker_runtime.web.app import create_web_app
from stocker_runtime.web.config import WebConfig

__all__ = ["WebConfig", "create_web_app"]
