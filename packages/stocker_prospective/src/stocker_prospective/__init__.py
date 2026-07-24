"""Prospective Stocker recorder and read-only web runtime.

This package has no order-submission boundary.  It records evidence only.
"""

from stocker_prospective.bundle import BUNDLE_MANIFEST_VERSION

__all__ = ["BUNDLE_MANIFEST_VERSION"]
