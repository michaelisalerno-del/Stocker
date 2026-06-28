"""Feature engineering placeholders.

Features should be pure transformations of audited data. They should not know about
order placement, broker state, or execution-specific risk checks.
"""

from typing import Any


def identity_features(frame: Any) -> Any:
    """Return input data unchanged as a placeholder feature pipeline."""

    return frame
