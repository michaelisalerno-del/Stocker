"""Optional official-local-install TWS API boundary.

IBKR states that its TWS API is distributed only through its official ZIP/MSI and that
package-index copies are not hosted, endorsed, or supported by IBKR. This module detects
an administrator-installed official client without substituting a wrapper or PyPI build.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class OfficialAPIStatus:
    """Installation feasibility without importing or connecting to TWS."""

    installed: bool
    module_importable: bool
    provenance_verified: bool
    module_name: str
    distribution_source: str
    installation_policy: str
    blocker: str | None


def official_api_status() -> OfficialAPIStatus:
    """Report whether an official local ``ibapi`` installation is importable."""

    module_importable = importlib.util.find_spec("ibapi") is not None
    return OfficialAPIStatus(
        installed=False,
        module_importable=module_importable,
        provenance_verified=False,
        module_name="ibapi",
        distribution_source="official_ibkr_tws_api_zip_or_msi_only",
        installation_policy=(
            "Install from the official IBKR TWS API ZIP/MSI; PyPI and unofficial wrappers "
            "are not accepted by this experiment."
        ),
        blocker=(
            "importable_ibapi_distribution_provenance_not_verified"
            if module_importable
            else "official_ibkr_tws_python_api_not_installed_in_repository_environment"
        ),
    )
