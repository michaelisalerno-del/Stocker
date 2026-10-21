"""The paper record's only settings: which Saxo environment and where its credentials live.

The trading rules themselves are frozen constants in `methods.py`, never configuration.
"""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

Environment = Literal["SAXO_SIM", "SAXO_LIVE"]
PAGE_SIZE = 100  # dashboard record rows per request


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SaxoSettings(Strict):
    # Secrets and account keys are read from a mode-0600 file, never from dashboard JSON.
    credentials_file: Path | None = None
    redirect_uri: str = "http://127.0.0.1:8765/oauth/saxo/callback"

    @field_validator("redirect_uri")
    @classmethod
    def redirect(cls, value: str) -> str:
        p = urlsplit(value)
        if (
            p.username
            or p.password
            or p.query
            or p.fragment
            or p.path != "/oauth/saxo/callback"
            or not p.hostname
            or (
                p.scheme != "https"
                and not (p.scheme == "http" and p.hostname in {"127.0.0.1", "localhost", "::1"})
            )
        ):
            raise ValueError("INVALID_OAUTH_REDIRECT")
        return value


class PaperConfig(Strict):
    data_environment: Environment = "SAXO_SIM"
    saxo: SaxoSettings = Field(default_factory=SaxoSettings)


def load(path: Path) -> PaperConfig:
    return PaperConfig.model_validate(yaml.safe_load(path.read_text()))
