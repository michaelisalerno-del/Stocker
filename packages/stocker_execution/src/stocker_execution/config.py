"""Saxo data and paper execution are independent, closed sets of capabilities."""

from datetime import date
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import yaml
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

# 2026-10-04, the user's decision: gas and silver removed (part-week same-day options, the widest
# spreads); E-mini S&P added, traded exactly like NQ.
Market = Literal["CL", "ES", "GC", "NQ"]
MARKETS: tuple[Market, ...] = ("CL", "ES", "GC", "NQ")
Environment = Literal["SAXO_SIM", "SAXO_LIVE"]
RULE_VERSION = "CLOCK60_23H_ES_20261004"
# One whole contract per trade, whatever it costs up to this ceiling. The ceiling guards against
# a bad quote; it is not a budget (2026-09-30: "the lowest possible amount, whether it's 200 or
# more"). Earlier policies were £10 and then £50; their reservations keep those amounts.
MAX_PREMIUM_RISK_GBP = 1000
MAX_PREMIUM_RISK_PENNIES = MAX_PREMIUM_RISK_GBP * 100
MAX_OPEN_POSITIONS = 4
MAX_SIMULTANEOUS_ENTRY_RISK_GBP = MAX_PREMIUM_RISK_GBP * MAX_OPEN_POSITIONS
# A quote counts as current for this long after it was last known to stand: its last change, or
# (2026-10-01, the user's request) any later moment the socket delivered while its subscription
# was heartbeated, unpaused and gap-free, since Saxo sends a price only when it changes.
QUOTE_MAX_AGE_SECONDS: Final = 5
# Saxo streaming subscriptions owned by this application: our own guard. Saxo publishes no count
# limit; scripts/saxo_subscription_capacity_probe.py held 55 in one session on 2026-10-04.
SUBSCRIPTION_LIMIT = 48
ROLLING_WINDOW_SECONDS = 15 * 60  # server-held L1/L2 prehistory per instrument
OPTION_METADATA_MAX_AGE_SECONDS = 15 * 60
PAGE_SIZE = 100  # dashboard history/execution rows per request


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RecorderConfig(Strict):
    minimum_post_event_minutes: int = Field(default=60, ge=60, le=1440)
    post_close_minutes: int = Field(default=5, ge=5, le=60)
    rolling_max_bytes: int = Field(default=32 * 1024**2, ge=65536, le=128 * 1024**2)
    max_message_bytes: int = Field(default=256 * 1024, ge=1024, le=1024**2)
    queue_max_bytes: int = Field(default=8 * 1024**2, ge=65536, le=32 * 1024**2)
    queue_max_items: int = Field(default=512, ge=2, le=4096)
    # 2026-10-04: clocks every hour of the CME session record almost continuously, and the chain
    # window grew to 61 strikes the same day; 80 GiB is the most the 116 GB server disk can hold
    # for the frozen tests' data to the end of November. The System page shows the measured rate.
    archive_max_bytes: int = Field(default=2 * 1024**3, ge=65536, le=80 * 1024**3)
    disk_reserve_bytes: int = Field(default=2 * 1024**3, ge=0)
    bar_max_bytes: int = Field(default=128 * 1024**2, ge=65536)
    max_open_captures: int = Field(default=32, ge=5, le=64)
    # Citation/permission record is required; an API schema is not a licence.
    persistent_capture: bool = False
    recording_permission_evidence: str | None = None

    @model_validator(mode="after")
    def permission(self) -> "RecorderConfig":
        if self.persistent_capture and not self.recording_permission_evidence:
            raise ValueError("RECORDING_PERMISSION_NOT_VERIFIED")
        return self


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


class AlertSettings(Strict):
    # A mode-0600 JSON file {"url": "https://..."} (for example an ntfy topic); never inline.
    url_file: Path | None = None
    stream_down_seconds: int = Field(default=120, ge=30, le=3600)
    # Saxo's refresh token lasts about an hour but rotates at every ~20-minute access-token
    # refresh, so a healthy session always shows 40-60 minutes left. Less means renewal stopped.
    login_warning_minutes: int = Field(default=15, ge=5, le=30)


class ContractSelection(Strict):
    """Explicit environment-specific reference selection; no guessed or portable UICs."""

    environment: Environment
    uic: int = Field(gt=0)
    symbol: str = Field(min_length=1, max_length=100)
    exchange: str = Field(min_length=1, max_length=40)
    contract_month: str = Field(pattern=r"^\d{4}-\d{2}$")
    approval: str = Field(min_length=10, max_length=1000)

    @field_validator("symbol")
    @classmethod
    def non_crypto(cls, value: str) -> str:
        # Positive family check is also enforced against returned reference metadata.
        if any(s in value.upper() for s in ("BTC", "MBT", "BFF", "BRR", "BITCOIN", "CRYPTO")):
            raise ValueError("CRYPTO_INSTRUMENT_REJECTED")
        return value


class OptionApproval(Strict):
    environment: Environment
    # The exchange lists one root per weekday and week (for example crude "Mon Weekly (1)"),
    # each with a single expiry, so a same-day option every weekday needs the approved family.
    option_root_ids: tuple[int, ...] = Field(min_length=1, max_length=32)
    delta_tolerance: float = Field(ge=0, lt=0.1)
    source: str = Field(min_length=10)
    approval: str = Field(min_length=10)
    fee_per_side_gbp: float = Field(gt=0, lt=10)
    fee_evidence: str = Field(min_length=10)
    # Actual cutoff comes from verified option-specific reference evidence, never this config.
    selection: Literal["NEAREST_FROZEN_MODEL_DELTA"] = "NEAREST_FROZEN_MODEL_DELTA"
    # ExpiryDate is a date, not the model's expiry instant. Evidence is specific
    # to this approved root and date; never infer it from exercise time. Either list the
    # instants, or approve the exchange's New York clock time: an expiry day then counts only
    # when Saxo's timestamped LastTradeDate falls on that day at exactly that time.
    expiry_instants: dict[str, AwareDatetime] = Field(default_factory=dict)
    expiry_clock_new_york: str | None = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    expiry_time_evidence: str | None = Field(default=None, min_length=10)
    # SAME_DAY is the frozen method. SAME_DAY_OR_NEXT_LISTED (the user's choice 2026-10-01, and for
    # every market from 2026-10-04 so the overnight clocks have an option) takes the nearest listed
    # expiry still trading after the 60-minute exit: today's when there is one, otherwise the next.
    expiry_rule: Literal["SAME_DAY", "SAME_DAY_OR_NEXT_LISTED"] = "SAME_DAY"

    @model_validator(mode="after")
    def expiry_evidence(self) -> "OptionApproval":
        if (
            len(set(self.option_root_ids)) != len(self.option_root_ids)
            or min(self.option_root_ids) <= 0
        ):
            raise ValueError("OPTION_ROOTS_MUST_BE_UNIQUE_POSITIVE_IDS")
        if self.expiry_instants and self.expiry_clock_new_york:
            raise ValueError("ONE_EXPIRY_TIME_SOURCE_ONLY")
        if (self.expiry_instants or self.expiry_clock_new_york) and not self.expiry_time_evidence:
            raise ValueError("OPTION_EXPIRY_INSTANT_EVIDENCE_REQUIRED")
        for day, instant in self.expiry_instants.items():
            if (
                date.fromisoformat(day).isoformat() != day
                or instant.astimezone(ZoneInfo("America/New_York")).date().isoformat() != day
            ):
                raise ValueError("OPTION_EXPIRY_INSTANT_DATE_MISMATCH")
        return self


class FuturesConfig(Strict):
    data_environment: Environment = "SAXO_SIM"
    execution_mode: Literal["DISABLED", "INTERNAL_PAPER", "SAXO_SIM"] = "DISABLED"
    armed: bool = False
    markets: tuple[Market, ...] = MARKETS
    saxo: SaxoSettings = Field(default_factory=SaxoSettings)
    contracts: dict[Market, ContractSelection] = Field(default_factory=dict)
    # The following contract month, streamed and recorded with every clock before its re-pin
    # (2026-10-04) so the roll starts with history; never a decision input.
    next_contracts: dict[Market, ContractSelection] = Field(default_factory=dict)
    mappings: dict[Market, OptionApproval] = Field(default_factory=dict)
    reference_selections_file: Path | None = None
    # Optional display/observation context; never an entry rule.
    event_calendar_file: Path | None = None
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    option_subscription_budget: int = Field(default=16, ge=4, le=16)
    option_candidate_window: int = Field(default=3, ge=1, le=3)
    # Strikes in the observation-only options-chain window (Saxo caps a chain at 100). 25 strikes
    # spanned about one expected move, too narrow for the open-interest gamma and wall columns, so
    # the bound was lifted on 2026-10-04 (live: 61); the recorder's archive rate pays for it.
    option_chain_strikes: int = Field(default=11, ge=3, le=100)
    # Scale of the chain's Greeks.MidVolatility, which Saxo does not document. A live SIM
    # probe (docs/saxo-field-probe-20260929.json) read annual fractions, e.g. NG 0.651 between
    # bid/ask volatilities 0.645/0.657. Set UNVERIFIED to hide the IV spread again.
    provider_volatility_scale: Literal["UNVERIFIED", "FRACTION", "PERCENT"] = "FRACTION"
    # Frozen: a configuration file may restate it but never change it.
    entry_deadline_seconds: Literal[20] = 20

    @model_validator(mode="after")
    def separate(self) -> "FuturesConfig":
        if self.markets != MARKETS:
            raise ValueError("EXACT_MARKET_UNIVERSE_REQUIRED")
        if self.execution_mode == "SAXO_SIM" and self.data_environment != "SAXO_SIM":
            raise ValueError("SIM_EXECUTION_REQUIRES_SIM_DATA_AND_ACCOUNT")
        if any(
            v.environment != self.data_environment
            for group in (self.contracts, self.next_contracts, self.mappings)
            for v in group.values()
        ):
            raise ValueError("CROSS_ENVIRONMENT_IDENTITY_REJECTED")
        if any(
            m in self.contracts and v.uic == self.contracts[m].uic
            for m, v in self.next_contracts.items()
        ):
            raise ValueError("NEXT_CONTRACT_MUST_DIFFER_FROM_PINNED_CONTRACT")
        if self.armed:
            raise ValueError("START_DISARMED_USE_EXPLICIT_PAPER_ARM_AFTER_PREFLIGHT")
        return self


def load(path: Path) -> FuturesConfig:
    return FuturesConfig.model_validate(yaml.safe_load(path.read_text()))
