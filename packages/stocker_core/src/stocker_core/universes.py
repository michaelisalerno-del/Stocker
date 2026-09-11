"""Small, broker-independent universe definitions and lookup."""

import csv
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

from stocker_core.markets import MarketUniverseSpec

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

NAMED_US_UNIVERSES = ("US_ALL", "NASDAQ", "NYSE")
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
_NAMED_US_PRIMARY_EXCHANGES = {
    "US_ALL": frozenset({"AMEX", "ARCA", "BATS", "IEX", "NASDAQ", "NYSE"}),
    "NASDAQ": frozenset({"NASDAQ"}),
    "NYSE": frozenset({"NYSE"}),
}
_NAMED_US_NAMES = {
    "US_ALL": "US listed non-ETF securities",
    "NASDAQ": "NASDAQ listed non-ETF securities",
    "NYSE": "NYSE listed non-ETF securities",
}


class InstrumentReference(BaseModel):
    """Unresolved stock identity sufficient for later IBKR qualification."""

    model_config = ConfigDict(frozen=True)

    symbol: Identifier
    exchange: Identifier
    primary_exchange: Identifier | None = None
    currency: Identifier
    security_type: Identifier = "STK"

    @field_validator("symbol", "exchange", "primary_exchange", "currency", "security_type")
    @classmethod
    def normalize_market_identity(cls, value: str | None) -> str | None:
        """Normalize identity fields so equal references compare equal across universes."""

        return value.upper() if value is not None else None


class UniverseDefinition(BaseModel):
    """One immutable named collection of eligible instrument references."""

    model_config = ConfigDict(frozen=True)

    universe_id: Identifier
    name: Identifier
    members: tuple[InstrumentReference, ...] = ()
    market_spec: MarketUniverseSpec | None = None

    @model_validator(mode="after")
    def require_members_or_market_spec(self) -> "UniverseDefinition":
        if not self.members and self.market_spec is None:
            raise ValueError("universe requires static members or a market specification")
        return self

    @field_validator("members")
    @classmethod
    def remove_duplicate_members(
        cls, members: tuple[InstrumentReference, ...]
    ) -> tuple[InstrumentReference, ...]:
        """Keep first occurrence order while removing exact normalized duplicates."""

        return tuple(dict.fromkeys(members))


@dataclass(frozen=True, slots=True)
class UsUniverseSnapshotMetadata:
    """Provenance and freshness metadata retained beside cached memberships."""

    source: str
    source_urls: str
    retrieved_at: datetime
    nasdaq_source_updated_at: str
    other_source_updated_at: str


@dataclass(frozen=True, slots=True)
class UsUniverseSnapshot:
    """One local Nasdaq Trader listing snapshot used by the named US universes."""

    metadata: UsUniverseSnapshotMetadata
    members: tuple[InstrumentReference, ...]

    def get_universe(self, universe_id: str) -> UniverseDefinition:
        """Materialize one supported broker-independent named universe."""

        normalized = universe_id.strip().upper()
        exchanges = _NAMED_US_PRIMARY_EXCHANGES.get(normalized)
        if exchanges is None:
            raise ValueError(f"Unsupported named US universe: {universe_id}")
        members = tuple(member for member in self.members if member.primary_exchange in exchanges)
        if not members:
            raise ValueError(f"Named universe {normalized} has no members in the snapshot")
        return UniverseDefinition(
            universe_id=normalized,
            name=_NAMED_US_NAMES[normalized],
            members=members,
        )


@dataclass(frozen=True, slots=True)
class UsUniverseSnapshotRefresh:
    """Summary of one explicit official-directory snapshot refresh."""

    output_path: Path
    retrieved_at: datetime
    us_all_count: int
    nasdaq_count: int
    nyse_count: int


def load_us_universe_snapshot(path: str | Path) -> UsUniverseSnapshot:
    """Load and validate one timestamped local US listing snapshot."""

    snapshot_path = Path(path)
    metadata: dict[str, str] = {}
    csv_lines: list[str] = []
    for line in snapshot_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            key, separator, value = line.removeprefix("#").strip().partition("=")
            if separator:
                metadata[key.strip()] = value.strip()
        elif line.strip():
            csv_lines.append(line)

    required_metadata = (
        "schema_version",
        "source",
        "source_urls",
        "retrieved_at",
        "nasdaq_source_updated_at",
        "other_source_updated_at",
        "universes",
    )
    missing_metadata = tuple(key for key in required_metadata if not metadata.get(key))
    if missing_metadata:
        raise ValueError(f"US universe snapshot is missing metadata: {', '.join(missing_metadata)}")
    if metadata["schema_version"] != "1":
        raise ValueError("Unsupported US universe snapshot schema version")
    declared_universes = tuple(
        item.strip().upper() for item in metadata["universes"].split(",") if item.strip()
    )
    if declared_universes != NAMED_US_UNIVERSES:
        raise ValueError("US universe snapshot does not declare US_ALL,NASDAQ,NYSE")
    retrieved_at = datetime.fromisoformat(metadata["retrieved_at"])
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("US universe snapshot retrieved_at must include a UTC offset")

    reader = csv.DictReader(csv_lines)
    if reader.fieldnames != ["symbol", "primary_exchange"]:
        raise ValueError("US universe snapshot must contain symbol,primary_exchange columns")
    members: list[InstrumentReference] = []
    for index, row in enumerate(reader, start=2):
        symbol = (row.get("symbol") or "").strip()
        primary_exchange = (row.get("primary_exchange") or "").strip().upper()
        if not symbol or primary_exchange not in _NAMED_US_PRIMARY_EXCHANGES["US_ALL"]:
            raise ValueError(f"Invalid US universe snapshot row {index}")
        members.append(
            InstrumentReference(
                symbol=symbol,
                exchange="SMART",
                primary_exchange=primary_exchange,
                currency="USD",
            )
        )
    normalized_members = tuple(
        sorted(
            dict.fromkeys(members),
            key=lambda item: (item.symbol, item.primary_exchange or ""),
        )
    )
    if not normalized_members:
        raise ValueError("US universe snapshot contains no members")
    return UsUniverseSnapshot(
        metadata=UsUniverseSnapshotMetadata(
            source=metadata["source"],
            source_urls=metadata["source_urls"],
            retrieved_at=retrieved_at,
            nasdaq_source_updated_at=metadata["nasdaq_source_updated_at"],
            other_source_updated_at=metadata["other_source_updated_at"],
        ),
        members=normalized_members,
    )


def refresh_us_universe_snapshot(
    path: str | Path,
    *,
    fetch_text: Callable[[str], str] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> UsUniverseSnapshotRefresh:
    """Fetch official US symbol directories and atomically refresh the local snapshot."""

    if fetch_text is None:
        import httpx

        def fetch_text(url: str) -> str:
            response = httpx.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            return response.text

    nasdaq_text = fetch_text(NASDAQ_LISTED_URL)
    other_text = fetch_text(OTHER_LISTED_URL)
    members = _parse_nasdaq_listed(nasdaq_text) + _parse_other_listed(other_text)
    normalized_members = tuple(
        sorted(
            dict.fromkeys(members),
            key=lambda item: (item.symbol, item.primary_exchange or ""),
        )
    )
    if not normalized_members:
        raise ValueError("Nasdaq Trader symbol directories contained no eligible members")
    retrieved_at = (clock or (lambda: datetime.now(tz=UTC)))()
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("US universe snapshot refresh clock must be timezone-aware")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = StringIO()
    buffer.write("# schema_version=1\n")
    buffer.write("# source=NASDAQ_TRADER_SYMBOL_DIRECTORY\n")
    buffer.write(f"# source_urls={NASDAQ_LISTED_URL},{OTHER_LISTED_URL}\n")
    buffer.write(f"# retrieved_at={retrieved_at.astimezone(UTC).isoformat()}\n")
    buffer.write(f"# nasdaq_source_updated_at={_source_updated_at(nasdaq_text)}\n")
    buffer.write(f"# other_source_updated_at={_source_updated_at(other_text)}\n")
    buffer.write(f"# universes={','.join(NAMED_US_UNIVERSES)}\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("symbol", "primary_exchange"))
    writer.writerows((member.symbol, member.primary_exchange) for member in normalized_members)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(buffer.getvalue(), encoding="utf-8")
    temporary.replace(output_path)
    snapshot = load_us_universe_snapshot(output_path)
    return UsUniverseSnapshotRefresh(
        output_path=output_path,
        retrieved_at=snapshot.metadata.retrieved_at,
        us_all_count=len(snapshot.get_universe("US_ALL").members),
        nasdaq_count=len(snapshot.get_universe("NASDAQ").members),
        nyse_count=len(snapshot.get_universe("NYSE").members),
    )


def _parse_nasdaq_listed(text: str) -> tuple[InstrumentReference, ...]:
    rows = _source_rows(text)
    required = {"Symbol", "Test Issue", "ETF", "NextShares"}
    if rows.fieldnames is None or not required.issubset(rows.fieldnames):
        raise ValueError("Invalid Nasdaq-listed symbol directory header")
    return tuple(
        InstrumentReference(
            symbol=row["Symbol"],
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
        )
        for row in rows
        if row.get("Test Issue") == "N" and row.get("ETF") == "N" and row.get("NextShares") != "Y"
    )


def _parse_other_listed(text: str) -> tuple[InstrumentReference, ...]:
    rows = _source_rows(text)
    required = {"ACT Symbol", "Exchange", "ETF", "Test Issue"}
    if rows.fieldnames is None or not required.issubset(rows.fieldnames):
        raise ValueError("Invalid other-listed symbol directory header")
    exchange_names = {
        "A": "AMEX",
        "N": "NYSE",
        "P": "ARCA",
        "Z": "BATS",
        "V": "IEX",
    }
    return tuple(
        InstrumentReference(
            symbol=row["ACT Symbol"],
            exchange="SMART",
            primary_exchange=exchange_names[row["Exchange"]],
            currency="USD",
        )
        for row in rows
        if row.get("Test Issue") == "N"
        and row.get("ETF") == "N"
        and row.get("Exchange") in exchange_names
    )


def _source_rows(text: str) -> csv.DictReader[str]:
    lines = [line for line in text.splitlines() if not line.startswith("File Creation Time:")]
    return csv.DictReader(lines, delimiter="|")


def _source_updated_at(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("File Creation Time:"):
            value = line.split("|", maxsplit=1)[0].removeprefix("File Creation Time:").strip()
            if value:
                return value
    raise ValueError("Nasdaq Trader symbol directory has no file creation timestamp")


class UniverseCatalog:
    """Explicit in-memory lookup for configured universe definitions."""

    def __init__(self, universes: Iterable[UniverseDefinition]) -> None:
        self._universes: dict[str, UniverseDefinition] = {}
        for universe in universes:
            if universe.universe_id in self._universes:
                raise ValueError(f"Duplicate universe_id: {universe.universe_id}")
            self._universes[universe.universe_id] = universe

    def get_universe(self, universe_id: str) -> UniverseDefinition:
        """Return one universe or reject an unknown identifier clearly."""

        try:
            return self._universes[universe_id]
        except KeyError as exc:
            raise ValueError(f"Unknown universe: {universe_id}") from exc

    def get_members(self, universe_id: str) -> tuple[InstrumentReference, ...]:
        """Return the immutable members of one configured universe."""

        return self.get_universe(universe_id).members

    def list_universes(self) -> tuple[UniverseDefinition, ...]:
        """List definitions in deterministic configuration order."""

        return tuple(self._universes.values())
