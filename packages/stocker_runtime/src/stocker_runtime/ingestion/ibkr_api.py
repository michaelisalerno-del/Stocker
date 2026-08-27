"""Verified provenance for the optional first-party IBKR Python API."""

from __future__ import annotations

import hashlib
import html as html_module
import importlib.util
import json
import os
import re
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Literal
from urllib.parse import urljoin, urlparse
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OFFICIAL_API_PAGE: Literal["https://interactivebrokers.github.io/"] = (
    "https://interactivebrokers.github.io/"
)
OFFICIAL_DOWNLOAD_HOST = "interactivebrokers.github.io"
OFFICIAL_DOWNLOAD_PREFIX = "/downloads/twsapi_macunix."
IBKR_DEPENDENCY_BLOCKER = "blocked_official_ibkr_api_not_installed"
IBKR_PROVENANCE_BLOCKER = "blocked_unverified_official_ibkr_api"
IBKR_API_UPDATE_MAX_AGE = timedelta(days=14)


class OfficialIBKRApiProvenanceError(ValueError):
    """The installed client cannot be tied to an official IBKR archive."""


class OfficialIBKRDependencyError(RuntimeError):
    """The operator has not installed a provenance-bound official client."""


class OfficialIBKRApiRelease(BaseModel):
    """Current first-party Mac/Unix release advertised by IBKR."""

    model_config = ConfigDict(extra="forbid")

    release_channel: Literal["latest"] = "latest"
    platform: Literal["mac_unix"] = "mac_unix"
    api_version: str = Field(pattern=r"^\d+\.\d+$")
    release_date: date
    source_url: str

    @field_validator("source_url")
    @classmethod
    def _official_download_required(cls, value: str) -> str:
        return _require_official_download_url(value)


class OfficialIBKRApiProvenance(BaseModel):
    """Immutable operator record for one installed official API archive."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    source: Literal["interactive_brokers_official_tws_api"]
    release_channel: Literal["latest"]
    platform: Literal["mac_unix"]
    api_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    release_date: date
    official_page_url: Literal["https://interactivebrokers.github.io/"]
    official_page_checked_at_utc: datetime
    source_url: str
    archive_filename: str = Field(pattern=r"^twsapi_macunix\.\d+\.\d+\.zip$")
    archive_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    installed_tree_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    registered_at_utc: datetime
    registered_by: str = Field(min_length=1)

    @field_validator(
        "official_page_checked_at_utc",
        "registered_at_utc",
    )
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("provenance timestamps must be timezone-aware")
        return value

    @field_validator("source_url")
    @classmethod
    def _official_download_required(cls, value: str) -> str:
        return _require_official_download_url(value)

    @model_validator(mode="after")
    def _archive_identity_matches_source(self) -> OfficialIBKRApiProvenance:
        source_name = Path(urlparse(self.source_url).path).name
        if self.archive_filename != source_name:
            raise ValueError("archive filename does not match the official source URL")
        if self.source_tree_sha256 != self.installed_tree_sha256:
            raise ValueError("installed source tree does not match the official archive")
        return self


class OfficialIBKRApiUpdateStatus(BaseModel):
    """Result of a read-only comparison with IBKR's current release page."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    checked_at_utc: datetime
    installed_api_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    installed_source_url: str
    latest_api_version: str = Field(pattern=r"^\d+\.\d+$")
    latest_release_date: date
    latest_source_url: str
    update_available: bool
    automatic_installation: Literal[False] = False

    @field_validator("checked_at_utc")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("update check timestamp must be timezone-aware")
        return value

    @field_validator("installed_source_url", "latest_source_url")
    @classmethod
    def _official_download_required(cls, value: str) -> str:
        return _require_official_download_url(value)


def _require_official_download_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != OFFICIAL_DOWNLOAD_HOST
        or not parsed.path.startswith(OFFICIAL_DOWNLOAD_PREFIX)
        or not re.fullmatch(r"/downloads/twsapi_macunix\.\d+\.\d+\.zip", parsed.path)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("source_url must be the official IBKR Mac/Unix archive")
    return value


_MAC_UNIX_ARCHIVE_ANCHOR_PATTERN = re.compile(
    r"""
    <a\b
    (?=[^>]*\bhref=["'](?P<source_url>[^"']*twsapi_macunix\.\d+\.\d+\.zip)["'])
    [^>]*>
    (?P<label>.*?)
    </a>
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)
_RELEASE_METADATA_PATTERN = re.compile(
    r"""
    Version:\s*<strong>\s*API\s*(?P<api_version>\d+\.\d+)\s*</strong>
    .*?
    Release\s+Date:\s*<strong>\s*(?P<release_date>[A-Za-z]{3}\s+\d{1,2}\s+\d{4})\s*</strong>
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def parse_latest_official_ibkr_api_release(html: str) -> OfficialIBKRApiRelease:
    """Parse only the Latest Mac/Unix row from IBKR's official licence page."""

    candidates: list[OfficialIBKRApiRelease] = []
    for anchor in _MAC_UNIX_ARCHIVE_ANCHOR_PATTERN.finditer(html):
        label = html_module.unescape(_HTML_TAG_PATTERN.sub(" ", anchor.group("label")))
        normalized_label = " ".join(label.split())
        if not re.search(r"\bLatest\b", normalized_label, re.IGNORECASE):
            continue
        if not re.search(r"\bMac\s*/\s*Unix\b", normalized_label, re.IGNORECASE):
            continue
        cell_end = html.lower().find("</td", anchor.end())
        if cell_end < 0:
            continue
        metadata = _RELEASE_METADATA_PATTERN.search(html, anchor.end(), cell_end)
        if metadata is None:
            continue
        try:
            candidates.append(
                OfficialIBKRApiRelease(
                    api_version=metadata.group("api_version"),
                    release_date=datetime.strptime(
                        metadata.group("release_date"),
                        "%b %d %Y",
                    ).date(),
                    source_url=urljoin(
                        OFFICIAL_API_PAGE,
                        html_module.unescape(anchor.group("source_url")),
                    ),
                )
            )
        except Exception as exc:
            raise OfficialIBKRApiProvenanceError(
                "official IBKR Latest Mac/Unix release metadata is invalid"
            ) from exc
    if len(candidates) != 1:
        raise OfficialIBKRApiProvenanceError(
            "official IBKR Latest Mac/Unix release was missing or ambiguous"
        )
    return candidates[0]


def evaluate_official_ibkr_api_update(
    installed: OfficialIBKRApiProvenance,
    latest: OfficialIBKRApiRelease,
    *,
    checked_at: datetime,
) -> OfficialIBKRApiUpdateStatus:
    """Report a newer archive; never mutate or install the runtime dependency."""

    installed_series = tuple(int(value) for value in installed.api_version.split(".")[:2])
    latest_series = tuple(int(value) for value in latest.api_version.split("."))
    update_available = latest_series > installed_series or latest.source_url != installed.source_url
    return OfficialIBKRApiUpdateStatus(
        checked_at_utc=checked_at,
        installed_api_version=installed.api_version,
        installed_source_url=installed.source_url,
        latest_api_version=latest.api_version,
        latest_release_date=latest.release_date,
        latest_source_url=latest.source_url,
        update_available=update_available,
    )


def fetch_latest_official_ibkr_api_release() -> OfficialIBKRApiRelease:
    """Read current release metadata from IBKR without downloading API code."""

    try:
        response = httpx.get(
            OFFICIAL_API_PAGE,
            timeout=15.0,
            follow_redirects=False,
            headers={"User-Agent": "Stocker-IBKR-API-version-check/1"},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OfficialIBKRApiProvenanceError("official IBKR API release check failed") from exc
    if len(response.content) > 2_000_000:
        raise OfficialIBKRApiProvenanceError(
            "official IBKR API release page exceeds the safety bound"
        )
    return parse_latest_official_ibkr_api_release(response.text)


def _write_json_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            os.chmod(temporary, 0o644)
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive_atomic(path: Path, payload: str) -> None:
    """Create a complete file without any check-then-replace race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            os.chmod(temporary, 0o644)
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_immutable_official_ibkr_api_provenance(
    path: str | Path,
    provenance: OfficialIBKRApiProvenance,
) -> None:
    """Create one provenance record idempotently; never overwrite another."""

    output = Path(path)
    payload = provenance.model_dump_json(indent=2)
    if output.is_symlink():
        raise OfficialIBKRApiProvenanceError("provenance path may not be a symlink")
    if output.exists():
        existing = load_official_ibkr_api_provenance(output)
        if existing != provenance:
            raise OfficialIBKRApiProvenanceError(
                "installed official IBKR API provenance is immutable"
            )
        return
    try:
        _write_json_exclusive_atomic(output, payload)
    except FileExistsError:
        if output.is_symlink():
            raise OfficialIBKRApiProvenanceError("provenance path may not be a symlink") from None
        existing = load_official_ibkr_api_provenance(output)
        if existing != provenance:
            raise OfficialIBKRApiProvenanceError(
                "installed official IBKR API provenance is immutable"
            ) from None


def write_official_ibkr_api_update_status(
    path: str | Path,
    status: OfficialIBKRApiUpdateStatus,
) -> None:
    """Atomically replace the latest read-only release comparison."""

    output = Path(path)
    if output.is_symlink():
        raise OfficialIBKRApiProvenanceError("update status path may not be a symlink")
    _write_json_atomic(output, status.model_dump_json(indent=2))


def load_official_ibkr_api_provenance(
    path: str | Path,
) -> OfficialIBKRApiProvenance:
    """Load a strict provenance record without accepting unknown fields."""

    provenance_path = Path(path)
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        return OfficialIBKRApiProvenance.model_validate(payload)
    except Exception as exc:
        raise OfficialIBKRApiProvenanceError("official IBKR API provenance is invalid") from exc


def load_official_ibkr_api_update_status(
    path: str | Path,
) -> OfficialIBKRApiUpdateStatus:
    """Load the latest strict read-only version comparison."""

    status_path = Path(path)
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        return OfficialIBKRApiUpdateStatus.model_validate(payload)
    except Exception as exc:
        raise OfficialIBKRApiProvenanceError("official IBKR API update status is invalid") from exc


def official_ibkr_api_available() -> bool:
    """Return whether the optional first-party package can be discovered."""

    try:
        _installed_ibkr_api_package_root()
    except OfficialIBKRApiProvenanceError:
        return False
    return True


def _module_package_root(module: ModuleType) -> Path:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, (str, os.PathLike)):
        raise OfficialIBKRApiProvenanceError("installed ibapi module path is invalid")
    package_file = Path(module_file)
    package_root = package_file.parent
    if (
        package_file.is_symlink()
        or not package_file.is_file()
        or package_root.is_symlink()
        or not package_root.is_dir()
    ):
        raise OfficialIBKRApiProvenanceError("installed ibapi module path is invalid")
    return package_root


def _installed_ibkr_api_package_root() -> Path:
    loaded = sys.modules.get("ibapi")
    if isinstance(loaded, ModuleType):
        return _module_package_root(loaded)
    try:
        specification = importlib.util.find_spec("ibapi")
    except (ImportError, AttributeError, ValueError) as error:
        raise OfficialIBKRApiProvenanceError(
            "installed ibapi package cannot be discovered"
        ) from error
    if specification is None or not isinstance(specification.origin, str):
        raise OfficialIBKRApiProvenanceError("installed ibapi package cannot be discovered")
    placeholder = ModuleType("ibapi")
    placeholder.__file__ = specification.origin
    return _module_package_root(placeholder)


def require_official_ibkr_api(provenance_path: str | Path | None = None) -> ModuleType:
    """Bind the importable ``ibapi`` tree to immutable official provenance."""

    if not official_ibkr_api_available():
        raise OfficialIBKRDependencyError(
            f"{IBKR_DEPENDENCY_BLOCKER}: install the official TWS API Python client "
            "from the IBKR Latest Mac/Unix distribution"
        )
    configured_path = provenance_path or os.environ.get("STOCKER_IBKR_API_PROVENANCE")
    if configured_path is None or not Path(configured_path).is_file():
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: official archive provenance is absent"
        )
    try:
        provenance = load_official_ibkr_api_provenance(configured_path)
    except OfficialIBKRApiProvenanceError as error:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: official archive provenance is invalid"
        ) from error
    try:
        discovered_root = _installed_ibkr_api_package_root()
        installed_tree_sha256 = python_package_tree_sha256(discovered_root)
    except OfficialIBKRApiProvenanceError as error:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: installed ibapi tree is invalid"
        ) from error
    if installed_tree_sha256 != provenance.installed_tree_sha256:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: installed ibapi tree hash mismatch"
        )
    try:
        module = __import__("ibapi")
    except Exception as error:
        raise OfficialIBKRDependencyError(
            f"{IBKR_DEPENDENCY_BLOCKER}: verified ibapi package import failed"
        ) from error
    if not isinstance(module, ModuleType):
        raise OfficialIBKRDependencyError(f"{IBKR_DEPENDENCY_BLOCKER}: invalid ibapi module")
    try:
        imported_root = _module_package_root(module)
        imported_tree_sha256 = python_package_tree_sha256(imported_root)
    except OfficialIBKRApiProvenanceError as error:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: imported ibapi tree is invalid"
        ) from error
    if imported_root.resolve() != discovered_root.resolve():
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: imported ibapi path changed during verification"
        )
    if imported_tree_sha256 != provenance.installed_tree_sha256:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: imported ibapi tree hash mismatch"
        )
    if getattr(module, "__version__", None) != provenance.api_version:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: installed ibapi version mismatch"
        )
    try:
        client_module = __import__("ibapi.client", fromlist=("EClient",))
    except Exception as error:
        raise OfficialIBKRDependencyError(
            f"{IBKR_DEPENDENCY_BLOCKER}: verified ibapi client import failed"
        ) from error
    if not isinstance(client_module, ModuleType) or not isinstance(
        getattr(client_module, "EClient", None), type
    ):
        raise OfficialIBKRDependencyError(
            f"{IBKR_DEPENDENCY_BLOCKER}: verified ibapi client is invalid"
        )
    try:
        client_root = _module_package_root(client_module)
        client_tree_sha256 = python_package_tree_sha256(client_root)
    except OfficialIBKRApiProvenanceError as error:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: imported ibapi client path is invalid"
        ) from error
    if client_root.resolve() != imported_root.resolve():
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: imported ibapi client path changed during verification"
        )
    if client_tree_sha256 != provenance.installed_tree_sha256:
        raise OfficialIBKRDependencyError(
            f"{IBKR_PROVENANCE_BLOCKER}: imported ibapi tree changed during client verification"
        )
    return module


def official_ibkr_api_projection() -> dict[str, Any]:
    """Return secret-free installed-client and update health."""

    provenance_path = os.environ.get("STOCKER_IBKR_API_PROVENANCE")
    update_status_path = os.environ.get("STOCKER_IBKR_API_UPDATE_STATUS")
    projection: dict[str, Any] = {
        "installed": official_ibkr_api_available(),
        "verified": False,
        "api_version": None,
        "release_channel": None,
        "release_date": None,
        "update_checked_at_utc": None,
        "latest_api_version": None,
        "update_available": None,
        "update_status_fresh": False,
        "automatic_installation": False,
        "blocker": None,
    }
    try:
        require_official_ibkr_api(provenance_path)
        provenance = load_official_ibkr_api_provenance(str(provenance_path))
    except OfficialIBKRDependencyError as error:
        projection["blocker"] = str(error).split(":", 1)[0]
        return projection
    projection.update(
        {
            "verified": True,
            "api_version": provenance.api_version,
            "release_channel": provenance.release_channel,
            "release_date": provenance.release_date.isoformat(),
        }
    )
    if update_status_path is None or not Path(update_status_path).is_file():
        projection["blocker"] = "blocked_ibkr_api_update_status_missing"
        return projection
    try:
        status = load_official_ibkr_api_update_status(update_status_path)
    except OfficialIBKRApiProvenanceError:
        projection["blocker"] = "blocked_ibkr_api_update_status_invalid"
        return projection
    if (
        status.installed_api_version != provenance.api_version
        or status.installed_source_url != provenance.source_url
    ):
        projection["blocker"] = "blocked_ibkr_api_update_status_invalid"
        return projection
    projection.update(
        {
            "update_checked_at_utc": status.checked_at_utc.isoformat(),
            "latest_api_version": status.latest_api_version,
        }
    )
    update_age = datetime.now(UTC) - status.checked_at_utc.astimezone(UTC)
    if update_age > IBKR_API_UPDATE_MAX_AGE or update_age < timedelta(0):
        projection["blocker"] = "blocked_ibkr_api_update_check_stale"
        return projection
    projection.update(
        {
            "update_available": status.update_available,
            "update_status_fresh": True,
            "blocker": "blocked_outdated_official_ibkr_api" if status.update_available else None,
        }
    )
    return projection


def python_package_tree_sha256(package_root: str | Path) -> str:
    """Hash Python source names and bytes in a stable, path-independent order."""

    root = Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise OfficialIBKRApiProvenanceError("installed ibapi package root is invalid")
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise OfficialIBKRApiProvenanceError("installed ibapi package has no Python sources")
    named_bytes: list[tuple[str, bytes]] = []
    for path in files:
        if path.is_symlink():
            raise OfficialIBKRApiProvenanceError("installed ibapi package contains a symlink")
        named_bytes.append((path.relative_to(root).as_posix(), path.read_bytes()))
    return _named_bytes_sha256(named_bytes)


def _named_bytes_sha256(items: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, content in sorted(items):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_python_tree_and_version(archive: Path) -> tuple[str, str]:
    prefix = "IBJts/source/pythonclient/ibapi/"
    try:
        with ZipFile(archive) as bundle:
            version_text = bundle.read("IBJts/API_VersionNum.txt").decode("ascii").strip()
            match = re.fullmatch(r"API_Version=(\d+)\.(\d+)\.(\d+)", version_text)
            if match is None:
                raise OfficialIBKRApiProvenanceError(
                    "official archive API_VersionNum.txt is invalid"
                )
            api_version = ".".join(str(int(value)) for value in match.groups())
            named_bytes: list[tuple[str, bytes]] = []
            seen: set[str] = set()
            total_size = 0
            for info in bundle.infolist():
                if not info.filename.startswith(prefix) or not info.filename.endswith(".py"):
                    continue
                relative = info.filename.removeprefix(prefix)
                if (
                    not relative
                    or relative in seen
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                    or info.flag_bits & 0x1
                ):
                    raise OfficialIBKRApiProvenanceError(
                        "official archive contains an unsafe Python package member"
                    )
                seen.add(relative)
                total_size += info.file_size
                if info.file_size > 10_000_000 or total_size > 100_000_000:
                    raise OfficialIBKRApiProvenanceError(
                        "official archive Python package exceeds safety bounds"
                    )
                named_bytes.append((relative, bundle.read(info)))
    except (BadZipFile, KeyError, UnicodeDecodeError) as exc:
        raise OfficialIBKRApiProvenanceError("official IBKR API archive is invalid") from exc
    if not named_bytes:
        raise OfficialIBKRApiProvenanceError(
            "official archive does not contain the Python API package"
        )
    return _named_bytes_sha256(named_bytes), api_version


def inspect_official_ibkr_api_archive(
    archive_path: str | Path,
    *,
    installed_package_root: str | Path,
    release: OfficialIBKRApiRelease,
    registered_by: str,
    checked_at: datetime,
) -> OfficialIBKRApiProvenance:
    """Bind an installed ``ibapi`` tree to the matching official release ZIP."""

    archive = Path(archive_path)
    if not archive.is_file() or archive.is_symlink():
        raise OfficialIBKRApiProvenanceError("official IBKR API archive is absent")
    expected_filename = Path(urlparse(release.source_url).path).name
    if archive.name != expected_filename:
        raise OfficialIBKRApiProvenanceError(
            "archive filename does not match the current official release"
        )
    source_tree_sha256, api_version = _archive_python_tree_and_version(archive)
    if ".".join(api_version.split(".")[:2]) != release.api_version:
        raise OfficialIBKRApiProvenanceError(
            "archive API version does not match the current official release"
        )
    installed_tree_sha256 = python_package_tree_sha256(installed_package_root)
    if installed_tree_sha256 != source_tree_sha256:
        raise OfficialIBKRApiProvenanceError(
            "installed ibapi sources do not match the official archive"
        )
    return OfficialIBKRApiProvenance(
        schema_version="1",
        source="interactive_brokers_official_tws_api",
        release_channel=release.release_channel,
        platform=release.platform,
        api_version=api_version,
        release_date=release.release_date,
        official_page_url=OFFICIAL_API_PAGE,
        official_page_checked_at_utc=checked_at,
        source_url=release.source_url,
        archive_filename=archive.name,
        archive_sha256=_file_sha256(archive),
        source_tree_sha256=source_tree_sha256,
        installed_tree_sha256=installed_tree_sha256,
        registered_at_utc=checked_at,
        registered_by=registered_by,
    )
