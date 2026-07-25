"""Verified provenance for the optional first-party IBKR Python API."""

from __future__ import annotations

import hashlib
import html as html_module
import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal
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


class OfficialIBKRApiProvenanceError(ValueError):
    """The installed client cannot be tied to an official IBKR archive."""


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


def python_package_tree_sha256(package_root: str | Path) -> str:
    """Hash Python source names and bytes in a stable, path-independent order."""

    root = Path(package_root)
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
