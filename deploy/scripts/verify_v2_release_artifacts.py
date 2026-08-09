#!/usr/bin/env python3
"""Fail-closed admission for the derived official IBKR API wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import stat
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import unquote, urlparse

API_VERSION = "10.49.1"
WHEEL_NAME = "ibapi-10.49.1-py3-none-any.whl"
DIST_INFO = "ibapi-10.49.1.dist-info"
ACTIVE_PROVENANCE = Path("/var/lib/stocker/ibkr-api/active-provenance.json")
PROVENANCE_TARGET = Path("/var/lib/stocker/ibkr-api/provenance/10.49.1.json")
PROVENANCE_ROOT = Path("/var/lib/stocker/ibkr-api")
PROVENANCE_TRUST_ANCHOR = Path("/")
REQUIRED_OWNER_UID = 0
MAX_WHEEL_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_ENTRIES = 2_048
MAX_PATH_BYTES = 512
WHEEL_METADATA_MEMBERS = frozenset(
    {
        f"{DIST_INFO}/METADATA",
        f"{DIST_INFO}/WHEEL",
        f"{DIST_INFO}/top_level.txt",
        f"{DIST_INFO}/RECORD",
    }
)
INSTALLER_METADATA_MEMBERS = frozenset(
    {
        f"{DIST_INFO}/INSTALLER",
        f"{DIST_INFO}/REQUESTED",
        f"{DIST_INFO}/direct_url.json",
        f"{DIST_INFO}/uv_cache.json",
    }
)
PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "source",
        "release_channel",
        "platform",
        "api_version",
        "release_date",
        "official_page_url",
        "official_page_checked_at_utc",
        "source_url",
        "archive_filename",
        "archive_sha256",
        "source_tree_sha256",
        "installed_tree_sha256",
        "registered_at_utc",
        "registered_by",
    }
)


class ArtifactVerificationError(RuntimeError):
    """A release artifact or its trust boundary failed admission."""


@dataclass(frozen=True)
class WheelEvidence:
    contents: dict[str, bytes]
    source_contents: dict[str, bytes]


def _raise(message: str) -> None:
    raise ArtifactVerificationError(message)


def _lstat(path: Path, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise ArtifactVerificationError(f"{label} is unavailable") from error


def _require_secure_directory(path: Path, label: str, required_uid: int) -> None:
    metadata = _lstat(path, label)
    if not stat.S_ISDIR(metadata.st_mode):
        _raise(f"{label} is not a real directory")
    if metadata.st_uid != required_uid:
        _raise(f"{label} owner is invalid")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        _raise(f"{label} is group/world writable")


def _require_secure_ancestry(path: Path, anchor: Path, required_uid: int) -> None:
    try:
        relative = path.relative_to(anchor)
    except ValueError as error:
        raise ArtifactVerificationError("provenance path escapes its trust anchor") from error
    current = anchor
    _require_secure_directory(current, "provenance parent", required_uid)
    for part in relative.parts:
        current /= part
        _require_secure_directory(current, "provenance parent", required_uid)


def _read_regular_bounded(
    path: Path,
    label: str,
    *,
    limit: int = MAX_MEMBER_BYTES,
) -> bytes:
    metadata = _lstat(path, label)
    if not stat.S_ISREG(metadata.st_mode):
        _raise(f"{label} is not a real file")
    if metadata.st_size < 0 or metadata.st_size > limit:
        _raise(f"{label} size is invalid")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArtifactVerificationError(f"{label} is unreadable") from error
    if len(payload) != metadata.st_size:
        _raise(f"{label} size changed while reading")
    return payload


def _json_without_duplicate_keys(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _raise(f"{label} contains duplicate keys")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactVerificationError(f"{label} is invalid JSON") from error
    if not isinstance(decoded, dict):
        _raise(f"{label} is not a JSON object")
    return cast(dict[str, Any], decoded)


def verify_provenance_link(
    provenance_link: Path = ACTIVE_PROVENANCE,
    *,
    expected_link: Path = ACTIVE_PROVENANCE,
    expected_target: Path = PROVENANCE_TARGET,
    trusted_root: Path = PROVENANCE_ROOT,
    trust_anchor: Path = PROVENANCE_TRUST_ANCHOR,
    required_uid: int = REQUIRED_OWNER_UID,
) -> dict[str, Any]:
    """Verify the exact root-owned active-provenance symlink and target."""

    if not all(
        path.is_absolute() for path in (expected_link, expected_target, trusted_root, trust_anchor)
    ):
        _raise("provenance boundary paths must be absolute")
    if provenance_link != expected_link:
        _raise("active provenance path is not the exact admitted link")
    if expected_link.parent != trusted_root:
        _raise("active provenance link parent is invalid")
    if expected_target.parent != trusted_root / "provenance":
        _raise("active provenance target parent is invalid")
    _require_secure_ancestry(trusted_root, trust_anchor, required_uid)
    _require_secure_ancestry(expected_target.parent, trust_anchor, required_uid)

    link_metadata = _lstat(provenance_link, "active provenance link")
    if not stat.S_ISLNK(link_metadata.st_mode):
        _raise("active provenance path is not a symlink")
    if link_metadata.st_uid != required_uid:
        _raise("active provenance link owner is invalid")
    try:
        literal_target = os.readlink(provenance_link)
    except OSError as error:
        raise ArtifactVerificationError("active provenance link is unreadable") from error
    if literal_target != str(expected_target):
        _raise("active provenance link literal target is invalid")

    target_metadata = _lstat(expected_target, "active provenance target")
    if not stat.S_ISREG(target_metadata.st_mode):
        _raise("active provenance target is not a real file")
    if target_metadata.st_uid != required_uid:
        _raise("active provenance target owner is invalid")
    if stat.S_IMODE(target_metadata.st_mode) & 0o022:
        _raise("active provenance target is group/world writable")
    try:
        resolved_link = provenance_link.resolve(strict=True)
        resolved_target = expected_target.resolve(strict=True)
    except OSError as error:
        raise ArtifactVerificationError("active provenance link is dangling") from error
    if resolved_target != expected_target or resolved_link != expected_target:
        _raise("active provenance resolved target is invalid")

    record = _json_without_duplicate_keys(
        _read_regular_bounded(
            expected_target,
            "active provenance target",
            limit=MAX_METADATA_BYTES,
        ),
        "active provenance",
    )
    if frozenset(record) != PROVENANCE_KEYS:
        _raise("active provenance fields are invalid")
    required_values = {
        "schema_version": "1",
        "source": "interactive_brokers_official_tws_api",
        "release_channel": "latest",
        "platform": "mac_unix",
        "api_version": API_VERSION,
        "official_page_url": "https://interactivebrokers.github.io/",
    }
    if any(record.get(key) != value for key, value in required_values.items()):
        _raise("active provenance identity is invalid")
    for field in ("archive_sha256", "source_tree_sha256", "installed_tree_sha256"):
        value = record.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            _raise(f"active provenance {field} is invalid")
    if record["source_tree_sha256"] != record["installed_tree_sha256"]:
        _raise("active provenance source and installed tree hashes differ")
    return record


def _canonical_member(name: str) -> str:
    if (
        not name
        or not name.isascii()
        or len(name.encode("ascii")) > MAX_PATH_BYTES
        or "\\" in name
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in name)
        or name.startswith("/")
        or name.endswith("/")
    ):
        _raise("wheel member path is invalid")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _raise("wheel member path is non-canonical")
    if PurePosixPath(name).as_posix() != name:
        _raise("wheel member path is non-canonical")
    return name


def _source_tree(source_root: Path) -> tuple[dict[str, bytes], str]:
    source_metadata = _lstat(source_root, "official source root")
    package_root = source_root / "ibapi"
    package_metadata = _lstat(package_root, "official ibapi source")
    if not stat.S_ISDIR(source_metadata.st_mode) or stat.S_ISLNK(source_metadata.st_mode):
        _raise("official source root is invalid")
    if not stat.S_ISDIR(package_metadata.st_mode) or stat.S_ISLNK(package_metadata.st_mode):
        _raise("official ibapi source is invalid")

    contents: dict[str, bytes] = {}
    digest_items: list[tuple[str, bytes]] = []
    total = 0
    visited = 0
    for current, directories, filenames in os.walk(package_root, followlinks=False):
        current_path = Path(current)
        visited += 1 + len(directories) + len(filenames)
        if visited > MAX_ENTRIES:
            _raise("official ibapi source tree has too many entries")
        for directory in directories:
            if (current_path / directory).is_symlink():
                _raise("official ibapi source contains a symlink")
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                _raise("official ibapi source contains a symlink")
            if path.suffix != ".py":
                continue
            relative = path.relative_to(package_root).as_posix()
            _canonical_member(relative)
            payload = _read_regular_bounded(path, "official ibapi source member")
            total += len(payload)
            if total > MAX_TOTAL_BYTES or len(contents) >= MAX_ENTRIES:
                _raise("official ibapi source tree is too large")
            contents[f"ibapi/{relative}"] = payload
            digest_items.append((relative, payload))
    if not contents:
        _raise("official ibapi source has no Python files")
    digest = hashlib.sha256()
    for relative, payload in sorted(digest_items):
        digest.update(relative.encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
        digest.update(b"\0")
    return contents, digest.hexdigest()


def _read_wheel(wheel: Path) -> dict[str, bytes]:
    metadata = _lstat(wheel, "IBAPI wheel")
    if wheel.name != WHEEL_NAME:
        _raise("IBAPI wheel filename is invalid")
    if not stat.S_ISREG(metadata.st_mode):
        _raise("IBAPI wheel is not a real file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_WHEEL_BYTES:
        _raise("IBAPI wheel size is invalid")
    contents: dict[str, bytes] = {}
    folded_names: set[str] = set()
    total = 0
    try:
        with zipfile.ZipFile(wheel) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_ENTRIES:
                _raise("IBAPI wheel entry count is invalid")
            for entry in entries:
                name = _canonical_member(entry.filename)
                folded = name.casefold()
                if name in contents or folded in folded_names:
                    _raise("IBAPI wheel contains duplicate member paths")
                folded_names.add(folded)
                if entry.is_dir() or entry.flag_bits & 0x1:
                    _raise("IBAPI wheel member type is invalid")
                unix_mode = entry.external_attr >> 16
                file_type = stat.S_IFMT(unix_mode)
                if file_type not in (0, stat.S_IFREG):
                    _raise("IBAPI wheel contains a non-regular member")
                if entry.file_size < 0 or entry.file_size > MAX_MEMBER_BYTES:
                    _raise("IBAPI wheel member size is invalid")
                total += entry.file_size
                if total > MAX_TOTAL_BYTES:
                    _raise("IBAPI wheel expanded size is too large")
                payload = archive.read(entry)
                if len(payload) != entry.file_size:
                    _raise("IBAPI wheel member size changed while reading")
                contents[name] = payload
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ArtifactVerificationError("IBAPI wheel archive is invalid") from error
    return contents


def _single_header(message: Message, name: str, label: str) -> str:
    values = message.get_all(name, [])
    if len(values) != 1:
        _raise(f"{label} {name} header is invalid")
    return str(values[0]).strip()


def _validate_metadata(contents: dict[str, bytes]) -> None:
    metadata_path = f"{DIST_INFO}/METADATA"
    wheel_path = f"{DIST_INFO}/WHEEL"
    metadata = BytesParser(policy=default).parsebytes(contents[metadata_path])
    if metadata.defects or metadata.is_multipart():
        _raise("wheel package metadata is malformed")
    if (
        _single_header(metadata, "Name", "wheel metadata") != "ibapi"
        or _single_header(metadata, "Version", "wheel metadata") != API_VERSION
    ):
        _raise("wheel package identity is invalid")
    requirements = [str(value).strip() for value in metadata.get_all("Requires-Dist", [])]
    if requirements != ["protobuf==5.29.5"]:
        _raise("wheel dependency declaration is invalid")

    wheel_metadata = BytesParser(policy=default).parsebytes(contents[wheel_path])
    if wheel_metadata.defects or wheel_metadata.is_multipart():
        _raise("wheel metadata is malformed")
    if _single_header(wheel_metadata, "Wheel-Version", "wheel") != "1.0":
        _raise("wheel format version is invalid")
    if _single_header(wheel_metadata, "Root-Is-Purelib", "wheel").lower() != "true":
        _raise("wheel is not pure Python")
    tags = [str(value).strip() for value in wheel_metadata.get_all("Tag", [])]
    if tags != ["py3-none-any"]:
        _raise("wheel compatibility tag is invalid")
    if contents[f"{DIST_INFO}/top_level.txt"] != b"ibapi\n":
        _raise("wheel top-level declaration is invalid")


def _expected_record_hash(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode("ascii")
    return f"sha256={encoded.rstrip('=')}"


def _validate_record(
    contents: dict[str, bytes],
    *,
    record_path: str,
    label: str,
) -> None:
    try:
        text = contents[record_path].decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (KeyError, UnicodeDecodeError, csv.Error) as error:
        raise ArtifactVerificationError(f"{label} RECORD is invalid") from error
    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            _raise(f"{label} RECORD row is invalid")
        path = _canonical_member(row[0])
        if path in records:
            _raise(f"{label} RECORD contains duplicate paths")
        records[path] = (row[1], row[2])
    if set(records) != set(contents):
        _raise(f"{label} RECORD coverage is incomplete")
    for path, payload in contents.items():
        digest, size = records[path]
        if path == record_path:
            if digest or size:
                _raise(f"{label} RECORD self row is invalid")
            continue
        if digest != _expected_record_hash(payload) or size != str(len(payload)):
            _raise(f"{label} RECORD hash or size is invalid")


def verify_wheel_artifact(
    wheel: Path,
    source_root: Path,
    *,
    provenance_link: Path = ACTIVE_PROVENANCE,
    expected_link: Path = ACTIVE_PROVENANCE,
    expected_target: Path = PROVENANCE_TARGET,
    trusted_root: Path = PROVENANCE_ROOT,
    trust_anchor: Path = PROVENANCE_TRUST_ANCHOR,
    required_uid: int = REQUIRED_OWNER_UID,
) -> WheelEvidence:
    """Validate the complete derived wheel before any installation."""

    provenance = verify_provenance_link(
        provenance_link,
        expected_link=expected_link,
        expected_target=expected_target,
        trusted_root=trusted_root,
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    source_contents, source_hash = _source_tree(source_root)
    if source_hash != provenance["source_tree_sha256"]:
        _raise("official source tree does not match active provenance")
    contents = _read_wheel(wheel)
    expected = set(source_contents) | set(WHEEL_METADATA_MEMBERS)
    if set(contents) != expected:
        _raise("IBAPI wheel contains non-official or missing members")
    for path, payload in source_contents.items():
        if contents[path] != payload:
            _raise("IBAPI wheel Python source differs from official source")
    _validate_metadata(contents)
    _validate_record(
        contents,
        record_path=f"{DIST_INFO}/RECORD",
        label="wheel",
    )
    return WheelEvidence(contents=contents, source_contents=source_contents)


def _locate_site_packages(venv: Path) -> Path:
    metadata = _lstat(venv, "V2 virtual environment")
    if not stat.S_ISDIR(metadata.st_mode):
        _raise("V2 virtual environment is not a real directory")
    lib = venv / "lib"
    lib_metadata = _lstat(lib, "V2 virtual environment lib")
    if not stat.S_ISDIR(lib_metadata.st_mode):
        _raise("V2 virtual environment lib is invalid")
    candidates = sorted(lib.glob("python*/site-packages"))
    if len(candidates) != 1:
        _raise("V2 site-packages path is missing or ambiguous")
    site_packages = candidates[0]
    for path in (site_packages.parent, site_packages):
        path_metadata = _lstat(path, "V2 site-packages boundary")
        if not stat.S_ISDIR(path_metadata.st_mode):
            _raise("V2 site-packages boundary is invalid")
    return site_packages


def _bounded_tree(root: Path, prefix: str) -> dict[str, bytes]:
    metadata = _lstat(root, f"installed {prefix}")
    if not stat.S_ISDIR(metadata.st_mode):
        _raise(f"installed {prefix} is not a real directory")
    contents: dict[str, bytes] = {}
    actual_directories: set[str] = set()
    total = 0
    visited = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        visited += 1 + len(directories) + len(filenames)
        if visited > MAX_ENTRIES:
            _raise(f"installed {prefix} tree has too many entries")
        relative_directory = current_path.relative_to(root).as_posix()
        if relative_directory != ".":
            actual_directories.add(f"{prefix}/{relative_directory}")
        for directory in directories:
            if (current_path / directory).is_symlink():
                _raise(f"installed {prefix} contains a symlink")
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                _raise(f"installed {prefix} contains a symlink")
            relative = path.relative_to(root).as_posix()
            name = _canonical_member(f"{prefix}/{relative}")
            payload = _read_regular_bounded(path, f"installed {prefix} member")
            total += len(payload)
            if total > MAX_TOTAL_BYTES or len(contents) >= MAX_ENTRIES:
                _raise(f"installed {prefix} tree is too large")
            contents[name] = payload
    expected_directories: set[str] = set()
    for member_name in contents:
        parent = PurePosixPath(member_name).parent
        while str(parent) != prefix:
            expected_directories.add(str(parent))
            parent = parent.parent
    if actual_directories != expected_directories:
        _raise(f"installed {prefix} contains unexpected directories")
    return contents


def _validate_uv_metadata(
    contents: dict[str, bytes],
    wheel: Path,
) -> None:
    if contents[f"{DIST_INFO}/INSTALLER"] != b"uv":
        _raise("installed distribution installer identity is invalid")
    if contents[f"{DIST_INFO}/REQUESTED"] != b"":
        _raise("installed distribution REQUESTED marker is invalid")
    direct_url = _json_without_duplicate_keys(
        contents[f"{DIST_INFO}/direct_url.json"],
        "installed direct_url",
    )
    if set(direct_url) != {"url", "archive_info"} or direct_url["archive_info"] != {}:
        _raise("installed direct_url metadata is invalid")
    if not isinstance(direct_url["url"], str):
        _raise("installed direct_url URL is invalid")
    parsed_url = urlparse(direct_url["url"])
    if parsed_url.scheme != "file" or parsed_url.netloc not in ("", "localhost"):
        _raise("installed direct_url is not a local wheel")
    direct_path = Path(unquote(parsed_url.path))
    try:
        if direct_path.resolve(strict=True) != wheel.resolve(strict=True):
            _raise("installed direct_url does not name the reviewed wheel")
    except OSError as error:
        raise ArtifactVerificationError("installed direct_url wheel is unavailable") from error

    uv_cache = _json_without_duplicate_keys(
        contents[f"{DIST_INFO}/uv_cache.json"],
        "installed uv_cache",
    )
    if set(uv_cache) != {"timestamp", "commit", "tags", "env", "directories"}:
        _raise("installed uv_cache metadata fields are invalid")
    timestamp = uv_cache["timestamp"]
    if not isinstance(timestamp, dict) or set(timestamp) != {
        "secs_since_epoch",
        "nanos_since_epoch",
    }:
        _raise("installed uv_cache timestamp is invalid")
    if not all(type(timestamp[key]) is int for key in timestamp):
        _raise("installed uv_cache timestamp values are invalid")
    if uv_cache["commit"] is not None or uv_cache["tags"] is not None:
        _raise("installed uv_cache source identity is invalid")
    if uv_cache["env"] != {} or uv_cache["directories"] != {}:
        _raise("installed uv_cache environment is invalid")


def verify_installed_distribution(
    wheel: Path,
    source_root: Path,
    venv: Path,
    *,
    provenance_link: Path = ACTIVE_PROVENANCE,
    expected_link: Path = ACTIVE_PROVENANCE,
    expected_target: Path = PROVENANCE_TARGET,
    trusted_root: Path = PROVENANCE_ROOT,
    trust_anchor: Path = PROVENANCE_TRUST_ANCHOR,
    required_uid: int = REQUIRED_OWNER_UID,
) -> None:
    """Validate the installed files before running anything from the new venv."""

    evidence = verify_wheel_artifact(
        wheel,
        source_root,
        provenance_link=provenance_link,
        expected_link=expected_link,
        expected_target=expected_target,
        trusted_root=trusted_root,
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    site_packages = _locate_site_packages(venv)
    package_contents = _bounded_tree(site_packages / "ibapi", "ibapi")
    dist_contents = _bounded_tree(site_packages / DIST_INFO, DIST_INFO)
    installed = package_contents | dist_contents
    expected_paths = (
        set(evidence.source_contents)
        | (set(WHEEL_METADATA_MEMBERS) - {f"{DIST_INFO}/RECORD"})
        | {f"{DIST_INFO}/RECORD"}
        | set(INSTALLER_METADATA_MEMBERS)
    )
    if set(installed) != expected_paths:
        _raise("installed IBAPI distribution contains extra or missing files")
    for path, payload in evidence.source_contents.items():
        if installed[path] != payload:
            _raise("installed IBAPI Python source differs from reviewed wheel")
    for path in WHEEL_METADATA_MEMBERS - {f"{DIST_INFO}/RECORD"}:
        if installed[path] != evidence.contents[path]:
            _raise("installed IBAPI metadata differs from reviewed wheel")
    _validate_uv_metadata(installed, wheel)
    _validate_record(
        installed,
        record_path=f"{DIST_INFO}/RECORD",
        label="installed distribution",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("preinstall", "postinstall"):
        command = subparsers.add_parser(operation)
        command.add_argument("--wheel", type=Path, required=True)
        command.add_argument("--official-source-root", type=Path, required=True)
        if operation == "postinstall":
            command.add_argument("--venv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.operation == "preinstall":
            verify_wheel_artifact(arguments.wheel, arguments.official_source_root)
        else:
            verify_installed_distribution(
                arguments.wheel,
                arguments.official_source_root,
                arguments.venv,
            )
    except Exception:
        print("release artifact verification failed", file=sys.stderr)
        return 78
    print(json.dumps({"operation": arguments.operation, "status": "ok"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
