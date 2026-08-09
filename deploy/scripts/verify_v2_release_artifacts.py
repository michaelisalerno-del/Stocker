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
import re
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
PROVENANCE_LITERAL_TARGET = "provenance/10.49.1.json"
PROVENANCE_ROOT = Path("/var/lib/stocker/ibkr-api")
PROVENANCE_TRUST_ANCHOR = Path("/")
REQUIRED_OWNER_UID = 0
UV_PATH = Path("/usr/local/bin/uv")
UV_SHA256 = "da15297d6879b2cfbe5ea3cb03725c1613d51ba72892cc996468d871f0a532fb"
MAX_WHEEL_BYTES = 64 * 1024 * 1024
MAX_EXECUTABLE_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_ENTRIES = 2_048
MAX_PROTECTED_ENTRIES = 100_000
MAX_PATH_BYTES = 512
ROOT_GROUP_GID = 0
POSIX_ACL_XATTRS = frozenset({"system.posix_acl_access", "system.posix_acl_default"})
VERIFIER_RELATIVE_PATH = Path("deploy/scripts/verify_v2_release_artifacts.py")
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
PROTOBUF_REQUIREMENT = re.compile(r"protobuf[ \t]*==[ \t]*5\.29\.5", flags=re.ASCII)


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


def _require_root_control(metadata: os.stat_result, label: str, required_uid: int) -> None:
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid != required_uid:
        _raise(f"{label} owner is invalid")
    if mode & stat.S_IWOTH:
        _raise(f"{label} is world writable")
    if mode & stat.S_IWGRP and metadata.st_gid != ROOT_GROUP_GID:
        _raise(f"{label} is writable by a non-root group")


def _require_no_extended_posix_acl(path: Path, label: str) -> None:
    listxattr: Any = getattr(os, "listxattr", None)
    if listxattr is None:
        _raise(f"{label} ACL inspection is unavailable")
    try:
        names = listxattr(path, follow_symlinks=False)
    except (OSError, TypeError, NotImplementedError) as error:
        raise ArtifactVerificationError(f"{label} ACL inspection failed") from error
    normalized = {
        name.decode("ascii", errors="strict") if isinstance(name, bytes) else name for name in names
    }
    if normalized & POSIX_ACL_XATTRS:
        _raise(f"{label} has an extended POSIX ACL")


def _require_secure_directory(path: Path, label: str, required_uid: int) -> None:
    metadata = _lstat(path, label)
    if not stat.S_ISDIR(metadata.st_mode):
        _raise(f"{label} is not a real directory")
    _require_root_control(metadata, label, required_uid)
    _require_no_extended_posix_acl(path, label)


def _require_secure_regular(path: Path, label: str, required_uid: int) -> os.stat_result:
    metadata = _lstat(path, label)
    if not stat.S_ISREG(metadata.st_mode):
        _raise(f"{label} is not a real file")
    _require_root_control(metadata, label, required_uid)
    _require_no_extended_posix_acl(path, label)
    return metadata


def _require_canonical_absolute(path: Path, label: str) -> None:
    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        _raise(f"{label} path is not canonical and absolute")


def _require_secure_ancestry(path: Path, anchor: Path, required_uid: int) -> None:
    _require_canonical_absolute(path, "protected")
    _require_canonical_absolute(anchor, "trust anchor")
    try:
        relative = path.relative_to(anchor)
    except ValueError as error:
        raise ArtifactVerificationError("provenance path escapes its trust anchor") from error
    current = anchor
    _require_secure_directory(current, "provenance parent", required_uid)
    for part in relative.parts:
        current /= part
        _require_secure_directory(current, "provenance parent", required_uid)


def _require_secure_file_boundary(
    path: Path,
    label: str,
    *,
    trust_anchor: Path,
    required_uid: int,
) -> os.stat_result:
    _require_canonical_absolute(path, label)
    _require_secure_ancestry(path.parent, trust_anchor, required_uid)
    return _require_secure_regular(path, label, required_uid)


def _reject_walk_error(error: OSError) -> None:
    raise ArtifactVerificationError("protected tree is unreadable") from error


def _require_secure_tree_entry(
    path: Path,
    label: str,
    *,
    trust_anchor: Path,
    required_uid: int,
) -> None:
    metadata = _lstat(path, label)
    if stat.S_ISLNK(metadata.st_mode):
        if metadata.st_uid != required_uid:
            _raise(f"{label} symlink owner is invalid")
        _require_no_extended_posix_acl(path, label)
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ArtifactVerificationError(f"{label} symlink is dangling") from error
        _require_secure_ancestry(resolved.parent, trust_anchor, required_uid)
        target_metadata = _lstat(resolved, f"{label} symlink target")
        if not (stat.S_ISREG(target_metadata.st_mode) or stat.S_ISDIR(target_metadata.st_mode)):
            _raise(f"{label} symlink target type is invalid")
        _require_root_control(target_metadata, f"{label} symlink target", required_uid)
        _require_no_extended_posix_acl(resolved, f"{label} symlink target")
    elif stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode):
        _require_root_control(metadata, label, required_uid)
        _require_no_extended_posix_acl(path, label)
    else:
        _raise(f"{label} type is invalid")


def _require_exact_lib64_symlink(
    path: Path,
    root: Path,
    label: str,
    required_uid: int,
) -> None:
    if path != root / ".venv/lib64":
        _raise(f"{label} directory symlink is not admitted")
    metadata = _lstat(path, label)
    if not stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != required_uid:
        _raise(f"{label} directory symlink identity is invalid")
    _require_no_extended_posix_acl(path, label)
    try:
        literal = os.readlink(path)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ArtifactVerificationError(f"{label} directory symlink is invalid") from error
    expected = root / ".venv/lib"
    if literal != "lib" or resolved != expected:
        _raise(f"{label} directory symlink target is invalid")
    _require_secure_directory(expected, f"{label} directory symlink target", required_uid)


def _require_secure_tree(
    root: Path,
    label: str,
    *,
    trust_anchor: Path,
    required_uid: int,
) -> None:
    visited = 0
    for current, directories, filenames in os.walk(
        root,
        followlinks=False,
        onerror=_reject_walk_error,
    ):
        current_path = Path(current)
        _require_secure_directory(current_path, f"{label} directory", required_uid)
        visited += 1 + len(directories) + len(filenames)
        if visited > MAX_PROTECTED_ENTRIES:
            _raise(f"{label} has too many entries")
        for directory in directories:
            directory_path = current_path / directory
            directory_metadata = _lstat(directory_path, f"{label} directory")
            if stat.S_ISLNK(directory_metadata.st_mode):
                _require_exact_lib64_symlink(
                    directory_path,
                    root,
                    f"{label} directory",
                    required_uid,
                )
            else:
                _require_secure_tree_entry(
                    directory_path,
                    f"{label} directory",
                    trust_anchor=trust_anchor,
                    required_uid=required_uid,
                )
        for filename in filenames:
            _require_secure_tree_entry(
                current_path / filename,
                f"{label} member",
                trust_anchor=trust_anchor,
                required_uid=required_uid,
            )


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
    _require_no_extended_posix_acl(provenance_link, "active provenance link")
    try:
        literal_target = os.readlink(provenance_link)
    except OSError as error:
        raise ArtifactVerificationError("active provenance link is unreadable") from error
    if literal_target != PROVENANCE_LITERAL_TARGET:
        _raise("active provenance link literal target is invalid")

    target_metadata = _lstat(expected_target, "active provenance target")
    if not stat.S_ISREG(target_metadata.st_mode):
        _raise("active provenance target is not a real file")
    _require_root_control(target_metadata, "active provenance target", required_uid)
    _require_no_extended_posix_acl(expected_target, "active provenance target")
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


def _source_tree(
    source_root: Path,
    *,
    trust_anchor: Path,
    required_uid: int,
) -> tuple[dict[str, bytes], str]:
    _require_secure_ancestry(source_root, trust_anchor, required_uid)
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
    for current, directories, filenames in os.walk(
        source_root,
        followlinks=False,
        onerror=_reject_walk_error,
    ):
        current_path = Path(current)
        _require_secure_directory(current_path, "official source directory", required_uid)
        visited += 1 + len(directories) + len(filenames)
        if visited > MAX_ENTRIES:
            _raise("official source tree has too many entries")
        for directory in directories:
            directory_path = current_path / directory
            _require_secure_directory(
                directory_path,
                "official source directory",
                required_uid,
            )
        for filename in filenames:
            path = current_path / filename
            metadata = _require_secure_regular(path, "official source member", required_uid)
            relative_source = path.relative_to(source_root).as_posix()
            _canonical_member(relative_source)
            total += metadata.st_size
            if total > MAX_TOTAL_BYTES:
                _raise("official source tree is too large")
            relative_path = PurePosixPath(relative_source)
            if not relative_path.parts or relative_path.parts[0] != "ibapi" or path.suffix != ".py":
                continue
            relative = path.relative_to(package_root).as_posix()
            payload = _read_regular_bounded(path, "official ibapi source member")
            if len(contents) >= MAX_ENTRIES:
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
    if len(requirements) != 1 or PROTOBUF_REQUIREMENT.fullmatch(requirements[0]) is None:
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
    _require_secure_file_boundary(
        wheel,
        "IBAPI wheel",
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    source_contents, source_hash = _source_tree(
        source_root,
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
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


def _verify_sha256_manifest(
    artifact: Path,
    manifest: Path,
    label: str,
    *,
    trust_anchor: Path,
    required_uid: int,
    artifact_limit: int = MAX_WHEEL_BYTES,
    expected_sha256: str | None = None,
    require_executable: bool = False,
) -> None:
    artifact_metadata = _require_secure_file_boundary(
        artifact,
        label,
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    _require_secure_file_boundary(
        manifest,
        f"{label} manifest",
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    artifact_payload = _read_regular_bounded(
        artifact,
        label,
        limit=artifact_limit,
    )
    manifest_payload = _read_regular_bounded(
        manifest,
        f"{label} manifest",
        limit=MAX_METADATA_BYTES,
    )
    digest = hashlib.sha256(artifact_payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        _raise(f"{label} hash is not the reviewed value")
    if require_executable and not stat.S_IMODE(artifact_metadata.st_mode) & 0o111:
        _raise(f"{label} is not executable")
    try:
        expected = f"{digest}  {artifact}\n".encode("ascii")
    except UnicodeEncodeError as error:
        raise ArtifactVerificationError(f"{label} path is not ASCII") from error
    if manifest_payload != expected:
        _raise(f"{label} manifest is not the exact one-entry hash binding")


def _verify_exact_sha256(path: Path, label: str, expected: str) -> None:
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or hashlib.sha256(
            _read_regular_bounded(path, label, limit=MAX_EXECUTABLE_BYTES)
        ).hexdigest()
        != expected
    ):
        _raise(f"{label} hash is not the reviewed value")


def verify_release_boundary(
    *,
    ibapi_wheel: Path,
    ibapi_manifest: Path,
    protobuf_wheel: Path,
    protobuf_manifest: Path,
    uv_bin: Path,
    uv_manifest: Path,
    source_root: Path,
    release_root: Path,
    venv: Path,
    verifier_path: Path,
    expected_verifier_sha256: str,
    provenance_link: Path = ACTIVE_PROVENANCE,
    expected_link: Path = ACTIVE_PROVENANCE,
    expected_target: Path = PROVENANCE_TARGET,
    provenance_root: Path = PROVENANCE_ROOT,
    trust_anchor: Path = PROVENANCE_TRUST_ANCHOR,
    required_uid: int = REQUIRED_OWNER_UID,
    expected_uv: Path = UV_PATH,
    expected_uv_sha256: str = UV_SHA256,
) -> WheelEvidence:
    """Bind immutable inputs and root-controlled execution paths for one phase."""

    if verifier_path != release_root / VERIFIER_RELATIVE_PATH:
        _raise("release verifier path is not exact")
    if venv != release_root / ".venv":
        _raise("V2 virtual environment path is not exact")
    if uv_bin != expected_uv:
        _raise("uv executable path is not exact")
    if (
        len(
            {
                ibapi_wheel,
                ibapi_manifest,
                protobuf_wheel,
                protobuf_manifest,
                uv_bin,
                uv_manifest,
            }
        )
        != 6
    ):
        _raise("release artifact paths are not distinct")
    if not (
        protobuf_wheel.name.startswith("protobuf-5.29.5-") and protobuf_wheel.name.endswith(".whl")
    ):
        _raise("protobuf wheel filename is invalid")

    _require_secure_ancestry(release_root, trust_anchor, required_uid)
    _require_secure_file_boundary(
        verifier_path,
        "release artifact verifier",
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    _verify_exact_sha256(
        verifier_path,
        "release artifact verifier",
        expected_verifier_sha256,
    )
    _locate_site_packages(
        venv,
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    _require_secure_tree(
        release_root,
        "V2 release",
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    _verify_sha256_manifest(
        uv_bin,
        uv_manifest,
        "uv executable",
        trust_anchor=trust_anchor,
        required_uid=required_uid,
        artifact_limit=MAX_EXECUTABLE_BYTES,
        expected_sha256=expected_uv_sha256,
        require_executable=True,
    )
    _verify_sha256_manifest(
        protobuf_wheel,
        protobuf_manifest,
        "protobuf wheel",
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    _verify_sha256_manifest(
        ibapi_wheel,
        ibapi_manifest,
        "IBAPI wheel",
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    return verify_wheel_artifact(
        ibapi_wheel,
        source_root,
        provenance_link=provenance_link,
        expected_link=expected_link,
        expected_target=expected_target,
        trusted_root=provenance_root,
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )


def _locate_site_packages(
    venv: Path,
    *,
    trust_anchor: Path,
    required_uid: int,
) -> Path:
    _require_secure_ancestry(venv, trust_anchor, required_uid)
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
        _require_secure_directory(path, "V2 site-packages boundary", required_uid)
    return site_packages


def _bounded_tree(
    root: Path,
    prefix: str,
    *,
    required_uid: int,
) -> dict[str, bytes]:
    metadata = _lstat(root, f"installed {prefix}")
    if not stat.S_ISDIR(metadata.st_mode):
        _raise(f"installed {prefix} is not a real directory")
    _require_root_control(metadata, f"installed {prefix}", required_uid)
    contents: dict[str, bytes] = {}
    actual_directories: set[str] = set()
    total = 0
    visited = 0
    for current, directories, filenames in os.walk(
        root,
        followlinks=False,
        onerror=_reject_walk_error,
    ):
        current_path = Path(current)
        _require_secure_directory(
            current_path,
            f"installed {prefix} directory",
            required_uid,
        )
        visited += 1 + len(directories) + len(filenames)
        if visited > MAX_ENTRIES:
            _raise(f"installed {prefix} tree has too many entries")
        relative_directory = current_path.relative_to(root).as_posix()
        if relative_directory != ".":
            actual_directories.add(f"{prefix}/{relative_directory}")
        for directory in directories:
            _require_secure_directory(
                current_path / directory,
                f"installed {prefix} directory",
                required_uid,
            )
        for filename in filenames:
            path = current_path / filename
            _require_secure_regular(
                path,
                f"installed {prefix} member",
                required_uid,
            )
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
    site_packages = _locate_site_packages(
        venv,
        trust_anchor=trust_anchor,
        required_uid=required_uid,
    )
    package_contents = _bounded_tree(
        site_packages / "ibapi",
        "ibapi",
        required_uid=required_uid,
    )
    dist_contents = _bounded_tree(
        site_packages / DIST_INFO,
        DIST_INFO,
        required_uid=required_uid,
    )
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
        command.add_argument("--ibapi-wheel", type=Path, required=True)
        command.add_argument("--ibapi-manifest", type=Path, required=True)
        command.add_argument("--protobuf-wheel", type=Path, required=True)
        command.add_argument("--protobuf-manifest", type=Path, required=True)
        command.add_argument("--uv-bin", type=Path, required=True)
        command.add_argument("--uv-manifest", type=Path, required=True)
        command.add_argument("--official-source-root", type=Path, required=True)
        command.add_argument("--release-root", type=Path, required=True)
        command.add_argument("--verifier-path", type=Path, required=True)
        command.add_argument("--verifier-sha256", required=True)
        command.add_argument("--venv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if Path(__file__) != arguments.verifier_path:
            _raise("running release verifier path is not exact")
        verify_release_boundary(
            ibapi_wheel=arguments.ibapi_wheel,
            ibapi_manifest=arguments.ibapi_manifest,
            protobuf_wheel=arguments.protobuf_wheel,
            protobuf_manifest=arguments.protobuf_manifest,
            uv_bin=arguments.uv_bin,
            uv_manifest=arguments.uv_manifest,
            source_root=arguments.official_source_root,
            release_root=arguments.release_root,
            venv=arguments.venv,
            verifier_path=arguments.verifier_path,
            expected_verifier_sha256=arguments.verifier_sha256,
        )
        if arguments.operation == "postinstall":
            verify_installed_distribution(
                arguments.ibapi_wheel,
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
