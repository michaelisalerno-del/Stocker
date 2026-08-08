#!/usr/bin/python3
"""Prepare SQLite read coordination without following mutable pathnames."""

from __future__ import annotations

import errno
import grp
import os
import pwd
import stat
import sys
from typing import NoReturn

PERSISTENT_ROOT = "/var/lib/stocker"
DATABASE_DIRECTORY = "/var/lib/stocker/prospective"
DATABASE_DIRECTORY_NAME = "prospective"
BUNDLE_DIRECTORY_NAME = "bundles"
DATABASE_NAME = "prospective.sqlite3"
WAL_NAME = f"{DATABASE_NAME}-wal"
SHM_NAME = f"{DATABASE_NAME}-shm"
RECORDER_USER = "stocker"
WEB_USER = "stocker-web"
READER_GROUP = "stocker-readers"
BUNDLE_CONTROL_FILES = frozenset({"active.json", "operator-actions.jsonl"})


def fail(reason: str) -> NoReturn:
    print(
        f"blocked_unsafe_runtime_configuration: web_sqlite_boundary:{reason}",
        file=sys.stderr,
    )
    raise SystemExit(78)


def require_regular_file(
    descriptor: int,
    *,
    owner_uid: int,
    allowed_group_gids: frozenset[int],
    label: str,
) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        fail(f"{label}_not_regular")
    if metadata.st_nlink != 1:
        fail(f"{label}_unexpected_link_count")
    if metadata.st_uid != owner_uid or metadata.st_gid not in allowed_group_gids:
        fail(f"{label}_unexpected_owner")


def require_directory(
    descriptor: int,
    *,
    owner_uid: int,
    allowed_group_gids: frozenset[int],
    label: str,
) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        fail(f"{label}_not_directory")
    if metadata.st_uid != owner_uid or metadata.st_gid not in allowed_group_gids:
        fail(f"{label}_unexpected_owner")


def open_directory(
    parent_descriptor: int,
    name: str,
    *,
    directory_flags: int,
    label: str,
) -> int:
    try:
        return os.open(name, directory_flags, dir_fd=parent_descriptor)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            fail(f"{label}_symlink")
        if exc.errno == errno.ENOENT:
            fail(f"{label}_missing")
        fail(f"{label}_open_failed")


def open_existing(
    directory_descriptor: int,
    name: str,
    *,
    writable: bool,
    label: str,
) -> int:
    flags = os.O_CLOEXEC | os.O_NOFOLLOW
    flags |= os.O_RDWR if writable else os.O_RDONLY
    try:
        return os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            fail(f"{label}_symlink")
        if exc.errno == errno.ENOENT:
            fail(f"{label}_missing")
        fail(f"{label}_open_failed")


def open_or_create_auxiliary(
    directory_descriptor: int,
    name: str,
    *,
    mode: int,
    owner_uid: int,
    group_gid: int,
    allowed_group_gids: frozenset[int],
    label: str,
) -> int:
    flags = os.O_CLOEXEC | os.O_NOFOLLOW | os.O_RDWR
    while True:
        try:
            descriptor = os.open(name, flags, dir_fd=directory_descriptor)
            require_regular_file(
                descriptor,
                owner_uid=owner_uid,
                allowed_group_gids=allowed_group_gids,
                label=label,
            )
            if os.fstat(descriptor).st_gid != group_gid:
                os.fchown(descriptor, owner_uid, group_gid)
            return descriptor
        except FileNotFoundError:
            try:
                descriptor = os.open(
                    name,
                    flags | os.O_CREAT | os.O_EXCL,
                    mode,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError:
                continue
            except OSError:
                fail(f"{label}_create_failed")
            os.fchown(descriptor, owner_uid, group_gid)
            return descriptor
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                fail(f"{label}_symlink")
            fail(f"{label}_open_failed")


def migrate_installed_bundle_tree(
    directory_descriptor: int,
    *,
    owner_uid: int,
    allowed_group_gids: frozenset[int],
    reader_gid: int,
) -> None:
    for name in os.listdir(directory_descriptor):
        metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child_descriptor = open_directory(
                directory_descriptor,
                name,
                directory_flags=(os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC),
                label="installed_bundle_directory",
            )
            try:
                require_directory(
                    child_descriptor,
                    owner_uid=owner_uid,
                    allowed_group_gids=allowed_group_gids,
                    label="installed_bundle_directory",
                )
                os.fchown(child_descriptor, owner_uid, reader_gid)
                os.fchmod(child_descriptor, 0o550)
                migrate_installed_bundle_tree(
                    child_descriptor,
                    owner_uid=owner_uid,
                    allowed_group_gids=allowed_group_gids,
                    reader_gid=reader_gid,
                )
            finally:
                os.close(child_descriptor)
        elif stat.S_ISREG(metadata.st_mode):
            file_descriptor = open_existing(
                directory_descriptor,
                name,
                writable=False,
                label="installed_bundle_file",
            )
            try:
                require_regular_file(
                    file_descriptor,
                    owner_uid=owner_uid,
                    allowed_group_gids=allowed_group_gids,
                    label="installed_bundle_file",
                )
                os.fchown(file_descriptor, owner_uid, reader_gid)
                os.fchmod(file_descriptor, 0o440)
            finally:
                os.close(file_descriptor)
        else:
            fail("installed_bundle_unsupported_file_type")


def migrate_bundle_store(
    bundle_descriptor: int,
    *,
    directory_flags: int,
    owner_uid: int,
    allowed_group_gids: frozenset[int],
    reader_gid: int,
) -> None:
    entries = frozenset(os.listdir(bundle_descriptor))
    unexpected = entries - BUNDLE_CONTROL_FILES - {"installed"}
    if unexpected:
        fail("bundle_directory_unexpected_entry")

    if "installed" in entries:
        installed_descriptor = open_directory(
            bundle_descriptor,
            "installed",
            directory_flags=directory_flags,
            label="installed_bundle_root",
        )
        try:
            require_directory(
                installed_descriptor,
                owner_uid=owner_uid,
                allowed_group_gids=allowed_group_gids,
                label="installed_bundle_root",
            )
            os.fchown(installed_descriptor, owner_uid, reader_gid)
            os.fchmod(installed_descriptor, 0o2750)
            migrate_installed_bundle_tree(
                installed_descriptor,
                owner_uid=owner_uid,
                allowed_group_gids=allowed_group_gids,
                reader_gid=reader_gid,
            )
        finally:
            os.close(installed_descriptor)

    for name in BUNDLE_CONTROL_FILES & entries:
        file_descriptor = open_existing(
            bundle_descriptor,
            name,
            writable=False,
            label="bundle_control_file",
        )
        try:
            require_regular_file(
                file_descriptor,
                owner_uid=owner_uid,
                allowed_group_gids=allowed_group_gids,
                label="bundle_control_file",
            )
            os.fchown(file_descriptor, owner_uid, reader_gid)
            os.fchmod(file_descriptor, 0o640)
        finally:
            os.close(file_descriptor)


def main(*, migrate_existing: bool = False) -> None:
    if os.geteuid() != 0:
        fail("root_required")
    try:
        recorder_identity = pwd.getpwnam(RECORDER_USER)
        recorder_uid = recorder_identity.pw_uid
        recorder_gid = recorder_identity.pw_gid
        web_uid = pwd.getpwnam(WEB_USER).pw_uid
        reader_gid = grp.getgrnam(READER_GROUP).gr_gid
    except KeyError:
        fail("service_identity_missing")
    if recorder_uid == web_uid:
        fail("service_identities_must_differ")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        root_descriptor = os.open(PERSISTENT_ROOT, directory_flags)
    except OSError:
        fail("persistent_root_open_failed")
    try:
        root_metadata = os.fstat(root_descriptor)
        if migrate_existing:
            if root_metadata.st_uid not in {0, recorder_uid}:
                fail("persistent_root_unexpected_owner")
            if root_metadata.st_gid not in {recorder_gid, reader_gid}:
                fail("persistent_root_unexpected_owner")
            os.fchown(root_descriptor, 0, reader_gid)
            os.fchmod(root_descriptor, 0o750)
        else:
            if root_metadata.st_uid != 0 or root_metadata.st_gid != reader_gid:
                fail("persistent_root_unexpected_owner")
            if stat.S_IMODE(root_metadata.st_mode) != 0o750:
                fail("persistent_root_unexpected_mode")

        allowed_group_gids = (
            frozenset({recorder_gid, reader_gid}) if migrate_existing else frozenset({reader_gid})
        )
        if migrate_existing:
            bundle_descriptor = open_directory(
                root_descriptor,
                BUNDLE_DIRECTORY_NAME,
                directory_flags=directory_flags,
                label="bundle_directory",
            )
            try:
                require_directory(
                    bundle_descriptor,
                    owner_uid=recorder_uid,
                    allowed_group_gids=allowed_group_gids,
                    label="bundle_directory",
                )
                os.fchown(bundle_descriptor, recorder_uid, reader_gid)
                os.fchmod(bundle_descriptor, 0o2750)
                migrate_bundle_store(
                    bundle_descriptor,
                    directory_flags=directory_flags,
                    owner_uid=recorder_uid,
                    allowed_group_gids=allowed_group_gids,
                    reader_gid=reader_gid,
                )
            finally:
                os.close(bundle_descriptor)

        directory_descriptor = open_directory(
            root_descriptor,
            DATABASE_DIRECTORY_NAME,
            directory_flags=directory_flags,
            label="database_directory",
        )
        try:
            directory_metadata = os.fstat(directory_descriptor)
            if (
                directory_metadata.st_uid != recorder_uid
                or directory_metadata.st_gid not in allowed_group_gids
            ):
                fail("database_directory_unexpected_owner")
            if directory_metadata.st_gid != reader_gid:
                os.fchown(directory_descriptor, recorder_uid, reader_gid)
            os.fchmod(directory_descriptor, 0o2750)

            database_descriptor = open_existing(
                directory_descriptor,
                DATABASE_NAME,
                writable=False,
                label="database",
            )
            try:
                require_regular_file(
                    database_descriptor,
                    owner_uid=recorder_uid,
                    allowed_group_gids=allowed_group_gids,
                    label="database",
                )
                if os.fstat(database_descriptor).st_gid != reader_gid:
                    os.fchown(database_descriptor, recorder_uid, reader_gid)
                os.fchmod(database_descriptor, 0o640)
            finally:
                os.close(database_descriptor)

            wal_descriptor = open_or_create_auxiliary(
                directory_descriptor,
                WAL_NAME,
                mode=0o640,
                owner_uid=recorder_uid,
                group_gid=reader_gid,
                allowed_group_gids=allowed_group_gids,
                label="wal",
            )
            try:
                os.fchmod(wal_descriptor, 0o640)
            finally:
                os.close(wal_descriptor)

            shm_descriptor = open_or_create_auxiliary(
                directory_descriptor,
                SHM_NAME,
                mode=0o660,
                owner_uid=recorder_uid,
                group_gid=reader_gid,
                allowed_group_gids=allowed_group_gids,
                label="shm",
            )
            try:
                os.fchmod(shm_descriptor, 0o660)
            finally:
                os.close(shm_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        os.close(root_descriptor)

    print("web_sqlite_boundary:verified")


def run() -> None:
    arguments = sys.argv[1:]
    if arguments == []:
        migrate_existing = False
    elif arguments == ["--migrate-existing"]:
        migrate_existing = True
    else:
        fail("unsupported_arguments")
    try:
        main(migrate_existing=migrate_existing)
    except OSError:
        fail("filesystem_operation_failed")


if __name__ == "__main__":
    run()
