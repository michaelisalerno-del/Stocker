#!/usr/bin/python3
"""Prepare the V2 SQLite reader coordination files without following symlinks."""

from __future__ import annotations

import errno
import grp
import os
import pwd
import stat
import sys
from typing import NoReturn

PERSISTENT_ROOT = "/var/lib/stocker"
DATABASE_DIRECTORY_NAME = "v2"
BACKUP_DIRECTORY_NAME = "backups-v2"
DATABASE_NAME = "stocker-v2.sqlite3"
WAL_NAME = f"{DATABASE_NAME}-wal"
SHM_NAME = f"{DATABASE_NAME}-shm"
RECORDER_USER = "stocker-recorder"
WEB_USER = "stocker-web"
BACKUP_USER = "stocker-backup"
READER_GROUP = "stocker-readers"
AUXILIARY_RACE_ATTEMPTS = 3


def fail(reason: str) -> NoReturn:
    print(f"blocked_unsafe_runtime_configuration:v2_sqlite_boundary:{reason}", file=sys.stderr)
    raise SystemExit(78)


def _open_directory(parent: int | None, name: str, *, label: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(name, flags) if parent is None else os.open(name, flags, dir_fd=parent)
    except OSError as error:
        if error.errno == errno.ELOOP:
            fail(f"{label}_symlink")
        if error.errno == errno.ENOENT:
            fail(f"{label}_missing")
        fail(f"{label}_open_failed")


def _require_directory(
    descriptor: int,
    *,
    owner_uid: int,
    group_gid: int,
    mode: int,
    label: str,
) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        fail(f"{label}_not_directory")
    if metadata.st_uid != owner_uid or metadata.st_gid != group_gid:
        fail(f"{label}_unexpected_owner")
    _set_mode_if_needed(descriptor, metadata=metadata, mode=mode, label=label)


def _set_mode_if_needed(
    descriptor: int,
    *,
    metadata: os.stat_result,
    mode: int,
    label: str,
) -> None:
    if stat.S_IMODE(metadata.st_mode) == mode:
        return
    try:
        os.fchmod(descriptor, mode)
    except OSError:
        fail(f"{label}_mode_update_failed")


def _set_owner(
    descriptor: int,
    *,
    owner_uid: int,
    group_gid: int,
    label: str,
) -> None:
    try:
        os.fchown(descriptor, owner_uid, group_gid)
    except OSError:
        fail(f"{label}_owner_update_failed")


def _open_regular(directory: int, name: str, *, writable: bool, label: str) -> int:
    flags = os.O_CLOEXEC | os.O_NOFOLLOW | (os.O_RDWR if writable else os.O_RDONLY)
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        if error.errno == errno.ELOOP:
            fail(f"{label}_symlink")
        if error.errno == errno.ENOENT:
            fail(f"{label}_missing")
        fail(f"{label}_open_failed")
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        fail(f"{label}_not_single_regular_file")
    return descriptor


def _prepare_auxiliary(
    directory: int,
    name: str,
    *,
    owner_uid: int,
    group_gid: int,
    mode: int,
    label: str,
) -> None:
    flags = os.O_CLOEXEC | os.O_NOFOLLOW | os.O_RDWR
    for _attempt in range(AUXILIARY_RACE_ATTEMPTS):
        try:
            descriptor = os.open(name, flags, dir_fd=directory)
            created = False
        except FileNotFoundError:
            try:
                descriptor = os.open(
                    name,
                    flags | os.O_CREAT | os.O_EXCL,
                    mode,
                    dir_fd=directory,
                )
                created = True
            except FileExistsError:
                continue
            except OSError:
                fail(f"{label}_create_failed")
        except OSError as error:
            if error.errno == errno.ELOOP:
                fail(f"{label}_symlink")
            fail(f"{label}_open_failed")
        try:
            if created:
                _set_owner(
                    descriptor,
                    owner_uid=owner_uid,
                    group_gid=group_gid,
                    label=label,
                )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink > 1:
                fail(f"{label}_not_single_regular_file")
            if metadata.st_nlink == 0:
                continue
            if metadata.st_uid != owner_uid or metadata.st_gid != group_gid:
                fail(f"{label}_unexpected_owner")
            _set_mode_if_needed(
                descriptor,
                metadata=metadata,
                mode=mode,
                label=label,
            )
            metadata = os.fstat(descriptor)
            if metadata.st_nlink == 0:
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                fail(f"{label}_not_single_regular_file")
            return
        finally:
            os.close(descriptor)
    fail(f"{label}_race_exhausted")


def main() -> None:
    if os.geteuid() != 0:
        fail("root_required")
    try:
        recorder_uid = pwd.getpwnam(RECORDER_USER).pw_uid
        web_uid = pwd.getpwnam(WEB_USER).pw_uid
        backup_uid = pwd.getpwnam(BACKUP_USER).pw_uid
        reader_gid = grp.getgrnam(READER_GROUP).gr_gid
    except KeyError:
        fail("service_identity_missing")
    if len({recorder_uid, web_uid, backup_uid}) != 3:
        fail("service_identities_must_differ")

    root = _open_directory(None, PERSISTENT_ROOT, label="persistent_root")
    try:
        _require_directory(
            root,
            owner_uid=0,
            group_gid=reader_gid,
            mode=0o750,
            label="persistent_root",
        )
        database_directory = _open_directory(
            root, DATABASE_DIRECTORY_NAME, label="database_directory"
        )
        try:
            _require_directory(
                database_directory,
                owner_uid=recorder_uid,
                group_gid=reader_gid,
                mode=0o2750,
                label="database_directory",
            )
            database = _open_regular(
                database_directory, DATABASE_NAME, writable=False, label="database"
            )
            try:
                metadata = os.fstat(database)
                if metadata.st_uid != recorder_uid or metadata.st_gid != reader_gid:
                    fail("database_unexpected_owner")
                _set_mode_if_needed(
                    database,
                    metadata=metadata,
                    mode=0o640,
                    label="database",
                )
            finally:
                os.close(database)
            _prepare_auxiliary(
                database_directory,
                WAL_NAME,
                owner_uid=recorder_uid,
                group_gid=reader_gid,
                mode=0o640,
                label="wal",
            )
            _prepare_auxiliary(
                database_directory,
                SHM_NAME,
                owner_uid=recorder_uid,
                group_gid=reader_gid,
                mode=0o660,
                label="shm",
            )
        finally:
            os.close(database_directory)

        backup_directory = _open_directory(root, BACKUP_DIRECTORY_NAME, label="backup_directory")
        try:
            _require_directory(
                backup_directory,
                owner_uid=backup_uid,
                group_gid=reader_gid,
                mode=0o2750,
                label="backup_directory",
            )
        finally:
            os.close(backup_directory)
    finally:
        os.close(root)
    print("v2_sqlite_boundary:verified")


def run() -> None:
    if sys.argv[1:]:
        fail("unsupported_arguments")
    try:
        main()
    except OSError:
        fail("filesystem_operation_failed")


if __name__ == "__main__":
    run()
