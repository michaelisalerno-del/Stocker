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
DATABASE_NAME = "prospective.sqlite3"
WAL_NAME = f"{DATABASE_NAME}-wal"
SHM_NAME = f"{DATABASE_NAME}-shm"
RECORDER_USER = "stocker"
WEB_USER = "stocker-web"
READER_GROUP = "stocker-readers"


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
    group_gid: int,
    label: str,
) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        fail(f"{label}_not_regular")
    if metadata.st_nlink != 1:
        fail(f"{label}_unexpected_link_count")
    if metadata.st_uid != owner_uid or metadata.st_gid != group_gid:
        fail(f"{label}_unexpected_owner")


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
    label: str,
) -> int:
    flags = os.O_CLOEXEC | os.O_NOFOLLOW | os.O_RDWR
    while True:
        try:
            descriptor = os.open(name, flags, dir_fd=directory_descriptor)
            require_regular_file(
                descriptor,
                owner_uid=owner_uid,
                group_gid=group_gid,
                label=label,
            )
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


def main() -> None:
    if os.geteuid() != 0:
        fail("root_required")
    try:
        recorder_uid = pwd.getpwnam(RECORDER_USER).pw_uid
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
        if root_metadata.st_uid != 0 or root_metadata.st_gid != reader_gid:
            fail("persistent_root_unexpected_owner")
        if stat.S_IMODE(root_metadata.st_mode) != 0o750:
            fail("persistent_root_unexpected_mode")

        try:
            directory_descriptor = os.open(
                DATABASE_DIRECTORY_NAME,
                directory_flags,
                dir_fd=root_descriptor,
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                fail("database_directory_symlink")
            fail("database_directory_open_failed")
        try:
            directory_metadata = os.fstat(directory_descriptor)
            if directory_metadata.st_uid != recorder_uid or directory_metadata.st_gid != reader_gid:
                fail("database_directory_unexpected_owner")
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
                    group_gid=reader_gid,
                    label="database",
                )
                os.fchmod(database_descriptor, 0o640)
            finally:
                os.close(database_descriptor)

            wal_descriptor = open_or_create_auxiliary(
                directory_descriptor,
                WAL_NAME,
                mode=0o640,
                owner_uid=recorder_uid,
                group_gid=reader_gid,
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
    try:
        main()
    except OSError:
        fail("filesystem_operation_failed")


if __name__ == "__main__":
    run()
