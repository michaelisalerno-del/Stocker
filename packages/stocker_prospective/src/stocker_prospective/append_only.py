"""Small durable primitives for immutable prospective evidence files."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path


def write_immutable_bytes(
    path: str | Path,
    content: bytes,
    *,
    conflict_message: str,
    before_link: Callable[[], None] | None = None,
) -> bool:
    """Link fully-written bytes once, accepting only an identical existing file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    created = False
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if before_link is not None:
            before_link()
        try:
            os.link(temporary, destination)
            created = True
        except FileExistsError:
            if (
                not destination.is_file()
                or destination.is_symlink()
                or destination.read_bytes() != content
            ):
                raise ValueError(conflict_message) from None
        if created:
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return created


def write_immutable_json(
    path: str | Path,
    payload: Mapping[str, object],
    *,
    conflict_message: str,
    before_link: Callable[[], None] | None = None,
) -> bool:
    """Canonical pretty JSON wrapper around :func:`write_immutable_bytes`."""

    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return write_immutable_bytes(
        path,
        content,
        conflict_message=conflict_message,
        before_link=before_link,
    )


__all__ = ["write_immutable_bytes", "write_immutable_json"]
