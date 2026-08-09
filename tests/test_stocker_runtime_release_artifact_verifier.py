from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import warnings
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[1]
VERIFIER_PATH = ROOT / "deploy/scripts/verify_v2_release_artifacts.py"
DIST_INFO = "ibapi-10.49.1.dist-info"


@pytest.fixture
def verifier() -> ModuleType:
    assert VERIFIER_PATH.is_file()
    specification = importlib.util.spec_from_file_location(
        "stocker_release_artifact_verifier",
        VERIFIER_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _record_hash(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return f"sha256={digest.decode('ascii').rstrip('=')}"


def _record_bytes(contents: dict[str, bytes], *, bad_path: str | None = None) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, payload in sorted(contents.items()):
        digest = "sha256=invalid" if name == bad_path else _record_hash(payload)
        writer.writerow((name, digest, str(len(payload))))
    writer.writerow((f"{DIST_INFO}/RECORD", "", ""))
    return output.getvalue().encode("utf-8")


def _source_files(source: Path) -> dict[str, bytes]:
    package = source / "ibapi"
    package.mkdir(parents=True)
    files = {
        "ibapi/__init__.py": b'__version__ = "10.49.1"\n',
        "ibapi/client.py": b"class EClient:\n    pass\n",
        "ibapi/messages/__init__.py": b"",
        "ibapi/messages/base.py": b"VALUE = 1\n",
    }
    for name, payload in files.items():
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return files


def _source_digest(source_files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(source_files.items()):
        relative = name.removeprefix("ibapi/")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _secure_provenance(
    tmp_path: Path,
    source_files: dict[str, bytes],
) -> tuple[Path, Path, Path]:
    trusted_root = tmp_path / "ibkr-api"
    provenance_directory = trusted_root / "provenance"
    provenance_directory.mkdir(parents=True)
    trusted_root.chmod(0o755)
    provenance_directory.chmod(0o755)
    target = provenance_directory / "10.49.1.json"
    tree_hash = _source_digest(source_files)
    target.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source": "interactive_brokers_official_tws_api",
                "release_channel": "latest",
                "platform": "mac_unix",
                "api_version": "10.49.1",
                "release_date": "2026-08-04",
                "official_page_url": "https://interactivebrokers.github.io/",
                "official_page_checked_at_utc": "2026-08-08T09:00:00Z",
                "source_url": (
                    "https://interactivebrokers.github.io/downloads/twsapi_macunix.1049.01.zip"
                ),
                "archive_filename": "twsapi_macunix.1049.01.zip",
                "archive_sha256": "a" * 64,
                "source_tree_sha256": tree_hash,
                "installed_tree_sha256": tree_hash,
                "registered_at_utc": "2026-08-08T09:05:00Z",
                "registered_by": "test-operator",
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    target.chmod(0o644)
    link = trusted_root / "active-provenance.json"
    link.symlink_to(target)
    return link, target, trusted_root


def _wheel_contents(source_files: dict[str, bytes]) -> dict[str, bytes]:
    return {
        **source_files,
        f"{DIST_INFO}/METADATA": (
            b"Metadata-Version: 2.4\n"
            b"Name: ibapi\n"
            b"Version: 10.49.1\n"
            b"Requires-Dist: protobuf==5.29.5\n"
            b"\n"
        ),
        f"{DIST_INFO}/WHEEL": (
            b"Wheel-Version: 1.0\n"
            b"Generator: reviewed-test-builder\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
            b"\n"
        ),
        f"{DIST_INFO}/top_level.txt": b"ibapi\n",
    }


def _write_wheel(
    wheel: Path,
    source_files: dict[str, bytes],
    *,
    extra: tuple[str, bytes] | None = None,
    bad_record_path: str | None = None,
    symlink_name: str | None = None,
    duplicate_name: str | None = None,
) -> None:
    contents = _wheel_contents(source_files)
    if extra is not None:
        contents[extra[0]] = extra[1]
    contents[f"{DIST_INFO}/RECORD"] = _record_bytes(
        contents,
        bad_path=bad_record_path,
    )
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in contents.items():
            archive.writestr(name, payload)
        if symlink_name is not None:
            info = zipfile.ZipInfo(symlink_name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "ibapi/client.py")
        if duplicate_name is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(duplicate_name, b"duplicate")


def _verify_wheel(
    verifier: ModuleType,
    wheel: Path,
    source: Path,
    link: Path,
    target: Path,
    trusted_root: Path,
) -> object:
    return verifier.verify_wheel_artifact(
        wheel,
        source,
        provenance_link=link,
        expected_link=link,
        expected_target=target,
        trusted_root=trusted_root,
        trust_anchor=trusted_root.parent,
        required_uid=os.getuid(),
    )


def _installed_fixture(venv: Path, wheel: Path) -> Path:
    site_packages = venv / "lib/python3.12/site-packages"
    site_packages.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name == f"{DIST_INFO}/RECORD":
                continue
            destination = site_packages / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    additions = {
        f"{DIST_INFO}/INSTALLER": b"uv",
        f"{DIST_INFO}/REQUESTED": b"",
        f"{DIST_INFO}/direct_url.json": json.dumps(
            {"url": wheel.resolve().as_uri(), "archive_info": {}},
            separators=(",", ":"),
        ).encode(),
        f"{DIST_INFO}/uv_cache.json": json.dumps(
            {
                "timestamp": {"secs_since_epoch": 1, "nanos_since_epoch": 2},
                "commit": None,
                "tags": None,
                "env": {},
                "directories": {},
            },
            separators=(",", ":"),
        ).encode(),
    }
    for name, payload in additions.items():
        destination = site_packages / name
        destination.write_bytes(payload)
    _rewrite_installed_record(site_packages)
    return site_packages


def _rewrite_installed_record(site_packages: Path) -> None:
    installed_contents = {
        path.relative_to(site_packages).as_posix(): path.read_bytes()
        for path in site_packages.rglob("*")
        if path.is_file() and path.name != "RECORD"
    }
    (site_packages / DIST_INFO / "RECORD").write_bytes(_record_bytes(installed_contents))


def test_valid_wheel_and_installed_distribution_are_accepted(
    tmp_path: Path,
    verifier: ModuleType,
) -> None:
    source = tmp_path / "pythonclient"
    source_files = _source_files(source)
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)
    wheel = tmp_path / "ibapi-10.49.1-py3-none-any.whl"
    _write_wheel(wheel, source_files)
    venv = tmp_path / "venv"
    _installed_fixture(venv, wheel)

    _verify_wheel(verifier, wheel, source, link, target, trusted_root)
    verifier.verify_installed_distribution(
        wheel,
        source,
        venv,
        provenance_link=link,
        expected_link=link,
        expected_target=target,
        trusted_root=trusted_root,
        trust_anchor=trusted_root.parent,
        required_uid=os.getuid(),
    )


@pytest.mark.parametrize(
    ("name", "payload"),
    (
        ("ibapi-loader.pth", b"import ibapi\n"),
        ("other_package/__init__.py", b""),
        ("ibapi/native.so", b"native"),
        ("ibapi/client.pyc", b"bytecode"),
        (f"{DIST_INFO}/entry_points.txt", b"[console_scripts]\nevil=ibapi.client:EClient\n"),
        ("../escape.py", b"escape"),
    ),
)
def test_wheel_rejects_non_official_members(
    tmp_path: Path,
    verifier: ModuleType,
    name: str,
    payload: bytes,
) -> None:
    source = tmp_path / "pythonclient"
    source_files = _source_files(source)
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)
    wheel = tmp_path / "ibapi-10.49.1-py3-none-any.whl"
    _write_wheel(wheel, source_files, extra=(name, payload))

    with pytest.raises(verifier.ArtifactVerificationError):
        _verify_wheel(verifier, wheel, source, link, target, trusted_root)


def test_wheel_rejects_symlink_duplicate_and_bad_record(
    tmp_path: Path,
    verifier: ModuleType,
) -> None:
    source = tmp_path / "pythonclient"
    source_files = _source_files(source)
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)
    cases = (
        {"symlink_name": "ibapi/link.py"},
        {"duplicate_name": "ibapi/client.py"},
        {"bad_record_path": "ibapi/client.py"},
    )
    for index, options in enumerate(cases):
        wheel = tmp_path / f"case-{index}" / "ibapi-10.49.1-py3-none-any.whl"
        wheel.parent.mkdir()
        _write_wheel(wheel, source_files, **options)
        with pytest.raises(verifier.ArtifactVerificationError):
            _verify_wheel(verifier, wheel, source, link, target, trusted_root)


def test_installed_distribution_rejects_tamper_and_extra(
    tmp_path: Path,
    verifier: ModuleType,
) -> None:
    source = tmp_path / "pythonclient"
    source_files = _source_files(source)
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)
    wheel = tmp_path / "ibapi-10.49.1-py3-none-any.whl"
    _write_wheel(wheel, source_files)

    tampered_venv = tmp_path / "tampered-venv"
    tampered_site = _installed_fixture(tampered_venv, wheel)
    (tampered_site / "ibapi/client.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_installed_distribution(
            wheel,
            source,
            tampered_venv,
            provenance_link=link,
            expected_link=link,
            expected_target=target,
            trusted_root=trusted_root,
            trust_anchor=trusted_root.parent,
            required_uid=os.getuid(),
        )

    extra_venv = tmp_path / "extra-venv"
    extra_site = _installed_fixture(extra_venv, wheel)
    (extra_site / "ibapi/extra.pyc").write_bytes(b"extra")
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_installed_distribution(
            wheel,
            source,
            extra_venv,
            provenance_link=link,
            expected_link=link,
            expected_target=target,
            trusted_root=trusted_root,
            trust_anchor=trusted_root.parent,
            required_uid=os.getuid(),
        )

    top_level_venv = tmp_path / "top-level-extra-venv"
    top_level_site = _installed_fixture(top_level_venv, wheel)
    (top_level_site / "ibapi-loader.pth").write_text("import ibapi\n", encoding="utf-8")
    _rewrite_installed_record(top_level_site)
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_installed_distribution(
            wheel,
            source,
            top_level_venv,
            provenance_link=link,
            expected_link=link,
            expected_target=target,
            trusted_root=trusted_root,
            trust_anchor=trusted_root.parent,
            required_uid=os.getuid(),
        )


def test_provenance_link_requires_exact_secure_root_owned_boundary(
    tmp_path: Path,
    verifier: ModuleType,
) -> None:
    source_files = _source_files(tmp_path / "pythonclient")
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)

    verifier.verify_provenance_link(
        link,
        expected_link=link,
        expected_target=target,
        trusted_root=trusted_root,
        trust_anchor=trusted_root.parent,
        required_uid=os.getuid(),
    )
    assert verifier.REQUIRED_OWNER_UID == 0
    assert Path("/var/lib/stocker/ibkr-api/active-provenance.json") == (verifier.ACTIVE_PROVENANCE)
    assert Path("/var/lib/stocker/ibkr-api/provenance/10.49.1.json") == (verifier.PROVENANCE_TARGET)

    wrong_link = trusted_root / "wrong.json"
    wrong_link.symlink_to(target)
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_provenance_link(
            wrong_link,
            expected_link=link,
            expected_target=target,
            trusted_root=trusted_root,
            trust_anchor=trusted_root.parent,
            required_uid=os.getuid(),
        )
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_provenance_link(
            link,
            expected_link=link,
            expected_target=target,
            trusted_root=trusted_root,
            trust_anchor=trusted_root.parent,
            required_uid=os.getuid() + 1,
        )


@pytest.mark.parametrize(
    "failure",
    ("escape", "dangling", "target", "parent", "root", "ancestor", "target_symlink"),
)
def test_provenance_link_rejects_unsafe_target_or_parent(
    tmp_path: Path,
    verifier: ModuleType,
    failure: str,
) -> None:
    source_files = _source_files(tmp_path / "pythonclient")
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)
    if failure == "escape":
        escaped = tmp_path / "escaped.json"
        escaped.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        link.unlink()
        link.symlink_to(escaped)
    elif failure == "dangling":
        link.unlink()
        link.symlink_to(target.with_name("missing.json"))
    elif failure == "target":
        target.chmod(0o666)
    elif failure == "parent":
        target.parent.chmod(0o777)
    elif failure == "root":
        trusted_root.chmod(0o777)
    elif failure == "ancestor":
        trusted_root.parent.chmod(0o777)
    else:
        real_target = target.with_name("real.json")
        target.rename(real_target)
        target.symlink_to(real_target)

    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_provenance_link(
            link,
            expected_link=link,
            expected_target=target,
            trusted_root=trusted_root,
            trust_anchor=trusted_root.parent,
            required_uid=os.getuid(),
        )


def test_cli_failure_output_is_bounded_and_secret_free(
    tmp_path: Path,
    verifier: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("/secret/operator/path")

    monkeypatch.setattr(verifier, "verify_wheel_artifact", fail)
    status = verifier.main(
        [
            "preinstall",
            "--wheel",
            str(tmp_path / "wheel"),
            "--official-source-root",
            str(tmp_path / "source"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 78
    assert captured.out == ""
    assert captured.err == "release artifact verification failed\n"
    assert "/secret" not in captured.err
