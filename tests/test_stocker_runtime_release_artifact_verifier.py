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
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
VERIFIER_PATH = ROOT / "deploy/scripts/verify_v2_release_artifacts.py"
DIST_INFO = "ibapi-10.49.1.dist-info"


@pytest.fixture
def verifier(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
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
    if not hasattr(module.os, "listxattr"):
        monkeypatch.setattr(
            module.os,
            "listxattr",
            lambda path, *, follow_symlinks=True: [],
            raising=False,
        )
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
    link.symlink_to(Path("provenance/10.49.1.json"))
    return link, target, trusted_root


def _wheel_contents(
    source_files: dict[str, bytes],
    *,
    requirement: str = "protobuf==5.29.5",
    top_level: bytes = b"ibapi\n",
) -> dict[str, bytes]:
    return {
        **source_files,
        f"{DIST_INFO}/METADATA": (
            "Metadata-Version: 2.4\n"
            "Name: ibapi\n"
            "Version: 10.49.1\n"
            f"Requires-Dist: {requirement}\n"
            "\n"
        ).encode("ascii"),
        f"{DIST_INFO}/WHEEL": (
            b"Wheel-Version: 1.0\n"
            b"Generator: reviewed-test-builder\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
            b"\n"
        ),
        f"{DIST_INFO}/top_level.txt": top_level,
    }


def _write_wheel(
    wheel: Path,
    source_files: dict[str, bytes],
    *,
    extra: tuple[str, bytes] | None = None,
    bad_record_path: str | None = None,
    symlink_name: str | None = None,
    duplicate_name: str | None = None,
    requirement: str = "protobuf==5.29.5",
    top_level: bytes = b"ibapi\n",
) -> None:
    contents = _wheel_contents(
        source_files,
        requirement=requirement,
        top_level=top_level,
    )
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


def _write_sha256_manifest(manifest: Path, artifact: Path) -> None:
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest.write_text(f"{digest}  {artifact}\n", encoding="ascii")


def _release_boundary_fixture(tmp_path: Path) -> dict[str, Path | int | str]:
    install_root = tmp_path / "install"
    source = install_root / "IBJts/source/pythonclient"
    source_files = _source_files(source)
    (source / "README.txt").write_text("reviewed source tree\n", encoding="utf-8")
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)

    ibapi_wheel = install_root / "ibapi-10.49.1-py3-none-any.whl"
    _write_wheel(ibapi_wheel, source_files)
    ibapi_manifest = install_root / "ibapi-10.49.1-wheel.sha256"
    _write_sha256_manifest(ibapi_manifest, ibapi_wheel)

    protobuf_wheel = install_root / "protobuf-5.29.5-py3-none-any.whl"
    protobuf_wheel.write_bytes(b"reviewed protobuf wheel")
    protobuf_manifest = install_root / "protobuf-5.29.5-wheel.sha256"
    _write_sha256_manifest(protobuf_manifest, protobuf_wheel)

    uv_bin = tmp_path / "usr/local/bin/uv"
    uv_bin.parent.mkdir(parents=True)
    uv_bin.write_bytes(b"reviewed uv executable")
    uv_bin.chmod(0o755)
    uv_manifest = install_root / "uv-0.11.32.sha256"
    _write_sha256_manifest(uv_manifest, uv_bin)
    uv_sha256 = hashlib.sha256(uv_bin.read_bytes()).hexdigest()

    release_root = tmp_path / "releases/reviewed"
    verifier_path = release_root / "deploy/scripts/verify_v2_release_artifacts.py"
    verifier_path.parent.mkdir(parents=True)
    verifier_path.write_text("# reviewed verifier\n", encoding="utf-8")
    verifier_sha256 = hashlib.sha256(verifier_path.read_bytes()).hexdigest()
    boundary_script = release_root / "deploy/scripts/prepare-v2-sqlite-boundary.py"
    boundary_script.write_text("# reviewed boundary helper\n", encoding="utf-8")
    venv = release_root / ".venv"
    site_packages = venv / "lib/python3.12/site-packages"
    site_packages.mkdir(parents=True)
    (venv / "lib64").symlink_to("lib", target_is_directory=True)
    (site_packages / "reviewed.pth").write_text("reviewed-release\n", encoding="utf-8")
    runtime = venv / "bin/stocker-runtime"
    runtime.parent.mkdir()
    runtime.write_text("#!/bin/sh\n", encoding="utf-8")
    (venv / "bin/python").symlink_to("stocker-runtime")

    return {
        "ibapi_wheel": ibapi_wheel,
        "ibapi_manifest": ibapi_manifest,
        "protobuf_wheel": protobuf_wheel,
        "protobuf_manifest": protobuf_manifest,
        "uv_bin": uv_bin,
        "uv_manifest": uv_manifest,
        "expected_uv": uv_bin,
        "expected_uv_sha256": uv_sha256,
        "source_root": source,
        "release_root": release_root,
        "venv": venv,
        "verifier_path": verifier_path,
        "expected_verifier_sha256": verifier_sha256,
        "provenance_link": link,
        "expected_link": link,
        "expected_target": target,
        "provenance_root": trusted_root,
        "trust_anchor": tmp_path,
        "required_uid": os.getuid(),
    }


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
    "requirement",
    (
        "protobuf ==5.29.5",
        "protobuf== 5.29.5",
        "protobuf == 5.29.5",
        "protobuf\t==\t5.29.5",
    ),
)
def test_wheel_accepts_only_semantically_exact_protobuf_pin_whitespace(
    tmp_path: Path,
    verifier: ModuleType,
    requirement: str,
) -> None:
    source = tmp_path / "pythonclient"
    source_files = _source_files(source)
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)
    wheel = tmp_path / "case/ibapi-10.49.1-py3-none-any.whl"
    wheel.parent.mkdir()
    _write_wheel(wheel, source_files, requirement=requirement)

    _verify_wheel(verifier, wheel, source, link, target, trusted_root)


@pytest.mark.parametrize(
    "requirement",
    (
        "Protobuf==5.29.5",
        "google-protobuf==5.29.5",
        "protobuf>=5.29.5",
        "protobuf===5.29.5",
        "protobuf==5.29.4",
        "protobuf[extra]==5.29.5",
        "protobuf==5.29.5; python_version >= '3.12'",
        "protobuf @ https://example.invalid/protobuf.whl",
        "protobuf==5.29.5\nRequires-Dist: other==1.0",
    ),
)
def test_wheel_rejects_every_non_exact_protobuf_requirement(
    tmp_path: Path,
    verifier: ModuleType,
    requirement: str,
) -> None:
    source = tmp_path / "pythonclient"
    source_files = _source_files(source)
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)
    wheel = tmp_path / "case/ibapi-10.49.1-py3-none-any.whl"
    wheel.parent.mkdir()
    _write_wheel(wheel, source_files, requirement=requirement)

    with pytest.raises(verifier.ArtifactVerificationError):
        _verify_wheel(verifier, wheel, source, link, target, trusted_root)


def test_wheel_accepts_production_official_builder_top_level_declaration(
    tmp_path: Path,
    verifier: ModuleType,
) -> None:
    source = tmp_path / "pythonclient"
    source_files = _source_files(source)
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)
    wheel = tmp_path / "case/ibapi-10.49.1-py3-none-any.whl"
    wheel.parent.mkdir()
    _write_wheel(
        wheel,
        source_files,
        top_level=b"ibapi\nibapi/protobuf\n",
    )

    _verify_wheel(verifier, wheel, source, link, target, trusted_root)


@pytest.mark.parametrize(
    "top_level",
    (
        b"ibapi/protobuf\nibapi\n",
        b"ibapi\nother\n",
        b"ibapi/protobuf\n",
        b"ibapi\nibapi.protobuf\n",
        b"ibapi\nibapi//protobuf\n",
        b"ibapi\r\nibapi/protobuf\r\n",
        b"ibapi\nibapi/protobuf\n\n",
        b"ibapi\nibapi/protobuf",
        b"ibapi\nibapi/protobuf/\n",
    ),
)
def test_wheel_rejects_every_other_top_level_declaration(
    tmp_path: Path,
    verifier: ModuleType,
    top_level: bytes,
) -> None:
    source = tmp_path / "pythonclient"
    source_files = _source_files(source)
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)
    wheel = tmp_path / "case/ibapi-10.49.1-py3-none-any.whl"
    wheel.parent.mkdir()
    _write_wheel(wheel, source_files, top_level=top_level)

    with pytest.raises(verifier.ArtifactVerificationError):
        _verify_wheel(verifier, wheel, source, link, target, trusted_root)


def test_release_boundary_accepts_exact_manifests_and_protected_paths(
    tmp_path: Path,
    verifier: ModuleType,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)

    verifier.verify_release_boundary(**boundary)
    assert os.readlink(Path(boundary["venv"]) / "lib64") == "lib"


@pytest.mark.parametrize(
    ("relative_path", "directory"),
    (
        (".venv/lib/python3.12/site-packages/ibapi/__pycache__", True),
        (".venv/lib/python3.12/site-packages/google/protobuf/message.pyc", False),
        ("src/stocker_runtime/runtime.pyo", False),
    ),
)
def test_release_boundary_rejects_generated_bytecode_anywhere(
    tmp_path: Path,
    verifier: ModuleType,
    relative_path: str,
    directory: bool,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    generated = Path(boundary["release_root"]) / relative_path
    if directory:
        generated.mkdir(parents=True)
    else:
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"generated bytecode")

    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**boundary)


@pytest.mark.parametrize(
    "target_name",
    (
        "ibapi_wheel",
        "ibapi_manifest",
        "protobuf_wheel",
        "protobuf_manifest",
        "uv_bin",
        "uv_manifest",
        "source_member",
        "verifier_path",
        "release_member",
        "release_root",
        "venv",
        "venv_member",
        "site_packages",
        "site_member",
        "artifact_ancestor",
    ),
)
def test_release_boundary_rejects_writable_protected_paths(
    tmp_path: Path,
    verifier: ModuleType,
    target_name: str,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    source_root = boundary["source_root"]
    venv = boundary["venv"]
    assert isinstance(source_root, Path)
    assert isinstance(venv, Path)
    targets = {
        "ibapi_wheel": boundary["ibapi_wheel"],
        "ibapi_manifest": boundary["ibapi_manifest"],
        "protobuf_wheel": boundary["protobuf_wheel"],
        "protobuf_manifest": boundary["protobuf_manifest"],
        "uv_bin": boundary["uv_bin"],
        "uv_manifest": boundary["uv_manifest"],
        "source_member": source_root / "README.txt",
        "verifier_path": boundary["verifier_path"],
        "release_member": Path(boundary["release_root"])
        / "deploy/scripts/prepare-v2-sqlite-boundary.py",
        "release_root": boundary["release_root"],
        "venv": venv,
        "venv_member": venv / "bin/stocker-runtime",
        "site_packages": venv / "lib/python3.12/site-packages",
        "site_member": venv / "lib/python3.12/site-packages/reviewed.pth",
        "artifact_ancestor": Path(boundary["ibapi_wheel"]).parent,
    }
    target = Path(targets[target_name])
    target.chmod(0o777 if target.is_dir() else 0o666)

    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**boundary)


def test_release_boundary_rejects_wrong_owner_and_non_root_group_write(
    tmp_path: Path,
    verifier: ModuleType,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    wrong_owner = boundary | {"required_uid": os.getuid() + 1}
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**wrong_owner)

    if os.getgid() != 0:
        wheel = Path(boundary["protobuf_wheel"])
        wheel.chmod(0o660)
        with pytest.raises(verifier.ArtifactVerificationError):
            verifier.verify_release_boundary(**boundary)


def test_root_control_mode_allows_only_root_group_write(verifier: ModuleType) -> None:
    root_group_writable = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o660,
        st_uid=os.getuid(),
        st_gid=0,
    )
    verifier._require_root_control(root_group_writable, "artifact", os.getuid())

    for metadata in (
        SimpleNamespace(
            st_mode=stat.S_IFREG | 0o660,
            st_uid=os.getuid(),
            st_gid=1,
        ),
        SimpleNamespace(
            st_mode=stat.S_IFREG | 0o606,
            st_uid=os.getuid(),
            st_gid=0,
        ),
        SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600,
            st_uid=os.getuid() + 1,
            st_gid=0,
        ),
    ):
        with pytest.raises(verifier.ArtifactVerificationError):
            verifier._require_root_control(metadata, "artifact", os.getuid())


def test_release_boundary_rejects_alternate_uv_path_or_hash(
    tmp_path: Path,
    verifier: ModuleType,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    alternate = tmp_path / "alternate/uv"
    alternate.parent.mkdir()
    alternate.write_bytes(Path(boundary["uv_bin"]).read_bytes())
    alternate.chmod(0o755)
    alternate_manifest = Path(boundary["uv_manifest"]).with_name("alternate-uv.sha256")
    _write_sha256_manifest(alternate_manifest, alternate)

    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(
            **(boundary | {"uv_bin": alternate, "uv_manifest": alternate_manifest})
        )
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**(boundary | {"expected_uv_sha256": "0" * 64}))

    Path(boundary["uv_bin"]).chmod(0o644)
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**boundary)


def test_release_boundary_rejects_uv_owner_mismatch(
    tmp_path: Path,
    verifier: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    uv_bin = Path(boundary["uv_bin"])
    real_lstat = verifier._lstat

    def wrong_uv_owner(path: Path, label: str) -> os.stat_result:
        metadata = real_lstat(path, label)
        if path != uv_bin:
            return metadata
        values = list(metadata)
        values[4] = metadata.st_uid + 1
        return os.stat_result(values)

    monkeypatch.setattr(verifier, "_lstat", wrong_uv_owner)
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**boundary)


def test_protected_tree_rejects_directory_symlink_without_skipping_descendants(
    tmp_path: Path,
    verifier: ModuleType,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    writable_tree = tmp_path / "writable-tree"
    writable_tree.mkdir()
    writable_member = writable_tree / "payload.py"
    writable_member.write_text("payload = True\n", encoding="utf-8")
    writable_member.chmod(0o666)
    venv = Path(boundary["venv"])
    (venv / "linked-tree").symlink_to(writable_tree, target_is_directory=True)

    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**boundary)


def test_release_boundary_rejects_alternate_lib64_target(
    tmp_path: Path,
    verifier: ModuleType,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    venv = Path(boundary["venv"])
    lib64 = venv / "lib64"
    lib64.unlink()
    alternate = venv / "alternate-lib"
    alternate.mkdir()
    lib64.symlink_to("alternate-lib", target_is_directory=True)

    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**boundary)


@pytest.mark.parametrize(
    "acl",
    ("system.posix_acl_access", "system.posix_acl_default"),
)
def test_release_boundary_rejects_extended_posix_acl(
    tmp_path: Path,
    verifier: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    acl: str,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    uv_bin = Path(boundary["uv_bin"])
    real_listxattr = verifier.os.listxattr

    def injected_acl(path: object, *, follow_symlinks: bool = True) -> list[str]:
        if Path(path) == uv_bin:
            return [acl]
        return list(real_listxattr(path, follow_symlinks=follow_symlinks))

    monkeypatch.setattr(verifier.os, "listxattr", injected_acl)
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**boundary)


def test_release_boundary_rejects_acl_inspection_error(
    tmp_path: Path,
    verifier: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    uv_bin = Path(boundary["uv_bin"])
    real_listxattr = verifier.os.listxattr

    def failed_acl_check(path: object, *, follow_symlinks: bool = True) -> list[str]:
        if Path(path) == uv_bin:
            raise OSError("acl inspection unavailable")
        return list(real_listxattr(path, follow_symlinks=follow_symlinks))

    monkeypatch.setattr(verifier.os, "listxattr", failed_acl_check)
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**boundary)


@pytest.mark.parametrize(
    "swap",
    (
        "protobuf_bytes",
        "uv_bytes",
        "release_member",
        "ibapi_manifest",
        "protobuf_manifest",
        "source_symlink",
        "verifier_bytes",
        "verifier_symlink",
    ),
)
def test_release_boundary_revalidates_swapped_prerequisites(
    tmp_path: Path,
    verifier: ModuleType,
    swap: str,
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    verifier.verify_release_boundary(**boundary)

    if swap == "protobuf_bytes":
        Path(boundary["protobuf_wheel"]).write_bytes(b"swapped after preinstall")
    elif swap == "uv_bytes":
        Path(boundary["uv_bin"]).write_bytes(b"swapped after preinstall")
    elif swap == "release_member":
        release_member = (
            Path(boundary["release_root"]) / "deploy/scripts/prepare-v2-sqlite-boundary.py"
        )
        release_member.write_text("# swapped after preinstall\n", encoding="utf-8")
        release_member.chmod(0o666)
    elif swap in {"ibapi_manifest", "protobuf_manifest"}:
        manifest = Path(boundary[swap])
        manifest.write_bytes(manifest.read_bytes() + b"0" * 64 + b"  second\n")
    elif swap == "source_symlink":
        source_member = Path(boundary["source_root"]) / "README.txt"
        source_member.unlink()
        source_member.symlink_to(Path(boundary["protobuf_wheel"]))
    elif swap == "verifier_bytes":
        Path(boundary["verifier_path"]).write_text("# swapped verifier\n", encoding="utf-8")
    else:
        verifier_path = Path(boundary["verifier_path"])
        replacement = verifier_path.with_name("replacement.py")
        replacement.write_text("# replacement\n", encoding="utf-8")
        verifier_path.unlink()
        verifier_path.symlink_to(replacement)

    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_release_boundary(**boundary)


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

    writable_venv = tmp_path / "writable-venv"
    writable_site = _installed_fixture(writable_venv, wheel)
    (writable_site / "ibapi/client.py").chmod(0o666)
    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_installed_distribution(
            wheel,
            source,
            writable_venv,
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
    assert os.readlink(link) == "provenance/10.49.1.json"

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
    "literal",
    (
        "/var/lib/stocker/ibkr-api/provenance/10.49.1.json",
        "./provenance/10.49.1.json",
        "provenance/../provenance/10.49.1.json",
        "provenance/other.json",
        "../ibkr-api/provenance/10.49.1.json",
    ),
)
def test_provenance_link_rejects_every_other_literal(
    tmp_path: Path,
    verifier: ModuleType,
    literal: str,
) -> None:
    source_files = _source_files(tmp_path / "pythonclient")
    link, target, trusted_root = _secure_provenance(tmp_path, source_files)
    link.unlink()
    link.symlink_to(literal)

    with pytest.raises(verifier.ArtifactVerificationError):
        verifier.verify_provenance_link(
            link,
            expected_link=link,
            expected_target=target,
            trusted_root=trusted_root,
            trust_anchor=trusted_root.parent,
            required_uid=os.getuid(),
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

    monkeypatch.setattr(verifier, "verify_release_boundary", fail)
    status = verifier.main(
        [
            "preinstall",
            "--ibapi-wheel",
            str(tmp_path / "ibapi-wheel"),
            "--ibapi-manifest",
            str(tmp_path / "ibapi-manifest"),
            "--protobuf-wheel",
            str(tmp_path / "protobuf-wheel"),
            "--protobuf-manifest",
            str(tmp_path / "protobuf-manifest"),
            "--uv-bin",
            str(tmp_path / "uv"),
            "--uv-manifest",
            str(tmp_path / "uv-manifest"),
            "--official-source-root",
            str(tmp_path / "source"),
            "--release-root",
            str(ROOT),
            "--verifier-path",
            str(VERIFIER_PATH),
            "--verifier-sha256",
            "0" * 64,
            "--venv",
            str(ROOT / ".venv"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 78
    assert captured.out == ""
    assert captured.err == "release artifact verification failed\n"
    assert "/secret" not in captured.err


def test_cli_revalidates_the_protected_boundary_after_install(
    tmp_path: Path,
    verifier: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    boundary = _release_boundary_fixture(tmp_path)
    verifier_path = Path(boundary["verifier_path"])
    monkeypatch.setattr(verifier, "__file__", str(verifier_path))
    boundary_calls: list[dict[str, object]] = []
    installed_calls: list[tuple[object, ...]] = []

    def record_boundary(**kwargs: object) -> None:
        boundary_calls.append(kwargs)

    def record_installed(*args: object, **kwargs: object) -> None:
        installed_calls.append((*args, kwargs))

    monkeypatch.setattr(verifier, "verify_release_boundary", record_boundary)
    monkeypatch.setattr(verifier, "verify_installed_distribution", record_installed)
    arguments = [
        "--ibapi-wheel",
        str(boundary["ibapi_wheel"]),
        "--ibapi-manifest",
        str(boundary["ibapi_manifest"]),
        "--protobuf-wheel",
        str(boundary["protobuf_wheel"]),
        "--protobuf-manifest",
        str(boundary["protobuf_manifest"]),
        "--uv-bin",
        str(boundary["uv_bin"]),
        "--uv-manifest",
        str(boundary["uv_manifest"]),
        "--official-source-root",
        str(boundary["source_root"]),
        "--release-root",
        str(boundary["release_root"]),
        "--verifier-path",
        str(verifier_path),
        "--verifier-sha256",
        str(boundary["expected_verifier_sha256"]),
        "--venv",
        str(boundary["venv"]),
    ]

    assert verifier.main(["preinstall", *arguments]) == 0
    assert verifier.main(["postinstall", *arguments]) == 0
    assert len(boundary_calls) == 2
    assert len(installed_calls) == 1
    assert boundary_calls[0] == boundary_calls[1]
    assert verifier.REQUIRED_OWNER_UID == 0
    assert verifier.ROOT_GROUP_GID == 0
    capsys.readouterr()
