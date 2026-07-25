from __future__ import annotations

import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import pytest

import stocker_prospective.ibkr as ibkr_module
from stocker_prospective.ibkr_api import (
    OfficialIBKRApiProvenance,
    OfficialIBKRApiRelease,
    OfficialIBKRApiUpdateStatus,
    evaluate_official_ibkr_api_update,
    inspect_official_ibkr_api_archive,
    parse_latest_official_ibkr_api_release,
    python_package_tree_sha256,
    write_immutable_official_ibkr_api_provenance,
    write_official_ibkr_api_update_status,
)

ROOT = Path(__file__).parents[1]


def test_installed_ibapi_without_official_provenance_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ibapi = ModuleType("ibapi")
    fake_ibapi.__file__ = str(tmp_path / "ibapi/__init__.py")
    monkeypatch.setitem(sys.modules, "ibapi", fake_ibapi)
    monkeypatch.setattr(ibkr_module, "official_ibkr_api_available", lambda: True)

    with pytest.raises(
        ibkr_module.OfficialIBKRDependencyError,
        match="blocked_unverified_official_ibkr_api",
    ):
        ibkr_module.require_official_ibkr_api(tmp_path / "missing-provenance.json")


def test_malformed_official_provenance_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ibapi = ModuleType("ibapi")
    fake_ibapi.__file__ = str(tmp_path / "ibapi/__init__.py")
    monkeypatch.setitem(sys.modules, "ibapi", fake_ibapi)
    monkeypatch.setattr(ibkr_module, "official_ibkr_api_available", lambda: True)
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ibkr_module.OfficialIBKRDependencyError,
        match="blocked_unverified_official_ibkr_api",
    ):
        ibkr_module.require_official_ibkr_api(provenance)


def test_installed_ibapi_tree_must_match_official_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "ibapi"
    package.mkdir()
    module_path = package / "__init__.py"
    module_path.write_text('VERSION = "unexpected"\n', encoding="utf-8")
    fake_ibapi = ModuleType("ibapi")
    fake_ibapi.__file__ = str(module_path)
    monkeypatch.setitem(sys.modules, "ibapi", fake_ibapi)
    monkeypatch.setattr(ibkr_module, "official_ibkr_api_available", lambda: True)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source": "interactive_brokers_official_tws_api",
                "release_channel": "latest",
                "platform": "mac_unix",
                "api_version": "10.48.1",
                "release_date": "2026-07-07",
                "official_page_url": "https://interactivebrokers.github.io/",
                "official_page_checked_at_utc": "2026-07-25T09:00:00Z",
                "source_url": (
                    "https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"
                ),
                "archive_filename": "twsapi_macunix.1048.01.zip",
                "archive_sha256": "0" * 64,
                "source_tree_sha256": "0" * 64,
                "installed_tree_sha256": "0" * 64,
                "registered_at_utc": "2026-07-25T09:05:00Z",
                "registered_by": "test-operator",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ibkr_module.OfficialIBKRDependencyError,
        match="blocked_unverified_official_ibkr_api",
    ):
        ibkr_module.require_official_ibkr_api(provenance)


def test_matching_installed_ibapi_and_provenance_are_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "ibapi"
    package.mkdir()
    module_path = package / "__init__.py"
    module_path.write_text('__version__ = "10.48.1"\n', encoding="utf-8")
    fake_ibapi = ModuleType("ibapi")
    fake_ibapi.__file__ = str(module_path)
    fake_ibapi.__version__ = "10.48.1"
    monkeypatch.setitem(sys.modules, "ibapi", fake_ibapi)
    monkeypatch.setattr(ibkr_module, "official_ibkr_api_available", lambda: True)
    tree_hash = python_package_tree_sha256(package)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source": "interactive_brokers_official_tws_api",
                "release_channel": "latest",
                "platform": "mac_unix",
                "api_version": "10.48.1",
                "release_date": "2026-07-07",
                "official_page_url": "https://interactivebrokers.github.io/",
                "official_page_checked_at_utc": "2026-07-25T09:00:00Z",
                "source_url": (
                    "https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"
                ),
                "archive_filename": "twsapi_macunix.1048.01.zip",
                "archive_sha256": "0" * 64,
                "source_tree_sha256": tree_hash,
                "installed_tree_sha256": tree_hash,
                "registered_at_utc": "2026-07-25T09:05:00Z",
                "registered_by": "test-operator",
            }
        ),
        encoding="utf-8",
    )

    assert ibkr_module.require_official_ibkr_api(provenance) is fake_ibapi


def test_installed_ibapi_version_must_match_official_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "ibapi"
    package.mkdir()
    module_path = package / "__init__.py"
    module_path.write_text('__version__ = "10.48.1"\n', encoding="utf-8")
    fake_ibapi = ModuleType("ibapi")
    fake_ibapi.__file__ = str(module_path)
    fake_ibapi.__version__ = "10.47.0"
    monkeypatch.setitem(sys.modules, "ibapi", fake_ibapi)
    monkeypatch.setattr(ibkr_module, "official_ibkr_api_available", lambda: True)
    tree_hash = python_package_tree_sha256(package)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source": "interactive_brokers_official_tws_api",
                "release_channel": "latest",
                "platform": "mac_unix",
                "api_version": "10.48.1",
                "release_date": "2026-07-07",
                "official_page_url": "https://interactivebrokers.github.io/",
                "official_page_checked_at_utc": "2026-07-25T09:00:00Z",
                "source_url": (
                    "https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"
                ),
                "archive_filename": "twsapi_macunix.1048.01.zip",
                "archive_sha256": "0" * 64,
                "source_tree_sha256": tree_hash,
                "installed_tree_sha256": tree_hash,
                "registered_at_utc": "2026-07-25T09:05:00Z",
                "registered_by": "test-operator",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ibkr_module.OfficialIBKRDependencyError,
        match="blocked_unverified_official_ibkr_api",
    ):
        ibkr_module.require_official_ibkr_api(provenance)


def test_provenance_rejects_source_and_installed_tree_disagreement() -> None:
    with pytest.raises(
        ValueError,
        match="installed source tree does not match",
    ):
        OfficialIBKRApiProvenance(
            schema_version="1",
            source="interactive_brokers_official_tws_api",
            release_channel="latest",
            platform="mac_unix",
            api_version="10.48.1",
            release_date=date(2026, 7, 7),
            official_page_url="https://interactivebrokers.github.io/",
            official_page_checked_at_utc=datetime(2026, 7, 25, 9, tzinfo=UTC),
            source_url=(
                "https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"
            ),
            archive_filename="twsapi_macunix.1048.01.zip",
            archive_sha256="0" * 64,
            source_tree_sha256="1" * 64,
            installed_tree_sha256="2" * 64,
            registered_at_utc=datetime(2026, 7, 25, 9, 5, tzinfo=UTC),
            registered_by="test-operator",
        )


def test_provenance_archive_name_must_match_official_source_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "ibapi"
    package.mkdir()
    module_path = package / "__init__.py"
    module_path.write_text('VERSION = "10.48.1"\n', encoding="utf-8")
    fake_ibapi = ModuleType("ibapi")
    fake_ibapi.__file__ = str(module_path)
    monkeypatch.setitem(sys.modules, "ibapi", fake_ibapi)
    monkeypatch.setattr(ibkr_module, "official_ibkr_api_available", lambda: True)
    provenance = tmp_path / "provenance.json"
    tree_hash = "2fb1a3296db30cc2ec0c21503856b06990ca7f0fc2cefcfe6f4cbf8c9c196a63"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source": "interactive_brokers_official_tws_api",
                "release_channel": "latest",
                "platform": "mac_unix",
                "api_version": "10.48.1",
                "release_date": "2026-07-07",
                "official_page_url": "https://interactivebrokers.github.io/",
                "official_page_checked_at_utc": "2026-07-25T09:00:00Z",
                "source_url": (
                    "https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"
                ),
                "archive_filename": "twsapi_macunix.9999.01.zip",
                "archive_sha256": "0" * 64,
                "source_tree_sha256": tree_hash,
                "installed_tree_sha256": tree_hash,
                "registered_at_utc": "2026-07-25T09:05:00Z",
                "registered_by": "test-operator",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ibkr_module.OfficialIBKRDependencyError,
        match="blocked_unverified_official_ibkr_api",
    ):
        ibkr_module.require_official_ibkr_api(provenance)


def test_latest_mac_unix_release_is_parsed_from_official_download_page() -> None:
    html = """
    <td>
      <a href="//interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip">
        TWS API <span>Latest</span> for Mac / Unix
      </a>
      <p>Version: <strong>API 10.48</strong><br />
      Release Date: <strong>Jul 7 2026</strong></p>
    </td>
    """

    release = parse_latest_official_ibkr_api_release(html)

    assert release.api_version == "10.48"
    assert release.release_date.isoformat() == "2026-07-07"
    assert (
        release.source_url
        == "https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"
    )
    assert release.release_channel == "latest"
    assert release.platform == "mac_unix"


def test_release_parser_does_not_mix_stable_url_with_latest_metadata() -> None:
    html = """
    <tr>
      <td>
        <a href="//interactivebrokers.github.io/downloads/twsapi_macunix.1045.01.zip">
          TWS API <span>Stable</span> for Mac / Unix
        </a>
        <p>Version: <strong>API 10.45</strong><br />
        Release Date: <strong>Mar 30 2026</strong></p>
      </td>
    </tr>
    <tr>
      <td>
        <a href="//interactivebrokers.github.io/downloads/TWS API Install 1048.01.msi">
          TWS API <span>Latest</span> for Windows
        </a>
        <p>Version: <strong>API 10.48</strong><br />
        Release Date: <strong>Jul 7 2026</strong></p>
      </td>
      <td>
        <a href="//interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip">
          TWS API <span>Latest</span> for Mac / Unix
        </a>
        <p>Version: <strong>API 10.48</strong><br />
        Release Date: <strong>Jul 7 2026</strong></p>
      </td>
    </tr>
    """

    release = parse_latest_official_ibkr_api_release(html)

    assert release.source_url.endswith("twsapi_macunix.1048.01.zip")
    assert release.api_version == "10.48"


def test_official_archive_is_tied_to_matching_installed_python_sources(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "twsapi_macunix.1048.01.zip"
    package_source = 'VERSION = {"major": 10, "minor": 48, "micro": 1}\n'
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("IBJts/API_VersionNum.txt", "API_Version=10.48.01\r\n")
        bundle.writestr(
            "IBJts/source/pythonclient/ibapi/__init__.py",
            package_source,
        )
    installed_package = tmp_path / "installed/ibapi"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text(package_source, encoding="utf-8")
    release = OfficialIBKRApiRelease(
        api_version="10.48",
        release_date=date(2026, 7, 7),
        source_url=("https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"),
    )

    provenance = inspect_official_ibkr_api_archive(
        archive,
        installed_package_root=installed_package,
        release=release,
        registered_by="test-operator",
        checked_at=datetime(2026, 7, 25, 9, tzinfo=UTC),
    )

    assert provenance.api_version == "10.48.1"
    assert provenance.archive_filename == archive.name
    assert provenance.source_tree_sha256 == provenance.installed_tree_sha256
    assert len(provenance.archive_sha256) == 64


def test_newer_official_release_is_reported_without_automatic_installation() -> None:
    tree_hash = "2fb1a3296db30cc2ec0c21503856b06990ca7f0fc2cefcfe6f4cbf8c9c196a63"
    installed = OfficialIBKRApiProvenance(
        schema_version="1",
        source="interactive_brokers_official_tws_api",
        release_channel="latest",
        platform="mac_unix",
        api_version="10.48.1",
        release_date=date(2026, 7, 7),
        official_page_url="https://interactivebrokers.github.io/",
        official_page_checked_at_utc=datetime(2026, 7, 25, 9, tzinfo=UTC),
        source_url=("https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"),
        archive_filename="twsapi_macunix.1048.01.zip",
        archive_sha256="0" * 64,
        source_tree_sha256=tree_hash,
        installed_tree_sha256=tree_hash,
        registered_at_utc=datetime(2026, 7, 25, 9, 5, tzinfo=UTC),
        registered_by="test-operator",
    )
    current = OfficialIBKRApiRelease(
        api_version="10.49",
        release_date=date(2026, 8, 4),
        source_url=("https://interactivebrokers.github.io/downloads/twsapi_macunix.1049.01.zip"),
    )

    status = evaluate_official_ibkr_api_update(
        installed,
        current,
        checked_at=datetime(2026, 8, 5, 9, tzinfo=UTC),
    )

    assert status.update_available is True
    assert status.installed_api_version == "10.48.1"
    assert status.latest_api_version == "10.49"
    assert status.automatic_installation is False


def test_update_status_for_another_install_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "ibapi"
    package.mkdir()
    module_path = package / "__init__.py"
    module_path.write_text('__version__ = "10.48.1"\n', encoding="utf-8")
    fake_ibapi = ModuleType("ibapi")
    fake_ibapi.__file__ = str(module_path)
    fake_ibapi.__version__ = "10.48.1"
    monkeypatch.setitem(sys.modules, "ibapi", fake_ibapi)
    monkeypatch.setattr(ibkr_module, "official_ibkr_api_available", lambda: True)
    tree_hash = python_package_tree_sha256(package)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        OfficialIBKRApiProvenance(
            schema_version="1",
            source="interactive_brokers_official_tws_api",
            release_channel="latest",
            platform="mac_unix",
            api_version="10.48.1",
            release_date=date(2026, 7, 7),
            official_page_url="https://interactivebrokers.github.io/",
            official_page_checked_at_utc=datetime(2026, 7, 25, 9, tzinfo=UTC),
            source_url=(
                "https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"
            ),
            archive_filename="twsapi_macunix.1048.01.zip",
            archive_sha256="0" * 64,
            source_tree_sha256=tree_hash,
            installed_tree_sha256=tree_hash,
            registered_at_utc=datetime(2026, 7, 25, 9, 5, tzinfo=UTC),
            registered_by="test-operator",
        ).model_dump_json(),
        encoding="utf-8",
    )
    status_path = tmp_path / "update-status.json"
    write_official_ibkr_api_update_status(
        status_path,
        OfficialIBKRApiUpdateStatus(
            checked_at_utc=datetime(2026, 7, 25, 10, tzinfo=UTC),
            installed_api_version="10.47.0",
            installed_source_url=(
                "https://interactivebrokers.github.io/downloads/twsapi_macunix.1047.02.zip"
            ),
            latest_api_version="10.48",
            latest_release_date=date(2026, 7, 7),
            latest_source_url=(
                "https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"
            ),
            update_available=True,
        ),
    )
    monkeypatch.setenv("STOCKER_IBKR_API_PROVENANCE", str(provenance))
    monkeypatch.setenv("STOCKER_IBKR_API_UPDATE_STATUS", str(status_path))

    projection = ibkr_module.official_ibkr_api_projection()

    assert projection["verified"] is True
    assert projection["blocker"] == "blocked_ibkr_api_update_status_invalid"


def test_verified_ibapi_without_update_status_is_not_reported_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "ibapi"
    package.mkdir()
    module_path = package / "__init__.py"
    module_path.write_text('__version__ = "10.48.1"\n', encoding="utf-8")
    fake_ibapi = ModuleType("ibapi")
    fake_ibapi.__file__ = str(module_path)
    fake_ibapi.__version__ = "10.48.1"
    monkeypatch.setitem(sys.modules, "ibapi", fake_ibapi)
    monkeypatch.setattr(ibkr_module, "official_ibkr_api_available", lambda: True)
    tree_hash = python_package_tree_sha256(package)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        OfficialIBKRApiProvenance(
            schema_version="1",
            source="interactive_brokers_official_tws_api",
            release_channel="latest",
            platform="mac_unix",
            api_version="10.48.1",
            release_date=date(2026, 7, 7),
            official_page_url="https://interactivebrokers.github.io/",
            official_page_checked_at_utc=datetime.now(UTC),
            source_url=(
                "https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"
            ),
            archive_filename="twsapi_macunix.1048.01.zip",
            archive_sha256="0" * 64,
            source_tree_sha256=tree_hash,
            installed_tree_sha256=tree_hash,
            registered_at_utc=datetime.now(UTC),
            registered_by="test-operator",
        ).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setenv("STOCKER_IBKR_API_PROVENANCE", str(provenance))
    monkeypatch.delenv("STOCKER_IBKR_API_UPDATE_STATUS", raising=False)

    projection = ibkr_module.official_ibkr_api_projection()

    assert projection["verified"] is True
    assert projection["update_status_fresh"] is False
    assert projection["blocker"] == "blocked_ibkr_api_update_status_missing"


@pytest.mark.parametrize(
    "checked_at",
    [
        datetime.now(UTC) - timedelta(days=15),
        datetime.now(UTC) + timedelta(minutes=10),
    ],
)
def test_stale_or_future_update_status_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checked_at: datetime,
) -> None:
    package = tmp_path / "ibapi"
    package.mkdir()
    module_path = package / "__init__.py"
    module_path.write_text('__version__ = "10.48.1"\n', encoding="utf-8")
    fake_ibapi = ModuleType("ibapi")
    fake_ibapi.__file__ = str(module_path)
    fake_ibapi.__version__ = "10.48.1"
    monkeypatch.setitem(sys.modules, "ibapi", fake_ibapi)
    monkeypatch.setattr(ibkr_module, "official_ibkr_api_available", lambda: True)
    tree_hash = python_package_tree_sha256(package)
    provenance_record = OfficialIBKRApiProvenance(
        schema_version="1",
        source="interactive_brokers_official_tws_api",
        release_channel="latest",
        platform="mac_unix",
        api_version="10.48.1",
        release_date=date(2026, 7, 7),
        official_page_url="https://interactivebrokers.github.io/",
        official_page_checked_at_utc=datetime.now(UTC),
        source_url=("https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"),
        archive_filename="twsapi_macunix.1048.01.zip",
        archive_sha256="0" * 64,
        source_tree_sha256=tree_hash,
        installed_tree_sha256=tree_hash,
        registered_at_utc=datetime.now(UTC),
        registered_by="test-operator",
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(provenance_record.model_dump_json(), encoding="utf-8")
    status_path = tmp_path / "update-status.json"
    write_official_ibkr_api_update_status(
        status_path,
        OfficialIBKRApiUpdateStatus(
            checked_at_utc=checked_at,
            installed_api_version="10.48.1",
            installed_source_url=provenance_record.source_url,
            latest_api_version="10.48",
            latest_release_date=date(2026, 7, 7),
            latest_source_url=provenance_record.source_url,
            update_available=False,
        ),
    )
    monkeypatch.setenv("STOCKER_IBKR_API_PROVENANCE", str(provenance))
    monkeypatch.setenv("STOCKER_IBKR_API_UPDATE_STATUS", str(status_path))

    projection = ibkr_module.official_ibkr_api_projection()

    assert projection["update_status_fresh"] is False
    assert projection["update_available"] is None
    assert projection["blocker"] == "blocked_ibkr_api_update_check_stale"


def test_immutable_provenance_race_never_overwrites_first_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree_hash = "2fb1a3296db30cc2ec0c21503856b06990ca7f0fc2cefcfe6f4cbf8c9c196a63"
    first = OfficialIBKRApiProvenance(
        schema_version="1",
        source="interactive_brokers_official_tws_api",
        release_channel="latest",
        platform="mac_unix",
        api_version="10.48.1",
        release_date=date(2026, 7, 7),
        official_page_url="https://interactivebrokers.github.io/",
        official_page_checked_at_utc=datetime(2026, 7, 25, 9, tzinfo=UTC),
        source_url=("https://interactivebrokers.github.io/downloads/twsapi_macunix.1048.01.zip"),
        archive_filename="twsapi_macunix.1048.01.zip",
        archive_sha256="0" * 64,
        source_tree_sha256=tree_hash,
        installed_tree_sha256=tree_hash,
        registered_at_utc=datetime(2026, 7, 25, 9, 5, tzinfo=UTC),
        registered_by="first-operator",
    )
    second = first.model_copy(update={"registered_by": "second-operator"})
    destination = tmp_path / "provenance.json"

    def racing_link(_source: object, target: object) -> None:
        Path(target).write_text(first.model_dump_json(indent=2), encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(os, "link", racing_link)

    with pytest.raises(
        ValueError,
        match="provenance is immutable",
    ):
        write_immutable_official_ibkr_api_provenance(destination, second)

    assert (
        OfficialIBKRApiProvenance.model_validate_json(destination.read_text(encoding="utf-8"))
        == first
    )


def test_weekly_update_timer_checks_but_never_installs_broker_code() -> None:
    service = (ROOT / "deploy/systemd/stocker-ibkr-api-update.service").read_text(encoding="utf-8")
    timer = (ROOT / "deploy/systemd/stocker-ibkr-api-update.timer").read_text(encoding="utf-8")

    assert "ibkr-api check-update" in service
    assert "User=stocker" in service
    assert "NoNewPrivileges=true" in service
    assert "EnvironmentFile=" not in service
    assert "pip install" not in service
    assert "uv pip" not in service
    assert "curl" not in service
    assert "wget" not in service
    assert "OnCalendar=weekly" in timer
