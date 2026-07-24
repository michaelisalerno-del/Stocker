"""Immutable deployment-bundle contract for the prospective server."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

BUNDLE_MANIFEST_VERSION: Final = "1"
SCIENTIFIC_CLASSIFICATION: Final = (
    "Previous-close front-options context + current intraday H0 stock condition -> "
    "improved prediction that near-term underlying movement exceeds previous-close "
    "option-implied movement."
)
ANCHOR_COHORT: Final = "anchor_frozen_20"
PROTECTED_START: Final = "2026-01-01"
DISALLOWED_BUNDLE_SUFFIXES = {
    ".csv",
    ".db",
    ".duckdb",
    ".parquet",
    ".sqlite",
    ".sqlite3",
}


class BundleError(RuntimeError):
    """A fail-closed bundle contract violation."""


class BundleBuildSpec(BaseModel):
    """Research-machine inputs used to construct a self-contained bundle."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str = Field(min_length=3, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]+$")
    created_at_utc: datetime
    m0_artifact: Path
    m1_artifact: Path
    preprocessor: Path
    feature_schema: Path
    universe: Path
    threshold: float = Field(ge=0.0, le=1.0)
    threshold_provenance: Path
    training_start: str
    training_end: str
    historical_reference_start: str
    historical_reference_end: str
    holdout_start: str
    holdout_end: str
    protected_start: str = PROTECTED_START
    code_feature_contract_version: str = Field(min_length=1)
    audit_references: list[Path] = Field(default_factory=list)
    determinism_references: list[Path] = Field(default_factory=list)


class FileIdentity(BaseModel):
    """Identity for one file copied into a deployment bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)
    format: str


class FeatureDefinition(BaseModel):
    """One ordered feature contract entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    dtype: Literal["float64", "int64", "bool", "string"]
    missing: Literal["reject", "allow"]


class FeatureSchemaIdentity(FileIdentity):
    """Hashed ordered feature schema."""

    schema_version: str
    features: list[FeatureDefinition]


class UniverseIdentity(FileIdentity):
    """Hashed frozen universe identity."""

    universe_id: str
    cohort: Literal["anchor_frozen_20"]
    symbol_count: int
    symbols: list[str]
    source_artifact: str


class FrozenThreshold(BaseModel):
    """Frozen M1 selection threshold and provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Literal["M1"] = "M1"
    value: float = Field(ge=0.0, le=1.0)
    source: Literal["weighted_2024_development_predictions"]
    frozen_before_holdout_outcomes: Literal[True]
    provenance: FileIdentity


class DateInterval(BaseModel):
    """Inclusive date interval recorded in the frozen handoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: str
    end: str


class BundleManifest(BaseModel):
    """Strict server-readable frozen deployment manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal["1"]
    bundle_id: str
    bundle_kind: Literal["frozen_m1"] = "frozen_m1"
    created_at_utc: datetime
    scientific_classification: Literal[
        "Previous-close front-options context + current intraday H0 stock condition -> "
        "improved prediction that near-term underlying movement exceeds previous-close "
        "option-implied movement."
    ]
    claim_limit: Literal["underlying_movement_selection_not_option_profitability"] = (
        "underlying_movement_selection_not_option_profitability"
    )
    code_feature_contract_version: str
    m0_artifact: FileIdentity
    m1_artifact: FileIdentity
    preprocessor: FileIdentity
    feature_schema: FeatureSchemaIdentity
    universe: UniverseIdentity
    threshold: FrozenThreshold
    training_interval: DateInterval
    historical_reference_interval: DateInterval
    holdout_interval: DateInterval
    protected_start: Literal["2026-01-01"]
    audit_references: list[FileIdentity]
    determinism_references: list[FileIdentity]
    files: dict[str, str]


class BundleVerification(BaseModel):
    """Result returned by every bundle verification."""

    manifest: BundleManifest
    verified: bool
    blockers: list[str]
    manifest_sha256: str


class ActiveBundle(BaseModel):
    """Atomic active-bundle pointer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    manifest_sha256: str
    activated_at_utc: datetime
    operator: str


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, relative_path: str) -> FileIdentity:
    return FileIdentity(
        path=relative_path,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        format=path.suffix.lstrip(".") or "binary",
    )


def _require_safe_source(path: Path, role: str) -> None:
    if not path.is_file():
        raise BundleError(
            f"blocked_missing_verified_frozen_bundle: missing {role} artifact at {path}"
        )
    if path.suffix.lower() in DISALLOWED_BUNDLE_SUFFIXES:
        raise BundleError(
            f"blocked_unsafe_bundle_content: {role} may not contain datasets or databases"
        )


def _copy_identity(source: Path, root: Path, relative_path: str) -> FileIdentity:
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return _identity(destination, relative_path)


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"blocked_feature_schema_mismatch: invalid {role} JSON") from exc
    if not isinstance(payload, dict):
        raise BundleError(f"blocked_feature_schema_mismatch: {role} must be an object")
    return payload


def _feature_schema(identity: FileIdentity, path: Path) -> FeatureSchemaIdentity:
    payload = _load_json(path, "feature schema")
    features = [FeatureDefinition.model_validate(item) for item in payload.get("features", [])]
    names = [feature.name for feature in features]
    if not features or len(names) != len(set(names)):
        raise BundleError("blocked_feature_schema_mismatch: features must be non-empty and unique")
    return FeatureSchemaIdentity(
        path=identity.path,
        sha256=identity.sha256,
        size_bytes=identity.size_bytes,
        format=identity.format,
        schema_version=str(payload.get("schema_version", "")),
        features=features,
    )


def _universe(identity: FileIdentity, path: Path) -> UniverseIdentity:
    payload = _load_json(path, "universe")
    symbols = payload.get("symbols")
    if (
        payload.get("cohort") != ANCHOR_COHORT
        or not isinstance(symbols, list)
        or len(symbols) != 20
        or len(set(symbols)) != 20
        or any(not isinstance(symbol, str) or symbol != symbol.upper() for symbol in symbols)
    ):
        raise BundleError(
            "blocked_frozen_universe_mismatch: anchor_frozen_20 must contain 20 unique "
            "registered uppercase symbols"
        )
    return UniverseIdentity(
        path=identity.path,
        sha256=identity.sha256,
        size_bytes=identity.size_bytes,
        format=identity.format,
        universe_id=str(payload.get("universe_id", "")),
        cohort=ANCHOR_COHORT,
        symbol_count=20,
        symbols=symbols,
        source_artifact=str(payload.get("source_artifact", "")),
    )


def _threshold(
    value: float,
    identity: FileIdentity,
    path: Path,
) -> FrozenThreshold:
    payload = _load_json(path, "threshold provenance")
    if (
        payload.get("model") != "M1"
        or payload.get("source") != "weighted_2024_development_predictions"
        or payload.get("frozen_before_holdout_outcomes") is not True
        or float(payload.get("value", -1.0)) != value
    ):
        raise BundleError("blocked_feature_schema_mismatch: frozen threshold provenance is invalid")
    return FrozenThreshold(
        value=value,
        source="weighted_2024_development_predictions",
        frozen_before_holdout_outcomes=True,
        provenance=identity,
    )


def build_bundle(spec: BundleBuildSpec, destination: str | Path) -> BundleManifest:
    """Copy frozen research artifacts into a self-contained immutable-format bundle."""

    output = Path(destination)
    if output.exists():
        raise BundleError(f"bundle destination already exists: {output}")
    if spec.protected_start != PROTECTED_START:
        raise BundleError("blocked_unsafe_bundle_content: protected_start must remain 2026-01-01")
    sources: list[tuple[Path, str]] = [
        (spec.m0_artifact, "M0"),
        (spec.m1_artifact, "M1"),
        (spec.preprocessor, "preprocessor"),
        (spec.feature_schema, "feature schema"),
        (spec.universe, "universe"),
        (spec.threshold_provenance, "threshold provenance"),
        *((path, "audit reference") for path in spec.audit_references),
        *((path, "determinism reference") for path in spec.determinism_references),
    ]
    for path, role in sources:
        _require_safe_source(path, role)

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        m0 = _copy_identity(spec.m0_artifact, temporary, f"artifacts/m0{spec.m0_artifact.suffix}")
        m1 = _copy_identity(spec.m1_artifact, temporary, f"artifacts/m1{spec.m1_artifact.suffix}")
        preprocessor = _copy_identity(
            spec.preprocessor,
            temporary,
            f"artifacts/preprocessor{spec.preprocessor.suffix}",
        )
        schema_file = _copy_identity(
            spec.feature_schema, temporary, "contracts/feature-schema.json"
        )
        universe_file = _copy_identity(spec.universe, temporary, "contracts/universe.json")
        threshold_file = _copy_identity(
            spec.threshold_provenance,
            temporary,
            "contracts/threshold-provenance.json",
        )
        audits = [
            _copy_identity(path, temporary, f"references/audit/{index:02d}-{path.name}")
            for index, path in enumerate(spec.audit_references, start=1)
        ]
        determinism = [
            _copy_identity(path, temporary, f"references/determinism/{index:02d}-{path.name}")
            for index, path in enumerate(spec.determinism_references, start=1)
        ]
        identities = [
            m0,
            m1,
            preprocessor,
            schema_file,
            universe_file,
            threshold_file,
            *audits,
            *determinism,
        ]
        manifest = BundleManifest(
            manifest_version=BUNDLE_MANIFEST_VERSION,
            bundle_id=spec.bundle_id,
            created_at_utc=spec.created_at_utc.astimezone(UTC),
            scientific_classification=SCIENTIFIC_CLASSIFICATION,
            code_feature_contract_version=spec.code_feature_contract_version,
            m0_artifact=m0,
            m1_artifact=m1,
            preprocessor=preprocessor,
            feature_schema=_feature_schema(schema_file, temporary / schema_file.path),
            universe=_universe(universe_file, temporary / universe_file.path),
            threshold=_threshold(
                spec.threshold,
                threshold_file,
                temporary / threshold_file.path,
            ),
            training_interval=DateInterval(start=spec.training_start, end=spec.training_end),
            historical_reference_interval=DateInterval(
                start=spec.historical_reference_start,
                end=spec.historical_reference_end,
            ),
            holdout_interval=DateInterval(start=spec.holdout_start, end=spec.holdout_end),
            protected_start=PROTECTED_START,
            audit_references=audits,
            determinism_references=determinism,
            files={identity.path: identity.sha256 for identity in identities},
        )
        (temporary / "manifest.json").write_bytes(_canonical_json(manifest.model_dump(mode="json")))
        os.replace(temporary, output)
        verified = verify_bundle(output)
        if not verified.verified:
            raise BundleError(", ".join(verified.blockers))
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _manifest(path: Path) -> BundleManifest:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(
            f"blocked_missing_verified_frozen_bundle: missing manifest at {manifest_path}"
        )
    try:
        return BundleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BundleError("blocked_feature_schema_mismatch: invalid bundle manifest") from exc


def verify_bundle(path: str | Path) -> BundleVerification:
    """Verify manifest schema, every file hash, universe, and embedded contracts."""

    root = Path(path)
    manifest = _manifest(root)
    blockers: list[str] = []
    for relative_path, expected_hash in manifest.files.items():
        candidate = root / relative_path
        try:
            candidate.relative_to(root)
        except ValueError:
            blockers.append("blocked_unsafe_bundle_content")
            continue
        if not candidate.is_file():
            blockers.append("blocked_missing_verified_frozen_bundle")
        elif _sha256(candidate) != expected_hash:
            blockers.append("blocked_frozen_artifact_hash_mismatch")
    try:
        schema = _feature_schema(
            manifest.feature_schema,
            root / manifest.feature_schema.path,
        )
        if schema != manifest.feature_schema:
            blockers.append("blocked_feature_schema_mismatch")
        universe = _universe(manifest.universe, root / manifest.universe.path)
        if universe != manifest.universe:
            blockers.append("blocked_frozen_universe_mismatch")
        threshold = _threshold(
            manifest.threshold.value,
            manifest.threshold.provenance,
            root / manifest.threshold.provenance.path,
        )
        if threshold != manifest.threshold:
            blockers.append("blocked_feature_schema_mismatch")
    except BundleError as exc:
        code = str(exc).split(":", 1)[0]
        blockers.append(code)
    manifest_hash = _sha256(root / "manifest.json")
    unique_blockers = list(dict.fromkeys(blockers))
    return BundleVerification(
        manifest=manifest,
        verified=not unique_blockers,
        blockers=unique_blockers,
        manifest_sha256=manifest_hash,
    )


def validate_feature_vector(
    manifest: BundleManifest,
    values: list[tuple[str, object]],
) -> None:
    """Validate ordered runtime values before any frozen-model invocation."""

    expected = manifest.feature_schema.features
    if [name for name, _ in values] != [feature.name for feature in expected]:
        raise BundleError("blocked_feature_schema_mismatch: feature ordering differs")
    for feature, (_, value) in zip(expected, values, strict=True):
        if value is None:
            if feature.missing == "reject":
                raise BundleError(
                    f"blocked_feature_schema_mismatch: {feature.name} may not be missing"
                )
            continue
        valid = {
            "float64": isinstance(value, (int, float)) and not isinstance(value, bool),
            "int64": isinstance(value, int) and not isinstance(value, bool),
            "bool": isinstance(value, bool),
            "string": isinstance(value, str),
        }[feature.dtype]
        if not valid:
            raise BundleError(
                f"blocked_feature_schema_mismatch: {feature.name} expected {feature.dtype}"
            )


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(
        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )


def install_bundle(
    bundle_path: str | Path,
    bundle_root: str | Path,
    *,
    operator: str,
) -> Path:
    """Copy a verified bundle into a versioned server directory exactly once."""

    if not operator.strip():
        raise BundleError("operator identity is required")
    verification = verify_bundle(bundle_path)
    if not verification.verified:
        raise BundleError(", ".join(verification.blockers))
    root = Path(bundle_root)
    installed_root = root / "installed"
    installed_root.mkdir(parents=True, exist_ok=True)
    destination = installed_root / verification.manifest.bundle_id
    if destination.exists():
        raise BundleError(f"installed bundle already exists: {destination}")
    temporary = installed_root / f".{verification.manifest.bundle_id}.{uuid.uuid4().hex}.tmp"
    shutil.copytree(bundle_path, temporary)
    copied = verify_bundle(temporary)
    if not copied.verified:
        shutil.rmtree(temporary)
        raise BundleError(", ".join(copied.blockers))
    os.replace(temporary, destination)
    _make_read_only(destination)
    _append_activation_audit(
        root,
        {
            "action": "install",
            "bundle_id": verification.manifest.bundle_id,
            "operator": operator,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "manifest_sha256": verification.manifest_sha256,
        },
    )
    return destination


def list_installed_bundles(bundle_root: str | Path) -> list[BundleManifest]:
    """List verified installed bundle manifests."""

    root = Path(bundle_root) / "installed"
    if not root.exists():
        return []
    manifests: list[BundleManifest] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        verification = verify_bundle(path)
        if verification.verified:
            manifests.append(verification.manifest)
    return manifests


def _active_pointer(bundle_root: Path) -> ActiveBundle | None:
    path = bundle_root / "active.json"
    if not path.is_file():
        return None
    try:
        return ActiveBundle.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BundleError("blocked_frozen_artifact_hash_mismatch: invalid active pointer") from exc


def _append_activation_audit(bundle_root: Path, payload: dict[str, object]) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    with (bundle_root / "operator-actions.jsonl").open("ab") as handle:
        handle.write(_canonical_json(payload))
        handle.flush()
        os.fsync(handle.fileno())


def activate_bundle(
    bundle_id: str,
    bundle_root: str | Path,
    *,
    operator: str,
    expected_current_bundle_id: str | None,
) -> ActiveBundle:
    """Atomically activate a verified installed bundle with compare-and-swap semantics."""

    if not operator.strip():
        raise BundleError("operator identity is required")
    root = Path(bundle_root)
    current = _active_pointer(root)
    current_id = None if current is None else current.bundle_id
    if current_id != expected_current_bundle_id:
        raise BundleError(
            f"active bundle changed: expected {expected_current_bundle_id!r}, found {current_id!r}"
        )
    installed = root / "installed" / bundle_id
    verification = verify_bundle(installed)
    if not verification.verified:
        raise BundleError(", ".join(verification.blockers))
    pointer = ActiveBundle(
        bundle_id=bundle_id,
        manifest_sha256=verification.manifest_sha256,
        activated_at_utc=datetime.now(UTC),
        operator=operator,
    )
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / ".active.tmp"
    temporary.write_bytes(_canonical_json(pointer.model_dump(mode="json")))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, root / "active.json")
    _append_activation_audit(
        root,
        {
            "action": "activate",
            **pointer.model_dump(mode="json"),
        },
    )
    return pointer


def load_active_bundle(bundle_root: str | Path) -> BundleVerification:
    """Load and reverify the active installed bundle before scoring."""

    root = Path(bundle_root)
    pointer = _active_pointer(root)
    if pointer is None:
        raise BundleError("blocked_missing_verified_frozen_bundle: no active bundle")
    verification = verify_bundle(root / "installed" / pointer.bundle_id)
    if not verification.verified or verification.manifest_sha256 != pointer.manifest_sha256:
        raise BundleError("blocked_frozen_artifact_hash_mismatch")
    return verification
