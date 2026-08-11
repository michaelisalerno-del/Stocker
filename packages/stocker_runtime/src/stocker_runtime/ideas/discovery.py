"""Deterministic discovery and validation of reviewed first-party idea plugins."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import multiprocessing
import queue
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import Field, model_validator

from stocker_runtime.domain import (
    DomainModel,
    JsonValue,
    ProtectedDataClass,
    canonical_json_bytes,
)
from stocker_runtime.ideas.contract import (
    IdeaActivation,
    IdeaManifest,
    IdeaPlugin,
    MarketDataRequirement,
)

FIRST_PARTY_PREFIX = "stocker_ideas.plugins."
FORBIDDEN_IMPORT_PREFIXES = (
    "ibapi",
    "sqlite3",
    "stocker_execution",
    "stocker_prospective",
    "stocker_runtime.config",
    "stocker_runtime.ingestion",
    "stocker_runtime.storage",
    "stocker_runtime.web",
)
FORBIDDEN_IMPORTS = {"os", "pathlib", "subprocess", "socket", "importlib"}
FORBIDDEN_ATTRIBUTES = {
    "environ",
    "placeOrder",
    "place_order",
    "submit_order",
    "cancel_order",
    "account_id",
    "broker_order_id",
    "risk_approval",
}
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
DISCOVERY_SECONDS = 5.0


class IdeaDiscoveryError(ValueError):
    """A configured plugin is invalid or crosses the authority boundary."""


class IdeaInstrumentConfig(DomainModel):
    """Broker-neutral instrument identity carried by one idea configuration entry."""

    instrument_id: str = Field(min_length=1)
    ibkr_con_id: int = Field(gt=0)
    kind: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    exchange: str = Field(min_length=1)
    currency: str = Field(min_length=1)


class IdeaConfig(DomainModel):
    """Explicit default-off configuration for one reviewed first-party idea."""

    module: str = Field(min_length=1)
    factory: str = Field(default="create_plugin", min_length=1)
    expected_code_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters: Mapping[str, JsonValue]
    universe: tuple[str, ...] = Field(min_length=1)
    instruments: tuple[IdeaInstrumentConfig, ...] = ()
    enabled: bool = False

    @model_validator(mode="after")
    def stable_values(self) -> IdeaConfig:
        if not self.module.startswith(FIRST_PARTY_PREFIX):
            raise ValueError("module must be in the first-party stocker_ideas.plugins package")
        if not _NAME.fullmatch(self.factory):
            raise ValueError("factory must be a stable Python identifier")
        if len(set(self.universe)) != len(self.universe):
            raise ValueError("universe must not contain duplicates")
        instrument_ids = tuple(item.instrument_id for item in self.instruments)
        if len(set(instrument_ids)) != len(instrument_ids):
            raise ValueError("configured instrument identities must not contain duplicates")
        return self


@dataclass(frozen=True)
class DiscoveredPlugin:
    config: IdeaConfig
    plugin: IdeaPlugin
    manifest: IdeaManifest
    code_hash: str
    manifest_hash: str
    manifest_json: str
    parameters_hash: str
    universe_hash: str
    requirements: tuple[MarketDataRequirement, ...]

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (
            self.plugin.manifest.idea_id,
            self.plugin.manifest.idea_version,
            self.code_hash,
            self.parameters_hash,
            self.universe_hash,
        )

    def activation(
        self,
        *,
        instance_id: str,
        run_id: str,
        data_class: ProtectedDataClass,
        activated_at_us: int,
    ) -> IdeaActivation:
        return IdeaActivation(
            instance_id=instance_id,
            parameters=self.config.parameters,
            parameters_hash=self.parameters_hash,
            plugin_code_hash=self.code_hash,
            activated_at_us=activated_at_us,
            run_id=run_id,
            protected_data_class=data_class,
            universe=self.config.universe,
        )


def _source(module_name: str) -> tuple[Path, bytes]:
    if module_name != "stocker_ideas" and not module_name.startswith("stocker_ideas."):
        raise IdeaDiscoveryError("plugin source must stay inside the first-party package")
    package_spec = importlib.util.find_spec("stocker_ideas")
    if package_spec is None:
        raise IdeaDiscoveryError(f"configured plugin module not found: {module_name}")
    roots = () if package_spec is None else tuple(package_spec.submodule_search_locations or ())
    relative = module_name.split(".")[1:]
    for root_value in roots:
        root = Path(root_value).resolve()
        module_path = root.joinpath(*relative)
        candidates = (
            (root / "__init__.py",)
            if not relative
            else (module_path.with_suffix(".py"), module_path / "__init__.py")
        )
        for path in candidates:
            if path.is_file() and path.resolve().is_relative_to(root):
                return path.resolve(), path.read_bytes()
    raise IdeaDiscoveryError(f"configured plugin module not found: {module_name}")


def _validate_source(module_name: str, source: bytes) -> tuple[str, ...]:
    try:
        tree = ast.parse(source, filename=module_name)
    except SyntaxError as error:
        raise IdeaDiscoveryError(f"invalid plugin source: {error}") from error
    first_party_imports: set[str] = set()
    for node in ast.walk(tree):
        imported: list[str] = []
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise IdeaDiscoveryError(
                    "relative plugin imports are forbidden; use a pinned absolute module"
                )
            imported = [node.module or ""]
        for name in imported:
            root = name.split(".", 1)[0]
            if root in FORBIDDEN_IMPORTS or name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                raise IdeaDiscoveryError(f"forbidden plugin import: {name}")
            if name.startswith("stocker_runtime") and not (
                name == "stocker_runtime.domain"
                or name.startswith("stocker_runtime.domain.")
                or name == "stocker_runtime.ideas.contract"
            ):
                raise IdeaDiscoveryError(f"plugin may only use public runtime contracts: {name}")
            if name.startswith("stocker_ideas"):
                if not name.startswith(FIRST_PARTY_PREFIX):
                    raise IdeaDiscoveryError(
                        f"plugin import is outside the reviewed plugin tree: {name}"
                    )
                if name == FIRST_PARTY_PREFIX.removesuffix("."):
                    raise IdeaDiscoveryError(
                        "plugin helpers must be imported through their full pinned module"
                    )
                first_party_imports.add(name)
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            raise IdeaDiscoveryError(f"forbidden plugin authority attribute: {node.attr}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_ATTRIBUTES:
            raise IdeaDiscoveryError(f"forbidden plugin authority name: {node.id}")
    return tuple(sorted(first_party_imports))


def _source_graph(module_name: str) -> tuple[bytes, tuple[str, ...]]:
    """Validate and hash the complete explicit first-party source dependency graph."""

    pending = [module_name]
    sources: dict[str, bytes] = {}
    while pending:
        current = pending.pop()
        if current in sources:
            continue
        parts = current.split(".")
        pending.extend(".".join(parts[:index]) for index in range(1, len(parts)))
        _, source = _source(current)
        sources[current] = source
        pending.extend(name for name in _validate_source(current, source) if name not in sources)
    framed = b"".join(
        len(name.encode()).to_bytes(4, "big")
        + name.encode()
        + len(source).to_bytes(8, "big")
        + source
        for name, source in sorted(sources.items())
    )
    return framed, tuple(sorted(sources))


def reviewed_code_hash(module_name: str) -> str:
    """Return the review pin for a plugin and all explicit first-party dependencies."""

    source_graph, _ = _source_graph(module_name)
    return hashlib.sha256(source_graph).hexdigest()


def _discovery_worker(
    results: Any,
    module_name: str,
    factory_name: str,
    activation_json: str,
) -> None:
    """Import and execute plugin startup hooks outside the recorder process."""

    try:
        module = importlib.import_module(module_name)
        candidates = [name for name in vars(module) if name == factory_name]
        if len(candidates) != 1:
            raise IdeaDiscoveryError("plugin module must expose exactly one configured factory")
        factory = getattr(module, factory_name)
        if not callable(factory) or inspect.signature(factory).parameters:
            raise IdeaDiscoveryError("plugin factory must be a no-argument callable")
        plugin = factory()
        if not isinstance(plugin, IdeaPlugin):
            raise IdeaDiscoveryError("factory result does not satisfy IdeaPlugin protocol")
        activation = IdeaActivation.model_validate_json(activation_json)
        requirements = tuple(plugin.requirements(activation))
        results.put(
            (
                "ok",
                plugin,
                plugin.manifest.model_dump(mode="python"),
                tuple(item.model_dump(mode="python") for item in requirements),
            )
        )
    except BaseException as error:
        results.put(("error", f"{type(error).__name__}:{error}"))


def _load_plugin_isolated(
    config: IdeaConfig, provisional: IdeaActivation
) -> tuple[IdeaPlugin, IdeaManifest, tuple[MarketDataRequirement, ...]]:
    context = multiprocessing.get_context("spawn")
    results = context.Queue(maxsize=1)
    process = context.Process(
        target=_discovery_worker,
        args=(results, config.module, config.factory, provisional.to_canonical_json().decode()),
        daemon=True,
    )
    process.start()
    try:
        try:
            result = results.get(timeout=DISCOVERY_SECONDS)
        except queue.Empty as error:
            raise IdeaDiscoveryError("plugin discovery exceeded startup bound") from error
        if not isinstance(result, tuple) or not result or result[0] != "ok":
            detail = (
                result[1] if isinstance(result, tuple) and len(result) > 1 else "worker crashed"
            )
            raise IdeaDiscoveryError(f"plugin requirements/startup failed: {detail}")
        _, plugin_value, manifest_value, requirement_values = result
        plugin = cast(IdeaPlugin, plugin_value)
        manifest = IdeaManifest.model_validate(manifest_value)
        requirements = tuple(
            MarketDataRequirement.model_validate(item) for item in requirement_values
        )
        return plugin, manifest, requirements
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        results.close()
        results.join_thread()


def _validate_parameter_value(value: JsonValue, schema: Mapping[str, Any], path: str) -> None:
    expected = schema.get("type")
    valid = (
        expected is None
        or (expected == "object" and isinstance(value, Mapping))
        or (expected == "array" and isinstance(value, tuple | list))
        or (expected == "string" and isinstance(value, str))
        or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (expected == "number" and isinstance(value, int | float) and not isinstance(value, bool))
        or (expected == "boolean" and isinstance(value, bool))
        or (expected == "null" and value is None)
    )
    if not valid:
        raise IdeaDiscoveryError(f"parameter {path} does not match declared type {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise IdeaDiscoveryError(f"parameter {path} is not in its declared enum")
    if isinstance(value, Mapping) and expected == "object":
        properties = cast(Mapping[str, Mapping[str, Any]], schema.get("properties", {}))
        required = set(cast(Sequence[str], schema.get("required", ())))
        missing = required.difference(value)
        if missing:
            raise IdeaDiscoveryError(f"parameter {path} is missing {sorted(missing)}")
        if schema.get("additionalProperties") is False:
            unknown = set(value).difference(properties)
            if unknown:
                raise IdeaDiscoveryError(f"parameter {path} has unknown fields {sorted(unknown)}")
        for key, nested in value.items():
            if key in properties:
                _validate_parameter_value(nested, properties[key], f"{path}.{key}")
    if isinstance(value, tuple | list) and expected == "array":
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise IdeaDiscoveryError(f"parameter {path} has too few items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise IdeaDiscoveryError(f"parameter {path} has too many items")
        item_schema = cast(Mapping[str, Any], schema.get("items", {}))
        for index, nested in enumerate(value):
            _validate_parameter_value(nested, item_schema, f"{path}[{index}]")


def _validate_parameter_schema(schema: Mapping[str, Any], path: str = "parameter_schema") -> None:
    allowed = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "minItems",
        "maxItems",
    }
    unknown = set(schema).difference(allowed)
    if unknown:
        raise IdeaDiscoveryError(f"{path} has unsupported keywords {sorted(unknown)}")
    schema_type = schema.get("type")
    allowed_types = {"object", "array", "string", "integer", "number", "boolean", "null"}
    if schema_type not in allowed_types:
        raise IdeaDiscoveryError(f"{path} has an unsupported type")
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", ())
        if not isinstance(properties, Mapping) or not isinstance(required, tuple | list):
            raise IdeaDiscoveryError(f"{path} object declaration is invalid")
        if not all(isinstance(name, str) for name in required):
            raise IdeaDiscoveryError(f"{path} required fields must be strings")
        if not set(required).issubset(properties):
            raise IdeaDiscoveryError(f"{path} requires undeclared properties")
        if schema.get("additionalProperties") is not False:
            raise IdeaDiscoveryError(f"{path} objects must reject additional properties")
        for name, nested in properties.items():
            if not isinstance(name, str) or not isinstance(nested, Mapping):
                raise IdeaDiscoveryError(f"{path} properties are invalid")
            _validate_parameter_schema(nested, f"{path}.{name}")
    if schema_type == "array":
        items = schema.get("items")
        if not isinstance(items, Mapping):
            raise IdeaDiscoveryError(f"{path} array requires an item schema")
        _validate_parameter_schema(items, f"{path}.items")


def _discover(config: IdeaConfig) -> DiscoveredPlugin:
    source_graph, _ = _source_graph(config.module)
    code_hash = hashlib.sha256(source_graph).hexdigest()
    if config.expected_code_hash != code_hash:
        raise IdeaDiscoveryError("configured plugin code hash does not match reviewed source")
    parameters_hash = hashlib.sha256(canonical_json_bytes(config.parameters)).hexdigest()
    universe_hash = hashlib.sha256(canonical_json_bytes(config.universe)).hexdigest()
    provisional = IdeaActivation(
        instance_id="discovery",
        parameters=config.parameters,
        parameters_hash=parameters_hash,
        plugin_code_hash=code_hash,
        activated_at_us=0,
        run_id="discovery",
        protected_data_class=ProtectedDataClass.PROSPECTIVE,
        universe=config.universe,
    )
    try:
        plugin, manifest, requirements = _load_plugin_isolated(config, provisional)
    except Exception as error:
        if isinstance(error, IdeaDiscoveryError):
            raise
        raise IdeaDiscoveryError(f"plugin startup failed validation: {error}") from error
    if manifest.api_version != 1 or not _NAME.fullmatch(manifest.idea_id):
        raise IdeaDiscoveryError("manifest has unsupported API version or unstable idea_id")
    _validate_parameter_schema(manifest.parameter_schema)
    _validate_parameter_value(config.parameters, manifest.parameter_schema, "parameters")
    manifest_json = manifest.to_canonical_json().decode()
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
    if config.expected_manifest_hash != manifest_hash:
        raise IdeaDiscoveryError("configured plugin manifest hash does not match reviewed manifest")
    if not requirements or len(set(requirements)) != len(requirements):
        raise IdeaDiscoveryError("plugin requirements must be nonempty and unique")
    return DiscoveredPlugin(
        config=config,
        plugin=plugin,
        manifest=manifest,
        code_hash=code_hash,
        manifest_hash=manifest_hash,
        manifest_json=manifest_json,
        parameters_hash=parameters_hash,
        universe_hash=universe_hash,
        requirements=tuple(sorted(requirements, key=lambda item: item.to_canonical_json())),
    )


def discover_plugins(configs: Sequence[IdeaConfig]) -> tuple[DiscoveredPlugin, ...]:
    """Discover enabled entries in deterministic order; an empty config is a safe no-op."""

    discovered = tuple(_discover(config) for config in configs if config.enabled)
    identities: set[tuple[str, str]] = set()
    modules: set[tuple[str, str]] = set()
    for item in discovered:
        identity = (item.manifest.idea_id, item.manifest.idea_version)
        module_factory = (item.config.module, item.config.factory)
        if identity in identities or module_factory in modules:
            raise IdeaDiscoveryError(f"duplicate configured plugin identity: {identity}")
        identities.add(identity)
        modules.add(module_factory)
    return tuple(sorted(discovered, key=lambda item: item.identity))


def load_idea_configs(path: str | Path) -> tuple[IdeaConfig, ...]:
    """Load one explicit bounded JSON configuration file; absent configuration is caller-owned."""

    raw = Path(path).read_bytes()
    if len(raw) > 1024 * 1024:
        raise IdeaDiscoveryError("idea configuration exceeds 1 MiB")
    try:
        values = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IdeaDiscoveryError(f"idea configuration is invalid JSON: {error}") from error
    if not isinstance(values, list):
        raise IdeaDiscoveryError("idea configuration root must be a list")
    try:
        return tuple(
            IdeaConfig.model_validate_json(json.dumps(value, allow_nan=False)) for value in values
        )
    except Exception as error:
        raise IdeaDiscoveryError(f"idea configuration entry is invalid: {error}") from error


def aggregate_requirements(
    plugins: Sequence[DiscoveredPlugin],
) -> tuple[MarketDataRequirement, ...]:
    """Merge identical feed/instrument/cadence needs; blocking flags merge conservatively."""

    merged: dict[tuple[str, str], MarketDataRequirement] = {}
    for plugin in plugins:
        for requirement in plugin.requirements:
            key = (requirement.instrument_id, requirement.feed_kind)
            prior = merged.get(key)
            if prior is not None and prior.cadence != requirement.cadence:
                raise IdeaDiscoveryError(f"conflicting cadence for {key}")
            merged[key] = MarketDataRequirement(
                instrument_id=requirement.instrument_id,
                feed_kind=requirement.feed_kind,
                event_kind=(
                    requirement.event_kind
                    if prior is None or prior.event_kind == requirement.event_kind
                    else None
                ),
                cadence=requirement.cadence,
                gaps_block=(requirement.gaps_block or (prior.gaps_block if prior else False)),
                staleness_block=(
                    requirement.staleness_block or (prior.staleness_block if prior else False)
                ),
            )
    return tuple(sorted(merged.values(), key=lambda item: item.to_canonical_json()))


def aggregate_instruments(
    plugins: Sequence[DiscoveredPlugin],
) -> tuple[IdeaInstrumentConfig, ...]:
    """Merge exact configured instrument identities and reject ambiguous metadata."""

    merged: dict[str, IdeaInstrumentConfig] = {}
    for plugin in plugins:
        for instrument in plugin.config.instruments:
            prior = merged.get(instrument.instrument_id)
            if prior is not None and prior != instrument:
                raise IdeaDiscoveryError(
                    f"conflicting instrument metadata for {instrument.instrument_id}"
                )
            merged[instrument.instrument_id] = instrument
    return tuple(merged[key] for key in sorted(merged))
