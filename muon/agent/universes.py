"""Universe plugins: manifest, discovery and registry.

A plugin is a folder `<universes_dir>/<prefix>/` with a `universe.toml` manifest and a
notes file. The engine finds plugins by their manifests and never imports one by name.
The T-Box files of a plugin come from `muon.ontology.model.list_ontology_paths`.
"""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from muon.files import TEXT_ENCODING
from muon.ontology.model import UNIVERSE_PREFIX_PATTERN, OntologyError, list_ontology_paths

MANIFEST_FILE_NAME = "universe.toml"
IGNORED_FOLDER_NAMES = frozenset({"__pycache__"})
HIDDEN_FOLDER_PREFIX = "."


class UniverseError(Exception):
    """A plugin folder or manifest that is missing, malformed or inconsistent."""


class UniverseManifest(BaseModel):
    """The contents of `universe.toml`. `notes` is a path relative to the plugin folder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prefix: str
    name: str
    notes: str


@dataclass(frozen=True)
class UniversePlugin:
    """A validated plugin: its manifest, folder, notes text and T-Box files."""

    manifest: UniverseManifest
    folder: Path
    notes_text: str
    ontology_paths: tuple[Path, ...]

    @property
    def prefix(self) -> str:
        return self.manifest.prefix

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def notes_path(self) -> Path:
        return self.folder / self.manifest.notes


class UniverseRegistry:
    """The plugins found by discovery, by prefix."""

    def __init__(self, plugins: Mapping[str, UniversePlugin]) -> None:
        self.plugins = dict(plugins)

    def list_prefixes(self) -> tuple[str, ...]:
        return tuple(sorted(self.plugins))

    def get(self, prefix: str) -> UniversePlugin:
        if prefix not in self.plugins:
            known = ", ".join(self.list_prefixes()) or "none"
            raise UniverseError(f"Unknown universe {prefix!r}; known: {known}")
        return self.plugins[prefix]


def read_manifest(folder: Path) -> UniverseManifest:
    """Parse `<folder>/universe.toml`."""
    path = folder / MANIFEST_FILE_NAME
    if not path.is_file():
        raise UniverseError(f"Plugin folder {folder} has no {MANIFEST_FILE_NAME}")
    try:
        raw_manifest = tomllib.loads(path.read_text(TEXT_ENCODING))
    except tomllib.TOMLDecodeError as error:
        raise UniverseError(f"Invalid TOML in {path}: {error}") from error
    try:
        return UniverseManifest.model_validate(raw_manifest)
    except ValidationError as error:
        raise UniverseError(f"Invalid manifest {path}: {error}") from error


def validate_plugin(folder: Path, manifest: UniverseManifest, ontology_dir: Path) -> UniversePlugin:
    """Check a manifest against its folder and the ontology files, then read its notes."""
    if not UNIVERSE_PREFIX_PATTERN.fullmatch(manifest.prefix):
        raise UniverseError(
            f"Invalid prefix {manifest.prefix!r} in {folder / MANIFEST_FILE_NAME}: "
            f"expected {UNIVERSE_PREFIX_PATTERN.pattern}"
        )
    if manifest.prefix != folder.name:
        raise UniverseError(
            f"Prefix {manifest.prefix!r} in {folder / MANIFEST_FILE_NAME} "
            f"differs from the folder name {folder.name!r}"
        )
    notes_path = folder / manifest.notes
    if not notes_path.resolve().is_relative_to(folder.resolve()):
        raise UniverseError(f"Notes path {manifest.notes!r} in {folder} leaves the plugin folder")
    if not notes_path.is_file():
        raise UniverseError(f"Missing notes file {notes_path} (field 'notes' of {folder})")
    try:
        ontology_paths = list_ontology_paths(ontology_dir, manifest.prefix)
    except OntologyError as error:
        raise UniverseError(f"Plugin folder {folder} has no usable T-Box: {error}") from error
    return UniversePlugin(
        manifest=manifest,
        folder=folder,
        notes_text=notes_path.read_text(TEXT_ENCODING),
        ontology_paths=ontology_paths,
    )


def discover_universes(universes_dir: Path, ontology_dir: Path) -> UniverseRegistry:
    """Validate every plugin folder under `universes_dir` and return the registry."""
    if not universes_dir.is_dir():
        raise UniverseError(f"Universes folder {universes_dir} does not exist")
    plugins: dict[str, UniversePlugin] = {}
    for folder in sorted(universes_dir.iterdir()):
        if not folder.is_dir() or is_ignored_folder(folder):
            continue
        manifest = read_manifest(folder)
        plugins[manifest.prefix] = validate_plugin(folder, manifest, ontology_dir)
    return UniverseRegistry(plugins)


def is_ignored_folder(folder: Path) -> bool:
    return folder.name.startswith(HIDDEN_FOLDER_PREFIX) or folder.name in IGNORED_FOLDER_NAMES
