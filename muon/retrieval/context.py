"""The schema context: introspected structure plus plugin notes, cached behind a fingerprint.

Map: `load_schema_context` computes the fingerprint (graph counts, T-Box bytes and
`SCHEMA_FORMAT_VERSION`), reads the cache file, and introspects only when the file is
missing, stale, invalid or a refresh was asked for. The notes text is never cached. Text
rendering lives here so the cached `GraphSchema` stays free of presentation.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, LiteralString

from pydantic import BaseModel, ConfigDict, ValidationError

from muon.files import write_text_atomically
from muon.graph import GraphClient
from muon.ontology.materializer import LabelMaterializer
from muon.plugins import UniversePlugin
from muon.retrieval.schema import (
    SCHEMA_FORMAT_VERSION,
    GraphSchema,
    NodeLabelSchema,
    PropertySchema,
    RelationshipPattern,
    RelationshipTypeSchema,
    introspect_schema,
)

READ_GRAPH_COUNTS: LiteralString = (
    "CALL db.stats.retrieve('GRAPH COUNTS') YIELD data "
    "RETURN data.nodes AS nodes, data.relationships AS relationships"
)
CACHE_FILE_SUFFIX = ".json"
JSON_INDENT = 2
JSON_SEPARATORS = (",", ":")
TYPE_SEPARATOR = " | "
VALUE_SEPARATOR = ", "
RELATIONSHIP_PROPERTY_SEPARATOR = "; "


class CacheStatus(StrEnum):
    """How a load obtained its schema."""

    HIT = "hit"
    MISSING = "missing"
    STALE = "stale"
    INVALID = "invalid"
    REFRESHED = "refreshed"


class SchemaCacheError(Exception):
    """A cache file that cannot be read or written."""


class InvalidSchemaCacheError(SchemaCacheError):
    """A cache file that is not valid JSON or does not match `SchemaCacheFile`."""


class SchemaCacheFile(BaseModel):
    """The cache file: the schema, the fingerprint it was built under and the format version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: int
    fingerprint: str
    graph_schema: GraphSchema


@dataclass(frozen=True)
class SchemaRequest:
    """What 1 load needs: the plugin, the label mapping of its universe and the refresh flag."""

    plugin: UniversePlugin
    materializer: LabelMaterializer
    should_refresh: bool


@dataclass(frozen=True)
class SchemaContext:
    """The schema of a universe, its notes, the text for the prompt and how it was obtained."""

    graph_schema: GraphSchema
    notes: str
    text: str
    cache_status: CacheStatus
    cache_path: Path


def build_canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=JSON_SEPARATORS)


async def compute_fingerprint(client: GraphClient, ontology_paths: Sequence[Path]) -> str:
    """Hash the format version, the graph counts and the bytes of every T-Box file.

    The counts cover the whole database, so a change in any universe changes the value.
    """
    (counts,) = await client.execute_read(READ_GRAPH_COUNTS)
    canonical_counts = {
        name: sorted(build_canonical_json(entry) for entry in counts[name])
        for name in ("nodes", "relationships")
    }
    hasher = hashlib.sha256()
    hasher.update(f"{SCHEMA_FORMAT_VERSION}\n".encode())
    hasher.update(build_canonical_json(canonical_counts).encode())
    for path in ontology_paths:
        hasher.update(hashlib.sha256(path.read_bytes()).digest())
    return hasher.hexdigest()


def read_schema_cache(path: Path) -> SchemaCacheFile | None:
    """Read the cache file at `path`, or `None` when there is none."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SchemaCacheError(f"Failed to read schema cache {path}: {error}") from error
    try:
        return SchemaCacheFile.model_validate_json(text)
    except ValidationError as error:
        raise InvalidSchemaCacheError(f"Invalid schema cache {path}: {error}") from error


def write_schema_cache(path: Path, cache_file: SchemaCacheFile) -> None:
    text = cache_file.model_dump_json(indent=JSON_INDENT) + "\n"
    try:
        write_text_atomically(path, text)
    except OSError as error:
        raise SchemaCacheError(f"Failed to write schema cache {path}: {error}") from error


def render_property(property_schema: PropertySchema) -> str:
    optional_mark = "" if property_schema.is_mandatory else "?"
    text = f"{property_schema.name}{optional_mark}: {TYPE_SEPARATOR.join(property_schema.types)}"
    if property_schema.enum_values:
        text += f"{VALUE_SEPARATOR}values: {VALUE_SEPARATOR.join(property_schema.enum_values)}"
    return text


def render_node_block(node_label: NodeLabelSchema) -> list[str]:
    noun = "node" if node_label.count == 1 else "nodes"
    header = f"{node_label.label} ({node_label.count:,} {noun}"
    if node_label.ancestor_labels:
        header += f"; also {VALUE_SEPARATOR.join(node_label.ancestor_labels)}"
    lines = [header + ")"]
    lines.extend(f"- {render_property(property_)}" for property_ in node_label.properties)
    return lines


def render_pattern(
    pattern: RelationshipPattern, relationship_types: dict[str, RelationshipTypeSchema]
) -> str:
    properties = relationship_types[pattern.type].properties
    property_text = ""
    if properties:
        rendered = RELATIONSHIP_PROPERTY_SEPARATOR.join(render_property(p) for p in properties)
        property_text = f" {{{rendered}}}"
    return (
        f"(:{pattern.start_label})-[:{pattern.type}{property_text}]->(:{pattern.end_label}) "
        f"{pattern.count:,}"
    )


def render_schema_text(graph_schema: GraphSchema, plugin: UniversePlugin) -> str:
    """Render the schema and the plugin notes as the plain text of the retrieve prompt."""
    relationship_types = {item.type: item for item in graph_schema.relationship_types}
    lines = [f"# Graph schema: {plugin.name} ({graph_schema.universe})", "", "## Node labels", ""]
    for node_label in graph_schema.node_labels:
        lines.extend(render_node_block(node_label))
        lines.append("")
    lines.extend(["## Relationships", ""])
    lines.extend(render_pattern(pattern, relationship_types) for pattern in graph_schema.patterns)
    lines.extend(["", "## Notes", "", plugin.notes_text.rstrip("\n")])
    return "\n".join(lines) + "\n"


def classify_cache(
    cached: SchemaCacheFile | None, fingerprint: str
) -> tuple[CacheStatus, GraphSchema | None]:
    """Return `HIT` with the cached schema, or the reason there is none."""
    if cached is None:
        return CacheStatus.MISSING, None
    if cached.format_version != SCHEMA_FORMAT_VERSION or cached.fingerprint != fingerprint:
        return CacheStatus.STALE, None
    return CacheStatus.HIT, cached.graph_schema


def read_cache_status(path: Path, fingerprint: str) -> tuple[CacheStatus, GraphSchema | None]:
    try:
        return classify_cache(read_schema_cache(path), fingerprint)
    except InvalidSchemaCacheError:
        return CacheStatus.INVALID, None


async def load_schema_context(
    client: GraphClient, request: SchemaRequest, cache_dir: Path
) -> SchemaContext:
    """Return the schema context of `request.plugin`, from the cache when it is current."""
    plugin = request.plugin
    if request.materializer.universe_label != plugin.prefix:
        raise ValueError(
            f"Materializer universe {request.materializer.universe_label!r} "
            f"differs from plugin prefix {plugin.prefix!r}"
        )
    cache_path = cache_dir / f"{plugin.prefix}{CACHE_FILE_SUFFIX}"
    fingerprint = await compute_fingerprint(client, plugin.ontology_paths)
    if request.should_refresh:
        status, graph_schema = CacheStatus.REFRESHED, None
    else:
        status, graph_schema = read_cache_status(cache_path, fingerprint)
    if graph_schema is None:
        graph_schema = await introspect_schema(client, request.materializer)
        write_schema_cache(
            cache_path,
            SchemaCacheFile(
                format_version=SCHEMA_FORMAT_VERSION,
                fingerprint=fingerprint,
                graph_schema=graph_schema,
            ),
        )
    return SchemaContext(
        graph_schema=graph_schema,
        notes=plugin.notes_text,
        text=render_schema_text(graph_schema, plugin),
        cache_status=status,
        cache_path=cache_path,
    )
