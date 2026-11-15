"""Introspection of a universe graph into a `GraphSchema`.

Map: `introspect_schema` runs 6 reads in order. (1) `COUNT_LABEL_SETS` groups the nodes
of the universe by label set, and each set collapses to its leaf class. (2)
`READ_NODE_PROPERTIES` gives the property types of every label set, merged per leaf.
(3) `COUNT_PATTERNS` counts the relationship patterns between leaf classes. (4)
`READ_RELATIONSHIP_PROPERTIES` gives the property types of the relationship types found in
(3). (5) and (6) sample the values of the string properties, so a small closed vocabulary
shows up as a list of values.

The module names no universe. The class hierarchy comes from the `LabelMaterializer`.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, LiteralString

from pydantic import BaseModel, ConfigDict

from muon.graph import GraphClient
from muon.ontology.materializer import LabelMaterializer

# Change this whenever the models below or the introspection rules change, or caches
# written by the old code are served as current.
SCHEMA_FORMAT_VERSION = 1
MAX_ENUM_VALUES = 30
KEY_PROPERTIES = ("id", "name")
STRING_TYPE = "STRING"
NOT_NULL_PATTERN = re.compile(r"\s*NOT NULL")
RELATIONSHIP_TYPE_PREFIX = ":`"
RELATIONSHIP_TYPE_SUFFIX = "`"

COUNT_LABEL_SETS: LiteralString = (
    "MATCH (n:$($universe)) RETURN labels(n) AS labels, count(*) AS count"
)
READ_NODE_PROPERTIES: LiteralString = (
    "CALL db.schema.nodeTypeProperties() "
    "YIELD nodeLabels, propertyName, propertyTypes, mandatory "
    "WHERE $universe IN nodeLabels AND propertyName IS NOT NULL "
    "RETURN nodeLabels AS owner, propertyName AS name, propertyTypes AS types, "
    "mandatory AS is_mandatory"
)
COUNT_PATTERNS: LiteralString = (
    "MATCH (a:$($universe))-[r]->(b:$($universe)) "
    "RETURN labels(a) AS start_labels, type(r) AS type, labels(b) AS end_labels, "
    "count(*) AS count"
)
READ_RELATIONSHIP_PROPERTIES: LiteralString = (
    "CALL db.schema.relTypeProperties() "
    "YIELD relType, propertyName, propertyTypes, mandatory "
    "WHERE propertyName IS NOT NULL "
    "RETURN relType AS owner, propertyName AS name, propertyTypes AS types, "
    "mandatory AS is_mandatory"
)
SAMPLE_NODE_VALUES: LiteralString = (
    "UNWIND $properties AS property "
    "CALL (property) { "
    "MATCH (n:$($label)&$($universe)) "
    "WHERE NOT any(descendant IN $descendant_labels WHERE descendant IN labels(n)) "
    "AND n[property] IS NOT NULL "
    "RETURN DISTINCT n[property] AS value LIMIT $limit "
    "} "
    "RETURN property, value"
)
SAMPLE_RELATIONSHIP_VALUES: LiteralString = (
    "UNWIND $properties AS property "
    "CALL (property) { "
    "MATCH (:$($universe))-[r:$($type)]->(:$($universe)) "
    "WHERE r[property] IS NOT NULL "
    "RETURN DISTINCT r[property] AS value LIMIT $limit "
    "} "
    "RETURN property, value"
)


class SchemaIntrospectionError(Exception):
    """The graph state does not allow a schema: an empty universe or an unplaceable label set."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PropertySchema(FrozenModel):
    """A property of a leaf class or relationship type.

    `types` lists every value type seen, with `NOT NULL` dropped. `is_mandatory` is true
    when every node (or relationship) has it. `enum_values` lists all values of a small
    string vocabulary, and is `None` otherwise.
    """

    name: str
    types: tuple[str, ...]
    is_mandatory: bool
    enum_values: tuple[str, ...] | None


class NodeLabelSchema(FrozenModel):
    """A leaf class: its node count, its ancestor class labels and its properties."""

    label: str
    count: int
    ancestor_labels: tuple[str, ...]
    properties: tuple[PropertySchema, ...]


class RelationshipTypeSchema(FrozenModel):
    type: str
    properties: tuple[PropertySchema, ...]


class RelationshipPattern(FrozenModel):
    """`count` relationships of `type` from a node of leaf class `start_label` to `end_label`."""

    start_label: str
    type: str
    end_label: str
    count: int


class GraphSchema(FrozenModel):
    """The structure of one universe graph. Every tuple is sorted."""

    universe: str
    node_labels: tuple[NodeLabelSchema, ...]
    relationship_types: tuple[RelationshipTypeSchema, ...]
    patterns: tuple[RelationshipPattern, ...]

    def list_labels(self) -> frozenset[str]:
        """Return every leaf class label and every ancestor label of the universe."""
        labels = {node_label.label for node_label in self.node_labels}
        for node_label in self.node_labels:
            labels.update(node_label.ancestor_labels)
        return frozenset(labels)

    def list_relationship_types(self) -> frozenset[str]:
        return frozenset(relationship.type for relationship in self.relationship_types)


@dataclass(frozen=True)
class PropertyRow:
    """One row of `db.schema.*TypeProperties`: a property of 1 label set or relationship type."""

    owner: frozenset[str]
    name: str
    types: tuple[str, ...]
    is_mandatory: bool


@dataclass(frozen=True)
class PropertyShape:
    name: str
    types: tuple[str, ...]
    is_mandatory: bool


@dataclass(frozen=True)
class LeafGroup:
    """The nodes whose label sets collapse to 1 leaf class."""

    count: int
    label_sets: frozenset[frozenset[str]]


def find_leaf_label(
    labels: Sequence[str], ancestor_labels: Mapping[str, frozenset[str]], universe: str
) -> str:
    """Return the class label of `labels` that no other class label of the set descends from.

    Labels that are not classes of the ontology, such as the universe label, are ignored.
    """
    class_labels = [label for label in labels if label in ancestor_labels]
    leaves = [
        label
        for label in class_labels
        if not any(label in ancestor_labels[other] for other in class_labels if other != label)
    ]
    if len(leaves) != 1:
        raise SchemaIntrospectionError(
            f"Label set {sorted(labels)} in universe {universe!r} has {len(leaves)} leaf "
            f"classes ({sorted(leaves)}), expected 1"
        )
    return leaves[0]


def normalize_property_type(raw_type: str) -> str:
    """Drop every `NOT NULL`: `LIST<STRING NOT NULL> NOT NULL` becomes `LIST<STRING>`."""
    return NOT_NULL_PATTERN.sub("", raw_type)


def sort_property_names(names: Iterable[str]) -> list[str]:
    """Sort names alphabetically with the key properties first."""

    def build_sort_key(name: str) -> tuple[int, str]:
        rank = KEY_PROPERTIES.index(name) if name in KEY_PROPERTIES else len(KEY_PROPERTIES)
        return (rank, name)

    return sorted(names, key=build_sort_key)


def merge_property_rows(
    rows: Iterable[PropertyRow], owners: frozenset[frozenset[str]]
) -> tuple[PropertyShape, ...]:
    """Merge the rows of every owner of 1 leaf class (or relationship type) per property.

    Types are unioned and sorted. A property is mandatory only if every owner in `owners`
    has it as mandatory.
    """
    rows_by_name: dict[str, list[PropertyRow]] = {}
    for row in rows:
        rows_by_name.setdefault(row.name, []).append(row)
    shapes = []
    for name in sort_property_names(rows_by_name):
        name_rows = rows_by_name[name]
        types = {normalize_property_type(raw) for row in name_rows for raw in row.types}
        mandatory_owners = {row.owner for row in name_rows if row.is_mandatory}
        shapes.append(
            PropertyShape(
                name=name, types=tuple(sorted(types)), is_mandatory=mandatory_owners == owners
            )
        )
    return tuple(shapes)


def list_enum_candidates(shapes: Iterable[PropertyShape]) -> list[str]:
    """Return the names of the string properties that can hold a small closed vocabulary."""
    return [
        shape.name
        for shape in shapes
        if shape.types == (STRING_TYPE,) and shape.name not in KEY_PROPERTIES
    ]


def build_property_schemas(
    shapes: Iterable[PropertyShape], enum_values: Mapping[str, tuple[str, ...]]
) -> tuple[PropertySchema, ...]:
    return tuple(
        PropertySchema(
            name=shape.name,
            types=shape.types,
            is_mandatory=shape.is_mandatory,
            enum_values=enum_values.get(shape.name),
        )
        for shape in shapes
    )


def select_enum_values(sample_rows: Iterable[Mapping[str, str]]) -> dict[str, tuple[str, ...]]:
    """Group sampled values by property; a property with over `MAX_ENUM_VALUES` has none."""
    values_by_property: dict[str, set[str]] = {}
    for row in sample_rows:
        values_by_property.setdefault(row["property"], set()).add(row["value"])
    return {
        name: tuple(sorted(values))
        for name, values in values_by_property.items()
        if len(values) <= MAX_ENUM_VALUES
    }


@dataclass(frozen=True)
class GraphReader:
    """What every read of 1 introspection run shares."""

    client: GraphClient
    universe: str
    ancestor_labels: Mapping[str, frozenset[str]]


async def read_leaf_groups(reader: GraphReader) -> dict[str, LeafGroup]:
    records = await reader.client.execute_read(COUNT_LABEL_SETS, {"universe": reader.universe})
    if not records:
        raise SchemaIntrospectionError(
            f"Universe {reader.universe!r} has no nodes; run `muon seed {reader.universe}` first"
        )
    counts: dict[str, int] = {}
    label_sets: dict[str, set[frozenset[str]]] = {}
    for record in records:
        leaf = find_leaf_label(record["labels"], reader.ancestor_labels, reader.universe)
        counts[leaf] = counts.get(leaf, 0) + record["count"]
        label_sets.setdefault(leaf, set()).add(frozenset(record["labels"]))
    return {leaf: LeafGroup(counts[leaf], frozenset(label_sets[leaf])) for leaf in counts}


def build_property_row(record: Mapping[str, Any], owner: frozenset[str]) -> PropertyRow:
    return PropertyRow(
        owner=owner,
        name=record["name"],
        types=tuple(record["types"]),
        is_mandatory=record["is_mandatory"],
    )


async def read_node_property_shapes(
    reader: GraphReader, groups: Mapping[str, LeafGroup]
) -> dict[str, tuple[PropertyShape, ...]]:
    records = await reader.client.execute_read(READ_NODE_PROPERTIES, {"universe": reader.universe})
    rows_by_leaf: dict[str, list[PropertyRow]] = {leaf: [] for leaf in groups}
    for record in records:
        leaf = find_leaf_label(record["owner"], reader.ancestor_labels, reader.universe)
        if leaf in rows_by_leaf:
            rows_by_leaf[leaf].append(build_property_row(record, frozenset(record["owner"])))
    return {
        leaf: merge_property_rows(rows, groups[leaf].label_sets)
        for leaf, rows in rows_by_leaf.items()
    }


async def read_patterns(reader: GraphReader) -> tuple[RelationshipPattern, ...]:
    records = await reader.client.execute_read(COUNT_PATTERNS, {"universe": reader.universe})
    counts: dict[tuple[str, str, str], int] = {}
    for record in records:
        key = (
            find_leaf_label(record["start_labels"], reader.ancestor_labels, reader.universe),
            record["type"],
            find_leaf_label(record["end_labels"], reader.ancestor_labels, reader.universe),
        )
        counts[key] = counts.get(key, 0) + record["count"]
    patterns = [
        RelationshipPattern(start_label=start, type=type_, end_label=end, count=count)
        for (start, type_, end), count in counts.items()
    ]
    return tuple(sorted(patterns, key=lambda p: (p.type, p.start_label, p.end_label)))


def strip_relationship_type(raw_type: str) -> str:
    """Turn the `:`HAS_TYPE`` form of `db.schema.relTypeProperties` into `HAS_TYPE`."""
    return raw_type.removeprefix(RELATIONSHIP_TYPE_PREFIX).removesuffix(RELATIONSHIP_TYPE_SUFFIX)


async def read_relationship_property_shapes(
    reader: GraphReader, types: frozenset[str]
) -> dict[str, tuple[PropertyShape, ...]]:
    records = await reader.client.execute_read(READ_RELATIONSHIP_PROPERTIES)
    rows_by_type: dict[str, list[PropertyRow]] = {type_: [] for type_ in types}
    for record in records:
        type_ = strip_relationship_type(record["owner"])
        if type_ in rows_by_type:
            rows_by_type[type_].append(build_property_row(record, frozenset({type_})))
    return {
        type_: merge_property_rows(rows, frozenset({frozenset({type_})}))
        for type_, rows in rows_by_type.items()
    }


def list_descendant_labels(leaf: str, ancestor_labels: Mapping[str, frozenset[str]]) -> list[str]:
    return sorted(label for label, ancestors in ancestor_labels.items() if leaf in ancestors)


async def sample_node_enums(
    reader: GraphReader, leaf: str, properties: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Sample the string properties of 1 leaf class, leaving out the nodes of its subclasses."""
    records = await reader.client.execute_read(
        SAMPLE_NODE_VALUES,
        {
            "universe": reader.universe,
            "label": leaf,
            "descendant_labels": list_descendant_labels(leaf, reader.ancestor_labels),
            "properties": list(properties),
            "limit": MAX_ENUM_VALUES + 1,
        },
    )
    return select_enum_values(records)


async def sample_relationship_enums(
    reader: GraphReader, type_: str, properties: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    records = await reader.client.execute_read(
        SAMPLE_RELATIONSHIP_VALUES,
        {
            "universe": reader.universe,
            "type": type_,
            "properties": list(properties),
            "limit": MAX_ENUM_VALUES + 1,
        },
    )
    return select_enum_values(records)


async def build_node_label_schema(
    reader: GraphReader, leaf: str, group: LeafGroup, shapes: tuple[PropertyShape, ...]
) -> NodeLabelSchema:
    candidates = list_enum_candidates(shapes)
    enum_values = await sample_node_enums(reader, leaf, candidates) if candidates else {}
    return NodeLabelSchema(
        label=leaf,
        count=group.count,
        ancestor_labels=tuple(sorted(reader.ancestor_labels[leaf])),
        properties=build_property_schemas(shapes, enum_values),
    )


async def build_relationship_type_schema(
    reader: GraphReader, type_: str, shapes: tuple[PropertyShape, ...]
) -> RelationshipTypeSchema:
    candidates = list_enum_candidates(shapes)
    enum_values = await sample_relationship_enums(reader, type_, candidates) if candidates else {}
    return RelationshipTypeSchema(
        type=type_, properties=build_property_schemas(shapes, enum_values)
    )


async def introspect_schema(client: GraphClient, materializer: LabelMaterializer) -> GraphSchema:
    """Read the structure of the universe of `materializer` from the graph. Reads only."""
    reader = GraphReader(client, materializer.universe_label, materializer.map_ancestor_labels())
    groups = await read_leaf_groups(reader)
    node_shapes = await read_node_property_shapes(reader, groups)
    patterns = await read_patterns(reader)
    relationship_shapes = await read_relationship_property_shapes(
        reader, frozenset(pattern.type for pattern in patterns)
    )
    node_labels = [
        await build_node_label_schema(reader, leaf, groups[leaf], node_shapes[leaf])
        for leaf in sorted(groups)
    ]
    relationship_types = [
        await build_relationship_type_schema(reader, type_, relationship_shapes[type_])
        for type_ in sorted(relationship_shapes)
    ]
    return GraphSchema(
        universe=reader.universe,
        node_labels=tuple(node_labels),
        relationship_types=tuple(relationship_types),
        patterns=patterns,
    )
