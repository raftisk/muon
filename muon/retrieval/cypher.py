"""The `run_cypher` tool: guardrails, the universe check on results and evidence building.

Map: `check_query_terms` rejects a query that names a label or relationship type outside
the scope. `build_run_cypher_tool` runs an accepted query in a read transaction, and
`check_universe_labels` rejects a result that holds a node of another universe. Every call,
accepted or not, becomes 1 `Evidence` item in the question's `EvidenceLog`.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, LiteralString

from langchain_core.tools import BaseTool, tool

from muon.agent.routing import UniverseScope
from muon.graph import (
    GraphClient,
    GraphNode,
    GraphPath,
    GraphQueryError,
    GraphRelationship,
)
from muon.retrieval.evidence import (
    Cell,
    Evidence,
    EvidenceStatus,
    GraphSlice,
    NodeCell,
    NodeKey,
    PathCell,
    RelationshipCell,
    SliceNode,
    SliceRelationship,
    render_evidence_content,
)
from muon.retrieval.schema import SchemaIntrospectionError, find_leaf_label

MAX_RESULT_RECORDS = 5000
TOOL_NAME = "run_cypher"
UNRESOLVED_LABEL = "Node"
NAME_PROPERTY = "name"
SOURCE_URL_PROPERTY = "sourceUrl"
ID_PROPERTY = "id"

NAME_PATTERN = r"[A-Za-z_]\w*|`(?:[^`]|``)*`"
MAP_KEY_PATTERN = re.compile(rf"([{{,]\s*)(?:{NAME_PATTERN})\s*:(?!:)")
NAME_REGEX = re.compile(NAME_PATTERN)
OPERATOR_CHARACTERS = "|&"
PREFIX_CHARACTERS = "!%"
DYNAMIC_TERM_START = "$("
NON_TERM_FOLLOWERS = ".("


class QueryCheckError(Exception):
    """A query or a result that the guardrails reject; the message goes to the model."""


def strip_comments_and_strings(query: str) -> str:
    """Replace comments and string literals with a space and keep backtick names."""
    parts: list[str] = []
    position = 0
    while position < len(query):
        character = query[position]
        if character in "'\"":
            position = skip_string(query, position)
            parts.append(" ")
        elif query.startswith("//", position):
            end = query.find("\n", position)
            position = len(query) if end == -1 else end
            parts.append(" ")
        elif query.startswith("/*", position):
            end = query.find("*/", position + 2)
            position = len(query) if end == -1 else end + 2
            parts.append(" ")
        elif character == "`":
            end = skip_backtick_name(query, position)
            parts.append(query[position:end])
            position = end
        else:
            parts.append(character)
            position += 1
    return "".join(parts)


def skip_string(query: str, start: int) -> int:
    """Return the index after the string literal that opens at `start`."""
    quote = query[start]
    position = start + 1
    while position < len(query):
        if query[position] == "\\":
            position += 2
            continue
        if query[position] == quote:
            return position + 1
        position += 1
    return len(query)


def skip_backtick_name(query: str, start: int) -> int:
    position = start + 1
    while position < len(query):
        if query[position] == "`":
            if query.startswith("``", position):
                position += 2
                continue
            return position + 1
        position += 1
    return len(query)


def unquote_name(name: str) -> str:
    return name[1:-1].replace("``", "`") if name.startswith("`") else name


def read_label_expression(text: str, start: int, terms: set[str]) -> None:
    """Add the names of the label expression at `start` to `terms`.

    Raises `QueryCheckError` on a dynamic term. Alternation and conjunction are read only
    when the operator follows a name directly, so `x:Label | x.name` in a list
    comprehension is not read as a label expression.
    """
    position = start
    while position < len(text):
        while position < len(text) and text[position] in PREFIX_CHARACTERS + "(":
            position += 1
        if text.startswith(DYNAMIC_TERM_START, position):
            raise QueryCheckError(
                "Dynamic labels and relationship types ($(...)) are not allowed; "
                "write the label or type by name"
            )
        match = NAME_REGEX.match(text, position)
        if match is None:
            return
        position = match.end()
        if text[position : position + 1] in NON_TERM_FOLLOWERS:
            return
        terms.add(unquote_name(match.group()))
        while position < len(text) and text[position] == ")":
            position += 1
        if text[position : position + 1] not in tuple(OPERATOR_CHARACTERS):
            return
        position += 1
        if text[position : position + 1] == ":":
            position += 1


def find_query_terms(query: str) -> set[str]:
    """Return every label and relationship type name that `query` uses."""
    text = MAP_KEY_PATTERN.sub(r"\1", strip_comments_and_strings(query))
    terms: set[str] = set()
    position = 0
    while (position := text.find(":", position)) != -1:
        if text.startswith("::", position):
            position += 2
            continue
        read_label_expression(text, position + 1, terms)
        position += 1
    return terms


def check_query_terms(query: str, allowed_terms: frozenset[str]) -> None:
    """Raise `QueryCheckError` when `query` uses a term outside `allowed_terms`.

    The check may miss a term, but it never rejects a query that only uses allowed terms.
    """
    unknown = sorted(find_query_terms(query) - allowed_terms)
    if unknown:
        raise QueryCheckError(
            f"Unknown label or relationship type: {', '.join(unknown)}. "
            "Use only the labels and relationship types of the schema."
        )


def iter_nodes(value: Any) -> Iterable[GraphNode]:
    """Yield every node inside a converted value, also inside paths, lists and maps."""
    if isinstance(value, GraphNode):
        yield value
    elif isinstance(value, GraphPath):
        yield from value.nodes
    elif isinstance(value, list):
        for item in value:
            yield from iter_nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_nodes(item)


def check_universe_labels(rows: list[dict[str, Any]], universe_label: str) -> None:
    """Raise `QueryCheckError` when a row holds a node without the universe label."""
    for row in rows:
        for node in iter_nodes(list(row.values())):
            if universe_label not in node.labels:
                raise QueryCheckError(
                    f"The result left the universe: a node with labels {sorted(node.labels)} "
                    f"does not carry the label {universe_label}. Start every MATCH at nodes "
                    f"labeled {universe_label}."
                )


def select_listed_properties(
    labels: Iterable[str], node_properties: Mapping[str, Any], listed: Mapping[str, tuple[str, ...]]
) -> dict[str, Any]:
    """Return the properties of a node that its labels list, with their raw values.

    The result is the union of the lists of every label the node carries, restricted to the
    properties the node has, in label order then list order.
    """
    selected: dict[str, Any] = {}
    for label in sorted(set(labels)):
        for name in listed.get(label, ()):
            if name in node_properties:
                selected.setdefault(name, node_properties[name])
    return selected


@dataclass
class SliceBuilder:
    """Collects the nodes and relationships of the rows of 1 result."""

    scope: UniverseScope
    nodes: dict[NodeKey, SliceNode] = field(default_factory=dict)
    relationships: dict[str, SliceRelationship] = field(default_factory=dict)
    keys_by_element_id: dict[str, NodeKey] = field(default_factory=dict)

    def key_node(self, node: GraphNode) -> NodeKey:
        existing = self.keys_by_element_id.get(node.element_id)
        if existing is not None:
            return existing
        leaf = find_leaf_label(
            sorted(node.labels),
            self.scope.materializer.map_ancestor_labels(),
            self.scope.universe_label,
        )
        node_id = node.properties.get(ID_PROPERTY)
        key = NodeKey(label=leaf, id=str(node_id) if node_id is not None else node.element_id)
        self.keys_by_element_id[node.element_id] = key
        name = node.properties.get(NAME_PROPERTY)
        source_url = node.properties.get(SOURCE_URL_PROPERTY)
        self.nodes[key] = SliceNode(
            key=key,
            name=name if isinstance(name, str) else None,
            labels=tuple(sorted(node.labels)),
            source_url=source_url if isinstance(source_url, str) else None,
            properties={
                property_name: self.convert_value(value)
                for property_name, value in select_listed_properties(
                    node.labels,
                    node.properties,
                    self.scope.plugin.manifest.artifact.properties,
                ).items()
            },
        )
        return key

    def key_endpoint(self, element_id: str) -> NodeKey:
        return self.keys_by_element_id.get(
            element_id, NodeKey(label=UNRESOLVED_LABEL, id=element_id)
        )

    def convert_relationship(self, relationship: GraphRelationship) -> RelationshipCell:
        start = self.key_endpoint(relationship.start_element_id)
        end = self.key_endpoint(relationship.end_element_id)
        self.relationships[relationship.element_id] = SliceRelationship(
            type=relationship.type,
            start=start,
            end=end,
            properties={
                name: self.convert_value(value) for name, value in relationship.properties.items()
            },
        )
        return RelationshipCell(type=relationship.type, start=start, end=end)

    def convert_value(self, value: Any) -> Cell:
        if isinstance(value, GraphNode):
            return NodeCell(key=self.key_node(value))
        if isinstance(value, GraphRelationship):
            return self.convert_relationship(value)
        if isinstance(value, GraphPath):
            return PathCell(
                nodes=tuple(self.key_node(node) for node in value.nodes),
                relationships=tuple(
                    self.convert_relationship(edge) for edge in value.relationships
                ),
            )
        if isinstance(value, list):
            return [self.convert_value(item) for item in value]
        if isinstance(value, dict):
            return {str(name): self.convert_value(item) for name, item in value.items()}
        if value is None or isinstance(value, bool | int | float | str):
            return value
        return str(value)

    def build_slice(self) -> GraphSlice:
        return GraphSlice(
            nodes=tuple(self.nodes.values()), relationships=tuple(self.relationships.values())
        )


def preload_node_keys(builder: SliceBuilder, rows: list[dict[str, Any]]) -> None:
    """Key every node first, so relationships can name their end nodes."""
    for row in rows:
        for node in iter_nodes(list(row.values())):
            builder.key_node(node)


@dataclass(frozen=True)
class CallDescription:
    """What 1 `run_cypher` call asked for."""

    evidence_id: str
    query: str
    purpose: str
    is_final: bool


def build_unanswered_evidence(
    call: CallDescription, scope: UniverseScope, status: EvidenceStatus, message: str
) -> Evidence:
    return Evidence(
        id=call.evidence_id,
        universe=scope.universe_label,
        tool=TOOL_NAME,
        query=call.query,
        purpose=call.purpose,
        status=status,
        message=message,
        is_final=call.is_final,
    )


def build_evidence(
    call: CallDescription, scope: UniverseScope, rows: list[dict[str, Any]]
) -> Evidence:
    """Turn the rows of an accepted query into an `Evidence` item.

    Cuts the rows to `MAX_RESULT_RECORDS` and sets `is_truncated` when there were more.
    Returns a `failed` item when a node has no single leaf class.
    """
    is_truncated = len(rows) > MAX_RESULT_RECORDS
    kept_rows = rows[:MAX_RESULT_RECORDS]
    builder = SliceBuilder(scope)
    try:
        preload_node_keys(builder, kept_rows)
        cell_rows = tuple(
            {name: builder.convert_value(value) for name, value in row.items()} for row in kept_rows
        )
    except SchemaIntrospectionError as error:
        return build_unanswered_evidence(call, scope, EvidenceStatus.FAILED, str(error))
    return Evidence(
        id=call.evidence_id,
        universe=scope.universe_label,
        tool=TOOL_NAME,
        query=call.query,
        purpose=call.purpose,
        status=EvidenceStatus.OK if cell_rows else EvidenceStatus.EMPTY,
        columns=tuple(kept_rows[0]) if kept_rows else (),
        rows=cell_rows,
        row_count=len(cell_rows),
        is_truncated=is_truncated,
        is_final=call.is_final,
        graph_slice=builder.build_slice(),
    )


class EvidenceLog:
    """The evidence items of 1 question and their id counter."""

    def __init__(self) -> None:
        self.counter = 0
        self.by_number: dict[int, Evidence] = {}

    def next_id(self) -> str:
        """Return the next id. It must run before the first `await` of a tool call."""
        self.counter += 1
        return f"e{self.counter}"

    def add(self, evidence: Evidence) -> None:
        self.by_number[int(evidence.id.removeprefix("e"))] = evidence

    @property
    def items(self) -> tuple[Evidence, ...]:
        return tuple(self.by_number[number] for number in sorted(self.by_number))


def describe_tool_failure(error: GraphQueryError) -> str:
    return str(error.__cause__ or error)


def build_run_cypher_tool(scope: UniverseScope, client: GraphClient, log: EvidenceLog) -> BaseTool:
    """Build the `run_cypher` tool for 1 question.

    The tool takes no universe parameter: the scope fixes it. A `GraphConnectionError`
    propagates and ends the run.
    """

    @tool(TOOL_NAME, response_format="content_and_artifact")
    async def run_cypher(query: str, purpose: str, is_final: bool = False) -> tuple[str, Evidence]:
        """Run 1 read-only Cypher query on the graph and return its rows.

        Args:
            query: The Cypher query, with literal values written into the text.
            purpose: One line on what the query fetches.
            is_final: true when you expect the rows of this query to answer the question in
                full. If it returns rows, retrieval ends and you get no further turn.
        """
        call = CallDescription(log.next_id(), query, purpose, is_final)
        evidence = await run_checked_query(call, scope, client)
        log.add(evidence)
        return render_evidence_content(evidence), evidence

    return run_cypher


async def run_checked_query(
    call: CallDescription, scope: UniverseScope, client: GraphClient
) -> Evidence:
    try:
        check_query_terms(call.query, scope.allowed_terms)
    except QueryCheckError as error:
        return build_unanswered_evidence(call, scope, EvidenceStatus.REJECTED, str(error))
    try:
        # Hazard: `call.query` is model text, not a literal. Typing it as LiteralString is
        # safe only because execute_read opens a read transaction, which Neo4j refuses to
        # write in, and check_query_terms and check_universe_labels bound what the query
        # may name and return. This is the only place that passes such text to the client.
        literal_query: LiteralString = call.query
        rows = await client.execute_read(literal_query, max_records=MAX_RESULT_RECORDS + 1)
    except GraphQueryError as error:
        return build_unanswered_evidence(
            call, scope, EvidenceStatus.FAILED, describe_tool_failure(error)
        )
    try:
        check_universe_labels(rows, scope.universe_label)
    except QueryCheckError as error:
        return build_unanswered_evidence(call, scope, EvidenceStatus.REJECTED, str(error))
    return build_evidence(call, scope, rows)
