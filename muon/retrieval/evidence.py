"""Evidence of 1 question: what each `run_cypher` call returned, and how the model sees it.

Map: an `Evidence` item holds the full rows (cells that point at nodes by `NodeKey`) and a
`GraphSlice` with the names, labels and source links of the nodes and relationships those
rows mention. `render_evidence_content` cuts an item to what the model reads: the first
`MAX_CONTENT_ROWS` rows with nodes shown as `(Leaf id "name")`. The full rows reach the
user through the renderer. Every model is serializable, so an `EvidenceBundle` is the
provenance record of an answer.
"""

import json
from collections.abc import Callable
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_CONTENT_ROWS = 50
CELL_SEPARATOR = " | "
LIST_SEPARATOR = ", "


class EvidenceModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NodeKey(EvidenceModel):
    """The leaf class and `id` of a node, unique inside 1 universe."""

    label: str
    id: str


class NodeCell(EvidenceModel):
    kind: Literal["node"] = "node"
    key: NodeKey


class RelationshipCell(EvidenceModel):
    kind: Literal["relationship"] = "relationship"
    type: str
    start: NodeKey
    end: NodeKey


class PathCell(EvidenceModel):
    """The nodes of a path in order, and the relationships between them."""

    kind: Literal["path"] = "path"
    nodes: tuple[NodeKey, ...]
    relationships: tuple[RelationshipCell, ...]


type Cell = (
    NodeCell
    | RelationshipCell
    | PathCell
    | list[Cell]
    | dict[str, Cell]
    | str
    | int
    | float
    | bool
    | None
)


class SliceNode(EvidenceModel):
    """A node of the slice. `properties` holds only the plugin's listed properties."""

    key: NodeKey
    name: str | None
    labels: tuple[str, ...]
    source_url: str | None
    properties: dict[str, Cell] = Field(default_factory=dict)


class SliceRelationship(EvidenceModel):
    type: str
    start: NodeKey
    end: NodeKey
    properties: dict[str, Cell]


class GraphSlice(EvidenceModel):
    """The nodes and relationships that the rows of 1 evidence item mention."""

    nodes: tuple[SliceNode, ...] = ()
    relationships: tuple[SliceRelationship, ...] = ()

    def find_node(self, key: NodeKey) -> SliceNode | None:
        return next((node for node in self.nodes if node.key == key), None)


class EvidenceStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    REJECTED = "rejected"
    FAILED = "failed"


class Evidence(EvidenceModel):
    """One `run_cypher` call: its query, outcome, rows and graph slice, in 1 universe."""

    id: str
    universe: str
    tool: str
    query: str
    purpose: str
    status: EvidenceStatus
    message: str | None = None
    columns: tuple[str, ...] = ()
    rows: tuple[dict[str, Cell], ...] = ()
    row_count: int = 0
    is_truncated: bool = False
    is_final: bool = False
    graph_slice: GraphSlice = Field(default_factory=GraphSlice)


class EvidenceBundle(EvidenceModel):
    """The evidence items of 1 question, in id order."""

    universe: str
    question: str
    items: tuple[Evidence, ...] = ()

    def get(self, evidence_id: str) -> Evidence | None:
        return next((item for item in self.items if item.id == evidence_id), None)


def render_node(key: NodeKey, graph_slice: GraphSlice) -> str:
    node = graph_slice.find_node(key)
    name = f' "{node.name}"' if node is not None and node.name is not None else ""
    return f"({key.label} {key.id}{name})"


type NodeRenderer = Callable[[NodeKey, GraphSlice], str]


def render_relationship(cell: RelationshipCell) -> str:
    return f"[:{cell.type}]"


def render_cell(
    cell: Cell, graph_slice: GraphSlice, node_renderer: NodeRenderer = render_node
) -> str:
    """Show a cell as the model reads it: nodes through `node_renderer`, scalars as JSON."""
    if isinstance(cell, NodeCell):
        return node_renderer(cell.key, graph_slice)
    if isinstance(cell, RelationshipCell):
        return render_relationship(cell)
    if isinstance(cell, PathCell):
        parts = [node_renderer(cell.nodes[0], graph_slice)] if cell.nodes else []
        for relationship, node in zip(cell.relationships, cell.nodes[1:], strict=False):
            parts.append(f"-{render_relationship(relationship)}-")
            parts.append(node_renderer(node, graph_slice))
        return "".join(parts)
    if isinstance(cell, list):
        items = (render_cell(item, graph_slice, node_renderer) for item in cell)
        return "[" + LIST_SEPARATOR.join(items) + "]"
    if isinstance(cell, dict):
        entries = (
            f"{json.dumps(key)}: {render_cell(item, graph_slice, node_renderer)}"
            for key, item in cell.items()
        )
        return "{" + LIST_SEPARATOR.join(entries) + "}"
    return json.dumps(cell, ensure_ascii=False)


def describe_row_count(evidence: Evidence) -> str:
    noun = "row" if evidence.row_count == 1 else "rows"
    parts = [f"{evidence.row_count} {noun}"]
    if evidence.is_truncated:
        parts.append("truncated at the record limit")
    if evidence.row_count > MAX_CONTENT_ROWS:
        parts.append(f"first {MAX_CONTENT_ROWS} shown")
    return ", ".join(parts)


def render_evidence_rows(
    evidence: Evidence, node_renderer: NodeRenderer = render_node
) -> list[str]:
    """Return the columns line and the first `MAX_CONTENT_ROWS` rows of an item, as text lines."""
    lines = []
    if evidence.columns:
        lines.append("columns: " + LIST_SEPARATOR.join(evidence.columns))
    for row in evidence.rows[:MAX_CONTENT_ROWS]:
        cells = (
            render_cell(row.get(column), evidence.graph_slice, node_renderer)
            for column in evidence.columns
        )
        lines.append(CELL_SEPARATOR.join(cells))
    return lines


def render_evidence_content(evidence: Evidence) -> str:
    """Return the text the model reads for 1 evidence item.

    A rejected or failed item shows its message and no rows. Any other item shows its row
    count, its columns and at most `MAX_CONTENT_ROWS` rows.
    """
    if evidence.status in (EvidenceStatus.REJECTED, EvidenceStatus.FAILED):
        return f"{evidence.id} {evidence.status}: {evidence.purpose}\n{evidence.message}"
    header = f"{evidence.id} {evidence.status}, {describe_row_count(evidence)}: {evidence.purpose}"
    return "\n".join([header, *render_evidence_rows(evidence)])
