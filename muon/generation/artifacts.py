"""The answer builder: sources, the attribution line, the graph artifact and table artifacts.

Map: `build_cited_answer` takes the rewritten text, what the rewriter cited and the handle
table, and returns a `CitedAnswer`. `build_graph_artifact` draws the cited nodes plus the
nodes that link 2 of them (`collect_relationships`, `find_hop_nodes`, `cap_nodes`).
`build_table_artifact` turns 1 evidence item into a table (`convert_table_cell`).
"""

import uuid
from collections.abc import Mapping, Sequence

from muon.generation.contract import (
    ARTIFACT_ID_PREFIX,
    GRAPH_ARTIFACT_TYPE,
    TABLE_ARTIFACT_TYPE,
    Artifact,
    ArtifactEdge,
    ArtifactEnvelope,
    ArtifactNode,
    CitedAnswer,
    CitedHandles,
    GraphData,
    Source,
    TableCell,
    TableColumn,
    TableData,
)
from muon.generation.handles import HandleTable
from muon.retrieval.evidence import (
    UNRESOLVED_LABEL,
    Cell,
    Evidence,
    EvidenceBundle,
    GraphSlice,
    NodeCell,
    NodeKey,
    PathCell,
    RelationshipCell,
    SliceNode,
    SliceRelationship,
    render_cell,
)

MAX_GRAPH_NODES = 60
MAX_TITLE_LENGTH = 60
ATTRIBUTION_SEPARATOR = "\n\n"
KEY_SEPARATOR = ":"


def new_artifact_id() -> str:
    return ARTIFACT_ID_PREFIX + uuid.uuid4().hex


def cut_title(text: str) -> str:
    """Collapse the whitespace of `text` and cut it to `MAX_TITLE_LENGTH` characters."""
    return " ".join(text.split())[:MAX_TITLE_LENGTH]


def format_node_key(key: NodeKey) -> str:
    return f"{key.label}{KEY_SEPARATOR}{key.id}"


def render_attribution(sources: Sequence[Source], attribution: str | None) -> str:
    """Return the attribution line behind a blank line, or empty text when no link is shown."""
    if not sources or not attribution:
        return ""
    return ATTRIBUTION_SEPARATOR + attribution


def build_cited_answer(
    text: str, cited: CitedHandles, table: HandleTable, attribution: str | None
) -> CitedAnswer:
    """Build the answer from the rewritten `text` and the handles that it cited."""
    graph = build_graph_artifact(cited, table) if cited.node_keys else None
    tables = tuple(
        build_table_artifact(get_evidence(table.bundle, evidence_id))
        for evidence_id in cited.table_ids
    )
    return CitedAnswer(
        text=text + render_attribution(cited.sources, attribution),
        sources=cited.sources,
        graph=graph,
        tables=tables,
        cited=cited,
        warnings=cited.warnings,
    )


def get_evidence(bundle: EvidenceBundle, evidence_id: str) -> Evidence:
    evidence = bundle.get(evidence_id)
    if evidence is None:
        raise LookupError(f"Cited table {evidence_id!r} is not in the evidence bundle")
    return evidence


def collect_relationships(bundle: EvidenceBundle) -> tuple[SliceRelationship, ...]:
    """Return the slice relationships of every item, once per start, type and end.

    A relationship with an end that no row returned as a node is dropped.
    """
    unique: dict[tuple[NodeKey, str, NodeKey], SliceRelationship] = {}
    for evidence in bundle.items:
        for relationship in evidence.graph_slice.relationships:
            ends = (relationship.start, relationship.end)
            if any(end.label == UNRESOLVED_LABEL for end in ends):
                continue
            unique.setdefault(
                (relationship.start, relationship.type, relationship.end), relationship
            )
    return tuple(unique.values())


def find_hop_nodes(
    cited_keys: Sequence[NodeKey], relationships: Sequence[SliceRelationship]
) -> list[NodeKey]:
    """Return the uncited nodes linked to at least 2 distinct cited nodes.

    The most linked come first, then by label and id.
    """
    cited = set(cited_keys)
    neighbors: dict[NodeKey, set[NodeKey]] = {}
    for relationship in relationships:
        for node, other in (
            (relationship.start, relationship.end),
            (relationship.end, relationship.start),
        ):
            if node not in cited and other in cited:
                neighbors.setdefault(node, set()).add(other)
    linking = [key for key, linked in neighbors.items() if len(linked) >= 2]
    return sorted(linking, key=lambda key: (-len(neighbors[key]), key.label, key.id))


def cap_nodes(keys: Sequence[NodeKey]) -> tuple[list[NodeKey], bool]:
    """Keep the first `MAX_GRAPH_NODES` keys and say whether any were dropped."""
    return list(keys[:MAX_GRAPH_NODES]), len(keys) > MAX_GRAPH_NODES


def index_slice_nodes(bundle: EvidenceBundle) -> dict[NodeKey, SliceNode]:
    """Map each node key to its first slice node in bundle order."""
    nodes: dict[NodeKey, SliceNode] = {}
    for evidence in bundle.items:
        for node in evidence.graph_slice.nodes:
            nodes.setdefault(node.key, node)
    return nodes


def build_artifact_node(
    key: NodeKey,
    slice_node: SliceNode | None,
    table: HandleTable,
    numbers_by_url: Mapping[str, int],
    is_cited: bool,
) -> ArtifactNode:
    url = slice_node.source_url if slice_node else None
    return ArtifactNode(
        key=format_node_key(key),
        handle=table.handles_by_key.get(key),
        label=key.label,
        name=slice_node.name if slice_node else None,
        is_cited=is_cited,
        source_number=numbers_by_url.get(url) if url else None,
        properties=dict(slice_node.properties) if slice_node else {},
    )


def build_graph_artifact(cited: CitedHandles, table: HandleTable) -> ArtifactEnvelope:
    """Build the graph artifact: the cited nodes, the nodes that link 2 of them and their edges."""
    bundle = table.bundle
    relationships = collect_relationships(bundle)
    slice_nodes = index_slice_nodes(bundle)
    hop_keys = [key for key in find_hop_nodes(cited.node_keys, relationships) if key in slice_nodes]
    kept, is_truncated = cap_nodes([*cited.node_keys, *hop_keys])
    kept_set = set(kept)
    cited_set = set(cited.node_keys)
    numbers_by_url = {source.url: source.number for source in cited.sources}
    nodes = tuple(
        build_artifact_node(key, slice_nodes.get(key), table, numbers_by_url, key in cited_set)
        for key in kept
    )
    edges = tuple(
        ArtifactEdge(
            type=relationship.type,
            start=format_node_key(relationship.start),
            end=format_node_key(relationship.end),
            properties=dict(relationship.properties),
        )
        for relationship in relationships
        if relationship.start in kept_set and relationship.end in kept_set
    )
    data = GraphData(nodes=nodes, edges=edges, sources=cited.sources, is_truncated=is_truncated)
    artifact = Artifact(
        id=new_artifact_id(),
        type=GRAPH_ARTIFACT_TYPE,
        title=cut_title(bundle.question),
        data=data,
    )
    return ArtifactEnvelope(artifact=artifact)


def convert_table_cell(cell: Cell, graph_slice: GraphSlice) -> TableCell:
    """Turn a cell into plain JSON: a node into its name, a relationship into its type."""
    if isinstance(cell, NodeCell):
        node = graph_slice.find_node(cell.key)
        return node.name if node and node.name is not None else format_node_key(cell.key)
    if isinstance(cell, RelationshipCell):
        return cell.type
    if isinstance(cell, PathCell):
        return render_cell(cell, graph_slice)
    if isinstance(cell, list):
        return [convert_table_cell(entry, graph_slice) for entry in cell]
    if isinstance(cell, dict):
        return {name: convert_table_cell(entry, graph_slice) for name, entry in cell.items()}
    return cell


def build_table_artifact(evidence: Evidence) -> ArtifactEnvelope:
    """Build the table artifact of 1 evidence item, with every row the item holds."""
    columns = tuple(TableColumn(key=name, label=name) for name in evidence.columns)
    rows = tuple(
        {name: convert_table_cell(row.get(name), evidence.graph_slice) for name in evidence.columns}
        for row in evidence.rows
    )
    artifact = Artifact(
        id=new_artifact_id(),
        type=TABLE_ARTIFACT_TYPE,
        title=cut_title(evidence.purpose),
        data=TableData(columns=columns, rows=rows),
    )
    return ArtifactEnvelope(artifact=artifact)
