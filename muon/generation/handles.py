"""Citation handles: short names for the nodes and tables the generator reads.

Map: `build_handle_table` walks a bundle and gives each node a handle (`n1`, `n2`, ...) and
each answerable evidence item a table handle (`t3` for `e3`). The walk takes the evidence
items in bundle order, the first `MAX_CONTENT_ROWS` rows of each, the columns in column
order, and each cell depth first with path nodes in path order. A node gets its handle at
its first sighting. Nodes seen only in later rows or only in slice relationships get none.
`render_bundle_with_handles` writes the evidence text the generator reads, with handles in
place of node ids.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from muon.retrieval.evidence import (
    MAX_CONTENT_ROWS,
    Cell,
    EvidenceBundle,
    EvidenceStatus,
    GraphSlice,
    NodeCell,
    NodeKey,
    NodeRenderer,
    PathCell,
    describe_row_count,
    render_evidence_content,
    render_evidence_rows,
)

NODE_HANDLE_PREFIX = "n"
TABLE_HANDLE_PREFIX = "t"
EVIDENCE_ID_PREFIX = "e"
ITEM_SEPARATOR = "\n\n"
REJECTED_STATUSES = (EvidenceStatus.REJECTED, EvidenceStatus.FAILED)


@dataclass(frozen=True)
class HandleNode:
    """A node that has a handle, with what the rewriter needs to cite it."""

    handle: str
    key: NodeKey
    name: str | None
    source_url: str | None


@dataclass(frozen=True)
class HandleTable:
    """The handles of 1 bundle.

    `tables` maps a table handle to its evidence id. `page_titles` maps a page URL to the
    name of the first named slice node with that URL.
    """

    bundle: EvidenceBundle
    nodes: Mapping[str, HandleNode]
    handles_by_key: Mapping[NodeKey, str]
    tables: Mapping[str, str]
    page_titles: Mapping[str, str]


def iter_cell_keys(cell: Cell) -> Iterator[NodeKey]:
    """Yield the node keys of a cell depth first; relationship ends are skipped."""
    if isinstance(cell, NodeCell):
        yield cell.key
    elif isinstance(cell, PathCell):
        yield from cell.nodes
    elif isinstance(cell, list):
        for entry in cell:
            yield from iter_cell_keys(entry)
    elif isinstance(cell, dict):
        for entry in cell.values():
            yield from iter_cell_keys(entry)


def build_table_handle(evidence_id: str) -> str:
    return TABLE_HANDLE_PREFIX + evidence_id.removeprefix(EVIDENCE_ID_PREFIX)


def build_handle_table(bundle: EvidenceBundle) -> HandleTable:
    """Assign handles to the nodes and tables of `bundle`, the same way for the same bundle."""
    nodes: dict[str, HandleNode] = {}
    handles_by_key: dict[NodeKey, str] = {}
    tables: dict[str, str] = {}
    page_titles: dict[str, str] = {}
    for evidence in bundle.items:
        if evidence.status is EvidenceStatus.OK and evidence.rows:
            tables[build_table_handle(evidence.id)] = evidence.id
        for row in evidence.rows[:MAX_CONTENT_ROWS]:
            for column in evidence.columns:
                for key in iter_cell_keys(row.get(column)):
                    if key in handles_by_key:
                        continue
                    handle = f"{NODE_HANDLE_PREFIX}{len(nodes) + 1}"
                    slice_node = evidence.graph_slice.find_node(key)
                    nodes[handle] = HandleNode(
                        handle=handle,
                        key=key,
                        name=slice_node.name if slice_node else None,
                        source_url=slice_node.source_url if slice_node else None,
                    )
                    handles_by_key[key] = handle
        for slice_node in evidence.graph_slice.nodes:
            if slice_node.source_url and slice_node.name:
                page_titles.setdefault(slice_node.source_url, slice_node.name)
    return HandleTable(
        bundle=bundle,
        nodes=nodes,
        handles_by_key=handles_by_key,
        tables=tables,
        page_titles=page_titles,
    )


def build_node_renderer(table: HandleTable) -> NodeRenderer:
    """Return a node renderer that writes `(Class "name") [n4]` and never a node id."""

    def render_handle_node(key: NodeKey, graph_slice: GraphSlice) -> str:
        slice_node = graph_slice.find_node(key)
        name = f' "{slice_node.name}"' if slice_node and slice_node.name is not None else ""
        handle = table.handles_by_key.get(key)
        marker = f" [{handle}]" if handle else ""
        return f"({key.label}{name}){marker}"

    return render_handle_node


def render_bundle_with_handles(table: HandleTable) -> str:
    """Return the evidence text of the generator: `render_evidence_content` with handles."""
    node_renderer = build_node_renderer(table)
    sections = []
    for evidence in table.bundle.items:
        if evidence.status in REJECTED_STATUSES:
            sections.append(render_evidence_content(evidence))
            continue
        table_handle = build_table_handle(evidence.id)
        handle_text = f" [{table_handle}]" if table_handle in table.tables else ""
        header = (
            f"{evidence.id}{handle_text} {evidence.status}, "
            f"{describe_row_count(evidence)}: {evidence.purpose}"
        )
        sections.append("\n".join([header, *render_evidence_rows(evidence, node_renderer)]))
    return ITEM_SEPARATOR.join(sections)
