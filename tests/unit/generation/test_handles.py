from muon.generation.handles import build_handle_table, iter_cell_keys, render_bundle_with_handles
from muon.retrieval.evidence import (
    MAX_CONTENT_ROWS,
    Cell,
    Evidence,
    EvidenceBundle,
    EvidenceStatus,
    GraphSlice,
    NodeCell,
    NodeKey,
    PathCell,
    RelationshipCell,
    SliceNode,
    render_evidence_content,
)

LAPRAS = NodeKey(label="Creature", id="lapras-id")
SURF = NodeKey(label="Attack", id="surf-id")
NAMELESS = NodeKey(label="Entry", id="entry-id")
LAPRAS_URL = "https://example.org/lapras"


def build_slice(*keys: NodeKey) -> GraphSlice:
    names = {LAPRAS: "Lapras", SURF: "Surf", NAMELESS: None}
    urls = {LAPRAS: LAPRAS_URL}
    return GraphSlice(
        nodes=tuple(
            SliceNode(
                key=key,
                name=names.get(key, key.id.upper()),
                labels=(key.label,),
                source_url=urls.get(key),
            )
            for key in keys
        )
    )


def build_item(
    item_id: str,
    rows: tuple[dict[str, Cell], ...],
    keys: tuple[NodeKey, ...] = (),
    status: EvidenceStatus = EvidenceStatus.OK,
) -> Evidence:
    return Evidence(
        id=item_id,
        universe="toy",
        tool="run_cypher",
        query="MATCH (n) RETURN n",
        purpose=f"purpose of {item_id}",
        status=status,
        message="no good" if status in (EvidenceStatus.REJECTED, EvidenceStatus.FAILED) else None,
        columns=tuple(rows[0]) if rows else (),
        rows=rows,
        row_count=len(rows),
        graph_slice=build_slice(*keys),
    )


def build_bundle(*items: Evidence) -> EvidenceBundle:
    return EvidenceBundle(universe="toy", question="q", items=items)


def node_row(*keys: NodeKey) -> dict[str, Cell]:
    return {f"c{index}": NodeCell(key=key) for index, key in enumerate(keys)}


def test_a_node_keeps_one_handle_across_items() -> None:
    first = build_item("e1", (node_row(LAPRAS),), (LAPRAS,))
    second = build_item("e2", (node_row(SURF, LAPRAS),), (SURF, LAPRAS))
    table = build_handle_table(build_bundle(first, second))

    assert table.handles_by_key == {LAPRAS: "n1", SURF: "n2"}
    text = render_bundle_with_handles(table)
    assert text.count('(Creature "Lapras") [n1]') == 2
    assert '(Attack "Surf") [n2]' in text


def test_80_rows_give_a_table_handle_and_handles_for_the_first_50_rows_only() -> None:
    rows = tuple(node_row(NodeKey(label="Creature", id=f"id-{index}")) for index in range(80))
    keys = tuple(NodeKey(label="Creature", id=f"id-{index}") for index in range(80))
    table = build_handle_table(build_bundle(build_item("e3", rows, keys)))

    text = render_bundle_with_handles(table)

    assert text.splitlines()[0] == "e3 [t3] ok, 80 rows, first 50 shown: purpose of e3"
    assert table.tables == {"t3": "e3"}
    assert len(table.nodes) == MAX_CONTENT_ROWS
    assert NodeKey(label="Creature", id="id-50") not in table.handles_by_key
    assert "id-" not in text


def test_the_same_bundle_gives_equal_tables() -> None:
    bundle = build_bundle(build_item("e1", (node_row(LAPRAS, SURF),), (LAPRAS, SURF)))

    assert build_handle_table(bundle) == build_handle_table(bundle)


def test_empty_rejected_and_failed_items_get_no_table_handle() -> None:
    bundle = build_bundle(
        build_item("e1", (), status=EvidenceStatus.EMPTY),
        build_item("e2", (), status=EvidenceStatus.REJECTED),
        build_item("e3", (), status=EvidenceStatus.FAILED),
    )

    table = build_handle_table(bundle)

    assert table.tables == {}
    text = render_bundle_with_handles(table)
    assert "e2 rejected: purpose of e2\nno good" in text
    assert "[t" not in text


def test_a_node_without_a_name_renders_the_class_and_the_handle() -> None:
    item = build_item("e1", (node_row(LAPRAS, SURF, NAMELESS),), (LAPRAS, SURF, NAMELESS))
    slice_without_name = item.graph_slice.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"name": None}) if node.key == NAMELESS else node
                for node in item.graph_slice.nodes
            )
        }
    )
    table = build_handle_table(
        build_bundle(item.model_copy(update={"graph_slice": slice_without_name}))
    )

    assert "(Entry) [n3]" in render_bundle_with_handles(table)


def test_nodes_in_paths_lists_and_maps_are_walked_in_order_and_relationship_ends_skipped() -> None:
    edge = RelationshipCell(type="KNOWS", start=LAPRAS, end=SURF)
    path = PathCell(nodes=(LAPRAS, SURF), relationships=(edge,))
    other = NodeKey(label="Creature", id="other")
    cells: list[Cell] = [edge, path, [NodeCell(key=other)], {"k": NodeCell(key=NAMELESS)}]

    keys = [key for cell in cells for key in iter_cell_keys(cell)]

    assert keys == [LAPRAS, SURF, other, NAMELESS]


def test_a_node_only_in_a_relationship_gets_no_handle() -> None:
    edge = RelationshipCell(type="KNOWS", start=LAPRAS, end=SURF)
    item = build_item("e1", ({"r": edge, "n": NodeCell(key=LAPRAS)},), (LAPRAS, SURF))

    table = build_handle_table(build_bundle(item))

    assert table.handles_by_key == {LAPRAS: "n1"}


def test_page_titles_take_the_first_named_node_with_the_url() -> None:
    item = build_item("e1", (node_row(LAPRAS),), (LAPRAS,))

    table = build_handle_table(build_bundle(item))

    assert table.page_titles == {LAPRAS_URL: "Lapras"}
    assert table.nodes["n1"].source_url == LAPRAS_URL


def test_the_retrieval_rendering_is_unchanged() -> None:
    item = build_item("e1", (node_row(LAPRAS),), (LAPRAS,))

    assert render_evidence_content(item).splitlines()[-1] == '(Creature lapras-id "Lapras")'
