import pytest

from muon.generation.artifacts import (
    MAX_GRAPH_NODES,
    MAX_TITLE_LENGTH,
    build_cited_answer,
    build_graph_artifact,
    collect_relationships,
    cut_title,
    find_hop_nodes,
)
from muon.generation.contract import (
    GRAPH_ARTIFACT_TYPE,
    TABLE_ARTIFACT_TYPE,
    CitedHandles,
    GraphData,
    TableData,
)
from muon.generation.handles import HandleTable, build_handle_table
from muon.generation.markers import MarkerRewriter
from muon.retrieval.evidence import (
    UNRESOLVED_LABEL,
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
    SliceRelationship,
)

ATTRIBUTION = "Text from the wiki, licensed under a license."
URL_1 = "https://example.org/squirtle"
URL_2 = "https://example.org/wartortle"
SQUIRTLE = NodeKey(label="Creature", id="squirtle")
WARTORTLE = NodeKey(label="Creature", id="wartortle")
STEP = NodeKey(label="Step", id="step-1")
LONE = NodeKey(label="Step", id="step-2")
FAR = NodeKey(label="Creature", id="far")
ENTRY = NodeKey(label="Entry", id="entry-1")


def build_node(key: NodeKey, name: str | None = None, **fields: object) -> SliceNode:
    values: dict[str, object] = {
        "key": key,
        "name": name,
        "labels": (key.label,),
        "source_url": None,
        **fields,
    }
    return SliceNode.model_validate(values)


def link(start: NodeKey, end: NodeKey, relationship_type: str = "NEXT") -> SliceRelationship:
    return SliceRelationship(type=relationship_type, start=start, end=end, properties={})


def build_evidence(
    rows: tuple[dict[str, Cell], ...],
    graph_slice: GraphSlice,
    evidence_id: str = "e1",
    status: EvidenceStatus = EvidenceStatus.OK,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        universe="toy",
        tool="run_cypher",
        query="MATCH (n) RETURN n",
        purpose="the creatures",
        status=status,
        columns=tuple(rows[0]) if rows else (),
        rows=rows,
        row_count=len(rows),
        graph_slice=graph_slice,
    )


def build_table(*items: Evidence, question: str = "Who evolves?") -> HandleTable:
    return build_handle_table(EvidenceBundle(universe="toy", question=question, items=items))


def cite(table: HandleTable, text: str) -> tuple[str, CitedHandles]:
    rewriter = MarkerRewriter(table)
    rewritten = rewriter.feed(text) + rewriter.finish()
    return rewritten, rewriter.cited()


def build_evolution_table() -> HandleTable:
    graph_slice = GraphSlice(
        nodes=(
            build_node(SQUIRTLE, "Squirtle", source_url=URL_1),
            build_node(WARTORTLE, "Wartortle", source_url=URL_2),
            build_node(STEP),
            build_node(LONE),
        ),
        relationships=(
            link(STEP, SQUIRTLE, "FROM"),
            link(STEP, WARTORTLE, "TO"),
            link(LONE, SQUIRTLE, "FROM"),
        ),
    )
    row: dict[str, Cell] = {"a": NodeCell(key=SQUIRTLE), "b": NodeCell(key=WARTORTLE)}
    return build_table(build_evidence((row,), graph_slice))


def test_links_are_numbered_and_the_text_ends_with_the_attribution() -> None:
    table = build_evolution_table()
    text, cited = cite(table, "Squirtle [n1] becomes Wartortle [n2].")

    answer = build_cited_answer(text, cited, table, ATTRIBUTION)

    assert [source.number for source in answer.sources] == [1, 2]
    assert answer.text == f"{text}\n\n{ATTRIBUTION}"
    assert "[[1]](" in answer.text
    assert "[[2]](" in answer.text
    assert "Sources" not in answer.text


def test_no_url_gives_no_links_no_attribution_and_no_sources() -> None:
    graph_slice = GraphSlice(nodes=(build_node(STEP, "Step"),))
    table = build_table(build_evidence(({"s": NodeCell(key=STEP)},), graph_slice))
    text, cited = cite(table, "A step [n1].")

    answer = build_cited_answer(text, cited, table, ATTRIBUTION)

    assert answer.text == "A step."
    assert answer.sources == ()


def test_a_missing_attribution_adds_nothing() -> None:
    table = build_evolution_table()
    text, cited = cite(table, "Squirtle [n1].")

    assert build_cited_answer(text, cited, table, None).text == text


def test_the_graph_holds_the_cited_nodes_and_the_node_that_links_2_of_them() -> None:
    table = build_evolution_table()
    _, cited = cite(table, "[n1] [n2]")

    graph = build_graph_artifact(cited, table)

    data = graph.artifact.data
    assert isinstance(data, GraphData)
    assert graph.artifact.type == GRAPH_ARTIFACT_TYPE
    assert graph.artifact.title == "Who evolves?"
    assert [node.key for node in data.nodes] == [
        "Creature:squirtle",
        "Creature:wartortle",
        "Step:step-1",
    ]
    step = data.nodes[2]
    assert (step.is_cited, step.handle, step.source_number) == (False, None, None)
    assert [(edge.type, edge.start, edge.end) for edge in data.edges] == [
        ("FROM", "Step:step-1", "Creature:squirtle"),
        ("TO", "Step:step-1", "Creature:wartortle"),
    ]
    assert [(node.handle, node.source_number) for node in data.nodes[:2]] == [("n1", 1), ("n2", 2)]
    assert data.sources == cited.sources
    assert data.is_truncated is False


def test_a_node_that_links_1_cited_node_is_not_in_the_graph() -> None:
    table = build_evolution_table()
    _, cited = cite(table, "[n1] [n2]")

    data = build_graph_artifact(cited, table).artifact.data

    assert isinstance(data, GraphData)
    assert "Step:step-2" not in [node.key for node in data.nodes]


def test_listed_properties_of_the_slice_node_reach_the_artifact_node() -> None:
    entry = build_node(ENTRY, properties={"method": "level-up", "level": 1})
    graph_slice = GraphSlice(nodes=(entry,))
    table = build_table(build_evidence(({"e": NodeCell(key=ENTRY)},), graph_slice))
    _, cited = cite(table, "[n1]")

    data = build_graph_artifact(cited, table).artifact.data

    assert isinstance(data, GraphData)
    assert data.nodes[0].properties == {"method": "level-up", "level": 1}


def test_70_cited_nodes_give_the_first_60_and_a_truncated_flag() -> None:
    keys = [NodeKey(label="Creature", id=f"c{index}") for index in range(70)]
    graph_slice = GraphSlice(nodes=tuple(build_node(key, key.id) for key in keys))
    row: dict[str, Cell] = {key.id: NodeCell(key=key) for key in keys}
    table = build_table(build_evidence((row,), graph_slice))
    _, cited = cite(table, " ".join(f"[n{number}]" for number in range(1, 71)))

    data = build_graph_artifact(cited, table).artifact.data

    assert isinstance(data, GraphData)
    assert [node.key for node in data.nodes] == [f"Creature:c{index}" for index in range(60)]
    assert len(data.nodes) == MAX_GRAPH_NODES
    assert data.is_truncated is True


def test_only_table_handles_give_no_graph() -> None:
    table = build_evolution_table()
    _, cited = cite(table, "All of them [t1].")

    answer = build_cited_answer("All of them.", cited, table, ATTRIBUTION)

    assert answer.graph is None
    assert len(answer.tables) == 1


def test_a_table_artifact_holds_every_row_with_node_names() -> None:
    keys = [NodeKey(label="Creature", id=f"c{index}") for index in range(80)]
    graph_slice = GraphSlice(nodes=tuple(build_node(key, key.id.upper()) for key in keys[:79]))
    rows = tuple({"creature": NodeCell(key=key), "n": index} for index, key in enumerate(keys))
    table = build_table(build_evidence(rows, graph_slice, evidence_id="e3"))
    _, cited = cite(table, "See [t3].")

    answer = build_cited_answer("See.", cited, table, ATTRIBUTION)

    data = answer.tables[0].artifact.data
    assert isinstance(data, TableData)
    assert answer.tables[0].artifact.type == TABLE_ARTIFACT_TYPE
    assert answer.tables[0].artifact.title == "the creatures"
    assert [column.key for column in data.columns] == ["creature", "n"]
    assert len(data.rows) == 80
    assert data.rows[0] == {"creature": "C0", "n": 0}
    assert data.rows[79] == {"creature": "Creature:c79", "n": 79}


def test_table_cells_keep_their_shape() -> None:
    edge = RelationshipCell(type="NEXT", start=SQUIRTLE, end=WARTORTLE)
    path = PathCell(nodes=(SQUIRTLE, WARTORTLE), relationships=(edge,))
    graph_slice = GraphSlice(nodes=(build_node(SQUIRTLE, "Squirtle"), build_node(WARTORTLE, "W")))
    row: dict[str, Cell] = {
        "edge": edge,
        "path": path,
        "list": [NodeCell(key=SQUIRTLE), 2],
        "map": {"k": NodeCell(key=WARTORTLE), "none": None},
    }
    table = build_table(build_evidence((row,), graph_slice))
    _, cited = cite(table, "[t1]")

    data = build_cited_answer("", cited, table, None).tables[0].artifact.data

    assert isinstance(data, TableData)
    assert data.rows[0] == {
        "edge": "NEXT",
        "path": '(Creature squirtle "Squirtle")-[:NEXT]-(Creature wartortle "W")',
        "list": ["Squirtle", 2],
        "map": {"k": "W", "none": None},
    }


def test_a_hop_node_keeps_the_handle_it_has_from_rendering() -> None:
    graph_slice = GraphSlice(
        nodes=(build_node(SQUIRTLE, "Squirtle"), build_node(WARTORTLE, "W"), build_node(STEP)),
        relationships=(link(STEP, SQUIRTLE), link(STEP, WARTORTLE)),
    )
    row: dict[str, Cell] = {
        "a": NodeCell(key=SQUIRTLE),
        "b": NodeCell(key=WARTORTLE),
        "s": NodeCell(key=STEP),
    }
    table = build_table(build_evidence((row,), graph_slice))
    _, cited = cite(table, "[n1][n2]")

    data = build_graph_artifact(cited, table).artifact.data

    assert isinstance(data, GraphData)
    step = data.nodes[2]
    assert (step.key, step.handle, step.is_cited) == ("Step:step-1", "n3", False)


def test_a_relationship_with_an_unresolved_end_never_shows() -> None:
    ghost = NodeKey(label=UNRESOLVED_LABEL, id="ghost")
    graph_slice = GraphSlice(
        nodes=(build_node(SQUIRTLE, "Squirtle"), build_node(WARTORTLE, "W")),
        relationships=(link(SQUIRTLE, ghost), link(ghost, WARTORTLE), link(SQUIRTLE, WARTORTLE)),
    )
    row: dict[str, Cell] = {"a": NodeCell(key=SQUIRTLE), "b": NodeCell(key=WARTORTLE)}
    table = build_table(build_evidence((row,), graph_slice))
    _, cited = cite(table, "[n1][n2]")

    data = build_graph_artifact(cited, table).artifact.data

    assert isinstance(data, GraphData)
    assert [(edge.start, edge.end) for edge in data.edges] == [
        ("Creature:squirtle", "Creature:wartortle")
    ]
    assert collect_relationships(table.bundle) == (link(SQUIRTLE, WARTORTLE),)


def test_relationships_are_deduplicated_across_items() -> None:
    graph_slice = GraphSlice(
        nodes=(build_node(SQUIRTLE, "S"), build_node(WARTORTLE, "W")),
        relationships=(link(SQUIRTLE, WARTORTLE),),
    )
    row: dict[str, Cell] = {"a": NodeCell(key=SQUIRTLE)}
    table = build_table(
        build_evidence((row,), graph_slice, "e1"), build_evidence((row,), graph_slice, "e2")
    )

    assert collect_relationships(table.bundle) == (link(SQUIRTLE, WARTORTLE),)


def test_hop_nodes_are_sorted_by_cited_neighbors_then_label_and_id() -> None:
    third = NodeKey(label="Creature", id="third")
    relationships = [
        *(link(STEP, cited_key) for cited_key in (SQUIRTLE, WARTORTLE, third)),
        *(link(LONE, cited_key) for cited_key in (SQUIRTLE, WARTORTLE)),
        *(link(ENTRY, cited_key) for cited_key in (SQUIRTLE, WARTORTLE)),
        *(link(FAR, cited_key) for cited_key in (SQUIRTLE, WARTORTLE)),
        link(NodeKey(label="Step", id="step-3"), SQUIRTLE),
    ]

    hops = find_hop_nodes([SQUIRTLE, WARTORTLE, third], relationships)

    assert hops == [STEP, FAR, ENTRY, LONE]


def test_two_builds_of_one_input_differ_only_in_artifact_ids() -> None:
    table = build_evolution_table()
    text, cited = cite(table, "[n1] [n2] [t1]")

    first = build_cited_answer(text, cited, table, ATTRIBUTION)
    second = build_cited_answer(text, cited, table, ATTRIBUTION)

    assert first.graph is not None
    assert second.graph is not None
    assert first.graph.artifact.id != second.graph.artifact.id
    first_graph = first.graph.artifact.model_copy(update={"id": "x"})
    second_graph = second.graph.artifact.model_copy(update={"id": "x"})
    assert first_graph == second_graph
    first_table = first.tables[0].artifact.model_copy(update={"id": "x"})
    second_table = second.tables[0].artifact.model_copy(update={"id": "x"})
    assert first_table == second_table


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  What\n is   this?  ", "What is this?"),
        ("x" * 100, "x" * MAX_TITLE_LENGTH),
    ],
)
def test_titles_are_collapsed_and_cut(text: str, expected: str) -> None:
    assert cut_title(text) == expected


def test_a_cited_table_missing_from_the_bundle_raises() -> None:
    table = build_evolution_table()
    cited = CitedHandles(node_keys=(), table_ids=("e9",), sources=(), warnings=())

    with pytest.raises(LookupError, match="'e9'"):
        build_cited_answer("", cited, table, None)
