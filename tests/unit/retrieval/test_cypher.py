from collections.abc import Mapping
from pathlib import Path
from typing import Any, LiteralString

import pytest

from muon.agent.routing import UniverseScope
from muon.graph import GraphNode, GraphPath, GraphQueryError, GraphRelationship, WriteSummary
from muon.graph.protocol import Statement
from muon.retrieval.cypher import (
    MAX_RESULT_RECORDS,
    EvidenceLog,
    QueryCheckError,
    build_run_cypher_tool,
    check_query_terms,
)
from muon.retrieval.evidence import Evidence, EvidenceStatus, NodeKey
from tests.unit.retrieval.scope import build_toy_scope

ALLOWED = frozenset({"ToyWidget", "ToyColor", "muontoy", "HAS_COLOR", "PAIRS_WITH"})
WIDGET_LABELS = frozenset({"ToyWidget", "ToyGadget", "Object", "Entity", "muontoy"})
OTHER_LABELS = frozenset({"AltWidget", "Entity", "muonalt"})


class StubReadClient:
    """Answers every read with canned rows, or raises the given error."""

    def __init__(self, rows: list[dict[str, Any]], error: Exception | None = None) -> None:
        self.rows = rows
        self.error = error
        self.reads: list[tuple[str, int | None]] = []

    async def execute_read(
        self,
        query: LiteralString,
        parameters: Mapping[str, Any] | None = None,
        max_records: int | None = None,
    ) -> list[dict[str, Any]]:
        self.reads.append((query, max_records))
        if self.error is not None:
            raise self.error
        return self.rows if max_records is None else self.rows[:max_records]

    async def execute_write(
        self, query: LiteralString, parameters: Mapping[str, Any] | None = None
    ) -> WriteSummary:
        raise AssertionError("run_cypher must not write")

    async def execute_write_batch(self, statements: list[Statement]) -> list[WriteSummary]:
        raise AssertionError("run_cypher must not write")


def widget(element_id: str, widget_id: str, name: str = "Widget") -> GraphNode:
    return GraphNode(element_id, WIDGET_LABELS, {"id": widget_id, "name": name})


@pytest.fixture
async def scope(tmp_path: Path) -> UniverseScope:
    return await build_toy_scope(tmp_path)


async def call_tool(
    scope: UniverseScope,
    client: StubReadClient,
    log: EvidenceLog,
    query: str = "MATCH (n) RETURN n",
) -> tuple[str, Evidence]:
    tool = build_run_cypher_tool(scope, client, log)  # type: ignore[arg-type]
    message = await tool.ainvoke(
        {
            "type": "tool_call",
            "name": "run_cypher",
            "id": "c1",
            "args": {"query": query, "purpose": "p"},
        }
    )
    return str(message.content), message.artifact


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (w:ToyWidget:muontoy) RETURN w",
        "MATCH (w:ToyWidget)-[:HAS_COLOR|PAIRS_WITH]->(c) RETURN c",
        "MATCH (w:muontoy) WHERE w:ToyWidget RETURN w",
        "MATCH (w:muontoy {name: 'Pokemon:Thing', id: 'x'}) RETURN w",
        "MATCH (w:muontoy) RETURN w, {kind: 'a', other: 1} AS m",
        "MATCH (w:muontoy) // a :Comment\nRETURN w",
        "MATCH (w:muontoy) /* :Block */ RETURN w",
        "MATCH (w:`ToyWidget`) RETURN w",
        "MATCH (w:muontoy) RETURN [x IN [w] WHERE x:ToyWidget | x.name]",
        "MATCH (w:ToyWidget&muontoy) RETURN w",
        "MATCH (w:!ToyColor) RETURN w.x::INTEGER",
        'MATCH (w:muontoy) WHERE w.name = "a:b" RETURN w',
    ],
)
def test_valid_queries_pass_the_term_check(query: str) -> None:
    check_query_terms(query, ALLOWED)


@pytest.mark.parametrize(
    ("query", "unknown"),
    [
        ("MATCH (p:Pokemon) RETURN p", "Pokemon"),
        ("MATCH (w:muontoy)-[:LOVES]->(c) RETURN c", "LOVES"),
        ("MATCH (w:ToyWidget|Pokemon) RETURN w", "Pokemon"),
        ("MATCH (w:muontoy) WHERE w:Pokemon RETURN w", "Pokemon"),
        ("MATCH (w:muontoy)-[:HAS_COLOR|:LOVES]->(c) RETURN c", "LOVES"),
        ("MATCH (w:`Pokemon`) RETURN w", "Pokemon"),
    ],
)
def test_unknown_terms_are_rejected_by_name(query: str, unknown: str) -> None:
    with pytest.raises(QueryCheckError, match=unknown):
        check_query_terms(query, ALLOWED)


@pytest.mark.parametrize(
    "query",
    ["MATCH (n:$($label)) RETURN n", "MATCH (n:ToyWidget|$($label)) RETURN n"],
)
def test_dynamic_terms_are_rejected(query: str) -> None:
    with pytest.raises(QueryCheckError, match="Dynamic"):
        check_query_terms(query, ALLOWED)


async def test_rejected_query_runs_no_read(scope: UniverseScope) -> None:
    client = StubReadClient([])

    content, evidence = await call_tool(scope, client, EvidenceLog(), "MATCH (p:Pokemon) RETURN p")

    assert client.reads == []
    assert evidence.status is EvidenceStatus.REJECTED
    assert "Pokemon" in content


async def test_dynamic_query_is_rejected_like_an_unknown_term(scope: UniverseScope) -> None:
    client = StubReadClient([])

    _, evidence = await call_tool(scope, client, EvidenceLog(), "MATCH (n:$($x)) RETURN n")

    assert client.reads == []
    assert evidence.status is EvidenceStatus.REJECTED


async def test_more_records_than_the_cap_are_cut_and_flagged(scope: UniverseScope) -> None:
    rows = [{"i": index} for index in range(MAX_RESULT_RECORDS + 1)]
    client = StubReadClient(rows)

    content, evidence = await call_tool(scope, client, EvidenceLog())

    assert client.reads[0][1] == MAX_RESULT_RECORDS + 1
    assert evidence.row_count == MAX_RESULT_RECORDS
    assert evidence.is_truncated
    assert "truncated" in content.splitlines()[0]


async def test_120_rows_give_50_in_content_and_120_in_the_artifact(scope: UniverseScope) -> None:
    client = StubReadClient([{"i": index} for index in range(120)])

    content, evidence = await call_tool(scope, client, EvidenceLog())

    assert len(content.splitlines()) == 2 + 50
    assert len(evidence.rows) == 120
    assert not evidence.is_truncated


async def test_node_without_the_universe_label_is_rejected_and_dropped(
    scope: UniverseScope,
) -> None:
    other = GraphNode("9", OTHER_LABELS, {"id": "alt"})
    client = StubReadClient([{"n": widget("1", "w1")}, {"n": other}])

    content, evidence = await call_tool(scope, client, EvidenceLog())

    assert evidence.status is EvidenceStatus.REJECTED
    assert evidence.rows == ()
    assert "left the universe" in content


async def test_nodes_in_paths_and_lists_are_checked(scope: UniverseScope) -> None:
    other = GraphNode("9", OTHER_LABELS, {"id": "alt"})
    path = GraphPath(nodes=(widget("1", "w1"), other), relationships=())
    client = StubReadClient([{"p": path}])

    _, evidence = await call_tool(scope, client, EvidenceLog())

    assert evidence.status is EvidenceStatus.REJECTED


async def test_nodes_become_keyed_cells_and_a_slice(scope: UniverseScope) -> None:
    color = GraphNode("2", frozenset({"ToyColor", "Concept", "Entity", "muontoy"}), {"id": "red"})
    edge = GraphRelationship("r1", "HAS_COLOR", "1", "2", {"strength": "high"})
    client = StubReadClient([{"w": widget("1", "w1", "Alpha"), "c": color, "r": edge}])

    _, evidence = await call_tool(scope, client, EvidenceLog())

    assert evidence.status is EvidenceStatus.OK
    assert evidence.columns == ("w", "c", "r")
    widget_key = NodeKey(label="ToyWidget", id="w1")
    node = evidence.graph_slice.find_node(widget_key)
    assert node is not None
    assert node.name == "Alpha"
    assert evidence.graph_slice.find_node(NodeKey(label="ToyColor", id="red")) is not None
    relationship = evidence.graph_slice.relationships[0]
    assert (relationship.start, relationship.end) == (
        widget_key,
        NodeKey(label="ToyColor", id="red"),
    )


async def test_empty_result_has_status_empty(scope: UniverseScope) -> None:
    _, evidence = await call_tool(scope, StubReadClient([]), EvidenceLog())

    assert evidence.status is EvidenceStatus.EMPTY
    assert evidence.columns == ()


async def test_graph_query_error_gives_a_failed_item_with_the_database_message(
    scope: UniverseScope,
) -> None:
    error = GraphQueryError("Neo4j rejected read on database")
    error.__cause__ = RuntimeError("Invalid input 'X'")
    client = StubReadClient([], error=error)

    content, evidence = await call_tool(scope, client, EvidenceLog())

    assert evidence.status is EvidenceStatus.FAILED
    assert "Invalid input 'X'" in content


async def test_ids_follow_call_order_including_rejected_calls(scope: UniverseScope) -> None:
    log = EvidenceLog()
    client = StubReadClient([{"i": 1}])

    await call_tool(scope, client, log, "MATCH (p:Pokemon) RETURN p")
    await call_tool(scope, client, log)

    assert [item.id for item in log.items] == ["e1", "e2"]
    assert [item.status for item in log.items] == [EvidenceStatus.REJECTED, EvidenceStatus.OK]
