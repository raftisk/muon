"""The `run_cypher` tool against the live Neo4j in `.env`, on the toy universes.

The `toy_runner` fixture purges `muontoy` afterwards, and the `graph_client` fixture
removes the `MuonTest` nodes of the run.
"""

from pathlib import Path
from typing import LiteralString

import pytest

from muon.agent.routing import UniverseScope, build_scope
from muon.agent.universes import discover_universes
from muon.graph import Neo4jGraphClient
from muon.ingestion.runner import Runner
from muon.retrieval.cypher import EvidenceLog, build_run_cypher_tool
from muon.retrieval.evidence import Evidence, EvidenceStatus
from tests.fixtures.toy_ontology import ALT_UNIVERSE, TOY_UNIVERSE, copy_toy_ontology
from tests.fixtures.toy_plugin import write_toy_plugin
from tests.unit.retrieval.scope import build_toy_settings

pytestmark = pytest.mark.integration

WRITE_GRAPH: LiteralString = """
CREATE (w:ToyWidget:ToyGadget:Object:Entity:muontoy {id: 'w1', name: 'W1', serial: 1})
CREATE (c:ToyColor:Concept:Entity:muontoy {id: 'c1', name: 'C1'})
CREATE (w)-[:HAS_COLOR]->(c)
CREATE (:AltThing:Entity:MuonTest:$($alt) {id: 'a1', name: 'A1', run_id: $run_id})
"""
COUNT_ALL_NODES: LiteralString = "MATCH (n) RETURN count(n) AS total"


async def count_all_nodes(client: Neo4jGraphClient) -> int:
    return int((await client.execute_read(COUNT_ALL_NODES))[0]["total"])


@pytest.fixture
async def toy_scope(
    graph_client: Neo4jGraphClient, toy_runner: Runner, run_id: str, tmp_path: Path
) -> UniverseScope:
    await graph_client.execute_write(WRITE_GRAPH, {"run_id": run_id, "alt": ALT_UNIVERSE})
    ontology_dir = copy_toy_ontology(tmp_path / "ontology")
    universes_dir = tmp_path / "plugins"
    universes_dir.mkdir()
    write_toy_plugin(universes_dir)
    plugin = discover_universes(universes_dir, ontology_dir).get(TOY_UNIVERSE)
    settings = build_toy_settings(tmp_path, ontology_dir)
    return await build_scope(graph_client, plugin, settings)


async def run_query(
    scope: UniverseScope, client: Neo4jGraphClient, query: str
) -> tuple[str, Evidence]:
    tool = build_run_cypher_tool(scope, client, EvidenceLog())
    message = await tool.ainvoke(
        {
            "type": "tool_call",
            "name": "run_cypher",
            "id": "c1",
            "args": {"query": query, "purpose": "test"},
        }
    )
    return str(message.content), message.artifact


async def test_write_query_fails_and_the_graph_is_unchanged(
    toy_scope: UniverseScope, graph_client: Neo4jGraphClient
) -> None:
    before = await count_all_nodes(graph_client)

    _, evidence = await run_query(toy_scope, graph_client, "MATCH (n:muontoy) SET n.x = 1")

    assert evidence.status is EvidenceStatus.FAILED
    assert evidence.message
    assert await count_all_nodes(graph_client) == before


async def test_read_returns_keyed_nodes(
    toy_scope: UniverseScope, graph_client: Neo4jGraphClient
) -> None:
    query = "MATCH (w:ToyWidget:muontoy)-[:HAS_COLOR]->(c) RETURN w, c"

    content, evidence = await run_query(toy_scope, graph_client, query)

    assert evidence.status is EvidenceStatus.OK
    assert '(ToyWidget w1 "W1")' in content
    assert '(ToyColor c1 "C1")' in content


async def test_node_of_another_universe_is_rejected(
    toy_scope: UniverseScope, graph_client: Neo4jGraphClient
) -> None:
    _, evidence = await run_query(toy_scope, graph_client, "MATCH (n:MuonTest) RETURN n")

    assert evidence.status is EvidenceStatus.REJECTED
    assert evidence.rows == ()
