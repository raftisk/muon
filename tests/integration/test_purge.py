from pathlib import Path
from typing import LiteralString

import pytest

from muon.graph import Neo4jGraphClient
from muon.ingestion.runner import Runner
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model
from tests.fixtures.toy_ontology import ALT_UNIVERSE, TOY_UNIVERSE, copy_toy_ontology

pytestmark = pytest.mark.integration

# Two toy nodes joined by an edge, and an alt node joined to one of them.
CREATE_NODES: LiteralString = """
CREATE (a:$(['MuonTest', $toy_universe]) {run_id: $run_id, id: 'a'})
  -[:LINKS_TO]->(b:$(['MuonTest', $toy_universe]) {run_id: $run_id, id: 'b'})
CREATE (b)-[:LINKS_TO]->(c:$(['MuonTest', $alt_universe]) {run_id: $run_id, id: 'c'})
"""

COUNT_UNIVERSE_NODES: LiteralString = "MATCH (n:$($universe)) RETURN count(n) AS total"

COUNT_RUN_NODES: LiteralString = "MATCH (n:$($universe) {run_id: $run_id}) RETURN count(n) AS total"


@pytest.fixture
def toy_runner(graph_client: Neo4jGraphClient, tmp_path: Path) -> Runner:
    model = load_ontology_model(copy_toy_ontology(tmp_path), TOY_UNIVERSE)
    return Runner(graph_client, LabelMaterializer(model))


async def count_nodes(
    client: Neo4jGraphClient, query: LiteralString, parameters: dict[str, str]
) -> int:
    records = await client.execute_read(query, parameters)
    return int(records[0]["total"])


async def test_purge_removes_one_universe(
    toy_runner: Runner, graph_client: Neo4jGraphClient, run_id: str
) -> None:
    await graph_client.execute_write(
        CREATE_NODES,
        {"toy_universe": TOY_UNIVERSE, "alt_universe": ALT_UNIVERSE, "run_id": run_id},
    )
    toy_nodes_before = await count_nodes(
        graph_client, COUNT_UNIVERSE_NODES, {"universe": TOY_UNIVERSE}
    )

    summary = await toy_runner.purge()

    toy_nodes_after = await count_nodes(
        graph_client, COUNT_UNIVERSE_NODES, {"universe": TOY_UNIVERSE}
    )
    alt_nodes_after = await count_nodes(
        graph_client, COUNT_RUN_NODES, {"universe": ALT_UNIVERSE, "run_id": run_id}
    )
    assert summary.nodes_deleted == toy_nodes_before
    assert summary.relationships_deleted >= 2
    assert toy_nodes_after == 0
    assert alt_nodes_after == 1
