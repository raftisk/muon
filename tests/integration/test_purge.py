from typing import LiteralString

import pytest

from muon.graph import Neo4jGraphClient
from muon.ingestion.runner import Runner
from tests.fixtures.toy_ontology import ALT_UNIVERSE, TOY_UNIVERSE

pytestmark = pytest.mark.integration

# Two toy nodes joined by an edge, and an alt node joined to one of them.
CREATE_NODES: LiteralString = """
CREATE (a:$(['MuonTest', $toy_universe]) {run_id: $run_id, id: 'a'})
  -[:LINKS_TO]->(b:$(['MuonTest', $toy_universe]) {run_id: $run_id, id: 'b'})
CREATE (b)-[:LINKS_TO]->(c:$(['MuonTest', $alt_universe]) {run_id: $run_id, id: 'c'})
"""

CREATE_FIVE_NODES: LiteralString = """
UNWIND range(1, 5) AS number
CREATE (:$(['MuonTest', $toy_universe]) {run_id: $run_id, id: toString(number)})
"""

COUNT_UNIVERSE_NODES: LiteralString = "MATCH (n:$($universe)) RETURN count(n) AS total"

COUNT_RUN_NODES: LiteralString = "MATCH (n:$($universe) {run_id: $run_id}) RETURN count(n) AS total"


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


async def test_purge_in_batches_smaller_than_the_universe(
    toy_runner: Runner, graph_client: Neo4jGraphClient, run_id: str
) -> None:
    await graph_client.execute_write(
        CREATE_FIVE_NODES, {"toy_universe": TOY_UNIVERSE, "run_id": run_id}
    )

    summary = await toy_runner.purge(batch_size=2)

    nodes_after = await count_nodes(graph_client, COUNT_UNIVERSE_NODES, {"universe": TOY_UNIVERSE})
    assert summary.nodes_deleted == 5
    assert nodes_after == 0
