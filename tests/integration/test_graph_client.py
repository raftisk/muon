import pytest
from pydantic import SecretStr

from muon.config import get_settings
from muon.graph import (
    GraphConnectionError,
    GraphNode,
    GraphPath,
    GraphQueryError,
    GraphRelationship,
    Neo4jGraphClient,
)

pytestmark = pytest.mark.integration

WRONG_PASSWORD = "wrong-password"


async def test_write_then_read_round_trip(graph_client: Neo4jGraphClient, run_id: str) -> None:
    summary = await graph_client.execute_write(
        "CREATE (:MuonTest {run_id: $run_id, name: $name})", {"run_id": run_id, "name": "x"}
    )
    records = await graph_client.execute_read(
        "MATCH (n:MuonTest {run_id: $run_id}) RETURN n.name AS name", {"run_id": run_id}
    )

    assert summary.nodes_created == 1
    assert records == [{"name": "x"}]


async def test_read_keeps_nodes_relationships_and_paths(
    graph_client: Neo4jGraphClient, run_id: str
) -> None:
    await graph_client.execute_write(
        "CREATE (:MuonTest:Extra {run_id: $run_id, name: 'a'})"
        "-[:LINKS {weight: 2}]->(:MuonTest {run_id: $run_id, name: 'b'})",
        {"run_id": run_id},
    )
    match = "MATCH p = (a:MuonTest {run_id: $run_id, name: 'a'})-[r:LINKS]->(b) "

    node_rows = await graph_client.execute_read(match + "RETURN a", {"run_id": run_id})
    edge_rows = await graph_client.execute_read(match + "RETURN r", {"run_id": run_id})
    path_rows = await graph_client.execute_read(match + "RETURN p", {"run_id": run_id})
    list_rows = await graph_client.execute_read(
        match + "RETURN collect(b) AS nodes", {"run_id": run_id}
    )

    node = node_rows[0]["a"]
    assert isinstance(node, GraphNode)
    assert node.labels == {"MuonTest", "Extra"}
    assert node.properties["name"] == "a"
    edge = edge_rows[0]["r"]
    assert isinstance(edge, GraphRelationship)
    assert edge.type == "LINKS"
    assert edge.properties == {"weight": 2}
    path = path_rows[0]["p"]
    assert isinstance(path, GraphPath)
    assert [item.properties["name"] for item in path.nodes] == ["a", "b"]
    assert [item.properties["name"] for item in list_rows[0]["nodes"]] == ["b"]


async def test_read_stops_at_max_records(graph_client: Neo4jGraphClient, run_id: str) -> None:
    await graph_client.execute_write(
        "UNWIND range(1, 3) AS i CREATE (:MuonTest {run_id: $run_id, i: i})", {"run_id": run_id}
    )

    records = await graph_client.execute_read(
        "MATCH (n:MuonTest {run_id: $run_id}) RETURN n.i AS i", {"run_id": run_id}, max_records=2
    )

    assert len(records) == 2


async def test_read_no_match_returns_empty_list(
    graph_client: Neo4jGraphClient, run_id: str
) -> None:
    records = await graph_client.execute_read(
        "MATCH (n:MuonTest {run_id: $run_id}) RETURN n", {"run_id": run_id}
    )

    assert records == []


async def test_bad_cypher_raises_query_error(graph_client: Neo4jGraphClient) -> None:
    with pytest.raises(GraphQueryError):
        await graph_client.execute_read("MATCH (n RETURN n")


@pytest.mark.usefixtures("name_constraint")
async def test_constraint_violation_rolls_back(graph_client: Neo4jGraphClient, run_id: str) -> None:
    with pytest.raises(GraphQueryError):
        await graph_client.execute_write(
            "CREATE (:MuonTest {run_id: $run_id, name: $name}), "
            "(:MuonTest {run_id: $run_id, name: $name})",
            {"run_id": run_id, "name": f"duplicate-{run_id}"},
        )

    records = await graph_client.execute_read(
        "MATCH (n:MuonTest {run_id: $run_id}) RETURN count(n) AS total", {"run_id": run_id}
    )
    assert records == [{"total": 0}]


async def test_bad_credentials_raise_connection_error() -> None:
    settings = get_settings().neo4j
    wrong_settings = settings.model_copy(update={"password": SecretStr(WRONG_PASSWORD)})

    with pytest.raises(GraphConnectionError):
        async with Neo4jGraphClient(wrong_settings):
            pass
