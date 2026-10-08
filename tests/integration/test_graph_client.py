import pytest
from pydantic import SecretStr

from muon.config import get_settings
from muon.graph import GraphConnectionError, GraphQueryError, Neo4jGraphClient

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
