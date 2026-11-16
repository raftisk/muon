import pytest

from muon.graph import Neo4jGraphClient
from muon.graph.indexes import ALIASES_FIELD, NAME_INDEX, NAME_INDEX_LABELS
from muon.ingestion.schema import ensure_name_index

pytestmark = pytest.mark.integration

SHOW_NAME_INDEX = "SHOW INDEXES YIELD name, state, labelsOrTypes, properties WHERE name = $name"
QUERY_NAMES = (
    "CALL db.index.fulltext.queryNodes($index, $text) YIELD node, score "
    "RETURN node.id AS id, score ORDER BY score DESC"
)
ALIAS_QUERY = "carchacrok"
ALIAS_OWNER_ID = "garchomp"


async def show_index(client: Neo4jGraphClient) -> list[dict[str, object]]:
    return await client.execute_read(SHOW_NAME_INDEX, {"name": NAME_INDEX})


async def test_index_covers_labels_and_properties(graph_client: Neo4jGraphClient) -> None:
    await ensure_name_index(graph_client)

    rows = await show_index(graph_client)

    assert len(rows) == 1
    assert set(rows[0]["labelsOrTypes"]) == set(NAME_INDEX_LABELS)
    assert ALIASES_FIELD in rows[0]["properties"]
    assert len(rows[0]["properties"]) == 2


async def test_second_call_changes_nothing(graph_client: Neo4jGraphClient) -> None:
    await ensure_name_index(graph_client)
    first = await show_index(graph_client)

    await ensure_name_index(graph_client)

    assert await show_index(graph_client) == first


async def test_alias_query_finds_owner(graph_client: Neo4jGraphClient) -> None:
    await ensure_name_index(graph_client)

    rows = await graph_client.execute_read(QUERY_NAMES, {"index": NAME_INDEX, "text": ALIAS_QUERY})

    assert ALIAS_OWNER_ID in {row["id"] for row in rows}
