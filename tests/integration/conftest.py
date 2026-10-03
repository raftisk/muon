"""Fixtures for tests against the live Neo4j configured in `.env`.

Test nodes carry the `:MuonTest` label and a per-run `run_id` property. Labels cannot
be Cypher parameters, so the property scopes cleanup instead of a per-run label.
"""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from muon.config import get_settings
from muon.graph import Neo4jGraphClient


@pytest.fixture
def run_id() -> str:
    return uuid4().hex


@pytest.fixture
async def graph_client(run_id: str) -> AsyncIterator[Neo4jGraphClient]:
    async with Neo4jGraphClient(get_settings().neo4j) as client:
        yield client
        await client.execute_write(
            "MATCH (n:MuonTest {run_id: $run_id}) DETACH DELETE n", {"run_id": run_id}
        )


@pytest.fixture
async def name_constraint(graph_client: Neo4jGraphClient) -> AsyncIterator[None]:
    await graph_client.execute_write(
        "CREATE CONSTRAINT muon_test_name IF NOT EXISTS "
        "FOR (n:MuonTest) REQUIRE n.name IS UNIQUE"
    )
    yield
    await graph_client.execute_write("DROP CONSTRAINT muon_test_name IF EXISTS")
