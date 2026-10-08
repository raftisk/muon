from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import CypherSyntaxError, ServiceUnavailable
from pydantic import SecretStr

from muon.config import Neo4jSettings
from muon.graph import GraphConnectionError, GraphQueryError, Neo4jGraphClient, WriteSummary
from muon.graph.client import consume_counters, fetch_records

URI = "neo4j://test-host:7687"
PASSWORD = "super-secret"
DATABASE = "testdb"
READ_QUERY = "MATCH (n:Foo {name: $name}) RETURN n.name AS name"
SECRET_PARAMETER_VALUE = "private-value"


@pytest.fixture
def settings() -> Neo4jSettings:
    return Neo4jSettings(uri=URI, username="neo4j", password=SecretStr(PASSWORD), database=DATABASE)


@pytest.fixture
def session() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch, session: AsyncMock) -> MagicMock:
    mock_driver = MagicMock()
    mock_driver.verify_connectivity = AsyncMock()
    mock_driver.close = AsyncMock()
    mock_driver.session.return_value.__aenter__.return_value = session
    monkeypatch.setattr(AsyncGraphDatabase, "driver", MagicMock(return_value=mock_driver))
    return mock_driver


async def test_enter_verifies_connectivity(settings: Neo4jSettings, driver: MagicMock) -> None:
    async with Neo4jGraphClient(settings):
        driver.verify_connectivity.assert_awaited_once_with(database=DATABASE)


async def test_exit_closes_driver(settings: Neo4jSettings, driver: MagicMock) -> None:
    async with Neo4jGraphClient(settings):
        pass

    driver.close.assert_awaited_once()


async def test_exit_closes_driver_on_exception(settings: Neo4jSettings, driver: MagicMock) -> None:
    with pytest.raises(ValueError):
        async with Neo4jGraphClient(settings):
            raise ValueError("boom")

    driver.close.assert_awaited_once()


async def test_connection_failure_raises_graph_connection_error(
    settings: Neo4jSettings, driver: MagicMock
) -> None:
    driver.verify_connectivity.side_effect = ServiceUnavailable("unreachable")

    with pytest.raises(GraphConnectionError) as error_info:
        async with Neo4jGraphClient(settings):
            pass

    message = str(error_info.value)
    assert URI in message
    assert PASSWORD not in message
    driver.close.assert_awaited_once()


async def test_execute_read_uses_managed_read_transaction(
    settings: Neo4jSettings, driver: MagicMock, session: AsyncMock
) -> None:
    session.execute_read.return_value = [{"name": "x"}]

    async with Neo4jGraphClient(settings) as client:
        records = await client.execute_read(READ_QUERY, {"name": "x"})

    assert records == [{"name": "x"}]
    driver.session.assert_called_once_with(database=DATABASE)
    session.execute_read.assert_awaited_once_with(fetch_records, READ_QUERY, {"name": "x"})


async def test_execute_write_uses_managed_write_transaction(
    settings: Neo4jSettings, driver: MagicMock, session: AsyncMock
) -> None:
    write_query = "CREATE (:Foo {name: $name})"

    async with Neo4jGraphClient(settings) as client:
        await client.execute_write(write_query, {"name": "x"})

    session.execute_write.assert_awaited_once_with(consume_counters, write_query, {"name": "x"})


async def test_query_error_is_wrapped(
    settings: Neo4jSettings, driver: MagicMock, session: AsyncMock
) -> None:
    session.execute_read.side_effect = CypherSyntaxError("bad syntax")

    async with Neo4jGraphClient(settings) as client:
        with pytest.raises(GraphQueryError) as error_info:
            await client.execute_read(READ_QUERY, {"name": SECRET_PARAMETER_VALUE})

    message = str(error_info.value)
    assert READ_QUERY in message
    assert "name" in message
    assert SECRET_PARAMETER_VALUE not in message


async def test_fetch_records_returns_dicts() -> None:
    records = [MagicMock(data=MagicMock(return_value={"name": name})) for name in ("a", "b")]
    result = MagicMock()
    result.__aiter__.return_value = records
    tx = AsyncMock()
    tx.run.return_value = result

    assert await fetch_records(tx, READ_QUERY, {}) == [{"name": "a"}, {"name": "b"}]


async def test_consume_counters_maps_fields() -> None:
    counter_values: dict[str, Any] = {
        "nodes_created": 1,
        "nodes_deleted": 2,
        "relationships_created": 3,
        "relationships_deleted": 4,
        "properties_set": 5,
    }
    result = AsyncMock()
    result.consume.return_value = MagicMock(counters=MagicMock(**counter_values))
    tx = AsyncMock()
    tx.run.return_value = result

    assert await consume_counters(tx, "CREATE (n)", {}) == WriteSummary(**counter_values)


async def test_use_outside_context_raises(settings: Neo4jSettings) -> None:
    with pytest.raises(RuntimeError, match="outside 'async with'"):
        await Neo4jGraphClient(settings).execute_read(READ_QUERY)
