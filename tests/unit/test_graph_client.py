from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import CypherSyntaxError, ServiceUnavailable, TransientError
from pydantic import SecretStr

from muon.config import Neo4jSettings
from muon.graph import (
    GraphConnectionError,
    GraphQueryError,
    Neo4jGraphClient,
    Statement,
    WriteSummary,
)
from muon.graph.client import consume_counters, fetch_records, run_statements

URI = "neo4j://test-host:7687"
PASSWORD = "super-secret"
DATABASE = "testdb"
READ_QUERY = "MATCH (n:Foo {name: $name}) RETURN n.name AS name"
SECRET_PARAMETER_VALUE = "private-value"
FIRST_WRITE_QUERY = "CREATE (:Foo {name: $name})"
SECOND_WRITE_QUERY = "MATCH (n:Foo {name: $name}) SET n.size = $size"
COUNTER_NAMES = (
    "nodes_created",
    "nodes_deleted",
    "relationships_created",
    "relationships_deleted",
    "labels_added",
    "labels_removed",
    "properties_set",
)


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
    counter_values: dict[str, Any] = {name: index for index, name in enumerate(COUNTER_NAMES)}
    tx = AsyncMock()
    tx.run.return_value = build_result(counter_values)

    assert await consume_counters(tx, "CREATE (n)", {}) == WriteSummary(**counter_values)


async def test_use_outside_context_raises(settings: Neo4jSettings) -> None:
    with pytest.raises(RuntimeError, match="outside 'async with'"):
        await Neo4jGraphClient(settings).execute_read(READ_QUERY)


def build_result(counter_values: dict[str, Any]) -> AsyncMock:
    result = AsyncMock()
    result.consume.return_value = MagicMock(counters=MagicMock(**counter_values))
    return result


def build_statements() -> list[Statement]:
    return [
        Statement(FIRST_WRITE_QUERY, {"name": "x"}),
        Statement(SECOND_WRITE_QUERY, {"name": SECRET_PARAMETER_VALUE, "size": 3}),
    ]


async def test_execute_write_batch_uses_one_managed_write_transaction(
    settings: Neo4jSettings, driver: MagicMock, session: AsyncMock
) -> None:
    statements = build_statements()
    session.execute_write.return_value = ["summary-1", "summary-2"]

    async with Neo4jGraphClient(settings) as client:
        summaries = await client.execute_write_batch(statements)

    assert summaries == ["summary-1", "summary-2"]
    session.execute_write.assert_awaited_once_with(run_statements, statements, DATABASE)


async def test_execute_write_batch_wraps_connection_loss(
    settings: Neo4jSettings, driver: MagicMock, session: AsyncMock
) -> None:
    session.execute_write.side_effect = ServiceUnavailable("gone")

    async with Neo4jGraphClient(settings) as client:
        with pytest.raises(GraphConnectionError, match="2 statements"):
            await client.execute_write_batch(build_statements())


async def test_run_statements_returns_summary_per_statement() -> None:
    first_counters = dict.fromkeys(COUNTER_NAMES, 0) | {"nodes_created": 1}
    second_counters = dict.fromkeys(COUNTER_NAMES, 0) | {"properties_set": 1}
    tx = AsyncMock()
    tx.run.side_effect = [build_result(first_counters), build_result(second_counters)]

    summaries = await run_statements(tx, build_statements(), DATABASE)

    assert summaries == [WriteSummary(**first_counters), WriteSummary(**second_counters)]
    assert [call.args[0] for call in tx.run.await_args_list] == [
        FIRST_WRITE_QUERY,
        SECOND_WRITE_QUERY,
    ]


async def test_run_statements_names_failing_statement() -> None:
    tx = AsyncMock()
    tx.run.side_effect = [
        build_result(dict.fromkeys(COUNTER_NAMES, 0)),
        CypherSyntaxError("bad syntax"),
    ]

    with pytest.raises(GraphQueryError) as error_info:
        await run_statements(tx, build_statements(), DATABASE)

    message = str(error_info.value)
    assert "statement 2 of 2" in message
    assert SECOND_WRITE_QUERY in message
    assert "size" in message
    assert SECRET_PARAMETER_VALUE not in message


async def test_run_statements_reraises_retryable_errors() -> None:
    transient_error = TransientError("deadlock")
    tx = AsyncMock()
    tx.run.side_effect = transient_error

    with pytest.raises(TransientError) as error_info:
        await run_statements(tx, build_statements(), DATABASE)

    assert error_info.value is transient_error
