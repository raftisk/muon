"""Neo4j implementation of `GraphClient`, built on the async driver."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import TracebackType
from typing import Any, LiteralString, Self

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction
from neo4j.exceptions import (
    ConfigurationError,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
)

from muon.config import Neo4jSettings
from muon.graph.errors import GraphConnectionError, GraphQueryError
from muon.graph.protocol import Statement, WriteSummary

READ_OPERATION = "read"
WRITE_OPERATION = "write"


class Neo4jGraphClient:
    """Owns one driver for the lifetime of an `async with` block.

    Entering the block creates the driver and verifies connectivity; leaving it
    closes the driver. Queries run in managed transactions, so the driver retries
    transient failures such as deadlocks.
    """

    def __init__(self, settings: Neo4jSettings) -> None:
        self.settings = settings
        self.driver: AsyncDriver | None = None

    async def __aenter__(self) -> Self:
        target = f"uri={self.settings.uri!r} database={self.settings.database!r}"
        try:
            driver = AsyncGraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.username, self.settings.password.get_secret_value()),
                connection_timeout=self.settings.connection_timeout,
                max_connection_pool_size=self.settings.max_connection_pool_size,
            )
        except ConfigurationError as error:
            raise GraphConnectionError(f"Failed to create Neo4j driver for {target}") from error

        try:
            await driver.verify_connectivity(database=self.settings.database)
        except (ServiceUnavailable, Neo4jError) as error:
            await driver.close()
            raise GraphConnectionError(f"Failed to connect to Neo4j at {target}") from error

        self.driver = driver
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.driver is not None:
            await self.driver.close()
            self.driver = None

    async def execute_read(
        self, query: LiteralString, parameters: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        driver = self.require_driver()
        query_parameters = dict(parameters or {})
        context = describe_query(READ_OPERATION, self.settings.database, query, query_parameters)
        with wrap_driver_errors(context):
            async with driver.session(database=self.settings.database) as session:
                return await session.execute_read(fetch_records, query, query_parameters)

    async def execute_write(
        self, query: LiteralString, parameters: Mapping[str, Any] | None = None
    ) -> WriteSummary:
        driver = self.require_driver()
        query_parameters = dict(parameters or {})
        context = describe_query(WRITE_OPERATION, self.settings.database, query, query_parameters)
        with wrap_driver_errors(context):
            async with driver.session(database=self.settings.database) as session:
                return await session.execute_write(consume_counters, query, query_parameters)

    async def execute_write_batch(self, statements: Sequence[Statement]) -> list[WriteSummary]:
        driver = self.require_driver()
        database = self.settings.database
        context = f"{WRITE_OPERATION} of {len(statements)} statements on database={database!r}"
        with wrap_driver_errors(context):
            async with driver.session(database=database) as session:
                return await session.execute_write(run_statements, statements, database)

    def require_driver(self) -> AsyncDriver:
        if self.driver is None:
            raise RuntimeError("Neo4jGraphClient used outside 'async with'")
        return self.driver


def describe_query(operation: str, database: str, query: str, parameters: Mapping[str, Any]) -> str:
    """Name a query for an error message.

    Carries parameter names only, because parameter values can hold arbitrary
    ingested data.
    """
    return f"{operation} on database={database!r} query={query!r} params={sorted(parameters)}"


@contextmanager
def wrap_driver_errors(context: str) -> Iterator[None]:
    """Translate driver errors into graph errors whose message ends with `context`."""
    try:
        yield
    except (ServiceUnavailable, SessionExpired) as error:
        raise GraphConnectionError(f"Lost connection during {context}") from error
    except Neo4jError as error:
        raise GraphQueryError(f"Neo4j rejected {context}") from error


async def fetch_records(
    tx: AsyncManagedTransaction, query: LiteralString, parameters: dict[str, Any]
) -> list[dict[str, Any]]:
    """Read every record inside the transaction; results are unusable after it ends."""
    result = await tx.run(query, parameters)
    return [record.data() async for record in result]


async def consume_counters(
    tx: AsyncManagedTransaction, query: LiteralString, parameters: dict[str, Any]
) -> WriteSummary:
    result = await tx.run(query, parameters)
    summary = await result.consume()
    counters = summary.counters
    return WriteSummary(
        nodes_created=counters.nodes_created,
        nodes_deleted=counters.nodes_deleted,
        relationships_created=counters.relationships_created,
        relationships_deleted=counters.relationships_deleted,
        labels_added=counters.labels_added,
        labels_removed=counters.labels_removed,
        properties_set=counters.properties_set,
    )


async def run_statements(
    tx: AsyncManagedTransaction, statements: Sequence[Statement], database: str
) -> list[WriteSummary]:
    """Run each statement in order inside one transaction.

    A retryable error propagates unchanged so the driver can retry the whole
    transaction. Any other driver error names the failing statement by its
    1-based position.
    """
    summaries: list[WriteSummary] = []
    for position, (query, parameters) in enumerate(statements, start=1):
        query_parameters = dict(parameters)
        try:
            summaries.append(await consume_counters(tx, query, query_parameters))
        except Neo4jError as error:
            if error.is_retryable():
                raise
            context = describe_query(WRITE_OPERATION, database, query, query_parameters)
            raise GraphQueryError(
                f"Neo4j rejected statement {position} of {len(statements)}: {context}"
            ) from error
    return summaries
