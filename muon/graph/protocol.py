"""The interface every graph consumer types against."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, LiteralString, NamedTuple, Protocol


@dataclass(frozen=True)
class WriteSummary:
    """Counts of what one write statement changed."""

    nodes_created: int
    nodes_deleted: int
    relationships_created: int
    relationships_deleted: int
    labels_added: int
    labels_removed: int
    properties_set: int


class Statement(NamedTuple):
    """One parameterized Cypher statement of a write batch."""

    query: LiteralString
    parameters: Mapping[str, Any]


class GraphClient(Protocol):
    """Async access to the graph.

    Callers pass values through `parameters` and never format them into `query`.
    Every method raises `GraphQueryError` when the database rejects a query and
    `GraphConnectionError` when it cannot be reached.
    """

    async def execute_read(
        self, query: LiteralString, parameters: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run `query` in a read transaction and return one dict per record."""
        ...

    async def execute_write(
        self, query: LiteralString, parameters: Mapping[str, Any] | None = None
    ) -> WriteSummary:
        """Run `query` in a write transaction and return what it changed."""
        ...

    async def execute_write_batch(self, statements: Sequence[Statement]) -> list[WriteSummary]:
        """Run `statements` in order in one write transaction.

        Returns one summary per statement, in order. If any statement fails, the
        transaction rolls back and nothing from the batch persists.
        """
        ...
