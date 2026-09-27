"""The interface every graph consumer types against."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, LiteralString, Protocol


@dataclass(frozen=True)
class WriteSummary:
    """Counts of what one write transaction changed."""

    nodes_created: int
    nodes_deleted: int
    relationships_created: int
    relationships_deleted: int
    properties_set: int


class GraphClient(Protocol):
    """Async access to the graph.

    Callers pass values through `parameters` and never format them into `query`.
    Both methods raise `GraphQueryError` when the database rejects the query and
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
