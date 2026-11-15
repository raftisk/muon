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


@dataclass(frozen=True)
class GraphNode:
    """A node returned by a read: its element id, labels and properties."""

    element_id: str
    labels: frozenset[str]
    properties: Mapping[str, Any]


@dataclass(frozen=True)
class GraphRelationship:
    """A relationship returned by a read, with the element ids of its end nodes."""

    element_id: str
    type: str
    start_element_id: str
    end_element_id: str
    properties: Mapping[str, Any]


@dataclass(frozen=True)
class GraphPath:
    """A path returned by a read: its nodes and relationships in path order."""

    nodes: tuple[GraphNode, ...]
    relationships: tuple[GraphRelationship, ...]


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
        self,
        query: LiteralString,
        parameters: Mapping[str, Any] | None = None,
        max_records: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run `query` in a read transaction and return one dict per record.

        Nodes, relationships and paths, also inside lists and maps, come back as
        `GraphNode`, `GraphRelationship` and `GraphPath`; temporal and spatial values
        come back as `str`. With `max_records` the read stops after that many records,
        and with `None` it reads them all.
        """
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
