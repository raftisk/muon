"""A `GraphClient` that answers reads from a mapping of query constants."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, LiteralString

from muon.graph import GraphClient, Statement, WriteSummary

Answer = Callable[[Mapping[str, Any]], list[dict[str, Any]]]


class QueryGraphClient:
    """Answers each read from `answers`, keyed by the query text, and records every read.

    A query with no answer fails the test. Writes are never expected and fail it too.
    """

    def __init__(self, answers: Mapping[str, Answer]) -> None:
        self.answers = dict(answers)
        self.reads: list[tuple[str, Mapping[str, Any]]] = []

    async def execute_read(
        self,
        query: LiteralString,
        parameters: Mapping[str, Any] | None = None,
        max_records: int | None = None,
    ) -> list[dict[str, Any]]:
        if query not in self.answers:
            raise AssertionError(f"Unexpected read: {query}")
        used_parameters = parameters or {}
        self.reads.append((query, used_parameters))
        rows = self.answers[query](used_parameters)
        return rows if max_records is None else rows[:max_records]

    async def execute_write(
        self, query: LiteralString, parameters: Mapping[str, Any] | None = None
    ) -> WriteSummary:
        raise AssertionError(f"Unexpected write: {query}")

    async def execute_write_batch(self, statements: Sequence[Statement]) -> list[WriteSummary]:
        raise AssertionError("Unexpected write batch")

    def list_queries(self) -> list[str]:
        return [query for query, _ in self.reads]


# mypy checks that the fake satisfies the protocol.
protocol_check: GraphClient = QueryGraphClient({})
