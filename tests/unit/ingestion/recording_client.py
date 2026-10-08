"""A `GraphClient` that records every write instead of reaching a database."""

from collections.abc import Mapping, Sequence
from typing import Any, LiteralString

from muon.graph import GraphClient, Statement, WriteSummary

EMPTY_SUMMARY = WriteSummary(
    nodes_created=0,
    nodes_deleted=0,
    relationships_created=0,
    relationships_deleted=0,
    labels_added=0,
    labels_removed=0,
    properties_set=0,
)


class RecordingGraphClient:
    """Records writes in call order and answers each statement with `summary`.

    `calls` holds the query of an `execute_write` or the statement list of an
    `execute_write_batch`.
    """

    def __init__(self, summary: WriteSummary = EMPTY_SUMMARY) -> None:
        self.summary = summary
        self.calls: list[str | list[Statement]] = []

    @property
    def writes(self) -> list[str]:
        return [call for call in self.calls if isinstance(call, str)]

    @property
    def batches(self) -> list[list[Statement]]:
        return [call for call in self.calls if isinstance(call, list)]

    async def execute_read(
        self, query: LiteralString, parameters: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return []

    async def execute_write(
        self, query: LiteralString, parameters: Mapping[str, Any] | None = None
    ) -> WriteSummary:
        self.calls.append(query)
        return self.summary

    async def execute_write_batch(self, statements: Sequence[Statement]) -> list[WriteSummary]:
        self.calls.append(list(statements))
        return [self.summary for _ in statements]


# mypy checks that the recorder satisfies the protocol.
protocol_check: GraphClient = RecordingGraphClient()
