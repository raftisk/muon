from collections.abc import Mapping
from typing import Any, LiteralString

from muon.graph import GraphClient, WriteSummary

CANNED_RECORDS = [{"name": "x"}]
EMPTY_SUMMARY = WriteSummary(
    nodes_created=0,
    nodes_deleted=0,
    relationships_created=0,
    relationships_deleted=0,
    properties_set=0,
)


class FakeGraphClient:
    async def execute_read(
        self, query: LiteralString, parameters: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return CANNED_RECORDS

    async def execute_write(
        self, query: LiteralString, parameters: Mapping[str, Any] | None = None
    ) -> WriteSummary:
        return EMPTY_SUMMARY


# mypy checks that the fake satisfies the protocol.
fake_client: GraphClient = FakeGraphClient()


async def test_fake_client_returns_canned_records() -> None:
    assert await fake_client.execute_read("MATCH (n) RETURN n.name AS name") == CANNED_RECORDS
