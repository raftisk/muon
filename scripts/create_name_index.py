"""Create the `entity_names` full-text index on the configured database and print its state."""

import asyncio

from muon.config import get_settings
from muon.graph import Neo4jGraphClient
from muon.graph.indexes import NAME_INDEX
from muon.ingestion.schema import ensure_name_index

SHOW_NAME_INDEX = "SHOW INDEXES YIELD name, state, labelsOrTypes, properties WHERE name = $name"


async def run() -> int:
    async with Neo4jGraphClient(get_settings().neo4j) as client:
        await ensure_name_index(client)
        rows = await client.execute_read(SHOW_NAME_INDEX, {"name": NAME_INDEX})
    for row in rows:
        print(row)
    return 0 if rows else 1


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
