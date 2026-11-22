"""Set `sourceUrl` on the existing nodes of the live graph from the Bulbapedia parse outputs.

The dry run (default) reads the graph and reports what a write would do. `--write` sets the
property on the nodes that have none. A node that is missing or already holds another URL
stops the run before any write.
"""

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import LiteralString

from muon.config import get_settings
from muon.graph import Neo4jGraphClient
from muon.graph.protocol import GraphClient
from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.batching import SOURCE_URL, TEXT_CHUNK, UNIVERSE
from muon.ingestion.bulbapedia.fetch import PAGE_TYPES
from muon.ingestion.bulbapedia.runs import DEFAULT_OUTPUT_ROOT, OUTPUT_SUFFIX
from muon.ingestion.runner import Runner, UniverseCensus
from muon.ontology.materializer import LabelMaterializer, build_plugin_materializer
from muon.ontology.model import load_ontology_model
from muon.plugins import discover_universes

BATCH_KEY = "batch"
BATCH_KEY_SUFFIX = "_batch"
WRITE_CHUNK_SIZE = 1000
SAMPLE_SIZE = 5
EXIT_OK = 0
EXIT_BLOCKED = 1

READ_SOURCE_URLS: LiteralString = """
UNWIND $rows AS row
MATCH (n:$(row.label):$($universe) {id: row.id})
RETURN row.label AS label, row.id AS id, n.sourceUrl AS current
"""

SET_SOURCE_URLS: LiteralString = """
UNWIND $rows AS row
MATCH (n:$(row.label):$($universe) {id: row.id})
SET n.sourceUrl = row.url
"""


@dataclass(frozen=True)
class SourceUrlRow:
    class_iri: str
    label: str
    id: str
    url: str


@dataclass(frozen=True)
class DryRunReport:
    """How the rows compare with the graph. `pending` holds the rows with no URL yet."""

    rows: tuple[SourceUrlRow, ...]
    pending: tuple[SourceUrlRow, ...]
    already_set: int
    conflicting: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def is_blocked(self) -> bool:
        return bool(self.conflicting or self.missing)


def read_page_batches(path: Path) -> list[GraphBatch]:
    """Return the batches of 1 parsed page file: `batch` and every key ending in `_batch`."""
    content = json.loads(path.read_text(encoding="utf-8"))
    return [
        GraphBatch.model_validate(value)
        for key, value in content.items()
        if (key == BATCH_KEY or key.endswith(BATCH_KEY_SUFFIX)) and value is not None
    ]


def read_source_url_rows(parsed_root: Path, materializer: LabelMaterializer) -> list[SourceUrlRow]:
    """Return 1 row per owner node that carries a `sourceUrl`, sorted by label and `id`.

    Equal duplicates collapse. Raises `ValueError` when 1 node has 2 URLs.
    """
    urls: dict[tuple[str, str], SourceUrlRow] = {}
    for page_type in PAGE_TYPES:
        for path in sorted((parsed_root / page_type).glob(f"*{OUTPUT_SUFFIX}")):
            for batch in read_page_batches(path):
                for node in batch.owner_nodes:
                    url = node.properties.get(SOURCE_URL)
                    if url is None or node.class_iri == TEXT_CHUNK:
                        continue
                    row = SourceUrlRow(
                        node.class_iri,
                        materializer.map_key_label(node.class_iri),
                        node.id,
                        str(url),
                    )
                    known = urls.setdefault((row.label, row.id), row)
                    if known.url != row.url:
                        raise ValueError(
                            f"{row.label} {row.id} has 2 sourceUrl values: "
                            f"{known.url} and {row.url}"
                        )
    return sorted(urls.values(), key=lambda row: (row.label, row.id))


def build_parameters(rows: Iterable[SourceUrlRow], universe: str) -> dict[str, object]:
    return {
        "universe": universe,
        "rows": [{"label": row.label, "id": row.id, "url": row.url} for row in rows],
    }


async def compare_with_graph(
    client: GraphClient, rows: Sequence[SourceUrlRow], universe: str
) -> DryRunReport:
    records = await client.execute_read(READ_SOURCE_URLS, build_parameters(rows, universe))
    current = {(record["label"], record["id"]): record["current"] for record in records}
    pending: list[SourceUrlRow] = []
    conflicting: list[str] = []
    missing: list[str] = []
    already_set = 0
    for row in rows:
        key = (row.label, row.id)
        if key not in current:
            missing.append(f"{row.label} {row.id}")
        elif current[key] is None:
            pending.append(row)
        elif current[key] == row.url:
            already_set += 1
        else:
            conflicting.append(f"{row.label} {row.id}: {current[key]} != {row.url}")
    return DryRunReport(
        tuple(rows), tuple(pending), already_set, tuple(conflicting), tuple(missing)
    )


async def write_source_urls(
    client: GraphClient, rows: Sequence[SourceUrlRow], universe: str
) -> int:
    """Set the URLs in transactions of `WRITE_CHUNK_SIZE` rows and return `properties_set`."""
    total = 0
    for start in range(0, len(rows), WRITE_CHUNK_SIZE):
        chunk = rows[start : start + WRITE_CHUNK_SIZE]
        summary = await client.execute_write(SET_SOURCE_URLS, build_parameters(chunk, universe))
        total += summary.properties_set
    return total


def print_report(report: DryRunReport) -> None:
    print(f"rows: {len(report.rows)}")
    for label, count in sorted(Counter(row.label for row in report.rows).items()):
        print(f"  {label}: {count}")
    matched = len(report.rows) - len(report.missing)
    print(f"matched: {matched}")
    print(f"already set: {report.already_set}")
    print(f"to write: {len(report.pending)}")
    print(f"conflicting: {len(report.conflicting)}")
    for line in report.conflicting[:SAMPLE_SIZE]:
        print(f"  {line}")
    print(f"missing: {len(report.missing)}")
    for line in report.missing[:SAMPLE_SIZE]:
        print(f"  {line}")


def print_census(title: str, census: UniverseCensus) -> None:
    print(f"census {title}:")
    print(f"  labels: {dict(census.labels)}")
    print(f"  relationship types: {dict(census.relationship_types)}")


async def run(should_write: bool, parsed_root: Path) -> int:
    settings = get_settings()
    ontology_dir = settings.ontology.ontology_dir
    model = load_ontology_model(ontology_dir, UNIVERSE)
    plugin = discover_universes(settings.universes.universes_dir, ontology_dir).get(UNIVERSE)
    materializer = build_plugin_materializer(model, plugin.manifest)
    rows = read_source_url_rows(parsed_root, materializer)
    async with Neo4jGraphClient(settings.neo4j) as client:
        runner = Runner(client, materializer)
        before = await runner.count_universe()
        report = await compare_with_graph(client, rows, materializer.universe_label)
        print_report(report)
        print_census("before", before)
        if report.is_blocked:
            return EXIT_BLOCKED
        if not should_write:
            return EXIT_OK
        properties_set = await write_source_urls(
            client, report.pending, materializer.universe_label
        )
        print(f"properties_set: {properties_set}")
        print_census("after", await runner.count_universe())
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="set the URLs, not only report")
    parser.add_argument("--parsed-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    return asyncio.run(run(arguments.write, arguments.parsed_root))


if __name__ == "__main__":
    raise SystemExit(main())
