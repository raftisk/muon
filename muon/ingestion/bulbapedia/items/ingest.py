"""Item ingest: the batches of an `ItemParseRun` go into the graph in 2 passes.

`ingest_items` checks the seed and ensures the constraints of every class in the run once,
then writes every pass 1 batch (Item, HeldItem, Berry), then every pass 2 batch (TextChunk),
and reads the census of the universe and the ids of the Item family nodes that are still
stubs. An item owner finds the stub a species page wrote with a sibling class through the
`Item` key label, so no `REQUIRES` edge of the species pass changes. The write loop, the stop
rule and the report models are the shared ones of `bulbapedia/ingest.py`.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from muon.graph import GraphClient
from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.batching import ITEM
from muon.ingestion.bulbapedia.ingest import (
    INGEST_OUTPUT_ROOT,
    IngestRequest,
    PassItem,
    PlanSummary,
    StubIngestReport,
    build_ingest_report,
    list_batch_classes,
    prepare_target,
    should_count,
    write_passes,
    write_report_file,
)
from muon.ingestion.bulbapedia.items.run import ItemPageOutput, ItemParseRun

ITEM_INGEST_REPORT_FILE_NAME = "item-ingest-report.json"
STUB_CLASSES = (ITEM,)


class ItemPass(StrEnum):
    """The passes in write order: chunks point at the item nodes pass 1 owns."""

    ITEM = "item"
    CHUNK = "chunk"


@dataclass(frozen=True)
class ItemIngestRun:
    report: StubIngestReport
    elapsed_seconds: Mapping[str, float]
    plan: PlanSummary | None = None


def select_item_batch(output: ItemPageOutput, ingest_pass: ItemPass) -> GraphBatch:
    if ingest_pass is ItemPass.ITEM:
        return output.batch
    return output.chunk_batch


def list_item_constraint_classes(outputs: Iterable[ItemPageOutput]) -> frozenset[str]:
    """Return the scope classes and the stub classes of every batch of the run."""
    return list_batch_classes(
        select_item_batch(output, ingest_pass) for output in outputs for ingest_pass in ItemPass
    )


async def ingest_items(
    client: GraphClient, run: ItemParseRun, request: IngestRequest
) -> ItemIngestRun:
    """Write the batches of `run` into the graph and report what happened.

    Raises `MissingSeedError` before any write when the A-Box is not in the graph. A
    `GraphError` from the seed check, the constraint step, the census or the stub list is
    not caught.
    """
    target = await prepare_target(client, request, list_item_constraint_classes(run.outputs))
    passes = [
        (
            ingest_pass,
            [
                PassItem(output.page.title, select_item_batch(output, ingest_pass))
                for output in run.outputs
            ],
        )
        for ingest_pass in ItemPass
    ]
    pass_run = await write_passes(target, passes, request.on_progress)
    is_counted = should_count(pass_run)
    census = await target.runner.count_universe() if is_counted else None
    stub_ids = await target.runner.list_stub_ids(STUB_CLASSES) if is_counted else None
    report = build_ingest_report(request, pass_run, run.report.pages, census)
    return ItemIngestRun(
        report=StubIngestReport(**dict(report), stub_ids=stub_ids),
        elapsed_seconds=pass_run.elapsed_seconds,
        plan=pass_run.plan,
    )


def write_item_ingest_report(report: StubIngestReport, root: Path = INGEST_OUTPUT_ROOT) -> Path:
    """Write the report under `root`, atomically, and return its path."""
    return write_report_file(report, ITEM_INGEST_REPORT_FILE_NAME, root)
