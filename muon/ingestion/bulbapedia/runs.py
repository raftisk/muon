"""Parse run code that every Bulbapedia page type shares.

Map:
- Report pieces: `ReportModel`, the page outcome (`PageStatus`, `PageResult`,
  `build_result`), the batch counters (`BatchCounts`, `count_batches`) and the dropped
  template ranking (`TemplateCount`, `rank_templates`).
- Requests: `find_unpinned_titles` lists the requested titles no pin has; each page type
  words its own error.
- Files: `dump_json` and `serialize_batch` give the same bytes for the same run, and
  `write_page_files` writes the page outputs of a run and, on a full run, deletes the rest.

Each page type keeps its own report model, notes and output layout.
"""

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.pages import DEFAULT_ROOT, Page, write_text_atomically
from muon.ontology.model import extract_local_name

DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "parsed"
OUTPUT_SUFFIX = ".json"
JSON_INDENT = 2


class ReportModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class PageStatus(StrEnum):
    PARSED = "parsed"
    SKIPPED = "skipped"
    FAILED = "failed"


class PageResult(ReportModel):
    title: str
    page_id: int
    revision_id: int
    status: PageStatus
    reason: str | None


class TemplateCount(ReportModel):
    """A template left out of chunk text and how often."""

    name: str
    count: int


class BatchCounts(ReportModel):
    """Counts by class local name (owner nodes and stubs) and property local name (edges
    and claims)."""

    owner_nodes: Mapping[str, int]
    stubs: Mapping[str, int]
    edges: Mapping[str, int]
    claims: Mapping[str, int] = Field(default_factory=dict)


def build_result(page: Page, status: PageStatus, reason: str | None = None) -> PageResult:
    return PageResult(
        title=page.title,
        page_id=page.page_id,
        revision_id=page.revision_id,
        status=status,
        reason=reason,
    )


def find_unpinned_titles(titles: Iterable[str], pinned: Iterable[str]) -> list[str]:
    """Return the titles that no pin has, sorted."""
    return sorted(set(titles) - set(pinned))


def count_batches(batches: Iterable[GraphBatch]) -> BatchCounts:
    owners: Counter[str] = Counter()
    stubs: Counter[str] = Counter()
    edges: Counter[str] = Counter()
    claims: Counter[str] = Counter()
    for batch in batches:
        owners.update(extract_local_name(node.class_iri) for node in batch.owner_nodes)
        stubs.update(extract_local_name(node.class_iri) for node in batch.stub_nodes)
        edges.update(extract_local_name(edge.property_iri) for edge in batch.edges)
        claims.update(extract_local_name(claim.property_iri) for claim in batch.claims)
    return BatchCounts(
        owner_nodes=dict(owners), stubs=dict(stubs), edges=dict(edges), claims=dict(claims)
    )


def rank_templates(counts: Mapping[str, int]) -> tuple[TemplateCount, ...]:
    """Rank dropped templates, most dropped first, then by name."""
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return tuple(TemplateCount(name=name, count=count) for name, count in ranked)


def count_nonzero(counts: Iterable[tuple[str, int]]) -> dict[str, int]:
    """Map each title with a nonzero count to that count, sorted by title."""
    return {title: count for title, count in sorted(counts) if count}


def dump_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=JSON_INDENT) + "\n"


def serialize_batch(batch: GraphBatch) -> dict[str, object]:
    """Dump a batch with its scope sorted: pydantic dumps a frozenset in hash order."""
    data = batch.model_dump(mode="json")
    data["scope"] = sorted(batch.scope)
    return data


def write_page_files(output_dir: Path, files: Mapping[str, str], is_full_run: bool) -> None:
    """Write each file name to its text under `output_dir`, atomically.

    A full run also deletes every output file that `files` lacks, so the directory depends
    on the run and not on the runs before it.
    """
    for name, text in files.items():
        write_text_atomically(output_dir / name, text)
    if is_full_run:
        for path in output_dir.glob(f"*{OUTPUT_SUFFIX}"):
            if path.name not in files:
                path.unlink()
