"""Plan: what a sequence of batches would write, as JSON files a person reads first.

Planning reads the graph and writes no data. A plan is 1 folder,
`<root>/<universe>/<run id>/`, with `plan.json` and 1 file per batch under `batches/`.

Map:
- Models: `ItemState`, the diff models (`BatchDiff` and its changes), `StoredStatement`,
  `PlannedItem` (1 batch file) and `PlanHeader` (`plan.json`).
- Diff: `read_batch_diff` compares a batch with the graph, with the semantics of an upsert
  write: only the properties a row sets are compared, and only ontology labels are removed.
- Delete guard: `check_deletes` rejects a preview that reaches a node outside the scope.
- Writing a plan: `open_plan` gives a `PlanWriter`, whose `plan_item` runs the policy check,
  the Validator, the delete preview and the diff, and stops at the 1st one that rejects.
  `finish` writes `plan.json`, so a folder without it is an incomplete plan.
  `plan_batches` does all of it for a sequence of batches.
"""

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, LiteralString, Protocol

from pydantic import BaseModel, ConfigDict, field_serializer

from muon.config import PROJECT_ROOT
from muon.files import dump_json, write_text_atomically
from muon.graph import GraphClient
from muon.ingestion.batch import GraphBatch, NodeRef, serialize_batch
from muon.ingestion.policy import WritePolicy, check_policy
from muon.ingestion.runner import (
    DeletePreview,
    Runner,
    WriteMode,
    build_edge_rows,
    build_node_rows,
    build_statements,
)
from muon.ingestion.seed import find_missing_nodes
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import list_ontology_paths
from muon.validation.report import ValidationReport, Violation
from muon.validation.shapes import SHAPES_DIR_NAME, SHAPES_SUFFIX

PLAN_ROOT = PROJECT_ROOT / "data" / "plans"
PLAN_FILE_NAME = "plan.json"
BATCHES_DIR_NAME = "batches"
BATCH_FILE_DIGITS = 5
RUN_ID_TIME_FORMAT = "%Y%m%d-%H%M%S"
RULE_DELETE_SCOPE = "D1"

# Row keys must match `build_node_rows`.
READ_OWNER_NODES: LiteralString = """
UNWIND $nodes AS row
MATCH (n:$(row.key_labels) {id: row.id})
RETURN row.key_labels[0] AS key_label, row.id AS id, labels(n) AS labels,
  [key IN keys(row.properties) WHERE n[key] IS NOT NULL | [key, n[key]]] AS properties
"""

# Row keys must match `build_edge_rows`.
READ_LISTED_EDGES: LiteralString = """
UNWIND $edges AS row
MATCH (a:$([row.start_label, $universe]) {id: row.start_id})
MATCH (b:$([row.end_label, $universe]) {id: row.end_id})
MATCH (a)-[r:$(row.type)]->(b)
RETURN row.type AS type, row.start_label AS start_label, row.start_id AS start_id,
  row.end_label AS end_label, row.end_id AS end_id,
  [key IN keys(row.properties) WHERE r[key] IS NOT NULL | [key, r[key]]] AS properties
"""

EdgeKey = tuple[str, str, str, str, str]


class PlanError(Exception):
    """A plan folder cannot be written or read. The message names the path."""


class ItemState(StrEnum):
    PLANNED = "planned"
    NO_CHANGE = "no_change"
    REJECTED = "rejected"


class PlanModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class PropertyChange(PlanModel):
    """A property a write sets to a new value. `before` is `None` for an absent property."""

    name: str
    before: Any
    after: Any


class NodeChange(PlanModel):
    """What a write changes on a node that exists."""

    node: NodeRef
    property_changes: tuple[PropertyChange, ...]
    labels_added: tuple[str, ...]
    labels_removed: tuple[str, ...]


class EdgeChange(PlanModel):
    """An edge a write creates, or one whose properties it changes."""

    type: str
    start_label: str
    start_id: str
    end_label: str
    end_id: str
    property_changes: tuple[PropertyChange, ...]


class BatchDiff(PlanModel):
    """What an upsert write of 1 batch creates and changes. Deletes are in the preview."""

    nodes_added: tuple[NodeRef, ...] = ()
    nodes_changed: tuple[NodeChange, ...] = ()
    edges_added: tuple[EdgeChange, ...] = ()
    edges_changed: tuple[EdgeChange, ...] = ()
    stubs_created: tuple[NodeRef, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (
            self.nodes_added
            or self.nodes_changed
            or self.edges_added
            or self.edges_changed
            or self.stubs_created
        )


class StoredStatement(PlanModel):
    cypher: str
    parameters: Mapping[str, Any]


class PlannedItem(PlanModel):
    """1 batch of a plan, as its file holds it.

    A rejected item holds the violations of the step that rejected it, no diff and no
    statements. It holds a preview only when the delete guard rejected it.
    """

    number: int
    source: str
    state: ItemState
    policy: WritePolicy | None
    batch: GraphBatch
    violations: tuple[Violation, ...] = ()
    diff: BatchDiff | None = None
    preview: DeletePreview | None = None
    statements: tuple[StoredStatement, ...] = ()

    @field_serializer("batch")
    def serialize_stored_batch(self, batch: GraphBatch) -> dict[str, object]:
        return serialize_batch(batch)


class DiffTotals(PlanModel):
    """The diffs and previews of a plan, summed per batch.

    A node or edge that 2 batches list counts once in each.
    """

    nodes_added: int = 0
    nodes_changed: int = 0
    nodes_deleted: int = 0
    stubs_created: int = 0
    edges_added: int = 0
    edges_changed: int = 0
    edges_deleted: int = 0


class PlanHeader(PlanModel):
    """`plan.json`. `counts` maps every item state to its number of items.

    `totals` covers the planned items only.
    """

    universe: str
    run_id: str
    run_label: str
    tbox_hash: str
    shapes_hash: str
    counts: Mapping[ItemState, int]
    totals: DiffTotals


class BatchValidator(Protocol):
    async def validate(self, batch: GraphBatch) -> ValidationReport: ...


@dataclass(frozen=True)
class PlanContext:
    """What planning reads. The plan folder goes under `root`."""

    client: GraphClient
    runner: Runner
    validator: BatchValidator
    materializer: LabelMaterializer
    ontology_dir: Path
    universe: str
    run_label: str
    root: Path = PLAN_ROOT


class PlanWriter:
    """Owns 1 plan folder: numbers the batches, writes their files and then `plan.json`."""

    def __init__(self, context: PlanContext, run_id: str, plan_dir: Path) -> None:
        self.context = context
        self.run_id = run_id
        self.plan_dir = plan_dir
        self.counts: Counter[ItemState] = Counter()
        self.totals: Counter[str] = Counter()

    async def plan_item(
        self, source: str, batch: GraphBatch, policy: WritePolicy | None
    ) -> PlannedItem:
        """Assess `batch`, write its batch file and return it."""
        number = self.counts.total() + 1
        unchecked = PlannedItem(
            number=number, source=source, state=ItemState.REJECTED, policy=policy, batch=batch
        )
        item = await assess_item(self.context, unchecked)
        path = build_batch_path(self.plan_dir, number)
        write_text_atomically(path, dump_json(item.model_dump(mode="json")))
        self.counts[item.state] += 1
        if item.state is ItemState.PLANNED:
            self.totals.update(count_changes(item))
        return item

    def finish(self) -> PlanHeader:
        """Write `plan.json` and return it."""
        context = self.context
        tbox_paths = list_ontology_paths(context.ontology_dir, context.universe)
        shapes_path = context.ontology_dir / SHAPES_DIR_NAME / f"{context.universe}{SHAPES_SUFFIX}"
        header = PlanHeader(
            universe=context.universe,
            run_id=self.run_id,
            run_label=context.run_label,
            tbox_hash=hash_files(tbox_paths),
            shapes_hash=hash_files([shapes_path]),
            counts={state: self.counts[state] for state in ItemState},
            totals=DiffTotals(**self.totals),
        )
        text = dump_json(header.model_dump(mode="json"))
        write_text_atomically(self.plan_dir / PLAN_FILE_NAME, text)
        return header


def open_plan(context: PlanContext) -> PlanWriter:
    """Create the folder of a new plan. Raises `PlanError` when it exists."""
    run_id = f"{datetime.now(UTC).strftime(RUN_ID_TIME_FORMAT)}-{context.run_label}"
    plan_dir = context.root / context.universe / run_id
    try:
        plan_dir.mkdir(parents=True)
    except FileExistsError as error:
        raise PlanError(f"Cannot open a plan: {plan_dir} exists") from error
    return PlanWriter(context, run_id, plan_dir)


async def plan_batches(
    context: PlanContext, items: Sequence[tuple[str, GraphBatch, WritePolicy | None]]
) -> PlanHeader:
    """Plan `items`, each a source label, a batch and its job policy, in order."""
    writer = open_plan(context)
    for source, batch, policy in items:
        await writer.plan_item(source, batch, policy)
    return writer.finish()


async def assess_item(context: PlanContext, item: PlannedItem) -> PlannedItem:
    """Return `item` with the state and the findings of the steps that ran on it."""
    batch = item.batch
    if item.policy is not None:
        violations = check_policy(batch, item.policy)
        if violations:
            return reject_item(item, violations)

    validation = await context.validator.validate(batch)
    if not validation.is_valid:
        return reject_item(item, validation.violations)

    preview = await context.runner.preview_deletes(batch)
    violations = check_deletes(preview, batch, context.materializer)
    if violations:
        return reject_item(item, violations).model_copy(update={"preview": preview})

    diff = await read_batch_diff(context.client, batch, context.materializer)
    is_unchanged = diff.is_empty and preview.is_empty
    return item.model_copy(
        update={
            "state": ItemState.NO_CHANGE if is_unchanged else ItemState.PLANNED,
            "diff": diff,
            "preview": preview,
            "statements": store_statements(batch, context.materializer),
        }
    )


def reject_item(item: PlannedItem, violations: Iterable[Violation]) -> PlannedItem:
    # Shapes report in the order of random blank node labels, so the order is fixed here.
    ordered = tuple(sorted(violations, key=lambda violation: violation.model_dump_json()))
    return item.model_copy(update={"state": ItemState.REJECTED, "violations": ordered})


def store_statements(
    batch: GraphBatch, materializer: LabelMaterializer
) -> tuple[StoredStatement, ...]:
    """Return the statements of an upsert write of `batch`, as a plan stores them."""
    return tuple(
        StoredStatement(cypher=statement.query, parameters=statement.parameters)
        for statement in build_statements(batch, WriteMode.UPSERT, materializer)
    )


def count_changes(item: PlannedItem) -> dict[str, int]:
    """Count the changes of 1 planned item under the field names of `DiffTotals`."""
    diff = item.diff or BatchDiff()
    preview = item.preview or DeletePreview()
    return {
        "nodes_added": len(diff.nodes_added),
        "nodes_changed": len(diff.nodes_changed),
        "nodes_deleted": len(preview.nodes),
        "stubs_created": len(diff.stubs_created),
        "edges_added": len(diff.edges_added),
        "edges_changed": len(diff.edges_changed),
        "edges_deleted": len(preview.edges),
    }


def check_deletes(
    preview: DeletePreview, batch: GraphBatch, materializer: LabelMaterializer
) -> tuple[Violation, ...]:
    """Return 1 D1 violation per node the batch would delete that has no scope class label."""
    scope_labels = {materializer.map_class_label(class_iri) for class_iri in batch.scope}
    return tuple(
        Violation(
            rule_id=RULE_DELETE_SCOPE,
            value=node.id,
            message=f"The claims would delete node {node.id!r} with labels "
            f"{', '.join(node.labels)}, none of the scope {', '.join(sorted(scope_labels))}",
        )
        for node in preview.nodes
        if scope_labels.isdisjoint(node.labels)
    )


async def read_batch_diff(
    client: GraphClient, batch: GraphBatch, materializer: LabelMaterializer
) -> BatchDiff:
    """Compare `batch` with the graph, with reads only."""
    nodes_added, nodes_changed = await read_node_changes(client, batch, materializer)
    edges_added, edges_changed = await read_edge_changes(client, batch, materializer)
    stubs_created = (
        await find_missing_nodes(client, materializer, batch.stub_nodes) if batch.stub_nodes else []
    )
    return BatchDiff(
        nodes_added=nodes_added,
        nodes_changed=nodes_changed,
        edges_added=edges_added,
        edges_changed=edges_changed,
        stubs_created=tuple(stubs_created),
    )


async def read_node_changes(
    client: GraphClient, batch: GraphBatch, materializer: LabelMaterializer
) -> tuple[tuple[NodeRef, ...], tuple[NodeChange, ...]]:
    """Return the owner nodes the graph lacks and the changes to those it has."""
    if not batch.owner_nodes:
        return (), ()
    rows = build_node_rows(batch.owner_nodes, materializer)
    records = await client.execute_read(READ_OWNER_NODES, {"nodes": rows})
    stored = {(record["key_label"], record["id"]): record for record in records}
    ontology_labels = materializer.list_ontology_labels()
    added: list[NodeRef] = []
    changed: list[NodeChange] = []
    for node, row in zip(batch.owner_nodes, rows, strict=True):
        record = stored.get((row["key_labels"][0], node.id))
        if record is None:
            added.append(node.node_ref)
            continue
        labels = set(record["labels"])
        change = NodeChange(
            node=node.node_ref,
            property_changes=list_property_changes(dict(record["properties"]), row["properties"]),
            labels_added=tuple(sorted(set(row["labels"]) - labels)),
            labels_removed=tuple(sorted((labels & ontology_labels) - set(row["labels"]))),
        )
        if change.property_changes or change.labels_added or change.labels_removed:
            changed.append(change)
    return (
        tuple(sorted(added, key=lambda ref: (ref.class_iri, ref.id))),
        tuple(sorted(changed, key=lambda change: (change.node.class_iri, change.node.id))),
    )


async def read_edge_changes(
    client: GraphClient, batch: GraphBatch, materializer: LabelMaterializer
) -> tuple[tuple[EdgeChange, ...], tuple[EdgeChange, ...]]:
    """Return the batch edges the graph lacks and those whose properties differ."""
    if not batch.edges:
        return (), ()
    rows = build_edge_rows(batch, materializer)
    parameters = {"universe": materializer.universe_label, "edges": rows}
    records = await client.execute_read(READ_LISTED_EDGES, parameters)
    stored = {build_edge_key(record): dict(record["properties"]) for record in records}
    added: dict[EdgeKey, EdgeChange] = {}
    changed: dict[EdgeKey, EdgeChange] = {}
    for row in rows:
        key = build_edge_key(row)
        property_changes = list_property_changes(stored.get(key, {}), row["properties"])
        if key in stored and not property_changes:
            continue
        change = EdgeChange(
            type=row["type"],
            start_label=row["start_label"],
            start_id=row["start_id"],
            end_label=row["end_label"],
            end_id=row["end_id"],
            property_changes=property_changes,
        )
        (changed if key in stored else added)[key] = change
    return (
        tuple(added[key] for key in sorted(added)),
        tuple(changed[key] for key in sorted(changed)),
    )


def build_edge_key(row: Mapping[str, Any]) -> EdgeKey:
    return (row["type"], row["start_label"], row["start_id"], row["end_label"], row["end_id"])


def list_property_changes(
    stored: Mapping[str, Any], wanted: Mapping[str, Any]
) -> tuple[PropertyChange, ...]:
    """Compare the properties a write sets with the stored ones, by value and by type."""
    changes: list[PropertyChange] = []
    for name in sorted(wanted):
        before, after = stored.get(name), wanted[name]
        if type(before) is not type(after) or before != after:
            changes.append(PropertyChange(name=name, before=before, after=after))
    return tuple(changes)


def build_batch_path(plan_dir: Path, number: int) -> Path:
    return plan_dir / BATCHES_DIR_NAME / f"{number:0{BATCH_FILE_DIGITS}d}.json"


def hash_files(paths: Iterable[Path]) -> str:
    """Return the SHA-256 over the names and contents of `paths`, in sorted order."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
