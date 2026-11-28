"""Plan and apply: what a sequence of batches would write, as JSON files a person reads
first, and the write of exactly that.

Planning reads the graph and writes no data. A plan is 1 folder,
`<root>/<universe>/<run id>/`, with `plan.json`, 1 file per batch under `batches/` and,
after an apply, the log `applied.jsonl`.

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
- Applying: `apply_plan` takes the batches in order. It checks each planned one again,
  rebuilds its statements, runs them in 1 transaction when they equal the stored ones and
  logs them. A batch in the log is never run again, so a 2nd apply resumes after a stop.
  Apply does not look at the graph for changes made since the plan.
"""

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, LiteralString, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError, field_serializer

from muon.config import PROJECT_ROOT
from muon.files import dump_json, write_text_atomically
from muon.graph import GraphClient, WriteSummary
from muon.ingestion.batch import GraphBatch, NodeRef, list_batch_classes, serialize_batch
from muon.ingestion.policy import WritePolicy, check_policy
from muon.ingestion.runner import (
    DeletePreview,
    Runner,
    WriteMode,
    add_summaries,
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
APPLIED_LOG_NAME = "applied.jsonl"
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

# Row keys must match `build_edge_rows`. The match expands from the start node and filters
# the end: with both ends bound first, Neo4j reads the relationships of end nodes that hold
# tens of thousands of them, seconds per batch.
READ_LISTED_EDGES: LiteralString = """
UNWIND $edges AS row
MATCH (a:$([row.start_label, $universe]) {id: row.start_id})
MATCH (a)-[r:$(row.type)]->(b)
WHERE b.id = row.end_id AND row.end_label IN labels(b) AND $universe IN labels(b)
RETURN row.type AS type, row.start_label AS start_label, row.start_id AS start_id,
  row.end_label AS end_label, row.end_id AS end_id,
  [key IN keys(row.properties) WHERE r[key] IS NOT NULL | [key, r[key]]] AS properties
"""

EdgeKey = tuple[str, str, str, str, str]


class PlanError(Exception):
    """A plan folder cannot be written or read. The message names the path."""


class ApplyError(Exception):
    """A batch of a plan cannot be applied. Earlier batches stay written and logged."""

    def __init__(self, number: int, reason: str) -> None:
        self.number = number
        super().__init__(f"Apply stopped at batch {number}: {reason}")


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


class LoggedStatement(PlanModel):
    """A log line for 1 statement apply ran, with the counters it returned."""

    batch: int
    cypher: str
    parameters: Mapping[str, Any]
    summary: WriteSummary
    time: datetime


class LoggedSkip(PlanModel):
    """A log line for a batch apply left alone because of its state."""

    batch: int
    state: ItemState
    time: datetime


class LoggedBatch(BaseModel):
    """What a resume reads from any log line."""

    batch: int


class ApplyReport(PlanModel):
    """What 1 apply did. `counters` sums the statements it ran."""

    batches_written: int
    batches_skipped: Mapping[ItemState, int]
    batches_already_applied: int
    counters: WriteSummary


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


@dataclass(frozen=True)
class ApplyContext:
    """What applying the plan in `plan_dir` reads and writes."""

    client: GraphClient
    runner: Runner
    validator: BatchValidator
    materializer: LabelMaterializer
    plan_dir: Path


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


def read_plan_header(plan_dir: Path) -> PlanHeader:
    path = plan_dir / PLAN_FILE_NAME
    try:
        return PlanHeader.model_validate_json(read_plan_file(path))
    except ValidationError as error:
        raise PlanError(f"Invalid plan file {path}: {error}") from error


def read_plan(plan_dir: Path) -> tuple[PlanHeader, tuple[PlannedItem, ...]]:
    """Read `plan.json` and every batch file it counts, in plan order."""
    header = read_plan_header(plan_dir)
    numbers = range(1, sum(header.counts.values()) + 1)
    return header, tuple(read_planned_item(plan_dir, number) for number in numbers)


def read_planned_item(plan_dir: Path, number: int) -> PlannedItem:
    path = build_batch_path(plan_dir, number)
    try:
        item = PlannedItem.model_validate_json(read_plan_file(path))
    except ValidationError as error:
        raise PlanError(f"Invalid batch file {path}: {error}") from error
    if item.number != number:
        raise PlanError(f"Invalid batch file {path}: it holds batch {item.number}")
    return item


def read_plan_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise PlanError(f"Cannot read plan file {path}: {error}") from error


def read_applied_numbers(plan_dir: Path) -> frozenset[int]:
    """Return the numbers of the batches the log of the plan holds, or none without a log."""
    path = plan_dir / APPLIED_LOG_NAME
    if not path.exists():
        return frozenset()
    numbers: set[int] = set()
    for line_number, line in enumerate(read_plan_file(path).splitlines(), start=1):
        try:
            numbers.add(LoggedBatch.model_validate_json(line).batch)
        except ValidationError as error:
            raise PlanError(f"Invalid line {line_number} of {path}: {error}") from error
    return frozenset(numbers)


async def apply_plan(context: ApplyContext) -> ApplyReport:
    """Write the planned batches that the log lacks, in order, and log each one.

    The `id` constraints of their classes are ensured once, up front. Raises `ApplyError`
    at the 1st batch that fails a check, and lets a `GraphError` of a write through.
    """
    _, items = read_plan(context.plan_dir)
    applied = read_applied_numbers(context.plan_dir)
    pending = [item for item in items if item.number not in applied]
    # A rejected batch is never written, and may name a class the ontology lacks.
    writable = (item.batch for item in pending if item.state is ItemState.PLANNED)
    await context.runner.ensure_constraints(list_batch_classes(writable))
    skipped: Counter[ItemState] = Counter()
    summaries: list[WriteSummary] = []
    written = 0
    for item in pending:
        if item.state is not ItemState.PLANNED:
            skip = LoggedSkip(batch=item.number, state=item.state, time=datetime.now(UTC))
            append_log(context.plan_dir, [skip])
            skipped[item.state] += 1
            continue
        summaries.extend(await apply_item(context, item))
        written += 1
    return ApplyReport(
        batches_written=written,
        batches_skipped=dict(skipped),
        batches_already_applied=len(items) - len(pending),
        counters=add_summaries(summaries),
    )


async def apply_item(context: ApplyContext, item: PlannedItem) -> list[WriteSummary]:
    """Check 1 planned batch again, write it in 1 transaction and log its statements.

    The statements that run are rebuilt from the batch, so no Cypher of a file reaches the
    graph. The log is written after the commit: a stop between the 2 makes the next apply
    run the batch again, which an upsert allows.
    """
    await check_item(context, item)
    statements = build_statements(item.batch, WriteMode.UPSERT, context.materializer)
    rebuilt = store_statements(item.batch, context.materializer)
    if dump_statements(rebuilt) != dump_statements(item.statements):
        reason = "the stored statements differ from the ones the batch builds now"
        raise ApplyError(item.number, reason)
    summaries = await context.client.execute_write_batch(statements)
    now = datetime.now(UTC)
    lines = [
        LoggedStatement(
            batch=item.number,
            cypher=statement.cypher,
            parameters=statement.parameters,
            summary=summary,
            time=now,
        )
        for statement, summary in zip(item.statements, summaries, strict=True)
    ]
    append_log(context.plan_dir, lines)
    return summaries


async def check_item(context: ApplyContext, item: PlannedItem) -> None:
    """Raise `ApplyError` unless the batch still passes its policy and the Validator."""
    try:
        context.runner.check_universe(item.batch)
    except ValueError as error:
        raise ApplyError(item.number, str(error)) from error
    if item.policy is not None:
        violations = check_policy(item.batch, item.policy)
        if violations:
            raise ApplyError(item.number, describe_violations(violations))
    validation = await context.validator.validate(item.batch)
    if not validation.is_valid:
        raise ApplyError(item.number, describe_violations(validation.violations))


def describe_violations(violations: Iterable[Violation]) -> str:
    return "; ".join(f"{violation.rule_id}: {violation.message}" for violation in violations)


def dump_statements(statements: Iterable[StoredStatement]) -> str:
    """Return the JSON text of the statements, which tells 1 from 1.0 and from true."""
    return dump_json([statement.model_dump(mode="json") for statement in statements])


def append_log(plan_dir: Path, lines: Sequence[LoggedStatement | LoggedSkip]) -> None:
    path = plan_dir / APPLIED_LOG_NAME
    text = "".join(f"{line.model_dump_json()}\n" for line in lines)
    try:
        with path.open("a", encoding="utf-8") as log:
            log.write(text)
    except OSError as error:
        numbers = sorted({line.batch for line in lines})
        raise PlanError(f"Cannot log batches {numbers} to {path}: {error}") from error
