"""Planning against a fake graph client: the order of the steps and the plan files."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, LiteralString

import pytest

from muon.ingestion import plan, runner, seed
from muon.ingestion.batch import ClaimDirection, EdgeClaim, EdgeRecord, GraphBatch, NodeRecord
from muon.ingestion.plan import ItemState, PlanContext, PlanError, open_plan, plan_batches
from muon.ingestion.policy import WritePolicy
from muon.ingestion.runner import Runner
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model
from muon.validation import ValidationReport, Violation
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.unit.ingestion.recording_client import RecordingGraphClient

TOY = "https://muon.dev/ns/muontoy#"
TOY_GADGET = TOY + "ToyGadget"
TOY_COLOR = TOY + "ToyColor"
TOY_HAS_COLOR = TOY + "hasColor"
RUN_LABEL = "toy-run"

ALPHA = NodeRecord(class_iri=TOY_GADGET, id="alpha", name="Alpha", properties={TOY + "serial": 7})
RED = NodeRecord(class_iri=TOY_COLOR, id="red", name="Red")
BATCH = GraphBatch(
    universe=TOY_UNIVERSE,
    scope=frozenset({TOY_GADGET}),
    nodes=(ALPHA, RED),
    edges=(EdgeRecord(property_iri=TOY_HAS_COLOR, start=ALPHA.node_ref, end=RED.node_ref),),
    claims=(
        EdgeClaim(
            anchor=ALPHA.node_ref,
            property_iri=TOY_HAS_COLOR,
            direction=ClaimDirection.OUT,
            other_class_iri=TOY_COLOR,
        ),
    ),
)
# A policy that owns nothing, so every batch with a scope breaks it.
EMPTY_POLICY = WritePolicy(
    job="toy_job",
    owned_classes=frozenset(),
    owner_properties=frozenset(),
    required_owner_properties=frozenset(),
    edge_properties=frozenset(),
    claim_rules=(),
)

STORED_NODE = {
    "key_label": "ToyGadget",
    "id": "alpha",
    "labels": ["muontoy", "ToyGadget", "Object", "Entity"],
    "properties": [["id", "alpha"], ["name", "Alpha"], ["serial", 7]],
}
STORED_EDGE = {
    "type": "HAS_COLOR",
    "start_label": "ToyGadget",
    "start_id": "alpha",
    "end_label": "ToyColor",
    "end_id": "red",
    "properties": [],
}


class GraphFake(RecordingGraphClient):
    """Answers each read by its query, with `records` of that query or no row."""

    def __init__(self, records: Mapping[str, list[dict[str, Any]]] | None = None) -> None:
        super().__init__()
        self.records = dict(records or {})

    async def execute_read(
        self,
        query: LiteralString,
        parameters: Mapping[str, Any] | None = None,
        max_records: int | None = None,
    ) -> list[dict[str, Any]]:
        self.reads.append((query, parameters))
        return self.records.get(query, [])


class ValidatorFake:
    def __init__(self, violations: Sequence[Violation] = ()) -> None:
        self.violations = tuple(violations)
        self.batches: list[GraphBatch] = []

    async def validate(self, batch: GraphBatch) -> ValidationReport:
        self.batches.append(batch)
        return ValidationReport(violations=self.violations)


@pytest.fixture(scope="module")
def ontology_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = copy_toy_ontology(tmp_path_factory.mktemp("ontology"))
    (directory / "shapes").mkdir()
    (directory / "shapes" / f"{TOY_UNIVERSE}.shacl.ttl").write_text("", encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def materializer(ontology_dir: Path) -> LabelMaterializer:
    return LabelMaterializer(load_ontology_model(ontology_dir, TOY_UNIVERSE))


@pytest.fixture
def build_context(ontology_dir: Path, materializer: LabelMaterializer, tmp_path: Path):
    def build(client: GraphFake, validator: ValidatorFake, folder: str = "plans") -> PlanContext:
        return PlanContext(
            client=client,
            runner=Runner(client, materializer),
            validator=validator,
            materializer=materializer,
            ontology_dir=ontology_dir,
            universe=TOY_UNIVERSE,
            run_label=RUN_LABEL,
            root=tmp_path / folder,
        )

    return build


def build_converged_client() -> GraphFake:
    """A graph that holds exactly what `BATCH` writes."""
    return GraphFake({plan.READ_OWNER_NODES: [STORED_NODE], plan.READ_LISTED_EDGES: [STORED_EDGE]})


async def test_policy_rejection_stops_before_the_validator(build_context) -> None:
    client, validator = GraphFake(), ValidatorFake()
    writer = open_plan(build_context(client, validator))

    item = await writer.plan_item("source", BATCH, EMPTY_POLICY)

    assert item.state is ItemState.REJECTED
    assert {violation.rule_id for violation in item.violations} == {"P1", "P3", "P5", "P6"}
    assert validator.batches == []
    assert client.reads == []
    assert item.statements == ()


async def test_validator_rejection_stops_before_the_preview(build_context) -> None:
    client = GraphFake()
    validator = ValidatorFake([Violation(rule_id="S1", message="broken")])
    writer = open_plan(build_context(client, validator))

    item = await writer.plan_item("source", BATCH, None)

    assert item.state is ItemState.REJECTED
    assert [violation.rule_id for violation in item.violations] == ["S1"]
    assert client.reads == []
    assert (item.preview, item.diff, item.statements) == (None, None, ())


async def test_delete_outside_the_scope_stops_before_the_diff(build_context) -> None:
    orphan = {
        "type": "HAS_COLOR",
        "start_labels": ["ToyGadget", "muontoy"],
        "start_id": "alpha",
        "end_labels": ["ToyColor", "muontoy"],
        "end_id": "blue",
        "orphan": {"labels": ["ToyColor", "muontoy"], "id": "blue"},
    }
    client = GraphFake({runner.PREVIEW_CLAIMED_OUT: [orphan]})
    writer = open_plan(build_context(client, ValidatorFake()))

    item = await writer.plan_item("source", BATCH, None)

    assert item.state is ItemState.REJECTED
    assert [violation.rule_id for violation in item.violations] == ["D1"]
    assert item.preview is not None and [node.id for node in item.preview.nodes] == ["blue"]
    assert [query for query, _ in client.reads] == [runner.PREVIEW_CLAIMED_OUT]


async def test_batch_that_changes_nothing_is_no_change(build_context) -> None:
    writer = open_plan(build_context(build_converged_client(), ValidatorFake()))

    item = await writer.plan_item("source", BATCH, None)

    assert item.state is ItemState.NO_CHANGE
    assert item.diff is not None and item.diff.is_empty
    assert len(item.statements) == 4


async def test_batch_against_an_empty_graph_adds_its_nodes_and_edges(build_context) -> None:
    missing_stub = {"class_iri": TOY_COLOR, "id": "red"}
    client = GraphFake({seed.FIND_MISSING_NODES: [missing_stub]})
    writer = open_plan(build_context(client, ValidatorFake()))

    item = await writer.plan_item("source", BATCH, None)

    assert item.state is ItemState.PLANNED
    assert item.diff is not None
    assert item.diff.nodes_added == (ALPHA.node_ref,)
    assert item.diff.stubs_created == (RED.node_ref,)
    assert [(edge.start_id, edge.end_id) for edge in item.diff.edges_added] == [("alpha", "red")]
    assert client.calls == []


async def test_diff_reports_changed_values_and_labels(build_context) -> None:
    stored = STORED_NODE | {
        "labels": ["muontoy", "ToyGadget", "ToyWidget", "Imported"],
        "properties": [["id", "alpha"], ["name", "Alpha"], ["serial", 7.0]],
    }
    client = GraphFake({plan.READ_OWNER_NODES: [stored], plan.READ_LISTED_EDGES: [STORED_EDGE]})
    writer = open_plan(build_context(client, ValidatorFake()))

    item = await writer.plan_item("source", BATCH, None)

    assert item.diff is not None
    [change] = item.diff.nodes_changed
    assert [(c.name, c.before, c.after) for c in change.property_changes] == [("serial", 7.0, 7)]
    assert change.labels_added == ("Entity", "Object")
    # A label outside the ontology stays on the node.
    assert change.labels_removed == ("ToyWidget",)


async def test_2_plans_of_1_input_give_equal_batch_files_with_sorted_keys(build_context) -> None:
    paths = []
    for folder in ("first", "second"):
        context = build_context(build_converged_client(), ValidatorFake(), folder)
        header = await plan_batches(context, [("source", BATCH, None)])
        plan_dir = context.root / TOY_UNIVERSE / header.run_id
        paths.append(plan_dir / plan.BATCHES_DIR_NAME / "00001.json")

    first, second = (path.read_bytes() for path in paths)

    assert first == second
    stored = json.loads(first)
    assert list(stored) == sorted(stored)
    assert stored["state"] == "no_change"
    assert plan.PlannedItem.model_validate_json(first).batch == BATCH


async def test_header_counts_the_items_by_state(build_context) -> None:
    context = build_context(build_converged_client(), ValidatorFake())

    header = await plan_batches(context, [("first", BATCH, None), ("second", BATCH, EMPTY_POLICY)])

    assert header.run_id.endswith(RUN_LABEL)
    assert header.counts == {
        ItemState.PLANNED: 0,
        ItemState.NO_CHANGE: 1,
        ItemState.REJECTED: 1,
    }
    assert header.totals == plan.DiffTotals()
    assert len(header.tbox_hash) == len(header.shapes_hash) == 64
    plan_dir = context.root / TOY_UNIVERSE / header.run_id
    stored = (plan_dir / plan.PLAN_FILE_NAME).read_text(encoding="utf-8")
    assert plan.PlanHeader.model_validate_json(stored) == header


async def test_header_totals_cover_the_planned_items(build_context) -> None:
    context = build_context(GraphFake(), ValidatorFake())

    header = await plan_batches(context, [("source", BATCH, None)])

    assert header.totals == plan.DiffTotals(nodes_added=1, edges_added=1)


async def test_opening_the_same_plan_twice_fails(build_context, monkeypatch) -> None:
    monkeypatch.setattr(plan, "RUN_ID_TIME_FORMAT", "fixed")
    context = build_context(GraphFake(), ValidatorFake())
    open_plan(context)

    with pytest.raises(PlanError, match="exists"):
        open_plan(context)
