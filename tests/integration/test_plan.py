"""Planning against the live Neo4j in `.env`.

Every test writes `muontoy` nodes only, and the shared `toy_runner` fixture purges that
universe afterwards. The Validator is a fake that accepts every batch.
"""

from pathlib import Path

import pytest

from muon.graph import Neo4jGraphClient
from muon.ingestion.batch import ClaimDirection, GraphBatch, NodeRecord
from muon.ingestion.plan import ItemState, PlanContext, check_deletes, open_plan
from muon.ingestion.runner import Runner, WriteMode
from muon.validation import ValidationReport
from tests.fixtures.toy_ontology import TOY_UNIVERSE
from tests.integration.test_runner_claims import (
    ALPHA,
    BLUE,
    FIRST_WIDGET,
    RED,
    TOY_COLOR,
    TOY_GADGET,
    TOY_HAS_COLOR,
    TOY_PAIRS_WITH,
    TOY_WIDGET,
    build_batch,
    build_claim,
    build_edge,
)

pytestmark = pytest.mark.integration

SCOPE = frozenset({TOY_GADGET, TOY_WIDGET})
CLAIMS = (build_claim(ALPHA, TOY_HAS_COLOR, ClaimDirection.OUT, TOY_COLOR),)
SERIAL = "https://muon.dev/ns/muontoy#serial"


class AcceptingValidator:
    async def validate(self, batch: GraphBatch) -> ValidationReport:
        return ValidationReport()


def build_gadget(serial: int) -> NodeRecord:
    return NodeRecord(class_iri=TOY_GADGET, id="alpha", name="Alpha", properties={SERIAL: serial})


def build_first_batch() -> GraphBatch:
    gadget = build_gadget(1)
    edges = (build_edge(TOY_HAS_COLOR, gadget, RED),)
    return build_batch(SCOPE, (gadget, RED), edges, CLAIMS)


def build_edited_batch() -> GraphBatch:
    """Alpha gets a new serial, swaps red for blue and pairs with a new widget."""
    gadget = build_gadget(2)
    edges = (
        build_edge(TOY_HAS_COLOR, gadget, BLUE),
        build_edge(TOY_PAIRS_WITH, gadget, FIRST_WIDGET),
    )
    return build_batch(SCOPE, (gadget, BLUE, FIRST_WIDGET), edges, CLAIMS)


@pytest.fixture
def plan_context(toy_runner: Runner, graph_client: Neo4jGraphClient, tmp_path: Path):
    shapes_dir = tmp_path / "shapes"
    shapes_dir.mkdir()
    (shapes_dir / f"{TOY_UNIVERSE}.shacl.ttl").write_text("", encoding="utf-8")
    return PlanContext(
        client=graph_client,
        runner=toy_runner,
        validator=AcceptingValidator(),
        materializer=toy_runner.materializer,
        ontology_dir=tmp_path,
        universe=TOY_UNIVERSE,
        run_label="toy",
        root=tmp_path / "plans",
    )


async def test_plan_lists_changes_deletes_and_statements_without_a_write(
    toy_runner: Runner, plan_context: PlanContext
) -> None:
    await toy_runner.write(build_first_batch(), WriteMode.UPSERT)
    census_before = await toy_runner.count_universe()

    item = await open_plan(plan_context).plan_item("toy", build_edited_batch(), None)

    assert item.state is ItemState.PLANNED
    assert item.diff is not None and item.preview is not None
    [change] = item.diff.nodes_changed
    assert [(c.name, c.before, c.after) for c in change.property_changes] == [("serial", 1, 2)]
    assert [ref.id for ref in item.diff.nodes_added] == ["w1"]
    assert [ref.id for ref in item.diff.stubs_created] == ["blue"]
    assert sorted((edge.type, edge.end_id) for edge in item.diff.edges_added) == [
        ("HAS_COLOR", "blue"),
        ("PAIRS_WITH", "w1"),
    ]
    assert [(edge.type, edge.end_id) for edge in item.preview.edges] == [("HAS_COLOR", "red")]
    assert item.preview.nodes == ()
    assert item.statements
    assert await toy_runner.count_universe() == census_before


async def test_plan_of_a_written_batch_is_no_change(
    toy_runner: Runner, plan_context: PlanContext
) -> None:
    await toy_runner.write(build_first_batch(), WriteMode.UPSERT)

    item = await open_plan(plan_context).plan_item("toy", build_first_batch(), None)

    assert item.state is ItemState.NO_CHANGE


async def test_claim_that_reaches_a_node_outside_the_scope_is_rejected(
    toy_runner: Runner,
) -> None:
    await toy_runner.write(build_first_batch(), WriteMode.UPSERT)
    census_before = await toy_runner.count_universe()
    gadget = build_gadget(1)
    # `GraphBatch` refuses to own an end outside the scope, so the batch skips validation. A
    # plan item would validate it again, so the test runs the preview and the guard itself.
    batch = GraphBatch.model_construct(
        universe=TOY_UNIVERSE,
        scope=SCOPE,
        nodes=(gadget,),
        edges=(),
        claims=(build_claim(gadget, TOY_HAS_COLOR, ClaimDirection.OUT, TOY_COLOR, True),),
        can_create_stubs=False,
    )

    preview = await toy_runner.preview_deletes(batch)
    violations = check_deletes(preview, batch, toy_runner.materializer)

    assert [(violation.rule_id, violation.value) for violation in violations] == [("D1", "red")]
    assert [node.id for node in preview.nodes] == ["red"]
    assert await toy_runner.count_universe() == census_before
