"""Key class lookups of the Runner, against the live Neo4j in `.env`.

The toy universe names `ToyGadget` as its key class, so a `ToyWidget` stub and a `ToyGizmo`
owner of the same id are 1 node. The `toy_runner` fixture of `conftest.py` has no key
classes, so this module builds its own runner and purges the universe afterwards.
"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import LiteralString

import pytest

from muon.graph import Neo4jGraphClient
from muon.ingestion.batch import EdgeRecord, GraphBatch, NodeRecord
from muon.ingestion.runner import Runner, RunReport, WriteMode
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology

pytestmark = pytest.mark.integration

TOY = "https://muon.dev/ns/muontoy#"
TOY_GADGET = TOY + "ToyGadget"
TOY_WIDGET = TOY + "ToyWidget"
TOY_GIZMO = TOY + "ToyGizmo"
TOY_PAIRS_WITH = TOY + "pairsWith"
SHARED_ID = "x"
PARTNER_ID = "partner"

READ_SHARED_NODE: LiteralString = """
MATCH (n:ToyGadget:$($universe) {id: $id})
OPTIONAL MATCH (:$($universe))-[r:PAIRS_WITH]->(n)
RETURN labels(n) AS labels, count(r) AS incoming
"""


@pytest.fixture
async def key_runner(graph_client: Neo4jGraphClient, tmp_path: Path) -> AsyncIterator[Runner]:
    model = load_ontology_model(copy_toy_ontology(tmp_path), TOY_UNIVERSE)
    runner = Runner(graph_client, LabelMaterializer(model, ["ToyGadget"]))
    yield runner
    await runner.purge()


def build_stub_batch() -> GraphBatch:
    partner = NodeRecord(class_iri=TOY_GIZMO, id=PARTNER_ID, name="Partner")
    stub = NodeRecord(class_iri=TOY_WIDGET, id=SHARED_ID, name="X")
    edge = EdgeRecord(property_iri=TOY_PAIRS_WITH, start=partner.node_ref, end=stub.node_ref)
    return GraphBatch(
        universe=TOY_UNIVERSE,
        scope=frozenset({TOY_GIZMO}),
        nodes=(partner, stub),
        edges=(edge,),
    )


def build_owner_batch() -> GraphBatch:
    owner = NodeRecord(class_iri=TOY_GIZMO, id=SHARED_ID, name="X")
    return GraphBatch(universe=TOY_UNIVERSE, scope=frozenset({TOY_GIZMO}), nodes=(owner,), edges=())


def has_changes(report: RunReport) -> bool:
    return any(
        (
            report.nodes_created,
            report.nodes_deleted,
            report.relationships_created,
            report.relationships_deleted,
        )
    )


async def test_owner_of_a_sibling_class_takes_over_the_stub(
    key_runner: Runner, graph_client: Neo4jGraphClient
) -> None:
    await key_runner.write(build_stub_batch(), WriteMode.UPSERT)
    await key_runner.write(build_owner_batch(), WriteMode.UPSERT)

    (record,) = await graph_client.execute_read(
        READ_SHARED_NODE, {"universe": TOY_UNIVERSE, "id": SHARED_ID}
    )

    assert "ToyGizmo" in record["labels"]
    assert "ToyGadget" in record["labels"]
    assert "ToyWidget" not in record["labels"]
    assert record["incoming"] == 1


async def test_rerun_of_the_stub_batch_changes_nothing_after_the_owner(
    key_runner: Runner,
) -> None:
    await key_runner.write(build_stub_batch(), WriteMode.UPSERT)
    await key_runner.write(build_owner_batch(), WriteMode.UPSERT)

    report = await key_runner.write(build_stub_batch(), WriteMode.UPSERT)

    assert not has_changes(report)
