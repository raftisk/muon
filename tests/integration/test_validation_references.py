"""Graph reference check of the Validator, against the live Neo4j in `.env`.

The test writes `muontoy` nodes only, and the shared `toy_runner` fixture purges that
universe afterwards.
"""

import pytest

from muon.graph import Neo4jGraphClient
from muon.ingestion.batch import GraphBatch, NodeRecord
from muon.ingestion.runner import Runner, WriteMode
from muon.ontology.materializer import LabelMaterializer
from muon.validation.references import find_unresolved_stubs
from muon.validation.report import RULE_REFERENCE
from tests.fixtures.toy_ontology import TOY_UNIVERSE

pytestmark = pytest.mark.integration

TOY = "https://muon.dev/ns/muontoy#"
TOY_WIDGET = TOY + "ToyWidget"
TOY_COLOR = TOY + "ToyColor"
STUB_ID = "gamma"


def build_batch(scope: frozenset[str], node: NodeRecord, can_create_stubs: bool) -> GraphBatch:
    return GraphBatch(
        universe=TOY_UNIVERSE,
        scope=scope,
        nodes=(node,),
        edges=(),
        can_create_stubs=can_create_stubs,
    )


async def test_strict_stub_fails_on_an_empty_universe_and_passes_once_written(
    toy_runner: Runner, graph_client: Neo4jGraphClient
) -> None:
    materializer: LabelMaterializer = toy_runner.materializer
    stub = NodeRecord(class_iri=TOY_WIDGET, id=STUB_ID, name="Gamma")
    strict_batch = build_batch(frozenset({TOY_COLOR}), stub, can_create_stubs=False)

    before = await find_unresolved_stubs(graph_client, materializer, strict_batch)
    owner = NodeRecord(class_iri=TOY_WIDGET, id=STUB_ID, name="Gamma")
    owner_batch = build_batch(frozenset({TOY_WIDGET}), owner, can_create_stubs=True)
    await toy_runner.write(owner_batch, WriteMode.UPSERT)
    after = await find_unresolved_stubs(graph_client, materializer, strict_batch)

    assert [(violation.rule_id, violation.value) for violation in before] == [
        (RULE_REFERENCE, STUB_ID)
    ]
    assert after == ()
