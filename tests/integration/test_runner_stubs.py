"""The stub list of the Runner, against the live Neo4j in `.env`.

The test writes `muontoy` nodes only, and the shared `toy_runner` fixture purges that
universe afterwards.
"""

import pytest

from muon.ingestion.batch import GraphBatch, NodeRecord
from muon.ingestion.runner import Runner, WriteMode
from tests.fixtures.toy_ontology import TOY_UNIVERSE

pytestmark = pytest.mark.integration

TOY = "https://muon.dev/ns/muontoy#"
TOY_WIDGET = TOY + "ToyWidget"
TOY_COLOR = TOY + "ToyColor"
TOY_WEIGHT = TOY + "weight"


def build_batch(scope: frozenset[str], node: NodeRecord) -> GraphBatch:
    return GraphBatch(universe=TOY_UNIVERSE, scope=scope, nodes=(node,), edges=())


async def test_stub_ids_leave_out_an_owner_of_the_same_label(toy_runner: Runner) -> None:
    owner = NodeRecord(class_iri=TOY_WIDGET, id="alpha", name="Alpha", properties={TOY_WEIGHT: 1.5})
    stub = NodeRecord(class_iri=TOY_WIDGET, id="gamma", name="Gamma")
    await toy_runner.write(build_batch(frozenset({TOY_WIDGET}), owner), WriteMode.UPSERT)
    await toy_runner.write(build_batch(frozenset({TOY_COLOR}), stub), WriteMode.UPSERT)

    stub_ids = await toy_runner.list_stub_ids([TOY_WIDGET])

    assert stub_ids == {"ToyWidget": ("gamma",)}
