import pytest
from rdflib import Graph

from muon.ingestion.schema import build_id_constraint, ensure_id_constraints
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyError, build_ontology_model
from tests.unit.ingestion.recording_client import RecordingGraphClient

EX = "https://example.org/fixture#"
TBOX = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <https://example.org/fixture#> .
ex:Type a owl:Class .
ex:Color a owl:Class .
<https://example.org/fixture#Bad-Name> a owl:Class .
"""


@pytest.fixture(scope="module")
def materializer() -> LabelMaterializer:
    tbox = Graph().parse(data=TBOX, format="turtle")
    return LabelMaterializer(build_ontology_model(tbox, "ex"))


def test_constraint_statement(materializer: LabelMaterializer) -> None:
    statement = build_id_constraint(materializer, EX + "Type")

    assert statement == (
        "CREATE CONSTRAINT `id_unique_Type` IF NOT EXISTS FOR (n:`Type`) REQUIRE n.id IS UNIQUE"
    )


def test_class_outside_model_raises(materializer: LabelMaterializer) -> None:
    with pytest.raises(OntologyError, match="not a named class"):
        build_id_constraint(materializer, EX + "Unknown")


def test_label_failing_pattern_raises(materializer: LabelMaterializer) -> None:
    with pytest.raises(OntologyError, match="'Bad-Name' does not match"):
        build_id_constraint(materializer, EX + "Bad-Name")


async def test_ensure_id_constraints_writes_one_per_class(
    materializer: LabelMaterializer,
) -> None:
    client = RecordingGraphClient()

    count = await ensure_id_constraints(client, materializer, [EX + "Type", EX + "Color"])

    assert count == 2
    assert [write.split("`")[1] for write in client.writes] == [
        "id_unique_Color",
        "id_unique_Type",
    ]
    assert client.batches == []
