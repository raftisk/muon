import pytest
from rdflib import Graph

from muon.ingestion.schema import (
    build_id_constraint,
    build_name_index_statement,
    ensure_id_constraints,
    ensure_name_index,
    list_constraint_labels,
)
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyError, build_ontology_model
from tests.unit.ingestion.recording_client import RecordingGraphClient

EX = "https://example.org/fixture#"
TBOX = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <https://example.org/fixture#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:Type a owl:Class .
ex:Color a owl:Class .
ex:Item a owl:Class .
ex:Berry a owl:Class ; rdfs:subClassOf ex:Item .
<https://example.org/fixture#Bad-Name> a owl:Class .
"""


@pytest.fixture(scope="module")
def materializer() -> LabelMaterializer:
    tbox = Graph().parse(data=TBOX, format="turtle")
    return LabelMaterializer(build_ontology_model(tbox, "ex"))


@pytest.fixture(scope="module")
def key_materializer() -> LabelMaterializer:
    tbox = Graph().parse(data=TBOX, format="turtle")
    return LabelMaterializer(build_ontology_model(tbox, "ex"), ["Item"])


def test_constraint_statement() -> None:
    statement = build_id_constraint("Type")

    assert statement == (
        "CREATE CONSTRAINT `id_unique_Type` IF NOT EXISTS FOR (n:`Type`) REQUIRE n.id IS UNIQUE"
    )


def test_class_outside_model_raises(materializer: LabelMaterializer) -> None:
    with pytest.raises(OntologyError, match="not a named class"):
        list_constraint_labels(materializer, [EX + "Unknown"])


def test_label_failing_pattern_raises() -> None:
    with pytest.raises(OntologyError, match="'Bad-Name', which does not match"):
        build_id_constraint("Bad-Name")


def test_constraint_labels_hold_leaf_and_key_labels(key_materializer: LabelMaterializer) -> None:
    labels = list_constraint_labels(key_materializer, [EX + "Berry", EX + "Item", EX + "Type"])

    assert labels == ("Berry", "Item", "Type")


def test_constraint_labels_without_key_classes_are_the_class_labels(
    materializer: LabelMaterializer,
) -> None:
    assert list_constraint_labels(materializer, [EX + "Type", EX + "Color"]) == ("Color", "Type")


async def test_ensure_id_constraints_adds_the_key_label(
    key_materializer: LabelMaterializer,
) -> None:
    client = RecordingGraphClient()

    count = await ensure_id_constraints(client, key_materializer, [EX + "Berry"])

    assert count == 2
    assert [write.split("`")[1] for write in client.writes] == [
        "id_unique_Berry",
        "id_unique_Item",
    ]


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


def test_name_index_statement() -> None:
    assert build_name_index_statement() == (
        "CREATE FULLTEXT INDEX `entity_names` IF NOT EXISTS "
        "FOR (n:`Entity`|`Work`|`Series`) ON EACH [n.`name`, n.`aliases`]"
    )


async def test_ensure_name_index_sends_one_write() -> None:
    client = RecordingGraphClient()

    await ensure_name_index(client)

    assert client.writes == [build_name_index_statement()]
