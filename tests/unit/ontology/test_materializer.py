import pytest
from rdflib.namespace import RDFS

from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel

PKMN = "https://muon.dev/ns/pkmn#"
TOTAL_CLASSES = 71


@pytest.fixture(scope="module")
def materializer(real_model: OntologyModel) -> LabelMaterializer:
    return LabelMaterializer(real_model)


@pytest.mark.parametrize(
    ("local_name", "labels"),
    [
        ("Type", ("Type", "Concept", "Entity", "pkmn")),
        ("Generation", ("Generation", "Meta", "Series", "Source", "pkmn")),
    ],
)
def test_class_labels(materializer: LabelMaterializer, local_name: str, labels: tuple[str]) -> None:
    assert materializer.map_class_labels(PKMN + local_name) == labels


def test_core_class_labels(materializer: LabelMaterializer) -> None:
    labels = materializer.map_class_labels("https://muon.dev/ns/core#Universe")

    assert labels == ("Universe", "Meta", "pkmn")


@pytest.mark.parametrize(
    ("local_name", "relationship_type"),
    [
        ("notVeryEffectiveAgainst", "NOT_VERY_EFFECTIVE_AGAINST"),
        ("superEffectiveAgainst", "SUPER_EFFECTIVE_AGAINST"),
        ("introducedIn", "INTRODUCED_IN"),
        ("boostsStat", "BOOSTS_STAT"),
    ],
)
def test_relationship_types(
    materializer: LabelMaterializer, local_name: str, relationship_type: str
) -> None:
    assert materializer.map_relationship_type(PKMN + local_name) == relationship_type


def test_property_names(materializer: LabelMaterializer) -> None:
    assert materializer.map_property_name(str(RDFS.label)) == "name"
    assert materializer.map_property_name(PKMN + "baseSpeed") == "baseSpeed"


def test_ontology_labels(materializer: LabelMaterializer) -> None:
    labels = materializer.list_ontology_labels()

    assert len(labels) == TOTAL_CLASSES
    assert "Concept" in labels
