import pytest
from rdflib.namespace import RDFS

from muon.ontology.materializer import LabelMaterializer, build_plugin_materializer
from muon.ontology.model import OntologyError, OntologyModel, load_ontology_model
from muon.plugins import UniverseManifest
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology

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


def test_ancestor_labels(materializer: LabelMaterializer) -> None:
    ancestor_labels = materializer.map_ancestor_labels()

    assert len(ancestor_labels) == TOTAL_CLASSES
    assert ancestor_labels["Region"] == {"Location", "Entity"}
    assert ancestor_labels["HeldItem"] == {"Item", "Object", "Entity"}
    assert ancestor_labels["Generation"] == {"Series", "Source", "Meta"}
    assert all("pkmn" not in labels for labels in ancestor_labels.values())


TOY = "https://muon.dev/ns/muontoy#"


@pytest.fixture(scope="module")
def toy_model(tmp_path_factory: pytest.TempPathFactory) -> OntologyModel:
    ontology_dir = copy_toy_ontology(tmp_path_factory.mktemp("ontology"))
    return load_ontology_model(ontology_dir, TOY_UNIVERSE)


@pytest.mark.parametrize(
    ("local_name", "key_label"),
    [
        ("ToyGadget", "ToyGadget"),
        ("ToyWidget", "ToyGadget"),
        ("ToyGizmo", "ToyGadget"),
        ("ToyColor", "ToyColor"),
    ],
)
def test_key_label_of_a_family(toy_model: OntologyModel, local_name: str, key_label: str) -> None:
    materializer = LabelMaterializer(toy_model, ["ToyGadget"])

    assert materializer.map_key_label(TOY + local_name) == key_label


def test_key_label_of_a_family_sibling_matches(toy_model: OntologyModel) -> None:
    materializer = LabelMaterializer(toy_model, ["ToyGadget"])

    assert materializer.map_key_label(TOY + "ToyWidget") == materializer.map_key_label(
        TOY + "ToyGizmo"
    )


def test_key_label_without_key_classes_is_the_class_label(toy_model: OntologyModel) -> None:
    materializer = LabelMaterializer(toy_model)

    assert materializer.map_key_label(TOY + "ToyWidget") == "ToyWidget"


def test_unknown_key_class_names_the_class(toy_model: OntologyModel) -> None:
    with pytest.raises(OntologyError, match="'ToyNothing'"):
        LabelMaterializer(toy_model, ["ToyNothing"])


def test_nested_key_classes_name_both(toy_model: OntologyModel) -> None:
    with pytest.raises(OntologyError, match="'ToyWidget'.*'ToyGadget'"):
        LabelMaterializer(toy_model, ["ToyGadget", "ToyWidget"])


def test_plugin_materializer_uses_the_manifest_key_classes(toy_model: OntologyModel) -> None:
    manifest = UniverseManifest(
        prefix="muontoy",
        name="Toy",
        notes="notes.md",
        namespace="https://muon.dev/ns/muontoy#",
        shapes_namespace="https://muon.dev/ns/muontoy-shapes#",
        key_classes=("ToyGadget",),
    )

    materializer = build_plugin_materializer(toy_model, manifest)

    assert materializer.map_key_label(TOY + "ToyGizmo") == "ToyGadget"
