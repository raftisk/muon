import shutil
from pathlib import Path

import pytest
from rdflib import Graph

from muon.ontology.model import (
    OntologyError,
    OntologyModel,
    build_ontology_model,
    extract_local_name,
    load_ontology_model,
)
from tests.unit.ontology.conftest import REAL_ONTOLOGY_DIR

MO = "https://muon.dev/ns/core#"
PKMN = "https://muon.dev/ns/pkmn#"
EX = "https://example.org/fixture#"
TOTAL_CLASSES = 71
TOTAL_OBJECT_PROPERTIES = 67
TOTAL_DATATYPE_PROPERTIES = 68
TBOX_PREFIXES = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <https://example.org/fixture#> .
"""


def parse_tbox(body: str) -> Graph:
    return Graph().parse(data=TBOX_PREFIXES + body, format="turtle")


def write_ontology_dir(target: Path, universe: str, universe_ttl: str | None) -> Path:
    shutil.copytree(REAL_ONTOLOGY_DIR / "core", target / "core")
    (target / "universes").mkdir()
    if universe_ttl is not None:
        (target / "universes" / f"{universe}.ttl").write_text(universe_ttl)
    return target


def test_real_model_counts(real_model: OntologyModel) -> None:
    assert len(real_model.classes) == TOTAL_CLASSES
    assert len(real_model.object_properties) == TOTAL_OBJECT_PROPERTIES
    assert len(real_model.datatype_properties) == TOTAL_DATATYPE_PROPERTIES


def test_real_model_ancestors(real_model: OntologyModel) -> None:
    assert real_model.ancestors[PKMN + "Type"] == {MO + "Concept", MO + "Entity"}
    assert real_model.ancestors[PKMN + "Generation"] == {MO + "Series", MO + "Source", MO + "Meta"}


def test_real_model_holds_plain_string_iris(real_model: OntologyModel) -> None:
    assert all(type(class_iri) is str for class_iri in real_model.classes)
    assert all(iri.startswith("https://") for iri in real_model.classes)


def test_subclass_cycle_terminates() -> None:
    tbox = parse_tbox(
        "ex:A a owl:Class ; rdfs:subClassOf ex:B .\nex:B a owl:Class ; rdfs:subClassOf ex:A .\n"
    )

    model = build_ontology_model(tbox, "ex")

    assert model.ancestors[EX + "A"] == {EX + "B"}
    assert model.ancestors[EX + "B"] == {EX + "A"}


def test_union_superclass_adds_nothing() -> None:
    tbox = parse_tbox(
        "ex:A a owl:Class ; rdfs:subClassOf [ a owl:Class ; owl:unionOf ( ex:B ex:C ) ] .\n"
        "ex:B a owl:Class .\nex:C a owl:Class .\n"
    )

    model = build_ontology_model(tbox, "ex")

    assert model.classes == {EX + "A", EX + "B", EX + "C"}
    assert model.ancestors[EX + "A"] == frozenset()


@pytest.mark.parametrize(
    ("iri", "local_name"),
    [
        ("https://muon.dev/ns/core#Entity", "Entity"),
        ("https://example.org/terms/Widget", "Widget"),
    ],
)
def test_extract_local_name(iri: str, local_name: str) -> None:
    assert extract_local_name(iri) == local_name


@pytest.mark.parametrize("universe", ["Pk-mn", "../x", ""])
def test_invalid_prefix_raises(tmp_path: Path, universe: str) -> None:
    with pytest.raises(OntologyError, match="Invalid universe prefix"):
        load_ontology_model(tmp_path, universe)


def test_empty_core_raises(tmp_path: Path) -> None:
    (tmp_path / "core").mkdir()

    with pytest.raises(OntologyError, match="No core ttl files"):
        load_ontology_model(tmp_path, "ex")


def test_missing_universe_file_names_path(tmp_path: Path) -> None:
    ontology_dir = write_ontology_dir(tmp_path, "ex", universe_ttl=None)

    with pytest.raises(OntologyError) as error_info:
        load_ontology_model(ontology_dir, "ex")

    assert str(ontology_dir / "universes" / "ex.ttl") in str(error_info.value)


def test_broken_turtle_names_path(tmp_path: Path) -> None:
    ontology_dir = write_ontology_dir(tmp_path, "ex", universe_ttl="ex:A a owl:Class")

    with pytest.raises(OntologyError) as error_info:
        load_ontology_model(ontology_dir, "ex")

    assert str(ontology_dir / "universes" / "ex.ttl") in str(error_info.value)
