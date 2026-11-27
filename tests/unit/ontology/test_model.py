import shutil
from pathlib import Path

import pytest
from rdflib import Graph

from muon.ontology.model import (
    OntologyError,
    OntologyModel,
    build_ontology_model,
    extract_local_name,
    list_ontology_paths,
    load_ontology_model,
    load_tbox,
)
from tests.unit.ontology.conftest import REAL_ONTOLOGY_DIR

MO = "https://muon.dev/ns/core#"
PKMN = "https://muon.dev/ns/pkmn#"
EX = "https://example.org/fixture#"
TOTAL_CLASSES = 73
TOTAL_OBJECT_PROPERTIES = 73
TOTAL_DATATYPE_PROPERTIES = 77
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


def test_union_range_expands_to_its_members() -> None:
    tbox = parse_tbox(
        "ex:A a owl:Class .\nex:B a owl:Class .\nex:C a owl:Class .\n"
        "ex:p a owl:ObjectProperty ; rdfs:domain ex:A ;"
        " rdfs:range [ a owl:Class ; owl:unionOf ( ex:B ex:C ) ] .\n"
    )

    model = build_ontology_model(tbox, "ex")

    assert model.domains[EX + "p"] == {EX + "A"}
    assert model.ranges[EX + "p"] == {EX + "B", EX + "C"}


def test_sub_property_inherits_domain_and_range() -> None:
    tbox = parse_tbox(
        "ex:A a owl:Class .\nex:B a owl:Class .\n"
        "ex:p a owl:ObjectProperty ; rdfs:domain ex:A ; rdfs:range ex:B .\n"
        "ex:q a owl:ObjectProperty ; rdfs:subPropertyOf ex:p .\n"
        "ex:r a owl:ObjectProperty ; rdfs:subPropertyOf ex:q .\n"
    )

    model = build_ontology_model(tbox, "ex")

    assert model.domains[EX + "r"] == {EX + "A"}
    assert model.ranges[EX + "q"] == {EX + "B"}


def test_own_range_wins_over_the_inherited_one() -> None:
    tbox = parse_tbox(
        "ex:A a owl:Class .\nex:B a owl:Class .\n"
        "ex:p a owl:ObjectProperty ; rdfs:range ex:A .\n"
        "ex:q a owl:ObjectProperty ; rdfs:subPropertyOf ex:p ; rdfs:range ex:B .\n"
    )

    model = build_ontology_model(tbox, "ex")

    assert model.ranges[EX + "q"] == {EX + "B"}


def test_property_without_domain_or_range_has_no_entry() -> None:
    tbox = parse_tbox("ex:p a owl:ObjectProperty .\n")

    model = build_ontology_model(tbox, "ex")

    assert EX + "p" not in model.domains
    assert EX + "p" not in model.ranges


def test_sub_property_cycle_terminates() -> None:
    tbox = parse_tbox(
        "ex:p a owl:ObjectProperty ; rdfs:subPropertyOf ex:q .\n"
        "ex:q a owl:ObjectProperty ; rdfs:subPropertyOf ex:p .\n"
    )

    model = build_ontology_model(tbox, "ex")

    assert model.ranges == {}


def test_datatype_range_keeps_only_xsd_ranges() -> None:
    tbox = parse_tbox(
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
        "ex:A a owl:Class .\n"
        "ex:n a owl:DatatypeProperty ; rdfs:range xsd:integer .\n"
        "ex:m a owl:DatatypeProperty ; rdfs:range ex:A .\n"
        "ex:o a owl:DatatypeProperty .\n"
    )

    model = build_ontology_model(tbox, "ex")

    assert model.datatype_ranges == {EX + "n": "http://www.w3.org/2001/XMLSchema#integer"}


def test_real_model_reads_pkmn_ranges(real_model: OntologyModel) -> None:
    assert real_model.ranges[PKMN + "hasType"] == {PKMN + "Type"}
    assert real_model.domains[PKMN + "hasType"] == {PKMN + "PokemonSpecies", PKMN + "Move"}
    assert real_model.datatype_ranges[PKMN + "amount"] == "http://www.w3.org/2001/XMLSchema#integer"


def test_real_sub_property_inherits_the_has_type_range(real_model: OntologyModel) -> None:
    assert real_model.ranges[PKMN + "primaryType"] == {PKMN + "Type"}


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


def test_list_ontology_paths_gives_core_files_sorted_then_the_universe_file() -> None:
    paths = list_ontology_paths(REAL_ONTOLOGY_DIR, "pkmn")

    core_paths = sorted((REAL_ONTOLOGY_DIR / "core").glob("*.ttl"))
    assert paths == (*core_paths, REAL_ONTOLOGY_DIR / "universes" / "pkmn.ttl")


@pytest.mark.parametrize("universe", ["Pk-mn", "../x", ""])
def test_list_ontology_paths_rejects_an_invalid_prefix(tmp_path: Path, universe: str) -> None:
    with pytest.raises(OntologyError, match="Invalid universe prefix"):
        list_ontology_paths(tmp_path, universe)


def test_list_ontology_paths_rejects_an_empty_core(tmp_path: Path) -> None:
    (tmp_path / "core").mkdir()

    with pytest.raises(OntologyError, match="No core ttl files"):
        list_ontology_paths(tmp_path, "ex")


def test_list_ontology_paths_names_a_missing_universe_file(tmp_path: Path) -> None:
    ontology_dir = write_ontology_dir(tmp_path, "ex", universe_ttl=None)

    with pytest.raises(OntologyError, match="Missing universe T-Box") as error_info:
        list_ontology_paths(ontology_dir, "ex")

    assert str(ontology_dir / "universes" / "ex.ttl") in str(error_info.value)


def test_load_tbox_raises_on_broken_turtle(tmp_path: Path) -> None:
    ontology_dir = write_ontology_dir(tmp_path, "ex", universe_ttl="ex:A a owl:Class")

    with pytest.raises(OntologyError, match="Failed to parse"):
        load_tbox(ontology_dir, "ex")
