from pathlib import Path

import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, SKOS

PKMN_TTL_PATH = Path(__file__).resolve().parents[3] / "ontology" / "universes" / "pkmn.ttl"
PKMN = Namespace("https://muon.dev/ns/pkmn#")
ONTOLOGY_IRI = URIRef("https://muon.dev/ns/pkmn")
VERSION = "0.6.0"
ENGLISH = "en"
DATA_SOURCES = ("bulbapedia", "pokeapi")
TERM_COUNTS = {OWL.Class: 37, OWL.ObjectProperty: 47, OWL.DatatypeProperty: 67}

MOVE_TERMS = (
    "Move",
    "MoveDamageClass",
    "MoveFlag",
    "MoveEffect",
    "Machine",
    "LearnsetEntry",
    "hasType",
    "hasDamageClass",
    "hasFlag",
    "hasEffect",
    "power",
    "accuracy",
    "pp",
    "priority",
    "target",
    "critStage",
    "minHits",
    "maxHits",
    "effectText",
    "teachesMove",
    "machineNumber",
    "machineKind",
    "isReusable",
    "hasLearnsetEntry",
    "inVersionGroup",
    "viaMachine",
    "viaParent",
    "canLearn",
    "level",
    "method",
    "isProtectable",
    "isMirrorable",
    "isOverworld",
    "minTurns",
    "maxTurns",
)

UNION_DOMAINS = {
    "isProtectable": {"Move", "AbilityEffect"},
    "isOverworld": {"MoveEffect", "AbilityEffect"},
    "minTurns": {"Move", "BattleCondition"},
    "maxTurns": {"Move", "BattleCondition"},
}


@pytest.fixture(scope="module")
def tbox() -> Graph:
    return Graph().parse(PKMN_TTL_PATH, format="turtle")


def read_union_domain(tbox: Graph, term: URIRef) -> set[str]:
    domain = tbox.value(term, RDFS.domain)
    members = tbox.value(domain, OWL.unionOf)
    assert members is not None, f"{term} has no union domain"
    return {str(member).removeprefix(str(PKMN)) for member in Collection(tbox, members)}


@pytest.mark.parametrize("local_name", MOVE_TERMS)
def test_move_term_has_english_label_and_definition(tbox: Graph, local_name: str) -> None:
    term = PKMN[local_name]
    labels = [label for label in tbox.objects(term, RDFS.label) if isinstance(label, Literal)]

    assert any(label.language == ENGLISH for label in labels)
    assert tbox.value(term, SKOS.definition) is not None


@pytest.mark.parametrize("local_name", MOVE_TERMS)
def test_move_term_definition_names_no_data_source(tbox: Graph, local_name: str) -> None:
    definition = str(tbox.value(PKMN[local_name], SKOS.definition)).lower()

    assert not any(source in definition for source in DATA_SOURCES)


def test_has_overworld_effect_is_gone(tbox: Graph) -> None:
    term = PKMN.hasOverworldEffect

    assert (term, None, None) not in tbox
    assert (None, None, term) not in tbox


@pytest.mark.parametrize(("local_name", "classes"), UNION_DOMAINS.items())
def test_union_domains(tbox: Graph, local_name: str, classes: set[str]) -> None:
    assert read_union_domain(tbox, PKMN[local_name]) == classes


def test_is_mirrorable_domain_is_move(tbox: Graph) -> None:
    assert tbox.value(PKMN.isMirrorable, RDFS.domain) == PKMN.Move


def test_version(tbox: Graph) -> None:
    assert str(tbox.value(ONTOLOGY_IRI, OWL.versionInfo)) == VERSION


@pytest.mark.parametrize(("kind", "count"), TERM_COUNTS.items())
def test_term_counts(tbox: Graph, kind: URIRef, count: int) -> None:
    terms = [term for term in tbox.subjects(RDF.type, kind) if str(term).startswith(str(PKMN))]

    assert len(terms) == count
