import pytest
from rdflib import RDF, RDFS, Graph, Namespace, URIRef

from tests.unit.shapes.abox_completion import (
    ALIASES,
    CLASS_COUNTS,
    REMOVED_LOCAL_NAMES,
    VOCABULARY_LABELS,
)
from tests.unit.shapes.vocabulary import PKMN

MO = Namespace("https://muon.dev/ns/core#")
NAMESPACES = {"pkmn": PKMN, "mo": MO}


def expand_class(name: str) -> URIRef:
    prefix, local_name = name.split(":")
    return NAMESPACES[prefix][local_name]


def read_local_name(node) -> str:
    return str(node).removeprefix(str(PKMN))


def read_labels(graph: Graph, class_name: str) -> list[str]:
    return [
        str(label)
        for individual in graph.subjects(RDF.type, expand_class(class_name))
        for label in graph.objects(individual, RDFS.label)
    ]


@pytest.mark.parametrize(("class_name", "count"), CLASS_COUNTS.items())
def test_class_counts(abox_graph, class_name, count):
    assert len(set(abox_graph.subjects(RDF.type, expand_class(class_name)))) == count


@pytest.mark.parametrize(("class_name", "labels"), VOCABULARY_LABELS.items())
def test_vocabulary_labels(abox_graph, class_name, labels):
    actual = read_labels(abox_graph, class_name)
    assert sorted(actual) == sorted(labels)


def test_removed_individuals_are_gone(abox_graph):
    for local_name in REMOVED_LOCAL_NAMES:
        assert (PKMN[local_name], None, None) not in abox_graph, local_name


def test_aliases_match_table(abox_graph):
    actual = {
        read_local_name(subject): tuple(
            sorted(str(alias) for alias in abox_graph.objects(subject, MO.aliases))
        )
        for subject in set(abox_graph.subjects(MO.aliases, None))
    }
    assert actual == ALIASES


def test_drowsy_exists_in_two_classes(abox_graph):
    assert (PKMN.statusDrowsy, RDF.type, PKMN.StatusCondition) in abox_graph
    assert (PKMN.volatileDrowsy, RDF.type, PKMN.VolatileCondition) in abox_graph
    assert "Drowsy" in read_labels(abox_graph, "pkmn:StatusCondition")
    assert "Drowsy" in read_labels(abox_graph, "pkmn:VolatileCondition")
