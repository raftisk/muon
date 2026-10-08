import pytest
from rdflib import RDF, RDFS, Graph, Literal, Namespace, URIRef

from muon.ingestion.batch import build_node_id
from tests.unit.shapes.abox_completion import (
    ABSENT_GAME_LABELS,
    ALIASES,
    CLASS_COUNTS,
    GAME_GROUPS,
    REMOVED_LOCAL_NAMES,
    VERSION_IDS_SPOT_CHECK,
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


def read_individuals_by_label(graph: Graph, class_name: str, label: str) -> list[URIRef]:
    return [
        individual
        for individual in graph.subjects(RDF.type, expand_class(class_name))
        if Literal(label, lang="en") in set(graph.objects(individual, RDFS.label))
    ]


def test_game_chain_matches_table(abox_graph):
    chain_edges = 0
    for group_label, _, generation, version_labels in GAME_GROUPS:
        groups = read_individuals_by_label(abox_graph, "pkmn:VersionGroup", group_label)
        assert len(groups) == 1, group_label
        assert set(abox_graph.objects(groups[0], MO.partOf)) == {PKMN[generation]}, group_label
        chain_edges += 1
        for version_label in version_labels:
            versions = read_individuals_by_label(abox_graph, "pkmn:Version", version_label)
            assert len(versions) == 1, version_label
            assert set(abox_graph.objects(versions[0], MO.partOf)) == {groups[0]}, version_label
            chain_edges += 1
    assert chain_edges == 64


def test_game_ids(abox_graph):
    for group_label, group_id, _, _ in GAME_GROUPS:
        assert build_node_id(group_label) == group_id
    version_ids = {build_node_id(label) for label in read_labels(abox_graph, "pkmn:Version")}
    assert set(VERSION_IDS_SPOT_CHECK) <= version_ids


def test_absent_games(abox_graph):
    all_labels = {str(label) for label in abox_graph.objects(None, RDFS.label)}
    assert all_labels.isdisjoint(ABSENT_GAME_LABELS)
