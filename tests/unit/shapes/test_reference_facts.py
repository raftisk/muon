from collections import Counter

from rdflib import Graph

from tests.unit.shapes.reference_facts import (
    EFFECTIVENESS_PROPERTIES,
    NATURE_GRID,
    NEUTRAL_NATURES,
    TYPE_CHART,
)
from tests.unit.shapes.vocabulary import PKMN

EXPECTED_EDGE_COUNTS = {
    "superEffectiveAgainst": 51,
    "notVeryEffectiveAgainst": 61,
    "noEffectAgainst": 8,
}
NATURE_STATS = ("attack", "defense", "specialAttack", "specialDefense", "speed")
EXPECTED_IMMUNITIES = {
    ("Normal", "Ghost"),
    ("Electric", "Ground"),
    ("Fighting", "Ghost"),
    ("Poison", "Steel"),
    ("Ground", "Flying"),
    ("Psychic", "Dark"),
    ("Ghost", "Normal"),
    ("Dragon", "Fairy"),
}


def read_local_name(node) -> str:
    return str(node).removeprefix(str(PKMN))


def read_edges(graph: Graph, property_name: str) -> set[tuple[str, str]]:
    return {
        (read_local_name(subject), read_local_name(target))
        for subject, target in graph.subject_objects(PKMN[property_name])
    }


def test_type_chart_edge_counts(abox_graph):
    counts = {name: len(read_edges(abox_graph, name)) for name in EFFECTIVENESS_PROPERTIES}
    assert counts == EXPECTED_EDGE_COUNTS
    assert sum(counts.values()) == 120


def test_type_chart_rows_match_spec(abox_graph):
    for attacker, rows in TYPE_CHART.items():
        for property_name, defenders in rows.items():
            actual = {
                defender
                for subject, defender in read_edges(abox_graph, property_name)
                if subject == attacker
            }
            assert actual == set(defenders), f"{attacker} {property_name}"


def test_type_chart_pair_has_one_property(abox_graph):
    pairs = Counter(
        pair for name in EFFECTIVENESS_PROPERTIES for pair in read_edges(abox_graph, name)
    )
    assert {pair for pair, count in pairs.items() if count > 1} == set()


def test_type_chart_immunities(abox_graph):
    assert read_edges(abox_graph, "noEffectAgainst") == EXPECTED_IMMUNITIES


def test_nature_edge_counts(abox_graph):
    assert len(read_edges(abox_graph, "boostsStat")) == 20
    assert len(read_edges(abox_graph, "reducesStat")) == 20


def test_nature_pairs_match_grid(abox_graph):
    boosts = dict(read_edges(abox_graph, "boostsStat"))
    reduces = dict(read_edges(abox_graph, "reducesStat"))
    assert {name: (boosts[name], reduces[name]) for name in boosts} == NATURE_GRID


def test_nature_pairs_cover_every_ordered_pair():
    expected = {(boosted, reduced) for boosted in NATURE_STATS for reduced in NATURE_STATS}
    expected -= {(stat, stat) for stat in NATURE_STATS}
    assert set(NATURE_GRID.values()) == expected
    assert len(NATURE_GRID) == len(expected)


def test_neutral_natures_have_no_edges(abox_graph):
    edged = {name for name, _ in read_edges(abox_graph, "boostsStat")}
    edged |= {name for name, _ in read_edges(abox_graph, "reducesStat")}
    assert edged.isdisjoint(NEUTRAL_NATURES)
    assert len(NEUTRAL_NATURES) == 5


def test_no_nature_touches_hp(abox_graph):
    targets = {t for _, t in read_edges(abox_graph, "boostsStat")}
    targets |= {t for _, t in read_edges(abox_graph, "reducesStat")}
    assert "hp" not in targets
