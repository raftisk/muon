from rdflib import BNode, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from muon.ontology.model import OntologyModel
from muon.validation.projection import ProjectionRules, project_batch, project_literal
from tests.unit.validation.conftest import PKMN, build_edge, build_node, build_test_batch

SPECIES = PKMN + "PokemonSpecies"
TYPE = PKMN + "Type"
EVOLUTION = PKMN + "Evolution"
ITEM = PKMN + "Item"
HAS_TYPE = PKMN + "hasType"
REQUIRES = PKMN + "requires"
NODE_BASE = PKMN + "node/"


def test_node_becomes_type_id_label_and_properties(
    pkmn_model: OntologyModel, pkmn_rules: ProjectionRules
) -> None:
    species = build_node(SPECIES, "pikachu", {PKMN + "pokedexNumber": 25})
    batch = build_test_batch([SPECIES], [species])

    projected = project_batch(batch, pkmn_model, pkmn_rules)

    term = URIRef(NODE_BASE + "PokemonSpecies/pikachu")
    graph = projected.graph
    assert (term, RDF.type, URIRef(SPECIES)) in graph
    assert (term, URIRef(PKMN + "id"), Literal("pikachu")) in graph
    assert (term, RDFS.label, Literal("pikachu")) in graph
    assert (term, URIRef(PKMN + "pokedexNumber"), Literal("25", datatype=XSD.integer)) in graph
    assert projected.nodes_by_term[term] == species.node_ref


def test_float_value_takes_the_declared_decimal_range(
    pkmn_model: OntologyModel, pkmn_rules: ProjectionRules
) -> None:
    species = build_node(SPECIES, "pikachu", {PKMN + "femaleRate": 0.5})

    projected = project_batch(build_test_batch([SPECIES], [species]), pkmn_model, pkmn_rules)

    term = URIRef(NODE_BASE + "PokemonSpecies/pikachu")
    assert (
        term,
        URIRef(PKMN + "femaleRate"),
        Literal("0.5", datatype=XSD.decimal),
    ) in projected.graph


def test_tuple_value_gives_1_literal_per_element() -> None:
    literals = project_literal(("a", "b"), None)

    assert literals == [Literal("a"), Literal("b")]


def test_untyped_values_follow_the_python_type() -> None:
    assert project_literal(True, None) == [Literal("true", datatype=XSD.boolean)]
    assert project_literal(3, None) == [Literal("3", datatype=XSD.integer)]
    assert project_literal(1e-05, None) == [Literal("0.00001", datatype=XSD.decimal)]


def test_slot_edge_maps_to_a_sub_property_without_a_reified_node(
    pkmn_model: OntologyModel, pkmn_rules: ProjectionRules
) -> None:
    species = build_node(SPECIES, "pikachu")
    electric = build_node(TYPE, "electric")
    flying = build_node(TYPE, "flying")
    edges = [
        build_edge(HAS_TYPE, species, electric, {"slot": 1}),
        build_edge(HAS_TYPE, species, flying, {"slot": 2}),
    ]

    projected = project_batch(
        build_test_batch([SPECIES], [species, electric, flying], edges), pkmn_model, pkmn_rules
    )

    graph = projected.graph
    start = URIRef(NODE_BASE + "PokemonSpecies/pikachu")
    assert (start, URIRef(PKMN + "primaryType"), URIRef(NODE_BASE + "Type/electric")) in graph
    assert (start, URIRef(PKMN + "secondaryType"), URIRef(NODE_BASE + "Type/flying")) in graph
    assert (start, URIRef(HAS_TYPE), None) not in graph
    assert list(graph.subjects(RDF.subject, None)) == []


def test_edge_property_gives_a_direct_triple_and_a_reified_node(
    pkmn_model: OntologyModel, pkmn_rules: ProjectionRules
) -> None:
    evolution = build_node(EVOLUTION, "pichu-pikachu")
    item = build_node(ITEM, "oran")
    edge = build_edge(REQUIRES, evolution, item, {"role": "hold"})

    projected = project_batch(
        build_test_batch([EVOLUTION], [evolution, item], [edge]), pkmn_model, pkmn_rules
    )

    graph = projected.graph
    start = URIRef(NODE_BASE + "Evolution/pichu-pikachu")
    end = URIRef(NODE_BASE + "Item/oran")
    assert (start, URIRef(REQUIRES), end) in graph
    reified = next(graph.subjects(RDF.predicate, URIRef(REQUIRES)))
    assert isinstance(reified, BNode)
    assert (reified, RDF.subject, start) in graph
    assert (reified, RDF.object, end) in graph
    assert (reified, URIRef(PKMN + "role"), Literal("hold")) in graph
    assert projected.nodes_by_term[reified] == evolution.node_ref


def test_stub_is_projected_like_a_node(
    pkmn_model: OntologyModel, pkmn_rules: ProjectionRules
) -> None:
    species = build_node(SPECIES, "pikachu")
    electric = build_node(TYPE, "electric")
    batch = build_test_batch(
        [SPECIES], [species, electric], [build_edge(HAS_TYPE, species, electric)]
    )

    projected = project_batch(batch, pkmn_model, pkmn_rules)

    term = URIRef(NODE_BASE + "Type/electric")
    assert (term, RDF.type, URIRef(TYPE)) in projected.graph
    assert projected.nodes_by_term[term] == electric.node_ref
