from rdflib import Graph
from rdflib.namespace import SH

from muon.ontology.model import OntologyModel
from muon.validation.projection import ProjectionRules, project_batch
from muon.validation.shapes import run_shapes, select_shapes
from tests.unit.validation.conftest import MO, PKMN, build_edge, build_node, build_test_batch

SHAPES_NAMESPACE = "https://muon.dev/ns/pkmn-shapes#"
SPECIES = PKMN + "PokemonSpecies"
MOVE = PKMN + "Move"
TYPE = PKMN + "Type"
EVOLUTION = PKMN + "Evolution"
ITEM = PKMN + "Item"
HAS_TYPE = PKMN + "hasType"
REQUIRES = PKMN + "requires"


def list_rule_ids(shapes: Graph) -> set[str]:
    return {str(name) for name in shapes.objects(None, SH.name)}


def test_species_scope_selects_species_rules_and_the_always_rule(
    pkmn_shapes: Graph, pkmn_model: OntologyModel
) -> None:
    selected = select_shapes(pkmn_shapes, frozenset({SPECIES}), pkmn_model, SHAPES_NAMESPACE)

    rule_ids = list_rule_ids(selected)
    assert {"S1", "S2", "S3", "S4", "L2"} <= rule_ids
    assert "M1" not in rule_ids


def test_move_scope_selects_move_rules_and_the_always_rule(
    pkmn_shapes: Graph, pkmn_model: OntologyModel
) -> None:
    selected = select_shapes(pkmn_shapes, frozenset({MOVE}), pkmn_model, SHAPES_NAMESPACE)

    rule_ids = list_rule_ids(selected)
    assert {"M1", "M2", "L2"} <= rule_ids
    assert "S1" not in rule_ids


def test_a_scope_class_selects_the_shapes_of_its_subclasses(
    pkmn_shapes: Graph, pkmn_model: OntologyModel
) -> None:
    selected = select_shapes(pkmn_shapes, frozenset({MO + "Entity"}), pkmn_model, SHAPES_NAMESPACE)

    assert {"S1", "M1"} <= list_rule_ids(selected)


def test_two_primary_types_break_s2_on_the_species(
    pkmn_shapes: Graph,
    pkmn_tbox: Graph,
    pkmn_model: OntologyModel,
    pkmn_rules: ProjectionRules,
) -> None:
    species = build_node(SPECIES, "pikachu", {PKMN + "pokedexNumber": 25})
    electric = build_node(TYPE, "electric")
    water = build_node(TYPE, "water")
    edges = [
        build_edge(HAS_TYPE, species, electric, {"slot": 1}),
        build_edge(HAS_TYPE, species, water, {"slot": 1}),
    ]
    batch = build_test_batch([SPECIES], [species, electric, water], edges)
    selected = select_shapes(pkmn_shapes, batch.scope, pkmn_model, SHAPES_NAMESPACE)

    violations = run_shapes(project_batch(batch, pkmn_model, pkmn_rules), selected, pkmn_tbox)

    s2 = [violation for violation in violations if violation.rule_id == "S2"]
    assert s2
    assert all(violation.node == species.node_ref for violation in s2)
    assert s2[0].property_name == "primaryType"


def test_requires_role_with_the_wrong_target_class_maps_to_the_start_node(
    pkmn_shapes: Graph,
    pkmn_tbox: Graph,
    pkmn_model: OntologyModel,
    pkmn_rules: ProjectionRules,
) -> None:
    evolution = build_node(EVOLUTION, "pichu-pikachu")
    move = build_node(MOVE, "thunderbolt")
    batch = build_test_batch(
        [EVOLUTION], [evolution, move], [build_edge(REQUIRES, evolution, move, {"role": "hold"})]
    )
    selected = select_shapes(pkmn_shapes, batch.scope, pkmn_model, SHAPES_NAMESPACE)

    violations = run_shapes(project_batch(batch, pkmn_model, pkmn_rules), selected, pkmn_tbox)

    r1 = [violation for violation in violations if violation.rule_id == "R1"]
    assert r1
    assert all(violation.node == evolution.node_ref for violation in r1)


def test_valid_requires_edge_raises_no_r1(
    pkmn_shapes: Graph,
    pkmn_tbox: Graph,
    pkmn_model: OntologyModel,
    pkmn_rules: ProjectionRules,
) -> None:
    evolution = build_node(EVOLUTION, "pichu-pikachu")
    item = build_node(ITEM, "oran")
    batch = build_test_batch(
        [EVOLUTION], [evolution, item], [build_edge(REQUIRES, evolution, item, {"role": "hold"})]
    )
    selected = select_shapes(pkmn_shapes, batch.scope, pkmn_model, SHAPES_NAMESPACE)

    violations = run_shapes(project_batch(batch, pkmn_model, pkmn_rules), selected, pkmn_tbox)

    assert "R1" not in {violation.rule_id for violation in violations}
