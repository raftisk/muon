from muon.ingestion.batch import ClaimDirection, EdgeClaim, GraphBatch
from muon.ontology.model import OntologyModel
from muon.validation.checks import check_abox_stubs, check_edge_typing, check_ids, check_iris
from muon.validation.report import RULE_ABOX, RULE_EDGE_TYPING, RULE_ID, RULE_IRI
from tests.unit.validation.conftest import (
    MO,
    PKMN,
    UNIVERSE,
    build_edge,
    build_node,
    build_test_batch,
)

SPECIES = PKMN + "PokemonSpecies"
MOVE = PKMN + "Move"
TYPE = PKMN + "Type"
ABILITY = PKMN + "Ability"
ITEM = PKMN + "Item"
BERRY = PKMN + "Berry"
REGION = MO + "Region"
HAS_TYPE = PKMN + "hasType"
HAS_ABILITY = PKMN + "hasAbility"
FOUND_IN = MO + "foundIn"
DATATYPE_PROPERTY = PKMN + "isLegendary"


def rule_ids(violations: tuple) -> set[str]:
    return {violation.rule_id for violation in violations}


def test_known_iris_pass(pkmn_model: OntologyModel) -> None:
    species = build_node(SPECIES, "pikachu")
    fire = build_node(TYPE, "fire")
    batch = build_test_batch([SPECIES], [species, fire], [build_edge(HAS_TYPE, species, fire)])

    assert check_iris(batch, pkmn_model) == ()


def test_unknown_node_class_fails(pkmn_model: OntologyModel) -> None:
    batch = build_test_batch([SPECIES], [build_node(PKMN + "Gadget", "widget")])

    violations = check_iris(batch, pkmn_model)

    assert rule_ids(violations) == {RULE_IRI}
    assert PKMN + "Gadget" in violations[0].message


def test_unknown_scope_class_fails(pkmn_model: OntologyModel) -> None:
    batch = build_test_batch([PKMN + "Gadget"], [])

    assert rule_ids(check_iris(batch, pkmn_model)) == {RULE_IRI}


def test_unknown_node_property_fails(pkmn_model: OntologyModel) -> None:
    node = build_node(SPECIES, "pikachu", {PKMN + "nonsense": 1})
    batch = build_test_batch([SPECIES], [node])

    violations = check_iris(batch, pkmn_model)

    assert rule_ids(violations) == {RULE_IRI}
    assert violations[0].property_name == "nonsense"


def test_label_property_passes(pkmn_model: OntologyModel) -> None:
    node = build_node(SPECIES, "pikachu", {"http://www.w3.org/2000/01/rdf-schema#label": "x"})

    assert check_iris(build_test_batch([SPECIES], [node]), pkmn_model) == ()


def test_unknown_claim_property_fails(pkmn_model: OntologyModel) -> None:
    species = build_node(SPECIES, "pikachu")
    claim = EdgeClaim(
        anchor=species.node_ref, property_iri=PKMN + "nonsense", direction=ClaimDirection.OUT
    )
    batch = GraphBatch(
        universe=UNIVERSE, scope=frozenset({SPECIES}), nodes=(species,), edges=(), claims=(claim,)
    )

    violations = check_iris(batch, pkmn_model)

    assert rule_ids(violations) == {RULE_IRI}
    assert violations[0].node is None


def test_unknown_claim_other_class_fails(pkmn_model: OntologyModel) -> None:
    species = build_node(SPECIES, "pikachu")
    claim = EdgeClaim(
        anchor=species.node_ref,
        property_iri=HAS_TYPE,
        direction=ClaimDirection.OUT,
        other_class_iri=PKMN + "Gadget",
    )
    batch = GraphBatch(
        universe=UNIVERSE, scope=frozenset({SPECIES}), nodes=(species,), edges=(), claims=(claim,)
    )

    assert rule_ids(check_iris(batch, pkmn_model)) == {RULE_IRI}


def test_datatype_iri_used_as_an_edge_fails(pkmn_model: OntologyModel) -> None:
    species = build_node(SPECIES, "pikachu")
    fire = build_node(TYPE, "fire")
    batch = build_test_batch(
        [SPECIES], [species, fire], [build_edge(DATATYPE_PROPERTY, species, fire)]
    )

    assert rule_ids(check_iris(batch, pkmn_model)) == {RULE_IRI}


def test_ancestor_and_descendant_sharing_an_id_fail(pkmn_model: OntologyModel) -> None:
    batch = build_test_batch([ITEM], [build_node(ITEM, "oran"), build_node(BERRY, "oran")])

    assert rule_ids(check_ids(batch, pkmn_model)) == {RULE_ID}


def test_unrelated_classes_sharing_an_id_pass(pkmn_model: OntologyModel) -> None:
    batch = build_test_batch([MOVE], [build_node(MOVE, "psychic"), build_node(TYPE, "psychic")])

    assert check_ids(batch, pkmn_model) == ()


def test_edge_to_a_range_class_passes(pkmn_model: OntologyModel) -> None:
    species = build_node(SPECIES, "pikachu")
    fire = build_node(TYPE, "fire")
    batch = build_test_batch([SPECIES], [species, fire], [build_edge(HAS_TYPE, species, fire)])

    assert check_edge_typing(batch, pkmn_model) == ()


def test_edge_to_a_wrong_class_fails_the_range(pkmn_model: OntologyModel) -> None:
    species = build_node(SPECIES, "pikachu")
    ability = build_node(ABILITY, "fire")
    batch = build_test_batch(
        [SPECIES], [species, ability], [build_edge(HAS_TYPE, species, ability)]
    )

    violations = check_edge_typing(batch, pkmn_model)

    assert rule_ids(violations) == {RULE_EDGE_TYPING}
    assert "hasType" in violations[0].message
    assert PKMN + "Type" in violations[0].message


def test_edge_from_a_wrong_class_fails_the_domain(pkmn_model: OntologyModel) -> None:
    ability = build_node(ABILITY, "static")
    fire = build_node(TYPE, "fire")
    batch = build_test_batch([ABILITY], [ability, fire], [build_edge(HAS_TYPE, ability, fire)])

    assert rule_ids(check_edge_typing(batch, pkmn_model)) == {RULE_EDGE_TYPING}


def test_union_domain_accepts_each_member(pkmn_model: OntologyModel) -> None:
    move = build_node(MOVE, "ember")
    species = build_node(SPECIES, "pikachu")
    fire = build_node(TYPE, "fire")
    edges = [build_edge(HAS_TYPE, move, fire), build_edge(HAS_TYPE, species, fire)]
    batch = build_test_batch([MOVE, SPECIES], [move, species, fire], edges)

    assert check_edge_typing(batch, pkmn_model) == ()


def test_subclass_of_a_range_class_passes(pkmn_model: OntologyModel) -> None:
    berry = build_node(BERRY, "oran")
    region = build_node(REGION, "kanto")
    batch = build_test_batch([BERRY], [berry, region], [build_edge(FOUND_IN, berry, region)])

    assert check_edge_typing(batch, pkmn_model) == ()


def test_property_without_domain_or_range_is_not_checked(pkmn_model: OntologyModel) -> None:
    assert PKMN + "undeclared" not in pkmn_model.ranges
    species = build_node(SPECIES, "pikachu")
    fire = build_node(TYPE, "fire")
    batch = build_test_batch(
        [SPECIES], [species, fire], [build_edge(PKMN + "undeclared", species, fire)]
    )

    assert check_edge_typing(batch, pkmn_model) == ()


def test_stub_missing_from_the_abox_fails(pkmn_abox: GraphBatch) -> None:
    species = build_node(SPECIES, "pikachu")
    shadow = build_node(TYPE, "shadow")
    batch = build_test_batch([SPECIES], [species, shadow], [build_edge(HAS_TYPE, species, shadow)])

    violations = check_abox_stubs(batch, pkmn_abox)

    assert rule_ids(violations) == {RULE_ABOX}
    assert violations[0].node == shadow.node_ref


def test_stub_in_the_abox_passes(pkmn_abox: GraphBatch) -> None:
    species = build_node(SPECIES, "pikachu")
    ghost = build_node(TYPE, "ghost")
    batch = build_test_batch([SPECIES], [species, ghost], [build_edge(HAS_TYPE, species, ghost)])

    assert check_abox_stubs(batch, pkmn_abox) == ()


def test_stub_of_a_class_outside_the_abox_passes(pkmn_abox: GraphBatch) -> None:
    species = build_node(SPECIES, "pikachu")
    ability = build_node(ABILITY, "static")
    batch = build_test_batch(
        [SPECIES], [species, ability], [build_edge(HAS_ABILITY, species, ability)]
    )

    assert check_abox_stubs(batch, pkmn_abox) == ()
