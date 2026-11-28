from muon.ingestion.batch import GraphBatch
from muon.validation.report import RULE_IRI, RULE_REFERENCE
from muon.validation.validator import Validator, ValidatorInputs
from tests.unit.ingestion.recording_client import RecordingGraphClient
from tests.unit.validation.conftest import PKMN, UNIVERSE, build_edge, build_node

SPECIES = PKMN + "PokemonSpecies"
TYPE = PKMN + "Type"
ABILITY = PKMN + "Ability"
HAS_TYPE = PKMN + "hasType"
HAS_ABILITY = PKMN + "hasAbility"
MOVE_EFFECT = PKMN + "MoveEffect"
GENERATION = PKMN + "Generation"


def build_species_batch(
    can_create_stubs: bool, species_properties: dict | None = None
) -> GraphBatch:
    properties = {
        PKMN + "pokedexNumber": 25,
        PKMN + "femaleRate": 0.5,
        PKMN + "growthRate": "medium_fast",
        PKMN + "isGenderless": False,
        **(species_properties or {}),
    }
    species = build_node(SPECIES, "pikachu", properties)
    electric = build_node(TYPE, "electric")
    flying = build_node(TYPE, "flying")
    static = build_node(ABILITY, "static")
    return GraphBatch(
        universe=UNIVERSE,
        scope=frozenset({SPECIES}),
        nodes=(species, electric, flying, static),
        edges=(
            build_edge(HAS_TYPE, species, electric, {"slot": 1}),
            build_edge(HAS_TYPE, species, flying, {"slot": 2}),
            build_edge(HAS_ABILITY, species, static, {"isHidden": False}),
        ),
        can_create_stubs=can_create_stubs,
    )


async def test_hand_built_species_batch_is_valid(pkmn_validator_inputs: ValidatorInputs) -> None:
    client = RecordingGraphClient()

    report = await Validator(pkmn_validator_inputs, client).validate(build_species_batch(True))

    assert report.violations == ()
    assert report.is_valid
    assert client.reads == []


async def test_iri_violation_returns_only_iri_violations_and_issues_no_query(
    pkmn_validator_inputs: ValidatorInputs,
) -> None:
    client = RecordingGraphClient()
    batch = build_species_batch(False, {PKMN + "nonsense": 1})

    report = await Validator(pkmn_validator_inputs, client).validate(batch)

    assert {violation.rule_id for violation in report.violations} == {RULE_IRI}
    assert client.reads == []


async def test_strict_batch_reads_the_graph_once_and_maps_misses_to_ref(
    pkmn_validator_inputs: ValidatorInputs,
) -> None:
    client = RecordingGraphClient(read_records=[[{"class_iri": ABILITY, "id": "static"}]])

    report = await Validator(pkmn_validator_inputs, client).validate(build_species_batch(False))

    assert len(client.reads) == 1
    assert [(violation.rule_id, violation.value) for violation in report.violations] == [
        (RULE_REFERENCE, "static")
    ]


async def test_shape_violation_is_reported_with_the_other_checks(
    pkmn_validator_inputs: ValidatorInputs,
) -> None:
    batch = build_species_batch(True, {PKMN + "pokedexNumber": 0})

    report = await Validator(pkmn_validator_inputs, RecordingGraphClient()).validate(batch)

    assert {violation.rule_id for violation in report.violations} == {"S1"}


async def test_shapes_are_selected_once_per_scope(pkmn_validator_inputs: ValidatorInputs) -> None:
    validator = Validator(pkmn_validator_inputs, RecordingGraphClient())

    await validator.validate(build_species_batch(True))
    await validator.validate(build_species_batch(True))

    assert list(validator.selected_shapes) == [frozenset({SPECIES})]


async def test_generation_scope_is_checked_on_abox_stubs(
    pkmn_validator_inputs: ValidatorInputs,
) -> None:
    effect = build_node(MOVE_EFFECT, "bite:1", {PKMN + "isOverworld": True})
    first = build_node(GENERATION, "generation-iv")
    last = build_node(GENERATION, "generation-iii")
    batch = GraphBatch(
        universe=UNIVERSE,
        scope=frozenset({MOVE_EFFECT}),
        nodes=(effect, first, last),
        edges=(
            build_edge(PKMN + "introducedIn", effect, first),
            build_edge(PKMN + "lastValidIn", effect, last),
        ),
        can_create_stubs=True,
    )

    report = await Validator(pkmn_validator_inputs, RecordingGraphClient()).validate(batch)

    assert [violation.rule_id for violation in report.violations] == ["G3"]
    assert "generation-iii" in report.violations[0].message
    assert "generation-iv" in report.violations[0].message
