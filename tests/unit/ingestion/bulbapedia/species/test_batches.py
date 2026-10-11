from collections.abc import Mapping
from pathlib import Path

import pytest

from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.species import batches
from muon.ingestion.bulbapedia.species.batches import (
    CAN_LEARN,
    EVOLUTION,
    EVOLUTION_SCOPE,
    FORM_CHANGE,
    HAS_FORM,
    HELD_ITEM,
    INTRODUCED_IN,
    ITEM,
    LEARNSET_ENTRY,
    MACHINE,
    MOVE,
    PKMN,
    POKEMON_SPECIES,
    REQUIRES,
    SPECIES_SCOPE,
    TYPE,
    VERSION_GROUP,
    AboxIndex,
    AboxMissError,
    BatchBuildError,
    SpeciesIndex,
    build_abox_index,
    build_change_id,
    build_change_name,
    build_entry_id,
    build_entry_name,
    build_evolution_batch,
    build_machine_stub,
    build_species_batch,
    build_species_index,
    resolve_abox,
    resolve_species,
    to_roman,
)
from muon.ingestion.bulbapedia.species.extract import (
    AbilitySlot,
    BaseStats,
    EvolutionFact,
    FormChangeFact,
    FormFacts,
    LearnMethod,
    LearnsetRow,
    NoteKind,
    Requirement,
    RequirementRole,
    SpeciesPageExtract,
    SpeciesRef,
)
from muon.ingestion.runner import (
    MERGE_EDGES_UPSERT,
    MERGE_NODES_UPSERT,
    MERGE_STUBS,
    WriteMode,
    build_statements,
)
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel, load_ontology_model

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
STATS = BaseStats(hp=10, attack=55, defense=25, sp_attack=35, sp_defense=45, speed=95)


@pytest.fixture(scope="module")
def model() -> OntologyModel:
    return load_ontology_model(ONTOLOGY_DIR, "pkmn")


@pytest.fixture(scope="module")
def abox(model: OntologyModel) -> AboxIndex:
    return build_abox_index(read_abox(model, ONTOLOGY_DIR))


def build_facts(form_id: str, name: str, **fields: object) -> FormFacts:
    values: dict[str, object] = {
        "id": form_id,
        "name": name,
        "form_index": 1,
        "is_default": True,
        "region": None,
        "generation": 1,
        "pokedex_number": 50,
        "types": ("Ground",),
        "abilities": (
            AbilitySlot(name="Sand Veil", is_hidden=False),
            AbilitySlot(name="Sand Force", is_hidden=True),
        ),
        "egg_groups": ("Field",),
        "height": 0.2,
        "weight": 0.8,
        "female_rate": 0.5,
        "is_genderless": False,
        "catch_rate": 255,
        "base_exp_yield": 53,
        "growth_rate": "medium_fast",
        "egg_cycles": 20,
        "base_friendship": 70,
        "color": "Brown",
        "shape": "blob",
        "genus": "Mole",
        "is_legendary": False,
        "is_mythical": False,
        "is_starter": False,
        "base_stats": STATS,
        "ev_yield": {"evsp": 1},
        "aliases": (),
        **fields,
    }
    return FormFacts(**values)  # type: ignore[arg-type]


def build_extract(title: str, forms: tuple[FormFacts, ...], **fields: object) -> SpeciesPageExtract:
    values: dict[str, object] = {
        "title": title,
        "page_id": 1,
        "revision_id": 2,
        "forms": forms,
        "candidates": (),
        "learnset": (),
        "form_changes": (),
        "evolutions": (),
        "residue": (),
        "notes": (),
        **fields,
    }
    return SpeciesPageExtract(**values)  # type: ignore[arg-type]


def row(form_id: str, move: str, method: LearnMethod, **fields: object) -> LearnsetRow:
    values: dict[str, object] = {"level": None, "machine": None, **fields}
    return LearnsetRow(form_id=form_id, move=move, method=method, **values)  # type: ignore[arg-type]


DIGLETT = build_facts("diglett", "Diglett")
ALOLAN_DIGLETT = build_facts(
    "diglett-alola",
    "Alolan Diglett",
    form_index=2,
    is_default=False,
    region="Alola",
    generation=None,
    types=("Ground", "Steel"),
    weight=1.0,
    aliases=(),
)
LEARNSET = (
    row("diglett", "Earthquake", LearnMethod.LEVEL, level=40),
    row("diglett", "Earthquake", LearnMethod.MACHINE, machine="TM149"),
    row("diglett", "Mud Shot", LearnMethod.EVOLUTION),
    row("diglett-alola", "Metal Claw", LearnMethod.REMINDER),
)


def level_26() -> EvolutionFact:
    return EvolutionFact(
        source=SpeciesRef(name="Diglett", code=""),
        target=SpeciesRef(name="Dugtrio", code=""),
        trigger="level",
        min_level=26,
        time_of_day=None,
        gender=None,
        requirements=(),
        condition=None,
        section_path="Game data > Evolution data",
    )


def build_diglett_extract() -> SpeciesPageExtract:
    return build_extract(
        "Diglett (Pokémon)",
        (DIGLETT, ALOLAN_DIGLETT),
        learnset=LEARNSET,
        evolutions=(level_26(),),
    )


def build_dugtrio_extract() -> SpeciesPageExtract:
    return build_extract("Dugtrio (Pokémon)", (build_facts("dugtrio", "Dugtrio"),))


def node_keys(batch: GraphBatch) -> set[tuple[str, str]]:
    return {(node.class_iri, node.id) for node in batch.nodes}


def edge_properties(
    batch: GraphBatch, property_iri: str, start: str, end: str
) -> Mapping[str, object]:
    (edge,) = (
        e
        for e in batch.edges
        if e.property_iri == property_iri and e.start.id == start and e.end.id == end
    )
    return edge.properties


def test_every_iri_constant_is_declared_in_the_ontology(model: OntologyModel) -> None:
    constants = {
        name: value
        for name, value in vars(batches).items()
        if name.isupper() and isinstance(value, str) and value.startswith("https://muon.dev/ns/")
    }
    declared = model.classes | model.object_properties | model.datatype_properties

    assert len(constants) > 60
    assert {name for name, value in constants.items() if value not in declared} == {
        "PKMN",
        "CORE",
    }


def test_abox_ids_come_from_the_real_abox(abox: AboxIndex) -> None:
    ground = resolve_abox(abox, TYPE, "Ground")
    water = resolve_abox(abox, PKMN + "EggGroup", "Water 1")
    generation = resolve_abox(abox, PKMN + "Generation", "Generation I")
    version_group = resolve_abox(abox, VERSION_GROUP, "Scarlet-Violet")

    assert (ground.id, ground.name) == ("ground", "Ground")
    assert water.id == "water-1"
    assert generation.id == "generation-i"
    assert version_group.id == "scarlet-violet"
    assert abox.region_generations["Alola"] == "Generation VII"
    assert abox.region_generations["Paldea"] == "Generation IX"


def test_resolve_abox_raises_on_a_miss(abox: AboxIndex) -> None:
    with pytest.raises(AboxMissError, match="Cosmic"):
        resolve_abox(abox, TYPE, "Cosmic")


@pytest.mark.parametrize(
    ("number", "numeral"), [(1, "I"), (4, "IV"), (7, "VII"), (8, "VIII"), (9, "IX"), (10, "X")]
)
def test_to_roman(number: int, numeral: str) -> None:
    assert to_roman(number) == numeral


def test_entry_and_machine_ids_and_names() -> None:
    assert (
        build_entry_id("diglett", "earthquake", LearnMethod.LEVEL, 40)
        == "diglett:earthquake:level:40:scarlet-violet"
    )
    assert (
        build_entry_id("diglett", "earthquake", LearnMethod.MACHINE, None)
        == "diglett:earthquake:machine:scarlet-violet"
    )
    assert (
        build_entry_name("Diglett", "Earthquake", LearnMethod.LEVEL, 40)
        == "Diglett learns Earthquake (level 40, Scarlet-Violet)"
    )
    assert (
        build_entry_name("Diglett", "Earthquake", LearnMethod.EGG, None)
        == "Diglett learns Earthquake (egg, Scarlet-Violet)"
    )
    machine = build_machine_stub("TM149")
    assert (machine.class_iri, machine.id, machine.name) == (
        MACHINE,
        "tm149-scarlet-violet",
        "TM149",
    )
    assert build_change_id("diglett", "dugtrio") == "diglett:dugtrio"
    assert build_change_name("Diglett", "Dugtrio") == "Diglett to Dugtrio"


def test_species_batch_scope_owners_and_stubs(abox: AboxIndex) -> None:
    batch = build_species_batch(build_diglett_extract(), abox)

    assert batch.universe == "pkmn"
    assert batch.scope == SPECIES_SCOPE
    assert {(n.class_iri, n.id) for n in batch.owner_nodes} == {
        (POKEMON_SPECIES, "diglett"),
        (POKEMON_SPECIES, "diglett-alola"),
        (LEARNSET_ENTRY, "diglett:earthquake:level:40:scarlet-violet"),
        (LEARNSET_ENTRY, "diglett:earthquake:machine:scarlet-violet"),
        (LEARNSET_ENTRY, "diglett:mud-shot:evolution:scarlet-violet"),
        (LEARNSET_ENTRY, "diglett-alola:metal-claw:reminder:scarlet-violet"),
    }
    stubs = {(n.class_iri.removeprefix(PKMN), n.id) for n in batch.stub_nodes}
    assert {
        ("Type", "ground"),
        ("Type", "steel"),
        ("Ability", "sand-veil"),
        ("Ability", "sand-force"),
        ("EggGroup", "field"),
        ("Stat", "speed"),
        ("Generation", "generation-i"),
        ("Generation", "generation-vii"),
        ("VersionGroup", "scarlet-violet"),
        ("Move", "earthquake"),
        ("Machine", "tm149-scarlet-violet"),
    } <= stubs


def test_species_records_carry_the_page_values(abox: AboxIndex) -> None:
    batch = build_species_batch(build_diglett_extract(), abox)
    records = {node.id: node for node in batch.owner_nodes}

    diglett = records["diglett"]
    assert diglett.name == "Diglett"
    assert diglett.properties[PKMN + "pokedexNumber"] == 50
    assert diglett.properties[PKMN + "isDefaultForm"] is True
    assert diglett.properties[PKMN + "baseSpeed"] == 95
    assert diglett.properties[PKMN + "growthRate"] == "medium_fast"
    assert diglett.properties[PKMN + "isGenderless"] is False
    assert PKMN + "isLegendary" in diglett.properties
    alola = records["diglett-alola"]
    assert alola.name == "Alolan Diglett"
    assert (alola.properties[PKMN + "weight"], alola.properties[PKMN + "isDefaultForm"]) == (
        1.0,
        False,
    )


def test_a_genderless_species_has_no_female_rate_and_unknown_values_are_left_out(
    abox: AboxIndex,
) -> None:
    magnemite = build_facts(
        "magnemite", "Magnemite", female_rate=None, is_genderless=True, genus=None, color=None
    )

    batch = build_species_batch(build_extract("Magnemite (Pokémon)", (magnemite,)), abox)

    properties = batch.owner_nodes[0].properties
    assert PKMN + "femaleRate" not in properties
    assert properties[PKMN + "isGenderless"] is True
    assert PKMN + "genus" not in properties
    assert PKMN + "color" not in properties


def test_aliases_are_a_list_property(abox: AboxIndex) -> None:
    diglett = build_facts("diglett", "Diglett", aliases=("ディグダ", "Digda"))

    batch = build_species_batch(build_extract("Diglett (Pokémon)", (diglett,)), abox)

    assert batch.owner_nodes[0].properties["https://muon.dev/ns/core#aliases"] == (
        "ディグダ",
        "Digda",
    )


def test_species_edges_and_their_properties(abox: AboxIndex) -> None:
    batch = build_species_batch(build_diglett_extract(), abox)

    assert edge_properties(batch, PKMN + "hasType", "diglett-alola", "steel") == {"slot": 2}
    assert edge_properties(batch, PKMN + "hasType", "diglett", "ground") == {"slot": 1}
    assert edge_properties(batch, PKMN + "hasAbility", "diglett", "sand-force") == {
        "isHidden": True
    }
    assert edge_properties(batch, PKMN + "hasAbility", "diglett", "sand-veil") == {
        "isHidden": False
    }
    assert edge_properties(batch, PKMN + "yieldsEv", "diglett", "speed") == {"amount": 1}
    assert edge_properties(batch, PKMN + "inEggGroup", "diglett", "field") == {}
    assert edge_properties(batch, HAS_FORM, "diglett", "diglett-alola") == {"region": "Alola"}


def test_introduced_in_follows_the_base_generation_and_the_region(abox: AboxIndex) -> None:
    batch = build_species_batch(build_diglett_extract(), abox)

    introduced = {(e.start.id, e.end.id) for e in batch.edges if e.property_iri == INTRODUCED_IN}
    assert introduced == {("diglett", "generation-i"), ("diglett-alola", "generation-vii")}


def test_a_form_that_is_not_regional_gets_a_bare_has_form_and_no_generation(
    abox: AboxIndex,
) -> None:
    zen = build_facts(
        "darmanitan-zen", "Darmanitan (Zen Mode)", form_index=2, is_default=False, generation=None
    )
    extract = build_extract("Darmanitan (Pokémon)", (build_facts("darmanitan", "Darmanitan"), zen))

    batch = build_species_batch(extract, abox)

    assert edge_properties(batch, HAS_FORM, "darmanitan", "darmanitan-zen") == {}
    assert [e.start.id for e in batch.edges if e.property_iri == INTRODUCED_IN] == ["darmanitan"]


def test_learnset_entries_and_can_learn(abox: AboxIndex) -> None:
    batch = build_species_batch(build_diglett_extract(), abox)
    entries = {n.id: n for n in batch.owner_nodes if n.class_iri == LEARNSET_ENTRY}

    level = entries["diglett:earthquake:level:40:scarlet-violet"]
    assert level.name == "Diglett learns Earthquake (level 40, Scarlet-Violet)"
    assert level.properties == {PKMN + "method": "level", PKMN + "level": 40}
    machine = entries["diglett:earthquake:machine:scarlet-violet"]
    assert machine.properties == {PKMN + "method": "machine"}
    reminder = entries["diglett-alola:metal-claw:reminder:scarlet-violet"]
    assert reminder.name == "Alolan Diglett learns Metal Claw (reminder, Scarlet-Violet)"
    links = {
        (e.property_iri.removeprefix(PKMN), e.start.id, e.end.id)
        for e in batch.edges
        if e.start.id == machine.id
    }
    assert links == {
        ("teachesMove", machine.id, "earthquake"),
        ("inVersionGroup", machine.id, "scarlet-violet"),
        ("viaMachine", machine.id, "tm149-scarlet-violet"),
    }
    can_learn = [(e.start.id, e.end.id) for e in batch.edges if e.property_iri == CAN_LEARN]
    assert can_learn == [
        ("diglett", "earthquake"),
        ("diglett", "mud-shot"),
        ("diglett-alola", "metal-claw"),
    ]


def test_repeated_learnset_rows_collapse(abox: AboxIndex) -> None:
    repeated = (*LEARNSET, LEARNSET[0])

    batch = build_species_batch(
        build_extract("Diglett (Pokémon)", (DIGLETT, ALOLAN_DIGLETT), learnset=repeated), abox
    )

    assert len([n for n in batch.owner_nodes if n.class_iri == LEARNSET_ENTRY]) == 4


def test_a_page_without_a_learnset_has_no_version_group_stub(abox: AboxIndex) -> None:
    batch = build_species_batch(build_extract("Diglett (Pokémon)", (DIGLETT,)), abox)

    assert VERSION_GROUP not in {n.class_iri for n in batch.nodes}
    assert CAN_LEARN not in {e.property_iri for e in batch.edges}


def test_mega_form_change_sits_in_the_species_batch(abox: AboxIndex) -> None:
    charizard = build_facts("charizard", "Charizard", types=("Fire", "Flying"))
    mega = build_facts(
        "charizard-mega-x",
        "Mega Charizard X",
        form_index=2,
        is_default=False,
        generation=None,
        types=("Fire", "Dragon"),
    )
    change = FormChangeFact(
        from_id="charizard",
        to_id="charizard-mega-x",
        form_kind="mega",
        is_battle_only=True,
        trigger=None,
        held_item="Charizardite X",
    )
    extract = build_extract("Charizard (Pokémon)", (charizard, mega), form_changes=(change,))

    batch = build_species_batch(extract, abox)

    record = next(n for n in batch.owner_nodes if n.class_iri == FORM_CHANGE)
    assert (record.id, record.name) == (
        "charizard:charizard-mega-x",
        "Charizard to Mega Charizard X",
    )
    assert record.properties == {PKMN + "formKind": "mega", PKMN + "isBattleOnly": True}
    assert edge_properties(batch, REQUIRES, record.id, "charizardite-x") == {"role": "hold"}
    assert any(n.class_iri == HELD_ITEM and n.id == "charizardite-x" for n in batch.stub_nodes)
    assert edge_properties(batch, PKMN + "fromSpecies", record.id, "charizard") == {}
    assert edge_properties(batch, PKMN + "toSpecies", record.id, "charizard-mega-x") == {}
    assert edge_properties(batch, HAS_FORM, "charizard", "charizard-mega-x") == {}


def test_a_primal_form_change_carries_its_trigger(abox: AboxIndex) -> None:
    primal = build_facts("groudon-primal", "Primal Groudon", form_index=2, is_default=False)
    change = FormChangeFact(
        from_id="groudon",
        to_id="groudon-primal",
        form_kind="primal",
        is_battle_only=True,
        trigger="switch_in",
        held_item="Red Orb",
    )
    extract = build_extract(
        "Groudon (Pokémon)", (build_facts("groudon", "Groudon"), primal), form_changes=(change,)
    )

    record = next(
        n for n in build_species_batch(extract, abox).owner_nodes if n.class_iri == FORM_CHANGE
    )

    assert record.properties[PKMN + "trigger"] == "switch_in"


@pytest.mark.parametrize(
    ("field", "facts", "value"),
    [
        ("type1", {"types": ("Cosmic",)}, "Cosmic"),
        ("egggroup", {"egg_groups": ("Imaginary",)}, "Imaginary"),
        ("evsp", {"ev_yield": {"evsp": 1}}, "Speed"),
    ],
)
def test_a_name_the_abox_lacks_fails_the_page(
    abox: AboxIndex, field: str, facts: dict[str, object], value: str
) -> None:
    extract = build_extract("Diglett (Pokémon)", (build_facts("diglett", "Diglett", **facts),))
    broken = AboxIndex(
        ids={k: v for k, v in abox.ids.items() if k[1] != value}, region_generations={}
    )

    with pytest.raises(BatchBuildError) as error:
        build_species_batch(extract, broken)

    assert error.value.page_title == "Diglett (Pokémon)"
    assert error.value.field == field
    assert error.value.value == value
    assert "Diglett (Pokémon)" in str(error.value)


def test_a_region_without_a_generation_fails_the_page(abox: AboxIndex) -> None:
    no_regions = AboxIndex(ids=abox.ids, region_generations={})

    with pytest.raises(BatchBuildError, match="Alola"):
        build_species_batch(build_diglett_extract(), no_regions)


def test_a_missing_version_group_fails_the_page(abox: AboxIndex) -> None:
    broken = AboxIndex(
        ids={k: v for k, v in abox.ids.items() if k[1] != "Scarlet-Violet"},
        region_generations=abox.region_generations,
    )

    with pytest.raises(BatchBuildError, match="Scarlet-Violet"):
        build_species_batch(build_diglett_extract(), broken)


def test_nodes_and_edges_are_sorted(abox: AboxIndex) -> None:
    batch = build_species_batch(build_diglett_extract(), abox)

    node_order = [(n.class_iri, n.id) for n in batch.nodes]
    edge_order = [
        (e.property_iri, e.start.class_iri, e.start.id, e.end.class_iri, e.end.id)
        for e in batch.edges
    ]
    assert node_order == sorted(node_order)
    assert edge_order == sorted(edge_order)


def test_the_species_batch_is_the_same_on_every_build(abox: AboxIndex) -> None:
    first = build_species_batch(build_diglett_extract(), abox)
    second = build_species_batch(build_diglett_extract(), abox)

    assert first == second


def test_the_species_batch_compiles_into_upsert_statements(
    abox: AboxIndex, model: OntologyModel
) -> None:
    batch = build_species_batch(build_diglett_extract(), abox)

    statements = build_statements(batch, WriteMode.UPSERT, LabelMaterializer(model))

    assert [s.query for s in statements] == [MERGE_NODES_UPSERT, MERGE_STUBS, MERGE_EDGES_UPSERT]


def test_species_index_and_reference_resolution() -> None:
    wormadam = build_facts("wormadam", "Wormadam")
    sandy = build_facts("wormadam-sandy", "Wormadam (Sandy Cloak)", form_index=2, is_default=False)
    index = build_species_index(
        [
            build_extract("Wormadam (Pokémon)", (wormadam, sandy)),
            build_extract("Nidoran♀ (Pokémon)", (build_facts("nidoran-f", "Nidoran♀"),)),
        ]
    )

    assert index.names["wormadam-sandy"] == "Wormadam (Sandy Cloak)"
    assert index.ids_by_name == {"Wormadam": "wormadam", "Nidoran♀": "nidoran-f"}
    assert resolve_species(index, SpeciesRef(name="Wormadam", code="")) == ("wormadam", False)
    assert resolve_species(index, SpeciesRef(name="Wormadam", code="Sandy")) == (
        "wormadam-sandy",
        False,
    )
    assert resolve_species(index, SpeciesRef(name="Wormadam", code="Plant")) == ("wormadam", True)
    assert resolve_species(index, SpeciesRef(name="Nidoran♀", code="")) == ("nidoran-f", False)
    assert resolve_species(index, SpeciesRef(name="Missingno", code="")) is None


def build_index() -> SpeciesIndex:
    return build_species_index([build_diglett_extract(), build_dugtrio_extract()])


def test_evolution_batch_owns_the_evolutions_of_the_page(abox: AboxIndex) -> None:
    build = build_evolution_batch(build_diglett_extract(), abox, build_index())

    assert build.batch is not None
    assert build.batch.scope == EVOLUTION_SCOPE
    assert (build.structured, build.total, build.notes) == (1, 1, ())
    (owner,) = build.batch.owner_nodes
    assert (owner.class_iri, owner.id, owner.name) == (
        EVOLUTION,
        "diglett:dugtrio",
        "Diglett to Dugtrio",
    )
    assert owner.properties == {PKMN + "evoTrigger": "level", PKMN + "minLevel": 26}
    assert {(n.class_iri, n.id, n.name) for n in build.batch.stub_nodes} == {
        (POKEMON_SPECIES, "diglett", "Diglett"),
        (POKEMON_SPECIES, "dugtrio", "Dugtrio"),
    }
    assert {
        (e.property_iri.removeprefix(PKMN), e.start.id, e.end.id) for e in build.batch.edges
    } == {
        ("fromSpecies", "diglett:dugtrio", "diglett"),
        ("toSpecies", "diglett:dugtrio", "dugtrio"),
        ("evolvesTo", "diglett", "dugtrio"),
    }


def test_evolution_batch_compiles(abox: AboxIndex, model: OntologyModel) -> None:
    build = build_evolution_batch(build_diglett_extract(), abox, build_index())
    assert build.batch is not None

    statements = build_statements(build.batch, WriteMode.UPSERT, LabelMaterializer(model))

    assert [s.query for s in statements] == [MERGE_NODES_UPSERT, MERGE_STUBS, MERGE_EDGES_UPSERT]


def test_evolutions_of_other_pages_are_cross_checks_and_skipped(abox: AboxIndex) -> None:
    extract = build_extract(
        "Dugtrio (Pokémon)", build_dugtrio_extract().forms, evolutions=(level_26(),)
    )

    build = build_evolution_batch(extract, abox, build_index())

    assert (build.batch, build.notes, build.total) == (None, (), 0)


def test_an_unknown_target_drops_the_evolution_and_creates_no_stub(abox: AboxIndex) -> None:
    unknown = level_26().model_copy(update={"target": SpeciesRef(name="Missingno", code="")})
    extract = build_extract("Diglett (Pokémon)", (DIGLETT,), evolutions=(unknown,))

    build = build_evolution_batch(extract, abox, build_index())

    assert build.batch is None
    assert [n.kind for n in build.notes] == [NoteKind.DROPPED_VALUE]
    assert "Missingno" in build.notes[0].detail


def test_a_cosmetic_form_reference_falls_back_to_the_base_id_and_is_noted(
    abox: AboxIndex,
) -> None:
    sandy = level_26().model_copy(update={"source": SpeciesRef(name="Diglett", code="Sandy")})
    extract = build_extract("Diglett (Pokémon)", (DIGLETT,), evolutions=(sandy,))

    build = build_evolution_batch(extract, abox, build_index())

    assert build.batch is not None
    assert build.batch.owner_nodes[0].id == "diglett:dugtrio"
    assert [n.kind for n in build.notes] == [NoteKind.REFERENCE_FALLBACK]
    assert "Sandy" in build.notes[0].detail


def test_repeated_evolutions_are_kept_once_and_a_conflict_is_noted(abox: AboxIndex) -> None:
    copy = level_26().model_copy(update={"section_path": "Game data > Evolution data > Other"})
    different = level_26().model_copy(update={"min_level": 30})
    same = build_extract("Diglett (Pokémon)", (DIGLETT,), evolutions=(level_26(), copy))
    conflict = build_extract("Diglett (Pokémon)", (DIGLETT,), evolutions=(level_26(), different))

    assert build_evolution_batch(same, abox, build_index()).notes == ()
    conflicting = build_evolution_batch(conflict, abox, build_index())
    assert [n.kind for n in conflicting.notes] == [NoteKind.DROPPED_VALUE]
    assert conflicting.total == 1


def test_an_unstructured_evolution_keeps_its_condition(abox: AboxIndex) -> None:
    tyrogue = level_26().model_copy(
        update={
            "trigger": None,
            "min_level": None,
            "condition": "Level 20 (Attack > Defense)",
        }
    )
    extract = build_extract("Diglett (Pokémon)", (DIGLETT,), evolutions=(tyrogue,))

    build = build_evolution_batch(extract, abox, build_index())

    assert build.batch is not None
    assert build.batch.owner_nodes[0].properties == {
        PKMN + "condition": "Level 20 (Attack > Defense)"
    }
    assert (build.structured, build.total) == (0, 1)


def requirement_evolution(requirement: Requirement) -> EvolutionFact:
    return level_26().model_copy(update={"requirements": (requirement,)})


@pytest.mark.parametrize(
    ("role", "name", "stub_class", "stub_id"),
    [
        (RequirementRole.USE, "Fire Stone", ITEM, "fire-stone"),
        (RequirementRole.HOLD, "Metal Coat", HELD_ITEM, "metal-coat"),
        (RequirementRole.KNOWN_MOVE, "Double Hit", MOVE, "double-hit"),
        (RequirementRole.KNOWN_MOVE_TYPE, "Fairy", TYPE, "fairy"),
        (RequirementRole.PARTY_TYPE, "Dark", TYPE, "dark"),
    ],
)
def test_requirement_stubs_follow_their_role(
    abox: AboxIndex, role: RequirementRole, name: str, stub_class: str, stub_id: str
) -> None:
    evolution = requirement_evolution(Requirement(role=role, name=name, species=None))
    extract = build_extract("Diglett (Pokémon)", (DIGLETT,), evolutions=(evolution,))

    build = build_evolution_batch(extract, abox, build_index())

    assert build.batch is not None
    assert (stub_class, stub_id) in node_keys(build.batch)
    assert edge_properties(build.batch, REQUIRES, "diglett:dugtrio", stub_id) == {
        "role": role.value
    }


def test_a_required_species_resolves_through_the_index(abox: AboxIndex) -> None:
    party = Requirement(
        role=RequirementRole.PARTY_SPECIES,
        name="Dugtrio",
        species=SpeciesRef(name="Dugtrio", code=""),
    )
    extract = build_extract(
        "Diglett (Pokémon)", (DIGLETT,), evolutions=(requirement_evolution(party),)
    )

    build = build_evolution_batch(extract, abox, build_index())

    assert build.batch is not None
    assert edge_properties(build.batch, REQUIRES, "diglett:dugtrio", "dugtrio") == {
        "role": "partySpecies"
    }


def test_an_unknown_required_species_drops_the_evolution(abox: AboxIndex) -> None:
    traded = Requirement(
        role=RequirementRole.TRADED_FOR,
        name="Missingno",
        species=SpeciesRef(name="Missingno", code=""),
    )
    extract = build_extract(
        "Diglett (Pokémon)", (DIGLETT,), evolutions=(requirement_evolution(traded),)
    )

    build = build_evolution_batch(extract, abox, build_index())

    assert build.batch is None
    assert [n.kind for n in build.notes] == [NoteKind.DROPPED_VALUE]


def test_a_requirement_type_the_abox_lacks_fails_the_page(abox: AboxIndex) -> None:
    cosmic = Requirement(role=RequirementRole.PARTY_TYPE, name="Cosmic", species=None)
    extract = build_extract(
        "Diglett (Pokémon)", (DIGLETT,), evolutions=(requirement_evolution(cosmic),)
    )

    with pytest.raises(BatchBuildError, match="Cosmic"):
        build_evolution_batch(extract, abox, build_index())
