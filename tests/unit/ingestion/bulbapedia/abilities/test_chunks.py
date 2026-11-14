from muon.ingestion.batch import EdgeRecord, GraphBatch, NodeRecord
from muon.ingestion.bulbapedia.abilities.chunks import (
    FLAVOR_SECTION_PATH,
    PART_OF,
    UnknownSupCode,
    build_ability_version_groups,
    read_ability_aspect,
    read_ability_chunks,
)
from muon.ingestion.bulbapedia.abilities.page import DescRow
from muon.ingestion.bulbapedia.batching import GENERATION, UNIVERSE, VERSION_GROUP
from muon.ingestion.bulbapedia.chunking import ChunkAspect
from muon.ingestion.bulbapedia.wikitext import split_sections

ABILITY_ID = "test-guard"
VERSION_GROUPS = {
    "Generation III": ("Ruby-Sapphire", "Emerald", "FireRed-LeafGreen", "Colosseum", "XD"),
    "Generation IV": ("Diamond-Pearl", "Platinum", "HeartGold-SoulSilver"),
    "Generation V": ("Black-White", "Black 2-White 2"),
    "Generation VIII": ("Sword-Shield", "Brilliant Diamond and Shining Pearl"),
}


def read_flavor(*rows: DescRow) -> list[tuple[str, tuple[str, ...]]]:
    reading = read_ability_chunks((), rows, ABILITY_ID, VERSION_GROUPS)
    return [(chunk.text, chunk.versions) for chunk in reading.chunks]


def test_effect_subsections_give_effect_chunks_except_champions_and_contests() -> None:
    assert read_ability_aspect(("Effect", "In battle", "Generation III")) is ChunkAspect.EFFECT
    assert read_ability_aspect(("Effect", "Outside of battle")) is ChunkAspect.EFFECT
    assert read_ability_aspect(("Effect", "Pokémon Champions")) is None
    assert read_ability_aspect(("Effect", "Contest")) is None
    assert read_ability_aspect(("Trivia",)) is ChunkAspect.TRIVIA
    assert read_ability_aspect(("In the anime",)) is None


def test_prose_chunks_come_before_flavor_chunks() -> None:
    sections = split_sections("Lead.\n==Effect==\nIt guards {{a|Trace}}.\n==Trivia==\n* A fact.\n")
    rows = (DescRow(generation_label="Generation V", wikitext="Guards."),)

    reading = read_ability_chunks(sections, rows, ABILITY_ID, VERSION_GROUPS)

    assert [chunk.aspect for chunk in reading.chunks] == [
        ChunkAspect.EFFECT,
        ChunkAspect.TRIVIA,
        ChunkAspect.FLAVOR,
    ]
    assert reading.chunks[2].section_path == FLAVOR_SECTION_PATH


def test_an_unmarked_row_is_shown_in_its_whole_generation() -> None:
    assert read_flavor(DescRow(generation_label="Generation III", wikitext="Heals.")) == [
        ("Heals.", VERSION_GROUPS["Generation III"])
    ]


def test_marked_parts_name_exactly_their_version_groups_and_merge_by_text() -> None:
    rows = (
        DescRow(
            generation_label="Generation IV",
            wikitext="Heals well.{{sup/4|DPP}}<br/>Heals.{{sup/4|HGSS}}",
        ),
        DescRow(generation_label="Generation V", wikitext="Heals."),
    )

    assert read_flavor(*rows) == [
        ("Heals well.", ("Diamond-Pearl", "Platinum")),
        ("Heals.", ("HeartGold-SoulSilver", "Black-White", "Black 2-White 2")),
    ]


def test_a_battle_revolution_part_gives_nothing() -> None:
    row = DescRow(
        generation_label="Generation IV",
        wikitext="Raises encounters.{{sup/4|DPPtHGSS}}<br>No effect in Pokétopia.{{sup/ss|PBR}}",
    )

    assert read_flavor(row) == [
        ("Raises encounters.", ("Diamond-Pearl", "Platinum", "HeartGold-SoulSilver"))
    ]


def test_a_legends_arceus_mark_names_that_game_alone() -> None:
    row = DescRow(
        generation_label="Generation VIII",
        wikitext="Halves stats.{{sup/8|SwShBDSP}}<br>Halves stats for a while.{{sup/8|LA}}",
    )

    assert read_flavor(row) == [
        ("Halves stats.", ("Sword-Shield", "Brilliant Diamond and Shining Pearl")),
        ("Halves stats for a while.", ("Legends: Arceus",)),
    ]


def test_an_unknown_code_keeps_the_generation_and_is_reported() -> None:
    row = DescRow(generation_label="Generation V", wikitext="Guards.{{sup/5|BW3}}")

    reading = read_ability_chunks((), (row,), ABILITY_ID, VERSION_GROUPS)

    assert reading.chunks[0].versions == VERSION_GROUPS["Generation V"]
    assert reading.unknown_codes == (UnknownSupCode("Generation V", "BW3"),)


def test_version_groups_per_generation_leave_out_the_games_without_abilities() -> None:
    generation = NodeRecord(class_iri=GENERATION, id="generation-viii", name="Generation VIII")
    groups = [
        NodeRecord(class_iri=VERSION_GROUP, id=name.lower(), name=name)
        for name in ("Legends: Arceus", "Sword-Shield")
    ]
    seed = GraphBatch(
        universe=UNIVERSE,
        scope=frozenset({GENERATION, VERSION_GROUP}),
        nodes=(generation, *groups),
        edges=tuple(
            EdgeRecord(property_iri=PART_OF, start=group.node_ref, end=generation.node_ref)
            for group in groups
        ),
    )

    assert build_ability_version_groups(seed) == {"Generation VIII": ("Sword-Shield",)}
