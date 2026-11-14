"""Golden tests of the ability run on the pinned pages, the PokeAPI snapshot and the species
parse output. They skip on a clone without the page cache or the snapshot."""

from pathlib import Path

import pytest

from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.abilities.batches import EFFECT_TEXT
from muon.ingestion.bulbapedia.abilities.run import (
    PAGE_TYPE,
    AbilityPageOutput,
    AbilityParseRequest,
    AbilityParseRun,
    AbilityRunInputs,
    parse_ability_pages,
)
from muon.ingestion.bulbapedia.batching import (
    ALIASES,
    INTRODUCED_IN,
    MENTIONED_IN,
    SECTION,
    TEXT,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect
from muon.ingestion.bulbapedia.pages import FETCH_HINT, PageStore
from muon.ingestion.pokeapi.snapshot import SnapshotError, SnapshotStore

pytestmark = pytest.mark.pages

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
NATURAL_CURE = "Natural Cure (Ability)"
TITLES = (NATURAL_CURE, "Slow Start (Ability)", "Mega Sol (Ability)", "Mind's Eye (Ability)")


@pytest.fixture(scope="module")
def run() -> AbilityParseRun:
    store = PageStore()
    manifest = store.read_manifest()
    for title in TITLES:
        pin = manifest.find_pin(PAGE_TYPE, title)
        if pin is None or not store.build_page_path(pin.page_id).exists():
            pytest.skip(
                f"{title!r} is not in the page cache: {FETCH_HINT.format(page_type=PAGE_TYPE)}"
            )
    inputs = AbilityRunInputs(AbilityParseRequest(TITLES), store, SnapshotStore(), ONTOLOGY_DIR)
    try:
        return parse_ability_pages(inputs)
    except SnapshotError as error:
        pytest.skip(str(error))


def find_output(run: AbilityParseRun, title: str) -> AbilityPageOutput:
    (output,) = [output for output in run.outputs if output.page.title == title]
    return output


def list_chunks(batch: GraphBatch, aspect: ChunkAspect) -> list[tuple[str, str]]:
    return [
        (str(node.properties.get(SECTION)), str(node.properties[TEXT]))
        for node in batch.owner_nodes
        if node.id.split(":")[3] == aspect.value
    ]


def test_natural_cure_pass_1(run: AbilityParseRun) -> None:
    batch = find_output(run, NATURAL_CURE).batch

    (ability,) = batch.owner_nodes
    assert ability.id == "natural-cure"
    assert {"Médic Nature", "Innere Kraft"} <= set(ability.properties[ALIASES])  # type: ignore[arg-type]
    assert EFFECT_TEXT in ability.properties
    assert [(e.end.id) for e in batch.edges if e.property_iri == INTRODUCED_IN] == [
        "generation-iii"
    ]


def test_natural_cure_effect_and_trivia_chunks(run: AbilityParseRun) -> None:
    batch = find_output(run, NATURAL_CURE).chunk_batch

    sections = {section for section, _ in list_chunks(batch, ChunkAspect.EFFECT)}
    assert sections == {
        "Effect > In battle > Generation III",
        "Effect > In battle > Generations IV to VII",
        "Effect > In battle > Generation VIII onward",
        "Effect > Outside of battle",
    }
    assert len(list_chunks(batch, ChunkAspect.TRIVIA)) == 2


def test_natural_cure_flavor_chunks(run: AbilityParseRun) -> None:
    extract = find_output(run, NATURAL_CURE).extract

    flavor = [(c.text, c.versions) for c in extract.chunks if c.aspect is ChunkAspect.FLAVOR]
    assert flavor[:3] == [
        (
            "Heals upon switching out.",
            ("Colosseum", "Emerald", "FireRed-LeafGreen", "Ruby-Sapphire", "XD"),
        ),
        ("All status problems are healed upon switching out.", ("Diamond-Pearl", "Platinum")),
        (
            "All status problems heal when it switches out.",
            ("HeartGold-SoulSilver", "Black 2-White 2", "Black-White"),
        ),
    ]
    assert flavor[3][0] == "All status conditions heal when the Pokémon switches out."
    assert flavor[4][1] == ("Scarlet-Violet",)
    assert len(flavor) == 5


def test_natural_cure_mentions_and_holders(run: AbilityParseRun) -> None:
    batch = find_output(run, NATURAL_CURE).chunk_batch

    mentioned = {e.start.id for e in batch.edges if e.property_iri == MENTIONED_IN}
    assert {"trace", "gastro-acid"} <= mentioned
    assert not [d for d in run.report.holder_differences if d.ability_id == "natural-cure"]


def test_slow_start_links_its_legends_arceus_text_to_that_game_alone(run: AbilityParseRun) -> None:
    extract = find_output(run, "Slow Start (Ability)").extract

    versions = [c.versions for c in extract.chunks if c.aspect is ChunkAspect.FLAVOR]
    assert ("Legends: Arceus",) in versions
    assert all("Legends: Arceus" not in labels for labels in versions if len(labels) > 1)


def test_an_ability_with_only_champions_rows_has_no_flavor_chunk(run: AbilityParseRun) -> None:
    extract = find_output(run, "Mega Sol (Ability)").extract

    assert extract.desc_rows == ()
    assert not [c for c in extract.chunks if c.aspect is ChunkAspect.FLAVOR]


def test_a_form_holder_counts_for_its_base_species(run: AbilityParseRun) -> None:
    extract = find_output(run, "Mind's Eye (Ability)").extract

    assert "Ursaluna" in {holder.species_name for holder in extract.holders}
