"""Golden tests of the item run on the pinned reference pages. They skip on a clone without
the page cache."""

from pathlib import Path

import pytest

from muon.ingestion.batch import NodeRecord
from muon.ingestion.bulbapedia.batching import (
    ASPECT,
    BERRY,
    HAS_FLAVOR,
    HELD_ITEM,
    INTRODUCED_IN,
    ITEM,
    SECTION,
    TEXT,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect
from muon.ingestion.bulbapedia.items.batches import (
    BUY_PRICE,
    CATEGORY,
    FLING_POWER,
    SELL_PRICE,
)
from muon.ingestion.bulbapedia.items.run import (
    PAGE_TYPE,
    ItemPageOutput,
    ItemParseRequest,
    ItemParseRun,
    ItemRunInputs,
    parse_item_pages,
)
from muon.ingestion.bulbapedia.pages import FETCH_HINT, PageStore

pytestmark = pytest.mark.pages

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
TITLES = ("Black Sludge", "Lava Cookie", "Metagrossite", "Rare Candy", "Oran Berry")


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> ItemParseRun:
    store = PageStore()
    manifest = store.read_manifest()
    for title in TITLES:
        pin = manifest.find_pin(PAGE_TYPE, title)
        if pin is None or not store.build_page_path(pin.page_id).exists():
            pytest.skip(
                f"{title!r} is not in the page cache: {FETCH_HINT.format(page_type=PAGE_TYPE)}"
            )
    output_root = tmp_path_factory.mktemp("parsed")
    return parse_item_pages(
        ItemRunInputs(ItemParseRequest(TITLES), store, ONTOLOGY_DIR, output_root)
    )


def find_output(run: ItemParseRun, title: str) -> ItemPageOutput:
    (output,) = [output for output in run.outputs if output.page.title == title]
    return output


def find_owner(output: ItemPageOutput) -> NodeRecord:
    (owner,) = output.batch.owner_nodes
    return owner


def list_flavor_groups(output: ItemPageOutput) -> list[tuple[str, ...]]:
    chunks = [n for n in output.chunk_batch.owner_nodes if n.properties.get(ASPECT) == "flavor"]
    groups = {
        chunk.id: tuple(
            e.end.id for e in output.chunk_batch.edges if e.start.id == chunk.id and e.end.id
        )
        for chunk in chunks
    }
    return sorted(group for group in groups.values() if group)


def count_chunks(output: ItemPageOutput, aspect: ChunkAspect) -> int:
    return sum(n.properties.get(ASPECT) == aspect.value for n in output.chunk_batch.owner_nodes)


def test_all_5_pages_parse(run: ItemParseRun) -> None:
    assert run.report.pages_parsed == len(TITLES)
    assert run.report.pages_failed == 0


def test_black_sludge(run: ItemParseRun) -> None:
    output = find_output(run, "Black Sludge")
    owner = find_owner(output)

    assert owner.class_iri == HELD_ITEM
    assert owner.id == "black-sludge"
    assert owner.properties[CATEGORY] == "Other Items"
    assert (owner.properties[BUY_PRICE], owner.properties[SELL_PRICE]) == (10000, 2500)
    assert owner.properties[FLING_POWER] == 30
    assert [e.end.id for e in output.batch.edges if e.property_iri == INTRODUCED_IN] == [
        "generation-iv"
    ]
    assert count_chunks(output, ChunkAspect.FLAVOR) == 4
    assert len(list_flavor_groups(output)) == 4


def test_lava_cookie(run: ItemParseRun) -> None:
    owner = find_owner(find_output(run, "Lava Cookie"))

    assert owner.class_iri == ITEM
    assert owner.properties[CATEGORY] == "Medicine"
    assert BUY_PRICE not in owner.properties
    assert owner.properties[FLING_POWER] == 30


def test_lava_cookie_flavor_has_no_colosseum_or_xd_group(run: ItemParseRun) -> None:
    groups = {g for group in list_flavor_groups(find_output(run, "Lava Cookie")) for g in group}

    assert {"colosseum", "xd"}.isdisjoint(groups)
    assert {"ruby-sapphire", "emerald"} <= groups


def test_metagrossite(run: ItemParseRun) -> None:
    owner = find_owner(find_output(run, "Metagrossite"))

    assert owner.class_iri == HELD_ITEM
    assert owner.properties[CATEGORY] == "Mega Stones"
    assert BUY_PRICE not in owner.properties
    assert owner.properties[FLING_POWER] == 80


def test_rare_candy(run: ItemParseRun) -> None:
    output = find_output(run, "Rare Candy")
    owner = find_owner(output)

    assert owner.class_iri == ITEM
    assert owner.properties[CATEGORY] == "Other Items"
    assert BUY_PRICE not in owner.properties
    assert owner.properties[SELL_PRICE] == 2500
    assert owner.properties[FLING_POWER] == 30
    assert count_chunks(output, ChunkAspect.TRIVIA) == 3


def test_oran_berry(run: ItemParseRun) -> None:
    output = find_output(run, "Oran Berry")
    owner = find_owner(output)

    assert owner.class_iri == BERRY
    assert owner.properties[CATEGORY] == "Berries"
    assert owner.properties[SELL_PRICE] == 20
    assert owner.properties[FLING_POWER] == 10
    flavors = {e.end.id: e.properties for e in output.batch.edges if e.property_iri == HAS_FLAVOR}
    assert flavors == {name: {"potency": 10} for name in ("spicy", "dry", "bitter", "sour")}


def test_oran_berry_tag_information_is_a_flavor_section(run: ItemParseRun) -> None:
    output = find_output(run, "Oran Berry")
    sections = {
        str(node.properties[SECTION])
        for node in output.chunk_batch.owner_nodes
        if node.properties.get(ASPECT) == "flavor"
    }

    assert any(section.startswith("Description") for section in sections)
    assert not any(TEXT in section for section in sections)
