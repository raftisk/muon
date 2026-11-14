import mwparserfromhell
import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import SH

from muon.ingestion.bulbapedia.batching import UNIVERSE
from muon.ingestion.bulbapedia.species.batches import VERSION
from muon.ingestion.bulbapedia.species.encounters import (
    ACQUISITION_LINKS,
    ENCOUNTER_KINDS,
    LOCATION_TEMPLATES,
    METHOD_KINDS,
    METHOD_LINKS,
    METHOD_MOVES,
    NO_METHOD_KIND,
    PAL_PARK_ROW,
    VERSION_ROWS,
    EncounterCounts,
    EncounterInputs,
    EncounterReading,
    EncounterTableError,
    check_version_rows,
    read_encounters,
    read_form_id,
    read_template_link,
    read_wikilink,
    split_parts,
)
from muon.ingestion.bulbapedia.species.forms import ResolvedForm
from muon.ingestion.bulbapedia.species.locations import LocationIndex
from muon.ingestion.bulbapedia.species.page import parse_species_page
from muon.ingestion.seed import read_abox
from muon.ontology.model import load_ontology_model
from tests.unit.ingestion.bulbapedia.species.synthetic_pages import (
    ALPHA,
    ONTOLOGY_DIR,
    build_page,
    species_wikitext,
)

SHAPES_PATH = ONTOLOGY_DIR / "shapes" / "pkmn.shacl.ttl"
PKMN = "https://muon.dev/ns/pkmn#"
VERSIONS = frozenset(
    {"Red", "Blue", "Diamond", "Pearl", "Platinum", "HeartGold", "SoulSilver", "Sword"}
    | {"Shield", "Scarlet", "Violet", "Colosseum", "Legends: Arceus"}
)
PLACES = frozenset({"Cave", "Lake", "Kanto Route 1", "Pal Park", "Field", "Mirelands"})
INDEX = LocationIndex(
    locations=PLACES,
    non_locations=frozenset({"Route", "Jade Orb"}),
    redirects={},
    foreign_titles=frozenset(),
)


def read_table(rows: str, index: LocationIndex = INDEX) -> EncounterReading:
    wikitext = species_wikitext("Alpha", 1, extra=f"===Game locations===\n{rows}\n")
    page = build_page(ALPHA, wikitext)
    extract = parse_species_page(page)
    assert extract is not None
    return read_encounters(page, extract, EncounterInputs(index, VERSIONS))


def entry(area: str, version: str = "Red") -> str:
    return f"{{{{Availability/Entry1|v={version}|area={area}}}}}"


def read_facts(rows: str) -> list[tuple[str, str, str | None, str]]:
    return [(f.location_title, f.version, f.method, f.kind) for f in read_table(rows).facts]


def test_abox_has_every_table_version() -> None:
    abox = read_abox(load_ontology_model(ONTOLOGY_DIR, UNIVERSE), ONTOLOGY_DIR)

    check_version_rows(frozenset(n.name for n in abox.nodes if n.class_iri == VERSION))


def test_table_version_outside_the_abox_stops_the_run() -> None:
    with pytest.raises(EncounterTableError, match="Legends: Z-A"):
        check_version_rows(VERSIONS)


def test_entry2_gives_both_versions() -> None:
    rows = "{{Availability/Entry2|v=Red|v2=Blue|area=[[Cave]]}}"

    assert read_facts(rows) == [
        ("Cave", "Red", None, "wild"),
        ("Cave", "Blue", None, "wild"),
    ]


@pytest.mark.parametrize(("row", "versions"), VERSION_ROWS.items())
def test_dlc_rows_give_their_base_versions(row: str, versions: tuple[str, ...]) -> None:
    reading = read_table(entry("[[Cave]]", row))

    assert [fact.version for fact in reading.facts] == list(versions)


@pytest.mark.parametrize("version", ["Dream World", "Pokéwalker", "Masters EX"])
def test_skipped_versions_give_nothing(version: str) -> None:
    reading = read_table(entry("[[Cave]]", version))

    assert reading.facts == ()
    assert reading.counts.skipped_versions == {version: 1}


def test_japanese_blue_row_is_skipped() -> None:
    reading = read_table("{{Availability/Entry1|v=Blue|ex=(Japan)|area=[[Cave]]}}")

    assert reading.facts == ()
    assert reading.counts.skipped_versions == {"Blue (Japan)": 1}


def test_unknown_version_skips_the_row() -> None:
    reading = read_table(entry("[[Cave]]", "Pokémon Pinball"))

    assert reading.facts == ()
    assert reading.counts.unknown_versions == {"Pokémon Pinball": 1}


def test_none_and_na_rows_are_counted_and_give_nothing() -> None:
    rows = "{{Availability/Entry1/None|v=Red|area=[[Poké Transfer]]}}\n{{Availability/NA|gen=II}}"
    reading = read_table(rows)

    assert reading.facts == ()
    assert reading.counts.rows == {"none": 1, "na": 1}


def test_pal_park_row_gives_its_place_in_its_versions() -> None:
    rows = (
        "{{Availability/Entry1|v=Pal Park|link=Pal Park|"
        "area={{DL|List of Pokémon by Pal Park location|Field}}}}"
    )

    assert read_facts(rows) == [
        ("Pal Park", version, None, PAL_PARK_ROW.kind) for version in PAL_PARK_ROW.versions
    ]


@pytest.mark.parametrize("separator", ["<br>", "<br/>", "<br />"])
def test_split_parts_cuts_at_every_break(separator: str) -> None:
    area = mwparserfromhell.parse(f"[[Cave]]{separator}[[Lake]] ({{{{m|Surf}}}})")

    parts = split_parts(area)

    assert ["".join(str(node) for node in part) for part in parts] == [
        "[[Cave]]",
        "[[Lake]] ({{m|Surf}})",
    ]


@pytest.mark.parametrize(
    ("template", "title"),
    [
        ("{{rt|228|Sinnoh}}", "Sinnoh Route 228"),
        ("{{rtn|5|Alola}}", "Alola Route 5"),
        ("{{rt|7|kalos}}", "kalos Route 7"),
        ("{{safari|Johto}}", "Johto Safari Zone"),
        ("{{ka|Victory Road}}", "Victory Road (Kanto)"),
        ("{{ho|Victory Road}}", "Victory Road (Hoenn)"),
        ("{{si|Iron Ruins}}", "Iron Ruins (Sinnoh)"),
        ("{{kal|Berry fields}}", "Berry fields (Kalos)"),
        ("{{al|Trainers' School}}", "Trainers' School (Alola)"),
        ("{{pal|Pokémon League}}", "Pokémon League (Paldea)"),
        ("{{FB|Kanto|Power Plant}}", "Kanto Power Plant"),
        ("{{OBP|Five Island|town}}", "Five Island (town)"),
        ("{{OBP|South Province|Area Six|Area Six}}", "South Province (Area Six)"),
        ("{{DL|Nacrene City|Nacrene Museum}}", "Nacrene City"),
    ],
)
def test_location_templates_give_their_link_titles(template: str, title: str) -> None:
    node = mwparserfromhell.parse(template).filter_templates()[0]
    link = read_template_link(node)

    assert link is not None
    assert link.title == title


def test_every_location_template_has_a_sample() -> None:
    assert set(LOCATION_TEMPLATES) == {
        "rt",
        "rtn",
        "safari",
        "ka",
        "ho",
        "si",
        "kal",
        "al",
        "pal",
        "fb",
        "obp",
    }


@pytest.mark.parametrize(
    ("wikilink", "title", "anchor"),
    [
        ("[[Bridge Field/Dens|Bridge Field]]", "Bridge Field", ""),
        ("[[Vermilion City#Port|Port]]", "Vermilion City", "Port"),
        ("[[#In events|event]]", "", "In events"),
    ],
)
def test_wikilinks_drop_anchors_and_dens(wikilink: str, title: str, anchor: str) -> None:
    link = read_wikilink(mwparserfromhell.parse(wikilink).filter_wikilinks()[0])

    assert link is not None
    assert (link.title, link.anchor) == (title, anchor)


def test_file_links_are_no_links() -> None:
    node = mwparserfromhell.parse("[[File:Alpha icon.png|20px]]").filter_wikilinks()[0]

    assert read_wikilink(node) is None


@pytest.mark.parametrize(("title", "kind"), ACQUISITION_LINKS.items())
def test_acquisition_parts_give_nothing(title: str, kind: str) -> None:
    reading = read_table(entry(f"[[{title}|Get]] it in [[Cave]]"))

    assert reading.facts == ()
    assert reading.counts.acquisitions == {kind: 1}


@pytest.mark.parametrize(
    "area",
    [
        "[[#In events|Event]] at [[Cave]]",
        "{{DL|Alpha (Pokémon)|In events|event}} at [[Cave]]",
        "{{pkmn2|Event}} at [[Cave]]",
    ],
)
def test_event_parts_give_nothing(area: str) -> None:
    reading = read_table(entry(area))

    assert reading.facts == ()
    assert reading.counts.acquisitions == {"event": 1}


def test_unobtainable_part_gives_nothing() -> None:
    reading = read_table(entry("Unobtainable"))

    assert reading.counts.acquisitions == {"unobtainable": 1}


@pytest.mark.parametrize(("title", "method"), METHOD_LINKS.items())
def test_method_links(title: str, method: str) -> None:
    assert read_facts(entry(f"[[Cave]] ([[{title}]])")) == [
        ("Cave", "Red", method, METHOD_KINDS[method])
    ]


@pytest.mark.parametrize(("move", "method"), METHOD_MOVES.items())
def test_method_moves(move: str, method: str) -> None:
    assert read_facts(entry(f"[[Cave]] ({{{{m|{move}}}}})")) == [("Cave", "Red", method, "wild")]


@pytest.mark.parametrize(
    "marker",
    ["{{pkmn2|Roaming}}", "{{DL|DexNav|hidden Pokémon}}", "[[mass outbreak]]s", "[[fishing]]"],
)
def test_template_and_lowercase_markers(marker: str) -> None:
    facts = read_facts(entry(f"[[Cave]] ({marker})"))

    assert [fact[2] for fact in facts] != [None]


def test_shadow_marker_gives_snag() -> None:
    area = "[[Cave]] {{color2|{{shadow color}}|Shadow Pokémon|(Shadow)}}"

    assert read_facts(entry(area, "Colosseum")) == [("Cave", "Colosseum", "snag", "snag")]


@pytest.mark.parametrize(
    ("display", "method"),
    [
        ("Only one", "static"),
        ("only two", "static"),
        ("One", "static"),
        ("Three", "static"),
        ("Only four", "static"),
        ("one of each form", "static"),
        ("Weekly", "static"),
        ("Received", "gift"),
        ("reward", "gift"),
        ("Buy", "gift"),
    ],
)
def test_event_list_displays(display: str, method: str) -> None:
    link = f"[[List of in-game event Pokémon in Generation II#Alpha|{display}]]"

    assert read_facts(entry(f"[[Cave]] ({link})")) == [("Cave", "Red", method, method)]


def test_event_list_section_link_reads_its_display() -> None:
    link = "{{DL|List of in-game event Pokémon in Pokémon Sword and Shield|Alpha|Only one}}"

    assert read_facts(entry(f"[[Cave]] ({link})", "Sword")) == [
        ("Cave", "Sword", "static", "static")
    ]


def test_unread_event_display_is_counted_and_gives_wild() -> None:
    link = "[[List of in-game event Pokémon in Generation II#Alpha|event]]"
    reading = read_table(entry(f"[[Cave]] {link}"))

    assert [fact.kind for fact in reading.facts] == [NO_METHOD_KIND]
    assert reading.counts.unread_markers == {"event list: event": 1}


def test_part_with_two_methods_gives_one_fact_per_method() -> None:
    area = "[[Lake]] ([[Good Rod]] or [[Super Rod]])"

    assert read_facts(entry(area)) == [
        ("Lake", "Red", "good-rod", "wild"),
        ("Lake", "Red", "super-rod", "wild"),
    ]


def test_method_applies_to_every_place_of_the_part() -> None:
    area = "[[Route]]s {{rtn|1|Kanto}} and [[Lake]] ({{m|Surf}})"

    assert read_facts(entry(area)) == [
        ("Kanto Route 1", "Red", "surf", "wild"),
        ("Lake", "Red", "surf", "wild"),
    ]


def test_unread_group_is_counted() -> None:
    reading = read_table(entry("[[Cave]] (tall grass with 5 points)"))

    assert [fact.method for fact in reading.facts] == [None]
    assert reading.counts.unread_markers == {"(tall grass with 5 points)": 1}


def test_group_with_a_marker_is_not_unread() -> None:
    reading = read_table(entry("[[Cave]] (also ([[mass outbreak]]s){{Sup/t|D}})"))

    assert reading.counts.unread_markers == {}


def test_bold_prefix_is_dropped_when_the_part_names_a_location() -> None:
    area = "'''[[Mirelands]]:''' [[Lake]]{{Sup/t|N}}, near [[Cave]]"

    assert [f[0] for f in read_facts(entry(area, "Legends: Arceus"))] == ["Lake", "Cave"]


def test_bold_prefix_is_the_location_when_the_part_names_no_other() -> None:
    area = "'''[[Mirelands]]:''' [[Golden Lowlands]], [[Unknown Bog]]"

    assert [f[0] for f in read_facts(entry(area, "Legends: Arceus"))] == ["Mirelands"]


def test_candidates_are_counted_by_verdict() -> None:
    reading = read_table(entry("[[Cave]], [[Jade Orb]], [[Somewhere]]"))

    assert reading.counts.locations == {"Cave": 1}
    assert reading.counts.not_a_location == {"Jade Orb": 1}
    assert reading.counts.unknown_location == {"Somewhere": 1}


def test_repeated_facts_collapse() -> None:
    area = "[[Cave]]<br>[[Cave]] (dark grass)"

    assert read_facts(entry(area)) == [("Cave", "Red", None, "wild")]


def test_page_without_game_locations_gives_nothing() -> None:
    page = build_page(ALPHA, species_wikitext("Alpha", 1))
    extract = parse_species_page(page)
    assert extract is not None

    reading = read_encounters(page, extract, EncounterInputs(INDEX, VERSIONS))

    assert reading.facts == ()


FORMS = (
    ResolvedForm(id="alpha", name="Alpha", index=1, infobox_name="", region=None, is_default=True),
    ResolvedForm(
        id="alpha-alola",
        name="Alolan Alpha",
        index=2,
        infobox_name="Alolan Form",
        region="Alola",
        is_default=False,
    ),
)


@pytest.mark.parametrize(
    ("caption", "form_id", "unmatched"),
    [
        ("", "alpha", {}),
        (" <small>('''Alolan Form''')</small>", "alpha-alola", {}),
        (" <small>('''Kantonian Form''')</small>", "alpha", {"Kantonian Form": 1}),
    ],
)
def test_form_caption(caption: str, form_id: str, unmatched: dict[str, int]) -> None:
    part = mwparserfromhell.parse(f"[[Cave]]{caption}").nodes
    counts = EncounterCounts()

    assert read_form_id(part, FORMS, counts) == form_id
    assert counts.unmatched_form_labels == unmatched


def read_shape_values(shape: str, path: str) -> set[str]:
    shapes = Graph().parse(SHAPES_PATH, format="turtle")
    node = URIRef(f"https://muon.dev/ns/pkmn-shapes#{shape}")
    for prop in shapes.objects(node, SH.property):
        if shapes.value(prop, SH.path) == URIRef(PKMN + path):
            return {str(value) for value in shapes.items(shapes.value(prop, SH["in"]))}
    raise AssertionError(f"{shape} has no property shape on {path}")


def test_method_kinds_match_shape_c3() -> None:
    assert read_shape_values("C3_EncounterMethod", "method") == set(METHOD_KINDS)


def test_kinds_match_shape_c1() -> None:
    assert read_shape_values("C1_Encounter", "encounterKind") == set(ENCOUNTER_KINDS)
    assert set(METHOD_KINDS.values()) <= set(ENCOUNTER_KINDS)
