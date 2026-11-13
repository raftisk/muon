import pytest

from muon.ingestion.bulbapedia.chunking import MIN_PARAGRAPH_LENGTH, ChunkAspect, MentionRef
from muon.ingestion.bulbapedia.species.chunks import (
    match_marker,
    merge_dex_rows,
    pick_biology_subject,
    read_aspect,
    read_chunks,
    read_dex_rows,
)
from muon.ingestion.bulbapedia.species.extract import NoteKind
from muon.ingestion.bulbapedia.species.forms import ResolvedForm
from muon.ingestion.bulbapedia.wikitext import split_sections

LONG = "x" * MIN_PARAGRAPH_LENGTH

DIGLETT = (
    ResolvedForm("diglett", "Diglett", 1, "", None, True),
    ResolvedForm("diglett-alola", "Alolan Diglett", 2, "Alolan Form", "Alola", False),
)
ROTOM = (
    ResolvedForm("rotom", "Rotom", 1, "", None, True),
    ResolvedForm("rotom-heat", "Heat Rotom", 2, "Heat Rotom", None, False),
    ResolvedForm("rotom-wash", "Wash Rotom", 3, "Wash Rotom", None, False),
)
DEOXYS = (
    ResolvedForm("deoxys", "Deoxys", 1, "Normal Forme", None, True),
    ResolvedForm("deoxys-attack", "Deoxys (Attack Forme)", 2, "Attack Forme", None, False),
)
TAUROS = (
    ResolvedForm("tauros", "Tauros", 1, "", None, True),
    ResolvedForm(
        "tauros-paldea-combat",
        "Paldean Tauros (Combat Breed)",
        2,
        "Paldean Form (Combat Breed)",
        "Paldea",
        False,
    ),
    ResolvedForm(
        "tauros-paldea-blaze",
        "Paldean Tauros (Blaze Breed)",
        3,
        "Paldean Form (Blaze Breed)",
        "Paldea",
        False,
    ),
)
ORICORIO = (
    ResolvedForm("oricorio", "Oricorio", 1, "Baile Style", None, True),
    ResolvedForm("oricorio-pau", "Oricorio (Pa’u Style)", 2, "Pa’u Style", None, False),
)
MINIOR = (
    ResolvedForm("minior", "Minior", 1, "Meteor Form", None, True),
    ResolvedForm("minior-red", "Minior (Red Core)", 2, "Red Core", None, False),
    ResolvedForm("minior-blue", "Minior (Blue Core)", 3, "Blue Core", None, False),
)


@pytest.mark.parametrize(
    ("path", "aspect"),
    [
        (("Biology",), ChunkAspect.BIOLOGY),
        (("Biology", "Forms"), ChunkAspect.BIOLOGY),
        (("Biology", "Forms", "Heat Rotom"), ChunkAspect.BIOLOGY),
        (("Biology", "Evolution"), None),
        (("Biology", "Evolution", "Galarian Corsola"), None),
        (("Trivia",), ChunkAspect.TRIVIA),
        (("Trivia", "Design quirks"), ChunkAspect.TRIVIA),
        (("Trivia", "Origin"), ChunkAspect.ORIGIN),
        (("Trivia", "Origin", "Name origin"), ChunkAspect.ORIGIN),
        (("Trivia", "Concept and development"), ChunkAspect.ORIGIN),
        (("Trivia", "Concept and development", "Name origin"), ChunkAspect.ORIGIN),
        (("Trivia", "Merchandise"), None),
        (("Trivia", "Merchandise", "Name origin"), None),
        (("Game data", "Pokédex entries"), ChunkAspect.POKEDEX),
        (("Game data", "Pokédex entries", "Generation I"), ChunkAspect.POKEDEX),
        (("Game data", "Stats"), None),
        (("In other languages",), None),
        ((), None),
    ],
)
def test_read_aspect(path: tuple[str, ...], aspect: ChunkAspect | None) -> None:
    assert read_aspect(path) is aspect


def test_read_aspect_compares_titles_case_insensitively() -> None:
    assert read_aspect(("BIOLOGY", "evolution")) is None
    assert read_aspect(("trivia", "ORIGIN")) is ChunkAspect.ORIGIN
    assert read_aspect(("Game Data", "POKÉDEX ENTRIES")) is ChunkAspect.POKEDEX


@pytest.mark.parametrize(
    ("marker", "forms", "expected"),
    [
        ("Heat Rotom", ROTOM, "rotom-heat"),
        ("heat rotom", ROTOM, "rotom-heat"),
        ("Rotom", ROTOM, "rotom"),
        ("Attack Forme", DEOXYS, "deoxys-attack"),
        ("Normal Forme", DEOXYS, "deoxys"),
        ("Alolan Form", DIGLETT, "diglett-alola"),
        ("Alolan Diglett", DIGLETT, "diglett-alola"),
        ("Pa'u Style", ORICORIO, "oricorio-pau"),
        ("Pa’u Style", ORICORIO, "oricorio-pau"),
        ("Combat Breed", TAUROS, "tauros-paldea-combat"),
        ("Paldean Tauros", TAUROS, None),
        ("Core", MINIOR, None),
        ("Gigantamax", DIGLETT, None),
        ("Combat", TAUROS, "tauros-paldea-combat"),
        ("Brea", TAUROS, None),
        ("", ROTOM, None),
    ],
)
def test_match_marker(marker: str, forms: tuple[ResolvedForm, ...], expected: str | None) -> None:
    assert match_marker(marker, forms) == expected


def test_match_marker_prefers_an_equal_name_over_a_containing_one() -> None:
    forms = (
        ResolvedForm("a", "Alpha", 1, "", None, True),
        ResolvedForm("a-red", "Alpha (Red)", 2, "Red", None, False),
        ResolvedForm("a-red-core", "Alpha (Red Core)", 3, "Red Core", None, False),
    )

    assert match_marker("Red", forms) == "a-red"


def test_match_marker_gives_nothing_for_2_equal_names() -> None:
    forms = (
        ResolvedForm("a", "Alpha", 1, "", None, True),
        ResolvedForm("a-x", "Alpha (X)", 2, "Same", None, False),
        ResolvedForm("a-y", "Alpha (Y)", 3, "Same", None, False),
    )

    assert match_marker("Same", forms) is None


def test_pick_biology_subject_describes_the_base_form_without_markers() -> None:
    assert pick_biology_subject(("Biology",), ROTOM) == ("rotom", None)
    assert pick_biology_subject(("Biology", "Forms"), ROTOM) == ("rotom", None)


def test_pick_biology_subject_uses_the_title_below_forms() -> None:
    subject, note = pick_biology_subject(("Biology", "Forms", "Heat Rotom"), ROTOM)

    assert (subject, note) == ("rotom-heat", None)


def test_pick_biology_subject_tries_the_deepest_title_first() -> None:
    path = ("Biology", "Forms", "Wash Rotom", "Heat Rotom")

    assert pick_biology_subject(path, ROTOM)[0] == "rotom-heat"


def test_pick_biology_subject_falls_back_to_a_shallower_title() -> None:
    path = ("Biology", "Wash Rotom", "Appearance")

    assert pick_biology_subject(path, ROTOM) == ("rotom-wash", None)


def test_pick_biology_subject_notes_a_label_that_matches_no_form() -> None:
    subject, note = pick_biology_subject(("Biology", "Forms", "Gigantamax"), DIGLETT)

    assert subject == "diglett"
    assert note is not None
    assert note.kind is NoteKind.UNMATCHED_FORM_LABEL
    assert note.section_path == "Biology > Forms > Gigantamax"
    assert note.detail == "Gigantamax matches no form node"


def test_pick_biology_subject_notes_a_label_that_matches_2_forms() -> None:
    subject, note = pick_biology_subject(("Biology", "Forms", "Core"), MINIOR)

    assert subject == "minior"
    assert note is not None
    assert note.detail == "Core matches no form node"


PAGE = f"""lead text that gives no chunk
==Game data==
===Stats===
stats text
==Biology==
{LONG} first biology paragraph with {{{{p|Onix|an Onix}}}}.

Diglett has a regional form: Alolan Diglett.

{LONG} the second long paragraph.
===Forms===
{LONG} forms text for the base form.
====Alolan Form====
{LONG} alolan text mentioning the {{{{type|Steel}}}} type.
====Gigantamax====
{LONG} gigantamax text.
===Evolution===
{LONG} evolution text is out.
==Trivia==
* trivia one {{{{m|Dig}}}}
** trivia one sub
* trivia two
===Origin===
{LONG} origin text with a {{{{roundy|x}}}} dropped template.
===Merchandise===
* merchandise is out
"""


def read_page_chunks() -> tuple[list[tuple[str, str, str, str]], list[str], dict[str, int]]:
    reading = read_chunks(split_sections(PAGE), DIGLETT)
    facts = [
        (chunk.subject_id, chunk.aspect.value, chunk.section_path, chunk.text[:40])
        for chunk in reading.chunks
    ]
    return facts, [note.detail for note in reading.notes], dict(reading.dropped_templates)


def test_read_chunks_gives_one_chunk_per_block_with_aspect_section_and_subject() -> None:
    facts, _, _ = read_page_chunks()

    assert [fact[:3] for fact in facts] == [
        ("diglett", "biology", "Biology"),
        ("diglett", "biology", "Biology"),
        ("diglett", "biology", "Biology > Forms"),
        ("diglett-alola", "biology", "Biology > Forms > Alolan Form"),
        ("diglett", "biology", "Biology > Forms > Gigantamax"),
        ("diglett", "trivia", "Trivia"),
        ("diglett", "trivia", "Trivia"),
        ("diglett", "origin", "Trivia > Origin"),
    ]


def test_read_chunks_merges_the_short_paragraph_into_the_next_one() -> None:
    reading = read_chunks(split_sections(PAGE), DIGLETT)

    assert reading.chunks[1].text.startswith("Diglett has a regional form: Alolan Diglett. xxx")


def test_read_chunks_keeps_a_bullet_item_with_its_sub_bullets() -> None:
    reading = read_chunks(split_sections(PAGE), DIGLETT)
    trivia = [chunk for chunk in reading.chunks if chunk.aspect is ChunkAspect.TRIVIA]

    assert [chunk.text for chunk in trivia] == ["trivia one Dig trivia one sub", "trivia two"]
    assert trivia[0].mentions == (MentionRef(template="m", target="Dig"),)


def test_read_chunks_collects_mentions_of_a_merged_paragraph() -> None:
    reading = read_chunks(split_sections(PAGE), DIGLETT)

    assert reading.chunks[0].mentions == (MentionRef(template="p", target="Onix"),)
    assert reading.chunks[3].mentions == (MentionRef(template="type", target="Steel"),)


def test_read_chunks_notes_unmatched_form_labels_and_counts_dropped_templates() -> None:
    _, notes, dropped = read_page_chunks()

    assert notes == ["Gigantamax matches no form node"]
    assert dropped == {"roundy": 1}


def test_read_chunks_skips_the_lead_stats_evolution_and_merchandise() -> None:
    reading = read_chunks(split_sections(PAGE), DIGLETT)
    joined = " ".join(chunk.text for chunk in reading.chunks)

    assert "lead text" not in joined
    assert "stats text" not in joined
    assert "evolution text" not in joined
    assert "merchandise" not in joined


def test_read_chunks_gives_nothing_for_a_page_without_chunk_sections() -> None:
    reading = read_chunks(split_sections("lead\n==Game data==\n===Stats===\nrows"), DIGLETT)

    assert reading.chunks == ()
    assert reading.notes == ()
    assert reading.dropped_templates == {}


POKEDEX_PATH = "Game data > Pokédex entries"
SKIN = "Its skin is very thin. If it is exposed to light, its blood heats up."


def read_dex(
    wikitext: str, forms: tuple[ResolvedForm, ...] = DIGLETT
) -> tuple[list[tuple[str, str, str, tuple[str, ...]]], list[str]]:
    section = split_sections(f"==Game data==\n===Pokédex entries===\n{wikitext}")[-1]
    rows, notes = read_dex_rows(section, forms)
    flat = [(row.subject_id, row.section_path, row.text, row.versions) for row in rows]
    return flat, [f"{note.section_path}: {note.detail}" for note in notes]


def test_read_dex_rows_reads_1_to_3_versions_in_order() -> None:
    rows, notes = read_dex(
        "{{Dex/Entry1|v=Gold|entry=One.}}\n"
        "{{Dex/Entry2|v=Black|v2=White|entry=Two.}}\n"
        "{{Dex/Entry3|v=Diamond|v2=Pearl|v3=Platinum|entry=Three.}}"
    )

    assert rows == [
        ("diglett", POKEDEX_PATH, "One.", ("Gold",)),
        ("diglett", POKEDEX_PATH, "Two.", ("Black", "White")),
        ("diglett", POKEDEX_PATH, "Three.", ("Diamond", "Pearl", "Platinum")),
    ]
    assert notes == []


def test_read_dex_rows_skips_side_game_rows_without_a_note() -> None:
    rows, notes = read_dex(
        "{{Dex/Entry1|v=Stadium|entry=A.}}\n"
        "{{Dex/Entry1|v=Stadium 2|entry=B.}}\n"
        "{{Dex/Entry1|v=Pokopia|entry=C.}}\n"
        "{{Dex/Entry1|v=Colosseum|entry=D.}}\n"
        "{{Dex/Entry1|v=XD|entry=E.}}\n"
        "{{Dex/Entry1|v=Battle Revolution|entry=F.}}\n"
        "{{Dex/Entry1|v=Champions|entry=G.}}"
    )

    assert rows == []
    assert notes == []


def test_read_dex_rows_drops_only_the_side_game_versions_of_a_mixed_row() -> None:
    rows, _ = read_dex("{{Dex/Entry2|v=Yellow|v2=Stadium|entry=Mixed.}}")

    assert [row[3] for row in rows] == [("Yellow",)]


def test_read_dex_rows_skips_rows_without_entry_text_or_version() -> None:
    rows, _ = read_dex(
        "{{Dex/Entry1|v=Gold|entry=}}\n{{Dex/Entry1|v=Gold}}\n{{Dex/Entry1|entry=No version.}}"
    )

    assert rows == []


def test_read_dex_rows_ignores_other_dex_templates_and_stray_text() -> None:
    rows, _ = read_dex(
        "{{Dex/Header|type=Ground}}\n{{Dex/Gen/1|gen=I}}\n{{Dex/NE|[[X]]}}\n|}\n|}\n"
        "{{Dex/Entry1|v=Red|entry=Kept.}}\n{{Dex/Footer}}"
    )

    assert rows == [("diglett", POKEDEX_PATH, "Kept.", ("Red",))]


def test_read_dex_rows_switches_the_subject_at_dex_form_and_keeps_the_label_in_the_path() -> None:
    rows, notes = read_dex(
        "{{Dex/Gen/3|gen=VII}}\n"
        "{{Dex/Entry1|v=Sun|entry=Base.}}\n"
        "{{Dex/Form|Alolan Form}}\n"
        "{{Dex/Entry1|v=Sun|entry=Alolan.}}\n"
        "{{Dex/Entry1|v=Moon|entry=Alolan two.}}"
    )

    assert rows == [
        ("diglett", POKEDEX_PATH, "Base.", ("Sun",)),
        ("diglett-alola", f"{POKEDEX_PATH} > Alolan Form", "Alolan.", ("Sun",)),
        ("diglett-alola", f"{POKEDEX_PATH} > Alolan Form", "Alolan two.", ("Moon",)),
    ]
    assert notes == []


def test_read_dex_rows_returns_to_the_base_form_at_the_next_dex_gen() -> None:
    rows, _ = read_dex(
        "{{Dex/Form|Alolan Form}}\n{{Dex/Entry1|v=Sun|entry=Alolan.}}\n"
        "{{Dex/Gen/5|gen=VIII}}\n{{Dex/Entry1|v=Sword|entry=Base again.}}"
    )

    assert [(row[0], row[1]) for row in rows] == [
        ("diglett-alola", f"{POKEDEX_PATH} > Alolan Form"),
        ("diglett", POKEDEX_PATH),
    ]


def test_read_dex_rows_keeps_the_label_and_the_base_form_when_a_label_matches_no_form() -> None:
    rows, notes = read_dex("{{Dex/Form|Gigantamax}}\n{{Dex/Entry1|v=Sword|entry=Max.}}")

    assert rows == [("diglett", f"{POKEDEX_PATH} > Gigantamax", "Max.", ("Sword",))]
    assert notes == [f"{POKEDEX_PATH} > Gigantamax: Gigantamax matches no form node"]


def test_read_dex_rows_renders_text_templates_and_collects_mentions() -> None:
    section = split_sections(
        "==Game data==\n===Pokédex entries===\n"
        "{{Dex/Entry1|v=Ruby|entry=<sc>Diglett</sc> lives {{tt|one yard|1 meter}} deep "
        "near {{p|Onix|an Onix}}. {{ScPkmn}}<small>(note)</small>{{roundy|x}}}}"
    )[-1]

    (row,), _ = read_dex_rows(section, DIGLETT)

    assert row.text == "Diglett lives one yard deep near an Onix. Pokémon"
    assert row.mentions == (MentionRef(template="p", target="Onix"),)
    assert row.dropped == ("roundy",)


def test_merge_dex_rows_gives_1_chunk_per_subject_and_text_with_the_version_union() -> None:
    section = split_sections(
        "==Game data==\n===Pokédex entries===\n"
        f"{{{{Dex/Gen/2|gen=II}}}}\n{{{{Dex/Entry1|v=Gold|entry={SKIN}}}}}\n"
        "{{Dex/Entry1|v=Silver|entry=Other text.}}\n"
        f"{{{{Dex/Gen/2|gen=IV}}}}\n{{{{Dex/Entry1|v=HeartGold|entry={SKIN}}}}}\n"
        f"{{{{Dex/Gen/1|gen=VI}}}}\n{{{{Dex/Entry2|v=Y|v2=Gold|entry={SKIN}}}}}"
    )[-1]
    rows, _ = read_dex_rows(section, DIGLETT)

    chunks = merge_dex_rows(rows)

    assert [(chunk.text, chunk.versions) for chunk in chunks] == [
        (SKIN, ("Gold", "HeartGold", "Y")),
        ("Other text.", ("Silver",)),
    ]
    assert all(chunk.aspect is ChunkAspect.POKEDEX for chunk in chunks)
    assert chunks[0].section_path == POKEDEX_PATH


def test_merge_dex_rows_keeps_the_same_text_of_2_subjects_apart() -> None:
    section = split_sections(
        "==Game data==\n===Pokédex entries===\n"
        "{{Dex/Entry1|v=Sun|entry=Same.}}\n{{Dex/Form|Alolan Form}}\n"
        "{{Dex/Entry1|v=Moon|entry=Same.}}"
    )[-1]
    rows, _ = read_dex_rows(section, DIGLETT)

    chunks = merge_dex_rows(rows)

    assert [(chunk.subject_id, chunk.versions) for chunk in chunks] == [
        ("diglett", ("Sun",)),
        ("diglett-alola", ("Moon",)),
    ]


def test_read_chunks_reads_the_pokedex_section_and_its_subsections() -> None:
    page = (
        "==Game data==\n===Pokédex entries===\n{{Dex/Entry1|v=Red|entry=Main.}}\n"
        "====In side games====\n{{Dex/Entry1|v=Yellow|entry=Side.}}\n"
        "==Trivia==\n* trivia\n"
    )

    reading = read_chunks(split_sections(page), DIGLETT)

    assert [(chunk.aspect.value, chunk.section_path, chunk.text) for chunk in reading.chunks] == [
        ("trivia", "Trivia", "trivia"),
        ("pokedex", POKEDEX_PATH, "Main."),
        ("pokedex", f"{POKEDEX_PATH} > In side games", "Side."),
    ]


def test_read_chunks_counts_the_templates_dropped_from_pokedex_rows() -> None:
    page = "==Game data==\n===Pokédex entries===\n{{Dex/Entry1|v=Red|entry=A {{roundy|x}} b.}}\n"

    reading = read_chunks(split_sections(page), DIGLETT)

    assert reading.dropped_templates == {"roundy": 1}
