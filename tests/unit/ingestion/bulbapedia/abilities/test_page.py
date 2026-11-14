from datetime import UTC, datetime

import pytest

from muon.ingestion.bulbapedia.abilities.page import (
    AbilityPageExtract,
    AbilityPageParseError,
    DescRow,
    HolderEntry,
    SkippedPage,
    SkipReason,
    parse_ability_page,
)
from muon.ingestion.bulbapedia.pages import Page, compute_sha1

PAGE_TITLE = "Test Guard (Ability)"
INFOBOX = (
    "{{AbilityInfobox/header\n|name=Test Guard\n|jpname=テストガード\n|jptrans=Test Guard\n"
    "|jptranslit=Tesuto Gādo\n|colorscheme=normal\n|gen={gen}\n}}\n"
    "{{AbilityInfobox/desc|III|Guards in tests.}}\n"
    "{{AbilityInfobox/desc|IV|Guards the test.{{sup/4|DPP}}<br>Guards tests.{{sup/4|HGSS}}}}\n"
    "{{AbilityInfobox/desc/Champions|Guards in Champions.}}\n"
    "{{AbilityInfobox/footer}}\n"
)
BODY = (
    "'''Test Guard''' is an Ability.\n"
    "==Effect==\n===In battle===\nGuards {{p|Pikachu}}.\n"
    "==Pokémon with Test Guard==\n{{Ability/head|normal}}\n"
    "{{Ability/entry|0025|Pikachu|Electric|Electric|Static|None|Test Guard}}\n"
    "{{Ability/entry|0026|form=-Alola|Raichu|Electric|Psychic|Surge Surfer|None|Test Guard"
    "|formlink={{color2|000|Alolan Raichu|Alolan Form}}}}\n"
    "{{Ability/foot|normal}}\n"
    "==In other languages==\n{{LANGTABLE|color={{normal color}}\n"
    "|fr=Garde Test\n|de=Testschutz\n}}\n"
    "[[Category:Abilities that guard]]\n[[Category:Generation III Abilities]]\n"
)


def build_page(wikitext: str, title: str = PAGE_TITLE) -> Page:
    return Page(
        title=title,
        page_id=1,
        revision_id=2,
        timestamp=datetime(2026, 11, 14, tzinfo=UTC),
        sha1=compute_sha1(wikitext),
        page_type="ability",
        wikitext=wikitext,
    )


def build_ability_wikitext(gen: str = "III", langtable: str = "Langtable") -> str:
    return INFOBOX.replace("{gen}", gen) + BODY.replace("LANGTABLE", langtable)


def parse(wikitext: str) -> AbilityPageExtract:
    extract = parse_ability_page(build_page(wikitext))
    assert isinstance(extract, AbilityPageExtract)
    return extract


def test_a_page_without_the_infobox_is_skipped() -> None:
    skipped = parse_ability_page(build_page("'''Test Guard''' is an Ability.\n"))

    assert skipped == SkippedPage(
        title=PAGE_TITLE, page_id=1, ability_id="test-guard", reason=SkipReason.NO_INFOBOX
    )


def test_id_name_and_generation_come_from_the_title_and_gen() -> None:
    extract = parse(build_ability_wikitext())

    assert (extract.id, extract.name, extract.generation_label) == (
        "test-guard",
        "Test Guard",
        "Generation III",
    )


def test_a_gen_that_is_no_roman_numeral_fails_the_page() -> None:
    with pytest.raises(AbilityPageParseError, match="gen"):
        parse_ability_page(build_page(build_ability_wikitext(gen="3")))


@pytest.mark.parametrize("langtable", ["Langtable", "langtable"])
def test_aliases_read_both_spellings_of_the_language_table(langtable: str) -> None:
    extract = parse(build_ability_wikitext(langtable=langtable))

    assert {"Garde Test", "Testschutz", "テストガード"} <= set(extract.aliases)


def test_desc_rows_keep_their_generation_and_raw_text_and_skip_champions() -> None:
    extract = parse(build_ability_wikitext())

    assert extract.desc_rows == (
        DescRow(generation_label="Generation III", wikitext="Guards in tests."),
        DescRow(
            generation_label="Generation IV",
            wikitext="Guards the test.{{sup/4|DPP}}<br>Guards tests.{{sup/4|HGSS}}",
        ),
    )


def test_holders_read_the_species_of_every_entry_row_named_parameters_aside() -> None:
    extract = parse(build_ability_wikitext())

    assert extract.holders == (
        HolderEntry(species_name="Pikachu"),
        HolderEntry(species_name="Raichu"),
    )


def test_only_the_abilities_categories_are_counted_apart() -> None:
    extract = parse(build_ability_wikitext())

    assert extract.ability_categories == ("Abilities that guard",)
    assert extract.categories == ("Abilities that guard", "Generation III Abilities")
