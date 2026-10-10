"""Extract-level expectations of the spec scenarios, checked on the pinned pilot pages."""

from collections import Counter
from collections.abc import Callable

import pytest

from muon.ingestion.bulbapedia.pages import Page
from muon.ingestion.bulbapedia.species.extract import (
    FormFacts,
    FormOutcome,
    NoteKind,
    ResidueKind,
    SpeciesPageExtract,
)
from muon.ingestion.bulbapedia.species.page import parse_species_page

pytestmark = pytest.mark.pages

Extracts = dict[str, SpeciesPageExtract]


def form_ids(extract: SpeciesPageExtract) -> list[str]:
    return [form.id for form in extract.forms]


def form_of(extract: SpeciesPageExtract, form_id: str) -> FormFacts:
    return next(form for form in extract.forms if form.id == form_id)


def abilities_of(extract: SpeciesPageExtract, form_id: str) -> list[tuple[str, bool]]:
    return [(slot.name, slot.is_hidden) for slot in form_of(extract, form_id).abilities]


def learnset_counts(extract: SpeciesPageExtract, form_id: str) -> dict[str, int]:
    methods = Counter(row.method.value for row in extract.learnset if row.form_id == form_id)
    return dict(methods)


def residue_forms(extract: SpeciesPageExtract, kind: ResidueKind) -> list[tuple[str, ...]]:
    return [item.form_ids for item in extract.residue if item.kind is kind]


def test_diglett(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Diglett (Pokémon)"]

    assert (extract.page_id, extract.revision_id) == (2482, 4630045)
    assert [(f.id, f.name, f.region) for f in extract.forms] == [
        ("diglett", "Diglett", None),
        ("diglett-alola", "Alolan Diglett", "Alola"),
    ]
    alola = form_of(extract, "diglett-alola")
    assert (alola.weight, alola.height, alola.types) == (1.0, 0.2, ("Ground", "Steel"))
    assert form_of(extract, "diglett").generation == 1
    assert learnset_counts(extract, "diglett") == {
        "level": 13,
        "machine": 40,
        "egg": 5,
        "special": 1,
    }
    assert learnset_counts(extract, "diglett-alola") == {"level": 13, "machine": 42, "egg": 7}
    assert [
        (e.source.code, e.target.name, e.target.code, e.trigger, e.min_level)
        for e in extract.evolutions
    ] == [("", "Dugtrio", "", "level", 26), ("Alola", "Dugtrio", "Alola", "level", 26)]


def test_meowth(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Meowth (Pokémon)"]

    assert form_ids(extract) == ["meowth", "meowth-alola", "meowth-galar"]
    assert [r.outcome for r in extract.candidates if r.infobox_name == "Gigantamax"] == [
        FormOutcome.SKIPPED
    ]
    assert ("Unnerve", True) in abilities_of(extract, "meowth")
    assert abilities_of(extract, "meowth-alola") == [("Rattled", True)]
    assert abilities_of(extract, "meowth-galar") == [("Unnerve", True)]
    assert residue_forms(extract, ResidueKind.ABILITIES) == [("meowth-alola",), ("meowth-galar",)]


def test_tauros(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Tauros (Pokémon)"]

    paldean = ["tauros-paldea-combat", "tauros-paldea-blaze", "tauros-paldea-aqua"]
    assert form_ids(extract) == ["tauros", *paldean]
    assert {form_of(extract, i).region for i in paldean} == {"Paldea"}
    assert len({form_of(extract, i).base_stats for i in paldean}) == 1
    assert form_of(extract, paldean[0]).base_stats != form_of(extract, "tauros").base_stats
    assert ("Sheer Force", True) in abilities_of(extract, "tauros")
    for form_id in paldean:
        assert ("Cud Chew", True) in abilities_of(extract, form_id)
        assert ("Sheer Force", True) not in abilities_of(extract, form_id)


def test_rotom(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Rotom (Pokémon)"]

    appliances = ["rotom-heat", "rotom-wash", "rotom-frost", "rotom-fan", "rotom-mow"]
    assert form_ids(extract) == ["rotom", *appliances]
    assert len({form_of(extract, i).base_stats for i in appliances}) == 1
    assert residue_forms(extract, ResidueKind.FORM_CHANGE) == [(i,) for i in appliances]
    assert [r.outcome for r in extract.candidates if r.infobox_name == "Stereo Rotom"] == [
        FormOutcome.OVERRIDDEN
    ]


def test_darmanitan(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Darmanitan (Pokémon)"]

    assert [(f.id, f.name, f.region) for f in extract.forms] == [
        ("darmanitan", "Darmanitan", None),
        ("darmanitan-zen", "Darmanitan (Zen Mode)", None),
        ("darmanitan-galar", "Galarian Darmanitan", "Galar"),
        ("darmanitan-galar-zen", "Galarian Darmanitan (Zen Mode)", None),
    ]
    assert ("Gorilla Tactics", False) in abilities_of(extract, "darmanitan-galar")
    assert ("Gorilla Tactics", False) not in abilities_of(extract, "darmanitan-zen")
    assert residue_forms(extract, ResidueKind.FORM_CHANGE) == [
        ("darmanitan-zen",),
        ("darmanitan-galar-zen",),
    ]
    assert [r.outcome for r in extract.candidates].count(FormOutcome.OVERRIDDEN) == 0
    assert any(n.kind is NoteKind.OVERRIDE_APPLIED for n in extract.notes)


def test_unown(pilot_extracts: Extracts) -> None:
    assert form_ids(pilot_extracts["Unown (Pokémon)"]) == ["unown"]


def test_gourgeist_lists_the_outcome_of_every_size_form(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Gourgeist (Pokémon)"]

    sizes = {r.infobox_name: (r.outcome, r.form_id) for r in extract.candidates if r.infobox_name}
    assert set(sizes) == {"Medium Variety", "Small Variety", "Large Variety", "Jumbo Variety"}
    assert all(outcome is FormOutcome.NODE for outcome, _ in sizes.values())
    assert sizes["Small Variety"][1] == "gourgeist-small-variety"


def test_charizard(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Charizard (Pokémon)"]

    assert form_ids(extract) == ["charizard", "charizard-mega-x", "charizard-mega-y"]
    assert form_of(extract, "charizard-mega-x").types == ("Fire", "Dragon")
    assert abilities_of(extract, "charizard-mega-x") == [("Tough Claws", False)]
    assert abilities_of(extract, "charizard-mega-y") == [("Drought", False)]
    assert [(c.from_id, c.to_id, c.form_kind, c.held_item) for c in extract.form_changes] == [
        ("charizard", "charizard-mega-x", "mega", "Charizardite X"),
        ("charizard", "charizard-mega-y", "mega", "Charizardite Y"),
    ]
    assert all(c.is_battle_only and c.trigger is None for c in extract.form_changes)
    assert FormOutcome.SKIPPED in {r.outcome for r in extract.candidates}


def test_groudon(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Groudon (Pokémon)"]

    assert [(f.id, f.name) for f in extract.forms] == [
        ("groudon", "Groudon"),
        ("groudon-primal", "Primal Groudon"),
    ]
    assert [(c.to_id, c.form_kind, c.trigger, c.held_item) for c in extract.form_changes] == [
        ("groudon-primal", "primal", "switch_in", "Red Orb")
    ]


def test_rayquaza(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Rayquaza (Pokémon)"]

    assert "rayquaza-mega" in form_ids(extract)
    assert extract.form_changes == ()
    assert residue_forms(extract, ResidueKind.FORM_CHANGE) == [("rayquaza-mega",)]


def test_eevee_evolutions_are_residue(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Eevee (Pokémon)"]

    assert extract.evolutions == ()
    (item,) = (r for r in extract.residue if r.kind is ResidueKind.EVOLUTION)
    assert item.section_path == "Game data > Evolution data"
    assert "Evobox/Setup" in item.wikitext


def test_tyrogue_evolutions_keep_their_condition(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Tyrogue (Pokémon)"]

    assert [(e.target.name, e.condition) for e in extract.evolutions] == [
        ("Hitmonlee", "Level 20 (Attack > Defense)"),
        ("Hitmonchan", "Level 20 (Attack < Defense)"),
        ("Hitmontop", "Level 20 (Attack = Defense)"),
    ]
    assert all(e.trigger is None and e.min_level is None for e in extract.evolutions)
    assert len(residue_forms(extract, ResidueKind.EVOLUTION)) == 3


def test_kyurem(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Kyurem (Pokémon)"]

    assert form_ids(extract) == ["kyurem", "kyurem-white", "kyurem-black"]
    assert extract.form_changes == ()
    assert residue_forms(extract, ResidueKind.FORM_CHANGE) == [
        ("kyurem-white",),
        ("kyurem-black",),
    ]
    assert "forme" in next(r for r in extract.residue if r.kind is ResidueKind.FORM_CHANGE).wikitext


def test_abra_has_no_scarlet_violet_learnset(pilot_extracts: Extracts) -> None:
    extract = pilot_extracts["Abra (Pokémon)"]

    assert extract.learnset == ()
    assert [n.kind for n in extract.notes if n.kind is NoteKind.MISSING_LEARNSET]
    assert [(e.target.name, e.trigger, e.min_level) for e in extract.evolutions][0] == (
        "Kadabra",
        "level",
        16,
    )


def test_pikachu_reports_the_partner_subsection_and_makes_no_cosmetic_node(
    pilot_extracts: Extracts,
) -> None:
    extract = pilot_extracts["Pikachu (Pokémon)"]

    assert form_ids(extract) == ["pikachu"]
    assert [n.section_path for n in extract.notes if n.kind is NoteKind.UNMATCHED_SUBSECTION] == [
        "Partner Pikachu"
    ]
    outcomes = {r.outcome for r in extract.candidates if r.infobox_name}
    assert outcomes <= {FormOutcome.COSMETIC, FormOutcome.SKIPPED}


def test_nidoran_ids(pilot_extracts: Extracts) -> None:
    assert form_ids(pilot_extracts["Nidoran♀ (Pokémon)"]) == ["nidoran-f"]
    assert form_ids(pilot_extracts["Nidoran♂ (Pokémon)"]) == ["nidoran-m"]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Mr. Mime (Pokémon)", "mr-mime"),
        ("Farfetch'd (Pokémon)", "farfetchd"),
        ("Type: Null (Pokémon)", "type-null"),
    ],
)
def test_punctuated_species_ids(
    read_pinned_page: Callable[[str], Page], title: str, expected: str
) -> None:
    extract = parse_species_page(read_pinned_page(title))

    assert extract is not None
    assert extract.forms[0].id == expected


def test_a_page_without_an_infobox_is_skipped(read_pinned_page: Callable[[str], Page]) -> None:
    assert parse_species_page(read_pinned_page("Browt (Pokémon)")) is None
