from collections.abc import Mapping

import pytest

from muon.ingestion.bulbapedia.pages import PageStore
from muon.ingestion.bulbapedia.species import forms
from muon.ingestion.bulbapedia.species.extract import FormOutcome, NoteKind
from muon.ingestion.bulbapedia.species.forms import (
    FormInputs,
    FormOverride,
    OverrideError,
    PageOverride,
    ResolvedForm,
    build_form_name,
    build_species_id,
    check_override_titles,
    match_subsection,
    read_form_candidates,
    read_region,
    read_sprite_code,
    resolve_forms,
)

PAGE_TITLE = "Testmon (Pokémon)"


def build_inputs(
    infobox: Mapping[str, str],
    stats_titles: tuple[str, ...] = (),
    learnset_titles: tuple[str, ...] = (),
) -> FormInputs:
    return FormInputs(PAGE_TITLE, infobox, stats_titles, learnset_titles)


def resolve_ids(
    infobox: Mapping[str, str],
    stats_titles: tuple[str, ...] = (),
    learnset_titles: tuple[str, ...] = (),
) -> list[str]:
    return [
        form.id
        for form in resolve_forms(build_inputs(infobox, stats_titles, learnset_titles)).forms
    ]


def outcomes(infobox: Mapping[str, str], **titles: tuple[str, ...]) -> dict[int, FormOutcome]:
    form_set = resolve_forms(build_inputs(infobox, **titles))
    return {result.index: result.outcome for result in form_set.candidates}


DIGLETT = {
    "name": "Diglett",
    "form2": "Alolan Form",
    "image2": "0050Diglett-Alola.png",
    "type1": "Ground",
    "form2type1": "Ground",
    "form2type2": "Steel",
}

DARMANITAN = {
    "name": "Darmanitan",
    "form1": "Standard Mode",
    "form2": "Zen Mode",
    "form3": "Galarian Form",
    "form4": "Galarian Form/Zen Mode",
    "image2": "0555Darmanitan-Zen.png",
    "image3": "0555Darmanitan-Galar.png",
    "image4": "0555Darmanitan-Galar-Zen.png",
    "form2type1": "Fire",
    "form3type1": "Ice",
    "form4type1": "Ice",
}

CHARIZARD = {
    "name": "Charizard",
    "form2": "Mega Charizard X",
    "form3": "Mega Charizard Y",
    "form4": "Gigantamax Form",
    "image2": "0006Charizard-Mega X.png",
    "image3": "0006Charizard-Mega Y.png",
    "form2type1": "Fire",
    "form3type1": "Fire",
    "form4type1": "Fire",
}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Diglett", "diglett"),
        ("Nidoran♀", "nidoran-f"),
        ("Nidoran♂", "nidoran-m"),
        ("Mr. Mime", "mr-mime"),
        ("Farfetch'd", "farfetchd"),
        ("Type: Null", "type-null"),
    ],
)
def test_build_species_id(name: str, expected: str) -> None:
    assert build_species_id(name) == expected


@pytest.mark.parametrize(
    ("file_or_number", "base_name", "expected"),
    [
        ("0050Diglett-Alola.png", "Diglett", "Alola"),
        ("0051Dugtrio-Alola", "Dugtrio", "Alola"),
        ("0050", "Diglett", ""),
        ("0050Diglett.png", "Diglett", ""),
        ("0006Charizard-Mega X.png", "Charizard", "Mega X"),
        ("006charizard-Mega X.PNG", "Charizard", "Mega X"),
        ("0128Tauros-Paldea Combat.png", "Tauros", "Paldea Combat"),
        ("0122Mr. Mime-Galar.png", "Mr. Mime", "Galar"),
        ("0555Darmanitan-Galar-Zen.png", "Darmanitan", "Galar-Zen"),
        ("HOME0669Y.png", "Flabébé", "HOME0669Y"),
        ("0439Mime Jr", "Mime Jr.", ""),
        ("0439Mime Jr-Galar", "Mime Jr.", "Galar"),
        ("0772Type Null", "Type: Null", ""),
    ],
)
def test_read_sprite_code(file_or_number: str, base_name: str, expected: str) -> None:
    assert read_sprite_code(file_or_number, base_name) == expected


@pytest.mark.parametrize(
    ("infobox_name", "expected"),
    [
        ("Alolan Form", "Alola"),
        ("Galarian Form", "Galar"),
        ("Hisuian Form", "Hisui"),
        ("Paldean Form (Combat Breed)", "Paldea"),
        ("Galarian Form/Zen Mode", None),
        ("Zen Mode", None),
        ("Mega Charizard X", None),
    ],
)
def test_read_region(infobox_name: str, expected: str | None) -> None:
    assert read_region(infobox_name) == expected


@pytest.mark.parametrize(
    ("base_name", "infobox_name", "expected"),
    [
        ("Diglett", "Alolan Form", "Alolan Diglett"),
        ("Tauros", "Paldean Form (Combat Breed)", "Paldean Tauros (Combat Breed)"),
        ("Darmanitan", "Galarian Form/Zen Mode", "Galarian Darmanitan (Zen Mode)"),
        ("Groudon", "Primal Reversion", "Primal Groudon"),
        ("Rotom", "Heat Rotom", "Heat Rotom"),
        ("Charizard", "Mega Charizard X", "Mega Charizard X"),
        ("Kyurem", "White Kyurem", "White Kyurem"),
        ("Darmanitan", "Zen Mode", "Darmanitan (Zen Mode)"),
        ("Aegislash", "Blade Forme", "Aegislash (Blade Forme)"),
    ],
)
def test_build_form_name(base_name: str, infobox_name: str, expected: str) -> None:
    assert build_form_name(base_name, infobox_name) == expected


def test_read_form_candidates_without_form_fields_gives_the_default_only() -> None:
    candidates = read_form_candidates({"name": "Unown", "image": "0201Unown.png"})

    assert [(c.index, c.infobox_name, c.image) for c in candidates] == [(1, "", "0201Unown.png")]


def test_read_form_candidates_reads_names_images_and_comments() -> None:
    infobox = {
        "name": "Rotom",
        "form1": "Rotom<!--Has form name-->",
        "form2": "Heat&nbsp;Rotom",
        "form3": "Stereo Rotom\n<!-- not a form -->",
        "form4": "  ",
        "image2": "0479Rotom-Heat.png",
        "image1": "0479Rotom.png",
    }

    candidates = read_form_candidates(infobox)

    assert [(c.index, c.infobox_name, c.image) for c in candidates] == [
        (1, "Rotom", "0479Rotom.png"),
        (2, "Heat Rotom", "0479Rotom-Heat.png"),
        (3, "Stereo Rotom", None),
    ]


def test_resolve_forms_gives_a_page_without_forms_one_node() -> None:
    form_set = resolve_forms(build_inputs({"name": "Unown", "type1": "Psychic"}))

    assert [form.id for form in form_set.forms] == ["unown"]
    assert [r.outcome for r in form_set.candidates] == [FormOutcome.NODE]
    assert form_set.forms[0].is_default


def test_resolve_forms_builds_regional_ids_names_and_regions() -> None:
    form_set = resolve_forms(build_inputs(DIGLETT))

    assert form_set.forms == (
        ResolvedForm("diglett", "Diglett", 1, "", None, True),
        ResolvedForm("diglett-alola", "Alolan Diglett", 2, "Alolan Form", "Alola", False),
    )


def test_resolve_forms_gives_a_combined_name_no_region() -> None:
    form_set = resolve_forms(build_inputs(DARMANITAN))

    assert [(f.id, f.name, f.region) for f in form_set.forms] == [
        ("darmanitan", "Darmanitan", None),
        ("darmanitan-zen", "Darmanitan (Zen Mode)", None),
        ("darmanitan-galar", "Galarian Darmanitan", "Galar"),
        ("darmanitan-galar-zen", "Galarian Darmanitan (Zen Mode)", None),
    ]


def test_resolve_forms_skips_gigantamax_even_with_own_types() -> None:
    form_set = resolve_forms(build_inputs(CHARIZARD))

    assert [form.id for form in form_set.forms] == [
        "charizard",
        "charizard-mega-x",
        "charizard-mega-y",
    ]
    assert form_set.candidates[3].outcome is FormOutcome.SKIPPED
    assert form_set.candidates[3].form_id is None


def test_resolve_forms_skips_gigantamax_named_by_a_subsection() -> None:
    infobox = {"name": "Meowth", "form2": "Gigantamax Form"}

    form_set = resolve_forms(build_inputs(infobox, stats_titles=("Gigantamax Meowth",)))

    assert [form.id for form in form_set.forms] == ["meowth"]
    assert [note.kind for note in form_set.notes] == [NoteKind.UNMATCHED_SUBSECTION]


@pytest.mark.parametrize(
    ("extra", "is_node"),
    [
        ({"form2type1": "Fire"}, True),
        ({"ability2-1": "Blaze"}, True),
        ({"ability2-2": "Blaze"}, True),
        ({"form2type1": "  "}, False),
        ({"type1": "Fire", "ability1": "Blaze"}, False),
    ],
)
def test_node_rule_own_types_or_abilities(extra: Mapping[str, str], is_node: bool) -> None:
    infobox = {"name": "Testmon", "form2": "Hot Form", **extra}

    assert (outcomes(infobox)[2] is FormOutcome.NODE) is is_node
    assert outcomes(infobox)[2] in (FormOutcome.NODE, FormOutcome.COSMETIC)


def test_node_rule_third_form_reads_its_own_ability_keys() -> None:
    infobox = {"name": "Testmon", "form2": "A Form", "form3": "B Form", "ability3-1": "Blaze"}

    assert outcomes(infobox) == {1: FormOutcome.NODE, 2: FormOutcome.COSMETIC, 3: FormOutcome.NODE}


def test_node_rule_mega_abilities_follow_mega_order() -> None:
    infobox = {
        "name": "Testmon",
        "form2": "Alolan Form",
        "form3": "Mega Testmon X",
        "form4": "Mega Testmon Y",
        "abilitym": "Tough Claws",
    }

    assert outcomes(infobox) == {
        1: FormOutcome.NODE,
        2: FormOutcome.COSMETIC,
        3: FormOutcome.NODE,
        4: FormOutcome.COSMETIC,
    }
    assert outcomes({**infobox, "abilitym2": "Drought"})[4] is FormOutcome.NODE


def test_node_rule_a_stats_or_learnset_title_names_the_form() -> None:
    infobox = {"name": "Testmon", "form2": "Hot Form", "form3": "Cold Form"}

    by_stats = outcomes(infobox, stats_titles=("Testmon (Hot Form)",))
    by_learnset = outcomes(infobox, learnset_titles=("Cold Form",))

    assert by_stats == {1: FormOutcome.NODE, 2: FormOutcome.NODE, 3: FormOutcome.COSMETIC}
    assert by_learnset == {1: FormOutcome.NODE, 2: FormOutcome.COSMETIC, 3: FormOutcome.NODE}


def test_cosmetic_forms_keep_no_id() -> None:
    infobox = {"name": "Pikachu", "form2": "Cap", "form3": "Pale"}

    form_set = resolve_forms(build_inputs(infobox))

    assert [form.id for form in form_set.forms] == ["pikachu"]
    assert [(r.outcome, r.form_id) for r in form_set.candidates[1:]] == [
        (FormOutcome.COSMETIC, None),
        (FormOutcome.COSMETIC, None),
    ]


def test_form_id_without_an_image_uses_the_form_name() -> None:
    infobox = {"name": "Gourgeist", "form2": "Small Variety"}

    assert resolve_ids(infobox, stats_titles=("Gourgeist (Small Variety)",)) == [
        "gourgeist",
        "gourgeist-small-variety",
    ]


def test_form_id_with_an_empty_sprite_code_uses_the_form_name() -> None:
    infobox = {"name": "Burmy", "form2": "Sandy Cloak", "image2": "0412Burmy.png"}

    assert resolve_ids(infobox, learnset_titles=("Sandy Cloak",)) == [
        "burmy",
        "burmy-sandy-cloak",
    ]


def forms_for(infobox: Mapping[str, str], titles: tuple[str, ...]) -> tuple[ResolvedForm, ...]:
    candidates = read_form_candidates(infobox)
    base = infobox["name"]
    return tuple(
        ResolvedForm(
            build_species_id(base) if c.index == 1 else f"{build_species_id(base)}-{c.index}",
            base if c.index == 1 else build_form_name(base, c.infobox_name),
            c.index,
            c.infobox_name,
            read_region(c.infobox_name),
            c.index == 1,
        )
        for c in candidates
    )


ROTOM = {
    "name": "Rotom",
    "form2": "Heat Rotom",
    "form3": "Wash Rotom",
    "form4": "Frost Rotom",
    "form5": "Fan Rotom",
    "form6": "Mow Rotom",
}


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Rotom", ("rotom",)),
        ("Heat Rotom", ("rotom-2",)),
        (
            "Heat, Wash, Frost, Fan, and Mow Rotom",
            ("rotom-2", "rotom-3", "rotom-4", "rotom-5", "rotom-6"),
        ),
        ("Heat and Wash Rotom", ("rotom-2", "rotom-3")),
        ("Heat Rotom, Wash Rotom", ("rotom-2", "rotom-3")),
        ("Heat, Cold, and Wash Rotom", ()),
        ("Partner Rotom", ()),
    ],
)
def test_match_subsection_names_and_lists(title: str, expected: tuple[str, ...]) -> None:
    assert match_subsection(title, forms_for(ROTOM, ())) == expected


def test_match_subsection_regional_title_covers_every_form_of_the_region() -> None:
    infobox = {
        "name": "Tauros",
        "form2": "Paldean Form (Combat Breed)",
        "form3": "Paldean Form (Blaze Breed)",
        "form4": "Paldean Form (Aqua Breed)",
    }
    resolved = forms_for(infobox, ())

    assert match_subsection("Paldean Tauros", resolved) == ("tauros-2", "tauros-3", "tauros-4")
    assert match_subsection("Paldean Tauros (Blaze Breed)", resolved) == ("tauros-3",)
    assert match_subsection("Tauros", resolved) == ("tauros",)
    assert match_subsection("Alolan Tauros", resolved) == ()


@pytest.mark.parametrize(
    ("infobox", "title", "expected"),
    [
        ({"name": "Calyrex", "form2": "Ice Rider"}, "Ice Rider Calyrex", ("calyrex-2",)),
        ({"name": "Indeedee", "form1": "Male", "form2": "Female"}, "Male Indeedee", ("indeedee",)),
        (
            {"name": "Basculin", "form1": "Red-Striped Form", "form2": "Blue-Striped Form"},
            "Red-Striped/Blue-Striped Basculin",
            ("basculin", "basculin-2"),
        ),
        ({"name": "Kyurem", "form2": "White Kyurem"}, "Kyurem (White Kyurem)", ("kyurem-2",)),
    ],
)
def test_match_subsection_spellings(
    infobox: Mapping[str, str], title: str, expected: tuple[str, ...]
) -> None:
    assert match_subsection(title, forms_for(infobox, ())) == expected


def test_resolve_forms_maps_titles_from_stats_and_learnset_to_node_ids() -> None:
    form_set = resolve_forms(
        build_inputs(
            DIGLETT,
            stats_titles=("Diglett", "Alolan Diglett"),
            learnset_titles=("Diglett", "Alolan Diglett", "Alolan Diglett"),
        )
    )

    assert form_set.subsections == {
        "Diglett": ("diglett",),
        "Alolan Diglett": ("diglett-alola",),
    }
    assert form_set.notes == ()


def test_resolve_forms_notes_an_unmatched_title_once_per_group() -> None:
    infobox = {"name": "Pikachu", "form2": "Cap"}

    form_set = resolve_forms(
        build_inputs(
            infobox,
            stats_titles=("Pikachu", "Partner Pikachu"),
            learnset_titles=("Partner Pikachu", "Partner Pikachu"),
        )
    )

    assert [(n.kind, n.section_path) for n in form_set.notes] == [
        (NoteKind.UNMATCHED_SUBSECTION, "Partner Pikachu"),
        (NoteKind.UNMATCHED_SUBSECTION, "Partner Pikachu"),
    ]
    assert "Base stats" in form_set.notes[0].detail
    assert "Learnset" in form_set.notes[1].detail
    assert "Partner Pikachu" not in form_set.subsections


@pytest.fixture
def overrides(monkeypatch: pytest.MonkeyPatch) -> dict[str, PageOverride]:
    registry: dict[str, PageOverride] = {}
    monkeypatch.setattr(forms, "FORM_OVERRIDES", registry)
    return registry


def test_override_sets_id_name_and_outcome(overrides: dict[str, PageOverride]) -> None:
    infobox = {"name": "Pikachu", "form2": "Cap", "form3": "Pale"}
    overrides[PAGE_TITLE] = PageOverride(
        forms={2: FormOverride(form_id="pikachu-cap", name="Capped Pikachu", is_node=True)}
    )

    form_set = resolve_forms(build_inputs(infobox))

    assert [(f.id, f.name) for f in form_set.forms] == [
        ("pikachu", "Pikachu"),
        ("pikachu-cap", "Capped Pikachu"),
    ]
    assert [(r.outcome, r.form_id) for r in form_set.candidates] == [
        (FormOutcome.NODE, "pikachu"),
        (FormOutcome.OVERRIDDEN, "pikachu-cap"),
        (FormOutcome.COSMETIC, None),
    ]
    assert [(n.kind, n.section_path) for n in form_set.notes] == [
        (NoteKind.OVERRIDE_APPLIED, "Cap")
    ]


def test_override_can_turn_a_rule_node_into_a_cosmetic_form(
    overrides: dict[str, PageOverride],
) -> None:
    infobox = {"name": "Rotom", "form2": "Stereo Rotom", "form2type1": "Electric"}
    overrides[PAGE_TITLE] = PageOverride(forms={2: FormOverride(is_node=False)})

    form_set = resolve_forms(build_inputs(infobox))

    assert [form.id for form in form_set.forms] == ["rotom"]
    assert form_set.candidates[1].outcome is FormOutcome.OVERRIDDEN
    assert form_set.candidates[1].form_id is None


def test_override_with_an_id_only_keeps_the_rule_outcome(
    overrides: dict[str, PageOverride],
) -> None:
    infobox = {"name": "Testmon", "form2": "Hot Form", "form2type1": "Fire"}
    overrides[PAGE_TITLE] = PageOverride(forms={2: FormOverride(form_id="testmon-hot")})

    form_set = resolve_forms(build_inputs(infobox))

    assert [form.id for form in form_set.forms] == ["testmon", "testmon-hot"]


def test_override_maps_a_subsection_title_to_form_ids(
    overrides: dict[str, PageOverride],
) -> None:
    infobox = {"name": "Deoxys", "form2": "Attack Forme", "form2type1": "Psychic"}
    overrides[PAGE_TITLE] = PageOverride(
        subsections={"All forms": ("deoxys", "deoxys-attack-forme")}
    )

    form_set = resolve_forms(build_inputs(infobox, learnset_titles=("All forms",)))

    assert form_set.subsections == {"All forms": ("deoxys", "deoxys-attack-forme")}
    assert [(n.kind, n.section_path) for n in form_set.notes] == [
        (NoteKind.OVERRIDE_APPLIED, "All forms")
    ]


def test_override_with_a_stale_form_index_raises(overrides: dict[str, PageOverride]) -> None:
    overrides[PAGE_TITLE] = PageOverride(forms={5: FormOverride(is_node=True)})

    with pytest.raises(OverrideError, match=r"Testmon \(Pokémon\).*\[5\]"):
        resolve_forms(build_inputs({"name": "Testmon", "form2": "Hot Form"}))


def test_override_with_a_stale_subsection_raises(overrides: dict[str, PageOverride]) -> None:
    overrides[PAGE_TITLE] = PageOverride(subsections={"All forms": ("testmon",)})

    with pytest.raises(OverrideError, match="All forms"):
        resolve_forms(build_inputs({"name": "Testmon"}, learnset_titles=("Testmon",)))


def test_override_mapping_a_subsection_to_an_unknown_form_raises(
    overrides: dict[str, PageOverride],
) -> None:
    overrides[PAGE_TITLE] = PageOverride(subsections={"All forms": ("testmon-ghost",)})

    with pytest.raises(OverrideError, match="testmon-ghost"):
        resolve_forms(build_inputs({"name": "Testmon"}, learnset_titles=("All forms",)))


def test_check_override_titles_rejects_an_unpinned_page(
    overrides: dict[str, PageOverride],
) -> None:
    overrides["Ghost (Pokémon)"] = PageOverride()

    with pytest.raises(OverrideError, match="Ghost"):
        check_override_titles(["Diglett (Pokémon)"])
    check_override_titles(["Diglett (Pokémon)", "Ghost (Pokémon)"])


def test_check_override_titles_passes_with_no_overrides(
    overrides: dict[str, PageOverride],
) -> None:
    check_override_titles([])


def test_the_committed_overrides_name_pinned_species_pages() -> None:
    pins = PageStore().read_manifest().list_pins("species")

    assert forms.FORM_OVERRIDES
    check_override_titles([pin.title for pin in pins])
