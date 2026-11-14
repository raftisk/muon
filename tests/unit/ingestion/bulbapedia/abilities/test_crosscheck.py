import json
from pathlib import Path

from muon.ingestion.bulbapedia.abilities.crosscheck import (
    HAS_ABILITY,
    HolderDifference,
    HolderSide,
    cross_check_species,
    read_species_abilities,
)
from muon.ingestion.bulbapedia.abilities.page import AbilityPageExtract, HolderEntry


def write_species_output(output_dir: Path, title: str, *edges: tuple[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    content = {
        "title": title,
        "species_batch": {
            "edges": [
                {"property_iri": HAS_ABILITY, "start": {"id": start}, "end": {"id": end}}
                for start, end in edges
            ]
        },
    }
    (output_dir / f"{len(list(output_dir.iterdir()))}.json").write_text(json.dumps(content))


def build_extract(*species_names: str) -> AbilityPageExtract:
    return AbilityPageExtract(
        title="Static (Ability)",
        page_id=1,
        revision_id=2,
        id="static",
        name="Static",
        generation_label="Generation III",
        holders=tuple(HolderEntry(species_name=name) for name in species_names),
    )


def test_a_missing_species_folder_gives_no_index(tmp_path: Path) -> None:
    assert read_species_abilities(tmp_path / "species") is None


def test_a_form_edge_and_a_form_row_both_count_for_the_base_species(tmp_path: Path) -> None:
    output_dir = tmp_path / "species"
    write_species_output(output_dir, "Raichu (Pokémon)", ("raichu-alola", "static"))
    index = read_species_abilities(output_dir)
    assert index is not None

    assert cross_check_species([build_extract("Raichu")], index) == ()


def test_both_sides_of_a_difference_are_reported(tmp_path: Path) -> None:
    output_dir = tmp_path / "species"
    write_species_output(output_dir, "Pikachu (Pokémon)", ("pikachu", "static"))
    write_species_output(output_dir, "Electabuzz (Pokémon)", ("electabuzz", "static"))
    write_species_output(output_dir, "Pichu (Pokémon)", ("pichu", "lightning-rod"))
    index = read_species_abilities(output_dir)
    assert index is not None

    differences = cross_check_species([build_extract("Pikachu", "Pichu")], index)

    assert differences == (
        HolderDifference(ability_id="static", species_id="pichu", side=HolderSide.TABLE_ONLY),
        HolderDifference(
            ability_id="static", species_id="electabuzz", side=HolderSide.SPECIES_ONLY
        ),
    )
