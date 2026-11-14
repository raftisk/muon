from pathlib import Path

from muon.ingestion.bulbapedia.abilities.pokeapi import (
    ABILITIES_FILE,
    PROSE_FILE,
    build_pokeapi_index,
    read_effect_text,
)
from tests.unit.ingestion.bulbapedia.moves.test_pokeapi import write_snapshot

TABLES = {
    ABILITIES_FILE: (
        "id,identifier,generation_id,is_main_series",
        "1,stench,3,1",
        "2,drizzle,3,1",
    ),
    PROSE_FILE: (
        "ability_id,local_language_id,short_effect,effect",
        "1,9,Has a 10% chance of making target flinch.,Long text.",
        "2,5,Fait tomber la pluie.,Texte long.",
    ),
}


def test_an_ability_with_english_text_gets_it(tmp_path: Path) -> None:
    index = build_pokeapi_index(write_snapshot(tmp_path, TABLES))

    assert read_effect_text(index, "stench") == "Has a 10% chance of making target flinch."


def test_an_ability_with_only_another_language_gets_none(tmp_path: Path) -> None:
    index = build_pokeapi_index(write_snapshot(tmp_path, TABLES))

    assert read_effect_text(index, "drizzle") is None
    assert "drizzle" in index.identifiers


def test_an_ability_without_a_row_gets_none(tmp_path: Path) -> None:
    index = build_pokeapi_index(write_snapshot(tmp_path, TABLES))

    assert read_effect_text(index, "cacophony") is None
    assert "cacophony" not in index.identifiers
