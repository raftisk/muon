"""The `effectText` of an ability: the English `short_effect` of its PokeAPI row.

PokeAPI identifiers are the slugs Bulbapedia ids use, so an ability matches by identifier
alone. An ability with no row, or a row with no English text, gets no `effectText`.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from muon.ingestion.pokeapi.snapshot import SnapshotStore

ABILITIES_FILE = "abilities.csv"
PROSE_FILE = "ability_prose.csv"
ENGLISH_LANGUAGE_ID = "9"
ID_COLUMN = "id"
IDENTIFIER_COLUMN = "identifier"


@dataclass(frozen=True)
class PokeapiAbilityIndex:
    """The English short effect of every PokeAPI ability that has one, by identifier, and
    every identifier of `abilities.csv`."""

    effect_texts: Mapping[str, str]
    identifiers: frozenset[str]


def build_pokeapi_index(store: SnapshotStore) -> PokeapiAbilityIndex:
    """Read the pinned snapshot into an index by identifier."""
    identifiers = {
        row[ID_COLUMN]: row[IDENTIFIER_COLUMN] for row in store.read_table(ABILITIES_FILE)
    }
    effect_texts = {
        identifiers[row["ability_id"]]: row["short_effect"]
        for row in store.read_table(PROSE_FILE)
        if row["local_language_id"] == ENGLISH_LANGUAGE_ID
        and row["short_effect"]
        and row["ability_id"] in identifiers
    }
    return PokeapiAbilityIndex(
        effect_texts=effect_texts, identifiers=frozenset(identifiers.values())
    )


def read_effect_text(index: PokeapiAbilityIndex, ability_id: str) -> str | None:
    """Return the English short effect of an ability, or `None` without a row or text."""
    return index.effect_texts.get(ability_id)
