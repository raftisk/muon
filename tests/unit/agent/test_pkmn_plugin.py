import re

from muon.agent.universes import discover_universes
from muon.config import DEFAULT_ONTOLOGY_DIR, DEFAULT_UNIVERSES_DIR

EXPECTED_HEADINGS = [
    "Node keys",
    "Stubs",
    "Forms",
    "Edge properties",
    "Evolutions and form changes",
    "Games",
    "Learnsets",
    "Encounters",
    "Type chart",
    "Text",
    "Gaps",
]
HEADING_PATTERN = re.compile(r"^### (.+)$", re.MULTILINE)


def test_discovery_lists_the_pkmn_plugin() -> None:
    registry = discover_universes(DEFAULT_UNIVERSES_DIR, DEFAULT_ONTOLOGY_DIR)

    assert registry.list_prefixes() == ("pkmn",)
    assert registry.get("pkmn").name == "Pokémon"


def test_ontology_paths_end_with_the_pkmn_t_box() -> None:
    plugin = discover_universes(DEFAULT_UNIVERSES_DIR, DEFAULT_ONTOLOGY_DIR).get("pkmn")

    assert plugin.ontology_paths[-1] == DEFAULT_ONTOLOGY_DIR / "universes" / "pkmn.ttl"


def test_notes_hold_the_topic_headings_in_order() -> None:
    plugin = discover_universes(DEFAULT_UNIVERSES_DIR, DEFAULT_ONTOLOGY_DIR).get("pkmn")

    assert HEADING_PATTERN.findall(plugin.notes_text) == EXPECTED_HEADINGS
