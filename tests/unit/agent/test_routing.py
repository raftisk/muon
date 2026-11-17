from pathlib import Path

import pytest
from pydantic import SecretStr

from muon.agent.routing import RoutingError, build_scope, select_universe
from muon.config import Neo4jSettings, OntologySettings, Settings, UniverseSettings
from muon.plugins import UniverseRegistry, discover_universes
from muon.retrieval.context import READ_GRAPH_COUNTS
from tests.fixtures.toy_ontology import (
    ALT_UNIVERSE,
    TOY_UNIVERSE,
    add_alt_universe_t_box,
    copy_toy_ontology,
)
from tests.fixtures.toy_plugin import write_toy_plugin
from tests.unit.retrieval.fakes import QueryGraphClient
from tests.unit.retrieval.toy_graph import build_toy_answers


@pytest.fixture
def ontology_dir(tmp_path: Path) -> Path:
    ontology_dir = copy_toy_ontology(tmp_path / "ontology")
    add_alt_universe_t_box(ontology_dir)
    return ontology_dir


def build_registry(tmp_path: Path, ontology_dir: Path, prefixes: list[str]) -> UniverseRegistry:
    universes_dir = tmp_path / "universes"
    universes_dir.mkdir()
    for prefix in prefixes:
        write_toy_plugin(universes_dir, prefix=prefix)
    return discover_universes(universes_dir, ontology_dir)


def test_only_plugin_is_selected_without_a_request(tmp_path: Path, ontology_dir: Path) -> None:
    registry = build_registry(tmp_path, ontology_dir, [TOY_UNIVERSE])

    assert select_universe(registry, None).prefix == TOY_UNIVERSE


def test_requested_plugin_is_selected(tmp_path: Path, ontology_dir: Path) -> None:
    registry = build_registry(tmp_path, ontology_dir, [TOY_UNIVERSE, ALT_UNIVERSE])

    assert select_universe(registry, ALT_UNIVERSE).prefix == ALT_UNIVERSE


def test_unknown_prefix_names_the_known_ones(tmp_path: Path, ontology_dir: Path) -> None:
    registry = build_registry(tmp_path, ontology_dir, [TOY_UNIVERSE])

    with pytest.raises(RoutingError, match=f"'zelda'.*{TOY_UNIVERSE}"):
        select_universe(registry, "zelda")


def test_two_plugins_and_no_request_raise(tmp_path: Path, ontology_dir: Path) -> None:
    registry = build_registry(tmp_path, ontology_dir, [TOY_UNIVERSE, ALT_UNIVERSE])

    with pytest.raises(RoutingError) as error_info:
        select_universe(registry, None)

    message = str(error_info.value)
    assert TOY_UNIVERSE in message
    assert ALT_UNIVERSE in message
    assert "--universe" in message


async def test_scope_allows_labels_types_and_the_universe_label(
    tmp_path: Path, ontology_dir: Path
) -> None:
    registry = build_registry(tmp_path, ontology_dir, [TOY_UNIVERSE])
    settings = Settings(
        neo4j=Neo4jSettings(uri="neo4j://x", username="u", password=SecretStr("p")),
        ontology=OntologySettings(ontology_dir=ontology_dir),
        universes=UniverseSettings(schema_cache_dir=tmp_path / "cache"),
    )
    answers = build_toy_answers()
    answers[READ_GRAPH_COUNTS] = lambda parameters: [
        {"nodes": [{"count": 10}], "relationships": [{"count": 5, "type": "HAS_COLOR"}]}
    ]

    scope = await build_scope(QueryGraphClient(answers), registry.get(TOY_UNIVERSE), settings)

    assert scope.universe_label == TOY_UNIVERSE
    assert {TOY_UNIVERSE, "ToyWidget", "Entity", "HAS_COLOR", "PAIRS_WITH"} <= scope.allowed_terms
