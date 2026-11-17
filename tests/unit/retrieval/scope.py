"""A `UniverseScope` of the toy universe, built from canned schema reads."""

from pathlib import Path

from pydantic import SecretStr

from muon.agent.routing import UniverseScope, build_scope
from muon.config import Neo4jSettings, OntologySettings, Settings, UniverseSettings
from muon.plugins import discover_universes
from muon.retrieval.context import READ_GRAPH_COUNTS
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.fixtures.toy_plugin import write_toy_plugin
from tests.unit.retrieval.fakes import QueryGraphClient
from tests.unit.retrieval.toy_graph import build_toy_answers

TOY_COUNTS = {"nodes": [{"count": 10}], "relationships": [{"count": 5, "type": "HAS_COLOR"}]}


def build_toy_settings(root: Path, ontology_dir: Path) -> Settings:
    return Settings(
        neo4j=Neo4jSettings(uri="neo4j://x", username="u", password=SecretStr("p")),
        ontology=OntologySettings(ontology_dir=ontology_dir),
        universes=UniverseSettings(schema_cache_dir=root / "cache"),
    )


async def build_toy_scope(root: Path) -> UniverseScope:
    ontology_dir = copy_toy_ontology(root / "ontology")
    universes_dir = root / "universes"
    universes_dir.mkdir()
    write_toy_plugin(universes_dir)
    plugin = discover_universes(universes_dir, ontology_dir).get(TOY_UNIVERSE)
    answers = build_toy_answers()
    answers[READ_GRAPH_COUNTS] = lambda parameters: [TOY_COUNTS]
    return await build_scope(
        QueryGraphClient(answers), plugin, build_toy_settings(root, ontology_dir)
    )
