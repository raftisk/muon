"""Answer artifacts built from a bundle of 2 live pkmn queries, with no model call.

The bundle holds an evolution chain with paths and a learnset query. The test cites every
node and table handle through the marker rewriter, in small chunks, and checks the graph
artifact against its model, the node cap and the size limit.

Needs the Neo4j of `.env` with the pkmn graph.
"""

import re

import pytest

from muon.agent.routing import build_scope
from muon.config import DEFAULT_ONTOLOGY_DIR, DEFAULT_UNIVERSES_DIR, get_settings
from muon.generation.artifacts import MAX_GRAPH_NODES, build_cited_answer
from muon.generation.contract import ArtifactEnvelope, GraphData, TableData
from muon.generation.handles import build_handle_table
from muon.generation.markers import MarkerRewriter
from muon.graph import Neo4jGraphClient
from muon.plugins import discover_universes
from muon.retrieval.cypher import CallDescription, run_checked_query
from muon.retrieval.evidence import EvidenceBundle, EvidenceStatus
from muon.retrieval.schema import SchemaIntrospectionError

pytestmark = pytest.mark.integration

PREFIX = "pkmn"
PLUGIN = discover_universes(DEFAULT_UNIVERSES_DIR, DEFAULT_ONTOLOGY_DIR).get(PREFIX)
QUESTION = "How does Squirtle evolve, and what does Pikachu learn by leveling up?"
CHUNK_SIZE = 7
MAX_GRAPH_JSON_BYTES = 64 * 1024
LEARNSET_LABEL = "LearnsetEntry"
LINK_URL_PATTERN = re.compile(r'\]\((\S+) "')
QUERIES = (
    (
        "the evolution chain from Squirtle to Blastoise",
        "MATCH p = (:PokemonSpecies {id: 'squirtle'})-[:EVOLVES_TO*]->"
        "(:PokemonSpecies {id: 'blastoise'}) RETURN p",
    ),
    (
        "the level-up learnset entries of Pikachu",
        "MATCH (s:PokemonSpecies {id: 'pikachu'})-[r:HAS_LEARNSET_ENTRY]->"
        "(e:LearnsetEntry {method: 'level'}) RETURN s, r, e",
    ),
)


async def test_a_live_bundle_gives_valid_artifacts(graph_client: Neo4jGraphClient) -> None:
    try:
        scope = await build_scope(graph_client, PLUGIN, get_settings())
    except SchemaIntrospectionError as error:
        pytest.skip(f"the {PREFIX} graph is not loaded: {error}")
    items = tuple(
        [
            await run_checked_query(
                CallDescription(f"e{number}", query, purpose, is_final=False),
                scope,
                graph_client,
            )
            for number, (purpose, query) in enumerate(QUERIES, start=1)
        ]
    )
    assert [item.status for item in items] == [EvidenceStatus.OK] * len(QUERIES)
    table = build_handle_table(EvidenceBundle(universe=PREFIX, question=QUESTION, items=items))
    markers = [f"[{handle}]" for handle in [*table.nodes, *table.tables]]
    text = " ".join(markers)

    rewriter = MarkerRewriter(table)
    pieces = [
        rewriter.feed(text[start : start + CHUNK_SIZE]) for start in range(0, len(text), CHUNK_SIZE)
    ]
    pieces.append(rewriter.finish())
    answer = build_cited_answer(
        "".join(pieces), rewriter.cited(), table, PLUGIN.manifest.attribution
    )

    assert answer.warnings == ()
    assert answer.graph is not None
    graph_json = answer.graph.to_json()
    assert len(graph_json.encode()) < MAX_GRAPH_JSON_BYTES
    graph = ArtifactEnvelope.model_validate_json(graph_json).artifact.data
    assert isinstance(graph, GraphData)
    assert len(graph.nodes) <= MAX_GRAPH_NODES
    learnset_nodes = [node for node in graph.nodes if node.label == LEARNSET_LABEL]
    assert learnset_nodes
    assert all({"level", "method"} <= set(node.properties) for node in learnset_nodes)
    source_urls = {source.url for source in answer.cited.sources}
    assert set(LINK_URL_PATTERN.findall(answer.text)) <= source_urls
    assert len(answer.tables) == len(items)
    for envelope, item in zip(answer.tables, items, strict=True):
        assert isinstance(envelope.artifact.data, TableData)
        assert len(envelope.artifact.data.rows) == item.row_count
