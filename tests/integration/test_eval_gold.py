"""Every gold query of the pkmn eval sets runs on the live graph and returns nodes.

A question with `has_empty_gold` returns none.

Needs the Neo4j of `.env` with the pkmn graph. It calls no model.
"""

import pytest

from muon.agent.routing import build_scope
from muon.config import DEFAULT_ONTOLOGY_DIR, DEFAULT_UNIVERSES_DIR, get_settings
from muon.evaluation.eval_scores import resolve_gold
from muon.evaluation.questions import EvalQuestion, load_eval_sets
from muon.graph import Neo4jGraphClient
from muon.plugins import discover_universes
from muon.retrieval.schema import SchemaIntrospectionError

pytestmark = pytest.mark.integration

PREFIX = "pkmn"
PLUGIN = discover_universes(DEFAULT_UNIVERSES_DIR, DEFAULT_ONTOLOGY_DIR).get(PREFIX)
QUESTIONS = [question for eval_set in load_eval_sets(PLUGIN, []) for question in eval_set.questions]


@pytest.mark.parametrize("question", QUESTIONS, ids=[question.id for question in QUESTIONS])
async def test_the_gold_query_returns_the_expected_nodes(
    question: EvalQuestion, graph_client: Neo4jGraphClient
) -> None:
    try:
        scope = await build_scope(graph_client, PLUGIN, get_settings())
    except SchemaIntrospectionError as error:
        pytest.skip(f"the {PREFIX} graph is not loaded: {error}")

    gold = await resolve_gold(question, scope, graph_client)

    assert bool(gold) != question.has_empty_gold
