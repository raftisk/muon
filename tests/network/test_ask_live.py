"""One live question through `ask()` against the graph in `.env` and the agent provider.

The test skips when `MUON_AGENT_*` is missing. It costs about 3 model calls.
"""

import time

import pytest
from pydantic import ValidationError

from muon.agent.ask import AskRequest, ask
from muon.agent.events import AnswerFinished, TextDelta, collect_answer
from muon.agent.routing import AgentRuntime
from muon.config import AgentSettings, AgentStep, get_settings
from muon.graph import Neo4jGraphClient
from muon.llm import create_chat_model
from muon.plugins import discover_universes
from muon.retrieval.evidence import NodeKey

# LangChain caches 1 HTTP client for all chat models, so the tests share 1 event loop.
pytestmark = [pytest.mark.network, pytest.mark.asyncio(loop_scope="module")]

QUESTION = "What type is Diglett?"
SPECIES_KEY = NodeKey(label="PokemonSpecies", id="diglett")
MULTI_HOP_QUESTION = "which Water-type Pokémon evolve from a Pokémon that learns Surf?"
MIN_DELTAS = 2


@pytest.fixture
def agent_settings() -> AgentSettings:
    try:
        return AgentSettings()
    except ValidationError as error:
        names = ", ".join(str(detail["loc"][0]) for detail in error.errors())
        pytest.skip(f"MUON_AGENT_* settings missing: {names}")


def build_runtime(client: Neo4jGraphClient, agent_settings: AgentSettings) -> AgentRuntime:
    return AgentRuntime(
        client,
        *[create_chat_model(agent_settings, agent_settings.model_for(step)) for step in AgentStep],
        agent_settings,
    )


async def test_live_ask_cites_the_species(agent_settings: AgentSettings) -> None:
    settings = get_settings()
    registry = discover_universes(settings.universes.universes_dir, settings.ontology.ontology_dir)
    async with Neo4jGraphClient(settings.neo4j) as client:
        runtime = build_runtime(client, agent_settings)

        events = ask(AskRequest(question=QUESTION), registry, settings, runtime)
        result = await collect_answer(events)

    assert result.universe == "pkmn"
    assert SPECIES_KEY in result.answer.cited.node_keys
    assert result.answer.warnings == ()


async def test_the_answer_streams_before_it_ends(agent_settings: AgentSettings) -> None:
    settings = get_settings()
    registry = discover_universes(settings.universes.universes_dir, settings.ontology.ontology_dir)
    delta_times: list[float] = []
    result = None
    async with Neo4jGraphClient(settings.neo4j) as client:
        runtime = build_runtime(client, agent_settings)
        started_at = time.perf_counter()

        async for event in ask(
            AskRequest(question=MULTI_HOP_QUESTION), registry, settings, runtime
        ):
            if isinstance(event, TextDelta):
                delta_times.append(time.perf_counter() - started_at)
            elif isinstance(event, AnswerFinished):
                result = event.result

    print(f"first delta {delta_times[0]:.2f} s, last delta {delta_times[-1]:.2f} s")
    assert len(delta_times) >= MIN_DELTAS
    assert delta_times[0] < delta_times[-1]
    assert result is not None
    assert result.answer.cited.table_ids or result.answer.cited.node_keys
