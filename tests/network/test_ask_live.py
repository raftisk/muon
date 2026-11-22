"""One live question through `ask()` against the graph in `.env` and the agent provider.

The test skips when `MUON_AGENT_*` is missing. It costs about 3 model calls.
"""

import pytest
from pydantic import ValidationError

from muon.agent.ask import AskRequest, ask
from muon.agent.events import collect_answer
from muon.agent.routing import AgentRuntime
from muon.config import AgentSettings, AgentStep, get_settings
from muon.graph import Neo4jGraphClient
from muon.llm import create_chat_model
from muon.plugins import discover_universes
from muon.retrieval.evidence import NodeKey

pytestmark = pytest.mark.network

QUESTION = "What type is Diglett?"
SPECIES_KEY = NodeKey(label="PokemonSpecies", id="diglett")


@pytest.fixture
def agent_settings() -> AgentSettings:
    try:
        return AgentSettings()
    except ValidationError as error:
        names = ", ".join(str(detail["loc"][0]) for detail in error.errors())
        pytest.skip(f"MUON_AGENT_* settings missing: {names}")


async def test_live_ask_cites_the_species(agent_settings: AgentSettings) -> None:
    settings = get_settings()
    registry = discover_universes(settings.universes.universes_dir, settings.ontology.ontology_dir)
    async with Neo4jGraphClient(settings.neo4j) as client:
        runtime = AgentRuntime(
            client,
            *[
                create_chat_model(agent_settings, agent_settings.model_for(step))
                for step in AgentStep
            ],
            agent_settings,
        )

        events = ask(AskRequest(question=QUESTION), registry, settings, runtime)
        result = await collect_answer(events)

    assert result.universe == "pkmn"
    assert SPECIES_KEY in result.answer.cited.node_keys
    assert result.answer.warnings == ()
