"""One live question through `ask()` against the graph in `.env` and the agent provider.

The test skips when `MUON_AGENT_*` is missing. It costs about 3 model calls.
"""

import pytest
from pydantic import ValidationError

from muon.agent.ask import AskRequest, ask
from muon.agent.routing import AgentRuntime
from muon.agent.universes import discover_universes
from muon.config import AgentSettings, AgentStep, get_settings
from muon.generation.answer import Claim
from muon.graph import Neo4jGraphClient
from muon.llm import create_chat_model
from muon.retrieval.evidence import EvidenceStatus, NodeKey

pytestmark = pytest.mark.network

QUESTION = "What type is Diglett?"
SPECIES_KEY = NodeKey(label="PokemonSpecies", id="diglett")
TYPE_KEY = NodeKey(label="Type", id="ground")


@pytest.fixture
def agent_settings() -> AgentSettings:
    try:
        return AgentSettings()
    except ValidationError as error:
        names = ", ".join(str(detail["loc"][0]) for detail in error.errors())
        pytest.skip(f"MUON_AGENT_* settings missing: {names}")


def find_claim_citing(claims: list[Claim], key: NodeKey) -> Claim | None:
    return next((claim for claim in claims if key in claim.node_keys), None)


async def test_live_ask_cites_the_species_and_its_type(agent_settings: AgentSettings) -> None:
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

        result = await ask(AskRequest(question=QUESTION), registry, settings, runtime)

    assert result.answer.universe == "pkmn"
    claim = find_claim_citing(result.answer.claims, SPECIES_KEY)
    assert claim is not None
    assert TYPE_KEY in claim.node_keys
    cited = [result.bundle.get(evidence_id) for evidence_id in claim.evidence_ids]
    assert all(item is not None and item.status is EvidenceStatus.OK for item in cited)
    assert result.answer.warnings == []
