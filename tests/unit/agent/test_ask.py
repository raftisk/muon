from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import SecretStr

from muon.agent.ask import AskRequest, ask
from muon.agent.routing import AgentRuntime, RoutingError
from muon.agent.universes import UniverseRegistry, discover_universes
from muon.config import AgentSettings, Settings
from muon.generation.answer import AnswerDraft, Claim
from muon.retrieval.context import READ_GRAPH_COUNTS
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.fixtures.toy_plugin import write_toy_plugin
from tests.unit.chat_fakes import ScriptedChatModel, build_final_message, build_tool_call_message
from tests.unit.retrieval.fakes import QueryGraphClient
from tests.unit.retrieval.scope import TOY_COUNTS, build_toy_settings
from tests.unit.retrieval.toy_graph import build_toy_answers

WIDGET_QUERY = "MATCH (w:ToyWidget:muontoy) RETURN w.name AS name"
QUESTION = "What are the widgets?"


def build_draft() -> AnswerDraft:
    claim = Claim(id="c1", text="W1 exists", evidence_ids=["e1"], node_keys=[])
    return AnswerDraft(
        text="W1 exists [c1].", claims=[claim], assumptions=[], table_evidence_ids=[]
    )


@dataclass
class Pipeline:
    client: QueryGraphClient
    model: ScriptedChatModel
    runtime: AgentRuntime
    registry: UniverseRegistry
    settings: Settings


def build_pipeline(tmp_path: Path) -> Pipeline:
    ontology_dir = copy_toy_ontology(tmp_path / "ontology")
    universes_dir = tmp_path / "universes"
    universes_dir.mkdir()
    write_toy_plugin(universes_dir)
    answers = build_toy_answers()
    answers[READ_GRAPH_COUNTS] = lambda parameters: [TOY_COUNTS]
    answers[WIDGET_QUERY] = lambda parameters: [{"name": "W1"}]
    client = QueryGraphClient(answers)
    model = ScriptedChatModel(
        replies=[build_tool_call_message(WIDGET_QUERY, "c1"), build_final_message()],
        drafts=[build_draft()],
    )
    agent_settings = AgentSettings(
        _env_file=None, base_url="https://a.test/v1", api_key=SecretStr("k"), model="m"
    )
    return Pipeline(
        client=client,
        model=model,
        runtime=AgentRuntime(client=client, chat_model=model, agent_settings=agent_settings),
        registry=discover_universes(universes_dir, ontology_dir),
        settings=build_toy_settings(tmp_path, ontology_dir),
    )


async def test_the_only_universe_answers_and_everything_carries_it(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    result = await ask(
        AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, pipeline.runtime
    )

    assert result.answer.universe == TOY_UNIVERSE
    assert result.bundle.universe == TOY_UNIVERSE
    assert {item.universe for item in result.bundle.items} == {TOY_UNIVERSE}
    assert result.answer.warnings == []


async def test_stats_hold_the_retrieve_and_generate_steps(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    result = await ask(
        AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, pipeline.runtime
    )

    assert [step.step for step in result.stats.steps] == ["retrieve", "generate"]
    assert (result.stats.tool_calls, result.stats.cypher_calls) == (1, 1)
    assert result.stats.total_latency_seconds >= 0


async def test_an_unknown_universe_raises_before_any_read_or_model_call(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    with pytest.raises(RoutingError, match="zelda"):
        await ask(
            AskRequest(question=QUESTION, universe="zelda"),
            pipeline.registry,
            pipeline.settings,
            pipeline.runtime,
        )

    assert pipeline.client.list_queries() == []
    assert pipeline.model.calls == []
