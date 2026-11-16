from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from muon.agent import ask as ask_module
from muon.agent.ask import ANSWER_STEP, QUERY_STEP, SCHEMA_STEP, AskRequest, ask
from muon.agent.routing import NO_PROGRESS, AgentRuntime, RoutingError
from muon.agent.universes import UniverseRegistry, discover_universes
from muon.config import AgentSettings, Settings
from muon.generation.answer import AnswerDraft, Claim
from muon.llm.client import ModelCallError
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
class RecordingTracker:
    """Collects `(name, "start")` and `(name, "end")` pairs in the order they happen."""

    events: list[tuple[str, str]] = field(default_factory=list)

    def step(self, name: str) -> AbstractContextManager[None]:
        return self.track(name)

    @contextmanager
    def track(self, name: str) -> Any:
        self.events.append((name, "start"))
        try:
            yield
        finally:
            self.events.append((name, "end"))


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


def test_the_runtime_defaults_to_no_progress(tmp_path: Path) -> None:
    assert build_pipeline(tmp_path).runtime.progress is NO_PROGRESS


async def test_ask_records_the_3_steps_in_order(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    tracker = RecordingTracker()
    runtime = replace(pipeline.runtime, progress=tracker)

    await ask(AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, runtime)

    assert tracker.events == [
        (SCHEMA_STEP, "start"),
        (SCHEMA_STEP, "end"),
        (QUERY_STEP, "start"),
        (QUERY_STEP, "end"),
        (ANSWER_STEP, "start"),
        (ANSWER_STEP, "end"),
    ]


async def test_a_retrieve_error_leaves_the_query_step_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = build_pipeline(tmp_path)
    tracker = RecordingTracker()
    runtime = replace(pipeline.runtime, progress=tracker)
    error = ModelCallError("retrieve", "m", "rate limited")

    async def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(ask_module, "retrieve", fail)

    with pytest.raises(ModelCallError) as raised:
        await ask(AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, runtime)

    assert raised.value is error
    assert tracker.events[-2:] == [(QUERY_STEP, "start"), (QUERY_STEP, "end")]
