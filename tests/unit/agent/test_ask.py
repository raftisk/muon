from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from muon.agent import ask as ask_module
from muon.agent.ask import (
    ANSWER_STEP,
    QUERY_STEP,
    RESOLVE_PROGRESS_STEP,
    SCHEMA_STEP,
    AskRequest,
    ask,
)
from muon.agent.routing import NO_PROGRESS, AgentRuntime, RoutingError
from muon.config import AgentSettings, Settings
from muon.generation.answer import AnswerDraft, Claim
from muon.graph import GraphNode
from muon.llm.client import ModelCallError
from muon.plugins import UniverseRegistry, discover_universes
from muon.retrieval.context import READ_GRAPH_COUNTS
from muon.retrieval.resolve import LOOK_UP_NAMES, NameMention, NameMentions
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.fixtures.toy_plugin import write_toy_plugin
from tests.unit.chat_fakes import ScriptedChatModel, build_final_message, build_tool_call_message
from tests.unit.retrieval.fakes import QueryGraphClient
from tests.unit.retrieval.scope import TOY_COUNTS, build_toy_settings
from tests.unit.retrieval.toy_graph import WIDGET_SET, build_toy_answers

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
        drafts=[NameMentions(names=[]), build_draft()],
    )
    agent_settings = AgentSettings(
        _env_file=None, base_url="https://a.test/v1", api_key=SecretStr("k"), default_model="m"
    )
    return Pipeline(
        client=client,
        model=model,
        runtime=AgentRuntime(
            client=client,
            resolve_model=model,
            retrieve_model=model,
            generate_model=model,
            agent_settings=agent_settings,
        ),
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


async def test_each_step_calls_only_its_own_model(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    resolve_model = ScriptedChatModel(drafts=[NameMentions(names=[])])
    retrieve_model = ScriptedChatModel(
        replies=[build_tool_call_message(WIDGET_QUERY, "c1"), build_final_message()]
    )
    generate_model = ScriptedChatModel(drafts=[build_draft()])
    runtime = replace(
        pipeline.runtime,
        resolve_model=resolve_model,
        retrieve_model=retrieve_model,
        generate_model=generate_model,
    )

    await ask(AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, runtime)

    assert len(resolve_model.calls) == 1
    assert len(retrieve_model.calls) == 2
    assert len(generate_model.calls) == 1


async def test_stats_hold_the_resolve_retrieve_and_generate_steps(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    result = await ask(
        AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, pipeline.runtime
    )

    assert [step.step for step in result.stats.steps] == ["resolve", "retrieve", "generate"]
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


async def test_ask_records_the_4_steps_in_order(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    tracker = RecordingTracker()
    runtime = replace(pipeline.runtime, progress=tracker)

    await ask(AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, runtime)

    assert tracker.events == [
        (SCHEMA_STEP, "start"),
        (SCHEMA_STEP, "end"),
        (RESOLVE_PROGRESS_STEP, "start"),
        (RESOLVE_PROGRESS_STEP, "end"),
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


async def test_resolved_names_are_recorded_and_reach_the_retrieve_message(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    mention = NameMention(mention="Widget One", label="ToyWidget")
    model = ScriptedChatModel(
        replies=[build_tool_call_message(WIDGET_QUERY, "c1"), build_final_message()],
        drafts=[NameMentions(names=[mention]), build_draft()],
    )
    row = {
        "position": 0,
        "node": GraphNode("el-w1", frozenset(WIDGET_SET), {"id": "w1", "name": "W1"}),
        "score": 6.3,
    }
    pipeline.client.answers[LOOK_UP_NAMES] = lambda parameters: [row]
    runtime = replace(
        pipeline.runtime, resolve_model=model, retrieve_model=model, generate_model=model
    )

    result = await ask(AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, runtime)

    assert [item.mention for item in result.resolved_names] == ["Widget One"]
    assert result.resolved_names[0].candidates[0].key.id == "w1"
    retrieve_human = model.calls[1][1].content
    assert "W1 [ToyWidget w1, score 6.3]" in str(retrieve_human)
    assert str(retrieve_human).endswith(f"Question: {QUESTION}")


async def test_a_question_without_names_runs_no_lookup_and_sends_the_bare_question(
    tmp_path: Path,
) -> None:
    pipeline = build_pipeline(tmp_path)

    result = await ask(
        AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, pipeline.runtime
    )

    assert result.resolved_names == []
    assert LOOK_UP_NAMES not in pipeline.client.list_queries()
    assert pipeline.model.calls[1][1].content == QUESTION
