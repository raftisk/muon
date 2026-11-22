from contextlib import AbstractContextManager, aclosing, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
from pydantic import SecretStr

from muon.agent import ask as ask_module
from muon.agent.ask import (
    ANSWER_STEP,
    QUERY_STEP,
    RESOLVE_PROGRESS_STEP,
    SCHEMA_STEP,
    STEP_MESSAGES,
    AskRequest,
    ask,
)
from muon.agent.events import (
    AnswerFinished,
    AskEvent,
    AskResult,
    StepFinished,
    StepStarted,
    TextDelta,
    collect_answer,
)
from muon.agent.routing import AgentRuntime, RoutingError
from muon.config import AgentSettings, Settings
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


TEXT_CHUNKS = ["W1 exists [t", "1] and more", " text."]
TABLE_TEXT = "W1 exists and more text."


@dataclass
class RecordingTracker:
    """Collects `(name, event)` pairs; an exception leaving a step records `failed`."""

    events: list[tuple[str, str]] = field(default_factory=list)

    def step(self, name: str) -> AbstractContextManager[None]:
        return self.track(name)

    @contextmanager
    def track(self, name: str) -> Any:
        self.events.append((name, "start"))
        try:
            yield
        except BaseException:
            self.events.append((name, "failed"))
            raise
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
        drafts=[NameMentions(names=[])],
        streams=[TEXT_CHUNKS],
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


async def read_events(
    pipeline: Pipeline, runtime: AgentRuntime | None = None, universe: str | None = None
) -> list[AskEvent]:
    stream = ask(
        AskRequest(question=QUESTION, universe=universe),
        pipeline.registry,
        pipeline.settings,
        runtime or pipeline.runtime,
    )
    return [event async for event in stream]


async def read_result(pipeline: Pipeline, runtime: AgentRuntime | None = None) -> AskResult:
    events = ask(
        AskRequest(question=QUESTION),
        pipeline.registry,
        pipeline.settings,
        runtime or pipeline.runtime,
    )
    return await collect_answer(events)


async def test_the_only_universe_answers_and_everything_carries_it(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    result = await read_result(pipeline)

    assert result.universe == TOY_UNIVERSE
    assert result.bundle.universe == TOY_UNIVERSE
    assert {item.universe for item in result.bundle.items} == {TOY_UNIVERSE}
    assert result.answer.warnings == ()


async def test_the_events_come_in_order_with_the_deltas_inside_the_answer_step(
    tmp_path: Path,
) -> None:
    pipeline = build_pipeline(tmp_path)

    events = await read_events(pipeline)

    kinds = [
        (event.kind, event.name) if isinstance(event, StepStarted | StepFinished) else event.kind
        for event in events
    ]
    steps = [SCHEMA_STEP, RESOLVE_PROGRESS_STEP, QUERY_STEP]
    expected: list[str | tuple[str, str]] = []
    for step in steps:
        expected.extend([("step_started", step), ("step_finished", step)])
    expected.append(("step_started", ANSWER_STEP))
    expected.extend(["text_delta"] * sum(isinstance(event, TextDelta) for event in events))
    expected.extend([("step_finished", ANSWER_STEP), "answer_finished"])
    assert kinds == expected
    assert sum(isinstance(event, TextDelta) for event in events) >= 2


async def test_the_deltas_join_to_the_answer_text_and_arrive_before_the_stream_ends(
    tmp_path: Path,
) -> None:
    pipeline = build_pipeline(tmp_path)
    progress_at_delta: list[int] = []
    deltas: list[str] = []
    result: AskResult | None = None

    async for event in ask(
        AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, pipeline.runtime
    ):
        if isinstance(event, TextDelta):
            deltas.append(event.text)
            progress_at_delta.append(len(pipeline.model.stream_progress))
            assert "[" not in event.text
        elif isinstance(event, AnswerFinished):
            result = event.result

    assert result is not None
    assert "".join(deltas) == result.answer.text == TABLE_TEXT
    assert progress_at_delta[0] < len(TEXT_CHUNKS)


async def test_a_table_handle_gives_the_table_through_ask(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    result = await read_result(pipeline)

    assert result.answer.cited.table_ids == ("e1",)
    assert len(result.answer.tables) == 1
    assert "[t1]" not in result.answer.text


async def test_an_unknown_handle_is_dropped_and_warned_through_ask(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    model = pipeline.model.model_copy(update={"streams": [["W1 exists [n99]."]]})
    runtime = replace(pipeline.runtime, generate_model=model)

    result = await read_result(pipeline, runtime)

    assert "[n99]" not in result.answer.text
    assert any("[n99]" in warning for warning in result.answer.warnings)


async def test_each_step_calls_only_its_own_model(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    resolve_model = ScriptedChatModel(drafts=[NameMentions(names=[])])
    retrieve_model = ScriptedChatModel(
        replies=[build_tool_call_message(WIDGET_QUERY, "c1"), build_final_message()]
    )
    generate_model = ScriptedChatModel(streams=[TEXT_CHUNKS])
    runtime = replace(
        pipeline.runtime,
        resolve_model=resolve_model,
        retrieve_model=retrieve_model,
        generate_model=generate_model,
    )

    await read_result(pipeline, runtime)

    assert len(resolve_model.calls) == 1
    assert len(retrieve_model.calls) == 2
    assert len(generate_model.calls) == 1


async def test_stats_hold_the_resolve_retrieve_and_generate_steps(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    result = await read_result(pipeline)

    assert [step.step for step in result.stats.steps] == ["resolve", "retrieve", "generate"]
    assert (result.stats.tool_calls, result.stats.cypher_calls) == (1, 1)
    assert result.stats.total_latency_seconds >= 0


async def test_an_unknown_universe_raises_before_any_event_read_or_model_call(
    tmp_path: Path,
) -> None:
    pipeline = build_pipeline(tmp_path)
    events: list[AskEvent] = []

    with pytest.raises(RoutingError, match="zelda"):
        async for event in ask(
            AskRequest(question=QUESTION, universe="zelda"),
            pipeline.registry,
            pipeline.settings,
            pipeline.runtime,
        ):
            events.append(event)

    assert events == []
    assert pipeline.client.list_queries() == []
    assert pipeline.model.calls == []


async def test_collect_answer_shows_the_4_steps_in_order(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    tracker = RecordingTracker()
    events = ask(
        AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, pipeline.runtime
    )

    await collect_answer(events, tracker)

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


async def test_a_retrieve_error_marks_the_query_step_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = build_pipeline(tmp_path)
    tracker = RecordingTracker()
    error = ModelCallError("retrieve", "m", "rate limited")

    async def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(ask_module, "retrieve", fail)
    events = ask(
        AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, pipeline.runtime
    )

    with pytest.raises(ModelCallError) as raised:
        await collect_answer(events, tracker)

    assert raised.value is error
    assert tracker.events[-2:] == [(QUERY_STEP, "start"), (QUERY_STEP, "failed")]


async def test_a_generate_error_arrives_after_the_deltas_and_leaves_the_step_open(
    tmp_path: Path,
) -> None:
    pipeline = build_pipeline(tmp_path)
    request = httpx2.Request("POST", "https://a.test/v1/chat/completions")
    model = pipeline.model.model_copy(
        update={
            "streams": [TEXT_CHUNKS],
            "stream_failure": openai.APIConnectionError(request=request),
            "fail_after": 2,
        }
    )
    runtime = replace(pipeline.runtime, generate_model=model)
    events: list[AskEvent] = []

    with pytest.raises(ModelCallError) as raised:
        async for event in ask(
            AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, runtime
        ):
            events.append(event)

    assert raised.value.step == "generate"
    assert sum(isinstance(event, TextDelta) for event in events) <= 2
    assert StepStarted(name=ANSWER_STEP) in events
    assert StepFinished(name=ANSWER_STEP) not in events


async def test_closing_ask_after_the_first_delta_closes_the_model_stream(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    stream = ask(
        AskRequest(question=QUESTION), pipeline.registry, pipeline.settings, pipeline.runtime
    )

    async with aclosing(stream):
        async for event in stream:
            if isinstance(event, TextDelta):
                break

    assert pipeline.model.is_stream_closed


async def test_resolved_names_are_recorded_and_reach_the_retrieve_message(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    mention = NameMention(mention="Widget One", label="ToyWidget")
    model = ScriptedChatModel(
        replies=[build_tool_call_message(WIDGET_QUERY, "c1"), build_final_message()],
        drafts=[NameMentions(names=[mention])],
        streams=[TEXT_CHUNKS],
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

    result = await read_result(pipeline, runtime)

    assert [item.mention for item in result.resolved_names] == ["Widget One"]
    assert result.resolved_names[0].candidates[0].key.id == "w1"
    retrieve_human = model.calls[1][1].content
    assert "W1 [ToyWidget w1, score 6.3]" in str(retrieve_human)
    assert str(retrieve_human).endswith(f"Question: {QUESTION}")


async def test_a_question_without_names_runs_no_lookup_and_sends_the_bare_question(
    tmp_path: Path,
) -> None:
    pipeline = build_pipeline(tmp_path)

    result = await read_result(pipeline)

    assert result.resolved_names == []
    assert LOOK_UP_NAMES not in pipeline.client.list_queries()
    assert pipeline.model.calls[1][1].content == QUESTION


def test_step_messages_cover_the_four_steps() -> None:
    assert set(STEP_MESSAGES) == {SCHEMA_STEP, RESOLVE_PROGRESS_STEP, QUERY_STEP, ANSWER_STEP}
    for messages in STEP_MESSAGES.values():
        assert messages.running and messages.completed and messages.failed
