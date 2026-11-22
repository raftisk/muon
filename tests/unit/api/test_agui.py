import asyncio
import json
from collections.abc import AsyncGenerator, Sequence

import pytest
from ag_ui.core import (
    ActivityMessage,
    AssistantMessage,
    BaseEvent,
    EventType,
    RunAgentInput,
    ToolMessage,
    UserMessage,
)

from muon.agent.ask import ANSWER_STEP, QUERY_STEP, RESOLVE_PROGRESS_STEP, SCHEMA_STEP, AskRequest
from muon.agent.events import (
    AnswerFinished,
    AskEvent,
    AskResult,
    StepFinished,
    StepStarted,
    TextDelta,
)
from muon.api.agui import StepBoard, stream_run
from muon.api.store import DEFAULT_THREAD_TITLE, ThreadStore
from muon.generation.contract import (
    GRAPH_ARTIFACT_TYPE,
    TABLE_ARTIFACT_TYPE,
    CitedAnswer,
)
from tests.unit.evaluation.fakes import build_ask_result
from tests.unit.generation.test_contract import build_envelope, build_graph_data, build_table_data

QUESTION = "  What does the widget do?  "
STEP_NAMES = (SCHEMA_STEP, RESOLVE_PROGRESS_STEP, QUERY_STEP, ANSWER_STEP)
TEXT_PARTS = ("Alpha ", "is ", "a widget.")


def build_result(has_artifacts: bool) -> AskResult:
    base = build_ask_result()
    if not has_artifacts:
        return base
    answer = CitedAnswer(
        text="Alpha is a widget.",
        sources=(),
        graph=build_envelope(build_graph_data(), GRAPH_ARTIFACT_TYPE),
        tables=(build_envelope(build_table_data(), TABLE_ARTIFACT_TYPE),),
        cited=base.answer.cited,
        warnings=(),
    )
    return base.model_copy(update={"answer": answer})


class FakeStream:
    """Plays scripted events, optionally raising after them, and records that it was closed."""

    def __init__(self, events: Sequence[AskEvent], error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.is_closed = False

    def __call__(self, request: AskRequest) -> AsyncGenerator[AskEvent]:
        return self.play()

    async def play(self) -> AsyncGenerator[AskEvent]:
        try:
            for event in self.events:
                yield event
            if self.error is not None:
                raise self.error
        finally:
            self.is_closed = True


def build_events(result: AskResult | None) -> list[AskEvent]:
    events: list[AskEvent] = []
    for name in STEP_NAMES[:-1]:
        events += [StepStarted(name=name), StepFinished(name=name)]
    events.append(StepStarted(name=ANSWER_STEP))
    events += [TextDelta(text=part) for part in TEXT_PARTS]
    events.append(StepFinished(name=ANSWER_STEP))
    if result is not None:
        events.append(AnswerFinished(result=result))
    return events


def build_run_input(thread_id: str) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id="run_1",
        state={},
        messages=[UserMessage(id="u1", content=QUESTION)],
        tools=[],
        context=[],
        forwarded_props={},
    )


def build_thread() -> tuple[ThreadStore, str]:
    store = ThreadStore()
    return store, store.create_thread().id


async def collect(generator: AsyncGenerator[BaseEvent]) -> list[BaseEvent]:
    return [event async for event in generator]


def event_types(events: Sequence[BaseEvent]) -> list[EventType]:
    return [event.type for event in events]


async def test_a_run_with_artifacts_streams_the_full_event_order_and_stores_the_turn() -> None:
    store, thread_id = build_thread()
    stream = FakeStream(build_events(build_result(has_artifacts=True)))

    events = await collect(stream_run(build_run_input(thread_id), QUESTION, stream, store))

    types = event_types(events)
    snapshot = EventType.ACTIVITY_SNAPSHOT
    assert types == [
        EventType.RUN_STARTED,
        *[snapshot] * 7,
        EventType.TEXT_MESSAGE_START,
        *[EventType.TEXT_MESSAGE_CONTENT] * len(TEXT_PARTS),
        snapshot,
        EventType.TEXT_MESSAGE_END,
        *[
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_ARGS,
            EventType.TOOL_CALL_END,
            EventType.TOOL_CALL_RESULT,
        ]
        * 2,
        EventType.RUN_FINISHED,
    ]
    snapshots = [event for event in events if event.type == snapshot]
    assert len({event.message_id for event in snapshots}) == 1  # type: ignore[attr-defined]
    assert [s["status"] for s in snapshots[-1].content["steps"]] == ["completed"] * 4  # type: ignore[attr-defined]
    stored = store.list_messages(thread_id)
    assert [type(message) for message in stored] == [
        AssistantMessage,
        AssistantMessage,
        ToolMessage,
        AssistantMessage,
        ToolMessage,
        ActivityMessage,
    ]
    assert stored[0].content == "Alpha is a widget."
    assert stored[1].tool_calls[0].function.name == "show_graph"  # type: ignore[attr-defined]
    assert stored[3].tool_calls[0].function.name == "show_table"  # type: ignore[attr-defined]
    assert json.loads(stored[2].content)["artifact"]["type"] == GRAPH_ARTIFACT_TYPE  # type: ignore[arg-type]
    assert store.get_thread(thread_id).title == "What does the widget do?"


async def test_a_run_without_artifacts_has_no_tool_call_events() -> None:
    store, thread_id = build_thread()
    stream = FakeStream(build_events(build_result(has_artifacts=False)))

    events = await collect(stream_run(build_run_input(thread_id), QUESTION, stream, store))

    assert not {
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_RESULT,
    } & set(event_types(events))
    assert [type(m) for m in store.list_messages(thread_id)] == [AssistantMessage, ActivityMessage]


async def test_a_failing_step_ends_with_an_error_snapshot_and_run_error() -> None:
    store, thread_id = build_thread()
    events_before_failure: list[AskEvent] = [
        StepStarted(name=SCHEMA_STEP),
        StepFinished(name=SCHEMA_STEP),
        StepStarted(name=QUERY_STEP),
    ]
    stream = FakeStream(events_before_failure, error=RuntimeError("neo4j down"))

    events = await collect(stream_run(build_run_input(thread_id), QUESTION, stream, store))

    assert event_types(events[-2:]) == [EventType.ACTIVITY_SNAPSHOT, EventType.RUN_ERROR]
    assert events[-1].message == "Could not query the graph"  # type: ignore[attr-defined]
    activity = store.list_messages(thread_id)[-1]
    assert isinstance(activity, ActivityMessage)
    assert [s["status"] for s in activity.content["steps"]] == ["completed", "error"]
    assert store.get_thread(thread_id).title == DEFAULT_THREAD_TITLE
    assert stream.is_closed


async def test_closing_the_run_after_the_first_text_stores_the_partial_turn() -> None:
    store, thread_id = build_thread()
    stream = FakeStream(build_events(build_result(has_artifacts=True)))
    run = stream_run(build_run_input(thread_id), QUESTION, stream, store)

    async for event in run:
        if event.type == EventType.TEXT_MESSAGE_CONTENT:
            break
    await run.aclose()

    assert stream.is_closed
    stored = store.list_messages(thread_id)
    assert stored[0].content == TEXT_PARTS[0]
    activity = stored[-1]
    assert isinstance(activity, ActivityMessage)
    assert activity.content["steps"][-1]["status"] == "error"
    assert store.get_thread(thread_id).title == DEFAULT_THREAD_TITLE


async def test_cancelling_the_run_task_stores_the_partial_turn() -> None:
    store, thread_id = build_thread()
    has_text = asyncio.Event()

    async def hang_after_text(request: AskRequest) -> AsyncGenerator[AskEvent]:
        yield StepStarted(name=ANSWER_STEP)
        yield TextDelta(text="partial")
        await asyncio.Event().wait()

    async def consume() -> None:
        run = stream_run(build_run_input(thread_id), QUESTION, hang_after_text, store)
        async for event in run:
            if event.type == EventType.TEXT_MESSAGE_CONTENT:
                has_text.set()

    task = asyncio.create_task(consume())
    await has_text.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    stored = store.list_messages(thread_id)
    assert stored[0].content == "partial"
    assert isinstance(stored[-1], ActivityMessage)
    assert store.get_thread(thread_id).title == DEFAULT_THREAD_TITLE


def test_the_board_starts_finishes_and_fails_steps() -> None:
    board = StepBoard().start(SCHEMA_STEP)
    assert board.open_step() == SCHEMA_STEP
    assert board.steps[0].status == "running"

    finished = board.finish(SCHEMA_STEP).start(QUERY_STEP)
    assert [step.status for step in finished.steps] == ["completed", "running"]

    failed = finished.fail_open()
    assert [step.status for step in failed.steps] == ["completed", "error"]
    assert failed.steps[1].message == "Could not query the graph"
    assert failed.open_step() is None
    assert board.steps[0].status == "running"
