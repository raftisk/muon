"""Turns an `ask()` run into AG-UI events and stores the turn.

Map: `StepBoard` tracks the steps and renders them as `toki.steps` activity snapshots.
`ArtifactCall` is one artifact reported as a synthetic tool call. `stream_run` is the run
coordinator: it reads the `AskEvent` stream, yields the AG-UI events and records the turn in
the thread store.
"""

import json
import logging
from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing
from typing import Literal, Protocol
from uuid import uuid4

from ag_ui.core import (
    ActivityMessage,
    ActivitySnapshotEvent,
    AssistantMessage,
    BaseEvent,
    FunctionCall,
    Message,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCall,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    ToolMessage,
)
from pydantic import BaseModel, ConfigDict

from muon.agent.ask import STEP_MESSAGES, AskRequest
from muon.agent.events import (
    AnswerFinished,
    AskEvent,
    AskResult,
    StepFinished,
    StepStarted,
    TextDelta,
)
from muon.api.store import ThreadRepository
from muon.generation.contract import ArtifactEnvelope, CitedAnswer

logger = logging.getLogger(__name__)

STEP_ACTIVITY_TYPE = "toki.steps"
GRAPH_TOOL_NAME = "show_graph"
TABLE_TOOL_NAME = "show_table"
MESSAGE_ID_PREFIX = "msg_"
TOOL_CALL_ID_PREFIX = "call_"
ACTIVITY_ID_PREFIX = "act_"
ASSISTANT_ROLE = "assistant"
TOOL_ROLE = "tool"
RUN_FAILED_MESSAGE = "The answer failed"

type StepStatus = Literal["running", "completed", "error"]


class AnswerStream(Protocol):
    def __call__(self, request: AskRequest) -> AsyncGenerator[AskEvent]: ...


def create_id(prefix: str) -> str:
    return f"{prefix}{uuid4().hex}"


class StepState(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    status: StepStatus
    message: str


class StepBoard(BaseModel):
    """The ordered steps of 1 run. Every change returns a new board.

    Step names must be keys of `STEP_MESSAGES`; an unknown name raises `KeyError`.
    """

    model_config = ConfigDict(frozen=True)

    steps: tuple[StepState, ...] = ()

    def start(self, name: str) -> "StepBoard":
        step = StepState(id=name, status="running", message=STEP_MESSAGES[name].running)
        return StepBoard(steps=(*self.steps, step))

    def finish(self, name: str) -> "StepBoard":
        done = StepState(id=name, status="completed", message=STEP_MESSAGES[name].completed)
        return StepBoard(steps=tuple(done if step.id == name else step for step in self.steps))

    def fail_open(self) -> "StepBoard":
        """Mark the running step, if any, as an error with its failed message."""
        return StepBoard(
            steps=tuple(
                StepState(id=step.id, status="error", message=STEP_MESSAGES[step.id].failed)
                if step.status == "running"
                else step
                for step in self.steps
            )
        )

    def open_step(self) -> str | None:
        return next((step.id for step in self.steps if step.status == "running"), None)

    def to_content(self) -> dict[str, list[dict[str, str]]]:
        return {"steps": [step.model_dump() for step in self.steps]}

    def to_snapshot(self, activity_id: str) -> ActivitySnapshotEvent:
        return ActivitySnapshotEvent(
            message_id=activity_id,
            activity_type=STEP_ACTIVITY_TYPE,
            content=self.to_content(),
            replace=True,
        )

    def to_message(self, activity_id: str) -> ActivityMessage:
        return ActivityMessage(
            id=activity_id, activity_type=STEP_ACTIVITY_TYPE, content=self.to_content()
        )


def iter_artifact_calls(answer: CitedAnswer) -> Iterator[tuple[str, ArtifactEnvelope]]:
    """Yield the graph first when there is one, then the tables, with their tool names."""
    if answer.graph is not None:
        yield GRAPH_TOOL_NAME, answer.graph
    for table in answer.tables:
        yield TABLE_TOOL_NAME, table


class ArtifactCall(BaseModel):
    """One artifact reported as a tool call, with the ids its events and messages share."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    arguments: str
    content: str
    assistant_message_id: str
    tool_call_id: str
    result_message_id: str

    @classmethod
    def build(cls, tool_name: str, envelope: ArtifactEnvelope) -> "ArtifactCall":
        return cls(
            tool_name=tool_name,
            arguments=json.dumps({"artifactId": envelope.artifact.id}),
            content=envelope.to_json(),
            assistant_message_id=create_id(MESSAGE_ID_PREFIX),
            tool_call_id=create_id(TOOL_CALL_ID_PREFIX),
            result_message_id=create_id(MESSAGE_ID_PREFIX),
        )

    def to_events(self) -> list[BaseEvent]:
        return [
            ToolCallStartEvent(
                tool_call_id=self.tool_call_id,
                tool_call_name=self.tool_name,
                parent_message_id=self.assistant_message_id,
            ),
            ToolCallArgsEvent(tool_call_id=self.tool_call_id, delta=self.arguments),
            ToolCallEndEvent(tool_call_id=self.tool_call_id),
            ToolCallResultEvent(
                message_id=self.result_message_id,
                tool_call_id=self.tool_call_id,
                content=self.content,
                role=TOOL_ROLE,
            ),
        ]

    def to_messages(self) -> list[Message]:
        call = ToolCall(
            id=self.tool_call_id,
            function=FunctionCall(name=self.tool_name, arguments=self.arguments),
        )
        return [
            AssistantMessage(id=self.assistant_message_id, tool_calls=[call]),
            ToolMessage(
                id=self.result_message_id, tool_call_id=self.tool_call_id, content=self.content
            ),
        ]


def build_text_message(message_id: str, text: str) -> list[Message]:
    return [AssistantMessage(id=message_id, content=text)] if text else []


async def stream_run(
    run_input: RunAgentInput,
    question: str,
    answer_stream: AnswerStream,
    store: ThreadRepository,
) -> AsyncGenerator[BaseEvent]:
    """Run `question` through `answer_stream` and yield the AG-UI events of the run.

    The stream always starts with `RUN_STARTED` and ends with `RUN_FINISHED` or `RUN_ERROR`.
    The turn is stored once the run finishes. When the run fails or the consumer stops reading
    (`GeneratorExit` or `CancelledError` on a client disconnect), the partial text and the
    steps with the open one marked as an error are stored instead, and the title stays.
    The user message is stored by the caller.
    """
    thread_id = run_input.thread_id
    message_id = create_id(MESSAGE_ID_PREFIX)
    activity_id = create_id(ACTIVITY_ID_PREFIX)
    board = StepBoard()
    text = ""
    is_text_open = False
    result: AskResult | None = None
    is_turn_stored = False

    yield RunStartedEvent(thread_id=thread_id, run_id=run_input.run_id)
    try:
        async with aclosing(answer_stream(AskRequest(question=question))) as events:
            async for event in events:
                if isinstance(event, StepStarted):
                    board = board.start(event.name)
                    yield board.to_snapshot(activity_id)
                elif isinstance(event, StepFinished):
                    board = board.finish(event.name)
                    yield board.to_snapshot(activity_id)
                elif isinstance(event, TextDelta):
                    if not is_text_open:
                        is_text_open = True
                        yield TextMessageStartEvent(message_id=message_id, role=ASSISTANT_ROLE)
                    text += event.text
                    yield TextMessageContentEvent(message_id=message_id, delta=event.text)
                elif isinstance(event, AnswerFinished):
                    result = event.result
        if is_text_open:
            yield TextMessageEndEvent(message_id=message_id)
        calls = (
            [
                ArtifactCall.build(name, envelope)
                for name, envelope in iter_artifact_calls(result.answer)
            ]
            if result is not None
            else []
        )
        for call in calls:
            for call_event in call.to_events():
                yield call_event
        for message in build_text_message(message_id, text):
            store.append_message(thread_id, message)
        for call in calls:
            for message in call.to_messages():
                store.append_message(thread_id, message)
        store.append_message(thread_id, board.to_message(activity_id))
        store.apply_first_run_title(thread_id, question)
        is_turn_stored = True
        yield RunFinishedEvent(thread_id=thread_id, run_id=run_input.run_id)
    except Exception:
        logger.exception("Run %s failed", run_input.run_id)
        open_step = board.open_step()
        failed_message = (
            STEP_MESSAGES[open_step].failed if open_step is not None else RUN_FAILED_MESSAGE
        )
        board = board.fail_open()
        yield board.to_snapshot(activity_id)
        yield RunErrorEvent(message=failed_message)
    finally:
        if not is_turn_stored:
            for message in build_text_message(message_id, text):
                store.append_message(thread_id, message)
            if board.steps:
                store.append_message(thread_id, board.fail_open().to_message(activity_id))
