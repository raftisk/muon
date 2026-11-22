"""The events of an `ask()` run: step brackets, answer text and the final result.

Map: `StepStarted` and `StepFinished` bracket each step, `TextDelta` carries answer text
and `AnswerFinished` ends a successful run with the `AskResult`. `collect_answer` reads a
stream to its result and drives a `StepTracker` from the step events.
"""

from collections.abc import AsyncIterator
from contextlib import ExitStack
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from muon.agent.routing import NO_PROGRESS, StepTracker
from muon.generation.contract import CitedAnswer
from muon.retrieval.evidence import EvidenceBundle
from muon.retrieval.resolved_names import ResolvedName
from muon.retrieval.retrieve import StepStats


class RunEventModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class StepStarted(RunEventModel):
    kind: Literal["step_started"] = "step_started"
    name: str


class StepFinished(RunEventModel):
    kind: Literal["step_finished"] = "step_finished"
    name: str


class TextDelta(RunEventModel):
    """Answer text that is safe to show: handles are already rewritten into links."""

    kind: Literal["text_delta"] = "text_delta"
    text: str


class AskStats(BaseModel):
    """What 1 question cost: executed tool calls, `run_cypher` calls, steps and wall time."""

    tool_calls: int
    cypher_calls: int
    steps: list[StepStats]
    total_latency_seconds: float


class AskResult(BaseModel):
    """The answer, the universe label, the evidence it rests on, the resolved names and the stats.

    `resolved_names` is recorded for logs and evals. It is not evidence and no handle cites it.
    """

    answer: CitedAnswer
    universe: str
    bundle: EvidenceBundle
    resolved_names: list[ResolvedName]
    stats: AskStats


class AnswerFinished(RunEventModel):
    """The last event of a successful run."""

    kind: Literal["answer_finished"] = "answer_finished"
    result: AskResult


type AskEvent = Annotated[
    StepStarted | StepFinished | TextDelta | AnswerFinished, Field(discriminator="kind")
]


class AskStreamError(Exception):
    """A stream ended with no `AnswerFinished`."""


async def collect_answer(
    events: AsyncIterator[AskEvent], tracker: StepTracker = NO_PROGRESS
) -> AskResult:
    """Read `events` to the end and return the result of its `AnswerFinished`.

    Each step is shown on `tracker` from its `StepStarted` to its `StepFinished`. A failure
    leaves the open step through its context manager, so the tracker marks it failed.
    Raises `AskStreamError` when the stream ends with no `AnswerFinished`.
    """
    result: AskResult | None = None
    with ExitStack() as open_steps:
        async for event in events:
            if isinstance(event, StepStarted):
                open_steps.enter_context(tracker.step(event.name))
            elif isinstance(event, StepFinished):
                open_steps.pop_all().close()
            elif isinstance(event, AnswerFinished):
                result = event.result
    if result is None:
        raise AskStreamError("The ask stream ended with no AnswerFinished event")
    return result
