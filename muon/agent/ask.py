"""The ask pipeline: select the universe, scope, resolve names, retrieve and write the answer.

`ask` is an async generator of `AskEvent`s. Consumers that only want the result read it with
`collect_answer` from `muon/agent/events.py`.
"""

import time
from collections.abc import AsyncIterator, Mapping
from contextlib import aclosing

from pydantic import BaseModel, ConfigDict

from muon.agent.events import (
    AnswerFinished,
    AskEvent,
    AskResult,
    AskStats,
    AskStreamError,
    StepFinished,
    StepStarted,
    TextDelta,
)
from muon.agent.routing import AgentRuntime, build_scope, select_universe
from muon.config import Settings
from muon.generation.generate import GeneratedAnswer, stream_answer
from muon.plugins import UniverseRegistry
from muon.retrieval.resolve import resolve_names
from muon.retrieval.retrieve import retrieve

SCHEMA_STEP = "reading schema"
RESOLVE_PROGRESS_STEP = "resolving names"
QUERY_STEP = "querying graph"
ANSWER_STEP = "writing answer"


class StepMessages(BaseModel):
    """The plain-English text of 1 step in each of its 3 states."""

    model_config = ConfigDict(frozen=True)

    running: str
    completed: str
    failed: str


STEP_MESSAGES: Mapping[str, StepMessages] = {
    SCHEMA_STEP: StepMessages(
        running="Reading the graph schema",
        completed="Read the graph schema",
        failed="Could not read the graph schema",
    ),
    RESOLVE_PROGRESS_STEP: StepMessages(
        running="Finding the names in the question",
        completed="Found the names in the question",
        failed="Could not resolve the names in the question",
    ),
    QUERY_STEP: StepMessages(
        running="Querying the graph",
        completed="Queried the graph",
        failed="Could not query the graph",
    ),
    ANSWER_STEP: StepMessages(
        running="Writing the answer",
        completed="Wrote the answer",
        failed="Could not write the answer",
    ),
}


class AskRequest(BaseModel):
    """One question, and the universe prefix to ask it in (`None` for the only one)."""

    model_config = ConfigDict(frozen=True)

    question: str
    universe: str | None = None


async def ask(
    request: AskRequest, registry: UniverseRegistry, settings: Settings, runtime: AgentRuntime
) -> AsyncIterator[AskEvent]:
    """Answer `request.question` from the graph of 1 universe, as a stream of events.

    The universe is chosen before the first event, so `RoutingError` for an unknown universe
    arrives before any graph read or model call. A failure raises out of the generator and
    leaves the running step with a `StepStarted` and no `StepFinished`; `ModelCallError` and
    `NameIndexError` arrive this way. Closing the generator closes the model stream.
    """
    started_at = time.perf_counter()
    plugin = select_universe(registry, request.universe)
    yield StepStarted(name=SCHEMA_STEP)
    scope = await build_scope(runtime.client, plugin, settings)
    yield StepFinished(name=SCHEMA_STEP)
    yield StepStarted(name=RESOLVE_PROGRESS_STEP)
    resolved = await resolve_names(request.question, scope, runtime)
    yield StepFinished(name=RESOLVE_PROGRESS_STEP)
    yield StepStarted(name=QUERY_STEP)
    retrieved = await retrieve(scope, request.question, runtime, resolved.names)
    yield StepFinished(name=QUERY_STEP)
    yield StepStarted(name=ANSWER_STEP)
    generated: GeneratedAnswer | None = None
    async with aclosing(
        stream_answer(request.question, retrieved.bundle, scope, runtime)
    ) as stream:
        async for item in stream:
            if isinstance(item, TextDelta):
                yield item
            else:
                generated = item
    if generated is None:
        raise AskStreamError("stream_answer ended with no answer")
    yield StepFinished(name=ANSWER_STEP)
    yield AnswerFinished(
        result=AskResult(
            answer=generated.answer,
            universe=scope.universe_label,
            bundle=retrieved.bundle,
            resolved_names=resolved.names,
            stats=AskStats(
                tool_calls=retrieved.tool_calls,
                cypher_calls=retrieved.cypher_calls,
                steps=[resolved.stats, retrieved.stats, generated.stats],
                total_latency_seconds=time.perf_counter() - started_at,
            ),
        )
    )
