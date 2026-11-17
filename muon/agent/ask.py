"""The ask pipeline: select the universe, scope, resolve names, retrieve, generate and check."""

import time

from pydantic import BaseModel, ConfigDict

from muon.agent.routing import AgentRuntime, build_scope, select_universe
from muon.config import Settings
from muon.generation.answer import Answer
from muon.generation.generate import generate_answer
from muon.plugins import UniverseRegistry
from muon.retrieval.evidence import EvidenceBundle
from muon.retrieval.resolve import resolve_names
from muon.retrieval.resolved_names import ResolvedName
from muon.retrieval.retrieve import StepStats, retrieve

SCHEMA_STEP = "reading schema"
RESOLVE_PROGRESS_STEP = "resolving names"
QUERY_STEP = "querying graph"
ANSWER_STEP = "writing answer"


class AskRequest(BaseModel):
    """One question, and the universe prefix to ask it in (`None` for the only one)."""

    model_config = ConfigDict(frozen=True)

    question: str
    universe: str | None = None


class AskStats(BaseModel):
    """What 1 question cost: executed tool calls, `run_cypher` calls, steps and wall time."""

    tool_calls: int
    cypher_calls: int
    steps: list[StepStats]
    total_latency_seconds: float


class AskResult(BaseModel):
    """The answer, the evidence it rests on, the names resolved from the question and the stats.

    `resolved_names` is recorded for logs and evals. It is not evidence and no claim cites it.
    """

    answer: Answer
    bundle: EvidenceBundle
    resolved_names: list[ResolvedName]
    stats: AskStats


async def ask(
    request: AskRequest, registry: UniverseRegistry, settings: Settings, runtime: AgentRuntime
) -> AskResult:
    """Answer `request.question` from the graph of 1 universe.

    The universe is chosen before any graph read or model call. Raises `RoutingError` when
    it cannot be chosen, `ModelCallError` when the provider fails and `NameIndexError` when
    the full-text index on names is missing.
    """
    started_at = time.perf_counter()
    plugin = select_universe(registry, request.universe)
    with runtime.progress.step(SCHEMA_STEP):
        scope = await build_scope(runtime.client, plugin, settings)
    with runtime.progress.step(RESOLVE_PROGRESS_STEP):
        resolved = await resolve_names(request.question, scope, runtime)
    with runtime.progress.step(QUERY_STEP):
        retrieved = await retrieve(scope, request.question, runtime, resolved.names)
    with runtime.progress.step(ANSWER_STEP):
        generated = await generate_answer(request.question, retrieved.bundle, scope, runtime)
    return AskResult(
        answer=generated.answer,
        bundle=retrieved.bundle,
        resolved_names=resolved.names,
        stats=AskStats(
            tool_calls=retrieved.tool_calls,
            cypher_calls=retrieved.cypher_calls,
            steps=[resolved.stats, retrieved.stats, *generated.stats],
            total_latency_seconds=time.perf_counter() - started_at,
        ),
    )
