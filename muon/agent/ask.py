"""The ask pipeline: select the universe, build its scope, retrieve, generate and check."""

import time

from pydantic import BaseModel, ConfigDict

from muon.agent.routing import AgentRuntime, build_scope, select_universe
from muon.agent.universes import UniverseRegistry
from muon.config import Settings
from muon.generation.answer import Answer
from muon.generation.generate import generate_answer
from muon.retrieval.evidence import EvidenceBundle
from muon.retrieval.retrieve import StepStats, retrieve


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
    """The answer, the evidence it rests on and the stats of the run."""

    answer: Answer
    bundle: EvidenceBundle
    stats: AskStats


async def ask(
    request: AskRequest, registry: UniverseRegistry, settings: Settings, runtime: AgentRuntime
) -> AskResult:
    """Answer `request.question` from the graph of 1 universe.

    The universe is chosen before any graph read or model call. Raises `RoutingError` when
    it cannot be chosen, and `ModelCallError` when the provider fails.
    """
    started_at = time.perf_counter()
    plugin = select_universe(registry, request.universe)
    scope = await build_scope(runtime.client, plugin, settings)
    retrieved = await retrieve(scope, request.question, runtime)
    generated = await generate_answer(request.question, retrieved.bundle, scope, runtime)
    return AskResult(
        answer=generated.answer,
        bundle=retrieved.bundle,
        stats=AskStats(
            tool_calls=retrieved.tool_calls,
            cypher_calls=retrieved.cypher_calls,
            steps=[retrieved.stats, *generated.stats],
            total_latency_seconds=time.perf_counter() - started_at,
        ),
    )
