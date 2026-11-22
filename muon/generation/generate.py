"""The generate step: writes the answer to a question from the evidence bundle.

Map: `stream_answer` builds the handle table, streams the generate model through a
`MarkerRewriter` and yields each rewritten piece as a `TextDelta`. It ends with a
`GeneratedAnswer` built by `build_cited_answer` after `check_answer` adds its warning.
What the model reads never goes past `MAX_CONTENT_ROWS` rows per item.
"""

import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import openai
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage

from muon.agent.events import TextDelta
from muon.agent.routing import AgentRuntime, UniverseScope
from muon.config import AgentStep
from muon.generation.artifacts import build_cited_answer, render_attribution
from muon.generation.contract import CitedAnswer, CitedHandles
from muon.generation.handles import HandleTable, build_handle_table, render_bundle_with_handles
from muon.generation.markers import MarkerRewriter
from muon.llm.client import ModelCallError
from muon.retrieval.evidence import EvidenceBundle, EvidenceStatus
from muon.retrieval.retrieve import StepStats, fill_prompt, read_step_usage

PROMPT_PATH = Path(__file__).parent / "prompts" / "generate.md"
GENERATE_STEP = "generate"
BLOCK_SEPARATOR = "\n\n"
NO_EVIDENCE_WARNING = "The answer cites no evidence"


@dataclass(frozen=True)
class GeneratedAnswer:
    """What `stream_answer` ends with: the finished answer and the stats of the model call."""

    answer: CitedAnswer
    stats: StepStats


def check_answer(cited: CitedHandles, bundle: EvidenceBundle) -> tuple[str, ...]:
    """Return the warnings that the rewriter cannot see: the answer cites nothing.

    The rewriter's own warnings are already in `cited.warnings`.
    """
    if cited.node_keys or cited.table_ids:
        return ()
    has_rows = any(
        evidence.status is EvidenceStatus.OK and evidence.rows for evidence in bundle.items
    )
    return (NO_EVIDENCE_WARNING,) if has_rows else ()


def build_stream_messages(
    question: str, table: HandleTable, universe_name: str
) -> list[BaseMessage]:
    """The system prompt, then the question and the evidence text with handles."""
    system_prompt = fill_prompt(PROMPT_PATH, {"universe_name": universe_name})
    human_text = BLOCK_SEPARATOR.join([f"Question: {question}", render_bundle_with_handles(table)])
    return [SystemMessage(system_prompt), HumanMessage(human_text)]


def build_stream_stats(total: AIMessageChunk | None, latency: float) -> StepStats:
    usage = read_step_usage([total] if total is not None else [])
    return StepStats(
        step=GENERATE_STEP,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        model_calls=usage.model_calls,
        latency_seconds=latency,
    )


async def stream_answer(
    question: str, bundle: EvidenceBundle, scope: UniverseScope, runtime: AgentRuntime
) -> AsyncGenerator[TextDelta | GeneratedAnswer, None]:
    """Stream the answer to `question` from `bundle`.

    Yields each piece of rewritten text as a `TextDelta`, the attribution line last, then
    1 `GeneratedAnswer`. Closing the generator closes the model stream.
    """
    table = build_handle_table(bundle)
    messages = build_stream_messages(question, table, scope.plugin.name)
    rewriter = MarkerRewriter(table)
    pieces: list[str] = []
    total: AIMessageChunk | None = None
    started_at = time.perf_counter()
    try:
        # `astream` is an async generator that the stubs type as an iterator without `aclose`.
        model_stream = cast(
            AsyncGenerator[AIMessageChunk, None], runtime.generate_model.astream(messages)
        )
        async with aclosing(model_stream) as chunks:
            async for chunk in chunks:
                total = chunk if total is None else total + chunk
                if not isinstance(chunk.content, str) or not chunk.content:
                    continue
                piece = rewriter.feed(chunk.content)
                if piece:
                    pieces.append(piece)
                    yield TextDelta(text=piece)
    except openai.APIError as error:
        model_name = runtime.agent_settings.model_for(AgentStep.GENERATE)
        raise ModelCallError(GENERATE_STEP, model_name, str(error)) from error
    latency = time.perf_counter() - started_at
    tail = rewriter.finish()
    if tail:
        pieces.append(tail)
        yield TextDelta(text=tail)
    cited = rewriter.cited()
    attribution = render_attribution(cited.sources, scope.plugin.manifest.attribution)
    if attribution:
        yield TextDelta(text=attribution)
    warned = cited.model_copy(update={"warnings": (*cited.warnings, *check_answer(cited, bundle))})
    answer = build_cited_answer("".join(pieces), warned, table, scope.plugin.manifest.attribution)
    yield GeneratedAnswer(answer=answer, stats=build_stream_stats(total, latency))
