"""The generate step: writes the answer to a question from the evidence bundle.

Map: `stream_answer` builds the handle table, streams the generate model through a
`MarkerRewriter` and yields each rewritten piece as a `TextDelta`. It ends with a
`GeneratedAnswer` built by `build_cited_answer` after `check_answer` adds its warning.
The old path, `generate_answer`, sends the question and the model-facing content of every evidence
item, parses the reply into an `AnswerDraft` under strict structured output, checks it with
`check_citations`, and reruns the call with the issues up to `max_check_retries` times.
What the model reads never goes past `MAX_CONTENT_ROWS` rows per item.
"""

import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import openai
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage

from muon.agent.events import TextDelta
from muon.agent.routing import AgentRuntime, UniverseScope
from muon.config import AgentStep
from muon.generation.answer import Answer, AnswerDraft
from muon.generation.artifacts import build_cited_answer, render_attribution
from muon.generation.check import CitationIssue, check_citations
from muon.generation.contract import CitedAnswer, CitedHandles
from muon.generation.handles import HandleTable, build_handle_table, render_bundle_with_handles
from muon.generation.markers import MarkerRewriter
from muon.llm.client import ModelCallError
from muon.retrieval.evidence import EvidenceBundle, EvidenceStatus, render_evidence_content
from muon.retrieval.retrieve import StepStats, fill_prompt, read_step_usage

PROMPT_PATH = Path(__file__).parent / "prompts" / "generate.md"
STREAM_PROMPT_PATH = Path(__file__).parent / "prompts" / "stream.md"
GENERATE_STEP = "generate"
BLOCK_SEPARATOR = "\n\n"
NO_EVIDENCE_WARNING = "The answer cites no evidence"


@dataclass(frozen=True)
class GeneratedAnswer:
    """What `stream_answer` ends with: the finished answer and the stats of the model call."""

    answer: CitedAnswer
    stats: StepStats


@dataclass(frozen=True)
class GenerateResult:
    answer: Answer
    stats: list[StepStats]


def build_human_message(
    question: str, bundle: EvidenceBundle, issues: list[CitationIssue]
) -> HumanMessage:
    """Join the question, the content of every evidence item and, on a rerun, the issues."""
    blocks = [f"Question: {question}"]
    blocks.extend(render_evidence_content(item) for item in bundle.items)
    if issues:
        listed = "\n".join(f"- {issue.reason}" for issue in issues)
        blocks.append(
            f"Your previous draft had these citation issues. Write the answer again and fix "
            f"them:\n{listed}"
        )
    return HumanMessage(BLOCK_SEPARATOR.join(blocks))


async def call_generator(
    messages: list[BaseMessage], step: str, runtime: AgentRuntime
) -> tuple[AnswerDraft, StepStats]:
    model_name = runtime.agent_settings.model_for(AgentStep.GENERATE)
    structured = runtime.generate_model.with_structured_output(
        AnswerDraft, method="json_schema", strict=True, include_raw=True
    )
    started_at = time.perf_counter()
    try:
        # With include_raw=True the result is a dict, which the stubs do not say.
        output = cast(dict[str, Any], await structured.ainvoke(messages))
    except openai.APIError as error:
        raise ModelCallError(GENERATE_STEP, model_name, str(error)) from error
    latency = time.perf_counter() - started_at
    parsed = output["parsed"]
    if parsed is None:
        raise ModelCallError(
            GENERATE_STEP, model_name, f"the reply is not a valid draft: {output['parsing_error']}"
        )
    usage = read_step_usage([output["raw"]])
    stats = StepStats(
        step=step,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        model_calls=usage.model_calls,
        latency_seconds=latency,
    )
    return AnswerDraft.model_validate(parsed, from_attributes=True), stats


async def generate_answer(
    question: str, bundle: EvidenceBundle, scope: UniverseScope, runtime: AgentRuntime
) -> GenerateResult:
    """Write the answer to `question` from `bundle`.

    The last draft is returned with 1 warning per citation issue it still has.
    """
    system_prompt = fill_prompt(PROMPT_PATH, {"universe_name": scope.plugin.name})
    issues: list[CitationIssue] = []
    stats: list[StepStats] = []
    for attempt in range(runtime.agent_settings.max_check_retries + 1):
        step = GENERATE_STEP if attempt == 0 else f"{GENERATE_STEP}_retry_{attempt}"
        messages = [SystemMessage(system_prompt), build_human_message(question, bundle, issues)]
        draft, step_stats = await call_generator(messages, step, runtime)
        stats.append(step_stats)
        issues = check_citations(draft, bundle)
        if not issues:
            break
    answer = Answer(
        **draft.model_dump(),
        universe=scope.universe_label,
        warnings=[issue.reason for issue in issues],
    )
    return GenerateResult(answer=answer, stats=stats)


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
    system_prompt = fill_prompt(STREAM_PROMPT_PATH, {"universe_name": universe_name})
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
