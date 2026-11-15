"""The generate step: writes an `Answer` from the evidence bundle and checks its citations.

Map: `generate_answer` sends the question and the model-facing content of every evidence
item, parses the reply into an `AnswerDraft` under strict structured output, checks it with
`check_citations`, and reruns the call with the issues up to `max_check_retries` times.
What the model reads never goes past `MAX_CONTENT_ROWS` rows per item.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from muon.agent.routing import AgentRuntime, UniverseScope
from muon.generation.answer import Answer, AnswerDraft
from muon.generation.check import CitationIssue, check_citations
from muon.llm.client import ModelCallError
from muon.retrieval.evidence import EvidenceBundle, render_evidence_content
from muon.retrieval.retrieve import StepStats, fill_prompt

PROMPT_PATH = Path(__file__).parent / "prompts" / "generate.md"
GENERATE_STEP = "generate"
BLOCK_SEPARATOR = "\n\n"


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


def read_usage(raw: Any) -> tuple[int, int]:
    usage = raw.usage_metadata if isinstance(raw, AIMessage) else None
    return (usage["input_tokens"], usage["output_tokens"]) if usage else (0, 0)


async def call_generator(
    messages: list[BaseMessage], step: str, runtime: AgentRuntime
) -> tuple[AnswerDraft, StepStats]:
    model_name = runtime.agent_settings.model
    structured = runtime.chat_model.with_structured_output(
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
    input_tokens, output_tokens = read_usage(output["raw"])
    stats = StepStats(
        step=step, input_tokens=input_tokens, output_tokens=output_tokens, latency_seconds=latency
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
