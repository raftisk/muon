"""The retrieve step: a `create_agent` loop over `run_cypher` that fills an `EvidenceBundle`.

Map: `retrieve` builds the system prompt from the scope, binds `run_cypher` to a fresh
`EvidenceLog`, runs the loop under 2 `ToolCallLimitMiddleware` instances (all tools, and
`run_cypher` alone) and reads the bundle from the log. The final message of the loop is
not used. `fill_prompt` is shared with the generate step.
"""

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import openai
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict

from muon.agent.routing import AgentRuntime, UniverseScope
from muon.llm.client import ModelCallError
from muon.retrieval.cypher import TOOL_NAME, EvidenceLog, build_run_cypher_tool
from muon.retrieval.evidence import EvidenceBundle

PROMPT_PATH = Path(__file__).parent / "prompts" / "retrieve.md"
MARKER_PATTERN = re.compile(r"\{\{(\w+)\}\}")
RETRIEVE_STEP = "retrieve"
RECURSION_MARGIN = 5
SUCCESS_STATUS = "success"


class PromptError(Exception):
    """A prompt file whose markers and values do not match."""


class StepStats(BaseModel):
    """Tokens and wall time of 1 model step."""

    model_config = ConfigDict(frozen=True)

    step: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


@dataclass(frozen=True)
class RetrieveResult:
    bundle: EvidenceBundle
    tool_calls: int
    cypher_calls: int
    stats: StepStats


def fill_prompt(path: Path, values: Mapping[str, str]) -> str:
    """Replace each `{{marker}}` of the file with its value.

    Only markers are replaced, so the braces and `$` of Cypher examples stay as written.
    Raises `PromptError` when a marker has no value or a value has no marker.
    """
    text = path.read_text(encoding="utf-8")
    markers = set(MARKER_PATTERN.findall(text))
    missing = sorted(markers - values.keys())
    unused = sorted(values.keys() - markers)
    if missing or unused:
        raise PromptError(
            f"Prompt {path.name}: markers without a value {missing}, "
            f"values without a marker {unused}"
        )
    return MARKER_PATTERN.sub(lambda match: values[match.group(1)], text)


def sum_token_usage(messages: list[object]) -> tuple[int, int]:
    """Return the input and output tokens that the AI messages report."""
    input_tokens = output_tokens = 0
    for message in messages:
        if isinstance(message, AIMessage) and message.usage_metadata:
            input_tokens += message.usage_metadata["input_tokens"]
            output_tokens += message.usage_metadata["output_tokens"]
    return input_tokens, output_tokens


def count_executed_tool_calls(messages: list[object]) -> int:
    """Count the tool calls that ran, leaving out those the limit middleware blocked."""
    return sum(
        1
        for message in messages
        if isinstance(message, ToolMessage) and message.status == SUCCESS_STATUS
    )


async def retrieve(scope: UniverseScope, question: str, runtime: AgentRuntime) -> RetrieveResult:
    """Run the tool loop for `question` and return the evidence it collected."""
    settings = runtime.agent_settings
    log = EvidenceLog()
    prompt = fill_prompt(
        PROMPT_PATH,
        {
            "universe_name": scope.plugin.name,
            "universe_label": scope.universe_label,
            "schema": scope.schema_context.text,
        },
    )
    agent = create_agent(
        runtime.retrieve_model,
        [build_run_cypher_tool(scope, runtime.client, log)],
        system_prompt=prompt,
        middleware=[
            ToolCallLimitMiddleware(run_limit=settings.max_tool_calls, exit_behavior="end"),
            ToolCallLimitMiddleware(
                tool_name=TOOL_NAME, run_limit=settings.max_cypher_calls, exit_behavior="continue"
            ),
        ],
    )
    config = RunnableConfig(recursion_limit=2 * settings.max_tool_calls + RECURSION_MARGIN)
    started_at = time.perf_counter()
    try:
        output = await agent.ainvoke({"messages": [HumanMessage(question)]}, config)
    except openai.APIError as error:
        raise ModelCallError(
            RETRIEVE_STEP, settings.resolve_retrieve_model(), str(error)
        ) from error
    latency = time.perf_counter() - started_at
    messages = list(output["messages"])
    input_tokens, output_tokens = sum_token_usage(messages)
    return RetrieveResult(
        bundle=EvidenceBundle(universe=scope.universe_label, question=question, items=log.items),
        tool_calls=count_executed_tool_calls(messages),
        cypher_calls=len(log.items),
        stats=StepStats(
            step=RETRIEVE_STEP,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
        ),
    )
