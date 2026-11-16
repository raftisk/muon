"""The eval run: ask the selected questions in order, score them and write the run folder.

Map: `run_eval` resolves every gold query first, then asks the questions 1 at a time and
returns an `EvalRun` with 1 `RunRecord` per question. `write_run` writes `run.json` (meta,
totals and the scores), `answers/<id>.json` (the full record, enough to re-score without a
model) and `review.md`, the sheet for the review by hand. The command renders the run through
`muon/console.py`. A provider failure on 1 question becomes an error
record and the run goes on. A lost graph connection ends the run and keeps the questions
that finished.
"""

import hashlib
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from muon.agent.ask import AskRequest, AskResult, ask
from muon.agent.routing import AgentRuntime, build_scope, select_universe
from muon.agent.universes import UniverseRegistry
from muon.config import PROJECT_ROOT, AgentSettings, ReasoningLevel, Settings
from muon.evaluation.eval_scores import (
    NodeComparison,
    QuestionScore,
    ScoredNode,
    compare_nodes,
    resolve_gold,
    score_answer,
)
from muon.evaluation.questions import SelectedQuestion
from muon.files import write_text_atomically
from muon.generation.generate import PROMPT_PATH as GENERATE_PROMPT_PATH
from muon.graph import GraphConnectionError
from muon.llm.client import ModelCallError
from muon.retrieval.retrieve import PROMPT_PATH as RETRIEVE_PROMPT_PATH
from muon.retrieval.schema import GraphSchema

DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "evals"
RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
RUN_FILE_NAME = "run.json"
REVIEW_FILE_NAME = "review.md"
ANSWERS_DIR_NAME = "answers"
JSON_SUFFIX = ".json"
JSON_INDENT = 2
MAX_LISTED_NODES = 20
GIT_COMMAND = ("git", "rev-parse", "HEAD")
HASH_QUESTIONS_PREFIX = "questions:"
HASH_RETRIEVE_PROMPT = "retrieve_prompt"
HASH_GENERATE_PROMPT = "generate_prompt"
HASH_NOTES = "notes"
# EvalDisplay in muon/console.py compares step names against this one to keep it out of
# the per-question list.
GOLD_STEP = "resolving gold queries"
VERDICT_LINE = "Verdict: [ ] correct  [ ] partial  [ ] wrong"

type AskFunction = Callable[
    [AskRequest, UniverseRegistry, Settings, AgentRuntime], Awaitable[AskResult]
]
type AskText = Callable[[str], Awaitable[AskResult]]


class RunStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class RunModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Census(RunModel):
    """What the graph held at run time: nodes per leaf class and relationships per type."""

    node_counts: dict[str, int]
    relationship_counts: dict[str, int]


class RunMeta(RunModel):
    """What a run needs to be compared with another: the code, prompts, limits and graph."""

    run_id: str
    started_at: datetime
    git_commit: str | None
    universe: str
    retrieve_model: str
    generate_model: str
    retrieve_reasoning: ReasoningLevel | None
    generate_reasoning: ReasoningLevel | None
    max_tool_calls: int
    max_cypher_calls: int
    max_check_retries: int
    hashes: dict[str, str]
    census: Census


class RunRecord(RunModel):
    """1 question of a run: its outcome, and for an answered one the nodes and the score."""

    question_id: str
    set_name: str
    family: str
    question: str
    reference: str
    status: RunStatus
    error: str | None
    result: AskResult | None
    comparison: NodeComparison | None
    score: QuestionScore | None


class EvalRun(RunModel):
    """The meta and the records of a run, and why it stopped early (`None` when it did not)."""

    meta: RunMeta
    records: tuple[RunRecord, ...]
    stop: str | None


class RunTotals(RunModel):
    """Sums and means over the questions of a run. Means cover the answered questions only."""

    question_count: int
    error_count: int
    valid_count: int
    mean_precision: float
    mean_recall: float
    tool_calls: int
    latency_seconds: float
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    model_calls: int


class QuestionSummary(RunModel):
    question_id: str
    set_name: str
    family: str
    status: RunStatus
    error: str | None
    score: QuestionScore | None


class RunFile(RunModel):
    """The contents of `run.json`: the scores without the node lists of `answers/`."""

    meta: RunMeta
    totals: RunTotals
    set_totals: dict[str, RunTotals]
    stop: str | None
    questions: tuple[QuestionSummary, ...]


@dataclass(frozen=True)
class EvalRequest:
    """The questions to ask, the universe to ask them in and the time the run started.

    `set_hashes` holds the sha256 of the question file of each set in the run. `on_result` is
    called with each record as soon as its question finishes.
    """

    prefix: str
    questions: tuple[SelectedQuestion, ...]
    set_hashes: dict[str, str]
    settings: Settings
    started_at: datetime
    on_result: Callable[[RunRecord], None] | None = None


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_git_commit() -> str | None:
    """Return the commit of the working tree, or `None` when git cannot say."""
    try:
        completed = subprocess.run(
            GIT_COMMAND, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def build_census(schema: GraphSchema) -> Census:
    relationship_counts: dict[str, int] = {}
    for pattern in schema.patterns:
        relationship_counts[pattern.type] = relationship_counts.get(pattern.type, 0) + pattern.count
    return Census(
        node_counts={node.label: node.count for node in schema.node_labels},
        relationship_counts=dict(sorted(relationship_counts.items())),
    )


async def ask_record(
    selected: SelectedQuestion, gold: Sequence[ScoredNode], ask_text: AskText
) -> RunRecord:
    """Ask 1 question and score its answer. A `ModelCallError` becomes an error record."""
    question = selected.question
    try:
        result = await ask_text(question.question)
    except ModelCallError as error:
        return RunRecord(
            question_id=question.id,
            set_name=selected.set_name,
            family=question.family,
            question=question.question,
            reference=question.reference,
            status=RunStatus.ERROR,
            error=str(error),
            result=None,
            comparison=None,
            score=None,
        )
    comparison = compare_nodes(result, gold)
    return RunRecord(
        question_id=question.id,
        set_name=selected.set_name,
        family=question.family,
        question=question.question,
        reference=question.reference,
        status=RunStatus.OK,
        error=None,
        result=result,
        comparison=comparison,
        score=score_answer(result, comparison),
    )


async def run_eval(
    request: EvalRequest,
    registry: UniverseRegistry,
    runtime: AgentRuntime,
    ask_question: AskFunction = ask,
) -> EvalRun:
    """Run the questions of `request` against the graph and the model of `runtime`.

    Every gold query runs before the first model call, and an `EvalError` ends the run
    there. A lost graph connection ends the loop and is recorded in `EvalRun.stop`.
    """
    plugin = select_universe(registry, request.prefix)
    scope = await build_scope(runtime.client, plugin, request.settings)
    with runtime.progress.step(GOLD_STEP):
        gold = {
            selected.question.id: await resolve_gold(selected.question, scope, runtime.client)
            for selected in request.questions
        }
    agent_settings = runtime.agent_settings
    meta = build_meta(request, agent_settings, scope.schema_context.graph_schema, plugin.notes_path)

    async def ask_text(text: str) -> AskResult:
        ask_request = AskRequest(question=text, universe=request.prefix)
        return await ask_question(ask_request, registry, request.settings, runtime)

    records: list[RunRecord] = []
    stop: str | None = None
    for selected in request.questions:
        question_id = selected.question.id
        try:
            record = await ask_record(selected, gold[question_id], ask_text)
        except GraphConnectionError as error:
            stop = f"{question_id}: {error}"
            break
        records.append(record)
        if request.on_result is not None:
            request.on_result(record)
    return EvalRun(meta=meta, records=tuple(records), stop=stop)


def build_meta(
    request: EvalRequest, agent_settings: AgentSettings, schema: GraphSchema, notes_path: Path
) -> RunMeta:
    return RunMeta(
        run_id=request.started_at.astimezone(UTC).strftime(RUN_ID_FORMAT),
        started_at=request.started_at,
        git_commit=read_git_commit(),
        universe=request.prefix,
        retrieve_model=agent_settings.resolve_retrieve_model(),
        generate_model=agent_settings.resolve_generate_model(),
        retrieve_reasoning=agent_settings.retrieve_reasoning,
        generate_reasoning=agent_settings.generate_reasoning,
        max_tool_calls=agent_settings.max_tool_calls,
        max_cypher_calls=agent_settings.max_cypher_calls,
        max_check_retries=agent_settings.max_check_retries,
        hashes={
            **{
                f"{HASH_QUESTIONS_PREFIX}{name}": value
                for name, value in request.set_hashes.items()
            },
            HASH_RETRIEVE_PROMPT: hash_file(RETRIEVE_PROMPT_PATH),
            HASH_GENERATE_PROMPT: hash_file(GENERATE_PROMPT_PATH),
            HASH_NOTES: hash_file(notes_path),
        },
        census=build_census(schema),
    )


def summarize(records: Sequence[RunRecord]) -> RunTotals:
    scores = [record.score for record in records if record.score is not None]
    count = len(scores)
    return RunTotals(
        question_count=len(records),
        error_count=len(records) - count,
        valid_count=sum(1 for score in scores if score.is_citation_valid),
        mean_precision=sum(score.precision for score in scores) / count if count else 0.0,
        mean_recall=sum(score.recall for score in scores) / count if count else 0.0,
        tool_calls=sum(score.tool_calls for score in scores),
        latency_seconds=sum(score.latency_seconds for score in scores),
        input_tokens=sum(score.input_tokens for score in scores),
        cached_input_tokens=sum(score.cached_input_tokens for score in scores),
        output_tokens=sum(score.output_tokens for score in scores),
        reasoning_tokens=sum(score.reasoning_tokens for score in scores),
        model_calls=sum(score.model_calls for score in scores),
    )


def summarize_sets(run: EvalRun) -> dict[str, RunTotals]:
    """The totals of each set that has a record, in order of first appearance."""
    by_set: dict[str, list[RunRecord]] = {}
    for record in run.records:
        by_set.setdefault(record.set_name, []).append(record)
    return {name: summarize(records) for name, records in by_set.items()}


def format_node(node: ScoredNode) -> str:
    return f"{node.name or node.key.id} ({node.key.label} {node.key.id})"


def format_node_list(nodes: Sequence[ScoredNode]) -> str:
    if not nodes:
        return "none"
    listed = ", ".join(format_node(node) for node in nodes[:MAX_LISTED_NODES])
    hidden = len(nodes) - MAX_LISTED_NODES
    return f"{listed} and {hidden} more" if hidden > 0 else listed


def render_record(record: RunRecord) -> str:
    """The review section of 1 question."""
    lines = [
        f"## {record.question_id} ({record.set_name}, {record.family})",
        "",
        f"Question: {record.question}",
        f"Reference: {record.reference}",
    ]
    result, score, comparison = record.result, record.score, record.comparison
    if result is None or score is None or comparison is None:
        lines.append(f"Error: {record.error}")
    else:
        answer = result.answer
        lines.append(f"Answer: {answer.text}")
        lines.extend(f"Assumption: {item}" for item in answer.assumptions)
        lines.extend(f"Warning: {item}" for item in answer.warnings)
        lines.append("Queries:")
        for number, item in enumerate(result.bundle.items, start=1):
            lines.append(f"{number}. {item.purpose} [{item.status}, {item.row_count} rows]")
            lines.extend(["", "```cypher", item.query, "```", ""])
        lines.append(
            f"Metrics: precision {score.precision:.2f}, recall {score.recall:.2f}, "
            f"citations valid {'yes' if score.is_citation_valid else 'no'}, "
            f"tool calls {score.tool_calls}, model calls {score.model_calls}, "
            f"{score.latency_seconds:.1f} s, {score.input_tokens} input "
            f"({score.cached_input_tokens} cached) and {score.output_tokens} output "
            f"({score.reasoning_tokens} reasoning) tokens"
        )
        lines.append(f"Missed gold nodes: {format_node_list(comparison.missed)}")
        lines.append(f"Extra cited nodes: {format_node_list(comparison.extra)}")
    lines.extend(["", VERDICT_LINE, "Notes:", ""])
    return "\n".join(lines)


def describe_step_model(model: str, reasoning: ReasoningLevel | None) -> str:
    return f"{model}, reasoning {reasoning or 'provider default'}"


def render_review(run: EvalRun) -> str:
    """The sheet for the review by hand: 1 section per question. Code never reads it back."""
    meta = run.meta
    header = [
        f"# Eval review {meta.run_id}",
        "",
        f"Universe: {meta.universe}, commit: {meta.git_commit or 'unknown'}",
        f"Retrieve: {describe_step_model(meta.retrieve_model, meta.retrieve_reasoning)}",
        f"Generate: {describe_step_model(meta.generate_model, meta.generate_reasoning)}",
        "",
    ]
    return "\n".join([*header, *(render_record(record) for record in run.records)])


def write_run(run: EvalRun, output_dir: Path) -> Path:
    """Write `run.json`, `answers/<id>.json` and `review.md` under `output_dir/<run id>/`."""
    folder = output_dir / run.meta.run_id
    for record in run.records:
        answer_path = folder / ANSWERS_DIR_NAME / f"{record.question_id}{JSON_SUFFIX}"
        write_text_atomically(answer_path, record.model_dump_json(indent=JSON_INDENT))
    run_file = RunFile(
        meta=run.meta,
        totals=summarize(run.records),
        set_totals=summarize_sets(run),
        stop=run.stop,
        questions=tuple(
            QuestionSummary(
                question_id=record.question_id,
                set_name=record.set_name,
                family=record.family,
                status=record.status,
                error=record.error,
                score=record.score,
            )
            for record in run.records
        ),
    )
    write_text_atomically(folder / RUN_FILE_NAME, run_file.model_dump_json(indent=JSON_INDENT))
    write_text_atomically(folder / REVIEW_FILE_NAME, render_review(run))
    return folder
