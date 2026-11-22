"""Builders for the eval tests: question files, ask results and a graph client for gold reads."""

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from muon.agent.events import AskResult, AskStats
from muon.evaluation.eval_scores import compare_nodes, score_answer
from muon.evaluation.questions import EvalQuestion
from muon.evaluation.run import Census, RunMeta, RunRecord, RunStatus
from muon.generation.contract import CitedAnswer, CitedHandles
from muon.graph import GraphNode
from muon.retrieval.context import READ_GRAPH_COUNTS
from muon.retrieval.evidence import (
    Evidence,
    EvidenceBundle,
    EvidenceStatus,
    GraphSlice,
    NodeCell,
    NodeKey,
    SliceNode,
)
from muon.retrieval.resolved_names import ResolvedName
from muon.retrieval.retrieve import StepStats
from tests.fixtures.toy_ontology import TOY_UNIVERSE
from tests.unit.retrieval.fakes import QueryGraphClient
from tests.unit.retrieval.scope import TOY_COUNTS
from tests.unit.retrieval.toy_graph import WIDGET_SET, build_toy_answers

EVALS_FILE_NAME = "evals.toml"
EVAL_SET_NAME = "general"
WIDGET_QUERY = "MATCH (w:ToyWidget:muontoy) RETURN w"
WIDGET_LABEL = "ToyWidget"
RETRIEVE_TOKENS = (100, 20)
GENERATE_TOKENS = (50, 10)
TOTAL_LATENCY_SECONDS = 2.5


def build_question_toml(question_id: str, **fields: str) -> str:
    """Write 1 `[[question]]` table; `fields` override or add to the default fields."""
    values = {
        "id": question_id,
        "family": "S1",
        "question": f"Question {question_id}?",
        "reference": f"Answer {question_id}.",
        "gold_query": WIDGET_QUERY,
        **fields,
    }
    lines = ["[[question]]", *(f'{name} = """{value}"""' for name, value in values.items())]
    return "\n".join(lines) + "\n"


def build_widget_key(widget_id: str) -> NodeKey:
    return NodeKey(label=WIDGET_LABEL, id=widget_id)


def build_widget_node(widget_id: str) -> GraphNode:
    return GraphNode(f"el-{widget_id}", frozenset(WIDGET_SET), {"id": widget_id, "name": widget_id})


def build_evidence(evidence_id: str, widget_ids: Sequence[str]) -> Evidence:
    """An `ok` evidence item whose rows hold 1 widget each, named after its id."""
    keys = [build_widget_key(widget_id) for widget_id in widget_ids]
    nodes = tuple(
        SliceNode(key=key, name=key.id, labels=(WIDGET_LABEL,), source_url=None) for key in keys
    )
    return Evidence(
        id=evidence_id,
        universe=TOY_UNIVERSE,
        tool="run_cypher",
        query=WIDGET_QUERY,
        purpose=f"widgets of {evidence_id}",
        status=EvidenceStatus.OK,
        columns=("w",),
        rows=tuple({"w": NodeCell(key=key)} for key in keys),
        row_count=len(keys),
        graph_slice=GraphSlice(nodes=nodes),
    )


def build_ask_result(
    cited_widget_ids: Sequence[str] = (),
    table_evidence_ids: Sequence[str] = (),
    items: Sequence[Evidence] = (),
    warnings: Sequence[str] = (),
    resolved_names: Sequence[ResolvedName] = (),
) -> AskResult:
    """An answer that cites the given widgets and tables, with the given evidence and warnings.

    The stats hold 1 tool call, 1 `run_cypher` call and the token constants above.
    """
    cited = CitedHandles(
        node_keys=tuple(build_widget_key(widget_id) for widget_id in cited_widget_ids),
        table_ids=tuple(table_evidence_ids),
        sources=(),
        warnings=tuple(warnings),
    )
    answer = CitedAnswer(
        text="Answer text.",
        sources=(),
        graph=None,
        tables=(),
        cited=cited,
        warnings=tuple(warnings),
    )
    steps = [
        StepStats(
            step="retrieve",
            input_tokens=RETRIEVE_TOKENS[0],
            output_tokens=RETRIEVE_TOKENS[1],
            reasoning_tokens=0,
            cached_input_tokens=0,
            model_calls=1,
            latency_seconds=1.0,
        ),
        StepStats(
            step="generate",
            input_tokens=GENERATE_TOKENS[0],
            output_tokens=GENERATE_TOKENS[1],
            reasoning_tokens=0,
            cached_input_tokens=0,
            model_calls=1,
            latency_seconds=1.0,
        ),
    ]
    return AskResult(
        answer=answer,
        universe=TOY_UNIVERSE,
        bundle=EvidenceBundle(universe=TOY_UNIVERSE, question="Q?", items=tuple(items)),
        resolved_names=list(resolved_names),
        stats=AskStats(
            tool_calls=1,
            cypher_calls=1,
            steps=steps,
            total_latency_seconds=TOTAL_LATENCY_SECONDS,
        ),
    )


def build_eval_client(
    gold_rows: Mapping[str, Callable[[Mapping[str, Any]], list[dict[str, Any]]]],
) -> QueryGraphClient:
    """A client that answers the schema reads of the toy universe and the given gold queries."""
    answers = build_toy_answers()
    answers[READ_GRAPH_COUNTS] = lambda parameters: [TOY_COUNTS]
    answers.update(gold_rows)
    return QueryGraphClient(answers)


def build_run_meta(run_id: str = "20261116-143005") -> RunMeta:
    return RunMeta(
        run_id=run_id,
        started_at=datetime(2026, 11, 16, 14, 30, 5, tzinfo=UTC),
        git_commit=None,
        universe=TOY_UNIVERSE,
        resolve_model="model-x",
        retrieve_model="model-x",
        generate_model="model-x",
        resolve_reasoning="none",
        retrieve_reasoning="none",
        generate_reasoning="none",
        max_tool_calls=10,
        max_cypher_calls=2,
        max_check_retries=1,
        hashes={f"questions:{EVAL_SET_NAME}": "0" * 64},
        census=Census(node_counts={WIDGET_LABEL: 1}, relationship_counts={}),
    )


def build_run_record(
    question: EvalQuestion, error: str | None = None, set_name: str = EVAL_SET_NAME
) -> RunRecord:
    """A record of `question`: an answered one with an empty gold set, or an error record."""
    base = {
        "question_id": question.id,
        "set_name": set_name,
        "family": question.family,
        "question": question.question,
        "reference": question.reference,
    }
    if error is not None:
        return RunRecord(
            **base, status=RunStatus.ERROR, error=error, result=None, comparison=None, score=None
        )
    result = build_ask_result()
    comparison = compare_nodes(result, [])
    return RunRecord(
        **base,
        status=RunStatus.OK,
        error=None,
        result=result,
        comparison=comparison,
        score=score_answer(result, comparison),
    )
