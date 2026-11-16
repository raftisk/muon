"""Gold resolution and scoring of 1 answer, with no model call.

Map: `resolve_gold` runs the gold query of a question through the same guardrails as the
agent's own queries and returns the nodes of its graph slice. `compare_nodes` sets the
nodes an answer cites against the gold nodes. `score_answer` turns that comparison and the
`AskStats` of the answer into the numbers of `QuestionScore`. A node counts as cited when a
claim names it or when it sits in the rows of an evidence item the answer prints in full.
"""

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from muon.agent.ask import AskResult
from muon.agent.routing import UniverseScope
from muon.evaluation.questions import EvalError, EvalQuestion
from muon.graph import GraphClient
from muon.retrieval.cypher import CallDescription, run_checked_query
from muon.retrieval.evidence import EvidenceBundle, EvidenceStatus, NodeKey, SliceNode

GOLD_EVIDENCE_ID = "gold"


class ScoredModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScoredNode(ScoredModel):
    """A graph node by key, with its name when an evidence slice holds it."""

    key: NodeKey
    name: str | None


class NodeComparison(ScoredModel):
    """The nodes an answer cites and the gold nodes, with the nodes only 1 side holds."""

    cited: tuple[ScoredNode, ...]
    gold: tuple[ScoredNode, ...]
    missed: tuple[ScoredNode, ...]
    extra: tuple[ScoredNode, ...]


class QuestionScore(ScoredModel):
    """The metrics of 1 answer. Tokens and counters come from its `AskStats`."""

    precision: float
    recall: float
    is_citation_valid: bool
    tool_calls: int
    cypher_calls: int
    latency_seconds: float
    input_tokens: int
    output_tokens: int


def order_key(node: ScoredNode | SliceNode) -> tuple[str, str]:
    return node.key.label, node.key.id


def sort_nodes(nodes: Iterable[ScoredNode]) -> tuple[ScoredNode, ...]:
    return tuple(sorted(nodes, key=order_key))


async def resolve_gold(
    question: EvalQuestion, scope: UniverseScope, client: GraphClient
) -> tuple[SliceNode, ...]:
    """Run the gold query of `question` and return the nodes of its result.

    Raises `EvalError` when the query is rejected or fails, returns no rows, or returns
    rows without a node.
    """
    call = CallDescription(GOLD_EVIDENCE_ID, question.gold_query, f"gold nodes of {question.id}")
    evidence = await run_checked_query(call, scope, client)
    prefix = f"Gold query of {question.id}"
    if evidence.status in (EvidenceStatus.REJECTED, EvidenceStatus.FAILED):
        raise EvalError(f"{prefix} is {evidence.status}: {evidence.message}")
    if evidence.status is EvidenceStatus.EMPTY:
        raise EvalError(f"{prefix} returned no rows")
    if not evidence.graph_slice.nodes:
        raise EvalError(f"{prefix} returned rows with no node; return the nodes themselves")
    return tuple(sorted(evidence.graph_slice.nodes, key=order_key))


def find_name(key: NodeKey, bundle: EvidenceBundle) -> str | None:
    for item in bundle.items:
        node = item.graph_slice.find_node(key)
        if node is not None:
            return node.name
    return None


def collect_cited(result: AskResult) -> dict[NodeKey, ScoredNode]:
    """Return the cited nodes by key: claim nodes plus the nodes of the printed tables."""
    answer, bundle = result.answer, result.bundle
    cited: dict[NodeKey, ScoredNode] = {}
    for claim in answer.claims:
        for key in claim.node_keys:
            cited[key] = ScoredNode(key=key, name=find_name(key, bundle))
    for evidence_id in answer.table_evidence_ids:
        item = bundle.get(evidence_id)
        if item is None:
            continue
        for node in item.graph_slice.nodes:
            cited[node.key] = ScoredNode(key=node.key, name=node.name)
    return cited


def compare_nodes(result: AskResult, gold: Sequence[SliceNode]) -> NodeComparison:
    """Compare the nodes `result` cites with the gold nodes, by key."""
    cited = collect_cited(result)
    gold_nodes = {node.key: ScoredNode(key=node.key, name=node.name) for node in gold}
    return NodeComparison(
        cited=sort_nodes(cited.values()),
        gold=sort_nodes(gold_nodes.values()),
        missed=sort_nodes(node for key, node in gold_nodes.items() if key not in cited),
        extra=sort_nodes(node for key, node in cited.items() if key not in gold_nodes),
    )


def compute_precision(comparison: NodeComparison) -> float:
    """Return the share of cited nodes that are gold.

    Citing nothing scores 1 against an empty gold set and 0 against a non-empty one.
    """
    if not comparison.cited:
        return 0.0 if comparison.gold else 1.0
    return (len(comparison.cited) - len(comparison.extra)) / len(comparison.cited)


def compute_recall(comparison: NodeComparison) -> float:
    """Return the share of gold nodes that are cited; an empty gold set scores 1."""
    if not comparison.gold:
        return 1.0
    return (len(comparison.gold) - len(comparison.missed)) / len(comparison.gold)


def score_answer(result: AskResult, comparison: NodeComparison) -> QuestionScore:
    """Score `result`: precision and recall from `comparison`, the rest from its stats.

    The citation is valid when the checker left no warning on the answer.
    """
    stats = result.stats
    return QuestionScore(
        precision=compute_precision(comparison),
        recall=compute_recall(comparison),
        is_citation_valid=not result.answer.warnings,
        tool_calls=stats.tool_calls,
        cypher_calls=stats.cypher_calls,
        latency_seconds=stats.total_latency_seconds,
        input_tokens=sum(step.input_tokens for step in stats.steps),
        output_tokens=sum(step.output_tokens for step in stats.steps),
    )
