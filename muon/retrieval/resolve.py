"""The resolve step: finds the names in a question and looks them up in the name index.

Map: `resolve_names` runs `find_names` (1 structured model call that returns the names as the
user wrote them, each with a label guess) and `look_up_names` (1 read query on the
full-text index for all names). `build_name_query` writes the Lucene query of 1 name,
`rank_candidates` orders and caps its matches, and `render_resolved_names` writes the block that
`retrieve` puts before the question.

Hazard: the query text, the index constants of `muon.graph.indexes` and the index created by
`muon.ingestion.schema` must stay in sync.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import openai
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from muon.agent.routing import AgentRuntime, UniverseScope
from muon.config import AgentStep
from muon.graph import GraphClient, GraphNode, GraphQueryError
from muon.graph.indexes import ALIASES_FIELD, NAME_FIELD, NAME_INDEX
from muon.llm.client import ModelCallError
from muon.retrieval.cypher import ID_PROPERTY
from muon.retrieval.evidence import NodeKey
from muon.retrieval.retrieve import StepStats, fill_prompt, read_step_usage
from muon.retrieval.schema import find_leaf_label

PROMPT_PATH = Path(__file__).parent / "prompts" / "resolve.md"
RESOLVE_STEP = AgentStep.RESOLVE.value
MIN_NAME_SCORE = 3.7  # largest value that keeps every expected node of the calibration set
FUZZY_MIN_LENGTH = 4
FUZZY_EDITS = 1
NAME_BOOST = 2
LUCENE_SPECIAL_CHARACTERS = frozenset('+-&|!(){}[]^"~*?:\\/')
SCORE_DIGITS = 1
FETCH_LIMIT = 50  # rows kept per name, by score, before ranking
MAX_CANDIDATES = 15  # candidates kept per name, after ranking
INDEX_SCRIPT = "scripts/create_name_index.py"
BLOCK_HEADER = "Resolved names (full-text match on name and aliases, best first):"
NO_MATCH_TEXT = "no node matched"
CANDIDATE_SEPARATOR = "; "

LOOK_UP_NAMES = """
UNWIND $queries AS query
CALL db.index.fulltext.queryNodes($index, query.text) YIELD node, score
WHERE $universe IN labels(node) AND score >= $min_score
WITH query.position AS position, node, score
ORDER BY position, score DESC
WITH position, collect({node: node, score: score})[0..$fetch_limit] AS hits
UNWIND hits AS hit
RETURN position, hit.node AS node, hit.score AS score
"""
SHOW_NAME_INDEX = "SHOW INDEXES YIELD name WHERE name = $name"


class NameIndexError(Exception):
    """The full-text index on names is missing from the database."""


class NameMention(BaseModel):
    """A name as written in the question, with the label the model guessed ("" for none)."""

    model_config = ConfigDict(extra="forbid")

    mention: str
    label: str


class NameMentions(BaseModel):
    """The structured reply of the model: every name of the question."""

    model_config = ConfigDict(extra="forbid")

    names: list[NameMention]


class NameCandidate(BaseModel):
    """A node the index matched to a name. `score` is the raw index score."""

    key: NodeKey
    name: str
    score: float


class ResolvedName(BaseModel):
    """A name with its ranked candidates. An empty list means no node matched."""

    mention: str
    label: str
    candidates: list[NameCandidate]


@dataclass(frozen=True)
class ResolveResult:
    names: list[ResolvedName]
    stats: StepStats


def escape_lucene(word: str) -> str:
    return "".join(f"\\{char}" if char in LUCENE_SPECIAL_CHARACTERS else char for char in word)


def build_name_query(mention: str) -> str:
    """Return the Lucene query of 1 name, or `""` when the name has no letter or digit.

    Words of `FUZZY_MIN_LENGTH` or more letters and digits allow `FUZZY_EDITS` edit, shorter
    words match exactly, and a match in the name field weighs `NAME_BOOST` times a match in
    the aliases.
    """
    terms = []
    for word in mention.split():
        length = sum(char.isalnum() for char in word)
        if length == 0:
            continue
        escaped = escape_lucene(word)
        terms.append(f"{escaped}~{FUZZY_EDITS}" if length >= FUZZY_MIN_LENGTH else escaped)
    if not terms:
        return ""
    joined = " ".join(terms)
    return f"{NAME_FIELD}:({joined})^{NAME_BOOST} {ALIASES_FIELD}:({joined})"


def rank_candidates(candidates: Sequence[NameCandidate], label: str) -> list[NameCandidate]:
    """Put the candidates whose leaf label is `label` first, then order by score, and cap.

    The cap is `MAX_CANDIDATES`. No candidate is dropped for its label.
    """
    ranked = sorted(candidates, key=lambda item: (item.key.label != label, -item.score))
    return ranked[:MAX_CANDIDATES]


def build_candidate(node: GraphNode, score: float, scope: UniverseScope) -> NameCandidate:
    leaf = find_leaf_label(
        sorted(node.labels), scope.materializer.map_ancestor_labels(), scope.universe_label
    )
    node_id = node.properties.get(ID_PROPERTY)
    name = node.properties.get(NAME_FIELD)
    return NameCandidate(
        key=NodeKey(label=leaf, id=str(node_id) if node_id is not None else node.element_id),
        name=name if isinstance(name, str) else "",
        score=score,
    )


async def find_names(
    question: str, scope: UniverseScope, runtime: AgentRuntime
) -> tuple[list[NameMention], StepStats]:
    """Ask the resolve model for the names in `question`, with a label guess for each.

    A label outside the leaf labels of the scope becomes `""`. Raises `ModelCallError` on a
    provider error and on a reply that does not parse.
    """
    labels = [node_label.label for node_label in scope.schema_context.graph_schema.node_labels]
    model_name = runtime.agent_settings.model_for(AgentStep.RESOLVE)
    system_prompt = fill_prompt(
        PROMPT_PATH, {"universe_name": scope.plugin.name, "labels": ", ".join(labels)}
    )
    structured = runtime.resolve_model.with_structured_output(
        NameMentions, method="json_schema", strict=True, include_raw=True
    )
    started_at = time.perf_counter()
    try:
        # With include_raw=True the result is a dict, which the stubs do not say.
        output = cast(
            dict[str, Any],
            await structured.ainvoke([SystemMessage(system_prompt), HumanMessage(question)]),
        )
    except openai.APIError as error:
        raise ModelCallError(RESOLVE_STEP, model_name, str(error)) from error
    latency = time.perf_counter() - started_at
    parsed = output["parsed"]
    if parsed is None:
        raise ModelCallError(
            RESOLVE_STEP,
            model_name,
            f"the reply is not a valid name list: {output['parsing_error']}",
        )
    usage = read_step_usage([output["raw"]])
    stats = StepStats(
        step=RESOLVE_STEP,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        model_calls=usage.model_calls,
        latency_seconds=latency,
    )
    mentions = NameMentions.model_validate(parsed, from_attributes=True).names
    return [
        NameMention(mention=item.mention, label=item.label if item.label in labels else "")
        for item in mentions
    ], stats


async def check_name_index(client: GraphClient, cause: GraphQueryError) -> None:
    """Raise `NameIndexError` when the index is absent; otherwise return."""
    rows = await client.execute_read(SHOW_NAME_INDEX, {"name": NAME_INDEX})
    if not rows:
        raise NameIndexError(
            f"The full-text index {NAME_INDEX!r} does not exist; "
            f"create it with `uv run python {INDEX_SCRIPT}`"
        ) from cause


async def look_up_names(
    mentions: Sequence[NameMention], scope: UniverseScope, client: GraphClient
) -> list[ResolvedName]:
    """Look every mention up in the name index with 1 read query and rank the matches.

    Sends no query when `mentions` is empty. Raises `NameIndexError` when the query fails
    because the index is missing, and re-raises any other `GraphQueryError`.
    """
    queries = [
        {"position": position, "text": build_name_query(item.mention)}
        for position, item in enumerate(mentions)
    ]
    queries = [query for query in queries if query["text"]]
    rows: list[dict[str, Any]] = []
    if queries:
        parameters = {
            "queries": queries,
            "index": NAME_INDEX,
            "universe": scope.universe_label,
            "min_score": MIN_NAME_SCORE,
            "fetch_limit": FETCH_LIMIT,
        }
        try:
            rows = await client.execute_read(LOOK_UP_NAMES, parameters)
        except GraphQueryError as error:
            await check_name_index(client, error)
            raise
    found: dict[int, list[NameCandidate]] = {}
    for row in rows:
        candidate = build_candidate(row["node"], row["score"], scope)
        found.setdefault(row["position"], []).append(candidate)
    return [
        ResolvedName(
            mention=item.mention,
            label=item.label,
            candidates=rank_candidates(found.get(position, []), item.label),
        )
        for position, item in enumerate(mentions)
    ]


async def resolve_names(
    question: str, scope: UniverseScope, runtime: AgentRuntime
) -> ResolveResult:
    """Find the names in `question` and resolve each to ranked graph nodes."""
    mentions, stats = await find_names(question, scope, runtime)
    names = await look_up_names(mentions, scope, runtime.client)
    return ResolveResult(names=names, stats=stats)


def render_candidate(candidate: NameCandidate) -> str:
    score = f"{candidate.score:.{SCORE_DIGITS}f}"
    return f"{candidate.name} [{candidate.key.label} {candidate.key.id}, score {score}]"


def render_resolved_names(names: Sequence[ResolvedName]) -> str:
    """Return the block that lists each name with its candidates, or `""` for no names."""
    if not names:
        return ""
    lines = [BLOCK_HEADER]
    for item in names:
        guess = f" (guess: {item.label})" if item.label else ""
        matches = (
            CANDIDATE_SEPARATOR.join(render_candidate(candidate) for candidate in item.candidates)
            or NO_MATCH_TEXT
        )
        lines.append(f'- "{item.mention}"{guess}: {matches}')
    return "\n".join(lines)
