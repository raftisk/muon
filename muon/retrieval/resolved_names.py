"""The resolved names of a question and the block that hands them to the retrieve step.

Kept apart from `muon.retrieval.resolve` so that `retrieve` can render the block without
importing the module that imports it.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from muon.retrieval.evidence import NodeKey

SCORE_DIGITS = 1
BLOCK_HEADER = "Resolved names (full-text match on name and aliases, best first):"
NO_MATCH_TEXT = "no node matched"
CANDIDATE_SEPARATOR = "; "


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
