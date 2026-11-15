"""The answer models.

`AnswerDraft` is what the model returns under strict structured output, so every field is
required and nothing has a default. `Answer` adds what code sets after the check.
"""

import re

from pydantic import BaseModel, ConfigDict

from muon.retrieval.evidence import NodeKey

CLAIM_MARKER_PATTERN = re.compile(r"\[(c\d+)\]")


class AnswerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(AnswerModel):
    """One statement of the answer with the evidence it rests on."""

    id: str
    text: str
    evidence_ids: list[str]
    node_keys: list[NodeKey]


class AnswerDraft(AnswerModel):
    """The generator's output: prose with `[c1]` markers, its claims and its readings."""

    text: str
    claims: list[Claim]
    assumptions: list[str]
    table_evidence_ids: list[str]


class Answer(AnswerDraft):
    """A checked draft with its universe and the citation warnings that remain."""

    universe: str
    warnings: list[str]
