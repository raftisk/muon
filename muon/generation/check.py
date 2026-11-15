"""The citation checker: tests a draft against the evidence bundle it must rest on."""

from dataclasses import dataclass

from muon.generation.answer import CLAIM_MARKER_PATTERN, AnswerDraft, Claim
from muon.retrieval.evidence import Evidence, EvidenceBundle, EvidenceStatus

UNUSABLE_STATUSES = frozenset({EvidenceStatus.REJECTED, EvidenceStatus.FAILED})


@dataclass(frozen=True)
class CitationIssue:
    """One miss of a draft. `claim_id` is `None` when no claim owns the miss."""

    claim_id: str | None
    reason: str


def find_cited_evidence(
    claim: Claim, bundle: EvidenceBundle, issues: list[CitationIssue]
) -> list[Evidence]:
    """Return the usable evidence items of `claim`, and add an issue per unusable id."""
    cited: list[Evidence] = []
    for evidence_id in claim.evidence_ids:
        evidence = bundle.get(evidence_id)
        if evidence is None:
            issues.append(
                CitationIssue(claim.id, f"{claim.id} cites {evidence_id}, which does not exist")
            )
        elif evidence.status in UNUSABLE_STATUSES:
            issues.append(
                CitationIssue(
                    claim.id, f"{claim.id} cites {evidence_id}, which is {evidence.status}"
                )
            )
        else:
            cited.append(evidence)
    return cited


def check_claim(claim: Claim, bundle: EvidenceBundle) -> list[CitationIssue]:
    if not claim.evidence_ids:
        return [CitationIssue(claim.id, f"{claim.id} cites no evidence")]
    issues: list[CitationIssue] = []
    cited = find_cited_evidence(claim, bundle, issues)
    if not cited:
        return issues
    for key in claim.node_keys:
        if not any(evidence.graph_slice.find_node(key) is not None for evidence in cited):
            issues.append(
                CitationIssue(
                    claim.id,
                    f"{claim.id} cites node {key.label} {key.id}, "
                    "which is in no cited evidence item",
                )
            )
    return issues


def check_citations(draft: AnswerDraft, bundle: EvidenceBundle) -> list[CitationIssue]:
    """Return 1 issue per citation miss of `draft`; an empty list means the draft is clean."""
    issues: list[CitationIssue] = []
    for claim in draft.claims:
        issues.extend(check_claim(claim, bundle))
    claim_ids = {claim.id for claim in draft.claims}
    for marker in dict.fromkeys(CLAIM_MARKER_PATTERN.findall(draft.text)):
        if marker not in claim_ids:
            issues.append(CitationIssue(marker, f"The text uses [{marker}], which has no claim"))
    for evidence_id in draft.table_evidence_ids:
        if bundle.get(evidence_id) is None:
            issues.append(
                CitationIssue(None, f"The table list names {evidence_id}, which does not exist")
            )
    return issues
