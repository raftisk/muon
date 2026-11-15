from muon.generation.answer import AnswerDraft, Claim
from muon.generation.check import check_citations
from muon.retrieval.evidence import (
    Evidence,
    EvidenceBundle,
    EvidenceStatus,
    GraphSlice,
    NodeKey,
    SliceNode,
)

SPECIES = NodeKey(label="PokemonSpecies", id="diglett")
TYPE = NodeKey(label="Type", id="ground")
OTHER = NodeKey(label="Type", id="fire")


def build_evidence(evidence_id: str, status: EvidenceStatus, keys: list[NodeKey]) -> Evidence:
    nodes = tuple(SliceNode(key=key, name=None, labels=(), source_url=None) for key in keys)
    return Evidence(
        id=evidence_id,
        universe="pkmn",
        tool="run_cypher",
        query="q",
        purpose="p",
        status=status,
        graph_slice=GraphSlice(nodes=nodes),
    )


BUNDLE = EvidenceBundle(
    universe="pkmn",
    question="q",
    items=(
        build_evidence("e1", EvidenceStatus.OK, [SPECIES, TYPE]),
        build_evidence("e2", EvidenceStatus.EMPTY, []),
        build_evidence("e3", EvidenceStatus.REJECTED, []),
        build_evidence("e4", EvidenceStatus.FAILED, []),
    ),
)


def build_draft(
    claim: Claim | None = None,
    text: str = "Diglett is Ground [c1].",
    tables: list[str] | None = None,
) -> AnswerDraft:
    claim = claim or build_claim()
    return AnswerDraft(text=text, claims=[claim], assumptions=[], table_evidence_ids=tables or [])


def build_claim(
    evidence_ids: list[str] | None = None, node_keys: list[NodeKey] | None = None
) -> Claim:
    return Claim(
        id="c1",
        text="Diglett is Ground type",
        evidence_ids=["e1"] if evidence_ids is None else evidence_ids,
        node_keys=[SPECIES, TYPE] if node_keys is None else node_keys,
    )


def test_a_clean_draft_has_no_issue() -> None:
    assert check_citations(build_draft(), BUNDLE) == []


def test_an_empty_evidence_item_can_be_cited() -> None:
    draft = build_draft(build_claim(["e2"], []), text="Nothing found [c1].")

    assert check_citations(draft, BUNDLE) == []


def test_an_absent_evidence_id_is_one_issue() -> None:
    issues = check_citations(build_draft(build_claim(["e9"], [])), BUNDLE)

    assert [issue.reason for issue in issues] == ["c1 cites e9, which does not exist"]


def test_a_node_key_outside_the_cited_slices_is_one_issue() -> None:
    issues = check_citations(build_draft(build_claim(["e1"], [OTHER])), BUNDLE)

    assert len(issues) == 1
    assert "Type fire" in issues[0].reason


def test_rejected_and_failed_items_are_one_issue_each() -> None:
    issues = check_citations(build_draft(build_claim(["e3", "e4"], [])), BUNDLE)

    assert [issue.reason for issue in issues] == [
        "c1 cites e3, which is rejected",
        "c1 cites e4, which is failed",
    ]


def test_a_marker_with_no_claim_is_one_issue() -> None:
    issues = check_citations(build_draft(text="Diglett [c1] and more [c2] [c2]."), BUNDLE)

    assert [issue.reason for issue in issues] == ["The text uses [c2], which has no claim"]


def test_a_claim_with_no_evidence_id_is_one_issue() -> None:
    issues = check_citations(build_draft(build_claim([], [])), BUNDLE)

    assert [issue.reason for issue in issues] == ["c1 cites no evidence"]


def test_an_unknown_table_id_is_one_issue() -> None:
    issues = check_citations(build_draft(tables=["e1", "e7"]), BUNDLE)

    assert [issue.claim_id for issue in issues] == [None]
