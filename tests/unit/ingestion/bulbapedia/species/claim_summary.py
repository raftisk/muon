"""Readable claim sets for the species batch tests."""

from collections import defaultdict

from muon.ingestion.batch import GraphBatch
from muon.ontology.model import extract_local_name

# A claim without its anchor: property, direction, other class (None for any), owns other end.
ClaimPattern = tuple[str, str, str | None, bool]


def claims_by_anchor(batch: GraphBatch) -> dict[str, set[ClaimPattern]]:
    """Map each anchor `id` to the patterns it claims, with IRIs cut to their local names."""
    patterns: defaultdict[str, set[ClaimPattern]] = defaultdict(set)
    for claim in batch.claims:
        other_class = (
            None if claim.other_class_iri is None else extract_local_name(claim.other_class_iri)
        )
        patterns[claim.anchor.id].add(
            (
                extract_local_name(claim.property_iri),
                claim.direction.value,
                other_class,
                claim.owns_other_end,
            )
        )
    return dict(patterns)
