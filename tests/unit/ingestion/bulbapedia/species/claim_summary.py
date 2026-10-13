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


# What each pass claims: the tables of the spec, as readable patterns.
SPECIES_CLAIM_PATTERNS: set[ClaimPattern] = {
    ("canLearn", "out", "Move", False),
    ("hasAbility", "out", "Ability", False),
    ("hasType", "out", "Type", False),
    ("inEggGroup", "out", "EggGroup", False),
    ("introducedIn", "out", "Generation", False),
    ("yieldsEv", "out", "Stat", False),
    ("hasLearnsetEntry", "out", "LearnsetEntry", True),
    ("hasForm", "out", "PokemonSpecies", True),
    ("fromSpecies", "in", "FormChange", True),
}
ENTRY_CLAIM_PATTERNS: set[ClaimPattern] = {("viaMachine", "out", "Machine", False)}
REQUIRES_CLAIM_PATTERNS: set[ClaimPattern] = {("requires", "out", None, False)}
EVOLUTION_ANCHOR_PATTERNS: set[ClaimPattern] = {
    ("evolvesTo", "out", "PokemonSpecies", False),
    ("fromSpecies", "in", "Evolution", True),
}
CHUNK_ANCHOR_PATTERNS: set[ClaimPattern] = {("describes", "in", "TextChunk", True)}
CHUNK_CLAIM_PATTERNS: set[ClaimPattern] = {
    ("fromWork", "out", "Version", False),
    ("mentionedIn", "in", None, False),
}
