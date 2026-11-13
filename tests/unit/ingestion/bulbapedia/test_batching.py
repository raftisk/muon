import pytest

from muon.ingestion.batch import NodeRecord
from muon.ingestion.bulbapedia.batching import (
    ABILITY,
    FROM_WORK,
    MENTIONED_IN,
    MOVE,
    SOURCE,
    TYPE,
    VERSION_GROUP,
    AboxIndex,
    ChunkBuildError,
    MentionContext,
    build_chunk_record,
    build_machine_id,
    build_mention_parts,
    build_stub,
    build_work_parts,
    describe_unresolved,
    hash_chunk_text,
    number_chunks,
    resolve_mention,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect, ChunkFact, MentionRef
from muon.ingestion.bulbapedia.species.batches import build_machine_stub

SHADOW_BALL = build_stub(MOVE, "Shadow Ball")
ABOX = AboxIndex(
    ids={(TYPE, "Ghost"): "ghost", (VERSION_GROUP, "Scarlet-Violet"): "scarlet-violet"},
    region_generations={},
)
MUK = NodeRecord(class_iri="https://muon.dev/ns/pkmn#PokemonSpecies", id="muk", name="Muk")


def resolve_known_species(target: str) -> NodeRecord | None:
    return MUK if target == "Muk" else None


CONTEXT = MentionContext(abox=ABOX, resolve_species=resolve_known_species)


def build_fact(aspect: ChunkAspect, text: str, *mentions: MentionRef) -> ChunkFact:
    return ChunkFact(
        subject_id="shadow-ball",
        aspect=aspect,
        section_path="Effect",
        text=text,
        mentions=mentions,
        versions=("Scarlet-Violet", "Pokopia") if aspect is ChunkAspect.FLAVOR else (),
    )


@pytest.mark.parametrize(
    ("kind", "number_text", "version_group_id", "expected"),
    [
        ("TM", "149", "scarlet-violet", "tm149-scarlet-violet"),
        ("TR", "33", "sword-shield", "tr33-sword-shield"),
        ("TM", "074", "legends-z-a", "tm074-legends-z-a"),
        ("HM", "05", "diamond-pearl", "hm05-diamond-pearl"),
        ("TM", "00", "sword-shield", "tm00-sword-shield"),
    ],
)
def test_machine_id_keeps_the_digits_as_written(
    kind: str, number_text: str, version_group_id: str, expected: str
) -> None:
    assert build_machine_id(kind, number_text, version_group_id) == expected


def test_species_machine_stub_and_move_owner_ids_agree() -> None:
    assert build_machine_stub("TM149").id == build_machine_id("TM", "149", "scarlet-violet")


def test_number_chunks_counts_effect_chunks_and_hashes_flavor_texts() -> None:
    facts = [
        build_fact(ChunkAspect.EFFECT, "First."),
        build_fact(ChunkAspect.FLAVOR, "Hurls a blob."),
        build_fact(ChunkAspect.EFFECT, "Second."),
    ]

    labels = [label for _, label in number_chunks("Shadow Ball (move)", facts)]

    assert labels == ["1", hash_chunk_text("Hurls a blob."), "2"]


def test_number_chunks_fails_when_2_flavor_texts_share_a_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "muon.ingestion.bulbapedia.batching.hash_chunk_text", lambda text: "deadbeef"
    )
    facts = [build_fact(ChunkAspect.FLAVOR, "One."), build_fact(ChunkAspect.FLAVOR, "Two.")]

    with pytest.raises(ChunkBuildError, match="texts share the hash deadbeef"):
        number_chunks("Shadow Ball (move)", facts)


def test_build_chunk_record_gives_a_flavor_chunk_a_hash_id_and_no_source() -> None:
    fact = build_fact(ChunkAspect.FLAVOR, "Hurls a blob.")
    digest = hash_chunk_text(fact.text)

    chunk = build_chunk_record(fact, digest, SHADOW_BALL, "url")

    assert chunk.id == f"bulbapedia:move:shadow-ball:flavor:{digest}"
    assert SOURCE not in chunk.properties


def test_build_chunk_record_gives_an_effect_chunk_an_ordinal_id_and_a_source() -> None:
    chunk = build_chunk_record(
        build_fact(ChunkAspect.EFFECT, "Deals damage."), "1", SHADOW_BALL, "url"
    )

    assert chunk.id == "bulbapedia:move:shadow-ball:effect:1"
    assert chunk.properties[SOURCE] == "bulbapedia"


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        (MentionRef(template="p", target="Muk"), ("muk", "Muk")),
        (MentionRef(template="p", target="Mukk"), None),
        (MentionRef(template="m", target="Night Shade"), ("night-shade", "Night Shade")),
        (MentionRef(template="a", target="Bulletproof"), ("bulletproof", "Bulletproof")),
        (MentionRef(template="t", target="Ghost"), ("ghost", "Ghost")),
        (MentionRef(template="type", target="Stellar"), None),
    ],
)
def test_resolve_mention(mention: MentionRef, expected: tuple[str, str] | None) -> None:
    resolved = resolve_mention(mention, CONTEXT)

    assert (None if resolved is None else (resolved.id, resolved.name)) == expected


def test_build_mention_parts_skips_the_subject_and_returns_the_misses() -> None:
    fact = build_fact(
        ChunkAspect.EFFECT,
        "Text.",
        MentionRef(template="m", target="Shadow Ball"),
        MentionRef(template="a", target="Bulletproof"),
        MentionRef(template="p", target="Nobody"),
    )
    chunk = build_chunk_record(fact, "1", SHADOW_BALL, "url")

    parts, unresolved = build_mention_parts(fact, chunk, SHADOW_BALL, CONTEXT)

    assert [(e.property_iri, e.start.id) for e in parts.edges] == [(MENTIONED_IN, "bulletproof")]
    assert parts.nodes[0].class_iri == ABILITY
    assert unresolved == (MentionRef(template="p", target="Nobody"),)
    assert describe_unresolved(unresolved[0]) == "{{p|Nobody}} names no species"


def test_build_work_parts_links_known_version_groups_and_returns_the_rest() -> None:
    fact = build_fact(ChunkAspect.FLAVOR, "Hurls a blob.")
    chunk = build_chunk_record(fact, hash_chunk_text(fact.text), SHADOW_BALL, "url")

    parts, missing = build_work_parts(fact, chunk, ABOX, VERSION_GROUP)

    assert [(e.property_iri, e.end.id) for e in parts.edges] == [(FROM_WORK, "scarlet-violet")]
    assert missing == ("Pokopia",)
