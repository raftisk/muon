from types import SimpleNamespace

from muon.ingestion.bulbapedia.runs import serialize_batch


def build_fake_batch(claims: list[dict[str, object]]) -> SimpleNamespace:
    data = {"scope": ["z", "a"], "claims": claims}
    return SimpleNamespace(scope=frozenset({"z", "a"}), model_dump=lambda mode: data)


def test_claim_without_an_id_prefix_leaves_the_key_out() -> None:
    claim = {"anchor": "x", "other_end_id_prefix": None}

    data = serialize_batch(build_fake_batch([claim]))  # type: ignore[arg-type]

    assert data["claims"] == [{"anchor": "x"}]
    assert data["scope"] == ["a", "z"]


def test_claim_with_an_id_prefix_keeps_the_key() -> None:
    claim = {"anchor": "x", "other_end_id_prefix": "cave:"}

    data = serialize_batch(build_fake_batch([claim]))  # type: ignore[arg-type]

    assert data["claims"] == [claim]


def test_claim_that_never_had_the_key_is_unchanged() -> None:
    data = serialize_batch(build_fake_batch([{"anchor": "x"}]))  # type: ignore[arg-type]

    assert data["claims"] == [{"anchor": "x"}]
