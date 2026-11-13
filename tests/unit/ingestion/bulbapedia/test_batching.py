import pytest

from muon.ingestion.bulbapedia.batching import build_machine_id
from muon.ingestion.bulbapedia.species.batches import build_machine_stub


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
