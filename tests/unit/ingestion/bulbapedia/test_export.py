import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.export import (
    ExportError,
    convert_base36_sha1,
    read_export_revisions,
)
from tests.unit.ingestion.bulbapedia.fake_export import (
    ExportFixturePage,
    build_export_xml,
    encode_base36_sha1,
    write_export,
)

TIMESTAMP = datetime(2026, 10, 20, 19, 18, 52, tzinfo=UTC)
SHADOW_BALL = ExportFixturePage(
    title="Shadow Ball (move)",
    page_id=2421,
    revision_id=4630100,
    timestamp=TIMESTAMP,
    wikitext="{{MoveInfobox|name=Shadow Ball}}\n==Effect==\nIt hurls a <shadowy> blob & more.",
)
WIND_MOVE = ExportFixturePage(
    title="Wind move",
    page_id=9001,
    revision_id=4630200,
    timestamp=TIMESTAMP,
    wikitext="A wind move is a move that wind-related Abilities react to.",
)
MOVES_CATEGORY = ExportFixturePage(
    title="Category:Moves",
    page_id=9002,
    revision_id=4630300,
    timestamp=TIMESTAMP,
    wikitext="Moves.",
    namespace=14,
)
REDIRECT = ExportFixturePage(
    title="Vice Grip (move)",
    page_id=9003,
    revision_id=4630400,
    timestamp=TIMESTAMP,
    wikitext="#REDIRECT [[Vise Grip (move)]]",
    is_redirect=True,
)
# Its SHA-1 starts with 2 zero digits, so both the hex and the base 36 form need padding.
LEADING_ZERO_TEXT = "text 43"


@pytest.mark.parametrize("text", ["abc", "Shadow Ball (move)\n", LEADING_ZERO_TEXT])
def test_convert_base36_sha1_matches_hashlib(text: str) -> None:
    hex_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()

    assert convert_base36_sha1(encode_base36_sha1(hex_sha1)) == hex_sha1


def test_leading_zero_text_pads_both_forms() -> None:
    hex_sha1 = hashlib.sha1(LEADING_ZERO_TEXT.encode("utf-8")).hexdigest()

    assert hex_sha1.startswith("00")
    assert encode_base36_sha1(hex_sha1).startswith("0")
    assert len(convert_base36_sha1(encode_base36_sha1(hex_sha1))) == 40


def test_reads_a_page_with_all_fields(tmp_path: Path) -> None:
    pages = read_export_revisions(write_export(tmp_path, [SHADOW_BALL]))

    assert len(pages) == 1
    page = pages[0]
    assert (page.namespace, page.is_redirect) == (0, False)
    revision = page.revision
    assert revision.title == SHADOW_BALL.title
    assert (revision.page_id, revision.revision_id) == (2421, 4630100)
    assert revision.timestamp == TIMESTAMP
    assert revision.wikitext == SHADOW_BALL.wikitext
    assert revision.sha1 == hashlib.sha1(SHADOW_BALL.wikitext.encode("utf-8")).hexdigest()


def test_reads_namespace_and_redirect(tmp_path: Path) -> None:
    pages = read_export_revisions(
        write_export(tmp_path, [SHADOW_BALL, WIND_MOVE, MOVES_CATEGORY, REDIRECT])
    )

    assert [(page.revision.title, page.namespace, page.is_redirect) for page in pages] == [
        (SHADOW_BALL.title, 0, False),
        (WIND_MOVE.title, 0, False),
        (MOVES_CATEGORY.title, 14, False),
        (REDIRECT.title, 0, True),
    ]


def test_reads_another_export_version(tmp_path: Path) -> None:
    path = tmp_path / "export-0.10.xml"
    path.write_text(build_export_xml([SHADOW_BALL], version="0.10"), encoding="utf-8")

    pages = read_export_revisions(path)

    assert [page.revision.title for page in pages] == [SHADOW_BALL.title]


def test_missing_file_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.xml"

    with pytest.raises(ExportError, match="missing.xml"):
        read_export_revisions(path)


def test_malformed_file_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.xml"
    path.write_text("<mediawiki><page><title>Shadow Ball", encoding="utf-8")

    with pytest.raises(ExportError, match="broken.xml"):
        read_export_revisions(path)


def test_two_revisions_fail_naming_the_page(tmp_path: Path) -> None:
    page = replace(SHADOW_BALL, extra_revisions=1)

    with pytest.raises(ExportError, match=r"Shadow Ball \(move\).*2 revisions"):
        read_export_revisions(write_export(tmp_path, [page]))


def test_page_without_revision_fails_naming_the_page(tmp_path: Path) -> None:
    path = tmp_path / "export.xml"
    xml = build_export_xml([SHADOW_BALL])
    start = xml.index("    <revision>")
    end = xml.index("    </revision>\n") + len("    </revision>\n")
    path.write_text(xml[:start] + xml[end:], encoding="utf-8")

    with pytest.raises(ExportError, match=r"Shadow Ball \(move\).*0 revisions"):
        read_export_revisions(path)


def test_wrong_sha1_fails_naming_the_page_and_both_values(tmp_path: Path) -> None:
    wrong_sha1 = encode_base36_sha1("0" * 39 + "1")
    page = replace(SHADOW_BALL, sha1=wrong_sha1)
    computed = hashlib.sha1(SHADOW_BALL.wikitext.encode("utf-8")).hexdigest()

    with pytest.raises(ExportError) as error:
        read_export_revisions(write_export(tmp_path, [page]))

    message = str(error.value)
    assert "export.xml" in message
    assert SHADOW_BALL.title in message
    assert computed in message
    assert "0" * 39 + "1" in message
