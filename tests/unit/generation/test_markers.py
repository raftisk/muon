import random
import re

import pytest

from muon.generation.handles import HandleNode, HandleTable
from muon.generation.markers import MarkerRewriter
from muon.retrieval.evidence import EvidenceBundle, NodeKey

URL_A = "https://example.org/a"
URL_B = "https://example.org/b"
OPEN_MARKER_AT_END = re.compile(r"\[(n|t)\d*$")


def build_node(handle: str, name: str | None, url: str | None) -> HandleNode:
    key = NodeKey(label="Creature", id=f"{handle}-id")
    return HandleNode(handle=handle, key=key, name=name, source_url=url)


def build_table() -> HandleTable:
    nodes = [
        build_node("n2", "Water", None),
        build_node("n4", "Lapras", URL_A),
        build_node("n5", "Lapras form", URL_A),
        build_node("n9", "Surf", URL_B),
        build_node("n6", 'The "Big" one', "https://example.org/c"),
    ]
    by_handle = {node.handle: node for node in nodes}
    return HandleTable(
        bundle=EvidenceBundle(universe="toy", question="q"),
        nodes=by_handle,
        handles_by_key={node.key: node.handle for node in nodes},
        tables={"t3": "e3"},
        page_titles={URL_A: "Lapras", URL_B: "Surf"},
    )


def rewrite(chunks: list[str]) -> tuple[str, MarkerRewriter]:
    rewriter = MarkerRewriter(build_table())
    pieces = [rewriter.feed(chunk) for chunk in chunks]
    pieces.append(rewriter.finish())
    return "".join(pieces), rewriter


def test_chunks_that_split_markers_never_show_part_of_one() -> None:
    rewriter = MarkerRewriter(build_table())

    pieces = [rewriter.feed(chunk) for chunk in ["Lapras learns Surf [n", "4][n", "9]."]]
    pieces.append(rewriter.finish())

    assert "".join(pieces) == (f'Lapras learns Surf [[1]]({URL_A} "Lapras")[[2]]({URL_B} "Surf").')
    assert not any(OPEN_MARKER_AT_END.search(piece) for piece in pieces)


def test_two_markers_with_one_url_render_one_link() -> None:
    text, _ = rewrite(["[n4][n5]"])

    assert text == f'[[1]]({URL_A} "Lapras")'


def test_a_node_without_a_url_renders_nothing_and_counts_as_cited() -> None:
    text, rewriter = rewrite(["Water type [n2]."])

    assert text == "Water type."
    assert rewriter.cited().node_keys == (build_table().nodes["n2"].key,)
    assert rewriter.cited().sources == ()


def test_an_unknown_handle_is_dropped_with_one_warning() -> None:
    text, rewriter = rewrite(["Fire [n99] and [n99] again."])

    assert text == "Fire and again."
    assert rewriter.cited().warnings == ("[n99] names no node in the evidence",)


@pytest.mark.parametrize("passthrough", ["[see below]", "[1]", "[n]", "[note] and [n4x]", "[nt1]"])
def test_text_that_is_not_a_marker_passes_through(passthrough: str) -> None:
    text, rewriter = rewrite([passthrough])

    assert text == passthrough
    assert rewriter.cited().warnings == ()


def test_the_stream_ends_inside_a_marker() -> None:
    rewriter = MarkerRewriter(build_table())

    shown = rewriter.feed("Lapras [n1")
    flushed = rewriter.finish()

    assert shown == "Lapras"
    assert flushed == " [n1"
    assert rewriter.cited().warnings == ("unclosed marker '[n1' at the end of the answer",)


def test_a_page_keeps_one_number_across_the_answer() -> None:
    text, rewriter = rewrite(["Lapras [n4] and Surf [n9] and Lapras again [n4]."])

    assert text == (
        f'Lapras [[1]]({URL_A} "Lapras") and Surf [[2]]({URL_B} "Surf") '
        f'and Lapras again [[1]]({URL_A} "Lapras").'
    )
    assert [(source.number, source.url) for source in rewriter.cited().sources] == [
        (1, URL_A),
        (2, URL_B),
    ]


def test_a_group_lists_its_links_in_ascending_number_order() -> None:
    text, _ = rewrite(["[n9][n4]"])

    assert text == f'[[1]]({URL_B} "Surf")[[2]]({URL_A} "Lapras")'


def test_a_table_marker_records_the_evidence_id_and_renders_nothing() -> None:
    text, rewriter = rewrite(["All of them [t3], and [t7] too."])

    assert text == "All of them, and too."
    assert rewriter.cited().table_ids == ("e3",)
    assert rewriter.cited().warnings == ("[t7] names no table in the evidence",)


def test_a_quote_in_a_title_is_escaped() -> None:
    text, _ = rewrite(["Big [n6]"])

    assert text == 'Big [[1]](https://example.org/c "The \\"Big\\" one")'


def test_whitespace_before_a_dropped_group_is_dropped_and_before_a_link_is_kept() -> None:
    text, _ = rewrite(["a  [n2] b [n4]\n[n2]\nc"])

    assert text == f'a b [[1]]({URL_A} "Lapras")\nc'


def test_trailing_whitespace_is_returned_at_finish() -> None:
    rewriter = MarkerRewriter(build_table())

    assert rewriter.feed("done \n") == "done"
    assert rewriter.finish() == " \n"


def test_the_record_is_not_available_before_finish() -> None:
    rewriter = MarkerRewriter(build_table())
    rewriter.feed("text")

    with pytest.raises(RuntimeError, match="finish"):
        rewriter.cited()


FIXED_ANSWERS = [
    "Lapras [n4] learns Surf [n9][n4], and Water [n2] is a type. See [1] and [t3]. Done [n99]",
    "Surf [n9]; Lapras [n5] again [n4].\n\n[t3][n9] end [n7] [see] [x] [",
    "[n4][n9] start, [note] [n2] [t7] [n4][n2][n5]  ends  \n ",
]


@pytest.mark.parametrize("answer", FIXED_ANSWERS)
def test_every_split_gives_the_whole_text_result(answer: str) -> None:
    whole_text, whole = rewrite([answer])

    for position in range(len(answer) + 1):
        text, split = rewrite([answer[:position], answer[position:]])
        assert text == whole_text
        assert split.cited() == whole.cited()


@pytest.mark.parametrize("answer", FIXED_ANSWERS)
def test_random_chunk_sizes_give_the_whole_text_result_and_no_open_marker(answer: str) -> None:
    whole_text, whole = rewrite([answer])
    generator = random.Random(2026)

    for _ in range(50):
        rewriter = MarkerRewriter(build_table())
        pieces = []
        position = 0
        while position < len(answer):
            size = generator.randint(1, 7)
            pieces.append(rewriter.feed(answer[position : position + size]))
            position += size
        pieces.append(rewriter.finish())
        assert "".join(pieces) == whole_text
        assert rewriter.cited() == whole.cited()
        assert not any(OPEN_MARKER_AT_END.search(piece) for piece in pieces[:-1])


def test_a_handle_longer_than_a_marker_is_text_for_every_split() -> None:
    answer = "Over [n12345678901] and [t1234567890] but [n123456789] fits."
    whole_text, whole = rewrite([answer])

    assert whole_text.startswith("Over [n12345678901] and [t1234567890] but ")
    assert whole.cited().warnings == ("[n123456789] names no node in the evidence",)
    for position in range(len(answer) + 1):
        text, split = rewrite([answer[:position], answer[position:]])
        assert text == whole_text
        assert split.cited() == whole.cited()
