from muon.ingestion.bulbapedia.chunking import (
    MIN_PARAGRAPH_LENGTH,
    Block,
    ChunkAspect,
    MentionRef,
    RenderedBlock,
    merge_short_paragraphs,
    normalize_chunk_text,
    read_prose_section,
    remove_tags,
    render_block,
    split_blocks,
)
from muon.ingestion.bulbapedia.wikitext import split_sections

LONG = "x" * MIN_PARAGRAPH_LENGTH
SHORT = "short sentence."


def prose(text: str) -> RenderedBlock:
    return RenderedBlock(is_bullet=False, text=text, mentions=(), dropped=())


def bullet(text: str) -> RenderedBlock:
    return RenderedBlock(is_bullet=True, text=text, mentions=(), dropped=())


def test_remove_tags_drops_tables_and_galleries_with_their_blank_lines() -> None:
    wikitext = (
        "before\n\n{|\n|a\n\n|b\n|}\n\n<gallery>\nFile:A.png\n\nFile:B.png\n</gallery>\nafter"
    )

    cleaned = remove_tags(wikitext, frozenset({"table", "gallery"}))

    assert "|a" not in cleaned
    assert "File:" not in cleaned
    assert "before" in cleaned
    assert "after" in cleaned


def test_remove_tags_removes_a_table_inside_another_element() -> None:
    cleaned = remove_tags("a<div>b\n{|\n|c\n|}\n</div>d", frozenset({"table"}))

    assert "c" not in cleaned
    assert "b" in cleaned
    assert "d" in cleaned


def test_remove_tags_unwraps_layout_divs_and_keeps_their_text() -> None:
    cleaned = remove_tags(
        'a<div class="box">kept\n\nalso kept</div>b<center>c</center>',
        frozenset({"table"}),
        frozenset({"div", "center"}),
    )

    assert cleaned == "akept\n\nalso keptbc"


def test_remove_tags_leaves_nothing_of_a_div_around_a_removed_table() -> None:
    wikitext = '<div class="scrollbox">\n{|\n|a\n\n|b\n|}\n<div>\n{|\n|c\n|}\n</div></div>'

    cleaned = remove_tags(wikitext, frozenset({"table"}), frozenset({"div"}))

    assert cleaned.strip() == ""


def lines(blocks: tuple[Block, ...]) -> list[tuple[bool, tuple[str, ...]]]:
    return [(block.is_bullet, block.lines) for block in blocks]


def test_split_blocks_cuts_paragraphs_on_blank_lines() -> None:
    blocks = split_blocks("First line\nsecond line\n\n\nSecond paragraph")

    assert lines(blocks) == [
        (False, ("First line", "second line")),
        (False, ("Second paragraph",)),
    ]


def test_split_blocks_gives_a_bullet_item_its_sub_bullets() -> None:
    blocks = split_blocks("* one\n** one a\n** one b\n* two\n*** two deep")

    assert lines(blocks) == [
        (True, ("* one", "** one a", "** one b")),
        (True, ("* two", "*** two deep")),
    ]


def test_split_blocks_separates_prose_and_bullets_in_one_block() -> None:
    blocks = split_blocks("intro\n* item\ntail line\nmore tail")

    assert lines(blocks) == [
        (False, ("intro",)),
        (True, ("* item",)),
        (False, ("tail line", "more tail")),
    ]


def test_split_blocks_opens_an_item_for_a_deep_line_without_one() -> None:
    assert lines(split_blocks("** orphan")) == [(True, ("** orphan",))]


def test_split_blocks_removes_tables_and_galleries() -> None:
    blocks = split_blocks(
        "text\n\n{|\n|cell\n\n|cell two\n|}\n\n<gallery>\nFile:A.png\n</gallery>\n"
    )

    assert lines(blocks) == [(False, ("text",))]


def test_split_blocks_leaves_no_div_text_around_a_removed_table() -> None:
    blocks = split_blocks(
        'intro\n\n<div class="scrollbox">\n{|\n|cell\n\n|cell\n|}\n</div>\n\nouter'
    )

    assert lines(blocks) == [(False, ("intro",)), (False, ("outer",))]


def test_split_blocks_removes_comments_without_splitting_a_paragraph() -> None:
    blocks = split_blocks("one\n<!-- note\n\nmore note -->\ntwo <!-- inline --> three")

    assert lines(blocks) == [(False, ("one", "two  three"))]


def test_split_blocks_gives_nothing_for_blank_text() -> None:
    assert split_blocks(" \n\n  \n") == ()


def test_normalize_chunk_text_replaces_line_breaks_and_removes_soft_hyphens() -> None:
    assert normalize_chunk_text("a\nb\fc  d­e \t f") == "a b c de f"


def test_render_block_strips_bullet_marks_and_joins_lines() -> None:
    block = Block(True, ("* first line", "** sub one", "*** sub two"))

    assert render_block(block).text == "first line sub one sub two"


def test_render_block_removes_references_and_keeps_display_text() -> None:
    block = Block(False, ("It digs{{rf|1}}<ref>cite</ref> {{wp|mole (animal)|mole}} caves.",))

    assert render_block(block).text == "It digs1 mole caves."


def test_render_block_collects_mentions_with_their_targets_not_their_display_text() -> None:
    block = Block(
        False,
        ("{{p|Dugtrio|its evolution}} learns {{m|Dig}} and has {{a|Tangling Hair}}.",),
    )

    rendered = render_block(block)

    assert rendered.text == "its evolution learns Dig and has Tangling Hair."
    assert rendered.mentions == (
        MentionRef(template="p", target="Dugtrio"),
        MentionRef(template="m", target="Dig"),
        MentionRef(template="a", target="Tangling Hair"),
    )


def test_render_block_ignores_mentions_in_removed_parts() -> None:
    block = Block(False, ("Text<ref>{{p|Onix}}</ref><small>{{m|Dig}}</small>.",))

    assert render_block(block).mentions == ()


def test_render_block_reports_dropped_templates() -> None:
    assert render_block(Block(False, ("a {{roundy|x}} b",))).dropped == ("roundy",)


def test_render_block_renders_the_text_templates_of_the_chunk_rules() -> None:
    block = Block(False, ("<sc>Diglett</sc> is {{tt|one yard|1 meter}} tall.<br>{{ScPkmn}}",))

    assert render_block(block).text == "Diglett is one yard tall. Pokémon"


def test_merge_keeps_long_paragraphs_and_all_bullets_unchanged() -> None:
    blocks = [prose(LONG), bullet(SHORT), prose(LONG + "y")]

    assert merge_short_paragraphs(blocks) == tuple(blocks)


def test_merge_joins_a_short_paragraph_into_the_next_one() -> None:
    merged = merge_short_paragraphs([prose(SHORT), prose(LONG)])

    assert [block.text for block in merged] == [f"{SHORT} {LONG}"]


def test_merge_repeats_while_the_result_stays_short() -> None:
    merged = merge_short_paragraphs([prose(SHORT), prose(SHORT), prose(LONG)])

    assert [block.text for block in merged] == [f"{SHORT} {SHORT} {LONG}"]


def test_merge_joins_a_trailing_short_paragraph_into_the_one_before() -> None:
    merged = merge_short_paragraphs([prose(LONG), prose(SHORT)])

    assert [block.text for block in merged] == [f"{LONG} {SHORT}"]


def test_merge_keeps_a_lone_short_paragraph() -> None:
    assert [block.text for block in merge_short_paragraphs([prose(SHORT)])] == [SHORT]


def test_merge_keeps_short_paragraphs_together_when_nothing_else_remains() -> None:
    merged = merge_short_paragraphs([prose(SHORT), prose(SHORT)])

    assert [block.text for block in merged] == [f"{SHORT} {SHORT}"]


def test_merge_skips_bullets_when_looking_for_the_next_paragraph() -> None:
    merged = merge_short_paragraphs([prose(SHORT), bullet(SHORT), prose(LONG)])

    assert [(block.is_bullet, block.text) for block in merged] == [
        (True, SHORT),
        (False, f"{SHORT} {LONG}"),
    ]


def test_merge_does_not_join_a_trailing_short_paragraph_into_a_bullet() -> None:
    merged = merge_short_paragraphs([bullet(LONG), prose(SHORT)])

    assert [(block.is_bullet, block.text) for block in merged] == [(True, LONG), (False, SHORT)]


def test_merge_keeps_the_mentions_of_both_paragraphs_in_order() -> None:
    first = RenderedBlock(False, SHORT, (MentionRef(template="p", target="Onix"),), ("a",))
    second = RenderedBlock(False, LONG, (MentionRef(template="m", target="Dig"),), ("b",))

    (merged,) = merge_short_paragraphs([first, second])

    assert [mention.target for mention in merged.mentions] == ["Onix", "Dig"]
    assert merged.dropped == ("a", "b")


def test_merge_measures_the_length_at_the_minimum() -> None:
    exact = prose("y" * MIN_PARAGRAPH_LENGTH)
    below = prose("y" * (MIN_PARAGRAPH_LENGTH - 1))

    assert merge_short_paragraphs([exact, prose(SHORT)])[0].text == f"{exact.text} {SHORT}"
    assert [block.text for block in merge_short_paragraphs([below, exact])] == [
        f"{below.text} {exact.text}"
    ]


def test_read_prose_section_gives_the_subject_aspect_and_section_path() -> None:
    wikitext = f"==Effect==\n===Generation IV===\n{LONG}\n\n{SHORT}\n\n* {{{{m|Dig}}}}"
    section = split_sections(wikitext)[-1]

    reading = read_prose_section(section, ChunkAspect.EFFECT, "shadow-ball")

    assert [(c.subject_id, c.aspect, c.section_path, c.text) for c in reading.chunks] == [
        ("shadow-ball", ChunkAspect.EFFECT, "Effect > Generation IV", f"{LONG} {SHORT}"),
        ("shadow-ball", ChunkAspect.EFFECT, "Effect > Generation IV", "Dig"),
    ]
    assert reading.chunks[1].mentions == (MentionRef(template="m", target="Dig"),)


def test_read_prose_section_reports_dropped_templates() -> None:
    section = split_sections("==Trivia==\n* a {{roundy|x}} b")[-1]

    assert read_prose_section(section, ChunkAspect.TRIVIA, "dig").dropped == ("roundy",)
