"""The marker rewriter: turns citation handles in streamed text into numbered source links.

Map: `MarkerRewriter.feed` appends a chunk to the pending text and `drain` reads it left to
right. The pending text is always in 1 of 4 states at its end:
- plain text, which is emitted at once;
- an open marker (`[`, `[n`, `[n12`), held until the marker closes or stops being a marker;
- a closed marker group (`[n1][n2]`), held until the next character shows whether the group
  goes on, because a group renders its links sorted and deduplicated;
- whitespace, held until the next non-space character, because whitespace before a group
  that renders nothing is dropped.
`finish` drains with no holding. Chunk boundaries never change the output or the record.
"""

import re
from dataclasses import dataclass
from typing import Literal

from muon.generation.contract import CitedHandles, Source
from muon.generation.handles import NODE_HANDLE_PREFIX, HandleNode, HandleTable
from muon.retrieval.evidence import NodeKey

MAX_MARKER_DIGITS = 9
MARKER_PATTERN = re.compile(rf"\[(n|t)(\d{{1,{MAX_MARKER_DIGITS}}})\]")
MAX_MARKER_LENGTH = MAX_MARKER_DIGITS + 3  # "[n", the digits and "]"
OPEN_MARKER_PATTERN = re.compile(r"\[(?:[nt]\d*)?")
UNCLOSED_MARKER_PATTERN = re.compile(r"\[[nt]\d*")
PLAIN_TEXT_PATTERN = re.compile(r"[^\[\s]+")
MARKER_START = "["

type GroupStatus = Literal["group", "text", "more"]


@dataclass(frozen=True)
class GroupScan:
    """The result of reading a marker group at a position: its markers and where it ends."""

    status: GroupStatus
    markers: tuple[str, ...] = ()
    end: int = 0


def is_open_marker(text: str) -> bool:
    """Return whether `text` could still become a marker when more characters arrive."""
    return len(text) < MAX_MARKER_LENGTH and OPEN_MARKER_PATTERN.fullmatch(text) is not None


def find_whitespace_end(text: str, start: int) -> int:
    end = start
    while end < len(text) and text[end].isspace():
        end += 1
    return end


def scan_group(text: str, start: int, is_final: bool) -> GroupScan:
    """Read the markers that start at `text[start]`, which is `[`.

    The status is `more` when the text so far cannot decide the group, `group` when it holds
    at least 1 marker and is closed, and `text` when `[` does not start a marker.
    """
    markers: list[str] = []
    position = start
    while match := MARKER_PATTERN.match(text, position):
        markers.append(match.group(0))
        position = match.end()
    remainder = text[position:]
    if not is_final and (not remainder or is_open_marker(remainder)):
        return GroupScan("more")
    if not markers:
        return GroupScan("text")
    return GroupScan("group", tuple(markers), position)


class MarkerRewriter:
    """Rewrites the handles of a chunked text stream into links, and records what it cited."""

    def __init__(self, table: HandleTable) -> None:
        self.table = table
        self.pending = ""
        self.is_finished = False
        self.node_keys: list[NodeKey] = []
        self.table_ids: list[str] = []
        self.sources: list[Source] = []
        self.numbers_by_url: dict[str, int] = {}
        self.warnings: list[str] = []

    def feed(self, chunk: str) -> str:
        """Add `chunk` to the text and return the part that is safe to show."""
        self.pending += chunk
        return self.drain(is_final=False)

    def finish(self) -> str:
        """Return everything still held. An unclosed marker is returned as text and warned."""
        flushed = self.drain(is_final=True)
        self.is_finished = True
        return flushed

    def cited(self) -> CitedHandles:
        """Return what the stream cited, in citation order. Valid after `finish()`."""
        if not self.is_finished:
            raise RuntimeError("cited() is valid only after finish()")
        return CitedHandles(
            node_keys=tuple(self.node_keys),
            table_ids=tuple(self.table_ids),
            sources=tuple(self.sources),
            warnings=tuple(self.warnings),
        )

    def drain(self, is_final: bool) -> str:
        text = self.pending
        emitted: list[str] = []
        position = 0
        while position < len(text):
            step = self.read_step(text, position, is_final)
            if step is None:
                break
            piece, position = step
            emitted.append(piece)
        self.pending = text[position:]
        return "".join(emitted)

    def read_step(self, text: str, position: int, is_final: bool) -> tuple[str, int] | None:
        """Return the text to emit and the next position, or `None` to hold from `position`."""
        plain = PLAIN_TEXT_PATTERN.match(text, position)
        if plain:
            return plain.group(0), plain.end()
        if text[position] == MARKER_START:
            return self.read_group(text, position, "", is_final)
        whitespace_end = find_whitespace_end(text, position)
        whitespace = text[position:whitespace_end]
        if whitespace_end == len(text):
            return (whitespace, whitespace_end) if is_final else None
        if text[whitespace_end] != MARKER_START:
            return whitespace, whitespace_end
        return self.read_group(text, whitespace_end, whitespace, is_final)

    def read_group(
        self, text: str, start: int, whitespace: str, is_final: bool
    ) -> tuple[str, int] | None:
        """Read the group at `start`. `whitespace` is the run held in front of it."""
        scan = scan_group(text, start, is_final)
        if scan.status == "more":
            return None
        if scan.status == "text":
            self.warn_if_unclosed(text[start:], is_final)
            return whitespace + MARKER_START, start + 1
        links = self.render_group(scan.markers)
        return (whitespace + links if links else ""), scan.end

    def warn_if_unclosed(self, remainder: str, is_final: bool) -> None:
        if is_final and UNCLOSED_MARKER_PATTERN.fullmatch(remainder):
            self.add_warning(f"unclosed marker '{remainder}' at the end of the answer")

    def render_group(self, markers: tuple[str, ...]) -> str:
        numbers = {number for marker in markers if (number := self.cite(marker)) is not None}
        return "".join(self.render_link(number) for number in sorted(numbers))

    def cite(self, marker: str) -> int | None:
        """Record the citation of 1 marker and return its source number, if it has one."""
        handle = marker[1:-1]
        if handle.startswith(NODE_HANDLE_PREFIX):
            return self.cite_node(marker, self.table.nodes.get(handle))
        self.cite_table(marker, self.table.tables.get(handle))
        return None

    def cite_node(self, marker: str, node: HandleNode | None) -> int | None:
        if node is None:
            self.add_warning(f"{marker} names no node in the evidence")
            return None
        if node.key not in self.node_keys:
            self.node_keys.append(node.key)
        if node.source_url is None:
            return None
        return self.number_url(node.source_url, node.name)

    def cite_table(self, marker: str, evidence_id: str | None) -> None:
        if evidence_id is None:
            self.add_warning(f"{marker} names no table in the evidence")
        elif evidence_id not in self.table_ids:
            self.table_ids.append(evidence_id)

    def number_url(self, url: str, node_name: str | None) -> int:
        """Return the number of `url`, giving it the next number on its first citation."""
        if url not in self.numbers_by_url:
            title = node_name or self.table.page_titles.get(url) or url
            self.numbers_by_url[url] = len(self.sources) + 1
            self.sources.append(Source(number=len(self.sources) + 1, url=url, title=title))
        return self.numbers_by_url[url]

    def render_link(self, number: int) -> str:
        source = self.sources[number - 1]
        title = source.title.replace('"', '\\"')
        return f'[[{number}]]({source.url} "{title}")'

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
