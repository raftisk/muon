"""Read Bulbapedia Special:Export files.

An export holds pages with their revisions as MediaWiki XML, in any `export-0.x`
schema. `read_export_revisions` reads and checks every page before it returns, so a
caller that writes nothing on `ExportError` writes nothing for a bad file. This
module never touches the network or the page store.
"""

import xml.etree.ElementTree as ElementTree
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from muon.ingestion.bulbapedia.api import RevisionContent
from muon.ingestion.bulbapedia.pages import compute_sha1

SHA1_BASE = 36
SHA1_HEX_FORMAT = "040x"
NAMESPACE_START = "{"
NAMESPACE_END = "}"
PARSE_EVENTS: tuple[Literal["start"], Literal["end"]] = ("start", "end")
START_EVENT = "start"
PAGE_TAG = "page"
TITLE_TAG = "title"
NAMESPACE_TAG = "ns"
ID_TAG = "id"
REDIRECT_TAG = "redirect"
REVISION_TAG = "revision"
TIMESTAMP_TAG = "timestamp"
TEXT_TAG = "text"
SHA1_TAG = "sha1"


class ExportError(Exception):
    """An export file that cannot be read, or a page in it that fails a check."""


class ExportPageError(Exception):
    """A page that fails a check. `read_export_revisions` turns it into an `ExportError`
    that names the file."""


@dataclass(frozen=True)
class ExportedPage:
    """1 page of an export with its only revision."""

    namespace: int
    is_redirect: bool
    revision: RevisionContent


def convert_base36_sha1(value: str) -> str:
    """Return the 40-digit hex form of a base 36 SHA-1, the form the API reports."""
    return format(int(value, SHA1_BASE), SHA1_HEX_FORMAT)


def read_export_revisions(path: Path) -> tuple[ExportedPage, ...]:
    """Read every page of the export at `path`, each with its SHA-1 checked."""
    try:
        return tuple(iterate_pages(path))
    except (OSError, ElementTree.ParseError) as error:
        raise ExportError(f"Cannot read export {path}: {error}") from error
    except ExportPageError as error:
        raise ExportError(f"Export {path}: {error}") from error


def iterate_pages(path: Path) -> Iterator[ExportedPage]:
    """Yield the pages of the export in file order, reading the XML namespace from the root tag."""
    namespace: str | None = None
    for event, element in ElementTree.iterparse(path, events=PARSE_EVENTS):
        if event == START_EVENT:
            if namespace is None:
                namespace = read_namespace(element.tag)
            continue
        if element.tag == f"{namespace}{PAGE_TAG}":
            yield read_page(element, namespace or "")
            element.clear()


def read_namespace(tag: str) -> str:
    """Return the `{uri}` prefix of a qualified tag, or an empty string for a plain tag."""
    if not tag.startswith(NAMESPACE_START):
        return ""
    return tag[: tag.index(NAMESPACE_END) + 1]


def read_page(page: ElementTree.Element, namespace: str) -> ExportedPage:
    title = read_child_text(page, namespace, TITLE_TAG, "a page")
    owner = f"page {title!r}"
    revisions = page.findall(f"{namespace}{REVISION_TAG}")
    if len(revisions) != 1:
        raise ExportPageError(
            f"{owner} has {len(revisions)} revisions; an export must hold exactly 1"
        )
    revision = revisions[0]
    wikitext = read_wikitext(revision, namespace, owner)
    export_sha1 = read_child_text(revision, namespace, SHA1_TAG, owner)
    hex_sha1 = convert_sha1(export_sha1, owner)
    computed_sha1 = compute_sha1(wikitext)
    if computed_sha1 != hex_sha1:
        raise ExportPageError(
            f"{owner} hashes to {computed_sha1}, the export reports {hex_sha1} "
            f"({export_sha1} in base 36)"
        )
    try:
        content = RevisionContent(
            title=title,
            page_id=read_child_int(page, namespace, ID_TAG, owner),
            revision_id=read_child_int(revision, namespace, ID_TAG, owner),
            timestamp=read_child_text(revision, namespace, TIMESTAMP_TAG, owner),
            sha1=hex_sha1,
            wikitext=wikitext,
        )
    except ValidationError as error:
        raise ExportPageError(f"{owner} has an invalid revision: {error}") from error
    return ExportedPage(
        namespace=read_child_int(page, namespace, NAMESPACE_TAG, owner),
        is_redirect=page.find(f"{namespace}{REDIRECT_TAG}") is not None,
        revision=content,
    )


def read_wikitext(revision: ElementTree.Element, namespace: str, owner: str) -> str:
    """Return the revision text. An empty `<text/>` is an empty page, not a missing one."""
    text = revision.find(f"{namespace}{TEXT_TAG}")
    if text is None:
        raise ExportPageError(f"{owner} has no <{TEXT_TAG}> in its revision")
    return text.text or ""


def read_child_text(element: ElementTree.Element, namespace: str, tag: str, owner: str) -> str:
    child = element.find(f"{namespace}{tag}")
    if child is None or child.text is None:
        raise ExportPageError(f"{owner} has no <{tag}>")
    return child.text


def read_child_int(element: ElementTree.Element, namespace: str, tag: str, owner: str) -> int:
    value = read_child_text(element, namespace, tag, owner)
    try:
        return int(value)
    except ValueError as error:
        raise ExportPageError(f"{owner} has a non-integer <{tag}> {value!r}") from error


def convert_sha1(value: str, owner: str) -> str:
    try:
        return convert_base36_sha1(value)
    except ValueError as error:
        raise ExportPageError(f"{owner} has an invalid base 36 SHA-1 {value!r}") from error
