"""Special:Export XML for offline tests, in the layout of MediaWiki export 0.11."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from muon.ingestion.bulbapedia.pages import compute_sha1
from tests.unit.ingestion.bulbapedia.fake_wiki import TIMESTAMP_FORMAT, FakeWiki

EXPORT_VERSION = "0.11"
EXPORT_NAMESPACE_TEMPLATE = "http://www.mediawiki.org/xml/export-{version}/"
EXPORT_FILE_NAME = "export.xml"
BASE36_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
HEX_BASE = 16
BASE36_WIDTH = 31


@dataclass(frozen=True)
class ExportFixturePage:
    """1 exported page. `sha1` replaces the right base 36 value; `extra_revisions` adds
    older revisions, which a current-only export never holds."""

    title: str
    page_id: int
    revision_id: int
    timestamp: datetime
    wikitext: str
    namespace: int = 0
    is_redirect: bool = False
    sha1: str | None = None
    extra_revisions: int = 0


def encode_base36_sha1(hex_sha1: str) -> str:
    """Return the base 36 form MediaWiki writes: lowercase, padded to 31 digits."""
    number = int(hex_sha1, HEX_BASE)
    digits: list[str] = []
    while number:
        number, remainder = divmod(number, len(BASE36_DIGITS))
        digits.append(BASE36_DIGITS[remainder])
    return "".join(reversed(digits)).rjust(BASE36_WIDTH, "0")


def build_revision_xml(page: ExportFixturePage, revision_id: int) -> str:
    sha1 = page.sha1 or encode_base36_sha1(compute_sha1(page.wikitext))
    return (
        "    <revision>\n"
        f"      <id>{revision_id}</id>\n"
        f"      <timestamp>{page.timestamp.strftime(TIMESTAMP_FORMAT)}</timestamp>\n"
        "      <contributor><username>Tester</username><id>1</id></contributor>\n"
        "      <model>wikitext</model>\n"
        "      <format>text/x-wiki</format>\n"
        f'      <text bytes="{len(page.wikitext.encode())}" sha1="{sha1}" xml:space="preserve">'
        f"{escape(page.wikitext)}</text>\n"
        f"      <sha1>{sha1}</sha1>\n"
        "    </revision>\n"
    )


def build_page_xml(page: ExportFixturePage) -> str:
    redirect = '    <redirect title="Target" />\n' if page.is_redirect else ""
    older = [page.revision_id - offset for offset in range(page.extra_revisions, 0, -1)]
    revisions = "".join(
        build_revision_xml(page, revision_id) for revision_id in [*older, page.revision_id]
    )
    return (
        "  <page>\n"
        f"    <title>{escape(page.title)}</title>\n"
        f"    <ns>{page.namespace}</ns>\n"
        f"    <id>{page.page_id}</id>\n"
        f"{redirect}{revisions}"
        "  </page>\n"
    )


def build_export_xml(pages: Sequence[ExportFixturePage], version: str = EXPORT_VERSION) -> str:
    namespace = EXPORT_NAMESPACE_TEMPLATE.format(version=version)
    return (
        f'<mediawiki xmlns={quoteattr(namespace)} version={quoteattr(version)} xml:lang="en">\n'
        "  <siteinfo>\n"
        "    <sitename>Bulbapedia</sitename>\n"
        "    <dbname>bulbapedia</dbname>\n"
        "  </siteinfo>\n"
        f"{''.join(build_page_xml(page) for page in pages)}"
        "</mediawiki>\n"
    )


def write_export(directory: Path, pages: Sequence[ExportFixturePage]) -> Path:
    path = directory / EXPORT_FILE_NAME
    path.write_text(build_export_xml(pages), encoding="utf-8")
    return path


def build_export_from_wiki(wiki: FakeWiki, titles: Sequence[str]) -> list[ExportFixturePage]:
    """The latest revision of each title, as an export of the same wiki would hold it."""
    pages = []
    for title in titles:
        page = wiki.pages[title]
        revision = page.revisions[-1]
        pages.append(
            ExportFixturePage(
                title=page.title,
                page_id=page.page_id,
                revision_id=revision.revision_id,
                timestamp=revision.timestamp,
                wikitext=revision.wikitext,
            )
        )
    return pages
