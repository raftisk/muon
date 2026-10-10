"""Pinned Bulbapedia pages on disk.

`<root>/manifest.json` pins 1 revision per page and is committed.
`<root>/pages/<page_id>.json` holds the wikitext of the pinned revision and is
gitignored. Every read checks a page file against its pin, so later ingestion
steps never parse a page that differs from the manifest. This module never
touches the network.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Self
from urllib.parse import quote

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from muon.config import PROJECT_ROOT

DEFAULT_ROOT = PROJECT_ROOT / "data" / "bulbapedia"
MANIFEST_FILE_NAME = "manifest.json"
PAGES_DIR_NAME = "pages"
PAGE_FILE_SUFFIX = ".json"
TEMP_FILE_SUFFIX = ".tmp"
TEXT_ENCODING = "utf-8"
JSON_INDENT = 2
PERMALINK_TEMPLATE = (
    "https://bulbapedia.bulbagarden.net/w/index.php?title={title}&oldid={revision_id}"
)
# MediaWiki writes spaces in URL titles as underscores and leaves parentheses unescaped.
TITLE_SPACE = " "
URL_TITLE_SPACE = "_"
PERMALINK_SAFE_CHARACTERS = "()"
FETCH_HINT = "run `muon fetch bulbapedia {page_type}`"


def compute_sha1(text: str) -> str:
    """Return the hex SHA-1 of the UTF-8 bytes of `text`, the value the API reports as `sha1`."""
    return hashlib.sha1(text.encode(TEXT_ENCODING)).hexdigest()


def write_text_atomically(path: Path, text: str) -> None:
    """Write `text` to `path`. A reader sees the old file or the new one, never a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=TEMP_FILE_SUFFIX
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding=TEXT_ENCODING) as handle:
            handle.write(text)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


class PagePin(BaseModel):
    """The pinned revision of 1 page. `sha1` is the hex SHA-1 of its wikitext."""

    model_config = ConfigDict(frozen=True)

    title: str
    page_id: int
    revision_id: int
    sha1: str


class Manifest(BaseModel):
    """The pins per page type. A page type with no pins is absent."""

    model_config = ConfigDict(frozen=True)

    page_types: Mapping[str, tuple[PagePin, ...]] = Field(default_factory=dict)

    def list_pins(self, page_type: str) -> tuple[PagePin, ...]:
        return tuple(self.page_types.get(page_type, ()))

    def find_pin(self, page_type: str, title: str) -> PagePin | None:
        return next((pin for pin in self.list_pins(page_type) if pin.title == title), None)

    def replace_pins(self, page_type: str, pins: Iterable[PagePin]) -> Self:
        """Return a manifest whose `page_type` holds `pins`, sorted by title."""
        sorted_pins = tuple(sorted(pins, key=lambda pin: pin.title))
        page_types = {name: kept for name, kept in self.page_types.items() if name != page_type}
        if sorted_pins:
            page_types[page_type] = sorted_pins
        return self.model_copy(update={"page_types": page_types})

    def list_page_ids(self) -> frozenset[int]:
        return frozenset(pin.page_id for pins in self.page_types.values() for pin in pins)


class Page(BaseModel):
    """1 page file: the pinned revision of a page, its page type and its wikitext."""

    model_config = ConfigDict(frozen=True)

    title: str
    page_id: int
    revision_id: int
    timestamp: AwareDatetime
    sha1: str
    page_type: str
    wikitext: str

    @property
    def permalink(self) -> str:
        """The URL of this revision, the `sourceUrl` of chunks cut from it."""
        url_title = quote(
            self.title.replace(TITLE_SPACE, URL_TITLE_SPACE), safe=PERMALINK_SAFE_CHARACTERS
        )
        return PERMALINK_TEMPLATE.format(title=url_title, revision_id=self.revision_id)

    @property
    def pin(self) -> PagePin:
        return PagePin(
            title=self.title, page_id=self.page_id, revision_id=self.revision_id, sha1=self.sha1
        )


class PageStoreError(Exception):
    """A manifest or page file that is missing, unreadable or differs from its pin."""


def describe_mismatch(page: Page, pin: PagePin) -> str | None:
    """Return why `page` is not the revision `pin` names, or `None` when it is."""
    if page.revision_id != pin.revision_id:
        return f"revision id {page.revision_id} in the file, {pin.revision_id} in the pin"
    if page.sha1 != pin.sha1:
        return f"SHA-1 {page.sha1} in the file, {pin.sha1} in the pin"
    computed_sha1 = compute_sha1(page.wikitext)
    if computed_sha1 != pin.sha1:
        return f"wikitext hashes to {computed_sha1}, the pin says {pin.sha1}"
    return None


def serialize_manifest(manifest: Manifest) -> str:
    """Return the manifest as JSON with page types sorted by name and pins sorted by title."""
    page_types = {
        page_type: [
            pin.model_dump()
            for pin in sorted(manifest.page_types[page_type], key=lambda p: p.title)
        ]
        for page_type in sorted(manifest.page_types)
    }
    return json.dumps({"page_types": page_types}, indent=JSON_INDENT, ensure_ascii=False) + "\n"


class PageStore:
    """The manifest and the page files under `root`."""

    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.manifest_path = root / MANIFEST_FILE_NAME
        self.pages_dir = root / PAGES_DIR_NAME

    def read_manifest(self) -> Manifest:
        """Return the manifest, or an empty one when the file does not exist yet."""
        if not self.manifest_path.exists():
            return Manifest()
        try:
            return Manifest.model_validate_json(
                self.manifest_path.read_text(encoding=TEXT_ENCODING)
            )
        except ValidationError as error:
            raise PageStoreError(f"Cannot read manifest {self.manifest_path}: {error}") from error

    def write_manifest(self, manifest: Manifest) -> None:
        write_text_atomically(self.manifest_path, serialize_manifest(manifest))

    def build_page_path(self, page_id: int) -> Path:
        return self.pages_dir / f"{page_id}{PAGE_FILE_SUFFIX}"

    def load_page_file(self, path: Path) -> Page:
        try:
            return Page.model_validate_json(path.read_text(encoding=TEXT_ENCODING))
        except ValidationError as error:
            raise PageStoreError(f"Cannot read page file {path}: {error}") from error

    def read_page(self, page_type: str, pin: PagePin) -> Page:
        """Return the page file of `pin`, checked against the pin and `page_type`."""
        path = self.build_page_path(pin.page_id)
        if not path.exists():
            hint = FETCH_HINT.format(page_type=page_type)
            raise PageStoreError(
                f"No page file for {pin.title!r} (page id {pin.page_id}) at {path}: {hint}"
            )
        page = self.load_page_file(path)
        mismatch = describe_mismatch(page, pin)
        if mismatch is not None:
            raise PageStoreError(f"Page file for {pin.title!r} differs from its pin: {mismatch}")
        if page.page_type != page_type:
            raise PageStoreError(
                f"Page file for {pin.title!r} has page type {page.page_type!r}, "
                f"expected {page_type!r}"
            )
        return page

    def is_current(self, pin: PagePin) -> bool:
        """Whether the page file of `pin` holds the pinned revision, with intact wikitext."""
        path = self.build_page_path(pin.page_id)
        if not path.exists():
            return False
        try:
            page = self.load_page_file(path)
        except PageStoreError:
            return False
        return describe_mismatch(page, pin) is None

    def write_page(self, page: Page) -> None:
        write_text_atomically(self.build_page_path(page.page_id), page.model_dump_json())

    def delete_pages_except(self, page_ids: frozenset[int]) -> tuple[int, ...]:
        """Delete every page file whose page id is not in `page_ids` and return the deleted ids."""
        if not self.pages_dir.exists():
            return ()
        stale_paths = sorted(
            path
            for path in self.pages_dir.glob(f"*{PAGE_FILE_SUFFIX}")
            if path.stem.isdigit() and int(path.stem) not in page_ids
        )
        for path in stale_paths:
            path.unlink()
        return tuple(int(path.stem) for path in stale_paths)
