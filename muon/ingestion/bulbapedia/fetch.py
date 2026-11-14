"""Fetch runs: which Bulbapedia pages to download, and the pins they leave behind.

A run covers 1 page type and takes 1 of 4 paths:

- pinned (the default): fetch the pinned revisions the cache lacks, so a fresh
  clone gets exactly the committed pages;
- update (`--update`, and any run on a page type with no pins): list the category
  and move the pins to the latest revisions;
- titles (`--title`): touch only the named pages;
- export (`--from-export`): pin every page of the page type found in an export
  file, with no request. Redirect pages are dropped, except for a page type that
  keeps them (`location`).

Page files are written per batch as they arrive, so a rerun after a failure
fetches only what is missing. The manifest is written once, at the end of a run
that changed it, and page files no pin points at are deleted after that.
"""

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

from muon.ingestion.bulbapedia.api import (
    MAIN_NAMESPACE,
    MAX_IDS_PER_REQUEST,
    BulbapediaClient,
    RevisionContent,
    RevisionMeta,
)
from muon.ingestion.bulbapedia.export import ExportedPage, read_export_revisions
from muon.ingestion.bulbapedia.pages import Manifest, Page, PagePin, PageStore


@dataclass(frozen=True)
class PageType:
    """A set of pages: the members of `category` whose title ends with `title_suffix`.

    With `keeps_redirects`, an export run also pins the redirect pages of the file. A
    species page may link a place by a redirect title, and the parser needs the redirect
    page to reach its target.
    """

    category: str
    title_suffix: str = ""
    keeps_redirects: bool = False

    def is_matching_title(self, title: str) -> bool:
        return title.endswith(self.title_suffix)


# Categories also hold overview pages ("Pokémon (species)", "Ability"); the suffix drops them.
PAGE_TYPES: Mapping[str, PageType] = {
    "species": PageType(category="Category:Pokémon", title_suffix=" (Pokémon)"),
    "ability": PageType(category="Category:Abilities", title_suffix=" (Ability)"),
    "item": PageType(category="Category:Items"),
    "move": PageType(category="Category:Moves", title_suffix=" (move)"),
    "location": PageType(category="Category:Locations", keeps_redirects=True),
}


@dataclass(frozen=True)
class FetchRequest:
    page_type: str
    titles: tuple[str, ...] = ()
    is_update: bool = False
    export_path: Path | None = None


@dataclass(frozen=True)
class FetchReport:
    """What 1 run did. `listed` counts the pages the run considered; `ignored` counts the
    pages of an export that are outside the page type."""

    page_type: str
    listed: int
    fetched: int
    skipped: int
    added: tuple[str, ...]
    changed: tuple[str, ...]
    removed: tuple[str, ...]
    redirects: tuple[tuple[str, str], ...]
    requests: int
    elapsed_seconds: float
    ignored: int = 0


@dataclass(frozen=True)
class FetchOutcome:
    """The pins a run leaves for its page type, and what it did to get there."""

    pins: tuple[PagePin, ...]
    listed: int
    fetched: int
    added: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    redirects: tuple[tuple[str, str], ...] = ()


class FetchError(Exception):
    """A run that cannot proceed: an unknown page type, an empty category or export, or a
    pin whose revision the API describes differently."""


def find_page_type(page_type: str) -> PageType:
    if page_type not in PAGE_TYPES:
        known = ", ".join(sorted(PAGE_TYPES))
        raise FetchError(f"Unknown page type {page_type!r}; known page types: {known}")
    return PAGE_TYPES[page_type]


def fetch_pages(request: FetchRequest, store: PageStore, client: BulbapediaClient) -> FetchReport:
    """Run `request`, write the manifest when the pins changed, and report."""
    find_page_type(request.page_type)
    started = time.monotonic()
    requests_before = client.request_count
    manifest = store.read_manifest()
    pins = manifest.list_pins(request.page_type)
    if request.titles:
        outcome = pin_titles(request, pins, store, client)
    elif request.is_update or not pins:
        outcome = update_pins(request.page_type, manifest, store, client)
    else:
        outcome = fetch_missing(request.page_type, pins, store, client)
    updated = manifest.replace_pins(request.page_type, outcome.pins)
    if updated != manifest:
        store.write_manifest(updated)
    store.delete_pages_except(updated.list_page_ids())
    return FetchReport(
        page_type=request.page_type,
        listed=outcome.listed,
        fetched=outcome.fetched,
        skipped=outcome.listed - outcome.fetched,
        added=outcome.added,
        changed=outcome.changed,
        removed=outcome.removed,
        redirects=outcome.redirects,
        requests=client.request_count - requests_before,
        elapsed_seconds=time.monotonic() - started,
    )


def import_pages(request: FetchRequest, store: PageStore) -> FetchReport:
    """The export run: pin every page of the page type that the export file holds.

    The pins of the page type are replaced as an update run replaces them, and only the
    page files that differ from their new pin are written. The run makes no request.
    """
    if request.export_path is None:
        raise FetchError(f"An export run of {request.page_type!r} needs an export file")
    definition = find_page_type(request.page_type)
    started = time.monotonic()
    exported = read_export_revisions(request.export_path)
    revisions = [page.revision for page in exported if is_importable(page, definition)]
    if not revisions:
        raise FetchError(
            f"{request.export_path} holds no page ending in {definition.title_suffix!r}"
        )
    manifest = store.read_manifest()
    old_pins = {pin.title: pin for pin in manifest.list_pins(request.page_type)}
    stale = [revision for revision in revisions if not store.is_current(build_pin(revision))]
    store_revisions(request.page_type, stale, store)
    pins = tuple(build_pin(revision) for revision in revisions)
    updated = manifest.replace_pins(request.page_type, pins)
    if updated != manifest:
        store.write_manifest(updated)
    store.delete_pages_except(updated.list_page_ids())
    titles = {pin.title for pin in pins}
    return FetchReport(
        page_type=request.page_type,
        listed=len(revisions),
        fetched=len(stale),
        skipped=len(revisions) - len(stale),
        added=tuple(sorted(titles - old_pins.keys())),
        changed=list_changed_titles(old_pins, pins),
        removed=tuple(sorted(old_pins.keys() - titles)),
        redirects=(),
        requests=0,
        elapsed_seconds=time.monotonic() - started,
        ignored=len(exported) - len(revisions),
    )


def is_importable(page: ExportedPage, definition: PageType) -> bool:
    """Whether an export page belongs to the page type: a main-namespace page with a
    matching title, and an article unless the page type keeps redirects."""
    return (
        page.namespace == MAIN_NAMESPACE
        and (not page.is_redirect or definition.keeps_redirects)
        and definition.is_matching_title(page.revision.title)
    )


def build_pin(revision: RevisionMeta) -> PagePin:
    return PagePin(
        title=revision.title,
        page_id=revision.page_id,
        revision_id=revision.revision_id,
        sha1=revision.sha1,
    )


def fetch_missing(
    page_type: str, pins: Sequence[PagePin], store: PageStore, client: BulbapediaClient
) -> FetchOutcome:
    """The pinned run: fetch the pinned revisions whose page file is missing or stale."""
    missing = {pin.revision_id: pin for pin in pins if not store.is_current(pin)}
    for batch in batched(missing, MAX_IDS_PER_REQUEST):
        revisions = client.fetch_content_by_ids(batch).revisions
        for revision in revisions:
            check_pinned_sha1(revision, missing[revision.revision_id])
        store_revisions(page_type, revisions, store)
    return FetchOutcome(pins=tuple(pins), listed=len(pins), fetched=len(missing))


def update_pins(
    page_type: str, manifest: Manifest, store: PageStore, client: BulbapediaClient
) -> FetchOutcome:
    """The update run: pin the latest revision of every matching category member.

    Known pages (pinned, or left on disk by a failed run) get a metadata request,
    and their content is fetched only when the latest revision differs or the page
    file is stale. Pages never seen are fetched by title.
    """
    definition = find_page_type(page_type)
    listed = [
        title
        for title in client.list_category_members(definition.category)
        if definition.is_matching_title(title)
    ]
    if not listed:
        raise FetchError(
            f"{definition.category} lists no page ending in {definition.title_suffix!r}"
        )
    old_pins = {pin.title: pin for pin in manifest.list_pins(page_type)}
    leftover_pins = {
        page.title: page.pin
        for page in store.read_unpinned_pages(manifest.list_page_ids())
        if page.page_type == page_type
    }
    known = leftover_pins | old_pins
    lookup = client.fetch_latest_metadata([title for title in listed if title in known])
    requested_title = {target: source for source, target in lookup.redirects.items()}
    current_pins: list[PagePin] = []
    stale_ids: list[int] = []
    for meta in lookup.revisions:
        pin = known[requested_title.get(meta.title, meta.title)]
        if meta.revision_id == pin.revision_id and store.is_current(pin):
            current_pins.append(pin)
        else:
            stale_ids.append(meta.revision_id)
    refetched_pins = download_by_ids(page_type, stale_ids, store, client)
    new_pins, _ = download_by_titles(
        page_type, [title for title in listed if title not in known], store, client
    )
    pins = tuple(current_pins) + refetched_pins + new_pins
    listed_titles = set(listed)
    return FetchOutcome(
        pins=pins,
        listed=len(listed),
        fetched=len(refetched_pins) + len(new_pins),
        added=tuple(sorted(listed_titles - old_pins.keys())),
        changed=list_changed_titles(old_pins, pins),
        removed=tuple(sorted(old_pins.keys() - listed_titles)),
    )


def pin_titles(
    request: FetchRequest, pins: Sequence[PagePin], store: PageStore, client: BulbapediaClient
) -> FetchOutcome:
    """The title run: pin the named pages and keep every other pin of the page type.

    Without `is_update`, a title that is already pinned keeps its pinned revision.
    """
    old_pins = {pin.title: pin for pin in pins}
    kept = [] if request.is_update else [old_pins[t] for t in request.titles if t in old_pins]
    kept_titles = {pin.title for pin in kept}
    kept_outcome = fetch_missing(request.page_type, kept, store, client)
    fetched_pins, redirects = download_by_titles(
        request.page_type, [t for t in request.titles if t not in kept_titles], store, client
    )
    fetched_titles = {pin.title for pin in fetched_pins}
    others = [pin for pin in pins if pin.title not in fetched_titles]
    return FetchOutcome(
        pins=tuple(others) + fetched_pins,
        listed=len(request.titles),
        fetched=kept_outcome.fetched + len(fetched_pins),
        added=tuple(sorted(title for title in fetched_titles if title not in old_pins)),
        changed=list_changed_titles(old_pins, fetched_pins),
        redirects=tuple(sorted(redirects.items())),
    )


def download_by_ids(
    page_type: str, revision_ids: Sequence[int], store: PageStore, client: BulbapediaClient
) -> tuple[PagePin, ...]:
    pins: list[PagePin] = []
    for batch in batched(revision_ids, MAX_IDS_PER_REQUEST):
        pins.extend(store_revisions(page_type, client.fetch_content_by_ids(batch).revisions, store))
    return tuple(pins)


def download_by_titles(
    page_type: str, titles: Sequence[str], store: PageStore, client: BulbapediaClient
) -> tuple[tuple[PagePin, ...], dict[str, str]]:
    """Fetch the latest revision of each title. Returns the pins and the title redirects."""
    pins: list[PagePin] = []
    redirects: dict[str, str] = {}
    for batch in batched(titles, MAX_IDS_PER_REQUEST):
        lookup = client.fetch_content_by_titles(batch)
        pins.extend(store_revisions(page_type, lookup.revisions, store))
        redirects |= lookup.redirects
    return tuple(pins), redirects


def store_revisions(
    page_type: str, revisions: Iterable[RevisionContent], store: PageStore
) -> tuple[PagePin, ...]:
    """Write 1 page file per revision and return their pins."""
    pages = [Page(**revision.model_dump(), page_type=page_type) for revision in revisions]
    for page in pages:
        store.write_page(page)
    return tuple(page.pin for page in pages)


def check_pinned_sha1(revision: RevisionContent, pin: PagePin) -> None:
    if revision.sha1 != pin.sha1:
        raise FetchError(
            f"Revision {revision.revision_id} of {pin.title!r} has SHA-1 {revision.sha1} "
            f"on Bulbapedia, but the manifest pins {pin.sha1}"
        )


def list_changed_titles(
    old_pins: Mapping[str, PagePin], new_pins: Iterable[PagePin]
) -> tuple[str, ...]:
    """Titles that were pinned before and now point at another revision."""
    return tuple(
        sorted(
            pin.title
            for pin in new_pins
            if pin.title in old_pins and old_pins[pin.title].revision_id != pin.revision_id
        )
    )
