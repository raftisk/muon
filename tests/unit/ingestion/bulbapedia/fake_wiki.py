"""An in-memory MediaWiki behind `httpx.MockTransport`, for offline Bulbapedia tests.

It answers the `action=query` requests the client sends: category members and
revisions by title or by revision id, with normalization, redirects, missing and
invalid titles, bad revision ids, hidden text and `continue` paging. Tests change
its pages between runs and schedule failures for given request numbers.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from muon.ingestion.bulbapedia.api import BulbapediaClient, RequestPolicy, build_http_client
from muon.ingestion.bulbapedia.pages import compute_sha1

BASE_TIME = datetime(2026, 9, 14, 11, 45, 57, tzinfo=UTC)
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
WRONG_SHA1 = "0" * 40
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
INVALID_TITLE_CHARACTER = "["
FIRST_PAGE_ID = 1
FIRST_REVISION_ID = 100


@dataclass(frozen=True)
class InjectedFailure:
    """A response that replaces the wiki's answer to 1 request."""

    status_code: int = 200
    body: str = ""
    content_type: str = JSON_CONTENT_TYPE
    retry_after: str | None = None
    is_timeout: bool = False


MAXLAG = InjectedFailure(
    body=json.dumps({"error": {"code": "maxlag", "info": "Waiting for a database server"}})
)
HTML_CHALLENGE = InjectedFailure(
    body="<!DOCTYPE html><title>Just a moment...</title>", content_type="text/html"
)
TIMEOUT = InjectedFailure(is_timeout=True)


def build_status_failure(status_code: int, retry_after: str | None = None) -> InjectedFailure:
    return InjectedFailure(status_code=status_code, body="busy", retry_after=retry_after)


@dataclass
class FakeRevision:
    revision_id: int
    wikitext: str
    timestamp: datetime
    is_hidden: bool = False


@dataclass
class FakePage:
    page_id: int
    title: str
    revisions: list[FakeRevision] = field(default_factory=list)


class FakeClock:
    """A clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def normalize_title(title: str) -> str:
    spaced = title.replace("_", " ")
    return spaced[:1].upper() + spaced[1:]


class FakeWiki:
    def __init__(self) -> None:
        self.pages: dict[str, FakePage] = {}
        self.categories: dict[str, list[str]] = {}
        self.redirects: dict[str, str] = {}
        self.requests: list[httpx.Request] = []
        self.scheduled_failures: dict[int, InjectedFailure] = {}
        self.category_page_size = 500
        self.content_page_size = 50
        self.wrong_sha1_titles: set[str] = set()
        self.next_page_id = FIRST_PAGE_ID
        self.next_revision_id = FIRST_REVISION_ID

    def build_client(self, clock: FakeClock | None = None) -> BulbapediaClient:
        fake_clock = clock or FakeClock()
        policy = RequestPolicy(sleep=fake_clock.sleep, clock=fake_clock.clock)
        return BulbapediaClient(build_http_client(httpx.MockTransport(self.handle)), policy)

    def add_page(self, title: str, wikitext: str, category: str | None = None) -> FakePage:
        page = FakePage(page_id=self.next_page_id, title=title)
        self.next_page_id += 1
        self.pages[title] = page
        self.edit_page(title, wikitext)
        if category is not None:
            self.categories.setdefault(category, []).append(title)
        return page

    def edit_page(self, title: str, wikitext: str) -> FakeRevision:
        page = self.pages[title]
        revision = FakeRevision(
            revision_id=self.next_revision_id,
            wikitext=wikitext,
            timestamp=BASE_TIME + timedelta(minutes=self.next_revision_id),
        )
        self.next_revision_id += 1
        page.revisions.append(revision)
        return revision

    def remove_from_category(self, category: str, title: str) -> None:
        self.categories[category].remove(title)

    def add_redirect(self, source: str, target: str) -> None:
        self.redirects[source] = target

    def hide_revision(self, revision_id: int) -> None:
        for page in self.pages.values():
            for revision in page.revisions:
                if revision.revision_id == revision_id:
                    revision.is_hidden = True

    def fail_at(self, request_number: int, failure: InjectedFailure) -> None:
        """Answer the `request_number`-th request (1-based, over the wiki's life) with `failure`."""
        self.scheduled_failures[request_number] = failure

    def fail_next(self, *failures: InjectedFailure) -> None:
        for offset, failure in enumerate(failures, start=1):
            self.fail_at(len(self.requests) + offset, failure)

    def requests_with(self, parameter: str) -> list[httpx.QueryParams]:
        return [request.url.params for request in self.requests if parameter in request.url.params]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        failure = self.scheduled_failures.pop(len(self.requests), None)
        if failure is not None:
            return self.build_failure(request, failure)
        if request.method != "GET":
            return httpx.Response(405)
        params = request.url.params
        if params.get("list") == "categorymembers":
            return build_json_response(self.list_category(params))
        if "revids" in params:
            return build_json_response(self.query_revision_ids(params))
        return build_json_response(self.query_titles(params))

    def build_failure(self, request: httpx.Request, failure: InjectedFailure) -> httpx.Response:
        if failure.is_timeout:
            raise httpx.ReadTimeout("timed out", request=request)
        headers = {"content-type": failure.content_type}
        if failure.retry_after is not None:
            headers["Retry-After"] = failure.retry_after
        return httpx.Response(failure.status_code, headers=headers, content=failure.body)

    def list_category(self, params: httpx.QueryParams) -> dict[str, Any]:
        members = self.categories.get(params["cmtitle"], [])
        offset = int(params.get("cmcontinue", "0"))
        chunk = members[offset : offset + self.category_page_size]
        body: dict[str, Any] = {
            "batchcomplete": True,
            "query": {
                "categorymembers": [
                    {"pageid": self.pages[title].page_id, "ns": 0, "title": title}
                    for title in chunk
                ]
            },
        }
        if offset + len(chunk) < len(members):
            body["continue"] = {"cmcontinue": str(offset + len(chunk)), "continue": "-||"}
        return body

    def query_titles(self, params: httpx.QueryParams) -> dict[str, Any]:
        normalized: list[dict[str, Any]] = []
        redirects: list[dict[str, str]] = []
        failed_pages: list[dict[str, Any]] = []
        found: list[tuple[FakePage, list[FakeRevision]]] = []
        for requested in params["titles"].split("|"):
            title = normalize_title(requested)
            if title != requested:
                normalized.append({"fromencoded": False, "from": requested, "to": title})
            if INVALID_TITLE_CHARACTER in title:
                failed_pages.append({"title": title, "invalidreason": "bad", "invalid": True})
                continue
            if title in self.redirects:
                redirects.append({"from": title, "to": self.redirects[title]})
                title = self.redirects[title]
            page = self.pages.get(title)
            if page is None:
                failed_pages.append({"ns": 0, "title": title, "missing": True})
                continue
            found.append((page, [page.revisions[-1]]))
        body = self.build_revisions_body(found, params)
        body["query"] |= {"normalized": normalized, "redirects": redirects}
        body["query"]["pages"] = failed_pages + body["query"]["pages"]
        return body

    def query_revision_ids(self, params: httpx.QueryParams) -> dict[str, Any]:
        by_id = {
            revision.revision_id: (page, revision)
            for page in self.pages.values()
            for revision in page.revisions
        }
        requested = [int(value) for value in params["revids"].split("|")]
        found: dict[int, tuple[FakePage, list[FakeRevision]]] = {}
        for revision_id in requested:
            if revision_id in by_id:
                page, revision = by_id[revision_id]
                found.setdefault(page.page_id, (page, []))[1].append(revision)
        body = self.build_revisions_body(list(found.values()), params)
        bad = [revision_id for revision_id in requested if revision_id not in by_id]
        if bad:
            body["query"]["badrevids"] = {
                str(revision_id): {"revid": revision_id, "missing": True} for revision_id in bad
            }
        return body

    def build_revisions_body(
        self, found: list[tuple[FakePage, list[FakeRevision]]], params: httpx.QueryParams
    ) -> dict[str, Any]:
        """Pages with their revisions. With content, only `content_page_size` pages per
        response carry revisions, and `rvcontinue` points at the next window."""
        is_content = "content" in params["rvprop"].split("|")
        offset = int(params.get("rvcontinue", "0"))
        size = self.content_page_size if is_content else len(found)
        pages = []
        for index, (page, revisions) in enumerate(found):
            entry: dict[str, Any] = {"pageid": page.page_id, "ns": 0, "title": page.title}
            if offset <= index < offset + size:
                entry["revisions"] = [
                    self.build_revision(page, revision, is_content) for revision in revisions
                ]
            pages.append(entry)
        body: dict[str, Any] = {"batchcomplete": True, "query": {"pages": pages}}
        if offset + size < len(found):
            body["continue"] = {"rvcontinue": str(offset + size), "continue": "||"}
        return body

    def build_revision(
        self, page: FakePage, revision: FakeRevision, is_content: bool
    ) -> dict[str, Any]:
        is_wrong = page.title in self.wrong_sha1_titles
        entry: dict[str, Any] = {
            "revid": revision.revision_id,
            "parentid": 0,
            "timestamp": revision.timestamp.strftime(TIMESTAMP_FORMAT),
            "sha1": WRONG_SHA1 if is_wrong else compute_sha1(revision.wikitext),
        }
        if is_content:
            slot = (
                {"texthidden": True}
                if revision.is_hidden
                else {"contentmodel": "wikitext", "content": revision.wikitext}
            )
            entry["slots"] = {"main": slot}
        return entry


def build_json_response(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": JSON_CONTENT_TYPE}, content=json.dumps(body).encode()
    )
