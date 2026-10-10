"""Client for the Bulbapedia MediaWiki API.

Only GET requests to `api.php` get through: Cloudflare challenges POST requests,
the edit view and `action=raw`. Requests run 1 at a time, at least `interval`
seconds apart. `maxlag`, HTTP 429, HTTP 503 and timeouts are retried; every other
failure stops at once. Every revision fetch checks that each requested title or
revision id came back, and every content fetch checks that the wikitext hashes to
the `sha1` the API reports.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from itertools import batched
from typing import Any

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from muon.ingestion.bulbapedia.pages import compute_sha1

API_URL = "https://bulbapedia.bulbagarden.net/w/api.php"
PROJECT_URL = "https://github.com/raftisk/muon"
USER_AGENT = f"muon/{version('muon')} (+{PROJECT_URL})"
USER_AGENT_HEADER = "User-Agent"
RETRY_AFTER_HEADER = "Retry-After"
CONTENT_TYPE_HEADER = "content-type"
JSON_MEDIA_TYPE = "application/json"

REQUEST_INTERVAL_SECONDS = 1.0
MAX_ATTEMPTS = 5
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_FACTOR = 2.0
HTTP_TIMEOUT_SECONDS = 30.0
MAX_IDS_PER_REQUEST = 50
CATEGORY_MEMBERS_LIMIT = 500
MAIN_NAMESPACE = 0
MAXLAG_SECONDS = 5

OK_STATUS_CODE = 200
FORBIDDEN_STATUS_CODE = 403
RETRYABLE_STATUS_CODES = frozenset({429, 503})
MAXLAG_ERROR_CODE = "maxlag"
FORBIDDEN_HINT = " (Cloudflare challenge or blocked User-Agent)"
CHALLENGE_HINT = " (likely a Cloudflare challenge page)"

TITLES_PARAMETER = "titles"
REVISION_IDS_PARAMETER = "revids"
CATEGORY_TITLE_PARAMETER = "cmtitle"
VALUE_SEPARATOR = "|"
MAIN_SLOT = "main"
FIXED_PARAMETERS: Mapping[str, str] = {
    "action": "query",
    "format": "json",
    "formatversion": "2",
    "maxlag": str(MAXLAG_SECONDS),
}
CATEGORY_MEMBERS_PARAMETERS: Mapping[str, str] = {
    "list": "categorymembers",
    "cmnamespace": str(MAIN_NAMESPACE),
    "cmtype": "page",
    "cmlimit": str(CATEGORY_MEMBERS_LIMIT),
}
METADATA_PARAMETERS: Mapping[str, str] = {"prop": "revisions", "rvprop": "ids|timestamp|sha1"}
CONTENT_PARAMETERS: Mapping[str, str] = {
    "prop": "revisions",
    "rvprop": "ids|timestamp|sha1|content",
    "rvslots": MAIN_SLOT,
}
TITLE_QUERY_PARAMETERS: Mapping[str, str] = {"redirects": "1"}


@dataclass(frozen=True)
class RequestPolicy:
    """Pacing and retry rules. `interval` and the backoff are in seconds."""

    interval: float = REQUEST_INTERVAL_SECONDS
    max_attempts: int = MAX_ATTEMPTS
    initial_backoff: float = INITIAL_BACKOFF_SECONDS
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic


DEFAULT_POLICY = RequestPolicy()


class BulbapediaAPIError(Exception):
    """A request or response the client cannot use."""


class RevisionMeta(BaseModel):
    """1 revision of a page, without its wikitext."""

    model_config = ConfigDict(frozen=True)

    title: str
    page_id: int
    revision_id: int
    timestamp: AwareDatetime
    sha1: str


class RevisionContent(RevisionMeta):
    """1 revision of a page with its wikitext, checked against `sha1`."""

    wikitext: str


@dataclass(frozen=True)
class Lookup[RevisionT: RevisionMeta]:
    """The revisions a fetch returned. `redirects` maps each requested title that was
    normalized or redirected to the canonical title of its page."""

    revisions: tuple[RevisionT, ...]
    redirects: Mapping[str, str]


class TitleChange(BaseModel):
    source: str = Field(alias="from")
    target: str = Field(alias="to")


class ApiSlot(BaseModel):
    content: str | None = None
    is_text_hidden: bool = Field(default=False, alias="texthidden")


class ApiRevision(BaseModel):
    revision_id: int = Field(alias="revid")
    timestamp: AwareDatetime
    sha1: str | None = None
    slots: Mapping[str, ApiSlot] = Field(default_factory=dict)


class ApiPage(BaseModel):
    title: str
    page_id: int | None = Field(default=None, alias="pageid")
    is_invalid: bool = Field(default=False, alias="invalid")
    invalid_reason: str | None = Field(default=None, alias="invalidreason")
    revisions: tuple[ApiRevision, ...] = ()


class QueryResult(BaseModel):
    pages: tuple[ApiPage, ...] = ()
    normalized: tuple[TitleChange, ...] = ()
    redirects: tuple[TitleChange, ...] = ()
    bad_revision_ids: Mapping[str, Any] = Field(default_factory=dict, alias="badrevids")
    category_members: tuple[ApiPage, ...] = Field(default=(), alias="categorymembers")


class ApiError(BaseModel):
    code: str
    info: str = ""


class QueryResponse(BaseModel):
    """The parts of an `action=query` response the client reads."""

    query: QueryResult = Field(default_factory=QueryResult)
    continuation: Mapping[str, str] | None = Field(default=None, alias="continue")
    error: ApiError | None = None
    warnings: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class QueryOutcome:
    """A revisions query merged over its continued responses.

    `revisions` maps revision ids to their page. `normalized` and `redirects` map
    titles as MediaWiki rewrote them, in that order.
    """

    revisions: Mapping[int, tuple[ApiPage, ApiRevision]]
    normalized: Mapping[str, str]
    redirects: Mapping[str, str]
    invalid_reasons: Mapping[str, str]
    bad_revision_ids: frozenset[str]

    def resolve_title(self, title: str) -> str:
        normalized_title = self.normalized.get(title, title)
        return self.redirects.get(normalized_title, normalized_title)


@dataclass(frozen=True)
class RetryableFailure:
    """A failed attempt worth retrying. `retry_after` is the server's wait in seconds."""

    reason: str
    retry_after: float | None


@dataclass(frozen=True)
class RevisionQuery:
    """1 revisions query: by titles or by revision ids, with or without wikitext."""

    selector: str
    values: tuple[str, ...]
    is_content: bool


type RevisionBuilder[RevisionT: RevisionMeta] = Callable[[ApiPage, ApiRevision], RevisionT]


def build_http_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Return an HTTP client with the muon User-Agent. Tests pass a mock `transport`."""
    return httpx.Client(
        headers={USER_AGENT_HEADER: USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
        transport=transport,
    )


def build_request_error(params: Mapping[str, str], problem: str) -> BulbapediaAPIError:
    """Return an error that names the request by its titles, revision ids or category."""
    asked = next(
        (
            f"{key}={params[key]}"
            for key in (TITLES_PARAMETER, REVISION_IDS_PARAMETER, CATEGORY_TITLE_PARAMETER)
            if key in params
        ),
        str(dict(params)),
    )
    return BulbapediaAPIError(f"GET {API_URL} for {asked} {problem}")


def build_revision_params(query: RevisionQuery, batch: Sequence[str]) -> dict[str, str]:
    is_title_query = query.selector == TITLES_PARAMETER
    return {
        **(CONTENT_PARAMETERS if query.is_content else METADATA_PARAMETERS),
        **(TITLE_QUERY_PARAMETERS if is_title_query else {}),
        query.selector: VALUE_SEPARATOR.join(batch),
    }


def parse_retry_after(response: httpx.Response) -> float | None:
    """Return `Retry-After` in seconds, or `None` when it is absent or an HTTP date."""
    header = response.headers.get(RETRY_AFTER_HEADER)
    if header is None or not header.strip().isdigit():
        return None
    return float(header)


def parse_body(response: httpx.Response, params: Mapping[str, str]) -> QueryResponse:
    """Return the body of a response that is not retried, or fail with its status."""
    if response.status_code != OK_STATUS_CODE:
        hint = FORBIDDEN_HINT if response.status_code == FORBIDDEN_STATUS_CODE else ""
        raise build_request_error(params, f"failed with HTTP {response.status_code}{hint}")
    content_type = response.headers.get(CONTENT_TYPE_HEADER, "")
    if not content_type.startswith(JSON_MEDIA_TYPE):
        raise build_request_error(params, f"returned {content_type!r}, not JSON{CHALLENGE_HINT}")
    try:
        return QueryResponse.model_validate_json(response.content)
    except ValidationError as error:
        raise build_request_error(params, f"returned an unexpected body: {error}") from error


def merge_responses(responses: Sequence[QueryResponse]) -> QueryOutcome:
    results = [response.query for response in responses]
    pages = [page for result in results for page in result.pages]
    return QueryOutcome(
        revisions={
            revision.revision_id: (page, revision) for page in pages for revision in page.revisions
        },
        normalized={
            change.source: change.target for result in results for change in result.normalized
        },
        redirects={
            change.source: change.target for result in results for change in result.redirects
        },
        invalid_reasons={
            page.title: page.invalid_reason or "invalid" for page in pages if page.is_invalid
        },
        bad_revision_ids=frozenset(key for result in results for key in result.bad_revision_ids),
    )


def check_titles_found(titles: Sequence[str], outcome: QueryOutcome) -> None:
    """Fail when a requested title has no page with a revision, naming each such title."""
    found_titles = {page.title for page, _ in outcome.revisions.values()}
    absent = [title for title in titles if outcome.resolve_title(title) not in found_titles]
    if absent:
        described = ", ".join(
            f"{title!r} ({outcome.invalid_reasons.get(title, 'missing')})" for title in absent
        )
        raise BulbapediaAPIError(f"Bulbapedia has no page for {described}")


def check_revisions_found(revision_ids: Sequence[str], outcome: QueryOutcome) -> None:
    """Fail when a requested revision id is bad or did not come back, naming each such id."""
    absent = outcome.bad_revision_ids | {
        value for value in revision_ids if int(value) not in outcome.revisions
    }
    if absent:
        raise BulbapediaAPIError(f"Bulbapedia has no revision {', '.join(sorted(absent))}")


def build_meta(page: ApiPage, revision: ApiRevision) -> RevisionMeta:
    if page.page_id is None or revision.sha1 is None:
        raise BulbapediaAPIError(
            f"Revision {revision.revision_id} of {page.title!r} has no page id or a hidden SHA-1"
        )
    return RevisionMeta(
        title=page.title,
        page_id=page.page_id,
        revision_id=revision.revision_id,
        timestamp=revision.timestamp,
        sha1=revision.sha1,
    )


def build_content(page: ApiPage, revision: ApiRevision) -> RevisionContent:
    """Return the revision with its wikitext, after checking the text against the API `sha1`."""
    meta = build_meta(page, revision)
    slot = revision.slots.get(MAIN_SLOT)
    if slot is None or slot.is_text_hidden or slot.content is None:
        raise BulbapediaAPIError(
            f"Revision {revision.revision_id} of {page.title!r} has hidden text"
        )
    computed_sha1 = compute_sha1(slot.content)
    if computed_sha1 != meta.sha1:
        raise BulbapediaAPIError(
            f"Revision {revision.revision_id} of {page.title!r} hashes to {computed_sha1}, "
            f"the API reports {meta.sha1}"
        )
    return RevisionContent(**meta.model_dump(), wikitext=slot.content)


class BulbapediaClient:
    """Reads category members and page revisions from the Bulbapedia API."""

    def __init__(self, http_client: httpx.Client, policy: RequestPolicy = DEFAULT_POLICY) -> None:
        self.http_client = http_client
        self.policy = policy
        self.requests_sent = 0
        self.last_request_start: float | None = None

    @property
    def request_count(self) -> int:
        """HTTP requests sent so far, retries included."""
        return self.requests_sent

    def list_category_members(self, category: str) -> tuple[str, ...]:
        """Return the titles of the main-namespace pages in `category`."""
        responses = self.query_all(
            {**CATEGORY_MEMBERS_PARAMETERS, CATEGORY_TITLE_PARAMETER: category}
        )
        return tuple(
            member.title for response in responses for member in response.query.category_members
        )

    def fetch_latest_metadata(self, titles: Sequence[str]) -> Lookup[RevisionMeta]:
        query = RevisionQuery(TITLES_PARAMETER, tuple(titles), is_content=False)
        return self.fetch_revisions(query, build_meta)

    def fetch_content_by_titles(self, titles: Sequence[str]) -> Lookup[RevisionContent]:
        query = RevisionQuery(TITLES_PARAMETER, tuple(titles), is_content=True)
        return self.fetch_revisions(query, build_content)

    def fetch_content_by_ids(self, revision_ids: Sequence[int]) -> Lookup[RevisionContent]:
        values = tuple(str(revision_id) for revision_id in revision_ids)
        query = RevisionQuery(REVISION_IDS_PARAMETER, values, is_content=True)
        return self.fetch_revisions(query, build_content)

    def fetch_revisions[RevisionT: RevisionMeta](
        self, query: RevisionQuery, build: RevisionBuilder[RevisionT]
    ) -> Lookup[RevisionT]:
        """Run `query` in batches and check that every requested title or id came back."""
        revisions: list[RevisionT] = []
        redirects: dict[str, str] = {}
        for batch in batched(query.values, MAX_IDS_PER_REQUEST):
            outcome = merge_responses(self.query_all(build_revision_params(query, batch)))
            if query.selector == TITLES_PARAMETER:
                check_titles_found(batch, outcome)
                resolved = {title: outcome.resolve_title(title) for title in batch}
                redirects |= {
                    title: target for title, target in resolved.items() if target != title
                }
            else:
                check_revisions_found(batch, outcome)
            revisions.extend(build(page, revision) for page, revision in outcome.revisions.values())
        return Lookup(revisions=tuple(revisions), redirects=redirects)

    def query_all(self, params: Mapping[str, str]) -> tuple[QueryResponse, ...]:
        """Send `params` and follow `continue` until the result is complete."""
        responses: list[QueryResponse] = []
        continuation: Mapping[str, str] = {}
        while True:
            response = self.send_query({**params, **continuation})
            responses.append(response)
            if response.continuation is None:
                return tuple(responses)
            continuation = response.continuation

    def send_query(self, params: Mapping[str, str]) -> QueryResponse:
        """Send 1 query, retrying throttling and timeouts up to `max_attempts` times."""
        failure: RetryableFailure | None = None
        for attempt in range(self.policy.max_attempts):
            if failure is not None:
                self.policy.sleep(self.compute_retry_delay(attempt, failure))
            outcome = self.send_once(params)
            if isinstance(outcome, QueryResponse):
                return outcome
            failure = outcome
        reason = failure.reason if failure is not None else "no attempt made"
        raise build_request_error(
            params, f"failed after {self.policy.max_attempts} attempts: {reason}"
        )

    def compute_retry_delay(self, attempt: int, failure: RetryableFailure) -> float:
        """The server's `Retry-After` when given, otherwise a backoff doubled per retry."""
        if failure.retry_after is not None:
            return failure.retry_after
        return self.policy.initial_backoff * BACKOFF_FACTOR ** (attempt - 1)

    def send_once(self, params: Mapping[str, str]) -> QueryResponse | RetryableFailure:
        self.wait_for_interval()
        self.requests_sent += 1
        try:
            response = self.http_client.get(API_URL, params={**FIXED_PARAMETERS, **params})
        except httpx.TimeoutException as error:
            return RetryableFailure(reason=f"timeout ({error})", retry_after=None)
        except httpx.HTTPError as error:
            raise build_request_error(params, f"failed: {error}") from error
        if response.status_code in RETRYABLE_STATUS_CODES:
            reason = f"HTTP {response.status_code}"
            return RetryableFailure(reason=reason, retry_after=parse_retry_after(response))
        body = parse_body(response, params)
        if body.error is not None and body.error.code == MAXLAG_ERROR_CODE:
            return RetryableFailure(
                reason=MAXLAG_ERROR_CODE, retry_after=parse_retry_after(response)
            )
        if body.error is not None:
            raise build_request_error(
                params, f"returned error {body.error.code}: {body.error.info}"
            )
        if body.warnings is not None:
            raise build_request_error(params, f"returned warnings {dict(body.warnings)}")
        return body

    def wait_for_interval(self) -> None:
        """Sleep until `interval` seconds have passed since the previous request started."""
        if self.last_request_start is not None:
            remaining = self.last_request_start + self.policy.interval - self.policy.clock()
            if remaining > 0:
                self.policy.sleep(remaining)
        self.last_request_start = self.policy.clock()
