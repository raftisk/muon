import pytest

from muon.ingestion.bulbapedia.api import (
    PROJECT_URL,
    USER_AGENT,
    BulbapediaAPIError,
    BulbapediaClient,
)
from muon.ingestion.bulbapedia.pages import compute_sha1
from tests.unit.ingestion.bulbapedia.fake_wiki import (
    HTML_CHALLENGE,
    MAXLAG,
    TIMEOUT,
    FakeClock,
    FakeWiki,
    InjectedFailure,
    build_status_failure,
)

CATEGORY = "Category:Pokémon"
DIGLETT = "Diglett (Pokémon)"
DUGTRIO = "Dugtrio (Pokémon)"
FIXED_PARAMETERS = {"action": "query", "format": "json", "formatversion": "2", "maxlag": "5"}


@pytest.fixture
def wiki() -> FakeWiki:
    fake = FakeWiki()
    fake.add_page(DIGLETT, "{{Pokémon Infobox|name=Diglett}}", CATEGORY)
    fake.add_page(DUGTRIO, "{{Pokémon Infobox|name=Dugtrio}}", CATEGORY)
    return fake


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def client(wiki: FakeWiki, clock: FakeClock) -> BulbapediaClient:
    return wiki.build_client(clock)


def add_species(wiki: FakeWiki, count: int) -> list[str]:
    titles = [f"Species {number} (Pokémon)" for number in range(count)]
    for title in titles:
        wiki.add_page(title, f"text of {title}", CATEGORY)
    return titles


def test_every_request_is_a_get_with_the_fixed_parameters(
    wiki: FakeWiki, client: BulbapediaClient
) -> None:
    client.list_category_members(CATEGORY)
    client.fetch_latest_metadata([DIGLETT])
    client.fetch_content_by_titles([DIGLETT])
    client.fetch_content_by_ids([wiki.pages[DIGLETT].revisions[-1].revision_id])

    assert len(wiki.requests) == 4
    for request in wiki.requests:
        assert request.method == "GET"
        assert FIXED_PARAMETERS.items() <= dict(request.url.params).items()


def test_every_request_carries_the_user_agent(wiki: FakeWiki, client: BulbapediaClient) -> None:
    client.fetch_latest_metadata([DIGLETT])

    assert wiki.requests[0].headers["User-Agent"] == USER_AGENT
    assert PROJECT_URL in USER_AGENT
    assert "@" not in USER_AGENT


def test_titles_are_sent_in_batches_of_50(wiki: FakeWiki, client: BulbapediaClient) -> None:
    titles = add_species(wiki, 120)

    lookup = client.fetch_latest_metadata(titles)

    batch_sizes = [len(params["titles"].split("|")) for params in wiki.requests_with("titles")]
    assert batch_sizes == [50, 50, 20]
    assert {revision.title for revision in lookup.revisions} == set(titles)


def test_category_listing_follows_continuation(wiki: FakeWiki, client: BulbapediaClient) -> None:
    add_species(wiki, 3)
    wiki.category_page_size = 2

    members = client.list_category_members(CATEGORY)

    assert len(members) == 5
    assert len(wiki.requests) == 3
    first = wiki.requests[0].url.params
    assert (first["list"], first["cmnamespace"], first["cmtype"]) == (
        "categorymembers",
        "0",
        "page",
    )


def test_content_continuation_is_merged(wiki: FakeWiki, client: BulbapediaClient) -> None:
    titles = add_species(wiki, 3) + [DIGLETT, DUGTRIO]
    wiki.content_page_size = 2

    lookup = client.fetch_content_by_titles(titles)

    assert len(wiki.requests) == 3
    assert {revision.title for revision in lookup.revisions} == set(titles)
    assert all(compute_sha1(revision.wikitext) == revision.sha1 for revision in lookup.revisions)


def test_content_by_id_returns_the_pinned_revision(
    wiki: FakeWiki, client: BulbapediaClient
) -> None:
    pinned = wiki.pages[DIGLETT].revisions[-1]
    wiki.edit_page(DIGLETT, "a newer revision")

    lookup = client.fetch_content_by_ids([pinned.revision_id])

    assert [revision.revision_id for revision in lookup.revisions] == [pinned.revision_id]
    assert lookup.revisions[0].wikitext == pinned.wikitext
    assert lookup.revisions[0].page_id == wiki.pages[DIGLETT].page_id


def test_normalized_and_redirected_titles_map_to_the_canonical_title(
    wiki: FakeWiki, client: BulbapediaClient
) -> None:
    wiki.add_redirect("Diglett", DIGLETT)

    lookup = client.fetch_content_by_titles(["diglett", "Dugtrio_(Pokémon)"])

    assert lookup.redirects == {"diglett": DIGLETT, "Dugtrio_(Pokémon)": DUGTRIO}
    assert {revision.title for revision in lookup.revisions} == {DIGLETT, DUGTRIO}


def test_missing_and_invalid_titles_are_named(client: BulbapediaClient) -> None:
    with pytest.raises(BulbapediaAPIError, match=r"'No Such Page' \(missing\).*'\[bad\]'"):
        client.fetch_content_by_titles([DIGLETT, "No Such Page", "[bad]"])


def test_bad_revision_ids_are_named(wiki: FakeWiki, client: BulbapediaClient) -> None:
    good = wiki.pages[DIGLETT].revisions[-1].revision_id

    with pytest.raises(BulbapediaAPIError, match="no revision 999999"):
        client.fetch_content_by_ids([good, 999999])


def test_hidden_text_fails(wiki: FakeWiki, client: BulbapediaClient) -> None:
    revision_id = wiki.pages[DIGLETT].revisions[-1].revision_id
    wiki.hide_revision(revision_id)

    with pytest.raises(BulbapediaAPIError, match=rf"Revision {revision_id} .*hidden text"):
        client.fetch_content_by_ids([revision_id])


def test_wikitext_that_differs_from_the_api_sha1_fails(
    wiki: FakeWiki, client: BulbapediaClient
) -> None:
    wiki.wrong_sha1_titles.add(DIGLETT)

    with pytest.raises(BulbapediaAPIError, match=r"Diglett.*hashes to .*the API reports 0{40}"):
        client.fetch_content_by_titles([DIGLETT])


@pytest.mark.parametrize(
    "failure",
    [MAXLAG, build_status_failure(429), build_status_failure(503), TIMEOUT],
    ids=["maxlag", "429", "503", "timeout"],
)
def test_retryable_failures_are_retried(
    wiki: FakeWiki, client: BulbapediaClient, clock: FakeClock, failure: InjectedFailure
) -> None:
    wiki.fail_next(failure)

    lookup = client.fetch_latest_metadata([DIGLETT])

    assert [revision.title for revision in lookup.revisions] == [DIGLETT]
    assert client.request_count == 2
    assert clock.sleeps == [2.0]


def test_retry_after_is_honored(wiki: FakeWiki, client: BulbapediaClient, clock: FakeClock) -> None:
    wiki.fail_next(build_status_failure(503, retry_after="7"))

    client.fetch_latest_metadata([DIGLETT])

    assert clock.sleeps == [7.0]


def test_backoff_doubles_per_retry(
    wiki: FakeWiki, client: BulbapediaClient, clock: FakeClock
) -> None:
    wiki.fail_next(MAXLAG, MAXLAG, MAXLAG)

    client.fetch_latest_metadata([DIGLETT])

    assert clock.sleeps == [2.0, 4.0, 8.0]


def test_client_gives_up_after_max_attempts(wiki: FakeWiki, client: BulbapediaClient) -> None:
    wiki.fail_next(*[build_status_failure(503)] * 5)

    with pytest.raises(BulbapediaAPIError, match=r"titles=Diglett.*after 5 attempts: HTTP 503"):
        client.fetch_latest_metadata([DIGLETT])

    assert client.request_count == 5


def test_forbidden_fails_without_retry(wiki: FakeWiki, client: BulbapediaClient) -> None:
    wiki.fail_next(build_status_failure(403))

    with pytest.raises(BulbapediaAPIError, match="HTTP 403 .*Cloudflare"):
        client.fetch_latest_metadata([DIGLETT])

    assert client.request_count == 1


def test_html_body_fails_without_retry(wiki: FakeWiki, client: BulbapediaClient) -> None:
    wiki.fail_next(HTML_CHALLENGE)

    with pytest.raises(BulbapediaAPIError, match="'text/html', not JSON"):
        client.list_category_members(CATEGORY)

    assert client.request_count == 1


def test_api_error_fails_without_retry(wiki: FakeWiki, client: BulbapediaClient) -> None:
    wiki.fail_next(InjectedFailure(body='{"error": {"code": "badvalue", "info": "Bad value"}}'))

    with pytest.raises(BulbapediaAPIError, match="returned error badvalue: Bad value"):
        client.fetch_latest_metadata([DIGLETT])

    assert client.request_count == 1


def test_warnings_fail(wiki: FakeWiki, client: BulbapediaClient) -> None:
    wiki.fail_next(InjectedFailure(body='{"warnings": {"main": {"warnings": "Too many values"}}}'))

    with pytest.raises(BulbapediaAPIError, match="returned warnings"):
        client.fetch_latest_metadata([DIGLETT])


def test_requests_keep_the_interval(client: BulbapediaClient, clock: FakeClock) -> None:
    client.fetch_latest_metadata([DIGLETT])
    client.fetch_latest_metadata([DUGTRIO])
    client.list_category_members(CATEGORY)

    assert clock.sleeps == [1.0, 1.0]
    assert client.request_count == 3
