from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from muon.ingestion.pokeapi.snapshot import (
    REPOSITORY,
    SNAPSHOT_FILES,
    SnapshotError,
    SnapshotManifest,
    SnapshotStore,
    compute_sha256,
    fetch_snapshot,
)

COMMIT = "a" * 40
NEXT_COMMIT = "b" * 40
COMMIT_PATH = "/repos/PokeAPI/pokeapi/commits/master"


def build_files(marker: str) -> dict[str, bytes]:
    return {name: f"id,identifier\n1,{marker}-{name}\n".encode() for name in SNAPSHOT_FILES}


class FakeGitHub:
    """Serves the commit endpoint and raw files per commit, and records every request."""

    def __init__(self, head: str, files_by_commit: Mapping[str, Mapping[str, bytes]]) -> None:
        self.head = head
        self.files_by_commit = files_by_commit
        self.requests: list[str] = []
        self.missing: set[str] = set()

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        if request.url.path == COMMIT_PATH:
            return httpx.Response(200, json={"sha": self.head})
        parts = request.url.path.split("/")
        commit, name = parts[3], parts[-1]
        content = self.files_by_commit.get(commit, {}).get(name)
        if content is None or name in self.missing:
            return httpx.Response(404)
        return httpx.Response(200, content=content)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))


def pin(store: SnapshotStore, commit: str, files: Mapping[str, bytes]) -> None:
    store.write_manifest(
        SnapshotManifest(
            repository=REPOSITORY,
            commit=commit,
            files={name: compute_sha256(content) for name, content in files.items()},
        )
    )


def test_update_pins_the_head_commit_and_every_file(tmp_path: Path) -> None:
    files = build_files("v1")
    github = FakeGitHub(COMMIT, {COMMIT: files})
    store = SnapshotStore(tmp_path)

    report = fetch_snapshot(store, github.client(), is_update=True)

    manifest = store.read_manifest()
    assert manifest is not None
    assert manifest.commit == COMMIT
    assert manifest.files == {name: compute_sha256(content) for name, content in files.items()}
    assert report.files_downloaded == len(SNAPSHOT_FILES)
    assert report.requests == len(SNAPSHOT_FILES) + 1
    assert store.build_file_path("moves.csv").read_bytes() == files["moves.csv"]


def test_plain_fetch_of_a_current_snapshot_makes_no_request(tmp_path: Path) -> None:
    github = FakeGitHub(COMMIT, {COMMIT: build_files("v1")})
    store = SnapshotStore(tmp_path)
    fetch_snapshot(store, github.client(), is_update=True)
    manifest_bytes = store.manifest_path.read_bytes()
    github.requests.clear()

    report = fetch_snapshot(store, github.client(), is_update=False)

    assert github.requests == []
    assert report.requests == 0
    assert store.manifest_path.read_bytes() == manifest_bytes


def test_plain_fetch_downloads_a_fresh_clone_at_the_pinned_commit(tmp_path: Path) -> None:
    pinned = build_files("v1")
    github = FakeGitHub(NEXT_COMMIT, {COMMIT: pinned, NEXT_COMMIT: build_files("v2")})
    store = SnapshotStore(tmp_path)
    pin(store, COMMIT, pinned)

    report = fetch_snapshot(store, github.client(), is_update=False)

    assert report.commit == COMMIT
    assert report.files_downloaded == len(SNAPSHOT_FILES)
    assert all(COMMIT in url for url in github.requests)
    assert store.read_table("moves.csv") == ({"id": "1", "identifier": "v1-moves.csv"},)


def test_plain_fetch_pins_a_file_the_manifest_lacks_at_the_pinned_commit(tmp_path: Path) -> None:
    pinned = build_files("v1")
    github = FakeGitHub(NEXT_COMMIT, {COMMIT: pinned, NEXT_COMMIT: build_files("v2")})
    store = SnapshotStore(tmp_path)
    fetch_snapshot(store, FakeGitHub(COMMIT, {COMMIT: pinned}).client(), is_update=True)
    old = {name: content for name, content in pinned.items() if name != "abilities.csv"}
    pin(store, COMMIT, old)

    report = fetch_snapshot(store, github.client(), is_update=False)

    manifest = store.read_manifest()
    assert manifest is not None
    assert manifest.commit == COMMIT
    assert manifest.files == {name: compute_sha256(content) for name, content in pinned.items()}
    assert report.files_downloaded == report.requests == 1
    assert github.requests == [
        f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/data/v2/csv/abilities.csv"
    ]
    assert store.read_table("abilities.csv") == ({"id": "1", "identifier": "v1-abilities.csv"},)


def test_a_download_that_differs_from_its_pin_writes_nothing(tmp_path: Path) -> None:
    github = FakeGitHub(COMMIT, {COMMIT: build_files("changed")})
    store = SnapshotStore(tmp_path)
    pin(store, COMMIT, build_files("v1"))

    with pytest.raises(SnapshotError, match="does not match its pin"):
        fetch_snapshot(store, github.client(), is_update=False)

    assert not store.build_file_path("moves.csv").exists()


def test_reading_a_changed_file_names_both_hashes(tmp_path: Path) -> None:
    files = build_files("v1")
    github = FakeGitHub(COMMIT, {COMMIT: files})
    store = SnapshotStore(tmp_path)
    fetch_snapshot(store, github.client(), is_update=True)
    store.build_file_path("moves.csv").write_bytes(b"id\n2\n")

    with pytest.raises(SnapshotError) as raised:
        store.read_table("moves.csv")

    message = str(raised.value)
    assert compute_sha256(files["moves.csv"]) in message
    assert compute_sha256(b"id\n2\n") in message


def test_reading_without_a_snapshot_gives_the_fetch_hint(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)

    with pytest.raises(SnapshotError, match="muon fetch pokeapi --update"):
        store.read_table("moves.csv")


def test_plain_fetch_without_a_pin_gives_the_update_hint(tmp_path: Path) -> None:
    github = FakeGitHub(COMMIT, {COMMIT: build_files("v1")})

    with pytest.raises(SnapshotError, match="--update"):
        fetch_snapshot(SnapshotStore(tmp_path), github.client(), is_update=False)


def test_a_failed_update_names_the_file_and_keeps_the_old_pin(tmp_path: Path) -> None:
    pinned = build_files("v1")
    github = FakeGitHub(NEXT_COMMIT, {COMMIT: pinned, NEXT_COMMIT: build_files("v2")})
    store = SnapshotStore(tmp_path)
    pin(store, COMMIT, pinned)
    manifest_bytes = store.manifest_path.read_bytes()
    github.missing.add("moves.csv")

    with pytest.raises(SnapshotError, match=r"moves\.csv .* HTTP 404"):
        fetch_snapshot(store, github.client(), is_update=True)

    assert store.manifest_path.read_bytes() == manifest_bytes
