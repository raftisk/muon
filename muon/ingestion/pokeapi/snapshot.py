"""A pinned snapshot of the PokeAPI CSV files.

`<root>/manifest.json` pins 1 commit of the PokeAPI repository and the SHA-256 of every
file in `SNAPSHOT_FILES`, and is committed. `<root>/csv/` holds the files and is
gitignored. A plain fetch downloads the pinned files a clone lacks, and pins a file of
`SNAPSHOT_FILES` that the manifest lacks at the pinned commit, so adding a file keeps every
other pin; `--update` moves the pin to the current commit of the default branch. Every read
checks a file against its hash, so a parser never reads bytes that differ from the pin. Only
`fetch_snapshot` touches the network.
"""

import csv
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from muon.config import PROJECT_ROOT
from muon.ingestion.bulbapedia.pages import write_text_atomically

REPOSITORY = "PokeAPI/pokeapi"
DEFAULT_BRANCH = "master"
COMMIT_URL_TEMPLATE = "https://api.github.com/repos/{repository}/commits/{branch}"
RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/{repository}/{commit}/data/v2/csv/{file_name}"
GITHUB_HEADERS = {"Accept": "application/vnd.github+json"}
OK_STATUS_CODE = 200

DEFAULT_ROOT = PROJECT_ROOT / "data" / "pokeapi"
MANIFEST_FILE_NAME = "manifest.json"
CSV_DIR_NAME = "csv"
TEMP_FILE_SUFFIX = ".tmp"
TEXT_ENCODING = "utf-8"
JSON_INDENT = 2
FETCH_HINT = "run `muon fetch pokeapi`"
UPDATE_HINT = "run `muon fetch pokeapi --update`"

SNAPSHOT_FILES = (
    "abilities.csv",
    "ability_prose.csv",
    "items.csv",
    "machines.csv",
    "move_damage_classes.csv",
    "move_effect_prose.csv",
    "move_meta.csv",
    "moves.csv",
    "types.csv",
    "version_groups.csv",
)


class SnapshotError(Exception):
    """A snapshot that is missing, does not match its pin or cannot be downloaded."""


class SnapshotManifest(BaseModel):
    """The pin: the repository, 1 commit and the SHA-256 hex of each file at that commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str
    commit: str
    files: dict[str, str]


class CommitResponse(BaseModel):
    """The part of the GitHub commit response the fetch reads."""

    sha: str


@dataclass(frozen=True)
class SnapshotFetchReport:
    commit: str
    files_downloaded: int
    requests: int


def compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_bytes_atomically(path: Path, content: bytes) -> None:
    """Write `content` to `path`. A reader sees the old file or the new one, never part."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=TEMP_FILE_SUFFIX
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


class SnapshotStore:
    """The manifest and CSV files under `root`. Never touches the network."""

    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = root

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE_NAME

    def build_file_path(self, name: str) -> Path:
        return self.root / CSV_DIR_NAME / name

    def read_manifest(self) -> SnapshotManifest | None:
        """Return the pin, or `None` when no manifest exists yet."""
        if not self.manifest_path.exists():
            return None
        try:
            return SnapshotManifest.model_validate_json(self.manifest_path.read_text(TEXT_ENCODING))
        except ValidationError as error:
            raise SnapshotError(
                f"invalid snapshot manifest {self.manifest_path}: {error}"
            ) from error

    def write_manifest(self, manifest: SnapshotManifest) -> None:
        """Write the pin with sorted keys, so the same pin is always the same bytes."""
        text = json.dumps(
            manifest.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, indent=JSON_INDENT
        )
        write_text_atomically(self.manifest_path, text + "\n")

    def is_current(self, name: str, sha256: str) -> bool:
        path = self.build_file_path(name)
        return path.exists() and compute_sha256(path.read_bytes()) == sha256

    def read_table(self, name: str) -> tuple[dict[str, str], ...]:
        """Return the rows of 1 pinned CSV file, after checking its SHA-256 against the pin."""
        manifest = self.read_manifest()
        if manifest is None:
            raise SnapshotError(f"no PokeAPI snapshot at {self.root}: {UPDATE_HINT}")
        expected = manifest.files.get(name)
        if expected is None:
            raise SnapshotError(f"the snapshot manifest does not pin {name}: {UPDATE_HINT}")
        path = self.build_file_path(name)
        if not path.exists():
            raise SnapshotError(f"{name} is missing from the snapshot: {FETCH_HINT}")
        content = path.read_bytes()
        actual = compute_sha256(content)
        if actual != expected:
            raise SnapshotError(
                f"{name} does not match its pin: expected SHA-256 {expected}, found {actual}"
            )
        reader = csv.DictReader(io.StringIO(content.decode(TEXT_ENCODING), newline=""))
        return tuple(dict(row) for row in reader)


def resolve_commit(client: httpx.Client) -> str:
    """Return the commit at the head of the default branch of the PokeAPI repository."""
    url = COMMIT_URL_TEMPLATE.format(repository=REPOSITORY, branch=DEFAULT_BRANCH)
    response = client.get(url, headers=GITHUB_HEADERS)
    if response.status_code != OK_STATUS_CODE:
        raise SnapshotError(
            f"resolving the {DEFAULT_BRANCH} commit failed: HTTP {response.status_code}"
        )
    try:
        return CommitResponse.model_validate_json(response.content).sha
    except ValidationError as error:
        raise SnapshotError(f"unexpected commit response from {url}: {error}") from error


def download_file(client: httpx.Client, commit: str, name: str) -> bytes:
    url = RAW_URL_TEMPLATE.format(repository=REPOSITORY, commit=commit, file_name=name)
    response = client.get(url)
    if response.status_code != OK_STATUS_CODE:
        raise SnapshotError(f"download of {name} at {commit} failed: HTTP {response.status_code}")
    return response.content


def download_pinned(
    client: httpx.Client, manifest: SnapshotManifest, names: list[str]
) -> dict[str, bytes]:
    """Download `names` at the pinned commit; a file whose hash differs from its pin fails."""
    contents: dict[str, bytes] = {}
    for name in names:
        content = download_file(client, manifest.commit, name)
        actual = compute_sha256(content)
        if actual != manifest.files[name]:
            raise SnapshotError(
                f"{name} at {manifest.commit} does not match its pin: "
                f"expected SHA-256 {manifest.files[name]}, found {actual}"
            )
        contents[name] = content
    return contents


def pin_missing_files(
    client: httpx.Client, manifest: SnapshotManifest, names: list[str]
) -> tuple[SnapshotManifest, dict[str, bytes]]:
    """Download `names` at the pinned commit and return the manifest with their hashes."""
    contents = {name: download_file(client, manifest.commit, name) for name in names}
    hashes = {name: compute_sha256(content) for name, content in contents.items()}
    return manifest.model_copy(update={"files": {**manifest.files, **hashes}}), contents


def fetch_snapshot(
    store: SnapshotStore, client: httpx.Client, is_update: bool
) -> SnapshotFetchReport:
    """Make the files on disk match the pin, or with `is_update` move the pin first.

    Every file is downloaded and checked before any is written, so a failed fetch leaves the
    manifest and the files as they were.
    """
    if is_update:
        commit = resolve_commit(client)
        contents = {name: download_file(client, commit, name) for name in SNAPSHOT_FILES}
        manifest = SnapshotManifest(
            repository=REPOSITORY,
            commit=commit,
            files={name: compute_sha256(content) for name, content in contents.items()},
        )
        requests = len(SNAPSHOT_FILES) + 1
    else:
        pinned = store.read_manifest()
        if pinned is None:
            raise SnapshotError(f"no PokeAPI snapshot pin at {store.root}: {UPDATE_HINT}")
        unpinned = [name for name in SNAPSHOT_FILES if name not in pinned.files]
        stale = [
            name
            for name in SNAPSHOT_FILES
            if name in pinned.files and not store.is_current(name, pinned.files[name])
        ]
        manifest, added = pin_missing_files(client, pinned, unpinned)
        contents = {**download_pinned(client, pinned, stale), **added}
        requests = len(stale) + len(unpinned)
    for name, content in contents.items():
        write_bytes_atomically(store.build_file_path(name), content)
    if manifest != store.read_manifest():
        store.write_manifest(manifest)
    return SnapshotFetchReport(
        commit=manifest.commit, files_downloaded=len(contents), requests=requests
    )
