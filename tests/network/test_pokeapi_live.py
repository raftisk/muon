from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.api import build_http_client
from muon.ingestion.pokeapi import snapshot
from muon.ingestion.pokeapi.snapshot import (
    SnapshotManifest,
    SnapshotStore,
    compute_sha256,
    download_file,
    resolve_commit,
)

pytestmark = pytest.mark.network

FILE_NAME = "move_damage_classes.csv"


def test_live_repository_serves_a_csv_file_at_its_current_commit(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    with build_http_client() as client:
        commit = resolve_commit(client)
        content = download_file(client, commit, FILE_NAME)
    snapshot.write_bytes_atomically(store.build_file_path(FILE_NAME), content)
    store.write_manifest(
        SnapshotManifest(
            repository=snapshot.REPOSITORY,
            commit=commit,
            files={FILE_NAME: compute_sha256(content)},
        )
    )

    rows = store.read_table(FILE_NAME)

    assert {row["identifier"] for row in rows} >= {"physical", "special", "status"}
