from pathlib import Path

import pytest

from muon.files import write_bytes_atomically, write_text_atomically


def test_write_creates_missing_parent_folders(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "file.bin"

    write_bytes_atomically(path, b"content")

    assert path.read_bytes() == b"content"


def test_second_write_replaces_the_content(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"

    write_bytes_atomically(path, b"first")
    write_bytes_atomically(path, b"second")

    assert path.read_bytes() == b"second"


def test_no_temp_file_remains_after_a_write(tmp_path: Path) -> None:
    write_bytes_atomically(tmp_path / "file.bin", b"content")

    assert [entry.name for entry in tmp_path.iterdir()] == ["file.bin"]


def test_failed_write_keeps_the_old_content_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file.bin"
    write_bytes_atomically(path, b"old")

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_bytes_atomically(path, b"new")

    assert path.read_bytes() == b"old"
    assert [entry.name for entry in tmp_path.iterdir()] == ["file.bin"]


def test_text_helper_writes_utf8_and_keeps_newlines(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"

    write_text_atomically(path, "Pokémon\n")

    assert path.read_bytes() == "Pokémon\n".encode()
