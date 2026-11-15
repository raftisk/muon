from pathlib import Path

import pytest

from muon.agent.universes import (
    MANIFEST_FILE_NAME,
    UniverseError,
    UniverseRegistry,
    discover_universes,
)
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.fixtures.toy_plugin import TOY_NAME, TOY_NOTES_TEXT, write_toy_plugin


@pytest.fixture
def ontology_dir(tmp_path: Path) -> Path:
    return copy_toy_ontology(tmp_path / "ontology")


@pytest.fixture
def universes_dir(tmp_path: Path) -> Path:
    path = tmp_path / "universes"
    path.mkdir()
    return path


def test_valid_plugin_is_discovered(universes_dir: Path, ontology_dir: Path) -> None:
    folder = write_toy_plugin(universes_dir)

    registry = discover_universes(universes_dir, ontology_dir)

    plugin = registry.get(TOY_UNIVERSE)
    assert registry.list_prefixes() == (TOY_UNIVERSE,)
    assert plugin.name == TOY_NAME
    assert plugin.notes_path == folder / "notes.md"
    assert plugin.notes_text == TOY_NOTES_TEXT
    assert plugin.ontology_paths[-1] == ontology_dir / "universes" / f"{TOY_UNIVERSE}.ttl"


def test_unknown_prefix_lists_the_known_ones(universes_dir: Path, ontology_dir: Path) -> None:
    write_toy_plugin(universes_dir)
    registry = discover_universes(universes_dir, ontology_dir)

    with pytest.raises(UniverseError, match=f"Unknown universe 'nope'; known: {TOY_UNIVERSE}"):
        registry.get("nope")


def test_empty_registry_says_none_is_known() -> None:
    with pytest.raises(UniverseError, match="known: none"):
        UniverseRegistry({}).get("pkmn")


def test_prefix_that_differs_from_the_folder_name_raises(
    universes_dir: Path, ontology_dir: Path
) -> None:
    write_toy_plugin(universes_dir, folder_name="muonalt")

    with pytest.raises(UniverseError, match="differs from the folder name 'muonalt'"):
        discover_universes(universes_dir, ontology_dir)


def test_prefix_that_fails_the_pattern_raises(universes_dir: Path, ontology_dir: Path) -> None:
    write_toy_plugin(universes_dir, prefix="Bad-Prefix")

    with pytest.raises(UniverseError, match="Invalid prefix 'Bad-Prefix'"):
        discover_universes(universes_dir, ontology_dir)


def test_extra_manifest_key_raises(universes_dir: Path, ontology_dir: Path) -> None:
    write_toy_plugin(universes_dir, description="not allowed")

    with pytest.raises(UniverseError, match="description"):
        discover_universes(universes_dir, ontology_dir)


def test_missing_notes_file_raises(universes_dir: Path, ontology_dir: Path) -> None:
    write_toy_plugin(universes_dir, notes="missing.md")

    with pytest.raises(UniverseError, match="Missing notes file .*missing.md"):
        discover_universes(universes_dir, ontology_dir)


def test_notes_path_outside_the_folder_raises(universes_dir: Path, ontology_dir: Path) -> None:
    write_toy_plugin(universes_dir, notes="../outside.md")

    with pytest.raises(UniverseError, match="leaves the plugin folder"):
        discover_universes(universes_dir, ontology_dir)


def test_missing_t_box_raises(universes_dir: Path, ontology_dir: Path) -> None:
    (ontology_dir / "universes" / f"{TOY_UNIVERSE}.ttl").unlink()
    write_toy_plugin(universes_dir)

    with pytest.raises(UniverseError, match="no usable T-Box"):
        discover_universes(universes_dir, ontology_dir)


def test_folder_without_manifest_raises(universes_dir: Path, ontology_dir: Path) -> None:
    (universes_dir / "empty").mkdir()

    with pytest.raises(UniverseError, match=f"empty has no {MANIFEST_FILE_NAME}"):
        discover_universes(universes_dir, ontology_dir)


def test_pycache_hidden_folders_and_plain_files_are_skipped(
    universes_dir: Path, ontology_dir: Path
) -> None:
    write_toy_plugin(universes_dir)
    (universes_dir / "__pycache__").mkdir()
    (universes_dir / ".hidden").mkdir()
    (universes_dir / "__init__.py").write_text("")

    registry = discover_universes(universes_dir, ontology_dir)

    assert registry.list_prefixes() == (TOY_UNIVERSE,)


def test_invalid_toml_names_the_file(universes_dir: Path, ontology_dir: Path) -> None:
    folder = write_toy_plugin(universes_dir)
    (folder / MANIFEST_FILE_NAME).write_text("prefix = ", encoding="utf-8")

    with pytest.raises(UniverseError, match="Invalid TOML in .*universe.toml"):
        discover_universes(universes_dir, ontology_dir)


def test_missing_universes_folder_raises(tmp_path: Path, ontology_dir: Path) -> None:
    with pytest.raises(UniverseError, match="does not exist"):
        discover_universes(tmp_path / "nowhere", ontology_dir)
