"""A toy universe plugin written into a temporary folder."""

import json
from pathlib import Path

from muon.plugins import MANIFEST_FILE_NAME
from tests.fixtures.toy_ontology import TOY_UNIVERSE

TOY_NAME = "Toy universe"
TOY_NOTES_FILE = "notes.md"
TOY_NAMESPACE = f"https://muon.dev/ns/{TOY_UNIVERSE}#"
TOY_SHAPES_NAMESPACE = f"https://muon.dev/ns/{TOY_UNIVERSE}-shapes#"
TOY_NOTES_TEXT = "### Widgets\n\nA widget is a gadget with a serial number.\n"


def write_toy_plugin(
    universes_dir: Path,
    prefix: str = TOY_UNIVERSE,
    folder_name: str | None = None,
    **manifest_overrides: str | dict[str, str],
) -> Path:
    """Write a plugin folder with a manifest and a notes file; return the folder.

    A dict override is written as a TOML table after the top-level keys.
    The folder is named `folder_name`, or `prefix` when it is `None`.
    """
    folder = universes_dir / (folder_name or prefix)
    folder.mkdir(parents=True)
    manifest = {
        "prefix": prefix,
        "name": TOY_NAME,
        "notes": TOY_NOTES_FILE,
        "namespace": TOY_NAMESPACE,
        "shapes_namespace": TOY_SHAPES_NAMESPACE,
        **manifest_overrides,
    }
    lines = [
        f"{key} = {json.dumps(value)}" for key, value in manifest.items() if isinstance(value, str)
    ]
    for key, table in manifest.items():
        if isinstance(table, dict):
            lines.extend(
                [
                    f"[{key}]",
                    *(f"{json.dumps(name)} = {json.dumps(path)}" for name, path in table.items()),
                ]
            )
    (folder / MANIFEST_FILE_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / TOY_NOTES_FILE).write_text(TOY_NOTES_TEXT, encoding="utf-8")
    return folder
