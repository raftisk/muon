"""Atomic writes for files that readers may open while a run writes them, and the JSON
text of the files a person reads."""

import json
import os
import tempfile
from pathlib import Path

TEMP_FILE_SUFFIX = ".tmp"
TEXT_ENCODING = "utf-8"
JSON_INDENT = 2


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


def write_text_atomically(path: Path, text: str) -> None:
    """Write `text` to `path` as UTF-8, with the guarantee of `write_bytes_atomically`."""
    write_bytes_atomically(path, text.encode(TEXT_ENCODING))


def dump_json(data: object) -> str:
    """Return `data` as indented JSON with sorted keys, so equal data gives equal bytes."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=JSON_INDENT) + "\n"
