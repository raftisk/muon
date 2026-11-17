"""The eval questions of a universe plugin and their loader.

A plugin names its eval sets in the `[evals]` table of `universe.toml`: set name to question
file. A file holds one `[[question]]` table per question, in the order of difficulty, easy
first. Question ids are unique across the sets of a run.
"""

import hashlib
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from muon.files import TEXT_ENCODING
from muon.plugins import EVALS_FIELD, UniversePlugin

QUESTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"


class EvalError(Exception):
    """An eval file, a gold query or a question selection that cannot be used."""


class EvalModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalQuestion(EvalModel):
    """One question with its gold query and a reference answer for the reviewer.

    `id` names the answer file of a run, so it holds no path character. `gold_query` returns
    the nodes the answer rests on, and returns no rows when `has_empty_gold` is set: the
    question asks about something the graph does not hold. `reference` is read by people
    only, never by code.
    """

    id: str = Field(pattern=QUESTION_ID_PATTERN)
    family: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    gold_query: str = Field(min_length=1)
    runbook: str | None = None
    has_empty_gold: bool = False


class EvalFile(EvalModel):
    question: list[EvalQuestion] = Field(min_length=1)


class EvalSet(EvalModel):
    """The questions of 1 named set in file order, and the sha256 of the file they came from."""

    name: str
    questions: tuple[EvalQuestion, ...]
    source_hash: str


@dataclass(frozen=True)
class SelectedQuestion:
    """A question of a run with the name of the set it belongs to."""

    set_name: str
    question: EvalQuestion


def load_eval_set(name: str, path: Path) -> EvalSet:
    """Read and validate the question file `path` as the set `name`.

    Raises `EvalError` when the file is unreadable or invalid, or 2 questions share an id.
    """
    try:
        content = path.read_bytes()
    except OSError as error:
        raise EvalError(f"Cannot read the eval file {path}: {error}") from error
    try:
        raw_file = tomllib.loads(content.decode(TEXT_ENCODING))
        eval_file = EvalFile.model_validate(raw_file)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise EvalError(f"Invalid TOML in {path}: {error}") from error
    except ValidationError as error:
        raise EvalError(f"Invalid eval file {path}: {error}") from error
    repeated = sorted(
        question_id
        for question_id, count in Counter(item.id for item in eval_file.question).items()
        if count > 1
    )
    if repeated:
        raise EvalError(f"Eval file {path} repeats question ids: {', '.join(repeated)}")
    return EvalSet(
        name=name,
        questions=tuple(eval_file.question),
        source_hash=hashlib.sha256(content).hexdigest(),
    )


def load_eval_sets(plugin: UniversePlugin, names: Sequence[str]) -> tuple[EvalSet, ...]:
    """Load the sets in `names`, or every declared set in manifest order when `names` is empty.

    Raises `EvalError` when the plugin declares no set, a name is not declared, or 2 loaded
    sets share a question id.
    """
    paths = plugin.eval_paths
    if not paths:
        raise EvalError(
            f"Universe {plugin.prefix!r} has no eval set (field {EVALS_FIELD!r} of its manifest)"
        )
    unknown = sorted(set(names) - set(paths))
    if unknown:
        raise EvalError(f"Unknown eval sets: {', '.join(unknown)}; declared: {', '.join(paths)}")
    chosen = [name for name in paths if not names or name in names]
    sets = tuple(load_eval_set(name, paths[name]) for name in chosen)
    owners: dict[str, str] = {}
    for eval_set in sets:
        for question in eval_set.questions:
            first_set = owners.setdefault(question.id, eval_set.name)
            if first_set != eval_set.name:
                raise EvalError(
                    f"Question id {question.id} is in the sets {first_set!r} and {eval_set.name!r}"
                )
    return sets


def select_questions(sets: Sequence[EvalSet], ids: Sequence[str]) -> tuple[SelectedQuestion, ...]:
    """Return the questions named in `ids` in set and file order, or every question when empty.

    Raises `EvalError` for an id that none of the sets holds.
    """
    selected = tuple(
        SelectedQuestion(eval_set.name, question)
        for eval_set in sets
        for question in eval_set.questions
    )
    known = [item.question.id for item in selected]
    unknown = sorted(set(ids) - set(known))
    if unknown:
        raise EvalError(f"Unknown question ids: {', '.join(unknown)}; known: {', '.join(known)}")
    if not ids:
        return selected
    return tuple(item for item in selected if item.question.id in ids)
