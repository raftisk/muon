"""The eval questions of a universe plugin and their loader.

A plugin names its question file in `universe.toml` (field `evals`). The file holds one
`[[question]]` table per question, in the order of difficulty, easy first.
"""

import hashlib
import tomllib
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from muon.agent.universes import EVALS_FIELD, UniversePlugin
from muon.files import TEXT_ENCODING

QUESTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"


class EvalError(Exception):
    """An eval file, a gold query or a question selection that cannot be used."""


class EvalModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalQuestion(EvalModel):
    """One question with its gold query and a reference answer for the reviewer.

    `id` names the answer file of a run, so it holds no path character. `gold_query` returns
    the nodes the answer rests on. `reference` is read by people only, never by code.
    """

    id: str = Field(pattern=QUESTION_ID_PATTERN)
    family: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    gold_query: str = Field(min_length=1)
    runbook: str | None = None


class EvalFile(EvalModel):
    question: list[EvalQuestion] = Field(min_length=1)


class EvalSet(EvalModel):
    """The questions of 1 plugin in file order, and the sha256 of the file they came from."""

    questions: tuple[EvalQuestion, ...]
    source_hash: str

    def select(self, ids: Sequence[str]) -> tuple[EvalQuestion, ...]:
        """Return the questions named in `ids`, in file order.

        Raises `EvalError` for an id that the file does not hold.
        """
        known = [question.id for question in self.questions]
        unknown = sorted(set(ids) - set(known))
        if unknown:
            raise EvalError(
                f"Unknown question ids: {', '.join(unknown)}; known: {', '.join(known)}"
            )
        return tuple(question for question in self.questions if question.id in ids)


def load_questions(plugin: UniversePlugin) -> EvalSet:
    """Read and validate the eval file of `plugin`.

    Raises `EvalError` when the plugin names no file, the file is unreadable or invalid, or
    2 questions share an id.
    """
    path = plugin.evals_path
    if path is None:
        raise EvalError(
            f"Universe {plugin.prefix!r} has no eval set (field {EVALS_FIELD!r} of its manifest)"
        )
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
        questions=tuple(eval_file.question), source_hash=hashlib.sha256(content).hexdigest()
    )
