"""The events of an `ask()` run: step brackets and answer text, told apart by `kind`."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class RunEventModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class StepStarted(RunEventModel):
    kind: Literal["step_started"] = "step_started"
    name: str


class StepFinished(RunEventModel):
    kind: Literal["step_finished"] = "step_finished"
    name: str


class TextDelta(RunEventModel):
    """Answer text that is safe to show: handles are already rewritten into links."""

    kind: Literal["text_delta"] = "text_delta"
    text: str
