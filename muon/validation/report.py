"""Violations and the report a validation run returns."""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from muon.ingestion.batch import NodeRef

RULE_IRI = "IRI"
RULE_ID = "ID"
RULE_EDGE_TYPING = "T1"
RULE_ABOX = "ABOX"
RULE_REFERENCE = "REF"


class Violation(BaseModel):
    """One rule a batch breaks.

    `rule_id` is one of the `RULE_*` constants or the `sh:name` of a shape. `node` is
    `None` for a claim or a scope class, which no single node owns. `property_name` is a
    property local name and `value` the offending value as a string, when there is one.
    """

    model_config = ConfigDict(frozen=True)

    rule_id: str
    node: NodeRef | None = None
    property_name: str | None = None
    value: str | None = None
    message: str


class ValidationReport(BaseModel):
    """Every violation found in one batch."""

    model_config = ConfigDict(frozen=True)

    violations: tuple[Violation, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.violations


class LabelSetFinding(BaseModel):
    """A set of ontology labels that no class produces, with the nodes that carry it.

    `labels` is sorted and `sample_ids` holds up to 5 node ids.
    """

    model_config = ConfigDict(frozen=True)

    labels: tuple[str, ...]
    count: int
    sample_ids: tuple[str, ...]


class EdgeTypingFinding(BaseModel):
    """Relationships of 1 type whose `end` node carries none of the labels the T-Box allows."""

    model_config = ConfigDict(frozen=True)

    relationship_type: str
    end: Literal["start", "end"]
    count: int
    sample_ids: tuple[str, ...]


class SweepReport(BaseModel):
    """What the checks that span batches found in the graph of 1 universe.

    `stubs` maps a class label to the ids of its nodes that hold only the stub keys, and
    `orphan_stubs` lists the stubs with no relationship. Neither makes the sweep fail.
    """

    model_config = ConfigDict(frozen=True)

    label_sets: tuple[LabelSetFinding, ...] = ()
    edge_typing: tuple[EdgeTypingFinding, ...] = ()
    stubs: Mapping[str, tuple[str, ...]] = {}
    orphan_stubs: tuple[NodeRef, ...] = ()
    missing_abox: tuple[NodeRef, ...] = ()

    @property
    def is_failing(self) -> bool:
        return bool(self.label_sets or self.edge_typing or self.missing_abox)
