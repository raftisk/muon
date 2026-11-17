"""Violations and the report a validation run returns."""

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
