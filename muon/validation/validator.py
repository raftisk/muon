"""The validation gate: runs the 6 checks of a batch in order."""

from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph

from muon.graph import GraphClient
from muon.ingestion.batch import GraphBatch
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer, build_plugin_materializer
from muon.ontology.model import OntologyModel, build_ontology_model, load_tbox
from muon.plugins import UniversePlugin
from muon.validation.checks import check_abox_stubs, check_edge_typing, check_ids, check_iris
from muon.validation.projection import ProjectionRules, build_projection_rules, project_batch
from muon.validation.references import find_unresolved_stubs
from muon.validation.report import ValidationReport, Violation
from muon.validation.shapes import load_shapes, run_shapes, select_shapes


@dataclass(frozen=True)
class ValidatorInputs:
    """Everything a `Validator` reads, loaded once per run."""

    model: OntologyModel
    materializer: LabelMaterializer
    tbox: Graph
    shapes: Graph
    abox: GraphBatch
    rules: ProjectionRules
    shapes_namespace: str


@dataclass
class Validator:
    """Validates batches of 1 universe. Owns a cache of the shapes selected per scope."""

    inputs: ValidatorInputs
    client: GraphClient
    selected_shapes: dict[frozenset[str], Graph] = field(default_factory=dict)

    async def validate(self, batch: GraphBatch) -> ValidationReport:
        """Run the checks in order and return every violation found.

        A violation in the T-Box IRIs ends the run, because the later checks cannot read an
        IRI the T-Box does not declare. Every other check runs even after an earlier one fails.
        """
        inputs = self.inputs
        iri_violations = check_iris(batch, inputs.model)
        if iri_violations:
            return ValidationReport(violations=iri_violations)

        violations: list[Violation] = [
            *check_ids(batch, inputs.model, inputs.materializer),
            *check_edge_typing(batch, inputs.model),
            *check_abox_stubs(batch, inputs.abox),
            *await find_unresolved_stubs(self.client, inputs.materializer, batch),
        ]
        projected = project_batch(batch, inputs.model, inputs.rules)
        violations.extend(run_shapes(projected, self.select_cached(batch.scope), inputs.tbox))
        return ValidationReport(violations=tuple(violations))

    def select_cached(self, scope: frozenset[str]) -> Graph:
        if scope not in self.selected_shapes:
            self.selected_shapes[scope] = select_shapes(
                self.inputs.shapes, scope, self.inputs.model, self.inputs.shapes_namespace
            )
        return self.selected_shapes[scope]


def build_validator(client: GraphClient, ontology_dir: Path, plugin: UniversePlugin) -> Validator:
    """Load the T-Box, A-Box, shapes and projection rules of `plugin` into a `Validator`."""
    universe = plugin.prefix
    tbox = load_tbox(ontology_dir, universe)
    model = build_ontology_model(tbox, universe)
    inputs = ValidatorInputs(
        model=model,
        materializer=build_plugin_materializer(model, plugin.manifest),
        tbox=tbox,
        shapes=load_shapes(ontology_dir, universe),
        abox=read_abox(model, ontology_dir),
        rules=build_projection_rules(plugin.manifest),
        shapes_namespace=plugin.manifest.shapes_namespace,
    )
    return Validator(inputs=inputs, client=client)
