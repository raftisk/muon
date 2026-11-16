"""Universe selection and the per-question scope.

`select_universe` is the only place that chooses a universe. `build_scope` turns the
choice into a `UniverseScope` that the retrieve step binds into its tools, so the model
never passes a universe and cannot leave the one chosen.
"""

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from muon.agent.universes import UniversePlugin, UniverseRegistry
from muon.config import AgentSettings, Settings
from muon.graph import GraphClient
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model
from muon.retrieval.context import SchemaContext, SchemaRequest, load_schema_context


class RoutingError(Exception):
    """No universe can be chosen for a question."""


@dataclass(frozen=True)
class UniverseScope:
    """The universe of 1 question.

    `allowed_terms` holds every label (leaf, ancestor and universe) and relationship type
    of the universe graph.
    """

    plugin: UniversePlugin
    materializer: LabelMaterializer
    schema_context: SchemaContext
    universe_label: str
    allowed_terms: frozenset[str]


class StepTracker(Protocol):
    """Shows the steps of a run.

    A step is shown from entry until exit. An exception leaving the block marks it failed.
    """

    def step(self, name: str) -> AbstractContextManager[None]: ...


class NullStepTracker:
    """A tracker that shows nothing."""

    def step(self, name: str) -> AbstractContextManager[None]:
        return nullcontext()


NO_PROGRESS = NullStepTracker()


@dataclass(frozen=True)
class AgentRuntime:
    """The per-run dependencies of the agent steps."""

    client: GraphClient
    chat_model: BaseChatModel
    agent_settings: AgentSettings
    progress: StepTracker = NO_PROGRESS


def select_universe(registry: UniverseRegistry, requested: str | None) -> UniversePlugin:
    """Return the requested plugin, or the only registered one when none is requested."""
    known = ", ".join(registry.list_prefixes()) or "none"
    if requested is not None:
        if requested not in registry.plugins:
            raise RoutingError(f"Unknown universe {requested!r}; known: {known}")
        return registry.plugins[requested]
    if len(registry.plugins) == 1:
        return next(iter(registry.plugins.values()))
    raise RoutingError(f"Several universes are registered ({known}); pass --universe PREFIX")


async def build_scope(
    client: GraphClient, plugin: UniversePlugin, settings: Settings
) -> UniverseScope:
    """Load the schema context of `plugin` and derive the allowed terms from it."""
    model = load_ontology_model(settings.ontology.ontology_dir, plugin.prefix)
    materializer = LabelMaterializer(model)
    request = SchemaRequest(plugin, materializer, should_refresh=False)
    context = await load_schema_context(client, request, settings.universes.schema_cache_dir)
    graph_schema = context.graph_schema
    allowed_terms = (
        graph_schema.list_labels()
        | graph_schema.list_relationship_types()
        | {materializer.universe_label}
    )
    return UniverseScope(
        plugin=plugin,
        materializer=materializer,
        schema_context=context,
        universe_label=materializer.universe_label,
        allowed_terms=frozenset(allowed_terms),
    )
