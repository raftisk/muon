"""The "Pokémon with X" table of each ability page against the species parse output.

The species pages own `hasAbility`, so this check only reports. It works at species
level: a table row of a form counts for its base species, and so does a `hasAbility` edge
of a form node. The species side is read from the species parse output files; without
them the check is skipped and the report says so.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from muon.ingestion.bulbapedia.abilities.page import AbilityPageExtract
from muon.ingestion.bulbapedia.batching import PKMN
from muon.ingestion.bulbapedia.runs import OUTPUT_SUFFIX, ReportModel
from muon.ingestion.bulbapedia.species.forms import build_species_id

HAS_ABILITY = PKMN + "hasAbility"
SPECIES_TITLE_SUFFIX = " (Pokémon)"
TEXT_ENCODING = "utf-8"


class CrossCheckError(Exception):
    """A species output file that does not have the shape the check reads."""


class OutputModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class OutputNodeRef(OutputModel):
    id: str


class OutputEdge(OutputModel):
    property_iri: str
    end: OutputNodeRef


class OutputBatch(OutputModel):
    edges: tuple[OutputEdge, ...]


class SpeciesOutputFile(OutputModel):
    """The part of a species parse output file the check reads."""

    title: str
    species_batch: OutputBatch


class HolderSide(StrEnum):
    TABLE_ONLY = "table_only"
    SPECIES_ONLY = "species_only"


class HolderDifference(ReportModel):
    """A species the ability table and the species output disagree on."""

    ability_id: str
    species_id: str
    side: HolderSide


class SpeciesAbilityIndex(BaseModel):
    """The base species ids of the species output per Ability id they point at."""

    model_config = ConfigDict(frozen=True)

    species_by_ability: Mapping[str, frozenset[str]]


def read_species_abilities(output_dir: Path) -> SpeciesAbilityIndex | None:
    """Read the `hasAbility` edges of every species output file, or `None` when there is
    none. Raises `CrossCheckError` for a file of another shape."""
    paths = sorted(output_dir.glob(f"*{OUTPUT_SUFFIX}")) if output_dir.is_dir() else []
    if not paths:
        return None
    species: defaultdict[str, set[str]] = defaultdict(set)
    for path in paths:
        try:
            output = SpeciesOutputFile.model_validate_json(path.read_text(TEXT_ENCODING))
        except ValidationError as error:
            raise CrossCheckError(f"cannot read species output {path}: {error}") from error
        species_id = build_species_id(output.title.removesuffix(SPECIES_TITLE_SUFFIX))
        for edge in output.species_batch.edges:
            if edge.property_iri == HAS_ABILITY:
                species[edge.end.id].add(species_id)
    return SpeciesAbilityIndex(
        species_by_ability={ability: frozenset(ids) for ability, ids in species.items()}
    )


def cross_check_species(
    extracts: Iterable[AbilityPageExtract], index: SpeciesAbilityIndex
) -> tuple[HolderDifference, ...]:
    """Return, per ability in order, the species only its table names, then the species
    only the species output names, each sorted."""
    differences: list[HolderDifference] = []
    for extract in extracts:
        table = {build_species_id(holder.species_name) for holder in extract.holders}
        graph = index.species_by_ability.get(extract.id, frozenset())
        differences.extend(
            HolderDifference(ability_id=extract.id, species_id=species_id, side=side)
            for side, ids in (
                (HolderSide.TABLE_ONLY, table - graph),
                (HolderSide.SPECIES_ONLY, graph - table),
            )
            for species_id in sorted(ids)
        )
    return tuple(differences)
