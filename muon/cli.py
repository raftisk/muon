"""Command line entry point: `muon seed <prefix>`, `muon purge <prefix>`,
`muon fetch bulbapedia <page-type> [--update] [--title TITLE ...] [--from-export FILE]`,
`muon fetch pokeapi [--update]`, `muon parse bulbapedia species|move|ability|item`,
`muon ingest bulbapedia species|move|ability|item`, `muon schema <prefix> [--refresh] [--json]` and
`muon ask QUESTION [--universe PREFIX] [--json] [--show-cypher]` and
`muon eval PREFIX [--set NAME ...] [--question ID ...] [--output-dir DIR]`."""

import argparse
import asyncio
import getpass
import secrets
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import SecretStr, ValidationError

from muon.agent.ask import AskRequest, ask
from muon.agent.events import AskResult, collect_answer
from muon.agent.routing import RoutingError, StepTracker, create_agent_runtime, select_universe
from muon.config import AgentSettings, OntologySettings, Settings, get_settings
from muon.console import (
    STDERR,
    STDOUT,
    EvalDisplay,
    StepDisplay,
    build_answer_renderable,
    build_eval_summary,
    format_validator_report,
)
from muon.evaluation.questions import EvalError, load_eval_sets, select_questions
from muon.evaluation.run import (
    DEFAULT_EVAL_DIR,
    EvalRequest,
    EvalRun,
    run_eval,
    write_run,
)
from muon.graph import GraphError, Neo4jGraphClient, WriteSummary
from muon.ingestion.bulbapedia.abilities.batches import SupCodeError
from muon.ingestion.bulbapedia.abilities.crosscheck import CrossCheckError
from muon.ingestion.bulbapedia.abilities.ingest import (
    AbilityIngestRun,
    ingest_abilities,
    write_ability_ingest_report,
)
from muon.ingestion.bulbapedia.abilities.run import PAGE_TYPE as ABILITY_PAGE_TYPE
from muon.ingestion.bulbapedia.abilities.run import (
    AbilityParseReport,
    AbilityParseRequest,
    AbilityParseRun,
    AbilityRunInputs,
    parse_ability_pages,
)
from muon.ingestion.bulbapedia.abilities.run import ParseRunError as AbilityParseRunError
from muon.ingestion.bulbapedia.abilities.run import write_outputs as write_ability_outputs
from muon.ingestion.bulbapedia.api import BulbapediaAPIError, BulbapediaClient, build_http_client
from muon.ingestion.bulbapedia.export import ExportError
from muon.ingestion.bulbapedia.fetch import (
    PAGE_TYPES,
    FetchError,
    FetchReport,
    FetchRequest,
    fetch_pages,
    import_pages,
)
from muon.ingestion.bulbapedia.ingest import (
    BatchRejection,
    IngestReport,
    IngestRequest,
    MissingSeedError,
    PassProgress,
    StubIngestReport,
)
from muon.ingestion.bulbapedia.items.ingest import (
    ItemIngestRun,
    ingest_items,
    write_item_ingest_report,
)
from muon.ingestion.bulbapedia.items.run import PAGE_TYPE as ITEM_PAGE_TYPE
from muon.ingestion.bulbapedia.items.run import (
    ItemParseReport,
    ItemParseRequest,
    ItemParseRun,
    ItemRunInputs,
    parse_item_pages,
)
from muon.ingestion.bulbapedia.items.run import ParseRunError as ItemParseRunError
from muon.ingestion.bulbapedia.items.run import write_outputs as write_item_outputs
from muon.ingestion.bulbapedia.locations.finds import FindTableError
from muon.ingestion.bulbapedia.locations.ingest import (
    LocationIngestRun,
    ingest_locations,
    write_location_ingest_report,
)
from muon.ingestion.bulbapedia.locations.page import LocationTableError
from muon.ingestion.bulbapedia.locations.run import (
    COVERAGE_FILE_NAME as LOCATION_COVERAGE_FILE_NAME,
)
from muon.ingestion.bulbapedia.locations.run import PAGE_TYPE as LOCATION_PAGE_TYPE
from muon.ingestion.bulbapedia.locations.run import (
    LocationParseReport,
    LocationParseRequest,
    LocationParseRun,
    LocationRunInputs,
    parse_location_pages,
)
from muon.ingestion.bulbapedia.locations.run import ParseRunError as LocationParseRunError
from muon.ingestion.bulbapedia.locations.run import write_outputs as write_location_outputs
from muon.ingestion.bulbapedia.moves.batches import GameCodeError, MachineFieldError
from muon.ingestion.bulbapedia.moves.ingest import (
    MoveIngestRun,
    ingest_moves,
    write_move_ingest_report,
)
from muon.ingestion.bulbapedia.moves.page import TableError
from muon.ingestion.bulbapedia.moves.run import (
    PAGE_TYPE as MOVE_PAGE_TYPE,
)
from muon.ingestion.bulbapedia.moves.run import (
    MachineConflictError,
    MoveParseReport,
    MoveParseRequest,
    MoveParseRun,
    MoveRunInputs,
    parse_move_pages,
)
from muon.ingestion.bulbapedia.moves.run import (
    ParseRunError as MoveParseRunError,
)
from muon.ingestion.bulbapedia.moves.run import (
    write_outputs as write_move_outputs,
)
from muon.ingestion.bulbapedia.pages import PageStore, PageStoreError
from muon.ingestion.bulbapedia.runs import DEFAULT_OUTPUT_ROOT, PageStatus
from muon.ingestion.bulbapedia.species.encounters import EncounterReport, EncounterTableError
from muon.ingestion.bulbapedia.species.forms import OverrideError
from muon.ingestion.bulbapedia.species.ingest import IngestRun, ingest_species, write_ingest_report
from muon.ingestion.bulbapedia.species.run import (
    CANDIDATE_FILE_NAME,
    ParseReport,
    ParseRequest,
    ParseRun,
    ParseRunError,
    RunInputs,
    parse_species_pages,
    write_outputs,
)
from muon.ingestion.bulbapedia.species.run import (
    PAGE_TYPE as PARSE_PAGE_TYPE,
)
from muon.ingestion.pokeapi.snapshot import (
    SnapshotError,
    SnapshotFetchReport,
    SnapshotStore,
    fetch_snapshot,
)
from muon.ingestion.runner import Runner, RunReport
from muon.ingestion.seed import ABoxError, read_abox, seed_universe
from muon.llm.client import ModelCallError
from muon.ontology.materializer import LabelMaterializer, build_plugin_materializer
from muon.ontology.model import OntologyError, extract_local_name, load_ontology_model
from muon.plugins import UniverseError, UniversePlugin, UniverseRegistry, discover_universes
from muon.retrieval.context import (
    JSON_INDENT as SCHEMA_JSON_INDENT,
)
from muon.retrieval.context import (
    CacheStatus,
    SchemaCacheError,
    SchemaContext,
    SchemaRequest,
    load_schema_context,
)
from muon.retrieval.evidence import (
    Cell,
    Evidence,
    EvidenceBundle,
    NodeCell,
    NodeKey,
    PathCell,
    RelationshipCell,
)
from muon.retrieval.resolve import NameIndexError
from muon.retrieval.schema import SchemaIntrospectionError
from muon.validation.report import SweepReport
from muon.validation.sweep import run_sweep, write_sweep_report

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
PROGRAM_NAME = "muon"
SEED_COMMAND = "seed"
PURGE_COMMAND = "purge"
VALIDATE_COMMAND = "validate"
FETCH_COMMAND = "fetch"
PARSE_COMMAND = "parse"
INGEST_COMMAND = "ingest"
SCHEMA_COMMAND = "schema"
ASK_COMMAND = "ask"
EVAL_COMMAND = "eval"
ASK_TARGET = "question"
BULBAPEDIA_SOURCE = "bulbapedia"
POKEAPI_SOURCE = "pokeapi"
ASK_JSON_INDENT = 2
TABLE_COLUMN_SEPARATOR = " | "
TABLE_PATH_SEPARATOR = " > "
TABLE_LIST_SEPARATOR = ", "
AGENT_VARIABLE_PREFIX = "MUON_AGENT_"
PASSWORD_PROMPT = "Neo4j password: "
PASSWORD_ENCODING = "utf-8"
PASSWORD_MISMATCH_MESSAGE = "the password does not match NEO4J_PASSWORD"
ABORT_MESSAGE = "aborted"
EXPORT_CONFLICT_MESSAGE = "--from-export cannot be combined with --update or --title"
MISSING_PAGE_TYPE_MESSAGE = "fetch bulbapedia needs a page type"
POKEAPI_OPTIONS_MESSAGE = "fetch pokeapi takes no page type, --title or --from-export"


SPECIES_PARSE_ERRORS = (
    ParseRunError,
    OverrideError,
    EncounterTableError,
    OntologyError,
    ABoxError,
    PageStoreError,
)
MOVE_PARSE_ERRORS = (
    MoveParseRunError,
    TableError,
    MachineFieldError,
    GameCodeError,
    MachineConflictError,
    SnapshotError,
    PageStoreError,
    OntologyError,
    ABoxError,
)

LOCATION_PARSE_ERRORS = (
    LocationParseRunError,
    LocationTableError,
    FindTableError,
    PageStoreError,
    OntologyError,
    ABoxError,
)

ITEM_PARSE_ERRORS = (
    ItemParseRunError,
    GameCodeError,
    PageStoreError,
    OntologyError,
    ABoxError,
)

ABILITY_PARSE_ERRORS = (
    AbilityParseRunError,
    SupCodeError,
    CrossCheckError,
    SnapshotError,
    PageStoreError,
    OntologyError,
    ABoxError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description="muon knowledge graph tools")
    commands = parser.add_subparsers(dest="command", required=True)
    seed = commands.add_parser(
        SEED_COMMAND, help="load a universe's A-Box into Neo4j and keep the graph in sync with it"
    )
    seed.add_argument("prefix", help="universe prefix, the stem of individuals/<prefix>.ttl")
    purge = commands.add_parser(
        PURGE_COMMAND,
        help="delete every node of a universe after a Neo4j password prompt",
    )
    purge.add_argument("prefix", help="universe prefix, the stem of universes/<prefix>.ttl")
    validate = commands.add_parser(
        VALIDATE_COMMAND,
        help="check the graph of a universe for faults that span batches and report them",
    )
    validate.add_argument("prefix", help="universe prefix, the stem of universes/<prefix>.ttl")
    fetch = commands.add_parser(
        FETCH_COMMAND, help="download pinned source pages into data/<source>/ for ingestion"
    )
    fetch.add_argument(
        "source", choices=(BULBAPEDIA_SOURCE, POKEAPI_SOURCE), help="the source to fetch from"
    )
    fetch.add_argument(
        "page_type",
        nargs="?",
        choices=sorted(PAGE_TYPES),
        help="the set of pages to fetch (bulbapedia only)",
    )
    fetch.add_argument(
        "--update", action="store_true", help="move the pins to the latest revisions or commit"
    )
    fetch.add_argument(
        "--title",
        action="append",
        default=[],
        dest="titles",
        metavar="TITLE",
        help="fetch only this page (repeatable)",
    )
    fetch.add_argument(
        "--from-export",
        type=Path,
        action="append",
        default=[],
        dest="export_paths",
        metavar="FILE",
        help="pin the pages of the page type from a Special:Export file (repeatable)",
    )
    parse = commands.add_parser(
        PARSE_COMMAND, help="parse pinned source pages into batches and a report under data/"
    )
    parse.add_argument("source", choices=(BULBAPEDIA_SOURCE,), help="the wiki to parse")
    parse.add_argument(
        "page_type",
        choices=(
            PARSE_PAGE_TYPE,
            MOVE_PAGE_TYPE,
            ABILITY_PAGE_TYPE,
            ITEM_PAGE_TYPE,
            LOCATION_PAGE_TYPE,
        ),
        help="the set of pages to parse",
    )
    parse.add_argument(
        "--title",
        action="append",
        default=[],
        dest="titles",
        metavar="TITLE",
        help="write outputs for only this page (repeatable)",
    )
    ingest = commands.add_parser(
        INGEST_COMMAND,
        help="parse pinned source pages and write their batches into Neo4j, then report",
    )
    ingest.add_argument("source", choices=(BULBAPEDIA_SOURCE,), help="the wiki to ingest")
    ingest.add_argument(
        "page_type",
        choices=(
            PARSE_PAGE_TYPE,
            MOVE_PAGE_TYPE,
            ABILITY_PAGE_TYPE,
            ITEM_PAGE_TYPE,
            LOCATION_PAGE_TYPE,
        ),
        help="the set of pages to ingest",
    )
    ingest.add_argument(
        "--title",
        action="append",
        default=[],
        dest="titles",
        metavar="TITLE",
        help="ingest only this page (repeatable)",
    )
    schema = commands.add_parser(
        SCHEMA_COMMAND, help="print the schema context of a universe, built from the live graph"
    )
    schema.add_argument("prefix", help="universe prefix, the folder name of the plugin")
    schema.add_argument(
        "--refresh", action="store_true", help="introspect the graph although the cache is current"
    )
    schema.add_argument(
        "--json",
        action="store_true",
        dest="is_json",
        help="print the schema as JSON instead of the prompt text",
    )
    ask_parser = commands.add_parser(
        ASK_COMMAND, help="answer a question from the graph of a universe"
    )
    ask_parser.add_argument("question", help="the question, in English")
    ask_parser.add_argument(
        "--universe", help="universe prefix; optional while only 1 universe is registered"
    )
    ask_parser.add_argument(
        "--json",
        action="store_true",
        dest="is_json",
        help="print the answer, the evidence and the stats as 1 JSON document",
    )
    ask_parser.add_argument(
        "--show-cypher",
        action="store_true",
        dest="should_show_cypher",
        help="list the queries that ran, after the answer",
    )
    eval_parser = commands.add_parser(
        EVAL_COMMAND, help="ask the eval questions of a universe and score the answers"
    )
    eval_parser.add_argument("prefix", help="universe prefix, the folder name of the plugin")
    eval_parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set_names",
        metavar="NAME",
        help="run only this eval set (repeatable; default: every set)",
    )
    eval_parser.add_argument(
        "--question",
        action="append",
        default=[],
        dest="question_ids",
        metavar="ID",
        help="ask only this question (repeatable)",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        metavar="DIR",
        help="folder that receives the run folder (default: data/evals)",
    )
    return parser


def describe_target(arguments: argparse.Namespace) -> str:
    """Name what a command works on: the universe prefix, or the source and page type."""
    if arguments.command == INGEST_COMMAND:
        return f"{arguments.source} {arguments.page_type}"
    if arguments.command == ASK_COMMAND:
        return str(arguments.universe or ASK_TARGET)
    return str(arguments.prefix)


def build_failure_prefix(command: str, prefix: str) -> str:
    return f"{command} {prefix} failed:"


def format_report(universe: str, report: RunReport) -> str:
    lines = [f"{SEED_COMMAND} {universe}: done"]
    lines.extend(
        f"{field.name.replace('_', ' ')}: {getattr(report, field.name)}"
        for field in fields(RunReport)
    )
    return "\n".join(lines)


def load_plugin(settings: Settings, prefix: str) -> UniversePlugin:
    registry = discover_universes(settings.universes.universes_dir, settings.ontology.ontology_dir)
    return registry.get(prefix)


async def run_seed(settings: Settings, prefix: str) -> RunReport:
    plugin = load_plugin(settings, prefix)
    async with Neo4jGraphClient(settings.neo4j) as client:
        return await seed_universe(
            client, settings.ontology.ontology_dir, prefix, plugin.manifest.key_classes
        )


def format_purge_report(prefix: str, summary: WriteSummary) -> str:
    return "\n".join(
        [
            f"{PURGE_COMMAND} {prefix}: done",
            f"nodes deleted: {summary.nodes_deleted}",
            f"relationships deleted: {summary.relationships_deleted}",
        ]
    )


def prompt_password() -> str:
    return getpass.getpass(PASSWORD_PROMPT)


def is_password_match(typed: str, expected: SecretStr) -> bool:
    return secrets.compare_digest(
        typed.encode(PASSWORD_ENCODING), expected.get_secret_value().encode(PASSWORD_ENCODING)
    )


async def run_purge(settings: Settings, prefix: str) -> WriteSummary:
    model = load_ontology_model(settings.ontology.ontology_dir, prefix)
    async with Neo4jGraphClient(settings.neo4j) as client:
        return await Runner(client, LabelMaterializer(model)).purge()


def run_fetch(request: FetchRequest) -> FetchReport:
    """Run an export run with no HTTP client, and any other run against the API."""
    if request.export_paths:
        return import_pages(request, PageStore())
    with build_http_client() as http_client:
        return fetch_pages(request, PageStore(), BulbapediaClient(http_client))


def format_fetch_report(request: FetchRequest, report: FetchReport) -> str:
    """The counts of a run, and for an update, title or export run the titles behind them."""
    lines = [
        f"{FETCH_COMMAND} {BULBAPEDIA_SOURCE} {report.page_type}: done",
        f"listed: {report.listed}",
        f"fetched: {report.fetched}",
        f"skipped: {report.skipped}",
        f"added: {len(report.added)}",
        f"changed: {len(report.changed)}",
        f"removed: {len(report.removed)}",
        f"redirects: {len(report.redirects)}",
        f"requests: {report.requests}",
    ]
    if request.export_paths:
        lines.append(f"ignored: {report.ignored}")
    lines.append(f"elapsed seconds: {report.elapsed_seconds:.1f}")
    if request.is_update or request.titles or request.export_paths:
        for label, titles in (
            ("added", report.added),
            ("changed", report.changed),
            ("removed", report.removed),
        ):
            lines.extend(f"{label}: {title}" for title in titles)
        lines.extend(f"redirect: {source} -> {target}" for source, target in report.redirects)
    return "\n".join(lines)


def execute_fetch(request: FetchRequest) -> int:
    failure_prefix = build_failure_prefix(FETCH_COMMAND, f"{BULBAPEDIA_SOURCE} {request.page_type}")
    try:
        report = run_fetch(request)
    except (BulbapediaAPIError, ExportError, FetchError, PageStoreError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_fetch_report(request, report))
    return EXIT_SUCCESS


def run_fetch_pokeapi(is_update: bool) -> SnapshotFetchReport:
    with build_http_client() as http_client:
        return fetch_snapshot(SnapshotStore(), http_client, is_update)


def execute_fetch_pokeapi(is_update: bool) -> int:
    failure_prefix = build_failure_prefix(FETCH_COMMAND, POKEAPI_SOURCE)
    try:
        report = run_fetch_pokeapi(is_update)
    except (SnapshotError, httpx.HTTPError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(
        "\n".join(
            [
                f"{FETCH_COMMAND} {POKEAPI_SOURCE}: done",
                f"commit: {report.commit}",
                f"files downloaded: {report.files_downloaded}",
                f"requests: {report.requests}",
            ]
        )
    )
    return EXIT_SUCCESS


def run_parse(request: ParseRequest) -> ParseRun:
    """Parse the pinned pages and write the outputs. No Neo4j setting is read."""
    inputs = RunInputs(
        request,
        PageStore(),
        OntologySettings().ontology_dir,
        DEFAULT_OUTPUT_ROOT / LOCATION_COVERAGE_FILE_NAME,
    )
    run = parse_species_pages(inputs)
    write_outputs(run, DEFAULT_OUTPUT_ROOT, is_full_run=not request.titles)
    return run


def format_parse_report(report: ParseReport, elapsed_seconds: float) -> str:
    """The counts of a run, the failed pages and the residue per kind."""
    lines = [
        f"{PARSE_COMMAND} {BULBAPEDIA_SOURCE} {PARSE_PAGE_TYPE}: done",
        f"pages parsed: {report.pages_parsed}",
        f"pages skipped: {report.pages_skipped}",
        f"pages failed: {report.pages_failed}",
    ]
    for label, counts in (
        ("species", report.species_counts),
        ("evolution", report.evolution_counts),
        ("chunk", report.chunk_counts),
    ):
        lines.append(f"{label} owner nodes: {sum(counts.owner_nodes.values())}")
        lines.append(f"{label} stubs: {sum(counts.stubs.values())}")
        lines.append(f"{label} edges: {sum(counts.edges.values())}")
    residue_counts = Counter(item.kind.value for item in report.residue)
    lines.extend(f"residue {kind}: {count}" for kind, count in sorted(residue_counts.items()))
    lines.extend(format_encounter_lines(report.encounters))
    lines.extend(
        f"failed: {page.title}: {page.reason}"
        for page in report.pages
        if page.status is PageStatus.FAILED
    )
    lines.append(f"elapsed seconds: {elapsed_seconds:.1f}")
    return "\n".join(lines)


def format_encounter_lines(report: EncounterReport) -> list[str]:
    return [
        f"encounter facts: {report.facts}",
        f"encounter owner nodes: {sum(report.encounter_counts.owner_nodes.values())}",
        f"encounter location stubs: {report.location_stubs}",
        *(f"encounter kind {kind}: {count}" for kind, count in report.per_kind.items()),
        f"encounter candidates: {report.candidate_occurrences}",
        f"encounter locations: {len(report.locations)}",
        f"encounter not_a_location: {len(report.not_a_location)}",
        f"encounter unknown_location: {len(report.unknown_location)} titles, "
        f"{report.unknown_location_occurrences} links",
        f"location candidate file: {DEFAULT_OUTPUT_ROOT / CANDIDATE_FILE_NAME}",
    ]


def execute_parse(request: ParseRequest) -> int:
    failure_prefix = build_failure_prefix(PARSE_COMMAND, f"{BULBAPEDIA_SOURCE} {PARSE_PAGE_TYPE}")
    started = time.monotonic()
    try:
        run = run_parse(request)
    except SPECIES_PARSE_ERRORS as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_parse_report(run.report, time.monotonic() - started))
    return EXIT_FAILURE if run.report.pages_failed else EXIT_SUCCESS


def run_move_parse(request: MoveParseRequest) -> MoveParseRun:
    """Parse the pinned move pages and write the outputs. No Neo4j or LLM setting is read."""
    inputs = MoveRunInputs(request, PageStore(), SnapshotStore(), OntologySettings().ontology_dir)
    run = parse_move_pages(inputs)
    write_move_outputs(run, DEFAULT_OUTPUT_ROOT, is_full_run=not request.titles)
    return run


def format_move_parse_report(report: MoveParseReport, elapsed_seconds: float) -> str:
    """The page counts, batch counts, notes per kind, cross-check per field and failures."""
    counts = report.batch_counts
    lines = [
        f"{PARSE_COMMAND} {BULBAPEDIA_SOURCE} {MOVE_PAGE_TYPE}: done",
        f"pages parsed: {report.pages_parsed}",
        f"pages skipped: {report.pages_skipped}",
        f"pages failed: {report.pages_failed}",
    ]
    lines.extend(
        f"owner nodes {name}: {count}" for name, count in sorted(counts.owner_nodes.items())
    )
    lines.append(f"stubs: {sum(counts.stubs.values())}")
    lines.append(f"edges: {sum(counts.edges.values())}")
    note_counts = Counter(note.kind.value for note in report.notes)
    lines.extend(f"notes {kind}: {count}" for kind, count in sorted(note_counts.items()))
    field_counts = Counter(difference.field for difference in report.cross_check)
    lines.extend(f"cross-check {field}: {count}" for field, count in sorted(field_counts.items()))
    lines.extend(
        f"failed: {page.title}: {page.reason}"
        for page in report.pages
        if page.status is PageStatus.FAILED
    )
    lines.append(f"elapsed seconds: {elapsed_seconds:.1f}")
    return "\n".join(lines)


def execute_move_parse(titles: tuple[str, ...]) -> int:
    failure_prefix = build_failure_prefix(PARSE_COMMAND, f"{BULBAPEDIA_SOURCE} {MOVE_PAGE_TYPE}")
    started = time.monotonic()
    try:
        run = run_move_parse(MoveParseRequest(titles=titles))
    except MOVE_PARSE_ERRORS as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_move_parse_report(run.report, time.monotonic() - started))
    return EXIT_FAILURE if run.report.pages_failed else EXIT_SUCCESS


def run_ability_parse(request: AbilityParseRequest) -> AbilityParseRun:
    """Parse the pinned ability pages and write the outputs. No Neo4j or LLM setting is read."""
    inputs = AbilityRunInputs(
        request, PageStore(), SnapshotStore(), OntologySettings().ontology_dir
    )
    run = parse_ability_pages(inputs)
    write_ability_outputs(run, DEFAULT_OUTPUT_ROOT, is_full_run=not request.titles)
    return run


def format_ability_parse_report(report: AbilityParseReport, elapsed_seconds: float) -> str:
    """The page counts, batch counts, notes per kind, cross-check results and failures."""
    counts = report.batch_counts
    lines = [
        f"{PARSE_COMMAND} {BULBAPEDIA_SOURCE} {ABILITY_PAGE_TYPE}: done",
        f"pages parsed: {report.pages_parsed}",
        f"pages skipped: {report.pages_skipped}",
        f"pages failed: {report.pages_failed}",
    ]
    lines.extend(
        f"owner nodes {name}: {count}" for name, count in sorted(counts.owner_nodes.items())
    )
    lines.append(f"chunks: {sum(report.chunks_per_aspect.values())}")
    lines.append(f"stubs: {sum(counts.stubs.values()) + sum(report.chunk_counts.stubs.values())}")
    lines.append(f"edges: {sum(counts.edges.values()) + sum(report.chunk_counts.edges.values())}")
    note_counts = Counter(note.kind.value for note in report.notes)
    lines.extend(f"notes {kind}: {count}" for kind, count in sorted(note_counts.items()))
    if report.species_output_missing:
        lines.append("cross-check: no species output, skipped")
    else:
        differences = Counter(difference.side.value for difference in report.holder_differences)
        lines.extend(f"cross-check {side}: {count}" for side, count in sorted(differences.items()))
    lines.append(f"unowned species stubs: {len(report.unowned_species_stubs)}")
    lines.extend(f"unowned: {stub_id}" for stub_id in report.unowned_species_stubs)
    lines.extend(
        f"failed: {page.title}: {page.reason}"
        for page in report.pages
        if page.status is PageStatus.FAILED
    )
    lines.append(f"elapsed seconds: {elapsed_seconds:.1f}")
    return "\n".join(lines)


def execute_ability_parse(titles: tuple[str, ...]) -> int:
    failure_prefix = build_failure_prefix(PARSE_COMMAND, f"{BULBAPEDIA_SOURCE} {ABILITY_PAGE_TYPE}")
    started = time.monotonic()
    try:
        run = run_ability_parse(AbilityParseRequest(titles=titles))
    except ABILITY_PARSE_ERRORS as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_ability_parse_report(run.report, time.monotonic() - started))
    return EXIT_FAILURE if run.report.pages_failed else EXIT_SUCCESS


def run_item_parse(request: ItemParseRequest) -> ItemParseRun:
    """Parse the pinned item pages and write the outputs. No Neo4j or LLM setting is read."""
    inputs = ItemRunInputs(request, PageStore(), OntologySettings().ontology_dir)
    run = parse_item_pages(inputs)
    write_item_outputs(run, DEFAULT_OUTPUT_ROOT, is_full_run=not request.titles)
    return run


def format_item_parse_report(report: ItemParseReport, elapsed_seconds: float) -> str:
    """The page counts, skip reasons, classes, batch counts, notes and unowned stubs."""
    counts = report.batch_counts
    lines = [
        f"{PARSE_COMMAND} {BULBAPEDIA_SOURCE} {ITEM_PAGE_TYPE}: done",
        f"pages parsed: {report.pages_parsed}",
        f"pages skipped: {report.pages_skipped}",
        f"pages failed: {report.pages_failed}",
    ]
    lines.extend(
        f"skipped {reason}: {count}" for reason, count in report.skipped_per_reason.items()
    )
    lines.extend(
        f"owner nodes {name}: {count}" for name, count in sorted(counts.owner_nodes.items())
    )
    lines.append(f"chunks: {sum(report.chunks_per_aspect.values())}")
    lines.append(f"stubs: {sum(counts.stubs.values()) + sum(report.chunk_counts.stubs.values())}")
    lines.append(f"edges: {sum(counts.edges.values()) + sum(report.chunk_counts.edges.values())}")
    note_counts = Counter(note.kind.value for note in report.notes)
    lines.extend(f"notes {kind}: {count}" for kind, count in sorted(note_counts.items()))
    lines.append(f"unknown game codes: {len(report.unknown_codes)}")
    lines.extend(f"stub source missing: {page_type}" for page_type in report.missing_stub_sources)
    lines.append(f"unowned item stubs: {len(report.unowned_stub_ids)}")
    lines.extend(f"unowned: {stub_id}" for stub_id in report.unowned_stub_ids)
    lines.extend(
        f"failed: {page.title}: {page.reason}"
        for page in report.pages
        if page.status is PageStatus.FAILED
    )
    lines.append(f"elapsed seconds: {elapsed_seconds:.1f}")
    return "\n".join(lines)


def execute_item_parse(titles: tuple[str, ...]) -> int:
    failure_prefix = build_failure_prefix(PARSE_COMMAND, f"{BULBAPEDIA_SOURCE} {ITEM_PAGE_TYPE}")
    started = time.monotonic()
    try:
        run = run_item_parse(ItemParseRequest(titles=titles))
    except ITEM_PARSE_ERRORS as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_item_parse_report(run.report, time.monotonic() - started))
    return EXIT_FAILURE if run.report.pages_failed else EXIT_SUCCESS


def run_location_parse(request: LocationParseRequest) -> LocationParseRun:
    """Parse the pinned location pages and write the outputs. No Neo4j or LLM setting is read."""
    inputs = LocationRunInputs(request, PageStore(), OntologySettings().ontology_dir)
    run = parse_location_pages(inputs)
    write_location_outputs(run, DEFAULT_OUTPUT_ROOT, is_full_run=not request.titles)
    return run


def format_location_parse_report(report: LocationParseReport, elapsed_seconds: float) -> str:
    """The page counts, skip reasons, classes, batch counts, notes and unowned references."""
    counts = report.batch_counts
    lines = [
        f"{PARSE_COMMAND} {BULBAPEDIA_SOURCE} {LOCATION_PAGE_TYPE}: done",
        f"pages parsed: {report.pages_parsed}",
        f"pages skipped: {report.pages_skipped}",
        f"pages failed: {report.pages_failed}",
    ]
    lines.extend(
        f"skipped {reason}: {count}" for reason, count in report.skipped_per_reason.items()
    )
    lines.extend(f"locations {name}: {count}" for name, count in report.locations_per_class.items())
    lines.append(f"areas: {report.areas}")
    lines.append(f"chunks: {sum(report.chunks_per_aspect.values())}")
    lines.extend(f"chunks {name}: {count}" for name, count in report.chunks_per_aspect.items())
    lines.append(f"mentions: {report.mentions} ({report.unresolved_mentions} unresolved)")
    lines.append(f"long chunks: {len(report.long_chunks)}")
    lines.append(f"item finds: {sum(report.finds_per_method.values())}")
    lines.extend(f"finds {name}: {count}" for name, count in report.finds_per_method.items())
    lines.append(f"hidden finds: {report.hidden_finds}")
    lines.extend(f"find skips {name}: {count}" for name, count in report.find_skips.items())
    lines.append(f"duplicate finds: {report.duplicate_finds}")
    lines.append(f"find item stubs: {len(report.item_stub_names)}")
    lines.append(f"encounters: {sum(report.encounters_per_kind.values())}")
    lines.extend(
        f"encounters {name}: {count}" for name, count in report.encounters_per_kind.items()
    )
    lines.extend(
        f"encounter skips {name}: {count}" for name, count in report.encounter_skips.items()
    )
    lines.append(f"merged encounters: {report.merged_encounters}")
    lines.append(f"duplicate encounters: {report.duplicate_encounters}")
    lines.append(f"unresolved species: {len(report.unresolved_species)}")
    lines.append(f"coverage pairs: {report.coverage_pairs}")
    if report.is_item_source_missing:
        lines.append("item source missing: item")
    if report.is_pokeapi_source_missing:
        lines.append("pokeapi source missing: locations.csv")
    lines.append(f"pokeapi locations with no owner: {len(report.pokeapi_unmatched)}")
    lines.extend(
        f"owner nodes {name}: {count}" for name, count in sorted(counts.owner_nodes.items())
    )
    others = (report.chunk_counts, report.find_counts, report.encounter_counts)
    lines.append(
        f"stubs: {sum(counts.stubs.values()) + sum(sum(c.stubs.values()) for c in others)}"
    )
    lines.append(
        f"edges: {sum(counts.edges.values()) + sum(sum(c.edges.values()) for c in others)}"
    )
    note_counts = Counter(note.kind.value for note in report.notes)
    lines.extend(f"notes {kind}: {count}" for kind, count in sorted(note_counts.items()))
    lines.append(f"duplicate ids: {len(report.duplicate_ids)}")
    lines.extend(f"duplicate: {entry}" for entry in report.duplicate_ids)
    lines.append(f"candidates: {report.candidate_count}")
    lines.append(f"unowned location stubs: {len(report.unowned_stub_titles)}")
    lines.extend(
        f"failed: {page.title}: {page.reason}"
        for page in report.pages
        if page.status is PageStatus.FAILED
    )
    lines.append(f"elapsed seconds: {elapsed_seconds:.1f}")
    return "\n".join(lines)


def execute_location_parse(titles: tuple[str, ...]) -> int:
    failure_prefix = build_failure_prefix(
        PARSE_COMMAND, f"{BULBAPEDIA_SOURCE} {LOCATION_PAGE_TYPE}"
    )
    started = time.monotonic()
    try:
        run = run_location_parse(LocationParseRequest(titles=titles))
    except LOCATION_PARSE_ERRORS as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_location_parse_report(run.report, time.monotonic() - started))
    return EXIT_FAILURE if run.report.pages_failed else EXIT_SUCCESS


def format_progress(progress: PassProgress) -> str:
    return (
        f"{progress.ingest_pass.value}: {progress.done}/{progress.total} batches, "
        f"{progress.elapsed_seconds:.1f} s"
    )


def print_progress(progress: PassProgress) -> None:
    print(format_progress(progress), flush=True)


async def run_ingest(settings: Settings, run: ParseRun, titles: tuple[str, ...]) -> IngestRun:
    request = IngestRequest(
        ontology_dir=settings.ontology.ontology_dir,
        universes_dir=settings.universes.universes_dir,
        titles=titles,
        on_progress=print_progress,
    )
    async with Neo4jGraphClient(settings.neo4j) as client:
        return await ingest_species(client, run, request)


def is_clean_ingest(pages_failed: int, report: IngestReport) -> bool:
    """A run is clean when no page failed to parse, no batch was rejected and nothing stopped it."""
    return not pages_failed and not report.rejected_batches and report.stop is None


def format_rejection(rejection: BatchRejection) -> list[str]:
    """One line per violation of a rejected batch."""
    lines = []
    for violation in rejection.violations:
        subject = ""
        if violation.node is not None:
            class_name = extract_local_name(violation.node.class_iri)
            subject = f" {class_name} {violation.node.id}"
        property_name = f" {violation.property_name}" if violation.property_name else ""
        lines.append(
            f"rejected: {rejection.page_title} ({rejection.ingest_pass}): "
            f"{violation.rule_id}{subject}{property_name}: {violation.message}"
        )
    return lines


def format_ingest_report(
    page_type: str, report: IngestReport, elapsed_seconds: Mapping[str, float]
) -> str:
    """The counters and seconds of each pass, the failed pages, the stop, the census and,
    for a report with stub ids, the stubs."""
    status = "stopped" if report.stop is not None else "done"
    lines = [f"{INGEST_COMMAND} {BULBAPEDIA_SOURCE} {page_type}: {status}"]
    for totals in report.passes:
        name = totals.ingest_pass
        seconds = elapsed_seconds[totals.ingest_pass]
        lines.append(f"{name}: {totals.batches_written} batches in {seconds:.1f} s")
        if totals.batches_rejected:
            lines.append(f"{name}: {totals.batches_rejected} batches rejected")
        lines.extend(
            f"{name} {counter.replace('_', ' ')}: {count}"
            for counter, count in totals.counters.items()
        )
    lines.extend(f"failed: {page.page_title}: {page.reason}" for page in report.failed_pages)
    for rejection in report.rejected_batches:
        lines.extend(format_rejection(rejection))
    if report.stop is not None:
        stop = report.stop
        lines.append(f"stopped: {stop.page_title} ({stop.ingest_pass}): {stop.error}")
    if report.census is not None:
        lines.extend(f"nodes {label}: {count}" for label, count in report.census.labels.items())
        lines.extend(
            f"relationships {name}: {count}"
            for name, count in report.census.relationship_types.items()
        )
    stub_ids = report.stub_ids if isinstance(report, StubIngestReport) else None
    for label, ids in (stub_ids or {}).items():
        lines.append(f"stubs {label}: {len(ids)}")
        lines.extend(f"stub {label}: {stub_id}" for stub_id in ids)
    return "\n".join(lines)


def execute_ingest(settings: Settings, request: ParseRequest) -> int:
    failure_prefix = build_failure_prefix(INGEST_COMMAND, f"{BULBAPEDIA_SOURCE} {PARSE_PAGE_TYPE}")
    started = time.monotonic()
    try:
        run = run_parse(request)
    except SPECIES_PARSE_ERRORS as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE
    parse_seconds = time.monotonic() - started

    try:
        ingest_run = asyncio.run(run_ingest(settings, run, request.titles))
    except (MissingSeedError, GraphError, OntologyError, ABoxError, UniverseError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    write_ingest_report(ingest_run.report)
    print(format_parse_report(run.report, parse_seconds))
    print(format_ingest_report(PARSE_PAGE_TYPE, ingest_run.report, ingest_run.elapsed_seconds))
    is_clean = is_clean_ingest(run.report.pages_failed, ingest_run.report)
    return EXIT_SUCCESS if is_clean else EXIT_FAILURE


async def run_move_ingest(
    settings: Settings, run: MoveParseRun, titles: tuple[str, ...]
) -> MoveIngestRun:
    request = IngestRequest(
        ontology_dir=settings.ontology.ontology_dir,
        universes_dir=settings.universes.universes_dir,
        titles=titles,
        on_progress=print_progress,
    )
    async with Neo4jGraphClient(settings.neo4j) as client:
        return await ingest_moves(client, run, request)


def execute_move_ingest(settings: Settings, titles: tuple[str, ...]) -> int:
    failure_prefix = build_failure_prefix(INGEST_COMMAND, f"{BULBAPEDIA_SOURCE} {MOVE_PAGE_TYPE}")
    started = time.monotonic()
    try:
        run = run_move_parse(MoveParseRequest(titles=titles))
    except MOVE_PARSE_ERRORS as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE
    parse_seconds = time.monotonic() - started

    try:
        ingest_run = asyncio.run(run_move_ingest(settings, run, titles))
    except (MissingSeedError, GraphError, OntologyError, ABoxError, UniverseError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    write_move_ingest_report(ingest_run.report)
    print(format_move_parse_report(run.report, parse_seconds))
    print(format_ingest_report(MOVE_PAGE_TYPE, ingest_run.report, ingest_run.elapsed_seconds))
    is_clean = is_clean_ingest(run.report.pages_failed, ingest_run.report)
    return EXIT_SUCCESS if is_clean else EXIT_FAILURE


async def run_ability_ingest(
    settings: Settings, run: AbilityParseRun, titles: tuple[str, ...]
) -> AbilityIngestRun:
    request = IngestRequest(
        ontology_dir=settings.ontology.ontology_dir,
        universes_dir=settings.universes.universes_dir,
        titles=titles,
        on_progress=print_progress,
    )
    async with Neo4jGraphClient(settings.neo4j) as client:
        return await ingest_abilities(client, run, request)


def execute_ability_ingest(settings: Settings, titles: tuple[str, ...]) -> int:
    failure_prefix = build_failure_prefix(
        INGEST_COMMAND, f"{BULBAPEDIA_SOURCE} {ABILITY_PAGE_TYPE}"
    )
    started = time.monotonic()
    try:
        run = run_ability_parse(AbilityParseRequest(titles=titles))
    except ABILITY_PARSE_ERRORS as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE
    parse_seconds = time.monotonic() - started

    try:
        ingest_run = asyncio.run(run_ability_ingest(settings, run, titles))
    except (MissingSeedError, GraphError, OntologyError, ABoxError, UniverseError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    write_ability_ingest_report(ingest_run.report)
    print(format_ability_parse_report(run.report, parse_seconds))
    print(format_ingest_report(ABILITY_PAGE_TYPE, ingest_run.report, ingest_run.elapsed_seconds))
    is_clean = is_clean_ingest(run.report.pages_failed, ingest_run.report)
    return EXIT_SUCCESS if is_clean else EXIT_FAILURE


async def run_item_ingest(
    settings: Settings, run: ItemParseRun, titles: tuple[str, ...]
) -> ItemIngestRun:
    request = IngestRequest(
        ontology_dir=settings.ontology.ontology_dir,
        universes_dir=settings.universes.universes_dir,
        titles=titles,
        on_progress=print_progress,
    )
    async with Neo4jGraphClient(settings.neo4j) as client:
        return await ingest_items(client, run, request)


def execute_item_ingest(settings: Settings, titles: tuple[str, ...]) -> int:
    failure_prefix = build_failure_prefix(INGEST_COMMAND, f"{BULBAPEDIA_SOURCE} {ITEM_PAGE_TYPE}")
    started = time.monotonic()
    try:
        run = run_item_parse(ItemParseRequest(titles=titles))
    except ITEM_PARSE_ERRORS as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE
    parse_seconds = time.monotonic() - started

    try:
        ingest_run = asyncio.run(run_item_ingest(settings, run, titles))
    except (MissingSeedError, GraphError, OntologyError, ABoxError, UniverseError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    write_item_ingest_report(ingest_run.report)
    print(format_item_parse_report(run.report, parse_seconds))
    print(format_ingest_report(ITEM_PAGE_TYPE, ingest_run.report, ingest_run.elapsed_seconds))
    is_clean = is_clean_ingest(run.report.pages_failed, ingest_run.report)
    return EXIT_SUCCESS if is_clean else EXIT_FAILURE


async def run_location_ingest(
    settings: Settings, run: LocationParseRun, titles: tuple[str, ...]
) -> LocationIngestRun:
    request = IngestRequest(
        ontology_dir=settings.ontology.ontology_dir,
        universes_dir=settings.universes.universes_dir,
        titles=titles,
        on_progress=print_progress,
    )
    async with Neo4jGraphClient(settings.neo4j) as client:
        return await ingest_locations(client, run, request)


def execute_location_ingest(settings: Settings, titles: tuple[str, ...]) -> int:
    failure_prefix = build_failure_prefix(
        INGEST_COMMAND, f"{BULBAPEDIA_SOURCE} {LOCATION_PAGE_TYPE}"
    )
    started = time.monotonic()
    try:
        run = run_location_parse(LocationParseRequest(titles=titles))
    except LOCATION_PARSE_ERRORS as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE
    parse_seconds = time.monotonic() - started

    try:
        ingest_run = asyncio.run(run_location_ingest(settings, run, titles))
    except (MissingSeedError, GraphError, OntologyError, ABoxError, UniverseError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    write_location_ingest_report(ingest_run.report)
    print(format_location_parse_report(run.report, parse_seconds))
    print(format_ingest_report(LOCATION_PAGE_TYPE, ingest_run.report, ingest_run.elapsed_seconds))
    is_clean = is_clean_ingest(run.report.pages_failed, ingest_run.report)
    return EXIT_SUCCESS if is_clean else EXIT_FAILURE


def execute_seed(settings: Settings, prefix: str) -> int:
    try:
        report = asyncio.run(run_seed(settings, prefix))
    except (OntologyError, ABoxError, GraphError, UniverseError) as error:
        print(f"{build_failure_prefix(SEED_COMMAND, prefix)} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_report(prefix, report))
    return EXIT_SUCCESS


async def run_validate(settings: Settings, prefix: str) -> SweepReport:
    ontology_dir = settings.ontology.ontology_dir
    model = load_ontology_model(ontology_dir, prefix)
    abox = read_abox(model, ontology_dir)
    materializer = build_plugin_materializer(model, load_plugin(settings, prefix).manifest)
    async with Neo4jGraphClient(settings.neo4j) as client:
        return await run_sweep(client, materializer, model, abox)


def execute_validate(settings: Settings, prefix: str) -> int:
    try:
        report = asyncio.run(run_validate(settings, prefix))
    except (OntologyError, ABoxError, GraphError, UniverseError) as error:
        print(f"{build_failure_prefix(VALIDATE_COMMAND, prefix)} {error}", file=sys.stderr)
        return EXIT_FAILURE

    report_path = write_sweep_report(report, prefix)
    status = "failing" if report.is_failing else "clean"
    title = f"{VALIDATE_COMMAND} {prefix}: {status}"
    STDOUT.print(format_validator_report(title, report, report_path))
    return EXIT_FAILURE if report.is_failing else EXIT_SUCCESS


def execute_purge(settings: Settings, prefix: str) -> int:
    failure_prefix = build_failure_prefix(PURGE_COMMAND, prefix)
    try:
        typed_password = prompt_password()
    except (EOFError, KeyboardInterrupt):
        print(f"{PURGE_COMMAND} {prefix} {ABORT_MESSAGE}", file=sys.stderr)
        return EXIT_FAILURE

    if not is_password_match(typed_password, settings.neo4j.password):
        print(f"{failure_prefix} {PASSWORD_MISMATCH_MESSAGE}", file=sys.stderr)
        return EXIT_FAILURE

    try:
        summary = asyncio.run(run_purge(settings, prefix))
    except (OntologyError, GraphError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_purge_report(prefix, summary))
    return EXIT_SUCCESS


SCHEMA_REBUILD_REASONS = {
    CacheStatus.MISSING: "missing",
    CacheStatus.STALE: "stale",
    CacheStatus.REFRESHED: "refresh",
}


def format_schema_status(prefix: str, context: SchemaContext, elapsed_seconds: float) -> str:
    """Name how the schema was obtained and how long it took, for stderr."""
    if context.cache_status is CacheStatus.HIT:
        outcome = "cache used"
    elif context.cache_status is CacheStatus.INVALID:
        outcome = f"rebuilt (invalid: {context.cache_path})"
    else:
        outcome = f"rebuilt ({SCHEMA_REBUILD_REASONS[context.cache_status]})"
    return f"{SCHEMA_COMMAND} {prefix}: {outcome} in {elapsed_seconds:.2f} s"


async def run_schema(
    settings: Settings, plugin: UniversePlugin, should_refresh: bool
) -> SchemaContext:
    model = load_ontology_model(settings.ontology.ontology_dir, plugin.prefix)
    request = SchemaRequest(plugin, LabelMaterializer(model), should_refresh)
    async with Neo4jGraphClient(settings.neo4j) as client:
        return await load_schema_context(client, request, settings.universes.schema_cache_dir)


def execute_schema(settings: Settings, arguments: argparse.Namespace) -> int:
    prefix = arguments.prefix
    failure_prefix = build_failure_prefix(SCHEMA_COMMAND, prefix)
    try:
        registry = discover_universes(
            settings.universes.universes_dir, settings.ontology.ontology_dir
        )
        plugin = registry.get(prefix)
    except UniverseError as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    started_at = time.perf_counter()
    try:
        context = asyncio.run(run_schema(settings, plugin, arguments.refresh))
    except (OntologyError, GraphError, SchemaIntrospectionError, SchemaCacheError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    elapsed_seconds = time.perf_counter() - started_at
    if arguments.is_json:
        print(context.graph_schema.model_dump_json(indent=SCHEMA_JSON_INDENT))
    else:
        print(context.text, end="")
    print(format_schema_status(prefix, context, elapsed_seconds), file=sys.stderr)
    return EXIT_SUCCESS


def render_node_name(key: NodeKey, evidence: Evidence) -> str:
    node = evidence.graph_slice.find_node(key)
    return node.name if node is not None and node.name else key.id


def render_table_cell(cell: Cell, evidence: Evidence) -> str:
    """Show a cell for a reader: nodes by name, relationships by type, paths by node names."""
    if isinstance(cell, NodeCell):
        return render_node_name(cell.key, evidence)
    if isinstance(cell, RelationshipCell):
        return cell.type
    if isinstance(cell, PathCell):
        return TABLE_PATH_SEPARATOR.join(render_node_name(key, evidence) for key in cell.nodes)
    if isinstance(cell, list):
        return TABLE_LIST_SEPARATOR.join(render_table_cell(item, evidence) for item in cell)
    if isinstance(cell, dict):
        return TABLE_LIST_SEPARATOR.join(
            f"{name}: {render_table_cell(item, evidence)}" for name, item in cell.items()
        )
    return "" if cell is None else str(cell)


def render_table(evidence: Evidence) -> str:
    """Print every row of the evidence item, columns padded to their widest cell."""
    rows = [
        [render_table_cell(row.get(column), evidence) for column in evidence.columns]
        for row in evidence.rows
    ]
    table = [list(evidence.columns), *rows]
    widths = [max(len(line[index]) for line in table) for index in range(len(evidence.columns))]
    lines = [
        TABLE_COLUMN_SEPARATOR.join(
            cell.ljust(width) for cell, width in zip(line, widths, strict=True)
        ).rstrip()
        for line in table
    ]
    return "\n".join([evidence.purpose, *lines])


def render_answer_text(result: AskResult) -> str:
    """Return the answer text, which holds its source links and its attribution line."""
    return result.answer.text


def render_answer_details(result: AskResult) -> str:
    """Return the cited tables in full, then the warnings; empty when there is neither."""
    answer, bundle = result.answer, result.bundle
    sections: list[str] = []
    for evidence_id in answer.cited.table_ids:
        evidence = bundle.get(evidence_id)
        if evidence is not None and evidence.rows:
            sections.append(render_table(evidence))
    if answer.warnings:
        sections.append("\n".join(["Warnings:", *(f"- {item}" for item in answer.warnings)]))
    return "\n\n".join(sections)


def render_queries(bundle: EvidenceBundle) -> str:
    """List every query in call order with its purpose, status and row count."""
    lines = ["Queries"]
    for number, item in enumerate(bundle.items, start=1):
        lines.append(f"{number}. {item.purpose} [{item.status}, {item.row_count} rows]")
        lines.append(f"   {item.query}")
    return "\n".join(lines)


def format_ask_status(result: AskResult) -> str:
    stats = result.stats
    model_calls = sum(step.model_calls for step in stats.steps)
    input_tokens = sum(step.input_tokens for step in stats.steps)
    cached_tokens = sum(step.cached_input_tokens for step in stats.steps)
    output_tokens = sum(step.output_tokens for step in stats.steps)
    reasoning_tokens = sum(step.reasoning_tokens for step in stats.steps)
    return (
        f"{ASK_COMMAND} {result.universe}: {stats.cypher_calls} queries, "
        f"{model_calls} model calls, {input_tokens:,} input ({cached_tokens:,} cached) and "
        f"{output_tokens:,} output ({reasoning_tokens:,} reasoning) tokens, "
        f"{stats.total_latency_seconds:.1f} s"
    )


def describe_missing_agent_settings(error: ValidationError) -> str:
    names = sorted(
        {f"{AGENT_VARIABLE_PREFIX}{str(detail['loc'][0]).upper()}" for detail in error.errors()}
    )
    return f"missing or invalid settings: {', '.join(names)}"


@dataclass(frozen=True)
class AgentContext:
    """What an agent command needs to run: the settings, the plugins and the step display."""

    settings: Settings
    agent_settings: AgentSettings
    registry: UniverseRegistry
    progress: StepTracker


async def run_ask(context: AgentContext, request: AskRequest) -> AskResult:
    async with Neo4jGraphClient(context.settings.neo4j) as client:
        runtime = create_agent_runtime(client, context.agent_settings)
        events = ask(request, context.registry, context.settings, runtime)
        return await collect_answer(events, context.progress)


def execute_ask(settings: Settings, arguments: argparse.Namespace) -> int:
    failure_prefix = build_failure_prefix(ASK_COMMAND, describe_target(arguments))
    try:
        agent_settings = AgentSettings()
    except ValidationError as error:
        STDERR.print(f"{failure_prefix} {describe_missing_agent_settings(error)}")
        return EXIT_FAILURE
    try:
        registry = discover_universes(
            settings.universes.universes_dir, settings.ontology.ontology_dir
        )
        plugin = select_universe(registry, arguments.universe)
        request = AskRequest(question=arguments.question, universe=plugin.prefix)
        with StepDisplay(STDERR) as display:
            context = AgentContext(settings, agent_settings, registry, display)
            result = asyncio.run(run_ask(context, request))
    except (
        UniverseError,
        RoutingError,
        OntologyError,
        GraphError,
        SchemaIntrospectionError,
        SchemaCacheError,
        ModelCallError,
        NameIndexError,
    ) as error:
        STDERR.print(f"{failure_prefix} {error}")
        return EXIT_FAILURE

    if arguments.is_json:
        STDOUT.print(result.model_dump_json(indent=ASK_JSON_INDENT, by_alias=True))
        return EXIT_SUCCESS
    STDOUT.print(build_answer_renderable(render_answer_text(result), STDOUT.is_terminal))
    details = render_answer_details(result)
    if details:
        STDOUT.print()
        STDOUT.print(details)
    if arguments.should_show_cypher:
        STDOUT.print()
        STDOUT.print(render_queries(result.bundle))
    STDERR.print(format_ask_status(result), style="dim")
    return EXIT_SUCCESS


async def run_eval_session(context: AgentContext, request: EvalRequest) -> EvalRun:
    async with Neo4jGraphClient(context.settings.neo4j) as client:
        runtime = create_agent_runtime(client, context.agent_settings)
        tracked = replace(request, progress=context.progress)
        return await run_eval(tracked, context.registry, runtime)


def execute_eval(settings: Settings, arguments: argparse.Namespace) -> int:
    failure_prefix = build_failure_prefix(EVAL_COMMAND, arguments.prefix)
    try:
        agent_settings = AgentSettings()
    except ValidationError as error:
        STDERR.print(f"{failure_prefix} {describe_missing_agent_settings(error)}")
        return EXIT_FAILURE
    try:
        registry = discover_universes(
            settings.universes.universes_dir, settings.ontology.ontology_dir
        )
        plugin = select_universe(registry, arguments.prefix)
        eval_sets = load_eval_sets(plugin, arguments.set_names)
        request = EvalRequest(
            prefix=plugin.prefix,
            questions=select_questions(eval_sets, arguments.question_ids),
            set_hashes={eval_set.name: eval_set.source_hash for eval_set in eval_sets},
            settings=settings,
            started_at=datetime.now(UTC),
        )
        set_names = ", ".join(eval_set.name for eval_set in eval_sets)
        STDOUT.print(
            f"{EVAL_COMMAND} {plugin.prefix} ({set_names}): {len(request.questions)} questions"
        )
        with EvalDisplay(STDOUT, request.questions) as display:
            context = AgentContext(settings, agent_settings, registry, display)
            tracked = replace(request, on_result=display.handle_result)
            run = asyncio.run(run_eval_session(context, tracked))
    except (
        UniverseError,
        RoutingError,
        OntologyError,
        GraphError,
        SchemaIntrospectionError,
        SchemaCacheError,
        EvalError,
        NameIndexError,
    ) as error:
        STDERR.print(f"{failure_prefix} {error}")
        return EXIT_FAILURE

    folder = write_run(run, arguments.output_dir)
    STDOUT.print(build_eval_summary(run))
    STDOUT.print(f"run folder: {folder}")
    has_error = any(record.error is not None for record in run.records)
    return EXIT_FAILURE if has_error or run.stop is not None else EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command in `argv` and return the process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == FETCH_COMMAND and arguments.source == POKEAPI_SOURCE:
        if arguments.page_type is not None or arguments.titles or arguments.export_paths:
            parser.error(POKEAPI_OPTIONS_MESSAGE)
        return execute_fetch_pokeapi(arguments.update)
    if arguments.command == FETCH_COMMAND:
        if arguments.page_type is None:
            parser.error(MISSING_PAGE_TYPE_MESSAGE)
        if arguments.export_paths and (arguments.update or arguments.titles):
            parser.error(EXPORT_CONFLICT_MESSAGE)
        request = FetchRequest(
            page_type=arguments.page_type,
            titles=tuple(arguments.titles),
            is_update=arguments.update,
            export_paths=tuple(arguments.export_paths),
        )
        return execute_fetch(request)
    if arguments.command == PARSE_COMMAND and arguments.page_type == MOVE_PAGE_TYPE:
        return execute_move_parse(tuple(arguments.titles))
    if arguments.command == PARSE_COMMAND and arguments.page_type == ABILITY_PAGE_TYPE:
        return execute_ability_parse(tuple(arguments.titles))
    if arguments.command == PARSE_COMMAND and arguments.page_type == ITEM_PAGE_TYPE:
        return execute_item_parse(tuple(arguments.titles))
    if arguments.command == PARSE_COMMAND and arguments.page_type == LOCATION_PAGE_TYPE:
        return execute_location_parse(tuple(arguments.titles))
    if arguments.command == PARSE_COMMAND:
        return execute_parse(ParseRequest(titles=tuple(arguments.titles)))

    try:
        settings = get_settings()
    except ValidationError as error:
        failure_prefix = build_failure_prefix(arguments.command, describe_target(arguments))
        print(f"{failure_prefix} invalid settings: {error}", file=sys.stderr)
        return EXIT_FAILURE

    if arguments.command == INGEST_COMMAND and arguments.page_type == MOVE_PAGE_TYPE:
        return execute_move_ingest(settings, tuple(arguments.titles))
    if arguments.command == INGEST_COMMAND and arguments.page_type == ABILITY_PAGE_TYPE:
        return execute_ability_ingest(settings, tuple(arguments.titles))
    if arguments.command == INGEST_COMMAND and arguments.page_type == ITEM_PAGE_TYPE:
        return execute_item_ingest(settings, tuple(arguments.titles))
    if arguments.command == INGEST_COMMAND and arguments.page_type == LOCATION_PAGE_TYPE:
        return execute_location_ingest(settings, tuple(arguments.titles))
    if arguments.command == INGEST_COMMAND:
        return execute_ingest(settings, ParseRequest(titles=tuple(arguments.titles)))
    if arguments.command == PURGE_COMMAND:
        return execute_purge(settings, arguments.prefix)
    if arguments.command == VALIDATE_COMMAND:
        return execute_validate(settings, arguments.prefix)
    if arguments.command == SCHEMA_COMMAND:
        return execute_schema(settings, arguments)
    if arguments.command == ASK_COMMAND:
        return execute_ask(settings, arguments)
    if arguments.command == EVAL_COMMAND:
        return execute_eval(settings, arguments)
    return execute_seed(settings, arguments.prefix)
