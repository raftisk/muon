"""Command line entry point: `muon seed <prefix>`, `muon purge <prefix>`,
`muon fetch bulbapedia <page-type>` and `muon parse bulbapedia species`."""

import argparse
import asyncio
import getpass
import secrets
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import fields

from pydantic import SecretStr, ValidationError

from muon.config import OntologySettings, Settings, get_settings
from muon.graph import GraphError, Neo4jGraphClient, WriteSummary
from muon.ingestion.bulbapedia.api import BulbapediaAPIError, BulbapediaClient, build_http_client
from muon.ingestion.bulbapedia.fetch import (
    PAGE_TYPES,
    FetchError,
    FetchReport,
    FetchRequest,
    fetch_pages,
)
from muon.ingestion.bulbapedia.pages import PageStore, PageStoreError
from muon.ingestion.bulbapedia.species.forms import OverrideError
from muon.ingestion.bulbapedia.species.run import (
    DEFAULT_OUTPUT_ROOT,
    PageStatus,
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
from muon.ingestion.runner import Runner, RunReport
from muon.ingestion.seed import ABoxError, seed_universe
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyError, load_ontology_model

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
PROGRAM_NAME = "muon"
SEED_COMMAND = "seed"
PURGE_COMMAND = "purge"
FETCH_COMMAND = "fetch"
PARSE_COMMAND = "parse"
BULBAPEDIA_SOURCE = "bulbapedia"
PASSWORD_PROMPT = "Neo4j password: "
PASSWORD_ENCODING = "utf-8"
PASSWORD_MISMATCH_MESSAGE = "the password does not match NEO4J_PASSWORD"
ABORT_MESSAGE = "aborted"


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
    fetch = commands.add_parser(
        FETCH_COMMAND, help="download pinned source pages into data/<source>/ for ingestion"
    )
    fetch.add_argument("source", choices=(BULBAPEDIA_SOURCE,), help="the wiki to fetch from")
    fetch.add_argument("page_type", choices=sorted(PAGE_TYPES), help="the set of pages to fetch")
    fetch.add_argument(
        "--update", action="store_true", help="move the pins to the latest revisions"
    )
    fetch.add_argument(
        "--title",
        action="append",
        default=[],
        dest="titles",
        metavar="TITLE",
        help="fetch only this page (repeatable)",
    )
    parse = commands.add_parser(
        PARSE_COMMAND, help="parse pinned source pages into batches and a report under data/"
    )
    parse.add_argument("source", choices=(BULBAPEDIA_SOURCE,), help="the wiki to parse")
    parse.add_argument("page_type", choices=(PARSE_PAGE_TYPE,), help="the set of pages to parse")
    parse.add_argument(
        "--title",
        action="append",
        default=[],
        dest="titles",
        metavar="TITLE",
        help="write outputs for only this page (repeatable)",
    )
    return parser


def build_failure_prefix(command: str, prefix: str) -> str:
    return f"{command} {prefix} failed:"


def format_report(universe: str, report: RunReport) -> str:
    lines = [f"{SEED_COMMAND} {universe}: done"]
    lines.extend(
        f"{field.name.replace('_', ' ')}: {getattr(report, field.name)}"
        for field in fields(RunReport)
    )
    return "\n".join(lines)


async def run_seed(settings: Settings, prefix: str) -> RunReport:
    async with Neo4jGraphClient(settings.neo4j) as client:
        return await seed_universe(client, settings.ontology.ontology_dir, prefix)


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
    with build_http_client() as http_client:
        return fetch_pages(request, PageStore(), BulbapediaClient(http_client))


def format_fetch_report(request: FetchRequest, report: FetchReport) -> str:
    """The counts of a run, and for an update or title run the titles behind them."""
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
        f"elapsed seconds: {report.elapsed_seconds:.1f}",
    ]
    if request.is_update or request.titles:
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
    except (BulbapediaAPIError, FetchError, PageStoreError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_fetch_report(request, report))
    return EXIT_SUCCESS


def run_parse(request: ParseRequest) -> ParseRun:
    """Parse the pinned pages and write the outputs. No Neo4j setting is read."""
    inputs = RunInputs(request, PageStore(), OntologySettings().ontology_dir)
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
    ):
        lines.append(f"{label} owner nodes: {sum(counts.owner_nodes.values())}")
        lines.append(f"{label} stubs: {sum(counts.stubs.values())}")
        lines.append(f"{label} edges: {sum(counts.edges.values())}")
    residue_counts = Counter(item.kind.value for item in report.residue)
    lines.extend(f"residue {kind}: {count}" for kind, count in sorted(residue_counts.items()))
    lines.extend(
        f"failed: {page.title}: {page.reason}"
        for page in report.pages
        if page.status is PageStatus.FAILED
    )
    lines.append(f"elapsed seconds: {elapsed_seconds:.1f}")
    return "\n".join(lines)


def execute_parse(request: ParseRequest) -> int:
    failure_prefix = build_failure_prefix(PARSE_COMMAND, f"{BULBAPEDIA_SOURCE} {PARSE_PAGE_TYPE}")
    started = time.monotonic()
    try:
        run = run_parse(request)
    except (ParseRunError, OverrideError, OntologyError, ABoxError, PageStoreError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_parse_report(run.report, time.monotonic() - started))
    return EXIT_FAILURE if run.report.pages_failed else EXIT_SUCCESS


def execute_seed(settings: Settings, prefix: str) -> int:
    try:
        report = asyncio.run(run_seed(settings, prefix))
    except (OntologyError, ABoxError, GraphError) as error:
        print(f"{build_failure_prefix(SEED_COMMAND, prefix)} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_report(prefix, report))
    return EXIT_SUCCESS


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command in `argv` and return the process exit code."""
    arguments = build_parser().parse_args(argv)
    if arguments.command == FETCH_COMMAND:
        request = FetchRequest(
            page_type=arguments.page_type,
            titles=tuple(arguments.titles),
            is_update=arguments.update,
        )
        return execute_fetch(request)
    if arguments.command == PARSE_COMMAND:
        return execute_parse(ParseRequest(titles=tuple(arguments.titles)))

    try:
        settings = get_settings()
    except ValidationError as error:
        failure_prefix = build_failure_prefix(arguments.command, arguments.prefix)
        print(f"{failure_prefix} invalid settings: {error}", file=sys.stderr)
        return EXIT_FAILURE

    if arguments.command == PURGE_COMMAND:
        return execute_purge(settings, arguments.prefix)
    return execute_seed(settings, arguments.prefix)
