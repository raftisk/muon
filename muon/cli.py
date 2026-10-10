"""Command line entry point: `muon seed <prefix>`, `muon purge <prefix>` and
`muon fetch bulbapedia <page-type>`."""

import argparse
import asyncio
import getpass
import secrets
import sys
from collections.abc import Sequence
from dataclasses import fields

from pydantic import SecretStr, ValidationError

from muon.config import Settings, get_settings
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

    try:
        settings = get_settings()
    except ValidationError as error:
        failure_prefix = build_failure_prefix(arguments.command, arguments.prefix)
        print(f"{failure_prefix} invalid settings: {error}", file=sys.stderr)
        return EXIT_FAILURE

    if arguments.command == PURGE_COMMAND:
        return execute_purge(settings, arguments.prefix)
    return execute_seed(settings, arguments.prefix)
