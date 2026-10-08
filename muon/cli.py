"""Command line entry point: `muon seed <prefix>`."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import fields

from pydantic import ValidationError

from muon.config import Settings, get_settings
from muon.graph import GraphError, Neo4jGraphClient
from muon.ingestion.runner import RunReport
from muon.ingestion.seed import ABoxError, seed_universe
from muon.ontology.model import OntologyError

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
PROGRAM_NAME = "muon"
SEED_COMMAND = "seed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description="muon knowledge graph tools")
    commands = parser.add_subparsers(dest="command", required=True)
    seed = commands.add_parser(
        SEED_COMMAND, help="load a universe's A-Box into Neo4j and keep the graph in sync with it"
    )
    seed.add_argument("prefix", help="universe prefix, the stem of individuals/<prefix>.ttl")
    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command in `argv` and return the process exit code."""
    arguments = build_parser().parse_args(argv)
    failure_prefix = f"{SEED_COMMAND} {arguments.prefix} failed:"
    try:
        settings = get_settings()
    except ValidationError as error:
        print(f"{failure_prefix} invalid settings: {error}", file=sys.stderr)
        return EXIT_FAILURE

    try:
        report = asyncio.run(run_seed(settings, arguments.prefix))
    except (OntologyError, ABoxError, GraphError) as error:
        print(f"{failure_prefix} {error}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_report(arguments.prefix, report))
    return EXIT_SUCCESS
