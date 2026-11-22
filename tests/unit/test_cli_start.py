import argparse
from functools import partial
from typing import Any

import pytest

from muon import cli
from muon.config import AgentSettings, OntologySettings, Settings, UniverseSettings
from muon.plugins import UniverseRegistry

AGENT_ENV = {
    "MUON_AGENT_BASE_URL": "https://a.test/v1",
    "MUON_AGENT_API_KEY": "secret",
    "MUON_AGENT_DEFAULT_MODEL": "model",
}


def build_settings() -> Settings:
    return Settings.model_construct(universes=UniverseSettings(), ontology=OntologySettings())


def record_server(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **options: calls.append(options))
    monkeypatch.setattr(cli, "discover_universes", lambda *args: UniverseRegistry({}))
    monkeypatch.setattr(cli, "select_universe", lambda registry, requested: None)
    return calls


def test_start_parser_takes_host_and_port() -> None:
    arguments = cli.build_parser().parse_args(["start", "--host", "0.0.0.0", "--port", "9000"])

    assert (arguments.host, arguments.port) == ("0.0.0.0", 9000)


def test_start_runs_uvicorn_on_the_given_port(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = record_server(monkeypatch)
    for name, value in AGENT_ENV.items():
        monkeypatch.setenv(name, value)
    arguments = argparse.Namespace(command="start", host=None, port=9000)

    exit_code = cli.execute_start(build_settings(), arguments)

    assert exit_code == cli.EXIT_SUCCESS
    assert calls[0]["port"] == 9000
    assert calls[0]["host"] == "127.0.0.1"


def test_start_without_agent_settings_fails_before_the_port_opens(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = record_server(monkeypatch)
    for name in AGENT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "AgentSettings", partial(AgentSettings, _env_file=None))
    arguments = argparse.Namespace(command="start", host=None, port=None)

    exit_code = cli.execute_start(build_settings(), arguments)

    assert exit_code == cli.EXIT_FAILURE
    assert "missing or invalid settings: MUON_AGENT_" in capsys.readouterr().err
    assert calls == []
