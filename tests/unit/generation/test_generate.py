from pathlib import Path

import pytest
from pydantic import SecretStr

from muon.agent.routing import AgentRuntime, UniverseScope
from muon.config import AgentSettings
from muon.generation.answer import AnswerDraft, Claim
from muon.generation.generate import PROMPT_PATH, generate_answer
from muon.retrieval.evidence import (
    Evidence,
    EvidenceBundle,
    EvidenceStatus,
    GraphSlice,
    NodeKey,
    SliceNode,
)
from muon.retrieval.retrieve import fill_prompt
from tests.unit.chat_fakes import ScriptedChatModel
from tests.unit.retrieval.scope import build_toy_scope
from tests.unit.retrieval.test_cypher import StubReadClient

KEY = NodeKey(label="ToyWidget", id="w1")
QUESTION = "What is W1?"


@pytest.fixture
async def scope(tmp_path: Path) -> UniverseScope:
    return await build_toy_scope(tmp_path)


def build_bundle(row_count: int = 1) -> EvidenceBundle:
    evidence = Evidence(
        id="e1",
        universe="muontoy",
        tool="run_cypher",
        query="MATCH (n) RETURN n",
        purpose="widgets",
        status=EvidenceStatus.OK,
        columns=("i",),
        rows=tuple({"i": index} for index in range(row_count)),
        row_count=row_count,
        graph_slice=GraphSlice(nodes=(SliceNode(key=KEY, name="W1", labels=(), source_url=None),)),
    )
    return EvidenceBundle(universe="muontoy", question=QUESTION, items=(evidence,))


def build_draft(evidence_id: str = "e1") -> AnswerDraft:
    claim = Claim(id="c1", text="W1 is a widget", evidence_ids=[evidence_id], node_keys=[KEY])
    return AnswerDraft(
        text="W1 is a widget [c1].", claims=[claim], assumptions=[], table_evidence_ids=[]
    )


def build_runtime(
    drafts: list[AnswerDraft], **limits: int
) -> tuple[AgentRuntime, ScriptedChatModel]:
    model = ScriptedChatModel(drafts=drafts)
    settings = AgentSettings(
        _env_file=None,
        base_url="https://a.test/v1",
        api_key=SecretStr("k"),
        default_model="m",
        **limits,
    )
    runtime = AgentRuntime(
        client=StubReadClient([]),
        resolve_model=model,
        retrieve_model=model,
        generate_model=model,
        agent_settings=settings,
    )  # type: ignore[arg-type]
    return runtime, model


def human_text(model: ScriptedChatModel, call: int) -> str:
    return str(model.calls[call][1].content)


async def test_the_human_message_holds_the_content_and_never_a_row_past_50(
    scope: UniverseScope,
) -> None:
    runtime, model = build_runtime([build_draft()])

    await generate_answer(QUESTION, build_bundle(row_count=120), scope, runtime)

    text = human_text(model, 0)
    assert QUESTION in text
    assert "e1 ok, 120 rows, first 50 shown: widgets" in text
    assert "\n49\n" in text + "\n"
    assert "\n50\n" not in text + "\n"


async def test_a_clean_first_draft_makes_one_call_and_no_warning(scope: UniverseScope) -> None:
    runtime, model = build_runtime([build_draft()])

    result = await generate_answer(QUESTION, build_bundle(), scope, runtime)

    assert result.answer.warnings == []
    assert result.answer.universe == scope.universe_label
    assert [stats.step for stats in result.stats] == ["generate"]
    assert len(model.calls) == 1


async def test_a_bad_first_draft_and_a_clean_second_give_no_warning(scope: UniverseScope) -> None:
    runtime, model = build_runtime([build_draft("e9"), build_draft()])

    result = await generate_answer(QUESTION, build_bundle(), scope, runtime)

    assert result.answer.warnings == []
    assert [stats.step for stats in result.stats] == ["generate", "generate_retry_1"]
    assert [stats.model_calls for stats in result.stats] == [1, 1]
    assert "c1 cites e9, which does not exist" in human_text(model, 1)
    assert "citation issues" not in human_text(model, 0)


async def test_two_bad_drafts_give_one_warning_per_issue(scope: UniverseScope) -> None:
    runtime, _ = build_runtime([build_draft("e9"), build_draft("e8")])

    result = await generate_answer(QUESTION, build_bundle(), scope, runtime)

    assert result.answer.warnings == ["c1 cites e8, which does not exist"]
    assert len(result.stats) == 2


async def test_zero_check_retries_make_one_call(scope: UniverseScope) -> None:
    runtime, model = build_runtime([build_draft("e9")], max_check_retries=0)

    result = await generate_answer(QUESTION, build_bundle(), scope, runtime)

    assert len(model.calls) == 1
    assert result.answer.warnings == ["c1 cites e9, which does not exist"]


def test_generate_prompt_fills_with_its_marker() -> None:
    text = fill_prompt(PROMPT_PATH, {"universe_name": "Toy"})

    assert "{{" not in text
    assert "Toy universe" in text
