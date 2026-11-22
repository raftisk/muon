from contextlib import aclosing
from pathlib import Path

import httpx2
import openai
import pytest
from pydantic import SecretStr

from muon.agent.events import TextDelta
from muon.agent.routing import AgentRuntime, UniverseScope
from muon.config import AgentSettings
from muon.generation.generate import (
    NO_EVIDENCE_WARNING,
    STREAM_PROMPT_PATH,
    GeneratedAnswer,
    stream_answer,
)
from muon.llm.client import ModelCallError
from muon.retrieval.evidence import (
    Evidence,
    EvidenceBundle,
    EvidenceStatus,
    GraphSlice,
    NodeCell,
    NodeKey,
    SliceNode,
)
from muon.retrieval.retrieve import fill_prompt
from tests.unit.chat_fakes import DEFAULT_USAGE, ScriptedChatModel
from tests.unit.retrieval.scope import build_toy_scope
from tests.unit.retrieval.test_cypher import StubReadClient

KEY = NodeKey(label="ToyWidget", id="w1")
QUESTION = "What is W1?"
URL = "https://example.org/w1"
ATTRIBUTION = "Text from the toy wiki."
CHUNKS = ["W1 is a widget [n", "1] and a gadget [t", "1]."]


@pytest.fixture
async def scope(tmp_path: Path) -> UniverseScope:
    return await build_toy_scope(tmp_path, extra_toml=f'attribution = "{ATTRIBUTION}"')


def build_evidence(
    row_count: int = 1, status: EvidenceStatus = EvidenceStatus.OK, evidence_id: str = "e1"
) -> Evidence:
    rows = tuple({"w": NodeCell(key=KEY), "i": index} for index in range(row_count))
    return Evidence(
        id=evidence_id,
        universe="muontoy",
        tool="run_cypher",
        query="MATCH (n) RETURN n",
        purpose="widgets",
        status=status,
        columns=("w", "i"),
        rows=rows,
        row_count=row_count,
        graph_slice=GraphSlice(nodes=(SliceNode(key=KEY, name="W1", labels=(), source_url=URL),)),
    )


def build_bundle(*items: Evidence) -> EvidenceBundle:
    return EvidenceBundle(universe="muontoy", question=QUESTION, items=items or (build_evidence(),))


def build_runtime(model: ScriptedChatModel) -> AgentRuntime:
    settings = AgentSettings(
        _env_file=None, base_url="https://a.test/v1", api_key=SecretStr("k"), default_model="m"
    )
    return AgentRuntime(
        client=StubReadClient([]),
        resolve_model=model,
        retrieve_model=model,
        generate_model=model,
        agent_settings=settings,
    )  # type: ignore[arg-type]


async def run_stream(
    model: ScriptedChatModel, bundle: EvidenceBundle, scope: UniverseScope
) -> tuple[list[TextDelta], GeneratedAnswer]:
    deltas: list[TextDelta] = []
    generated: GeneratedAnswer | None = None
    async for event in stream_answer(QUESTION, bundle, scope, build_runtime(model)):
        if isinstance(event, TextDelta):
            deltas.append(event)
        else:
            generated = event
    assert generated is not None
    return deltas, generated


async def test_a_delta_arrives_before_the_last_chunk_and_holds_no_partial_marker(
    scope: UniverseScope,
) -> None:
    model = ScriptedChatModel(streams=[CHUNKS])
    seen_progress: list[int] = []

    async for event in stream_answer(QUESTION, build_bundle(), scope, build_runtime(model)):
        if isinstance(event, TextDelta):
            seen_progress.append(len(model.stream_progress))
            assert "[n" not in event.text
            assert "[t" not in event.text

    assert seen_progress[0] < len(CHUNKS)


async def test_the_joined_deltas_equal_the_answer_text_with_its_attribution(
    scope: UniverseScope,
) -> None:
    deltas, generated = await run_stream(ScriptedChatModel(streams=[CHUNKS]), build_bundle(), scope)

    assert "".join(delta.text for delta in deltas) == generated.answer.text
    assert generated.answer.text.endswith(ATTRIBUTION)
    assert deltas[-1].text.strip() == ATTRIBUTION


async def test_a_node_handle_gives_a_source_and_a_graph(scope: UniverseScope) -> None:
    _, generated = await run_stream(ScriptedChatModel(streams=[CHUNKS]), build_bundle(), scope)

    assert [source.url for source in generated.answer.sources] == [URL]
    assert generated.answer.graph is not None
    assert generated.answer.cited.node_keys == (KEY,)
    assert generated.answer.warnings == ()


async def test_a_table_handle_gives_a_table_with_every_row(scope: UniverseScope) -> None:
    model = ScriptedChatModel(streams=[["The full table follows [t1]."]])

    _, generated = await run_stream(model, build_bundle(build_evidence(row_count=80)), scope)

    assert len(generated.answer.tables) == 1
    table_data = generated.answer.tables[0].artifact.data
    assert len(table_data.rows) == 80  # type: ignore[union-attr]
    assert "[t1]" not in generated.answer.text


async def test_an_unknown_handle_is_dropped_and_warned(scope: UniverseScope) -> None:
    model = ScriptedChatModel(streams=[["W1 is a widget [n99]."]])

    _, generated = await run_stream(model, build_bundle(), scope)

    assert "[n99]" not in generated.answer.text
    assert any("[n99]" in warning for warning in generated.answer.warnings)


async def test_an_answer_with_no_handle_and_an_ok_item_with_rows_is_warned(
    scope: UniverseScope,
) -> None:
    model = ScriptedChatModel(streams=[["W1 is a widget."]])

    _, generated = await run_stream(model, build_bundle(), scope)

    assert NO_EVIDENCE_WARNING in generated.answer.warnings


async def test_an_answer_with_no_handle_and_only_empty_items_is_not_warned(
    scope: UniverseScope,
) -> None:
    model = ScriptedChatModel(streams=[["The graph holds nothing on W1."]])
    bundle = build_bundle(
        Evidence(
            id="e1",
            universe="muontoy",
            tool="run_cypher",
            query="MATCH (n) RETURN n",
            purpose="widgets",
            status=EvidenceStatus.EMPTY,
            columns=(),
            rows=(),
            row_count=0,
            graph_slice=GraphSlice(),
        )
    )

    _, generated = await run_stream(model, bundle, scope)

    assert generated.answer.warnings == ()


async def test_an_api_error_after_two_chunks_becomes_a_model_call_error(
    scope: UniverseScope,
) -> None:
    request = httpx2.Request("POST", "https://a.test/v1/chat/completions")
    failure = openai.APIConnectionError(request=request)
    model = ScriptedChatModel(streams=[CHUNKS], stream_failure=failure, fail_after=2)
    deltas: list[TextDelta] = []

    with pytest.raises(ModelCallError) as caught:
        async for event in stream_answer(QUESTION, build_bundle(), scope, build_runtime(model)):
            assert isinstance(event, TextDelta)
            deltas.append(event)

    assert caught.value.step == "generate"
    assert len(deltas) <= 2


async def test_closing_the_stream_after_the_first_delta_closes_the_model_stream(
    scope: UniverseScope,
) -> None:
    model = ScriptedChatModel(streams=[CHUNKS])
    stream = stream_answer(QUESTION, build_bundle(), scope, build_runtime(model))

    async with aclosing(stream):
        await anext(stream)
        assert not model.is_stream_closed

    assert model.is_stream_closed


async def test_the_human_message_holds_handles_and_no_node_id(scope: UniverseScope) -> None:
    model = ScriptedChatModel(streams=[CHUNKS])

    await run_stream(model, build_bundle(), scope)

    text = str(model.calls[0][1].content)
    assert QUESTION in text
    assert "[n1]" in text
    assert "[t1]" in text
    assert KEY.id not in text


async def test_the_stats_count_one_model_call_with_the_fake_usage(scope: UniverseScope) -> None:
    _, generated = await run_stream(ScriptedChatModel(streams=[CHUNKS]), build_bundle(), scope)

    assert generated.stats.step == "generate"
    assert generated.stats.model_calls == 1
    assert generated.stats.input_tokens == DEFAULT_USAGE["input_tokens"]
    assert generated.stats.output_tokens == DEFAULT_USAGE["output_tokens"]


def test_the_stream_prompt_fills_and_forbids_urls_and_json() -> None:
    text = fill_prompt(STREAM_PROMPT_PATH, {"universe_name": "Toy"})

    assert "{{" not in text
    assert "Toy universe" in text
