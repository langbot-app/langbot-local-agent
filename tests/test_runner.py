"""Tests for local-agent runner functionality."""

from __future__ import annotations

import asyncio
import os

# Import modules to test
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from langbot_plugin.api.entities.builtin.agent_runner.context import AdapterContext, AgentRunContext
from langbot_plugin.api.entities.builtin.agent_runner.context_access import (
    ContextAccess,
    ContextAPICapabilities,
)
from langbot_plugin.api.entities.builtin.agent_runner.delivery import DeliveryContext
from langbot_plugin.api.entities.builtin.agent_runner.event import AgentEventContext
from langbot_plugin.api.entities.builtin.agent_runner.input import AgentInput
from langbot_plugin.api.entities.builtin.agent_runner.resources import (
    AgentResources,
    KnowledgeBaseResource,
    ModelResource,
    ToolResource,
)
from langbot_plugin.api.entities.builtin.agent_runner.result import (
    AgentRunResultType,
)
from langbot_plugin.api.entities.builtin.agent_runner.runtime import AgentRuntimeContext
from langbot_plugin.api.entities.builtin.agent_runner.trigger import AgentTrigger
from langbot_plugin.api.entities.builtin.provider.message import (
    ContentElement,
    FunctionCall,
    Message,
    MessageChunk,
    ToolCall,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkg.config import get_knowledge_base_ids, get_rerank_config, parse_model_config
from pkg.context_pipeline import ContextBudget, ContextCompactor
from pkg.messages import build_messages, format_rag_results, get_effective_prompt_config

# ==================== Fixtures ====================


class FakeAgentRunAPIProxy:
    """Fake API proxy for testing."""

    def __init__(
        self,
        models: list[ModelResource] | None = None,
        tools: list[ToolResource] | None = None,
        knowledge_bases: list[KnowledgeBaseResource] | None = None,
    ):
        self._models = models or []
        self._tools = tools or []
        self._knowledge_bases = knowledge_bases or []

        # Mock methods
        self.invoke_llm = AsyncMock()
        self.invoke_llm_stream = AsyncMock()
        self.get_tool_detail = AsyncMock(
            side_effect=lambda tool_name: {
                "name": tool_name,
                "description": f"Tool {tool_name}",
                "parameters": {"type": "object", "properties": {}},
            }
        )
        self.call_tool = AsyncMock()
        self.retrieve_knowledge = AsyncMock()
        self.invoke_rerank = AsyncMock()
        self.history_page = AsyncMock(
            return_value={
                "items": [],
                "next_cursor": None,
                "prev_cursor": None,
                "has_more": False,
            }
        )

    def get_allowed_models(self) -> list[ModelResource]:
        return self._models

    def get_allowed_tools(self) -> list[ToolResource]:
        return self._tools

    def get_allowed_knowledge_bases(self) -> list[KnowledgeBaseResource]:
        return self._knowledge_bases


def make_context(
    run_id: str = "test-run-id",
    config: dict[str, Any] | None = None,
    resources: AgentResources | None = None,
    input_text: str = "test input",
    input_contents: list[ContentElement] | None = None,
    runtime_metadata: dict[str, Any] | None = None,
    adapter_extra: dict[str, Any] | None = None,
    history_available: bool = False,
    conversation_id: str = "conv-test",
) -> AgentRunContext:
    """Create a test AgentRunContext."""
    return AgentRunContext(
        run_id=run_id,
        trigger=AgentTrigger(type="message.received"),
        event=AgentEventContext(
            event_id=f"evt-{run_id}",
            event_type="message.received",
            source="pipeline_adapter",
        ),
        input=AgentInput(text=input_text, contents=input_contents or []),
        delivery=DeliveryContext(
            surface="pipeline",
            supports_streaming=(runtime_metadata or {}).get("streaming_supported", True),
        ),
        resources=resources or AgentResources(),
        context=ContextAccess(
            conversation_id=conversation_id,
            available_apis=ContextAPICapabilities(
                history_page=history_available,
                history_search=history_available,
            ),
        ),
        runtime=AgentRuntimeContext(query_id=1, metadata=runtime_metadata or {}),
        config=config or {},
        adapter=AdapterContext(extra=adapter_extra or {}),
    )


# ==================== Config Parsing Tests ====================


class TestParseModelConfig:
    """Tests for model configuration parsing."""

    def test_string_format_primary_success(self):
        """String format: primary model configured and allowed."""
        result = parse_model_config("model-123", {"model-123", "model-456"})
        assert result == ["model-123"]

    def test_string_format_primary_not_allowed(self):
        """String format: primary model not in allowed set."""
        result = parse_model_config("model-999", {"model-123", "model-456"})
        assert result == []

    def test_string_format_empty_string(self):
        """String format: empty string returns empty."""
        result = parse_model_config("", {"model-123"})
        assert result == []

    def test_string_format_none_value(self):
        """String format: __none__ returns empty."""
        result = parse_model_config("__none__", {"model-123"})
        assert result == []

    def test_dict_format_primary_only(self):
        """Dict format: only primary model configured."""
        result = parse_model_config(
            {"primary": "model-123"},
            {"model-123", "model-456"},
        )
        assert result == ["model-123"]

    def test_dict_format_with_fallbacks(self):
        """Dict format: primary + fallbacks in order."""
        result = parse_model_config(
            {"primary": "model-1", "fallbacks": ["model-2", "model-3"]},
            {"model-1", "model-2", "model-3", "model-4"},
        )
        assert result == ["model-1", "model-2", "model-3"]

    def test_dict_format_fallback_not_allowed(self):
        """Dict format: fallback not in allowed set is filtered out."""
        result = parse_model_config(
            {"primary": "model-1", "fallbacks": ["model-999", "model-2"]},
            {"model-1", "model-2"},
        )
        assert result == ["model-1", "model-2"]

    def test_dict_format_no_duplicates(self):
        """Dict format: duplicate model IDs are deduplicated."""
        result = parse_model_config(
            {"primary": "model-1", "fallbacks": ["model-1", "model-2"]},
            {"model-1", "model-2"},
        )
        assert result == ["model-1", "model-2"]

    def test_dict_format_primary_not_allowed(self):
        """Dict format: primary not allowed, fallbacks used."""
        result = parse_model_config(
            {"primary": "model-999", "fallbacks": ["model-1", "model-2"]},
            {"model-1", "model-2"},
        )
        assert result == ["model-1", "model-2"]

    def test_none_config(self):
        """None config returns empty list."""
        result = parse_model_config(None, {"model-1"})
        assert result == []

    def test_invalid_format(self):
        """Invalid format returns empty list."""
        result = parse_model_config(123, {"model-1"})
        assert result == []


class TestGetKnowledgeBaseIds:
    """Tests for knowledge base ID parsing."""

    def test_empty_config(self):
        """Empty config returns empty list."""
        result = get_knowledge_base_ids({}, {"kb-1"})
        assert result == []

    def test_single_kb(self):
        """Single KB ID is returned."""
        result = get_knowledge_base_ids({"knowledge-bases": ["kb-1"]}, {"kb-1", "kb-2"})
        assert result == ["kb-1"]

    def test_multiple_kbs(self):
        """Multiple KB IDs are returned."""
        result = get_knowledge_base_ids(
            {"knowledge-bases": ["kb-1", "kb-2"]},
            {"kb-1", "kb-2", "kb-3"},
        )
        assert result == ["kb-1", "kb-2"]

    def test_kb_not_allowed(self):
        """KB not in allowed set is filtered out."""
        result = get_knowledge_base_ids(
            {"knowledge-bases": ["kb-1", "kb-999"]},
            {"kb-1"},
        )
        assert result == ["kb-1"]

    def test_none_value(self):
        """__none__ is filtered out."""
        result = get_knowledge_base_ids(
            {"knowledge-bases": ["kb-1", "__none__"]},
            {"kb-1"},
        )
        assert result == ["kb-1"]

    def test_legacy_single_knowledge_base(self):
        """Legacy singular knowledge-base config is still supported."""
        result = get_knowledge_base_ids(
            {"knowledge-base": "kb-1"},
            {"kb-1", "kb-2"},
        )
        assert result == ["kb-1"]


class TestGetRerankConfig:
    """Tests for rerank configuration parsing."""

    def test_empty_config(self):
        """Empty config disables rerank and keeps default top-k."""
        assert get_rerank_config({}) == (None, 5)

    def test_valid_config(self):
        """Valid rerank model and top-k are returned."""
        assert get_rerank_config({"rerank-model": "rerank-1", "rerank-top-k": 2}) == ("rerank-1", 2)

    def test_none_model_disables_rerank(self):
        """__none__ disables rerank."""
        assert get_rerank_config({"rerank-model": "__none__", "rerank-top-k": 2}) == (None, 2)

    def test_invalid_top_k_uses_default(self):
        """Invalid rerank top-k falls back to default."""
        assert get_rerank_config({"rerank-model": "rerank-1", "rerank-top-k": 0}) == ("rerank-1", 5)


# ==================== Message Building Tests ====================


class TestBuildMessages:
    """Tests for message building."""

    def test_basic_messages(self):
        """Basic message building with prompt, history, and input."""
        prompt = [{"role": "system", "content": "You are helpful."}]
        history = [Message(role="user", content="Hi"), Message(role="assistant", content="Hello")]

        messages = build_messages(
            prompt_config=prompt,
            history_messages=history,
            user_text="How are you?",
        )

        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[0].content == "You are helpful."
        assert messages[-1].role == "user"
        assert messages[-1].content == "How are you?"

    def test_history_is_preserved_as_provided(self):
        """History selection is owned by Host APIs and runner policy."""
        history = []
        for i in range(15):
            history.append(Message(role="user", content=f"User {i}"))
            history.append(Message(role="assistant", content=f"Assistant {i}"))

        messages = build_messages(
            prompt_config=[],
            history_messages=history,
            user_text="New message",
        )

        assert len(messages) == 31
        assert messages[0].content == "User 0"
        assert messages[-2].content == "Assistant 14"
        assert messages[-1].content == "New message"

    def test_rag_context(self):
        """RAG context is prepended to user message."""
        messages = build_messages(
            prompt_config=[],
            history_messages=[],
            user_text="What is X?",
            rag_context="[1] X is a variable.",
        )

        assert len(messages) == 1
        assert "<context>" in messages[0].content
        assert "[1] X is a variable." in messages[0].content
        assert "<user_message>" in messages[0].content
        assert "What is X?" in messages[0].content

    def test_static_prompt_config_is_used(self):
        """Static binding prompt config is used for system prompt."""
        messages = build_messages(
            prompt_config=[{"role": "system", "content": "Static prompt"}],
            history_messages=[],
            user_text="Hello",
        )

        assert messages[0].content == "Static prompt"

    def test_effective_prompt_prefers_host_adapter_prompt(self):
        """Host effective prompt replaces static runner prompt."""
        ctx = make_context(
            config={"prompt": [{"role": "system", "content": "Static prompt"}]},
            adapter_extra={"prompt": [{"role": "system", "content": "Host effective prompt"}]},
        )

        assert get_effective_prompt_config(ctx) == [
            {"role": "system", "content": "Host effective prompt"}
        ]

    def test_effective_prompt_allows_host_to_clear_prompt(self):
        """An explicit empty host prompt should not fall back to static config."""
        ctx = make_context(
            config={"prompt": [{"role": "system", "content": "Static prompt"}]},
            adapter_extra={"prompt": []},
        )

        assert get_effective_prompt_config(ctx) == []

    def test_effective_prompt_falls_back_to_static_config_without_adapter_prompt(self):
        """Static runner config is still used outside Pipeline adapter prompt handoff."""
        ctx = make_context(
            config={"prompt": [{"role": "system", "content": "Static prompt"}]},
            adapter_extra={"params": {"public": "value"}},
        )

        assert get_effective_prompt_config(ctx) == [
            {"role": "system", "content": "Static prompt"}
        ]

    def test_multimodal_input_contents_are_preserved(self):
        """Structured input contents are preserved in the current user message."""
        contents = [
            ContentElement.from_text("Look at this"),
            ContentElement.from_image_base64("base64-image"),
        ]

        messages = build_messages(
            prompt_config=[],
            history_messages=[],
            user_text="Look at this",
            input_contents=contents,
        )

        assert isinstance(messages[0].content, list)
        assert messages[0].content[0].text == "Look at this"
        assert messages[0].content[1].type == "image_base64"

    def test_rag_context_replaces_text_and_preserves_multimodal_parts(self):
        """RAG modifies only the text content and keeps attachments."""
        contents = [
            ContentElement.from_text("What is in this image?"),
            ContentElement.from_image_base64("base64-image"),
        ]

        messages = build_messages(
            prompt_config=[],
            history_messages=[],
            user_text="What is in this image?",
            input_contents=contents,
            rag_context="[1] Image metadata",
        )

        assert isinstance(messages[0].content, list)
        assert "<context>" in messages[0].content[0].text
        assert messages[0].content[1].type == "image_base64"


class TestContextCompaction:
    """Tests for runner-owned context budgeting and compaction."""

    def test_budget_from_config_uses_character_fields(self):
        """Context budget uses character limits instead of round counts."""
        budget = ContextBudget.from_config(
            {
                "context-window-chars": 300,
                "context-reserve-chars": 60,
                "context-keep-recent-chars": 80,
                "context-summary-chars": 120,
            }
        )

        assert budget.window_chars == 300
        assert budget.input_chars == 240
        assert budget.keep_recent_chars == 80
        assert budget.summary_chars == 120

    def test_compactor_summarizes_old_history_and_keeps_recent_tail(self):
        """Older history is summarized while recent context and current input remain."""
        history = [
            Message(role="user", content="old user " + "u" * 220),
            Message(role="assistant", content="old assistant " + "a" * 220),
            Message(role="user", content="recent user sentinel"),
            Message(role="assistant", content="recent assistant sentinel"),
        ]
        current = [Message(role="user", content="current request")]
        budget = ContextBudget(
            window_chars=320,
            reserve_chars=60,
            keep_recent_chars=100,
            summary_chars=120,
        )

        assembly = ContextCompactor(budget).compact(
            prompt_messages=[Message(role="system", content="Static prompt")],
            history_messages=history,
            current_messages=current,
        )

        assert assembly.compacted is True
        assert assembly.summary_message is not None
        assert assembly.messages[0].content == "Static prompt"
        assert "<conversation_summary>" in assembly.summary_message.content
        assert "old user" in assembly.summary_message.content
        contents = [message.content for message in assembly.messages]
        assert "recent user sentinel" in contents
        assert "recent assistant sentinel" in contents
        assert contents[-1] == "current request"


class TestFormatRagResults:
    """Tests for RAG result formatting."""

    def test_empty_results(self):
        """Empty results return empty string."""
        assert format_rag_results([]) == ""

    def test_single_result_text(self):
        """Single text result is formatted correctly."""
        results = [{"content": [{"type": "text", "text": "Hello world"}]}]
        result = format_rag_results(results)
        assert "[1] Hello world" in result

    def test_multiple_results(self):
        """Multiple results are numbered."""
        results = [
            {"content": [{"type": "text", "text": "First"}]},
            {"content": [{"type": "text", "text": "Second"}]},
        ]
        result = format_rag_results(results)
        assert "[1] First" in result
        assert "[2] Second" in result

    def test_string_content(self):
        """String content (not list) is handled."""
        results = [{"content": "Plain text content"}]
        result = format_rag_results(results)
        assert "[1] Plain text content" in result


# ==================== Runner Integration Tests ====================


class TestDefaultAgentRunner:
    """Tests for DefaultAgentRunner behavior."""

    def test_manifest_declares_event_context_capability(self):
        """The local runner consumes Protocol v1 event-first context from Host."""
        manifest = yaml.safe_load((Path(__file__).resolve().parents[1] / "components/agent_runner/default.yaml").read_text())

        assert manifest["spec"]["capabilities"]["event_context"] is True
        assert manifest["spec"]["permissions"]["history"] == ["page", "search"]
        config_names = {item["name"] for item in manifest["spec"]["config"]}
        assert "context-window-chars" in config_names
        assert "context-keep-recent-chars" in config_names

    @pytest.fixture
    def runner(self):
        """Create a runner instance."""
        from components.agent_runner.default import DefaultAgentRunner
        runner = DefaultAgentRunner()
        runner.bind_runtime(
            plugin_runtime_handler=MagicMock(),
            plugin_identity="langbot/local-agent",
        )
        return runner

    @pytest.mark.asyncio
    async def test_no_model_returns_failed(self, runner):
        """No authorized model returns run.failed with code=runner.no_model."""
        ctx = make_context(
            config={"model": "model-1"},
            resources=AgentResources(models=[]),  # No models authorized
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        assert len(results) == 1
        assert results[0].type == AgentRunResultType.RUN_FAILED
        assert results[0].data.get("code") == "runner.no_model"

    @pytest.mark.asyncio
    async def test_primary_success_no_fallback(self, runner, monkeypatch):
        """Primary model succeeds, no fallback needed."""
        # Setup fake API
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
        )

        # Mock invoke_llm_stream to yield successful chunks
        async def mock_stream(*args, **kwargs):
            yield MessageChunk(role="assistant", content="Hello", is_final=False)
            yield MessageChunk(role="assistant", content=" world", is_final=True)

        fake_api.invoke_llm_stream = mock_stream

        # Patch get_run_api
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": {"primary": "model-1", "fallbacks": ["model-2"]}},
            resources=AgentResources(models=[ModelResource(model_id="model-1")]),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        # Should have message deltas and run.completed
        assert any(r.type == AgentRunResultType.MESSAGE_DELTA for r in results)
        assert any(r.type == AgentRunResultType.RUN_COMPLETED for r in results)
        fake_api.history_page.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_history_is_pulled_from_host_api(self, runner, monkeypatch):
        """Conversation history comes from Host history API, not adapter bootstrap."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
        )
        fake_api.history_page = AsyncMock(
            return_value={
                "items": [
                    {
                        "event_id": "evt-run-history",
                        "role": "user",
                        "content": "current input",
                    },
                    {
                        "event_id": "evt-old-assistant",
                        "content_json": {"role": "assistant", "content": "previous answer"},
                    },
                    {
                        "event_id": "evt-old-user",
                        "role": "user",
                        "content": "previous question",
                    },
                ],
                "next_cursor": None,
                "prev_cursor": None,
                "has_more": False,
            }
        )
        captured_messages: list[Message] = []

        async def mock_stream(*args, **kwargs):
            captured_messages.extend(kwargs["messages"])
            yield MessageChunk(role="assistant", content="Response", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            run_id="run-history",
            config={
                "model": "model-1",
                "prompt": [{"role": "system", "content": "Static prompt"}],
            },
            resources=AgentResources(models=[ModelResource(model_id="model-1")]),
            input_text="current input",
            history_available=True,
        )

        results = [result async for result in runner.run(ctx)]

        assert any(r.type == AgentRunResultType.RUN_COMPLETED for r in results)
        fake_api.history_page.assert_awaited_once_with(
            conversation_id="conv-test",
            limit=50,
            direction="backward",
            include_artifacts=True,
        )
        assert [(msg.role, msg.content) for msg in captured_messages] == [
            ("system", "Static prompt"),
            ("user", "previous question"),
            ("assistant", "previous answer"),
            ("user", "current input"),
        ]

    @pytest.mark.asyncio
    async def test_pipeline_effective_prompt_replaces_static_prompt(self, runner, monkeypatch):
        """PromptPreProcessing output from Host is the model-facing prompt."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
        )
        captured_messages: list[Message] = []

        async def mock_stream(*args, **kwargs):
            captured_messages.extend(kwargs["messages"])
            yield MessageChunk(role="assistant", content="Response", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={
                "model": "model-1",
                "prompt": [{"role": "system", "content": "Static prompt"}],
            },
            adapter_extra={
                "prompt": [{"role": "system", "content": "Host effective prompt"}],
            },
            resources=AgentResources(models=[ModelResource(model_id="model-1")]),
            input_text="qa-effective-prompt",
        )

        results = [result async for result in runner.run(ctx)]

        assert any(r.type == AgentRunResultType.RUN_COMPLETED for r in results)
        assert [(msg.role, msg.content) for msg in captured_messages] == [
            ("system", "Host effective prompt"),
            ("user", "qa-effective-prompt"),
        ]

    @pytest.mark.asyncio
    async def test_context_compaction_runs_before_model_call(self, runner, monkeypatch):
        """Long host history is compacted by budget, not by a max-round setting."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
        )
        fake_api.history_page = AsyncMock(
            return_value={
                "items": [
                    {
                        "event_id": "evt-run-compact",
                        "role": "user",
                        "content": "current compact request",
                    },
                    {
                        "event_id": "evt-recent-assistant",
                        "role": "assistant",
                        "content": "recent assistant sentinel",
                    },
                    {
                        "event_id": "evt-recent-user",
                        "role": "user",
                        "content": "recent user sentinel",
                    },
                    {
                        "event_id": "evt-old-assistant",
                        "role": "assistant",
                        "content": "old assistant " + "a" * 220,
                    },
                    {
                        "event_id": "evt-old-user",
                        "role": "user",
                        "content": "old user " + "u" * 220,
                    },
                ],
                "next_cursor": None,
                "prev_cursor": None,
                "has_more": False,
            }
        )
        captured_messages: list[Message] = []

        async def mock_stream(*args, **kwargs):
            captured_messages.extend(kwargs["messages"])
            yield MessageChunk(role="assistant", content="Response", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            run_id="run-compact",
            config={
                "model": "model-1",
                "prompt": [{"role": "system", "content": "Static prompt"}],
                "context-window-chars": 340,
                "context-reserve-chars": 80,
                "context-keep-recent-chars": 90,
                "context-summary-chars": 120,
            },
            resources=AgentResources(models=[ModelResource(model_id="model-1")]),
            input_text="current compact request",
            history_available=True,
        )

        results = [result async for result in runner.run(ctx)]

        assert any(r.type == AgentRunResultType.RUN_COMPLETED for r in results)
        assert captured_messages[0].content == "Static prompt"
        assert captured_messages[1].role == "system"
        assert "<conversation_summary>" in captured_messages[1].content
        assert "old user" in captured_messages[1].content
        contents = [message.content for message in captured_messages]
        assert "recent user sentinel" in contents
        assert "recent assistant sentinel" in contents
        assert contents[-1] == "current compact request"

    @pytest.mark.asyncio
    async def test_streaming_skips_none_chunks(self, runner, monkeypatch):
        """Provider heartbeat/no-op chunks do not fail a committed stream."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
        )

        async def mock_stream(*args, **kwargs):
            yield None
            yield MessageChunk(role="assistant", content="Hello", is_final=False)
            yield None
            yield MessageChunk(role="assistant", content=" world", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": "model-1"},
            resources=AgentResources(models=[ModelResource(model_id="model-1")]),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        deltas = [
            result.data["chunk"]["content"]
            for result in results
            if result.type == AgentRunResultType.MESSAGE_DELTA
        ]
        assert deltas == ["Hello world"]
        assert any(r.type == AgentRunResultType.RUN_COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_runs_do_not_share_per_run_state(self, runner, monkeypatch):
        """The same runner instance can process multiple runs concurrently."""
        api_a = FakeAgentRunAPIProxy(models=[ModelResource(model_id="model-a")])
        api_b = FakeAgentRunAPIProxy(models=[ModelResource(model_id="model-b")])

        async def stream_a(*args, **kwargs):
            await asyncio.sleep(0.01)
            yield MessageChunk(role="assistant", content="response-a", is_final=True)

        async def stream_b(*args, **kwargs):
            yield MessageChunk(role="assistant", content="response-b", is_final=True)

        api_a.invoke_llm_stream = stream_a
        api_b.invoke_llm_stream = stream_b

        def get_api(ctx):
            return api_a if ctx.run_id == "run-a" else api_b

        monkeypatch.setattr(runner, "get_run_api", get_api)

        ctx_a = make_context(
            run_id="run-a",
            config={"model": "model-a"},
            resources=AgentResources(models=[ModelResource(model_id="model-a")]),
            input_text="input a",
        )
        ctx_b = make_context(
            run_id="run-b",
            config={"model": "model-b"},
            resources=AgentResources(models=[ModelResource(model_id="model-b")]),
            input_text="input b",
        )

        async def collect(ctx):
            return [result async for result in runner.run(ctx)]

        results_a, results_b = await asyncio.gather(collect(ctx_a), collect(ctx_b))

        chunks_a = [
            result.data["chunk"]["content"]
            for result in results_a
            if result.type == AgentRunResultType.MESSAGE_DELTA
        ]
        chunks_b = [
            result.data["chunk"]["content"]
            for result in results_b
            if result.type == AgentRunResultType.MESSAGE_DELTA
        ]

        assert chunks_a == ["response-a"]
        assert chunks_b == ["response-b"]

    @pytest.mark.asyncio
    async def test_streaming_fallback_before_first_chunk(self, runner, monkeypatch):
        """Streaming: primary fails before first chunk, fallback succeeds."""
        fake_api = FakeAgentRunAPIProxy(
            models=[
                ModelResource(model_id="model-1"),
                ModelResource(model_id="model-2"),
            ],
        )

        # First call raises exception, second call succeeds
        call_count = [0]

        async def mock_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Primary model fails
                raise Exception("Primary model error")
            # Fallback succeeds
            yield MessageChunk(role="assistant", content="Fallback response", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": {"primary": "model-1", "fallbacks": ["model-2"]}},
            resources=AgentResources(
                models=[
                    ModelResource(model_id="model-1"),
                    ModelResource(model_id="model-2"),
                ]
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        # Should have message from fallback
        assert any(r.type == AgentRunResultType.MESSAGE_DELTA for r in results)
        assert any(r.type == AgentRunResultType.RUN_COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_tool_call_only_authorized_tools(self, runner, monkeypatch):
        """Tool calls only execute for authorized tools."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
            tools=[ToolResource(tool_name="allowed_tool")],
        )

        # Mock streaming with tool call on first call, then no tool calls
        call_count = [0]

        async def mock_stream(*args, **kwargs):
            from langbot_plugin.api.entities.builtin.provider.message import (
                FunctionCall,
                ToolCall,
            )
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: has tool call
                yield MessageChunk(
                    role="assistant",
                    content="",
                    is_final=True,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            type="function",
                            function=FunctionCall(name="allowed_tool", arguments='{"arg": "value"}'),
                        )
                    ],
                )
            else:
                # Second call (after tool result): no tool calls, just response
                yield MessageChunk(role="assistant", content="Done!", is_final=True)

        fake_api.invoke_llm_stream = mock_stream

        # Tool call returns success
        fake_api.call_tool = AsyncMock(return_value={"result": "success"})
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": "model-1"},
            resources=AgentResources(
                models=[ModelResource(model_id="model-1")],
                tools=[ToolResource(tool_name="allowed_tool")],
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        # Should have tool.call.started and tool.call.completed
        started = [r for r in results if r.type == AgentRunResultType.TOOL_CALL_STARTED]
        completed = [r for r in results if r.type == AgentRunResultType.TOOL_CALL_COMPLETED]

        assert len(started) == 1
        assert started[0].data.get("tool_name") == "allowed_tool"
        assert len(completed) == 1
        assert completed[0].data.get("error") is None  # No error

    @pytest.mark.asyncio
    async def test_authorized_tool_detail_is_fetched_and_passed_to_model(self, runner, monkeypatch):
        """Allowed tools are resolved through get_tool_detail and passed as LLM tools."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
            tools=[ToolResource(tool_name="allowed_tool")],
        )
        fake_api.get_tool_detail = AsyncMock(
            return_value={
                "name": "allowed_tool",
                "description": "Allowed test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                },
            }
        )

        captured_funcs = []

        async def mock_stream(*args, **kwargs):
            captured_funcs.extend(kwargs.get("funcs", []))
            yield MessageChunk(role="assistant", content="No tools needed", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": "model-1"},
            resources=AgentResources(
                models=[ModelResource(model_id="model-1")],
                tools=[ToolResource(tool_name="allowed_tool")],
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        fake_api.get_tool_detail.assert_awaited_once_with("allowed_tool")
        assert len(captured_funcs) == 1
        assert captured_funcs[0].name == "allowed_tool"
        assert captured_funcs[0].parameters["properties"]["arg"]["type"] == "string"
        assert any(r.type == AgentRunResultType.MESSAGE_DELTA for r in results)

    @pytest.mark.asyncio
    async def test_streaming_tool_loop_preserves_visible_prefix_and_tools(self, runner, monkeypatch):
        """Tool follow-up chunks keep earlier visible content and available tools."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
            tools=[ToolResource(tool_name="allowed_tool")],
        )
        fake_api.call_tool = AsyncMock(return_value={"result": "success"})

        stream_funcs: list[list[Any]] = []

        async def mock_stream(*args, **kwargs):
            stream_funcs.append(kwargs.get("funcs", []))
            if len(stream_funcs) == 1:
                yield MessageChunk(role="assistant", content="before ", is_final=False)
                yield MessageChunk(
                    role="assistant",
                    content="",
                    is_final=True,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            type="function",
                            function=FunctionCall(name="allowed_tool", arguments="{}"),
                        )
                    ],
                )
            else:
                yield MessageChunk(role="assistant", content="after", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": "model-1"},
            resources=AgentResources(
                models=[ModelResource(model_id="model-1")],
                tools=[ToolResource(tool_name="allowed_tool")],
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        deltas = [r.data.get("chunk", {}).get("content") for r in results if r.type == AgentRunResultType.MESSAGE_DELTA]
        assert deltas == ["before ", "before after"]
        assert stream_funcs[1], "tool follow-up stream should keep tools available"

    @pytest.mark.asyncio
    async def test_tool_call_unauthorized_tool_fails(self, runner, monkeypatch):
        """Tool call to unauthorized tool returns error."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
            tools=[ToolResource(tool_name="allowed_tool")],  # Only allowed_tool is authorized
        )

        from langbot_plugin.api.entities.builtin.provider.message import (
            FunctionCall,
            ToolCall,
        )

        # Mock streaming with tool call on first call, then no tool calls
        call_count = [0]

        async def mock_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Tool call to unauthorized tool
                yield MessageChunk(
                    role="assistant",
                    content="",
                    is_final=True,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            type="function",
                            function=FunctionCall(name="unauthorized_tool", arguments="{}"),
                        )
                    ],
                )
            else:
                # Second call: no tool calls
                yield MessageChunk(role="assistant", content="Done", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": "model-1"},
            resources=AgentResources(
                models=[ModelResource(model_id="model-1")],
                tools=[ToolResource(tool_name="allowed_tool")],
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        # Should have tool.call.completed with error
        completed = [r for r in results if r.type == AgentRunResultType.TOOL_CALL_COMPLETED]
        assert len(completed) == 1
        assert "not authorized" in completed[0].data.get("error", "")

    @pytest.mark.asyncio
    async def test_kb_retrieval_only_authorized(self, runner, monkeypatch):
        """KB retrieval only calls authorized knowledge bases."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
            knowledge_bases=[KnowledgeBaseResource(kb_id="kb-1")],  # Only kb-1 authorized
        )

        async def mock_stream(*args, **kwargs):
            yield MessageChunk(role="assistant", content="Response", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        fake_api.retrieve_knowledge = AsyncMock(return_value=[{"content": [{"type": "text", "text": "KB content"}]}])
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": "model-1", "knowledge-bases": ["kb-1", "kb-2"]},  # Request kb-1 and kb-2
            resources=AgentResources(
                models=[ModelResource(model_id="model-1")],
                knowledge_bases=[KnowledgeBaseResource(kb_id="kb-1")],  # Only kb-1 authorized
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        # Should only call retrieve_knowledge for kb-1 (authorized)
        # kb-2 is not in allowed set, so it's filtered out before API call
        assert fake_api.retrieve_knowledge.call_count == 1
        call_args = fake_api.retrieve_knowledge.call_args
        assert call_args.kwargs.get("kb_id") == "kb-1" or call_args.args[0] == "kb-1"

    @pytest.mark.asyncio
    async def test_kb_not_authorized_not_called(self, runner, monkeypatch):
        """KB not in authorized set is not called."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
            knowledge_bases=[],  # No KBs authorized
        )

        async def mock_stream(*args, **kwargs):
            yield MessageChunk(role="assistant", content="Response", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        fake_api.retrieve_knowledge = AsyncMock()
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": "model-1", "knowledge-bases": ["kb-1"]},  # Request kb-1
            resources=AgentResources(
                models=[ModelResource(model_id="model-1")],
                knowledge_bases=[],  # No KBs authorized
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        # Should not call retrieve_knowledge at all
        assert fake_api.retrieve_knowledge.call_count == 0

    @pytest.mark.asyncio
    async def test_rag_uses_authorized_rerank_model(self, runner, monkeypatch):
        """RAG retrieval invokes configured rerank model and uses its order."""
        fake_api = FakeAgentRunAPIProxy(
            models=[
                ModelResource(model_id="model-1"),
                ModelResource(model_id="rerank-1", model_type="rerank"),
            ],
            knowledge_bases=[KnowledgeBaseResource(kb_id="kb-1")],
        )
        fake_api.retrieve_knowledge = AsyncMock(
            return_value=[
                {"content": [{"type": "text", "text": "low relevance"}]},
                {"content": [{"type": "text", "text": "high relevance sentinel"}]},
            ]
        )
        fake_api.invoke_rerank = AsyncMock(
            return_value=[{"index": 1, "relevance_score": 0.99}]
        )
        captured_messages: list[Message] = []

        async def mock_stream(*args, **kwargs):
            captured_messages.extend(kwargs["messages"])
            yield MessageChunk(role="assistant", content="Response", is_final=True)

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={
                "model": "model-1",
                "knowledge-bases": ["kb-1"],
                "rerank-model": "rerank-1",
                "rerank-top-k": 1,
            },
            resources=AgentResources(
                models=[
                    ModelResource(model_id="model-1"),
                    ModelResource(model_id="rerank-1", model_type="rerank"),
                ],
                knowledge_bases=[KnowledgeBaseResource(kb_id="kb-1")],
            ),
            input_text="find sentinel",
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        fake_api.invoke_rerank.assert_awaited_once_with(
            rerank_model_uuid="rerank-1",
            query="find sentinel",
            documents=["low relevance", "high relevance sentinel"],
            top_k=1,
        )
        assert any(r.type == AgentRunResultType.RUN_COMPLETED for r in results)
        assert "high relevance sentinel" in captured_messages[-1].content
        assert "low relevance" not in captured_messages[-1].content

    @pytest.mark.asyncio
    async def test_streaming_failure_after_first_chunk_no_fallback(self, runner, monkeypatch):
        """Streaming: model fails after first chunk, should NOT fallback to next model.

        This verifies the critical rule: once first chunk is yielded, the model is committed
        and subsequent failures should result in controlled failure, not fallback.
        """
        fake_api = FakeAgentRunAPIProxy(
            models=[
                ModelResource(model_id="model-1"),
                ModelResource(model_id="model-2"),
            ],
        )

        # Model-1 yields first chunk then fails
        async def mock_stream_model1(*args, **kwargs):
            yield MessageChunk(role="assistant", content="First chunk", is_final=False)
            # Simulate failure after first chunk
            raise Exception("Model-1 failed after yielding first chunk")

        # Model-2 should NOT be called, but set up for verification
        async def mock_stream_model2(*args, **kwargs):
            yield MessageChunk(role="assistant", content="Model-2 response", is_final=True)

        # Track which model is called
        call_count = {"model-1": 0, "model-2": 0}

        async def mock_stream(llm_model_uuid, *args, **kwargs):
            call_count[llm_model_uuid] = call_count.get(llm_model_uuid, 0) + 1
            if llm_model_uuid == "model-1":
                async for chunk in mock_stream_model1(*args, **kwargs):
                    yield chunk
            else:
                async for chunk in mock_stream_model2(*args, **kwargs):
                    yield chunk

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": {"primary": "model-1", "fallbacks": ["model-2"]}},
            resources=AgentResources(
                models=[
                    ModelResource(model_id="model-1"),
                    ModelResource(model_id="model-2"),
                ]
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        # Should have exactly one call to model-1, zero calls to model-2
        assert call_count["model-1"] == 1
        assert call_count["model-2"] == 0

        # Should end with run.failed (not run.completed with model-2 content)
        failed = [r for r in results if r.type == AgentRunResultType.RUN_FAILED]
        assert len(failed) == 1
        assert "no fallback possible" in failed[0].data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_streaming_tool_loop_stops_at_iteration_limit(self, runner, monkeypatch):
        """Repeated tool requests stop with runner.tool_loop_limit."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
            tools=[ToolResource(tool_name="allowed_tool")],
        )
        fake_api.call_tool = AsyncMock(return_value={"result": "again"})
        stream_call_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal stream_call_count
            stream_call_count += 1
            yield MessageChunk(
                role="assistant",
                content="",
                is_final=True,
                tool_calls=[
                    ToolCall(
                        id=f"call-{stream_call_count}",
                        type="function",
                        function=FunctionCall(name="allowed_tool", arguments="{}"),
                    )
                ],
            )

        fake_api.invoke_llm_stream = mock_stream
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": "model-1"},
            resources=AgentResources(
                models=[ModelResource(model_id="model-1")],
                tools=[ToolResource(tool_name="allowed_tool")],
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        failed = [r for r in results if r.type == AgentRunResultType.RUN_FAILED]
        assert len(failed) == 1
        assert failed[0].data.get("code") == "runner.tool_loop_limit"
        assert fake_api.call_tool.await_count == 10
        assert stream_call_count == 11

    @pytest.mark.asyncio
    async def test_non_streaming_mode_uses_invoke_llm(self, runner, monkeypatch):
        """Non-streaming mode uses invoke_llm and yields message.completed."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
        )

        # Mock non-streaming invoke
        fake_api.invoke_llm = AsyncMock(
            return_value=Message(role="assistant", content="Non-streaming response")
        )

        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": "model-1", "streaming": False},
            resources=AgentResources(models=[ModelResource(model_id="model-1")]),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        # Should have called invoke_llm (not invoke_llm_stream)
        assert fake_api.invoke_llm.call_count == 1

        # Should have message.completed and run.completed
        completed = [r for r in results if r.type == AgentRunResultType.MESSAGE_COMPLETED]
        run_completed = [r for r in results if r.type == AgentRunResultType.RUN_COMPLETED]

        assert len(completed) == 1
        assert completed[0].data.get("message", {}).get("content") == "Non-streaming response"
        assert len(run_completed) == 1

    @pytest.mark.asyncio
    async def test_runtime_metadata_can_disable_default_streaming(self, runner, monkeypatch):
        """When config omits streaming, host adapter capability decides the mode."""
        fake_api = FakeAgentRunAPIProxy(
            models=[ModelResource(model_id="model-1")],
        )
        fake_api.invoke_llm = AsyncMock(
            return_value=Message(role="assistant", content="Adapter cannot stream")
        )
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": "model-1"},
            resources=AgentResources(models=[ModelResource(model_id="model-1")]),
            runtime_metadata={"streaming_supported": False},
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        fake_api.invoke_llm.assert_awaited_once()
        assert fake_api.invoke_llm_stream.call_count == 0
        completed = [r for r in results if r.type == AgentRunResultType.MESSAGE_COMPLETED]
        assert completed[0].data.get("message", {}).get("content") == "Adapter cannot stream"

    @pytest.mark.asyncio
    async def test_non_streaming_fallback(self, runner, monkeypatch):
        """Non-streaming: primary fails, fallback succeeds."""
        fake_api = FakeAgentRunAPIProxy(
            models=[
                ModelResource(model_id="model-1"),
                ModelResource(model_id="model-2"),
            ],
        )

        call_count = [0]

        async def mock_invoke_llm(llm_model_uuid, *args, **kwargs):
            call_count[0] += 1
            if llm_model_uuid == "model-1":
                raise Exception("Primary model error")
            return Message(role="assistant", content="Fallback response")

        fake_api.invoke_llm = mock_invoke_llm
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": {"primary": "model-1", "fallbacks": ["model-2"]}, "streaming": False},
            resources=AgentResources(
                models=[
                    ModelResource(model_id="model-1"),
                    ModelResource(model_id="model-2"),
                ]
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        # Should have called both models
        assert call_count[0] == 2

        # Should have successful completion from fallback
        completed = [r for r in results if r.type == AgentRunResultType.MESSAGE_COMPLETED]
        assert len(completed) == 1
        assert completed[0].data.get("message", {}).get("content") == "Fallback response"

    @pytest.mark.asyncio
    async def test_non_streaming_tool_loop_uses_committed_fallback_model(self, runner, monkeypatch):
        """Tool loop continues with the model that succeeded during fallback."""
        fake_api = FakeAgentRunAPIProxy(
            models=[
                ModelResource(model_id="model-1"),
                ModelResource(model_id="model-2"),
            ],
            tools=[ToolResource(tool_name="allowed_tool")],
        )
        fake_api.call_tool = AsyncMock(return_value={"result": "success"})

        calls: list[tuple[str, list[Any]]] = []

        async def mock_invoke_llm(llm_model_uuid, *args, **kwargs):
            calls.append((llm_model_uuid, kwargs.get("funcs", [])))
            if llm_model_uuid == "model-1":
                raise Exception("Primary model error")
            if len(calls) == 2:
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            type="function",
                            function=FunctionCall(name="allowed_tool", arguments="{}"),
                        )
                    ],
                )
            return Message(role="assistant", content="Done")

        fake_api.invoke_llm = mock_invoke_llm
        monkeypatch.setattr(runner, "get_run_api", lambda ctx: fake_api)

        ctx = make_context(
            config={"model": {"primary": "model-1", "fallbacks": ["model-2"]}, "streaming": False},
            resources=AgentResources(
                models=[
                    ModelResource(model_id="model-1"),
                    ModelResource(model_id="model-2"),
                ],
                tools=[ToolResource(tool_name="allowed_tool")],
            ),
        )

        results = []
        async for result in runner.run(ctx):
            results.append(result)

        assert [model_id for model_id, _ in calls] == ["model-1", "model-2", "model-2"]
        assert calls[-1][1], "tool loop should keep tools available for multi-step calls"
        completed = [r for r in results if r.type == AgentRunResultType.MESSAGE_COMPLETED]
        assert completed[0].data.get("message", {}).get("content") == "Done"
