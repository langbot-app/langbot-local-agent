"""Tests for the LangBot-native agent core loop."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock

import pytest
from langbot_plugin.api.entities.builtin.provider.message import FunctionCall, Message, MessageChunk, ToolCall

from pkg.agent_core import (
    AgentLoop,
    AgentLoopEventType,
    AgentLoopHooks,
    LangBotModelAdapter,
    LangBotToolExecutor,
    ModelTurnEvent,
    ModelTurnResult,
    ToolCallRequest,
)
from pkg.model_calling import INTERNAL_ARTIFACT_READ_TOOL_NAME, ModelCallError


async def _collect_events(loop: AgentLoop):
    return [event async for event in loop.run()]


@pytest.mark.asyncio
async def test_streaming_loop_emits_turn_message_and_tool_events():
    """The core loop exposes a Pi-style lifecycle independent from AgentRunResult."""

    class FakeAPI:
        def __init__(self):
            self.call_tool = AsyncMock(return_value={"value": "tool-result"})
            self.stream_calls = 0

        def invoke_llm_stream(self, *args, **kwargs):
            self.stream_calls += 1

            async def stream():
                if self.stream_calls == 1:
                    yield MessageChunk(
                        role="assistant",
                        content="Checking",
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
                    yield MessageChunk(role="assistant", content="Done", is_final=True)

            return stream()

    api = FakeAPI()
    loop = AgentLoop(
        model_adapter=LangBotModelAdapter(api),
        tool_executor=LangBotToolExecutor(api, {"allowed_tool"}),
        model_ids=["model-1"],
        messages=[Message(role="user", content="Use the tool")],
        tools=[],
        streaming=True,
        max_tool_iterations=10,
    )

    events = [event async for event in loop.run()]
    event_types = [event.type for event in events]

    assert event_types == [
        AgentLoopEventType.AGENT_START,
        AgentLoopEventType.TURN_START,
        AgentLoopEventType.MESSAGE_START,
        AgentLoopEventType.MESSAGE_UPDATE,
        AgentLoopEventType.MESSAGE_END,
        AgentLoopEventType.TOOL_EXECUTION_START,
        AgentLoopEventType.TOOL_EXECUTION_END,
        AgentLoopEventType.MESSAGE_START,
        AgentLoopEventType.MESSAGE_END,
        AgentLoopEventType.TURN_END,
        AgentLoopEventType.TURN_START,
        AgentLoopEventType.MESSAGE_START,
        AgentLoopEventType.MESSAGE_UPDATE,
        AgentLoopEventType.MESSAGE_END,
        AgentLoopEventType.TURN_END,
        AgentLoopEventType.AGENT_END,
    ]

    deltas = [
        event.chunk.content
        for event in events
        if event.type == AgentLoopEventType.MESSAGE_UPDATE and event.chunk is not None
    ]
    assert deltas == ["Checking", "CheckingDone"]
    api.call_tool.assert_awaited_once_with(tool_name="allowed_tool", parameters={"arg": "value"})


@pytest.mark.asyncio
async def test_artifact_read_tool_uses_host_artifact_api_not_tool_call():
    class FakeAPI:
        def __init__(self):
            self.call_tool = AsyncMock()
            self.artifact_read = AsyncMock(
                return_value={
                    "artifact_id": "artifact-1",
                    "mime_type": "text/plain; charset=utf-8",
                    "size_bytes": 11,
                    "offset": 2,
                    "length": 5,
                    "has_more": True,
                    "content_base64": base64.b64encode("hello".encode("utf-8")).decode("ascii"),
                }
            )

    api = FakeAPI()
    executor = LangBotToolExecutor(
        api,
        {INTERNAL_ARTIFACT_READ_TOOL_NAME},
        artifact_read_available=True,
    )
    prepared = executor.prepare(
        ToolCallRequest(
            id="call-artifact",
            name=INTERNAL_ARTIFACT_READ_TOOL_NAME,
            arguments='{"artifact_id": "artifact-1", "offset": 2, "limit": 5}',
        )
    )

    outcome = await executor.execute(prepared)

    assert outcome.error is None
    assert outcome.result["artifact_id"] == "artifact-1"
    assert outcome.result["text"] == "hello"
    assert outcome.result["has_more"] is True
    api.artifact_read.assert_awaited_once_with(artifact_id="artifact-1", offset=2, limit=5)
    api.call_tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_loop_executes_same_batch_tools_in_parallel_and_preserves_source_order(streaming):
    class BatchModelAdapter:
        def __init__(self):
            self.turns = 0
            self.messages_by_turn: list[list[Message]] = []
            self.tool_calls = [
                ToolCallRequest(id="call-slow", name="slow_tool", arguments='{"value": "slow"}'),
                ToolCallRequest(id="call-fast", name="fast_tool", arguments='{"value": "fast"}'),
            ]

        def _next_turn(self, messages: list[Message]) -> ModelTurnResult:
            self.messages_by_turn.append(list(messages))
            self.turns += 1
            if self.turns == 1:
                return ModelTurnResult(
                    message=Message(
                        role="assistant",
                        content="Need tools",
                        tool_calls=[tool_call.to_tool_call() for tool_call in self.tool_calls],
                    ),
                    tool_calls=self.tool_calls,
                    committed_model_id="model-1",
                    visible_content="Need tools",
                )

            return ModelTurnResult(
                message=Message(role="assistant", content="Done"),
                tool_calls=[],
                committed_model_id="model-1",
                visible_content="Done",
            )

        async def invoke_turn(self, *, model_ids, messages, tools):
            return self._next_turn(messages)

        async def invoke_committed_turn(self, *, committed_model_id, messages, tools):
            return self._next_turn(messages)

        async def stream_turn(self, *, model_ids, messages, tools, visible_prefix=""):
            result = self._next_turn(messages)
            yield ModelTurnEvent.message_delta(
                MessageChunk(role="assistant", content=result.visible_content, is_final=True)
            )
            yield ModelTurnEvent.message_end(result)

    class ConcurrentToolAPI:
        def __init__(self):
            self.active_calls = 0
            self.max_active_calls = 0
            self.both_started = asyncio.Event()

        async def call_tool(self, *, tool_name, parameters):
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            if self.active_calls == 2:
                self.both_started.set()

            try:
                await self.both_started.wait()
                if tool_name == "slow_tool":
                    await asyncio.sleep(0.05)
                else:
                    await asyncio.sleep(0.001)
                return {"tool": tool_name, "value": parameters["value"]}
            finally:
                self.active_calls -= 1

    model_adapter = BatchModelAdapter()
    api = ConcurrentToolAPI()
    loop = AgentLoop(
        model_adapter=model_adapter,
        tool_executor=LangBotToolExecutor(api, {"slow_tool", "fast_tool"}),
        model_ids=["model-1"],
        messages=[Message(role="user", content="Use both tools")],
        tools=[],
        streaming=streaming,
        max_tool_iterations=10,
    )

    events = await asyncio.wait_for(_collect_events(loop), timeout=1)

    assert api.max_active_calls == 2

    started_tool_call_ids = [
        event.tool_call_id for event in events if event.type == AgentLoopEventType.TOOL_EXECUTION_START
    ]
    assert started_tool_call_ids == ["call-slow", "call-fast"]

    ended_tool_call_ids = [
        event.tool_call_id for event in events if event.type == AgentLoopEventType.TOOL_EXECUTION_END
    ]
    assert ended_tool_call_ids == ["call-fast", "call-slow"]

    result_message_ids = [
        event.message.tool_call_id
        for event in events
        if event.type == AgentLoopEventType.MESSAGE_END
        and event.message is not None
        and event.message.role == "tool"
    ]
    assert result_message_ids == ["call-slow", "call-fast"]

    tool_result_turns = [event for event in events if event.type == AgentLoopEventType.TURN_END and event.tool_results]
    assert len(tool_result_turns) == 1
    assert [message.tool_call_id for message in tool_result_turns[0].tool_results] == ["call-slow", "call-fast"]

    second_model_turn_tool_results = [
        message.tool_call_id for message in model_adapter.messages_by_turn[1] if message.role == "tool"
    ]
    assert second_model_turn_tool_results == ["call-slow", "call-fast"]


@pytest.mark.asyncio
async def test_loop_runs_pi_style_hooks_around_turns_and_tools():
    class HookedModelAdapter:
        def __init__(self):
            self.turns = 0

        async def invoke_turn(self, *, model_ids, messages, tools):
            self.turns += 1
            return ModelTurnResult(
                message=Message(
                    role="assistant",
                    content="Need tool",
                    tool_calls=[
                        ToolCallRequest(id="call-1", name="allowed_tool", arguments='{"arg": "value"}').to_tool_call()
                    ],
                ),
                tool_calls=[ToolCallRequest(id="call-1", name="allowed_tool", arguments='{"arg": "value"}')],
                committed_model_id="model-1",
                visible_content="Need tool",
            )

        async def invoke_committed_turn(self, *, committed_model_id, messages, tools):
            self.turns += 1
            return ModelTurnResult(
                message=Message(role="assistant", content="Done"),
                tool_calls=[],
                committed_model_id=committed_model_id,
                visible_content="Done",
            )

    class HookAPI:
        async def call_tool(self, *, tool_name, parameters):
            return {"ok": True, "arg": parameters["arg"]}

    class TrackingHooks(AgentLoopHooks):
        def __init__(self):
            self.prepare_model_turn_calls = 0
            self.before_tool_ids: list[str] = []
            self.after_tool_ids: list[str] = []
            self.prepare_next_turn_tool_ids: list[str] = []
            self.stop_checks = 0

        async def prepare_model_turn(self, messages: list[Message]) -> list[Message]:
            self.prepare_model_turn_calls += 1
            return [message.model_copy(deep=True) for message in messages]

        async def before_tool_call(self, prepared):
            self.before_tool_ids.append(prepared.request.id)
            return prepared

        async def after_tool_call(self, outcome):
            self.after_tool_ids.append(outcome.request.id)
            return outcome

        async def should_stop_after_turn(self, result, messages):
            self.stop_checks += 1
            return False

        async def prepare_next_turn(self, messages, result, tool_results):
            self.prepare_next_turn_tool_ids = [message.tool_call_id for message in tool_results]
            return [message.model_copy(deep=True) for message in messages]

    hooks = TrackingHooks()
    loop = AgentLoop(
        model_adapter=HookedModelAdapter(),
        tool_executor=LangBotToolExecutor(HookAPI(), {"allowed_tool"}),
        model_ids=["model-1"],
        messages=[Message(role="user", content="Use tool")],
        tools=[],
        streaming=False,
        max_tool_iterations=10,
        hooks=hooks,
    )

    events = await _collect_events(loop)

    assert AgentLoopEventType.RUN_FAILED not in [event.type for event in events]
    assert hooks.prepare_model_turn_calls == 2
    assert hooks.before_tool_ids == ["call-1"]
    assert hooks.after_tool_ids == ["call-1"]
    assert hooks.prepare_next_turn_tool_ids == ["call-1"]
    assert hooks.stop_checks == 2


@pytest.mark.asyncio
async def test_non_streaming_loop_recovers_context_overflow_before_visible_failure():
    class OverflowModelAdapter:
        def __init__(self):
            self.calls: list[list[Message]] = []

        async def invoke_turn(self, *, model_ids, messages, tools):
            self.calls.append([message.model_copy(deep=True) for message in messages])
            if len(self.calls) == 1:
                raise ModelCallError("context length exceeded", retryable=True)
            return ModelTurnResult(
                message=Message(role="assistant", content="Recovered"),
                tool_calls=[],
                committed_model_id="model-1",
                visible_content="Recovered",
            )

        async def invoke_committed_turn(self, *, committed_model_id, messages, tools):
            raise AssertionError("committed turn should not be used")

    class RecoveryHooks(AgentLoopHooks):
        def __init__(self):
            self.recover_calls = 0

        async def recover_context_overflow(self, messages: list[Message], error: Exception) -> list[Message] | None:
            self.recover_calls += 1
            return [Message(role="user", content="short retry context")]

    model_adapter = OverflowModelAdapter()
    hooks = RecoveryHooks()
    loop = AgentLoop(
        model_adapter=model_adapter,
        tool_executor=LangBotToolExecutor(AsyncMock(), set()),
        model_ids=["model-1"],
        messages=[Message(role="user", content="long context")],
        tools=[],
        streaming=False,
        max_tool_iterations=10,
        hooks=hooks,
    )

    events = await _collect_events(loop)

    assert hooks.recover_calls == 1
    assert len(model_adapter.calls) == 2
    assert model_adapter.calls[1][0].content == "short retry context"
    assert [event.type for event in events].count(AgentLoopEventType.TURN_START) == 1
    assert AgentLoopEventType.RUN_FAILED not in [event.type for event in events]
