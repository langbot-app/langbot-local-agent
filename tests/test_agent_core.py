"""Tests for the LangBot-native agent core loop."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langbot_plugin.api.entities.builtin.provider.message import FunctionCall, Message, MessageChunk, ToolCall

from pkg.agent_core import AgentLoop, AgentLoopEventType, LangBotModelAdapter, LangBotToolExecutor


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
