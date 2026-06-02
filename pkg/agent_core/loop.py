"""Pi-style agent loop implemented with LangBot-native adapters."""

from __future__ import annotations

import typing

from langbot_plugin.api.entities.builtin.provider.message import Message
from langbot_plugin.api.entities.builtin.resource.tool import LLMTool

from pkg.model_calling import ModelCallError

from .langbot import LangBotModelAdapter, LangBotToolExecutor
from .types import AgentLoopEvent, ModelTurnEventType, ModelTurnResult


class AgentLoop:
    """Turn/message/tool lifecycle loop for local-agent.

    The loop is deliberately independent from AgentRunResult so the runner can
    keep LangBot protocol adaptation at the boundary.
    """

    def __init__(
        self,
        *,
        model_adapter: LangBotModelAdapter,
        tool_executor: LangBotToolExecutor,
        model_ids: list[str],
        messages: list[Message],
        tools: list[LLMTool],
        streaming: bool,
        max_tool_iterations: int,
    ):
        self.model_adapter = model_adapter
        self.tool_executor = tool_executor
        self.model_ids = list(model_ids)
        self.messages = [message.model_copy(deep=True) for message in messages]
        self.tools = list(tools)
        self.streaming = streaming
        self.max_tool_iterations = max_tool_iterations

    async def run(self) -> typing.AsyncGenerator[AgentLoopEvent, None]:
        if self.streaming:
            async for event in self._run_streaming():
                yield event
            return

        async for event in self._run_non_streaming():
            yield event

    async def _run_streaming(self) -> typing.AsyncGenerator[AgentLoopEvent, None]:
        yield AgentLoopEvent.agent_start()

        committed_model_id: str | None = None
        pending_tool_calls = None
        visible_content_prefix = ""
        tool_iterations = 0
        last_assistant_message: Message | None = None

        while True:
            if pending_tool_calls is not None:
                if tool_iterations >= self.max_tool_iterations:
                    yield AgentLoopEvent.run_failed(
                        f"Tool call iteration limit reached ({self.max_tool_iterations})",
                        code="runner.tool_loop_limit",
                    )
                    return

                tool_results: list[Message] = []
                tool_iterations += 1
                for tool_call in pending_tool_calls:
                    prepared = self.tool_executor.prepare(tool_call)
                    yield AgentLoopEvent.tool_execution_start(tool_call, prepared.parameters)
                    outcome = await self.tool_executor.execute(prepared)
                    yield AgentLoopEvent.tool_execution_end(outcome)
                    if outcome.message is not None:
                        self.messages.append(outcome.message)
                        tool_results.append(outcome.message)
                        yield AgentLoopEvent.message_start(outcome.message)
                        yield AgentLoopEvent.message_end(outcome.message)

                if last_assistant_message is not None:
                    yield AgentLoopEvent.turn_end(last_assistant_message, tool_results)

            if committed_model_id is None:
                turn_model_ids = self.model_ids
            else:
                turn_model_ids = [committed_model_id]

            yield AgentLoopEvent.turn_start()
            yield AgentLoopEvent.message_start(Message(role="assistant", content=""))

            try:
                model_turn: ModelTurnResult | None = None
                async for model_event in self.model_adapter.stream_turn(
                    model_ids=turn_model_ids,
                    messages=self.messages,
                    tools=self.tools,
                    visible_prefix=visible_content_prefix,
                ):
                    if model_event.type == ModelTurnEventType.MESSAGE_DELTA and model_event.chunk is not None:
                        yield AgentLoopEvent.message_update(model_event.chunk)
                    elif model_event.type == ModelTurnEventType.MESSAGE_END and model_event.result is not None:
                        model_turn = model_event.result
            except ModelCallError as e:
                yield AgentLoopEvent.run_failed(str(e), code="runner.llm_error", retryable=e.retryable)
                return
            except Exception as e:
                yield AgentLoopEvent.run_failed(str(e), code="runner.error")
                return

            if model_turn is None:
                yield AgentLoopEvent.run_failed("Model stream ended without a final message", code="runner.llm_error")
                return

            committed_model_id = model_turn.committed_model_id or committed_model_id
            self.messages.append(model_turn.message)
            last_assistant_message = model_turn.message
            yield AgentLoopEvent.message_end(model_turn.message)

            pending_tool_calls = model_turn.tool_calls
            if not pending_tool_calls:
                yield AgentLoopEvent.turn_end(model_turn.message, [])
                yield AgentLoopEvent.agent_end(self.messages)
                return

            visible_content_prefix += model_turn.visible_content

    async def _run_non_streaming(self) -> typing.AsyncGenerator[AgentLoopEvent, None]:
        yield AgentLoopEvent.agent_start()

        committed_model_id: str | None = None
        pending_tool_calls = None
        tool_iterations = 0
        last_assistant_message: Message | None = None

        while True:
            if pending_tool_calls is not None:
                if tool_iterations >= self.max_tool_iterations:
                    yield AgentLoopEvent.run_failed(
                        f"Tool call iteration limit reached ({self.max_tool_iterations})",
                        code="runner.tool_loop_limit",
                    )
                    return

                tool_results: list[Message] = []
                tool_iterations += 1
                for tool_call in pending_tool_calls:
                    prepared = self.tool_executor.prepare(tool_call)
                    yield AgentLoopEvent.tool_execution_start(tool_call, prepared.parameters)
                    outcome = await self.tool_executor.execute(prepared)
                    yield AgentLoopEvent.tool_execution_end(outcome)
                    if outcome.message is not None:
                        self.messages.append(outcome.message)
                        tool_results.append(outcome.message)
                        yield AgentLoopEvent.message_start(outcome.message)
                        yield AgentLoopEvent.message_end(outcome.message)

                if last_assistant_message is not None:
                    yield AgentLoopEvent.turn_end(last_assistant_message, tool_results)

            yield AgentLoopEvent.turn_start()
            yield AgentLoopEvent.message_start(Message(role="assistant", content=""))

            try:
                if committed_model_id is None:
                    model_turn = await self.model_adapter.invoke_turn(
                        model_ids=self.model_ids,
                        messages=self.messages,
                        tools=self.tools,
                    )
                else:
                    model_turn = await self.model_adapter.invoke_committed_turn(
                        committed_model_id=committed_model_id,
                        messages=self.messages,
                        tools=self.tools,
                    )
            except ModelCallError as e:
                yield AgentLoopEvent.run_failed(str(e), code="runner.llm_error", retryable=e.retryable)
                return
            except Exception as e:
                yield AgentLoopEvent.run_failed(str(e), code="runner.error")
                return

            committed_model_id = model_turn.committed_model_id or committed_model_id
            self.messages.append(model_turn.message)
            last_assistant_message = model_turn.message
            yield AgentLoopEvent.message_end(model_turn.message)

            pending_tool_calls = model_turn.tool_calls
            if not pending_tool_calls:
                yield AgentLoopEvent.turn_end(model_turn.message, [])
                yield AgentLoopEvent.agent_end(self.messages)
                return
