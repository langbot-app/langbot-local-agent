"""Pi-style agent loop implemented with LangBot-native adapters."""

from __future__ import annotations

import asyncio
import typing

from langbot_plugin.api.entities.builtin.provider.message import Message
from langbot_plugin.api.entities.builtin.resource.tool import LLMTool

from pkg.model_calling import ModelCallError

from .langbot import LangBotModelAdapter, LangBotToolExecutor
from .types import (
    AgentLoopEvent,
    AgentLoopHooks,
    ModelTurnEventType,
    ModelTurnResult,
    PreparedToolCall,
    ToolExecutionMode,
    ToolExecutionOutcome,
)

STATEFUL_TOOL_NAMES = frozenset(
    {
        "activate",
        "register_skill",
        "exec",
        "sandbox_exec",
        "write",
        "edit",
    }
)


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
        tool_execution_mode: ToolExecutionMode | str = ToolExecutionMode.AUTO,
        hooks: AgentLoopHooks | None = None,
    ):
        self.model_adapter = model_adapter
        self.tool_executor = tool_executor
        self.model_ids = list(model_ids)
        self.messages = [message.model_copy(deep=True) for message in messages]
        self.tools = list(tools)
        self.streaming = streaming
        self.max_tool_iterations = max_tool_iterations
        self.tool_execution_mode = self._normalize_tool_execution_mode(tool_execution_mode)
        self.hooks = hooks or AgentLoopHooks()

    @staticmethod
    def _normalize_tool_execution_mode(mode: ToolExecutionMode | str) -> ToolExecutionMode:
        if isinstance(mode, ToolExecutionMode):
            return mode
        try:
            return ToolExecutionMode(str(mode))
        except ValueError:
            return ToolExecutionMode.AUTO

    async def run(self) -> typing.AsyncGenerator[AgentLoopEvent, None]:
        if self.streaming:
            async for event in self._run_streaming():
                yield event
            return

        async for event in self._run_non_streaming():
            yield event

    async def _execute_prepared_tool_call(
        self,
        index: int,
        prepared: PreparedToolCall,
    ) -> tuple[int, ToolExecutionOutcome]:
        outcome = await self.tool_executor.execute(prepared)
        outcome = await self.hooks.after_tool_call(outcome)
        return index, outcome

    async def _prepare_model_turn(self) -> None:
        messages = await self.hooks.prepare_model_turn(self.messages)
        self.messages = [message.model_copy(deep=True) for message in messages]

    async def _recover_context_overflow(self, error: ModelCallError) -> bool:
        messages = await self.hooks.recover_context_overflow(self.messages, error)
        if messages is None:
            return False
        self.messages = [message.model_copy(deep=True) for message in messages]
        return True

    def _should_execute_serially(self, prepared_calls: list[PreparedToolCall]) -> bool:
        if self.tool_execution_mode == ToolExecutionMode.SERIAL:
            return True
        if self.tool_execution_mode == ToolExecutionMode.PARALLEL:
            return False
        return any(prepared.request.name in STATEFUL_TOOL_NAMES for prepared in prepared_calls)

    async def _execute_prepared_tool_calls_parallel(
        self,
        prepared_calls: list[PreparedToolCall],
    ) -> typing.AsyncGenerator[tuple[int, ToolExecutionOutcome], None]:
        tasks = [
            asyncio.create_task(self._execute_prepared_tool_call(index, prepared))
            for index, prepared in enumerate(prepared_calls)
        ]

        try:
            for completed in asyncio.as_completed(tasks):
                yield await completed
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _execute_tool_batch(
        self,
        pending_tool_calls: typing.Iterable[typing.Any],
        last_model_turn: ModelTurnResult | None,
    ) -> typing.AsyncGenerator[AgentLoopEvent, None]:
        prepared_calls: list[PreparedToolCall] = []
        for tool_call in pending_tool_calls:
            prepared = self.tool_executor.prepare(tool_call)
            prepared = await self.hooks.before_tool_call(prepared)
            prepared_calls.append(prepared)

        outcomes_by_source_order: list[ToolExecutionOutcome | None] = [None] * len(prepared_calls)
        if self._should_execute_serially(prepared_calls):
            for index, prepared in enumerate(prepared_calls):
                yield AgentLoopEvent.tool_execution_start(prepared.request, prepared.parameters)
                _, outcome = await self._execute_prepared_tool_call(index, prepared)
                outcomes_by_source_order[index] = outcome
                yield AgentLoopEvent.tool_execution_end(outcome)
        else:
            for prepared in prepared_calls:
                yield AgentLoopEvent.tool_execution_start(prepared.request, prepared.parameters)
            async for index, outcome in self._execute_prepared_tool_calls_parallel(prepared_calls):
                outcomes_by_source_order[index] = outcome
                yield AgentLoopEvent.tool_execution_end(outcome)

        tool_results: list[Message] = []
        for outcome in outcomes_by_source_order:
            if outcome is None or outcome.message is None:
                continue
            self.messages.append(outcome.message)
            tool_results.append(outcome.message)
            yield AgentLoopEvent.message_start(outcome.message)
            yield AgentLoopEvent.message_end(outcome.message)

        if last_model_turn is not None:
            yield AgentLoopEvent.turn_end(last_model_turn.message, tool_results)
            messages = await self.hooks.prepare_next_turn(self.messages, last_model_turn, tool_results)
            self.messages = [message.model_copy(deep=True) for message in messages]

    async def _run_streaming(self) -> typing.AsyncGenerator[AgentLoopEvent, None]:
        yield AgentLoopEvent.agent_start()

        committed_model_id: str | None = None
        pending_tool_calls = None
        visible_content_prefix = ""
        tool_iterations = 0
        last_model_turn: ModelTurnResult | None = None

        while True:
            if pending_tool_calls is not None:
                if tool_iterations >= self.max_tool_iterations:
                    yield AgentLoopEvent.run_failed(
                        f"Tool call iteration limit reached ({self.max_tool_iterations})",
                        code="runner.tool_loop_limit",
                    )
                    return

                tool_iterations += 1
                async for event in self._execute_tool_batch(pending_tool_calls, last_model_turn):
                    yield event

            if committed_model_id is None:
                turn_model_ids = self.model_ids
            else:
                turn_model_ids = [committed_model_id]

            yield AgentLoopEvent.turn_start()
            yield AgentLoopEvent.message_start(Message(role="assistant", content=""))

            context_retry_used = False
            while True:
                stream_started = False
                try:
                    await self._prepare_model_turn()
                    model_turn: ModelTurnResult | None = None
                    async for model_event in self.model_adapter.stream_turn(
                        model_ids=turn_model_ids,
                        messages=self.messages,
                        tools=self.tools,
                        visible_prefix=visible_content_prefix,
                    ):
                        if model_event.type == ModelTurnEventType.MESSAGE_DELTA and model_event.chunk is not None:
                            stream_started = True
                            yield AgentLoopEvent.message_update(model_event.chunk)
                        elif model_event.type == ModelTurnEventType.MESSAGE_END and model_event.result is not None:
                            model_turn = model_event.result
                    break
                except ModelCallError as e:
                    if not stream_started and not context_retry_used and await self._recover_context_overflow(e):
                        context_retry_used = True
                        continue
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
            last_model_turn = model_turn
            yield AgentLoopEvent.message_end(model_turn.message)

            if await self.hooks.should_stop_after_turn(model_turn, self.messages):
                yield AgentLoopEvent.turn_end(model_turn.message, [])
                yield AgentLoopEvent.agent_end(self.messages)
                return

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
        last_model_turn: ModelTurnResult | None = None

        while True:
            if pending_tool_calls is not None:
                if tool_iterations >= self.max_tool_iterations:
                    yield AgentLoopEvent.run_failed(
                        f"Tool call iteration limit reached ({self.max_tool_iterations})",
                        code="runner.tool_loop_limit",
                    )
                    return

                tool_iterations += 1
                async for event in self._execute_tool_batch(pending_tool_calls, last_model_turn):
                    yield event

            yield AgentLoopEvent.turn_start()
            yield AgentLoopEvent.message_start(Message(role="assistant", content=""))

            context_retry_used = False
            while True:
                try:
                    await self._prepare_model_turn()
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
                    break
                except ModelCallError as e:
                    if not context_retry_used and await self._recover_context_overflow(e):
                        context_retry_used = True
                        continue
                    yield AgentLoopEvent.run_failed(str(e), code="runner.llm_error", retryable=e.retryable)
                    return
                except Exception as e:
                    yield AgentLoopEvent.run_failed(str(e), code="runner.error")
                    return

            committed_model_id = model_turn.committed_model_id or committed_model_id
            self.messages.append(model_turn.message)
            last_model_turn = model_turn
            yield AgentLoopEvent.message_end(model_turn.message)

            if await self.hooks.should_stop_after_turn(model_turn, self.messages):
                yield AgentLoopEvent.turn_end(model_turn.message, [])
                yield AgentLoopEvent.agent_end(self.messages)
                return

            pending_tool_calls = model_turn.tool_calls
            if not pending_tool_calls:
                yield AgentLoopEvent.turn_end(model_turn.message, [])
                yield AgentLoopEvent.agent_end(self.messages)
                return
