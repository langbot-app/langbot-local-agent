"""LangBot Host adapters used by the local-agent loop."""

from __future__ import annotations

import json
import typing

from langbot_plugin.api.entities.builtin.provider.message import Message, MessageChunk
from langbot_plugin.api.entities.builtin.resource.tool import LLMTool
from langbot_plugin.api.proxies.agent_run_api import AgentRunAPIProxy, PermissionDeniedError

from pkg.config import DEFAULT_MAX_TOOL_RESULT_CHARS
from pkg.context_pipeline import ContextBudget, ContextCompactor
from pkg.model_calling import (
    ModelCallError,
    StreamingModelCaller,
    build_tool_call_message,
    invoke_with_fallback,
)

from .types import (
    AgentLoopHooks,
    ModelTurnEvent,
    ModelTurnResult,
    PreparedToolCall,
    ToolCallRequest,
    ToolExecutionOutcome,
)


def build_assistant_message(content: str, tool_calls: list[ToolCallRequest]) -> Message:
    """Build an assistant message that preserves provider tool call IDs."""
    return Message(
        role="assistant",
        content=content,
        tool_calls=[tool_call.to_tool_call() for tool_call in tool_calls] if tool_calls else None,
    )


def _prefix_chunk_content(chunk: MessageChunk, prefix: str) -> MessageChunk:
    if not prefix:
        return chunk

    copied = chunk.model_copy(deep=True)
    if isinstance(copied.content, str):
        copied.content = prefix + copied.content
    elif isinstance(copied.content, list):
        for content in copied.content:
            if content.type == "text" and isinstance(content.text, str):
                content.text = prefix + content.text
                break
    return copied


class LangBotModelAdapter:
    """Model invocation adapter that keeps all model access behind Host APIs."""

    def __init__(self, api: AgentRunAPIProxy):
        self.api = api

    async def stream_turn(
        self,
        *,
        model_ids: list[str],
        messages: list[Message],
        tools: list[LLMTool],
        visible_prefix: str = "",
    ) -> typing.AsyncGenerator[ModelTurnEvent, None]:
        caller = StreamingModelCaller(
            api=self.api,
            model_ids=model_ids,
            messages=messages,
            tools=tools,
        )

        async for chunk, _is_delta in caller.stream():
            if chunk.content:
                yield ModelTurnEvent.message_delta(_prefix_chunk_content(chunk, visible_prefix))

        tool_calls = [ToolCallRequest.from_raw(tool_call) for tool_call in caller.get_tool_calls()]
        content = caller.get_accumulated_content()
        yield ModelTurnEvent.message_end(
            ModelTurnResult(
                message=build_assistant_message(content, tool_calls),
                tool_calls=tool_calls,
                committed_model_id=caller.get_committed_model_id(),
                visible_content=content,
            )
        )

    async def invoke_turn(
        self,
        *,
        model_ids: list[str],
        messages: list[Message],
        tools: list[LLMTool],
    ) -> ModelTurnResult:
        response, committed_model_id = await invoke_with_fallback(
            api=self.api,
            model_ids=model_ids,
            messages=messages,
            tools=tools,
        )
        tool_calls = [ToolCallRequest.from_raw(tool_call) for tool_call in response.tool_calls or []]
        return ModelTurnResult(
            message=response,
            tool_calls=tool_calls,
            committed_model_id=committed_model_id,
            visible_content=response.content if isinstance(response.content, str) else "",
        )

    async def invoke_committed_turn(
        self,
        *,
        committed_model_id: str,
        messages: list[Message],
        tools: list[LLMTool],
    ) -> ModelTurnResult:
        try:
            response = await self.api.invoke_llm(
                llm_model_uuid=committed_model_id,
                messages=messages,
                funcs=tools,
            )
        except Exception as e:
            raise ModelCallError(f"LLM call in tool loop failed: {e}", retryable=True) from e

        tool_calls = [ToolCallRequest.from_raw(tool_call) for tool_call in response.tool_calls or []]
        return ModelTurnResult(
            message=response,
            tool_calls=tool_calls,
            committed_model_id=committed_model_id,
            visible_content=response.content if isinstance(response.content, str) else "",
        )


class LangBotToolExecutor:
    """Tool preflight, execution, and model-result adaptation."""

    def __init__(
        self,
        api: AgentRunAPIProxy,
        allowed_tools: set[str],
        max_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
    ):
        self.api = api
        self.allowed_tools = allowed_tools
        self.max_result_chars = max_result_chars

    def prepare(self, request: ToolCallRequest) -> PreparedToolCall:
        parameters: dict[str, typing.Any] = {}
        if request.arguments:
            try:
                parsed = json.loads(request.arguments)
                if isinstance(parsed, dict):
                    parameters = parsed
                else:
                    return PreparedToolCall(
                        request=request,
                        parameters={},
                        error=f"Invalid JSON arguments for tool '{request.name}': expected object",
                    )
            except json.JSONDecodeError as e:
                return PreparedToolCall(
                    request=request,
                    parameters={},
                    error=f"Invalid JSON arguments for tool '{request.name}': {e}",
                )

        if request.name not in self.allowed_tools:
            return PreparedToolCall(
                request=request,
                parameters=parameters,
                error=f"Tool '{request.name}' is not authorized. Allowed tools: {list(self.allowed_tools)}",
            )

        return PreparedToolCall(request=request, parameters=parameters)

    async def execute(self, prepared: PreparedToolCall) -> ToolExecutionOutcome:
        if prepared.error is not None:
            return self.finalize(prepared, result=None, error=prepared.error)

        try:
            result = await self.api.call_tool(
                tool_name=prepared.request.name,
                parameters=prepared.parameters,
            )
            return self.finalize(prepared, result=result, error=None)
        except PermissionDeniedError as e:
            return self.finalize(prepared, result=None, error=str(e))
        except Exception as e:
            return self.finalize(prepared, result=None, error=f"Tool execution failed: {e}")

    def finalize(
        self,
        prepared: PreparedToolCall,
        *,
        result: typing.Any,
        error: str | None,
    ) -> ToolExecutionOutcome:
        message = build_tool_call_message(
            prepared.request.id,
            prepared.request.name,
            error if error is not None else result,
            is_error=error is not None,
            max_result_chars=self.max_result_chars,
        )
        return ToolExecutionOutcome(
            request=prepared.request,
            parameters=prepared.parameters,
            result=result,
            error=error,
            message=message,
        )


class LangBotContextHooks(AgentLoopHooks):
    """LangBot-specific loop hooks for Pi-style per-turn context management."""

    def __init__(self, budget: ContextBudget):
        self.budget = budget

    async def prepare_model_turn(self, messages: list[Message]) -> list[Message]:
        assembly = ContextCompactor(self.budget).compact_messages(messages)
        return assembly.messages

    async def recover_context_overflow(self, messages: list[Message], error: Exception) -> list[Message] | None:
        is_overflow = error.is_context_overflow if isinstance(error, ModelCallError) else False
        if not is_overflow or not self.budget.enabled:
            return None

        assembly = ContextCompactor(self._overflow_retry_budget()).compact_messages(messages)
        if not assembly.compacted or assembly.tokens_after >= assembly.tokens_before:
            return None
        return assembly.messages

    def _overflow_retry_budget(self) -> ContextBudget:
        retry_input_tokens = max(1, (self.budget.input_tokens * 3) // 4)
        return ContextBudget(
            window_tokens=retry_input_tokens,
            reserve_tokens=0,
            keep_recent_tokens=min(self.budget.keep_recent_tokens, max(1, retry_input_tokens // 2)),
            summary_tokens=min(self.budget.summary_tokens, max(0, retry_input_tokens // 4)),
            history_fetch_limit=self.budget.history_fetch_limit,
        )
