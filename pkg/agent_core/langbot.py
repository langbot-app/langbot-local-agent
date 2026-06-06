"""LangBot Host adapters used by the local-agent loop."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import typing
import uuid

from langbot_plugin.api.entities.builtin.provider.message import Message, MessageChunk
from langbot_plugin.api.entities.builtin.resource.tool import LLMTool
from langbot_plugin.api.proxies.agent_run_api import AgentRunAPIProxy, PermissionDeniedError

from pkg.config import DEFAULT_MAX_TOOL_RESULT_ARTIFACT_BYTES, DEFAULT_MAX_TOOL_RESULT_CHARS
from pkg.context_pipeline import ContextBudget, ContextCompactor
from pkg.model_calling import (
    INTERNAL_ARTIFACT_READ_TOOL_NAME,
    ModelCallError,
    StreamingModelCaller,
    build_tool_artifact_message,
    build_tool_call_message,
    build_tool_reference_message,
    extract_tool_result_references,
    invoke_with_fallback,
    serialize_tool_result_content,
)

from .types import (
    AgentLoopHooks,
    ModelTurnEvent,
    ModelTurnResult,
    PreparedToolCall,
    ToolCallRequest,
    ToolExecutionOutcome,
    ToolResultArtifact,
)

THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking_blocks(content: str) -> str:
    return THINK_BLOCK_RE.sub("", content).lstrip()


def build_assistant_message(content: str, tool_calls: list[ToolCallRequest]) -> Message:
    """Build an assistant message that preserves provider tool call IDs."""
    return Message(
        role="assistant",
        content=_strip_thinking_blocks(content) if tool_calls else content,
        tool_calls=[tool_call.to_tool_call() for tool_call in tool_calls] if tool_calls else None,
    )


def _context_message_for_tool_follow_up(message: Message, tool_calls: list[ToolCallRequest]) -> Message:
    if not tool_calls or not isinstance(message.content, str):
        return message

    copied = message.model_copy(deep=True)
    copied.content = _strip_thinking_blocks(message.content)
    return copied


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

    def __init__(self, api: AgentRunAPIProxy, *, remove_think: bool = False):
        self.api = api
        self.remove_think = remove_think

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
            remove_think=self.remove_think,
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
            remove_think=self.remove_think,
        )
        tool_calls = [ToolCallRequest.from_raw(tool_call) for tool_call in response.tool_calls or []]
        return ModelTurnResult(
            message=_context_message_for_tool_follow_up(response, tool_calls),
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
                remove_think=self.remove_think,
            )
        except Exception as e:
            raise ModelCallError(f"LLM call in tool loop failed: {e}", retryable=True) from e

        tool_calls = [ToolCallRequest.from_raw(tool_call) for tool_call in response.tool_calls or []]
        return ModelTurnResult(
            message=_context_message_for_tool_follow_up(response, tool_calls),
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
        max_artifact_bytes: int = DEFAULT_MAX_TOOL_RESULT_ARTIFACT_BYTES,
        artifact_read_available: bool = False,
    ):
        self.api = api
        self.allowed_tools = allowed_tools
        self.max_result_chars = max_result_chars
        self.max_artifact_bytes = max_artifact_bytes
        self.artifact_read_available = artifact_read_available

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
            if prepared.request.name == INTERNAL_ARTIFACT_READ_TOOL_NAME:
                result = await self._read_artifact(prepared.parameters)
            else:
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
        event_result: typing.Any = None
        if error is None:
            references = extract_tool_result_references(result)
            if references["artifact_refs"] or references["file_refs"]:
                message, event_result = build_tool_reference_message(
                    prepared.request.id,
                    result=result,
                    references=references,
                    max_result_chars=self.max_result_chars,
                )
                return ToolExecutionOutcome(
                    request=prepared.request,
                    parameters=prepared.parameters,
                    result=result,
                    event_result=event_result,
                    error=None,
                    message=message,
                )

            content = serialize_tool_result_content(result, is_error=False)
            if len(content) > self.max_result_chars:
                preview = content[: self.max_result_chars]
                content_bytes = content.encode("utf-8")
                if self.artifact_read_available and len(content_bytes) <= self.max_artifact_bytes:
                    artifact = self._build_tool_result_artifact(prepared, content, content_bytes=content_bytes)
                    message = build_tool_artifact_message(
                        prepared.request.id,
                        artifact_ref=artifact.to_reference(),
                        preview=preview,
                        original_chars=len(content),
                        kept_chars=len(preview),
                    )
                    event_result = {
                        "type": "langbot_tool_result_artifact",
                        "artifact": artifact.to_reference(),
                        "original_chars": len(content),
                        "kept_preview_chars": len(preview),
                    }
                    return ToolExecutionOutcome(
                        request=prepared.request,
                        parameters=prepared.parameters,
                        result=result,
                        event_result=event_result,
                        error=None,
                        message=message,
                        artifact=artifact,
                    )

                message = build_tool_call_message(
                    prepared.request.id,
                    prepared.request.name,
                    result,
                    is_error=False,
                    max_result_chars=self.max_result_chars,
                )
                reason = "artifact_too_large" if self.artifact_read_available else "artifact_read_unavailable"
                event_result = {
                    "type": "langbot_tool_result_preview",
                    "truncated": True,
                    "reason": reason,
                    "original_chars": len(content),
                    "original_bytes": len(content_bytes),
                    "kept_preview_chars": len(preview),
                    "preview": preview,
                }
                return ToolExecutionOutcome(
                    request=prepared.request,
                    parameters=prepared.parameters,
                    result=result,
                    event_result=event_result,
                    error=None,
                    message=message,
                )

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
            event_result=event_result,
            error=error,
            message=message,
        )

    async def _read_artifact(self, parameters: dict[str, typing.Any]) -> dict[str, typing.Any]:
        if not self.artifact_read_available:
            raise PermissionDeniedError("Artifact read API is not available for this run")

        artifact_id = str(parameters.get("artifact_id") or "").strip()
        if not artifact_id:
            raise ValueError("artifact_id is required")

        offset = _non_negative_int(parameters.get("offset"), default=0)
        limit = min(_positive_int(parameters.get("limit"), default=8000), 20_000)
        result = await self.api.artifact_read(artifact_id=artifact_id, offset=offset, limit=limit)

        content_base64 = _model_or_mapping_get(result, "content_base64")
        text: str | None = None
        if isinstance(content_base64, str) and content_base64:
            text = base64.b64decode(content_base64).decode("utf-8", errors="replace")

        return {
            "artifact_id": _model_or_mapping_get(result, "artifact_id", artifact_id),
            "mime_type": _model_or_mapping_get(result, "mime_type"),
            "size_bytes": _model_or_mapping_get(result, "size_bytes"),
            "offset": _model_or_mapping_get(result, "offset", offset),
            "length": _model_or_mapping_get(result, "length"),
            "has_more": bool(_model_or_mapping_get(result, "has_more", False)),
            "text": text,
            "file_key": _model_or_mapping_get(result, "file_key"),
        }

    def _build_tool_result_artifact(
        self,
        prepared: PreparedToolCall,
        content: str,
        *,
        content_bytes: bytes | None = None,
    ) -> ToolResultArtifact:
        if content_bytes is None:
            content_bytes = content.encode("utf-8")
        artifact_id = f"tool-result-{uuid.uuid4().hex}"
        return ToolResultArtifact(
            artifact_id=artifact_id,
            artifact_type="tool_result",
            mime_type="text/plain; charset=utf-8",
            name=f"{prepared.request.name}-{prepared.request.id}.txt",
            size_bytes=len(content_bytes),
            sha256=hashlib.sha256(content_bytes).hexdigest(),
            content_base64=base64.b64encode(content_bytes).decode("ascii"),
            metadata={
                "tool_name": prepared.request.name,
                "tool_call_id": prepared.request.id,
                "stored_by": "langbot-local-agent",
            },
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


def _non_negative_int(value: typing.Any, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _positive_int(value: typing.Any, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value


def _model_or_mapping_get(value: typing.Any, key: str, default: typing.Any = None) -> typing.Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
