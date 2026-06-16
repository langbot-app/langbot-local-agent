"""Model calling utilities with fallback support."""

from __future__ import annotations

import asyncio
import json
import logging
import typing
import uuid
from dataclasses import dataclass

from langbot_plugin.api.entities.builtin.provider.message import Message, MessageChunk
from langbot_plugin.api.entities.builtin.resource.tool import LLMTool
from langbot_plugin.api.proxies.agent_run_api import AgentRunAPIProxy

from pkg.config import DEFAULT_MAX_TOOL_RESULT_CHARS

TOOL_RESULT_TRUNCATION_MARKER = "tool result truncated"
TOOL_RESULT_ARTIFACT_MARKER = "tool result stored as artifact"
TOOL_RESULT_REFERENCE_MARKER = "tool result references"
INTERNAL_ARTIFACT_READ_TOOL_NAME = "langbot_artifact_read"
REFERENCE_RESULT_MAX_REFS = 20
CONTEXT_OVERFLOW_PATTERNS = (
    "context length",
    "context window",
    "context too long",
    "maximum context",
    "max context",
    "too many tokens",
    "token limit",
    "tokens exceed",
    "prompt is too long",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMCallResult:
    message: Message
    usage: dict[str, typing.Any] | None = None


class ModelCallError(Exception):
    """Error during model invocation."""

    def __init__(self, message: str, retryable: bool = False, code: str | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.code = code

    @property
    def is_context_overflow(self) -> bool:
        if self.code == "context_overflow":
            return True
        return is_context_overflow_error(self)


def is_context_overflow_error(error: Exception) -> bool:
    """Best-effort provider-neutral context overflow detection."""
    text = str(error).lower()
    return any(pattern in text for pattern in CONTEXT_OVERFLOW_PATTERNS)


async def build_llm_tools(
    api: AgentRunAPIProxy,
    allowed_tools: set[str],
) -> list[LLMTool]:
    """Build LLMTool list from allowed tool names.

    Fetches tool details from LangBot and converts to LLMTool format
    for LLM function calling.

    Args:
        api: AgentRunAPIProxy for authorized access
        allowed_tools: Set of tool names authorized for this run

    Returns:
        List of LLMTool objects ready for LLM invocation
    """
    tools = await asyncio.gather(
        *(_build_llm_tool(api, tool_name) for tool_name in sorted(allowed_tools)),
    )
    return [tool for tool in tools if tool is not None]


async def _build_llm_tool(api: AgentRunAPIProxy, tool_name: str) -> LLMTool | None:
    try:
        detail = await api.get_tool_detail(tool_name)
        description = detail.get("description", "")

        async def _placeholder_func(**kwargs):
            return kwargs

        return LLMTool(
            name=detail.get("name", tool_name),
            human_desc=description,
            description=description,
            parameters=detail.get("parameters", {}),
            func=_placeholder_func,
        )
    except Exception:
        logger.warning("Tool detail fetch failed; skipping tool: %s", tool_name, exc_info=True)
        return None


def build_artifact_read_tool() -> LLMTool:
    """Build the runner-owned Host artifact read tool."""

    async def _placeholder_func(**kwargs):
        return kwargs

    return LLMTool(
        name=INTERNAL_ARTIFACT_READ_TOOL_NAME,
        human_desc="Read a slice of a LangBot Host artifact for this run.",
        description=(
            "Read a bounded text slice from a LangBot Host artifact by artifact_id. "
            "Use this when a tool result says the full output was stored as an artifact."
        ),
        parameters={
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "Artifact ID returned in a prior tool result reference.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "Byte offset to start reading from.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20000,
                    "default": 8000,
                    "description": "Maximum bytes to read.",
                },
            },
            "required": ["artifact_id"],
            "additionalProperties": False,
        },
        func=_placeholder_func,
    )


async def invoke_with_fallback(
    api: AgentRunAPIProxy,
    model_ids: list[str],
    messages: list[Message],
    tools: list[LLMTool] | None = None,
    remove_think: bool = False,
) -> tuple[Message, str]:
    """Invoke LLM with sequential fallback on failure.

    Args:
        api: AgentRunAPIProxy for authorized access
        model_ids: Ordered list of model IDs to try
        messages: Conversation messages for LLM
        tools: Optional tools for function calling

    Returns:
        Tuple of (message from first successful model, committed model ID)

    Raises:
        ModelCallError: All models failed
    """
    if not model_ids:
        raise ModelCallError("No model configured", retryable=False)

    last_error: Exception | None = None
    for index, model_id in enumerate(model_ids):
        try:
            response = await api.invoke_llm(
                llm_model_uuid=model_id,
                messages=messages,
                funcs=tools or [],
                remove_think=remove_think,
            )
            return response, model_id
        except Exception as e:
            last_error = e
            if index < len(model_ids) - 1:
                logger.warning("LLM model failed; falling back to next configured model: %s", model_id, exc_info=True)
            else:
                logger.warning("LLM model failed and no fallback remains: %s", model_id, exc_info=True)
            continue

    raise ModelCallError(
        f"All models failed. Last error: {last_error}",
        retryable=True,
    )


async def invoke_with_fallback_result(
    api: AgentRunAPIProxy,
    model_ids: list[str],
    messages: list[Message],
    tools: list[LLMTool] | None = None,
    remove_think: bool = False,
) -> tuple[LLMCallResult, str]:
    if not model_ids:
        raise ModelCallError("No model configured", retryable=False)

    last_error: Exception | None = None
    for index, model_id in enumerate(model_ids):
        try:
            invoke = getattr(api, "invoke_llm_with_usage", None)
            if not callable(invoke):
                invoke = api.invoke_llm
            response = await invoke(
                llm_model_uuid=model_id,
                messages=messages,
                funcs=tools or [],
                remove_think=remove_think,
            )
            return normalize_llm_call_result(response), model_id
        except Exception as e:
            last_error = e
            if index < len(model_ids) - 1:
                logger.warning("LLM model failed; falling back to next configured model: %s", model_id, exc_info=True)
            else:
                logger.warning("LLM model failed and no fallback remains: %s", model_id, exc_info=True)
            continue

    raise ModelCallError(
        f"All models failed. Last error: {last_error}",
        retryable=True,
    )


class StreamingModelCaller:
    """Handles streaming LLM calls with fallback and accumulation.

    Fallback is only possible before the first chunk is yielded.
    Once streaming starts, the model is committed.
    """

    def __init__(
        self,
        api: AgentRunAPIProxy,
        model_ids: list[str],
        messages: list[Message],
        tools: list[LLMTool] | None = None,
        remove_think: bool = False,
    ):
        self.api = api
        self.model_ids = model_ids
        self.messages = messages
        self.tools = tools or []
        self.remove_think = remove_think

        self._committed_model_id: str | None = None
        self._accumulated_content = ""
        self._tool_calls_map: dict[str, dict[str, typing.Any]] = {}
        self._tool_call_id_keys: dict[str, str] = {}
        self._tool_call_position_keys: dict[str, str] = {}
        self._msg_idx = 0
        self._msg_sequence = 0
        self._usage: dict[str, typing.Any] | None = None

    async def _next_non_empty_chunk(self, stream: typing.AsyncIterator[typing.Any]) -> MessageChunk:
        while True:
            raw_chunk = await stream.__anext__()
            chunk, usage = _normalize_stream_chunk(raw_chunk)
            if usage is not None:
                self._usage = usage
            if chunk is not None:
                return chunk

    async def stream(
        self,
    ) -> typing.AsyncGenerator[tuple[MessageChunk, bool], None]:
        """Stream chunks with accumulation.

        Fallback rules:
        - Before first chunk: can fallback to next model on failure
        - After first chunk (committed): no fallback, failure is terminal

        Yields:
            Tuple of (MessageChunk, is_delta) where is_delta=False for final accumulated chunk
        """
        if not self.model_ids:
            raise ModelCallError("No model configured", retryable=False)

        # Try each model until one succeeds (before first chunk)
        stream = None
        last_error: Exception | None = None
        model_id = None

        for model_id in self.model_ids:
            try:
                # Try to get first chunk to verify stream works
                stream_invoke = getattr(self.api, "invoke_llm_stream_events", None)
                if not callable(stream_invoke):
                    stream_invoke = self.api.invoke_llm_stream
                stream = stream_invoke(
                    llm_model_uuid=model_id,
                    messages=self.messages,
                    funcs=self.tools,
                    remove_think=self.remove_think,
                )
                first_chunk = await self._next_non_empty_chunk(stream)
                # First chunk received - model is now committed
                self._committed_model_id = model_id

                # Yield first chunk
                chunk, is_final = self._process_chunk(first_chunk)
                yield chunk, not is_final

                # BREAK THE FALLBACK LOOP - we are now committed
                break

            except StopAsyncIteration:
                # Empty stream before any model-visible chunk is a model failure.
                last_error = ModelCallError(
                    f"Model {model_id} stream ended before first chunk",
                    retryable=True,
                )
                if model_id != self.model_ids[-1]:
                    logger.warning(
                        "Streaming LLM model ended before first chunk; falling back to next configured model: %s",
                        model_id,
                    )
                else:
                    logger.warning(
                        "Streaming LLM model ended before first chunk and no fallback remains: %s",
                        model_id,
                    )
                continue
            except Exception as e:
                # Failure before first chunk - try next model
                last_error = e
                if model_id != self.model_ids[-1]:
                    logger.warning(
                        "Streaming LLM model failed before first chunk; falling back to next configured model: %s",
                        model_id,
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        "Streaming LLM model failed before first chunk and no fallback remains: %s",
                        model_id,
                        exc_info=True,
                    )
                continue
        else:
            # All models failed before first chunk
            raise ModelCallError(
                f"All models failed during streaming setup. Last error: {last_error}",
                retryable=True,
            )

        # Continue with rest of stream (no fallback allowed after this point)
        # Any failure here is terminal for the run
        try:
            async for raw_stream_item in stream:
                raw_chunk, usage = _normalize_stream_chunk(raw_stream_item)
                if usage is not None:
                    self._usage = usage
                if raw_chunk is None:
                    continue
                chunk, is_final = self._process_chunk(raw_chunk)
                yield chunk, not is_final
        except Exception as e:
            # Post-commit failure - no fallback, raise terminal error
            raise ModelCallError(
                f"Model {model_id} failed after first chunk (no fallback possible): {e}",
                retryable=False,
            )

    def _process_chunk(self, raw_chunk: MessageChunk) -> tuple[MessageChunk, bool]:
        """Process and accumulate a chunk.

        Returns:
            Tuple of (processed_chunk, is_final)
        """
        self._msg_idx += 1

        # Accumulate content
        if raw_chunk.content:
            if isinstance(raw_chunk.content, str):
                self._accumulated_content += raw_chunk.content
            else:
                # Handle list content
                for ce in raw_chunk.content:
                    if hasattr(ce, "type") and ce.type == "text" and ce.text:
                        self._accumulated_content += ce.text

        # Accumulate tool calls
        if raw_chunk.tool_calls:
            for position, tc in enumerate(raw_chunk.tool_calls):
                key = self._tool_call_accumulation_key(tc, position)
                tc_id = _normalized_tool_call_id(tc.id)
                if key not in self._tool_calls_map:
                    self._tool_calls_map[key] = {
                        "id": tc_id or _new_tool_call_id(),
                        "type": tc.type or "function",
                        "function_name": "",
                        "function_arguments": "",
                    }
                elif tc_id:
                    self._tool_calls_map[key]["id"] = tc_id
                if tc.function:
                    if tc.function.name:
                        self._tool_calls_map[key]["function_name"] = tc.function.name
                    if tc.function.arguments:
                        self._tool_calls_map[key]["function_arguments"] += tc.function.arguments

        # Yield every 8 chunks or on final
        if self._msg_idx % 8 == 0 or raw_chunk.is_final:
            self._msg_sequence += 1

            # Build accumulated chunk
            tool_calls = None
            if self._tool_calls_map and raw_chunk.is_final:
                from langbot_plugin.api.entities.builtin.provider.message import FunctionCall, ToolCall

                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type=tc["type"],
                        function=FunctionCall(
                            name=tc["function_name"],
                            arguments=tc["function_arguments"],
                        ),
                    )
                    for tc in self._tool_calls_map.values()
                ]

            chunk = MessageChunk(
                role=raw_chunk.role or "assistant",
                content=self._accumulated_content,
                tool_calls=tool_calls,
                is_final=raw_chunk.is_final,
                msg_sequence=self._msg_sequence,
            )
            return chunk, raw_chunk.is_final

        # Don't yield yet
        return MessageChunk(role="assistant", content="", is_final=False), False

    def get_accumulated_content(self) -> str:
        """Get accumulated content so far."""
        return self._accumulated_content

    def get_tool_calls(self) -> list[dict[str, typing.Any]]:
        """Get accumulated tool calls as raw dicts."""
        return list(self._tool_calls_map.values())

    def get_committed_model_id(self) -> str | None:
        """Get the model ID that was committed for this stream."""
        return self._committed_model_id

    def get_usage(self) -> dict[str, typing.Any] | None:
        """Get provider usage reported by the committed stream, if available."""
        return dict(self._usage) if self._usage is not None else None

    def _tool_call_accumulation_key(self, tool_call: typing.Any, position: int) -> str:
        tc_id = _normalized_tool_call_id(getattr(tool_call, "id", None))
        position_key = _tool_call_position_key(tool_call, position)
        if tc_id:
            key = self._tool_call_id_keys.get(tc_id) or self._tool_call_position_keys.get(position_key) or f"id:{tc_id}"
            self._tool_call_id_keys[tc_id] = key
            self._tool_call_position_keys[position_key] = key
            return key

        key = self._tool_call_position_keys.get(position_key)
        if key is None:
            key = position_key
            self._tool_call_position_keys[position_key] = key
        return key


def build_tool_call_message(
    tool_call_id: str,
    tool_name: str,
    result: typing.Any,
    is_error: bool = False,
    max_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
) -> Message:
    """Build a tool result message.

    Args:
        tool_call_id: Tool call ID from LLM
        tool_name: Name of the tool that was called
        result: Tool execution result
        is_error: Whether the result is an error
        max_result_chars: Maximum result content characters before fallback truncation

    Returns:
        Message with role="tool" containing the result
    """
    content = bound_tool_result_content(
        result,
        is_error=is_error,
        max_result_chars=max_result_chars,
    )

    return Message(
        role="tool",
        content=content,
        tool_call_id=tool_call_id,
    )


def build_tool_artifact_message(
    tool_call_id: str,
    *,
    artifact_ref: dict[str, typing.Any],
    preview: str,
    original_chars: int,
    kept_chars: int,
) -> Message:
    """Build a model-facing tool message that references a Host artifact."""
    payload = {
        "type": TOOL_RESULT_ARTIFACT_MARKER,
        "truncated": True,
        "original_chars": original_chars,
        "kept_preview_chars": kept_chars,
        "artifact": artifact_ref,
        "preview": preview,
        "next_step": (
            f"Use the {INTERNAL_ARTIFACT_READ_TOOL_NAME} tool with this artifact_id "
            "and an offset/limit when you need more of the full tool output."
        ),
    }

    return Message(
        role="tool",
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=tool_call_id,
    )


def build_tool_reference_message(
    tool_call_id: str,
    *,
    result: typing.Any,
    references: dict[str, list[dict[str, typing.Any]]],
    max_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
) -> tuple[Message, dict[str, typing.Any]]:
    """Build a model-facing tool message for Host/sandbox artifact or file refs."""
    content = serialize_tool_result_content(result, is_error=False)
    if isinstance(max_result_chars, bool) or not isinstance(max_result_chars, int) or max_result_chars < 1:
        max_result_chars = DEFAULT_MAX_TOOL_RESULT_CHARS

    payload: dict[str, typing.Any] = {
        "type": TOOL_RESULT_REFERENCE_MARKER,
        "artifact_refs": references["artifact_refs"],
        "file_refs": references["file_refs"],
        "original_chars": len(content),
        "next_step": (
            f"Use {INTERNAL_ARTIFACT_READ_TOOL_NAME} with an artifact_id when you need more artifact content. "
            "For sandbox files, call the sandbox-provided file tools with the returned file reference."
        ),
    }
    if len(content) <= max_result_chars:
        payload["result"] = result
        payload["truncated"] = False
    else:
        preview = content[:max_result_chars]
        payload.update(
            {
                "truncated": True,
                "kept_preview_chars": len(preview),
                "preview": preview,
            }
        )

    return (
        Message(
            role="tool",
            content=json.dumps(payload, ensure_ascii=False, default=str),
            tool_call_id=tool_call_id,
        ),
        payload,
    )


def extract_tool_result_references(result: typing.Any) -> dict[str, list[dict[str, typing.Any]]]:
    """Extract explicit Host/sandbox artifact and file references from a tool result."""
    artifact_refs: list[dict[str, typing.Any]] = []
    file_refs: list[dict[str, typing.Any]] = []
    seen_artifacts: set[str] = set()
    seen_files: set[str] = set()

    def add_artifact_ref(value: dict[str, typing.Any]) -> None:
        artifact_id = value.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in seen_artifacts:
            return
        ref = _copy_reference_fields(
            value,
            (
                "artifact_id",
                "artifact_type",
                "mime_type",
                "size",
                "size_bytes",
                "name",
                "source",
                "url",
                "sha256",
                "digest",
                "summary",
                "expires_at",
                "permissions",
                "metadata",
            ),
        )
        if "size_bytes" not in ref and "size" in ref:
            ref["size_bytes"] = ref["size"]
        seen_artifacts.add(artifact_id)
        artifact_refs.append(ref)

    def add_file_ref(value: dict[str, typing.Any]) -> None:
        file_id = value.get("file_key") or value.get("file_id")
        if not isinstance(file_id, str) or not file_id or file_id in seen_files:
            return
        ref = _copy_reference_fields(
            value,
            (
                "file_key",
                "file_id",
                "file_name",
                "name",
                "mime_type",
                "size",
                "size_bytes",
                "source",
                "summary",
                "metadata",
            ),
        )
        seen_files.add(file_id)
        file_refs.append(ref)

    def walk(value: typing.Any, depth: int = 0) -> None:
        if depth > 6 or len(artifact_refs) + len(file_refs) >= REFERENCE_RESULT_MAX_REFS:
            return
        if isinstance(value, dict):
            add_artifact_ref(value)
            add_file_ref(value)
            for key in ("artifact_refs", "artifacts", "file_refs", "files", "attachments", "items", "result"):
                nested = value.get(key)
                if isinstance(nested, (dict, list, tuple)):
                    walk(nested, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item, depth + 1)

    walk(result)
    return {"artifact_refs": artifact_refs, "file_refs": file_refs}


def _normalized_tool_call_id(value: typing.Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    return text


def _tool_call_position_key(tool_call: typing.Any, position: int) -> str:
    index = getattr(tool_call, "index", None)
    if index is None:
        extra = getattr(tool_call, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            index = extra.get("index")
    if index is not None:
        return f"index:{index}"
    return f"position:{position}"


def _new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


def normalize_llm_call_result(response: typing.Any) -> LLMCallResult:
    if isinstance(response, Message):
        return LLMCallResult(message=response, usage=_normalize_usage(_model_or_mapping_get(response, "usage")))

    message = _model_or_mapping_get(response, "message")
    if message is None:
        data = _model_or_mapping_get(response, "data")
        if isinstance(data, dict):
            message = data.get("message")
    if isinstance(message, Message):
        normalized_message = message
    elif isinstance(message, dict):
        normalized_message = Message.model_validate(message)
    else:
        normalized_message = Message.model_validate(response)

    usage = _normalize_usage(_model_or_mapping_get(response, "usage"))
    if usage is None and isinstance(response, dict):
        usage = _normalize_usage(response.get("token_usage") or response.get("model_usage"))
    return LLMCallResult(message=normalized_message, usage=usage)


def _normalize_stream_chunk(raw_chunk: typing.Any) -> tuple[MessageChunk | None, dict[str, typing.Any] | None]:
    if raw_chunk is None:
        return None, None
    if isinstance(raw_chunk, MessageChunk):
        return raw_chunk, _normalize_usage(_model_or_mapping_get(raw_chunk, "usage"))

    usage = _normalize_usage(_model_or_mapping_get(raw_chunk, "usage"))
    missing = object()
    chunk = _model_or_mapping_get(raw_chunk, "chunk", missing)
    if chunk is missing:
        chunk = _model_or_mapping_get(raw_chunk, "message", missing)
    if chunk is None:
        return None, usage
    if chunk is missing:
        chunk = None
    if isinstance(chunk, MessageChunk):
        return chunk, usage
    if isinstance(chunk, dict):
        return MessageChunk.model_validate(chunk), usage
    if isinstance(raw_chunk, dict):
        chunk_payload = raw_chunk.get("chunk") or raw_chunk.get("message")
        if isinstance(chunk_payload, MessageChunk):
            return chunk_payload, usage
        if isinstance(chunk_payload, dict):
            return MessageChunk.model_validate(chunk_payload), usage
        if usage is not None:
            return None, usage
    return MessageChunk.model_validate(raw_chunk), usage


def _normalize_usage(usage: typing.Any) -> dict[str, typing.Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return None


def _model_or_mapping_get(value: typing.Any, key: str, default: typing.Any = None) -> typing.Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _copy_reference_fields(
    value: dict[str, typing.Any],
    allowed_fields: tuple[str, ...],
) -> dict[str, typing.Any]:
    ref: dict[str, typing.Any] = {}
    for key in allowed_fields:
        if key in value and value[key] is not None:
            ref[key] = value[key]
    return ref


def bound_tool_result_content(
    result: typing.Any,
    *,
    is_error: bool = False,
    max_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
) -> str:
    """Serialize and bound a tool result for model-facing tool messages."""
    content = serialize_tool_result_content(result, is_error=is_error)
    if isinstance(max_result_chars, bool) or not isinstance(max_result_chars, int) or max_result_chars < 1:
        max_result_chars = DEFAULT_MAX_TOOL_RESULT_CHARS

    original_chars = len(content)
    if original_chars <= max_result_chars:
        return content

    kept_content = content[:max_result_chars]
    marker = (
        "\n\n"
        f"[{TOOL_RESULT_TRUNCATION_MARKER}: original_chars={original_chars}, "
        f"kept_chars={len(kept_content)}. "
        "Only the leading content is included. No readable Host artifact reference was emitted by the runner. "
        "For large files, tools should return Host artifact or file references instead of inline content.]"
    )
    return kept_content + marker


def serialize_tool_result_content(result: typing.Any, *, is_error: bool = False) -> str:
    """Serialize a raw tool result into provider tool-message content."""
    if is_error:
        return f"Error: {result}"
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)
