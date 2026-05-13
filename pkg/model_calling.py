"""Model calling utilities with fallback support."""

from __future__ import annotations

import json
import typing

from langbot_plugin.api.entities.builtin.provider.message import Message, MessageChunk
from langbot_plugin.api.entities.builtin.resource.tool import LLMTool
from langbot_plugin.api.proxies.agent_run_api import AgentRunAPIProxy


class ModelCallError(Exception):
    """Error during model invocation."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


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
    tools: list[LLMTool] = []

    for tool_name in allowed_tools:
        try:
            detail = await api.get_tool_detail(tool_name)
            tool = LLMTool(
                name=detail.get('name', tool_name),
                description=detail.get('description', ''),
                parameters=detail.get('parameters', {}),
            )
            tools.append(tool)
        except Exception:
            # Tool detail fetch failed - skip this tool
            continue

    return tools


async def invoke_with_fallback(
    api: AgentRunAPIProxy,
    model_ids: list[str],
    messages: list[Message],
    tools: list[LLMTool] | None = None,
) -> Message:
    """Invoke LLM with sequential fallback on failure.

    Args:
        api: AgentRunAPIProxy for authorized access
        model_ids: Ordered list of model IDs to try
        messages: Conversation messages for LLM
        tools: Optional tools for function calling

    Returns:
        Message from first successful model

    Raises:
        ModelCallError: All models failed
    """
    if not model_ids:
        raise ModelCallError("No model configured", retryable=False)

    last_error: Exception | None = None
    for model_id in model_ids:
        try:
            return await api.invoke_llm(
                llm_model_uuid=model_id,
                messages=messages,
                funcs=tools or [],
            )
        except Exception as e:
            last_error = e
            # Log and continue to next model
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
    ):
        self.api = api
        self.model_ids = model_ids
        self.messages = messages
        self.tools = tools or []

        self._committed_model_id: str | None = None
        self._accumulated_content = ""
        self._tool_calls_map: dict[str, dict[str, typing.Any]] = {}
        self._msg_idx = 0
        self._msg_sequence = 0

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
                stream = self.api.invoke_llm_stream(
                    llm_model_uuid=model_id,
                    messages=self.messages,
                    funcs=self.tools,
                )
                first_chunk = await stream.__anext__()
                # First chunk received - model is now committed
                self._committed_model_id = model_id

                # Yield first chunk
                chunk, is_final = self._process_chunk(first_chunk)
                yield chunk, not is_final

                # BREAK THE FALLBACK LOOP - we are now committed
                break

            except StopAsyncIteration:
                # Empty stream - treat as success, commit this model
                self._committed_model_id = model_id
                return
            except Exception as e:
                # Failure before first chunk - try next model
                last_error = e
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
            async for raw_chunk in stream:
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
            for tc in raw_chunk.tool_calls:
                tc_id = tc.id
                if tc_id not in self._tool_calls_map:
                    self._tool_calls_map[tc_id] = {
                        "id": tc_id,
                        "type": tc.type,
                        "function_name": "",
                        "function_arguments": "",
                    }
                if tc.function:
                    if tc.function.name:
                        self._tool_calls_map[tc_id]["function_name"] = tc.function.name
                    if tc.function.arguments:
                        self._tool_calls_map[tc_id]["function_arguments"] += tc.function.arguments

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


def build_tool_call_message(
    tool_call_id: str,
    tool_name: str,
    result: typing.Any,
    is_error: bool = False,
) -> Message:
    """Build a tool result message.

    Args:
        tool_call_id: Tool call ID from LLM
        tool_name: Name of the tool that was called
        result: Tool execution result
        is_error: Whether the result is an error

    Returns:
        Message with role="tool" containing the result
    """
    content: str
    if is_error:
        content = f"Error: {result}"
    elif isinstance(result, str):
        content = result
    else:
        content = json.dumps(result, ensure_ascii=False)

    return Message(
        role="tool",
        content=content,
        tool_call_id=tool_call_id,
    )
