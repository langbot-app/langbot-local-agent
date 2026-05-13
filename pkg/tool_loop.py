"""Tool calling loop implementation."""

from __future__ import annotations

import json
import typing

from langbot_plugin.api.entities.builtin.agent_runner.result import AgentRunResult
from langbot_plugin.api.entities.builtin.provider.message import Message, MessageChunk
from langbot_plugin.api.proxies.agent_run_api import AgentRunAPIProxy, PermissionDeniedError

from .model_calling import ModelCallError, StreamingModelCaller, build_tool_call_message

# Maximum tool call iterations to prevent infinite loops
DEFAULT_MAX_TOOL_ITERATIONS = 10


class ToolCallLoop:
    """Manages tool calling loop with iteration limits.

    Only allows tools from ctx.resources.tools.
    Uses AgentRunAPIProxy.call_tool for authorized access.
    """

    def __init__(
        self,
        api: AgentRunAPIProxy,
        allowed_tools: set[str],
        max_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ):
        self.api = api
        self.allowed_tools = allowed_tools
        self.max_iterations = max_iterations
        self._iteration_count = 0

    async def execute_tool_call(
        self,
        tool_call: dict[str, typing.Any],
    ) -> tuple[typing.Any, str | None]:
        """Execute a single tool call with validation.

        Args:
            tool_call: Dict with id, type, function_name, function_arguments

        Returns:
            Tuple of (result, error_message)
            error_message is None on success, or error string on failure
        """
        tool_name = tool_call.get("function_name", "")
        arguments_str = tool_call.get("function_arguments", "{}")

        # Check if tool is authorized
        if tool_name not in self.allowed_tools:
            return (
                None,
                f"Tool '{tool_name}' is not authorized. Allowed tools: {list(self.allowed_tools)}",
            )

        # Parse arguments
        try:
            parameters = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON arguments for tool '{tool_name}': {e}"

        # Call tool via API proxy
        try:
            result = await self.api.call_tool(
                tool_name=tool_name,
                parameters=parameters,
            )
            return result, None
        except PermissionDeniedError as e:
            return None, str(e)
        except Exception as e:
            return None, f"Tool execution failed: {e}"

    def check_iteration_limit(self) -> bool:
        """Check if we've reached the iteration limit.

        Returns:
            True if we can continue, False if limit reached
        """
        return self._iteration_count < self.max_iterations

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self._iteration_count += 1

    def get_iteration_count(self) -> int:
        """Get current iteration count."""
        return self._iteration_count


async def run_tool_call_loop_streaming(
    api: AgentRunAPIProxy,
    allowed_tools: set[str],
    initial_messages: list[Message],
    committed_model_id: str,
    initial_content: str,
    initial_tool_calls: list[dict[str, typing.Any]],
    max_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
) -> typing.AsyncGenerator[
    tuple[MessageChunk | None, AgentRunResult | None, list[dict[str, typing.Any]]],
    None,
]:
    """Run tool calling loop for streaming mode.

    Yields:
        Tuple of (chunk_to_yield, tool_result_to_yield, updated_tool_calls)
        chunk_to_yield may be None if only tool results to process
        tool_result_to_yield is the tool.call.started/completed result

    The loop:
    1. Execute pending tool calls
    2. Build tool result messages
    3. Call LLM again
    4. Stream response
    5. Repeat until no more tool calls or limit reached
    """
    tool_loop = ToolCallLoop(api, allowed_tools, max_iterations)
    messages = initial_messages.copy()

    # Add the initial assistant message with tool calls
    if initial_tool_calls:
        # Build assistant message with tool calls
        from langbot_plugin.api.entities.builtin.provider.message import FunctionCall, ToolCall
        assistant_msg = Message(
            role="assistant",
            content=initial_content,
            tool_calls=[
                ToolCall(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=FunctionCall(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", ""),
                    ),
                )
                for tc in initial_tool_calls
            ],
        )
        messages.append(assistant_msg)

    pending_tool_calls = initial_tool_calls

    while pending_tool_calls and tool_loop.check_iteration_limit():
        tool_loop.increment_iteration()

        # Execute each tool call
        for tool_call in pending_tool_calls:
            tool_call_id = tool_call.get("id", str(hash(tool_call)))
            tool_name = tool_call.get("function_name", "unknown")
            parameters = {}
            try:
                args_str = tool_call.get("function_arguments", "{}")
                parameters = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                parameters = {}

            # Yield tool.call.started
            yield None, AgentRunResult.tool_call_started(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                parameters=parameters,
            ), []

            # Execute tool
            result, error = await tool_loop.execute_tool_call(tool_call)

            # Yield tool.call.completed
            yield None, AgentRunResult.tool_call_completed(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result=result,
                error=error,
            ), []

            # Build tool result message
            if error:
                tool_msg = build_tool_call_message(tool_call_id, tool_name, error, is_error=True)
            else:
                tool_msg = build_tool_call_message(tool_call_id, tool_name, result)
            messages.append(tool_msg)

        # Call LLM again with tool results
        caller = StreamingModelCaller(
            api=api,
            model_ids=[committed_model_id],
            messages=messages,
            tools=[],  # Tools already passed in initial call
        )

        pending_tool_calls = []
        accumulated_content = initial_content

        try:
            async for chunk, is_delta in caller.stream():
                if chunk.content:  # Only yield non-empty chunks
                    accumulated_content = caller.get_accumulated_content()
                    yield chunk, None, []

            # Check for new tool calls
            new_tool_calls = caller.get_tool_calls()
            if new_tool_calls:
                pending_tool_calls = new_tool_calls
                # Update initial_content for next iteration
                initial_content = accumulated_content

        except ModelCallError as e:
            # Tool loop failed - return error
            yield None, AgentRunResult.run_failed(
                error=str(e),
                code="runner.tool_loop_error",
                retryable=e.retryable,
            ), []
            return

    # Check if we hit iteration limit
    if pending_tool_calls and not tool_loop.check_iteration_limit():
        yield None, AgentRunResult.run_failed(
            error=f"Tool call iteration limit reached ({max_iterations})",
            code="runner.tool_loop_limit",
            retryable=False,
        ), []
