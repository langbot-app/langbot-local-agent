"""Local Agent default runner implementation.

Supports:
- Model fallback (primary + fallbacks)
- Streaming and non-streaming
- Tool calling loop with max iterations
- Knowledge retrieval with permission validation
- Protocol v1 AgentRunResult output
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from langbot_plugin.api.definition.components.agent_runner.runner import AgentRunner
from langbot_plugin.api.entities.builtin.agent_runner import (
    AgentRunContext,
    AgentRunnerCapabilities,
    AgentRunnerPermissions,
    AgentRunResult,
)
from langbot_plugin.api.entities.builtin.provider.message import FunctionCall, Message, ToolCall

from pkg.config import get_knowledge_base_ids, get_rerank_config, parse_model_config
from pkg.messages import build_messages
from pkg.model_calling import (
    ModelCallError,
    StreamingModelCaller,
    build_llm_tools,
    build_tool_call_message,
    invoke_with_fallback,
)
from pkg.rag import retrieve_from_knowledge_bases
from pkg.tool_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    ToolCallLoop,
)


def _build_assistant_tool_call_message(
    content: str,
    tool_calls: list[dict[str, Any]],
) -> Message:
    """Build an assistant message that preserves tool call IDs for tool results."""
    return Message(
        role="assistant",
        content=content,
        tool_calls=[
            ToolCall(
                id=tc["id"],
                type=tc.get("type", "function"),
                function=FunctionCall(
                    name=tc.get("function_name", ""),
                    arguments=tc.get("function_arguments", ""),
                ),
            )
            for tc in tool_calls
        ],
    )


DEFAULT_HISTORY_LIMIT = 50


def _message_from_transcript_item(item: dict[str, Any], current_event_id: str) -> Message | None:
    """Convert one Host transcript item into a model message."""
    if item.get("event_id") == current_event_id:
        return None

    content_json = item.get("content_json")
    if isinstance(content_json, dict):
        try:
            return Message.model_validate(content_json)
        except Exception:
            pass

    role = item.get("role")
    content = item.get("content")
    if isinstance(role, str) and isinstance(content, str) and content:
        return Message(role=role, content=content)

    return None


async def _get_history_messages(api: Any, ctx: AgentRunContext) -> list[Message]:
    """Pull conversation history from Host APIs when authorized."""
    context = ctx.context
    if not context.available_apis.history_page or not context.conversation_id:
        return []

    page = await api.history_page(
        conversation_id=context.conversation_id,
        limit=DEFAULT_HISTORY_LIMIT,
        direction="backward",
        include_artifacts=True,
    )

    messages: list[Message] = []
    for item in reversed(page.get("items", [])):
        if not isinstance(item, dict):
            continue
        message = _message_from_transcript_item(item, ctx.event.event_id)
        if message is not None:
            messages.append(message)

    return messages


class DefaultAgentRunner(AgentRunner):
    """Default AgentRunner for Local Agent.

    Full-featured LLM runner with:
    - Model primary/fallback selection
    - Streaming and non-streaming output
    - Tool calling loop
    - Knowledge retrieval (RAG)

    All resource access goes through AgentRunAPIProxy for authorization.
    """

    @classmethod
    def get_capabilities(cls) -> AgentRunnerCapabilities:
        """Get runner capabilities."""
        return AgentRunnerCapabilities(
            streaming=True,
            tool_calling=True,
            knowledge_retrieval=True,
            multimodal_input=True,
            stateful_session=True,
        )

    @classmethod
    def get_permissions(cls) -> AgentRunnerPermissions:
        """Get runner permissions for resource access."""
        return AgentRunnerPermissions(
            models=["invoke", "stream"],
            tools=["detail", "call"],
            knowledge_bases=["list", "retrieve"],
            history=["page", "search"],
        )

    async def run(
        self, ctx: AgentRunContext
    ) -> AsyncGenerator[AgentRunResult, None]:
        """Run the agent with full LLM capabilities.

        Implementation:
        1. Get authorized models and parse model config
        2. Retrieve knowledge base context (if configured)
        3. Build messages from prompt + history + input
        4. Stream/Invoke LLM with fallback support
        5. Handle tool calling loop
        6. Yield AgentRunResult events
        """
        api = self.get_run_api(ctx)

        # Get authorized models
        allowed_model_ids = set(m.model_id for m in api.get_allowed_models())

        # Parse model config (supports string and dict formats)
        model_config = ctx.config.get("model")
        model_ids = parse_model_config(model_config, allowed_model_ids)

        if not model_ids:
            yield AgentRunResult.run_failed(
                ctx.run_id,
                error="No authorized model for local-agent",
                code="runner.no_model",
            )
            return

        # Get allowed tools for tool calling.
        allowed_tools = set(t.tool_name for t in api.get_allowed_tools())

        # Get allowed KB IDs from config (intersection with authorized)
        allowed_kb_ids = set(kb.kb_id for kb in api.get_allowed_knowledge_bases())
        kb_ids = get_knowledge_base_ids(ctx.config, allowed_kb_ids)

        # Get rerank configuration
        rerank_model_id, rerank_top_k = get_rerank_config(ctx.config)

        # Get user input text
        user_text = ctx.input.to_text()

        # Retrieve from knowledge bases if configured
        rag_context = ""
        if kb_ids and user_text:
            rag_context = await retrieve_from_knowledge_bases(
                api=api,
                kb_ids=kb_ids,
                query_text=user_text,
                top_k=5,
                rerank_model_id=rerank_model_id,
                rerank_top_k=rerank_top_k,
            )

        history_messages = await _get_history_messages(api, ctx)

        # Build messages for LLM
        prompt_config = ctx.config.get("prompt", [])
        messages = build_messages(
            prompt_config=prompt_config,
            history_messages=history_messages,
            user_text=user_text,
            input_contents=ctx.input.contents,
            rag_context=rag_context if rag_context else None,
        )

        # Prefer host runtime capability so non-streaming adapters keep the
        # same behavior as the original built-in local-agent runner.
        use_streaming = ctx.config.get("streaming")
        if use_streaming is None:
            use_streaming = ctx.runtime.metadata.get("streaming_supported", True)

        if use_streaming:
            async for result in self._run_streaming(
                run_id=ctx.run_id,
                api=api,
                model_ids=model_ids,
                messages=messages,
                allowed_tools=allowed_tools,
            ):
                yield result
        else:
            async for result in self._run_non_streaming(
                run_id=ctx.run_id,
                api=api,
                model_ids=model_ids,
                messages=messages,
                allowed_tools=allowed_tools,
            ):
                yield result

    async def _run_streaming(
        self,
        run_id: str,
        api: Any,
        model_ids: list[str],
        messages: list[Message],
        allowed_tools: set[str],
    ) -> AsyncGenerator[AgentRunResult, None]:
        """Run with streaming output and tool calling support."""
        # Build LLM tools from allowed tool names
        llm_tools = await build_llm_tools(api, allowed_tools)

        try:
            caller = StreamingModelCaller(
                api=api,
                model_ids=model_ids,
                messages=messages,
                tools=llm_tools,
            )

            # Stream LLM response
            async for chunk, is_delta in caller.stream():
                if chunk.content:  # Only yield non-empty chunks
                    yield AgentRunResult.message_delta(run_id, chunk)

            # Check for tool calls after streaming
            tool_calls = caller.get_tool_calls()

            if tool_calls:
                # Run tool calling loop
                committed_model_id = caller.get_committed_model_id()
                if not committed_model_id:
                    yield AgentRunResult.run_failed(
                        run_id,
                        error="No model committed for tool loop",
                        code="runner.no_model",
                    )
                    return

                accumulated_content = caller.get_accumulated_content()

                # Add assistant message with tool calls to messages
                assistant_msg = _build_assistant_tool_call_message(
                    accumulated_content,
                    tool_calls,
                )
                updated_messages = messages + [assistant_msg]

                # Execute tool calls
                tool_loop = ToolCallLoop(
                    api=api,
                    allowed_tools=allowed_tools,
                    max_iterations=DEFAULT_MAX_TOOL_ITERATIONS,
                )

                pending_tool_calls = tool_calls
                current_messages = updated_messages
                visible_content_prefix = accumulated_content

                while pending_tool_calls and tool_loop.check_iteration_limit():
                    tool_loop.increment_iteration()

                    # Execute each tool call
                    for tool_call in pending_tool_calls:
                        tool_call_id = tool_call.get("id", "")
                        tool_name = tool_call.get("function_name", "")
                        parameters = {}
                        try:
                            args_str = tool_call.get("function_arguments", "{}")
                            parameters = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            parameters = {}

                        # Yield tool.call.started
                        yield AgentRunResult.tool_call_started(
                            run_id,
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            parameters=parameters,
                        )

                        # Execute tool
                        result, error = await tool_loop.execute_tool_call(tool_call)

                        # Yield tool.call.completed
                        yield AgentRunResult.tool_call_completed(
                            run_id,
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            result=result,
                            error=error,
                        )

                        # Build tool result message
                        tool_msg = build_tool_call_message(
                            tool_call_id, tool_name, error if error else result, is_error=bool(error)
                        )
                        current_messages.append(tool_msg)

                    # Call LLM again with tool results
                    try:
                        tool_caller = StreamingModelCaller(
                            api=api,
                            model_ids=[committed_model_id],
                            messages=current_messages,
                            tools=llm_tools,
                        )

                        async for chunk, is_delta in tool_caller.stream():
                            if chunk.content:
                                chunk = chunk.model_copy(deep=True)
                                if isinstance(chunk.content, str):
                                    chunk.content = visible_content_prefix + chunk.content
                                yield AgentRunResult.message_delta(run_id, chunk)

                        # Check for more tool calls
                        pending_tool_calls = tool_caller.get_tool_calls()
                        visible_content_prefix += tool_caller.get_accumulated_content()
                        if pending_tool_calls:
                            # Update messages with assistant response
                            current_messages.append(
                                _build_assistant_tool_call_message(
                                    tool_caller.get_accumulated_content(),
                                    pending_tool_calls,
                                )
                            )

                    except ModelCallError as e:
                        yield AgentRunResult.run_failed(
                            run_id,
                            error=str(e),
                            code="runner.tool_loop_error",
                            retryable=e.retryable,
                        )
                        return

                # Check if we hit iteration limit
                if pending_tool_calls and not tool_loop.check_iteration_limit():
                    yield AgentRunResult.run_failed(
                        run_id,
                        error=f"Tool call iteration limit reached ({DEFAULT_MAX_TOOL_ITERATIONS})",
                        code="runner.tool_loop_limit",
                    )
                    return

            # Successful completion
            yield AgentRunResult.run_completed(run_id, finish_reason="stop")

        except ModelCallError as e:
            yield AgentRunResult.run_failed(
                run_id,
                error=str(e),
                code="runner.llm_error",
                retryable=e.retryable,
            )
        except Exception as e:
            yield AgentRunResult.run_failed(
                run_id,
                error=str(e),
                code="runner.error",
            )

    async def _run_non_streaming(
        self,
        run_id: str,
        api: Any,
        model_ids: list[str],
        messages: list[Message],
        allowed_tools: set[str],
    ) -> AsyncGenerator[AgentRunResult, None]:
        """Run with non-streaming output and tool calling support."""
        # Build LLM tools from allowed tool names
        llm_tools = await build_llm_tools(api, allowed_tools)

        try:
            # Non-streaming invocation with fallback
            response, committed_model_id = await invoke_with_fallback(
                api=api,
                model_ids=model_ids,
                messages=messages,
                tools=llm_tools,
            )

            # Check for tool calls
            tool_calls = response.tool_calls

            if tool_calls:
                # Convert to internal format
                pending_tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function_name": tc.function.name if tc.function else "",
                        "function_arguments": tc.function.arguments if tc.function else "",
                    }
                    for tc in tool_calls
                ]

                current_messages = messages + [response]

                # Execute tool calls in a loop
                tool_loop = ToolCallLoop(
                    api=api,
                    allowed_tools=allowed_tools,
                    max_iterations=DEFAULT_MAX_TOOL_ITERATIONS,
                )

                while pending_tool_calls and tool_loop.check_iteration_limit():
                    tool_loop.increment_iteration()

                    for tool_call in pending_tool_calls:
                        tool_call_id = tool_call.get("id", "")
                        tool_name = tool_call.get("function_name", "")
                        parameters = {}
                        try:
                            args_str = tool_call.get("function_arguments", "{}")
                            parameters = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            parameters = {}

                        yield AgentRunResult.tool_call_started(
                            run_id,
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            parameters=parameters,
                        )

                        result, error = await tool_loop.execute_tool_call(tool_call)

                        yield AgentRunResult.tool_call_completed(
                            run_id,
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            result=result,
                            error=error,
                        )

                        tool_msg = build_tool_call_message(
                            tool_call_id, tool_name, error if error else result, is_error=bool(error)
                        )
                        current_messages.append(tool_msg)

                    # Call LLM again with tool results using the committed
                    # model from the successful first call.
                    try:
                        response = await api.invoke_llm(
                            llm_model_uuid=committed_model_id,
                            messages=current_messages,
                            funcs=llm_tools,
                        )

                        # Update pending tool calls
                        if response.tool_calls:
                            pending_tool_calls = [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function_name": tc.function.name if tc.function else "",
                                    "function_arguments": tc.function.arguments if tc.function else "",
                                }
                                for tc in response.tool_calls
                            ]
                            current_messages.append(response)
                        else:
                            pending_tool_calls = None

                    except Exception as e:
                        yield AgentRunResult.run_failed(
                            run_id,
                            error=f"LLM call in tool loop failed: {e}",
                            code="runner.tool_loop_error",
                            retryable=True,
                        )
                        return

                if pending_tool_calls and not tool_loop.check_iteration_limit():
                    yield AgentRunResult.run_failed(
                        run_id,
                        error=f"Tool call iteration limit reached ({DEFAULT_MAX_TOOL_ITERATIONS})",
                        code="runner.tool_loop_limit",
                    )
                    return

                # Yield final message
                yield AgentRunResult.message_completed(run_id, response)
            else:
                # No tool calls - yield completed message
                yield AgentRunResult.message_completed(run_id, response)

            yield AgentRunResult.run_completed(run_id, finish_reason="stop")

        except ModelCallError as e:
            yield AgentRunResult.run_failed(
                run_id,
                error=str(e),
                code="runner.llm_error",
                retryable=e.retryable,
            )
        except Exception as e:
            yield AgentRunResult.run_failed(
                run_id,
                error=str(e),
                code="runner.error",
            )
