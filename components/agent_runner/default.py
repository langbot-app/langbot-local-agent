"""Local Agent default runner implementation.

Supports:
- Model fallback (primary + fallbacks)
- Streaming and non-streaming
- Tool calling loop with max iterations
- Knowledge retrieval with permission validation
- Protocol v1 AgentRunResult output
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from langbot_plugin.api.definition.components.agent_runner.runner import AgentRunner
from langbot_plugin.api.entities.builtin.agent_runner import (
    AgentRunContext,
    AgentRunResult,
)
from langbot_plugin.api.entities.builtin.provider.message import Message

from pkg.agent_core import (
    AgentLoop,
    AgentLoopEvent,
    AgentLoopEventType,
    LangBotContextHooks,
    LangBotModelAdapter,
    LangBotToolExecutor,
)
from pkg.config import (
    get_max_tool_iterations,
    get_max_tool_result_artifact_bytes,
    get_max_tool_result_chars,
    parse_model_config,
)
from pkg.context_pipeline import ContextAssembler, ContextBudget
from pkg.model_calling import INTERNAL_ARTIFACT_READ_TOOL_NAME, build_artifact_read_tool, build_llm_tools


class DefaultAgentRunner(AgentRunner):
    """Default AgentRunner for Local Agent.

    Full-featured LLM runner with:
    - Model primary/fallback selection
    - Streaming and non-streaming output
    - Tool calling loop
    - Knowledge retrieval (RAG)

    All resource access goes through AgentRunAPIProxy for authorization.
    """

    async def run(self, ctx: AgentRunContext) -> AsyncGenerator[AgentRunResult, None]:
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

        # Parse current model-fallback-selector config.
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
        artifact_read_available = bool(getattr(ctx.context.available_apis, "artifact_read", False))
        if artifact_read_available:
            allowed_tools.add(INTERNAL_ARTIFACT_READ_TOOL_NAME)
        max_tool_iterations = get_max_tool_iterations(ctx.config)
        max_tool_result_chars = get_max_tool_result_chars(ctx.config)
        max_tool_result_artifact_bytes = get_max_tool_result_artifact_bytes(ctx.config)
        context_budget = ContextBudget.from_context(ctx)

        # Build the model context through the runner-owned context pipeline.
        context_assembly = await ContextAssembler(api, ctx, budget=context_budget).assemble()
        messages = context_assembly.messages

        # Prefer host runtime capability so non-streaming adapters keep the
        # same behavior as the original built-in local-agent runner.
        use_streaming = bool(ctx.runtime.metadata.get("streaming_supported", True))

        async for result in self._run_agent_loop(
            run_id=ctx.run_id,
            api=api,
            model_ids=model_ids,
            messages=messages,
            allowed_tools=allowed_tools,
            streaming=use_streaming,
            max_tool_iterations=max_tool_iterations,
            max_tool_result_chars=max_tool_result_chars,
            max_tool_result_artifact_bytes=max_tool_result_artifact_bytes,
            context_budget=context_budget,
            artifact_read_available=artifact_read_available,
        ):
            yield result

    async def _run_agent_loop(
        self,
        run_id: str,
        api: Any,
        model_ids: list[str],
        messages: list[Message],
        allowed_tools: set[str],
        streaming: bool,
        max_tool_iterations: int,
        max_tool_result_chars: int,
        max_tool_result_artifact_bytes: int,
        context_budget: ContextBudget,
        artifact_read_available: bool,
    ) -> AsyncGenerator[AgentRunResult, None]:
        """Run the LangBot-native Pi-style agent loop."""
        host_tool_names = {tool_name for tool_name in allowed_tools if tool_name != INTERNAL_ARTIFACT_READ_TOOL_NAME}
        llm_tools = await build_llm_tools(api, host_tool_names)
        if artifact_read_available:
            llm_tools.append(build_artifact_read_tool())
        loop = AgentLoop(
            model_adapter=LangBotModelAdapter(api),
            tool_executor=LangBotToolExecutor(
                api,
                allowed_tools,
                max_result_chars=max_tool_result_chars,
                max_artifact_bytes=max_tool_result_artifact_bytes,
                artifact_read_available=artifact_read_available,
            ),
            model_ids=model_ids,
            messages=messages,
            tools=llm_tools,
            streaming=streaming,
            max_tool_iterations=max_tool_iterations,
            hooks=LangBotContextHooks(context_budget),
        )

        final_message: Message | None = None
        async for event in loop.run():
            if event.type == AgentLoopEventType.TOOL_EXECUTION_END and event.artifact is not None:
                artifact = event.artifact
                yield AgentRunResult.artifact_created(
                    run_id,
                    artifact_id=artifact.artifact_id,
                    artifact_type=artifact.artifact_type,
                    mime_type=artifact.mime_type,
                    name=artifact.name,
                    size_bytes=artifact.size_bytes,
                    sha256=artifact.sha256,
                    metadata=artifact.metadata,
                    content_base64=artifact.content_base64,
                )

            result = self._loop_event_to_result(run_id, event, streaming=streaming)
            if result is not None:
                yield result
                if result.type.value == "run.failed":
                    return

            if (
                not streaming
                and event.type == AgentLoopEventType.MESSAGE_END
                and event.message is not None
                and event.message.role == "assistant"
                and not event.message.tool_calls
            ):
                final_message = event.message

            if event.type == AgentLoopEventType.AGENT_END:
                if not streaming and final_message is not None:
                    yield AgentRunResult.message_completed(run_id, final_message)
                yield AgentRunResult.run_completed(run_id, finish_reason="stop")

    def _loop_event_to_result(
        self,
        run_id: str,
        event: AgentLoopEvent,
        *,
        streaming: bool,
    ) -> AgentRunResult | None:
        if event.type == AgentLoopEventType.MESSAGE_UPDATE and streaming and event.chunk is not None:
            return AgentRunResult.message_delta(run_id, event.chunk)

        if event.type == AgentLoopEventType.TOOL_EXECUTION_START and event.tool_call_id and event.tool_name:
            return AgentRunResult.tool_call_started(
                run_id,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                parameters=event.parameters,
            )

        if event.type == AgentLoopEventType.TOOL_EXECUTION_END and event.tool_call_id and event.tool_name:
            return AgentRunResult.tool_call_completed(
                run_id,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                result=event.result,
                error=event.error,
            )

        if event.type == AgentLoopEventType.RUN_FAILED:
            return AgentRunResult.run_failed(
                run_id,
                error=event.error or "Agent loop failed",
                code=event.code or "runner.error",
                retryable=event.retryable,
            )

        return None
