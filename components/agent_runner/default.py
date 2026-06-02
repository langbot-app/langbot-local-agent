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
    AgentRunnerCapabilities,
    AgentRunnerPermissions,
    AgentRunResult,
)
from langbot_plugin.api.entities.builtin.provider.message import Message

from pkg.agent_core import (
    AgentLoop,
    AgentLoopEvent,
    AgentLoopEventType,
    LangBotModelAdapter,
    LangBotToolExecutor,
)
from pkg.config import get_max_tool_iterations, parse_model_config
from pkg.context_pipeline import ContextAssembler
from pkg.model_calling import build_llm_tools


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
        max_tool_iterations = get_max_tool_iterations(ctx.config)

        # Build the model context through the runner-owned context pipeline.
        context_assembly = await ContextAssembler(api, ctx).assemble()
        messages = context_assembly.messages

        # Prefer host runtime capability so non-streaming adapters keep the
        # same behavior as the original built-in local-agent runner.
        use_streaming = ctx.config.get("streaming")
        if use_streaming is None:
            use_streaming = ctx.runtime.metadata.get("streaming_supported", True)

        if use_streaming:
            async for result in self._run_agent_loop(
                run_id=ctx.run_id,
                api=api,
                model_ids=model_ids,
                messages=messages,
                allowed_tools=allowed_tools,
                streaming=True,
                max_tool_iterations=max_tool_iterations,
            ):
                yield result
        else:
            async for result in self._run_agent_loop(
                run_id=ctx.run_id,
                api=api,
                model_ids=model_ids,
                messages=messages,
                allowed_tools=allowed_tools,
                streaming=False,
                max_tool_iterations=max_tool_iterations,
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
    ) -> AsyncGenerator[AgentRunResult, None]:
        """Run the LangBot-native Pi-style agent loop."""
        llm_tools = await build_llm_tools(api, allowed_tools)
        loop = AgentLoop(
            model_adapter=LangBotModelAdapter(api),
            tool_executor=LangBotToolExecutor(api, allowed_tools),
            model_ids=model_ids,
            messages=messages,
            tools=llm_tools,
            streaming=streaming,
            max_tool_iterations=max_tool_iterations,
        )

        final_message: Message | None = None
        async for event in loop.run():
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
