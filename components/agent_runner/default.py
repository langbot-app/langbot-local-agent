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
    LangBotModelAdapter,
)
from pkg.run_assembly import AgentRunAssembler, AgentRunAssembly, NoAuthorizedModelError


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

        try:
            assembly = await AgentRunAssembler(api, ctx).assemble()
        except NoAuthorizedModelError:
            yield AgentRunResult.run_failed(
                ctx.run_id,
                error="No authorized model for local-agent",
                code="runner.no_model",
            )
            return

        async for result in self._run_agent_loop(
            run_id=ctx.run_id,
            api=api,
            assembly=assembly,
        ):
            yield result

    async def _run_agent_loop(
        self,
        run_id: str,
        api: Any,
        assembly: AgentRunAssembly,
    ) -> AsyncGenerator[AgentRunResult, None]:
        """Run the LangBot-native Pi-style agent loop."""
        loop = AgentLoop(
            model_adapter=LangBotModelAdapter(api, remove_think=assembly.remove_think),
            tool_executor=assembly.tool_executor,
            model_ids=assembly.model_ids,
            messages=assembly.messages,
            tools=assembly.tools,
            streaming=assembly.streaming,
            max_tool_iterations=assembly.max_tool_iterations,
            tool_execution_mode=assembly.tool_execution_mode,
            hooks=assembly.hooks,
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

            result = self._loop_event_to_result(run_id, event, streaming=assembly.streaming)
            if result is not None:
                yield result
                if result.type.value == "run.failed":
                    return

            if (
                not assembly.streaming
                and event.type == AgentLoopEventType.MESSAGE_END
                and event.message is not None
                and event.message.role == "assistant"
                and not event.message.tool_calls
            ):
                final_message = event.message

            if event.type == AgentLoopEventType.AGENT_END:
                if not assembly.streaming and final_message is not None:
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
