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
from pkg.config import get_knowledge_base_ids, get_rerank_config, parse_model_config
from pkg.messages import build_messages, get_effective_prompt_config
from pkg.model_calling import build_llm_tools
from pkg.rag import retrieve_from_knowledge_bases
from pkg.tool_loop import DEFAULT_MAX_TOOL_ITERATIONS

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

        # Build messages for LLM. Pipeline adapter runs provide the host
        # effective prompt after PromptPreProcessing in adapter.extra.prompt.
        prompt_config = get_effective_prompt_config(ctx)
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
            async for result in self._run_agent_loop(
                run_id=ctx.run_id,
                api=api,
                model_ids=model_ids,
                messages=messages,
                allowed_tools=allowed_tools,
                streaming=True,
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
            max_tool_iterations=DEFAULT_MAX_TOOL_ITERATIONS,
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
