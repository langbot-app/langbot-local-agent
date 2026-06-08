"""Run-scoped capability assembly for the local-agent runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langbot_plugin.api.entities.builtin.agent_runner import AgentRunContext
from langbot_plugin.api.entities.builtin.provider.message import Message
from langbot_plugin.api.entities.builtin.resource.tool import LLMTool

from pkg.agent_core import AgentLoopHooks, LangBotContextHooks, LangBotToolExecutor
from pkg.config import (
    get_max_tool_iterations,
    get_max_tool_result_artifact_bytes,
    get_max_tool_result_chars,
    get_remove_think,
    get_tool_execution_mode,
    parse_model_config,
)
from pkg.context_pipeline import ContextAssembler, ContextBudget
from pkg.model_calling import INTERNAL_ARTIFACT_READ_TOOL_NAME, build_artifact_read_tool, build_llm_tools


class NoAuthorizedModelError(Exception):
    """Raised when no configured model is available for this run."""


@dataclass(frozen=True)
class AgentRunAssembly:
    """Everything the LangBot adapter needs to construct an AgentLoop."""

    model_ids: list[str]
    messages: list[Message]
    tools: list[LLMTool]
    tool_executor: LangBotToolExecutor
    hooks: AgentLoopHooks
    streaming: bool
    max_tool_iterations: int
    tool_execution_mode: str
    remove_think: bool


class AgentRunAssembler:
    """Assemble LangBot-authorized run capabilities into AgentLoop inputs."""

    def __init__(self, api: Any, ctx: AgentRunContext):
        self.api = api
        self.ctx = ctx

    async def assemble(self) -> AgentRunAssembly:
        model_ids = self._resolve_model_ids()
        if not model_ids:
            raise NoAuthorizedModelError("No authorized model for local-agent")

        allowed_tools = self._allowed_tool_names()
        artifact_read_available = self._artifact_read_available()
        if artifact_read_available:
            allowed_tools.add(INTERNAL_ARTIFACT_READ_TOOL_NAME)

        max_tool_iterations = get_max_tool_iterations(self.ctx.config)
        max_tool_result_chars = get_max_tool_result_chars(self.ctx.config)
        max_tool_result_artifact_bytes = get_max_tool_result_artifact_bytes(self.ctx.config)
        tool_execution_mode = get_tool_execution_mode(self.ctx.config)
        remove_think = get_remove_think(self.ctx.config)
        context_budget = ContextBudget.from_context(self.ctx)

        context_assembly = await ContextAssembler(self.api, self.ctx, budget=context_budget).assemble()
        tools = await self._build_tools(allowed_tools, artifact_read_available)

        return AgentRunAssembly(
            model_ids=model_ids,
            messages=context_assembly.messages,
            tools=tools,
            tool_executor=LangBotToolExecutor(
                self.api,
                allowed_tools,
                max_result_chars=max_tool_result_chars,
                max_artifact_bytes=max_tool_result_artifact_bytes,
                artifact_read_available=artifact_read_available,
            ),
            hooks=LangBotContextHooks(context_budget),
            streaming=self._streaming_supported(),
            max_tool_iterations=max_tool_iterations,
            tool_execution_mode=tool_execution_mode,
            remove_think=remove_think,
        )

    def _resolve_model_ids(self) -> list[str]:
        allowed_model_ids = {model.model_id for model in self.api.get_allowed_models()}
        return parse_model_config(self.ctx.config.get("model"), allowed_model_ids)

    def _allowed_tool_names(self) -> set[str]:
        return {tool.tool_name for tool in self.api.get_allowed_tools()}

    def _artifact_read_available(self) -> bool:
        return bool(getattr(self.ctx.context.available_apis, "artifact_read", False))

    def _streaming_supported(self) -> bool:
        metadata = getattr(self.ctx.runtime, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            return True
        return bool(metadata.get("streaming_supported", True))

    async def _build_tools(self, allowed_tools: set[str], artifact_read_available: bool) -> list[LLMTool]:
        host_tool_names = {tool_name for tool_name in allowed_tools if tool_name != INTERNAL_ARTIFACT_READ_TOOL_NAME}
        tools = await build_llm_tools(self.api, host_tool_names)
        if artifact_read_available:
            tools.append(build_artifact_read_tool())
        return tools


__all__ = ["AgentRunAssembler", "AgentRunAssembly", "NoAuthorizedModelError"]
