from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langbot_plugin.api.entities.builtin.agent_runner.resources import AgentResources, ModelResource, ToolResource

from pkg.agent_core import LangBotContextHooks, LangBotToolExecutor
from pkg.model_calling import INTERNAL_ARTIFACT_READ_TOOL_NAME
from pkg.run_assembly import AgentRunAssembler, NoAuthorizedModelError
from tests.test_runner import FakeAgentRunAPIProxy, make_context


@pytest.mark.asyncio
async def test_assembler_builds_authorized_loop_inputs_with_artifact_read_tool() -> None:
    api = FakeAgentRunAPIProxy(
        models=[ModelResource(model_id="model-primary"), ModelResource(model_id="model-fallback")],
        tools=[ToolResource(tool_name="qa_plugin_echo")],
    )
    ctx = make_context(
        config={
            "model": {"primary": "model-primary", "fallbacks": ["model-missing", "model-fallback"]},
            "prompt": [{"role": "system", "content": "Static prompt"}],
            "remove-think": True,
            "max-tool-iterations": 7,
            "max-tool-result-chars": 32,
            "max-tool-result-artifact-bytes": 1024,
            "tool-execution-mode": "serial",
        },
        resources=AgentResources(
            models=[ModelResource(model_id="model-primary"), ModelResource(model_id="model-fallback")],
            tools=[ToolResource(tool_name="qa_plugin_echo")],
        ),
        input_text="hello",
        runtime_metadata={"streaming_supported": False},
        artifact_read=True,
    )

    assembly = await AgentRunAssembler(api, ctx).assemble()

    assert assembly.model_ids == ["model-primary", "model-fallback"]
    assert [tool.name for tool in assembly.tools] == ["qa_plugin_echo", INTERNAL_ARTIFACT_READ_TOOL_NAME]
    assert isinstance(assembly.tool_executor, LangBotToolExecutor)
    assert assembly.tool_executor.allowed_tools == {"qa_plugin_echo", INTERNAL_ARTIFACT_READ_TOOL_NAME}
    assert assembly.tool_executor.max_result_chars == 32
    assert assembly.tool_executor.max_artifact_bytes == 1024
    assert assembly.tool_executor.artifact_read_available is True
    assert isinstance(assembly.hooks, LangBotContextHooks)
    assert assembly.streaming is False
    assert assembly.max_tool_iterations == 7
    assert assembly.tool_execution_mode == "serial"
    assert assembly.remove_think is True
    assert [message.role for message in assembly.messages] == ["system", "user"]
    assert assembly.messages[-1].content == "hello"
    api.get_tool_detail.assert_awaited_once_with("qa_plugin_echo")


@pytest.mark.asyncio
async def test_assembler_disables_streaming_when_delivery_does_not_support_it() -> None:
    api = FakeAgentRunAPIProxy(
        models=[ModelResource(model_id="model-primary")],
    )
    ctx = make_context(
        config={"model": {"primary": "model-primary", "fallbacks": []}},
        resources=AgentResources(models=[ModelResource(model_id="model-primary")]),
        runtime_metadata={"streaming_supported": True},
        delivery_supports_streaming=False,
    )

    assembly = await AgentRunAssembler(api, ctx).assemble()

    assert assembly.streaming is False


@pytest.mark.asyncio
async def test_assembler_omits_internal_artifact_tool_when_host_api_is_unavailable() -> None:
    api = FakeAgentRunAPIProxy(
        models=[ModelResource(model_id="model-primary")],
        tools=[ToolResource(tool_name="qa_plugin_echo")],
    )
    ctx = make_context(
        config={"model": {"primary": "model-primary", "fallbacks": []}},
        resources=AgentResources(
            models=[ModelResource(model_id="model-primary")],
            tools=[ToolResource(tool_name="qa_plugin_echo")],
        ),
        artifact_read=False,
    )

    assembly = await AgentRunAssembler(api, ctx).assemble()

    assert [tool.name for tool in assembly.tools] == ["qa_plugin_echo"]
    assert assembly.tool_executor.allowed_tools == {"qa_plugin_echo"}
    assert assembly.tool_executor.artifact_read_available is False


@pytest.mark.asyncio
async def test_assembler_raises_when_configured_models_are_not_authorized() -> None:
    api = FakeAgentRunAPIProxy(models=[ModelResource(model_id="authorized-model")])
    api.get_tool_detail = AsyncMock()
    ctx = make_context(
        config={"model": {"primary": "unauthorized-model", "fallbacks": []}},
        resources=AgentResources(models=[ModelResource(model_id="authorized-model")]),
    )

    with pytest.raises(NoAuthorizedModelError):
        await AgentRunAssembler(api, ctx).assemble()

    api.get_tool_detail.assert_not_awaited()
