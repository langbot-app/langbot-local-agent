"""Message building utilities for local-agent runner."""

from __future__ import annotations

import typing

from langbot_plugin.api.entities.builtin.provider.message import ContentElement, Message


def get_effective_prompt_config(ctx: typing.Any) -> list[dict[str, typing.Any]]:
    """Return the prompt that should be sent to the model.

    Host effective prompts should be pulled through AgentRunAPIProxy.get_prompt()
    before calling this helper. This helper only returns the static binding
    fallback from ctx.config.prompt.
    """
    config = getattr(ctx, "config", None)
    if isinstance(config, dict):
        prompt = config.get("prompt", [])
        if isinstance(prompt, list):
            return prompt

    return []


def build_prompt_messages(
    prompt_config: list[dict[str, typing.Any]] | None,
) -> list[Message]:
    """Build model prompt messages from runner/host prompt config."""
    messages: list[Message] = []

    if prompt_config:
        for prompt_item in prompt_config:
            if isinstance(prompt_item, dict):
                role = prompt_item.get("role", "system")
                content = prompt_item.get("content", "")
                if content and isinstance(content, str):
                    messages.append(Message(role=role, content=content))

    return messages


def build_user_message(
    user_text: str,
    input_contents: list[ContentElement] | None = None,
) -> Message | None:
    """Build the current user message, preserving structured/multimodal input."""
    if input_contents:
        contents = [content.model_copy(deep=True) for content in input_contents]
        return Message(role="user", content=contents)

    if user_text:
        return Message(role="user", content=user_text)

    return None


def build_rag_context_message(rag_context: str | None) -> Message | None:
    """Build a separate model-facing RAG message.

    The runner keeps RAG separate from the current user input internally and
    renders only the chunk text to the model. Source metadata stays out of the
    prompt so repeated equivalent retrievals keep a stable rendered shape.
    """
    if not rag_context:
        return None

    return Message(
        role="user",
        content=f"""Retrieved context for the next user message.
Use it only when it is relevant.

<retrieved_context>
{rag_context}
</retrieved_context>""",
    )
