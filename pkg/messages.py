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
    rag_context: str | None = None,
) -> Message | None:
    """Build the current user message, preserving structured/multimodal input."""
    final_user_text = _build_rag_prompt(rag_context, user_text) if rag_context else user_text

    if input_contents:
        contents = [content.model_copy(deep=True) for content in input_contents]
        if rag_context:
            for content in contents:
                if content.type == "text":
                    content.text = final_user_text
                    break
            else:
                contents.insert(0, ContentElement.from_text(final_user_text))
        return Message(role="user", content=contents)

    if final_user_text:
        return Message(role="user", content=final_user_text)

    return None


def _build_rag_prompt(rag_context: str, user_text: str) -> str:
    """Build user message with RAG context.

    Matches original localagent template:
    - Instructions
    - <context>...</context>
    - <user_message>...</user_message>
    """
    return f"""The following are relevant context entries retrieved from the knowledge base.
Please use them to answer the user's message.
Respond in the same language as the user's input.

<context>
{rag_context}
</context>

<user_message>
{user_text}
</user_message>"""


def format_rag_results(results: list[dict[str, typing.Any]]) -> str:
    """Format knowledge base retrieval results for context.

    Args:
        results: List of retrieval result entries from API

    Returns:
        Formatted context string
    """
    if not results:
        return ""

    texts: list[str] = []
    idx = 1

    for entry in results:
        content = entry.get("content", [])
        if isinstance(content, list):
            for ce in content:
                if isinstance(ce, dict) and ce.get("type") == "text":
                    text = ce.get("text", "")
                    if text:
                        texts.append(f"[{idx}] {text}")
                        idx += 1
        elif isinstance(content, str) and content:
            texts.append(f"[{idx}] {content}")
            idx += 1

    return "\n\n".join(texts)
