"""Message building utilities for local-agent runner."""

from __future__ import annotations

import typing

from langbot_plugin.api.entities.builtin.provider.message import ContentElement, Message


def build_messages(
    prompt_config: list[dict[str, typing.Any]] | None,
    history_messages: list[Message],
    user_text: str,
    max_round: int,
    rag_context: str | None = None,
    prompt_messages: list[Message] | None = None,
    input_contents: list[ContentElement] | None = None,
) -> list[Message]:
    """Build messages list for LLM invocation.

    Structure:
    1. System prompt from config.prompt
    2. Historical messages (truncated by max-round)
    3. Current user input (with RAG context if provided)

    Args:
        prompt_config: System prompt configuration
        history_messages: Conversation history
        user_text: Current user input text
        max_round: Maximum conversation rounds to keep
        rag_context: Optional RAG context to prepend to user message

    Returns:
        List of Message objects ready for LLM
    """
    messages: list[Message] = []

    # Prefer the effective host-provided prompt. It includes changes made by
    # host preprocessing and prompt-related plugin events.
    if prompt_messages:
        messages.extend([msg.model_copy(deep=True) for msg in prompt_messages])
    elif prompt_config:
        for prompt_item in prompt_config:
            if isinstance(prompt_item, dict):
                role = prompt_item.get("role", "system")
                content = prompt_item.get("content", "")
                if content and isinstance(content, str):
                    messages.append(Message(role=role, content=content))

    truncated_history = _truncate_history_by_round(history_messages, max_round)

    messages.extend(truncated_history)

    user_message = build_user_message(
        user_text=user_text,
        input_contents=input_contents,
        rag_context=rag_context,
    )
    if user_message is not None:
        messages.append(user_message)

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


def _truncate_history_by_round(history_messages: list[Message], max_round: int) -> list[Message]:
    """Truncate history using the same user-round semantics as LangBot host."""
    if max_round < 1:
        return history_messages

    temp_messages: list[Message] = []
    current_round = 0

    for msg in reversed(history_messages):
        if current_round < max_round:
            temp_messages.append(msg)
            if msg.role == "user":
                current_round += 1
        else:
            break

    return list(reversed(temp_messages))


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
