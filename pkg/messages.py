"""Message building utilities for local-agent runner."""

from __future__ import annotations

import typing

from langbot_plugin.api.entities.builtin.provider.message import Message


def build_messages(
    prompt_config: list[dict[str, typing.Any]] | None,
    history_messages: list[Message],
    user_text: str,
    max_round: int,
    rag_context: str | None = None,
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

    # Add system prompt from config
    if prompt_config:
        for prompt_item in prompt_config:
            if isinstance(prompt_item, dict):
                role = prompt_item.get("role", "system")
                content = prompt_item.get("content", "")
                if content and isinstance(content, str):
                    messages.append(Message(role=role, content=content))

    # Truncate history if exceeds max-round
    # Each round = 1 user + 1 assistant message
    truncated_history = history_messages
    if len(history_messages) > max_round * 2:
        truncated_history = history_messages[-(max_round * 2):]

    messages.extend(truncated_history)

    # Add current user input
    final_user_text = user_text
    if rag_context:
        final_user_text = _build_rag_prompt(rag_context, user_text)

    if final_user_text:
        messages.append(Message(role="user", content=final_user_text))

    return messages


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
