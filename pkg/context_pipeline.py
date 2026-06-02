"""Context assembly, budgeting, and deterministic compaction for local-agent."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from langbot_plugin.api.entities.builtin.agent_runner import AgentRunContext
from langbot_plugin.api.entities.builtin.provider.message import Message

from pkg.config import get_knowledge_base_ids, get_rerank_config
from pkg.messages import build_prompt_messages, build_user_message, get_effective_prompt_config
from pkg.rag import retrieve_from_knowledge_bases

DEFAULT_HISTORY_FETCH_LIMIT = 50
DEFAULT_CONTEXT_WINDOW_CHARS = 32000
DEFAULT_CONTEXT_RESERVE_CHARS = 8000
DEFAULT_CONTEXT_KEEP_RECENT_CHARS = 12000
DEFAULT_CONTEXT_SUMMARY_CHARS = 4000

ESTIMATED_ATTACHMENT_CHARS = 4800


@dataclass(frozen=True)
class ContextBudget:
    """Character-based context budget used before token metadata is available."""

    window_chars: int = DEFAULT_CONTEXT_WINDOW_CHARS
    reserve_chars: int = DEFAULT_CONTEXT_RESERVE_CHARS
    keep_recent_chars: int = DEFAULT_CONTEXT_KEEP_RECENT_CHARS
    summary_chars: int = DEFAULT_CONTEXT_SUMMARY_CHARS
    history_fetch_limit: int = DEFAULT_HISTORY_FETCH_LIMIT

    @classmethod
    def from_config(cls, config: dict[str, typing.Any]) -> "ContextBudget":
        return cls(
            window_chars=_config_int(config, "context-window-chars", DEFAULT_CONTEXT_WINDOW_CHARS, minimum=0),
            reserve_chars=_config_int(config, "context-reserve-chars", DEFAULT_CONTEXT_RESERVE_CHARS, minimum=0),
            keep_recent_chars=_config_int(
                config,
                "context-keep-recent-chars",
                DEFAULT_CONTEXT_KEEP_RECENT_CHARS,
                minimum=0,
            ),
            summary_chars=_config_int(config, "context-summary-chars", DEFAULT_CONTEXT_SUMMARY_CHARS, minimum=0),
            history_fetch_limit=_config_int(
                config,
                "context-history-fetch-limit",
                DEFAULT_HISTORY_FETCH_LIMIT,
                minimum=1,
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.window_chars > 0

    @property
    def input_chars(self) -> int:
        if not self.enabled:
            return 0
        return max(self.window_chars - self.reserve_chars, 1)


@dataclass(frozen=True)
class ContextAssembly:
    """Final model context plus compaction diagnostics."""

    messages: list[Message]
    compacted: bool
    chars_before: int
    chars_after: int
    summary_message: Message | None = None


class ContextAssembler:
    """Build the model-facing context for one AgentRunner run."""

    def __init__(
        self,
        api: typing.Any,
        ctx: AgentRunContext,
        budget: ContextBudget | None = None,
    ):
        self.api = api
        self.ctx = ctx
        self.budget = budget or ContextBudget.from_config(ctx.config)

    async def assemble(self) -> ContextAssembly:
        user_text = self.ctx.input.to_text()
        rag_context = await self._retrieve_rag_context(user_text)
        history_messages = await self._get_history_messages()

        prompt_messages = build_prompt_messages(get_effective_prompt_config(self.ctx))
        current_messages = []
        user_message = build_user_message(
            user_text=user_text,
            input_contents=self.ctx.input.contents,
            rag_context=rag_context if rag_context else None,
        )
        if user_message is not None:
            current_messages.append(user_message)

        return ContextCompactor(self.budget).compact(
            prompt_messages=prompt_messages,
            history_messages=history_messages,
            current_messages=current_messages,
        )

    async def _retrieve_rag_context(self, user_text: str) -> str:
        allowed_kb_ids = set(kb.kb_id for kb in self.api.get_allowed_knowledge_bases())
        kb_ids = get_knowledge_base_ids(self.ctx.config, allowed_kb_ids)
        if not kb_ids or not user_text:
            return ""

        rerank_model_id, rerank_top_k = get_rerank_config(self.ctx.config)
        return await retrieve_from_knowledge_bases(
            api=self.api,
            kb_ids=kb_ids,
            query_text=user_text,
            top_k=5,
            rerank_model_id=rerank_model_id,
            rerank_top_k=rerank_top_k,
        )

    async def _get_history_messages(self) -> list[Message]:
        context = self.ctx.context
        if not context.available_apis.history_page or not context.conversation_id:
            return []

        page = await self.api.history_page(
            conversation_id=context.conversation_id,
            limit=self.budget.history_fetch_limit,
            direction="backward",
            include_artifacts=True,
        )

        messages: list[Message] = []
        for item in reversed(page.get("items", [])):
            if not isinstance(item, dict):
                continue
            message = _message_from_transcript_item(item, self.ctx.event.event_id)
            if message is not None:
                messages.append(message)

        return messages


class ContextCompactor:
    """Compact old history into a summary message while keeping recent context."""

    def __init__(self, budget: ContextBudget):
        self.budget = budget

    def compact(
        self,
        *,
        prompt_messages: list[Message],
        history_messages: list[Message],
        current_messages: list[Message],
    ) -> ContextAssembly:
        prompt = _copy_messages(prompt_messages)
        history = _copy_messages(history_messages)
        current = _copy_messages(current_messages)
        original_messages = prompt + history + current
        chars_before = estimate_messages_chars(original_messages)

        if not self.budget.enabled or chars_before <= self.budget.input_chars:
            return ContextAssembly(
                messages=original_messages,
                compacted=False,
                chars_before=chars_before,
                chars_after=chars_before,
            )

        history_budget = max(self.budget.input_chars - estimate_messages_chars(prompt + current), 0)
        if not history or history_budget <= 0:
            messages = prompt + current
            return ContextAssembly(
                messages=messages,
                compacted=bool(history),
                chars_before=chars_before,
                chars_after=estimate_messages_chars(messages),
            )

        summary_budget = self._summary_budget(history_budget)
        recent_budget = max(history_budget - summary_budget, 0)
        first_kept_index = self._select_first_kept_index(history, recent_budget)

        omitted = history[:first_kept_index]
        recent = history[first_kept_index:]
        summary_message = self._build_summary_message(omitted, summary_budget)
        compacted_history = ([summary_message] if summary_message is not None else []) + recent
        messages = prompt + compacted_history + current

        return ContextAssembly(
            messages=messages,
            compacted=bool(omitted),
            chars_before=chars_before,
            chars_after=estimate_messages_chars(messages),
            summary_message=summary_message,
        )

    def _summary_budget(self, history_budget: int) -> int:
        if self.budget.summary_chars <= 0:
            return 0
        if history_budget > self.budget.keep_recent_chars:
            return min(self.budget.summary_chars, history_budget - self.budget.keep_recent_chars)
        if history_budget >= 120:
            return min(self.budget.summary_chars, max(80, history_budget // 3))
        return 0

    def _select_first_kept_index(self, history: list[Message], recent_budget: int) -> int:
        if recent_budget <= 0:
            return len(history)

        total = 0
        first_kept_index = len(history)
        for index in range(len(history) - 1, -1, -1):
            message_chars = estimate_message_chars(history[index])
            if first_kept_index < len(history) and total + message_chars > recent_budget:
                break
            if first_kept_index == len(history) and message_chars > recent_budget:
                break
            total += message_chars
            first_kept_index = index

        return _move_cut_before_tool_result(history, first_kept_index)

    def _build_summary_message(self, omitted: list[Message], summary_budget: int) -> Message | None:
        if not omitted or summary_budget <= 0:
            return None

        summary = summarize_messages(omitted, summary_budget)
        if not summary:
            return None
        return Message(role="system", content=summary)


def summarize_messages(messages: list[Message], max_chars: int) -> str:
    """Create a deterministic compacted-history summary.

    This is intentionally not a model call yet. It provides the same structural
    shape as Pi compaction, while future iterations can replace the summary
    generator with an LLM + host state checkpoint.
    """
    if max_chars <= 0:
        return ""

    lines = ["<conversation_summary>"]
    for index, message in enumerate(messages, start=1):
        text = message_to_text(message)
        if not text and message.tool_calls:
            text = "; ".join(tool_call.function.name for tool_call in message.tool_calls if tool_call.function)

        remaining = max_chars - len("\n".join(lines)) - len("\n") - len("\n</conversation_summary>")
        if remaining <= 0:
            break
        prefix = f"{index}. {message.role}: "
        if remaining <= len(prefix):
            break
        line = prefix + _truncate_text(text, remaining - len(prefix))
        candidate = "\n".join(lines + [line, "</conversation_summary>"])
        if len(candidate) > max_chars:
            break
        lines.append(line)

    count_line = f"Compacted {len(messages)} older messages."
    with_count = "\n".join([lines[0], count_line, *lines[1:], "</conversation_summary>"])
    if len(with_count) <= max_chars:
        return with_count

    closing_candidate = "\n".join(lines + ["</conversation_summary>"])
    if len(closing_candidate) <= max_chars:
        return closing_candidate

    return _truncate_text("\n".join(lines), max_chars)


def estimate_messages_chars(messages: list[Message]) -> int:
    return sum(estimate_message_chars(message) for message in messages)


def estimate_message_chars(message: Message) -> int:
    chars = len(message.role) + 8
    chars += _content_chars(message.content)
    if message.name:
        chars += len(message.name)
    if message.tool_call_id:
        chars += len(message.tool_call_id)
    if message.tool_calls:
        for tool_call in message.tool_calls:
            chars += len(tool_call.id) + len(tool_call.type)
            if tool_call.function:
                chars += len(tool_call.function.name) + len(tool_call.function.arguments)
    return chars


def message_to_text(message: Message) -> str:
    parts: list[str] = []
    content = message.content
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if item.type == "text" and item.text:
                parts.append(item.text)
            elif item.type.startswith("image"):
                parts.append("[image]")
            elif item.type.startswith("file"):
                parts.append(f"[file:{item.file_name or 'unnamed'}]")

    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function:
                parts.append(f"[tool_call:{tool_call.function.name} {tool_call.function.arguments}]")

    return "\n".join(parts)


def _message_from_transcript_item(item: dict[str, typing.Any], current_event_id: str) -> Message | None:
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


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [message.model_copy(deep=True) for message in messages]


def _content_chars(content: typing.Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        chars = 0
        for item in content:
            if item.type == "text" and item.text:
                chars += len(item.text)
            elif item.type.startswith(("image", "file")):
                chars += ESTIMATED_ATTACHMENT_CHARS
        return chars
    return 0


def _move_cut_before_tool_result(history: list[Message], first_kept_index: int) -> int:
    while first_kept_index > 0 and first_kept_index < len(history) and history[first_kept_index].role == "tool":
        first_kept_index -= 1
    return first_kept_index


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 32:
        return text[:max_chars]

    omitted = len(text) - max_chars
    suffix = f"\n[... {omitted} characters compacted]"
    return text[: max_chars - len(suffix)] + suffix


def _config_int(
    config: dict[str, typing.Any],
    key: str,
    default: int,
    *,
    minimum: int,
) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        return default
    if not isinstance(value, int):
        return default
    if value < minimum:
        return default
    return value
