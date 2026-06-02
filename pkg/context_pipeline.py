"""Context assembly, budgeting, and deterministic compaction for local-agent."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from langbot_plugin.api.entities.builtin.agent_runner import AgentRunContext
from langbot_plugin.api.entities.builtin.provider.message import Message

from pkg.config import get_knowledge_base_ids, get_rerank_config, get_retrieval_top_k
from pkg.messages import build_prompt_messages, build_user_message, get_effective_prompt_config
from pkg.rag import retrieve_from_knowledge_bases

DEFAULT_HISTORY_FETCH_LIMIT = 50
DEFAULT_CONTEXT_WINDOW_TOKENS = 200_000
DEFAULT_CONTEXT_RESERVE_TOKENS = 16_384
DEFAULT_CONTEXT_KEEP_RECENT_TOKENS = 20_000
DEFAULT_CONTEXT_SUMMARY_TOKENS = 8_000

ESTIMATED_ATTACHMENT_CHARS = 4800
ESTIMATED_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ContextBudget:
    """Token-style context budget with conservative local estimates."""

    window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    reserve_tokens: int = DEFAULT_CONTEXT_RESERVE_TOKENS
    keep_recent_tokens: int = DEFAULT_CONTEXT_KEEP_RECENT_TOKENS
    summary_tokens: int = DEFAULT_CONTEXT_SUMMARY_TOKENS
    history_fetch_limit: int = DEFAULT_HISTORY_FETCH_LIMIT

    @classmethod
    def from_context(cls, ctx: AgentRunContext) -> "ContextBudget":
        config = ctx.config if isinstance(ctx.config, dict) else {}
        runtime = getattr(ctx, "runtime", None)
        metadata = getattr(runtime, "metadata", {}) if runtime is not None else {}
        if not isinstance(metadata, dict):
            metadata = {}

        host_window_tokens = _runtime_context_window_tokens(metadata)
        config_window_tokens = _config_token_value(
            config,
            "context-window-tokens",
            None,
            minimum=0,
        )

        return cls.from_config(
            config,
            window_tokens=host_window_tokens if host_window_tokens is not None else config_window_tokens,
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, typing.Any],
        *,
        window_tokens: int | None = None,
    ) -> "ContextBudget":
        return cls(
            window_tokens=window_tokens
            if window_tokens is not None
            else _config_token_value(
                config,
                "context-window-tokens",
                DEFAULT_CONTEXT_WINDOW_TOKENS,
                minimum=0,
            ),
            reserve_tokens=_config_token_value(
                config,
                "context-reserve-tokens",
                DEFAULT_CONTEXT_RESERVE_TOKENS,
                minimum=0,
            ),
            keep_recent_tokens=_config_token_value(
                config,
                "context-keep-recent-tokens",
                DEFAULT_CONTEXT_KEEP_RECENT_TOKENS,
                minimum=0,
            ),
            summary_tokens=_config_token_value(
                config,
                "context-summary-tokens",
                DEFAULT_CONTEXT_SUMMARY_TOKENS,
                minimum=0,
            ),
            history_fetch_limit=_config_int(
                config,
                "context-history-fetch-limit",
                DEFAULT_HISTORY_FETCH_LIMIT,
                minimum=1,
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.window_tokens > 0

    @property
    def input_tokens(self) -> int:
        if not self.enabled:
            return 0
        return max(self.window_tokens - self.reserve_tokens, 1)


@dataclass(frozen=True)
class ContextAssembly:
    """Final model context plus compaction diagnostics."""

    messages: list[Message]
    compacted: bool
    tokens_before: int
    tokens_after: int
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
        self.budget = budget or ContextBudget.from_context(ctx)

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
            top_k=get_retrieval_top_k(self.ctx.config),
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
        tokens_before = estimate_messages_tokens(original_messages)

        if not self.budget.enabled or tokens_before <= self.budget.input_tokens:
            return ContextAssembly(
                messages=original_messages,
                compacted=False,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        history_budget = max(self.budget.input_tokens - estimate_messages_tokens(prompt + current), 0)
        if not history or history_budget <= 0:
            messages = prompt + current
            return ContextAssembly(
                messages=messages,
                compacted=bool(history),
                tokens_before=tokens_before,
                tokens_after=estimate_messages_tokens(messages),
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
            tokens_before=tokens_before,
            tokens_after=estimate_messages_tokens(messages),
            summary_message=summary_message,
        )

    def _summary_budget(self, history_budget: int) -> int:
        if self.budget.summary_tokens <= 0:
            return 0
        if history_budget > self.budget.keep_recent_tokens:
            return min(self.budget.summary_tokens, history_budget - self.budget.keep_recent_tokens)
        if history_budget >= 30:
            return min(self.budget.summary_tokens, max(20, history_budget // 3))
        return 0

    def _select_first_kept_index(self, history: list[Message], recent_budget: int) -> int:
        if recent_budget <= 0:
            return len(history)

        total = 0
        first_kept_index = len(history)
        for index in range(len(history) - 1, -1, -1):
            message_tokens = estimate_message_tokens(history[index])
            if first_kept_index < len(history) and total + message_tokens > recent_budget:
                break
            if first_kept_index == len(history) and message_tokens > recent_budget:
                break
            total += message_tokens
            first_kept_index = index

        return _move_cut_before_tool_result(history, first_kept_index)

    def _build_summary_message(self, omitted: list[Message], summary_budget: int) -> Message | None:
        if not omitted or summary_budget <= 0:
            return None

        # TODO(compaction-state): after Host model usage metadata and summary
        # storage land, replace this deterministic summary with a persisted
        # checkpoint loaded through Host state/storage.
        summary = summarize_messages(omitted, _tokens_to_chars(summary_budget))
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


def estimate_messages_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


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


def estimate_message_tokens(message: Message) -> int:
    return _chars_to_tokens(estimate_message_chars(message))


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


def _runtime_context_window_tokens(metadata: dict[str, typing.Any]) -> int | None:
    for key in (
        "model_context_window_tokens",
        "context_window_tokens",
        "model_context_tokens",
        "contextWindowTokens",
        "context_window",
    ):
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value

    model = metadata.get("model")
    if isinstance(model, dict):
        for key in ("context_window_tokens", "contextWindowTokens", "context_window"):
            value = model.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value > 0:
                return value

    return None


def _config_token_value(
    config: dict[str, typing.Any],
    key: str,
    default: int | None,
    *,
    minimum: int,
) -> int | None:
    if key in config:
        return _config_int(config, key, default, minimum=minimum)
    return default


def _chars_to_tokens(chars: int) -> int:
    if chars <= 0:
        return 0
    return max(1, (chars + ESTIMATED_CHARS_PER_TOKEN - 1) // ESTIMATED_CHARS_PER_TOKEN)


def _tokens_to_chars(tokens: int) -> int:
    return max(tokens, 0) * ESTIMATED_CHARS_PER_TOKEN


def _config_int(
    config: dict[str, typing.Any],
    key: str,
    default: int | None,
    *,
    minimum: int,
) -> int | None:
    value = config.get(key, default)
    if isinstance(value, bool):
        return default
    if not isinstance(value, int):
        return default
    if value < minimum:
        return default
    return value
