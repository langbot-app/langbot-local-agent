# Local Agent Runner

Official LangBot AgentRunner plugin for the local, LangBot-hosted agent path.

This runner is a consumer of LangBot AgentRunner Protocol v1. LangBot provides
the host infrastructure, authorization, facts, and pull APIs; the runner owns
the model-facing agent behavior such as prompt assembly, history selection,
tool loop, RAG orchestration, and optional context compaction.

## Scope

This repository does not define the LangBot host protocol. It consumes the
Protocol v1 run context produced by LangBot:

- `ctx.event`: event-first metadata for the current trigger.
- `ctx.input`: current structured input, including text, multimodal contents,
  and artifact/file references.
- `ctx.context`: context handles, inline policy, and available pull APIs. Local
  Agent uses the Host history API for conversation history instead of adapter
  bootstrap.
- `ctx.resources`: run-scoped authorized models, tools, knowledge bases, files,
  and storage capabilities.
- `ctx.runtime`: runtime metadata such as deadline, trace id, query id from
  Pipeline adapter paths, and adapter capabilities.
- `ctx.delivery`: host delivery surface and streaming/edit capabilities.
- `ctx.adapter`: Pipeline adapter fields; not part of Protocol v1 core.

LangBot does not inline full conversation history by default. When the runner
needs more context, it should use authorized Host APIs through
`AgentRunAPIProxy`, for example history, event, artifact, state, model, tool,
knowledge-base, and storage APIs.

AgentRunner components should obtain that proxy with `self.get_run_api(ctx)`.
They should not use the legacy `self.plugin` proxy that regular non-runner
plugin components use.

## Runner ID

`plugin:langbot/local-agent/default`

## Configuration

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| model | model-fallback-selector | yes | primary: '', fallbacks: [] | LLM model with fallbacks |
| timeout | integer | no | 300 | Total runner execution timeout in seconds. Set to `0` or `null` to disable the host deadline. |
| prompt | prompt-editor | yes | system: "You are a helpful assistant." | Default system prompt edited in LangBot UI |
| knowledge-bases | knowledge-base-multi-selector | no | [] | Knowledge bases for RAG |
| retrieval-top-k | integer | no | 5 | Retrieval results requested per knowledge base |
| rerank-model | rerank-model-selector | no | '' | Rerank model for improved retrieval |
| rerank-top-k | integer | no | 5 | Top-K results after reranking |
| max-tool-iterations | integer | no | 10 | Maximum tool-call follow-up iterations |
| max-tool-result-chars | integer | no | 20000 | Maximum serialized tool result characters injected into the next model request |
| context-history-fetch-limit | integer | no | 50 | Transcript messages pulled from the Host history API |
| context-window-tokens | integer | no | 200000 | Approximate context window when Host model metadata is not available |
| context-reserve-tokens | integer | no | 16384 | Tokens reserved for the model response and provider overhead |
| context-keep-recent-tokens | integer | no | 20000 | Approximate recent history tokens to retain when compaction triggers |
| context-summary-tokens | integer | no | 8000 | Maximum deterministic summary tokens inserted for compacted older history |

`prompt` is the static binding default. When LangBot exposes
`ctx.context.available_apis.prompt_get`, Local Agent pulls the
post-preprocessing effective prompt through `AgentRunAPIProxy.get_prompt()` and
uses it instead of the static default so `PromptPreProcessing` changes are
preserved. If the prompt API is unavailable, Local Agent falls back to
`ctx.config.prompt`.

Legacy singular `knowledge-base` values must be normalized by LangBot
configuration migration before runner execution. Local Agent only reads the
manifest-defined `knowledge-bases` binding config.

`max-tool-result-chars` is a runner-level safety fallback for model-facing tool
messages. String results, serialized JSON results, and error results are bounded
before they are appended as `role="tool"` messages for the next model request.
Oversized content keeps the leading characters and appends a marker with the
original and kept character counts. This does not implement Host artifact
persistence yet; full-output references should move to Host artifact/storage APIs
in a later phase when that API surface is available.

## Context Management

The local agent should be treated as a self-managed or hybrid-context runner:

- LangBot inlines the current event/input and context handles.
- The runner pulls transcript history through the authorized Host history API.
- The runner decides whether to search history, read
  artifacts, load state, summarize, compact, or construct a model request from
  scratch.
- Large files, images, audio, and tool outputs should be consumed as artifact
  references instead of large inline payloads.

Local Agent currently uses a runner-owned context pipeline:

1. Assemble effective prompt, host transcript history, RAG context, and current
   structured input.
2. Use the Host-provided model context window from `ctx.runtime.metadata` when
   available; otherwise use the runner binding's `context-window-tokens`,
   which defaults to 200k tokens.
3. Estimate message tokens with a conservative local heuristic until LangBot
   exposes tokenizer/model usage metadata to runner plugins.
4. When the assembled context exceeds `context-window-tokens - context-reserve-tokens`,
   replace older history with a `system` message containing
   `<conversation_summary>...</conversation_summary>` and keep a recent history
   tail bounded by `context-keep-recent-tokens`.
5. Re-run the context transform before every model turn, including tool-call
   follow-up turns, so tool results and assistant tool calls are budgeted before
   the next provider request.
6. If a provider fails before producing any streamed content with a
   context-overflow style error, compact the current loop context with a more
   aggressive retry budget and retry the model turn once before surfacing
   `run.failed`.

This is not `max-round` behavior. History is not selected by number of rounds;
the runner budgets prompt, current input, summary, and recent history together,
following the Pi-style context threshold and per-turn transform shape. Future
iterations can replace the deterministic summary generator with an LLM summary
and persist compaction checkpoints through Host state/storage after LangBot
exposes tokenizer/model usage metadata from the LiteLLM model-info work.

Pipeline adapter data is intentionally narrow. Local Agent does not consume
`ctx.adapter.extra.prompt`; prompt handoff goes through the run-scoped Host
prompt API when available. New runner logic should prefer event-first context
and Host APIs over adapter fields.

## Host APIs Consumed

Model, tool, knowledge-base, artifact, history, event, state, and storage access
go through `AgentRunAPIProxy`. LangBot validates these calls with the current
`run_id`, runner permissions, resource policy, and caller plugin identity.

Typical local-agent usage:

- Invoke authorized LangBot-hosted models.
- Call authorized tools.
- Retrieve authorized knowledge bases and rerank results.
- Page transcript history for the model request and search history when the
  runner decides it is needed.
- Read artifact metadata/content for files, images, or large tool results.
- Save optional summary/checkpoint/session state through Host state or storage.

The runner must not bypass `ctx.resources` or call host-private managers to
access unauthorized models, tools, knowledge bases, files, or storage.

## Capabilities

- `streaming`: yes
- `tool_calling`: yes
- `knowledge_retrieval`: yes
- `multimodal_input`: yes
- `skill_authoring`: yes
- `skill_injection`: yes
- `event_context`: yes
- `stateful_session`: yes

`stateful_session` means the runner can participate in cross-run state through
Host-owned state/storage or through runner-owned external state. It does not
mean the plugin instance should keep mutable per-conversation state in memory.
The plugin process is shared and must remain reentrant.

## Event System Boundary

This plugin does not implement LangBot EventGateway, event subscription, event
notification, scheduler, or event fanout. Those systems belong to LangBot host
or separate event-focused branches. This runner only consumes the run context
that LangBot delivers through AgentRunner Protocol v1.

## Current Boundary

This plugin is the target external implementation of LangBot's local agent
runner. LangBot's internal runner code can be used as reference material, but
its host-private structures must not become plugin API.

## Contributing

We welcome contributions. Useful areas include:

- Protocol v1 adapter fixes
- history/artifact/state API consumption
- tool loop and RAG behavior
- multimodal input handling
- focused tests and documentation improvements
