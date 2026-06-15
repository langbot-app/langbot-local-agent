# Local Agent Runner

Official LangBot AgentRunner plugin for the local, LangBot-hosted agent path.

This runner is a consumer of LangBot AgentRunner Protocol v1. LangBot provides
the host infrastructure, authorization, facts, and pull APIs; the runner owns
the model-facing agent behavior such as prompt assembly, history selection,
tool loop, RAG orchestration, and optional context compaction.

## Scope

This repository does not define the LangBot host protocol. It consumes the
Protocol v1 run context produced by LangBot. The canonical protocol source is
`LangBot/docs/agent-runner-pluginization/PROTOCOL_V1.md`; this README only
documents how Local Agent consumes that contract:

- `ctx.event`: event-first metadata for the current trigger.
- `ctx.conversation`, `ctx.actor`, `ctx.subject`: current run scope metadata.
- `ctx.input`: current structured input, including text, multimodal contents,
  and artifact/file references.
- `ctx.context`: context handles, inline policy, and available pull APIs. Local
  Agent uses the Host history API for conversation history instead of adapter
  bootstrap.
- `ctx.resources`: run-scoped authorized models, tools, knowledge bases, skills,
  files, and storage capabilities.
- `ctx.state`: small Host-projected state for the current run.
- `ctx.runtime`: runtime metadata such as deadline, trace id, query id from
  migration adapter paths, and Host metadata.
- `ctx.delivery`: host delivery surface and streaming/edit capabilities.
- `ctx.config`: runner binding config.
- `ctx.adapter`: migration adapter fields; not part of Protocol v1 core and not
  a place for prompt, history, RAG results, tool schemas, or authorized
  resources.

LangBot does not inline full conversation history by default. When the runner
needs more context, it should use authorized Host APIs through
`AgentRunAPIProxy`, for example model, prompt, history, artifact, tool, and
knowledge-base APIs.

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
| remove-think | boolean | no | false | Ask Host model APIs to remove provider thinking output when supported |
| knowledge-bases | knowledge-base-multi-selector | no | [] | Knowledge bases for RAG |
| retrieval-top-k | integer | no | 5 | Retrieval results requested per knowledge base |
| rerank-model | rerank-model-selector | no | '' | Rerank model for improved retrieval |
| rerank-top-k | integer | no | 5 | Top-K results after reranking |
| max-tool-iterations | integer | no | 20 | Maximum tool-call follow-up iterations |
| tool-execution-mode | select | no | parallel | Same-batch tool execution: `parallel` or `serial` |
| max-tool-result-chars | integer | no | 20000 | Maximum serialized tool result characters injected into the next model request |
| max-tool-result-artifact-bytes | integer | no | 1048576 | Maximum inline artifact payload bytes emitted by the runner for oversized tool text results |
| context-history-fetch-limit | integer | no | 50 | Transcript messages pulled from the Host history API |
| context-window-tokens | integer | no | 200000 | Fallback context window, and an upper cap when Host model metadata is available |
| context-reserve-tokens | integer | no | 16384 | Tokens reserved for the model response and provider overhead, clamped to at most 25% of the effective window |
| context-keep-recent-tokens | integer | no | 20000 | Approximate recent history tokens to retain when compaction triggers |
| context-summary-tokens | integer | no | 8000 | Maximum deterministic summary tokens inserted for compacted older history |

`prompt` is the static binding default. When LangBot exposes
`ctx.context.available_apis.prompt_get`, Local Agent pulls the
post-preprocessing effective prompt through `AgentRunAPIProxy.get_prompt()` and
uses it instead of the static default so `PromptPreProcessing` changes are
preserved. If the prompt API is unavailable, Local Agent falls back to
`ctx.config.prompt`.

`remove-think` is the first supported thinking-output control for Local Agent.
When enabled, the runner passes `remove_think=True` to Host model APIs for both
streaming and non-streaming calls. It is not Pi-style thinking-level control; it
only requests provider thinking output removal when the active Host model
adapter supports that flag.

Skill support is Host-mediated. When Local Agent advertises
`skill_authoring`, LangBot lists the current pipeline-visible skill facts in
`ctx.resources.skills` and exposes the Host-owned `activate` and
`register_skill` tools according to the same visibility policy. Calling
`activate` returns the full `SKILL.md` instructions as a tool result and
registers the skill package for Box mount resolution under
`/workspace/.skills/<skill-name>`. Local Agent consumes skill facts and tools
through Host APIs; it decides how tool schemas, tool results, prompt context,
or MCP surfaces are presented to the model.

Legacy singular `knowledge-base` values must be normalized by LangBot
configuration migration before runner execution. Local Agent only reads the
manifest-defined `knowledge-bases` binding config.

`max-tool-result-chars` is a runner-level safety fallback for model-facing tool
messages. String results, serialized JSON results, and error results are bounded
before they are appended as `role="tool"` messages for the next model request.
When Host exposes `ctx.context.available_apis.artifact_read`, oversized
non-error tool results are emitted as Host `artifact.created` results and the
model receives only an artifact reference plus a bounded preview. Follow-up
reads go through the runner-owned `langbot_artifact_read` tool, which calls
`AgentRunAPIProxy.artifact_read()`. This inline artifact path is capped by
`max-tool-result-artifact-bytes`; if artifact reads are unavailable or the
serialized result exceeds that byte cap, Local Agent falls back to a bounded
preview event and does not emit a full-result payload. Large files should be
returned by sandbox tools as Host artifact/file references, not inline file
content.

`tool-execution-mode` controls tool calls emitted in the same model turn.
`parallel` runs the batch concurrently and still writes tool-result messages
back in source order. `serial` executes them one by one.

Tools can return a top-level `terminate: true` runtime hint when the tool action
already completes the user-visible work and the automatic follow-up model call
should be skipped. Local Agent stops early only when every finalized tool result
in that batch sets `terminate: true`; mixed batches continue normally. The hint
is stripped from the model-facing `role="tool"` message so it does not become
business data for the next provider request.

When a sandbox or Host tool already returns explicit `artifact_refs`,
`artifact_id`, `file_refs`, `file_key`, or `file_id` fields, Local Agent treats
those as authoritative references. It does not create another runner artifact
for the same tool result; if the surrounding result is large, the model and Host
events receive the references plus a bounded preview only.

## Context Management

The local agent should be treated as a runner-owned or hybrid-context runner:

- LangBot inlines the current event/input and context handles.
- The runner pulls transcript history through the authorized Host history API.
- The runner decides whether to page history, read artifacts, summarize,
  compact, or construct a model request from scratch.
- Large files, images, audio, and tool outputs should be consumed as artifact
  references instead of large inline payloads.

Local Agent currently uses a runner-owned context pipeline:

1. Assemble effective prompt, host transcript history, RAG context, and current
   structured input.
2. Use the Host-provided model context window from `ctx.runtime.metadata` when
   available, capped by the runner binding's `context-window-tokens`. If Host
   metadata is unavailable, `context-window-tokens` is the fallback window and
   defaults to 200k tokens.
3. Estimate message tokens with a conservative local heuristic until LangBot
   exposes tokenizer/model usage metadata to runner plugins.
4. When the assembled context exceeds the effective input budget
   (`window - reserve`, with reserve clamped for small windows),
   use the authorized Host model API to generate a structured checkpoint summary,
   wrap it in a `system` message containing
   `<conversation_summary>...</conversation_summary>`, and keep a recent history
   tail bounded by `context-keep-recent-tokens`. If Host exposes the state API,
   Local Agent persists that summary as a conversation-scoped compaction
   checkpoint at `runner.compaction.checkpoint`, anchored by `covers_until`, and
   later runs reuse it before pulling transcript entries after that cursor. If
   model summarization fails or returns empty content, Local Agent falls back to
   a deterministic bounded summary.
5. Re-run the context transform before every model turn, including tool-call
   follow-up turns, so tool results and assistant tool calls are budgeted before
   the next provider request.
6. If a provider fails before producing any streamed content with a
   context-overflow style error, compact the current loop context with a more
   aggressive retry budget and retry the model turn once before surfacing
   `run.failed`.

This is not `max-round` behavior. History is not selected by number of rounds;
the runner budgets prompt, current input, summary, and recent history together,
following the Pi-style context threshold and per-turn transform shape. When the
Host does not expose the state API or a checkpoint cannot be parsed, Local Agent
falls back to the previous tail-history behavior. Future iterations can replace
local estimates with tokenizer/model usage metadata from the LiteLLM model-info
work.

Pipeline adapter data is intentionally narrow. Local Agent does not consume
`ctx.adapter.extra.prompt`; prompt handoff goes through the run-scoped Host
prompt API when available. New runner logic should prefer event-first context
and Host APIs over adapter fields.

## Host APIs Consumed

Model, prompt, history, state, artifact, tool, knowledge-base, rerank, and
steering access go through `AgentRunAPIProxy`. LangBot validates these calls
with the current `run_id`, run-scoped resource policy / available APIs, and
caller plugin identity.

Local Agent must not expose the runner process filesystem as an agent
capability. In sandboxed deployments, file access is mediated by Host/sandbox
tools registered in `ctx.resources.tools`; the model can request those tools,
and the runner invokes them through `AgentRunAPIProxy.call_tool()`. "Local" here
means the agent loop runs locally as a LangBot plugin, not that the model can
read or write arbitrary files on the runner machine.

Skill activation uses the same tool path. If Host exposes `activate` in the
run's allowed tools, the model calls `activate` like any other function tool and
Local Agent forwards it through `AgentRunAPIProxy.call_tool()`; no separate
runner action is required for skill activation.

Typical local-agent usage:

- Invoke authorized LangBot-hosted models.
- Call authorized tools.
- Retrieve authorized knowledge bases and rerank results.
- Page transcript history for the model request.
- Pull authorized steering inputs at turn boundaries.
- Read artifact content for files, images, or large tool results.

The runner must not bypass `ctx.resources` or call host-private managers to
access unauthorized models, tools, knowledge bases, files, storage, or platform
APIs.

## Capabilities

- `streaming`: yes
- `tool_calling`: yes
- `knowledge_retrieval`: yes
- `multimodal_input`: yes
- `skill_authoring`: yes
- `interrupt`: yes
- `steering`: yes

`interrupt` is cooperative. When Host exposes the run ledger API, Local Agent
polls the current run through `AgentRunAPIProxy.run_get()` at run boundaries and
streaming event boundaries. If Host has recorded `cancel_requested_at`, the
runner stops and emits `run.failed` with `code="cancelled"`.

`skill_authoring` means Local Agent can receive LangBot's Host-owned
`ctx.resources.skills` facts plus `activate`/`register_skill` tools when skills
are available. Skills remain owned by LangBot/Box; the runner owns how
model-facing prompts, tool schemas, tool results, or MCP adapters are assembled
from those Host capabilities.

Local Agent is reentrant and does not keep mutable per-conversation state in
the plugin instance. It can pull Host history each run. When Host state APIs
are available, it persists compacted summary checkpoints through
`AgentRunAPIProxy` so later runs can resume from
`runner.compaction.checkpoint`. It does not persist external session IDs or
runner-owned memory outside Host-managed state/storage.

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
