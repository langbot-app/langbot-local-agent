# Local Agent Runner

Official LangBot AgentRunner plugin for the former built-in `local-agent` runner.

The migration goal is external behavior parity with the built-in runner, while
moving the implementation behind the AgentRunner protocol boundary.

## Runner ID

`plugin:langbot/local-agent/default`

## Configuration

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| model | model-fallback-selector | yes | primary: '', fallbacks: [] | LLM model with fallbacks |
| max-round | integer | yes | 10 | Maximum conversation rounds |
| prompt | prompt-editor | yes | system: "You are a helpful assistant." | System prompt |
| knowledge-bases | knowledge-base-multi-selector | no | [] | Knowledge bases for RAG |
| rerank-model | rerank-model-selector | no | '' | Rerank model for improved retrieval |
| rerank-top-k | integer | no | 5 | Top-K results after reranking |

`prompt` remains in the static runner config for defaults and UI editing. During
execution the runner prefers `ctx.prompt`, which is the effective prompt prepared
by LangBot after pipeline preprocessing and `PromptPreProcessing` events. This
matches the old built-in behavior where model calls used:

```text
query.prompt.messages + query.messages + query.user_message
```

The plugin runner consumes the equivalent AgentRunner context:

```text
ctx.prompt + ctx.messages + current user message from ctx.input
```

The legacy singular `knowledge-base` config key is still accepted during
migration and is treated as a one-item `knowledge-bases` list.

## Host Context Consumed

- `ctx.prompt`: effective host-preprocessed prompt.
- `ctx.messages`: conversation history supplied by LangBot.
- `ctx.input.contents`: structured current input, including text, images, and files.
- `ctx.resources`: authorized models, tools, knowledge bases, and storage.
- `ctx.runtime.metadata.streaming_supported`: adapter streaming capability.

Model, tool, knowledge-base, and rerank calls go through `AgentRunAPIProxy`, so
LangBot can enforce run-scoped authorization and restore Query-bound behavior on
the host side.

## Capabilities

- `streaming`: yes
- `tool_calling`: yes
- `knowledge_retrieval`: yes
- `multimodal_input`: yes
- `stateful_session`: yes

## Legacy Runner

Migrated from `local-agent` in LangBot. The old `RequestRunner` implementation
may remain in LangBot during migration as the parity reference; it is not the
long-term runtime path.

## Contributing

We welcome contributions! Feel free to:

- Submit issues for bugs or feature requests
- Fork the repo and submit pull requests
- Improve documentation or add examples
- Share your ideas and feedback

Star the repo if you find it useful!
