# Local Agent

Built-in agent with model fallback, tools, and knowledge retrieval.

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

## Capabilities

- `streaming`: yes
- `tool_calling`: yes
- `knowledge_retrieval`: yes
- `multimodal_input`: yes
- `stateful_session`: yes

## Legacy Runner

Migrated from `local-agent` in LangBot.

## Contributing

We welcome contributions! Feel free to:

- Submit issues for bugs or feature requests
- Fork the repo and submit pull requests
- Improve documentation or add examples
- Share your ideas and feedback

Star the repo if you find it useful!