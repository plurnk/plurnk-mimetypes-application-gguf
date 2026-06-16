# @plurnk/plurnk-mimetypes-application-gguf

`application/x-gguf` (GGUF local-LLM model) mimetype handler for the [plurnk](https://github.com/plurnk) ecosystem. Hand-rolled binary reader, no parser dependency.

## install

```
npm i @plurnk/plurnk-mimetypes-application-gguf
```

## what it does

GGUF (llama.cpp / Ollama) packs a model's weights behind a structured key-value metadata **header**: architecture, quantization, context length, embedding length, tokenizer, chat template. This handler reads **only that header** — never the gigabytes of tensor data — so an agent in a model folder can answer "what model is this, what quant, what context window" for free.

- `extractRaw(content)` — the metadata keys (`general.architecture`, `llama.context_length`, …) as `field` symbols, in file order.
- `deepJson(content)` — `{ version, tensorCount, metadata }`, a jsonpath target (`$.metadata['llama.context_length']`).
- `toText` (regex/glob + embed-source) — the metadata rendered as a `key: value` table. Large arrays (tokenizer vocabularies) are summarized as `<type[N]>`, never expanded.
- `validate` — throws on bad magic or a truncated header; every other channel degrades to empty.

Per-tensor inventory is out of scope for v1 — the metadata is the "what is this" answer. References are not applicable.

## license

MIT.
