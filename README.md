# tiny-transformer

A small educational decoder-only transformer implementation for learning LLM fundamentals.

This project starts with a character-level tokenizer and a minimal autoregressive model so the full path is inspectable:

text -> tokens -> embeddings -> attention -> logits -> loss -> generation

Primary goals:

- understand tokenization and embeddings
- implement causal self-attention
- train a tiny model on a small dataset
- generate text end-to-end
- record shape/debugging notes and validation evidence

Framework record:

<https://github.com/trinidad-lopez/technical-growth-framework/tree/main/projects/planned/llm-fundamentals-lab>