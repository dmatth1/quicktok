# Python API

[← back to README](../README.md)

```sh
pip install quicktok-v1
```

The PyPI package is `quicktok-v1`; you still `import quicktok`. Runs on Linux,
Mac, and Windows. Python ≥ 3.9 required.

## Basics

```python
import quicktok
enc = quicktok.get_encoding("cl100k_base")
ids = enc.encode("hello world")                   # tiktoken semantics (raises on a stray special)
text = enc.decode(ids)

quicktok.encoding_for_model("meta-llama/Llama-3.1-8B").count("...")   # model-name lookup

# other byte-level-BPE tokenizers: import once (exactness-verified), then use by name
quicktok.import_tokenizer("mistralai/Mistral-Nemo-Instruct-2407", "tekken")  # HF repo id, URL, or local file
quicktok.get_encoding("tekken")
```

All encoding names — bundled, gated, imported — are in
[docs/encodings.md](encodings.md).

## Drop-in for tiktoken

Same method names and semantics, so a tiktoken `Encoding` swaps for
`quicktok.get_encoding(...)`. Like tiktoken, `encode` raises on a stray special
token unless you pass `allowed_special` (or call `encode_ordinary`).

## Bulk / batch

For bulk work (dataset prep, corpus token counting), `encode_batch` tokenizes
documents in parallel and returns one flat `uint32` token array plus `int64`
offsets:

```python
tokens, offsets = enc.encode_batch(docs)    # doc i = tokens[offsets[i]:offsets[i+1]]
tokens.tofile("corpus.tokens.bin")          # flat binary, ready for training
counts = quicktok.count_batch(enc, docs)    # per-doc token counts for budgeting

enc.encode_batch(chats, with_special=True)  # chat-templated data: "<|im_start|>..." -> special ids
```

## `encode_to_numpy()`

`encode_to_numpy()` returns a `uint32` array directly, skipping the per-token
Python-list marshalling — so from Python it runs at **near-native speed** (~3×
over bpe-openai, ~7× over tiktoken) on large inputs. The plain `encode()` builds
a `list[int]` and is the convenient default for smaller inputs. See the
[benchmarks](../bench/README.md#results) for the native / numpy / list split.

## Imported encodings & data location

- Imported encodings are stored in `$QUICKTOK_DATA` (default `~/.cache/quicktok`);
  `get_encoding` finds them automatically.
- `import_tokenizer` verifies against the model's own tokenizer, so the reference
  must be installed: `pip install tokenizers` (HF models) or `mistral-common`
  (Tekken) — plus `huggingface_hub` to fetch from a repo id (and its auth for
  gated repos).
- Full importing rules (which pretokenizer grammars are supported, what's
  refused) are in [docs/encodings.md](encodings.md#importing-other-tokenizers).
