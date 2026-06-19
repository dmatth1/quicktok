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

## Drop-in for HuggingFace `AutoTokenizer`

Most code reaches a tokenizer through `transformers.AutoTokenizer`, not tiktoken.
`patch_transformers()` makes `AutoTokenizer.from_pretrained(...)` return a
quicktok-backed tokenizer **when quicktok supports the model's grammar**, and the
unmodified HF tokenizer otherwise — one call, existing code unchanged:

```python
import quicktok
quicktok.patch_transformers()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")  # quicktok-backed
ids = tok.encode(text)                       # fast path through quicktok
out = tok(text)["input_ids"]                 # same

quicktok.unpatch_transformers()              # restore the original
```

Exactness, not approximation. The wrapper only fast-paths a call whose output it
can reproduce **byte-for-byte**: plain `str` input, no per-call options it doesn't
model (padding, truncation, `return_tensors`, …), and special tokens only when the
model adds them as a verified static prefix/suffix (e.g. a single BOS). Everything
else — decode, batching, chat templates, an unsupported model — delegates to the
real HF tokenizer, so behaviour never changes. Models whose grammar quicktok
doesn't have (or that need `import_tokenizer` first) pass straight through.

- `wrap_pretrained(hf_tokenizer, quicktok_encoding_or_None)` wraps an
  already-constructed HF tokenizer directly; `None` returns it untouched.
- Coverage extends automatically to anything `import_tokenizer` /
  `encoding_for_model` resolve.

## Drop-in for tiktoken

Same method names and semantics, so a tiktoken `Encoding` swaps for
`quicktok.get_encoding(...)`. `encode` carries tiktoken's full signature:

```python
enc.encode(text, *, allowed_special=set(), disallowed_special="all")
```

Like tiktoken, a special-token string in the input **raises `ValueError`** by
default. Pass `allowed_special="all"` (or a set) to encode specials as ids, or
`disallowed_special=()` to disable the check; `encode_ordinary(text)` is the
"specials as plain text, never raises" path. `decode` raises `KeyError` on an
unknown id and takes `errors=` (default `"replace"`); `decode_bytes` is the
lossless path.

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

## Per-token offsets (token ↔ text spans)

`encode_with_offsets(text)` returns `(ids, spans)` where `spans[i]` is the
`(start, end)` range of the input that token `i` covers — for NER, span
highlighting, streaming detokenization, and training alignment:

```python
ids, spans = enc.encode_with_offsets("hello world")     # byte spans (default)
for tid, (lo, hi) in zip(ids, spans):
    assert enc.decode_bytes([tid]) == "hello world".encode()[lo:hi]   # exact

ids, spans = enc.encode_with_offsets(text, unit="char")  # HF offset_mapping shape
```

- **`unit="byte"`** (default): UTF-8 byte offsets, **exact and gap-free** — the
  spans tile the input and each is precisely the token's bytes (ordinary encode
  round-trips losslessly, so this is exact by construction).
- **`unit="char"`**: code-point offsets — **byte-identical to HuggingFace's
  `return_offsets_mapping=True`** (CI-verified against `AutoTokenizer`). Exact for
  non-NFC encodings; for NFC encodings (Qwen) offsets index the normalized text.

## Method reference

A `Tokenizer` mirrors tiktoken's `Encoding` across the common surface.

**Encode**

| method | returns | notes |
|---|---|---|
| `encode(text, *, allowed_special=set(), disallowed_special="all")` | `list[int]` | raises on a disallowed special, like tiktoken |
| `encode_ordinary(text)` | `list[int]` | specials as plain text, never raises |
| `encode_with_special(text)` / `encode_with_special_tokens(text)` | `list[int]` | all specials → ids |
| `encode_to_numpy(text, ...)` | `uint32` array | fastest single-encode path (see above) |
| `encode_batch(texts, threads=0, with_special=False)` | `(uint32 tokens, int64 offsets)` | parallel; doc i = `tokens[offsets[i]:offsets[i+1]]` |
| `encode_single_token(text_or_bytes)` | `int` | the id for exactly these bytes |
| `encode_with_offsets(text, *, unit="byte")` | `(list[int], list[(int,int)])` | ids + per-token spans; `unit="byte"` exact byte offsets, `unit="char"` == HF `offset_mapping` |
| `count(text)` | `int` | tokens `encode` would produce |

**Decode**

| method | returns | notes |
|---|---|---|
| `decode(ids, errors="replace")` | `str` | raises `KeyError` on unknown id |
| `decode_bytes(ids)` | `bytes` | lossless |
| `decode_batch(batch, errors="replace")` | `list[str]` | |
| `decode_single_token_bytes(id)` | `bytes` | |

**Introspect**

| member | returns | notes |
|---|---|---|
| `n_vocab` | `int` | max id + 1, **including** specials (cl100k → 100277) |
| `max_token_value` | `int` | |
| `eot_token` | `int` | |
| `special_tokens_set` | `set[str]` | |
| `is_special_token(id)` | `bool` | |
| `token_byte_values()` | `list[bytes]` | lexicographic, like tiktoken's `sorted_token_bytes` |
| `name` | `str` | the encoding name |

## Imported encodings & data location

- Imported encodings are stored in `$QUICKTOK_DATA` (default `~/.cache/quicktok`);
  `get_encoding` finds them automatically.
- `import_tokenizer` verifies against the model's own tokenizer, so the reference
  must be installed: `pip install tokenizers` (HF models) or `mistral-common`
  (Tekken) — plus `huggingface_hub` to fetch from a repo id (and its auth for
  gated repos).
- Full importing rules (which pretokenizer grammars are supported, what's
  refused) are in [docs/encodings.md](encodings.md#importing-other-tokenizers).
