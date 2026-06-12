# quicktok

A fast, exact BPE tokenizer for OpenAI and open-model encodings, written in C++.
Token ids are byte-identical to [tiktoken](https://github.com/openai/tiktoken);
encoding runs 2–3.4× faster than the fastest exact tokenizer we know of
([bpe-openai](https://github.com/github/rust-gems)) and 3.5–11× faster than tiktoken
itself. See [benchmarks](#benchmarks).

- **Exact** — ids match each encoding's reference (tiktoken / Hugging Face / Meta) byte-for-byte; every benchmark is exactness-checked before timing.
- **Drop-in** — Python wheels with a tiktoken-style API, a stable C ABI, CMake support.
- **Self-contained** — C++20, no external dependencies; cl100k_base, o200k_base, o200k_harmony, Llama-3, and Qwen2.5/3 ship in the repo.
- **Thread-safe** — load once, call `encode()` from as many threads as you like.

## Install

**Python** (the PyPI package is `quicktok-v1`; you still `import quicktok`):

```sh
pip install quicktok-v1
```

Wheels for Linux (x86_64, aarch64), macOS (arm64, x86_64), and Windows; Python ≥ 3.9.

**C++** — via CMake (`find_package` or `FetchContent`), or `make install` and
pkg-config. There's also a stable **C ABI** (`quicktok.h`) for FFI from any language.

```cmake
find_package(quicktok REQUIRED)
target_link_libraries(app PRIVATE quicktok::quicktok)
```

## Python

```python
import quicktok
enc = quicktok.get_encoding("cl100k_base")
ids = enc.encode("hello world")                   # == tiktoken.encode_ordinary
text = enc.decode(ids)

quicktok.encoding_for_model("meta-llama/Llama-3.1-8B").count("...")   # model-name lookup

# other byte-level-BPE tokenizers: import once (exactness-verified), then use by name
quicktok.import_tokenizer("mistralai/Mistral-Nemo-Instruct-2407", "tekken")  # HF repo id, URL, or local file
quicktok.get_encoding("tekken")
```

All encoding names — bundled, gated, imported — are in [Encodings](#encodings).

For bulk work (dataset prep, corpus token counting), `encode_batch` tokenizes
documents in parallel and returns one flat `uint32` token array plus `int64`
offsets — no per-document Python lists; ~550 MB/s from Python on an M1:

```python
tokens, offsets = enc.encode_batch(docs)    # doc i = tokens[offsets[i]:offsets[i+1]]
tokens.tofile("corpus.tokens.bin")          # flat binary, ready for training
counts = quicktok.count_batch(enc, docs)    # per-doc token counts for budgeting

enc.encode_batch(chats, with_special=True)  # chat-templated data: "<|im_start|>..." -> special ids
```

- `encode()` is tiktoken's `encode_ordinary` (special tokens treated as plain text); `encode_with_special()` is tiktoken's `encode(text, allowed_special="all")`. Both byte-exact vs the reference, both tested.
- Any byte sequence is accepted; invalid UTF-8 round-trips through encode/decode unchanged. (NFC encodings decode valid-but-non-NFC input to its normalized form — see [Encodings](#encodings).)
- Imported encodings are stored in `$QUICKTOK_DATA` (default `~/.cache/quicktok`); `get_encoding` finds them automatically.
- `import_tokenizer` verifies against the model's own tokenizer, so the reference must be installed: `pip install tokenizers` (HF models) or `mistral-common` (Tekken) — plus `huggingface_hub` to fetch from a repo id (and its auth for gated repos).

## C++

```sh
git clone https://github.com/dmatth1/quicktok
cd quicktok
make            # builds build/libquicktok.{a, dylib/so}
make test       # verifies exact ids vs the references (all encodings) + C ABI
```

```cpp
#include <quicktok.hpp>

auto tok = quicktok::Tokenizer::load_dir("data");              // cl100k_base
auto gpt4o = quicktok::Tokenizer::load_dir("data", "o200k_base");

auto ids = tok.encode("Hello, quicktok! 日本語 🚀");  // std::vector<uint32_t>
std::string text = tok.decode(ids);                   // lossless round-trip
size_t n = tok.count("how many tokens is this?");
auto with_sp = tok.encode_with_special("a<|endoftext|>b");  // specials -> ids
auto batch = tok.encode_batch(texts);                       // parallel
```

```cpp
namespace quicktok {
class Tokenizer {
    // encoding: a built-in name (see Encodings) or any imported one
    static Tokenizer load_dir(const std::string& dir, const std::string& encoding = "cl100k_base");

    std::vector<uint32_t> encode(std::string_view text) const;          // encode_ordinary semantics
    std::vector<uint32_t> encode_with_special(std::string_view) const;  // allowed_special="all"
    void encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const;
    std::vector<std::vector<uint32_t>> encode_batch(const std::vector<std::string_view>&,
                                                    unsigned threads = 0, bool with_special = false) const;
    size_t count(std::string_view text) const;

    std::string decode(const std::vector<uint32_t>& ids) const;         // handles special ids too
    void decode(const uint32_t* ids, size_t n, std::string& out) const;

    size_t vocab_size() const;
    const std::string& encoding() const;
};
}
```

- `load_dir` takes the directory holding the data files (`data/` in this repo;
  `make install` puts them in `share/quicktok`). It throws `std::runtime_error`
  on missing or corrupt files; nothing throws on the encode hot path (one
  exception: inputs over 4 GiB per call are rejected).
- A loaded `Tokenizer` is safe to share across threads — concurrent
  `encode()`/`decode()` is supported and tested.

## Benchmarks

Five encoders, same machine (Apple M1), single thread, every output verified
token-for-token identical before timing, every reference called through the same
raw API its own benchmark uses. Three 25 MB corpora streamed from their real
sources — **The Pile** (diverse), **GitHub code**, **Common Crawl**
(multilingual) — across both common OpenAI encodings (throughput in **MB/s**):

**cl100k_base** (GPT-3.5 / GPT-4)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** | **118.3** | **142.6** | **72.3** |
| bpe-openai | 37.5 | 41.9 | 28.6 |
| tiktoken-rs | 15.7 | 14.1 | 13.4 |
| tiktoken (Python) | 14.2 | 13.1 | 12.0 |
| TokenDagger | 11.2 | 11.7 | 10.7 |

**o200k_base** (GPT-4o)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** | **99.3** | **122.3** | **54.7** |
| bpe-openai | 35.4 | 39.2 | 28.2 |
| tiktoken-rs | 23.2 | 21.6 | 17.2 |
| tiktoken (Python) | 20.1 | 18.5 | 15.4 |
| TokenDagger | 10.5 | 11.3 | 9.7 |

**Reproduce these tables:** `make bench-compare` — see [bench/README.md](https://github.com/dmatth1/quicktok/blob/main/bench/README.md).

<details>
<summary><b>Parallel / batch scaling</b> (Apple M1, 8 threads; measured at v0.3.2)</summary>

<br>Native `encode_batch()` (`make bench`), cl100k:

| threads | MB/s | speedup |
|---:|---:|---:|
| 1 | 110 | 1.0× |
| 2 | 210 | 1.9× |
| 4 | 397 | 3.6× |
| 8 | **706** | **6.4×** |

From Python (`make bench-py`), cl100k, 10 threads:

| Python API | quicktok | tiktoken | speedup |
|---|---:|---:|---:|
| single-thread | 77 MB/s | 15 MB/s | **5.0×** |
| `encode_batch` | **550 MB/s** | 24 MB/s (batch) | **24×** |
</details>

<details>
<summary><b>x86 cross-check</b> (cl100k + o200k, The Pile / Code / Common Crawl; measured at v0.3.2 — re-validation of the current engine pending)</summary>

<br>Same encoders, corpora, and method on an x86 server (Intel Xeon @ 2.8 GHz,
single thread, MB/s). quicktok is shown both as the native C++ kernel and as the
Python wheel:

**cl100k_base** (GPT-3.5 / GPT-4)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** (native) | **75.0** | **86.6** | **47.9** |
| quicktok (Python) | 42.5 | 47.4 | 29.6 |
| bpe-openai | 25.9 | 30.8 | 22.9 |
| tiktoken-rs | 11.1 | 10.2 | 11.1 |
| tiktoken (Python) | 10.2 | 9.1 | 9.5 |
| TokenDagger | 7.1 | 7.6 | 7.1 |

**o200k_base** (GPT-4o)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** (native) | **60.9** | **69.5** | **37.3** |
| quicktok (Python) | 36.6 | 41.1 | 26.7 |
| bpe-openai | 24.4 | 28.9 | 23.9 |
| tiktoken-rs | 17.0 | 15.5 | 15.1 |
| tiktoken (Python) | 14.2 | 13.3 | 13.2 |
| TokenDagger | 6.6 | 7.3 | 6.2 |

Same ordering as the M1 tables, the Python wheel included. (One footnote:
TokenDagger diverges from the other four by a single token on Pile/cl100k — a
known TokenDagger edge case, not an encoder bug.)
</details>

<details>
<summary><b>Open-model encodings</b> (Apple M1: vs llama.cpp on Llama-3, vs Hugging Face tokenizers on Qwen3, vs TokenDagger on Llama-4)</summary>

<br>Same corpora and method as the headline tables (single thread, best-of-5, MB/s).

**Llama-3** — quicktok vs `libllama`'s own `llama_tokenize` (same 128k vocab,
vocab-only GGUF, llama.cpp built with `-DGGML_NATIVE=ON`):

| | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** | **124.2** | **139.8** | **66.5** |
| llama.cpp | 9.8 | 10.6 | 5.8 |

Token agreement is 99.999% on The Pile and Code and 99.81% on multilingual
Common Crawl — the known tiktoken-rank vs merge-list divergence (see
[Encodings](#encodings)); quicktok matches Meta's original tokenizer.

**Qwen3** — quicktok vs Hugging Face `tokenizers` (the Rust core behind
`AutoTokenizer`), timed per document (its best case; a single 25 MB string
drops it below 2.2 MB/s):

| | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** | **100.4** | **124.5** | **61.5** |
| HF tokenizers | 3.4 | 3.8 | 3.1 |

quicktok's output was verified token-for-token identical to HF's on all three
raw corpora (quicktok implements the same NFC normalization HF runs — see the
Qwen note in [Encodings](#encodings)).

**Llama-4** — quicktok vs TokenDagger, on the vocab TokenDagger's own headline
numbers come from (its native `CoreBPE`; both encoders verified token-for-token
against Meta's tiktoken construction before timing; tiktoken itself shown for
scale):

| | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** | **92.2** | **119.2** | **56.7** |
| tiktoken | 21.1 | 19.3 | 16.5 |
| TokenDagger | 10.5 | 11.8 | 10.1 |

Meta gates the Llama-4 vocab, so it isn't bundled — this bench takes a local
`tokenizer.model` (a TokenDagger clone happens to redistribute one).

Reproduce: `python bench/hf_qwen_bench.py`, `python bench/llama4_bench.py`,
and `bench/llamacpp_bench.cpp`.
</details>

## How it's fast

Same algorithm as bpe-openai (exact backtracking BPE) — the speed is data-structure engineering:

- **2-byte trie** — the longest-match walk reads 2 input bytes per single 8-byte slot load, with a zero-lookup direct table for CJK characters.
- **Dense validity memos** — merge-validity checks hit exactly-keyed caches (2 MB for 17-bit token ids, a second wide one for 200k-vocab ids; a bijective mixer means no aliasing, ever).
- **Specialized pretokenizers** — the fixed cl100k/o200k-family regexes are compiled by hand into SIMD scanners; no general regex engine anywhere.
- **Single-pass product machines** — for ASCII text (most of code and English), one loop owns both the pretokenizer's boundary rules and token emission: contractions, prefix-words, digit triples, punct runs, and the whitespace cascade are handled inline with no per-piece scanner dispatch; only Unicode contact falls back to the general scanner, one piece at a time.

## Encodings

Five encodings ship in the repo; Llama-4's code path ships too, but its vocab is
gated. Each is byte-exact vs its reference:

| name | model family | reference |
|---|---|---|
| `cl100k_base` | GPT-3.5 / GPT-4 | tiktoken (the default) |
| `o200k_base` | GPT-4o | tiktoken |
| `o200k_harmony` | GPT-OSS | tiktoken — o200k_base plus the harmony chat specials |
| `llama3` | Llama 3 | Meta's tiktoken-rank BPE |
| `qwen3` | Qwen2.5 / Qwen3 | Hugging Face tokenizers, including its NFC normalization |
| `llama4` | Llama 4 | Meta — **not bundled** (gated; bring your own vocab) |

- **qwen3** reproduces the HF pipeline including NFC normalization: clean input
  pays one cheap scan, only spans with non-NFC codepoints are normalized, and —
  like HF — decode returns the normalized text for such input.
- **llama3** matches Meta's original tiktoken-rank tokenizer. Hugging Face and
  llama.cpp infer the same vocab from a merge list and agree on ~99.9998% of
  tokens; the rare differences are non-Latin+symbol sequences where rank order
  and merge order pick different splits.
- **llama4** shares o200k_base's pretokenizer, but Meta gates the vocab. With
  repo access, import it like any other tokenizer —
  `quicktok.import_tokenizer("meta-llama/Llama-4-Scout-17B-16E-Instruct", "llama4")`
  (the import verifies, so a wrong guess can't ship; this checks against HF's
  merge-list tokenizer, the same rank-vs-merges nuance as llama3). To match
  Meta's original rank file instead, export from a checkout:
  `python tools/export_llama4.py <tokenizer.model> data`.

Vocabs regenerate from their references with `tools/export_*.py`; the Unicode and
NFC tables are pinned, version-stamped, and exhaustively re-derivable
(`tools/export_unicode.py verify`, `tools/export_nfc.py verify`). Third-party
vocab licenses: [NOTICE](https://github.com/dmatth1/quicktok/blob/main/NOTICE).

### Importing other tokenizers

Any byte-level-BPE tokenizer whose pretokenizer matches one of quicktok's
hand-compiled grammars (cl100k / o200k / llama3 / qwen / tekken) can be imported —
`quicktok.import_tokenizer()` from Python (shown above), or:

```sh
python -m quicktok.importer path/to/tokenizer.json myenc --corpus big.txt
```

The import checks the normalizer (none/NFC), classifies the pretokenizer regex,
writes the data files, then encodes a stress suite plus any `--corpus` files with
both the reference tokenizer and quicktok and compares token-for-token. A
mismatch fails the import; an unrecognized pattern is refused with the pattern
printed. There is no fallback regex engine and no approximate mode — each grammar
is compiled by hand, which is where the speed comes from.

- **Mistral Tekken v3** imports and verifies: the o200k grammar minus
  contractions, single-digit numbers, ids offset by its 1,000 reserved special
  slots — exact vs `mistral-common`, at full o200k-class speed.
- **DeepSeek V3/R1** is refused: a pipeline of three sequential Split regexes, a
  different grammar shape. Supporting it would be a new scanner, not an import.
- **SentencePiece models** (Gemma, T5, Llama-2) are a different algorithm
  entirely, not a missing grammar — out of scope.

## Notes

- Builds tune to the host CPU by default (`-march=native`); set `CXXFLAGS_ARCH` for portable binaries.
- To regenerate the data files from the references:

  ```sh
  pip install tiktoken regex tokenizers
  python tools/export_fixtures.py        # cl100k/o200k/o200k_harmony from tiktoken
  python tools/export_qwen.py --download  # data/qwen3.vocab (Apache-2.0)
  python tools/export_unicode.py         # data/uniclass.bin + version stamp
  python tools/export_nfc.py             # data/nfc.bin (NFC tables) + version stamp
  python tools/gen_vectors.py            # test vectors (tiktoken encodings)
  ```

## License

MIT — see [LICENSE](https://github.com/dmatth1/quicktok/blob/main/LICENSE).
