# quicktok

A fast, exact BPE tokenizer for OpenAI and open-model encodings, written in C++.
Token ids are byte-identical to [tiktoken](https://github.com/openai/tiktoken);
encoding runs 2–3.5× faster than the fastest exact tokenizer we know of
([bpe-openai](https://github.com/github/rust-gems)) and 4–11× faster than tiktoken
itself. See [benchmarks](#benchmarks).

- **Exact** — ids match tiktoken byte-for-byte; every benchmark is exactness-checked before timing.
- **Drop-in** — Python wheels with a tiktoken-style API, a stable C ABI, CMake support.
- **Self-contained** — C++20, no external dependencies; cl100k_base, o200k_base, o200k_harmony, Llama-3, and Qwen2.5/3 ship in the repo.
- **Thread-safe** — load once, call `encode()` from as many threads as you like.

## Install

**Python** (the PyPI package is `quicktok-v1`; you still `import quicktok`):

```sh
pip install quicktok-v1
```

```python
import quicktok
enc = quicktok.get_encoding("cl100k_base")        # or "o200k_base", "o200k_harmony", "llama3", "qwen3"
ids = enc.encode("hello world")                   # == tiktoken.encode_ordinary
text = enc.decode(ids)
quicktok.encoding_for_model("gpt-4o").count("...")  # tiktoken-style model lookup
```

**C++** — via CMake (`find_package` or `FetchContent`), or `make install` and
pkg-config. There's also a stable **C ABI** (`quicktok.h`) for FFI from any language.

```cmake
find_package(quicktok REQUIRED)
target_link_libraries(app PRIVATE quicktok::quicktok)
```

## Build from source

```sh
git clone https://github.com/dmatth1/quicktok
cd quicktok
make            # builds build/libquicktok.{a, dylib/so}
make test       # verifies exact ids vs tiktoken (all encodings) + C ABI
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

The data files install to `share/quicktok`.

## Benchmarks

Five encoders, same machine (Apple M1), single thread, every output verified
token-for-token identical before timing. Three 25 MB corpora streamed from their
real sources — **The Pile** (diverse), **GitHub code**, **Common Crawl**
(multilingual) — across both common OpenAI encodings (throughput in **MB/s**):

**cl100k_base** (GPT-3.5 / GPT-4)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** | **116.1** | **144.2** | **75.2** |
| bpe-openai | 36.5 | 41.6 | 29.2 |
| tiktoken-rs | 15.3 | 14.3 | 13.5 |
| tiktoken (Python) | 14.7 | 13.2 | 12.3 |
| TokenDagger | 11.5 | 12.0 | 11.2 |

**o200k_base** (GPT-4o)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** | **100.6** | **117.1** | **59.2** |
| bpe-openai | 36.1 | 40.1 | 29.9 |
| tiktoken-rs | 23.1 | 20.9 | 17.9 |
| tiktoken (Python) | 21.6 | 19.3 | 16.3 |
| TokenDagger | 11.0 | 11.7 | 10.2 |

**Reproduce these tables:** `make bench-compare` — see [bench/README.md](https://github.com/dmatth1/quicktok/blob/main/bench/README.md).

<details>
<summary><b>Parallel / batch scaling</b> (Apple M1, 8 threads)</summary>

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
<summary><b>x86 cross-check</b> (cl100k + o200k, The Pile / Code / Common Crawl)</summary>

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
corpora, on NFC-normalized input — see the Qwen note in
[Encodings](#encodings).

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

<details>
<summary><b>Notes on fairness</b></summary>

<br>Every reference is called through the same raw API its own benchmark uses
(e.g. `encode_ordinary`, `encode_via_backtracking`) — no convenience-wrapper
handicaps. TokenDagger's README claims 2–4× over tiktoken, but that's on
Llama-4/Mistral vocabs on AMD EPYC; on cl100k/o200k here it lands around Python
tiktoken's level, and on its own Llama-4 vocab about half of it (table above).
We checked our setup wasn't the cause: PCRE2 JIT is verified active, and
rebuilding TokenDagger with `-O3 -mcpu=native` (its own Makefile uses `-O2`)
changes its numbers by under 1%. The gap is structural — a backtracking regex
engine called per pretoken, and byte-string-keyed merge lookups.
</details>

## How it's fast

Same algorithm as bpe-openai (exact backtracking BPE) — the speed is data-structure engineering:

- **2-byte trie** — the longest-match walk reads 2 input bytes per single 8-byte slot load, with a zero-lookup direct table for CJK characters.
- **Dense validity memos** — merge-validity checks hit exactly-keyed caches (2 MB for 17-bit token ids, a second wide one for 200k-vocab ids; a bijective mixer means no aliasing, ever).
- **Specialized pretokenizers** — the fixed cl100k/o200k-family regexes are compiled by hand into SIMD scanners; no general regex engine anywhere.

## API

```cpp
namespace quicktok {
class Tokenizer {
    // encoding: "cl100k_base" (default), "o200k_base", "o200k_harmony", "llama3", "qwen3", "llama4"
    static Tokenizer load_dir(const std::string& dir, const std::string& encoding = "cl100k_base");

    std::vector<uint32_t> encode(std::string_view text) const;          // encode_ordinary semantics
    std::vector<uint32_t> encode_with_special(std::string_view) const;  // allowed_special="all"
    void encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const;
    std::vector<std::vector<uint32_t>> encode_batch(const std::vector<std::string_view>&, unsigned threads = 0) const;
    size_t count(std::string_view text) const;

    std::string decode(const std::vector<uint32_t>& ids) const;         // handles special ids too
    void decode(const uint32_t* ids, size_t n, std::string& out) const;

    size_t vocab_size() const;
    const std::string& encoding() const;
};
}
```

- `encode()` is tiktoken's `encode_ordinary` (special tokens treated as plain text); `encode_with_special()` is tiktoken's `encode(text, allowed_special="all")`. Both byte-exact vs the reference, both tested.
- Any byte sequence is accepted; invalid UTF-8 round-trips through encode/decode unchanged.
- Python's `encode_batch` returns a flat `uint32` token array plus `int64` offsets (`tokens[offsets[i]:offsets[i+1]]` is document *i*, no per-document Python lists); `count_batch(enc, texts)` gives per-document counts.
- `load*` throws `std::runtime_error` on missing or corrupt data files. Nothing throws on the encode hot path (one exception: inputs over 4 GiB per call are rejected).
- A loaded `Tokenizer` is safe to share across threads — concurrent `encode()`/`decode()` is supported and tested.

## Encodings

Five encodings ship in the repo, each byte-exact vs its reference:

| name | model family | reference | notes |
|---|---|---|---|
| `cl100k_base` | GPT-3.5 / GPT-4 | tiktoken | the default |
| `o200k_base` | GPT-4o | tiktoken | ~85% of cl100k speed (2× vocab) |
| `o200k_harmony` | GPT-OSS (20b/120b) | tiktoken | same pattern + ranks as o200k_base, extra chat specials |
| `llama3` | Llama 3 | Meta tiktoken-rank | full cl100k speed; see exactness note |
| `qwen3` | Qwen2.5 / Qwen3 | HF tokenizers | cl100k speed; single-digit numbers; NFC note below |
| `llama4` | Llama 4 | Meta tiktoken-rank | **not bundled** (gated — bring your own vocab, see below) |

Load one by encoding name. The Python wheel bundles all the data files, so a name
is all you need; in C++ you also point at the directory holding them (`data/` in
this repo, or wherever `make install` put them):

```python
enc = quicktok.get_encoding("qwen3")                        # Python: data ships in the wheel
```
```cpp
auto tok = quicktok::Tokenizer::load_dir("data", "qwen3");  // C++: same thing, explicit data dir
```

- **Qwen2.5 / Qwen3** share one byte-level BPE; quicktok reproduces the Hugging
  Face tokenizer byte-for-byte **on NFC-normalized input**. HF's pipeline runs an
  NFC normalizer before tokenizing, which quicktok does not yet implement — input
  containing non-NFC codepoints (rare: 6–450 bytes per 25 MB across our bench
  corpora) tokenizes the raw bytes instead, and round-trips losslessly where HF's
  output decodes to the normalized text. Apache-2.0; regenerate with
  `tools/export_qwen.py --download`.
- **o200k_harmony** is o200k_base plus the harmony chat specials (`<|start|>`,
  `<|channel|>`, `<|return|>`, …) — ordinary text encodes identically to o200k_base.
- **Llama-3** reproduces Meta's original **tiktoken-rank** BPE byte-for-byte
  ([Meta Llama 3 Community License](https://llama.meta.com/llama3/license/), see
  [NOTICE](https://github.com/dmatth1/quicktok/blob/main/NOTICE); regenerate with `tools/export_llama3.py <tokenizer.json> data`).
  Hugging Face / llama.cpp infer the same vocab from a **merge list** and agree on
  ~99.9998% of tokens — the rare differences are non-Latin+symbol sequences (e.g.
  Cyrillic next to `€`) where rank order and merge order pick different splits.
- **Llama-4** uses the same pretokenizer as o200k_base, so the code path ships —
  but Meta's vocab is gated and **not bundled**. Export your own, then load it:

  ```sh
  python tools/export_llama4.py <tokenizer.model> data
  ```
  ```python
  enc = quicktok.get_encoding("llama4", "data")
  ```

## Notes

- Builds tune to the host CPU by default (`-march=native`); set `CXXFLAGS_ARCH` for portable binaries.
- The bundled Unicode table is pinned and version-stamped; `python tools/export_unicode.py verify` re-derives all 1.1 M codepoints against the live reference and diffs them — see [`data/uniclass.bin.meta`](https://github.com/dmatth1/quicktok/blob/main/data/uniclass.bin.meta).
- To regenerate the data files from the references:

  ```sh
  pip install tiktoken regex tokenizers
  python tools/export_fixtures.py        # cl100k/o200k/o200k_harmony from tiktoken
  python tools/export_qwen.py --download  # data/qwen3.vocab (Apache-2.0)
  python tools/export_unicode.py         # data/uniclass.bin + version stamp
  python tools/gen_vectors.py            # test vectors (tiktoken encodings)
  ```

## License

MIT — see [LICENSE](https://github.com/dmatth1/quicktok/blob/main/LICENSE).
