# quicktok

A fast, exact BPE tokenizer for OpenAI and open-model encodings, written in C++.
Token ids are byte-identical to [tiktoken](https://github.com/openai/tiktoken), and
encoding runs about 3× faster than the fastest exact tokenizer we know of
([bpe-openai](https://github.com/github/rust-gems)) — 6–8× faster than tiktoken
itself. cl100k, o200k, GPT-OSS, Llama-3, and Qwen all bundled. See
[benchmarks](#benchmarks).

- **Exact** — ids match tiktoken byte-for-byte; every benchmark is exactness-checked before timing.
- **Drop-in** — Python wheels with a tiktoken-style API, a stable C ABI, CMake support.
- **Self-contained** — C++20, no external dependencies; cl100k_base, o200k_base, o200k_harmony, Llama-3, and Qwen2.5/3 ship in the repo.
- **Thread-safe** — load once, call `encode()` from as many threads as you like.

## Install

**Python**:

```sh
pip install quicktok
```

```python
import quicktok
enc = quicktok.get_encoding("cl100k_base")        # or "o200k_base", "llama3"
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
| **quicktok** | **110.2** | **132.3** | **66.5** |
| bpe-openai | 33.9 | 39.5 | 27.2 |
| tiktoken-rs | 15.1 | 13.6 | 12.7 |
| tiktoken (Python) | 13.8 | 12.7 | 11.7 |
| TokenDagger | 10.7 | 11.5 | 10.5 |

**o200k_base** (GPT-4o)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** | **77.5** | **98.7** | **44.2** |
| bpe-openai | 32.4 | 37.3 | 27.6 |
| tiktoken-rs | 21.3 | 19.7 | 15.8 |
| tiktoken (Python) | 19.6 | 18.1 | 14.9 |
| TokenDagger | 10.3 | 11.1 | 9.2 |

quicktok is fastest in every case: **2.4–3.4× bpe-openai** and **6–10× tiktoken**
on cl100k, **1.6–2.6×** and **3–5×** on o200k (the 2× vocab narrows the gap, and
helps tiktoken more than us). Common Crawl — the most multilingual corpus — is our
weakest ratio. Absolute MB/s moves with corpus and host; the ordering doesn't.

**Reproduce it yourself:** `make bench` (native, single-thread + parallel scaling)
and `make bench-py` (quicktok vs tiktoken, needs `pip install tiktoken`) run on a
bundled 1 MB public-domain corpus — no network, no setup.

<details>
<summary><b>Parallel / batch scaling</b> (Apple M1, 8 threads)</summary>

<br>`encode_batch()` work-steals across a shared tokenizer. Native kernel throughput (`make bench`):

| threads | cl100k | speedup |
|---:|---:|---:|
| 1 | 110 MB/s | 1.0× |
| 2 | 210 MB/s | 1.9× |
| 4 | 397 MB/s | 3.6× |
| 8 | **706 MB/s** | **6.4×** |

From Python (`make bench-py`), cl100k, 10 threads, exact-checked first:

| Python API | quicktok | tiktoken | speedup |
|---|---:|---:|---:|
| single-thread | 77 MB/s | 15 MB/s | **5.0×** |
| `encode_batch` | **550 MB/s** | 24 MB/s (batch) | **24×** |

`encode_batch` returns one flat `uint32` token buffer plus an `int64` offsets
array (`tokens[offsets[i]:offsets[i+1]]` is document *i*) rather than building
thousands of Python lists — the marshalling cost that caps tiktoken's own batch
at ~1.5× its single-thread. That's how it keeps the full native scaling.
`count_batch(enc, texts)` returns per-document counts the same way.
</details>

<details>
<summary><b>x86 cross-check + size stability</b> (cl100k)</summary>

<br>The ordering holds on an x86 server too, exact-checked (`quicktok == bpe-openai
== tiktoken`), across more datasets:

| dataset | size | quicktok | bpe-openai | tiktoken-rs | vs bpe |
|---|---:|---:|---:|---:|---:|
| FineWeb | 15 MB | **80.9** | 28.2 | 12.8 | 2.87× |
| FineWeb | 100 MB | **74.5** | 26.5 | 12.1 | 2.81× |
| The Pile | 15 MB | **79.9** | 25.9 | 10.7 | 3.08× |
| C4 | 15 MB | **80.1** | 26.8 | 12.6 | 2.99× |
| SlimPajama | 15 MB | **76.4** | 25.8 | 11.8 | 2.96× |
| Wikipedia | 15 MB | **75.1** | 26.1 | 11.8 | 2.88× |

The 100 MB row matches the 15 MB row — the ratio is size-stable. (This particular
x86 box is slower in absolute MB/s than the M1 above; the *ratio* is what travels.)
</details>

<details>
<summary><b>vs llama.cpp</b> (x86 server, Llama-3 vocab, FineWeb 15 MB)</summary>

<br>Same Llama-3 128k vocab (`ggml-vocab-llama-bpe.gguf`), quicktok's kernel vs `libllama`'s own `llama_tokenize`:

| | MB/s | Mtok/s |
|---|---:|---:|
| **quicktok (Llama-3)** | **~78** | ~17 |
| llama.cpp | ~5.4 | ~1.2 |

Token agreement is 99.9998% (3,274,274 of 3,274,281); the 7 differing tokens are
a known tiktoken-rank vs HF-merges divergence on rare Cyrillic+symbol sequences,
not an encoder bug. See [Encodings](#encodings).
</details>

<details>
<summary><b>Notes on fairness</b></summary>

<br>Every reference is called through the same raw API its own benchmark uses
(e.g. `encode_ordinary`, `encode_via_backtracking`) — no convenience-wrapper
handicaps. Every comparison is exact-checked on the same bytes before timing.
TokenDagger's README claims 2–4× over tiktoken, but that's on Llama-4/Mistral
vocabs on AMD EPYC; on cl100k/o200k here it lands around Python tiktoken's level
(a little below, on these corpora). Building this repo reproduces ~120–135 MB/s on
the bundled M1 fixtures (`make example`).
</details>

## How it's fast

Same algorithm as bpe-openai (exact backtracking BPE) — the speed is data-structure engineering:

- **2-byte trie** — the longest-match walk reads 2 input bytes per single 16-byte cache load, with a zero-lookup direct table for CJK characters.
- **Dense validity memo** — merge-validity checks hit a 2 MB exactly-keyed cache (a bijective mixer means no aliasing, ever).
- **Specialized pretokenizer** — the fixed cl100k regex is compiled by hand into a scanner; no general regex engine anywhere.

## API

```cpp
namespace quicktok {
class Tokenizer {
    // encoding: "cl100k_base" (default), "o200k_base", or "llama3"
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
- `load*` throws `std::runtime_error` on missing or corrupt data files. Nothing throws on the encode hot path (one exception: inputs over 4 GiB per call are rejected).
- A loaded `Tokenizer` is safe to share across threads — concurrent `encode()`/`decode()` is supported and tested.

## Encodings

Five encodings ship in the repo, each byte-exact vs its reference:

| name | model family | reference | notes |
|---|---|---|---|
| `cl100k_base` | GPT-3.5 / GPT-4 | tiktoken | the default |
| `o200k_base` | GPT-4o | tiktoken | ~80% of cl100k speed (2× vocab), still ~2× bpe-openai |
| `o200k_harmony` | GPT-OSS (20b/120b) | tiktoken | same pattern + ranks as o200k_base, extra chat specials |
| `llama3` | Llama 3 | Meta tiktoken-rank | full cl100k speed; see exactness note |
| `qwen3` | Qwen2.5 / Qwen3 | HF tokenizers | cl100k speed; single-digit numbers |

```python
enc = quicktok.get_encoding("qwen3")          # or "o200k_harmony", "llama3", ...
```
```cpp
auto tok = quicktok::Tokenizer::load_dir("data", "qwen3");
```

**Qwen2.5 / Qwen3** share one byte-level BPE; quicktok reproduces the Hugging Face
tokenizer's merge-list output byte-for-byte by rank order (verified token-for-token
on a multilingual corpus). Apache-2.0 — `tools/export_qwen.py --download` regenerates
the vocab.

**o200k_harmony** (GPT-OSS) is o200k_base with the harmony chat special tokens
(`<|start|>`, `<|channel|>`, `<|return|>`, `<|call|>`, …); it reuses o200k's merge
ranks, so encoding ordinary text is identical to o200k_base.

**Llama-3** is derived from Meta's Llama 3 tokenizer, governed by the
[Meta Llama 3 Community License](https://llama.meta.com/llama3/license/) (see
[NOTICE](NOTICE)) — redistributed for interoperability following llama.cpp's
precedent. Regenerate from a Hugging Face `tokenizer.json` or llama.cpp GGUF with
`python tools/export_llama3.py <tokenizer.json> data`. quicktok reproduces Llama-3's
original **tiktoken-rank** BPE byte-for-byte; Hugging Face / llama.cpp infer the
same vocab via a **merge list** and agree on ~99.9998% of tokens, differing only on
rare non-Latin+symbol sequences (e.g. Cyrillic next to `€`) where rank order and
merge order pick different splits.

**Llama-4** has the same pretokenizer as o200k_base, so its encoding is wired in —
but Meta's Llama-4 vocab is gated and **not bundled**. Supply your own and quicktok
encodes it exactly:

```sh
python tools/export_llama4.py <tokenizer.model> data   # from Meta's gated tokenizer
```
```python
enc = quicktok.get_encoding("llama4", "data")
```

## Notes

- Builds tune to the host CPU by default (`-march=native`); set `CXXFLAGS_ARCH` for portable binaries.
- The bundled Unicode table is pinned and version-stamped; `python tools/export_unicode.py verify` re-derives all 1.1 M codepoints against the live reference and diffs them — see [`data/uniclass.bin.meta`](data/uniclass.bin.meta).
- To regenerate the data files from the references:

  ```sh
  pip install tiktoken regex tokenizers
  python tools/export_fixtures.py        # cl100k/o200k/o200k_harmony from tiktoken
  python tools/export_qwen.py --download  # data/qwen3.vocab (Apache-2.0)
  python tools/export_unicode.py         # data/uniclass.bin + version stamp
  python tools/gen_vectors.py            # test vectors (tiktoken encodings)
  ```

## License

MIT — see [LICENSE](LICENSE).
