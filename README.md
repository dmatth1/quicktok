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

**Reproduce it yourself:** `make bench` (native, single-thread + parallel scaling)
and `make bench-py` (quicktok vs tiktoken, needs `pip install tiktoken`) run on a
bundled 1 MB public-domain corpus — no network, no setup.

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
| **quicktok** (native) | **76.3** | **87.4** | **46.5** |
| quicktok (Python) | 42.5 | 47.4 | 29.6 |
| bpe-openai | 25.9 | 30.8 | 22.9 |
| tiktoken-rs | 11.1 | 10.2 | 11.1 |
| tiktoken (Python) | 10.2 | 9.1 | 9.5 |
| TokenDagger | 7.1 | 7.6 | 7.1 |

**o200k_base** (GPT-4o)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** (native) | **60.3** | **69.0** | **36.7** |
| quicktok (Python) | 36.6 | 41.1 | 26.7 |
| bpe-openai | 24.4 | 28.9 | 23.9 |
| tiktoken-rs | 17.0 | 15.5 | 15.1 |
| tiktoken (Python) | 14.2 | 13.3 | 13.2 |
| TokenDagger | 6.6 | 7.3 | 6.2 |

Same ordering as the M1 tables; even through the Python binding quicktok beats
bpe-openai in every cell and runs 2–5× Python tiktoken. (One footnote: TokenDagger
diverges from the other four by a single token on Pile/cl100k — a known TokenDagger
edge case, not an encoder bug.)
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
- Python's `encode_batch` returns a flat `uint32` token array plus `int64` offsets (`tokens[offsets[i]:offsets[i+1]]` is document *i*, no per-document Python lists); `count_batch(enc, texts)` gives per-document counts.
- `load*` throws `std::runtime_error` on missing or corrupt data files. Nothing throws on the encode hot path (one exception: inputs over 4 GiB per call are rejected).
- A loaded `Tokenizer` is safe to share across threads — concurrent `encode()`/`decode()` is supported and tested.

## Encodings

Five encodings ship in the repo, each byte-exact vs its reference:

| name | model family | reference | notes |
|---|---|---|---|
| `cl100k_base` | GPT-3.5 / GPT-4 | tiktoken | the default |
| `o200k_base` | GPT-4o | tiktoken | ~85% of cl100k speed (2× vocab), 2–2.9× bpe-openai |
| `o200k_harmony` | GPT-OSS (20b/120b) | tiktoken | same pattern + ranks as o200k_base, extra chat specials |
| `llama3` | Llama 3 | Meta tiktoken-rank | full cl100k speed; see exactness note |
| `qwen3` | Qwen2.5 / Qwen3 | HF tokenizers | cl100k speed; single-digit numbers |

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
  Face tokenizer byte-for-byte. Apache-2.0; regenerate with
  `tools/export_qwen.py --download`.
- **o200k_harmony** is o200k_base plus the harmony chat specials (`<|start|>`,
  `<|channel|>`, `<|return|>`, …) — ordinary text encodes identically to o200k_base.
- **Llama-3** reproduces Meta's original **tiktoken-rank** BPE byte-for-byte
  ([Meta Llama 3 Community License](https://llama.meta.com/llama3/license/), see
  [NOTICE](NOTICE); regenerate with `tools/export_llama3.py <tokenizer.json> data`).
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
