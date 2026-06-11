# quicktok

A fast, exact BPE tokenizer for OpenAI encodings, written in C++. Token ids are
byte-identical to [tiktoken](https://github.com/openai/tiktoken), and encoding runs
about 3× faster than the fastest exact tokenizer we know of
([bpe-openai](https://github.com/github/rust-gems)) — 6–8× faster than tiktoken
itself. See [benchmarks](#benchmarks).

- **Exact** — ids match tiktoken byte-for-byte; every benchmark is exactness-checked before timing.
- **Drop-in** — Python wheels with a tiktoken-style API, a stable C ABI, CMake support.
- **Self-contained** — C++20, no external dependencies; cl100k_base, o200k_base, and Llama-3 ship in the repo.
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

Five encoders, same corpus, same vocab, same machine (Apple M1), single thread,
and every encoder's output verified token-for-token identical before timing:

| encoder | The Pile (40 MB) | code (20 MB) |
|---|---:|---:|
| **quicktok** | **112.8 MB/s** | **149.3 MB/s** |
| bpe-openai | 35.9 | 37.7 |
| tiktoken-rs | 14.9 | 12.9 |
| tiktoken (Python) | 14.0 | 12.1 |
| TokenDagger | 10.8 | 12.2 |

The ratio holds everywhere we've measured: **2.8–3.1× over bpe-openai** across
FineWeb, The Pile, C4, SlimPajama, and Wikipedia on x86; **2.5–2.8×** under
bpe-openai's own benchmark setup; **~14×** over llama.cpp on the Llama-3 vocab.
Absolute MB/s moves with corpus and host — the ratios don't.

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
| `encode_batch` → `list[list[int]]` | 150 MB/s | 24 MB/s (batch) | 6.7× |
| `encode_batch_numpy` → flat arrays | **550 MB/s** | 24 MB/s (batch) | **24×** |

`encode_batch` is bound by building thousands of Python lists (tiktoken's batch
is too). `encode_batch_numpy` skips that: it returns one flat `uint32` token
buffer plus an `int64` offsets array (`tokens[offsets[i]:offsets[i+1]]` is
document *i*), so it keeps the full native scaling. `count_batch(enc, texts)`
returns per-document counts the same way.
</details>

<details>
<summary><b>Real-world corpora</b> (x86 server, cl100k, MB/s)</summary>

<br>Each dataset streamed from the real source, exact-checked (`quicktok == bpe-openai == tiktoken` ids):

| dataset | size | quicktok | bpe-openai | tiktoken-rs | vs bpe | vs tiktoken-rs |
|---|---:|---:|---:|---:|---:|---:|
| FineWeb | 15 MB | **80.9** | 28.2 | 12.8 | 2.87× | 6.3× |
| FineWeb | 100 MB | **74.5** | 26.5 | 12.1 | 2.81× | 6.2× |
| FineWeb-Edu | 15 MB | **80.7** | 28.7 | 12.6 | 2.81× | 6.4× |
| The Pile | 15 MB | **79.9** | 25.9 | 10.7 | 3.08× | 7.5× |
| C4 | 15 MB | **80.1** | 26.8 | 12.6 | 2.99× | 6.3× |
| SlimPajama | 15 MB | **76.4** | 25.8 | 11.8 | 2.96× | 6.5× |
| Wikipedia | 15 MB | **75.1** | 26.1 | 11.8 | 2.88× | 6.4× |
| Common Crawl | 15 MB | **45.9** | 20.2 | 9.7 | 2.28× | 4.7× |

Common Crawl is the most multilingual corpus and our weakest ratio (2.28×).
The 100 MB row matches the 15 MB row — the ratio is size-stable.
</details>

<details>
<summary><b>bpe-openai's own microbenchmark</b> (x86 server, synthetic text, size sweep)</summary>

<br>Reproduces the reference's `performance.rs` setup exactly:

| input size | quicktok | bpe-openai | tiktoken-rs | vs bpe |
|---|---:|---:|---:|---:|
| 10 B | **24.9** | 9.9 | 5.3 | 2.51× |
| 100 B | **41.1** | 16.5 | 8.2 | 2.49× |
| 1 KB | **45.9** | 18.1 | 9.0 | 2.54× |
| 10 KB | **46.5** | 16.7 | 9.1 | 2.79× |

The win is *larger* on natural text than on synthetic — the real-corpus numbers
above are the conservative ones.
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
vocabs on AMD EPYC; on cl100k it lands at parity with Python tiktoken. Building
this repo reproduces ~120–135 MB/s on the bundled M1 fixtures (`make example`).
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

**cl100k_base** (GPT-3.5/4), **o200k_base** (GPT-4o), and **Llama-3** all ship in
the repo. o200k runs at ~80% of cl100k throughput (its vocab is twice the size) —
still about 2× bpe-openai. Llama-3 shares cl100k's grammar and runs at full speed:

```python
enc = quicktok.get_encoding("llama3")
```

```cpp
auto tok = quicktok::Tokenizer::load_dir("data", "llama3");
```

The bundled `data/llama3.vocab` is derived from Meta's Llama 3 tokenizer and is
governed by the [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/)
(see [NOTICE](NOTICE)) — redistributed for interoperability following llama.cpp's
precedent. Regenerate it from a Hugging Face `tokenizer.json` or llama.cpp GGUF
with `python tools/export_llama3.py <tokenizer.json> data` if you prefer.

**Llama-3 exactness:** quicktok reproduces Llama-3's original **tiktoken-rank**
BPE byte-for-byte. Hugging Face and llama.cpp infer the same vocab via a **merge
list**; the two agree on ~99.9998% of tokens, differing only on rare
non-Latin+symbol sequences (e.g. Cyrillic next to `€`), where the rank order and
the merge order pick different splits. Neither is "wrong"; quicktok matches
Meta's original tiktoken-format tokenizer.

## Notes

- Builds tune to the host CPU by default (`-march=native`); set `CXXFLAGS_ARCH` for portable binaries.
- The bundled Unicode table is pinned and version-stamped; `python tools/export_unicode.py verify` re-derives all 1.1 M codepoints against the live reference and diffs them — see [`data/uniclass.bin.meta`](data/uniclass.bin.meta).
- To regenerate the data files from the references:

  ```sh
  pip install tiktoken regex
  python tools/export_fixtures.py   # data/cl100k.vocab from tiktoken
  python tools/export_unicode.py    # data/uniclass.bin + version stamp
  python tools/gen_vectors.py       # test/vectors.bin
  ```

## License

MIT — see [LICENSE](LICENSE).
