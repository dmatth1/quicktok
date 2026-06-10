# quicktok

A fast, exact BPE tokenizer for OpenAI's `cl100k_base` (GPT-3.5 / GPT-4), written in C++.

- **Exact** — token ids are byte-identical to [tiktoken](https://github.com/openai/tiktoken). The reference is the spec, and the test suite enforces it.
- **Fast** — about **3×** the fastest exact tokenizer we know of ([bpe-openai](https://github.com/github/rust-gems)), **6–8×** tiktoken, **10×+** [TokenDagger](https://github.com/M4THYOU/TokenDagger) and llama.cpp. Single-threaded, measured against each project's own API ([benchmarks](#benchmarks)).
- **Self-contained** — C++20, no external dependencies, ships its own data files (~1.3 MB).
- **Thread-safe** — load once, call `encode()` from as many threads as you like.

## Quick start

```sh
git clone https://github.com/dmatth1/quicktok
cd quicktok
make            # builds build/libquicktok.{a, dylib/so}
make test       # verifies exact ids vs tiktoken + decode round-trip
```

```cpp
#include <quicktok.hpp>

auto tok = quicktok::Tokenizer::load_dir("data");
auto ids = tok.encode("Hello, quicktok! 日本語 🚀");  // std::vector<uint32_t>
std::string text = tok.decode(ids);                   // lossless round-trip
```

Link against `build/libquicktok.a`, or `make install` and use `pkg-config --cflags --libs quicktok`. The data files install to `share/quicktok`.

## Benchmarks

Five encoders, same corpus, same vocab, same machine (Apple M1), single thread, and every encoder's output verified token-for-token identical before timing:

| encoder | The Pile (40 MB) | code (20 MB) |
|---|---:|---:|
| **quicktok** | **112.8 MB/s** | **149.3 MB/s** |
| bpe-openai | 35.9 | 37.7 |
| tiktoken-rs | 14.9 | 12.9 |
| tiktoken (Python) | 14.0 | 12.1 |
| TokenDagger | 10.8 | 12.2 |

The ratio holds everywhere we've measured: **2.8–3.1× over bpe-openai** across FineWeb, The Pile, C4, SlimPajama, and Wikipedia on x86; **2.5–2.8×** under bpe-openai's *own* synthetic benchmark methodology; **~14×** over llama.cpp's tokenizer on the Llama-3 vocab. Throughput in MB/s varies with corpus and host — the ratios don't. Details below; full methodology in [evolve `BENCHMARKING.md`](https://github.com/dmatth1/evolve/blob/evolve-tokenizer/tokenizer/BENCHMARKING.md).

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

Common Crawl is the most multilingual corpus and our relative weak spot (2.28×). The 100 MB row matches the 15 MB row — the ratio is size-stable.
</details>

<details>
<summary><b>bpe-openai's own microbenchmark</b> (x86 server, synthetic text, size sweep)</summary>

<br>Reproduces the reference's `performance.rs` setup exactly — synthetic BPE-random text, swept input sizes:

| input size | quicktok | bpe-openai | tiktoken-rs | vs bpe |
|---|---:|---:|---:|---:|
| 10 B | **24.9** | 9.9 | 5.3 | 2.51× |
| 100 B | **41.1** | 16.5 | 8.2 | 2.49× |
| 1 KB | **45.9** | 18.1 | 9.0 | 2.54× |
| 10 KB | **46.5** | 16.7 | 9.1 | 2.79× |

The win is *larger* on natural text than on synthetic — the real-corpus numbers above are the conservative ones.
</details>

<details>
<summary><b>vs llama.cpp</b> (x86 server, Llama-3 vocab, FineWeb 15 MB)</summary>

<br>Same Llama-3 128k vocab (`ggml-vocab-llama-bpe.gguf`), quicktok's kernel vs `libllama`'s own `llama_tokenize`:

| | MB/s | Mtok/s |
|---|---:|---:|
| **quicktok kernel (Llama-3)** | **~78** | ~17 |
| llama.cpp | ~5.4 | ~1.2 |

Token agreement is 99.9998% (3,274,274 of 3,274,281); the 7 differing tokens are a known tiktoken-rank vs HF-merges divergence on rare Cyrillic+symbol sequences, not an encoder bug. The Llama-3 and o200k (GPT-4o) kernels live in the [evolve](https://github.com/dmatth1/evolve) lab today; packaging them into this library is in progress.
</details>

<details>
<summary><b>Notes on fairness</b></summary>

<br>Every reference is called through the same raw API its own benchmark uses (e.g. `encode_ordinary`, `encode_via_backtracking`) — no convenience-wrapper handicaps. Every comparison is exact-checked on the same bytes before timing. TokenDagger's README claims 2–4× over tiktoken, but that's on Llama-4/Mistral vocabs on AMD EPYC; on cl100k it lands at parity with Python tiktoken. Building this repo reproduces ~120–135 MB/s on the bundled M1 fixtures (`make example`).
</details>

## How it's fast

Same algorithm as bpe-openai (exact backtracking BPE) — the speed is data-structure engineering:

- **2-byte trie** — the longest-match walk reads 2 input bytes per single 16-byte cache load, with a zero-lookup direct table for CJK characters.
- **Dense validity memo** — merge-validity checks hit a 2 MB exactly-keyed cache (a bijective mixer means no aliasing, ever).
- **Specialized pretokenizer** — the fixed cl100k regex is compiled by hand into a scanner; no general regex engine anywhere.

The full measurement trail — every design decision, dead end, and benchmark — is in the [evolve](https://github.com/dmatth1/evolve) lab (`tokenizer/TOKENIZER_LOG.md`).

## API

```cpp
namespace quicktok {
class Tokenizer {
    static Tokenizer load_dir(const std::string& dir);   // expects cl100k.vocab + uniclass.bin
    static Tokenizer load(const std::string& vocab_path, const std::string& uniclass_path);

    std::vector<uint32_t> encode(std::string_view text) const;
    void encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const;

    std::string decode(const std::vector<uint32_t>& ids) const;
    void decode(const uint32_t* ids, size_t n, std::string& out) const;

    size_t vocab_size() const;
};
}
```

- `load*` throws `std::runtime_error` on missing or corrupt data files. Nothing throws on the encode hot path (one exception: inputs over 4 GiB per call are rejected).
- A loaded `Tokenizer` is safe to share across threads — concurrent `encode()`/`decode()` is supported and tested.
- Any byte sequence is accepted; invalid UTF-8 round-trips through encode/decode unchanged.
- Encoding is tiktoken's `encode_ordinary`: special tokens like `<|endoftext|>` are treated as plain text, not parsed.

## Notes & limitations

- **cl100k_base only** for now. o200k (GPT-4o) and Llama-3 support is in progress.
- Builds tune to the host CPU by default (`-march=native`); set `CXXFLAGS_ARCH` for portable binaries.
- The bundled Unicode table is pinned and version-stamped; `python tools/export_unicode.py verify` re-derives all 1.1 M codepoints against the live reference and diffs them — see [`data/uniclass.bin.meta`](data/uniclass.bin.meta).

## Regenerating data

```sh
pip install tiktoken regex
python tools/export_fixtures.py   # data/cl100k.vocab from tiktoken
python tools/export_unicode.py    # data/uniclass.bin + version stamp
python tools/gen_vectors.py       # test/vectors.bin
```

## License

MIT — see [LICENSE](LICENSE).
