# quicktok

Fast **exact** cl100k_base (GPT-3.5 / GPT-4) BPE tokenizer in C++. Token ids are
**byte-identical to tiktoken** — the reference *is* the spec, enforced by the test
suite. A faithful port of the `bpe` crate's backtracking encoder, made fast by
cache engineering: a 2-byte-radix trie, a bijective-mixed dense validity memo, and
a fused hand-coded cl100k pretokenizer.

**~2.8–3.1× faster than [`bpe-openai`](https://github.com/github/rust-gems)** (the
fastest exact single-core GPT byte-BPE reference), **~6–7× tiktoken**, **~10–12×
[TokenDagger](https://github.com/M4THYOU/TokenDagger)**, and **~14× llama.cpp's
tokenizer** — single-thread, exact-correct, measured in-harness against each
reference's raw API. See [Performance](#performance).

## Quick start

```sh
git clone https://github.com/dmatth1/quicktok
cd quicktok
make            # build/libquicktok.{a,dylib/so}
make test       # verify exact ids vs tiktoken + decode round-trip
make example
```

```cpp
#include <quicktok.hpp>

auto tok = quicktok::Tokenizer::load_dir("data");   // ships cl100k.vocab + uniclass.bin
auto ids = tok.encode("Hello, quicktok! 日本語 🚀"); // -> std::vector<uint32_t>
std::string text = tok.decode(ids);                 // lossless round-trip
```

Link with `build/libquicktok.a` (or `-lquicktok` after `make install`), and point
`load_dir` at the installed `share/quicktok` data dir. C++20, header is
`include/quicktok.hpp`.

## Why it's fast

Same algorithm as `bpe-openai` (exact backtracking); the wins are all in the data
structures (full trail in the [evolve](https://github.com/dmatth1/evolve) lab):

- **2-byte-radix trie** — the longest-match walk consumes 2 bytes per single
  aligned-16B load (child + that step's best token packed together), ~1.55× fewer
  probes/byte than a byte trie at the *same* memory footprint. Odd-depth tokens use
  a 1 MB side table probed only on miss; 3-byte UTF-8 (CJK) starts resolve their
  first char with a zero-probe direct table.
- **Dense validity memo** — `is_valid_token_pair` is memoized in a u16/slot table
  (2 MB) via a bijective 34-bit mixer, so the pair key round-trips exactly (no
  aliasing) at 4× the density of a naive 64-bit memo, freeing L2 for the trie.
- **Fused, specialized pretokenizer** — the fixed cl100k regex is hand-coded
  (no general regex engine); the dominant ASCII-word case is fused straight into
  the merge walk with no per-piece dispatch. (vs llama.cpp, whose BPE pretok runs
  a general regex and whose merge is a heap-based bigram loop.)

## Exactness

`make test` checks 15 stress vectors (ASCII, Latin diacritics, CJK, emoji,
whitespace runs, contractions, URLs, symbol soup, mixed scripts) for `encode ==
tiktoken cl100k_base` *and* `decode(encode(x)) == x`. The bundled
`data/uniclass.bin` (the `\p{L}`/`\p{N}`/`\s` table the pretokenizer needs) is
pinned and version-stamped (`data/uniclass.bin.meta`); `tools/export_unicode.py
verify` re-derives all 1,114,112 codepoints against the live `regex` engine and
diffs — a complete equivalence proof per Unicode version, not a snapshot you have
to trust.

## Performance

All numbers are single-thread, exact-correct (ids verified identical to the
reference on the same bytes), measured in our own harness calling each reference
through the **same raw API its own benchmark uses**. Tokenizer MB/s is
data-dependent and host-dependent — **lead with the ratios**, which are stable
across hosts; absolute MB/s is labeled per host. Methodology:
[evolve `BENCHMARKING.md`](https://github.com/dmatth1/evolve/blob/evolve-tokenizer/tokenizer/BENCHMARKING.md).

### Real-world corpora — the workload where tokenizer speed matters

cl100k, single-thread, MB/s *(x86 server host)*. Each corpus streamed from the real
dataset, capped, exact-checked (`quicktok == bpe-openai == tiktoken` ids):

| dataset | size | tok/byte | **quicktok** | bpe-openai | tiktoken-rs | vs bpe | vs tiktoken-rs |
|---|---:|---:|---:|---:|---:|---:|---:|
| FineWeb         | 15 MB  | 0.218 | **80.9** | 28.2 | 12.8 | **2.87×** | 6.3× |
| FineWeb         | 100 MB | 0.217 | **74.5** | 26.5 | 12.1 | **2.81×** | 6.2× |
| FineWeb-Edu     | 15 MB  | 0.213 | **80.7** | 28.7 | 12.6 | **2.81×** | 6.4× |
| The Pile        | 15 MB  | 0.244 | **79.9** | 25.9 | 10.7 | **3.08×** | 7.5× |
| C4              | 15 MB  | 0.212 | **80.1** | 26.8 | 12.6 | **2.99×** | 6.3× |
| SlimPajama      | 15 MB  | 0.226 | **76.4** | 25.8 | 11.8 | **2.96×** | 6.5× |
| Wikipedia       | 15 MB  | 0.224 | **75.1** | 26.1 | 11.8 | **2.88×** | 6.4× |
| Common Crawl    | 15 MB  | 0.344 | **45.9** | 20.2 |  9.7 | **2.28×** | 4.7× |

The win holds on every corpus: **~2.8–3.1×** over bpe-openai on English web /
encyclopedic text, **~6–7.5×** over tiktoken-rs. Common Crawl is the most
multilingual (highest tok/byte) and our relative weak spot at 2.28×. The 100 MB
row ≈ the 15 MB row — the ratio is size-stable.

### Head-to-head incl. TokenDagger & Python tiktoken

cl100k, single-thread, MB/s *(Apple M1 host)*, every encoder exact-correct on the
same corpus + same vocab + same oracle:

| corpus | **quicktok** | bpe-openai | tiktoken-rs | tiktoken (py) | TokenDagger | vs TokenDagger |
|---|---:|---:|---:|---:|---:|---:|
| The Pile (40 MB) | **112.8** | 35.9 | 14.9 | 14.0 | 10.8 | **10.4×** |
| code (20 MB)     | **149.3** | 37.7 | 12.9 | 12.1 | 12.2 | **12.3×** |

[TokenDagger](https://github.com/M4THYOU/TokenDagger) is a PCRE2-based tiktoken
drop-in; on cl100k it lands ~tied with Python tiktoken (its README's 2×/4× claim is
vs Python tiktoken on *Llama-4/Mistral* vocabs on an AMD EPYC, a different regex
path). bpe-openai remains the only real exact-BPE competitor, and quicktok beats it
~3–4× here.

### Microbenchmark — bpe-openai's own synthetic methodology

The reference's own `performance.rs` setup: synthetic BPE-random text, input-size
sweep, cl100k, MB/s *(x86 server host)*, exact on the synthetic string:

| input size | **quicktok** | bpe-openai | tiktoken-rs | vs bpe |
|---|---:|---:|---:|---:|
| 10 B    | **24.9** |  9.9 | 5.3 | 2.51× |
| 100 B   | **41.1** | 16.5 | 8.2 | 2.49× |
| 1 KB    | **45.9** | 18.1 | 9.0 | 2.54× |
| 10 KB   | **46.5** | 16.7 | 9.1 | **2.79×** |

The win is, if anything, *larger* on natural text than on bpe's synthetic data —
natural text has the recurring structure our pretok specialization exploits. So the
real-corpus tables above are conservative relative to the reference's own data.

### vs llama.cpp's tokenizer

Same Llama-3 128k vocab (`ggml-vocab-llama-bpe.gguf`), `quicktok` kernel vs
`libllama`'s own `llama_tokenize`, FineWeb 15 MB *(x86 server host)*:

| | MB/s | Mtok/s | vs llama.cpp |
|---|---:|---:|---:|
| **quicktok kernel (Llama-3)** | **~78** | ~17 | **~14×** |
| llama.cpp `llama_tokenize`     | ~5.4 | ~1.2 | 1× |

llama.cpp's BPE merge is a heap-based bigram loop with string-pair-keyed lookups
and a general-regex pretok — structurally the slow design. Agreement is
**99.9998%** (3,274,274 / 3,274,281 tokens); the ~7 differing tokens are a genuine
*tiktoken-rank vs HF-merges* BPE divergence on rare Cyrillic+symbol sequences, not
an encoder bug (quicktok is byte-exact vs tiktoken on cl100k/o200k).

> **Vocab support:** cl100k ships in this repo today. The same kernel runs **o200k**
> (GPT-4o; ~2.1× bpe-openai, ~2.8× tiktoken-rs) and **Llama-3** in the
> [evolve](https://github.com/dmatth1/evolve) lab — packaging those vocabs into the
> library is in progress.

**Building this repo** reproduces ~120–135 MB/s on the bundled M1 fixtures
(`make example`); the tables above are from the evolve harness's interleaved /
best-of comparisons. Absolute MB/s shifts with corpus and host; the ratios don't.

## Using it as a library

- **Errors:** `Tokenizer::load*` throws `std::runtime_error` on missing/corrupt
  data files (with validation: token count, ranks, record bounds — never `exit()`).
  `encode()` throws `std::invalid_argument` past 4 GiB per call. Nothing else throws
  on the hot path.
- **Thread safety:** a loaded `Tokenizer` is immutable except an internal memo whose
  slots are written with relaxed atomics (each slot value is self-consistent), so
  **concurrent `encode()`/`decode()` on one shared `Tokenizer` is safe** — covered by
  an 8-thread test in CI. Loading is not thread-safe with concurrent use of the same
  object (load first, then share).
- **Input handling:** any byte sequence is accepted; invalid UTF-8 is treated as
  1-byte units (matching the reference's behavior on the valid-UTF-8 inputs it
  defines; round-trip through `decode` is preserved either way). `decode()` skips
  ids `>= vocab_size()`.
- **Symbols:** all internals live under `quicktok::detail`; the only public surface
  is `quicktok::Tokenizer`.
- **Limitations (v0.1):** cl100k_base only (o200k/Llama-3 kernels exist in the
  [evolve](https://github.com/dmatth1/evolve) lab, packaging in progress; vocabs
  >131,072 ids are rejected at load by design — the id packing is 17-bit). Special
  tokens (`<|endoftext|>` etc.) are not parsed — input is encoded as ordinary text,
  i.e. tiktoken's `encode_ordinary`. Default build tunes `-mcpu/-march=native`;
  override `CXXFLAGS_ARCH` for portable binaries.

## Regenerating data

```sh
pip install tiktoken regex
python tools/export_fixtures.py   # data/cl100k.vocab from tiktoken
python tools/export_unicode.py    # data/uniclass.bin + .meta stamp
python tools/gen_vectors.py       # test/vectors.bin
```

## License

MIT — see [LICENSE](LICENSE).
