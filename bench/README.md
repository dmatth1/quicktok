# Benchmarks

Two tiers: a quick offline bench for development, and the full cross-encoder
harness that produces the [Results](#results) tables below. The
[main README](../README.md#benchmarks) shows just the cl100k headline; the full
set lives here.

## Results

Same machine (Apple M1), single thread, every output verified token-for-token
identical before timing, every reference called through the same raw API its own
benchmark uses, one back-to-back run. Three 25 MB corpora streamed from their
real sources — **The Pile** (diverse), **GitHub code**, **Common Crawl**
(multilingual) — across both common OpenAI encodings (throughput in **MB/s**).
quicktok is shown three ways: native C++, the Python `encode_to_numpy()` fast
path, and the plain Python `encode()` (which builds a `list[int]`):

**cl100k_base** (GPT-3.5 / GPT-4)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok (native)** | **92.8** | **114.9** | **55.6** |
| **quicktok (Python, numpy)** | **92.5** | **109.5** | **54.9** |
| quicktok (Python) | 64.3 | 75.9 | 41.6 |
| bpe-openai | 29.8 | 34.1 | 24.0 |
| tiktoken-rs | 13.6 | 12.9 | 11.9 |
| tiktoken (Python) | 12.6 | 11.8 | 10.9 |
| TokenDagger | 9.7 | 10.4 | 9.3 |

**o200k_base** (GPT-4o)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok (native)** | **73.4** | **91.7** | **39.0** |
| **quicktok (Python, numpy)** | **72.0** | **93.7** | **39.0** |
| quicktok (Python) | 57.1 | 65.4 | 33.5 |
| bpe-openai | 27.4 | 32.4 | 22.5 |
| tiktoken-rs | 19.9 | 18.9 | 14.8 |
| tiktoken (Python) | 17.3 | 16.6 | 13.4 |
| TokenDagger | 8.9 | 9.9 | 7.9 |

`encode_to_numpy()` returns a `uint32` array directly, skipping the per-token
Python-list marshalling — so from Python it runs at **near-native speed** (~3× over
bpe-openai, ~7× over tiktoken). Absolute MB/s is machine- and thermal-dependent
(~15% swing on M1); the same-run ratios are the stable signal. (rs-bpe is omitted
from this run — no Python 3.14 wheel; it's a binding over the same `bpe` crate as
the bpe-openai row.)

### Parallel / batch scaling (Apple M1, 8 threads; measured at v0.3.2)

Native `encode_batch()` (`make bench`), cl100k:

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

### x86 cross-check (cl100k + o200k, current engine)

Same encoders, corpora, and method on an x86 server (Intel Xeon @ 2.8 GHz,
single thread, MB/s). quicktok is shown both as the native C++ kernel and as the
Python wheel:

**cl100k_base** (GPT-3.5 / GPT-4)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** (native) | **72.3** | **88.7** | **46.3** |
| quicktok (Python) | 59.4 | 86.7 | 37.8 |
| bpe-openai | 27.0 | 31.5 | 23.6 |
| tiktoken-rs | 11.7 | 10.3 | 11.2 |
| tiktoken (Python) | 10.4 | 9.2 | 9.6 |
| TokenDagger | 7.3 | 7.7 | 7.2 |

**o200k_base** (GPT-4o)

| encoder | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** (native) | **59.1** | **70.7** | **36.6** |
| quicktok (Python) | 50.3 | 68.5 | 32.6 |
| bpe-openai | 25.3 | 29.3 | 23.8 |
| tiktoken-rs | 17.3 | 15.7 | 15.4 |
| tiktoken (Python) | 14.4 | 13.6 | 13.9 |
| TokenDagger | 6.7 | 7.4 | 6.4 |

Same ordering as the M1 tables, the Python wheel included (o200k runs at a steady
~0.80× of cl100k native). (One footnote: TokenDagger diverges from the other four
by a single token on Pile/cl100k — a known TokenDagger edge case, not an encoder
bug.)

### Open-model encodings (Apple M1)

Same corpora and method as the headline tables (single thread, best-of-5, MB/s).

**Llama-3** — quicktok vs `libllama`'s own `llama_tokenize` (same 128k vocab,
vocab-only GGUF, llama.cpp built with `-DGGML_NATIVE=ON`):

| | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** | **124.2** | **139.8** | **66.5** |
| llama.cpp | 9.8 | 10.6 | 5.8 |

Token agreement is 99.999% on The Pile and Code and 99.81% on multilingual
Common Crawl — the known tiktoken-rank vs merge-list divergence (see
[encodings](../docs/encodings.md)); quicktok matches Meta's original tokenizer.

**Qwen3** — quicktok vs Hugging Face `tokenizers` (the Rust core behind
`AutoTokenizer`), timed per document (its best case; a single 25 MB string
drops it below 2.2 MB/s):

| | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok** | **100.4** | **124.5** | **61.5** |
| HF tokenizers | 3.4 | 3.8 | 3.1 |

quicktok's output was verified token-for-token identical to HF's on all three
raw corpora (quicktok implements the same NFC normalization HF runs — see the
Qwen note in [encodings](../docs/encodings.md)).

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

### vs tokie (chonkie-inc/tokie, x86 server, single thread)

tokie is a fast Rust BPE that targets **Hugging Face** output, not tiktoken — on
these 25 MB corpora it diverges from tiktoken on 0.02–0.14% of tokens (worst on
multilingual text), so it isn't a byte-exact tiktoken drop-in. Its `encode()` also
spreads one call across cores; both encoders are pinned to a single core here for an
equal-thread comparison (quicktok parallelizes via `encode_batch`), best-of-5, MB/s:

| cl100k_base | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok (native)** | **66.1** | **84.6** | **42.9** |
| tokie (native) | 37.5 | 50.4 | 19.4 |

| o200k_base | The Pile | Code | Common Crawl |
|---|---:|---:|---:|
| **quicktok (native)** | **56.5** | **74.6** | **34.1** |
| tokie (native) | 34.3 | 44.6 | 21.0 |

quicktok is ~1.6–2.2× tokie at equal threads and byte-exact vs tiktoken throughout.
Measured with the tokie Rust crate (`Tokenizer::from_json` on Xenova's gpt-4 /
gpt-4o `tokenizer.json`); numbers are core-pinned, so a touch below the unpinned x86
table above.

## Quick (offline, no setup)

```sh
make bench      # quicktok native: single-thread + parallel batch scaling
make bench-py   # quicktok vs tiktoken from Python (pip install tiktoken)
```

Both run on `corpus.txt` — ~1 MB of *Moby-Dick* (public domain, Project
Gutenberg), bundled so results are reproducible offline. Use these to sanity-check
a change; absolute MB/s on this corpus is *not* comparable to the README tables
(different data).

## Full README tables

```sh
pip install tiktoken datasets zstandard
make bench-compare          # or: python bench/compare.py
```

This fetches the same three 25 MB corpora the README uses (streamed from their
real sources into `bench/corpus/`, deterministic order), runs every available
encoder on each corpus × encoding, and prints the README-format markdown tables.

| row | requirement |
|---|---|
| quicktok (native) | none — built automatically |
| quicktok (Python) | `pip install quicktok-v1` — `encode_ordinary()` (list) |
| quicktok (Python, numpy) | same wheel — `encode_to_numpy()` (uint32 array; fast path for large inputs) |
| tiktoken (Python) | `pip install tiktoken` (required — it is the reference) |
| bpe-openai, tiktoken-rs | a Rust toolchain (`cargo`); compiled on first run from `comparators/` |
| TokenDagger | `TOKENDAGGER_DIR` pointing at a built clone — see `tokendagger_bench.cpp` |

Missing comparators are skipped with a note, so `pip install tiktoken` alone
already gets you quicktok-vs-tiktoken on the real corpora.

## Open-model comparisons (optional)

- **Qwen3 vs Hugging Face `tokenizers`**: `python bench/hf_qwen_bench.py`
  (needs `pip install tokenizers`). Fetches Qwen's tokenizer.json (Apache-2.0),
  NFC-normalizes the corpora (HF normalizes internally; see the main README's
  Qwen note), exact-checks quicktok against HF, times HF per document (its best
  case).
- **Llama-3 vs llama.cpp**: build libllama (`cmake -B build -DGGML_NATIVE=ON
  -DLLAMA_CURL=OFF && cmake --build build -j --target llama` in a llama.cpp
  clone), compile `bench/llamacpp_bench.cpp` against it (header comment has the
  command), run it per corpus (it dumps its ids), then time quicktok with
  `build/bench_file <corpus> llama3 data`. Agreement differs from 100% by the
  known rank-vs-merges divergence (worst on multilingual text).
- **Llama-4 vs TokenDagger**: `python bench/llama4_bench.py
  /path/to/tokenizer.model` (Meta's gated tiktoken-format vocab — a TokenDagger
  clone redistributes one at `src/tokenizer.model`). Reference is Meta's own
  tiktoken construction; set `TOKENDAGGER_DIR` for the TokenDagger row.

## Method

- **Single thread, best-of-5** wall-clock over the whole file; MB/s = bytes / best.
- **Exactness is a gate, not a footnote**: tiktoken defines the expected ids;
  every encoder's full output is compared token-for-token *before* it is timed,
  and any mismatch fails the run.
- **Raw APIs**: each comparator is called through the same API its own benchmark
  uses (`encode_ordinary`, bpe's `Tokenizer::encode`, TokenDagger's
  `CoreBPE::encode_ordinary`) — no convenience-wrapper handicaps.
- **One machine, one session**: ratios are only quoted between numbers measured
  in the same run; thermal drift makes cross-run comparisons of absolute MB/s
  unreliable.
- **Corpora**: The Pile (diverse English), GitHub Python (code), Common Crawl WET
  (multilingual — deliberately included as quicktok's weakest case). 25 MB each,
  documents joined by blank lines.
