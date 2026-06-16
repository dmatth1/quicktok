# Benchmarks

Two tiers: a quick offline bench for development, and the full cross-encoder
harness that produces the tables in the main README.

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
