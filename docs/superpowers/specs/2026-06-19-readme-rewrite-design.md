# quicktok README rewrite — design

**Date:** 2026-06-19
**Branch:** `readme-rewrite`
**Goal:** Restructure the README into a lean, visual landing page (matching the
shape of comparable fast-tokenizer projects), relocating intricate detail into
topic-split sub-READMEs. Pure documentation change — no code, no re-benchmarking.

## Motivation

The current `README.md` is ~373 lines. Strong content, but the shape buries the
pitch: the benchmark section alone is ~150 lines (tables + four collapsible
`<details>` blocks), and full C++/Python API references plus a long
encodings/import section sit inline.

Comparable projects (fastokens, TokenDagger, tokie, tiktoken-rs) all converge on
the same shape: one-line headline + a bold speed claim, a **benchmark chart up
top** as visual proof, badges, a tight install + quickstart, and details pushed
down or out. This rewrite adopts that shape.

All benchmark numbers are carried over **verbatim** from the current README. This
is a restructure, not a re-measurement.

## Front page — `README.md` (~130–150 lines)

Section order (note: **Benchmarks moves above Install** — visual proof first):

1. **Title + badges** — PyPI (`quicktok-v1`), License MIT, CI (`ci.yml`),
   a "C++20 · zero-deps" shield.
2. **Pitch** — the existing one-liner + bold speed claim (2–3.4× the fastest
   exact tokenizer, 3.5–11× tiktoken; byte-exact). Short nav links:
   Install · Quickstart · Benchmarks · Encodings · Docs.
3. **Benchmarks**
   - One `mermaid xychart-beta` bar chart: encoder → MB/s, **cl100k_base · The
     Pile**, single thread, M1. ~5–6 encoders on the x-axis (quicktok native,
     quicktok Python, bpe-openai, tiktoken-rs, tiktoken, TokenDagger).
   - One compact headline table: cl100k, 3 corpora, the same encoder rows.
   - One line: "x86 cross-check, o200k, open-model, and batch-scaling numbers →
     [bench/README.md]".
4. **Install** — Python `pip install quicktok-v1` + C++ `find_package` snippet.
5. **Quickstart**
   - Python: ~8 lines (`get_encoding`, `encode`/`decode`, one `encode_batch`
     line). → "Full Python API → docs/python.md".
   - C++: ~6 lines (`load_dir`, `encode`, `decode`, `count`). → "Full C++ / C
     ABI → docs/cpp.md".
6. **How it's fast** — the existing four bullets, kept as-is (already tight).
7. **Encodings** — the 6-row encoding table only; per-encoding prose moves out.
   → "Encoding details + importing other tokenizers → docs/encodings.md".
8. **License** — MIT.

## Relocated detail — four homes

### `bench/README.md` (exists; gains a "## Results" section at top)
Absorbs everything pulled off the front page:
- Full cl100k + o200k **M1** tables (all 3 corpora, all 7 encoders).
- The **x86 cross-check** tables.
- **Open-model** tables (Llama-3 vs llama.cpp, Qwen3 vs HF tokenizers, Llama-4
  vs TokenDagger).
- **Parallel / batch-scaling** tables.
- **vs-tokie** tables.
- The narrative footnotes attached to each (thermal caveat, rs-bpe omission,
  TokenDagger single-token divergence, token-agreement percentages, etc.).
- The existing reproduce/method content stays **below** Results.

### `docs/python.md` (new)
- Every Python method, `encode_to_numpy`, `encode_batch` / `count_batch`,
  `import_tokenizer`, `encoding_for_model`.
- `$QUICKTOK_DATA` cache location, tiktoken-semantics notes (`allowed_special`,
  `encode_ordinary`, raises-on-stray-special), reference-install requirements.

### `docs/cpp.md` (new)
- The full `class Tokenizer` declaration block.
- `load_dir` directory semantics, throwing behavior, the >4 GiB rejection,
  thread-safety notes.
- **C ABI** pointer (`quicktok.h`) for FFI from other languages.

### `docs/encodings.md` (new)
- Per-encoding detail prose: qwen3 NFC normalization, llama3 rank-vs-merges
  divergence, llama4 gating + import/export instructions.
- The entire **"Importing other tokenizers"** section (the
  `python -m quicktok.importer` CLI; Tekken supported, DeepSeek refused,
  SentencePiece out of scope).
- The **"Notes"** block (data-file regeneration with `tools/export_*.py`,
  `-march=native` / `CXXFLAGS_ARCH`).

## The mermaid chart

Single `xychart-beta` bar chart (one bar series — `xychart-beta` does not do
grouped multi-series bars, so the headline chart shows one encoding/corpus; the
full grouped numbers live in the table beneath it and in `bench/README.md`):

- Title: `cl100k_base · The Pile · MB/s (single thread, M1)`
- x-axis: encoder names
- y-axis: MB/s
- bars: the cl100k / The Pile column from the current headline table
  (quicktok native 92.8, quicktok Python 64.3, bpe-openai 29.8, tiktoken-rs
  13.6, tiktoken 12.6, TokenDagger 9.7).

## Out of scope (YAGNI)
- No re-benchmarking; numbers are carried over verbatim.
- No committed SVG/PNG (mermaid renders inline, hand-editable).
- No code, build, or API changes.
- No new badges beyond the four listed.

## Verification before commit
- Mermaid block renders (valid `xychart-beta` syntax).
- Every relative link resolves (`bench/README.md`, `docs/python.md`,
  `docs/cpp.md`, `docs/encodings.md`, `LICENSE`, `NOTICE`).
- No benchmark number altered vs the current README (diff-check the values).
- CI badge points at the real workflow (`ci.yml`).
