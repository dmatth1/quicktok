# Changelog

All notable changes to quicktok. Format follows [Keep a Changelog](https://keepachangelog.com);
versioning is [SemVer](https://semver.org).

## [Unreleased]

### Added
- Open-model benchmark comparisons, bench-grade (M1, single thread, best-of-5,
  three real corpora, exactness-gated): vs llama.cpp on Llama-3 (~11–13×), vs
  Hugging Face `tokenizers` on Qwen3 (~20–33×, per-document timing), and vs
  TokenDagger on Llama-4 — its own headline vocab (~5.6–10×; BYO gated vocab).
  Reproduce with `bench/hf_qwen_bench.py`, `bench/llama4_bench.py`, and
  `bench/llamacpp_bench.cpp`.

### Known limitations
- `qwen3` does not yet apply the NFC normalization HF's pipeline runs before
  tokenizing: output is byte-exact vs HF on NFC input; input with non-NFC
  codepoints (rare) tokenizes the raw bytes instead. Found by the 25 MB
  real-corpus exactness gate.

### Changed
- **o200k-class encodings are faster** (o200k_base, o200k_harmony, Llama-4) —
  closes part of the large-vocab gap vs cl100k (x86, single thread: +16% on
  The Pile, +8% on multilingual Common Crawl, +3% on code; o200k went from
  ~0.68× to ~0.77× of cl100k's MB/s on the same corpora):
  - the o200k pretokenizer got the same ASCII SIMD fast paths the cl100k scanner
    already had (case runs for the UPPER*/LOWER+ alternatives, punct/whitespace
    runs), plus an empty-run pre-check so probing UPPER* on a lowercase word
    costs one byte-compare;
  - the o200k encode loop got the fused ASCII-word fast path cl100k already had
    (single `next_match` walk; single-token words skip the merge driver), with
    o200k's case-split and attached-contraction semantics;
  - the wide-id validity memo (vocabs with ids ≥ 2^17) now uses the same dense
    bijective-mixer scheme as cl100k, widened to a 36-bit pair key: u32 slots,
    exact (no aliasing), 2× entries per cache line; capacity is per-arch
    (`IVBITS_W`, 2^22 on x86 — sweeps best under a 33 MB L3 — 2^20 elsewhere,
    half the bytes of the u64 memo it replaces at equal capacity).
- **Packed trie + pair tables** — the 2-byte-radix trie's edge table now stores
  8-byte tagged slots instead of 16-byte key+value pairs (a bijective 36-bit
  mixer makes index+tag reconstruct the key exactly; a build-time cluster-length
  invariant makes tag matching deterministic under linear probing — no aliasing,
  ever). Halves the dominant hot table (o200k 8.4→4.2 MB, cl100k 4.2→2.1 MB)
  and each probe step is one load instead of two. `pair_lookup` likewise packs
  key+rank into one u64 slot (12→8 B). With the smaller hot set the wide-memo
  optimum re-swept to 2^21 on x86. Net on top of the scanner/fused/memo work
  (x86, single thread): o200k The Pile +6.6%, code +4%, Common Crawl +3.3%;
  cl100k flat to +4%. New constraint: vocabs are limited to 2^20 trie nodes
  (~2.5× o200k's; `load()` fails cleanly past it).

  Exactness unchanged and re-verified after every step: token-for-token vs
  tiktoken on 3×25 MB corpora (Pile / code / multilingual CC) for both cl100k
  and o200k, plus 20k-case differential fuzz over case-transition/contraction/
  Unicode edge inputs. ARM is functionally identical but unmeasured (scalar
  fallbacks).

## [0.3.1]

Packaging and docs only — no code, API, or tokenization changes.

### Fixed
- The wheel now bundles **NOTICE** (`license-files`), so the attribution for the
  redistributed tiktoken (OpenAI) and Llama-3 (Meta) vocab data travels with the
  package.
- README links are absolute, so they resolve on the PyPI project page; stale
  package-name references corrected.

## [0.3.0]

First release intended for general use. Adds two encodings, language bindings, and
a build/packaging story for downstream consumers.

### Added
- **o200k_base** (GPT-4o) encoding — exact vs tiktoken, bundled.
- **o200k_harmony** (GPT-OSS) encoding — exact vs tiktoken, bundled. Shares
  o200k_base's merge ranks; adds the harmony chat special tokens.
- **Qwen2.5 / Qwen3** (`qwen3`) encoding — exact vs the Hugging Face tokenizer
  (rank-order backtracking reproduces its merge list), bundled (Apache-2.0).
- **Llama-4** (`llama4`) encoding — pretokenizer wired in (identical to o200k_base);
  vocab is gated and not bundled (bring your own via `tools/export_llama4.py`).
- **Llama-3** encoding — exact vs Llama-3's tiktoken-rank BPE, bundled (Meta Llama 3
  Community License; see [NOTICE](NOTICE)).
- **Special tokens**: `encode_with_special()` (tiktoken `allowed_special="all"`
  semantics) and special-id decoding; `count()` for token budgeting.
- **Parallel batch**: C++ `encode_batch()`; Python `encode_batch()` (flat token
  buffer + offsets — `tokens[offsets[i]:offsets[i+1]]` is document *i* — ~24×
  tiktoken's batch) and `count_batch()`.
- **Python package** (`pip install quicktok-v1`) with a tiktoken-compatible API
  (`get_encoding`, `encoding_for_model`, `encode`/`encode_ordinary`, `decode`, …).
- **C ABI** (`quicktok.h`) for FFI from any language.
- **CMake** support: `find_package(quicktok)`, FetchContent, install export.
- `make bench` / `make bench-py` reproducible benchmarks on a bundled corpus.

### Changed
- Internals namespaced under `quicktok::detail`; only `quicktok::Tokenizer` is public.
- Loads throw `std::runtime_error` (never `exit()`); inputs are validated.
- Memo access is relaxed-atomic — concurrent `encode()` on one tokenizer is well-defined.
- Portable by default (baseline SSE2/NEON); `-march=native` is opt-in.

### Fixed
- **Out-of-bounds read fixed** in the pretokenizer: a prefix probe could read one
  byte past the input when the last piece was a lone non-letter (benign result, but
  UB / crash risk on a tightly-bounded buffer). Found by the fuzzer.
- **Lossless on invalid UTF-8.** Ill-formed multibyte sequences (e.g. truncated
  streams) no longer corrupt `encode` output — the 3-byte-char fast table is used
  only for well-formed input, ill-formed bytes take the byte-accurate path. Found
  by the new fuzzer; valid UTF-8 is unaffected.
- `make install` now ships every encoding's data (previously omitted o200k).

## [0.2.0] — o200k groundwork
## [0.1.0] — initial cl100k_base release
