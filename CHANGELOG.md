# Changelog

All notable changes to quicktok. Format follows [Keep a Changelog](https://keepachangelog.com);
versioning is [SemVer](https://semver.org).

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
- **Python package** (`pip install quicktok`) with a tiktoken-compatible API
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
