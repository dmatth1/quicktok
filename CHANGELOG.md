# Changelog

All notable changes to quicktok. Format follows [Keep a Changelog](https://keepachangelog.com);
versioning is [SemVer](https://semver.org).

## [0.4.0]

### Changed
- **`encode()` now matches tiktoken's full signature and semantics** (behavior
  change): `encode(text, *, allowed_special=set(), disallowed_special="all")`.
  A special-token string in the input now raises `ValueError` by default (as
  tiktoken does), instead of being silently encoded as ordinary text — this
  closes a prompt-injection footgun for code that swapped quicktok in expecting
  tiktoken's guard. `allowed_special="all"` (or a set) encodes specials as ids;
  `disallowed_special=()` disables the check. The old "specials as plain text,
  never raises" behavior is exactly `encode_ordinary()`. Verified token-for-token
  vs tiktoken across the default-raise, allowed-set, disallowed-set, and ordinary
  paths for cl100k and o200k. (C++ `encode()` keeps ordinary semantics, matching
  tiktoken-rs, which also doesn't raise.)

### Added
- **`encode_to_numpy()`** (Python): like `encode()` but returns a `uint32` numpy
  array instead of a `list[int]`. On large inputs the list-building (millions of
  Python int objects) dominates the wheel's time — returning one contiguous
  buffer skips it, lifting single-call throughput ~50% (x86: cl100k 42→63,
  o200k 38→57 MB/s on a 25 MB doc, ~native). Matches tiktoken's
  `encode_to_numpy`, same `allowed_special`/`disallowed_special` args; pass
  `disallowed_special=()` for raw-text throughput. (`encode_batch` already
  returned flat arrays; this brings the single-input path to parity.)
- **More tiktoken-parity methods** (Python): `encode_single_token`,
  `is_special_token`, `max_token_value`, `token_byte_values` (lexicographic, as
  tiktoken's `sorted_token_bytes`), `decode_batch`, and an
  `encode_with_special_tokens` alias (the name tiktoken-rs / TokenDagger use).
  C++ gains `encode_ordinary`/`encode_with_special_tokens` aliases and
  `token_id(bytes)`. A tiktoken `Encoding` now swaps for `get_encoding(...)`
  across the common surface.
- **tiktoken-parity surface in Python**: `n_vocab` now means max token id + 1
  *including specials* (cl100k -> 100277, matching tiktoken — it previously
  returned the 100256 base count, a silent embedding-sizing footgun);
  `eot_token`, `special_tokens_set`, and `decode_single_token_bytes`;
  `decode()` raises `KeyError` on unknown ids (was: silently skipped) and takes
  `errors=` (default `"replace"`, like tiktoken) for tokens that split a UTF-8
  character — `decode_bytes()` remains the lossless path. C++ gains `n_vocab()`,
  `special_tokens()`, `known_id()`; `vocab_size()` keeps base-count semantics.
- `quicktok.import_tokenizer()` raises a catchable `ImportRefused(ValueError)`
  instead of calling `sys.exit` (the CLIs still exit nonzero); Hugging Face
  auth/gating errors are reported as such instead of "file not found"; specials
  beyond the loader caps are refused at import time with the offending token
  named; `python -m quicktok.import_tokenizer` works as an alias.


### Changed
- **Single-pass product machines for both pretok families** — `encode_core` now
  owns the full ASCII grammar inline (cl100k/Llama-3/Qwen family: standalone
  contractions, general 1-byte-prefix words, digit rules per family, punct runs,
  per-family whitespace cascade; o200k family: prefix-consumed-first word
  alternatives, case-aware UPPER*/LOWER+ letters, attached contractions, the
  `/`-tail on punct, o200k whitespace order). Non-ASCII falls back to the exact
  scalar scanner for one piece, so output is byte-exact by construction
  (vector + special suites for all encodings; 18/18 corpus exactness gates vs
  tiktoken). M1, interleaved old-vs-new on the 25 MB corpora: cl100k Pile
  +1.6% / code +4.9%; o200k code **+10.1%**; multilingual flat. Tekken keeps
  the plain fused loop (no product machine for its grammar yet).
  Note: a whole-piece hash fast path was evaluated and REJECTED at this corpus
  scale (L2 displacement outweighs the saved walk; wins seen on small inputs
  did not transfer).

### Added
- Open-model benchmark comparisons, bench-grade (M1, single thread, best-of-5,
  three real corpora, exactness-gated): vs llama.cpp on Llama-3 (~11–13×), vs
  Hugging Face `tokenizers` on Qwen3 (~20–33×, per-document timing), and vs
  TokenDagger on Llama-4 — its own headline vocab (~5.6–10×; BYO gated vocab).
  Reproduce with `bench/hf_qwen_bench.py`, `bench/llama4_bench.py`, and
  `bench/llamacpp_bench.cpp`.

### Added
- `encode_batch(..., with_special=True)` (C++ and Python): parallel batch
  encoding with special-token parsing — the chat-templated fine-tuning-data
  path. Same flat-arrays output; `count_batch` gains the flag too.
- **The importer ships in the wheel**: `quicktok.import_tokenizer(source, name)`
  (also `python -m quicktok.importer`) — source can be a local
  tokenizer.json/tekken.json, a URL, or a Hugging Face repo id (with
  `huggingface_hub`). Imported encodings are written to `$QUICKTOK_DATA` (default
  `~/.cache/quicktok`) and `get_encoding(name)` finds them automatically, so the
  flow from pip is: install → import once (verified) → use by name.
- `encoding_for_model` resolves HF-style ids (`meta-llama/Llama-3.1-8B`) and
  Llama-3 model names; unknown models point at `import_tokenizer`.
- Linux aarch64 wheels (native arm runner in the wheels matrix).
- README: a bulk-processing recipe (`encode_batch` → flat token file + offsets,
  `count_batch` for budgeting) — the dataset-prep path.
- **Tokenizer importer with built-in verification** (`tools/import_tokenizer.py`):
  imports any byte-level-BPE `tokenizer.json` / `tekken.json` whose pretokenizer
  matches a supported grammar (cl100k / o200k / llama3 / qwen / tekken) and whose
  normalizer is none/NFC. Emits `<name>.{vocab,special,enc}`; `load_dir` loads
  non-builtin names via the `.enc` sidecar. The import encodes a stress suite
  plus optional corpora with the reference and with quicktok (real library, via
  the C ABI) and fails on any token mismatch — no fallback modes, unrecognized
  patterns are refused with the pattern printed.
- **Mistral Tekken v3 support** (via the importer): the tekken pattern is the
  o200k grammar minus contractions with single-digit `\p{N}` — now a parametric
  scanner variant — and Tekken's 1,000 reserved special slots are handled with
  an id offset, so ids match `mistral-common` exactly (verified, 471k tokens).
  Full o200k-class speed. DeepSeek V3/R1 is the documented refusal case: a
  3-stage sequential-Split pipeline, structurally a different grammar.

### Fixed
- **`qwen3` now applies NFC normalization** (the step HF's pipeline runs before
  tokenizing), making it byte-exact vs the Hugging Face tokenizer on arbitrary
  input — previously exact only on NFC input (gap found by the 25 MB real-corpus
  exactness gate). Implementation is a quick-check fast path: clean input pays
  one cheap scan (throughput unchanged, A/B-verified); only spans containing
  non-NFC codepoints are normalized. Tables derived from Unicode 16 and pinned
  (`data/nfc.bin` + `tools/export_nfc.py verify`); malformed UTF-8 passes
  through normalization verbatim. Like HF, decode returns normalized text for
  non-NFC input (decode→re-encode is idempotent; tested).

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
