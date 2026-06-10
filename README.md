# quicktok

Fast **exact** cl100k_base (GPT-3.5 / GPT-4) BPE tokenizer in C++. Token ids are
**byte-identical to tiktoken** — the reference *is* the spec, enforced by the test
suite. A faithful port of the `bpe` crate's backtracking encoder, made fast by
cache engineering: a 2-byte-radix trie, a bijective-mixed dense validity memo, and
a fused hand-coded cl100k pretokenizer.

On Apple M1 it encodes at **~125 MB/s on prose, ~130 MB/s on code** single-threaded
— measured **~3.4× the `bpe-openai` crate** (the fastest exact single-core GPT
byte-BPE reference) and ~20–35× tiktoken/HuggingFace, end-to-end, on the same
machine in the same thermal window. See [Performance](#performance).

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
  the merge walk with no per-piece dispatch.

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

Single-thread, Apple M1, end-to-end (raw text → ids), best-of-7, library build:

| corpus | throughput | tokens/s |
|--------|-----------:|---------:|
| code   | ~132 MB/s  | 30.9 Mtok/s |
| prose  | ~126 MB/s  | 29.8 Mtok/s |
| CJK    | ~97 MB/s   | 46.6 Mtok/s |

The **~3.4× vs `bpe-openai`** headline is from the evolve harness's interleaved
same-thermal-window comparison (`tokenizer/TOKENIZER_LOG.md`), exact-correct in
every cell. Absolute MB/s is data-dependent; numbers above are this repo's own
bench, not extrapolated. x86_64 builds and is correct (scalar/SSE2 paths); the
trie/memo wins are validated on aarch64 — x86 tuning is in progress.

## Regenerating data

```sh
pip install tiktoken regex
python tools/export_fixtures.py   # data/cl100k.vocab from tiktoken
python tools/export_unicode.py    # data/uniclass.bin + .meta stamp
python tools/gen_vectors.py       # test/vectors.bin
```

## License

MIT — see [LICENSE](LICENSE).
