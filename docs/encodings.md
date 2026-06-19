# Encodings

[← back to README](../README.md)

Five encodings ship in the repo; Llama-4's code path ships too, but its vocab is
gated. Each is byte-exact vs its reference:

| name | model family | reference |
|---|---|---|
| `cl100k_base` | GPT-3.5 / GPT-4 | tiktoken (the default) |
| `o200k_base` | GPT-4o | tiktoken |
| `o200k_harmony` | GPT-OSS | tiktoken — o200k_base plus the harmony chat specials |
| `llama3` | Llama 3 | Meta's tiktoken-rank BPE |
| `qwen3` | Qwen2.5 / Qwen3 | Hugging Face tokenizers, including its NFC normalization |
| `llama4` | Llama 4 | Meta — **not bundled** (gated; bring your own vocab) |

- **qwen3** reproduces the HF pipeline including NFC normalization: clean input
  pays one cheap scan, only spans with non-NFC codepoints are normalized, and —
  like HF — decode returns the normalized text for such input.
- **llama3** matches Meta's original tiktoken-rank tokenizer. Hugging Face and
  llama.cpp infer the same vocab from a merge list and agree on ~99.9998% of
  tokens; the rare differences are non-Latin+symbol sequences where rank order
  and merge order pick different splits.
- **llama4** shares o200k_base's pretokenizer, but Meta gates the vocab. With
  repo access, import it like any other tokenizer —
  `quicktok.import_tokenizer("meta-llama/Llama-4-Scout-17B-16E-Instruct", "llama4")`
  (the import verifies, so a wrong guess can't ship; this checks against HF's
  merge-list tokenizer, the same rank-vs-merges nuance as llama3). To match
  Meta's original rank file instead, export from a checkout:
  `python tools/export_llama4.py <tokenizer.model> data`.

Vocabs regenerate from their references with `tools/export_*.py`; the Unicode and
NFC tables are pinned, version-stamped, and exhaustively re-derivable
(`tools/export_unicode.py verify`, `tools/export_nfc.py verify`). Third-party
vocab licenses: [NOTICE](../NOTICE).

## Importing other tokenizers

Any byte-level-BPE tokenizer whose pretokenizer matches one of quicktok's
hand-compiled grammars (cl100k / o200k / llama3 / qwen / tekken) can be imported —
`quicktok.import_tokenizer()` from Python (see [docs/python.md](python.md)), or:

```sh
python -m quicktok.importer path/to/tokenizer.json myenc --corpus big.txt
```

The import checks the normalizer (none/NFC), classifies the pretokenizer regex,
writes the data files, then encodes a stress suite plus any `--corpus` files with
both the reference tokenizer and quicktok and compares token-for-token. A
mismatch fails the import; an unrecognized pattern is refused with the pattern
printed. There is no fallback regex engine and no approximate mode — each grammar
is compiled by hand, which is where the speed comes from.

- **Mistral Tekken v3** imports and verifies: the o200k grammar minus
  contractions, single-digit numbers, ids offset by its 1,000 reserved special
  slots — exact vs `mistral-common`, at full o200k-class speed.
- **DeepSeek V3/R1** is refused: a pipeline of three sequential Split regexes, a
  different grammar shape. Supporting it would be a new scanner, not an import.
- **SentencePiece models** (Gemma, T5, Llama-2) are a different algorithm
  entirely, not a missing grammar — out of scope.

## Notes

- Builds tune to the host CPU by default (`-march=native`); set `CXXFLAGS_ARCH`
  for portable binaries.
- To regenerate the data files from the references:

  ```sh
  pip install tiktoken regex tokenizers
  python tools/export_fixtures.py        # cl100k/o200k/o200k_harmony from tiktoken
  python tools/export_qwen.py --download  # data/qwen3.vocab (Apache-2.0)
  python tools/export_unicode.py         # data/uniclass.bin + version stamp
  python tools/export_nfc.py             # data/nfc.bin (NFC tables) + version stamp
  python tools/gen_vectors.py            # test vectors (tiktoken encodings)
  ```
