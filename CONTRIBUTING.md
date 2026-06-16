# Contributing

Thanks for your interest. quicktok values **exactness above all** — token ids must
match the reference tokenizer byte-for-byte.

## Building & testing

```sh
make test          # C++ + C ABI, all encodings exact vs committed vectors
make bench         # native throughput + parallel scaling
```

For the Python API: `pip install ".[test]" && pytest tests/` (parity vs tiktoken,
batch, importer). `make test` is the gate for the C++ core: it verifies every encoding against reference-derived vectors (tiktoken / Hugging Face / Meta)
and round-trips decode. CI also runs ASan+UBSan and a cross-platform build matrix.

## Ground rules

- **No exactness regressions.** If you touch the kernel or a pretokenizer, the vector
  tests must still pass. New behavior needs new vectors generated from the reference
  (`tools/gen_vectors.py`).
- **Measure speed claims.** Use `make bench`; report the machine.
- **Keep the public surface small.** Implementation lives in `quicktok::detail`.
- **Bump `VERSION`** for releases; `python tools/check_version.py` enforces that every
  file agrees.

## Reporting bugs

Open an issue with a minimal input that reproduces. For a tokenization mismatch,
include the input bytes and both the expected (tiktoken) and actual ids.
