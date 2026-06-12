#!/usr/bin/env python3
"""Import a byte-level-BPE tokenizer into quicktok — with verification built in.

  import_tokenizer.py <tokenizer.json|tekken.json> <name> [options]

  --data-dir DIR     where to write <name>.{vocab,special,enc}  (default: data/)
  --corpus FILE...   extra text files for the exactness verification
  --lib PATH         libquicktok dylib/so (default: build it via `make lib`)

What it does, in order — and it HARD-FAILS rather than emit a wrong encoding:

  1. Parse. HF `tokenizer.json` (normalizer / pre_tokenizer / BPE model /
     added_tokens) or Mistral-style `tekken.json` (config.pattern + base64
     vocab). Anything structurally unexpected is an error.
  2. Normalizer: absent or NFC are supported (NFC is implemented natively);
     anything else is refused.
  3. Pretokenizer regex: classified against quicktok's hand-compiled scanner
     grammars (cl100k / o200k / llama3 / qwen). quicktok has no general regex
     engine — that's why it's fast — so an unrecognized pattern is refused
     with the pattern printed for comparison. New grammars are deliberate
     engineering decisions, not silent fallbacks.
  4. Emit <name>.vocab / <name>.special / <name>.enc.
  5. VERIFY: encode a builtin stress suite (plus any --corpus files) with the
     reference implementation and with quicktok (via the C ABI), and compare
     token-for-token. On any mismatch the .enc is deleted and the import fails.

Exactness means the whole pipeline (normalizer + pretokenizer + merges), and
rank-order BPE reproducing a merge-list tokenizer is an empirical property of
each vocab — which is why verification is part of the import, not optional.
"""
import argparse, ctypes, os, struct, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pure logic shared with the packaged importer (quicktok/_import_core.py),
# loaded from the source tree so this CLI works without a built wheel.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "qt_import_core", os.path.join(ROOT, "python", "quicktok", "_import_core.py"))
_core = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_core)
KNOWN_PATTERNS = _core.KNOWN_PATTERNS
parse_hf, parse_tekken = _core.parse_hf, _core.parse_tekken
write_files, ensure_tables, STRESS = _core.write_files, _core.ensure_tables, _core.STRESS


class QT:
    """quicktok via the C ABI (so verification runs the real library)."""
    def __init__(self, lib_path, data_dir, name):
        L = ctypes.CDLL(lib_path)
        L.qt_load_dir.restype = ctypes.c_void_p
        L.qt_load_dir.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        L.qt_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
                                ctypes.POINTER(ctypes.c_size_t)]
        L.qt_ids_free.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        err = ctypes.create_string_buffer(256)
        self.h = L.qt_load_dir(data_dir.encode(), name.encode(), err, 256)
        if not self.h:
            raise RuntimeError(f"qt_load_dir failed: {err.value.decode()}")
        self.L = L

    def encode(self, b):
        ids = ctypes.POINTER(ctypes.c_uint32)()
        n = ctypes.c_size_t()
        if self.L.qt_encode(self.h, b, len(b), ctypes.byref(ids), ctypes.byref(n)):
            raise RuntimeError("qt_encode failed")
        out = ids[:n.value]
        self.L.qt_ids_free(ids)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source"); ap.add_argument("name")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--corpus", nargs="*", default=[])
    ap.add_argument("--lib", default=None)
    a = ap.parse_args()

    is_tekken = "tekken" in os.path.basename(a.source).lower()
    parsed = parse_tekken(a.source) if is_tekken else parse_hf(a.source)
    ranks, specials, pattern, nfc, id_offset = parsed

    scanner = KNOWN_PATTERNS.get(pattern)
    if scanner is None:
        sys.exit("REFUSED: unrecognized pretokenizer pattern — quicktok compiles each\n"
                 "grammar by hand (no general regex engine; that is why it is fast).\n"
                 "If this model matters, file an issue with the pattern below; a close\n"
                 "variant of a supported grammar is usually a small change.\n\n"
                 f"  pattern: {pattern}\n\n"
                 "  supported grammars: cl100k, o200k (Llama-4), llama3, qwen")
    # the cl100k-string vs llama3 ambiguity is real: identical regex, but tiktoken's
    # cl100k uses possessive quantifiers with different whitespace endings. The HF
    # Split form matches llama3/qwen semantics; keep the mapping as classified.
    print(f"classified pretokenizer -> scanner '{scanner}'  (nfc={'yes' if nfc else 'no'}"
          + (f", id_offset={id_offset}" if id_offset else "") + ")")
    if max(ranks.values()) >= (1 << 18):
        sys.exit(f"REFUSED: max token id {max(ranks.values())} exceeds the 18-bit engine limit.")

    write_files(a.data_dir, a.name, ranks, specials, scanner, nfc, id_offset)
    ensure_tables(a.data_dir, os.path.join(ROOT, "data"), scanner, nfc)
    print(f"wrote {a.name}.vocab ({len(ranks)} tokens), .special ({len(specials)}), .enc")

    # ---- verification (the point of this tool) ----
    lib = a.lib
    if not lib:
        subprocess.run(["make", "-C", ROOT, "lib"], check=True, capture_output=True)
        import glob
        cands = glob.glob(os.path.join(ROOT, "build", "libquicktok.dylib")) + \
                glob.glob(os.path.join(ROOT, "build", "libquicktok.so"))
        lib = cands[0]
    qt = QT(lib, a.data_dir, a.name)

    if is_tekken:
        # preferred oracle: mistral-common (Mistral's own tokenizer — real model
        # ids, including the special-token offset). Fallback: tiktoken over the
        # same ranks + the offset (the construction tekken documents).
        try:
            from mistral_common.tokens.tokenizers.tekken import Tekkenizer, SpecialTokenPolicy
            tk = Tekkenizer.from_file(a.source)
            tk.special_token_policy = SpecialTokenPolicy.IGNORE
            ref_encode = lambda s: tk.encode(s, bos=False, eos=False)
            print("reference: mistral-common Tekkenizer")
        except ImportError:
            import tiktoken
            enc = tiktoken.Encoding(name=a.name, pat_str=pattern, mergeable_ranks=ranks,
                                    special_tokens={})
            ref_encode = lambda s: [i + id_offset for i in enc.encode_ordinary(s)]
            print("reference: tiktoken construction + id offset "
                  "(pip install mistral-common for the authoritative oracle)")
    else:
        from tokenizers import Tokenizer
        hf = Tokenizer.from_file(a.source)
        ref_encode = lambda s: hf.encode(s, add_special_tokens=False).ids

    texts = list(STRESS)
    for p in a.corpus:
        texts.append(open(p, encoding="utf-8", errors="strict").read())
    total, bad = 0, 0
    for s in texts:
        ref = list(ref_encode(s))
        got = qt.encode(s.encode("utf-8"))
        total += len(ref)
        if got != ref:
            bad += 1
            i = next((k for k in range(min(len(got), len(ref))) if got[k] != ref[k]),
                     min(len(got), len(ref)))
            print(f"MISMATCH on {s[:48]!r}...: first diff at token {i} "
                  f"(ours {got[max(0,i-2):i+3]} vs ref {ref[max(0,i-2):i+3]})")
    if bad:
        os.remove(os.path.join(a.data_dir, a.name + ".enc"))
        sys.exit(f"\nVERIFICATION FAILED: {bad}/{len(texts)} inputs diverged — "
                 f"removed {a.name}.enc. This vocab's merge list is not reproducible "
                 f"by rank order, or the scanner classification is wrong.")
    print(f"VERIFIED: {total} tokens across {len(texts)} inputs, all byte-exact. "
          f"Load with quicktok.get_encoding({a.name!r}, {a.data_dir!r}).")


if __name__ == "__main__":
    main()
