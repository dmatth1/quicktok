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
import argparse, base64, ctypes, json, os, struct, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The exact pretokenizer regexes quicktok has hand-compiled scanners for.
# Keys are the literal pattern strings as they appear in the wild. (tiktoken's
# cl100k pattern is the possessive variant; the lazy Split form of the same
# grammar is what Llama-3-style tokenizer.json files carry.)
CL100K_PAT_POSSESSIVE = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
O200K_PAT = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
LLAMA3_PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
QWEN_PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
# Tekken v3: the o200k case-aware grammar minus the contraction suffix, with \p{N}
TEKKEN_PAT = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

KNOWN_PATTERNS = {
    QWEN_PAT: "qwen",      # before llama3: identical except \p{N} (single digit)
    LLAMA3_PAT: "llama3",
    O200K_PAT: "o200k",
    TEKKEN_PAT: "tekken",
    CL100K_PAT_POSSESSIVE: "cl100k",
}

# Builtin stress cases for verification (diverse + the failure modes we've hit)
STRESS = [
    "Hello, world!", "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return [i**2 for i in range(100)]  # 1+2+3==6",
    "café résumé naïve — Zürich",
    "café ole", "variance 〈*ξ*~*i*〉",
    "한국", "á̛z",
    "日本語のトークナイザー 中文 한국어",
    "emoji \U0001f680\U0001f525\U0001f44d mixed",
    "   leading\n\ttabs and  spaces   \r\n windows lines \r\n",
    "ALLCAPS lowercase MixedCase 1234567890 3.14159 0xDEADBEEF 99.99%",
    "don't can't I'll we've they're 'tis 'TWAS",
    "https://example.com/path?q=1&x=2#frag <html><body>x</body></html>",
    "", "a", "\n\n\n", "Ω≈ç√∫ Икс हिन्दी",
    "久久夜色精品国产噜噜亚洲AV中文ENGLISH",
]


def gpt2_byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def parse_hf(path):
    j = json.load(open(path))
    norm = j.get("normalizer")
    if norm is None or (isinstance(norm, dict) and norm.get("type") == "Sequence"
                        and not norm.get("normalizers")):
        nfc = False   # absent, or an empty Sequence (a no-op, e.g. DeepSeek)
    elif isinstance(norm, dict) and norm.get("type") == "NFC":
        nfc = True
    else:
        sys.exit(f"REFUSED: unsupported normalizer {json.dumps(norm)[:120]} — "
                 f"quicktok implements none/NFC. Wrong normalization would be a "
                 f"silent exactness bug, so this import stops here.")
    # pre_tokenizer: find the Split(Regex) (possibly inside a Sequence with ByteLevel)
    pt = j.get("pre_tokenizer") or {}
    nodes = pt.get("pretokenizers", [pt]) if pt.get("type") == "Sequence" else [pt]
    splits = [(nd.get("pattern") or {}).get("Regex") for nd in nodes if nd.get("type") == "Split"]
    if len(splits) > 1:
        sys.exit("REFUSED: this tokenizer pretokenizes with a PIPELINE of "
                 f"{len(splits)} sequential Split regexes — a different grammar "
                 "shape, not a variant of a supported one. Patterns:\n  " +
                 "\n  ".join(repr(s) for s in splits))
    pattern, bytelevel = (splits[0] if splits else None), False
    for nd in nodes:
        if nd.get("type") == "Split":
            pass
        elif nd.get("type") == "ByteLevel":
            bytelevel = True
            if nd.get("add_prefix_space"):
                sys.exit("REFUSED: ByteLevel add_prefix_space=true is not supported.")
            if nd.get("use_regex") and pattern is None:
                pattern = "GPT2_BYTELEVEL_DEFAULT"   # gpt2-style internal regex
        else:
            sys.exit(f"REFUSED: unsupported pre_tokenizer node {nd.get('type')!r}.")
    if not bytelevel:
        sys.exit("REFUSED: not a byte-level BPE (no ByteLevel pre_tokenizer/decoder).")
    if j.get("model", {}).get("type") != "BPE":
        sys.exit(f"REFUSED: model.type={j.get('model',{}).get('type')!r}, need BPE.")
    if j["model"].get("byte_fallback"):
        sys.exit("REFUSED: byte_fallback BPE is not supported.")
    inv = gpt2_byte_decoder()
    ranks = {}
    for s, i in j["model"]["vocab"].items():
        try:
            ranks[bytes(inv[c] for c in s)] = i
        except KeyError:
            pass
    specials = {t["content"]: t["id"] for t in j.get("added_tokens", [])}
    return ranks, specials, pattern, nfc, 0


def parse_tekken(path):
    j = json.load(open(path))
    cfg = j.get("config", {})
    pattern = cfg.get("pattern")
    if not pattern:
        sys.exit("REFUSED: tekken.json without config.pattern.")
    n_special = cfg.get("default_num_special_tokens", 0)
    # model ids = vocab rank + n_special; only the first
    # (default_vocab_size - n_special) vocab entries are live at default size
    n_live = cfg.get("default_vocab_size", 0) - n_special
    ranks, specials = {}, {}
    for ent in j.get("vocab", []):
        if ent["rank"] < n_live:
            ranks[base64.b64decode(ent["token_bytes"])] = ent["rank"]
    for s in j.get("special_tokens", []) or []:
        specials[s.get("token_str") or s.get("content", f"<special_{s['rank']}>")] = s["rank"]
    return ranks, specials, pattern, False, n_special


def write_files(data_dir, name, ranks, specials, scanner, nfc, id_offset=0):
    recs = sorted(ranks.items(), key=lambda kv: kv[1])
    with open(os.path.join(data_dir, name + ".vocab"), "wb") as f:
        f.write(struct.pack("<I", len(recs)))
        for raw, r in recs:
            f.write(struct.pack("<H", len(raw))); f.write(raw); f.write(struct.pack("<I", r))
    sp = sorted(specials.items(), key=lambda kv: kv[1])
    with open(os.path.join(data_dir, name + ".special"), "wb") as f:
        f.write(struct.pack("<I", len(sp)))
        for tok, tid in sp:
            b = tok.encode()
            f.write(struct.pack("<I", tid)); f.write(struct.pack("<H", len(b))); f.write(b)
    with open(os.path.join(data_dir, name + ".enc"), "w") as f:
        f.write(f"# imported by tools/import_tokenizer.py — verified exact before writing\n")
        f.write(f"scanner={scanner}\n")
        if nfc:
            f.write("nfc=1\n")
        if id_offset:
            f.write(f"id_offset={id_offset}\n")


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
