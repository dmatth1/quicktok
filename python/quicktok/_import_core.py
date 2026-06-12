"""Pure import logic shared by the packaged importer (quicktok.importer) and
the repo-dev CLI (tools/import_tokenizer.py): pattern classification, config
parsing (HF tokenizer.json / Mistral tekken.json), data-file writing, and the
verification stress suite. No compiled-extension imports — safe to load from a
source tree. See tools/import_tokenizer.py for the full design rationale."""
import base64, json, os, struct, sys

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




def tables_for(scanner, nfc):
    """Data tables an encoding needs next to its vocab (beyond <name>.*)."""
    t = ["uniclass_o200k.bin"] if scanner in ("o200k", "tekken") else ["uniclass.bin"]
    if nfc:
        t.append("nfc.bin")
    return t


def ensure_tables(out_dir, tables_src, scanner, nfc):
    """Copy the required class/NFC tables into out_dir if not already there."""
    import shutil
    for f in tables_for(scanner, nfc):
        dst = os.path.join(out_dir, f)
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(tables_src, f), dst)
