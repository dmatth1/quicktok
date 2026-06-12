#!/usr/bin/env python3
"""Regenerate test/vectors.bin from tiktoken cl100k_base (the reference spec).
  format: u32 ncase; per case (u32 blen, utf8 bytes, u32 nids, nids*u32)."""
import os, struct, sys, tiktoken
TESTDIR = os.path.join(os.path.dirname(__file__), "..", "test")
CASES = [
    "Hello, world!", "The quick brown fox jumps over the lazy dog.",
    "def encode(text): return tokenizer.encode(text)  # 1234567890",
    "café résumé naïve — Zürich", "日本語のトークナイザー", "emoji test 🚀🔥👍 mixed",
    "   leading\n\ttabs and  spaces   ", "ALLCAPS lowercase MixedCase 42 3.14",
    "don't can't I'll we've they're", "https://example.com/path?q=1&x=2",
    "", "a", "\n\n\n", "Ω≈ç√∫˜µ≤≥÷ åß∂ƒ©˙∆˚¬", "中文 English 混合 text 123",
    # CJK/letters directly adjacent to Latin CAPS — exercises o200k's UPPER*/LOWER+
    # pretok backtrack (Lo is in both classes; Lu is UPPER-only). Regression for the
    # "亚洲AV" boundary bug. No spaces: the adjacency is the whole point.
    "噜噜亚洲AV中文ENGLISH日本語XY한국어KRΩβγXYZ",
]
SPECIAL_CASES = [
    "<|endoftext|>",
    "before<|endoftext|>after",
    "a<|endoftext|><|endofprompt|>b",
    "no specials here",
    "tricky <|endof and <|endoftext|",
]

def write_vectors(path, cases, encode_fn):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(cases)))
        for s in cases:
            b = s.encode("utf-8"); ids = encode_fn(s)
            f.write(struct.pack("<I", len(b))); f.write(b)
            f.write(struct.pack("<I", len(ids))); f.write(struct.pack(f"<{len(ids)}I", *ids))
    print(f"wrote {len(cases)} vectors -> {path}")

def _llama3(json_path):
    import json
    j = json.load(open(json_path))
    bs = list(range(33,127))+list(range(161,173))+list(range(174,256)); cs = bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    inv = {chr(c): b for b,c in zip(bs,cs)}
    ranks = {}
    for s,i in j["model"]["vocab"].items():
        try: ranks[bytes(inv[c] for c in s)] = i
        except KeyError: pass
    PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    sp = {t["content"]: t["id"] for t in j["added_tokens"]}
    enc = tiktoken.Encoding(name="llama3", pat_str=PAT, mergeable_ranks=ranks, special_tokens=sp)
    SP = ["<|begin_of_text|>", "before<|end_of_text|>after", "a<|begin_of_text|><|end_of_text|>b", "no specials"]
    write_vectors(os.path.join(TESTDIR, "vectors_llama3.bin"), CASES, enc.encode_ordinary)
    write_vectors(os.path.join(TESTDIR, "vectors_llama3_special.bin"), SP,
                  lambda s: enc.encode(s, allowed_special="all"))


def _qwen3(json_path):
    """Reference = the Hugging Face tokenizer itself (the merge-list BPE quicktok
    reproduces by rank). Needs: pip install tokenizers."""
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(json_path)
    enc_ord = lambda s: hf.encode(s, add_special_tokens=False).ids
    # NFC regressions (explicit escapes — these strings must be NON-NFC):
    # singleton U+2329/U+232A (the Common Crawl find), decomposed accents,
    # decomposed Hangul jamo, marks out of canonical order, a mark with no
    # composition (stays decomposed in NFC), dirty at start and end.
    NFC_CASES = [
        "variance \u2329*\u03be*~*i*\u232a done",
        "cafe\u0301 ole cafe\u0301",
        "\u1112\u1161\u11ab\u1100\u116e\u11a8\u110b\u1165",
        "a\u0301\u031bz",
        "x\u0367y",
        "\u2329start endx\u0301",
    ]
    import unicodedata
    # all but x+U+0367 must be genuinely non-NFC; that one is already-NFC text
    # that still contains a "suspicious" codepoint (exercises the identity window)
    for s in NFC_CASES:
        if s == "x\u0367y":
            assert unicodedata.normalize("NFC", s) == s
        else:
            assert unicodedata.normalize("NFC", s) != s, f"case is already NFC: {s!r}"
    QWEN_SP = ["<|im_start|>", "before<|im_end|>after", "a<|endoftext|><|im_start|>b",
               "no specials here", "<think>reasoning</think>"]
    write_vectors(os.path.join(TESTDIR, "vectors_qwen3.bin"), CASES + NFC_CASES, enc_ord)
    write_vectors(os.path.join(TESTDIR, "vectors_qwen3_special.bin"), QWEN_SP, enc_ord)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--llama3":
        _llama3(sys.argv[2]); return
    if len(sys.argv) > 2 and sys.argv[1] == "--qwen3":
        _qwen3(sys.argv[2]); return
    for name, stem in [("cl100k_base", "vectors"), ("o200k_base", "vectors_o200k"),
                       ("o200k_harmony", "vectors_o200k_harmony")]:
        enc = tiktoken.get_encoding(name)
        write_vectors(os.path.join(TESTDIR, stem + ".bin"), CASES, enc.encode_ordinary)
        write_vectors(os.path.join(TESTDIR, stem + "_special.bin"), SPECIAL_CASES,
                      lambda s: enc.encode(s, allowed_special="all"))
if __name__ == "__main__":
    main()
