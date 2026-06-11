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


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--llama3":
        _llama3(sys.argv[2]); return
    for name, stem in [("cl100k_base", "vectors"), ("o200k_base", "vectors_o200k")]:
        enc = tiktoken.get_encoding(name)
        write_vectors(os.path.join(TESTDIR, stem + ".bin"), CASES, enc.encode_ordinary)
        write_vectors(os.path.join(TESTDIR, stem + "_special.bin"), SPECIAL_CASES,
                      lambda s: enc.encode(s, allowed_special="all"))
if __name__ == "__main__":
    main()
