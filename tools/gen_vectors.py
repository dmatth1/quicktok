#!/usr/bin/env python3
"""Regenerate test/vectors.bin from tiktoken cl100k_base (the reference spec).
  format: u32 ncase; per case (u32 blen, utf8 bytes, u32 nids, nids*u32)."""
import os, struct, tiktoken
OUT = os.path.join(os.path.dirname(__file__), "..", "test", "vectors.bin")
CASES = [
    "Hello, world!", "The quick brown fox jumps over the lazy dog.",
    "def encode(text): return tokenizer.encode(text)  # 1234567890",
    "café résumé naïve — Zürich", "日本語のトークナイザー", "emoji test 🚀🔥👍 mixed",
    "   leading\n\ttabs and  spaces   ", "ALLCAPS lowercase MixedCase 42 3.14",
    "don't can't I'll we've they're", "https://example.com/path?q=1&x=2",
    "", "a", "\n\n\n", "Ω≈ç√∫˜µ≤≥÷ åß∂ƒ©˙∆˚¬", "中文 English 混合 text 123",
]
def main():
    enc = tiktoken.get_encoding("cl100k_base")
    with open(OUT, "wb") as f:
        f.write(struct.pack("<I", len(CASES)))
        for s in CASES:
            b = s.encode("utf-8"); ids = enc.encode_ordinary(s)
            f.write(struct.pack("<I", len(b))); f.write(b)
            f.write(struct.pack("<I", len(ids))); f.write(struct.pack(f"<{len(ids)}I", *ids))
    print(f"wrote {len(CASES)} vectors -> {OUT}")
if __name__ == "__main__":
    main()
