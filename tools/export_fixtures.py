#!/usr/bin/env python3
"""Regenerate quicktok's data/ from the live reference (tiktoken + regex).
Needs: pip install tiktoken regex.

  cl100k.vocab : u32 n; n records of (u16 blen, blen bytes, u32 rank)
                 == tiktoken cl100k_base mergeable_ranks.
  uniclass.bin : \\p{L}/\\p{N}/\\s codepoint ranges (see tools/export_unicode.py),
                 byte-exact to the regex engine that matches tiktoken's split.

Run tools/export_unicode.py separately (it owns uniclass.bin + its .meta stamp).
"""
import os, struct, tiktoken

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

def export(name, stem):
    enc = tiktoken.get_encoding(name)
    ranks = enc._mergeable_ranks
    with open(os.path.join(DATA, stem + ".vocab"), "wb") as f:
        f.write(struct.pack("<I", len(ranks)))
        for b, r in ranks.items():
            f.write(struct.pack("<H", len(b))); f.write(b); f.write(struct.pack("<I", r))
    # specials: u32 n; per entry u32 id, u16 len, bytes
    sp = sorted(enc._special_tokens.items(), key=lambda kv: kv[1])
    with open(os.path.join(DATA, stem + ".special"), "wb") as f:
        f.write(struct.pack("<I", len(sp)))
        for tok, tid in sp:
            b = tok.encode()
            f.write(struct.pack("<I", tid)); f.write(struct.pack("<H", len(b))); f.write(b)
    print(f"{name}: {len(ranks)} tokens + {len(sp)} specials -> data/{stem}.vocab/.special")

def main():
    os.makedirs(DATA, exist_ok=True)
    export("cl100k_base", "cl100k")
    export("o200k_base", "o200k")

if __name__ == "__main__":
    main()
