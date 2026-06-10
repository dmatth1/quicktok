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

def main():
    os.makedirs(DATA, exist_ok=True)
    enc = tiktoken.get_encoding("cl100k_base")
    ranks = enc._mergeable_ranks
    with open(os.path.join(DATA, "cl100k.vocab"), "wb") as f:
        f.write(struct.pack("<I", len(ranks)))
        for b, r in ranks.items():
            f.write(struct.pack("<H", len(b))); f.write(b); f.write(struct.pack("<I", r))
    print(f"vocab: {len(ranks)} tokens -> data/cl100k.vocab")

if __name__ == "__main__":
    main()
